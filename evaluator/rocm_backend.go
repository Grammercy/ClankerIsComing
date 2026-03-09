//go:build rocm

package evaluator

import (
	"fmt"
	"math"
	"sync"

	"github.com/jgillich/go-opencl/cl"
)

const mlpOpenCLSource = `
#define TILE_SIZE 16

__kernel void dense_forward(
    __global const float* weights,
    __global const float* biases,
    __global const float* bnMean,
    __global const float* bnVar,
    __global const float* in,
    __global float* out,
    const int inSize,
    const int outSize,
    const int activation,
    const int useBatchNorm
) {
    int gid = get_global_id(0);
    if (gid >= outSize) return;

    float sum = biases[gid];
    int wBase = gid * inSize;
    for (int i = 0; i < inSize; i++) {
        sum += weights[wBase + i] * in[i];
    }

    if (useBatchNorm != 0) {
        sum = (sum - bnMean[gid]) * rsqrt(bnVar[gid] + 1.0e-5f);
    }

    if (activation == 0) {
        out[gid] = (sum > 0.0f) ? sum : (0.01f * sum);
    } else if (activation == 1) {
        out[gid] = 1.0f / (1.0f + exp(-sum));
    } else {
        out[gid] = sum;
    }
}

__kernel void output_delta_bce(
    __global const float* outputs,
    __global const float* targets,
    __global float* deltas,
    const int outSize,
    const float eloWeight
) {
    int gid = get_global_id(0);
    if (gid >= outSize) return;

    float t = targets[gid];
    if (t < 0.0f) {
        deltas[gid] = 0.0f;
    } else {
        deltas[gid] = (t - outputs[gid]) * eloWeight;
    }
}

__kernel void set_output_deltas(
    __global const float* outputs,
    __global const float* outputDeltas,
    __global float* deltas,
    const int outSize,
    const int linearOutput
) {
    int gid = get_global_id(0);
    if (gid >= outSize) return;

    float d = outputDeltas[gid];
    if (linearOutput != 0) {
        deltas[gid] = d;
    } else {
        float o = outputs[gid];
        deltas[gid] = d * o * (1.0f - o);
    }
}

__kernel void hidden_delta(
    __global const float* nextWeights,
    __global const float* nextDeltas,
    __global const float* bnVar,
    __global const float* currentOutputs,
    __global float* currentDeltas,
    const int currentSize,
    const int nextSize,
    const int useBatchNorm
) {
    int gid = get_global_id(0);
    if (gid >= currentSize) return;

    float err = 0.0f;
    for (int k = 0; k < nextSize; k++) {
        err += nextWeights[k * currentSize + gid] * nextDeltas[k];
    }

    float o = currentOutputs[gid];
    float deriv = (o > 0.0f) ? 1.0f : 0.01f;
    float bnScale = 1.0f;
    if (useBatchNorm != 0) {
        bnScale = rsqrt(bnVar[gid] + 1.0e-5f);
    }
    currentDeltas[gid] = err * deriv * bnScale;
}

__kernel void accumulate_weight_grads(
    __global const float* deltas,
    __global const float* inputs,
    __global float* gradWeights,
    const int inSize,
    const int outSize,
    const float scale
) {
    int gid = get_global_id(0);
    int total = inSize * outSize;
    if (gid >= total) return;

    int row = gid / inSize;
    int col = gid % inSize;
    gradWeights[gid] += deltas[row] * inputs[col] * scale;
}

__kernel void accumulate_bias_grads(
    __global const float* deltas,
    __global float* gradBiases,
    const int outSize,
    const float scale
) {
    int gid = get_global_id(0);
    if (gid >= outSize) return;
    gradBiases[gid] += deltas[gid] * scale;
}

__kernel void adam_update(
    __global float* weights,
    __global float* m,
    __global float* v,
    __global const float* grads,
    const int size,
    const float beta1,
    const float beta2,
    const float beta1CorrInv,
    const float beta2CorrInv,
    const float lr,
    const float weightDecay,
    const float epsilon,
    const float invBatch
) {
    int gid = get_global_id(0);
    if (gid >= size) return;

    float g = grads[gid] * invBatch + weightDecay * weights[gid];
    float m_new = beta1 * m[gid] + (1.0f - beta1) * g;
    float v_new = beta2 * v[gid] + (1.0f - beta2) * g * g;

    m[gid] = m_new;
    v[gid] = v_new;

    float m_hat = m_new * beta1CorrInv;
    float v_hat = v_new * beta2CorrInv;

    weights[gid] -= lr * m_hat * rsqrt(v_hat + epsilon);
}

__kernel void first_layer_input_grads(
    __global const float* weights,
    __global const float* deltas,
    __global float* outGrads,
    const int inSize,
    const int outSize,
    const int inputOffset,
    const int gradCount
) {
    int gid = get_global_id(0);
    if (gid >= gradCount) return;

    int inIdx = inputOffset + gid;
    float acc = 0.0f;
    for (int i = 0; i < outSize; i++) {
        acc += weights[i * inSize + inIdx] * deltas[i];
    }
    outGrads[gid] = acc;
}

__kernel void dense_forward_batch_optimized(
    __global const float* weights,
    __global const float* biases,
    __global const float* bnMean,
    __global const float* bnVar,
    __global const float* in,
    __global float* out,
    const int inSize,
    const int outSize,
    const int batchSize,
    const int activation,
    const int useBatchNorm
) {
    int gid = get_global_id(0);
    int total = outSize * batchSize;
    if (gid >= total) return;

    int sample = gid / outSize;
    int neuron = gid % outSize;

    float sum = biases[neuron];
    int wBase = neuron * inSize;
    int inBase = sample * inSize;
    
    for (int i = 0; i < inSize; i++) {
        sum += weights[wBase + i] * in[inBase + i];
    }

    if (useBatchNorm != 0) {
        sum = (sum - bnMean[neuron]) * rsqrt(bnVar[neuron] + 1.0e-5f);
    }

    if (activation == 0) {
        out[gid] = (sum > 0.0f) ? sum : (0.01f * sum);
    } else if (activation == 1) {
        out[gid] = 1.0f / (1.0f + exp(-sum));
    } else {
        out[gid] = sum;
    }
}

__kernel void accumulate_weight_grads_batch_tiled(
    __global const float* deltas,
    __global const float* inputs,
    __global float* gradWeights,
    const int inSize,
    const int outSize,
    const int batchSize
) {
    int gid = get_global_id(0);
    int total = inSize * outSize;
    if (gid >= total) return;

    int row = gid / inSize;
    int col = gid % inSize;

    float sum = 0.0f;
    for (int s = 0; s < batchSize; s++) {
        sum += deltas[s * outSize + row] * inputs[s * inSize + col];
    }
    gradWeights[gid] += sum;
}

__kernel void output_delta_bce_batch(
    __global const float* outputs,
    __global const float* targets,
    __global const float* sampleScale,
    __global float* deltas,
    const int outSize,
    const int batchSize
) {
    int gid = get_global_id(0);
    int total = outSize * batchSize;
    if (gid >= total) return;

    int sample = gid / outSize;
    float t = targets[gid];
    if (t < 0.0f) {
        deltas[gid] = 0.0f;
    } else {
        deltas[gid] = (t - outputs[gid]) * sampleScale[sample];
    }
}

__kernel void set_output_deltas_batch(
    __global const float* outputs,
    __global const float* outputDeltas,
    __global const float* sampleWeights,
    __global float* deltas,
    const int outSize,
    const int batchSize,
    const int linearOutput
) {
    int gid = get_global_id(0);
    int total = outSize * batchSize;
    if (gid >= total) return;

    int sample = gid / outSize;
    float d = outputDeltas[gid] * sampleWeights[sample];
    if (linearOutput != 0) {
        deltas[gid] = d;
    } else {
        float o = outputs[gid];
        deltas[gid] = d * o * (1.0f - o);
    }
}

__kernel void hidden_delta_batch(
    __global const float* nextWeights,
    __global const float* nextDeltas,
    __global const float* bnVar,
    __global const float* currentOutputs,
    __global float* currentDeltas,
    const int currentSize,
    const int nextSize,
    const int batchSize,
    const int useBatchNorm
) {
    int gid = get_global_id(0);
    int total = currentSize * batchSize;
    if (gid >= total) return;

    int sample = gid / currentSize;
    int cur = gid % currentSize;

    float err = 0.0f;
    int nextBase = sample * nextSize;
    for (int k = 0; k < nextSize; k++) {
        err += nextWeights[k * currentSize + cur] * nextDeltas[nextBase + k];
    }

    float o = currentOutputs[gid];
    float deriv = (o > 0.0f) ? 1.0f : 0.01f;
    float bnScale = 1.0f;
    if (useBatchNorm != 0) {
        bnScale = rsqrt(bnVar[cur] + 1.0e-5f);
    }
    currentDeltas[gid] = err * deriv * bnScale;
}

__kernel void accumulate_bias_grads_batch(
    __global const float* deltas,
    __global float* gradBiases,
    const int outSize,
    const int batchSize
) {
    int gid = get_global_id(0);
    if (gid >= outSize) return;

    float sum = 0.0f;
    for (int s = 0; s < batchSize; s++) {
        sum += deltas[s * outSize + gid];
    }
    gradBiases[gid] += sum;
}

__kernel void first_layer_input_grads_batch(
    __global const float* weights,
    __global const float* deltas,
    __global float* outGrads,
    const int inSize,
    const int outSize,
    const int inputOffset,
    const int gradCount,
    const int batchSize
) {
    int gid = get_global_id(0);
    int total = gradCount * batchSize;
    if (gid >= total) return;

    int sample = gid / gradCount;
    int gIdx = gid % gradCount;
    int inIdx = inputOffset + gIdx;

    float acc = 0.0f;
    int dBase = sample * outSize;
    for (int i = 0; i < outSize; i++) {
        acc += weights[i * inSize + inIdx] * deltas[dBase + i];
    }
    outGrads[gid] = acc;
}

__kernel void attention_output_deltas_from_input_grads(
    __global const float* mainInputGrads,
    __global const float* rawSlots,
    __global const float* attentionWeights,
    __global const float* sampleWeights,
    __global float* outDeltas,
    const int featuresPerSlot,
    const int slotCount,
    const int batchSize
) {
    int gid = get_global_id(0);
    if (gid >= batchSize) return;

    int rBase = gid * (slotCount * featuresPerSlot);
    int gBase = gid * (slotCount * featuresPerSlot);
    int outBase = gid * slotCount;

    float sw = sampleWeights[gid];

    for (int s = 0; s < slotCount; s++) {
        float acc = 0.0f;
        int sBase = s * featuresPerSlot;
        for (int f = 0; f < featuresPerSlot; f++) {
            acc += mainInputGrads[gBase + sBase + f] * rawSlots[rBase + sBase + f];
        }
        outDeltas[outBase + s] = acc * sw;
    }
}

__kernel void attention_output_deltas_from_first_layer_batch(
    __global const float* firstLayerWeights,
    __global const float* firstLayerDeltas,
    __global const float* rawSlots,
    __global const float* attentionWeights,
    __global const float* sampleWeights,
    __global float* outDeltas,
    const int firstLayerInSize,
    const int firstLayerOutSize,
    const int inputOffset,
    const int featuresPerSlot,
    const int slotCount,
    const int batchSize
) {
    int gid = get_global_id(0);
    if (gid >= batchSize * slotCount) return;

    int sample = gid / slotCount;
    int s = gid % slotCount;

    int rBase = sample * (slotCount * featuresPerSlot);
    int dBase = sample * firstLayerOutSize;
    float sw = sampleWeights[sample];
    float aw = attentionWeights[sample * slotCount + s];

    float acc = 0.0f;
    int sBase = s * featuresPerSlot;
    for (int f = 0; f < featuresPerSlot; f++) {
        int inIdx = inputOffset + sBase + f;
        float inputGrad = 0.0f;
        for (int i = 0; i < firstLayerOutSize; i++) {
            inputGrad += firstLayerWeights[i * firstLayerInSize + inIdx] * firstLayerDeltas[dBase + i];
        }
        acc += inputGrad * rawSlots[rBase + sBase + f];
    }
    outDeltas[gid] = acc * sw * aw;
}
`

type openclLayerBuffers struct {
	inSize      int
	outSize     int
	weights     *cl.MemObject
	biases      *cl.MemObject
	bnMean      *cl.MemObject
	bnVar       *cl.MemObject
	bnBatchMean *cl.MemObject
	bnBatchVar  *cl.MemObject
	weightM     *cl.MemObject
	weightV     *cl.MemObject
	biasM       *cl.MemObject
	biasV       *cl.MemObject
	outputs     *cl.MemObject
	deltas      *cl.MemObject
	batchOut    *cl.MemObject
	batchDel    *cl.MemObject
	gradWBuf    *cl.MemObject
	gradBBuf    *cl.MemObject
}

type openclMLPBackend struct {
	ctx                                            *cl.Context
	queue                                          *cl.CommandQueue
	device                                         *cl.Device
	program                                        *cl.Program
	denseForwardKernel                             *cl.Kernel
	outputDeltaBCEKernel                           *cl.Kernel
	setOutputDeltasKernel                          *cl.Kernel
	hiddenDeltaKernel                              *cl.Kernel
	denseForwardBatchKernel                        *cl.Kernel
	outputDeltaBCEBatchKernel                      *cl.Kernel
	setOutputDeltasBatchKernel                     *cl.Kernel
	hiddenDeltaBatchKernel                         *cl.Kernel
	accWeightGradsBatchKernel                      *cl.Kernel
	accBiasGradsBatchKernel                        *cl.Kernel
	accWeightGradsKernel                           *cl.Kernel
	accBiasGradsKernel                             *cl.Kernel
	adamUpdateKernel                               *cl.Kernel
	firstInputGradsKernel                          *cl.Kernel
	firstInputGradsBatchKernel                     *cl.Kernel
	attentionDeltaInputKernel                      *cl.Kernel
	attentionOutputDeltasFromFirstLayerBatchKernel *cl.Kernel

	mu                sync.Mutex
	maxBatchSize      int
	workBufSize       int
	lastBatchSize     int
	bnNeutralMean     *cl.MemObject
	bnNeutralVar      *cl.MemObject
	scratchA          *cl.MemObject
	scratchB          *cl.MemObject
	targetsBuf        *cl.MemObject
	inputGradBuf      *cl.MemObject
	batchInputBuf     *cl.MemObject
	batchTargetsBuf   *cl.MemObject
	batchScaleBuf     *cl.MemObject
	attentionDeltaBuf *cl.MemObject
	inputGradBatchBuf *cl.MemObject
	batchSlotsBuffer  *cl.MemObject

	kernelLocal1D map[*cl.Kernel]int
}

func (b *openclMLPBackend) maxBatchSizeLimit() int {
	return b.maxBatchSize
}

func (b *openclMLPBackend) enqueueKernel1D(k *cl.Kernel, globalSize int) error {
	local := b.kernelLocal1D[k]
	if local <= 0 {
		local = 64
	}
	g := ((globalSize + local - 1) / local) * local
	_, err := b.queue.EnqueueNDRangeKernel(k, []int{0}, []int{g}, []int{local}, nil)
	return err
}

func newOpenCLMLPBackend() (*openclMLPBackend, error) {
	dev, err := pickBestGPU()
	if err != nil {
		return nil, err
	}

	ctx, err := cl.CreateContext([]*cl.Device{dev})
	if err != nil {
		return nil, fmt.Errorf("create OpenCL context: %w", err)
	}

	queue, err := ctx.CreateCommandQueue(dev, 0)
	if err != nil {
		ctx.Release()
		return nil, fmt.Errorf("create OpenCL command queue: %w", err)
	}

	program, err := ctx.CreateProgramWithSource([]string{mlpOpenCLSource})
	if err != nil {
		queue.Release()
		ctx.Release()
		return nil, fmt.Errorf("create OpenCL program: %w", err)
	}
	if err := program.BuildProgram(nil, ""); err != nil {
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, fmt.Errorf("build OpenCL program: %w", err)
	}

	fmt.Println("ROCm backend initialized (stable flags)")

	mk := func(name string) (*cl.Kernel, error) {
		k, kErr := program.CreateKernel(name)
		if kErr != nil {
			return nil, fmt.Errorf("create kernel %q: %w", name, kErr)
		}
		return k, nil
	}

	denseForwardKernel, _ := mk("dense_forward")
	outputDeltaBCEKernel, _ := mk("output_delta_bce")
	setOutputDeltasKernel, _ := mk("set_output_deltas")
	hiddenDeltaKernel, _ := mk("hidden_delta")
	accWeightGradsKernel, _ := mk("accumulate_weight_grads")
	accBiasGradsKernel, _ := mk("accumulate_bias_grads")
	adamUpdateKernel, _ := mk("adam_update")
	firstInputGradsKernel, _ := mk("first_layer_input_grads")
	denseForwardBatchKernel, _ := mk("dense_forward_batch_optimized")
	outputDeltaBCEBatchKernel, _ := mk("output_delta_bce_batch")
	setOutputDeltasBatchKernel, _ := mk("set_output_deltas_batch")
	hiddenDeltaBatchKernel, _ := mk("hidden_delta_batch")
	accWeightGradsBatchKernel, _ := mk("accumulate_weight_grads_batch_tiled")
	accBiasGradsBatchKernel, _ := mk("accumulate_bias_grads_batch")
	firstInputGradsBatchKernel, _ := mk("first_layer_input_grads_batch")
	attentionDeltaInputKernel, _ := mk("attention_output_deltas_from_input_grads")
	attentionOutputDeltasFromFirstLayerBatchKernel, _ := mk("attention_output_deltas_from_first_layer_batch")

	backend := &openclMLPBackend{
		ctx:                        ctx,
		queue:                      queue,
		device:                     dev,
		program:                    program,
		denseForwardKernel:         denseForwardKernel,
		outputDeltaBCEKernel:       outputDeltaBCEKernel,
		setOutputDeltasKernel:      setOutputDeltasKernel,
		hiddenDeltaKernel:          hiddenDeltaKernel,
		denseForwardBatchKernel:    denseForwardBatchKernel,
		outputDeltaBCEBatchKernel:  outputDeltaBCEBatchKernel,
		setOutputDeltasBatchKernel: setOutputDeltasBatchKernel,
		hiddenDeltaBatchKernel:     hiddenDeltaBatchKernel,
		accWeightGradsBatchKernel:  accWeightGradsBatchKernel,
		accBiasGradsBatchKernel:    accBiasGradsBatchKernel,
		accWeightGradsKernel:       accWeightGradsKernel,
		accBiasGradsKernel:         accBiasGradsKernel,
		adamUpdateKernel:           adamUpdateKernel,
		firstInputGradsKernel:      firstInputGradsKernel,
		firstInputGradsBatchKernel: firstInputGradsBatchKernel,
		attentionDeltaInputKernel:  attentionDeltaInputKernel,
		attentionOutputDeltasFromFirstLayerBatchKernel: attentionOutputDeltasFromFirstLayerBatchKernel,
		kernelLocal1D: make(map[*cl.Kernel]int, 20),
	}

	for _, kernel := range []*cl.Kernel{
		denseForwardKernel, outputDeltaBCEKernel, setOutputDeltasKernel, hiddenDeltaKernel,
		denseForwardBatchKernel, outputDeltaBCEBatchKernel, setOutputDeltasBatchKernel, hiddenDeltaKernel,
		accWeightGradsBatchKernel, accBiasGradsBatchKernel, accWeightGradsKernel, accBiasGradsKernel,
		adamUpdateKernel, firstInputGradsKernel, firstInputGradsBatchKernel, attentionDeltaInputKernel,
		attentionOutputDeltasFromFirstLayerBatchKernel,
	} {
		if kernel != nil {
			backend.kernelLocal1D[kernel] = chooseKernelLocalSize(dev, kernel)
		}
	}

	maxBatch := 1024
	maxWidth := 1024
	backend.maxBatchSize = maxBatch
	backend.workBufSize = maxWidth

	backend.bnNeutralMean, _ = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth)
	backend.bnNeutralVar, _ = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth)
	backend.scratchA, _ = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth)
	backend.scratchB, _ = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth)
	backend.targetsBuf, _ = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth)
	backend.inputGradBuf, _ = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth)
	backend.batchInputBuf, _ = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth*maxBatch)
	backend.batchTargetsBuf, _ = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth*maxBatch)
	backend.batchScaleBuf, _ = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxBatch)
	backend.attentionDeltaBuf, _ = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth*maxBatch)
	backend.inputGradBatchBuf, _ = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth*maxBatch)
	backend.batchSlotsBuffer, _ = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth*maxBatch)

	return backend, nil
}

func (b *openclMLPBackend) InitMLP(mlp *MLP) error {
	maxBatch := b.maxBatchSize
	mlp.gpuLayers = make([]*openclLayerBuffers, len(mlp.Layers))
	for i, layer := range mlp.Layers {
		inSize := len(layer.Weights[0])
		outSize := len(layer.Weights)
		weights := flatten2DF64(layer.Weights)
		biases := float64To32(layer.Biases)
		bnMean := float64To32(layer.BNRunningMean)
		bnVar := float64To32(layer.BNRunningVar)

		wBuf, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*len(weights))
		b.queue.EnqueueWriteBufferFloat32(wBuf, true, 0, weights, nil)
		bBuf, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*len(biases))
		b.queue.EnqueueWriteBufferFloat32(bBuf, true, 0, biases, nil)
		bnMeanBuf, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*len(bnMean))
		b.queue.EnqueueWriteBufferFloat32(bnMeanBuf, true, 0, bnMean, nil)
		bnVarBuf, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*len(bnVar))
		b.queue.EnqueueWriteBufferFloat32(bnVarBuf, true, 0, bnVar, nil)

		bnBatchMeanBuf, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*outSize)
		unitVar := make([]float32, outSize)
		for j := range unitVar {
			unitVar[j] = 1.0
		}
		bnBatchVarBuf, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*outSize)
		b.queue.EnqueueWriteBufferFloat32(bnBatchVarBuf, true, 0, unitVar, nil)

		wmBuf, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*len(weights))
		wvBuf, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*len(weights))
		bmBuf, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*len(biases))
		bvBuf, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*len(biases))

		outBuf, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*outSize)
		deltaBuf, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*outSize)
		batchOutBuf, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*outSize*maxBatch)
		batchDelBuf, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*outSize*maxBatch)
		gradW, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*len(weights))
		gradB, _ := b.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*len(biases))

		mlp.gpuLayers[i] = &openclLayerBuffers{
			inSize:      inSize,
			outSize:     outSize,
			weights:     wBuf,
			biases:      bBuf,
			bnMean:      bnMeanBuf,
			bnVar:       bnVarBuf,
			bnBatchMean: bnBatchMeanBuf,
			bnBatchVar:  bnBatchVarBuf,
			weightM:     wmBuf,
			weightV:     wvBuf,
			biasM:       bmBuf,
			biasV:       bvBuf,
			outputs:     outBuf,
			deltas:      deltaBuf,
			batchOut:    batchOutBuf,
			batchDel:    batchDelBuf,
			gradWBuf:    gradW,
			gradBBuf:    gradB,
		}
	}
	return nil
}

func (b *openclMLPBackend) Release(mlp *MLP) {
	if b == nil || mlp == nil {
		return
	}
	for _, l := range mlp.gpuLayers {
		if l == nil {
			continue
		}
		l.weights.Release()
		l.biases.Release()
		l.bnMean.Release()
		l.bnVar.Release()
		l.bnBatchMean.Release()
		l.bnBatchVar.Release()
		l.weightM.Release()
		l.weightV.Release()
		l.biasM.Release()
		l.biasV.Release()
		l.outputs.Release()
		l.deltas.Release()
		l.batchOut.Release()
		l.batchDel.Release()
		l.gradWBuf.Release()
		l.gradBBuf.Release()
	}
}

func (b *openclMLPBackend) syncParamsFromHost(mlp *MLP) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	for i, layer := range mlp.Layers {
		w := flatten2DF64(layer.Weights)
		bv := float64To32(layer.Biases)
		bnMean := float64To32(layer.BNRunningMean)
		bnVar := float64To32(layer.BNRunningVar)
		b.queue.EnqueueWriteBufferFloat32(mlp.gpuLayers[i].weights, true, 0, w, nil)
		b.queue.EnqueueWriteBufferFloat32(mlp.gpuLayers[i].biases, true, 0, bv, nil)
		b.queue.EnqueueWriteBufferFloat32(mlp.gpuLayers[i].bnMean, true, 0, bnMean, nil)
		b.queue.EnqueueWriteBufferFloat32(mlp.gpuLayers[i].bnVar, true, 0, bnVar, nil)
	}
	return nil
}

func (b *openclMLPBackend) syncParamsToHost(mlp *MLP) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	for i := range mlp.Layers {
		layer := mlp.gpuLayers[i]
		wOut := make([]float32, layer.inSize*layer.outSize)
		bOut := make([]float32, layer.outSize)
		bnMeanOut := make([]float32, layer.outSize)
		bnVarOut := make([]float32, layer.outSize)
		b.queue.EnqueueReadBufferFloat32(layer.weights, true, 0, wOut, nil)
		b.queue.EnqueueReadBufferFloat32(layer.biases, true, 0, bOut, nil)
		b.queue.EnqueueReadBufferFloat32(layer.bnMean, true, 0, bnMeanOut, nil)
		b.queue.EnqueueReadBufferFloat32(layer.bnVar, true, 0, bnVarOut, nil)
		unflattenTo2DF64(mlp.Layers[i].Weights, wOut)
		for j := range bOut {
			mlp.Layers[i].Biases[j] = float64(bOut[j])
			mlp.Layers[i].BNRunningMean[j] = float64(bnMeanOut[j])
			mlp.Layers[i].BNRunningVar[j] = float64(bnVarOut[j])
		}
	}
	return b.queue.Finish()
}

func (b *openclMLPBackend) clearGradients(mlp *MLP) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, layer := range mlp.gpuLayers {
		zeroW := make([]float32, layer.inSize*layer.outSize)
		zeroB := make([]float32, layer.outSize)
		b.queue.EnqueueWriteBufferFloat32(layer.gradWBuf, false, 0, zeroW, nil)
		b.queue.EnqueueWriteBufferFloat32(layer.gradBBuf, false, 0, zeroB, nil)
	}
	return b.queue.Finish()
}

func isHiddenLayer(layerIdx int, totalLayers int) bool {
	return layerIdx >= 0 && layerIdx < totalLayers-1
}

func (b *openclMLPBackend) bnBuffersForLayer(mlp *MLP, layerIdx int, useBatchStats bool) (*cl.MemObject, *cl.MemObject, int32) {
	if !isHiddenLayer(layerIdx, len(mlp.Layers)) {
		return b.bnNeutralMean, b.bnNeutralVar, 0
	}
	layer := mlp.gpuLayers[layerIdx]
	if useBatchStats {
		return layer.bnBatchMean, layer.bnBatchVar, 1
	}
	return layer.bnMean, layer.bnVar, 1
}

func (b *openclMLPBackend) refreshBatchNormStats(mlp *MLP, layerIdx int, batchSize int) error {
	if !isHiddenLayer(layerIdx, len(mlp.Layers)) {
		return nil
	}
	layer := mlp.gpuLayers[layerIdx]
	flat := make([]float32, layer.outSize*batchSize)
	b.queue.EnqueueReadBufferFloat32(layer.batchOut, true, 0, flat, nil)

	batchMean := make([]float32, layer.outSize)
	batchVar := make([]float32, layer.outSize)
	hostLayer := mlp.Layers[layerIdx]
	invBatch := 1.0 / float64(batchSize)
	for j := 0; j < layer.outSize; j++ {
		mean := 0.0
		for s := 0; s < batchSize; s++ {
			mean += float64(flat[s*layer.outSize+j])
		}
		mean *= invBatch
		variance := 0.0
		for s := 0; s < batchSize; s++ {
			d := float64(flat[s*layer.outSize+j]) - mean
			variance += d * d
		}
		variance *= invBatch
		if variance < 1e-5 {
			variance = 1e-5
		}
		batchMean[j] = float32(mean)
		batchVar[j] = float32(variance)
		hostLayer.BNRunningMean[j] = 0.9*hostLayer.BNRunningMean[j] + 0.1*mean
		hostLayer.BNRunningVar[j] = 0.9*hostLayer.BNRunningVar[j] + 0.1*variance
	}
	b.queue.EnqueueWriteBufferFloat32(layer.bnBatchMean, true, 0, batchMean, nil)
	b.queue.EnqueueWriteBufferFloat32(layer.bnBatchVar, true, 0, batchVar, nil)
	b.queue.EnqueueWriteBufferFloat32(layer.bnMean, true, 0, float64To32(hostLayer.BNRunningMean), nil)
	b.queue.EnqueueWriteBufferFloat32(layer.bnVar, true, 0, float64To32(hostLayer.BNRunningVar), nil)
	return nil
}

func (b *openclMLPBackend) forwardAndStore(mlp *MLP, inputs []float64, _ *WorkerCache) error {
	b.queue.EnqueueWriteBufferFloat32(b.scratchA, false, 0, float64To32(inputs), nil)
	current := b.scratchA
	lastSize := len(inputs)
	for i, layer := range mlp.gpuLayers {
		activation := int32(0)
		if i == len(mlp.Layers)-1 {
			activation = 1
			if mlp.LinearOutput {
				activation = 2
			}
		}
		bnMean, bnVar, useBN := b.bnBuffersForLayer(mlp, i, false)
		b.denseForwardKernel.SetArgs(layer.weights, layer.biases, bnMean, bnVar, current, layer.outputs, int32(lastSize), int32(layer.outSize), activation, useBN)
		b.enqueueKernel1D(b.denseForwardKernel, layer.outSize)
		current = layer.outputs
		lastSize = layer.outSize
	}
	return nil
}

func (b *openclMLPBackend) calculateBCELocalGradients(mlp *MLP, inputs []float64, targets []float64, eloWeight float64, cache *WorkerCache) (float64, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.forwardAndStore(mlp, inputs, cache); err != nil {
		return 0, err
	}
	last := mlp.gpuLayers[len(mlp.gpuLayers)-1]
	out32 := make([]float32, last.outSize)
	b.queue.EnqueueReadBufferFloat32(last.outputs, true, 0, out32, nil)
	loss := 0.0
	deltas32 := make([]float32, last.outSize)
	validTargets := 0.0
	for i := range out32 {
		t := float32(targets[i])
		o := out32[i]
		if t < 0 {
			deltas32[i] = 0
		} else {
			validTargets++
			o64 := float64(o)
			if o64 < 1e-7 {
				o64 = 1e-7
			}
			if o64 > 1.0-1e-7 {
				o64 = 1.0 - 1e-7
			}
			loss -= float64(t)*math.Log(o64) + float64(1-t)*math.Log(1.0-o64)
		}
	}
	scale := float32(0.0)
	if validTargets > 0 {
		loss /= validTargets
		scale = float32(eloWeight / validTargets)
	}
	for i := range out32 {
		t := float32(targets[i])
		if t < 0 {
			continue
		}
		deltas32[i] = (t - out32[i]) * scale
	}
	b.queue.EnqueueWriteBufferFloat32(last.deltas, false, 0, deltas32, nil)
	for i := len(mlp.gpuLayers) - 1; i >= 0; i-- {
		layer := mlp.gpuLayers[i]
		prevOut, pSize := b.scratchA, len(inputs)
		if i > 0 {
			prevOut, pSize = mlp.gpuLayers[i-1].outputs, mlp.gpuLayers[i-1].outSize
		}
		b.accWeightGradsKernel.SetArgs(layer.deltas, prevOut, layer.gradWBuf, int32(pSize), int32(layer.outSize), float32(1.0))
		b.enqueueKernel1D(b.accWeightGradsKernel, pSize*layer.outSize)
		b.accBiasGradsKernel.SetArgs(layer.deltas, layer.gradBBuf, int32(layer.outSize), float32(1.0))
		b.enqueueKernel1D(b.accBiasGradsKernel, layer.outSize)
		if i > 0 {
			prev := mlp.gpuLayers[i-1]
			_, bnVar, useBN := b.bnBuffersForLayer(mlp, i, false)
			b.hiddenDeltaKernel.SetArgs(layer.weights, layer.deltas, bnVar, prev.outputs, prev.deltas, int32(prev.outSize), int32(layer.outSize), useBN)
			b.enqueueKernel1D(b.hiddenDeltaKernel, prev.outSize)
		}
	}
	return loss, nil
}

func (b *openclMLPBackend) backpropGivenDeltas(mlp *MLP, inputs []float64, outputDeltas []float64, eloWeight float64, cache *WorkerCache) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.forwardAndStore(mlp, inputs, cache); err != nil {
		return err
	}
	lastIdx := len(mlp.gpuLayers) - 1
	last := mlp.gpuLayers[lastIdx]
	linearFlag := int32(0)
	if mlp.LinearOutput {
		linearFlag = 1
	}
	b.queue.EnqueueWriteBufferFloat32(b.targetsBuf, false, 0, float64To32(outputDeltas), nil)
	b.setOutputDeltasKernel.SetArgs(last.outputs, b.targetsBuf, last.deltas, int32(last.outSize), linearFlag)
	b.enqueueKernel1D(b.setOutputDeltasKernel, last.outSize)

	for i := len(mlp.gpuLayers) - 1; i >= 0; i-- {
		layer := mlp.gpuLayers[i]
		prevOut, pSize := b.scratchA, len(inputs)
		if i > 0 {
			prevOut, pSize = mlp.gpuLayers[i-1].outputs, mlp.gpuLayers[i-1].outSize
		}
		b.accWeightGradsKernel.SetArgs(layer.deltas, prevOut, layer.gradWBuf, int32(pSize), int32(layer.outSize), float32(eloWeight))
		b.enqueueKernel1D(b.accWeightGradsKernel, pSize*layer.outSize)
		b.accBiasGradsKernel.SetArgs(layer.deltas, layer.gradBBuf, int32(layer.outSize), float32(eloWeight))
		b.enqueueKernel1D(b.accBiasGradsKernel, layer.outSize)
		if i > 0 {
			prev := mlp.gpuLayers[i-1]
			_, bnVar, useBN := b.bnBuffersForLayer(mlp, i, false)
			b.hiddenDeltaKernel.SetArgs(layer.weights, layer.deltas, bnVar, prev.outputs, prev.deltas, int32(prev.outSize), int32(layer.outSize), useBN)
			b.enqueueKernel1D(b.hiddenDeltaKernel, prev.outSize)
		}
	}
	return nil
}

func (b *openclMLPBackend) backpropGivenDeltasBatch(mlp *MLP, inputsBatch [][]float64, outputDeltasBatch [][]float64, sampleWeights []float64) error {
	batchSize := len(inputsBatch)
	if batchSize == 0 {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	inSize := len(inputsBatch[0])
	flatIn := make([]float32, batchSize*inSize)
	for i, inputs := range inputsBatch {
		copy(flatIn[i*inSize:], float64To32(inputs))
	}
	b.queue.EnqueueWriteBufferFloat32(b.batchInputBuf, false, 0, flatIn, nil)

	current := b.batchInputBuf
	lastSize := inSize
	for i, layer := range mlp.gpuLayers {
		activation := int32(0)
		if i == len(mlp.Layers)-1 {
			activation = 1
			if mlp.LinearOutput {
				activation = 2
			}
		}
		bnMean, bnVar, useBN := b.bnBuffersForLayer(mlp, i, false)
		b.denseForwardBatchKernel.SetArgs(layer.weights, layer.biases, bnMean, bnVar, current, layer.batchOut, int32(lastSize), int32(layer.outSize), int32(batchSize), activation, useBN)
		b.enqueueKernel1D(b.denseForwardBatchKernel, layer.outSize*batchSize)
		current, lastSize = layer.batchOut, layer.outSize
	}

	last := mlp.gpuLayers[len(mlp.gpuLayers)-1]
	flatD := make([]float32, batchSize*last.outSize)
	for s := 0; s < batchSize; s++ {
		copy(flatD[s*last.outSize:], float64To32(outputDeltasBatch[s]))
	}
	b.queue.EnqueueWriteBufferFloat32(b.batchTargetsBuf, false, 0, flatD, nil)
	b.queue.EnqueueWriteBufferFloat32(b.batchScaleBuf, false, 0, float64To32(sampleWeights), nil)

	linearFlag := int32(0)
	if mlp.LinearOutput {
		linearFlag = 1
	}
	b.setOutputDeltasBatchKernel.SetArgs(last.batchOut, b.batchTargetsBuf, b.batchScaleBuf, last.batchDel, int32(last.outSize), int32(batchSize), linearFlag)
	b.enqueueKernel1D(b.setOutputDeltasBatchKernel, last.outSize*batchSize)

	for i := len(mlp.gpuLayers) - 1; i >= 0; i-- {
		layer := mlp.gpuLayers[i]
		prevOut, pSize := b.batchInputBuf, inSize
		if i > 0 {
			prevOut, pSize = mlp.gpuLayers[i-1].batchOut, mlp.gpuLayers[i-1].outSize
		}
		b.accWeightGradsBatchKernel.SetArgs(layer.batchDel, prevOut, layer.gradWBuf, int32(pSize), int32(layer.outSize), int32(batchSize))
		b.enqueueKernel1D(b.accWeightGradsBatchKernel, pSize*layer.outSize)
		b.accBiasGradsBatchKernel.SetArgs(layer.batchDel, layer.gradBBuf, int32(layer.outSize), int32(batchSize))
		b.enqueueKernel1D(b.accBiasGradsBatchKernel, layer.outSize)
		if i > 0 {
			prev := mlp.gpuLayers[i-1]
			_, bnVar, useBN := b.bnBuffersForLayer(mlp, i, false)
			b.hiddenDeltaBatchKernel.SetArgs(layer.weights, layer.batchDel, bnVar, prev.batchOut, prev.batchDel, int32(prev.outSize), int32(layer.outSize), int32(batchSize), useBN)
			b.enqueueKernel1D(b.hiddenDeltaBatchKernel, prev.outSize*batchSize)
		}
	}
	return nil
}

func (b *openclMLPBackend) forward(mlp *MLP, inputs []float64, cache *InferenceCache) ([]float64, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.queue.EnqueueWriteBufferFloat32(b.scratchA, false, 0, float64To32(inputs), nil)
	current := b.scratchA
	next := b.scratchB
	lastSize := len(inputs)
	for i, layer := range mlp.gpuLayers {
		activation := int32(0)
		if i == len(mlp.Layers)-1 {
			activation = 1
			if mlp.LinearOutput {
				activation = 2
			}
		}
		bnMean, bnVar, useBN := b.bnBuffersForLayer(mlp, i, false)
		b.denseForwardKernel.SetArgs(layer.weights, layer.biases, bnMean, bnVar, current, next, int32(lastSize), int32(layer.outSize), activation, useBN)
		b.enqueueKernel1D(b.denseForwardKernel, layer.outSize)
		current, next = next, current
		lastSize = layer.outSize
	}
	out32 := make([]float32, lastSize)
	b.queue.EnqueueReadBufferFloat32(current, true, 0, out32, nil)
	return float32To64(out32), nil
}

func (b *openclMLPBackend) forwardBatch(mlp *MLP, inputsBatch [][]float64) ([][]float64, error) {
	batchSize := len(inputsBatch)
	if batchSize == 0 {
		return nil, nil
	}
	inSize := len(inputsBatch[0])
	b.mu.Lock()
	defer b.mu.Unlock()
	flatIn := make([]float32, batchSize*inSize)
	for i, inputs := range inputsBatch {
		copy(flatIn[i*inSize:], float64To32(inputs))
	}
	b.queue.EnqueueWriteBufferFloat32(b.batchInputBuf, false, 0, flatIn, nil)
	current := b.batchInputBuf
	lastSize := inSize
	for i, layer := range mlp.gpuLayers {
		activation := int32(0)
		if i == len(mlp.Layers)-1 {
			activation = 1
			if mlp.LinearOutput {
				activation = 2
			}
		}
		bnMean, bnVar, useBN := b.bnBuffersForLayer(mlp, i, false)
		b.denseForwardBatchKernel.SetArgs(layer.weights, layer.biases, bnMean, bnVar, current, layer.batchOut, int32(lastSize), int32(layer.outSize), int32(batchSize), activation, useBN)
		b.enqueueKernel1D(b.denseForwardBatchKernel, layer.outSize*batchSize)
		current = layer.batchOut
		lastSize = layer.outSize
	}
	outFlat := make([]float32, batchSize*lastSize)
	b.queue.EnqueueReadBufferFloat32(current, true, 0, outFlat, nil)
	out := make([][]float64, batchSize)
	for s := 0; s < batchSize; s++ {
		row := make([]float64, lastSize)
		for i := 0; i < lastSize; i++ {
			row[i] = float64(outFlat[s*lastSize+i])
		}
		out[s] = row
	}
	return out, nil
}

func (b *openclMLPBackend) calculateBCELocalGradientsBatch(mlp *MLP, inputsBatch [][]float64, targetsBatch [][]float64, eloWeights []float64) (float64, [][]float64, error) {
	batchSize := len(targetsBatch)
	if batchSize == 0 {
		return 0, nil, nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	inSize := 0
	if inputsBatch != nil {
		inSize = len(inputsBatch[0])
		flatIn := make([]float32, batchSize*inSize)
		for i := 0; i < batchSize; i++ {
			copy(flatIn[i*inSize:], float64To32(inputsBatch[i]))
		}
		b.queue.EnqueueWriteBufferFloat32(b.batchInputBuf, false, 0, flatIn, nil)
	} else {
		inSize = 1012
	} // TotalFeatures (Globals + 12*80)

	current := b.batchInputBuf
	lastSize := inSize
	for i, layer := range mlp.gpuLayers {
		activation := int32(0)
		if i == len(mlp.Layers)-1 {
			activation = 1
			if mlp.LinearOutput {
				activation = 2
			}
		}
		bnMean, bnVar, useBN := b.bnBuffersForLayer(mlp, i, false)
		b.denseForwardBatchKernel.SetArgs(layer.weights, layer.biases, bnMean, bnVar, current, layer.batchOut, int32(lastSize), int32(layer.outSize), int32(batchSize), activation, useBN)
		b.enqueueKernel1D(b.denseForwardBatchKernel, layer.outSize*batchSize)
		current, lastSize = layer.batchOut, layer.outSize
	}

	last := mlp.gpuLayers[len(mlp.gpuLayers)-1]
	outFlat := make([]float32, batchSize*last.outSize)
	b.queue.EnqueueReadBufferFloat32(last.batchOut, true, 0, outFlat, nil)
	targets32, scale32 := make([]float32, batchSize*last.outSize), make([]float32, batchSize)
	totalLoss := 0.0
	for s := 0; s < batchSize; s++ {
		valid := 0.0
		sampleLoss := 0.0
		for i := 0; i < last.outSize; i++ {
			idx := s*last.outSize + i
			t := float32(targetsBatch[s][i])
			targets32[idx] = t
			if t >= 0 {
				valid++
				o := float64(outFlat[idx])
				if o < 1e-7 {
					o = 1e-7
				}
				if o > 1.0-1e-7 {
					o = 1.0 - 1e-7
				}
				sampleLoss -= float64(t)*math.Log(o) + (1.0-float64(t))*math.Log(1.0-o)
			}
		}
		if valid > 0 {
			totalLoss += sampleLoss / valid
			scale32[s] = float32(eloWeights[s] / valid)
		} else {
			scale32[s] = 0.0
		}
	}
	b.queue.EnqueueWriteBufferFloat32(b.batchTargetsBuf, false, 0, targets32, nil)
	b.queue.EnqueueWriteBufferFloat32(b.batchScaleBuf, false, 0, scale32, nil)
	b.outputDeltaBCEBatchKernel.SetArgs(last.batchOut, b.batchTargetsBuf, b.batchScaleBuf, last.batchDel, int32(last.outSize), int32(batchSize))
	b.enqueueKernel1D(b.outputDeltaBCEBatchKernel, last.outSize*batchSize)

	for i := len(mlp.gpuLayers) - 1; i >= 0; i-- {
		layer := mlp.gpuLayers[i]
		prevOut, pSize := b.batchInputBuf, inSize
		if i > 0 {
			prevOut, pSize = mlp.gpuLayers[i-1].batchOut, mlp.gpuLayers[i-1].outSize
		}
		b.accWeightGradsBatchKernel.SetArgs(layer.batchDel, prevOut, layer.gradWBuf, int32(pSize), int32(layer.outSize), int32(batchSize))
		b.enqueueKernel1D(b.accWeightGradsBatchKernel, pSize*layer.outSize)
		b.accBiasGradsBatchKernel.SetArgs(layer.batchDel, layer.gradBBuf, int32(layer.outSize), int32(batchSize))
		b.enqueueKernel1D(b.accBiasGradsBatchKernel, layer.outSize)
		if i > 0 {
			prev := mlp.gpuLayers[i-1]
			_, bnVar, useBN := b.bnBuffersForLayer(mlp, i, false)
			b.hiddenDeltaBatchKernel.SetArgs(layer.weights, layer.batchDel, bnVar, prev.batchOut, prev.batchDel, int32(prev.outSize), int32(layer.outSize), int32(batchSize), useBN)
			b.enqueueKernel1D(b.hiddenDeltaBatchKernel, prev.outSize*batchSize)
		}
	}
	out64 := make([][]float64, batchSize)
	for s := 0; s < batchSize; s++ {
		row := make([]float64, last.outSize)
		for i := 0; i < last.outSize; i++ {
			row[i] = float64(outFlat[s*last.outSize+i])
		}
		out64[s] = row
	}
	return totalLoss, out64, nil
}

func (b *openclMLPBackend) backpropAttentionFromInputGradsBatch(mlp *MLP, rawSlotsBatch [][]float64, attentionWeightsBatch [][]float64, mainInputGradsFlat []float64, sampleWeights []float64, featuresPerSlot int) error {
	batchSize := len(rawSlotsBatch)
	if batchSize == 0 {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	rawSize := len(rawSlotsBatch[0])
	flatR := make([]float32, batchSize*rawSize)
	for s := 0; s < batchSize; s++ {
		copy(flatR[s*rawSize:], float64To32(rawSlotsBatch[s]))
	}
	b.queue.EnqueueWriteBufferFloat32(b.batchSlotsBuffer, false, 0, flatR, nil)
	b.queue.EnqueueWriteBufferFloat32(b.inputGradBatchBuf, false, 0, float64To32(mainInputGradsFlat), nil)

	slotCount := len(attentionWeightsBatch[0])
	flatA := make([]float32, batchSize*slotCount)
	for s := 0; s < batchSize; s++ {
		copy(flatA[s*slotCount:], float64To32(attentionWeightsBatch[s]))
	}
	b.queue.EnqueueWriteBufferFloat32(b.batchTargetsBuf, false, 0, flatA, nil)
	b.queue.EnqueueWriteBufferFloat32(b.batchScaleBuf, false, 0, float64To32(sampleWeights), nil)

	b.attentionDeltaInputKernel.SetArgs(b.inputGradBatchBuf, b.batchSlotsBuffer, b.batchTargetsBuf, b.batchScaleBuf, b.attentionDeltaBuf, int32(featuresPerSlot), int32(slotCount), int32(batchSize))
	b.enqueueKernel1D(b.attentionDeltaInputKernel, batchSize)
	return nil
}

func (b *openclMLPBackend) firstLayerInputGradSlice(mlp *MLP, inputOffset int, gradCount int) ([]float64, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	first := mlp.gpuLayers[0]
	b.firstInputGradsKernel.SetArgs(first.weights, first.deltas, b.targetsBuf, int32(first.inSize), int32(first.outSize), int32(inputOffset), int32(gradCount))
	b.enqueueKernel1D(b.firstInputGradsKernel, gradCount)
	out32 := make([]float32, gradCount)
	b.queue.EnqueueReadBufferFloat32(b.targetsBuf, true, 0, out32, nil)
	return float32To64(out32), nil
}

func (b *openclMLPBackend) firstLayerInputGradSliceBatch(mlp *MLP, inputOffset int, gradCount int, batchSize int) ([]float64, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	first := mlp.gpuLayers[0]
	b.firstInputGradsBatchKernel.SetArgs(first.weights, first.batchDel, b.batchInputBuf, int32(first.inSize), int32(first.outSize), int32(inputOffset), int32(gradCount), int32(batchSize))
	b.enqueueKernel1D(b.firstInputGradsBatchKernel, gradCount*batchSize)
	out32 := make([]float32, gradCount*batchSize)
	b.queue.EnqueueReadBufferFloat32(b.batchInputBuf, true, 0, out32, nil)
	return float32To64(out32), nil
}

func (b *openclMLPBackend) attentionOutputDeltasFromFirstLayerBatch(mlp *MLP, inputOffset int, rawSlotsBatch [][]float64, attentionWeightsBatch [][]float64, sampleWeights []float64, featuresPerSlot int, slotCount int) ([]float64, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	batchSize := len(rawSlotsBatch)
	flatS := make([]float32, batchSize*slotCount*featuresPerSlot)
	for s := 0; s < batchSize; s++ {
		copy(flatS[s*slotCount*featuresPerSlot:], float64To32(rawSlotsBatch[s]))
	}
	b.queue.EnqueueWriteBufferFloat32(b.batchSlotsBuffer, false, 0, flatS, nil)
	flatA := make([]float32, batchSize*slotCount)
	for s := 0; s < batchSize; s++ {
		copy(flatA[s*slotCount:], float64To32(attentionWeightsBatch[s]))
	}
	b.queue.EnqueueWriteBufferFloat32(b.batchTargetsBuf, false, 0, flatA, nil)
	b.queue.EnqueueWriteBufferFloat32(b.batchScaleBuf, false, 0, float64To32(sampleWeights), nil)
	first := mlp.gpuLayers[0]
	b.attentionOutputDeltasFromFirstLayerBatchKernel.SetArgs(first.weights, first.batchDel, b.batchSlotsBuffer, b.batchTargetsBuf, b.batchScaleBuf, b.batchInputBuf, int32(first.inSize), int32(first.outSize), int32(inputOffset), int32(featuresPerSlot), int32(slotCount), int32(batchSize))
	b.enqueueKernel1D(b.attentionOutputDeltasFromFirstLayerBatchKernel, slotCount*batchSize)
	out32 := make([]float32, slotCount*batchSize)
	b.queue.EnqueueReadBufferFloat32(b.batchInputBuf, true, 0, out32, nil)
	return float32To64(out32), nil
}

func (b *openclMLPBackend) applyAdamGradients(mlp *MLP, _ *WorkerCache, batchSize float64, lr, weightDecay, beta1, beta2, epsilon float64, beta1CorrInv, beta2CorrInv float64) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	invBatch := float32(1.0 / batchSize)
	for _, layer := range mlp.gpuLayers {
		wCount := layer.inSize * layer.outSize
		b.adamUpdateKernel.SetArgs(layer.weights, layer.weightM, layer.weightV, layer.gradWBuf, int32(wCount), float32(beta1), float32(beta2), float32(beta1CorrInv), float32(beta2CorrInv), float32(lr), float32(weightDecay), float32(epsilon), invBatch)
		b.enqueueKernel1D(b.adamUpdateKernel, wCount)
		b.adamUpdateKernel.SetArgs(layer.biases, layer.biasM, layer.biasV, layer.gradBBuf, int32(layer.outSize), float32(beta1), float32(beta2), float32(beta1CorrInv), float32(beta2CorrInv), float32(lr), float32(weightDecay), float32(epsilon), invBatch)
		b.enqueueKernel1D(b.adamUpdateKernel, layer.outSize)
	}
	return nil
}

func pickBestGPU() (*cl.Device, error) {
	platforms, _ := cl.GetPlatforms()
	var all []*cl.Device
	for _, p := range platforms {
		devs, _ := p.GetDevices(cl.DeviceTypeGPU)
		all = append(all, devs...)
	}
	if len(all) == 0 {
		return nil, fmt.Errorf("no GPU")
	}
	return all[0], nil
}

func adamBiasCorrectionInv(beta float64, step int64) float64 {
	pow := 1.0
	for i := int64(0); i < step; i++ {
		pow *= beta
	}
	return 1.0 / (1.0 - pow)
}

func chooseKernelLocalSize(dev *cl.Device, k *cl.Kernel) int {
	return 64
}

func flatten2DF64(m [][]float64) []float32 {
	rows, cols := len(m), len(m[0])
	res := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			res[i*cols+j] = float32(m[i][j])
		}
	}
	return res
}

func unflattenTo2DF64(m [][]float64, flat []float32) {
	rows, cols := len(m), len(m[0])
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m[i][j] = float64(flat[i*cols+j])
		}
	}
}

func float64To32(s []float64) []float32 {
	r := make([]float32, len(s))
	for i, v := range s {
		r[i] = float32(v)
	}
	return r
}

func float32To64(s []float32) []float64 {
	r := make([]float64, len(s))
	for i, v := range s {
		r[i] = float64(v)
	}
	return r
}

const (
	bnRunningMomentum = 0.9
	bnEpsilon         = 1.0e-5
)
