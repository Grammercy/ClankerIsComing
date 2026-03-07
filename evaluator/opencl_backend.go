//go:build opencl

package evaluator

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/jgillich/go-opencl/cl"
)

const mlpOpenCLSource = `
__kernel void dense_forward(
    __global const float* weights,
    __global const float* biases,
    __global const float* in,
    __global float* out,
    const int inSize,
    const int outSize,
    const int activation
) {
    int gid = get_global_id(0);
    if (gid >= outSize) return;

    float sum = biases[gid];
    int base = gid * inSize;
    for (int i = 0; i < inSize; i++) {
        sum += weights[base + i] * in[i];
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
    const float invValid
) {
    int gid = get_global_id(0);
    if (gid >= outSize) return;

    float t = targets[gid];
    if (t < 0.0f) {
        deltas[gid] = 0.0f;
    } else {
        deltas[gid] = (t - outputs[gid]) * invValid;
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
    __global const float* currentOutputs,
    __global float* currentDeltas,
    const int currentSize,
    const int nextSize
) {
    int gid = get_global_id(0);
    if (gid >= currentSize) return;

    float err = 0.0f;
    for (int k = 0; k < nextSize; k++) {
        err += nextWeights[k * currentSize + gid] * nextDeltas[k];
    }

    float o = currentOutputs[gid];
    float deriv = (o > 0.0f) ? 1.0f : 0.01f;
    currentDeltas[gid] = err * deriv;
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
    int col = gid - row * inSize;
    gradWeights[gid] += scale * deltas[row] * inputs[col];
}

__kernel void accumulate_bias_grads(
    __global const float* deltas,
    __global float* gradBiases,
    const int outSize,
    const float scale
) {
    int gid = get_global_id(0);
    if (gid >= outSize) return;

    gradBiases[gid] += scale * deltas[gid];
}

__kernel void adam_update(
    __global float* params,
    __global float* m,
    __global float* v,
    __global float* grads,
    const int count,
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
    if (gid >= count) return;

    float g = grads[gid] * invBatch;
    float mNew = beta1 * m[gid] + (1.0f - beta1) * g;
    float vNew = beta2 * v[gid] + (1.0f - beta2) * g * g;

    m[gid] = mNew;
    v[gid] = vNew;
    params[gid] = params[gid] * (1.0f - weightDecay) + lr * (mNew * beta1CorrInv) / (sqrt(vNew * beta2CorrInv) + epsilon);
    grads[gid] = 0.0f;
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

    int inputIdx = inputOffset + gid;
    float acc = 0.0f;
    for (int row = 0; row < outSize; row++) {
        acc += deltas[row] * weights[row * inSize + inputIdx];
    }
    outGrads[gid] = acc;
}

__kernel void dense_forward_batch(
    __global const float* weights,
    __global const float* biases,
    __global const float* in,
    __global float* out,
    const int inSize,
    const int outSize,
    const int batchSize,
    const int activation
) {
    int gid = get_global_id(0);
    int total = outSize * batchSize;
    if (gid >= total) return;

    int sample = gid / outSize;
    int neuron = gid - sample * outSize;

    float sum = biases[neuron];
    int wBase = neuron * inSize;
    int inBase = sample * inSize;
    for (int i = 0; i < inSize; i++) {
        sum += weights[wBase + i] * in[inBase + i];
    }

    if (activation == 0) {
        out[gid] = (sum > 0.0f) ? sum : (0.01f * sum);
    } else if (activation == 1) {
        out[gid] = 1.0f / (1.0f + exp(-sum));
    } else {
        out[gid] = sum;
    }
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

__kernel void hidden_delta_batch(
    __global const float* nextWeights,
    __global const float* nextDeltas,
    __global const float* currentOutputs,
    __global float* currentDeltas,
    const int currentSize,
    const int nextSize,
    const int batchSize
) {
    int gid = get_global_id(0);
    int total = currentSize * batchSize;
    if (gid >= total) return;

    int sample = gid / currentSize;
    int cur = gid - sample * currentSize;

    float err = 0.0f;
    int nextBase = sample * nextSize;
    for (int k = 0; k < nextSize; k++) {
        err += nextWeights[k * currentSize + cur] * nextDeltas[nextBase + k];
    }

    float o = currentOutputs[gid];
    float deriv = (o > 0.0f) ? 1.0f : 0.01f;
    currentDeltas[gid] = err * deriv;
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

__kernel void accumulate_weight_grads_batch(
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
    int col = gid - row * inSize;

    float sum = 0.0f;
    for (int s = 0; s < batchSize; s++) {
        sum += deltas[s * outSize + row] * inputs[s * inSize + col];
    }
    gradWeights[gid] += sum;
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
    int k = gid - sample * gradCount;
    int inputIdx = inputOffset + k;

    float acc = 0.0f;
    int deltaBase = sample * outSize;
    for (int row = 0; row < outSize; row++) {
        acc += deltas[deltaBase + row] * weights[row * inSize + inputIdx];
    }
    outGrads[gid] = acc;
}

__kernel void attention_output_deltas_from_input_grads(
    __global const float* inputGrads,
    __global const float* rawSlots,
    __global const float* attentionWeights,
    __global const float* sampleWeights,
    __global float* outDeltas,
    const int featuresPerSlot,
    const int slotCount,
    const int batchSize
) {
    int sample = get_global_id(0);
    if (sample >= batchSize) return;

    float dLocal[16];
    float dotDA = 0.0f;
    int gradBase = sample * featuresPerSlot * slotCount;
    int slotBase = sample * featuresPerSlot * slotCount;
    int attBase = sample * slotCount;

    for (int j = 0; j < slotCount; j++) {
        float d = 0.0f;
        int featBase = j * featuresPerSlot;
        for (int f = 0; f < featuresPerSlot; f++) {
            int idx = featBase + f;
            d += inputGrads[gradBase + idx] * rawSlots[slotBase + idx];
        }
        dLocal[j] = d;
        dotDA += d * attentionWeights[attBase + j];
    }

    float scale = sampleWeights[sample];
    for (int j = 0; j < slotCount; j++) {
        float a = attentionWeights[attBase + j];
        outDeltas[attBase + j] = scale * a * (dLocal[j] - dotDA);
    }
}
`

type openclLayerBuffers struct {
	inSize   int
	outSize  int
	weights  *cl.MemObject
	biases   *cl.MemObject
	weightM  *cl.MemObject
	weightV  *cl.MemObject
	biasM    *cl.MemObject
	biasV    *cl.MemObject
	outputs  *cl.MemObject
	deltas   *cl.MemObject
	batchOut *cl.MemObject
	batchDel *cl.MemObject
	gradWBuf *cl.MemObject
	gradBBuf *cl.MemObject
}

type openclMLPBackend struct {
	mu sync.Mutex

	ctx    *cl.Context
	queue  *cl.CommandQueue
	device *cl.Device

	program                    *cl.Program
	denseForwardKernel         *cl.Kernel
	outputDeltaBCEKernel       *cl.Kernel
	setOutputDeltasKernel      *cl.Kernel
	hiddenDeltaKernel          *cl.Kernel
	denseForwardBatchKernel    *cl.Kernel
	outputDeltaBCEBatchKernel  *cl.Kernel
	setOutputDeltasBatchKernel *cl.Kernel
	hiddenDeltaBatchKernel     *cl.Kernel
	accWeightGradsBatchKernel  *cl.Kernel
	accBiasGradsBatchKernel    *cl.Kernel
	accWeightGradsKernel       *cl.Kernel
	accBiasGradsKernel         *cl.Kernel
	adamUpdateKernel           *cl.Kernel
	firstInputGradsKernel      *cl.Kernel
	firstInputGradsBatchKernel *cl.Kernel
	attentionDeltaInputKernel  *cl.Kernel

	layers []*openclLayerBuffers

	scratchA          *cl.MemObject
	scratchB          *cl.MemObject
	targetsBuf        *cl.MemObject
	batchInputBuf     *cl.MemObject
	batchTargetsBuf   *cl.MemObject
	batchScaleBuf     *cl.MemObject
	attentionDeltaBuf *cl.MemObject
	inputGradBuf      *cl.MemObject
	inputGradBatchBuf *cl.MemObject
	workBufSize       int
	maxBatchSize      int
	lastBatchSize     int
}

func (b *openclMLPBackend) float32To64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i := range in {
		out[i] = float64(in[i])
	}
	return out
}

func (b *openclMLPBackend) maxBatchSizeLimit() int {
	if b == nil {
		return 1
	}
	if b.maxBatchSize < 1 {
		return 1
	}
	return b.maxBatchSize
}

func newOpenCLMLPBackend(mlp *MLP) (*openclMLPBackend, error) {
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

	mk := func(name string) (*cl.Kernel, error) {
		k, kErr := program.CreateKernel(name)
		if kErr != nil {
			return nil, fmt.Errorf("create kernel %q: %w", name, kErr)
		}
		return k, nil
	}

	denseForwardKernel, err := mk("dense_forward")
	if err != nil {
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	outputDeltaBCEKernel, err := mk("output_delta_bce")
	if err != nil {
		denseForwardKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	setOutputDeltasKernel, err := mk("set_output_deltas")
	if err != nil {
		denseForwardKernel.Release()
		outputDeltaBCEKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	hiddenDeltaKernel, err := mk("hidden_delta")
	if err != nil {
		denseForwardKernel.Release()
		outputDeltaBCEKernel.Release()
		setOutputDeltasKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	accWeightGradsKernel, err := mk("accumulate_weight_grads")
	if err != nil {
		denseForwardKernel.Release()
		outputDeltaBCEKernel.Release()
		setOutputDeltasKernel.Release()
		hiddenDeltaKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	accBiasGradsKernel, err := mk("accumulate_bias_grads")
	if err != nil {
		denseForwardKernel.Release()
		outputDeltaBCEKernel.Release()
		setOutputDeltasKernel.Release()
		hiddenDeltaKernel.Release()
		accWeightGradsKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	adamUpdateKernel, err := mk("adam_update")
	if err != nil {
		denseForwardKernel.Release()
		outputDeltaBCEKernel.Release()
		setOutputDeltasKernel.Release()
		hiddenDeltaKernel.Release()
		accWeightGradsKernel.Release()
		accBiasGradsKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	firstInputGradsKernel, err := mk("first_layer_input_grads")
	if err != nil {
		denseForwardKernel.Release()
		outputDeltaBCEKernel.Release()
		setOutputDeltasKernel.Release()
		hiddenDeltaKernel.Release()
		accWeightGradsKernel.Release()
		accBiasGradsKernel.Release()
		adamUpdateKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	denseForwardBatchKernel, err := mk("dense_forward_batch")
	if err != nil {
		denseForwardKernel.Release()
		outputDeltaBCEKernel.Release()
		setOutputDeltasKernel.Release()
		hiddenDeltaKernel.Release()
		accWeightGradsKernel.Release()
		accBiasGradsKernel.Release()
		adamUpdateKernel.Release()
		firstInputGradsKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	outputDeltaBCEBatchKernel, err := mk("output_delta_bce_batch")
	if err != nil {
		denseForwardKernel.Release()
		outputDeltaBCEKernel.Release()
		setOutputDeltasKernel.Release()
		hiddenDeltaKernel.Release()
		accWeightGradsKernel.Release()
		accBiasGradsKernel.Release()
		adamUpdateKernel.Release()
		firstInputGradsKernel.Release()
		denseForwardBatchKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	setOutputDeltasBatchKernel, err := mk("set_output_deltas_batch")
	if err != nil {
		denseForwardKernel.Release()
		outputDeltaBCEKernel.Release()
		setOutputDeltasKernel.Release()
		hiddenDeltaKernel.Release()
		accWeightGradsKernel.Release()
		accBiasGradsKernel.Release()
		adamUpdateKernel.Release()
		firstInputGradsKernel.Release()
		denseForwardBatchKernel.Release()
		outputDeltaBCEBatchKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	hiddenDeltaBatchKernel, err := mk("hidden_delta_batch")
	if err != nil {
		denseForwardKernel.Release()
		outputDeltaBCEKernel.Release()
		setOutputDeltasKernel.Release()
		hiddenDeltaKernel.Release()
		accWeightGradsKernel.Release()
		accBiasGradsKernel.Release()
		adamUpdateKernel.Release()
		firstInputGradsKernel.Release()
		denseForwardBatchKernel.Release()
		outputDeltaBCEBatchKernel.Release()
		setOutputDeltasBatchKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	accWeightGradsBatchKernel, err := mk("accumulate_weight_grads_batch")
	if err != nil {
		denseForwardKernel.Release()
		outputDeltaBCEKernel.Release()
		setOutputDeltasKernel.Release()
		hiddenDeltaKernel.Release()
		accWeightGradsKernel.Release()
		accBiasGradsKernel.Release()
		adamUpdateKernel.Release()
		firstInputGradsKernel.Release()
		denseForwardBatchKernel.Release()
		outputDeltaBCEBatchKernel.Release()
		setOutputDeltasBatchKernel.Release()
		hiddenDeltaBatchKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	accBiasGradsBatchKernel, err := mk("accumulate_bias_grads_batch")
	if err != nil {
		denseForwardKernel.Release()
		outputDeltaBCEKernel.Release()
		setOutputDeltasKernel.Release()
		hiddenDeltaKernel.Release()
		accWeightGradsKernel.Release()
		accBiasGradsKernel.Release()
		adamUpdateKernel.Release()
		firstInputGradsKernel.Release()
		denseForwardBatchKernel.Release()
		outputDeltaBCEBatchKernel.Release()
		setOutputDeltasBatchKernel.Release()
		hiddenDeltaBatchKernel.Release()
		accWeightGradsBatchKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	firstInputGradsBatchKernel, err := mk("first_layer_input_grads_batch")
	if err != nil {
		denseForwardKernel.Release()
		outputDeltaBCEKernel.Release()
		setOutputDeltasKernel.Release()
		hiddenDeltaKernel.Release()
		accWeightGradsKernel.Release()
		accBiasGradsKernel.Release()
		adamUpdateKernel.Release()
		firstInputGradsKernel.Release()
		denseForwardBatchKernel.Release()
		outputDeltaBCEBatchKernel.Release()
		setOutputDeltasBatchKernel.Release()
		hiddenDeltaBatchKernel.Release()
		accWeightGradsBatchKernel.Release()
		accBiasGradsBatchKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	attentionDeltaInputKernel, err := mk("attention_output_deltas_from_input_grads")
	if err != nil {
		denseForwardKernel.Release()
		outputDeltaBCEKernel.Release()
		setOutputDeltasKernel.Release()
		hiddenDeltaKernel.Release()
		accWeightGradsKernel.Release()
		accBiasGradsKernel.Release()
		adamUpdateKernel.Release()
		firstInputGradsKernel.Release()
		denseForwardBatchKernel.Release()
		outputDeltaBCEBatchKernel.Release()
		setOutputDeltasBatchKernel.Release()
		hiddenDeltaBatchKernel.Release()
		accWeightGradsBatchKernel.Release()
		accBiasGradsBatchKernel.Release()
		firstInputGradsBatchKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}

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
		layers:                     make([]*openclLayerBuffers, len(mlp.Layers)),
	}

	createRWBufferWithData := func(data []float32) (*cl.MemObject, error) {
		buf, e := ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*len(data))
		if e != nil {
			return nil, e
		}
		if len(data) > 0 {
			if _, e = queue.EnqueueWriteBufferFloat32(buf, false, 0, data, nil); e != nil {
				buf.Release()
				return nil, e
			}
		}
		return buf, nil
	}

	maxBatch := 1024
	if env := strings.TrimSpace(os.Getenv("OPENCL_MAX_BATCH")); env != "" {
		if v, convErr := strconv.Atoi(env); convErr == nil && v > 0 {
			maxBatch = v
		}
	}
	backend.maxBatchSize = maxBatch

	maxWidth := 0
	for i, layer := range mlp.Layers {
		inSize := len(layer.Weights[0])
		outSize := len(layer.Weights)
		if inSize > maxWidth {
			maxWidth = inSize
		}
		if outSize > maxWidth {
			maxWidth = outSize
		}

		weights := flatten2DF64(layer.Weights)
		biases := float64To32(layer.Biases)
		weightM := make([]float32, len(weights))
		weightV := make([]float32, len(weights))
		biasM := make([]float32, len(biases))
		biasV := make([]float32, len(biases))

		wBuf, e := createRWBufferWithData(weights)
		if e != nil {
			backend.Release()
			return nil, fmt.Errorf("create weights buffer layer %d: %w", i, e)
		}
		bBuf, e := createRWBufferWithData(biases)
		if e != nil {
			wBuf.Release()
			backend.Release()
			return nil, fmt.Errorf("create bias buffer layer %d: %w", i, e)
		}
		wmBuf, e := createRWBufferWithData(weightM)
		if e != nil {
			wBuf.Release()
			bBuf.Release()
			backend.Release()
			return nil, fmt.Errorf("create weight m buffer layer %d: %w", i, e)
		}
		wvBuf, e := createRWBufferWithData(weightV)
		if e != nil {
			wBuf.Release()
			bBuf.Release()
			wmBuf.Release()
			backend.Release()
			return nil, fmt.Errorf("create weight v buffer layer %d: %w", i, e)
		}
		bmBuf, e := createRWBufferWithData(biasM)
		if e != nil {
			wBuf.Release()
			bBuf.Release()
			wmBuf.Release()
			wvBuf.Release()
			backend.Release()
			return nil, fmt.Errorf("create bias m buffer layer %d: %w", i, e)
		}
		bvBuf, e := createRWBufferWithData(biasV)
		if e != nil {
			wBuf.Release()
			bBuf.Release()
			wmBuf.Release()
			wvBuf.Release()
			bmBuf.Release()
			backend.Release()
			return nil, fmt.Errorf("create bias v buffer layer %d: %w", i, e)
		}
		outBuf, e := ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*outSize)
		if e != nil {
			wBuf.Release()
			bBuf.Release()
			wmBuf.Release()
			wvBuf.Release()
			bmBuf.Release()
			bvBuf.Release()
			backend.Release()
			return nil, fmt.Errorf("create output buffer layer %d: %w", i, e)
		}
		deltaBuf, e := ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*outSize)
		if e != nil {
			wBuf.Release()
			bBuf.Release()
			wmBuf.Release()
			wvBuf.Release()
			bmBuf.Release()
			bvBuf.Release()
			outBuf.Release()
			backend.Release()
			return nil, fmt.Errorf("create delta buffer layer %d: %w", i, e)
		}
		batchOutBuf, e := ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*outSize*maxBatch)
		if e != nil {
			wBuf.Release()
			bBuf.Release()
			wmBuf.Release()
			wvBuf.Release()
			bmBuf.Release()
			bvBuf.Release()
			outBuf.Release()
			deltaBuf.Release()
			backend.Release()
			return nil, fmt.Errorf("create batch output buffer layer %d: %w", i, e)
		}
		batchDelBuf, e := ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*outSize*maxBatch)
		if e != nil {
			wBuf.Release()
			bBuf.Release()
			wmBuf.Release()
			wvBuf.Release()
			bmBuf.Release()
			bvBuf.Release()
			outBuf.Release()
			deltaBuf.Release()
			batchOutBuf.Release()
			backend.Release()
			return nil, fmt.Errorf("create batch delta buffer layer %d: %w", i, e)
		}
		gradW, e := createRWBufferWithData(make([]float32, len(weights)))
		if e != nil {
			wBuf.Release()
			bBuf.Release()
			wmBuf.Release()
			wvBuf.Release()
			bmBuf.Release()
			bvBuf.Release()
			outBuf.Release()
			deltaBuf.Release()
			batchOutBuf.Release()
			batchDelBuf.Release()
			backend.Release()
			return nil, fmt.Errorf("create gradW buffer layer %d: %w", i, e)
		}
		gradB, e := createRWBufferWithData(make([]float32, len(biases)))
		if e != nil {
			wBuf.Release()
			bBuf.Release()
			wmBuf.Release()
			wvBuf.Release()
			bmBuf.Release()
			bvBuf.Release()
			outBuf.Release()
			deltaBuf.Release()
			batchOutBuf.Release()
			batchDelBuf.Release()
			gradW.Release()
			backend.Release()
			return nil, fmt.Errorf("create gradB buffer layer %d: %w", i, e)
		}

		backend.layers[i] = &openclLayerBuffers{
			inSize:   inSize,
			outSize:  outSize,
			weights:  wBuf,
			biases:   bBuf,
			weightM:  wmBuf,
			weightV:  wvBuf,
			biasM:    bmBuf,
			biasV:    bvBuf,
			outputs:  outBuf,
			deltas:   deltaBuf,
			batchOut: batchOutBuf,
			batchDel: batchDelBuf,
			gradWBuf: gradW,
			gradBBuf: gradB,
		}
	}

	if maxWidth < 1 {
		maxWidth = 1
	}
	backend.workBufSize = maxWidth
	backend.scratchA, err = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth)
	if err != nil {
		backend.Release()
		return nil, fmt.Errorf("create scratchA buffer: %w", err)
	}
	backend.scratchB, err = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth)
	if err != nil {
		backend.Release()
		return nil, fmt.Errorf("create scratchB buffer: %w", err)
	}
	// Targets/output-delta staging buffers must accommodate the widest output
	// layer (not a fixed action count), since MoE router output can be larger.
	backend.targetsBuf, err = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth)
	if err != nil {
		backend.Release()
		return nil, fmt.Errorf("create targets buffer: %w", err)
	}
	backend.inputGradBuf, err = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth)
	if err != nil {
		backend.Release()
		return nil, fmt.Errorf("create input gradient buffer: %w", err)
	}
	backend.batchInputBuf, err = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth*maxBatch)
	if err != nil {
		backend.Release()
		return nil, fmt.Errorf("create batch input buffer: %w", err)
	}
	backend.batchTargetsBuf, err = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth*maxBatch)
	if err != nil {
		backend.Release()
		return nil, fmt.Errorf("create batch targets buffer: %w", err)
	}
	backend.batchScaleBuf, err = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxBatch)
	if err != nil {
		backend.Release()
		return nil, fmt.Errorf("create batch scale buffer: %w", err)
	}
	backend.attentionDeltaBuf, err = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth*maxBatch)
	if err != nil {
		backend.Release()
		return nil, fmt.Errorf("create attention delta buffer: %w", err)
	}
	backend.inputGradBatchBuf, err = ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*maxWidth*maxBatch)
	if err != nil {
		backend.Release()
		return nil, fmt.Errorf("create batch input gradient buffer: %w", err)
	}

	return backend, nil
}

func (b *openclMLPBackend) Release() {
	if b == nil {
		return
	}

	for _, l := range b.layers {
		if l == nil {
			continue
		}
		if l.weights != nil {
			l.weights.Release()
		}
		if l.biases != nil {
			l.biases.Release()
		}
		if l.weightM != nil {
			l.weightM.Release()
		}
		if l.weightV != nil {
			l.weightV.Release()
		}
		if l.biasM != nil {
			l.biasM.Release()
		}
		if l.biasV != nil {
			l.biasV.Release()
		}
		if l.outputs != nil {
			l.outputs.Release()
		}
		if l.deltas != nil {
			l.deltas.Release()
		}
		if l.batchOut != nil {
			l.batchOut.Release()
		}
		if l.batchDel != nil {
			l.batchDel.Release()
		}
		if l.gradWBuf != nil {
			l.gradWBuf.Release()
		}
		if l.gradBBuf != nil {
			l.gradBBuf.Release()
		}
	}
	if b.scratchA != nil {
		b.scratchA.Release()
	}
	if b.scratchB != nil {
		b.scratchB.Release()
	}
	if b.targetsBuf != nil {
		b.targetsBuf.Release()
	}
	if b.batchInputBuf != nil {
		b.batchInputBuf.Release()
	}
	if b.batchTargetsBuf != nil {
		b.batchTargetsBuf.Release()
	}
	if b.batchScaleBuf != nil {
		b.batchScaleBuf.Release()
	}
	if b.attentionDeltaBuf != nil {
		b.attentionDeltaBuf.Release()
	}
	if b.inputGradBuf != nil {
		b.inputGradBuf.Release()
	}
	if b.inputGradBatchBuf != nil {
		b.inputGradBatchBuf.Release()
	}

	if b.denseForwardKernel != nil {
		b.denseForwardKernel.Release()
	}
	if b.outputDeltaBCEKernel != nil {
		b.outputDeltaBCEKernel.Release()
	}
	if b.setOutputDeltasKernel != nil {
		b.setOutputDeltasKernel.Release()
	}
	if b.hiddenDeltaKernel != nil {
		b.hiddenDeltaKernel.Release()
	}
	if b.denseForwardBatchKernel != nil {
		b.denseForwardBatchKernel.Release()
	}
	if b.outputDeltaBCEBatchKernel != nil {
		b.outputDeltaBCEBatchKernel.Release()
	}
	if b.setOutputDeltasBatchKernel != nil {
		b.setOutputDeltasBatchKernel.Release()
	}
	if b.hiddenDeltaBatchKernel != nil {
		b.hiddenDeltaBatchKernel.Release()
	}
	if b.accWeightGradsBatchKernel != nil {
		b.accWeightGradsBatchKernel.Release()
	}
	if b.accBiasGradsBatchKernel != nil {
		b.accBiasGradsBatchKernel.Release()
	}
	if b.accWeightGradsKernel != nil {
		b.accWeightGradsKernel.Release()
	}
	if b.accBiasGradsKernel != nil {
		b.accBiasGradsKernel.Release()
	}
	if b.adamUpdateKernel != nil {
		b.adamUpdateKernel.Release()
	}
	if b.firstInputGradsKernel != nil {
		b.firstInputGradsKernel.Release()
	}
	if b.firstInputGradsBatchKernel != nil {
		b.firstInputGradsBatchKernel.Release()
	}
	if b.attentionDeltaInputKernel != nil {
		b.attentionDeltaInputKernel.Release()
	}
	if b.program != nil {
		b.program.Release()
	}
	if b.queue != nil {
		b.queue.Release()
	}
	if b.ctx != nil {
		b.ctx.Release()
	}
}

func (b *openclMLPBackend) syncParamsFromHost(mlp *MLP) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	for i, layer := range mlp.Layers {
		w := flatten2DF64(layer.Weights)
		bv := float64To32(layer.Biases)
		if _, err := b.queue.EnqueueWriteBufferFloat32(b.layers[i].weights, true, 0, w, nil); err != nil {
			return fmt.Errorf("upload weights layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueWriteBufferFloat32(b.layers[i].biases, true, 0, bv, nil); err != nil {
			return fmt.Errorf("upload biases layer %d: %w", i, err)
		}
	}
	return nil
}

func (b *openclMLPBackend) syncParamsToHost(mlp *MLP) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	for i := range mlp.Layers {
		layer := b.layers[i]
		weightCount := layer.inSize * layer.outSize
		wOut := make([]float32, weightCount)
		bOut := make([]float32, layer.outSize)

		if _, err := b.queue.EnqueueReadBufferFloat32(layer.weights, true, 0, wOut, nil); err != nil {
			return fmt.Errorf("download weights layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueReadBufferFloat32(layer.biases, true, 0, bOut, nil); err != nil {
			return fmt.Errorf("download biases layer %d: %w", i, err)
		}

		unflattenTo2DF64(mlp.Layers[i].Weights, wOut)
		for j := range bOut {
			mlp.Layers[i].Biases[j] = float64(bOut[j])
		}
	}
	return b.queue.Finish()
}

func (b *openclMLPBackend) clearGradients() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	for i, layer := range b.layers {
		zeroW := make([]float32, layer.inSize*layer.outSize)
		zeroB := make([]float32, layer.outSize)
		if _, err := b.queue.EnqueueWriteBufferFloat32(layer.gradWBuf, false, 0, zeroW, nil); err != nil {
			return fmt.Errorf("clear weight gradients layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueWriteBufferFloat32(layer.gradBBuf, false, 0, zeroB, nil); err != nil {
			return fmt.Errorf("clear bias gradients layer %d: %w", i, err)
		}
	}
	return b.queue.Finish()
}

func (b *openclMLPBackend) forward(mlp *MLP, inputs []float64, cache *InferenceCache) ([]float64, error) {
	if len(inputs) > b.workBufSize {
		return nil, fmt.Errorf("input size %d exceeds OpenCL work buffer %d", len(inputs), b.workBufSize)
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	in32 := float64To32(inputs)
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.scratchA, false, 0, in32, nil); err != nil {
		return nil, fmt.Errorf("write forward input: %w", err)
	}

	current := b.scratchA
	next := b.scratchB
	lastOutSize := len(inputs)

	for i, layer := range b.layers {
		activation := int32(0)
		if i == len(b.layers)-1 {
			if mlp.LinearOutput {
				activation = 2
			} else {
				activation = 1
			}
		}

		if err := b.denseForwardKernel.SetArgs(layer.weights, layer.biases, current, next, int32(lastOutSize), int32(layer.outSize), activation); err != nil {
			return nil, fmt.Errorf("set dense_forward args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.denseForwardKernel, nil, []int{layer.outSize}, nil, nil); err != nil {
			return nil, fmt.Errorf("enqueue dense_forward layer %d: %w", i, err)
		}

		if cache != nil && i < len(cache.Outputs) {
			tmp := make([]float32, layer.outSize)
			if _, err := b.queue.EnqueueReadBufferFloat32(next, true, 0, tmp, nil); err != nil {
				return nil, fmt.Errorf("read forward output layer %d: %w", i, err)
			}
			for j := range tmp {
				cache.Outputs[i][j] = float64(tmp[j])
			}
		}

		current, next = next, current
		lastOutSize = layer.outSize
	}

	out32 := make([]float32, lastOutSize)
	if _, err := b.queue.EnqueueReadBufferFloat32(current, true, 0, out32, nil); err != nil {
		return nil, fmt.Errorf("read final forward output: %w", err)
	}
	out := float32To64(out32)
	return out, nil
}

func (b *openclMLPBackend) forwardBatch(mlp *MLP, inputsBatch [][]float64) ([][]float64, error) {
	batchSize := len(inputsBatch)
	if batchSize == 0 {
		return nil, nil
	}
	if batchSize > b.maxBatchSize {
		return nil, fmt.Errorf("batch size %d exceeds OPENCL_MAX_BATCH %d", batchSize, b.maxBatchSize)
	}

	inSize := len(inputsBatch[0])
	if inSize > b.workBufSize {
		return nil, fmt.Errorf("input size %d exceeds OpenCL work buffer %d", inSize, b.workBufSize)
	}
	for i := 1; i < batchSize; i++ {
		if len(inputsBatch[i]) != inSize {
			return nil, fmt.Errorf("input batch element %d has size %d, expected %d", i, len(inputsBatch[i]), inSize)
		}
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	inFlat := make([]float32, batchSize*inSize)
	for s := 0; s < batchSize; s++ {
		base := s * inSize
		for i := 0; i < inSize; i++ {
			inFlat[base+i] = float32(inputsBatch[s][i])
		}
	}
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.batchInputBuf, false, 0, inFlat, nil); err != nil {
		return nil, fmt.Errorf("write batch inputs: %w", err)
	}

	currentInputBuf := b.batchInputBuf
	currentInputSize := inSize
	for i, layer := range b.layers {
		activation := int32(0)
		if i == len(b.layers)-1 {
			if mlp.LinearOutput {
				activation = 2
			} else {
				activation = 1
			}
		}

		if err := b.denseForwardBatchKernel.SetArgs(
			layer.weights, layer.biases, currentInputBuf, layer.batchOut,
			int32(currentInputSize), int32(layer.outSize), int32(batchSize), activation,
		); err != nil {
			return nil, fmt.Errorf("set dense_forward_batch args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.denseForwardBatchKernel, nil, []int{layer.outSize * batchSize}, nil, nil); err != nil {
			return nil, fmt.Errorf("enqueue dense_forward_batch layer %d: %w", i, err)
		}

		currentInputBuf = layer.batchOut
		currentInputSize = layer.outSize
	}

	outSize := b.layers[len(b.layers)-1].outSize
	outFlat := make([]float32, batchSize*outSize)
	if _, err := b.queue.EnqueueReadBufferFloat32(b.layers[len(b.layers)-1].batchOut, true, 0, outFlat, nil); err != nil {
		return nil, fmt.Errorf("read batch outputs: %w", err)
	}

	out := make([][]float64, batchSize)
	for s := 0; s < batchSize; s++ {
		row := make([]float64, outSize)
		base := s * outSize
		for i := 0; i < outSize; i++ {
			row[i] = float64(outFlat[base+i])
		}
		out[s] = row
	}
	return out, nil
}

func (b *openclMLPBackend) forwardAndStore(mlp *MLP, inputs []float64, cache *WorkerCache) error {
	in32 := float64To32(inputs)
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.scratchA, false, 0, in32, nil); err != nil {
		return fmt.Errorf("write training input: %w", err)
	}

	currentInputBuf := b.scratchA
	currentInputSize := len(inputs)
	for i, layer := range b.layers {
		activation := int32(0)
		if i == len(b.layers)-1 {
			if mlp.LinearOutput {
				activation = 2
			} else {
				activation = 1
			}
		}
		if err := b.denseForwardKernel.SetArgs(layer.weights, layer.biases, currentInputBuf, layer.outputs, int32(currentInputSize), int32(layer.outSize), activation); err != nil {
			return fmt.Errorf("set dense_forward(train) args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.denseForwardKernel, nil, []int{layer.outSize}, nil, nil); err != nil {
			return fmt.Errorf("enqueue dense_forward(train) layer %d: %w", i, err)
		}

		if cache != nil && i < len(cache.Outputs) {
			tmp := make([]float32, layer.outSize)
			if _, err := b.queue.EnqueueReadBufferFloat32(layer.outputs, true, 0, tmp, nil); err != nil {
				return fmt.Errorf("read train output layer %d: %w", i, err)
			}
			for j := range tmp {
				cache.Outputs[i][j] = float64(tmp[j])
			}
		}

		currentInputBuf = layer.outputs
		currentInputSize = layer.outSize
	}

	return nil
}

func (b *openclMLPBackend) calculateBCELocalGradients(mlp *MLP, inputs []float64, targets []float64, eloWeight float64, cache *WorkerCache) (float64, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if err := b.forwardAndStore(mlp, inputs, cache); err != nil {
		return 0, err
	}

	outIdx := len(b.layers) - 1
	outSize := b.layers[outIdx].outSize
	if len(targets) != outSize {
		return 0, fmt.Errorf("targets size mismatch: got %d want %d", len(targets), outSize)
	}

	out32 := make([]float32, outSize)
	if _, err := b.queue.EnqueueReadBufferFloat32(b.layers[outIdx].outputs, true, 0, out32, nil); err != nil {
		return 0, fmt.Errorf("read output for BCE: %w", err)
	}

	validTargets := 0.0
	loss := 0.0
	for i := 0; i < outSize; i++ {
		t := targets[i]
		if t < 0 {
			continue
		}
		validTargets++
		loss += bceLoss(t, float64(out32[i]))
	}
	if validTargets > 0 {
		loss /= validTargets
	}

	targets32 := float64To32(targets)
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.targetsBuf, false, 0, targets32, nil); err != nil {
		return 0, fmt.Errorf("write BCE targets: %w", err)
	}
	invValid := float32(0.0)
	if validTargets > 0 {
		invValid = float32(1.0 / validTargets)
	}
	if err := b.outputDeltaBCEKernel.SetArgs(b.layers[outIdx].outputs, b.targetsBuf, b.layers[outIdx].deltas, int32(outSize), invValid); err != nil {
		return 0, fmt.Errorf("set output_delta_bce args: %w", err)
	}
	if _, err := b.queue.EnqueueNDRangeKernel(b.outputDeltaBCEKernel, nil, []int{outSize}, nil, nil); err != nil {
		return 0, fmt.Errorf("enqueue output_delta_bce: %w", err)
	}

	for i := len(b.layers) - 2; i >= 0; i-- {
		cur := b.layers[i]
		next := b.layers[i+1]
		if err := b.hiddenDeltaKernel.SetArgs(next.weights, next.deltas, cur.outputs, cur.deltas, int32(cur.outSize), int32(next.outSize)); err != nil {
			return 0, fmt.Errorf("set hidden_delta args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.hiddenDeltaKernel, nil, []int{cur.outSize}, nil, nil); err != nil {
			return 0, fmt.Errorf("enqueue hidden_delta layer %d: %w", i, err)
		}
	}

	for i, layer := range b.layers {
		inputBuf := b.scratchA
		inSize := layer.inSize
		if i > 0 {
			inputBuf = b.layers[i-1].outputs
			inSize = b.layers[i-1].outSize
		}

		scale := float32(eloWeight)
		weightCount := layer.inSize * layer.outSize
		if err := b.accWeightGradsKernel.SetArgs(layer.deltas, inputBuf, layer.gradWBuf, int32(inSize), int32(layer.outSize), scale); err != nil {
			return 0, fmt.Errorf("set accumulate_weight_grads args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.accWeightGradsKernel, nil, []int{weightCount}, nil, nil); err != nil {
			return 0, fmt.Errorf("enqueue accumulate_weight_grads layer %d: %w", i, err)
		}

		if err := b.accBiasGradsKernel.SetArgs(layer.deltas, layer.gradBBuf, int32(layer.outSize), scale); err != nil {
			return 0, fmt.Errorf("set accumulate_bias_grads args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.accBiasGradsKernel, nil, []int{layer.outSize}, nil, nil); err != nil {
			return 0, fmt.Errorf("enqueue accumulate_bias_grads layer %d: %w", i, err)
		}
	}

	return loss, nil
}

func (b *openclMLPBackend) calculateBCELocalGradientsBatch(mlp *MLP, inputsBatch [][]float64, targetsBatch [][]float64, eloWeights []float64) (float64, [][]float64, error) {
	batchSize := len(inputsBatch)
	if batchSize == 0 {
		return 0, nil, nil
	}
	if len(targetsBatch) != batchSize || len(eloWeights) != batchSize {
		return 0, nil, fmt.Errorf("batch sizes mismatch: inputs=%d targets=%d weights=%d", batchSize, len(targetsBatch), len(eloWeights))
	}
	if batchSize > b.maxBatchSize {
		return 0, nil, fmt.Errorf("batch size %d exceeds OPENCL_MAX_BATCH %d", batchSize, b.maxBatchSize)
	}

	inSize := len(inputsBatch[0])
	if inSize > b.workBufSize {
		return 0, nil, fmt.Errorf("input size %d exceeds OpenCL work buffer %d", inSize, b.workBufSize)
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	for i := 1; i < batchSize; i++ {
		if len(inputsBatch[i]) != inSize {
			return 0, nil, fmt.Errorf("input batch element %d has size %d, expected %d", i, len(inputsBatch[i]), inSize)
		}
	}

	outSize := b.layers[len(b.layers)-1].outSize
	for i := 0; i < batchSize; i++ {
		if len(targetsBatch[i]) != outSize {
			return 0, nil, fmt.Errorf("target batch element %d has size %d, expected %d", i, len(targetsBatch[i]), outSize)
		}
	}

	inFlat := make([]float32, batchSize*inSize)
	for s := 0; s < batchSize; s++ {
		base := s * inSize
		for i := 0; i < inSize; i++ {
			inFlat[base+i] = float32(inputsBatch[s][i])
		}
	}
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.batchInputBuf, false, 0, inFlat, nil); err != nil {
		return 0, nil, fmt.Errorf("write batch inputs: %w", err)
	}

	currentInputBuf := b.batchInputBuf
	currentInputSize := inSize
	for i, layer := range b.layers {
		activation := int32(0)
		if i == len(b.layers)-1 {
			if mlp.LinearOutput {
				activation = 2
			} else {
				activation = 1
			}
		}

		if err := b.denseForwardBatchKernel.SetArgs(
			layer.weights, layer.biases, currentInputBuf, layer.batchOut,
			int32(currentInputSize), int32(layer.outSize), int32(batchSize), activation,
		); err != nil {
			return 0, nil, fmt.Errorf("set dense_forward_batch args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.denseForwardBatchKernel, nil, []int{layer.outSize * batchSize}, nil, nil); err != nil {
			return 0, nil, fmt.Errorf("enqueue dense_forward_batch layer %d: %w", i, err)
		}

		currentInputBuf = layer.batchOut
		currentInputSize = layer.outSize
	}

	outputFlat := make([]float32, batchSize*outSize)
	if _, err := b.queue.EnqueueReadBufferFloat32(b.layers[len(b.layers)-1].batchOut, true, 0, outputFlat, nil); err != nil {
		return 0, nil, fmt.Errorf("read batch outputs: %w", err)
	}

	outputs64 := make([][]float64, batchSize)
	for s := 0; s < batchSize; s++ {
		row := make([]float64, outSize)
		base := s * outSize
		for i := 0; i < outSize; i++ {
			row[i] = float64(outputFlat[base+i])
		}
		outputs64[s] = row
	}

	targetFlat := make([]float32, batchSize*outSize)
	sampleScale := make([]float32, batchSize)
	totalLoss := 0.0
	for s := 0; s < batchSize; s++ {
		valid := 0.0
		loss := 0.0
		base := s * outSize
		for i := 0; i < outSize; i++ {
			t := targetsBatch[s][i]
			targetFlat[base+i] = float32(t)
			if t < 0 {
				continue
			}
			valid++
			loss += bceLoss(t, float64(outputFlat[base+i]))
		}
		if valid > 0 {
			totalLoss += loss / valid
			sampleScale[s] = float32(eloWeights[s] / valid)
		} else {
			sampleScale[s] = 0
		}
	}

	if _, err := b.queue.EnqueueWriteBufferFloat32(b.batchTargetsBuf, false, 0, targetFlat, nil); err != nil {
		return 0, nil, fmt.Errorf("write batch targets: %w", err)
	}
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.batchScaleBuf, false, 0, sampleScale, nil); err != nil {
		return 0, nil, fmt.Errorf("write batch scales: %w", err)
	}

	last := b.layers[len(b.layers)-1]
	if err := b.outputDeltaBCEBatchKernel.SetArgs(last.batchOut, b.batchTargetsBuf, b.batchScaleBuf, last.batchDel, int32(outSize), int32(batchSize)); err != nil {
		return 0, nil, fmt.Errorf("set output_delta_bce_batch args: %w", err)
	}
	if _, err := b.queue.EnqueueNDRangeKernel(b.outputDeltaBCEBatchKernel, nil, []int{outSize * batchSize}, nil, nil); err != nil {
		return 0, nil, fmt.Errorf("enqueue output_delta_bce_batch: %w", err)
	}

	for i := len(b.layers) - 2; i >= 0; i-- {
		cur := b.layers[i]
		next := b.layers[i+1]
		if err := b.hiddenDeltaBatchKernel.SetArgs(next.weights, next.batchDel, cur.batchOut, cur.batchDel, int32(cur.outSize), int32(next.outSize), int32(batchSize)); err != nil {
			return 0, nil, fmt.Errorf("set hidden_delta_batch args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.hiddenDeltaBatchKernel, nil, []int{cur.outSize * batchSize}, nil, nil); err != nil {
			return 0, nil, fmt.Errorf("enqueue hidden_delta_batch layer %d: %w", i, err)
		}
	}

	for i, layer := range b.layers {
		inputBuf := b.batchInputBuf
		inputSize := inSize
		if i > 0 {
			inputBuf = b.layers[i-1].batchOut
			inputSize = b.layers[i-1].outSize
		}

		weightCount := layer.inSize * layer.outSize
		if err := b.accWeightGradsBatchKernel.SetArgs(layer.batchDel, inputBuf, layer.gradWBuf, int32(inputSize), int32(layer.outSize), int32(batchSize)); err != nil {
			return 0, nil, fmt.Errorf("set accumulate_weight_grads_batch args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.accWeightGradsBatchKernel, nil, []int{weightCount}, nil, nil); err != nil {
			return 0, nil, fmt.Errorf("enqueue accumulate_weight_grads_batch layer %d: %w", i, err)
		}

		if err := b.accBiasGradsBatchKernel.SetArgs(layer.batchDel, layer.gradBBuf, int32(layer.outSize), int32(batchSize)); err != nil {
			return 0, nil, fmt.Errorf("set accumulate_bias_grads_batch args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.accBiasGradsBatchKernel, nil, []int{layer.outSize}, nil, nil); err != nil {
			return 0, nil, fmt.Errorf("enqueue accumulate_bias_grads_batch layer %d: %w", i, err)
		}
	}

	b.lastBatchSize = batchSize
	return totalLoss, outputs64, nil
}

func (b *openclMLPBackend) backpropGivenDeltas(mlp *MLP, inputs []float64, outputDeltas []float64, eloWeight float64, cache *WorkerCache) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if err := b.forwardAndStore(mlp, inputs, cache); err != nil {
		return err
	}

	outIdx := len(b.layers) - 1
	outSize := b.layers[outIdx].outSize
	if len(outputDeltas) != outSize {
		return fmt.Errorf("output deltas size mismatch: got %d want %d", len(outputDeltas), outSize)
	}
	outDelta32 := float64To32(outputDeltas)
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.targetsBuf, false, 0, outDelta32, nil); err != nil {
		return fmt.Errorf("write output deltas: %w", err)
	}
	linearFlag := int32(0)
	if mlp.LinearOutput {
		linearFlag = 1
	}
	if err := b.setOutputDeltasKernel.SetArgs(b.layers[outIdx].outputs, b.targetsBuf, b.layers[outIdx].deltas, int32(outSize), linearFlag); err != nil {
		return fmt.Errorf("set set_output_deltas args: %w", err)
	}
	if _, err := b.queue.EnqueueNDRangeKernel(b.setOutputDeltasKernel, nil, []int{outSize}, nil, nil); err != nil {
		return fmt.Errorf("enqueue set_output_deltas: %w", err)
	}

	for i := len(b.layers) - 2; i >= 0; i-- {
		cur := b.layers[i]
		next := b.layers[i+1]
		if err := b.hiddenDeltaKernel.SetArgs(next.weights, next.deltas, cur.outputs, cur.deltas, int32(cur.outSize), int32(next.outSize)); err != nil {
			return fmt.Errorf("set hidden_delta args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.hiddenDeltaKernel, nil, []int{cur.outSize}, nil, nil); err != nil {
			return fmt.Errorf("enqueue hidden_delta layer %d: %w", i, err)
		}
	}

	for i, layer := range b.layers {
		inputBuf := b.scratchA
		inSize := layer.inSize
		if i > 0 {
			inputBuf = b.layers[i-1].outputs
			inSize = b.layers[i-1].outSize
		}

		scale := float32(eloWeight)
		weightCount := layer.inSize * layer.outSize
		if err := b.accWeightGradsKernel.SetArgs(layer.deltas, inputBuf, layer.gradWBuf, int32(inSize), int32(layer.outSize), scale); err != nil {
			return fmt.Errorf("set accumulate_weight_grads args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.accWeightGradsKernel, nil, []int{weightCount}, nil, nil); err != nil {
			return fmt.Errorf("enqueue accumulate_weight_grads layer %d: %w", i, err)
		}

		if err := b.accBiasGradsKernel.SetArgs(layer.deltas, layer.gradBBuf, int32(layer.outSize), scale); err != nil {
			return fmt.Errorf("set accumulate_bias_grads args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.accBiasGradsKernel, nil, []int{layer.outSize}, nil, nil); err != nil {
			return fmt.Errorf("enqueue accumulate_bias_grads layer %d: %w", i, err)
		}
	}

	return nil
}

func (b *openclMLPBackend) backpropGivenDeltasBatch(mlp *MLP, inputsBatch [][]float64, outputDeltasBatch [][]float64, sampleWeights []float64) error {
	batchSize := len(inputsBatch)
	if batchSize == 0 {
		return nil
	}
	if len(outputDeltasBatch) != batchSize || len(sampleWeights) != batchSize {
		return fmt.Errorf("batch sizes mismatch: inputs=%d outputDeltas=%d sampleWeights=%d", batchSize, len(outputDeltasBatch), len(sampleWeights))
	}
	if batchSize > b.maxBatchSize {
		return fmt.Errorf("batch size %d exceeds OPENCL_MAX_BATCH %d", batchSize, b.maxBatchSize)
	}

	inSize := len(inputsBatch[0])
	if inSize > b.workBufSize {
		return fmt.Errorf("input size %d exceeds OpenCL work buffer %d", inSize, b.workBufSize)
	}
	for i := 1; i < batchSize; i++ {
		if len(inputsBatch[i]) != inSize {
			return fmt.Errorf("input batch element %d has size %d, expected %d", i, len(inputsBatch[i]), inSize)
		}
	}

	outSize := b.layers[len(b.layers)-1].outSize
	for i := 0; i < batchSize; i++ {
		if len(outputDeltasBatch[i]) != outSize {
			return fmt.Errorf("output delta batch element %d has size %d, expected %d", i, len(outputDeltasBatch[i]), outSize)
		}
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	inFlat := make([]float32, batchSize*inSize)
	for s := 0; s < batchSize; s++ {
		base := s * inSize
		for i := 0; i < inSize; i++ {
			inFlat[base+i] = float32(inputsBatch[s][i])
		}
	}
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.batchInputBuf, false, 0, inFlat, nil); err != nil {
		return fmt.Errorf("write batch inputs: %w", err)
	}

	currentInputBuf := b.batchInputBuf
	currentInputSize := inSize
	for i, layer := range b.layers {
		activation := int32(0)
		if i == len(b.layers)-1 {
			if mlp.LinearOutput {
				activation = 2
			} else {
				activation = 1
			}
		}
		if err := b.denseForwardBatchKernel.SetArgs(
			layer.weights, layer.biases, currentInputBuf, layer.batchOut,
			int32(currentInputSize), int32(layer.outSize), int32(batchSize), activation,
		); err != nil {
			return fmt.Errorf("set dense_forward_batch args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.denseForwardBatchKernel, nil, []int{layer.outSize * batchSize}, nil, nil); err != nil {
			return fmt.Errorf("enqueue dense_forward_batch layer %d: %w", i, err)
		}
		currentInputBuf = layer.batchOut
		currentInputSize = layer.outSize
	}

	outDeltaFlat := make([]float32, batchSize*outSize)
	scaleFlat := make([]float32, batchSize)
	for s := 0; s < batchSize; s++ {
		base := s * outSize
		for i := 0; i < outSize; i++ {
			outDeltaFlat[base+i] = float32(outputDeltasBatch[s][i])
		}
		scaleFlat[s] = float32(sampleWeights[s])
	}
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.batchTargetsBuf, false, 0, outDeltaFlat, nil); err != nil {
		return fmt.Errorf("write batch output deltas: %w", err)
	}
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.batchScaleBuf, false, 0, scaleFlat, nil); err != nil {
		return fmt.Errorf("write batch sample weights: %w", err)
	}

	linearFlag := int32(0)
	if mlp.LinearOutput {
		linearFlag = 1
	}
	last := b.layers[len(b.layers)-1]
	if err := b.setOutputDeltasBatchKernel.SetArgs(last.batchOut, b.batchTargetsBuf, b.batchScaleBuf, last.batchDel, int32(outSize), int32(batchSize), linearFlag); err != nil {
		return fmt.Errorf("set set_output_deltas_batch args: %w", err)
	}
	if _, err := b.queue.EnqueueNDRangeKernel(b.setOutputDeltasBatchKernel, nil, []int{outSize * batchSize}, nil, nil); err != nil {
		return fmt.Errorf("enqueue set_output_deltas_batch: %w", err)
	}

	for i := len(b.layers) - 2; i >= 0; i-- {
		cur := b.layers[i]
		next := b.layers[i+1]
		if err := b.hiddenDeltaBatchKernel.SetArgs(next.weights, next.batchDel, cur.batchOut, cur.batchDel, int32(cur.outSize), int32(next.outSize), int32(batchSize)); err != nil {
			return fmt.Errorf("set hidden_delta_batch args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.hiddenDeltaBatchKernel, nil, []int{cur.outSize * batchSize}, nil, nil); err != nil {
			return fmt.Errorf("enqueue hidden_delta_batch layer %d: %w", i, err)
		}
	}

	for i, layer := range b.layers {
		inputBuf := b.batchInputBuf
		inputSize := inSize
		if i > 0 {
			inputBuf = b.layers[i-1].batchOut
			inputSize = b.layers[i-1].outSize
		}

		weightCount := layer.inSize * layer.outSize
		if err := b.accWeightGradsBatchKernel.SetArgs(layer.batchDel, inputBuf, layer.gradWBuf, int32(inputSize), int32(layer.outSize), int32(batchSize)); err != nil {
			return fmt.Errorf("set accumulate_weight_grads_batch args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.accWeightGradsBatchKernel, nil, []int{weightCount}, nil, nil); err != nil {
			return fmt.Errorf("enqueue accumulate_weight_grads_batch layer %d: %w", i, err)
		}

		if err := b.accBiasGradsBatchKernel.SetArgs(layer.batchDel, layer.gradBBuf, int32(layer.outSize), int32(batchSize)); err != nil {
			return fmt.Errorf("set accumulate_bias_grads_batch args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.accBiasGradsBatchKernel, nil, []int{layer.outSize}, nil, nil); err != nil {
			return fmt.Errorf("enqueue accumulate_bias_grads_batch layer %d: %w", i, err)
		}
	}

	return nil
}

func (b *openclMLPBackend) backpropAttentionFromInputGradsBatch(
	mlp *MLP,
	rawSlotsBatch [][]float64,
	attentionWeightsBatch [][]float64,
	mainInputGradsFlat []float64,
	sampleWeights []float64,
	featuresPerSlot int,
) error {
	batchSize := len(rawSlotsBatch)
	if batchSize == 0 {
		return nil
	}
	if len(attentionWeightsBatch) != batchSize || len(sampleWeights) != batchSize {
		return fmt.Errorf("batch sizes mismatch: raw=%d attn=%d weights=%d", batchSize, len(attentionWeightsBatch), len(sampleWeights))
	}
	if batchSize > b.maxBatchSize {
		return fmt.Errorf("batch size %d exceeds OPENCL_MAX_BATCH %d", batchSize, b.maxBatchSize)
	}
	if len(b.layers) == 0 {
		return fmt.Errorf("MLP has no layers")
	}
	if featuresPerSlot <= 0 {
		return fmt.Errorf("featuresPerSlot must be > 0")
	}

	inSize := len(rawSlotsBatch[0])
	if inSize > b.workBufSize {
		return fmt.Errorf("input size %d exceeds OpenCL work buffer %d", inSize, b.workBufSize)
	}
	for i := 1; i < batchSize; i++ {
		if len(rawSlotsBatch[i]) != inSize {
			return fmt.Errorf("raw slot batch element %d has size %d, expected %d", i, len(rawSlotsBatch[i]), inSize)
		}
	}

	outSize := b.layers[len(b.layers)-1].outSize
	if outSize <= 0 {
		return fmt.Errorf("invalid attention output size %d", outSize)
	}
	if outSize > 16 {
		return fmt.Errorf("attention output size %d exceeds kernel local limit 16", outSize)
	}
	for i := 0; i < batchSize; i++ {
		if len(attentionWeightsBatch[i]) != outSize {
			return fmt.Errorf("attention weight batch element %d has size %d, expected %d", i, len(attentionWeightsBatch[i]), outSize)
		}
	}

	expectedGradCount := batchSize * outSize * featuresPerSlot
	if len(mainInputGradsFlat) != expectedGradCount {
		return fmt.Errorf("main input gradient flat size mismatch: got %d expected %d", len(mainInputGradsFlat), expectedGradCount)
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	rawFlat := make([]float32, batchSize*inSize)
	for s := 0; s < batchSize; s++ {
		base := s * inSize
		for i := 0; i < inSize; i++ {
			rawFlat[base+i] = float32(rawSlotsBatch[s][i])
		}
	}
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.batchInputBuf, false, 0, rawFlat, nil); err != nil {
		return fmt.Errorf("write raw slot batch inputs: %w", err)
	}

	currentInputBuf := b.batchInputBuf
	currentInputSize := inSize
	for i, layer := range b.layers {
		activation := int32(0)
		if i == len(b.layers)-1 {
			if mlp.LinearOutput {
				activation = 2
			} else {
				activation = 1
			}
		}
		if err := b.denseForwardBatchKernel.SetArgs(
			layer.weights, layer.biases, currentInputBuf, layer.batchOut,
			int32(currentInputSize), int32(layer.outSize), int32(batchSize), activation,
		); err != nil {
			return fmt.Errorf("set dense_forward_batch args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.denseForwardBatchKernel, nil, []int{layer.outSize * batchSize}, nil, nil); err != nil {
			return fmt.Errorf("enqueue dense_forward_batch layer %d: %w", i, err)
		}
		currentInputBuf = layer.batchOut
		currentInputSize = layer.outSize
	}

	mainGrad32 := float64To32(mainInputGradsFlat)
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.inputGradBatchBuf, false, 0, mainGrad32, nil); err != nil {
		return fmt.Errorf("write main input gradients batch: %w", err)
	}

	attnWeightsFlat := make([]float32, batchSize*outSize)
	for s := 0; s < batchSize; s++ {
		base := s * outSize
		for i := 0; i < outSize; i++ {
			attnWeightsFlat[base+i] = float32(attentionWeightsBatch[s][i])
		}
	}
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.batchTargetsBuf, false, 0, attnWeightsFlat, nil); err != nil {
		return fmt.Errorf("write attention weights batch: %w", err)
	}

	scaleFlat := make([]float32, batchSize)
	for s := 0; s < batchSize; s++ {
		scaleFlat[s] = float32(sampleWeights[s])
	}
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.batchScaleBuf, false, 0, scaleFlat, nil); err != nil {
		return fmt.Errorf("write batch sample weights: %w", err)
	}

	last := b.layers[len(b.layers)-1]
	if err := b.attentionDeltaInputKernel.SetArgs(
		b.inputGradBatchBuf,
		b.batchInputBuf,
		b.batchTargetsBuf,
		b.batchScaleBuf,
		last.batchDel,
		int32(featuresPerSlot),
		int32(outSize),
		int32(batchSize),
	); err != nil {
		return fmt.Errorf("set attention_output_deltas_from_input_grads args: %w", err)
	}
	if _, err := b.queue.EnqueueNDRangeKernel(b.attentionDeltaInputKernel, nil, []int{batchSize}, nil, nil); err != nil {
		return fmt.Errorf("enqueue attention_output_deltas_from_input_grads: %w", err)
	}

	for i := len(b.layers) - 2; i >= 0; i-- {
		cur := b.layers[i]
		next := b.layers[i+1]
		if err := b.hiddenDeltaBatchKernel.SetArgs(next.weights, next.batchDel, cur.batchOut, cur.batchDel, int32(cur.outSize), int32(next.outSize), int32(batchSize)); err != nil {
			return fmt.Errorf("set hidden_delta_batch args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.hiddenDeltaBatchKernel, nil, []int{cur.outSize * batchSize}, nil, nil); err != nil {
			return fmt.Errorf("enqueue hidden_delta_batch layer %d: %w", i, err)
		}
	}

	for i, layer := range b.layers {
		inputBuf := b.batchInputBuf
		inputSize := inSize
		if i > 0 {
			inputBuf = b.layers[i-1].batchOut
			inputSize = b.layers[i-1].outSize
		}

		weightCount := layer.inSize * layer.outSize
		if err := b.accWeightGradsBatchKernel.SetArgs(layer.batchDel, inputBuf, layer.gradWBuf, int32(inputSize), int32(layer.outSize), int32(batchSize)); err != nil {
			return fmt.Errorf("set accumulate_weight_grads_batch args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.accWeightGradsBatchKernel, nil, []int{weightCount}, nil, nil); err != nil {
			return fmt.Errorf("enqueue accumulate_weight_grads_batch layer %d: %w", i, err)
		}

		if err := b.accBiasGradsBatchKernel.SetArgs(layer.batchDel, layer.gradBBuf, int32(layer.outSize), int32(batchSize)); err != nil {
			return fmt.Errorf("set accumulate_bias_grads_batch args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.accBiasGradsBatchKernel, nil, []int{layer.outSize}, nil, nil); err != nil {
			return fmt.Errorf("enqueue accumulate_bias_grads_batch layer %d: %w", i, err)
		}
	}

	return nil
}

func (b *openclMLPBackend) applyAdamGradients(mlp *MLP, _ *WorkerCache, batchSize float64, lr, weightDecay, beta1, beta2, epsilon float64, beta1CorrInv, beta2CorrInv float64) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if batchSize <= 0 {
		return fmt.Errorf("batch size must be > 0")
	}
	invBatch := float32(1.0 / batchSize)

	for i, layer := range b.layers {
		weightCount := layer.inSize * layer.outSize
		if err := b.adamUpdateKernel.SetArgs(
			layer.weights,
			layer.weightM,
			layer.weightV,
			layer.gradWBuf,
			int32(weightCount),
			float32(beta1),
			float32(beta2),
			float32(beta1CorrInv),
			float32(beta2CorrInv),
			float32(lr),
			float32(weightDecay),
			float32(epsilon),
			invBatch,
		); err != nil {
			return fmt.Errorf("set Adam weights args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.adamUpdateKernel, nil, []int{weightCount}, nil, nil); err != nil {
			return fmt.Errorf("enqueue Adam weights layer %d: %w", i, err)
		}

		if err := b.adamUpdateKernel.SetArgs(
			layer.biases,
			layer.biasM,
			layer.biasV,
			layer.gradBBuf,
			int32(layer.outSize),
			float32(beta1),
			float32(beta2),
			float32(beta1CorrInv),
			float32(beta2CorrInv),
			float32(lr),
			float32(weightDecay),
			float32(epsilon),
			invBatch,
		); err != nil {
			return fmt.Errorf("set Adam biases args layer %d: %w", i, err)
		}
		if _, err := b.queue.EnqueueNDRangeKernel(b.adamUpdateKernel, nil, []int{layer.outSize}, nil, nil); err != nil {
			return fmt.Errorf("enqueue Adam biases layer %d: %w", i, err)
		}
	}

	return nil
}

func (b *openclMLPBackend) firstLayerInputGradSlice(mlp *MLP, inputOffset int, gradCount int) ([]float64, error) {
	if len(b.layers) == 0 {
		return nil, fmt.Errorf("MLP has no layers")
	}
	if gradCount <= 0 {
		return nil, fmt.Errorf("gradient count must be > 0")
	}
	first := b.layers[0]
	if inputOffset < 0 || inputOffset+gradCount > first.inSize {
		return nil, fmt.Errorf("gradient slice out of bounds: offset=%d count=%d inSize=%d", inputOffset, gradCount, first.inSize)
	}
	if gradCount > b.workBufSize {
		return nil, fmt.Errorf("gradient count %d exceeds work buffer size %d", gradCount, b.workBufSize)
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if err := b.firstInputGradsKernel.SetArgs(
		first.weights,
		first.deltas,
		b.inputGradBuf,
		int32(first.inSize),
		int32(first.outSize),
		int32(inputOffset),
		int32(gradCount),
	); err != nil {
		return nil, fmt.Errorf("set first_layer_input_grads args: %w", err)
	}
	if _, err := b.queue.EnqueueNDRangeKernel(b.firstInputGradsKernel, nil, []int{gradCount}, nil, nil); err != nil {
		return nil, fmt.Errorf("enqueue first_layer_input_grads: %w", err)
	}

	out32 := make([]float32, gradCount)
	if _, err := b.queue.EnqueueReadBufferFloat32(b.inputGradBuf, true, 0, out32, nil); err != nil {
		return nil, fmt.Errorf("read first-layer input gradients: %w", err)
	}
	if err := b.queue.Finish(); err != nil {
		return nil, err
	}
	return float32To64(out32), nil
}

func (b *openclMLPBackend) firstLayerInputGradSliceBatch(mlp *MLP, inputOffset int, gradCount int, batchSize int) ([]float64, error) {
	if len(b.layers) == 0 {
		return nil, fmt.Errorf("MLP has no layers")
	}
	if gradCount <= 0 || batchSize <= 0 {
		return nil, fmt.Errorf("gradient count and batch size must be > 0")
	}
	if batchSize > b.maxBatchSize {
		return nil, fmt.Errorf("batch size %d exceeds OPENCL_MAX_BATCH %d", batchSize, b.maxBatchSize)
	}
	first := b.layers[0]
	if inputOffset < 0 || inputOffset+gradCount > first.inSize {
		return nil, fmt.Errorf("gradient slice out of bounds: offset=%d count=%d inSize=%d", inputOffset, gradCount, first.inSize)
	}
	if gradCount > b.workBufSize {
		return nil, fmt.Errorf("gradient count %d exceeds work buffer size %d", gradCount, b.workBufSize)
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if err := b.firstInputGradsBatchKernel.SetArgs(
		first.weights,
		first.batchDel,
		b.inputGradBatchBuf,
		int32(first.inSize),
		int32(first.outSize),
		int32(inputOffset),
		int32(gradCount),
		int32(batchSize),
	); err != nil {
		return nil, fmt.Errorf("set first_layer_input_grads_batch args: %w", err)
	}
	if _, err := b.queue.EnqueueNDRangeKernel(b.firstInputGradsBatchKernel, nil, []int{batchSize * gradCount}, nil, nil); err != nil {
		return nil, fmt.Errorf("enqueue first_layer_input_grads_batch: %w", err)
	}

	out32 := make([]float32, batchSize*gradCount)
	if _, err := b.queue.EnqueueReadBufferFloat32(b.inputGradBatchBuf, true, 0, out32, nil); err != nil {
		return nil, fmt.Errorf("read first-layer input gradients batch: %w", err)
	}
	return float32To64(out32), nil
}

func (b *openclMLPBackend) attentionOutputDeltasFromFirstLayerBatch(
	_ *MLP,
	inputOffset int,
	rawSlotsBatch [][]float64,
	attentionWeightsBatch [][]float64,
	sampleWeights []float64,
	featuresPerSlot int,
	slotCount int,
) ([]float64, error) {
	batchSize := len(rawSlotsBatch)
	if batchSize == 0 {
		return nil, nil
	}
	if len(attentionWeightsBatch) != batchSize || len(sampleWeights) != batchSize {
		return nil, fmt.Errorf("batch sizes mismatch: raw=%d attn=%d weights=%d", batchSize, len(attentionWeightsBatch), len(sampleWeights))
	}
	if batchSize > b.maxBatchSize {
		return nil, fmt.Errorf("batch size %d exceeds OPENCL_MAX_BATCH %d", batchSize, b.maxBatchSize)
	}
	if featuresPerSlot <= 0 || slotCount <= 0 {
		return nil, fmt.Errorf("featuresPerSlot and slotCount must be > 0")
	}
	if slotCount > 16 {
		return nil, fmt.Errorf("slotCount %d exceeds kernel local limit 16", slotCount)
	}
	gradCount := featuresPerSlot * slotCount

	if len(b.layers) == 0 {
		return nil, fmt.Errorf("MLP has no layers")
	}
	first := b.layers[0]
	if inputOffset < 0 || inputOffset+gradCount > first.inSize {
		return nil, fmt.Errorf("gradient slice out of bounds: offset=%d count=%d inSize=%d", inputOffset, gradCount, first.inSize)
	}
	if gradCount > b.workBufSize {
		return nil, fmt.Errorf("gradient count %d exceeds work buffer size %d", gradCount, b.workBufSize)
	}

	rawSize := len(rawSlotsBatch[0])
	if rawSize != gradCount {
		return nil, fmt.Errorf("raw slot size %d does not match expected %d", rawSize, gradCount)
	}
	for i := 1; i < batchSize; i++ {
		if len(rawSlotsBatch[i]) != rawSize {
			return nil, fmt.Errorf("raw slot batch element %d has size %d, expected %d", i, len(rawSlotsBatch[i]), rawSize)
		}
	}
	for i := 0; i < batchSize; i++ {
		if len(attentionWeightsBatch[i]) != slotCount {
			return nil, fmt.Errorf("attention weight batch element %d has size %d, expected %d", i, len(attentionWeightsBatch[i]), slotCount)
		}
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if err := b.firstInputGradsBatchKernel.SetArgs(
		first.weights,
		first.batchDel,
		b.inputGradBatchBuf,
		int32(first.inSize),
		int32(first.outSize),
		int32(inputOffset),
		int32(gradCount),
		int32(batchSize),
	); err != nil {
		return nil, fmt.Errorf("set first_layer_input_grads_batch args: %w", err)
	}
	if _, err := b.queue.EnqueueNDRangeKernel(b.firstInputGradsBatchKernel, nil, []int{batchSize * gradCount}, nil, nil); err != nil {
		return nil, fmt.Errorf("enqueue first_layer_input_grads_batch: %w", err)
	}

	rawFlat := make([]float32, batchSize*rawSize)
	for s := 0; s < batchSize; s++ {
		base := s * rawSize
		for i := 0; i < rawSize; i++ {
			rawFlat[base+i] = float32(rawSlotsBatch[s][i])
		}
	}
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.batchInputBuf, false, 0, rawFlat, nil); err != nil {
		return nil, fmt.Errorf("write raw slots batch: %w", err)
	}

	attnFlat := make([]float32, batchSize*slotCount)
	for s := 0; s < batchSize; s++ {
		base := s * slotCount
		for i := 0; i < slotCount; i++ {
			attnFlat[base+i] = float32(attentionWeightsBatch[s][i])
		}
	}
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.batchTargetsBuf, false, 0, attnFlat, nil); err != nil {
		return nil, fmt.Errorf("write attention weights batch: %w", err)
	}

	scaleFlat := make([]float32, batchSize)
	for s := 0; s < batchSize; s++ {
		scaleFlat[s] = float32(sampleWeights[s])
	}
	if _, err := b.queue.EnqueueWriteBufferFloat32(b.batchScaleBuf, false, 0, scaleFlat, nil); err != nil {
		return nil, fmt.Errorf("write sample weights batch: %w", err)
	}

	if err := b.attentionDeltaInputKernel.SetArgs(
		b.inputGradBatchBuf,
		b.batchInputBuf,
		b.batchTargetsBuf,
		b.batchScaleBuf,
		b.attentionDeltaBuf,
		int32(featuresPerSlot),
		int32(slotCount),
		int32(batchSize),
	); err != nil {
		return nil, fmt.Errorf("set attention_output_deltas_from_input_grads args: %w", err)
	}
	if _, err := b.queue.EnqueueNDRangeKernel(b.attentionDeltaInputKernel, nil, []int{batchSize}, nil, nil); err != nil {
		return nil, fmt.Errorf("enqueue attention_output_deltas_from_input_grads: %w", err)
	}

	out32 := make([]float32, batchSize*slotCount)
	if _, err := b.queue.EnqueueReadBufferFloat32(b.attentionDeltaBuf, true, 0, out32, nil); err != nil {
		return nil, fmt.Errorf("read attention output deltas: %w", err)
	}
	return float32To64(out32), nil
}

func pickBestGPU() (*cl.Device, error) {
	platforms, err := cl.GetPlatforms()
	if err != nil {
		return nil, fmt.Errorf("list OpenCL platforms: %w", err)
	}

	var allGPUs []*cl.Device
	for _, p := range platforms {
		devs, devErr := p.GetDevices(cl.DeviceTypeGPU)
		if devErr != nil {
			continue
		}
		allGPUs = append(allGPUs, devs...)
	}
	if len(allGPUs) == 0 {
		return nil, fmt.Errorf("no OpenCL GPU devices found")
	}
	candidates := allGPUs

	// Optional explicit selection for debugging/tuning:
	// OPENCL_DEVICE_INDEX=N or OPENCL_DEVICE_SUBSTR="radeon|arc|..."
	if idxStr := strings.TrimSpace(os.Getenv("OPENCL_DEVICE_INDEX")); idxStr != "" {
		if idx, convErr := strconv.Atoi(idxStr); convErr == nil && idx >= 0 && idx < len(candidates) {
			return candidates[idx], nil
		}
	}
	if needle := strings.ToLower(strings.TrimSpace(os.Getenv("OPENCL_DEVICE_SUBSTR"))); needle != "" {
		for _, d := range candidates {
			name := strings.ToLower(d.Name())
			vendor := strings.ToLower(d.Vendor())
			if strings.Contains(name, needle) || strings.Contains(vendor, needle) {
				return d, nil
			}
		}
	}

	best := candidates[0]
	bestScore := deviceScore(best)
	for i := 1; i < len(candidates); i++ {
		score := deviceScore(candidates[i])
		if score > bestScore {
			best = candidates[i]
			bestScore = score
		}
	}
	return best, nil
}

func deviceScore(d *cl.Device) int64 {
	score := int64(d.MaxComputeUnits()) * int64(d.MaxClockFrequency())
	score += d.GlobalMemSize() / (1024 * 1024)

	// Prefer discrete GPUs for heavier training throughput.
	if !d.HostUnifiedMemory() {
		score += 1_000_000
	}

	return score
}

func float64To32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i := range in {
		out[i] = float32(in[i])
	}
	return out
}

func float32To64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i := range in {
		out[i] = float64(in[i])
	}
	return out
}

func flatten2DF64(in [][]float64) []float32 {
	if len(in) == 0 {
		return nil
	}
	cols := len(in[0])
	out := make([]float32, len(in)*cols)
	o := 0
	for i := range in {
		for j := 0; j < cols; j++ {
			out[o] = float32(in[i][j])
			o++
		}
	}
	return out
}

func unflattenTo2DF64(dst [][]float64, flat []float32) {
	if len(dst) == 0 {
		return
	}
	cols := len(dst[0])
	o := 0
	for i := range dst {
		for j := 0; j < cols; j++ {
			dst[i][j] = float64(flat[o])
			o++
		}
	}
}

func adamBiasCorrectionInv(beta float64, step int64) float64 {
	return 1.0 / (1.0 - math.Pow(beta, float64(step)))
}
