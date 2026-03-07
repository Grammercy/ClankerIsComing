package evaluator

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// Layer represents a single layer in the MLP
type Layer struct {
	Mu      sync.Mutex `json:"-"` // Layer-specific lock to prevent global bottleneck
	Weights [][]float64
	Biases  []float64
	Outputs []float64
	Deltas  []float64
	// Adam optimizer moment estimates
	WeightM [][]float64 `json:"-"`
	WeightV [][]float64 `json:"-"`
	BiasM   []float64   `json:"-"`
	BiasV   []float64   `json:"-"`
}

// MLP represents the Multi-Layer Perceptron neural network
type MLP struct {
	Layers       []*Layer
	LinearOutput bool
	AdamStep     int64

	gpu *openclMLPBackend
}

func (mlp *MLP) HasLayerSizes(sizes []int) bool {
	if mlp == nil || len(sizes) < 2 || len(mlp.Layers) != len(sizes)-1 {
		return false
	}
	for i, layer := range mlp.Layers {
		if len(layer.Weights) != sizes[i+1] || len(layer.Biases) != sizes[i+1] {
			return false
		}
		for _, row := range layer.Weights {
			if len(row) != sizes[i] {
				return false
			}
		}
	}
	return true
}

// NewMLP initializes an MLP with the given layer sizes.
func NewMLP(sizes []int) *MLP {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	mlp := &MLP{Layers: make([]*Layer, len(sizes)-1)}

	for i := 0; i < len(sizes)-1; i++ {
		inputSize := sizes[i]
		outputSize := sizes[i+1]

		layer := &Layer{
			Weights: make([][]float64, outputSize),
			Biases:  make([]float64, outputSize),
			Outputs: make([]float64, outputSize),
			Deltas:  make([]float64, outputSize),
			WeightM: make([][]float64, outputSize),
			WeightV: make([][]float64, outputSize),
			BiasM:   make([]float64, outputSize),
			BiasV:   make([]float64, outputSize),
		}

		variance := math.Sqrt(2.0 / float64(inputSize))
		for j := 0; j < outputSize; j++ {
			layer.Weights[j] = make([]float64, inputSize)
			layer.WeightM[j] = make([]float64, inputSize)
			layer.WeightV[j] = make([]float64, inputSize)
			for k := 0; k < inputSize; k++ {
				layer.Weights[j][k] = rng.NormFloat64() * variance
			}
		}
		mlp.Layers[i] = layer
	}

	backend, err := newOpenCLMLPBackend(mlp)
	if err != nil {
		fmt.Printf("Warning: failed to initialize OpenCL MLP backend (falling back to CPU): %v\n", err)
	} else {
		mlp.gpu = backend
	}

	return mlp
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func bceLoss(target, output float64) float64 {
	const epsilon = 1e-15
	if output < epsilon {
		output = epsilon
	}
	if output > 1.0-epsilon {
		output = 1.0 - epsilon
	}
	loss := -(target*math.Log(output) + (1.0-target)*math.Log(1.0-output))
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return 0
	}
	return loss
}

// Caches
type InferenceCache struct {
	Outputs [][]float64
}

func NewInferenceCache(mlp *MLP) *InferenceCache {
	if mlp == nil {
		return nil
	}
	c := &InferenceCache{Outputs: make([][]float64, len(mlp.Layers))}
	for i, layer := range mlp.Layers {
		c.Outputs[i] = make([]float64, len(layer.Outputs))
	}
	return c
}

type WorkerCache struct {
	Outputs         [][]float64
	Deltas          [][]float64
	WeightGradients [][][]float64
	BiasGradients   [][]float64
}


func (mlp *MLP) ensureGPU() {
	// Re-initialization attempt if nil, but don't panic
	if mlp.gpu == nil {
		backend, err := newOpenCLMLPBackend(mlp)
		if err != nil {
			// Log once and don't try again if it's a permanent failure (stub)
			return
		}
		mlp.gpu = backend
	}
}

func (mlp *MLP) OpenCLMaxBatchSize() int {
	mlp.ensureGPU()
	return mlp.gpu.maxBatchSizeLimit()
}

// Forward runs entirely on OpenCL.
func (mlp *MLP) Forward(inputs []float64, cache *InferenceCache) []float64 {
	mlp.ensureGPU()
	if mlp.gpu == nil {
		lastLayer := mlp.Layers[len(mlp.Layers)-1]
		return make([]float64, len(lastLayer.Biases))
	}
	out, err := mlp.gpu.forward(mlp, inputs, cache)
	if err != nil {
		panic(fmt.Sprintf("OpenCL forward failed: %v", err))
	}
	return out
}

func (mlp *MLP) ForwardBatch(inputsBatch [][]float64) [][]float64 {
	mlp.ensureGPU()
	if mlp.gpu == nil {
		lastLayer := mlp.Layers[len(mlp.Layers)-1]
		out := make([][]float64, len(inputsBatch))
		for i := range out {
			out[i] = make([]float64, len(lastLayer.Biases))
		}
		return out
	}
	out, err := mlp.gpu.forwardBatch(mlp, inputsBatch)
	if err != nil {
		panic(fmt.Sprintf("OpenCL batch forward failed: %v", err))
	}
	return out
}

// CalculateBCELocalGradients runs forward/backprop gradient accumulation on OpenCL.
func (mlp *MLP) CalculateBCELocalGradients(inputs []float64, targets []float64, eloWeight float64, cache *WorkerCache) float64 {
	mlp.ensureGPU()
	loss, err := mlp.gpu.calculateBCELocalGradients(mlp, inputs, targets, eloWeight, cache)
	if err != nil {
		panic(fmt.Sprintf("OpenCL BCE backprop failed: %v", err))
	}
	return loss
}

func (mlp *MLP) CalculateBCELocalGradientsBatch(inputsBatch [][]float64, targetsBatch [][]float64, eloWeights []float64) float64 {
	mlp.ensureGPU()
	loss, err := mlp.gpu.calculateBCELocalGradientsBatch(mlp, inputsBatch, targetsBatch, eloWeights)
	if err != nil {
		panic(fmt.Sprintf("OpenCL BCE batch backprop failed: %v", err))
	}
	return loss
}

func (mlp *MLP) BackpropGivenDeltas(inputs []float64, outputDeltas []float64, eloWeight float64, cache *WorkerCache) {
	mlp.ensureGPU()
	if err := mlp.gpu.backpropGivenDeltas(mlp, inputs, outputDeltas, eloWeight, cache); err != nil {
		panic(fmt.Sprintf("OpenCL delta backprop failed: %v", err))
	}
}

func (mlp *MLP) BackpropGivenDeltasBatch(inputsBatch [][]float64, outputDeltasBatch [][]float64, sampleWeights []float64) {
	mlp.ensureGPU()
	if err := mlp.gpu.backpropGivenDeltasBatch(mlp, inputsBatch, outputDeltasBatch, sampleWeights); err != nil {
		panic(fmt.Sprintf("OpenCL batch delta backprop failed: %v", err))
	}
}

func (mlp *MLP) BackpropAttentionFromInputGradsBatch(
	rawSlotsBatch [][]float64,
	attentionWeightsBatch [][]float64,
	mainInputGradsFlat []float64,
	sampleWeights []float64,
	featuresPerSlot int,
) {
	mlp.ensureGPU()
	if err := mlp.gpu.backpropAttentionFromInputGradsBatch(mlp, rawSlotsBatch, attentionWeightsBatch, mainInputGradsFlat, sampleWeights, featuresPerSlot); err != nil {
		panic(fmt.Sprintf("OpenCL attention delta batch backprop failed: %v", err))
	}
}

func (mlp *MLP) ApplyAdamGradients(cache *WorkerCache, batchSize float64, lr, weightDecay, beta1, beta2, epsilon float64) {
	mlp.ensureGPU()
	step := atomic.AddInt64(&mlp.AdamStep, 1)
	beta1CorrInv := adamBiasCorrectionInv(beta1, step)
	beta2CorrInv := adamBiasCorrectionInv(beta2, step)

	for _, layer := range mlp.Layers {
		layer.Mu.Lock()
	}
	defer func() {
		for _, layer := range mlp.Layers {
			layer.Mu.Unlock()
		}
	}()

	if err := mlp.gpu.applyAdamGradients(mlp, cache, batchSize, lr, weightDecay, beta1, beta2, epsilon, beta1CorrInv, beta2CorrInv); err != nil {
		panic(fmt.Sprintf("OpenCL Adam update failed: %v", err))
	}
}

func (mlp *MLP) ClearGradients() {
	mlp.ensureGPU()
	if err := mlp.gpu.clearGradients(); err != nil {
		panic(fmt.Sprintf("OpenCL clear gradients failed: %v", err))
	}
}

func (mlp *MLP) FirstLayerInputGradSlice(inputOffset int, gradCount int) []float64 {
	mlp.ensureGPU()
	grads, err := mlp.gpu.firstLayerInputGradSlice(mlp, inputOffset, gradCount)
	if err != nil {
		panic(fmt.Sprintf("OpenCL first-layer input gradient failed: %v", err))
	}
	return grads
}

func (mlp *MLP) FirstLayerInputGradSliceBatch(inputOffset int, gradCount int, batchSize int) []float64 {
	mlp.ensureGPU()
	grads, err := mlp.gpu.firstLayerInputGradSliceBatch(mlp, inputOffset, gradCount, batchSize)
	if err != nil {
		panic(fmt.Sprintf("OpenCL first-layer input gradient batch failed: %v", err))
	}
	return grads
}

func (mlp *MLP) AttentionOutputDeltasFromFirstLayerBatch(
	inputOffset int,
	rawSlotsBatch [][]float64,
	attentionWeightsBatch [][]float64,
	sampleWeights []float64,
	featuresPerSlot int,
	slotCount int,
) []float64 {
	mlp.ensureGPU()
	deltas, err := mlp.gpu.attentionOutputDeltasFromFirstLayerBatch(mlp, inputOffset, rawSlotsBatch, attentionWeightsBatch, sampleWeights, featuresPerSlot, slotCount)
	if err != nil {
		panic(fmt.Sprintf("OpenCL attention output deltas from first layer failed: %v", err))
	}
	return deltas
}

func (mlp *MLP) SaveWeights(filename string) error {
	mlp.ensureGPU()
	if err := mlp.gpu.syncParamsToHost(mlp); err != nil {
		return fmt.Errorf("sync OpenCL params to host before save: %w", err)
	}

	for _, l := range mlp.Layers {
		l.Mu.Lock()
		defer l.Mu.Unlock()
	}
	data, err := json.Marshal(mlp.Layers)
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

func (mlp *MLP) LoadWeights(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &mlp.Layers); err != nil {
		return err
	}

	for _, layer := range mlp.Layers {
		if len(layer.Outputs) != len(layer.Weights) {
			layer.Outputs = make([]float64, len(layer.Weights))
		}
		if len(layer.Deltas) != len(layer.Weights) {
			layer.Deltas = make([]float64, len(layer.Weights))
		}
		if len(layer.WeightM) != len(layer.Weights) {
			layer.WeightM = make([][]float64, len(layer.Weights))
		}
		if len(layer.WeightV) != len(layer.Weights) {
			layer.WeightV = make([][]float64, len(layer.Weights))
		}
		if len(layer.BiasM) != len(layer.Biases) {
			layer.BiasM = make([]float64, len(layer.Biases))
		}
		if len(layer.BiasV) != len(layer.Biases) {
			layer.BiasV = make([]float64, len(layer.Biases))
		}
		for i := range layer.Weights {
			if len(layer.WeightM[i]) != len(layer.Weights[i]) {
				layer.WeightM[i] = make([]float64, len(layer.Weights[i]))
			}
			if len(layer.WeightV[i]) != len(layer.Weights[i]) {
				layer.WeightV[i] = make([]float64, len(layer.Weights[i]))
			}
		}
	}

	mlp.ensureGPU()
	if err := mlp.gpu.syncParamsFromHost(mlp); err != nil {
		return fmt.Errorf("sync loaded weights to OpenCL backend: %w", err)
	}
	return nil
}
