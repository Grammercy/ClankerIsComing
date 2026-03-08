//go:build !opencl && !rocm

package evaluator

import "fmt"

type openclLayerBuffers struct{}
type openclMLPBackend struct{}

func newOpenCLMLPBackend() (*openclMLPBackend, error) {
	return nil, fmt.Errorf("OpenCL/ROCm backend is required; rebuild with '-tags opencl' or '-tags rocm' and install OpenCL SDK/runtime headers")
}

func (b *openclMLPBackend) InitMLP(_ *MLP) error {
	return fmt.Errorf("OpenCL backend unavailable")
}

func (b *openclMLPBackend) Release(_ *MLP) {}

func (b *openclMLPBackend) syncParamsFromHost(_ *MLP) error {
	return fmt.Errorf("OpenCL backend unavailable in non-opencl build")
}

func (b *openclMLPBackend) syncParamsToHost(_ *MLP) error {
	return fmt.Errorf("OpenCL backend unavailable in non-opencl build")
}

func (b *openclMLPBackend) forward(_ *MLP, _ []float64, _ *InferenceCache) ([]float64, error) {
	return nil, fmt.Errorf("OpenCL backend unavailable in non-opencl build")
}

func (b *openclMLPBackend) forwardBatch(_ *MLP, _ [][]float64) ([][]float64, error) {
	return nil, fmt.Errorf("OpenCL backend unavailable in non-opencl build")
}

func (b *openclMLPBackend) calculateBCELocalGradients(_ *MLP, _ []float64, _ []float64, _ float64, _ *WorkerCache) (float64, error) {
	return 0, fmt.Errorf("OpenCL backend unavailable in non-opencl build")
}

func (b *openclMLPBackend) calculateBCELocalGradientsBatch(_ *MLP, _ [][]float64, _ [][]float64, _ []float64) (float64, [][]float64, error) {
	return 0, nil, fmt.Errorf("OpenCL backend unavailable in non-opencl build")
}

func (b *openclMLPBackend) backpropGivenDeltas(_ *MLP, _ []float64, _ []float64, _ float64, _ *WorkerCache) error {
	return fmt.Errorf("OpenCL backend unavailable in non-opencl build")
}

func (b *openclMLPBackend) backpropGivenDeltasBatch(_ *MLP, _ [][]float64, _ [][]float64, _ []float64) error {
	return fmt.Errorf("OpenCL backend unavailable in non-opencl build")
}

func (b *openclMLPBackend) backpropAttentionFromInputGradsBatch(_ *MLP, _ [][]float64, _ [][]float64, _ []float64, _ []float64, _ int) error {
	return fmt.Errorf("OpenCL backend unavailable in non-opencl build")
}

func (b *openclMLPBackend) applyAdamGradients(_ *MLP, _ *WorkerCache, _ float64, _, _, _, _, _ float64, _, _ float64) error {
	return fmt.Errorf("OpenCL backend unavailable in non-opencl build")
}

func (b *openclMLPBackend) clearGradients(_ *MLP) error {
	return fmt.Errorf("OpenCL backend unavailable in non-opencl build")
}

func (b *openclMLPBackend) firstLayerInputGradSlice(_ *MLP, _, _ int) ([]float64, error) {
	return nil, fmt.Errorf("OpenCL backend unavailable in non-opencl build")
}

func (b *openclMLPBackend) firstLayerInputGradSliceBatch(_ *MLP, _, _, _ int) ([]float64, error) {
	return nil, fmt.Errorf("OpenCL backend unavailable in non-opencl build")
}

func (b *openclMLPBackend) attentionOutputDeltasFromFirstLayerBatch(_ *MLP, _ int, _ [][]float64, _ [][]float64, _ []float64, _, _ int) ([]float64, error) {
	return nil, fmt.Errorf("OpenCL backend unavailable in non-opencl build")
}

func (b *openclMLPBackend) maxBatchSizeLimit() int {
	return 1
}

func adamBiasCorrectionInv(beta float64, step int64) float64 {
	// Keep this helper in non-opencl builds so nn.go compiles.
	pow := 1.0
	for i := int64(0); i < step; i++ {
		pow *= beta
	}
	return 1.0 / (1.0 - pow)
}
