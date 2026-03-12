package neuralv2

import "fmt"

func newONNXBackend(path string) (backend, error) {
	return nil, fmt.Errorf("onnx backend is not available in this build (path: %s)", path)
}
