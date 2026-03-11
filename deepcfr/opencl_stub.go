//go:build !cgo || !opencl

package deepcfr

import "fmt"

func newOpenCLTrainer(_ *Model, _ TrainingHyperParams, _ int, _ string, _ string) (exampleTrainer, error) {
	return nil, fmt.Errorf("opencl training backend requires cgo")
}
