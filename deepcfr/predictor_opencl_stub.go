//go:build !cgo || !opencl

package deepcfr

import "fmt"

func newOpenCLBatchPredictBackend(_ *Model, _ string, _ string, _ int) (batchPredictBackend, error) {
	return nil, fmt.Errorf("opencl target predictor requires cgo")
}
