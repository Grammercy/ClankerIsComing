package deepcfr

import (
	"fmt"
	"strings"
)

type exampleTrainer interface {
	Train(example TrainingExample) ([]TrainingMetrics, error)
	Flush() ([]TrainingMetrics, error)
	Close() error
	Name() string
}

type cpuTrainer struct {
	model *Model
	hp    TrainingHyperParams
}

func newExampleTrainer(model *Model, hp TrainingHyperParams, cfg TrainConfig) (exampleTrainer, error) {
	accelerator := strings.ToLower(strings.TrimSpace(cfg.Accelerator))
	if accelerator == "" {
		accelerator = "auto"
	}
	batchSize := cfg.BatchSize
	if batchSize <= 0 {
		if accelerator == "cpu" {
			batchSize = 1
		} else {
			batchSize = 128
		}
	}

	switch accelerator {
	case "auto":
		trainer, err := newOpenCLTrainer(model, hp, batchSize, cfg.OpenCLPlatform, cfg.OpenCLDevice)
		if err == nil {
			return trainer, nil
		}
		return &cpuTrainer{model: model, hp: hp}, nil
	case "opencl":
		return newOpenCLTrainer(model, hp, batchSize, cfg.OpenCLPlatform, cfg.OpenCLDevice)
	case "cpu":
		return &cpuTrainer{model: model, hp: hp}, nil
	default:
		return nil, fmt.Errorf("unknown training accelerator %q", cfg.Accelerator)
	}
}

func (t *cpuTrainer) Train(example TrainingExample) ([]TrainingMetrics, error) {
	return []TrainingMetrics{t.model.TrainExample(example, t.hp)}, nil
}

func (t *cpuTrainer) Flush() ([]TrainingMetrics, error) {
	return nil, nil
}

func (t *cpuTrainer) Close() error {
	return nil
}

func (t *cpuTrainer) Name() string {
	return "cpu"
}
