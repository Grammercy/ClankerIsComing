package deepcfr

import (
	"os"
	"strings"
	"testing"
	"time"
)

func TestTrainingProgressRenderLine(t *testing.T) {
	progress := &trainingProgress{
		start:       time.Date(2026, time.March, 10, 12, 0, 0, 0, time.UTC),
		totalEpochs: 3,
		totalFiles:  40,
	}
	progress.currentFile.Store("gen9randombattle-super-long-training-replay-name.log")
	progress.currentEpoch.Store(2)
	progress.filesSeen.Store(12)
	progress.attemptedFiles.Store(14)
	progress.parseErrors.Store(2)
	progress.emptyFiles.Store(3)
	progress.positions.Store(250)
	atomicAddFloat64(&progress.lossSum, 125.0)
	atomicAddFloat64(&progress.valueErrorSum, 50.0)
	atomicAddFloat64(&progress.policyCrossSum, 75.0)

	line := progress.renderLine(progress.start.Add(25 * time.Second))

	checks := []string{
		"epoch 2/3",
		"file 12/40",
		"attempt 14",
		"err 2",
		"empty 3",
		"pos 250",
		"loss 0.500000",
		"mae 0.200000",
		"ce 0.300000",
		"pos/s 10.0",
		"file/s 0.48",
		"elapsed 25s",
		"eta 58s",
		"current=...",
	}
	for _, check := range checks {
		if !strings.Contains(line, check) {
			t.Fatalf("expected line to contain %q, got %q", check, line)
		}
	}
}

func TestCountReplayLogs(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	for _, name := range []string{"a.log", "b.log", "notes.txt"} {
		if err := os.WriteFile(dir+"/"+name, []byte("test"), 0644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	if err := os.Mkdir(dir+"/subdir", 0755); err != nil {
		t.Fatalf("mkdir subdir: %v", err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	if got := countReplayLogs(entries); got != 2 {
		t.Fatalf("expected 2 replay logs, got %d", got)
	}
}

func TestNewExampleTrainerCPU(t *testing.T) {
	model := NewModel(1)
	trainer, err := newExampleTrainer(model, TrainingHyperParams{LearningRate: 0.001}, TrainConfig{
		Accelerator: "cpu",
	})
	if err != nil {
		t.Fatalf("newExampleTrainer cpu: %v", err)
	}
	t.Cleanup(func() {
		_ = trainer.Close()
	})
	if got := trainer.Name(); got != "cpu" {
		t.Fatalf("expected cpu trainer, got %q", got)
	}
}

func TestModelCloneCopiesWeights(t *testing.T) {
	model := NewModel(1)
	model.Priors = &Priors{SpeciesCounts: map[string]int{"pikachu": 3}}
	cloned := model.Clone()
	if cloned == model {
		t.Fatal("expected clone to allocate a new model")
	}
	cloned.W1[0] += 1
	cloned.B1[0] += 1
	if model.W1[0] == cloned.W1[0] {
		t.Fatal("expected cloned weights to be independent")
	}
	if model.B1[0] == cloned.B1[0] {
		t.Fatal("expected cloned biases to be independent")
	}
	if cloned.Priors != model.Priors {
		t.Fatal("expected priors pointer to be shared read-only")
	}
}
