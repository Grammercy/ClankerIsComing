package deepcfr

import (
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/pokemon-engine/simulator"
)

func TestAsyncPredictorMatchesDirectModel(t *testing.T) {
	model := NewModel(7)
	predictor, warning, err := newStatePredictor(model, TrainConfig{
		TargetPredictor: targetPredictorCPUBatch,
		TargetBatchSize: 32,
		TargetQueueSize: 128,
	})
	if err != nil {
		t.Fatalf("newStatePredictor: %v", err)
	}
	if warning != "" {
		t.Fatalf("unexpected warning: %s", warning)
	}
	t.Cleanup(func() {
		_ = predictor.Close()
	})

	rng := rand.New(rand.NewSource(99))
	for sample := 0; sample < 64; sample++ {
		features := make([]float64, FeatureSize)
		mask := make([]float64, simulator.MaxActions)
		for i := range features {
			features[i] = rng.Float64()*2 - 1
		}
		legal := 0
		for i := range mask {
			if rng.Float64() < 0.45 {
				mask[i] = 1
				legal++
			}
		}
		if legal == 0 {
			mask[0] = 1
		}

		wantRegret, wantPolicy, wantValue := model.Predict(features, mask)
		gotRegret, gotPolicy, gotValue := predictor.Predict(features, mask)

		for i := 0; i < simulator.MaxActions; i++ {
			if math.Abs(wantRegret[i]-gotRegret[i]) > 1e-4 {
				t.Fatalf("regret mismatch at sample %d action %d: want %.6f got %.6f", sample, i, wantRegret[i], gotRegret[i])
			}
			if math.Abs(wantPolicy[i]-gotPolicy[i]) > 1e-4 {
				t.Fatalf("policy mismatch at sample %d action %d: want %.6f got %.6f", sample, i, wantPolicy[i], gotPolicy[i])
			}
		}
		if math.Abs(wantValue-gotValue) > 1e-4 {
			t.Fatalf("value mismatch at sample %d: want %.6f got %.6f", sample, wantValue, gotValue)
		}
	}
}

func TestAsyncPredictorBatchesConcurrentRequests(t *testing.T) {
	model := NewModel(11)
	predictor, warning, err := newStatePredictor(model, TrainConfig{
		TargetPredictor: targetPredictorCPUBatch,
		TargetBatchSize: 64,
		TargetQueueSize: 2048,
	})
	if err != nil {
		t.Fatalf("newStatePredictor: %v", err)
	}
	if warning != "" {
		t.Fatalf("unexpected warning: %s", warning)
	}
	t.Cleanup(func() {
		_ = predictor.Close()
	})

	resetPredictorMetrics()

	var wg sync.WaitGroup
	start := make(chan struct{})
	for g := 0; g < 64; g++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed))
			<-start
			for i := 0; i < 8; i++ {
				features := make([]float64, FeatureSize)
				mask := make([]float64, simulator.MaxActions)
				for j := range features {
					features[j] = rng.Float64()
				}
				mask[rng.Intn(simulator.MaxActions)] = 1
				predictor.Predict(features, mask)
			}
		}(int64(g + 1))
	}
	close(start)
	wg.Wait()

	time.Sleep(5 * time.Millisecond)
	metrics := snapshotPredictorMetrics()
	if metrics.States == 0 {
		t.Fatalf("expected predictor metrics states > 0")
	}
	if metrics.Batches == 0 {
		t.Fatalf("expected predictor metrics batches > 0")
	}
}
