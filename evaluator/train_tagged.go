package evaluator

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/pokemon-engine/gamedata"
	"github.com/pokemon-engine/parser"
	"github.com/pokemon-engine/simulator"
)

type taggedDataLoadStats struct {
	filesLoaded      int
	samplesLoaded    int
	samplesSkipped   int
	filesParseFailed int
}

func (s *taggedDataLoadStats) merge(other taggedDataLoadStats) {
	s.filesLoaded += other.filesLoaded
	s.samplesLoaded += other.samplesLoaded
	s.samplesSkipped += other.samplesSkipped
	s.filesParseFailed += other.filesParseFailed
}

func bceLossClamped(target, output float64) float64 {
	const epsilon = 1e-15
	if output < epsilon {
		output = epsilon
	}
	if output > 1.0-epsilon {
		output = 1.0 - epsilon
	}
	return -(target*math.Log(output) + (1.0-target)*math.Log(1.0-output))
}

func buildInputFromPrepared(sample preparedSnapshot, attentionMLP *MLP) []float64 {
	mainInputs := make([]float64, TotalFeatures)
	copy(mainInputs, sample.prefix)

	rawAttention := attentionMLP.Forward(sample.rawSlots, nil)
	attentionWeights := make([]float64, len(rawAttention))
	maxScore := -math.MaxFloat64
	for _, score := range rawAttention {
		if score > maxScore {
			maxScore = score
		}
	}
	sumExp := 0.0
	for i, score := range rawAttention {
		attentionWeights[i] = math.Exp(score - maxScore)
		sumExp += attentionWeights[i]
	}
	if sumExp > 0 {
		for i := range attentionWeights {
			attentionWeights[i] /= sumExp
		}
	}

	idx := TotalGlobals
	for slotIdx := 0; slotIdx < len(attentionWeights); slotIdx++ {
		w := attentionWeights[slotIdx]
		base := slotIdx * FeaturesPerSlot
		for j := 0; j < FeaturesPerSlot; j++ {
			mainInputs[idx] = sample.rawSlots[base+j] * w
			idx++
		}
	}

	return mainInputs
}

func evaluateReplaySetBCE(testDir string, mlp *MLP, attentionMLP *MLP) (float64, int, error) {
	entries, err := os.ReadDir(testDir)
	if err != nil {
		return 0, 0, err
	}

	totalLoss := 0.0
	totalTargets := 0

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".log") {
			continue
		}
		filePath := fmt.Sprintf("%s/%s", testDir, entry.Name())
		replay, err := parser.ParseLogFile(filePath)
		if err != nil || replay.Turns < 5 {
			continue
		}

		matchWinner := 0.5
		if strings.EqualFold(replay.Winner, replay.P1) {
			matchWinner = 1.0
		} else if strings.EqualFold(replay.Winner, replay.P2) {
			matchWinner = 0.0
		}

		samples, _ := buildReplayChosenActionSamples(replay, matchWinner, 1.0)
		for _, sample := range samples {
			mainInputs := buildInputFromPrepared(sample, attentionMLP)
			output := mlp.Forward(mainInputs, nil)
			for action := 0; action < simulator.MaxActions; action++ {
				target := sample.targets[action]
				if target < 0 {
					continue
				}
				if action < len(output) {
					totalLoss += bceLossClamped(target, output[action])
					totalTargets++
				}
			}
		}
	}

	if totalTargets == 0 {
		return 0, 0, nil
	}
	return totalLoss / float64(totalTargets), totalTargets, nil
}

func TrainNetworkFromTagged(taggedDir string, epochs int) error {
	return TrainNetworkFromTaggedWithValidation(taggedDir, "", epochs)
}

func TrainNetworkFromTaggedWithValidation(taggedDir string, testDir string, epochs int) error {
	if err := gamedata.LoadPokedex("data/pokedex.json"); err != nil {
		return fmt.Errorf("failed to load pokedex: %w", err)
	}

	entries, err := os.ReadDir(taggedDir)
	if err != nil {
		return err
	}

	fmt.Printf("Initializing Main MLP [%d -> 2048 -> 1024 -> 512 -> %d]...\n", TotalFeatures, simulator.MaxActions)
	mlp := NewMLP([]int{TotalFeatures, 2048, 1024, 512, simulator.MaxActions})
	if err := mlp.LoadWeights("evaluator_weights.json"); err == nil {
		fmt.Println("Loaded existing evaluator_weights.json...")
	} else {
		fmt.Println("Starting Main MLP from scratch...")
	}

	fmt.Printf("Initializing Attention MLP [%d -> 256 -> 128 -> 12]...\n", TotalSlotFeatures)
	attentionMLP := NewMLP([]int{TotalSlotFeatures, 256, 128, 12})
	attentionMLP.LinearOutput = true
	if err := attentionMLP.LoadWeights("attention_weights.json"); err == nil {
		fmt.Println("Loaded existing attention_weights.json...")
	} else {
		fmt.Println("Starting Attention MLP from scratch...")
	}

	learningRate := 0.001
	bestMetric := math.MaxFloat64
	patienceCounter := 0
	startEpoch := 1

	statePath := "training_state_tagged.json"
	if state, err := LoadTrainingState(statePath); err == nil {
		fmt.Printf("Resuming tagged training from Epoch %d (LR: %.6f, AdamStep: %d)...\n", state.Epoch+1, state.LearningRate, state.MainAdamStep)
		learningRate = state.LearningRate
		bestMetric = state.BestMSE
		patienceCounter = state.PatienceCounter
		startEpoch = state.Epoch + 1
		mlp.AdamStep = state.MainAdamStep
		attentionMLP.AdamStep = state.AttnAdamStep
	}

	const patience = 3
	const lrDecay = 0.5
	const minLR = 1e-5
	numWorkers := runtime.NumCPU()
	kernelBatchSize := tuneKernelBatchSize(mlp, attentionMLP)
	trainingStart := time.Now()
	var cumulativeEpochTime time.Duration

	for epoch := startEpoch; epoch <= epochs; epoch++ {
		epochStart := time.Now()
		epochEntries := make([]os.DirEntry, len(entries))
		copy(epochEntries, entries)
		rand.Shuffle(len(epochEntries), func(i, j int) { epochEntries[i], epochEntries[j] = epochEntries[j], epochEntries[i] })
		maxPerEpoch := 2000000
		if len(epochEntries) > maxPerEpoch {
			epochEntries = epochEntries[:maxPerEpoch]
		}

		jobs := make(chan os.DirEntry, len(epochEntries))
		samples := make(chan preparedSnapshot, kernelBatchSize*8)
		statsCh := make(chan gpuEpochStats, 1)
		go func() {
			statsCh <- runGPUEpochTrainer(mlp, attentionMLP, kernelBatchSize, learningRate, samples)
		}()

		var workerWG sync.WaitGroup
		loadStats := taggedDataLoadStats{}
		var loadStatsMu sync.Mutex

		for w := 0; w < numWorkers; w++ {
			workerWG.Add(1)
			go func() {
				defer workerWG.Done()
				local := taggedDataLoadStats{}
				for entry := range jobs {
					if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
						continue
					}

					filePath := fmt.Sprintf("%s/%s", taggedDir, entry.Name())
					data, err := os.ReadFile(filePath)
					if err != nil {
						local.filesParseFailed++
						continue
					}

					var dataset TaggedReplayDataset
					if err := json.Unmarshal(data, &dataset); err != nil {
						local.filesParseFailed++
						continue
					}
					local.filesLoaded++

					for _, tagged := range dataset.Samples {
						prepared, err := taggedSampleToPrepared(tagged)
						if err != nil {
							local.samplesSkipped++
							continue
						}
						samples <- prepared
						local.samplesLoaded++
					}
				}

				loadStatsMu.Lock()
				loadStats.merge(local)
				loadStatsMu.Unlock()
			}()
		}

		for _, entry := range epochEntries {
			jobs <- entry
		}
		close(jobs)
		workerWG.Wait()
		close(samples)

		stats := <-statsCh
		fmt.Printf("Epoch %d Tagged Data - files: %d loaded, %d failed, samples: %d loaded, %d skipped\n",
			epoch,
			loadStats.filesLoaded,
			loadStats.filesParseFailed,
			loadStats.samplesLoaded,
			loadStats.samplesSkipped,
		)

		totalLoss := stats.totalLoss
		validSnapshots := stats.validSnapshots
		avgLoss := math.MaxFloat64
		if validSnapshots > 0 {
			avgLoss = totalLoss / float64(validSnapshots)
		}
		fmt.Printf("Epoch %d/%d - Train Loss (BCE): %.6f (Trained on %d snapshots, LR: %.6f)\n", epoch, epochs, avgLoss, validSnapshots, learningRate)
		epochElapsed := time.Since(epochStart)
		cumulativeEpochTime += epochElapsed
		completedEpochs := epoch - startEpoch + 1
		remainingEpochs := epochs - epoch
		avgEpoch := cumulativeEpochTime / time.Duration(completedEpochs)
		eta := avgEpoch * time.Duration(remainingEpochs)
		fmt.Printf("  -> Epoch time: %s | Elapsed: %s | ETA: %s\n",
			formatDurationCompact(epochElapsed), formatDurationCompact(time.Since(trainingStart)), formatDurationCompact(eta))

		currentMetric := avgLoss
		if strings.TrimSpace(testDir) != "" {
			testBCE, testTargets, err := evaluateReplaySetBCE(testDir, mlp, attentionMLP)
			if err != nil {
				fmt.Printf("  -> Test evaluation failed: %v\n", err)
			} else if testTargets > 0 {
				fmt.Printf("  -> Test BCE: %.6f (targets=%d)\n", testBCE, testTargets)
				currentMetric = testBCE
			} else {
				fmt.Println("  -> Test BCE skipped (no valid targets)")
			}
		}

		if currentMetric < bestMetric {
			fmt.Printf("  -> New best metric (%.6f -> %.6f). Saving weights...\n", bestMetric, currentMetric)
			attentionMLP.SaveWeights("attention_weights.json")
			mlp.SaveWeights("evaluator_weights.json")

			if currentMetric < bestMetric-0.0001 {
				patienceCounter = 0
			} else {
				patienceCounter++
			}
			bestMetric = currentMetric
		} else {
			patienceCounter++
		}

		if patienceCounter >= patience && learningRate > minLR {
			learningRate *= lrDecay
			if learningRate < minLR {
				learningRate = minLR
			}
			patienceCounter = 0
			fmt.Printf("  -> Reducing learning rate to %.6f (best metric: %.6f)\n", learningRate, bestMetric)
		}

		SaveTrainingState(statePath, TrainingState{
			Epoch:           epoch,
			LearningRate:    learningRate,
			BestMSE:         bestMetric,
			PatienceCounter: patienceCounter,
			MainAdamStep:    mlp.AdamStep,
			AttnAdamStep:    attentionMLP.AdamStep,
		})
	}

	fmt.Println("Saving updated weights...")
	attentionMLP.SaveWeights("attention_weights.json")
	return mlp.SaveWeights("evaluator_weights.json")
}
