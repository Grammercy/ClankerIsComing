package evaluator

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/pokemon-engine/gamedata"
	"github.com/pokemon-engine/parser"
	"github.com/pokemon-engine/simulator"
)

// BatchSize is the number of snapshots to accumulate before applying gradients
const BatchSize = 256
const defaultKernelBatchSize = 64
const kernelTuneIterations = 8

// TrainingState preserves progress across restarts
type TrainingState struct {
	Epoch           int     `json:"epoch"`
	LearningRate    float64 `json:"learningRate"`
	BestMSE         float64 `json:"bestMSE"`
	PatienceCounter int     `json:"patienceCounter"`
	MainAdamStep    int64   `json:"mainAdamStep"`
	AttnAdamStep    int64   `json:"attnAdamStep"`
}

func SaveTrainingState(path string, state TrainingState) {
	data, _ := json.MarshalIndent(state, "", "  ")
	os.WriteFile(path, data, 0644)
}

func LoadTrainingState(path string) (TrainingState, error) {
	var state TrainingState
	data, err := os.ReadFile(path)
	if err != nil {
		return state, err
	}
	err = json.Unmarshal(data, &state)
	return state, err
}

func parseKernelBatchEnv() (int, bool) {
	raw := strings.TrimSpace(os.Getenv("OPENCL_KERNEL_BATCH"))
	if raw == "" {
		return 0, false
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		return 0, false
	}
	return v, true
}

func buildKernelBatchCandidates(maxBatch int) []int {
	base := []int{16, 32, 48, 64, 96, 128, 192, 256, 384, 512}
	seen := make(map[int]struct{}, len(base)+2)
	out := make([]int, 0, len(base)+2)

	add := func(v int) {
		if v <= 0 || v > maxBatch {
			return
		}
		if _, ok := seen[v]; ok {
			return
		}
		seen[v] = struct{}{}
		out = append(out, v)
	}

	add(defaultKernelBatchSize)
	for _, v := range base {
		add(v)
	}
	add(maxBatch)

	if len(out) == 0 {
		return []int{1}
	}
	return out
}

func tuneKernelBatchSize(mlp *MLP, attentionMLP *MLP) int {
	maxBatch := BatchSize
	if v := mlp.OpenCLMaxBatchSize(); v < maxBatch {
		maxBatch = v
	}
	if v := attentionMLP.OpenCLMaxBatchSize(); v < maxBatch {
		maxBatch = v
	}
	if maxBatch < 1 {
		maxBatch = 1
	}

	if envBatch, ok := parseKernelBatchEnv(); ok {
		if envBatch > maxBatch {
			envBatch = maxBatch
		}
		fmt.Printf("Using OPENCL_KERNEL_BATCH=%d (max=%d)\n", envBatch, maxBatch)
		return envBatch
	}

	candidates := buildKernelBatchCandidates(maxBatch)
	if len(candidates) == 1 {
		return candidates[0]
	}

	mainInputs := make([][]float64, maxBatch)
	mainTargets := make([][]float64, maxBatch)
	attnInputs := make([][]float64, maxBatch)
	attnWeights := make([][]float64, maxBatch)
	sampleWeights := make([]float64, maxBatch)
	for s := 0; s < maxBatch; s++ {
		mainRow := make([]float64, TotalFeatures)
		targetRow := make([]float64, simulator.MaxActions)
		attnRow := make([]float64, TotalSlotFeatures)
		weightsRow := make([]float64, 12)
		for i := 0; i < TotalFeatures; i++ {
			mainRow[i] = float64(((s+1)*(i+3))%29) / 29.0
		}
		for i := 0; i < TotalSlotFeatures; i++ {
			attnRow[i] = float64(((s+2)*(i+5))%31) / 31.0
		}
		sum := 0.0
		for i := 0; i < 12; i++ {
			w := float64(((s+3)*(i+7))%37 + 1)
			weightsRow[i] = w
			sum += w
		}
		for i := 0; i < 12; i++ {
			weightsRow[i] /= sum
		}
		for i := 0; i < simulator.MaxActions; i++ {
			if i%3 == 0 {
				targetRow[i] = -1
			} else {
				targetRow[i] = float64((s + i) % 2)
			}
		}
		mainInputs[s] = mainRow
		mainTargets[s] = targetRow
		attnInputs[s] = attnRow
		attnWeights[s] = weightsRow
		sampleWeights[s] = 1.0
	}

	bestBatch := candidates[0]
	bestRate := -1.0

	fmt.Printf("Auto-tuning OpenCL kernel batch size using train-step kernels (max=%d)...\n", maxBatch)
	for _, batch := range candidates {
		// Warm up driver caches and command submission paths with a real train-step batch.
		_ = mlp.CalculateBCELocalGradientsBatch(mainInputs[:batch], mainTargets[:batch], sampleWeights[:batch])
		warmGrads := mlp.FirstLayerInputGradSliceBatch(TotalGlobals, TotalSlotFeatures, batch)
		attentionMLP.BackpropAttentionFromInputGradsBatch(attnInputs[:batch], attnWeights[:batch], warmGrads, sampleWeights[:batch], FeaturesPerSlot)
		mlp.ClearGradients()
		attentionMLP.ClearGradients()

		start := time.Now()
		for i := 0; i < kernelTuneIterations; i++ {
			_ = mlp.CalculateBCELocalGradientsBatch(mainInputs[:batch], mainTargets[:batch], sampleWeights[:batch])
			grads := mlp.FirstLayerInputGradSliceBatch(TotalGlobals, TotalSlotFeatures, batch)
			attentionMLP.BackpropAttentionFromInputGradsBatch(attnInputs[:batch], attnWeights[:batch], grads, sampleWeights[:batch], FeaturesPerSlot)
		}
		elapsed := time.Since(start)
		mlp.ClearGradients()
		attentionMLP.ClearGradients()
		if elapsed <= 0 {
			continue
		}
		rate := float64(batch*kernelTuneIterations) / elapsed.Seconds()
		fmt.Printf("  candidate %d -> %.0f train-samples/s\n", batch, rate)
		if rate > bestRate {
			bestRate = rate
			bestBatch = batch
		}
	}

	fmt.Printf("Selected kernel batch size: %d\n", bestBatch)
	return bestBatch
}

type preparedSnapshot struct {
	prefix    []float64
	rawSlots  []float64
	targets   []float64
	eloWeight float64
}

type gpuEpochStats struct {
	totalLoss      float64
	validSnapshots int
}

type replayLabelStats struct {
	candidates        int
	mapped            int
	skipped           int
	moveSlotFallbacks int
	skipBy            map[string]int
	actionFreq        [simulator.MaxActions]int
}

func newReplayLabelStats() replayLabelStats {
	return replayLabelStats{
		skipBy: make(map[string]int),
	}
}

func (s *replayLabelStats) skip(reason string) {
	s.skipped++
	s.skipBy[reason]++
}

func (s *replayLabelStats) merge(other replayLabelStats) {
	s.candidates += other.candidates
	s.mapped += other.mapped
	s.skipped += other.skipped
	s.moveSlotFallbacks += other.moveSlotFallbacks
	for k, v := range other.skipBy {
		s.skipBy[k] += v
	}
	for i := 0; i < simulator.MaxActions; i++ {
		s.actionFreq[i] += other.actionFreq[i]
	}
}

func normalizeReplayMoveName(move string) string {
	s := strings.ToLower(move)
	s = strings.ReplaceAll(s, " ", "")
	s = strings.ReplaceAll(s, "-", "")
	s = strings.ReplaceAll(s, "'", "")
	s = strings.ReplaceAll(s, ".", "")
	s = strings.ReplaceAll(s, ":", "")
	return s
}

func mapReplaySwitchAction(state *simulator.BattleState, event parser.Event) (int, string) {
	species := strings.TrimSpace(event.Value)
	if species == "" {
		return -1, "empty_switch_species"
	}
	for i := 0; i < state.P1.TeamSize; i++ {
		teamMon := &state.P1.Team[i]
		if strings.EqualFold(teamMon.Species, species) || strings.EqualFold(teamMon.Name, species) {
			return simulator.ActionSwitchBase + i, ""
		}
	}
	return -1, "unknown_switch_slot"
}

func mapReplayMoveSlot(state *simulator.BattleState, moveName string) (int, string, bool) {
	active := state.P1.GetActive()
	if active == nil {
		return -1, "no_active_pokemon", false
	}
	if active.Fainted {
		return -1, "active_fainted", false
	}

	needle := normalizeReplayMoveName(moveName)
	known := active.NumMoves
	if known < 0 {
		known = 0
	}
	if known > 4 {
		known = 4
	}
	for i := 0; i < known; i++ {
		if normalizeReplayMoveName(active.Moves[i]) == needle {
			return i, "", false
		}
	}

	// Fallback for first-seen move usage: if this move is not in the known list yet
	// and there is an unfilled move slot, assume it occupied the next slot.
	if known < 4 && needle != "" {
		return known, "", true
	}
	return -1, "unknown_move_slot", false
}

func isActionValidForState(state *simulator.BattleState, action int) bool {
	validActions, n := simulator.GetSearchActions(&state.P1)
	for i := 0; i < n; i++ {
		if validActions[i] == action {
			return true
		}
	}
	return false
}

func buildPreparedSnapshot(state *simulator.BattleState, chosenAction int, matchWinner float64, eloWeight float64) (preparedSnapshot, string, bool) {
	if chosenAction < 0 || chosenAction >= simulator.MaxActions {
		return preparedSnapshot{}, "mapped_action_out_of_range", false
	}
	if !isActionValidForState(state, chosenAction) {
		return preparedSnapshot{}, "mapped_action_not_valid_in_state", false
	}

	p1Globals, p1Slots := vectorizePlayerFeatures(&state.P1, state)
	p2Globals, p2Slots := vectorizePlayerFeatures(&state.P2, state)

	targets := make([]float64, simulator.MaxActions)
	for i := 0; i < simulator.MaxActions; i++ {
		targets[i] = -1.0
	}
	targets[chosenAction] = matchWinner

	rawSlots := make([]float64, 0, TotalSlotFeatures)
	rawSlots = append(rawSlots, p1Slots[:]...)
	rawSlots = append(rawSlots, p2Slots[:]...)

	var prefixArr [TotalFeatures]float64
	idx := 0
	prefixArr[idx] = p1Globals[0]
	idx++
	prefixArr[idx] = p1Globals[1]
	idx++
	prefixArr[idx] = p2Globals[0]
	idx++
	prefixArr[idx] = p2Globals[1]
	idx++
	vectorizeFieldConditions(&state.Field, &prefixArr, &idx)
	vectorizeSideConditions(&state.P1.Side, &prefixArr, &idx)
	vectorizeSideConditions(&state.P2.Side, &prefixArr, &idx)
	vectorizeBoosts(state.P1.GetActive(), &prefixArr, &idx)
	vectorizeBoosts(state.P2.GetActive(), &prefixArr, &idx)
	vectorizeMatchup(state, &prefixArr, &idx)

	return preparedSnapshot{
		prefix:    append([]float64(nil), prefixArr[:TotalGlobals]...),
		rawSlots:  rawSlots,
		targets:   targets,
		eloWeight: eloWeight,
	}, "", true
}

func findNextMoveInTurn(events []parser.Event, startIdx int, player string) int {
	if startIdx < 0 || startIdx >= len(events) {
		return -1
	}
	turn := events[startIdx].Turn
	for i := startIdx + 1; i < len(events); i++ {
		if events[i].Turn != turn {
			break
		}
		if events[i].Player == player && events[i].Type == "move" {
			return i
		}
	}
	return -1
}

func hasPriorTerastallizeInTurn(events []parser.Event, idx int, player string) bool {
	if idx < 0 || idx >= len(events) {
		return false
	}
	turn := events[idx].Turn
	for i := idx - 1; i >= 0; i-- {
		if events[i].Turn != turn {
			break
		}
		if events[i].Player == player && events[i].Type == "terastallize" {
			return true
		}
	}
	return false
}

func actionLabelForIndex(action int) string {
	if action >= simulator.ActionMove1 && action <= simulator.ActionMove4 {
		return fmt.Sprintf("move%d", action-simulator.ActionMove1+1)
	}
	if action >= simulator.ActionTeraMove1 && action <= simulator.ActionTeraMove4 {
		return fmt.Sprintf("teramove%d", action-simulator.ActionTeraMove1+1)
	}
	if action >= simulator.ActionSwitchBase && action < simulator.MaxActions {
		return fmt.Sprintf("switch%d", action-simulator.ActionSwitchBase)
	}
	return fmt.Sprintf("action%d", action)
}

func buildReplayChosenActionSamples(replay *parser.Replay, matchWinner float64, eloWeight float64) ([]preparedSnapshot, replayLabelStats) {
	stats := newReplayLabelStats()
	samples := make([]preparedSnapshot, 0, len(replay.Events)/2)

	for i, event := range replay.Events {
		if event.Player != "p1" {
			continue
		}

		switch event.Type {
		case "switch":
			stats.candidates++
			state, err := simulator.FastForwardToEvent(replay, i-1)
			if err != nil {
				stats.skip("fastforward_error")
				continue
			}
			action, reason := mapReplaySwitchAction(state, event)
			if reason != "" {
				stats.skip(reason)
				continue
			}
			snapshot, reason, ok := buildPreparedSnapshot(state, action, matchWinner, eloWeight)
			if !ok {
				stats.skip(reason)
				continue
			}
			samples = append(samples, snapshot)
			stats.mapped++
			stats.actionFreq[action]++

		case "move":
			// Tera move training sample is sourced from the terastallize event
			// to capture pre-tera state. Skip this move if tera already happened this turn.
			if hasPriorTerastallizeInTurn(replay.Events, i, "p1") {
				continue
			}
			stats.candidates++
			state, err := simulator.FastForwardToEvent(replay, i-1)
			if err != nil {
				stats.skip("fastforward_error")
				continue
			}
			slot, reason, usedFallback := mapReplayMoveSlot(state, event.Value)
			if reason != "" {
				stats.skip(reason)
				continue
			}
			if usedFallback {
				stats.moveSlotFallbacks++
			}
			action := simulator.ActionMove1 + slot
			snapshot, reason, ok := buildPreparedSnapshot(state, action, matchWinner, eloWeight)
			if !ok {
				stats.skip(reason)
				continue
			}
			samples = append(samples, snapshot)
			stats.mapped++
			stats.actionFreq[action]++

		case "terastallize":
			stats.candidates++
			moveIdx := findNextMoveInTurn(replay.Events, i, "p1")
			if moveIdx == -1 {
				stats.skip("tera_without_followup_move")
				continue
			}
			state, err := simulator.FastForwardToEvent(replay, i-1)
			if err != nil {
				stats.skip("fastforward_error")
				continue
			}
			slot, reason, usedFallback := mapReplayMoveSlot(state, replay.Events[moveIdx].Value)
			if reason != "" {
				stats.skip(reason)
				continue
			}
			if usedFallback {
				stats.moveSlotFallbacks++
			}
			action := simulator.ActionTeraMove1 + slot
			snapshot, reason, ok := buildPreparedSnapshot(state, action, matchWinner, eloWeight)
			if !ok {
				stats.skip(reason)
				continue
			}
			samples = append(samples, snapshot)
			stats.mapped++
			stats.actionFreq[action]++
		}
	}

	return samples, stats
}

func runGPUEpochTrainer(mlp *MLP, attentionMLP *MLP, kernelBatchSize int, learningRate float64, samples <-chan preparedSnapshot) gpuEpochStats {
	stats := gpuEpochStats{}
	workerBatchCount := 0
	batch := make([]preparedSnapshot, 0, kernelBatchSize)

	flushKernelBatch := func() {
		n := len(batch)
		if n == 0 {
			return
		}

		rawSlotsBatch := make([][]float64, n)
		targetsBatch := make([][]float64, n)
		eloWeightsBatch := make([]float64, n)
		for i, sample := range batch {
			rawSlotsBatch[i] = sample.rawSlots
			targetsBatch[i] = sample.targets
			eloWeightsBatch[i] = sample.eloWeight
		}

		rawAttentionBatch := attentionMLP.ForwardBatch(rawSlotsBatch)
		attentionWeightsBatch := make([][]float64, n)
		mainInputsBatch := make([][]float64, n)
		attentionSlotCount := 0
		for sampleIdx, sample := range batch {
			rawAttention := rawAttentionBatch[sampleIdx]
			if attentionSlotCount == 0 {
				attentionSlotCount = len(rawAttention)
			}
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

			mainInputs := make([]float64, TotalFeatures)
			copy(mainInputs, sample.prefix)
			idx := TotalGlobals
			for slotIdx := 0; slotIdx < len(attentionWeights); slotIdx++ {
				w := attentionWeights[slotIdx]
				base := slotIdx * FeaturesPerSlot
				for j := 0; j < FeaturesPerSlot; j++ {
					mainInputs[idx] = sample.rawSlots[base+j] * w
					idx++
				}
			}

			attentionWeightsBatch[sampleIdx] = attentionWeights
			mainInputsBatch[sampleIdx] = mainInputs
		}

		batchLoss := mlp.CalculateBCELocalGradientsBatch(mainInputsBatch, targetsBatch, eloWeightsBatch)
		stats.totalLoss += batchLoss
		stats.validSnapshots += n

		attentionDeltaFlat := mlp.AttentionOutputDeltasFromFirstLayerBatch(
			TotalGlobals,
			rawSlotsBatch,
			attentionWeightsBatch,
			eloWeightsBatch,
			FeaturesPerSlot,
			attentionSlotCount,
		)
		attentionDeltaBatch := make([][]float64, n)
		for sample := 0; sample < n; sample++ {
			base := sample * attentionSlotCount
			row := make([]float64, attentionSlotCount)
			copy(row, attentionDeltaFlat[base:base+attentionSlotCount])
			attentionDeltaBatch[sample] = row
		}
		attentionMLP.BackpropGivenDeltasBatch(rawSlotsBatch, attentionDeltaBatch, eloWeightsBatch)

		workerBatchCount += n
		for workerBatchCount >= BatchSize {
			mlp.ApplyAdamGradients(nil, float64(BatchSize), learningRate, 0.9, 0.999, 1e-8)
			attentionMLP.ApplyAdamGradients(nil, float64(BatchSize), learningRate, 0.9, 0.999, 1e-8)
			workerBatchCount -= BatchSize
		}
		batch = batch[:0]
	}

	for sample := range samples {
		batch = append(batch, sample)
		if len(batch) >= kernelBatchSize {
			flushKernelBatch()
		}
	}
	flushKernelBatch()

	if workerBatchCount > 0 {
		mlp.ApplyAdamGradients(nil, float64(workerBatchCount), learningRate, 0.9, 0.999, 1e-8)
		attentionMLP.ApplyAdamGradients(nil, float64(workerBatchCount), learningRate, 0.9, 0.999, 1e-8)
	}
	return stats
}

// TrainNetwork runs the ELO-weighted training loop
func TrainNetwork(replaysDir string, epochs int) error {
	// Load Pokedex data for feature extraction
	if err := gamedata.LoadPokedex("data/pokedex.json"); err != nil {
		return fmt.Errorf("failed to load pokedex: %w", err)
	}

	entries, err := os.ReadDir(replaysDir)
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
	bestMSE := math.MaxFloat64
	patienceCounter := 0
	startEpoch := 1

	// Attempt to load previous training state
	if state, err := LoadTrainingState("training_state.json"); err == nil {
		fmt.Printf("Resuming from Epoch %d (LR: %.6f, AdamStep: %d)...\n", state.Epoch+1, state.LearningRate, state.MainAdamStep)
		learningRate = state.LearningRate
		bestMSE = state.BestMSE
		patienceCounter = state.PatienceCounter
		startEpoch = state.Epoch + 1
		mlp.AdamStep = state.MainAdamStep
		attentionMLP.AdamStep = state.AttnAdamStep
	}

	const patience = 3  // epochs without improvement before reducing LR
	const lrDecay = 0.5 // multiply LR by this on plateau
	const minLR = 1e-5  // floor to prevent vanishing updates
	numWorkers := runtime.NumCPU()
	kernelBatchSize := tuneKernelBatchSize(mlp, attentionMLP)
	trainingStart := time.Now()
	var cumulativeEpochTime time.Duration

	for epoch := startEpoch; epoch <= epochs; epoch++ {
		epochStart := time.Now()
		// Shuffle and sample 50k random games per epoch
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

		var parserWG sync.WaitGroup
		labelStats := newReplayLabelStats()
		var labelStatsMu sync.Mutex
		for w := 0; w < numWorkers; w++ {
			parserWG.Add(1)
			go func() {
				defer parserWG.Done()
				localStats := newReplayLabelStats()
				for entry := range jobs {
					filePath := fmt.Sprintf("%s/%s", replaysDir, entry.Name())
					eloWeight := 1.0
					matchWinner := 0.5

					if strings.HasSuffix(entry.Name(), ".json") {
						// Self-play snapshots do not contain chosen-action labels.
						localStats.skip("json_without_action_labels")
						continue
					} else if strings.HasSuffix(entry.Name(), ".log") {
						replay, err := parser.ParseLogFile(filePath)
						if err != nil || replay.Turns < 5 {
							localStats.skip("invalid_or_short_replay")
							continue
						}
						avgRating := float64(replay.P1Rating+replay.P2Rating) / 2.0
						if avgRating < 750.0 {
							avgRating = 750.0
						}
						eloWeight = avgRating / 1500.0
						if strings.EqualFold(replay.Winner, replay.P1) {
							matchWinner = 1.0
						} else if strings.EqualFold(replay.Winner, replay.P2) {
							matchWinner = 0.0
						}

						replaySamples, replayStats := buildReplayChosenActionSamples(replay, matchWinner, eloWeight)
						localStats.merge(replayStats)
						for _, sample := range replaySamples {
							samples <- sample
						}
					} else {
						localStats.skip("unsupported_file_extension")
						continue
					}
				}

				labelStatsMu.Lock()
				labelStats.merge(localStats)
				labelStatsMu.Unlock()
			}()
		}

		for _, entry := range epochEntries {
			jobs <- entry
		}
		close(jobs)
		parserWG.Wait()
		close(samples)

		stats := <-statsCh
		coverage := 0.0
		if labelStats.candidates > 0 {
			coverage = 100.0 * float64(labelStats.mapped) / float64(labelStats.candidates)
		}
		fmt.Printf("Epoch %d Label Mapping - candidates: %d, mapped: %d, skipped: %d (coverage: %.1f%%)\n",
			epoch, labelStats.candidates, labelStats.mapped, labelStats.skipped, coverage)
		if labelStats.moveSlotFallbacks > 0 {
			fmt.Printf("  move_slot_fallbacks=%d\n", labelStats.moveSlotFallbacks)
		}
		if len(labelStats.skipBy) > 0 {
			keys := make([]string, 0, len(labelStats.skipBy))
			for k := range labelStats.skipBy {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				fmt.Printf("  skip[%s]=%d\n", k, labelStats.skipBy[k])
			}
		}
		fmt.Printf("Epoch %d Action Frequency:", epoch)
		for action := 0; action < simulator.MaxActions; action++ {
			if labelStats.actionFreq[action] == 0 {
				continue
			}
			fmt.Printf(" %s=%d", actionLabelForIndex(action), labelStats.actionFreq[action])
		}
		fmt.Printf("\n")

		totalLoss := stats.totalLoss
		validSnapshots := stats.validSnapshots
		avgLoss := math.MaxFloat64
		if validSnapshots > 0 {
			avgLoss = totalLoss / float64(validSnapshots)
		}
		fmt.Printf("Epoch %d/%d - Average Loss (BCE): %.6f (Trained on %d snapshots, LR: %.6f)\n", epoch, epochs, avgLoss, validSnapshots, learningRate)
		epochElapsed := time.Since(epochStart)
		cumulativeEpochTime += epochElapsed
		completedEpochs := epoch - startEpoch + 1
		remainingEpochs := epochs - epoch
		avgEpoch := cumulativeEpochTime / time.Duration(completedEpochs)
		eta := avgEpoch * time.Duration(remainingEpochs)
		fmt.Printf("  -> Epoch time: %s | Elapsed: %s | ETA: %s\n",
			formatDurationCompact(epochElapsed), formatDurationCompact(time.Since(trainingStart)), formatDurationCompact(eta))

		// Save weights if loss improved (went down)
		if avgLoss < bestMSE {
			fmt.Printf("  -> New best loss (%.6f -> %.6f). Saving weights...\n", bestMSE, avgLoss)
			attentionMLP.SaveWeights("attention_weights.json")
			mlp.SaveWeights("evaluator_weights.json")

			if avgLoss < bestMSE-0.0001 {
				patienceCounter = 0
			} else {
				patienceCounter++
			}
			bestMSE = avgLoss
		} else {
			patienceCounter++
		}

		// Adaptive LR: reduce on plateau
		if patienceCounter >= patience && learningRate > minLR {
			learningRate *= lrDecay
			if learningRate < minLR {
				learningRate = minLR
			}
			patienceCounter = 0
			fmt.Printf("  -> Reducing learning rate to %.6f (best MSE: %.6f)\n", learningRate, bestMSE)
		}

		// Save state after each epoch
		SaveTrainingState("training_state.json", TrainingState{
			Epoch:           epoch,
			LearningRate:    learningRate,
			BestMSE:         bestMSE,
			PatienceCounter: patienceCounter,
			MainAdamStep:    mlp.AdamStep,
			AttnAdamStep:    attentionMLP.AdamStep,
		})
	}

	fmt.Println("Saving updated weights...")
	attentionMLP.SaveWeights("attention_weights.json")
	return mlp.SaveWeights("evaluator_weights.json")
}

func formatDurationCompact(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	d = d.Round(time.Second)
	h := d / time.Hour
	d -= h * time.Hour
	m := d / time.Minute
	d -= m * time.Minute
	s := d / time.Second
	if h > 0 {
		return fmt.Sprintf("%dh%02dm%02ds", h, m, s)
	}
	if m > 0 {
		return fmt.Sprintf("%dm%02ds", m, s)
	}
	return fmt.Sprintf("%ds", s)
}
