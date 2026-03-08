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
	"sync/atomic"
	"time"

	"github.com/pokemon-engine/gamedata"
	"github.com/pokemon-engine/parser"
	"github.com/pokemon-engine/simulator"
)

// BatchSize is the number of snapshots to accumulate before applying gradients
const BatchSize = 256
const defaultKernelBatchSize = 64
const kernelTuneIterations = 8
const WeightDecay = 0.0001
const TrainingTurnThreshold = 20
const trainLatentBootstrapPasses = 5

// TrainingState preserves progress across restarts
type TrainingState struct {
	Epoch           int     `json:"epoch"`
	LearningRate    float64 `json:"learningRate"`
	BestMSE         float64 `json:"bestMSE"`
	PatienceCounter int     `json:"patienceCounter"`
	MainAdamStep    int64   `json:"mainAdamStep"`
	AttnAdamStep    int64   `json:"attnAdamStep"`
}

var globalGameCounter uint64

type GamePhaseStats struct {
	Correct [3]int
	Total   [3]int
}

type GameStatsTracker struct {
	mu          sync.Mutex
	last1000    []uint64
	gameMap     map[uint64]*GamePhaseStats
	historySize int
}

func NewGameStatsTracker(size int) *GameStatsTracker {
	return &GameStatsTracker{
		gameMap:     make(map[uint64]*GamePhaseStats),
		historySize: size,
	}
}

func (t *GameStatsTracker) Record(gameID uint64, turnPct float64, correct bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	stats, exists := t.gameMap[gameID]
	if !exists {
		stats = &GamePhaseStats{}
		t.gameMap[gameID] = stats
		t.last1000 = append(t.last1000, gameID)
		if len(t.last1000) > t.historySize {
			oldID := t.last1000[0]
			t.last1000 = t.last1000[1:]
			delete(t.gameMap, oldID)
		}
	}

	phase := 0
	if turnPct >= 0.66 {
		phase = 2
	} else if turnPct >= 0.33 {
		phase = 1
	}

	stats.Total[phase]++
	if correct {
		stats.Correct[phase]++
	}
}

func (t *GameStatsTracker) GetPhaseAccuracy() [3]float64 {
	t.mu.Lock()
	defer t.mu.Unlock()

	var totals [3]int
	var corrects [3]int
	for _, id := range t.last1000 {
		s := t.gameMap[id]
		for i := 0; i < 3; i++ {
			totals[i] += s.Total[i]
			corrects[i] += s.Correct[i]
		}
	}

	var acc [3]float64
	for i := 0; i < 3; i++ {
		if totals[i] > 0 {
			acc[i] = float64(corrects[i]) / float64(totals[i])
		}
	}
	return acc
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

func mainTargetsWithMaskedLatents(actionTargets []float64) []float64 {
	targets := make([]float64, MainOutputSize)
	for i := range targets {
		targets[i] = -1.0
	}
	for i := 0; i < MainActionOutputSize && i < len(actionTargets); i++ {
		targets[i] = actionTargets[i]
	}
	return targets
}

func setPrefixLatentsFromOutput(prefix []float64, out []float64) {
	if len(prefix) < TotalGlobals || len(out) <= PredictionTokenOutputIndex {
		return
	}
	prefix[TotalGlobals-LatentTokenFeatures] = out[ReasoningTokenOutputIndex]
	prefix[TotalGlobals-1] = out[PredictionTokenOutputIndex]
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
	sampleWeights := make([]float64, maxBatch)
	for s := 0; s < maxBatch; s++ {
		mainRow := make([]float64, TotalFeatures)
		targetRow := make([]float64, MainOutputSize)
		attnRow := make([]float64, TotalSlotFeatures)
		for i := 0; i < TotalFeatures; i++ {
			mainRow[i] = float64(((s+1)*(i+3))%29) / 29.0
		}
		for i := 0; i < TotalSlotFeatures; i++ {
			attnRow[i] = float64(((s+2)*(i+5))%31) / 31.0
		}
		for i := 0; i < simulator.MaxActions; i++ {
			if i%3 == 0 {
				targetRow[i] = -1
			} else {
				targetRow[i] = float64((s + i) % 2)
			}
		}
		targetRow[ReasoningTokenOutputIndex] = -1
		targetRow[PredictionTokenOutputIndex] = -1
		mainInputs[s] = mainRow
		mainTargets[s] = targetRow
		attnInputs[s] = attnRow
		sampleWeights[s] = 1.0
	}

	bestBatch := candidates[0]
	bestRate := -1.0

	fmt.Printf("Auto-tuning OpenCL kernel batch size using train-step kernels (max=%d)...\n", maxBatch)
	for _, batch := range candidates {
		runOneStep := func() {
			rawAttentionBatch := attentionMLP.ForwardBatch(attnInputs[:batch])
			slotWeightsBatch := make([][]float64, batch)
			defaultWeights := uniformSlotWeights()
			for i, rawAttention := range rawAttentionBatch {
				slotWeights, ok := attentionWeightsFromOutput(rawAttention)
				if !ok {
					slotWeights = defaultWeights
				}
				row := make([]float64, SlotAttentionSlots)
				copy(row, slotWeights[:])
				slotWeightsBatch[i] = row
			}

			_, _ = mlp.CalculateBCELocalGradientsBatch(mainInputs[:batch], mainTargets[:batch], sampleWeights[:batch])
			slotDeltaFlat := mlp.AttentionOutputDeltasFromFirstLayerBatch(
				TotalGlobals,
				attnInputs[:batch],
				slotWeightsBatch,
				sampleWeights[:batch],
				FeaturesPerSlot,
				SlotAttentionSlots,
			)
			slotDeltaBatch := make([][]float64, batch)
			for i := 0; i < batch; i++ {
				base := i * SlotAttentionSlots
				row := make([]float64, SlotAttentionSlots)
				copy(row, slotDeltaFlat[base:base+SlotAttentionSlots])
				slotDeltaBatch[i] = row
			}
			attentionDeltas := buildAttentionOutputDeltasBatch(rawAttentionBatch, slotDeltaBatch)
			attentionMLP.BackpropGivenDeltasBatch(attnInputs[:batch], attentionDeltas, sampleWeights[:batch])
		}

		// Warm up driver caches and command submission paths with a real train-step batch.
		runOneStep()
		mlp.ClearGradients()
		attentionMLP.ClearGradients()

		start := time.Now()
		for i := 0; i < kernelTuneIterations; i++ {
			runOneStep()
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
	prefix       []float64
	rawSlots     []float64
	targets      []float64
	validActions []int
	eloWeight    float64
	isSearchTag  bool
	turn         int
	GameID       uint64
	TurnPercent  float64
}

type gpuEpochStats struct {
	totalLoss      float64
	totalBCELoss   float64
	validSnapshots int
	PhaseAccuracy  [3]float64
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

func buildPreparedSnapshot(state *simulator.BattleState, chosenAction int, matchWinner float64, eloWeight float64, turn int, totalTurns int, gameID uint64) (preparedSnapshot, string, bool) {
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

	va, nVa := simulator.GetSearchActions(&state.P1)
	validActions := make([]int, nVa)
	copy(validActions, va[:nVa])

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
	vectorizeLatentTokens(&prefixArr, &idx, DefaultLatentReasoningToken, DefaultLatentPredictionToken)

	turnPct := 0.0
	if totalTurns > 0 {
		turnPct = float64(turn) / float64(totalTurns)
	}

	return preparedSnapshot{
		prefix:       append([]float64(nil), prefixArr[:TotalGlobals]...),
		rawSlots:     rawSlots,
		targets:      targets,
		validActions: validActions,
		eloWeight:    eloWeight,
		turn:         turn,
		GameID:       gameID,
		TurnPercent:  turnPct,
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

func appendLatentBootstrapSnapshots(samples []preparedSnapshot, base preparedSnapshot) []preparedSnapshot {
	for i := 0; i < trainLatentBootstrapPasses; i++ {
		clone := base
		clone.prefix = append([]float64(nil), base.prefix...)
		clone.prefix[TotalGlobals-LatentTokenFeatures] = rand.Float64()*2.0 - 1.0
		clone.prefix[TotalGlobals-1] = rand.Float64()*2.0 - 1.0
		samples = append(samples, clone)
	}
	return samples
}

func buildReplayChosenActionSamples(replay *parser.Replay, matchWinner float64, eloWeight float64, gameID uint64) ([]preparedSnapshot, replayLabelStats) {
	stats := newReplayLabelStats()
	samples := make([]preparedSnapshot, 0, (len(replay.Events)/2)*trainLatentBootstrapPasses)

	for i, event := range replay.Events {
		if event.Player != "p1" || event.Turn < TrainingTurnThreshold {
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
			snapshot, reason, ok := buildPreparedSnapshot(state, action, matchWinner, eloWeight, event.Turn, replay.Turns, gameID)
			if !ok {
				stats.skip(reason)
				continue
			}
			samples = appendLatentBootstrapSnapshots(samples, snapshot)
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
			snapshot, reason, ok := buildPreparedSnapshot(state, action, matchWinner, eloWeight, event.Turn, replay.Turns, gameID)
			if !ok {
				stats.skip(reason)
				continue
			}
			samples = appendLatentBootstrapSnapshots(samples, snapshot)
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
			snapshot, reason, ok := buildPreparedSnapshot(state, action, matchWinner, eloWeight, event.Turn, replay.Turns, gameID)
			if !ok {
				stats.skip(reason)
				continue
			}
			samples = appendLatentBootstrapSnapshots(samples, snapshot)
			stats.mapped++
			stats.actionFreq[action]++
		}
	}

	return samples, stats
}

func runGPUEpochTrainer(mlp *MLP, attentionMLP *MLP, kernelBatchSize int, learningRate float64, samples <-chan preparedSnapshot, progressCh chan<- gpuEpochStats) gpuEpochStats {
	stats := gpuEpochStats{}
	tracker := NewGameStatsTracker(1000)
	workerBatchCount := 0
	batch := make([]preparedSnapshot, 0, kernelBatchSize)
	defaultWeights := uniformSlotWeights()

	flushKernelBatch := func() {
		n := len(batch)
		if n == 0 {
			return
		}

		rawSlotsBatch := make([][]float64, n)
		targetsBatch := make([][]float64, n)
		eloWeightsBatch := make([]float64, n)
		prefixBatch := make([][]float64, n)
		for i, sample := range batch {
			rawSlotsBatch[i] = sample.rawSlots
			targetsBatch[i] = mainTargetsWithMaskedLatents(sample.targets)
			eloWeightsBatch[i] = sample.eloWeight
			prefixBatch[i] = append([]float64(nil), sample.prefix...)
		}

		var rawAttentionBatch [][]float64
		var bceLoss float64
		var outputsBatch [][]float64
		var attentionWeightsBatch [][]float64

		rawAttentionBatch = attentionMLP.ForwardBatch(rawSlotsBatch)
		attentionWeightsBatch = make([][]float64, n)
		for sampleIdx := range batch {
			slotWeights, ok := attentionWeightsFromOutput(rawAttentionBatch[sampleIdx])
			if !ok {
				slotWeights = defaultWeights
			}
			attentionWeights := make([]float64, SlotAttentionSlots)
			copy(attentionWeights, slotWeights[:])

			attentionWeightsBatch[sampleIdx] = attentionWeights
		}

		for step := 0; step < LatentRecurrenceSteps; step++ {
			mainInputsBatch := make([][]float64, n)
			for sampleIdx := range batch {
				mainInputs := make([]float64, TotalFeatures)
				copy(mainInputs, prefixBatch[sampleIdx])
				idx := TotalGlobals
				for slotIdx := 0; slotIdx < SlotAttentionSlots; slotIdx++ {
					w := attentionWeightsBatch[sampleIdx][slotIdx]
					base := slotIdx * FeaturesPerSlot
					for j := 0; j < FeaturesPerSlot; j++ {
						mainInputs[idx] = rawSlotsBatch[sampleIdx][base+j] * w
						idx++
					}
				}
				mainInputsBatch[sampleIdx] = mainInputs
			}

			if step == LatentRecurrenceSteps-1 {
				bceLoss, outputsBatch = mlp.CalculateBCELocalGradientsBatch(mainInputsBatch, targetsBatch, eloWeightsBatch)
			} else {
				stepOutputs := mlp.ForwardBatch(mainInputsBatch)
				for i := 0; i < n; i++ {
					setPrefixLatentsFromOutput(prefixBatch[i], stepOutputs[i])
				}
			}
		}
		stats.validSnapshots += n

		for i := 0; i < n; i++ {
			output := outputsBatch[i]
			target := targetsBatch[i]
			chosenAction := -1
			for j := 0; j < len(target); j++ {
				if target[j] >= 0 {
					chosenAction = j
					break
				}
			}
			if chosenAction == -1 {
				continue
			}

			targetWin := target[chosenAction] >= 0.5
			predictedWin := output[chosenAction] >= 0.5
			tracker.Record(batch[i].GameID, batch[i].TurnPercent, predictedWin == targetWin)
		}
		stats.PhaseAccuracy = tracker.GetPhaseAccuracy()

		attentionDeltaFlat := mlp.AttentionOutputDeltasFromFirstLayerBatch(
			TotalGlobals,
			rawSlotsBatch,
			attentionWeightsBatch,
			eloWeightsBatch,
			FeaturesPerSlot,
			SlotAttentionSlots,
		)
		slotDeltaBatch := make([][]float64, n)
		for sample := 0; sample < n; sample++ {
			base := sample * SlotAttentionSlots
			row := make([]float64, SlotAttentionSlots)
			copy(row, attentionDeltaFlat[base:base+SlotAttentionSlots])
			slotDeltaBatch[sample] = row
		}

		attentionDeltaBatch := buildAttentionOutputDeltasBatch(rawAttentionBatch, slotDeltaBatch)
		attentionMLP.BackpropGivenDeltasBatch(rawSlotsBatch, attentionDeltaBatch, eloWeightsBatch)

		stats.totalBCELoss += bceLoss
		stats.totalLoss += bceLoss

		workerBatchCount += n
		for workerBatchCount >= BatchSize {
			mlp.ApplyAdamGradients(nil, float64(BatchSize), learningRate, WeightDecay, 0.9, 0.999, 1e-8)
			attentionMLP.ApplyAdamGradients(nil, float64(BatchSize), learningRate, WeightDecay, 0.9, 0.999, 1e-8)
			workerBatchCount -= BatchSize
		}
		batch = batch[:0]

		progressCh <- stats
	}

	for sample := range samples {
		batch = append(batch, sample)
		if len(batch) >= kernelBatchSize {
			flushKernelBatch()
		}
	}
	flushKernelBatch()

	if workerBatchCount > 0 {
		mlp.ApplyAdamGradients(nil, float64(workerBatchCount), learningRate, WeightDecay, 0.9, 0.999, 1e-8)
		attentionMLP.ApplyAdamGradients(nil, float64(workerBatchCount), learningRate, WeightDecay, 0.9, 0.999, 1e-8)
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

	mainSizes := mainMLPLayerSizes()
	attnSizes := attentionMLPLayerSizes()
	fmt.Printf("Initializing Main MLP %v (%d params)...\n", mainSizes, mlpParamCount(mainSizes))
	fmt.Printf("Model total (main + attention): %d params\n", totalEvaluatorParamCount())
	mlp := NewMLP(mainSizes)
	if err := mlp.LoadWeights("evaluator_weights.json"); err == nil {
		if mlp.HasLayerSizes(mainSizes) {
			fmt.Println("Loaded existing evaluator_weights.json...")
		} else {
			fmt.Println("Ignoring incompatible evaluator_weights.json (old architecture); starting Main MLP from scratch...")
			mlp = NewMLP(mainSizes)
		}
	} else {
		fmt.Println("Starting Main MLP from scratch...")
	}

	fmt.Printf("Initializing Attention MLP %v (%d params)...\n", attnSizes, mlpParamCount(attnSizes))
	attentionMLP := NewMLP(attnSizes)
	attentionMLP.LinearOutput = true
	if err := attentionMLP.LoadWeights("attention_weights.json"); err == nil {
		if attentionMLP.HasLayerSizes(attnSizes) {
			fmt.Println("Loaded existing attention_weights.json...")
		} else {
			fmt.Println("Ignoring incompatible attention_weights.json (old architecture); starting Attention MLP from scratch...")
			attentionMLP = NewMLP(attnSizes)
			attentionMLP.LinearOutput = true
		}
	} else {
		fmt.Println("Starting Attention MLP from scratch...")
	}

	learningRate := 0.05
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

	const rapidLRDecay = 0.4       // aggressively decay LR each epoch
	const normalScheduleLR = 0.001 // switch to plateau schedule at this LR
	const patience = 3             // epochs without improvement before reducing LR
	const lrDecay = 0.5            // multiply LR by this on plateau
	const minLR = 1e-5             // floor to prevent vanishing updates
	const exploratorySamplesPerEpoch int64 = 4000
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
		explorationPhase := learningRate > normalScheduleLR
		explorationBudget := int64(0)
		if explorationPhase {
			explorationBudget = exploratorySamplesPerEpoch
			fmt.Printf("  -> Exploration epoch sample budget: %d\n", explorationBudget)
		}
		var samplesQueued int64
		reserveSample := func() bool {
			if explorationBudget <= 0 {
				return true
			}
			for {
				cur := atomic.LoadInt64(&samplesQueued)
				if cur >= explorationBudget {
					return false
				}
				if atomic.CompareAndSwapInt64(&samplesQueued, cur, cur+1) {
					return true
				}
			}
		}

		jobs := make(chan os.DirEntry, len(epochEntries))
		samples := make(chan preparedSnapshot, kernelBatchSize*8)
		statsCh := make(chan gpuEpochStats, 1)
		progressCh := make(chan gpuEpochStats, 1)

		go func() {
			lastReport := time.Now()
			for s := range progressCh {
				if time.Since(lastReport) < 5*time.Second {
					continue
				}
				avgLoss := 0.0
				if s.validSnapshots > 0 {
					avgLoss = s.totalLoss / float64(s.validSnapshots)
				}
				if math.IsNaN(avgLoss) || math.IsInf(avgLoss, 0) {
					avgLoss = 0.0
				}
				elapsed := time.Since(epochStart)
				rate := float64(s.validSnapshots) / elapsed.Seconds()
				fmt.Printf("\r  -> Progress: %d snapshots | Loss: %.6f | W/L: E:%.1f%% M:%.1f%% L:%.1f%% | Speed: %.0f/s        ",
					s.validSnapshots, avgLoss, s.PhaseAccuracy[0]*100, s.PhaseAccuracy[1]*100, s.PhaseAccuracy[2]*100, rate)
				lastReport = time.Now()
			}
		}()

		go func() {
			stats := runGPUEpochTrainer(mlp, attentionMLP, kernelBatchSize, learningRate, samples, progressCh)
			close(progressCh)
			statsCh <- stats
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
					if explorationBudget > 0 && atomic.LoadInt64(&samplesQueued) >= explorationBudget {
						break
					}
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

						gameID := atomic.AddUint64(&globalGameCounter, 1)
						replaySamples, replayStats := buildReplayChosenActionSamples(replay, matchWinner, eloWeight, gameID)
						localStats.merge(replayStats)
						for _, sample := range replaySamples {
							if !reserveSample() {
								break
							}
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
		parserWG.Wait()

		stats := <-statsCh
		fmt.Printf("\n") // Clear progress line
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

		validSnapshots := stats.validSnapshots
		avgBCE := math.MaxFloat64
		avgLoss := math.MaxFloat64
		if validSnapshots > 0 {
			avgBCE = stats.totalBCELoss / float64(validSnapshots)
			avgLoss = avgBCE
		}
		fmt.Printf("Epoch %d/%d - Average Loss: %.6f (BCE: %.6f, snapshots: %d, LR: %.6f)\n",
			epoch, epochs, avgLoss, avgBCE, validSnapshots, learningRate)
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

		if learningRate > normalScheduleLR {
			// Exploration phase: force fast decay to quickly find a good LR basin.
			nextLR := learningRate * rapidLRDecay
			if nextLR < normalScheduleLR {
				nextLR = normalScheduleLR
			}
			if nextLR != learningRate {
				fmt.Printf("  -> Rapid LR decay: %.6f -> %.6f\n", learningRate, nextLR)
			}
			if nextLR == normalScheduleLR {
				patienceCounter = 0
				fmt.Printf("  -> Switched to normal plateau LR schedule at %.6f\n", nextLR)
			}
			learningRate = nextLR
		} else if patienceCounter >= patience && learningRate > minLR {
			// Normal schedule: reduce LR only on plateau.
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
