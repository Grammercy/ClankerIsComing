package deepcfr

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/pokemon-engine/parser"
	"github.com/pokemon-engine/simulator"
)

type TrainConfig struct {
	ReplayDir            string
	ModelPath            string
	Epochs               int
	MaxFiles             int
	Seed                 int64
	BeliefSamples        int
	OpponentSamples      int
	TargetDepth          int
	LearningRate         float64
	ReplayPolicyBlend    float64
	ChosenActionBonus    float64
	ProgressWriter       io.Writer
	ProgressInterval     time.Duration
	Accelerator          string
	BatchSize            int
	OpenCLPlatform       string
	OpenCLDevice         string
	TargetWorkers        int
	TargetPredictor      string
	TargetBatchSize      int
	TargetQueueSize      int
	SnapshotFilesPerSync int
	TrainProfilePath     string
	StopChan             <-chan struct{}
}

type TrainStats struct {
	Backend         string
	FilesSeen       int
	Positions       int
	AverageLoss     float64
	AverageValueMAE float64
	AveragePolicyCE float64
	Interrupted     bool
}

type replayPosition struct {
	State        simulator.BattleState
	ChosenAction int
	Outcome      float64
}

type trainProfile struct {
	StartedAt        time.Time          `json:"startedAt"`
	FinishedAt       time.Time          `json:"finishedAt"`
	DurationSec      float64            `json:"durationSec"`
	Positions        int                `json:"positions"`
	FilesSeen        int                `json:"filesSeen"`
	StagesSec        map[string]float64 `json:"stagesSec"`
	StagesPct        map[string]float64 `json:"stagesPct"`
	PosPerSec        float64            `json:"positionsPerSec"`
	FilePerSec       float64            `json:"filesPerSec"`
	PredictBatches   int64              `json:"predictBatches"`
	PredictStates    int64              `json:"predictStates"`
	PredictBatchSec  float64            `json:"predictBatchSec"`
	PredictFallbacks int64              `json:"predictFallbacks"`
}

type trainingProgress struct {
	writer         io.Writer
	interval       time.Duration
	start          time.Time
	totalEpochs    int
	totalFiles     int
	currentEpoch   atomic.Int64
	filesSeen      atomic.Int64
	attemptedFiles atomic.Int64
	parseErrors    atomic.Int64
	emptyFiles     atomic.Int64
	positions      atomic.Int64
	lossSum        atomic.Uint64
	valueErrorSum  atomic.Uint64
	policyCrossSum atomic.Uint64
	currentFile    atomic.Value
	stop           chan struct{}
	done           chan struct{}
}

func newTrainingProgress(cfg TrainConfig, totalFiles int) *trainingProgress {
	if cfg.ProgressWriter == nil {
		return nil
	}
	interval := cfg.ProgressInterval
	if interval <= 0 {
		interval = 10 * time.Second
	}
	p := &trainingProgress{
		writer:      cfg.ProgressWriter,
		interval:    interval,
		start:       time.Now(),
		totalEpochs: cfg.Epochs,
		totalFiles:  totalFiles * cfg.Epochs,
		stop:        make(chan struct{}),
		done:        make(chan struct{}),
	}
	p.currentFile.Store("")
	return p
}

func (p *trainingProgress) startLoop() {
	if p == nil {
		return
	}
	go func() {
		ticker := time.NewTicker(p.interval)
		defer ticker.Stop()
		lastWidth := 0
		for {
			select {
			case <-ticker.C:
				lastWidth = p.writeSnapshot(time.Now(), lastWidth)
			case <-p.stop:
				lastWidth = p.writeSnapshot(time.Now(), lastWidth)
				if lastWidth > 0 {
					fmt.Fprintln(p.writer)
				}
				close(p.done)
				return
			}
		}
	}()
}

func (p *trainingProgress) finish() {
	if p == nil {
		return
	}
	close(p.stop)
	<-p.done
}

func (p *trainingProgress) setEpoch(epoch int) {
	if p == nil {
		return
	}
	p.currentEpoch.Store(int64(epoch))
}

func (p *trainingProgress) setCurrentFile(name string) {
	if p == nil {
		return
	}
	p.currentFile.Store(name)
}

func (p *trainingProgress) recordAttempt() {
	if p == nil {
		return
	}
	p.attemptedFiles.Add(1)
}

func (p *trainingProgress) recordParseError() {
	if p == nil {
		return
	}
	p.parseErrors.Add(1)
}

func (p *trainingProgress) recordFileComplete(empty bool) {
	if p == nil {
		return
	}
	p.filesSeen.Add(1)
	if empty {
		p.emptyFiles.Add(1)
	}
}

func (p *trainingProgress) recordExample(metrics TrainingMetrics) {
	if p == nil {
		return
	}
	p.positions.Add(1)
	atomicAddFloat64(&p.lossSum, metrics.Loss)
	atomicAddFloat64(&p.valueErrorSum, metrics.ValueError)
	atomicAddFloat64(&p.policyCrossSum, metrics.PolicyCross)
}

func (p *trainingProgress) writeSnapshot(now time.Time, lastWidth int) int {
	if p == nil {
		return lastWidth
	}
	line := p.renderLine(now)
	padding := ""
	if lastWidth > len(line) {
		padding = strings.Repeat(" ", lastWidth-len(line))
	}
	fmt.Fprintf(p.writer, "\r%s%s", line, padding)
	if len(line) > lastWidth {
		return len(line)
	}
	return lastWidth
}

func (p *trainingProgress) renderLine(now time.Time) string {
	if p == nil {
		return ""
	}
	elapsed := now.Sub(p.start)
	if elapsed < 0 {
		elapsed = 0
	}
	epoch := int(p.currentEpoch.Load())
	if epoch <= 0 {
		epoch = 1
	}
	filesSeen := p.filesSeen.Load()
	attempted := p.attemptedFiles.Load()
	parseErrors := p.parseErrors.Load()
	emptyFiles := p.emptyFiles.Load()
	positions := p.positions.Load()

	lossAvg := 0.0
	valueAvg := 0.0
	policyAvg := 0.0
	if positions > 0 {
		lossAvg = atomicLoadFloat64(&p.lossSum) / float64(positions)
		valueAvg = atomicLoadFloat64(&p.valueErrorSum) / float64(positions)
		policyAvg = atomicLoadFloat64(&p.policyCrossSum) / float64(positions)
	}

	posRate := 0.0
	fileRate := 0.0
	if elapsed > 0 {
		seconds := elapsed.Seconds()
		posRate = float64(positions) / seconds
		fileRate = float64(filesSeen) / seconds
	}

	eta := "n/a"
	if p.totalFiles > 0 && fileRate > 0 {
		remaining := float64(p.totalFiles) - float64(filesSeen)
		if remaining < 0 {
			remaining = 0
		}
		eta = formatProgressDuration(time.Duration(remaining / fileRate * float64(time.Second)))
	}

	currentFile, _ := p.currentFile.Load().(string)
	currentFile = truncateProgressFile(currentFile, 32)

	return fmt.Sprintf(
		"train epoch %d/%d file %d/%d attempt %d err %d empty %d pos %d loss %.6f mae %.6f ce %.6f pos/s %.1f file/s %.2f elapsed %s eta %s current=%s",
		epoch,
		p.totalEpochs,
		filesSeen,
		p.totalFiles,
		attempted,
		parseErrors,
		emptyFiles,
		positions,
		lossAvg,
		valueAvg,
		policyAvg,
		posRate,
		fileRate,
		formatProgressDuration(elapsed),
		eta,
		currentFile,
	)
}

func truncateProgressFile(name string, maxLen int) string {
	if maxLen <= 0 {
		return ""
	}
	if name == "" {
		return "-"
	}
	if len(name) <= maxLen {
		return name
	}
	if maxLen <= 3 {
		return name[:maxLen]
	}
	return "..." + name[len(name)-maxLen+3:]
}

func formatProgressDuration(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	totalSeconds := int64(d.Round(time.Second) / time.Second)
	hours := totalSeconds / 3600
	minutes := (totalSeconds % 3600) / 60
	seconds := totalSeconds % 60
	if hours > 0 {
		return fmt.Sprintf("%dh%02dm%02ds", hours, minutes, seconds)
	}
	if minutes > 0 {
		return fmt.Sprintf("%dm%02ds", minutes, seconds)
	}
	return fmt.Sprintf("%ds", seconds)
}

func atomicAddFloat64(dst *atomic.Uint64, delta float64) {
	for {
		current := dst.Load()
		next := math.Float64bits(math.Float64frombits(current) + delta)
		if dst.CompareAndSwap(current, next) {
			return
		}
	}
}

func atomicLoadFloat64(src *atomic.Uint64) float64 {
	return math.Float64frombits(src.Load())
}

func TrainFromReplayDir(cfg TrainConfig) (*Model, TrainStats, error) {
	startedAt := time.Now()
	stageDurations := map[string]time.Duration{}
	addStage := func(name string, d time.Duration) {
		stageDurations[name] += d
	}

	if cfg.Epochs <= 0 {
		cfg.Epochs = 1
	}
	if cfg.BeliefSamples <= 0 {
		cfg.BeliefSamples = 6
	}
	if cfg.OpponentSamples <= 0 {
		cfg.OpponentSamples = 3
	}
	if cfg.TargetDepth <= 0 {
		cfg.TargetDepth = 1
	}
	if cfg.LearningRate <= 0 {
		cfg.LearningRate = 0.0005
	}
	if cfg.ReplayPolicyBlend <= 0 || cfg.ReplayPolicyBlend > 1 {
		cfg.ReplayPolicyBlend = 0.35
	}
	if cfg.ChosenActionBonus == 0 {
		cfg.ChosenActionBonus = 0.10
	}
	if cfg.TargetWorkers <= 0 {
		cfg.TargetWorkers = max(1, runtime.GOMAXPROCS(0)-1)
	}
	if cfg.TargetBatchSize <= 0 {
		cfg.TargetBatchSize = 2048
	}
	if cfg.TargetQueueSize <= 0 {
		cfg.TargetQueueSize = 8192
	}
	if cfg.SnapshotFilesPerSync <= 0 {
		cfg.SnapshotFilesPerSync = 16
	}

	if cfg.ProgressWriter != nil {
		fmt.Fprintln(cfg.ProgressWriter, "stage: building replay priors")
	}
	resetPredictorMetrics()
	priorsStart := time.Now()
	priors, _, err := BuildPriorsFromReplayDirWithProgress(cfg.ReplayDir, cfg.MaxFiles, cfg.ProgressWriter, cfg.ProgressInterval)
	addStage("build_priors", time.Since(priorsStart))
	if err != nil {
		return nil, TrainStats{}, err
	}
	model := NewModel(cfg.Seed)
	model.Priors = priors

	entries, err := os.ReadDir(cfg.ReplayDir)
	if err != nil {
		return nil, TrainStats{}, fmt.Errorf("failed to read replay dir: %w", err)
	}
	totalFiles := countReplayLogs(entries)
	if cfg.MaxFiles > 0 && totalFiles > cfg.MaxFiles {
		totalFiles = cfg.MaxFiles
	}
	progress := newTrainingProgress(cfg, totalFiles)
	hp := TrainingHyperParams{
		LearningRate:   cfg.LearningRate,
		RegretWeight:   1.0,
		StrategyWeight: 0.75,
		ValueWeight:    0.5,
	}
	trainer, trainerWarning, err := newExampleTrainer(model, hp, cfg)
	if err != nil {
		return nil, TrainStats{}, err
	}
	defer trainer.Close()
	stats := TrainStats{Backend: trainer.Name()}
	if progress != nil {
		fmt.Fprintf(cfg.ProgressWriter, "training backend: %s\n", trainer.Name())
		if trainerWarning != "" {
			fmt.Fprintf(cfg.ProgressWriter, "warning: %s\n", trainerWarning)
		}
		progress.startLoop()
		defer progress.finish()
	} else if cfg.ProgressWriter != nil && trainerWarning != "" {
		fmt.Fprintf(cfg.ProgressWriter, "warning: %s\n", trainerWarning)
	}
	recordMetrics := func(all []TrainingMetrics) {
		for _, metrics := range all {
			stats.Positions++
			stats.AverageLoss += metrics.Loss
			stats.AverageValueMAE += metrics.ValueError
			stats.AveragePolicyCE += metrics.PolicyCross
			if progress != nil {
				progress.recordExample(metrics)
			}
		}
	}
	shouldStop := func() bool {
		if cfg.StopChan == nil {
			return false
		}
		select {
		case <-cfg.StopChan:
			return true
		default:
			return false
		}
	}
	interrupted := false
	writeProfile := func(stats TrainStats) error {
		if strings.TrimSpace(cfg.TrainProfilePath) == "" {
			return nil
		}
		now := time.Now()
		total := now.Sub(startedAt)
		if total <= 0 {
			total = time.Nanosecond
		}
		predictMetrics := snapshotPredictorMetrics()
		stages := make(map[string]float64, len(stageDurations))
		stagePct := make(map[string]float64, len(stageDurations))
		for name, d := range stageDurations {
			stages[name] = d.Seconds()
			stagePct[name] = (d.Seconds() / total.Seconds()) * 100.0
		}
		profile := trainProfile{
			StartedAt:        startedAt,
			FinishedAt:       now,
			DurationSec:      total.Seconds(),
			Positions:        stats.Positions,
			FilesSeen:        stats.FilesSeen,
			StagesSec:        stages,
			StagesPct:        stagePct,
			PosPerSec:        float64(stats.Positions) / total.Seconds(),
			FilePerSec:       float64(stats.FilesSeen) / total.Seconds(),
			PredictBatches:   predictMetrics.Batches,
			PredictStates:    predictMetrics.States,
			PredictBatchSec:  predictMetrics.Duration.Seconds(),
			PredictFallbacks: predictMetrics.Fallbacks,
		}
		encoded, err := json.MarshalIndent(profile, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to encode train profile: %w", err)
		}
		if err := os.WriteFile(cfg.TrainProfilePath, encoded, 0644); err != nil {
			return fmt.Errorf("failed to write train profile: %w", err)
		}
		return nil
	}

	var targetSnapshot *Model
	var targetPredictor statePredictor
	filesSinceSnapshot := 0
	var predictorWarningPrinted bool
	closeTargetPredictor := func() {
		if targetPredictor != nil {
			_ = targetPredictor.Close()
			targetPredictor = nil
		}
		targetSnapshot = nil
		filesSinceSnapshot = 0
	}
	defer closeTargetPredictor()

	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		filesSeen := 0
		if progress != nil {
			progress.setEpoch(epoch + 1)
		}
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].Name() < entries[j].Name()
		})
		for _, entry := range entries {
			if shouldStop() {
				interrupted = true
				break
			}
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".log") {
				continue
			}
			if cfg.MaxFiles > 0 && filesSeen >= cfg.MaxFiles {
				break
			}
			if progress != nil {
				progress.setCurrentFile(entry.Name())
				progress.recordAttempt()
			}

			parseStart := time.Now()
			replay, err := parser.ParseLogFile(cfg.ReplayDir + "/" + entry.Name())
			addStage("parse_replay", time.Since(parseStart))
			if err != nil {
				if progress != nil {
					progress.recordParseError()
				}
				continue
			}

			extractStart := time.Now()
			positions := extractReplayPositions(replay)
			addStage("extract_positions", time.Since(extractStart))
			if len(positions) == 0 {
				filesSeen++
				stats.FilesSeen++
				if progress != nil {
					progress.recordFileComplete(true)
				}
				continue
			}

			if targetSnapshot == nil || filesSinceSnapshot >= cfg.SnapshotFilesPerSync {
				closeTargetPredictor()
				targetSnapshot = model.Clone()
				predictor, warning, err := newStatePredictor(targetSnapshot, cfg)
				if err != nil {
					return nil, stats, err
				}
				targetPredictor = predictor
				if warning != "" && cfg.ProgressWriter != nil && !predictorWarningPrinted {
					fmt.Fprintf(cfg.ProgressWriter, "warning: %s\n", warning)
					predictorWarningPrinted = true
				}
			}

			targetStart := time.Now()
			examples, err := prepareTrainingExamples(positions, targetSnapshot, targetPredictor, cfg, epoch, filesSeen, cfg.Seed)
			addStage("prepare_examples", time.Since(targetStart))
			if err != nil {
				return nil, stats, err
			}
			for _, example := range examples {
				if shouldStop() {
					interrupted = true
					break
				}
				trainStepStart := time.Now()
				metrics, err := trainer.Train(example)
				addStage("trainer_train", time.Since(trainStepStart))
				if err != nil {
					return nil, stats, err
				}
				recordMetrics(metrics)
			}
			if interrupted {
				break
			}
			filesSinceSnapshot++
			filesSeen++
			stats.FilesSeen++
			if progress != nil {
				progress.recordFileComplete(false)
			}
		}
		flushStart := time.Now()
		metrics, err := trainer.Flush()
		addStage("trainer_flush", time.Since(flushStart))
		if err != nil {
			return nil, stats, err
		}
		recordMetrics(metrics)
		if interrupted {
			break
		}
		closeTargetPredictor()
	}
	flushStart := time.Now()
	metrics, err := trainer.Flush()
	addStage("trainer_flush", time.Since(flushStart))
	if err != nil {
		return nil, stats, err
	}
	recordMetrics(metrics)
	stats.Interrupted = interrupted
	if interrupted && cfg.ProgressWriter != nil {
		fmt.Fprintln(cfg.ProgressWriter, "training interrupted: checkpointing current model state")
	}

	if stats.Positions > 0 {
		stats.AverageLoss /= float64(stats.Positions)
		stats.AverageValueMAE /= float64(stats.Positions)
		stats.AveragePolicyCE /= float64(stats.Positions)
	}
	if cfg.ModelPath != "" {
		saveStart := time.Now()
		if err := model.Save(cfg.ModelPath); err != nil {
			return nil, stats, err
		}
		addStage("save_model", time.Since(saveStart))
	}
	if err := writeProfile(stats); err != nil {
		return nil, stats, err
	}
	return model, stats, nil
}

func prepareTrainingExamples(positions []replayPosition, model *Model, predictor statePredictor, cfg TrainConfig, epoch int, fileIndex int, seed int64) ([]TrainingExample, error) {
	if len(positions) == 0 {
		return nil, nil
	}
	if predictor == nil {
		predictor = &directModelPredictor{model: model}
	}
	if cfg.TargetWorkers <= 1 || len(positions) == 1 {
		engine := NewEngineWithPredictor(model, predictor, seed+17)
		rng := rand.New(rand.NewSource(seed))
		examples := make([]TrainingExample, 0, len(positions))
		for _, pos := range positions {
			example, ok := buildTrainingExample(pos, model, engine, rng, cfg)
			if !ok {
				continue
			}
			examples = append(examples, example)
		}
		return examples, nil
	}

	type task struct {
		index int
		pos   replayPosition
	}
	type result struct {
		index   int
		example TrainingExample
		ok      bool
	}

	workerCount := min(cfg.TargetWorkers, len(positions))
	queueCap := cfg.TargetQueueSize
	if queueCap <= 0 {
		queueCap = 8192
	}
	taskCap := min(queueCap, len(positions))
	if taskCap <= 0 {
		taskCap = workerCount
	}
	tasks := make(chan task, taskCap)
	results := make(chan result, len(positions))

	baseSeed := seed + int64(epoch)*1_000_003 + int64(fileIndex)*10_007
	for worker := 0; worker < workerCount; worker++ {
		workerSeed := baseSeed + int64(worker+1)*7919
		go func(localSeed int64) {
			engine := NewEngineWithPredictor(model, predictor, localSeed+17)
			rng := rand.New(rand.NewSource(localSeed))
			for job := range tasks {
				example, ok := buildTrainingExample(job.pos, model, engine, rng, cfg)
				results <- result{index: job.index, example: example, ok: ok}
			}
		}(workerSeed)
	}

	for index, pos := range positions {
		tasks <- task{index: index, pos: pos}
	}
	close(tasks)

	ordered := make([]*TrainingExample, len(positions))
	for range positions {
		res := <-results
		if !res.ok {
			continue
		}
		example := res.example
		ordered[res.index] = &example
	}

	examples := make([]TrainingExample, 0, len(positions))
	for _, example := range ordered {
		if example != nil {
			examples = append(examples, *example)
		}
	}
	return examples, nil
}

func buildTrainingExample(pos replayPosition, model *Model, engine *Engine, rng *rand.Rand, cfg TrainConfig) (TrainingExample, bool) {
	state := simulator.CloneBattleState(&pos.State)
	if model.Priors != nil {
		model.Priors.CompleteState(state, rng)
	}
	legalMask := buildLegalMask(state)
	if !actionLegal(legalMask, pos.ChosenAction) {
		return TrainingExample{}, false
	}

	regretTargets, strategyTargets := deriveTargets(engine, state, pos.ChosenAction, pos.Outcome, cfg)
	return TrainingExample{
		Features:        encodeState(state, legalMask),
		LegalMask:       legalMask,
		RegretTargets:   regretTargets,
		StrategyTargets: strategyTargets,
		ValueTarget:     pos.Outcome,
	}, true
}

func countReplayLogs(entries []os.DirEntry) int {
	total := 0
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".log") {
			continue
		}
		total++
	}
	return total
}

func deriveTargets(engine *Engine, state *simulator.BattleState, chosenAction int, outcome float64, cfg TrainConfig) ([simulator.MaxActions]float64, [simulator.MaxActions]float64) {
	var regrets [simulator.MaxActions]float64
	var strategy [simulator.MaxActions]float64

	actions, n := simulator.GetSearchActions(&state.P1)
	if n == 0 {
		return regrets, strategy
	}
	mask := buildLegalMask(state)
	_, predictedStrategy, _ := engine.predictor.Predict(encodeState(state, mask), mask)

	actionValues := make(map[int]float64, n)
	baseline := 0.0
	weightSum := 0.0
	for i := 0; i < n; i++ {
		action := actions[i]
		value := 0.0
		for s := 0; s < cfg.BeliefSamples; s++ {
			sampled := simulator.CloneBattleState(state)
			if engine.Model.Priors != nil {
				engine.Model.Priors.CompleteState(sampled, engine.rng)
			}
			value += engine.actionValue(sampled, action, cfg.TargetDepth, cfg.OpponentSamples)
		}
		value /= float64(cfg.BeliefSamples)
		actionValues[action] = value
		weight := predictedStrategy[action]
		if weight <= 0 {
			weight = 1.0 / float64(n)
		}
		baseline += weight * value
		weightSum += weight
	}
	if weightSum > 0 {
		baseline /= weightSum
	}

	for action, value := range actionValues {
		regrets[action] = value - baseline
	}
	if chosenAction >= 0 && chosenAction < simulator.MaxActions {
		regrets[chosenAction] += cfg.ChosenActionBonus + (outcome-0.5)*0.25
	}

	totalPositive := 0.0
	for i := 0; i < simulator.MaxActions; i++ {
		if mask[i] > 0 && regrets[i] > 0 {
			totalPositive += regrets[i]
		}
	}
	if totalPositive > 0 {
		for i := 0; i < simulator.MaxActions; i++ {
			if mask[i] > 0 && regrets[i] > 0 {
				strategy[i] = regrets[i] / totalPositive
			}
		}
	} else {
		uniform := 1.0 / float64(n)
		for i := 0; i < n; i++ {
			strategy[actions[i]] = uniform
		}
	}

	if chosenAction >= 0 && chosenAction < simulator.MaxActions {
		for i := 0; i < simulator.MaxActions; i++ {
			strategy[i] *= (1.0 - cfg.ReplayPolicyBlend)
		}
		strategy[chosenAction] += cfg.ReplayPolicyBlend
	}
	return regrets, strategy
}

func extractReplayPositions(replay *parser.Replay) []replayPosition {
	state := newObservedState()
	positions := make([]replayPosition, 0, len(replay.Events)/2)

	for idx, event := range replay.Events {
		if isDecisionEvent(replay.Events, idx) {
			canonical := canonicalizePerspective(state, event.Player)
			ensureUnknownSlots(&canonical.P1)
			ensureUnknownSlots(&canonical.P2)
			chosen, ok := mapReplayEventChosenActionCanonical(replay.Events, &canonical, idx, event.Player)
			if ok {
				outcome := 0.0
				winner := replay.Winner
				if event.Player == "p1" && winner == replay.P1 {
					outcome = 1
				}
				if event.Player == "p2" && winner == replay.P2 {
					outcome = 1
				}
				positions = append(positions, replayPosition{
					State:        canonical,
					ChosenAction: chosen,
					Outcome:      outcome,
				})
			}
		}
		simulator.ApplyEvent(state, event)
	}

	return positions
}

func DecisionStateForTurn(replay *parser.Replay, targetTurn int, player string) (*simulator.BattleState, int, bool) {
	state := newObservedState()
	for idx, event := range replay.Events {
		if event.Turn > targetTurn {
			break
		}
		if event.Turn == targetTurn && event.Player == player && isDecisionEvent(replay.Events, idx) {
			canonical := canonicalizePerspective(state, player)
			ensureUnknownSlots(&canonical.P1)
			ensureUnknownSlots(&canonical.P2)
			chosen, ok := mapReplayEventChosenActionCanonical(replay.Events, &canonical, idx, player)
			return &canonical, chosen, ok
		}
		simulator.ApplyEvent(state, event)
	}
	canonical := canonicalizePerspective(state, player)
	ensureUnknownSlots(&canonical.P1)
	ensureUnknownSlots(&canonical.P2)
	return &canonical, -1, false
}

func newObservedState() *simulator.BattleState {
	return &simulator.BattleState{
		P1: simulator.PlayerState{
			ID:              "p1",
			ActiveIdx:       -1,
			CanTerastallize: true,
		},
		P2: simulator.PlayerState{
			ID:              "p2",
			ActiveIdx:       -1,
			CanTerastallize: true,
		},
	}
}

func canonicalizePerspective(state *simulator.BattleState, player string) simulator.BattleState {
	cloned := *simulator.CloneBattleState(state)
	if player == "p2" {
		cloned.P1, cloned.P2 = cloned.P2, cloned.P1
	}
	cloned.P1.ID = "p1"
	cloned.P2.ID = "p2"
	return cloned
}

func ensureUnknownSlots(player *simulator.PlayerState) {
	if player == nil {
		return
	}
	if player.TeamSize < 6 {
		player.TeamSize = 6
	}
	for i := 0; i < player.TeamSize; i++ {
		if player.Team[i].Species == "" {
			player.Team[i] = simulator.PokemonState{
				Name:     "Unknown",
				Species:  "Unknown",
				HP:       100,
				MaxHP:    100,
				Boosts:   simulator.NeutralBoosts,
				IsActive: i == player.ActiveIdx,
			}
		}
		if player.Team[i].Boosts == 0 {
			player.Team[i].Boosts = simulator.NeutralBoosts
		}
		player.Team[i].IsActive = (i == player.ActiveIdx)
	}
}

func isDecisionEvent(events []parser.Event, idx int) bool {
	if idx < 0 || idx >= len(events) {
		return false
	}
	event := events[idx]
	if event.Turn == 0 {
		return false
	}
	switch event.Type {
	case "switch":
		return true
	case "move":
		return !hasPriorTerastallize(events, idx, event.Player)
	case "terastallize":
		return true
	default:
		return false
	}
}

func hasPriorTerastallize(events []parser.Event, idx int, player string) bool {
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

func mapReplayEventChosenActionCanonical(events []parser.Event, state *simulator.BattleState, idx int, player string) (int, bool) {
	if idx < 0 || idx >= len(events) {
		return -1, false
	}
	event := events[idx]
	switch event.Type {
	case "switch":
		return mapReplaySwitchAction(state, event)
	case "move":
		slot, ok := mapReplayMoveSlot(state, event.Value)
		if !ok {
			return -1, false
		}
		return simulator.ActionMove1 + slot, true
	case "terastallize":
		moveIdx := findNextMoveInTurn(events, idx, player)
		if moveIdx == -1 {
			return -1, false
		}
		slot, ok := mapReplayMoveSlot(state, events[moveIdx].Value)
		if !ok {
			return -1, false
		}
		return simulator.ActionTeraMove1 + slot, true
	default:
		return -1, false
	}
}

func mapReplaySwitchAction(state *simulator.BattleState, event parser.Event) (int, bool) {
	species := strings.TrimSpace(event.Value)
	if species == "" {
		return -1, false
	}
	for i := 0; i < state.P1.TeamSize; i++ {
		if strings.EqualFold(state.P1.Team[i].Species, species) || strings.EqualFold(state.P1.Team[i].Name, species) {
			return simulator.ActionSwitchBase + i, true
		}
	}
	for i := 0; i < state.P1.TeamSize; i++ {
		if !state.P1.Team[i].IsActive && (state.P1.Team[i].Species == "" || strings.EqualFold(state.P1.Team[i].Species, "Unknown")) {
			return simulator.ActionSwitchBase + i, true
		}
	}
	return -1, false
}

func mapReplayMoveSlot(state *simulator.BattleState, moveName string) (int, bool) {
	active := state.P1.GetActive()
	if active == nil || active.Fainted {
		return -1, false
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
			return i, true
		}
	}
	if known < 4 && needle != "" {
		return known, true
	}
	return -1, false
}

func normalizeReplayMoveName(move string) string {
	s := strings.ToLower(move)
	replacer := strings.NewReplacer(" ", "", "-", "", "'", "", ".", "", ":", "")
	return replacer.Replace(s)
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

func actionLegal(mask []float64, action int) bool {
	return action >= 0 && action < len(mask) && mask[action] > 0
}

func regretMatching(regrets [simulator.MaxActions]float64, mask []float64) [simulator.MaxActions]float64 {
	var out [simulator.MaxActions]float64
	total := 0.0
	valid := 0
	for i := 0; i < simulator.MaxActions; i++ {
		if i >= len(mask) || mask[i] == 0 {
			continue
		}
		valid++
		if regrets[i] > 0 {
			out[i] = regrets[i]
			total += regrets[i]
		}
	}
	if total > 0 {
		for i := 0; i < simulator.MaxActions; i++ {
			out[i] /= total
		}
		return out
	}
	if valid == 0 {
		return out
	}
	uniform := 1.0 / float64(valid)
	for i := 0; i < simulator.MaxActions; i++ {
		if i < len(mask) && mask[i] > 0 {
			out[i] = uniform
		}
	}
	return out
}

func stableAverage(current float64, count int, next float64) float64 {
	if count <= 0 {
		return next
	}
	return current + (next-current)/float64(count)
}

func positivePart(v float64) float64 {
	return math.Max(0, v)
}
