package neuralv2

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/pokemon-engine/deepcfr"
	"github.com/pokemon-engine/simulator"
)

const (
	BackendDeepCFRJSON = "deepcfr-json"
	BackendONNX        = "onnx"
)

type SearchConfig struct {
	BeliefSamples   int
	OpponentSamples int
	Depth           int
	TimeBudget      time.Duration
	MaxSimulations  int
	TopK            int
}

type SearchResult struct {
	BestAction     int
	WinProbability float64
	ActionValues   map[int]float64
	ActionPolicy   map[int]float64
	Depth          int
	Simulations    int
	Latency        time.Duration
}

type Model struct {
	backend backend
}

type LoadConfig struct {
	Path    string
	Backend string
}

type backend interface {
	Name() string
	Evaluate(state *simulator.BattleState, cfg SearchConfig) SearchResult
	Close() error
}

func LoadModel(cfg LoadConfig) (*Model, error) {
	backendName := strings.ToLower(strings.TrimSpace(cfg.Backend))
	if backendName == "" {
		if strings.HasSuffix(strings.ToLower(cfg.Path), ".onnx") {
			backendName = BackendONNX
		} else {
			backendName = BackendDeepCFRJSON
		}
	}

	var b backend
	switch backendName {
	case BackendDeepCFRJSON:
		loaded, err := deepcfr.LoadModel(cfg.Path)
		if err != nil {
			return nil, fmt.Errorf("failed to load deepcfr-json model: %w", err)
		}
		b = &deepCFRBackend{model: loaded}
	case BackendONNX:
		onnxBackend, err := newONNXBackend(cfg.Path)
		if err != nil {
			return nil, err
		}
		b = onnxBackend
	default:
		return nil, fmt.Errorf("unknown neuralv2 backend %q", cfg.Backend)
	}

	return &Model{backend: b}, nil
}

func (m *Model) Close() error {
	if m == nil || m.backend == nil {
		return nil
	}
	return m.backend.Close()
}

func (m *Model) BackendName() string {
	if m == nil || m.backend == nil {
		return ""
	}
	return m.backend.Name()
}

func (m *Model) Evaluate(state *simulator.BattleState, cfg SearchConfig) SearchResult {
	if m == nil || m.backend == nil {
		return SearchResult{
			BestAction:     -1,
			WinProbability: 0.5,
			ActionValues:   map[int]float64{},
			ActionPolicy:   map[int]float64{},
			Depth:          cfg.Depth,
		}
	}
	cfg = normalizeConfig(cfg)
	start := time.Now()
	deadline := start.Add(cfg.TimeBudget)

	valueSum := map[int]float64{}
	policySum := map[int]float64{}
	bestActionCounts := map[int]int{}
	winProbSum := 0.0

	sims := 0
	for sims < cfg.MaxSimulations {
		if cfg.TimeBudget > 0 && sims > 0 && time.Now().After(deadline) {
			break
		}
		res := m.backend.Evaluate(state, cfg)
		winProbSum += res.WinProbability
		if res.BestAction >= 0 {
			bestActionCounts[res.BestAction]++
		}
		for action, val := range res.ActionValues {
			valueSum[action] += val
		}
		for action, prob := range res.ActionPolicy {
			policySum[action] += prob
		}
		sims++
	}

	if sims == 0 {
		sims = 1
		res := m.backend.Evaluate(state, cfg)
		winProbSum = res.WinProbability
		bestActionCounts[res.BestAction]++
		for action, val := range res.ActionValues {
			valueSum[action] = val
		}
		for action, prob := range res.ActionPolicy {
			policySum[action] = prob
		}
	}

	values := make(map[int]float64, len(valueSum))
	policy := make(map[int]float64, len(policySum))
	for action, v := range valueSum {
		values[action] = clamp01(v / float64(sims))
	}
	for action, p := range policySum {
		policy[action] = clamp01(p / float64(sims))
	}

	if cfg.TopK > 0 {
		values = keepTopK(values, cfg.TopK)
		policy = keepTopK(policy, cfg.TopK)
	}

	bestAction := selectBestAction(values, bestActionCounts)
	if bestAction == -1 {
		bestAction = firstLegalAction(state)
	}

	return SearchResult{
		BestAction:     bestAction,
		WinProbability: clamp01(winProbSum / float64(sims)),
		ActionValues:   values,
		ActionPolicy:   policy,
		Depth:          cfg.Depth,
		Simulations:    sims,
		Latency:        time.Since(start),
	}
}

func normalizeConfig(cfg SearchConfig) SearchConfig {
	if cfg.BeliefSamples <= 0 {
		cfg.BeliefSamples = 10
	}
	if cfg.OpponentSamples <= 0 {
		cfg.OpponentSamples = 3
	}
	if cfg.Depth <= 0 {
		cfg.Depth = 2
	}
	if cfg.MaxSimulations <= 0 {
		cfg.MaxSimulations = 96
	}
	if cfg.TimeBudget <= 0 {
		cfg.TimeBudget = 650 * time.Millisecond
	}
	if cfg.TopK < 0 {
		cfg.TopK = 0
	}
	return cfg
}

func keepTopK(values map[int]float64, k int) map[int]float64 {
	if k <= 0 || len(values) <= k {
		return values
	}
	type pair struct {
		action int
		value  float64
	}
	pairs := make([]pair, 0, len(values))
	for action, value := range values {
		pairs = append(pairs, pair{action: action, value: value})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].value == pairs[j].value {
			return pairs[i].action < pairs[j].action
		}
		return pairs[i].value > pairs[j].value
	})

	out := make(map[int]float64, k)
	for i := 0; i < k && i < len(pairs); i++ {
		out[pairs[i].action] = pairs[i].value
	}
	return out
}

func selectBestAction(values map[int]float64, vote map[int]int) int {
	bestAction := -1
	bestValue := math.Inf(-1)
	bestVotes := -1
	for action, value := range values {
		votes := vote[action]
		if value > bestValue || (value == bestValue && votes > bestVotes) || (value == bestValue && votes == bestVotes && (bestAction == -1 || action < bestAction)) {
			bestAction = action
			bestValue = value
			bestVotes = votes
		}
	}
	if bestAction != -1 {
		return bestAction
	}
	for action, votes := range vote {
		if votes > bestVotes || (votes == bestVotes && (bestAction == -1 || action < bestAction)) {
			bestAction = action
			bestVotes = votes
		}
	}
	return bestAction
}

func firstLegalAction(state *simulator.BattleState) int {
	if state == nil {
		return -1
	}
	actions, n := simulator.GetSearchActions(&state.P1)
	if n == 0 {
		return -1
	}
	return actions[0]
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

type deepCFRBackend struct {
	model *deepcfr.Model
}

func (b *deepCFRBackend) Name() string {
	return BackendDeepCFRJSON
}

func (b *deepCFRBackend) Close() error {
	return nil
}

func (b *deepCFRBackend) Evaluate(state *simulator.BattleState, cfg SearchConfig) SearchResult {
	engine := deepcfr.NewEngine(b.model, time.Now().UnixNano())
	res := engine.Evaluate(state, deepcfr.SearchConfig{
		BeliefSamples:   cfg.BeliefSamples,
		OpponentSamples: cfg.OpponentSamples,
		Depth:           cfg.Depth,
	})
	return SearchResult{
		BestAction:     res.BestAction,
		WinProbability: res.WinProbability,
		ActionValues:   res.ActionValues,
		ActionPolicy:   res.ActionPolicy,
		Depth:          res.Depth,
		Simulations:    1,
	}
}
