package deepcfr

import (
	"math"
	"math/rand"
	"sort"

	"github.com/pokemon-engine/evaluator"
	"github.com/pokemon-engine/simulator"
)

type SearchConfig struct {
	BeliefSamples   int
	OpponentSamples int
	Depth           int
}

type SearchResult struct {
	BestAction     int
	WinProbability float64
	ActionValues   map[int]float64
	ActionPolicy   map[int]float64
	Depth          int
}

type Engine struct {
	Model *Model
	rng   *rand.Rand
}

func NewEngine(model *Model, seed int64) *Engine {
	if seed == 0 {
		seed = 1
	}
	return &Engine{
		Model: model,
		rng:   rand.New(rand.NewSource(seed)),
	}
}

func (e *Engine) Evaluate(state *simulator.BattleState, cfg SearchConfig) SearchResult {
	if cfg.BeliefSamples <= 0 {
		cfg.BeliefSamples = 8
	}
	if cfg.OpponentSamples <= 0 {
		cfg.OpponentSamples = 3
	}
	if cfg.Depth <= 0 {
		cfg.Depth = 1
	}
	if terminal, ok := terminalValue(state); ok {
		return SearchResult{
			BestAction:     -1,
			WinProbability: terminal,
			ActionValues:   map[int]float64{},
			ActionPolicy:   map[int]float64{},
			Depth:          cfg.Depth,
		}
	}

	legalActions, n := simulator.GetSearchActions(&state.P1)
	if n == 0 {
		return SearchResult{
			BestAction:     -1,
			WinProbability: e.leafValue(state),
			ActionValues:   map[int]float64{},
			ActionPolicy:   map[int]float64{},
			Depth:          cfg.Depth,
		}
	}

	mask := buildLegalMask(state)
	_, policy, _ := e.Model.Predict(encodeState(state, mask), mask)
	actionValues := make(map[int]float64, n)
	actionPolicy := make(map[int]float64, n)

	bestAction := legalActions[0]
	bestValue := math.Inf(-1)
	baseline := 0.0
	policyMass := 0.0

	for i := 0; i < n; i++ {
		action := legalActions[i]
		actionPolicy[action] = policy[action]
		value := 0.0
		for s := 0; s < cfg.BeliefSamples; s++ {
			sampled := simulator.CloneBattleState(state)
			if e.Model != nil && e.Model.Priors != nil {
				e.Model.Priors.CompleteState(sampled, e.rng)
			}
			value += e.actionValue(sampled, action, cfg.Depth, cfg.OpponentSamples)
		}
		value /= float64(cfg.BeliefSamples)
		actionValues[action] = clamp01(value)
		baseline += actionPolicy[action] * value
		policyMass += actionPolicy[action]
		if value > bestValue {
			bestValue = value
			bestAction = action
		}
	}

	if policyMass <= 0 {
		baseline = bestValue
	} else {
		baseline /= policyMass
	}

	return SearchResult{
		BestAction:     bestAction,
		WinProbability: clamp01((bestValue + baseline) / 2.0),
		ActionValues:   actionValues,
		ActionPolicy:   actionPolicy,
		Depth:          cfg.Depth,
	}
}

func (e *Engine) actionValue(state *simulator.BattleState, ourAction int, depth int, opponentSamples int) float64 {
	oppActions, oppLen := simulator.GetSearchActions(&state.P2)
	if oppLen == 0 {
		next := simulator.CloneBattleState(state)
		simulator.ExecuteSpecificTurn(next, ourAction, -1)
		return e.counterfactualValue(next, depth-1)
	}

	swapped := swapPerspective(state)
	oppMask := buildLegalMask(swapped)
	_, oppPolicy, _ := e.Model.Predict(encodeState(swapped, oppMask), oppMask)

	type weightedAction struct {
		action int
		weight float64
	}
	weighted := make([]weightedAction, 0, oppLen)
	for i := 0; i < oppLen; i++ {
		action := oppActions[i]
		weight := oppPolicy[action]
		if weight <= 0 {
			weight = 1.0 / float64(oppLen)
		}
		weighted = append(weighted, weightedAction{action: action, weight: weight})
	}
	sort.Slice(weighted, func(i, j int) bool {
		return weighted[i].weight > weighted[j].weight
	})
	if opponentSamples > 0 && opponentSamples < len(weighted) {
		weighted = weighted[:opponentSamples]
	}

	sum := 0.0
	weightSum := 0.0
	for _, choice := range weighted {
		next := simulator.CloneBattleState(state)
		simulator.ExecuteSpecificTurn(next, ourAction, choice.action)
		v := e.counterfactualValue(next, depth-1)
		sum += choice.weight * v
		weightSum += choice.weight
	}
	if weightSum == 0 {
		return e.leafValue(state)
	}
	return sum / weightSum
}

func (e *Engine) counterfactualValue(state *simulator.BattleState, depth int) float64 {
	if terminal, ok := terminalValue(state); ok {
		return terminal
	}
	if depth <= 0 {
		return e.leafValue(state)
	}

	actions, n := simulator.GetSearchActions(&state.P1)
	if n == 0 {
		return e.leafValue(state)
	}

	best := math.Inf(-1)
	limit := min(n, 3)
	mask := buildLegalMask(state)
	regrets, policy, _ := e.Model.Predict(encodeState(state, mask), mask)

	type ranked struct {
		action int
		score  float64
	}
	candidates := make([]ranked, 0, n)
	for i := 0; i < n; i++ {
		action := actions[i]
		candidates = append(candidates, ranked{
			action: action,
			score:  regrets[action] + policy[action],
		})
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	for i := 0; i < limit; i++ {
		value := e.actionValue(state, candidates[i].action, depth, 3)
		if value > best {
			best = value
		}
	}
	if math.IsInf(best, -1) {
		return e.leafValue(state)
	}
	return best
}

func (e *Engine) leafValue(state *simulator.BattleState) float64 {
	base := evaluator.EvaluateState(state)
	if e == nil || e.Model == nil {
		return base
	}
	mask := buildLegalMask(state)
	_, _, value := e.Model.Predict(encodeState(state, mask), mask)
	return clamp01((base + value) / 2.0)
}

func terminalValue(state *simulator.BattleState) (float64, bool) {
	p1Alive := countAlive(&state.P1)
	p2Alive := countAlive(&state.P2)
	switch {
	case p1Alive == 0 && p2Alive == 0:
		return 0.5, true
	case p1Alive == 0:
		return 0, true
	case p2Alive == 0:
		return 1, true
	default:
		return 0.5, false
	}
}

func countAlive(player *simulator.PlayerState) int {
	if player == nil {
		return 0
	}
	alive := 0
	for i := 0; i < player.TeamSize; i++ {
		if !player.Team[i].Fainted {
			alive++
		}
	}
	return alive
}

func swapPerspective(state *simulator.BattleState) *simulator.BattleState {
	cloned := simulator.CloneBattleState(state)
	cloned.P1, cloned.P2 = cloned.P2, cloned.P1
	cloned.P1.ID = "p1"
	cloned.P2.ID = "p2"
	return cloned
}
