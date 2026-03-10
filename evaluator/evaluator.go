package evaluator

import "github.com/pokemon-engine/simulator"

// EvaluateState returns a deterministic scalar from P1 perspective.
// Terminal states map to 0/0.5/1. Non-terminal states use a neutral fallback.
func EvaluateState(state *simulator.BattleState) float64 {
	if state == nil {
		return 0.5
	}

	p1Alive := countAlive(&state.P1)
	p2Alive := countAlive(&state.P2)
	switch {
	case p1Alive == 0 && p2Alive == 0:
		return 0.5
	case p1Alive == 0:
		return 0.0
	case p2Alive == 0:
		return 1.0
	default:
		return 0.5
	}
}

// EvaluateBatchStates evaluates many states in one call.
func EvaluateBatchStates(states []simulator.BattleState) []float64 {
	out := make([]float64, len(states))
	for i := range states {
		out[i] = EvaluateState(&states[i])
	}
	return out
}

func countAlive(player *simulator.PlayerState) int {
	alive := 0
	for i := 0; i < player.TeamSize; i++ {
		if !player.Team[i].Fainted {
			alive++
		}
	}
	return alive
}
