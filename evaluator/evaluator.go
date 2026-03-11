package evaluator

import "github.com/pokemon-engine/simulator"

// EvaluateState returns a deterministic scalar from P1 perspective.
// Terminal states map to 0/0.5/1. Non-terminal states use a lightweight
// material-and-board heuristic centered at 0.5.
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
		p1Score := teamScore(&state.P1, &state.P2)
		p2Score := teamScore(&state.P2, &state.P1)
		score := 0.5 + (p1Score-p2Score)*0.5
		if score < 0.01 {
			return 0.01
		}
		if score > 0.99 {
			return 0.99
		}
		return score
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

func teamScore(player *simulator.PlayerState, opponent *simulator.PlayerState) float64 {
	if player == nil {
		return 0
	}

	aliveRatio := float64(countAlive(player)) / 6.0
	hpTotal := 0.0
	statusPenalty := 0.0
	for i := 0; i < player.TeamSize; i++ {
		poke := &player.Team[i]
		if poke.MaxHP > 0 {
			hpTotal += float64(max(poke.HP, 0)) / float64(poke.MaxHP)
		}
		switch poke.Status {
		case "brn", "par", "psn":
			statusPenalty += 0.03
		case "tox", "slp", "frz":
			statusPenalty += 0.05
		}
	}
	hpRatio := hpTotal / 6.0

	activeBoard := 0.0
	if active := player.GetActive(); active != nil {
		activeBoard += float64(active.GetBoost(simulator.AtkShift)+active.GetBoost(simulator.SpaShift)) * 0.01
		activeBoard += float64(active.GetBoost(simulator.DefShift)+active.GetBoost(simulator.SpdShift)+active.GetBoost(simulator.SpeShift)) * 0.007
		if active.Terastallized {
			activeBoard += 0.02
		}
	}
	if opponent != nil {
		hazardAdv := 0.0
		if opponent.Side.StealthRock {
			hazardAdv += 0.03
		}
		hazardAdv += float64(opponent.Side.Spikes) * 0.015
		hazardAdv += float64(opponent.Side.ToxicSpikes) * 0.012
		if opponent.Side.StickyWeb {
			hazardAdv += 0.015
		}
		if player.Side.StealthRock {
			hazardAdv -= 0.03
		}
		hazardAdv -= float64(player.Side.Spikes) * 0.015
		hazardAdv -= float64(player.Side.ToxicSpikes) * 0.012
		if player.Side.StickyWeb {
			hazardAdv -= 0.015
		}
		activeBoard += hazardAdv
	}

	score := aliveRatio*0.55 + hpRatio*0.35 + activeBoard - statusPenalty
	if score < 0 {
		return 0
	}
	if score > 1 {
		return 1
	}
	return score
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
