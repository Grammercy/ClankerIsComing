package arena

import (
	"github.com/pokemon-engine/evaluator"
	"github.com/pokemon-engine/simulator"
)

type Agent interface {
	Name() string
	Choose(state *simulator.BattleState) int
}

type Config struct {
	MaxTurns int
}

type TurnTrace struct {
	Turn     int
	P1Action int
	P2Action int
	Value    float64
}

type Result struct {
	Winner     string
	Turns      int
	FinalValue float64
	Trace      []TurnTrace
}

func PlayFromState(start simulator.BattleState, p1 Agent, p2 Agent, cfg Config) Result {
	if cfg.MaxTurns <= 0 {
		cfg.MaxTurns = 24
	}

	state := *simulator.CloneBattleState(&start)
	trace := make([]TurnTrace, 0, cfg.MaxTurns)

	for turn := 0; turn < cfg.MaxTurns; turn++ {
		if terminalWinner(&state) != "" {
			break
		}
		p1Action := sanitizeAction(&state, p1.Choose(&state))
		swapped := swapPerspective(&state)
		p2Action := sanitizeAction(swapped, p2.Choose(swapped))

		simulator.ExecuteSpecificTurn(&state, p1Action, p2Action)
		trace = append(trace, TurnTrace{
			Turn:     state.Turn,
			P1Action: p1Action,
			P2Action: p2Action,
			Value:    evaluator.EvaluateState(&state),
		})
	}

	finalValue := evaluator.EvaluateState(&state)
	winner := terminalWinner(&state)
	if winner == "" {
		if finalValue > 0.55 {
			winner = "p1"
		} else if finalValue < 0.45 {
			winner = "p2"
		} else {
			winner = "draw"
		}
	}

	return Result{
		Winner:     winner,
		Turns:      state.Turn,
		FinalValue: finalValue,
		Trace:      trace,
	}
}

func sanitizeAction(state *simulator.BattleState, action int) int {
	actions, n := simulator.GetSearchActions(&state.P1)
	if n == 0 {
		return -1
	}
	for i := 0; i < n; i++ {
		if actions[i] == action {
			return action
		}
	}
	return actions[0]
}

func swapPerspective(state *simulator.BattleState) *simulator.BattleState {
	cloned := simulator.CloneBattleState(state)
	cloned.P1, cloned.P2 = cloned.P2, cloned.P1
	cloned.P1.ID = "p1"
	cloned.P2.ID = "p2"
	return cloned
}

func terminalWinner(state *simulator.BattleState) string {
	p1Alive := countAlive(&state.P1)
	p2Alive := countAlive(&state.P2)
	switch {
	case p1Alive == 0 && p2Alive == 0:
		return "draw"
	case p1Alive == 0:
		return "p2"
	case p2Alive == 0:
		return "p1"
	default:
		return ""
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
