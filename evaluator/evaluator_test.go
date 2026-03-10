package evaluator

import (
	"testing"

	"github.com/pokemon-engine/simulator"
)

func TestEvaluateStateTerminal(t *testing.T) {
	state := &simulator.BattleState{}
	state.P1.TeamSize = 1
	state.P2.TeamSize = 1
	state.P1.Team[0] = simulator.PokemonState{Fainted: false}
	state.P2.Team[0] = simulator.PokemonState{Fainted: false}

	if got := EvaluateState(state); got != 0.5 {
		t.Fatalf("expected neutral score for non-terminal state, got %v", got)
	}

	state.P2.Team[0].Fainted = true
	if got := EvaluateState(state); got != 1.0 {
		t.Fatalf("expected win score for P1 terminal state, got %v", got)
	}

	state.P1.Team[0].Fainted = true
	if got := EvaluateState(state); got != 0.5 {
		t.Fatalf("expected draw score when both are fainted, got %v", got)
	}
}

func TestEvaluateBatchStates(t *testing.T) {
	states := []simulator.BattleState{{}, {}}
	states[0].P1.TeamSize = 1
	states[0].P2.TeamSize = 1
	states[1].P1.TeamSize = 1
	states[1].P2.TeamSize = 1
	states[1].P2.Team[0].Fainted = true

	out := EvaluateBatchStates(states)
	if len(out) != 2 {
		t.Fatalf("expected 2 outputs, got %d", len(out))
	}
	if out[0] != 0.5 || out[1] != 1.0 {
		t.Fatalf("unexpected batch outputs: %v", out)
	}
}
