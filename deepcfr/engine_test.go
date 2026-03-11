package deepcfr

import (
	"testing"

	"github.com/pokemon-engine/simulator"
)

func TestEngineTerminalEvaluation(t *testing.T) {
	model := NewModel(1)
	engine := NewEngine(model, 1)
	state := &simulator.BattleState{}
	state.P1.TeamSize = 1
	state.P2.TeamSize = 1
	state.P2.Team[0].Fainted = true

	result := engine.Evaluate(state, SearchConfig{BeliefSamples: 1, OpponentSamples: 1, Depth: 1})
	if result.WinProbability < 0.9 {
		t.Fatalf("expected winning terminal position, got %.4f", result.WinProbability)
	}
}
