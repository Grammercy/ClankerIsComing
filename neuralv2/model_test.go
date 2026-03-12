package neuralv2

import (
	"testing"
	"time"

	"github.com/pokemon-engine/deepcfr"
	"github.com/pokemon-engine/simulator"
)

func TestEvaluateWithDeepCFRBackend(t *testing.T) {
	model := deepcfr.NewModel(1)
	state := &simulator.BattleState{}
	state.P1.TeamSize = 1
	state.P1.ActiveIdx = 0
	state.P1.Team[0] = simulator.PokemonState{
		Species:  "Pikachu",
		Name:     "Pikachu",
		HP:       100,
		MaxHP:    100,
		IsActive: true,
		NumMoves: 2,
		Moves:    [4]string{"thunderbolt", "voltswitch"},
		Boosts:   simulator.NeutralBoosts,
	}
	state.P2.TeamSize = 1
	state.P2.ActiveIdx = 0
	state.P2.Team[0] = simulator.PokemonState{
		Species:  "Bulbasaur",
		Name:     "Bulbasaur",
		HP:       100,
		MaxHP:    100,
		IsActive: true,
		NumMoves: 1,
		Moves:    [4]string{"sludgebomb"},
		Boosts:   simulator.NeutralBoosts,
	}

	loaded := &Model{backend: &deepCFRBackend{model: model}}
	result := loaded.Evaluate(state, SearchConfig{
		BeliefSamples:   1,
		OpponentSamples: 1,
		Depth:           1,
		MaxSimulations:  2,
		TimeBudget:      100 * time.Millisecond,
		TopK:            4,
	})

	if result.BestAction < 0 {
		t.Fatalf("expected valid action, got %d", result.BestAction)
	}
	if result.WinProbability < 0 || result.WinProbability > 1 {
		t.Fatalf("expected probability in [0,1], got %.4f", result.WinProbability)
	}
	if result.Simulations <= 0 {
		t.Fatalf("expected positive simulation count, got %d", result.Simulations)
	}
}

func TestLoadModelUnknownBackend(t *testing.T) {
	if _, err := LoadModel(LoadConfig{Path: "data/model.json", Backend: "bogus"}); err == nil {
		t.Fatal("expected error for unknown backend")
	}
}
