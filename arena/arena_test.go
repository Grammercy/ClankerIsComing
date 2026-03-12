package arena

import (
	"testing"

	"github.com/pokemon-engine/simulator"
)

type firstActionAgent struct {
	name string
}

func (a firstActionAgent) Name() string {
	return a.name
}

func (a firstActionAgent) Choose(state *simulator.BattleState) int {
	actions, n := simulator.GetSearchActions(&state.P1)
	if n == 0 {
		return -1
	}
	return actions[0]
}

func TestPlayFromStateProducesResult(t *testing.T) {
	state := simulator.BattleState{}
	state.P1.TeamSize = 1
	state.P1.ActiveIdx = 0
	state.P1.Team[0] = simulator.PokemonState{
		Species:  "Pikachu",
		Name:     "Pikachu",
		HP:       100,
		MaxHP:    100,
		IsActive: true,
		NumMoves: 1,
		Moves:    [4]string{"thunderbolt"},
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

	result := PlayFromState(state, firstActionAgent{name: "a"}, firstActionAgent{name: "b"}, Config{MaxTurns: 2})
	if result.Winner == "" {
		t.Fatal("expected non-empty winner")
	}
	if result.FinalValue < 0 || result.FinalValue > 1 {
		t.Fatalf("expected value in [0,1], got %.4f", result.FinalValue)
	}
}
