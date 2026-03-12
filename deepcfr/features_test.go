package deepcfr

import (
	"testing"

	"github.com/pokemon-engine/simulator"
)

func TestEncodeStateSizeAndMask(t *testing.T) {
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

	mask := buildLegalMask(state)
	features := encodeState(state, mask)
	mask32 := make([]float32, len(mask))
	for i := range mask {
		mask32[i] = float32(mask[i])
	}
	features32 := encodeState32(state, mask32)
	if len(features) != FeatureSize {
		t.Fatalf("expected feature size %d, got %d", FeatureSize, len(features))
	}
	if len(features32) != FeatureSize {
		t.Fatalf("expected float32 feature size %d, got %d", FeatureSize, len(features32))
	}
	if mask[simulator.ActionMove1] != 1 || mask[simulator.ActionMove2] != 1 {
		t.Fatalf("expected first two move actions to be legal, got %v", mask[:4])
	}
}
