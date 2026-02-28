package bot

import (
	"testing"

	"github.com/pokemon-engine/evaluator"
	"github.com/pokemon-engine/simulator"
)

func TestMCTSSearch(t *testing.T) {
	// Setup a simple battle state
	state := &simulator.BattleState{
		P1: simulator.PlayerState{
			TeamSize: 1,
			Team: [6]simulator.PokemonState{
				{Species: "Pikachu", HP: 100, MaxHP: 100, Moves: [4]string{"Thunderbolt"}},
			},
			ActiveIdx: 0,
		},
		P2: simulator.PlayerState{
			TeamSize: 1,
			Team: [6]simulator.PokemonState{
				{Species: "Squirtle", HP: 100, MaxHP: 100, Moves: [4]string{"Water Gun"}},
			},
			ActiveIdx: 0,
		},
	}
	state.P1.Team[0].IsActive = true
	state.P2.Team[0].IsActive = true

	// Mock evaluator to avoid OpenCL dependency
	evaluator.GlobalMLP = nil

	// Run MCTS search
	// We expect it to fallback to baseEval (0.5) if MLP is nil, but still run simulations
	result := SearchBestMoveWithSims(state, 1, 10, nil, nil, nil)

	if result.NodesSearched == 0 {
		t.Errorf("Expected some nodes to be searched, got 0")
	}
}
