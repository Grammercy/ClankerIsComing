package simulator

import (
	"testing"

	"github.com/pokemon-engine/parser"
)

func TestSideConditionDurationDecrements(t *testing.T) {
	ensureGameData(t)

	state := &BattleState{
		P1: PlayerState{
			ID:        "p1",
			TeamSize:  1,
			ActiveIdx: 0,
			Team: [6]PokemonState{
				{Species: "Pikachu", IsActive: true, MaxHP: 100, HP: 100, Boosts: NeutralBoosts},
			},
			Side: SideConditions{
				ReflectTurns:  2,
				TailwindTurns: 1,
			},
		},
		P2: PlayerState{
			ID:        "p2",
			TeamSize:  1,
			ActiveIdx: 0,
			Team: [6]PokemonState{
				{Species: "Venusaur", IsActive: true, MaxHP: 100, HP: 100, Boosts: NeutralBoosts},
			},
		},
	}

	// Turn 1
	ExecuteSpecificTurn(state, -1, -1)
	if state.P1.Side.ReflectTurns != 1 {
		t.Errorf("expected ReflectTurns to be 1 after one turn, got %d", state.P1.Side.ReflectTurns)
	}
	if state.P1.Side.TailwindTurns != 0 {
		t.Errorf("expected TailwindTurns to be 0 after one turn, got %d", state.P1.Side.TailwindTurns)
	}

	// Turn 2
	ExecuteSpecificTurn(state, -1, -1)
	if state.P1.Side.ReflectTurns != 0 {
		t.Errorf("expected ReflectTurns to be 0 after two turns, got %d", state.P1.Side.ReflectTurns)
	}
}

func TestApplyEventSetsCorrectDurations(t *testing.T) {
	ensureGameData(t)

	state := &BattleState{
		P1: PlayerState{
			ID:        "p1",
			TeamSize:  1,
			ActiveIdx: 0,
			Team: [6]PokemonState{
				{Species: "Pikachu", IsActive: true, Item: "lightclay", Boosts: NeutralBoosts},
			},
		},
		P2: PlayerState{
			ID:        "p2",
			TeamSize:  1,
			ActiveIdx: 0,
			Team: [6]PokemonState{
				{Species: "Venusaur", IsActive: true, Boosts: NeutralBoosts},
			},
		},
	}

	// Reflect with Light Clay
	ApplyEvent(state, parser.Event{Type: "sidestart", Player: "p1", Value: "Reflect"})
	if state.P1.Side.ReflectTurns != 8 {
		t.Errorf("expected ReflectTurns to be 8 with Light Clay, got %d", state.P1.Side.ReflectTurns)
	}

	// Light Screen without Light Clay (p2)
	ApplyEvent(state, parser.Event{Type: "sidestart", Player: "p2", Value: "Light Screen"})
	if state.P2.Side.LightScreenTurns != 5 {
		t.Errorf("expected LightScreenTurns to be 5, got %d", state.P2.Side.LightScreenTurns)
	}

	// Tailwind
	ApplyEvent(state, parser.Event{Type: "sidestart", Player: "p1", Value: "Tailwind"})
	if state.P1.Side.TailwindTurns != 4 {
		t.Errorf("expected TailwindTurns to be 4, got %d", state.P1.Side.TailwindTurns)
	}
}
