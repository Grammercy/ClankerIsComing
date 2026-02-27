package simulator

import (
	"testing"
)

func TestEndOfTurnSpeedOrder(t *testing.T) {
	// P1 is Slow (Spe 50), P2 is Fast (Spe 100).
	// Both are at 1 HP and Poisoned.
	// P1 has Leftovers.

	// Expectations:
	// 1. P2 (Fast) takes Poison damage first -> HP 0.
	// 2. P1 (Slow) takes Poison damage -> HP 0.
	// 3. Since both HP are 0, Leftovers should NOT trigger for either.

	state := &BattleState{
		P1: PlayerState{
			ID:        "p1",
			TeamSize:  1,
			ActiveIdx: 0,
			Team: [6]PokemonState{
				{
					Species:  "Slowpoke",
					IsActive: true,
					MaxHP:    160,
					HP:       1,
					Status:   "psn",
					Item:     "Leftovers",
					Stats:    Stats{Spe: 50},
					Boosts:   NeutralBoosts,
				},
			},
		},
		P2: PlayerState{
			ID:        "p2",
			TeamSize:  1,
			ActiveIdx: 0,
			Team: [6]PokemonState{
				{
					Species:  "Ninjask",
					IsActive: true,
					MaxHP:    160,
					HP:       1,
					Status:   "psn",
					Stats:    Stats{Spe: 100},
					Boosts:   NeutralBoosts,
				},
			},
		},
	}

	ExecuteSpecificTurn(state, -1, -1)

	// Currently, our simulator will:
	// 1. applyStatusResidual(P1) -> HP 0
	// 2. applyStatusResidual(P2) -> HP 0
	// 3. applyItemResidual(P1) -> HEALS P1 because HP 0 is not checked!
	// 4. markActiveFainted(P1), markActiveFainted(P2)

	if state.P1.Team[0].HP > 0 {
		t.Errorf("expected Slowpoke (P1) to stay at 0 HP after Poison, but found heal to %d", state.P1.Team[0].HP)
	}
	if state.P2.Team[0].HP > 0 {
		t.Errorf("expected Ninjask (P2) to stay at 0 HP after Poison, but found heal to %d", state.P2.Team[0].HP)
	}
}
