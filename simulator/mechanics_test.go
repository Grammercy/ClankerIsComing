package simulator

import (
	"path/filepath"
	"sync"
	"testing"

	"github.com/pokemon-engine/gamedata"
)

var loadGameDataOnce sync.Once

func ensureGameData(t *testing.T) {
	t.Helper()
	loadGameDataOnce.Do(func() {
		if err := gamedata.LoadPokedex(filepath.Join("..", "data", "pokedex.json")); err != nil {
			t.Fatalf("failed to load pokedex: %v", err)
		}
		if err := gamedata.LoadMovedex(filepath.Join("..", "data", "moves.json")); err != nil {
			t.Fatalf("failed to load movedex: %v", err)
		}
	})
}

func TestExecuteSpecificTurnAppliesStatusResidual(t *testing.T) {
	state := &BattleState{
		RNGState: 1,
		P1: PlayerState{
			ID:        "p1",
			TeamSize:  1,
			ActiveIdx: 0,
			Team: [6]PokemonState{
				{
					Species:  "Charizard",
					IsActive: true,
					MaxHP:    160,
					HP:       160,
					Status:   "brn",
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
					Species:  "Venusaur",
					IsActive: true,
					MaxHP:    160,
					HP:       160,
					Status:   "psn",
					Boosts:   NeutralBoosts,
				},
			},
		},
	}

	ExecuteSpecificTurn(state, -1, -1)

	if got := state.P1.Team[0].HP; got != 150 {
		t.Fatalf("expected burned pokemon HP to be 150, got %d", got)
	}
	if got := state.P2.Team[0].HP; got != 140 {
		t.Fatalf("expected poisoned pokemon HP to be 140, got %d", got)
	}
}

func TestApplyMoveDamageUsesAbsoluteHPDamage(t *testing.T) {
	ensureGameData(t)

	state := &BattleState{
		RNGState: 42,
		P1: PlayerState{
			ID:        "p1",
			TeamSize:  1,
			ActiveIdx: 0,
			Team: [6]PokemonState{
				{
					Species:  "Charizard",
					Level:    100,
					IsActive: true,
					MaxHP:    300,
					HP:       300,
					Stats: Stats{
						SpA: 320,
						Spe: 100,
					},
					Moves:    [4]string{"flamethrower"},
					NumMoves: 1,
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
					Species:  "Venusaur",
					Level:    100,
					IsActive: true,
					MaxHP:    600,
					HP:       600,
					Stats: Stats{
						SpD: 200,
						Spe: 90,
					},
					Boosts: NeutralBoosts,
				},
			},
		},
	}

	applyMoveDamage(state, &state.P1.Team[0], &state.P2.Team[0], 0, &state.P2.Side)

	damage := 600 - state.P2.Team[0].HP
	if damage < 80 {
		move := gamedata.LookupMove(state.P1.Team[0].Moves[0])
		acc, alwaysHit := getMoveAccuracy(move)
		hit := moveHits(state, &state.P1.Team[0], &state.P2.Team[0], move)
		t.Fatalf(
			"expected absolute damage (>=80) against high MaxHP target, got %d (p2HP=%d move=%q moveExists=%v acc=%v always=%v hit=%v move=%+v)",
			damage,
			state.P2.Team[0].HP,
			state.P1.Team[0].Moves[0],
			gamedata.LookupMove(state.P1.Team[0].Moves[0]) != nil,
			acc,
			alwaysHit,
			hit,
			move,
		)
	}
}

func TestExecuteSpecificTurnMoveActionDealsDamage(t *testing.T) {
	ensureGameData(t)

	state := &BattleState{
		RNGState: 7,
		P1: PlayerState{
			ID:        "p1",
			TeamSize:  1,
			ActiveIdx: 0,
			Team: [6]PokemonState{
				{
					Species:  "Charizard",
					Level:    100,
					IsActive: true,
					MaxHP:    300,
					HP:       300,
					Stats: Stats{
						SpA: 320,
						Spe: 100,
					},
					Moves:    [4]string{"flamethrower"},
					NumMoves: 1,
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
					Species:  "Venusaur",
					Level:    100,
					IsActive: true,
					MaxHP:    600,
					HP:       600,
					Stats: Stats{
						SpD: 200,
						Spe: 90,
					},
					Boosts: NeutralBoosts,
				},
			},
		},
	}

	ExecuteSpecificTurn(state, ActionMove1, -1)

	if state.P2.Team[0].HP >= 600 {
		t.Fatalf("expected move action to reduce defender HP, got %d", state.P2.Team[0].HP)
	}
}

func TestSpeedTieUsesRandomChance(t *testing.T) {
	ensureGameData(t)

	seenP1First := false
	seenP2First := false

	for seed := uint64(1); seed <= 40; seed++ {
		state := &BattleState{
			RNGState: seed,
			P1: PlayerState{
				ID:        "p1",
				TeamSize:  1,
				ActiveIdx: 0,
				Team: [6]PokemonState{
					{
						Species:  "Charizard",
						Level:    100,
						IsActive: true,
						MaxHP:    60,
						HP:       60,
						Stats: Stats{
							Atk: 500,
							Def: 50,
							Spe: 100,
						},
						Moves:    [4]string{"dummyhit"},
						NumMoves: 1,
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
						Species:  "Venusaur",
						Level:    100,
						IsActive: true,
						MaxHP:    60,
						HP:       60,
						Stats: Stats{
							Atk: 500,
							Def: 50,
							Spe: 100,
						},
						Moves:    [4]string{"dummyhit"},
						NumMoves: 1,
						Boosts:   NeutralBoosts,
					},
				},
			},
		}

		ExecuteSpecificTurn(state, ActionMove1, ActionMove1)
		if state.P1.ActiveIdx == -1 && state.P2.ActiveIdx != -1 {
			seenP2First = true
		}
		if state.P2.ActiveIdx == -1 && state.P1.ActiveIdx != -1 {
			seenP1First = true
		}
		if seenP1First && seenP2First {
			return
		}
	}

	t.Fatalf("expected both speed-tie outcomes across seeds, got p1First=%v p2First=%v", seenP1First, seenP2First)
}
