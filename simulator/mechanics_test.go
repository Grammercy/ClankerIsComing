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

	applyMoveDamage(state, &state.P1, &state.P2, 0)

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

func TestPriorityMoveActsBeforeFasterPokemon(t *testing.T) {
	ensureGameData(t)

	state := &BattleState{
		RNGState: 5,
		P1:       PlayerState{ID: "p1", TeamSize: 1, ActiveIdx: 0},
		P2:       PlayerState{ID: "p2", TeamSize: 1, ActiveIdx: 0},
	}
	state.P1.Team[0] = PokemonState{
		Species:  "Scizor",
		Level:    100,
		IsActive: true,
		MaxHP:    40,
		HP:       40,
		Stats:    Stats{Atk: 500, Def: 60, Spe: 30},
		Moves:    [4]string{"quickattack"},
		NumMoves: 1,
		Boosts:   NeutralBoosts,
	}
	state.P2.Team[0] = PokemonState{
		Species:  "Jolteon",
		Level:    100,
		IsActive: true,
		MaxHP:    40,
		HP:       40,
		Stats:    Stats{Atk: 500, Def: 50, Spe: 200},
		Moves:    [4]string{"tackle"},
		NumMoves: 1,
		Boosts:   NeutralBoosts,
	}

	ExecuteSpecificTurn(state, ActionMove1, ActionMove1)
	if state.P2.ActiveIdx != -1 {
		t.Fatalf("expected faster target to faint to priority move first, got p2 active idx %d", state.P2.ActiveIdx)
	}
}

func TestTrickRoomReversesMoveOrder(t *testing.T) {
	ensureGameData(t)

	state := &BattleState{
		RNGState: 9,
		Field:    FieldConditions{TrickRoom: true},
		P1:       PlayerState{ID: "p1", TeamSize: 1, ActiveIdx: 0},
		P2:       PlayerState{ID: "p2", TeamSize: 1, ActiveIdx: 0},
	}
	state.P1.Team[0] = PokemonState{
		Species:  "Snorlax",
		Level:    100,
		IsActive: true,
		MaxHP:    40,
		HP:       40,
		Stats:    Stats{Atk: 500, Def: 60, Spe: 20},
		Moves:    [4]string{"tackle"},
		NumMoves: 1,
		Boosts:   NeutralBoosts,
	}
	state.P2.Team[0] = PokemonState{
		Species:  "Alakazam",
		Level:    100,
		IsActive: true,
		MaxHP:    40,
		HP:       40,
		Stats:    Stats{Atk: 500, Def: 50, Spe: 200},
		Moves:    [4]string{"tackle"},
		NumMoves: 1,
		Boosts:   NeutralBoosts,
	}

	ExecuteSpecificTurn(state, ActionMove1, ActionMove1)
	if state.P2.ActiveIdx != -1 {
		t.Fatalf("expected slower attacker to move first under Trick Room and KO target")
	}
}

func TestToxicCounterRampsAndResetsOnSwitch(t *testing.T) {
	state := &BattleState{
		P1: PlayerState{ID: "p1", TeamSize: 2, ActiveIdx: 0},
		P2: PlayerState{ID: "p2", TeamSize: 1, ActiveIdx: 0},
	}
	state.P1.Team[0] = PokemonState{
		Species:  "Venusaur",
		IsActive: true,
		MaxHP:    160,
		HP:       160,
		Status:   "tox",
		Boosts:   NeutralBoosts,
	}
	state.P1.Team[1] = PokemonState{
		Species:  "Blastoise",
		IsActive: false,
		MaxHP:    160,
		HP:       160,
		Boosts:   NeutralBoosts,
	}
	state.P2.Team[0] = PokemonState{
		Species:  "Charizard",
		IsActive: true,
		MaxHP:    160,
		HP:       160,
		Boosts:   NeutralBoosts,
	}

	ExecuteSpecificTurn(state, -1, -1)
	ExecuteSpecificTurn(state, -1, -1)
	if got := state.P1.Team[0].HP; got != 130 {
		t.Fatalf("expected toxic ramp damage to reach 130 HP, got %d", got)
	}

	ExecuteSpecificTurn(state, ActionSwitchBase+1, -1)
	if state.P1.Team[0].ToxicCounter != 0 {
		t.Fatalf("expected toxic counter reset on switch out, got %d", state.P1.Team[0].ToxicCounter)
	}

	ExecuteSpecificTurn(state, ActionSwitchBase+0, -1)
	if got := state.P1.Team[0].HP; got != 120 {
		t.Fatalf("expected toxic to restart at 1/16 after re-entry, got HP=%d", got)
	}
}

func TestSleepCounterBlocksThenWakes(t *testing.T) {
	ensureGameData(t)

	state := &BattleState{
		RNGState: 11,
		P1:       PlayerState{ID: "p1", TeamSize: 1, ActiveIdx: 0},
		P2:       PlayerState{ID: "p2", TeamSize: 1, ActiveIdx: 0},
	}
	state.P1.Team[0] = PokemonState{
		Species:    "Breloom",
		Level:      100,
		IsActive:   true,
		MaxHP:      160,
		HP:         160,
		Status:     "slp",
		SleepTurns: 2,
		Stats:      Stats{Atk: 300, Def: 120, Spe: 80},
		Moves:      [4]string{"tackle"},
		NumMoves:   1,
		Boosts:     NeutralBoosts,
	}
	state.P2.Team[0] = PokemonState{
		Species:  "Pikachu",
		Level:    100,
		IsActive: true,
		MaxHP:    160,
		HP:       160,
		Stats:    Stats{Def: 100, Spe: 70},
		Boosts:   NeutralBoosts,
	}

	ExecuteSpecificTurn(state, ActionMove1, -1)
	if state.P2.Team[0].HP != 160 || state.P1.Team[0].SleepTurns != 1 {
		t.Fatalf("expected first sleep turn to block action; hp=%d sleep=%d", state.P2.Team[0].HP, state.P1.Team[0].SleepTurns)
	}

	ExecuteSpecificTurn(state, ActionMove1, -1)
	if state.P2.Team[0].HP != 160 || state.P1.Team[0].Status != "" {
		t.Fatalf("expected second sleep turn to consume final sleep and wake next turn; hp=%d status=%q", state.P2.Team[0].HP, state.P1.Team[0].Status)
	}

	ExecuteSpecificTurn(state, ActionMove1, -1)
	if state.P2.Team[0].HP >= 160 {
		t.Fatalf("expected woken pokemon to attack on third turn")
	}
}

func TestMoveHitCountLoadedDicePrefersFourToFiveHits(t *testing.T) {
	ensureGameData(t)

	move := gamedata.LookupMove("armthrust")
	if move == nil {
		t.Fatalf("expected armthrust to exist in movedex")
	}
	attacker := &PokemonState{
		Species: "Breloom",
		Item:    "Loaded Dice",
	}

	seen4 := false
	seen5 := false
	for seed := uint64(1); seed <= 20; seed++ {
		state := &BattleState{RNGState: seed}
		hits := moveHitCount(state, attacker, "armthrust", move)
		if hits < 4 || hits > 5 {
			t.Fatalf("expected loaded dice to force 4-5 hits, got %d", hits)
		}
		if hits == 4 {
			seen4 = true
		}
		if hits == 5 {
			seen5 = true
		}
	}
	if !seen4 || !seen5 {
		t.Fatalf("expected to observe both 4-hit and 5-hit rolls, got seen4=%v seen5=%v", seen4, seen5)
	}
}

func TestElectricTerrainPreventsSleepStatus(t *testing.T) {
	ensureGameData(t)

	state := &BattleState{
		RNGState: 13,
		Field:    FieldConditions{Terrain: "electricterrain"},
		P1:       PlayerState{ID: "p1", TeamSize: 1, ActiveIdx: 0},
		P2:       PlayerState{ID: "p2", TeamSize: 1, ActiveIdx: 0},
	}
	state.P1.Team[0] = PokemonState{
		Species:  "Amoonguss",
		Level:    100,
		IsActive: true,
		MaxHP:    160,
		HP:       160,
		Stats:    Stats{Spe: 30},
		Moves:    [4]string{"spore"},
		NumMoves: 1,
		Boosts:   NeutralBoosts,
	}
	state.P2.Team[0] = PokemonState{
		Species:  "Pikachu",
		Level:    100,
		IsActive: true,
		MaxHP:    160,
		HP:       160,
		Stats:    Stats{Spe: 20},
		Boosts:   NeutralBoosts,
	}

	ExecuteSpecificTurn(state, ActionMove1, -1)
	if state.P2.Team[0].Status == "slp" {
		t.Fatalf("expected electric terrain to block sleep on grounded target")
	}
}

func TestRoarForceSwitchesTarget(t *testing.T) {
	ensureGameData(t)

	state := &BattleState{
		RNGState: 21,
		P1:       PlayerState{ID: "p1", TeamSize: 1, ActiveIdx: 0},
		P2:       PlayerState{ID: "p2", TeamSize: 2, ActiveIdx: 0},
	}
	state.P1.Team[0] = PokemonState{
		Species:  "Arcanine",
		Level:    100,
		IsActive: true,
		MaxHP:    180,
		HP:       180,
		Moves:    [4]string{"roar"},
		NumMoves: 1,
		Boosts:   NeutralBoosts,
	}
	state.P2.Team[0] = PokemonState{
		Species:  "Garchomp",
		Level:    100,
		IsActive: true,
		MaxHP:    180,
		HP:       180,
		Boosts:   NeutralBoosts,
	}
	state.P2.Team[1] = PokemonState{
		Species:  "Dragonite",
		Level:    100,
		IsActive: false,
		MaxHP:    180,
		HP:       180,
		Boosts:   NeutralBoosts,
	}

	ExecuteSpecificTurn(state, ActionMove1, -1)
	if state.P2.ActiveIdx != 1 {
		t.Fatalf("expected roar to force a random switch to bench slot, got active idx %d", state.P2.ActiveIdx)
	}
}

func TestSwiftSwimInRainChangesMoveOrder(t *testing.T) {
	ensureGameData(t)

	state := &BattleState{
		RNGState: 29,
		Field:    FieldConditions{Weather: "raindance"},
		P1:       PlayerState{ID: "p1", TeamSize: 1, ActiveIdx: 0},
		P2:       PlayerState{ID: "p2", TeamSize: 1, ActiveIdx: 0},
	}
	state.P1.Team[0] = PokemonState{
		Species:  "Ludicolo",
		Ability:  "Swift Swim",
		Level:    100,
		IsActive: true,
		MaxHP:    40,
		HP:       40,
		Stats:    Stats{Atk: 500, Def: 60, Spe: 50},
		Moves:    [4]string{"tackle"},
		NumMoves: 1,
		Boosts:   NeutralBoosts,
	}
	state.P2.Team[0] = PokemonState{
		Species:  "Alakazam",
		Level:    100,
		IsActive: true,
		MaxHP:    40,
		HP:       40,
		Stats:    Stats{Atk: 500, Def: 50, Spe: 90},
		Moves:    [4]string{"tackle"},
		NumMoves: 1,
		Boosts:   NeutralBoosts,
	}

	ExecuteSpecificTurn(state, ActionMove1, ActionMove1)
	if state.P2.ActiveIdx != -1 {
		t.Fatalf("expected Swift Swim user to outspeed and KO in rain")
	}
}
