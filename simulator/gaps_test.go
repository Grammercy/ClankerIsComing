package simulator

import (
	"testing"

	"github.com/pokemon-engine/gamedata"
)

func TestForcedSwitchImmunities(t *testing.T) {
	ensureGameData(t)

	// Test Suction Cups
	state := &BattleState{
		RNGState: 1,
		P1:       PlayerState{ID: "p1", TeamSize: 1, ActiveIdx: 0},
		P2: PlayerState{ID: "p2", TeamSize: 2, ActiveIdx: 0, Team: [6]PokemonState{
			{Species: "Octillery", Ability: "Suction Cups", IsActive: true, MaxHP: 100, HP: 100},
			{Species: "Pikachu", IsActive: false, MaxHP: 100, HP: 100},
		}},
	}
	state.P1.Team[0] = PokemonState{Species: "Arcanine", IsActive: true, Moves: [4]string{"roar"}, NumMoves: 1}

	ExecuteSpecificTurn(state, ActionMove1, -1)
	if state.P2.ActiveIdx != 0 {
		t.Fatalf("expected Suction Cups to prevent being forced out, but switched to %d", state.P2.ActiveIdx)
	}

	// Test Ingrain
	state.P2.Team[0].Ability = ""
	state.P2.Team[0].Volatiles |= VolatileIngrain
	ExecuteSpecificTurn(state, ActionMove1, -1)
	if state.P2.ActiveIdx != 0 {
		t.Fatalf("expected Ingrain to prevent being forced out, but switched to %d", state.P2.ActiveIdx)
	}
}

func TestForcedSwitchTriggersAbilitiesAndHazards(t *testing.T) {
	ensureGameData(t)

	state := &BattleState{
		RNGState: 1,
		P1:       PlayerState{ID: "p1", TeamSize: 1, ActiveIdx: 0},
		P2: PlayerState{ID: "p2", TeamSize: 2, ActiveIdx: 0,
			Side: SideConditions{StealthRock: true},
			Team: [6]PokemonState{
				{Species: "Arcanine", IsActive: true, MaxHP: 100, HP: 100},
				{Species: "Gyarados", Ability: "Intimidate", IsActive: false, MaxHP: 100, HP: 100, Boosts: NeutralBoosts},
			},
		},
	}
	state.P1.Team[0] = PokemonState{Species: "Arcanine", IsActive: true, Moves: [4]string{"roar"}, NumMoves: 1, Boosts: NeutralBoosts}

	ExecuteSpecificTurn(state, ActionMove1, -1)

	if state.P2.ActiveIdx != 1 {
		t.Fatalf("expected forced switch to slot 1")
	}

	// Check Hazads: Gyarados is Flying/Water, 2x weak to Rock (25% damage)
	if state.P2.Team[1].HP != 75 {
		t.Fatalf("expected Gyarados to take 25 hazard damage, got filter/HP=%d", state.P2.Team[1].HP)
	}

	// Check Ability: Intimidate should lower P1's Attack
	if state.P1.Team[0].GetBoost(AtkShift) != -1 {
		t.Fatalf("expected Intimidate to trigger after forced switch, got Atk boost %d", state.P1.Team[0].GetBoost(AtkShift))
	}
}

func TestAegislashStanceChange(t *testing.T) {
	ensureGameData(t)

	state := &BattleState{
		RNGState: 1,
		P1:       PlayerState{ID: "p1", TeamSize: 1, ActiveIdx: 0},
		P2: PlayerState{ID: "p2", TeamSize: 2, ActiveIdx: 0, Team: [6]PokemonState{
			{Species: "Blissey", IsActive: true, MaxHP: 7000, HP: 7000, Stats: Stats{Def: 100, SpD: 100, Spe: 10}},
			{Species: "Blissey", IsActive: false, MaxHP: 700, HP: 700, Stats: Stats{Def: 100, SpD: 100, Spe: 10}},
		}},
	}
	state.P1.Team[0] = PokemonState{
		Species:  "Aegislash", // Shield Forme: 50 Atk, 140 Def
		Ability:  "Stance Change",
		IsActive: true,
		MaxHP:    100,
		HP:       100,
		Level:    100,
		Stats:    Stats{Atk: 136, Def: 316, SpA: 136, SpD: 316, Spe: 156},
		Moves:    [4]string{"ironhead", "kingsshield"},
		NumMoves: 2,
		Boosts:   NeutralBoosts,
	}

	// Use Iron Head (Attack)
	ExecuteSpecificTurn(state, ActionMove1, -1)

	if state.P1.Team[0].Species != "Aegislash-Blade" {
		t.Fatalf("expected Aegislash to transform to Blade form, got %s", state.P1.Team[0].Species)
	}
	if state.P1.Team[0].Stats.Atk <= 136 {
		t.Fatalf("expected Aegislash-Blade to have higher Atk, got %d", state.P1.Team[0].Stats.Atk)
	}

	// Use King's Shield (back to Shield form)
	// King's Shield is slot 2 (ActionMove2)
	ExecuteSpecificTurn(state, ActionMove2, -1)
	if state.P1.Team[0].Species != "Aegislash" {
		t.Fatalf("expected Aegislash to transform to Shield form, got %s", state.P1.Team[0].Species)
	}
}

func TestPalafinZeroToHero(t *testing.T) {
	ensureGameData(t)

	state := &BattleState{
		RNGState: 1,
		P1:       PlayerState{ID: "p1", TeamSize: 2, ActiveIdx: 0},
		P2:       PlayerState{ID: "p2", TeamSize: 1, ActiveIdx: 0, Team: [6]PokemonState{{Species: "Magikarp", IsActive: true}}},
	}
	state.P1.Team[0] = PokemonState{
		Species:  "Palafin",
		Ability:  "Zero to Hero",
		IsActive: true,
		MaxHP:    341,
		HP:       341,
		Level:    100,
		Stats:    Stats{Atk: 176, Def: 180, Spe: 236},
		Boosts:   NeutralBoosts,
	}
	state.P1.Team[1] = PokemonState{Species: "Pikachu", IsActive: false, MaxHP: 100, HP: 100}

	// Switch out
	ExecuteSpecificTurn(state, ActionSwitchBase+1, -1)
	if state.P1.Team[0].Species != "Palafin-Hero" {
		t.Fatalf("expected Palafin to transform to Hero form on switch out, got %s", state.P1.Team[0].Species)
	}
	if state.P1.Team[0].Stats.Atk <= 176 {
		t.Fatalf("expected Palafin-Hero to have higher Atk, got %d", state.P1.Team[0].Stats.Atk)
	}
}

func TestWeatherAccuracy(t *testing.T) {
	ensureGameData(t)

	state := &BattleState{
		RNGState: 1,
		Field:    FieldConditions{Weather: "raindance"},
		P1:       PlayerState{ID: "p1", TeamSize: 1, ActiveIdx: 0},
		P2: PlayerState{ID: "p2", TeamSize: 1, ActiveIdx: 0, Team: [6]PokemonState{
			{Species: "Blissey", IsActive: true, MaxHP: 700, HP: 700, Stats: Stats{Def: 100, SpD: 100, Spe: 10}, Boosts: NeutralBoosts},
		}},
	}
	state.P1.Team[0] = PokemonState{
		Species:  "Zapdos",
		IsActive: true,
		Stats:    Stats{SpA: 300, Spe: 100},
		Moves:    [4]string{"thunder"},
		NumMoves: 1,
		Boosts:   NeutralBoosts,
	}

	// Thunder in Rain (100% acc)
	move := gamedata.LookupMove("thunder")
	if !moveHits(state, &state.P1.Team[0], &state.P2.Team[0], move) {
		t.Fatalf("expected Thunder to always hit in Rain")
	}

	// Thunder in Sun (50% acc)
	state.Field.Weather = "sunnyday"
	hits := 0
	for i := uint64(0); i < 100; i++ {
		state.RNGState = i
		if moveHits(state, &state.P1.Team[0], &state.P2.Team[0], move) {
			hits++
		}
	}
	if hits == 100 || hits == 0 {
		t.Fatalf("expected Thunder to have ~50%% accuracy in Sun, got %d hits", hits)
	}
}
