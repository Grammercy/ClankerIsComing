package simulator

import "testing"

func TestGetSearchActionsIncludesTeraMoves(t *testing.T) {
	player := &PlayerState{
		ID:              "p1",
		TeamSize:        1,
		ActiveIdx:       0,
		CanTerastallize: true,
	}
	player.Team[0] = PokemonState{
		Species:       "Pikachu",
		IsActive:      true,
		NumMoves:      2,
		TeraType:      "Electric",
		Boosts:        114420174,
		Moves:         [4]string{"thunderbolt", "voltswitch"},
		MaxHP:         100,
		HP:            100,
		Terastallized: false,
	}

	actions, n := GetSearchActions(player)
	if n != 4 {
		t.Fatalf("expected 4 actions (2 moves + 2 tera moves), got %d", n)
	}

	want := []int{ActionMove1, ActionMove2, ActionTeraMove1, ActionTeraMove2}
	for i, w := range want {
		if actions[i] != w {
			t.Fatalf("action[%d] = %d, want %d", i, actions[i], w)
		}
	}
}

func TestExecuteSpecificTurnConsumesTerastallize(t *testing.T) {
	state := &BattleState{
		P1: PlayerState{
			ID:              "p1",
			TeamSize:        1,
			ActiveIdx:       0,
			CanTerastallize: true,
		},
		P2: PlayerState{
			ID:        "p2",
			TeamSize:  1,
			ActiveIdx: 0,
		},
	}
	state.P1.Team[0] = PokemonState{
		Species:       "Pikachu",
		IsActive:      true,
		NumMoves:      1,
		Moves:         [4]string{"thunderbolt"},
		TeraType:      "Electric",
		Boosts:        114420174,
		MaxHP:         100,
		HP:            100,
		Terastallized: false,
	}
	state.P2.Team[0] = PokemonState{
		Species:  "Eevee",
		IsActive: true,
		Boosts:   114420174,
		MaxHP:    100,
		HP:       100,
	}

	ExecuteSpecificTurn(state, ActionTeraMove1, -1)

	if !state.P1.Team[0].Terastallized {
		t.Fatalf("expected active Pokemon to be terastallized after tera move")
	}
	if state.P1.CanTerastallize {
		t.Fatalf("expected CanTerastallize to be false after using tera")
	}
}
