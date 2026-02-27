package client

import (
	"testing"
)

func TestOpponentTracking(t *testing.T) {
	bot := &ShowdownBot{
		battles: make(map[string]*BattleContext),
	}

	roomID := "battle-gen9randombattle-1"
	bot.battles[roomID] = &BattleContext{
		RoomID:     roomID,
		OpponentID: "p2",
	}

	// Simulate an opponent switching in
	bot.onOpponentAction(roomID, "p2a: Gengar|Gengar, L85, M|100/100", true)

	ctx := bot.battles[roomID]
	if ctx.State == nil {
		t.Fatalf("Expected State to be initialized")
	}

	oppTeam := ctx.State.P2.Team
	if oppTeam[0].Species != "Gengar" {
		t.Errorf("Expected species to be Gengar, got %s", oppTeam[0].Species)
	}

	if oppTeam[0].HP != 100 {
		t.Errorf("Expected HP to be 100, got %d", oppTeam[0].HP)
	}

	// Create a dummy request to check RequestToBattleState
	req := &ShowdownRequest{
		Side: ShowdownSide{
			ID: "p1",
		},
	}

	state := RequestToBattleState(req, ctx.State)
	if state.P2.Team[0].Species != "Gengar" {
		t.Errorf("Expected RequestToBattleState to populate opponent state, got %s", state.P2.Team[0].Species)
	}
}
