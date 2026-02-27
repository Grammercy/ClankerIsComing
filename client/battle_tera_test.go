package client

import (
	"testing"

	"github.com/pokemon-engine/simulator"
)

func TestActionToShowdownTeraMove(t *testing.T) {
	req := &ShowdownRequest{
		Active: []ShowdownActive{
			{
				CanTerastallize: true,
				Moves: []ShowdownMove{
					{ID: "thunderbolt", PP: 24, Disabled: false},
					{ID: "voltswitch", PP: 20, Disabled: false},
				},
			},
		},
	}

	got := actionToShowdown(simulator.ActionTeraMove2, req)
	if got != "move 2 terastallize" {
		t.Fatalf("expected tera choice, got %q", got)
	}
}
