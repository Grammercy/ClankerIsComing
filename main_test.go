package main

import (
    "testing"
    "github.com/pokemon-engine/parser"
    "github.com/pokemon-engine/simulator"
)

func createTestReplay() *parser.Replay {
    replay := &parser.Replay{
        P1: "p1", P2: "p2",
        Teams: map[string]map[string]bool{
            "p1": {"pikachu": true},
            "p2": {"charmander": true},
        },
        Events: []parser.Event{
            {Turn: 0, Type: "switch", Player: "p1", Value: "pikachu", Detail: "p1a: pikachu switched to pikachu"},
            {Turn: 0, Type: "switch", Player: "p2", Value: "charmander", Detail: "p2a: charmander switched to charmander"},
        },
    }

    // Add 1000 turns of events
    for t := 1; t <= 1000; t++ {
        replay.Events = append(replay.Events,
            parser.Event{Turn: t, Type: "move", Player: "p1", Value: "thunderbolt"},
            parser.Event{Turn: t, Type: "damage", Player: "p2", Value: "charmander", Detail: "charmander->90/100"},
            parser.Event{Turn: t, Type: "move", Player: "p2", Value: "ember"},
            parser.Event{Turn: t, Type: "damage", Player: "p1", Value: "pikachu", Detail: "pikachu->90/100"},
        )
    }
    return replay
}

func BenchmarkQuadraticLoop(b *testing.B) {
    replay := createTestReplay()
    indices := collectReplayDecisionEventIndices(replay)

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        for _, eventIdx := range indices {
            _, err := simulator.FastForwardToEvent(replay, eventIdx-1)
            if err != nil {
                continue
            }
        }
    }
}

func BenchmarkLinearLoop(b *testing.B) {
    replay := createTestReplay()
    indices := collectReplayDecisionEventIndices(replay)

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        if len(indices) == 0 {
            continue
        }

        state, err := simulator.FastForwardToEvent(replay, indices[0]-1)
        if err != nil {
            continue
        }

        lastEventIdx := indices[0] - 1
        for _, eventIdx := range indices {
            for j := lastEventIdx + 1; j < eventIdx; j++ {
                simulator.ApplyEvent(state, replay.Events[j])
            }
            simulator.UpdateRNGState(state, replay, eventIdx-1)
            lastEventIdx = eventIdx - 1
        }
    }
}
