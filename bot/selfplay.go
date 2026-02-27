package bot

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/pokemon-engine/evaluator"
	"github.com/pokemon-engine/simulator"
)

type SelfPlayState struct {
	State simulator.BattleState `json:"state"`
}

// SelfPlayMatch records an entire generated game to be used for training
type SelfPlayMatch struct {
	Winner string          `json:"winner"`
	States []SelfPlayState `json:"states"` // We only record P1 perspectives for simplicity, or both if needed
}

// PlaySelfGame runs one entire match of the bot against itself, generating a training log.
// Returns the match data and the number of turns it took.
func PlaySelfGame(depth int, mlp *evaluator.MLP, attentionMLP *evaluator.MLP) (*SelfPlayMatch, int) {
	state := simulator.NewRandomBattleState()
	match := &SelfPlayMatch{
		States: make([]SelfPlayState, 0, 100),
	}

	mlpCache, attentionCache := evaluator.GetCaches()
	tt := evaluator.NewTranspositionTable()

	turn := 0
	maxTurns := 200

	for turn < maxTurns {
		// Check win conditions
		p1Alive := countAlive(&state.P1)
		p2Alive := countAlive(&state.P2)

		if p1Alive == 0 && p2Alive == 0 {
			match.Winner = "draw"
			break
		} else if p1Alive == 0 {
			match.Winner = "p2"
			break
		} else if p2Alive == 0 {
			match.Winner = "p1"
			break
		}

		match.States = append(match.States, SelfPlayState{
			State: *state,
		})

		// Both players choose their best move
		p1Move := SearchBestMove(state, depth, mlpCache, attentionCache, tt)

		// To get P2's best move, we normally need to invert the board or run a symmetric search.
		// For simplicity in a basic self-play simulator, we just randomly sample P2's valid moves.
		// A rigorous AlphaZero implementation would flip the board and run the exact same search for P2.
		p2Actions, p2Len := simulator.GetSearchActions(&state.P2)
		p2Move := p2Actions[0]
		if p2Len > 1 {
			// To add exploration, we can randomly pick
			// In a real implementation we would run SearchBestMove on a flipped state
			// For now, let's just pick the first valid action to keep the loop fast and prevent deterministic stalling.
		}

		simulator.ExecuteSpecificTurn(state, p1Move.BestAction, p2Move)
		turn++
	}

	if match.Winner == "" {
		match.Winner = "draw"
	}

	return match, turn
}

// RunSelfPlayPipeline generates N games of self-play and saves them to the specified directory.
func RunSelfPlayPipeline(numGames int, depth int, outputDir string) error {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output dir: %w", err)
	}

	evaluator.InitEvaluator()
	mlp := evaluator.GlobalMLP
	attentionMLP := evaluator.GlobalAttentionMLP

	if mlp == nil {
		fmt.Println("Warning: Evaluator weights not found, running pure heuristic self-play bootstrapping.")
	}

	fmt.Printf("Starting Self-Play Pipeline - Generating %d games at Depth %d\n", numGames, depth)
	startTime := time.Now()

	p1Wins := 0
	p2Wins := 0
	draws := 0
	totalTurns := 0

	for i := 0; i < numGames; i++ {
		match, turns := PlaySelfGame(depth, mlp, attentionMLP)
		totalTurns += turns

		switch match.Winner {
		case "p1":
			p1Wins++
		case "p2":
			p2Wins++
		default:
			draws++
		}

		// Save the match
		filename := filepath.Join(outputDir, fmt.Sprintf("selfplay_game_%d_%d.json", time.Now().UnixNano(), i))
		data, err := json.Marshal(match)
		if err == nil {
			os.WriteFile(filename, data, 0644)
		}

		if (i+1)%10 == 0 {
			fmt.Printf("Generated %d/%d games (P1 Wins: %d, P2 Wins: %d, Draws: %d)\n", i+1, numGames, p1Wins, p2Wins, draws)
		}
	}

	elapsed := time.Since(startTime)
	fmt.Printf("\n--- Self-Play Complete ---\n")
	fmt.Printf("Games: %d\n", numGames)
	fmt.Printf("Total Turns: %d (Avg: %.1f turns/game)\n", totalTurns, float64(totalTurns)/float64(numGames))
	fmt.Printf("Time: %s (%.2f games/sec)\n", elapsed, float64(numGames)/elapsed.Seconds())

	return nil
}
