package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pokemon-engine/bot"
	"github.com/pokemon-engine/client"
	"github.com/pokemon-engine/deepcfr"
	"github.com/pokemon-engine/evaluator"
	"github.com/pokemon-engine/gamedata"
	"github.com/pokemon-engine/parser"
	"github.com/pokemon-engine/scraper"
	"github.com/pokemon-engine/simulator"
)

func main() {
	cmd := flag.String("cmd", "", "Command to run: 'scrape', 'parse', 'actions', 'verify-actions', 'evaluate', 'bulk-evaluate', 'search-evaluate', 'train-deepcfr', 'deep-evaluate', 'import', 'live'")
	format := flag.String("format", "gen9randombattle", "Pokemon Showdown format to scrape")
	numGames := flag.Int("num", 100, "Number of games to scrape")
	outDir := flag.String("out", "data/replays", "Output directory for scraped replays")
	inDir := flag.String("in", "data/replays", "Input directory containing scraped replays to parse")
	concurrency := flag.Int("jobs", 5, "Number of concurrent download routines")
	// Actions command flags
	file := flag.String("file", "", "Path to a specific replay log to run actions command against")
	turn := flag.Int("turn", 1, "Target turn number to simulate up to")
	player := flag.String("player", "p1", "Player ID to generate actions for (e.g. 'p1' or 'p2')")
	depth := flag.Int("depth", 2, "Search depth for Alpha-Beta engine (search-evaluate command)")
	sims := flag.Int("sims", 0, "MCTS simulation count (0 = use depth-based default)")
	engineName := flag.String("engine", "mcts", "Decision engine for live play: 'mcts' or 'deepcfr'")
	modelPath := flag.String("model", "data/deepcfr_model.json", "Path to a Deep CFR model file")
	epochs := flag.Int("epochs", 1, "Training epochs for train-deepcfr")
	maxFiles := flag.Int("max-files", 0, "Optional cap on replay files used for train-deepcfr")
	beliefSamples := flag.Int("belief-samples", 8, "Belief samples for Deep CFR training/evaluation")
	trainAccelerator := flag.String("train-accelerator", "auto", "Training accelerator for train-deepcfr: 'auto', 'cpu', or 'opencl'")
	trainBatchSize := flag.Int("train-batch-size", 128, "Mini-batch size for GPU-backed Deep CFR training")
	targetWorkers := flag.Int("target-workers", 0, "Parallel workers for CPU target generation in train-deepcfr (0 = auto)")
	openclPlatform := flag.String("opencl-platform", "", "Optional OpenCL platform name substring for train-deepcfr")
	openclDevice := flag.String("opencl-device", "", "Optional OpenCL device name substring for train-deepcfr")
	moveTimeMs := flag.Int("move-time-ms", int(client.SearchMoveTime/time.Millisecond), "Per-move time budget in milliseconds (live command)")
	url := flag.String("url", "", "Pokemon Showdown replay URL for import command")
	// Live bot flags
	user := flag.String("user", "", "Pokemon Showdown username for live bot")
	pass := flag.String("pass", "", "Pokemon Showdown password (empty for guest)")

	flag.Parse()

	if *cmd == "" {
		fmt.Println("Error: --cmd flag is required.")
		fmt.Println("Example (Windows): .\\pokemon-engine.exe -cmd parse -in data\\replays")
		fmt.Println("Example (all platforms): go run . -cmd parse -in data/replays")
		fmt.Println("\nAvailable Commands:")
		fmt.Println("  scrape           Download replays from Pokemon Showdown")
		fmt.Println("  parse            Extract events and data from downloaded replay logs")
		fmt.Println("  actions          List valid actions for a specific turn in a replay")
		fmt.Println("  verify-actions   Bulk verify simulator action generation against replays")
		fmt.Println("  evaluate         Predict win probability for a specific game state")
		fmt.Println("  bulk-evaluate    Run win probability prediction on a large set of replays")
		fmt.Println("  search-evaluate  Run search-enhanced evaluation (Alpha-Beta) on replays")
		fmt.Println("  train-deepcfr    Train an imperfect-information Deep CFR style model from replay logs")
		fmt.Println("  deep-evaluate    Evaluate a replay decision state with the Deep CFR engine")
		fmt.Println("  import           Download, parse, and analyze a specific Showdown replay URL")
		fmt.Println("  live             Run the bot live on Pokemon Showdown (supports -engine deepcfr)")
		fmt.Println("")
		flag.Usage()
		os.Exit(1)
	}

	switch *cmd {
	case "scrape":
		cfg := scraper.ScrapeConfig{
			Format:      *format,
			NumGames:    *numGames,
			OutputDir:   *outDir,
			PageLimit:   100000, // Safe default max pages
			Concurrency: *concurrency,
		}

		err := scraper.ScrapeReplays(cfg)
		if err != nil {
			fmt.Printf("Scraping failed: %v\n", err)
			os.Exit(1)
		}
	case "parse":
		err := runParseCommand(*inDir)
		if err != nil {
			fmt.Printf("Parsing failed: %v\n", err)
			os.Exit(1)
		}
	case "actions":
		if *file == "" {
			fmt.Println("Error: --file is required for the 'actions' command.")
			os.Exit(1)
		}
		err := runActionsCommand(*file, *turn, *player)
		if err != nil {
			fmt.Printf("Actions generation failed: %v\n", err)
			os.Exit(1)
		}
	case "verify-actions":
		err := runVerifyActionsCommand(*inDir)
		if err != nil {
			fmt.Printf("Verify actions failed: %v\n", err)
			os.Exit(1)
		}
	case "evaluate":
		if *file == "" {
			fmt.Println("Error: --file is required for the 'evaluate' command.")
			os.Exit(1)
		}
		err := runEvaluateCommand(*file, *turn)
		if err != nil {
			fmt.Printf("Evaluation failed: %v\n", err)
			os.Exit(1)
		}
	case "bulk-evaluate":
		if err := gamedata.LoadPokedex("data/pokedex.json"); err != nil {
			fmt.Printf("Warning: Pokedex not loaded: %v\n", err)
		}
		err := runEvaluateBulkCommand(*inDir, *turn, *depth, *sims)
		if err != nil {
			fmt.Printf("Bulk Evaluation failed: %v\n", err)
			os.Exit(1)
		}
	case "search-evaluate":
		if err := gamedata.LoadPokedex("data/pokedex.json"); err != nil {
			fmt.Printf("Warning: Pokedex not loaded: %v\n", err)
		}
		if err := gamedata.LoadMovedex("data/moves.json"); err != nil {
			fmt.Printf("Warning: Movedex not loaded: %v\n", err)
		}
		err := runSearchEvaluateBulkCommand(*inDir, *turn, *depth, *sims)
		if err != nil {
			fmt.Printf("Search evaluation failed: %v\n", err)
			os.Exit(1)
		}
	case "train-deepcfr":
		if err := gamedata.LoadPokedex("data/pokedex.json"); err != nil {
			fmt.Printf("Warning: Pokedex not loaded: %v\n", err)
		}
		if err := gamedata.LoadMovedex("data/moves.json"); err != nil {
			fmt.Printf("Warning: Movedex not loaded: %v\n", err)
		}
		err := runTrainDeepCFRCommand(*inDir, *modelPath, *epochs, *maxFiles, *beliefSamples, *depth, *trainAccelerator, *trainBatchSize, *targetWorkers, *openclPlatform, *openclDevice)
		if err != nil {
			fmt.Printf("Deep CFR training failed: %v\n", err)
			os.Exit(1)
		}
	case "deep-evaluate":
		if *file == "" {
			fmt.Println("Error: --file is required for the 'deep-evaluate' command.")
			os.Exit(1)
		}
		if err := gamedata.LoadPokedex("data/pokedex.json"); err != nil {
			fmt.Printf("Warning: Pokedex not loaded: %v\n", err)
		}
		if err := gamedata.LoadMovedex("data/moves.json"); err != nil {
			fmt.Printf("Warning: Movedex not loaded: %v\n", err)
		}
		err := runDeepEvaluateCommand(*file, *turn, *player, *modelPath, *beliefSamples, *depth)
		if err != nil {
			fmt.Printf("Deep CFR evaluation failed: %v\n", err)
			os.Exit(1)
		}
	case "import":
		if *url == "" {
			fmt.Println("Error: --url is required for the 'import' command.")
			fmt.Println("Example: go run . -cmd import -url https://replay.pokemonshowdown.com/gen9ou-1234567890")
			os.Exit(1)
		}
		if err := gamedata.LoadPokedex("data/pokedex.json"); err != nil {
			fmt.Printf("Warning: Pokedex not loaded: %v\n", err)
		}
		err := runImportCommand(*url)
		if err != nil {
			fmt.Printf("Import failed: %v\n", err)
			os.Exit(1)
		}
	case "live":
		if *user == "" {
			fmt.Println("Error: --user is required for the 'live' command.")
			fmt.Println("Example: go run . -cmd live -user MyBot -pass MyPassword")
			os.Exit(1)
		}
		err := client.RunBot(*user, *pass, time.Duration(*moveTimeMs)*time.Millisecond, *engineName, *modelPath)
		if err != nil {
			fmt.Printf("Bot error: %v\n", err)
			os.Exit(1)
		}
	default:
		fmt.Printf("Unknown command: %s\n", *cmd)
		os.Exit(1)
	}
}

func runTrainDeepCFRCommand(inDir string, modelPath string, epochs int, maxFiles int, beliefSamples int, depth int, accelerator string, batchSize int, targetWorkers int, openclPlatform string, openclDevice string) error {
	fmt.Printf("Training Deep CFR model from %s\n", inDir)
	model, stats, err := deepcfr.TrainFromReplayDir(deepcfr.TrainConfig{
		ReplayDir:         inDir,
		ModelPath:         modelPath,
		Epochs:            epochs,
		MaxFiles:          maxFiles,
		Seed:              1,
		BeliefSamples:     beliefSamples,
		OpponentSamples:   3,
		TargetDepth:       depth,
		LearningRate:      0.0005,
		ReplayPolicyBlend: 0.35,
		ChosenActionBonus: 0.10,
		ProgressWriter:    os.Stdout,
		ProgressInterval:  10 * time.Second,
		Accelerator:       accelerator,
		BatchSize:         batchSize,
		TargetWorkers:     targetWorkers,
		OpenCLPlatform:    openclPlatform,
		OpenCLDevice:      openclDevice,
	})
	if err != nil {
		return err
	}
	fmt.Printf("Saved model to %s\n", modelPath)
	fmt.Printf("Backend: %s\n", stats.Backend)
	fmt.Printf("Files Seen: %d\n", stats.FilesSeen)
	fmt.Printf("Positions: %d\n", stats.Positions)
	fmt.Printf("Average Loss: %.6f\n", stats.AverageLoss)
	fmt.Printf("Average Value MAE: %.6f\n", stats.AverageValueMAE)
	fmt.Printf("Average Policy CE: %.6f\n", stats.AveragePolicyCE)
	if model != nil && model.Priors != nil {
		fmt.Printf("Species Priors Learned: %d\n", len(model.Priors.SpeciesCounts))
	}
	return nil
}

func runDeepEvaluateCommand(filePath string, targetTurn int, playerID string, modelPath string, beliefSamples int, depth int) error {
	replay, err := parser.ParseLogFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to parse replay: %w", err)
	}
	model, err := deepcfr.LoadModel(modelPath)
	if err != nil {
		return err
	}
	state, actualAction, hasActual := deepcfr.DecisionStateForTurn(replay, targetTurn, playerID)
	engine := deepcfr.NewEngine(model, 1)
	result := engine.Evaluate(state, deepcfr.SearchConfig{
		BeliefSamples:   beliefSamples,
		OpponentSamples: 3,
		Depth:           depth,
	})

	fmt.Printf("Deep CFR evaluation for %s turn %d (%s perspective)\n", filePath, targetTurn, playerID)
	fmt.Printf("Win Probability: %.2f%%\n", result.WinProbability*100.0)
	fmt.Printf("Recommended Action: %s\n", bot.ActionToString(state, &state.P1, result.BestAction))
	if hasActual {
		fmt.Printf("Replay Action: %s\n", bot.ActionToString(state, &state.P1, actualAction))
	}
	fmt.Println("Action Values:")
	actions := make([]int, 0, len(result.ActionValues))
	for action := range result.ActionValues {
		actions = append(actions, action)
	}
	sort.Ints(actions)
	for _, action := range actions {
		fmt.Printf("  %-28s value=%.4f policy=%.4f\n",
			bot.ActionToString(state, &state.P1, action),
			result.ActionValues[action],
			result.ActionPolicy[action],
		)
	}
	return nil
}

func runParseCommand(inDir string) error {
	fmt.Printf("Starting parse command on directory: %s\n", inDir)
	entries, err := os.ReadDir(inDir)
	if err != nil {
		return fmt.Errorf("could not read directory: %w", err)
	}

	successCount := 0
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".log") {
			continue
		}

		filePath := fmt.Sprintf("%s/%s", inDir, entry.Name())
		fmt.Printf("Parsing file: %s...\n", filePath)
		replay, err := parser.ParseLogFile(filePath)
		if err != nil {
			fmt.Printf("Error parsing %s: %v\n", entry.Name(), err)
			continue
		}

		fmt.Printf("Parsed %s: %s vs %s (Winner: %s) in %d turns. Extracted %d events.\n", entry.Name(), replay.P1, replay.P2, replay.Winner, replay.Turns, len(replay.Events))
		successCount++
	}

	fmt.Printf("\nSuccessfully parsed %d files.\n", successCount)
	return nil
}

func runActionsCommand(filePath string, targetTurn int, playerID string) error {
	fmt.Printf("Parsing replay file: %s\n", filePath)
	replay, err := parser.ParseLogFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to parse log: %w", err)
	}

	fmt.Printf("Simulating game up to Turn %d...\n", targetTurn)
	state, err := simulator.FastForward(replay, targetTurn)
	if err != nil {
		return fmt.Errorf("failed to simulate state: %w", err)
	}

	actions, err := simulator.GetValidActions(state, replay, playerID)
	if err != nil {
		return fmt.Errorf("failed to get valid actions: %w", err)
	}

	fmt.Printf("\n--- Valid Actions for %s at Turn %d ---\n", playerID, targetTurn)
	for i, action := range actions {
		fmt.Printf("[%d] %s: %s\n", i+1, strings.ToUpper(action.Type), action.Name)
	}
	fmt.Println("---------------------------------------")

	return nil
}
func runVerifyActionsCommand(inDir string) error {
	fmt.Printf("Starting bulk verification on directory: %s\n", inDir)
	entries, err := os.ReadDir(inDir)
	if err != nil {
		return fmt.Errorf("could not read directory: %w", err)
	}

	totalFiles := 0
	successfulChecks := 0
	failedChecks := 0

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".log") {
			continue
		}

		filePath := fmt.Sprintf("%s/%s", inDir, entry.Name())
		replay, err := parser.ParseLogFile(filePath)
		if err != nil {
			fmt.Printf("Error parsing %s: %v\n", entry.Name(), err)
			continue
		}

		totalFiles++
		fileSuccess := true

		// We want to simulate state and verify every move/switch action against GetValidActions
		// We'll iterate through events. To know the state BEFORE the event, we simulate up to Turn-1.
		// NOTE: In Showdown, multiple events happen per turn. We'll group actions made by players.
		// For a simpler first pass, we check every "move" and "switch" event in the replay
		// and ensure it was in the ValidActions pool at that Turn.

		for i, event := range replay.Events {
			if event.Type != "move" && event.Type != "switch" {
				continue
			}

			// Turn 0 switches are initial sent-outs (leads).
			// Players select leads at Team Preview, the simulator targets mid-battle state where
			// you pick a switch. Thus, ignore Turn 0 switches.
			if event.Turn == 0 {
				continue
			}

			// state before this exact action
			targetIndex := i - 1
			if targetIndex < 0 {
				continue
			}

			state, err := simulator.FastForwardToEvent(replay, targetIndex)
			if err != nil {
				fmt.Printf("Error simulating %s up to event %d: %v\n", entry.Name(), targetIndex, err)
				fileSuccess = false
				break
			}

			actions, err := simulator.GetValidActions(state, replay, event.Player)
			if err != nil {
				fmt.Printf("Error getting valid actions for %s on event %d: %v\n", entry.Name(), targetIndex, err)
				fileSuccess = false
				break
			}

			found := false
			for _, action := range actions {
				if action.Type == event.Type {
					if action.Type == "move" {
						// For bulk verification, just check if ANY Pokemon on the team knows this move.
						// This avoids tight Turn/Active state tracking bugs misaligning KnownMoves.
						for _, moves := range replay.KnownMoves[event.Player] {
							if _, knowsMove := moves[event.Value]; knowsMove {
								found = true
								break
							}
						}
					} else if action.Type == "switch" {
						// e.g "event.Value: Kyurem", "action.Name: Kyurem-Black"
						// e.g "event.Value: Zamazenta-Crowned", "action.Name: Zamazenta"
						baseEventVal := strings.Split(event.Value, "-")[0]
						baseActionVal := strings.Split(action.Name, "-")[0]

						if strings.Contains(event.Value, action.Name) || strings.Contains(action.Name, event.Value) || baseEventVal == baseActionVal {
							found = true
							break
						}
					}
				}
			}

			if !found && event.Type == "switch" {
				playerState := &state.P1
				if event.Player == "p2" {
					playerState = &state.P2
				}

				hasZoroark := false
				for i := 0; i < playerState.TeamSize; i++ {
					if strings.HasPrefix(playerState.Team[i].Species, "Zoroark") {
						hasZoroark = true
						break
					}
				}

				if hasZoroark && playerState.GetActive() != nil {
					baseEventVal := strings.Split(event.Value, "-")[0]
					activeSp := strings.Split(playerState.GetActive().Species, "-")[0]
					if baseEventVal == activeSp {
						found = true
					}
				}
			}

			if !found {
				fmt.Printf("[!] Verification Failed in %s Turn %d (Event %d): Player %s took a %s '%s'.\n", entry.Name(), event.Turn, i, event.Player, event.Type, event.Value)
				fmt.Printf("Valid Actions Pool:\n")
				for _, a := range actions {
					fmt.Printf("- %s: %s\n", a.Type, a.Name)
				}

				// Let's print Team state!
				playerState := &state.P1
				if event.Player == "p2" {
					playerState = &state.P2
				}
				fmt.Printf("Team State & Active Pokemon: (Active: %v)\n", playerState.GetActive())
				for i := 0; i < playerState.TeamSize; i++ {
					p := &playerState.Team[i]
					fmt.Printf(" - %s: Active=%v, Fainted=%v\n", p.Species, p.IsActive, p.Fainted)
				}

				return fmt.Errorf("early exit on first failure")
			}
		}

		if fileSuccess {
			successfulChecks++
		} else {
			failedChecks++
		}

		if totalFiles%100 == 0 {
			fmt.Printf("...Processed %d files...\n", totalFiles)
		}
	}

	fmt.Printf("\n--- Bulk Verification Complete ---\n")
	fmt.Printf("Total Replays Checked: %d\n", totalFiles)
	fmt.Printf("Successful (100%% matches): %d\n", successfulChecks)
	fmt.Printf("Failed (Missing Action): %d\n", failedChecks)

	if failedChecks > 0 {
		return fmt.Errorf("%d replays failed verification", failedChecks)
	}

	return nil
}

func runEvaluateCommand(filePath string, targetTurn int) error {
	fmt.Printf("Parsing replay file: %s\n", filePath)
	replay, err := parser.ParseLogFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to parse log: %w", err)
	}

	fmt.Printf("Simulating game up to Turn %d...\n", targetTurn)
	state, err := simulator.FastForward(replay, targetTurn)
	if err != nil {
		return fmt.Errorf("failed to simulate state: %w", err)
	}

	winProb := evaluator.EvaluateState(state)

	fmt.Printf("\n--- Evaluation at Turn %d ---\n", targetTurn)
	fmt.Printf("Player 1 (%s) vs Player 2 (%s)\n", replay.P1, replay.P2)
	fmt.Printf("P1 Win Probability: %.2f%%\n", winProb*100)
	fmt.Printf("P2 Win Probability: %.2f%%\n", (1.0-winProb)*100)
	fmt.Println("-------------------------------")

	return nil
}

func runEvaluateBulkCommand(inDir string, targetTurn int, searchDepth int, sims int) error {
	fmt.Printf("Starting bulk evaluation on directory: %s at Turn %d (Depth: %d, Sims: %d)\n", inDir, targetTurn, searchDepth, sims)
	entries, err := os.ReadDir(inDir)
	if err != nil {
		return fmt.Errorf("could not read directory: %w", err)
	}

	// Shuffle and sample random games for benchmark
	rand.Shuffle(len(entries), func(i, j int) { entries[i], entries[j] = entries[j], entries[i] })
	maxSample := 10000
	if len(entries) > maxSample {
		entries = entries[:maxSample]
	}
	fmt.Printf("Sampling %d random replays...\n", len(entries))

	var totalFiles int32
	var correctPredictions int32
	var totalNodes int64
	startTime := time.Now()

	numWorkers := 16
	if searchDepth > 0 {
		// Reduce workers if searching to avoid massive overhead, but keep batching for rollout evals
		numWorkers = 8
	}

	jobs := make(chan string, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".log") {
			continue
		}
		jobs <- fmt.Sprintf("%s/%s", inDir, entry.Name())
	}
	close(jobs)

	// If depth is 0, we can use the high-performance batching evaluator
	if searchDepth <= 0 {
		return runBatchEvaluateBulk(jobs, len(entries), targetTurn, &totalFiles, &correctPredictions, startTime)
	}

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for filePath := range jobs {
				replay, err := parser.ParseLogFile(filePath)
				if err != nil {
					continue
				}

				evalTurn := targetTurn
				if targetTurn == 0 {
					if replay.Turns < 5 {
						continue
					}
					evalTurn = rand.Intn(replay.Turns-2) + 2
				} else if targetTurn < 0 {
					evalTurn = replay.Turns + targetTurn
					if evalTurn < 1 {
						continue
					}
				} else if replay.Turns < targetTurn {
					continue
				}

				state, err := simulator.FastForward(replay, evalTurn)
				if err != nil {
					continue
				}

				tf := atomic.AddInt32(&totalFiles, 1)

				// Use SearchEvaluate with specified depth and sims
				winProb, nodes := bot.SearchEvaluateWithSims(state, searchDepth, sims)
				atomic.AddInt64(&totalNodes, int64(nodes))

				actualWinner := "tie"
				if replay.Winner == replay.P1 {
					actualWinner = "p1"
				} else if replay.Winner == replay.P2 {
					actualWinner = "p2"
				}

				predictedWinner := "tie"
				if winProb > 0.5 {
					predictedWinner = "p1"
				} else if winProb < 0.5 {
					predictedWinner = "p2"
				}

				if predictedWinner == actualWinner {
					atomic.AddInt32(&correctPredictions, 1)
				}

				if tf%1000 == 0 {
					fmt.Printf("\r...Evaluated %d matches...", tf)
				}
			}
		}()
	}
	wg.Wait()

	printBulkReport(totalFiles, correctPredictions, totalNodes, startTime, targetTurn)
	return nil
}

func runBatchEvaluateBulk(jobs <-chan string, estimatedTotal int, targetTurn int, totalFiles *int32, correctPredictions *int32, startTime time.Time) error {
	numWorkers := 16
	stateChan := make(chan struct {
		state  simulator.BattleState
		winner string
	}, 1024)

	var wg sync.WaitGroup
	// Workers to parse and fast-forward states
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for filePath := range jobs {
				replay, err := parser.ParseLogFile(filePath)
				if err != nil {
					continue
				}

				evalTurn := targetTurn
				if targetTurn == 0 {
					if replay.Turns < 5 {
						continue
					}
					evalTurn = rand.Intn(replay.Turns-2) + 2
				} else if targetTurn < 0 {
					evalTurn = replay.Turns + targetTurn
					if evalTurn < 1 {
						continue
					}
				} else if replay.Turns < targetTurn {
					continue
				}

				state, err := simulator.FastForward(replay, evalTurn)
				if err != nil {
					continue
				}

				actualWinner := "tie"
				if replay.Winner == replay.P1 {
					actualWinner = "p1"
				} else if replay.Winner == replay.P2 {
					actualWinner = "p2"
				}

				stateChan <- struct {
					state  simulator.BattleState
					winner string
				}{*state, actualWinner}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(stateChan)
	}()

	// Batch consumer
	batchSize := 256
	var states []simulator.BattleState
	var winners []string

	for item := range stateChan {
		states = append(states, item.state)
		winners = append(winners, item.winner)

		if len(states) >= batchSize {
			processBatch(states, winners, totalFiles, correctPredictions)
			states = states[:0]
			winners = winners[:0]
			fmt.Printf("\r...Evaluated %d matches...", atomic.LoadInt32(totalFiles))
		}
	}

	if len(states) > 0 {
		processBatch(states, winners, totalFiles, correctPredictions)
	}

	printBulkReport(*totalFiles, *correctPredictions, 0, startTime, targetTurn)
	return nil
}

func processBatch(states []simulator.BattleState, winners []string, totalFiles *int32, correctPredictions *int32) {
	evals := evaluator.EvaluateBatchStates(states)
	for i, winProb := range evals {
		actualWinner := winners[i]
		predictedWinner := "tie"
		if winProb > 0.5 {
			predictedWinner = "p1"
		} else if winProb < 0.5 {
			predictedWinner = "p2"
		}

		if predictedWinner == actualWinner {
			atomic.AddInt32(correctPredictions, 1)
		}
		atomic.AddInt32(totalFiles, 1)
	}
}

func printBulkReport(totalFiles int32, correctPredictions int32, totalNodes int64, startTime time.Time, targetTurn int) {
	fmt.Printf("\n--- Bulk Prediction Report ---\n")
	if targetTurn == 0 {
		fmt.Printf("Total Valid Replays (Random Midgame Turns): %d\n", totalFiles)
	} else {
		fmt.Printf("Total Valid Replays (Reaching Turn %d): %d\n", targetTurn, totalFiles)
	}

	if totalFiles > 0 {
		accuracy := float64(correctPredictions) / float64(totalFiles) * 100.0
		elapsed := time.Since(startTime).Seconds()
		fmt.Printf("Correct Predictions: %d (%.2f%% Accuracy)\n", correctPredictions, accuracy)
		fmt.Printf("Time Elapsed: %.2f seconds (%.1f replays/sec)\n", elapsed, float64(totalFiles)/elapsed)
		if totalNodes > 0 {
			nps := float64(totalNodes) / elapsed
			fmt.Printf("Nodes Evaluated: %d\n", totalNodes)
			fmt.Printf("Nodes Per Second (NPS): %.0f\n", nps)
		}
	}
}

func runSearchEvaluateBulkCommand(inDir string, targetTurn int, searchDepth int, sims int) error {
	fmt.Printf("Starting search-enhanced bulk evaluation (depth=%d, sims=%d) on directory: %s at Turn %d\n", searchDepth, sims, inDir, targetTurn)
	entries, err := os.ReadDir(inDir)
	if err != nil {
		return fmt.Errorf("could not read directory: %w", err)
	}

	// Shuffle and sample 1000 random games for deep search eval
	rand.Shuffle(len(entries), func(i, j int) { entries[i], entries[j] = entries[j], entries[i] })
	maxSample := 1000
	if len(entries) > maxSample {
		entries = entries[:maxSample]
	}
	fmt.Printf("Sampling %d random replays...\n", len(entries))

	var totalFiles int32
	var correctPredictions int32
	var totalNodes int64
	startTime := time.Now()

	numWorkers := 16
	jobs := make(chan string, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".log") {
			continue
		}
		jobs <- fmt.Sprintf("%s/%s", inDir, entry.Name())
	}
	close(jobs)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for filePath := range jobs {
				replay, err := parser.ParseLogFile(filePath)
				if err != nil {
					continue
				}

				evalTurn := targetTurn
				if targetTurn == 0 {
					if replay.Turns < 5 {
						continue
					}
					evalTurn = rand.Intn(replay.Turns-2) + 2
				} else if replay.Turns < targetTurn {
					continue
				}

				state, err := simulator.FastForward(replay, evalTurn)
				if err != nil {
					continue
				}

				tf := atomic.AddInt32(&totalFiles, 1)

				// Use SearchEvaluate with specified depth and sims
				winProb, nodes := bot.SearchEvaluateWithSims(state, searchDepth, sims)
				atomic.AddInt64(&totalNodes, int64(nodes))

				actualWinner := "tie"
				if replay.Winner == replay.P1 {
					actualWinner = "p1"
				} else if replay.Winner == replay.P2 {
					actualWinner = "p2"
				}

				predictedWinner := "tie"
				if winProb > 0.5 {
					predictedWinner = "p1"
				} else if winProb < 0.5 {
					predictedWinner = "p2"
				}

				if predictedWinner == actualWinner {
					atomic.AddInt32(&correctPredictions, 1)
				}

				if tf%1000 == 0 {
					accuracy := float64(atomic.LoadInt32(&correctPredictions)) / float64(tf) * 100.0
					fmt.Printf("\r...Evaluated %d matches (%.2f%% accuracy so far)...", tf, accuracy)
				}
			}
		}()
	}
	wg.Wait()

	fmt.Printf("\n--- Search-Enhanced Prediction Report (Depth %d) ---\n", searchDepth)
	if targetTurn == 0 {
		fmt.Printf("Total Valid Replays (Random Midgame Turns): %d\n", totalFiles)
	} else {
		fmt.Printf("Total Valid Replays (Reaching Turn %d): %d\n", targetTurn, totalFiles)
	}

	if totalFiles > 0 {
		accuracy := float64(correctPredictions) / float64(totalFiles) * 100.0
		elapsed := time.Since(startTime).Seconds()
		nps := float64(totalNodes) / elapsed

		fmt.Printf("Correct Predictions: %d (%.2f%% Accuracy)\n", correctPredictions, accuracy)
		fmt.Printf("Time Elapsed: %.2f seconds\n", elapsed)
		fmt.Printf("Nodes Evaluated: %d\n", totalNodes)
		fmt.Printf("Nodes Per Second (NPS): %.0f\n", nps)
	}
	return nil
}

func hasPriorTerastallizeInTurn(events []parser.Event, idx int, player string) bool {
	if idx < 0 || idx >= len(events) {
		return false
	}
	turn := events[idx].Turn
	for i := idx - 1; i >= 0; i-- {
		if events[i].Turn != turn {
			break
		}
		if events[i].Player == player && events[i].Type == "terastallize" {
			return true
		}
	}
	return false
}

func collectReplayDecisionEventIndices(replay *parser.Replay) []int {
	indices := make([]int, 0, len(replay.Events)/2)
	for i, event := range replay.Events {
		if event.Player != "p1" {
			continue
		}
		switch event.Type {
		case "switch":
			indices = append(indices, i)
		case "move":
			// Moves that happen after a tera in the same turn share the same pre-decision state.
			if hasPriorTerastallizeInTurn(replay.Events, i, "p1") {
				continue
			}
			indices = append(indices, i)
		case "terastallize":
			indices = append(indices, i)
		}
	}
	return indices
}

func normalizeReplayMoveName(move string) string {
	s := strings.ToLower(move)
	s = strings.ReplaceAll(s, " ", "")
	s = strings.ReplaceAll(s, "-", "")
	s = strings.ReplaceAll(s, "'", "")
	s = strings.ReplaceAll(s, ".", "")
	s = strings.ReplaceAll(s, ":", "")
	return s
}

func mapReplaySwitchAction(state *simulator.BattleState, event parser.Event) (int, bool) {
	species := strings.TrimSpace(event.Value)
	if species == "" {
		return -1, false
	}
	for i := 0; i < state.P1.TeamSize; i++ {
		teamMon := &state.P1.Team[i]
		if strings.EqualFold(teamMon.Species, species) || strings.EqualFold(teamMon.Name, species) {
			return simulator.ActionSwitchBase + i, true
		}
	}
	return -1, false
}

func mapReplayMoveSlot(state *simulator.BattleState, moveName string) (int, bool) {
	active := state.P1.GetActive()
	if active == nil || active.Fainted {
		return -1, false
	}

	needle := normalizeReplayMoveName(moveName)
	known := active.NumMoves
	if known < 0 {
		known = 0
	}
	if known > 4 {
		known = 4
	}

	for i := 0; i < known; i++ {
		if normalizeReplayMoveName(active.Moves[i]) == needle {
			return i, true
		}
	}

	// First-seen move fallback.
	if known < 4 && needle != "" {
		return known, true
	}
	return -1, false
}

func findNextMoveInTurn(events []parser.Event, startIdx int, player string) int {
	if startIdx < 0 || startIdx >= len(events) {
		return -1
	}
	turn := events[startIdx].Turn
	for i := startIdx + 1; i < len(events); i++ {
		if events[i].Turn != turn {
			break
		}
		if events[i].Player == player && events[i].Type == "move" {
			return i
		}
	}
	return -1
}

func mapReplayEventChosenAction(replay *parser.Replay, state *simulator.BattleState, eventIdx int) (int, bool) {
	if eventIdx < 0 || eventIdx >= len(replay.Events) {
		return -1, false
	}
	event := replay.Events[eventIdx]
	if event.Player != "p1" {
		return -1, false
	}

	switch event.Type {
	case "switch":
		return mapReplaySwitchAction(state, event)
	case "move":
		slot, ok := mapReplayMoveSlot(state, event.Value)
		if !ok {
			return -1, false
		}
		return simulator.ActionMove1 + slot, true
	case "terastallize":
		moveIdx := findNextMoveInTurn(replay.Events, eventIdx, "p1")
		if moveIdx == -1 {
			return -1, false
		}
		slot, ok := mapReplayMoveSlot(state, replay.Events[moveIdx].Value)
		if !ok {
			return -1, false
		}
		return simulator.ActionTeraMove1 + slot, true
	default:
		return -1, false
	}
}

func runImportCommand(rawURL string) error {
	// Normalize URL: strip trailing slash, extract replay ID
	rawURL = strings.TrimRight(rawURL, "/")

	// Extract the replay ID from the URL
	// Supports: https://replay.pokemonshowdown.com/gen9ou-1234567890
	//           gen9ou-1234567890
	replayID := rawURL
	if strings.Contains(rawURL, "pokemonshowdown.com/") {
		parts := strings.Split(rawURL, "pokemonshowdown.com/")
		replayID = parts[len(parts)-1]
	}
	replayID = strings.TrimSuffix(replayID, ".log")

	fmt.Printf("Importing replay: %s\n", replayID)

	// Download the log
	logURL := fmt.Sprintf("https://replay.pokemonshowdown.com/%s.log", replayID)
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Get(logURL)
	if err != nil {
		return fmt.Errorf("failed to download replay: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("replay not found (HTTP %d): %s", resp.StatusCode, logURL)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	// Save to data/replays/
	os.MkdirAll("data/replays", 0755)
	filePath := path.Join("data/replays", replayID+".log")
	if err := os.WriteFile(filePath, body, 0644); err != nil {
		return fmt.Errorf("failed to save replay: %w", err)
	}
	fmt.Printf("Saved to: %s\n", filePath)

	// Parse the replay
	replay, err := parser.ParseLogFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to parse replay: %w", err)
	}

	fmt.Printf("\n=== %s vs %s ===\n", replay.P1, replay.P2)
	fmt.Printf("Format: %s | Turns: %d | Winner: %s\n", replay.Tier, replay.Turns, replay.Winner)
	if replay.P1Rating > 0 || replay.P2Rating > 0 {
		fmt.Printf("Ratings: %s (%d) vs %s (%d)\n", replay.P1, replay.P1Rating, replay.P2, replay.P2Rating)
	}

	// Show teams
	fmt.Printf("\n%s's team: ", replay.P1)
	for species := range replay.Teams["p1"] {
		fmt.Printf("%s ", species)
	}
	fmt.Printf("\n%s's team: ", replay.P2)
	for species := range replay.Teams["p2"] {
		fmt.Printf("%s ", species)
	}
	fmt.Println()

	// Turn-by-turn evaluation
	fmt.Printf("\n--- Turn-by-Turn Win Probability (P1: %s) ---\n", replay.P1)
	fmt.Printf("%-6s  %-8s  %-40s\n", "Turn", "P1 Win%", "Bar")
	fmt.Println(strings.Repeat("-", 58))
	var history []float64
	for t := 1; t <= replay.Turns; t++ {
		state, err := simulator.FastForward(replay, t)
		if err != nil {
			continue
		}

		winProb, _ := bot.SearchEvaluate(state, 3)
		history = append(history, winProb)

		// Draw a visual bar
		barLen := 40
		filled := int(winProb * float64(barLen))
		if filled < 0 {
			filled = 0
		}
		if filled > barLen {
			filled = barLen
		}
		bar := strings.Repeat("█", filled) + strings.Repeat("░", barLen-filled)

		fmt.Printf("%-6d  %-8.1f  %s\n", t, winProb*100, bar)
	}

	// Final verdict
	fmt.Println()
	if len(history) > 0 {
		// Use moving average of last 3 turns for final verdict
		window := 10
		if len(history) < window {
			window = len(history)
		}
		sum := 0.0
		for i := len(history) - window; i < len(history); i++ {
			sum += history[i]
		}
		finalProb := sum / float64(window)

		predictedWinner := replay.P2
		if finalProb > 0.5 {
			predictedWinner = replay.P1
		}
		fmt.Printf("Search Engine Prediction (depth 4): %s wins (%.1f%% confidence)\n", predictedWinner, math.Abs(finalProb-0.5)*200)
		if predictedWinner == replay.Winner {
			fmt.Println("✅ Correct!")
		} else {
			fmt.Printf("❌ Wrong (actual winner: %s)\n", replay.Winner)
		}
	}

	return nil
}
