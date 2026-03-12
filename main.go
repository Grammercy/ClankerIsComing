package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/pokemon-engine/arena"
	"github.com/pokemon-engine/bot"
	"github.com/pokemon-engine/client"
	"github.com/pokemon-engine/deepcfr"
	"github.com/pokemon-engine/evaluator"
	"github.com/pokemon-engine/gamedata"
	"github.com/pokemon-engine/neuralv2"
	"github.com/pokemon-engine/parser"
	"github.com/pokemon-engine/scraper"
	"github.com/pokemon-engine/simulator"
)

func main() {
	cmd := flag.String("cmd", "", "Command to run: 'scrape', 'parse', 'actions', 'verify-actions', 'evaluate', 'bulk-evaluate', 'search-evaluate', 'train-neuralv2', 'neural-evaluate', 'eval-latency', 'eval-elo', 'selfplay-generate', 'export-onnx', 'import', 'live'")
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
	engineName := flag.String("engine", "mcts", "Decision engine for live play: 'mcts' or 'neuralv2'")
	modelPath := flag.String("model", "data/deepcfr_model.json", "Path to a model file (Deep CFR JSON or ONNX for neuralv2)")
	epochs := flag.Int("epochs", 1, "Training epochs for train-neuralv2")
	maxFiles := flag.Int("max-files", 0, "Optional cap on replay files used for train-neuralv2")
	beliefSamples := flag.Int("belief-samples", 8, "Belief samples for neuralv2 training/evaluation")
	trainAccelerator := flag.String("train-accelerator", "auto", "Training accelerator for train-neuralv2: 'auto', 'cpu', or 'opencl'")
	trainBatchSize := flag.Int("train-batch-size", 512, "Mini-batch size for GPU-backed Deep CFR training")
	targetWorkers := flag.Int("target-workers", 0, "Parallel workers for CPU target generation in train-neuralv2 (0 = auto)")
	targetPredictor := flag.String("target-predictor", "auto", "Target-generation predictor backend for train-neuralv2: 'auto', 'cpu', or 'opencl'")
	targetBatchSize := flag.Int("target-batch-size", 2048, "Micro-batch size for target predictor requests")
	targetQueueSize := flag.Int("target-queue-size", 8192, "Buffered queue size for target predictor requests")
	snapshotFilesPerSync := flag.Int("snapshot-files-per-sync", 16, "Replay files processed per target snapshot before predictor refresh")
	trainProfileJSON := flag.String("train-profile-json", "", "Optional JSON output path for training stage/profile timings")
	openclPlatform := flag.String("opencl-platform", "", "Optional OpenCL platform name substring for train-neuralv2")
	openclDevice := flag.String("opencl-device", "", "Optional OpenCL device name substring for train-neuralv2")
	moveTimeMs := flag.Int("move-time-ms", int(client.SearchMoveTime/time.Millisecond), "Per-move time budget in milliseconds (live command)")
	engineA := flag.String("engine-a", "neuralv2", "Engine A for eval-elo/selfplay-generate: 'mcts' or 'neuralv2'")
	engineB := flag.String("engine-b", "mcts", "Engine B for eval-elo/selfplay-generate: 'mcts' or 'neuralv2'")
	modelAPath := flag.String("model-a", "data/deepcfr_model.json", "Model path for engine A")
	modelBPath := flag.String("model-b", "data/deepcfr_model.json", "Model path for engine B")
	latencySamples := flag.Int("latency-samples", 100, "Number of decision states to time in eval-latency")
	matches := flag.Int("matches", 100, "Number of matches for eval-elo/selfplay-generate")
	maxTurns := flag.Int("max-turns", 24, "Turn cap for eval-elo/selfplay-generate rollouts")
	outFile := flag.String("out-file", "data/selfplay/selfplay.jsonl", "Output file for selfplay-generate or export-onnx")
	onnxOutPath := flag.String("onnx-out", "data/neuralv2_model.onnx", "Output ONNX path for export-onnx")
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
		fmt.Println("  train-neuralv2   Bootstrap neuralv2 training (non-CUDA path)")
		fmt.Println("  neural-evaluate  Evaluate a replay decision state with the neuralv2 engine")
		fmt.Println("  eval-latency     Benchmark per-decision latency on replay states")
		fmt.Println("  eval-elo         Run head-to-head Elo evaluation between two engines")
		fmt.Println("  selfplay-generate Generate self-play rollout traces as JSONL")
		fmt.Println("  export-onnx      Export Deep CFR JSON weights to ONNX (CPU/ROCm workflow)")
		fmt.Println("  import           Download, parse, and analyze a specific Showdown replay URL")
		fmt.Println("  live             Run the bot live on Pokemon Showdown (supports -engine neuralv2)")
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
	case "train-neuralv2":
		if err := gamedata.LoadPokedex("data/pokedex.json"); err != nil {
			fmt.Printf("Warning: Pokedex not loaded: %v\n", err)
		}
		if err := gamedata.LoadMovedex("data/moves.json"); err != nil {
			fmt.Printf("Warning: Movedex not loaded: %v\n", err)
		}
		err := runTrainNeuralV2Command(*inDir, *modelPath, *epochs, *maxFiles, *beliefSamples, *depth, *trainAccelerator, *trainBatchSize, *targetWorkers, *targetPredictor, *targetBatchSize, *targetQueueSize, *snapshotFilesPerSync, *trainProfileJSON, *openclPlatform, *openclDevice)
		if err != nil {
			fmt.Printf("Neuralv2 training failed: %v\n", err)
			os.Exit(1)
		}
	case "neural-evaluate":
		if *file == "" {
			fmt.Println("Error: --file is required for the 'neural-evaluate' command.")
			os.Exit(1)
		}
		if err := gamedata.LoadPokedex("data/pokedex.json"); err != nil {
			fmt.Printf("Warning: Pokedex not loaded: %v\n", err)
		}
		if err := gamedata.LoadMovedex("data/moves.json"); err != nil {
			fmt.Printf("Warning: Movedex not loaded: %v\n", err)
		}
		err := runNeuralEvaluateCommand(*file, *turn, *player, *modelPath, *beliefSamples, *depth, time.Duration(*moveTimeMs)*time.Millisecond)
		if err != nil {
			fmt.Printf("Neuralv2 evaluation failed: %v\n", err)
			os.Exit(1)
		}
	case "eval-latency":
		if err := gamedata.LoadPokedex("data/pokedex.json"); err != nil {
			fmt.Printf("Warning: Pokedex not loaded: %v\n", err)
		}
		if err := gamedata.LoadMovedex("data/moves.json"); err != nil {
			fmt.Printf("Warning: Movedex not loaded: %v\n", err)
		}
		err := runEvalLatencyCommand(*inDir, *engineA, *modelAPath, *latencySamples, *beliefSamples, *depth, time.Duration(*moveTimeMs)*time.Millisecond)
		if err != nil {
			fmt.Printf("Latency evaluation failed: %v\n", err)
			os.Exit(1)
		}
	case "eval-elo":
		if err := gamedata.LoadPokedex("data/pokedex.json"); err != nil {
			fmt.Printf("Warning: Pokedex not loaded: %v\n", err)
		}
		if err := gamedata.LoadMovedex("data/moves.json"); err != nil {
			fmt.Printf("Warning: Movedex not loaded: %v\n", err)
		}
		err := runEvalEloCommand(*inDir, *engineA, *modelAPath, *engineB, *modelBPath, *matches, *maxTurns, *beliefSamples, *depth, time.Duration(*moveTimeMs)*time.Millisecond)
		if err != nil {
			fmt.Printf("Elo evaluation failed: %v\n", err)
			os.Exit(1)
		}
	case "selfplay-generate":
		if err := gamedata.LoadPokedex("data/pokedex.json"); err != nil {
			fmt.Printf("Warning: Pokedex not loaded: %v\n", err)
		}
		if err := gamedata.LoadMovedex("data/moves.json"); err != nil {
			fmt.Printf("Warning: Movedex not loaded: %v\n", err)
		}
		err := runSelfPlayGenerateCommand(*inDir, *engineA, *modelAPath, *engineB, *modelBPath, *matches, *maxTurns, *beliefSamples, *depth, time.Duration(*moveTimeMs)*time.Millisecond, *outFile)
		if err != nil {
			fmt.Printf("Self-play generation failed: %v\n", err)
			os.Exit(1)
		}
	case "export-onnx":
		err := runExportONNXCommand(*modelPath, *onnxOutPath)
		if err != nil {
			fmt.Printf("ONNX export failed: %v\n", err)
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

func runTrainNeuralV2Command(inDir string, modelPath string, epochs int, maxFiles int, beliefSamples int, depth int, accelerator string, batchSize int, targetWorkers int, targetPredictor string, targetBatchSize int, targetQueueSize int, snapshotFilesPerSync int, trainProfilePath string, openclPlatform string, openclDevice string) error {
	fmt.Printf("Training neuralv2 bootstrap model from %s\n", inDir)
	fmt.Println("Press Ctrl+C or type 'q' then Enter to stop early and save a checkpoint.")
	if strings.EqualFold(accelerator, "cuda") {
		return fmt.Errorf("cuda backend is not supported; use cpu, opencl, or auto")
	}

	stopCh := make(chan struct{})
	var stopOnce sync.Once
	requestStop := func(reason string) {
		stopOnce.Do(func() {
			fmt.Printf("\nStop requested (%s). Finishing current step before checkpointing...\n", reason)
			close(stopCh)
		})
	}

	signalCh := make(chan os.Signal, 1)
	signal.Notify(signalCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(signalCh)
	go func() {
		<-signalCh
		requestStop("signal")
	}()

	if info, err := os.Stdin.Stat(); err == nil && (info.Mode()&os.ModeCharDevice) != 0 {
		go func() {
			scanner := bufio.NewScanner(os.Stdin)
			for scanner.Scan() {
				text := strings.TrimSpace(strings.ToLower(scanner.Text()))
				if text == "q" || text == "quit" || text == "exit" || text == "stop" {
					requestStop("keyboard")
					return
				}
			}
		}()
	}

	model, stats, err := deepcfr.TrainFromReplayDir(deepcfr.TrainConfig{
		ReplayDir:            inDir,
		ModelPath:            modelPath,
		Epochs:               epochs,
		MaxFiles:             maxFiles,
		Seed:                 1,
		BeliefSamples:        beliefSamples,
		OpponentSamples:      3,
		TargetDepth:          depth,
		LearningRate:         0.0005,
		ReplayPolicyBlend:    0.35,
		ChosenActionBonus:    0.10,
		ProgressWriter:       os.Stdout,
		ProgressInterval:     10 * time.Second,
		Accelerator:          accelerator,
		BatchSize:            batchSize,
		TargetWorkers:        targetWorkers,
		TargetPredictor:      targetPredictor,
		TargetBatchSize:      targetBatchSize,
		TargetQueueSize:      targetQueueSize,
		SnapshotFilesPerSync: snapshotFilesPerSync,
		TrainProfilePath:     trainProfilePath,
		OpenCLPlatform:       openclPlatform,
		OpenCLDevice:         openclDevice,
		StopChan:             stopCh,
	})
	if err != nil {
		return err
	}
	if stats.Interrupted {
		fmt.Printf("Training interrupted early. Saved checkpoint to %s\n", modelPath)
	} else {
		fmt.Printf("Saved model to %s\n", modelPath)
	}
	fmt.Printf("Backend: %s\n", stats.Backend)
	fmt.Printf("Files Seen: %d\n", stats.FilesSeen)
	fmt.Printf("Positions: %d\n", stats.Positions)
	fmt.Printf("Average Loss: %.6f\n", stats.AverageLoss)
	fmt.Printf("Average Value MAE: %.6f\n", stats.AverageValueMAE)
	fmt.Printf("Average Policy CE: %.6f\n", stats.AveragePolicyCE)
	if strings.TrimSpace(trainProfilePath) != "" {
		fmt.Printf("Training profile JSON: %s\n", trainProfilePath)
	}
	if model != nil && model.Priors != nil {
		fmt.Printf("Species Priors Learned: %d\n", len(model.Priors.SpeciesCounts))
	}
	return nil
}

func runNeuralEvaluateCommand(filePath string, targetTurn int, playerID string, modelPath string, beliefSamples int, depth int, moveBudget time.Duration) error {
	replay, err := parser.ParseLogFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to parse replay: %w", err)
	}
	model, err := neuralv2.LoadModel(neuralv2.LoadConfig{
		Path: modelPath,
	})
	if err != nil {
		return err
	}
	defer model.Close()

	state, actualAction, hasActual := deepcfr.DecisionStateForTurn(replay, targetTurn, playerID)
	result := model.Evaluate(state, neuralv2.SearchConfig{
		BeliefSamples:   beliefSamples,
		OpponentSamples: 3,
		Depth:           depth,
		TimeBudget:      minDuration(moveBudget, 650*time.Millisecond),
		MaxSimulations:  96,
		TopK:            6,
	})

	fmt.Printf("Neuralv2 evaluation for %s turn %d (%s perspective)\n", filePath, targetTurn, playerID)
	fmt.Printf("Backend: %s\n", model.BackendName())
	fmt.Printf("Win Probability: %.2f%%\n", result.WinProbability*100.0)
	fmt.Printf("Recommended Action: %s\n", bot.ActionToString(state, &state.P1, result.BestAction))
	if hasActual {
		fmt.Printf("Replay Action: %s\n", bot.ActionToString(state, &state.P1, actualAction))
	}
	fmt.Printf("Simulations: %d | Latency: %s\n", result.Simulations, result.Latency.Round(time.Millisecond))

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

type mctsArenaAgent struct {
	depth int
	sims  int
}

func (a *mctsArenaAgent) Name() string {
	return "mcts"
}

func (a *mctsArenaAgent) Choose(state *simulator.BattleState) int {
	depth := a.depth
	if depth <= 0 {
		depth = 2
	}
	result := bot.SearchBestMoveWithSims(state, depth, a.sims)
	return result.BestAction
}

type neuralArenaAgent struct {
	model *neuralv2.Model
	cfg   neuralv2.SearchConfig
}

func (a *neuralArenaAgent) Name() string {
	return "neuralv2/" + a.model.BackendName()
}

func (a *neuralArenaAgent) Choose(state *simulator.BattleState) int {
	result := a.model.Evaluate(state, a.cfg)
	return result.BestAction
}

func buildArenaAgent(engineName string, modelPath string, beliefSamples int, depth int, moveBudget time.Duration) (arena.Agent, func(), error) {
	switch strings.ToLower(strings.TrimSpace(engineName)) {
	case "mcts":
		return &mctsArenaAgent{depth: depth, sims: 0}, func() {}, nil
	case "neuralv2":
		model, err := neuralv2.LoadModel(neuralv2.LoadConfig{
			Path: modelPath,
		})
		if err != nil {
			return nil, nil, err
		}
		return &neuralArenaAgent{
			model: model,
			cfg: neuralv2.SearchConfig{
				BeliefSamples:   beliefSamples,
				OpponentSamples: 3,
				Depth:           depth,
				TimeBudget:      minDuration(moveBudget, 650*time.Millisecond),
				MaxSimulations:  96,
				TopK:            6,
			},
		}, func() { _ = model.Close() }, nil
	default:
		return nil, nil, fmt.Errorf("unknown engine %q (expected mcts or neuralv2)", engineName)
	}
}

func collectDecisionStates(inDir string, maxStates int) ([]simulator.BattleState, error) {
	if maxStates <= 0 {
		maxStates = 1
	}
	entries, err := os.ReadDir(inDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read replay dir: %w", err)
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Name() < entries[j].Name()
	})

	states := make([]simulator.BattleState, 0, maxStates)
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".log") {
			continue
		}
		replay, err := parser.ParseLogFile(path.Join(inDir, entry.Name()))
		if err != nil {
			continue
		}
		for turn := 1; turn <= replay.Turns; turn++ {
			state, _, ok := deepcfr.DecisionStateForTurn(replay, turn, "p1")
			if !ok || state == nil {
				continue
			}
			states = append(states, *state)
			if len(states) >= maxStates {
				return states, nil
			}
		}
	}
	if len(states) == 0 {
		return nil, fmt.Errorf("no decision states found in %s", inDir)
	}
	return states, nil
}

func runEvalLatencyCommand(inDir string, engineName string, modelPath string, sampleCount int, beliefSamples int, depth int, moveBudget time.Duration) error {
	if sampleCount <= 0 {
		sampleCount = 100
	}
	states, err := collectDecisionStates(inDir, sampleCount*2)
	if err != nil {
		return err
	}
	agent, cleanup, err := buildArenaAgent(engineName, modelPath, beliefSamples, depth, moveBudget)
	if err != nil {
		return err
	}
	defer cleanup()

	rng := rand.New(rand.NewSource(1))
	latencies := make([]time.Duration, 0, sampleCount)
	invalid := 0
	for i := 0; i < sampleCount; i++ {
		state := states[rng.Intn(len(states))]
		start := time.Now()
		action := agent.Choose(&state)
		latencies = append(latencies, time.Since(start))
		if action < 0 {
			invalid++
		}
	}

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	total := time.Duration(0)
	for _, d := range latencies {
		total += d
	}

	fmt.Printf("Latency benchmark (%s)\n", agent.Name())
	fmt.Printf("Samples: %d | Invalid Actions: %d\n", len(latencies), invalid)
	fmt.Printf("Mean: %s\n", (total / time.Duration(len(latencies))).Round(time.Microsecond))
	fmt.Printf("P50:  %s\n", percentileDuration(latencies, 0.50).Round(time.Microsecond))
	fmt.Printf("P95:  %s\n", percentileDuration(latencies, 0.95).Round(time.Microsecond))
	fmt.Printf("P99:  %s\n", percentileDuration(latencies, 0.99).Round(time.Microsecond))
	return nil
}

func runEvalEloCommand(inDir string, engineA string, modelAPath string, engineB string, modelBPath string, matches int, maxTurns int, beliefSamples int, depth int, moveBudget time.Duration) error {
	if matches <= 0 {
		matches = 100
	}
	states, err := collectDecisionStates(inDir, max(128, matches*2))
	if err != nil {
		return err
	}

	agentA, cleanupA, err := buildArenaAgent(engineA, modelAPath, beliefSamples, depth, moveBudget)
	if err != nil {
		return err
	}
	defer cleanupA()
	agentB, cleanupB, err := buildArenaAgent(engineB, modelBPath, beliefSamples, depth, moveBudget)
	if err != nil {
		return err
	}
	defer cleanupB()

	rng := rand.New(rand.NewSource(1))
	eloA, eloB := 1500.0, 1500.0
	k := 24.0
	winsA, winsB, draws := 0, 0, 0

	for i := 0; i < matches; i++ {
		start := states[rng.Intn(len(states))]

		res1 := arena.PlayFromState(start, agentA, agentB, arena.Config{MaxTurns: maxTurns})
		scoreA1 := scoreAsP1(res1.Winner)
		eloA, eloB = updateElo(eloA, eloB, scoreA1, k)
		switch scoreA1 {
		case 1:
			winsA++
		case 0:
			winsB++
		default:
			draws++
		}

		res2 := arena.PlayFromState(start, agentB, agentA, arena.Config{MaxTurns: maxTurns})
		scoreA2 := scoreAsP2(res2.Winner)
		eloA, eloB = updateElo(eloA, eloB, scoreA2, k)
		switch scoreA2 {
		case 1:
			winsA++
		case 0:
			winsB++
		default:
			draws++
		}
	}

	totalGames := matches * 2
	fmt.Printf("Elo Arena (%s vs %s)\n", agentA.Name(), agentB.Name())
	fmt.Printf("Games: %d | A Wins: %d | B Wins: %d | Draws: %d\n", totalGames, winsA, winsB, draws)
	fmt.Printf("Final Elo: %s = %.1f, %s = %.1f (delta %.1f)\n", agentA.Name(), eloA, agentB.Name(), eloB, eloA-eloB)
	return nil
}

type selfPlayRecord struct {
	Match    int     `json:"match"`
	Ply      int     `json:"ply"`
	Turn     int     `json:"turn"`
	P1Engine string  `json:"p1Engine"`
	P2Engine string  `json:"p2Engine"`
	P1Action int     `json:"p1Action"`
	P2Action int     `json:"p2Action"`
	Value    float64 `json:"value"`
	Winner   string  `json:"winner"`
}

func runSelfPlayGenerateCommand(inDir string, engineA string, modelAPath string, engineB string, modelBPath string, matches int, maxTurns int, beliefSamples int, depth int, moveBudget time.Duration, outPath string) error {
	if matches <= 0 {
		matches = 100
	}
	states, err := collectDecisionStates(inDir, max(128, matches))
	if err != nil {
		return err
	}

	agentA, cleanupA, err := buildArenaAgent(engineA, modelAPath, beliefSamples, depth, moveBudget)
	if err != nil {
		return err
	}
	defer cleanupA()
	agentB, cleanupB, err := buildArenaAgent(engineB, modelBPath, beliefSamples, depth, moveBudget)
	if err != nil {
		return err
	}
	defer cleanupB()

	if err := os.MkdirAll(path.Dir(outPath), 0755); err != nil {
		return fmt.Errorf("failed to create output dir: %w", err)
	}
	file, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer file.Close()

	enc := json.NewEncoder(file)
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < matches; i++ {
		start := states[rng.Intn(len(states))]
		result := arena.PlayFromState(start, agentA, agentB, arena.Config{MaxTurns: maxTurns})
		for ply, step := range result.Trace {
			record := selfPlayRecord{
				Match:    i + 1,
				Ply:      ply + 1,
				Turn:     step.Turn,
				P1Engine: agentA.Name(),
				P2Engine: agentB.Name(),
				P1Action: step.P1Action,
				P2Action: step.P2Action,
				Value:    step.Value,
				Winner:   result.Winner,
			}
			if err := enc.Encode(record); err != nil {
				return fmt.Errorf("failed to encode self-play record: %w", err)
			}
		}
	}

	fmt.Printf("Wrote %d self-play matches to %s\n", matches, outPath)
	return nil
}

func runExportONNXCommand(modelPath string, outPath string) error {
	if modelPath == "" {
		return fmt.Errorf("model path is required")
	}
	if !strings.HasSuffix(strings.ToLower(modelPath), ".json") {
		return fmt.Errorf("expected Deep CFR JSON model for export, got %s", modelPath)
	}
	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("failed to read model file %s: %w", modelPath, err)
	}
	if err := os.MkdirAll(path.Dir(outPath), 0755); err != nil {
		return fmt.Errorf("failed to create output dir: %w", err)
	}

	scriptPath := "tools/export_deepcfr_to_onnx.py"
	if _, err := os.Stat(scriptPath); err != nil {
		return fmt.Errorf("missing exporter script %s", scriptPath)
	}
	cmd := exec.Command("python3", scriptPath, "--in", modelPath, "--out", outPath)
	output, err := cmd.CombinedOutput()
	if len(output) > 0 {
		fmt.Print(string(output))
	}
	if err != nil {
		return fmt.Errorf("python exporter failed: %w", err)
	}
	fmt.Printf("Exported ONNX model to %s\n", outPath)
	return nil
}

func percentileDuration(samples []time.Duration, p float64) time.Duration {
	if len(samples) == 0 {
		return 0
	}
	if p <= 0 {
		return samples[0]
	}
	if p >= 1 {
		return samples[len(samples)-1]
	}
	idx := int(math.Ceil(float64(len(samples))*p)) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(samples) {
		idx = len(samples) - 1
	}
	return samples[idx]
}

func scoreAsP1(winner string) float64 {
	switch winner {
	case "p1":
		return 1
	case "p2":
		return 0
	default:
		return 0.5
	}
}

func scoreAsP2(winner string) float64 {
	switch winner {
	case "p2":
		return 1
	case "p1":
		return 0
	default:
		return 0.5
	}
}

func updateElo(eloA float64, eloB float64, scoreA float64, k float64) (float64, float64) {
	expectA := 1.0 / (1.0 + math.Pow(10, (eloB-eloA)/400.0))
	expectB := 1.0 - expectA
	scoreB := 1.0 - scoreA
	return eloA + k*(scoreA-expectA), eloB + k*(scoreB-expectB)
}

func minDuration(a time.Duration, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}

func max(a int, b int) int {
	if a > b {
		return a
	}
	return b
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
