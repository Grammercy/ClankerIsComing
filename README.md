# Pokemon Engine AI

A Go-based Pokemon Showdown engine with a live client, deterministic battle simulator, MCTS search, and neural imperfect-information engines for random battles.

## Overview

This project provides an end-to-end workflow for:
- downloading and parsing Showdown replays,
- validating simulator action generation,
- evaluating positions with search,
- running a live bot against Showdown.

The evaluator uses a hybrid of deterministic heuristics and neural evaluation:
- terminal states return `1.0` (P1 win), `0.0` (P1 loss), or `0.5` (draw),
- non-terminal leaf states use a lightweight board heuristic,
- the `deepcfr` package trains a belief-sampled policy/value model from replay logs,
- the `neuralv2` package provides a non-CUDA runtime abstraction (Deep CFR JSON backend today, ONNX target workflow).

## Architecture

- `client/`: Showdown websocket client and live battle state tracking.
- `simulator/`: offline battle mechanics and legal action generation.
- `bot/`: MCTS search and move selection.
- `deepcfr/`: replay training, belief completion, feature encoding, and imperfect-information search.
- `neuralv2/`: neuralv2 runtime model loading and time-budgeted evaluation.
- `arena/`: head-to-head rollout harness for Elo and self-play trace generation.
- `evaluator/`: deterministic leaf evaluation utilities.
- `parser/`: replay log parser.
- `scraper/`: replay downloader.
- `gamedata/`: static game datasets.
- `tools/`: utility scripts (including Deep CFR JSON -> ONNX export).
- `main.go`: CLI entry point.

## CLI Commands

Run with `go run . -cmd <name>`.
Windows binary usage follows the same flags, for example:
`.\pokemon-engine.exe -cmd parse -in data\replays`.

- `scrape`: Download replays from Showdown.
- `parse`: Parse replay logs and print summaries.
- `actions`: Print valid actions for a replay turn.
- `verify-actions`: Compare simulator action availability against replay actions.
- `evaluate`: Evaluate a replay position at a turn.
- `bulk-evaluate`: Evaluate many replay positions.
- `search-evaluate`: Run search-based bulk evaluation.
- `train-deepcfr`: Train the Deep CFR style model from replay logs.
  - For OpenCL runs, increase `-train-batch-size` (default `512`) to improve GPU occupancy.
  - Target generation is simulator-heavy and CPU-bound; tune `-target-workers` if CPU usage is too high.
- `train-neuralv2`: Bootstrap neuralv2 training through the non-CUDA Deep CFR trainer path.
- `deep-evaluate`: Evaluate a replay decision state with the Deep CFR engine.
- `neural-evaluate`: Evaluate a replay decision state with the neuralv2 engine.
- `eval-latency`: Benchmark per-decision latency on replay decision states.
- `eval-elo`: Run a head-to-head Elo arena between two engines.
- `selfplay-generate`: Generate JSONL self-play traces between two engines.
- `export-onnx`: Export a Deep CFR JSON checkpoint to ONNX using `tools/export_deepcfr_to_onnx.py`.
- `import`: Download + parse a single replay URL/ID.
- `live`: Run the bot on Pokemon Showdown with `-engine mcts`, `-engine deepcfr`, or `-engine neuralv2`.

## Build

- Local build: `make build-local`
- Tests: `make test`
- Windows build: `make windows` (or `make build`, alias)
  - Single Windows target builds with OpenCL enabled (`-tags opencl`) and is compatible with OpenCL drivers (including ROCm OpenCL stacks).
  - Uses self-contained linker flags for MinGW runtime libraries (`-static-libgcc`, `-static-libstdc++`).
  - OpenCL itself remains provided by the installed driver runtime (e.g. `OpenCL.dll` via vendor ICD/ROCm).
  - Requires cross OpenCL headers/libs discoverable via `x86_64-w64-mingw32-pkg-config`.

## ONNX Export

To export Deep CFR JSON weights for neuralv2 ONNX workflows:

`go run . -cmd export-onnx -model data/deepcfr_model.json -onnx-out data/neuralv2_model.onnx`

Requirements:
- `python3`
- `torch` with ONNX export support installed in the active Python environment
