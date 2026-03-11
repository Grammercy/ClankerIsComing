# Pokemon Engine AI

A Go-based Pokemon Showdown engine with a live client, deterministic battle simulator, MCTS search, and a Deep CFR style imperfect-information engine for random battles.

## Overview

This project provides an end-to-end workflow for:
- downloading and parsing Showdown replays,
- validating simulator action generation,
- evaluating positions with search,
- running a live bot against Showdown.

The evaluator uses a hybrid of deterministic heuristics and neural evaluation:
- terminal states return `1.0` (P1 win), `0.0` (P1 loss), or `0.5` (draw),
- non-terminal leaf states use a lightweight board heuristic,
- the `deepcfr` package trains a belief-sampled policy/value model from replay logs.

## Architecture

- `client/`: Showdown websocket client and live battle state tracking.
- `simulator/`: offline battle mechanics and legal action generation.
- `bot/`: MCTS search and move selection.
- `deepcfr/`: replay training, belief completion, feature encoding, and imperfect-information search.
- `evaluator/`: deterministic leaf evaluation utilities.
- `parser/`: replay log parser.
- `scraper/`: replay downloader.
- `gamedata/`: static game datasets.
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
- `deep-evaluate`: Evaluate a replay decision state with the Deep CFR engine.
- `import`: Download + parse a single replay URL/ID.
- `live`: Run the bot on Pokemon Showdown with `-engine mcts` or `-engine deepcfr`.

## Build

- Local build: `make build-local`
- Tests: `make test`
- Windows build (CPU backend): `make build` or `make windows`
  - This target uses pure-Go (`CGO_ENABLED=0`) to avoid external MinGW/OpenCL runtime DLL dependencies.
- Windows build (OpenCL backend): `make windows-opencl` (requires cross OpenCL/CLBlast headers and libs discoverable via `x86_64-w64-mingw32-pkg-config`)
