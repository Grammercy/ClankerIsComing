# Pokemon Engine AI

A Go-based Pokemon Showdown engine with a live client, deterministic battle simulator, and MCTS search.

## Overview

This project provides an end-to-end workflow for:
- downloading and parsing Showdown replays,
- validating simulator action generation,
- evaluating positions with search,
- running a live bot against Showdown.

The evaluator is currently deterministic and non-neural:
- terminal states return `1.0` (P1 win), `0.0` (P1 loss), or `0.5` (draw),
- non-terminal leaf states return `0.5`.

## Architecture

- `client/`: Showdown websocket client and live battle state tracking.
- `simulator/`: offline battle mechanics and legal action generation.
- `bot/`: MCTS search and move selection.
- `evaluator/`: deterministic leaf evaluation utilities.
- `parser/`: replay log parser.
- `scraper/`: replay downloader.
- `gamedata/`: static game datasets.
- `main.go`: CLI entry point.

## CLI Commands

Run with `go run . -cmd <name>`.

- `scrape`: Download replays from Showdown.
- `parse`: Parse replay logs and print summaries.
- `actions`: Print valid actions for a replay turn.
- `verify-actions`: Compare simulator action availability against replay actions.
- `evaluate`: Evaluate a replay position at a turn.
- `bulk-evaluate`: Evaluate many replay positions.
- `search-evaluate`: Run search-based bulk evaluation.
- `import`: Download + parse a single replay URL/ID.
- `live`: Run the bot on Pokemon Showdown.

## Build

- Local build: `make build`
- Tests: `make test`
- Windows build: `make windows`
