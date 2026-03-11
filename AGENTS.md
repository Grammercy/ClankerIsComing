# Repository Guidelines

## Project Structure & Module Organization
This repository is a Go CLI for Pokemon Showdown tooling and battle simulation. Keep package boundaries clear:

- `main.go`: CLI entry point and command wiring.
- `client/`: live Showdown websocket client and battle state syncing.
- `simulator/`: battle rules, state transitions, and action generation.
- `bot/`: search logic and move selection.
- `evaluator/`, `parser/`, `scraper/`, `gamedata/`: evaluation, replay parsing, scraping, and static datasets.
- `data/`: local replay logs and JSON datasets such as `data/pokedex.json` and `data/moves.json`. This directory is ignored by Git.
- `reports/` and `gaps_report.md`: audits and simulator coverage notes.

## Build, Test, and Development Commands
- `make build`: cross-compile a Windows binary (`pokemon-engine.exe` by default).
- `make build-local`: build the local binary.
- `make test`: run the full Go test suite with `go test ./...`.
- `make windows`: cross-compile a Windows binary; requires `x86_64-w64-mingw32-gcc/g++`.
- `make windows-opencl`: cross-compile a Windows binary with OpenCL backend; requires cross OpenCL/CLBlast dev packages available to `x86_64-w64-mingw32-pkg-config`.
- `go run . -cmd parse -in data/replays`: parse downloaded replay logs.
- `go run . -cmd live -user <name> -pass <pass>`: run the live Showdown bot.

Use `go run . -cmd <name>` for CLI work; supported commands are documented in `README.md` and `main.go`.

## Coding Style & Naming Conventions
Format all Go code with `gofmt` before committing. Follow standard Go conventions:

- tabs for indentation, grouped imports, and lowercase package names.
- exported identifiers in `CamelCase`; unexported helpers in `camelCase`.
- keep files focused on one package concern; prefer descriptive names like `search.go`, `simulator_test.go`.

## Testing Guidelines
Tests use Go’s built-in `testing` package and live alongside code as `*_test.go`. Add or update tests in the same package you change, especially for simulator, client, and search behavior. Run `make test` before opening a PR. There is no enforced coverage threshold, but behavior changes should ship with regression coverage.

## Commit & Pull Request Guidelines
Recent history favors short, descriptive commit subjects, sometimes with a scope prefix like `docs:`. Prefer imperative messages such as `simulator: fix speed tie handling`.

Pull requests should include:

- a brief summary of the behavior change,
- linked issue or context when applicable,
- the commands you ran to validate the change,
- replay IDs, logs, or terminal snippets for simulator/client behavior changes.

## Security & Configuration Tips
Do not commit credentials, replay dumps you do not intend to share, or generated binaries. Keep Showdown credentials in local environment or shell history-safe workflows, not in source files.
