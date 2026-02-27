package parser

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Event represents a specific action in a turn
type Event struct {
	Turn    int
	Type    string // "switch", "move", "damage", "heal", "faint"
	Player  string // "p1" or "p2"
	Pokemon string // The nickname/ident of the Pokemon
	Value   string // The move name, species, or HP
	Detail  string // Original detail string
}

// Replay represents a parsed Pokemon Showdown replay
type Replay struct {
	P1         string
	P2         string
	P1Rating   int
	P2Rating   int
	Winner     string
	Tier       string
	Turns      int
	Events     []Event
	RawLines   []string
	Teams      map[string]map[string]bool            // player -> base species -> true
	KnownMoves map[string]map[string]map[string]bool // player -> pokemon -> map[move]bool
}

// ParseLogFile reads a Showdown replay log file and extracts information
func ParseLogFile(filePath string) (*Replay, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	replay := &Replay{
		Events:     make([]Event, 0),
		RawLines:   make([]string, 0),
		Teams:      map[string]map[string]bool{"p1": make(map[string]bool), "p2": make(map[string]bool)},
		KnownMoves: map[string]map[string]map[string]bool{"p1": make(map[string]map[string]bool), "p2": make(map[string]map[string]bool)},
	}

	scanner := bufio.NewScanner(file)
	// Some Showdown JSON requests at the end of files can be extremely long
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024) // up to 1MB lines

	currentTurn := 0

	// Map to track which identifier (e.g., "p1a: Pikachu") maps to which species ("Pikachu")
	activePokemonIdent := make(map[string]string)

	for scanner.Scan() {
		line := scanner.Text()
		replay.RawLines = append(replay.RawLines, line)

		if !strings.HasPrefix(line, "|") {
			continue // Skip empty or unrecognized lines
		}

		parts := strings.Split(line, "|")
		if len(parts) < 2 {
			continue
		}

		command := parts[1]

		switch command {
		case "poke":
			if len(parts) >= 4 {
				player := parts[2]
				species := strings.TrimSpace(strings.Split(parts[3], ",")[0])
				if replay.Teams[player] == nil {
					replay.Teams[player] = make(map[string]bool)
				}
				replay.Teams[player][species] = true
			}
		case "player":
			if len(parts) >= 4 && parts[3] != "" {
				if parts[2] == "p1" {
					if replay.P1 == "" {
						replay.P1 = parts[3]
					}
					if len(parts) >= 6 && parts[5] != "" {
						if rating, err := strconv.Atoi(parts[5]); err == nil {
							replay.P1Rating = rating
						}
					}
				} else if parts[2] == "p2" {
					if replay.P2 == "" {
						replay.P2 = parts[3]
					}
					if len(parts) >= 6 && parts[5] != "" {
						if rating, err := strconv.Atoi(parts[5]); err == nil {
							replay.P2Rating = rating
						}
					}
				}
			}
		case "tier":
			if len(parts) >= 3 {
				replay.Tier = parts[2]
			}
		case "turn":
			replay.Turns++
			currentTurn++
		case "win":
			if len(parts) >= 3 {
				replay.Winner = parts[2]
			}
		case "switch", "drag", "replace":
			if len(parts) >= 4 {
				player := parts[2][:2] // "p1a: Pikachu" -> "p1"
				identParts := strings.SplitN(parts[2], ": ", 2)
				pokemon := parts[2]
				if len(identParts) == 2 {
					pokemon = strings.TrimSpace(identParts[1])
				}

				species := strings.TrimSpace(strings.Split(parts[3], ",")[0])

				// Track the species for this identifier so moves can be mapped correctly
				activePokemonIdent[parts[2]] = species

				// Also track the benched identifier (p1: Nickname) in case of Revival Blessing
				if len(identParts) == 2 {
					benchIdent := fmt.Sprintf("%s: %s", player, pokemon)
					activePokemonIdent[benchIdent] = species
				}

				replay.Events = append(replay.Events, Event{
					Turn:    currentTurn,
					Type:    command,
					Player:  player,
					Pokemon: pokemon,
					Value:   species,
					Detail:  fmt.Sprintf("%s switched to %s", parts[2], parts[3]),
				})
			}
		case "move":
			if len(parts) >= 4 {
				player := parts[2][:2]
				identParts := strings.SplitN(parts[2], ": ", 2)
				pokemon := parts[2]
				if len(identParts) == 2 {
					pokemon = strings.TrimSpace(identParts[1])
				}
				move := parts[3]

				// Try to map to the actual species if tracked, otherwise fallback to Ident/Pokemon
				species := pokemon
				if mappedSpecies, ok := activePokemonIdent[parts[2]]; ok {
					species = mappedSpecies
				}

				if replay.KnownMoves[player] == nil {
					replay.KnownMoves[player] = make(map[string]map[string]bool)
				}
				if replay.KnownMoves[player][species] == nil {
					replay.KnownMoves[player][species] = make(map[string]bool)
				}
				replay.KnownMoves[player][species][move] = true

				replay.Events = append(replay.Events, Event{
					Turn:    currentTurn,
					Type:    "move",
					Player:  player,
					Pokemon: pokemon,
					Value:   move,
					Detail:  fmt.Sprintf("%s used %s", parts[2], parts[3]),
				})
			}
		case "-damage", "-heal":
			if len(parts) >= 4 {
				player := parts[2][:2]

				identParts := strings.SplitN(parts[2], ": ", 2)
				pokemon := parts[2]
				if len(identParts) == 2 {
					pokemon = strings.TrimSpace(identParts[1])
				}

				species := pokemon
				if mappedSpecies, ok := activePokemonIdent[parts[2]]; ok {
					species = mappedSpecies
				}

				replay.Events = append(replay.Events, Event{
					Turn:    currentTurn,
					Type:    command[1:], // "damage" or "heal"
					Player:  player,
					Pokemon: parts[2],
					Value:   species,
					Detail:  fmt.Sprintf("%s HP -> %s", parts[2], parts[3]),
				})
			}
		case "faint":
			if len(parts) >= 3 {
				player := parts[2][:2]
				identParts := strings.SplitN(parts[2], ": ", 2)
				pokemon := parts[2]
				if len(identParts) == 2 {
					pokemon = strings.TrimSpace(identParts[1])
				}

				replay.Events = append(replay.Events, Event{
					Turn:    currentTurn,
					Type:    "faint",
					Player:  player,
					Pokemon: pokemon,
					Detail:  fmt.Sprintf("%s fainted", parts[2]),
				})
			}
		case "-status":
			if len(parts) >= 4 {
				player := parts[2][:2]
				pokemon := parts[2]
				identParts := strings.SplitN(parts[2], ": ", 2)
				if len(identParts) == 2 {
					pokemon = strings.TrimSpace(identParts[1])
				}
				species := pokemon
				if mappedSpecies, ok := activePokemonIdent[parts[2]]; ok {
					species = mappedSpecies
				}
				replay.Events = append(replay.Events, Event{
					Turn:    currentTurn,
					Type:    "status",
					Player:  player,
					Pokemon: parts[2],
					Value:   species,
					Detail:  parts[3], // "brn", "par", etc.
				})
			}
		case "-curestatus":
			if len(parts) >= 4 {
				player := parts[2][:2]
				pokemon := parts[2]
				identParts := strings.SplitN(parts[2], ": ", 2)
				if len(identParts) == 2 {
					pokemon = strings.TrimSpace(identParts[1])
				}
				species := pokemon
				if mappedSpecies, ok := activePokemonIdent[parts[2]]; ok {
					species = mappedSpecies
				}
				replay.Events = append(replay.Events, Event{
					Turn:    currentTurn,
					Type:    "curestatus",
					Player:  player,
					Pokemon: parts[2],
					Value:   species,
					Detail:  parts[3],
				})
			}
		case "-weather":
			if len(parts) >= 3 {
				weather := parts[2]
				// Skip upkeep notifications
				if len(parts) >= 4 && strings.Contains(parts[3], "[upkeep]") {
					continue
				}
				replay.Events = append(replay.Events, Event{
					Turn:   currentTurn,
					Type:   "weather",
					Value:  weather,
					Detail: weather,
				})
			}
		case "-fieldstart":
			if len(parts) >= 3 {
				field := parts[2]
				// Extract terrain/trick room name from "move: Psychic Terrain"
				if strings.HasPrefix(field, "move: ") {
					field = strings.TrimPrefix(field, "move: ")
				}
				replay.Events = append(replay.Events, Event{
					Turn:   currentTurn,
					Type:   "fieldstart",
					Value:  field,
					Detail: field,
				})
			}
		case "-fieldend":
			if len(parts) >= 3 {
				field := parts[2]
				if strings.HasPrefix(field, "move: ") {
					field = strings.TrimPrefix(field, "move: ")
				}
				replay.Events = append(replay.Events, Event{
					Turn:   currentTurn,
					Type:   "fieldend",
					Value:  field,
					Detail: field,
				})
			}
		case "-sidestart":
			if len(parts) >= 4 {
				player := parts[2][:2]
				condition := parts[3]
				if strings.HasPrefix(condition, "move: ") {
					condition = strings.TrimPrefix(condition, "move: ")
				}
				replay.Events = append(replay.Events, Event{
					Turn:   currentTurn,
					Type:   "sidestart",
					Player: player,
					Value:  condition,
					Detail: condition,
				})
			}
		case "-sideend":
			if len(parts) >= 4 {
				player := parts[2][:2]
				condition := parts[3]
				if strings.HasPrefix(condition, "move: ") {
					condition = strings.TrimPrefix(condition, "move: ")
				}
				replay.Events = append(replay.Events, Event{
					Turn:   currentTurn,
					Type:   "sideend",
					Player: player,
					Value:  condition,
					Detail: condition,
				})
			}
		case "-boost", "-unboost":
			if len(parts) >= 5 {
				player := parts[2][:2]
				stat := parts[3]
				amount := parts[4]
				species := ""
				if mappedSpecies, ok := activePokemonIdent[parts[2]]; ok {
					species = mappedSpecies
				}
				replay.Events = append(replay.Events, Event{
					Turn:    currentTurn,
					Type:    command[1:], // "boost" or "unboost"
					Player:  player,
					Value:   species,
					Pokemon: stat,
					Detail:  amount,
				})
			}
		case "-setboost":
			if len(parts) >= 5 {
				player := parts[2][:2]
				stat := parts[3]
				amount := parts[4]
				species := ""
				if mappedSpecies, ok := activePokemonIdent[parts[2]]; ok {
					species = mappedSpecies
				}
				replay.Events = append(replay.Events, Event{
					Turn:    currentTurn,
					Type:    "setboost",
					Player:  player,
					Value:   species,
					Pokemon: stat,
					Detail:  amount,
				})
			}
		case "-clearallboost":
			if len(parts) >= 3 {
				player := parts[2][:2]
				replay.Events = append(replay.Events, Event{
					Turn:   currentTurn,
					Type:   "clearallboost",
					Player: player,
				})
			}
		case "-clearboost", "-clearnegativeboost", "-clearpositiveboost":
			if len(parts) >= 3 {
				player := parts[2][:2]
				replay.Events = append(replay.Events, Event{
					Turn:   currentTurn,
					Type:   command[1:], // "clearboost", etc.
					Player: player,
				})
			}
		case "-start":
			if len(parts) >= 4 {
				player := parts[2][:2]
				pokemon := parts[2]
				identParts := strings.SplitN(parts[2], ": ", 2)
				if len(identParts) == 2 {
					pokemon = strings.TrimSpace(identParts[1])
				}
				effect := parts[3]
				if strings.HasPrefix(effect, "move: ") {
					effect = strings.TrimPrefix(effect, "move: ")
				}
				replay.Events = append(replay.Events, Event{
					Turn:    currentTurn,
					Type:    "start",
					Player:  player,
					Pokemon: pokemon,
					Value:   effect,
					Detail:  line,
				})
			}
		case "-end":
			if len(parts) >= 3 {
				player := parts[2][:2]
				pokemon := parts[2]
				identParts := strings.SplitN(parts[2], ": ", 2)
				if len(identParts) == 2 {
					pokemon = strings.TrimSpace(identParts[1])
				}
				effect := parts[3]
				if strings.HasPrefix(effect, "move: ") {
					effect = strings.TrimPrefix(effect, "move: ")
				}
				replay.Events = append(replay.Events, Event{
					Turn:    currentTurn,
					Type:    "end",
					Player:  player,
					Pokemon: pokemon,
					Value:   effect,
					Detail:  line,
				})
			}
		case "-terastallize":
			if len(parts) >= 4 {
				player := parts[2][:2]
				pokemon := parts[2]
				identParts := strings.SplitN(parts[2], ": ", 2)
				if len(identParts) == 2 {
					pokemon = strings.TrimSpace(identParts[1])
				}
				replay.Events = append(replay.Events, Event{
					Turn:    currentTurn,
					Type:    "terastallize",
					Player:  player,
					Pokemon: pokemon,
					Value:   parts[3],
					Detail:  line,
				})
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	return replay, nil
}
