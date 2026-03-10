package client

import (
	"fmt"
	"log"
	"strconv"
	"strings"
	"time"

	"github.com/pokemon-engine/bot"
	"github.com/pokemon-engine/gamedata"
	"github.com/pokemon-engine/simulator"
)

func normalizeTypeName(typeName string) string {
	for _, t := range gamedata.AllTypes {
		if strings.EqualFold(t, typeName) {
			return t
		}
	}
	return ""
}

// ParseCondition parses "227/227" or "0 fnt" into (currentHP, maxHP, fainted)
func ParseCondition(cond string) (int, int, bool) {
	cond = strings.TrimSpace(cond)
	if cond == "0 fnt" {
		return 0, 100, true
	}

	// Strip status like " par", " brn", etc.
	parts := strings.Fields(cond)
	hpPart := parts[0]

	hpParts := strings.Split(hpPart, "/")
	if len(hpParts) == 2 {
		current, _ := strconv.Atoi(hpParts[0])
		max, _ := strconv.Atoi(hpParts[1])
		return current, max, current <= 0
	}

	// Handle single number (percentage)
	val, _ := strconv.Atoi(hpPart)
	return val, 100, val <= 0
}

// ParseStatus extracts the status condition from a condition string like "227/227 par"
func ParseStatus(cond string) string {
	parts := strings.Fields(cond)
	if len(parts) >= 2 && parts[1] != "fnt" {
		return parts[1]
	}
	return ""
}

// ParseDetails extracts species, level, and gender from a details string like "Ledian, L83, M"
func ParseDetails(details string) (string, int, string) {
	parts := strings.Split(details, ",")
	species := strings.TrimSpace(parts[0])
	level := 100
	gender := "N"

	for i := 1; i < len(parts); i++ {
		p := strings.TrimSpace(parts[i])
		if strings.HasPrefix(p, "L") {
			if l, err := strconv.Atoi(p[1:]); err == nil {
				level = l
			}
		} else if p == "M" || p == "F" {
			gender = p
		}
	}
	return species, level, gender
}

// RequestToBattleState converts a ShowdownRequest into a simulator.BattleState
// We only have full info about our side. The opponent side is unknown and will
// be filled with minimal defaults unless provided.
func RequestToBattleState(req *ShowdownRequest, currentBattleState *simulator.BattleState) *simulator.BattleState {
	state := &simulator.BattleState{
		Turn:     1,
		RNGState: uint64(time.Now().UnixNano()),
	}

	if currentBattleState != nil {
		state.Field = currentBattleState.Field
		state.Turn = currentBattleState.Turn
		state.RNGState = currentBattleState.RNGState
	}

	// Build our side
	var ourCurrentPlayer *simulator.PlayerState
	if currentBattleState != nil {
		ourCurrentPlayer = &currentBattleState.P1
	}
	ourPlayer := buildPlayerState(req.Side.Pokemon, req.Side.ID, ourCurrentPlayer)
	if len(req.ForceSwitch) > 0 {
		ourPlayer.ForceSwitch = req.ForceSwitch[0]
	}
	if len(req.Active) > 0 {
		ourPlayer.CanTerastallize = req.Active[0].CanTerastallize
	} else if ourCurrentPlayer != nil {
		ourPlayer.CanTerastallize = ourCurrentPlayer.CanTerastallize
	} else {
		ourPlayer.CanTerastallize = true
	}

	// Override active Pokemon's moves from the active move list (has canonical IDs)
	if len(req.Active) > 0 && ourPlayer.ActiveIdx >= 0 {
		active := &ourPlayer.Team[ourPlayer.ActiveIdx]
		active.NumMoves = 0
		for i, move := range req.Active[0].Moves {
			if i >= 4 {
				break
			}
			active.Moves[i] = move.ID
			active.NumMoves++
		}
	}

	// Build opponent side
	var oppPlayer simulator.PlayerState
	if currentBattleState != nil {
		oppPlayer = currentBattleState.P2
		// Ensure we assume a full team (matching our own size) even if not all revealed
		if oppPlayer.TeamSize < ourPlayer.TeamSize {
			oppPlayer.TeamSize = ourPlayer.TeamSize
		}
		// Ensure unknown slots are at least "Unknown" so they get base 105 stats
		for i := 0; i < oppPlayer.TeamSize; i++ {
			if oppPlayer.Team[i].Species == "" || oppPlayer.Team[i].Species == "Unknown" {
				oppPlayer.Team[i].Species = "Unknown"
				oppPlayer.Team[i].Name = "Unknown"
				if oppPlayer.Team[i].MaxHP == 0 {
					oppPlayer.Team[i].HP = 100
					oppPlayer.Team[i].MaxHP = 100
				}
			}
			// Sync active flag
			oppPlayer.Team[i].IsActive = (i == oppPlayer.ActiveIdx)
		}
	} else {
		oppID := "p2"
		if req.Side.ID == "p2" {
			oppID = "p1"
		}
		oppPlayer = simulator.PlayerState{
			ID:              oppID,
			TeamSize:        ourPlayer.TeamSize,
			ActiveIdx:       0,
			CanTerastallize: true,
		}
		for i := 0; i < oppPlayer.TeamSize; i++ {
			oppPlayer.Team[i] = simulator.PokemonState{
				Name:     "Unknown",
				Species:  "Unknown",
				HP:       100,
				MaxHP:    100,
				IsActive: (i == 0),
				Boosts:   simulator.NeutralBoosts,
			}
		}
	}

	// Always place our team as P1: the search engine maximizes P1's score,
	// so "us = P1" regardless of which side Showdown assigned us.
	state.P1 = ourPlayer
	state.P2 = oppPlayer

	return state
}

func buildPlayerState(pokemon []ShowdownPokemon, playerID string, currentPlayer *simulator.PlayerState) simulator.PlayerState {
	player := simulator.PlayerState{
		ID:        playerID,
		ActiveIdx: -1,
		TeamSize:  len(pokemon),
	}
	if currentPlayer != nil {
		player.Side = currentPlayer.Side
	}
	if player.TeamSize > 6 {
		player.TeamSize = 6
	}

	for i := 0; i < player.TeamSize; i++ {
		poke := pokemon[i]
		hp, maxHP, fainted := ParseCondition(poke.Condition)
		status := ParseStatus(poke.Condition)
		species, level, gender := ParseDetails(poke.Details)

		ps := simulator.PokemonState{
			Name:          species,
			Species:       species,
			TeraType:      normalizeTypeName(poke.TeraType),
			HP:            hp,
			MaxHP:         maxHP,
			Status:        status,
			IsActive:      poke.Active,
			Fainted:       fainted,
			Terastallized: poke.Terastallized != "",
			Boosts:        simulator.NeutralBoosts,
			Level:         level,
			Gender:        gender,
			Ability:       poke.Ability,
			Item:          poke.Item,
			Stats: simulator.Stats{
				HP:  poke.Stats.HP,
				Atk: poke.Stats.Atk,
				Def: poke.Stats.Def,
				SpA: poke.Stats.SpA,
				SpD: poke.Stats.SpD,
				Spe: poke.Stats.Spe,
			},
		}

		// Try to preserve boosts and volatiles from the current state if it's the same pokemon
		if currentPlayer != nil {
			for j := 0; j < currentPlayer.TeamSize; j++ {
				oldPoke := &currentPlayer.Team[j]
				if oldPoke.Species == ps.Species {
					ps.Boosts = oldPoke.Boosts
					ps.Volatiles = oldPoke.Volatiles
					ps.SleepTurns = oldPoke.SleepTurns
					ps.FreezeTurns = oldPoke.FreezeTurns
					ps.ToxicCounter = oldPoke.ToxicCounter
					if ps.TeraType == "" {
						ps.TeraType = oldPoke.TeraType
					}
					if !ps.Terastallized {
						ps.Terastallized = oldPoke.Terastallized
					}
					break
				}
			}
		}

		// Populate moves from the team data
		for j, moveID := range poke.Moves {
			if j >= 4 {
				break
			}
			ps.Moves[j] = moveID
			ps.NumMoves = j + 1
		}

		player.Team[i] = ps

		if poke.Active {
			player.ActiveIdx = i
		}
	}

	return player
}

// ChooseBestAction runs iterative deepening search and converts the result to a Showdown /choose command.
// Returns (choiceString, actionIndex, searchResult). actionIndex is -1 for non-combat decisions.
func ChooseBestAction(req *ShowdownRequest, moveTime time.Duration, currentBattleState *simulator.BattleState) (string, int, bot.SearchResult) {

	// Handle team preview
	if req.TeamPreview {
		return chooseTeamOrder(req), -1, bot.SearchResult{}
	}

	// Handle wait
	if req.Wait {
		return "", -1, bot.SearchResult{}
	}

	state := RequestToBattleState(req, currentBattleState)
	result := bot.IterativeDeepeningSearch(state, moveTime)

	return actionToShowdown(result.BestAction, req), result.BestAction, result
}

// DebugPrintState logs the internal state of both players for debugging
func DebugPrintState(roomID string, state *simulator.BattleState) {
	log.Printf("[%s] --- DEBUG STATE (Turn %d) ---", roomID, state.Turn)
	printPlayer(roomID, "P1 (Us)", &state.P1)
	printPlayer(roomID, "P2 (Opp)", &state.P2)
	log.Printf("[%s] -------------------------------", roomID)
}

func printPlayer(roomID, label string, p *simulator.PlayerState) {
	log.Printf("[%s] %s: ID=%s, ActiveIdx=%d, TeamSize=%d, ForceSwitch=%v",
		roomID, label, p.ID, p.ActiveIdx, p.TeamSize, p.ForceSwitch)
	for i := 0; i < p.TeamSize; i++ {
		poke := &p.Team[i]
		activeStr := ""
		if poke.IsActive {
			activeStr = "[ACTIVE]"
		}
		faintStr := ""
		if poke.Fainted {
			faintStr = "[FAINTED]"
		}
		log.Printf("[%s]   [%d] %s %d/%d %s %s Volatiles=%0x",
			roomID, i, poke.Species, poke.HP, poke.MaxHP, activeStr, faintStr, poke.Volatiles)
	}
}

// actionToShowdown converts our internal action index to Showdown's /choose format
func actionToShowdown(action int, req *ShowdownRequest) string {
	// Move actions (including tera move actions)
	if simulator.IsAttackAction(action) {
		moveIdx := simulator.BaseMoveIndex(action)
		if len(req.Active) > 0 && moveIdx >= 0 && moveIdx < len(req.Active[0].Moves) {
			move := req.Active[0].Moves[moveIdx]
			if !move.Disabled && move.PP > 0 {
				if simulator.IsTeraAction(action) && req.Active[0].CanTerastallize {
					return fmt.Sprintf("move %d terastallize", moveIdx+1)
				}
				return fmt.Sprintf("move %d", moveIdx+1) // Showdown 1-indexed
			}
		}
		// Fallback: pick first usable move
		if len(req.Active) > 0 {
			for i, move := range req.Active[0].Moves {
				if !move.Disabled && move.PP > 0 {
					if simulator.IsTeraAction(action) && req.Active[0].CanTerastallize {
						return fmt.Sprintf("move %d terastallize", i+1)
					}
					return fmt.Sprintf("move %d", i+1)
				}
			}
		}
		return "move 1"
	}

	// Switch action: ActionSwitchBase + team slot index
	switchTarget := action - simulator.ActionSwitchBase + 1 // 1-based switch target
	return fmt.Sprintf("switch %d", switchTarget)
}

// chooseTeamOrder returns the default team order for team preview
func chooseTeamOrder(req *ShowdownRequest) string {
	n := len(req.Side.Pokemon)
	order := make([]string, n)
	for i := range order {
		order[i] = strconv.Itoa(i + 1)
	}
	return "team " + strings.Join(order, "")
}
