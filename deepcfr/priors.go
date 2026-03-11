package deepcfr

import (
	"fmt"
	"io"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/pokemon-engine/gamedata"
	"github.com/pokemon-engine/parser"
	"github.com/pokemon-engine/simulator"
)

type Priors struct {
	SpeciesCounts  map[string]int            `json:"speciesCounts"`
	MoveCounts     map[string]map[string]int `json:"moveCounts"`
	ItemCounts     map[string]map[string]int `json:"itemCounts"`
	AbilityCounts  map[string]map[string]int `json:"abilityCounts"`
	TeraTypeCounts map[string]map[string]int `json:"teraTypeCounts"`
	LevelCounts    map[string]map[string]int `json:"levelCounts"`
}

func BuildPriorsFromReplayDir(dir string, maxFiles int) (*Priors, error) {
	priors, _, err := buildPriorsFromReplayDir(dir, maxFiles, nil, 0)
	return priors, err
}

type PriorsBuildStats struct {
	Candidates int
	Attempts   int
	Parsed     int
	Errors     int
	Elapsed    time.Duration
}

func BuildPriorsFromReplayDirWithProgress(dir string, maxFiles int, writer io.Writer, interval time.Duration) (*Priors, PriorsBuildStats, error) {
	return buildPriorsFromReplayDir(dir, maxFiles, writer, interval)
}

func buildPriorsFromReplayDir(dir string, maxFiles int, writer io.Writer, interval time.Duration) (*Priors, PriorsBuildStats, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, PriorsBuildStats{}, fmt.Errorf("failed to read replay dir: %w", err)
	}
	priors := &Priors{
		SpeciesCounts:  make(map[string]int),
		MoveCounts:     make(map[string]map[string]int),
		ItemCounts:     make(map[string]map[string]int),
		AbilityCounts:  make(map[string]map[string]int),
		TeraTypeCounts: make(map[string]map[string]int),
		LevelCounts:    make(map[string]map[string]int),
	}

	if interval <= 0 {
		interval = 10 * time.Second
	}
	start := time.Now()
	stats := PriorsBuildStats{}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".log") {
			continue
		}
		stats.Candidates++
	}
	if maxFiles > 0 && stats.Candidates > maxFiles {
		stats.Candidates = maxFiles
	}
	lastReport := time.Now().Add(-interval)
	if writer != nil {
		fmt.Fprintf(writer, "building priors from %d replay logs\n", stats.Candidates)
	}

	seen := 0
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".log") {
			continue
		}
		if maxFiles > 0 && seen >= maxFiles {
			break
		}
		stats.Attempts++
		if writer != nil {
			now := time.Now()
			if now.Sub(lastReport) >= interval {
				fmt.Fprintf(writer, "priors progress: %d/%d parsed, %d errors, current=%s\n", stats.Parsed, stats.Candidates, stats.Errors, entry.Name())
				lastReport = now
			}
		}

		replay, err := parser.ParseLogFile(dir + "/" + entry.Name())
		if err != nil {
			stats.Errors++
			continue
		}
		priors.ingestReplay(replay)
		seen++
		stats.Parsed = seen
	}
	stats.Elapsed = time.Since(start)
	if writer != nil {
		fmt.Fprintf(writer, "priors complete: %d/%d parsed, %d errors in %s\n", stats.Parsed, stats.Candidates, stats.Errors, stats.Elapsed.Round(time.Second))
	}
	return priors, stats, nil
}

func (p *Priors) CompleteState(state *simulator.BattleState, rng *rand.Rand) {
	if state == nil || p == nil {
		return
	}
	p.completePlayer(&state.P1, rng)
	p.completePlayer(&state.P2, rng)
}

func (p *Priors) completePlayer(player *simulator.PlayerState, rng *rand.Rand) {
	if player == nil {
		return
	}
	if player.TeamSize < 6 {
		player.TeamSize = 6
	}
	revealed := make(map[string]bool)
	for i := 0; i < player.TeamSize && i < len(player.Team); i++ {
		if player.Team[i].Species != "" && normalizeID(player.Team[i].Species) != "unknown" {
			revealed[normalizeID(player.Team[i].Species)] = true
		}
	}

	for i := 0; i < player.TeamSize && i < len(player.Team); i++ {
		poke := &player.Team[i]
		if poke.Species == "" || normalizeID(poke.Species) == "unknown" {
			species := p.sampleSpecies(rng, revealed)
			if species == "" {
				species = "Unknown"
			}
			player.Team[i] = defaultPokemon(species, p.mostLikelyLevel(species))
			player.Team[i].IsActive = poke.IsActive
			player.Team[i].Fainted = poke.Fainted
			player.Team[i].HP = max(player.Team[i].HP, poke.HP)
			player.Team[i].MaxHP = max(player.Team[i].MaxHP, poke.MaxHP)
			poke = &player.Team[i]
			revealed[normalizeID(species)] = true
		}

		if poke.MaxHP <= 0 {
			defaulted := defaultPokemon(poke.Species, p.mostLikelyLevel(poke.Species))
			defaulted.IsActive = poke.IsActive
			defaulted.Fainted = poke.Fainted
			defaulted.Status = poke.Status
			defaulted.NumMoves = poke.NumMoves
			for m := 0; m < 4; m++ {
				defaulted.Moves[m] = poke.Moves[m]
				defaulted.MovePP[m] = poke.MovePP[m]
				defaulted.MoveMaxPP[m] = poke.MoveMaxPP[m]
			}
			defaulted.Ability = poke.Ability
			defaulted.Item = poke.Item
			defaulted.TeraType = poke.TeraType
			player.Team[i] = defaulted
			poke = &player.Team[i]
		}

		if poke.Ability == "" {
			poke.Ability = p.mostLikelyValue(p.AbilityCounts[normalizeID(poke.Species)])
		}
		if poke.Item == "" {
			poke.Item = p.mostLikelyValue(p.ItemCounts[normalizeID(poke.Species)])
		}
		if poke.TeraType == "" {
			poke.TeraType = p.mostLikelyValue(p.TeraTypeCounts[normalizeID(poke.Species)])
		}
		p.fillMoves(poke)
	}
}

func (p *Priors) fillMoves(poke *simulator.PokemonState) {
	if poke == nil {
		return
	}
	existing := make(map[string]bool)
	for i := 0; i < poke.NumMoves; i++ {
		if poke.Moves[i] != "" {
			existing[normalizeID(poke.Moves[i])] = true
			if poke.MoveMaxPP[i] <= 0 {
				poke.MoveMaxPP[i] = defaultMovePP(poke.Moves[i])
			}
			if poke.MovePP[i] <= 0 {
				poke.MovePP[i] = poke.MoveMaxPP[i]
			}
		}
	}
	for _, moveID := range p.topMovesForSpecies(poke.Species, 4) {
		if poke.NumMoves >= 4 {
			break
		}
		if existing[normalizeID(moveID)] {
			continue
		}
		slot := poke.NumMoves
		poke.Moves[slot] = moveID
		poke.MoveMaxPP[slot] = defaultMovePP(moveID)
		poke.MovePP[slot] = poke.MoveMaxPP[slot]
		poke.NumMoves++
		existing[normalizeID(moveID)] = true
	}
}

func (p *Priors) topMovesForSpecies(species string, limit int) []string {
	counts := p.MoveCounts[normalizeID(species)]
	if len(counts) == 0 {
		return nil
	}
	type pair struct {
		key   string
		count int
	}
	pairs := make([]pair, 0, len(counts))
	for key, count := range counts {
		pairs = append(pairs, pair{key: key, count: count})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].count == pairs[j].count {
			return pairs[i].key < pairs[j].key
		}
		return pairs[i].count > pairs[j].count
	})
	if limit > len(pairs) {
		limit = len(pairs)
	}
	out := make([]string, 0, limit)
	for i := 0; i < limit; i++ {
		out = append(out, pairs[i].key)
	}
	return out
}

func (p *Priors) mostLikelyValue(counts map[string]int) string {
	bestKey := ""
	bestCount := -1
	for key, count := range counts {
		if count > bestCount || (count == bestCount && key < bestKey) {
			bestKey = key
			bestCount = count
		}
	}
	return bestKey
}

func (p *Priors) mostLikelyLevel(species string) int {
	counts := p.LevelCounts[normalizeID(species)]
	best := 80
	bestCount := -1
	for raw, count := range counts {
		level, err := strconv.Atoi(raw)
		if err != nil {
			continue
		}
		if count > bestCount {
			best = level
			bestCount = count
		}
	}
	return best
}

func (p *Priors) sampleSpecies(rng *rand.Rand, exclude map[string]bool) string {
	total := 0
	for species, count := range p.SpeciesCounts {
		if exclude[species] {
			continue
		}
		total += count
	}
	if total == 0 {
		return ""
	}
	if rng == nil {
		return p.mostLikelyAvailableSpecies(exclude)
	}
	target := rng.Intn(total)
	for species, count := range p.SpeciesCounts {
		if exclude[species] {
			continue
		}
		target -= count
		if target < 0 {
			return species
		}
	}
	return p.mostLikelyAvailableSpecies(exclude)
}

func (p *Priors) mostLikelyAvailableSpecies(exclude map[string]bool) string {
	best := ""
	bestCount := -1
	for species, count := range p.SpeciesCounts {
		if exclude[species] {
			continue
		}
		if count > bestCount {
			best = species
			bestCount = count
		}
	}
	return best
}

func (p *Priors) ingestReplay(replay *parser.Replay) {
	activeSpecies := make(map[string]string)
	for _, event := range replay.Events {
		switch event.Type {
		case "switch", "drag", "replace":
			species := normalizeID(event.Value)
			if species != "" {
				p.SpeciesCounts[species]++
				activeSpecies[event.Player] = species
			}
		case "move":
			species := activeSpecies[event.Player]
			if species == "" {
				continue
			}
			if p.MoveCounts[species] == nil {
				p.MoveCounts[species] = make(map[string]int)
			}
			p.MoveCounts[species][normalizeID(event.Value)]++
		}
	}

	identToSpecies := make(map[string]string)
	for _, line := range replay.RawLines {
		if !strings.HasPrefix(line, "|") {
			continue
		}
		parts := strings.Split(line, "|")
		if len(parts) < 2 {
			continue
		}
		switch parts[1] {
		case "switch", "drag", "replace":
			if len(parts) >= 4 {
				identToSpecies[parts[2]] = normalizeID(strings.TrimSpace(strings.Split(parts[3], ",")[0]))
				level := parseLevel(parts[3])
				species := identToSpecies[parts[2]]
				if species != "" && level > 0 {
					if p.LevelCounts[species] == nil {
						p.LevelCounts[species] = make(map[string]int)
					}
					p.LevelCounts[species][strconv.Itoa(level)]++
				}
			}
		case "-item", "-enditem":
			if len(parts) >= 4 {
				species := identToSpecies[parts[2]]
				if species != "" {
					if p.ItemCounts[species] == nil {
						p.ItemCounts[species] = make(map[string]int)
					}
					p.ItemCounts[species][normalizeID(parts[3])]++
				}
			}
		case "-ability":
			if len(parts) >= 4 {
				species := identToSpecies[parts[2]]
				if species != "" {
					if p.AbilityCounts[species] == nil {
						p.AbilityCounts[species] = make(map[string]int)
					}
					p.AbilityCounts[species][normalizeID(parts[3])]++
				}
			}
		case "-terastallize":
			if len(parts) >= 4 {
				species := identToSpecies[parts[2]]
				if species != "" {
					if p.TeraTypeCounts[species] == nil {
						p.TeraTypeCounts[species] = make(map[string]int)
					}
					p.TeraTypeCounts[species][parts[3]]++
				}
			}
		}
	}
}

func defaultPokemon(species string, level int) simulator.PokemonState {
	if level <= 0 {
		level = 80
	}
	maxHP := 100
	stats := simulator.Stats{
		HP:  100,
		Atk: 100,
		Def: 100,
		SpA: 100,
		SpD: 100,
		Spe: 100,
	}
	if entry := gamedata.LookupSpecies(species); entry != nil {
		maxHP = entry.BaseStats.HP*2 + 141
		stats = simulator.Stats{
			HP:  maxHP,
			Atk: entry.BaseStats.Atk*2 + 36,
			Def: entry.BaseStats.Def*2 + 36,
			SpA: entry.BaseStats.SpA*2 + 36,
			SpD: entry.BaseStats.SpD*2 + 36,
			Spe: entry.BaseStats.Spe*2 + 36,
		}
	}
	return simulator.PokemonState{
		Name:        species,
		Species:     species,
		Level:       level,
		HP:          maxHP,
		MaxHP:       maxHP,
		Stats:       stats,
		Boosts:      simulator.NeutralBoosts,
		NumMoves:    0,
		TeraType:    "",
		Fainted:     false,
		IsActive:    false,
		Ability:     "",
		Item:        "",
		TurnsActive: 0,
	}
}

func parseLevel(details string) int {
	parts := strings.Split(details, ",")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if strings.HasPrefix(part, "L") {
			level, err := strconv.Atoi(strings.TrimPrefix(part, "L"))
			if err == nil {
				return level
			}
		}
	}
	return 0
}

func defaultMovePP(moveID string) int {
	move := gamedata.LookupMove(moveID)
	if move == nil || move.PP <= 0 {
		return 16
	}
	return move.PP
}
