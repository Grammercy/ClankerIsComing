package gamedata

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// BaseStats holds the 6 base stats for a Pokemon species
type BaseStats struct {
	HP  int `json:"hp"`
	Atk int `json:"atk"`
	Def int `json:"def"`
	SpA int `json:"spa"`
	SpD int `json:"spd"`
	Spe int `json:"spe"`
}

// PokedexEntry holds the data we need from each Pokemon
type PokedexEntry struct {
	Name        string            `json:"name"`
	Types       []string          `json:"types"`
	BaseStats   BaseStats         `json:"baseStats"`
	Abilities   map[string]string `json:"abilities"`
	GenderRatio map[string]float64 `json:"genderRatio"`
	Weight      float64           `json:"weightkg"`
}

// Pokedex is the global lookup map: normalized name -> entry
var Pokedex map[string]*PokedexEntry

// normalizeName converts a species name to the Showdown key format:
// lowercase, strip spaces, hyphens, dots, apostrophes, colons
func normalizeName(name string) string {
	s := strings.ToLower(name)
	s = strings.ReplaceAll(s, " ", "")
	s = strings.ReplaceAll(s, "-", "")
	s = strings.ReplaceAll(s, ".", "")
	s = strings.ReplaceAll(s, "'", "")
	s = strings.ReplaceAll(s, "'", "")
	s = strings.ReplaceAll(s, ":", "")
	s = strings.ReplaceAll(s, "%", "")
	return s
}

// LoadPokedex parses the pokedex.json file into the global Pokedex map
func LoadPokedex(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read pokedex: %w", err)
	}

	raw := make(map[string]*PokedexEntry)
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("failed to parse pokedex: %w", err)
	}

	Pokedex = make(map[string]*PokedexEntry, len(raw))
	for key, entry := range raw {
		Pokedex[key] = entry
		// Also index by normalized display name for fuzzy matching
		normalized := normalizeName(entry.Name)
		if normalized != key {
			Pokedex[normalized] = entry
		}
	}

	// Inject "Unknown" species as a base 105 Normal-type for unrevealed Pokemon
	unknownEntry := &PokedexEntry{
		Name:  "Unknown",
		Types: []string{"Normal"},
		BaseStats: BaseStats{
			HP:  105,
			Atk: 105,
			Def: 105,
			SpA: 105,
			SpD: 105,
			Spe: 105,
		},
	}
	Pokedex["unknown"] = unknownEntry
	Pokedex["Unknown"] = unknownEntry

	fmt.Printf("Pokedex loaded: %d species indexed\n", len(Pokedex))
	return nil
}

// LookupSpecies finds a PokedexEntry by species name (case-insensitive, handles formes)
func LookupSpecies(species string) *PokedexEntry {
	if Pokedex == nil {
		return nil
	}

	// Try direct normalized lookup
	key := normalizeName(species)
	if entry, ok := Pokedex[key]; ok {
		return entry
	}

	// Try without forme suffix (e.g., "Charizard-Mega-X" -> "charizard")
	if idx := strings.Index(key, "mega"); idx > 0 {
		base := key[:idx]
		if entry, ok := Pokedex[base]; ok {
			return entry
		}
	}

	return nil
}
