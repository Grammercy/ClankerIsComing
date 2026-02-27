package gamedata

import (
	"encoding/json"
	"fmt"
	"os"
)

// MoveEntry holds the data we need from each move
type MoveEntry struct {
	Name           string          `json:"name"`
	BasePower      int             `json:"basePower"`
	Type           string          `json:"type"`
	Category       string          `json:"category"` // "Physical", "Special", "Status"
	Accuracy       any             `json:"accuracy"` // can be int, float64, or bool (true = always hits)
	PP             int             `json:"pp"`
	Priority       int             `json:"priority"`
	CritRatio      int             `json:"critRatio"`
	WillCrit       bool            `json:"willCrit"`
	Status         string          `json:"status"`
	Boosts         map[string]int  `json:"boosts"`
	Secondary      *MoveSecondary  `json:"secondary"`
	Secondaries    []MoveSecondary `json:"secondaries"`
	Drain          [2]int          `json:"drain"`
	Recoil         [2]int          `json:"recoil"`
	Healing        [2]int          `json:"heal"`
	VolatileStatus string          `json:"volatileStatus"`
	ForceSwitch    bool            `json:"forceSwitch"`
	Target         string          `json:"target"`
}

type MoveSecondary struct {
	Chance         int            `json:"chance"`
	Status         string         `json:"status"`
	VolatileStatus string         `json:"volatileStatus"`
	Boosts         map[string]int `json:"boosts"`
	Self           *MoveSelf      `json:"self"`
}

type MoveSelf struct {
	Boosts map[string]int `json:"boosts"`
}

// Movedex is the global lookup map: normalized ID -> entry
var Movedex map[string]*MoveEntry

// LoadMovedex parses the moves.json file into the global Movedex map
func LoadMovedex(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read movedex: %w", err)
	}

	raw := make(map[string]*MoveEntry)
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("failed to parse movedex: %w", err)
	}

	Movedex = make(map[string]*MoveEntry, len(raw))
	for key, entry := range raw {
		Movedex[key] = entry
		// Also index by normalized display name
		normalized := normalizeName(entry.Name)
		if normalized != key {
			Movedex[normalized] = entry
		}
	}

	fmt.Printf("Movedex loaded: %d moves indexed\n", len(Movedex))
	return nil
}

// LookupMove finds a MoveEntry by move name or ID (case-insensitive)
func LookupMove(name string) *MoveEntry {
	if Movedex == nil {
		return nil
	}
	key := NormalizeID(name)
	if entry, ok := Movedex[key]; ok {
		return entry
	}
	return nil
}
