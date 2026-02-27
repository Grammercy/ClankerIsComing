package client

import (
	"github.com/pokemon-engine/evaluator"
	"github.com/pokemon-engine/simulator"
)

// ShowdownMove represents a single move option from the |request| JSON
type ShowdownMove struct {
	Move     string `json:"move"`
	ID       string `json:"id"`
	PP       int    `json:"pp"`
	MaxPP    int    `json:"maxpp"`
	Target   string `json:"target"`
	Disabled bool   `json:"disabled"`
}

// ShowdownActive represents the active Pokemon's move options
type ShowdownActive struct {
	Moves           []ShowdownMove `json:"moves"`
	CanTerastallize bool           `json:"canTerastallize"`
	// Trapped, canMegaEvo, etc. could go here
}

// ShowdownStats holds computed battle stats for a team member
type ShowdownStats struct {
	HP  int `json:"hp"`
	Atk int `json:"atk"`
	Def int `json:"def"`
	SpA int `json:"spa"`
	SpD int `json:"spd"`
	Spe int `json:"spe"`
}

// ShowdownPokemon represents a team member in the |request| JSON
type ShowdownPokemon struct {
	Ident         string        `json:"ident"`
	Details       string        `json:"details"`
	Condition     string        `json:"condition"` // "227/227" or "0 fnt"
	Active        bool          `json:"active"`
	Stats         ShowdownStats `json:"stats"`
	Moves         []string      `json:"moves"`
	BaseAbility   string        `json:"baseAbility"`
	Item          string        `json:"item"`
	Ability       string        `json:"ability"`
	TeraType      string        `json:"teraType"`
	Terastallized string        `json:"terastallized"`
}

// ShowdownSide represents our side's info
type ShowdownSide struct {
	Name    string            `json:"name"`
	ID      string            `json:"id"` // "p1" or "p2"
	Pokemon []ShowdownPokemon `json:"pokemon"`
}

// ShowdownRequest is the parsed |request| JSON from Showdown
type ShowdownRequest struct {
	Active      []ShowdownActive `json:"active"`
	Side        ShowdownSide     `json:"side"`
	Rqid        int              `json:"rqid"`
	ForceSwitch []bool           `json:"forceSwitch"` // If set, must switch (fainted)
	Wait        bool             `json:"wait"`        // If true, waiting for opponent
	TeamPreview bool             // Set manually when we detect team preview
}

// Experience records a single decision point for reinforcement learning
type Experience struct {
	Features [evaluator.TotalFeatures]float64
	Action   int // The action that was chosen (0=attack, 1-5=switch)
}

// BattleContext holds the live state for one active battle
type BattleContext struct {
	RoomID              string
	PlayerID            string // "p1" or "p2"
	OpponentID          string // the other player
	Request             *ShowdownRequest
	State               *simulator.BattleState
	IsOver              bool
	Experiences         []Experience                      // Our decisions for post-game learning
	OpponentExperiences []Experience                      // Opponent's observed decisions (learned on loss)
	LastFeatures        *[evaluator.TotalFeatures]float64 // State snapshot for pairing with opponent actions
}
