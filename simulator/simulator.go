package simulator

import (
	"math"
	"strconv"
	"strings"

	"github.com/pokemon-engine/gamedata"
	"github.com/pokemon-engine/parser"
)

func GetStatShift(stat string) (uint32, bool) {
	switch stat {
	case "atk":
		return AtkShift, true
	case "def":
		return DefShift, true
	case "spa":
		return SpaShift, true
	case "spd":
		return SpdShift, true
	case "spe":
		return SpeShift, true
	case "eva", "evasion":
		return EvaShift, true
	case "acc", "accuracy":
		return AccShift, true
	}
	return 0, false
}

const (
	AtkShift  uint32 = 0
	DefShift  uint32 = 4
	SpaShift  uint32 = 8
	SpdShift  uint32 = 12
	SpeShift  uint32 = 16
	EvaShift  uint32 = 20
	AccShift  uint32 = 24
	BoostMask uint32 = 0xF
)

const NeutralBoosts uint32 = (6 << AtkShift) | (6 << DefShift) | (6 << SpaShift) | (6 << SpdShift) | (6 << SpeShift) | (6 << EvaShift) | (6 << AccShift)

const (
	VolatileSubstitute  uint32 = 1 << 0
	VolatileConfusion   uint32 = 1 << 1
	VolatileEncore      uint32 = 1 << 2
	VolatileTaunt       uint32 = 1 << 3
	VolatileLeechSeed   uint32 = 1 << 4
	VolatilePerishSong  uint32 = 1 << 5
	VolatileAttract     uint32 = 1 << 6
	VolatileFocusEnergy uint32 = 1 << 7
	VolatileDestinyBond uint32 = 1 << 8
	VolatileMagnetRise  uint32 = 1 << 9
	VolatileTelekinesis uint32 = 1 << 10
	VolatileGastroAcid  uint32 = 1 << 11
	VolatileIngrain     uint32 = 1 << 12
	VolatileAquaRing    uint32 = 1 << 13
	VolatileCurse       uint32 = 1 << 14
	VolatileEmbargo     uint32 = 1 << 15
	VolatileHealBlock   uint32 = 1 << 16
	VolatileProtection  uint32 = 1 << 17
)

// Stats holds the 6 primary battle stats
type Stats struct {
	HP  int `json:"hp"`
	Atk int `json:"atk"`
	Def int `json:"def"`
	SpA int `json:"spa"`
	SpD int `json:"spd"`
	Spe int `json:"spe"`
}

// PokemonState tracks the current battle state of a single Pokemon
type PokemonState struct {
	Name          string
	Species       string
	TeraType      string
	HP            int
	MaxHP         int
	Status        string // TODO: convert to enum/flag if needed
	Volatiles     uint32 // Bitfield for volatile statuses
	IsActive      bool
	Fainted       bool
	Terastallized bool
	Boosts        uint32    // atk, def, spa, spd, spe, eva, acc boosts (packed 4 bits each: 0-15, center is 6)
	Moves         [4]string // move IDs (Showdown format, e.g. "thunderbolt")
	NumMoves      int       // how many move slots are filled (0-4)

	// Full details for Pokedex compliance
	Level   int
	Gender  string // "M", "F", "N" (none)
	Nature  string
	Ability string
	Item    string
	EVs     Stats
	IVs     Stats
	Stats   Stats // Computed actual stats (before boosts/modifiers)
}

// GetBoost returns the boost value (-6 to +6) for a given shift
func (p *PokemonState) GetBoost(shift uint32) int {
	val := (p.Boosts >> shift) & BoostMask
	return int(val) - 6
}

// SetBoost sets the boost value (-6 to +6) for a given shift
func (p *PokemonState) SetBoost(shift uint32, val int) {
	if val < -6 {
		val = -6
	}
	if val > 6 {
		val = 6
	}
	packed := uint32(val+6) & BoostMask
	p.Boosts = (p.Boosts & ^(BoostMask << shift)) | (packed << shift)
}

// ClearVolatiles resets all volatile statuses
func (p *PokemonState) ClearVolatiles() {
	p.Volatiles = 0
}

// ToggleVolatile sets or clears a volatile status bit
func (p *PokemonState) ToggleVolatile(bit uint32, active bool) {
	if active {
		p.Volatiles |= bit
	} else {
		p.Volatiles &= ^bit
	}
}

func MapVolatileToBit(v string) uint32 {
	switch gamedata.NormalizeID(v) {
	case "substitute":
		return VolatileSubstitute
	case "confusion":
		return VolatileConfusion
	case "encore":
		return VolatileEncore
	case "taunt":
		return VolatileTaunt
	case "leechseed":
		return VolatileLeechSeed
	case "perishsong":
		return VolatilePerishSong
	case "attract":
		return VolatileAttract
	case "focusenergy":
		return VolatileFocusEnergy
	case "destinybond":
		return VolatileDestinyBond
	case "magnetrise":
		return VolatileMagnetRise
	case "telekinesis":
		return VolatileTelekinesis
	case "gastroacid":
		return VolatileGastroAcid
	case "ingrain":
		return VolatileIngrain
	case "aquaring":
		return VolatileAquaRing
	case "curse":
		return VolatileCurse
	case "embargo":
		return VolatileEmbargo
	case "healblock":
		return VolatileHealBlock
	case "protect", "endure", "spikyshield", "banefulbunker", "kingsshield", "silktrap", "obstruct", "burningbulwark":
		return VolatileProtection
	}
	return 0
}

// FieldConditions tracks global battle field state
type FieldConditions struct {
	Weather   string // "Sun", "RainDance", "Sandstorm", "Snow"/"Snowscape", ""
	Terrain   string // "Electric Terrain", "Grassy Terrain", "Psychic Terrain", "Misty Terrain", ""
	TrickRoom bool
	Gravity   bool
}

// SideConditions tracks per-side hazards and screens
type SideConditions struct {
	StealthRock bool
	Spikes      int // 0-3 layers
	ToxicSpikes int // 0-2 layers
	StickyWeb   bool
	Reflect     bool
	LightScreen bool
	AuroraVeil  bool
	Tailwind    bool
	Safeguard   bool
	Mist        bool
}

// PlayerState tracks the team and active Pokemon for a player
type PlayerState struct {
	ID              string
	Team            [6]PokemonState
	TeamSize        int
	ActiveIdx       int // Index into Team array. -1 if none.
	ForceSwitch     bool
	CanTerastallize bool
	Side            SideConditions
}

// GetActive returns a pointer to the active Pokemon, or nil
func (p *PlayerState) GetActive() *PokemonState {
	if p.ActiveIdx >= 0 && p.ActiveIdx < p.TeamSize {
		return &p.Team[p.ActiveIdx]
	}
	return nil
}

// BattleState tracks the full simulation up to the current turn
type BattleState struct {
	Turn     int
	P1       PlayerState
	P2       PlayerState
	Field    FieldConditions
	RNGState uint64
}

// Action represents a possible move or switch
type Action struct {
	Type string // "move" or "switch"
	Name string // The move name or the Pokemon to switch to
}

func seedHash(parts ...string) uint64 {
	var h uint64 = 1469598103934665603
	for _, part := range parts {
		for i := 0; i < len(part); i++ {
			h ^= uint64(part[i])
			h *= 1099511628211
		}
		h ^= 0xff
		h *= 1099511628211
	}
	if h == 0 {
		return 0x9e3779b97f4a7c15
	}
	return h
}

func nextRand(state *BattleState) uint64 {
	if state == nil {
		return 0
	}
	x := state.RNGState
	if x == 0 {
		x = 0x9e3779b97f4a7c15 ^ uint64(state.Turn+1)
	}
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	state.RNGState = x
	return x * 2685821657736338717
}

func randomInt(state *BattleState, n int) int {
	if n <= 1 {
		return 0
	}
	return int(nextRand(state) % uint64(n))
}

func randomChance(state *BattleState, numerator, denominator int) bool {
	if numerator <= 0 {
		return false
	}
	if numerator >= denominator {
		return true
	}
	return randomInt(state, denominator) < numerator
}

// FastForward runs the events in the replay up to `targetTurn` to rebuild the state
// FastForward reconstructs the state up to a specified turn.
// It includes all events with Turn <= targetTurn.
func FastForward(replay *parser.Replay, targetTurn int) (*BattleState, error) {
	// Find the last event index for the target turn
	targetIndex := -1
	for i, event := range replay.Events {
		if event.Turn <= targetTurn {
			targetIndex = i
		} else {
			break
		}
	}
	return FastForwardToEvent(replay, targetIndex)
}

func initPokemonArray(p *PlayerState, replayTeam map[string]bool) {
	idx := 0
	for species := range replayTeam {
		if idx >= 6 {
			break
		}
		maxHP := 100
		stats := Stats{}
		if entry := gamedata.LookupSpecies(species); entry != nil {
			// Replay logs for the opponent often expose percentages, but for the simulator
			// internals we keep canonical level-100 neutral stats as a baseline.
			maxHP = entry.BaseStats.HP*2 + 141
			stats = Stats{
				HP:  maxHP,
				Atk: entry.BaseStats.Atk*2 + 36,
				Def: entry.BaseStats.Def*2 + 36,
				SpA: entry.BaseStats.SpA*2 + 36,
				SpD: entry.BaseStats.SpD*2 + 36,
				Spe: entry.BaseStats.Spe*2 + 36,
			}
		}
		// Reset boosts to neutral (0 stages for all packed stats).
		p.Team[idx] = PokemonState{
			Name:     species,
			Species:  species,
			Level:    100,
			HP:       maxHP,
			MaxHP:    maxHP,
			Stats:    stats,
			IsActive: false,
			Fainted:  false,
			Boosts:   NeutralBoosts,
		}
		idx++
	}
	p.TeamSize = idx
	p.ActiveIdx = -1
	p.CanTerastallize = true
}

func findPokemonByName(p *PlayerState, name string) int {
	needle := gamedata.NormalizeID(name)
	for i := 0; i < p.TeamSize; i++ {
		if gamedata.NormalizeID(p.Team[i].Name) == needle || gamedata.NormalizeID(p.Team[i].Species) == needle {
			return i
		}
	}
	return -1
}

func parseConditionHP(cond string) (hp int, maxHP int, fainted bool, ok bool) {
	cond = strings.TrimSpace(cond)
	if cond == "" {
		return 0, 0, false, false
	}
	if cond == "0 fnt" {
		return 0, 100, true, true
	}

	parts := strings.Fields(cond)
	hpPart := parts[0]
	if strings.Contains(hpPart, "/") {
		hpParts := strings.SplitN(hpPart, "/", 2)
		if len(hpParts) == 2 {
			cur, errCur := strconv.Atoi(hpParts[0])
			max, errMax := strconv.Atoi(hpParts[1])
			if errCur == nil && errMax == nil && max > 0 {
				return cur, max, cur <= 0, true
			}
		}
	}
	if v, err := strconv.Atoi(hpPart); err == nil {
		return v, 100, v <= 0, true
	}
	return 0, 0, false, false
}

// FastForwardToEvent reconstructs the state exactly up to (and including) the specified event index.
// ApplyEvent applies a single parser.Event to the given BattleState.
func ApplyEvent(state *BattleState, event parser.Event) {
	state.Turn = event.Turn
		playerState := &state.P1
		if event.Player == "p2" {
			playerState = &state.P2
		}

		switch event.Type {
		case "switch", "drag", "replace":
			if event.Value != "" {
				speciesName := event.Value
				if playerState.ActiveIdx != -1 {
					playerState.Team[playerState.ActiveIdx].IsActive = false
				}
				idx := findPokemonByName(playerState, speciesName)
				if idx != -1 {
					playerState.Team[idx].IsActive = true
					playerState.ActiveIdx = idx
				} else if playerState.TeamSize < 6 {
					maxHP := 100
					stats := Stats{}
					if entry := gamedata.LookupSpecies(speciesName); entry != nil {
						maxHP = entry.BaseStats.HP*2 + 141
						stats = Stats{
							HP:  maxHP,
							Atk: entry.BaseStats.Atk*2 + 36,
							Def: entry.BaseStats.Def*2 + 36,
							SpA: entry.BaseStats.SpA*2 + 36,
							SpD: entry.BaseStats.SpD*2 + 36,
							Spe: entry.BaseStats.Spe*2 + 36,
						}
					}
					newIdx := playerState.TeamSize
					playerState.Team[newIdx] = PokemonState{
						Name:     speciesName,
						Species:  speciesName,
						Level:    100,
						Stats:    stats,
						HP:       maxHP,
						MaxHP:    maxHP,
						IsActive: true, Fainted: false,
						Boosts: NeutralBoosts,
					}
					playerState.TeamSize++
					playerState.ActiveIdx = newIdx
				}
			}

		case "faint":
			if playerState.ActiveIdx != -1 {
				playerState.Team[playerState.ActiveIdx].Fainted = true
				playerState.Team[playerState.ActiveIdx].HP = 0
			}

		case "damage", "heal":
			if event.Value != "" {
				idx := findPokemonByName(playerState, event.Value)
				if idx != -1 {
					poke := &playerState.Team[idx]
					if split := strings.SplitN(event.Detail, "->", 2); len(split) == 2 {
						if hp, maxHP, fainted, ok := parseConditionHP(strings.TrimSpace(split[1])); ok {
							// Replay logs often use percentage bars (x/100). If we already have a
							// species-derived max HP, project the new percent into that max HP.
							if maxHP == 100 && poke.MaxHP > 100 {
								poke.HP = int(math.Round((float64(hp) / 100.0) * float64(poke.MaxHP)))
							} else {
								poke.HP = hp
								poke.MaxHP = maxHP
							}
							if poke.HP < 0 {
								poke.HP = 0
							}
							if event.Type == "heal" && poke.HP > 0 {
								poke.Fainted = false
							} else if fainted && event.Type == "damage" {
								// Keep this conservative for damage lines; explicit faint events
								// still remain the source of truth for edge cases like Illusion.
								poke.HP = 0
							}
						}
					}
					if event.Type == "heal" {
						poke.Fainted = false
					}
				}
			}

		case "status":
			if event.Value != "" {
				idx := findPokemonByName(playerState, event.Value)
				if idx != -1 {
					playerState.Team[idx].Status = event.Detail
				}
			}

		case "curestatus":
			if event.Value != "" {
				idx := findPokemonByName(playerState, event.Value)
				if idx != -1 {
					playerState.Team[idx].Status = ""
				}
			}

		case "weather":
			if event.Value == "none" {
				state.Field.Weather = ""
			} else {
				state.Field.Weather = event.Value
			}

		case "fieldstart":
			val := event.Value
			switch {
			case strings.Contains(val, "Electric Terrain"):
				state.Field.Terrain = "Electric Terrain"
			case strings.Contains(val, "Grassy Terrain"):
				state.Field.Terrain = "Grassy Terrain"
			case strings.Contains(val, "Psychic Terrain"):
				state.Field.Terrain = "Psychic Terrain"
			case strings.Contains(val, "Misty Terrain"):
				state.Field.Terrain = "Misty Terrain"
			case strings.Contains(val, "Trick Room"):
				state.Field.TrickRoom = true
			case strings.Contains(val, "Gravity"):
				state.Field.Gravity = true
			}

		case "fieldend":
			val := event.Value
			switch {
			case strings.Contains(val, "Terrain"):
				state.Field.Terrain = ""
			case strings.Contains(val, "Trick Room"):
				state.Field.TrickRoom = false
			case strings.Contains(val, "Gravity"):
				state.Field.Gravity = false
			}

		case "sidestart":
			side := &state.P1.Side
			if event.Player == "p2" {
				side = &state.P2.Side
			}
			switch {
			case strings.Contains(event.Value, "Stealth Rock"):
				side.StealthRock = true
			case event.Value == "Spikes":
				if side.Spikes < 3 {
					side.Spikes++
				}
			case strings.Contains(event.Value, "Toxic Spikes"):
				if side.ToxicSpikes < 2 {
					side.ToxicSpikes++
				}
			case strings.Contains(event.Value, "Sticky Web"):
				side.StickyWeb = true
			case strings.Contains(event.Value, "Reflect"):
				side.Reflect = true
			case strings.Contains(event.Value, "Light Screen"):
				side.LightScreen = true
			case strings.Contains(event.Value, "Aurora Veil"):
				side.AuroraVeil = true
			case strings.Contains(event.Value, "Tailwind"):
				side.Tailwind = true
			case strings.Contains(event.Value, "Safeguard"):
				side.Safeguard = true
			case strings.Contains(event.Value, "Mist"):
				side.Mist = true
			}

		case "sideend":
			side := &state.P1.Side
			if event.Player == "p2" {
				side = &state.P2.Side
			}
			switch {
			case strings.Contains(event.Value, "Stealth Rock"):
				side.StealthRock = false
			case event.Value == "Spikes":
				side.Spikes = 0
			case strings.Contains(event.Value, "Toxic Spikes"):
				side.ToxicSpikes = 0
			case strings.Contains(event.Value, "Sticky Web"):
				side.StickyWeb = false
			case strings.Contains(event.Value, "Reflect"):
				side.Reflect = false
			case strings.Contains(event.Value, "Light Screen"):
				side.LightScreen = false
			case strings.Contains(event.Value, "Aurora Veil"):
				side.AuroraVeil = false
			case strings.Contains(event.Value, "Tailwind"):
				side.Tailwind = false
			case strings.Contains(event.Value, "Safeguard"):
				side.Safeguard = false
			case strings.Contains(event.Value, "Mist"):
				side.Mist = false
			}

		case "start":
			if playerState.ActiveIdx != -1 {
				bit := MapVolatileToBit(event.Value)
				playerState.Team[playerState.ActiveIdx].Volatiles |= bit
			}

		case "end":
			if playerState.ActiveIdx != -1 {
				bit := MapVolatileToBit(event.Value)
				playerState.Team[playerState.ActiveIdx].Volatiles &= ^bit
			}

		case "boost":
			if playerState.ActiveIdx != -1 {
				amount, _ := strconv.Atoi(event.Detail)
				stat := event.Pokemon // stat name stored in Pokemon field
				if shift, ok := GetStatShift(stat); ok {
					cur := playerState.Team[playerState.ActiveIdx].GetBoost(shift)
					playerState.Team[playerState.ActiveIdx].SetBoost(shift, cur+amount)
				}
			}

		case "unboost":
			if playerState.ActiveIdx != -1 {
				amount, _ := strconv.Atoi(event.Detail)
				stat := event.Pokemon
				if shift, ok := GetStatShift(stat); ok {
					cur := playerState.Team[playerState.ActiveIdx].GetBoost(shift)
					playerState.Team[playerState.ActiveIdx].SetBoost(shift, cur-amount)
				}
			}

		case "setboost":
			if playerState.ActiveIdx != -1 {
				amount, _ := strconv.Atoi(event.Detail)
				stat := event.Pokemon
				if shift, ok := GetStatShift(stat); ok {
					playerState.Team[playerState.ActiveIdx].SetBoost(shift, amount)
				}
			}

		case "clearallboost", "clearboost":
			if playerState.ActiveIdx != -1 {
				playerState.Team[playerState.ActiveIdx].Boosts = NeutralBoosts
			}

		case "clearnegativeboost":
			if playerState.ActiveIdx != -1 {
				for _, shift := range []uint32{AtkShift, DefShift, SpaShift, SpdShift, SpeShift, EvaShift, AccShift} {
					if playerState.Team[playerState.ActiveIdx].GetBoost(shift) < 0 {
						playerState.Team[playerState.ActiveIdx].SetBoost(shift, 0)
					}
				}
			}

		case "clearpositiveboost":
			if playerState.ActiveIdx != -1 {
				for _, shift := range []uint32{AtkShift, DefShift, SpaShift, SpdShift, SpeShift, EvaShift, AccShift} {
					if playerState.Team[playerState.ActiveIdx].GetBoost(shift) > 0 {
						playerState.Team[playerState.ActiveIdx].SetBoost(shift, 0)
					}
				}
			}

		case "move":
			if playerState.ActiveIdx != -1 {
				active := &playerState.Team[playerState.ActiveIdx]
				moveName := event.Value
				// Add move to the pokemon's known moves if not already present
				found := false
				for i := 0; i < active.NumMoves; i++ {
					if active.Moves[i] == moveName {
						found = true
						break
					}
				}
				if !found && active.NumMoves < 4 {
					active.Moves[active.NumMoves] = moveName
					active.NumMoves++
				}
			}
		case "terastallize":
			if playerState.ActiveIdx != -1 {
				active := &playerState.Team[playerState.ActiveIdx]
				active.Terastallized = true
				active.TeraType = normalizeTypeName(event.Value)
				playerState.CanTerastallize = false
			}
		}
}

// UpdateRNGState updates the RNGState for a specific event index.
func UpdateRNGState(state *BattleState, replay *parser.Replay, targetIndex int) {
	state.RNGState = seedHash(replay.P1, replay.P2, strconv.Itoa(targetIndex))
}
func FastForwardToEvent(replay *parser.Replay, targetIndex int) (*BattleState, error) {
	state := &BattleState{
		Turn: 0,
		P1: PlayerState{
			ID:              "p1",
			ActiveIdx:       -1,
			CanTerastallize: true,
		},
		P2: PlayerState{
			ID:              "p2",
			ActiveIdx:       -1,
			CanTerastallize: true,
		},
		RNGState: seedHash(replay.P1, replay.P2, strconv.Itoa(targetIndex)),
	}

	// 1. Initialize Teams from the replay
	initPokemonArray(&state.P1, replay.Teams["p1"])
	initPokemonArray(&state.P2, replay.Teams["p2"])

	// Turn 0 initialization: Find the first switch for each player and make them active
	for _, event := range replay.Events {
		if event.Turn > 0 {
			break
		}
		if event.Type == "switch" {
			playerState := &state.P1
			if event.Player == "p2" {
				playerState = &state.P2
			}
			parts := strings.Split(event.Detail, " switched to ")
			if len(parts) == 2 {
				speciesName := event.Value
				idx := findPokemonByName(playerState, speciesName)
				if idx != -1 {
					playerState.Team[idx].IsActive = true
					playerState.ActiveIdx = idx
				}
			}
		}
	}

	// 2. Process events sequentially up to targetIndex
	for i, event := range replay.Events {
		if i > targetIndex {
			break
		}

		ApplyEvent(state, event)
	}

	return state, nil
}

// GetValidActions returns all legally selectable actions for a player at the current state.
func GetValidActions(state *BattleState, replay *parser.Replay, playerID string) ([]Action, error) {
	playerState := &state.P1
	if playerID == "p2" {
		playerState = &state.P2
	}

	var actions []Action

	// 1. Valid Moves
	activePoke := playerState.GetActive()
	if activePoke != nil && !activePoke.Fainted {
		if playerMoves, playerExists := replay.KnownMoves[playerID]; playerExists {
			if moves, exists := playerMoves[activePoke.Species]; exists {
				for move := range moves {
					actions = append(actions, Action{
						Type: "move",
						Name: move,
					})
				}
			}
		}
	}

	// 2. Valid Switches
	for i := 0; i < playerState.TeamSize; i++ {
		poke := &playerState.Team[i]
		if !poke.IsActive && !poke.Fainted {
			actions = append(actions, Action{
				Type: "switch",
				Name: poke.Species,
			})
		}
	}

	return actions, nil
}

// Search Action Enums (Zero-Allocation)
const (
	ActionMove1 = 0 // Use move slot 1
	ActionMove2 = 1 // Use move slot 2
	ActionMove3 = 2 // Use move slot 3
	ActionMove4 = 3 // Use move slot 4
	// Terastallized move actions (same move slots, with tera activation this turn)
	ActionTeraMove1 = 4
	ActionTeraMove2 = 5
	ActionTeraMove3 = 6
	ActionTeraMove4 = 7
	// 8-13 are switch indices (ActionSwitchBase + team slot idx)
	ActionSwitchBase = 8
	MaxActions       = 14
)

// IsAttackAction returns true if the action is a move (normal or tera move).
func IsAttackAction(action int) bool {
	return (action >= ActionMove1 && action <= ActionMove4) || IsTeraAction(action)
}

// IsTeraAction returns true if the action is a tera-move action (4-7).
func IsTeraAction(action int) bool {
	return action >= ActionTeraMove1 && action <= ActionTeraMove4
}

// BaseMoveIndex maps both normal and tera move actions to move slots 0-3.
func BaseMoveIndex(action int) int {
	switch {
	case action >= ActionMove1 && action <= ActionMove4:
		return action - ActionMove1
	case action >= ActionTeraMove1 && action <= ActionTeraMove4:
		return action - ActionTeraMove1
	default:
		return -1
	}
}

func normalizeTypeName(typeName string) string {
	for _, t := range gamedata.AllTypes {
		if strings.EqualFold(t, typeName) {
			return t
		}
	}
	return ""
}

func getCurrentTypes(p *PokemonState, entry *gamedata.PokedexEntry) []string {
	if p.Terastallized && p.TeraType != "" {
		return []string{p.TeraType}
	}
	if entry == nil {
		return nil
	}
	return entry.Types
}

// GetSearchActions generates valid actions for Alpha-Beta search WITHOUT requiring replay context.
// Returns a fixed array and length to completely eliminate slice allocations.
// 0-3 = Move 1-4, 4-7 = Tera Move 1-4, 8-13 = Switch to slot 0-5.
func GetSearchActions(player *PlayerState) ([MaxActions]int, int) {
	var actions [MaxActions]int
	count := 0

	// 1. Move actions (only for valid move slots, unless forced to switch)
	active := player.GetActive()
	if active != nil && !active.Fainted && !player.ForceSwitch {
		numMoves := active.NumMoves
		if numMoves == 0 {
			numMoves = 4 // Assume 4 moves for search if none are known
		}
		for m := 0; m < numMoves; m++ {
			if count < MaxActions {
				actions[count] = ActionMove1 + m
				count++
			}
		}
		// Tera move actions are legal if tera hasn't been used and this active has a tera type.
		if player.CanTerastallize && !active.Terastallized && active.TeraType != "" {
			for m := 0; m < numMoves; m++ {
				if count < MaxActions {
					actions[count] = ActionTeraMove1 + m
					count++
				}
			}
		}
	}

	// 2. Switches
	for i := 0; i < player.TeamSize; i++ {
		poke := &player.Team[i]
		if !poke.IsActive && !poke.Fainted {
			if count < MaxActions {
				actions[count] = ActionSwitchBase + i
				count++
			}
		}
	}

	return actions, count
}

// CloneBattleState creates a deep copy of a BattleState so MCTS can safely simulate futures
func CloneBattleState(state *BattleState) *BattleState {
	// Because BattleState and all its children (PlayerState, PokemonState, etc)
	// are now completely flat struct values with NO pointers, maps, or slices,
	// this assignment operator forces the Go runtime to do a blazing fast,
	// zero-allocation flat memory block copy! GC pressure = 0.
	cloned := *state
	return &cloned
}

func applyTerastallizeIfNeeded(player *PlayerState, action int) {
	if !IsTeraAction(action) || !player.CanTerastallize {
		return
	}
	active := player.GetActive()
	if active == nil || active.Fainted || active.Terastallized || active.TeraType == "" {
		return
	}
	active.Terastallized = true
	player.CanTerastallize = false
}

func markActiveFainted(player *PlayerState) {
	active := player.GetActive()
	if active == nil || active.HP > 0 {
		return
	}
	active.HP = 0
	active.Fainted = true
	active.IsActive = false
	player.ActiveIdx = -1
}

func canActThisTurn(state *BattleState, p *PokemonState) bool {
	if p == nil || p.Fainted {
		return false
	}
	switch p.Status {
	case "par":
		// Full paralysis: 25%
		if randomChance(state, 1, 4) {
			return false
		}
	case "slp":
		// Sleep approximation: 2/3 skip, 1/3 wake and move.
		if randomChance(state, 2, 3) {
			return false
		}
		p.Status = ""
	case "frz":
		// Freeze thaw chance each turn: 20%.
		if randomChance(state, 4, 5) {
			return false
		}
		p.Status = ""
	}
	return true
}

func applyStatusResidual(p *PokemonState) {
	if p == nil || p.Fainted || p.MaxHP <= 0 {
		return
	}
	switch p.Status {
	case "brn":
		dmg := p.MaxHP / 16
		if dmg < 1 {
			dmg = 1
		}
		p.HP -= dmg
	case "psn", "tox":
		dmg := p.MaxHP / 8
		if dmg < 1 {
			dmg = 1
		}
		p.HP -= dmg
	}
}

// ExecuteSpecificTurn applies two integer actions directly to the BattleState
// 0-3 = Move 1-4, 4-7 = Tera Move 1-4, 8-13 = Switch to slot 0-5
func ExecuteSpecificTurn(state *BattleState, p1Action int, p2Action int) {
	state.Turn++

	// Execute Switches first
	if p1Action >= ActionSwitchBase {
		idx := p1Action - ActionSwitchBase
		if state.P1.ActiveIdx != -1 {
			state.P1.Team[state.P1.ActiveIdx].IsActive = false
		}
		if idx < state.P1.TeamSize {
			state.P1.Team[idx].IsActive = true
			state.P1.ActiveIdx = idx
		}
	}
	if p2Action >= ActionSwitchBase {
		idx := p2Action - ActionSwitchBase
		if state.P2.ActiveIdx != -1 {
			state.P2.Team[state.P2.ActiveIdx].IsActive = false
		}
		if idx < state.P2.TeamSize {
			state.P2.Team[idx].IsActive = true
			state.P2.ActiveIdx = idx
		}
	}

	// Tera declarations occur before move execution.
	applyTerastallizeIfNeeded(&state.P1, p1Action)
	applyTerastallizeIfNeeded(&state.P2, p2Action)

	// Move-Aware Damage Engine
	p1Active := state.P1.GetActive()
	p2Active := state.P2.GetActive()

	p1Attacks := IsAttackAction(p1Action) && p1Active != nil
	p2Attacks := IsAttackAction(p2Action) && p2Active != nil

	if p1Attacks && p2Attacks {
		// Determine speed order
		p1Spe := getEffectiveStat(p1Active, SpeShift, &state.P1.Side)
		p2Spe := getEffectiveStat(p2Active, SpeShift, &state.P2.Side)

		p1First := false
		switch {
		case p1Spe > p2Spe:
			p1First = true
		case p1Spe < p2Spe:
			p1First = false
		default:
			// Speed ties are random.
			p1First = randomChance(state, 1, 2)
		}

		if p1First {
			p2Flinched := false
			if canActThisTurn(state, p1Active) {
				_, p2Flinched = applyMoveDamage(state, p1Active, p2Active, BaseMoveIndex(p1Action), &state.P2.Side)
			}
			markActiveFainted(&state.P2)
			if state.P2.GetActive() != nil && !p2Flinched && canActThisTurn(state, state.P2.GetActive()) {
				applyMoveDamage(state, state.P2.GetActive(), state.P1.GetActive(), BaseMoveIndex(p2Action), &state.P1.Side)
			}
		} else {
			p1Flinched := false
			if canActThisTurn(state, p2Active) {
				_, p1Flinched = applyMoveDamage(state, p2Active, p1Active, BaseMoveIndex(p2Action), &state.P1.Side)
			}
			markActiveFainted(&state.P1)
			if state.P1.GetActive() != nil && !p1Flinched && canActThisTurn(state, state.P1.GetActive()) {
				applyMoveDamage(state, state.P1.GetActive(), state.P2.GetActive(), BaseMoveIndex(p1Action), &state.P2.Side)
			}
		}
	} else {
		if p1Attacks && p2Active != nil && canActThisTurn(state, p1Active) {
			applyMoveDamage(state, p1Active, p2Active, BaseMoveIndex(p1Action), &state.P2.Side)
		} else if p2Attacks && p1Active != nil && canActThisTurn(state, p2Active) {
			applyMoveDamage(state, p2Active, p1Active, BaseMoveIndex(p2Action), &state.P1.Side)
		}
	}

	// End-of-turn residual statuses.
	applyStatusResidual(state.P1.GetActive())
	applyStatusResidual(state.P2.GetActive())

	// Check for faints
	markActiveFainted(&state.P1)
	markActiveFainted(&state.P2)
}

func normalizedName(s string) string {
	return gamedata.NormalizeID(s)
}

func statBoostMultiplier(boost int) float64 {
	if boost >= 0 {
		return float64(2+boost) / 2.0
	}
	return 2.0 / float64(2-boost)
}

func accuracyStageMultiplier(stage int) float64 {
	if stage >= 0 {
		return float64(3+stage) / 3.0
	}
	return 3.0 / float64(3-stage)
}

func calcNonHPStat(base, iv, ev, level int) int {
	return ((2*base+iv+(ev/4))*level)/100 + 5
}

// getEffectiveStat computes the stat including boosts and common battle modifiers.
func getEffectiveStat(p *PokemonState, shift uint32, side *SideConditions) float64 {
	base := 0.0
	switch shift {
	case AtkShift:
		base = float64(p.Stats.Atk)
	case DefShift:
		base = float64(p.Stats.Def)
	case SpaShift:
		base = float64(p.Stats.SpA)
	case SpdShift:
		base = float64(p.Stats.SpD)
	case SpeShift:
		base = float64(p.Stats.Spe)
	}

	if base == 0 {
		entry := gamedata.LookupSpecies(p.Species)
		if entry != nil {
			level := 100
			if p.Level > 0 {
				level = p.Level
			}
			iv, ev := 31, 0
			switch shift {
			case AtkShift:
				if p.IVs.Atk > 0 || p.EVs.Atk > 0 {
					iv, ev = p.IVs.Atk, p.EVs.Atk
				}
				base = float64(calcNonHPStat(entry.BaseStats.Atk, iv, ev, level))
			case DefShift:
				if p.IVs.Def > 0 || p.EVs.Def > 0 {
					iv, ev = p.IVs.Def, p.EVs.Def
				}
				base = float64(calcNonHPStat(entry.BaseStats.Def, iv, ev, level))
			case SpaShift:
				if p.IVs.SpA > 0 || p.EVs.SpA > 0 {
					iv, ev = p.IVs.SpA, p.EVs.SpA
				}
				base = float64(calcNonHPStat(entry.BaseStats.SpA, iv, ev, level))
			case SpdShift:
				if p.IVs.SpD > 0 || p.EVs.SpD > 0 {
					iv, ev = p.IVs.SpD, p.EVs.SpD
				}
				base = float64(calcNonHPStat(entry.BaseStats.SpD, iv, ev, level))
			case SpeShift:
				if p.IVs.Spe > 0 || p.EVs.Spe > 0 {
					iv, ev = p.IVs.Spe, p.EVs.Spe
				}
				base = float64(calcNonHPStat(entry.BaseStats.Spe, iv, ev, level))
			}
		}
	}
	if base == 0 {
		base = 105
	}

	val := base * statBoostMultiplier(p.GetBoost(shift))
	if shift == SpeShift {
		if p.Status == "par" {
			val *= 0.5
		}
		if side != nil && side.Tailwind {
			val *= 2.0
		}
	}

	if shift == AtkShift {
		if normalizedName(p.Ability) == "hugepower" || normalizedName(p.Ability) == "purepower" {
			val *= 2.0
		}
		if normalizedName(p.Item) == "choiceband" {
			val *= 1.5
		}
	}
	if shift == DefShift && normalizedName(p.Item) == "eviolite" {
		val *= 1.5
	}
	if shift == SpaShift && normalizedName(p.Item) == "choicespecs" {
		val *= 1.5
	}
	if shift == SpdShift && normalizedName(p.Item) == "assaultvest" {
		val *= 1.5
	}

	if val < 1 {
		val = 1
	}
	return val
}

func getMoveAccuracy(move *gamedata.MoveEntry) (float64, bool) {
	if move == nil {
		return 100.0, false
	}
	switch v := move.Accuracy.(type) {
	case bool:
		return 0, v
	case int:
		return float64(v), false
	case int64:
		return float64(v), false
	case float64:
		return v, false
	default:
		return 100.0, false
	}
}

func moveHits(state *BattleState, attacker *PokemonState, defender *PokemonState, move *gamedata.MoveEntry) bool {
	baseAcc, alwaysHit := getMoveAccuracy(move)
	if alwaysHit {
		return true
	}
	if baseAcc <= 0 {
		return false
	}
	acc := baseAcc * accuracyStageMultiplier(attacker.GetBoost(AccShift)) / accuracyStageMultiplier(defender.GetBoost(EvaShift))
	if acc <= 0 {
		return false
	}
	if acc >= 100 {
		return true
	}
	roll := float64(randomInt(state, 10000)) / 100.0
	return roll < acc
}

func critChanceNumerator(stage int) int {
	switch {
	case stage <= 0:
		return 1 // 1/24
	case stage == 1:
		return 1 // 1/8
	case stage == 2:
		return 1 // 1/2
	default:
		return 1 // guaranteed at denominator 1
	}
}

func critChanceDenominator(stage int) int {
	switch {
	case stage <= 0:
		return 24
	case stage == 1:
		return 8
	case stage == 2:
		return 2
	default:
		return 1
	}
}

func moveIsCritical(state *BattleState, attacker *PokemonState, move *gamedata.MoveEntry) bool {
	if move == nil {
		return false
	}
	if move.WillCrit {
		return true
	}
	stage := move.CritRatio
	if (attacker.Volatiles & VolatileFocusEnergy) != 0 {
		stage += 2
	}
	return randomChance(state, critChanceNumerator(stage), critChanceDenominator(stage))
}

func hasType(types []string, t string) bool {
	for _, existing := range types {
		if existing == t {
			return true
		}
	}
	return false
}

func stabModifier(attacker *PokemonState, atkEntry *gamedata.PokedexEntry, moveType string) float64 {
	if moveType == "" || atkEntry == nil {
		return 1.0
	}
	ability := normalizedName(attacker.Ability)
	hasOriginalType := hasType(atkEntry.Types, moveType)
	adaptability := ability == "adaptability"

	if attacker.Terastallized && attacker.TeraType != "" {
		if moveType == attacker.TeraType && hasOriginalType {
			if adaptability {
				return 2.25
			}
			return 2.0
		}
		if moveType == attacker.TeraType || hasOriginalType {
			if adaptability {
				return 2.0
			}
			return 1.5
		}
		return 1.0
	}

	if hasOriginalType {
		if adaptability {
			return 2.0
		}
		return 1.5
	}
	return 1.0
}

func clampHP(p *PokemonState) {
	if p.MaxHP <= 0 {
		p.MaxHP = 1
	}
	if p.HP > p.MaxHP {
		p.HP = p.MaxHP
	}
	if p.HP < 0 {
		p.HP = 0
	}
}

func applyBoostMap(target *PokemonState, boosts map[string]int) {
	if target == nil || len(boosts) == 0 {
		return
	}
	for stat, delta := range boosts {
		shift, ok := GetStatShift(stat)
		if !ok {
			continue
		}
		target.SetBoost(shift, target.GetBoost(shift)+delta)
	}
}

func applySecondary(state *BattleState, sec gamedata.MoveSecondary, attacker *PokemonState, defender *PokemonState) bool {
	if sec.Chance > 0 && !randomChance(state, sec.Chance, 100) {
		return false
	}
	if sec.Status != "" && defender.Status == "" {
		defender.Status = sec.Status
	}
	if sec.VolatileStatus == "flinch" {
		return true
	}
	if sec.VolatileStatus != "" {
		defender.ToggleVolatile(MapVolatileToBit(sec.VolatileStatus), true)
	}
	applyBoostMap(defender, sec.Boosts)
	if sec.Self != nil {
		applyBoostMap(attacker, sec.Self.Boosts)
	}
	return false
}

func applyMoveEffects(state *BattleState, move *gamedata.MoveEntry, attacker *PokemonState, defender *PokemonState, damageDealt int) bool {
	if move == nil {
		return false
	}
	flinch := false

	if move.Status != "" && defender.Status == "" {
		defender.Status = move.Status
	}
	if move.VolatileStatus == "flinch" {
		flinch = true
	} else if move.VolatileStatus != "" {
		defender.ToggleVolatile(MapVolatileToBit(move.VolatileStatus), true)
	}

	if len(move.Boosts) > 0 {
		if move.Target == "self" {
			applyBoostMap(attacker, move.Boosts)
		} else {
			applyBoostMap(defender, move.Boosts)
		}
	}

	if move.Secondary != nil {
		if applySecondary(state, *move.Secondary, attacker, defender) {
			flinch = true
		}
	}
	for _, sec := range move.Secondaries {
		if applySecondary(state, sec, attacker, defender) {
			flinch = true
		}
	}

	if damageDealt > 0 {
		if move.Drain[1] > 0 && move.Drain[0] > 0 {
			heal := int(math.Round(float64(damageDealt) * float64(move.Drain[0]) / float64(move.Drain[1])))
			if heal < 1 {
				heal = 1
			}
			attacker.HP += heal
		}
		if move.Recoil[1] > 0 && move.Recoil[0] > 0 {
			recoil := int(math.Round(float64(damageDealt) * float64(move.Recoil[0]) / float64(move.Recoil[1])))
			if recoil < 1 {
				recoil = 1
			}
			attacker.HP -= recoil
		}
	}
	if move.Healing[1] > 0 && move.Healing[0] > 0 {
		heal := int(math.Round(float64(attacker.MaxHP) * float64(move.Healing[0]) / float64(move.Healing[1])))
		if heal < 1 {
			heal = 1
		}
		attacker.HP += heal
	}

	clampHP(attacker)
	clampHP(defender)
	return flinch
}

// applyMoveDamage applies a single move resolution.
// Returns (moveHit, defenderFlinched).
func applyMoveDamage(state *BattleState, attacker *PokemonState, defender *PokemonState, moveIdx int, defenderSide *SideConditions) (bool, bool) {
	if attacker == nil || defender == nil || moveIdx < 0 || moveIdx >= 4 {
		return false, false
	}

	moveEntry := gamedata.LookupMove(attacker.Moves[moveIdx])
	if !moveHits(state, attacker, defender, moveEntry) {
		return false, false
	}

	if moveEntry != nil && moveEntry.Category == "Status" {
		flinch := applyMoveEffects(state, moveEntry, attacker, defender, 0)
		clampHP(attacker)
		clampHP(defender)
		return true, flinch
	}

	atkEntry := gamedata.LookupSpecies(attacker.Species)
	defEntry := gamedata.LookupSpecies(defender.Species)
	if atkEntry == nil || defEntry == nil {
		defender.HP -= 15
		clampHP(defender)
		return true, false
	}

	power := 80.0
	moveType := ""
	isPhysical := true
	if moveEntry != nil {
		if moveEntry.BasePower > 0 {
			power = float64(moveEntry.BasePower)
		}
		moveType = moveEntry.Type
		isPhysical = moveEntry.Category != "Special"
	}
	if power <= 0 {
		flinch := applyMoveEffects(state, moveEntry, attacker, defender, 0)
		clampHP(attacker)
		clampHP(defender)
		return true, flinch
	}

	attackStat := 1.0
	defenseStat := 1.0
	if isPhysical {
		attackStat = getEffectiveStat(attacker, AtkShift, nil)
		defenseStat = getEffectiveStat(defender, DefShift, defenderSide)
	} else {
		attackStat = getEffectiveStat(attacker, SpaShift, nil)
		defenseStat = getEffectiveStat(defender, SpdShift, defenderSide)
	}
	if defenseStat < 1 {
		defenseStat = 1
	}

	level := 100.0
	if attacker.Level > 0 {
		level = float64(attacker.Level)
	}
	baseDamage := math.Floor((math.Floor((2.0*level)/5.0+2.0) * power * attackStat / defenseStat / 50.0) + 2.0)
	if baseDamage < 1 {
		baseDamage = 1
	}

	critical := moveIsCritical(state, attacker, moveEntry)
	modifier := float64(randomInt(state, 16)+85) / 100.0
	if critical {
		modifier *= 1.5
	}

	if moveType != "" {
		modifier *= stabModifier(attacker, atkEntry, moveType)
		effectiveness := gamedata.CalcTypeEffectiveness(moveType, getCurrentTypes(defender, defEntry))
		if effectiveness == 0 {
			return true, false
		}
		modifier *= effectiveness
	}

	if isPhysical && attacker.Status == "brn" && normalizedName(attacker.Ability) != "guts" {
		modifier *= 0.5
	}

	if defenderSide != nil && !critical {
		if isPhysical && (defenderSide.Reflect || defenderSide.AuroraVeil) {
			modifier *= 0.5
		} else if !isPhysical && (defenderSide.LightScreen || defenderSide.AuroraVeil) {
			modifier *= 0.5
		}
	}

	if normalizedName(attacker.Item) == "lifeorb" {
		modifier *= 1.3
	}
	if normalizedName(attacker.Ability) == "rivalry" && attacker.Gender != "N" && defender.Gender != "N" {
		if attacker.Gender == defender.Gender {
			modifier *= 1.25
		} else {
			modifier *= 0.75
		}
	}

	finalDamage := int(math.Floor(baseDamage * modifier))
	if finalDamage < 1 && modifier > 0 {
		finalDamage = 1
	}
	if finalDamage > defender.HP {
		finalDamage = defender.HP
	}
	defender.HP -= finalDamage
	clampHP(defender)

	flinch := applyMoveEffects(state, moveEntry, attacker, defender, finalDamage)
	clampHP(attacker)
	return true, flinch
}
