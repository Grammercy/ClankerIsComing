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
	Name               string
	Species            string
	TeraType           string
	HP                 int
	MaxHP              int
	Status             string // TODO: convert to enum/flag if needed
	SleepTurns         int    // Remaining turns unable to move while asleep
	FreezeTurns        int    // Consecutive turns spent frozen
	ToxicCounter       int    // Toxic counter increments each residual while active
	TookDamageThisTurn bool
	ActedThisTurn      bool
	Volatiles          uint32 // Bitfield for volatile statuses
	IsActive           bool
	Fainted            bool
	Terastallized      bool
	Boosts             uint32    // atk, def, spa, spd, spe, eva, acc boosts (packed 4 bits each: 0-15, center is 6)
	Moves              [4]string // move IDs (Showdown format, e.g. "thunderbolt")
	NumMoves           int       // how many move slots are filled (0-4)

	// Full details for Pokedex compliance
	Level          int
	Gender         string // "M", "F", "N" (none)
	Nature         string
	Ability        string
	Item           string
	EVs            Stats
	IVs            Stats
	Stats          Stats // Computed actual stats (before boosts/modifiers)
	DisguiseBroken bool
	IceFaceBroken  bool
	TurnsActive    int
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
		ability := normalizedName(p.Ability)
		if bit == VolatileConfusion && ability == "owntempo" {
			return
		}
		if (bit == VolatileTaunt || bit == VolatileAttract) && ability == "oblivious" {
			return
		}
		if (bit == VolatileTaunt || bit == VolatileEncore || bit == VolatileHealBlock) && ability == "aromaveil" {
			return
		}
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
	Weather        string // "Sun", "RainDance", "Sandstorm", "Snow"/"Snowscape", ""
	WeatherTurns   int
	Terrain        string // "Electric Terrain", "Grassy Terrain", "Psychic Terrain", "Misty Terrain", ""
	TerrainTurns   int
	TrickRoom      bool
	TrickRoomTurns int
	Gravity        bool
	GravityTurns   int
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
				onSwitchOut(&playerState.Team[playerState.ActiveIdx])
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
				status := normalizedName(event.Detail)
				playerState.Team[idx].Status = status
				if status != "slp" {
					playerState.Team[idx].SleepTurns = 0
				}
				if status != "frz" {
					playerState.Team[idx].FreezeTurns = 0
				}
				playerState.Team[idx].ToxicCounter = 0
			}
		}

	case "curestatus":
		if event.Value != "" {
			idx := findPokemonByName(playerState, event.Value)
			if idx != -1 {
				playerState.Team[idx].Status = ""
				playerState.Team[idx].SleepTurns = 0
				playerState.Team[idx].FreezeTurns = 0
				playerState.Team[idx].ToxicCounter = 0
			}
		}

	case "weather":
		if event.Value == "none" {
			state.Field.Weather = ""
			state.Field.WeatherTurns = 0
		} else {
			state.Field.Weather = normalizeWeatherID(event.Value)
			state.Field.WeatherTurns = 0
		}

	case "fieldstart":
		val := normalizedName(event.Value)
		switch {
		case val == "electricterrain":
			state.Field.Terrain = "electricterrain"
			state.Field.TerrainTurns = 0
		case val == "grassyterrain":
			state.Field.Terrain = "grassyterrain"
			state.Field.TerrainTurns = 0
		case val == "psychicterrain":
			state.Field.Terrain = "psychicterrain"
			state.Field.TerrainTurns = 0
		case val == "mistyterrain":
			state.Field.Terrain = "mistyterrain"
			state.Field.TerrainTurns = 0
		case val == "trickroom":
			state.Field.TrickRoom = true
			state.Field.TrickRoomTurns = 0
		case val == "gravity":
			state.Field.Gravity = true
			state.Field.GravityTurns = 0
		}

	case "fieldend":
		val := normalizedName(event.Value)
		switch {
		case strings.Contains(val, "terrain"):
			state.Field.Terrain = ""
			state.Field.TerrainTurns = 0
		case val == "trickroom":
			state.Field.TrickRoom = false
			state.Field.TrickRoomTurns = 0
		case val == "gravity":
			state.Field.Gravity = false
			state.Field.GravityTurns = 0
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

func normalizeWeatherID(weather string) string {
	switch normalizedName(weather) {
	case "raindance", "primordialsea":
		return "raindance"
	case "sunnyday", "desolateland":
		return "sunnyday"
	case "sandstorm":
		return "sandstorm"
	case "hail", "snowscape", "snow":
		return "snowscape"
	default:
		return ""
	}
}

func normalizeTerrainID(terrain string) string {
	switch normalizedName(terrain) {
	case "electricterrain":
		return "electricterrain"
	case "grassyterrain":
		return "grassyterrain"
	case "psychicterrain":
		return "psychicterrain"
	case "mistyterrain":
		return "mistyterrain"
	default:
		return ""
	}
}

func onSwitchOut(p *PokemonState) {
	if p == nil {
		return
	}
	if p.Status == "tox" {
		p.ToxicCounter = 0
	}
	p.ClearVolatiles()
}

func executeSwitchAction(player *PlayerState, action int) bool {
	if action < ActionSwitchBase {
		return false
	}
	idx := action - ActionSwitchBase
	if idx < 0 || idx >= player.TeamSize || idx == player.ActiveIdx {
		return false
	}
	if player.Team[idx].Fainted {
		return false
	}
	if player.ActiveIdx >= 0 && player.ActiveIdx < player.TeamSize {
		onSwitchOut(&player.Team[player.ActiveIdx])
		player.Team[player.ActiveIdx].IsActive = false
	}
	player.Team[idx].IsActive = true
	player.ActiveIdx = idx
	return true
}

func applyOnSwitchInAbilities(state *BattleState, player *PlayerState, opponent *PlayerState) {
	active := player.GetActive()
	if active == nil || active.Fainted {
		return
	}

	ability := normalizedName(active.Ability)
	switch ability {
	case "drizzle":
		if normalizeWeatherID(state.Field.Weather) != "raindance" {
			state.Field.Weather = "raindance"
			state.Field.WeatherTurns = 5
			if normalizedName(active.Item) == "damprock" {
				state.Field.WeatherTurns = 8
			}
		}
	case "drought":
		if normalizeWeatherID(state.Field.Weather) != "sunnyday" {
			state.Field.Weather = "sunnyday"
			state.Field.WeatherTurns = 5
			if normalizedName(active.Item) == "heatrock" {
				state.Field.WeatherTurns = 8
			}
		}
	case "sandstream":
		if normalizeWeatherID(state.Field.Weather) != "sandstorm" {
			state.Field.Weather = "sandstorm"
			state.Field.WeatherTurns = 5
			if normalizedName(active.Item) == "smoothrock" {
				state.Field.WeatherTurns = 8
			}
		}
	case "snowwarning":
		if normalizeWeatherID(state.Field.Weather) != "snowscape" {
			state.Field.Weather = "snowscape"
			state.Field.WeatherTurns = 5
			if normalizedName(active.Item) == "icyrock" {
				state.Field.WeatherTurns = 8
			}
		}
	case "electricsurge":
		if normalizeTerrainID(state.Field.Terrain) != "electricterrain" {
			state.Field.Terrain = "electricterrain"
			state.Field.TerrainTurns = 5
			if normalizedName(active.Item) == "terrainextender" {
				state.Field.TerrainTurns = 8
			}
		}
	case "grassysurge":
		if normalizeTerrainID(state.Field.Terrain) != "grassyterrain" {
			state.Field.Terrain = "grassyterrain"
			state.Field.TerrainTurns = 5
			if normalizedName(active.Item) == "terrainextender" {
				state.Field.TerrainTurns = 8
			}
		}
	case "mistysurge":
		if normalizeTerrainID(state.Field.Terrain) != "mistyterrain" {
			state.Field.Terrain = "mistyterrain"
			state.Field.TerrainTurns = 5
			if normalizedName(active.Item) == "terrainextender" {
				state.Field.TerrainTurns = 8
			}
		}
	case "psychicsurge":
		if normalizeTerrainID(state.Field.Terrain) != "psychicterrain" {
			state.Field.Terrain = "psychicterrain"
			state.Field.TerrainTurns = 5
			if normalizedName(active.Item) == "terrainextender" {
				state.Field.TerrainTurns = 8
			}
		}
	case "intimidate":
		oppActive := opponent.GetActive()
		if oppActive != nil && !oppActive.Fainted {
			oppAbility := normalizedName(oppActive.Ability)
			if oppAbility != "clearbody" && oppAbility != "whitesmoke" && oppAbility != "hypercutter" && oppAbility != "fullmetalbody" {
				curBoost := oppActive.GetBoost(AtkShift)
				if curBoost > -6 {
					oppActive.SetBoost(AtkShift, curBoost-1)
					if oppAbility == "defiant" {
						curAtk := oppActive.GetBoost(AtkShift)
						if curAtk < 6 {
							oppActive.SetBoost(AtkShift, curAtk+2)
						}
					} else if oppAbility == "competitive" {
						curSpa := oppActive.GetBoost(SpaShift)
						if curSpa < 6 {
							oppActive.SetBoost(SpaShift, curSpa+2)
						}
					}
				}
			}
		}
	case "download":
		oppActive := opponent.GetActive()
		if oppActive != nil && !oppActive.Fainted {
			oppDef := getEffectiveStat(state, oppActive, DefShift, &opponent.Side, false)
			oppSpD := getEffectiveStat(state, oppActive, SpdShift, &opponent.Side, false)
			if oppDef < oppSpD {
				curSpa := active.GetBoost(SpaShift)
				if curSpa < 6 {
					active.SetBoost(SpaShift, curSpa+1)
				}
			} else {
				curAtk := active.GetBoost(AtkShift)
				if curAtk < 6 {
					active.SetBoost(AtkShift, curAtk+1)
				}
			}
		}
	}
}

func canActThisTurn(state *BattleState, p *PokemonState) bool {
	if p == nil || p.Fainted {
		return false
	}
	ability := normalizedName(p.Ability)
	if ability == "truant" && p.TurnsActive%2 != 0 {
		return false
	}
	switch p.Status {
	case "par":
		if randomChance(state, 1, 4) {
			return false
		}
	case "slp":
		if p.SleepTurns <= 0 {
			p.SleepTurns = randomInt(state, 3) + 1
		}
		p.SleepTurns--
		if p.SleepTurns <= 0 {
			p.Status = ""
			p.SleepTurns = 0
		}
		return false
	case "frz":
		p.FreezeTurns++
		if randomChance(state, 1, 5) {
			p.Status = ""
			p.FreezeTurns = 0
			return true
		}
		return false
	}
	return true
}

func applyStatusResidual(p *PokemonState) {
	if p == nil || p.Fainted || p.MaxHP <= 0 {
		return
	}
	ability := normalizedName(p.Ability)
	if ability == "magicguard" {
		if p.Status == "tox" {
			p.ToxicCounter++
		}
		return
	}
	switch p.Status {
	case "brn":
		dmg := p.MaxHP / 16
		if dmg < 1 {
			dmg = 1
		}
		p.HP -= dmg
	case "psn":
		if ability == "poisonheal" {
			p.HP += p.MaxHP / 8
			clampHP(p)
			return
		}
		dmg := p.MaxHP / 8
		if dmg < 1 {
			dmg = 1
		}
		p.HP -= dmg
	case "tox":
		p.ToxicCounter++
		if p.ToxicCounter < 1 {
			p.ToxicCounter = 1
		}
		if ability == "poisonheal" {
			p.HP += p.MaxHP / 8
			clampHP(p)
			return
		}
		dmg := (p.MaxHP * p.ToxicCounter) / 16
		if dmg < 1 {
			dmg = 1
		}
		p.HP -= dmg
	}
}

func applyWeatherResidual(state *BattleState, p *PokemonState) {
	if state == nil || p == nil || p.Fainted || p.MaxHP <= 0 {
		return
	}
	weather := normalizeWeatherID(state.Field.Weather)
	ability := normalizedName(p.Ability)

	if weather == "raindance" {
		if ability == "raindish" || ability == "dryskin" {
			p.HP += p.MaxHP / 16
			clampHP(p)
		} else if ability == "hydration" && p.Status != "" {
			clearStatus(p)
		}
	} else if weather == "snowscape" || weather == "hail" {
		if ability == "icebody" {
			p.HP += p.MaxHP / 16
			clampHP(p)
		}
	} else if weather == "sunnyday" {
		if ability == "solarpower" || ability == "dryskin" {
			dmg := p.MaxHP / 8
			if dmg < 1 {
				dmg = 1
			}
			p.HP -= dmg
		}
	}

	if weather != "sandstorm" {
		return
	}
	if ability == "magicguard" || ability == "overcoat" || ability == "sandveil" || ability == "sandrush" || ability == "sandspit" || ability == "sandforce" {
		return
	}
	entry := gamedata.LookupSpecies(p.Species)
	types := getCurrentTypes(p, entry)
	if hasType(types, "Rock") || hasType(types, "Ground") || hasType(types, "Steel") {
		return
	}
	dmg := p.MaxHP / 16
	if dmg < 1 {
		dmg = 1
	}
	p.HP -= dmg
}

func applyTerrainResidual(state *BattleState, p *PokemonState) {
	if state == nil || p == nil || p.Fainted || p.MaxHP <= 0 {
		return
	}
	if normalizeTerrainID(state.Field.Terrain) != "grassyterrain" || !isGrounded(state, p) {
		return
	}
	heal := p.MaxHP / 16
	if heal < 1 {
		heal = 1
	}
	p.HP += heal
	clampHP(p)
}

func applyAbilityResidual(state *BattleState, p *PokemonState) {
	if state == nil || p == nil || p.Fainted || p.MaxHP <= 0 {
		return
	}
	ability := normalizedName(p.Ability)
	if ability == "speedboost" {
		curSpe := p.GetBoost(SpeShift)
		if curSpe < 6 {
			p.SetBoost(SpeShift, curSpe+1)
		}
	} else if ability == "moody" {
		stats := []uint32{AtkShift, DefShift, SpaShift, SpdShift, SpeShift}
		raiseNode := randomInt(state, 5)
		lowerNode := randomInt(state, 5)
		for lowerNode == raiseNode {
			lowerNode = randomInt(state, 5)
		}
		p.SetBoost(stats[raiseNode], p.GetBoost(stats[raiseNode])+2)
		p.SetBoost(stats[lowerNode], p.GetBoost(stats[lowerNode])-1)
	} else if ability == "shedskin" {
		if p.Status != "" && randomChance(state, 33, 100) {
			clearStatus(p)
		}
	}
}

func decrementFieldTimers(field *FieldConditions) {
	if field == nil {
		return
	}
	if field.WeatherTurns > 0 {
		field.WeatherTurns--
		if field.WeatherTurns == 0 {
			field.Weather = ""
		}
	}
	if field.TerrainTurns > 0 {
		field.TerrainTurns--
		if field.TerrainTurns == 0 {
			field.Terrain = ""
		}
	}
	if field.TrickRoomTurns > 0 {
		field.TrickRoomTurns--
		if field.TrickRoomTurns == 0 {
			field.TrickRoom = false
		}
	}
	if field.GravityTurns > 0 {
		field.GravityTurns--
		if field.GravityTurns == 0 {
			field.Gravity = false
		}
	}
}

func actionPriority(state *BattleState, player *PlayerState, action int) int {
	if action >= ActionSwitchBase {
		return 6
	}
	if !IsAttackAction(action) {
		return 0
	}
	active := player.GetActive()
	moveIdx := BaseMoveIndex(action)
	if active == nil || moveIdx < 0 || moveIdx >= active.NumMoves {
		return 0
	}
	move := gamedata.LookupMove(active.Moves[moveIdx])
	if move == nil {
		return 0
	}
	priority := move.Priority
	ability := normalizedName(active.Ability)
	if ability == "prankster" && move.Category == "Status" {
		priority++
	}
	if ability == "triage" && ((move.Healing[0] > 0 && move.Healing[1] > 0) || (move.Drain[0] > 0 && move.Drain[1] > 0)) {
		priority += 3
	}
	if ability == "galewings" && move.Type == "Flying" && active.HP == active.MaxHP {
		priority++
	}
	if normalizedName(active.Moves[moveIdx]) == "grassyglide" && normalizeTerrainID(state.Field.Terrain) == "grassyterrain" {
		priority++
	}
	return priority
}

type queuedAction struct {
	player    *PlayerState
	opponent  *PlayerState
	action    int
	actorIdx  int
	priority  int
	speed     float64
	isMove    bool
	isSwitch  bool
	playerNum int
}

func buildQueuedAction(state *BattleState, player *PlayerState, opponent *PlayerState, action int, playerNum int) queuedAction {
	q := queuedAction{
		player:    player,
		opponent:  opponent,
		action:    action,
		actorIdx:  player.ActiveIdx,
		priority:  actionPriority(state, player, action),
		isMove:    IsAttackAction(action),
		isSwitch:  action >= ActionSwitchBase,
		playerNum: playerNum,
	}
	active := player.GetActive()
	if active != nil {
		q.speed = getEffectiveStat(state, active, SpeShift, &player.Side, false)
	}
	return q
}

func queuedActionOrder(state *BattleState, a queuedAction, b queuedAction) bool {
	if !a.isMove && !a.isSwitch {
		return false
	}
	if !b.isMove && !b.isSwitch {
		return true
	}
	if a.priority != b.priority {
		return a.priority > b.priority
	}
	trickRoomApplies := state.Field.TrickRoom && a.isMove && b.isMove
	if a.speed != b.speed {
		if trickRoomApplies {
			return a.speed < b.speed
		}
		return a.speed > b.speed
	}
	return randomChance(state, 1, 2)
}

// ExecuteSpecificTurn applies two integer actions directly to the BattleState
// 0-3 = Move 1-4, 4-7 = Tera Move 1-4, 8-13 = Switch to slot 0-5
func ExecuteSpecificTurn(state *BattleState, p1Action int, p2Action int) {
	state.Turn++
	for i := 0; i < state.P1.TeamSize; i++ {
		state.P1.Team[i].TookDamageThisTurn = false
		state.P1.Team[i].ActedThisTurn = false
	}
	for i := 0; i < state.P2.TeamSize; i++ {
		state.P2.Team[i].TookDamageThisTurn = false
		state.P2.Team[i].ActedThisTurn = false
	}

	applyTerastallizeIfNeeded(&state.P1, p1Action)
	applyTerastallizeIfNeeded(&state.P2, p2Action)

	p1Queued := buildQueuedAction(state, &state.P1, &state.P2, p1Action, 1)
	p2Queued := buildQueuedAction(state, &state.P2, &state.P1, p2Action, 2)

	p1Flinched := false
	p2Flinched := false
	executeQueued := func(q queuedAction) {
		if q.isSwitch {
			if executeSwitchAction(q.player, q.action) {
				applyOnSwitchInAbilities(state, q.player, q.opponent)
			}
			return
		}
		if !q.isMove {
			return
		}
		if q.player.ActiveIdx != q.actorIdx {
			return
		}
		attacker := q.player.GetActive()
		defender := q.opponent.GetActive()
		if attacker == nil || defender == nil || attacker.Fainted {
			return
		}
		if q.playerNum == 1 && p1Flinched {
			p1Flinched = false
			return
		}
		if q.playerNum == 2 && p2Flinched {
			p2Flinched = false
			return
		}
		moveIdx := BaseMoveIndex(q.action)
		if attacker.Status == "frz" && moveThawsUser(attacker, moveIdx) {
			clearStatus(attacker)
		}
		if !canActThisTurn(state, attacker) {
			return
		}
		attacker.ActedThisTurn = true
		_, defenderFlinched := applyMoveDamage(state, q.player, q.opponent, moveIdx)
		markActiveFainted(q.opponent)
		if defenderFlinched && q.opponent.GetActive() != nil {
			if q.playerNum == 1 {
				p2Flinched = true
			} else {
				p1Flinched = true
			}
		}
	}

	if queuedActionOrder(state, p1Queued, p2Queued) {
		executeQueued(p1Queued)
		executeQueued(p2Queued)
	} else {
		executeQueued(p2Queued)
		executeQueued(p1Queued)
	}

	applyStatusResidual(state.P1.GetActive())
	applyStatusResidual(state.P2.GetActive())
	applyWeatherResidual(state, state.P1.GetActive())
	applyWeatherResidual(state, state.P2.GetActive())
	applyTerrainResidual(state, state.P1.GetActive())
	applyTerrainResidual(state, state.P2.GetActive())
	applyAbilityResidual(state, state.P1.GetActive())
	applyAbilityResidual(state, state.P2.GetActive())

	if act := state.P1.GetActive(); act != nil {
		act.TurnsActive++
	}
	if act := state.P2.GetActive(); act != nil {
		act.TurnsActive++
	}

	markActiveFainted(&state.P1)
	markActiveFainted(&state.P2)
	decrementFieldTimers(&state.Field)
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
func getEffectiveStat(state *BattleState, p *PokemonState, shift uint32, side *SideConditions, ignoreBoosts bool) float64 {
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

	val := base
	if !ignoreBoosts {
		val *= statBoostMultiplier(p.GetBoost(shift))
	}

	ability := normalizedName(p.Ability)
	if ability == "slowstart" && p.TurnsActive < 5 && (shift == AtkShift || shift == SpeShift) {
		val *= 0.5
	}

	weatherID := ""
	if state != nil {
		weatherID = normalizeWeatherID(state.Field.Weather)
	}
	if shift == SpeShift {
		if p.Status == "par" {
			val *= 0.5
		}
		if side != nil && side.Tailwind {
			val *= 2.0
		}
		switch normalizedName(p.Ability) {
		case "swiftswim":
			if weatherID == "raindance" {
				val *= 2.0
			}
		case "chlorophyll":
			if weatherID == "sunnyday" {
				val *= 2.0
			}
		case "sandrush":
			if weatherID == "sandstorm" {
				val *= 2.0
			}
		case "slushrush":
			if weatherID == "snowscape" {
				val *= 2.0
			}
		}
		if normalizedName(p.Item) == "choicescarf" {
			val *= 1.5
		}
		if normalizedName(p.Item) == "ironball" {
			val *= 0.5
		}
	}

	if shift == AtkShift {
		ability := normalizedName(p.Ability)
		if ability == "hugepower" || ability == "purepower" {
			val *= 2.0
		}
		if ability == "guts" && p.Status != "" {
			val *= 1.5
		}
		if ability == "toxicboost" && (p.Status == "psn" || p.Status == "tox") {
			val *= 1.5
		}
		if normalizedName(p.Item) == "choiceband" {
			val *= 1.5
		}
	}
	if shift == DefShift && normalizedName(p.Item) == "eviolite" {
		val *= 1.5
	}
	if shift == SpaShift {
		if normalizedName(p.Ability) == "flareboost" && p.Status == "brn" {
			val *= 1.5
		}
		if normalizedName(p.Item) == "choicespecs" {
			val *= 1.5
		}
	}
	if shift == SpdShift && normalizedName(p.Item) == "assaultvest" {
		val *= 1.5
	}
	if shift == SpdShift && weatherID == "sandstorm" {
		entry := gamedata.LookupSpecies(p.Species)
		if hasType(getCurrentTypes(p, entry), "Rock") {
			val *= 1.5
		}
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

func clearStatus(target *PokemonState) {
	if target == nil {
		return
	}
	target.Status = ""
	target.SleepTurns = 0
	target.FreezeTurns = 0
	target.ToxicCounter = 0
}

func isGrounded(state *BattleState, p *PokemonState) bool {
	if p == nil {
		return false
	}
	if state != nil && state.Field.Gravity {
		return true
	}
	if (p.Volatiles & VolatileIngrain) != 0 {
		return true
	}
	if normalizedName(p.Ability) == "levitate" || normalizedName(p.Item) == "airballoon" {
		return false
	}
	if (p.Volatiles&VolatileMagnetRise) != 0 || (p.Volatiles&VolatileTelekinesis) != 0 {
		return false
	}
	entry := gamedata.LookupSpecies(p.Species)
	if hasType(getCurrentTypes(p, entry), "Flying") {
		return false
	}
	return true
}

func trySetStatus(state *BattleState, target *PokemonState, targetSide *SideConditions, status string) bool {
	if target == nil || target.Fainted {
		return false
	}
	if normalizedName(target.Ability) == "comatose" {
		return false
	}
	status = normalizedName(status)
	if status == "" || target.Status != "" {
		return false
	}
	if targetSide != nil && targetSide.Safeguard {
		return false
	}

	entry := gamedata.LookupSpecies(target.Species)
	types := getCurrentTypes(target, entry)
	terrain := ""
	if state != nil {
		terrain = normalizeTerrainID(state.Field.Terrain)
	}
	if terrain == "mistyterrain" && isGrounded(state, target) {
		return false
	}
	if status == "slp" && terrain == "electricterrain" && isGrounded(state, target) {
		return false
	}

	switch status {
	case "brn":
		if hasType(types, "Fire") || normalizedName(target.Ability) == "waterveil" || normalizedName(target.Ability) == "waterbubble" || normalizedName(target.Ability) == "thermalexchange" {
			return false
		}
	case "par":
		if hasType(types, "Electric") || normalizedName(target.Ability) == "limber" {
			return false
		}
	case "psn", "tox":
		if hasType(types, "Poison") || hasType(types, "Steel") || normalizedName(target.Ability) == "immunity" || normalizedName(target.Ability) == "pastelveil" {
			return false
		}
	case "frz":
		if hasType(types, "Ice") {
			return false
		}
		if state != nil && normalizeWeatherID(state.Field.Weather) == "sunnyday" {
			return false
		}
	case "slp":
		if normalizedName(target.Ability) == "insomnia" || normalizedName(target.Ability) == "vitalspirit" || normalizedName(target.Ability) == "sweetveil" {
			return false
		}
	}

	target.Status = status
	target.SleepTurns = 0
	target.FreezeTurns = 0
	target.ToxicCounter = 0
	if status == "slp" {
		target.SleepTurns = randomInt(state, 3) + 1
	}
	return true
}

func moveThawsUser(attacker *PokemonState, moveIdx int) bool {
	if attacker == nil || moveIdx < 0 || moveIdx >= 4 {
		return false
	}
	move := gamedata.LookupMove(attacker.Moves[moveIdx])
	if move == nil || move.Flags == nil {
		return false
	}
	return move.Flags["defrost"] > 0
}

func moveHitCount(state *BattleState, attacker *PokemonState, moveID string, move *gamedata.MoveEntry) int {
	if move == nil || move.MultiHit == nil {
		return 1
	}
	switch v := move.MultiHit.(type) {
	case int:
		if v > 0 {
			return v
		}
	case float64:
		if int(v) > 0 {
			return int(v)
		}
	case []any:
		if len(v) == 2 {
			minRaw, okMin := v[0].(float64)
			maxRaw, okMax := v[1].(float64)
			if !okMin || !okMax {
				return 1
			}
			minHits := int(minRaw)
			maxHits := int(maxRaw)
			if minHits >= maxHits {
				return minHits
			}
			if normalizedName(attacker.Ability) == "skilllink" {
				return maxHits
			}
			if normalizedName(attacker.Item) == "loadeddice" && maxHits >= 5 {
				if randomChance(state, 1, 2) {
					return 4
				}
				return 5
			}
			if minHits == 2 && maxHits == 5 {
				r := randomInt(state, 100)
				switch {
				case r < 35:
					return 2
				case r < 70:
					return 3
				case r < 85:
					return 4
				default:
					return 5
				}
			}
			return minHits + randomInt(state, maxHits-minHits+1)
		}
	}
	switch moveID {
	case "surgingstrikes", "tripleaxel", "triplekick", "tripledive":
		return 3
	case "populationbomb":
		return 10
	}
	return 1
}

func effectiveMoveType(state *BattleState, attacker *PokemonState, moveID string, move *gamedata.MoveEntry) string {
	if move == nil {
		return ""
	}
	t := move.Type
	switch moveID {
	case "weatherball":
		switch normalizeWeatherID(state.Field.Weather) {
		case "sunnyday":
			return "Fire"
		case "raindance":
			return "Water"
		case "sandstorm":
			return "Rock"
		case "snowscape":
			return "Ice"
		}
	}
	if attacker != nil {
		ability := normalizedName(attacker.Ability)
		if t == "Normal" {
			switch ability {
			case "aerilate":
				return "Flying"
			case "pixilate":
				return "Fairy"
			case "refrigerate":
				return "Ice"
			case "galvanize":
				return "Electric"
			}
		}
		if ability == "liquidvoice" && move.Flags["sound"] == 1 {
			return "Water"
		}
	}
	return t
}

func effectiveBasePower(state *BattleState, attacker *PokemonState, defender *PokemonState, moveID string, move *gamedata.MoveEntry) float64 {
	if move == nil {
		return 80
	}
	power := float64(move.BasePower)
	if power <= 0 {
		return power
	}
	switch moveID {
	case "acrobatics":
		if attacker.Item == "" {
			power *= 2.0
		}
	case "waterspout", "eruption":
		if attacker.MaxHP > 0 {
			power = math.Floor((150.0 * float64(attacker.HP)) / float64(attacker.MaxHP))
		}
	case "hex":
		if defender.Status != "" || normalizedName(defender.Ability) == "comatose" {
			power = 130
		}
	case "venoshock":
		if defender.Status == "psn" || defender.Status == "tox" {
			power = 130
		}
	case "facade":
		if attacker.Status == "brn" || attacker.Status == "par" || attacker.Status == "psn" || attacker.Status == "tox" {
			power = 140
		}
	case "assurance":
		if defender.TookDamageThisTurn {
			power *= 2.0
		}
	case "avalanche", "revenge":
		if attacker.TookDamageThisTurn {
			power *= 2.0
		}
	case "flail", "reversal":
		if attacker.MaxHP > 0 {
			ratio := (attacker.HP * 48) / attacker.MaxHP
			switch {
			case ratio <= 1:
				power = 200
			case ratio <= 4:
				power = 150
			case ratio <= 9:
				power = 100
			case ratio <= 16:
				power = 80
			case ratio <= 32:
				power = 40
			default:
				power = 20
			}
		}
	case "electroball":
		attackerSpeed := getEffectiveStat(state, attacker, SpeShift, nil, false)
		defenderSpeed := getEffectiveStat(state, defender, SpeShift, nil, false)
		if defenderSpeed <= 0 {
			power = 150
		} else {
			ratio := attackerSpeed / defenderSpeed
			switch {
			case ratio >= 4:
				power = 150
			case ratio >= 3:
				power = 120
			case ratio >= 2:
				power = 80
			case ratio >= 1:
				power = 60
			default:
				power = 40
			}
		}
	case "gyroball":
		attackerSpeed := getEffectiveStat(state, attacker, SpeShift, nil, false)
		defenderSpeed := getEffectiveStat(state, defender, SpeShift, nil, false)
		if attackerSpeed <= 0 {
			power = 150
		} else {
			power = math.Floor((25.0 * defenderSpeed) / attackerSpeed)
			if power > 150 {
				power = 150
			}
			if power < 1 {
				power = 1
			}
		}
	case "weatherball":
		if normalizeWeatherID(state.Field.Weather) != "" {
			power = 100
		}
	}
	if move.BasePowerCallback && power < 1 {
		power = 1
	}
	return power
}

func setWeatherFromMove(state *BattleState, attacker *PokemonState, move *gamedata.MoveEntry) {
	if state == nil || move == nil || move.Weather == "" {
		return
	}
	weather := normalizeWeatherID(move.Weather)
	if weather == "" {
		return
	}
	duration := 5
	switch weather {
	case "sunnyday":
		if normalizedName(attacker.Item) == "heatrock" {
			duration = 8
		}
	case "raindance":
		if normalizedName(attacker.Item) == "damprock" {
			duration = 8
		}
	case "sandstorm":
		if normalizedName(attacker.Item) == "smoothrock" {
			duration = 8
		}
	case "snowscape":
		if normalizedName(attacker.Item) == "icyrock" {
			duration = 8
		}
	}
	state.Field.Weather = weather
	state.Field.WeatherTurns = duration
}

func setTerrainOrPseudoWeatherFromMove(state *BattleState, attacker *PokemonState, move *gamedata.MoveEntry) {
	if state == nil || move == nil {
		return
	}
	if move.Terrain != "" {
		terrain := normalizeTerrainID(move.Terrain)
		if terrain != "" {
			duration := 5
			if normalizedName(attacker.Item) == "terrainextender" {
				duration = 8
			}
			state.Field.Terrain = terrain
			state.Field.TerrainTurns = duration
		}
	}
	switch normalizedName(move.PseudoWeather) {
	case "trickroom":
		state.Field.TrickRoom = true
		state.Field.TrickRoomTurns = 5
	case "gravity":
		state.Field.Gravity = true
		state.Field.GravityTurns = 5
	}
}

func switchToRandomBench(state *BattleState, player *PlayerState) bool {
	if state == nil || player == nil {
		return false
	}
	choices := 0
	for i := 0; i < player.TeamSize; i++ {
		if i != player.ActiveIdx && !player.Team[i].Fainted {
			choices++
		}
	}
	if choices == 0 {
		return false
	}
	choice := randomInt(state, choices)
	for i := 0; i < player.TeamSize; i++ {
		if i == player.ActiveIdx || player.Team[i].Fainted {
			continue
		}
		if choice == 0 {
			return executeSwitchAction(player, ActionSwitchBase+i)
		}
		choice--
	}
	return false
}

func applySecondary(state *BattleState, sec gamedata.MoveSecondary, attackerPlayer *PlayerState, defenderPlayer *PlayerState) bool {
	attacker := attackerPlayer.GetActive()
	defender := defenderPlayer.GetActive()
	if attacker == nil || defender == nil {
		return false
	}
	chance := sec.Chance
	if chance > 0 && normalizedName(attacker.Ability) == "serenegrace" {
		chance *= 2
	}
	if chance > 0 && !randomChance(state, chance, 100) {
		return false
	}
	if sec.Status != "" {
		if trySetStatus(state, defender, &defenderPlayer.Side, sec.Status) {
			if normalizedName(defender.Ability) == "synchronize" && (sec.Status == "brn" || sec.Status == "par" || sec.Status == "psn" || sec.Status == "tox") {
				trySetStatus(state, attacker, &attackerPlayer.Side, sec.Status)
			}
		}
	}
	if sec.VolatileStatus == "flinch" {
		return !defender.ActedThisTurn
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

func applyMoveEffects(state *BattleState, moveID string, move *gamedata.MoveEntry, attackerPlayer *PlayerState, defenderPlayer *PlayerState, damageDealt int) bool {
	attacker := attackerPlayer.GetActive()
	defender := defenderPlayer.GetActive()
	if move == nil || attacker == nil || defender == nil {
		return false
	}
	flinch := false

	setWeatherFromMove(state, attacker, move)
	setTerrainOrPseudoWeatherFromMove(state, attacker, move)
	if move.ThawsTarget && defender.Status == "frz" {
		clearStatus(defender)
	}

	if move.Status != "" {
		if move.Target == "self" {
			trySetStatus(state, attacker, &attackerPlayer.Side, move.Status)
		} else {
			if trySetStatus(state, defender, &defenderPlayer.Side, move.Status) {
				if normalizedName(defender.Ability) == "synchronize" && (move.Status == "brn" || move.Status == "par" || move.Status == "psn" || move.Status == "tox") {
					trySetStatus(state, attacker, &attackerPlayer.Side, move.Status)
				}
			}
		}
	}
	if move.VolatileStatus == "flinch" {
		flinch = !defender.ActedThisTurn
	} else if move.VolatileStatus != "" {
		if move.Target == "self" {
			attacker.ToggleVolatile(MapVolatileToBit(move.VolatileStatus), true)
		} else {
			defender.ToggleVolatile(MapVolatileToBit(move.VolatileStatus), true)
		}
	}

	if len(move.Boosts) > 0 {
		if move.Target == "self" {
			applyBoostMap(attacker, move.Boosts)
		} else {
			applyBoostMap(defender, move.Boosts)
		}
	}
	if move.Self != nil {
		applyBoostMap(attacker, move.Self.Boosts)
	}
	if move.SelfBoost != nil && damageDealt > 0 {
		applyBoostMap(attacker, move.SelfBoost.Boosts)
	}

	applySec := true
	if normalizedName(attacker.Ability) == "sheerforce" && (move.Secondary != nil || len(move.Secondaries) > 0) {
		applySec = false
	}

	if applySec {
		if move.Secondary != nil {
			if applySecondary(state, *move.Secondary, attackerPlayer, defenderPlayer) {
				flinch = true
			}
		}
		for _, sec := range move.Secondaries {
			if applySecondary(state, sec, attackerPlayer, defenderPlayer) {
				flinch = true
			}
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

	if move.ForceSwitch && (damageDealt > 0 || move.Category == "Status") && defender.HP > 0 {
		switchToRandomBench(state, defenderPlayer)
	}
	if move.SelfSwitch != nil && damageDealt > 0 && attacker.HP > 0 {
		switchToRandomBench(state, attackerPlayer)
	}
	if move.Selfdestruct != "" {
		if normalizedName(move.Selfdestruct) != "ifhit" || damageDealt > 0 {
			attacker.HP = 0
			attacker.Fainted = true
		}
	}

	clampHP(attacker)
	clampHP(defender)
	_ = moveID

	if damageDealt > 0 && !flinch && normalizedName(attacker.Ability) == "stench" {
		if randomChance(state, 10, 100) {
			flinch = true
		}
	}

	return flinch
}

// applyMoveDamage applies a single move resolution.
// Returns (moveHit, defenderFlinched).
func applyMoveDamage(state *BattleState, attackerPlayer *PlayerState, defenderPlayer *PlayerState, moveIdx int) (bool, bool) {
	attacker := attackerPlayer.GetActive()
	defender := defenderPlayer.GetActive()
	if attacker == nil || defender == nil || moveIdx < 0 || moveIdx >= 4 {
		return false, false
	}

	moveID := normalizedName(attacker.Moves[moveIdx])
	moveEntry := gamedata.LookupMove(attacker.Moves[moveIdx])
	if moveEntry == nil {
		moveEntry = gamedata.LookupMove(moveID)
	}
	moveType := "Normal"
	moveCategory := "Physical"
	movePriority := 0
	if moveEntry != nil {
		moveType = effectiveMoveType(state, attacker, moveID, moveEntry)
		moveCategory = moveEntry.Category
		movePriority = moveEntry.Priority
	}

	if normalizeTerrainID(state.Field.Terrain) == "psychicterrain" && movePriority > 0 && isGrounded(state, defender) {
		return true, false
	}

	atkAbility := normalizedName(attacker.Ability)
	ability := normalizedName(defender.Ability)
	if atkAbility == "moldbreaker" || atkAbility == "teravolt" || atkAbility == "turboblaze" {
		ability = ""
	}
	defEntry := gamedata.LookupSpecies(defender.Species)
	switch {
	case moveType == "Ground" && !isGrounded(state, defender):
		return true, false
	case moveType == "Fire" && ability == "flashfire":
		return true, false
	case moveType == "Fire" && ability == "wellbakedbody":
		defender.SetBoost(DefShift, defender.GetBoost(DefShift)+2)
		return true, false
	case moveType == "Water" && (ability == "waterabsorb" || ability == "stormdrain" || ability == "dryskin"):
		defender.HP += defender.MaxHP / 4
		clampHP(defender)
		return true, false
	case moveType == "Electric" && (ability == "voltabsorb" || ability == "lightningrod"):
		defender.HP += defender.MaxHP / 4
		clampHP(defender)
		return true, false
	case moveType == "Electric" && ability == "motordrive":
		defender.SetBoost(SpeShift, defender.GetBoost(SpeShift)+1)
		return true, false
	case moveType == "Grass" && ability == "sapsipper":
		defender.SetBoost(AtkShift, defender.GetBoost(AtkShift)+1)
		return true, false
	case moveType == "Ground" && ability == "eartheater":
		defender.HP += defender.MaxHP / 4
		clampHP(defender)
		return true, false
	case ability == "bulletproof" && moveEntry != nil && moveEntry.Flags["bullet"] == 1:
		return true, false
	case ability == "soundproof" && moveEntry != nil && moveEntry.Flags["sound"] == 1:
		return true, false
	case (ability == "dazzling" || ability == "queenlymajesty" || ability == "armortail") && movePriority > 0:
		return true, false
	case ability == "windrider" && moveEntry != nil && moveEntry.Flags["wind"] == 1:
		defender.SetBoost(AtkShift, defender.GetBoost(AtkShift)+1)
		return true, false
	case ability == "wonderguard" && moveCategory != "Status" && moveType != "":
		eff := gamedata.CalcTypeEffectiveness(moveType, getCurrentTypes(defender, defEntry))
		if eff <= 1.0 {
			return true, false
		}
	}

	perHitAccuracy := moveID == "tripleaxel" || moveID == "triplekick" || moveID == "populationbomb"
	if moveEntry != nil && !perHitAccuracy && !moveHits(state, attacker, defender, moveEntry) {
		return false, false
	}

	if moveCategory == "Status" {
		flinch := applyMoveEffects(state, moveID, moveEntry, attackerPlayer, defenderPlayer, 0)
		clampHP(attacker)
		clampHP(defender)
		return true, flinch
	}

	atkEntry := gamedata.LookupSpecies(attacker.Species)
	if atkEntry == nil || defEntry == nil {
		defender.HP -= 15
		defender.TookDamageThisTurn = true
		clampHP(defender)
		return true, false
	}

	power := 80.0
	if moveEntry != nil {
		power = effectiveBasePower(state, attacker, defender, moveID, moveEntry)
	}
	if power <= 0 {
		flinch := applyMoveEffects(state, moveID, moveEntry, attackerPlayer, defenderPlayer, 0)
		clampHP(attacker)
		clampHP(defender)
		return true, flinch
	}

	isPhysical := moveCategory != "Special"

	if ability == "disguise" && !defender.DisguiseBroken {
		defender.DisguiseBroken = true
		defender.HP -= defender.MaxHP / 8
		clampHP(defender)
		return true, false
	}
	if ability == "iceface" && !defender.IceFaceBroken && isPhysical {
		defender.IceFaceBroken = true
		return true, false
	}

	effectiveness := 1.0
	if moveType != "" {
		effectiveness = gamedata.CalcTypeEffectiveness(moveType, getCurrentTypes(defender, defEntry))
	}
	if effectiveness == 0 {
		return true, false
	}

	totalDamage := 0
	flinch := false
	hits := moveHitCount(state, attacker, moveID, moveEntry)
	if hits < 1 {
		hits = 1
	}
	for hit := 1; hit <= hits; hit++ {
		if moveEntry != nil && perHitAccuracy && !moveHits(state, attacker, defender, moveEntry) {
			if hit == 1 {
				return false, false
			}
			break
		}
		hitPower := power
		if moveID == "tripleaxel" || moveID == "triplekick" {
			hitPower = power * float64(hit)
		}

		critical := moveIsCritical(state, attacker, moveEntry)

		ignoreAtkBoosts := false
		ignoreDefBoosts := false

		if ability == "unaware" {
			ignoreAtkBoosts = true
		}
		if atkAbility == "unaware" {
			ignoreDefBoosts = true
		}
		if critical {
			ignoreDefBoosts = true
			if isPhysical && attacker.GetBoost(AtkShift) < 0 {
				ignoreAtkBoosts = true
			} else if !isPhysical && attacker.GetBoost(SpaShift) < 0 {
				ignoreAtkBoosts = true
			}
		}

		attackStat := 1.0
		defenseStat := 1.0
		if isPhysical {
			attackStat = getEffectiveStat(state, attacker, AtkShift, &attackerPlayer.Side, ignoreAtkBoosts)
			defenseStat = getEffectiveStat(state, defender, DefShift, &defenderPlayer.Side, ignoreDefBoosts)
		} else {
			attackStat = getEffectiveStat(state, attacker, SpaShift, &attackerPlayer.Side, ignoreAtkBoosts)
			defenseStat = getEffectiveStat(state, defender, SpdShift, &defenderPlayer.Side, ignoreDefBoosts)
		}
		if defenseStat < 1 {
			defenseStat = 1
		}

		level := 100.0
		if attacker.Level > 0 {
			level = float64(attacker.Level)
		}
		baseDamage := math.Floor((math.Floor((2.0*level)/5.0+2.0) * hitPower * attackStat / defenseStat / 50.0) + 2.0)
		if baseDamage < 1 {
			baseDamage = 1
		}

		modifier := float64(randomInt(state, 16)+85) / 100.0
		if critical {
			modifier *= 1.5
		}
		if moveType != "" {
			modifier *= stabModifier(attacker, atkEntry, moveType)
			modifier *= effectiveness
		}

		weather := normalizeWeatherID(state.Field.Weather)
		if weather == "sunnyday" {
			if moveType == "Fire" {
				modifier *= 1.5
			}
			if moveType == "Water" {
				modifier *= 0.5
			}
		}
		if weather == "raindance" {
			if moveType == "Water" {
				modifier *= 1.5
			}
			if moveType == "Fire" {
				modifier *= 0.5
			}
		}

		terrain := normalizeTerrainID(state.Field.Terrain)
		if terrain == "electricterrain" && moveType == "Electric" && isGrounded(state, attacker) {
			modifier *= 1.3
		}
		if terrain == "grassyterrain" && moveType == "Grass" && isGrounded(state, attacker) {
			modifier *= 1.3
		}
		if terrain == "psychicterrain" && moveType == "Psychic" && isGrounded(state, attacker) {
			modifier *= 1.3
		}
		if terrain == "mistyterrain" && moveType == "Dragon" && isGrounded(state, defender) {
			modifier *= 0.5
		}
		if terrain == "grassyterrain" && (moveID == "earthquake" || moveID == "bulldoze" || moveID == "magnitude") && isGrounded(state, defender) {
			modifier *= 0.5
		}

		if isPhysical && attacker.Status == "brn" && normalizedName(attacker.Ability) != "guts" {
			modifier *= 0.5
		}

		atkAbility := normalizedName(attacker.Ability)
		if atkAbility == "technician" && hitPower <= 60 {
			modifier *= 1.5
		} else if atkAbility == "aerilate" && moveEntry != nil && moveEntry.Type == "Normal" {
			modifier *= 1.2
		} else if atkAbility == "pixilate" && moveEntry != nil && moveEntry.Type == "Normal" {
			modifier *= 1.2
		} else if atkAbility == "refrigerate" && moveEntry != nil && moveEntry.Type == "Normal" {
			modifier *= 1.2
		} else if atkAbility == "galvanize" && moveEntry != nil && moveEntry.Type == "Normal" {
			modifier *= 1.2
		} else if atkAbility == "overgrow" && moveType == "Grass" && attacker.HP <= attacker.MaxHP/3 {
			modifier *= 1.5
		} else if atkAbility == "blaze" && moveType == "Fire" && attacker.HP <= attacker.MaxHP/3 {
			modifier *= 1.5
		} else if atkAbility == "torrent" && moveType == "Water" && attacker.HP <= attacker.MaxHP/3 {
			modifier *= 1.5
		} else if atkAbility == "swarm" && moveType == "Bug" && attacker.HP <= attacker.MaxHP/3 {
			modifier *= 1.5
		} else if atkAbility == "toughclaws" && moveEntry != nil && moveEntry.Flags["contact"] == 1 {
			modifier *= 1.3
		} else if atkAbility == "strongjaw" && moveEntry != nil && moveEntry.Flags["bite"] == 1 {
			modifier *= 1.5
		} else if atkAbility == "ironfist" && moveEntry != nil && moveEntry.Flags["punch"] == 1 {
			modifier *= 1.2
		} else if atkAbility == "megalauncher" && moveEntry != nil && moveEntry.Flags["pulse"] == 1 {
			modifier *= 1.5
		} else if atkAbility == "sharpness" && moveEntry != nil && moveEntry.Flags["slicing"] == 1 {
			modifier *= 1.5
		} else if atkAbility == "punkrock" && moveEntry != nil && moveEntry.Flags["sound"] == 1 {
			modifier *= 1.3
		} else if atkAbility == "reckless" && moveEntry != nil && (moveEntry.Recoil[0] > 0 || moveID == "jumpkick" || moveID == "highjumpkick" || moveID == "axekick") {
			modifier *= 1.2
		} else if atkAbility == "sheerforce" && moveEntry != nil && (moveEntry.Secondary != nil || len(moveEntry.Secondaries) > 0) {
			modifier *= 1.3
		} else if atkAbility == "sniper" && critical {
			modifier *= 1.5
		}

		if moveEntry != nil && moveEntry.Flags["sound"] == 1 && ability == "punkrock" {
			modifier *= 0.5
		}
		if moveEntry != nil && moveEntry.Flags["contact"] == 1 && ability == "fluffy" {
			modifier *= 0.5
		}
		if moveType == "Fire" && ability == "fluffy" {
			modifier *= 2.0
		}

		if !critical {
			if isPhysical && (defenderPlayer.Side.Reflect || defenderPlayer.Side.AuroraVeil) {
				modifier *= 0.5
			} else if !isPhysical && (defenderPlayer.Side.LightScreen || defenderPlayer.Side.AuroraVeil) {
				modifier *= 0.5
			}
		}
		if (moveType == "Fire" || moveType == "Ice") && normalizedName(defender.Ability) == "thickfat" {
			modifier *= 0.5
		}
		if effectiveness > 1 {
			defenderAbility := normalizedName(defender.Ability)
			if defenderAbility == "filter" || defenderAbility == "solidrock" || defenderAbility == "prismarmor" {
				modifier *= 0.75
			}
			if normalizedName(attacker.Item) == "expertbelt" {
				modifier *= 1.2
			}
		}

		if normalizedName(attacker.Item) == "lifeorb" {
			modifier *= 1.3
		}
		if isPhysical && normalizedName(attacker.Item) == "muscleband" {
			modifier *= 1.1
		}
		if !isPhysical && normalizedName(attacker.Item) == "wiseglasses" {
			modifier *= 1.1
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
		if finalDamage > 0 {
			defender.TookDamageThisTurn = true
		}
		totalDamage += finalDamage
		clampHP(defender)
		if defender.HP <= 0 {
			break
		}
	}

	flinch = applyMoveEffects(state, moveID, moveEntry, attackerPlayer, defenderPlayer, totalDamage)
	clampHP(attacker)
	return true, flinch
}
