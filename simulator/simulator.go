package simulator

import (
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
	switch v {
	case "Substitute":
		return VolatileSubstitute
	case "confusion":
		return VolatileConfusion
	case "Encore":
		return VolatileEncore
	case "Taunt":
		return VolatileTaunt
	case "Leech Seed":
		return VolatileLeechSeed
	case "Perish Song":
		return VolatilePerishSong
	case "Attract":
		return VolatileAttract
	case "Focus Energy":
		return VolatileFocusEnergy
	case "Destiny Bond":
		return VolatileDestinyBond
	case "Magnet Rise":
		return VolatileMagnetRise
	case "Telekinesis":
		return VolatileTelekinesis
	case "Gastro Acid":
		return VolatileGastroAcid
	case "Ingrain":
		return VolatileIngrain
	case "Aqua Ring":
		return VolatileAquaRing
	case "Curse":
		return VolatileCurse
	case "Embargo":
		return VolatileEmbargo
	case "Heal Block":
		return VolatileHealBlock
	case "Protect", "Endure", "Spiky Shield", "Baneful Bunker", "King's Shield", "Silk Trap", "Obstruct", "Burning Bulwark":
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
	Turn  int
	P1    PlayerState
	P2    PlayerState
	Field FieldConditions
}

// Action represents a possible move or switch
type Action struct {
	Type string // "move" or "switch"
	Name string // The move name or the Pokemon to switch to
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
		// Reset boost correctly (+6 to 0 shift = 6)
		// For 7 stats, that is (6<<0)|(6<<4)|(6<<8)|(6<<12)|(6<<16)|(6<<20)|(6<<24) = 114420174
		p.Team[idx] = PokemonState{
			Name:     species,
			Species:  species,
			HP:       100,
			MaxHP:    100,
			IsActive: false,
			Fainted:  false,
			Boosts:   114420174,
		}
		idx++
	}
	p.TeamSize = idx
	p.ActiveIdx = -1
	p.CanTerastallize = true
}

func findPokemonByName(p *PlayerState, name string) int {
	for i := 0; i < p.TeamSize; i++ {
		if p.Team[i].Name == name || p.Team[i].Species == name {
			return i
		}
	}
	return -1
}

// FastForwardToEvent reconstructs the state exactly up to (and including) the specified event index.
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
					newIdx := playerState.TeamSize
					playerState.Team[newIdx] = PokemonState{
						Name:    speciesName,
						Species: speciesName,
						HP:      100, MaxHP: 100,
						IsActive: true, Fainted: false,
						Boosts: 114420174,
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
			// We no longer guess faints from HP drops because Illusion (Zoroark)
			// causes the disguised pokemon's name to drop to 0 HP before the 'replace' event.
			// Showdown always logs an explicit 'faint' event after 'replace', which we handle natively.

			// However, we MUST handle revives. If a Pokemon is healed (e.g. Revival Blessing), it is no longer fainted.
			if event.Type == "heal" && event.Value != "" {
				idx := findPokemonByName(playerState, event.Value)
				if idx != -1 {
					playerState.Team[idx].Fainted = false
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
				playerState.Team[playerState.ActiveIdx].Boosts = 114420174
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

		var firstAtk, secondAtk *PokemonState
		var firstDef, secondDef *PokemonState
		var firstSide, secondSide *SideConditions
		var firstMove, secondMove int

		if p1Spe > p2Spe || (p1Spe == p2Spe && p1Active.HP >= p2Active.HP) {
			firstAtk, firstDef, firstMove = p1Active, p2Active, p1Action
			firstSide, secondSide = &state.P1.Side, &state.P2.Side
			secondAtk, secondDef, secondMove = p2Active, p1Active, p2Action
		} else {
			firstAtk, firstDef, firstMove = p2Active, p1Active, p2Action
			firstSide, secondSide = &state.P2.Side, &state.P1.Side
			secondAtk, secondDef, secondMove = p1Active, p2Active, p1Action
		}

		applyMoveDamage(firstAtk, firstDef, BaseMoveIndex(firstMove), secondSide)
		if firstDef.HP <= 0 {
			firstDef.HP = 0
			firstDef.Fainted = true
		} else {
			applyMoveDamage(secondAtk, secondDef, BaseMoveIndex(secondMove), firstSide)
		}
	} else {
		if p1Attacks && p2Active != nil {
			applyMoveDamage(p1Active, p2Active, BaseMoveIndex(p1Action), &state.P2.Side)
		} else if p2Attacks && p1Active != nil {
			applyMoveDamage(p2Active, p1Active, BaseMoveIndex(p2Action), &state.P1.Side)
		}
	}

	// Check for faints
	if p1Active != nil && p1Active.HP <= 0 {
		p1Active.HP = 0
		p1Active.Fainted = true
		p1Active.IsActive = false
		state.P1.ActiveIdx = -1
	}
	if p2Active != nil && p2Active.HP <= 0 {
		p2Active.HP = 0
		p2Active.Fainted = true
		p2Active.IsActive = false
		state.P2.ActiveIdx = -1
	}
}

// getEffectiveStat computes the stat including base stats, boosts, and status conditions.
func getEffectiveStat(p *PokemonState, shift uint32, side *SideConditions) float64 {
	base := 0.0
	// 1. Check if actual stats are already computed
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

	// 2. Fallback to base stat calculation if actual stats are missing
	if base == 0 {
		entry := gamedata.LookupSpecies(p.Species)
		if entry != nil {
			lvl := 100
			if p.Level > 0 {
				lvl = p.Level
			}

			var b int
			switch shift {
			case AtkShift:
				b = entry.BaseStats.Atk
			case DefShift:
				b = entry.BaseStats.Def
			case SpaShift:
				b = entry.BaseStats.SpA
			case SpdShift:
				b = entry.BaseStats.SpD
			case SpeShift:
				b = entry.BaseStats.Spe
			}
			base = float64(((2*b+31)*lvl)/100 + 5)
		} else {
			base = 105
		}
	}

	boost := p.GetBoost(shift)
	var multiplier float64
	if boost >= 0 {
		multiplier = float64(2+boost) / 2.0
	} else {
		multiplier = 2.0 / float64(2-boost)
	}

	val := base * multiplier
	if shift == SpeShift {
		if p.Status == "par" {
			val *= 0.5
		}
		if side != nil && side.Tailwind {
			val *= 2.0
		}
	}

	if shift == AtkShift {
		if p.Status == "brn" {
			val *= 0.5
		}
		if p.Ability == "hugepower" || p.Ability == "purepower" {
			val *= 2.0
		}
		if p.Item == "choiceband" {
			val *= 1.5
		}
	}
	if shift == DefShift && p.Item == "eviolite" {
		val *= 1.5
	}
	if shift == SpaShift && p.Item == "choicespecs" {
		val *= 1.5
	}
	if shift == SpdShift && p.Item == "assaultvest" {
		val *= 1.5
	}

	if val < 1 {
		val = 1
	}
	return val
}

// applyMoveDamage calculates type-aware damage from attacker to defender using actual move data.
func applyMoveDamage(attacker *PokemonState, defender *PokemonState, moveIdx int, side *SideConditions) {
	atkEntry := gamedata.LookupSpecies(attacker.Species)
	defEntry := gamedata.LookupSpecies(defender.Species)

	var moveEntry *gamedata.MoveEntry
	if moveIdx >= 0 && moveIdx < 4 {
		moveEntry = gamedata.LookupMove(attacker.Moves[moveIdx])
	}

	if moveEntry != nil && moveEntry.Category == "Status" {
		return
	}

	if atkEntry == nil || defEntry == nil {
		defender.HP -= 15
		return
	}

	power := 80.0
	moveType := ""
	isPhys := true

	if moveEntry != nil && moveEntry.BasePower > 0 {
		power = float64(moveEntry.BasePower)
		moveType = moveEntry.Type
		isPhys = moveEntry.Category == "Physical"
	}

	var atkStat, defStat float64
	if isPhys {
		atkStat = getEffectiveStat(attacker, AtkShift, nil)
		defStat = getEffectiveStat(defender, DefShift, side)
	} else {
		atkStat = getEffectiveStat(attacker, SpaShift, nil)
		defStat = getEffectiveStat(defender, SpdShift, side)
	}

	ratio := atkStat / defStat
	level := 100.0
	if attacker.Level > 0 {
		level = float64(attacker.Level)
	}

	// Gen 9 damage formula
	damage := ((2.0*level/5.0+2.0)*power*ratio)/50.0 + 2.0
	modifiers := 0.925 // average random roll

	// STAB (Gen 9 Terastallization-aware)
	if moveType != "" {
		if attacker.Terastallized && attacker.TeraType != "" {
			if moveType == attacker.TeraType {
				stab := 1.5
				for _, t := range atkEntry.Types {
					if t == moveType {
						stab = 2.0
						break
					}
				}
				modifiers *= stab
			}
		} else {
			for _, t := range atkEntry.Types {
				if t == moveType {
					modifiers *= 1.5
					break
				}
			}
		}
	}

	// Type Effectiveness
	if moveType != "" {
		modifiers *= gamedata.CalcTypeEffectiveness(moveType, getCurrentTypes(defender, defEntry))
	}

	// Item/Ability/Gender Modifiers
	if attacker.Item == "lifeorb" {
		modifiers *= 1.3
	}
	if attacker.Ability == "rivalry" && attacker.Gender != "N" && defender.Gender != "N" {
		if attacker.Gender == defender.Gender {
			modifiers *= 1.25
		} else {
			modifiers *= 0.75
		}
	}

	// Screens
	if side != nil {
		if isPhys && (side.Reflect || side.AuroraVeil) {
			modifiers *= 0.5
		} else if !isPhys && (side.LightScreen || side.AuroraVeil) {
			modifiers *= 0.5
		}
	}

	finalDamage := damage * modifiers
	defenderMaxHP := float64(defender.MaxHP)
	if defenderMaxHP == 0 {
		defenderMaxHP = float64(defEntry.BaseStats.HP*2 + 141)
	}
	damagePercent := (finalDamage / defenderMaxHP) * 100.0
	defender.HP -= int(damagePercent)
}
