package simulator

import (
	"math/rand"

	"github.com/pokemon-engine/gamedata"
)

// RandomSpecies returns a random valid species key from the Pokedex
func RandomSpecies() string {
	if len(gamedata.Pokedex) == 0 {
		return ""
	}
	keys := make([]string, 0, len(gamedata.Pokedex))
	for k := range gamedata.Pokedex {
		keys = append(keys, k)
	}
	return keys[rand.Intn(len(keys))]
}

// GenerateRandomPlayerState establishes a randomized player team.
func GenerateRandomPlayerState(playerID string) PlayerState {
	player := PlayerState{
		ID:        playerID,
		ActiveIdx: -1,
		TeamSize:  6,
	}

	for i := 0; i < 6; i++ {
		species := RandomSpecies()
		entry := gamedata.Pokedex[species]

		// Minimum sane defaults for self-play
		player.Team[i] = PokemonState{
			Name:     entry.Name,
			Species:  entry.Name,
			TeraType: gamedata.AllTypes[rand.Intn(len(gamedata.AllTypes))],
			MaxHP:    entry.BaseStats.HP*2 + 141, // Rough lvl 100 calc mapping
			HP:       entry.BaseStats.HP*2 + 141,
			Status:   "",
			IsActive: false,
			Fainted:  false,
			Boosts:   (6 << AtkShift) | (6 << DefShift) | (6 << SpaShift) | (6 << SpdShift) | (6 << SpeShift) | (6 << EvaShift) | (6 << AccShift),
		}
	}
	player.CanTerastallize = true
	return player
}

// NewRandomBattleState creates a completely automated game state seeded with random Pokemon.
func NewRandomBattleState() *BattleState {
	state := &BattleState{
		Turn:     0,
		P1:       GenerateRandomPlayerState("p1"),
		P2:       GenerateRandomPlayerState("p2"),
		RNGState: rand.Uint64(),
	}

	// Make the first slot active
	state.P1.Team[0].IsActive = true
	state.P1.ActiveIdx = 0

	state.P2.Team[0].IsActive = true
	state.P2.ActiveIdx = 0

	return state
}
