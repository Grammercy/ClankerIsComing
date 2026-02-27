package gamedata

import (
	"math"
)

// AllTypes is the canonical ordering of the 18 Pokemon types.
// This ordering is used for the one-hot encoding in the feature vector.
var AllTypes = []string{
	"Normal", "Fire", "Water", "Electric", "Grass", "Ice",
	"Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
	"Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
}

// TypeIndex maps type name -> index in AllTypes for fast lookup
var TypeIndex map[string]int

// TypeChart holds the effectiveness multiplier: TypeChart[attacker][defender] = multiplier
// 0.0 = immune, 0.5 = not very effective, 1.0 = neutral, 2.0 = super effective
var TypeChart map[string]map[string]float64

func init() {
	// Build type index
	TypeIndex = make(map[string]int, len(AllTypes))
	for i, t := range AllTypes {
		TypeIndex[t] = i
	}

	// Build the full 18x18 type effectiveness chart (Gen 6+ with Fairy)
	TypeChart = map[string]map[string]float64{
		"Normal": {
			"Rock": 0.5, "Ghost": 0.0, "Steel": 0.5,
		},
		"Fire": {
			"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 2.0,
			"Bug": 2.0, "Rock": 0.5, "Dragon": 0.5, "Steel": 2.0,
		},
		"Water": {
			"Fire": 2.0, "Water": 0.5, "Grass": 0.5, "Ground": 2.0,
			"Rock": 2.0, "Dragon": 0.5,
		},
		"Electric": {
			"Water": 2.0, "Electric": 0.5, "Grass": 0.5, "Ground": 0.0,
			"Flying": 2.0, "Dragon": 0.5,
		},
		"Grass": {
			"Fire": 0.5, "Water": 2.0, "Grass": 0.5, "Poison": 0.5,
			"Ground": 2.0, "Flying": 0.5, "Bug": 0.5, "Rock": 2.0,
			"Dragon": 0.5, "Steel": 0.5,
		},
		"Ice": {
			"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 0.5,
			"Ground": 2.0, "Flying": 2.0, "Dragon": 2.0, "Steel": 0.5,
		},
		"Fighting": {
			"Normal": 2.0, "Ice": 2.0, "Poison": 0.5, "Flying": 0.5,
			"Psychic": 0.5, "Bug": 0.5, "Rock": 2.0, "Ghost": 0.0,
			"Dark": 2.0, "Steel": 2.0, "Fairy": 0.5,
		},
		"Poison": {
			"Grass": 2.0, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5,
			"Ghost": 0.5, "Steel": 0.0, "Fairy": 2.0,
		},
		"Ground": {
			"Fire": 2.0, "Electric": 2.0, "Grass": 0.5, "Poison": 2.0,
			"Flying": 0.0, "Bug": 0.5, "Rock": 2.0, "Steel": 2.0,
		},
		"Flying": {
			"Electric": 0.5, "Grass": 2.0, "Fighting": 2.0, "Bug": 2.0,
			"Rock": 0.5, "Steel": 0.5,
		},
		"Psychic": {
			"Fighting": 2.0, "Poison": 2.0, "Psychic": 0.5, "Dark": 0.0,
			"Steel": 0.5,
		},
		"Bug": {
			"Fire": 0.5, "Grass": 2.0, "Fighting": 0.5, "Poison": 0.5,
			"Flying": 0.5, "Psychic": 2.0, "Ghost": 0.5, "Dark": 2.0,
			"Steel": 0.5, "Fairy": 0.5,
		},
		"Rock": {
			"Fire": 2.0, "Ice": 2.0, "Fighting": 0.5, "Ground": 0.5,
			"Flying": 2.0, "Bug": 2.0, "Steel": 0.5,
		},
		"Ghost": {
			"Normal": 0.0, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5,
		},
		"Dragon": {
			"Dragon": 2.0, "Steel": 0.5, "Fairy": 0.0,
		},
		"Dark": {
			"Fighting": 0.5, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5,
			"Fairy": 0.5,
		},
		"Steel": {
			"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2.0,
			"Rock": 2.0, "Steel": 0.5, "Fairy": 2.0,
		},
		"Fairy": {
			"Fire": 0.5, "Fighting": 2.0, "Poison": 0.5, "Dragon": 2.0,
			"Dark": 2.0, "Steel": 0.5,
		},
	}
}

// CalcTypeEffectiveness returns the multiplier for an attack type vs a defending Pokemon's types.
// For dual-typed defenders, multipliers are multiplied together (e.g., 2.0 * 2.0 = 4.0).
func CalcTypeEffectiveness(attackType string, defenderTypes []string) float64 {
	multiplier := 1.0
	attackMap, exists := TypeChart[attackType]
	if !exists {
		return 1.0 // Unknown type, assume neutral
	}

	for _, defType := range defenderTypes {
		if eff, ok := attackMap[defType]; ok {
			multiplier *= eff
		}
		// If not in the map, it's 1.0 (neutral), so no change
	}
	return multiplier
}

// TypeOneHot returns an 18-element slice with 1.0 for each matching type
func TypeOneHot(types []string) []float64 {
	vec := make([]float64, 18)
	for _, t := range types {
		if idx, ok := TypeIndex[t]; ok {
			vec[idx] = 1.0
		}
	}
	return vec
}

// CalcMatchupScore returns a normalized type advantage score for attackerTypes vs defenderTypes.
// The raw effectiveness (product across defender types) is mapped to [-1, 1]:
//
//	0.0 (immune)    -> -1.0
//	0.25            -> -0.75
//	0.5 (NVE)       -> -0.5
//	1.0 (neutral)   ->  0.0
//	2.0 (SE)        ->  0.5
//	4.0 (double SE) ->  1.0
//
// Returns the best (max) score across all attacker types.
func CalcMatchupScore(attackerTypes, defenderTypes []string) float64 {
	if len(attackerTypes) == 0 || len(defenderTypes) == 0 {
		return 0.0
	}

	bestEff := 0.0
	for _, atkType := range attackerTypes {
		eff := CalcTypeEffectiveness(atkType, defenderTypes)
		if eff > bestEff {
			bestEff = eff
		}
	}

	// Normalize: log2 mapping clamped to [-1, 1]
	// 0 -> -1, 0.25 -> -0.75, 0.5 -> -0.5, 1 -> 0, 2 -> 0.5, 4 -> 1.0
	if bestEff == 0.0 {
		return -1.0
	}
	// log2(eff) / 2 maps: log2(0.5)=-1 -> -0.5, log2(1)=0 -> 0, log2(2)=1 -> 0.5, log2(4)=2 -> 1.0
	normalizedScore := math.Log2(bestEff) / 2.0
	if normalizedScore < -1.0 {
		return -1.0
	}
	if normalizedScore > 1.0 {
		return 1.0
	}
	return normalizedScore
}
