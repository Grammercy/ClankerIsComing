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

// typeChartMatrix provides O(1) lookup for type effectiveness
// Rows represent attackers, columns represent defenders.
var typeChartMatrix = [18][18]float64{
	// Normal
	{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.5, 1.0},
	// Fire
	{1.0, 0.5, 0.5, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0},
	// Water
	{1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0, 1.0},
	// Electric
	{1.0, 1.0, 2.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0},
	// Grass
	{1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 1.0, 0.5, 2.0, 0.5, 1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 0.5, 1.0},
	// Ice
	{1.0, 0.5, 0.5, 1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0},
	// Fighting
	{2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5, 2.0, 0.0, 1.0, 2.0, 2.0, 0.5},
	// Poison
	{1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 2.0},
	// Ground
	{1.0, 2.0, 1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0},
	// Flying
	{1.0, 1.0, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0},
	// Psychic
	{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0, 0.5, 1.0},
	// Bug
	{1.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.5, 0.5, 1.0, 0.5, 2.0, 1.0, 1.0, 0.5, 1.0, 2.0, 0.5, 0.5},
	// Rock
	{1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0},
	// Ghost
	{0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0},
	// Dragon
	{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 0.0},
	// Dark
	{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 0.5},
	// Steel
	{1.0, 0.5, 0.5, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.5, 2.0},
	// Fairy
	{1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.5, 1.0},
}

// GetTypeIndex returns the index of a type in AllTypes, or -1 if not found.
// This uses a switch statement for performance, avoiding map lookups.
func GetTypeIndex(t string) int {
	switch t {
	case "Normal":
		return 0
	case "Fire":
		return 1
	case "Water":
		return 2
	case "Electric":
		return 3
	case "Grass":
		return 4
	case "Ice":
		return 5
	case "Fighting":
		return 6
	case "Poison":
		return 7
	case "Ground":
		return 8
	case "Flying":
		return 9
	case "Psychic":
		return 10
	case "Bug":
		return 11
	case "Rock":
		return 12
	case "Ghost":
		return 13
	case "Dragon":
		return 14
	case "Dark":
		return 15
	case "Steel":
		return 16
	case "Fairy":
		return 17
	}
	return -1
}

// CalcTypeEffectiveness returns the multiplier for an attack type vs a defending Pokemon's types.
// For dual-typed defenders, multipliers are multiplied together (e.g., 2.0 * 2.0 = 4.0).
func CalcTypeEffectiveness(attackType string, defenderTypes []string) float64 {
	atkIdx := GetTypeIndex(attackType)
	if atkIdx == -1 {
		return 1.0 // Unknown type, assume neutral
	}

	multiplier := 1.0
	for _, defType := range defenderTypes {
		defIdx := GetTypeIndex(defType)
		if defIdx != -1 {
			multiplier *= typeChartMatrix[atkIdx][defIdx]
		}
	}
	return multiplier
}

// TypeOneHot returns an 18-element slice with 1.0 for each matching type
func TypeOneHot(types []string) [18]float64 {
	var vec [18]float64
	for _, t := range types {
		idx := GetTypeIndex(t)
		if idx != -1 {
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
