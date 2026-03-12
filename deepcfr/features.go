package deepcfr

import (
	"hash/fnv"
	"math"
	"math/bits"
	"strings"

	"github.com/pokemon-engine/gamedata"
	"github.com/pokemon-engine/simulator"
)

const (
	typeFeatureCount        = 18
	speciesHashBins         = 16
	itemHashBins            = 8
	abilityHashBins         = 8
	moveHashBins            = 8
	baseGlobalFeatureCount  = 30
	extraGlobalFeatureCount = 2
	globalFeatureCount      = baseGlobalFeatureCount + extraGlobalFeatureCount
	baseSlotFeatureCount    = 29
	extraSlotFeatureCount   = 6 + 4 + typeFeatureCount + speciesHashBins + itemHashBins + abilityHashBins
	slotFeatureCount        = baseSlotFeatureCount + extraSlotFeatureCount
	baseMoveFeatureCount    = 7
	extraMoveFeatureCount   = 3 + typeFeatureCount + moveHashBins
	moveFeatureCount        = baseMoveFeatureCount + extraMoveFeatureCount
	activeMoveSlots         = 8

	FeatureSize = globalFeatureCount + (slotFeatureCount * 12) + (moveFeatureCount * activeMoveSlots) + simulator.MaxActions
)

func encodeState(state *simulator.BattleState, legalMask []float64) []float64 {
	features := make([]float64, FeatureSize)
	idx := 0

	push := func(v float64) {
		if idx >= len(features) {
			return
		}
		features[idx] = v
		idx++
	}

	if state == nil {
		return features
	}

	push(clamp01(float64(min(state.Turn, 40)) / 40.0))
	push(float64(countAlive(&state.P1)) / 6.0)
	push(float64(countAlive(&state.P2)) / 6.0)
	push(boolToFloat(state.P1.CanTerastallize))
	push(boolToFloat(state.P2.CanTerastallize))

	for _, weather := range []string{"sunnyday", "raindance", "sandstorm", "snow", "snowscape"} {
		push(boolToFloat(normalizeID(state.Field.Weather) == weather))
	}
	for _, terrain := range []string{"electricterrain", "grassyterrain", "psychicterrain", "mistyterrain"} {
		push(boolToFloat(normalizeID(state.Field.Terrain) == terrain))
	}

	push(boolToFloat(state.Field.TrickRoom))
	push(clamp01(float64(state.Field.TrickRoomTurns) / 7.0))
	push(boolToFloat(state.Field.Gravity))
	push(clamp01(float64(state.Field.GravityTurns) / 7.0))
	push(boolToFloat(state.P1.ForceSwitch))
	push(boolToFloat(state.P2.ForceSwitch))

	encodeSide(&state.P1.Side, push)
	encodeSide(&state.P2.Side, push)

	for i := 0; i < 6; i++ {
		encodePokemonSlot(getTeamSlot(&state.P1, i), push)
	}
	for i := 0; i < 6; i++ {
		encodePokemonSlot(getTeamSlot(&state.P2, i), push)
	}

	encodeActiveMoves(state.P1.GetActive(), push)
	encodeActiveMoves(state.P2.GetActive(), push)

	for i := 0; i < simulator.MaxActions; i++ {
		if i < len(legalMask) {
			push(clamp01(legalMask[i]))
		} else {
			push(0)
		}
	}

	return features
}

func encodeState32(state *simulator.BattleState, legalMask []float32) []float32 {
	mask64 := make([]float64, len(legalMask))
	for i := range legalMask {
		mask64[i] = float64(legalMask[i])
	}
	features64 := encodeState(state, mask64)
	features32 := make([]float32, len(features64))
	for i := range features64 {
		features32[i] = float32(features64[i])
	}
	return features32
}

func buildLegalMask(state *simulator.BattleState) []float64 {
	mask := make([]float64, simulator.MaxActions)
	if state == nil {
		return mask
	}
	actions, n := simulator.GetSearchActions(&state.P1)
	for i := 0; i < n; i++ {
		a := actions[i]
		if a >= 0 && a < len(mask) {
			mask[a] = 1
		}
	}
	return mask
}

func encodeSide(side *simulator.SideConditions, push func(float64)) {
	if side == nil {
		for i := 0; i < 10; i++ {
			push(0)
		}
		return
	}
	push(boolToFloat(side.StealthRock))
	push(clamp01(float64(side.Spikes) / 3.0))
	push(clamp01(float64(side.ToxicSpikes) / 2.0))
	push(boolToFloat(side.StickyWeb))
	push(clamp01(float64(side.ReflectTurns) / 8.0))
	push(clamp01(float64(side.LightScreenTurns) / 8.0))
	push(clamp01(float64(side.AuroraVeilTurns) / 8.0))
	push(clamp01(float64(side.TailwindTurns) / 4.0))
	push(clamp01(float64(side.SafeguardTurns) / 5.0))
	push(clamp01(float64(side.MistTurns) / 5.0))
}

func encodePokemonSlot(p *simulator.PokemonState, push func(float64)) {
	if p == nil {
		for i := 0; i < slotFeatureCount; i++ {
			push(0)
		}
		return
	}

	hidden := p.Species == "" || normalizeID(p.Species) == "unknown"
	push(1)
	push(boolToFloat(hidden))
	push(boolToFloat(p.IsActive))
	push(boolToFloat(p.Fainted))

	hpFrac := 0.0
	if p.MaxHP > 0 {
		hpFrac = float64(max(p.HP, 0)) / float64(p.MaxHP)
	}
	push(clamp01(hpFrac))
	push(clamp01(float64(max(p.MaxHP, 0)) / 450.0))
	push(clamp01(float64(max(p.Level, 0)) / 100.0))
	push(boolToFloat(p.Terastallized))
	push(clamp01(float64(max(p.NumMoves, 0)) / 4.0))
	push(boolToFloat(p.Item != ""))
	push(boolToFloat(p.Ability != ""))
	push(clamp01(float64(max(p.TurnsActive, 0)) / 10.0))

	push(clamp01(float64(max(p.Stats.HP, 0)) / 450.0))
	push(clamp01(float64(max(p.Stats.Atk, 0)) / 450.0))
	push(clamp01(float64(max(p.Stats.Def, 0)) / 450.0))
	push(clamp01(float64(max(p.Stats.SpA, 0)) / 450.0))
	push(clamp01(float64(max(p.Stats.SpD, 0)) / 450.0))
	push(clamp01(float64(max(p.Stats.Spe, 0)) / 450.0))

	for _, shift := range []uint32{
		simulator.AtkShift,
		simulator.DefShift,
		simulator.SpaShift,
		simulator.SpdShift,
		simulator.SpeShift,
		simulator.EvaShift,
		simulator.AccShift,
	} {
		push((float64(p.GetBoost(shift)) + 6.0) / 12.0)
	}

	for _, status := range []string{"brn", "par", "slp", "frz", "psn", "tox"} {
		push(boolToFloat(p.Status == status))
	}

	push(clamp01(float64(max(p.SleepTurns, 0)) / 3.0))
	push(clamp01(float64(max(p.FreezeTurns, 0)) / 5.0))
	push(clamp01(float64(max(p.ToxicCounter, 0)) / 15.0))
	push(boolToFloat(p.LockedMove != ""))
	push(boolToFloat(p.TookDamageThisTurn))
	push(boolToFloat(p.ActedThisTurn))
	push(boolToFloat((p.Volatiles & simulator.VolatileSubstitute) != 0))
	push(boolToFloat((p.Volatiles & simulator.VolatileConfusion) != 0))
	push(boolToFloat((p.Volatiles & simulator.VolatileProtection) != 0))
	push(clamp01(float64(bits.OnesCount32(p.Volatiles)) / 10.0))

	currentTypes := currentTypesForPokemon(p)
	encodeTypeOneHot(currentTypes, push)
	if hidden {
		pushHashOneHot("", speciesHashBins, push)
	} else {
		pushHashOneHot(normalizeID(p.Species), speciesHashBins, push)
	}
	pushHashOneHot(normalizeID(p.Item), itemHashBins, push)
	pushHashOneHot(normalizeID(p.Ability), abilityHashBins, push)
}

func encodeActiveMoves(active *simulator.PokemonState, push func(float64)) {
	for i := 0; i < 4; i++ {
		if active == nil || i >= active.NumMoves {
			for j := 0; j < moveFeatureCount; j++ {
				push(0)
			}
			continue
		}

		moveID := active.Moves[i]
		move := gamedata.LookupMove(moveID)
		ppFrac := 0.0
		if active.MoveMaxPP[i] > 0 {
			ppFrac = float64(max(active.MovePP[i], 0)) / float64(active.MoveMaxPP[i])
		}

		basePower := 0.0
		accuracy := 1.0
		priority := 0.0
		category := ""
		if move != nil {
			basePower = clamp01(float64(max(move.BasePower, 0)) / 160.0)
			switch v := move.Accuracy.(type) {
			case int:
				accuracy = clamp01(float64(v) / 100.0)
			case float64:
				accuracy = clamp01(v / 100.0)
			case bool:
				if !v {
					accuracy = 0
				}
			}
			priority = math.Max(-2.0, math.Min(2.0, float64(move.Priority)/3.0))
			category = move.Category
		}

		push(1)
		push(clamp01(ppFrac))
		push(basePower)
		push(clamp01(accuracy))
		push((priority + 2.0) / 4.0)
		push(boolToFloat(category == "Physical"))
		push(boolToFloat(category == "Special"))
		push(boolToFloat(category == "Status"))
		push(boolToFloat(move != nil && move.Flags != nil && move.Flags["contact"] > 0))
		push(boolToFloat(move != nil && move.Flags != nil && move.Flags["protect"] > 0))
		moveType := ""
		if move != nil {
			moveType = move.Type
		}
		encodeTypeOneHot([]string{moveType}, push)
		pushHashOneHot(normalizeID(moveID), moveHashBins, push)
	}
}

func currentTypesForPokemon(p *simulator.PokemonState) []string {
	if p == nil {
		return nil
	}
	if p.Terastallized && p.TeraType != "" {
		return []string{p.TeraType}
	}
	if entry := gamedata.LookupSpecies(p.Species); entry != nil && len(entry.Types) > 0 {
		return entry.Types
	}
	if p.TeraType != "" {
		return []string{p.TeraType}
	}
	return nil
}

func encodeTypeOneHot(types []string, push func(float64)) {
	hot := [typeFeatureCount]float64{}
	for _, t := range types {
		idx := typeIndex(t)
		if idx >= 0 {
			hot[idx] = 1
		}
	}
	for i := 0; i < typeFeatureCount; i++ {
		push(hot[i])
	}
}

func typeIndex(typeName string) int {
	for i, t := range gamedata.AllTypes {
		if strings.EqualFold(t, typeName) {
			return i
		}
	}
	return -1
}

func pushHashOneHot(value string, bins int, push func(float64)) {
	if bins <= 0 {
		return
	}
	hotIdx := -1
	if value != "" {
		h := fnv.New32a()
		_, _ = h.Write([]byte(value))
		hotIdx = int(h.Sum32() % uint32(bins))
	}
	for i := 0; i < bins; i++ {
		if i == hotIdx {
			push(1)
			continue
		}
		push(0)
	}
}

func getTeamSlot(player *simulator.PlayerState, idx int) *simulator.PokemonState {
	if player == nil || idx < 0 || idx >= player.TeamSize || idx >= len(player.Team) {
		return nil
	}
	return &player.Team[idx]
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func boolToFloat(v bool) float64 {
	if v {
		return 1
	}
	return 0
}

func normalizeID(s string) string {
	return gamedata.NormalizeID(s)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
