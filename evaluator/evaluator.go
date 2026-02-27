package evaluator

import (
	"math"
	"sort"

	"github.com/pokemon-engine/gamedata"
	"github.com/pokemon-engine/simulator"
)

// FeaturesPerSlot is the number of features extracted per Pokemon slot
// 74 base + (24 per move * 4 moves) = 170
const FeaturesPerSlot = 170

// TotalSlotFeatures is 12 slots * 170 features
const TotalSlotFeatures = 12 * FeaturesPerSlot // 2040

// FieldFeatures = 5 weather one-hot + 5 terrain one-hot + 1 trick room + 1 gravity = 12
const FieldFeatures = 12

// SideFeatures = 10 per side (StealthRock, Spikes, ToxicSpikes, StickyWeb, Reflect, LightScreen, AuroraVeil, Tailwind, Safeguard, Mist)
const SideFeatures = 10

// BoostFeatures = 5 per active Pokemon (atk, def, spa, spd, spe) * 2 actives = 10
const BoostFeatures = 10

// MatchupFeatures = 2 (P1 vs P2 type effectiveness, P2 vs P1 type effectiveness)
const MatchupFeatures = 2

// TotalGlobals = 6 player globals + 12 field + 20 side (10*2) + 10 boosts (5*2) + 2 matchup
const TotalGlobals = 6 + FieldFeatures + 2*SideFeatures + BoostFeatures + MatchupFeatures // 50

// TotalFeatures is globals + slot features
const TotalFeatures = TotalGlobals + TotalSlotFeatures // 2090

var (
	GlobalMLP          *MLP
	GlobalAttentionMLP *MLP
	MLPLoadAttempted   bool
)

// TranspositionTable caches evaluation results to avoid redundant neural network passes
type TranspositionTable struct {
	Cache map[uint64]float64
}

// NewTranspositionTable creates a preallocated TT
func NewTranspositionTable() *TranspositionTable {
	return &TranspositionTable{
		Cache: make(map[uint64]float64, 100000),
	}
}

// HashFeatures computes a fast 64-bit FNV-1a-like hash of the feature array
func HashFeatures(f *[TotalFeatures]float64) uint64 {
	var h uint64 = 14695981039346656037
	for i := 0; i < len(f); i++ {
		h ^= math.Float64bits(f[i])
		h *= 1099511628211
	}
	return h
}

func InitEvaluator() {
	if !MLPLoadAttempted {
		GlobalMLP = NewMLP([]int{TotalFeatures, 2048, 1024, 512, simulator.MaxActions})
		GlobalAttentionMLP = NewMLP([]int{TotalSlotFeatures, 256, 128, 12})
		GlobalAttentionMLP.LinearOutput = true

		if err := GlobalMLP.LoadWeights("evaluator_weights.json"); err == nil {
			GlobalAttentionMLP.LoadWeights("attention_weights.json")
		} else {
			GlobalMLP = nil
			GlobalAttentionMLP = nil
		}
		MLPLoadAttempted = true
	}
}

func GetCaches() (*InferenceCache, *InferenceCache) {
	InitEvaluator()
	return NewInferenceCache(GlobalMLP), NewInferenceCache(GlobalAttentionMLP)
}

func EvaluateAll(state *simulator.BattleState, mlpCache *InferenceCache, attentionCache *InferenceCache) [simulator.MaxActions]float64 {
	InitEvaluator()

	var result [simulator.MaxActions]float64

	if GlobalMLP != nil {
		var features [TotalFeatures]float64
		Vectorize(state, GlobalAttentionMLP, &features, attentionCache)
		output := GlobalMLP.Forward(features[:], mlpCache)

		for i := 0; i < simulator.MaxActions; i++ {
			if i < len(output) {
				result[i] = output[i]
			}
		}
		return result
	}

	// Default to neutral score if no MLP is available
	for i := 0; i < simulator.MaxActions; i++ {
		result[i] = 0.5
	}
	return result
}

// AttentionWeights returns the softmax-normalized 12-slot attention weights
// (P1 slots 0-5 followed by P2 slots 0-5) used during vectorization.
// The bool return is false when the attention network is unavailable.
func AttentionWeights(state *simulator.BattleState, cache *InferenceCache) ([12]float64, bool) {
	var weights [12]float64
	if state == nil {
		return weights, false
	}

	InitEvaluator()
	if GlobalAttentionMLP == nil {
		return weights, false
	}

	_, p1Slots := vectorizePlayerFeatures(&state.P1, state)
	_, p2Slots := vectorizePlayerFeatures(&state.P2, state)
	rawSlots := make([]float64, 0, TotalSlotFeatures)
	rawSlots = append(rawSlots, p1Slots[:]...)
	rawSlots = append(rawSlots, p2Slots[:]...)

	rawAttention := GlobalAttentionMLP.Forward(rawSlots, cache)
	if len(rawAttention) < len(weights) {
		return weights, false
	}

	maxScore := -math.MaxFloat64
	for i := 0; i < len(weights); i++ {
		if rawAttention[i] > maxScore {
			maxScore = rawAttention[i]
		}
	}

	sumExp := 0.0
	for i := 0; i < len(weights); i++ {
		weights[i] = math.Exp(rawAttention[i] - maxScore)
		sumExp += weights[i]
	}
	if sumExp <= 0 {
		return weights, false
	}
	for i := range weights {
		weights[i] /= sumExp
	}
	return weights, true
}

func Evaluate(state *simulator.BattleState, action int, mlp *MLP, attentionMLP *MLP, mlpCache *InferenceCache, attentionCache *InferenceCache, tt *TranspositionTable) float64 {
	if mlp != nil {
		var features [TotalFeatures]float64
		Vectorize(state, attentionMLP, &features, attentionCache)

		if tt != nil {
			hash := HashFeatures(&features)
			if q, exists := tt.Cache[hash]; exists {
				// Action -1 (state evaluation) can be served from TT
				if action == -1 {
					return q
				}
			}
		}

		output := mlp.Forward(features[:], mlpCache)
		if action >= 0 && action < len(output) {
			return output[action]
		}

		// If action is -1 (just general state evaluation), return the max Q-value over VALID actions
		validActions, validLen := simulator.GetSearchActions(&state.P1)
		if validLen == 0 {
			return 0.5 // Default tie/unknown
		}

		maxQ := -math.MaxFloat64
		for i := 0; i < validLen; i++ {
			a := validActions[i]
			if a >= 0 && a < len(output) {
				if output[a] > maxQ {
					maxQ = output[a]
				}
			}
		}

		if maxQ == -math.MaxFloat64 {
			// Fallback if somehow no valid actions mapped to 0-9
			for _, q := range output {
				if q > maxQ {
					maxQ = q
				}
			}
		}

		if tt != nil {
			hash := HashFeatures(&features)
			tt.Cache[hash] = maxQ
		}
		return maxQ
	}

	// 2. Default to neutral score
	return 0.5
}

// EvaluateBatchStates evaluates multiple battle states in batches on GPU and returns
// state values from P1 perspective (max Q over valid P1 actions for each state).
func EvaluateBatchStates(states []simulator.BattleState) []float64 {
	InitEvaluator()
	results := make([]float64, len(states))
	if len(states) == 0 {
		return results
	}
	if GlobalMLP == nil {
		for i := range results {
			results[i] = 0.5
		}
		return results
	}

	mainInputs := make([][]float64, len(states))
	rawSlotsBatch := make([][]float64, len(states))
	validActionsBatch := make([][simulator.MaxActions]int, len(states))
	validLens := make([]int, len(states))

	for i := range states {
		state := &states[i]
		p1Globals, p1Slots := vectorizePlayerFeatures(&state.P1, state)
		p2Globals, p2Slots := vectorizePlayerFeatures(&state.P2, state)

		mainInputs[i] = make([]float64, TotalFeatures)
		rawSlots := make([]float64, 0, TotalSlotFeatures)
		rawSlots = append(rawSlots, p1Slots[:]...)
		rawSlots = append(rawSlots, p2Slots[:]...)
		rawSlotsBatch[i] = rawSlots

		idx := 0
		mainInputs[i][idx] = p1Globals[0]
		idx++
		mainInputs[i][idx] = p1Globals[1]
		idx++
		mainInputs[i][idx] = p2Globals[0]
		idx++
		mainInputs[i][idx] = p2Globals[1]
		idx++
		mainInputs[i][idx] = p1Globals[2]
		idx++
		mainInputs[i][idx] = p2Globals[2]
		idx++
		var tmp [TotalFeatures]float64
		copy(tmp[:TotalGlobals], mainInputs[i][:TotalGlobals])
		tmpIdx := idx
		vectorizeFieldConditions(&state.Field, &tmp, &tmpIdx)
		vectorizeSideConditions(&state.P1.Side, &tmp, &tmpIdx)
		vectorizeSideConditions(&state.P2.Side, &tmp, &tmpIdx)
		vectorizeBoosts(state.P1.GetActive(), &tmp, &tmpIdx)
		vectorizeBoosts(state.P2.GetActive(), &tmp, &tmpIdx)
		vectorizeMatchup(state, &tmp, &tmpIdx)
		copy(mainInputs[i][:TotalGlobals], tmp[:TotalGlobals])

		validActions, validLen := simulator.GetSearchActions(&state.P1)
		validActionsBatch[i] = validActions
		validLens[i] = validLen
	}

	if GlobalAttentionMLP != nil {
		rawAttentionBatch := GlobalAttentionMLP.ForwardBatch(rawSlotsBatch)
		for sampleIdx, rawAttention := range rawAttentionBatch {
			weights := make([]float64, len(rawAttention))
			maxScore := -math.MaxFloat64
			for _, score := range rawAttention {
				if score > maxScore {
					maxScore = score
				}
			}
			sumExp := 0.0
			for i, score := range rawAttention {
				weights[i] = math.Exp(score - maxScore)
				sumExp += weights[i]
			}
			if sumExp > 0 {
				for i := range weights {
					weights[i] /= sumExp
				}
			}

			dst := mainInputs[sampleIdx][TotalGlobals:]
			src := rawSlotsBatch[sampleIdx]
			for slotIdx := 0; slotIdx < len(weights); slotIdx++ {
				w := weights[slotIdx]
				base := slotIdx * FeaturesPerSlot
				for j := 0; j < FeaturesPerSlot; j++ {
					dst[base+j] = src[base+j] * w
				}
			}
		}
	} else {
		for i := range states {
			copy(mainInputs[i][TotalGlobals:], rawSlotsBatch[i])
		}
	}

	outputs := GlobalMLP.ForwardBatch(mainInputs)
	for i := range outputs {
		validLen := validLens[i]
		if validLen == 0 {
			results[i] = 0.5
			continue
		}

		maxQ := -math.MaxFloat64
		for j := 0; j < validLen; j++ {
			action := validActionsBatch[i][j]
			if action >= 0 && action < len(outputs[i]) && outputs[i][action] > maxQ {
				maxQ = outputs[i][action]
			}
		}
		if maxQ == -math.MaxFloat64 {
			results[i] = 0.5
		} else {
			results[i] = maxQ
		}
	}

	return results
}

// Vectorize converts a BattleState into a fixed-length float64 array for the Neural Network.
// It writes directly into the given 'out' buffer to completely eliminate slice allocations during search.
func Vectorize(state *simulator.BattleState, attention *MLP, out *[TotalFeatures]float64, cache *InferenceCache) {
	idx := 0

	// 6 player globals
	p1Globals, p1Slots := vectorizePlayerFeatures(&state.P1, state)
	p2Globals, p2Slots := vectorizePlayerFeatures(&state.P2, state)

	out[idx] = p1Globals[0]
	idx++
	out[idx] = p1Globals[1]
	idx++
	out[idx] = p2Globals[0]
	idx++
	out[idx] = p2Globals[1]
	idx++
	out[idx] = p1Globals[2]
	idx++
	out[idx] = p2Globals[2]
	idx++

	// 11 field condition features
	vectorizeFieldConditions(&state.Field, out, &idx)

	// 14 side condition features (7 per side)
	vectorizeSideConditions(&state.P1.Side, out, &idx)
	vectorizeSideConditions(&state.P2.Side, out, &idx)

	// 10 active boost features (5 per active)
	vectorizeBoosts(state.P1.GetActive(), out, &idx)
	vectorizeBoosts(state.P2.GetActive(), out, &idx)

	// 2 type matchup features
	vectorizeMatchup(state, out, &idx)

	// Track where slots begin for attention processing
	slotsStartIdx := idx

	// Combine all 12 slots (2040 features) into out buffer
	for i := 0; i < len(p1Slots); i++ {
		out[idx] = p1Slots[i]
		idx++
	}
	for i := 0; i < len(p2Slots); i++ {
		out[idx] = p2Slots[i]
		idx++
	}

	// Apply Self-Attention if network is provided
	if attention != nil {
		rawAttention := attention.Forward(out[slotsStartIdx:idx], cache)

		// Softmax the 12 scores
		attentionWeights := make([]float64, 12)
		maxScore := -math.MaxFloat64
		for _, score := range rawAttention {
			if score > maxScore {
				maxScore = score
			}
		}

		sumExp := 0.0
		for i, score := range rawAttention {
			attentionWeights[i] = math.Exp(score - maxScore)
			sumExp += attentionWeights[i]
		}
		for i := range attentionWeights {
			attentionWeights[i] /= sumExp
		}

		// Multiply slot features by their respective Attention Weights in-place
		for slotIdx := 0; slotIdx < 12; slotIdx++ {
			weight := attentionWeights[slotIdx]
			baseIdx := slotsStartIdx + (slotIdx * FeaturesPerSlot)
			for j := 0; j < FeaturesPerSlot; j++ {
				out[baseIdx+j] *= weight
			}
		}
	}

}

// vectorizeFieldConditions writes 11 features for global field state into the out array
func vectorizeFieldConditions(field *simulator.FieldConditions, out *[TotalFeatures]float64, idx *int) {
	// Weather one-hot (5 bits): Sun, Rain, Sand, Snow, None
	wIdx := 4 // Default to "None"
	w := field.Weather
	if w == "SunnyDay" || w == "DesolateLand" || w == "Sun" {
		wIdx = 0
	} else if w == "RainDance" || w == "PrimordialSea" || w == "Rain" {
		wIdx = 1
	} else if w == "Sandstorm" || w == "Sand" {
		wIdx = 2
	} else if w == "Snowscape" || w == "Hail" || w == "Snow" {
		wIdx = 3
	}
	out[*idx+wIdx] = 1.0
	*idx += 5

	// Terrain one-hot (5 bits): Electric, Grassy, Psychic, Misty, None
	tIdx := 4 // Default to "None"
	t := field.Terrain
	if t == "Electric Terrain" || t == "Electric" {
		tIdx = 0
	} else if t == "Grassy Terrain" || t == "Grassy" {
		tIdx = 1
	} else if t == "Psychic Terrain" || t == "Psychic" {
		tIdx = 2
	} else if t == "Misty Terrain" || t == "Misty" {
		tIdx = 3
	}
	out[*idx+tIdx] = 1.0
	*idx += 5

	// Trick Room
	if field.TrickRoom {
		out[*idx] = 1.0
	}
	*idx++

	// Gravity
	if field.Gravity {
		out[*idx] = 1.0
	}
	*idx++
}

// vectorizeSideConditions writes 10 features for one side's conditions
func vectorizeSideConditions(side *simulator.SideConditions, out *[TotalFeatures]float64, idx *int) {
	if side.StealthRock {
		out[*idx] = 1.0
	}
	out[*idx+1] = float64(side.Spikes) / 3.0
	out[*idx+2] = float64(side.ToxicSpikes) / 2.0
	if side.StickyWeb {
		out[*idx+3] = 1.0
	}
	if side.Reflect {
		out[*idx+4] = 1.0
	}
	if side.LightScreen {
		out[*idx+5] = 1.0
	}
	if side.AuroraVeil {
		out[*idx+6] = 1.0
	}
	if side.Tailwind {
		out[*idx+7] = 1.0
	}
	if side.Safeguard {
		out[*idx+8] = 1.0
	}
	if side.Mist {
		out[*idx+9] = 1.0
	}
	*idx += 10
}

// vectorizeBoosts writes 5 features for the active Pokemon's stat boosts
func vectorizeBoosts(active *simulator.PokemonState, out *[TotalFeatures]float64, idx *int) {
	if active == nil {
		*idx += 5
		return
	}

	out[*idx] = float64(active.GetBoost(simulator.AtkShift)) / 6.0
	out[*idx+1] = float64(active.GetBoost(simulator.DefShift)) / 6.0
	out[*idx+2] = float64(active.GetBoost(simulator.SpaShift)) / 6.0
	out[*idx+3] = float64(active.GetBoost(simulator.SpdShift)) / 6.0
	out[*idx+4] = float64(active.GetBoost(simulator.SpeShift)) / 6.0
	*idx += 5
}

func getCurrentTypes(p *simulator.PokemonState) []string {
	if p == nil {
		return nil
	}
	if p.Terastallized && p.TeraType != "" {
		return []string{p.TeraType}
	}
	entry := gamedata.LookupSpecies(p.Species)
	if entry == nil {
		return nil
	}
	return entry.Types
}

// vectorizeMatchup writes 2 type-effectiveness features for P1 active vs P2 active
func vectorizeMatchup(state *simulator.BattleState, out *[TotalFeatures]float64, idx *int) {
	p1Active := state.P1.GetActive()
	p2Active := state.P2.GetActive()

	if p1Active != nil && p2Active != nil {
		p1Types := getCurrentTypes(p1Active)
		p2Types := getCurrentTypes(p2Active)
		if len(p1Types) > 0 && len(p2Types) > 0 {
			out[*idx] = gamedata.CalcMatchupScore(p1Types, p2Types)   // P1 attacks P2
			out[*idx+1] = gamedata.CalcMatchupScore(p2Types, p1Types) // P2 attacks P1
		}
	}
	*idx += 2
}

// vectorizePlayerFeatures extracts 3 globals and 1020 slot features (6 slots × 170 features)
func vectorizePlayerFeatures(player *simulator.PlayerState, state *simulator.BattleState) ([3]float64, [6 * FeaturesPerSlot]float64) {
	var globals [3]float64
	var slots [6 * FeaturesPerSlot]float64
	slotsIdx := 0
	aliveCount := 0.0
	totalHP := 0.0

	var team []*simulator.PokemonState
	for i := 0; i < player.TeamSize; i++ {
		team = append(team, &player.Team[i])
	}

	slotsAdded := 0

	// 1. ACTIVE POKEMON ALWAYS IN SLOT 0
	active := player.GetActive()
	if active != nil {
		extractPokemon(active, slots[:], &slotsIdx, state)
		slotsAdded++
		if !active.Fainted {
			aliveCount += 1.0
		}
		totalHP += float64(active.HP) / 100.0
	}

	// 2. SORT REMAINING TEAM BY NAME FOR CONSISTENCY
	var remaining []*simulator.PokemonState
	for _, poke := range team {
		if poke.IsActive {
			continue
		}
		remaining = append(remaining, poke)
	}
	sort.Slice(remaining, func(i, j int) bool {
		return remaining[i].Species < remaining[j].Species
	})

	for _, poke := range remaining {
		if slotsAdded >= 6 {
			break
		}
		extractPokemon(poke, slots[:], &slotsIdx, state)
		if !poke.Fainted {
			aliveCount += 1.0
		}
		totalHP += float64(poke.HP) / 100.0
		slotsAdded++
	}

	// Pad remaining slots
	for slotsAdded < 6 {
		extractPokemon(&simulator.PokemonState{Species: "None", Fainted: true}, slots[:], &slotsIdx, state)
		slotsAdded++
	}

	globals[0] = aliveCount / 6.0
	globals[1] = totalHP / 6.0
	if player.CanTerastallize {
		globals[2] = 1.0
	}

	return globals, slots
}

// extractPokemon is a helper to vectorize a single Pokemon into the slots array.
// Now includes Move-Action features for all 4 move slots.
func extractPokemon(poke *simulator.PokemonState, slots []float64, slotsIdx *int, state *simulator.BattleState) {
	isAlive := 0.0
	if !poke.Fainted {
		isAlive = 1.0
	}

	isActive := 0.0
	if poke.IsActive {
		isActive = 1.0
	}

	hpPct := float64(poke.HP) / 100.0

	brn, par, slp, psn := 0.0, 0.0, 0.0, 0.0
	switch poke.Status {
	case "brn":
		brn = 1.0
	case "par":
		par = 1.0
	case "slp", "frz":
		slp = 1.0
	case "psn", "tox":
		psn = 1.0
	}

	slots[*slotsIdx] = isAlive
	*slotsIdx++
	slots[*slotsIdx] = isActive
	*slotsIdx++
	slots[*slotsIdx] = hpPct
	*slotsIdx++
	slots[*slotsIdx] = brn
	*slotsIdx++
	slots[*slotsIdx] = par
	*slotsIdx++
	slots[*slotsIdx] = slp
	*slotsIdx++
	slots[*slotsIdx] = psn
	*slotsIdx++

	// Level (normalized)
	lvl := 100.0
	if poke.Level > 0 {
		lvl = float64(poke.Level)
	}
	slots[*slotsIdx] = lvl / 100.0
	*slotsIdx++

	// Gender (2 bits: IsMale, IsFemale)
	isMale, isFemale := 0.0, 0.0
	if poke.Gender == "M" {
		isMale = 1.0
	} else if poke.Gender == "F" {
		isFemale = 1.0
	}
	slots[*slotsIdx] = isMale
	*slotsIdx++
	slots[*slotsIdx] = isFemale
	*slotsIdx++

	slots[*slotsIdx] = float64(poke.NumMoves) / 4.0
	*slotsIdx++

	// Ability Flags (10)
	abilityFlags := map[string]int{
		"hugepower": 0, "purepower": 0, "rivalry": 1, "regenerator": 2, "intimidate": 3,
		"levitate": 4, "drought": 5, "drizzle": 6, "sandstream": 7, "snowwarning": 8, "sturdy": 9,
	}
	aVec := make([]float64, 10)
	if idx, ok := abilityFlags[poke.Ability]; ok {
		aVec[idx] = 1.0
	}
	for i := 0; i < 10; i++ {
		slots[*slotsIdx] = aVec[i]
		*slotsIdx++
	}

	// Item Flags (10)
	itemFlags := map[string]int{
		"choiceband": 0, "choicespecs": 1, "choicescarf": 2, "lifeorb": 3, "leftovers": 4,
		"blacksludge": 4, "focussash": 5, "eviolite": 6, "assaultvest": 7, "heavydutyboots": 8, "rockyhelmet": 9,
	}
	iVec := make([]float64, 10)
	if idx, ok := itemFlags[poke.Item]; ok {
		iVec[idx] = 1.0
	}
	for i := 0; i < 10; i++ {
		slots[*slotsIdx] = iVec[i]
		*slotsIdx++
	}

	// Volatiles (18 bits)
	for i := 0; i < 18; i++ {
		if (poke.Volatiles & (1 << uint32(i))) != 0 {
			slots[*slotsIdx] = 1.0
		} else {
			slots[*slotsIdx] = 0.0
		}
		*slotsIdx++
	}

	entry := gamedata.LookupSpecies(poke.Species)
	if entry != nil {
		typeVec := gamedata.TypeOneHot(getCurrentTypes(poke))
		for i := 0; i < len(typeVec); i++ {
			slots[*slotsIdx] = typeVec[i]
			*slotsIdx++
		}

		// Use Actual Computed Stats if available, otherwise Base Stats
		if poke.Stats.HP > 0 {
			slots[*slotsIdx] = float64(poke.Stats.HP) / 500.0
			*slotsIdx++
			slots[*slotsIdx] = float64(poke.Stats.Atk) / 500.0
			*slotsIdx++
			slots[*slotsIdx] = float64(poke.Stats.Def) / 500.0
			*slotsIdx++
			slots[*slotsIdx] = float64(poke.Stats.SpA) / 500.0
			*slotsIdx++
			slots[*slotsIdx] = float64(poke.Stats.SpD) / 500.0
			*slotsIdx++
			slots[*slotsIdx] = float64(poke.Stats.Spe) / 500.0
			*slotsIdx++
		} else {
			slots[*slotsIdx] = float64(entry.BaseStats.HP) / 255.0
			*slotsIdx++
			slots[*slotsIdx] = float64(entry.BaseStats.Atk) / 255.0
			*slotsIdx++
			slots[*slotsIdx] = float64(entry.BaseStats.Def) / 255.0
			*slotsIdx++
			slots[*slotsIdx] = float64(entry.BaseStats.SpA) / 255.0
			*slotsIdx++
			slots[*slotsIdx] = float64(entry.BaseStats.SpD) / 255.0
			*slotsIdx++
			slots[*slotsIdx] = float64(entry.BaseStats.Spe) / 255.0
			*slotsIdx++
		}

		// Weight (normalized)
		slots[*slotsIdx] = entry.Weight / 1000.0
		*slotsIdx++
	} else {
		for i := 0; i < 25; i++ { // 18 types + 6 stats + 1 weight
			slots[*slotsIdx] = 0.0
			*slotsIdx++
		}
	}

	// MOVE-ACTION FEATURES (24 per move * 4 moves = 96)
	// We determine opponent active to calculate effectiveness
	var opponentActive *simulator.PokemonState
	if state != nil {
		if poke.IsActive {
			// Check if poke belongs to P1 or P2 using ActiveIdx index
			if state.P1.ActiveIdx != -1 && &state.P1.Team[state.P1.ActiveIdx] == poke {
				opponentActive = state.P2.GetActive()
			} else if state.P2.ActiveIdx != -1 && &state.P2.Team[state.P2.ActiveIdx] == poke {
				opponentActive = state.P1.GetActive()
			}
		}
	}

	for m := 0; m < 4; m++ {
		moveName := poke.Moves[m]
		move := gamedata.LookupMove(moveName)
		if move != nil {
			// 1. Type One-Hot (18)
			typeVec := gamedata.TypeOneHot([]string{move.Type})
			for i := 0; i < 18; i++ {
				slots[*slotsIdx] = typeVec[i]
				*slotsIdx++
			}
			// 2. Category (3: Phys, Spec, Stat)
			p, s, st := 0.0, 0.0, 0.0
			if move.Category == "Physical" {
				p = 1.0
			} else if move.Category == "Special" {
				s = 1.0
			} else {
				st = 1.0
			}
			slots[*slotsIdx] = p
			*slotsIdx++
			slots[*slotsIdx] = s
			*slotsIdx++
			slots[*slotsIdx] = st
			*slotsIdx++

			// 3. Base Power (1)
			slots[*slotsIdx] = float64(move.BasePower) / 250.0
			*slotsIdx++

			// 4. Accuracy (1)
			acc := 1.0
			if val, ok := move.Accuracy.(int); ok {
				acc = float64(val) / 100.0
			}
			slots[*slotsIdx] = acc
			*slotsIdx++

			// 5. Effectiveness (1)
			eff := 1.0
			if opponentActive != nil {
				oppTypes := getCurrentTypes(opponentActive)
				if len(oppTypes) > 0 {
					eff = gamedata.CalcTypeEffectiveness(move.Type, oppTypes) / 4.0
				}
			}
			slots[*slotsIdx] = eff
			*slotsIdx++
		} else {
			// Zero pad empty move slots
			for i := 0; i < 24; i++ {
				slots[*slotsIdx] = 0.0
				*slotsIdx++
			}
		}
	}
}
