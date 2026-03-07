package evaluator

import (
	"math"
	"sort"

	"github.com/pokemon-engine/gamedata"
	"github.com/pokemon-engine/simulator"
)

// FeaturesPerSlot is the number of features extracted per Pokemon slot
// 53 base + (6 per move * 4 moves) + 3 padding = 80
const FeaturesPerSlot = 80

// TotalSlotFeatures is 12 slots * 80 features
const TotalSlotFeatures = 12 * FeaturesPerSlot // 960

// FieldFeatures = 5 weather one-hot + 5 terrain one-hot + 1 trick room + 1 gravity = 12
const FieldFeatures = 12

// SideFeatures = 10 per side (StealthRock, Spikes, ToxicSpikes, StickyWeb, Reflect, LightScreen, AuroraVeil, Tailwind, Safeguard, Mist)
const SideFeatures = 10

// BoostFeatures = 5 per active Pokemon (atk, def, spa, spd, spe) * 2 actives = 10
const BoostFeatures = 10

// MatchupFeatures = 2 (P1 vs P2 type effectiveness, P2 vs P1 type effectiveness)
const MatchupFeatures = 2

// LatentTokenFeatures = 2 (reasoning token, prediction token)
const LatentTokenFeatures = 2

const (
	DefaultLatentReasoningToken  = 0.0
	DefaultLatentPredictionToken = 0.0
)

// TotalGlobals = 6 player globals + 12 field + 20 side (10*2) + 10 boosts (5*2) + 2 matchup + 2 latent tokens
const TotalGlobals = 6 + FieldFeatures + 2*SideFeatures + BoostFeatures + MatchupFeatures + LatentTokenFeatures // 52

// TotalFeatures is globals + slot features
const TotalFeatures = TotalGlobals + TotalSlotFeatures // 1012

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
		mainSizes := mainMLPLayerSizes()
		attnSizes := attentionMLPLayerSizes()
		GlobalMLP = NewMLP(mainSizes)
		GlobalAttentionMLP = NewMLP(attnSizes)
		GlobalAttentionMLP.LinearOutput = true

		mainOk := false
		if err := GlobalMLP.LoadWeights("evaluator_weights.json"); err == nil && GlobalMLP.HasLayerSizes(mainSizes) {
			mainOk = true
		}

		if mainOk {
			if err := GlobalAttentionMLP.LoadWeights("attention_weights.json"); err != nil || !GlobalAttentionMLP.HasLayerSizes(attnSizes) {
				GlobalAttentionMLP = NewMLP(attnSizes)
				GlobalAttentionMLP.LinearOutput = true
			}
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
	return EvaluateAllWithLatentTokens(state, mlpCache, attentionCache, DefaultLatentReasoningToken, DefaultLatentPredictionToken)
}

func EvaluateAllWithLatentTokens(state *simulator.BattleState, mlpCache *InferenceCache, attentionCache *InferenceCache, latentReasoningToken float64, latentPredictionToken float64) [simulator.MaxActions]float64 {
	InitEvaluator()

	var result [simulator.MaxActions]float64

	if GlobalMLP != nil {
		var features [TotalFeatures]float64
		VectorizeWithLatentTokens(state, GlobalAttentionMLP, &features, attentionCache, latentReasoningToken, latentPredictionToken)
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

// AttentionWeights returns the MoE-composed softmax-normalized 12-slot attention weights
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
	moeWeights, ok := attentionWeightsFromMoEOutput(rawAttention)
	if !ok {
		return weights, false
	}
	for i := 0; i < len(weights); i++ {
		weights[i] = moeWeights[i]
	}
	return weights, true
}

func Evaluate(state *simulator.BattleState, action int, mlp *MLP, attentionMLP *MLP, mlpCache *InferenceCache, attentionCache *InferenceCache, tt *TranspositionTable) float64 {
	return EvaluateWithLatentTokens(state, action, mlp, attentionMLP, mlpCache, attentionCache, tt, DefaultLatentReasoningToken, DefaultLatentPredictionToken)
}

func EvaluateWithLatentTokens(state *simulator.BattleState, action int, mlp *MLP, attentionMLP *MLP, mlpCache *InferenceCache, attentionCache *InferenceCache, tt *TranspositionTable, latentReasoningToken float64, latentPredictionToken float64) float64 {
	if mlp != nil {
		var features [TotalFeatures]float64
		VectorizeWithLatentTokens(state, attentionMLP, &features, attentionCache, latentReasoningToken, latentPredictionToken)

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
	return EvaluateBatchStatesWithLatentTokens(states, DefaultLatentReasoningToken, DefaultLatentPredictionToken)
}

func EvaluateBatchStatesWithLatentTokens(states []simulator.BattleState, latentReasoningToken float64, latentPredictionToken float64) []float64 {
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
		vectorizeLatentTokens(&tmp, &tmpIdx, latentReasoningToken, latentPredictionToken)
		copy(mainInputs[i][:TotalGlobals], tmp[:TotalGlobals])

		validActions, validLen := simulator.GetSearchActions(&state.P1)
		validActionsBatch[i] = validActions
		validLens[i] = validLen
	}

	if GlobalAttentionMLP != nil {
		rawMoEBatch := GlobalAttentionMLP.ForwardBatch(rawSlotsBatch)
		defaultWeights := uniformSlotWeights()
		for sampleIdx, rawMoE := range rawMoEBatch {
			slotWeights, ok := attentionWeightsFromMoEOutput(rawMoE)
			if !ok {
				slotWeights = defaultWeights
			}

			dst := mainInputs[sampleIdx][TotalGlobals:]
			src := rawSlotsBatch[sampleIdx]
			for slotIdx := 0; slotIdx < SlotAttentionSlots; slotIdx++ {
				w := slotWeights[slotIdx]
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
	VectorizeWithLatentTokens(state, attention, out, cache, DefaultLatentReasoningToken, DefaultLatentPredictionToken)
}

func VectorizeWithLatentTokens(state *simulator.BattleState, attention *MLP, out *[TotalFeatures]float64, cache *InferenceCache, latentReasoningToken float64, latentPredictionToken float64) {
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

	// 2 latent features
	vectorizeLatentTokens(out, &idx, latentReasoningToken, latentPredictionToken)

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

	// Apply MoE slot routing if network is provided
	if attention != nil {
		rawMoE := attention.Forward(out[slotsStartIdx:idx], cache)
		attentionWeights, ok := attentionWeightsFromMoEOutput(rawMoE)
		if !ok {
			attentionWeights = uniformSlotWeights()
		}

		// Multiply slot features by their respective Attention Weights in-place
		for slotIdx := 0; slotIdx < SlotAttentionSlots; slotIdx++ {
			weight := attentionWeights[slotIdx]
			baseIdx := slotsStartIdx + (slotIdx * FeaturesPerSlot)
			for j := 0; j < FeaturesPerSlot; j++ {
				out[baseIdx+j] *= weight
			}
		}
	}

}

func vectorizeLatentTokens(out *[TotalFeatures]float64, idx *int, latentReasoningToken float64, latentPredictionToken float64) {
	out[*idx] = latentReasoningToken
	*idx++
	out[*idx] = latentPredictionToken
	*idx++
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
	if side.ReflectTurns > 0 {
		out[*idx+4] = 1.0
	}
	if side.LightScreenTurns > 0 {
		out[*idx+5] = 1.0
	}
	if side.AuroraVeilTurns > 0 {
		out[*idx+6] = 1.0
	}
	if side.TailwindTurns > 0 {
		out[*idx+7] = 1.0
	}
	if side.SafeguardTurns > 0 {
		out[*idx+8] = 1.0
	}
	if side.MistTurns > 0 {
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
		if active.MaxHP > 0 {
			totalHP += float64(active.HP) / float64(active.MaxHP)
		}
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
		if poke.MaxHP > 0 {
			totalHP += float64(poke.HP) / float64(poke.MaxHP)
		}
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
	startOffset := *slotsIdx

	isAlive := 0.0
	if !poke.Fainted {
		isAlive = 1.0
	}

	isActive := 0.0
	if poke.IsActive {
		isActive = 1.0
	}

	hpPct := 0.0
	if poke.MaxHP > 0 {
		hpPct = float64(poke.HP) / float64(poke.MaxHP)
		if hpPct < 0 {
			hpPct = 0
		} else if hpPct > 1 {
			hpPct = 1
		}
	}

	slots[*slotsIdx] = isAlive
	*slotsIdx++
	slots[*slotsIdx] = isActive
	*slotsIdx++
	slots[*slotsIdx] = hpPct
	*slotsIdx++

	// Status (1 float)
	statusVal := 0.0
	switch poke.Status {
	case "brn":
		statusVal = 0.25
	case "par":
		statusVal = 0.5
	case "slp", "frz":
		statusVal = 0.75
	case "psn", "tox":
		statusVal = 1.0
	}
	slots[*slotsIdx] = statusVal
	*slotsIdx++

	// Level (normalized)
	lvl := 100.0
	if poke.Level > 0 {
		lvl = float64(poke.Level)
	}
	slots[*slotsIdx] = lvl / 100.0
	*slotsIdx++

	// Gender (1 float: M=1, F=0.5, N=0)
	genderVal := 0.0
	if poke.Gender == "M" {
		genderVal = 1.0
	} else if poke.Gender == "F" {
		genderVal = 0.5
	}
	slots[*slotsIdx] = genderVal
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

	// Volatiles (1 bitpacked float)
	vPacked := 0.0
	for i := 0; i < 18; i++ {
		if (poke.Volatiles & (1 << uint32(i))) != 0 {
			vPacked += math.Pow(2, float64(i))
		}
	}
	slots[*slotsIdx] = vPacked / 262144.0 // 2^18
	*slotsIdx++

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

	// MOVE-ACTION FEATURES
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
			// 1. Type ID (normalized) (1)
			typeIdx := gamedata.GetTypeIndex(move.Type)
			slots[*slotsIdx] = float64(typeIdx) / 18.0
			*slotsIdx++

			// 2. Category (1: Phys=1, Spec=0.5, Stat=0)
			cat := 0.0
			if move.Category == "Physical" {
				cat = 1.0
			} else if move.Category == "Special" {
				cat = 0.5
			}
			slots[*slotsIdx] = cat
			*slotsIdx++

			// 3. Base Power (1)
			slots[*slotsIdx] = float64(move.BasePower) / 250.0
			*slotsIdx++

			// 4. Accuracy (1)
			acc := 1.0
			switch val := move.Accuracy.(type) {
			case int:
				acc = float64(val) / 100.0
			case float64:
				acc = val / 100.0
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

			// 6. Extra Padding for alignment
			slots[*slotsIdx] = 0.0
			*slotsIdx++
		} else {
			// Zero pad empty move slots (6 features)
			for i := 0; i < 6; i++ {
				slots[*slotsIdx] = 0.0
				*slotsIdx++
			}
		}
	}

	// Final Padding to reach FeaturesPerSlot (80 total)
	expectedEnd := startOffset + FeaturesPerSlot
	for *slotsIdx < expectedEnd {
		slots[*slotsIdx] = 0.0
		*slotsIdx++
	}
}
