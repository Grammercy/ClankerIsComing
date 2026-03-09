package evaluator

import (
	"math"
	"sort"

	"github.com/pokemon-engine/gamedata"
	"github.com/pokemon-engine/simulator"
)

const (
	AbilityHashBuckets = 16
	ItemHashBuckets    = 16
	VolatileBits       = 18
	MoveFeatures       = 8
	PlayerGlobals      = 7
)

// FeaturesPerSlot is the number of features extracted per Pokemon slot.
// 7 base + ability(16) + item(16) + volatiles(18) + species(25) + (8 per move * 4 moves) = 114
const FeaturesPerSlot = 114

// TotalSlotFeatures is 12 slots * FeaturesPerSlot features
const TotalSlotFeatures = 12 * FeaturesPerSlot // 1368

// FieldFeatures = 5 weather one-hot + weather turns + 5 terrain one-hot + terrain turns + trick room turns + gravity turns = 16
const FieldFeatures = 16

// SideFeatures = 10 per side (StealthRock, Spikes, ToxicSpikes, StickyWeb, Reflect, LightScreen, AuroraVeil, Tailwind, Safeguard, Mist)
const SideFeatures = 10

// BoostFeatures = 5 per active Pokemon (atk, def, spa, spd, spe) * 2 actives = 10
const BoostFeatures = 10

// MatchupFeatures = 2 (P1 vs P2 type effectiveness, P2 vs P1 type effectiveness)
const MatchupFeatures = 2

// SpeedOrderFeatures = 4 (P1 speed, P2 speed, speed diff, P1 moves first at equal priority)
const SpeedOrderFeatures = 4

// ActiveProgressFeatures = 8 (sleep/freeze/toxic/turns-active for both actives)
const ActiveProgressFeatures = 8

// LatentTokenFeatures = 2 (reasoning token, prediction token)
const LatentTokenFeatures = 2

const (
	DefaultLatentReasoningToken  = 0.0
	DefaultLatentPredictionToken = 0.0
)

const (
	// Main evaluator outputs action logits plus next-step latent token predictions.
	MainActionOutputSize = simulator.MaxActions
	MainOutputSize       = MainActionOutputSize + LatentTokenFeatures
)

const (
	ReasoningTokenOutputIndex  = MainActionOutputSize
	PredictionTokenOutputIndex = MainActionOutputSize + 1
	LatentRecurrenceSteps      = 5
)

// TotalGlobals = 14 player globals + field + 20 side + 10 boosts + 2 matchup + 4 speed/order + 8 status progress + 2 latent
const TotalGlobals = 2*PlayerGlobals + FieldFeatures + 2*SideFeatures + BoostFeatures + MatchupFeatures + SpeedOrderFeatures + ActiveProgressFeatures + LatentTokenFeatures // 76

// TotalFeatures is globals + slot features
const TotalFeatures = TotalGlobals + TotalSlotFeatures // 1444

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
		output := make([]float64, 0, MainOutputSize)
		reasoningToken := latentReasoningToken
		predictionToken := latentPredictionToken
		for step := 0; step < LatentRecurrenceSteps; step++ {
			VectorizeWithLatentTokens(state, GlobalAttentionMLP, &features, attentionCache, reasoningToken, predictionToken)
			output = GlobalMLP.Forward(features[:], mlpCache)
			if len(output) > PredictionTokenOutputIndex {
				reasoningToken = output[ReasoningTokenOutputIndex]
				predictionToken = output[PredictionTokenOutputIndex]
			}
		}

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

// AttentionWeights returns softmax-normalized 12-slot attention weights
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
	attnWeights, ok := attentionWeightsFromOutput(rawAttention)
	if !ok {
		return weights, false
	}
	for i := 0; i < len(weights); i++ {
		weights[i] = attnWeights[i]
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

		output := make([]float64, 0, MainOutputSize)
		reasoningToken := latentReasoningToken
		predictionToken := latentPredictionToken
		for step := 0; step < LatentRecurrenceSteps; step++ {
			VectorizeWithLatentTokens(state, attentionMLP, &features, attentionCache, reasoningToken, predictionToken)
			output = mlp.Forward(features[:], mlpCache)
			if len(output) > PredictionTokenOutputIndex {
				reasoningToken = output[ReasoningTokenOutputIndex]
				predictionToken = output[PredictionTokenOutputIndex]
			}
		}
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

	for i := range states {
		results[i] = EvaluateWithLatentTokens(&states[i], -1, GlobalMLP, GlobalAttentionMLP, nil, nil, nil, latentReasoningToken, latentPredictionToken)
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

	// 14 player globals (7 per side)
	p1Globals, p1Slots := vectorizePlayerFeatures(&state.P1, state)
	p2Globals, p2Slots := vectorizePlayerFeatures(&state.P2, state)
	for i := 0; i < len(p1Globals); i++ {
		out[idx] = p1Globals[i]
		idx++
	}
	for i := 0; i < len(p2Globals); i++ {
		out[idx] = p2Globals[i]
		idx++
	}

	// Field condition features
	vectorizeFieldConditions(&state.Field, out, &idx)

	// Side condition features
	vectorizeSideConditions(&state.P1.Side, out, &idx)
	vectorizeSideConditions(&state.P2.Side, out, &idx)

	// 10 active boost features (5 per active)
	vectorizeBoosts(state.P1.GetActive(), out, &idx)
	vectorizeBoosts(state.P2.GetActive(), out, &idx)

	// 2 type matchup features
	vectorizeMatchup(state, out, &idx)

	// 4 speed/order features
	vectorizeSpeedOrderFeatures(state, out, &idx)

	// 8 active status progression features
	vectorizeActiveProgressFeatures(state, out, &idx)

	// 2 latent features
	vectorizeLatentTokens(out, &idx, latentReasoningToken, latentPredictionToken)

	// Track where slots begin for attention processing
	slotsStartIdx := idx

	// Combine all 12 slots into out buffer
	for i := 0; i < len(p1Slots); i++ {
		out[idx] = p1Slots[i]
		idx++
	}
	for i := 0; i < len(p2Slots); i++ {
		out[idx] = p2Slots[i]
		idx++
	}

	// Apply learned slot attention if network is provided
	if attention != nil {
		rawAttention := attention.Forward(out[slotsStartIdx:idx], cache)
		attentionWeights, ok := attentionWeightsFromOutput(rawAttention)
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

// vectorizeFieldConditions writes field identity and exact timer features.
func vectorizeFieldConditions(field *simulator.FieldConditions, out *[TotalFeatures]float64, idx *int) {
	for i := 0; i < 16; i++ {
		out[*idx+i] = 0.0
	}

	// Weather one-hot (5 bits): Sun, Rain, Sand, Snow, None
	wIdx := 4 // Default to "None"
	w := normalizeWeatherForVectorize(field.Weather)
	if w == "sunnyday" {
		wIdx = 0
	} else if w == "raindance" {
		wIdx = 1
	} else if w == "sandstorm" {
		wIdx = 2
	} else if w == "snowscape" {
		wIdx = 3
	}
	out[*idx+wIdx] = 1.0
	*idx += 5
	out[*idx] = math.Min(float64(field.WeatherTurns)/8.0, 1.0)
	*idx++

	// Terrain one-hot (5 bits): Electric, Grassy, Psychic, Misty, None
	tIdx := 4 // Default to "None"
	t := normalizeTerrainForVectorize(field.Terrain)
	if t == "electricterrain" {
		tIdx = 0
	} else if t == "grassyterrain" {
		tIdx = 1
	} else if t == "psychicterrain" {
		tIdx = 2
	} else if t == "mistyterrain" {
		tIdx = 3
	}
	out[*idx+tIdx] = 1.0
	*idx += 5
	out[*idx] = math.Min(float64(field.TerrainTurns)/8.0, 1.0)
	*idx++
	out[*idx] = math.Min(float64(field.TrickRoomTurns)/5.0, 1.0)
	*idx++
	out[*idx] = math.Min(float64(field.GravityTurns)/5.0, 1.0)
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
	out[*idx+4] = math.Min(float64(side.ReflectTurns)/8.0, 1.0)
	out[*idx+5] = math.Min(float64(side.LightScreenTurns)/8.0, 1.0)
	out[*idx+6] = math.Min(float64(side.AuroraVeilTurns)/8.0, 1.0)
	out[*idx+7] = math.Min(float64(side.TailwindTurns)/4.0, 1.0)
	out[*idx+8] = math.Min(float64(side.SafeguardTurns)/5.0, 1.0)
	out[*idx+9] = math.Min(float64(side.MistTurns)/5.0, 1.0)
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

func vectorizeSpeedOrderFeatures(state *simulator.BattleState, out *[TotalFeatures]float64, idx *int) {
	p1Active := state.P1.GetActive()
	p2Active := state.P2.GetActive()
	if p1Active == nil || p2Active == nil {
		*idx += SpeedOrderFeatures
		return
	}

	p1Speed := effectiveSpeedForOrder(state, p1Active, &state.P1.Side)
	p2Speed := effectiveSpeedForOrder(state, p2Active, &state.P2.Side)
	out[*idx] = math.Min(p1Speed/1200.0, 1.0)
	out[*idx+1] = math.Min(p2Speed/1200.0, 1.0)
	diff := (p1Speed - p2Speed) / 1200.0
	if diff < -1.0 {
		diff = -1.0
	}
	if diff > 1.0 {
		diff = 1.0
	}
	out[*idx+2] = (diff + 1.0) / 2.0

	p1First := 0.0
	if state.Field.TrickRoom {
		if p1Speed < p2Speed {
			p1First = 1.0
		}
	} else if p1Speed > p2Speed {
		p1First = 1.0
	}
	out[*idx+3] = p1First

	*idx += SpeedOrderFeatures
}

func vectorizeActiveProgressFeatures(state *simulator.BattleState, out *[TotalFeatures]float64, idx *int) {
	p1Active := state.P1.GetActive()
	p2Active := state.P2.GetActive()
	if p1Active != nil {
		out[*idx] = math.Min(float64(p1Active.SleepTurns)/3.0, 1.0)
		out[*idx+2] = math.Min(float64(p1Active.FreezeTurns)/5.0, 1.0)
		out[*idx+4] = math.Min(float64(p1Active.ToxicCounter)/15.0, 1.0)
		out[*idx+6] = math.Min(float64(p1Active.TurnsActive)/20.0, 1.0)
	}
	if p2Active != nil {
		out[*idx+1] = math.Min(float64(p2Active.SleepTurns)/3.0, 1.0)
		out[*idx+3] = math.Min(float64(p2Active.FreezeTurns)/5.0, 1.0)
		out[*idx+5] = math.Min(float64(p2Active.ToxicCounter)/15.0, 1.0)
		out[*idx+7] = math.Min(float64(p2Active.TurnsActive)/20.0, 1.0)
	}
	*idx += ActiveProgressFeatures
}

func normalizeWeatherForVectorize(weather string) string {
	switch gamedata.NormalizeID(weather) {
	case "raindance", "primordialsea", "rain":
		return "raindance"
	case "sunnyday", "desolateland", "sun":
		return "sunnyday"
	case "sandstorm", "sand":
		return "sandstorm"
	case "hail", "snowscape", "snow":
		return "snowscape"
	default:
		return ""
	}
}

func normalizeTerrainForVectorize(terrain string) string {
	switch gamedata.NormalizeID(terrain) {
	case "electricterrain", "electric":
		return "electricterrain"
	case "grassyterrain", "grassy":
		return "grassyterrain"
	case "psychicterrain", "psychic":
		return "psychicterrain"
	case "mistyterrain", "misty":
		return "mistyterrain"
	default:
		return ""
	}
}

func boostMultiplier(stages int) float64 {
	if stages >= 0 {
		return float64(2+stages) / 2.0
	}
	return 2.0 / float64(2-stages)
}

func effectiveSpeedForOrder(state *simulator.BattleState, poke *simulator.PokemonState, side *simulator.SideConditions) float64 {
	base := float64(poke.Stats.Spe)
	if base <= 0 {
		if entry := gamedata.LookupSpecies(poke.Species); entry != nil {
			base = float64(entry.BaseStats.Spe*2 + 36)
		} else {
			base = 200.0
		}
	}
	speed := base * boostMultiplier(poke.GetBoost(simulator.SpeShift))

	if side != nil && side.TailwindTurns > 0 {
		speed *= 2.0
	}

	item := gamedata.NormalizeID(poke.Item)
	if item == "choicescarf" {
		speed *= 1.5
	}

	ability := gamedata.NormalizeID(poke.Ability)
	weather := ""
	if state != nil {
		weather = normalizeWeatherForVectorize(state.Field.Weather)
	}
	switch ability {
	case "swiftswim":
		if weather == "raindance" {
			speed *= 2.0
		}
	case "chlorophyll":
		if weather == "sunnyday" {
			speed *= 2.0
		}
	case "sandrush":
		if weather == "sandstorm" {
			speed *= 2.0
		}
	case "slushrush":
		if weather == "snowscape" {
			speed *= 2.0
		}
	}

	if poke.Status == "par" && ability != "quickfeet" {
		speed *= 0.5
	}
	if ability == "quickfeet" && poke.Status != "" {
		speed *= 1.5
	}
	return speed
}

func writeHashedOneHot(out []float64, offset int, buckets int, raw string) {
	for i := 0; i < buckets; i++ {
		out[offset+i] = 0.0
	}
	key := gamedata.NormalizeID(raw)
	if key == "" || buckets <= 0 {
		return
	}
	var h uint32 = 2166136261
	for i := 0; i < len(key); i++ {
		h ^= uint32(key[i])
		h *= 16777619
	}
	out[offset+int(h%uint32(buckets))] = 1.0
}

func secondaryEffectScore(move *gamedata.MoveEntry) float64 {
	if move == nil {
		return 0.0
	}
	score := 0.0
	if move.Status != "" {
		score += 0.2
	}
	if move.VolatileStatus != "" {
		score += 0.2
	}
	if len(move.Boosts) > 0 || (move.Self != nil && len(move.Self.Boosts) > 0) || (move.SelfBoost != nil && len(move.SelfBoost.Boosts) > 0) {
		score += 0.2
	}
	if move.Secondary != nil {
		chance := math.Max(float64(move.Secondary.Chance), 10.0) / 100.0
		score += 0.25 * math.Min(chance, 1.0)
	}
	if len(move.Secondaries) > 0 {
		strongest := 0.0
		for _, sec := range move.Secondaries {
			chance := math.Max(float64(sec.Chance), 10.0) / 100.0
			if chance > strongest {
				strongest = chance
			}
		}
		score += 0.25 * math.Min(strongest, 1.0)
	}
	if score > 1.0 {
		score = 1.0
	}
	return score
}

func moveDisabledForActive(poke *simulator.PokemonState, moveName string, move *gamedata.MoveEntry, slot int) float64 {
	if poke == nil || slot < 0 || slot >= 4 || moveName == "" || move == nil {
		return 0.0
	}
	if poke.MovePP[slot] == 0 && poke.MoveMaxPP[slot] > 0 {
		return 1.0
	}
	if (poke.Volatiles&simulator.VolatileTaunt) != 0 && move.Category == "Status" {
		return 1.0
	}
	if (poke.Volatiles&simulator.VolatileEncore) != 0 && poke.LockedMove != "" && gamedata.NormalizeID(moveName) != gamedata.NormalizeID(poke.LockedMove) {
		return 1.0
	}
	if simulator.IsAttackAction(simulator.ActionMove1+slot) && poke.LockedMove != "" {
		if choice := gamedata.NormalizeID(poke.Item); choice == "choiceband" || choice == "choicespecs" || choice == "choicescarf" {
			if gamedata.NormalizeID(moveName) != gamedata.NormalizeID(poke.LockedMove) {
				return 1.0
			}
		}
	}
	return 0.0
}

// vectorizePlayerFeatures extracts 7 globals and 6*FeaturesPerSlot slot features.
func vectorizePlayerFeatures(player *simulator.PlayerState, state *simulator.BattleState) ([PlayerGlobals]float64, [6 * FeaturesPerSlot]float64) {
	var globals [PlayerGlobals]float64
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
	if player.TeamSize > 0 {
		unknownAbilities := 0.0
		unknownItems := 0.0
		unknownMoves := 0.0
		for i := 0; i < player.TeamSize; i++ {
			mon := &player.Team[i]
			if gamedata.NormalizeID(mon.Ability) == "" {
				unknownAbilities++
			}
			if gamedata.NormalizeID(mon.Item) == "" {
				unknownItems++
			}
			known := 0.0
			for m := 0; m < 4; m++ {
				if gamedata.NormalizeID(mon.Moves[m]) != "" {
					known++
				}
			}
			unknownMoves += 4.0 - known
		}
		invTeam := 1.0 / float64(player.TeamSize)
		globals[3] = float64(6-player.TeamSize) / 6.0
		globals[4] = unknownAbilities * invTeam
		globals[5] = unknownItems * invTeam
		globals[6] = unknownMoves / (4.0 * float64(player.TeamSize))
	} else {
		globals[3] = 1.0
		globals[4] = 1.0
		globals[5] = 1.0
		globals[6] = 1.0
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

	// Ability/Item identity with hashed one-hot buckets.
	writeHashedOneHot(slots, *slotsIdx, AbilityHashBuckets, poke.Ability)
	*slotsIdx += AbilityHashBuckets
	writeHashedOneHot(slots, *slotsIdx, ItemHashBuckets, poke.Item)
	*slotsIdx += ItemHashBuckets

	// Volatiles as explicit binary features.
	for i := 0; i < VolatileBits; i++ {
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

			ppRatio := 0.0
			if poke.MoveMaxPP[m] > 0 {
				ppRatio = math.Min(float64(poke.MovePP[m])/float64(poke.MoveMaxPP[m]), 1.0)
			} else if move.PP > 0 {
				ppRatio = 1.0
			}
			slots[*slotsIdx] = ppRatio
			*slotsIdx++

			slots[*slotsIdx] = moveDisabledForActive(poke, moveName, move, m)
			*slotsIdx++

			slots[*slotsIdx] = secondaryEffectScore(move)
			*slotsIdx++
		} else {
			// Zero pad empty move slots.
			for i := 0; i < MoveFeatures; i++ {
				slots[*slotsIdx] = 0.0
				*slotsIdx++
			}
		}
	}

	// Final Padding to reach FeaturesPerSlot.
	expectedEnd := startOffset + FeaturesPerSlot
	for *slotsIdx < expectedEnd {
		slots[*slotsIdx] = 0.0
		*slotsIdx++
	}
}
