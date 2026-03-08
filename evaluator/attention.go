package evaluator

import "math"

const (
	SlotAttentionSlots      = 12
	SlotAttentionOutputSize = SlotAttentionSlots
)

func attentionMLPLayerSizes() []int {
	return []int{TotalSlotFeatures, 95, 710, 580, SlotAttentionOutputSize}
}

func softmax(logits []float64, probs []float64) bool {
	if len(logits) == 0 || len(probs) < len(logits) {
		return false
	}

	maxLogit := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxLogit {
			maxLogit = logits[i]
		}
	}

	sumExp := 0.0
	for i := 0; i < len(logits); i++ {
		p := math.Exp(logits[i] - maxLogit)
		probs[i] = p
		sumExp += p
	}
	if sumExp <= 0 || math.IsNaN(sumExp) || math.IsInf(sumExp, 0) {
		return false
	}

	inv := 1.0 / sumExp
	for i := 0; i < len(logits); i++ {
		probs[i] *= inv
	}
	return true
}

func uniformSlotWeights() [SlotAttentionSlots]float64 {
	var weights [SlotAttentionSlots]float64
	v := 1.0 / float64(SlotAttentionSlots)
	for i := range weights {
		weights[i] = v
	}
	return weights
}

func attentionWeightsFromOutput(raw []float64) ([SlotAttentionSlots]float64, bool) {
	var slotWeights [SlotAttentionSlots]float64
	if len(raw) != SlotAttentionOutputSize {
		return slotWeights, false
	}
	if !softmax(raw, slotWeights[:]) {
		return slotWeights, false
	}
	return slotWeights, true
}

func buildAttentionOutputDeltasBatch(
	rawAttentionBatch [][]float64,
	slotDeltasBatch [][]float64,
) [][]float64 {
	n := len(rawAttentionBatch)
	deltaBatch := make([][]float64, n)
	if n == 0 || len(slotDeltasBatch) != n {
		return deltaBatch
	}

	for i := 0; i < n; i++ {
		row := make([]float64, SlotAttentionOutputSize)
		slotDeltas := slotDeltasBatch[i]
		if len(slotDeltas) < SlotAttentionSlots {
			tmp := make([]float64, SlotAttentionSlots)
			copy(tmp, slotDeltas)
			slotDeltas = tmp
		}
		for slot := 0; slot < SlotAttentionSlots; slot++ {
			row[slot] = slotDeltas[slot]
		}
		deltaBatch[i] = row
	}

	return deltaBatch
}
