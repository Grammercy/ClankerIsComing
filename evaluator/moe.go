package evaluator

import "math"

const (
	SlotAttentionSlots      = 12
	SlotAttentionExperts    = 4
	SlotMoEOutputSize       = SlotAttentionExperts*SlotAttentionSlots + SlotAttentionExperts
	SlotMoEBalanceLossScale = 0.02
)

func attentionMLPLayerSizes() []int {
	return []int{TotalSlotFeatures, 256, 128, SlotMoEOutputSize}
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

func decodeMoERouting(
	raw []float64,
	slotWeights *[SlotAttentionSlots]float64,
	gateProbs *[SlotAttentionExperts]float64,
	expertLogits *[SlotAttentionExperts][SlotAttentionSlots]float64,
) bool {
	if len(raw) != SlotMoEOutputSize {
		return false
	}

	offset := 0
	for expert := 0; expert < SlotAttentionExperts; expert++ {
		for slot := 0; slot < SlotAttentionSlots; slot++ {
			expertLogits[expert][slot] = raw[offset]
			offset++
		}
	}

	var gateLogits [SlotAttentionExperts]float64
	for expert := 0; expert < SlotAttentionExperts; expert++ {
		gateLogits[expert] = raw[offset]
		offset++
	}
	if !softmax(gateLogits[:], gateProbs[:]) {
		return false
	}

	var mixedSlotLogits [SlotAttentionSlots]float64
	for slot := 0; slot < SlotAttentionSlots; slot++ {
		v := 0.0
		for expert := 0; expert < SlotAttentionExperts; expert++ {
			v += gateProbs[expert] * expertLogits[expert][slot]
		}
		mixedSlotLogits[slot] = v
	}
	if !softmax(mixedSlotLogits[:], slotWeights[:]) {
		return false
	}
	return true
}

func attentionWeightsFromMoEOutput(raw []float64) ([SlotAttentionSlots]float64, bool) {
	var (
		slotWeights  [SlotAttentionSlots]float64
		gateProbs    [SlotAttentionExperts]float64
		expertLogits [SlotAttentionExperts][SlotAttentionSlots]float64
	)
	ok := decodeMoERouting(raw, &slotWeights, &gateProbs, &expertLogits)
	return slotWeights, ok
}

func buildMoEOutputDeltasBatch(
	rawMoEBatch [][]float64,
	slotDeltasBatch [][]float64,
	sampleWeights []float64,
	balanceLossScale float64,
) ([][]float64, float64) {
	n := len(rawMoEBatch)
	deltaBatch := make([][]float64, n)
	if n == 0 || len(slotDeltasBatch) != n {
		return deltaBatch, 0
	}

	gateBatch := make([][SlotAttentionExperts]float64, n)
	expertBatch := make([][SlotAttentionExperts][SlotAttentionSlots]float64, n)

	totalSampleWeight := 0.0
	var gateUsage [SlotAttentionExperts]float64
	for i := 0; i < n; i++ {
		var slotWeights [SlotAttentionSlots]float64
		if !decodeMoERouting(rawMoEBatch[i], &slotWeights, &gateBatch[i], &expertBatch[i]) {
			for expert := 0; expert < SlotAttentionExperts; expert++ {
				gateBatch[i][expert] = 1.0 / float64(SlotAttentionExperts)
			}
		}

		w := 1.0
		if len(sampleWeights) == n && sampleWeights[i] > 0 {
			w = sampleWeights[i]
		}
		totalSampleWeight += w
		for expert := 0; expert < SlotAttentionExperts; expert++ {
			gateUsage[expert] += w * gateBatch[i][expert]
		}
	}

	var usageMean [SlotAttentionExperts]float64
	if totalSampleWeight > 0 {
		invTotal := 1.0 / totalSampleWeight
		for expert := 0; expert < SlotAttentionExperts; expert++ {
			usageMean[expert] = gateUsage[expert] * invTotal
		}
	}

	balanceLoss := 0.0
	targetUsage := 1.0 / float64(SlotAttentionExperts)
	for expert := 0; expert < SlotAttentionExperts; expert++ {
		diff := usageMean[expert] - targetUsage
		balanceLoss += diff * diff
	}
	balanceLoss *= balanceLossScale

	balanceGradScale := 0.0
	if balanceLossScale > 0 && totalSampleWeight > 0 {
		balanceGradScale = 2.0 * balanceLossScale / totalSampleWeight
	}

	for i := 0; i < n; i++ {
		row := make([]float64, SlotMoEOutputSize)
		slotDeltas := slotDeltasBatch[i]
		if len(slotDeltas) < SlotAttentionSlots {
			tmp := make([]float64, SlotAttentionSlots)
			copy(tmp, slotDeltas)
			slotDeltas = tmp
		}

		var gateProbDeltas [SlotAttentionExperts]float64
		for expert := 0; expert < SlotAttentionExperts; expert++ {
			base := expert * SlotAttentionSlots
			g := gateBatch[i][expert]
			dot := 0.0
			for slot := 0; slot < SlotAttentionSlots; slot++ {
				delta := g * slotDeltas[slot]
				row[base+slot] = delta
				dot += slotDeltas[slot] * expertBatch[i][expert][slot]
			}
			gateProbDeltas[expert] = dot
		}

		if balanceGradScale > 0 {
			sampleWeight := 1.0
			if len(sampleWeights) == n && sampleWeights[i] > 0 {
				sampleWeight = sampleWeights[i]
			}
			for expert := 0; expert < SlotAttentionExperts; expert++ {
				gateProbDeltas[expert] += balanceGradScale * sampleWeight * (usageMean[expert] - targetUsage)
			}
		}

		// dL/dgate_logits = softmax_j * (dL/dgate_prob_j - sum_k softmax_k*dL/dgate_prob_k)
		weightedMean := 0.0
		for expert := 0; expert < SlotAttentionExperts; expert++ {
			weightedMean += gateBatch[i][expert] * gateProbDeltas[expert]
		}
		gateOffset := SlotAttentionExperts * SlotAttentionSlots
		for expert := 0; expert < SlotAttentionExperts; expert++ {
			row[gateOffset+expert] = gateBatch[i][expert] * (gateProbDeltas[expert] - weightedMean)
		}

		deltaBatch[i] = row
	}

	return deltaBatch, balanceLoss
}
