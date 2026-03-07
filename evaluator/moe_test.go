package evaluator

import "testing"

func TestAttentionWeightsFromMoEOutputUniform(t *testing.T) {
	raw := make([]float64, SlotMoEOutputSize)
	weights, ok := attentionWeightsFromMoEOutput(raw)
	if !ok {
		t.Fatalf("expected successful MoE decode")
	}

	expected := 1.0 / float64(SlotAttentionSlots)
	for i, w := range weights {
		if diff := w - expected; diff > 1e-9 || diff < -1e-9 {
			t.Fatalf("slot %d weight mismatch: got %.12f expected %.12f", i, w, expected)
		}
	}
}

func TestAttentionWeightsFromMoEOutputInvalidSize(t *testing.T) {
	if _, ok := attentionWeightsFromMoEOutput(make([]float64, 12)); ok {
		t.Fatalf("expected decode failure for incompatible output size")
	}
}

func TestBuildMoEOutputDeltasBatchBalanceGradient(t *testing.T) {
	rawBatch := make([][]float64, 2)
	for i := range rawBatch {
		row := make([]float64, SlotMoEOutputSize)
		gateOffset := SlotAttentionExperts * SlotAttentionSlots
		row[gateOffset] = 10.0 // heavily route both samples to expert 0
		rawBatch[i] = row
	}

	slotDeltaBatch := make([][]float64, 2)
	for i := range slotDeltaBatch {
		slotDeltaBatch[i] = make([]float64, SlotAttentionSlots) // isolate balance-only gradient
	}

	deltaBatch, balanceLoss := buildMoEOutputDeltasBatch(rawBatch, slotDeltaBatch, nil, SlotMoEBalanceLossScale)
	if balanceLoss <= 0 {
		t.Fatalf("expected positive balance loss")
	}
	if len(deltaBatch) != len(rawBatch) {
		t.Fatalf("unexpected delta batch size: got %d expected %d", len(deltaBatch), len(rawBatch))
	}

	gateOffset := SlotAttentionExperts * SlotAttentionSlots
	for i, row := range deltaBatch {
		if len(row) != SlotMoEOutputSize {
			t.Fatalf("row %d has unexpected delta size %d", i, len(row))
		}
		if row[gateOffset] <= 0 {
			t.Fatalf("expert 0 gate delta should be positive when over-utilized, got %.8f", row[gateOffset])
		}
		for expert := 1; expert < SlotAttentionExperts; expert++ {
			if row[gateOffset+expert] >= 0 {
				t.Fatalf("expert %d gate delta should be negative when under-utilized, got %.8f", expert, row[gateOffset+expert])
			}
		}
	}
}
