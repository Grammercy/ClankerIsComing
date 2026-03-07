package evaluator

import "testing"

func TestAttentionRouterParamCountStableAfterExpertIncrease(t *testing.T) {
	const target = 285620
	got := mlpParamCount(attentionMLPLayerSizes())
	if got != target {
		t.Fatalf("router param count mismatch: got %d expected %d", got, target)
	}
}

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

func TestAttentionWeightsFromMoEOutputTop2Average(t *testing.T) {
	raw := make([]float64, SlotMoEOutputSize)

	// expert 0 focuses slot 0, expert 1 focuses slot 1, expert 2 has extreme slot 2
	// but should be ignored if not in top-2 gates.
	raw[0*SlotAttentionSlots+0] = 10.0
	raw[1*SlotAttentionSlots+1] = 10.0
	raw[2*SlotAttentionSlots+2] = 100.0

	gateOffset := SlotAttentionExperts * SlotAttentionSlots
	raw[gateOffset+0] = 10.0
	raw[gateOffset+1] = 9.0
	raw[gateOffset+2] = -10.0

	weights, ok := attentionWeightsFromMoEOutput(raw)
	if !ok {
		t.Fatalf("expected successful MoE decode")
	}
	if !(weights[0] > weights[2] && weights[1] > weights[2]) {
		t.Fatalf("top-2 routing did not suppress non-selected expert output: w0=%.6f w1=%.6f w2=%.6f", weights[0], weights[1], weights[2])
	}
	diff := weights[0] - weights[1]
	if diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("expected averaged top-2 experts to make slots 0 and 1 equal, got %.12f vs %.12f", weights[0], weights[1])
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
