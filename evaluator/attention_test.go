package evaluator

import "testing"

func TestAttentionHeadOutputSize(t *testing.T) {
	sizes := attentionMLPLayerSizes()
	if sizes[len(sizes)-1] != SlotAttentionOutputSize {
		t.Fatalf("unexpected attention output size: got %d expected %d", sizes[len(sizes)-1], SlotAttentionOutputSize)
	}
}

func TestAttentionWeightsFromOutputUniform(t *testing.T) {
	raw := make([]float64, SlotAttentionOutputSize)
	weights, ok := attentionWeightsFromOutput(raw)
	if !ok {
		t.Fatalf("expected successful attention decode")
	}
	expected := 1.0 / float64(SlotAttentionSlots)
	for i, w := range weights {
		if diff := w - expected; diff > 1e-9 || diff < -1e-9 {
			t.Fatalf("slot %d weight mismatch: got %.12f expected %.12f", i, w, expected)
		}
	}
}

func TestAttentionWeightsFromOutputInvalidSize(t *testing.T) {
	if _, ok := attentionWeightsFromOutput(make([]float64, SlotAttentionOutputSize-1)); ok {
		t.Fatalf("expected decode failure for incompatible output size")
	}
}

func TestBuildAttentionOutputDeltasBatch(t *testing.T) {
	rawBatch := [][]float64{
		make([]float64, SlotAttentionOutputSize),
		make([]float64, SlotAttentionOutputSize),
	}
	slotDeltaBatch := make([][]float64, 2)
	for i := range slotDeltaBatch {
		row := make([]float64, SlotAttentionSlots)
		for j := range row {
			row[j] = float64(i + j)
		}
		slotDeltaBatch[i] = row
	}

	deltaBatch := buildAttentionOutputDeltasBatch(rawBatch, slotDeltaBatch)
	if len(deltaBatch) != len(rawBatch) {
		t.Fatalf("unexpected delta batch size: got %d expected %d", len(deltaBatch), len(rawBatch))
	}
	for i := range deltaBatch {
		if len(deltaBatch[i]) != SlotAttentionOutputSize {
			t.Fatalf("row %d has unexpected delta size %d", i, len(deltaBatch[i]))
		}
		for j := 0; j < SlotAttentionSlots; j++ {
			if deltaBatch[i][j] != slotDeltaBatch[i][j] {
				t.Fatalf("delta mismatch row=%d slot=%d got=%.6f expected=%.6f", i, j, deltaBatch[i][j], slotDeltaBatch[i][j])
			}
		}
	}
}
