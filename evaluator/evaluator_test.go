package evaluator

import (
	"testing"
)

func TestHashFeatures(t *testing.T) {
	t.Run("Determinism", func(t *testing.T) {
		var f1, f2 [TotalFeatures]float64
		for i := 0; i < TotalFeatures; i++ {
			f1[i] = float64(i) * 0.1
			f2[i] = float64(i) * 0.1
		}

		h1 := HashFeatures(&f1)
		h2 := HashFeatures(&f2)

		if h1 != h2 {
			t.Errorf("Expected identical hashes for identical inputs, got %d and %d", h1, h2)
		}
	})

	t.Run("Difference", func(t *testing.T) {
		var f1, f2 [TotalFeatures]float64
		for i := 0; i < TotalFeatures; i++ {
			f1[i] = float64(i) * 0.1
			f2[i] = float64(i) * 0.1
		}

		// Change a single element
		f2[0] = 1.0

		h1 := HashFeatures(&f1)
		h2 := HashFeatures(&f2)

		if h1 == h2 {
			t.Errorf("Expected different hashes for different inputs, got %d", h1)
		}
	})

	t.Run("ZeroVsNonZero", func(t *testing.T) {
		var f1, f2 [TotalFeatures]float64
		// f1 is all zeros
		for i := 0; i < TotalFeatures; i++ {
			f2[i] = 1.0
		}

		h1 := HashFeatures(&f1)
		h2 := HashFeatures(&f2)

		if h1 == h2 {
			t.Errorf("Expected different hashes for zero vs non-zero inputs, got %d", h1)
		}
	})

	t.Run("SwappedElements", func(t *testing.T) {
		var f1, f2 [TotalFeatures]float64
		f1[0] = 1.0
		f1[1] = 2.0

		f2[0] = 2.0
		f2[1] = 1.0

		h1 := HashFeatures(&f1)
		h2 := HashFeatures(&f2)

		if h1 == h2 {
			t.Errorf("Expected different hashes for swapped elements, got %d", h1)
		}
	})

	t.Run("ZeroValuesHash", func(t *testing.T) {
		var f [TotalFeatures]float64
		h := HashFeatures(&f)
		if h == 0 {
			t.Errorf("Hash of zero array should not be 0")
		}
	})
}
