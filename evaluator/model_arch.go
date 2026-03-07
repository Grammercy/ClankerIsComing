package evaluator

import "github.com/pokemon-engine/simulator"

func mainMLPLayerSizes() []int {
	// 14,102,058 params in main MLP (exactly 3x previous 4,700,686).
	return []int{TotalFeatures, 3374, 2690, 596, simulator.MaxActions}
}

func mlpParamCount(sizes []int) int {
	if len(sizes) < 2 {
		return 0
	}
	total := 0
	for i := 0; i < len(sizes)-1; i++ {
		total += sizes[i]*sizes[i+1] + sizes[i+1]
	}
	return total
}

func totalEvaluatorParamCount() int {
	return mlpParamCount(mainMLPLayerSizes()) + mlpParamCount(attentionMLPLayerSizes())
}
