package evaluator

import "github.com/pokemon-engine/simulator"

func mainMLPLayerSizes() []int {
	// 7,048,044 params in main MLP (50% of previous 14,096,088).
	return []int{TotalFeatures, 2255, 1802, 389, simulator.MaxActions}
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
