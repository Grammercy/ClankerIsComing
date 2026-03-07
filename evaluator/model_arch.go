package evaluator

import "github.com/pokemon-engine/simulator"

func mainMLPLayerSizes() []int {
	// 675,626 params in main MLP for much faster training/inference.
	return []int{TotalFeatures, 512, 256, 96, simulator.MaxActions}
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
