package evaluator

func mainMLPLayerSizes() []int {
	// Main head outputs actions plus 2 recurrent latent token channels.
	return []int{TotalFeatures, 1700, 1760, 1130, MainOutputSize}
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
