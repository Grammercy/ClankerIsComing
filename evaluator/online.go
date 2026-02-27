package evaluator

import (
	"fmt"
	"log"

	"github.com/pokemon-engine/simulator"
)

// ExperienceTuple holds a single (features, action) pair for online RL
type ExperienceTuple struct {
	Features [TotalFeatures]float64
	Action   int
}

// OnlineLearn performs a single reinforcement learning update pass over a batch of experiences.
// For each experience, it pushes the Q-value of the chosen action toward the game outcome (reward),
// while leaving other Q-values untouched (zero gradient).
//
// reward: 1.0 for win, 0.0 for loss, 0.5 for draw
func OnlineLearn(experiences []ExperienceTuple, reward float64) {
	if GlobalMLP == nil || GlobalAttentionMLP == nil {
		log.Println("[OnlineLearn] Skipping: no loaded weights")
		return
	}
	if len(experiences) == 0 {
		return
	}

	learningRate := 0.001 // Smaller for online updates to avoid catastrophic forgetting
	mlp := GlobalMLP
	attentionMLP := GlobalAttentionMLP

	totalLoss := 0.0

	for _, exp := range experiences {
		// Forward pass to get current predictions
		currentPredictions := mlp.Forward(exp.Features[:], nil)

		// Build targets: keep current predictions for all actions (zero gradient),
		// except for the chosen action which we push toward the reward
		targets := make([]float64, simulator.MaxActions)
		for i := 0; i < simulator.MaxActions; i++ {
			targets[i] = currentPredictions[i]
		}

		// Only train the action that was actually taken
		if exp.Action >= 0 && exp.Action < simulator.MaxActions {
			targets[exp.Action] = reward
		}

		// Backprop through the main MLP (BCE LOSS)
		loss := mlp.CalculateBCELocalGradients(exp.Features[:], targets, 1.0, nil)
		totalLoss += loss

		// Also backprop the attention MLP with the feature gradients
		// For simplicity in online mode, we skip attention backprop
		// since feature vectors are pre-computed
	}

	// Apply accumulated gradients using Adam
	mlp.ApplyAdamGradients(nil, float64(len(experiences)), learningRate, 0.9, 0.999, 1e-8)

	avgLoss := totalLoss / float64(len(experiences))
	fmt.Printf("[OnlineLearn] Trained on %d experiences (reward=%.1f, avg_loss=%.6f)\n",
		len(experiences), reward, avgLoss)

	// Save updated weights immediately
	if err := mlp.SaveWeights("evaluator_weights.json"); err != nil {
		log.Printf("[OnlineLearn] Failed to save weights: %v", err)
	}
	if err := attentionMLP.SaveWeights("attention_weights.json"); err != nil {
		log.Printf("[OnlineLearn] Failed to save attention weights: %v", err)
	}
}
