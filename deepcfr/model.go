package deepcfr

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/pokemon-engine/simulator"
)

type Model struct {
	InputSize int       `json:"inputSize"`
	Hidden1   int       `json:"hidden1"`
	Hidden2   int       `json:"hidden2"`
	W1        []float64 `json:"w1"`
	B1        []float64 `json:"b1"`
	W2        []float64 `json:"w2"`
	B2        []float64 `json:"b2"`
	WRegret   []float64 `json:"wRegret"`
	BRegret   []float64 `json:"bRegret"`
	WStrategy []float64 `json:"wStrategy"`
	BStrategy []float64 `json:"bStrategy"`
	WValue    []float64 `json:"wValue"`
	BValue    float64   `json:"bValue"`
	Priors    *Priors   `json:"priors,omitempty"`
}

type TrainingExample struct {
	Features        []float64
	LegalMask       []float64
	RegretTargets   [simulator.MaxActions]float64
	StrategyTargets [simulator.MaxActions]float64
	ValueTarget     float64
}

type TrainingHyperParams struct {
	LearningRate   float64
	RegretWeight   float64
	StrategyWeight float64
	ValueWeight    float64
}

type TrainingMetrics struct {
	Loss        float64
	ValueError  float64
	PolicyCross float64
}

type forwardPass struct {
	h1Pre          []float64
	h1             []float64
	h2Pre          []float64
	h2             []float64
	regret         []float64
	strategyLogits []float64
	strategy       []float64
	value          float64
}

func NewModel(seed int64) *Model {
	rng := rand.New(rand.NewSource(seed))
	m := &Model{
		InputSize: FeatureSize,
		Hidden1:   128,
		Hidden2:   96,
		W1:        make([]float64, FeatureSize*128),
		B1:        make([]float64, 128),
		W2:        make([]float64, 128*96),
		B2:        make([]float64, 96),
		WRegret:   make([]float64, 96*simulator.MaxActions),
		BRegret:   make([]float64, simulator.MaxActions),
		WStrategy: make([]float64, 96*simulator.MaxActions),
		BStrategy: make([]float64, simulator.MaxActions),
		WValue:    make([]float64, 96),
	}
	initXavier(m.W1, FeatureSize, 128, rng)
	initXavier(m.W2, 128, 96, rng)
	initXavier(m.WRegret, 96, simulator.MaxActions, rng)
	initXavier(m.WStrategy, 96, simulator.MaxActions, rng)
	initXavier(m.WValue, 96, 1, rng)
	return m
}

func LoadModel(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read model: %w", err)
	}
	var model Model
	if err := json.Unmarshal(data, &model); err != nil {
		return nil, fmt.Errorf("failed to parse model: %w", err)
	}
	if model.InputSize != FeatureSize {
		return nil, fmt.Errorf("model input size %d does not match current encoder size %d", model.InputSize, FeatureSize)
	}
	return &model, nil
}

func (m *Model) Save(path string) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write model: %w", err)
	}
	return nil
}

func (m *Model) Clone() *Model {
	if m == nil {
		return nil
	}
	cloned := &Model{
		InputSize: m.InputSize,
		Hidden1:   m.Hidden1,
		Hidden2:   m.Hidden2,
		W1:        append([]float64(nil), m.W1...),
		B1:        append([]float64(nil), m.B1...),
		W2:        append([]float64(nil), m.W2...),
		B2:        append([]float64(nil), m.B2...),
		WRegret:   append([]float64(nil), m.WRegret...),
		BRegret:   append([]float64(nil), m.BRegret...),
		WStrategy: append([]float64(nil), m.WStrategy...),
		BStrategy: append([]float64(nil), m.BStrategy...),
		WValue:    append([]float64(nil), m.WValue...),
		BValue:    m.BValue,
		Priors:    m.Priors,
	}
	return cloned
}

func (m *Model) Predict(features []float64, legalMask []float64) ([simulator.MaxActions]float64, [simulator.MaxActions]float64, float64) {
	pass := m.forward(features, legalMask)
	var regrets [simulator.MaxActions]float64
	var strategy [simulator.MaxActions]float64
	for i := 0; i < simulator.MaxActions; i++ {
		regrets[i] = pass.regret[i]
		strategy[i] = pass.strategy[i]
	}
	return regrets, strategy, pass.value
}

func (m *Model) TrainExample(ex TrainingExample, hp TrainingHyperParams) TrainingMetrics {
	if hp.LearningRate <= 0 {
		hp.LearningRate = 0.0005
	}
	if hp.RegretWeight <= 0 {
		hp.RegretWeight = 1.0
	}
	if hp.StrategyWeight <= 0 {
		hp.StrategyWeight = 1.0
	}
	if hp.ValueWeight <= 0 {
		hp.ValueWeight = 0.5
	}

	pass := m.forward(ex.Features, ex.LegalMask)
	dh2 := make([]float64, m.Hidden2)

	loss := 0.0
	policyCross := 0.0
	valueError := math.Abs(pass.value - ex.ValueTarget)

	valueGrad := (pass.value - ex.ValueTarget) * hp.ValueWeight
	loss += valueError * valueError * hp.ValueWeight
	for i := 0; i < m.Hidden2; i++ {
		dh2[i] += valueGrad * m.WValue[i]
		m.WValue[i] -= hp.LearningRate * clipGrad(valueGrad*pass.h2[i])
	}
	m.BValue -= hp.LearningRate * clipGrad(valueGrad)

	for out := 0; out < simulator.MaxActions; out++ {
		mask := 0.0
		if out < len(ex.LegalMask) {
			mask = ex.LegalMask[out]
		}
		if mask == 0 {
			continue
		}

		dRegret := 2.0 * (pass.regret[out] - ex.RegretTargets[out]) * hp.RegretWeight
		loss += (pass.regret[out] - ex.RegretTargets[out]) * (pass.regret[out] - ex.RegretTargets[out]) * hp.RegretWeight
		for i := 0; i < m.Hidden2; i++ {
			dh2[i] += dRegret * m.WRegret[out*m.Hidden2+i]
			m.WRegret[out*m.Hidden2+i] -= hp.LearningRate * clipGrad(dRegret*pass.h2[i])
		}
		m.BRegret[out] -= hp.LearningRate * clipGrad(dRegret)

		dStrategy := (pass.strategy[out] - ex.StrategyTargets[out]) * hp.StrategyWeight
		if ex.StrategyTargets[out] > 0 && pass.strategy[out] > 1e-9 {
			policyCross += -ex.StrategyTargets[out] * math.Log(pass.strategy[out])
		}
		loss += dStrategy * dStrategy
		for i := 0; i < m.Hidden2; i++ {
			dh2[i] += dStrategy * m.WStrategy[out*m.Hidden2+i]
			m.WStrategy[out*m.Hidden2+i] -= hp.LearningRate * clipGrad(dStrategy*pass.h2[i])
		}
		m.BStrategy[out] -= hp.LearningRate * clipGrad(dStrategy)
	}

	dh1 := make([]float64, m.Hidden1)
	for i := 0; i < m.Hidden2; i++ {
		if pass.h2Pre[i] <= 0 {
			dh2[i] = 0
		}
		for j := 0; j < m.Hidden1; j++ {
			dh1[j] += dh2[i] * m.W2[i*m.Hidden1+j]
			m.W2[i*m.Hidden1+j] -= hp.LearningRate * clipGrad(dh2[i]*pass.h1[j])
		}
		m.B2[i] -= hp.LearningRate * clipGrad(dh2[i])
	}

	for i := 0; i < m.Hidden1; i++ {
		if pass.h1Pre[i] <= 0 {
			dh1[i] = 0
		}
		for j := 0; j < m.InputSize; j++ {
			m.W1[i*m.InputSize+j] -= hp.LearningRate * clipGrad(dh1[i]*ex.Features[j])
		}
		m.B1[i] -= hp.LearningRate * clipGrad(dh1[i])
	}

	return TrainingMetrics{
		Loss:        loss,
		ValueError:  valueError,
		PolicyCross: policyCross,
	}
}

func (m *Model) forward(features []float64, legalMask []float64) forwardPass {
	h1Pre := make([]float64, m.Hidden1)
	h1 := make([]float64, m.Hidden1)
	for i := 0; i < m.Hidden1; i++ {
		sum := m.B1[i]
		row := i * m.InputSize
		for j := 0; j < m.InputSize && j < len(features); j++ {
			sum += m.W1[row+j] * features[j]
		}
		h1Pre[i] = sum
		if sum > 0 {
			h1[i] = sum
		}
	}

	h2Pre := make([]float64, m.Hidden2)
	h2 := make([]float64, m.Hidden2)
	for i := 0; i < m.Hidden2; i++ {
		sum := m.B2[i]
		row := i * m.Hidden1
		for j := 0; j < m.Hidden1; j++ {
			sum += m.W2[row+j] * h1[j]
		}
		h2Pre[i] = sum
		if sum > 0 {
			h2[i] = sum
		}
	}

	regret := make([]float64, simulator.MaxActions)
	strategyLogits := make([]float64, simulator.MaxActions)
	for out := 0; out < simulator.MaxActions; out++ {
		rSum := m.BRegret[out]
		sSum := m.BStrategy[out]
		row := out * m.Hidden2
		for i := 0; i < m.Hidden2; i++ {
			rSum += m.WRegret[row+i] * h2[i]
			sSum += m.WStrategy[row+i] * h2[i]
		}
		regret[out] = rSum
		strategyLogits[out] = sSum
	}

	valueLogit := m.BValue
	for i := 0; i < m.Hidden2; i++ {
		valueLogit += m.WValue[i] * h2[i]
	}

	return forwardPass{
		h1Pre:          h1Pre,
		h1:             h1,
		h2Pre:          h2Pre,
		h2:             h2,
		regret:         regret,
		strategyLogits: strategyLogits,
		strategy:       maskedSoftmax(strategyLogits, legalMask),
		value:          sigmoid(valueLogit),
	}
}

func maskedSoftmax(logits []float64, legalMask []float64) []float64 {
	out := make([]float64, len(logits))
	maxLogit := math.Inf(-1)
	valid := 0
	for i := range logits {
		if i < len(legalMask) && legalMask[i] > 0 {
			valid++
			if logits[i] > maxLogit {
				maxLogit = logits[i]
			}
		}
	}
	if valid == 0 {
		return out
	}
	sum := 0.0
	for i := range logits {
		if i >= len(legalMask) || legalMask[i] == 0 {
			continue
		}
		out[i] = math.Exp(logits[i] - maxLogit)
		sum += out[i]
	}
	if sum == 0 {
		uniform := 1.0 / float64(valid)
		for i := range out {
			if i < len(legalMask) && legalMask[i] > 0 {
				out[i] = uniform
			}
		}
		return out
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func initXavier(weights []float64, fanIn int, fanOut int, rng *rand.Rand) {
	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
	for i := range weights {
		weights[i] = (rng.Float64()*2.0 - 1.0) * limit
	}
}

func clipGrad(v float64) float64 {
	if v > 5 {
		return 5
	}
	if v < -5 {
		return -5
	}
	return v
}

func sigmoid(v float64) float64 {
	if v >= 0 {
		z := math.Exp(-v)
		return 1.0 / (1.0 + z)
	}
	z := math.Exp(v)
	return z / (1.0 + z)
}
