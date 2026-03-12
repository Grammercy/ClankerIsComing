package deepcfr

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pokemon-engine/simulator"
)

const (
	targetPredictorAuto     = "auto"
	targetPredictorCPU      = "cpu"
	targetPredictorCPUBatch = "cpu-batch"
	targetPredictorOpenCL   = "opencl"
)

type statePredictor interface {
	Predict(features []float64, legalMask []float64) ([simulator.MaxActions]float64, [simulator.MaxActions]float64, float64)
	Close() error
	Name() string
}

type directModelPredictor struct {
	model *Model
}

func (p *directModelPredictor) Predict(features []float64, legalMask []float64) ([simulator.MaxActions]float64, [simulator.MaxActions]float64, float64) {
	if p == nil || p.model == nil {
		var regrets [simulator.MaxActions]float64
		var policy [simulator.MaxActions]float64
		return regrets, policy, 0.5
	}
	return p.model.Predict(features, legalMask)
}

func (p *directModelPredictor) Close() error {
	return nil
}

func (p *directModelPredictor) Name() string {
	return "cpu-direct"
}

type batchPredictBackend interface {
	PredictBatch(features []float32, legalMasks []float32, batch int, outRegret []float32, outPolicy []float32, outValue []float32) error
	Close() error
	Name() string
}

type cpuBatchPredictBackend struct {
	model *Model
}

func (b *cpuBatchPredictBackend) PredictBatch(features []float32, legalMasks []float32, batch int, outRegret []float32, outPolicy []float32, outValue []float32) error {
	if batch < 0 {
		batch = 0
	}
	for i := 0; i < batch; i++ {
		featureOffset := i * FeatureSize
		maskOffset := i * simulator.MaxActions
		features64 := make([]float64, FeatureSize)
		mask64 := make([]float64, simulator.MaxActions)
		for j := 0; j < FeatureSize; j++ {
			features64[j] = float64(features[featureOffset+j])
		}
		for j := 0; j < simulator.MaxActions; j++ {
			mask64[j] = float64(legalMasks[maskOffset+j])
		}
		regret, policy, value := b.model.Predict(features64, mask64)
		for j := 0; j < simulator.MaxActions; j++ {
			outRegret[maskOffset+j] = float32(regret[j])
			outPolicy[maskOffset+j] = float32(policy[j])
		}
		outValue[i] = float32(value)
	}
	return nil
}

func (b *cpuBatchPredictBackend) Close() error {
	return nil
}

func (b *cpuBatchPredictBackend) Name() string {
	return "cpu"
}

type predictRequest struct {
	features []float64
	mask     []float64
	resp     chan predictResponse
}

type predictResponse struct {
	regrets [simulator.MaxActions]float64
	policy  [simulator.MaxActions]float64
	value   float64
}

type asyncPredictor struct {
	model      *Model
	backend    batchPredictBackend
	batchSize  int
	flushAfter time.Duration
	requests   chan predictRequest
	wg         sync.WaitGroup
	closeOnce  sync.Once
}

type predictorMetricsSnapshot struct {
	Batches   int64
	States    int64
	Fallbacks int64
	Duration  time.Duration
}

var predictorMetrics struct {
	batches   atomic.Int64
	states    atomic.Int64
	fallbacks atomic.Int64
	nanos     atomic.Int64
}

func resetPredictorMetrics() {
	predictorMetrics.batches.Store(0)
	predictorMetrics.states.Store(0)
	predictorMetrics.fallbacks.Store(0)
	predictorMetrics.nanos.Store(0)
}

func snapshotPredictorMetrics() predictorMetricsSnapshot {
	return predictorMetricsSnapshot{
		Batches:   predictorMetrics.batches.Load(),
		States:    predictorMetrics.states.Load(),
		Fallbacks: predictorMetrics.fallbacks.Load(),
		Duration:  time.Duration(predictorMetrics.nanos.Load()),
	}
}

func newAsyncPredictor(model *Model, backend batchPredictBackend, batchSize int, queueSize int) *asyncPredictor {
	if batchSize <= 0 {
		batchSize = 2048
	}
	if queueSize <= 0 {
		queueSize = 8192
	}
	p := &asyncPredictor{
		model:      model,
		backend:    backend,
		batchSize:  batchSize,
		flushAfter: 200 * time.Microsecond,
		requests:   make(chan predictRequest, queueSize),
	}
	p.wg.Add(1)
	go p.loop()
	return p
}

func (p *asyncPredictor) Predict(features []float64, legalMask []float64) ([simulator.MaxActions]float64, [simulator.MaxActions]float64, float64) {
	if p == nil || p.backend == nil {
		var regrets [simulator.MaxActions]float64
		var policy [simulator.MaxActions]float64
		return regrets, policy, 0.5
	}
	req := predictRequest{
		features: append([]float64(nil), features...),
		mask:     append([]float64(nil), legalMask...),
		resp:     make(chan predictResponse, 1),
	}
	p.requests <- req
	res := <-req.resp
	return res.regrets, res.policy, res.value
}

func (p *asyncPredictor) Close() error {
	if p == nil {
		return nil
	}
	p.closeOnce.Do(func() {
		close(p.requests)
		p.wg.Wait()
	})
	if p.backend != nil {
		return p.backend.Close()
	}
	return nil
}

func (p *asyncPredictor) Name() string {
	if p == nil || p.backend == nil {
		return "unknown"
	}
	return p.backend.Name()
}

func (p *asyncPredictor) loop() {
	defer p.wg.Done()

	batch := make([]predictRequest, 0, p.batchSize)
	timer := time.NewTimer(0)
	if !timer.Stop() {
		<-timer.C
	}
	timerActive := false

	flush := func() {
		if len(batch) == 0 {
			return
		}
		p.runBatch(batch)
		batch = batch[:0]
	}

	for {
		if len(batch) == 0 {
			req, ok := <-p.requests
			if !ok {
				flush()
				return
			}
			batch = append(batch, req)
			timer.Reset(p.flushAfter)
			timerActive = true
			continue
		}

		select {
		case req, ok := <-p.requests:
			if !ok {
				if timerActive {
					if !timer.Stop() {
						select {
						case <-timer.C:
						default:
						}
					}
					timerActive = false
				}
				flush()
				return
			}
			batch = append(batch, req)
			if len(batch) >= p.batchSize {
				if timerActive {
					if !timer.Stop() {
						select {
						case <-timer.C:
						default:
						}
					}
					timerActive = false
				}
				flush()
			}
		case <-timer.C:
			timerActive = false
			flush()
		}
	}
}

func (p *asyncPredictor) runBatch(requests []predictRequest) {
	n := len(requests)
	if n == 0 {
		return
	}
	features := make([]float32, n*FeatureSize)
	masks := make([]float32, n*simulator.MaxActions)
	for row, req := range requests {
		featureOffset := row * FeatureSize
		maskOffset := row * simulator.MaxActions
		for i := 0; i < FeatureSize && i < len(req.features); i++ {
			features[featureOffset+i] = float32(req.features[i])
		}
		for i := 0; i < simulator.MaxActions && i < len(req.mask); i++ {
			masks[maskOffset+i] = float32(req.mask[i])
		}
	}

	regret := make([]float32, n*simulator.MaxActions)
	policy := make([]float32, n*simulator.MaxActions)
	value := make([]float32, n)
	start := time.Now()
	if err := p.backend.PredictBatch(features, masks, n, regret, policy, value); err != nil {
		predictorMetrics.fallbacks.Add(1)
		for _, req := range requests {
			r, s, v := p.model.Predict(req.features, req.mask)
			req.resp <- predictResponse{regrets: r, policy: s, value: v}
		}
		return
	}
	predictorMetrics.batches.Add(1)
	predictorMetrics.states.Add(int64(n))
	predictorMetrics.nanos.Add(time.Since(start).Nanoseconds())

	for row, req := range requests {
		maskOffset := row * simulator.MaxActions
		var outRegret [simulator.MaxActions]float64
		var outPolicy [simulator.MaxActions]float64
		for i := 0; i < simulator.MaxActions; i++ {
			outRegret[i] = float64(regret[maskOffset+i])
			outPolicy[i] = float64(policy[maskOffset+i])
		}
		req.resp <- predictResponse{
			regrets: outRegret,
			policy:  outPolicy,
			value:   float64(value[row]),
		}
	}
}

func newStatePredictor(model *Model, cfg TrainConfig) (statePredictor, string, error) {
	if model == nil {
		return &directModelPredictor{model: model}, "", nil
	}
	mode := strings.ToLower(strings.TrimSpace(cfg.TargetPredictor))
	if mode == "" {
		mode = targetPredictorAuto
	}

	batchSize := cfg.TargetBatchSize
	if batchSize <= 0 {
		batchSize = 2048
	}
	queueSize := cfg.TargetQueueSize
	if queueSize <= 0 {
		queueSize = 8192
	}

	switch mode {
	case targetPredictorCPU:
		return &directModelPredictor{model: model}, "", nil
	case targetPredictorCPUBatch:
		return newAsyncPredictor(model, &cpuBatchPredictBackend{model: model}, batchSize, queueSize), "", nil
	case targetPredictorOpenCL:
		backend, err := newOpenCLBatchPredictBackend(model, cfg.OpenCLPlatform, cfg.OpenCLDevice, batchSize)
		if err != nil {
			return nil, "", err
		}
		return newAsyncPredictor(model, backend, batchSize, queueSize), "", nil
	case targetPredictorAuto:
		return &directModelPredictor{model: model}, "", nil
	default:
		return nil, "", fmt.Errorf("unknown target predictor %q", cfg.TargetPredictor)
	}
}
