package bot

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pokemon-engine/evaluator"
	"github.com/pokemon-engine/simulator"
)

// SearchResult holds the best action and its evaluated score
type SearchResult struct {
	BestAction    int
	Score         float64
	Depth         int
	NodesSearched int
	ActionScores  map[int]float64 // per-action eval scores
	TimedOut      bool
}

type orderedAction struct {
	action   int
	estimate float64
}

type searchContext struct {
	deadline              time.Time
	useDeadline           bool
	timedOut              bool
	useLatentTokens       bool
	latentReasoningToken  float64
	latentPredictionToken float64
}

type mctsActionStat struct {
	visits      int
	virtualLoss int
	valueSum    float64
}

type mctsNode struct {
	mu           sync.Mutex
	visits       int
	virtualLoss  int
	p1ActionStat map[int]*mctsActionStat
	p2ActionStat map[int]*mctsActionStat
	children     map[int]*mctsNode
}

type mctsPathStep struct {
	node   *mctsNode
	p1Hero int
	p2Hero int
}

type mctsConfig struct {
	rolloutDepth int
	simulations  int
	workers      int
	exploreC     float64
}

type mctsLeafEvalRequest struct {
	state simulator.BattleState
	resp  chan float64
}

type mctsLeafBatcher struct {
	reqCh                 chan mctsLeafEvalRequest
	stopCh                chan struct{}
	wg                    sync.WaitGroup
	batchSize             int
	flushWindow           time.Duration
	useLatentTokens       bool
	latentReasoningToken  float64
	latentPredictionToken float64
}

func (ctx *searchContext) shouldStop(nodesSearched int) bool {
	if ctx == nil || !ctx.useDeadline || ctx.timedOut {
		return ctx != nil && ctx.timedOut
	}
	if nodesSearched&63 == 0 && time.Now().After(ctx.deadline) {
		ctx.timedOut = true
	}
	return ctx.timedOut
}

func parsePositiveIntEnv(name string) (int, bool) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return 0, false
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		return 0, false
	}
	return v, true
}

func parsePositiveFloatEnv(name string) (float64, bool) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil || v <= 0 {
		return 0, false
	}
	return v, true
}

func evaluateWithContext(state *simulator.BattleState, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable, ctx *searchContext) float64 {
	if ctx != nil && ctx.useLatentTokens {
		return evaluator.EvaluateWithLatentTokens(state, -1, evaluator.GlobalMLP, evaluator.GlobalAttentionMLP, mlpCache, attentionCache, tt, ctx.latentReasoningToken, ctx.latentPredictionToken)
	}
	return evaluator.Evaluate(state, -1, evaluator.GlobalMLP, evaluator.GlobalAttentionMLP, mlpCache, attentionCache, tt)
}

func newMCTSLeafBatcher(batchSize int, flushWindow time.Duration, ctx *searchContext) *mctsLeafBatcher {
	if batchSize < 1 {
		batchSize = 1
	}
	if flushWindow <= 0 {
		flushWindow = 200 * time.Microsecond
	}
	b := &mctsLeafBatcher{
		reqCh:       make(chan mctsLeafEvalRequest, batchSize*8),
		stopCh:      make(chan struct{}),
		batchSize:   batchSize,
		flushWindow: flushWindow,
	}
	if ctx != nil && ctx.useLatentTokens {
		b.useLatentTokens = true
		b.latentReasoningToken = ctx.latentReasoningToken
		b.latentPredictionToken = ctx.latentPredictionToken
	}
	b.wg.Add(1)
	go b.loop()
	return b
}

func (b *mctsLeafBatcher) Evaluate(state simulator.BattleState) float64 {
	resp := make(chan float64, 1)
	req := mctsLeafEvalRequest{state: state, resp: resp}
	select {
	case b.reqCh <- req:
	case <-b.stopCh:
		return 0.5
	}
	return <-resp
}

func (b *mctsLeafBatcher) Stop() {
	close(b.stopCh)
	b.wg.Wait()
}

func (b *mctsLeafBatcher) loop() {
	defer b.wg.Done()
	pending := make([]mctsLeafEvalRequest, 0, b.batchSize)

	flush := func() {
		if len(pending) == 0 {
			return
		}
		states := make([]simulator.BattleState, len(pending))
		for i, req := range pending {
			states[i] = req.state
		}
		var values []float64
		if b.useLatentTokens {
			values = evaluator.EvaluateBatchStatesWithLatentTokens(states, b.latentReasoningToken, b.latentPredictionToken)
		} else {
			values = evaluator.EvaluateBatchStates(states)
		}
		for i, req := range pending {
			req.resp <- values[i]
			close(req.resp)
		}
		pending = pending[:0]
	}

	for {
		if len(pending) == 0 {
			select {
			case req := <-b.reqCh:
				pending = append(pending, req)
			case <-b.stopCh:
				return
			}
		}

		timer := time.NewTimer(b.flushWindow)
		collecting := true
		stopping := false
		for collecting && len(pending) < b.batchSize {
			select {
			case req := <-b.reqCh:
				pending = append(pending, req)
			case <-timer.C:
				collecting = false
			case <-b.stopCh:
				stopping = true
				collecting = false
			}
		}
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}

		flush()
		if stopping {
			for {
				select {
				case req := <-b.reqCh:
					req.resp <- 0.5
					close(req.resp)
				default:
					return
				}
			}
		}
	}
}

func beamLimitForDepth(depth int) int {
	switch {
	case depth >= 5:
		return 3
	case depth >= 3:
		return 4
	case depth >= 2:
		return 6
	default:
		return 10
	}
}

func orderedActionsByEstimate(candidates []orderedAction, maximizing bool, limit int) []int {
	sort.Slice(candidates, func(i, j int) bool {
		if maximizing {
			return candidates[i].estimate > candidates[j].estimate
		}
		return candidates[i].estimate < candidates[j].estimate
	})
	if limit > 0 && len(candidates) > limit {
		candidates = candidates[:limit]
	}
	ordered := make([]int, 0, len(candidates))
	for _, c := range candidates {
		ordered = append(ordered, c.action)
	}
	return ordered
}

func orderRootActions(state *simulator.BattleState, p1Actions [simulator.MaxActions]int, p1Len int, preferredAction int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable, ctx *searchContext) []int {
	candidates := make([]orderedAction, 0, p1Len)
	for i := 0; i < p1Len; i++ {
		p1Action := p1Actions[i]
		if ctx != nil && ctx.shouldStop(i+1) {
			break
		}
		if p1Action == preferredAction {
			candidates = append(candidates, orderedAction{action: p1Action, estimate: math.MaxFloat64})
			continue
		}

		p2Actions, p2Len := simulator.GetSearchActions(&state.P2)
		if p2Len == 0 {
			p2Actions[0] = -1
			p2Len = 1
		}

		worstCase := math.MaxFloat64
		for j := 0; j < p2Len; j++ {
			newState := *state
			simulator.ExecuteSpecificTurn(&newState, p1Action, p2Actions[j])
			estimate := evaluateWithContext(&newState, mlpCache, attentionCache, tt, ctx)
			if estimate < worstCase {
				worstCase = estimate
			}
		}
		candidates = append(candidates, orderedAction{action: p1Action, estimate: worstCase})
	}

	return orderedActionsByEstimate(candidates, true, 0)
}

func orderP2Responses(state simulator.BattleState, depth int, p1Action int, p2Actions [simulator.MaxActions]int, p2Len int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable, ctx *searchContext) []int {
	candidates := make([]orderedAction, 0, p2Len)
	for i := 0; i < p2Len; i++ {
		p2Action := p2Actions[i]
		newState := state
		simulator.ExecuteSpecificTurn(&newState, p1Action, p2Action)
		estimate := evaluateWithContext(&newState, mlpCache, attentionCache, tt, ctx)
		candidates = append(candidates, orderedAction{action: p2Action, estimate: estimate})
	}
	return orderedActionsByEstimate(candidates, false, beamLimitForDepth(depth))
}

func orderP1Responses(state simulator.BattleState, depth int, p2Action int, p1Actions [simulator.MaxActions]int, p1Len int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable, ctx *searchContext) []int {
	candidates := make([]orderedAction, 0, p1Len)
	for i := 0; i < p1Len; i++ {
		p1Action := p1Actions[i]
		newState := state
		simulator.ExecuteSpecificTurn(&newState, p1Action, p2Action)
		estimate := evaluateWithContext(&newState, mlpCache, attentionCache, tt, ctx)
		candidates = append(candidates, orderedAction{action: p1Action, estimate: estimate})
	}
	return orderedActionsByEstimate(candidates, true, beamLimitForDepth(depth))
}

// ActionToString returns a human-readable label for an action index.
func ActionToString(state *simulator.BattleState, player *simulator.PlayerState, action int) string {
	if simulator.IsAttackAction(action) {
		moveIdx := simulator.BaseMoveIndex(action)
		if simulator.IsTeraAction(action) {
			return fmt.Sprintf("move %d [tera]", moveIdx+1)
		}
		return fmt.Sprintf("move %d", moveIdx+1)
	} else if action >= simulator.ActionSwitchBase {
		idx := action - simulator.ActionSwitchBase
		if idx < player.TeamSize {
			return "switch " + player.Team[idx].Species
		}
	}
	return "pass"
}

// AlphaBeta performs Negamax search with Alpha-Beta pruning.
// We pass BattleState BY VALUE (`state simulator.BattleState`) to force it onto the goroutine stack,
// completely eliminating heap allocations for CloneBattleState.
func AlphaBeta(state simulator.BattleState, depth int, alpha float64, beta float64, isP1Turn bool, nodesSearched *int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable, ctx *searchContext) float64 {
	*nodesSearched++
	if ctx != nil && ctx.shouldStop(*nodesSearched) {
		return evaluateWithContext(&state, mlpCache, attentionCache, tt, ctx)
	}

	// Terminal: depth exhausted — use NN evaluation
	if depth == 0 {
		return evaluateWithContext(&state, mlpCache, attentionCache, tt, ctx)
	}

	// Check for terminal game states (one side has no living Pokemon)
	p1Alive := countAlive(&state.P1)
	p2Alive := countAlive(&state.P2)
	if p1Alive == 0 {
		return 0.0 // P1 lost
	}
	if p2Alive == 0 {
		if depth >= 1 {
			// Only log if we found a win within the search tree (not at depth 0)
			// log.Printf("  [Search] Terminal Win Detected at depth %d", depth)
		}
		return 1.0 // P1 won
	}

	if isP1Turn {
		// MAXIMIZING: P1 wants the highest score
		maxEval := -math.MaxFloat64
		p1Actions, p1Len := simulator.GetSearchActions(&state.P1)
		if p1Len == 0 {
			p1Actions[0] = -1 // Pass
			p1Len = 1
		}
		orderedP1Actions := orderP1Responses(state, depth, -1, p1Actions, p1Len, mlpCache, attentionCache, tt, ctx)

		for _, p1Action := range orderedP1Actions {
			// For each P1 action, P2 gets to respond (minimizing layer)
			eval := alphaBetaP2Response(state, depth, alpha, beta, p1Action, nodesSearched, mlpCache, attentionCache, tt, ctx)

			if eval > maxEval {
				maxEval = eval
			}
			if eval > alpha {
				alpha = eval
			}
			if beta <= alpha {
				break // Beta cutoff — P2 would never allow this branch
			}
		}
		return maxEval
	} else {
		// MINIMIZING: P2 wants the lowest score
		minEval := math.MaxFloat64
		p2Actions, p2Len := simulator.GetSearchActions(&state.P2)
		if p2Len == 0 {
			p2Actions[0] = -1 // Pass
			p2Len = 1
		}
		orderedP2Actions := orderP2Responses(state, depth, -1, p2Actions, p2Len, mlpCache, attentionCache, tt, ctx)

		for _, p2Action := range orderedP2Actions {
			eval := alphaBetaP1Response(state, depth, alpha, beta, p2Action, nodesSearched, mlpCache, attentionCache, tt, ctx)

			if eval < minEval {
				minEval = eval
			}
			if eval < beta {
				beta = eval
			}
			if beta <= alpha {
				break // Alpha cutoff — P1 would never allow this branch
			}
		}
		return minEval
	}
}

// alphaBetaP2Response: Given P1's chosen action, search over P2's responses
func alphaBetaP2Response(state simulator.BattleState, depth int, alpha float64, beta float64, p1Action int, nodesSearched *int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable, ctx *searchContext) float64 {
	minEval := math.MaxFloat64
	p2Actions, p2Len := simulator.GetSearchActions(&state.P2)
	if p2Len == 0 {
		p2Actions[0] = -1
		p2Len = 1
	}
	orderedP2Actions := orderP2Responses(state, depth, p1Action, p2Actions, p2Len, mlpCache, attentionCache, tt, ctx)

	for _, p2Action := range orderedP2Actions {
		// Clone state (stack allocated value) and execute the turn
		newState := state // Flat copy on the stack
		simulator.ExecuteSpecificTurn(&newState, p1Action, p2Action)

		eval := AlphaBeta(newState, depth-1, alpha, beta, true, nodesSearched, mlpCache, attentionCache, tt, ctx)

		if eval < minEval {
			minEval = eval
		}
		if eval < beta {
			beta = eval
		}
		if beta <= alpha {
			break // Alpha cutoff
		}
	}
	return minEval
}

// alphaBetaP1Response: Given P2's chosen action, search over P1's responses
func alphaBetaP1Response(state simulator.BattleState, depth int, alpha float64, beta float64, p2Action int, nodesSearched *int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable, ctx *searchContext) float64 {
	maxEval := -math.MaxFloat64
	p1Actions, p1Len := simulator.GetSearchActions(&state.P1)
	if p1Len == 0 {
		p1Actions[0] = -1
		p1Len = 1
	}
	orderedP1Actions := orderP1Responses(state, depth, p2Action, p1Actions, p1Len, mlpCache, attentionCache, tt, ctx)

	for _, p1Action := range orderedP1Actions {
		newState := state // Flat stack copy
		simulator.ExecuteSpecificTurn(&newState, p1Action, p2Action)

		eval := AlphaBeta(newState, depth-1, alpha, beta, false, nodesSearched, mlpCache, attentionCache, tt, ctx)

		if eval > maxEval {
			maxEval = eval
		}
		if eval > alpha {
			alpha = eval
		}
		if beta <= alpha {
			break // Beta cutoff
		}
	}
	return maxEval
}

func defaultMCTSConfig(rolloutDepth int) mctsConfig {
	if rolloutDepth < 1 {
		rolloutDepth = 1
	}

	// Previously rolloutDepth * 4096, which was too heavy for low depths.
	// Now we use a more progressive scale.
	simulations := 256
	if rolloutDepth == 2 {
		simulations = 1024
	} else if rolloutDepth >= 3 {
		simulations = rolloutDepth * 1024
	}
	if v, ok := parsePositiveIntEnv("MCTS_SIMS"); ok {
		simulations = v
	}

	workers := runtime.NumCPU() * 8
	if workers < 8 {
		workers = 8
	}
	if v, ok := parsePositiveIntEnv("MCTS_WORKERS"); ok {
		workers = v
	}

	exploreC := 1.25
	if v, ok := parsePositiveFloatEnv("MCTS_C"); ok {
		exploreC = v
	}

	return mctsConfig{
		rolloutDepth: rolloutDepth,
		simulations:  simulations,
		workers:      workers,
		exploreC:     exploreC,
	}
}

func encodeJointAction(p1Action int, p2Action int) int {
	p2Code := p2Action + 1
	if p2Code < 0 {
		p2Code = 0
	}
	return p1Action*32 + p2Code
}

func terminalValue(state *simulator.BattleState) (float64, bool) {
	p1Alive := countAlive(&state.P1)
	p2Alive := countAlive(&state.P2)
	if p1Alive == 0 && p2Alive == 0 {
		return 0.5, true
	}
	if p1Alive == 0 {
		return 0.0, true
	}
	if p2Alive == 0 {
		return 1.0, true
	}
	return 0, false
}

func ensureNodeActions(node *mctsNode, p1Actions [simulator.MaxActions]int, p1Len int, p2Actions [simulator.MaxActions]int, p2Len int) {
	if node.p1ActionStat != nil {
		return
	}
	node.p1ActionStat = make(map[int]*mctsActionStat, p1Len)
	for i := 0; i < p1Len; i++ {
		node.p1ActionStat[p1Actions[i]] = &mctsActionStat{}
	}
	node.p2ActionStat = make(map[int]*mctsActionStat, p2Len)
	for i := 0; i < p2Len; i++ {
		node.p2ActionStat[p2Actions[i]] = &mctsActionStat{}
	}
	if node.children == nil {
		node.children = make(map[int]*mctsNode)
	}
}

func selectUCTAction(stats map[int]*mctsActionStat, actions [simulator.MaxActions]int, n int, totalVisits int, exploreC float64, rng *rand.Rand, maximize bool) int {
	bestAction := actions[rng.Intn(n)]
	bestScore := -math.MaxFloat64
	if !maximize {
		bestScore = math.MaxFloat64
	}

	logVisits := math.Log(float64(totalVisits + 1))
	for i := 0; i < n; i++ {
		action := actions[i]
		stat := stats[action]
		if stat == nil || (stat.visits+stat.virtualLoss) == 0 {
			return action
		}

		v := stat.visits + stat.virtualLoss
		mean := stat.valueSum / float64(stat.visits) // Note: valueSum is only from real visits
		if !maximize {
			mean = 1.0 - mean
		}

		ucb := mean + exploreC*math.Sqrt(logVisits/float64(v))
		if maximize {
			if ucb > bestScore {
				bestScore = ucb
				bestAction = action
			}
		} else {
			if ucb < bestScore {
				bestScore = ucb
				bestAction = action
			}
		}
	}
	return bestAction
}

func chooseOpponentAction(state *simulator.BattleState, rng *rand.Rand) int {
	p2Actions, p2Len := simulator.GetSearchActions(&state.P2)
	if p2Len == 0 {
		return -1
	}
	return p2Actions[rng.Intn(p2Len)]
}

func runMCTSSimulation(root *mctsNode, rootState simulator.BattleState, cfg mctsConfig, batcher *mctsLeafBatcher, ctx *searchContext, rng *rand.Rand) int {
	node := root
	state := rootState
	path := make([]mctsPathStep, 0, cfg.rolloutDepth)
	steps := 0

	// Exploration phase
	for depth := 0; depth < cfg.rolloutDepth; depth++ {
		if _, terminal := terminalValue(&state); terminal {
			break
		}
		if ctx != nil && ctx.useDeadline && time.Now().After(ctx.deadline) {
			ctx.timedOut = true
			break
		}

		p1Actions, p1Len := simulator.GetSearchActions(&state.P1)
		p2Actions, p2Len := simulator.GetSearchActions(&state.P2)
		if p1Len == 0 {
			p1Actions[0] = -1
			p1Len = 1
		}
		if p2Len == 0 {
			p2Actions[0] = -1
			p2Len = 1
		}

		node.mu.Lock()
		ensureNodeActions(node, p1Actions, p1Len, p2Actions, p2Len)

		totalV := node.visits + node.virtualLoss
		p1Action := selectUCTAction(node.p1ActionStat, p1Actions, p1Len, totalV, cfg.exploreC, rng, true)
		p2Action := selectUCTAction(node.p2ActionStat, p2Actions, p2Len, totalV, cfg.exploreC, rng, false)

		// Apply Virtual Loss
		node.virtualLoss++
		node.p1ActionStat[p1Action].virtualLoss++
		node.p2ActionStat[p2Action].virtualLoss++

		jointKey := encodeJointAction(p1Action, p2Action)
		child := node.children[jointKey]
		if child == nil {
			child = &mctsNode{}
			node.children[jointKey] = child
		}
		path = append(path, mctsPathStep{node: node, p1Hero: p1Action, p2Hero: p2Action})
		node.mu.Unlock()

		next := state
		simulator.ExecuteSpecificTurn(&next, p1Action, p2Action)
		state = next
		node = child
		steps++
	}

	value := batcher.Evaluate(state)
	if v, terminal := terminalValue(&state); terminal {
		value = v
	}

	// Backpropagation phase
	for i := len(path) - 1; i >= 0; i-- {
		step := path[i]
		step.node.mu.Lock()
		step.node.virtualLoss--
		step.node.visits++

		p1Stat := step.node.p1ActionStat[step.p1Hero]
		p1Stat.virtualLoss--
		p1Stat.visits++
		p1Stat.valueSum += value

		p2Stat := step.node.p2ActionStat[step.p2Hero]
		p2Stat.virtualLoss--
		p2Stat.visits++
		p2Stat.valueSum += value // Both track value from P1 perspective

		step.node.mu.Unlock()
	}

	return steps + 1
}

func runMCTSSearch(state *simulator.BattleState, cfg mctsConfig, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable, ctx *searchContext) SearchResult {
	p1Actions, p1Len := simulator.GetSearchActions(&state.P1)
	if p1Len == 0 {
		return SearchResult{BestAction: -1, Score: 0.5, Depth: cfg.rolloutDepth}
	}

	root := &mctsNode{}
	var simsDone int64
	var nodesSearched int64
	var wg sync.WaitGroup
	evalBatchSize := 128
	if v, ok := parsePositiveIntEnv("MCTS_EVAL_BATCH"); ok {
		evalBatchSize = v
	}
	flushWindow := 200 * time.Microsecond
	if v, ok := parsePositiveIntEnv("MCTS_EVAL_FLUSH_US"); ok {
		flushWindow = time.Duration(v) * time.Microsecond
	}
	leafBatcher := newMCTSLeafBatcher(evalBatchSize, flushWindow, ctx)
	defer leafBatcher.Stop()

	for w := 0; w < cfg.workers; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			seed := time.Now().UnixNano() + int64(worker)*7919
			rng := rand.New(rand.NewSource(seed))
			for {
				if ctx != nil && ctx.useDeadline && time.Now().After(ctx.deadline) {
					ctx.timedOut = true
					return
				}
				simID := int(atomic.AddInt64(&simsDone, 1))
				if cfg.simulations > 0 && simID > cfg.simulations {
					return
				}
				nodes := runMCTSSimulation(root, *state, cfg, leafBatcher, ctx, rng)
				atomic.AddInt64(&nodesSearched, int64(nodes))
			}
		}(w)
	}
	wg.Wait()

	baseEval := evaluateWithContext(state, mlpCache, attentionCache, tt, ctx)
	actionScores := make(map[int]float64, p1Len)
	bestAction := p1Actions[0]
	bestScore := -math.MaxFloat64

	root.mu.Lock()
	for i := 0; i < p1Len; i++ {
		action := p1Actions[i]
		score := baseEval
		if root.p1ActionStat != nil {
			if stat := root.p1ActionStat[action]; stat != nil && stat.visits > 0 {
				score = stat.valueSum / float64(stat.visits)
			}
		}
		actionScores[action] = score
		if score > bestScore {
			bestScore = score
			bestAction = action
		}
	}
	root.mu.Unlock()

	if bestScore == -math.MaxFloat64 {
		bestScore = baseEval
	}

	return SearchResult{
		BestAction:    bestAction,
		Score:         bestScore,
		Depth:         cfg.rolloutDepth,
		NodesSearched: int(nodesSearched),
		ActionScores:  actionScores,
		TimedOut:      ctx != nil && ctx.timedOut,
	}
}

// SearchBestMove runs parallel MCTS and returns the best P1 action.
func SearchBestMove(state *simulator.BattleState, depth int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable) SearchResult {
	return SearchBestMoveWithSims(state, depth, 0, mlpCache, attentionCache, tt)
}

func SearchBestMoveWithSims(state *simulator.BattleState, depth int, sims int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable) SearchResult {
	cfg := defaultMCTSConfig(depth)
	if sims > 0 {
		cfg.simulations = sims
	}
	return runMCTSSearch(state, cfg, mlpCache, attentionCache, tt, nil)
}

func searchBestMoveInternal(state *simulator.BattleState, depth int, preferredAction int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable, ctx *searchContext) SearchResult {
	_ = preferredAction
	cfg := defaultMCTSConfig(depth)
	return runMCTSSearch(state, cfg, mlpCache, attentionCache, tt, ctx)
}

// IterativeDeepeningSearch runs time-bounded MCTS within the move time budget.
func IterativeDeepeningSearch(state *simulator.BattleState, maxDuration time.Duration, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable) SearchResult {
	rolloutDepth := 3
	if v, ok := parsePositiveIntEnv("MCTS_ROLLOUT"); ok {
		rolloutDepth = v
	}
	cfg := defaultMCTSConfig(rolloutDepth)
	cfg.simulations = 0 // run until deadline
	ctx := &searchContext{
		deadline:    time.Now().Add(maxDuration),
		useDeadline: true,
	}
	result := runMCTSSearch(state, cfg, mlpCache, attentionCache, tt, ctx)
	actionStr := ActionToString(state, &state.P1, result.BestAction)
	fmt.Printf("  MCTS: Best=%s Score=%.4f Sims=%d\n", actionStr, result.Score, result.NodesSearched)
	return result
}

// SearchEvaluate uses MCTS at a given rollout depth to evaluate who is winning.
// Returns the score from P1's perspective (>0.5 = P1 favored), and the number of nodes searched.
func SearchEvaluate(state *simulator.BattleState, depth int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable) (float64, int) {
	return SearchEvaluateWithSims(state, depth, 0, mlpCache, attentionCache, tt)
}

func SearchEvaluateWithSims(state *simulator.BattleState, depth int, sims int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable) (float64, int) {
	if depth <= 0 {
		score := evaluateWithContext(state, mlpCache, attentionCache, tt, nil)
		return score, 1
	}
	result := SearchBestMoveWithSims(state, depth, sims, mlpCache, attentionCache, tt)
	return result.Score, result.NodesSearched
}

func countAlive(player *simulator.PlayerState) int {
	count := 0
	for i := 0; i < player.TeamSize; i++ {
		if !player.Team[i].Fainted {
			count++
		}
	}
	return count
}

// GetDetailedTags runs MCTS and returns a [MaxActions]float64 array containing the
// Q-values (predicted win probability) for the possible valid actions from this state.
func GetDetailedTags(state *simulator.BattleState, depth int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable) [simulator.MaxActions]float64 {
	return GetDetailedTagsWithLatents(state, depth, mlpCache, attentionCache, tt, evaluator.DefaultLatentReasoningToken, evaluator.DefaultLatentPredictionToken)
}

func GetDetailedTagsWithLatents(state *simulator.BattleState, depth int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable, latentReasoningToken float64, latentPredictionToken float64) [simulator.MaxActions]float64 {
	var tags [simulator.MaxActions]float64

	// Depth <= 0 is a pure-network tagging mode (no MCTS).
	if depth <= 0 {
		return evaluator.EvaluateAllWithLatentTokens(state, mlpCache, attentionCache, latentReasoningToken, latentPredictionToken)
	}

	ctx := &searchContext{
		useLatentTokens:       true,
		latentReasoningToken:  latentReasoningToken,
		latentPredictionToken: latentPredictionToken,
	}

	// Fallback/fill values for invalid actions. We'll pre-fill with the base evaluation.
	baseEval := evaluateWithContext(state, mlpCache, attentionCache, tt, ctx)
	for i := 0; i < simulator.MaxActions; i++ {
		tags[i] = baseEval
	}

	_, p1Len := simulator.GetSearchActions(&state.P1)
	if p1Len == 0 {
		return tags
	}

	cfg := defaultMCTSConfig(depth)
	result := runMCTSSearch(state, cfg, mlpCache, attentionCache, tt, ctx)
	for action, score := range result.ActionScores {
		if action >= 0 && action < simulator.MaxActions {
			tags[action] = score
		}
	}

	return tags
}

// GetDetailedTagsWithBudget runs MCTS tagging with an optional simulation override.
// If simsOverride <= 0, the default simulation budget for the depth is used.
func GetDetailedTagsWithBudget(state *simulator.BattleState, depth int, simsOverride int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable) [simulator.MaxActions]float64 {
	return GetDetailedTagsWithBudgetAndLatents(state, depth, simsOverride, mlpCache, attentionCache, tt, evaluator.DefaultLatentReasoningToken, evaluator.DefaultLatentPredictionToken)
}

func GetDetailedTagsWithBudgetAndLatents(state *simulator.BattleState, depth int, simsOverride int, mlpCache *evaluator.InferenceCache, attentionCache *evaluator.InferenceCache, tt *evaluator.TranspositionTable, latentReasoningToken float64, latentPredictionToken float64) [simulator.MaxActions]float64 {
	var tags [simulator.MaxActions]float64

	if depth <= 0 {
		return evaluator.EvaluateAllWithLatentTokens(state, mlpCache, attentionCache, latentReasoningToken, latentPredictionToken)
	}

	ctx := &searchContext{
		useLatentTokens:       true,
		latentReasoningToken:  latentReasoningToken,
		latentPredictionToken: latentPredictionToken,
	}

	baseEval := evaluateWithContext(state, mlpCache, attentionCache, tt, ctx)
	for i := 0; i < simulator.MaxActions; i++ {
		tags[i] = baseEval
	}

	_, p1Len := simulator.GetSearchActions(&state.P1)
	if p1Len == 0 {
		return tags
	}

	cfg := defaultMCTSConfig(depth)
	if simsOverride > 0 {
		cfg.simulations = simsOverride
	}
	result := runMCTSSearch(state, cfg, mlpCache, attentionCache, tt, ctx)
	for action, score := range result.ActionScores {
		if action >= 0 && action < simulator.MaxActions {
			tags[action] = score
		}
	}
	return tags
}
