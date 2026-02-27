package evaluator

import (
	"fmt"

	"github.com/pokemon-engine/simulator"
)

const TaggedReplayVersion = 1

type TaggedSample struct {
	Prefix    []float64 `json:"prefix"`
	RawSlots  []float64 `json:"rawSlots"`
	Targets   []float64 `json:"targets"`
	EloWeight float64   `json:"eloWeight"`
	Turn      int       `json:"turn,omitempty"`
	EventIdx  int       `json:"eventIdx,omitempty"`
}

type TaggedReplayDataset struct {
	Version      int            `json:"version"`
	SourceReplay string         `json:"sourceReplay"`
	Depth        int            `json:"depth"`
	Samples      []TaggedSample `json:"samples"`
}

func BuildTaggedSampleFromState(state *simulator.BattleState, targets []float64, eloWeight float64) (TaggedSample, string, bool) {
	if len(targets) != simulator.MaxActions {
		return TaggedSample{}, "target_length_mismatch", false
	}

	hasLabel := false
	for _, t := range targets {
		if t >= 0.0 {
			hasLabel = true
			break
		}
	}
	if !hasLabel {
		return TaggedSample{}, "no_valid_targets", false
	}

	p1Globals, p1Slots := vectorizePlayerFeatures(&state.P1, state)
	p2Globals, p2Slots := vectorizePlayerFeatures(&state.P2, state)

	rawSlots := make([]float64, 0, TotalSlotFeatures)
	rawSlots = append(rawSlots, p1Slots[:]...)
	rawSlots = append(rawSlots, p2Slots[:]...)

	var prefixArr [TotalFeatures]float64
	idx := 0
	prefixArr[idx] = p1Globals[0]
	idx++
	prefixArr[idx] = p1Globals[1]
	idx++
	prefixArr[idx] = p2Globals[0]
	idx++
	prefixArr[idx] = p2Globals[1]
	idx++
	vectorizeFieldConditions(&state.Field, &prefixArr, &idx)
	vectorizeSideConditions(&state.P1.Side, &prefixArr, &idx)
	vectorizeSideConditions(&state.P2.Side, &prefixArr, &idx)
	vectorizeBoosts(state.P1.GetActive(), &prefixArr, &idx)
	vectorizeBoosts(state.P2.GetActive(), &prefixArr, &idx)
	vectorizeMatchup(state, &prefixArr, &idx)

	return TaggedSample{
		Prefix:    append([]float64(nil), prefixArr[:TotalGlobals]...),
		RawSlots:  rawSlots,
		Targets:   append([]float64(nil), targets...),
		EloWeight: eloWeight,
	}, "", true
}

func taggedSampleToPrepared(sample TaggedSample) (preparedSnapshot, error) {
	if len(sample.Prefix) != TotalGlobals {
		return preparedSnapshot{}, fmt.Errorf("invalid prefix length %d (expected %d)", len(sample.Prefix), TotalGlobals)
	}
	if len(sample.RawSlots) != TotalSlotFeatures {
		return preparedSnapshot{}, fmt.Errorf("invalid rawSlots length %d (expected %d)", len(sample.RawSlots), TotalSlotFeatures)
	}
	if len(sample.Targets) != simulator.MaxActions {
		return preparedSnapshot{}, fmt.Errorf("invalid targets length %d (expected %d)", len(sample.Targets), simulator.MaxActions)
	}
	return preparedSnapshot{
		prefix:    append([]float64(nil), sample.Prefix...),
		rawSlots:  append([]float64(nil), sample.RawSlots...),
		targets:   append([]float64(nil), sample.Targets...),
		eloWeight: sample.EloWeight,
	}, nil
}
