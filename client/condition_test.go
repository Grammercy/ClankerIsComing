package client

import "testing"

func TestParseConditionPreservesAbsoluteHP(t *testing.T) {
	hp, maxHP, fainted := ParseCondition("227/227 par")
	if hp != 227 || maxHP != 227 || fainted {
		t.Fatalf("unexpected parse result: hp=%d maxHP=%d fainted=%v", hp, maxHP, fainted)
	}
}

func TestParseConditionFainted(t *testing.T) {
	hp, maxHP, fainted := ParseCondition("0 fnt")
	if hp != 0 || maxHP != 100 || !fainted {
		t.Fatalf("unexpected faint parse result: hp=%d maxHP=%d fainted=%v", hp, maxHP, fainted)
	}
}
