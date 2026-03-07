package client

import (
	"encoding/json"
	"testing"
)

func TestShowdownActiveUnmarshalCanTerastallizeBool(t *testing.T) {
	var got ShowdownActive
	err := json.Unmarshal([]byte(`{"moves":[],"canTerastallize":true}`), &got)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !got.CanTerastallize {
		t.Fatalf("expected CanTerastallize=true")
	}
}

func TestShowdownActiveUnmarshalCanTerastallizeString(t *testing.T) {
	var got ShowdownActive
	err := json.Unmarshal([]byte(`{"moves":[],"canTerastallize":"Ghost"}`), &got)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !got.CanTerastallize {
		t.Fatalf("expected CanTerastallize=true for non-empty string")
	}
}

func TestShowdownActiveUnmarshalCanTerastallizeStringFalse(t *testing.T) {
	var got ShowdownActive
	err := json.Unmarshal([]byte(`{"moves":[],"canTerastallize":"false"}`), &got)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.CanTerastallize {
		t.Fatalf("expected CanTerastallize=false for string false")
	}
}
