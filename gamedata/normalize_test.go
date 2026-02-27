package gamedata

import "testing"

func TestNormalizeIDSpecialCases(t *testing.T) {
	cases := map[string]string{
		"Nidoran♀":    "nidoranf",
		"Nidoran♂":    "nidoranm",
		"Flabébé":     "flabebe",
		"Farfetch’d":  "farfetchd",
		"Will-O-Wisp": "willowisp",
	}
	for in, want := range cases {
		if got := NormalizeID(in); got != want {
			t.Fatalf("NormalizeID(%q) = %q, want %q", in, got, want)
		}
	}
}
