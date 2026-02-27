package simulator

import "testing"

func TestMapVolatileToBit(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected uint32
	}{
		{"exact match", "substitute", VolatileSubstitute},
		{"uppercase match", "Confusion", VolatileConfusion},
		{"mixed case and space", "Leech Seed", VolatileLeechSeed},
		{"special character and space", "Perish Song", VolatilePerishSong},
		{"multiple valid case 1", "protect", VolatileProtection},
		{"multiple valid case 2", "Spiky Shield", VolatileProtection},
		{"multiple valid case 3", "King's Shield", VolatileProtection},
		{"invalid match", "Not A Real Volatile", 0},
		{"empty string", "", 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MapVolatileToBit(tt.input); got != tt.expected {
				t.Errorf("MapVolatileToBit(%q) = %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}
