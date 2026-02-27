package gamedata

import (
	"testing"
)

func BenchmarkTypeOneHot(b *testing.B) {
	types := []string{"Fire", "Flying"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = TypeOneHot(types)
	}
}
