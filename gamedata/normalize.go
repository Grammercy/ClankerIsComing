package gamedata

import (
	"regexp"
	"strings"
)

var (
	idStripper = regexp.MustCompile(`[^a-z0-9]+`)
	idReplacer = strings.NewReplacer(
		"♀", "f",
		"♂", "m",
		"é", "e",
		"É", "e",
		"’", "",
		"'", "",
		"`", "",
		".", "",
		":", "",
		"%", "",
	)
)

// NormalizeID mirrors Showdown-style ID normalization (toID) with a few
// battle-relevant Unicode replacements for species names and move names.
func NormalizeID(text string) string {
	s := strings.TrimSpace(text)
	if s == "" {
		return ""
	}
	s = idReplacer.Replace(strings.ToLower(s))
	return idStripper.ReplaceAllString(s, "")
}
