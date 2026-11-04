package main

import (
	"strings"
	"testing"
)

// Not 2>&1, /dev/null, variables or quoted names.
func TestRedirectTargets(t *testing.T) {
	got := redirectTargets(`cd rust && (just test > /tmp/a.log 2>&1; echo "exit=$?" >> /tmp/a.log); cargo check 2> err.txt &> all.txt; x > $OUT; y > "q q"; z > /dev/null`, "/w")
	want := []string{"/tmp/a.log", "/tmp/a.log", "/w/err.txt", "/w/all.txt"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("targets = %v, want %v", got, want)
	}
}
