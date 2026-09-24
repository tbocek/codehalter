package main

import (
	"strings"
	"testing"
)

// TestRedirectTargets pins what the watchdog watches: files after >, >> and
// &>, an fd-prefixed 2>, relative paths under cwd; not 2>&1, /dev/null,
// variables or quoted names, and nothing for a command that is not bash -c.
func TestRedirectTargets(t *testing.T) {
	got := redirectTargets("bash", []string{"-c", `cd rust && (just test > /tmp/a.log 2>&1; echo "exit=$?" >> /tmp/a.log); cargo check 2> err.txt &> all.txt; x > $OUT; y > "q q"; z > /dev/null`}, "/w")
	want := []string{"/tmp/a.log", "/tmp/a.log", "/w/err.txt", "/w/all.txt"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("targets = %v, want %v", got, want)
	}
	if got := redirectTargets("just", []string{"test"}, "/w"); got != nil {
		t.Errorf("non-shell command watched %v", got)
	}
}
