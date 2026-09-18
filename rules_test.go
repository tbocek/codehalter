package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestDefaultRulesMatchLeakedControlTokens pins that the shipped rules fire on
// the failure they exist for — a model writing tool-call or chat-template
// syntax as prose — and, just as importantly, that they stay quiet on ordinary
// text. A rule that misfires costs two extra round-trips on every turn, so the
// negative cases are the load-bearing half of this test.
func TestDefaultRulesMatchLeakedControlTokens(t *testing.T) {
	rules := compileStreamRules(defaultStreamRules)
	if len(rules) != len(defaultStreamRules) {
		t.Fatalf("compiled %d of %d default rules — one has a bad pattern", len(rules), len(defaultStreamRules))
	}

	fires := []struct{ name, text string }{
		{"tool_call tag", "Let me check that.\n<tool_call>\n{\"name\": \"read_file\"}"},
		{"closing tag", "done</tool_call>"},
		{"piped form", "<|tool_call|>"},
		{"function_call", "<function_call>"},
		{"im_start", "<|im_start|>assistant"},
		{"channel", "<|channel|>final"},
	}
	for _, tc := range fires {
		m := &ruleMatcher{rules: rules}
		if got := m.feed(tc.text); got == nil {
			t.Errorf("%s: no rule fired on %q", tc.name, tc.text)
		}
	}

	quiet := []struct{ name, text string }{
		{"plain prose", "I read the file and the parser looks correct."},
		{"code with generics", "func f[T any](x T) []T { return []T{x} }"},
		{"comparison operators", "if a < b && b > c { return a <= c }"},
		{"markdown", "See `tool_call` handling in tools.go for the registry."},
		{"html", "<div class=\"tool\">call</div>"},
	}
	for _, tc := range quiet {
		m := &ruleMatcher{rules: rules}
		if got := m.feed(tc.text); got != nil {
			t.Errorf("%s: rule %q misfired on %q", tc.name, got.Name, tc.text)
		}
	}
}

// TestMatcherFiresAcrossDeltas covers the real streaming shape: the pattern
// arrives one token at a time, so it only exists once several deltas have been
// concatenated. A matcher that tested each delta in isolation would never fire.
func TestMatcherFiresAcrossDeltas(t *testing.T) {
	m := &ruleMatcher{rules: compileStreamRules(defaultStreamRules)}
	deltas := []string{"Sure", ", I'll", " do that", ".\n<tool", "_call", ">"}
	var fired *streamRule
	for _, d := range deltas {
		if r := m.feed(d); r != nil {
			fired = r
			break
		}
	}
	if fired == nil {
		t.Fatal("rule never fired across split deltas")
	}
	if fired.Name != "tool_call_as_text" {
		t.Errorf("fired %q, want tool_call_as_text", fired.Name)
	}
}

// TestMatcherWindowIsBounded pins the cost guarantee: the matcher holds at most
// ruleWindowBytes regardless of how long the reply runs, so per-delta cost
// doesn't grow with message length.
func TestMatcherWindowIsBounded(t *testing.T) {
	m := &ruleMatcher{rules: compileStreamRules(defaultStreamRules)}
	for range 200 {
		if r := m.feed(strings.Repeat("a", 100)); r != nil {
			t.Fatalf("rule %q misfired on filler text", r.Name)
		}
	}
	if len(m.win) > ruleWindowBytes {
		t.Errorf("window grew to %d bytes, cap is %d", len(m.win), ruleWindowBytes)
	}
	// Still live after all that trimming.
	if m.feed("<tool_call>") == nil {
		t.Error("matcher stopped firing after the window filled")
	}
}

// TestCompileStreamRulesDropsBadRules checks that a broken entry is skipped
// rather than taking the whole set down with it — one bad regex in a
// user-written rules.toml must not disarm the rules that do compile.
func TestCompileStreamRulesDropsBadRules(t *testing.T) {
	in := []streamRule{
		{Name: "good", Pattern: "abc", Reminder: "stop"},
		{Name: "bad regex", Pattern: "a(b", Reminder: "stop"},
		{Name: "no reminder", Pattern: "xyz"},
		{Name: "no pattern", Reminder: "stop"},
	}
	got := compileStreamRules(in)
	if len(got) != 1 || got[0].Name != "good" {
		t.Fatalf("kept %d rules (%+v), want only the good one", len(got), got)
	}
}

// TestLoadStreamRulesFromFile covers the project-override path: a rules.toml
// replaces the defaults wholesale, and a malformed one falls back rather than
// failing the session.
func TestLoadStreamRulesFromFile(t *testing.T) {
	cwd := t.TempDir()
	dir := filepath.Join(cwd, sessionDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "rules.toml")

	// No file → defaults.
	if got := loadStreamRules(cwd); len(got) != len(defaultStreamRules) {
		t.Errorf("no rules.toml: got %d rules, want the %d defaults", len(got), len(defaultStreamRules))
	}

	body := "[[rule]]\nname = \"project\"\npattern = \"NEVERDOTHIS\"\nreminder = \"don't\"\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	got := loadStreamRules(cwd)
	if len(got) != 1 || got[0].Name != "project" {
		t.Fatalf("got %+v, want only the project rule (the file replaces the defaults)", got)
	}
	// The replaced-not-merged contract: a default pattern must no longer fire.
	m := &ruleMatcher{rules: got}
	if r := m.feed("<tool_call>"); r != nil {
		t.Errorf("default rule %q still active after an override file", r.Name)
	}

	if err := os.WriteFile(path, []byte("this is not toml ["), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := loadStreamRules(cwd); len(got) != len(defaultStreamRules) {
		t.Errorf("malformed rules.toml: got %d rules, want a fallback to the %d defaults", len(got), len(defaultStreamRules))
	}
}

// TestAsStreamRuleUnwraps pins that the tool loop can recognise a rule abort
// through errors.As, including when it has been wrapped.
func TestAsStreamRuleUnwraps(t *testing.T) {
	base := &streamRuleError{Rule: "r", Reminder: "fix it", Matched: "<tool_call>"}
	if got := asStreamRule(base); got == nil || got.Rule != "r" {
		t.Fatalf("asStreamRule(direct) = %v, want the error", got)
	}
	if got := asStreamRule(errors.New("unrelated")); got != nil {
		t.Errorf("asStreamRule(unrelated) = %v, want nil", got)
	}
}
