package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestConfirmPlanFixAutoExecSkipsGate pins that a user-accepted fix dispatch
// (fixAutoExec set) skips confirmPlan's "Execute?" gate. Every fix card —
// missing-tools/npm, lsmcp, mcp errors — routes through proposeFix, which sets
// the flag, so this one check covers them all. With a nil conn, reaching the
// gate would call conn.AskChoice and panic; returning nil proves it was skipped.
func TestConfirmPlanFixAutoExecSkipsGate(t *testing.T) {
	a, s := newTestAgent(t)
	plan := &planResult{Subtasks: []subtask{{Description: "install npm + prettier"}}}

	s.fixAutoExec = true
	if err := a.confirmPlan(context.Background(), s.ID, plan, false); err != nil {
		t.Errorf("fixAutoExec set: confirmPlan should skip the gate and return nil, got %v", err)
	}
}

// TestSubagentStatusFolding pins the subagent-meter fold: a subagent's live
// status (which Zed would otherwise drop with its sub_* sid) is recorded under
// its parent and rendered as a compact, labelled, sorted join on the parent's
// row — so N parallel subagents each show ↑sent/↓tokens/elapsed instead of
// clobbering one line. The meter's outer "( )" framing and redundant "llm[slot]"
// prefix are stripped (the label already names the slot); an empty suffix clears
// that subagent's fragment and prunes the parent once none remain.
func TestSubagentStatusFolding(t *testing.T) {
	a, _ := newTestAgent(t)
	ctx := context.Background()
	const parent = "parent-1"

	got := a.setSubagentStatus(ctx, parent, "sub_a", "llm1@0/0", " (llm[0] ↑12.00kb ↓1.2k…)")
	if want := " (llm1@0/0 ↑12.00kb ↓1.2k…)"; got != want {
		t.Errorf("single sub: got %q, want %q", got, want)
	}

	// A second parallel subagent: both fragments, sorted by label, joined " · ".
	// The non-meter "(running …)" suffix has no llm[slot] prefix to strip.
	got = a.setSubagentStatus(ctx, parent, "sub_b", "llm2@1/0", " (running read_file…)")
	if want := " (llm1@0/0 ↑12.00kb ↓1.2k… · llm2@1/0 running read_file…)"; got != want {
		t.Errorf("two subs: got %q, want %q", got, want)
	}

	// Clearing the first subagent drops only its fragment.
	got = a.setSubagentStatus(ctx, parent, "sub_a", "llm1@0/0", "")
	if want := " (llm2@1/0 running read_file…)"; got != want {
		t.Errorf("after clear: got %q, want %q", got, want)
	}

	// Clearing the last empties the row and prunes the parent entry.
	if got = a.setSubagentStatus(ctx, parent, "sub_b", "llm2@1/0", ""); got != "" {
		t.Errorf("after final clear: got %q, want empty", got)
	}
	a.subagentMeterMu.Lock()
	_, present := a.subagentMeter[parent]
	a.subagentMeterMu.Unlock()
	if present {
		t.Error("parent entry should be pruned once no subagents remain")
	}
}

// TestParseLineRange covers the line-range encodings Zed may put in a
// resource_link fragment.
func TestParseLineRange(t *testing.T) {
	cases := []struct {
		frag             string
		wantStart, wantE int
	}{
		{"L810-845", 810, 845},
		{"810:845", 810, 845},
		{"810-845", 810, 845},
		{"L810", 810, 810},
		{"", 0, 0},
		{"nodigits", 0, 0},
	}
	for _, c := range cases {
		if s, e := parseLineRange(c.frag); s != c.wantStart || e != c.wantE {
			t.Errorf("parseLineRange(%q) = %d,%d, want %d,%d", c.frag, s, e, c.wantStart, c.wantE)
		}
	}
}

// TestReadLinkedResource verifies a resource_link is inlined with its line
// range, full-file when no range, and refused outside the workspace.
func TestReadLinkedResource(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.go")
	if err := os.WriteFile(path, []byte("a\nb\nc\nd\ne\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Line range 2-4 → "b\nc\nd".
	if snip, label, ok := readLinkedResource(dir, "file://"+path+"#L2-4"); !ok || snip != "b\nc\nd" || label != "f.go:2-4" {
		t.Errorf("range read = %q,%q,%v", snip, label, ok)
	}
	// No range → whole file, label is basename.
	if snip, label, ok := readLinkedResource(dir, "file://"+path); !ok || snip != "a\nb\nc\nd\ne\n" || label != "f.go" {
		t.Errorf("full read = %q,%q,%v", snip, label, ok)
	}
	// Outside cwd → refused.
	if _, _, ok := readLinkedResource(dir, "file:///etc/passwd"); ok {
		t.Errorf("expected refusal for path outside workspace")
	}
	// Missing file → refused.
	if _, _, ok := readLinkedResource(dir, "file://"+filepath.Join(dir, "nope.go")); ok {
		t.Errorf("expected refusal for missing file")
	}
}

// TestResourcePath pins the URI → read_file path mapping used when an embedded
// "resource" / "resource_link" block is folded into the prompt: file:// URIs
// collapse to their percent-decoded path with the fragment stripped, non-file
// URIs pass through verbatim.
func TestResourcePath(t *testing.T) {
	cases := map[string]string{
		"file:///workspaces/codehalter/llm.go":          "/workspaces/codehalter/llm.go",
		"file:///workspaces/codehalter/llm.go#L801-836": "/workspaces/codehalter/llm.go",
		"file:///a%20b/c.go":                            "/a b/c.go",
		"/plain/path.go":                                "/plain/path.go",
		"https://example.com/x":                         "https://example.com/x",
		"":                                              "",
	}
	for uri, want := range cases {
		if got := resourcePath(uri); got != want {
			t.Errorf("resourcePath(%q) = %q, want %q", uri, got, want)
		}
	}
}

// TestResourceLabel checks the human-readable header: the path plus the URI
// fragment (an editor line range) in parentheses when present.
func TestResourceLabel(t *testing.T) {
	cases := map[string]string{
		"file:///workspaces/codehalter/llm.go":          "/workspaces/codehalter/llm.go",
		"file:///workspaces/codehalter/llm.go#L801-836": "/workspaces/codehalter/llm.go (L801-836)",
		"": "attachment",
	}
	for uri, want := range cases {
		if got := resourceLabel(uri); got != want {
			t.Errorf("resourceLabel(%q) = %q, want %q", uri, got, want)
		}
	}
}

func TestHumanFormatters(t *testing.T) {
	for _, c := range []struct {
		n    int
		want string
	}{
		{0, "0"}, {543, "543"}, {1234, "1.2k"}, {12000, "12k"},
		{543490, "543k"}, {1_500_000, "1.5m"}, {2_000_000_000, "2g"},
	} {
		if got := humanCount(c.n); got != c.want {
			t.Errorf("humanCount(%d)=%q want %q", c.n, got, c.want)
		}
	}
	for _, c := range []struct {
		ms   int64
		want string
	}{
		{5500, "5.5s"}, {61000, "1m1s"}, {3661000, "1h1m1s"}, {120000, "2m0s"},
	} {
		if got := humanDuration(c.ms); got != c.want {
			t.Errorf("humanDuration(%d)=%q want %q", c.ms, got, c.want)
		}
	}
	// 200 tokens in 500ms = 400/s
	if got := humanRate(200, 500); got != "400" {
		t.Errorf("humanRate(200,500)=%q want 400", got)
	}
}

// TestSystemPromptCarriesPhaseGuidance pins that PLAN.md/EXECUTE.md live in the
// system prompt (the stable, cached prefix) instead of being re-injected as a
// per-turn user message — the fix for the primer bloat that stacked 7-8 KB
// copies in the history each (re)plan and forced repeated compactions.
func TestSystemPromptCarriesPhaseGuidance(t *testing.T) {
	a, s := newTestAgent(t)
	dir := filepath.Join(s.Cwd, ".codehalter")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{
		"PLAN.md":    "PLAN_SENTINEL planning guidance",
		"EXECUTE.md": "EXEC_SENTINEL execution guidance",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	sp, err := a.systemPrompt(s.ID)
	if err != nil {
		t.Fatalf("systemPrompt: %v", err)
	}
	for _, want := range []string{"PLAN_SENTINEL", "EXEC_SENTINEL"} {
		if !strings.Contains(sp, want) {
			t.Errorf("system prompt missing %q — phase guidance not carried in the prefix", want)
		}
	}
}

// TestLoadAgentsFile pins the AGENTS.md scan: project-root only, priority order
// (AGENTS.md first), trimmed, whitespace-only skipped.
func TestLoadAgentsFile(t *testing.T) {
	dir := t.TempDir()
	if name, content := loadAgentsFile(dir); name != "" || content != "" {
		t.Errorf("no agents file → empty, got %q / %q", name, content)
	}

	if err := os.WriteFile(filepath.Join(dir, "agent.md"), []byte("\n  be concise  \n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if name, content := loadAgentsFile(dir); name != "agent.md" || content != "be concise" {
		t.Errorf("agent.md should be found+trimmed, got %q / %q", name, content)
	}

	// AGENTS.md outranks the lowercase variant.
	if err := os.WriteFile(filepath.Join(dir, "AGENTS.md"), []byte("canonical conventions"), 0o644); err != nil {
		t.Fatal(err)
	}
	if name, content := loadAgentsFile(dir); name != "AGENTS.md" || content != "canonical conventions" {
		t.Errorf("AGENTS.md should win priority, got %q / %q", name, content)
	}

	empty := t.TempDir()
	if err := os.WriteFile(filepath.Join(empty, "AGENTS.md"), []byte("   \n\t\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if name, _ := loadAgentsFile(empty); name != "" {
		t.Errorf("whitespace-only AGENTS.md should be skipped, got %q", name)
	}
}

// TestSystemPromptIncludesAgentsFile verifies the root AGENTS.md is folded into
// the system prompt (the cached prefix) at build time.
func TestSystemPromptIncludesAgentsFile(t *testing.T) {
	a, s := newTestAgent(t)
	if err := os.WriteFile(filepath.Join(s.Cwd, "AGENTS.md"), []byte("Always use tabs."), 0o644); err != nil {
		t.Fatal(err)
	}
	sp, err := a.systemPrompt(s.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(sp, "## Project instructions (AGENTS.md)") || !strings.Contains(sp, "Always use tabs.") {
		t.Errorf("system prompt should fold in AGENTS.md:\n%s", sp)
	}
}
