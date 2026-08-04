package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestProbeArms pins the arm derivation: a statement present in the loaded
// skill yields a removal check (WITHOUT = skill minus statement), an absent one
// an addition check (WITH = skill plus statement), an empty file a context-free
// probe, and bad inputs error.
func TestProbeArms(t *testing.T) {
	dir := t.TempDir()
	ch := filepath.Join(dir, ".codehalter")
	if err := os.MkdirAll(ch, 0o755); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(ch, "SKILL-go.md"), []byte("line one\n- DROP ME\nline three\n"), 0o644)

	without, with, mode, err := probeArms(dir, "", probeSpec{File: "SKILL-go.md", Statement: "- DROP ME\n"})
	if err != nil {
		t.Fatalf("removal: %v", err)
	}
	if strings.Contains(without, "DROP ME") || !strings.Contains(with, "DROP ME") || !strings.Contains(mode, "removal") {
		t.Errorf("removal arms wrong: without=%q with=%q mode=%q", without, with, mode)
	}

	without, with, mode, err = probeArms(dir, "", probeSpec{File: "SKILL-go.md", Statement: "- NEW RULE"})
	if err != nil {
		t.Fatalf("addition: %v", err)
	}
	if strings.Contains(without, "NEW RULE") || !strings.Contains(with, "NEW RULE") || !strings.Contains(mode, "addition") {
		t.Errorf("addition arms wrong: without=%q with=%q mode=%q", without, with, mode)
	}

	without, with, _, err = probeArms(dir, "", probeSpec{Statement: "- SOLO"})
	if err != nil {
		t.Fatalf("context-free: %v", err)
	}
	if without != "" || with != "- SOLO" {
		t.Errorf("context-free arms wrong: without=%q with=%q", without, with)
	}

	// The variant copy is the loaded one, so arms derive from it.
	vdir := filepath.Join(ch, "skills", "v1")
	os.MkdirAll(vdir, 0o755)
	os.WriteFile(filepath.Join(vdir, "SKILL-go.md"), []byte("variant body\n- V RULE\n"), 0o644)
	_, with, _, err = probeArms(dir, "v1", probeSpec{File: "SKILL-go.md", Statement: "- V RULE\n"})
	if err != nil {
		t.Fatalf("variant: %v", err)
	}
	if !strings.Contains(with, "variant body") {
		t.Errorf("variant probe should read the variant copy, got %q", with)
	}

	for name, p := range map[string]probeSpec{
		"empty statement": {File: "SKILL-go.md"},
		"not a skill":     {File: "PLAN.md", Statement: "x"},
		"path escape":     {File: "../SKILL-evil.md", Statement: "x"},
		"missing skill":   {File: "SKILL-rust.md", Statement: "x"},
	} {
		if _, _, _, err := probeArms(dir, "", p); err == nil {
			t.Errorf("%s: should error", name)
		}
	}
}

// TestProbeStatementExecute drives one batched probe through the tool against a
// mock main model: two calls (WITHOUT then WITH), both answers verbatim in the
// report, and the WITH arm's system prompt carrying the statement.
func TestProbeStatementExecute(t *testing.T) {
	m := newMockLLM(t, sseText("answer-without"), sseText("answer-with"))
	defer m.Close()

	a, s := newTestAgent(t)
	a.settings = Settings{LLM: []LLMConnection{*m.conn("main")}}
	ch := filepath.Join(s.Cwd, ".codehalter")
	os.MkdirAll(ch, 0o755)
	os.WriteFile(filepath.Join(ch, "SKILL-go.md"), []byte("head\n- PROBED RULE\ntail\n"), 0o644)
	s.improving.Store(true)

	args, _ := json.Marshal(map[string]any{
		"probes":  `[{"file":"SKILL-go.md","statement":"- PROBED RULE\n","question":"What do you do?"}]`,
		"samples": 1,
	})
	var tc toolCall
	tc.Function.Name = probeStatementToolName
	tc.Function.Arguments = string(args)
	out, failed := a.executeTool(context.Background(), s.ID, tc)
	if failed {
		t.Fatalf("failed=true: %s", out)
	}
	if !strings.Contains(out, "answer-without") || !strings.Contains(out, "answer-with") {
		t.Errorf("report must carry both answers verbatim: %s", out)
	}
	if !strings.Contains(out, "removal check") {
		t.Errorf("report must name the mode: %s", out)
	}
	if m.callCount() != 2 {
		t.Fatalf("want 2 main-model calls, got %d", m.callCount())
	}
	sysOf := func(i int) string {
		msgs, _ := m.request(i)["messages"].([]any)
		for _, mm := range msgs {
			if mo, ok := mm.(map[string]any); ok && mo["role"] == "system" {
				str, _ := mo["content"].(string)
				return str
			}
		}
		return ""
	}
	if strings.Contains(sysOf(0), "PROBED RULE") {
		t.Errorf("first call is the WITHOUT arm, its system prompt must lack the statement: %q", sysOf(0))
	}
	if !strings.Contains(sysOf(1), "PROBED RULE") {
		t.Errorf("second call is the WITH arm, its system prompt must carry the statement: %q", sysOf(1))
	}
}

// TestProbeStatementGating pins the guards: outside an /improve run the tool
// refuses (probes burn real main-model calls), and malformed args error without
// any LLM traffic.
func TestProbeStatementGating(t *testing.T) {
	a, s := newTestAgent(t)
	a.settings = Settings{LLM: []LLMConnection{{Server: "http://127.0.0.1:0", Model: "m"}}}
	call := func(argsMap map[string]any) (string, bool) {
		b, _ := json.Marshal(argsMap)
		var tc toolCall
		tc.Function.Name = probeStatementToolName
		tc.Function.Arguments = string(b)
		return a.executeTool(context.Background(), s.ID, tc)
	}

	if out, failed := call(map[string]any{"probes": `[{"statement":"x","question":"q"}]`}); !failed || !strings.Contains(out, "only available during an /improve run") {
		t.Errorf("non-improve call must refuse: failed=%v %s", failed, out)
	}
	s.improving.Store(true)
	if out, failed := call(map[string]any{"probes": ""}); !failed || !strings.Contains(out, "probes is required") {
		t.Errorf("empty probes: failed=%v %s", failed, out)
	}
	if out, failed := call(map[string]any{"probes": "not json"}); !failed || !strings.Contains(out, "invalid probes JSON") {
		t.Errorf("bad json: failed=%v %s", failed, out)
	}
	if out, failed := call(map[string]any{"probes": "[]"}); !failed || !strings.Contains(out, "nothing to test") {
		t.Errorf("empty array: failed=%v %s", failed, out)
	}
}
