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

// TestProbeStatementRewriteCheck pins the `replaces` mode used by Step 3b's
// validation round: the superseded statement is stripped from BOTH arms, so the
// WITH arm measures the rewrite alone instead of rewrite+original side by side.
func TestProbeStatementRewriteCheck(t *testing.T) {
	m := newMockLLM(t, sseText("ans-without"), sseText("ans-with"))
	defer m.Close()

	a, s := newTestAgent(t)
	a.settings = Settings{LLM: []LLMConnection{*m.conn("main")}}
	ch := filepath.Join(s.Cwd, ".codehalter")
	os.MkdirAll(ch, 0o755)
	os.WriteFile(filepath.Join(ch, "SKILL-go.md"), []byte("head\n- OLD RULE\ntail\n"), 0o644)
	s.improving.Store(true)

	args, _ := json.Marshal(map[string]any{
		"probes":  `[{"file":"SKILL-go.md","statement":"- NEW RULE","replaces":"- OLD RULE","question":"What do you do?"}]`,
		"samples": 1,
	})
	var tc toolCall
	tc.Function.Name = probeStatementToolName
	tc.Function.Arguments = string(args)
	out, failed := a.executeTool(context.Background(), s.ID, tc)
	if failed {
		t.Fatalf("failed=true: %s", out)
	}
	if !strings.Contains(out, "rewrite check") {
		t.Errorf("report must name the rewrite mode: %s", out)
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
	if strings.Contains(sysOf(0), "OLD RULE") || strings.Contains(sysOf(0), "NEW RULE") {
		t.Errorf("WITHOUT arm must strip the replaced statement and lack the rewrite: %q", sysOf(0))
	}
	if !strings.Contains(sysOf(1), "NEW RULE") || strings.Contains(sysOf(1), "OLD RULE") {
		t.Errorf("WITH arm must carry the rewrite alone (original stripped): %q", sysOf(1))
	}

	// A `replaces` that isn't in the file byte-exactly must error, not probe.
	args, _ = json.Marshal(map[string]any{
		"probes": `[{"file":"SKILL-go.md","statement":"- NEW RULE","replaces":"- NOT PRESENT","question":"q"}]`,
	})
	tc.Function.Arguments = string(args)
	if out, _ := a.executeTool(context.Background(), s.ID, tc); !strings.Contains(out, "`replaces` text not found") {
		t.Errorf("missing replaces text must be reported: %s", out)
	}
}

// TestProbeArmCacheAndRecord pins the run-scoped arm cache (a repeated probe
// costs zero extra main-model calls; a higher-sample re-probe pays only the
// shortfall) and the durable evidence log: every probe lands as a line in
// .codehalter/improve-probes.jsonl with its answers.
func TestProbeArmCacheAndRecord(t *testing.T) {
	m := newMockLLM(t, sseText("w1"), sseText("v1"), sseText("w2"), sseText("v2"))
	defer m.Close()

	a, s := newTestAgent(t)
	a.settings = Settings{LLM: []LLMConnection{*m.conn("main")}}
	ch := filepath.Join(s.Cwd, ".codehalter")
	os.MkdirAll(ch, 0o755)
	os.WriteFile(filepath.Join(ch, "SKILL-go.md"), []byte("head\n- PROBED RULE\ntail\n"), 0o644)
	s.improving.Store(true)

	run := func(samples int) string {
		args, _ := json.Marshal(map[string]any{
			"probes":  `[{"file":"SKILL-go.md","statement":"- PROBED RULE","question":"What do you do?"}]`,
			"samples": samples,
		})
		var tc toolCall
		tc.Function.Name = probeStatementToolName
		tc.Function.Arguments = string(args)
		out, failed := a.executeTool(context.Background(), s.ID, tc)
		if failed {
			t.Fatalf("failed=true: %s", out)
		}
		return out
	}

	run(1)
	if m.callCount() != 2 {
		t.Fatalf("first probe: want 2 main-model calls, got %d", m.callCount())
	}
	// Identical probe again: both arms served from the cache, zero new calls.
	if out := run(1); m.callCount() != 2 {
		t.Fatalf("repeated probe must reuse cached arms, got %d calls: %s", m.callCount(), out)
	}
	// samples=2 on the same arms: only the shortfall (1 per arm) is generated.
	out := run(2)
	if m.callCount() != 4 {
		t.Fatalf("shortfall generation: want 4 calls total, got %d", m.callCount())
	}
	for _, want := range []string{"w1", "v1", "w2", "v2"} {
		if !strings.Contains(out, want) {
			t.Errorf("samples=2 report must carry cached + fresh answers, missing %q: %s", want, out)
		}
	}

	// Durable record: three probes → three jsonl lines, answers included.
	data, err := os.ReadFile(filepath.Join(ch, "improve-probes.jsonl"))
	if err != nil {
		t.Fatalf("improve-probes.jsonl not written: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 3 {
		t.Fatalf("want 3 probe records, got %d:\n%s", len(lines), data)
	}
	var rec probeRecord
	if err := json.Unmarshal([]byte(lines[0]), &rec); err != nil {
		t.Fatalf("record does not parse: %v", err)
	}
	if rec.Statement != "- PROBED RULE" || rec.Question != "What do you do?" ||
		len(rec.Without) != 1 || rec.Without[0] != "w1" || len(rec.With) != 1 || rec.With[0] != "v1" || rec.Time == "" {
		t.Errorf("record fields wrong: %+v", rec)
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
