package main

import (
	"strings"
	"testing"
)

// buildSessionLog fabricates a session log in the exact logSession format: the
// last REQUEST carries the full message history (like a real cumulative log).
func buildSessionLog(t *testing.T, lastRequestJSON string, extra ...string) string {
	t.Helper()
	var b strings.Builder
	b.WriteString("=== 2026-07-10T10:00:00+02:00 [llm[0] execute model=m REQUEST] ===\n{\"messages\":[]}\n")
	for _, e := range extra {
		b.WriteString(e)
	}
	b.WriteString("=== 2026-07-10T10:30:00+02:00 [llm[0] execute model=m REQUEST] ===\n")
	b.WriteString(lastRequestJSON)
	b.WriteString("\n")
	return b.String()
}

func TestDigestLogLoopsFailuresAndPairing(t *testing.T) {
	req := `{"messages":[
	 {"role":"user","content":"fix the bug"},
	 {"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"search_text","arguments":"{\"pattern\":\"LoadAll\"}"}}]},
	 {"role":"tool","tool_call_id":"c1","content":"3 matches"},
	 {"role":"assistant","tool_calls":[{"id":"c2","function":{"name":"search_text","arguments":"{\"pattern\":\"LoadAll\"}"}}]},
	 {"role":"tool","tool_call_id":"c2","content":"3 matches"},
	 {"role":"assistant","tool_calls":[{"id":"c3","function":{"name":"run_command","arguments":"{\"command\":\"go build ./...\"}"}}]},
	 {"role":"tool","tool_call_id":"c3","content":"error: undefined symbol foo\nmore lines"},
	 {"role":"user","content":[{"type":"text","text":"we should replan this"}]}
	]}`
	log := buildSessionLog(t, req,
		"=== 2026-07-10T10:10:00+02:00 [RECOVER] ===\nmodel stuck in <think> — retrying with thinking disabled\n",
		"=== 2026-07-10T10:11:00+02:00 [llm[0] execute model=m] ===\n[transport error] connection refused\n")

	d := digestLog("session_x.log", log)
	if d.requests != 2 || d.userTurns != 2 {
		t.Fatalf("requests=%d userTurns=%d, want 2/2", d.requests, d.userTurns)
	}
	if len(d.recovers) != 1 || !strings.Contains(d.recovers[0], "stuck in <think>") {
		t.Fatalf("recovers = %v", d.recovers)
	}
	if len(d.transport) != 1 {
		t.Fatalf("transport = %v", d.transport)
	}
	if d.replans != 1 {
		t.Fatalf("replans = %d, want 1 (multimodal content must be scanned)", d.replans)
	}
	if d.buildRuns != 1 || d.testRuns != 0 {
		t.Fatalf("buildRuns=%d testRuns=%d", d.buildRuns, d.testRuns)
	}

	var rendered strings.Builder
	d.render(&rendered)
	out := rendered.String()
	if !strings.Contains(out, "LOOP: search_text") || !strings.Contains(out, "2× with identical args") {
		t.Fatalf("loop not surfaced:\n%s", out)
	}
	if !strings.Contains(out, "FAIL ×1: run_command") || !strings.Contains(out, "undefined symbol foo") {
		t.Fatalf("failing call not paired via tool_call_id:\n%s", out)
	}
	if !strings.Contains(out, "VERIFY GAP") {
		t.Fatalf("verify gap not flagged:\n%s", out)
	}
}

func TestDigestLogFailureNeedsFirstLineMatch(t *testing.T) {
	// A result that merely MENTIONS "error" deep in its output is not a failure.
	req := `{"messages":[
	 {"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"read_file","arguments":"{\"path\":\"a.go\"}"}}]},
	 {"role":"tool","tool_call_id":"c1","content":"package main\n// handles error cases gracefully"}
	]}`
	d := digestLog("s.log", buildSessionLog(t, req))
	for _, st := range d.calls {
		if st.fails != 0 {
			t.Fatalf("mention of 'error' mid-output flagged as failure: %+v", st)
		}
	}
}

func TestParseLogEntries(t *testing.T) {
	entries := parseLogEntries("=== 2026-07-10T10:00:00Z [WEB] ===\nline1\nline2\n=== 2026-07-10T10:01:00Z [RECOVER] ===\nnudged\n")
	if len(entries) != 2 || entries[0].tag != "WEB" || entries[0].body != "line1\nline2" || entries[1].tag != "RECOVER" {
		t.Fatalf("parseLogEntries = %+v", entries)
	}
}

func TestDigestLogDegradesGracefully(t *testing.T) {
	d := digestLog("s.log", "not a session log at all")
	if len(d.parseNotes) == 0 {
		t.Fatal("garbage input should carry a parse note")
	}
	d = digestLog("s.log", "=== 2026-07-10T10:00:00Z [llm[0] REQUEST] ===\n{malformed json\n")
	if len(d.parseNotes) == 0 {
		t.Fatal("unparseable REQUEST should carry a parse note")
	}
}
