package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// The shapes are the ones Qwen3.8 flash sent for tools that do not exist here.
func TestCanonicalCall(t *testing.T) {
	for _, c := range []struct{ name, args, want string }{
		{"read", `{"paths": {"inv/inv.go": [1, 80]}}`, "sed -n '1,80p' inv/inv.go"},
		{"read", `{"limit": 80, "offset": 1, "path": "inv/inv.go"}`, "sed -n '1,80p' inv/inv.go"},
		{"read", `{"limit": 80, "start": 1, "target_file": "inv/inv.go"}`, "sed -n '1,80p' inv/inv.go"},
		{"read", `{"limit": 80, "start_line": 1, "path": "inv/inv.go"}`, "sed -n '1,80p' inv/inv.go"},
		{"read", `{"limit": 80, "path": "inv/inv.go"}`, "sed -n '1,80p' inv/inv.go"},
		{"read", `{"limit": 120, "path": "inv/inv.go", "range": [{"end": 80, "start": 1}]}`, "sed -n '1,120p' inv/inv.go"},
		{"read", `{"arguments": {"path": "inv/inv.go"}, "tool_call_id": "toolu_x"}`, "cat inv/inv.go"},
		{"read_file", `{"path": "inv/inv.go"}`, "cat inv/inv.go"},
		{"read_file", `{"file_path": "/w/my file.go", "offset": 30}`, "sed -n '30,$p' '/w/my file.go'"},
		{"read", `{"paths": ["a.go", "b.go"]}`, "echo '== a.go'; cat a.go; echo '== b.go'; cat b.go"},
		{"read_command", `{"command": "sed -n '1,90p' inv/inv.go"}`, "sed -n '1,90p' inv/inv.go"},
		{"exec", `{"command": "ls -R && cat go.mod"}`, "ls -R && cat go.mod"},
		{"sed", `{"path": "inv/inv_test.go"}`, "cat inv/inv_test.go"},
	} {
		var tc toolCall
		tc.Function.Name, tc.Function.Arguments = c.name, c.args
		got, ok := canonicalCall(tc)
		var a map[string]string
		_ = json.Unmarshal([]byte(got.Function.Arguments), &a)
		if !ok || got.Function.Name != "run_command" || a["command"] != c.want {
			t.Errorf("%s %s: got %v %s %q, want run_command %q", c.name, c.args, ok, got.Function.Name, a["command"], c.want)
		}
	}
	for _, c := range []struct{ name, args string }{
		{"read", `{"query": "MaxItems"}`},    // nothing to read
		{"deploy", `{"path": "inv/inv.go"}`}, // not a read
		{"bash", `{"script": "ls"}`},         // no command
		{"read", `not json`},
	} {
		var tc toolCall
		tc.Function.Name, tc.Function.Arguments = c.name, c.args
		if got, ok := canonicalCall(tc); ok {
			t.Errorf("%s %s: rewritten to %s, want left alone", c.name, c.args, got.Function.Arguments)
		}
	}
}

// A call to a read tool that does not exist runs as the shell read it means, and the
// history the next request carries shows that run_command, not the made-up tool.
func TestExecuteRunsAMadeUpReadAsShell(t *testing.T) {
	mock := newMockLLM(t,
		sseToolCall("r1", "read", `{"file_path": "notes.txt", "offset": 2, "limit": 1}`),
		sseToolCall("r2", respondToolName, `{"message":"done"}`))
	defer mock.Close()
	a, s := newTestAgent(t)
	writeTree(t, s.Cwd, map[string]string{"notes.txt": "one\ntwo\nthree\n"})
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}}
	var ran string
	a.tools.add(fakeTool("run_command", func(args string) (string, bool) {
		ran = parseArgs(args).str("command")
		return "exit 0\n\ntwo\n", false
	}))
	if out := a.runExecutePhase(context.Background(), s.ID, subtask{Description: "read line 2"}, 0, 1); out.Reason != "" {
		t.Fatalf("executor error: %s", out.Reason)
	}
	sent, _ := mock.request(1)["messages"].([]any)
	var call, result string
	for _, m := range sent {
		mm, _ := m.(map[string]any)
		if tcs, _ := mm["tool_calls"].([]any); len(tcs) > 0 {
			f, _ := tcs[0].(map[string]any)["function"].(map[string]any)
			call = fmt.Sprint(f["name"], " ", f["arguments"])
		}
		if mm["role"] == "tool" {
			result = fmt.Sprint(mm["content"])
		}
	}
	if !strings.Contains(call, "run_command") || !strings.Contains(call, `sed -n '2,2p' notes.txt`) {
		t.Errorf("history carries %q, want the run_command it ran as", call)
	}
	if ran != "sed -n '2,2p' notes.txt" || !strings.Contains(result, "two") {
		t.Errorf("result = %q, want line two", result)
	}
}
