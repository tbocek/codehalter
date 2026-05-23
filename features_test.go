package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// newTestAgent returns an agent with one session rooted at a fresh tempdir.
// a.conn is left nil so sendUpdate becomes a no-op (covered by the nil-check).
func newTestAgent(t *testing.T) (*agent, *Session) {
	t.Helper()
	s, err := newSession(t.TempDir())
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	return &agent{sessions: map[string]*Session{s.ID: s}}, s
}

// withFreshToolRegistry saves and restores the package-global registeredTools
// around a test body, so registering fake tools doesn't leak into other tests.
func withFreshToolRegistry(t *testing.T) {
	t.Helper()
	saved := registeredTools
	registeredTools = nil
	t.Cleanup(func() { registeredTools = saved })
}

func toolDef(name string) map[string]any {
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": name, "description": "test tool",
			"parameters": map[string]any{"type": "object"},
		},
	}
}

func toolNames(defs []map[string]any) []string {
	names := make([]string, 0, len(defs))
	for _, d := range defs {
		fn, _ := d["function"].(map[string]any)
		n, _ := fn["name"].(string)
		names = append(names, n)
	}
	return names
}

// contentString asserts that an llmMessage carries a string payload and
// returns it; it fails the test if the content was something else.
func contentString(t *testing.T, m llmMessage) string {
	t.Helper()
	s, ok := m.Content.(string)
	if !ok {
		t.Fatalf("expected string content, got %T", m.Content)
	}
	return s
}

// ---------------------------------------------------------------------------
// Path security
// ---------------------------------------------------------------------------

// TestResolvePath covers the sandbox boundary: relative paths are joined to
// sess.Cwd; absolute paths must already live inside it; `..` that would escape
// the root is rejected.
func TestResolvePath(t *testing.T) {
	a, s := newTestAgent(t)
	cwd := s.Cwd

	cases := []struct {
		name    string
		in      string
		want    string // expected resolved value; "" means expect error
		wantErr bool
	}{
		{name: "relative file", in: "foo.go", want: filepath.Join(cwd, "foo.go")},
		{name: "relative nested", in: "a/b/c.go", want: filepath.Join(cwd, "a/b/c.go")},
		{name: "dot", in: ".", want: cwd},
		{name: "empty", in: "", want: cwd},
		{name: "absolute inside cwd", in: filepath.Join(cwd, "x.go"), want: filepath.Join(cwd, "x.go")},
		{name: "escape via dotdot", in: "../../../etc/passwd", wantErr: true},
		{name: "absolute outside cwd", in: "/etc/passwd", wantErr: true},
		{name: "trailing dotdot escape", in: "sub/../../escaped", wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := a.resolvePath(s.ID, tc.in)
			if tc.wantErr {
				if err == nil {
					t.Errorf("expected error for %q, got %q", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}

	if _, err := a.resolvePath("missing", "foo.go"); err == nil {
		t.Error("expected error for unknown session id, got nil")
	}
}

// TestCwdOrDefaultAbsolutes pins the contract that sess.Cwd is always an
// absolute path. The bench harness sends Cwd: "." over ACP, and an unresolved
// "." breaks resolvePath's prefix check (filepath.Clean drops the leading
// "./" so "go.mod" matches neither "./" nor "."). resolvePath then rejects
// every project-relative path with "outside project directory".
func TestCwdOrDefaultAbsolutes(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	for _, in := range []string{".", "", "./"} {
		got := cwdOrDefault(in)
		if !filepath.IsAbs(got) {
			t.Errorf("cwdOrDefault(%q) = %q, want absolute path", in, got)
		}
		if got != cwd {
			t.Errorf("cwdOrDefault(%q) = %q, want %q", in, got, cwd)
		}
	}
}

// ---------------------------------------------------------------------------
// Tool filter
// ---------------------------------------------------------------------------

// TestToolFilter pins the exclude semantics: an empty filter passes every
// registered tool, and the exclude set drops the named tools. The
// "this tool mutates" gate is gone — every phase sees every tool.
func TestToolFilter(t *testing.T) {
	withFreshToolRegistry(t)
	RegisterTool(Tool{Def: toolDef("read")})
	RegisterTool(Tool{Def: toolDef("write")})
	RegisterTool(Tool{Def: toolDef("other")})

	cases := []struct {
		name   string
		filter toolFilter
		want   []string
	}{
		{"no filter", toolFilter{}, []string{"read", "write", "other"}},
		{"exclude read", toolFilter{exclude: map[string]bool{"read": true}}, []string{"write", "other"}},
		{"exclude two", toolFilter{exclude: map[string]bool{"read": true, "other": true}}, []string{"write"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := toolNames(llmToolDefinitionsFiltered(tc.filter))
			if !slices.Equal(got, tc.want) {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Session mutators with branching behavior
// ---------------------------------------------------------------------------

// TestUpsertLastAssistant pins both branches of UpsertLastAssistant: append a
// new assistant turn when the trailing role is not assistant, overwrite the
// existing one otherwise.
func TestUpsertLastAssistant(t *testing.T) {
	s := &Session{}

	// Empty → append.
	s.UpsertLastAssistant("first")
	if len(s.Messages) != 1 || s.Messages[0].Role != "assistant" || s.Messages[0].Content != "first" {
		t.Fatalf("empty case: got %+v", s.Messages)
	}

	// Trailing assistant → overwrite.
	s.UpsertLastAssistant("replaced")
	if len(s.Messages) != 1 || s.Messages[0].Content != "replaced" {
		t.Fatalf("overwrite case: got %+v", s.Messages)
	}

	// Trailing user → append.
	s.AddUser("question")
	s.UpsertLastAssistant("answer")
	if len(s.Messages) != 3 {
		t.Fatalf("append case: got %d messages, want 3", len(s.Messages))
	}
	if s.Messages[2].Role != "assistant" || s.Messages[2].Content != "answer" {
		t.Errorf("tail: got %+v", s.Messages[2])
	}
}

// ---------------------------------------------------------------------------
// History shape
// ---------------------------------------------------------------------------

// TestBuildLLMHistoryShape verifies the header injection when a summary
// exists (one user message), message ordering, and that skipIdx is honoured.
func TestBuildLLMHistoryShape(t *testing.T) {
	a := &agent{}
	s := &Session{
		Summary: "earlier summary",
		Messages: []Message{
			{Role: "user", Content: "q1"},
			{Role: "assistant", Content: "a1"},
			{Role: "user", Content: "q2"},
		},
	}

	msgs := a.buildLLMHistory(s, 2)

	if len(msgs) != 3 {
		t.Fatalf("got %d messages, want 3: %+v", len(msgs), msgs)
	}
	if msgs[0].Role != "user" {
		t.Errorf("header role: got %q, want user", msgs[0].Role)
	}
	if !strings.Contains(contentString(t, msgs[0]), "earlier summary") {
		t.Errorf("intro missing summary content: %q", contentString(t, msgs[0]))
	}
	if got := contentString(t, msgs[1]); got != "q1" {
		t.Errorf("msgs[1]: got %q, want q1", got)
	}
	if got := contentString(t, msgs[2]); got != "a1" {
		t.Errorf("msgs[2]: got %q, want a1", got)
	}

	// No summary → no header; skipIdx still honoured.
	s2 := &Session{Messages: []Message{
		{Role: "user", Content: "q1"},
		{Role: "assistant", Content: "a1"},
	}}
	msgs2 := a.buildLLMHistory(s2, 0)
	if len(msgs2) != 1 {
		t.Fatalf("no-summary case: got %d, want 1", len(msgs2))
	}
	if got := contentString(t, msgs2[0]); got != "a1" {
		t.Errorf("no-summary case content: got %q, want a1", got)
	}
}

// ---------------------------------------------------------------------------
// historyMessage image handling
// ---------------------------------------------------------------------------

// TestHistoryMessageImageHandling covers the imagesSupported branch: text
// fallback when false, structured content blocks when true.
func TestHistoryMessageImageHandling(t *testing.T) {
	img := ImageData{MimeType: "image/png", Data: base64.StdEncoding.EncodeToString([]byte("pngbytes"))}
	msg := Message{Role: "user", Content: "look at this", Images: []ImageData{img, img}}

	// images not supported → text with annotation.
	a := &agent{imagesSupported: false}
	if got := contentString(t, a.historyMessage(msg)); !strings.Contains(got, "[Images: 2 attached]") {
		t.Errorf("expected '[Images: 2 attached]' in %q", got)
	}

	// images supported → []any with one text block + N image_url blocks.
	a.imagesSupported = true
	parts, ok := a.historyMessage(msg).Content.([]any)
	if !ok {
		t.Fatalf("expected []any content, got different type")
	}
	if len(parts) != 3 {
		t.Fatalf("expected 3 parts (text + 2 images), got %d", len(parts))
	}
	text, _ := parts[0].(map[string]any)
	if text["type"] != "text" || text["text"] != "look at this" {
		t.Errorf("parts[0] wrong: %+v", text)
	}
	for i, part := range parts[1:] {
		block, _ := part.(map[string]any)
		if block["type"] != "image_url" {
			t.Errorf("parts[%d] type: got %v, want image_url", i+1, block["type"])
		}
		url, _ := block["image_url"].(map[string]string)
		if !strings.HasPrefix(url["url"], "data:image/png;base64,") {
			t.Errorf("parts[%d] url prefix wrong: %q", i+1, url["url"])
		}
	}

	// Message with no images → plain string, untouched.
	if got := contentString(t, a.historyMessage(Message{Role: "user", Content: "no imgs"})); got != "no imgs" {
		t.Errorf("plain: got %q, want 'no imgs'", got)
	}

	// Combined: tool uses + images together. The tool-call summary should be
	// inlined into the text block; the image URLs follow.
	combined := Message{
		Role:     "assistant",
		Content:  "done",
		Images:   []ImageData{img},
		ToolUses: []ToolUse{{Name: "read_file", Input: `{"path":"x"}`, Output: "ok"}},
	}
	parts, ok = a.historyMessage(combined).Content.([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("combined: expected []any of len 2, got %+v", parts)
	}
	text, _ = parts[0].(map[string]any)
	textStr, _ := text["text"].(string)
	if !strings.Contains(textStr, "done") || !strings.Contains(textStr, "read_file") {
		t.Errorf("combined text block missing content or tool summary: %q", textStr)
	}
}

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

// TestEstimateMessagesTokensCountsToolUses guards the fix where tool Name +
// Input + Output each contribute to the budget, not just Content. Tests each
// field individually so a regression that drops any one of them is caught.
func TestEstimateMessagesTokensCountsToolUses(t *testing.T) {
	baseline := estimateMessagesTokens([]Message{{Role: "assistant", Content: "hello"}})
	big := strings.Repeat("x", 1000) // 250 tokens @ 4 chars/token
	mkMsg := func(tu ToolUse) []Message {
		return []Message{{Role: "assistant", Content: "hello", ToolUses: []ToolUse{tu}}}
	}

	// Each tool-use field, in isolation, must push the count past baseline.
	cases := []struct {
		field string
		msg   []Message
	}{
		{"Name", mkMsg(ToolUse{Name: big})},
		{"Input", mkMsg(ToolUse{Input: big})},
		{"Output", mkMsg(ToolUse{Output: big})},
	}
	for _, tc := range cases {
		t.Run(tc.field, func(t *testing.T) {
			got := estimateMessagesTokens(tc.msg)
			// -roleTokenOverhead accounts for the per-tool-use overhead
			// added on top of the field contents.
			if got-baseline < 250 {
				t.Errorf("ToolUse.%s not counted: baseline=%d, got=%d, want delta >=250",
					tc.field, baseline, got)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// verify.go JSON parse fallback
// ---------------------------------------------------------------------------

// TestVerifyJSONParseFallback exercises the graceful-degradation path: when
// the verify LLM returns non-JSON, we issue one corrective retry; if that
// also fails to parse, we swallow the error and treat verification as
// successful so the pipeline still progresses.
func TestVerifyJSONParseFallback(t *testing.T) {
	a, s := newTestAgent(t)

	// Drop a VERIFY.md into the session's .codehalter dir.
	verifyPath := filepath.Join(s.Cwd, ".codehalter", "VERIFY.md")
	if err := os.WriteFile(verifyPath, []byte("Check that X is true."), 0644); err != nil {
		t.Fatalf("write VERIFY.md: %v", err)
	}

	mock := newMockLLM(t,
		sseText("this is not JSON at all"),
		sseText("still not JSON"),
	)
	defer mock.Close()

	res := toolLoopResult{Text: "my previous response"}
	out, vr, err := a.verify(context.Background(), s.ID, mock.conn("execute"), nil, res)
	if err != nil {
		t.Fatalf("verify returned error: %v", err)
	}
	if vr == nil || !vr.Success {
		t.Errorf("expected Success=true fallback, got %+v", vr)
	}
	if out.Text != res.Text {
		t.Errorf("expected res unchanged, got %q", out.Text)
	}
	if mock.callCount() != 2 {
		t.Errorf("expected 2 LLM calls (initial + retry), got %d", mock.callCount())
	}
}

// TestPlanResultSubtasksDeserialize ensures the new `subtasks` field on
// planResult round-trips from the JSON the planner emits. This is the seam
// Prompt branches on to enter the decomposed plan→execute→verify flow.
func TestPlanResultSubtasksDeserialize(t *testing.T) {
	raw := `{"clear":true,"complexity":"complex","steps":[],"subtasks":["refactor storage","update API","write migration"]}`
	var p planResult
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !slices.Equal(p.Subtasks, []string{"refactor storage", "update API", "write migration"}) {
		t.Errorf("subtasks: got %v", p.Subtasks)
	}
	if len(p.Steps) != 0 {
		t.Errorf("steps should be empty when subtasks set, got %v", p.Steps)
	}

	// Backward-compat: old-style plan with only steps leaves Subtasks nil.
	rawOld := `{"clear":true,"complexity":"simple","steps":["edit foo.go"]}`
	var p2 planResult
	if err := json.Unmarshal([]byte(rawOld), &p2); err != nil {
		t.Fatalf("unmarshal legacy: %v", err)
	}
	if len(p2.Subtasks) != 0 {
		t.Errorf("expected no subtasks for legacy plan, got %v", p2.Subtasks)
	}
}

// ---------------------------------------------------------------------------
// Runner discovery + classification
// ---------------------------------------------------------------------------

// TestClassifyTask covers the task-name heuristic. The classifier must be
// narrow enough that "checkout" doesn't get labelled "lint" just because it
// contains "check".
func TestClassifyTask(t *testing.T) {
	cases := map[string]string{
		"build":      "build",
		"compile":    "build",
		"bundle":     "build",
		"test":       "test",
		"unit-test":  "test",
		"spec":       "test",
		"lint":       "lint",
		"vet":        "lint",
		"check":      "lint",
		"clippy":     "lint",
		"fmt":        "format",
		"format":     "format",
		"prettier":   "format",
		"clean":      "", // not in any category yet
		"start":      "",
		"checkout":   "",     // segment match: "checkout" is one segment, not "check"
		"checkstyle": "",     // same — "checkstyle" is one segment
		"test-build": "test", // first segment wins
		"test:unit":  "test", // colon separator
		"build.all":  "build",
	}
	for task, want := range cases {
		if got := classifyTask(task); got != want {
			t.Errorf("classifyTask(%q): got %q, want %q", task, got, want)
		}
	}
}

// TestClassifyRunners verifies each task gets slotted into its category and
// the distinct runner list is preserved.
func TestClassifyRunners(t *testing.T) {
	runners := []taskRunner{
		{Name: "go", Tasks: []string{"build", "test", "vet", "fmt"}},
		{Name: "make", Tasks: []string{"build", "clean", "release"}},
	}
	caps := classifyRunners(runners)

	if !slices.Equal(caps.runners, []string{"go", "make"}) {
		t.Errorf("runners: got %v, want [go make]", caps.runners)
	}
	if !slices.Equal(caps.build, []string{"go:build", "make:build"}) {
		t.Errorf("build: got %v", caps.build)
	}
	if !slices.Equal(caps.test, []string{"go:test"}) {
		t.Errorf("test: got %v", caps.test)
	}
	if !slices.Equal(caps.lint, []string{"go:vet"}) {
		t.Errorf("lint: got %v", caps.lint)
	}
	if !slices.Equal(caps.format, []string{"go:fmt"}) {
		t.Errorf("format: got %v", caps.format)
	}
}

// TestDiscoverGoAndCargo verifies the zero-parse runners fire when their
// manifest exists and produce the standard subcommand list. Go discovery
// is a fallback that defers to a justfile/Makefile when one is present.
func TestDiscoverGoAndCargo(t *testing.T) {
	goDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(goDir, "go.mod"), []byte("module x\n"), 0644); err != nil {
		t.Fatalf("go.mod: %v", err)
	}
	r := discoverGo(goDir)
	if r == nil || r.Name != "go" || !slices.Contains(r.Tasks, "test") || !slices.Contains(r.Tasks, "vet") {
		t.Errorf("discoverGo: got %+v", r)
	}
	if args := r.Args("test"); !slices.Equal(args, []string{"test", "./..."}) {
		t.Errorf("go Args(test): got %v, want [test ./...]", args)
	}

	// A Makefile alongside go.mod should suppress the Go fallback.
	if err := os.WriteFile(filepath.Join(goDir, "Makefile"), []byte("test:\n\tgo test ./...\n"), 0644); err != nil {
		t.Fatalf("Makefile: %v", err)
	}
	if r := discoverGo(goDir); r != nil {
		t.Errorf("discoverGo with Makefile present: got %+v, want nil", r)
	}

	cargoDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(cargoDir, "Cargo.toml"), []byte("[package]\nname=\"x\"\n"), 0644); err != nil {
		t.Fatalf("Cargo.toml: %v", err)
	}
	r = discoverCargo(cargoDir)
	if r == nil || r.Name != "cargo" || !slices.Contains(r.Tasks, "clippy") {
		t.Errorf("discoverCargo: got %+v", r)
	}

	// Absent manifests return nil cleanly.
	if r := discoverGo(t.TempDir()); r != nil {
		t.Errorf("discoverGo on empty dir: got %+v, want nil", r)
	}
	if r := discoverCargo(t.TempDir()); r != nil {
		t.Errorf("discoverCargo on empty dir: got %+v, want nil", r)
	}
}

// TestEmptyProjectFlag covers the deferred-bootstrap path: a fresh empty
// dir sets a.emptyProject so the first user turn can inject a hint asking
// what language/runner to use. Populated dirs must not be flagged.
func TestEmptyProjectFlag(t *testing.T) {
	dir := t.TempDir()
	if !isEmptyProject(dir) {
		t.Fatal("expected fresh tempdir to be empty")
	}

	s, err := newSession(dir) // creates .codehalter/ but no source files
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	if !isEmptyProject(dir) {
		t.Error("expected dir with only .codehalter/ to still count as empty")
	}

	a := &agent{sessions: map[string]*Session{s.ID: s}}
	a.discoverRunners(dir)

	if !a.emptyProject {
		t.Error("expected emptyProject=true for empty dir")
	}
	if len(a.capabilities.runners) != 0 {
		t.Errorf("expected no runners discovered, got %v", a.capabilities.runners)
	}
	if _, err := os.Stat(filepath.Join(dir, "Makefile")); err == nil {
		t.Error("bootstrap must be deferred — no Makefile should be written")
	}

	// Non-empty project: isEmptyProject=false and flag stays off.
	populated := t.TempDir()
	if err := os.WriteFile(filepath.Join(populated, "main.go"), []byte("package main\n"), 0644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if isEmptyProject(populated) {
		t.Error("expected dir with main.go to not be empty")
	}
	a2 := &agent{sessions: map[string]*Session{}}
	a2.discoverRunners(populated)
	if a2.emptyProject {
		t.Error("expected emptyProject=false when source files are present")
	}
}

// ---------------------------------------------------------------------------
// search_text matcher (literal vs regex)
// ---------------------------------------------------------------------------

// TestSearchInFileMatchers verifies the matcher callback cleanly swaps
// literal substring and regex behavior. Literal mode treats '.' as a dot;
// regex mode treats it as any-char.
func TestSearchInFileMatchers(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	body := "alpha\nbeta.go\nbeta_go\ngamma\n"
	if err := os.WriteFile(path, []byte(body), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}

	// Literal: "beta.go" matches only line 2 (the actual dot).
	lit := searchInFile(path, func(s string) bool { return strings.Contains(s, "beta.go") }, 10)
	if !slices.Equal(lit, []int{2}) {
		t.Errorf("literal: got %v, want [2]", lit)
	}

	// Regex: "beta.go" matches both lines 2 and 3 (. = any char).
	re := regexp.MustCompile("beta.go")
	rx := searchInFile(path, re.MatchString, 10)
	if !slices.Equal(rx, []int{2, 3}) {
		t.Errorf("regex: got %v, want [2 3]", rx)
	}

	// Case-insensitive via inline flag.
	reI := regexp.MustCompile("(?i)ALPHA")
	ci := searchInFile(path, reI.MatchString, 10)
	if !slices.Equal(ci, []int{1}) {
		t.Errorf("case-insensitive: got %v, want [1]", ci)
	}

	// Limit is honoured.
	anyLine := regexp.MustCompile(".+")
	limited := searchInFile(path, anyLine.MatchString, 2)
	if len(limited) != 2 {
		t.Errorf("limit=2: got %d matches, want 2", len(limited))
	}
}

// ---------------------------------------------------------------------------
// parseArgs + trimJSON utilities
// ---------------------------------------------------------------------------

func TestParseArgs(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want map[string]string
	}{
		{name: "valid flat", in: `{"key":"value","a":"b"}`, want: map[string]string{"key": "value", "a": "b"}},
		{name: "empty object", in: `{}`, want: map[string]string{}},
		{name: "empty string", in: ``, want: map[string]string{}},
		{name: "invalid JSON", in: `not json`, want: map[string]string{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := parseArgs(tc.in)
			if got == nil {
				t.Fatalf("parseArgs must never return nil")
			}
			if len(got) != len(tc.want) {
				t.Errorf("got %v, want %v", got, tc.want)
			}
			for k, v := range tc.want {
				if got[k] != v {
					t.Errorf("key %q: got %q, want %q", k, got[k], v)
				}
			}
		})
	}
}

func TestTrimJSON(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{name: "plain", in: `{"ok":true}`, want: `{"ok":true}`},
		{name: "leading whitespace", in: "  \n{\"ok\":true}\n  ", want: `{"ok":true}`},
		{name: "json fence", in: "```json\n{\"ok\":true}\n```", want: `{"ok":true}`},
		{name: "bare fence", in: "```\n{\"ok\":true}\n```", want: `{"ok":true}`},
		{name: "prose prefix", in: "Sure, here's the JSON:\n{\"ok\":true}", want: `{"ok":true}`},
		{name: "prose suffix", in: "{\"ok\":true}\nLet me know if you need more.", want: `{"ok":true}`},
		{name: "prose both sides", in: "Here you go: {\"ok\":true} — that's it!", want: `{"ok":true}`},
		{name: "nested", in: "noise {\"a\":{\"b\":1}} noise", want: `{"a":{"b":1}}`},
		{name: "brace in string", in: `{"s":"} not the end"}`, want: `{"s":"} not the end"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := trimJSON(tc.in); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// TestIssueBagTokenisation pins the bag-of-words tokeniser used for fuzzy
// failure matching: lowercase, punctuation-stripped, order-independent. The
// reworded-near-duplicate case ("missing import" vs "import is missing") is
// the one that motivates the fuzzy approach over exact key matching.
func TestIssueBagTokenisation(t *testing.T) {
	// Casing, punctuation and word order are all discarded.
	a := issueBag([]string{"Missing import!", "Syntax error."})
	b := issueBag([]string{"syntax  ERROR", "missing\timport"})
	if !slices.Equal(sortedKeys(a), sortedKeys(b)) {
		t.Errorf("expected equivalent bags, got %v vs %v", sortedKeys(a), sortedKeys(b))
	}

	// Adjacent non-alphanumeric runs collapse to a single separator (no empty
	// tokens leak into the bag).
	bag := issueBag([]string{"foo--bar...baz"})
	want := []string{"bar", "baz", "foo"}
	if !slices.Equal(sortedKeys(bag), want) {
		t.Errorf("got %v, want %v", sortedKeys(bag), want)
	}
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	slices.Sort(out)
	return out
}

// TestJaccardSimilarity covers the failure-loop bail decision. The reworded
// near-duplicate must score above the configured threshold so the retry
// loop bails; unrelated failures must stay below.
func TestJaccardSimilarity(t *testing.T) {
	// Two empty bags are treated as identical (degenerate but well-defined).
	if got := jaccard(map[string]bool{}, map[string]bool{}); got != 1 {
		t.Errorf("empty/empty: got %v, want 1", got)
	}

	// Reworded duplicate: {"missing","import"} vs {"import","is","missing"}.
	// |∩|=2, |∪|=3 → 0.666… → must exceed the threshold so a retry bails.
	a := issueBag([]string{"missing import"})
	b := issueBag([]string{"import is missing"})
	if s := jaccard(a, b); s < failureSimilarityThreshold {
		t.Errorf("reworded duplicate: got %v, want >= %v", s, failureSimilarityThreshold)
	}

	// Unrelated failures must NOT collapse — exact wording chosen so the
	// Jaccard score is comfortably under the threshold.
	c := issueBag([]string{"missing import in foo.go"})
	d := issueBag([]string{"unused variable x"})
	if s := jaccard(c, d); s >= failureSimilarityThreshold {
		t.Errorf("disjoint issues: got %v, want < %v", s, failureSimilarityThreshold)
	}

	// Symmetric.
	if jaccard(a, b) != jaccard(b, a) {
		t.Errorf("expected jaccard to be symmetric")
	}
}
