package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

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
	// A symlink inside the project pointing out of it passes a prefix test;
	// the file tools refuse it, and an attachment must too.
	outside := filepath.Join(t.TempDir(), "secret.txt")
	if err := os.WriteFile(outside, []byte("secret"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link.txt")
	if err := os.Symlink(outside, link); err != nil {
		t.Skip("symlinks unavailable:", err)
	}
	if snip, _, ok := readLinkedResource(dir, "file://"+link); ok {
		t.Errorf("a symlink out of the project was inlined: %q", snip)
	}
}

// TestParseResourceURI pins the URI → path mapping for attached resources:
// file:// URIs collapse to their percent-decoded path with the fragment split
// off (editors put a line range there), anything else passes through as its own
// path so it can still be named.
func TestParseResourceURI(t *testing.T) {
	cases := map[string][2]string{
		"file:///workspaces/codehalter/llm.go":          {"/workspaces/codehalter/llm.go", ""},
		"file:///workspaces/codehalter/llm.go#L801-836": {"/workspaces/codehalter/llm.go", "L801-836"},
		"file:///a%20b/c.go":                            {"/a b/c.go", ""},
		"/plain/path.go":                                {"/plain/path.go", ""},
		"https://example.com/x":                         {"https://example.com/x", ""},
		"":                                              {"", ""},
	}
	for uri, want := range cases {
		if path, frag := parseResourceURI(uri); path != want[0] || frag != want[1] {
			t.Errorf("parseResourceURI(%q) = %q, %q, want %q, %q", uri, path, frag, want[0], want[1])
		}
	}
}

// TestPromptContent pins what the model is handed for a prompt with
// attachments: an embedded selection is inlined under a header naming the file
// and its line range, a linked file outside the project is named rather than
// read, and an image becomes a content-addressed reference, not bytes.
func TestPromptContent(t *testing.T) {
	dir := t.TempDir()
	text, images := promptContent(dir, []ContentBlock{
		{Type: "text", Text: "why do we need this?"},
		{Type: "resource", Resource: &EmbeddedResource{URI: "file:///x/llm.go#L801-836", Text: "func f() {}"}},
		{Type: "resource_link", URI: "file:///etc/passwd", Name: "passwd"},
		{Type: "image", MimeType: "image/png", Data: "aGVsbG8="},
	})
	for _, want := range []string{
		"why do we need this?",
		"[Attached context from /x/llm.go (L801-836)]",
		"func f() {}",
		"[Referenced file: passwd (/etc/passwd)]",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("text is missing %q; got:\n%s", want, text)
		}
	}
	if strings.Contains(text, "root:") {
		t.Error("a file outside the project was inlined")
	}
	if len(images) != 1 || !strings.HasPrefix(images[0].ID, "img_") || images[0].MimeType != "image/png" {
		t.Errorf("images = %+v, want one content-addressed png", images)
	}
	// Same bytes, same id: a re-pasted screenshot does not grow the store.
	_, again := promptContent(dir, []ContentBlock{{Type: "image", MimeType: "image/png", Data: "aGVsbG8="}})
	if len(again) != 1 || again[0].ID != images[0].ID {
		t.Errorf("re-pasting the same image gave %+v, want id %s", again, images[0].ID)
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

// TestDeriveTitle pins the thread-naming rules: one line, whitespace collapsed,
// cut on a word boundary. The title goes to the client as session_info_update
// and is all the user sees in their thread list, so a mid-word cut or a title
// containing half a pasted stack trace is the visible failure.
func TestDeriveTitle(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   string
		want string
	}{
		{"short prompt is used as-is", "Fix the login bug", "Fix the login bug"},
		{"leading blank lines skipped", "\n\n  Add a healthcheck  ", "Add a healthcheck"},
		{"only the first line", "Update the parser\n\nIt panics on empty input.", "Update the parser"},
		{"runs of whitespace collapse", "Add   a\ttest", "Add a test"},
		{"macro name survives", "/commit", "/commit"},
		{"no text at all", "\n  \n", ""},
		{
			"long prompt cut on a word boundary",
			"Please refactor the terminal watchdog so that it polls the client instead of streaming",
			"Please refactor the terminal watchdog so that it polls the…",
		},
		{
			"unbreakable token is cut mid-token rather than vanishing",
			strings.Repeat("x", 100),
			strings.Repeat("x", sessionTitleMax) + "…",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := deriveTitle(tc.in); got != tc.want {
				t.Errorf("deriveTitle(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestSetSessionTitleAnnouncesOnce pins that the thread name reaches the client
// as a session_info_update and is not re-sent when it hasn't changed — a
// notification per turn would make the client re-render the thread list for
// nothing.
func TestSetSessionTitleAnnouncesOnce(t *testing.T) {
	h := newTerminalHarness(t)

	h.agent.setSessionTitle(context.Background(), h.sess, "Wire up the MCP import")
	u := h.waitForKind("session_info_update")
	if u == nil {
		t.Fatalf("no session_info_update sent; got %v", h.sentUpdates())
	}
	if u["title"] != "Wire up the MCP import" {
		t.Errorf("title = %v, want the first line of the prompt", u["title"])
	}
	if h.sess.Title != "Wire up the MCP import" {
		t.Errorf("sess.Title = %q, want it persisted on the session", h.sess.Title)
	}

	h.agent.setSessionTitle(context.Background(), h.sess, "Wire up the MCP import")
	h.waitFor(func() bool { return len(h.updatesOfKind("session_info_update")) > 1 })
	if n := len(h.updatesOfKind("session_info_update")); n != 1 {
		t.Errorf("sent %d session_info_updates, want 1 — the title didn't change", n)
	}
}

// TestSpecStopIsAnInstructionNotSteer: "/spec stop" typed while a /spec loop
// holds the turn sets the loop's stop flag and is NOT queued as text for the
// model; with no loop running it only says so. Anything else typed during a
// turn still steers it.
func TestSpecStopIsAnInstructionNotSteer(t *testing.T) {
	a, s := newTestAgent(t)
	ctx, release, ok := a.holdTurn(context.Background(), s, true)
	if !ok {
		t.Fatal("could not hold the turn")
	}
	defer release()
	_ = ctx
	prompt := func(text string) {
		t.Helper()
		if _, err := a.Prompt(context.Background(), PromptRequest{SessionId: s.ID, Content: []ContentBlock{{Type: "text", Text: text}}}); err != nil {
			t.Fatalf("Prompt(%q): %v", text, err)
		}
	}

	prompt("/spec stop") // no loop running
	if s.takeSpecStop() {
		t.Error("a stop was requested with no loop running")
	}
	if q := s.takeSteer(); len(q) != 0 {
		t.Errorf("/spec stop was queued as steer text: %v", q)
	}

	s.setSpecFence(filepath.Join(s.Cwd, "spec"))
	defer s.setSpecFence("")
	prompt("/spec stop")
	if !s.takeSpecStop() {
		t.Error("the running loop did not get the stop request")
	}
	if q := s.takeSteer(); len(q) != 0 {
		t.Errorf("/spec stop was queued as steer text: %v", q)
	}
	prompt("also update the README")
	if q := s.takeSteer(); len(q) != 1 {
		t.Errorf("an ordinary message during the loop must still steer: %v", q)
	}
}
