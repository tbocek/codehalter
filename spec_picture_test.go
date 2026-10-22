package main

import (
	"bytes"
	"image"
	"image/png"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// A screen item that looked at its picture still fails when codehalter's own render, put
// beside the picture by a fresh call, does not match; the call sees both images, and the
// differences it names become the reason. A match passes the item.
func TestSpecRoundPictureMustMatch(t *testing.T) {
	if _, err := exec.LookPath("just"); err != nil {
		t.Skip("no just")
	}
	a, sess := newTestAgent(t)
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 8, 6))); err != nil {
		t.Fatal(err)
	}
	pic := buf.String()
	writeTree(t, sess.Cwd, map[string]string{
		".gitignore":             ".codehalter/\n",
		"spec/03-shell.md":       "# 03 Shell\n\n## 1. Screen\n\nThe window, the log under the page. ![window](img/03-window.png)\n\n## 2. Run bar\n\nThe prototype's chain dropdown is gone in the rewrite.\n",
		"spec/img/03-window.png": pic,
		"render.png":             pic,
		"rust/justfile":          "test:\n\ttrue\n\nsnapshot name:\n\tmkdir -p shots && cp ../render.png shots/{{name}}.png && cp ../render.png shots/{{name}}.page.png\n",
		"rust/src/lib.rs":        "// v1\n",
	})
	idx, err := scanSpec(filepath.Join(sess.Cwd, "spec"), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	item := ""
	for _, id := range idx.order {
		if strings.Contains(id, "1-screen") {
			item = id
		}
	}
	if item == "" {
		t.Fatalf("no screen item in %v", idx.order)
	}
	gitInit(t, sess.Cwd)
	// The round's own test, so only the picture decides.
	writeTree(t, sess.Cwd, map[string]string{"rust/tests/screen.rs": "#[test]\nfn " + specTestToken(item) + "_window() {}\n"})

	differs := `{"verdict": "MISMATCH", "differences": ["top of the page: the picture shows the sources list but the render starts mid-page", "under the run bar: a thin row in the picture but an empty band in the render"]}`
	m := newMockLLM(t, sseText("Looking at both.\n"+differs), sseText(`{"verdict": "MATCH", "differences": []}`))
	defer m.Close()
	a.settings = Settings{LLM: []LLMConnection{*m.conn("execute")}}
	a.imagesSupported.Store(true)
	cfg := &specConfig{SpecDir: "spec", OutDir: "rust", TestCmd: "true", LintCmd: "off", Items: map[string]specLedger{}}
	r := &specRun{a: a, sid: sess.ID, sess: sess, cfg: cfg, idx: idx, outAbs: filepath.Join(sess.Cwd, "rust"), reasons: map[string]string{}}
	defer sess.startToolLog()()
	round := func(change string) bool {
		t.Helper()
		writeTree(t, sess.Cwd, map[string]string{"rust/src/lib.rs": change})
		mark := sess.toolMark()
		// The round put a render beside the picture, as the gate before this one asks.
		a.recordToolUse(sess.ID, toolCall{ID: "c-" + change}, ToolUse{Output: "exit 0\n[codehalter, not the user: " + sideBySideMark + "spec/img/03-window.png. ...]"})
		done, block, err := r.finishRound(t.Context(), specWork{Item: item}, nil, mark, time.Now())
		if err != nil || block {
			t.Fatalf("done=%v block=%v err=%v", done, block, err)
		}
		return done
	}

	if round("// v2\n") {
		t.Fatal("a render the check found different counted the item as done")
	}
	if why := r.reasons[item]; strings.Contains(why, "no test this round") || !strings.Contains(why, "an empty band in the render") || !strings.Contains(why, "rust/shots/03-window.png beside spec/img/03-window.png") {
		t.Errorf("reason = %q, want the named differences of the subject's render", why)
	}
	parts, _ := m.request(0)["messages"].([]any)[0].(map[string]any)["content"].([]any)
	images := 0
	for _, p := range parts {
		if p.(map[string]any)["type"] == "image_url" {
			images++
		}
	}
	// The whole chapter: a window's picture also shows what other sections describe.
	if text := parts[0].(map[string]any)["text"].(string); images != 2 || !strings.Contains(text, "The window, the log under the page.") || !strings.Contains(text, "chain dropdown is gone in the rewrite") {
		t.Errorf("the check got %d images and prompt %v, want the picture, the render and the whole chapter", images, parts[0])
	}

	if !round("// v3, the fix\n") {
		t.Errorf("a render the check found matching did not pass: %q", r.reasons[item])
	}
	if m.callCount() != 2 {
		t.Errorf("%d checks, want one per round", m.callCount())
	}
}
