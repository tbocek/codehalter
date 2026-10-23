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
// differences it names become the reason. A match passes the item. A later round on another
// item has the done screen looked at again only when its render changed, and fails when it
// no longer matches.
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
		"spec/03-shell.md":       "# 03 Shell\n\n## 1. Screen\n\nThe window, the log under the page, the [sources](02-list.md#1-sources). ![window](img/03-window.png)\n\n## 2. Run bar\n\nThe prototype's chain dropdown is gone in the rewrite.\n",
		"spec/02-list.md":        "# 02 List\n\n## 1. Sources\n\nEach row ends in 🗣 (prototype: ✂, not reused).\n",
		"spec/QUESTIONS.md":      "## §03-shell#1-screen · Which run bar?\n\n**Answer:** The chain dropdown in the picture is the prototype's; I'm feeling lucky replaces it.\n",
		"spec/img/03-window.png": pic,
		"render.png":             pic,
		"rust/justfile":          "test:\n\ttrue\n\nsnapshot name:\n\tmkdir -p shots && cp ../render.png shots/{{name}}.png && cp ../render.png shots/{{name}}.page.png\n",
		"rust/src/lib.rs":        "// v1\n",
	})
	idx, err := scanSpec(filepath.Join(sess.Cwd, "spec"), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	item, runBar := "", ""
	for _, id := range idx.order {
		switch {
		case strings.Contains(id, "1-screen"):
			item = id
		case strings.Contains(id, "2-run-bar"):
			runBar = id
		}
	}
	if item == "" || runBar == "" {
		t.Fatalf("no screen and run bar items in %v", idx.order)
	}
	gitInit(t, sess.Cwd)
	// The round's own test, so only the picture decides.
	writeTree(t, sess.Cwd, map[string]string{"rust/tests/screen.rs": "#[test]\nfn " + specTestToken(item) + "_window() {}\n"})

	differs := `{"verdict": "MISMATCH", "differences": ["top of the page: the picture shows the sources list but the render starts mid-page", "under the run bar: a thin row in the picture but an empty band in the render"]}`
	broke := `{"verdict": "MISMATCH", "differences": ["run bar: one row in the picture but two in the render"]}`
	m := newMockLLM(t, sseText("Looking at both.\n"+differs), sseText(`{"verdict": "MATCH", "differences": []}`), sseText(broke))
	defer m.Close()
	a.settings = Settings{LLM: []LLMConnection{*m.conn("execute")}}
	a.imagesSupported.Store(true)
	cfg := &specConfig{SpecDir: "spec", OutDir: "rust", TestCmd: "true", LintCmd: "off", Items: map[string]specLedger{}}
	r := &specRun{a: a, sid: sess.ID, sess: sess, cfg: cfg, idx: idx, outAbs: filepath.Join(sess.Cwd, "rust"), reasons: map[string]string{}}
	defer sess.startToolLog()()
	var roundOn func(item, change string) bool
	round := func(change string) bool {
		t.Helper()
		return roundOn(item, change)
	}
	roundOn = func(item, change string) bool {
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
	// The whole chapter, the sections it links in other chapters once each, and the item's
	// answers: a window's picture also shows what other sections describe.
	if text := parts[0].(map[string]any)["text"].(string); images != 2 || strings.Count(text, "The window, the log under the page") != 1 || !strings.Contains(text, "chain dropdown is gone in the rewrite") ||
		!strings.Contains(text, "prototype: ✂, not reused") || !strings.Contains(text, "I'm feeling lucky replaces it") {
		t.Errorf("the check got %d images and prompt %v, want the picture, the render, the whole chapter, the linked section and the answer", images, parts[0])
	}

	if !round("// v3, the fix\n") {
		t.Errorf("a render the check found matching did not pass: %q", r.reasons[item])
	}
	if m.callCount() != 2 {
		t.Errorf("%d checks, want one per round", m.callCount())
	}

	// The run bar's round changes the window's render: the done screen is looked at again, and
	// its differences fail this round, named with the item they belong to.
	var other bytes.Buffer
	shifted := image.NewRGBA(image.Rect(0, 0, 8, 6))
	copy(shifted.Pix, []byte{255, 0, 0, 255})
	if err := png.Encode(&other, shifted); err != nil {
		t.Fatal(err)
	}
	writeTree(t, sess.Cwd, map[string]string{"render.png": other.String(), "rust/tests/run_bar.rs": "#[test]\nfn " + specTestToken(runBar) + "_row() {}\n"})
	if roundOn(runBar, "// v4, the run bar\n") {
		t.Error("a round that broke a done screen counted")
	}
	if why := r.reasons[runBar]; !strings.Contains(why, "screens of items already done") || !strings.Contains(why, item+"'s rust/shots/03-window.png beside spec/img/03-window.png: run bar: one row in the picture but two in the render") {
		t.Errorf("reason = %q, want the done screen's differences", why)
	}
	// Put back: the render is the one that matched, so it is not looked at again.
	writeTree(t, sess.Cwd, map[string]string{"render.png": pic})
	if !roundOn(runBar, "// v5, put back\n") {
		t.Errorf("the round that put the screen back did not pass: %q", r.reasons[runBar])
	}
	if m.callCount() != 3 {
		t.Errorf("%d checks, want none for a render that is the one that matched", m.callCount())
	}
}
