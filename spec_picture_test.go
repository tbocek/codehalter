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
		"spec/00-principles.md":  "# 00 Principles\n\nNo styles: the prototype's Style dropdown is gone.\n",
		"spec/Readme.md":         "# The prototype\n\n" + strings.Repeat("The prototype's own manual, page after page.\n", 2000),
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
	// The whole chapter, the sections it links in other chapters once each, the standing rules
	// that fit and the item's answers: a window's picture also shows what other sections describe.
	text := parts[0].(map[string]any)["text"].(string)
	if images != 2 || strings.Count(text, "The window, the log under the page") != 1 || !strings.Contains(text, "chain dropdown is gone in the rewrite") ||
		!strings.Contains(text, "prototype: ✂, not reused") || !strings.Contains(text, "I'm feeling lucky replaces it") || !strings.Contains(text, "the prototype's Style dropdown is gone") {
		t.Errorf("the check got %d images and prompt %v, want the picture, the render, the whole chapter, the linked section, the standing rules and the answer", images, parts[0])
	}
	if strings.Contains(text, "The prototype's own manual") {
		t.Error("a standing-rules file larger than the check's budget went in")
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

// A screen that draws differently twice on the same code is not looked at and does not count
// against the item: the next item round is asked to make the snapshot deterministic first, and
// the note stays until a round draws it the same twice.
func TestSpecUnsteadySnapshotIsNotTheItemsFault(t *testing.T) {
	if _, err := exec.LookPath("just"); err != nil {
		t.Skip("no just")
	}
	a, sess := newTestAgent(t)
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 8, 6))); err != nil {
		t.Fatal(err)
	}
	writeTree(t, sess.Cwd, map[string]string{
		".gitignore":             ".codehalter/\n",
		"spec/03-shell.md":       "# 03 Shell\n\n## 1. Screen\n\nThe window. ![window](img/03-window.png)\n",
		"spec/img/03-window.png": buf.String(),
		"render.png":             buf.String(),
		// Until ../steady exists, every render holds other bytes, as a picture taken before the screen settled.
		"rust/justfile":   "test:\n\ttrue\n\nsnapshot name:\n\tmkdir -p shots && if [ -e ../steady ]; then cp ../render.png shots/{{name}}.png; else head -c 64 /dev/urandom > shots/{{name}}.png; fi\n",
		"rust/src/lib.rs": "// v1\n",
	})
	idx, err := scanSpec(filepath.Join(sess.Cwd, "spec"), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	item := idx.order[0]
	gitInit(t, sess.Cwd)
	m := newMockLLM(t, sseText(`{"verdict": "MATCH", "differences": []}`))
	defer m.Close()
	a.settings = Settings{LLM: []LLMConnection{*m.conn("execute")}}
	a.imagesSupported.Store(true)
	cfg := &specConfig{SpecDir: "spec", OutDir: "rust", TestCmd: "true", LintCmd: "off", Items: map[string]specLedger{}}
	r := &specRun{a: a, sid: sess.ID, sess: sess, cfg: cfg, idx: idx, outAbs: filepath.Join(sess.Cwd, "rust"), reasons: map[string]string{}}
	defer sess.startToolLog()()
	round := func(w specWork, change string) bool {
		t.Helper()
		// Each round's own test, so only the picture decides.
		writeTree(t, sess.Cwd, map[string]string{"rust/src/lib.rs": change, "rust/tests/screen.rs": change + "#[test]\nfn " + specTestToken(item) + "_window() {}\n"})
		mark := sess.toolMark()
		a.recordToolUse(sess.ID, toolCall{ID: "c-" + change}, ToolUse{Output: "exit 0\n[codehalter, not the user: " + sideBySideMark + "spec/img/03-window.png. ...]"})
		done, block, err := r.finishRound(t.Context(), w, nil, mark, time.Now())
		if err != nil || block {
			t.Fatalf("done=%v block=%v err=%v", done, block, err)
		}
		return done
	}

	if !round(specWork{Item: item}, "// v2\n") {
		t.Errorf("an unsteady snapshot counted against the item: %q", r.reasons[item])
	}
	if m.callCount() != 0 || !strings.Contains(cfg.Snapshot, "`just snapshot 03-window` in rust/") {
		t.Errorf("%d looks, snapshot note %q: want no look and the command named", m.callCount(), cfg.Snapshot)
	}
	// The next round is asked first; drawing it differently again is not a fix.
	w := specWork{Item: item, Snapshot: cfg.Snapshot}
	if prompt, _ := a.specRoundPrompt(sess.ID, cfg, idx, w, "true", "off"); !strings.Contains(prompt, "First: a snapshot that is not deterministic") || !strings.Contains(prompt, "`just snapshot 03-window`") {
		t.Errorf("the next round was not asked to fix the snapshot:\n%s", prompt)
	}
	round(w, "// v3, not fixed\n")
	if cfg.Snapshot == "" {
		t.Error("the note was cleared by a round that drew the screen differently again")
	}
	writeTree(t, sess.Cwd, map[string]string{"steady": ""})
	if !round(specWork{Item: item, Snapshot: cfg.Snapshot}, "// v4, the fix\n") || m.callCount() != 1 || cfg.Snapshot != "" {
		t.Errorf("after the fix: %d looks, note %q, reason %q; want one look, no note, done", m.callCount(), cfg.Snapshot, r.reasons[item])
	}
}
