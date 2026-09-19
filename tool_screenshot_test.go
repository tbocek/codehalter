package main

import (
	"context"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tbocek/codehalter/llm"
)

// TestScreenshotRejectsBadPaths: the two ways a path argument can be wrong
// (absent, or pointing outside the project) both fail with a message the model
// can act on, and neither launches a browser.
func TestScreenshotRejectsBadPaths(t *testing.T) {
	a, s := newTestAgent(t)
	a.imagesSupported = true

	cases := []struct {
		name string
		args string
		want string
	}{
		{"no path", `{}`, "missing `path`"},
		{"escapes the project", `{"path":"../../etc/passwd"}`, "outside project directory"},
		{"not a file", `{"path":"."}`, "not a readable file"},
		{"missing file", `{"path":"nope.html"}`, "not a readable file"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			text, parts, id, failed := dispatchScreenshot(context.Background(), a, s.ID, tc.args)
			if !failed {
				t.Fatalf("failed=false, text=%q", text)
			}
			if !strings.Contains(text, tc.want) {
				t.Errorf("text = %q, want it to contain %q", text, tc.want)
			}
			if parts != nil || id != "" {
				t.Errorf("a failed call must deliver nothing: parts=%v id=%q", parts, id)
			}
		})
	}
}

// TestScreenshotNoVisionTellsTheUser: with an LLM that takes no images the
// dispatcher never intercepts, so the call lands in the fallback. That path has
// to say so in the TRANSCRIPT, because "your model has no vision" is a
// settings.toml fact only the user can act on.
func TestScreenshotNoVisionTellsTheUser(t *testing.T) {
	a, s := newTestAgent(t)
	a.imagesSupported = false

	text, failed := screenshotExecuteFallback(context.Background(), a, s.ID, `{"path":"out/index.html"}`)
	if !failed {
		t.Fatalf("failed=false, text=%q", text)
	}
	// The model is told what to do instead, not just that it went wrong.
	if !strings.Contains(text, "NUMBER") {
		t.Errorf("model-facing text should point at numeric verification, got %q", text)
	}
	if !strings.Contains(text, "the user has been told") {
		t.Errorf("model-facing text should say the user was informed, got %q", text)
	}
}

// TestWriteInstrumentedInjects: the instrumented copy gets a <base> pointing at
// the ORIGINAL directory (or its relative CSS breaks) and the probe script, and
// the original file is left untouched.
func TestWriteInstrumentedInjects(t *testing.T) {
	src := filepath.Join(t.TempDir(), "page.html")
	original := `<html><head><title>t</title></head><body><p class="x">hi</p></body></html>`
	if err := os.WriteFile(src, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()

	out, err := writeInstrumented(dir, src, ".x", "http://127.0.0.1:1234/")
	if err != nil {
		t.Fatalf("writeInstrumented: %v", err)
	}
	if out == src {
		t.Fatal("instrumented page must be a COPY: the project tree is never mutated")
	}
	got, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	html := string(got)
	for _, want := range []string{
		`<base href="file://` + filepath.Dir(src) + `/">`,
		`document.querySelector(".x")`,
		`http://127.0.0.1:1234/`,
		`x.open('GET',`, // synchronous XHR: async loses the race with firefox exiting
		`, false)`,
		`translateY(`,
	} {
		if !strings.Contains(html, want) {
			t.Errorf("instrumented page missing %q\n---\n%s", want, html)
		}
	}
	if i, j := strings.Index(html, "<script>"), strings.Index(html, "</body>"); i == -1 || i > j {
		t.Errorf("probe script must sit before </body>, got script@%d body@%d", i, j)
	}
	back, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	if string(back) != original {
		t.Errorf("source page was modified:\n%s", back)
	}
}

// TestWriteInstrumentedFragment: a file with no <head> and no </body> (a bare
// SVG or an HTML fragment) still gets both injections, at the ends.
func TestWriteInstrumentedFragment(t *testing.T) {
	src := filepath.Join(t.TempDir(), "frag.html")
	if err := os.WriteFile(src, []byte(`<p class="x">hi</p>`), 0o600); err != nil {
		t.Fatal(err)
	}
	out, err := writeInstrumented(t.TempDir(), src, ".x", "http://127.0.0.1:1/")
	if err != nil {
		t.Fatalf("writeInstrumented: %v", err)
	}
	got, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	html := string(got)
	if !strings.HasPrefix(html, "<base href=") {
		t.Errorf("no </head> → base goes first, got %q", html)
	}
	if !strings.HasSuffix(html, "</script>") {
		t.Errorf("no </body> → script goes last, got %q", html)
	}
}

// TestSelectorNoteReportsAMiss: a selector that matched nothing is the trap
// this note exists to close. Without it the model gets the top of the page and
// no reason to doubt it's looking at what it asked for.
func TestSelectorNoteReportsAMiss(t *testing.T) {
	cases := []struct {
		name, report, want string
	}{
		{"hit", "matched=1 top=1204px height=310px", "top=1204px"},
		{"miss", "matched=0", "matched NO element"},
		{"silent", "", "could not be resolved"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := selectorNote(".grid", tc.report)
			if !strings.Contains(got, tc.want) {
				t.Errorf("selectorNote(%q) = %q, want it to contain %q", tc.report, got, tc.want)
			}
			if !strings.Contains(got, ".grid") {
				t.Errorf("note should quote the selector, got %q", got)
			}
		})
	}
}

// TestClampDimension: absent/zero/negative fall back to the default, oversized
// gets capped rather than rejected (a capped shot is still useful, and the
// reply states the size actually used).
func TestClampDimension(t *testing.T) {
	cases := []struct {
		args string
		want int
	}{
		{`{}`, screenshotDefaultWidth},
		{`{"width":0}`, screenshotDefaultWidth},
		{`{"width":-5}`, screenshotDefaultWidth},
		{`{"width":"nope"}`, screenshotDefaultWidth}, // junk → default, not a crash
		{`{"width":800}`, 800},
		{`{"width":"800"}`, 800}, // toolArgs.num takes a numeric string: models emit them
		{`{"width":99999}`, screenshotMaxWidth},
	}
	for _, tc := range cases {
		got := clampDimension(parseArgs(tc.args), "width", screenshotDefaultWidth, screenshotMaxWidth)
		if got != tc.want {
			t.Errorf("clampDimension(%s) = %d, want %d", tc.args, got, tc.want)
		}
	}
}

// TestBeaconRoundTrip: the beacon is how a number leaves a headless Firefox
// that has no --dump-dom. Anything that can GET the URL can report through it.
func TestBeaconRoundTrip(t *testing.T) {
	b, err := startBeacon()
	if err != nil {
		t.Fatalf("startBeacon: %v", err)
	}
	defer b.Close()

	if got := b.report(); got != "" {
		t.Errorf("nothing sent yet, report() = %q, want empty", got)
	}
	resp, err := http.Get(b.URL() + "?" + url.QueryEscape("matched=1 top=42px"))
	if err != nil {
		t.Fatalf("GET beacon: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatal(err)
	}
	if got := b.report(); got != "matched=1 top=42px" {
		t.Errorf("report() = %q, want %q", got, "matched=1 top=42px")
	}
}

// TestScreenshotEndToEnd drives the real browser: the only way to know the
// flags still work and that the selector shift lands. Skipped where Firefox
// isn't installed (CI, a slim container) rather than failing the suite.
func TestScreenshotEndToEnd(t *testing.T) {
	if _, err := firefoxPath(); err != nil {
		t.Skipf("no firefox: %v", err)
	}
	if testing.Short() {
		t.Skip("launches a browser")
	}
	a, s := newTestAgent(t)
	a.imagesSupported = true
	page := filepath.Join(s.Cwd, "page.html")
	// A tall spacer so the target is well below the fold: without the
	// translateY shift the capture would start at the top and miss it.
	body := `<html><head><style>body{margin:0}#pad{height:3000px;background:#eee}
	  #target{height:200px;background:#f0f}</style></head>
	  <body><div id="pad"></div><div id="target"></div></body></html>`
	if err := os.WriteFile(page, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}

	text, parts, id, failed := dispatchScreenshot(context.Background(), a, s.ID, `{"path":"page.html","selector":"#target","width":400,"height":400}`)
	if failed {
		t.Fatalf("dispatchScreenshot: %s", text)
	}
	if !strings.HasPrefix(id, "img_") {
		t.Errorf("image id = %q, want img_ prefix", id)
	}
	if len(parts) != 2 {
		t.Fatalf("want [text, image_url], got %d parts", len(parts))
	}
	if !strings.Contains(text, "top=3000px") {
		t.Errorf("selector note should carry the measured offset, got %q", text)
	}
	// The bytes must be in the store under the reported id, because that is
	// the ONLY thing replay has: it never re-renders.
	data, mime, err := readImageFile(s.Cwd, id)
	if err != nil {
		t.Fatalf("readImageFile(%s): %v", id, err)
	}
	if mime != "image/png" || !strings.HasPrefix(string(data), "\x89PNG") {
		t.Errorf("stored file is not a PNG: mime=%q head=%q", mime, data[:min(8, len(data))])
	}
}

// TestScreenshotReplayUsesStoredBytes: replay must rebuild the parts from
// ImageID, never by re-running the tool. Re-rendering a page that changed since
// would put different bytes in the middle of the prompt and reprocess every
// message behind them.
func TestScreenshotReplayUsesStoredBytes(t *testing.T) {
	a, s := newTestAgent(t)
	a.imagesSupported = true
	id := "img_deadbeef"
	if err := writeImageFile(s.Cwd, id, "image/png", []byte("stored pixels")); err != nil {
		t.Fatal(err)
	}
	tu := ToolUse{
		ID:      "tu_1",
		Name:    "screenshot",
		Input:   `{"path":"gone.html"}`,
		Output:  "[Screenshot of gone.html (400x400) attached as img_deadbeef.]",
		ImageID: id,
	}

	got := a.replayToolOutput(s, tu)
	parts, ok := got.([]any)
	if !ok {
		t.Fatalf("replay returned %T, want multimodal parts", got)
	}
	// parts[0] is the STORED output verbatim, not a re-derived truncation
	// hint: anything else changes wire bytes the model already saw.
	if want := imageParts(tu.Output, "image/png", []byte("stored pixels")); !reflect.DeepEqual(parts, want) {
		t.Errorf("replayed parts differ from a live call:\ngot  %v\nwant %v", parts, want)
	}

	// Bytes deleted out from under us → fall back to text, don't fail the turn.
	matches, err := filepath.Glob(filepath.Join(s.Cwd, ".codehalter", "images", id+".*"))
	if err != nil || len(matches) == 0 {
		t.Fatalf("glob stored image: %v %v", matches, err)
	}
	if err := os.Remove(matches[0]); err != nil {
		t.Fatal(err)
	}
	if _, ok := a.replayToolOutput(s, tu).([]any); ok {
		t.Error("missing bytes should replay as text, not as parts")
	}
}

// TestScreenshotFallbackWhenModelIsBlind: with imagesSupported=false the
// registered tool still exists (the tools array is a byte-identical superset
// across phases), but dispatching it takes the fallback and fails loudly
// instead of burning a browser launch to deliver something the model cannot
// see. Goes through executeTool so the registration itself is covered.
func TestScreenshotFallbackWhenModelIsBlind(t *testing.T) {
	a, s := newTestAgent(t)
	a.imagesSupported = false
	page := filepath.Join(s.Cwd, "page.html")
	if err := os.WriteFile(page, []byte("<p>hi</p>"), 0o600); err != nil {
		t.Fatal(err)
	}

	tc := llm.ToolCall{}
	tc.Function.Name = "screenshot"
	tc.Function.Arguments = `{"path":"page.html"}`
	text, failed := a.executeTool(context.Background(), s.ID, tc)
	if !failed {
		t.Fatalf("failed=false, text=%q", text)
	}
	if strings.Contains(text, "unknown tool") {
		t.Fatalf("screenshot is not registered: %q", text)
	}
	if !strings.Contains(text, "image inputs") {
		t.Errorf("text = %q, want it to name the missing capability", text)
	}
}
