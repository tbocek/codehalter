package main

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// Absent and outside-project paths fail with an actionable message and launch no
// browser.
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

// The fallback tells the user in the transcript: only they can fix settings.toml.
func TestScreenshotNoVisionTellsTheUser(t *testing.T) {
	a, s := newTestAgent(t)
	a.imagesSupported = false

	text, failed := screenshotExecuteFallback(context.Background(), a, s.ID, `{"path":"out/index.html"}`)
	if !failed {
		t.Fatalf("failed=false, text=%q", text)
	}
	if !strings.Contains(text, "NUMBER") {
		t.Errorf("model-facing text should point at numeric verification, got %q", text)
	}
	if !strings.Contains(text, "the user has been told") {
		t.Errorf("model-facing text should say the user was informed, got %q", text)
	}
}

// <base> points at the ORIGINAL directory, and the original file is untouched.
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

// No <head> and no </body>: both injections still land, at the ends.
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

func TestClampDimension(t *testing.T) {
	cases := []struct {
		args string
		want int
	}{
		{`{}`, screenshotDefaultWidth},
		{`{"width":0}`, screenshotDefaultWidth},
		{`{"width":-5}`, screenshotDefaultWidth},
		{`{"width":"nope"}`, screenshotDefaultWidth},
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

func TestScreenshotEndToEnd(t *testing.T) {
	if _, err := findFirefox(); err != nil {
		t.Skipf("no firefox: %v", err)
	}
	if testing.Short() {
		t.Skip("launches a browser")
	}
	a, s := newTestAgent(t)
	a.imagesSupported = true
	page := filepath.Join(s.Cwd, "page.html")
	// Well below the fold: without the translateY shift the capture misses it.
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
	// The stored bytes are the ONLY thing replay has: it never re-renders.
	data, mime, err := readImageFile(s.Cwd, id)
	if err != nil {
		t.Fatalf("readImageFile(%s): %v", id, err)
	}
	if mime != "image/png" || !strings.HasPrefix(string(data), "\x89PNG") {
		t.Errorf("stored file is not a PNG: mime=%q head=%q", mime, data[:min(8, len(data))])
	}
}

func TestScreenshotReplayUsesStoredBytes(t *testing.T) {
	a, s := newTestAgent(t)
	a.imagesSupported = true
	id, err := storeImage(s.Cwd, "image/png", []byte("stored pixels"))
	if err != nil {
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
	// parts[0] is the stored output verbatim: anything else changes bytes the
	// model already saw.
	if want := imageParts(tu.Output, "image/png", []byte("stored pixels")); !reflect.DeepEqual(parts, want) {
		t.Errorf("replayed parts differ from a live call:\ngot  %v\nwant %v", parts, want)
	}

	// Bytes gone from the store: fall back to text, do not fail the turn.
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

// Goes through runToolCall so the registration itself is covered.
func TestScreenshotFallbackWhenModelIsBlind(t *testing.T) {
	a, s := newTestAgent(t)
	a.imagesSupported = false
	page := filepath.Join(s.Cwd, "page.html")
	if err := os.WriteFile(page, []byte("<p>hi</p>"), 0o600); err != nil {
		t.Fatal(err)
	}

	tc := toolCall{}
	tc.Function.Name = "screenshot"
	tc.Function.Arguments = `{"path":"page.html"}`
	tu, _ := a.runToolCall(context.Background(), s.ID, tc)
	text, failed := tu.Output, tu.Failed
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

// A failed render, or a command that is not a render, attaches nothing.
func TestRunCommandAttachesRenderedScreen(t *testing.T) {
	h := newTerminalHarness(t)
	h.agent.imagesSupported = true
	h.agent.tools.add(Tool{Def: map[string]any{"type": "function", "function": map[string]any{"name": "run_command", "parameters": map[string]any{"type": "object"}}}, Execute: runCmdExecute})
	shots := filepath.Join(h.sess.Cwd, "rust", "shots")
	if err := os.MkdirAll(shots, 0o755); err != nil {
		t.Fatal(err)
	}
	// Any bytes will do: the attachment is the file's content, unread.
	png := "PNG-BYTES"
	var tc toolCall
	tc.ID, tc.Function.Name = "c1", "run_command"
	tc.Function.Arguments = `{"command":"printf '` + png + `' > rust/shots/05-cut.png; echo 'just snapshot 05-cut -> rust/shots/05-cut.png'"}`
	tu, content := h.agent.runToolCall(context.Background(), h.sess.ID, tc)
	if tu.ImageID == "" || !strings.Contains(tu.Output, "attached below as "+tu.ImageID) || !strings.Contains(tu.Output, "rust/shots/05-cut.png") {
		t.Fatalf("no render attached: %+v", tu)
	}
	parts, ok := content.([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("content = %T %v, want text + image parts", content, content)
	}
	if got := uiEditedUnseen([]ToolUse{{Name: "edit_file", Input: `{"path":"rust/src/ui.rs"}`}, tu}, h.sess.Cwd); got != nil {
		t.Errorf("an attached render did not count as a look: %v", got)
	}

	// The same pixels again are a note, not a second copy, and still a look.
	first := tu.ImageID
	tc.ID, tc.Function.Arguments = "c1b", `{"command":"printf '`+png+`' > rust/shots/05-cut.png; echo 'just snapshot 05-cut again'"}`
	again, content := h.agent.runToolCall(context.Background(), h.sess.ID, tc)
	if again.ImageID != "" || again.SameImageAs != first || !strings.Contains(again.Output, "is the same picture as "+first) {
		t.Errorf("an unchanged render was attached again: %+v", again)
	}
	if _, isText := content.(string); !isText {
		t.Errorf("an unchanged render sent %T, want text only", content)
	}
	if got := uiEditedUnseen([]ToolUse{{Name: "edit_file", Input: `{"path":"rust/src/ui.rs"}`}, again}, h.sess.Cwd); got != nil {
		t.Errorf("an unchanged render did not count as a look: %v", got)
	}
	tc.ID, tc.Function.Arguments = "c1c", `{"command":"printf 'OTHER-BYTES' > rust/shots/05-cut.png; echo 'just snapshot 05-cut'"}`
	if changed, _ := h.agent.runToolCall(context.Background(), h.sess.ID, tc); changed.ImageID == "" || changed.ImageID == first {
		t.Errorf("a changed render was not attached: %+v", changed)
	}

	tc.ID, tc.Function.Arguments = "c2", `{"command":"echo snapshot failed; exit 1"}`
	if tu, _ := h.agent.runToolCall(context.Background(), h.sess.ID, tc); tu.ImageID != "" {
		t.Error("a failed render attached a picture")
	}
	tc.ID, tc.Function.Arguments = "c3", `{"command":"printf '`+png+`' > rust/shots/06-lane.png; echo wrote"}`
	if tu, _ := h.agent.runToolCall(context.Background(), h.sess.ID, tc); tu.ImageID != "" {
		t.Error("a command that is not a render attached a picture")
	}
}

// A 2x render comes back at exactly half; a small image and a non-PNG untouched.
func TestDownscalePNG(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 40, 30))
	for y := 0; y < 30; y++ {
		for x := 0; x < 40; x++ {
			if (x+y)%2 == 0 {
				img.Set(x, y, color.White)
			} else {
				img.Set(x, y, color.Black)
			}
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	out := downscalePNG(buf.Bytes(), 20)
	got, err := png.Decode(bytes.NewReader(out))
	if err != nil {
		t.Fatal(err)
	}
	if b := got.Bounds(); b.Dx() != 20 || b.Dy() != 15 {
		t.Errorf("scaled to %dx%d, want 20x15", b.Dx(), b.Dy())
	}
	if r, _, _, _ := got.At(3, 3).RGBA(); r < 0x7000 || r > 0x9000 {
		t.Errorf("a 2x2 checker box averaged to %x, want mid grey", r)
	}
	if same := downscalePNG(buf.Bytes(), 100); !bytes.Equal(same, buf.Bytes()) {
		t.Error("a small image was re-encoded")
	}
	if same := downscalePNG([]byte("PNG-BYTES"), 20); string(same) != "PNG-BYTES" {
		t.Error("undecodable bytes were changed")
	}
}

// A region outside the picture says how big the picture is.
func TestScreenshotPictureRegion(t *testing.T) {
	h := newTerminalHarness(t)
	h.agent.imagesSupported = true
	img := image.NewRGBA(image.Rect(0, 0, 300, 200))
	for y := 0; y < 200; y++ {
		for x := 0; x < 300; x++ {
			if y >= 100 {
				img.Set(x, y, color.Black)
			} else {
				img.Set(x, y, color.White)
			}
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(h.sess.Cwd, "shot.png"), buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	text, parts, id, failed := dispatchScreenshot(context.Background(), h.agent, h.sess.ID, `{"path":"shot.png"}`)
	if failed || id == "" || len(parts) != 2 || !strings.Contains(text, "shot.png (300x200 px)") {
		t.Fatalf("whole picture = %q %v", text, failed)
	}
	text, _, id, failed = dispatchScreenshot(context.Background(), h.agent, h.sess.ID, `{"path":"shot.png","region":[0,90,100,20]}`)
	if failed || !strings.Contains(text, "region x 0-100, y 90-110, enlarged 14x") {
		t.Fatalf("region = %q %v", text, failed)
	}
	data, _, err := readImageFile(h.sess.Cwd, id)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := png.Decode(bytes.NewReader(data))
	if b := got.Bounds(); b.Dx() != 1400 || b.Dy() != 280 {
		t.Errorf("region size = %v, want 1400x280", b)
	}
	if r, _, _, _ := got.At(5, 5).RGBA(); r != 0xffff {
		t.Error("top of the region should be white")
	}
	if r, _, _, _ := got.At(5, 275).RGBA(); r != 0 {
		t.Error("bottom of the region should be black")
	}
	if text, _, _, failed := dispatchScreenshot(context.Background(), h.agent, h.sess.ID, `{"path":"shot.png","region":[500,500,10,10]}`); !failed || !strings.Contains(text, "300x200") {
		t.Errorf("region outside = %q %v", text, failed)
	}
}

// Looking at the same picture again gets a note: the pixels are already in view.
func TestScreenshotOfAPictureAlreadyInView(t *testing.T) {
	a, s := newTestAgent(t)
	a.imagesSupported = true
	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(s.Cwd, "shot.png"), buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	var tc toolCall
	tc.ID, tc.Function.Name, tc.Function.Arguments = "s1", "screenshot", `{"path":"shot.png"}`
	first, content := a.runToolCall(context.Background(), s.ID, tc)
	if _, isParts := content.([]any); first.ImageID == "" || !isParts {
		t.Fatalf("the first look sent no picture: %+v", first)
	}
	tc.ID = "s2"
	again, content := a.runToolCall(context.Background(), s.ID, tc)
	if _, isText := content.(string); !isText || again.ImageID != "" || again.SameImageAs != first.ImageID || !strings.Contains(again.Output, "already in your view") {
		t.Errorf("the same picture was sent again: %+v (%T)", again, content)
	}
}
