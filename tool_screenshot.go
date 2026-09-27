package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// Replay rebuilds a screenshot from the stored bytes (ToolUse.ImageID), never by
// re-rendering: a changed image mid-prompt would reprocess everything after it.

const (
	screenshotDefaultWidth  = 1400
	screenshotDefaultHeight = 1200
	screenshotMaxWidth      = 2560
	screenshotMaxHeight     = 4000
	// Past this the model should scope with `selector`; it bounds an attached
	// render too.
	maxScreenshotBytes = 4 << 20
	screenshotTimeout  = 90 * time.Second
	// Page pixels left above a `selector` element.
	screenshotMargin = 40
)

var (
	headOpenRe  = regexp.MustCompile(`(?i)<head[^>]*>`)
	bodyCloseRe = regexp.MustCompile(`(?i)</body>`)
)

var screenshotTool = Tool{
	Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": "screenshot",
			"description": "Render a local HTML/SVG/image file with headless Firefox and look at the result. " +
				"Use it to CHECK a rendering you changed (CSS, template, generated page) instead of asking the user whether it looks right. " +
				"Point `path` at the BUILT file the user sees, not the source template it was generated from. " +
				"`selector` puts one element at the top of the shot: that is how you inspect a block of a long page without pushing a huge image into the prompt. " +
				"A screenshot tells you WHETHER something is off, not by how much: for a distance, measure it as a number instead.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path"},
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Project-relative path of the file to render, e.g. `out/index.html`. Must be inside the project; there is no URL mode.",
					},
					"region": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "integer"},
						"description": "Only for a picture file (.png, .jpg): [x, y, width, height] in the picture's own pixels, returned enlarged, for a close look at one part of a screen: {\"path\": \"shots/06-lane.png\", \"region\": [0, 560, 1500, 260]}. The reply for a whole picture states its size in pixels. Use this instead of cropping with a script.",
					},
					"selector": map[string]any{
						"type":        "string",
						"description": "Optional CSS selector. Its first match is scrolled to the top of the shot, and the reply says whether it matched and where it sits.",
					},
					"width": map[string]any{
						"type":        "integer",
						"description": fmt.Sprintf("Viewport width in px. Default %d, max %d.", screenshotDefaultWidth, screenshotMaxWidth),
					},
					"height": map[string]any{
						"type":        "integer",
						"description": fmt.Sprintf("Viewport height in px. Default %d, max %d. A taller shot captures more of the page and costs more; prefer `selector`.", screenshotDefaultHeight, screenshotMaxHeight),
					},
				},
			},
		},
	},
	// Only a fallback: real success goes through dispatchScreenshot.
	Execute: screenshotExecuteFallback,
}

// screenshotExecuteFallback is reached when the LLM takes no images: tell the
// user, who can fix the config, rather than burn a browser launch.
func screenshotExecuteFallback(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	if !a.imagesSupported {
		a.say(ctx, sid, "❕ The model asked for a screenshot, but this LLM reports no image support, so it can't be shown one. If your model does accept images, set `image_support = true` on its [[llm]] entry in settings.toml.\n")
		return "screenshot: this LLM doesn't accept image inputs, so the rendering can't be delivered (the user has been told). Verify it as a NUMBER instead: render the page and print the measurement as text.", true
	}
	return "screenshot: internal: dispatch missed the intercept. Try once more; if it persists, verify the rendering as a number instead.", true
}

// dispatchScreenshot's text is both ToolUse.Output and parts[0], so a replay
// from the stored output and bytes is byte-identical to the live call.
func dispatchScreenshot(ctx context.Context, a *agent, sid string, rawArgs string) (string, []any, string, bool) {
	sess := a.getSession(sid)
	if sess == nil {
		return "screenshot: no session", nil, "", true
	}
	args := parseArgs(rawArgs)
	rel := strings.TrimSpace(args.str("path"))
	if rel == "" {
		return "screenshot: missing `path`. Pass the project-relative path of the file to render, e.g. `out/index.html`.", nil, "", true
	}
	abs, err := a.resolvePath(sid, rel)
	if err != nil {
		return "screenshot: " + err.Error(), nil, "", true
	}
	if info, err := os.Stat(abs); err != nil || info.IsDir() {
		return fmt.Sprintf("screenshot: %s is not a readable file. Only files inside the project can be rendered; there is no URL mode.", rel), nil, "", true
	}
	switch strings.ToLower(filepath.Ext(abs)) {
	case ".png", ".jpg", ".jpeg":
		return a.viewPicture(ctx, sid, sess, rel, abs, args)
	}
	bin, err := findFirefox()
	if err != nil {
		return "screenshot: " + err.Error() + ". Install it (the OS skill has the package name) and call screenshot again, or verify the rendering as a number instead.", nil, "", true
	}
	width := clampDimension(args, "width", screenshotDefaultWidth, screenshotMaxWidth)
	height := clampDimension(args, "height", screenshotDefaultHeight, screenshotMaxHeight)
	selector := strings.TrimSpace(args.str("selector"))

	title := fmt.Sprintf("Screenshot: %s (%dx%d)", rel, width, height)
	tcID := a.StartToolCall(ctx, sid, title, "read", []ToolCallLocation{{Path: rel}})

	png, note, err := renderPage(ctx, bin, abs, selector, width, height)
	if err != nil {
		msg := "screenshot: " + err.Error()
		a.FailToolCall(ctx, sid, tcID, msg)
		return msg, nil, "", true
	}
	id, err := storeImage(sess.Cwd, "image/png", png)
	if err != nil {
		msg := "screenshot: rendered but could not be stored: " + err.Error()
		a.FailToolCall(ctx, sid, tcID, msg)
		return msg, nil, "", true
	}
	text := fmt.Sprintf("[Screenshot of %s (%dx%d) attached as %s.%s]", rel, width, height, id, note)
	a.CompleteToolCall(ctx, sid, tcID, []ToolCallContent{TextContent(fmt.Sprintf("%s, %d KiB%s", id, len(png)/1024, note))})
	return text, imageParts(text, "image/png", png), id, false
}

// imageParts is shared by live dispatch and history replay, which keeps a
// replayed turn byte-identical and the prefix cache intact.
func imageParts(text, mime string, data []byte) []any {
	return []any{map[string]any{"type": "text", "text": text}, imageURLPart(mime, data)}
}

func imageURLPart(mime string, data []byte) map[string]any {
	return map[string]any{
		"type": "image_url",
		"image_url": map[string]string{
			"url": fmt.Sprintf("data:%s;base64,%s", mime, base64.StdEncoding.EncodeToString(data)),
		},
	}
}

func clampDimension(args toolArgs, key string, def, max int) int {
	n, ok := args.num(key)
	if !ok || n <= 0 {
		return def
	}
	if n > max {
		return max
	}
	return n
}

// renderPage never writes to the project tree, not even to instrument the page.
func renderPage(ctx context.Context, bin, page, selector string, width, height int) ([]byte, string, error) {
	tmp, err := os.MkdirTemp("", "codehalter-shot-")
	if err != nil {
		return nil, "", err
	}
	defer func() {
		if err := os.RemoveAll(tmp); err != nil {
			slog.Debug("screenshot: could not remove temp dir", "dir", tmp, "err", err)
		}
	}()
	// Without its own --profile (and --no-remote --new-instance) Firefox attaches
	// to an already-running instance and hangs until the timeout.
	profile := filepath.Join(tmp, "profile")
	if err := os.Mkdir(profile, 0o700); err != nil {
		return nil, "", err
	}
	shot := filepath.Join(tmp, "shot.png")

	target, note := page, ""
	var b *beacon
	if selector != "" {
		b, err = startBeacon()
		if err != nil {
			return nil, "", err
		}
		defer b.Close()
		target, err = writeInstrumented(tmp, page, selector, b.URL())
		if err != nil {
			return nil, "", err
		}
	}

	runCtx, cancel := context.WithTimeout(ctx, screenshotTimeout)
	defer cancel()
	cmd := exec.CommandContext(runCtx, bin,
		"--headless", "--no-remote", "--new-instance",
		"--profile", profile,
		"--window-size", fmt.Sprintf("%d,%d", width, height),
		"--screenshot", shot,
		"file://"+target)
	// HOME in the temp dir, so a first run cannot scatter ~/.mozilla state.
	cmd.Env = append(os.Environ(), "HOME="+tmp, "MOZ_HEADLESS=1")
	if out, err := cmd.CombinedOutput(); err != nil {
		return nil, "", fmt.Errorf("firefox failed: %v (%s)", err, truncate(strings.TrimSpace(string(out)), 300))
	}

	data, err := os.ReadFile(shot)
	if err != nil {
		return nil, "", fmt.Errorf("firefox exited 0 but wrote no screenshot: %v", err)
	}
	if len(data) == 0 {
		return nil, "", errors.New("firefox wrote an empty screenshot (out of disk space?)")
	}
	if len(data) > maxScreenshotBytes {
		return nil, "", fmt.Errorf("screenshot is %d KiB, over the %d KiB cap. Narrow it with `selector` or a smaller `height`",
			len(data)/1024, maxScreenshotBytes/1024)
	}
	if b != nil {
		note = " " + selectorNote(selector, b.report())
	}
	return data, note, nil
}

// selectorNote exists for the selector that matched nothing: otherwise the model
// gets the top of the page with no reason to doubt it.
func selectorNote(selector, report string) string {
	switch {
	case strings.HasPrefix(report, "matched=1"):
		return fmt.Sprintf("Selector %q is at the top of the shot (%s).", selector, strings.TrimSpace(strings.TrimPrefix(report, "matched=1")))
	case report == "matched=0":
		return fmt.Sprintf("Selector %q matched NO element, so this is the top of the page, not what you asked for.", selector)
	default:
		return fmt.Sprintf("Selector %q could not be resolved (the page reported nothing), so this is the top of the page.", selector)
	}
}

func writeInstrumented(dir, page, selector, beaconURL string) (string, error) {
	raw, err := os.ReadFile(page)
	if err != nil {
		return "", err
	}
	selJSON, err := json.Marshal(selector)
	if err != nil {
		return "", err
	}
	urlJSON, err := json.Marshal(beaconURL)
	if err != nil {
		return "", err
	}
	html := string(raw)
	base := fmt.Sprintf("<base href=\"file://%s/\">", filepath.Dir(page))
	if loc := headOpenRe.FindStringIndex(html); loc != nil {
		html = html[:loc[1]] + base + html[loc[1]:]
	} else {
		html = base + html
	}
	// translateY, not scrollTop: --screenshot renders from the document top
	// however it was scrolled. The XHR is synchronous so it beats Firefox's exit.
	script := fmt.Sprintf(`<script>window.addEventListener('load', function () {
  var el = document.querySelector(%s), out = 'matched=0';
  if (el) {
    var r = el.getBoundingClientRect(), top = r.top + window.scrollY;
    document.documentElement.style.transform = 'translateY(' + (%d - top) + 'px)';
    out = 'matched=1 top=' + top.toFixed(0) + 'px height=' + r.height.toFixed(0) + 'px';
  }
  var x = new XMLHttpRequest();
  x.open('GET', %s + '?' + encodeURIComponent(out), false);
  try { x.send(); } catch (e) {}
});</script>`, selJSON, screenshotMargin, urlJSON)
	if loc := bodyCloseRe.FindStringIndex(html); loc != nil {
		html = html[:loc[0]] + script + html[loc[0]:]
	} else {
		html += script
	}
	out := filepath.Join(dir, "page.html")
	if err := os.WriteFile(out, []byte(html), 0o600); err != nil {
		return "", err
	}
	return out, nil
}

// beacon gets a value out of headless Firefox, which has no --dump-dom: the page
// GETs a loopback URL with the payload as its query string.
type beacon struct {
	ln net.Listener
	ch chan string
}

func startBeacon() (*beacon, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("could not open the loopback listener the page reports to: %w", err)
	}
	b := &beacon{ln: ln, ch: make(chan string, 1)}
	srv := &http.Server{
		ReadHeaderTimeout: 5 * time.Second,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			select {
			case b.ch <- r.URL.RawQuery:
			default: // first report wins
			}
			w.Header().Set("Content-Length", "0")
			w.WriteHeader(http.StatusOK)
		}),
	}
	go func() {
		if err := srv.Serve(ln); err != nil && !errors.Is(err, net.ErrClosed) {
			slog.Debug("screenshot beacon stopped", "err", err)
		}
	}()
	return b, nil
}

func (b *beacon) URL() string { return "http://" + b.ln.Addr().String() + "/" }

func (b *beacon) Close() {
	if err := b.ln.Close(); err != nil {
		slog.Debug("screenshot beacon close", "err", err)
	}
}

// report does not block: the XHR is synchronous, so once Firefox has exited the
// payload is already in the channel.
func (b *beacon) report() string {
	select {
	case q := <-b.ch:
		v, err := url.QueryUnescape(q)
		if err != nil {
			return q
		}
		return v
	default:
		return ""
	}
}

// attachRenderedScreen returns an empty id when there is nothing to attach. It
// stores and builds parts like screenshot, so replay rebuilds the same message.
func (a *agent) attachRenderedScreen(ctx context.Context, sid, rawArgs, result string, since time.Time) (string, []any, string) {
	cmd := parseArgs(rawArgs).str("command")
	if !strings.Contains(cmd, "snapshot") || !strings.HasPrefix(result, "exit 0\n") {
		return "", nil, ""
	}
	sess := a.getSession(sid)
	if sess == nil {
		return "", nil, ""
	}
	// Whole-second mtimes on some filesystems: a file written in the same
	// second the command started must still count.
	png := newestPNGSince(sess.Cwd, since.Truncate(time.Second).Add(-time.Nanosecond))
	if png == "" {
		return "", nil, ""
	}
	data, err := os.ReadFile(png)
	if err != nil || len(data) == 0 || len(data) > maxScreenshotBytes {
		return "", nil, ""
	}
	data = downscalePNG(data, attachMaxSide)
	id, err := storeImage(sess.Cwd, "image/png", data)
	if err != nil {
		slog.Debug("attachRenderedScreen: could not store the render", "png", png, "err", err)
		return "", nil, ""
	}
	rel, _ := filepath.Rel(sess.Cwd, png)
	rel = filepath.ToSlash(rel)
	size := ""
	if cfg, _, err := image.DecodeConfig(bytes.NewReader(data)); err == nil {
		size = fmt.Sprintf(", %dx%d px as attached", cfg.Width, cfg.Height)
	}
	text := result + fmt.Sprintf("\n[codehalter, not the user: the screen this command rendered, %s%s, is attached below as %s. For a closer look at one part, call screenshot on that file with \"region\": [x, y, width, height] in the file's own pixels; never crop with a script. Look at it now, before anything else: is every widget the spec names on it, in the order it says; is anything empty, overlapping, cut off or unlabeled? Where the spec has its own picture of this screen (its image with the same name), look at that too with `screenshot` and compare widgets, order and labels, not pixels. Fix what you see, then render again.]", rel, size, id)
	a.say(ctx, sid, fmt.Sprintf("👁 attached the rendered screen %s to the model's view\n", rel))
	return text, imageParts(text, "image/png", data), id
}

// attachMaxSide halves a HiDPI (2x) snapshot back to window size, where text is
// still sharp, at a quarter of the vision cost.
const attachMaxSide = 1600

// downscalePNG uses a whole factor, so a 2x render comes back at exactly 1x.
func downscalePNG(data []byte, max int) []byte {
	src, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		return data
	}
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	f := 1
	for (w+f-1)/f > max || (h+f-1)/f > max {
		f++
	}
	if f == 1 {
		return data
	}
	nw, nh := w/f, h/f
	dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
	n := uint32(f * f)
	for y := 0; y < nh; y++ {
		for x := 0; x < nw; x++ {
			var r, g, bl, a uint32
			for dy := 0; dy < f; dy++ {
				for dx := 0; dx < f; dx++ {
					cr, cg, cb, ca := src.At(b.Min.X+x*f+dx, b.Min.Y+y*f+dy).RGBA()
					r, g, bl, a = r+cr, g+cg, bl+cb, a+ca
				}
			}
			dst.Set(x, y, color.RGBA64{uint16(r / n), uint16(g / n), uint16(bl / n), uint16(a / n)})
		}
	}
	var out bytes.Buffer
	if err := png.Encode(&out, dst); err != nil {
		return data
	}
	return out.Bytes()
}

func newestPNGSince(root string, t time.Time) string {
	best, bestT := "", t
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			name := d.Name()
			if path != root && skipWalkDir(name) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.EqualFold(filepath.Ext(path), ".png") {
			return nil
		}
		if info, err := d.Info(); err == nil && info.ModTime().After(bestT) {
			best, bestT = path, info.ModTime()
		}
		return nil
	})
	return best
}

const pictureRegionMaxSide = 1400

func (a *agent) viewPicture(ctx context.Context, sid string, sess *Session, rel, abs string, args toolArgs) (string, []any, string, bool) {
	data, err := os.ReadFile(abs)
	if err != nil {
		return "screenshot: " + err.Error(), nil, "", true
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return fmt.Sprintf("screenshot: %s could not be decoded as an image: %v", rel, err), nil, "", true
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	what := fmt.Sprintf("%s (%dx%d px)", rel, w, h)
	var out []byte
	if r, ok := args["region"].([]any); ok {
		if len(r) != 4 {
			return "screenshot: `region` is [x, y, width, height] in the picture's pixels, four numbers.", nil, "", true
		}
		ta := toolArgs{"x": r[0], "y": r[1], "w": r[2], "h": r[3]}
		x, _ := ta.num("x")
		y, _ := ta.num("y")
		rw, _ := ta.num("w")
		rh, _ := ta.num("h")
		cut := image.Rect(x, y, x+rw, y+rh).Intersect(image.Rect(0, 0, w, h))
		if cut.Empty() {
			return fmt.Sprintf("screenshot: region %v lies outside %s; the picture is %dx%d.", r, rel, w, h), nil, "", true
		}
		f := 1
		for (f+1)*max(cut.Dx(), cut.Dy()) <= pictureRegionMaxSide {
			f++
		}
		dst := image.NewRGBA(image.Rect(0, 0, cut.Dx()*f, cut.Dy()*f))
		for yy := 0; yy < dst.Bounds().Dy(); yy++ {
			for xx := 0; xx < dst.Bounds().Dx(); xx++ {
				dst.Set(xx, yy, img.At(b.Min.X+cut.Min.X+xx/f, b.Min.Y+cut.Min.Y+yy/f))
			}
		}
		var buf bytes.Buffer
		if err := png.Encode(&buf, dst); err != nil {
			return "screenshot: " + err.Error(), nil, "", true
		}
		out = buf.Bytes()
		what = fmt.Sprintf("%s, region x %d-%d, y %d-%d, enlarged %dx", what, cut.Min.X, cut.Max.X, cut.Min.Y, cut.Max.Y, f)
	} else {
		var buf bytes.Buffer
		if err := png.Encode(&buf, img); err != nil {
			return "screenshot: " + err.Error(), nil, "", true
		}
		out = downscalePNG(buf.Bytes(), attachMaxSide)
	}
	id, err := storeImage(sess.Cwd, "image/png", out)
	if err != nil {
		return "screenshot: could not be stored: " + err.Error(), nil, "", true
	}
	text := fmt.Sprintf("[Picture %s attached as %s. For a closer look at one part, pass \"region\": [x, y, width, height] in these pixels; it comes back enlarged.]", what, id)
	tcID := a.StartToolCall(ctx, sid, "Picture: "+what, "read", []ToolCallLocation{{Path: rel}})
	a.CompleteToolCall(ctx, sid, tcID, []ToolCallContent{TextContent(id)})
	return text, imageParts(text, "image/png", out), id, false
}
