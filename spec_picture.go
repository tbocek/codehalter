package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// specPictureTimeout bounds one look; the main model may think before it answers.
const specPictureTimeout = 10 * time.Minute

// specPictureSectionBytes caps the spec text sent with the two images.
const specPictureSectionBytes = 24 * 1024

type specPictureVerdict struct {
	Verdict     string   `json:"verdict"`
	Differences []string `json:"differences"`
}

// checkPictures renders each of the item's own spec pictures with the snapshot recipe and asks a
// fresh call, which has no stake in the round, whether the render's arrangement matches. Being
// shown the picture is not matching it: a round looked at a page cut off at the top, an empty
// band under the run bar and labels drawn over each other, called it "content, not layout", and
// passed. It returns one line per picture that differs; a look that could not be had is said and
// not counted, so a broken verifier never costs the item an attempt.
func (r *specRun) checkPictures(ctx context.Context, item string, pics []string) []string {
	if len(pics) == 0 || !r.a.imagesSupported.Load() {
		return nil
	}
	r.a.cfgMu.RLock()
	conn := r.a.settings.ConnAt(0, "execute")
	r.a.cfgMu.RUnlock()
	if conn == nil {
		return nil
	}
	section := clipBytes(r.idx.slice(item, r.cfg.SpecDir).Text, specPictureSectionBytes)
	var differs []string
	for _, pic := range pics {
		stem := strings.TrimSuffix(filepath.Base(pic), filepath.Ext(pic))
		// The round's own render may be older than its last edit: this one is of the code as committed.
		since := time.Now()
		cmd := "just snapshot " + stem
		ok, tail, _ := r.a.runCheckCommand(ctx, r.sid, r.outAbs, fmt.Sprintf("Rendering %s for the picture check", stem), cmd)
		if ctx.Err() != nil {
			return differs
		}
		render := ""
		for _, p := range pngsWrittenSince(r.outAbs, since) {
			if strings.EqualFold(strings.TrimSuffix(filepath.Base(p), filepath.Ext(p)), stem) {
				render = p
				break
			}
		}
		if !ok || render == "" {
			differs = append(differs, fmt.Sprintf("%s: `%s` in %s/ wrote no %s.png, so the screen could not be put beside its picture. Its output ends:\n```\n%s\n```",
				pic, cmd, r.cfg.OutDir, stem, strings.TrimSpace(tailUTF8(tail, 1200))))
			continue
		}
		renderRel, err := filepath.Rel(r.sess.Cwd, render)
		if err != nil {
			renderRel = render
		}
		renderRel = filepath.ToSlash(renderRel)
		v, err := r.lookAtPicture(ctx, conn, pic, renderRel, render, section)
		switch {
		case err != nil:
			r.a.logSession(r.sid, "PICTURE", "%s beside %s: no verdict: %v", pic, renderRel, err)
			r.say(ctx, fmt.Sprintf("⚠ picture check of %s could not be had, so it does not count against %s: %v\n", pic, item, err))
		case strings.EqualFold(v.Verdict, "MATCH"):
			r.a.logSession(r.sid, "PICTURE", "%s beside %s: MATCH", pic, renderRel)
			r.say(ctx, fmt.Sprintf("🖼 %s matches its picture %s\n", renderRel, pic))
		default:
			found := strings.Join(v.Differences, "; ")
			if strings.TrimSpace(found) == "" {
				found = "the check found the arrangement differs but named no difference: put the render beside the picture and compare region by region"
			}
			r.a.logSession(r.sid, "PICTURE", "%s beside %s: MISMATCH: %s", pic, renderRel, found)
			r.say(ctx, fmt.Sprintf("🖼 %s differs from its picture %s: %s\n", renderRel, pic, found))
			differs = append(differs, fmt.Sprintf("%s beside %s: %s", renderRel, pic, found))
		}
	}
	return differs
}

// lookAtPicture asks for one verdict on the spec's picture and the render, both attached.
func (r *specRun) lookAtPicture(ctx context.Context, conn *LLMConnection, pic, renderRel, renderAbs, section string) (specPictureVerdict, error) {
	var v specPictureVerdict
	picData, err := os.ReadFile(filepath.Join(r.sess.Cwd, pic))
	if err != nil {
		return v, err
	}
	renderData, err := os.ReadFile(renderAbs)
	if err != nil {
		return v, err
	}
	size := ""
	if c, _, err := image.DecodeConfig(bytes.NewReader(renderData)); err == nil {
		size = fmt.Sprintf(" (%dx%d px)", c.Width, c.Height)
	}
	mime := "image/png"
	switch strings.ToLower(filepath.Ext(pic)) {
	case ".jpg", ".jpeg":
		mime = "image/jpeg"
	case ".webp":
		mime = "image/webp"
	}
	prompt := r.a.renderSpecPrompt(r.sid, "SPEC-PICTURE.md", []string{"{{picture}}", pic, "{{render}}", renderRel, "{{section}}", section})
	msgs := []llmMessage{{Role: "user", Content: []any{
		map[string]any{"type": "text", "text": prompt},
		map[string]any{"type": "text", "text": "1. The spec's picture, " + pic + ":"},
		imageURLPart(mime, downscalePNG(picData, attachMaxSide)),
		map[string]any{"type": "text", "text": "2. The program's render, " + renderRel + size + ":"},
		imageURLPart("image/png", downscalePNG(renderData, attachMaxSide)),
	}}}
	tctx, cancel := context.WithTimeout(ctx, specPictureTimeout)
	defer cancel()
	out, _, _, err := r.a.llmStream(tctx, r.sid, conn, msgs, nil, nil, nil, nil)
	if err != nil {
		return v, err
	}
	// The object is the answer; a model may still talk around it.
	start, end := strings.Index(out, "{"), strings.LastIndex(out, "}")
	if start < 0 || end < start {
		return v, fmt.Errorf("the reply holds no JSON object: %q", clipBytes(strings.TrimSpace(out), 400))
	}
	if err := json.Unmarshal([]byte(out[start:end+1]), &v); err != nil {
		return v, fmt.Errorf("the reply's JSON does not parse (%v): %q", err, clipBytes(out[start:end+1], 400))
	}
	if !strings.EqualFold(v.Verdict, "MATCH") && !strings.EqualFold(v.Verdict, "MISMATCH") {
		return v, fmt.Errorf("the verdict is %q, neither MATCH nor MISMATCH", v.Verdict)
	}
	return v, nil
}
