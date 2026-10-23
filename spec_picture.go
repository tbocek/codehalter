package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
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

// specPictureSpecBytes caps the spec text sent with the two images: a whole chapter fits.
const specPictureSpecBytes = 64 * 1024

type specPictureVerdict struct {
	Verdict     string   `json:"verdict"`
	Differences []string `json:"differences"`
}

// checkPictures renders each of the item's spec pictures with the snapshot recipe and asks a
// fresh call, which has no stake in the round, whether the render's arrangement matches. Being
// shown the picture is not matching it: a round looked at a page cut off at the top, an empty
// band under the run bar and labels drawn over each other, called it "content, not layout", and
// passed. It returns one line per picture that differs; a look that could not be had is said and
// not counted, so a broken verifier never costs the item an attempt.
//
// A render that matches is remembered by its bytes (specConfig.Pictures). With onlyChanged, the
// check is of a done item's screens: a picture that never matched is not rendered, and a render
// with the bytes that last matched is not looked at, so an unchanged screen costs one render and
// no call. Rounds on one screen had broken another (a Cut round lost the Prepare rows' ✂), and
// only that screen's own next round saw it, which broke a third.
func (r *specRun) checkPictures(ctx context.Context, item string, pics []string, onlyChanged bool) []string {
	if len(pics) == 0 || !r.a.imagesSupported.Load() {
		return nil
	}
	r.a.cfgMu.RLock()
	conn := r.a.settings.ConnAt(0, "execute")
	r.a.cfgMu.RUnlock()
	if conn == nil {
		return nil
	}
	// The item's whole chapter, not only its section: a window's picture shows parts other
	// sections describe (the run bar, a sources list whose frame-size control the rewrite drops),
	// and the check had listed those as missing with the sentence that drops them out of sight.
	// A chapter over the cap keeps the item's own section whole and shortens the rest.
	doc := r.idx.docs[r.idx.items[item].Doc]
	section := strings.Join(doc.lines, "\n")
	// …and the sections the item links in other chapters: the Prepare screen's source rows are
	// specified in the shell's chapter, whose "prototype: ✂ … so it is not reused" the check had
	// not seen, and it listed the ✂ the rewrite replaces as missing.
	sl := r.idx.slice(item, r.cfg.SpecDir)
	if _, linked, ok := strings.Cut(sl.Text, "\n\n--- "); ok {
		for _, block := range strings.Split("--- "+linked, "\n\n--- ") {
			if !strings.HasPrefix(strings.TrimPrefix(block, "--- "), r.cfg.SpecDir+"/"+doc.rel+" ") {
				section += "\n\n--- " + strings.TrimPrefix(block, "--- ")
			}
		}
	}
	if len(section) > specPictureSpecBytes {
		own := clipBytes(sl.Text, specPictureSpecBytes/2)
		section = own + "\n\n--- the rest of " + doc.rel + ", shortened:\n\n" + clipBytes(section, specPictureSpecBytes-len(own))
	}
	var differs []string
	for _, pic := range pics {
		if onlyChanged && r.cfg.Pictures[pic] == "" {
			continue
		}
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
		renderData, err := os.ReadFile(render)
		if err != nil {
			r.say(ctx, fmt.Sprintf("⚠ picture check of %s could not read the render %s, so it does not count against %s: %v\n", pic, renderRel, item, err))
			continue
		}
		sum := sha256.Sum256(renderData)
		renderHash := hex.EncodeToString(sum[:])
		if onlyChanged && r.cfg.Pictures[pic] == renderHash {
			continue
		}
		v, err := r.lookAtPicture(ctx, conn, pic, renderRel, renderData, section)
		switch {
		case err != nil:
			r.a.logSession(r.sid, "PICTURE", "%s beside %s: no verdict: %v", pic, renderRel, err)
			r.say(ctx, fmt.Sprintf("⚠ picture check of %s could not be had, so it does not count against %s: %v\n", pic, item, err))
		case strings.EqualFold(v.Verdict, "MATCH"):
			if r.cfg.Pictures == nil {
				r.cfg.Pictures = map[string]string{}
			}
			r.cfg.Pictures[pic] = renderHash
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
func (r *specRun) lookAtPicture(ctx context.Context, conn *LLMConnection, pic, renderRel string, renderData []byte, section string) (specPictureVerdict, error) {
	var v specPictureVerdict
	picData, err := os.ReadFile(filepath.Join(r.sess.Cwd, pic))
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
