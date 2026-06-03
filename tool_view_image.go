package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
)

// view_image redelivers a previously-attached image to the model. The image
// store is content-addressed at <cwd>/.codehalter/images/<id>.<ext>, so a
// reference surviving in Summary (after compaction has rotated the owning
// message out) is enough to fetch the bytes back. Dispatch is special-cased
// in runToolLoop: instead of the usual text Role:"tool" message, the loop
// appends a multimodal Role:"tool" message whose content is [text, image_url]
// — that delivers the bytes within the current turn so the next llmStream
// call sees the image as fresh context (NOT "next turn").

func init() {
	RegisterTool(Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": "view_image",
				"description": "Re-fetch a previously-attached image into the current context. " +
					"Pass the `id` (img_<hex>) surfaced in a `[Image img_… — call view_image id=… to view]` reference. " +
					"References appear in Summary after compaction has rotated the original user turn out, OR alongside any image still in live history when image bytes failed to read from disk. " +
					"Only call this when you actually need to look at the image — every retrieval re-injects the full bytes into the prompt.",
				"parameters": map[string]any{
					"type":     "object",
					"required": []string{"id"},
					"properties": map[string]any{
						"id": map[string]any{
							"type":        "string",
							"description": "The image id from a view_image reference, e.g. `img_a1b2c3d4e5f60718`.",
						},
					},
				},
			},
		},
		// Execute is the fallback path: if a server is configured to disable
		// the in-loop multimodal intercept (e.g. the LLM doesn't support image
		// inputs), this returns a plain-text error rather than silently
		// pretending the bytes were delivered. Real success goes through
		// dispatchViewImage and never reaches here.
		Execute: viewImageExecuteFallback,
	})
}

func viewImageExecuteFallback(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	if !a.imagesSupported {
		return "view_image: this LLM doesn't support image inputs — call other tools (read_file, run_command) to inspect the attachment indirectly.", true
	}
	// imagesSupported and we reached the fallback? The dispatcher should have
	// intercepted. Surface that so the bug isn't silent.
	return "view_image: internal — dispatch missed the intercept. Try again, or use view_output on a prior screenshotting tool if this persists.", true
}

// dispatchViewImage parses view_image arguments, reads the file from the
// content-addressed store, and returns (textForToolUseOutput, multimodalParts,
// failed). The caller wires multimodalParts as the Role:"tool" content. When
// failed=true the multimodal parts slice is empty and textForToolUseOutput is
// the plain-text error to feed back to the model.
func dispatchViewImage(sess *Session, rawArgs string) (string, []any, bool) {
	var args struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
		return fmt.Sprintf("view_image: invalid arguments: %v", err), nil, true
	}
	if args.ID == "" {
		return "view_image: missing `id`. Pass the image id from a view_image reference, e.g. img_a1b2c3d4e5f60718.", nil, true
	}
	if sess == nil {
		return "view_image: no session", nil, true
	}
	data, mime, err := readImageFile(sess.Cwd, args.ID)
	if err != nil {
		return fmt.Sprintf("view_image: image %q not found in the session image store. References to images live in Summary (and inline in live history). Check the id matches a `view_image id=…` hint exactly.", args.ID), nil, true
	}
	parts := []any{
		map[string]any{"type": "text", "text": fmt.Sprintf("[Image %s re-delivered.]", args.ID)},
		map[string]any{
			"type": "image_url",
			"image_url": map[string]string{
				"url": fmt.Sprintf("data:%s;base64,%s", mime, base64.StdEncoding.EncodeToString(data)),
			},
		},
	}
	return fmt.Sprintf("[Image %s re-delivered.]", args.ID), parts, false
}
