package main

import (
	"context"
	"encoding/json"
	"fmt"
)

// view_image re-fetches an image by id from the content-addressed store, so a
// reference kept in Summary still works after compaction.

var viewImageTool = Tool{
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
	// Only a fallback: real success goes through dispatchViewImage.
	Execute: viewImageExecuteFallback,
}

func viewImageExecuteFallback(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	if !a.imagesSupported {
		return "view_image: this LLM doesn't support image inputs — call other tools (read_file, run_command) to inspect the attachment indirectly.", true
	}
	return "view_image: internal — dispatch missed the intercept. Try again.", true
}

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
	text := fmt.Sprintf("[Image %s re-delivered.]", args.ID)
	return text, imageParts(text, mime, data), false
}
