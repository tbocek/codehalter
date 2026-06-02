package main

import (
	"encoding/base64"
	"fmt"
	"strings"
	"testing"
)

// TestDispatchViewImageHappyPath: a known image id returns the text receipt
// and a 2-part multimodal slice (text + image_url data URL with correct mime).
func TestDispatchViewImageHappyPath(t *testing.T) {
	dir := t.TempDir()
	payload := []byte("PNG bytes here")
	id := "img_test_happy"
	if err := writeImageFile(dir, id, "image/png", payload); err != nil {
		t.Fatalf("writeImageFile: %v", err)
	}
	sess := &Session{Cwd: dir}

	text, parts, failed := dispatchViewImage(sess, fmt.Sprintf(`{"id":%q}`, id))
	if failed {
		t.Fatalf("dispatchViewImage: failed=true, text=%q", text)
	}
	if !strings.Contains(text, id) {
		t.Errorf("text receipt missing id: %q", text)
	}
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts (text + image_url), got %d", len(parts))
	}
	textBlock, _ := parts[0].(map[string]any)
	if textBlock["type"] != "text" {
		t.Errorf("parts[0] type: got %v, want text", textBlock["type"])
	}
	imgBlock, _ := parts[1].(map[string]any)
	if imgBlock["type"] != "image_url" {
		t.Errorf("parts[1] type: got %v, want image_url", imgBlock["type"])
	}
	url, _ := imgBlock["image_url"].(map[string]string)
	wantPrefix := "data:image/png;base64,"
	if !strings.HasPrefix(url["url"], wantPrefix) {
		t.Errorf("url prefix: got %q, want prefix %q", url["url"], wantPrefix)
	}
	// Decode the embedded base64 and confirm bytes round-trip.
	gotB64 := strings.TrimPrefix(url["url"], wantPrefix)
	got, err := base64.StdEncoding.DecodeString(gotB64)
	if err != nil {
		t.Fatalf("decode embedded base64: %v", err)
	}
	if string(got) != string(payload) {
		t.Errorf("payload round-trip: got %q, want %q", got, payload)
	}
}

// TestDispatchViewImageMissingID: a hallucinated id produces a clean,
// model-readable error with no multimodal parts.
func TestDispatchViewImageMissingID(t *testing.T) {
	dir := t.TempDir()
	sess := &Session{Cwd: dir}

	text, parts, failed := dispatchViewImage(sess, `{"id":"img_doesnotexist"}`)
	if !failed {
		t.Errorf("expected failed=true on missing id, got false")
	}
	if parts != nil {
		t.Errorf("expected nil parts on miss, got %+v", parts)
	}
	if !strings.Contains(text, "img_doesnotexist") {
		t.Errorf("error text missing id: %q", text)
	}
}

// TestDispatchViewImageBadArgs covers two argument failures: malformed JSON,
// and an empty id. Both must fail loudly rather than silently doing nothing.
func TestDispatchViewImageBadArgs(t *testing.T) {
	sess := &Session{Cwd: t.TempDir()}

	if text, _, failed := dispatchViewImage(sess, "not json"); !failed || !strings.Contains(text, "invalid arguments") {
		t.Errorf("bad JSON: failed=%v text=%q", failed, text)
	}
	if text, _, failed := dispatchViewImage(sess, `{}`); !failed || !strings.Contains(text, "missing `id`") {
		t.Errorf("empty id: failed=%v text=%q", failed, text)
	}
}

// TestDispatchViewImageNoSession: defensive — a nil session must fail cleanly,
// not panic.
func TestDispatchViewImageNoSession(t *testing.T) {
	text, _, failed := dispatchViewImage(nil, `{"id":"img_anything"}`)
	if !failed {
		t.Errorf("expected failed=true with nil session")
	}
	if !strings.Contains(text, "no session") {
		t.Errorf("nil-session error text: %q", text)
	}
}
