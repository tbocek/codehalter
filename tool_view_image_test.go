package main

import (
	"encoding/base64"
	"fmt"
	"strings"
	"testing"
)

// Two parts: the text receipt and an image_url data URL with the right mime.
func TestDispatchViewImageHappyPath(t *testing.T) {
	dir := t.TempDir()
	payload := []byte("PNG bytes here")
	id, err := storeImage(dir, "image/png", payload)
	if err != nil {
		t.Fatalf("storeImage: %v", err)
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
	gotB64 := strings.TrimPrefix(url["url"], wantPrefix)
	got, err := base64.StdEncoding.DecodeString(gotB64)
	if err != nil {
		t.Fatalf("decode embedded base64: %v", err)
	}
	if string(got) != string(payload) {
		t.Errorf("payload round-trip: got %q, want %q", got, payload)
	}
}

// Every way a view_image call can fail says why and returns no picture.
func TestDispatchViewImageRefusals(t *testing.T) {
	sess := &Session{Cwd: t.TempDir()}
	for _, c := range []struct {
		sess       *Session
		args, want string
	}{
		{sess, `{"id":"img_doesnotexist"}`, "img_doesnotexist"},
		{sess, "not json", "invalid arguments"},
		{sess, `{}`, "missing `id`"},
		{nil, `{"id":"img_anything"}`, "no session"},
	} {
		text, parts, failed := dispatchViewImage(c.sess, c.args)
		if !failed || parts != nil || !strings.Contains(text, c.want) {
			t.Errorf("%s: failed=%v parts=%v text=%q, want a failure naming %q", c.args, failed, parts != nil, text, c.want)
		}
	}
}
