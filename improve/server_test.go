package improve

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestServerPostImprovement(t *testing.T) {
	tmpDir := t.TempDir()
	handler := NewServer(tmpDir)

	payload := map[string]any{
		"improvements": []map[string]any{
			{
				"title":     "test improvement",
				"file":      "PLAN.md",
				"type":      "replace",
				"original":  "old",
				"new":       "new",
				"reasoning": "test",
			},
		},
	}

	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/v1/improvements", bytes.NewReader(body))
	req.Header.Set("X-Client-IP", "10.0.0.1")
	req.Header.Set("X-Model", "qwen3.6-27b")
	req.Header.Set("X-License", "MIT")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("parse response: %v", err)
	}

	if stored, ok := resp["stored"].(float64); !ok || stored != 1 {
		t.Errorf("stored = %v, want 1", resp["stored"])
	}

	// Verify file was written
	entries, err := LoadAll(tmpDir)
	if err != nil {
		t.Fatalf("LoadAll: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("LoadAll returned %d entries, want 1", len(entries))
	}
	if entries[0].Title != "test improvement" {
		t.Errorf("title = %q, want %q", entries[0].Title, "test improvement")
	}
	if entries[0].Ip != "10.0.0.1" {
		t.Errorf("ip = %q, want %q", entries[0].Ip, "10.0.0.1")
	}
	if entries[0].Model != "qwen3.6-27b" {
		t.Errorf("model = %q, want %q", entries[0].Model, "qwen3.6-27b")
	}
	if entries[0].License != "MIT" {
		t.Errorf("license = %q, want %q", entries[0].License, "MIT")
	}
}

func TestServerPostWithSensitiveData(t *testing.T) {
	tmpDir := t.TempDir()
	handler := NewServer(tmpDir)

	payload := map[string]any{
		"improvements": []map[string]any{
			{
				"title":     "sensitive",
				"file":      "PLAN.md",
				"type":      "replace",
				"original":  `api_key = "secret123"`,
				"new":       "new text",
				"reasoning": "test",
			},
		},
	}

	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/v1/improvements", bytes.NewReader(body))
	req.Header.Set("X-Client-IP", "10.0.0.2")
	req.Header.Set("X-Model", "gemma-27b")
	req.Header.Set("X-License", "Apache-2.0")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)

	redacted, ok := resp["redacted"].([]any)
	if !ok || len(redacted) == 0 {
		t.Errorf("expected redacted notes, got: %v", resp["redacted"])
	}

	// Verify stored file does not contain the secret
	entries, err := LoadAll(tmpDir)
	if err != nil {
		t.Fatalf("LoadAll: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("no entries stored")
	}
	if bytes.Contains([]byte(entries[0].Original), []byte("secret123")) {
		t.Errorf("stored file still contains secret: %q", entries[0].Original)
	}
}

func TestServerPostInvalidJSON(t *testing.T) {
	tmpDir := t.TempDir()
	handler := NewServer(tmpDir)

	req := httptest.NewRequest(http.MethodPost, "/v1/improvements", bytes.NewReader([]byte("not json")))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestServerPostReadBodyError(t *testing.T) {
	tmpDir := t.TempDir()
	handler := NewServer(tmpDir)

	req := httptest.NewRequest(http.MethodPost, "/v1/improvements", &errorReader{})
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

type errorReader struct{}

func (e *errorReader) Read(p []byte) (int, error) { return 0, io.EOF }
