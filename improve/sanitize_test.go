package improve

import (
	"strings"
	"testing"
)

func TestSanitizeNoSensitive(t *testing.T) {
	entry := ImprovementEntry{
		Title:    "clean",
		Original: "some normal text",
		New:      "more normal text",
	}
	sanitized, notes := Sanitize(entry)
	if len(notes) != 0 {
		t.Errorf("expected no redaction notes, got: %v", notes)
	}
	if sanitized.Original != "some normal text" {
		t.Errorf("original changed: %q", sanitized.Original)
	}
}

func TestSanitizeApiKeyRedacted(t *testing.T) {
	entry := ImprovementEntry{
		Original: `api_key = "sk-abc123secret"`,
	}
	sanitized, notes := Sanitize(entry)
	if len(notes) == 0 {
		t.Fatal("expected redaction notes")
	}
	if !strings.Contains(sanitized.Original, "[REDACTED]") {
		t.Errorf("api_key not redacted: %q", sanitized.Original)
	}
	if strings.Contains(sanitized.Original, "sk-abc123secret") {
		t.Errorf("api_key value still present: %q", sanitized.Original)
	}
}

func TestSanitizeBearerTokenRedacted(t *testing.T) {
	entry := ImprovementEntry{
		Original: "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9",
	}
	sanitized, notes := Sanitize(entry)
	if len(notes) == 0 {
		t.Fatal("expected redaction notes")
	}
	if !strings.Contains(sanitized.Original, "[REDACTED]") {
		t.Errorf("Bearer token not redacted: %q", sanitized.Original)
	}
	if strings.Contains(sanitized.Original, "eyJhbGciOiJIUzI1NiJ9") {
		t.Errorf("Bearer token value still present: %q", sanitized.Original)
	}
}

func TestSanitizeMultipleFields(t *testing.T) {
	entry := ImprovementEntry{
		Original: `api_key = "key123" and password = "pass456"`,
		New:      `token: tok789`,
	}
	sanitized, notes := Sanitize(entry)
	if len(notes) < 3 {
		t.Errorf("expected at least 3 redaction notes, got %d: %v", len(notes), notes)
	}
	if strings.Contains(sanitized.Original, "key123") || strings.Contains(sanitized.Original, "pass456") {
		t.Errorf("original still contains secrets: %q", sanitized.Original)
	}
	if strings.Contains(sanitized.New, "tok789") {
		t.Errorf("new still contains secret: %q", sanitized.New)
	}
}
