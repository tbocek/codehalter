package improve

import (
	"os"
	"path/filepath"
	"testing"
)

func TestStoreAndLoadAll(t *testing.T) {
	tmpDir := t.TempDir()

	payload := ImprovementPayload{
		Improvements: []ImprovementEntry{
			{Title: "fix typo", File: "PLAN.md", Type: "replace", Original: "old text", New: "new text", Reasoning: "clarity", Ip: "10.0.0.1", Model: "qwen3.6-27b", License: "MIT"},
			{Title: "remove redundant", File: "EXECUTE.md", Type: "remove", Original: "verbose line", New: "", Reasoning: "brevity", Ip: "10.0.0.1", Model: "qwen3.6-27b", License: "MIT"},
		},
	}

	stored, _, err := Store(tmpDir, payload, nil)
	if err != nil {
		t.Fatalf("Store error: %v", err)
	}
	if stored != 2 {
		t.Errorf("stored = %d, want 2", stored)
	}

	entries, err := LoadAll(tmpDir)
	if err != nil {
		t.Fatalf("LoadAll error: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("LoadAll returned %d entries, want 2", len(entries))
	}

	// Verify round-trip
	if entries[0].Title != "fix typo" {
		t.Errorf("entries[0].Title = %q, want %q", entries[0].Title, "fix typo")
	}
	if entries[0].Original != "old text" {
		t.Errorf("entries[0].Original = %q, want %q", entries[0].Original, "old text")
	}
	if entries[0].New != "new text" {
		t.Errorf("entries[0].New = %q, want %q", entries[0].New, "new text")
	}
	if entries[1].File != "EXECUTE.md" {
		t.Errorf("entries[1].File = %q, want %q", entries[1].File, "EXECUTE.md")
	}
	if entries[0].Ip != "10.0.0.1" {
		t.Errorf("entries[0].Ip = %q, want %q", entries[0].Ip, "10.0.0.1")
	}
	if entries[0].Model != "qwen3.6-27b" {
		t.Errorf("entries[0].Model = %q, want %q", entries[0].Model, "qwen3.6-27b")
	}
	if entries[0].License != "MIT" {
		t.Errorf("entries[0].License = %q, want %q", entries[0].License, "MIT")
	}
}

func TestStoreCreatesDataDir(t *testing.T) {
	tmpDir := filepath.Join(t.TempDir(), "nested", "data")
	payload := ImprovementPayload{
		Improvements: []ImprovementEntry{
			{Title: "test", File: "PLAN.md", Type: "add", Original: "", New: "text", Reasoning: "test", Ip: "10.0.0.1", Model: "qwen3.6-27b", License: "MIT"},
		},
	}
	_, _, err := Store(tmpDir, payload, nil)
	if err != nil {
		t.Fatalf("Store failed on nested dir: %v", err)
	}
	if _, err := os.Stat(tmpDir); os.IsNotExist(err) {
		t.Fatal("data dir was not created")
	}
}
