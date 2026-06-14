package improve

import (
	"os"
	"path/filepath"
	"testing"
)

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
