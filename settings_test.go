package main

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/BurntSushi/toml"
)

// samplerParams are the request fields that only steer generation. They never
// reach the server's chat template, so the two roles may differ in them freely.
// Anything else is assumed to change the rendered prompt.
var samplerParams = map[string]bool{
	"frequency_penalty": true, "max_tokens": true, "min_p": true,
	"n": true, "presence_penalty": true, "repeat_penalty": true,
	"seed": true, "stop": true, "temperature": true, "top_k": true, "top_p": true,
}

// TestLoadSettingsProjectLocalFirst pins the settings precedence: the
// project-local .codehalter/settings.toml wins over the global
// ~/.config/codehalter/settings.toml (which is only the fallback for projects
// without a local file). Whole-file selection, no merging.
func TestLoadSettingsProjectLocalFirst(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // isolate the global dir from the developer's
	cwd := t.TempDir()
	localPath := filepath.Join(cwd, sessionDir, "settings.toml")
	if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
		t.Fatal(err)
	}
	globalPath := filepath.Join(os.Getenv("HOME"), ".config", "codehalter", "settings.toml")
	if err := os.MkdirAll(filepath.Dir(globalPath), 0o755); err != nil {
		t.Fatal(err)
	}
	writeLocal := func() {
		t.Helper()
		cfg := "[[llm]]\nserver = \"http://local.example\"\nmodel = \"local-model\"\n"
		if err := os.WriteFile(localPath, []byte(cfg), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	writeGlobal := func() {
		t.Helper()
		cfg := "[[llm]]\nserver = \"http://g.example\"\nmodel = \"global-model\"\n"
		if err := os.WriteFile(globalPath, []byte(cfg), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// (a) both files exist: project-local must win.
	writeLocal()
	writeGlobal()
	s, err := loadSettings(cwd)
	if err != nil {
		t.Fatalf("loadSettings(both): %v", err)
	}
	if s.path != localPath {
		t.Errorf("both files: path = %q, want project-local %q", s.path, localPath)
	}
	if len(s.LLM) != 1 || s.LLM[0].Model != "local-model" {
		t.Errorf("both files: LLM = %+v, want one entry with model local-model", s.LLM)
	}

	// (b) global only: the fallback must be used.
	if err := os.Remove(localPath); err != nil {
		t.Fatal(err)
	}
	s, err = loadSettings(cwd)
	if err != nil {
		t.Fatalf("loadSettings(global only): %v", err)
	}
	if s.path != globalPath {
		t.Errorf("global only: path = %q, want %q", s.path, globalPath)
	}
	if len(s.LLM) != 1 || s.LLM[0].Model != "global-model" {
		t.Errorf("global only: LLM = %+v, want one entry with model global-model", s.LLM)
	}

	// (c) local only: the local file is used.
	if err := os.Remove(globalPath); err != nil {
		t.Fatal(err)
	}
	writeLocal()
	s, err = loadSettings(cwd)
	if err != nil {
		t.Fatalf("loadSettings(local only): %v", err)
	}
	if s.path != localPath {
		t.Errorf("local only: path = %q, want %q", s.path, localPath)
	}
	if len(s.LLM) != 1 || s.LLM[0].Model != "local-model" {
		t.Errorf("local only: LLM = %+v, want one entry with model local-model", s.LLM)
	}
}

// TestSeedSettingsRolesDifferInSamplersOnly pins the rule that made a turn pay a
// full prefill twice: params_execute carried chat_template_kwargs =
// { enable_thinking = false } and params_thinking did not. That field is an
// argument to the server's chat template, not a sampler, so the plan → execute
// role switch re-rendered the entire prompt — 9088 of 14436 tokens
// re-evaluated, 15s, with the next call (same field) hot again. Reasoning is
// turned off for execute in the message text now (noThinkSwitch), which the
// cache doesn't see. Sampler differences stay allowed: they don't enter the KV
// cache key.
func TestSeedSettingsRolesDifferInSamplersOnly(t *testing.T) {
	var s Settings
	if _, err := toml.Decode(defaultSettingsTOML, &s); err != nil {
		t.Fatalf("decode res/settings.toml: %v", err)
	}
	if len(s.LLM) == 0 {
		t.Fatal("res/settings.toml declares no [[llm]] entry")
	}
	for i, c := range s.LLM {
		keys := map[string]bool{}
		for k := range c.ParamsThinking {
			keys[k] = true
		}
		for k := range c.ParamsExecute {
			keys[k] = true
		}
		for k := range keys {
			if samplerParams[k] {
				continue
			}
			// DeepEqual, not ==: a nested TOML table decodes to a map, and == on two
			// maps of the same type panics.
			tv, ev := c.ParamsThinking[k], c.ParamsExecute[k]
			if !reflect.DeepEqual(tv, ev) {
				t.Errorf("llm[%d] %q differs between roles (thinking=%v execute=%v): non-sampler params re-render the prompt and break the prefix cache on every phase switch", i, k, tv, ev)
			}
		}
	}
}
