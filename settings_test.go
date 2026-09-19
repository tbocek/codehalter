package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/BurntSushi/toml"

	"github.com/tbocek/codehalter/llm"
)

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

// TestSeedSettingsRolesDifferInSamplersOnly pins the rule that the SEEDED
// settings.toml gives both roles the same llm.RenderKey. Samplers may differ freely
// (temperature, max_tokens and friends never reach the chat template); anything
// else means the two roles ask the server for two different token sequences of
// the same conversation, and on a one-slot server they evict each other at
// every plan -> execute switch.
//
// Measured over one 11.6h session: 35 role switches carrying 2414262 prompt
// tokens between them at 483 tok/s prefill. 4.2 minutes with identical
// renderings (121748 tokens actually re-read), 83.3 if each re-prefills. A user may still opt into the divergence per
// connection (it turns reasoning off for execute, worth ~50 minutes of decode
// on that session) and res/settings.toml says when it pays. What ships by
// default must not, because the default server holds one slot.
//
// Asserted through llm.RenderKey itself, not a re-implementation of it: the
// fingerprint the rewind detector compares at runtime is the thing that has to
// match.
func TestSeedSettingsRolesDifferInSamplersOnly(t *testing.T) {
	var s Settings
	if _, err := toml.Decode(defaultSettingsTOML, &s); err != nil {
		t.Fatalf("decode res/settings.toml: %v", err)
	}
	if len(s.LLM) == 0 {
		t.Fatal("res/settings.toml declares no [[llm]] entry")
	}
	for i, c := range s.LLM {
		think, exec := llm.RenderKey(c.ParamsThinking), llm.RenderKey(c.ParamsExecute)
		if think != exec {
			t.Errorf("llm[%d] renders differently per role (thinking=%s execute=%s): "+
				"non-sampler params re-render the prompt, so every phase switch re-prefills the context",
				i, think, exec)
		}
	}
}

// TestRenderSettingsSourcesMarksShadowed pins what /settings exists to show: a
// project-local settings.toml silently shadowing the global one. Precedence
// itself is pinned by TestLoadSettingsProjectLocalFirst; what matters here is
// that the LOSING file is still named, because the failure this command exists
// to diagnose is a user editing the global file while a forgotten local copy
// is the one being read.
func TestRenderSettingsSourcesMarksShadowed(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	globalPath := filepath.Join(home, ".config", "codehalter", "settings.toml")
	if err := os.MkdirAll(filepath.Dir(globalPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(globalPath, []byte("[[llm]]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cwd := t.TempDir()
	localPath := filepath.Join(cwd, sessionDir, "settings.toml")

	// (a) global only: it is in use, and the missing local file is still listed
	// so the user can see where to put an override.
	got := renderSettingsSources(cwd)
	if !strings.Contains(got, "✅ in use: `"+globalPath+"`") {
		t.Errorf("global-only: %q does not mark the global file in use", got)
	}
	if !strings.Contains(got, "absent: `"+localPath+"`") {
		t.Errorf("global-only: %q does not list the absent project path", got)
	}

	// (b) both exist: the project file wins and the global one is named as
	// shadowed, not omitted.
	if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(localPath, []byte("[[llm]]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got = renderSettingsSources(cwd)
	if !strings.Contains(got, "✅ in use: `"+localPath+"`") {
		t.Errorf("both: %q does not mark the project file in use", got)
	}
	if !strings.Contains(got, "❕ shadowed: `"+globalPath+"`") {
		t.Errorf("both: %q does not name the shadowed global file", got)
	}

	// (c) neither: say so outright rather than printing two absent paths and
	// leaving the user to conclude it.
	if err := os.Remove(localPath); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(globalPath); err != nil {
		t.Fatal(err)
	}
	if got = renderSettingsSources(cwd); !strings.Contains(got, "No settings.toml at either path") {
		t.Errorf("neither: %q does not say no settings file was found", got)
	}
}
