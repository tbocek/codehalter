package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/BurntSushi/toml"
)

// Whole-file selection, never a merge.
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

// Samplers may differ between roles; anything else gives two renderings that evict each other on
// a one-slot server at every plan/execute switch. Asserted through renderKey, the runtime fingerprint.
func TestSeedSettingsRolesDifferInSamplersOnly(t *testing.T) {
	var s Settings
	if _, err := toml.Decode(defaultSettingsTOML, &s); err != nil {
		t.Fatalf("decode res/settings.toml: %v", err)
	}
	if len(s.LLM) == 0 {
		t.Fatal("res/settings.toml declares no [[llm]] entry")
	}
	for i, c := range s.LLM {
		think, exec := renderKey(c.ParamsThinking), renderKey(c.ParamsExecute)
		if think != exec {
			t.Errorf("llm[%d] renders differently per role (thinking=%s execute=%s): "+
				"non-sampler params re-render the prompt, so every phase switch re-prefills the context",
				i, think, exec)
		}
	}
}

// Per-role kwargs would re-prefill at every phase switch on a one-slot server; reasoning-off
// uses an appended <think></think> (withThinkingDisabled) instead.
func TestParamsForInventsNoChatTemplateKwargs(t *testing.T) {
	c := LLMConnection{
		ParamsThinking: map[string]any{"temperature": 1.0},
		ParamsExecute:  map[string]any{"temperature": 0.6},
	}
	for _, role := range []string{"thinking", "execute"} {
		if _, set := c.paramsFor(role)["chat_template_kwargs"]; set {
			t.Errorf("%s: codehalter invented chat_template_kwargs: %v", role, c.paramsFor(role))
		}
	}
	legacy := LLMConnection{Params: map[string]any{"temperature": 0.7}}
	if _, set := legacy.paramsFor("execute")["chat_template_kwargs"]; set {
		t.Errorf("legacy params: codehalter invented chat_template_kwargs: %v", legacy.paramsFor("execute"))
	}

	opted := LLMConnection{
		ParamsThinking: map[string]any{"temperature": 1.0},
		ParamsExecute: map[string]any{
			"temperature":          0.6,
			"chat_template_kwargs": map[string]any{"enable_thinking": false},
		},
	}
	ctk, _ := opted.paramsFor("execute")["chat_template_kwargs"].(map[string]any)
	if ctk["enable_thinking"] != false {
		t.Errorf("the user's own kwargs did not survive: %v", opted.paramsFor("execute"))
	}
	if _, set := opted.paramsFor("thinking")["chat_template_kwargs"]; set {
		t.Error("params_execute kwargs leaked onto the thinking role")
	}
	if got := opted.paramsFor("execute")["temperature"]; got != 0.6 {
		t.Errorf("temperature = %v, want 0.6", got)
	}
}

// The losing file must still be named: users edit the global file while a forgotten local copy is read.
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

	got := renderSettingsSources(cwd)
	if !strings.Contains(got, "✅ in use: `"+globalPath+"`") {
		t.Errorf("global-only: %q does not mark the global file in use", got)
	}
	if !strings.Contains(got, "absent: `"+localPath+"`") {
		t.Errorf("global-only: %q does not list the absent project path", got)
	}

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

// llama.cpp turns tool_choice into a grammar and leaves the prompt tokens alone.
func TestRenderKeyIgnoresToolChoice(t *testing.T) {
	base := map[string]any{"temperature": 0.7}
	required := map[string]any{"temperature": 0.7, "tool_choice": "required"}
	none := map[string]any{"temperature": 0.7, "tool_choice": "none"}
	if renderKey(base) != renderKey(required) || renderKey(required) != renderKey(none) {
		t.Errorf("tool_choice must not change the render fingerprint: %q vs %q vs %q",
			renderKey(base), renderKey(required), renderKey(none))
	}
	if renderKey(base) == renderKey(map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}}) {
		t.Error("chat_template_kwargs must still change the fingerprint")
	}
}

// An unparsable value falls back to the default rather than disabling the refresh.
func TestKeepWarmInterval(t *testing.T) {
	for raw, want := range map[string]time.Duration{
		"":       keepWarmEvery,
		"90s":    90 * time.Second,
		"5m":     5 * time.Minute,
		"off":    0,
		"0":      0,
		"no":     0,
		"banana": keepWarmEvery,
	} {
		a := &agent{settings: Settings{KeepWarm: raw}}
		if got := a.keepWarmInterval(); got != want {
			t.Errorf("keep_warm=%q -> %v, want %v", raw, got, want)
		}
	}
}
