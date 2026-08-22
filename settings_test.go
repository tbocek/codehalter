package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/BurntSushi/toml"
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
// settings.toml gives both roles the same renderKey. Samplers may differ freely
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
// Asserted through renderKey itself, not a re-implementation of it: the
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
		think, exec := renderKey(c.ParamsThinking), renderKey(c.ParamsExecute)
		if think != exec {
			t.Errorf("llm[%d] renders differently per role (thinking=%s execute=%s): "+
				"non-sampler params re-render the prompt, so every phase switch re-prefills the context",
				i, think, exec)
		}
	}
}

// TestParamsForInventsNoChatTemplateKwargs pins that codehalter never puts a
// chat-template argument on the wire that the user did not write. It is
// tempting to: turning reasoning off for the execute role is worth ~50 minutes
// of decode on an 11.6h session against Qwen3.8-27B, and Qwen's /no_think text
// switch does not deliver it (237 of 388 execute responses reasoned anyway).
//
// But it gives the two roles different renderings, and on a one-slot server
// they evict each other on every plan -> execute switch: 35 switches in that
// session carrying 2414262 prompt tokens, at 483 tok/s prefill, so 83.3 minutes
// of re-prefill against the 50 it saves. It is the user's call per connection
// (res/settings.toml costs it), so paramsFor hands back exactly what was
// configured.
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
	// The legacy single `params` set gets the same treatment.
	legacy := LLMConnection{Params: map[string]any{"temperature": 0.7}}
	if _, set := legacy.paramsFor("execute")["chat_template_kwargs"]; set {
		t.Errorf("legacy params: codehalter invented chat_template_kwargs: %v", legacy.paramsFor("execute"))
	}

	// What the user DID write is passed through untouched, on the role they put
	// it on and only that role. The divergence is theirs to place.
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
