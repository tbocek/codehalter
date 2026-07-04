package main

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTemp(t *testing.T, name, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadConfigValid(t *testing.T) {
	p := writeTemp(t, "crafter.toml", `
[settings]
samples = 5

[judge]
server = "http://j"
model = "judge-m"

[[model]]
name = "gemma"
server = "http://g"
model = "gemma-m"

[[model]]
name = "qwen"
server = "http://q"
model = "qwen-m"
`)
	cfg, err := loadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Settings.Samples != 5 {
		t.Fatalf("samples = %d", cfg.Settings.Samples)
	}
	if len(cfg.Models) != 2 || cfg.Models[0].Name != "gemma" {
		t.Fatalf("models = %+v", cfg.Models)
	}
	if cfg.Judge.Name != "main" {
		t.Fatalf("judge name should default to %q, got %q", "main", cfg.Judge.Name)
	}
	// Endpoint semaphores: default parallel=1, channel wired on judge + models.
	if cfg.Judge.Parallel != 1 || cap(cfg.Judge.sem) != 1 {
		t.Fatalf("judge parallel/sem = %d/%d, want 1/1", cfg.Judge.Parallel, cap(cfg.Judge.sem))
	}
	for _, m := range cfg.Models {
		if m.Parallel != 1 || cap(m.sem) != 1 {
			t.Fatalf("model %s parallel/sem = %d/%d, want 1/1", m.Name, m.Parallel, cap(m.sem))
		}
	}
}

func TestLoadConfigSamplesDefault(t *testing.T) {
	p := writeTemp(t, "c.toml", `
[judge]
server = "http://j"
model = "m"
[[model]]
name = "a"
server = "http://a"
model = "am"
`)
	cfg, err := loadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Settings.Samples != 3 {
		t.Fatalf("default samples = %d, want 3", cfg.Settings.Samples)
	}
}

func TestLoadConfigErrors(t *testing.T) {
	cases := map[string]string{
		"no judge": `
[[model]]
name = "a"
server = "http://a"
model = "am"
`,
		"no models": `
[judge]
server = "http://j"
model = "m"
`,
		"dup names": `
[judge]
server = "http://j"
model = "m"
[[model]]
name = "a"
server = "http://a"
model = "am"
[[model]]
name = "a"
server = "http://b"
model = "bm"
`,
		"model missing name": `
[judge]
server = "http://j"
model = "m"
[[model]]
server = "http://a"
model = "am"
`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			p := writeTemp(t, "c.toml", body)
			if _, err := loadConfig(p); err == nil {
				t.Fatalf("expected error for %s", name)
			}
		})
	}
}

func TestWantSkill(t *testing.T) {
	all := Settings{}
	if !all.wantSkill("go") || !all.wantSkill("anything") {
		t.Fatal("empty filter should allow all")
	}
	only := Settings{Skills: []string{"go", "base"}}
	if !only.wantSkill("go") || only.wantSkill("ts") {
		t.Fatal("filter mismatch")
	}
}
