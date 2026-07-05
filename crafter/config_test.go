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
	// loadConfig leaves parallelism unresolved (0 = auto) and the semaphore nil
	// until resolveParallelism runs.
	if cfg.Judge.Parallel != 0 || cfg.Judge.sem != nil {
		t.Fatalf("loadConfig should not resolve parallelism: parallel=%d sem=%v", cfg.Judge.Parallel, cfg.Judge.sem)
	}
}

func TestResolveParallelism(t *testing.T) {
	cfg := &Config{
		Judge: ModelSpec{Name: "main", Model: "j"}, // unset → auto-detect
		Models: []ModelSpec{
			{Name: "auto", Model: "a"},                  // unset → auto-detect
			{Name: "explicit", Model: "e", Parallel: 5}, // configured, kept
			{Name: "undetectable", Model: "u"},          // detect fails → 1
		},
	}
	detect := func(m ModelSpec) int {
		switch m.Model {
		case "j":
			return 3
		case "a":
			return 2
		default:
			return 0 // detection failed
		}
	}
	cfg.resolveParallelism(detect, func(string, ...any) {})

	want := map[string]int{"main": 3, "auto": 2, "explicit": 5, "undetectable": 1}
	check := func(m ModelSpec) {
		if m.Parallel != want[m.Name] {
			t.Fatalf("%s parallel = %d, want %d", m.Name, m.Parallel, want[m.Name])
		}
		if cap(m.sem) != want[m.Name] {
			t.Fatalf("%s sem cap = %d, want %d", m.Name, cap(m.sem), want[m.Name])
		}
	}
	check(cfg.Judge)
	for _, m := range cfg.Models {
		check(m)
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
