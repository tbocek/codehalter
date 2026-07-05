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

[[judge]]
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
	if len(cfg.Judges) != 1 || cfg.Judges[0].Name != "main" {
		t.Fatalf("single [judge] should give one judge named %q, got %+v", "main", cfg.Judges)
	}
	// loadConfig leaves parallelism unresolved (0 = auto) and the semaphore nil
	// until resolveParallelism runs.
	if cfg.Judges[0].Parallel != 0 || cfg.Judges[0].sem != nil {
		t.Fatalf("loadConfig should not resolve parallelism: parallel=%d sem=%v", cfg.Judges[0].Parallel, cfg.Judges[0].sem)
	}
}

func TestResolveParallelism(t *testing.T) {
	cfg := &Config{
		Judges: []ModelSpec{{Name: "main", Model: "j"}}, // unset → auto-detect
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
	check(cfg.Judges[0])
	for _, m := range cfg.Models {
		check(m)
	}
}

func TestLoadConfigSamplesDefault(t *testing.T) {
	p := writeTemp(t, "c.toml", `
[[judge]]
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
[[judge]]
server = "http://j"
model = "m"
`,
		"dup names": `
[[judge]]
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
[[judge]]
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

func TestLoadConfigMultipleJudges(t *testing.T) {
	p := writeTemp(t, "c.toml", `
[[judge]]
server = "http://j1"
model  = "m1"

[[judge]]
server = "http://j2"
model  = "m2"

[[model]]
name = "a"
server = "http://a"
model = "am"
`)
	cfg, err := loadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Judges) != 2 {
		t.Fatalf("judges = %d, want 2", len(cfg.Judges))
	}
	// Unnamed judges default to main, judge2, …
	if cfg.Judges[0].Name != "main" || cfg.Judges[1].Name != "judge2" {
		t.Fatalf("judge names = %q,%q, want main,judge2", cfg.Judges[0].Name, cfg.Judges[1].Name)
	}
	if cfg.Judges[0].Model != "m1" || cfg.Judges[1].Server != "http://j2" {
		t.Fatalf("judge fields wrong: %+v", cfg.Judges)
	}
}

func TestLoadConfigRejectsDuplicateJudgeNames(t *testing.T) {
	p := writeTemp(t, "c.toml", `
[[judge]]
name = "dup"
server = "http://j1"
model  = "m1"
[[judge]]
name = "dup"
server = "http://j2"
model  = "m2"
[[model]]
name = "a"
server = "http://a"
model = "am"
`)
	if _, err := loadConfig(p); err == nil {
		t.Fatal("duplicate judge names must error")
	}
}
