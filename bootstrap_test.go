package main

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func writeFiles(t *testing.T, dir string, names ...string) {
	t.Helper()
	for _, n := range names {
		p := filepath.Join(dir, n)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", filepath.Dir(p), err)
		}
		if err := os.WriteFile(p, nil, 0o644); err != nil {
			t.Fatalf("write %s: %v", n, err)
		}
	}
}

func TestDetectStacksEmpty(t *testing.T) {
	if got := detectStacks(t.TempDir()); len(got) != 0 {
		t.Errorf("empty dir: want no stacks, got %v", got)
	}
}

func TestDetectStacksSingle(t *testing.T) {
	cases := []struct {
		name  string
		files []string
		want  string
	}{
		{"go", []string{"go.mod"}, "go"},
		{"rust", []string{"Cargo.toml"}, "rust"},
		{"zig-build", []string{"build.zig"}, "zig"},
		{"zig-zon", []string{"build.zig.zon"}, "zig"},
		{"java-pom", []string{"pom.xml"}, "java"},
		{"java-gradle", []string{"build.gradle"}, "java"},
		{"java-gradle-kts", []string{"build.gradle.kts"}, "java"},
		{"ts-tsconfig", []string{"tsconfig.json"}, "ts"},
		{"ts-fileonly", []string{"app.ts"}, "ts"},
		{"js", []string{"package.json"}, "js"},
		{"bash", []string{"run.sh"}, "bash"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			writeFiles(t, dir, c.files...)
			got := detectStacks(dir)
			if len(got) != 1 || got[0] != c.want {
				t.Errorf("files %v: want [%s], got %v", c.files, c.want, got)
			}
		})
	}
}

// TS wins over JS when both package.json and a .ts file are present —
// otherwise polyglot TS projects would get the JS skill instead.
func TestDetectStacksTSBeatsJS(t *testing.T) {
	dir := t.TempDir()
	writeFiles(t, dir, "package.json", "app.ts")
	got := detectStacks(dir)
	for _, s := range got {
		if s == "js" {
			t.Errorf("expected ts to suppress js; got %v", got)
		}
	}
	if !contains(got, "ts") {
		t.Errorf("want ts in %v", got)
	}
}

func TestDetectStacksDevcontainerDir(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, ".devcontainer"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if !contains(detectStacks(dir), "devcontainer") {
		t.Errorf("want devcontainer detected when .devcontainer/ dir exists")
	}
}

// Polyglot project: every stack at once, in the documented stable order.
func TestDetectStacksMulti(t *testing.T) {
	dir := t.TempDir()
	writeFiles(t, dir,
		"go.mod", "package.json", "tsconfig.json",
		"pom.xml", "Cargo.toml", "build.zig", "run.sh",
	)
	if err := os.MkdirAll(filepath.Join(dir, ".devcontainer"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	want := []string{"go", "ts", "java", "rust", "zig", "bash", "devcontainer"}
	got := detectStacks(dir)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("multi: want %v, got %v", want, got)
	}
}

func TestHasFileWithExt(t *testing.T) {
	dir := t.TempDir()
	writeFiles(t, dir, "a.go", "b.md")
	if err := os.MkdirAll(filepath.Join(dir, "sub.ts"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if !hasFileWithExt(dir, ".go") {
		t.Errorf("want .go detected")
	}
	if !hasFileWithExt(dir, ".rs", ".md") {
		t.Errorf("want .md detected via multi-ext")
	}
	if hasFileWithExt(dir, ".rs") {
		t.Errorf("want .rs not detected")
	}
	if hasFileWithExt(dir, ".ts") {
		t.Errorf("dir named *.ts must not count as a .ts file")
	}
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}
