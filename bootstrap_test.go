package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
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
		{"c-source", []string{"main.c"}, "c"},
		{"cpp-source", []string{"main.cpp"}, "c"},
		{"c-header-only", []string{"lib.h"}, "c"},
		{"cmake", []string{"CMakeLists.txt"}, "c"},
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
		"pom.xml", "Cargo.toml", "build.zig", "main.c", "run.sh",
	)
	if err := os.MkdirAll(filepath.Join(dir, ".devcontainer"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	want := []string{"go", "ts", "java", "rust", "zig", "c", "bash", "devcontainer"}
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

// devMounts unmarshals buildDevcontainerJSON output (it must stay valid plain
// JSON) and returns the mounts list + containerEnv.
func devMounts(t *testing.T, raw string) ([]string, map[string]any) {
	t.Helper()
	var m struct {
		ContainerEnv map[string]any `json:"containerEnv"`
		Mounts       []string       `json:"mounts"`
	}
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, raw)
	}
	return m.Mounts, m.ContainerEnv
}

func anyHas(ss []string, sub string) bool {
	for _, s := range ss {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

// TestBuildDevcontainerJSON pins the scaffold-time mount splicing: the base has
// only the codehalter config mount; .git / .gitconfig / ssh are added only when
// opted in, and ssh also sets the SSH_AUTH_SOCK env. Output stays valid JSON.
func TestBuildDevcontainerJSON(t *testing.T) {
	// Base: nothing opted in.
	bm, benv := devMounts(t, buildDevcontainerJSON(false, false, false))
	if len(bm) != 1 || !anyHas(bm, "/.config/codehalter") {
		t.Errorf("base must be just the config mount, got %v", bm)
	}
	if _, ok := benv["SSH_AUTH_SOCK"]; ok {
		t.Errorf("base must not set SSH_AUTH_SOCK env, got %v", benv)
	}

	// git writable, no gitconfig.
	gm, _ := devMounts(t, buildDevcontainerJSON(true, false, false))
	if !anyHas(gm, "containerWorkspaceFolder}/.git") {
		t.Errorf("gitWritable must add the .git mount, got %v", gm)
	}
	if anyHas(gm, "/.gitconfig") || anyHas(gm, "ssh-agent") {
		t.Errorf("git-only must not add gitconfig/ssh, got %v", gm)
	}

	// git + gitconfig.
	gcm, _ := devMounts(t, buildDevcontainerJSON(true, true, false))
	if !anyHas(gcm, "containerWorkspaceFolder}/.git") || !anyHas(gcm, "/.gitconfig") {
		t.Errorf("git+gitconfig must add both, got %v", gcm)
	}

	// ssh only: socket mount + env, no git.
	sm, senv := devMounts(t, buildDevcontainerJSON(false, false, true))
	if !anyHas(sm, "ssh-agent") || anyHas(sm, "/.git,") {
		t.Errorf("ssh-only mounts wrong, got %v", sm)
	}
	if senv["SSH_AUTH_SOCK"] != "/ssh-agent" {
		t.Errorf("ssh must set SSH_AUTH_SOCK=/ssh-agent, got %v", senv)
	}

	// everything on → 4 mounts.
	am, _ := devMounts(t, buildDevcontainerJSON(true, true, true))
	if len(am) != 4 {
		t.Errorf("all-on should have 4 mounts, got %v", am)
	}
}

func TestHasGitFolder(t *testing.T) {
	dir := t.TempDir()
	if hasGitFolder(dir) {
		t.Errorf("no .git → false")
	}
	if err := os.Mkdir(filepath.Join(dir, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	if !hasGitFolder(dir) {
		t.Errorf(".git dir → true")
	}
	d2 := t.TempDir()
	if err := os.WriteFile(filepath.Join(d2, ".git"), []byte("gitdir: ../x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if hasGitFolder(d2) {
		t.Errorf(".git FILE (worktree link) must not count")
	}
}

func TestHostSSHAgentAvailable(t *testing.T) {
	t.Setenv("SSH_AUTH_SOCK", "")
	if hostSSHAgentAvailable() {
		t.Errorf("unset SSH_AUTH_SOCK → false")
	}
	t.Setenv("SSH_AUTH_SOCK", filepath.Join(t.TempDir(), "nope.sock"))
	if hostSSHAgentAvailable() {
		t.Errorf("missing socket → false")
	}
	sock := filepath.Join(t.TempDir(), "agent.sock")
	if err := os.WriteFile(sock, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("SSH_AUTH_SOCK", sock)
	if !hostSSHAgentAvailable() {
		t.Errorf("existing socket → true")
	}
}

func TestLoadGlobalConfig(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if loadGlobalConfig().HasGitconfigInHome {
		t.Errorf("missing global.toml → false")
	}
	cfg := filepath.Join(home, ".config", "codehalter")
	if err := os.MkdirAll(cfg, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cfg, "global.toml"), []byte("has_gitconfig_in_home = true\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !loadGlobalConfig().HasGitconfigInHome {
		t.Errorf("global.toml has_gitconfig_in_home=true → true")
	}
}

// TestEnsureSettingsGitignored pins the secrets-file ignore: settings.toml is
// added to .gitignore (created if the project is a git repo), idempotently, and
// not at all in a non-git directory.
func TestEnsureSettingsGitignored(t *testing.T) {
	// No git, no .gitignore → no-op, no file created.
	bare := t.TempDir()
	if ensureSettingsGitignored(bare) {
		t.Errorf("non-git dir with no .gitignore must not gitignore")
	}
	if _, err := os.Stat(filepath.Join(bare, ".gitignore")); !os.IsNotExist(err) {
		t.Errorf("non-git dir: .gitignore must not be created")
	}

	// Git repo, no .gitignore → creates it with the entry; idempotent.
	repo := t.TempDir()
	if err := os.Mkdir(filepath.Join(repo, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	if !ensureSettingsGitignored(repo) {
		t.Fatalf("git repo: must gitignore settings.toml")
	}
	data, _ := os.ReadFile(filepath.Join(repo, ".gitignore"))
	if !strings.Contains(string(data), gitignoreSettingsEntry) {
		t.Errorf("entry missing:\n%s", data)
	}
	ensureSettingsGitignored(repo) // idempotent
	data2, _ := os.ReadFile(filepath.Join(repo, ".gitignore"))
	if strings.Count(string(data2), gitignoreSettingsEntry) != 1 {
		t.Errorf("entry duplicated:\n%s", data2)
	}

	// Existing .gitignore without trailing newline → appends on a fresh line.
	repo2 := t.TempDir()
	if err := os.Mkdir(filepath.Join(repo2, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo2, ".gitignore"), []byte("node_modules"), 0o644); err != nil {
		t.Fatal(err)
	}
	ensureSettingsGitignored(repo2)
	data3, _ := os.ReadFile(filepath.Join(repo2, ".gitignore"))
	if !strings.Contains(string(data3), "node_modules\n"+gitignoreSettingsEntry) {
		t.Errorf("must append on a fresh line:\n%q", string(data3))
	}
}
