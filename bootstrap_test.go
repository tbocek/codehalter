package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// want is in the fixed order skillSet relies on.
func TestDetectStacks(t *testing.T) {
	for _, c := range []struct {
		name  string
		files []string
		dirs  []string
		want  []string
	}{
		{name: "empty"},
		{name: "go", files: []string{"go.mod"}, want: []string{"go"}},
		{name: "rust", files: []string{"Cargo.toml"}, want: []string{"rust"}},
		{name: "zig-build", files: []string{"build.zig"}, want: []string{"zig"}},
		{name: "zig-zon", files: []string{"build.zig.zon"}, want: []string{"zig"}},
		{name: "java-pom", files: []string{"pom.xml"}, want: []string{"java"}},
		{name: "java-gradle", files: []string{"build.gradle"}, want: []string{"java"}},
		{name: "java-gradle-kts", files: []string{"build.gradle.kts"}, want: []string{"java"}},
		{name: "ts-tsconfig", files: []string{"tsconfig.json"}, want: []string{"ts"}},
		{name: "ts-fileonly", files: []string{"app.ts"}, want: []string{"ts"}},
		{name: "js", files: []string{"package.json"}, want: []string{"js"}},
		{name: "c-source", files: []string{"main.c"}, want: []string{"c"}},
		{name: "cpp-source", files: []string{"main.cpp"}, want: []string{"c"}},
		{name: "c-header-only", files: []string{"lib.h"}, want: []string{"c"}},
		{name: "cmake", files: []string{"CMakeLists.txt"}, want: []string{"c"}},
		{name: "css", files: []string{"layout.css"}, want: []string{"css"}},
		{name: "css-html", files: []string{"index.html"}, want: []string{"css"}},
		{name: "ts beats js", files: []string{"package.json", "app.ts"}, want: []string{"ts"}},
		{name: "scaffolding only", files: []string{"run.sh", "build.bash"}, dirs: []string{".devcontainer"}},
		{name: "a directory named *.ts is no ts file", dirs: []string{"sub.ts"}},
		{
			name:  "multi, fixed order",
			files: []string{"go.mod", "package.json", "tsconfig.json", "pom.xml", "Cargo.toml", "build.zig", "main.c", "run.sh", "layout.css"},
			dirs:  []string{".devcontainer"},
			want:  []string{"go", "ts", "css", "java", "rust", "zig", "c"},
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			writeFiles(t, dir, c.files...)
			for _, d := range c.dirs {
				if err := os.MkdirAll(filepath.Join(dir, d), 0o755); err != nil {
					t.Fatal(err)
				}
			}
			if got := detectStacks(dir); !slices.Equal(got, c.want) {
				t.Errorf("files %v dirs %v: want %v, got %v", c.files, c.dirs, c.want, got)
			}
		})
	}
}

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

func TestBuildDevcontainerJSON(t *testing.T) {
	has := func(mounts []string, sub string) bool {
		return slices.ContainsFunc(mounts, func(m string) bool { return strings.Contains(m, sub) })
	}
	bm, benv := devMounts(t, buildDevcontainerJSON(false, false, false))
	if len(bm) != 1 || !has(bm, "/.config/codehalter") {
		t.Errorf("base must be just the config mount, got %v", bm)
	}
	if _, ok := benv["SSH_AUTH_SOCK"]; ok {
		t.Errorf("base must not set SSH_AUTH_SOCK env, got %v", benv)
	}

	gm, _ := devMounts(t, buildDevcontainerJSON(true, false, false))
	if !has(gm, "containerWorkspaceFolder}/.git") {
		t.Errorf("gitWritable must add the .git mount, got %v", gm)
	}
	if has(gm, "/.gitconfig") || has(gm, "ssh-agent") {
		t.Errorf("git-only must not add gitconfig/ssh, got %v", gm)
	}

	gcm, _ := devMounts(t, buildDevcontainerJSON(true, true, false))
	if !has(gcm, "containerWorkspaceFolder}/.git") || !has(gcm, "/.gitconfig") {
		t.Errorf("git+gitconfig must add both, got %v", gcm)
	}

	sm, senv := devMounts(t, buildDevcontainerJSON(false, false, true))
	if !has(sm, "ssh-agent") || has(sm, "/.git,") {
		t.Errorf("ssh-only mounts wrong, got %v", sm)
	}
	if senv["SSH_AUTH_SOCK"] != "/ssh-agent" {
		t.Errorf("ssh must set SSH_AUTH_SOCK=/ssh-agent, got %v", senv)
	}

	am, _ := devMounts(t, buildDevcontainerJSON(true, true, true))
	if len(am) != 4 {
		t.Errorf("all-on should have 4 mounts, got %v", am)
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

func TestEnsureSettingsGitignored(t *testing.T) {
	bare := t.TempDir()
	if ensureSettingsGitignored(bare) {
		t.Errorf("non-git dir with no .gitignore must not gitignore")
	}
	if _, err := os.Stat(filepath.Join(bare, ".gitignore")); !os.IsNotExist(err) {
		t.Errorf("non-git dir: .gitignore must not be created")
	}

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

	worktree := t.TempDir()
	if err := os.WriteFile(filepath.Join(worktree, ".git"), []byte("gitdir: /elsewhere/.git/worktrees/x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !ensureSettingsGitignored(worktree) {
		t.Errorf("linked worktree: must gitignore settings.toml")
	}
}

// Autopilot answers the card with its first option, ignoring .codehalter/.
func TestEnsureGitignore(t *testing.T) {
	for _, c := range []struct {
		name      string
		git       bool
		gitignore *string // nil: no .gitignore
		want      *string // nil: no .gitignore afterwards
	}{
		{name: "neither git nor .gitignore: nothing written", git: false},
		{name: "git repo without .gitignore: created", git: true, want: ptr(".codehalter/\n")},
		{
			name:      "the settings-only entry does not count as the decision",
			git:       true,
			gitignore: ptr("node_modules\n" + gitignoreSettingsEntry),
			want:      ptr("node_modules\n" + gitignoreSettingsEntry + "\n.codehalter/\n"),
		},
		{name: "already decided: untouched", git: true, gitignore: ptr("# .codehalter/ is intentionally tracked\n"), want: ptr("# .codehalter/ is intentionally tracked\n")},
	} {
		t.Run(c.name, func(t *testing.T) {
			a, s := newTestAgent(t)
			a.mode = "Autopilot"
			if c.git {
				if err := os.Mkdir(filepath.Join(s.Cwd, ".git"), 0o755); err != nil {
					t.Fatal(err)
				}
			}
			path := filepath.Join(s.Cwd, ".gitignore")
			if c.gitignore != nil {
				if err := os.WriteFile(path, []byte(*c.gitignore), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			a.ensureGitignore(context.Background(), s.Cwd, s.ID)
			data, err := os.ReadFile(path)
			switch {
			case c.want == nil && !os.IsNotExist(err):
				t.Errorf(".gitignore written: %q, %v", data, err)
			case c.want != nil && string(data) != *c.want:
				t.Errorf(".gitignore = %q, want %q (err %v)", data, *c.want, err)
			}
		})
	}
}
