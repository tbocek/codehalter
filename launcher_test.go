package main

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// HOME and SSH_AUTH_SOCK point into the temp tree: composeFile refuses a missing bind source.
func scaffoldWorkspace(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	ws := filepath.Join(root, "myproject")
	home := filepath.Join(root, "home")
	for _, d := range []string{
		filepath.Join(ws, ".devcontainer"),
		filepath.Join(ws, ".git"),
		filepath.Join(home, ".config", "codehalter"),
	} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", d, err)
		}
	}
	sock := filepath.Join(home, "agent.sock")
	for _, f := range []struct{ path, body string }{
		{filepath.Join(ws, ".devcontainer", "devcontainer.json"), buildDevcontainerJSON(true, true, true)},
		{filepath.Join(ws, ".devcontainer", "Dockerfile"), "FROM alpine\n"},
		{filepath.Join(home, ".gitconfig"), "[user]\n"},
		{sock, ""},
	} {
		if err := os.WriteFile(f.path, []byte(f.body), 0o644); err != nil {
			t.Fatalf("write %s: %v", f.path, err)
		}
	}
	t.Setenv("HOME", home)
	t.Setenv("SSH_AUTH_SOCK", sock)
	return ws
}

func writeConfig(t *testing.T, files map[string]string) string {
	t.Helper()
	ws := t.TempDir()
	for rel, body := range files {
		full := filepath.Join(ws, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return ws
}

// dig fails on a missing key, so a renamed compose key shows up as the assertion it broke.
func dig(t *testing.T, doc any, path ...string) any {
	t.Helper()
	cur := doc
	for i, p := range path {
		m, ok := cur.(map[string]any)
		if !ok {
			t.Fatalf("%s: not an object", strings.Join(path[:i], "."))
		}
		cur, ok = m[p]
		if !ok {
			t.Fatalf("%s: missing", strings.Join(path[:i+1], "."))
		}
	}
	return cur
}

// Pins the derived ${containerWorkspaceFolder} mount and the compose escaping of $.
func TestScaffoldConfigBecomesCompose(t *testing.T) {
	ws := scaffoldWorkspace(t)
	cfg, err := loadDevcontainerConfig(ws)
	if err != nil {
		t.Fatalf("loadDevcontainerConfig: %v", err)
	}
	body, err := cfg.composeFile()
	if err != nil {
		t.Fatalf("composeFile: %v", err)
	}
	if strings.Contains(string(body), "${") {
		t.Errorf("unexpanded variable left in the compose file:\n%s", body)
	}
	var doc any
	if err := json.Unmarshal(body, &doc); err != nil {
		t.Fatalf("generated compose is not valid JSON: %v\n%s", err, body)
	}
	svc := dig(t, doc, "services", "dev")

	if got := dig(t, svc, "working_dir"); got != "/workspaces/myproject" {
		t.Errorf("working_dir: got %v", got)
	}
	if got := dig(t, svc, "container_name"); got != "codehalter-myproject" {
		t.Errorf("container_name: got %v (runArgs --name must survive)", got)
	}
	if got := dig(t, svc, "extra_hosts").([]any); len(got) != 1 || got[0] != "host.docker.internal:host-gateway" {
		t.Errorf("extra_hosts: got %v", got)
	}
	if got := dig(t, svc, "environment", "DEVCONTAINER"); got != "true" {
		t.Errorf("containerEnv did not become environment: got %v", got)
	}
	if got := dig(t, svc, "build", "dockerfile"); got != filepath.Join(ws, ".devcontainer", "Dockerfile") {
		t.Errorf("build.dockerfile: got %v", got)
	}
	if got := dig(t, svc, "build", "context"); got != filepath.Join(ws, ".devcontainer") {
		t.Errorf("build.context: got %v", got)
	}
	// The entrypoint's $! must reach compose doubled.
	if got := dig(t, svc, "entrypoint").([]any); !strings.Contains(got[2].(string), "$$!") {
		t.Errorf("entrypoint not escaped for compose: %v", got)
	}

	// The .git mount is written in terms of ${containerWorkspaceFolder}, which the file never states.
	vols := dig(t, svc, "volumes").([]any)
	want := map[string]struct {
		source string
		ro     bool
	}{
		"/workspaces/myproject":        {ws, false},
		"/workspaces/myproject/.git":   {filepath.Join(ws, ".git"), false},
		"/home/dev/.config/codehalter": {filepath.Join(os.Getenv("HOME"), ".config", "codehalter"), true},
		"/home/dev/.gitconfig":         {filepath.Join(os.Getenv("HOME"), ".gitconfig"), true},
		"/ssh-agent":                   {os.Getenv("SSH_AUTH_SOCK"), false},
	}
	if len(vols) != len(want) {
		t.Fatalf("got %d mounts, want %d: %v", len(vols), len(want), vols)
	}
	for _, v := range vols {
		m := v.(map[string]any)
		target, _ := m["target"].(string)
		exp, ok := want[target]
		if !ok {
			t.Errorf("unexpected mount target %q", target)
			continue
		}
		if m["source"] != exp.source {
			t.Errorf("%s: source %v, want %v", target, m["source"], exp.source)
		}
		if ro, _ := m["read_only"].(bool); ro != exp.ro {
			t.Errorf("%s: read_only %v, want %v", target, ro, exp.ro)
		}
		if m["type"] != "bind" {
			t.Errorf("%s: type %v, want bind", target, m["type"])
		}
	}
}

func TestGeneratedComposeParses(t *testing.T) {
	rt := containerTool()
	if rt == "" {
		t.Skip("no container runtime with a compose plugin")
	}
	ws := scaffoldWorkspace(t)
	cfg, err := loadDevcontainerConfig(ws)
	if err != nil {
		t.Fatalf("loadDevcontainerConfig: %v", err)
	}
	body, err := cfg.composeFile()
	if err != nil {
		t.Fatalf("composeFile: %v", err)
	}
	path := filepath.Join(t.TempDir(), "compose.yaml")
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	out, err := exec.Command(rt, "compose", "-f", path, "config").CombinedOutput()
	if err != nil {
		t.Fatalf("%s compose config rejected the generated file: %v\n%s\n--- file ---\n%s", rt, err, out, body)
	}

	// $HOME probes the escaping: compose would substitute it from the host if it were not doubled.
	ws2 := writeConfig(t, map[string]string{
		".devcontainer/devcontainer.json": `{"image": "alpine", "containerEnv": {"LITERAL": "$HOME/x"}}`,
	})
	cfg2, err := loadDevcontainerConfig(ws2)
	if err != nil {
		t.Fatalf("loadDevcontainerConfig: %v", err)
	}
	body2, err := cfg2.composeFile()
	if err != nil {
		t.Fatalf("composeFile: %v", err)
	}
	path2 := filepath.Join(t.TempDir(), "compose.yaml")
	if err := os.WriteFile(path2, body2, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	out, err = exec.Command(rt, "compose", "-f", path2, "config").CombinedOutput()
	if err != nil {
		t.Fatalf("%s compose config: %v\n%s", rt, err, out)
	}
	if strings.Contains(string(out), os.Getenv("HOME")+"/x") {
		t.Errorf("compose interpolated a value that should have been escaped:\n%s", out)
	}
}

func TestRefusesWhatItCannotBuild(t *testing.T) {
	ws := writeConfig(t, map[string]string{".devcontainer/devcontainer.json": `{
	  "image": "alpine",
	  "features": { "ghcr.io/devcontainers/features/node:1": {} },
	  "postCreateCommand": "npm install"
	}`})
	_, err := loadDevcontainerConfig(ws)
	var unsupported *unsupportedConfig
	if !errors.As(err, &unsupported) {
		t.Fatalf("got %v, want an unsupportedConfig", err)
	}
	if strings.Join(unsupported.keys, ",") != "features,postCreateCommand" {
		t.Errorf("refused keys: got %v", unsupported.keys)
	}
}

func TestNotesInsteadOfRefusing(t *testing.T) {
	ws := writeConfig(t, map[string]string{
		".devcontainer/devcontainer.json": `{"image": "alpine", "updateRemoteUserUID": true, "name": "whatever"}`,
	})
	cfg, err := loadDevcontainerConfig(ws)
	if err != nil {
		t.Fatalf("loadDevcontainerConfig: %v", err)
	}
	if len(cfg.notes) != 1 || !strings.HasPrefix(cfg.notes[0], "updateRemoteUserUID:") {
		t.Errorf("notes: got %v, want one about updateRemoteUserUID", cfg.notes)
	}
}

func TestStripJSONC(t *testing.T) {
	for _, tc := range []struct{ name, in, want string }{
		{"line comment", "{\n // hi\n \"a\": 1\n}", "{\n \n \"a\": 1\n}"},
		{"block comment", `{/* hi */"a": 1}`, `{"a": 1}`},
		{"trailing comma", "{\"a\": 1,\n}", "{\"a\": 1\n}"},
		{"trailing comma in array", `{"a": [1, 2, ]}`, `{"a": [1, 2 ]}`},
		{"comma then comment", "{\"a\": 1, // why\n}", "{\"a\": 1 \n}"},
		{"slashes in a string", `{"a": "https://x/y"}`, `{"a": "https://x/y"}`},
		{"escaped quote", `{"a": "he said \"hi\" // no"}`, `{"a": "he said \"hi\" // no"}`},
		{"byte order mark", "\ufeff{\"a\": 1}", `{"a": 1}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := string(stripJSONC([]byte(tc.in))); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
			var v any
			if err := json.Unmarshal(stripJSONC([]byte(tc.in)), &v); err != nil {
				t.Errorf("result does not parse: %v", err)
			}
		})
	}
}

func TestMountForms(t *testing.T) {
	for _, tc := range []struct {
		name, in string
		want     map[string]any
	}{
		{"long", `"source=/a,target=/b,type=bind,readonly"`,
			map[string]any{"type": "bind", "source": "/a", "target": "/b", "read_only": true}},
		{"aliases", `"src=/a,dst=/b"`,
			map[string]any{"type": "bind", "source": "/a", "target": "/b"}},
		{"volume", `"source=vol,target=/b,type=volume"`,
			map[string]any{"type": "volume", "source": "vol", "target": "/b"}},
		{"object", `{"source": "/a", "target": "/b", "type": "bind"}`,
			map[string]any{"type": "bind", "source": "/a", "target": "/b"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseMountJSON([]byte(tc.in))
			if err != nil {
				t.Fatalf("parseMountJSON: %v", err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
			for k, v := range tc.want {
				if got[k] != v {
					t.Errorf("%s: got %v, want %v", k, got[k], v)
				}
			}
		})
	}
}

func TestRunArgsWithoutComposeEquivalent(t *testing.T) {
	err := applyRunArgs(map[string]any{}, []string{"--gpus", "all"})
	if err == nil || !strings.Contains(err.Error(), "--gpus") {
		t.Fatalf("got %v, want an error naming --gpus", err)
	}
	err = applyRunArgs(map[string]any{}, []string{"-v", "/a:/b"})
	if err == nil || !strings.Contains(err.Error(), "mounts") {
		t.Fatalf("got %v, want an error pointing at the mounts key", err)
	}
}

// A bare -e NAME takes the host's value, as docker does.
func TestRunArgsEnvPassthrough(t *testing.T) {
	t.Setenv("CODEHALTER_PROBE", "from-the-host")
	svc := map[string]any{}
	if err := applyRunArgs(svc, []string{"-e", "CODEHALTER_PROBE", "-e", "OTHER=literal"}); err != nil {
		t.Fatalf("applyRunArgs: %v", err)
	}
	env := svc["environment"].(map[string]string)
	if env["CODEHALTER_PROBE"] != "from-the-host" || env["OTHER"] != "literal" {
		t.Errorf("environment: got %v", env)
	}
}

func TestExpandVarsRejectsTheUnknown(t *testing.T) {
	_, err := expandVars([]byte(`{"a": "${nonsense}"}`), map[string]string{})
	if err == nil || !strings.Contains(err.Error(), "nonsense") {
		t.Fatalf("got %v, want an error naming the variable", err)
	}
	// containerEnv is left for resolveContainerEnv.
	kept, err := expandVars([]byte(`{"a": "${containerEnv:PATH}:/x"}`), map[string]string{})
	if err != nil || string(kept) != `{"a": "${containerEnv:PATH}:/x"}` {
		t.Fatalf("containerEnv: got %s, %v; want it left in place", kept, err)
	}
	got, err := expandVars([]byte(`{"a": "${localEnv:NOPE:fallback}"}`), map[string]string{})
	if err != nil {
		t.Fatalf("localEnv default: %v", err)
	}
	if string(got) != `{"a": "fallback"}` {
		t.Errorf("localEnv default: got %s", got)
	}
}

// Several side-by-side configurations are named, not guessed.
func TestConfigLocations(t *testing.T) {
	for _, tc := range []struct {
		name  string
		files map[string]string
		want  string
	}{
		{"folder", map[string]string{".devcontainer/devcontainer.json": `{"image":"a"}`}, "a"},
		{"dotfile", map[string]string{".devcontainer.json": `{"image":"b"}`}, "b"},
		{"named subfolder", map[string]string{".devcontainer/rust/devcontainer.json": `{"image":"c"}`}, "c"},
		{"folder wins over dotfile", map[string]string{
			".devcontainer/devcontainer.json": `{"image":"d"}`,
			".devcontainer.json":              `{"image":"e"}`,
		}, "d"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := loadDevcontainerConfig(writeConfig(t, tc.files))
			if err != nil {
				t.Fatalf("loadDevcontainerConfig: %v", err)
			}
			if cfg.Image != tc.want {
				t.Errorf("read %q, want the config with image %q", cfg.Image, tc.want)
			}
		})
	}

	ws := writeConfig(t, map[string]string{
		".devcontainer/rust/devcontainer.json": `{"image":"a"}`,
		".devcontainer/node/devcontainer.json": `{"image":"b"}`,
	})
	_, err := loadDevcontainerConfig(ws)
	if err == nil || !strings.Contains(err.Error(), "node, rust") {
		t.Errorf("got %v, want an error naming both configurations", err)
	}

	if _, err := loadDevcontainerConfig(t.TempDir()); err != os.ErrNotExist {
		t.Errorf("empty project: got %v, want os.ErrNotExist so the CLI runs on the host", err)
	}
}

func TestCosmeticKeysAreNotExpanded(t *testing.T) {
	ws := writeConfig(t, map[string]string{".devcontainer/devcontainer.json": `{
	  "image": "alpine",
	  "customizations": {"vscode": {"settings": {
	    "go.testFlags": ["${workspaceFolder}/x"],
	    "terminal.integrated.env.linux": {"A": "${env:HOME}"}
	  }}},
	  "portsAttributes": {"3000": {"label": "${localWorkspaceFolderBasename} app"}}
	}`})
	if _, err := loadDevcontainerConfig(ws); err != nil {
		t.Fatalf("loadDevcontainerConfig: %v", err)
	}
}

func TestBlankKeysAreNotRefused(t *testing.T) {
	ws := writeConfig(t, map[string]string{
		".devcontainer/devcontainer.json": `{"image":"alpine","features":{},"postCreateCommand":null,"initializeCommand":""}`,
	})
	if _, err := loadDevcontainerConfig(ws); err != nil {
		t.Fatalf("loadDevcontainerConfig: %v", err)
	}
}

// Docker creates a missing working dir, so both cases would otherwise start in an empty directory.
func TestWorkspaceFolderMustBeReachable(t *testing.T) {
	ws := writeConfig(t, map[string]string{
		".devcontainer/devcontainer.json": `{"dockerComposeFile":"compose.yml","service":"app"}`,
		".devcontainer/compose.yml":       "services:\n  app:\n    image: alpine\n",
	})
	_, err := loadDevcontainerConfig(ws)
	if err == nil || !strings.Contains(err.Error(), "workspaceFolder") {
		t.Errorf("dockerComposeFile without workspaceFolder: got %v, want an error naming it", err)
	}

	ws = writeConfig(t, map[string]string{
		".devcontainer/devcontainer.json": `{"image":"alpine","workspaceMount":"source=${localWorkspaceFolder},target=/src,type=bind"}`,
	})
	cfg, err := loadDevcontainerConfig(ws)
	if err != nil {
		t.Fatalf("loadDevcontainerConfig: %v", err)
	}
	if _, err := cfg.composeFile(); err == nil || !strings.Contains(err.Error(), "/src") {
		t.Errorf("workspaceMount pointing elsewhere: got %v, want an error naming the mismatch", err)
	}
}

func TestComposeFileCorners(t *testing.T) {
	svcOf := func(t *testing.T, config string) map[string]any {
		t.Helper()
		cfg, err := loadDevcontainerConfig(writeConfig(t, map[string]string{".devcontainer/devcontainer.json": config}))
		if err != nil {
			t.Fatalf("loadDevcontainerConfig: %v", err)
		}
		body, err := cfg.composeFile()
		if err != nil {
			t.Fatalf("composeFile: %v", err)
		}
		var doc map[string]map[string]map[string]any
		if err := json.Unmarshal(body, &doc); err != nil {
			t.Fatalf("generated compose is not valid JSON: %v", err)
		}
		return doc["services"]["dev"]
	}

	ports := svcOf(t, `{"image":"alpine","forwardPorts":[3000]}`)["ports"]
	if got, ok := ports.([]any); !ok || len(got) != 1 || got[0] != "127.0.0.1:3000:3000" {
		t.Errorf("forwardPorts: got %v, want it bound to the loopback address", ports)
	}

	if got := svcOf(t, `{"image":"alpine","forwardPorts":[3000],"runArgs":["--network=host"]}`); got["ports"] != nil {
		t.Errorf("network host: ports %v, want none", got["ports"])
	}

	vols := svcOf(t, `{"image":"alpine","mounts":["target=/cache,type=volume"]}`)["volumes"].([]any)
	if len(vols) != 2 || vols[1].(map[string]any)["source"] != nil {
		t.Errorf("anonymous volume: got %v", vols)
	}

	cfg, err := loadDevcontainerConfig(writeConfig(t, map[string]string{
		".devcontainer/devcontainer.json": `{"image":"alpine","mounts":["source=./data,target=/data,type=bind"]}`,
		"data/keep":                       "",
	}))
	if err != nil {
		t.Fatalf("loadDevcontainerConfig: %v", err)
	}
	if _, err := cfg.composeFile(); err == nil || !strings.Contains(err.Error(), "relative") {
		t.Errorf("relative bind source: got %v, want it refused", err)
	}
}

// ${containerEnv:...} is only knowable once the container is up, so it is refused outside remoteEnv.
func TestRemoteEnvReadsTheContainer(t *testing.T) {
	ws := writeConfig(t, map[string]string{".devcontainer/devcontainer.json": `{
	  "image": "alpine",
	  "remoteEnv": {"PATH": "${containerEnv:PATH}:/opt/bin", "HOME": "${containerEnv:NOPE:/root}"}
	}`})
	cfg, err := loadDevcontainerConfig(ws)
	if err != nil {
		t.Fatalf("loadDevcontainerConfig: %v", err)
	}
	calls := 0
	if err := resolveContainerEnv(cfg.RemoteEnv, func() ([]byte, error) {
		calls++
		return []byte("PATH=/usr/bin\nSHELL=/bin/sh\n"), nil
	}); err != nil {
		t.Fatalf("resolveContainerEnv: %v", err)
	}
	if calls != 1 {
		t.Errorf("asked the container %d times, want exactly one round trip", calls)
	}
	if got := cfg.RemoteEnv["PATH"]; got != "/usr/bin:/opt/bin" {
		t.Errorf("PATH: got %q", got)
	}
	if got := cfg.RemoteEnv["HOME"]; got != "/root" {
		t.Errorf("HOME: got %q, want the default for a variable the container does not set", got)
	}

	calls = 0
	plain := map[string]string{"A": "b"}
	if err := resolveContainerEnv(plain, func() ([]byte, error) { calls++; return nil, nil }); err != nil || calls != 0 {
		t.Errorf("plain remoteEnv: %d calls, %v; want none", calls, err)
	}

	ws = writeConfig(t, map[string]string{
		".devcontainer/devcontainer.json": `{"image":"alpine","containerEnv":{"PATH":"${containerEnv:PATH}:/x"}}`,
	})
	if _, err := loadDevcontainerConfig(ws); err == nil || !strings.Contains(err.Error(), "remoteEnv") {
		t.Errorf("containerEnv: got %v, want it refused with the reason", err)
	}
}
