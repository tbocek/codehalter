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

// scaffoldWorkspace writes the devcontainer.json codehalter itself scaffolds
// (with every optional mount switched on, which is the widest config it ever
// produces) into a temp workspace, plus every file those mounts bind. HOME and
// SSH_AUTH_SOCK point at the temp tree so ${localEnv:...} resolves to something
// that exists: composeFile refuses a bind whose source is missing.
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

// writeConfig lays out a workspace from relative path to content and returns
// it. Every test here starts by writing a devcontainer.json somewhere under it.
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

// dig walks a decoded JSON document. Missing keys fail the test rather than
// returning nil, so a renamed compose key shows up as the assertion it broke.
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

// TestScaffoldConfigBecomesCompose is the end-to-end of the translation: the
// devcontainer.json codehalter writes, through the parser and the compose
// renderer, checked key by key. It pins the two substitutions that are easy to
// get wrong (a mount written in terms of ${containerWorkspaceFolder}, which is
// itself derived) and the compose escaping of $.
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
	// The keep-alive entrypoint holds a $! that compose would interpolate, so
	// it has to arrive doubled and nothing else may.
	if got := dig(t, svc, "entrypoint").([]any); !strings.Contains(got[2].(string), "$$!") {
		t.Errorf("entrypoint not escaped for compose: %v", got)
	}

	// Mounts: the workspace bind comes first, then the four from the config.
	// The .git one is the interesting one: it is written in terms of
	// ${containerWorkspaceFolder}, which nothing in the file states.
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

// TestGeneratedComposeParses feeds the generated file to the real compose
// parser. Emitting JSON and calling it YAML is the whole quoting strategy, so
// it is worth one check against the thing that has to accept it.
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

	// And the escaping, which only the real parser can confirm: compose reads
	// $VAR out of the host environment when it loads a file, so a value that
	// survived to here unescaped would come back replaced. $HOME is the cheap
	// probe because it is always set and never what the config meant.
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

// TestRefusesWhatItCannotBuild pins the hand-off. features and the lifecycle
// commands change what ends up inside the container, and a compose file cannot
// express either, so a config using them must be named and refused rather than
// half-built.
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

// TestNotesInsteadOfRefusing covers the other half of the policy: keys that are
// spec defaults cannot be refused (every config has them, written down or not),
// so they produce one line saying how the result differs.
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

// TestStripJSONC pins the comment and trailing-comma handling, including the
// case that makes it more than a regex: a // inside a string literal.
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

// TestMountForms covers the spellings a devcontainer.json may use for the same
// mount: the docker --mount string with its aliases and bare readonly, and the
// object form.
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

// TestRunArgsWithoutComposeEquivalent: compose has no passthrough for raw
// docker flags, so an unmapped one has to be an error. Dropping it would start
// a container quietly missing whatever the flag was for.
func TestRunArgsWithoutComposeEquivalent(t *testing.T) {
	err := applyRunArgs(map[string]any{}, []string{"--gpus", "all"})
	if err == nil || !strings.Contains(err.Error(), "--gpus") {
		t.Fatalf("got %v, want an error naming --gpus", err)
	}
	// One that does have a home in devcontainer.json: saying where it belongs
	// beats the bare news that compose cannot take it.
	err = applyRunArgs(map[string]any{}, []string{"-v", "/a:/b"})
	if err == nil || !strings.Contains(err.Error(), "mounts") {
		t.Fatalf("got %v, want an error pointing at the mounts key", err)
	}
}

// TestRunArgsEnvPassthrough: docker reads a bare -e NAME out of the calling
// environment, and a config that says it means the host's value, not an empty
// one, which is what a plain Cut on = would have produced.
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

// TestExpandVarsRejectsTheUnknown: a ${...} nobody resolved would travel on as
// a literal into a path or an image tag and fail somewhere unrecognisable.
func TestExpandVarsRejectsTheUnknown(t *testing.T) {
	_, err := expandVars([]byte(`{"a": "${nonsense}"}`), map[string]string{})
	if err == nil || !strings.Contains(err.Error(), "nonsense") {
		t.Fatalf("got %v, want an error naming the variable", err)
	}
	// The one variable that is deliberately not resolved here: nothing knows
	// the container's environment until the container is running.
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

// TestConfigLocations: the spec allows three places, and a project keeping
// several configurations side by side has no single answer, so that one is
// named instead of guessed at. Missing the folder-per-config form would be
// worse than an error: the CLI would run on the host and the agent would then
// offer to scaffold a second devcontainer next to the one already there.
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

// TestCosmeticKeysAreNotExpanded: customizations carries the editor's own
// settings, which have their own ${...} vocabulary. Expanding those would
// refuse a perfectly ordinary config over a value nothing here reads.
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

// TestBlankKeysAreNotRefused: a template that had its last feature deleted
// keeps an empty "features": {}, which asks for nothing at all. Refusing that
// would be refusing a config over punctuation.
func TestBlankKeysAreNotRefused(t *testing.T) {
	ws := writeConfig(t, map[string]string{
		".devcontainer/devcontainer.json": `{"image":"alpine","features":{},"postCreateCommand":null,"initializeCommand":""}`,
	})
	if _, err := loadDevcontainerConfig(ws); err != nil {
		t.Fatalf("loadDevcontainerConfig: %v", err)
	}
}

// TestWorkspaceFolderMustBeReachable covers the two ways the agent could end up
// in a directory that holds nothing. Docker creates a missing working
// directory, so neither of these fails on its own: the container starts, and
// the project is simply not in it.
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

// TestComposeFileCorners pins the translations that a wrong answer would hide
// rather than announce: a port on the wrong interface, a mount compose would
// resolve against its own directory, a volume that has no name to give.
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

	// A forwarded port is for the person at the keyboard. Published on every
	// interface it would be for the coffee shop as well.
	ports := svcOf(t, `{"image":"alpine","forwardPorts":[3000]}`)["ports"]
	if got, ok := ports.([]any); !ok || len(got) != 1 || got[0] != "127.0.0.1:3000:3000" {
		t.Errorf("forwardPorts: got %v, want it bound to the loopback address", ports)
	}

	// Host networking already puts every port on the host, and compose refuses
	// a service that asks for both.
	if got := svcOf(t, `{"image":"alpine","forwardPorts":[3000],"runArgs":["--network=host"]}`); got["ports"] != nil {
		t.Errorf("network host: ports %v, want none", got["ports"])
	}

	// An anonymous volume has no name to pin, and asking for one refused a
	// mount docker is perfectly happy with.
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
	// Compose resolves a relative source against the directory holding the file
	// it read, which is a cache directory nobody wrote that path against.
	if _, err := cfg.composeFile(); err == nil || !strings.Contains(err.Error(), "relative") {
		t.Errorf("relative bind source: got %v, want it refused", err)
	}
}

// TestRemoteEnvReadsTheContainer: "${containerEnv:PATH}:/opt/bin" is the
// documented way to extend PATH for the tools a devcontainer installs, and the
// value is only knowable once the container is up. Anywhere else it has to be
// refused, because the answer would be needed before there is anything to ask.
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

	// Nothing to substitute means nothing to ask, so no container is started
	// on account of remoteEnv alone.
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
