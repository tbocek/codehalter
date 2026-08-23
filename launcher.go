package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
)

// ---------------------------------------------------------------------------
// Starting the devcontainer for the standalone CLI.
//
// codehalter refuses to run outside a container. The editor front end gets that
// for free: Zed reopens the project in the container and starts the agent
// there. At a shell prompt nobody does it for you, so --cli does it itself. It
// reads .devcontainer/devcontainer.json, translates the keys it understands
// into a generated compose file, and runs
//
//	<runtime> compose up -d      then     <runtime> compose exec ... codehalter --cli
//
// with the child's stdio wired straight to the terminal. What you end up
// talking to is an ordinary codehalter CLI that happens to live in a container,
// not a second front end.
//
// Why compose rather than plain `docker run`: compose already does the two
// things this would otherwise have to reimplement. It hashes a service
// definition and recreates the container when the definition changes, and it
// gives the whole thing one name to start, exec into and stop. Both are easy to
// get subtly wrong by hand, and "I edited the Dockerfile and nothing happened"
// is a bug this project would hit often, because the agent is forever proposing
// Dockerfile edits.
//
// This is deliberately NOT a devcontainer implementation. Keys that change what
// ends up inside the container but that a compose file cannot express (features
// above all, plus every lifecycle command) are refused BY NAME, with the
// devcontainer CLI command to run instead. A config we cannot model completely
// is one we hand off, never one we half-build: a container silently missing
// half its setup is a worse outcome than a message saying so. The same reason
// ensureTerminals refuses rather than falling back to running commands itself.
//
// Known differences from `devcontainer up`, all listed in the README:
//   - updateRemoteUserUID is not implemented, so the container user keeps the
//     uid baked into the image.
//   - userEnvProbe is not implemented: commands get the image's environment
//     rather than one probed from a login shell.
//   - ${devcontainerId} is derived from the workspace path, so it does not
//     match the id the devcontainer CLI would compute for the same folder.
//   - shutdownAction defaults to leaving the container running rather than to
//     stopping it, because the next `codehalter --cli` is then instant.
// ---------------------------------------------------------------------------

// launcherService is the service name in a generated compose file. Fixed,
// because a generated project has exactly one service.
const launcherService = "dev"

// keepAlive is the entrypoint a generated service runs when overrideCommand is
// on (the spec's default). The image's own CMD would exit immediately and take
// the container with it. `sleep & wait` rather than a bare sleep so a TERM
// during the sleep is acted on at once instead of ten seconds later, and sh
// rather than `sleep infinity` because busybox sleep (Alpine) wants a number.
var keepAlive = []string{"/bin/sh", "-c", `trap 'exit 0' TERM; while sleep 3600 & wait $!; do :; done`}

// dcImplemented lists the devcontainer.json keys the launcher translates.
var dcImplemented = map[string]bool{
	"image": true, "build": true, "dockerFile": true, "context": true,
	"runArgs": true, "containerEnv": true, "remoteEnv": true,
	"containerUser": true, "remoteUser": true, "mounts": true,
	"workspaceFolder": true, "workspaceMount": true, "forwardPorts": true,
	"overrideCommand": true, "init": true, "privileged": true,
	"capAdd": true, "securityOpt": true, "shutdownAction": true,
	"dockerComposeFile": true, "service": true, "runServices": true,
}

// dcCosmetic lists keys that describe the editor's view of the container rather
// than the container itself, so ignoring them changes nothing that runs.
var dcCosmetic = map[string]bool{
	"name": true, "customizations": true, "portsAttributes": true,
	"otherPortsAttributes": true, "hostRequirements": true, "waitFor": true,
}

// dcNoted lists keys that DO change the container but that the launcher does
// not implement. Each prints one line saying how the result will differ, rather
// than refusing the file: both are spec defaults, so a config that names them
// is usually just writing down what it was getting anyway.
var dcNoted = map[string]string{
	"updateRemoteUserUID": "the container user keeps the uid baked into the image, so files written in the workspace may not end up owned by you",
	"userEnvProbe":        "commands run with the image's environment, not one probed from a login shell",
}

// devcontainerConfig is the subset of devcontainer.json the launcher models.
// Every string in here has already been through variable substitution.
type devcontainerConfig struct {
	Image string `json:"image"`
	Build struct {
		Dockerfile string            `json:"dockerfile"`
		Context    string            `json:"context"`
		Target     string            `json:"target"`
		Args       map[string]string `json:"args"`
	} `json:"build"`
	// Pre-spec spelling, still found in older configs and still accepted by
	// the devcontainer CLI.
	DockerFile string `json:"dockerFile"`
	Context    string `json:"context"`

	RunArgs         []string          `json:"runArgs"`
	ContainerEnv    map[string]string `json:"containerEnv"`
	RemoteEnv       map[string]string `json:"remoteEnv"`
	ContainerUser   string            `json:"containerUser"`
	RemoteUser      string            `json:"remoteUser"`
	Mounts          []json.RawMessage `json:"mounts"`
	WorkspaceFolder string            `json:"workspaceFolder"`
	WorkspaceMount  string            `json:"workspaceMount"`
	ForwardPorts    []any             `json:"forwardPorts"`
	OverrideCommand *bool             `json:"overrideCommand"`
	Init            *bool             `json:"init"`
	Privileged      *bool             `json:"privileged"`
	CapAdd          []string          `json:"capAdd"`
	SecurityOpt     []string          `json:"securityOpt"`
	ShutdownAction  string            `json:"shutdownAction"`

	// Compose flavour: the user's own compose files are used as they are, so
	// none of the fields above apply.
	ComposeFiles []string `json:"-"`
	Service      string   `json:"service"`
	RunServices  []string `json:"runServices"`

	path      string   // the devcontainer.json we read
	dir       string   // directory holding it, the base for relative paths
	workspace string   // absolute host workspace folder
	notes     []string // dcNoted lines to show once at startup
}

// unsupportedConfig is the refusal: the named keys are real devcontainer
// features that a generated compose file cannot express.
type unsupportedConfig struct {
	path string
	keys []string
}

func (e *unsupportedConfig) Error() string {
	return fmt.Sprintf("%s uses %s, which the built-in launcher does not implement",
		e.path, strings.Join(e.keys, ", "))
}

// stripJSONC removes what devcontainer.json is allowed to contain and
// encoding/json is not: // and /* */ comments, and a comma before a closing
// brace or bracket. Two passes, because a trailing comma is only visible as one
// once the comment that followed it is gone. Both passes track string literals,
// which is the whole difficulty: a // inside a URL is not a comment.
func stripJSONC(src []byte) []byte {
	src = bytes.TrimPrefix(src, []byte("\xef\xbb\xbf")) // a BOM, which some editors still write
	out := make([]byte, 0, len(src))
	inStr, esc := false, false
	for i := 0; i < len(src); i++ {
		c := src[i]
		if inStr {
			out = append(out, c)
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch {
		case c == '"':
			inStr = true
			out = append(out, c)
		case c == '/' && i+1 < len(src) && src[i+1] == '/':
			for i < len(src) && src[i] != '\n' {
				i++
			}
			out = append(out, '\n')
		case c == '/' && i+1 < len(src) && src[i+1] == '*':
			for i += 2; i < len(src); i++ {
				if src[i] == '*' && i+1 < len(src) && src[i+1] == '/' {
					i++
					break
				}
			}
		default:
			out = append(out, c)
		}
	}

	final := make([]byte, 0, len(out))
	inStr, esc = false, false
	for i := 0; i < len(out); i++ {
		c := out[i]
		if inStr {
			final = append(final, c)
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		if c == '"' {
			inStr = true
		}
		if c == ',' {
			j := i + 1
			for j < len(out) && (out[j] == ' ' || out[j] == '\t' || out[j] == '\n' || out[j] == '\r') {
				j++
			}
			if j < len(out) && (out[j] == '}' || out[j] == ']') {
				continue
			}
		}
		final = append(final, c)
	}
	return final
}

// expandVars replaces ${...} in the raw JSON text, before it is parsed, so one
// pass covers every value in the document. Replacements are JSON-escaped
// because they land inside string literals. An unknown variable is an error:
// leaving it in place would produce a path or an image tag with a literal
// ${...} in it, which fails later and further from the cause.
func expandVars(src []byte, vars map[string]string) ([]byte, error) {
	var out []byte
	for i := 0; i < len(src); i++ {
		if src[i] != '$' || i+1 >= len(src) || src[i+1] != '{' {
			out = append(out, src[i])
			continue
		}
		end := strings.IndexByte(string(src[i:]), '}')
		if end < 0 {
			return nil, fmt.Errorf("unterminated ${ at byte %d", i)
		}
		name := string(src[i+2 : i+end])
		val, ok := vars[name]
		switch {
		case ok:
		case strings.HasPrefix(name, "containerEnv:"):
			// The only variable whose value lives inside a container that does
			// not exist yet. Left standing for resolveContainerEnv to fill in
			// once `up` has run, which is why it is legal in remoteEnv and
			// nowhere else.
			out = append(out, src[i:i+end+1]...)
			i += end
			continue
		case strings.HasPrefix(name, "localEnv:"):
			key, def, _ := strings.Cut(strings.TrimPrefix(name, "localEnv:"), ":")
			val = os.Getenv(key)
			if val == "" {
				val = def
			}
		default:
			return nil, fmt.Errorf("${%s} is not a variable the launcher can resolve", name)
		}
		quoted, err := json.Marshal(val)
		if err != nil {
			return nil, fmt.Errorf("escaping ${%s}: %w", name, err)
		}
		out = append(out, quoted[1:len(quoted)-1]...) // drop the quotes json.Marshal added
		i += end
	}
	return out, nil
}

// devcontainerPath finds the project's configuration file. The spec allows
// three locations, and the third one, a folder per configuration under
// .devcontainer/, can hold several: there is no way to say which of those is
// meant, so that is named rather than guessed at.
func devcontainerPath(workspace string) (string, error) {
	for _, p := range []string{
		filepath.Join(workspace, ".devcontainer", "devcontainer.json"),
		filepath.Join(workspace, ".devcontainer.json"),
	} {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	found, err := filepath.Glob(filepath.Join(workspace, ".devcontainer", "*", "devcontainer.json"))
	if err != nil {
		return "", err
	}
	switch len(found) {
	case 0:
		return "", os.ErrNotExist
	case 1:
		return found[0], nil
	}
	names := make([]string, len(found))
	for i, f := range found {
		names[i] = filepath.Base(filepath.Dir(f))
	}
	sort.Strings(names)
	return "", fmt.Errorf("%s holds several configurations (%s), and the launcher has no flag to pick one",
		filepath.Join(workspace, ".devcontainer"), strings.Join(names, ", "))
}

// blank reports whether a key carries no instruction: an empty object, array or
// string, or null. Templates leave "features": {} behind when the last feature
// is deleted, and refusing a config over a key that asks for nothing would be
// refusing it over punctuation.
func blank(raw json.RawMessage) bool {
	switch strings.TrimSpace(string(raw)) {
	case "{}", "[]", `""`, "null", "":
		return true
	}
	return false
}

// loadDevcontainerConfig reads and validates the project's devcontainer.json.
// It returns os.ErrNotExist when the project has none, which is not a failure:
// the CLI then runs on the host and the agent's own bootstrap offers to
// scaffold one.
func loadDevcontainerConfig(workspace string) (*devcontainerConfig, error) {
	path, err := devcontainerPath(workspace)
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	clean := stripJSONC(raw)

	var keys map[string]json.RawMessage
	if err := json.Unmarshal(clean, &keys); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	cfg := &devcontainerConfig{path: path, dir: filepath.Dir(path), workspace: workspace}
	var refused []string
	for k, v := range keys {
		switch {
		case blank(v), dcImplemented[k], dcCosmetic[k]:
		case dcNoted[k] != "":
			cfg.notes = append(cfg.notes, k+": "+dcNoted[k])
		default:
			refused = append(refused, k)
		}
	}
	if len(refused) > 0 {
		sort.Strings(refused)
		return nil, &unsupportedConfig{path: path, keys: refused}
	}
	sort.Strings(cfg.notes)

	// containerWorkspaceFolder has to be resolved first: other values are
	// written in terms of it (the .git mount codehalter scaffolds is), and it
	// may itself be written in terms of the local ones.
	base := filepath.Base(workspace)
	vars := map[string]string{
		"localWorkspaceFolder":         workspace,
		"localWorkspaceFolderBasename": base,
		"devcontainerId":               shortHash(workspace),
	}
	container := "/workspaces/" + base
	stated := !blank(keys["workspaceFolder"])
	if wf := keys["workspaceFolder"]; stated {
		expanded, err := expandVars(wf, vars)
		if err != nil {
			return nil, fmt.Errorf("%s: workspaceFolder: %w", path, err)
		}
		if err := json.Unmarshal(expanded, &container); err != nil {
			return nil, fmt.Errorf("%s: workspaceFolder: %w", path, err)
		}
	}
	vars["containerWorkspaceFolder"] = container
	vars["containerWorkspaceFolderBasename"] = filepath.Base(container)

	// Only the keys the launcher reads are expanded and parsed. customizations
	// is the reason: it carries the editor's own settings, which have their own
	// ${...} vocabulary (${workspaceFolder}, ${env:HOME}, ${config:...}), and
	// failing on one of those would refuse a config over a value that nothing
	// here ever looks at. remoteEnv is held back separately because it is the
	// one place ${containerEnv:...} can be answered.
	mine := map[string]json.RawMessage{}
	for k, v := range keys {
		if dcImplemented[k] {
			mine[k] = v
		}
	}
	remote := mine["remoteEnv"]
	delete(mine, "remoteEnv")
	body, err := json.Marshal(mine)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	expanded, err := expandVars(body, vars)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if i := bytes.Index(expanded, []byte("${containerEnv:")); i >= 0 {
		name, _, _ := strings.Cut(string(expanded[i+2:]), "}")
		return nil, fmt.Errorf("%s: ${%s} works in remoteEnv only: everything else here has to be "+
			"final before the container it would be read from exists", path, name)
	}
	if err := json.Unmarshal(expanded, cfg); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if remote != nil {
		expanded, err := expandVars(remote, vars)
		if err != nil {
			return nil, fmt.Errorf("%s: remoteEnv: %w", path, err)
		}
		if err := json.Unmarshal(expanded, &cfg.RemoteEnv); err != nil {
			return nil, fmt.Errorf("%s: remoteEnv: %w", path, err)
		}
	}
	cfg.WorkspaceFolder = container

	// dockerComposeFile is a string or a list of them.
	if v, ok := keys["dockerComposeFile"]; ok && !blank(v) {
		var one string
		if err := json.Unmarshal(v, &one); err == nil {
			cfg.ComposeFiles = []string{one}
		} else if err := json.Unmarshal(v, &cfg.ComposeFiles); err != nil {
			return nil, fmt.Errorf("%s: dockerComposeFile: %w", path, err)
		}
		for i, f := range cfg.ComposeFiles {
			cfg.ComposeFiles[i] = cfg.abs(f)
		}
		if cfg.Service == "" {
			return nil, fmt.Errorf("%s: dockerComposeFile without service", path)
		}
		// The launcher writes no mount in this flavour, so the only thing that
		// knows where the project ends up inside the container is the config.
		// Guessing /workspaces/<name> would start the agent in an empty
		// directory that docker helpfully creates for it.
		if !stated {
			return nil, fmt.Errorf("%s: dockerComposeFile needs workspaceFolder as well, to say where "+
				"%s mounts the project inside the container", path, filepath.Base(cfg.ComposeFiles[0]))
		}
	}
	if cfg.ComposeFiles == nil && cfg.Image == "" && cfg.dockerfile() == "" {
		return nil, fmt.Errorf("%s: needs one of image, build.dockerfile or dockerComposeFile", path)
	}
	return cfg, nil
}

// abs resolves a path written in devcontainer.json, which is relative to the
// directory the file lives in.
func (d *devcontainerConfig) abs(p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Clean(filepath.Join(d.dir, p))
}

// dockerfile is the Dockerfile this config builds its image from, and "" when
// it names a prebuilt image instead. The spec spells it build.dockerfile; the
// pre-1.0 top-level dockerFile is still common in the wild, so both are read.
func (d *devcontainerConfig) dockerfile() string {
	return orElse(d.Build.Dockerfile, d.DockerFile)
}

// service returns the compose service name to exec into.
func (d *devcontainerConfig) service() string {
	if d.ComposeFiles != nil {
		return d.Service
	}
	return launcherService
}

// composeFile renders the generated compose project. It is emitted as JSON,
// which every YAML parser accepts, so encoding/json does the quoting instead of
// a hand-written YAML writer. The one post-step is doubling every $: compose
// interpolates ${...} and $VAR out of the host environment when it loads the
// file, and every value here is already final. $ cannot appear in JSON outside
// a string literal, so a blind replace is exact.
func (d *devcontainerConfig) composeFile() ([]byte, error) {
	svc := map[string]any{"working_dir": d.WorkspaceFolder}
	switch {
	case d.Build.Dockerfile != "" || d.DockerFile != "":
		dockerfile, context := d.Build.Dockerfile, d.Build.Context
		if dockerfile == "" {
			dockerfile, context = d.DockerFile, d.Context
		}
		build := map[string]any{"context": d.abs(orElse(context, ".")), "dockerfile": d.abs(dockerfile)}
		if d.Build.Target != "" {
			build["target"] = d.Build.Target
		}
		if len(d.Build.Args) > 0 {
			build["args"] = d.Build.Args
		}
		svc["build"] = build
	default:
		svc["image"] = d.Image
	}

	// The workspace mount is the whole point of the exercise: without it the
	// container sees none of the project. The spec's default binds the
	// workspace folder to containerWorkspaceFolder, read-write.
	mounts := []map[string]any{{
		"type": "bind", "source": d.workspace, "target": d.WorkspaceFolder,
	}}
	if d.WorkspaceMount != "" {
		m, err := parseMount(d.WorkspaceMount)
		if err != nil {
			return nil, fmt.Errorf("workspaceMount: %w", err)
		}
		// A workspaceFolder outside the mount is the one misconfiguration that
		// fails quietly: docker creates a missing working directory, so the
		// container starts and the agent finds an empty project.
		if target, _ := m["target"].(string); !within(d.WorkspaceFolder, target) {
			return nil, fmt.Errorf("workspaceMount puts the project at %s, but workspaceFolder says %s, "+
				"so the container would start in an empty directory", target, d.WorkspaceFolder)
		}
		mounts[0] = m
	}
	for _, raw := range d.Mounts {
		m, err := parseMountJSON(raw)
		if err != nil {
			return nil, err
		}
		mounts = append(mounts, m)
	}
	named := map[string]any{}
	for _, m := range mounts {
		src, _ := m["source"].(string)
		switch m["type"] {
		case "bind":
			// Relative is not a thing a bind source can be: compose resolves
			// what it reads against the generated file's own directory, which
			// is a cache directory nobody wrote that path against.
			switch {
			case src == "":
				return nil, fmt.Errorf("the bind mount to %v needs a source", m["target"])
			case !filepath.IsAbs(src):
				return nil, fmt.Errorf("mount source %q is relative: write it absolute, or as ${localWorkspaceFolder}/...", src)
			}
			if _, err := os.Stat(src); err != nil {
				return nil, fmt.Errorf("mount source does not exist on this machine: %s", src)
			}
		case "volume":
			if src != "" {
				// Pin the volume's real name: left to itself compose would
				// prefix it with the project name, and the config asked for
				// this one. A volume with no source is anonymous, which is
				// exactly what compose does with it.
				named[src] = map[string]any{"name": src}
			}
		}
	}
	svc["volumes"] = mounts

	if len(d.ContainerEnv) > 0 {
		svc["environment"] = d.ContainerEnv
	}
	if d.ContainerUser != "" {
		svc["user"] = d.ContainerUser
	}
	if d.Init != nil {
		svc["init"] = *d.Init
	}
	if d.Privileged != nil {
		svc["privileged"] = *d.Privileged
	}
	if len(d.CapAdd) > 0 {
		svc["cap_add"] = d.CapAdd
	}
	if len(d.SecurityOpt) > 0 {
		svc["security_opt"] = d.SecurityOpt
	}
	for _, p := range d.ForwardPorts {
		n, ok := p.(float64)
		if !ok {
			return nil, fmt.Errorf("forwardPorts: %v names another service, and a generated project has "+
				"only the one: use a compose file of your own for that", p)
		}
		port := strconv.Itoa(int(n))
		// Bound to the loopback address, which is what the editors do with a
		// forwarded port: it is for the person at this keyboard, not for the
		// network the laptop happens to be on.
		svc["ports"] = append(strs(svc["ports"]), "127.0.0.1:"+port+":"+port)
	}
	if err := applyRunArgs(svc, d.RunArgs); err != nil {
		return nil, err
	}
	// Host networking already puts every listening port on the host, and
	// compose refuses a service that asks for both.
	if svc["network_mode"] == "host" {
		delete(svc, "ports")
	}
	if d.OverrideCommand == nil || *d.OverrideCommand {
		svc["entrypoint"] = keepAlive
	}

	doc := map[string]any{"services": map[string]any{launcherService: svc}}
	if len(named) > 0 {
		doc["volumes"] = named
	}
	out, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return nil, err
	}
	return []byte(strings.ReplaceAll(string(out), "$", "$$")), nil
}

// runArgsElsewhere names the devcontainer.json key that covers a docker flag
// the launcher has no compose mapping for. Where it belongs is more use than
// the bare news that it is not supported.
var runArgsElsewhere = map[string]string{
	"-v": "mounts", "--volume": "mounts", "--mount": "mounts",
	"-p": "forwardPorts", "--publish": "forwardPorts",
	"-u": "containerUser", "--user": "containerUser",
	"-w": "workspaceFolder", "--workdir": "workspaceFolder",
}

// applyRunArgs folds the docker-run flags a devcontainer.json may carry into
// the compose service. Compose has no passthrough for raw docker flags, so
// anything without a mapping is refused by name rather than dropped.
func applyRunArgs(svc map[string]any, args []string) error {
	next := func(i *int) (string, error) {
		if *i+1 >= len(args) {
			return "", fmt.Errorf("runArgs: %s needs a value", args[*i])
		}
		*i++
		return args[*i], nil
	}
	for i := 0; i < len(args); i++ {
		flag, inline, split := strings.Cut(args[i], "=")
		val := inline
		var err error
		if !split {
			switch flag {
			case "--add-host", "--name", "--network", "--cap-add", "--security-opt", "--shm-size",
				"-e", "--env", "--hostname", "--userns", "--ipc", "--pid", "--group-add", "--device":
				if val, err = next(&i); err != nil {
					return err
				}
			}
		}
		switch flag {
		case "--add-host":
			svc["extra_hosts"] = append(strs(svc["extra_hosts"]), val)
		case "--name":
			svc["container_name"] = val
		case "--network":
			svc["network_mode"] = val
		case "--hostname":
			svc["hostname"] = val
		case "--cap-add":
			svc["cap_add"] = append(strs(svc["cap_add"]), val)
		case "--security-opt":
			svc["security_opt"] = append(strs(svc["security_opt"]), val)
		case "--shm-size":
			svc["shm_size"] = val
		case "--userns":
			svc["userns_mode"] = val // podman configs carry --userns=keep-id
		case "--ipc":
			svc["ipc"] = val
		case "--pid":
			svc["pid"] = val
		case "--group-add":
			svc["group_add"] = append(strs(svc["group_add"]), val)
		case "--device":
			svc["devices"] = append(strs(svc["devices"]), val)
		case "-e", "--env":
			k, v, ok := strings.Cut(val, "=")
			if !ok {
				v = os.Getenv(k) // docker's "pass this one through from my environment"
			}
			env, _ := svc["environment"].(map[string]string)
			if env == nil {
				env = map[string]string{}
				svc["environment"] = env
			}
			env[k] = v
		case "--init":
			svc["init"] = true
		case "--privileged":
			svc["privileged"] = true
		default:
			if key := runArgsElsewhere[flag]; key != "" {
				return fmt.Errorf("runArgs: %s has no compose equivalent, but %s in devcontainer.json "+
					"says the same thing, and every tool that reads the file understands it", flag, key)
			}
			return fmt.Errorf("runArgs: %s has no compose equivalent the launcher knows", flag)
		}
	}
	return nil
}

// parseMountJSON takes one entry of the mounts array, which the spec allows to
// be either the docker --mount string or an object with the same fields.
func parseMountJSON(raw json.RawMessage) (map[string]any, error) {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return parseMount(s)
	}
	var obj struct {
		Source, Target, Type string
		ReadOnly             bool `json:"readonly"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, fmt.Errorf("mounts: %s: %w", raw, err)
	}
	m := map[string]any{"type": orElse(obj.Type, "bind"), "target": obj.Target}
	if obj.Source != "" {
		m["source"] = obj.Source
	}
	if obj.ReadOnly {
		m["read_only"] = true
	}
	return m, nil
}

// parseMount takes the docker --mount form, "source=X,target=Y,type=bind,readonly".
func parseMount(spec string) (map[string]any, error) {
	m := map[string]any{"type": "bind"}
	for _, field := range strings.Split(spec, ",") {
		k, v, hasVal := strings.Cut(field, "=")
		switch k {
		case "source", "src":
			m["source"] = v
		case "target", "destination", "dst":
			m["target"] = v
		case "type":
			m["type"] = v
		case "readonly", "ro":
			m["read_only"] = !hasVal || v == "true"
		case "consistency", "bind-propagation":
			// Mac-only performance hints; nothing to do on a Linux daemon.
		default:
			return nil, fmt.Errorf("mount %q: unsupported field %q", spec, k)
		}
	}
	if m["target"] == nil {
		return nil, fmt.Errorf("mount %q: needs a target", spec)
	}
	return m, nil
}

func strs(v any) []string {
	s, _ := v.([]string)
	return s
}

// within reports whether path is dir or something under it.
func within(path, dir string) bool {
	rel, err := filepath.Rel(dir, path)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func orElse(v, fallback string) string {
	if v == "" {
		return fallback
	}
	return v
}

func shortHash(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])[:12]
}

// composeProject is the -p name: compose wants lowercase alphanumerics, and two
// projects that share a name share their containers, so the workspace path goes
// in as a hash.
func composeProject(workspace string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(filepath.Base(workspace)) {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
			b.WriteRune(r)
		} else {
			b.WriteRune('-')
		}
	}
	name := strings.Trim(b.String(), "-")
	if name == "" {
		name = "codehalter"
	}
	return name + "-" + shortHash(workspace)[:6]
}

// containerTool finds the container runtime and its compose plugin. Both podman
// and docker ship one and speak the same subcommands, so the only thing that
// varies is argv[0].
func containerTool() string {
	for _, rt := range []string{"docker", "podman"} {
		if _, err := exec.LookPath(rt); err != nil {
			continue
		}
		if err := exec.Command(rt, "compose", "version").Run(); err == nil {
			return rt
		}
		slog.Debug("launcher: runtime has no compose plugin", "runtime", rt)
	}
	return ""
}

// launchInDevcontainer runs the CLI inside the project's devcontainer and
// reports whether it handled the run at all. Not handled means the caller
// carries on in this process: no devcontainer.json to work from, or no
// container runtime to work with, and in both cases the agent's own bootstrap
// gives the better message.
func launchInDevcontainer(workspace string, inner []string, rebuild bool) (int, bool) {
	cfg, err := loadDevcontainerConfig(workspace)
	switch {
	case err == os.ErrNotExist:
		return 0, false
	case err != nil:
		fmt.Fprintf(os.Stderr, "%s\n\nStart it with the devcontainer CLI instead:\n"+
			"  npm i -g @devcontainers/cli\n"+
			"  devcontainer up   --workspace-folder %s\n"+
			"  devcontainer exec --workspace-folder %s codehalter --cli\n", err, workspace, workspace)
		return 1, true
	}
	rt := containerTool()
	if rt == "" {
		fmt.Fprintln(os.Stderr, "This project has a devcontainer, but neither docker nor podman "+
			"with the compose plugin is on PATH, so codehalter cannot start it.")
		return 0, false
	}

	project := composeProject(workspace)
	files := cfg.ComposeFiles
	if files == nil {
		body, err := cfg.composeFile()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", cfg.path, err)
			return 1, true
		}
		dir := filepath.Join(cacheDir(), "launcher", project)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "creating %s: %v\n", dir, err)
			return 1, true
		}
		path := filepath.Join(dir, "compose.yaml")
		if err := os.WriteFile(path, body, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "writing %s: %v\n", path, err)
			return 1, true
		}
		files = []string{path}
		// Compose rebuilds nothing on its own once an image exists, so the
		// Dockerfile is fingerprinted here instead. Without this an edited
		// Dockerfile, which is a thing codehalter actively suggests, would
		// silently keep the old image.
		rebuild = rebuild || stampChanged(filepath.Join(dir, "stamp"), body, cfg)
	}

	notice(rt, project, cfg, rebuild)
	compose := func(args ...string) *exec.Cmd {
		full := []string{"compose", "-p", project}
		for _, f := range files {
			full = append(full, "-f", f)
		}
		cmd := exec.Command(rt, append(full, args...)...)
		cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
		return cmd
	}

	up := []string{"up", "-d"}
	if rebuild {
		up = append(up, "--build")
	}
	// runServices narrows what starts, and unset means all of them, which is
	// what a plain `up` does. It only says anything for a config that brought
	// its own compose files: the generated project has the one service.
	if cfg.ComposeFiles != nil && len(cfg.RunServices) > 0 {
		up = append(up, cfg.RunServices...)
		if !slices.Contains(cfg.RunServices, cfg.Service) {
			up = append(up, cfg.Service)
		}
	}
	if err := compose(up...).Run(); err != nil {
		fmt.Fprintf(os.Stderr, "\n%s compose up failed: %v\n", rt, err)
		return 1, true
	}

	if err := resolveContainerEnv(cfg.RemoteEnv, func() ([]byte, error) {
		cmd := compose("exec", "-T", cfg.service(), "env")
		cmd.Stdin, cmd.Stdout = nil, nil
		return cmd.Output()
	}); err != nil {
		fmt.Fprintf(os.Stderr, "reading the container's environment, which remoteEnv is written in terms of: %v\n", err)
		return 1, true
	}

	args := []string{"exec", "-w", cfg.WorkspaceFolder}
	if !stdinIsTTY() {
		args = append(args, "-T")
	}
	if cfg.RemoteUser != "" {
		args = append(args, "-u", cfg.RemoteUser)
	}
	// Sorted so the same config produces the same exec line every run.
	for _, k := range slices.Sorted(maps.Keys(cfg.RemoteEnv)) {
		args = append(args, "-e", k+"="+cfg.RemoteEnv[k])
	}
	// What the host already worked out about updates: the resolved release tag
	// and the answer the user gave to the question about it. Without these the
	// copy inside the container spends its own API call and asks again.
	for _, k := range []string{envLatest, envUpdate} {
		if v := os.Getenv(k); v != "" {
			args = append(args, "-e", k+"="+v)
		}
	}
	args = append(args, cfg.service(), "codehalter", "--cli", "--cwd", cfg.WorkspaceFolder)

	// Ctrl+C belongs to the CLI in the container, which cancels the turn with
	// it. The signal reaches it over the exec's tty; ignoring it here keeps the
	// launcher from dying underneath and leaving the child holding the
	// terminal.
	signal.Ignore(os.Interrupt)
	code := 0
	if err := compose(append(args, inner...)...).Run(); err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			code = ee.ExitCode()
		} else {
			fmt.Fprintf(os.Stderr, "%s compose exec failed: %v\n", rt, err)
			code = 1
		}
		if code == 126 || code == 127 {
			fmt.Fprintf(os.Stderr, "\ncodehalter is not installed in this image. Add it to %s "+
				"(see the Dockerfiles codehalter scaffolds) and re-run with --rebuild.\n",
				filepath.Join(cfg.dir, "Dockerfile"))
		}
	}
	signal.Reset(os.Interrupt)

	// shutdownAction is the config's own answer to "what happens when the
	// client goes away". Anything but the two stop values leaves it running,
	// which is also the default here because the next start is then instant.
	if cfg.ShutdownAction == "stopContainer" || cfg.ShutdownAction == "stopCompose" {
		if err := compose("stop").Run(); err != nil {
			fmt.Fprintf(os.Stderr, "%s compose stop failed: %v\n", rt, err)
		}
	}
	return code, true
}

// resolveContainerEnv fills in ${containerEnv:VAR} in remoteEnv, the one
// substitution that cannot be done while reading the file: the answer lives in
// a container that does not exist until `up` has run. read produces the output
// of `env` in there, and is only called when something asks for it.
func resolveContainerEnv(remoteEnv map[string]string, read func() ([]byte, error)) error {
	const marker = "${containerEnv:"
	need := false
	for _, v := range remoteEnv {
		need = need || strings.Contains(v, marker)
	}
	if !need {
		return nil
	}
	out, err := read()
	if err != nil {
		return err
	}
	env := map[string]string{}
	for _, line := range strings.Split(string(out), "\n") {
		if k, v, ok := strings.Cut(strings.TrimRight(line, "\r"), "="); ok {
			env[k] = v
		}
	}
	for k, v := range remoteEnv {
		var b strings.Builder
		for {
			i := strings.Index(v, marker)
			if i < 0 {
				break
			}
			j := strings.IndexByte(v[i:], '}')
			if j < 0 {
				break
			}
			name, def, _ := strings.Cut(v[i+len(marker):i+j], ":")
			b.WriteString(v[:i])
			b.WriteString(orElse(env[name], def))
			v = v[i+j+1:]
		}
		remoteEnv[k] = b.String() + v
	}
	return nil
}

// notice says what is about to happen before anything slow starts: which
// runtime, that compose is driving it, and above all which host directory is
// about to be mounted where. The mount is the part worth being sure about,
// because everything the agent edits lands there.
func notice(rt, project string, cfg *devcontainerConfig, building bool) {
	bold, dim, reset := ansiBold, ansiDim, ansiReset
	fi, err := os.Stdout.Stat()
	if err != nil || fi.Mode()&os.ModeCharDevice == 0 || os.Getenv("TERM") == "dumb" || os.Getenv("NO_COLOR") != "" {
		bold, dim, reset = "", "", ""
	}
	from := cfg.Image
	if from == "" {
		from = filepath.Base(cfg.abs(cfg.dockerfile()))
	}
	if cfg.ComposeFiles != nil {
		from = filepath.Base(cfg.ComposeFiles[0]) + ", service " + cfg.Service
	}
	fmt.Printf("%scodehalter cli%s\n", bold, reset)
	fmt.Printf("%s  no container here, so %s compose starts one: project %s, from %s%s\n", dim, rt, project, from, reset)
	if cfg.ComposeFiles != nil {
		fmt.Printf("%s  your compose file does the mounting; the CLI starts in %s%s\n", dim, cfg.WorkspaceFolder, reset)
	} else {
		fmt.Printf("%s  %s → %s  (bind mount, read-write: edits inside are edits here)%s\n",
			dim, cfg.workspace, cfg.WorkspaceFolder, reset)
	}
	// Only for a config that actually has something to build. The first run of
	// any project has no stamp to compare against, so it passes --build either
	// way, and for an "image": ... config that is a no-op compose skips: saying
	// "building the image" there promises a wait that never comes.
	if building && cfg.dockerfile() != "" {
		fmt.Printf("%s  building the image first, which takes a while; later runs reuse it%s\n", dim, reset)
	}
	for _, n := range cfg.notes {
		fmt.Printf("%s  note: %s%s\n", dim, n, reset)
	}
	if runtime.GOOS == "linux" && os.Getuid() != 1000 {
		fmt.Printf("%s  note: your uid is %d, and container users are usually 1000, so files "+
			"written in there may come back owned by someone else%s\n", dim, os.Getuid(), reset)
	}
}

// stampChanged reports whether the build inputs differ from the last run, and
// records the current ones either way.
func stampChanged(path string, compose []byte, cfg *devcontainerConfig) bool {
	h := sha256.New()
	h.Write(compose)
	if df := cfg.dockerfile(); df != "" {
		body, err := os.ReadFile(cfg.abs(df))
		if err != nil {
			slog.Debug("launcher: cannot fingerprint dockerfile", "err", err)
			return true
		}
		h.Write(body)
	}
	want := hex.EncodeToString(h.Sum(nil))
	got, err := os.ReadFile(path)
	if err := os.WriteFile(path, []byte(want), 0o644); err != nil {
		slog.Debug("launcher: cannot write stamp", "err", err)
	}
	return err != nil || string(got) != want
}

func cacheDir() string {
	if d, err := os.UserCacheDir(); err == nil {
		return filepath.Join(d, "codehalter")
	}
	return filepath.Join(os.TempDir(), "codehalter")
}
