package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// containerLabelKey is the docker label we set on every bench-spawned
// devcontainer so we can find and remove the previous one before a re-run
// without touching any unrelated containers on the host. The matching value
// is unique per test (containerLabel(tc.Name)).
const containerLabelKey = "codehalter-bench"

// containerLabel returns the full label=value string passed to both
// `devcontainer up --id-label` and `docker ps --filter label=`. Keeping the
// test name in the value makes `docker ps` output trivially scannable for a
// human debugging a stuck container.
func containerLabel(testName string) string {
	return containerLabelKey + "=codehalter-bench-" + testName
}

// testCase is one row from a bench/tests/*.toml file. Verify is a shell line
// run inside the same devcontainer codehalter executed in — its exit code
// (0 = pass) is the binary success signal; stdout/stderr are captured for
// debugging in the result row.
//
// Settings is an optional path to the settings.toml this test should copy
// into the project before spawning codehalter. Resolved relative to the
// test TOML file's directory so tests are movable. Empty means: use the
// fallback path passed on the command line (-settings flag).
type testCase struct {
	Name         string `toml:"name"`
	Repo         string `toml:"repo"`
	Commit       string `toml:"commit"`
	Prompt       string `toml:"prompt"`
	Verify       string `toml:"verify"`
	Timeout      string `toml:"timeout"`
	Settings     string `toml:"settings"`
	Devcontainer string `toml:"devcontainer"`

	// sourcePath is the absolute path to the test TOML file this case was
	// parsed from. Not a TOML field — populated by collectTests so we can
	// resolve Settings / Devcontainer relative to where the test lives.
	sourcePath string `toml:"-"`
}

// resolveTestPath expands $VAR / ${VAR} env refs and a leading `~/`, then
// returns the path as-is if absolute or resolves it against the test TOML's
// directory so paths in the .toml are unambiguous. Empty input returns the
// fallback verbatim — the caller decides whether the fallback itself is OK.
func (tc testCase) resolveTestPath(input, fallback string) string {
	if input == "" {
		return fallback
	}
	p := os.ExpandEnv(input)
	if strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			p = filepath.Join(home, p[2:])
		}
	}
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(filepath.Dir(tc.sourcePath), p)
}

// settingsPath returns the absolute path to the settings.toml to copy in.
// Per-test `settings` field overrides the -settings flag fallback.
func (tc testCase) settingsPath(fallback string) string {
	return tc.resolveTestPath(tc.Settings, fallback)
}

// devcontainerPath returns the absolute path to the directory whose contents
// get overlaid into the test repo's `.devcontainer/`. Per-test `devcontainer`
// field overrides the -devcontainer flag fallback. The overlay is only
// applied when the clone doesn't already ship a devcontainer.json.
func (tc testCase) devcontainerPath(fallback string) string {
	return tc.resolveTestPath(tc.Devcontainer, fallback)
}

// testResult is one row appended to bench/results.jsonl.
type testResult struct {
	Name       string    `json:"name"`
	StartedAt  time.Time `json:"started_at"`
	AgentMs    int64     `json:"agent_ms"`
	VerifyMs   int64     `json:"verify_ms"`
	StopReason string    `json:"stop_reason"`
	VerifyExit int       `json:"verify_exit"`
	OK         bool      `json:"ok"`
	Note       string    `json:"note,omitempty"`
}

// runTest drives a single test end-to-end. Returns a result row plus the path
// to the per-test workdir (which is kept on disk so the user can re-inspect).
// fallbackSettings / fallbackDevcontainer are used only when the test TOML
// doesn't carry its own `settings` / `devcontainer` path.
func runTest(ctx context.Context, tc testCase, fallbackSettings, fallbackDevcontainer, codehalterBin, workRoot string) (testResult, string, error) {
	result := testResult{Name: tc.Name, StartedAt: time.Now()}
	workDir := filepath.Join(workRoot, tc.Name)
	settingsPath := tc.settingsPath(fallbackSettings)
	if _, err := os.Stat(settingsPath); err != nil {
		result.Note = "settings: " + err.Error()
		return result, workDir, fmt.Errorf("settings file %s missing", settingsPath)
	}
	dcPath := tc.devcontainerPath(fallbackDevcontainer)

	timeout := 30 * time.Minute
	if tc.Timeout != "" {
		if d, err := time.ParseDuration(tc.Timeout); err == nil {
			timeout = d
		}
	}
	tctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	if err := prepareWorkdir(workDir, tc.Repo, tc.Commit, settingsPath, dcPath); err != nil {
		result.Note = "prepare: " + err.Error()
		return result, workDir, err
	}

	// Wipe any container left over from a previous run of this test before
	// bringing a new one up — each bench run starts from a clean container
	// so cached state (mutated files, daemons, env) can't bleed across runs.
	// The filter is scoped to our label, so unrelated containers on the host
	// are never touched.
	if err := removeOldContainer(tctx, tc.Name); err != nil {
		// Best-effort: a stale docker daemon or missing CLI shouldn't block
		// the run. devcontainer up will error out shortly if docker really
		// isn't usable, with a clearer message.
		fmt.Fprintf(os.Stderr, "warn: remove old container: %v\n", err)
	}

	// `devcontainer up` brings up the project's container, tagged with our
	// per-test label so we can find it again on the next run.
	if err := devUp(tctx, workDir, tc.Name); err != nil {
		result.Note = "devcontainer up: " + err.Error()
		return result, workDir, err
	}

	// Replace the noisy auto-generated docker name (vsc-<repo>-<hash>) with a
	// stable, scannable one — `docker ps` then shows `bench-<test>` instead.
	// Best-effort: a failure here doesn't affect anything the bench actually
	// does, since every other operation already routes via id-label.
	if err := renameContainer(tctx, tc.Name); err != nil {
		fmt.Fprintf(os.Stderr, "warn: rename container: %v\n", err)
	}

	// Codehalter must exist inside the container. Easiest path: copy the
	// pre-built binary into a tmp location the container can see (the workdir
	// is bind-mounted) and exec it from there. Avoids touching the image.
	inContainerBin, err := stageBinary(tctx, workDir, codehalterBin)
	if err != nil {
		result.Note = "stage binary: " + err.Error()
		return result, workDir, err
	}

	agentStart := time.Now()
	stopReason, agentErr := runAgent(tctx, workDir, tc.Name, inContainerBin, tc.Prompt)
	result.AgentMs = time.Since(agentStart).Milliseconds()
	result.StopReason = string(stopReason)
	if agentErr != nil {
		result.Note = "agent: " + agentErr.Error()
		return result, workDir, agentErr
	}

	verifyStart := time.Now()
	exit, verr := devExec(tctx, workDir, tc.Name, "bash", "-lc", tc.Verify)
	result.VerifyMs = time.Since(verifyStart).Milliseconds()
	result.VerifyExit = exit
	result.OK = exit == 0 && stopReason == StopReasonEndTurn
	if verr != nil && exit == 0 {
		result.Note = "verify wrapper: " + verr.Error()
	}
	return result, workDir, nil
}

// prepareWorkdir clones the repo (fresh — wipes any prior workdir), checks
// out the target commit, drops the bench's settings.toml under .codehalter/,
// and overlays a devcontainer config under .devcontainer/ when the cloned
// repo doesn't already ship one. dcPath is the source directory to copy from.
// The clone lives on the host filesystem; the devcontainer mounts it.
func prepareWorkdir(workDir, repo, commit, settingsPath, dcPath string) error {
	if err := os.RemoveAll(workDir); err != nil {
		return fmt.Errorf("rm old workdir: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(workDir), 0o755); err != nil {
		return err
	}
	if err := runHost("git", "clone", repo, workDir); err != nil {
		return fmt.Errorf("git clone: %w", err)
	}
	if err := runHostIn(workDir, "git", "checkout", "--detach", commit); err != nil {
		return fmt.Errorf("git checkout %s: %w", commit, err)
	}
	dotdir := filepath.Join(workDir, ".codehalter")
	if err := os.MkdirAll(dotdir, 0o755); err != nil {
		return err
	}
	if err := copyFile(settingsPath, filepath.Join(dotdir, "settings.toml")); err != nil {
		return fmt.Errorf("copy settings: %w", err)
	}
	if err := ensureDevcontainer(workDir, dcPath); err != nil {
		return fmt.Errorf("devcontainer overlay: %w", err)
	}
	return nil
}

// ensureDevcontainer guarantees the cloned repo has a usable .devcontainer/
// before we hand it to the devcontainer CLI. If the clone already ships a
// devcontainer.json we trust it — projects with custom container setups
// (Cargo, JDK, …) almost certainly know better than the bench default. When
// the clone has nothing, we copy every file from dcPath into <workDir>/.devcontainer/.
// A missing dcPath when no devcontainer.json exists is a hard error — the
// devcontainer CLI would fail downstream with the same effect, but the
// up-front check yields a cleaner result.Note for the row.
func ensureDevcontainer(workDir, dcPath string) error {
	existing := filepath.Join(workDir, ".devcontainer", "devcontainer.json")
	if _, err := os.Stat(existing); err == nil {
		return nil
	}
	if dcPath == "" {
		return fmt.Errorf("clone has no .devcontainer/devcontainer.json and no default configured")
	}
	if _, err := os.Stat(dcPath); err != nil {
		return fmt.Errorf("devcontainer source %s missing", dcPath)
	}
	dst := filepath.Join(workDir, ".devcontainer")
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return err
	}
	entries, err := os.ReadDir(dcPath)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if e.IsDir() {
			// Devcontainer configs are flat in practice (devcontainer.json
			// + Dockerfile + maybe a script). A subdirectory in the source
			// is unusual — skip rather than walk, to keep the overlay
			// behavior boring and predictable.
			continue
		}
		if err := copyFile(filepath.Join(dcPath, e.Name()), filepath.Join(dst, e.Name())); err != nil {
			return err
		}
	}
	return nil
}

// stageBinary places the codehalter binary inside the workdir (which the
// devcontainer bind-mounts as /workspaces/<name>) so it shows up at a known
// path inside the container. Returns the in-container path. Using the bind
// mount avoids docker cp / image rebuilds — works for any devcontainer that
// mounts the workspace folder (which is the default).
func stageBinary(ctx context.Context, workDir, codehalterBin string) (string, error) {
	dst := filepath.Join(workDir, ".codehalter", "codehalter")
	if err := copyFile(codehalterBin, dst); err != nil {
		return "", err
	}
	if err := os.Chmod(dst, 0o755); err != nil {
		return "", err
	}
	// The devcontainer CLI mounts workDir under /workspaces/<basename>, so
	// the staged binary appears at /workspaces/<basename>/.codehalter/codehalter
	// inside the container. We don't rely on knowing the exact mount path —
	// we run devcontainer exec from the project root, which sets it as CWD,
	// so a relative path works regardless of mount naming.
	return "./.codehalter/codehalter", nil
}

// devUp brings up a freshly-tagged devcontainer for workDir. The id-label
// scopes both this `up` and every subsequent `exec` to a container we own
// — so a host with other (preveltekit, codehalter, …) dev containers running
// doesn't get its work disturbed.
func devUp(ctx context.Context, workDir, testName string) error {
	cmd := exec.CommandContext(ctx, "devcontainer", "up",
		"--workspace-folder", workDir,
		"--id-label", containerLabel(testName),
	)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// devExec runs a single shell command inside this test's devcontainer and
// returns its exit code. The id-label routes the exec to the same container
// devUp tagged — no chance of landing in some other dev container that
// happens to share the workspace folder name.
func devExec(ctx context.Context, workDir, testName string, argv ...string) (int, error) {
	args := append([]string{
		"exec",
		"--workspace-folder", workDir,
		"--id-label", containerLabel(testName),
	}, argv...)
	cmd := exec.CommandContext(ctx, "devcontainer", args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err == nil {
		return 0, nil
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return ee.ExitCode(), nil
	}
	return -1, err
}

// renameContainer gives the just-started devcontainer a stable name like
// `bench-<testname>`. The devcontainer CLI auto-names containers
// `vsc-<repo>-<hash>` which is impossible to scan visually in `docker ps`.
// Idempotent across re-runs: removeOldContainer wipes the prior container by
// label first, so the target name is always free when we get here.
func renameContainer(ctx context.Context, testName string) error {
	psCmd := exec.CommandContext(ctx, "docker", "ps", "-q",
		"--filter", "label="+containerLabel(testName))
	out, err := psCmd.Output()
	if err != nil {
		return fmt.Errorf("docker ps: %w", err)
	}
	ids := strings.Fields(strings.TrimSpace(string(out)))
	if len(ids) == 0 {
		return fmt.Errorf("no container found for label %s", containerLabel(testName))
	}
	// Multiple matches shouldn't happen — removeOldContainer ran moments ago.
	// Rename only the first; the label filter already isolates them to us.
	rn := exec.CommandContext(ctx, "docker", "rename", ids[0], "bench-"+testName)
	rn.Stdout = os.Stderr
	rn.Stderr = os.Stderr
	return rn.Run()
}

// removeOldContainer force-removes any container previously tagged for this
// test. Idempotent: zero matches is fine. We shell out to `docker` rather
// than `devcontainer` because the devcontainer CLI has no first-class
// "remove by label" subcommand across all versions.
func removeOldContainer(ctx context.Context, testName string) error {
	psCmd := exec.CommandContext(ctx, "docker", "ps", "-aq",
		"--filter", "label="+containerLabel(testName))
	psOut, err := psCmd.Output()
	if err != nil {
		return fmt.Errorf("docker ps: %w", err)
	}
	ids := strings.Fields(strings.TrimSpace(string(psOut)))
	if len(ids) == 0 {
		return nil
	}
	rmCmd := exec.CommandContext(ctx, "docker", append([]string{"rm", "-f"}, ids...)...)
	rmCmd.Stdout = os.Stderr
	rmCmd.Stderr = os.Stderr
	return rmCmd.Run()
}

// runAgent spawns codehalter inside the devcontainer with stdio piped back,
// runs one full turn through the ACP wire, and returns the stop reason.
// Notifications (session/update) are tail-logged into the workdir so a
// long-running turn can be inspected mid-flight.
func runAgent(ctx context.Context, workDir, testName, inContainerBin, prompt string) (StopReason, error) {
	logPath := filepath.Join(workDir, ".codehalter", "bench_agent.log")
	logF, err := os.Create(logPath)
	if err != nil {
		return "", fmt.Errorf("open agent log: %w", err)
	}
	defer logF.Close()

	cmd := exec.CommandContext(ctx,
		"devcontainer", "exec",
		"--workspace-folder", workDir,
		"--id-label", containerLabel(testName),
		inContainerBin,
	)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return "", err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", err
	}
	// Tee codehalter's stderr (slog output) to both the per-test log file and
	// the bench's own stderr so a `just bench` run shows progress live without
	// the user having to tail the log file.
	cmd.Stderr = io.MultiWriter(logF, os.Stderr)
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("start codehalter: %w", err)
	}
	defer func() {
		// Closing stdin makes codehalter's stdin-reader exit cleanly; if it
		// hangs (shouldn't, but defensive), the context cancel will reap it.
		_ = stdin.Close()
		_ = cmd.Wait()
	}()

	client := newACPClient(stdin, stdout, func(method string, params json.RawMessage) {
		line := fmt.Sprintf("[%s] %s %s\n", time.Now().Format(time.RFC3339), method, string(params))
		fmt.Fprint(logF, line)
		fmt.Fprint(os.Stderr, "[acp] "+line)
	})

	var initResp initializeResponse
	if err := client.call(ctx, "initialize", initializeRequest{ProtocolVersion: ProtocolVersion}, &initResp); err != nil {
		return "", fmt.Errorf("initialize: %w", err)
	}

	// session/new with cwd. Inside the container the CWD is set by
	// `devcontainer exec` to the workspace folder, so a relative "." works.
	var nsResp newSessionResponse
	if err := client.call(ctx, "session/new", newSessionRequest{Cwd: "."}, &nsResp); err != nil {
		return "", fmt.Errorf("session/new: %w", err)
	}

	// Flip into autopilot before the prompt so codehalter never sends a
	// session/request_permission back — bench is unattended and any such
	// request would hang the run until the timeout fires.
	var smResp setSessionModeResponse
	if err := client.call(ctx, "session/set_mode", setSessionModeRequest{
		SessionId: nsResp.SessionId,
		ModeId:    "autopilot",
	}, &smResp); err != nil {
		return "", fmt.Errorf("session/set_mode autopilot: %w", err)
	}

	// session/prompt blocks for the entire turn — could be minutes. The
	// notification handler keeps writing to bench_agent.log so the user can
	// `tail -f` it without the bench appearing hung.
	var pr promptResponse
	err = client.call(ctx, "session/prompt", promptRequest{
		SessionId: nsResp.SessionId,
		Content:   []contentBlockText{{Type: "text", Text: prompt}},
	}, &pr)
	if err != nil {
		return "", fmt.Errorf("session/prompt: %w", err)
	}
	return pr.StopReason, nil
}

// runHost executes a command on the host with stderr forwarded.
func runHost(name string, args ...string) error {
	return runHostIn("", name, args...)
}

func runHostIn(dir, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	if dir != "" {
		cmd.Dir = dir
	}
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}
