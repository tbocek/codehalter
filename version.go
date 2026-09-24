package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// ---------------------------------------------------------------------------
// Update check and self-update.
//
// Releases are integer tags (v1, v2, ...), stamped into `version` by the build,
// so "is there a newer one" is an integer comparison. Any other build reports
// "dev", compares as older than nothing, and is never offered an update.
//
// Env vars carry a decision already made, so it is made once per run and not
// once per process (the launcher starts a second codehalter in the container, a
// self-update re-executes a third). They are the manual controls too:
//
//	CODEHALTER_LATEST=v42   the newest release, already resolved: no API call.
//	CODEHALTER_UPDATE=yes   the user consented: update, don't ask.
//	CODEHALTER_UPDATE=skip  don't check, don't ask, don't update.
//
// Nothing here may stop codehalter from starting: no network, a rate limit or
// an unwritable install directory is one reported line, then the run goes on.
// ---------------------------------------------------------------------------

// version is the release tag this binary was built from, stamped by the build
// workflow with -ldflags "-X main.version=$tag". Every other build reports the
// default and never checks for updates.
var version = "dev"

const (
	updateRepo    = "tbocek/codehalter"
	updateTTL     = 24 * time.Hour
	updateTimeout = 4 * time.Second
	// A release binary is tens of megabytes, and anything far below that is an
	// error page or a truncated transfer rather than a program. Renaming one of
	// those over the running binary would leave nothing to run, so the download
	// is rejected before it can replace anything.
	updateMinBytes = 4 << 20
)

// The two upstream endpoints and the way this process finds its own binary.
// Vars rather than consts so the tests can point them at a local server and at
// a throwaway file: replacing a running binary is the one thing here that
// cannot be tried out on the real thing.
var (
	updateLatestURL = "https://api.github.com/repos/" + updateRepo + "/releases/latest"
	updateAssetURL  = "https://github.com/" + updateRepo + "/releases/download/%s/codehalter-%s-%s"
	executable      = os.Executable
)

// envLatest and envUpdate carry an already-made decision to the next codehalter
// in the chain (see the header). The launcher forwards exactly these two into
// the container, and a self-update's re-exec inherits them through os.Environ.
const (
	envLatest = "CODEHALTER_LATEST"
	envUpdate = "CODEHALTER_UPDATE"
)

// remember records a fact for the rest of the run, where "the run" is longer
// than this process: the container it may start and the binary it may re-exec
// into both read these back out of the environment (see the header). A failure
// costs a repeated question or a repeated API call downstream, never
// correctness, so it is logged and the run continues.
func remember(key, value string) {
	if err := os.Setenv(key, value); err != nil {
		slog.Debug("update: could not record a decision", "key", key, "err", err)
	}
}

// releaseNum turns a release tag into the integer release.sh incremented.
// Anything that is not "v" + digits, including "dev", yields 0, which is older
// than every real release and newer than none: an unstamped build never offers
// an update, and an unparsable tag from the API is ignored rather than acted on.
func releaseNum(tag string) int {
	n, err := strconv.Atoi(strings.TrimPrefix(tag, "v"))
	if err != nil || n <= 0 {
		return 0
	}
	return n
}

// updateCachePath is where the last answer from the GitHub API is kept, with
// the ETag it came with. The cache is not there to skip the check: a check that
// said "nothing newer" is exactly the one worth repeating after a release, and
// trusting it for a day is how a container missed v67 for sixteen hours. It
// holds the ETag so the check can be a conditional request, which GitHub does
// not count against the 60 requests per hour it allows unauthenticated, and it
// is the answer when GitHub cannot be reached at all.
func updateCachePath() string { return filepath.Join(cacheDir(), "update.json") }

type updateCache struct {
	Tag     string    `json:"tag"`
	ETag    string    `json:"etag,omitempty"`
	Checked time.Time `json:"checked"`
}

// latestRelease reports the newest release tag. It answers from $CODEHALTER_LATEST
// when the process that started us already resolved it; otherwise it asks
// GitHub every time, conditionally: with the cached ETag a 304 costs nothing,
// and only a new release costs a counted request. When GitHub is unreachable or
// rate-limits us, a cached answer younger than updateTTL stands in.
func latestRelease(ctx context.Context) (string, error) {
	if tag := os.Getenv(envLatest); tag != "" {
		return tag, nil
	}
	var c updateCache
	if buf, err := os.ReadFile(updateCachePath()); err == nil {
		if err := json.Unmarshal(buf, &c); err != nil {
			c = updateCache{}
		}
	}
	fallback := func(err error) (string, error) {
		if c.Tag != "" && time.Since(c.Checked) < updateTTL {
			slog.Debug("update: GitHub unavailable, answering from the cache", "tag", c.Tag, "age", time.Since(c.Checked), "err", err)
			return c.Tag, nil
		}
		return "", err
	}

	ctx, cancel := context.WithTimeout(ctx, updateTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, updateLatestURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	if c.ETag != "" {
		req.Header.Set("If-None-Match", c.ETag)
	}
	resp, err := metaHTTPClient.Do(req)
	if err != nil {
		return fallback(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotModified && c.Tag != "" {
		c.Checked = time.Now()
		writeUpdateCache(c)
		return c.Tag, nil
	}
	if resp.StatusCode != http.StatusOK {
		// 403/429 is the rate limit, and the body says so; it is the one failure
		// here a user can act on (wait, or use a different network), so it is
		// worth keeping in the message rather than reporting a bare status.
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 400))
		return fallback(fmt.Errorf("GitHub API: %s: %s", resp.Status, strings.TrimSpace(string(body))))
	}
	var rel struct {
		TagName string `json:"tag_name"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&rel); err != nil {
		return "", err
	}
	if rel.TagName == "" {
		return "", errors.New("GitHub API returned a release with no tag_name")
	}
	writeUpdateCache(updateCache{Tag: rel.TagName, ETag: resp.Header.Get("ETag"), Checked: time.Now()})
	return rel.TagName, nil
}

// writeUpdateCache is best effort: a read-only cache dir costs a counted API
// call per start, which is a reason to log, not to fail a check that succeeded.
func writeUpdateCache(c updateCache) {
	buf, err := json.Marshal(c)
	if err != nil {
		return
	}
	if err := os.MkdirAll(filepath.Dir(updateCachePath()), 0o755); err != nil {
		slog.Debug("update: cache dir not writable", "err", err)
	} else if err := os.WriteFile(updateCachePath(), buf, 0o644); err != nil {
		slog.Debug("update: cache not written", "err", err)
	}
}

// newerRelease returns the latest release tag when it is newer than this
// binary, and "" when it is not, when the check is switched off, or when it
// failed. The tag it resolved (newer or not) is published in $CODEHALTER_LATEST
// so the container this process may start next, and the process a self-update
// may re-exec, both skip the API call.
//
// cwd selects the settings.toml that can hold update_check = false; it is the
// project directory, which is where the rest of codehalter's config lives.
func newerRelease(ctx context.Context, cwd string) string {
	if os.Getenv(envUpdate) == "skip" {
		return ""
	}
	if version == "dev" {
		return "" // own build: nothing to offer, and nothing to compare against
	}
	if s, err := loadSettings(cwd); err == nil && s.UpdateCheck != nil && !*s.UpdateCheck {
		return ""
	}
	tag, err := latestRelease(ctx)
	if err != nil {
		slog.Debug("update: check failed (ignored)", "err", err)
		return ""
	}
	remember(envLatest, tag)
	if releaseNum(tag) <= releaseNum(version) {
		return ""
	}
	return tag
}

// selfUpdate installs this platform's release asset over the running binary
// and reports the path it replaced. The download lands in the install directory
// (a rename is atomic only within one filesystem) and is renamed into place
// only after it has RUN once and reported the tag asked for. So a truncated,
// wrong-architecture or error-page "binary" never replaces a working one, and
// an update that does not change the version cannot loop.
func selfUpdate(ctx context.Context, tag string) (string, error) {
	self, err := installTarget(ctx)
	if err != nil {
		return "", err
	}

	// The download has to land in the install directory: a rename is atomic
	// only within one filesystem, and anything less than atomic here can leave
	// a half-written binary where a working one used to be.
	tmp, err := os.CreateTemp(filepath.Dir(self), ".codehalter-update-*")
	if err != nil {
		return "", fmt.Errorf("%s is not writable (try sudo codehalter --update): %w", filepath.Dir(self), err)
	}
	defer os.Remove(tmp.Name()) // no-op once the rename below succeeded

	url := fmt.Sprintf(updateAssetURL, tag, runtime.GOOS, runtime.GOARCH)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		tmp.Close()
		return "", err
	}
	resp, err := metaHTTPClient.Do(req)
	if err != nil {
		tmp.Close()
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		tmp.Close()
		return "", fmt.Errorf("downloading %s: %s", url, resp.Status)
	}
	n, err := io.Copy(tmp, resp.Body)
	if err == nil {
		err = tmp.Chmod(0o755)
	}
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return "", err
	}
	if n < updateMinBytes {
		return "", fmt.Errorf("%s returned %d bytes, too small to be the binary", url, n)
	}

	got, err := reportedVersion(ctx, tmp.Name())
	if err != nil {
		return "", fmt.Errorf("the downloaded binary does not run here: %w", err)
	}
	if got != versionLine(tag) {
		return "", fmt.Errorf("the downloaded binary reports %q, not %q", got, versionLine(tag))
	}
	if err := os.Rename(tmp.Name(), self); err != nil {
		return "", err
	}
	return self, nil
}

// installTarget is the file a self-update replaces: this program's own binary,
// found by asking each candidate path to identify itself, never by trusting a
// path alone.
//
// Trusting /proc/self/exe alone bricked a container once. On Alpine, gcompat
// runs a glibc-linked binary by re-executing it through /lib/ld-musl-x86_64.so.1,
// so the "executable" of the running process was the musl loader; the update
// renamed the download over it, and from then on no program in that container
// could start. The path the user invoked (argv[0] on PATH) is tried first
// because it survives such re-execs, the kernel's answer second, and whichever
// is chosen must print this program's exact version line before it can be
// replaced: a loader, a wrapper script or some other codehalter all fail that.
func installTarget(ctx context.Context) (string, error) {
	var candidates []string
	if p, err := exec.LookPath(os.Args[0]); err == nil {
		candidates = append(candidates, p)
	}
	if p, err := executable(); err == nil {
		candidates = append(candidates, p)
	}
	var tried []string
	for _, c := range candidates {
		// Follow the symlink: replacing the link itself would leave the real
		// binary in place and the next start would run the old one again.
		if resolved, err := filepath.EvalSymlinks(c); err == nil {
			c = resolved
		}
		if got, err := reportedVersion(ctx, c); err == nil && got == versionLine(version) {
			return c, nil
		}
		tried = append(tried, c)
	}
	return "", fmt.Errorf("refusing to update: nothing at %s reports %q, so none of them is the running binary", strings.Join(tried, " or "), versionLine(version))
}

// reportedVersion runs path with --version and returns its first line trimmed,
// the only proof accepted that a file is a codehalter binary. Later lines (the
// build stamp) are for people and never compared.
func reportedVersion(ctx context.Context, path string) (string, error) {
	probe, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(probe, path, "--version").Output()
	if err != nil {
		return "", err
	}
	first, _, _ := strings.Cut(string(out), "\n")
	return strings.TrimSpace(first), nil
}

// versionLine is the first line of what --version prints, and the exact string
// selfUpdate matches a binary against. Everything after it is decoration.
func versionLine(tag string) string { return "codehalter " + tag }

// buildStamp is when and from what this binary was built: the commit date and
// short hash Go records in every binary built from a git checkout, release
// builds included ("2026-09-24, 73b7d58", "+dirty" when the tree had edits).
// "" when the binary carries no such record (a `go test` binary does not).
// A version number says which release; this says how old that release is.
func buildStamp() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return ""
	}
	var date, rev, dirty string
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.time":
			if len(s.Value) >= 10 {
				date = s.Value[:10]
			}
		case "vcs.revision":
			if len(s.Value) >= 7 {
				rev = s.Value[:7]
			}
		case "vcs.modified":
			if s.Value == "true" {
				dirty = "+dirty"
			}
		}
	}
	if date == "" && rev == "" {
		return ""
	}
	return strings.TrimPrefix(date+", "+rev+dirty, ", ")
}

// versionStamp is the version with its build stamp, "v78 (2026-09-24,
// 73b7d58)": what the banner shows after "codehalter" and what the spec ledger
// records against each item. The stamp is what tells two builds of one day
// apart.
func versionStamp() string {
	if stamp := buildStamp(); stamp != "" {
		return version + " (" + stamp + ")"
	}
	return version
}

// versionBanner is the banner's first words: "codehalter v78 (2026-09-24, 73b7d58)".
func versionBanner() string { return "codehalter " + versionStamp() }

// offerUpdate is the whole interaction for a terminal run: check, ask, install
// and re-exec, before anything else starts, so no session or container is
// lost. ask is false for a one-shot (-p) run or a non-terminal stdin, which
// print the notice and carry on. $CODEHALTER_UPDATE=yes skips the question,
// which is how the launcher stops the copy in the container from asking again.
func offerUpdate(ctx context.Context, cwd string, ask bool) {
	tag := newerRelease(ctx, cwd)
	if tag == "" {
		return
	}
	consented := os.Getenv(envUpdate) == "yes"
	if !consented && !ask {
		fmt.Printf("codehalter %s is available (running %s). Run codehalter --update to install it.\n", tag, version)
		// A run nobody is watching cannot consent, so the answer for the whole
		// run is no. Recording it keeps the copy in the container from printing
		// the same line about the same release a second time.
		remember(envUpdate, "skip")
		return
	}
	if !consented {
		fmt.Printf("codehalter %s is available (running %s). Update now? [Y/n] ", tag, version)
		// One unbuffered read, because the CLI's own stdin reader does not exist
		// yet and a bufio.Reader here could swallow the bytes it is about to
		// want. ask is only true for a terminal, where the line discipline hands
		// over exactly one line per read.
		buf := make([]byte, 64)
		n, err := os.Stdin.Read(buf)
		if err != nil {
			fmt.Println()
			return
		}
		switch strings.ToLower(strings.TrimSpace(string(buf[:n]))) {
		case "", "y", "yes":
		default:
			// Declining is a decision for the whole run, not for this process:
			// without it the copy inside the container asks the same question.
			remember(envUpdate, "skip")
			return
		}
		remember(envUpdate, "yes")
	}

	fmt.Printf("downloading codehalter %s ...\n", tag)
	path, err := selfUpdate(ctx, tag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "update failed: %v\ncontinuing with %s\n", err, version)
		// One failure is enough: a container that would hit the same unwritable
		// mount or the same missing asset should report it, not retry it.
		remember(envUpdate, "skip")
		return
	}
	fmt.Printf("installed codehalter %s to %s, restarting\n", tag, path)
	// Replace this process rather than return: the binary that was started is
	// gone, and the point of updating before anything else runs is that the run
	// the user asked for happens on the new version.
	if err := syscall.Exec(path, os.Args, os.Environ()); err != nil {
		fmt.Fprintf(os.Stderr, "could not restart %s: %v\nrun it again to use %s\n", path, err, tag)
	}
}

// offerSelfUpdate is the editor's half of offerUpdate: same check, same
// install, but it never re-execs. This process holds the editor's stdio and a
// live session, and replacing the process would drop both.
//
// Installing over the running binary is safe even so. The rename in selfUpdate
// unlinks the inode this process is executing, which keeps running to the end
// of the session, and the new file is what the editor spawns next. So the user
// accepts a card, finishes what they were doing, and starts a new thread on the
// new version: no shell inside the container, no image rebuild.
//
// $CODEHALTER_UPDATE=yes means the terminal half already asked on the host and
// was told yes, so the copy in the container installs without asking twice.
func (a *agent) offerSelfUpdate(ctx context.Context, sess *Session, sid string) {
	// Free after the banner's own call: the tag is in $CODEHALTER_LATEST.
	tag := newerRelease(ctx, sess.Cwd)
	if tag == "" {
		return
	}
	tcId := ""
	if os.Getenv(envUpdate) != "yes" {
		ok, id, err := a.askYesNoWithCard(ctx, sid,
			fmt.Sprintf("codehalter %s is available (running %s). Install it into this container?", tag, version),
			"think", "Install", "Not now")
		tcId = id
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return
		}
		if !ok {
			// For the life of this process, which is every thread of this editor
			// window: one decline answers for the release, not for the thread.
			remember(envUpdate, "skip")
			a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(
				"Staying on " + version + ". `codehalter --update` installs it whenever you want.")})
			return
		}
	}
	if tcId == "" {
		tcId = a.StartToolCall(ctx, sid, "Installing codehalter "+tag, "think", nil)
	}
	// The asset is several MB over a link codehalter does not control, so the
	// thread says something while it downloads.
	stopBeat := a.heartbeat(ctx, sid)
	path, err := selfUpdate(ctx, tag)
	stopBeat()
	// Answered either way, for the life of this process: a container whose
	// install dir is read-only would report the same failure at every thread,
	// and a success has nothing left to offer.
	remember(envUpdate, "skip")
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, fmt.Sprintf("update failed: %v (staying on %s)", err, version))
		a.say(ctx, sid, fmt.Sprintf("\n⚠ Update to %s failed: %v. Staying on %s.\n", tag, err, version))
		return
	}
	// The card's content is collapsed in Zed until clicked; its title and a
	// plain line are what the user actually sees, so the outcome goes there.
	done := fmt.Sprintf("Installed codehalter %s to %s. Restart Zed to use %s: this process keeps running %s until then, and the thread comes back as it is. The container does not need rebuilding.",
		tag, path, tag, version)
	a.CompleteToolCallTitled(ctx, sid, tcId, "Installed codehalter "+tag+": restart Zed to use it", []ToolCallContent{TextContent(done)})
	a.say(ctx, sid, "\n✅ "+done+"\n")
}

// runUpdate is --update: check and install with no question asked, for the run
// that is not a terminal. It prints what it did and never re-executes, because
// installing is all it was asked to do.
func runUpdate() int {
	ctx := context.Background()
	if version == "dev" {
		fmt.Println("this is a development build (no release tag), so there is nothing to update from")
		return 1
	}
	tag, err := latestRelease(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "could not check for updates: %v\n", err)
		return 1
	}
	if releaseNum(tag) <= releaseNum(version) {
		fmt.Printf("codehalter %s is the latest release\n", version)
		return 0
	}
	fmt.Printf("downloading codehalter %s (running %s) ...\n", tag, version)
	path, err := selfUpdate(ctx, tag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "update failed: %v\n", err)
		return 1
	}
	fmt.Printf("installed codehalter %s to %s\n", tag, path)
	return 0
}
