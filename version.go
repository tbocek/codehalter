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
	"strconv"
	"strings"
	"syscall"
	"time"
)

// ---------------------------------------------------------------------------
// Update check and self-update.
//
// Releases are plain integer tags (v1, v2, ...): release.sh reads the last one,
// adds one, pushes the tag, and the build workflow attaches one binary per
// GOOS/GOARCH and stamps the tag into `version` with -ldflags. So "is there a
// newer one" is an integer comparison, and a binary built any other way reports
// "dev", compares as older than nothing, and is never offered an update: a
// developer running their own build does not want it overwritten by a download.
//
// Three env vars carry a decision that was already made, so it is made once per
// run rather than once per process. The launcher starts a second codehalter
// inside the container and a self-update re-executes a third, and asking the
// same question in each of them would be three prompts for one intent:
//
//	CODEHALTER_LATEST=v42   the newest release, already resolved. Skips the API
//	                        call (and its cache) entirely.
//	CODEHALTER_UPDATE=yes   the user consented in this run: update, don't ask.
//	CODEHALTER_UPDATE=skip  don't check, don't ask, don't update.
//
// They are also the manual controls: CODEHALTER_UPDATE=skip in CI keeps the
// check off the network, and CODEHALTER_UPDATE=yes makes an unattended run
// self-update.
//
// Nothing here is allowed to stop codehalter from starting. No network, a rate
// limit, an unwritable install directory: each one is reported in a line and
// the run continues on the version already installed.
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

// updateCachePath is where the last answer from the GitHub API is kept. The
// unauthenticated API allows 60 requests per hour per IP (release.sh runs into
// this too), and codehalter starts often enough that a check per start would
// spend that budget on a version number that changes a few times a month.
func updateCachePath() string { return filepath.Join(cacheDir(), "update.json") }

type updateCache struct {
	Tag     string    `json:"tag"`
	Checked time.Time `json:"checked"`
}

// latestRelease reports the newest release tag. It answers from $CODEHALTER_LATEST
// when the process that started us already resolved it, then from the cache file
// while that is younger than updateTTL, and only then asks GitHub.
func latestRelease(ctx context.Context) (string, error) {
	if tag := os.Getenv(envLatest); tag != "" {
		return tag, nil
	}
	var c updateCache
	if buf, err := os.ReadFile(updateCachePath()); err == nil {
		if err := json.Unmarshal(buf, &c); err == nil && c.Tag != "" && time.Since(c.Checked) < updateTTL {
			slog.Debug("update: answered from cache", "tag", c.Tag, "age", time.Since(c.Checked))
			return c.Tag, nil
		}
	}

	ctx, cancel := context.WithTimeout(ctx, updateTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, updateLatestURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	resp, err := metaHTTPClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		// 403/429 is the rate limit, and the body says so; it is the one failure
		// here a user can act on (wait, or use a different network), so it is
		// worth keeping in the message rather than reporting a bare status.
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 400))
		return "", fmt.Errorf("GitHub API: %s: %s", resp.Status, strings.TrimSpace(string(body)))
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

	// Best-effort cache write: a read-only cache dir costs an API call per run,
	// which is a reason to log, not to fail a check that already succeeded.
	if buf, err := json.Marshal(updateCache{Tag: rel.TagName, Checked: time.Now()}); err == nil {
		if err := os.MkdirAll(filepath.Dir(updateCachePath()), 0o755); err != nil {
			slog.Debug("update: cache dir not writable", "err", err)
		} else if err := os.WriteFile(updateCachePath(), buf, 0o644); err != nil {
			slog.Debug("update: cache not written", "err", err)
		}
	}
	return rel.TagName, nil
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

// selfUpdate installs the release asset for this platform over the running
// binary and reports the path it replaced.
//
// The download lands in the install directory (a rename is atomic only within
// one filesystem) and is only renamed into place once it has been run once and
// reported the tag that was asked for. A wrong-architecture, truncated or
// error-page "binary" therefore never replaces a working one, and neither does
// a build whose version stamp is missing, which is also what stops a re-exec
// loop: an update that does not change the version cannot commit.
func selfUpdate(ctx context.Context, tag string) (string, error) {
	self, err := executable()
	if err != nil {
		return "", err
	}
	// Follow the symlink: replacing the link itself would leave the real
	// binary in place and the next start would run the old one again.
	if resolved, err := filepath.EvalSymlinks(self); err == nil {
		self = resolved
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

	probe, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(probe, tmp.Name(), "--version").Output()
	if err != nil {
		return "", fmt.Errorf("the downloaded binary does not run here: %w", err)
	}
	if got := strings.TrimSpace(string(out)); got != versionLine(tag) {
		return "", fmt.Errorf("the downloaded binary reports %q, not %q", got, versionLine(tag))
	}
	if err := os.Rename(tmp.Name(), self); err != nil {
		return "", err
	}
	return self, nil
}

// versionLine is the whole of what --version prints, and the exact string
// selfUpdate matches the downloaded binary against.
func versionLine(tag string) string { return "codehalter " + tag }

// offerUpdate is the whole interaction for a terminal run: check, ask, install,
// and re-exec into the new binary so the run continues on it. It returns before
// anything else starts, so there is no session or container to lose.
//
// ask is false for a one-shot (-p) run and for a run whose stdin is not a
// terminal: a script must not block on a question nobody is there to answer, so
// those print the notice and carry on. $CODEHALTER_UPDATE=yes skips the
// question the other way, which is how the launcher stops the copy inside the
// container from asking again what the user already answered on the host.
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
