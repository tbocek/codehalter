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

// version is stamped by release builds with -ldflags "-X main.version=$tag"; "dev" never updates.
var version = "dev"

const (
	updateRepo    = "tbocek/codehalter"
	updateTTL     = 24 * time.Hour
	updateTimeout = 4 * time.Second
	// Anything smaller is an error page or a truncated transfer, never the binary.
	updateMinBytes = 4 << 20
)

// Vars so tests can point them at a local server and a throwaway binary.
var (
	updateLatestURL = "https://api.github.com/repos/" + updateRepo + "/releases/latest"
	updateAssetURL  = "https://github.com/" + updateRepo + "/releases/download/%s/codehalter-%s-%s"
	executable      = os.Executable
)

// envLatest (resolved tag) and envUpdate ("yes" or "skip") carry a decision already made to the
// launcher's container copy and a self-update's re-exec, so each is made once per run.
const (
	envLatest = "CODEHALTER_LATEST"
	envUpdate = "CODEHALTER_UPDATE"
)

func remember(key, value string) {
	if err := os.Setenv(key, value); err != nil {
		slog.Debug("update: could not record a decision", "key", key, "err", err)
	}
}

func releaseNum(tag string) int {
	n, err := strconv.Atoi(strings.TrimPrefix(tag, "v"))
	if err != nil || n <= 0 {
		return 0
	}
	return n
}

// The cache holds the ETag for a conditional request (a 304 does not count against GitHub's
// rate limit) and answers when GitHub is unreachable. It never skips the check.
func updateCachePath() string { return filepath.Join(cacheDir(), "update.json") }

type updateCache struct {
	Tag     string    `json:"tag"`
	ETag    string    `json:"etag,omitempty"`
	Checked time.Time `json:"checked"`
}

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
		// Keep the body: for 403/429 it explains the rate limit, which the user can act on.
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

func newerRelease(ctx context.Context, cwd string) string {
	if os.Getenv(envUpdate) == "skip" {
		return ""
	}
	if version == "dev" {
		return ""
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

// selfUpdate replaces the running binary only after the download has run and reported the
// requested tag, so a broken download never replaces it and an update cannot loop.
func selfUpdate(ctx context.Context, tag string) (string, error) {
	self, err := installTarget(ctx)
	if err != nil {
		return "", err
	}

	// Same directory as the target: rename is atomic only within one filesystem.
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

// installTarget accepts only a path that prints this binary's exact version line. Under Alpine
// gcompat /proc/self/exe is the musl loader, and replacing that bricks the container; argv[0]
// on PATH is tried first because it survives such re-execs.
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
		// Replace the symlink's target, not the link, or the next start runs the old binary.
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

func versionLine(tag string) string { return "codehalter " + tag }

// buildStamp is "2026-09-24, 73b7d58[+dirty]" from the VCS build info, "" when absent.
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

func versionStamp() string {
	if stamp := buildStamp(); stamp != "" {
		return version + " (" + stamp + ")"
	}
	return version
}

func versionBanner() string { return "codehalter " + versionStamp() }

// offerUpdate runs before anything else starts; with ask false (no terminal) it only prints a notice.
func offerUpdate(ctx context.Context, cwd string, ask bool) {
	tag := newerRelease(ctx, cwd)
	if tag == "" {
		return
	}
	consented := os.Getenv(envUpdate) == "yes"
	if !consented && !ask {
		fmt.Printf("codehalter %s is available (running %s). Run codehalter --update to install it.\n", tag, version)
		remember(envUpdate, "skip")
		return
	}
	if !consented {
		fmt.Printf("codehalter %s is available (running %s). Update now? [Y/n] ", tag, version)
		// One unbuffered read: a bufio.Reader could swallow input meant for the CLI's own
		// reader, and a terminal hands over one line per read.
		buf := make([]byte, 64)
		n, err := os.Stdin.Read(buf)
		if err != nil {
			fmt.Println()
			return
		}
		switch strings.ToLower(strings.TrimSpace(string(buf[:n]))) {
		case "", "y", "yes":
		default:
			remember(envUpdate, "skip")
			return
		}
		remember(envUpdate, "yes")
	}

	fmt.Printf("downloading codehalter %s ...\n", tag)
	path, err := selfUpdate(ctx, tag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "update failed: %v\ncontinuing with %s\n", err, version)
		// The container copy would hit the same failure: report once, don't retry.
		remember(envUpdate, "skip")
		return
	}
	fmt.Printf("installed codehalter %s to %s, restarting\n", tag, path)
	if err := syscall.Exec(path, os.Args, os.Environ()); err != nil {
		fmt.Fprintf(os.Stderr, "could not restart %s: %v\nrun it again to use %s\n", path, err, tag)
	}
}

// offerSelfUpdate never re-execs: this process holds the editor's stdio and a live session.
// Renaming over the running binary is safe; the old inode runs until the editor respawns us.
func (a *agent) offerSelfUpdate(ctx context.Context, sess *Session, sid string) {
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
			// Covers every thread of this editor window, which all share this process.
			remember(envUpdate, "skip")
			a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(
				"Staying on " + version + ". `codehalter --update` installs it whenever you want.")})
			return
		}
	}
	if tcId == "" {
		tcId = a.StartToolCall(ctx, sid, "Installing codehalter "+tag, "think", nil)
	}
	stopBeat := a.heartbeat(ctx, sid)
	path, err := selfUpdate(ctx, tag)
	stopBeat()
	// Final either way for this process: a read-only install dir would fail at every thread.
	remember(envUpdate, "skip")
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, fmt.Sprintf("update failed: %v (staying on %s)", err, version))
		a.say(ctx, sid, fmt.Sprintf("\n⚠ Update to %s failed: %v. Staying on %s.\n", tag, err, version))
		return
	}
	// Zed collapses card content, so the outcome goes in the title and a plain line.
	done := fmt.Sprintf("Installed codehalter %s to %s. Restart Zed to use %s: this process keeps running %s until then, and the thread comes back as it is. The container does not need rebuilding.",
		tag, path, tag, version)
	a.CompleteToolCallTitled(ctx, sid, tcId, "Installed codehalter "+tag+": restart Zed to use it", []ToolCallContent{TextContent(done)})
	a.say(ctx, sid, "\n✅ "+done+"\n")
}

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
