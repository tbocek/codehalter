package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// isolateUpdate points the update machinery at a private cache directory and
// clears the two env vars, so a test never reads the developer's real cache,
// never writes to it, and never inherits a decision from the environment it
// runs in. XDG_CACHE_HOME covers Linux, HOME covers macOS (os.UserCacheDir
// reads a different variable on each).
func isolateUpdate(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", dir)
	t.Setenv("HOME", dir)
	t.Setenv(envLatest, "")
	t.Setenv(envUpdate, "")
	old := version
	t.Cleanup(func() { version = old })
}

// releaseServer serves the GitHub endpoints: the latest-release JSON, and the
// per-platform asset, whose body is whatever the test wants the "binary" to be.
// It counts API calls so a test can prove the cache prevented one.
func releaseServer(t *testing.T, tag, asset string) (*httptest.Server, *int) {
	t.Helper()
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/releases/latest") {
			calls++
			fmt.Fprintf(w, `{"tag_name": %q}`, tag)
			return
		}
		fmt.Fprint(w, asset)
	}))
	t.Cleanup(srv.Close)
	updateLatestURL, updateAssetURL = srv.URL+"/releases/latest", srv.URL+"/download/%s/%s-%s"
	t.Cleanup(func() {
		updateLatestURL = "https://api.github.com/repos/" + updateRepo + "/releases/latest"
		updateAssetURL = "https://github.com/" + updateRepo + "/releases/download/%s/codehalter-%s-%s"
	})
	return srv, &calls
}

// fakeBinary is a runnable stand-in for a downloaded release: a shell script
// that answers --version like the real one, padded past updateMinBytes so it
// gets through the "too small to be the binary" gate.
func fakeBinary(reports string, size int) string {
	s := "#!/bin/sh\necho '" + reports + "'\n#"
	return s + strings.Repeat("x", max(0, size-len(s)))
}

func TestReleaseNum(t *testing.T) {
	for _, c := range []struct {
		tag  string
		want int
	}{
		{"v1", 1},
		{"v42", 42},
		{"42", 42},    // the API returns the tag; tolerate a missing v
		{"dev", 0},    // an own build is older than every release
		{"", 0},       //
		{"v0", 0},     // release.sh starts at v1, so v0 is not a release
		{"v1.2", 0},   // not the scheme release.sh writes
		{"v-3", 0},    //
		{"vNaN", 0},   //
		{"latest", 0}, //
	} {
		if got := releaseNum(c.tag); got != c.want {
			t.Errorf("releaseNum(%q) = %d, want %d", c.tag, got, c.want)
		}
	}
}

// TestLatestReleaseSources: three answers, cheapest first. The environment is
// what the launcher fills in for the container, the cache is what keeps a
// restart off the rate-limited API, and only a stale cache reaches the network.
func TestLatestReleaseSources(t *testing.T) {
	isolateUpdate(t)
	_, calls := releaseServer(t, "v42", "")

	t.Setenv(envLatest, "v99")
	if tag, err := latestRelease(context.Background()); err != nil || tag != "v99" {
		t.Fatalf("with %s set: got %q, %v; want v99 and no error", envLatest, tag, err)
	}
	if *calls != 0 {
		t.Errorf("the environment already had the answer, but the API was called %d times", *calls)
	}
	t.Setenv(envLatest, "")

	if err := os.MkdirAll(cacheDir(), 0o755); err != nil {
		t.Fatal(err)
	}
	fresh, _ := json.Marshal(updateCache{Tag: "v7", Checked: time.Now().Add(-time.Hour)})
	if err := os.WriteFile(updateCachePath(), fresh, 0o644); err != nil {
		t.Fatal(err)
	}
	if tag, err := latestRelease(context.Background()); err != nil || tag != "v7" {
		t.Fatalf("fresh cache: got %q, %v; want v7 and no error", tag, err)
	}
	if *calls != 0 {
		t.Errorf("a %v-old cache entry should answer on its own, but the API was called %d times", updateTTL, *calls)
	}

	stale, _ := json.Marshal(updateCache{Tag: "v7", Checked: time.Now().Add(-2 * updateTTL)})
	if err := os.WriteFile(updateCachePath(), stale, 0o644); err != nil {
		t.Fatal(err)
	}
	if tag, err := latestRelease(context.Background()); err != nil || tag != "v42" {
		t.Fatalf("stale cache: got %q, %v; want the API's v42", tag, err)
	}
	if *calls != 1 {
		t.Errorf("stale cache: %d API calls, want exactly 1", *calls)
	}
	var back updateCache
	buf, err := os.ReadFile(updateCachePath())
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(buf, &back); err != nil || back.Tag != "v42" {
		t.Errorf("the fresh answer was not cached: %+v, %v", back, err)
	}
}

// TestNewerReleaseGates covers every way the check declines to nag, plus the
// one way it speaks up. The tag it resolved is published to the environment
// either way, because the container started next should not re-resolve it.
func TestNewerReleaseGates(t *testing.T) {
	cwd := t.TempDir()

	t.Run("own build never offers", func(t *testing.T) {
		isolateUpdate(t)
		_, calls := releaseServer(t, "v42", "")
		version = "dev"
		if got := newerRelease(context.Background(), cwd); got != "" {
			t.Errorf("got %q, want no offer for a dev build", got)
		}
		if *calls != 0 {
			t.Errorf("a dev build should not even ask: %d API calls", *calls)
		}
	})

	t.Run("skip decided elsewhere", func(t *testing.T) {
		isolateUpdate(t)
		_, calls := releaseServer(t, "v42", "")
		version = "v1"
		t.Setenv(envUpdate, "skip")
		if got := newerRelease(context.Background(), cwd); got != "" {
			t.Errorf("got %q, want silence when the answer was already no", got)
		}
		if *calls != 0 {
			t.Errorf("%s=skip should keep it off the network: %d API calls", envUpdate, *calls)
		}
	})

	t.Run("switched off in settings", func(t *testing.T) {
		isolateUpdate(t)
		_, calls := releaseServer(t, "v42", "")
		version = "v1"
		off := t.TempDir()
		if err := os.MkdirAll(filepath.Join(off, sessionDir), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(off, sessionDir, "settings.toml"),
			[]byte("update_check = false\n[[llm]]\nserver = \"http://x\"\nmodel = \"m\"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if got := newerRelease(context.Background(), off); got != "" {
			t.Errorf("got %q, want silence with update_check = false", got)
		}
		if *calls != 0 {
			t.Errorf("update_check = false should keep it off the network: %d API calls", *calls)
		}
	})

	t.Run("already current", func(t *testing.T) {
		isolateUpdate(t)
		releaseServer(t, "v42", "")
		version = "v42"
		if got := newerRelease(context.Background(), cwd); got != "" {
			t.Errorf("got %q, want no offer when running the latest", got)
		}
		if os.Getenv(envLatest) != "v42" {
			t.Errorf("%s = %q, want the resolved tag published even when there is nothing to do",
				envLatest, os.Getenv(envLatest))
		}
	})

	t.Run("newer release", func(t *testing.T) {
		isolateUpdate(t)
		releaseServer(t, "v42", "")
		version = "v41"
		if got := newerRelease(context.Background(), cwd); got != "v42" {
			t.Errorf("got %q, want v42", got)
		}
		if os.Getenv(envLatest) != "v42" {
			t.Errorf("%s = %q, want v42", envLatest, os.Getenv(envLatest))
		}
	})
}

// TestSelfUpdateReplacesTheBinary: the happy path, and the two ways a download
// is rejected. Both rejections must leave the installed binary untouched: this
// is the one operation in codehalter that can destroy the program itself.
func TestSelfUpdateReplacesTheBinary(t *testing.T) {
	install := func(t *testing.T) string {
		t.Helper()
		dir := t.TempDir()
		self := filepath.Join(dir, "codehalter")
		if err := os.WriteFile(self, []byte("#!/bin/sh\necho 'codehalter v41'\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		executable = func() (string, error) { return self, nil }
		t.Cleanup(func() { executable = os.Executable })
		return self
	}
	unchanged := func(t *testing.T, self string) {
		t.Helper()
		buf, err := os.ReadFile(self)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(buf), "v41") {
			t.Fatalf("the installed binary was replaced by a download that should have been refused:\n%.80s", buf)
		}
		entries, err := os.ReadDir(filepath.Dir(self))
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 1 {
			t.Errorf("a refused download left files behind: %v", entries)
		}
	}

	t.Run("installs and reports the path", func(t *testing.T) {
		isolateUpdate(t)
		self := install(t)
		releaseServer(t, "v42", fakeBinary(versionLine("v42"), updateMinBytes+1))
		got, err := selfUpdate(context.Background(), "v42")
		if err != nil {
			t.Fatalf("selfUpdate: %v", err)
		}
		if got != self {
			t.Errorf("replaced %q, want %q", got, self)
		}
		buf, err := os.ReadFile(self)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(buf), versionLine("v42")) {
			t.Errorf("the new binary is not in place:\n%.80s", buf)
		}
	})

	t.Run("refuses a body too small to be a binary", func(t *testing.T) {
		isolateUpdate(t)
		self := install(t)
		releaseServer(t, "v42", "404: Not Found")
		if _, err := selfUpdate(context.Background(), "v42"); err == nil {
			t.Fatal("selfUpdate accepted a 14-byte error page")
		}
		unchanged(t, self)
	})

	t.Run("refuses a binary reporting another version", func(t *testing.T) {
		isolateUpdate(t)
		self := install(t)
		// The stamp is what stops a re-exec loop: a binary that does not
		// report the tag that was asked for never gets to replace anything.
		releaseServer(t, "v42", fakeBinary(versionLine("dev"), updateMinBytes+1))
		if _, err := selfUpdate(context.Background(), "v42"); err == nil {
			t.Fatal("selfUpdate accepted a binary reporting a different version")
		}
		unchanged(t, self)
	})
}
