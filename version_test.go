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

// XDG_CACHE_HOME covers Linux, HOME covers macOS: os.UserCacheDir reads a different variable on each.
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

// The returned counter counts latest-release API calls.
func releaseServer(t *testing.T, tag, asset string) (*httptest.Server, *int) {
	t.Helper()
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/releases/latest") {
			etag := fmt.Sprintf("%q", tag)
			if r.Header.Get("If-None-Match") == etag {
				w.WriteHeader(http.StatusNotModified)
				return
			}
			calls++
			w.Header().Set("ETag", etag)
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

func fakeBinary(reports string, size int) string {
	s := "#!/bin/sh\necho '" + reports + "'\n#"
	return s + strings.Repeat("x", max(0, size-len(s)))
}

// A young cache must not stand in for the check; it answers only when GitHub is unreachable.
func TestLatestReleaseSources(t *testing.T) {
	isolateUpdate(t)
	srv, calls := releaseServer(t, "v42", "")

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
	young, _ := json.Marshal(updateCache{Tag: "v7", ETag: `"v7"`, Checked: time.Now().Add(-time.Hour)})
	if err := os.WriteFile(updateCachePath(), young, 0o644); err != nil {
		t.Fatal(err)
	}
	if tag, err := latestRelease(context.Background()); err != nil || tag != "v42" {
		t.Fatalf("young cache with an old answer: got %q, %v; want GitHub's v42", tag, err)
	}
	if *calls != 1 {
		t.Errorf("%d counted API calls, want 1", *calls)
	}
	var back updateCache
	buf, err := os.ReadFile(updateCachePath())
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(buf, &back); err != nil || back.Tag != "v42" || back.ETag != `"v42"` {
		t.Errorf("the fresh answer and its ETag were not cached: %+v, %v", back, err)
	}

	if tag, err := latestRelease(context.Background()); err != nil || tag != "v42" {
		t.Fatalf("after a 304: got %q, %v; want v42", tag, err)
	}
	if *calls != 1 {
		t.Errorf("a 304 was counted: %d calls", *calls)
	}

	srv.Close()
	if tag, err := latestRelease(context.Background()); err != nil || tag != "v42" {
		t.Fatalf("unreachable with a young cache: got %q, %v; want v42", tag, err)
	}
	stale, _ := json.Marshal(updateCache{Tag: "v42", ETag: `"v42"`, Checked: time.Now().Add(-2 * updateTTL)})
	if err := os.WriteFile(updateCachePath(), stale, 0o644); err != nil {
		t.Fatal(err)
	}
	if tag, err := latestRelease(context.Background()); err == nil {
		t.Fatalf("unreachable with a stale cache answered %q; want an error", tag)
	}
}

// The resolved tag is published to the environment even when no update is offered.
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

// Every rejection must leave the installed binary untouched.
func TestSelfUpdateReplacesTheBinary(t *testing.T) {
	install := func(t *testing.T) string {
		t.Helper()
		dir := t.TempDir()
		self := filepath.Join(dir, "codehalter")
		if err := os.WriteFile(self, []byte("#!/bin/sh\necho 'codehalter v41'\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		executable = func() (string, error) { return self, nil }
		// installTarget accepts only the file that reports the running version.
		version = "v41"
		t.Cleanup(func() { executable = os.Executable; version = "dev" })
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
		// Older releases printed a build stamp on a second --version line; it must be tolerated.
		releaseServer(t, "v42", fakeBinary(versionLine("v42")+"\nbuilt 2026-09-24, abc1234", updateMinBytes+1))
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

	// Under Alpine gcompat the process's executable is the musl loader.
	t.Run("refuses to replace a file that is not this program", func(t *testing.T) {
		isolateUpdate(t)
		self := install(t)
		loader := filepath.Join(filepath.Dir(self), "ld-musl-x86_64.so.1")
		if err := os.WriteFile(loader, []byte("#!/bin/sh\necho 'musl libc (x86_64)' >&2; exit 1\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		executable = func() (string, error) { return loader, nil }
		releaseServer(t, "v42", fakeBinary(versionLine("v42"), updateMinBytes+1))
		_, err := selfUpdate(context.Background(), "v42")
		if err == nil || !strings.Contains(err.Error(), "refusing to update") {
			t.Fatalf("selfUpdate over a loader: err = %v, want a refusal", err)
		}
		if buf, _ := os.ReadFile(loader); !strings.Contains(string(buf), "musl libc") {
			t.Fatalf("the loader was replaced:\n%.80s", buf)
		}
		if _, err := os.Stat(self); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("refuses a binary reporting another version", func(t *testing.T) {
		isolateUpdate(t)
		self := install(t)
		// Otherwise an update that does not change the version would re-exec forever.
		releaseServer(t, "v42", fakeBinary(versionLine("dev"), updateMinBytes+1))
		if _, err := selfUpdate(context.Background(), "v42"); err == nil {
			t.Fatal("selfUpdate accepted a binary reporting a different version")
		}
		unchanged(t, self)
	})
}
