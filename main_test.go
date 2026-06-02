package main

import (
	"os"
	"path/filepath"
	"testing"
)

// newTestAgent returns an agent with one session rooted at a fresh tempdir.
// a.conn is left nil so sendUpdate becomes a no-op (covered by the nil-check).
func newTestAgent(t *testing.T) (*agent, *Session) {
	t.Helper()
	s, err := newSession(t.TempDir())
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	return &agent{sessions: map[string]*Session{s.ID: s}}, s
}

// withFreshToolRegistry saves and restores the package-global registeredTools
// around a test body, so registering fake tools doesn't leak into other tests.
func withFreshToolRegistry(t *testing.T) {
	t.Helper()
	saved := registeredTools
	registeredTools = nil
	t.Cleanup(func() { registeredTools = saved })
}

// TestCwdOrDefaultAbsolutes pins the contract that sess.Cwd is always an
// absolute path. The bench harness sends Cwd: "." over ACP, and an unresolved
// "." breaks resolvePath's prefix check (filepath.Clean drops the leading
// "./" so "go.mod" matches neither "./" nor "."). resolvePath then rejects
// every project-relative path with "outside project directory".
func TestCwdOrDefaultAbsolutes(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	for _, in := range []string{".", "", "./"} {
		got := cwdOrDefault(in)
		if !filepath.IsAbs(got) {
			t.Errorf("cwdOrDefault(%q) = %q, want absolute path", in, got)
		}
		if got != cwd {
			t.Errorf("cwdOrDefault(%q) = %q, want %q", in, got, cwd)
		}
	}
}

// TestCwdAvailable pins the guard that stops session/new and session/load from
// scaffolding .codehalter under a workspace root that isn't mounted here — the
// case where Zed restores an agent thread pinned to another project's cwd.
func TestCwdAvailable(t *testing.T) {
	dir := t.TempDir()
	if err := cwdAvailable(dir); err != nil {
		t.Errorf("cwdAvailable(existing dir) = %v, want nil", err)
	}

	missing := filepath.Join(dir, "does-not-exist")
	if err := cwdAvailable(missing); err == nil {
		t.Errorf("cwdAvailable(missing) = nil, want error")
	}

	file := filepath.Join(dir, "afile")
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if err := cwdAvailable(file); err == nil {
		t.Errorf("cwdAvailable(file) = nil, want error")
	}
}
