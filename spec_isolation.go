package main

import (
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

// specStrayShown caps the paths named; the rest is a count.
const specStrayShown = 20

type fileStamp struct {
	size int64
	mod  time.Time
}

// writeWatch tells what a test run wrote outside its own temporary directory: the project's
// files (build and dot directories aside, as everywhere) and the names at the top of the
// system's temp directory. A test that writes a fixed path collides with the same test run
// beside it: two suites at once into one tree once failed 196 test files.
type writeWatch struct {
	root, fresh string
	files       map[string]fileStamp
	tmpNames    map[string]bool
}

// watchWrites takes the state before the run; fresh is the run's own $TMPDIR.
func watchWrites(root, fresh string) *writeWatch {
	return &writeWatch{root: root, fresh: fresh, files: projectStamps(root), tmpNames: tmpNames(filepath.Dir(fresh))}
}

// strays lists what the run wrote outside fresh: project paths relative to the project,
// temp entries absolute, at most specStrayShown and a count of the rest.
func (w *writeWatch) strays() []string {
	var out []string
	for rel, now := range projectStamps(w.root) {
		if before, ok := w.files[rel]; !ok || before != now {
			out = append(out, rel)
		}
	}
	tmp := filepath.Dir(w.fresh)
	for name := range tmpNames(tmp) {
		// codehalter's own pid and log files live there too.
		if !w.tmpNames[name] && name != filepath.Base(w.fresh) && !strings.HasPrefix(name, "codehalter-") {
			out = append(out, filepath.Join(tmp, name))
		}
	}
	slices.Sort(out)
	if len(out) > specStrayShown {
		out = append(out[:specStrayShown], fmt.Sprintf("and %d more", len(out)-specStrayShown))
	}
	return out
}

func projectStamps(root string) map[string]fileStamp {
	stamps := map[string]fileStamp{}
	walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // a file gone mid-walk is no write of the run's
		}
		if d.IsDir() {
			if path != root && skipWalkDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		if rel, err := filepath.Rel(root, path); err == nil {
			stamps[filepath.ToSlash(rel)] = fileStamp{info.Size(), info.ModTime()}
		}
		return nil
	})
	if walkErr != nil {
		slog.Debug("spec: write watch walk", "root", root, "err", walkErr)
	}
	return stamps
}

func tmpNames(dir string) map[string]bool {
	names := map[string]bool{}
	entries, err := os.ReadDir(dir)
	if err != nil {
		slog.Debug("spec: write watch temp dir", "dir", dir, "err", err)
		return names
	}
	for _, e := range entries {
		names[e.Name()] = true
	}
	return names
}
