package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/BurntSushi/toml"
)

// bench drives codehalter through a sequence of test cases, spawning it
// inside each test repo's devcontainer over ACP. Each run writes one row
// to bench/results.jsonl with timing + pass/fail. Designed to be run on
// the host (or any environment with `devcontainer` CLI + git on PATH).
//
// Usage:
//   ./bench                               # run every test under bench/tests/
//   ./bench tests/preveltekit_go126.toml  # run one specific test

func main() {
	settingsFlag := flag.String("settings", "settings.toml", "settings.toml copied into each test project's .codehalter/")
	dcFlag := flag.String("devcontainer", "devcontainer", "directory whose contents are overlaid into clones that don't ship .devcontainer/")
	binFlag := flag.String("codehalter", "../codehalter", "pre-built codehalter binary to stage inside the devcontainer")
	workFlag := flag.String("work", "/tmp/codehalter-bench", "directory where per-test clones live (host path bind-mounted into the devcontainer)")
	resultsFlag := flag.String("results", "results.jsonl", "append one JSON row per test here")
	archiveFlag := flag.String("archive", "results", "directory under which per-run timestamped artifact folders are created (empty disables)")
	noteFlag := flag.String("note", "", "free-form tag recorded on every result row of this run (e.g. \"testing MTP\")")
	flag.Parse()

	settings, err := filepath.Abs(*settingsFlag)
	if err != nil {
		log.Fatalf("resolve settings: %v", err)
	}
	// Soft check only — a missing fallback is fine if every test specifies
	// its own `settings` path. runTest re-validates per test and produces a
	// clean error row when neither path resolves.
	if _, err := os.Stat(settings); err != nil {
		fmt.Fprintf(os.Stderr, "warn: fallback settings %s missing — tests without their own `settings` field will fail\n", settings)
	}
	dc, err := filepath.Abs(*dcFlag)
	if err != nil {
		log.Fatalf("resolve devcontainer: %v", err)
	}
	// Same soft check — only tests whose clone already ships .devcontainer/
	// and don't override `devcontainer` get away without it.
	if _, err := os.Stat(dc); err != nil {
		fmt.Fprintf(os.Stderr, "warn: fallback devcontainer %s missing — tests whose clone has no .devcontainer/ will fail\n", dc)
	}
	bin, err := filepath.Abs(*binFlag)
	if err != nil {
		log.Fatalf("resolve codehalter: %v", err)
	}
	if _, err := os.Stat(bin); err != nil {
		log.Fatalf("codehalter binary %s missing — run `just build` in the repo root first", bin)
	}
	work, err := filepath.Abs(*workFlag)
	if err != nil {
		log.Fatalf("resolve workdir: %v", err)
	}
	if err := os.MkdirAll(work, 0o755); err != nil {
		log.Fatalf("create workdir: %v", err)
	}
	resultsPath, err := filepath.Abs(*resultsFlag)
	if err != nil {
		log.Fatalf("resolve results: %v", err)
	}
	// One timestamp per `just bench` invocation, shared across every test —
	// makes it trivial to grab "the run I just did" from results/<ts>/.
	// Filesystem-safe format (no colons, no spaces) so `cd` autocompletes.
	var archiveRun string
	if *archiveFlag != "" {
		archiveBase, err := filepath.Abs(*archiveFlag)
		if err != nil {
			log.Fatalf("resolve archive: %v", err)
		}
		archiveRun = filepath.Join(archiveBase, time.Now().Format("2006-01-02_15-04-05"))
		if err := os.MkdirAll(archiveRun, 0o755); err != nil {
			log.Fatalf("create archive dir: %v", err)
		}
		fmt.Fprintf(os.Stderr, "archive: %s\n", archiveRun)
	}

	tests, err := collectTests(flag.Args())
	if err != nil {
		log.Fatalf("collect tests: %v", err)
	}
	if len(tests) == 0 {
		log.Fatalf("no tests found")
	}

	resultsF, err := os.OpenFile(resultsPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatalf("open results: %v", err)
	}
	defer resultsF.Close()

	ctx := context.Background()
	if *noteFlag != "" {
		fmt.Fprintf(os.Stderr, "note: %s\n", *noteFlag)
	}
	for _, tc := range tests {
		fmt.Fprintf(os.Stderr, "\n=== %s ===\n", tc.Name)
		res, workDir, runErr := runTest(ctx, tc, settings, dc, bin, work, *noteFlag, archiveRun)
		if runErr != nil {
			fmt.Fprintf(os.Stderr, "FAIL %s: %v (workdir: %s)\n", tc.Name, runErr, workDir)
		} else if res.OK {
			fmt.Fprintf(os.Stderr, "PASS %s — agent %dms, verify %dms\n", tc.Name, res.AgentMs, res.VerifyMs)
		} else {
			fmt.Fprintf(os.Stderr, "FAIL %s — stop=%s verify_exit=%d (workdir: %s)\n",
				tc.Name, res.StopReason, res.VerifyExit, workDir)
		}
		row, _ := json.Marshal(res)
		fmt.Fprintln(resultsF, string(row))
	}
}

// collectTests resolves either explicit test files passed on the command line
// or, when none given, every *.toml under bench/tests/. Paths are returned
// in stable lexicographic order so re-runs are reproducible.
func collectTests(args []string) ([]testCase, error) {
	var paths []string
	if len(args) == 0 {
		matches, err := filepath.Glob("tests/*.toml")
		if err != nil {
			return nil, err
		}
		paths = matches
	} else {
		paths = args
	}

	var out []testCase
	for _, p := range paths {
		var tc testCase
		if _, err := toml.DecodeFile(p, &tc); err != nil {
			return nil, fmt.Errorf("decode %s: %w", p, err)
		}
		abs, err := filepath.Abs(p)
		if err != nil {
			return nil, fmt.Errorf("abs %s: %w", p, err)
		}
		tc.sourcePath = abs
		if tc.Name == "" {
			tc.Name = filepath.Base(p[:len(p)-len(filepath.Ext(p))])
		}
		out = append(out, tc)
	}
	return out, nil
}
