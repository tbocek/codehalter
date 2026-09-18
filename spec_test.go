package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// writeSpecFixture lays out a miniature spec shaped like the one /spec was built
// against: numbered chapters with navigation bars, flows as headings, prose
// sections with no id, a parameter table, an index, a raw inventory repeating a
// flow id, and a heading that only mentions an id in parentheses.
func writeSpecFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"README.md":        "# The spec\n\nRead this first.\n\n## How to read\n\nOrientation only.\n",
		"00-principles.md": "# 00 Principles\n\n## 1. Rules\n\nStanding rules.\n",
		"01-files.md":      "# 01 Files\n\n## 1. Layout\n\nThe project folder.\n\n## 2. cut.json\n\nThe cut file format.\n",
		"03-shell.md": "# 03 Shell\n\n<!-- nav -->\n[← 01](01-files.md) · [↑](README.md)\n<!-- /nav -->\n\n" +
			"## 1. Screen\n\nThe window. See ![window](img/window.png) and [F0.1](#f01-switch-tab).\n\n" +
			"## 2. Flows\n\n" +
			"### F0.1 Switch tab\n\n<sub>[← start](#1-screen) · [F0.2 →](#f02-press-)</sub>\n\n" +
			"S1 Switch. Uses P.policy.padSeconds and writes [the cut file](01-files.md#2-cutjson). Next is [F0.2](#f02-press-).\n\n" +
			"### F0.2 Press ▶\n\nS1 Press. REVIEW: keep or drop.\n\n" +
			"### 3.1 Derivation (F0.1, P.policy.padSeconds)\n\nHow the policy is derived, which PREVIEWING does not change.\n",
		"09-llm.md":        "# 09 LLM\n\n## 2. Tool protocol (F6.1)\n\nThe protocol.\n",
		"10-parameters.md": "# 10 Parameters\n\n## 1. Policy\n\n| parameter | default |\n|---|---|\n| P.policy.padSeconds | 10 |\n| P.policy.minTake | 2 |\n",
		"11-index.md":      "# 11 Index\n\n## 1. All flows\n\n| id | flow |\n|---|---|\n| [F0.1](03-shell.md#f01-switch-tab) | Switch |\n| [F0.2](03-shell.md#f02-press-) | Press |\n| F6.1 | Tool protocol |\n",
		"inventory/cut.md": "# Raw inventory\n\n## F0.1 raw notes\n\nFrom the prototype.\n",
	}
	for rel, body := range files {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// TestScanSpecLedger pins what becomes an item, where each item is defined, and
// the order the loop takes them in.
func TestScanSpecLedger(t *testing.T) {
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	where := func(id string) (string, int) {
		t.Helper()
		it := idx.items[id]
		if it == nil {
			t.Fatalf("%s is not an item; items: %v", id, idx.order)
		}
		return idx.docs[it.Doc].rel, it.Kind
	}

	// A chapter heading defines a flow, not the raw inventory repeating it.
	if rel, kind := where("F0.1"); rel != "03-shell.md" || kind != specDefHeading {
		t.Errorf("F0.1 defined at %s kind %d, want the chapter heading", rel, kind)
	}
	// A heading that only names an id in parentheses is still the section about
	// it when no heading opens with it, and it beats the index row.
	if rel, kind := where("F6.1"); rel != "09-llm.md" || kind != specDefHeading {
		t.Errorf("F6.1 defined at %s kind %d, want 09's heading over the index row", rel, kind)
	}
	// "### 3.1 Derivation (F0.1, …)" mentions F0.1 and the parameter; neither
	// moves there, and the parameter stays defined by its table row.
	if rel, kind := where("P.policy.padSeconds"); rel != "10-parameters.md" || kind != specDefTableRow {
		t.Errorf("P.policy.padSeconds defined at %s kind %d, want the parameter table row", rel, kind)
	}

	// Prose sections with no id become items; context files and page titles don't.
	for _, id := range []string{"§01-files#1-layout", "§01-files#2-cutjson", "§03-shell#1-screen"} {
		if _, kind := where(id); kind != specDefSection {
			t.Errorf("%s kind %d, want a section item", id, kind)
		}
	}
	for id := range idx.items {
		if strings.HasPrefix(id, "§README") || strings.HasPrefix(id, "§00-principles") {
			t.Errorf("context file became an item: %s", id)
		}
		if strings.HasPrefix(id, "§11-index") || strings.HasPrefix(id, "§10-parameters") {
			t.Errorf("an index or a parameter table became an item: %s", id)
		}
		if strings.HasPrefix(id, "§inventory") {
			t.Errorf("a subdirectory section became an item: %s", id)
		}
	}

	// Document order, formats before the flows that use them; table ids last.
	want := []string{"§01-files#1-layout", "§01-files#2-cutjson", "§03-shell#1-screen", "F0.1", "F0.2"}
	if got := idx.order[:len(want)]; !reflect.DeepEqual(got, want) {
		t.Errorf("ledger starts %v, want %v", got, want)
	}
	last := idx.order[len(idx.order)-1]
	if idx.items[last].Kind == specDefHeading || idx.items[last].Kind == specDefSection {
		t.Errorf("ledger ends with %s, want parameter rows after every flow and section", last)
	}
}

// TestSpecSlice pins what one round is shown: its own section without the
// navigation bar, the rows of the parameters it cites, the section it links to,
// and a pointer (not a copy) to another flow.
func TestSpecSlice(t *testing.T) {
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	s := idx.slice("F0.1", "spec")
	for _, want := range []string{
		"### F0.1 Switch tab",
		"| P.policy.padSeconds | 10 |", // the cited parameter's row
		"| parameter | default |",      // with its column names
		"The cut file format.",         // the linked section
	} {
		if !strings.Contains(s.Text, want) {
			t.Errorf("slice lacks %q:\n%s", want, s.Text)
		}
	}
	if strings.Contains(s.Text, "<sub>") {
		t.Error("slice carries the navigation bar")
	}
	if strings.Contains(s.Text, "S1 Press.") {
		t.Error("slice pasted F0.2's section, which is another round's item")
	}
	if !reflect.DeepEqual(s.Related, []string{"spec/03-shell.md#f02-press-"}) {
		t.Errorf("related = %v, want a pointer to F0.2", s.Related)
	}

	screen := idx.slice("§03-shell#1-screen", "spec")
	if !reflect.DeepEqual(screen.Images, []string{"spec/img/window.png"}) {
		t.Errorf("images = %v, want the screen's picture", screen.Images)
	}
}

// TestSpecCoverage pins what counts as a test naming an id, and what doesn't.
func TestSpecCoverage(t *testing.T) {
	out := t.TempDir()
	write := func(rel, body string) {
		t.Helper()
		p := filepath.Join(out, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("tests/shell.rs", "#[test]\nfn f0_1_s1_switches() {}\n#[test]\nfn f0_10_other() {}\n")
	// Production code naming an id in a doc comment is not a test; its inline
	// test module is.
	write("src/lib.rs", "//! F0.2 lives here\npub fn press() {}\n#[cfg(test)]\nmod tests {\n    // P.policy.padSeconds\n    #[test] fn pad() {}\n}\n")
	write("src/cut.rs", "// §01-files#2-cutjson\npub fn load() {}\n")
	write("tests/files.rs", "#[test] fn sec_01_files_2_cutjson_round_trips() {}\n")
	write("target/debug/stale.rs", "#[test] fn f0_2_would_count() {}\n") // build output: never scanned

	ids := []string{"F0.1", "F0.2", "F0.10", "P.policy.padSeconds", "§01-files#2-cutjson", "F1.0"}
	covered, files, err := specCoverage(out, ids)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"F0.1", "F0.10", "P.policy.padSeconds", "§01-files#2-cutjson"} {
		if _, ok := covered[id]; !ok {
			t.Errorf("%s not covered, want it covered", id)
		}
	}
	for _, id := range []string{"F0.2", "F1.0"} {
		if f, ok := covered[id]; ok {
			t.Errorf("%s covered by %s, want uncovered", id, f)
		}
	}
	if files != 3 {
		t.Errorf("test sources = %d, want 3 (tests/shell.rs, tests/files.rs, src/lib.rs)", files)
	}
}

func TestSpecTestToken(t *testing.T) {
	for id, want := range map[string]string{
		"F2.3":                    "f2_3",
		"P.policy.minTakeSeconds": "p_policy_mintakeseconds",
		"tool:set_policy":         "tool_set_policy",
		"§05-cut#7-rules":         "sec_05_cut_7_rules",
	} {
		if got := specTestToken(id); got != want {
			t.Errorf("specTestToken(%q) = %q, want %q", id, got, want)
		}
	}
}

// TestGithubSlug pins the anchors the spec's own links are written against.
func TestGithubSlug(t *testing.T) {
	for title, want := range map[string]string{
		"F0.2 Press ▶":                        "f02-press-",
		"3. cut/cut.json":                     "3-cutcutjson",
		"2. Model roles (which job asks)":     "2-model-roles-which-job-asks",
		"05 — Cut":                            "05--cut",
		"8. Details confirmed (verification)": "8-details-confirmed-verification",
	} {
		if got := githubSlug(title); got != want {
			t.Errorf("githubSlug(%q) = %q, want %q", title, got, want)
		}
	}
}

func TestParseSpecArgs(t *testing.T) {
	cmd, spec, out, target, err := parseSpecArgs("spec/ rust/ use gtk4-rs  libadwaita")
	if err != nil || cmd != "setup" || spec != "spec" || out != "rust" || target != "use gtk4-rs  libadwaita" {
		t.Errorf("setup = (%q %q %q %q %v)", cmd, spec, out, target, err)
	}
	if cmd, _, _, _, _ := parseSpecArgs("  "); cmd != "resume" {
		t.Errorf("empty args = %q, want resume", cmd)
	}
	if cmd, _, _, _, _ := parseSpecArgs("status"); cmd != "status" {
		t.Errorf("status = %q", cmd)
	}
	if _, _, _, _, err := parseSpecArgs("spec/"); err == nil {
		t.Error("a spec dir without an output dir parsed")
	}
	if _, _, _, target, _ := parseSpecArgs("spec out"); target != "" {
		t.Errorf("target = %q, want empty (it is optional)", target)
	}
}

// TestDetectSpecTestCmd: a justfile test recipe wins, since that is where a
// project puts what the bare toolchain command doesn't know.
func TestDetectSpecTestCmd(t *testing.T) {
	dir := t.TempDir()
	if got := detectSpecTestCmd(dir); got != "" {
		t.Errorf("empty dir = %q, want none", got)
	}
	os.WriteFile(filepath.Join(dir, "Cargo.toml"), []byte("[package]\n"), 0o644)
	if got := detectSpecTestCmd(dir); got != "cargo test" {
		t.Errorf("cargo project = %q", got)
	}
	os.WriteFile(filepath.Join(dir, "justfile"), []byte("build:\n\tcargo build\n\ntest:\n\txvfb-run -a cargo test\n"), 0o644)
	if got := detectSpecTestCmd(dir); got != "just test" {
		t.Errorf("with a justfile test recipe = %q, want just test", got)
	}
}

func TestNextSpecItem(t *testing.T) {
	idx := &specIndex{order: []string{"a", "b", "c", "d"}}
	cfg := &specConfig{Blocked: []specBlock{{ID: "b", Reason: "stuck"}, {ID: "c", Reason: "asked", Answer: "use SQLite"}}}
	covered := map[string]string{"a": "tests/a.rs"}
	if id, ans := nextSpecItem(idx, covered, cfg); id != "c" || ans != "use SQLite" {
		t.Errorf("next = %q %q, want the answered block c before d", id, ans)
	}
	cfg.unblock("c")
	covered["c"] = "tests/c.rs"
	if id, _ := nextSpecItem(idx, covered, cfg); id != "d" {
		t.Errorf("next = %q, want d (b stays blocked without an answer)", id)
	}
	covered["d"] = "tests/d.rs"
	if id, _ := nextSpecItem(idx, covered, cfg); id != "" {
		t.Errorf("next = %q, want nothing left", id)
	}
}

func TestSpecOpenMarkers(t *testing.T) {
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	got := idx.openMarkers(defaultSpecOpenMarkers, "spec")
	// "REVIEW:" counts, "PREVIEWING" does not.
	if len(got) != 1 || !strings.HasPrefix(got[0], "spec/03-shell.md:") {
		t.Errorf("markers = %v, want exactly the one REVIEW line in 03-shell.md", got)
	}
}

func TestSpecConfigRoundTrip(t *testing.T) {
	cwd := t.TempDir()
	if cfg, err := loadSpecConfig(cwd); cfg != nil || err != nil {
		t.Fatalf("no file = (%v, %v), want (nil, nil)", cfg, err)
	}
	in := &specConfig{SpecDir: "spec", OutDir: "rust", Target: "use gtk4-rs", Attempts: map[string]int{"F0.1": 1},
		Blocked: []specBlock{{ID: "F0.2", Reason: "asked", Question: "keep it?"}}}
	if err := saveSpecConfig(cwd, in); err != nil {
		t.Fatal(err)
	}
	out, err := loadSpecConfig(cwd)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, out) {
		t.Errorf("round trip:\n got %+v\nwant %+v", out, in)
	}
}
