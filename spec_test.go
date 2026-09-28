package main

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// writeSpecFixture has nav bars, heading flows, id-less prose sections, a parameter table, an
// index, an inventory repeating a flow id, and a heading naming an id only in parentheses.
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

// What becomes an item, where each is defined, and the order the loop takes them in.
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
	// A heading naming an id in parentheses beats the index row when no heading opens with it.
	if rel, kind := where("F6.1"); rel != "09-llm.md" || kind != specDefHeading {
		t.Errorf("F6.1 defined at %s kind %d, want 09's heading over the index row", rel, kind)
	}
	// A heading naming several ids moves neither; the parameter stays defined by its table row.
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

// Own section without nav, cited parameter rows, the linked section, only a pointer to another flow.
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
	// An id in a production doc comment is not a test; the inline test module is.
	write("src/lib.rs", "//! F0.2 lives here\npub fn press() {}\n#[cfg(test)]\nmod tests {\n    #[test] fn p_policy_padseconds_holds() {}\n}\n")
	// A comment or a string inside a test does not name it: 130 items passed that way.
	write("tests/said.rs", "// F1.1 is covered here\n#[test]\nfn checks() { assert_eq!(ITEM, \"F1.2\"); let s = r#\"F1.3\"#; }\n")
	write("web/a.test.js", "it(\"F1.4 opens the file\", () => {});\nconst note = \"F1.5\";\n")
	write("src/cut.rs", "// §01-files#2-cutjson\npub fn load() {}\n")
	write("tests/files.rs", "#[test] fn sec_01_files_2_cutjson_round_trips() {}\n")
	write("target/debug/stale.rs", "#[test] fn f0_2_would_count() {}\n") // build output: never scanned

	ids := []string{"F0.1", "F0.2", "F0.10", "P.policy.padSeconds", "§01-files#2-cutjson", "F1.0", "F1.1", "F1.2", "F1.3", "F1.4", "F1.5"}
	covered, files, err := specCoverage(out, ids)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"F0.1", "F0.10", "P.policy.padSeconds", "§01-files#2-cutjson", "F1.4"} {
		if _, ok := covered[id]; !ok {
			t.Errorf("%s not covered, want it covered", id)
		}
	}
	for _, id := range []string{"F0.2", "F1.0", "F1.1", "F1.2", "F1.3", "F1.5"} {
		if f, ok := covered[id]; ok {
			t.Errorf("%s covered by %s, want uncovered", id, f)
		}
	}
	if files != 5 {
		t.Errorf("test sources = %d, want 5 (tests/shell.rs, tests/files.rs, tests/said.rs, web/a.test.js, src/lib.rs)", files)
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
	for args, want := range map[string]string{"": "resume", "  ": "resume", "status": "status", " stop ": "stop", "redo": "redo"} {
		if cmd, err := parseSpecArgs(args); err != nil || cmd != want {
			t.Errorf("parseSpecArgs(%q) = %q, %v; want %q", args, cmd, err, want)
		}
	}
	// Item ids and the positional spec-dir/out-dir form are refused.
	for _, args := range []string{"redo F2.3", "redo 03-shell.md", "spec/ rust/ use gtk4-rs"} {
		if _, err := parseSpecArgs(args); err == nil {
			t.Errorf("%q parsed", args)
		}
	}
}

// Reopened items stay open though their tests name them, by id or file; an unknown target reopens nothing.
func TestSpecRedoReopens(t *testing.T) {
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &specConfig{SpecDir: "spec", Items: map[string]specLedger{}}
	covered := map[string]string{}
	for _, id := range idx.order {
		cfg.Items[id] = specLedger{Hash: specItemHash(idx, id)}
		covered[id] = "tests/all.rs"
	}
	byFile := idx.docs[idx.items[idx.order[0]].Doc].rel
	ids, unknown := specRedoTargets(cfg, idx, []string{"spec/" + byFile, idx.order[len(idx.order)-1]})
	if len(unknown) != 0 || len(ids) < 2 {
		t.Fatalf("targets = %v unknown = %v", ids, unknown)
	}
	if _, unknown := specRedoTargets(cfg, idx, []string{"F9.9", byFile}); len(unknown) != 1 || unknown[0] != "F9.9" {
		t.Errorf("unknown = %v, want the typo alone", unknown)
	}
	// A paraphrased slug resolves by its number; a number the file lacks does not.
	var section string
	for _, id := range idx.order {
		if strings.HasPrefix(id, "§") && strings.Contains(id, "#") {
			section = id
			break
		}
	}
	if section != "" {
		stem := section[:strings.Index(section, "#")+1]
		num := strings.SplitN(section[len(stem):], "-", 2)[0]
		got, unknown := specRedoTargets(cfg, idx, []string{stem + num + "-something-else"})
		if len(unknown) != 0 || len(got) != 1 || got[0] != section {
			t.Errorf("paraphrased section = %v %v, want %q", got, unknown, section)
		}
		if _, unknown := specRedoTargets(cfg, idx, []string{stem + "99-nothing"}); len(unknown) != 1 {
			t.Errorf("a section number the file lacks resolved: %v", unknown)
		}
	}

	cfg.reopen(ids)
	for _, id := range ids {
		if _, known := cfg.Items[id]; known {
			t.Errorf("%s still in the ledger", id)
		}
	}
	if d := specReconcile(cfg, idx, covered); d.Adopted != 0 {
		t.Errorf("adopted %d reopened items back", d.Adopted)
	}
	if next := nextSpecItem(idx, cfg, nil); next != ids[0] {
		t.Errorf("next = %q, want the first reopened item %q", next, ids[0])
	}
	if cfg.Redo[ids[0]] == "" {
		t.Error("the redo mark did not stick")
	}
	// Status counts a reopened item as open although its test still names it.
	status := renderSpecStatus(cfg, idx, 1, "just test")
	want := fmt.Sprintf("Covered: %d/%d flows", countKind(idx, specDefHeading)-countKindIn(idx, ids, specDefHeading), countKind(idx, specDefHeading))
	if !strings.Contains(status, want) {
		t.Errorf("status lacks %q:\n%s", want, status)
	}
}

func countKind(idx *specIndex, kind int) int {
	n := 0
	for _, id := range idx.order {
		if idx.items[id].Kind == kind {
			n++
		}
	}
	return n
}

func countKindIn(idx *specIndex, ids []string, kind int) int {
	n := 0
	for _, id := range ids {
		if idx.items[id].Kind == kind {
			n++
		}
	}
	return n
}

// A justfile test recipe wins over the bare toolchain command.
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

// Done is the ledger's word; an item waits while its question has no answer.
func TestNextSpecItem(t *testing.T) {
	idx := &specIndex{order: []string{"a", "b", "c", "d"}, questions: map[string][]specQuestion{
		"b": {{ID: "b", Question: "which store?"}},
		"c": {{ID: "c", Question: "which db?", Answer: "use SQLite"}},
	}}
	cfg := &specConfig{Items: map[string]specLedger{"a": {}}}
	if id := nextSpecItem(idx, cfg, nil); id != "c" {
		t.Errorf("next = %q, want c: b waits on its question, c is answered", id)
	}
	if id := nextSpecItem(idx, cfg, func(id string) bool { return id == "c" }); id != "d" {
		t.Errorf("next = %q, want d past c blocked in this run", id)
	}
	cfg.Items["c"], cfg.Items["d"] = specLedger{}, specLedger{}
	if id := nextSpecItem(idx, cfg, nil); id != "" {
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

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.md"), []byte("# A\n\nTODO: decide\nsee [open] here\nxTODO: not a marker\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if idx, err = scanSpec(dir, defaultSpecIDPatterns, nil, nil); err != nil {
		t.Fatal(err)
	}
	// Markers may start or end with punctuation.
	if got := idx.openMarkers([]string{"TODO:", "[open]"}, "spec"); !slices.Equal(got, []string{"spec/a.md:3", "spec/a.md:4"}) {
		t.Errorf("punctuated markers = %v, want lines 3 and 4", got)
	}
}

func TestSpecConfigRoundTrip(t *testing.T) {
	cwd := t.TempDir()
	if cfg, err := loadSpecConfig(cwd); cfg != nil || err != nil {
		t.Fatalf("no file = (%v, %v), want (nil, nil)", cfg, err)
	}
	in := &specConfig{SpecDir: "spec", OutDir: "rust", Target: "use gtk4-rs", Attempts: map[string]int{"F0.1": 1},
		LintCmd: "off", MaxFileLines: 2000, RefactorEvery: -1, RefactorAt: 12}
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
	// A ledger from before blocks lasted one run still loads; saving drops its blocks.
	old, _ := os.ReadFile(specConfigPath(cwd))
	old = append(old, "\n[[blocked]]\n  id = \"F0.2\"\n  reason = \"stuck\"\n  answer = \"\"\n"...)
	if err := os.WriteFile(specConfigPath(cwd), old, 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err = loadSpecConfig(cwd); err != nil || !reflect.DeepEqual(in, out) {
		t.Errorf("old ledger = %+v, %v; want it read as %+v", out, err, in)
	}
}

// Rewording must still re-open an item.
func TestSpecItemHashIgnoresFormatting(t *testing.T) {
	root := writeSpecFixture(t)
	idx, err := scanSpec(root, defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	const id = "§01-files#1-layout"
	before := specItemHash(idx, id)
	if before == "" {
		t.Fatalf("no hash for %s; items: %v", id, idx.order)
	}

	path := filepath.Join(root, "01-files.md")
	body, _ := os.ReadFile(path)
	reflowed := strings.Replace(string(body), "The project folder.", "  The project\n  folder.  ", 1)
	if err := os.WriteFile(path, []byte(reflowed), 0o644); err != nil {
		t.Fatal(err)
	}
	idx2, err := scanSpec(root, defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := specItemHash(idx2, id); got != before {
		t.Errorf("reformatting changed the hash: %s -> %s", before, got)
	}

	reworded := strings.Replace(string(body), "The project folder.", "The project folder, now with a lock file.", 1)
	if err := os.WriteFile(path, []byte(reworded), 0o644); err != nil {
		t.Fatal(err)
	}
	idx3, err := scanSpec(root, defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := specItemHash(idx3, id); got == before {
		t.Error("rewording the section left the hash unchanged")
	}
}

// A rename carries its record to the new id; covered work predating the ledger is adopted, not changed.
func TestSpecReconcile(t *testing.T) {
	root := writeSpecFixture(t)
	idx, err := scanSpec(root, defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	const (
		unchanged = "§01-files#1-layout"
		edited    = "§01-files#2-cutjson"
	)
	cfg := &specConfig{Items: map[string]specLedger{
		unchanged: {Hash: specItemHash(idx, unchanged), Title: idx.items[unchanged].Title},
		edited:    {Hash: "stale", Title: idx.items[edited].Title},
		// Gone from the spec entirely.
		"§99-old#1-dropped": {Hash: "whatever", Title: "1. Dropped"},
		// Same heading, different file: a section that moved.
		"§98-moved#1-screen": {Hash: "moved", Title: idx.items["§03-shell#1-screen"].Title},
	}}
	// A parameter row a test is named after, absent from the ledger: adopted, not
	// reported as changed forever. A section likewise named: done only by a round.
	adoptable, roundOnly := "", ""
	for _, id := range idx.order {
		if _, known := cfg.Items[id]; known || id == "§03-shell#1-screen" {
			continue
		}
		switch {
		case idx.items[id].Kind == specDefTableRow && adoptable == "":
			adoptable = id
		case idx.items[id].Kind != specDefTableRow && roundOnly == "":
			roundOnly = id
		}
	}
	if adoptable == "" || roundOnly == "" {
		t.Fatalf("fixture has no spare row or section; order: %v", idx.order)
	}
	covered := map[string]string{adoptable: "tests/flows.rs", roundOnly: "tests/flows.rs",
		unchanged: "tests/a.rs", edited: "tests/a.rs", "§03-shell#1-screen": "tests/a.rs"}

	d := specReconcile(cfg, idx, covered)
	if want := []string{edited}; !reflect.DeepEqual(d.Changed, want) {
		t.Errorf("Changed = %v, want %v", d.Changed, want)
	}
	if want := []string{"§99-old#1-dropped"}; !reflect.DeepEqual(d.Removed, want) {
		t.Errorf("Removed = %v, want %v", d.Removed, want)
	}
	if len(d.Renamed) != 1 || !strings.Contains(d.Renamed[0], "§03-shell#1-screen") {
		t.Errorf("Renamed = %v, want the moved section", d.Renamed)
	}
	if _, ok := cfg.Items["§98-moved#1-screen"]; ok {
		t.Error("the old id must not stay in the ledger after a rename")
	}
	if led, ok := cfg.Items["§03-shell#1-screen"]; !ok || led.Hash != specItemHash(idx, "§03-shell#1-screen") {
		t.Errorf("the renamed item must be recorded at its new text, got %+v", led)
	}
	if d.Adopted != 1 || cfg.Items[adoptable].CoveredBy != "tests/flows.rs" {
		t.Errorf("a named parameter row predating the ledger should be adopted, got %d and %+v", d.Adopted, cfg.Items[adoptable])
	}
	if _, ok := cfg.Items[roundOnly]; ok {
		t.Errorf("%s was adopted without a round; only parameter and tool rows are", roundOnly)
	}
	if e := cfg.Items[adoptable]; e.At.IsZero() || e.Version == "" || e.Commit != "" {
		t.Errorf("an adopted entry must say when and by which codehalter it was recorded, and carry no commit: %+v", e)
	}
	// A second pass adopts and renames nothing; the redo and the delete stay reported until acted on.
	d2 := specReconcile(cfg, idx, covered)
	if d2.Adopted != 0 || len(d2.Renamed) != 0 {
		t.Errorf("a second pass must be quiet about renames and adoptions, got %+v", d2)
	}
	if !reflect.DeepEqual(d2.Changed, d.Changed) || !reflect.DeepEqual(d2.Removed, d.Removed) {
		t.Errorf("unacted work must still be reported: %+v then %+v", d, d2)
	}
}

// A done item no test is named after any more is open again, unless most of the
// ledger would reopen at once, which reads as a scan problem. An entry from before
// the test-name rule stays done and is reported.
func TestSpecReconcileReopensUnnamedItems(t *testing.T) {
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	ledger := func() *specConfig {
		cfg := &specConfig{Items: map[string]specLedger{}}
		for _, id := range idx.order[:4] {
			cfg.Items[id] = specLedger{Hash: specItemHash(idx, id), Title: idx.items[id].Title, Named: true}
		}
		return cfg
	}
	named := map[string]string{}
	for _, id := range idx.order[:3] {
		named[id] = "tests/a.rs"
	}
	cfg := ledger()
	d := specReconcile(cfg, idx, named)
	if gone := idx.order[3]; !reflect.DeepEqual(d.Reopened, []string{gone}) || cfg.done(gone) {
		t.Errorf("Reopened = %v, want %s open again", d.Reopened, gone)
	}
	if !strings.Contains(renderSpecDelta(d), "open again") {
		t.Errorf("the reopen is not reported:\n%s", renderSpecDelta(d))
	}

	cfg = ledger()
	d = specReconcile(cfg, idx, map[string]string{idx.order[0]: "tests/a.rs"})
	if d.Unnamed != 3 || len(d.Reopened) != 0 || len(cfg.Items) != 4 {
		t.Errorf("three of four unnamed: Unnamed=%d Reopened=%v items=%d, want none reopened", d.Unnamed, d.Reopened, len(cfg.Items))
	}

	cfg = ledger()
	legacy := idx.order[3]
	cfg.Items[legacy] = specLedger{Hash: specItemHash(idx, legacy), Title: idx.items[legacy].Title}
	d = specReconcile(cfg, idx, named)
	if !cfg.done(legacy) || !reflect.DeepEqual(d.Legacy, []string{legacy}) || len(d.Reopened) != 0 {
		t.Errorf("an entry from before the rule: done=%v Legacy=%v Reopened=%v, want it kept and reported", cfg.done(legacy), d.Legacy, d.Reopened)
	}
	if !strings.Contains(renderSpecDelta(d), "/spec redo") {
		t.Errorf("the kept entries are not reported:\n%s", renderSpecDelta(d))
	}
}

// Which directory holds the spec, which page is read first, and what stack it asks for.
func TestSpecSetupDetection(t *testing.T) {
	root := t.TempDir()
	for path, body := range map[string]string{
		"notes/a.md":     "# a",
		"notes/b.md":     "# b",
		"spec/README.md": "# The spec\n\nA desktop editor. The rewrite targets Rust with gtk4-rs and libadwaita.\n",
		"spec/01.md":     "# 01",
		"spec/02.md":     "# 02",
		"target/x.md":    "# ignored build output",
		"target/y.md":    "# ignored build output",
	} {
		full := filepath.Join(root, path)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cands := specDirCandidates(root)
	if len(cands) == 0 || cands[0] != "spec" {
		t.Errorf("candidates = %v, want spec first (it is named like a spec)", cands)
	}
	for _, c := range cands {
		if c == "target" {
			t.Error("build output must not be offered as a spec directory")
		}
		if strings.HasPrefix(c, "spec/") {
			t.Errorf("%q is part of the spec already offered, not a rival to it", c)
		}
	}
	rel, entry := specEntryPage(filepath.Join(root, "spec"))
	if rel != "README.md" || !strings.Contains(entry, "desktop editor") {
		t.Errorf("entry page = %q (%d bytes), want README.md", rel, len(entry))
	}
	out, target := specGuessTarget(entry)
	if out != "rust" || !strings.Contains(target, "gtk4-rs") {
		t.Errorf("guess = %q / %q, want rust and gtk4-rs", out, target)
	}
	if out, target := specGuessTarget("# a spec with no stack in it\n"); out != "" || target != "" {
		t.Errorf("a page naming no stack must guess nothing, got %q / %q", out, target)
	}
}

// The section stops at the next heading of its level or above.
func TestSpecSectionFromText(t *testing.T) {
	doc := "# 01 Files\n\n## 1. Layout\n\nThe folder.\n\n### 1.1 Detail\n\nMore.\n\n## 2. cut.json\n\nThe format.\n"
	got := specSectionFromText(doc, "§01-files#1-layout")
	if !strings.Contains(got, "The folder.") || !strings.Contains(got, "1.1 Detail") {
		t.Errorf("section should carry its subsections, got %q", got)
	}
	if strings.Contains(got, "cut.json") {
		t.Errorf("section must stop at the next heading of its level, got %q", got)
	}
	if got := specSectionFromText(doc, "§01-files#nope"); got != "" {
		t.Errorf("an unknown slug should find nothing, got %q", got)
	}
}

// Out-dir: the page's suggestion, then manifest dirs (never the spec dir or build output).
// Target: the page's answer, then what the chosen directory's manifest says.
func TestSpecSetupOptions(t *testing.T) {
	root := t.TempDir()
	for path, body := range map[string]string{
		"spec/README.md":  "# The spec\n\nThe rewrite targets Rust with gtk4-rs and libadwaita.\n",
		"spec/01.md":      "# 01",
		"rust/Cargo.toml": "[package]\nname = \"x\"\n\n[dependencies]\ngtk4 = \"0.11\"\nlibadwaita = \"0.9\"\nserde_json = \"1\"\ncairo-rs = \"0.20\"\n\n[dev-dependencies]\ntempfile = \"3\"\n",
		"gui/go.mod":      "module example.com/gui\n\ngo 1.26\n\nrequire (\n\tgithub.com/diamondburned/gotk4 v0.4.1\n\tgithub.com/coder/websocket v1.8.0 // indirect\n)\n",
		"target/x.rs":     "",
		"notes.txt":       "",
	} {
		full := filepath.Join(root, path)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if lang, deps := manifestStack(filepath.Join(root, "rust")); lang != "rust" || strings.Join(deps, ",") != "gtk4,libadwaita,serde_json" {
		t.Errorf("rust manifest = %q %v, want rust and the first three [dependencies]", lang, deps)
	}
	if lang, deps := manifestStack(filepath.Join(root, "gui")); lang != "go" || strings.Join(deps, ",") != "gotk4" {
		t.Errorf("go manifest = %q %v, want go and gotk4 (indirect skipped)", lang, deps)
	}
	if lang, _ := manifestStack(filepath.Join(root, "spec")); lang != "" {
		t.Errorf("a directory with no manifest reports %q", lang)
	}

	outs := specOutDirOptions(root, "spec", "rust")
	if strings.Join(outs, ",") != "rust,gui" {
		t.Errorf("out-dir options = %v, want the suggestion first, then the other project, never spec or target", outs)
	}
	if outs := specOutDirOptions(root, "spec", "app"); outs[0] != "app" || len(outs) != 3 {
		t.Errorf("a suggested new directory must lead: %v", outs)
	}

	entry := "The rewrite targets Rust with gtk4-rs and libadwaita."
	got := specTargetOptions(root, "rust", "rust with gtk4-rs (v4_14) and libadwaita", entry)
	want := []string{"rust with gtk4-rs (v4_14) and libadwaita", "rust with gtk4, libadwaita, serde_json", "rust with gtk4-rs and libadwaita"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("target options = %v, want %v", got, want)
	}
	if got := specTargetOptions(root, "gui", "", entry); got[0] != "go with gotk4" {
		t.Errorf("with no page suggestion the directory's own stack leads: %v", got)
	}
}

// After a failed round the test exists but the suite did not pass, so it must not be adopted.
func TestSpecFailedRoundStaysOpen(t *testing.T) {
	root := writeSpecFixture(t)
	idx, err := scanSpec(root, defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	// A parameter row: the one kind a named test adopts without a round.
	item := ""
	for _, id := range idx.order {
		if idx.items[id].Kind == specDefTableRow {
			item = id
			break
		}
	}
	covered := map[string]string{item: "tests/x.rs"}
	cfg := &specConfig{Attempts: map[string]int{item: 1}}

	if d := specReconcile(cfg, idx, covered); d.Adopted != 0 {
		t.Errorf("adopted %d, want 0: the item's last round failed", d.Adopted)
	}
	if _, known := cfg.Items[item]; known {
		t.Error("a failed item landed in the ledger")
	}
	if cfg.done(item) {
		t.Errorf("%s counts as done after a failed round", item)
	}

	// A ledgered item is done whatever a stale attempt count says, and reconciling drops it.
	cfg.Items[item] = specLedger{Hash: specItemHash(idx, item)}
	if !cfg.done(item) {
		t.Error("a ledgered item is not done because of a stale attempt count")
	}
	if !strings.Contains(renderSpecStatus(cfg, idx, 1, "just test"), "| "+idx.docs[idx.items[item].Doc].rel+" | 1 |") {
		t.Error("status does not count a ledgered item with a stale attempt count as covered")
	}
	specReconcile(cfg, idx, covered)
	if cfg.Attempts[item] != 0 {
		t.Errorf("attempts = %d after reconcile, want the stale count dropped", cfg.Attempts[item])
	}
	// A changed item keeps its count: its change rounds are counted there.
	cfg.Items[item] = specLedger{Hash: "stale"}
	cfg.Attempts[item] = 1
	specReconcile(cfg, idx, covered)
	if cfg.Attempts[item] != 1 {
		t.Error("reconcile dropped the attempt count of a changed item")
	}
	delete(cfg.Items, item)

	delete(cfg.Attempts, item) // the item passed
	if d := specReconcile(cfg, idx, covered); d.Adopted != 1 {
		t.Errorf("adopted %d, want 1 once no failed round stands", d.Adopted)
	}
	if id := nextSpecItem(idx, cfg, nil); id == item {
		t.Error("a covered item with no failed round was picked again")
	}
}

// An existing file is never overwritten and a path that leaves the directory is refused.
func TestWriteSpecFiles(t *testing.T) {
	cwd := t.TempDir()
	written, err := writeSpecFiles(cwd, "spec", []specFile{{Path: "01-files.md", Content: "# Files\n\n## 1. Layout\n\nF1.1 the layout"}, {Path: "spec/02-ui.md", Content: "# UI"}})
	if err != nil || strings.Join(written, ",") != "spec/01-files.md,spec/02-ui.md" {
		t.Fatalf("written = %v, err = %v", written, err)
	}
	if b, _ := os.ReadFile(filepath.Join(cwd, "spec", "02-ui.md")); string(b) != "# UI\n" {
		t.Errorf("02-ui.md = %q", b)
	}
	if _, err := writeSpecFiles(cwd, "spec", []specFile{{Path: "01-files.md", Content: "again"}}); err == nil {
		t.Error("an existing spec file was overwritten")
	}
	if _, err := writeSpecFiles(cwd, "spec", []specFile{{Path: "../evil.md", Content: "x"}}); err == nil {
		t.Error("a path outside the spec dir was written")
	}
}

// More than half of the ledger gone at once, or all of a ledger under 4, reads as a missing spec.
func TestSpecReconcileVanished(t *testing.T) {
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		live, gone int
		want       bool
	}{
		{3, 2, false}, {2, 2, false}, {2, 3, true}, // half is not more than half
		{1, 2, false}, {0, 3, true}, {0, 1, true}, // under 4, only all of it
	} {
		cfg := &specConfig{Items: map[string]specLedger{}}
		for _, id := range idx.order[:tc.live] {
			cfg.Items[id] = specLedger{Hash: specItemHash(idx, id)}
		}
		for i := range tc.gone {
			cfg.Items[fmt.Sprintf("§99-old#%d-dropped", i)] = specLedger{Hash: "gone"}
		}
		if d := specReconcile(cfg, idx, nil); d.Vanished != tc.want || len(d.Removed) != tc.gone {
			t.Errorf("%d live, %d gone: vanished=%v removed=%v, want vanished=%v", tc.live, tc.gone, d.Vanished, d.Removed, tc.want)
		}
	}
}
