package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// Comments and strings go, line breaks stay; a test call's title is kept.
func TestCodeText(t *testing.T) {
	for _, tc := range []struct {
		rel, src      string
		mode          stringsMode
		want, notWant []string
	}{
		{"a.rs", "fn f() { // F1.1\n  let s = r#\"F1.2 \"quoted\"\"#; /* F1.3 */ let c = '\"'; }\n#[test]\nfn f1_4_ok<'a>(x: &'a str) {}\n",
			blankStrings, []string{"fn f()", "fn f1_4_ok"}, []string{"F1.1", "F1.2", "F1.3", "quoted"}},
		{"a.py", "def test_f2_1():\n    '''F2.2 docstring'''\n    x = \"F2.3\"  # F2.4\n", blankStrings,
			[]string{"test_f2_1"}, []string{"F2.2", "F2.3", "F2.4"}},
		{"a.test.ts", "describe(\"F3.1 file\", () => {\n  it('F3.2 opens', () => { expect(`F3.3`).toBe(\"F3.4\") })\n})\n", keepTitles,
			[]string{"F3.1", "F3.2"}, []string{"F3.3", "F3.4"}},
		{"a_test.go", "func TestX(t *testing.T) {\n\tt.Run(\"f4_1 case\", func(t *testing.T) { _ = \"F4.2\" })\n}\n", keepTitles,
			[]string{"f4_1"}, []string{"F4.2"}},
		{"a.js", "import { run } from './transcribe' // why\n", keepStrings, []string{"./transcribe"}, []string{"why"}},
		{"a.py", "r\"\"\"F5.1 raw docstring\"\"\"\ndef test_f5_2():\n    pass\n", blankStrings, []string{"test_f5_2"}, []string{"F5.1"}},
		{"a.rs", "fn f5_3(s: &'static str) -> char { let c = \"can't\"; 'x' }\nfn f5_4() {}\n", blankStrings, []string{"fn f5_3", "fn f5_4"}, []string{"can"}},
		{"a.js", "const re = /[/*]/g\nfunction f5_5() {}\n", blankStrings, []string{"function f5_5"}, nil},
	} {
		got := codeText(tc.rel, tc.src, tc.mode)
		if strings.Count(got, "\n") != strings.Count(tc.src, "\n") {
			t.Errorf("%s: %d lines, want %d", tc.rel, strings.Count(got, "\n"), strings.Count(tc.src, "\n"))
		}
		for _, w := range tc.want {
			if !strings.Contains(got, w) {
				t.Errorf("%s: lost %q:\n%s", tc.rel, w, got)
			}
		}
		for _, w := range tc.notWant {
			if strings.Contains(got, w) {
				t.Errorf("%s: kept %q:\n%s", tc.rel, w, got)
			}
		}
	}
}

// Go and Java name tests in camel case: TestF4_1Opens names f4_1, TestF4_10 does not.
func TestNamesItemCamelCase(t *testing.T) {
	for text, want := range map[string]bool{
		"func TestF4_1Opens(t *testing.T)":  true,
		"void f4_1OpensTheFile()":           true,
		"fn f4_1_opens()":                   true,
		"func TestF4_10Opens(t *testing.T)": false,
		"func LatestF4_1()":                 false,
	} {
		if got := namesItem(text, "F4.1"); got != want {
			t.Errorf("namesItem(%q) = %v, want %v", text, got, want)
		}
	}
}

func gitRepo(t *testing.T, files map[string]string) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("no git")
	}
	dir := t.TempDir()
	run := func(args ...string) {
		t.Helper()
		if out, err := exec.Command("git", append([]string{"-C", dir, "-c", "user.email=t@t", "-c", "user.name=t"}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v %s", args, err, out)
		}
	}
	run("init", "-q")
	writeTree(t, dir, files)
	run("add", "-A")
	run("commit", "-q", "-m", "base")
	return dir
}

func writeTree(t *testing.T, dir string, files map[string]string) {
	t.Helper()
	for rel, body := range files {
		p := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// The round's added lines and growth, per file, relative to the output directory.
func TestSpecRoundChanges(t *testing.T) {
	cwd := gitRepo(t, map[string]string{"out/src/a.rs": "one\ntwo\nthree\n", "README.md": "x\n"})
	writeTree(t, cwd, map[string]string{
		"out/src/a.rs": "one\nTWO\nthree\nfour\n",
		"out/src/b.rs": "new\nfile\n",
		"README.md":    "outside\n",
	})
	ch := specRoundChanges(t.Context(), cwd, "out", "HEAD")
	if !ch.ok {
		t.Fatal("no changes read")
	}
	if !slices.Equal(ch.added["src/a.rs"], []int{2, 4}) || ch.grown["src/a.rs"] != 1 {
		t.Errorf("a.rs added %v grown %d, want [2 4] and 1", ch.added["src/a.rs"], ch.grown["src/a.rs"])
	}
	if !slices.Equal(ch.added["src/b.rs"], []int{1, 2}) || ch.grown["src/b.rs"] != 2 {
		t.Errorf("untracked b.rs added %v grown %d", ch.added["src/b.rs"], ch.grown["src/b.rs"])
	}
	if _, ok := ch.added["README.md"]; ok || len(ch.added) != 2 {
		t.Errorf("files = %v, want only the output directory's", ch.files())
	}
	if got := specNamedIn(filepath.Join(cwd, "out"), "F1.1", ch.files()); got != "" {
		t.Errorf("no test in the round names F1.1, got %q", got)
	}
}

// A new free function must be called by the program: tests alone or nothing do
// not count. A qualified call, an import, an alias, a match arm, a sibling in a Go
// package, and a helper named for tests do; so does a changed signature of an old
// function, which is no new one.
func TestSpecUnreachable(t *testing.T) {
	base := map[string]string{
		"out/src/main.rs":       "mod transcribe;\nmod other;\n#[cfg(test)]\nmod tests;\nfn main() {\n    transcribe::start();\n    other::run();\n}\n",
		"out/src/other.rs":      "pub fn run() {}\n\npub fn clear() {}\n",
		"out/src/transcribe.rs": "pub fn start() {}\n\npub fn legacy_rows(a: u32) {}\n",
		"out/go/a.go":           "package app\n\nfunc Serve() { helper() }\n",
		"out/py/app.py":         "from py.store import load\n\ndef main():\n    load()\n",
		"out/web/app.js":        "import { draw } from './canvas'\ndraw()\n",
		"out/gopkg/other/o.go":  "package other\n\nfunc Run() {}\n",
		"out/gopkg/cmd/main.go": "package main\n\nfunc main() {}\n",
		"out/src/lone.rs":       "pub fn multi() {}\n",
	}
	cwd := gitRepo(t, base)
	writeTree(t, cwd, map[string]string{
		"out/src/transcribe.rs": "pub fn start() {}\n\npub fn legacy_rows(a: u32, b: u32) {}\n\npub fn run() {}\n\npub fn wired() {}\n\npub fn kept() -> bool { true }\n\npub fn rows_for_test() {}\n\n#[wasm_bindgen]\n#[allow(dead_code)]\npub fn exported() {}\n\npub fn arm() {}\n\npub fn aliased() {}\n\npub fn clear() {}\n",
		"out/src/main.rs":       "mod transcribe;\nmod other;\nuse crate::transcribe as tr;\n#[cfg(test)]\nmod tests;\nfn main() {\n    transcribe::start();\n    other::run();\n    transcribe::wired();\n    let _ = transcribe::kept();\n    match 1 { _ => transcribe::arm() }\n    tr::aliased();\n    tr::clear();\n    after();\n}\nfn after() {}\n",
		"out/tests/t.rs":        "#[test]\nfn f1_1_runs() { naivepost::transcribe::run(); naivepost::transcribe::legacy_rows(1, 2); }\n",
		"out/go/b.go":           "package app\n\nfunc helper() {}\n\nfunc Orphan() {}\n",
		"out/py/store.py":       "def load():\n    pass\n\n@app.route(\n    '/x',\n)\ndef route():\n    pass\n",
		"out/web/canvas.js":     "export function draw() {}\nexport const erase = () => {}\nexport default function Page() {}\n",
		"out/gopkg/render/r.go": "package render\n\nfunc Run() {}\n",
		"out/gopkg/cmd/main.go": "package main\n\nimport (\n\tr \"example.com/app/gopkg/render\"\n)\n\nfunc main() { r.Run() }\n",
		"out/src/tail.rs":       "#[cfg(test)]\nuse std::fmt;\n\npub fn multi() {}\n\npub fn entry() { deep(); }\n\nfn deep() {}\n\npub fn orphan_after() {}\n",
		"out/src/uses.rs":       "use crate::tail::{\n    entry,\n    multi,\n};\n\npub fn go() { entry(); multi(); }\n",
	})
	out := filepath.Join(cwd, "out")
	ch := specRoundChanges(t.Context(), cwd, "out", "HEAD")
	got := strings.Join(specUnreachable(out, ch, loadSpecProgram(out)), "\n")
	for _, want := range []string{"`run` (src/transcribe.rs:5): only tests call it",
		"`Orphan` (go/b.go:5): nothing calls it", "`erase` (web/canvas.js:2): nothing calls it",
		"`orphan_after` (src/tail.rs:10): nothing calls it"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	for _, fine := range []string{"`wired`", "`kept`", "`arm`", "`aliased`", "`clear`", "`Run`", "`multi`", "`entry`", "`deep`", "`after`", "`legacy_rows`", "`rows_for_test`",
		"`exported`", "`helper`", "`load`", "`route`", "`draw`", "`Page`", "`start`"} {
		if strings.Contains(got, fine) {
			t.Errorf("%s flagged, but the program calls it or it is exempt:\n%s", fine, got)
		}
	}
}

// The naivepost stand-ins, and the same shapes in Go, Python and JS; old lines,
// comments, tests and the for-test helpers themselves are not the round's stand-ins.
func TestSpecStandIns(t *testing.T) {
	cwd := gitRepo(t, map[string]string{
		"out/src/upload.rs": "pub fn old() -> Result<(), String> { Err(\"not implemented\".into()) }\n",
	})
	writeTree(t, cwd, map[string]string{
		"out/src/upload.rs": "pub fn old() -> Result<(), String> { Err(\"not implemented\".into()) }\n\n" +
			"pub fn reply_for_test() -> Option<String> { SCRIPT.with(|c| c.borrow().clone()) }\n\n" +
			"pub fn scripted_or_live_for_test() -> Option<String> { reply_for_test() }\n\n" +
			"// a refusal is reachable, so this is exercised rather than stubbed out\n" +
			"pub fn scripted_ask() -> Box<AskModel> {\n    match reply_for_test() {\n        Some(s) => Box::new(move |_| Ok(s.clone())),\n        None => Box::new(|_| Err(\"no llm server here\".to_string())),\n    }\n}\n\n" +
			"pub fn speak() { todo!() }\n\npub fn set_placeholder() { entry.set_placeholder_text(Some(\"Search\")); }\n\n" +
			"#[cfg(test)]\nmod tests {\n    fn t() { super::reply_for_test(); todo!() }\n}\n",
		"out/go/svc.go":   "package svc\n\nfunc Ask() error { return errors.New(\"not yet implemented\") }\n\nfunc Live() string { return clientForTest().Get() }\n",
		"out/py/tts.py":   "def speak(line):\n    raise NotImplementedError\n",
		"out/kt/Tts.kt":   "fun speak(line: String): Unit = TODO(\"speech\")\n",
		"out/web/draw.js": "export function draw() { return 'stub' }\n",
		"out/tests/t.rs":  "#[test]\nfn f1_1() { naivepost::upload::reply_for_test(); unimplemented!() }\n",
	})
	ch := specRoundChanges(t.Context(), cwd, "out", "HEAD")
	got := strings.Join(specStandIns(filepath.Join(cwd, "out"), ch), "\n")
	for _, want := range []string{"`reply_for_test` (src/upload.rs:9): the program asks a helper that exists for tests",
		"src/upload.rs:15: `pub fn speak() { todo!() }`", "go/svc.go:3: `func Ask() error { return errors.New(\"not yet implemented\") }`",
		"`clientForTest` (go/svc.go:5)", "py/tts.py:2: `raise NotImplementedError`", "web/draw.js:1:", "kt/Tts.kt:1:"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	for _, fine := range []string{"upload.rs:1:", "upload.rs:3", "upload.rs:5", "upload.rs:7", "upload.rs:17", "upload.rs:21", "tests/t.rs"} {
		if strings.Contains(got, fine) {
			t.Errorf("%s flagged (an old line, a helper, a comment, a placeholder text or a test):\n%s", fine, got)
		}
	}
}

// A file over the budget may grow by the slack, not more; one under it is free.
func TestSpecOversize(t *testing.T) {
	cwd := gitRepo(t, map[string]string{"out/src/big.rs": strings.Repeat("x\n", 30), "out/src/small.rs": "x\n"})
	writeTree(t, cwd, map[string]string{
		"out/src/big.rs":   strings.Repeat("x\n", 30+specFileGrowthSlack+1),
		"out/src/small.rs": strings.Repeat("x\n", 15),
	})
	ch := specRoundChanges(t.Context(), cwd, "out", "HEAD")
	got := specOversize(filepath.Join(cwd, "out"), &specConfig{MaxFileLines: 20}, ch)
	if len(got) != 1 || !strings.Contains(got[0], "`src/big.rs` is 51 lines, over the 20-line budget, and this round grew it by 21") {
		t.Errorf("oversize = %v", got)
	}
	if got := specOversize(filepath.Join(cwd, "out"), &specConfig{MaxFileLines: -1}, ch); got != nil {
		t.Errorf("a budget turned off still reports %v", got)
	}
}

// Only findings in lines the round wrote count; a missing linter is no finding.
func TestSpecLint(t *testing.T) {
	out := t.TempDir()
	ch := specChanges{added: map[string][]int{"src/a.rs": {3}, "b.go": {9}}, ok: true}
	cmd := `printf 'warning: unused variable x\n  --> src/a.rs:3:9\nwarning: old\n  --> src/a.rs:40:1\n./b.go:9:2: printf format %%d has arg of wrong type\n'; exit 1`
	findings, missing := specLint(t.Context(), out, cmd, ch)
	if missing != "" || len(findings) != 2 {
		t.Fatalf("missing=%q findings=%q, want the two in written lines", missing, findings)
	}
	if !strings.Contains(findings[0], "unused variable x") || !strings.Contains(findings[1], "b.go:9:2: printf format") {
		t.Errorf("findings = %q", findings)
	}
	// eslint's default format and tsc's.
	ch.added["web/a.ts"] = []int{4, 7}
	eslint := `printf '\n%s/web/a.ts\n  4:3  error  Unexpected any  no-explicit-any\n  9:1  warning  old  no-console\n\nweb/a.ts(7,2): error TS2322: bad type\n'`
	findings, _ = specLint(t.Context(), out, fmt.Sprintf(eslint, out), ch)
	if len(findings) != 2 || !strings.Contains(findings[0], "web/a.ts:4: 4:3  error  Unexpected any") || !strings.Contains(findings[1], "TS2322") {
		t.Errorf("eslint and tsc findings = %q", findings)
	}
	// A header with parentheses (a route group) is still a header; a finding under
	// it never lands on the file before.
	ch.added["lib/cart.ts"] = []int{2, 4}
	ch.added["app/(shop)/cart/page.tsx"] = []int{2}
	groups := `printf '%s/lib/cart.ts\n  9:1  error  x  r\n\n%s/app/(shop)/cart/page.tsx\n  2:7  error  Unexpected any  no-explicit-any\n\n%s/README.md\n  4:1  error  y  r\n'`
	findings, _ = specLint(t.Context(), out, fmt.Sprintf(groups, out, out, out), ch)
	if len(findings) != 1 || !strings.Contains(findings[0], "app/(shop)/cart/page.tsx:2") {
		t.Errorf("route-group findings = %q, want only page.tsx:2", findings)
	}
	if _, missing := specLint(t.Context(), out, "no-such-linter-xyz --check", ch); !strings.Contains(missing, "no-such-linter-xyz") {
		t.Errorf("a missing linter: missing=%q, want the shell's own line naming it", missing)
	}
}

// The refactor measure counts lines over budget, dead functions and copied test
// helpers; better needs one down and none up.
func TestMeasureSpecDebt(t *testing.T) {
	out := t.TempDir()
	writeTree(t, out, map[string]string{
		"src/main.rs":   "mod ui;\nfn main() { ui::show(); }\n",
		"src/ui.rs":     "pub fn show() {}\n" + strings.Repeat("// filler\n", 30) + "pub fn unused() {}\n",
		"tests/a.rs":    "fn fixture_dir() {}\n#[test]\nfn f1_1() { fixture_dir(); }\n",
		"tests/b.rs":    "fn fixture_dir() {}\n#[test]\nfn f1_2() { fixture_dir(); }\n",
		"tests/c.rs":    "fn fixture_dir() {}\n#[test]\nfn f1_3() { fixture_dir(); }\n",
		"examples/x.rs": "fn main() { naive::ui::show(); }\n",
	})
	cfg := &specConfig{MaxFileLines: 20}
	d := measureSpecDebt(out, cfg)
	if d.overLines != 12 || d.dead != 1 || d.copies != 2 {
		t.Fatalf("debt = %+v, want 12 over, 1 dead, 2 copies", d)
	}
	for _, want := range []string{"`src/ui.rs`: 32 lines", "`unused` (src/ui.rs:32)", "`fixture_dir`: 3 files"} {
		if !strings.Contains(d.targets, want) {
			t.Errorf("targets lack %q:\n%s", want, d.targets)
		}
	}
	if !(specDebt{overLines: 12, dead: 0, copies: 2}).better(d) || (specDebt{overLines: 13, dead: 0, copies: 0}).better(d) || d.better(d) {
		t.Error("better must need one measure down and none up")
	}
}
