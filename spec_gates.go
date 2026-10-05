package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
)

// The /spec gates work for any language: comment syntax and the shape of a
// definition come from the file extension, the round's changes from git, and
// every other check is plain text.

// commentMarks: an unknown language keeps its comments, which can only make a
// gate more lenient.
func commentMarks(ext string) (line []string, open, close string) {
	switch ext {
	case ".rs", ".go", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".kt", ".kts", ".scala",
		".swift", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cs", ".dart", ".zig", ".groovy", ".vue", ".svelte":
		return []string{"//"}, "/*", "*/"
	case ".php":
		return []string{"//", "#"}, "/*", "*/"
	case ".py", ".rb", ".sh", ".bash", ".pl", ".r", ".ex", ".exs", ".jl", ".nim", ".cr":
		return []string{"#"}, "", ""
	case ".lua", ".sql", ".hs", ".elm":
		return []string{"--"}, "", ""
	}
	return nil, "", ""
}

// The title of a test call: it("..."), test("..."), describe "...", t.Run("...").
var testTitleRe = regexp.MustCompile(`(?:\b(?:test|it|describe|context|specify|scenario|feature|suite|should)|\.Run|\btest\.\w+)\s*\(?\s*$`)

type stringsMode int

const (
	blankStrings stringsMode = iota
	keepTitles               // a test call's title string stays
	keepStrings              // import paths are strings in JS and friends
)

// codeText drops comments and, by mode, string literals, keeping every line
// break so line numbers hold.
func codeText(rel, src string, mode stringsMode) string {
	ext := strings.ToLower(filepath.Ext(rel))
	lineMarks, open, close := commentMarks(ext)
	out := make([]byte, 0, len(src))
	for i := 0; i < len(src); {
		rest := src[i:]
		if open != "" && strings.HasPrefix(rest, open) {
			n := len(rest)
			if j := strings.Index(rest[len(open):], close); j >= 0 {
				n = len(open) + j + len(close)
			}
			out = append(out, strings.Repeat("\n", strings.Count(rest[:n], "\n"))...)
			i += n
			continue
		}
		if slices.ContainsFunc(lineMarks, func(m string) bool { return strings.HasPrefix(rest, m) }) {
			j := strings.IndexByte(rest, '\n')
			if j < 0 {
				break
			}
			i += j
			continue
		}
		c := src[i]
		end, from, to := -1, 0, 0
		switch {
		case c == '"' || c == '\'' || c == '`':
			end, from, to = literalEnd(src, i, out, ext == ".rs")
		case c == '/' && jsExts[ext]:
			end, from, to = regexLiteralEnd(src, i, out)
		}
		if end < 0 {
			out = append(out, c)
			i++
			continue
		}
		lit := src[i:end]
		switch {
		case mode == keepStrings, mode == keepTitles && testTitleRe.Match(out[max(0, len(out)-40):]):
			out = append(out, ' ')
			out = append(out, src[from:to]...)
			out = append(out, ' ')
		default:
			out = append(out, '"', '"')
			out = append(out, strings.Repeat("\n", strings.Count(lit, "\n"))...)
		}
		i = end
	}
	return string(out)
}

var jsExts = map[string]bool{".js": true, ".jsx": true, ".ts": true, ".tsx": true, ".mjs": true, ".cjs": true}

// regexLiteralEnd: a JS regex literal (`/[/*]/`) would otherwise open a comment.
// A / starts one where an operand is due: after ( = , : [ ! & | ? { ; or at a line start.
func regexLiteralEnd(src string, i int, before []byte) (end, from, to int) {
	if i+1 < len(src) && (src[i+1] == '/' || src[i+1] == '*') {
		return -1, 0, 0
	}
	k := len(before) - 1
	for k >= 0 && (before[k] == ' ' || before[k] == '\t') {
		k--
	}
	if k >= 0 && before[k] != '\n' && !strings.ContainsRune("(=,:[!&|?{;", rune(before[k])) {
		return -1, 0, 0
	}
	inClass := false
	for j := i + 1; j < len(src); j++ {
		switch c := src[j]; {
		case c == '\\':
			j++
		case c == '\n':
			return -1, 0, 0
		case c == '[':
			inClass = true
		case c == ']':
			inClass = false
		case c == '/' && !inClass:
			return j + 1, i + 1, j
		}
	}
	return -1, 0, 0
}

// literalEnd finds the end of the string literal opening at src[i], and its
// content; -1 when it is not one (a Rust lifetime, an unclosed quote).
func literalEnd(src string, i int, before []byte, rust bool) (end, from, to int) {
	c := src[i]
	if rust && c == '\'' {
		// 'a' and '\n' are chars; 'a in <'a> or &'static is a lifetime.
		switch {
		case i+2 < len(src) && src[i+1] != '\\' && src[i+2] == '\'':
			return i + 3, i + 1, i + 2
		case i+1 < len(src) && src[i+1] == '\\':
			if j := strings.IndexByte(src[i+2:min(len(src), i+14)], '\''); j >= 0 {
				return i + 3 + j, i + 1, i + 2 + j
			}
		}
		return -1, 0, 0
	}
	if rust && c == '"' {
		// Rust raw string: r"..", r#".."#.
		k := len(before)
		for k > 0 && before[k-1] == '#' {
			k--
		}
		if k > 0 && before[k-1] == 'r' && (k == 1 || !isWordByte(before[k-2]) || before[k-2] == 'b') {
			closing := `"` + strings.Repeat("#", len(before)-k)
			if j := strings.Index(src[i+1:], closing); j >= 0 {
				return i + 1 + j + len(closing), i + 1, i + 1 + j
			}
			return -1, 0, 0
		}
	}
	if c != '`' && strings.HasPrefix(src[i:], strings.Repeat(string(c), 3)) {
		if j := strings.Index(src[i+3:], strings.Repeat(string(c), 3)); j >= 0 {
			return i + 6 + j, i + 3, i + 3 + j
		}
		return -1, 0, 0
	}
	for j := i + 1; j < len(src); j++ {
		switch src[j] {
		case '\\':
			if c != '`' {
				j++
			}
		case c:
			return j + 1, i + 1, j
		case '\n':
			if c == '\'' || c == '"' && !rust {
				return -1, 0, 0
			}
		}
	}
	return -1, 0, 0
}

func isWordByte(c byte) bool { return c == '_' || c == '$' || isAlnumByte(c) }

// testNameText is what can name a test in a file: its test part without
// comments and strings, except test-call titles. An id in a comment or in a
// string constant used to count and let 130 items pass without a round.
func testNameText(rel, content string) string {
	t := testSourceText(rel, content)
	if t == "" {
		return ""
	}
	return codeText(rel, t, keepTitles)
}

// namesItem reports whether test-name text names id: the id itself, or its token
// as a word or a camel-case part (`TestF4_1Opens` and `f4_1Opens` name f4_1 as
// `f4_1_opens` does). '_' is not alphanumeric, so "f2_3" matches in
// "fn f2_3_s1_greyed" but not in "f2_30".
func namesItem(text, id string) bool {
	if idBoundaryIndex(text, id) >= 0 {
		return true
	}
	token := specTestToken(id)
	lower := []byte(text)
	for i, c := range lower {
		if c >= 'A' && c <= 'Z' {
			lower[i] = c + 'a' - 'A'
		}
	}
	for from := 0; ; {
		i := strings.Index(string(lower[from:]), token)
		if i < 0 {
			return false
		}
		i += from
		end := i + len(token)
		start := i == 0 || !isAlnumByte(text[i-1]) || i >= 4 && string(lower[i-4:i]) == "test" && (i == 4 || !isAlnumByte(text[i-5]))
		stop := end == len(text) || !isAlnumByte(text[end]) || text[end] >= 'A' && text[end] <= 'Z'
		if start && stop {
			return true
		}
		from = i + 1
	}
}

// specNamedIn is the first of files (relative to outAbs) whose tests name id,
// skipping what specCoverage skips, or a done item would reopen at the next scan.
func specNamedIn(outAbs, id string, files []string) string {
	for _, rel := range files {
		if slices.ContainsFunc(strings.Split(filepath.Dir(rel), "/"), skipWalkDir) {
			continue
		}
		data, err := os.ReadFile(filepath.Join(outAbs, rel))
		if err != nil || len(data) > 2<<20 || looksBinary(data) {
			continue
		}
		if text := testNameText(rel, string(data)); text != "" && namesItem(text, id) {
			return rel
		}
	}
	return ""
}

// specChanges is what the round changed under the output directory, relative
// to it: each file's added lines (new numbering) and its net growth in lines.
type specChanges struct {
	added map[string][]int
	grown map[string]int
	// removed names the free functions whose definition line the round took out: a
	// changed signature or a moved function is not a new one.
	removed map[string]bool
	// moved: added lines whose text the round deleted elsewhere; moved code keeps
	// its old findings.
	moved map[string]map[int]bool
	ok    bool // false without git or without a first commit: the gates that need it are skipped
}

func (c specChanges) files() []string {
	files := make([]string, 0, len(c.added))
	for f := range c.added {
		files = append(files, f)
	}
	sort.Strings(files)
	return files
}

var hunkRe = regexp.MustCompile(`^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@`)

// specRoundChanges diffs the work tree against base, the commit before the
// item's first attempt (see specRoundBase).
func specRoundChanges(ctx context.Context, cwd, outRel, base string) specChanges {
	ch := specChanges{added: map[string][]int{}, grown: map[string]int{}, removed: map[string]bool{}, moved: map[string]map[int]bool{}}
	type addedLine struct {
		file string
		line int
		text string
	}
	var addedLines []addedLine
	deleted := map[string]bool{}
	diff, err := specGit(ctx, cwd, "-c", "core.quotePath=false", "diff", base, "--no-color", "--no-ext-diff", "--no-prefix", "--relative", "-U0", "--", outRel)
	if err != nil {
		return ch
	}
	prefix := filepath.ToSlash(outRel) + "/"
	cur, line, inHunk := "", 0, false
	for _, l := range strings.Split(diff, "\n") {
		switch {
		case strings.HasPrefix(l, "diff --git "):
			inHunk = false
		case !inHunk && strings.HasPrefix(l, "+++ "):
			cur = strings.TrimPrefix(strings.TrimRight(l[4:], "\t"), prefix)
			if l[4:] == "/dev/null" {
				cur = ""
			}
		case strings.HasPrefix(l, "@@"):
			inHunk = true
			if m := hunkRe.FindStringSubmatch(l); m != nil {
				line, _ = strconv.Atoi(m[1])
			}
		case !inHunk:
		case cur != "" && strings.HasPrefix(l, "+"):
			ch.added[cur] = append(ch.added[cur], line)
			addedLines = append(addedLines, addedLine{cur, line, l[1:]})
			ch.grown[cur]++
			line++
		case strings.HasPrefix(l, "-"):
			if m := topLevelDefRe.FindStringSubmatch(l[1:]); m != nil {
				ch.removed[m[1]+m[2]] = true
			}
			deleted[movedKey(l[1:])] = true
			if cur != "" {
				ch.grown[cur]--
			}
		}
	}
	untracked, err := specGit(ctx, cwd, "-c", "core.quotePath=false", "ls-files", "--others", "--exclude-standard", "--", outRel)
	if err != nil {
		return ch
	}
	for _, p := range strings.Split(strings.TrimSpace(untracked), "\n") {
		rel := strings.TrimPrefix(p, prefix)
		data, err := os.ReadFile(filepath.Join(cwd, p))
		if p == "" || err != nil || looksBinary(data) {
			continue
		}
		n := strings.Count(string(data), "\n")
		for i, text := range strings.SplitN(string(data), "\n", n+1)[:n] {
			ch.added[rel] = append(ch.added[rel], i+1)
			addedLines = append(addedLines, addedLine{rel, i + 1, text})
		}
		ch.grown[rel] = n
	}
	for _, a := range addedLines {
		if k := movedKey(a.text); k != "" && deleted[k] {
			if ch.moved[a.file] == nil {
				ch.moved[a.file] = map[int]bool{}
			}
			ch.moved[a.file][a.line] = true
		}
	}
	ch.ok = true
	return ch
}

// movedKey is a line as a move is recognised: without its indentation, which a
// move may change. Short lines (a brace, `} else {`) match anywhere, so they
// are never taken for moved.
func movedKey(line string) string {
	if t := strings.TrimSpace(line); len(t) >= 12 {
		return t
	}
	return ""
}

// The extensions the size budget and the reachability scan read as program code.
var codeExts = map[string]bool{
	".rs": true, ".go": true, ".py": true, ".js": true, ".jsx": true, ".ts": true, ".tsx": true, ".mjs": true,
	".cjs": true, ".java": true, ".kt": true, ".kts": true, ".scala": true, ".swift": true, ".c": true,
	".cc": true, ".cpp": true, ".cxx": true, ".h": true, ".hpp": true, ".cs": true, ".dart": true,
	".rb": true, ".php": true, ".lua": true, ".ex": true, ".exs": true, ".vue": true, ".svelte": true,
}

// Neither the program nor its tests: a function only an example calls is still dead.
var specSideDirs = map[string]bool{"examples": true, "example": true, "benches": true, "bench": true}

// prodSource is a code file's program part: nothing for a test file, and the
// part before an inline test module.
func prodSource(rel, content string) string {
	if !codeExts[strings.ToLower(filepath.Ext(rel))] {
		return ""
	}
	for _, part := range strings.Split(filepath.ToSlash(filepath.Dir(rel)), "/") {
		if specSideDirs[part] {
			return ""
		}
	}
	t := testSourceText(rel, content)
	return content[:len(content)-len(t)]
}

// topLevelDefRe finds a free function at column 0: a Rust fn, a Go func without
// a receiver, a Python or Ruby def, a JS/TS function or arrow constant, a Kotlin
// fun. Methods are skipped: an interface or trait calls them where no text search looks.
var topLevelDefRe = regexp.MustCompile(`^(?:pub(?:\([^)]*\))?\s+)?(?:export\s+)?(?:default\s+)?(?:public\s+|internal\s+|private\s+)?(?:const\s+|async\s+|unsafe\s+|suspend\s+|local\s+)*(?:fn|func|def|function\*?|fun)\s+([A-Za-z_$][\w$]*)\s*(?:[(<\[]|$)|^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::\s*[^=]+)?=>`)

// A decorator or an export attribute hands the function to something outside.
var externalDefRe = regexp.MustCompile(`^(?:@|//go:|//export|#\[(?:no_mangle|export_name|wasm_bindgen|proc_macro|\w+::))`)

// Names a framework calls by convention, never the program: route loaders, Next.js
// data functions, handlers.
var frameworkNames = map[string]bool{"loader": true, "action": true, "meta": true, "links": true, "handler": true,
	"middleware": true, "getServerSideProps": true, "getStaticProps": true, "getStaticPaths": true, "generateMetadata": true}

type codeDef struct {
	name, file string
	line       int // 1-based
}

// defsIn lists the free functions defined on the given lines of src (nil: all lines).
func defsIn(rel, src string, lines []int) []codeDef {
	all := strings.Split(src, "\n")
	want := map[int]bool{}
	for _, n := range lines {
		want[n] = true
	}
	var defs []codeDef
	for i, l := range all {
		if lines != nil && !want[i+1] {
			continue
		}
		m := topLevelDefRe.FindStringSubmatch(l)
		if m == nil {
			continue
		}
		name := m[1] + m[2]
		switch lower := strings.ToLower(name); {
		case lower == "main", lower == "init", lower == "setup", lower == "teardown", strings.Contains(lower, "test"),
			frameworkNames[name], strings.Contains(l, "export default"):
			continue
		}
		// A decorator or attribute block above it (stacked, or spread over lines)
		// hands it to something outside the program's own calls.
		external := false
	attrs:
		for j := i - 1; j >= 0 && j >= i-8; j-- {
			switch t := strings.TrimSpace(all[j]); {
			case t == "":
				break attrs
			case externalDefRe.MatchString(t):
				external = true
				break attrs
			case all[j] == t && !strings.HasPrefix(t, "#") && !strings.HasPrefix(t, ")") && !strings.HasPrefix(t, "//"):
				break attrs // code at column 0: the attribute block is over
			}
		}
		if external {
			continue
		}
		defs = append(defs, codeDef{name: name, file: rel, line: i + 1})
	}
	return defs
}

// codeFile indexes one program file for reference lookups.
type codeFile struct {
	rel, dir, ext, stem string
	lines               []string
	words               map[string][]int  // word → 0-based lines
	aliases             map[string]string // module stem → the names this file imports it as
}

// multiImportRe opens an import that goes on over lines: `use crate::x::{`, `import {`.
var multiImportRe = regexp.MustCompile(`^\s*(?:pub(?:\([^)]*\))?\s+)?(?:use\s+[\w:]+::\{|import\s+\{|from\s+[\w.]+\s+import\s+\()`)

// importModule is the module a multi-line import names: the path before `::{` or
// after `from`, as the last segment.
var importFromRe = regexp.MustCompile(`from\s+['"]?([^'"\s;(]+)`)

func importModule(stmt string) string {
	if i := strings.Index(stmt, "::{"); i >= 0 {
		path := strings.Fields(stmt[:i])
		return filepath.Base(strings.ReplaceAll(path[len(path)-1], "::", "/"))
	}
	if m := importFromRe.FindStringSubmatch(stmt); m != nil {
		p := strings.ReplaceAll(m[1], ".", "/")
		return strings.TrimSuffix(filepath.Base(p), filepath.Ext(p))
	}
	return ""
}

// Import aliases: `use a::render_mod as r;`, `import x as y`, `import * as r from './render_mod'`,
// and Go's `r "example.com/app/render_mod"`.
var (
	asAliasRe   = regexp.MustCompile(`^\s*(?:pub(?:\([^)]*\))?\s+)?(?:use|import|from)\b.*?(\w+)\s+as\s+(\w+)`)
	starAliasRe = regexp.MustCompile(`import\s+\*\s+as\s+(\w+)\s+from\s+['"]([^'"]+)['"]`)
	goAliasRe   = regexp.MustCompile(`^\s*(?:import\s+)?(\w+)\s+"([^"]+)"`)
)

// indexCode indexes text; raw is the same code as written, where the import
// aliases are read (text may have lost its quotes).
func indexCode(rel, text, raw string) *codeFile {
	stem := strings.TrimSuffix(filepath.Base(rel), filepath.Ext(rel))
	switch {
	case stem == "mod", stem == "index", stem == "__init__", stem == "lib",
		strings.HasSuffix(rel, ".go"): // a Go file is called by its package, the directory
		stem = filepath.Base(filepath.Dir(rel))
	}
	f := &codeFile{rel: rel, dir: filepath.Dir(rel), ext: strings.ToLower(filepath.Ext(rel)), stem: stem,
		lines: strings.Split(text, "\n"), words: map[string][]int{}, aliases: map[string]string{}}
	rawLines := strings.Split(raw, "\n")
	block := -1 // first line of a multi-line import still open
	for n, l := range rawLines {
		switch m := starAliasRe.FindStringSubmatch(l); {
		case m != nil:
			f.aliases[strings.TrimSuffix(filepath.Base(m[2]), filepath.Ext(m[2]))] = m[1]
		case asAliasRe.MatchString(l):
			m := asAliasRe.FindStringSubmatch(l)
			f.aliases[m[1]] = m[2]
		case f.ext == ".go" && goAliasRe.MatchString(l):
			m := goAliasRe.FindStringSubmatch(l)
			f.aliases[filepath.Base(m[2])] = m[1]
		}
		// rustfmt and prettier put a long import on many lines; each name then
		// carries the module, as it would on one line.
		switch {
		case multiImportRe.MatchString(l) && !strings.Contains(l, "}"):
			block = n
		case block >= 0 && strings.Contains(l, "}"):
			module := importModule(rawLines[block] + " " + l)
			for k := block; k <= n && k < len(f.lines); k++ {
				f.lines[k] += " " + module
			}
			block = -1
		}
	}
	for n, l := range f.lines {
		for i := 0; i < len(l); {
			if !isWordByte(l[i]) {
				i++
				continue
			}
			j := i
			for j < len(l) && isWordByte(l[j]) {
				j++
			}
			if w, ws := l[i:j], f.words[l[i:j]]; len(ws) == 0 || ws[len(ws)-1] != n {
				f.words[w] = append(ws, n)
			}
			i = j
		}
	}
	return f
}

// Languages whose files share one namespace per directory: a sibling file calls
// a function by its bare name.
var packageScoped = map[string]bool{".go": true, ".java": true, ".kt": true, ".cs": true, ".scala": true,
	".swift": true, ".c": true, ".cc": true, ".cpp": true, ".h": true, ".hpp": true}

// specProgram is the program's code, indexed, and the words its tests use.
type specProgram struct {
	files     []*codeFile
	byRel     map[string]*codeFile
	globbed   map[string]bool // stems re-exported whole (`pub use x::*`, `export * from './x'`)
	testWords map[string]bool
	defined   map[string]int // how many free functions of each name the program defines
}

func loadSpecProgram(outAbs string) *specProgram {
	p := &specProgram{byRel: map[string]*codeFile{}, globbed: map[string]bool{}, testWords: map[string]bool{}, defined: map[string]int{}}
	_ = filepath.WalkDir(outAbs, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if path != outAbs && skipWalkDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		rel, _ := filepath.Rel(outAbs, path)
		rel = filepath.ToSlash(rel)
		if !codeExts[strings.ToLower(filepath.Ext(rel))] {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil || len(data) > 2<<20 || looksBinary(data) {
			return nil
		}
		content := string(data)
		if t := testSourceText(rel, content); t != "" {
			for w := range indexCode(rel, codeText(rel, t, blankStrings), t).words {
				p.testWords[w] = true
			}
		}
		if prod := prodSource(rel, content); prod != "" {
			f := indexCode(rel, codeText(rel, prod, keepStrings), prod)
			p.files = append(p.files, f)
			p.byRel[rel] = f
			for _, l := range strings.Split(prod, "\n") {
				if m := topLevelDefRe.FindStringSubmatch(l); m != nil {
					p.defined[m[1]+m[2]]++
				}
			}
		}
		return nil
	})
	for _, f := range p.files {
		for _, l := range f.lines {
			if strings.Contains(l, "*") {
				for _, g := range p.files {
					if g != f && strings.Contains(l, g.stem) {
						p.globbed[g.stem] = true
					}
				}
			}
		}
	}
	return p
}

// called reports whether program code other than the definition itself uses it.
func (p *specProgram) called(d codeDef) bool {
	f := p.byRel[d.file]
	if f == nil {
		return true
	}
	uses := func(g *codeFile, line func(string) bool) bool {
		for _, n := range g.words[d.name] {
			if (g != f || n != d.line-1) && line(g.lines[n]) {
				return true
			}
		}
		return false
	}
	any := func(string) bool { return true }
	if uses(f, any) {
		return true
	}
	// A name no other function has is that function wherever it appears: an alias
	// or a multi-line import hides the module, not the name. A name like `run` needs
	// its module on the line: a qualified call (stem::run, stem.run) or an import.
	unique := p.defined[d.name] <= 1
	for _, g := range p.files {
		if g == f {
			continue
		}
		alias := g.aliases[f.stem]
		switch {
		case unique && uses(g, any),
			uses(g, func(l string) bool { return strings.Contains(l, f.stem) || alias != "" && strings.Contains(l, alias) }),
			packageScoped[f.ext] && g.dir == f.dir && g.ext == f.ext && uses(g, any),
			p.globbed[f.stem] && uses(g, any):
			return true
		}
	}
	return false
}

// specUnreachable lists the free functions the round added to the program that
// only tests call, or nothing does. In naivepost 21 modules and ~580 public
// functions were reachable from tests alone while their items counted as done.
func specUnreachable(outAbs string, ch specChanges, prog *specProgram) []string {
	var out []string
	for _, rel := range ch.files() {
		data, err := os.ReadFile(filepath.Join(outAbs, rel))
		if err != nil || prog.byRel[rel] == nil {
			continue
		}
		prod := prodSource(rel, string(data))
		for _, d := range defsIn(rel, prod, ch.added[rel]) {
			if ch.removed[d.name] || prog.called(d) {
				continue
			}
			who := "nothing calls it"
			if prog.testWords[d.name] {
				who = "only tests call it"
			}
			out = append(out, fmt.Sprintf("`%s` (%s:%d): %s", d.name, rel, d.line, who))
		}
	}
	return out
}

var (
	// forTestCallRe: a call of a helper named for tests, the naming rule the round prompt sets.
	forTestCallRe = regexp.MustCompile(`\b([A-Za-z_]\w*(?:_for_tests?|_for_testing|ForTests?|ForTesting))\s*\(`)
	// Rust's macros, Kotlin's TODO(), Python's, .NET's and Java-land's not-implemented errors.
	notBuiltCodeRe = regexp.MustCompile(`\b(?:todo|unimplemented)!\s*[(\[{]|\bTODO\s*\(|\bNotImplemented(?:Error|Exception)?\b`)
	// notBuiltTextRe runs on code with its strings kept, comments gone: a message, not a remark.
	notBuiltTextRe = regexp.MustCompile(`(?i)\bnot (?:yet )?implemented\b|\bunimplemented\b|\bstub(?:bed)?\b`)
)

// specStandIns lists added program lines that fake the work instead of doing it:
// asking a for-test hook, or saying "not implemented".
func specStandIns(outAbs string, ch specChanges) []string {
	var out []string
	for _, rel := range ch.files() {
		data, err := os.ReadFile(filepath.Join(outAbs, rel))
		if err != nil {
			continue
		}
		prod := prodSource(rel, string(data))
		if prod == "" {
			continue
		}
		raw := strings.Split(prod, "\n")
		code := strings.Split(codeText(rel, prod, blankStrings), "\n")
		kept := strings.Split(codeText(rel, prod, keepStrings), "\n")
		added := map[int]bool{}
		for _, n := range ch.added[rel] {
			added[n] = true
		}
		inHelper := false // inside a top-level function that is itself a for-test helper
		for i, ln := range code {
			if m := topLevelDefRe.FindStringSubmatch(ln); m != nil {
				inHelper = forTestCallRe.MatchString(m[1] + m[2] + "(")
			}
			if !added[i+1] || inHelper {
				continue
			}
			at := fmt.Sprintf("%s:%d", rel, i+1)
			switch {
			case forTestCallRe.MatchString(ln):
				name := forTestCallRe.FindStringSubmatch(ln)[1]
				out = append(out, fmt.Sprintf("`%s` (%s): the program asks a helper that exists for tests, so outside a test it never gets the real answer", name, at))
			case notBuiltCodeRe.MatchString(ln) || i < len(kept) && notBuiltTextRe.MatchString(kept[i]):
				out = append(out, fmt.Sprintf("%s: `%s` says the work is not built", at, strings.TrimSpace(clipUTF8(raw[i], 120))))
			}
		}
	}
	return out
}

// Defaults for the size budget: a file over maxFileLines may still grow by
// fileGrowthSlack lines, enough for the call into a new module.
const (
	specDefaultMaxFileLines = 1500
	specFileGrowthSlack     = 20
	specDefaultRefactorEach = 10
)

func (c *specConfig) maxFileLines() int {
	if c.MaxFileLines == 0 {
		return specDefaultMaxFileLines
	}
	return c.MaxFileLines
}

func (c *specConfig) refactorEvery() int {
	if c.RefactorEvery == 0 {
		return specDefaultRefactorEach
	}
	return c.RefactorEvery
}

// specOversize: in naivepost one UI file took 60% of all new code and reached
// 12,500 lines, too long for the model to read, so it copied blocks instead.
func specOversize(outAbs string, cfg *specConfig, ch specChanges) []string {
	max := cfg.maxFileLines()
	if max < 0 {
		return nil
	}
	var out []string
	for _, rel := range ch.files() {
		data, err := os.ReadFile(filepath.Join(outAbs, rel))
		if err != nil || prodSource(rel, string(data)) == "" {
			continue
		}
		n, grew := strings.Count(string(data), "\n"), ch.grown[rel]
		if n > max && grew > specFileGrowthSlack {
			out = append(out, fmt.Sprintf("`%s` is %d lines, over the %d-line budget, and this round grew it by %d: put the new code in a new file and keep this one's growth under %d lines", rel, n, max, grew, specFileGrowthSlack))
		}
	}
	return out
}

// lintCmd is the linter the project already has, or spec.toml's; "" is none.
func (r *specRun) lintCmd() string {
	switch r.cfg.LintCmd {
	case "off":
		return ""
	case "":
	default:
		return r.cfg.LintCmd
	}
	var pkg struct {
		Scripts map[string]string `json:"scripts"`
	}
	data, _ := os.ReadFile(filepath.Join(r.outAbs, "package.json")) // absent: no npm lint script
	switch dir := r.outAbs; {
	case justRecipe(dir, "lint"):
		return "just lint"
	case fileExists(dir, "Cargo.toml"):
		return "cargo clippy --all-targets --quiet"
	case fileExists(dir, "go.mod"):
		return "go vet ./..."
	case json.Unmarshal(data, &pkg) == nil && pkg.Scripts["lint"] != "":
		return "npm run --silent lint"
	case fileExists(dir, "ruff.toml"), fileExists(dir, "pyproject.toml"):
		return "ruff check --output-format concise ."
	}
	return ""
}

// Any "path.ext:line" in a linter's output: clippy's `--> src/a.rs:10:5`, go
// vet's `./a.go:10:2: msg`, ruff's and eslint's unix format.
// tsc writes `a.ts(10,5)`; eslint's default names the file on a line of its own
// and each finding under it as `  10:5  error  ...`.
var (
	lintLocRe     = regexp.MustCompile(`([\w./\\-]+\.[A-Za-z0-9]+)(?::(\d+)|\((\d+),\d+\))`)
	lintStylishRe = regexp.MustCompile(`^\s+(\d+):\d+\s+(?:error|warning)\b`)
)

// specLint runs the linter and keeps what it reports in the lines this round
// wrote: the code before the round is not the round's to fix. missing is why the
// linter did not run (not installed), which is no reason to fail a round.
func specLint(ctx context.Context, outAbs, cmd string, ch specChanges) (findings []string, missing string) {
	tctx, cancel := context.WithTimeout(ctx, specTestTimeout)
	defer cancel()
	c := exec.CommandContext(tctx, specShell(), "-lc", cmd)
	c.Dir = outAbs
	raw, err := c.CombinedOutput()
	out := string(raw)
	var ee *exec.ExitError
	if tctx.Err() != nil || err != nil && !errors.As(err, &ee) || ee != nil && ee.ExitCode() == 127 ||
		strings.Contains(out, "no such command") || strings.Contains(out, "command not found") || strings.Contains(out, "is not installed") {
		why := "it did not finish"
		for _, l := range strings.Split(out, "\n") {
			if l = strings.TrimSpace(l); l != "" {
				why = truncate(l, 200)
				break
			}
		}
		return nil, why
	}
	lines := strings.Split(out, "\n")
	seen := map[string]bool{}
	stylishFile := ""
	for i, l := range lines {
		type loc struct{ path, line, text string }
		var locs []loc
		switch t := strings.TrimSpace(l); {
		case lintStylishRe.MatchString(l):
			if stylishFile != "" {
				locs = append(locs, loc{stylishFile, lintStylishRe.FindStringSubmatch(l)[1], ""})
			}
		// A header is a path alone at column 0; it may hold spaces or parentheses (a route group).
		case t == l && filepath.Ext(t) != "" && strings.Contains(t, "/") && !strings.Contains(t, ": "):
			stylishFile = t
		default:
			stylishFile = "" // a blank line or a summary ends the file's block
		}
		for _, m := range lintLocRe.FindAllStringSubmatch(l, -1) {
			locs = append(locs, loc{m[1], m[2] + m[3], m[0]})
		}
		for _, lc := range locs {
			p := filepath.ToSlash(lc.path)
			if filepath.IsAbs(p) {
				if rel, err := filepath.Rel(outAbs, p); err == nil {
					p = filepath.ToSlash(rel)
				}
			}
			p = strings.TrimPrefix(p, "./")
			n, _ := strconv.Atoi(lc.line)
			if !slices.Contains(ch.added[p], n) || ch.moved[p][n] || seen[p+":"+lc.line] {
				continue
			}
			seen[p+":"+lc.line] = true
			// clippy puts the message a few lines above its `-->` location.
			msg := strings.TrimSpace(l)
			if lc.text == "" {
				msg = p + ":" + lc.line + ": " + msg
			}
			if strings.HasPrefix(msg, "-->") || msg == lc.text {
				for j := i - 1; j >= 0 && j >= i-5; j-- {
					if t := strings.TrimSpace(lines[j]); strings.HasPrefix(t, "warning") || strings.HasPrefix(t, "error") {
						msg = t + " " + msg
						break
					}
				}
			}
			findings = append(findings, msg)
		}
		if len(findings) >= 12 {
			break
		}
	}
	return findings, ""
}

// specDebt is what a refactor round must shrink: lines over the size budget,
// functions nothing calls, and test helpers copied into several test files.
type specDebt struct {
	overLines, dead, copies int
	targets                 string
}

func (d specDebt) String() string {
	return fmt.Sprintf("lines over the size budget %d, functions nothing calls %d, copied test helpers %d", d.overLines, d.dead, d.copies)
}

// better: something shrank and nothing grew.
func (d specDebt) better(before specDebt) bool {
	return d.overLines <= before.overLines && d.dead <= before.dead && d.copies <= before.copies &&
		(d.overLines < before.overLines || d.dead < before.dead || d.copies < before.copies)
}

func measureSpecDebt(outAbs string, cfg *specConfig) specDebt {
	var d specDebt
	var b strings.Builder
	prog := loadSpecProgram(outAbs)
	max := cfg.maxFileLines()
	type big struct {
		rel   string
		lines int
	}
	var bigs []big
	var dead []string
	for _, f := range prog.files {
		data, err := os.ReadFile(filepath.Join(outAbs, f.rel))
		if err != nil {
			continue
		}
		if n := strings.Count(string(data), "\n"); max > 0 && n > max {
			d.overLines += n - max
			bigs = append(bigs, big{f.rel, n})
		}
		for _, def := range defsIn(f.rel, prodSource(f.rel, string(data)), nil) {
			if !prog.called(def) && !prog.testWords[def.name] {
				dead = append(dead, fmt.Sprintf("`%s` (%s:%d)", def.name, def.file, def.line))
			}
		}
	}
	d.dead = len(dead)
	helpers := map[string][]string{}
	_ = filepath.WalkDir(outAbs, func(path string, e os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if e.IsDir() {
			if path != outAbs && skipWalkDir(e.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		rel, _ := filepath.Rel(outAbs, path)
		rel = filepath.ToSlash(rel)
		data, err := os.ReadFile(path)
		if err != nil || !codeExts[strings.ToLower(filepath.Ext(rel))] || looksBinary(data) {
			return nil
		}
		if t := testSourceText(rel, string(data)); t == string(data) {
			for _, def := range defsIn(rel, t, nil) {
				helpers[def.name] = append(helpers[def.name], rel)
			}
		}
		return nil
	})
	type copied struct {
		name  string
		files int
	}
	var copies []copied
	for name, files := range helpers {
		if len(files) > 1 {
			d.copies += len(files) - 1
			copies = append(copies, copied{name, len(files)})
		}
	}
	slices.SortFunc(bigs, func(x, y big) int { return y.lines - x.lines })
	slices.SortFunc(copies, func(x, y copied) int { return y.files - x.files })
	slices.Sort(dead)
	if len(bigs) > 0 {
		fmt.Fprintf(&b, "Files over the %d-line budget, largest first:\n", max)
		for _, x := range bigs[:min(len(bigs), 5)] {
			fmt.Fprintf(&b, "- `%s`: %d lines\n", x.rel, x.lines)
		}
	}
	if len(dead) > 0 {
		b.WriteString("Functions nothing calls, not even a test (delete them, or call them where the spec needs them):\n")
		for _, x := range dead[:min(len(dead), 15)] {
			b.WriteString("- " + x + "\n")
		}
	}
	if len(copies) > 0 {
		b.WriteString("Test helpers defined in several test files (move each into one shared test module and use it from there):\n")
		for _, x := range copies[:min(len(copies), 10)] {
			fmt.Fprintf(&b, "- `%s`: %d files\n", x.name, x.files)
		}
	}
	d.targets = b.String()
	return d
}
