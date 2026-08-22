package main

import (
	"strings"
	"testing"
)

// planArgs is a realistic submit_plan argument payload: a pipe inside a shell
// command, a newline inside a description, and a verify list on every subtask.
const planArgs = `{"clear":true,"subtasks":[` +
	`{"description":"Add humanBytes to prompt.go","verify":["go build ./...","gofmt -l ."]},` +
	`{"description":"Count rows with grep -c x | wc -l","verify":["go vet ./..."]},` +
	`{"description":"Wrap up\nsecond line","verify":[]}` +
	`],"report_only":false}`

// feedAll replays args one byte at a time, the worst case for the partial
// parser, and returns everything the table emitted.
func feedAll(t *testing.T, args string, chunk int) string {
	t.Helper()
	var out strings.Builder
	tbl := &planTable{}
	for i := 0; i < len(args); i += chunk {
		end := i + chunk
		if end > len(args) {
			end = len(args)
		}
		tbl.feed(args[i:end], func(s string) { out.WriteString(s) })
	}
	return out.String()
}

func TestPlanTableStreamsSameRowsAtAnyChunkSize(t *testing.T) {
	// The whole point of holding the tail back is that chunking must not change
	// the result: llama.cpp delivers ~4 chars a chunk, vLLM delivers almost all
	// of it at once, and both must render identically.
	want := feedAll(t, planArgs, len(planArgs))
	for _, chunk := range []int{1, 3, 17, 64} {
		if got := feedAll(t, planArgs, chunk); got != want {
			t.Errorf("chunk=%d:\n got %q\nwant %q", chunk, got, want)
		}
	}
	if n := strings.Count(want, "\n|"); n != 5 { // header + separator + 3 rows
		t.Errorf("row count: got %d lines, want 5\n%s", n, want)
	}
}

func TestPlanTableEscapesCellsSoRowsCannotSplit(t *testing.T) {
	got := feedAll(t, planArgs, 1)
	if !strings.Contains(got, `grep -c x \| wc -l`) {
		t.Errorf("pipe not escaped, row would split into phantom columns:\n%s", got)
	}
	if strings.Contains(got, "Wrap up\nsecond") {
		t.Errorf("embedded newline reached the cell, table would end early:\n%s", got)
	}
	if !strings.Contains(got, "Wrap up<br>second line") {
		t.Errorf("newline should survive as a break, not vanish:\n%s", got)
	}
	if !strings.Contains(got, `go build ./...<br>gofmt -l .`) {
		t.Errorf("verify steps should each get their own line:\n%s", got)
	}
	// Every emitted line must be a complete table row: same pipe count throughout.
	for _, line := range strings.Split(strings.TrimSpace(got), "\n") {
		if n := strings.Count(line, "|") - strings.Count(line, `\|`); n != 3 {
			t.Errorf("line has %d structural pipes, want 3: %q", n, line)
		}
	}
}

func TestPlanTableHoldsTheTailUntilItsObjectCloses(t *testing.T) {
	// Truncated mid-description: the two closed subtasks render, the third must
	// not, because a row already sent can never be corrected.
	cut := strings.Index(planArgs, `{"description":"Wrap up`) + 15
	got := feedAll(t, planArgs[:cut], 1)
	if strings.Contains(got, "Wrap up") {
		t.Errorf("emitted a row whose object never closed:\n%s", got)
	}
	if !strings.Contains(got, "humanBytes") || !strings.Contains(got, "grep -c x") {
		t.Errorf("dropped rows that had closed:\n%s", got)
	}
}

func TestPlanTableEmitsNothingWithoutSubtasks(t *testing.T) {
	// The clarification path (clear=false) carries no subtasks, so no table and
	// in particular no dangling header.
	if got := feedAll(t, `{"clear":false,"question":"which one?","subtasks":[],"report_only":false}`, 1); got != "" {
		t.Errorf("want no output, got %q", got)
	}
}

func TestRepairJSONClosesOpenStructures(t *testing.T) {
	for _, tc := range []struct{ name, in, want string }{
		{"open object", `{"a":1`, `{"a":1}`},
		{"open array", `{"a":[1,2`, `{"a":[1,2]}`},
		{"open string", `{"a":"hi`, `{"a":"hi"}`},
		{"dangling escape", `{"a":"hi\`, `{"a":"hi\\"}`},
		{"trailing comma", `{"a":[1],`, `{"a":[1]}`},
		{"dangling key", `{"a":`, `{"a":null}`},
		{"brackets in string", `{"a":"x{[", "b":2`, `{"a":"x{[", "b":2}`},
		{"already complete", `{"a":1}`, `{"a":1}`},
	} {
		if got := repairJSON(tc.in); got != tc.want {
			t.Errorf("%s: repairJSON(%q) = %q, want %q", tc.name, tc.in, got, tc.want)
		}
	}
}

func TestPlanCellKeepsLineStructureInsideOneRow(t *testing.T) {
	if got := planCell("first line\nsecond line"); got != "first line<br>second line" {
		t.Errorf("newline not turned into a break: %q", got)
	}
	// Whatever the input, a real line break must never reach the cell: one ends
	// the table at that row and the rest of the plan renders as loose text.
	for _, in := range []string{"a\nb", "a\r\nb", "a\n\n\n\nb", "\n\nlead", "trail\n\n"} {
		if got := planCell(in); strings.ContainsAny(got, "\n\r") {
			t.Errorf("planCell(%q) leaked a line break: %q", in, got)
		}
	}
	if got := planCell("a\n\n\n\nb"); got != "a<br><br>b" {
		t.Errorf("blank run should collapse to one gap: %q", got)
	}
	if got := planCell("\n\nmiddle\n\n"); got != "middle" {
		t.Errorf("blank lines at the edges should go: %q", got)
	}
	// A shell continuation ends in a backslash, which would escape the `<` of the
	// following <br> and print the tag as text instead of breaking the line.
	if got := planCell("cmd \\\n--flag"); got != `cmd \ <br>--flag` {
		t.Errorf("dangling backslash not defused: %q", got)
	}
	// Nothing is clipped: the cell carries the full instruction.
	long := strings.Repeat("run the command and check it ", 60) // ~1700 chars, as real plans are
	if got := planCell(long); got != strings.TrimSpace(long) {
		t.Errorf("cell was altered: %d chars from %d", len(got), len(long))
	}
	// Leading indentation survives so heredoc'd source keeps its shape, while
	// interior runs still collapse.
	if got := planCell("func f() {\n\tif x {\n\t\treturn  y\n"); got != "func f() {<br>    if x {<br>        return y" {
		t.Errorf("indentation not preserved: %q", got)
	}
	// Multi-byte text passes through intact.
	if got := planCell("überprüfen die Änderung"); got != "überprüfen die Änderung" {
		t.Errorf("multi-byte text mangled: %q", got)
	}
}
