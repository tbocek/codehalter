package main

import (
	"strings"
	"testing"
)

// planArgs has a pipe in a command, a newline in a description and a verify list per subtask.
const planArgs = `{"clear":true,"subtasks":[` +
	`{"description":"Add humanBytes to prompt.go","verify":["go build ./...","gofmt -l ."]},` +
	`{"description":"Count rows with grep -c x | wc -l","verify":["go vet ./..."]},` +
	`{"description":"Wrap up\nsecond line","verify":[]}` +
	`],"report_only":false}`

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
	// llama.cpp streams a few chars a chunk, vLLM nearly everything at once.
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
	for _, line := range strings.Split(strings.TrimSpace(got), "\n") {
		if n := strings.Count(line, "|") - strings.Count(line, `\|`); n != 3 {
			t.Errorf("line has %d structural pipes, want 3: %q", n, line)
		}
	}
}

func TestPlanTableHoldsTheTailUntilItsObjectCloses(t *testing.T) {
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
	if got := planCell("cmd \\\n--flag"); got != `cmd \ <br>--flag` {
		t.Errorf("dangling backslash not defused: %q", got)
	}
	long := strings.Repeat("run the command and check it ", 60)
	if got := planCell(long); got != strings.TrimSpace(long) {
		t.Errorf("cell was altered: %d chars from %d", len(got), len(long))
	}
	// Leading indentation survives; interior runs still collapse.
	if got := planCell("func f() {\n\tif x {\n\t\treturn  y\n"); got != "func f() {<br>    if x {<br>        return y" {
		t.Errorf("indentation not preserved: %q", got)
	}
	if got := planCell("überprüfen die Änderung"); got != "überprüfen die Änderung" {
		t.Errorf("multi-byte text mangled: %q", got)
	}
}
