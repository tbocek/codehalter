package main

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// Qwen3.8 flash reads through the shell when the history shows it doing so, but
// where it does not (a fresh step, after a compaction) it calls a read tool it was
// trained on, in the shapes of other harnesses (`file_path`/`path`/`target_file`,
// `offset`/`start_line` + `limit`): 29 of 48 first calls replayed at a bare rename step,
// and 8 calls (`read`, `read_file`, `read_command`, `exec`, `sed`) in 24 replay runs.
// An "unknown tool" answer costs a call and teaches nothing; the same read as the
// run_command it means, put into history in its place, does both.
var (
	readAliases  = []string{"read", "read_file", "readfile", "file_read", "view", "view_file", "open_file", "cat", "get_file", "sed", "head"}
	shellAliases = []string{"read_command", "bash", "shell", "run", "exec", "execute", "run_shell", "terminal", "execute_command"}
	pathKeys     = []string{"path", "file_path", "filepath", "target_file", "file", "filename", "absolute_path"}
)

// canonicalCall rewrites a call to a tool that does not exist into the run_command
// it stands for, when its arguments say plainly what to run or read.
func canonicalCall(tc toolCall) (toolCall, bool) {
	name := strings.ToLower(tc.Function.Name)
	var a map[string]any
	if json.Unmarshal([]byte(tc.Function.Arguments), &a) != nil {
		return tc, false
	}
	// Some calls nest their real arguments: {"arguments": {"path": ...}}.
	switch inner := a["arguments"].(type) {
	case map[string]any:
		a = inner
	case string:
		var m map[string]any
		if json.Unmarshal([]byte(inner), &m) == nil {
			a = m
		}
	}
	var cmd string
	switch {
	case slices.Contains(shellAliases, name):
		cmd, _ = a["command"].(string)
	case slices.Contains(readAliases, name):
		cmd = readCommand(a)
	}
	if strings.TrimSpace(cmd) == "" {
		return tc, false
	}
	b, _ := json.Marshal(map[string]string{"command": cmd})
	out := tc
	out.Function.Name, out.Function.Arguments = "run_command", string(b)
	return out, true
}

// readCommand: `sed -n 'a,bp' path` for a range, `cat path` for a whole file.
func readCommand(a map[string]any) string {
	type read struct {
		path       string
		start, end int
	}
	num := func(keys ...string) int {
		for _, k := range keys {
			switch v := a[k].(type) {
			case float64:
				return int(v)
			case string:
				if n, err := strconv.Atoi(v); err == nil {
					return n
				}
			}
		}
		return 0
	}
	pair := func(v any) (int, int) {
		switch r := v.(type) {
		case []any:
			if len(r) == 2 {
				s, _ := r[0].(float64)
				e, _ := r[1].(float64)
				return int(s), int(e)
			}
			if len(r) == 1 {
				if m, ok := r[0].(map[string]any); ok {
					s, _ := m["start"].(float64)
					e, _ := m["end"].(float64)
					return int(s), int(e)
				}
			}
		case map[string]any:
			s, _ := r["start"].(float64)
			e, _ := r["end"].(float64)
			return int(s), int(e)
		}
		return 0, 0
	}
	var reads []read
	for _, k := range pathKeys {
		if p, ok := a[k].(string); ok && p != "" {
			s := num("offset", "start", "start_line", "line", "from", "line_start")
			e := num("end", "end_line", "to", "line_end")
			if l := num("limit", "lines", "count"); l > 0 && e == 0 {
				e = max(s, 1) + l - 1
			}
			if s == 0 && e == 0 {
				s, e = pair(a["range"])
			}
			reads = append(reads, read{p, s, e})
			break
		}
	}
	switch ps := a["paths"].(type) {
	case []any:
		for _, p := range ps {
			if s, ok := p.(string); ok && s != "" {
				reads = append(reads, read{path: s})
			}
		}
	case map[string]any:
		keys := make([]string, 0, len(ps))
		for k := range ps {
			keys = append(keys, k)
		}
		slices.Sort(keys)
		for _, k := range keys {
			s, e := pair(ps[k])
			reads = append(reads, read{k, s, e})
		}
	}
	var parts []string
	for _, r := range reads {
		q := shellQuote(r.path)
		var c string
		if r.start > 0 || r.end > 0 {
			end := "$"
			if r.end > 0 {
				end = strconv.Itoa(r.end)
			}
			c = fmt.Sprintf("sed -n '%d,%sp' %s", max(r.start, 1), end, q)
		} else {
			c = "cat " + q
		}
		if len(reads) > 1 {
			c = fmt.Sprintf("echo '== %s'; %s", strings.ReplaceAll(r.path, "'", ""), c)
		}
		parts = append(parts, c)
	}
	return strings.Join(parts, "; ")
}

// shellQuote leaves a plain path bare, as the model writes it.
func shellQuote(s string) string {
	if s != "" && strings.Trim(s, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-+@") == "" {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}
