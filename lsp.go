package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
	"time"
)

// Post-write diagnostics: after every successful write_file / edit_file,
// codehalter asks a wired language server what it thinks of the file and
// appends the answer to the tool result.
//
// Why this is done in code rather than by telling the model to check: because
// telling it does not work. SKILL-go.md has instructed the model to navigate
// with go_definition / go_references instead of search_text since it was
// written, and a measured session made zero LSP calls and grepped instead. A
// weak model will not spend a tool call on a check it doesn't believe it needs.
// So the check is not offered to it, it is performed for it, on the one event
// that most reliably introduces a compile error: a write.
//
// The payoff is latency, not accuracy. The break is already detectable — it
// surfaces at the next `just build`, or in the subtask's verify recipe, two or
// three model round-trips later. Each of those round-trips is a full generation.
// Folding the diagnostic into the write's own tool result puts it in front of
// the model before it plans its next call, at the cost of one MCP round-trip
// and zero tokens when the file is clean.
//
// Everything here is best-effort. No language server, no diagnostics tool, a
// tool whose schema we can't fill, a timeout, an error: all of them return ""
// and the write behaves exactly as it did before.

// diagTimeout bounds the diagnostics call. gopls answers from its already-built
// package graph in tens of milliseconds; a cold or wedged server must not stall
// the write behind it. On timeout the write returns undecorated.
const diagTimeout = 6 * time.Second

// diagMaxBytes clips the diagnostics text appended to a tool result. A file
// with a syntax error near the top can produce hundreds of cascading errors,
// and pasting all of them costs more context than the edit itself. The first
// few are the actionable ones.
const diagMaxBytes = 1200

// diagSkipExts are extensions never worth a diagnostics round-trip: prose and
// lockfiles, where no language server has anything to say. Everything else is
// tried — a server that doesn't handle the language returns an error or nothing
// and we fall through to "".
var diagSkipExts = map[string]bool{
	".md": true, ".txt": true, ".rst": true, ".adoc": true,
	".lock": true, ".sum": true, ".csv": true, ".svg": true,
	".png": true, ".jpg": true, ".jpeg": true, ".gif": true, ".pdf": true,
}

// postWriteDiagnostics runs the wired language server over a just-written file
// and returns the text to append to the write's tool result, or "" when there
// is nothing to say. Never returns an error: a diagnostics failure must not
// turn a successful write into a failed tool call.
func (a *agent) postWriteDiagnostics(ctx context.Context, sid, path string) string {
	if !a.diagnosticsEnabled() {
		return ""
	}
	if diagSkipExts[strings.ToLower(filepath.Ext(path))] {
		return ""
	}
	sess := a.getSession(sid)
	if sess == nil {
		return ""
	}

	client, tool, ok := a.findDiagnosticsTool(path)
	if !ok {
		return ""
	}

	rel, err := filepath.Rel(sess.Cwd, path)
	if err != nil {
		rel = path
	}
	args, ok := diagArgs(tool.InputSchema, path, rel, sess.Cwd)
	if !ok {
		// The server exposes a diagnostics tool whose schema has no argument we
		// recognise as "the file". Log it rather than dropping it: this is how a
		// new language-server bridge with an unfamiliar schema surfaces.
		slog.Debug("diagnostics: no usable path argument in tool schema", "server", client.name, "tool", tool.Name)
		return ""
	}
	raw, err := json.Marshal(args)
	if err != nil {
		slog.Debug("diagnostics: marshalling args failed", "tool", tool.Name, "err", err)
		return ""
	}

	callCtx, cancel := context.WithTimeout(ctx, diagTimeout)
	defer cancel()
	out, isErr, err := client.callTool(callCtx, tool.Name, raw)
	switch {
	case err != nil:
		slog.Debug("diagnostics: call failed", "server", client.name, "tool", tool.Name, "path", path, "err", err)
		return ""
	case isErr:
		// The server rejected the call — commonly "not a Go file" when gopls is
		// asked about something outside its languages. Expected, not a fault.
		slog.Debug("diagnostics: server reported an error", "server", client.name, "tool", tool.Name, "path", path, "out", truncate(out, 200))
		return ""
	}

	if !diagHasFindings(out) {
		return ""
	}
	a.logSession(sid, "DIAGNOSTICS", "%s__%s on %s:\n%s", client.name, tool.Name, rel, out)
	return fmt.Sprintf("\n\n[%s reports on %s — these are from the file you just wrote, fix them before moving on]\n%s",
		client.name, rel, truncate(strings.TrimSpace(out), diagMaxBytes))
}

// diagnosticsEnabled reports the `diagnostics` settings key (default on).
func (a *agent) diagnosticsEnabled() bool {
	a.cfgMu.RLock()
	defer a.cfgMu.RUnlock()
	return a.settings.Diagnostics == nil || *a.settings.Diagnostics
}

// findDiagnosticsTool picks the language server to ask about path. Selection is
// by tool NAME, not by server name or a configured language map, because the
// name is the one thing every bridge agrees on: gopls exposes `go_diagnostics`,
// lsmcp exposes `lsp_get_diagnostics`, and a future bridge will call it
// something containing "diagnostic" too.
//
// A name that starts with a language token (`go_`) also gates on extension, so
// a project with both gopls and lsmcp wired doesn't ask gopls about a .ts file.
// Bridges with a generic name (`lsp_`, `get_`) are language-agnostic and are
// tried for anything.
func (a *agent) findDiagnosticsTool(path string) (*MCPClient, mcpTool, bool) {
	a.mu.Lock()
	clients := make([]*MCPClient, 0, len(a.mcp.clients))
	for _, c := range a.mcp.clients {
		clients = append(clients, c)
	}
	a.mu.Unlock()

	ext := strings.ToLower(filepath.Ext(path))
	var generic *MCPClient
	var genericTool mcpTool
	for _, c := range clients {
		for _, t := range c.tools {
			if !strings.Contains(strings.ToLower(t.Name), "diagnostic") {
				continue
			}
			if want, ok := diagToolLangExt[langPrefix(t.Name)]; ok {
				// Language-specific tool: exact match wins immediately.
				if want[ext] {
					return c, t, true
				}
				continue
			}
			if generic == nil {
				generic, genericTool = c, t
			}
		}
	}
	if generic != nil {
		return generic, genericTool, true
	}
	return nil, mcpTool{}, false
}

// langPrefix returns the token before the first underscore, lowercased —
// "go_diagnostics" → "go". Used only to look up diagToolLangExt.
func langPrefix(name string) string {
	if i := strings.Index(name, "_"); i > 0 {
		return strings.ToLower(name[:i])
	}
	return ""
}

// diagToolLangExt maps a language-specific diagnostics-tool prefix to the
// extensions that tool can actually answer for. A prefix absent from this map
// is treated as a generic bridge (tried for any file). Only prefixes we know to
// be language-scoped belong here; guessing wrong costs a wasted round-trip.
var diagToolLangExt = map[string]map[string]bool{
	"go":     {".go": true},
	"rust":   {".rs": true},
	"clangd": {".c": true, ".h": true, ".cc": true, ".cpp": true, ".hpp": true, ".cxx": true},
}

// diagPathArgs lists the argument names a diagnostics tool might use for "the
// file", in the order we prefer them, paired with how to render the path.
// Checked against the tool's declared input schema so we send what that
// specific server asks for instead of guessing one convention.
var diagPathArgs = []struct {
	key  string
	kind string // "abs" | "rel" | "uri"
}{
	{"files", "abs"},
	{"paths", "abs"},
	{"relativePath", "rel"},
	{"relative_path", "rel"},
	{"filePath", "abs"},
	{"file_path", "abs"},
	{"file", "abs"},
	{"path", "abs"},
	{"uri", "uri"},
	{"textDocument", "uri"},
}

// diagArgs builds the call arguments for a diagnostics tool from its declared
// JSON schema: find the first property that names the file, fill it in the
// shape the schema declares (array vs string), and add a workspace root if the
// schema wants one. Returns ok=false when no property looks like a file path,
// which is the signal to skip this server rather than send a call that will
// certainly fail.
func diagArgs(schema map[string]any, abs, rel, root string) (map[string]any, bool) {
	props, _ := schema["properties"].(map[string]any)
	if len(props) == 0 {
		return nil, false
	}
	args := map[string]any{}
	filled := false
	for _, cand := range diagPathArgs {
		spec, ok := props[cand.key].(map[string]any)
		if !ok {
			continue
		}
		declared, _ := spec["type"].(string)
		if declared == "object" {
			// A structured argument we'd have to know the inner shape of — LSP's own
			// `textDocument` is `{uri}`, not a string. Filling it with a path would
			// produce a call that is certain to be rejected, so treat it as if the
			// property weren't there and keep looking for a plain one.
			continue
		}
		var val string
		switch cand.kind {
		case "rel":
			val = rel
		case "uri":
			val = "file://" + abs
		default:
			val = abs
		}
		if declared == "array" {
			args[cand.key] = []string{val}
		} else {
			args[cand.key] = val
		}
		filled = true
		break
	}
	if !filled {
		return nil, false
	}
	// Bridges that serve several workspaces need to be told which one. Only set
	// these when the schema declares them — an unexpected property makes a
	// strict server reject the whole call.
	for _, k := range []string{"root", "rootPath", "root_path"} {
		if _, ok := props[k]; ok {
			args[k] = root
			break
		}
	}
	if _, ok := props["rootUri"]; ok {
		args["rootUri"] = "file://" + root
	}
	return args, true
}

// diagHasFindings reports whether the server's response is worth showing. A
// clean file is the common case and must cost zero tokens, but servers spell it
// differently: empty output, an empty JSON collection, or a sentence saying so.
func diagHasFindings(out string) bool {
	s := strings.TrimSpace(out)
	switch s {
	case "", "[]", "{}", "null":
		return false
	}
	lower := strings.ToLower(s)
	for _, phrase := range []string{"no diagnostics", "no errors", "no issues", "0 diagnostics"} {
		if strings.Contains(lower, phrase) {
			return false
		}
	}
	return true
}
