package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/BurntSushi/toml"

	"github.com/tbocek/codehalter/acp"
	"github.com/tbocek/codehalter/mcp"
)

// registerMCPTools registers the given tools into codehalter's tool
// registry, prefixed with `<server>__` to avoid collisions across servers.
// The tool's description and JSON schema flow through verbatim — the MCP
// server is the source of truth for both. Caller must have already fetched
// the list via ListTools so any startup failure is observed before
// registration (avoiding partial-state if tools/list errors mid-flight).
func (a *agent) registerMCPTools(c *mcp.Client, tools []mcp.Tool) {
	for _, t := range tools {
		fullName := c.Name + "__" + t.Name
		description := t.Description
		if description == "" {
			description = "(no description provided by MCP server " + c.Name + ")"
		}
		params := t.InputSchema
		if params == nil {
			params = map[string]any{"type": "object"}
		}
		client := c
		remoteName := t.Name
		a.tools.add(Tool{
			Def: map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        fullName,
					"description": description,
					"parameters":  params,
				},
			},
			Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
				tcId := a.StartToolCall(ctx, sid, fullName, "search", nil)
				output, isErr, err := client.CallTool(ctx, remoteName, json.RawMessage(rawArgs))
				if err != nil {
					a.FailToolCall(ctx, sid, tcId, err.Error())
					return "error: " + err.Error(), false
				}
				if isErr {
					a.FailToolCall(ctx, sid, tcId, output)
					return output, true
				}
				a.CompleteToolCall(ctx, sid, tcId, []acp.ToolCallContent{acp.TextContent(output)})
				return output, false
			},
		})
	}
	slog.Info("mcp tools registered", "server", c.Name, "count", len(tools))
}

// ---------------------------------------------------------------------------
// Lifecycle wired into the agent
// ---------------------------------------------------------------------------

// mcpChange describes one outcome of a reconciliation pass. The reconciler
// turns these into tool-call cards in the chat so the user sees additions,
// removals, restarts, and failures distinctly.
type mcpChange struct {
	action string // "started" | "stopped" | "restarted" | "failed" | "parse_error"
	name   string // server name; "" for parse_error
	err    error  // populated when action == "failed" or "parse_error"
	tools  int    // tools the server advertised; "started"/"restarted" only
}

// schedule runs `run` in the background, at most one at a time. A request that
// arrives while a run is in flight is coalesced into exactly ONE follow-up run,
// however many arrive: the file is read fresh at the top of each run, so a
// single catch-up pass sees the latest state. A request that arrives when idle
// starts immediately.
//
// This is the whole "apply MCP changes at a quiescent point" rule: callers
// schedule from the turn boundary and never call reconcileMCP mid-turn.
func (m *mcpState) schedule(run func()) {
	m.flushMu.Lock()
	if m.flushing {
		// Already one more queued → nothing to add; the queued pass will read
		// the same file this one would have.
		m.flushPending = true
		m.flushMu.Unlock()
		return
	}
	m.flushing = true
	done := make(chan struct{})
	m.flushDone = done
	m.flushMu.Unlock()

	go func() {
		defer close(done)
		for {
			run()
			m.flushMu.Lock()
			if !m.flushPending {
				m.flushing = false
				m.flushDone = nil
				m.flushMu.Unlock()
				return
			}
			m.flushPending = false
			m.flushMu.Unlock()
		}
	}()
}

// wait blocks until no scheduled flush is in flight, so a turn never starts
// while the tool registry is being rewritten. A no-op when idle, which is the
// common case: the flush from the previous turn's end has long finished by the
// time the user types again.
func (m *mcpState) wait() {
	m.flushMu.Lock()
	done := m.flushDone
	m.flushMu.Unlock()
	if done != nil {
		<-done
	}
}

// takeFixes empties the cards a background flush left behind (see flushFixes).
// takePending hands over what background flushes have parked, exactly once.
func (m *mcpState) takePending() (notes []string, fixes []fixProblem) {
	m.flushMu.Lock()
	defer m.flushMu.Unlock()
	notes, fixes = m.flushNotes, m.flushFixes
	m.flushNotes, m.flushFixes = nil, nil
	return notes, fixes
}

// shutdownMCP closes every running MCP child on app exit. stdio servers are
// spawned with exec.Command (no context), so without this they orphan and keep
// running after codehalter is gone. Snapshot under the lock (don't race reconcile),
// then Close unlocked with an overall deadline so a wedged server can't hang exit.
func (a *agent) shutdownMCP() {
	a.mcp.mu.Lock()
	clients := make([]*mcp.Client, 0, len(a.mcp.clients))
	for _, c := range a.mcp.clients {
		clients = append(clients, c)
	}
	a.mcp.clients = nil
	a.mcp.mu.Unlock()
	if len(clients) == 0 {
		return
	}
	done := make(chan struct{})
	go func() {
		for _, c := range clients {
			c.Close()
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		slog.Warn("mcp shutdown: timed out closing clients")
	}
}

// ---------------------------------------------------------------------------
// Importing the editor's own MCP servers
// ---------------------------------------------------------------------------

// mcpConfigPath is the single file MCP is configured from.
func mcpConfigPath(cwd string) string { return filepath.Join(cwd, sessionDir, "mcp.toml") }

// elicitMCPKey is the one multi-select field offerMCPImport asks for. The name
// is arbitrary but must match between the requested schema and the lookup in
// the response content.
const elicitMCPKey = "servers"

// offerMCPImport asks which of the editor's own MCP servers to adopt, and
// writes the answer into .codehalter/mcp.toml. session/new (and session/load)
// carry the list the user configured in Zed; codehalter used to drop it on the
// floor, because MCP here is file-driven, so those servers were simply
// invisible with no hint that they existed.
//
// Every offered server is written either way: chosen ones as live entries, the
// rest commented out. That is what makes this a one-time question. The next
// session finds the name already in the file and stays quiet, and the user
// enables one later by deleting a '#', which is the enable/disable convention
// the file already documents (see mcp.ServerConfig).
func (a *agent) offerMCPImport(ctx context.Context, cwd, sid string) {
	sess := a.getSession(sid)
	if sess == nil || len(sess.mcpOffer) == 0 {
		return
	}
	path := mcpConfigPath(cwd)
	raw, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		slog.Warn("mcp import: reading config", "path", path, "err", err)
		return
	}
	var fresh []acp.MCPServer
	for _, s := range sess.mcpOffer {
		// SSE is the one transport we can't run (mcpTransport does stdio and
		// Streamable HTTP), and Initialize doesn't advertise it, so a
		// spec-following client never sends one. Skip rather than write an
		// entry the reconciler would then fail to start.
		if s.Name == "" || s.Type == "sse" || mcpNameInFile(string(raw), s.Name) {
			continue
		}
		fresh = append(fresh, s)
	}
	if len(fresh) == 0 {
		return
	}

	chosen := a.askMCPImport(ctx, sid, fresh)
	var body strings.Builder
	var added []string
	for _, s := range fresh {
		enabled := chosen[s.Name]
		if enabled {
			added = append(added, s.Name)
		}
		body.WriteString(mcpTOMLEntry(s, !enabled))
	}
	// The session dir normally exists by now (initSession scaffolds it), but a
	// session that has never saved may not have one, and a failed MkdirAll would
	// otherwise surface as a confusing "no such file" from the append.
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		slog.Warn("mcp import: creating config dir", "path", path, "err", err)
		return
	}
	if err := appendFile(path, body.String()); err != nil {
		slog.Warn("mcp import: writing config", "path", path, "err", err)
		return
	}

	msg := fmt.Sprintf("Wrote %d MCP server(s) from your editor's settings into .codehalter/mcp.toml, commented out. Uncomment one to enable it.", len(fresh))
	if len(added) > 0 {
		msg = fmt.Sprintf("Added %s to .codehalter/mcp.toml. Starting with your next message.", strings.Join(added, ", "))
		if rest := len(fresh) - len(added); rest > 0 {
			msg += fmt.Sprintf(" The other %d are in the file commented out.", rest)
		}
	}
	a.say(ctx, sid, msg+"\n\n")
}

// askMCPImport puts the offered servers up as one multi-select form and returns
// the picked names. A client with no elicitation support, a declined form or a
// transport error all mean "none": the servers still get recorded (commented
// out), so nothing is lost and the question isn't repeated.
func (a *agent) askMCPImport(ctx context.Context, sid string, fresh []acp.MCPServer) map[string]bool {
	if a.conn == nil || !a.clientCan("elicitation") {
		return nil
	}
	options := make([]map[string]any, 0, len(fresh))
	for _, s := range fresh {
		summary := "stdio " + strings.TrimSpace(s.Command+" "+strings.Join(s.Args, " "))
		if s.URL != "" {
			summary = "http " + s.URL
		}
		options = append(options, map[string]any{"const": s.Name, "title": s.Name + " — " + summary})
	}
	raw, err := a.conn.SendRequest(ctx, "elicitation/create", map[string]any{
		"sessionId": sid,
		"mode":      "form",
		"message": "Your editor is configured with MCP servers codehalter isn't using yet. " +
			"Pick the ones to add to .codehalter/mcp.toml. The rest are written commented out, so you won't be asked again.",
		"requestedSchema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				elicitMCPKey: map[string]any{
					"type":  "array",
					"title": "MCP servers to enable",
					"items": map[string]any{"anyOf": options},
				},
			},
		},
	})
	if err != nil {
		slog.Warn("mcp import: elicitation failed", "err", err)
		return nil
	}
	// Content values are a union (string, number, bool, string array), so this
	// decodes into any and type-asserts rather than map[string]string.
	var resp struct {
		Action  string         `json:"action"`
		Content map[string]any `json:"content"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		slog.Warn("mcp import: undecodable elicitation response", "err", err)
		return nil
	}
	if resp.Action != "accept" {
		return nil
	}
	picked := map[string]bool{}
	values, _ := resp.Content[elicitMCPKey].([]any)
	for _, v := range values {
		if name, ok := v.(string); ok {
			picked[name] = true
		}
	}
	return picked
}

// mcpNameInFile reports whether mcp.toml already mentions a server by this
// name, INCLUDING inside a comment. Commented-out entries are how a declined
// server is remembered, so a parse of the live entries alone would re-ask every
// session.
func mcpNameInFile(raw, name string) bool {
	quoted := strconv.Quote(name)
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(strings.TrimLeft(strings.TrimSpace(line), "#"))
		key, value, ok := strings.Cut(line, "=")
		if ok && strings.TrimSpace(key) == "name" && strings.TrimSpace(value) == quoted {
			return true
		}
	}
	return false
}

// mcpTOMLEntry renders one server as a [[server]] block, optionally with every
// line commented out.
func mcpTOMLEntry(s acp.MCPServer, commented bool) string {
	var b strings.Builder
	b.WriteString("[[server]]\n")
	fmt.Fprintf(&b, "name = %q\n", s.Name)
	if s.URL != "" {
		fmt.Fprintf(&b, "url = %q\n", s.URL)
		if t := tomlInlineTable(s.Headers); t != "" {
			fmt.Fprintf(&b, "headers = %s\n", t)
		}
	} else {
		fmt.Fprintf(&b, "command = %q\n", s.Command)
		if len(s.Args) > 0 {
			quoted := make([]string, len(s.Args))
			for i, arg := range s.Args {
				quoted[i] = strconv.Quote(arg)
			}
			fmt.Fprintf(&b, "args = [%s]\n", strings.Join(quoted, ", "))
		}
		if t := tomlInlineTable(s.Env); t != "" {
			fmt.Fprintf(&b, "env = %s\n", t)
		}
	}
	if !commented {
		return "\n" + b.String()
	}
	var out strings.Builder
	out.WriteString("\n# Offered by the editor, not enabled. Uncomment to use.\n")
	for _, line := range strings.Split(strings.TrimRight(b.String(), "\n"), "\n") {
		fmt.Fprintf(&out, "# %s\n", line)
	}
	return out.String()
}

// tomlInlineTable renders ACP's [{name, value}] list as a TOML inline table.
// Header names contain '-', which is a legal TOML bare key, so only genuinely
// odd keys get quoted.
func tomlInlineTable(kv []acp.NameValue) string {
	if len(kv) == 0 {
		return ""
	}
	parts := make([]string, 0, len(kv))
	for _, e := range kv {
		parts = append(parts, tomlKey(e.Name)+" = "+strconv.Quote(e.Value))
	}
	return "{ " + strings.Join(parts, ", ") + " }"
}

func tomlKey(k string) string {
	if k == "" {
		return `""`
	}
	for _, r := range k {
		bare := r == '-' || r == '_' ||
			(r >= '0' && r <= '9') || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
		if !bare {
			return strconv.Quote(k)
		}
	}
	return k
}

// appendFile appends to path, creating it if absent. Close is checked: it's
// where a deferred write actually fails.
func appendFile(path, body string) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.WriteString(body); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// reconcileMCP brings the running MCP clients in line with .codehalter/mcp.toml.
// It is idempotent and cheap to call at every turn boundary: if the file is
// unchanged at the semantic level, no UI is emitted. Failures don't block the
// caller — the user's turn proceeds with whatever set of tools is currently
// registered.
//
// Callers are the two halves of the boundary, checkMCP (before a turn) and
// flushMCP (after one), never a running turn: registering tools rewrites the
// `tools` array the whole conversation is rendered behind.
//
// Restart semantics are start-then-stop: a config change brings up the new
// process first and only kills the old one after the new client is verified
// (initialize + tools/list both succeeded). That way a typo in args doesn't
// take down a working server.
func (a *agent) reconcileMCP(ctx context.Context, cwd string) []mcpChange {
	a.mcp.mu.Lock()
	defer a.mcp.mu.Unlock()

	// Read .codehalter/mcp.toml (MCP is opt-in, so a missing file is silently
	// fine). mtime lets the unchanged-file check below skip the diff so a
	// persistent start failure doesn't re-emit the same failed card every turn.
	path := mcpConfigPath(cwd)
	var cfgs []mcp.ServerConfig
	var mtime time.Time
	var err error
	if info, serr := os.Stat(path); serr == nil {
		mtime = info.ModTime()
		var f struct {
			Server []mcp.ServerConfig `toml:"server"`
		}
		if _, derr := toml.DecodeFile(path, &f); derr != nil {
			err = fmt.Errorf("loading %s: %w", path, derr)
		} else {
			cfgs = f.Server
		}
	} else if !os.IsNotExist(serr) {
		err = serr
	}
	if err != nil {
		// Parse errors are reported once per mtime change. If the user's
		// editor saved a half-written file at t0, we surface it once; if
		// they don't touch it again, we don't keep nagging on every prompt.
		if !mtime.IsZero() && mtime.Equal(a.mcp.appliedMtime) {
			return nil
		}
		a.mcp.appliedMtime = mtime
		return []mcpChange{{action: "parse_error", err: err}}
	}
	// File unchanged since last reconcile — skip the diff entirely. This
	// also suppresses re-emitting a failed-start card every turn when the
	// user has a server configured incorrectly; they have to actually edit
	// the file (which bumps mtime) to trigger another attempt.
	if !mtime.IsZero() && mtime.Equal(a.mcp.appliedMtime) {
		return nil
	}
	a.mcp.appliedMtime = mtime

	// Last-write-wins on duplicate names. The mcp.toml schema doesn't define
	// behavior here, and the user probably meant the second entry to override.
	desired := make(map[string]mcp.ServerConfig, len(cfgs))
	for _, c := range cfgs {
		if c.Name == "" {
			continue
		}
		if c.Command == "" && c.URL == "" {
			continue
		}
		desired[c.Name] = c
	}

	applied := make(map[string]mcp.ServerConfig, len(a.mcp.applied))
	for _, c := range a.mcp.applied {
		applied[c.Name] = c
	}

	var changes []mcpChange

	// Pass 1: start brand-new + restart changed. Start-then-stop, so we
	// verify the new client works before tearing down the old one.
	for name, want := range desired {
		old, existed := applied[name]
		// Unchanged in every field that affects runtime behavior → no-op.
		if existed && old.Command == want.Command && old.URL == want.URL &&
			slices.Equal(old.Args, want.Args) &&
			maps.Equal(old.Env, want.Env) && maps.Equal(old.Headers, want.Headers) {
			continue
		}

		// Bound the start+handshake+listTools: a stdio server that never answers
		// (npx still fetching, a broken --bin, a wedged process) must NOT hang the
		// prompt's prepare phase forever (the stdio transport has no client timeout
		// like httpTransport does). On timeout it's recorded as "failed" and the
		// prompt proceeds without that server's tools.
		startCtx, cancel := context.WithTimeout(ctx, mcp.StartTimeout)
		newClient, err := mcp.Start(startCtx, want, cwd)
		if err != nil {
			cancel()
			changes = append(changes, mcpChange{action: "failed", name: name, err: err})
			continue
		}
		tools, err := newClient.ListTools(startCtx)
		cancel()
		if err != nil {
			newClient.Close()
			changes = append(changes, mcpChange{action: "failed", name: name, err: fmt.Errorf("tools/list: %w", err)})
			continue
		}
		// Enforce the modern-HTTP x-mcp-header contract (and precompute the
		// Mcp-Param-* mirrors) before the list is retained or registered.
		tools = newClient.VetTools(tools)

		// Retain the advertised tool list on the client before publishing it (see
		// mcp.Client.Tools). Set here, pre-publication, so it is written while no
		// other goroutine can reach the client.
		newClient.Tools = tools

		// New client is ready. Atomically swap: unregister old tools, register
		// new tools, replace the client handle, close the old client.
		a.mu.Lock()
		if a.mcp.clients == nil {
			a.mcp.clients = make(map[string]*mcp.Client)
		}
		oldClient := a.mcp.clients[name]
		a.mcp.clients[name] = newClient
		a.mu.Unlock()

		if oldClient != nil {
			a.tools.removePrefix(name + "__")
		}
		a.registerMCPTools(newClient, tools)
		if oldClient != nil {
			oldClient.Close()
			changes = append(changes, mcpChange{action: "restarted", name: name, tools: len(tools)})
		} else {
			changes = append(changes, mcpChange{action: "started", name: name, tools: len(tools)})
		}
	}

	// Pass 2: stop entries that disappeared from the file (or were disabled).
	for name := range applied {
		if _, stillWanted := desired[name]; stillWanted {
			continue
		}
		a.mu.Lock()
		oldClient := a.mcp.clients[name]
		delete(a.mcp.clients, name)
		a.mu.Unlock()

		a.tools.removePrefix(name + "__")
		if oldClient != nil {
			oldClient.Close()
		}
		changes = append(changes, mcpChange{action: "stopped", name: name})
	}

	// Snapshot the running set for the next diff. Only entries that actually
	// started go in — a server that failed to start stays out, so once the user
	// fixes the file (bumping mtime) the next reconcile sees it as "missing" and
	// retries the start. desired is already the validated, deduped set.
	a.mcp.applied = a.mcp.applied[:0]
	for name, c := range desired {
		a.mu.Lock()
		_, running := a.mcp.clients[name]
		a.mu.Unlock()
		if running {
			a.mcp.applied = append(a.mcp.applied, c)
		}
	}

	return changes
}
