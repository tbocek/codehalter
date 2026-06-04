package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// fixProblem
// ---------------------------------------------------------------------------

// fixProblem describes a missing tool or environmental gap that prepare
// detected and wants to offer the user a one-click fix for. The desc is
// the human-readable banner line; the prompt is the synthetic user
// message dispatched through orchestrate when the user accepts the card.
type fixProblem struct {
	desc   string
	prompt string
}

// stackProbeBinary returns the binary whose presence on PATH means the
// stack's required dev tooling is installed. "" means no probe known —
// checkEnv ignores that stack. One binary per stack by design: gopls
// stands in for Go because installing it implicitly requires the toolchain.
func stackProbeBinary(stack string) string {
	switch stack {
	case "go":
		return "gopls"
	}
	return ""
}

// detectRunnerConfigs returns runner-kind names purely from config-file
// presence in cwd, regardless of whether the runner binary is installed.
// Used by checkEnv to flag "user has a justfile but `just` isn't on PATH"
// so the consolidated install card can offer to install the missing tool.
// Order is deterministic so envSnapshot diffs don't false-positive on
// reordering. go.mod is included unconditionally (the user wants to run
// `go vet` / `go test` even when a justfile or Makefile is also present —
// missing `go` blocks the executor's self-verify recipe regardless).
func detectRunnerConfigs(cwd string) []string {
	var kinds []string
	for _, name := range []string{"justfile", "Justfile", ".justfile"} {
		if _, err := os.Stat(filepath.Join(cwd, name)); err == nil {
			kinds = append(kinds, "just")
			break
		}
	}
	for _, name := range []string{"Makefile", "makefile", "GNUmakefile"} {
		if _, err := os.Stat(filepath.Join(cwd, name)); err == nil {
			kinds = append(kinds, "make")
			break
		}
	}
	if _, err := os.Stat(filepath.Join(cwd, "package.json")); err == nil {
		kinds = append(kinds, "npm")
	}
	if _, err := os.Stat(filepath.Join(cwd, "Cargo.toml")); err == nil {
		kinds = append(kinds, "cargo")
	}
	if _, err := os.Stat(filepath.Join(cwd, "go.mod")); err == nil {
		kinds = append(kinds, "go")
	}
	return kinds
}

// runnerProbeBinary returns the binary that must be on PATH for a runner
// kind to be usable. Mirrors stackProbeBinary's shape — kept as a switch
// rather than a map so adding a new runner is one obvious place to edit.
func runnerProbeBinary(kind string) string {
	switch kind {
	case "just":
		return "just"
	case "make":
		return "make"
	case "npm":
		return "npm"
	case "cargo":
		return "cargo"
	case "go":
		return "go"
	}
	return ""
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

// prepareChecks runs the pre-turn freshness pass: re-advertise the slash-macro
// menu, re-verify a reachable LLM (looping on a Retry card when not), refresh
// the per-session environment snapshot (stacks, container, firefox,
// run_command, per-stack probe binaries), and reconcile .codehalter/mcp.toml so
// the turn runs against current config. Each check short-circuits on an
// unchanged settings hash / env snapshot / mcp.toml mtime, so a steady-state
// turn pays almost nothing. Returns the missing-tool / broken-MCP problems for
// drainFixes to offer AFTER the turn — surfacing a fix card here, in front of
// the user's prompt, would let an accepted card dispatch a whole orchestrate
// cycle ahead of the request it interrupted.
func (a *agent) prepareChecks(ctx context.Context, sess *Session, sid string) []fixProblem {
	if sess == nil {
		slog.Debug("prepareChecks: nil sess, skipping")
		return nil
	}
	slog.Debug("prepareChecks: start", "sid", sid, "cwd", sess.Cwd)
	a.sendAvailableCommands(ctx, sid) // re-advertise the slash-macro menu each turn
	llmChanged := a.ensureLLM(ctx, sess, sid)
	envChanged, envProblems := a.checkEnv(sess, sid)
	mcpChanged, mcpProblems := a.checkMCP(ctx, sess, sid)
	// Full capabilities banner: emit it ONCE per session (the first prepare,
	// at bootstrap, when state is established — so it never fires mid-session).
	// After that, suppress the re-dump on routine changes — a tool getting
	// installed, an MCP server starting, a re-probe — those are surfaced as
	// one-line notices (checkMCP) or fix cards (drainFixes), not by re-printing
	// the whole setup screen in the middle of an unrelated turn. It stays here
	// with the checks because it reads the change flags they produce.
	if !sess.capabilitiesShown && (llmChanged || envChanged || mcpChanged) {
		a.notifyCapabilities(ctx, sess, sid)
		sess.capabilitiesShown = true
	}
	slog.Debug("prepareChecks: done", "sid", sid, "hasLLM", a.hasReachableLLM(), "stacks", sess.knownStacks, "envChanged", envChanged, "mcpChanged", mcpChanged, "llmChanged", llmChanged)
	return append(envProblems, mcpProblems...)
}

// drainFixes offers each problem prepareChecks detected as a one-click "fix it
// for me?" ack card. Runs post-turn so an accepted fix dispatches its synthetic
// prompt through the normal plan/execute/verify/document phases AFTER the user's
// actual request, not ahead of it.
func (a *agent) drainFixes(ctx context.Context, sid string, fixes []fixProblem) {
	for _, p := range fixes {
		if ctx.Err() != nil {
			break
		}
		a.proposeFix(ctx, sid, p)
	}
}

// prepare runs the full pre-flight — checks then fix cards — in one shot. Used
// at bootstrap, where there is no surrounding turn to split it across; the
// per-Prompt path calls prepareChecks pre-turn and drainFixes post-turn instead.
func (a *agent) prepare(ctx context.Context, sess *Session, sid string) {
	a.drainFixes(ctx, sid, a.prepareChecks(ctx, sess, sid))
}

// ---------------------------------------------------------------------------
// ensureLLM — settings load, probe, Retry-card loop
// ---------------------------------------------------------------------------

// minSlotTokens is the smallest per-slot n_ctx codehalter accepts at startup.
// Below this, even a single turn's system prompt + skills + a normal-sized
// reply doesn't leave room for a useful compaction tail, so the agent refuses
// to run and surfaces a Retry card the same way an unreachable LLM does.
const minSlotTokens = 32 * 1024

// minTotalSlots is the smallest total slot count (summed across every [[llm]]
// entry's parallelCap) codehalter accepts at startup. Compaction folds older
// turns into Summary via the background summariser, which must run on a slot
// separate from the foreground turn to keep the foreground's prefix cache
// warm. Below 2 slots there is nowhere to run it and compaction has no path.
const minTotalSlots = 2

// totalSlots returns the slot count summed across every [[llm]] entry as of the
// last probeAllLLMs (which back-fills auto-detected total_slots into each conn's
// parallelCap before summing into a.detectedSlots). It reads the stored value
// rather than recomputing live because ensureLLM resets settings.Parallel via
// loadSettings each pass — a live sum would see the post-reset default before
// the next probe and defeat the "nothing changed, skip the probe" short-circuit.
// 0 before the first probe. Used by ensureLLM to gate startup on having a slot
// for the background summariser separate from the foreground turn.
func (a *agent) totalSlots() int {
	return a.detectedSlots
}

// ensureLLM blocks until every startup gate passes: at least one [[llm]]
// connection answers a probe, llm[0] reports a per-slot context window of at
// least minSlotTokens, AND the configured slot count totals at least
// minTotalSlots (so the background summariser has somewhere to run separate
// from the foreground). First call scaffolds .codehalter/settings.toml if
// neither the global nor the project-local file exists. The probe is skipped
// when the merged settings-file hash matches sess.llmHash AND every gate is
// already satisfied — i.e. the file the user could have edited hasn't
// actually changed AND we know we have a working route. A single "Retry"
// tool card is shown when any gate fails; clicking it always re-probes
// regardless of hash (the user may have changed network or launch settings
// outside the file).
//
// Returns true when something observable changed since the last call
// (new hash, new reachability, or n_ctx newly discovered) — prepare uses
// this to decide whether to re-emit the consolidated capabilities banner.
// Steady-state turns short-circuit and return false.
//
// There is no Abort: codehalter cannot function without an LLM. In auto-
// answer modes (autopilot, subagents) we cap retries at 3 to avoid
// spinning forever — those callers handle "no LLM" gracefully via
// connForSession.
func (a *agent) ensureLLM(ctx context.Context, sess *Session, sid string) bool {
	auto, _ := a.shouldAutoAnswer(sid)
	const autoCap = 3
	prevHash := sess.llmHash
	ready := func() bool {
		return a.hasReachableLLM() && a.mainSlotTokens >= minSlotTokens && a.totalSlots() >= minTotalSlots
	}
	prevReady := ready()
	forceRetry := false
	for attempt := 0; ; attempt++ {
		if loaded, err := loadSettings(sess.Cwd); err == nil {
			a.settings = loaded
			a.buildConnSems()
		}
		if a.settings.path == "" {
			a.scaffoldSettings(ctx, sess.Cwd, sid)
			if loaded, err := loadSettings(sess.Cwd); err == nil {
				a.settings = loaded
				a.buildConnSems()
			}
		}
		currentHash := hashSettingsFiles(sess.Cwd)
		if !forceRetry && currentHash != "" && currentHash == sess.llmHash && ready() {
			return false
		}
		a.probeAllLLMs(ctx)
		sess.llmHash = currentHash
		if ready() {
			return sess.llmHash != prevHash || !prevReady
		}
		if auto && attempt >= autoCap-1 {
			return sess.llmHash != prevHash || prevReady
		}
		var msg string
		switch {
		case !a.hasReachableLLM():
			msg = "LLM not reachable — edit settings.toml, then click Retry"
		case a.mainSlotTokens == 0:
			msg = "LLM reachable but server didn't report a context size (n_ctx). Codehalter needs this to size compaction safely. Restart your server with the context size set on the launch command (llama.cpp: `-c N`, vLLM: `--max-model-len N`, llama-server: ensure /props is enabled), then click Retry."
		case a.mainSlotTokens < minSlotTokens:
			msg = fmt.Sprintf("LLM reachable but per-slot context window is only %d tokens — codehalter requires at least %d. Restart your server with a larger `-c N` (llama.cpp) / `--max-model-len N` (vLLM), or reduce the `parallel` slot count in settings.toml, then click Retry.", a.mainSlotTokens, minSlotTokens)
		default:
			msg = fmt.Sprintf("LLM reachable but only %d total slot(s) — codehalter requires at least %d so the background summariser can run separate from the foreground turn. Set `parallel = 2` (or more) on your [[llm]] entry, or add a second [[llm]] entry, then click Retry.", a.totalSlots(), minTotalSlots)
		}
		tcId, err := a.askAcknowledgeWithCard(ctx, sid, msg, "think", "Retry")
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return sess.llmHash != prevHash || prevReady
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Retrying LLM probe")})
		forceRetry = true
	}
}

// scaffoldSettings writes .codehalter/settings.toml with the embedded
// placeholder template and prints a short hint to chat. No-op when a
// settings file already exists. The placeholder won't reach any real
// server — the next probe will fail and the Retry card explains the
// situation.
func (a *agent) scaffoldSettings(ctx context.Context, cwd string, sid string) {
	path := filepath.Join(cwd, sessionDir, "settings.toml")
	if _, err := os.Stat(path); err == nil {
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "Failed to create " + filepath.Dir(path) + ": " + err.Error() + "\n"}})
		return
	}
	if err := os.WriteFile(path, []byte(defaultSettingsTOML), 0o644); err != nil {
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "Failed to write " + path + ": " + err.Error() + "\n"}})
		return
	}
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "Wrote " + path + " with placeholder values. Edit `server` and `model` to match your LLM server, then click Retry below. Optional: move the edited file to ~/.config/codehalter/settings.toml to share it across every project.\n\n"}})
}

// hashSettingsFiles returns hex sha256 of the concatenated contents of the
// global and project-local settings files. Either missing file contributes
// no bytes; "" only when both are absent.
func hashSettingsFiles(cwd string) string {
	h := sha256.New()
	written := false
	if home, err := os.UserHomeDir(); err == nil {
		if data, err := os.ReadFile(filepath.Join(home, ".config", "codehalter", "settings.toml")); err == nil {
			h.Write(data)
			written = true
		}
	}
	h.Write([]byte{0})
	if data, err := os.ReadFile(filepath.Join(cwd, sessionDir, "settings.toml")); err == nil {
		h.Write(data)
		written = true
	}
	if !written {
		return ""
	}
	return hex.EncodeToString(h.Sum(nil))
}

// hasReachableLLM returns true when the last probe found at least one
// answering [[llm]] entry.
func (a *agent) hasReachableLLM() bool {
	for _, r := range a.connReachable {
		if r {
			return true
		}
	}
	return false
}

// probeAllLLMs probes every configured [[llm]] in parallel and updates
// a.connReachable, a.mainSlotTokens, and a.imagesSupported. Config values
// (context_size / image_support on [[llm]]) take precedence over probe
// discovery — the probe is the auto-detect shortcut for llama.cpp/llama-swap;
// every other backend (OpenAI, Ollama, vLLM, LiteLLM, …) configures
// explicitly. The human-readable status is rendered separately by
// renderLLMStatus so the consolidated banner can diff and re-emit when
// state changes.
func (a *agent) probeAllLLMs(ctx context.Context) {
	conns := a.settings.allConnections()
	a.connReachable = make(map[string]bool, len(conns))
	a.mainSlotTokens = 0
	a.detectedSlots = 0
	if len(conns) == 0 {
		a.imagesSupported = false
		return
	}
	results := make([]probeResult, len(conns))
	parallel(len(conns), len(conns), func(i int) {
		c := conns[i]
		results[i] = a.probeLLM(ctx, &c)
	})
	// Record reachability and auto-detect the slot count: when an [[llm]] left
	// `parallel` unset, adopt llama.cpp's reported total_slots (-np) so connSems,
	// totalSlots, the summariser's separate-slot gate, and subagent pinning all
	// see real server capacity without the user declaring it. An explicit
	// `parallel` always wins. Re-detected each probe — ensureLLM reloads settings
	// (resetting Parallel to the file value) right before calling us.
	for i := range conns {
		a.connReachable[conns[i].Server+"\x00"+conns[i].Model] = results[i].Reachable
		if a.settings.LLM[i].Parallel == 0 && results[i].TotalSlots > 0 {
			a.settings.LLM[i].Parallel = results[i].TotalSlots
		}
	}
	a.buildConnSems() // resize the per-conn semaphores to the back-filled caps

	// Persist the total slot count (across all entries) so ensureLLM's pre-probe
	// short-circuit reads the real number after loadSettings has reset Parallel.
	a.detectedSlots = 0
	for i := range a.settings.LLM {
		a.detectedSlots += a.settings.LLM[i].parallelCap()
	}

	// LLM[0] owns the foreground session's KV cache, so its per-slot context
	// window drives compaction sizing. Prefer the server's directly-reported
	// per-slot n_ctx (no division, robust to the total ÷ -np split); else divide
	// a known total — an explicit context_size (the total the user declared) or
	// /v1/models' -c launch arg — by the slot count.
	slots := a.settings.LLM[0].parallelCap()
	switch {
	case conns[0].ContextSize > 0:
		a.mainSlotTokens = conns[0].ContextSize / slots
	case results[0].SlotCtx > 0:
		a.mainSlotTokens = results[0].SlotCtx
	case results[0].ContextSize > 0:
		a.mainSlotTokens = results[0].ContextSize / slots
	}
	slog.Info("probeAllLLMs", "slots", slots, "mainSlotTokens", a.mainSlotTokens,
		"slotCtx", results[0].SlotCtx, "totalCtx", results[0].ContextSize, "totalSlots", results[0].TotalSlots)
	// Image support is a property of LLM[0] alone: the foreground model is the
	// only image consumer (view_image and the execute loop route there), and ACP
	// advertises a single agent-wide image capability. Explicit config wins over
	// probe discovery; a down LLM[0] with no declared value falls back to false.
	switch {
	case conns[0].ImageSupport != nil:
		a.imagesSupported = *conns[0].ImageSupport
	case results[0].Reachable:
		a.imagesSupported = results[0].ImageSupport
	default:
		a.imagesSupported = false
	}
}

// renderLLMStatus formats the LLM probe results into the chat summary used
// by notifyCapabilities. Pure function over agent state — produces the same
// string until probeAllLLMs or settings changes.
func (a *agent) renderLLMStatus() string {
	conns := a.settings.allConnections()
	var b strings.Builder
	if len(conns) == 0 {
		b.WriteString("🟡 LLM: no [[llm]] in settings.toml — codehalter cannot run until you add one.\n\n")
		return b.String()
	}
	if settingsLooksPlaceholder(a.settings) {
		fmt.Fprintf(&b, "🟡 LLM: %s still has the placeholder model \"your-model-id\". Edit it with your real url and model, then click Retry below.\n\n", a.settings.path)
		return b.String()
	}
	firstReachable := -1
	for i := range conns {
		c := conns[i]
		label := fmt.Sprintf("llm[%d]", i)
		if i > 0 && c.Tag != "" {
			label += " " + c.Tag
		}
		if !a.connReachable[c.Server+"\x00"+c.Model] {
			fmt.Fprintf(&b, "🟡 %s: unreachable at %s — start your server or fix the server url.\n\n", label, c.Server)
			continue
		}
		fmt.Fprintf(&b, "✅ %s: %s @ %s (parallel=%d)\n\n", label, c.Model, c.Server, c.parallelCap())
		if firstReachable < 0 {
			firstReachable = i
		}
	}
	if firstReachable < 0 {
		b.WriteString("🟡 No LLM reachable — every connection above failed. Codehalter cannot run any prompt until at least one comes back.\n\n")
		return b.String()
	}
	switch {
	case a.imagesSupported:
		b.WriteString("✅ Image support: enabled\n\n")
	case conns[0].ImageSupport != nil:
		b.WriteString("Image support: disabled (declared image_support = false in settings.toml)\n\n")
	default:
		b.WriteString("❕ Image support: undetected — codehalter assumed disabled. If your model accepts images, set `image_support = true` on the [[llm]] entry in settings.toml.\n\n")
	}
	switch {
	case a.mainSlotTokens == 0:
		b.WriteString("🟡 Context window: unknown — set `context_size = N` on the [[llm]] entry in settings.toml. For llama.cpp/vLLM you can also restart with the launch flag (`-c N` / `--max-model-len N`) so the probe discovers it.\n\n")
	case a.mainSlotTokens < minSlotTokens:
		fmt.Fprintf(&b, "🟡 Context window: only %d tokens/slot — codehalter requires at least %d. Raise `context_size` in settings.toml, increase your server's launch flag (`-c N` / `--max-model-len N`), or reduce `parallel`.\n\n", a.mainSlotTokens, minSlotTokens)
	default:
		trigger := a.mainSlotTokens * compactTriggerPct / 100
		if pc := conns[0].parallelCap(); pc > 1 {
			fmt.Fprintf(&b, "✅ Context window: %d tokens/slot (n_ctx %d ÷ %d slots, compact at %d)\n\n", a.mainSlotTokens, a.mainSlotTokens*pc, pc, trigger)
		} else {
			fmt.Fprintf(&b, "✅ Context window: %d tokens (compact at %d)\n\n", a.mainSlotTokens, trigger)
		}
	}
	if total := a.totalSlots(); total < minTotalSlots {
		fmt.Fprintf(&b, "🟡 Slots: only %d configured — codehalter requires at least %d so the background summariser has a slot separate from the foreground. Set `parallel = 2` on your [[llm]] entry or add a second [[llm]].\n\n", total, minTotalSlots)
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// checkEnv — stacks, container, firefox, run_command, per-stack probes
// ---------------------------------------------------------------------------

// envSnapshot builds the canonical string that represents the entire
// environment as currently observable. Two calls return identical strings
// iff nothing the user can see in the capabilities banner has changed.
// Stack and runner-config lists are taken in their detection functions'
// fixed orders so reordering can't false-positive a diff.
func (a *agent) envSnapshot(sess *Session) string {
	var b strings.Builder
	fmt.Fprintf(&b, "container=%s\n", containerKind())
	if _, err := findFirefox(); err == nil {
		b.WriteString("firefox=ok\n")
	} else {
		b.WriteString("firefox=missing\n")
	}
	// run_command availability is fully determined by container= above (it's
	// registered iff in a container), so it needs no separate snapshot line.
	fmt.Fprintf(&b, "stacks=%s\n", strings.Join(sess.knownStacks, ","))
	for _, s := range sess.knownStacks {
		bin := stackProbeBinary(s)
		if bin == "" {
			continue
		}
		if _, err := exec.LookPath(bin); err == nil {
			fmt.Fprintf(&b, "tool[%s]=ok\n", bin)
		} else {
			fmt.Fprintf(&b, "tool[%s]=missing\n", bin)
		}
	}
	fmt.Fprintf(&b, "runners=%s\n", strings.Join(sess.knownRunners, ","))
	for _, k := range sess.knownRunners {
		bin := runnerProbeBinary(k)
		if bin == "" {
			continue
		}
		if _, err := exec.LookPath(bin); err == nil {
			fmt.Fprintf(&b, "runner[%s]=ok\n", bin)
		} else {
			fmt.Fprintf(&b, "runner[%s]=missing\n", bin)
		}
	}
	return b.String()
}

// checkEnv refreshes sess.knownStacks and sess.knownRunners, probes the
// environment (container, firefox, run_command, per-stack dev-tool
// binaries, runner-config binaries on PATH), and reports whether anything
// is different from sess.envSnapshot. All missing binaries are collapsed
// into ONE fixProblem so the user sees a single "Install fix? gopls,
// make, just" card instead of one card per tool. Strictly silent — emits
// no chat output of its own. Bash and devcontainer are filtered out of
// knownStacks because they're meta-tooling, not stacks.
func (a *agent) checkEnv(sess *Session, sid string) (bool, []fixProblem) {
	var stacks []string
	for _, s := range detectStacks(sess.Cwd) {
		if s == "bash" || s == "devcontainer" {
			continue
		}
		stacks = append(stacks, s)
	}
	sess.knownStacks = stacks
	sess.knownRunners = detectRunnerConfigs(sess.Cwd)

	// Seed SKILL-*.md for a stack / runner config / distro added since session
	// start so its skill is on disk before the next turn's systemPrompt loads
	// it. Keyed on FILE presence (justfile / Makefile / go.mod / ...), not on
	// installed tools — the LLM needs the skill to know how to install the
	// missing tool. ensureSkills seeds each skill once (leaving existing copies)
	// and prunes other-OS copies.
	osi := readOSInfo()
	if err := ensureSkills(sess.Cwd, sess.knownStacks, osi); err != nil {
		slog.Warn("prepare: ensureSkills failed", "sid", sid, "err", err)
	}
	// Rebuild the system prompt every turn but assign only on an actual byte
	// diff. An unchanged prompt keeps the LLM's KV prefix cache warm; a skill
	// added/edited/removed on disk this turn — whether codehalter-managed or a
	// SKILL-*.md the user hand-dropped — gets picked up for the next LLM call
	// (including proposeFix's orchestrate below) instead of waiting for a new
	// session.
	if sp, err := a.systemPrompt(sid); err != nil {
		slog.Warn("prepare: systemPrompt rebuild failed", "sid", sid, "err", err)
	} else if sp != sess.SystemPrompt {
		sess.SystemPrompt = sp
	}

	snap := a.envSnapshot(sess)
	changed := snap != sess.envSnapshot
	sess.envSnapshot = snap

	// Build the "gopls (go stack), make (Makefile), …" detail as we find each
	// missing probe binary — one consolidated fixProblem covers them all.
	var detail strings.Builder
	note := func(bin, reason string) {
		if detail.Len() > 0 {
			detail.WriteString(", ")
		}
		fmt.Fprintf(&detail, "%s (%s)", bin, reason)
	}
	for _, s := range stacks {
		if bin := stackProbeBinary(s); bin != "" {
			if _, err := exec.LookPath(bin); err != nil {
				note(bin, s+" stack")
			}
		}
	}
	for _, k := range sess.knownRunners {
		if bin := runnerProbeBinary(k); bin != "" {
			if _, err := exec.LookPath(bin); err != nil {
				note(bin, k+" runner")
			}
		}
	}
	if detail.Len() == 0 {
		return changed, nil
	}
	// Embed the OS we already detected so the LLM doesn't waste a tool
	// call rediscovering it. The bootstrap step (ensureDevcontainer) only
	// scaffolds containers based on one of the five supported distros, so
	// osi.ID is a non-empty supported value by the time prepare runs.
	distro := osi.Fields["PRETTY_NAME"]
	if distro == "" {
		distro = strings.ToUpper(osi.ID[:1]) + osi.ID[1:]
	}
	return changed, []fixProblem{{
		desc: fmt.Sprintf("🟡 Missing dev tools: %s", detail.String()),
		prompt: fmt.Sprintf("Missing dev tools in this %s devcontainer: %s. "+
			"PLAN ONLY — do not install anything yourself. Produce execute-phase "+
			"steps in this order for each tool: (1) install via the OS package "+
			"manager, (2) verify it runs (e.g. `<tool> --version`), (3) persist "+
			"by editing .devcontainer/Dockerfile, (4) if the tool is MCP-capable "+
			"(e.g. gopls via `gopls mcp`), add a [[server]] entry to "+
			".codehalter/mcp.toml. Verify-phase checks: each tool on PATH, "+
			"Dockerfile contains the persist line.",
			distro, detail.String()),
	}}
}

// ---------------------------------------------------------------------------
// checkMCP — mtime-gated reconcile + parse/start fix proposals
// ---------------------------------------------------------------------------

// checkMCP runs the per-prompt mcp.toml reconcile (mtime-gated inside
// reconcileMCP) and converts parse_error / failed changes into fixProblem
// proposals so prepare can offer the user a one-click "fix the file"
// prompt. Returns changed=true when reconcileMCP reported anything at all
// (started / stopped / restarted / failed / parse_error) so prepare re-
// emits the consolidated banner that now reflects the new MCP state. Also
// folds in cleanupGitCommitIfClean — same per-prompt cadence, no point
// running it from anywhere else.
func (a *agent) checkMCP(ctx context.Context, sess *Session, sid string) (bool, []fixProblem) {
	changes := a.reconcileMCP(ctx, sess.Cwd)
	a.cleanupGitCommitIfClean(sess.Cwd, sid)
	if len(changes) == 0 {
		return false, nil
	}
	var problems []fixProblem
	var notices []string
	for _, ch := range changes {
		switch ch.action {
		case "parse_error":
			problems = append(problems, fixProblem{
				desc: fmt.Sprintf("🟡 .codehalter/mcp.toml parse error: %s", ch.err),
				prompt: fmt.Sprintf("The MCP configuration file .codehalter/mcp.toml failed to parse: %s. "+
					"Read the file, fix the syntax error, and confirm it parses cleanly by re-reading it. "+
					"Do not start any servers — codehalter will pick up the fix on the next prompt.", ch.err),
			})
		case "failed":
			problems = append(problems, fixProblem{
				desc: fmt.Sprintf("🟡 MCP server %q failed to start: %s", ch.name, ch.err),
				prompt: fmt.Sprintf("The MCP server %q in .codehalter/mcp.toml failed to start: %s. "+
					"Inspect the [[server]] entry, confirm the command is on PATH, and check any required "+
					"args / env. If the server's binary is missing, install it via run_command, verify it "+
					"runs by hand, then persist the install in .devcontainer/Dockerfile so it survives a "+
					"container rebuild.", ch.name, ch.err),
			})
		case "started":
			notices = append(notices, fmt.Sprintf("✅ MCP server %q started", ch.name))
		case "restarted":
			notices = append(notices, fmt.Sprintf("✅ MCP server %q restarted", ch.name))
		case "stopped":
			notices = append(notices, fmt.Sprintf("MCP server %q stopped", ch.name))
		}
	}
	// Benign starts/stops/restarts get a one-line notice, NOT a re-dump of the
	// whole capabilities banner — that full re-emit on a routine server start
	// (e.g. gopls coming up the turn after it was added) was pure noise. Only
	// an actionable problem (failed / parse_error) forces the consolidated
	// banner, via the changed=true return below.
	for _, n := range notices {
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: n + "\n"}})
	}
	return len(problems) > 0, problems
}

// ---------------------------------------------------------------------------
// notifyCapabilities — consolidated banner
// ---------------------------------------------------------------------------

// notifyCapabilities renders ONE consolidated banner covering everything
// the user needs to see at the top of a turn: settings.toml path, LLM
// status, project tooling (runners), detected stacks, container, firefox,
// run_command, MCP servers, and per-stack dev-tool probes (✅ found / 🟡
// missing). Called by prepare only when one of ensureLLM / checkEnv /
// checkMCP reported a change since the last turn. Steady-state turns
// emit nothing.
func (a *agent) notifyCapabilities(ctx context.Context, sess *Session, sid string) {
	var b strings.Builder

	if a.settings.path != "" {
		fmt.Fprintf(&b, "Using %s\n\n", a.settings.path)
	}
	b.WriteString(a.renderLLMStatus())

	// Sandbox / browser / shell — these gate everything else, so they
	// belong above project-specific state.
	if kind := containerKind(); kind != "" {
		fmt.Fprintf(&b, "✅ Container: %s\n\n", kind)
	} else {
		b.WriteString("🟡 Container: none (running on host — file edits and tasks hit your real filesystem)\n\n")
	}
	if _, err := findFirefox(); err == nil {
		b.WriteString("✅ Firefox: found (web_search/web_read enabled)\n\n")
	} else {
		b.WriteString("🟡 Firefox: not found — web_search/web_read disabled. Install firefox or set FIREFOX_PATH.\n\n")
	}
	// Reaching this banner means the session is actually running, which only
	// happens inside a container (ensureDevcontainer aborts otherwise), so
	// run_command is always registered by discoverSandbox at this point.
	b.WriteString("✅ run_command: available (probes and test installs; `.git` is bind-mounted read-only — destructive git commands fail at the FS layer)\n\n")

	// Stacks paired with their probe binary so the user sees stack →
	// required-binary status on consecutive lines.
	if len(sess.knownStacks) > 0 {
		fmt.Fprintf(&b, "Stacks: %s", strings.Join(sess.knownStacks, ", "))
		if len(sess.knownStacks) > 1 {
			b.WriteString(" (monorepo)")
		}
		b.WriteString("\n\n")
		for _, s := range sess.knownStacks {
			bin := stackProbeBinary(s)
			if bin == "" {
				continue
			}
			if _, err := exec.LookPath(bin); err == nil {
				fmt.Fprintf(&b, "✅ %s: found (%s stack)\n\n", bin, s)
			} else {
				fmt.Fprintf(&b, "🟡 %s: not on PATH — required for the %s stack\n\n", bin, s)
			}
		}
	}

	// Task-runner block. Three states:
	//   - empty project → bootstrap hint
	//   - no runner config at all → "add one" nudge
	//   - one or more runner configs detected → list populated runners + per-
	//     kind 🟡 lines for any kind whose binary is missing.
	a.mu.Lock()
	caps := a.capabilities
	empty := a.emptyProject
	a.mu.Unlock()

	if empty {
		b.WriteString("Empty project — I'll ask about language and runner on your first message.\n\n")
	} else if len(sess.knownRunners) == 0 {
		b.WriteString("🟡 No task runner detected (just, make, npm, go, cargo). Add one so I can build/test/lint.\n\n")
	} else {
		if len(caps.runners) > 0 {
			fmt.Fprintf(&b, "Project tooling (%s):\n", strings.Join(caps.runners, ", "))
			row := func(label string, entries []string, hint string) {
				if len(entries) > 0 {
					fmt.Fprintf(&b, "  %-7s %s\n", label+":", strings.Join(entries, ", "))
				} else {
					fmt.Fprintf(&b, "  %-7s (none — %s)\n", label+":", hint)
				}
			}
			row("build", caps.build, "consider adding a `build` target")
			row("test", caps.test, "consider adding a `test` target")
			row("lint", caps.lint, "consider adding a `lint`/`vet`/`check` target")
			row("format", caps.format, "consider adding a `fmt`/`format` target")
			b.WriteString("\n")
		}
		for _, k := range sess.knownRunners {
			bin := runnerProbeBinary(k)
			if bin == "" {
				continue
			}
			if _, err := exec.LookPath(bin); err != nil {
				fmt.Fprintf(&b, "🟡 %s config detected but `%s` not on PATH\n\n", k, bin)
			}
		}
	}

	if skills := listSkills(sess.Cwd); len(skills) > 0 {
		fmt.Fprintf(&b, "🧠 Skills: %s\n\n", strings.Join(skills, ", "))
	}

	a.mu.Lock()
	var mcpRunning []string
	for name := range a.mcp.clients {
		mcpRunning = append(mcpRunning, name)
	}
	a.mu.Unlock()
	if len(mcpRunning) > 0 {
		sort.Strings(mcpRunning)
		fmt.Fprintf(&b, "✅ MCP: %s\n\n", strings.Join(mcpRunning, ", "))
	}

	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: b.String()}})
}

// ---------------------------------------------------------------------------
// proposeFix — ack card + synthetic prompt → orchestrate
// ---------------------------------------------------------------------------

// proposeFix shows the user a single yes/no card carrying the problem
// description. On accept we synthesise a user message with the proposed
// prompt and dispatch it through orchestrate exactly as if the user had
// typed it — full plan / per-subtask self-verifying loop / document phases,
// real tool calls, real cards. Skip just closes the card; the problem stays
// visible in the banner so the user can address it manually whenever they
// choose.
func (a *agent) proposeFix(ctx context.Context, sid string, p fixProblem) {
	title := p.desc + " — install fix?"
	ok, tcId, err := a.askYesNoWithCard(ctx, sid, title, "think", "Install fix", "Skip")
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}
	if !ok {
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Skipped — fix it manually when convenient")})
		return
	}
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Dispatching: " + p.prompt)})
	sess := a.getSession(sid)
	if sess == nil {
		return
	}
	sess.AddUser(p.prompt)
	sess.saveOrLog()
	if _, err := a.orchestrate(ctx, sid); err != nil {
		// A cancelled fix dispatch is routine (user stopped it); a real failure
		// is not — surface it at Warn so a fix that silently never ran is
		// visible in the log rather than buried at Debug.
		if isCancelled(err) {
			slog.Debug("proposeFix: fix dispatch cancelled", "sid", sid, "err", err)
		} else {
			slog.Warn("proposeFix: fix dispatch failed", "sid", sid, "err", err)
		}
	}
}
