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
	"slices"
	"sort"
	"strings"
)

// Fix-card prompts: what the executor is sent when the user accepts a 🟡 card.
// Each is a THIN trigger, since the how-to lives in the SKILL it points at.
// Inline rather than res/*.md because nothing reads them at runtime: a file
// would add no editability, only distance between the format string and the
// fmt.Sprintf whose verbs must match it.
const (
	// cardSetupHeader opens the combined setup card; every cardSetup* line below
	// is appended to it as one bullet. Split this way so that N problems produce
	// ONE prompt carrying one PLAN ONLY directive, instead of N prompts that each
	// repeat it and each cost a full plan/execute/document cycle. They are one job
	// anyway: the pkg-mgr command that installs the missing tools is the same one
	// that provides the language server the wiring bullet then points at.
	cardSetupHeader = "Container setup needed in this %s devcontainer.\n" +
		"\n" +
		"PLAN ONLY → produce execute-phase steps covering every item below, then PERSIST every install in `.devcontainer/Dockerfile`. Follow SKILL-base.md (install order + install/persist loop).\n" +
		"\n"

	cardInstallTools = "- Missing dev tools: %s. Install each, verify each runs.\n"

	// cardFormatHeader is its own card rather than a bullet on cardSetupHeader:
	// nothing here is installed and nothing belongs in the Dockerfile, and the
	// work is judgement (read the code, infer the style it is already written in)
	// rather than a package-manager command. Holes: the formatter list, then the
	// config files to write.
	cardFormatHeader = "This project has no formatter config (%s), so every tool that touches it formats by its own defaults.\n" +
		"\n" +
		"That is not cosmetic. codehalter formats what it writes, the editor may format on save, CI may check a third way, and when they disagree a file is rewritten between the moment the model reads it and its next edit — so the edit fails on text that was correct when it was read.\n" +
		"\n" +
		"PLAN ONLY → produce execute-phase steps that:\n" +
		"1. MEASURE the style already in the repo. Do NOT impose defaults. Read several of the largest existing source files and count: indent width, tabs vs spaces, quote style, semicolons, trailing commas, the line width the code actually respects. State the numbers you measured.\n" +
		"2. Write %s encoding exactly those numbers, so the formatter is a no-op on code that already matches the project.\n" +
		"3. Prove it: run the formatter in check mode over the whole tree and report how many files it would still change. A large number means the config does not describe this codebase — go back to step 1 and fix the config. Do NOT reformat the repo to match a guess.\n" +
		"4. Only once step 3 is small: format the whole tree and commit that as ONE commit containing formatting and nothing else. If `git status` is not clean, skip this step and say so — a formatting commit must not sweep up someone's work in progress.\n" +
		"\n" +
		"SKILL-base.md (\"Formatter config\") has the exact config files and flags. Change no behavior anywhere in this task.\n"

	// cardAgentsFile writes the project brief codehalter loads into the system
	// prompt at session start. Two things make it worth a card of its own: the
	// facts are the user's to confirm, not the model's to guess, and the failure
	// mode of getting it wrong is a line that stays wrong for every later
	// session. No holes to fill.
	//
	// Step 3 is emphatic for a measured reason. An earlier draft asked for the
	// file "under 60 lines" and listed words it must not contain, and the model
	// read both as acceptance criteria: it wrote the file, then ran wc -l and a
	// grep over it and trimmed, 26 edits across 42 rounds and 12k generated
	// tokens for a 59-line file. A target a tool can measure is a target the
	// model will iterate against, so brevity is stated without a number.
	cardAgentsFile = "This project has no AGENT.md, so every session starts by rediscovering what the project is.\n" +
		"\n" +
		"PLAN ONLY → produce execute-phase steps that:\n" +
		"1. Read enough of the tree to answer for yourself: what this project IS in one sentence, its language and version, the frameworks and notable libraries, how it is built / run / tested, which directory holds what, and any convention the code plainly follows that no file states.\n" +
		"2. Use `ask_user` ONCE, as a single free-text box, to put those draft answers to the user: they correct what is wrong and add what cannot be read off the tree (who it is for, what it must never do, decisions already settled). Do not ask what the tree already answers.\n" +
		"3. Compose the whole file, write it in ONE `write_file` call, and stop. It is prose, not code: there is nothing to verify, so do not read it back, count its lines, grep it for words you were told to avoid, or edit it into shape a line at a time. Short enough to read in a minute, and you are the judge of that.\n" +
		"4. It holds ONLY what stays true between sessions: purpose, stack, layout, the build and test commands, standing constraints. NO task list, NO status, NO roadmap, NO \"currently working on\". A line that expires is worse than no line, because the next session believes it.\n" +
		"5. Name no command you have not run: a build or test line goes in only once it worked.\n" +
		"\n" +
		"codehalter folds AGENT.md into the system prompt at session start, so it is read once per session, not once per turn. Do not commit it; that is the user's call.\n"

	cardMCPParseError = "MCP config `.codehalter/mcp.toml` failed to parse: %s.\n" +
		"\n" +
		"Read file (header comments = schema), fix syntax, re-read to confirm parses. Do NOT start servers → codehalter reconciles next prompt.\n"

	cardMCPStartError = "MCP server %q in `.codehalter/mcp.toml` failed to start: %s.\n" +
		"\n" +
		"Inspect its `[[server]]` entry; confirm command on PATH + args/env right. Binary missing → install + persist in `.devcontainer/Dockerfile` (see SKILL-base.md).\n"
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

// projectIsEmpty reports the startup verdict: this agent opened a directory
// with nothing in it. Read from three places (the banner, the first user
// message, the AGENT.md card), all of them off the session goroutine.
func (a *agent) projectIsEmpty() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.emptyProject
}

// isEmptyProject reports whether cwd is a pristine directory: nothing in it
// but the .codehalter we create ourselves, if that. A hidden directory such as
// .git means the project exists and merely has no files here yet, which is not
// the same thing, so only .codehalter is ignored.
func isEmptyProject(cwd string) bool {
	entries, err := os.ReadDir(cwd)
	if err != nil {
		return false
	}
	for _, e := range entries {
		name := e.Name()
		if name == ".codehalter" {
			continue
		}
		if strings.HasPrefix(name, ".") && e.IsDir() {
			// Hidden dirs like .git/.idea don't count as "real" content, but
			// their presence means this isn't a pristine mkdir — bail out.
			return false
		}
		return false
	}
	return true
}

// emptyProjectHint is injected onto the first user turn when the working
// directory has no source files, manifests, or runner config. It tells the
// LLM to ask what language/framework the user wants before writing anything.
const emptyProjectHint = `[Note: this project directory is empty — no source files or build manifests were found. Before doing anything else, use the ask_user tool to confirm:
1. What language/framework should this project use? (Rust, Go, Node.js, Python, C, etc.)
2. Which build runner do they prefer? (Cargo, go modules, npm/pnpm, just, Make)

Only then create the appropriate skeleton — Cargo.toml for Rust, go.mod for Go, package.json for Node, justfile/Makefile otherwise — with sensible build/test/lint/format targets.]
`

// formatterNeed is a formatter this project would use (from a detected stack or
// a formatter config file) plus a human-readable reason, for the install card.
type formatterNeed struct {
	bin    string
	reason string
}

// detectFormatters returns the formatters defensive auto-formatting (format.go)
// would invoke here, EXCLUDING ones that ship with their language toolchain
// (gofmt, rustfmt, zig fmt — present whenever the language is). Derived from
// detected stacks AND formatter config files, so the install card can offer a
// missing one (e.g. prettier on a fresh TS repo, or ruff when pyproject pins it
// even though detectStacks doesn't model Python).
func detectFormatters(stacks []string, cwd string) []formatterNeed {
	var needs []formatterNeed
	seen := map[string]bool{}
	add := func(bin, reason string) {
		if seen[bin] {
			return
		}
		seen[bin] = true
		needs = append(needs, formatterNeed{bin, reason})
	}
	for _, s := range stacks {
		switch s {
		case "ts", "js":
			add("prettier", s+" stack")
		case "bash":
			add("shfmt", "bash stack")
		case "c":
			add("clang-format", "c stack")
		}
	}
	if hasPrettierConfig(cwd) {
		add("prettier", "prettier config")
	}
	if fileExists(cwd, ".clang-format") {
		add("clang-format", ".clang-format")
	}
	if pyprojectHasTable(cwd, "[tool.ruff") {
		add("ruff", "ruff config")
	}
	if pyprojectHasTable(cwd, "[tool.black]") {
		add("black", "black config")
	}
	return needs
}

func fileExists(cwd, name string) bool {
	_, err := os.Stat(filepath.Join(cwd, name))
	return err == nil
}

// formatterConfigNeed pairs a formatter that will run over this project with the
// config file that would pin HOW it formats.
type formatterConfigNeed struct {
	bin, config string
}

// formatterConfigNeeds returns the installed formatters that will reformat
// this project with nothing in the repo saying what its style is. gofmt,
// rustfmt and zig fmt are absent on purpose: they have no style options, so
// there is nothing to pin. A formatter with arbitrary defaults can drift from
// the code, and an editor's format-on-save then invalidates the text the model
// just read, which costs failed edits. A missing formatter is the install
// card's business and comes first.
func formatterConfigNeeds(stacks []string, cwd string) []formatterConfigNeed {
	var needs []formatterConfigNeed
	// .editorconfig counts as pinned for prettier, which reads it; clang-format
	// does not, so it gets no such exemption below.
	prettierUnpinned := prettierBin(cwd) != "" && !hasPrettierConfig(cwd) && !fileExists(cwd, ".editorconfig")
	if prettierUnpinned && (slices.Contains(stacks, "ts") || slices.Contains(stacks, "js") || slices.Contains(stacks, "css")) {
		needs = append(needs, formatterConfigNeed{"prettier", ".prettierrc"})
	}
	if slices.Contains(stacks, "c") && onPath("clang-format") && !fileExists(cwd, ".clang-format") {
		needs = append(needs, formatterConfigNeed{"clang-format", ".clang-format"})
	}
	return needs
}

// formatConfigEnabled reports the `format_config` settings key (default on).
func (a *agent) formatConfigEnabled() bool {
	a.cfgMu.RLock()
	defer a.cfgMu.RUnlock()
	return a.settings.FormatConfig == nil || *a.settings.FormatConfig
}

// hasPrettierConfig reports whether the project pins prettier — a dotfile, a
// prettier.config.*, or a "prettier" key in package.json.
func hasPrettierConfig(cwd string) bool {
	for _, n := range []string{
		".prettierrc", ".prettierrc.json", ".prettierrc.yaml", ".prettierrc.yml",
		".prettierrc.json5", ".prettierrc.js", ".prettierrc.cjs", ".prettierrc.mjs",
		".prettierrc.toml", "prettier.config.js", "prettier.config.cjs", "prettier.config.mjs",
	} {
		if fileExists(cwd, n) {
			return true
		}
	}
	if data, err := os.ReadFile(filepath.Join(cwd, "package.json")); err == nil {
		return strings.Contains(string(data), "\"prettier\"")
	}
	return false
}

// pyprojectHasTable reports whether cwd/pyproject.toml contains the given table
// header prefix ("[tool.ruff" matches both [tool.ruff] and [tool.ruff.lint]).
func pyprojectHasTable(cwd, prefix string) bool {
	data, err := os.ReadFile(filepath.Join(cwd, "pyproject.toml"))
	return err == nil && strings.Contains(string(data), prefix)
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

// prepareChecks is the pre-turn freshness pass: re-advertise the slash menu,
// re-verify a reachable LLM, refresh the environment snapshot and reconcile
// mcp.toml. Each check short-circuits on an unchanged hash, snapshot or mtime,
// so a steady turn pays almost nothing. It RETURNS the problems for drainFixes
// to offer after the turn: a card shown here could dispatch a whole orchestrate
// cycle ahead of the request it interrupted.
func (a *agent) prepareChecks(ctx context.Context, sess *Session, sid string) []fixProblem {
	if sess == nil {
		slog.Debug("prepareChecks: nil sess, skipping")
		return nil
	}
	slog.Debug("prepareChecks: start", "sid", sid, "cwd", sess.Cwd)
	a.sendAvailableCommands(ctx, sid) // re-advertise the slash-macro menu each turn
	a.ensureLLM(ctx, sess, sid)
	envProblems := a.checkEnv(sess, sid)
	mcpProblems := a.checkMCP(ctx, sess, sid)
	// Full capabilities banner: once per session, on the first prepare, and
	// unconditionally, so an unchanged setup does not leave the user staring at
	// an empty thread unable to tell setup from a hang. Later prepares stay
	// silent: mid-session changes surface as a one-line notice or a fix card.
	if !sess.capabilitiesShown {
		a.notifyCapabilities(ctx, sess, sid)
		sess.capabilitiesShown = true
		// After the banner, so the card lands under the line announcing it.
		a.offerSelfUpdate(ctx, sess, sid)
	}
	slog.Debug("prepareChecks: done", "sid", sid, "hasLLM", a.hasReachableLLM(), "stacks", sess.knownStacks)
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

// ---------------------------------------------------------------------------
// ensureLLM — settings load, probe, Retry-card loop
// ---------------------------------------------------------------------------

// minSlotTokens is the smallest per-slot n_ctx codehalter accepts at startup.
// Below this, even a single turn's system prompt + skills + a normal-sized
// reply doesn't leave room for a useful compaction tail, so the agent refuses
// to run and surfaces a Retry card the same way an unreachable LLM does.
const minSlotTokens = 32 * 1024

// ensureLLM blocks until the startup gates pass: some [[llm]] answers a probe
// and llm[0] reports at least minSlotTokens per slot. The first call scaffolds
// settings.toml when none exists. The probe is skipped while the settings hash
// is unchanged AND the gates already hold. On failure it shows a Retry card,
// which always re-probes. There is no Abort, since codehalter cannot work
// without an LLM; autopilot caps the retries at 3.
func (a *agent) ensureLLM(ctx context.Context, sess *Session, sid string) {
	auto := a.isAutopilot()
	const autoCap = 3
	ready := func() bool {
		return a.hasReachableLLM() && a.mainSlotTokens >= minSlotTokens
	}
	forceRetry := false
	// reload swaps in the settings on disk and reports whether a settings file
	// exists at all. A file that fails to parse keeps the previous settings.
	reload := func() bool {
		loaded, err := loadSettings(sess.Cwd)
		a.cfgMu.Lock()
		defer a.cfgMu.Unlock()
		if err == nil {
			a.settings = loaded
			a.buildConnSems()
		} else {
			slog.Warn("ensureLLM: keeping previous settings, reload failed", "err", err)
		}
		return a.settings.path != ""
	}
	for attempt := 0; ; attempt++ {
		if !reload() {
			a.scaffoldSettings(ctx, sess.Cwd, sid)
			reload()
		}
		currentHash := hashSettingsFiles(sess.Cwd)
		if !forceRetry && currentHash != "" && currentHash == sess.llmHash && ready() {
			return
		}
		// Every configured server gets probed here, each one a network round
		// trip that a local backend answers only once the model is resident:
		// the single slowest silent stretch of a session open.
		stopBeat := a.heartbeat(ctx, sid)
		a.probeAllLLMs(ctx)
		stopBeat()
		sess.llmHash = currentHash
		if ready() {
			return
		}
		if auto && attempt >= autoCap-1 {
			return
		}
		var msg string
		switch {
		case !a.hasReachableLLM():
			msg = "LLM not reachable — edit settings.toml, then click Retry"
		case a.mainSlotTokens == 0:
			// Name the exact probe URLs so the user can curl them and see why
			// n_ctx came back empty, and point at the context_size escape hatch
			// in settings.toml — the fix when the backend simply doesn't expose
			// it (OpenAI, Ollama, vLLM, …) and restarting the server won't help.
			probed := "GET /v1/models and GET /props"
			if len(a.settings.LLM) > 0 {
				c := &a.settings.LLM[0]
				probed = "GET " + c.endpoint("/v1/models") + " and GET " + c.endpoint("/props")
			}
			where := a.settings.path
			if where == "" {
				where = "your settings.toml"
			}
			msg = fmt.Sprintf("LLM reachable but neither metadata endpoint reported a context size (n_ctx) — codehalter probed %s. It needs the model's context window to size compaction safely. Fix it one of two ways: (1) set `context_size = N` (the model's max prompt+output tokens) on the [[llm]] entry in %s — use this when your backend doesn't expose n_ctx; or (2) restart your server with the size on the launch command (llama.cpp: `-c N`, vLLM: `--max-model-len N`, llama-server: ensure /props is enabled). Then click Retry.", probed, where)
		default:
			msg = fmt.Sprintf("LLM reachable but per-slot context window is only %d tokens — codehalter requires at least %d. Restart your server with a larger `-c N` (llama.cpp) / `--max-model-len N` (vLLM), or reduce the `parallel` slot count in settings.toml, then click Retry.", a.mainSlotTokens, minSlotTokens)
		}
		// One button and no decline path: codehalter cannot go on without an
		// LLM, so the only useful answer is "I fixed it, try again".
		_, tcId, err := a.askCard(ctx, sid, msg, "think", []permissionOption{{OptionId: "ack", Name: "Retry", Kind: "allow_once"}})
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return
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
		a.say(ctx, sid, "Failed to create "+filepath.Dir(path)+": "+err.Error()+"\n")
		return
	}
	if err := os.WriteFile(path, []byte(defaultSettingsTOML), 0o644); err != nil {
		a.say(ctx, sid, "Failed to write "+path+": "+err.Error()+"\n")
		return
	}
	// Read the file back before claiming success. On some devcontainer mounts a
	// WriteFile reports nil yet nothing persists (read-only overlay), or a
	// workspace-reset hook reaps it right away (.codehalter/ is gitignored, so a
	// `git clean -fdX` would). Only say "Wrote" for a file we can actually read;
	// otherwise name the likely cause instead of a misleading success message.
	if data, err := os.ReadFile(path); err != nil || len(data) == 0 {
		reason := "read back empty"
		if err != nil {
			reason = err.Error()
		}
		a.say(ctx, sid, "Wrote "+path+" but could not read it back ("+reason+"). The directory may be read-only or wiped by a reset hook (.codehalter/ is gitignored). Add a global ~/.config/codehalter/settings.toml instead.\n\n")
		return
	}
	// Keep settings.toml out of git regardless of whether the user tracks the
	// rest of .codehalter/ — it can hold an api_key.
	gitignoreNote := ""
	if ensureSettingsGitignored(cwd) {
		gitignoreNote = " It's listed in .gitignore so your api_key isn't committed."
	}
	a.say(ctx, sid, "Wrote "+path+" with placeholder values."+gitignoreNote+" Edit `server` and `model` to match your LLM server, then click Retry below. If it is not in your editor's file tree, refresh: agent-created files do not always show up live. Optional: move the edited file to ~/.config/codehalter/settings.toml to share it across every project.\n\n")
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
	for _, p := range a.connProbe {
		if p.Reachable {
			return true
		}
	}
	return false
}

// probeAllLLMs probes every configured [[llm]] in parallel and updates
// a.connProbe, a.mainSlotTokens, and a.imagesSupported. Config values
// (context_size / image_support on [[llm]]) take precedence over probe
// discovery — the probe is the auto-detect shortcut for llama.cpp/llama-swap;
// every other backend (OpenAI, Ollama, vLLM, LiteLLM, …) configures
// explicitly. The human-readable status is rendered separately by
// renderLLMStatus so the consolidated banner can diff and re-emit when
// state changes.
func (a *agent) probeAllLLMs(ctx context.Context) {
	conns := a.settings.allConnections()
	a.connProbe = make(map[string]probeResult, len(conns))
	a.setMainSlotTokens(0)
	if len(conns) == 0 {
		a.imagesSupported = false
		return
	}
	results := make([]probeResult, len(conns))
	parallel(len(conns), len(conns), func(i int) {
		c := conns[i]
		results[i] = probeLLM(ctx, &c)
	})
	// Record reachability and auto-detect the slot count: when an [[llm]] left
	// `parallel` unset, adopt llama.cpp's reported total_slots (-np) so connSems,
	// and the summariser's separate-slot gate both
	// see real server capacity without the user declaring it. An explicit
	// `parallel` always wins. Re-detected each probe — ensureLLM reloads settings
	// (resetting Parallel to the file value) right before calling us.
	for i := range conns {
		a.connProbe[conns[i].Server+"\x00"+conns[i].Model] = results[i]
	}
	// Back-fill the detected parallelism and resize the semaphores under cfgMu: a
	// prior turn's background LLM call may be reading a.settings.LLM / a.connSems
	// right now (connForBackgroundLLM / the slot gate).
	a.cfgMu.Lock()
	for i := range conns {
		if a.settings.LLM[i].Parallel == nil && results[i].TotalSlots > 0 {
			val := results[i].TotalSlots
			a.settings.LLM[i].Parallel = &val
		}
	}
	a.buildConnSems() // resize the per-conn semaphores to the back-filled caps
	a.cfgMu.Unlock()

	// LLM[0] owns the foreground session's KV cache, so its per-slot context
	// window drives compaction sizing. Prefer the server's directly-reported
	// per-slot n_ctx (no division, robust to the total ÷ -np split); else divide
	// a known total — an explicit context_size (the total the user declared) or
	// /v1/models' -c launch arg — by the slot count.
	slots := a.settings.LLM[0].parallelCap()
	switch {
	case conns[0].ContextSize != nil && *conns[0].ContextSize > 0:
		a.setMainSlotTokens(*conns[0].ContextSize / slots)
	case results[0].SlotCtx > 0:
		a.setMainSlotTokens(results[0].SlotCtx)
	case results[0].ContextSize > 0:
		a.setMainSlotTokens(results[0].ContextSize / slots)
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
	// Still the skeleton's placeholder: say "edit your settings.toml" rather than
	// a generic "unreachable" or "model not loaded".
	if len(a.settings.LLM) > 0 && a.settings.LLM[0].Model == "your-model-id" {
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
		if !a.connProbe[c.Server+"\x00"+c.Model].Reachable {
			fmt.Fprintf(&b, "🟡 %s: unreachable at %s — start your server or fix the server url.\n\n", label, c.Server)
			continue
		}
		// Reachable, but /v1/models answered without listing the configured id.
		// The connection works yet requests for this model often come back empty
		// (the gateway routes an unknown/unloaded name to nothing, returning a
		// clean 200 with no content). Said in the banner, because otherwise the
		// first sign is a turn that fails to parse.
		if pr := a.connProbe[c.Server+"\x00"+c.Model]; pr.ModelKnown && !pr.ModelLoaded {
			avail := "its model list came back empty"
			if len(pr.AvailableModels) > 0 {
				avail = "it offers: " + strings.Join(pr.AvailableModels, ", ")
			}
			fmt.Fprintf(&b, "🟡 %s: reachable at %s, but model `%s` isn't in its /v1/models list — requests may return an empty response. Check `model =` in settings.toml (%s). Harmless if your gateway lists models under different ids or doesn't enumerate them.\n\n", label, c.Server, c.Model, avail)
			if firstReachable < 0 {
				firstReachable = i
			}
			continue
		}
		fmt.Fprintf(&b, "✅ %s: %s @ %s (parallel=%d)\n\n", label, c.Model, c.Server, c.parallelCap())
		// The one settings mistake that costs real time and shows no symptom:
		// roles that differ in anything but samplers ask for two renderings, and
		// each phase switch then re-evaluates what the OTHER role appended
		// (measured: 99582 tokens across two switches). The rewind detector
		// reports it only after the tokens are spent, so say it up front.
		if think, exec := renderKey(c.paramsFor("thinking")), renderKey(c.paramsFor("execute")); think != exec {
			// "(none)" rather than an empty string: no template params at all is
			// the good configuration and should not read like missing data.
			show := func(k string) string {
				if k == "" {
					return "(none)"
				}
				return k
			}
			fmt.Fprintf(&b, "❕ %s: the two roles ask for different renderings — `params_thinking` %s vs `params_execute` %s. "+
				"Anything that is not a sampler is an argument to the chat template, so every plan ↔ execute switch re-evaluates "+
				"whatever the other role appended in between. Make them agree, or keep the split deliberately if you have priced it.\n\n",
				label, show(think), show(exec))
		}
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
		inputCap := a.mainSlotTokens * compactTriggerPct / 100
		if pc := conns[0].parallelCap(); pc > 1 {
			fmt.Fprintf(&b, "✅ Context window: %d tokens/slot (n_ctx %d ÷ %d slots, max prompt %d)\n\n", a.mainSlotTokens, a.mainSlotTokens*pc, pc, inputCap)
		} else {
			fmt.Fprintf(&b, "✅ Context window: %d tokens (max prompt %d)\n\n", a.mainSlotTokens, inputCap)
		}
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// checkEnv — stacks, container, firefox, run_command, per-stack probes
// ---------------------------------------------------------------------------

func onPath(bin string) bool { _, err := exec.LookPath(bin); return err == nil }

// hasFormatter reports whether the formatter f can actually run here. prettier
// is the exception: a JS project usually has it in node_modules rather than on
// PATH, so prettierBin looks there too.
func hasFormatter(cwd string, f formatterNeed) bool {
	if f.bin == "prettier" {
		return prettierBin(cwd) != ""
	}
	return onPath(f.bin)
}

// checkEnv refreshes sess.knownStacks, seeds any newly-applicable skill, and
// reports what this project is missing as fix cards: the formatters that would
// reformat it but are not installed, a formatter that runs with nothing pinning
// its style, and a project with no AGENT.md. Bash and devcontainer are filtered
// out of knownStacks: they are meta-tooling every project has, not stacks.
// Strictly silent, emitting no chat output of its own.
//
// There is no task-runner detection. Knowing a justfile exists told us only
// that `just` should be installed, which the model finds out anyway the first
// time a recipe fails, and the classification built on top of it (build / test /
// lint / format targets) was never read by anything but a banner line.
func (a *agent) checkEnv(sess *Session, sid string) []fixProblem {
	stacks := projectStacks(sess.Cwd)
	sess.knownStacks = stacks
	osi := readOSInfo()

	// The system prompt is the leading message of every request, so changing it
	// mid-session busts the LLM's KV prefix cache — which only compaction may do.
	// So: set it on the FIRST build; afterward, a skill that became applicable
	// this session (a justfile appeared, a stack was installed) is injected as a
	// user message for this turn instead — cache-safe, since it appends to the
	// tail. The next compaction re-renders the prompt (history.go) and is where
	// the skill finally enters the cached prefix. promptSkills tracks what the
	// current prompt already holds, so each new skill is injected exactly once.
	if sess.promptSkills == nil {
		sess.promptSkills = skillSet(sess.Cwd, stacks)
	}
	if sp, err := a.systemPrompt(sid); err != nil {
		slog.Warn("prepare: systemPrompt rebuild failed", "sid", sid, "err", err)
	} else if sess.SystemPrompt == "" {
		sess.SystemPrompt = sp
		sess.promptSkills = skillSet(sess.Cwd, stacks)
	} else if sp != sess.SystemPrompt {
		for _, name := range skillSet(sess.Cwd, stacks) {
			if slices.Contains(sess.promptSkills, name) {
				continue
			}
			if body := skillBody(sess.Cwd, name); body != "" {
				sess.AddUser("[New skill available this session — " + name +
					". It enters the system prompt at the next history compaction; until then it's here.]\n\n" + body)
				sess.promptSkills = append(sess.promptSkills, name)
			}
		}
	}

	// Build the "just (just runner), prettier (ts stack formatter), …" detail as we find each
	// missing probe binary — one consolidated fixProblem covers them all.
	var detail strings.Builder
	note := func(bin, reason string) {
		if detail.Len() > 0 {
			detail.WriteString(", ")
		}
		fmt.Fprintf(&detail, "%s (%s)", bin, reason)
	}
	// Formatters defensive auto-formatting would use: a missing one is the
	// whole content of the install card now that runners are not probed.
	for _, f := range detectFormatters(stacks, sess.Cwd) {
		if !hasFormatter(sess.Cwd, f) {
			note(f.bin, f.reason+" formatter")
		}
	}

	var probs []fixProblem
	if detail.Len() > 0 {
		// Embed the OS we already detected so the LLM doesn't waste a tool call
		// rediscovering it. The bootstrap step (ensureDevcontainer) only scaffolds
		// containers based on one of the five supported distros, so osi.ID normally
		// has a supported value here; the plain "Linux" fallback is for a container
		// the user built themselves with no usable /etc/os-release.
		distro := osi.Fields["PRETTY_NAME"]
		if distro == "" && osi.ID != "" {
			distro = strings.ToUpper(osi.ID[:1]) + osi.ID[1:]
		}
		if distro == "" {
			distro = "Linux"
		}
		probs = append(probs, fixProblem{
			desc:   "🟡 Container setup: install " + detail.String(),
			prompt: fmt.Sprintf(cardSetupHeader, distro) + fmt.Sprintf(cardInstallTools, detail.String()),
		})
	}

	// Formatter config: once per PROJECT, and never when it opted out with
	// format_config = false. A separate card from the container setup above, for
	// the reasons on cardFormatHeader.
	if !checkDone(sess.Cwd, checkFormatConfig) && a.formatConfigEnabled() {
		if needs := formatterConfigNeeds(stacks, sess.Cwd); len(needs) > 0 {
			markCheckDone(sess.Cwd, checkFormatConfig)
			bins := make([]string, len(needs))
			cfgs := make([]string, len(needs))
			for i, n := range needs {
				bins[i], cfgs[i] = n.bin, n.config
			}
			probs = append(probs, fixProblem{
				desc:   "🟡 No formatter config (" + strings.Join(bins, ", ") + "): pin the style this code is already written in?",
				prompt: fmt.Sprintf(cardFormatHeader, strings.Join(bins, ", "), strings.Join(cfgs, " + ")),
			})
		}
	}

	// The project brief. AGENT.md rides in the system prompt from session start
	// (loadAgentsFile), so writing it once is what stops every later session
	// spending its first turns working out what the project is.
	//
	// Only worth asking once there is something to read: the card's first step
	// is "read the tree", and a directory holding nothing but a .git has no
	// answers in it. Checked live and left unmarked below the threshold, so a
	// project scaffolded during this session is offered the card as soon as it
	// has a shape, rather than at the next session or never.
	//
	// The order of these three tests is load-bearing: checkDone is two syscalls
	// and is true forever after the first offer, so the walk below happens at
	// most once per project, and never at all for a project that ships a brief.
	if !checkDone(sess.Cwd, checkAgentsFile) {
		if _, content := loadAgentsFile(sess.Cwd); content == "" && len(listProjectFiles(sess.Cwd)) >= minFilesForBrief {
			markCheckDone(sess.Cwd, checkAgentsFile)
			probs = append(probs, fixProblem{
				desc:   "🟡 No AGENT.md: write down what this project is, so every session starts knowing it?",
				prompt: cardAgentsFile,
			})
		}
	}
	return probs
}

// minFilesForBrief is how many files (listProjectFiles rules: no .git, no
// .codehalter, no node_modules) a project needs before being asked for an
// AGENT.md. A bare scaffold has nothing to say that its own go.mod does not,
// and asking about it wastes the one time the card is offered.
const minFilesForBrief = 5

// Names recorded in checksDoneFile. Each gates a card that asks for work the
// project only ever needs once.
const (
	checkFormatConfig = "format-config"
	checkAgentsFile   = "agent-md"
)

// checksDoneFile records, per project, the one-time cards codehalter has
// already offered. Without it a declined card comes back at every session
// start, which nags rather than helps. One name per line: the only question
// asked of the file is whether a name is in it, and deleting a line is how the
// user re-arms that card.
const checksDoneFile = "checks.done"

func checkDone(cwd, name string) bool {
	data, err := os.ReadFile(filepath.Join(cwd, ".codehalter", checksDoneFile))
	return err == nil && slices.Contains(strings.Fields(string(data)), name)
}

// markCheckDone records name as offered. Failing to write it costs one repeat
// offer next session, which is not worth failing a turn over, so it warns.
func markCheckDone(cwd, name string) {
	if checkDone(cwd, name) {
		return
	}
	path := filepath.Join(cwd, ".codehalter", checksDoneFile)
	data, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		slog.Warn("reading the completed-checks file failed", "path", path, "err", err)
		return
	}
	if err := os.WriteFile(path, append(data, (name+"\n")...), 0o644); err != nil {
		slog.Warn("recording a completed check failed", "path", path, "err", err)
	}
}

// ---------------------------------------------------------------------------
// checkMCP — mtime-gated reconcile + parse/start fix proposals
// ---------------------------------------------------------------------------

// checkMCP is the pre-turn half of the reconcile. The turn-end flush
// (flushMCP) applies whatever the model changed during a turn, so by here there
// is usually nothing to do: wait out a flush still starting its child, say the
// notices and collect the cards it parked, and run one mtime-gated reconcile of
// its own. That
// last pass is what catches an mcp.toml edited by hand while no turn was
// running — nothing watches the file, so this stat is how such an edit is seen.
func (a *agent) checkMCP(ctx context.Context, sess *Session, sid string) []fixProblem {
	// Starting a stdio MCP child can take seconds before it
	// answers the handshake, and neither the wait nor reconcile says anything
	// until it is done.
	stopBeat := a.heartbeat(ctx, sid)
	a.mcp.wait()
	changes := a.reconcileMCP(ctx, sess.Cwd)
	stopBeat()

	notes, fixes := a.mcp.takePending()
	ownNotes, ownFixes := renderMCPChanges(changes)
	// Benign starts/stops/restarts get a one-line notice, NOT a re-dump of the
	// whole capabilities banner — that full re-emit on a routine server start
	// (a server coming up the turn after it was added) was pure noise. Said
	// here, from inside the turn, including the ones a background flush parked.
	for _, n := range append(notes, ownNotes...) {
		a.say(ctx, sid, n+"\n")
	}
	return append(fixes, ownFixes...)
}

// flushMCP is the turn-end half: apply the mcp.toml the just-finished turn may
// have written, now that no turn is in flight to have its prompt moved under
// it. Runs on a background context (the turn's is done, possibly cancelled) via
// mcpState.schedule, so a slow stdio start happens while the user reads the
// answer instead of in front of their next prompt. Its notices and cards are
// parked for the next checkMCP rather than said here, since nothing it emits
// belongs to a turn.
func (a *agent) flushMCP(sid string) {
	sess := a.getSession(sid)
	if sess == nil {
		return
	}
	notes, fixes := renderMCPChanges(a.reconcileMCP(context.Background(), sess.Cwd))
	if len(notes) == 0 && len(fixes) == 0 {
		return
	}
	a.mcp.flushMu.Lock()
	a.mcp.flushNotes = append(a.mcp.flushNotes, notes...)
	a.mcp.flushFixes = append(a.mcp.flushFixes, fixes...)
	a.mcp.flushMu.Unlock()
}

// renderMCPChanges splits a reconcile's changes into one-line notices and
// fixProblem proposals (from which prepare offers a one-click "fix the file"
// prompt). Pure, so the caller decides when the notices are safe to say.
func renderMCPChanges(changes []mcpChange) (notices []string, problems []fixProblem) {
	for _, ch := range changes {
		switch ch.action {
		case "parse_error":
			problems = append(problems, fixProblem{
				desc:   fmt.Sprintf("🟡 .codehalter/mcp.toml parse error: %s", ch.err),
				prompt: fmt.Sprintf(cardMCPParseError, ch.err),
			})
		case "failed":
			problems = append(problems, fixProblem{
				desc:   fmt.Sprintf("🟡 MCP server %q failed to start: %s", ch.name, ch.err),
				prompt: fmt.Sprintf(cardMCPStartError, ch.name, ch.err),
			})
		case "started", "restarted":
			// Name the tool count and the one-off cost: those tools join the
			// `tools` array, which the chat template renders ahead of the whole
			// conversation, so the next call re-reads the context once. Without
			// this line that shows up as an unexplained slow turn.
			notices = append(notices, fmt.Sprintf("✅ MCP server %q %s (%d tools). This turn re-reads its context once to pick them up.",
				ch.name, ch.action, ch.tools))
		case "stopped":
			notices = append(notices, fmt.Sprintf("MCP server %q stopped", ch.name))
		}
	}
	return notices, problems
}

// ---------------------------------------------------------------------------
// notifyCapabilities — consolidated banner
// ---------------------------------------------------------------------------

// notifyCapabilities renders ONE consolidated banner covering everything
// the user needs to see at the top of a turn: settings.toml path, LLM status,
// detected stacks, container, firefox, run_command, the skills in force and MCP
// servers. Called by prepare exactly once per session, on the first prepare;
// later turns emit nothing.
func (a *agent) notifyCapabilities(ctx context.Context, sess *Session, sid string) {
	var b strings.Builder

	// The version first: which binary is talking is the first question when a
	// prompt change or an update seems not to have arrived, and it is the one
	// fact the rest of the banner cannot be read without.
	b.WriteString(versionBanner())
	if a.settings.path != "" {
		fmt.Fprintf(&b, " · settings: %s", a.settings.path)
	}
	b.WriteString("\n\n")
	// One line, once per session, and the card that installs it follows the
	// banner (offerSelfUpdate). The check answers from a day-old cache most of
	// the time and says nothing at all when it cannot reach GitHub.
	if tag := newerRelease(ctx, sess.Cwd); tag != "" {
		fmt.Fprintf(&b, "🟡 Update: codehalter %s is available (running %s).\n\n", tag, version)
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

	// A stack's language server is not listed here: it's covered by the ✅ MCP
	// line at the bottom once wired, and by its setup card until then.
	if len(sess.knownStacks) > 0 {
		fmt.Fprintf(&b, "Stacks: %s", strings.Join(sess.knownStacks, ", "))
		if len(sess.knownStacks) > 1 {
			b.WriteString(" (monorepo)")
		}
		b.WriteString("\n\n")
	}

	if a.projectIsEmpty() {
		b.WriteString("Empty project: I'll ask about language and runner on your first message.\n\n")
	}

	if files := skillSet(sess.Cwd, sess.knownStacks); len(files) > 0 {
		names := make([]string, len(files))
		for i, n := range files {
			names[i] = strings.TrimSuffix(strings.TrimPrefix(n, "SKILL-"), ".md")
		}
		fmt.Fprintf(&b, "🧠 Skills: %s\n\n", strings.Join(names, ", "))
	}
	// A file in .codehalter that carries a built-in's name replaces the shipped
	// text, and nothing else on screen says so. Twice now a copy written by an
	// older codehalter took over silently and the model ran on prompts naming
	// tools that no longer exist; this line is what makes that visible.
	if over := overriddenBuiltins(sess.Cwd); len(over) > 0 {
		fmt.Fprintf(&b, "🟡 .codehalter overrides %d built-in file(s): %s. Each replaces the shipped text; delete it to use the built-in.\n\n",
			len(over), strings.Join(over, ", "))
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

	a.say(ctx, sid, b.String())
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
	// A card that asks for something already ends its desc in a question mark;
	// the container-setup card states a fact, so it gets the question added.
	// Not every card is an install any more, so the button says neither.
	title := p.desc
	if !strings.HasSuffix(title, "?") {
		title += " (fix it?)"
	}
	ok, tcId, err := a.askYesNoWithCard(ctx, sid, title, "think", "Do it", "Skip")
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
	// The accepted fix is its own turn, inside the turn the caller holds, run
	// through the same path as a typed prompt so it gets the "✅ Done" stats
	// line and compaction.
	if err := a.runPromptTurn(ctx, sess, p.prompt); err != nil {
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
