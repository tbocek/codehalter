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

// Fix-card prompts stay inline, not in res/*.md: nothing reads them at runtime, and a file would
// only separate each format string from the fmt.Sprintf verbs that must match it.
const (
	// One card for all missing tools. Holes: distro, then tools ("bin (reason)", comma-joined).
	cardContainerSetup = "Container setup needed in this %s devcontainer.\n" +
		"\n" +
		"PLAN ONLY → produce execute-phase steps covering every item below, then PERSIST every install in `.devcontainer/Dockerfile`. Follow SKILL-base.md (install order + install/persist loop).\n" +
		"\n" +
		"- Missing dev tools: %s. Install each, verify each runs.\n"

	// Separate card: nothing is installed, the work is judgement. Holes: formatter list, config files.
	cardFormatHeader = "This project has no formatter config (%s), so every tool that touches it formats by its own defaults.\n" +
		"\n" +
		"That is not cosmetic. codehalter formats what it writes, the editor may format on save, CI may check a third way, and when they disagree a file is rewritten between the moment the model reads it and its next edit, so the edit fails on text that was correct when it was read.\n" +
		"\n" +
		"PLAN ONLY → produce execute-phase steps that:\n" +
		"1. MEASURE the style already in the repo. Do NOT impose defaults. Read several of the largest existing source files and count: indent width, tabs vs spaces, quote style, semicolons, trailing commas, the line width the code actually respects. State the numbers you measured.\n" +
		"2. Write %s encoding exactly those numbers, so the formatter is a no-op on code that already matches the project.\n" +
		"3. Prove it: run the formatter in check mode over the whole tree and report how many files it would still change. A large number means the config does not describe this codebase: go back to step 1 and fix the config. Do NOT reformat the repo to match a guess.\n" +
		"4. Only once step 3 is small: format the whole tree and commit that as ONE commit containing formatting and nothing else. If `git status` is not clean, skip this step and say so: a formatting commit must not sweep up someone's work in progress.\n" +
		"\n" +
		"SKILL-base.md (\"Formatter config\") has the exact config files and flags. Change no behavior anywhere in this task.\n"

	// Brevity is stated without a number: given a measurable target ("under 60 lines") the
	// model iterates against it with wc and grep instead of writing the file once.
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

type fixProblem struct {
	desc   string
	prompt string
}

func (a *agent) projectIsEmpty() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.emptyProject
}

// Only .codehalter is ignored: a .git means the project exists, it just has no files here yet.
func isEmptyProject(cwd string) bool {
	entries, err := os.ReadDir(cwd)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if e.Name() != ".codehalter" {
			return false
		}
	}
	return true
}

const emptyProjectHint = `[Note: this project directory is empty: no source files or build manifests were found. Before doing anything else, use the ask_user tool to confirm:
1. What language/framework should this project use? (Rust, Go, Node.js, Python, C, etc.)
2. Which build runner do they prefer? (Cargo, go modules, npm/pnpm, just, Make)

Only then create the appropriate skeleton (Cargo.toml for Rust, go.mod for Go, package.json for Node, justfile/Makefile otherwise) with sensible build/test/lint/format targets.]
`

type formatterNeed struct {
	bin    string
	reason string
}

// detectFormatters omits toolchain-bundled formatters (gofmt, rustfmt, zig fmt).
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
	// "[tool.ruff" matches both [tool.ruff] and [tool.ruff.lint].
	if data, err := os.ReadFile(filepath.Join(cwd, "pyproject.toml")); err == nil && strings.Contains(string(data), "[tool.ruff") {
		add("ruff", "ruff config")
	}
	return needs
}

func fileExists(cwd, name string) bool {
	_, err := os.Stat(filepath.Join(cwd, name))
	return err == nil
}

type formatterConfigNeed struct {
	bin, config string
}

// An unpinned formatter can disagree with the editor's format-on-save and rewrite text the
// model just read. gofmt, rustfmt and zig fmt have no style options, so nothing to pin.
func formatterConfigNeeds(stacks []string, cwd string) []formatterConfigNeed {
	var needs []formatterConfigNeed
	// .editorconfig pins prettier, which reads it; clang-format does not.
	prettierUnpinned := prettierBin(cwd) != "" && !hasPrettierConfig(cwd) && !fileExists(cwd, ".editorconfig")
	if prettierUnpinned && (slices.Contains(stacks, "ts") || slices.Contains(stacks, "js") || slices.Contains(stacks, "css")) {
		needs = append(needs, formatterConfigNeed{"prettier", ".prettierrc"})
	}
	if slices.Contains(stacks, "c") && onPath("clang-format") && !fileExists(cwd, ".clang-format") {
		needs = append(needs, formatterConfigNeed{"clang-format", ".clang-format"})
	}
	return needs
}

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

// prepareChecks returns fix problems instead of offering them: a card accepted here would run a
// whole fix cycle ahead of the request it interrupted. drainFixes offers them after the turn.
func (a *agent) prepareChecks(ctx context.Context, sess *Session, sid string) []fixProblem {
	if sess == nil {
		slog.Debug("prepareChecks: nil sess, skipping")
		return nil
	}
	slog.Debug("prepareChecks: start", "sid", sid, "cwd", sess.Cwd)
	a.sendAvailableCommands(ctx, sid)
	a.ensureLLM(ctx, sess, sid)
	envProblems := a.checkEnv(sess, sid)
	mcpProblems := a.checkMCP(ctx, sess, sid)
	// Banner on each session's first prepare, even if unchanged, so setup never looks like a hang.
	if !sess.capabilitiesShown {
		a.notifyCapabilities(ctx, sess, sid)
		sess.capabilitiesShown = true
		// After the banner, so the card lands under the line announcing it.
		a.offerSelfUpdate(ctx, sess, sid)
	}
	slog.Debug("prepareChecks: done", "sid", sid, "hasLLM", a.hasReachableLLM(), "stacks", sess.knownStacks)
	return append(envProblems, mcpProblems...)
}

func (a *agent) drainFixes(ctx context.Context, sid string, fixes []fixProblem) {
	for _, p := range fixes {
		if ctx.Err() != nil {
			break
		}
		a.proposeFix(ctx, sid, p)
	}
}

// Below this per-slot n_ctx there is no room for a compaction tail, so startup refuses.
const minSlotTokens = 32 * 1024

// ensureLLM blocks until some [[llm]] answers and llm[0] has minSlotTokens per slot. The Retry
// card has no Abort (codehalter cannot work without an LLM); autopilot stops after 3 tries.
func (a *agent) ensureLLM(ctx context.Context, sess *Session, sid string) {
	auto := a.isAutopilot()
	const autoCap = 3
	ready := func() bool {
		return a.hasReachableLLM() && a.mainSlotTokens.Load() >= minSlotTokens
	}
	forceRetry := false
	// reload reports whether a settings file exists; one that fails to parse keeps the previous settings.
	reload := func() bool {
		loaded, err := loadSettings(sess.Cwd)
		a.cfgMu.Lock()
		defer a.cfgMu.Unlock()
		if err == nil {
			a.setSettings(loaded)
		} else {
			slog.Warn("ensureLLM: keeping previous settings, reload failed", "err", err)
		}
		return a.settings.path != ""
	}
	for attempt := 0; ; attempt++ {
		// Unchanged file and gates holding: nothing to reload or probe.
		currentHash := hashSettingsFiles(sess.Cwd)
		if !forceRetry && currentHash != "" && currentHash == sess.llmHash && ready() {
			return
		}
		if !reload() {
			a.scaffoldSettings(ctx, sess.Cwd, sid)
			reload()
			currentHash = hashSettingsFiles(sess.Cwd)
		}
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
			msg = "LLM not reachable: edit settings.toml, then click Retry"
		case a.mainSlotTokens.Load() == 0:
			probed := "GET /v1/models and GET /props"
			a.cfgMu.RLock()
			if len(a.settings.LLM) > 0 {
				c := &a.settings.LLM[0]
				probed = "GET " + c.endpoint("/v1/models") + " and GET " + c.endpoint("/props")
			}
			where := orElse(a.settings.path, "your settings.toml")
			a.cfgMu.RUnlock()
			msg = fmt.Sprintf("LLM reachable but neither metadata endpoint reported a context size (n_ctx): codehalter probed %s. It needs the model's context window to size compaction safely. Fix it one of two ways: (1) set `context_size = N` (the model's max prompt+output tokens) on the [[llm]] entry in %s, use this when your backend doesn't expose n_ctx; or (2) restart your server with the size on the launch command (llama.cpp: `-c N`, vLLM: `--max-model-len N`, llama-server: ensure /props is enabled). Then click Retry.", probed, where)
		default:
			msg = fmt.Sprintf("LLM reachable but per-slot context window is only %d tokens; codehalter requires at least %d. Restart your server with a larger `-c N` (llama.cpp) / `--max-model-len N` (vLLM), or reduce the `parallel` slot count in settings.toml, then click Retry.", a.mainSlotTokens.Load(), minSlotTokens)
		}
		_, tcId, err := a.askCard(ctx, sid, msg, "think", []permissionOption{{OptionId: "ack", Name: "Retry", Kind: "allow_once"}})
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Retrying LLM probe")})
		forceRetry = true
	}
}

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
	// Read back before claiming success: some devcontainer mounts accept a write that never
	// persists, and a reset hook (`git clean -fdX`) may reap the gitignored .codehalter/.
	if data, err := os.ReadFile(path); err != nil || len(data) == 0 {
		reason := "read back empty"
		if err != nil {
			reason = err.Error()
		}
		a.say(ctx, sid, "Wrote "+path+" but could not read it back ("+reason+"). The directory may be read-only or wiped by a reset hook (.codehalter/ is gitignored). Add a global ~/.config/codehalter/settings.toml instead.\n\n")
		return
	}
	gitignoreNote := ""
	if ensureSettingsGitignored(cwd) {
		gitignoreNote = " It's listed in .gitignore so your api_key isn't committed."
	}
	a.say(ctx, sid, "Wrote "+path+" with placeholder values."+gitignoreNote+" Edit `server` and `model` to match your LLM server, then click Retry below. If it is not in your editor's file tree, refresh: agent-created files do not always show up live. Optional: move the edited file to ~/.config/codehalter/settings.toml to share it across every project.\n\n")
}

// hashSettingsFiles returns "" only when neither settings file exists.
func hashSettingsFiles(cwd string) string {
	h := sha256.New()
	written := false
	if path, err := globalConfigPath("settings.toml"); err == nil {
		if data, err := os.ReadFile(path); err == nil {
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

func (a *agent) hasReachableLLM() bool {
	a.cfgMu.RLock()
	defer a.cfgMu.RUnlock()
	for _, p := range a.connProbe {
		if p.Reachable {
			return true
		}
	}
	return false
}

func (a *agent) probeAllLLMs(ctx context.Context) {
	a.cfgMu.RLock()
	conns := slices.Clone(a.settings.LLM)
	a.cfgMu.RUnlock()
	a.mainSlotTokens.Store(0)
	if len(conns) == 0 {
		a.cfgMu.Lock()
		a.connProbe = map[string]probeResult{}
		a.cfgMu.Unlock()
		a.imagesSupported.Store(false)
		return
	}
	results := make([]probeResult, len(conns))
	parallel(len(conns), len(conns), func(i int) {
		c := conns[i]
		results[i] = probeLLM(ctx, &c)
	})
	probes := make(map[string]probeResult, len(conns))
	for i := range conns {
		probes[conns[i].Server+"\x00"+conns[i].Model] = results[i]
	}
	// Under cfgMu: a prior turn's background call may be reading a.settings.LLM and a.connSems.
	a.cfgMu.Lock()
	a.connProbe = probes
	a.setSettings(a.settings)
	// LLM[0] holds the session's KV cache, so its per-slot window sizes compaction. A declared
	// context_size or /v1/models' -c is a total and is divided by the slot count.
	slots := a.settings.LLM[0].parallelCap()
	a.cfgMu.Unlock()

	switch {
	case conns[0].ContextSize != nil && *conns[0].ContextSize > 0:
		a.mainSlotTokens.Store(int64(*conns[0].ContextSize / slots))
	case results[0].SlotCtx > 0:
		a.mainSlotTokens.Store(int64(results[0].SlotCtx))
	case results[0].ContextSize > 0:
		a.mainSlotTokens.Store(int64(results[0].ContextSize / slots))
	}
	slog.Info("probeAllLLMs", "slots", slots, "mainSlotTokens", a.mainSlotTokens.Load(),
		"slotCtx", results[0].SlotCtx, "totalCtx", results[0].ContextSize, "totalSlots", results[0].TotalSlots)
	// Only LLM[0] consumes images, and ACP advertises one agent-wide image capability.
	switch {
	case conns[0].ImageSupport != nil:
		a.imagesSupported.Store(*conns[0].ImageSupport)
	case results[0].Reachable:
		a.imagesSupported.Store(results[0].ImageSupport)
	default:
		a.imagesSupported.Store(false)
	}
}

func connPurpose(conns []LLMConnection, i int) string {
	hasSummariser := false
	for j := 1; j < len(conns); j++ {
		if strings.EqualFold(conns[j].Purpose, purposeSummary) {
			hasSummariser = true
		}
	}
	switch {
	case i == 0 && hasSummariser:
		return "the session (planning, execution, documentation)"
	case i == 0:
		return "the session (planning, execution, documentation) and the background summariser"
	case strings.EqualFold(conns[i].Purpose, purposeSummary):
		return "background work (the summariser, side questions), off the session's cache"
	default:
		return "unused: nothing routes here (purpose = \"summary\" would host the summariser)"
	}
}

func (a *agent) renderLLMStatus() string {
	a.cfgMu.RLock()
	conns := slices.Clone(a.settings.LLM)
	path := a.settings.path
	probes := a.connProbe
	a.cfgMu.RUnlock()
	var b strings.Builder
	if len(conns) == 0 {
		b.WriteString("🟡 LLM: no [[llm]] in settings.toml: codehalter cannot run until you add one.\n\n")
		return b.String()
	}
	if conns[0].Model == "your-model-id" {
		fmt.Fprintf(&b, "🟡 LLM: %s still has the placeholder model \"your-model-id\". Edit it with your real url and model, then click Retry below.\n\n", path)
		return b.String()
	}
	firstReachable := -1
	for i := range conns {
		c := conns[i]
		label := fmt.Sprintf("llm[%d]", i)
		if i > 0 && c.Tag != "" {
			label += " " + c.Tag
		}
		pr := probes[c.Server+"\x00"+c.Model]
		if !pr.Reachable {
			fmt.Fprintf(&b, "🟡 %s: unreachable at %s: start your server or fix the server url.\n\n", label, c.Server)
			continue
		}
		// Gateways answer an unknown model id with an empty 200, so warn before a turn fails to parse.
		if pr.ModelKnown && !pr.ModelLoaded {
			avail := "its model list came back empty"
			if len(pr.AvailableModels) > 0 {
				avail = "it offers: " + strings.Join(pr.AvailableModels, ", ")
			}
			fmt.Fprintf(&b, "🟡 %s: reachable at %s, but model `%s` isn't in its /v1/models list, so requests may return an empty response. Check `model =` in settings.toml (%s). Harmless if your gateway lists models under different ids or doesn't enumerate them.\n\n", label, c.Server, c.Model, avail)
			if firstReachable < 0 {
				firstReachable = i
			}
			continue
		}
		fmt.Fprintf(&b, "✅ %s: %s @ %s (parallel=%d) · %s\n\n", label, c.Model, c.Server, c.parallelCap(), connPurpose(conns, i))
		// Roles differing in non-sampler params get two renderings, so every phase switch re-reads
		// what the other role appended; the rewind detector only notices after the tokens are spent.
		if think, exec := renderKey(c.paramsFor("thinking")), renderKey(c.paramsFor("execute")); think != exec {
			fmt.Fprintf(&b, "❕ %s: the two roles ask for different renderings: `params_thinking` %s vs `params_execute` %s. "+
				"Anything that is not a sampler is an argument to the chat template, so every plan ↔ execute switch re-evaluates "+
				"whatever the other role appended in between. Make them agree, or keep the split deliberately if you have priced it.\n\n",
				label, orElse(think, "(none)"), orElse(exec, "(none)"))
		}
		if firstReachable < 0 {
			firstReachable = i
		}
	}
	if firstReachable < 0 {
		b.WriteString("🟡 No LLM reachable: every connection above failed. Codehalter cannot run any prompt until at least one comes back.\n\n")
		return b.String()
	}
	switch {
	case a.imagesSupported.Load():
		b.WriteString("✅ Image support: enabled\n\n")
	case conns[0].ImageSupport != nil:
		b.WriteString("Image support: disabled (declared image_support = false in settings.toml)\n\n")
	default:
		b.WriteString("❕ Image support: undetected, codehalter assumed disabled. If your model accepts images, set `image_support = true` on the [[llm]] entry in settings.toml.\n\n")
	}
	switch mst := int(a.mainSlotTokens.Load()); {
	case mst == 0:
		b.WriteString("🟡 Context window: unknown. Set `context_size = N` on the [[llm]] entry in settings.toml. For llama.cpp/vLLM you can also restart with the launch flag (`-c N` / `--max-model-len N`) so the probe discovers it.\n\n")
	case mst < minSlotTokens:
		fmt.Fprintf(&b, "🟡 Context window: only %d tokens/slot; codehalter requires at least %d. Raise `context_size` in settings.toml, increase your server's launch flag (`-c N` / `--max-model-len N`), or reduce `parallel`.\n\n", mst, minSlotTokens)
	default:
		inputCap := mst * compactTriggerPct / 100
		if pc := conns[0].parallelCap(); pc > 1 {
			fmt.Fprintf(&b, "✅ Context window: %d tokens/slot (n_ctx %d ÷ %d slots, max prompt %d)\n\n", mst, mst*pc, pc, inputCap)
		} else {
			fmt.Fprintf(&b, "✅ Context window: %d tokens (max prompt %d)\n\n", mst, inputCap)
		}
	}
	return b.String()
}

func onPath(bin string) bool { _, err := exec.LookPath(bin); return err == nil }

// checkEnv emits no chat output of its own; everything it finds becomes a fix card.
func (a *agent) checkEnv(sess *Session, sid string) []fixProblem {
	stacks := detectStacks(sess.Cwd)
	sess.knownStacks = stacks
	osi := readOSInfo()

	// The system prompt leads every request, so changing it mid-session busts the prefix cache.
	// A skill that becomes applicable later is appended as a user message and enters the prompt
	// at the next compaction; promptSkills tracks what the prompt already holds.
	skills := skillSet(sess.Cwd, stacks)
	if sess.promptSkills == nil {
		sess.promptSkills = skills
	}
	if sess.SystemPrompt == "" {
		if sp, err := a.systemPrompt(sid); err != nil {
			slog.Warn("prepare: systemPrompt build failed", "sid", sid, "err", err)
		} else {
			sess.SystemPrompt = sp
			sess.promptSkills = skills
		}
	} else {
		for _, name := range skills {
			if slices.Contains(sess.promptSkills, name) {
				continue
			}
			if body := skillBody(sess.Cwd, name); body != "" {
				sess.AddUser("[New skill available this session: " + name +
					". It enters the system prompt at the next history compaction; until then it's here.]\n\n" + body)
				sess.promptSkills = append(sess.promptSkills, name)
			}
		}
	}

	var missing []string
	for _, f := range detectFormatters(stacks, sess.Cwd) {
		// prettier usually lives in node_modules rather than on PATH.
		found := onPath(f.bin)
		if f.bin == "prettier" {
			found = prettierBin(sess.Cwd) != ""
		}
		if !found {
			missing = append(missing, f.bin+" ("+f.reason+" formatter)")
		}
	}

	var probs []fixProblem
	if len(missing) > 0 {
		// Name the OS so the model need not rediscover it; "Linux" covers a container without os-release.
		distro := osi.Fields["PRETTY_NAME"]
		if distro == "" && osi.ID != "" {
			distro = strings.ToUpper(osi.ID[:1]) + osi.ID[1:]
		}
		if distro == "" {
			distro = "Linux"
		}
		tools := strings.Join(missing, ", ")
		probs = append(probs, fixProblem{
			desc:   "🟡 Container setup: install " + tools,
			prompt: fmt.Sprintf(cardContainerSetup, distro, tools),
		})
	}

	if !checkDone(sess.Cwd, checkFormatConfig) {
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

	// Test order matters: checkDone is cheap and true forever after the first offer, so the tree
	// walk runs at most once. Unmarked below the threshold, so a project scaffolded now is offered later.
	if !checkDone(sess.Cwd, checkAgentsFile) {
		if _, content := loadAgentsFile(sess.Cwd); content == "" && hasProjectFiles(sess.Cwd, minFilesForBrief) {
			markCheckDone(sess.Cwd, checkAgentsFile)
			probs = append(probs, fixProblem{
				desc:   "🟡 No AGENT.md: write down what this project is, so every session starts knowing it?",
				prompt: cardAgentsFile,
			})
		}
	}
	return probs
}

const minFilesForBrief = 5

const (
	checkFormatConfig = "format-config"
	checkAgentsFile   = "agent-md"
)

// checksDoneFile lists one-time cards already offered, one per line; deleting a line re-arms that card.
const checksDoneFile = "checks.done"

func checkDone(cwd, name string) bool {
	data, err := os.ReadFile(filepath.Join(cwd, ".codehalter", checksDoneFile))
	return err == nil && slices.Contains(strings.Fields(string(data)), name)
}

func markCheckDone(cwd, name string) {
	if checkDone(cwd, name) {
		return
	}
	path := filepath.Join(cwd, ".codehalter", checksDoneFile)
	if err := appendFile(path, name+"\n"); err != nil {
		slog.Warn("recording a completed check failed", "path", path, "err", err)
	}
}

// The only MCP reconcile, before a turn: the `tools` array renders ahead of the conversation,
// so a changed server re-reads the whole context and a mid-turn change would break the prefix cache.
func (a *agent) checkMCP(ctx context.Context, sess *Session, sid string) []fixProblem {
	stopBeat := a.heartbeat(ctx, sid)
	changes := a.reconcileMCP(ctx, sess.Cwd)
	stopBeat()

	var problems []fixProblem
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
			// Said out loud, or the re-read looks like an unexplained slow turn.
			a.say(ctx, sid, fmt.Sprintf("✅ MCP server %q %s (%d tools). This turn re-reads its context once to pick them up.\n",
				ch.name, ch.action, ch.tools))
		case "stopped":
			a.say(ctx, sid, fmt.Sprintf("MCP server %q stopped\n", ch.name))
		}
	}
	return problems
}

func (a *agent) notifyCapabilities(ctx context.Context, sess *Session, sid string) {
	var b strings.Builder

	b.WriteString(versionBanner())
	a.cfgMu.RLock()
	path := a.settings.path
	a.cfgMu.RUnlock()
	if path != "" {
		fmt.Fprintf(&b, " · settings: %s", path)
	}
	b.WriteString("\n\n")
	if tag := newerRelease(ctx, sess.Cwd); tag != "" {
		fmt.Fprintf(&b, "🟡 Update: codehalter %s is available (running %s).\n\n", tag, version)
	}
	b.WriteString(a.renderLLMStatus())

	if kind := containerKind(); kind != "" {
		fmt.Fprintf(&b, "✅ Container: %s\n\n", kind)
	} else {
		b.WriteString("🟡 Container: none (running on host: file edits and tasks hit your real filesystem)\n\n")
	}
	if _, err := findFirefox(); err == nil {
		b.WriteString("✅ Firefox: found (web_search/web_read enabled)\n\n")
	} else {
		b.WriteString("🟡 Firefox: not found, web_search/web_read disabled. Install firefox or set FIREFOX_PATH.\n\n")
	}
	// Only reachable inside a container (ensureDevcontainer), where run_command is always registered.
	b.WriteString("✅ run_command: available (probes and test installs; `.git` is bind-mounted read-only, destructive git commands fail at the FS layer)\n\n")

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
	// An override silently replaces the shipped text, and a stale one can name tools that no longer exist.
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

func (a *agent) proposeFix(ctx context.Context, sid string, p fixProblem) {
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
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Skipped: fix it manually when convenient")})
		return
	}
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Dispatching: " + p.prompt)})
	sess := a.getSession(sid)
	if sess == nil {
		return
	}
	// Runs as a full turn nested in the caller's, through the same path as a typed prompt.
	if err := a.runPromptTurn(ctx, sess, p.prompt); err != nil {
		if isCancelled(err) {
			slog.Debug("proposeFix: fix dispatch cancelled", "sid", sid, "err", err)
		} else {
			slog.Warn("proposeFix: fix dispatch failed", "sid", sid, "err", err)
		}
	}
}
