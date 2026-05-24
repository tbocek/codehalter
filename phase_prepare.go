package main

import (
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/BurntSushi/toml"
)

// ---------------------------------------------------------------------------
// Embedded PREPARE templates
// ---------------------------------------------------------------------------

//go:embed docs/PREPARE-alpine-go.md
var prepareAlpineGo string

//go:embed docs/PREPARE-arch-go.md
var prepareArchGo string

//go:embed docs/PREPARE-debian-go.md
var prepareDebianGo string

//go:embed docs/PREPARE-fedora-go.md
var prepareFedoraGo string

//go:embed docs/PREPARE-ubuntu-go.md
var prepareUbuntuGo string

//go:embed docs/PREPARE-monorepo.md
var prepareMonorepoMD string

// defaultPrepares is keyed by "<os>-<stack>". prepareStacks injects the
// matching body when a stack's probe binary is missing AND the user hasn't
// already declined. Only Go is wired in today; add more stacks as their
// PREPARE templates land (and extend stackProbeBinary alongside).
var defaultPrepares = map[string]string{
	"alpine-go": prepareAlpineGo,
	"arch-go":   prepareArchGo,
	"debian-go": prepareDebianGo,
	"fedora-go": prepareFedoraGo,
	"ubuntu-go": prepareUbuntuGo,
}

// stackProbeBinary returns the binary whose presence on PATH means the
// stack's required dev tooling is installed. "" means no probe known —
// prepareStacks skips that stack. One binary per stack by design (gopls
// stands in for Go because it transitively needs `go` to install).
func stackProbeBinary(stack string) string {
	switch stack {
	case "go":
		return "gopls"
	}
	return ""
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

// prepare runs at the top of every Prompt cycle: re-verify LLM reachability
// (blocking on a Retry card when none is reachable) and assemble a PREPARE
// composite when a detected stack's tooling is missing. Returns the composite
// so Prompt can fold it into the user's first message of the turn — "" when
// nothing needs injection. Safe to call repeatedly; each step is a no-op when
// nothing has changed since the previous cycle.
func (a *agent) prepare(ctx context.Context, sess *Session, sid string) string {
	if sess == nil {
		return ""
	}
	a.ensureLLM(ctx, sess, sid)
	return a.prepareStacks(sess)
}

// ---------------------------------------------------------------------------
// ensureLLM — settings load, probe, Retry-card loop
// ---------------------------------------------------------------------------

// ensureLLM blocks until at least one [[llm]] connection answers a probe.
// First call scaffolds .codehalter/settings.toml if neither the global nor
// the project-local file exists. The probe is skipped when both the merged
// settings-file hash matches sess.llmHash AND at least one connection answered
// the previous probe — i.e. the file the user edited hasn't actually changed
// AND we know we have a working route. A single "Retry" tool card is shown
// when no connection is reachable; clicking it always re-probes regardless of
// hash (the user may have changed network settings outside the file).
//
// There is no Abort: codehalter cannot function without an LLM. In auto-
// answer modes (autopilot, subagents) we cap retries at 3 to avoid spinning
// forever — those callers handle "no LLM" gracefully via pickAvailable.
func (a *agent) ensureLLM(ctx context.Context, sess *Session, sid string) {
	auto, _ := a.shouldAutoAnswer(sid)
	const autoCap = 3
	forceRetry := false
	for attempt := 0; ; attempt++ {
		if loaded, err := loadSettings(sess.Cwd); err == nil {
			a.settings = loaded
			a.buildSlotSems()
		}
		if a.settings.path == "" {
			a.scaffoldSettings(ctx, sess.Cwd, sid)
			if loaded, err := loadSettings(sess.Cwd); err == nil {
				a.settings = loaded
				a.buildSlotSems()
			}
		}
		currentHash := hashSettingsFiles(sess.Cwd)
		if !forceRetry && currentHash != "" && currentHash == sess.llmHash && a.hasReachableLLM() {
			return
		}
		a.probeAllLLMs(ctx)
		sess.llmHash = currentHash
		a.notifyCapabilitiesLLM(ctx, sess, sid)
		if a.hasReachableLLM() {
			return
		}
		if auto && attempt >= autoCap-1 {
			return
		}
		tcId, err := a.askAcknowledgeWithCard(ctx, sid, "LLM not reachable — edit settings.toml, then click Retry", "think", "Retry")
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Retrying LLM probe")})
		forceRetry = true
	}
}

// scaffoldSettings writes .codehalter/settings.toml with the embedded
// placeholder template and prints a short hint to chat. No-op when a settings
// file already exists. The placeholder won't reach any real server — the
// next probe will fail and the Retry card explains the situation.
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
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "Wrote " + path + " with placeholder values. Edit `url` and `model` to match your LLM server, then click Retry below. Optional: move the edited file to ~/.config/codehalter/settings.toml to share it across every project.\n\n"}})
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
// a.connReachable, a.mainSlotTokens, and a.imagesSupported. The human-
// readable status is rendered separately by renderLLMStatus so
// notifyCapabilitiesLLM can diff against the snapshot.
func (a *agent) probeAllLLMs(ctx context.Context) {
	conns := a.settings.allConnections()
	a.connReachable = make(map[string]bool, len(conns))
	a.mainSlotTokens = 0
	if len(conns) == 0 {
		a.imagesSupported = false
		return
	}
	results := make([]probeResult, len(conns))
	parallel(len(conns), maxParallel, func(i int) {
		c := conns[i]
		results[i] = a.probeLLM(ctx, &c)
	})
	firstReachable := -1
	for i, r := range results {
		c := conns[i]
		a.connReachable[connKey(&c)] = r.Reachable
		if r.Reachable && firstReachable < 0 {
			firstReachable = i
		}
	}
	// LLM[0] owns the foreground session's KV cache, so its context size
	// drives compaction sizing. llama.cpp's `-c N -np K` splits the total
	// across K slots, so divide.
	if results[0].ContextSize > 0 {
		a.mainSlotTokens = results[0].ContextSize / conns[0].parallelCap()
	}
	if firstReachable < 0 {
		a.imagesSupported = false
	} else {
		a.imagesSupported = results[firstReachable].ImageSupport
	}
}

// renderLLMStatus formats the LLM probe results into the chat summary used
// by notifyCapabilitiesLLM. Pure function over agent state — produces the
// same string until probeAllLLMs or settings changes.
func (a *agent) renderLLMStatus() string {
	conns := a.settings.allConnections()
	var b strings.Builder
	if len(conns) == 0 {
		b.WriteString("🟡 LLM: no [[llm]] in settings.toml — codehalter cannot run until you add one.\n\n")
		return b.String()
	}
	if settingsLooksPlaceholder(a.settings) {
		b.WriteString("🟡 LLM: " + a.settings.path + " still has the placeholder model \"your-model-id\". Edit it with your real url and model, then click Retry below.\n\n")
		return b.String()
	}
	firstReachable := -1
	for i := range conns {
		c := conns[i]
		label := fmt.Sprintf("llm[%d]", i)
		if i > 0 && c.Tag != "" {
			label += " " + c.Tag
		}
		if !a.connReachable[connKey(&c)] {
			fmt.Fprintf(&b, "🟡 %s: unreachable at %s — start your server or fix the url.\n\n", label, c.URL)
			continue
		}
		fmt.Fprintf(&b, "✅ %s: %s @ %s (parallel=%d)\n\n", label, c.Model, c.URL, c.parallelCap())
		if firstReachable < 0 {
			firstReachable = i
		}
	}
	if firstReachable < 0 {
		b.WriteString("🟡 No LLM reachable — every connection above failed. Codehalter cannot run any prompt until at least one comes back.\n\n")
		return b.String()
	}
	if a.imagesSupported {
		b.WriteString("✅ Image support: enabled\n\n")
	} else {
		b.WriteString("Image support: disabled\n\n")
	}
	if a.mainSlotTokens > 0 {
		if pc := conns[0].parallelCap(); pc > 1 {
			fmt.Fprintf(&b, "✅ Context window: %d tokens/slot (n_ctx %d ÷ %d slots, compact at ~%d)\n\n", a.mainSlotTokens, a.mainSlotTokens*pc, pc, a.compactTriggerTokens())
		} else {
			fmt.Fprintf(&b, "✅ Context window: %d tokens (compact at ~%d)\n\n", a.mainSlotTokens, a.compactTriggerTokens())
		}
	} else {
		fmt.Fprintf(&b, "🟡 Context window: unknown — server didn't report n_ctx, using default compact trigger %d\n\n", rawBufferTokens)
	}
	return b.String()
}

// notifyCapabilitiesLLM prints the LLM-side status to chat when it differs
// from the snapshot the session last saw. Steady-state runs emit nothing.
func (a *agent) notifyCapabilitiesLLM(ctx context.Context, sess *Session, sid string) {
	status := a.renderLLMStatus()
	if status == sess.llmStatusSnapshot {
		return
	}
	sess.llmStatusSnapshot = status
	if a.settings.path != "" {
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "Using " + a.settings.path + "\n\n"}})
	}
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: status}})
}

// ---------------------------------------------------------------------------
// prepareStacks — detect, probe, compose PREPARE injection
// ---------------------------------------------------------------------------

// osID returns a lowercase OS family identifier suitable for indexing
// defaultPrepares ("alpine", "arch", "debian", "fedora", "ubuntu"). Reads
// /etc/os-release's ID field. Returns "" when the file is missing or the
// ID isn't recognised — caller skips PREPARE injection in that case.
func osID() string {
	data, err := os.ReadFile("/etc/os-release")
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "ID=") {
			continue
		}
		id := strings.TrimPrefix(line, "ID=")
		id = strings.Trim(id, `"'`)
		return strings.ToLower(id)
	}
	return ""
}

// prepareState is the parsed .codehalter/prepare-state.toml. The LLM (told
// by PREPARE-*-go.md) appends to `declined` when the user refuses an
// install — Go just reads to suppress re-injection.
type prepareState struct {
	Declined []string `toml:"declined"`
}

// readPrepareState loads .codehalter/prepare-state.toml as a set of declined
// stack names. Missing or unreadable file → empty map; a corrupted state
// file shouldn't permanently hide the install prompt.
func readPrepareState(cwd string) map[string]bool {
	path := filepath.Join(cwd, sessionDir, "prepare-state.toml")
	var ps prepareState
	if _, err := toml.DecodeFile(path, &ps); err != nil {
		return nil
	}
	out := make(map[string]bool, len(ps.Declined))
	for _, s := range ps.Declined {
		out[s] = true
	}
	return out
}

// prepareStacks detects active stacks and returns a PREPARE composite when
// any stack's probe binary is missing AND the user hasn't already declined
// (persistently via prepare-state.toml, or this session via
// sess.preparedStacks). Returns "" when nothing to inject. Bash and
// devcontainer aren't stacks — they're meta-tooling every project uses, so
// we filter them out before recording knownStacks/monorepo.
func (a *agent) prepareStacks(sess *Session) string {
	var stacks []string
	for _, s := range detectStacks(sess.Cwd) {
		if s == "bash" || s == "devcontainer" {
			continue
		}
		stacks = append(stacks, s)
	}
	sess.knownStacks = stacks
	sess.monorepo = len(stacks) > 1

	if !a.hasReachableLLM() {
		return ""
	}

	declined := readPrepareState(sess.Cwd)
	injected := make(map[string]bool, len(sess.preparedStacks))
	for _, s := range sess.preparedStacks {
		injected[s] = true
	}

	var toPrepare []string
	for _, s := range stacks {
		bin := stackProbeBinary(s)
		if bin == "" {
			continue
		}
		if _, err := exec.LookPath(bin); err == nil {
			continue
		}
		if declined[s] || injected[s] {
			continue
		}
		toPrepare = append(toPrepare, s)
	}
	if len(toPrepare) == 0 {
		return ""
	}
	osName := osID()
	if osName == "" {
		return ""
	}

	var b strings.Builder
	if sess.monorepo {
		b.WriteString(prepareMonorepoMD)
		if !strings.HasSuffix(prepareMonorepoMD, "\n") {
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}
	wrote := false
	for _, s := range toPrepare {
		body, ok := defaultPrepares[osName+"-"+s]
		if !ok {
			continue
		}
		b.WriteString(body)
		if !strings.HasSuffix(body, "\n") {
			b.WriteString("\n")
		}
		b.WriteString("\n")
		sess.preparedStacks = append(sess.preparedStacks, s)
		wrote = true
	}
	if !wrote {
		return ""
	}
	return b.String()
}
