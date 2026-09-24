package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
)

func hasFormatterNeed(needs []formatterNeed, bin string) bool {
	for _, n := range needs {
		if n.bin == bin {
			return true
		}
	}
	return false
}

// TestCheckEnvInjectsMidSessionSkillNotPrompt pins the cache-safety rule: a
// skill seeded mid-session is injected as a user message, NOT folded into the
// system prompt (which would bust the KV prefix cache — only compaction may).
func TestCheckEnvInjectsMidSessionSkillNotPrompt(t *testing.T) {
	a, s := newTestAgent(t)
	cfgDir := filepath.Join(s.Cwd, ".codehalter")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Baseline: freeze the skills that already apply as "already in the prompt",
	// so only a NEW skill counts as added mid-session.
	const frozen = "EXISTING PROMPT — do not mutate"
	s.SystemPrompt = frozen
	s.promptSkills = skillSet(s.Cwd, nil)

	body := "# Zzz skill\n\nuse the zzz tool wisely\n"
	if err := os.WriteFile(filepath.Join(cfgDir, "SKILL-zzz.md"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	a.checkEnv(s, s.ID)

	if s.SystemPrompt != frozen {
		t.Errorf("system prompt was mutated mid-session — KV cache bust")
	}
	var injected bool
	for _, m := range s.Messages {
		if m.Role == "user" && strings.Contains(m.Content, "SKILL-zzz.md") &&
			strings.Contains(m.Content, "use the zzz tool wisely") {
			injected = true
		}
	}
	if !injected {
		t.Errorf("a mid-session skill should be injected as a user message, got none")
	}
}

// TestCheckEnvSetupIsOneCard pins the one-card, one-turn rule. Every accepted
// card dispatches a full plan/execute/document cycle, so every missing dev tool
// is folded into a SINGLE fixProblem, with one PLAN ONLY directive. Two stacks
// wanting two different formatters is the case that would otherwise be two
// cards; a runner config (the justfile here) is deliberately not probed at all.
func TestCheckEnvSetupIsOneCard(t *testing.T) {
	a, s := newTestAgent(t)
	// Empty PATH so every probed binary reads as missing regardless of the
	// developer's machine.
	t.Setenv("PATH", "")
	for name, body := range map[string]string{
		"tsconfig.json": "{}\n",
		"main.c":        "int main(void){return 0;}\n",
		"justfile":      "test:\n\tgo test ./...\n",
	} {
		if err := os.WriteFile(filepath.Join(s.Cwd, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	setup := firstCard(t, a.checkEnv(s, s.ID), "Missing dev tools:")
	for _, want := range []string{"prettier (", "clang-format ("} {
		if !strings.Contains(setup.prompt, want) {
			t.Errorf("prompt lacks %q: %q", want, setup.prompt)
		}
	}
	if strings.Contains(setup.prompt, "just") {
		t.Errorf("a runner binary was probed: %q", setup.prompt)
	}
	if n := strings.Count(setup.prompt, "PLAN ONLY"); n != 1 {
		t.Errorf("want exactly 1 PLAN ONLY directive, got %d: %q", n, setup.prompt)
	}
}

// firstCard returns the one problem whose prompt carries want, failing when
// none or several do. checkEnv answers with several unrelated cards in one
// pass, and a test that indexed [0] would break whenever their order changed.
func firstCard(t *testing.T, probs []fixProblem, want string) fixProblem {
	t.Helper()
	var found []fixProblem
	for _, p := range probs {
		if strings.Contains(p.prompt, want) {
			found = append(found, p)
		}
	}
	if len(found) != 1 {
		t.Fatalf("want exactly one card mentioning %q, got %d of %+v", want, len(found), probs)
	}
	return found[0]
}

// TestCheckEnvOneTimeCards pins that the two cards asking for work a project
// only ever needs once are offered once per PROJECT, not once per session: the
// mark lands in .codehalter/checks.done, so a second agent opening the same
// directory (a restart, the ordinary case) stays quiet. Before that file
// existed, declining meant being asked again at every session start.
func TestCheckEnvOneTimeCards(t *testing.T) {
	a, s := newTestAgent(t)
	// A ts project with a local prettier and no config: formatterConfigNeeds
	// wants a .prettierrc, and there is no AGENT.md either.
	bin := filepath.Join(s.Cwd, "node_modules", ".bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	for path, body := range map[string]string{
		filepath.Join(s.Cwd, "tsconfig.json"): "{}\n",
		filepath.Join(bin, "prettier"):        "#!/bin/sh\n",
	} {
		if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.MkdirAll(filepath.Join(s.Cwd, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}

	// One file to read: .git and node_modules are skipped by the walk. The
	// formatter card fires, the AGENT.md card does not. A directory that is
	// merely non-pristine has nothing for the card's "read the tree" step.
	probs := a.checkEnv(s, s.ID)
	firstCard(t, probs, "no formatter config")
	for _, p := range probs {
		if strings.Contains(p.prompt, "AGENT.md") {
			t.Fatalf("asked for an AGENT.md with one file to read: %q", p.desc)
		}
	}

	// Content is what flips it, and it is re-checked live, so a project that
	// grows during the session is asked then rather than at the next one.
	for _, n := range []string{"a.ts", "b.ts", "c.ts", "d.ts", "README.md"} {
		if err := os.WriteFile(filepath.Join(s.Cwd, n), []byte("x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	probs = a.checkEnv(s, s.ID)
	agents := firstCard(t, probs, "AGENT.md")
	if !strings.Contains(agents.prompt, "ask_user") {
		t.Errorf("the AGENT.md card must confirm the facts with the user: %q", agents.prompt)
	}

	// Same session, and after a restart: neither comes back.
	for _, again := range []func() []fixProblem{
		func() []fixProblem { return a.checkEnv(s, s.ID) },
		func() []fixProblem {
			b, s2 := newTestAgent(t)
			s2.Cwd = s.Cwd
			return b.checkEnv(s2, s2.ID)
		},
	} {
		for _, p := range again() {
			if strings.Contains(p.prompt, "no formatter config") || strings.Contains(p.prompt, "AGENT.md") {
				t.Errorf("a one-time card was offered twice: %q", p.desc)
			}
		}
	}

	// A project that ships AGENT.md is never asked in the first place.
	c, s3 := newTestAgent(t)
	if err := os.WriteFile(filepath.Join(s3.Cwd, "AGENT.md"), []byte("# x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, p := range c.checkEnv(s3, s3.ID) {
		if strings.Contains(p.prompt, "AGENT.md") {
			t.Errorf("asked for an AGENT.md that exists: %q", p.desc)
		}
	}
}

// TestDetectFormatters covers both drivers: detected stack (ts → prettier) and
// formatter config files (.clang-format → clang-format, pyproject [tool.ruff] →
// ruff), and that an empty project needs nothing.
// TestPrepareChecksBannerAlwaysShowsOnce: the capabilities banner is all a
// session open prints, so it must NOT be gated on something having changed
// since the last run. A project whose settings, tools and MCP servers are
// exactly as they were is the ordinary case, and gating on change left the
// thread empty after the gitignore card with no way to tell setup from a
// hang. First prepareChecks emits it; the second (settings hash unchanged,
// probe short-circuited) adds nothing.
func TestPrepareChecksBannerAlwaysShowsOnce(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/v1/models") {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"gpt-4o"}]}`))
	}))
	defer ts.Close()

	// loadSettings prefers the project-local settings.toml over the global
	// ~/.config/codehalter/settings.toml, so without an isolated HOME this
	// test probes the developer's real LLM servers over the network.
	t.Setenv("HOME", t.TempDir())

	h := newTerminalHarness(t)
	a, s := h.agent, h.sess
	ch := filepath.Join(s.Cwd, ".codehalter")
	if err := os.MkdirAll(ch, 0o755); err != nil {
		t.Fatal(err)
	}
	// A real settings file: ensureLLM reloads it each call, so the second call
	// sees an unchanged hash plus a satisfied gate and skips the re-probe —
	// the "nothing changed" state that used to suppress the banner entirely.
	cfg := fmt.Sprintf("[[llm]]\nserver = %q\nmodel = \"gpt-4o\"\nparallel = 1\ncontext_size = 128000\n", ts.URL)
	if err := os.WriteFile(filepath.Join(ch, "settings.toml"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}

	banners := func() int {
		n := 0
		for _, u := range h.updatesOfKind("agent_message_chunk") {
			content, _ := u["content"].(map[string]any)
			text, _ := content["text"].(string)
			if strings.Contains(text, "Container:") {
				n++
			}
		}
		return n
	}

	a.prepareChecks(context.Background(), s, s.ID)
	h.waitFor(func() bool { return banners() > 0 })
	if got := banners(); got != 1 {
		t.Fatalf("first prepareChecks: %d capabilities banners, want 1", got)
	}
	a.prepareChecks(context.Background(), s, s.ID)
	if got := banners(); got != 1 {
		t.Errorf("second prepareChecks: %d banners, want the first one only (no mid-session re-dump)", got)
	}
}

func TestDetectFormatters(t *testing.T) {
	if !hasFormatterNeed(detectFormatters([]string{"ts"}, t.TempDir()), "prettier") {
		t.Errorf("ts stack should need prettier")
	}
	if !hasFormatterNeed(detectFormatters([]string{"c"}, t.TempDir()), "clang-format") {
		t.Errorf("c stack should need clang-format")
	}

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ".clang-format"), []byte("BasedOnStyle: LLVM\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "pyproject.toml"), []byte("[tool.ruff.lint]\nselect = [\"E\"]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := detectFormatters(nil, dir)
	if !hasFormatterNeed(got, "clang-format") || !hasFormatterNeed(got, "ruff") {
		t.Errorf("config files should need clang-format+ruff, got %v", got)
	}

	if n := detectFormatters(nil, t.TempDir()); len(n) != 0 {
		t.Errorf("empty project should need no formatters, got %v", n)
	}
}

// TestProbeAllLLMsConfigBeatsProbe asserts the precedence rule: explicit
// context_size / image_support on the [[llm]] entry win over whatever the
// probe discovered. This is the path OpenAI/Ollama/vLLM users rely on —
// their /v1/models response carries no metadata, but the user declared the
// values in settings.toml.
func TestProbeAllLLMsConfigBeatsProbe(t *testing.T) {
	// Mock a metadata-bare /v1/models response (no status.args). Mirrors
	// what OpenAI and Ollama return — the probe gleans model presence but
	// nothing else, so any non-zero ContextSize / ImageSupport must come
	// from config.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/v1/models") {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"gpt-4o"}]}`))
	}))
	defer ts.Close()

	yes := true
	a := &agent{
		settings: Settings{
			LLM: []LLMConnection{{
				Server:       ts.URL,
				Model:        "gpt-4o",
				Parallel:     ptr(1),
				ContextSize:  ptr(128000),
				ImageSupport: &yes,
			}},
		},
	}
	a.probeAllLLMs(context.Background())

	if a.mainSlotTokens != 128000 {
		t.Errorf("mainSlotTokens: got %d, want 128000 (from settings.toml context_size)", a.mainSlotTokens)
	}
	if !a.imagesSupported {
		t.Errorf("imagesSupported: got false, want true (from settings.toml image_support)")
	}
}

// TestProbeAllLLMsUndetectedFallsThroughToFalse covers the warn path: probe
// finds no metadata AND config declares nothing — both signals end up at
// their safe defaults (mainSlotTokens=0, imagesSupported=false) so the
// renderLLMStatus banner can surface the "set this in settings.toml" hint.
func TestProbeAllLLMsUndetectedFallsThroughToFalse(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/v1/models") {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"gpt-4o"}]}`))
	}))
	defer ts.Close()

	a := &agent{
		settings: Settings{
			LLM: []LLMConnection{{
				Server:   ts.URL,
				Model:    "gpt-4o",
				Parallel: ptr(1),
			}},
		},
	}
	a.probeAllLLMs(context.Background())

	if a.mainSlotTokens != 0 {
		t.Errorf("mainSlotTokens: got %d, want 0 (probe metadata-bare, no config override)", a.mainSlotTokens)
	}
	if a.imagesSupported {
		t.Errorf("imagesSupported: got true, want false (probe metadata-bare, no config override)")
	}
}

// TestRenderLLMStatusWarnsModelNotInList pins the surfaced warning for the
// silent failure mode behind "plan not valid JSON": the server is reachable and
// /v1/models enumerates models, but the configured id isn't among them. The
// gateway then routes the unknown name to an empty 200, which only surfaces
// three layers down at plan time. probeLLM used to log loaded=false and move on
// (bare ✅ in the banner); now renderLLMStatus names the mismatch and lists the
// ids the server actually offers.
func TestRenderLLMStatusWarnsModelNotInList(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/v1/models") {
			http.NotFound(w, r) // /props 404s — keeps the /v1/models result
			return
		}
		w.Header().Set("Content-Type", "application/json")
		// The configured model "qwopus3.6-27b" is NOT in this list.
		_, _ = w.Write([]byte(`{"data":[{"id":"gpt-4o"},{"id":"llama-3.1-70b"}]}`))
	}))
	defer ts.Close()

	a := &agent{
		settings: Settings{
			LLM: []LLMConnection{{
				Server:      ts.URL,
				Model:       "qwopus3.6-27b",
				Parallel:    ptr(1),
				ContextSize: ptr(128000), // keep the ctx gate quiet; not under test
			}},
		},
	}
	a.probeAllLLMs(context.Background())

	pr := a.connProbe[ts.URL+"\x00"+"qwopus3.6-27b"]
	if !pr.ModelKnown {
		t.Fatalf("ModelKnown: got false, want true (the server enumerated /v1/models)")
	}
	if pr.ModelLoaded {
		t.Errorf("ModelLoaded: got true, want false (configured id not in the list)")
	}
	if len(pr.AvailableModels) != 2 {
		t.Errorf("AvailableModels: got %v, want the two offered ids", pr.AvailableModels)
	}

	status := a.renderLLMStatus()
	for _, want := range []string{"qwopus3.6-27b", "isn't in its /v1/models list", "gpt-4o", "llama-3.1-70b"} {
		if !strings.Contains(status, want) {
			t.Errorf("renderLLMStatus output missing %q.\nGot:\n%s", want, status)
		}
	}
	// No bare ✅ connection line for a model we could not confirm.
	if strings.Contains(status, "✅ llm[0]: qwopus3.6-27b") {
		t.Errorf("renderLLMStatus showed a bare ✅ for an unconfirmed model:\n%s", status)
	}
}

// TestRenderLLMStatusWarnsRoleRenderSplit pins the startup warning for the one
// settings mistake that costs real time and produces no symptom at all: the two
// roles asking the server for two different renderings of the same
// conversation. Anything that is not a sampler is an argument to the chat
// template, so the server keeps a prompt state per rendering and every plan <->
// execute switch re-evaluates whatever the other role appended in between. One
// 11.6h session paid 99582 tokens over two switches.
//
// The rewind detector already catches it, but only after the tokens are spent,
// and prompt/cached alone never name the key at fault. This is the same
// diagnosis for free, before the first call.
func TestRenderLLMStatusWarnsRoleRenderSplit(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"m"}]}`))
	}))
	defer ts.Close()

	base := LLMConnection{Server: ts.URL, Model: "m", Parallel: ptr(1), ContextSize: ptr(128000)}
	status := func(t *testing.T, think, exec map[string]any) string {
		t.Helper()
		c := base
		c.ParamsThinking, c.ParamsExecute = think, exec
		a := &agent{settings: Settings{LLM: []LLMConnection{c}}}
		a.probeAllLLMs(context.Background())
		return a.renderLLMStatus()
	}

	// Samplers may differ freely: they never reach the chat template, so the two
	// roles still render identically. Warning here would be noise on a correct
	// and quite common configuration.
	quiet := status(t,
		map[string]any{"temperature": 1.0, "top_p": 0.95},
		map[string]any{"temperature": 0.6, "max_tokens": 8192})
	if strings.Contains(quiet, "different renderings") {
		t.Errorf("warned about a sampler-only difference:\n%s", quiet)
	}

	// A chat-template argument on one role only: the real shape of the fault.
	loud := status(t,
		map[string]any{"temperature": 1.0},
		map[string]any{"temperature": 0.6, "chat_template_kwargs": map[string]any{"enable_thinking": false}})
	for _, want := range []string{"different renderings", "enable_thinking", "params_execute", "(none)"} {
		if !strings.Contains(loud, want) {
			t.Errorf("banner missing %q.\nGot:\n%s", want, loud)
		}
	}
}

// TestProbeAllLLMsExplicitFalseHonoured: a model that DOES auto-detect as
// vision-capable but the user wants disabled via image_support = false must
// stay disabled — *bool lets us distinguish "not set" from "explicitly off".
func TestProbeAllLLMsExplicitFalseHonoured(t *testing.T) {
	// Mock /v1/models with llama-swap-style status.args carrying --mmproj —
	// the probe would normally detect vision support here.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/v1/models") {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"qwen-vl","status":{"args":["--mmproj","/path","--ctx-size","32768"]}}]}`))
	}))
	defer ts.Close()

	no := false
	a := &agent{
		settings: Settings{
			LLM: []LLMConnection{{
				Server:       ts.URL,
				Model:        "qwen-vl",
				Parallel:     ptr(1),
				ImageSupport: &no,
			}},
		},
	}
	a.probeAllLLMs(context.Background())

	if a.imagesSupported {
		t.Errorf("imagesSupported: probe detected vision but user explicitly disabled — config must win")
	}
}

// llamaCppServer mocks a llama.cpp endpoint: a bare /v1/models plus a /props
// carrying a PER-SLOT n_ctx (default_generation_settings.n_ctx) and total_slots.
func llamaCppServer(perSlotCtx, totalSlots int) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.HasSuffix(r.URL.Path, "/v1/models"):
			_, _ = w.Write([]byte(`{"data":[{"id":"qwen"}]}`))
		case strings.HasSuffix(r.URL.Path, "/props"):
			fmt.Fprintf(w, `{"default_generation_settings":{"n_ctx":%d},"total_slots":%d}`, perSlotCtx, totalSlots)
		default:
			http.NotFound(w, r)
		}
	}))
}

// TestProbeAllLLMsAutoDetectsSlots: with `parallel` unset, the slot count is
// back-filled from /props total_slots and the per-slot n_ctx is used directly
// (no division), so the user no longer has to declare -np in settings.toml.
func TestProbeAllLLMsAutoDetectsSlots(t *testing.T) {
	ts := llamaCppServer(16384, 2)
	defer ts.Close()

	a := &agent{settings: Settings{LLM: []LLMConnection{{Server: ts.URL, Model: "qwen"}}}}
	a.probeAllLLMs(context.Background())

	if got := a.settings.LLM[0].Parallel; *got != 2 {
		t.Errorf("Parallel: got %d, want 2 (auto-detected from /props total_slots)", *got)
	}
	if a.mainSlotTokens != 16384 {
		t.Errorf("mainSlotTokens: got %d, want 16384 (per-slot n_ctx used directly, no division)", a.mainSlotTokens)
	}
}

// TestProbeAllLLMsRouterModelProps pins the llama.cpp router-mode fix: bare /props
// reports n_ctx=0 (role:"router"), and the real per-slot n_ctx is only returned
// when the request is routed to the model via ?model=<id>. The id here carries a
// space and a semicolon ("q3 (a; b)") to prove the query is encoded — a raw ';'
// is a query separator that would truncate the name to "model not found".
func TestProbeAllLLMsRouterModelProps(t *testing.T) {
	const modelID = "q3 (a; b)"
	var gotModel atomic.Pointer[string] // written from the handler goroutine
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.HasSuffix(r.URL.Path, "/v1/models"):
			_, _ = w.Write([]byte(`{"data":[{"id":"q3"}]}`))
		case strings.HasSuffix(r.URL.Path, "/props"):
			if m := r.URL.Query().Get("model"); m != "" {
				gotModel.Store(&m) // real model props (router routed by ?model=)
				fmt.Fprint(w, `{"default_generation_settings":{"n_ctx":16384},"total_slots":2}`)
			} else {
				fmt.Fprint(w, `{"role":"router","default_generation_settings":{"n_ctx":0},"total_slots":0}`)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	a := &agent{settings: Settings{LLM: []LLMConnection{{Server: ts.URL, Model: modelID}}}}
	a.probeAllLLMs(context.Background())

	if got := gotModel.Load(); got == nil || *got != modelID {
		t.Fatalf("router ?model= param: got %v, want %q (encoding bug truncates at ';')", got, modelID)
	}
	if got := a.settings.LLM[0].Parallel; *got != 2 {
		t.Errorf("Parallel: got %d, want 2 (from ?model= /props total_slots)", *got)
	}
	if a.mainSlotTokens != 16384 {
		t.Errorf("mainSlotTokens: got %d, want 16384 (per-slot n_ctx from ?model= /props)", a.mainSlotTokens)
	}
}

// TestProbeAllLLMsExplicitParallelWins: an explicit `parallel` is never
// overwritten by the probed total_slots.
func TestProbeAllLLMsExplicitParallelWins(t *testing.T) {
	ts := llamaCppServer(16384, 4)
	defer ts.Close()

	a := &agent{settings: Settings{LLM: []LLMConnection{{Server: ts.URL, Model: "qwen", Parallel: ptr(1)}}}}
	a.probeAllLLMs(context.Background())

	if got := a.settings.LLM[0].Parallel; *got != 1 {
		t.Errorf("Parallel: got %d, want 1 (explicit value must not be overwritten by total_slots)", *got)
	}
}

// TestScaffoldSettings pins that the skeleton settings.toml is written from the
// embedded default when none exists, and that a second call is a no-op (never
// clobbers an existing, possibly user-edited, file). This is the "if there is none,
// create a skeleton" behavior the prepare phase relies on.
func TestScaffoldSettings(t *testing.T) {
	a, s := newTestAgent(t)
	path := filepath.Join(s.Cwd, sessionDir, "settings.toml")
	if _, err := os.Stat(path); err == nil {
		t.Fatal("precondition: no settings.toml should exist yet")
	}

	a.scaffoldSettings(context.Background(), s.Cwd, s.ID)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("scaffoldSettings did not create the skeleton: %v", err)
	}
	if len(data) == 0 || string(data) != defaultSettingsTOML {
		t.Errorf("skeleton doesn't match the embedded default (%d bytes)", len(data))
	}

	// Idempotent: a second call must NOT clobber an existing file.
	os.WriteFile(path, []byte("# user edited\n"), 0o644)
	a.scaffoldSettings(context.Background(), s.Cwd, s.ID)
	if again, _ := os.ReadFile(path); string(again) != "# user edited\n" {
		t.Error("scaffoldSettings clobbered an existing settings.toml")
	}
}

// TestFormatterConfigNeeds pins who gets offered a formatter config and, more
// importantly, who does not: a Go project has nothing to pin (gofmt exposes no
// style options), and a project that already declares its style is left alone
// rather than asked about at the start of every session.
func TestFormatterConfigNeeds(t *testing.T) {
	// proj builds a project dir carrying a local prettier (so the "formatter is
	// installed" gate passes without depending on the host PATH) plus whatever
	// config files the case declares.
	proj := func(t *testing.T, files ...string) string {
		t.Helper()
		dir := t.TempDir()
		bin := filepath.Join(dir, "node_modules", ".bin")
		if err := os.MkdirAll(bin, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(bin, "prettier"), []byte("#!/bin/sh\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		for _, f := range files {
			if err := os.WriteFile(filepath.Join(dir, f), []byte("{}"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		return dir
	}

	tests := []struct {
		name   string
		stacks []string
		files  []string
		want   []formatterConfigNeed
	}{
		{
			name:   "js project with nothing pinning its style",
			stacks: []string{"js"},
			want:   []formatterConfigNeed{{"prettier", ".prettierrc"}},
		},
		{
			name:   "ts and css are the same prettier, asked once",
			stacks: []string{"ts", "css"},
			want:   []formatterConfigNeed{{"prettier", ".prettierrc"}},
		},
		{
			name:   "prettier config present",
			stacks: []string{"ts"},
			files:  []string{".prettierrc"},
		},
		{
			// prettier reads .editorconfig, so the style IS pinned.
			name:   "editorconfig counts as pinned",
			stacks: []string{"js"},
			files:  []string{".editorconfig"},
		},
		{
			// The whole point of the exclusion list: gofmt has no options, so
			// there is no config to write and nothing to disagree about.
			name:   "go project is already pinned by gofmt",
			stacks: []string{"go"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := formatterConfigNeeds(tc.stacks, proj(t, tc.files...))
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got %+v, want %+v", got, tc.want)
			}
		})
	}

	// An uninstalled formatter is the install card's business and comes first,
	// so nothing is offered here. Skipped on a machine with a global prettier,
	// where the gate legitimately passes.
	t.Run("formatter not installed", func(t *testing.T) {
		dir := t.TempDir()
		if prettierBin(dir) != "" {
			t.Skip("prettier is on PATH here")
		}
		if got := formatterConfigNeeds([]string{"js"}, dir); got != nil {
			t.Errorf("got %+v, want nothing offered for an uninstalled formatter", got)
		}
	})

	t.Run("c project without clang-format config", func(t *testing.T) {
		dir := t.TempDir()
		got := formatterConfigNeeds([]string{"c"}, dir)
		if !onPath("clang-format") {
			if got != nil {
				t.Errorf("got %+v with no clang-format installed, want nothing", got)
			}
			return
		}
		want := []formatterConfigNeed{{"clang-format", ".clang-format"}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %+v, want %+v", got, want)
		}
		if err := os.WriteFile(filepath.Join(dir, ".clang-format"), []byte("BasedOnStyle: GNU\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if got := formatterConfigNeeds([]string{"c"}, dir); got != nil {
			t.Errorf("got %+v with a .clang-format present, want nothing", got)
		}
	})
}

// TestEmptyProject covers the deferred-bootstrap path: a fresh directory
// reads as empty, so initSession sets the flag whose hint asks the user what
// language and runner to use. A populated one must not, and nothing is
// scaffolded either way.
func TestEmptyProject(t *testing.T) {
	dir := t.TempDir()
	if !isEmptyProject(dir) {
		t.Fatal("expected fresh tempdir to be empty")
	}

	if _, err := newSession(dir); err != nil { // creates .codehalter/, no sources
		t.Fatalf("newSession: %v", err)
	}
	if !isEmptyProject(dir) {
		t.Error("expected dir with only .codehalter/ to still count as empty")
	}

	if _, err := os.Stat(filepath.Join(dir, "Makefile")); err == nil {
		t.Error("bootstrap must be deferred — no Makefile should be written")
	}

	// Non-empty project: isEmptyProject=false and flag stays off.
	populated := t.TempDir()
	if err := os.WriteFile(filepath.Join(populated, "main.go"), []byte("package main\n"), 0644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if isEmptyProject(populated) {
		t.Error("expected dir with main.go to not be empty")
	}
}

// TestRenderLLMStatusNamesPurpose: the banner says what each [[llm]] entry is
// for, so a second entry is not a mystery: llm[0] carries the session and,
// with nobody else designated, the summariser too; an entry with
// purpose = "summary" takes the summariser off it; any other extra is idle.
func TestRenderLLMStatusNamesPurpose(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"m"}]}`))
	}))
	defer ts.Close()
	base := LLMConnection{Server: ts.URL, Model: "m", Parallel: ptr(1), ContextSize: ptr(128000)}
	status := func(conns ...LLMConnection) string {
		a := &agent{settings: Settings{LLM: conns}}
		a.probeAllLLMs(context.Background())
		return a.renderLLMStatus()
	}

	alone := status(base)
	if !strings.Contains(alone, "llm[0]: m @ "+ts.URL+" (parallel=1) · the session (planning, execution, documentation) and the background summariser") {
		t.Errorf("a lone entry must carry everything:\n%s", alone)
	}
	summ := base
	summ.Purpose = "summary"
	two := status(base, summ)
	for _, want := range []string{"llm[0]: m @ " + ts.URL + " (parallel=1) · the session (planning, execution, documentation)\n", "llm[1]: m @ " + ts.URL + " (parallel=1) · background work (the summariser, side questions), off the session's cache"} {
		if !strings.Contains(two, want) {
			t.Errorf("missing %q:\n%s", want, two)
		}
	}
	if idle := status(base, base); !strings.Contains(idle, "llm[1]: m @ "+ts.URL+" (parallel=1) · unused: nothing routes here") {
		t.Errorf("an extra with no purpose must say it is idle:\n%s", idle)
	}
}
