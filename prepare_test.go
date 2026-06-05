package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
	// Baseline: seed the always-on skills and freeze them as "already in the
	// prompt", so only a NEW skill counts as added mid-session.
	if err := ensureSkills(s.Cwd, nil, readOSInfo()); err != nil {
		t.Fatal(err)
	}
	const frozen = "EXISTING PROMPT — do not mutate"
	s.SystemPrompt = frozen
	s.promptSkills = skillFiles(s.Cwd)

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

// TestDetectFormatters covers both drivers: detected stack (ts → prettier) and
// formatter config files (.clang-format → clang-format, pyproject [tool.ruff] →
// ruff), and that an empty project needs nothing.
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

// TestMcpServerConfigured pins lsmcp detection: no file / commented entry read
// as not-configured (so the setup card fires), an active entry as configured.
func TestMcpServerConfigured(t *testing.T) {
	dir := t.TempDir()
	if mcpServerConfigured(dir, "lsmcp") {
		t.Error("no mcp.toml should be not-configured")
	}
	cfgDir := filepath.Join(dir, sessionDir)
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	mcpPath := filepath.Join(cfgDir, "mcp.toml")

	if err := os.WriteFile(mcpPath, []byte("# [[server]]\n# name = \"lsmcp\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if mcpServerConfigured(dir, "lsmcp") {
		t.Error("commented lsmcp should be not-configured")
	}

	if err := os.WriteFile(mcpPath, []byte("[[server]]\nname = \"lsmcp\"\ncommand = \"npx\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !mcpServerConfigured(dir, "lsmcp") {
		t.Error("active lsmcp should be configured")
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
				Parallel:     1,
				ContextSize:  128000,
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
				Parallel: 1,
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
				Parallel:     1,
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

	if got := a.settings.LLM[0].Parallel; got != 2 {
		t.Errorf("Parallel: got %d, want 2 (auto-detected from /props total_slots)", got)
	}
	if a.mainSlotTokens != 16384 {
		t.Errorf("mainSlotTokens: got %d, want 16384 (per-slot n_ctx used directly, no division)", a.mainSlotTokens)
	}
}

// TestProbeAllLLMsRouterUpstreamProps pins the llama-swap fix: the router's own
// /props always reports n_ctx=0 (role:"router") — even with a model loaded — so
// probeLLM must read /upstream/<model>/props to get the real per-slot n_ctx +
// total_slots (that path also auto-loads the model). Without it the n_ctx-unknown
// gate would block forever.
func TestProbeAllLLMsRouterUpstreamProps(t *testing.T) {
	var upstreamHit atomic.Bool
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.HasSuffix(r.URL.Path, "/v1/models"):
			_, _ = w.Write([]byte(`{"data":[{"id":"qwen"}]}`))
		case strings.Contains(r.URL.Path, "/upstream/qwen/props"):
			upstreamHit.Store(true) // real model props behind the router (auto-loads)
			fmt.Fprint(w, `{"default_generation_settings":{"n_ctx":16384},"total_slots":2}`)
		case strings.HasSuffix(r.URL.Path, "/props"):
			fmt.Fprint(w, `{"role":"router","model_path":"none","default_generation_settings":{"n_ctx":0},"total_slots":0}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	a := &agent{settings: Settings{LLM: []LLMConnection{{Server: ts.URL, Model: "qwen"}}}}
	a.probeAllLLMs(context.Background())

	if !upstreamHit.Load() {
		t.Fatal("never read /upstream/qwen/props — router /props (n_ctx=0) was taken at face value")
	}
	if got := a.settings.LLM[0].Parallel; got != 2 {
		t.Errorf("Parallel: got %d, want 2 (from upstream /props total_slots)", got)
	}
	if a.mainSlotTokens != 16384 {
		t.Errorf("mainSlotTokens: got %d, want 16384 (per-slot n_ctx from upstream /props)", a.mainSlotTokens)
	}
}

// TestProbeAllLLMsExplicitParallelWins: an explicit `parallel` is never
// overwritten by the probed total_slots.
func TestProbeAllLLMsExplicitParallelWins(t *testing.T) {
	ts := llamaCppServer(16384, 4)
	defer ts.Close()

	a := &agent{settings: Settings{LLM: []LLMConnection{{Server: ts.URL, Model: "qwen", Parallel: 1}}}}
	a.probeAllLLMs(context.Background())

	if got := a.settings.LLM[0].Parallel; got != 1 {
		t.Errorf("Parallel: got %d, want 1 (explicit value must not be overwritten by total_slots)", got)
	}
}
