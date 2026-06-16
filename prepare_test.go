package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
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
				Parallel:    1,
				ContextSize: 128000, // keep the ctx gate quiet; not under test
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
	if got := a.settings.LLM[0].Parallel; got != 2 {
		t.Errorf("Parallel: got %d, want 2 (from ?model= /props total_slots)", got)
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

	a := &agent{settings: Settings{LLM: []LLMConnection{{Server: ts.URL, Model: "qwen", Parallel: 1}}}}
	a.probeAllLLMs(context.Background())

	if got := a.settings.LLM[0].Parallel; got != 1 {
		t.Errorf("Parallel: got %d, want 1 (explicit value must not be overwritten by total_slots)", got)
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

// TestProbeToolBinsMatchesInlineProbe pins that probeToolBins computes exactly the
// per-binary presence the three reporters (envSnapshot / checkEnv /
// notifyCapabilities) used to compute inline, so collapsing them into one helper
// changed no behavior. It independently re-implements the original inline loops
// and compares; both sides read the same PATH/cwd, so the result is deterministic
// regardless of which dev tools happen to be installed.
func TestProbeToolBinsMatchesInlineProbe(t *testing.T) {
	a, s := newTestAgent(t)
	s.knownStacks = []string{"go", "rust"}           // go→gopls; rust→"" (skipped)
	s.knownRunners = []string{"make", "go", "bogus"} // make→make, go→go; bogus→"" (skipped)

	gotStacks, gotRunners, gotFormatters := a.probeToolBins(s)

	// Independent re-implementation of the pre-refactor inline probe loops.
	var wantStacks, wantRunners, wantFormatters []toolPresence
	for _, st := range s.knownStacks {
		if bin := stackProbeBinary(st); bin != "" {
			_, err := exec.LookPath(bin)
			wantStacks = append(wantStacks, toolPresence{bin: bin, label: st, present: err == nil})
		}
	}
	for _, k := range s.knownRunners {
		if bin := runnerProbeBinary(k); bin != "" {
			_, err := exec.LookPath(bin)
			wantRunners = append(wantRunners, toolPresence{bin: bin, label: k, present: err == nil})
		}
	}
	for _, f := range detectFormatters(s.knownStacks, s.Cwd) {
		present := false
		if f.bin == "prettier" {
			present = prettierBin(s.Cwd) != ""
		} else if _, err := exec.LookPath(f.bin); err == nil {
			present = true
		}
		wantFormatters = append(wantFormatters, toolPresence{bin: f.bin, label: f.reason, present: present})
	}

	if !reflect.DeepEqual(gotStacks, wantStacks) {
		t.Errorf("stacks: got %+v, want %+v", gotStacks, wantStacks)
	}
	if !reflect.DeepEqual(gotRunners, wantRunners) {
		t.Errorf("runners: got %+v, want %+v", gotRunners, wantRunners)
	}
	if !reflect.DeepEqual(gotFormatters, wantFormatters) {
		t.Errorf("formatters: got %+v, want %+v", gotFormatters, wantFormatters)
	}
}
