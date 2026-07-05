package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/BurntSushi/toml"
)

// Config is the whole crafter.toml. One [judge] table names the strong
// reference model that segments skills, authors questions, and judges A/B
// answers. Each [[model]] table is one target model being profiled — the
// thing we decide statements for. [settings] holds run-wide knobs.
//
// Endpoints are OpenAI-compatible (server = host root only, no /v1 path);
// the client appends /v1/chat/completions itself, matching how the main
// codehalter LLMConnection works.
type Config struct {
	Settings Settings    `toml:"settings"`
	Judge    ModelSpec   `toml:"judge"`
	Models   []ModelSpec `toml:"model"`
}

// Settings are run-wide knobs. Samples is the number of A/B repetitions per
// claim; the keep/drop verdict is the majority across them. Skills optionally
// restricts which ground-skills/SKILL-*.md files are probed (bare stack names,
// e.g. ["go","base"]); empty means every SKILL-*.md in ground-skills/.
type Settings struct {
	Samples int      `toml:"samples"`
	Skills  []string `toml:"skills"`
}

// ModelSpec is one OpenAI-compatible endpoint. Name is the folder under
// models/ where this target's results and pruned skills land (the judge has
// no folder, so its Name is ignored). Server is the host root only. Model is
// the id the server exposes at /v1/models. Temperature/TopP/MaxTokens are
// optional sampler overrides sent verbatim in the request body.
//
// Parallel caps concurrent calls to this endpoint, matching the server's slot
// count (llama-server -np); default 1. sem enforces it client-side: llama.cpp
// queues a request for a busy slot WITHOUT sending headers, so an
// over-parallel client turns queue wait into "timeout awaiting response
// headers" (measured: prefetch segmentation vs probe verdicts on a 1-slot
// judge). Queuing on our side has no timeout. The chan is shared by all
// copies of the spec.
type ModelSpec struct {
	Name        string   `toml:"name"`
	Server      string   `toml:"server"`
	APIKey      string   `toml:"api_key"`
	Model       string   `toml:"model"`
	Temperature *float64 `toml:"temperature"`
	TopP        *float64 `toml:"top_p"`
	MaxTokens   *int     `toml:"max_tokens"`
	Parallel    int      `toml:"parallel"`

	// Params is merged verbatim into every request body for this endpoint
	// (crafter's version of the main package's ExtraBody) — top_k, min_p,
	// presence_penalty, chat_template_kwargs, anything the server accepts.
	// Set AFTER the named fields above, so a key here overrides them.
	Params map[string]any `toml:"params"`

	sem chan struct{}
}

// endpoint joins the server root with an API path — endpoint("/v1/models") →
// "http://host:8080/v1/models". Trailing slashes on Server are tolerated. Same
// contract as the main package's LLMConnection.endpoint.
func (m ModelSpec) endpoint(path string) string {
	return strings.TrimRight(m.Server, "/") + path
}

// loadConfig reads and validates crafter.toml. A run is worthless without a
// reachable judge and at least one target, so both are hard errors here rather
// than a confusing failure deep in the probe loop.
func loadConfig(path string) (*Config, error) {
	var cfg Config
	if _, err := toml.DecodeFile(path, &cfg); err != nil {
		return nil, fmt.Errorf("decode %s: %w", path, err)
	}

	if cfg.Settings.Samples <= 0 {
		cfg.Settings.Samples = 3
	}

	if cfg.Judge.Server == "" || cfg.Judge.Model == "" {
		return nil, fmt.Errorf("%s: [judge] needs both server and model", path)
	}
	// Label for logs and the llm.jsonl trace — the judge is the main/reference
	// model, so calls read "call main (…)" unless the user names it.
	if cfg.Judge.Name == "" {
		cfg.Judge.Name = "main"
	}
	// Parallel is resolved later (resolveParallelism): 0/absent means
	// auto-detect from the server's slot count, a positive value is an
	// explicit cap. The semaphore is built there too, once the endpoints are
	// known reachable.
	if len(cfg.Models) == 0 {
		return nil, fmt.Errorf("%s: at least one [[model]] required", path)
	}
	seen := map[string]bool{}
	for i, m := range cfg.Models {
		if m.Name == "" {
			return nil, fmt.Errorf("%s: [[model]] #%d needs a name (used as the models/ folder)", path, i+1)
		}
		if m.Server == "" || m.Model == "" {
			return nil, fmt.Errorf("%s: model %q needs both server and model", path, m.Name)
		}
		if seen[m.Name] {
			return nil, fmt.Errorf("%s: duplicate model name %q", path, m.Name)
		}
		seen[m.Name] = true
	}
	return &cfg, nil
}

// resolveParallelism finalizes each endpoint's concurrency cap and builds its
// semaphore. An endpoint with an explicit positive `parallel` in the config
// keeps it; one left unset (0) is auto-detected from the server's slot count
// (llama.cpp -np via /props total_slots), falling back to 1 when the server
// can't be probed. detect is injected so it can be faked in tests. Call once,
// after preflight has confirmed the endpoints are reachable.
//
// The cap must not exceed the server's real slots: a client that sends more
// requests than there are slots makes the server queue the extras WITHOUT
// sending headers (the "timeout awaiting response headers" failure) — so an
// explicit value is the operator's responsibility, and the auto path uses the
// server's own reported count, which is correct by construction.
func (cfg *Config) resolveParallelism(detect func(ModelSpec) int, logf func(string, ...any)) {
	finalize := func(m *ModelSpec) {
		switch {
		case m.Parallel >= 1:
			logf("  %s (%s): parallel = %d (configured)", m.Name, m.Model, m.Parallel)
		default:
			if n := detect(*m); n >= 1 {
				m.Parallel = n
				logf("  %s (%s): auto-detected %d slot(s) → parallel = %d", m.Name, m.Model, n, n)
			} else {
				m.Parallel = 1
				logf("  %s (%s): slot count not detected → parallel = 1 (set it in crafter.toml to override)", m.Name, m.Model)
			}
		}
		m.sem = make(chan struct{}, m.Parallel)
	}
	finalize(&cfg.Judge)
	for i := range cfg.Models {
		finalize(&cfg.Models[i])
	}
}

// wantSkill reports whether a bare stack name (e.g. "go" from SKILL-go.md)
// should be probed given the Settings.Skills filter. Empty filter = probe all.
func (s Settings) wantSkill(stack string) bool {
	if len(s.Skills) == 0 {
		return true
	}
	for _, w := range s.Skills {
		if w == stack {
			return true
		}
	}
	return false
}

// mustExist is a small guard used by main to fail fast with a clear message
// when a required path (crafter.toml, ground-skills/) is missing.
func mustExist(path, what string) error {
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("%s not found at %s: %w", what, path, err)
	}
	return nil
}
