package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

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
