package main

import (
	"testing"
	"time"
)

// TestTurnServerCache pins the server-driven accounting: with the cache split
// reported (evaluated, cached per call) the turn sums them and flags
// haveServerCache; with -1/-1 (no backend report) it flags no cache info and
// keeps the final context size for the fallback line — no guessing.
func TestTurnServerCache(t *testing.T) {
	s := &Session{}
	s.resetTurnStats(time.Now())
	s.addTurnTokens(1000, 50, 1000, 0)   // cold: 1000 evaluated, 0 cached
	s.addTurnTokens(1100, 40, 100, 1000) // 100 evaluated, 1000 reused
	r := s.turnStats()
	if r.sentPrompt != 2100 {
		t.Errorf("sentPrompt: got %d, want 2100", r.sentPrompt)
	}
	if r.evaluatedPrompt != 1100 {
		t.Errorf("evaluatedPrompt: got %d, want 1100", r.evaluatedPrompt)
	}
	if r.cachedPrompt != 1000 {
		t.Errorf("cachedPrompt: got %d, want 1000", r.cachedPrompt)
	}
	if !r.haveServerCache || r.completion != 90 {
		t.Errorf("haveServerCache=%v completion=%d", r.haveServerCache, r.completion)
	}

	// No server cache info → no claim, keep the final context size.
	s2 := &Session{}
	s2.resetTurnStats(time.Now())
	s2.addTurnTokens(5000, 30, -1, -1)
	s2.addTurnTokens(5200, 20, -1, -1)
	r2 := s2.turnStats()
	if r2.haveServerCache {
		t.Error("haveServerCache should be false with no server data")
	}
	if r2.lastPrompt != 5200 {
		t.Errorf("lastPrompt: got %d, want 5200", r2.lastPrompt)
	}
}

// TestUpsertLastAssistant pins both branches of UpsertLastAssistant: append a
// new assistant turn when the trailing role is not assistant, overwrite the
// existing one otherwise.
func TestUpsertLastAssistant(t *testing.T) {
	s := &Session{}

	// Empty → append.
	s.UpsertLastAssistant("first")
	if len(s.Messages) != 1 || s.Messages[0].Role != "assistant" || s.Messages[0].Content != "first" {
		t.Fatalf("empty case: got %+v", s.Messages)
	}

	// Trailing assistant → overwrite.
	s.UpsertLastAssistant("replaced")
	if len(s.Messages) != 1 || s.Messages[0].Content != "replaced" {
		t.Fatalf("overwrite case: got %+v", s.Messages)
	}

	// Trailing user → append.
	s.AddUser("question")
	s.UpsertLastAssistant("answer")
	if len(s.Messages) != 3 {
		t.Fatalf("append case: got %d messages, want 3", len(s.Messages))
	}
	if s.Messages[2].Role != "assistant" || s.Messages[2].Content != "answer" {
		t.Errorf("tail: got %+v", s.Messages[2])
	}
}

// TestSessionTurnControl pins the cancel-vs-redirect distinction that fixes the
// limbo: the Cancel button stops the turn (not a redirect); a new Prompt cancels
// the in-flight turn AND marks it interrupted (continue with the new message).
func TestSessionTurnControl(t *testing.T) {
	s := &Session{}
	cancelled := false
	s.beginTurn(func() { cancelled = true })
	if s.wasInterrupted() {
		t.Error("a fresh turn should not be interrupted")
	}

	s.cancelTurn() // Cancel button
	if !cancelled || s.wasInterrupted() {
		t.Errorf("cancelTurn: cancelled=%v interrupted=%v, want true/false", cancelled, s.wasInterrupted())
	}

	cancelled = false
	s.beginTurn(func() { cancelled = true })
	s.interruptForPrompt() // user typed a new message
	if !cancelled || !s.wasInterrupted() {
		t.Errorf("interruptForPrompt: cancelled=%v interrupted=%v, want true/true", cancelled, s.wasInterrupted())
	}

	s.beginTurn(func() {}) // a new turn clears the redirect flag
	if s.wasInterrupted() {
		t.Error("beginTurn should clear interrupted")
	}
}
