package main

import (
	"fmt"
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

// TestSessionTurnControl pins the turn-cancel handle: beginTurn registers the
// in-flight cancel, cancelTurn fires it (Cancel button OR a new prompt
// superseding). The caller never waits, so a wedged turn can't block the next.
func TestSessionTurnControl(t *testing.T) {
	s := &Session{}
	cancelled := false
	s.beginTurn(func() { cancelled = true })
	s.cancelTurn()
	if !cancelled {
		t.Error("cancelTurn should fire the registered cancel")
	}

	// A new turn overwrites the cancel; cancelTurn now fires the NEW one only.
	first, second := false, false
	s.beginTurn(func() { first = true })
	s.beginTurn(func() { second = true })
	s.cancelTurn()
	if first || !second {
		t.Errorf("cancelTurn should fire only the latest turn: first=%v second=%v", first, second)
	}
}

// TestSessionSupersedeFlag pins the supersede flag that lets a cancelled turn
// tell an editor abort (surface the reason) from a new-prompt supersede (stay
// silent). markSuperseding sets it, superseded reads it, adoptTurn clears it
// once the replacement turn has taken over.
func TestSessionSupersedeFlag(t *testing.T) {
	s := &Session{}
	if s.superseded() {
		t.Error("fresh session should not be flagged superseded")
	}
	s.markSuperseding()
	if !s.superseded() {
		t.Error("markSuperseding should set the flag")
	}
	s.adoptTurn()
	if s.superseded() {
		t.Error("adoptTurn should clear the flag")
	}
}

// TestKeepWindowStart pins the 400-recovery keep window sized by REAL server
// prompt_tokens: the unfinished small turn plus the most recent completed small
// turns under the budget — never the whole oversized in-flight turn (the 194 KB
// bug), and only the unfinished turn when no token usage is reported.
func TestKeepWindowStart(t *testing.T) {
	dir := t.TempDir()

	// Oversized turn: cumulative prompt_tokens grow [1k,4k,7k,15k,25k]. Keeping
	// from K costs ref(25k) - PromptTokens[K]; under a 10k budget only step 3
	// (25k-15k=10k) fits, step 2 (25k-7k=18k) does not.
	s, _ := newSession(dir)
	s.AddUser("task prompt")
	s.markTurnStart()
	for i := 0; i < 5; i++ {
		s.AddAssistant(fmt.Sprintf("step %d", i))
	}
	base := s.turnStartIndex() + 1
	for i, pt := range []int{1000, 4000, 7000, 15000, 25000} {
		s.Messages[base+i].PromptTokens = pt
	}
	keep := s.keepWindowStart(10_000)
	if kept := len(s.Messages) - keep; kept != 2 {
		t.Errorf("oversized: kept %d, want 2 (unfinished + step 3)", kept)
	}
	if s.Messages[keep].Content != "step 3" {
		t.Errorf("oversized: kept window starts at %q, want 'step 3'", s.Messages[keep].Content)
	}

	// Small turn: all completed small turns fit under budget, so all are kept and
	// only the prompt folds (keepFrom = the first assistant message).
	s2, _ := newSession(dir)
	s2.AddUser("p")
	s2.markTurnStart()
	for i := 0; i < 3; i++ {
		s2.AddAssistant(fmt.Sprintf("a%d", i))
	}
	b2 := s2.turnStartIndex() + 1
	for i, pt := range []int{500, 1500, 3000} {
		s2.Messages[b2+i].PromptTokens = pt
	}
	if got := s2.keepWindowStart(10_000); got != b2 {
		t.Errorf("small turn: keepWindowStart=%d, want %d (keep all small turns)", got, b2)
	}

	// No token usage reported (PromptTokens == 0) → keep only the unfinished turn.
	s3, _ := newSession(dir)
	s3.AddUser("p")
	s3.markTurnStart()
	s3.AddAssistant("a1")
	s3.AddAssistant("a2")
	if got := s3.keepWindowStart(10_000); got != s3.lastAssistantIndex() {
		t.Errorf("no usage: keepWindowStart=%d, want lastAssistantIndex=%d", got, s3.lastAssistantIndex())
	}
}
