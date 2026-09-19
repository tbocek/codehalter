package main

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/tbocek/codehalter/llm"
)

// TestTurnServerCache pins the server-driven accounting: with the evaluated
// (sent-but-not-cached) count reported per call the turn sums just that and flags
// haveServerCache; with -1 (no backend report) it flags no cache info and keeps
// the final context size for the fallback line — no guessing. The gross prompt
// total is no longer tracked.
func TestTurnServerCache(t *testing.T) {
	s := &Session{}
	s.startTurn(time.Now())
	s.addTurnTokens(1000, 50, 1000) // cold: 1000 evaluated (uncached)
	s.addTurnTokens(1100, 40, 100)  // 100 evaluated, the rest served from cache
	r := s.turnStats()
	if r.evaluatedPrompt != 1100 {
		t.Errorf("evaluatedPrompt: got %d, want 1100", r.evaluatedPrompt)
	}
	if !r.haveServerCache || r.completion != 90 {
		t.Errorf("haveServerCache=%v completion=%d", r.haveServerCache, r.completion)
	}

	// No server cache info → no claim, keep the final context size.
	s2 := &Session{}
	s2.startTurn(time.Now())
	s2.addTurnTokens(5000, 30, -1)
	s2.addTurnTokens(5200, 20, -1)
	r2 := s2.turnStats()
	if r2.haveServerCache {
		t.Error("haveServerCache should be false with no server data")
	}
	if r2.lastPrompt != 5200 {
		t.Errorf("lastPrompt: got %d, want 5200", r2.lastPrompt)
	}
}

// TestCacheLineage pins the rewind detector against the numbers that motivated
// it. A healthy tool-loop call gets back everything the previous call sent minus
// llama.cpp's dropped chunk; the turn that ignored preserve_thinking got back
// thousands less on four calls. First call, no-cache-info calls and compaction
// have no comparison point and must never be reported.
func TestCacheLineage(t *testing.T) {
	at := lineageClock()
	s := &Session{}
	s.startTurn(time.Now())

	// Real trace of the fixed run: cached is always prompt(n-1) - 4.
	healthy := [][2]int{{11307, 5348}, {11483, 11303}, {11666, 11479}, {12732, 11662}}
	for _, c := range healthy {
		if n := s.noteCacheLineage(c[0], c[1], "", at()).tokens; n != 0 {
			t.Errorf("prompt=%d cached=%d: reported a %d-token rewind, want none", c[0], c[1], n)
		}
	}
	if r := s.turnStats(); r.cacheRewinds != 0 {
		t.Errorf("healthy turn: cacheRewinds=%d, want 0", r.cacheRewinds)
	}

	// Real trace of the broken run: the boundary moved, the server re-read the
	// tail behind it. 65 is the same fault with nothing behind the boundary:
	// under the slack, deliberately not reported.
	s2 := &Session{}
	s2.startTurn(time.Now())
	s2.noteCacheLineage(15346, 5348, "", at()) // first call: no comparison point
	broken := [][3]int{{15346, 8475, 6871}, {17000, 7002, 8344}, {17100, 17035, 0}}
	for _, c := range broken {
		if n := s2.noteCacheLineage(c[0], c[1], "", at()).tokens; n != c[2] {
			t.Errorf("prompt=%d cached=%d: got %d, want %d", c[0], c[1], n, c[2])
		}
	}
	r2 := s2.turnStats()
	if r2.cacheRewinds != 2 || r2.cacheRewound != 6871+8344 {
		t.Errorf("broken turn: rewinds=%d rewound=%d, want 2 and %d", r2.cacheRewinds, r2.cacheRewound, 6871+8344)
	}

	// A backend that reports no cache split (cached = -1) can't be judged, and
	// must not make the NEXT call look like a rewind either.
	s3 := &Session{}
	s3.startTurn(time.Now())
	s3.noteCacheLineage(9000, -1, "", at())
	if n := s3.noteCacheLineage(9100, -1, "", at()).tokens; n != 0 {
		t.Errorf("no cache info: got %d, want 0", n)
	}
	if n := s3.noteCacheLineage(9200, 9096, "", at()).tokens; n != 0 {
		t.Errorf("after no cache info: got %d, want 0", n)
	}

	// A prompt far SMALLER than the previous one is a context reset, not a
	// rewind: nothing could have been reused because it is not the same list any
	// more. Real trace, 2026-08-21T21:24Z: a 115135-token execute call followed
	// by a 5970-token plan call at cached=0. Counting prev-cached there reports a
	// 115135-token fault that never happened. The genuine loss three hours later
	// (72033 -> 71997, a 36-token shrink from the thinking-off flip) still counts.
	s5 := &Session{}
	s5.startTurn(time.Now())
	s5.noteCacheLineage(115135, 115000, "", at())
	if n := s5.noteCacheLineage(5970, 0, "", at()).tokens; n != 0 {
		t.Errorf("context reset reported as a %d-token rewind", n)
	}
	s6 := &Session{}
	s6.startTurn(time.Now())
	s6.noteCacheLineage(72033, 71900, "", at())
	if n := s6.noteCacheLineage(71997, 0, "", at()).tokens; n != 72033 {
		t.Errorf("genuine loss after a 36-token shrink: got %d, want 72033", n)
	}

	// Compaction rewrites the front of the context on purpose.
	s4 := &Session{}
	s4.startTurn(time.Now())
	s4.noteCacheLineage(30000, 29996, "", at())
	s4.resetCacheLineage()
	if n := s4.noteCacheLineage(12000, 0, "", at()).tokens; n != 0 {
		t.Errorf("after compaction: got %d, want 0", n)
	}
}

// TestCacheLineageSpansTurns pins the blind spot that made every rewind above
// invisible in practice: the turn reset used to zero the comparison point, so
// the FIRST call of every turn was exempt — and a turn boundary is the one place
// the message list actually gets mutated (compaction, a summariser fold, a tool
// result that replays differently than it was sent). One 11.6h session logged
// zero CACHE lines while carrying a 30466-token rewind at exactly such a
// boundary. The next turn's first call is still the previous list plus one user
// message, so the premise holds and the check must survive the reset.
func TestCacheLineageSpansTurns(t *testing.T) {
	at := lineageClock()
	s := &Session{}
	s.startTurn(time.Now())
	s.noteCacheLineage(51594, 51590, "", at()) // last call of turn 1

	s.startTurn(time.Now()) // turn 2 begins
	// First call of turn 2: an image dropped out of the middle of the prompt, so
	// the server could only reuse the part in front of it.
	if n := s.noteCacheLineage(51163, 21128, "", at()).tokens; n != 51594-21128 {
		t.Errorf("first call after a turn boundary: got %d, want %d", n, 51594-21128)
	}
	if r := s.turnStats(); r.cacheRewinds != 1 {
		t.Errorf("cacheRewinds=%d, want 1 — the rewind was not reported", r.cacheRewinds)
	}

	// The per-turn counters DO reset, so turn 3 starts its report clean while
	// keeping the comparison point.
	s.startTurn(time.Now())
	if r := s.turnStats(); r.cacheRewinds != 0 || r.cacheRewound != 0 {
		t.Errorf("turn 3: rewinds=%d rewound=%d, want 0 and 0", r.cacheRewinds, r.cacheRewound)
	}
	if n := s.noteCacheLineage(51500, 51159, "", at()).tokens; n != 0 {
		t.Errorf("healthy first call of turn 3: got %d, want 0", n)
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

// TestKeepWindowStartDegenerate pins that the keep-window doesn't panic on an
// empty session or an out-of-range turnStartIdx (the old unguarded
// s.Messages[last] dereference).
func TestKeepWindowStartDegenerate(t *testing.T) {
	if got := (&Session{}).keepWindowStart(10_000); got != 0 {
		t.Errorf("empty session: keepWindowStart = %d, want 0", got)
	}
	s := &Session{}
	s.AddUser("p")
	s.AddAssistant("a")
	s.turnStartIdx = 99           // past the end
	_ = s.keepWindowStart(10_000) // must not panic
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
	base := s.turnStartIdx + 1
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
	b2 := s2.turnStartIdx + 1
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

// TestCacheLineageNamesTheRenderChange pins the attribution half of the rewind
// detector. The token counts alone say the prompt was re-rendered; they cannot
// say who did it. Recording the llm.RenderKey of each call answers that, and the
// two answers have nothing in common: a rendering that changed is one line of
// settings.toml, a rendering that held is compaction, a tool result that
// replayed differently, or a server-side eviction.
//
// The trace here is the real one from an 11.6h session against a one-slot
// server: the thinking-off retry flipped enable_thinking mid-run, the next call
// came back cached=0 on a 71997-token prompt, and switching back re-read 27585
// more. 99582 tokens for one setting, and nothing in the log said so.
//
// codehalter no longer causes this: the retry appends a closed <think></think>
// instead (see llm.Conn.WithThinkingDisabled). The detector stays because a settings.toml
// whose two roles disagree on chat_template_kwargs reproduces it exactly, and
// the numbers below are what that costs.
func TestCacheLineageNamesTheRenderChange(t *testing.T) {
	at := lineageClock()
	const (
		think = `{"chat_template_kwargs":{"preserve_thinking":true}}`
		exec  = `{"chat_template_kwargs":{"enable_thinking":false}}`
	)
	s := &Session{}
	s.startTurn(time.Now())
	s.noteCacheLineage(71997, 71000, think, at())

	// Same conversation, different rendering asked for: the server had nothing
	// to reuse.
	rw := s.noteCacheLineage(71997, 0, exec, at())
	if rw.tokens != 71997 || !rw.renderChanged {
		t.Errorf("flip to %s: got n=%d changed=%v, want 71997 and true", exec, rw.tokens, rw.renderChanged)
	}
	if rw.prevRender != think {
		t.Errorf("previous rendering = %q, want %q", rw.prevRender, think)
	}

	// The run continues under the new rendering and the prefix holds again: the
	// cost is the switch, not the setting.
	if rw := s.noteCacheLineage(99614, 71993, exec, at()); rw.tokens != 0 || rw.renderChanged {
		t.Errorf("second call under the new rendering: got n=%d changed=%v, want 0 and false", rw.tokens, rw.renderChanged)
	}

	// Switching back re-reads what the other rendering left behind.
	if rw := s.noteCacheLineage(99614, 72029, think, at()); rw.tokens != 27585 || !rw.renderChanged {
		t.Errorf("flip back: got n=%d changed=%v, want 27585 and true", rw.tokens, rw.renderChanged)
	}
	if r := s.turnStats(); r.cacheRewinds != 2 || r.cacheRewindsRender != 2 {
		t.Errorf("rewinds=%d of which render changes=%d, want 2 and 2", r.cacheRewinds, r.cacheRewindsRender)
	}

	// A rewind with the rendering unchanged is a different fault and must not be
	// blamed on the settings.
	s2 := &Session{}
	s2.startTurn(time.Now())
	s2.noteCacheLineage(51594, 51590, think, at())
	if rw := s2.noteCacheLineage(51163, 21128, think, at()); rw.tokens == 0 || rw.renderChanged {
		t.Errorf("rewind with a stable rendering: got n=%d changed=%v, want a rewind and false", rw.tokens, rw.renderChanged)
	}
	if r := s2.turnStats(); r.cacheRewinds != 1 || r.cacheRewindsRender != 0 {
		t.Errorf("rewinds=%d of which render changes=%d, want 1 and 0", r.cacheRewinds, r.cacheRewindsRender)
	}

	// Compaction drops the whole comparison point, rendering included, so the
	// call after it is neither a rewind nor a render change.
	s2.resetCacheLineage()
	if rw := s2.noteCacheLineage(9000, 0, exec, at()); rw.tokens != 0 || rw.renderChanged || rw.prevRender != "" {
		t.Errorf("after compaction: got n=%d prev=%q changed=%v, want 0, \"\" and false", rw.tokens, rw.prevRender, rw.renderChanged)
	}
}

// TestCacheLineageTimesTheGap pins the field that tells the two remaining
// causes apart once the rendering is ruled out. A prompt/cached pair looks
// exactly the same whether something rewrote the middle of the prompt or the
// server simply reclaimed a slot we left sitting, and only one of those is
// worth acting on. The gap before the call is the discriminator: on the 11.6h
// session that motivated the detector every same-rendering rewind sat behind a
// gap of minutes or hours, while the hundreds of calls seconds apart never lost
// a prefix.
func TestCacheLineageTimesTheGap(t *testing.T) {
	base := time.Date(2026, 8, 21, 22, 0, 0, 0, time.UTC)
	s := &Session{}
	s.startTurn(base)

	// First call of a lineage: there is no previous call to be idle since, and
	// reporting the process uptime here would read as a two-hour stall.
	if rw := s.noteCacheLineage(51594, 51590, "", base); rw.idle != 0 {
		t.Errorf("first call: idle=%v, want 0", rw.idle)
	}
	// A tool loop's own cadence, measured whether or not anything went wrong.
	if rw := s.noteCacheLineage(51700, 51590, "", base.Add(3*time.Second)); rw.idle != 3*time.Second {
		t.Errorf("healthy call: idle=%v, want 3s", rw.idle)
	}
	// The real shape of the logged eviction: two hours between turns, then a
	// rewind the rendering cannot explain. The gap has to reach the call that
	// reports the rewind, not just the ones before it.
	rw := s.noteCacheLineage(51700, 0, "", base.Add(2*time.Hour+6*time.Minute))
	if rw.tokens == 0 {
		t.Fatal("the rewind itself went unreported")
	}
	if rw.idle < idleEvictionSuspect {
		t.Errorf("idle=%v, want at least %v so the log can name it an eviction", rw.idle, idleEvictionSuspect)
	}
	// Compaction drops the clock along with the rest of the comparison point:
	// the call after it must not be charged for a gap it never sat through.
	s.resetCacheLineage()
	if rw := s.noteCacheLineage(9000, 0, "", base.Add(3*time.Hour)); rw.idle != 0 {
		t.Errorf("after compaction: idle=%v, want 0", rw.idle)
	}
}

// TestTurnStatsNamesDiscardedDecode pins the split between generation the user
// got and generation the harness threw away. They are the same tokens to the
// server and the same seconds on the clock, so the discarded half has to be
// counted inside the completion total; but left unnamed there it reads as
// output, and the worst turns report as the most productive ones. On this
// hardware it is the single most expensive thing that can go wrong in a turn:
// one measured <think> stall burned the full 8192-token cap, 3m36s at 37.7
// tok/s, against 57s for the prefix loss the same event caused.
func TestTurnStatsNamesDiscardedDecode(t *testing.T) {
	s := &Session{}
	s.startTurn(time.Now())
	s.addTurnTokens(11000, 8192, 900) // the stalled call, generated then dropped
	s.addWastedCompletion(8192)
	s.addTurnTokens(11400, 400, 300) // the retry that produced the answer

	r := s.turnStats()
	if r.completion != 8592 {
		t.Errorf("completion=%d, want 8592: discarded decode is still decode the server performed", r.completion)
	}
	if r.wastedCompletion != 8192 {
		t.Errorf("wastedCompletion=%d, want 8192", r.wastedCompletion)
	}

	// Per turn, like every other figure on the Done line: a stall in the turn
	// before this one is not this turn's cost.
	s.startTurn(time.Now())
	if r := s.turnStats(); r.wastedCompletion != 0 {
		t.Errorf("after startTurn: wastedCompletion=%d, want 0", r.wastedCompletion)
	}
}

// TestRenderKeyIgnoresSamplers pins what goes into the fingerprint. Samplers
// never reach the chat template, so two roles may differ in them without
// costing a re-prefill; anything else must be treated as a rendering change,
// including fields nobody has thought of yet (Qwen3.8 reads a top-level
// reasoning_effort, for instance).
func TestRenderKeyIgnoresSamplers(t *testing.T) {
	plan := llm.RenderKey(map[string]any{"temperature": 1.0, "top_p": 0.95, "max_tokens": 8000})
	exec := llm.RenderKey(map[string]any{"temperature": 0.6, "top_p": 0.8, "max_tokens": 4000})
	if plan != "" || exec != "" {
		t.Errorf("samplers entered the key: thinking=%q execute=%q", plan, exec)
	}

	// Same kwargs, written in a different order, must fingerprint identically:
	// Go map iteration is randomised, and a key that flapped would report a
	// rewind on every other call.
	a := llm.RenderKey(map[string]any{"temperature": 1.0, "chat_template_kwargs": map[string]any{"preserve_thinking": true, "enable_thinking": true}})
	b := llm.RenderKey(map[string]any{"temperature": 0.6, "chat_template_kwargs": map[string]any{"enable_thinking": true, "preserve_thinking": true}})
	if a != b || a == "" {
		t.Errorf("kwargs key is not canonical: %q vs %q", a, b)
	}

	// An unknown non-sampler counts: assuming it is harmless is how the
	// expensive kind of rewind goes unnoticed.
	if k := llm.RenderKey(map[string]any{"reasoning_effort": "low"}); k == "" {
		t.Error("reasoning_effort was dropped from the key")
	}
}

// TestSessionRoundtrip verifies the TOML schema for Session is stable: what we
// write in memory comes back byte-for-byte on reload.
func TestSessionRoundtrip(t *testing.T) {
	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.AddUser("hello")
	s.AddAssistant("hi there")
	s.AppendToolUse(ToolUse{Name: "read_file", Input: `{"path":"x.go"}`, Output: "file content"})
	s.Summary = "earlier summary"
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := loadSession(dir, s.ID)
	if err != nil {
		t.Fatalf("loadSession: %v", err)
	}
	if got := len(loaded.Messages); got != 2 {
		t.Fatalf("Messages len: got %d, want 2", got)
	}
	if loaded.Messages[0].Role != "user" || loaded.Messages[0].Content != "hello" {
		t.Errorf("Messages[0]: got %+v", loaded.Messages[0])
	}
	if loaded.Messages[1].Role != "assistant" || loaded.Messages[1].Content != "hi there" {
		t.Errorf("Messages[1]: got %+v", loaded.Messages[1])
	}
	if got := len(loaded.Messages[1].ToolUses); got != 1 {
		t.Fatalf("ToolUses len: got %d, want 1", got)
	}
	tu := loaded.Messages[1].ToolUses[0]
	if tu.Name != "read_file" || tu.Output != "file content" {
		t.Errorf("ToolUse: got %+v", tu)
	}
	if loaded.Summary != "earlier summary" {
		t.Errorf("Summary mismatch: got %q, want %q", loaded.Summary, "earlier summary")
	}
}

// TestNewSessionKeepsTheOneBeforeIt pins the collision guard. Session ids have
// second granularity, so opening a second session in the same second (the CLI's
// /new does exactly that) used to hand back the id already on disk, and the
// first save wiped the conversation it named.
func TestNewSessionKeepsTheOneBeforeIt(t *testing.T) {
	dir := t.TempDir()
	first, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	first.AddUser("keep me")
	if err := first.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	second, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession (second): %v", err)
	}
	if second.ID == first.ID {
		t.Fatalf("second session reused id %q", first.ID)
	}
	second.AddUser("and me")
	if err := second.Save(); err != nil {
		t.Fatalf("Save (second): %v", err)
	}

	loaded, err := loadSession(dir, first.ID)
	if err != nil {
		t.Fatalf("loadSession(%q): %v", first.ID, err)
	}
	if len(loaded.Messages) != 1 || loaded.Messages[0].Content != "keep me" {
		t.Errorf("first session was overwritten: %+v", loaded.Messages)
	}
}

// TestAppendToolUseCreatesAssistantMessage verifies that recording a tool use
// when the last message is a user turn creates a new empty assistant message
// to hold it (rather than attaching to the user).
func TestAppendToolUseCreatesAssistantMessage(t *testing.T) {
	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.AddUser("do a thing")
	s.AppendToolUse(ToolUse{Name: "read_file", Input: "{}", Output: "ok"})

	if got := len(s.Messages); got != 2 {
		t.Fatalf("Messages len: got %d, want 2", got)
	}
	if s.Messages[1].Role != "assistant" {
		t.Errorf("expected assistant message to be created, got role %q", s.Messages[1].Role)
	}
	if len(s.Messages[1].ToolUses) != 1 {
		t.Errorf("tool use not appended to assistant message: %+v", s.Messages[1])
	}

	// A second tool use must stay on the same assistant message.
	s.AppendToolUse(ToolUse{Name: "write_file", Input: "{}", Output: "ok"})
	if got := len(s.Messages); got != 2 {
		t.Fatalf("Messages len after second AppendToolUse: got %d, want 2", got)
	}
	if got := len(s.Messages[1].ToolUses); got != 2 {
		t.Fatalf("ToolUses len: got %d, want 2", got)
	}
}

// TestConcurrentSessionWritesAreRaceFree exercises the Session mutex by
// racing session mutations against concurrent Save() calls. Run with -race to
// catch regressions in the locking that fix #4 introduced.
func TestConcurrentSessionWritesAreRaceFree(t *testing.T) {
	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}

	const workers = 8
	const perWorker = 50
	var wg sync.WaitGroup
	wg.Add(workers * 3)

	// Writers: exercise each mutator type concurrently.
	for i := 0; i < workers; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < perWorker; j++ {
				s.AddUser("u")
			}
		}()
		go func() {
			defer wg.Done()
			for j := 0; j < perWorker; j++ {
				s.AppendToolUse(ToolUse{Name: "x", Input: "{}", Output: "ok"})
			}
		}()
		go func() {
			defer wg.Done()
			for j := 0; j < perWorker; j++ {
				_ = s.Save()
			}
		}()
	}
	wg.Wait()

	// Loosely assert that all AddUser calls landed — exact count verifies
	// that no append was lost to a concurrent slice-grow race.
	userCount := 0
	for _, m := range s.Messages {
		if m.Role == "user" {
			userCount++
		}
	}
	if want := workers * perWorker; userCount != want {
		t.Errorf("user message count: got %d, want %d", userCount, want)
	}
}

// TestExternalChangeDrift covers the detector that tells the model a file was
// rewritten behind it. The failure it exists for: an editor's format-on-save
// reindents a file after codehalter writes it, and the model's next edit_file
// fails on old_text that was correct when it read it.
func TestExternalChangeDrift(t *testing.T) {
	s := &Session{}
	const path = "/w/proj/src/app.js"

	// A file we never wrote is not tracked: it changing is not drift.
	s.checkExternalChange(path, "whatever")
	if note := s.takeDriftNote(path); note != "" {
		t.Errorf("untracked file produced a note: %q", note)
	}

	s.recordWrite(path, "let a = 1;\n")
	s.checkExternalChange(path, "let a = 1;\n")
	if note := s.takeDriftNote(path); note != "" {
		t.Errorf("unchanged file produced a note: %q", note)
	}

	// Reindented behind us: exactly one note, delivered once.
	s.checkExternalChange(path, "let  a  =  1;\n")
	note := s.takeDriftNote(path)
	if note == "" {
		t.Fatal("a file rewritten behind us produced no note")
	}
	if again := s.takeDriftNote(path); again != "" {
		t.Errorf("note delivered twice: %q", again)
	}
	// The hash advanced to what is on disk now, so the same content is no longer
	// drift — only a NEW external change is.
	s.checkExternalChange(path, "let  a  =  1;\n")
	if n := s.takeDriftNote(path); n != "" {
		t.Errorf("already-reported drift reported again: %q", n)
	}
	s.checkExternalChange(path, "let a = 2;\n")
	if n := s.takeDriftNote(path); n == "" {
		t.Error("a second, distinct external change produced no note")
	}

	// Our own write settles the question: whatever the other writer did, we have
	// just replaced it, so there is nothing left to warn about.
	s.recordWrite(path, "let a = 3;\n")
	s.checkExternalChange(path, "let a = 4;\n")
	s.recordWrite(path, "let a = 5;\n")
	if n := s.takeDriftNote(path); n != "" {
		t.Errorf("pending note survived our own write: %q", n)
	}
}
