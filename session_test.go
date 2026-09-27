package main

import (
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"
	"unicode/utf8"
)

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

// First call, no-cache-info calls and compaction have no comparison point and are never reported.
func TestCacheLineage(t *testing.T) {
	at := lineageClock()
	s := &Session{}
	s.startTurn(time.Now())

	// Healthy: cached is always the previous prompt minus 4.
	healthy := [][2]int{{11307, 5348}, {11483, 11303}, {11666, 11479}, {12732, 11662}}
	for _, c := range healthy {
		if n := s.noteCacheLineage(c[0], c[1], "", at()).tokens; n != 0 {
			t.Errorf("prompt=%d cached=%d: reported a %d-token rewind, want none", c[0], c[1], n)
		}
	}
	if r := s.turnStats(); r.cacheRewinds != 0 {
		t.Errorf("healthy turn: cacheRewinds=%d, want 0", r.cacheRewinds)
	}

	// The last case loses 65 tokens: under the slack, deliberately not reported.
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

	// No cache split can't be judged and must not make the next call look like a rewind.
	s3 := &Session{}
	s3.startTurn(time.Now())
	s3.noteCacheLineage(9000, -1, "", at())
	if n := s3.noteCacheLineage(9100, -1, "", at()).tokens; n != 0 {
		t.Errorf("no cache info: got %d, want 0", n)
	}
	if n := s3.noteCacheLineage(9200, 9096, "", at()).tokens; n != 0 {
		t.Errorf("after no cache info: got %d, want 0", n)
	}

	// A much smaller prompt is a context reset, not a rewind; a small shrink still counts.
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

	s4 := &Session{}
	s4.startTurn(time.Now())
	s4.noteCacheLineage(30000, 29996, "", at())
	s4.resetCacheLineage()
	if n := s4.noteCacheLineage(12000, 0, "", at()).tokens; n != 0 {
		t.Errorf("after compaction: got %d, want 0", n)
	}
}

// A turn boundary is where the message list mutates, so the comparison point must survive startTurn.
func TestCacheLineageSpansTurns(t *testing.T) {
	at := lineageClock()
	s := &Session{}
	s.startTurn(time.Now())
	s.noteCacheLineage(51594, 51590, "", at())

	s.startTurn(time.Now())
	if n := s.noteCacheLineage(51163, 21128, "", at()).tokens; n != 51594-21128 {
		t.Errorf("first call after a turn boundary: got %d, want %d", n, 51594-21128)
	}
	if r := s.turnStats(); r.cacheRewinds != 1 {
		t.Errorf("cacheRewinds=%d, want 1 — the rewind was not reported", r.cacheRewinds)
	}

	// The per-turn counters do reset; the comparison point does not.
	s.startTurn(time.Now())
	if r := s.turnStats(); r.cacheRewinds != 0 || r.cacheRewound != 0 {
		t.Errorf("turn 3: rewinds=%d rewound=%d, want 0 and 0", r.cacheRewinds, r.cacheRewound)
	}
	if n := s.noteCacheLineage(51500, 51159, "", at()).tokens; n != 0 {
		t.Errorf("healthy first call of turn 3: got %d, want 0", n)
	}
}

func TestKeepWindowStartDegenerate(t *testing.T) {
	if got := (&Session{}).keepWindowStart(10_000); got != 0 {
		t.Errorf("empty session: keepWindowStart = %d, want 0", got)
	}
}

func TestKeepWindowStart(t *testing.T) {
	dir := t.TempDir()

	// Keeping from K costs 25k - PromptTokens[K]: under 10k only step 3 (25k-15k) fits.
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

	s3, _ := newSession(dir)
	s3.AddUser("p")
	s3.markTurnStart()
	s3.AddAssistant("a1")
	s3.AddAssistant("a2")
	if got := s3.keepWindowStart(10_000); got != s3.lastAssistantIndex() {
		t.Errorf("no usage: keepWindowStart=%d, want lastAssistantIndex=%d", got, s3.lastAssistantIndex())
	}
}

func TestCacheLineageNamesTheRenderChange(t *testing.T) {
	at := lineageClock()
	const (
		think = `{"chat_template_kwargs":{"preserve_thinking":true}}`
		exec  = `{"chat_template_kwargs":{"enable_thinking":false}}`
	)
	s := &Session{}
	s.startTurn(time.Now())
	s.noteCacheLineage(71997, 71000, think, at())

	rw := s.noteCacheLineage(71997, 0, exec, at())
	if rw.tokens != 71997 || !rw.renderChanged {
		t.Errorf("flip to %s: got n=%d changed=%v, want 71997 and true", exec, rw.tokens, rw.renderChanged)
	}
	if rw.prevRender != think {
		t.Errorf("previous rendering = %q, want %q", rw.prevRender, think)
	}

	if rw := s.noteCacheLineage(99614, 71993, exec, at()); rw.tokens != 0 || rw.renderChanged {
		t.Errorf("second call under the new rendering: got n=%d changed=%v, want 0 and false", rw.tokens, rw.renderChanged)
	}

	if rw := s.noteCacheLineage(99614, 72029, think, at()); rw.tokens != 27585 || !rw.renderChanged {
		t.Errorf("flip back: got n=%d changed=%v, want 27585 and true", rw.tokens, rw.renderChanged)
	}
	if r := s.turnStats(); r.cacheRewinds != 2 || r.cacheRewindsRender != 2 {
		t.Errorf("rewinds=%d of which render changes=%d, want 2 and 2", r.cacheRewinds, r.cacheRewindsRender)
	}

	s2 := &Session{}
	s2.startTurn(time.Now())
	s2.noteCacheLineage(51594, 51590, think, at())
	if rw := s2.noteCacheLineage(51163, 21128, think, at()); rw.tokens == 0 || rw.renderChanged {
		t.Errorf("rewind with a stable rendering: got n=%d changed=%v, want a rewind and false", rw.tokens, rw.renderChanged)
	}
	if r := s2.turnStats(); r.cacheRewinds != 1 || r.cacheRewindsRender != 0 {
		t.Errorf("rewinds=%d of which render changes=%d, want 1 and 0", r.cacheRewinds, r.cacheRewindsRender)
	}

	s2.resetCacheLineage()
	if rw := s2.noteCacheLineage(9000, 0, exec, at()); rw.tokens != 0 || rw.renderChanged || rw.prevRender != "" {
		t.Errorf("after compaction: got n=%d prev=%q changed=%v, want 0, \"\" and false", rw.tokens, rw.prevRender, rw.renderChanged)
	}
}

// The idle gap tells a server-side eviction from a mid-prompt rewrite.
func TestCacheLineageTimesTheGap(t *testing.T) {
	base := time.Date(2026, 8, 21, 22, 0, 0, 0, time.UTC)
	s := &Session{}
	s.startTurn(base)

	// No previous call: reporting uptime would read as a stall.
	if rw := s.noteCacheLineage(51594, 51590, "", base); rw.idle != 0 {
		t.Errorf("first call: idle=%v, want 0", rw.idle)
	}
	if rw := s.noteCacheLineage(51700, 51590, "", base.Add(3*time.Second)); rw.idle != 3*time.Second {
		t.Errorf("healthy call: idle=%v, want 3s", rw.idle)
	}
	// The gap must reach the call that reports the rewind.
	rw := s.noteCacheLineage(51700, 0, "", base.Add(2*time.Hour+6*time.Minute))
	if rw.tokens == 0 {
		t.Fatal("the rewind itself went unreported")
	}
	if rw.idle < idleEvictionSuspect {
		t.Errorf("idle=%v, want at least %v so the log can name it an eviction", rw.idle, idleEvictionSuspect)
	}
	s.resetCacheLineage()
	if rw := s.noteCacheLineage(9000, 0, "", base.Add(3*time.Hour)); rw.idle != 0 {
		t.Errorf("after compaction: idle=%v, want 0", rw.idle)
	}
}

// Discarded decode stays inside completion and is also named on its own.
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

	s.startTurn(time.Now())
	if r := s.turnStats(); r.wastedCompletion != 0 {
		t.Errorf("after startTurn: wastedCompletion=%d, want 0", r.wastedCompletion)
	}
}

// Anything that is not a sampler, unknown fields included, counts as a rendering change.
func TestRenderKeyIgnoresSamplers(t *testing.T) {
	plan := renderKey(map[string]any{"temperature": 1.0, "top_p": 0.95, "max_tokens": 8000})
	exec := renderKey(map[string]any{"temperature": 0.6, "top_p": 0.8, "max_tokens": 4000})
	if plan != "" || exec != "" {
		t.Errorf("samplers entered the key: thinking=%q execute=%q", plan, exec)
	}

	// Map order is random; a flapping key would report a rewind every other call.
	a := renderKey(map[string]any{"temperature": 1.0, "chat_template_kwargs": map[string]any{"preserve_thinking": true, "enable_thinking": true}})
	b := renderKey(map[string]any{"temperature": 0.6, "chat_template_kwargs": map[string]any{"enable_thinking": true, "preserve_thinking": true}})
	if a != b || a == "" {
		t.Errorf("kwargs key is not canonical: %q vs %q", a, b)
	}

	if k := renderKey(map[string]any{"reasoning_effort": "low"}); k == "" {
		t.Error("reasoning_effort was dropped from the key")
	}
}

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

// Ids are second-granular; a same-second session must not reuse the id on disk.
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

	s.AppendToolUse(ToolUse{Name: "write_file", Input: "{}", Output: "ok"})
	if got := len(s.Messages); got != 2 {
		t.Fatalf("Messages len after second AppendToolUse: got %d, want 2", got)
	}
	if got := len(s.Messages[1].ToolUses); got != 2 {
		t.Fatalf("ToolUses len: got %d, want 2", got)
	}
}

// Run with -race.
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

	// An exact count catches an append lost to a concurrent slice grow.
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

func TestExternalChangeDrift(t *testing.T) {
	s := &Session{}
	const path = "/w/proj/src/app.js"

	s.checkExternalChange(path, "whatever")
	if note := s.takeDriftNote(path); note != "" {
		t.Errorf("untracked file produced a note: %q", note)
	}

	s.recordWrite(path, "let a = 1;\n")
	s.checkExternalChange(path, "let a = 1;\n")
	if note := s.takeDriftNote(path); note != "" {
		t.Errorf("unchanged file produced a note: %q", note)
	}

	s.checkExternalChange(path, "let  a  =  1;\n")
	note := s.takeDriftNote(path)
	if note == "" {
		t.Fatal("a file rewritten behind us produced no note")
	}
	if again := s.takeDriftNote(path); again != "" {
		t.Errorf("note delivered twice: %q", again)
	}
	s.checkExternalChange(path, "let  a  =  1;\n")
	if n := s.takeDriftNote(path); n != "" {
		t.Errorf("already-reported drift reported again: %q", n)
	}
	s.checkExternalChange(path, "let a = 2;\n")
	if n := s.takeDriftNote(path); n == "" {
		t.Error("a second, distinct external change produced no note")
	}

	s.recordWrite(path, "let a = 3;\n")
	s.checkExternalChange(path, "let a = 4;\n")
	s.recordWrite(path, "let a = 5;\n")
	if n := s.takeDriftNote(path); n != "" {
		t.Errorf("pending note survived our own write: %q", n)
	}
}

// The repaired text must marshal to the same JSON as the broken original: same wire bytes.
func TestLoadSessionRepairsInvalidUTF8(t *testing.T) {
	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatal(err)
	}
	broken := "the session clock of \xe2\x86\n[... 390 bytes truncated ...]"
	s.Summary = broken
	s.AddUser("go on")
	if err := s.Save(); err != nil {
		t.Fatal(err)
	}
	got, err := loadSession(dir, s.ID)
	if err != nil {
		t.Fatalf("loadSession: %v", err)
	}
	if !utf8.ValidString(got.Summary) {
		t.Errorf("Summary still invalid: %q", got.Summary)
	}
	was, _ := json.Marshal(broken)
	now, _ := json.Marshal(got.Summary)
	if string(was) != string(now) {
		t.Errorf("wire bytes changed: before %s, after %s", was, now)
	}
	if len(got.Messages) != 1 || got.Messages[0].Content != "go on" {
		t.Errorf("messages not restored: %+v", got.Messages)
	}
}
