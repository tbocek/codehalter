package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// TestCompressHistoryRecordsSummary is the headline history test: once the
// server-reported prompt_tokens crosses the trigger, a boundary compaction
// (midTurn=false) should rotate the session — freeze the pre-rotation state to
// a "session_archive_*" file, fold the WHOLE shadow buffer into Summary, keep
// NOTHING verbatim (start fresh), and persist everything. The mock LLM has zero
// responses queued: folding pre-computed notes is fully local and any LLM call
// would fail the test.
func TestCompressHistoryRecordsSummary(t *testing.T) {
	mock := newMockLLM(t)
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}

	// Build 10 user+assistant pairs; a boundary compaction folds them all into
	// the summary and keeps nothing verbatim.
	filler := strings.Repeat("lorem ipsum ", 100)
	for i := 0; i < 10; i++ {
		s.AddUser(fmt.Sprintf("user msg %d %s", i, filler))
		s.AddAssistant(fmt.Sprintf("asst msg %d %s", i, filler))
	}
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}
	originalMsgCount := len(s.Messages)

	// Seed shadow with three completed-turn notes; all fold (no anchor held back).
	s.appendShadow("Goal: ship feature\nProgress: scaffolded module")
	s.appendShadow("Goal: ship feature\nProgress: wired up handler")
	s.appendShadow("Goal: ship feature\nProgress: shipped it")

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}},
		},
		mainSlotTokens: 90_000,
	}

	s.turnStartIdx = len(s.Messages) // all turns completed → foldHistory(len) folds them all via shadow
	a.foldHistory(context.Background(), s, len(s.Messages))

	for _, want := range []string{"scaffolded module", "wired up handler", "shipped it"} {
		if !strings.Contains(s.Summary, want) {
			t.Errorf("summary missing folded shadow entry %q; got %q", want, s.Summary)
		}
	}
	if peek := s.peekShadow(); peek != "" {
		t.Errorf("shadow buffer should be empty after a fold-all compaction; got %q", peek)
	}
	if len(s.Messages) >= originalMsgCount {
		t.Errorf("Messages not trimmed: before %d, after %d", originalMsgCount, len(s.Messages))
	}
	if len(s.Messages) != 0 {
		t.Errorf("boundary compaction should keep nothing verbatim; got %d messages", len(s.Messages))
	}

	// An archive file should exist holding the pre-rotation full state, and
	// the live session must keep its original ID + path.
	archives, err := filepath.Glob(filepath.Join(dir, sessionDir, "session_archive_*.toml"))
	if err != nil {
		t.Fatalf("glob archives: %v", err)
	}
	if len(archives) != 1 {
		t.Fatalf("expected 1 archive file, got %d: %v", len(archives), archives)
	}
	archiveID := strings.TrimSuffix(strings.TrimPrefix(filepath.Base(archives[0]), "session_"), ".toml")
	archived, err := loadSession(dir, archiveID)
	if err != nil {
		t.Fatalf("loadSession archive: %v", err)
	}
	if len(archived.Messages) != originalMsgCount {
		t.Errorf("archive Messages: got %d, want %d", len(archived.Messages), originalMsgCount)
	}

	// Persistence — the drained shadow must survive a reload.
	loaded, err := loadSession(dir, s.ID)
	if err != nil {
		t.Fatalf("loadSession: %v", err)
	}
	if !strings.Contains(loaded.Summary, "wired up handler") {
		t.Errorf("persisted summary missing drained shadow entries; got %q", loaded.Summary)
	}

	if mock.callCount() != 0 {
		t.Errorf("LLM calls: got %d, want 0 (shadow fast path is fully local)", mock.callCount())
	}
}

// TestPrefixStableAcrossTurns is the cache-correctness contract: a second
// Prompt() turn must reproduce the previous turn's wire bytes byte-for-byte
// for every message that's already on record. llama.cpp / vLLM / etc. only
// reuse their KV cache when the leading tokens of the new request match the
// leading tokens of the old one, so any drift in the prefix bytes silently
// reprocesses the entire history each turn.
//
// The append-only transcript model makes this trivial: each phase pushes a
// new user/assistant pair onto sess.Messages and never mutates earlier
// entries, so buildLLMContext replays the same bytes. The test exercises the
// load-bearing case — the first turn populates sess.SystemPrompt (skills +
// project context), and that leading message must remain identical on later
// turns so the LLM's prefix cache keeps hitting.
func TestPrefixStableAcrossTurns(t *testing.T) {
	dir := t.TempDir()

	// Seed a SKILL file so loadSkills returns a non-empty system prompt —
	// otherwise the bug (sysPrompt set turn 1, dropped turn 2) is invisible
	// because sysPrompt is empty.
	cfgDir := filepath.Join(dir, ".codehalter")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(cfgDir, "SKILL-go.md"), []byte("# Go skill\nsome conventions\n"), 0o644); err != nil {
		t.Fatalf("write SKILL: %v", err)
	}

	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	a := &agent{sessions: map[string]*Session{s.ID: s}}

	// Sanity: systemPrompt must be non-empty so the bug we're guarding
	// against (sysPrompt set turn 1, dropped turn 2) is observable.
	sysPrompt, _ := a.systemPrompt(s.ID)
	if sysPrompt == "" {
		t.Fatal("expected non-empty systemPrompt — SKILL seed didn't take effect")
	}

	// --- Turn 1: first prompt of the session ---
	// Prompt() sets sess.SystemPrompt (emitted by buildLLMContext as the
	// leading user message); subsequent phases (PLAN/EXECUTE/VERIFY/
	// DOCUMENT.md) get their own user turns appended to Messages.
	s.SystemPrompt = sysPrompt
	s.AddUser("first prompt")
	msgs1 := a.buildLLMContext(s)

	// Assistant replies (planner JSON, executor text, etc. — collapsed to
	// one assistant message here since the test only cares about the
	// user/assistant alternation that lands in history).
	s.UpsertLastAssistant("done with turn 1")

	// --- Turn 2: a follow-up prompt ---
	// Subsequent user turns store just the raw text — sysPrompt is already
	// in sess.SystemPrompt from turn 1.
	s.AddUser("second prompt")
	msgs2 := a.buildLLMContext(s)

	if len(msgs2) <= len(msgs1) {
		t.Fatalf("turn 2 should extend turn 1's history; got len1=%d len2=%d",
			len(msgs1), len(msgs2))
	}
	// Every message turn 1 sent must reappear byte-identically as the prefix
	// of turn 2's wire — this is exactly what the prefix cache keys on.
	for i := range msgs1 {
		b1, _ := json.Marshal(msgs1[i])
		b2, _ := json.Marshal(msgs2[i])
		if !bytes.Equal(b1, b2) {
			t.Errorf("prefix message %d drifted between turns:\n  turn 1: %s\n  turn 2: %s",
				i, b1, b2)
		}
	}
}

// TestCompressHistoryNoopWhenBelowBudget verifies that sessions with a
// prompt_tokens reading below the trigger don't call the LLM at all — no
// summary.
func TestFoldHistoryNoopWhenNothingToFold(t *testing.T) {
	mock := newMockLLM(t) // zero responses queued → any call fails the test.
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.AddUser("short")
	s.AddAssistant("also short")

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}},
		},
		mainSlotTokens: 90_000,
	}

	a.foldHistory(context.Background(), s, 0) // keepFrom=0 → nothing to fold

	if s.Summary != "" {
		t.Errorf("expected empty summary, got %q", s.Summary)
	}
	if len(s.Messages) != 2 {
		t.Errorf("expected messages untouched, got %d", len(s.Messages))
	}
	if mock.callCount() != 0 {
		t.Errorf("expected no LLM calls, got %d", mock.callCount())
	}
}

// TestCompressHistoryShadowFastPath verifies the background-summariser fast
// path: when the shadow buffer already has structured notes (populated during
// the turns), compaction folds the whole buffer into Summary with no LLM call.
func TestCompressHistoryShadowFastPath(t *testing.T) {
	mock := newMockLLM(t) // zero responses queued → any call fails the test.
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}

	filler := strings.Repeat("lorem ipsum ", 100)
	for i := 0; i < 10; i++ {
		s.AddUser(fmt.Sprintf("user msg %d %s", i, filler))
		s.AddAssistant(fmt.Sprintf("asst msg %d %s", i, filler))
	}
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	s.appendShadow("Goal: do thing\nProgress: did thing")
	s.appendShadow("Goal: do thing\nProgress: refined thing")
	s.appendShadow("Goal: do thing\nProgress: finished thing")

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}},
		},
		mainSlotTokens: 90_000,
	}

	s.turnStartIdx = len(s.Messages)
	a.foldHistory(context.Background(), s, len(s.Messages))

	// Every note folds into Summary; no anchor is held back.
	for _, want := range []string{"did thing", "refined thing", "finished thing"} {
		if !strings.Contains(s.Summary, want) {
			t.Errorf("expected Summary to contain folded shadow chunk %q, got %q", want, s.Summary)
		}
	}
	if mock.callCount() != 0 {
		t.Errorf("LLM calls: got %d, want 0 (shadow fast path is fully local)", mock.callCount())
	}
	// The buffer is fully drained after a fold-all compaction.
	if peek := s.peekShadow(); peek != "" {
		t.Errorf("shadow buffer should be empty after compaction; got %q", peek)
	}
}

// contentString asserts that an llmMessage carries a string payload and
// returns it; it fails the test if the content was something else.
func contentString(t *testing.T, m llmMessage) string {
	t.Helper()
	s, ok := m.Content.(string)
	if !ok {
		t.Fatalf("expected string content, got %T", m.Content)
	}
	return s
}

// TestBuildLLMHistoryShape verifies the header injection when a summary
// exists (one leading user message) and that stored messages follow in
// order. The no-summary case verifies the header is omitted.
func TestBuildLLMHistoryShape(t *testing.T) {
	a := &agent{}
	s := &Session{
		Summary: "earlier summary",
		Messages: []Message{
			{Role: "user", Content: "q1"},
			{Role: "assistant", Content: "a1"},
			{Role: "user", Content: "q2"},
		},
	}

	msgs := a.buildLLMContext(s)

	if len(msgs) != 4 {
		t.Fatalf("got %d messages, want 4: %+v", len(msgs), msgs)
	}
	if msgs[0].Role != "user" {
		t.Errorf("header role: got %q, want user", msgs[0].Role)
	}
	if !strings.Contains(contentString(t, msgs[0]), "earlier summary") {
		t.Errorf("intro missing summary content: %q", contentString(t, msgs[0]))
	}
	if got := contentString(t, msgs[1]); got != "q1" {
		t.Errorf("msgs[1]: got %q, want q1", got)
	}
	if got := contentString(t, msgs[2]); got != "a1" {
		t.Errorf("msgs[2]: got %q, want a1", got)
	}
	if got := contentString(t, msgs[3]); got != "q2" {
		t.Errorf("msgs[3]: got %q, want q2", got)
	}

	// No summary → no header; stored messages pass through unchanged.
	s2 := &Session{Messages: []Message{
		{Role: "user", Content: "q1"},
		{Role: "assistant", Content: "a1"},
	}}
	msgs2 := a.buildLLMContext(s2)
	if len(msgs2) != 2 {
		t.Fatalf("no-summary case: got %d, want 2", len(msgs2))
	}
	if got := contentString(t, msgs2[0]); got != "q1" {
		t.Errorf("no-summary case msgs2[0]: got %q, want q1", got)
	}
	if got := contentString(t, msgs2[1]); got != "a1" {
		t.Errorf("no-summary case msgs2[1]: got %q, want a1", got)
	}
}

// TestBuildLLMHistoryToolUseProtocolShape verifies that a stored assistant
// message with ToolUses is rebuilt in the OpenAI protocol shape — assistant
// with ToolCalls field, followed by one tool-role message per call carrying
// the (truncated) output and a ToolCallID pointer back. tu.ID is reused as
// tool_call_id. Also covers the empty-assistant-content path (model emitted
// only tool calls) and the long-output truncation path.
func TestBuildLLMHistoryToolUseProtocolShape(t *testing.T) {
	long := strings.Repeat("X", truncateThreshold+500)

	a := &agent{}
	s := &Session{Messages: []Message{
		{Role: "user", Content: "do it"},
		{Role: "assistant", Content: "running", ToolUses: []ToolUse{
			{ID: "tu_1", Name: "read_file", Input: `{"path":"x"}`, Output: "file content"},
			{ID: "tu_2", Name: "search", Input: `{"q":"foo"}`, Output: long},
		}},
		{Role: "user", Content: "thanks"},
		{Role: "assistant", Content: "", ToolUses: []ToolUse{
			{ID: "tu_3", Name: "respond", Input: `{}`, Output: "done"},
		}},
	}}

	msgs := a.buildLLMContext(s)
	// user + (assistant + 2 tool) + user + (assistant + 1 tool) = 7
	if len(msgs) != 7 {
		t.Fatalf("got %d messages, want 7: %+v", len(msgs), msgs)
	}

	if msgs[0].Role != "user" || contentString(t, msgs[0]) != "do it" {
		t.Errorf("msgs[0] wrong: %+v", msgs[0])
	}

	asst := msgs[1]
	if asst.Role != "assistant" || contentString(t, asst) != "running" {
		t.Errorf("assistant Role/Content wrong: %+v", asst)
	}
	if len(asst.ToolCalls) != 2 {
		t.Fatalf("assistant ToolCalls len: got %d, want 2", len(asst.ToolCalls))
	}
	if asst.ToolCalls[0].ID != "tu_1" || asst.ToolCalls[0].Type != "function" ||
		asst.ToolCalls[0].Function.Name != "read_file" || asst.ToolCalls[0].Function.Arguments != `{"path":"x"}` {
		t.Errorf("ToolCalls[0] wrong: %+v", asst.ToolCalls[0])
	}
	if asst.ToolCalls[1].ID != "tu_2" || asst.ToolCalls[1].Function.Name != "search" {
		t.Errorf("ToolCalls[1] wrong: %+v", asst.ToolCalls[1])
	}

	if msgs[2].Role != "tool" || msgs[2].ToolCallID != "tu_1" || contentString(t, msgs[2]) != "file content" {
		t.Errorf("tool message [0] wrong: %+v", msgs[2])
	}
	// Long output runs through truncateForLLM — the wire copy is shorter than
	// the stored Output and carries the "to see more" hint.
	tool2Content := contentString(t, msgs[3])
	if msgs[3].Role != "tool" || msgs[3].ToolCallID != "tu_2" {
		t.Errorf("tool message [1] Role/ID wrong: %+v", msgs[3])
	}
	if len(tool2Content) >= len(long) {
		t.Errorf("long tool output not truncated: wire len %d >= stored len %d", len(tool2Content), len(long))
	}
	if !strings.Contains(tool2Content, "To see more:") {
		t.Errorf("truncated tool message missing the hint: %q", tool2Content)
	}

	if msgs[4].Role != "user" || contentString(t, msgs[4]) != "thanks" {
		t.Errorf("msgs[4] wrong: %+v", msgs[4])
	}
	// Empty-content assistant turn (model emitted only tool calls) still
	// gets a properly-shaped assistant message.
	if msgs[5].Role != "assistant" || contentString(t, msgs[5]) != "" || len(msgs[5].ToolCalls) != 1 {
		t.Errorf("empty-content assistant wrong: %+v", msgs[5])
	}
	if msgs[6].Role != "tool" || msgs[6].ToolCallID != "tu_3" {
		t.Errorf("tool message [2] wrong: %+v", msgs[6])
	}
}

// TestBuildLLMHistoryImageHandling covers the imagesSupported branch and the
// cache-consistency rule: every stored image gets its bytes inlined every turn
// — there is no trailing-vs-older split. After compaction the image lives in
// Summary as a reference and view_image fetches it on demand.
func TestBuildLLMHistoryImageHandling(t *testing.T) {
	dir := t.TempDir()
	bytes1 := []byte("pngbytes-1")
	bytes2 := []byte("pngbytes-2-different")
	id1 := "img_test_a"
	id2 := "img_test_b"
	if err := writeImageFile(dir, id1, "image/png", bytes1); err != nil {
		t.Fatalf("writeImageFile id1: %v", err)
	}
	if err := writeImageFile(dir, id2, "image/png", bytes2); err != nil {
		t.Fatalf("writeImageFile id2: %v", err)
	}
	img1 := ImageData{ID: id1, MimeType: "image/png"}
	img2 := ImageData{ID: id2, MimeType: "image/png"}

	// Images NOT supported → text fallback with per-image placeholders.
	a := &agent{imagesSupported: false}
	s := &Session{Cwd: dir, Messages: []Message{{Role: "user", Content: "look at this", Images: []ImageData{img1, img2}}}}
	out := a.buildLLMContext(s)
	if len(out) != 1 {
		t.Fatalf("expected 1 message, got %d", len(out))
	}
	got := contentString(t, out[0])
	if !strings.Contains(got, "[Image "+id1) || !strings.Contains(got, "[Image "+id2) {
		t.Errorf("expected per-image placeholders in %q", got)
	}
	if !strings.Contains(got, "view_image id="+id1) || !strings.Contains(got, "view_image id="+id2) {
		t.Errorf("expected view_image hints in %q", got)
	}

	// Images supported → []any with one text block + N image_url blocks
	// containing the actual data: URLs read from disk.
	a.imagesSupported = true
	out = a.buildLLMContext(s)
	parts, ok := out[0].Content.([]any)
	if !ok {
		t.Fatalf("expected []any content, got different type")
	}
	if len(parts) != 3 {
		t.Fatalf("expected 3 parts (text + 2 images), got %d", len(parts))
	}
	text, _ := parts[0].(map[string]any)
	if text["type"] != "text" || text["text"] != "look at this" {
		t.Errorf("parts[0] wrong: %+v", text)
	}
	for i, part := range parts[1:] {
		block, _ := part.(map[string]any)
		if block["type"] != "image_url" {
			t.Errorf("parts[%d] type: got %v, want image_url", i+1, block["type"])
		}
		url, _ := block["image_url"].(map[string]string)
		if !strings.HasPrefix(url["url"], "data:image/png;base64,") {
			t.Errorf("parts[%d] url prefix wrong: %q", i+1, url["url"])
		}
	}

	// Cache-consistency: two messages with images both inline bytes every
	// turn. No trailing/older split — the older image is NOT degraded to a
	// text placeholder.
	s4 := &Session{Cwd: dir, Messages: []Message{
		{Role: "user", Content: "earlier", Images: []ImageData{img1}},
		{Role: "user", Content: "now look at this one", Images: []ImageData{img2}},
	}}
	out = a.buildLLMContext(s4)
	if len(out) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(out))
	}
	olderParts, ok := out[0].Content.([]any)
	if !ok || len(olderParts) != 2 {
		t.Fatalf("older: expected []any of len 2 (text + image_url), got %+v", out[0].Content)
	}
	olderImg, _ := olderParts[1].(map[string]any)
	if olderImg["type"] != "image_url" {
		t.Errorf("older parts[1] type: got %v, want image_url", olderImg["type"])
	}
	trailingParts, ok := out[1].Content.([]any)
	if !ok || len(trailingParts) != 2 {
		t.Fatalf("trailing: expected []any of len 2, got %+v", out[1].Content)
	}

	// Wire bytes for the same stored message must be identical turn-over-turn.
	first := a.buildLLMContext(s4)
	second := a.buildLLMContext(s4)
	firstJSON, err := json.Marshal(first)
	if err != nil {
		t.Fatalf("marshal first: %v", err)
	}
	secondJSON, err := json.Marshal(second)
	if err != nil {
		t.Fatalf("marshal second: %v", err)
	}
	if string(firstJSON) != string(secondJSON) {
		t.Errorf("cache consistency: wire bytes differ between consecutive rebuilds\nfirst:  %s\nsecond: %s", firstJSON, secondJSON)
	}

	// Missing file on disk → text fallback for that image only, no panic and
	// the rest of the request still goes through.
	imgMissing := ImageData{ID: "img_deadbeef00000000", MimeType: "image/png"}
	sMissing := &Session{Cwd: dir, Messages: []Message{{Role: "user", Content: "missing", Images: []ImageData{imgMissing}}}}
	outMissing := a.buildLLMContext(sMissing)
	missingParts, ok := outMissing[0].Content.([]any)
	if !ok || len(missingParts) != 2 {
		t.Fatalf("missing: expected []any of len 2 (text + fallback), got %+v", outMissing[0].Content)
	}
	fallback, _ := missingParts[1].(map[string]any)
	if fallback["type"] != "text" {
		t.Errorf("missing image fallback type: got %v, want text", fallback["type"])
	}
	if fallbackText, _ := fallback["text"].(string); !strings.Contains(fallbackText, "img_deadbeef00000000") {
		t.Errorf("missing image fallback missing id reference: %q", fallbackText)
	}

	// Message with no images → plain string, untouched.
	s2 := &Session{Cwd: dir, Messages: []Message{{Role: "user", Content: "no imgs"}}}
	if got := contentString(t, a.buildLLMContext(s2)[0]); got != "no imgs" {
		t.Errorf("plain: got %q, want 'no imgs'", got)
	}

	// Combined: tool uses + images on a message. The image is inlined as an
	// image_url part; the tool use lives in ToolCalls on the assistant message
	// and produces a follow-up tool-role message.
	combined := Message{
		Role:     "assistant",
		Content:  "done",
		Images:   []ImageData{img1},
		ToolUses: []ToolUse{{ID: "tu_77", Name: "read_file", Input: `{"path":"x"}`, Output: "ok"}},
	}
	s3 := &Session{Cwd: dir, Messages: []Message{combined}}
	outCombined := a.buildLLMContext(s3)
	if len(outCombined) != 2 {
		t.Fatalf("combined: expected 2 messages (assistant + tool), got %d: %+v", len(outCombined), outCombined)
	}
	parts, ok = outCombined[0].Content.([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("combined assistant Content: expected []any of len 2, got %+v", outCombined[0].Content)
	}
	text, _ = parts[0].(map[string]any)
	textStr, _ := text["text"].(string)
	if textStr != "done" {
		t.Errorf("combined text block: got %q, want %q", textStr, "done")
	}
	if len(outCombined[0].ToolCalls) != 1 || outCombined[0].ToolCalls[0].ID != "tu_77" || outCombined[0].ToolCalls[0].Function.Name != "read_file" {
		t.Errorf("combined ToolCalls wrong: %+v", outCombined[0].ToolCalls)
	}
	if outCombined[1].Role != "tool" || outCombined[1].ToolCallID != "tu_77" || outCombined[1].Content.(string) != "ok" {
		t.Errorf("combined tool message wrong: %+v", outCombined[1])
	}
}

// TestBackgroundSummariseAppendsImageRefsThroughCompaction is the end-to-end
// post-compaction view_image story: a user turn with images + an assistant
// turn → backgroundSummarise produces a shadow chunk that includes the
// `Attached images:` ref block → foldHistory folds it into Session.Summary
// so the handle survives even after the original message rotates out.
func TestBackgroundSummariseAppendsImageRefsThroughCompaction(t *testing.T) {
	mock := newMockLLM(t, sseText("Goal: inspect screenshot\nProgress: looked at it"))
	defer mock.Close()

	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, ".codehalter"), 0o755); err != nil {
		t.Fatalf("mkdir .codehalter: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".codehalter", "SUMMARISE.md"), []byte("SUMMARISE PROMPT\n"), 0o644); err != nil {
		t.Fatalf("write SUMMARISE.md: %v", err)
	}

	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	bytes := []byte("PNG screenshot bytes")
	imgID := "img_test_compaction"
	if err := writeImageFile(dir, imgID, "image/png", bytes); err != nil {
		t.Fatalf("writeImageFile: %v", err)
	}
	s.AddUserWithImages("look at this", []ImageData{{ID: imgID, MimeType: "image/png"}})
	s.AddAssistant("I see a screenshot.")

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}},
		},
		mainSlotTokens: 90_000,
	}

	a.backgroundSummarise(s)

	deadline := time.Now().Add(2 * time.Second)
	for {
		if peek := s.peekShadow(); strings.Contains(peek, "Attached images:") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("backgroundSummarise never appended image refs to shadow; got %q", s.peekShadow())
		}
		time.Sleep(10 * time.Millisecond)
	}

	peek := s.peekShadow()
	if !strings.Contains(peek, "Goal: inspect screenshot") {
		t.Errorf("shadow chunk missing summariser output: %q", peek)
	}
	if !strings.Contains(peek, "view_image id="+imgID) {
		t.Errorf("shadow chunk missing view_image handle %q: %q", imgID, peek)
	}

	// Drive a boundary compaction and confirm the image reference folds into
	// Summary (a second note alongside it, both fold — no anchor held back).
	s.appendShadow("Goal: follow-up\nProgress: follow-up turn")
	s.turnStartIdx = len(s.Messages)
	a.foldHistory(context.Background(), s, len(s.Messages))

	if !strings.Contains(s.Summary, "view_image id="+imgID) {
		t.Errorf("Summary lost the image handle after compaction; got %q", s.Summary)
	}
}

// TestCompressHistoryShadowPreservesPriorSummary verifies that when the
// shadow fast path runs and a previous Summary is already in place, the
// previous Summary is kept and the shadow is appended after it.
func TestCompressHistoryShadowPreservesPriorSummary(t *testing.T) {
	mock := newMockLLM(t) // shadow fast path is fully local — no LLM calls expected.
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.Summary = "PRIOR SUMMARY FROM AN EARLIER COMPACTION"

	filler := strings.Repeat("lorem ipsum ", 100)
	for i := 0; i < 10; i++ {
		s.AddUser(fmt.Sprintf("user %d %s", i, filler))
		s.AddAssistant(fmt.Sprintf("asst %d %s", i, filler))
	}

	s.appendShadow("Goal: x\nProgress: y")
	s.appendShadow("Goal: x\nProgress: anchor")

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}},
		},
		mainSlotTokens: 90_000,
	}

	s.turnStartIdx = len(s.Messages)
	a.foldHistory(context.Background(), s, len(s.Messages))

	if !strings.Contains(s.Summary, "PRIOR SUMMARY") {
		t.Errorf("prior Summary dropped during shadow fast path; got %q", s.Summary)
	}
	if !strings.Contains(s.Summary, "Goal: x") {
		t.Errorf("shadow chunk missing from new Summary; got %q", s.Summary)
	}
}

// TestShadowPersistsAcrossReload pins the persistence fix: turn notes live in
// the Shadow field and must survive a Save/loadSession round-trip, so the notes
// accumulated across turns are not lost when the process restarts.
func TestShadowPersistsAcrossReload(t *testing.T) {
	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.AddUser("hi")
	s.appendShadow("Goal: a\nProgress: one")
	s.appendShadow("Goal: a\nProgress: two")
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := loadSession(dir, s.ID)
	if err != nil {
		t.Fatalf("loadSession: %v", err)
	}
	if len(loaded.Shadow) != 2 {
		t.Fatalf("Shadow not persisted: got %d entries, want 2 (%q)", len(loaded.Shadow), loaded.Shadow)
	}
	if loaded.Shadow[0] != "Goal: a\nProgress: one" || loaded.Shadow[1] != "Goal: a\nProgress: two" {
		t.Errorf("Shadow entries corrupted on reload: %q", loaded.Shadow)
	}
}

// TestCompressHistoryMidTurnKeepsInFlightTurn covers the mid-turn policy: above
// the 90% trigger, compaction keeps the in-flight turn (Messages[turnStart:])
// verbatim and folds only the completed turns ahead of it into Summary.
func TestCompressHistoryMidTurnKeepsInFlightTurn(t *testing.T) {
	mock := newMockLLM(t) // folding is local; any LLM call fails the test.
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}

	filler := strings.Repeat("lorem ipsum ", 100)
	// Completed turn 1.
	s.AddUser("turn1 user " + filler)
	s.AddAssistant("turn1 asst " + filler)
	// In-flight turn 2 begins at the next user message.
	s.AddUser("turn2 user " + filler)
	s.markTurnStart()
	s.AddAssistant("turn2 asst step " + filler)

	// One note for the completed turn 1 (the in-flight turn has none yet).
	s.appendShadow("Goal: do\nProgress: finished turn 1")

	a := &agent{
		sessions:       map[string]*Session{s.ID: s},
		settings:       Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}},
		mainSlotTokens: 90_000,
	}

	if !a.foldHistory(context.Background(), s, s.turnStartIdx) {
		t.Fatalf("foldHistory(turnStartIdx) did not fold the completed turn")
	}
	if !strings.Contains(s.Summary, "finished turn 1") {
		t.Errorf("Summary missing the completed turn's note; got %q", s.Summary)
	}
	if len(s.Messages) != 2 {
		t.Fatalf("expected the 2 in-flight messages kept verbatim, got %d", len(s.Messages))
	}
	if !strings.Contains(s.Messages[0].Content, "turn2 user") {
		t.Errorf("kept window must start at the in-flight turn; got %q", s.Messages[0].Content)
	}
	if peek := s.peekShadow(); peek != "" {
		t.Errorf("shadow should be drained after compaction; got %q", peek)
	}
	if s.turnStartIdx != 0 {
		t.Errorf("turnStart not reset after rotation; got %d", s.turnStartIdx)
	}
	if mock.callCount() != 0 {
		t.Errorf("LLM calls: got %d, want 0", mock.callCount())
	}
}

// TestBackgroundSummariseRendersWholeTurn verifies the summariser is fed the
// ENTIRE turn — the user prompt plus every assistant step and its tool calls —
// not just the last assistant message, so one note covers the whole turn.
func TestBackgroundSummariseRendersWholeTurn(t *testing.T) {
	mock := newMockLLM(t, sseText("Goal: g\nProgress: did it"))
	defer mock.Close()

	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, ".codehalter"), 0o755); err != nil {
		t.Fatalf("mkdir .codehalter: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".codehalter", "SUMMARISE.md"), []byte("SUMMARISE PROMPT\n"), 0o644); err != nil {
		t.Fatalf("write SUMMARISE.md: %v", err)
	}

	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.AddUser("please do the thing")
	s.markTurnStart()
	s.AddAssistantWithTools("reading files", []ToolUse{{ID: "tu_1", Name: "read_file", Input: `{"path":"a.go"}`, Output: "package a"}})
	s.AddAssistant("done with the thing")

	a := &agent{
		sessions:       map[string]*Session{s.ID: s},
		settings:       Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}},
		mainSlotTokens: 90_000,
	}
	a.backgroundSummarise(s)

	deadline := time.Now().Add(2 * time.Second)
	for s.peekShadow() == "" {
		if time.Now().After(deadline) {
			t.Fatalf("backgroundSummarise never appended a note")
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Single [[llm]] entry → prefix-extension mode: the request replays the
	// conversation itself (reusing the foreground KV cache) instead of pasting
	// a transcript, with the SUMMARISE instruction as the FINAL message,
	// anchored to the turn's opening user message.
	req := mock.request(0)
	msgs, _ := req["messages"].([]any)
	if len(msgs) < 2 {
		t.Fatalf("summariser request should carry the conversation + instruction, got %d messages: %v", len(msgs), req)
	}
	wire, _ := json.Marshal(msgs)
	for _, want := range []string{"please do the thing", "read_file", "done with the thing"} {
		if !strings.Contains(string(wire), want) {
			t.Errorf("summariser request missing %q; got:\n%s", want, wire)
		}
	}
	// Reasoning off: the closed think block follows the instruction, and the
	// server continues it. A note written after 20 KB of reasoning missed the
	// deadline on a 27B.
	prefill, _ := msgs[len(msgs)-1].(map[string]any)
	if prefill["role"] != "assistant" || prefill["content"] != noThinkPrefillContent || req["continue_final_message"] != true {
		t.Errorf("summariser should run with reasoning off (closed think block, continued), got last=%v continue=%v", prefill, req["continue_final_message"])
	}
	last, _ := msgs[len(msgs)-2].(map[string]any)
	instr, _ := last["content"].(string)
	if last["role"] != "user" || !strings.Contains(instr, "SUMMARISE PROMPT") {
		t.Errorf("the SUMMARISE instruction should precede the prefill, got role=%v content:\n%s", last["role"], instr)
	}
	if !strings.Contains(instr, "please do the thing") {
		t.Errorf("instruction should anchor the turn's opening user message, got:\n%s", instr)
	}
	// Prefix-extension MUST carry the foreground's tools array: the chat
	// template renders tools into the head of the prompt, so a tools-less
	// request diverges from the foreground prefix at the first token (cold
	// re-eval + evicts the single slot's KV). tool_choice=none keeps the
	// answer a note instead of a tool call.
	if tools, _ := req["tools"].([]any); len(tools) == 0 {
		t.Errorf("prefix-extension summarise request must carry the foreground tools array, got none")
	}
	if tc, _ := req["tool_choice"].(string); tc != "none" {
		t.Errorf("prefix-extension summarise request should set tool_choice=none, got %v", req["tool_choice"])
	}
}

// TestBuildContextUsesModelCallID pins the cache fix: a rebuilt context must use
// the MODEL's tool_call id (CallID) on the wire — both the assistant tool call
// and its tool result — so replaying from history is byte-identical to what was
// sent live. The internal useID is only the fallback for a model that sends none.
func TestBuildContextUsesModelCallID(t *testing.T) {
	a := &agent{}
	s := &Session{}
	s.AddUser("go")
	s.AddAssistant("")
	s.AppendToolUse(ToolUse{ID: "tu_1", CallID: "call_abc", Name: "read_file", Input: "{}", Output: "data"})

	var gotCall, gotResult string
	for _, m := range a.buildLLMContext(s) {
		if m.Role == "assistant" && len(m.ToolCalls) == 1 {
			gotCall = m.ToolCalls[0].ID
		}
		if m.Role == "tool" {
			gotResult = m.ToolCallID
		}
	}
	if gotCall != "call_abc" || gotResult != "call_abc" {
		t.Errorf("wire ids: tool_call=%q tool_call_id=%q, want both %q (model's id)", gotCall, gotResult, "call_abc")
	}

	// Fallback for an old session with no CallID → the useID.
	if got := wireCallID(ToolUse{ID: "tu_9"}); got != "tu_9" {
		t.Errorf("fallback: got %q, want tu_9", got)
	}
}

// peekShadow joins every accumulated turn note without draining the buffer.
// Test-only: the summariser tests poll it concurrently with the
// backgroundSummarise goroutine, so it takes the same mu that guards Shadow.
func (s *Session) peekShadow() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return strings.Join(s.Shadow, "\n\n")
}

// TestSummarisePrefixIdentity pins the cache-consistency contract of the
// prefix-extension summariser at the byte level: its request must render as
// the foreground request EXTENDED — the exact same tools array, and the
// foreground's messages as a byte-identical prefix — so the server reuses the
// foreground's KV cache whole. Anything less (a missing tools array, a
// re-rendered tool output) diverges the rendered prompt near token 0, which
// on a single slot evicts the foreground cache AND makes the next user turn
// re-evaluate cold (the 12k-uncached regression).
func TestSummarisePrefixIdentity(t *testing.T) {
	mock := newMockLLM(t,
		sseText("did it"),                    // foreground call
		sseText("Goal: g\nProgress: did it"), // prefix-extension summarise
	)
	defer mock.Close()

	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, ".codehalter"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".codehalter", "SUMMARISE.md"), []byte("SUMMARISE PROMPT\n"), 0o644); err != nil {
		t.Fatalf("write SUMMARISE.md: %v", err)
	}
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	a := &agent{
		sessions:       map[string]*Session{s.ID: s},
		settings:       Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}},
		mainSlotTokens: 90_000,
	}

	s.AddUser("do the thing")
	s.markTurnStart()
	fgMsgs := a.buildLLMContext(s)
	if _, _, _, err := a.llmStream(context.Background(), s.ID, a.settings.ConnAt(0, "execute"), fgMsgs, a.tools.defs(), nil, nil, nil); err != nil {
		t.Fatalf("foreground call: %v", err)
	}
	s.AddAssistant("did it")

	a.backgroundSummarise(s)
	deadline := time.Now().Add(2 * time.Second)
	for s.peekShadow() == "" {
		if time.Now().After(deadline) {
			t.Fatalf("backgroundSummarise never appended a note")
		}
		time.Sleep(10 * time.Millisecond)
	}

	fg, sm := mock.request(0), mock.request(1)
	fgTools, _ := json.Marshal(fg["tools"])
	smTools, _ := json.Marshal(sm["tools"])
	if !bytes.Equal(fgTools, smTools) {
		t.Errorf("tools arrays differ between foreground and summarise:\nforeground: %s\nsummarise:  %s", fgTools, smTools)
	}
	fgM, _ := fg["messages"].([]any)
	smM, _ := sm["messages"].([]any)
	if len(smM) <= len(fgM) {
		t.Fatalf("summarise request should EXTEND the foreground context: %d vs %d messages", len(smM), len(fgM))
	}
	for i := range fgM {
		fb, _ := json.Marshal(fgM[i])
		sb, _ := json.Marshal(smM[i])
		if !bytes.Equal(fb, sb) {
			t.Errorf("message %d diverges between foreground and summarise:\nforeground: %s\nsummarise:  %s", i, fb, sb)
		}
	}
}

// TestReplayToolOutputViewImage pins the byte-for-byte replay of a view_image
// tool result. The live call puts []any{text, image_url} on the wire but stores
// only the text half in tu.Output, so rebuilding from tu.Output alone drops an
// image out of the MIDDLE of the prompt and shifts every message behind it. A
// real session paid 30035 re-evaluated tokens to save 431.
func TestReplayToolOutputViewImage(t *testing.T) {
	dir := t.TempDir()
	id := "img_00ff11ee22dd33cc"
	if err := writeImageFile(dir, id, "image/png", []byte("PNG bytes here")); err != nil {
		t.Fatalf("writeImageFile: %v", err)
	}
	sess := &Session{Cwd: dir}
	a := &agent{imagesSupported: true}

	// What the live call put on the wire (tools.go runToolCall).
	text, live, failed := dispatchViewImage(sess, fmt.Sprintf(`{"id":%q}`, id))
	if failed {
		t.Fatalf("dispatchViewImage: failed=true (%s)", text)
	}
	tu := ToolUse{ID: "tu_1", Name: "view_image", Input: fmt.Sprintf(`{"id":%q}`, id), Output: text}

	got := a.replayToolOutput(sess, tu)
	parts, ok := got.([]any)
	if !ok {
		t.Fatalf("replay returned %T, want []any — the image was dropped from history", got)
	}
	if !reflect.DeepEqual(parts, live) {
		t.Errorf("replay differs from the live wire:\n got %#v\nwant %#v", parts, live)
	}

	// A failed view_image never carried parts, so it must replay as text.
	if bad := a.replayToolOutput(sess, ToolUse{ID: "tu_2", Name: "view_image", Failed: true,
		Input: `{"id":"img_gone"}`, Output: "view_image: image not found"}); bad != "view_image: image not found" {
		t.Errorf("failed view_image replayed as %#v, want the stored text", bad)
	}
	// A server without image support never carried parts either.
	noImg := &agent{imagesSupported: false}
	if bad := noImg.replayToolOutput(sess, tu); bad != text {
		t.Errorf("images-off replay = %#v, want the stored text", bad)
	}
	// Any other tool is untouched.
	if got := a.replayToolOutput(sess, ToolUse{ID: "tu_3", Name: "read_file", Output: "hello"}); got != "hello" {
		t.Errorf("read_file replay = %#v, want %q", got, "hello")
	}
}

// TestReplayToolOutputViewImageFileGone: the bytes vanished between the live
// call and the rebuild. Nothing can make that replay identical, so it degrades
// to the stored text instead of failing the turn.
func TestReplayToolOutputViewImageFileGone(t *testing.T) {
	sess := &Session{Cwd: t.TempDir()}
	a := &agent{imagesSupported: true}
	tu := ToolUse{ID: "tu_1", Name: "view_image", Input: `{"id":"img_deadbeefdeadbeef"}`,
		Output: "[Image img_deadbeefdeadbeef re-delivered.]"}
	if got := a.replayToolOutput(sess, tu); got != tu.Output {
		t.Errorf("replay with the file gone = %#v, want the stored text", got)
	}
}

// TestKeepImageRefs: the fold is told to copy image references through
// verbatim, but a reference it paraphrases away is unrecoverable — nothing else
// in the session ever names that id again. They are restored deterministically.
func TestKeepImageRefs(t *testing.T) {
	old := "Goal: one\n\nAttached images:\n" +
		"- img_1111111111111111 (image/png) — call view_image id=img_1111111111111111 to view\n" +
		"- img_2222222222222222 (image/png) — call view_image id=img_2222222222222222 to view\n" +
		"\nGoal: two\n\nAttached images:\n" +
		"- img_1111111111111111 (image/png) — call view_image id=img_1111111111111111 to view"

	// The fold kept one id and dropped the other.
	got := keepImageRefs(old, "Goal: merged\nCritical Context: img_2222222222222222 is the screenshot")
	if !strings.Contains(got, "id=img_1111111111111111 to view") {
		t.Errorf("dropped reference not restored:\n%s", got)
	}
	if n := strings.Count(got, "img_2222222222222222"); n != 1 {
		t.Errorf("id already present was re-appended (%d occurrences):\n%s", n, got)
	}
	// The same id referenced twice in the old summary is restored once.
	got = keepImageRefs(old, "Goal: merged, no ids at all")
	if n := strings.Count(got, "id=img_1111111111111111 to view"); n != 1 {
		t.Errorf("duplicate reference restored %d times, want 1:\n%s", n, got)
	}
	// Nothing missing: the fold is returned untouched.
	folded := "Goal: merged\n" + old
	if got := keepImageRefs(old, folded); got != folded {
		t.Errorf("no-op case rewrote the fold:\n%s", got)
	}
}

// TestBoundSummaryUnderBound: below maxSummaryBytes nothing is folded and no
// LLM call is made — a.settings has no connection here, so a call would be
// visible as an empty/failed result rather than the input coming back.
func TestSummaryFoldIsDeferredAndConsumedAtNextCompaction(t *testing.T) {
	// Under the bound there is nothing to fold and nothing is queued.
	a0, s0 := newTestAgent(t)
	a0.scheduleSummaryFold(s0, strings.Repeat("Goal: small\n", 8))
	s0.waitSummarise()
	if s0.FoldedSummary != "" {
		t.Errorf("a summary under the bound was folded: %q", s0.FoldedSummary)
	}

	folded := "Goal: everything so far, in one line."
	mock := newMockLLM(t, sseText(folded))
	defer mock.Close()

	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, ".codehalter"), 0o755); err != nil {
		t.Fatalf("mkdir .codehalter: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".codehalter", "RESUMMARISE.md"), []byte("RESUMMARISE PROMPT\n"), 0o644); err != nil {
		t.Fatalf("write RESUMMARISE.md: %v", err)
	}
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	big := strings.Repeat("Goal: big\n", maxSummaryBytes/10+64)
	s.Summary = big
	a := &agent{
		sessions:       map[string]*Session{s.ID: s},
		settings:       Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}},
		mainSlotTokens: 90_000,
	}

	// Deferred: the call that schedules the fold returns before the LLM does,
	// and Summary is untouched. Summary leads every request, so rewriting it
	// here would re-prefill the whole prompt behind it.
	a.scheduleSummaryFold(s, big)
	s.waitSummarise()
	if s.Summary != big {
		t.Error("the fold rewrote Summary; it must only ever write FoldedSummary")
	}
	if s.FoldedSummary != folded {
		t.Fatalf("FoldedSummary = %q, want %q", s.FoldedSummary, folded)
	}
	if req := mock.request(0); !strings.Contains(req["messages"].([]any)[0].(map[string]any)["content"].(string), "RESUMMARISE PROMPT") {
		t.Errorf("fold request did not carry RESUMMARISE.md: %v", req)
	}

	// The next compaction consumes it as the base, in place of the long
	// Summary, and clears it. Without an LLM for the notes the in-flight slice
	// falls back to a raw excerpt, which is fine: what matters is the base.
	s.AddUser("next question")
	s.markTurnStart()
	s.AddAssistant("next answer")
	s.appendShadow("Goal: the turn after the fold\nProgress: done")
	if !a.foldHistory(context.Background(), s, len(s.Messages)) {
		t.Fatal("foldHistory did not fold")
	}
	if strings.Contains(s.Summary, big) || !strings.HasPrefix(s.Summary, folded) {
		t.Errorf("compaction did not build on the folded summary; got %d bytes starting %q", len(s.Summary), clipBytes(s.Summary, 80))
	}
	if !strings.Contains(s.Summary, "the turn after the fold") {
		t.Error("compaction dropped the notes it was folding in")
	}
	if s.FoldedSummary != "" {
		t.Errorf("FoldedSummary survived the compaction that consumed it: %q", s.FoldedSummary)
	}
}

// TestSummaryFoldKeepsSummaryOnFailure pins the direction every failure falls
// in: an unreachable or unhelpful summariser must leave the record alone rather
// than replace it with something shorter and worse. Growing beats losing.
func TestSummaryFoldKeepsSummaryOnFailure(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, ".codehalter"), 0o755); err != nil {
		t.Fatalf("mkdir .codehalter: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".codehalter", "RESUMMARISE.md"), []byte("RESUMMARISE PROMPT\n"), 0o644); err != nil {
		t.Fatalf("write RESUMMARISE.md: %v", err)
	}
	big := strings.Repeat("Goal: big\n", maxSummaryBytes/10+64)

	// No reachable summariser at all.
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.Summary = big
	a := &agent{sessions: map[string]*Session{s.ID: s}}
	a.scheduleSummaryFold(s, big)
	s.waitSummarise()
	if s.FoldedSummary != "" || s.Summary != big {
		t.Errorf("unreachable summariser changed the record: folded=%d summary=%d", len(s.FoldedSummary), len(s.Summary))
	}

	// A "fold" that came back longer than its input is not a fold.
	mock := newMockLLM(t, sseText(big+"and more"))
	defer mock.Close()
	s2, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s2.Summary = big
	a2 := &agent{
		sessions:       map[string]*Session{s2.ID: s2},
		settings:       Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}},
		mainSlotTokens: 90_000,
	}
	a2.scheduleSummaryFold(s2, big)
	s2.waitSummarise()
	if s2.FoldedSummary != "" {
		t.Errorf("a longer rewrite was accepted as a fold: %d bytes", len(s2.FoldedSummary))
	}
}

// TestPasteSummariseCarriesPriorSummary pins that a paste-mode note is written
// with the rolling Summary in front of it. Prefix-extension mode replays the
// real wire context, which already renders Summary; a paste sees only the turn
// slice, so without this the first note after a compaction is written by
// something that does not know the session's own goal, and it restates what is
// already recorded directly above where the note will land.
func TestPasteSummariseCarriesPriorSummary(t *testing.T) {
	note := sseText("Goal: g\nProgress: did it")
	mock := newMockLLM(t, note, note)
	defer mock.Close()

	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, ".codehalter"), 0o755); err != nil {
		t.Fatalf("mkdir .codehalter: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".codehalter", "SUMMARISE.md"), []byte("SUMMARISE PROMPT\n"), 0o644); err != nil {
		t.Fatalf("write SUMMARISE.md: %v", err)
	}

	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.Summary = "Goal: port xdocc to the new API\nConstraint: never touch vendor/"
	s.AddUser("keep going")
	s.markTurnStart()
	s.AddAssistant("kept going")

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		// A dedicated summariser takes the paste branch: the conversation is not
		// in that conn's cache, so prefix extension would not line up.
		settings: Settings{LLM: []LLMConnection{
			{Server: mock.ts.URL, Model: "m"},
			{Server: mock.ts.URL, Model: "s", Purpose: purposeSummary},
		}},
		mainSlotTokens: 90_000,
	}
	a.buildConnSems()
	a.backgroundSummarise(s)

	deadline := time.Now().Add(2 * time.Second)
	for s.peekShadow() == "" {
		if time.Now().After(deadline) {
			t.Fatalf("backgroundSummarise never appended a note")
		}
		time.Sleep(10 * time.Millisecond)
	}

	// One pasted message, then the closed think block that turns reasoning off.
	msgs, _ := mock.request(0)["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("paste mode should send the paste plus the prefill, got %d messages", len(msgs))
	}
	m, _ := msgs[0].(map[string]any)
	paste, _ := m["content"].(string)
	for _, want := range []string{
		"SUMMARISE PROMPT",
		"port xdocc to the new API",
		"never touch vendor/",
		"Do NOT repeat any of it",
		"kept going",
	} {
		if !strings.Contains(paste, want) {
			t.Errorf("paste missing %q; got:\n%s", want, paste)
		}
	}
	// Order matters: the recorded block is background for the turn, so it has to
	// arrive before the transcript it is meant to contextualise.
	if strings.Index(paste, "</already_recorded>") > strings.Index(paste, "kept going") {
		t.Errorf("prior summary must precede the turn transcript; got:\n%s", paste)
	}

	// An empty Summary adds nothing: no stray tags, no wasted prompt tokens.
	s2, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s2.AddUser("first turn")
	s2.markTurnStart()
	s2.AddAssistant("done")
	a.sessions[s2.ID] = s2
	a.backgroundSummarise(s2)
	deadline = time.Now().Add(2 * time.Second)
	for s2.peekShadow() == "" {
		if time.Now().After(deadline) {
			t.Fatalf("second backgroundSummarise never appended a note")
		}
		time.Sleep(10 * time.Millisecond)
	}
	msgs, _ = mock.request(1)["messages"].([]any)
	m, _ = msgs[0].(map[string]any)
	if paste, _ := m["content"].(string); strings.Contains(paste, "already_recorded") {
		t.Errorf("empty Summary should add no block; got:\n%s", paste)
	}
}

// TestFallbackTurnNoteIsValidUTF8 pins the note that made a session
// unloadable: a respond call whose input had a two-byte character exactly at
// the 100-byte cut. The note went into Shadow, compaction folded it into
// Summary, and the TOML decoder refused the whole file over that one byte.
func TestFallbackTurnNoteIsValidUTF8(t *testing.T) {
	// fallbackTurnNote keeps the first 100 bytes of a tool input; ± (0xC2 0xB1,
	// the same lead byte as the real file) starts at byte 99.
	prefix := `{"message":"`
	input := prefix + strings.Repeat("a", 99-len(prefix)) + "±" + strings.Repeat("x", 400) + `"}`
	turn := []Message{{Role: "assistant", ToolUses: []ToolUse{{Name: "respond", Input: input}}}}
	if note := fallbackTurnNote(turn); !utf8.ValidString(note) {
		t.Errorf("note is not valid UTF-8: %q", note)
	}
}
