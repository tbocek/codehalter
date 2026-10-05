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

// A boundary compaction archives the old state, folds the whole Shadow into
// Summary after the prior one, keeps nothing verbatim and persists, all without
// an LLM call.
func TestFoldHistoryRecordsSummary(t *testing.T) {
	mock := newMockLLM(t)
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.Summary = "PRIOR SUMMARY FROM AN EARLIER COMPACTION"

	filler := strings.Repeat("lorem ipsum ", 100)
	for i := 0; i < 10; i++ {
		s.AddUser(fmt.Sprintf("user msg %d %s", i, filler))
		s.AddAssistant(fmt.Sprintf("asst msg %d %s", i, filler))
	}
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}
	originalMsgCount := len(s.Messages)

	s.appendShadow("Goal: ship feature\nProgress: scaffolded module")
	s.appendShadow("Goal: ship feature\nProgress: wired up handler")
	s.appendShadow("Goal: ship feature\nProgress: shipped it")

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}},
		},
	}

	s.turnStartIdx = len(s.Messages) // all turns completed → foldHistory(len) folds them all via shadow
	a.foldHistory(context.Background(), s, len(s.Messages))

	if !strings.HasPrefix(s.Summary, "PRIOR SUMMARY") {
		t.Errorf("the prior Summary was dropped or moved; got %q", s.Summary)
	}
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

	// The live session keeps its ID and path; the archive holds the old state.
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

// Every message already sent, the system prompt first, must replay byte-identically
// on the next turn, or the server's KV prefix cache misses.
func TestPrefixStableAcrossTurns(t *testing.T) {
	dir := t.TempDir()

	// A non-empty system prompt, or dropping it on turn 2 would go unnoticed.
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

	sysPrompt, _ := a.systemPrompt(s.ID)
	if sysPrompt == "" {
		t.Fatal("expected non-empty systemPrompt: SKILL seed didn't take effect")
	}

	s.SystemPrompt = sysPrompt
	s.AddUser("first prompt")
	msgs1 := a.buildLLMContext(s)

	s.AddAssistant("done with turn 1")

	s.AddUser("second prompt")
	msgs2 := a.buildLLMContext(s)

	if len(msgs2) <= len(msgs1) {
		t.Fatalf("turn 2 should extend turn 1's history; got len1=%d len2=%d",
			len(msgs1), len(msgs2))
	}
	for i := range msgs1 {
		b1, _ := json.Marshal(msgs1[i])
		b2, _ := json.Marshal(msgs2[i])
		if !bytes.Equal(b1, b2) {
			t.Errorf("prefix message %d drifted between turns:\n  turn 1: %s\n  turn 2: %s",
				i, b1, b2)
		}
	}
}

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
	}

	a.foldHistory(context.Background(), s, 0)

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

func contentString(t *testing.T, m llmMessage) string {
	t.Helper()
	s, ok := m.Content.(string)
	if !ok {
		t.Fatalf("expected string content, got %T", m.Content)
	}
	return s
}

// A Summary becomes one leading header message; without one there is no header.
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

// Also covers an assistant message with only tool calls and a truncated long output.
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
	if msgs[5].Role != "assistant" || contentString(t, msgs[5]) != "" || len(msgs[5].ToolCalls) != 1 {
		t.Errorf("empty-content assistant wrong: %+v", msgs[5])
	}
	if msgs[6].Role != "tool" || msgs[6].ToolCallID != "tu_3" {
		t.Errorf("tool message [2] wrong: %+v", msgs[6])
	}
}

// Every stored image is inlined every turn, so wire bytes stay identical until compaction.
func TestBuildLLMHistoryImageHandling(t *testing.T) {
	dir := t.TempDir()
	id1, err := storeImage(dir, "image/png", []byte("pngbytes-1"))
	if err != nil {
		t.Fatalf("storeImage id1: %v", err)
	}
	id2, err := storeImage(dir, "image/png", []byte("pngbytes-2-different"))
	if err != nil {
		t.Fatalf("storeImage id2: %v", err)
	}
	img1 := ImageData{ID: id1, MimeType: "image/png"}
	img2 := ImageData{ID: id2, MimeType: "image/png"}
	a := &agent{}
	a.imagesSupported.Store(true)

	t.Run("images off: placeholders with view_image hints", func(t *testing.T) {
		s := &Session{Cwd: dir, Messages: []Message{{Role: "user", Content: "look at this", Images: []ImageData{img1, img2}}}}
		out := (&agent{}).buildLLMContext(s)
		if len(out) != 1 {
			t.Fatalf("expected 1 message, got %d", len(out))
		}
		got := contentString(t, out[0])
		for _, id := range []string{id1, id2} {
			if !strings.Contains(got, "[Image "+id) || !strings.Contains(got, "view_image id="+id) {
				t.Errorf("expected a placeholder and a view_image hint for %s in %q", id, got)
			}
		}
	})

	t.Run("images on: text then image parts", func(t *testing.T) {
		s := &Session{Cwd: dir, Messages: []Message{{Role: "user", Content: "look at this", Images: []ImageData{img1, img2}}}}
		parts, ok := a.buildLLMContext(s)[0].Content.([]any)
		if !ok || len(parts) != 3 {
			t.Fatalf("expected 3 parts (text + 2 images), got %+v", parts)
		}
		if text, _ := parts[0].(map[string]any); text["type"] != "text" || text["text"] != "look at this" {
			t.Errorf("parts[0] wrong: %+v", text)
		}
		for i, part := range parts[1:] {
			block, _ := part.(map[string]any)
			url, _ := block["image_url"].(map[string]string)
			if block["type"] != "image_url" || !strings.HasPrefix(url["url"], "data:image/png;base64,") {
				t.Errorf("parts[%d] = %+v, want a png image_url", i+1, block)
			}
		}
	})

	t.Run("an older image stays inlined and rebuilds are identical", func(t *testing.T) {
		s := &Session{Cwd: dir, Messages: []Message{
			{Role: "user", Content: "earlier", Images: []ImageData{img1}},
			{Role: "user", Content: "now look at this one", Images: []ImageData{img2}},
		}}
		out := a.buildLLMContext(s)
		if len(out) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(out))
		}
		for i, m := range out {
			parts, ok := m.Content.([]any)
			if !ok || len(parts) != 2 {
				t.Fatalf("message %d: expected []any of len 2 (text + image_url), got %+v", i, m.Content)
			}
			if img, _ := parts[1].(map[string]any); img["type"] != "image_url" {
				t.Errorf("message %d parts[1] type: got %v, want image_url", i, img["type"])
			}
		}
		first, err := json.Marshal(out)
		if err != nil {
			t.Fatalf("marshal first: %v", err)
		}
		second, err := json.Marshal(a.buildLLMContext(s))
		if err != nil {
			t.Fatalf("marshal second: %v", err)
		}
		if !bytes.Equal(first, second) {
			t.Errorf("cache consistency: wire bytes differ between consecutive rebuilds\nfirst:  %s\nsecond: %s", first, second)
		}
	})

	t.Run("a missing file becomes a text part naming the id", func(t *testing.T) {
		s := &Session{Cwd: dir, Messages: []Message{{Role: "user", Content: "missing", Images: []ImageData{{ID: "img_deadbeef00000000", MimeType: "image/png"}}}}}
		parts, ok := a.buildLLMContext(s)[0].Content.([]any)
		if !ok || len(parts) != 2 {
			t.Fatalf("expected []any of len 2 (text + fallback), got %+v", parts)
		}
		fallback, _ := parts[1].(map[string]any)
		if text, _ := fallback["text"].(string); fallback["type"] != "text" || !strings.Contains(text, "img_deadbeef00000000") {
			t.Errorf("fallback = %+v, want a text part naming the id", fallback)
		}
	})

	t.Run("no images: plain string content", func(t *testing.T) {
		s := &Session{Cwd: dir, Messages: []Message{{Role: "user", Content: "no imgs"}}}
		if got := contentString(t, a.buildLLMContext(s)[0]); got != "no imgs" {
			t.Errorf("plain: got %q, want 'no imgs'", got)
		}
	})

	t.Run("images and tool calls on one message", func(t *testing.T) {
		s := &Session{Cwd: dir, Messages: []Message{{
			Role:     "assistant",
			Content:  "done",
			Images:   []ImageData{img1},
			ToolUses: []ToolUse{{ID: "tu_77", Name: "read_file", Input: `{"path":"x"}`, Output: "ok"}},
		}}}
		out := a.buildLLMContext(s)
		if len(out) != 2 {
			t.Fatalf("expected 2 messages (assistant + tool), got %d: %+v", len(out), out)
		}
		parts, ok := out[0].Content.([]any)
		if !ok || len(parts) != 2 {
			t.Fatalf("assistant Content: expected []any of len 2, got %+v", out[0].Content)
		}
		if text, _ := parts[0].(map[string]any); text["text"] != "done" {
			t.Errorf("text block: got %v, want %q", text["text"], "done")
		}
		if len(out[0].ToolCalls) != 1 || out[0].ToolCalls[0].ID != "tu_77" || out[0].ToolCalls[0].Function.Name != "read_file" {
			t.Errorf("ToolCalls wrong: %+v", out[0].ToolCalls)
		}
		if c, _ := out[1].Content.(string); out[1].Role != "tool" || out[1].ToolCallID != "tu_77" || c != "ok" {
			t.Errorf("tool message wrong: %+v", out[1])
		}
	})
}

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
	imgID, err := storeImage(dir, "image/png", bytes)
	if err != nil {
		t.Fatalf("storeImage: %v", err)
	}
	s.AddUser("look at this", ImageData{ID: imgID, MimeType: "image/png"})
	s.AddAssistant("I see a screenshot.")

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}},
		},
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

	s.appendShadow("Goal: follow-up\nProgress: follow-up turn")
	s.turnStartIdx = len(s.Messages)
	a.foldHistory(context.Background(), s, len(s.Messages))

	if !strings.Contains(s.Summary, "view_image id="+imgID) {
		t.Errorf("Summary lost the image handle after compaction; got %q", s.Summary)
	}
}

func TestFoldHistoryMidTurnKeepsInFlightTurn(t *testing.T) {
	mock := newMockLLM(t) // folding is local; any LLM call fails the test.
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}

	filler := strings.Repeat("lorem ipsum ", 100)
	s.AddUser("turn1 user " + filler)
	s.AddAssistant("turn1 asst " + filler)
	s.AddUser("turn2 user " + filler)
	s.markTurnStart()
	s.AddAssistant("turn2 asst step " + filler)

	s.appendShadow("Goal: do\nProgress: finished turn 1")

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}},
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
	s.AddAssistant("reading files")
	s.AppendToolUse(ToolUse{ID: "tu_1", Name: "read_file", Input: `{"path":"a.go"}`, Output: "package a"})
	s.AddAssistant("done with the thing")

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}},
	}
	a.backgroundSummarise(s)

	deadline := time.Now().Add(2 * time.Second)
	for s.peekShadow() == "" {
		if time.Now().After(deadline) {
			t.Fatalf("backgroundSummarise never appended a note")
		}
		time.Sleep(10 * time.Millisecond)
	}

	// One [[llm]] entry means prefix-extension: the replayed conversation, then the
	// instruction anchored to the turn's opening user message.
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
	// Reasoning off: a closed think block follows the instruction.
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
	// The template renders tools at the prompt head, so without them the prefix diverges.
	if tools, _ := req["tools"].([]any); len(tools) == 0 {
		t.Errorf("prefix-extension summarise request must carry the foreground tools array, got none")
	}
	if tc, _ := req["tool_choice"].(string); tc != "none" {
		t.Errorf("prefix-extension summarise request should set tool_choice=none, got %v", req["tool_choice"])
	}
}

// Both the tool call and its result carry the model's id, so replay is byte-identical.
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
}

// Takes mu: tests poll it while the backgroundSummarise goroutine writes Shadow.
func (s *Session) peekShadow() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return strings.Join(s.Shadow, "\n\n")
}

// The summariser request must be the foreground request extended: same tools
// array and a byte-identical message prefix, so the server reuses the whole cache.
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
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}},
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

// Only the text half is stored, so replay must re-render the image parts or the
// prompt shifts from that message on.
func TestReplayToolOutputViewImage(t *testing.T) {
	dir := t.TempDir()
	id, err := storeImage(dir, "image/png", []byte("PNG bytes here"))
	if err != nil {
		t.Fatalf("storeImage: %v", err)
	}
	sess := &Session{Cwd: dir}
	a := &agent{}
	a.imagesSupported.Store(true)

	// What the live call put on the wire (tools.go runToolCall).
	text, live, failed := dispatchViewImage(sess, fmt.Sprintf(`{"id":%q}`, id))
	if failed {
		t.Fatalf("dispatchViewImage: failed=true (%s)", text)
	}
	tu := ToolUse{ID: "tu_1", Name: "view_image", Input: fmt.Sprintf(`{"id":%q}`, id), Output: text}

	got := a.replayToolOutput(sess, tu)
	parts, ok := got.([]any)
	if !ok {
		t.Fatalf("replay returned %T, want []any: the image was dropped from history", got)
	}
	if !reflect.DeepEqual(parts, live) {
		t.Errorf("replay differs from the live wire:\n got %#v\nwant %#v", parts, live)
	}

	if bad := a.replayToolOutput(sess, ToolUse{ID: "tu_2", Name: "view_image", Failed: true,
		Input: `{"id":"img_gone"}`, Output: "view_image: image not found"}); bad != "view_image: image not found" {
		t.Errorf("failed view_image replayed as %#v, want the stored text", bad)
	}
	noImg := &agent{}
	if bad := noImg.replayToolOutput(sess, tu); bad != text {
		t.Errorf("images-off replay = %#v, want the stored text", bad)
	}
	if got := a.replayToolOutput(sess, ToolUse{ID: "tu_3", Name: "read_file", Output: "hello"}); got != "hello" {
		t.Errorf("read_file replay = %#v, want %q", got, "hello")
	}
}

// Degrades to the stored text instead of failing the turn.
func TestReplayToolOutputViewImageFileGone(t *testing.T) {
	sess := &Session{Cwd: t.TempDir()}
	a := &agent{}
	a.imagesSupported.Store(true)
	tu := ToolUse{ID: "tu_1", Name: "view_image", Input: `{"id":"img_deadbeefdeadbeef"}`,
		Output: "[Image img_deadbeefdeadbeef re-delivered.]"}
	if got := a.replayToolOutput(sess, tu); got != tu.Output {
		t.Errorf("replay with the file gone = %#v, want the stored text", got)
	}
}

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
	folded := "Goal: merged\n" + old
	if got := keepImageRefs(old, folded); got != folded {
		t.Errorf("no-op case rewrote the fold:\n%s", got)
	}
}

func TestSummaryFoldIsDeferredAndConsumedAtNextCompaction(t *testing.T) {
	// Under the bound there is nothing to fold and nothing is queued.
	a0, s0 := newTestAgent(t)
	a0.scheduleSummaryFold(s0, strings.Repeat("Goal: small\n", 8))
	s0.waitSummarise()
	if s0.FoldedSummary != "" {
		t.Errorf("a summary under the bound was folded: %q", s0.FoldedSummary)
	}

	folded := "Goal: everything so far, in one line."
	// The fold, then the note for the in-flight slice the compaction rotates out.
	mock := newMockLLM(t, sseText(folded), sseText("Goal: the slice that rotated out"))
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
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}},
	}

	// Deferred, Summary untouched: rewriting it would re-prefill the whole prompt.
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
		sessions: map[string]*Session{s2.ID: s2},
		settings: Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}},
	}
	a2.scheduleSummaryFold(s2, big)
	s2.waitSummarise()
	if s2.FoldedSummary != "" {
		t.Errorf("a longer rewrite was accepted as a fold: %d bytes", len(s2.FoldedSummary))
	}
}

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
		// A dedicated summariser takes the paste branch.
		settings: Settings{LLM: []LLMConnection{
			{Server: mock.ts.URL, Model: "m"},
			{Server: mock.ts.URL, Model: "s", Purpose: purposeSummary},
		}},
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
	if strings.Index(paste, "</already_recorded>") > strings.Index(paste, "kept going") {
		t.Errorf("prior summary must precede the turn transcript; got:\n%s", paste)
	}

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

// A rune split at the clip makes the session TOML unloadable.
func TestFallbackTurnNoteIsValidUTF8(t *testing.T) {
	// clipBytes keeps a head and a tail of half the limit each: put a two-byte
	// rune across both cuts of the tool input (200) and of the message (800).
	straddle := func(prefix string, limit, total int) string {
		half := limit / 2
		head := prefix + strings.Repeat("a", half-1-len(prefix)) + "±"
		tail := "±" + strings.Repeat("z", half-1)
		return head + strings.Repeat("x", total-len(head)-len(tail)) + tail
	}
	input := straddle(`{"message":"`, 200, 600)
	content := straddle("", 800, 2000)
	turn := []Message{{Role: "assistant", Content: content, ToolUses: []ToolUse{{Name: "respond", Input: input}}}}
	if note := fallbackTurnNote(turn); !utf8.ValidString(note) {
		t.Errorf("note is not valid UTF-8: %q", note)
	}
}
