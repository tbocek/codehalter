package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// sse builds a minimal SSE body from chunk payloads plus the [DONE] frame.
func sse(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: ")
		b.WriteString(c)
		b.WriteString("\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func TestPing(t *testing.T) {
	cases := []struct {
		name    string
		status  int
		body    string
		wantErr bool
	}{
		{"ok", 200, sse(`{"choices":[{"delta":{"content":"ok"}}]}`), false},
		{"ok reasoning only", 200, sse(`{"choices":[{"delta":{"reasoning_content":"hmm"}}]}`), false}, // thinking model — still reachable
		{"ok empty stream", 200, sse(), false},
		{"unauthorized", 401, `{"error":{"message":"bad key"}}`, true},
		{"model not found", 400, `{"error":{"code":400,"message":"model 'x' not found"}}`, true},
		{"mid-stream error", 200, sse(`{"error":{"message":"no slot available"}}`), true},
		{"bare message error", 200, sse(`{"message":"context exceeded"}`), true},
		{"gateway down", 502, `bad gateway`, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(c.status)
				_, _ = w.Write([]byte(c.body))
			}))
			defer ts.Close()
			err := ping(context.Background(), ModelSpec{Name: "t", Server: ts.URL, Model: "m"})
			if (err != nil) != c.wantErr {
				t.Fatalf("ping err = %v, wantErr = %v", err, c.wantErr)
			}
		})
	}
}

func TestChatStreamAccumulates(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Request must carry stream=true and the model id.
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || !req.Stream || req.Model != "m" {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		_, _ = w.Write([]byte(sse(
			`{"choices":[{"delta":{"reasoning_content":"thinking..."}}]}`,
			`{"choices":[{"delta":{"content":"hello "}}]}`,
			`{"choices":[{"delta":{"content":"world"}}]}`,
		)))
	}))
	defer ts.Close()
	got, err := chat(context.Background(), ModelSpec{Name: "t", Server: ts.URL, Model: "m"}, "sys", "user")
	if err != nil {
		t.Fatal(err)
	}
	if got != "hello world" {
		t.Fatalf("chat = %q, want %q", got, "hello world")
	}
}

func TestChatReasoningOnlyIsError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"reasoning_content":"endless pondering"}}]}`)))
	}))
	defer ts.Close()
	_, err := chat(context.Background(), ModelSpec{Name: "t", Server: ts.URL, Model: "m"}, "", "q")
	if err == nil || !strings.Contains(err.Error(), "reasoning channel") {
		t.Fatalf("want reasoning-channel error, got: %v", err)
	}
}

// loopHandler streams the same sentence until the client hangs up (guard trip)
// or the frame budget runs out.
func loopHandler(w http.ResponseWriter, r *http.Request) {
	loop := `{"choices":[{"delta":{"content":"the same thing again and again and again. "}}]}`
	fl, _ := w.(http.Flusher)
	for i := 0; i < 500; i++ {
		if _, err := fmt.Fprintf(w, "data: %s\n\n", loop); err != nil {
			return
		}
		if fl != nil {
			fl.Flush()
		}
	}
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
}

func TestChatNudgeRecoversFromRepetition(t *testing.T) {
	// First call loops; the retry must carry the nudge in the user message and
	// gets a clean answer.
	var calls, nudged int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		last := req.Messages[len(req.Messages)-1].Content
		if strings.Contains(last, "aborted because its reasoning kept repeating") {
			nudged++
			_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"final answer"}}]}`)))
			return
		}
		loopHandler(w, r)
	}))
	defer ts.Close()
	got, err := chat(context.Background(), ModelSpec{Name: "t", Server: ts.URL, Model: "m"}, "sys", "q")
	if err != nil {
		t.Fatalf("nudged retry should succeed: %v", err)
	}
	if got != "final answer" || calls != 2 || nudged != 1 {
		t.Fatalf("got %q, calls=%d nudged=%d — want answer via exactly one nudged retry", got, calls, nudged)
	}
}

func TestChatRepetitionPersistsAfterNudge(t *testing.T) {
	// Loops on every attempt: chat retries once (nudge), then surfaces the error.
	var calls int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		loopHandler(w, r)
	}))
	defer ts.Close()
	_, err := chat(context.Background(), ModelSpec{Name: "t", Server: ts.URL, Model: "m"}, "", "q")
	if err == nil || !strings.Contains(err.Error(), "looping") {
		t.Fatalf("want looping error after failed nudge, got: %v", err)
	}
	if calls != 2 {
		t.Fatalf("want exactly 2 attempts (original + nudged), got %d", calls)
	}
}

func TestRepetitionGuardPassesVariedText(t *testing.T) {
	var g repetitionGuard
	// 8KB of genuinely varied text: distinct words per window.
	for i := 0; i < 2000; i++ {
		if err := g.add(fmt.Sprintf("word%d item%d ", i, i*7)); err != nil {
			t.Fatalf("varied text tripped the guard: %v", err)
		}
	}
	// Structural repetition with varied content (like the segmenter's claims
	// JSON: shared keys, different values) must pass too.
	g = repetitionGuard{}
	for i := 0; i < 200; i++ {
		line := fmt.Sprintf(`{"text":"claim about topic %d field %d","source":"- bullet %d says thing %d"},`, i, i*3, i, i*5)
		if err := g.add(line); err != nil {
			t.Fatalf("claims-shaped JSON tripped the guard: %v", err)
		}
	}
}

func TestWordBagAndJaccard(t *testing.T) {
	a := wordBag("Hello, world! HELLO?")
	if len(a) != 2 || !a["hello"] || !a["world"] {
		t.Fatalf("wordBag = %v", a)
	}
	if j := jaccard(a, wordBag("hello world")); j != 1.0 {
		t.Fatalf("identical bags jaccard = %v", j)
	}
	if j := jaccard(a, wordBag("совершенно other words")); j >= 0.5 {
		t.Fatalf("disjoint-ish bags jaccard = %v", j)
	}
	if j := jaccard(map[string]bool{}, map[string]bool{}); j != 1.0 {
		t.Fatalf("empty bags jaccard = %v", j)
	}
}

func TestLLMLogTracesCalls(t *testing.T) {
	path := filepath.Join(t.TempDir(), "llm.jsonl")
	if err := openLLMLog(path); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { llmLog.f.Close(); llmLog.f = nil })

	ok := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(sse(
			`{"choices":[{"delta":{"reasoning_content":"let me think"}}]}`,
			`{"choices":[{"delta":{"content":"the answer"}}]}`,
		)))
	}))
	defer ok.Close()
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"model not found"}}`))
	}))
	defer bad.Close()

	if _, err := chat(context.Background(), ModelSpec{Name: "good", Server: ok.URL, Model: "m1"}, "sys prompt", "user prompt"); err != nil {
		t.Fatal(err)
	}
	if _, err := chat(context.Background(), ModelSpec{Name: "broken", Server: bad.URL, Model: "m2"}, "", "hello"); err == nil {
		t.Fatal("expected error from bad server")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// Keep only final lines — short streams still emit one live chunk line
	// each, tested separately in TestLLMLogChunksStreamLive.
	var finals []llmLogEntry
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var e llmLogEntry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatal(err)
		}
		if !e.Chunk {
			finals = append(finals, e)
		}
	}
	if len(finals) != 2 {
		t.Fatalf("want 2 final trace lines, got %d: %s", len(finals), data)
	}
	e1, e2 := finals[0], finals[1]
	// Success line: full messages as sent, the reply, and the thinking channel.
	if e1.Name != "good" || len(e1.Messages) != 2 || e1.Messages[0].Content != "sys prompt" ||
		e1.Response != "the answer" || e1.Reasoning != "let me think" {
		t.Fatalf("success trace wrong: %+v", e1)
	}
	// Failure line: messages still captured, error recorded.
	if e2.Name != "broken" || len(e2.Messages) != 1 || !strings.Contains(e2.Error, "400") {
		t.Fatalf("error trace wrong: %+v", e2)
	}
}

func TestParamsMergeIntoRequestBody(t *testing.T) {
	var got map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"ok"}}]}`)))
	}))
	defer ts.Close()

	temp := 0.6
	spec := ModelSpec{
		Name: "t", Server: ts.URL, Model: "m", Temperature: &temp,
		Params: map[string]any{"top_k": 20, "presence_penalty": 1.5, "temperature": 0.9},
	}
	if _, err := chat(context.Background(), spec, "", "q"); err != nil {
		t.Fatal(err)
	}
	if got["top_k"] != float64(20) || got["presence_penalty"] != 1.5 {
		t.Fatalf("params not merged: %v", got)
	}
	// A params key overrides the named field.
	if got["temperature"] != 0.9 {
		t.Fatalf("params should override named temperature, got %v", got["temperature"])
	}
	if got["stream"] != true || got["model"] != "m" {
		t.Fatalf("base fields missing: %v", got)
	}
}

func TestChatSerializesPerEndpoint(t *testing.T) {
	var inFlight, peak int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cur := atomic.AddInt32(&inFlight, 1)
		for {
			p := atomic.LoadInt32(&peak)
			if cur <= p || atomic.CompareAndSwapInt32(&peak, p, cur) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond) // hold the "slot" long enough to overlap
		atomic.AddInt32(&inFlight, -1)
		_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"ok"}}]}`)))
	}))
	defer ts.Close()

	spec := ModelSpec{Name: "t", Server: ts.URL, Model: "m", sem: make(chan struct{}, 1)}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := chat(context.Background(), spec, "", "q"); err != nil {
				t.Errorf("chat: %v", err)
			}
		}()
	}
	wg.Wait()
	if p := atomic.LoadInt32(&peak); p != 1 {
		t.Fatalf("parallel=1 must serialize; peak concurrent requests = %d", p)
	}
}

func TestLLMLogChunksStreamLive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "llm.jsonl")
	if err := openLLMLog(path); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { llmLog.f.Close(); llmLog.f = nil })

	// ~3.4KB of varied content in 100 SSE frames → expect ≥3 chunk lines.
	var frames []string
	for i := 0; i < 100; i++ {
		frames = append(frames, fmt.Sprintf(`{"choices":[{"delta":{"content":"piece %03d of the answer "}}]}`, i))
	}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(sse(frames...)))
	}))
	defer ts.Close()

	got, err := chat(context.Background(), ModelSpec{Name: "t", Server: ts.URL, Model: "m"}, "", "q")
	if err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var chunks []llmLogEntry
	var finals []llmLogEntry
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var e llmLogEntry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatal(err)
		}
		if e.Chunk {
			chunks = append(chunks, e)
		} else {
			finals = append(finals, e)
		}
	}
	if len(chunks) < 3 {
		t.Fatalf("want ≥3 chunk lines for a ~3.4KB stream, got %d", len(chunks))
	}
	if len(finals) != 1 {
		t.Fatalf("want exactly 1 final line, got %d", len(finals))
	}
	// Chunk lines carry no messages (kept light); final line carries them.
	if len(chunks[0].Messages) != 0 || len(finals[0].Messages) != 1 {
		t.Fatalf("messages placement wrong: chunk=%d final=%d", len(chunks[0].Messages), len(finals[0].Messages))
	}
	// Concatenated chunk deltas reproduce the final line's raw response
	// exactly; chat() returns it whitespace-trimmed.
	var cat strings.Builder
	for _, c := range chunks {
		cat.WriteString(c.Response)
	}
	if cat.String() != finals[0].Response || strings.TrimSpace(cat.String()) != got {
		t.Fatalf("chunk concat (%d B) != final raw (%d B) / trimmed response (%d B)", cat.Len(), len(finals[0].Response), len(got))
	}
}

func TestExtractJSON(t *testing.T) {
	cases := []struct {
		name, in, want string
	}{
		{"bare", `{"a":1}`, `{"a":1}`},
		{"prefixed", "sure, here:\n{\"a\":1}", `{"a":1}`},
		{"fenced", "```json\n{\"a\":1}\n```", `{"a":1}`},
		{"brace in string", `{"a":"has } brace"}`, `{"a":"has } brace"}`},
		{"escaped quote", `{"a":"say \"hi\""}`, `{"a":"say \"hi\""}`},
		{"nested", `pre {"a":{"b":2}} post`, `{"a":{"b":2}}`},
		{"none", `no json here`, ``},
		{"unbalanced", `{"a":1`, ``},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := extractJSON(c.in); got != c.want {
				t.Fatalf("extractJSON(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

func TestTruncate(t *testing.T) {
	if got := truncate("hello", 10); got != "hello" {
		t.Fatalf("short truncate = %q", got)
	}
	if got := truncate("hello world", 5); got != "hello…" {
		t.Fatalf("long truncate = %q", got)
	}
}
