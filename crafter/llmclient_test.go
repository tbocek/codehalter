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

func TestChatWithToolsParsesStreamedCalls(t *testing.T) {
	var gotBody map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		// Name arrives on the first frame, arguments fragmented over three.
		_, _ = w.Write([]byte(sse(
			`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"web_search","arguments":"{\"que"}}]}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ry\":\"kubectl inst"}}]}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"all\"}"}}]}}]}`,
		)))
	}))
	defer ts.Close()

	tools := buildProbeTools([]string{"web_search", "run_command"})
	content, calls, err := chatWithTools(context.Background(), ModelSpec{Name: "t", Server: ts.URL, Model: "m"}, "sys", "install kubectl", tools)
	if err != nil {
		t.Fatal(err)
	}
	// No visible text, one call: valid for a tool probe.
	if content != "" || len(calls) != 1 {
		t.Fatalf("content=%q calls=%d, want empty content + 1 call", content, len(calls))
	}
	if got := calls[0].render(); got != `web_search({"query":"kubectl install"})` {
		t.Fatalf("call = %q", got)
	}
	// The request body must carry the tools array.
	reqTools, ok := gotBody["tools"].([]any)
	if !ok || len(reqTools) != 2 {
		t.Fatalf("tools not sent: %v", gotBody["tools"])
	}
}

func TestChatWithToolsEmptyBothIsError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(sse()))
	}))
	defer ts.Close()
	_, _, err := chatWithTools(context.Background(), ModelSpec{Name: "t", Server: ts.URL, Model: "m"}, "", "q", buildProbeTools([]string{"web_search"}))
	if err == nil || !strings.Contains(err.Error(), "no tool calls") {
		t.Fatalf("want empty-answer-and-no-calls error, got: %v", err)
	}
}

func TestBuildProbeToolsSkipsUnknown(t *testing.T) {
	tools := buildProbeTools([]string{"web_search", "made_up_tool"})
	if len(tools) != 1 {
		t.Fatalf("want 1 resolved tool, got %d", len(tools))
	}
}

// parallel=2 must allow exactly two in-flight requests: the cap is enforced
// (never 3+) and actually used (two overlap given enough concurrent callers).
func TestDetectSlots(t *testing.T) {
	t.Run("router props?model", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Router: bare /props reports no slots, /props?model= reports 3.
			if r.URL.Query().Get("model") == "" {
				_, _ = w.Write([]byte(`{"total_slots":0}`))
				return
			}
			_, _ = w.Write([]byte(`{"total_slots":3}`))
		}))
		defer ts.Close()
		if n := detectSlots(context.Background(), ModelSpec{Server: ts.URL, Model: "Qwen (x; y)"}); n != 3 {
			t.Fatalf("detected %d, want 3", n)
		}
	})
	t.Run("direct bare props", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"total_slots":2}`))
		}))
		defer ts.Close()
		if n := detectSlots(context.Background(), ModelSpec{Server: ts.URL, Model: "m"}); n != 2 {
			t.Fatalf("detected %d, want 2", n)
		}
	})
	t.Run("non-llamacpp backend", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotFound)
		}))
		defer ts.Close()
		if n := detectSlots(context.Background(), ModelSpec{Server: ts.URL, Model: "m"}); n != 0 {
			t.Fatalf("undetectable server should yield 0, got %d", n)
		}
	})
}

func TestChatParallelTwoAllowsTwoInFlight(t *testing.T) {
	var inFlight, peak int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cur := atomic.AddInt32(&inFlight, 1)
		for {
			p := atomic.LoadInt32(&peak)
			if cur <= p || atomic.CompareAndSwapInt32(&peak, p, cur) {
				break
			}
		}
		time.Sleep(50 * time.Millisecond)
		atomic.AddInt32(&inFlight, -1)
		_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"ok"}}]}`)))
	}))
	defer ts.Close()

	spec := ModelSpec{Name: "t", Server: ts.URL, Model: "m", sem: make(chan struct{}, 2)}
	var wg sync.WaitGroup
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := chat(context.Background(), spec, "", "q"); err != nil {
				t.Errorf("chat: %v", err)
			}
		}()
	}
	wg.Wait()
	if p := atomic.LoadInt32(&peak); p != 2 {
		t.Fatalf("parallel=2: peak in-flight = %d, want exactly 2", p)
	}
}

// Concurrent calls (judge prefetch + probe verdicts in the real run) write the
// trace from multiple goroutines — every line must stay valid, unmangled JSON.
func TestConcurrentCallsTraceCleanly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "llm.jsonl")
	if err := openLLMLog(path); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { llmLog.f.Close(); llmLog.f = nil })

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// >1KB of VARIED text so every call emits chunk lines concurrently
		// without tripping the repetition guard.
		var big strings.Builder
		for i := 0; i < 300; i++ {
			fmt.Fprintf(&big, "word%d ", i)
		}
		_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, big.String()))))
	}))
	defer ts.Close()

	const callers = 8
	var wg sync.WaitGroup
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			spec := ModelSpec{Name: fmt.Sprintf("m%d", i), Server: ts.URL, Model: "m", sem: make(chan struct{}, 1)}
			if _, err := chat(context.Background(), spec, "", "q"); err != nil {
				t.Errorf("chat: %v", err)
			}
		}(i)
	}
	wg.Wait()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	finals := 0
	for n, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var e llmLogEntry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("trace line %d corrupted by concurrent writes: %v", n+1, err)
		}
		if !e.Chunk {
			finals++
		}
	}
	if finals != callers {
		t.Fatalf("finals = %d, want %d", finals, callers)
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

func TestParseJSONReply(t *testing.T) {
	type verdict struct {
		A int    `json:"a"`
		S string `json:"s"`
	}
	cases := []struct {
		name, in string
		ok       bool
		want     verdict
	}{
		{"bare", `{"a":1}`, true, verdict{A: 1}},
		{"prefixed", "sure, here:\n{\"a\":1}", true, verdict{A: 1}},
		{"fenced", "```json\n{\"a\":1}\n```", true, verdict{A: 1}},
		{"brace in string", `{"a":2,"s":"has } brace"}`, true, verdict{A: 2, S: "has } brace"}},
		{"escaped quote", `{"a":3,"s":"say \"hi\""}`, true, verdict{A: 3, S: `say "hi"`}},
		// The bash#c74e12bf failure: brace prose BEFORE the object must not
		// shadow it — the old first-span extractor returned `{var}` and errored.
		{"code prose before object", "the answer quotes ${var} correctly, so:\n{\"a\":7}", true, verdict{A: 7}},
		{"multiple junk candidates", "try {1..5} or {} then {\"a\":9}", true, verdict{A: 9}},
		// A bare {} in prose parses to a zero value: skipped in favor of the
		// real object; accepted only when nothing non-zero exists (the caller's
		// field validation then produces the precise error).
		{"only empty object", `here: {}`, true, verdict{}},
		{"none", `no json here`, false, verdict{}},
		{"unbalanced only", `{"a":1`, false, verdict{}},
		// A truncated outer object must not stop the scan: a later complete one
		// still wins.
		{"truncated then complete", `{"a":1  ... oops. Retry: {"a":5}`, true, verdict{A: 5}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var got verdict
			ok := parseJSONReply(c.in, &got)
			if ok != c.ok {
				t.Fatalf("parseJSONReply(%q) ok = %v, want %v", c.in, ok, c.ok)
			}
			if got != c.want {
				t.Fatalf("parseJSONReply(%q) = %+v, want %+v", c.in, got, c.want)
			}
		})
	}
}

// A failed candidate must not leave partial fields behind in out.
func TestParseJSONReplyNoPartialPollution(t *testing.T) {
	type v struct {
		A int    `json:"a"`
		B string `json:"b"`
	}
	// First candidate {"a":1,"b":3} type-errors on b AFTER a would be set;
	// the real object follows. out must contain exactly the second candidate.
	var got v
	if ok := parseJSONReply(`{"a":1,"b":3} then {"a":2,"b":"ok"}`, &got); !ok {
		t.Fatal("second candidate should parse")
	}
	if got.A != 2 || got.B != "ok" {
		t.Fatalf("partial pollution from failed candidate: %+v", got)
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

// Callers pre-populate fields the JSON doesn't carry (judgeOne seeds Sample
// with the answers before parsing the verdict) — the winning candidate must
// MERGE into out, not replace it. Pinned because the first multi-candidate
// implementation did dst.Set(fresh) and wiped those fields.
func TestParseJSONReplyMergesIntoPrepopulated(t *testing.T) {
	type v struct {
		Answer  string `json:"answer_zz"` // never in the JSON
		Verdict string `json:"verdict"`
	}
	got := v{Answer: "pre-set"}
	if ok := parseJSONReply(`the ${var} case again: {"verdict":"keep"}`, &got); !ok {
		t.Fatal("should parse")
	}
	if got.Answer != "pre-set" || got.Verdict != "keep" {
		t.Fatalf("merge semantics broken: %+v", got)
	}
}
