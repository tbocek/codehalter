package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestLocateSpan(t *testing.T) {
	file := strings.Split(strings.TrimPrefix(`
# Go skill
## Errors
- Errors = values. Check: if err != nil.
- NO panic for control flow.

## Idioms
- := declare, = reassign.
User=non-root dev, sudo NOPASSWD. Write ops need sudo. Read probes no sudo.
`, "\n"), "\n")

	cases := []struct {
		name, source       string
		wantStart, wantEnd int
		wantFrag           bool
	}{
		{"single bullet", "- NO panic for control flow.", 4, 4, false},
		{"first bullet", "- Errors = values. Check: if err != nil.", 3, 3, false},
		{"no such line", "- Idioms line? no", 0, 0, false},
		{"multi contiguous", "## Idioms\n- := declare, = reassign.", 6, 7, false},
		{"absent", "this is not present", 0, 0, false},
		// Sub-line fragments: one sentence of a multi-sentence line.
		{"fragment middle sentence", "Write ops need sudo.", 8, 8, true},
		{"fragment first sentence", "User=non-root dev, sudo NOPASSWD.", 8, 8, true},
		{"fragment without bullet marker", "NO panic for control flow.", 4, 4, true},
		// "sudo" appears in several lines → ambiguous, must not match.
		{"ambiguous fragment", "sudo", 0, 0, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gs, ge, frag := locateSpan(file, c.source)
			if gs != c.wantStart || ge != c.wantEnd || frag != c.wantFrag {
				t.Fatalf("locateSpan(%q) = (%d,%d,%v), want (%d,%d,%v)", c.source, gs, ge, frag, c.wantStart, c.wantEnd, c.wantFrag)
			}
		})
	}
}

func TestLocateSpanSkipsBlankLines(t *testing.T) {
	file := []string{"- first", "", "- second"}
	// A source quoting both bullets with the blank between should still match.
	gs, ge, frag := locateSpan(file, "- first\n- second")
	if gs != 1 || ge != 3 || frag {
		t.Fatalf("got (%d,%d,%v), want (1,3,false)", gs, ge, frag)
	}
}

func TestClaimCacheInvalidation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "go.json")
	claims := []Claim{{ID: "go#abc", Skill: "go", Text: "t", Source: "- s", StartLine: 1, EndLine: 1}}
	if err := writeClaimCache(path, "hashA", claims); err != nil {
		t.Fatal(err)
	}
	// Same content hash → cache hit.
	if got, ok := readClaimCache(path, "hashA"); !ok || len(got) != 1 || got[0].ID != "go#abc" {
		t.Fatalf("expected hit, got ok=%v claims=%v", ok, got)
	}
	// Changed content hash → miss, forcing re-segmentation.
	if _, ok := readClaimCache(path, "hashB"); ok {
		t.Fatal("stale cache must miss when content hash differs")
	}
}

func TestHashOfStable(t *testing.T) {
	// Distinct []byte allocations with equal content must hash equal —
	// separate variables also keep staticcheck SA4000 quiet.
	a, b := hashOf([]byte("x")), hashOf([]byte("x"))
	if a != b {
		t.Fatal("hash not deterministic")
	}
	if a == hashOf([]byte("y")) {
		t.Fatal("distinct inputs collided")
	}
}

func TestNonBlankTrimmed(t *testing.T) {
	got := nonBlankTrimmed([]string{"  a  ", "", "   ", "b"})
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("nonBlankTrimmed = %v", got)
	}
}

// segJudge serves a fixed segmentation/repair reply (as the model's JSON
// content) and counts calls.
func segJudge(t *testing.T, replyJSON string) (ModelSpec, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, replyJSON))))
	}))
	t.Cleanup(s.Close)
	return ModelSpec{Name: "j", Server: s.URL, Model: "j"}, &calls
}

const segSkill = "# Skill\n- run git status before committing.\n- push only when explicitly asked.\n"

// A first-pass claim whose source is paraphrased (unlocatable) is recovered by
// the repair pass on judge B, which re-quotes the exact verbatim span.
func TestSegmentSkillRepairsUnlocatableSource(t *testing.T) {
	root := t.TempDir()
	src := filepath.Join(root, "SKILL-go.md")
	if err := os.WriteFile(src, []byte(segSkill), 0o644); err != nil {
		t.Fatal(err)
	}
	// Pass 1: one locatable claim + one paraphrased ("push when the user asks"
	// is not in the text, which says "push only when explicitly asked").
	judgeA, _ := segJudge(t, `{"claims":[{"text":"check status first","source":"- run git status before committing."},{"text":"push only when asked","source":"push when the user asks"}]}`)
	judgeB, bCalls := segJudge(t, `{"claims":[{"text":"push only when asked","source":"- push only when explicitly asked."}]}`)

	claims, err := segmentSkill(context.Background(), judgeA, judgeB, "go", src, filepath.Join(root, "seg"))
	if err != nil {
		t.Fatal(err)
	}
	if len(claims) != 2 {
		t.Fatalf("claims = %d, want 2 (1 located + 1 repaired)", len(claims))
	}
	if bCalls.Load() != 1 {
		t.Fatalf("repair judge called %d times, want 1", bCalls.Load())
	}
	found := false
	for _, c := range claims {
		if c.StartLine == 0 {
			t.Fatalf("claim not located: %+v", c)
		}
		if c.Text == "push only when asked" && c.Source == "- push only when explicitly asked." {
			found = true
		}
	}
	if !found {
		t.Fatalf("repaired claim missing/wrong: %+v", claims)
	}
}

// When the repair pass gives up ("" source), the unlocatable claim stays
// dropped but the located one survives.
func TestSegmentSkillRepairGivesUp(t *testing.T) {
	root := t.TempDir()
	src := filepath.Join(root, "SKILL-go.md")
	if err := os.WriteFile(src, []byte(segSkill), 0o644); err != nil {
		t.Fatal(err)
	}
	judgeA, _ := segJudge(t, `{"claims":[{"text":"check status first","source":"- run git status before committing."},{"text":"push only when asked","source":"push when the user asks"}]}`)
	judgeB, bCalls := segJudge(t, `{"claims":[{"text":"push only when asked","source":""}]}`)

	claims, err := segmentSkill(context.Background(), judgeA, judgeB, "go", src, filepath.Join(root, "seg"))
	if err != nil {
		t.Fatal(err)
	}
	if len(claims) != 1 || claims[0].Text != "check status first" {
		t.Fatalf("claims = %+v, want only the located one", claims)
	}
	if bCalls.Load() != 1 {
		t.Fatalf("repair judge called %d times, want 1", bCalls.Load())
	}
}

// A claim whose source is a {{cmd:}} templating line is a FACT: never probed,
// filtered at build time and again when loading a pre-filter cache.
func TestSegmentSkillDropsCmdFactClaims(t *testing.T) {
	root := t.TempDir()
	src := filepath.Join(root, "SKILL-go.md")
	content := "# Skill\n- Base: Fedora ({{cmd:. /etc/os-release && echo x}}).\n- run git status before committing.\n"
	if err := os.WriteFile(src, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	// The segmenter (wrongly) claims the fact line alongside a real claim.
	judgeA, _ := segJudge(t, `{"claims":[{"text":"verify the base environment","source":"- Base: Fedora ({{cmd:. /etc/os-release && echo x}})."},{"text":"check status first","source":"- run git status before committing."}]}`)
	judgeB, bCalls := segJudge(t, `{"claims":[]}`)

	cacheDir := filepath.Join(root, "seg")
	claims, err := segmentSkill(context.Background(), judgeA, judgeB, "go", src, cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(claims) != 1 || claims[0].Text != "check status first" {
		t.Fatalf("fact claim not filtered: %+v", claims)
	}
	if bCalls.Load() != 0 {
		t.Fatalf("fact claims must be skipped, not repaired (repair called %dx)", bCalls.Load())
	}

	// Simulate a pre-filter cache that still carries the fact claim: the read
	// path must retire it without a cache bust.
	hash := hashOf([]byte(content + segmentPrompt + repairPrompt))
	stale := append([]Claim{{
		ID: "go#facts", Skill: "go", Text: "verify the base environment",
		Source: "- Base: Fedora ({{cmd:. /etc/os-release && echo x}}).", StartLine: 2, EndLine: 2,
	}}, claims...)
	if err := writeClaimCache(filepath.Join(cacheDir, "go.json"), hash, stale); err != nil {
		t.Fatal(err)
	}
	got, err := segmentSkill(context.Background(), judgeA, judgeB, "go", src, cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Text != "check status first" {
		t.Fatalf("cached fact claim not retired on read: %+v", got)
	}
}
