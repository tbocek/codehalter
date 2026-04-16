package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/BurntSushi/toml"
)

const (
	maxPreviewLines = 100
	maxPreviewBytes = maxPreviewLines * 200

)

type FileCache struct {
	mu    sync.Mutex                `toml:"-"`
	Files map[string]FileCacheEntry `toml:"files"`
}

type FileCacheEntry struct {
	Hash    string `toml:"hash"`
	Size    int64  `toml:"size"`
	Summary string `toml:"summary"`
}

func cachePath(cwd string) string {
	return filepath.Join(cwd, sessionDir, "cache.toml")
}

func loadFileCache(cwd string) *FileCache {
	var c FileCache
	_, _ = toml.DecodeFile(cachePath(cwd), &c)
	if c.Files == nil {
		c.Files = make(map[string]FileCacheEntry)
	}
	return &c
}

func (c *FileCache) Save(cwd string) error {
	f, err := os.Create(cachePath(cwd))
	if err != nil {
		return err
	}
	defer f.Close()
	return toml.NewEncoder(f).Encode(c)
}

// updateFileCache checks all project files against the cache and returns
// which files need re-summarization. Updates hashes for changed files.
func updateFileCache(cwd string, cache *FileCache) []string {
	files := listProjectFiles(cwd)
	var stale []string

	current := make(map[string]bool)

	for _, rel := range files {
		current[rel] = true
		abs := filepath.Join(cwd, rel)
		hash := hashFileQuick(abs)

		var size int64
		if info, err := os.Stat(abs); err == nil {
			size = info.Size()
		}

		entry, exists := cache.Files[rel]
		if !exists || entry.Hash != hash {
			cache.Files[rel] = FileCacheEntry{Hash: hash, Size: size}
			stale = append(stale, rel)
		} else if entry.Size != size {
			entry.Size = size
			cache.Files[rel] = entry
		}
	}

	for rel := range cache.Files {
		if !current[rel] {
			delete(cache.Files, rel)
		}
	}

	return stale
}

func hashFileQuick(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()

	h := sha256.New()
	scanner := bufio.NewScanner(f)
	lines := 0
	for scanner.Scan() {
		h.Write(scanner.Bytes())
		h.Write([]byte{'\n'})
		lines++
		if lines >= maxPreviewLines {
			break
		}
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

var binaryExtensions = map[string]bool{
	".br": true, ".gz": true, ".zst": true, ".xz": true, ".bz2": true,
	".zip": true, ".tar": true, ".rar": true, ".7z": true,
	".png": true, ".jpg": true, ".jpeg": true, ".gif": true, ".webp": true, ".ico": true, ".svg": true,
	".woff": true, ".woff2": true, ".ttf": true, ".otf": true, ".eot": true,
	".pdf": true, ".doc": true, ".docx": true,
	".wasm": true, ".so": true, ".dylib": true, ".dll": true, ".exe": true,
	".o": true, ".a": true, ".pyc": true, ".class": true,
	".mp3": true, ".mp4": true, ".wav": true, ".ogg": true, ".webm": true,
	".bin": true, ".dat": true, ".db": true, ".sqlite": true,
}

func isBinaryFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	if binaryExtensions[ext] {
		return true
	}

	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()

	buf := make([]byte, 512)
	n, _ := f.Read(buf)
	for _, b := range buf[:n] {
		if b == 0 {
			return true
		}
	}
	return false
}

func readPreview(cwd, rel string) string {
	path := filepath.Join(cwd, rel)
	if isBinaryFile(path) {
		return ""
	}

	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()

	var b strings.Builder
	scanner := bufio.NewScanner(f)
	lines := 0
	for scanner.Scan() {
		b.WriteString(scanner.Text())
		b.WriteByte('\n')
		lines++
		if lines >= maxPreviewLines {
			break
		}
	}
	if b.Len() > maxPreviewBytes {
		return ""
	}
	return b.String()
}

// summarizeStaleFiles summarizes stale files in chunks and updates the cache.
func (a *agent) summarizeStaleFiles(ctx context.Context, cwd string, cache *FileCache, staleFiles []string, sid SessionId) error {
	conn := a.settings.SummaryLLM()
	if conn == nil {
		return fmt.Errorf("no 'summary' or 'thinking' LLM connection configured")
	}

	// Filter to text files only, mark binary.
	var toSummarize []string
	for _, rel := range staleFiles {
		preview := readPreview(cwd, rel)
		if preview == "" {
			if entry, ok := cache.Files[rel]; ok {
				entry.Summary = "(binary file)"
				cache.Files[rel] = entry
			}
			continue
		}
		toSummarize = append(toSummarize, rel)
	}

	if len(toSummarize) == 0 {
		return cache.Save(cwd)
	}

	total := len(toSummarize)
	var okCount, failedCount atomic.Int32
	var doneCount atomic.Int32

	parallel(total, func(i int) {
		rel := toSummarize[i]
		n := doneCount.Add(1)
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(fmt.Sprintf("Indexing %d/%d: %s\n\n", n, total, rel))))
		if a.summarizeFile(ctx, cwd, cache, conn, rel) == "ok" {
			okCount.Add(1)
		} else {
			failedCount.Add(1)
		}
	})

	fmt.Fprintf(os.Stderr, "filecache: done. ok=%d failed=%d total=%d\n", okCount.Load(), failedCount.Load(), total)
	return cache.Save(cwd)
}

// summarizeFile returns "ok" or "failed".
func (a *agent) summarizeFile(ctx context.Context, cwd string, cache *FileCache, conn *LLMConnection, rel string) string {
	preview := readPreview(cwd, rel)
	if preview == "" {
		return "failed"
	}

	summaryPrompt := loadSummaryPrompt(cwd)
	prompt := summaryPrompt + fmt.Sprintf("=== %s ===\n%s", rel, preview)
	messages := []llmMessage{{Role: "user", Content: prompt}}
	text, err := a.llmSimple(ctx, conn, messages)
	if err != nil {
		fmt.Fprintf(os.Stderr, "filecache: error for %s: %v\n", rel, err)
		return "failed"
	}

	// Take first sentence or first non-empty line, cap at 200 chars.
	text = strings.TrimSpace(text)
	summary := text
	if idx := strings.Index(summary, "."); idx > 0 {
		summary = summary[:idx+1]
	} else if idx := strings.Index(summary, "\n"); idx > 0 {
		summary = summary[:idx]
	}
	summary = strings.TrimSpace(summary)
	if len(summary) > 200 {
		summary = summary[:200]
	}

	if summary != "" {
		cache.mu.Lock()
		if entry, ok := cache.Files[rel]; ok {
			entry.Summary = summary
			cache.Files[rel] = entry
		}
		cache.mu.Unlock()
		return "ok"
	}

	return "failed"
}

func loadSummaryPrompt(cwd string) string {
	data, err := os.ReadFile(filepath.Join(cwd, ".codehalter", "SUMMARY.md"))
	if err != nil {
		return ""
	}
	return string(data) + "\n\n"
}

// buildProjectContext returns the project structure with summaries,
// suitable for prepending to a prompt. Not stored in history.
func buildProjectContext(cwd string, cache *FileCache) string {
	files := listProjectFiles(cwd)
	if len(files) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString("[Project structure — not part of conversation history]\n")
	for _, rel := range files {
		if entry, ok := cache.Files[rel]; ok && entry.Summary != "" {
			fmt.Fprintf(&b, "  %s — %s\n", rel, entry.Summary)
		} else {
			b.WriteString("  " + rel + "\n")
		}
	}
	return b.String()
}
