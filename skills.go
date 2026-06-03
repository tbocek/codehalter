package main

import (
	"bytes"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

//go:embed docs/SKILL-go.md
var skillGo string

//go:embed docs/SKILL-ts.md
var skillTS string

//go:embed docs/SKILL-js.md
var skillJS string

//go:embed docs/SKILL-java.md
var skillJava string

//go:embed docs/SKILL-bash.md
var skillBash string

//go:embed docs/SKILL-container.md
var skillContainer string

//go:embed docs/SKILL-makefile.md
var skillMakefile string

//go:embed docs/SKILL-justfile.md
var skillJustfile string

//go:embed docs/SKILL-alpine.md
var skillAlpine string

//go:embed docs/SKILL-arch.md
var skillArch string

//go:embed docs/SKILL-debian.md
var skillDebian string

//go:embed docs/SKILL-fedora.md
var skillFedora string

//go:embed docs/SKILL-ubuntu.md
var skillUbuntu string

// defaultSkills maps a per-stack key (language stack from detectStacks)
// to the embedded skill body. The container / per-OS / per-runner skills
// live in their own seed paths inside ensureSkills, not here, because
// they're not driven by detectStacks.
var defaultSkills = map[string]string{
	"go":   skillGo,
	"ts":   skillTS,
	"js":   skillJS,
	"java": skillJava,
	"bash": skillBash,
}

// osSkills maps an /etc/os-release ID (as returned by readOSInfo) to the
// embedded skill body. Only IDs we have a SKILL-<id>.md for are present;
// readOSInfo filters everything else to "" before lookup.
var osSkills = map[string]string{
	"alpine": skillAlpine,
	"arch":   skillArch,
	"debian": skillDebian,
	"fedora": skillFedora,
	"ubuntu": skillUbuntu,
}

// skillStateFile holds, per SKILL-*.md, the hex SHA-256 of the bytes
// codehalter last wrote. It is how syncSkill tells a copy it owns and may
// refresh from an updated embed apart from one the user has hand-edited and
// must leave alone. loadSkills/listSkills only ever read SKILL-*.md, so this
// sidecar never leaks into the system prompt or the capabilities banner.
const skillStateFile = ".skillstate.json"

func loadSkillState(dir string) map[string]string {
	data, err := os.ReadFile(filepath.Join(dir, skillStateFile))
	if err != nil {
		return map[string]string{}
	}
	var m map[string]string
	if json.Unmarshal(data, &m) != nil || m == nil {
		return map[string]string{}
	}
	return m
}

func skillHash(body string) string {
	sum := sha256.Sum256([]byte(body))
	return hex.EncodeToString(sum[:])
}

// syncSkill reconciles one SKILL-*.md against its desired embed body and
// refreshes ONLY copies codehalter wrote that the user hasn't since edited —
// detected by comparing the file's current hash against the provenance hash in
// `state`. A hand-edited copy (current hash differs from the recorded one) is
// left untouched. `authoritative` covers the one-time migration of a copy with
// no provenance yet: per-OS skills were historically rewritten every turn, so
// an untracked copy is certainly codehalter's and is adopted+refreshed; user-
// editable skills (stack/runner/container) are left alone when their provenance
// is unknown, since an untracked difference might be a genuine edit. Either way
// a copy already identical to the embed is recorded so future edits are caught.
// Returns whether the file changed on disk; `state` is mutated in place.
func syncSkill(dir, name, desired string, authoritative bool, state map[string]string) (bool, error) {
	if desired == "" {
		return false, nil
	}
	path := filepath.Join(dir, name)
	desiredHash := skillHash(desired)

	cur, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		if werr := os.WriteFile(path, []byte(desired), 0o644); werr != nil {
			return false, fmt.Errorf("seeding %s: %w", path, werr)
		}
		state[name] = desiredHash
		return true, nil
	}
	if err != nil {
		return false, fmt.Errorf("reading %s: %w", path, err)
	}

	curHash := skillHash(string(cur))
	if curHash == desiredHash {
		state[name] = desiredHash // already current — record provenance so later edits are detected
		return false, nil
	}

	recorded, tracked := state[name]
	unedited := tracked && recorded == curHash
	if unedited || (!tracked && authoritative) {
		if werr := os.WriteFile(path, []byte(desired), 0o644); werr != nil {
			return false, fmt.Errorf("refreshing %s: %w", path, werr)
		}
		state[name] = desiredHash
		return true, nil
	}
	// Hand-edited (or untracked + non-authoritative): leave it, keep no provenance.
	return false, nil
}

// ensureSkills reconciles .codehalter/SKILL-*.md with the embeds based on what
// is PRESENT in the project tree (justfile / Makefile / language stacks) and in
// the container (/etc/os-release ID), NOT on what tooling is installed on PATH.
// This is load-bearing: the LLM needs SKILL-justfile.md loaded BEFORE it
// installs `just` for the user, otherwise the fix-dispatch turn has no idea what
// justfile syntax looks like.
//
// Every skill is refreshed from its embed only while still un-edited (see
// syncSkill) — so codehalter's own skill updates reach existing projects, but a
// user's hand-edits to any SKILL-*.md survive. Per-OS handling additionally
// prunes every SKILL-<other-os>.md unconditionally, because codehalter supports
// exactly one OS per session and a stale skill from a prior run on a different
// host would otherwise keep getting concatenated into every system prompt.
//
// It only makes the on-disk set current; the caller (checkEnv) rebuilds the
// system prompt every turn and diffs it, so ensureSkills no longer needs to
// report whether anything changed.
func ensureSkills(cwd string, stacks []string, osi osInfo) error {
	dir := filepath.Join(cwd, ".codehalter")
	state := loadSkillState(dir)
	sync := func(name, body string, authoritative bool) error {
		_, err := syncSkill(dir, name, body, authoritative, state)
		return err
	}
	exists := func(names ...string) bool {
		for _, n := range names {
			if _, err := os.Stat(filepath.Join(cwd, n)); err == nil {
				return true
			}
		}
		return false
	}

	// Always-on container skill.
	if err := sync("SKILL-container.md", skillContainer, false); err != nil {
		return err
	}
	// Per-stack skills (driven by files in the tree).
	for _, stack := range stacks {
		if body, ok := defaultSkills[stack]; ok {
			if err := sync("SKILL-"+stack+".md", body, false); err != nil {
				return err
			}
		}
	}
	// Per-runner skills.
	if exists("justfile", "Justfile", ".justfile") {
		if err := sync("SKILL-justfile.md", skillJustfile, false); err != nil {
			return err
		}
	}
	if exists("Makefile", "makefile", "GNUmakefile") {
		if err := sync("SKILL-makefile.md", skillMakefile, false); err != nil {
			return err
		}
	}
	// Per-OS skill: prune the other-OS copies, then sync the active one. It's
	// authoritative so an untracked copy from before provenance existed (or from
	// the old always-overwrite behaviour) still adopts the current embed.
	if osi.ID != "" {
		for other := range osSkills {
			if other == osi.ID {
				continue
			}
			name := "SKILL-" + other + ".md"
			path := filepath.Join(dir, name)
			if _, err := os.Stat(path); err != nil {
				continue
			}
			if err := os.Remove(path); err != nil {
				return fmt.Errorf("pruning stale %s: %w", path, err)
			}
			delete(state, name)
		}
		if body, ok := osSkills[osi.ID]; ok {
			rendered := renderOSSkill(body, osi.Fields)
			if err := sync("SKILL-"+osi.ID+".md", rendered, true); err != nil {
				return err
			}
		}
	}

	// Persist provenance only when it actually moved, so a steady-state turn
	// neither rewrites the sidecar nor churns the .codehalter git snapshot.
	newState, _ := json.MarshalIndent(state, "", "  ")
	curState, _ := os.ReadFile(filepath.Join(dir, skillStateFile))
	if !bytes.Equal(curState, newState) && (len(state) > 0 || len(curState) > 0) {
		if err := os.WriteFile(filepath.Join(dir, skillStateFile), newState, 0o644); err != nil {
			return fmt.Errorf("saving skill state: %w", err)
		}
	}
	return nil
}

// renderOSSkill substitutes {{KEY}} placeholders in the per-OS skill body
// with the matching /etc/os-release field. Unknown placeholders are
// replaced with the empty string so the LLM doesn't see literal `{{X}}`
// when a field is missing on a non-standard image.
func renderOSSkill(body string, fields map[string]string) string {
	var b strings.Builder
	b.Grow(len(body))
	for {
		i := strings.Index(body, "{{")
		if i < 0 {
			b.WriteString(body)
			return b.String()
		}
		j := strings.Index(body[i+2:], "}}")
		if j < 0 {
			b.WriteString(body)
			return b.String()
		}
		b.WriteString(body[:i])
		key := body[i+2 : i+2+j]
		b.WriteString(fields[key])
		body = body[i+2+j+2:]
	}
}

// loadSkills concatenates every SKILL-*.md present in .codehalter/. Detection
// (detectStacks) decides which to seed initially, but loading honors whatever
// the user actually has on disk — drop a SKILL-rust.md in there manually and
// it gets picked up; delete one and it stops loading. checkEnv rebuilds the
// system prompt every turn and assigns only on a byte diff, so a skill added or
// removed mid-session takes effect on the next turn while an unchanged set
// keeps the cache prefix stable.
func loadSkills(cwd string) string {
	dir := filepath.Join(cwd, ".codehalter")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		n := e.Name()
		if strings.HasPrefix(n, "SKILL-") && strings.HasSuffix(n, ".md") {
			names = append(names, n)
		}
	}
	if len(names) == 0 {
		return ""
	}
	sort.Strings(names) // deterministic order → stable cache prefix
	var b strings.Builder
	for _, n := range names {
		data, err := os.ReadFile(filepath.Join(dir, n))
		if err != nil {
			continue
		}
		b.Write(data)
		if !strings.HasSuffix(string(data), "\n") {
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}
	return b.String()
}
