package main

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"

	"github.com/BurntSushi/toml"
)

// Stream rules abort a generation the moment its *content* matches a pattern
// that means the reply has gone off the rails, then re-ask with a reminder. A
// small model emitting a literal `<tool_call>` block has committed to a dead
// turn, so every further token is waste; aborting costs only what was produced,
// and the retry appends to the same prefix, so the KV cache is untouched.
//
// Deliberately narrow: content only, not reasoning (a model may think about
// `<tool_call>`) and not tool arguments. The default set is two patterns, both
// chat-template control tokens leaking into visible text. Rules for
// behavioural drift belong in `.codehalter/rules.toml`, written from failures
// seen in the session log, never guessed: a misfiring rule costs two round
// trips on every turn.
type streamRule struct {
	// Name identifies the rule in logs and in the "rule fired" notice.
	Name string `toml:"name"`
	// Pattern is a Go regexp matched against a rolling window of the tail of
	// the content stream.
	Pattern string `toml:"pattern"`
	// Reminder is injected as a user message on the retry. Write it as a direct
	// instruction about what to do instead; the model never sees the pattern.
	Reminder string `toml:"reminder"`

	re *regexp.Regexp
}

// ruleWindowBytes bounds the text each pattern is run against. Rules are
// checked on every content delta, so matching the whole accumulated reply would
// be quadratic in the length of the message. A trailing window is enough for
// the control-token patterns these rules are for, but it does mean a pattern
// that needs to span more than this many bytes will not fire — keep patterns
// local.
const ruleWindowBytes = 512

// defaultStreamRules is the built-in set. See the streamRule doc for why it is
// this short.
var defaultStreamRules = []streamRule{
	{
		Name:    "tool_call_as_text",
		Pattern: `(?i)<\|?\s*/?\s*(tool_call|function_call|tool_use|tool_response)s?\s*\|?>`,
		Reminder: "STOP. You wrote a literal `<tool_call>` block as message text. That is not a tool call — it is plain prose, nothing ran, and no result will come back. " +
			"Emit the call through the tool-calling API (the same mechanism your earlier successful calls used), or, if you are finished, call respond. Do not describe a call in text.",
	},
	{
		Name:    "chat_template_token",
		Pattern: `<\|(im_start|im_end|channel|constrain|assistant|end_of_turn)\|>`,
		Reminder: "STOP. You emitted a chat-template control token (`<|im_start|>` or similar) inside your reply. Those belong to the prompt format, not to your message; the reply is being discarded. " +
			"Write the answer as plain text, or make a tool call.",
	},
}

// loadStreamRules returns the rules for a project: `.codehalter/rules.toml`
// when it exists, else the built-in defaults. The file REPLACES the defaults
// rather than extending them, so a project that wants the built-ins plus its
// own must restate them — matching how the other .codehalter/ files behave, and
// letting a project switch the mechanism off entirely with an empty file.
//
// A malformed file is not fatal: it logs and falls back to the defaults. Rules
// are a guard rail, and failing a session over a bad regex would be worse than
// running without the guard.
func loadStreamRules(cwd string) []streamRule {
	path := filepath.Join(cwd, sessionDir, "rules.toml")
	data, err := os.ReadFile(path)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.Warn("rules: unreadable rules.toml, using defaults", "path", path, "err", err)
		}
		return compileStreamRules(defaultStreamRules)
	}
	var f struct {
		Rule []streamRule `toml:"rule"`
	}
	if _, err := toml.Decode(string(data), &f); err != nil {
		slog.Warn("rules: malformed rules.toml, using defaults", "path", path, "err", err)
		return compileStreamRules(defaultStreamRules)
	}
	slog.Info("rules: loaded from file", "path", path, "count", len(f.Rule))
	return compileStreamRules(f.Rule)
}

// compileStreamRules compiles each pattern, dropping (with a log) any that
// doesn't compile or is missing a field. A half-configured rule is skipped
// rather than silently matching nothing.
func compileStreamRules(in []streamRule) []streamRule {
	out := make([]streamRule, 0, len(in))
	for _, r := range in {
		switch {
		case r.Pattern == "":
			slog.Warn("rules: skipping rule with no pattern", "name", r.Name)
			continue
		case r.Reminder == "":
			slog.Warn("rules: skipping rule with no reminder", "name", r.Name)
			continue
		}
		re, err := regexp.Compile(r.Pattern)
		if err != nil {
			slog.Warn("rules: skipping rule with an invalid pattern", "name", r.Name, "pattern", r.Pattern, "err", err)
			continue
		}
		if r.Name == "" {
			r.Name = r.Pattern
		}
		r.re = re
		out = append(out, r)
	}
	return out
}

// ruleMatcher feeds content deltas through the rule set, keeping only a
// trailing window so cost per delta is bounded by ruleWindowBytes rather than
// by the length of the reply so far. Not safe for concurrent use; one matcher
// belongs to one in-flight stream.
type ruleMatcher struct {
	rules []streamRule
	win   []byte
}

// feed appends a content delta and returns the first rule that now matches the
// trailing window, or nil. The caller aborts the stream on a non-nil return, so
// a rule never fires twice for one generation.
func (m *ruleMatcher) feed(delta string) *streamRule {
	if m == nil || len(m.rules) == 0 || delta == "" {
		return nil
	}
	m.win = append(m.win, delta...)
	if len(m.win) > ruleWindowBytes {
		m.win = m.win[len(m.win)-ruleWindowBytes:]
	}
	for i := range m.rules {
		if m.rules[i].re.Match(m.win) {
			return &m.rules[i]
		}
	}
	return nil
}

// streamRuleError is the error llmStream returns when it aborted a generation
// because a rule fired. It carries the reminder so the tool loop can re-ask
// with it appended, without knowing anything about the rule set.
type streamRuleError struct {
	Rule     string
	Reminder string
	// Matched is the text that tripped the rule, for the session log. Not shown
	// to the model: telling it which bytes matched invites it to work around
	// the pattern instead of fixing the behaviour.
	Matched string
}

func (e *streamRuleError) Error() string {
	return fmt.Sprintf("stream rule %q fired mid-generation: %s", e.Rule, truncate(e.Matched, 120))
}

// asStreamRule reports whether err is a rule abort, returning the detail.
func asStreamRule(err error) *streamRuleError {
	var e *streamRuleError
	if errors.As(err, &e) {
		return e
	}
	return nil
}
