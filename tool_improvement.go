package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// submitImprovementToolName is the terminal tool for an /improve execute pass.
// The model only PROPOSES improvements (structured); codehalter drives the rest
// in code: present each change, ask the user Apply/Skip, apply accepted edits to
// the .codehalter/ prompt file, then ask whether to submit the applied ones.
// Moving the apply/submit loop out of the (weak) model is what makes /improve
// reliably ask before changing anything, instead of analysing and bailing with a
// prose respond.
const submitImprovementToolName = "submit_improvement"

type improvementEntry struct {
	Title     string `json:"title"`
	File      string `json:"file"`
	Type      string `json:"type"`
	Original  string `json:"original"`
	New       string `json:"new"`
	Reasoning string `json:"reasoning"`
	// Model is stamped by codehalter (the LLM only writes the change fields, and
	// the backend can't know which model ran). ip is left to the backend, which
	// sees the request source; license rides the X-License header.
	Model string `json:"model,omitempty"`
}

type improvementPayload struct {
	Improvements []improvementEntry `json:"improvements"`
}

// openSourceLicenses matches common open-source license identifiers in license
// files. The pattern is case-insensitive and matches the license name as a
// word boundary.
var openSourceLicenses = regexp.MustCompile(`(?i)\b(MIT|BSD|Apache-2\.0|Apache 2\.0|GPL|GNU General Public License|LGPL|GNU Lesser General Public License|AGPL|GNU Affero General Public License|MPL|Mozilla Public License|ISC|Unlicense|WTFPL)\b`)

// checkLicense reads LICENSE/LICENCE in projectDir and returns the detected
// open-source license name, or an error if no open-source license is found.
func checkLicense(projectDir string) (string, error) {
	for _, name := range []string{"LICENSE", "LICENCE"} {
		path := filepath.Join(projectDir, name)
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		match := openSourceLicenses.FindString(string(data))
		if match != "" {
			return match, nil
		}
	}
	return "", fmt.Errorf("no open-source license found in project root")
}

// applyImprovement applies one structured change to its .codehalter/ prompt
// file: replace/remove swap out the `original` text, add appends `new` (after
// `original` when given, else at the end). The user already approved this entry
// via the Apply card, so the write is direct. Returns a per-entry error the
// caller surfaces without aborting the rest.
func applyImprovement(cwd string, e improvementEntry) error {
	name := strings.TrimSpace(e.File)
	if name == "" {
		return fmt.Errorf("no file given")
	}
	// Improvements target the prompt files under .codehalter/ by bare filename;
	// reject anything that escapes that directory.
	if strings.ContainsAny(name, `/\`) || name == ".." {
		return fmt.Errorf("file %q must be a bare .codehalter/ prompt filename", name)
	}
	path := filepath.Join(cwd, ".codehalter", name)
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read %s: %w", name, err)
	}
	content := string(data)
	switch strings.ToLower(strings.TrimSpace(e.Type)) {
	case "replace":
		if e.Original == "" {
			return fmt.Errorf("replace needs `original` text")
		}
		if !strings.Contains(content, e.Original) {
			return fmt.Errorf("`original` text not found in %s", name)
		}
		content = strings.Replace(content, e.Original, e.New, 1)
	case "remove":
		if e.Original == "" {
			return fmt.Errorf("remove needs `original` text")
		}
		if !strings.Contains(content, e.Original) {
			return fmt.Errorf("`original` text not found in %s", name)
		}
		content = strings.Replace(content, e.Original, "", 1)
	case "add":
		if e.New == "" {
			return fmt.Errorf("add needs `new` text")
		}
		if e.Original != "" && strings.Contains(content, e.Original) {
			content = strings.Replace(content, e.Original, e.Original+"\n\n"+e.New, 1)
		} else {
			content = strings.TrimRight(content, "\n") + "\n\n" + e.New + "\n"
		}
	default:
		return fmt.Errorf("unknown type %q (want add, replace, or remove)", e.Type)
	}
	return os.WriteFile(path, []byte(content), 0o644)
}

// submitImprovements POSTs the applied entries to the feedback endpoint, gated
// on the project carrying an open-source license. Returns a one-line result
// (success, or the reason it didn't submit) for the user-facing summary.
func (a *agent) submitImprovements(ctx context.Context, cwd, endpoint string, entries []improvementEntry) string {
	if endpoint == "" {
		endpoint = "https://ai.jos.li/improve"
	}
	license, err := checkLicense(cwd)
	if err != nil {
		return fmt.Sprintf("not submitted: %v (only open-source projects are eligible)", err)
	}
	// Stamp the model that produced these onto every entry — the LLM writes only
	// the change fields, and the backend has no other way to know which model ran.
	model := ""
	if c := a.settings.MainLLM("execute"); c != nil {
		model = c.Model
	}
	for i := range entries {
		entries[i].Model = model
	}
	body, err := json.Marshal(improvementPayload{Improvements: entries})
	if err != nil {
		return fmt.Sprintf("not submitted: marshalling payload: %v", err)
	}
	// No auth token: the endpoint takes anonymous submissions. The gate is the
	// user's Submit click, the open-source license, and "no secrets in the change
	// text" — not an API key.
	client := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(string(body)))
	if err != nil {
		return fmt.Sprintf("not submitted: creating request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-License", license)
	resp, err := client.Do(req)
	if err != nil {
		slog.Debug("submit_improvement: request failed", "err", err)
		return fmt.Sprintf("not submitted: request failed: %v", err)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return fmt.Sprintf("submitted %d improvement(s) to the feedback API", len(entries))
	}
	return fmt.Sprintf("not submitted: HTTP %d: %s", resp.StatusCode, string(respBody))
}

// renderImprovement is the card body shown before each Apply/Skip prompt, so the
// user sees exactly what they're approving.
func renderImprovement(i, n int, e improvementEntry) string {
	var b strings.Builder
	fmt.Fprintf(&b, "### Improvement %d/%d: %s\n`.codehalter/%s` · %s\n\n", i, n, e.Title, e.File, e.Type)
	if e.Reasoning != "" {
		fmt.Fprintf(&b, "%s\n\n", e.Reasoning)
	}
	switch strings.ToLower(strings.TrimSpace(e.Type)) {
	case "add":
		fmt.Fprintf(&b, "```\n+ %s\n```\n", truncate(strings.TrimSpace(e.New), 800))
	case "remove":
		fmt.Fprintf(&b, "```\n- %s\n```\n", truncate(strings.TrimSpace(e.Original), 800))
	default: // replace
		fmt.Fprintf(&b, "```\n- %s\n+ %s\n```\n", truncate(strings.TrimSpace(e.Original), 600), truncate(strings.TrimSpace(e.New), 600))
	}
	return b.String()
}

func init() {
	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        submitImprovementToolName,
			"description": "Hand off ALL your proposed prompt improvements in ONE structured call. codehalter then drives the rest itself: it shows the user each change, asks Apply/Skip, applies the accepted ones to the .codehalter/ prompt file, and (for open-source projects) asks whether to submit the applied changes to the feedback API. Do NOT ask_user or edit_file yourself; this single call IS the apply step. `improvements` is a JSON array; each object: title (string), file (bare .codehalter prompt filename, e.g. \"PLAN.md\"), type (add|replace|remove), original (the exact current text to match, for replace/remove), new (the added/replacement text), reasoning (string).",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"improvements"},
				"properties": map[string]any{
					"endpoint": map[string]any{
						"type":        "string",
						"description": "Feedback API endpoint. Default: https://ai.jos.li/improve",
					},
					"improvements": map[string]any{
						"type":        "string",
						"description": "JSON array of improvement objects (see above).",
					},
				},
			},
		},
	}, Execute: improvementExecute})
}

// improvementExecute is the code-driven /improve apply loop: the model proposes
// (structured), codehalter asks Apply/Skip per change, applies the accepted
// edits, then asks whether to submit. Keeping the loop here (not in the prompt)
// stops a weak model from skipping the approval.
func improvementExecute(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	args := parseArgs(rawArgs)
	improvementsJSON := args["improvements"]
	if improvementsJSON == "" {
		return "error: improvements is required (a JSON array of changes)", true
	}
	sess := a.getSession(sid)
	if sess == nil {
		return "error: no session", true
	}
	var improvements []improvementEntry
	if err := json.Unmarshal([]byte(improvementsJSON), &improvements); err != nil {
		return fmt.Sprintf("error: invalid improvements JSON: %v", err), true
	}
	if len(improvements) == 0 {
		return "No improvements proposed — nothing to apply.", false
	}
	// The apply loop has side effects (file edits + a POST), so it runs at most ONCE
	// per /improve turn: a duplicate call (two in one batch, or a replan re-entry)
	// no-ops here instead of re-applying. Claim the delivery only now that the call
	// is well-formed, so a malformed first call above can still be retried.
	if !sess.improveDelivered.CompareAndSwap(false, true) {
		return "Improvements were already handled this /improve run; ignoring this repeat submit_improvement call.", false
	}
	// Enforce the top-N cap in code (the prompt asks for the top 3, but a chatty
	// model may send more); report what was dropped rather than silently trimming.
	dropped := 0
	if len(improvements) > improveAskCap {
		dropped = len(improvements) - improveAskCap
		improvements = improvements[:improveAskCap]
	}

	var applied []improvementEntry
	var summary strings.Builder
	if dropped > 0 {
		fmt.Fprintf(&summary, "(%d further proposal(s) beyond the top %d were not shown)\n", dropped, improveAskCap)
	}
	for i, e := range improvements {
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "\n" + renderImprovement(i+1, len(improvements), e) + "\n"}})
		ok, tcId, err := a.askYesNoWithCard(ctx, sid, fmt.Sprintf("Apply %d/%d: %s", i+1, len(improvements), e.Title), "edit", "Apply", "Skip")
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return fmt.Sprintf("Stopped: the Apply prompt failed: %v\n\n%s", err, summary.String()), false
		}
		if !ok {
			a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Skipped")})
			fmt.Fprintf(&summary, "skipped: %s\n", e.Title)
			continue
		}
		if err := applyImprovement(sess.Cwd, e); err != nil {
			a.FailToolCall(ctx, sid, tcId, "apply failed: "+err.Error())
			fmt.Fprintf(&summary, "could not apply %q: %v\n", e.Title, err)
			continue
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Applied to .codehalter/" + e.File)})
		applied = append(applied, e)
		fmt.Fprintf(&summary, "applied: %s (.codehalter/%s)\n", e.Title, e.File)
	}

	if len(applied) == 0 {
		return "No improvements applied (all skipped or failed):\n\n" + summary.String(), false
	}

	// Submission only makes sense for open-source projects; when we already know
	// there's no license (cached at /improve start), skip the ask entirely.
	if sess.improveNoLicense.Load() {
		return fmt.Sprintf("✅ Applied %d improvement(s). Not submitted (no open-source license).\n\n%s", len(applied), summary.String()), false
	}
	ok, tcId, err := a.askYesNoWithCard(ctx, sid, fmt.Sprintf("Submit %d applied improvement(s) to the feedback API?", len(applied)), "think", "Submit", "Keep local")
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return fmt.Sprintf("✅ Applied %d improvement(s); the submit prompt failed: %v\n\n%s", len(applied), err, summary.String()), false
	}
	if !ok {
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Kept local")})
		return fmt.Sprintf("✅ Applied %d improvement(s); kept local (not submitted).\n\n%s", len(applied), summary.String()), false
	}
	result := a.submitImprovements(ctx, sess.Cwd, args["endpoint"], applied)
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(result)})
	return fmt.Sprintf("✅ Applied %d improvement(s); %s.\n\n%s", len(applied), result, summary.String()), false
}
