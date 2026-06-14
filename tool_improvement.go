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

func init() {
	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "submit_improvement",
			"description": "Submit prompt improvement changes to the feedback API. The improvements parameter is a JSON string with an array of objects, each having: title (string), file (string), type (string: remove|add|replace), original (string), new (string), reasoning (string).",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"improvements"},
				"properties": map[string]any{
					"endpoint": map[string]any{
						"type":        "string",
						"description": "The API endpoint URL. Default: https://ai.jos.li/improve",
					},
					"improvements": map[string]any{
						"type":        "string",
						"description": "JSON string with the improvements array.",
					},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		endpoint := args["endpoint"]
		improvementsJSON := args["improvements"]

		if endpoint == "" {
			endpoint = "https://ai.jos.li/improve"
		}
		// No auth token: the endpoint takes anonymous submissions. The submission
		// gate is the user's Yes/No, the open-source license, and "no secrets in the
		// change text" — not an API key.
		if improvementsJSON == "" {
			return "error: improvements is required", false
		}

		// Check that the project has an open-source license before submitting.
		sess := a.getSession(sid)
		if sess == nil {
			return "error: no session", false
		}
		license, err := checkLicense(sess.Cwd)
		if err != nil {
			return fmt.Sprintf("error: cannot submit improvements — %v. Only projects with an open-source license (MIT, BSD, Apache, GPL, etc.) are eligible.", err), false
		}

		// The model passes a bare JSON array (per the tool description and
		// TEMPLATE-improve.md), which we wrap in {"improvements":[...]} for the API.
		var improvements []improvementEntry
		if err := json.Unmarshal([]byte(improvementsJSON), &improvements); err != nil {
			return fmt.Sprintf("error: invalid improvements JSON: %v", err), false
		}
		if len(improvements) == 0 {
			return "error: improvements array is empty", false
		}

		// Stamp the model that produced these onto every entry — the LLM writes
		// only the change fields, and the backend has no other way to know which
		// model ran. (ip is filled backend-side from the request source.)
		model := ""
		if c := a.settings.MainLLM("execute"); c != nil {
			model = c.Model
		}
		for i := range improvements {
			improvements[i].Model = model
		}

		body, err := json.Marshal(improvementPayload{Improvements: improvements})
		if err != nil {
			return fmt.Sprintf("error: marshalling payload: %v", err), false
		}

		client := &http.Client{Timeout: 30 * time.Second}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(string(body)))
		if err != nil {
			return fmt.Sprintf("error: creating request: %v", err), false
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-License", license)

		resp, err := client.Do(req)
		if err != nil {
			slog.Debug("submit_improvement: request failed", "err", err)
			return fmt.Sprintf("error: request failed: %v", err), false
		}
		defer resp.Body.Close()

		respBody, _ := io.ReadAll(resp.Body)

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return fmt.Sprintf("Submitted %d improvement(s) successfully", len(improvements)), false
		}

		return fmt.Sprintf("error: HTTP %d: %s", resp.StatusCode, string(respBody)), false
	}})
}
