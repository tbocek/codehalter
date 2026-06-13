package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
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
}

type improvementPayload struct {
	Improvements []improvementEntry `json:"improvements"`
}

func init() {
	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "submit_improvement",
			"description": "Submit prompt improvement changes to the feedback API. The improvements parameter is a JSON string with an array of objects, each having: title (string), file (string), type (string: remove|add|replace), original (string), new (string), reasoning (string).",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"endpoint", "api_key", "improvements"},
				"properties": map[string]any{
					"endpoint": map[string]any{
						"type":        "string",
						"description": "The API endpoint URL. Default: https://api.codehalter.dev/v1/improvements",
					},
					"api_key": map[string]any{
						"type":        "string",
						"description": "Bearer token for authentication.",
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
		apiKey := args["api_key"]
		improvementsJSON := args["improvements"]

		if endpoint == "" {
			endpoint = "https://api.codehalter.dev/v1/improvements"
		}
		if apiKey == "" {
			return "error: api_key is required", false
		}
		if improvementsJSON == "" {
			return "error: improvements is required", false
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
		req.Header.Set("Authorization", "Bearer "+apiKey)

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
