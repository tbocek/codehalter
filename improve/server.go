package improve

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// NewServer returns an http.Handler that serves the improvement API.
func NewServer(dataDir string) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("POST /improve", func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body", http.StatusBadRequest)
			return
		}

		var payload ImprovementPayload
		if err := json.Unmarshal(body, &payload); err != nil {
			http.Error(w, fmt.Sprintf("invalid JSON: %v", err), http.StatusBadRequest)
			return
		}

		var allNotes []string
		clientIP := r.Header.Get("X-Client-IP")
		model := r.Header.Get("X-Model")
		license := r.Header.Get("X-License")
		for i := range payload.Improvements {
			sanitized, notes := Sanitize(payload.Improvements[i])
			sanitized.Ip = clientIP
			sanitized.Model = model
			sanitized.License = license
			payload.Improvements[i] = sanitized
			allNotes = append(allNotes, notes...)
		}

		stored, _, err := Store(dataDir, payload, allNotes)
		if err != nil {
			http.Error(w, fmt.Sprintf("store error: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"stored":   stored,
			"redacted": allNotes,
		})
	})

	return mux
}
