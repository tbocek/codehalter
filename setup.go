package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

// runSetup runs an interactive terminal flow that configures the LLM
// connection and writes ~/.config/codehalter/settings.toml.
// It exits the process on completion (os.Exit(0) on success, os.Exit(1) on error).
func runSetup() {
	fmt.Println("=== codehalter LLM Setup ===")
	fmt.Println()

	reader := bufio.NewReader(os.Stdin)

	// Prompt for server URL
	fmt.Print("LLM server URL (e.g. http://localhost:8080): ")
	server, err := reader.ReadString('\n')
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		os.Exit(1)
	}
	server = strings.TrimSpace(server)
	if server == "" {
		fmt.Fprintln(os.Stderr, "Server URL is required.")
		os.Exit(1)
	}

	// Prompt for API key (optional)
	fmt.Print("API key (optional, press Enter to skip): ")
	apiKey, err := reader.ReadString('\n')
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		os.Exit(1)
	}
	apiKey = strings.TrimSpace(apiKey)

	// Prompt for model name
	fmt.Print("Model name (e.g. llama-3.1-8b): ")
	model, err := reader.ReadString('\n')
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		os.Exit(1)
	}
	model = strings.TrimSpace(model)
	if model == "" {
		fmt.Fprintln(os.Stderr, "Model name is required.")
		os.Exit(1)
	}

	// Build a temporary settings object and validate the connection
	settings := Settings{
		LLM: []LLMConnection{{
			Server: server,
			Model:  model,
		}},
	}
	if apiKey != "" {
		settings.LLM[0].APIKey = apiKey
	}

	fmt.Println()
	fmt.Println("Testing connection...")

	agent := &agent{sessions: make(map[string]*Session), mode: "Interactive"}
	agent.cfgMu.Lock()
	agent.settings = settings
	agent.buildConnSems()
	agent.cfgMu.Unlock()

	conn := settings.MainLLM("execute")
	if conn == nil {
		fmt.Fprintln(os.Stderr, "Failed to build LLM connection.")
		os.Exit(1)
	}

	result := agent.probeLLM(context.Background(), conn)
	if !result.ModelKnown {
		// Try a simple HTTP check to distinguish network vs model issues
		endpoint := conn.endpoint("/v1/models")
		req, _ := http.NewRequest("GET", endpoint, nil)
		if apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+apiKey)
		}
		resp, httpErr := http.DefaultClient.Do(req)
		if httpErr != nil {
			fmt.Fprintf(os.Stderr, "Connection failed: %v\n", httpErr)
			os.Exit(1)
		}
		resp.Body.Close()
		if resp.StatusCode != 200 {
			fmt.Fprintf(os.Stderr, "Connection failed: server returned HTTP %d\n", resp.StatusCode)
			os.Exit(1)
		}
		// Server is reachable but model not found — still useful, warn
		fmt.Println("Server is reachable, but model not found in /v1/models list.")
		fmt.Println("You may still be able to use it — double-check the model name.")
	} else {
		fmt.Println("Connection successful!")
		if result.ModelLoaded {
			fmt.Println("Model is loaded and ready.")
		}
	}

	// Write settings.toml to ~/.config/codehalter/settings.toml
	home, err := os.UserHomeDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not determine home directory: %v\n", err)
		os.Exit(1)
	}
	configDir := filepath.Join(home, ".config", "codehalter")
	if err := os.MkdirAll(configDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Could not create config directory: %v\n", err)
		os.Exit(1)
	}
	configPath := filepath.Join(configDir, "settings.toml")

	// Backup existing settings.toml before overwriting
	if _, err := os.Stat(configPath); err == nil {
		old, readErr := os.ReadFile(configPath)
		if readErr == nil {
			hash := sha256.Sum256(old)
			shaPrefix := hex.EncodeToString(hash[:])[:8]
			ts := time.Now().Format("20060102150405")
			backupPath := filepath.Join(configDir, fmt.Sprintf("settings.backup-%s-%s", ts, shaPrefix))
			if writeErr := os.WriteFile(backupPath, old, 0o644); writeErr != nil {
				fmt.Fprintf(os.Stderr, "Warning: could not create backup: %v\n", writeErr)
			}
		}
	}

	f, err := os.Create(configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not create settings.toml: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	enc := toml.NewEncoder(f)
	if err := enc.Encode(settings); err != nil {
		fmt.Fprintf(os.Stderr, "Could not write settings.toml: %v\n", err)
		os.Exit(1)
	}

	fmt.Println()
	fmt.Printf("Settings written to %s\n", configPath)
	fmt.Println("Setup complete! You can now start codehalter.")
}
