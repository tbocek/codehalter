package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

// runSetup exits the process with status 1 on any error.
func runSetup() {
	fmt.Println("=== codehalter LLM Setup ===")
	fmt.Println()

	reader := bufio.NewReader(os.Stdin)
	// required, if non-empty, is printed before exiting on an empty answer.
	ask := func(prompt, required string) string {
		fmt.Print(prompt)
		line, err := reader.ReadString('\n')
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
			os.Exit(1)
		}
		line = strings.TrimSpace(line)
		if line == "" && required != "" {
			fmt.Fprintln(os.Stderr, required)
			os.Exit(1)
		}
		return line
	}
	server := ask("LLM server URL (e.g. http://localhost:8080): ", "Server URL is required.")
	apiKey := ask("API key (optional, press Enter to skip): ", "")
	model := ask("Model name (e.g. llama-3.1-8b): ", "Model name is required.")

	settings := Settings{
		LLM: []LLMConnection{{
			Server: server,
			APIKey: apiKey,
			Model:  model,
		}},
	}

	fmt.Println()
	fmt.Println("Testing connection...")

	// probeLLM already logs the failure reason to stderr via slog.
	result := probeLLM(context.Background(), &settings.LLM[0])
	switch {
	case !result.Reachable:
		c := &settings.LLM[0]
		fmt.Fprintf(os.Stderr, "Connection failed: neither %s nor %s answered with 200. Check the URL and the API key.\n", c.endpoint("/v1/models"), c.endpoint("/props"))
		os.Exit(1)
	case !result.ModelKnown:
		fmt.Println("Server is reachable, but model not found in /v1/models list.")
		fmt.Println("You may still be able to use it — double-check the model name.")
	default:
		fmt.Println("Connection successful!")
		if result.ModelLoaded {
			fmt.Println("Model is loaded and ready.")
		}
	}

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
