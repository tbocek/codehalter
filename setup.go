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
		fmt.Println("Connection successful, but the server does not list its models, so the model name could not be checked.")
	case !result.ModelLoaded:
		fmt.Println("Server is reachable, but model not found in /v1/models list.")
		fmt.Println("You may still be able to use it: double-check the model name.")
	default:
		fmt.Println("Connection successful!")
		fmt.Println("Model is loaded and ready.")
	}

	configPath, err := globalConfigPath("settings.toml")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not determine home directory: %v\n", err)
		os.Exit(1)
	}
	configDir := filepath.Dir(configPath)
	if err := os.MkdirAll(configDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Could not create config directory: %v\n", err)
		os.Exit(1)
	}

	// 0600 for the backup and the file: both can hold an api_key.
	if _, err := os.Stat(configPath); err == nil {
		old, readErr := os.ReadFile(configPath)
		if readErr == nil {
			hash := sha256.Sum256(old)
			shaPrefix := hex.EncodeToString(hash[:])[:8]
			ts := time.Now().Format("20060102150405")
			backupPath := filepath.Join(configDir, fmt.Sprintf("settings.backup-%s-%s", ts, shaPrefix))
			if writeErr := os.WriteFile(backupPath, old, 0o600); writeErr != nil {
				fmt.Fprintf(os.Stderr, "Warning: could not create backup: %v\n", writeErr)
			}
		}
	}

	f, err := os.OpenFile(configPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not create settings.toml: %v\n", err)
		os.Exit(1)
	}
	// OpenFile keeps the mode of a file that already exists.
	if err := f.Chmod(0o600); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: could not restrict settings.toml to your user: %v\n", err)
	}
	if err := toml.NewEncoder(f).Encode(settings); err != nil {
		fmt.Fprintf(os.Stderr, "Could not write settings.toml: %v\n", err)
		os.Exit(1)
	}
	if err := f.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "Could not write settings.toml: %v\n", err)
		os.Exit(1)
	}

	fmt.Println()
	fmt.Printf("Settings written to %s\n", configPath)
	fmt.Println("Setup complete! You can now start codehalter.")
}
