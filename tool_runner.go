package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// discoverRunners checks for known task runners in the project and registers
// a "run_task" tool if any are found. Also classifies discovered tasks by
// purpose (build/test/lint/format) and stores the result on the agent for the
// startup notification.
//
// If the project is completely empty, we don't write any skeleton files —
// we just flag the empty state so the first user turn can inject a hint
// telling the LLM to ask the user which language/runner to bootstrap with.
func (a *agent) discoverRunners(cwd string) {
	runners := detectRunners(cwd)

	a.mu.Lock()
	a.capabilities = classifyRunners(runners)
	a.emptyProject = len(runners) == 0 && isEmptyProject(cwd)
	a.mu.Unlock()

	if len(runners) == 0 {
		return
	}

	// Build the description with available tasks.
	var desc strings.Builder
	desc.WriteString("Run a project task. Available tasks:\n")
	for _, r := range runners {
		for _, t := range r.Tasks {
			fmt.Fprintf(&desc, "  %s:%s\n", r.Name, t)
		}
	}

	// Build enum of valid task names.
	var validTasks []string
	for _, r := range runners {
		for _, t := range r.Tasks {
			validTasks = append(validTasks, r.Name+":"+t)
		}
	}

	a.mu.Lock()
	a.runners = runners
	a.mu.Unlock()

	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "run_task",
			"description": desc.String(),
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"task"},
				"properties": map[string]any{
					"task": map[string]any{
						"type":        "string",
						"description": "Task to run in the format runner:target (e.g. just:build, npm:test)",
						"enum":        validTasks,
					},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) string {
		args := parseArgs(rawArgs)
		task := args["task"]
		parts := strings.SplitN(task, ":", 2)
		if len(parts) != 2 {
			return "error: task must be in format runner:target"
		}
		runnerName, target := parts[0], parts[1]

		// Find the runner.
		a.mu.Lock()
		var runner *taskRunner
		for i := range a.runners {
			if a.runners[i].Name == runnerName {
				runner = &a.runners[i]
				break
			}
		}
		a.mu.Unlock()

		if runner == nil {
			return fmt.Sprintf("error: unknown runner %q", runnerName)
		}

		// Verify the target is valid.
		valid := false
		for _, t := range runner.Tasks {
			if t == target {
				valid = true
				break
			}
		}
		if !valid {
			return fmt.Sprintf("error: unknown target %q for runner %q", target, runnerName)
		}

		sess := a.getSession(sid)
		if sess == nil {
			return "error: no session"
		}

		tcId := a.StartToolCall(ctx, sid, "Running "+task, "execute", nil)

		cmd := exec.CommandContext(ctx, runner.Command, runner.Args(target)...)
		cmd.Dir = sess.Cwd

		// Stream stdout+stderr to the UI line-by-line so the user sees
		// progress from long-running builds, while still collecting the
		// full transcript to hand back to the LLM as tool output.
		pipeR, pipeW := io.Pipe()
		cmd.Stdout = pipeW
		cmd.Stderr = pipeW

		if err := cmd.Start(); err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error starting task: " + err.Error()
		}

		waitErr := make(chan error, 1)
		go func() {
			waitErr <- cmd.Wait()
			_ = pipeW.Close()
		}()

		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("\n```\n$ "+task+"\n")))

		var collected strings.Builder
		scanner := bufio.NewScanner(pipeR)
		// go test -v / verbose build output can produce long single lines;
		// the default 64 KB ceiling silently drops them.
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			line := scanner.Text() + "\n"
			collected.WriteString(line)
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(line)))
		}
		runErr := <-waitErr
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("```\n")))

		result := collected.String()
		if runErr != nil {
			result += "\nexit: " + runErr.Error()
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(result)})
		return result
	}})
}

type taskRunner struct {
	Name    string
	Command string
	Tasks   []string
}

func (r *taskRunner) Args(target string) []string {
	switch r.Name {
	case "just":
		return []string{target}
	case "make":
		return []string{target}
	case "npm":
		return []string{"run", target}
	case "gradle":
		return []string{target}
	case "cargo":
		return []string{target}
	default:
		return []string{target}
	}
}

// detectRunners runs every supported runner-discovery probe and returns what
// was found. Split out from discoverRunners so the empty-dir fallback can
// re-probe after writing a placeholder Makefile.
func detectRunners(cwd string) []taskRunner {
	var runners []taskRunner
	probes := []func(string) *taskRunner{
		discoverJust, discoverMake, discoverNpm, discoverCargo, discoverGradle,
	}
	for _, p := range probes {
		if r := p(cwd); r != nil {
			runners = append(runners, *r)
		}
	}
	return runners
}

// capabilities groups runner tasks by what they accomplish so the startup
// notification can flag what's missing per category.
type capabilities struct {
	build   []string
	test    []string
	lint    []string
	format  []string
	runners []string // distinct runner names found
}

// classifyTask returns the category (build/test/lint/format) that a task name
// belongs to, or "" if it doesn't match. Matches on segment boundaries so
// "checkout" isn't mistaken for "check" — we split on `-_:/.` and require an
// exact segment match against the keyword set. First match wins.
func classifyTask(task string) string {
	segments := strings.FieldsFunc(strings.ToLower(task), func(r rune) bool {
		return r == '-' || r == '_' || r == ':' || r == '.' || r == '/'
	})
	for _, seg := range segments {
		switch seg {
		case "build", "compile", "bundle", "dist":
			return "build"
		case "test", "tests", "spec", "specs":
			return "test"
		case "lint", "vet", "check", "verify", "clippy", "audit":
			return "lint"
		case "fmt", "format", "prettier":
			return "format"
		}
	}
	return ""
}

func classifyRunners(runners []taskRunner) capabilities {
	var c capabilities
	seenRunner := map[string]bool{}
	for _, r := range runners {
		if !seenRunner[r.Name] {
			c.runners = append(c.runners, r.Name)
			seenRunner[r.Name] = true
		}
		for _, task := range r.Tasks {
			entry := r.Name + ":" + task
			switch classifyTask(task) {
			case "build":
				c.build = append(c.build, entry)
			case "test":
				c.test = append(c.test, entry)
			case "lint":
				c.lint = append(c.lint, entry)
			case "format":
				c.format = append(c.format, entry)
			}
		}
	}
	return c
}

// isEmptyProject returns true when cwd has no meaningful source or config
// files — a fresh `mkdir foo && cd foo` case where bootstrapping a Makefile
// is safe. The .codehalter/ dir we create ourselves is ignored.
func isEmptyProject(cwd string) bool {
	entries, err := os.ReadDir(cwd)
	if err != nil {
		return false
	}
	for _, e := range entries {
		name := e.Name()
		if name == ".codehalter" {
			continue
		}
		if strings.HasPrefix(name, ".") && e.IsDir() {
			// Hidden dirs like .git/.idea don't count as "real" content, but
			// their presence means this isn't a pristine mkdir — bail out.
			return false
		}
		return false
	}
	return true
}

// emptyProjectHint is injected onto the first user turn when the working
// directory has no source files, manifests, or runner config. It tells the
// LLM to ask what language/framework the user wants before writing anything.
const emptyProjectHint = `[Note: this project directory is empty — no source files or build manifests were found. Before doing anything else, use the ask_user tool to confirm:
1. What language/framework should this project use? (Rust, Go, Node.js, Python, C, etc.)
2. Which build runner do they prefer? (Cargo, go modules, npm/pnpm, just, Make)

Only then create the appropriate skeleton — Cargo.toml for Rust, go.mod for Go, package.json for Node, justfile/Makefile otherwise — with sensible build/test/lint/format targets.]
`

func discoverJust(cwd string) *taskRunner {
	for _, name := range []string{"justfile", "Justfile", ".justfile"} {
		if _, err := os.Stat(filepath.Join(cwd, name)); err == nil {
			cmd := exec.Command("just", "--list", "--unsorted")
			cmd.Dir = cwd
			out, err := cmd.Output()
			if err != nil {
				return nil
			}
			var tasks []string
			for _, line := range strings.Split(string(out), "\n") {
				line = strings.TrimSpace(line)
				if line == "" || strings.HasPrefix(line, "Available") {
					continue
				}
				// "target # description" or just "target"
				name := strings.Fields(line)[0]
				tasks = append(tasks, name)
			}
			if len(tasks) == 0 {
				return nil
			}
			return &taskRunner{Name: "just", Command: "just", Tasks: tasks}
		}
	}
	return nil
}

func discoverMake(cwd string) *taskRunner {
	for _, name := range []string{"Makefile", "makefile", "GNUmakefile"} {
		if _, err := os.Stat(filepath.Join(cwd, name)); err == nil {
			// Parse .PHONY targets and simple targets from the Makefile.
			data, err := os.ReadFile(filepath.Join(cwd, name))
			if err != nil {
				return nil
			}
			var tasks []string
			seen := map[string]bool{}
			for _, line := range strings.Split(string(data), "\n") {
				if strings.HasPrefix(line, ".PHONY:") {
					for _, t := range strings.Fields(line)[1:] {
						if !seen[t] {
							tasks = append(tasks, t)
							seen[t] = true
						}
					}
				} else if len(line) > 0 && line[0] != '\t' && line[0] != '#' && line[0] != '.' {
					if idx := strings.Index(line, ":"); idx > 0 {
						t := strings.TrimSpace(line[:idx])
						if !strings.ContainsAny(t, " $%") && !seen[t] {
							tasks = append(tasks, t)
							seen[t] = true
						}
					}
				}
			}
			if len(tasks) == 0 {
				return nil
			}
			return &taskRunner{Name: "make", Command: "make", Tasks: tasks}
		}
	}
	return nil
}

func discoverNpm(cwd string) *taskRunner {
	path := filepath.Join(cwd, "package.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var pkg struct {
		Scripts map[string]string `json:"scripts"`
	}
	if err := json.Unmarshal(data, &pkg); err != nil {
		return nil
	}
	var tasks []string
	for name := range pkg.Scripts {
		tasks = append(tasks, name)
	}
	if len(tasks) == 0 {
		return nil
	}
	return &taskRunner{Name: "npm", Command: "npm", Tasks: tasks}
}

// discoverCargo exposes the standard `cargo` subcommands when Cargo.toml is
// present. `cargo check` and `cargo clippy` cover the lint slot.
func discoverCargo(cwd string) *taskRunner {
	if _, err := os.Stat(filepath.Join(cwd, "Cargo.toml")); err != nil {
		return nil
	}
	return &taskRunner{
		Name:    "cargo",
		Command: "cargo",
		Tasks:   []string{"build", "test", "check", "clippy", "fmt", "clean"},
	}
}

func discoverGradle(cwd string) *taskRunner {
	for _, name := range []string{"build.gradle", "build.gradle.kts"} {
		if _, err := os.Stat(filepath.Join(cwd, name)); err == nil {
			// Try to find the wrapper first.
			cmdName := "gradle"
			if _, err := os.Stat(filepath.Join(cwd, "gradlew")); err == nil {
				cmdName = "./gradlew"
			} else if _, err := os.Stat(filepath.Join(cwd, "gradlew.bat")); err == nil {
				// For Windows, but we assume Unix-like environment for now.
				// If we were strictly Windows, we'd use gradlew.bat.
				cmdName = "gradlew"
			}

			// List tasks.
			cmd := exec.Command("gradle", "tasks")
			if cmdName != "gradle" {
				cmd = exec.Command(cmdName, "tasks")
			}
			cmd.Dir = cwd
			out, err := cmd.Output()
			if err != nil {
				return nil
			}

			var tasks []string
			for _, line := range strings.Split(string(out), "\n") {
				line = strings.TrimSpace(line)
				if line == "" || strings.HasPrefix(line, "Tasks:") || strings.HasPrefix(line, "---") {
					continue
				}
				// Gradle tasks often look like: "build" or "test"
				// and often have indentation.
				if len(line) > 0 && line[0] == ' ' {
					// It's likely a task description or something.
					// We only want the task name.
					// But Gradle's `tasks` output is often formatted.
					// Let's just try to split by space and take the first part.
					parts := strings.Fields(line)
					if len(parts) > 0 {
						tasks = append(tasks, parts[0])
					}
				}
			}
			if len(tasks) == 0 {
				// If we can't parse the tasks, just provide basic ones.
				return &taskRunner{
					Name:    "gradle",
					Command: cmdName,
					Tasks:   []string{"build", "test", "clean"},
				}
			}
			return &taskRunner{
				Name:    "gradle",
				Command: cmdName,
				Tasks:   tasks,
			}
		}
	}
	return nil
}
