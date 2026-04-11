package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// discoverRunners checks for known task runners in the project and registers
// a "run_task" tool if any are found.
func (a *agent) discoverRunners(cwd string) {
	var runners []taskRunner

	if r := discoverJust(cwd); r != nil {
		runners = append(runners, *r)
	}
	if r := discoverMake(cwd); r != nil {
		runners = append(runners, *r)
	}
	if r := discoverNpm(cwd); r != nil {
		runners = append(runners, *r)
	}

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
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, args map[string]string) string {
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
		output, err := cmd.CombinedOutput()
		result := string(output)
		if err != nil {
			result += "\nexit: " + err.Error()
			a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(result)})
			return result
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
	default:
		return []string{target}
	}
}

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
