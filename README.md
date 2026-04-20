# Codehalter

An [ACP](https://spec.anthropic.com/acp)-compatible AI coding agent that connects [Zed](https://zed.dev) to any OpenAI-compatible LLM server (llama.cpp, Ollama, vLLM, etc.). Runs entirely on your machine. Designed for small to medium sized repositories.

## Features

- **Plan → Execute → Verify pipeline** — every prompt is routed through a planning pass, an execution pass, and a self-verification pass. Failed verifications re-plan with the failure context; up to 5 attempts before giving up.
- **Pre-planning verification** — a read-only project health check runs before planning so broken baseline state is surfaced instead of silently corrupted by new work.
- **Big-task decomposition** — when the planner splits a request into multiple subtasks, the user picks once: *Interactive* (approve each subtask), *Automatic* (run the whole batch under autopilot), or *Cancel*. Each subtask gets its own full plan → execute → verify cycle.
- **Subagents** — `launch_subagent` spawns parallel sub-tasks for independent work; each runs its own plan/execute/verify, up to 2 levels deep.
- **Agentic tool loop** — the LLM can read, edit, search, and run tasks iteratively with file edits surfaced as diffs.
- **Built-in tools** — `read_file`, `write_file`, `edit_file`, `list_files`, `search_text` (literal or regex), `run_task`, `ask_user`, `web_search`, `web_read`, `launch_subagent`.
- **Task runner integration** — auto-discovers targets from `justfile`, `Makefile`, `package.json`, `go.mod`, or `Cargo.toml` and classifies them as build/test/lint/format for the startup report.
- **Empty-project bootstrap** — fresh directories are flagged; the LLM asks on the first turn which language/runner to scaffold before writing anything.
- **Web tools** — `web_search` (DuckDuckGo) and `web_read` open results in Firefox for review, then summarize. Restricted to the planning phase so execution stays offline.
- **Image support** — when the active LLM advertises vision, prompt images are passed through as OpenAI-style content blocks.
- **Session persistence** — conversations are saved as TOML files under `.codehalter/` and can be resumed across editor restarts.
- **History compression** — older messages are hierarchically summarized to stay within token budgets; `CodeRef` hashes flag summaries whose referenced code has since changed.
- **Two modes** — *Interactive* (ask before non-trivial actions) and *Autopilot* (auto-answer prompts, no interruption). Selectable per-session from the Zed mode picker.
- **Configurable LLM endpoints** — different roles (`thinking`, `execute`, `summary`) can point at different models or servers.

## Prerequisites

- Go 1.26+
- An OpenAI-compatible LLM server (e.g. [llama.cpp](https://github.com/ggml-org/llama.cpp), [Ollama](https://ollama.com), [vLLM](https://github.com/vllm-project/vllm))
- Firefox, if you want `web_search` / `web_read` to work

## Build

```sh
just build
```

Or directly:

```sh
go build -o codehalter .
```

## Configuration

Codehalter looks for settings in two places (project-local takes priority):

1. `.codehalter/settings.toml` in the project root
2. `~/.config/codehalter/settings.toml` as a global fallback

### Example settings

```toml
[[llmconnections]]
name = "thinking"
url = "http://localhost:8080/v1/chat/completions"
model = "qwen3"

[[llmconnections]]
name = "execute"
url = "http://localhost:8080/v1/chat/completions"
model = "qwen3-coder"

[[llmconnections]]
name = "summary"
url = "http://localhost:8080/v1/chat/completions"
model = "qwen3-fast"
```

All three roles (`thinking`, `execute`, `summary`) are required — codehalter refuses to start a session if any are missing. If you only have one model available, point all three `[[llmconnections]]` entries at the same URL and model.

Each connection also accepts an optional `extra_body` table whose keys are merged into every request. Useful for toggling reasoning modes on hybrid models so one loaded model can serve both `thinking` and `execute`:

```toml
[[llmconnections]]
name = "thinking"
url = "http://localhost:8080/v1/chat/completions"
model = "qwen3"
extra_body = { enable_thinking = true }

[[llmconnections]]
name = "execute"
url = "http://localhost:8080/v1/chat/completions"
model = "qwen3"
extra_body = { enable_thinking = false }
```

Core fields (`model`, `messages`, `stream`, `tools`) always win over `extra_body`.

Each connection has a role:

| Name | Purpose |
|------|---------|
| `thinking` | Planning, pre-verification, and post-execution verification |
| `execute` | Running the tool loop that does the work |
| `summary` | History compression and web-page summarization |

### Prompt files

On first run, `ensureDefaults` drops five templates into `.codehalter/`:

| File | Role |
|------|------|
| `AGENT.md` | System-level rules the LLM sees on every turn |
| `PLAN.md` | Planning-phase instructions (clarity check, info retrieval, steps/subtasks JSON schema) |
| `EXECUTE.md` | Execution-phase directives prepended to the user message |
| `VERIFY.md` | Self-verification rubric applied to execute output |
| `SUMMARY.md` | History-compression prompt used by the `summary` LLM |

Edit any of them to customize behavior for your project.

## Running the LLM server

With llama.cpp:

```sh
llama-server -m your-model.gguf --port 8080
```

With Ollama:

```sh
ollama serve
```

Then adjust the `url` in your settings to match.

## Zed setup

Add to your Zed settings (`~/.config/zed/settings.json`):

```json
{
  "agent_servers": {
    "Codehalter": {
      "type": "custom",
      "command": "/absolute/path/to/codehalter",
      "args": [],
      "env": {}
    }
  }
}
```

Open the agent panel (`Cmd+?` / `Ctrl+?`), click `+`, and select "Codehalter". Pick Interactive or Autopilot from the mode selector in the session header.

## How it works

1. Zed spawns `codehalter` as a subprocess and communicates via JSON-RPC 2.0 over stdio.
2. On session start, the agent indexes project files, probes image support on the configured LLM, discovers task runners, and reports which of build/test/lint/format are covered.
3. Each prompt runs through the pipeline:
   - **Pre-verify** — read-only check of project health; failures are folded into the planning input.
   - **Plan** — the `thinking` LLM analyzes the request with read-only tools, gathers external info via web tools if needed, and emits either `steps` (simple task) or `subtasks` (big task). In Interactive mode the user confirms before execution.
   - **Execute** — the `execute` LLM runs the agentic tool loop; file edits are shown as diffs and require user approval in Interactive mode.
   - **Verify** — the output is self-checked against `VERIFY.md`. On failure with fix steps, the loop re-plans with failure context (up to 5 attempts total).
4. For big tasks, each subtask repeats the plan → execute → verify cycle independently; prior subtasks' assistant replies stay in history so later subtasks have context.
5. Conversation history is persisted to `.codehalter/session_<id>.toml`. When it grows large, the `summary` LLM compresses older messages into cascading summary levels; `CodeRef` hashes flag summaries whose files have since changed.
