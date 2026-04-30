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
- **History compression** — older messages are hierarchically summarized by the `summary` LLM to stay within token budgets.
- **Two modes** — *Interactive* (ask before non-trivial actions) and *Autopilot* (auto-answer prompts, no interruption). Selectable per-session from the Zed mode picker.
- **Configurable LLM endpoints** — different roles (`thinking`, `execute`, `summary`) can point at different models or servers.
- **Slash commands** — `/improve` prints a markdown snapshot of the current session (settings, runners, messages, log tail; API keys redacted) without invoking the LLM, designed to paste into another assistant when reporting issues. `/clean` archives the session's full state to `session_archive_*.toml` and resets the live session — same SessionId, empty Messages/History; Zed's chat panel is not cleared (ACP has no agent → client reload), so close+reopen the session for a visually empty view.

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

Position-based: index 0 of `[[llmconnections]]` is the **main** tier (your foreground agent), indices 1+ are the **subagent** tier (parallel/offloaded work like `launch_subagent` or web summarisation). When the agent picks a connection, a session running at depth 0 prefers the main tier; subagents (depth > 0) prefer subagent tier. Each falls back to the other if needed.

Each connection declares per-role sampler overrides via `extra_body_thinking`, `extra_body_execute`, `extra_body_summary`. The right one is merged into every OpenAI request. Core fields (`model`, `messages`, `stream`, `tools`) always win over `extra_body_*`.

Single-machine setup — one connection serves every role; sampler swaps between phases (KV cache stays shared because samplers don't affect cache keys):

```toml
[[llmconnections]]
url = "http://localhost:8080/v1/chat/completions"
model = "qwen3.6-27b"
extra_body_thinking = { temperature = 1.0, top_p = 0.95, top_k = 20, min_p = 0.0 }
extra_body_execute  = { temperature = 0.6, top_p = 0.95, top_k = 20, min_p = 0.0 }
extra_body_summary  = { temperature = 0.6, top_p = 0.95, top_k = 20 }
```

Multi-machine setup — main agent on the 7900XTX, subagents on a Strix Halo, summary on a 4b box:

```toml
# index 0 — main tier
[[llmconnections]]
url = "http://7900xtx:8080/v1/chat/completions"
model = "qwen3-coder-30b"
extra_body_thinking = { temperature = 1.0, top_p = 0.95 }
extra_body_execute  = { temperature = 0.6, top_p = 0.95 }

# index 1 — subagent tier (parallel work)
[[llmconnections]]
url = "http://strix-halo:8080/v1/chat/completions"
model = "qwen3-120b"
extra_body_thinking = { temperature = 1.0, top_p = 0.95 }
extra_body_execute  = { temperature = 0.6, top_p = 0.95 }

# index 2 — also subagent tier; declares only summary, so summary calls land here
[[llmconnections]]
url = "http://summary-box:8080/v1/chat/completions"
model = "qwen3-4b"
extra_body_summary = { temperature = 0.6, top_p = 0.95 }
```

For a hybrid-reasoning model, drive the reasoning toggle from the same `extra_body_<role>` keys:

```toml
[[llmconnections]]
url = "http://localhost:8080/v1/chat/completions"
model = "qwen3"
extra_body_thinking = { enable_thinking = true }
extra_body_execute  = { enable_thinking = false }
```

| Role | Purpose | Suggested temperature |
|------|---------|------------------------|
| `thinking` | Planning, pre-verification, post-execution verification | ~1.0 (diverse hypotheses, edge-case exploration) |
| `execute` | Running the tool loop that does the work | ~0.6 (precise tool calls, fewer hallucinated args) |
| `summary` | History compression and web-page summarisation | ~0.6 (faithful, not creative) |

If a role has no `extra_body_<role>` declared on any connection, codehalter still falls back to the first connection in tier order — the override is purely opt-in.

### Prompt files

On first run, `ensureDefaults` drops three phase templates into `.codehalter/`:

| File | Role |
|------|------|
| `PLAN.md` | Planning-phase instructions (clarity check, info retrieval, steps/subtasks JSON schema) |
| `EXECUTE.md` | Execution-phase directives prepended to the user message |
| `VERIFY.md` | Self-verification rubric applied to execute output |

Plus per-stack `SKILL-{lang}.md` files for any language detected in the project root (`go.mod` → SKILL-go.md, `package.json`+`tsconfig.json` → SKILL-ts.md, plain `package.json` → SKILL-js.md, `pom.xml`/`build.gradle` → SKILL-java.md, `*.sh`/`*.bash` files → SKILL-bash.md). Skills are concatenated into the system prompt on the first turn so they ride along in cached history. Designed for smaller local models (Qwen3, Gemma) that need explicit language conventions; larger models can usually have them deleted.

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

## Sandboxing with a devcontainer

Codehalter edits files and runs build/test commands. Running it inside a [devcontainer](https://containers.dev) sandboxes those actions to a throwaway environment, keeps the project's toolchain pinned, and means the same setup works on every machine. Zed launches ACP servers *inside* the container when you "Reopen in Container," so codehalter, Firefox, and the LLM endpoint must be reachable from there. The startup banner reports `Container: devcontainer` when this is wired up correctly.

Minimal Debian-based example. Drop `.devcontainer/Dockerfile` and `.devcontainer/devcontainer.json` into your project:

```dockerfile
# .devcontainer/Dockerfile
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git just make firefox-esr golang-go \
    && rm -rf /var/lib/apt/lists/*

ENV GOBIN=/usr/local/bin
RUN go install github.com/tbocek/codehalter@latest
```

```jsonc
// .devcontainer/devcontainer.json
{
  "name": "codehalter",
  "build": { "dockerfile": "Dockerfile" },
  // Lets settings.toml reach an LLM running on the host as host.docker.internal:8080
  "runArgs": ["--add-host=host.docker.internal:host-gateway"],
  "containerEnv": { "DEVCONTAINER": "true" }
}
```

Then in your Zed settings, point `command` at the in-container path:

```json
"Codehalter": { "type": "custom", "command": "/usr/local/bin/codehalter", "args": [], "env": {} }
```

If your LLM server runs on the host, change the URL in `.codehalter/settings.toml` to `http://host.docker.internal:8080/v1/chat/completions`. For an existing language-stack devcontainer (e.g. `mcr.microsoft.com/devcontainers/go:1`), just add `firefox-esr` and `go install github.com/tbocek/codehalter@latest` to your image — no need to start from this Debian base.

## How it works

1. Zed spawns `codehalter` as a subprocess and communicates via JSON-RPC 2.0 over stdio.
2. On session start, the agent indexes project files, probes image support on the configured LLM, discovers task runners, and reports which of build/test/lint/format are covered.
3. Each prompt runs through the pipeline:
   - **Pre-verify** — read-only check of project health; failures are folded into the planning input.
   - **Plan** — the `thinking` LLM analyzes the request with read-only tools, gathers external info via web tools if needed, and emits either `steps` (simple task) or `subtasks` (big task). In Interactive mode the user confirms before execution.
   - **Execute** — the `execute` LLM runs the agentic tool loop; file edits are shown as diffs and require user approval in Interactive mode.
   - **Verify** — the output is self-checked against `VERIFY.md`. On failure with fix steps, the loop re-plans with failure context (up to 5 attempts total).
4. For big tasks, each subtask repeats the plan → execute → verify cycle independently; prior subtasks' assistant replies stay in history so later subtasks have context.
5. Conversation history is persisted to `.codehalter/session_<id>.toml`. When it grows large, the `summary` LLM compresses older messages into cascading summary levels.
