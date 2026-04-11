# Codehalter

An [ACP](https://spec.anthropic.com/acp)-compatible AI coding agent that connects [Zed](https://zed.dev) to any OpenAI-compatible LLM server (llama.cpp, Ollama, vLLM, etc.). Runs entirely on your machine. Designed for small to medium sized repositories.

## Features

- **Agentic tool loop** — the LLM can read, edit, and create files, search code, and run tasks iteratively (up to 10 rounds per prompt)
- **Built-in tools** — `read_file`, `write_file`, `edit_file`, `list_files`, `search_text`, `run_task`, `ask_user`
- **Task runner integration** — automatically discovers and runs targets from `justfile`, `Makefile`, or `package.json`
- **Session persistence** — conversations are saved as TOML files and can be resumed across editor restarts
- **History compression** — older messages are hierarchically summarized to stay within token budgets
- **Project indexing** — builds file summaries on session start so the LLM has project-wide context
- **Planning** — analyzes each request for clarity and complexity, routes to the appropriate model, and asks for clarification when needed
- **Two modes** — *discussion* (read-only) and *execution* (read/write with user approval for changes)
- **Configurable LLM endpoints** — point different roles (thinking, summary) at different models or servers

## Prerequisites

- Go 1.26+
- An OpenAI-compatible LLM server (e.g. [llama.cpp](https://github.com/ggml-org/llama.cpp), [Ollama](https://ollama.com), [vLLM](https://github.com/vllm-project/vllm))

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
name = "summary"
url = "http://localhost:8080/v1/chat/completions"
model = "qwen3-fast"
```

Each connection has a role:

| Name | Purpose |
|------|---------|
| `thinking` | Used for complex requests that need deeper reasoning |
| `summary` | Used for history compression |

### Agent behavior

Drop an `AGENT.md` file in your project root to customize the LLM's behavior (see `AGENT.md.example` for a starting point).

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

Open the agent panel (`Cmd+?` / `Ctrl+?`), click `+`, and select "Codehalter".

## How it works

1. Zed spawns `codehalter` as a subprocess and communicates via JSON-RPC 2.0 over stdio
2. On session start, the agent indexes project files and builds summaries for context
3. Each prompt goes through a planning phase that assesses clarity and complexity
4. The LLM runs an agentic loop — calling tools, reading results, and iterating until the task is complete
5. File edits are shown as diffs and require user approval before writing
6. Conversation history is persisted to `.codehalter/` and compressed when it grows large
