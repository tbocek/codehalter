# Codehalter

An [ACP](https://agentclientprotocol.com)-compatible AI coding agent that connects [Zed](https://zed.dev), and any other Agent Client Protocol editor, to any OpenAI-compatible LLM server. **Devcontainer-first and local-LLM-first**: built around running inside a devcontainer, **Alpine, Arch, Debian, Fedora, or Ubuntu**, against [llama.cpp](https://github.com/ggml-org/llama.cpp) (or Ollama / vLLM) on the host. Your real `.git` and identity are available inside the container, but the agent is **instructed never to commit, push, or rewrite history on its own**, only when you explicitly ask. Targets small-to-medium repositories.

## Highlights

- **Small and auditable**: under 10k lines of code (≈8.4k SLOC; ~12k including its thorough comments), a single dependency, one static binary.
- **Local-LLM-first**: designed and tuned for small/medium models on your own [llama.cpp](https://github.com/ggml-org/llama.cpp) / Ollama / vLLM box, and works with hosted OpenAI-compatible APIs too.
- **ACP-only, zero UI**: codehalter ships no interface of its own; it speaks the [Agent Client Protocol](https://agentclientprotocol.com) over stdio, so the editor is the UI. Native in [Zed](https://zed.dev); usable from JetBrains and VS Code through their ACP integrations.
- **Prefix-cache consistency as a first principle**: the rendered prompt stays byte- and position-stable turn over turn, so the server's KV cache keeps ~90%+ warm. The *only* deliberate cache break is a new MCP server's tools coming online; skills added mid-session ride as user messages until the next compaction folds them into the cached system prompt.
- **Smart background compaction**: **after every turn**, a *separate* LLM slot condenses the whole turn into a structured note in the background while you read the reply. The notes accumulate (persisted across restarts), so when the context crosses the trigger they are already prepared and compaction is fast instead of a stall. Two tiers: at **80%** of the slot budget it compacts at the turn boundary (folds every completed turn, starts the next turn fresh); above **90%** it compacts mid-turn, keeping the in-flight turn verbatim and folding only the completed turns before it. **Needs ≥2 parallel LLM slots** (two `[[llm]]` entries, or one with `parallel ≥ 2`); with a single slot the summariser evicts the foreground cache and throughput tanks.
- **Lightweight plan → execute → document loop**: each prompt is planned, executed with per-subtask self-verification, and documented only when the change is user-visible, no heavyweight agent framework.
- **Devcontainer-first**: on the first session codehalter scaffolds a throwaway devcontainer from your pick of **Alpine, Arch, Debian, Fedora, or Ubuntu**, `go install`s itself + the toolchain, and runs entirely inside it. Edits and build/test commands are sandboxed to the container. It works with your real `.git`, but is **instructed never to commit, push, or rewrite history on its own**, that stays your call.
- **Project onboarding**: detects your stack and task runners, seeds per-language skills, wires up code intelligence, and bootstraps empty repositories. When a tool is missing or code intelligence isn't set up, it offers a one-click **"fix it?" card** rather than just failing.
- **Reusable prompt macros**: type `/<name> <args>` to expand a `TEMPLATE-<name>.md` into a full prompt. Ships **`/grill-me`** (the agent interviews you about a plan until the whole design tree is resolved); drop your own `TEMPLATE-*.md` for project- or team-specific workflows.
- **Git commit messages on the fly**: a background drafter refreshes `.codehalter/.git_commit` to match the working tree **after every turn**, so a message is always ready. The agent never commits on its own; when you ask, it runs `git commit -F …` with that message (or hands you the command).
- **Per-turn usage stats**: every reply ends with a server-measured line such as `Done in 1m22s · 11k prompt + 2.1k gen (122k sent, 91% cached) · 388 pp/s · 39 tg/s` (evaluated vs cached tokens, prompt/generation throughput).
- **Code intelligence beyond Go**: gopls is wired in for Go out of the box; every other popular language bridges its language server to MCP via [lsmcp](https://github.com/mizchi/lsmcp), TypeScript/JavaScript, C/C++ (clangd), Rust (rust-analyzer), Python, …, so the model gets real go-to-definition / find-references / symbol search everywhere, not just in Go. *(non-Go bridges are set up per project via a skill or a setup card; experimental.)*

### Roadmap

- **Per-tool usage statistics**: call counts, latency, and failure rates per tool.

## Features

- **Devcontainer scaffolding (5 base distros)**: the first session prompts `Alpine / Arch / Debian / Fedora / Ubuntu` and writes `.devcontainer/Dockerfile` + `devcontainer.json` for the chosen base. Each image installs a minimal toolchain (git, curl, sudo, Go, Firefox) and `go install`s codehalter into `/usr/local/bin` under a passwordless `dev` sudo user; the Arch variant additionally builds `yay` from AUR. Language toolchains (gopls, node, clang, …) are added per project on demand. The seeded `devcontainer.json` bind-mounts the host's `.gitconfig` (so the agent's git sees your identity) and the project's `.git` directory (read-write), the agent works with real git but is **instructed never to commit, push, or rewrite history on its own**, only when you explicitly ask.
- **Commit workflow**: the agent never commits or pushes on its own. A background updater on `llm[2+]` keeps `.codehalter/.git_commit` in sync with `git diff HEAD` after every turn; when you ask for a commit it regenerates the message if stale and runs `git commit -F .codehalter/.git_commit` (and `git push`) itself, falling back to handing you the host command on an older read-only container. After the next prompt, codehalter detects the clean tree and deletes `.git_commit` so the next round starts fresh.
- **Bring-your-own MCP**: `.codehalter/mcp.toml` declares each `[[server]]` as a child process to spawn (`name`, `command`, `args`, `env`); tools it advertises are registered as `<name>__<tool>`. Projects with a `go.mod` get a pre-seeded gopls entry (`gopls mcp` exposes `go_symbols`, `go_references`, `go_definition`, `go_hover`, …). Reconciled on every prompt, add or remove a `[[server]]` between turns and the change applies without restarting; failed startups surface as a red tool-call card and mtime-gate so they don't re-emit until you edit the file.
- **Parallel LLM slots for local servers**: three `[[llm]]` entries is the sweet spot: `llm[0]` owns the foreground KV cache, the per-turn structured summariser pins to `llm[1]`, and the `.git_commit` drafter prefers `llm[2+]`. `launch_subagent` fans out breadth-first across all slots with a per-conn semaphore. Each slot's `parallel = N` caps concurrent requests to match its server's capacity.
- **Plan → Execute → Verify → Document pipeline**: every prompt is routed through a planning pass, an execution pass, and a self-verification pass. Failed verifications re-plan with the failure context; up to 20 attempts before giving up, with fuzzy duplicate-failure detection (Jaccard similarity over issue words) that bails early when the model keeps rephrasing the same problem. Turns that wrote files end with a documentation pass that updates (or creates) the README when the change is user-visible.
- **Big-task decomposition**: when the planner splits a request into multiple subtasks, the user picks once: *Interactive* (approve each subtask), *Automatic* (run the whole batch under autopilot), or *Cancel*. Each subtask gets its own full plan → execute → verify cycle.
- **Subagents**: `launch_subagent` spawns parallel sub-tasks for independent work; each runs its own tool loop, up to 2 levels deep. The first subagent in a batch pins to `llm[0]` and inherits the parent's full history (prefix cache already warm); the rest start fresh on `llm[1+]` with just their `instructions` + `context`.
- **Agentic tool loop**: the LLM can read, edit, search, and run tasks iteratively with file edits surfaced as diffs.
- **Built-in tools**: `read_file`, `continue_read` (paging through a large read), `write_file`, `edit_file`, `list_files`, `search_text` (literal or regex), `run_task`, `run_command` (in-container only), `view_output` (paging spilled large outputs), `view_image` (when the active model has vision), `ask_user`, `web_search`, `web_read`, `web_read_raw`, `launch_subagent`, `respond`. Plus any MCP-server tool, registered as `<server-name>__<tool>`.
- **Synthetic `respond` terminal tool**: the execute and subagent phases expose a `respond(message)` tool that captures the model's final user-facing reply. The turn ends only when `respond` is called, so small local models stay inside the tool-calling grammar they handle best instead of having to decide "tool call vs free text" at exit time. Inspired by [forge](https://github.com/antoinezambelli/forge)'s `respond_tool`.
- **Stuck-tool-loop detection**: a model that re-issues the same call gets nudged once, then the next attempt escalates to a thinking-sampler retry instead of spinning indefinitely.
- **Stack-aware skills**: auto-detected stacks (Go, TypeScript, JavaScript, Java, C/C++, Bash), runner configs (Makefile → `SKILL-makefile.md`, justfile → `SKILL-justfile.md`), and container distro (Alpine/Arch/Debian/Fedora/Ubuntu via `/etc/os-release`) seed `SKILL-*.md` files in `.codehalter/`. `SKILL-base.md` is always seeded (codehalter assumes containerised execution and instructs the agent never to commit, push, or rewrite the project's git on its own, only when you ask). All SKILL files are concatenated into the system prompt on the first turn so they ride along in cached history. Drop your own `SKILL-*.md` to add conventions for any language.
- **Task runner integration**: auto-discovers targets from `justfile`, `Makefile`, `package.json`, `go.mod`, or `Cargo.toml` and classifies them as build/test/lint/format for the startup report.
- **Empty-project bootstrap**: fresh directories are flagged; the LLM asks on the first turn which language/runner to scaffold before writing anything.
- **Proactive fix / setup cards**: before each turn codehalter audits the environment and, when something's missing, shows a single yes/no card instead of failing mid-task: install missing tools (gopls, node, clang, a task runner, …), set up code intelligence (lsmcp / clangd), or report an `mcp.toml` parse/start error. Accepting the card dispatches a synthetic fix turn; declining (or editing the file) dismisses it without re-nagging.
- **Slash-command prompt macros**: a message of the form `/<name> <args>` expands the matching `TEMPLATE-<name>.md` (your copy in `.codehalter/`, else the embedded default) into a full prompt and runs it as a normal turn. Ships `/grill-me` (relentless plan interview); macros that take an argument use `{{}}` and stop with a note if it's missing. Drop your own `TEMPLATE-*.md` to add reusable workflows.
- **Web tools**: `web_search` (DuckDuckGo) returns a list of result titles/URLs/snippets for the LLM to triage; `web_read` then opens a chosen URL in headless Firefox and returns a summary (or `web_read_raw` for verbatim text). Restricted to the planning phase so execution stays offline.
- **Image support**: when the active LLM advertises vision, prompt images are passed through as OpenAI-style content blocks.
- **Session persistence**: conversations are saved as TOML files under `.codehalter/` and can be resumed across editor restarts.
- **History compression**: older turns are summarised to stay within token budgets. After every turn a background goroutine condenses the **whole turn** (the user prompt plus all assistant steps and tool calls, not just the last reply) into a seven-section structured note (Goal / Constraints / Tasks / Progress / Decisions / Next Steps / Critical Context) on a free `llm[1+]` slot. The notes accumulate one per completed turn in a shadow buffer that is **persisted in the session TOML**, so they survive a restart. When the context crosses the trigger, the whole buffer folds into the rolling summary and the turns it covers rotate into an archive file (no synchronous compress pass). Two triggers, both scaled to the LLM's discovered `n_ctx`: at **80%** it compacts at the turn boundary, folding every completed turn and keeping nothing verbatim so the next turn starts fresh; above **90%** it compacts mid-turn, folding the completed turns but keeping the in-flight turn verbatim so the agent keeps working. If the summariser's call fails, the turn still leaves a clipped raw note, so a turn is never rotated out unsummarised. Requires ≥2 `[[llm]]` entries so the summariser runs on a slot separate from the foreground; with only one entry it cannot run and compaction is skipped (a startup card warns).
- **Two modes**: *Interactive* (ask before non-trivial actions) and *Autopilot* (auto-answer prompts, no interruption). Selectable per-session from the Zed mode picker.
- **Configurable LLM endpoints**: different roles (`thinking`, `execute`) can point at different models or servers.

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

Codehalter looks for settings in two places, in this order:

1. `~/.config/codehalter/settings.toml` (global, preferred, used across every project)
2. `<project>/.codehalter/settings.toml` (project-local fallback)

On first run with neither file present, codehalter prompts to write a commented skeleton into `<project>/.codehalter/settings.toml`. Edit it, then move it to `~/.config/codehalter/` to share across projects (the project-local copy can be deleted once the global one exists).

### Example settings

One `[[llm]]` array. Order matters: `llm[0]` is the main connection, `llm[1+]` are extras.

- `llm[0]`, owns the foreground session's KV cache, so its prefix stays warm across turns. Every plan/execute/verify/document call on the main session lands here. Typically a smaller/faster model.
- `llm[1+]`, used to fan out `launch_subagent` tasks in parallel and to host background work routed off `llm[0]` so its prefix cache isn't evicted. The per-turn structured summariser pins to `llm[1]`; the `.git_commit` drafter prefers `llm[2+]` and falls back to `llm[1]` only after the summariser finishes.

`parallel = N` per entry caps how many concurrent LLM calls that entry accepts. The token is held for the duration of one LLM round-trip only, not the lifetime of a subagent, so between LLM calls the slot is free for another caller, and pool size 1 simply serialises everything (no deadlock even when a subagent nests another). For one `launch_subagent` batch, codehalter fans out using a breadth-first interleave of the configured caps: the first subagent pins to `llm[0]` (it inherits the parent's full history so the prefix cache stays warm), the next N to `llm[1]`, then `llm[2]`, …; excess tasks queue.

`params` is forwarded verbatim as the OpenAI request's extra body, put samplers and any model-specific knobs (`enable_thinking`, `reasoning_mode`, …) there. Core fields (`model`, `messages`, `stream`, `tools`) always win over `params`.

`server` is the base URL of your OpenAI-compatible server, the host root only (e.g. `http://localhost:8080`). codehalter appends the API paths itself: `/v1/chat/completions` for completions, plus `/v1/models` and `/props` for probing. Don't include a path.

```toml
[[llm]]
server = "http://localhost:8080"
model = "qwen3.6-27b"
parallel = 1
params_thinking = { temperature = 1.0, top_p = 0.95, top_k = 20, min_p = 0.0 }
params_execute  = { temperature = 0.6, top_p = 0.95, top_k = 20, min_p = 0.0 }

[[llm]]
server = "http://other-host:9001"
model = "qwen3.5-122b"
parallel = 3
params_thinking = { temperature = 1.0, top_p = 0.95, top_k = 20, min_p = 0.0 }
params_execute  = { temperature = 0.6, top_p = 0.8,  top_k = 20, min_p = 0.0 }
```

With the caps above (1 + 3) one `launch_subagent` batch can run up to 4 tasks at once; task 0 pins to `llm[0]`, tasks 1..3 to `llm[1]`, and a 5th task queues on `llm[0]` until task 0 finishes.

> **Recommendation:** configure **at least two `[[llm]]` entries, three is ideal**. Each entry should point at a different physical slot (different llama-server process, different port). With two entries the per-turn structured summariser runs on `llm[1]` while the user reads the reply; with three the `.git_commit` drafter on `llm[2+]` runs in parallel with the summariser instead of queuing behind it. With only one entry every background call would evict `llm[0]`'s prefix cache, so the background features self-disable and fall back to synchronous paths.

| Role | Purpose | Suggested temperature |
|------|---------|------------------------|
| `thinking` | Planning, document, history compaction | ~1.0 (diverse hypotheses, edge-case exploration) |
| `execute` | Tool loop, verify, web-page summarisation | ~0.6 (precise, faithful) |

### Prompt files

On first run, codehalter drops the phase templates into `.codehalter/`:

| File | Role |
|------|------|
| `PLAN.md` | Planning-phase instructions (clarity check, info retrieval, subtask JSON schema with per-subtask verify recipe) |
| `EXECUTE.md` | Execution-phase directives prepended to the user message; instructs the executor to run the verify recipe itself before calling `respond` |
| `DOCUMENT.md` | Decides when the change is user-visible enough to update the README, then edits it minimally |
| `COMMIT.md` | Instructions for the background commit-message drafter that keeps `.codehalter/.git_commit` matching the current uncommitted diff |

Plus per-stack `SKILL-{lang}.md` files for any language detected in the project root (`go.mod` → SKILL-go.md, `package.json`+`tsconfig.json` → SKILL-ts.md, plain `package.json` → SKILL-js.md, `pom.xml`/`build.gradle` → SKILL-java.md, `CMakeLists.txt` or `.c`/`.cpp`/`.h` files → SKILL-c.md, `*.sh`/`*.bash` files → SKILL-bash.md). Skills are concatenated into the system prompt on the first turn so they ride along in cached history. Designed for smaller local models (Qwen3, Gemma) that need explicit language conventions; larger models can usually have them deleted.

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

### Example: a single 24 GB GPU

My setup: AMD Radeon RX 7900 XTX (24 GB VRAM), 128 GB system RAM, llama.cpp, running Qwopus3.6 27B (Q5_K_S) at 80k context per slot. The config below is a llama.cpp router config (each key maps to a `llama-server` flag), curated to the settings that actually matter for codehalter on this hardware. I left out a couple of cosmetic startup flags (`no-mmap`, `no-warmup`).

```ini
# Global defaults for every hosted model:
[*]
n-gpu-layers  = 99      # offload all layers to the GPU
flash-attn    = on      # smaller KV cache, faster attention
cache-type-k  = q8_0    # quantize the KV cache so a long context fits in 24 GB
cache-type-v  = q5_1
cache-reuse   = 64      # reuse a cached prefix down to 64-token chunks
cache-ram     = 65536   # host-RAM prompt cache, 64 GiB ceiling (see below)
batch-size    = 1024    # prefill (prompt-processing) batch
ubatch-size   = 512
jinja         = true    # use the model's chat template (correct tool-call format)
threads       = 8
threads-batch = 8

# The model codehalter points at:
[Qwopus3.6-27B]
# url = https://huggingface.co/Jackrong/Qwopus3.6-27B-v2-MTP-GGUF
model              = /mnt/models/Qwopus3.6-27B-v2-MTP-Q5_K_S.gguf
parallel           = 2         # two slots: codehalter's foreground turn + the background summariser
ctx-size           = 160000    # total across slots, so 80k per slot at parallel = 2
spec-type          = draft-mtp # speculative decoding via the model's MTP head
spec-draft-n-max   = 3
cache-type-k-draft = q8_0      # KV quant for the speculative draft
cache-type-v-draft = q8_0
```

Why these matter for codehalter specifically:

- **`parallel = 2`, `ctx-size = 160000`.** codehalter needs at least two slots so the per-turn background summariser runs without evicting the foreground prefix cache. Two slots split a 160k context into 80k each. A third slot would let the commit drafter run in parallel too, if VRAM allows.
- **`cache-reuse = 64`.** Lets the server reuse a cached prefix even when a chunk changes, down to 64-token granularity. This pairs directly with codehalter keeping the rendered prompt byte-stable turn over turn, so most turns hit the cache instead of reprefilling.
- **`cache-type-k = q8_0` / `cache-type-v = q5_1` / `flash-attn = on`.** Quantizing the KV cache plus flash attention are what let an 80k context sit in 24 GB alongside the Q5 weights.
- **`spec-type = draft-mtp`, `spec-draft-n-max = 3`.** The model ships a Multi-Token-Prediction head; speculative decoding with it drafts a few tokens per step and lifts generation throughput.

#### Does 64 GiB of `cache-ram` make sense for one user?

`cache-ram` is llama.cpp [host-memory prompt caching](https://github.com/ggml-org/llama.cpp/discussions/20574): the server keeps computed KV states of past prompts in system RAM, so a request sharing a prefix with a cached one restores it instead of recomputing. It is a ceiling, not a reservation (default 8192 MiB; `-1` unlimited, `0` off), so it only uses what it actually caches.

The common advice is to skip it for single-user chat, where every prompt is unique and nothing hits the cache. codehalter is the opposite workload: it keeps the prompt byte-stable across turns and runs multiple slots, so prefix reuse is the norm (for example, restoring the foreground prefix after the summariser and commit slots have churned). So enabling it is right here.

64 GiB specifically is more than one user needs, but harmless. A single user's working set is a few states (the growing foreground context plus a couple of side ones), which stays well under 64 GiB, and because the value is only a ceiling it never costs RAM it is not using. On 128 GiB there is no downside to the high cap. If you want the headroom back, something in the low tens of GiB covers one user with margin. Keep it finite rather than `0`: there is an [open bug](https://github.com/ggml-org/llama.cpp/issues/22629) where the ceiling can be ineffective on Linux and OOM the server when caching image prompts, and a finite cap with free RAM (your 64 of 128 GiB) stays on the safe side of it.

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

**Codehalter scaffolds the devcontainer for you.** On the first session in a project without a `.devcontainer/` directory, it prompts for a base distro, **`Alpine / Arch / Debian / Fedora / Ubuntu`**, and writes the chosen `Dockerfile` + `devcontainer.json`:

| Distro | Base image |
|--------|-----------|
| Alpine | `alpine:3.23` |
| Arch | `quay.io/archlinux/archlinux:base-devel` (+ `makepkg`-built `yay` for AUR) |
| Debian | `debian:trixie` |
| Fedora | `fedora:44` |
| Ubuntu | `ubuntu:26.04` |

Each image installs a **minimal** toolchain (git, curl, sudo, openssh, Go, Firefox), creates a passwordless `dev` sudo user, and `go install`s codehalter into `/usr/local/bin`. Language toolchains (gopls, node, clang, rust-analyzer, …) are added per project on demand, codehalter's skills and fix-cards offer to install what a given repo needs rather than baking everything into the base. All five seed a `devcontainer.json` that:

1. Bind-mounts `${localWorkspaceFolder}/.git` (read-write) plus the SSH agent socket, so the agent can run git fully, including `git commit` / `git push`, **but it is instructed to do so only when you explicitly ask**, never on its own initiative or as a side effect. (Want a hard guarantee rather than an instruction? Add `,readonly` to this mount; the agent then falls back to handing you the commit command.)
2. Bind-mounts `${localEnv:HOME}/.gitconfig` read-only into the dev user's home, so the in-container `git` sees your real `user.name` / `user.email` / signing key without any per-project setup.
3. Bind-mounts `~/.config/codehalter` read-only so `settings.toml` is shared across every project.
4. Sets `containerEnv.DEVCONTAINER=true` so codehalter's startup banner shows `Container: devcontainer` and the `run_command` tool is registered.

After the container is built, point Zed at the in-container codehalter binary:

```json
"Codehalter": { "type": "custom", "command": "/usr/local/bin/codehalter", "args": [], "env": {} }
```

If your LLM server runs on the host, change `server` in `~/.config/codehalter/settings.toml` to `http://host.docker.internal:8080`. For an existing language-stack devcontainer (e.g. `mcr.microsoft.com/devcontainers/go:1`), copy the four mounts above into its `devcontainer.json` and `go install github.com/tbocek/codehalter@latest` in its image, no need to start from one of the scaffolded bases.

### Commits and pushes from inside the container

The agent never commits or pushes on its own, only when you explicitly ask. When you do:

1. A background updater (running on `llm[2+]` after every turn) has already written a draft message to `.codehalter/.git_commit` that matches `git diff HEAD`. The agent regenerates it if missing or visibly stale.
2. The agent runs the commit (and push, if you asked) itself, with that message:

   ```sh
   git commit -F .codehalter/.git_commit && git push
   ```

3. It reports the commit subject (and branch, if pushed) in its reply. On your next prompt, codehalter sees the working tree is clean and deletes `.git_commit` so the next round starts fresh.

**Fallback:** on an older container that mounts `.git` read-only, or where `git push` lacks SSH auth, the agent doesn't fight it, it suggests the exact host-side `git commit … && git push` for you to run instead.

`.codehalter/` is gitignored on first bootstrap, so the draft file never accidentally gets staged.

## How it works

1. Zed spawns `codehalter` as a subprocess and communicates via JSON-RPC 2.0 over stdio.
2. On session start, the agent indexes project files, probes image support on the configured LLM, discovers task runners, and reports which of build/test/lint/format are covered.
3. Each prompt runs through the pipeline:
   - **Plan**, the `thinking` LLM analyzes the request with read-only tools and gathers external info via web tools if needed. If the request is ambiguous it asks you a multiple-choice clarification first instead of guessing; if it's a pure question it answers directly with no execution pass (`report_only`). Otherwise it emits an array of `subtasks`, each with its own `verify` recipe, and in Interactive mode you confirm before execution (Execute / Automatic / Cancel).
   - **Work**, for each subtask the `execute` LLM runs a single bounded tool-calling loop where it edits, runs commands, and then runs the verify recipe itself before calling `respond`. Mid-run it can call `submit_plan` to revise the *remaining* subtasks in place (a plan-upsert) when execution reveals the plan needs adjusting, no full replan. A stuck subtask bounces back to replanning after 8 failed rounds, with a 100-iteration backstop against runaways. File edits are shown as diffs and require user approval in Interactive mode.
   - **Replan on failure**, if a subtask bounces, the executor returns without calling `respond`, or a tool exits non-zero, the orchestrator records the failure reason and re-plans (up to 20 replans per prompt). When the same failure recurs (Jaccard over issue words ≥ 0.6), the replan note escalates to "the prior fix didn't work; propose a structurally different approach."
   - **Document**, fires once at the end of a successful prompt: the `thinking` LLM checks against `DOCUMENT.md` whether the change is user-visible (new feature/flag/API/dep, install or config change) and, if so, updates or creates the project README. Routine refactors and bug fixes are skipped.
4. Prior subtasks' assistant replies stay in history so later subtasks have context.
5. Conversation history is persisted to `.codehalter/session_<id>.toml`. After every turn a background goroutine on a free `llm[1+]` slot condenses the whole turn into a seven-section structured note that accumulates in a shadow buffer (itself persisted in the session TOML, so it survives a restart). When message tokens cross the trigger, the buffer folds into the summary and the turns it covers rotate into an archive file: at 80% this happens at the turn boundary (folding every completed turn, keeping nothing verbatim), above 90% it happens mid-turn (keeping the in-flight turn verbatim). With only one `[[llm]]` entry the summariser cannot run on a separate slot, so compaction is skipped (the startup checks warn).

## Compared to forge and pi/coding-agent

Codehalter borrows two ideas: the synthetic `respond` terminal tool from [forge](https://github.com/antoinezambelli/forge), and explicit terminal tools from [pi/coding-agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent), though codehalter declares *which* tools end the loop per phase (a `phasePolicy`: plan exits on `submit_plan`, execute on `respond`) rather than with a per-tool flag. Beyond those, codehalter adds:

- **Devcontainer-first sandboxing posture**: auto-scaffolded `.devcontainer/` from any of five base distros (Alpine/Arch/Debian/Fedora/Ubuntu), host `.gitconfig` mounted read-only, plus a `.codehalter/.git_commit` drafter and a standing instruction never to commit or push on its own, so a commit is always your explicit call (forge and pi don't ship a devcontainer story).
- **Native MCP client**: `.codehalter/mcp.toml` spawns each `[[server]]` as a long-lived child, registers its tools under `<name>__<tool>`, and reconciles on every prompt; gopls is pre-seeded for Go projects.
- **Multi-slot parallel LLM architecture for local servers**: `[[llm]]` array with per-conn `parallel = N` semaphores, breadth-first subagent pinning, and background work (per-turn summariser on `llm[1]`, git-commit drafter on `llm[2+]`) routed off the foreground KV-cache slot so the main session's prefix cache stays warm.
- **Plan → per-subtask self-verifying loop → Document pipeline** built in, not delegated to an extension, the planner decomposes into subtasks each carrying its own verify recipe; the executor runs the recipe itself before declaring done; failed subtasks trigger a replan (≤20 per prompt) with fuzzy duplicate-failure detection (Jaccard over issue words) escalating the note when the same problem recurs; the document pass updates the README only when the change is user-visible.
- **Big-task decomposition** with a once-per-plan Interactive / Automatic / Cancel choice; the same gate re-fires on each replan so the user sees the new approach before it runs.
- **Web tools restricted to the planning phase** (`web_search` via DuckDuckGo + `web_read[_raw]` via headless Firefox) so execution stays offline.
- **ACP / Zed-native** integration with file-edit diffs, permission prompts, and per-session mode switching from Zed's UI.
- **Empty-project bootstrap, stuck-tool-loop detection, dynamic `n_ctx`-aware two-tier compaction trigger (80% at the turn boundary, 90% mid-turn keeping the in-flight turn verbatim), bench harness with isolated devcontainer per test.**

What codehalter does NOT have that those projects emphasise: forge's published 26-scenario eval suite with ablation studies (codehalter has a bench harness but no published eval set), pi's JSONL session trees with in-place branching, pi's 30+ provider adapters and TypeScript-based extensions, pi's `@`-mention fuzzy file references.

## Credits

- [forge](https://github.com/antoinezambelli/forge) by Antoine Zambelli, the synthetic `respond` terminal tool used in execute/subagent is borrowed from forge's playbook for keeping small local models inside structured tool-calling grammar.
- [pi/coding-agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent) by earendil-works, codehalter's explicit terminal tools (which tool ends the loop is declared per phase via `phasePolicy`) are inspired by pi's `terminate: true` tool-result signal.
- Code harness idea exchanging with Clemens
