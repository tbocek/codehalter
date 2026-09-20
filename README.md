# Codehalter

An [ACP](https://agentclientprotocol.com)-compatible AI coding agent that connects [Zed](https://zed.dev), and any other Agent Client Protocol editor, to any OpenAI-compatible LLM server. **Devcontainer-first and local-LLM-first**: built around running inside a devcontainer, **Alpine, Arch, Debian, Fedora, or Ubuntu**, against [llama.cpp](https://github.com/ggml-org/llama.cpp) (or Ollama / vLLM) on the host. Your real `.git` and identity are available inside the container, but the agent is **instructed never to commit, push, or rewrite history on its own**, only when you explicitly ask. Targets small-to-medium repositories.

## Highlights

- **Small and auditable**: about 16k lines of Go (SLOC; ~24k including its thorough comments), two small dependencies, one static binary.
- **Local-LLM-first**: designed and tuned for small/medium models on your own [llama.cpp](https://github.com/ggml-org/llama.cpp) / Ollama / vLLM box, and works with hosted OpenAI-compatible APIs too.
- **Spec-driven development (`/spec`)**: point it at a directory of specification documents and it implements them one item per round, unattended, for as long as you let it run. An item is done only when a test *names* it and the whole suite passes; each one is its own commit. The bookkeeping is code, not model memory: a ledger of items, a coverage scan of your tests, and a fingerprint of the spec text each item was built from. **The spec is the master** — edit a section later and the next run redoes exactly that item, showing the model the `git diff` of your edit; delete a section and it offers to remove the code and tests with it. The spec can be any markdown: headings become items, no id scheme required. First run just asks where the spec is and what to build from it.
- **One protocol, two front ends**: codehalter speaks the [Agent Client Protocol](https://agentclientprotocol.com) over stdio, so the editor is the UI. Native in [Zed](https://zed.dev); usable from JetBrains and VS Code through their ACP integrations. No editor at hand? `codehalter --cli` runs the client half in-process over a real pipe and gives you an inline TUI in the terminal, same agent, same protocol, no extra dependencies.
- **Prefix-cache consistency as a first principle**: the rendered prompt stays byte- and position-stable turn over turn, so the server's KV cache keeps ~90%+ warm. The *only* deliberate cache break is a new MCP server's tools coming online; skills added mid-session ride as user messages until the next compaction folds them into the cached system prompt. At session open codehalter **pre-warms** the cache with a background 1-token call, so your first turn skips the prompt-processing wait (`prewarm = false` to disable), and between turns, plus while a long tool runs, it **keeps that prefix warm** with the same call: one measured session lost a 174k-token prefix to a reclaimed slot during a 2m59s test run, about six minutes of prompt processing to re-read (`keep_warm = "off"` to disable).
- **Smart background compaction**: **after every turn**, a *separate* LLM slot condenses the whole turn into a structured note in the background while you read the reply, so the notes are ready before they're needed. A **large turn** is one user prompt and everything codehalter does to answer it; a **small turn** is a single model call plus its tool results (no user involved). Compaction is **purely reactive** — no token estimate. It's triggered by the server's context-overflow **400** and escalates: first keep the **unfinished small turn** plus the most recent **~10k tokens of completed small turns** (folding everything older into the summary); if that still overflows, keep only the **unfinished small turn**. Then retry. With ≥2 parallel LLM slots the summariser runs beside the foreground turn; on a **single slot** it appends its instruction to the conversation itself (a prefix-extension), reusing the warm cache instead of evicting it — it just serialises briefly with your next prompt, so `parallel = 1` works fully.
- **Lightweight plan → execute → document loop**: each prompt is planned, executed with per-subtask self-verification, and documented only when the change is user-visible, no heavyweight agent framework.
- **Devcontainer-first**: on the first session codehalter scaffolds a throwaway devcontainer from your pick of **Alpine, Arch, Debian, Fedora, or Ubuntu**, installs itself via `curl` + the toolchain, and runs entirely inside it. Edits and build/test commands are sandboxed to the container. It works with your real `.git`, but is **instructed never to commit, push, or rewrite history on its own**, that stays your call.
- **Project onboarding**: detects your stack, loads the skills for it and for your container, and bootstraps empty repositories. When a formatter is missing, its style is unpinned, or the project has no `AGENT.md`, it offers a one-click **"fix it?" card** rather than just failing.
- **Reusable prompt macros**: type `/<name> <args>` to expand a `TEMPLATE-<name>.md` into a full prompt. Ships **`/grill-me`** (the agent interviews you about a plan until the whole design tree is resolved), **`/commit`** (drafts a commit message from your diff plus the conversation's intent, then commits — never pushes unless asked), **`/clean`** (deletes session log files from `.codehalter/`), and **`/settings`** (names the `settings.toml` in force, the ones it shadows, and re-probes every configured model); drop your own `TEMPLATE-*.md` for project- or team-specific workflows.
- **Steer a running turn**: type while it works and your message is queued, not a cancellation: the turn picks it up before its next model call and carries on, so "also update the README" costs one round instead of the whole turn (a `/spec` round can be an hour). It arrives as an ordinary user message, which is an append, so the prefix cache is untouched. The editor's stop button still interrupts.
- **Commit on request, message included**: ask for a commit (or type `/commit`) and the agent drafts the message from `git status` + `git diff HEAD`, using the conversation for the *why*, writes it to `.codehalter/.git_commit`, and runs `git commit -F …` itself (or hands you the command). It never commits or pushes on its own.
- **Per-turn usage stats**: every reply ends with a server-measured line such as `Done in 1m22s · 11k prompt + 2.1k gen (122k sent, 91% cached) · 388 pp/s · 39 tg/s` (evaluated vs cached tokens, prompt/generation throughput), and Zed's context-window ring tracks the session via ACP `usage_update`.
- **External-change detection**: codehalter remembers the exact bytes it wrote to each file, so when the next read finds something else — an editor's format-on-save is the usual cause — the tool result says so instead of leaving the model to work out on its own why its `old_text` stopped matching.

## Features

- **Devcontainer scaffolding (5 base distros)**: the first session prompts `Alpine / Arch / Debian / Fedora / Ubuntu` and writes `.devcontainer/Dockerfile` + `devcontainer.json` for the chosen base. Each image installs a minimal toolchain (git, curl, sudo, Firefox) and installs codehalter into `~/.local/bin` (added to the container `PATH`) under a passwordless `dev` sudo user; for community-repo installs (the last resort in the skills' install order) the Arch variant adds the `yay` AUR helper, Fedora the COPR plugin, Ubuntu `add-apt-repository`, and Debian pre-enables backports. Language toolchains (go, node, clang, …) are added per project on demand. The seeded `devcontainer.json` bind-mounts the host's `.gitconfig` (so the agent's git sees your identity) and the project's `.git` directory (read-write), the agent works with real git but is **instructed never to commit, push, or rewrite history on its own**, only when you explicitly ask.
- **Commit workflow**: the agent never commits or pushes on its own. When you ask (or type `/commit`), it drafts the message from `git status --porcelain` + `git diff HEAD` with the conversation as intent, writes it to `.codehalter/.git_commit`, and runs `git commit -F .codehalter/.git_commit` (and `git push` if asked) itself, falling back to handing you the host command on an older read-only container.
- **Bring-your-own MCP**: `.codehalter/mcp.toml` declares each `[[server]]` as a child process to spawn (`name`, `command`, `args`, `env`); tools it advertises are registered as `<name>__<tool>`. The seeded file declares no servers. Reconciled at turn boundaries (never mid-turn, since the tool list is part of the prompt prefix): add or remove a `[[server]]` and the change applies without restarting, flushed as soon as the running turn ends; failed startups surface as a red tool-call card and mtime-gate so they don't re-emit until you edit the file.
- **One slot is enough**: one `[[llm]]` entry works fully, even at `parallel = 1`: the background summariser rides the foreground context as a prefix-extension, so the KV cache stays warm. `llm[0]` owns the foreground KV cache; a second entry marked `purpose = "summary"` hosts the per-turn summariser and `web_read`'s page reader instead (falling back to `llm[0]` when it is unreachable or keeps failing). Each entry's `parallel = N` caps concurrent requests to match its server's capacity.
- **Plan → Execute → Verify → Document pipeline**: every prompt is routed through a planning pass, an execution pass, and a self-verification pass. Failed verifications re-plan with the failure context; up to 20 attempts before giving up, with fuzzy duplicate-failure detection (Jaccard similarity over issue words) that bails early when the model keeps rephrasing the same problem. Turns that wrote files end with a documentation pass that updates (or creates) the README when the change is user-visible.
- **Big-task decomposition**: when the planner splits a request into multiple subtasks, the numbered plan is posted to the thread and execution starts, with no "Execute?" gate: the devcontainer is the approval, and every subtask is confined to it. A mid-run revision is re-posted as *Plan updated*, and each subtask gets its own full plan → execute → verify cycle.
- **Agentic tool loop**: the LLM can read, edit, search, and run tasks iteratively with file edits surfaced as diffs.
- **Built-in tools**: `read_file`, `continue_read` (paging through a large read), `write_file`, `edit_file`, `list_files`, `search_text` (literal or regex, with `context` lines around each hit like `grep -C`, and `path` may name a single file, so a partial `read_file` points there for "less" and at `continue_read` for "more"), `run_command` (in-container only), `run_background` (a dev server, or a long experiment: the process outlives the tool call and the turn, the chat stays usable, and codehalter reports the exit code and last output by itself when it finishes; mid-conversation that is one line after the running turn ends, never an interruption, and when idle the agent looks at the result and tells you; jobs die with the session), `view_image` (when the active model has vision), `ask_user` (a pick-one list, a free-text box, or both, sent as an ACP elicitation form), `web_search`, `web_read` (with a `question`, a separate reader answers it from the page on the background connection, so the conversation carries the answer instead of the page; without one, the raw page text), `respond`, `submit_plan`. An output too large for the context is cut to head and tail with a hint on how to narrow the call; there is no tool to page a cached output. Plus any MCP-server tool, registered as `<server-name>__<tool>`.
- **Synthetic `respond` terminal tool**: the execute phase exposes a `respond(message)` tool that captures the model's final user-facing reply. The turn ends only when `respond` is called, so small local models stay inside the tool-calling grammar they handle best instead of having to decide "tool call vs free text" at exit time. Inspired by [forge](https://github.com/antoinezambelli/forge)'s `respond_tool`.
- **Stuck-tool-loop detection**: a model that re-issues the same call gets nudged once, then the next attempt escalates to a thinking-sampler retry instead of spinning indefinitely.
- **Skills for what the model cannot know**: there are no per-language skills. A capable model knows its languages; the skills cover the environment instead. `SKILL-base.md` always applies: containerised execution, the install order and install/persist policy, toolchain rules (patch vs minor version gaps, one package manager per JS project), how to read a "failed on line N" wrapper error, how to pin a formatter config that matches the code, plus the house code style (early return over nesting, named constants, short names, no bool parameters, comments that say why, private by default) and reply tone (no praise, no superlatives). One `SKILL-<distro>.md` follows the container's `/etc/os-release` (Alpine/Arch/Debian/Fedora/Ubuntu), `SKILL-justfile.md` follows a `justfile` in the tree, and `SKILL-layout.md` (check a rendered page with `screenshot`, measure before editing) applies when the project has stylesheets or HTML. The rule that the agent never commits, pushes or rewrites the project's git on its own, only when you ask, lives in `EXECUTE.md`. They are read out of the binary, never copied into your project, so a skill improved upstream reaches every project at once; a file of the same name in `.codehalter/` overrides one, and a `SKILL-*.md` of your own is added to the set. Every applicable skill rides in the system prompt (they total a few KB, cached once). One that becomes applicable mid-session arrives as a user message until the next compaction folds it into the prompt, so the cached prefix never moves.
- **`AGENT.md` in the system prompt**: a project brief (`AGENT.md`, `AGENTS.md` and the lowercase spellings, project root only) is folded into the cached prefix at session start, so every session begins already knowing what the project is, what it is built with and how it is tested, instead of spending its first turns finding out. A project without one is offered a card that writes it, but only once there is something to read (five files, ignoring `.git`, `.codehalter` and `node_modules`): the model reads the tree, puts its draft understanding to you in a single `ask_user` box to correct, and saves the answer. That check is live, so a directory you scaffold during a session is asked at the end of that turn rather than at the next session. It holds only what stays true between sessions, so nothing in it expires, and the prompt tells the model to update it in the same task that makes one of its lines wrong. There is no task-runner detection to go with it: the model reads your `justfile` like any other file and runs `just test` through `run_command` like any other command.
- **Empty-project bootstrap**: fresh directories are flagged; the LLM asks on the first turn which language/runner to scaffold before writing anything.
- **Proactive fix / setup cards**: before each turn codehalter audits the environment and, when something's missing, shows a single yes/no card instead of failing mid-task: every formatter the container is missing is folded into **one** card whose accepted turn installs *and* wires all of it, plus separate cards for an `mcp.toml` parse/start error and for a project with no `AGENT.md`. A project with no formatter config gets its own card, offered once per **project** (the ones that ask for work a project only ever needs once are recorded in `.codehalter/checks.done`, so declining is not re-asked at every session start; delete a line there to be asked again): the accepted turn measures the style the code is already written in (indent width, quotes, semicolons, line width), writes the config that encodes exactly that, and formats the tree in one formatting-only commit. This matters beyond tidiness, since an editor's format-on-save reindenting a file after a write invalidates the text the model just read and costs it failed edits. Set `format_config = false` for a project that deliberately has none. Accepting a card dispatches a synthetic fix turn; declining (or editing the file) dismisses it without re-nagging.
- **Slash-command prompt macros**: a message of the form `/<name> <args>` expands the matching `TEMPLATE-<name>.md` (your copy in `.codehalter/`, else the embedded default) into a full prompt and runs it as a normal turn. Ships `/grill-me` (relentless plan interview) and `/clean` (deletes session log files from `.codehalter/`); macros that take an argument use `{{}}` and stop with a note if it's missing. `/clean` and `/settings` are code-level commands rather than templates: `/settings` prints which `settings.toml` is being read and which candidate it shadows, then re-probes every `[[llm]]` so "which models work" is answered live rather than from the probe cached at session start. Drop your own `TEMPLATE-*.md` to add reusable workflows.
- **Spec loop (`/spec`)**: plain `/spec` in a project that has no loop yet asks two questions (which directory holds the spec, then what to build from it and where, with a suggestion read off the spec's own entry page), so `/spec <spec-dir> <out-dir> [technology]` is the shortcut rather than the entry point. Either way it implements a specification item by item until every requirement has a passing test. The bookkeeping is code, not model memory:
  - **Ledger:** requirement ids (flows `F2.3`, parameters `P.policy.x`, tools `tool:x`, patterns configurable) are read from the spec by regex, and every headed section that defines no id (a file format, a screen, a rules list) becomes an item of its own.
  - **Done:** an item counts as done when a test under the output dir names it (`f2_3_…`, or the id in a comment) and the project's test command passes. Codehalter checks both after every round and commits each item that passes.
  - **Per round:** the model gets only its item's section, the rows of the parameters it cites and the sections it links to.
  - **Pre-flight:** the loop refuses to start while the spec has open `REVIEW`/`TBD` markers or while git would ignore the files the rewrite creates.
  - **Guards:** the spec dir is read-only to the file tools while the loop runs. Under Autopilot a planner question blocks its item instead of being guessed.
  - **The spec is the master:** every finished item is recorded with a fingerprint of the spec text it was built from. Edit that section later and the next run redoes the item, showing the model the `git diff` of the edit rather than the section again; delete the section and the run offers to delete its code and tests too. A section that moves to another file or is retitled keeps its record instead of looking like one deletion plus one new item.
  - **Stop and resume:** the loop stops after three blocked items in a row. `/spec` resumes and `/spec status` shows coverage per chapter plus anything that changed in the spec since; state lives in `.codehalter/spec.toml`.
- **Web tools**: `web_search` (DuckDuckGo) returns a list of result titles/URLs/snippets for the LLM to triage; `web_read` then opens a chosen URL in headless Firefox and answers a standalone `question` about it (omit the question for verbatim text). Restricted to the planning phase so execution stays offline.
- **Image support**: when the active LLM advertises vision, prompt images are passed through as OpenAI-style content blocks.
- **Session persistence**: conversations are saved as TOML files under `.codehalter/` and can be resumed across editor restarts.
- **History compression**: older turns are summarised to stay within token budgets. The unit of work is a **large turn** (one user-or-synthetic prompt plus every assistant step and tool call that answers it); within it, each model call plus its tool results is a **small turn**. After every large turn a background goroutine condenses the **whole turn** (not just the last reply) into a seven-section structured note (Goal / Constraints / Tasks / Progress / Decisions / Next Steps / Critical Context) on the `purpose = "summary"` slot. The notes accumulate one per completed large turn in a shadow buffer that is **persisted in the session TOML**, so they survive a restart. Compaction is **purely reactive — no token estimate at all**. It is driven by the server's context-overflow **400** and escalates in two steps. Step 1: keep the **unfinished small turn** plus the most recent **~10k tokens of completed small turns** verbatim, and fold everything older — the completed large turns (instant, from their ready shadow notes) plus a synchronous summary of the older completed small turns — into the rolling summary, rotating it into an archive file; retry. A small in-flight turn (under the budget) is kept whole. Step 2 (only if the retry still 400s — the unfinished small turn alone is huge): keep only the **unfinished small turn**, retry. Each step strictly shrinks the context, so it terminates: once only the unfinished small turn is left, the 400 surfaces as a normal failure. If the summariser's call fails, the slice still leaves a clipped raw note, so nothing is rotated out unsummarised. With ≥2 slots the summariser runs on its own slot; on a single slot (`parallel = 1`, one `[[llm]]` entry) it extends the foreground context as a prefix-extension — cache-safe, just briefly serialised with your next prompt.
- **Standalone terminal client (`--cli`)**: the same agent driven from a terminal instead of an editor. `--cli` starts the ACP *client* half in-process and connects it to the agent over an `io.Pipe` pair, so the wire carries the identical JSON-RPC an editor would send, and there is no shortcut path that could drift from it. The TUI is **inline**: the transcript is append-only ordinary output (scrollback, selection and copy all keep working, no alternate screen) while a redrawn live region at the bottom holds the streaming reply, the current plan, in-flight tool cards and the tail of a running command. It advertises the `terminal` and `elicitation` capabilities (so `run_command` and `ask_user` work) and deliberately not `fs`, so the agent reads and writes the disk directly. `Ctrl+C` cancels the running turn rather than the process, `Ctrl+D` or `/quit` leaves, and logs are redirected to `.codehalter/cli.log` so they never fight the screen. Piped or redirected output carries no escape sequences, so it doubles as a plain transcript. Started on the host in a project that has a `.devcontainer/devcontainer.json`, it also **starts that container itself**: the config is translated into a compose file on the fly (no devcontainer CLI, no new dependency), `docker` or `podman` `compose` brings it up, and the CLI re-runs itself inside with your terminal attached, printing which runtime it uses and which folder is mounted where. A key it cannot translate faithfully (`features`, lifecycle commands) is refused by name with the `devcontainer up` command to use instead, never silently dropped. See [Standalone CLI](#standalone-cli).
- **Update check that covers the container too**: at most once a day, `codehalter --cli` asks the GitHub releases API whether a newer release exists and, if so, asks **you** before doing anything: yes downloads the binary for your platform, replaces the running one and restarts into it. The answer and the resolved tag then travel into the devcontainer, so the copy in there (baked into the image, otherwise stale until the next rebuild) updates itself in the same run without a second question or a second API call. A run with nobody watching (`-p`, redirected stdin) is told rather than asked. An **editor session installs it in place**: the banner names the release and a card offers to install it into the running container, with no shell in there and no image rebuild. The thread you are in finishes on the old binary (its inode survives the replace) and the next Agent Thread starts on the new one. A downloaded binary must run and report the release tag that was asked for before it is allowed to replace anything, so a truncated file or a wrong-architecture build cannot break the install. Off with `update_check = false` or `CODEHALTER_UPDATE=skip`. See [Updating](#updating).
- **Two modes**: *Interactive* (ask before setup steps and anything reaching outside the container) and *Autopilot* (auto-answer those prompts too, no interruption). Selectable per-session from the Zed mode picker. Work inside the container is never gated in either mode.
- **Configurable LLM endpoints**: different roles (`thinking`, `execute`) can point at different models or servers.

## Prerequisites

- Go 1.26+ — only to build from source; [Install](#install) uses prebuilt binaries and needs no toolchain
- An OpenAI-compatible LLM server (e.g. [llama.cpp](https://github.com/ggml-org/llama.cpp), [Ollama](https://ollama.com), [vLLM](https://github.com/vllm-project/vllm))
- Firefox, if you want `web_search` / `web_read` to work

## Install

Prebuilt binaries — no toolchain needed. The script detects your OS/arch, downloads the matching binary from the [latest release](https://github.com/tbocek/codehalter/releases/latest), and installs it.

**Linux & macOS** (Intel and Apple Silicon) — installs to `/usr/local/bin`, or `~/.local/bin` when that isn't writable:

```sh
curl -sL https://raw.githubusercontent.com/tbocek/codehalter/main/install.sh | bash
```

**Windows** — codehalter runs inside a Linux devcontainer, so install it under **WSL**: run `wsl --install` (once), open your WSL distro, and run the same Linux command above inside it. Native Windows isn't supported — the devcontainer's host-path bind mounts resolve `${localEnv:HOME}`, which WSL provides and native Windows does not.

### Updating

Releases are numbered `v1`, `v2`, ..., and an installed binary knows which one it is (`codehalter --version`; a build from source says `dev` and is left alone by everything below).

`codehalter --cli` checks at most once a day whether a newer release exists, and asks before doing anything about it:

```
codehalter v42 is available (running v41). Update now? [Y/n]
```

Answer yes and it downloads the binary for your platform, replaces the one you are running, and restarts into it, so the session you asked for starts on the new version. There is nothing to reinstall and no package manager involved. **A run that nobody is watching is told rather than asked**: with `-p` or with stdin redirected it prints the same line, adds `Run codehalter --update to install it`, and carries on.

**One answer covers the container too.** When the launcher starts a devcontainer for you, it hands the resolved release tag and your answer to the copy of codehalter inside it, so that one updates itself in the same run without asking a second question or spending a second API call. This matters because the container's binary was baked into the image and does not otherwise change until the image is rebuilt.

The other ways in:

| | |
|---|---|
| `codehalter --update` | check and install, no question asked, for a run nobody is watching |
| `codehalter --version` | the release tag this binary was built from |
| `update_check = false` | in `settings.toml`: never contact GitHub about releases |
| `CODEHALTER_UPDATE=skip` | the same for one run (CI); `=yes` updates without asking |

The check is one unauthenticated call to the GitHub releases API, cached in `~/.cache/codehalter/update.json` for a day, so an unreachable network, a rate limit or an unwritable install directory each cost a line of output and nothing else: codehalter starts on the version you already have. A download is only allowed to replace your binary once it has been run and has reported the release tag that was asked for, so a truncated file, an error page or a wrong-architecture build never lands on top of a working install.

## Build

To build from source instead (requires Go 1.26+):

```sh
just build
```

Or directly:

```sh
go build -o codehalter .
```

## Configuration

Codehalter looks for settings in two places, in this order:

1. `<project>/.codehalter/settings.toml` (project-local, preferred)
2. `~/.config/codehalter/settings.toml` (global fallback, used across every project that has no local file)

On first run with neither file present, codehalter prompts to write a commented skeleton into `<project>/.codehalter/settings.toml`. Edit it, then move it to `~/.config/codehalter/` to share across projects (a project-local file always wins over the global one, so delete it if you want the project to use the shared config).

### Interactive setup

Alternatively, run the `--setup` flag to configure the LLM connection interactively:

```sh
codehalter --setup
```

This walks through server URL, model name, and API key (optional), validates the connection, and writes `~/.config/codehalter/settings.toml` automatically.

### Example settings

One `[[llm]]` array. Order matters: `llm[0]` is the main connection, `llm[1+]` are extras.

- `llm[0]`, owns the foreground session's KV cache, so its prefix stays warm across turns. Every plan/execute/verify/document call on the main session lands here. Typically a smaller/faster model.
- `llm[1]`, optional, marked `purpose = "summary"`: hosts the per-turn structured summariser and `web_read`'s page reader off `llm[0]`, so its prefix cache isn't evicted. Without it the summariser extends `llm[0]`'s own context as a prefix-extension — cache-safe, just serialised.

`parallel = N` per entry caps how many concurrent LLM calls that entry accepts. The token is held for the duration of one LLM round-trip only, so between calls the slot is free for another caller, and pool size 1 simply serialises everything.

`params` is forwarded verbatim as the OpenAI request's extra body, put samplers and any model-specific knobs (`enable_thinking`, `reasoning_mode`, …) there. Core fields (`model`, `messages`, `stream`, `tools`) always win over `params`.

Keep `params_thinking` and `params_execute` identical in everything that is **not** a sampler. Samplers (`temperature`, `top_p`, `max_tokens`, …) never enter the KV cache key, so they can differ per role for free. Anything the server feeds to its chat template does not: `chat_template_kwargs` changes the rendering, and the server keeps a separate prompt state per rendering, so a role switch mid-turn (plan → execute) re-evaluates everything the other role appended since this one last ran ([why](#why-a-role-switch-can-cost-a-full-re-prefill)). That is why codehalter turns reasoning off for the execute phase by appending an already-closed `<think></think>` for the model to continue, which is a pure extension of the prompt, instead of with `enable_thinking = false`. `logy` flags this class of change as `⚠ params CHANGED`.

`server` is the base URL of your OpenAI-compatible server, the host root only (e.g. `http://localhost:8080`). codehalter appends the API paths itself: `/v1/chat/completions` for completions, plus `/v1/models` and `/props` for probing. Don't include a path.

Optional **top-level keys** (they must appear above the `[[llm]]` tables): `prewarm = true` (default on) fires a background 1-token call at session open so the server tokenizes and caches the prompt prefix before your first message; `keep_warm = "3m"` (default) re-sends that 1-token call while the session is idle and while a long tool runs, so an idle slot is not reclaimed with your prefix in it, and gives up 30 minutes after the last real call (`"off"` for a hosted endpoint, which caches on its own and bills per request); `format_config = false` stops the formatter-config card; `update_check = false` stops the daily release check.

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
purpose = "summary"
parallel = 1
params_thinking = { temperature = 1.0, top_p = 0.95, top_k = 20, min_p = 0.0 }
params_execute  = { temperature = 0.6, top_p = 0.8,  top_k = 20, min_p = 0.0 }
```

> **Recommendation:** a single `[[llm]]` entry with `parallel = 1` works fully — the per-turn summariser rides the foreground prefix, so nothing is evicted and the one slot gets the whole context window. Use `parallel = 2` (or a second entry marked `purpose = "summary"`) if you want the summariser to run beside your typing instead of briefly serialising with it.

| Role | Purpose | Suggested temperature |
|------|---------|------------------------|
| `thinking` | Planning, document, history compaction | ~1.0 (diverse hypotheses, edge-case exploration) |
| `execute` | Tool loop, verify, web-page summarisation | ~0.6 (precise, faithful) |

### Prompt files

The phase prompts live in the binary. Nothing is copied into your project, so a prompt improved upstream reaches every project on its next session instead of only the ones created after the change:

| File | Role |
|------|------|
| `PLAN.md` | Planning-phase instructions (clarity check, info retrieval, subtask JSON schema with per-subtask verify recipe) |
| `EXECUTE.md` | Execution-phase directives prepended to the user message; instructs the executor to run the verify recipe itself before calling `respond` |
| `DOCUMENT.md` | Decides when the change is user-visible enough to update the README, then edits it minimally |
| `SUMMARISE.md` | The per-turn note format the background summariser distils each turn into (feeds compaction) |

Plus `SPEC.md`, `SPEC-SETUP.md` and `SPEC-REMOVE.md` for the three kinds of `/spec` round, and the skills: `SKILL-base.md` (container, install order, toolchains, formatter config), one `SKILL-<distro>.md` for the container's OS, `SKILL-justfile.md` when the project has a `justfile`, and `SKILL-layout.md` (check a rendered page with `screenshot`, measure before editing) when it has stylesheets or HTML templates. There are no per-language skills: a capable model knows its languages, and the skills cover what it cannot know.

**To change one for a project, put a file of that name in `.codehalter/`** and it replaces the built-in text; delete it and the built-in comes back. That is the only reason one of these names exists on disk, so an override is something you chose rather than a copy you forgot. Emptying `PLAN.md` is still how you turn planning off. A `SKILL-<anything>.md` of your own is added to the set rather than replacing anything, and the same rule covers `TEMPLATE-*.md` macros. Only `settings.toml` and `mcp.toml` are written for you, because they are configuration rather than prompts and their comments are the schema.

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
parallel           = 2         # two slots: foreground turn + concurrent summariser (1 works too, see below)
ctx-size           = 160000    # total across slots, so 80k per slot at parallel = 2
spec-type          = draft-mtp # speculative decoding via the model's MTP head
spec-draft-n-max   = 3
cache-type-k-draft = q8_0      # KV quant for the speculative draft
cache-type-v-draft = q8_0
```

Why these matter for codehalter specifically:

- **`parallel = 2`, `ctx-size = 160000`.** Two slots let the per-turn background summariser run beside the foreground turn; they split a 160k context into 80k each. `parallel = 1` also works: the summariser then extends the foreground prefix (cache-safe, briefly serialised) and the single slot gets the full 160k.
- **`cache-reuse = 64`.** Lets the server reuse a cached prefix even when a chunk changes, down to 64-token granularity. This pairs directly with codehalter keeping the rendered prompt byte-stable turn over turn, so most turns hit the cache instead of reprefilling.
- **`cache-type-k = q8_0` / `cache-type-v = q5_1` / `flash-attn = on`.** Quantizing the KV cache plus flash attention are what let an 80k context sit in 24 GB alongside the Q5 weights.
- **`spec-type = draft-mtp`, `spec-draft-n-max = 3`.** The model ships a Multi-Token-Prediction head; speculative decoding with it drafts a few tokens per step and lifts generation throughput.

#### Does 64 GiB of `cache-ram` make sense for one user?

`cache-ram` is llama.cpp [host-memory prompt caching](https://github.com/ggml-org/llama.cpp/discussions/20574): the server keeps computed KV states of past prompts in system RAM, so a request sharing a prefix with a cached one restores it instead of recomputing. It is a ceiling, not a reservation (default 8192 MiB; `-1` unlimited, `0` off), so it only uses what it actually caches.

The common advice is to skip it for single-user chat, where every prompt is unique and nothing hits the cache. codehalter is the opposite workload: it keeps the prompt byte-stable across turns, so prefix reuse is the norm (for example, restoring the foreground prefix after another request churned the slot). So enabling it is right here.

64 GiB specifically is more than one user needs, but harmless. A single user's working set is a few states (the growing foreground context plus a couple of side ones), which stays well under 64 GiB, and because the value is only a ceiling it never costs RAM it is not using. On 128 GiB there is no downside to the high cap. If you want the headroom back, something in the low tens of GiB covers one user with margin. Keep it finite rather than `0`: there is an [open bug](https://github.com/ggml-org/llama.cpp/issues/22629) where the ceiling can be ineffective on Linux and OOM the server when caching image prompts, and a finite cap with free RAM (your 64 of 128 GiB) stays on the safe side of it.

## Standalone CLI

No editor required. `--cli` runs codehalter's own ACP client in the same process and drives the agent over a pipe, so what reaches the agent is byte-for-byte the protocol Zed would speak:

```
codehalter --cli [--cwd DIR] [--resume [SESSION_ID]] [-p PROMPT] [--rebuild]

  --cli               run the standalone terminal client instead of an ACP server
  --cwd DIR           project directory (default: current directory)
  --resume [ID]       continue a stored session; without an ID, the most recent
                      one for this directory
  -p PROMPT           run one turn and exit, instead of opening a prompt. The
                      exit status is 0 only when the turn finished normally,
                      so a script can branch on it.
  --rebuild           rebuild the devcontainer image before starting it

Other flags: --version prints the release tag, --update installs the newest
release over this binary, --setup reconfigures the LLM connection.
```

Like every other codehalter run this one is **devcontainer-first**: it refuses to start outside a container. Starting that container is its own job, though.

**The launcher.** Run `codehalter --cli` on the host in a project that has `.devcontainer/devcontainer.json` and it does not start the agent at all. It reads that file, translates it into a compose file, brings it up with `docker compose` (or `podman compose`), and re-runs itself inside the container with your terminal attached:

```
$ codehalter --cli
codehalter cli
  no container here, so docker compose starts one: project myproject-1c881a, from Dockerfile
  /home/you/src/myproject → /workspaces/myproject  (bind mount, read-write: edits inside are edits here)
  building the image first, which takes a while; later runs reuse it
```

Inside a container it skips all that and just starts, so the same command works in both places. `Ctrl+C` still belongs to the agent inside; the launcher ignores it and passes the child's exit status back, which keeps `-p` scriptable from the host.

The container is left running when you leave, so the next `codehalter --cli` is one `compose exec` away, a tenth of a second rather than a start. Set `"shutdownAction": "stopContainer"` in the devcontainer file to have it stopped instead. Compose recreates the container by itself when the translated service changes, an image built from a `Dockerfile` is rebuilt when that `Dockerfile` changes, and `--rebuild` forces a rebuild.

This is a launcher, not a devcontainer implementation. It reads the config from any of the three places the spec allows (`.devcontainer/devcontainer.json`, `.devcontainer.json`, or a single `.devcontainer/<name>/devcontainer.json`) and translates the keys that have a compose equivalent (`image`, `build`, `runArgs`, `containerEnv`, `remoteEnv`, `containerUser`, `remoteUser`, `mounts`, `workspaceFolder`, `workspaceMount`, `forwardPorts`, `overrideCommand`, `init`, `privileged`, `capAdd`, `securityOpt`, `shutdownAction`, and a `dockerComposeFile` project it hands through). It expands `${localEnv:NAME}`, `${localWorkspaceFolder}`, `${containerWorkspaceFolder}`, `${devcontainerId}` and their basename forms, plus `${containerEnv:NAME}` in `remoteEnv`, which is read out of the container once it is up. Forwarded ports are published on `127.0.0.1`, as the editors do. Editor-only keys are ignored **without** being expanded, so the `${workspaceFolder}` and `${env:HOME}` that live in `customizations.vscode.settings` are left alone. Anything else it refuses **by name**, rather than quietly starting a container that is not the one you asked for:

```
$ codehalter --cli
.devcontainer/devcontainer.json uses features, postCreateCommand, which the built-in launcher does not implement

Start it with the devcontainer CLI instead:
  npm i -g @devcontainers/cli
  devcontainer up   --workspace-folder /home/you/src/myproject
  devcontainer exec --workspace-folder /home/you/src/myproject codehalter --cli
```

Four things differ from `devcontainer up` even for a config it accepts, and the first two print a `note:` line when they apply:

- `updateRemoteUserUID` is not honoured: the container user keeps the uid baked into the image, so files written in the workspace can come back owned by someone else.
- `userEnvProbe` is not honoured: commands run with the image's environment, not with one probed from a login shell.
- `${devcontainerId}` is derived from the workspace path, so it does not match the id the devcontainer CLI computes for the same project.
- `shutdownAction` defaults to leaving the container running, where the spec's default is to stop it. Write it down explicitly to get the other behaviour.

None of this is required: `devcontainer up` yourself and run `devcontainer exec --workspace-folder . codehalter --cli`, and the launcher never runs.

The interface is an **inline** TUI, not a full-screen one. The transcript is plain append-only output, so your scrollback, mouse selection and copy behave exactly as they do for any other command, while a small live region at the bottom is redrawn in place with the streaming reply, the current plan, running tool cards and the last few lines of any command in flight. Redirect the output to a file and the escape sequences vanish, leaving a readable log.

Client-side commands:

| Command | Effect |
|---------|--------|
| `/help` | client commands, then the agent's own commands and `TEMPLATE-*.md` macros |
| `/mode [name]` | show or switch the session mode (`Interactive` / `Autopilot`) |
| `/sessions` | stored sessions for this directory, newest first |
| `/new` | start a fresh session in this directory |
| `/resume [id]` | switch to a stored session, or pick one from the list |
| `/cwd` | project directory and current session id |
| `/quit`, `/exit` | leave (same as `Ctrl+D`) |

Anything else starting with `/` is handed to the agent, so `/commit`, `/clean`, `/settings`, `/grill-me` and your own macros work unchanged.

`Ctrl+C` cancels the **turn**, not the process: the agent stops where it is and reports why, so the conversation survives an interrupt. While idle it says so instead of exiting; a second `Ctrl+C` within two seconds does leave, as does `Ctrl+D`. Permission cards and `ask_user` forms are rendered as numbered lists, with an empty line to dismiss. Diagnostics go to `.codehalter/cli.log` instead of the screen (`tail -f` it in another pane when you want to watch the protocol); the file is truncated on every run.

**One turn, no prompt.** `-p` runs a single turn and exits, which is the form to reach for in a script, a git hook or CI:

```
codehalter --cli -p "run the tests and fix what fails" || echo "turn did not finish"
```

It prints the same transcript, opens no prompt row, and exits `0` only when the turn ended normally. A cancellation, a refusal or a protocol error exits `1`, so the shell can branch on it. Questions still read from stdin, so run it with stdin closed (`< /dev/null`) for unattended use: every permission card and `ask_user` form then reads EOF and is cancelled, which in `Interactive` mode cancels the tool behind it. For a run that should actually get work done unattended, put the session in `Autopilot` first and `--resume` it.

At startup it also checks, at most once a day, whether a newer release exists, and offers to install it before anything else happens (see [Updating](#updating)). The answer travels into the container with the launcher, so one yes updates both.

Terminal geometry is read with `stty size` at startup and at each prompt, keeping the binary dependency-free and platform-neutral, at the cost of not reacting to a resize mid-turn: the next prompt picks up the new width.

## Zed setup

Add to your Zed settings (`~/.config/zed/settings.json`):

```json
{
  "agent_servers": {
    "Codehalter": {
      "type": "custom",
      "command": "codehalter"
    }
  }
}
```

Open the agent panel (`Cmd+?` / `Ctrl+?`), click `+`, and select "Codehalter". Pick Interactive or Autopilot from the mode selector in the session header.

## Sandboxing with a devcontainer

Codehalter edits files and runs build/test commands. Running it inside a [devcontainer](https://containers.dev) sandboxes those actions to a throwaway environment, keeps the project's toolchain pinned, and means the same setup works on every machine. Zed launches ACP servers *inside* the container when you open the project in one (`Ctrl-Shift-P`, then type "open dev container"), so codehalter, Firefox, and the LLM endpoint must be reachable from there. The startup banner reports `Container: devcontainer` when this is wired up correctly. From a terminal there is nothing to wire up: `codehalter --cli` starts the container itself (see [Standalone CLI](#standalone-cli)).

**Codehalter scaffolds the devcontainer for you.** On the first session in a project without a `.devcontainer/` directory, it prompts for a base distro, **`Alpine / Arch / Debian / Fedora / Ubuntu`**, and writes the chosen `Dockerfile` + `devcontainer.json`:

| Distro | Base image |
|--------|-----------|
| Alpine | `alpine:3.24` |
| Arch | `quay.io/archlinux/archlinux:base-devel` (+ `yay` AUR helper) |
| Debian | `debian:trixie` |
| Fedora | `fedora:44` |
| Ubuntu | `ubuntu:26.04` |

Each image installs a **minimal** toolchain (git, curl, sudo, openssh, Firefox), creates a passwordless `dev` sudo user, and installs codehalter into `~/.local/bin` (added to `PATH`). Language toolchains (go, node, clang, rust, …) are added per project on demand, codehalter's skills and fix-cards offer to install what a given repo needs rather than baking everything into the base. All five seed a `devcontainer.json` that:

1. Bind-mounts `${localWorkspaceFolder}/.git` (read-write) plus the SSH agent socket, so the agent can run git fully, including `git commit` / `git push`, **but it is instructed to do so only when you explicitly ask**, never on its own initiative or as a side effect. (Want a hard guarantee rather than an instruction? Add `,readonly` to this mount; the agent then falls back to handing you the commit command.)
2. Bind-mounts `${localEnv:HOME}/.gitconfig` read-only into the dev user's home, so the in-container `git` sees your real `user.name` / `user.email` / signing key without any per-project setup.
3. Bind-mounts `~/.config/codehalter` read-only so `settings.toml` is shared across every project as the global fallback (a project-local `.codehalter/settings.toml` still wins when present).
4. Sets `containerEnv.DEVCONTAINER=true` so codehalter's startup banner shows `Container: devcontainer` and the `run_command` tool is registered.

After the container is built, point Zed at the in-container codehalter binary:

```json
"Codehalter": { "type": "custom", "command": "codehalter", "args": [], "env": {} }
```

If your LLM server runs on the host, change `server` in `~/.config/codehalter/settings.toml` to `http://host.docker.internal:8080`. For an existing language-stack devcontainer (e.g. `mcr.microsoft.com/devcontainers/go:1`), copy the four mounts above into its `devcontainer.json` and run `curl -sL https://raw.githubusercontent.com/tbocek/codehalter/main/install.sh | bash` in its image, no need to start from one of the scaffolded bases.

### Commits and pushes from inside the container

The agent never commits or pushes on its own, only when you explicitly ask. When you do (a plain request, or the `/commit` macro):

1. It drafts the commit message from `git status --porcelain` + `git diff HEAD`, using the conversation for the *why*, and writes it to `.codehalter/.git_commit`.
2. It runs the commit (and push, if you asked) itself, with that message:

   ```sh
   git commit -F .codehalter/.git_commit && git push
   ```

3. It reports the commit subject (and branch, if pushed) in its reply.

**Fallback:** on an older container that mounts `.git` read-only, or where `git push` lacks SSH auth, the agent doesn't fight it, it suggests the exact host-side `git commit … && git push` for you to run instead.

`.codehalter/` is gitignored on first bootstrap, so the draft file never accidentally gets staged.

## Glossary

These words mean one specific thing each. The code uses them in comments; this is the reference.

**Session**: one Zed thread. Persisted to `.codehalter/session_<id>.toml` and reloaded on restart. One user, one project, one conversation.

**Large turn** (also "user turn"): one user prompt until the `✅ Done` line, i.e. until you can type again. One `orchestrate()` call. It contains the whole pipeline: plan, every subtask, every replan, document. Elsewhere in this README the word "prompt" means this.

**Small turn** (also "harness turn"): one LLM request/response plus the tool results it triggered. Stored as one assistant message, stamped with its phase and with that call's `prompt_tokens`. A large turn is typically tens of small turns.

**In-flight turn**: the large turn currently running, and inside it the small turn not yet answered. Compaction always keeps the in-flight small turn; that is the one thing it can never fold away.

**Phase**: which stage of the pipeline a small turn belongs to: `plan`, `execute` or `document`. Recorded per message.

**Role**: which params table a call is sent with: `thinking` (plan and replan) or `execute` (subtasks, document, web page reading). Both roles are the *same* `[[llm]]` entry, i.e. the same server and the same model. Only `params_thinking` vs `params_execute` differ.

**LLM call**: one HTTP request to the server. This is the unit the prefix cache sees. One small turn is one LLM call.

**Rendering**: the flat token sequence the server's chat template produces from (messages, tools, `chat_template_kwargs`). You send JSON; the server caches *tokens*. The same conversation sent with two different `chat_template_kwargs` produces two different renderings.

**Prefix cache / KV slot**: the server-side KV cache holding one rendering. `parallel = N` gives the server N slots. Reuse is longest-common-prefix from token 0: position 40,000 is reusable only if tokens 0 to 39,999 are identical to last time.

**Rewind**: an LLM call whose `cached` count came back below the previous call's `prompt` count, meaning the server re-read tokens it already had. Logged as `CACHE` lines in the session log and counted on the `✅ Done` line.

**Compaction**: the reactive fold triggered by a context-overflow `400`. Keeps the in-flight small turn (plus recent completed small turns on the first step) and folds everything older into the Summary.

**Turn note**: a seven-section structured note the background LLM writes about each *completed large turn*.

**Shadow buffer**: where turn notes accumulate between compactions. Persisted, so it survives a restart.

**Summary**: the rolling prose that leads the context after a compaction. **Folded summary** is a shorter rewrite of it, produced in the background and used as the base of the *next* compaction, so summaries do not grow without bound.

**Background LLM**: the `purpose = "summary"` entry. Runs turn notes and the git-commit drafter off the foreground slot. With only one slot available it rides the foreground context as a prefix extension instead.

### Why a role switch can cost a full re-prefill

Inside a *single* large turn the role changes at least once: plan runs on `thinking`, then the first subtask switches to `execute`. Every replan switches back to `thinking` and forward again, so a turn that replans twice switches five times. If `params_thinking` and `params_execute` differ in anything that reaches the chat template (`chat_template_kwargs` above all), those calls ask the server for two different renderings of the same conversation. A rendering the server has never held costs a full prefill; after that it may keep both and serve either, in which case a switch costs only what was appended under the other rendering since you last used this one. Probed against llama.cpp on a 13,972-token prompt: adding `enable_thinking = false` returned `cached = 0`, and the same request repeated returned `cached = 13,932`. So the floor is one full re-prefill per rendering and the running cost is roughly double the prefill work of a single rendering, since each switch re-reads what the other role wrote. Samplers (`temperature`, `top_p`, `max_tokens`, …) never enter the rendering and may differ freely.

Measured on one 11.6 h session against a single-slot 27B: 436 LLM calls on the foreground connection, 35 role switches, one every ~12 calls. With both roles rendering identically those switches re-read 121,748 tokens in total (median 1,526 each, 97.4 % cached), which is 4.2 min at the server's measured 483 tok/s. The same 35 calls carried 2,414,262 prompt tokens between them, so re-prefilling each from scratch is 83.3 min. Six calls in that session did lose a prefix, and only one was a rendering change: the stuck-in-`<think>` retry, which then flipped `enable_thinking`. Entering that window cost 71,997 tokens at `cached = 0` and leaving it 27,585 more, the span generated while the flag was on. The other four losses were idle evictions and noise.

codehalter no longer does that. Every call that wants reasoning off, the whole `execute` role (subtasks, document) as well as that stall retry, appends an already-closed `<think></think>` as a trailing assistant message and asks the server to continue it, which is exactly what the template emits for `enable_thinking = false` but reached by extension rather than re-rendering. Same suppression, no divergence: probed on that server with a tools array attached, the prefill returned `cached = 14,263` of 14,273 and still emitted its tool call, where the kwargs flip returned `cached = 0`. The detector above stays because a `settings.toml` whose two roles disagree reproduces the old cost exactly. `res/settings.toml` carries the same note next to the params tables.

## How it works

1. Zed spawns `codehalter` as a subprocess and communicates via JSON-RPC 2.0 over stdio.
2. On session start, the agent indexes project files, probes image support on the configured LLM, folds the project's `AGENT.md` and the applicable skills into the system prompt, and reports what the container has and is missing.
3. Each prompt runs through the pipeline:
   - **Plan**, the `thinking` LLM analyzes the request with read-only tools and gathers external info via web tools if needed. If the request is ambiguous it asks you a multiple-choice clarification first instead of guessing; if it's a pure question it answers directly with no execution pass (`report_only`). Otherwise it emits an array of `subtasks`, each with its own `verify` recipe, which are posted to the thread as a numbered plan and then run.
   - **Work**, for each subtask the `execute` LLM runs a single bounded tool-calling loop where it edits, runs commands, and then runs the verify recipe itself before calling `respond`. Mid-run it can call `submit_plan` to revise the *remaining* subtasks in place (a plan-upsert) when execution reveals the plan needs adjusting, no full replan. A stuck subtask bounces back to replanning after 8 failed rounds, with a 100-iteration backstop against runaways. File edits are shown as diffs.
   - **Replan on failure**, if a subtask bounces, the executor returns without calling `respond`, or a tool exits non-zero, the orchestrator records the failure reason and re-plans (up to 20 replans per prompt). When the same failure recurs (Jaccard over issue words ≥ 0.6), the replan note escalates to "the prior fix didn't work; propose a structurally different approach."
   - **Document**, fires once at the end of a successful prompt: the `execute` LLM (the same connection the subtasks just ran on, so it inherits their warm prefix) checks against `DOCUMENT.md` whether the change is user-visible (new feature/flag/API/dep, install or config change) and, if so, updates or creates the project README. Routine refactors and bug fixes are skipped.
4. Prior subtasks' assistant replies stay in history so later subtasks have context.
5. Conversation history is persisted to `.codehalter/session_<id>.toml`. After every turn a background goroutine on the `purpose = "summary"` slot condenses the whole turn into a seven-section structured note that accumulates in a shadow buffer (itself persisted in the session TOML, so it survives a restart). Compaction is **purely reactive** — driven by the server's context-overflow **400** and escalates in two steps. Step 1: keep the **unfinished small turn** plus the most recent **~10k tokens of completed small turns** verbatim, and fold everything older into the rolling summary; retry. Step 2 (only if the retry still 400s): keep only the **unfinished small turn**; retry. Each step strictly shrinks the context, so it terminates. With only one `[[llm]]` entry (or `parallel = 1`) the summariser extends the foreground context as a prefix-extension on the same slot — cache-safe, briefly serialised with the next prompt.

## Compared to other harnesses

Codehalter borrows two ideas: the synthetic `respond` terminal tool from [forge](https://github.com/antoinezambelli/forge), and explicit terminal tools from [pi/coding-agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent), though codehalter declares *which* tools end the loop per phase (a `phasePolicy`: plan exits on `submit_plan`, execute on `respond`) rather than with a per-tool flag.

### Where the field stands (September 2026)

- **Language servers are being removed, not added.** opencode v2 "does not run language servers, expose LSP tools, or produce LSP diagnostics"; pi, hermes and Claude Code never shipped them. [Crush](https://github.com/charmbracelet/crush) is the exception, and it does the opposite of a tool: diagnostics and symbols are fed into the context rather than offered for the model to call. Codehalter dropped its LSP tools for the reason that shape addresses: the model kept reaching for `search_text` instead.
- **Model-visible todo lists are being withdrawn.** Claude Code stopped offering todo tools to current models, opencode v2 deleted `todowrite`, Codex ships `update_plan` off by default; [Cline](https://docs.cline.bot/features/focus-chain) went the other way and reinjects a Focus Chain every few messages. Codehalter has no todo tool: the plan is a `submit_plan` table the executor revises in place.
- **Compaction has converged** on a token threshold with a reserve buffer, a summary with required headings, and a retained tail whose tool outputs are clipped. Codehalter's trigger is the odd one out: it compacts on the server's context-overflow 400 rather than on an estimate, and writes one structured note per completed turn as it goes.
- **Local models are table stakes.** opencode v2 auto-discovers Ollama, LM Studio and vLLM; [hermes](https://github.com/NousResearch/hermes-agent) ships a managed llama.cpp runtime that sizes the quant to your GPU; pi drives `llama-server` router mode and manages GGUFs from the CLI. Codehalter assumes that world instead of adapting to it: router-mode probes per model id, per-connection slot semaphores, and a prefix-cache rewind detector that names the cause. Keeping an idle prefix alive is a shared answer: opencode has `warming`, pi gates a refresh on expected savings, and codehalter re-sends its 1-token warm call while the session is idle and while a long tool runs.
- **Typing during a turn queues rather than cancels.** pi polls for steering messages between iterations and Claude Code queues them; codehalter now does the same, with the stop button as the way to interrupt. It used to replace the turn in flight, which during a long `/spec` round threw the round away.
- **Sandboxing is mostly left to the user.** opencode states plainly that its shell "runs with the host user's filesystem, process, and network authority". Codehalter refuses to run outside a container at all.
- **ACP is winning as the editor surface.** [Goose](https://goose-docs.ai/docs/gdk/acp/zed/) ships a native ACP server, opencode has `opencode acp`, Codex and Claude Code integrate through external adapters, and Amp deleted its VS Code and Cursor extensions outright. Codehalter speaks ACP natively and nothing else.

### The harnesses compared

Surveyed September 2026. "Local" means what the harness does for a self-hosted model beyond accepting an OpenAI-compatible base URL.

| Harness | Local models | Editor integration | Context strategy | Stance |
| --- | --- | --- | --- | --- |
| **codehalter** | llama.cpp router probed per model id, slot semaphores, rewind detector, keeps an idle prefix warm | ACP, and nothing else | Reactive on the server's 400, one structured note per completed turn | Refuses to run outside a container |
| [opencode](https://github.com/anomalyco/opencode) v2 | Auto-discovers Ollama, LM Studio, vLLM (vLLM starts with tools off: no capability metadata) | `opencode acp`, plus its own server with TUI/desktop/web clients | Threshold with reserve buffer, required-heading summary with one retry, provider checkpoints | Deleted language servers and todo tools in v2; no sandbox by design |
| [pi](https://github.com/earendil-works/pi) | `llama-server` router mode first class, loads and downloads GGUFs from the CLI | None: RPC, SDK and TypeScript extensions | Auto-compact on a reserve threshold, fixed-template summary, cost-gated cache warming | Ships no MCP, subagents, permission popups, plan mode or todos on purpose |
| [forge](https://github.com/antoinezambelli/forge) | A reliability proxy in front of any local server, 26 per-model sampling profiles | None: it is an OpenAI and Anthropic proxy other harnesses point at | Tiered compaction that drops nudges, then tool results, then reasoning | One tool, `respond`, so a small model never picks between text and tool call |
| [hermes-agent](https://github.com/NousResearch/hermes-agent) | Managed llama.cpp runtime: sizes the quant to the GPU, grows the window, spills experts before KV | ACP adapter, plus CLI, TUI, desktop and chat gateways | Phased batch compression, plus opt-in per-turn micro-compaction that never compacts user turns | Deterministic guardrails (repeat streaks, cycle detection) over prompt wording |
| [Goose](https://goose-docs.ai/docs/gdk/acp/zed/) | Explicit roadmap to tune the loop and tool schemas for small open models | Native ACP server, installable from Zed's registry | Auto-compact at 80%, summarises older tool outputs while recent ones stay verbatim | YAML recipes and sub-recipe subagents with isolated context |
| [Crush](https://github.com/charmbracelet/crush) | Auto-discovers Ollama, llama.cpp and LM Studio | No first-party ACP, only a third-party bridge | LSP diagnostics and symbols fed into context generation | The one harness still betting on LSP, and not as a tool |
| [Claude Code](https://code.claude.com/docs/en/changelog) | None: frontier models only | ACP through an adapter | Auto-compaction plus a system-prompt boundary marker so the stable prefix caches | Withdrew todo tools from current models; subagent fan-out is the default unit of work |
| [Codex CLI](https://learn.chatgpt.com/docs/changelog) | `--oss` mode against Ollama, LM Studio and MLX | ACP through an external server | Full-history handoff summary at a threshold capped below the advertised window | `update_plan` off by default; deleted its MCP-server subcommands |
| [Cline](https://docs.cline.bot/features/focus-chain) / Roo | Any OpenAI-compatible endpoint | VS Code extension, no ACP | Focus Chain: a todo list reinjected every few messages, surviving compaction | The loudest argument that weak models need a visible task list |
| [Aider](https://aider.chat/HISTORY.html) | Any endpoint; last release Feb 2026 | None | Repo map plus SEARCH/REPLACE retries | Architect and editor split: one model plans, a cheaper one applies the edit |

### What codehalter adds

- **Devcontainer-first sandboxing posture**: auto-scaffolded `.devcontainer/` from any of five base distros (Alpine/Arch/Debian/Fedora/Ubuntu), host `.gitconfig` mounted read-only, plus an on-request commit flow (`/commit`) and a standing instruction never to commit or push on its own, so a commit is always your explicit call.
- **Native MCP client**: `.codehalter/mcp.toml` spawns each `[[server]]` as a long-lived child, registers its tools under `<name>__<tool>`, and reconciles at turn boundaries. Dual-era: each server is probed via `server/discover` and spoken to with the stateless 2026-07-28 revision ("MCP 2", per-request `_meta` + mirrored `Mcp-Method`/`Mcp-Name`/`Mcp-Param-*` headers, no sessions) when it supports it, with automatic fallback to the 2025-06-18 initialize/session handshake for everything else.
- **Slot-aware LLM routing for local servers**: per-conn `parallel = N` semaphores, and the per-turn summariser routed off the foreground KV-cache slot — or, on a single slot, riding it as a prefix-extension — so the main session's prefix cache stays warm.
- **Plan → per-subtask self-verifying loop → Document pipeline** built in, not delegated to an extension: the planner decomposes into subtasks each carrying its own verify recipe; the executor runs the recipe itself before declaring done; failed subtasks trigger a replan (≤20 per prompt) with fuzzy duplicate-failure detection (Jaccard over issue words) escalating the note when the same problem recurs; the document pass updates the README only when the change is user-visible.
- **Spec mode** (`/spec`): a deterministic ledger over a spec directory, one item per turn, each item committed only once a test that names it passes, and each recorded with the spec text it was built from, so editing the spec later re-opens exactly the items that changed. Goose's recipes and Cline's Focus Chain are the nearest things elsewhere, and neither ties completion to a passing test.
- **Web tools restricted to the planning phase** (`web_search` via DuckDuckGo + `web_read` via headless Firefox) so execution stays offline.
- **Empty-project bootstrap, stuck-tool-loop detection, reactive 400-driven two-tier compaction** (keep unfinished turn plus recent completed small turns, then keep only the unfinished turn if that still overflows).

### What codehalter does not have

forge's published 26-scenario eval suite with ablation studies. pi's session branching, 30+ provider adapters, TypeScript extensions and `@`-mention file references. opencode's durable server-side sessions that survive a disconnected client, its browser tool namespace, and Code Mode (the model writes a confined program over the tools instead of calling them one at a time). hermes's managed llama.cpp runtime and FTS5 search across past sessions. Goose's YAML recipes and sub-recipe subagents. Cline's Focus Chain. No subagents, no LSP, no embedding index: all three were removed or never added on purpose.

## Credits

- [forge](https://github.com/antoinezambelli/forge) by Antoine Zambelli, the synthetic `respond` terminal tool used in execute is borrowed from forge's playbook for keeping small local models inside structured tool-calling grammar.
- [pi/coding-agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent) by earendil-works, codehalter's explicit terminal tools (which tool ends the loop is declared per phase via `phasePolicy`) are inspired by pi's `terminate: true` tool-result signal.
- Code harness idea exchanging with Clemens
