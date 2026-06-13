# Container skill

You run inside a container. The workspace is bind-mounted from the host;
the container is your sandbox — pkg-mgr / pip / npm writes persist for the
container's lifetime (wiped on rebuild), so test installs are cheap and
reversible.

## Git — writable, commit/push when asked

`.git` is bind-mounted writable and your `~/.gitconfig` plus SSH agent are
mounted, so commit and push work from inside the container. But act on git ONLY
when the user explicitly asks (see EXECUTE.md) — never commit/push on your own.

- **Fine on request**: `git commit`, `git push`, and the always-safe reads
  `git status`/`log`/`diff`/`show`/`blame`/`fetch`.
- **Still avoid unless explicitly asked**: history rewrites on shared branches —
  `git reset --hard`, `git push --force`, `git filter-branch`, `git gc --prune`.

If a git write fails (read-only `.git`, or no credentials mounted — an older
container), surface the exact host command and stop rather than fighting it.

## When a task fails with "command not found"

Applies in execute and verify-failure replans. In plan, emit install +
Dockerfile-edit steps and let execute run them.

1. Confirm with `run_command`: `which <tool>` (exit 1 = missing).
2. Read `.devcontainer/devcontainer.json` and its `Dockerfile`.
   - If declared → image is stale; point at the install line.
   - If not declared → propose adding it (but TEST IT first, see below).
3. Do NOT retry the same failing command without changing anything.

Package-manager commands depend on the base image — check the matching
`SKILL-<os>.md` (alpine, arch, debian, fedora, ubuntu) or fall back to
`/etc/os-release`.

## Test installs in the live container before patching the Dockerfile

**Execute phase only.** Plan must stay read-only.

Before writing `RUN <pkg-manager> install <pkg>` (or `pip`, `npm i -g`)
into the Dockerfile, first run the same install via `run_command`:

```
run_command: <install-command> && <tool> --version
```

The install persists for this container's lifetime, so re-run the project
build/test to confirm end-to-end success.

If the install fails (wrong package name, repo missing it, version
mismatch), debug HERE — try alternative sources, search the release page —
before committing to a Dockerfile change. An untested patch is a guess.

When the test succeeds, propose the Dockerfile edit with the exact verified
commands and tell the user to rebuild.

## Use `run_command` for probes too

- `which <tool>`, `<tool> --version` — does it exist?
- Type-check / lint (`cargo check`, `tsc --noEmit`, `go vet ./...`).
- `ls -la <path>`, `cat <config>` — inspect state.

Workspace writes from probes (e.g. `cargo check` populating `target/`) are
real. Fine for build artifacts; don't run destructive shell commands
against source — use `edit_file`/`write_file`.

## What `run_command` is NOT for

- Long-running services — they die with codehalter. Don't start daemons.
- Replacing `run_task` — for declared project tasks (`just build`,
  `npm test`), keep using `run_task` so the user sees it in the same UI.
- Editing project files — use `edit_file`/`write_file` so the change goes
  through the diff/approval UI. Raw `sed -i` or `>` skip that.

Exit code is in the output and tool-card title. `which <tool>` exit 1 =
binary missing, not tool error.
