# Container skill

You are running inside a container. The workspace is bind-mounted in from
the host, but the container itself is your sandbox — package-manager / pip /
npm writes persist for the container's lifetime (wiped on rebuild), so test
installs are cheap and reversible.

## Git — read-only history, mutable working tree

`.git` is bind-mounted read-only. The working tree can change, but the
object/ref store cannot, so destructive history-rewriting commands fail at
the filesystem layer:

- **Blocked**: `git push` (any form), `git push --force`, `git reset --hard`,
  `git branch -D`, `git filter-branch`, `git gc --prune`, anything that
  writes under `.git/`.
- **Fine**: `git status`, `git log`, `git diff`, `git show`, `git blame`,
  `git ls-remote`, `git fetch`, `git clone <other-repo>` (into a fresh dir),
  `git archive`. Local read-only and remote fetches both work.

If the user wants to push, commit, amend, or rewrite history, surface the
exact command they should run on the host and stop — don't try to do it
yourself, the FS layer will block you and the error is unhelpful.

## When a task fails with "command not found"

1. Confirm with `run_command`: `which <tool>` (or `command -v <tool>`).
   Exit 1 = missing.
2. Read `.devcontainer/devcontainer.json` and any `Dockerfile` it references.
   - If the missing tool IS declared → image is stale. Fix is "rebuild the
     devcontainer". Point at the line that should have installed it.
   - If the missing tool is NOT declared → propose adding it. But before you
     emit the Dockerfile patch, TEST IT (see below).
3. Do NOT retry the same failing command without changing anything first.

The package-manager commands for installing depend on the base image —
check the matching `SKILL-<os>.md` (alpine, arch, debian, fedora, ubuntu)
for the right tool. Fall back to `/etc/os-release` if you're not sure which
one applies here.

## Test installs in the live container before patching the Dockerfile

When you're about to write `RUN <pkg-manager> install <pkg>` (or `pip
install`, `npm install -g`, etc.) into the Dockerfile, first run the same
install via `run_command` and verify it actually works:

```
run_command: <install-command> && <tool> --version
```

The install persists for the lifetime of this container, so you can then
re-run the project's build/test task to confirm the build now succeeds
end-to-end.

If the install fails (wrong package name, repo doesn't have it, version
mismatch), debug HERE — try alternative package sources, search the
project's release page, find the canonical install method — before
committing to a Dockerfile change. A Dockerfile patch you haven't tested
is a guess.

When the test succeeds, propose the Dockerfile edit with the exact commands
you just verified, and tell the user to rebuild.

## Use `run_command` for probes, too

- `which <tool>`, `<tool> --version` — does the tool exist?
- Type-check / lint targets (`cargo check`, `tsc --noEmit`, `go vet ./...`,
  …) — would compilation succeed?
- `ls -la <path>`, `cat <config>` — inspect state.

Workspace writes from these probes (e.g. `cargo check` populating `target/`)
are real. That's fine for build artifacts; just don't run destructive shell
commands against source files — use `edit_file`/`write_file` for that.

## What `run_command` is NOT for

- Long-running services — the process dies with codehalter; when the agent
  exits, the shell exits too. Don't start daemons in the foreground here.
- Replacing `run_task` — for declared project tasks (`just build`, `npm
  test`), keep using `run_task` so the user sees it in the same UI as when
  they invoke the same target by hand.
- Editing project files — use `edit_file` / `write_file` so the change goes
  through the diff/approval UI. Raw `sed -i` or `>` skip that.

Exit code is always in the output and the tool card title. Interpret it:
`which <tool>` exiting 1 means the binary is missing, not that the tool
itself erred.
