# Devcontainer skill

This project ships a `.devcontainer/` directory, so the agent is running inside
a container and the host workspace is bind-mounted in. `run_command` runs
shell commands directly: the container itself is your sandbox (it's
throwaway, and package-manager / pip / npm writes persist for the container's
lifetime — wiped on rebuild). Workspace writes are real, but recoverable
from `.git/`.

The base image varies per project — could be Debian/Ubuntu (`apt-get`),
Arch (`pacman` / `yay`), Alpine (`apk`), Fedora (`dnf`), etc. Check the
Dockerfile's `FROM` line (or `/etc/os-release` if you need to be sure) to
know which one applies here, then use the matching tool from the
examples below.

**Blocked**: `.git` is bind-mounted into the container read-only, so
destructive history-rewriting commands (`git reset --hard`,
`git push --force`, `git branch -D`, `git filter-branch`) fail at the
filesystem layer — the working tree can change but the object/ref store
cannot. Read code with `read_file`; the user runs destructive git themselves.
Read-only git is fine (`clone`, `log`, `ls-remote`, `archive`).

## When a task fails with "command not found"

1. Confirm with `run_command`: `which <tool>` (or `command -v <tool>`).
   Exit 1 = missing.
2. Read `.devcontainer/devcontainer.json` and any `Dockerfile` it references.
   - If the missing tool IS declared → image is stale. Fix is "rebuild the
     devcontainer". Point at the line that should have installed it.
   - If the missing tool is NOT declared → propose adding it. But before you
     emit the Dockerfile patch, TEST IT (see next section).
3. Do NOT retry the same failing command without changing anything first.

## Searching for packages

Distro repos lag upstream — sometimes by months for fast-moving toolchains.
Search before installing, and prefer the package manager that surfaces the
freshest version available on this image:

- **Debian/Ubuntu**: `apt-cache search <pkg>`, `apt-cache policy <pkg>` (shows
  candidate version). Stable repos can be very old; backports / PPAs may
  carry newer ones.
- **Arch**: ALWAYS prefer `yay` over `pacman` here — `yay` searches and
  installs from both the official repos AND the AUR in one pass, and the AUR
  routinely carries fresher versions (and packages that aren't in the
  official repos at all). Use `yay -Ss <pkg>` to search
  both sources, `yay -Si <pkg>` for details, `yay -S --noconfirm <pkg>` to
  install. Do NOT start with `pacman -S` and only fall back to `yay` when it
  fails — that wastes a round trip. If you must use `pacman` for some reason
  (e.g. yay isn't on PATH), `pacman -Ss` only covers the official repos and
  will miss AUR packages.
- **Alpine**: `apk search <pkg>`, `apk info <pkg>`.

If the candidate version looks older than you expected, or you're not sure
whether a given version even exists, look it up with `web_search` against
the upstream project's release page **before** claiming "version X is not
available." Don't guess version strings — confirm.

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
mismatch), debug HERE — try alternative package sources (AUR, backports,
upstream tarball), search the project's release page, find the canonical
install method — before committing to a Dockerfile change. A Dockerfile
patch you haven't tested is a guess.

When the test succeeds, propose the Dockerfile edit with the exact commands
you just verified, and tell the user to rebuild.

## Use it for probes, too

- `which <tool>`, `<tool> --version` — does the tool exist?
- Type-check / lint targets (`cargo check`, `tsc --noEmit`, `go vet ./...`,
  …) — would compilation succeed?
- Query the installed-package database — `dpkg -l <pkg>` (Debian),
  `pacman -Qi <pkg>` (Arch), `apk info <pkg>` (Alpine) — what version is
  actually pinned?
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
