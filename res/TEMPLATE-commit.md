Commit my uncommitted changes (do NOT push unless I explicitly said so in this message).

1. Inspect what changed: `git status --porcelain` and `git diff HEAD`.
2. Draft the commit message from the actual changes, using our conversation for the WHY:
   - Line 1: imperative subject, no trailing period, ≤72 chars ("add ...", "fix ...", "refactor ..."). It must complete "If applied, this commit will <subject>".
   - Blank line.
   - Body wrapped at 72 cols: 1-3 short bullets/sentences on WHY, not a diff restatement.
3. Follow the EXECUTE.md git flow: write the message to `.codehalter/.git_commit` with write_file, then commit via `git commit -F .codehalter/.git_commit`.
4. Report the commit subject (and branch, if pushed) in your final message.
