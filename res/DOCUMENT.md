# Documentation phase
The work is done and verified. Keep the README in sync with what users see. Docs only: no source edits, no tests, no redoing work.

If the request named the files to change, or the turn changed nothing a user or contributor must know, reply exactly `No documentation change needed.` and call no tools. They must know about: a feature, command, flag, env var or settings key; a public API or CLI change; a dependency, prerequisite or supported platform; install, build or run steps; a breaking change or removal. They need not know about refactors, small bug fixes, tests, formatting, internal renames, or changes only to `.devcontainer/`, `.codehalter/`, `.github/`, CI or lint config, or lockfiles.

Otherwise:
1. Find the README with `ls`. Read it once, and do not re-read what you already have.
2. `edit_file` the relevant section with the smallest change, in the README's style. No README: `write_file` a `README.md` with a short description and a section for the change.
3. Write only what the executor's reply supports. Add no changelog section unless there is one.

End with one short sentence: `Updated README.md: Configuration section, new FOO_BAR env var.`
