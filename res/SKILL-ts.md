# TypeScript skill
- Strict null checks ON: `T | undefined` is NOT `T`. Use `!` ONLY when you can prove non-null.
- Prefer named exports; ESM imports, not `require`, in new code.
- Always braces, even on a one-line `if`/`for`: without them a line added to the body later silently falls outside it (goto fail).

## Package manager — ONE per project, detect BEFORE installing
`package.json` `packageManager` field → lockfile (`pnpm-lock.yaml`→pnpm, `yarn.lock`→yarn, `bun.lockb`→bun, else npm) → an existing `node_modules/.pnpm` dir (→pnpm; the lockfile is often gitignored, so this is the real signal). Use that ONE for EVERYTHING (formatter, scripts). Never mix npm into a pnpm/yarn project, never `npm install` as a fallback. Get pnpm/yarn via `corepack enable`.
The base image has no node → install `nodejs npm` (OS pkg mgr) + persist FIRST. pnpm in a devcontainer drops `.pnpm-store` in the repo (the store can't hardlink across the bind mount) → gitignore `.pnpm-store` + `node_modules`.

## Type-check + lint through the project's scripts
Its package-manager run (`pnpm run`/`npm run`/`yarn`, matching the lockfile) of `typecheck`/`build`/`lint`, whatever `package.json` declares. Don't reach for `npx tsc --noEmit` or eslint directly when a script exists.

## Formatter config — pin the style that is already there
No `.prettierrc`/`.editorconfig` → prettier, your editor and CI each use their own defaults and fight over the file. Fix by MEASURING, never by imposing:
- indent: `grep -h "^ *[^ ]" src/*.ts | sed "s/[^ ].*//" | awk "{print length}" | sort -n | uniq -c` → the smallest non-zero width that repeats is the indent. Tabs: `grep -lP "^\t" src/*.ts`.
- quotes: count `'` vs `"` string delimiters. semicolons: does a line end in `;`? trailing commas: does a multi-line literal end `,\n)`? width: longest existing line, rounded.
- Write `.prettierrc` with exactly those (`tabWidth`, `useTabs`, `singleQuote`, `semi`, `trailingComma`, `printWidth`).
- Prove it: `prettier --list-different .` → few files = the config describes the code. Many = the config is wrong; fix it, do NOT reformat the repo to match a guess.
- `.prettierignore` vendored/generated files (a copied upstream lib, `dist/`) BEFORE any `--write`.
- Then `prettier --write .` as ONE commit, formatting only, and only on a clean tree.
