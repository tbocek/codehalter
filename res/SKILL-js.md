# JavaScript skill
- ESM (`import`/`export`) for new code. CommonJS `require` ONLY when matching the existing file's style — check the file before writing either.
- Optional chaining (`?.`) + nullish coalescing (`??`) for defaults and safe access, not `&&` chains or `x === undefined` branches.
- Every promise rejection handled: `try`/`catch` around `await`, or an attached `.catch`. No fire-and-forget promises.
- Always braces, even on a one-line `if`/`for`: without them a line added to the body later silently falls outside it (goto fail).
- Match the project's existing linter (eslint, biome). Don't change `.editorconfig`/lint config unless asked.

## Package manager — ONE per project, detect BEFORE installing
`package.json` `packageManager` field → lockfile (`pnpm-lock.yaml`→pnpm, `yarn.lock`→yarn, else npm) → an existing `node_modules/.pnpm` dir (→pnpm; the lockfile is often gitignored). Use that ONE for everything — never mix npm into a pnpm/yarn project. pnpm/yarn via `corepack enable`. The base image has no node → install `nodejs npm` + persist FIRST. pnpm drops `.pnpm-store` in the repo in a devcontainer → gitignore `.pnpm-store` + `node_modules`.

## Checking a change — the project's own scripts
No type checker applies to plain JS: there is nothing to check against. Verify with what `package.json` declares (`lint`, `test`, `build`) through the project's package-manager run, not with a checker you install. A JS-only repo needs NO code-intelligence setup — don't propose one.
Adding TypeScript to the project (a `tsconfig.json`) changes that: see SKILL-ts.md, where codehalter type-checks every write for you.
