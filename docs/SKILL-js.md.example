# JavaScript skill

## Variables
- `const` by default. `let` for variables you reassign. Never `var`.
- Always declare; no implicit globals.

## Equality
- Use `===` and `!==`. Never `==` or `!=` (silent type coercion).

## Async
- `async`/`await` over `.then()` chains.
- Always handle promise rejection — `try`/`catch` around `await`, or attach `.catch`.
- No fire-and-forget promises.

## Modules
- ESM (`import`/`export`) for new code. Use CommonJS `require` only when matching an existing file's style.
- Imports: built-ins → third-party → local, with blank lines between groups.

## Idioms
- Destructure objects and arrays for clarity: `const { x, y } = obj`.
- Spread for immutable updates: `[...arr, x]`, `{...obj, field: x}`. Avoid mutating shared state.
- Optional chaining (`?.`) and nullish coalescing (`??`) for safe access — not `&&` chains.
- Arrow functions for callbacks; `function` for top-level definitions when hoisting matters.

## Style
- Match the project's existing linter (eslint, biome). Don't change `.editorconfig`/lint config without being asked.
- Don't add JSDoc unless the rest of the file uses it.
