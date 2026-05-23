# TypeScript skill

## Types
- Strict null checks are on. `T | undefined` is NOT the same as `T`.
- Use `?.` and `??` for safe access. Use `!` only when you can prove non-null.
- `interface` for object shapes, `type` for unions and aliases.
- Avoid `any`. Prefer `unknown` and narrow with type guards. Reach for generics before `any`.
- Don't use the non-standard `Object` or `Function` types.

## Async
- `async`/`await`, not `.then()` chains.
- Always handle promise rejection (`try`/`catch` or pass error up).
- Don't fire-and-forget promises — `await` or assign to a variable handled later.

## Modules
- ESM imports: `import { x } from "./y"`. Avoid CommonJS `require` in new code.
- Prefer named exports over default exports.
- Imports go: built-ins → third-party → local, with blank lines between groups.

## Idioms
- `const` for everything except locals you reassign; `let` for those; never `var`.
- Use destructuring for objects and arrays.
- Triple-equals `===` always. Never `==`.
- Prefer immutable updates (`{...obj, field: x}`, `[...arr, x]`) over mutation.

## Tooling
- Type-check and lint through the project's build target (see SKILL-buildfile) — `npm run typecheck`/`build`/`lint` or whatever the `package.json` scripts declare. Don't call `tsc` or eslint directly.
