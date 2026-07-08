# Java skill
## Modern Java
- `var` for local var type inference.

## Null
- Always inspect existing code for @Nullable/@NonNull usage BEFORE writing new methods; ONLY annotate boundaries with these annotations if they're already used in the project — NEVER introduce them unconditionally even when you know modern alternatives exist, because annotation conventions vary by codebase.

## Style
- BEFORE writing code, locate the project's checkstyle/spotless config (e.g. `checkstyle.xml`, pom/gradle plugin) and read the surrounding file to match its indentation, brace style, and import grouping; then add ONLY the new method — never reformat, re-indent, or reorder unrelated lines — and always run the checkstyle check on the modified file before declaring done. The project's style wins even when a cleaner modern pattern exists in your head.

## Tests
- JUnit 5 (`@Test`, `org.junit.jupiter.api`). AssertJ if project already does; else plain JUnit asserts.
