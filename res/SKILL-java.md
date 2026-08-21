# Java skill
- Records for immutable data carriers, not hand-written getter classes or lombok.
- Never return null → `Optional<T>` for "may be absent", throw for hard errors. No `Optional.get()` without a check.
- ALWAYS try-with-resources for `AutoCloseable` (streams, readers, connections).
- Don't mutate the source collection inside a stream pipeline.
- Match the project's existing checkstyle/spotless config — don't reformat the world.
- Tests: JUnit 5 (`org.junit.jupiter.api`). AssertJ only if the project already uses it.
