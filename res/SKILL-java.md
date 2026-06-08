# Java skill

## Modern Java (11+)
- `var` for local variable type inference. Don't use `var` for fields or method signatures.
- Records for immutable data carriers, not lombok-heavy classes.
- `switch` expressions where supported.

## Null
- Avoid returning `null`. Return `Optional<T>` for "may be absent" or throw for hard errors.
- Don't `Optional.get()` without checking; use `orElse`, `orElseThrow`, `map`, `ifPresent`.
- Annotate boundaries with `@Nullable` / `@NonNull` if the project uses them.

## Resources
- Always use try-with-resources for `AutoCloseable` (streams, readers, connections):
  `try (var f = Files.newBufferedReader(p)) { ... }`.

## Collections and streams
- Prefer `List.of`, `Set.of`, `Map.of` for small immutables.
- Streams are fine when they make code clearer; don't force them when a `for` loop reads better.
- Don't mutate the source collection inside a stream pipeline.

## Style
- Fields are `private final` by default; mutate only when you have to.
- Constructor injection over field/setter injection (Spring, Guice, etc.).
- Match the project's existing checkstyle / spotless config — don't reformat the world.

## Tests
- JUnit 5 (`@Test`, `org.junit.jupiter.api`). Use AssertJ if the project already does; otherwise plain JUnit asserts.
