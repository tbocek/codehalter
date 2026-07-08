# Java skill
## Modern Java
- Targeting Java 11+.
- `var` for local var type inference.
- NO `var` for fields or method signatures.
- Records for immutable data carriers, not lombok-heavy classes.
- `switch` expressions where supported.

## Null
- Avoid returning null → return `Optional<T>` for "may be absent" or throw for hard errors.
- No `Optional.get()` without checking → use orElse, orElseThrow, map, ifPresent.
- Annotate boundaries `@Nullable`/`@NonNull` if project uses them.

## Resources
- ALWAYS try-with-resources for `AutoCloseable` (streams, readers, connections): `try (var f = Files.newBufferedReader(p)) { ... }`.

## Collections / streams
- Prefer `List.of`, `Set.of`, `Map.of` for small immutables.
- Streams fine when clearer; don't force when `for` loop reads better.
- Don't mutate source collection inside stream pipeline.

## Style
- Fields `private final` by default; mutate only when you must.
- Constructor injection over field/setter (Spring, Guice, etc.).
- Match project's existing checkstyle/spotless config — don't reformat the world.

## Tests
- JUnit 5 (`@Test`, `org.junit.jupiter.api`). AssertJ if project already does; else plain JUnit asserts.
