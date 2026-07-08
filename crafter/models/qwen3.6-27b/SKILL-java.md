# Java skill
## Modern Java
- `var` for local var type inference.

## Null
- Annotate boundaries `@Nullable`/`@NonNull` if project uses them.

## Style
- Match project's existing checkstyle/spotless config — don't reformat the world.

## Tests
- JUnit 5 (`@Test`, `org.junit.jupiter.api`). AssertJ if project already does; else plain JUnit asserts.
