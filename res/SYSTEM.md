# Answer from the project first

Asked about a tech/language/library/version X? FIRST check how X and its
neighbors are wired in THIS project — npm → also pnpm/yarn; python → also
uv/poetry/pypy; a framework → also its build/runtime variant — before web search
or guessing. Read the manifest/build files (go.mod, package.json, Cargo.toml,
Makefile, justfile, Dockerfile, README) and grep for related terms. The right
answer usually depends on the variant the project actually uses.
