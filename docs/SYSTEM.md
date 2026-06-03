# Answer from the project first

When the user asks about a technology, language, library, or version X, first
check how X and its adjacent tooling are configured in this project (e.g. npm →
also pnpm/yarn; python → also uv/poetry/pypy; a framework → also the
build/runtime variant) before searching the web or making assumptions. Read
manifest/build files (go.mod, package.json, Cargo.toml, Makefile, justfile,
Dockerfile, README) and grep for related terms — the right answer often depends
on which variant the project actually uses.
