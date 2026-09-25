# Spec final pass: `{{spec_dir}}/` → `{{out_dir}}/`

Every item of the specification is covered by a passing test: {{items}} items, {{blocked_count}} blocked. This round implements no item. It checks that the pieces make one program a person can build, run and try, and it writes down how.

## Target

{{target}}

The spec in `{{spec_dir}}/` is READ-ONLY; codehalter refuses edits there. {{context}}

## Do, in this order

1. Build it the way a user would: the release build for the target (`cargo build --release`, `npm run build`, `go build ./...`, whatever the target uses), from `{{out_dir}}/`. A warning about dead code or an unused import is a leftover of some round: fix it.
2. Run `{{test_cmd}}` from `{{out_dir}}/` once, into a file, and read its summary. It must pass. If it does not, fix the code, not the test.
3. Start the program the way a user would and look at it: the entry point with `--help`, then for a desktop app every screen through the `snapshot` recipe, then `screenshot` each image and compare it with the spec's image of that screen (widgets, their order, labels and tooltips; not pixels). A screen that does not match, or an item whose code exists but cannot be reached from the UI or the command line, is a gap: fix it when it is small, otherwise list it.
4. Go over the spec file by file ({{files}}) against the program, looking for the seams between items, which no single item's test can see: is every flow reachable from where the spec says it starts, does every format both read and write, is every parameter used where its row says, does the shell show every screen.
5. Write `{{out_dir}}/README.md` (create it, or update it if the project has one; short and true): what this program is, in one paragraph from the spec; how to build it; how to run it, with the exact commands, any environment the container needed (such as `GSK_RENDERER=cairo`), and where a sample project or fixture is; how to run the tests; what the spec covers; and what is not done: the blocked items below, and every gap you found and did not fix.
6. Do not commit: codehalter commits this pass once the test command passes.

Finish with `respond`: three lines the user can act on, how to build, how to run, what to try first; then the gaps, if any.

## Blocked items

{{blocked}}
