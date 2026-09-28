# Spec refactor round: `{{out_dir}}/`

A specification in `{{spec_dir}}/` is being implemented one item per round. Every few items one round goes to the shape of the code instead of a new item. This is that round: it implements NO spec item and changes NO behaviour.

This round runs as your usual two phases. In the PLANNING phase, read the targets below and submit a plan of small, separate refactoring steps; the item is already chosen, so `redo` and `spec` do not apply here. In the EXECUTION phase, do the steps.

## Target

{{target}}

The spec in `{{spec_dir}}/` is READ-ONLY. All changes stay in `{{out_dir}}/`.

{{context}}

## What codehalter measured

{{debt}}

{{targets}}

## When this round counts as done

codehalter measures again after the round. It counts when:

1. At least one of the three numbers above went down and none went up.
2. `{{test_cmd}}`, run from `{{out_dir}}/`, passes: the WHOLE suite, with every test that passed before.
3. The project's linter finds nothing in the lines this round wrote.
4. A program file over {{max_lines}} lines does not grow by more than {{growth_slack}} lines.

## How to work

- Behaviour stays exactly as it is. Move code, do not rewrite it: a function moved to a new file keeps its body, and its callers change only their path. Keep every public name the tests use, or update those tests in the same step.
- A file over the budget: move whole groups of related functions (one screen, one flow, one widget family) into a new module next to it, and leave the calls in place. One group per step, and the suite passes after each step.
- A function nothing calls: delete it, and its now unused helpers. If the spec needs it, the item it belongs to calls it in its own round; do not wire it here.
- A test helper copied into several test files: put one copy in a shared test module (Rust: `tests/common/mod.rs` with `mod common;` in each test file; Go: a `_test.go` helper file in the package; Python: `conftest.py`; JS: a helper module the tests import), and delete the copies.
- Do not touch what no target names. Do not add features, tests for new behaviour, or comments that retell the code.
- Run the suite once, last, after your final edit, as its own command: `cd {{out_dir}} && {{test_cmd}}`.

{{previous}}
