# Spec round: {{id}} {{title}}

You are implementing a specification one item per round. codehalter picked this round's item from the spec's ledger: **{{id}}** ({{title}}), defined in `{{file}}`. Progress so far: {{progress}}.

This round runs as your usual two phases. In the PLANNING phase, read what you need and submit a plan whose subtasks build this one item and its tests; the item is already chosen, so `redo` and `spec` do not apply here, and an answer instead of a plan does not finish an item. In the EXECUTION phase, do the subtasks. Everything below is for both phases.

## Target

{{target}}

All new code goes in `{{out_dir}}/`. The spec in `{{spec_dir}}/` is the source of truth and is READ-ONLY: codehalter refuses edits there. Anything outside both (toolchain config, `.devcontainer/`) you may change when the work needs it.

{{context}}

## When this item counts as done

codehalter checks this itself after the round. Nothing else counts, not a summary and not a claim:

1. A test this round adds or changes under `{{out_dir}}/` is named after this item: `{{token}}` in the test function's name (for example `{{token}}_s1_...`), or in the title of a test call (`it("{{token}} ...")`). A comment or a string that mentions the item does not count, and neither does an older test the round did not touch.
2. `{{test_cmd}}`, run from `{{out_dir}}/`, passes. The WHOLE suite, not only the new tests: an earlier item's test that you break counts against this round.
3. If the spec text starts this item with a user action (a button, a menu entry, a shortcut, a drop, a click on a row), one of its tests goes through the real widget: build the window or the screen, find the widget by its name, fire the action the way the toolkit does (GTK: `emit_by_name::<()>("clicked", &[])`, `activate_action`, a key event; Qt: `QTest::mouseClick`, `trigger()`), and assert the effect through the same state the logic test checks. The logic test proves the rule; this one proves the wire. A program where every rule is tested and no click reaches any of them passed every round once, and nothing in it worked.
4. Every function this round adds outside the tests is called by the program, not only by a test: call it from the path the spec describes (a UI handler, `main`, the flow that owns it), or do not write it. A helper that exists only for tests says so in its name (`rows_for_test`).
5. The project's linter finds nothing in the lines this round wrote. Linter: {{lint_cmd}}.
6. A program file over {{max_lines}} lines grows by at most {{growth_slack}} lines in a round: put new code in a new file, and call it from there.

## How to work

- Read what already exists in `{{out_dir}}/` before writing. Earlier rounds implemented other items; extend and reuse their code, do not write a second version of something that is there.
- Implement exactly what the spec text below says. Every numbered step (S1, S2, ...) and every branch should be exercised by a test; name step tests `{{token}}_s1_...`, `{{token}}_s2_...`.
- Parameters (`P.*`) and tools (`tool:*`) this text cites are items of their own, listed with their rows. Where your code uses one, name a test that checks it after the row's token; the row is then done without a round of its own. The rows cited here, with their tokens: {{row_tokens}}.
- Keep behaviour in plain, testable code (state, rules, data formats, flow steps) and the UI layer thin: it renders state and forwards user actions. A step whose logic lives inside a UI callback cannot be tested, so it can never count as done. Give every interactive widget you add a stable name (`set_widget_name`, from the spec's label, for example `add-source`), so tests and the snapshot entry point can find it.
- Where the spec is silent, decide in its spirit and in line with the target, and write what you decided in a code comment. Where the spec contradicts itself or the target, ask instead of guessing. A layout, a label or an order the spec states elsewhere (the chapter's screen description, its image) is not a question: follow it, even outside this item's own section. Code that does not match the spec yet is work to do, not a contradiction to ask about.
- If you changed a UI source file, look at the result before you finish, whether or not the spec has a picture of it: render the screen with the `snapshot` recipe and look at the image with `screenshot`. Is every widget the text names on it, in the order it says; is nothing empty, overlapping, cut off or unlabeled? If this item's text shows a screen (the spec images listed under "Screens" at the end), look at the spec's image too and compare: widgets, their order, labels and tooltips must match; exact pixels, colours and spacing need not. Everything runs inside this container, headless: never ask for a display. A UI change without a look does not count as done.
- Run the suite once, last, after your final edit, as its own command: `cd {{out_dir}} && {{test_cmd}}` (a redirection to a file is fine). A green run then counts as the check and codehalter does not run the suite a second time. Do not wrap it in a subshell or append `echo exit=$?`: the exit code is then the echo's and the run does not count.
- Stay on this item. Another item's section is linked below only so you know it exists; its round will come.

{{previous}}

## The spec for this item

{{slice}}
{{screens}}
{{related}}
