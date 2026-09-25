# Spec round: {{id}} {{title}}

You are implementing a specification one item per round. codehalter picked this round's item from the spec's ledger: **{{id}}** ({{title}}), defined in `{{file}}`. Progress so far: {{progress}}.

## Target

{{target}}

All new code goes in `{{out_dir}}/`. The spec in `{{spec_dir}}/` is the source of truth and is READ-ONLY: codehalter refuses edits there. Anything outside both (toolchain config, `.devcontainer/`) you may change when the work needs it.

{{context}}

## When this item counts as done

codehalter checks this itself after the round. Nothing else counts, not a summary and not a claim:

1. A test under `{{out_dir}}/` names this item. Either put `{{token}}` in the test function's name (for example `{{token}}_s1_...`), or write `{{id}}` in a comment or string inside the test.
2. `{{test_cmd}}`, run from `{{out_dir}}/`, passes. The WHOLE suite, not only the new tests: an earlier item's test that you break counts against this round.
3. If the spec text starts this item with a user action (a button, a menu entry, a shortcut, a drop, a click on a row), one of its tests goes through the real widget: build the window or the screen, find the widget by its name, fire the action the way the toolkit does (GTK: `emit_by_name::<()>("clicked", &[])`, `activate_action`, a key event; Qt: `QTest::mouseClick`, `trigger()`), and assert the effect through the same state the logic test checks. The logic test proves the rule; this one proves the wire. A program where every rule is tested and no click reaches any of them passed every round once, and nothing in it worked.

## How to work

- Read what already exists in `{{out_dir}}/` before writing. Earlier rounds implemented other items; extend and reuse their code, do not write a second version of something that is there.
- Implement exactly what the spec text below says. Every numbered step (S1, S2, ...) and every branch should be exercised by a test; name step tests `{{token}}_s1_...`, `{{token}}_s2_...`.
- Parameters (`P.*`) and tools (`tool:*`) this text cites are listed with their rows. Where your code uses one, name it in a test too (`// P.policy.reviewPadSeconds`): codehalter tracks those ids as well, and a test that names them now saves a round later.
- Keep behaviour in plain, testable code (state, rules, data formats, flow steps) and the UI layer thin: it renders state and forwards user actions. A step whose logic lives inside a UI callback cannot be tested, so it can never count as done. Thin is not absent: the callback must exist and call the tested function, and for a user action that is what the widget test above checks. Give every interactive widget you add a stable name (`set_widget_name`, from the spec's label, for example `add-source`), so tests and the snapshot entry point can find it.
- Where the spec is silent, decide in its spirit and in line with the target, and write what you decided in a code comment. Where the spec contradicts itself or the target, ask instead of guessing.
- If this item's text shows a screen (the spec images listed under "Screens" at the end), compare it with the real thing: render the app's matching screen with the `snapshot` recipe, then look at both images with `screenshot`, the spec's and yours. Widgets, their order, labels and tooltips must match; exact pixels, colours and spacing need not. Everything runs inside this container, headless: never ask for a display.
- Run the suite once, last, after your final edit, as its own command: `cd {{out_dir}} && {{test_cmd}}` (a redirection to a file is fine). A green run then counts as the check and codehalter does not run the suite a second time. Do not wrap it in a subshell or append `echo exit=$?`: the exit code is then the echo's and the run does not count.
- If you changed a UI source file, look at the result before you finish, whether or not the spec has a picture of it: render the screen with the `snapshot` recipe and look at the image with `screenshot`. Is every widget the text names on it, in the order it says; is nothing empty, overlapping, cut off or unlabeled? A UI change without a look does not count as done.
- Stay on this item. Another item's section is linked below only so you know it exists; its round will come.

{{previous}}

## The spec for this item

{{slice}}
{{screens}}
{{related}}
