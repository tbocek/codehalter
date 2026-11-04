# Spec round: {{id}} {{title}}

You implement a specification one item per round. codehalter chose this round's item: **{{id}}** ({{title}}), defined in `{{file}}`. Progress so far: {{progress}}.

In the PLANNING phase, read what you need and submit a plan whose subtasks build this one item and its tests; an answer instead of a plan does not finish an item. In the EXECUTION phase, do the subtasks.

## Target

{{target}}

All new code goes in `{{out_dir}}/`. The spec in `{{spec_dir}}/` is the source of truth and READ-ONLY. You may change anything outside both (toolchain config, `.devcontainer/`) when the work needs it.

{{context}}

## When this item counts as done

codehalter checks this itself after the round; a summary or a claim does not count.

1. A test this round adds or changes under `{{out_dir}}/` has `{{token}}` in its function name (`{{token}}_s1_...`) or call title (`it("{{token}} ...")`). A comment or string does not count, nor does an older test the round did not touch.
2. `{{test_cmd}}`, run from `{{out_dir}}/`, passes: the WHOLE suite, including earlier items' tests.
3. If the spec starts this item with a user action (a button, menu entry, shortcut, drop, click on a row), one test goes through the real widget: build the screen, find the widget by name, fire the action the way the toolkit does (GTK: `emit_by_name::<()>("clicked", &[])`, `activate_action`, a key event; Qt: `QTest::mouseClick`, `trigger()`), and assert the effect through the state the logic test checks.
4. The program calls every function this round adds outside the tests, from the path the spec describes (a UI handler, `main`, the owning flow). A helper only for tests says so in its name (`rows_for_test`).
5. The program does the work itself. When the item needs a service, file format or device, the program calls, writes or drives the real one, and a test replaces it through a seam the program itself uses, such as a server URL pointed at a local fake. Program code never asks a for-test helper for its answer, and never answers with a refusal, "not implemented", `todo!()` or a scripted reply.
6. The project's linter finds nothing in the lines this round wrote. Linter: {{lint_cmd}}.
7. A program file over {{max_lines}} lines grows by at most {{growth_slack}} lines in a round: put new code in a new file, and call it from there.

## Read the spec before you plan

The section at the end of this prompt is where the item starts, rarely all of it. In the PLANNING phase, before the plan:

1. Collect every place that specifies this item: its section; the sections it links (appended below); the chapter's screen description; any table the chapter calls normative (labels, options, defaults, tooltips, sizes); the standing-rules files named above; the answers to this item's questions. Open with the shell those not included below.
2. If the item shows a screen, list every part its picture shows, with the sentence that specifies it and that sentence's file. For a part with no sentence, search the whole spec for its name before deciding the spec is silent. Do not build a part the text calls gone or replaced. Build to the sentences, and take from the picture what they leave open: arrangement, order, which control is a menu.
3. Where two places disagree, the more specific decides: an answer over the section, the section over the picture, a normative table over a summary sentence. A thing has one place: do not draw a second copy of what the window or another page already shows.
4. For every number, find what it measures and which code already uses it. Where the program already computes what the screen shows, draw from that same code.
5. Name the facts you will test, each beside its sentence. A test asserts the number the spec states, not the one your code produces.

Put the list from 2 and the facts from 5 into the plan's first subtask.

## How to work

- Read what exists in `{{out_dir}}/` first; extend and reuse earlier rounds' code.
- Implement exactly what the spec text says. Test every numbered step and branch; name step tests `{{token}}_s1_...`, `{{token}}_s2_...`.
- Parameters (`P.*`) and tools (`tool:*`) are items of their own. Where your code uses one, name a test that checks it after the row's token; the row is then done without its own round. The rows cited here, with their tokens: {{row_tokens}}.
- Keep behaviour in plain, testable code and the UI layer thin: logic inside a UI callback cannot be tested. Give every interactive widget a stable name (`set_widget_name`, from the spec's label, such as `add-source`).
- Write only the tests this item needs: one per fact its text states, through the item's own entry point. Do not assert another item's facts. Extend a test that already checks a fact rather than writing a second. Assert a pixel number only where the spec states it; codehalter judges arrangement from your render. codehalter reviews the tests and sends back any that check another item's fact, repeat a test, wait a guessed amount, or use a fixed path, port or display.
- Tests are deterministic under load and may run in parallel. They use the project's loopback fakes, never a live server, and assert the state their action changes, not a status line other code also writes.
- A new test joins the project's one test program. Where every test file builds its own program (Rust's `tests/*.rs`): put it in `tests/suite/` with a `mod` line in `tests/suite/mod.rs` (create `tests/all.rs` with the shared test module and `mod suite;` if missing), never a new `[[test]]` target. A widget test runs on the toolkit's thread through its helper (gtk4-rs: `#[gtk::test]`) and builds the screen it checks, without starting the application.
- A test never waits a guessed amount: no fixed sleep, no fixed number of event-loop turns, no "until nothing is pending". Await a condition through the one wait helper in the shared test module (add it if missing), failing after a wall-clock deadline (10 s, say) with a message naming what never happened. To make one file newer, set its modification time. A test with a fake clock must make every timer the code relies on read that clock.
- Each test writes only into its own temporary directory (the platform's temp-dir call, which follows `$TMPDIR`), binds port 0 for a fake server, and uses the display it is given (`$DISPLAY`). codehalter runs the tests with a fresh `$TMPDIR` and reports anything written outside it.
- Where the spec is silent, decide in its spirit and record the decision in a code comment. Where it contradicts itself or the target, ask. A layout, label or order stated elsewhere in the spec is not a question: follow it. Code that does not match the spec yet is work, not a contradiction. To ask, call `submit_plan` with clear=false, no subtasks, `question` (one sentence naming the decision, understandable without the code), `spec_quote` (the closest spec text, copied exactly) and `options` (2 or 3, your pick first, each a short `choice` and an `example` of what the user would see or the program would do).
- If you changed a UI source file, render the screen with the `snapshot` recipe and look at it with `screenshot`: is every widget the text names there, in order, with nothing empty, overlapping, cut off or unlabeled? If the item shows a screen (see "Screens" at the end), compare with the spec's image: widgets, order, labels and tooltips must match, exact pixels, colours and spacing need not. Everything runs here headless: never ask for a display. A UI change without a look does not count.
- Run the suite once, last, after your final edit, as its own command: `cd {{out_dir}} && {{test_cmd}}` (a redirection to a file is fine, a subshell does not count). A green run then counts as the check.
- Build only this item. Linked sections are there to be read; their rounds will come.
- Format only the files you changed (`rustfmt <file>`, `gofmt -w <file>`), never the whole project.

{{previous}}

## The spec for this item

{{slice}}
{{screens}}
{{related}}
