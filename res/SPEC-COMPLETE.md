# Completion check: `{{file}}`

The items below of the specification in `{{spec_dir}}/` count as done: a test is named after each, and the suite passes. That does not say the program does what the spec says. A test can pass against a stand-in (a refusal, a scripted answer, a helper only tests fill), and a screen can have every widget in the wrong place. This round checks each item against its spec text. It changes nothing.

## Target

{{target}}

## The items

{{items}}

## Do, for each item

1. Read its section in `{{file}}` (the line is given) and what it links to.
2. Find the code in `{{out_dir}}/` that does it, starting where a user starts it: the button, menu entry or command in the UI, or `main`. Follow the calls. A test that calls a function directly proves nothing here.
3. Check that the program does every step and branch the text names: what the user sees (every widget, label and message, and where it sits), what it reads and writes (the files, their format), what it calls (a server, a tool), and what happens when that fails. A service the program never calls, a function that only refuses or only answers when a test loads a script, a screen or a step left out: that is missing.
4. Where the spec pictures the item, render the screen the way this project renders its screens (the `snapshot` recipe in `{{out_dir}}/`), and look at it and at the spec's picture with `screenshot`. What sits where (left or right, top or bottom, which panel holds what), which widgets and labels, and their order must match; pixels, colours, fonts and spacing need not.

## Answer

Finish with `respond`, one line per item and nothing else:

- `F1.2: DONE` when the program does all of it.
- `F1.2: MISSING what is missing` otherwise, naming each gap concretely, for example `F1.2: MISSING the Save button is not wired; the sources list is on the right, the spec has it on the left`.

Edit nothing: codehalter reopens the missing items with your words, and later rounds build what is missing.
