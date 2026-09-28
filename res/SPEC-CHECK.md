# Before {{id}} counts

The round's work is in place, and codehalter checked it before counting it. This still stands in the way:

{{findings}}

Fix exactly this, in `{{out_dir}}/`, and nothing else:

- A function the program does not call: call it from the path the spec describes (a UI handler, `main`, the flow that owns it), or delete it. A helper that exists only for tests says so in its name (`rows_for_test`).
- A lint finding: fix the line it names.
- A UI change nobody looked at: render the screen with the `snapshot` recipe and look at it with `screenshot`.
- A test that does not carry the item's name: rename it or add one; a comment or a string does not count.
- A file over its size budget: move the new code into a new file and call it from there.

Then run the suite once, last, after your final edit, as its own command: `cd {{out_dir}} && {{test_cmd}}`.
