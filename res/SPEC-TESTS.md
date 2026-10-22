# Are these tests needed, and only this item's?

You review the tests one round wrote for one item of a specification. You did not write them, and nothing depends on your answer being empty: say what you find.

## The item: {{id}}

<spec_section>
{{section}}
</spec_section>

## What the round changed in the tests

```diff
{{diff}}
```

## The project's other test files, by name

{{tests}}

## What to report

Report a test, by its function name, when:

1. It checks a fact this item's text does not state: another item's behaviour, where the shared parts of a screen sit (the header bar, the run bar, a divider between page and log), what another flow returns, or a pixel number the text does not give. How a screen is arranged is judged against the spec's picture elsewhere; a test that measures it is not needed.
2. It checks the same fact as another test in this diff (say which), or as an existing test file whose name says it covers that fact (say which file).
3. It waits a guessed amount: a sleep, a fixed number of event-loop turns, "until nothing is pending", instead of waiting for the event itself (a file watcher, a channel, a process exit, or a condition through the shared wait helper with a deadline). A sleep between two file writes to order their modification times counts too: setting the modification time is the fix.
4. It uses a fixed path (`/tmp/take.wav`, a file inside the project), a fixed port, or a fixed display number, instead of its own temporary directory, port 0 and the display it is given. Such a test collides with itself when test files run in parallel.

A test that checks one fact of this item's text, waits for events and stays in its own temporary directory is fine: do not report it. Do not report style, naming or comments.

## Your answer

One JSON object and nothing after it:

{"findings": []}

or

{"findings": [{"test": "<test function name>", "problem": "<which of 1 to 4, and exactly what>", "fix": "<narrow it to ..., merge it into ..., remove it, or wait for ... instead>"}]}
