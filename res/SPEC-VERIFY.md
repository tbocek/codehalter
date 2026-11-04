# Spec verify: do these changes contradict `{{spec_dir}}/`?

The user asked for changes in chat, and codehalter made them without looking at the specification. The user now asks whether any of them contradicts it. This round finds that out and reports it. It changes nothing: no file, no spec, no ledger.

## The changes

What the chat sessions changed since the last check: each request, and the file edits made for it. Long edits are cut short.

```
{{changes}}
```

These are the steps, not the result: a later edit may have replaced an earlier one, and a change the user committed or undid since looks the same here. Read the file as it is now (`sed -n`, `grep -n`) before you call anything a contradiction.

## Do

1. For each change still in the file, say to yourself what it does for someone using the program.
2. Look in {{files}} for what the spec states about that: grep for the feature, the label, the parameter and other words for them, then read the section. {{context}}
3. A contradiction is a change that makes the program do something other than what a spec line states: a different value, label, order, layout or flow, or something the spec rules out. A change the spec says nothing about, and a change that only adds to or tidies what the spec states, is no contradiction. A build, container or test change is none unless the spec states that very thing.

## Answer

Call `submit_plan` with no subtasks and the whole report in `answer`:

- For each contradiction: the changed file and line, what the change does, the spec file with the line quoted exactly, and the item id if the line has one.
- Then one line naming the changes you could not place in the spec, if any.
- No contradiction: say so, in one line.

Propose no fix. Which side wins is the user's decision: change the code back, or change the spec.
