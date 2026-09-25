# Spec audit: what to redo in `{{out_dir}}/`

Every item of the specification in `{{spec_dir}}/` is recorded as done: a test names it and the suite passes. The user says the program still does not do what the spec says, and asked for the items that fall short to be rebuilt, without naming them. This round names them. It changes nothing.

## Target

{{target}}

{{context}}

## Do

1. Build it (`just build` or the target's release build) and run `{{test_cmd}}` once, into a file. Both should be green; if not, that is the first finding.
2. Render every screen with the `snapshot` recipe and look at each image with `screenshot`, next to the spec's image of that screen where there is one. A bare page, a missing widget, a wrong order, a control the spec names that is not there: note the item ids that screen belongs to.
3. Read the UI sources for what is wired: which buttons, menu entries, shortcuts and rows have a handler that calls the tested code, and which do nothing. Compare with the flows in the spec ({{files}}): a flow whose trigger has no handler, or whose effect never reaches the screen, is a finding.
4. For each finding, name the spec item ids it belongs to. The ids are in the spec files: flows (`F2.3`), parameters (`P.policy.x`), tools (`tool:x`), and headed sections (`§03-shell#1-screen`). Prefer the flow or screen item over the parameter it uses.

## Answer

Call `submit_plan` with `redo` set to the list of item ids, nothing else: no subtasks, no answer. The user sees the list and confirms before anything is reopened. If everything delivers, call `submit_plan` with `report_only=true` and say so in `answer`.
