Compress this running set of turn notes into a shorter set of turn notes. It is the entire memory of everything that happened before the messages still in context, so what you drop is gone for good.

Keep the same shape as the input: notes in chronological order, each note using these section lines, skipping sections that don't apply. One line per section, terse, no fluff. Reply with just the notes, no preamble.

Goal: what the user wants overall
Constraints: rules/requirements/preferences the user stated
Tasks: the active plan's subtasks, numbered, status + description
Progress: concrete progress (files changed, cmds run, info gathered)
Decisions: choices made or directions taken
Next Steps: what's queued or open
Critical Context: paths, identifiers, versions, or state that must not be lost

How to compress:
- MERGE consecutive notes about the same piece of work into one note. Most of the input is the same task described several times as it progressed; the final state is what matters.
- COLLAPSE finished work. A task that is done, verified, and superseded becomes one Progress line. Its intermediate steps, failed attempts, and abandoned approaches go.
- KEEP open work at full detail. Anything in Next Steps, any `[todo]` or `[doing]` task, and anything the user has not yet answered stays as-is.
- DROP notes that record no outcome at all (a phase that only "initiated" or "reviewed" something and produced nothing).

Never drop, no matter how old:
- Any constraint recorded as locked ("… per user, do not revert"). These are standing user instructions.
- Any `img_<hex>` image id, and any `- img_…` reference line that names a `view_image` id. Copy those lines through verbatim into the note they belong to. Image retrieval breaks without them.
- Absolute paths, version numbers, process ids, port numbers, and identifiers that later work will need to name.

Aim for roughly half the input length. Losing detail about finished work is correct. Losing an open question, a locked constraint, or an identifier is not.
