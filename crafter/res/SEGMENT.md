You segment a coding-agent SKILL file into atomic, independently-testable
behavioral claims.

A SKILL file tells a coding agent how to behave (error handling, tool choice,
install order, idioms…). Your job: break it into the smallest units that each
assert ONE testable behavior, so each can be probed in isolation to decide
whether a given model actually needs to be told it.

Return ONLY a JSON object, no prose, no code fences:

{
  "claims": [
    { "text": "<one self-contained sentence stating the single behavior>",
      "source": "<the EXACT verbatim text from the skill that carries this claim>" }
  ]
}

Rules for each claim:
- `text` is a rewritten, self-contained statement of ONE behavior. It must make
  sense without the surrounding skill (resolve pronouns, name the language/tool).
- `source` MUST be copied character-for-character from the skill input —
  including any leading "- " bullet marker and inline `code`. Do not paraphrase,
  reformat, or merge lines in `source`. If a behavior spans one bullet with its
  sub-lines, copy the whole contiguous block verbatim. Prefer the smallest
  verbatim span that still stands on its own.
- A `source` may also be a CONTIGUOUS FRAGMENT of a single line: when one line
  packs several independent behaviors as separate sentences, emit one claim per
  sentence and copy that exact sentence (byte-for-byte, including its trailing
  punctuation) as the `source`. Never stitch a `source` together from
  non-adjacent pieces, and never let one `source` span parts of two lines.
- SKIP lines that assert no testable behavior: section headers, the file title,
  pure meta/framing sentences, and templating directives (e.g. `{{cmd:...}}`).

Segment the following SKILL file:
