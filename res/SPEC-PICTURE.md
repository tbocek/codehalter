# Does the screen match its picture?

You check one screen of a program against the picture the spec gives of it. You did not build this screen and nothing depends on your answer being a pass: say what you see.

Below are two images:

1. The spec's picture, `{{picture}}`. It was taken of another version of the program, on other data.
2. The program's own render of the same screen, `{{render}}`, as the program draws it now.

## What to ignore

The data differs, and that is not a difference: other file names, other numbers, other counts of rows or clips, other text in the text boxes, other thumbnails or frames, another theme colour. A list with eight rows in the picture and one row in the render is the same list.

The picture can be of an older version (a prototype), and the spec text below says where the program is meant to differ from it. A control the spec text calls gone, removed, replaced, or prototype only is not a difference when the render lacks it, and the control that replaces it is not a difference when the render has it. For example, if the spec text says "chain (prototype; replaced by I'm feeling lucky)", a render with an "I'm feeling lucky" button where the picture has a chain dropdown matches. Read the spec text for each control before you list it as missing or extra.

## What to compare

Compare the ARRANGEMENT, region by region, starting at the top left of the spec's picture:

- Is every region and control the picture shows also in the render, in the same place relative to the others (left or right, above or below, in the header bar or under it, which side of a divider)?
- Is anything in the render cut off: a heading, a label, a row or a control that runs past the top, bottom, left or right edge of the window or of its panel, or a page that starts in the middle of its content because its top is scrolled out of view?
- Do any two things in the render overlap: a label drawn over another label or over a control, text drawn over text?
- Is there a large empty area in the render where the picture has content, or one region taking far more of the window than in the picture (for example an empty band under a bar that is a thin row in the picture)?
- Are controls in a row or a grid in the same order and grouping as in the picture (for example two label and control pairs per line in the picture, three in the render)?

Below is the whole spec chapter this screen belongs to. The screen has its own section in it, and other sections describe parts the picture also shows (a bar, a list, a panel). The spec's words decide what must match, wherever in the chapter they stand; where they say nothing, the picture does. Before you list a control as missing or extra, look for it in the whole chapter: a sentence in another section that calls it gone, replaced or prototype only settles it.

<spec_chapter>
{{section}}
</spec_chapter>

## Your answer

Answer with one JSON object and nothing after it:

{"verdict": "MATCH", "differences": []}

or

{"verdict": "MISMATCH", "differences": ["<where on the screen>: <what the picture shows> but <what the render shows>", "..."]}

Name every difference in arrangement you found, each with where it is, so someone who cannot see the images can fix it. A difference only in data is not listed, and neither is a control the spec text says the program dropped or replaced. If the render has any part cut off, overlapping or scrolled out of view, the verdict is MISMATCH.
