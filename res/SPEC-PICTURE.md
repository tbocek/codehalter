# Does the screen match its picture?

Check one screen of a program against the spec's picture of it. You did not build it, and nothing depends on a pass: say what you see.

1. The first image is the spec's picture, `{{picture}}`, taken of another version on other data.
2. The second is the program's render of the same screen, `{{render}}`.

## What to ignore

Data differs and is not a difference: file names, numbers, counts of rows or clips, text in text boxes, thumbnails, theme colour. Eight rows in the picture and one in the render is the same list.

The picture may be of a prototype. A control the spec text calls gone, removed, replaced or prototype only is not a difference when the render lacks it, and its replacement is not a difference when the render has it. Check the spec text before listing a control as missing or extra, in all of the text below.

## What to compare

Compare the ARRANGEMENT, region by region, from the top left of the picture:

- Is every region and control there, in the same place relative to the others (left or right, above or below, in or under the header bar, which side of a divider)?
- Is anything cut off, or does a page start mid-content because its top is scrolled out of view?
- Do any two things overlap?
- Is there a large empty area where the picture has content, or a region far larger than in the picture?
- Are controls in a row or grid in the same order and grouping?

Below are the spec chapter, sections of other chapters it links to (headed `--- <file> · <title> ---`), the standing rules, and the user's answers, which are spec too. The spec's words decide what must match; where they say nothing, the picture does.

<spec_chapter>
{{section}}
</spec_chapter>

## Your answer

Answer with one JSON object and nothing after it:

{"verdict": "MATCH", "differences": []}

or

{"verdict": "MISMATCH", "differences": ["<where on the screen>: <what the picture shows> but <what the render shows>", "..."]}

Name every difference in arrangement, each with where it is, for someone who cannot see the images. Anything cut off, overlapping or scrolled out of view makes the verdict MISMATCH.
