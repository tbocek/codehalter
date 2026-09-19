# Rendered-layout skill (CSS, HTML templates)
A stylesheet or template edit has no compile step and no unit test: CSS always parses, so a WRONG rule fails exactly like a right one, silently. "Should look better now" is not a check. It costs a user turn per attempt and answers only "still wrong", never why. Measure the rendered page instead.

## Look with `screenshot`, measure with the probe
Two different questions, two different tools. Reach for `screenshot` first: it needs no `run_command`, so it works on a host run too.
- `screenshot(path="out/page/index.html", selector=".grid-video-2")` renders with headless Firefox and puts the picture in front of you. `selector` lifts one block to the top of the shot, so a long page costs one block, not a poster. Use it to find WHETHER something is off and WHERE: overlap, clipping, a caption hidden behind a player, the wrong column.
- The reply says whether the selector matched. `matched NO element` means you are looking at the top of the page, not at your block: fix the selector before reading anything into the image.
- A picture cannot give you a distance. "Too big a gap" needs a pixel count, and that is the probe below.
- The tool reports that this LLM takes no image input? Then you cannot see, and neither can the probe help you skip that fact: say so and use the numbers alone.

## Measure BEFORE editing, in the BUILT page
Turn the complaint into ONE number read off the artifact the user actually sees (the generated `.html` in the output dir, NOT the source template it came from). Take it before the first edit: with no baseline you cannot tell a fix from a coincidence.
Headless Firefox lays the page out and a small probe reports the number as TEXT. Needs `run_command` (devcontainer), `firefox` and `python3`.

1. Receiver: prints the first request's query, then exits:
```
mkdir -p /tmp/ffp && cat > /tmp/rx.py <<'EOF'
import http.server, urllib.parse
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        print(urllib.parse.unquote_plus(urllib.parse.urlparse(self.path).query), flush=True)
        self.send_response(200); self.send_header('Content-Length','0'); self.end_headers()
        raise SystemExit(0)
    def log_message(self, *a): pass
http.server.HTTPServer(('127.0.0.1', 8099), H).serve_forever()
EOF
```
2. Instrumented COPY of the built page (never edit the original). `<base>` keeps the copy finding the real CSS/images/fonts; edit only `SEL` and the arithmetic:
```
python3 - /path/to/out/page/index.html <<'EOF'
import sys, os
src = sys.argv[1]; html = open(src, encoding='utf-8').read()
probe = r'''<script>window.addEventListener('load', function () {
  var out = [];
  document.querySelectorAll('SEL').forEach(function (el, i) {
    var r = el.getBoundingClientRect(), pr = el.previousElementSibling.getBoundingClientRect();
    out.push(i + ' gapAbove=' + (r.top - pr.bottom).toFixed(1));
  });
  var x = new XMLHttpRequest();
  x.open('GET', 'http://127.0.0.1:8099/?' + encodeURIComponent(out.join('\n')), false);
  try { x.send(); } catch (e) {}
});</script>'''
base = '<base href="file://%s/">' % os.path.dirname(os.path.abspath(src))
open('/tmp/m.html','w',encoding='utf-8').write(
    html.replace('<head>', '<head>' + base, 1).replace('</body>', probe + '</body>', 1))
EOF
```
3. Render and read the numbers. `--screenshot` is only a "load the page, then quit" driver:
```
(python3 /tmp/rx.py > /tmp/m.log 2>&1 &) ; sleep 1
firefox --headless --no-remote --new-instance --profile /tmp/ffp --window-size 1400,1200 \
        --screenshot /tmp/m.png file:///tmp/m.html ; cat /tmp/m.log
```
- The XHR MUST stay synchronous (the `false` argument), or the request loses the race with Firefox exiting.
- `--no-remote --new-instance --profile` or Firefox attaches to a running instance and hangs until the timeout.
- Measuring text, not an element? `var r = document.createRange(); r.selectNodeContents(node);` then `r.getBoundingClientRect()`.
- Report EVERY match, not the first: one bad block among 40 good ones is the whole bug.

## One rule per measurement
- Change ONE rule, take the SAME number again, report both (`16.0 -> 1.9`). Never "should be better now".
- Number unchanged? Your DIAGNOSIS was wrong, not your value. REVERT that edit before trying another. Never stack a second guess on the first.
- A number that only proves what you were looking at: check the rest too (no overlap, nothing clipped, other pages using the same class). A metric that says "clean" for a broken layout is worse than none.

## Space you did not write comes from something you did not look at
Unwanted space between A and B is exactly one of: margin, padding, gap, line-height/descender, LEFTOVER TRACK SPACE. Name WHICH from the measurement before touching a property.
- Leftover track space: the row is taller than its content because something ELSE sizes it, usually an item in a DIFFERENT column spanning those rows. `align-items: end` then drops that slack ABOVE every item, which reads as "a gap under the title".
- So don't ask "which property adds this space", ask "which ELEMENT makes this track that tall", and measure that track's items to find it.
- Fix the sizer, not the symptom: give the tall spanning item its own `1fr` spacer track to absorb (an item spanning a flexible track stops contributing to the intrinsic tracks it spans, CSS Grid 12.5), and put that spacer LAST or auto-placed items land in it.
- Every child of a grid/flex container is its own item, including each inline element and each text run. A caption with `<b>` in it is 3 items, not 1.
- A comment claiming WHY a rule works is a claim like any other: measured, or not written.

## Nothing to render with
No `screenshot`, no `run_command`, no browser and no way to install one: say exactly that in your FIRST reply and ask how the user wants it verified. Do not guess a second time.
