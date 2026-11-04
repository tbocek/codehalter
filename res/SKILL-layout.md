# Rendered-layout skill (CSS, HTML templates)
A wrong CSS rule fails as silently as a right one: it parses, and no test catches it. Measure the rendered page; "should look better now" is not a check.

## Look with `screenshot`, measure with the probe
- `screenshot(path="out/page/index.html", selector=".grid-video-2")` renders with headless Firefox and shows you the picture, with `selector` at the top. Use it to find whether and where something is off: overlap, clipping, the wrong column. It needs no `run_command`.
- `matched NO element` in the reply means you are looking at the top of the page: fix the selector first.
- A picture gives no distance. A gap needs a pixel count, from the probe below.
- If this LLM takes no images, say so and use the numbers alone.

## Measure BEFORE editing, in the BUILT page
Turn the complaint into ONE number from the generated `.html`, not the source template. Take it before the first edit, or you cannot tell a fix from a coincidence. Needs `run_command`, `firefox` and `python3`.

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
- Keep the XHR synchronous (the `false` argument), or it loses the race with Firefox exiting.
- Without `--no-remote --new-instance --profile`, Firefox attaches to a running instance and hangs.
- For text, not an element: `var r = document.createRange(); r.selectNodeContents(node);` then `r.getBoundingClientRect()`.
- Report EVERY match: one bad block among 40 good ones is the whole bug.

## One rule per measurement
- Change ONE rule, take the SAME number again, report both (`16.0 -> 1.9`).
- Number unchanged: your diagnosis was wrong. REVERT that edit before trying another.
- Then check the rest: no overlap, nothing clipped, other pages using the same class.

## Space you did not write
Unwanted space between A and B is one of: margin, padding, gap, line-height or descender, or leftover track space. Name which from the measurement before touching a property.
- Leftover track space: something else sizes the row, usually an item in another column spanning it, and `align-items: end` drops the slack above every item. Ask which ELEMENT makes the track that tall, and measure that track's items.
- Fix the sizer: give the tall spanning item its own `1fr` spacer track, placed LAST, or auto-placed items land in it.
- Every child of a grid or flex container is its own item, including each inline element and text run: a caption with `<b>` in it is 3 items.

## Nothing to render with
No `screenshot`, `run_command` or browser, and no way to install one: say so in your FIRST reply and ask how the user wants it verified.
