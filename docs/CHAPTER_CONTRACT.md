# Chapter authoring contract

Every chapter is ONE plain JavaScript file in `app/chapters/NN-slug.js`, loaded by a `<script>` tag
from `index.html` (already listed there — use the exact filename). No ES modules, no `import`,
no external libraries, no build step. The app must work from `file://` and from GitHub Pages.

## Registration

```js
(function () {
  ZTA.registerChapter({
    id: '02-neural-networks',        // must equal the filename without .js
    num: 2,
    part: 'I',                        // 'I' | 'II' | 'III' | 'IV' | 'V'
    title: 'Neural networks & backpropagation',
    tagline: 'One sentence a curious adult would find intriguing.',
    render(root, ctx) {
      // build DOM with ctx helpers and root.append(...)
    },
  });
})();
```

`render` is called every time the user opens the chapter and its DOM is discarded when they leave.
Anything started with `ctx.loop` / `ctx.interval` is stopped automatically. Register any other
cleanup with `ctx.onCleanup(fn)`.

## The `ctx` toolkit (defined in `app/js/core.js` — read it)

| helper | purpose |
|---|---|
| `ctx.h(tag, attrs, ...children)` | DOM builder. attrs: `class`, `style` (object), `html` (innerHTML), `onclick`/`oninput`/…, `dataset`. |
| `ctx.p(html)` / `ctx.ul([...html])` / `ctx.ol([...])` | prose. Strings are HTML, so `<b>`, `<em>`, `<code class="inline">` work. |
| `ctx.section(title, ...children)` | `<h2>` section. `ctx.sub(title, ...)` for `<h3>`. |
| `ctx.callout(kind, title, html)` | kinds: `example` (real-world example), `key` (key idea), `history` (who/when), `tryit` (instructions for the interactive), `warning`. |
| `ctx.code(text)` | code block. `ctx.table(headers, rows)`, `ctx.cards([{title, body}])`, `ctx.pill(text, color)`. |
| `ctx.canvas(w, h)` → `[canvasEl, ctx2d]` | DPR-aware canvas with logical size w×h (use 720×360-ish; it scales to container width). `canvasEl.pos(event)` gives logical mouse coords. |
| `ctx.figure(body, caption, controlsArray, readoutEl)` | card wrapper around a canvas or element, with caption and controls row. |
| `ctx.slider({label, min, max, step, value, onChange, fmt})` | returns control element with `.value`. |
| `ctx.select({label, options:[{value,label}], value, onChange})` | dropdown, `.value`. |
| `ctx.button(label, onClick, extraClass)` | `.btn.small` button ('primary' as extraClass for emphasis). |
| `ctx.textarea({label, value, onChange})` | multi-line input, `.value`. |
| `ctx.readout()` | element with `.set({ 'loss': '0.123', 'epoch': 42 })` for live numbers. |
| `ctx.loop(fn(dt, t))` | requestAnimationFrame loop; returns `{stop(), start(), running}`. |
| `ctx.interval(fn, ms)` | setInterval with auto-cleanup. |
| `ctx.tabs([{label, content}])` | tabbed panels. |
| `ctx.quiz([{q, options:[...], answer: idx, explain}])` | 3–5 questions at the end of every chapter. |
| `ctx.rand(a,b)`, `ctx.randn()`, `ctx.clamp`, `ctx.lerp`, `ctx.heat(v)`, `ctx.colors.*` | numerics + palette. |

Canvas background is `#0a0e16`; draw with `ctx.colors` (accent blue `#7c9cff`, green `#38d9a9`,
warn yellow `#fbbf24`, danger red `#fb7185`, pink, purple, orange, muted grey). Text colour `#e6ebf5`.
Always `clearRect` and redraw the whole canvas each frame. Keep each animation cheap (<2 ms/frame).

## Pedagogy — INTERACTIVE FIRST. This is the whole product.

The reader is a smart adult with no ML background who feels behind, and who learns by doing,
not by reading. **`app/chapters/01-what-is-learning.js` is the reference implementation. Read it
before writing anything.**

The governing rule: **the reader touches something before they read anything, and every paragraph
that exists is there to explain something they just did with their own hands.** A paragraph that
explains a thing the reader has not yet experienced is in the wrong place. Move the interactive up.

### The loop, repeated 5–9 times per chapter

```
tryit callout  ->  INTERACTIVE  ->  1–2 short paragraphs on what they just saw  ->  next idea
```

Never: three paragraphs, then a demo. Always: demo, then the short explanation that now lands.

### Hard budget (enforced by `node scripts/smoke.js`, which FAILS the chapter otherwise)

| Rule | Limit |
|---|---|
| Interactives (`ctx.figure`) per chapter | **5 minimum**, 6–9 is the target |
| Prose words before the FIRST interactive | **120 max** (chapter 01 does it in 24) |
| Any single paragraph | **110 words max**, aim for 25–70 |
| Total rendered words | 900 minimum |

### Thorough, not thin

Short paragraphs are not an excuse to cover less. **Be longer and more thorough than the old
text-heavy chapters were, not shorter.** Aim for **1,500–2,500 words of prose**, chopped into
25–70 word blocks and spread between many interactives. Nothing from the old chapter's substance
should be lost. Redistribute it: a long explanation usually becomes a short paragraph plus a
figure caption plus a callout, and is better for the split.

Figure captions carry real teaching weight. Use them for the detail that would otherwise bloat a
paragraph.

### Still required, woven through

- **Jargon after the experience.** Let them do the thing, then name it with `<em>` on first use.
- **Notation after the intuition.** The same rule, applied to maths. A symbol may only appear
  after the reader has already done the thing it describes, and its job is to *name that
  experience*, not to introduce it. **There is no standalone maths chapter and never will be** —
  a wall of maths is where a reader quits. Instead every chapter carries **one maths beat**:
  at most one interactive plus one short callout, 3–5 minutes, always attached to something
  their hands already did. If a beat needs more room than that, it is in the wrong chapter.
  Prefer the statistical framing over the calculus framing wherever both are available.
  See `docs/ROADMAP.md` for which beat belongs to which chapter.
- **A worked numeric example**, ideally as a live panel where they change the numbers.
- **`example` callouts**: where this shows up in products they already use.
- **A `history` callout**: who, when, why it mattered.
- **Why it matters for modern AI**: connect to today's models and agents.
- **`quiz`**: 5 questions, each with an explanation.
- **Go deeper**: 3–5 real links as a `ctx.ul` of `<a href target="_blank" rel="noopener">`.

### Designing an interactive that teaches

Good ones let the reader **fail informatively**, or make an invisible quantity visible and
draggable. The best in chapter 01 lets you try an impossible task until you feel why it is
impossible. Prefer:

- a thing the reader drives by hand, before the machine does it automatically
- a live counter or score that reacts to what they changed
- a preset that is deliberately broken, so they can see the failure mode
- a slider that crosses a threshold where behaviour visibly changes

Avoid decorative animations that the reader cannot influence. If it has no control, it should be
a diagram that responds to hover at minimum.

Write like a great teacher: short sentences, vivid analogies, no hedging, no fluff. Be precise;
if you simplify, say so.

## Quality bar

- Run `node --check app/chapters/NN-slug.js` — it must pass.
- No runtime errors. Guard against NaN in training demos (clip gradients, reset button).
- Interactives must work with mouse AND touch (use `pointerdown/pointermove/pointerup`).
- Never block the main thread: do at most a few hundred training steps per animation frame.
- Do not use `alert`, `fetch`, or any network call.

## Cache busting

Every asset URL in `index.html` carries a `?v=N` query string. GitHub Pages serves
JS and CSS with a long cache lifetime, so without this a reader who has visited
before keeps running the old chapter files after you push changes.

**When you change any file under `app/`, bump every `?v=N` in `index.html`:**

```bash
sed -i -E 's/\?v=[0-9]+/?v=3/g' index.html
```

There is no build step to do this automatically. If a change does not show up in a
browser, this is almost always why; a hard reload (Ctrl+Shift+R) confirms it.
