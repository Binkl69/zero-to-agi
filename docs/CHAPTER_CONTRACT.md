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

## Pedagogy — the non-negotiable structure

The reader is a smart adult with no ML background who feels "behind". Each chapter:

1. **Hook** (2–3 paragraphs): a concrete real-world situation, then the question this chapter answers.
2. **Mechanism**: explain the actual thing, with a diagram or animation. Introduce jargon only after
   the idea, then name it (`<em>` the term the first time). Use one worked numeric example where possible.
3. **Interactive(s)**: at least ONE substantial interactive per chapter (two is better) where the reader
   changes something and sees the consequence. Precede each with a `tryit` callout that tells them
   exactly what to try and what to notice.
4. **Real-world examples** (`example` callouts): where this idea shows up in products they use.
5. **History** (`history` callout): who, when, and why it mattered — one or two paragraphs.
6. **Why it matters for modern AI**: connect to today's LLMs/agents.
7. **Quiz**: 3–5 questions with explanations.
8. **Go deeper**: 3–5 links (papers, videos, code) as a `ctx.ul` of `<a href target="_blank">`.

Aim for 1200–2000 words of prose per chapter plus the interactives. Write like a great teacher,
not a textbook: short sentences, vivid analogies, no hedging, no fluff. Be precise and correct;
if simplifying, say so.

## Quality bar

- Run `node --check app/chapters/NN-slug.js` — it must pass.
- No runtime errors. Guard against NaN in training demos (clip gradients, reset button).
- Interactives must work with mouse AND touch (use `pointerdown/pointermove/pointerup`).
- Never block the main thread: do at most a few hundred training steps per animation frame.
- Do not use `alert`, `fetch`, or any network call.
