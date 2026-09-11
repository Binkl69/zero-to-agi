#!/usr/bin/env node
/* Headless smoke test for chapters.
   Usage: node scripts/smoke.js [chapter-id ...]
   Loads core.js + every chapter in a tiny fake DOM, calls render(), ticks every animation loop
   a number of frames, fires input handlers on sliders/buttons, then runs cleanup.
   Any thrown error is a failure. This catches undefined variables, bad ctx usage and
   NaN-producing training loops; it does not check what things look like. */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.join(__dirname, '..');
const FRAMES = parseInt(process.env.SMOKE_FRAMES || '40', 10);

/* Interactive-first format budget. Chapter 01 is the reference implementation. */
const MIN_INTERACTIVES = 5;   // things the reader can touch
const MAX_INTRO_WORDS = 120;  // prose before the first one
const MAX_PARA_WORDS = 110;   // any single paragraph
const STRUCTURE_ONLY_OK = new Set([]); // ids exempt from the format checks

/* ---------- fake DOM (shared with paint.js) ---------- */
const makeEnv = require('./lib/fakedom.js');
const { window, document, Element, walk, rafQueue, timers } = makeEnv();

const context = vm.createContext(window);
function load(file) { vm.runInContext(fs.readFileSync(file, 'utf8'), context, { filename: file }); }

load(path.join(ROOT, 'app/js/core.js'));
const chapterDir = path.join(ROOT, 'app/chapters');
const files = fs.readdirSync(chapterDir).filter(f => f.endsWith('.js')).sort();
const only = process.argv.slice(2);
let failures = 0;
for (const f of files) {
  const id = f.replace(/\.js$/, '');
  if (only.length && !only.some(o => id.includes(o))) continue;
  const errors = [];
  const origError = console.error;
  console.error = (...a) => errors.push(a.map(String).join(' '));
  const before = window.ZTA.chapters.length;
  try {
    load(path.join(chapterDir, f));
    const ch = window.ZTA.chapters.find(c => c.id === id);
    if (!ch) throw new Error('did not register a chapter with id ' + id + ' (registered: ' + window.ZTA.chapters.slice(before).map(c => c.id).join(',') + ')');
    if (!ch.title || !ch.part || !ch.num) throw new Error('missing title/part/num');
    const root = new Element('div');
    const ctx = window.ZTA.makeCtx();
    ch.render(root, ctx);
    // tick animation loops
    for (let i = 0; i < FRAMES; i++) {
      const q = rafQueue.splice(0);
      for (const fn of q) fn(Date.now() + i * 16);
    }
    // fire timers once
    for (const fn of timers.splice(0)) { try { fn(); } catch (e) { errors.push('timer: ' + (e.stack || e)); } }
    // poke every control: sliders, selects, buttons, canvases
    const ev = (type, extra) => Object.assign({ type, preventDefault() {}, stopPropagation() {}, clientX: 100, clientY: 100, touches: null, pointerId: 1, button: 0, key: 'a', deltaY: 1 }, extra || {});
    let poked = 0;
    walk(root, el => {
      try {
        if (el.tagName === 'INPUT' && el.attributes.type === 'range') { const lo = +el.attributes.min, hi = +el.attributes.max, st = +el.attributes.step || 1; const mid = Math.min(hi, lo + Math.round(((lo + hi) / 2 - lo) / st) * st); el.value = String(+mid.toFixed(10)); el.dispatchEvent(ev('input')); el.dispatchEvent(ev('change')); poked++; }
        else if (el.tagName === 'SELECT') { const o = el.options; if (o.length > 1) { el.value = o[1].attributes.value || o[1].textContent; el.dispatchEvent(ev('change')); poked++; } }
        else if (el.tagName === 'BUTTON') { el.dispatchEvent(ev('click')); poked++; }
        else if (el.tagName === 'CANVAS') { el.dispatchEvent(ev('pointerdown')); el.dispatchEvent(ev('pointermove', { clientX: 150, clientY: 120 })); el.dispatchEvent(ev('pointerup')); el.dispatchEvent(ev('mousedown')); el.dispatchEvent(ev('mousemove')); el.dispatchEvent(ev('mouseup')); el.dispatchEvent(ev('click')); el.dispatchEvent(ev('wheel')); el.dispatchEvent(ev('mouseleave')); poked++; }
        else if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') { el.dispatchEvent(ev('input')); poked++; }
      } catch (e) { errors.push('control ' + el.tagName + ': ' + (e.stack || e)); }
    });
    for (let i = 0; i < FRAMES; i++) { const q = rafQueue.splice(0); for (const fn of q) fn(Date.now() + i * 16); }
    ctx._cleanup();
    rafQueue.length = 0; timers.length = 0;
    const words = root.textContent.split(/\s+/).filter(Boolean).length;
    const quiz = root.querySelectorAll('.quiz').length, canvases = root.querySelectorAll('canvas').length, callouts = root.querySelectorAll('.callout').length;

    /* ---- interactive-first format checks (see docs/CHAPTER_CONTRACT.md) ----
       Walk the rendered tree in document order and measure how much reading the
       reader must do before they can touch anything, and whether any single
       block of prose is a wall. These are failures, not warnings: the format is
       the product. */
    const figures = root.querySelectorAll('.figure');
    let wordsBeforeFirstFigure = 0, sawFigure = false, maxPara = 0, longParas = 0;
    (function walkFmt(el) {
      for (const c of el.children) {
        if (!(c instanceof Element)) continue;
        if (c.classList.contains('figure')) { sawFigure = true; continue; }
        if (c.classList.contains('quiz')) continue;
        if (c.tagName === 'P') {
          const w = c.textContent.split(/\s+/).filter(Boolean).length;
          if (w > maxPara) maxPara = w;
          if (w > MAX_PARA_WORDS) longParas++;
          if (!sawFigure) wordsBeforeFirstFigure += w;
        } else if (c.tagName === 'DIV' || c.tagName === 'SECTION') walkFmt(c);
      }
    })(root);

    const problems = [];
    if (errors.length) problems.push(errors.length + ' runtime error(s): ' + errors[0].split('\n').slice(0, 3).join(' | '));
    if (!quiz) problems.push('no quiz');
    if (!canvases) problems.push('no canvas');
    if (words < 900) problems.push('thin prose (' + words + ' words rendered)');
    if (!STRUCTURE_ONLY_OK.has(id)) {
      if (figures.length < MIN_INTERACTIVES) problems.push('only ' + figures.length + ' interactive(s), need ' + MIN_INTERACTIVES + '+');
      if (wordsBeforeFirstFigure > MAX_INTRO_WORDS) problems.push(wordsBeforeFirstFigure + ' words before the first interactive, max ' + MAX_INTRO_WORDS);
      if (longParas) problems.push(longParas + ' paragraph(s) over ' + MAX_PARA_WORDS + ' words (longest ' + maxPara + ')');
    }
    if (problems.length) { failures++; console.log('FAIL ' + id + ': ' + problems.join('; ')); }
    else console.log('ok   ' + id + ' — ' + words + 'w, ' + figures.length + ' interactives, ' + wordsBeforeFirstFigure + 'w intro, longest para ' + maxPara + 'w, ' + callouts + ' callouts, ' + poked + ' controls');
  } catch (e) {
    failures++;
    console.log('FAIL ' + id + ': ' + String(e.stack || e).split('\n').slice(0, 4).join(' | '));
  } finally { console.error = origError; }
}
console.log(failures ? failures + ' chapter(s) failed' : 'all chapters passed');
process.exit(failures ? 1 : 0);
