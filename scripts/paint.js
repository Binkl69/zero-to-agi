#!/usr/bin/env node
/* Rendering check for chapters.
   Usage: node scripts/paint.js [chapter-id ...]

   smoke.js answers "does the code run". This answers "does the picture make
   sense". It renders every chapter into the same fake DOM, but with a canvas
   context that records the real geometry of every paint, then drives each
   interactive through a spread of states (slider extremes, every button, every
   select option) and looks for the things a reader would notice:

     text-overlap     two labels printed on top of each other
     line-over-text   a line or curve drawn through a label
     offcanvas-text   a label that runs off the edge of the canvas
     offcanvas-paint  drawing far outside the canvas — always a coordinate bug

   Anything drawn inside an active clip region is accounted for, so a chart that
   correctly clips its plot area reports nothing. */
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const makeEnv = require('./lib/fakedom.js');
const { makeRecorder } = require('./lib/recorder.js');

const ROOT = path.join(__dirname, '..');
const WARM = 4;


const recorders = [];
function canvasFactory(canvas) {
  const r = makeRecorder(canvas);
  const st = new Error().stack || '';
  const line = st.split('\n').find(l => l.includes('app/chapters/') || l.includes('app\\chapters\\'));
  r.__where = line ? line.trim().replace(/^at\s+/, '').replace(ROOT + '/', '') : '?';
  recorders.push(r);
  return r;
}

const env = makeEnv({ canvasFactory });
const { window, document, Element, walk, rafQueue, timers } = env;
const context = vm.createContext(window);
function load(file) { vm.runInContext(fs.readFileSync(file, 'utf8'), context, { filename: file }); }

const { checkFrame: runChecks } = require('./lib/checks.js');

function checkFrame(rec, out, scene) {
  const W = rec.__canvas.width || 720, H = rec.__canvas.height || 400;
  for (const f of runChecks(rec.ops, W, H)) {
    const key = rec.__where + '|' + f.kind + '|' + f.detail;
    if (!out.has(key)) out.set(key, { where: rec.__where, kind: f.kind, detail: f.detail, scene, w: W, h: H });
  }
}

/* ---------- driving a chapter through its states ---------- */
function runFrames(n, record, out, scene) {
  for (let i = 0; i < n; i++) {
    /* No startFrame here: the recorder is always on and delimits frames on a
       full-canvas clear, so a figure that draws on a control's onChange instead
       of inside ctx.loop is captured too. Thirteen of the course's ninety
       canvases never call ctx.loop and were silently unchecked before. */
    const q = rafQueue.splice(0);
    for (const fn of q) { try { fn(Date.now() + i * 16); } catch (e) { /* smoke.js owns runtime errors */ } }
    if (record && i === n - 1) { recorders.forEach(r => { r.endFrame(); if (r.ops.length) checkFrame(r, out, scene); }); }
  }
}
const ev = (type, extra) => Object.assign({ type, preventDefault() {}, stopPropagation() {}, clientX: 100, clientY: 100, touches: null, pointerId: 1, button: 0, key: 'a', deltaY: 1 }, extra || {});

load(path.join(ROOT, 'app/js/core.js'));
const chapterDir = path.join(ROOT, 'app/chapters');
const files = fs.readdirSync(chapterDir).filter(f => f.endsWith('.js')).sort();
const only = process.argv.slice(2);
let total = 0, chaptersWithIssues = 0;

for (const f of files) {
  const id = f.replace(/\.js$/, '');
  if (only.length && !only.some(o => id.includes(o))) continue;
  recorders.length = 0; rafQueue.length = 0; timers.length = 0;
  const out = new Map();
  const origError = console.error; console.error = () => {};
  let ctx;
  try {
    load(path.join(chapterDir, f));
    const ch = window.ZTA.chapters.find(c => c.id === id);
    const root = new Element('div');
    ctx = window.ZTA.makeCtx();
    ch.render(root, ctx);
    runFrames(WARM, true, out, 'initial');
    for (const fn of timers.splice(0)) { try { fn(); } catch (e) {} }

    const ranges = [], buttons = [], selects = [];
    walk(root, el => {
      if (el.tagName === 'INPUT' && el.attributes.type === 'range') ranges.push(el);
      else if (el.tagName === 'BUTTON') buttons.push(el);
      else if (el.tagName === 'SELECT') selects.push(el);
    });

    /* every slider swept together, then each slider swept on its own: the first
       finds states the reader reaches by dragging, the second stops one stuck
       slider from hiding everything else. */
    const sweep = [0, 0.25, 0.5, 0.75, 1];
    /* A browser snaps a range input to its step, so states between steps are
       unreachable for a reader and must not be tested — a degree-3.5 polynomial
       is a bug in the harness, not in the chapter. */
    const setR = (el, frac) => {
      const lo = +el.attributes.min, hi = +el.attributes.max;
      const step = +el.attributes.step || 1;
      const raw = lo + frac * (hi - lo);
      const snapped = Math.min(hi, Math.max(lo, lo + Math.round((raw - lo) / step) * step));
      el.value = String(+snapped.toFixed(10));
      el.dispatchEvent(ev('input')); el.dispatchEvent(ev('change'));
    };
    for (const frac of sweep) {
      ranges.forEach(r => setR(r, frac));
      runFrames(WARM, true, out, 'all sliders at ' + Math.round(frac * 100) + '%');
    }
    for (const r of ranges.slice(0, 40)) {
      for (const frac of [0, 1]) { setR(r, frac); runFrames(3, true, out, 'one slider at ' + Math.round(frac * 100) + '%'); }
      setR(r, 0.5);
    }
    for (const s of selects.slice(0, 30)) {
      for (const o of s.options) {
        s.value = o.attributes.value || o.textContent;
        s.dispatchEvent(ev('change'));
        runFrames(3, true, out, 'select ' + JSON.stringify(String(s.value).slice(0, 20)));
      }
    }
    for (const b of buttons.slice(0, 90)) {
      b.dispatchEvent(ev('click'));
      runFrames(3, true, out, 'after button ' + JSON.stringify(String(b.textContent).slice(0, 24)));
    }
    ctx._cleanup();
  } catch (e) {
    console.log('SKIP ' + id + ' — ' + String(e.stack || e).split('\n').slice(0,6).join(' | '));
    try { ctx && ctx._cleanup(); } catch (_) {}
    console.error = origError;
    continue;
  } finally { console.error = origError; rafQueue.length = 0; timers.length = 0; }

  const issues = [...out.values()];
  total += issues.length;
  if (!issues.length) { console.log('ok   ' + id); continue; }
  chaptersWithIssues++;
  console.log('PAINT ' + id + ' — ' + issues.length + ' issue(s)');
  const byWhere = new Map();
  for (const i of issues) { if (!byWhere.has(i.where)) byWhere.set(i.where, []); byWhere.get(i.where).push(i); }
  for (const [where, list] of byWhere) {
    console.log('  ' + where + '  (' + list.length + ')');
    for (const i of list.slice(0, 12)) console.log('      [' + i.kind + '] ' + i.detail + '   — ' + i.scene);
    if (list.length > 12) console.log('      … and ' + (list.length - 12) + ' more');
  }
}
console.log(total ? '\n' + total + ' rendering issue(s) in ' + chaptersWithIssues + ' chapter(s)' : '\nno rendering issues');
process.exit(total ? 1 : 0);
