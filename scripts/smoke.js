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

/* ---------- fake DOM ---------- */
function makeCanvasCtx() {
  const noop = () => {};
  const state = {};
  return new Proxy(state, {
    get(t, k) {
      if (k === 'measureText') return () => ({ width: 10 });
      if (k === 'getImageData') return (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h });
      if (k === 'createImageData') return (w, h) => ({ data: new Uint8ClampedArray((h ? w : w.width) * (h || w.height) * 4), width: h ? w : w.width, height: h || w.height });
      if (k === 'createLinearGradient' || k === 'createRadialGradient') return () => ({ addColorStop: noop });
      if (k === 'createPattern') return () => ({});
      if (k === 'canvas') return t.__canvas;
      if (k in t) return t[k];
      return noop;
    },
    set(t, k, v) { t[k] = v; return true; },
  });
}

class ClassList {
  constructor(el) { this.el = el; this.set = new Set(); }
  add(...c) { c.forEach(x => this.set.add(x)); }
  remove(...c) { c.forEach(x => this.set.delete(x)); }
  toggle(c, force) { const on = force == null ? !this.set.has(c) : !!force; on ? this.set.add(c) : this.set.delete(c); return on; }
  contains(c) { return this.set.has(c); }
}

class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.children = []; this.childNodes = this.children; this.parentNode = null;
    this.style = {}; this.dataset = {}; this.attributes = {}; this.listeners = {}; this.classList = new ClassList(this);
    this._text = ''; this.value = ''; this.checked = false; this.disabled = false; this.selected = false;
    this.width = 300; this.height = 150; this.scrollTop = 0; this.scrollHeight = 0; this.offsetWidth = 720; this.clientWidth = 720; this.offsetHeight = 360; this.clientHeight = 360;
  }
  get className() { return [...this.classList.set].join(' '); }
  set className(v) { this.classList.set = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get innerHTML() { return this._text; }
  set innerHTML(v) { this.children.length = 0; this._text = String(v); }
  get textContent() { return this._text + this.children.map(c => c.textContent || '').join(''); }
  set textContent(v) { this.children.length = 0; this._text = String(v); }
  get innerText() { return this.textContent; } set innerText(v) { this.textContent = v; }
  get firstChild() { return this.children[0] || null; }
  get lastChild() { return this.children[this.children.length - 1] || null; }
  get isConnected() { return true; }
  append(...nodes) { for (const n of nodes.flat(Infinity)) { if (n == null) continue; if (n instanceof Element) { n.parentNode = this; this.children.push(n); } else this.children.push(new TextNode(String(n))); } }
  appendChild(n) { this.append(n); return n; }
  prepend(...nodes) { const saved = this.children.splice(0); this.append(...nodes); this.children.push(...saved); }
  removeChild(n) { const i = this.children.indexOf(n); if (i >= 0) this.children.splice(i, 1); return n; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  replaceChildren(...n) { this.children.length = 0; this.append(...n); }
  insertBefore(n, ref) { const i = this.children.indexOf(ref); if (i < 0) this.append(n); else { n.parentNode = this; this.children.splice(i, 0, n); } return n; }
  setAttribute(k, v) { this.attributes[k] = String(v); if (k === 'value') this.value = String(v); if (k === 'width') this.width = +v; if (k === 'height') this.height = +v; if (k === 'id') this.id = v; }
  getAttribute(k) { return this.attributes[k] == null ? null : this.attributes[k]; }
  removeAttribute(k) { delete this.attributes[k]; }
  hasAttribute(k) { return k in this.attributes; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  removeEventListener(type, fn) { const l = this.listeners[type]; if (l) { const i = l.indexOf(fn); if (i >= 0) l.splice(i, 1); } }
  dispatchEvent(ev) { ev.target = ev.target || this; ev.currentTarget = this; (this.listeners[ev.type] || []).forEach(fn => fn.call(this, ev)); const h = this['on' + ev.type]; if (typeof h === 'function') h.call(this, ev); return true; }
  getContext() { if (!this._ctx) { this._ctx = makeCanvasCtx(); this._ctx.__canvas = this; } return this._ctx; }
  getBoundingClientRect() { return { left: 0, top: 0, width: this.W || 720, height: this.H || 360, right: this.W || 720, bottom: this.H || 360 }; }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  querySelectorAll(sel) {
    // supports comma lists and combinators loosely: only the LAST simple selector of each part is matched (descendant search)
    const out = [];
    for (const part of String(sel).split(',')) {
      const last = part.trim().split(/\s*[>+~]\s*|\s+/).filter(Boolean).pop() || '';
      const m = last.match(/^([a-z0-9]+)?((?:\.[\w-]+)*)(?:\[[^\]]*\])?(?::[\w-]+(?:\([^)]*\))?)*$/i);
      if (!m) continue;
      const tag = m[1], classes = (m[2] || '').split('.').filter(Boolean);
      walk(this, el => { if ((!tag || el.tagName === tag.toUpperCase()) && classes.every(c => el.classList.contains(c)) && !out.includes(el)) out.push(el); });
    }
    return out;
  }
  focus() {} blur() {} click() { this.dispatchEvent({ type: 'click' }); } select() {} scrollIntoView() {} setPointerCapture() {} releasePointerCapture() {}
  getElementsByTagName(t) { return this.querySelectorAll(t); }
  toDataURL() { return ''; }
  get options() { return this.querySelectorAll('option'); }
  get selectedIndex() { return 0; }
}
class TextNode { constructor(t) { this.textContent = t; this.nodeType = 3; } }
function walk(el, fn) { for (const c of el.children) { if (c instanceof Element) { fn(c); walk(c, fn); } } }

const rafQueue = [];
const timers = [];
const window = {
  devicePixelRatio: 1, innerWidth: 1200, innerHeight: 800,
  addEventListener() {}, removeEventListener() {},
  requestAnimationFrame(fn) { rafQueue.push(fn); return rafQueue.length; },
  cancelAnimationFrame() {},
  setTimeout(fn, ms) { timers.push(fn); return timers.length; }, clearTimeout() {},
  setInterval(fn) { timers.push(fn); return timers.length; }, clearInterval() {},
  localStorage: { _d: {}, getItem(k) { return k in this._d ? this._d[k] : null; }, setItem(k, v) { this._d[k] = String(v); }, removeItem(k) { delete this._d[k]; } },
  performance: { now: () => Date.now() },
  location: { hash: '' },
  getComputedStyle: () => ({ getPropertyValue: () => '' }),
  scrollTo() {},
  matchMedia: () => ({ matches: false, addEventListener() {} }),
};
const document = {
  createElement: (t) => new Element(t),
  createElementNS: (ns, t) => new Element(t),
  createTextNode: (t) => new TextNode(t),
  createDocumentFragment: () => new Element('fragment'),
  body: new Element('body'), documentElement: new Element('html'),
  addEventListener() {}, removeEventListener() {},
  getElementById: () => null, querySelector: () => null, querySelectorAll: () => [],
  activeElement: null, hidden: false,
};
window.document = document; window.window = window; window.Node = Element; window.HTMLElement = Element; window.Element = Element;
window.Event = class { constructor(type, init) { this.type = type; Object.assign(this, init || {}); } };
window.CustomEvent = window.Event; window.PointerEvent = window.Event; window.MouseEvent = window.Event; window.KeyboardEvent = window.Event;
window.navigator = { userAgent: 'smoke', maxTouchPoints: 0, clipboard: { writeText: async () => {} } };
window.console = console; window.Math = Math; window.JSON = JSON; window.Date = Date;
window.Float32Array = Float32Array; window.Float64Array = Float64Array; window.Uint8Array = Uint8Array; window.Uint8ClampedArray = Uint8ClampedArray; window.Int32Array = Int32Array; window.Uint32Array = Uint32Array; window.Int8Array = Int8Array; window.Int16Array = Int16Array; window.Uint16Array = Uint16Array;
window.Map = Map; window.Set = Set; window.Promise = Promise; window.Object = Object; window.Array = Array; window.Number = Number; window.String = String; window.Boolean = Boolean; window.Symbol = Symbol; window.Error = Error; window.TypeError = TypeError; window.RangeError = RangeError; window.RegExp = RegExp; window.Infinity = Infinity; window.NaN = NaN; window.isNaN = isNaN; window.isFinite = isFinite; window.parseInt = parseInt; window.parseFloat = parseFloat; window.encodeURIComponent = encodeURIComponent; window.decodeURIComponent = decodeURIComponent; window.structuredClone = (x) => JSON.parse(JSON.stringify(x));
window.TextEncoder = TextEncoder; window.TextDecoder = TextDecoder; window.queueMicrotask = queueMicrotask; window.Intl = Intl;
window.requestIdleCallback = (fn) => { timers.push(fn); return 1; };
window.alert = () => { throw new Error('alert() is not allowed in chapters'); };
window.fetch = () => { throw new Error('fetch() is not allowed in chapters'); };

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
        if (el.tagName === 'INPUT' && el.attributes.type === 'range') { const mid = (+el.attributes.min + +el.attributes.max) / 2; el.value = String(mid); el.dispatchEvent(ev('input')); el.dispatchEvent(ev('change')); poked++; }
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
    const problems = [];
    if (errors.length) problems.push(errors.length + ' runtime error(s): ' + errors[0].split('\n').slice(0, 3).join(' | '));
    if (!quiz) problems.push('no quiz');
    if (!canvases) problems.push('no canvas');
    if (words < 900) problems.push('thin prose (' + words + ' words rendered)');
    if (problems.length) { failures++; console.log('FAIL ' + id + ': ' + problems.join('; ')); }
    else console.log('ok   ' + id + ' — ' + words + ' words, ' + canvases + ' canvases, ' + callouts + ' callouts, ' + poked + ' controls poked');
  } catch (e) {
    failures++;
    console.log('FAIL ' + id + ': ' + String(e.stack || e).split('\n').slice(0, 4).join(' | '));
  } finally { console.error = origError; }
}
console.log(failures ? failures + ' chapter(s) failed' : 'all chapters passed');
process.exit(failures ? 1 : 0);
