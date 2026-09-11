/* A canvas 2D context that draws nothing and remembers everything.

   smoke.js proves a chapter's code does not throw. It cannot prove the picture
   is right, because its fake context swallows every call. This one keeps the
   full transform stack, the clip stack and a real text-metric model, so after a
   frame we can ask geometric questions: did a label land on top of another
   label, did a line cut through the truth table, did anything fall off the
   canvas. Those are the failures a reader sees and a thrown-error test never
   will. */

/* Advance widths as a fraction of the font size. Approximate Inter/system-ui;
   JetBrains Mono is a true 0.6em monospace. Within ~8% of the browser, which is
   plenty for collision detection given the tolerances below. */
function charW(c, mono) {
  if (mono) return 0.6;
  if (c === ' ') return 0.26;
  if ('iIl|.,:;\'`!'.indexOf(c) >= 0) return 0.28;
  if ('[]()/\\{}'.indexOf(c) >= 0) return 0.33;
  if ('fjtr-'.indexOf(c) >= 0) return 0.37;
  if ('mw'.indexOf(c) >= 0) return 0.84;
  if ('MW'.indexOf(c) >= 0) return 0.86;
  if (c >= 'a' && c <= 'z') return 0.54;
  if (c >= '0' && c <= '9') return 0.57;
  if (c >= 'A' && c <= 'Z') return 0.66;
  if (c.charCodeAt(0) > 0x2000) return 0.9;   /* arrows, box drawing, emoji-ish */
  return 0.55;
}

function parseFont(f) {
  const s = String(f || '13px sans-serif');
  const m = s.match(/(\d+(?:\.\d+)?)px/);
  return { size: m ? +m[1] : 13, mono: /mono/i.test(s), bold: /bold|[6-9]00/.test(s) };
}

/* Effective opacity of a paint: globalAlpha times any alpha carried by the
   colour itself. A chapter that fades something out with rgba(...,0) or the
   literal 'transparent' is not drawing it, and must not be reported. */
function styleAlpha(style) {
  if (style == null) return 1;
  if (typeof style !== 'string') return 1;             /* gradient / pattern */
  const s = style.trim();
  if (s === 'transparent' || s === 'none') return 0;
  const m = s.match(/rgba?\(([^)]*)\)/i);
  if (m) { const p = m[1].split(','); return p.length > 3 ? Math.max(0, Math.min(1, parseFloat(p[3]) || 0)) : 1; }
  const h = s.match(/^#([0-9a-f]{8})$/i);
  if (h) return parseInt(h[1].slice(6), 16) / 255;
  return 1;
}

const I = () => ({ a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
const apply = (m, x, y) => ({ x: m.a * x + m.c * y + m.e, y: m.b * x + m.d * y + m.f });
const scaleOf = (m) => Math.sqrt(Math.abs(m.a * m.d - m.b * m.c)) || 1;

function bboxOf(pts) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of pts) { if (p.x < x0) x0 = p.x; if (p.y < y0) y0 = p.y; if (p.x > x1) x1 = p.x; if (p.y > y1) y1 = p.y; }
  return { x0, y0, x1, y1 };
}
const rectsOverlap = (a, b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
function intersect(a, b) {
  return { x0: Math.max(a.x0, b.x0), y0: Math.max(a.y0, b.y0), x1: Math.min(a.x1, b.x1), y1: Math.min(a.y1, b.y1) };
}

const NOOP = () => {};
const PASSTHROUGH = new Set(['fillStyle', 'strokeStyle', 'lineWidth', 'globalAlpha', 'font', 'textAlign', 'textBaseline',
  'lineCap', 'lineJoin', 'miterLimit', 'lineDashOffset', 'shadowBlur', 'shadowColor', 'shadowOffsetX', 'shadowOffsetY',
  'globalCompositeOperation', 'filter', 'imageSmoothingEnabled', 'imageSmoothingQuality', 'direction', 'letterSpacing']);

class Recorder {
  constructor(canvas) {
    this.__canvas = canvas;
    this.recording = false;
    this.ops = [];          /* visible paint operations in this frame, in order */
    this.seq = 0;
    this.reset();
  }
  reset() {
    this.m = I();
    this.clipRect = null;   /* null = whole canvas */
    this.stack = [];
    this.fillStyle = '#000'; this.strokeStyle = '#000'; this.lineWidth = 1; this.globalAlpha = 1;
    this.font = '13px sans-serif'; this.textAlign = 'start'; this.textBaseline = 'alphabetic';
    this.path = []; this.cur = null; this.pathPts = [];
  }
  get W() { return this.__canvas.width || 720; }
  get H() { return this.__canvas.height || 400; }

  /* ---- state ---- */
  save() {
    this.stack.push({ m: Object.assign({}, this.m), clipRect: this.clipRect, fillStyle: this.fillStyle, strokeStyle: this.strokeStyle,
      lineWidth: this.lineWidth, globalAlpha: this.globalAlpha, font: this.font, textAlign: this.textAlign, textBaseline: this.textBaseline });
  }
  restore() { const s = this.stack.pop(); if (s) Object.assign(this, s); }
  translate(tx, ty) { const m = this.m; m.e += m.a * tx + m.c * ty; m.f += m.b * tx + m.d * ty; }
  scale(sx, sy) { const m = this.m; m.a *= sx; m.b *= sx; m.c *= sy; m.d *= sy; }
  rotate(r) {
    const m = this.m, cs = Math.cos(r), sn = Math.sin(r);
    const a = m.a * cs + m.c * sn, b = m.b * cs + m.d * sn;
    const c = m.a * -sn + m.c * cs, d = m.b * -sn + m.d * cs;
    m.a = a; m.b = b; m.c = c; m.d = d;
  }
  transform(a, b, c, d, e, f) {
    const m = this.m;
    const na = m.a * a + m.c * b, nb = m.b * a + m.d * b;
    const nc = m.a * c + m.c * d, nd = m.b * c + m.d * d;
    const ne = m.a * e + m.c * f + m.e, nf = m.b * e + m.d * f + m.f;
    m.a = na; m.b = nb; m.c = nc; m.d = nd; m.e = ne; m.f = nf;
  }
  setTransform(a, b, c, d, e, f) { this.m = { a: a, b: b, c: c, d: d, e: e, f: f }; }
  resetTransform() { this.m = I(); }

  /* ---- paths ---- */
  beginPath() { this.path = []; this.cur = null; this.pathPts = []; }
  closePath() { if (this.cur && this.cur.length > 1) this.cur.push(this.cur[0]); }
  moveTo(x, y) { const p = apply(this.m, x, y); this.cur = [p]; this.path.push(this.cur); this.pathPts.push(p); }
  lineTo(x, y) { const p = apply(this.m, x, y); if (!this.cur) { this.cur = [p]; this.path.push(this.cur); } else this.cur.push(p); this.pathPts.push(p); }
  _curveTo() { const a = arguments; this.lineTo(a[a.length - 2], a[a.length - 1]); }
  quadraticCurveTo(cx, cy, x, y) { this.lineTo(x, y); }
  bezierCurveTo(a, b, c, d, x, y) { this.lineTo(x, y); }
  arcTo(a, b, x, y) { this.lineTo(x, y); }
  rect(x, y, w, h) {
    const pts = [apply(this.m, x, y), apply(this.m, x + w, y), apply(this.m, x + w, y + h), apply(this.m, x, y + h)];
    this.cur = pts.concat([pts[0]]); this.path.push(this.cur); this.pathPts.push(...pts);
    this._lastRect = bboxOf(pts);
  }
  roundRect(x, y, w, h) { this.rect(x, y, w, h); }
  arc(x, y, r, s, e) {
    const c = apply(this.m, x, y), rr = r * scaleOf(this.m);
    this.pathPts.push({ x: c.x - rr, y: c.y - rr }, { x: c.x + rr, y: c.y + rr });
    (this._arcs = this._arcs || []).push({ x: c.x, y: c.y, r: rr });
    this.cur = null;
  }
  ellipse(x, y, rx, ry) { this.arc(x, y, Math.max(rx, ry), 0, 7); }
  clip() {
    const b = this.pathPts.length ? bboxOf(this.pathPts) : null;
    if (b) this.clipRect = this.clipRect ? intersect(this.clipRect, b) : b;
  }

  /* ---- text ---- */
  measureText(s) {
    const f = parseFont(this.font);
    let w = 0; const str = String(s);
    for (let i = 0; i < str.length; i++) w += charW(str[i], f.mono) * f.size;
    if (f.bold) w *= 1.04;
    return { width: w, actualBoundingBoxAscent: f.size * 0.72, actualBoundingBoxDescent: f.size * 0.22 };
  }
  _textBox(s, x, y) {
    const f = parseFont(this.font), w = this.measureText(s).width;
    let x0 = x;
    if (this.textAlign === 'center') x0 = x - w / 2;
    else if (this.textAlign === 'right' || this.textAlign === 'end') x0 = x - w;
    let yTop = y - f.size * 0.74;
    if (this.textBaseline === 'top' || this.textBaseline === 'hanging') yTop = y;
    else if (this.textBaseline === 'middle') yTop = y - f.size * 0.5;
    else if (this.textBaseline === 'bottom') yTop = y - f.size;
    const h = f.size * 0.96;
    const corners = [apply(this.m, x0, yTop), apply(this.m, x0 + w, yTop), apply(this.m, x0 + w, yTop + h), apply(this.m, x0, yTop + h)];
    return bboxOf(corners);
  }
  fillText(s, x, y) { this._text(s, x, y, styleAlpha(this.fillStyle)); }
  strokeText(s, x, y) { this._text(s, x, y, styleAlpha(this.strokeStyle)); }
  _text(s, x, y, sa) {
    const str = String(s);
    if (!this.recording || !str.trim()) return;
    const alpha = this.globalAlpha * sa;
    const box = this._textBox(str, x, y);
    this.ops.push({ kind: 'text', str: str, box: box, alpha: alpha, clip: this.clipRect, seq: this.seq++,
      size: parseFont(this.font).size, rotated: Math.abs(this.m.b) > 0.01 || Math.abs(this.m.c) > 0.01 });
  }

  /* ---- painting ---- */
  fillRect(x, y, w, h) {
    if (!this.recording || !w || !h) return;
    const b = bboxOf([apply(this.m, x, y), apply(this.m, x + w, y), apply(this.m, x + w, y + h), apply(this.m, x, y + h)]);
    this.ops.push({ kind: 'rect', box: b, alpha: this.globalAlpha * styleAlpha(this.fillStyle), clip: this.clipRect, seq: this.seq++ });
  }
  strokeRect(x, y, w, h) {
    if (!this.recording) return;
    const p = [apply(this.m, x, y), apply(this.m, x + w, y), apply(this.m, x + w, y + h), apply(this.m, x, y + h)];
    const alpha = this.globalAlpha * styleAlpha(this.strokeStyle);
    this._segments([p.concat([p[0]])], alpha, this.lineWidth);
    /* A large stroked rectangle is nearly always a plot frame or a panel
       border. Remembering it lets the checker ask whether a chart's own
       drawing stayed inside its box. */
    this.ops.push({ kind: 'frame', box: bboxOf(p), alpha: alpha, clip: this.clipRect, seq: this.seq++ });
  }
  clearRect() {}
  stroke() {
    if (!this.recording) return;
    this._segments(this.path, this.globalAlpha * styleAlpha(this.strokeStyle), this.lineWidth);
  }
  fill() {
    if (!this.recording) { this._arcs = null; return; }
    const alpha = this.globalAlpha * styleAlpha(this.fillStyle);
    for (const a of this._arcs || []) this.ops.push({ kind: 'dot', box: { x0: a.x - a.r, y0: a.y - a.r, x1: a.x + a.r, y1: a.y + a.r }, alpha: alpha, clip: this.clipRect, seq: this.seq++ });
    if (this.pathPts.length && !(this._arcs || []).length) this.ops.push({ kind: 'shape', box: bboxOf(this.pathPts), alpha: alpha, clip: this.clipRect, seq: this.seq++ });
    this._arcs = null;
  }
  _segments(subpaths, alpha, lw) {
    for (const sp of subpaths) {
      for (let i = 1; i < sp.length; i++) {
        const a = sp[i - 1], b = sp[i];
        if (!isFinite(a.x) || !isFinite(a.y) || !isFinite(b.x) || !isFinite(b.y)) continue;
        this.ops.push({ kind: 'seg', a: a, b: b, box: bboxOf([a, b]), alpha: alpha, lw: lw, clip: this.clipRect, seq: this.seq++ });
      }
    }
  }

  /* ---- images and stubs ---- */
  drawImage() {} putImageData() {}
  getImageData(x, y, w, h) {
    const ww = Math.max(1, Math.abs(w | 0)), hh = Math.max(1, Math.abs(h | 0));
    return { data: new Uint8ClampedArray(ww * hh * 4), width: ww, height: hh };
  }
  createImageData(w, h) {
    const ww = Math.max(1, Math.abs(Math.round((h != null ? w : (w && w.width)) || 1)));
    const hh = Math.max(1, Math.abs(Math.round((h != null ? h : (w && w.height)) || 1)));
    return { data: new Uint8ClampedArray(ww * hh * 4), width: ww, height: hh };
  }
  createLinearGradient() { return { addColorStop: NOOP }; }
  createRadialGradient() { return { addColorStop: NOOP }; }
  createConicGradient() { return { addColorStop: NOOP }; }
  createPattern() { return {}; }
  setLineDash() {} getLineDash() { return []; }
  isPointInPath() { return false; }
  get canvas() { return this.__canvas; }

  /* ---- frame control ---- */
  startFrame() { this.recording = true; this.ops = []; this.seq = 0; }
  endFrame() { this.recording = false; return this.ops; }
}

/* Unknown members must not explode: chapters use a wide slice of the API. */
function makeRecorder(canvas) {
  const r = new Recorder(canvas);
  return new Proxy(r, {
    get(t, k) {
      if (k in t) { const v = t[k]; return typeof v === 'function' ? v.bind(t) : v; }
      return NOOP;
    },
    set(t, k, v) { t[k] = v; return true; },
  });
}

module.exports = { makeRecorder, Recorder, bboxOf, rectsOverlap, intersect, parseFont };
