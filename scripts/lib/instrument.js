/* In-page instrumentation for the browser rendering check.

   Wraps CanvasRenderingContext2D so every paint is recorded with the geometry
   the browser will actually use — real font metrics, the real transform stack —
   and nothing is approximated. Injected before the page's own scripts run. */
module.exports = function instrumentSource() {
  return `(function () {
  var P = CanvasRenderingContext2D.prototype;
  var O = {};
  ['save','restore','beginPath','closePath','moveTo','lineTo','rect','roundRect','arc','ellipse',
   'quadraticCurveTo','bezierCurveTo','arcTo','clip','stroke','fill','fillRect','strokeRect',
   'clearRect','fillText','strokeText'].forEach(function (k) { if (P[k]) O[k] = P[k]; });

  function S(ctx) {
    if (!ctx.__zta) ctx.__zta = { ops: [], last: [], seq: 0, clip: null, stack: [], pts: [], subs: [], cur: null, arcs: [] };
    return ctx.__zta;
  }
  function T(ctx, x, y) { var m = ctx.getTransform(); return { x: m.a * x + m.c * y + m.e, y: m.b * x + m.d * y + m.f }; }
  function bbox(pts) {
    var x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (var i = 0; i < pts.length; i++) { var p = pts[i];
      if (p.x < x0) x0 = p.x; if (p.y < y0) y0 = p.y; if (p.x > x1) x1 = p.x; if (p.y > y1) y1 = p.y; }
    return { x0: x0, y0: y0, x1: x1, y1: y1 };
  }
  function alphaOf(style, ga) {
    if (typeof style !== 'string') return ga;
    var s = style.trim();
    if (s === 'transparent' || s === 'none') return 0;
    var m = s.match(/rgba?\\(([^)]*)\\)/i);
    if (m) { var p = m[1].split(','); if (p.length > 3) return ga * Math.max(0, Math.min(1, parseFloat(p[3]) || 0)); return ga; }
    var h8 = s.match(/^#([0-9a-fA-F]{8})$/);
    if (h8) return ga * parseInt(h8[1].slice(6), 16) / 255;
    return ga;
  }
  function fontSize(f) { var m = String(f || '13px').match(/(\\d+(?:\\.\\d+)?)px/); return m ? +m[1] : 13; }
  function rec(ctx, op) { var s = S(ctx); if (!window.__ZTA_RECORD) return; op.seq = s.seq++; op.clip = s.clip; s.ops.push(op); }

  P.save = function () { var s = S(this); s.stack.push(s.clip); return O.save.apply(this, arguments); };
  P.restore = function () { var s = S(this); if (s.stack.length) s.clip = s.stack.pop(); return O.restore.apply(this, arguments); };
  P.beginPath = function () { var s = S(this); s.pts = []; s.subs = []; s.cur = null; s.arcs = []; return O.beginPath.apply(this, arguments); };
  P.moveTo = function (x, y) { var s = S(this), p = T(this, x, y); s.cur = [p]; s.subs.push(s.cur); s.pts.push(p); return O.moveTo.apply(this, arguments); };
  P.lineTo = function (x, y) { var s = S(this), p = T(this, x, y); if (!s.cur) { s.cur = [p]; s.subs.push(s.cur); } else s.cur.push(p); s.pts.push(p); return O.lineTo.apply(this, arguments); };
  P.closePath = function () { var s = S(this); if (s.cur && s.cur.length > 1) s.cur.push(s.cur[0]); return O.closePath.apply(this, arguments); };
  P.quadraticCurveTo = function (a, b, x, y) { P.lineTo.call(this, x, y); return O.quadraticCurveTo.apply(this, arguments); };
  P.bezierCurveTo = function (a, b, c, d, x, y) { P.lineTo.call(this, x, y); return O.bezierCurveTo.apply(this, arguments); };
  P.arcTo = function (a, b, x, y) { P.lineTo.call(this, x, y); return O.arcTo.apply(this, arguments); };
  P.rect = function (x, y, w, h) {
    var s = S(this), p = [T(this, x, y), T(this, x + w, y), T(this, x + w, y + h), T(this, x, y + h)];
    s.cur = p.concat([p[0]]); s.subs.push(s.cur); s.pts.push.apply(s.pts, p);
    return O.rect.apply(this, arguments);
  };
  if (O.roundRect) P.roundRect = function (x, y, w, h) { P.rect.call(this, x, y, w, h); return O.roundRect.apply(this, arguments); };
  P.arc = function (x, y, r) {
    var s = S(this), c = T(this, x, y), m = this.getTransform();
    var sc = Math.sqrt(Math.abs(m.a * m.d - m.b * m.c)) || 1, rr = r * sc;
    s.pts.push({ x: c.x - rr, y: c.y - rr }, { x: c.x + rr, y: c.y + rr });
    s.arcs.push({ x: c.x, y: c.y, r: rr }); s.cur = null;
    return O.arc.apply(this, arguments);
  };
  P.ellipse = function (x, y, rx, ry) { var s = S(this); P.arc.call(this, x, y, Math.max(rx, ry), 0, 7); s.arcs.pop(); var c = T(this, x, y); s.arcs.push({ x: c.x, y: c.y, r: Math.max(rx, ry) }); return O.ellipse.apply(this, arguments); };
  P.clip = function () {
    var s = S(this);
    if (s.pts.length) { var b = bbox(s.pts); s.clip = s.clip ? { x0: Math.max(s.clip.x0, b.x0), y0: Math.max(s.clip.y0, b.y0), x1: Math.min(s.clip.x1, b.x1), y1: Math.min(s.clip.y1, b.y1) } : b; }
    return O.clip.apply(this, arguments);
  };
  function segs(ctx, subs, alpha, lw) {
    for (var i = 0; i < subs.length; i++) { var sp = subs[i];
      for (var j = 1; j < sp.length; j++) {
        var a = sp[j - 1], b = sp[j];
        if (!isFinite(a.x) || !isFinite(a.y) || !isFinite(b.x) || !isFinite(b.y)) continue;
        rec(ctx, { kind: 'seg', a: a, b: b, box: bbox([a, b]), alpha: alpha, lw: lw });
      }
    }
  }
  P.stroke = function () { var s = S(this); segs(this, s.subs, alphaOf(this.strokeStyle, this.globalAlpha), this.lineWidth); return O.stroke.apply(this, arguments); };
  P.fill = function () {
    var s = S(this), al = alphaOf(this.fillStyle, this.globalAlpha);
    for (var i = 0; i < s.arcs.length; i++) { var a = s.arcs[i]; rec(this, { kind: 'dot', box: { x0: a.x - a.r, y0: a.y - a.r, x1: a.x + a.r, y1: a.y + a.r }, alpha: al }); }
    if (s.pts.length && !s.arcs.length) rec(this, { kind: 'shape', box: bbox(s.pts), alpha: al });
    return O.fill.apply(this, arguments);
  };
  P.fillRect = function (x, y, w, h) {
    if (w && h) rec(this, { kind: 'rect', box: bbox([T(this, x, y), T(this, x + w, y), T(this, x + w, y + h), T(this, x, y + h)]), alpha: alphaOf(this.fillStyle, this.globalAlpha) });
    return O.fillRect.apply(this, arguments);
  };
  P.strokeRect = function (x, y, w, h) {
    var p = [T(this, x, y), T(this, x + w, y), T(this, x + w, y + h), T(this, x, y + h)];
    var al = alphaOf(this.strokeStyle, this.globalAlpha);
    segs(this, [p.concat([p[0]])], al, this.lineWidth);
    rec(this, { kind: 'frame', box: bbox(p), alpha: al });
    return O.strokeRect.apply(this, arguments);
  };
  P.clearRect = function (x, y, w, h) {
    var s = S(this), cv = this.canvas;
    /* every chapter starts a frame by clearing the whole canvas */
    if (x <= 1 && y <= 1 && w >= cv.width * 0.9 / (cv.__dpr || 1) && h >= cv.height * 0.9 / (cv.__dpr || 1)) {
      if (s.ops.length) s.last = s.ops;
      s.ops = []; s.seq = 0;
    }
    return O.clearRect.apply(this, arguments);
  };
  function text(ctx, str, x, y, al) {
    str = String(str);
    if (!window.__ZTA_RECORD || !str.trim()) return;
    var w = ctx.measureText(str).width, size = fontSize(ctx.font);
    var x0 = x;
    if (ctx.textAlign === 'center') x0 = x - w / 2;
    else if (ctx.textAlign === 'right' || ctx.textAlign === 'end') x0 = x - w;
    var yTop = y - size * 0.74;
    if (ctx.textBaseline === 'top' || ctx.textBaseline === 'hanging') yTop = y;
    else if (ctx.textBaseline === 'middle') yTop = y - size * 0.5;
    else if (ctx.textBaseline === 'bottom') yTop = y - size;
    var hgt = size * 0.96;
    var c = [T(ctx, x0, yTop), T(ctx, x0 + w, yTop), T(ctx, x0 + w, yTop + hgt), T(ctx, x0, yTop + hgt)];
    var m = ctx.getTransform();
    rec(ctx, { kind: 'text', str: str, box: bbox(c), alpha: al, size: size, rotated: Math.abs(m.b) > 0.01 || Math.abs(m.c) > 0.01 });
  }
  P.fillText = function (s, x, y) { text(this, s, x, y, alphaOf(this.fillStyle, this.globalAlpha)); return O.fillText.apply(this, arguments); };
  P.strokeText = function (s, x, y) { text(this, s, x, y, alphaOf(this.strokeStyle, this.globalAlpha)); return O.strokeText.apply(this, arguments); };

  window.__ZTA_RECORD = false;
  window.__ZTA_frames = function () {
    var out = [];
    document.querySelectorAll('canvas').forEach(function (cv, i) {
      var ctx = cv.__ztaCtx || (cv.getContext ? null : null);
      var s = cv.__zta_ctx_state;
      if (!s) return;
      var ops = s.ops.length ? s.ops : s.last;
      if (!ops || !ops.length) return;
      out.push({ i: i, w: cv.W || cv.width, h: cv.H || cv.height, ops: ops });
    });
    return out;
  };
  /* remember the state object on the canvas so __ZTA_frames can find it */
  var gc = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function () {
    var c = gc.apply(this, arguments);
    if (c && c.canvas === this && typeof c.fillText === 'function') { this.__zta_ctx_state = S(c); }
    return c;
  };
})();`;
};
