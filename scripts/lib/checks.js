/* The rendering checks, over a recorded list of paint operations.

   Shared by the fast headless checker (paint.js, modelled text metrics) and the
   browser checker (render-check.js, real Chromium metrics) so both report the
   same things and a fix silences both. */

const VISIBLE = 0.3;        /* below this alpha a thing is fading, not drawn */
const LINE_VISIBLE = 0.5;   /* faint enough to be a gridline is fine under text */
const LINE_MIN_W = 1.4;     /* hairlines and rules are fine under text */
const OVERLAP_FRAC = 0.35;  /* share of the smaller label that must be covered */
const TEXT_INSET_X = 0.22;  /* glyphs have side bearings; boxes overstate them */
const TEXT_INSET_Y = 0.16;
const OFFCANVAS_PAINT = 24; /* px outside the canvas before a shape is a bug */
const OFFCANVAS_TEXT = 4;   /* px of a label allowed past the edge */
const FRAME_PAD = 18;       /* px a plot may spill past its own frame */
const MAX_PER_KIND = 25;    /* a collapsed scatter would otherwise report n² times */

function intersect(a, b) {
  return { x0: Math.max(a.x0, b.x0), y0: Math.max(a.y0, b.y0), x1: Math.min(a.x1, b.x1), y1: Math.min(a.y1, b.y1) };
}
function area(b) { return Math.max(0, b.x1 - b.x0) * Math.max(0, b.y1 - b.y0); }
function inset(b, fx, fy) {
  const dx = (b.x1 - b.x0) * fx, dy = (b.y1 - b.y0) * fy;
  return { x0: b.x0 + dx, y0: b.y0 + dy, x1: b.x1 - dx, y1: b.y1 - dy };
}
function overlapArea(a, b) { const i = intersect(a, b); return (i.x1 <= i.x0 || i.y1 <= i.y0) ? 0 : area(i); }
/* Liang–Barsky */
function segHitsBox(a, b, r) {
  let t0 = 0, t1 = 1;
  const dx = b.x - a.x, dy = b.y - a.y;
  const p = [-dx, dx, -dy, dy];
  const q = [a.x - r.x0, r.x1 - a.x, a.y - r.y0, r.y1 - a.y];
  for (let i = 0; i < 4; i++) {
    if (p[i] === 0) { if (q[i] < 0) return false; continue; }
    const t = q[i] / p[i];
    if (p[i] < 0) { if (t > t1) return false; if (t > t0) t0 = t; }
    else { if (t < t0) return false; if (t < t1) t1 = t; }
  }
  return t1 > t0;
}
function visibleBox(op) {
  if (!op.clip) return op.box;
  const b = intersect(op.box, op.clip);
  return (b.x1 <= b.x0 || b.y1 <= b.y0) ? null : b;
}
const r0 = Math.round;

/* ops: the ordered paint operations of ONE frame on ONE canvas.
   Returns [{kind, detail}] — the reader-visible defects in that frame. */
function checkFrame(ops, W, H) {
  const found = [];
  const counts = {};
  const add = (kind, detail) => {
    counts[kind] = (counts[kind] || 0) + 1;
    if (counts[kind] <= MAX_PER_KIND) found.push({ kind, detail });
  };

  const texts = [], frames = [], blocks = [];
  for (const op of ops) {
    const b = visibleBox(op);
    if (!b) continue;
    op.vis = b;
    if (op.kind === 'text' && op.alpha >= VISIBLE) texts.push({ op, b, backedBy: -1 });
    if (op.kind === 'frame' && area(b) > 9000) frames.push(b);
    /* an opaque block hides what is under it: labels on badges and bars are
       normal, and without this every one of them would be reported */
    if ((op.kind === 'rect' || op.kind === 'shape') && op.alpha >= 0.85) blocks.push({ b, seq: op.seq });
    /* A solid marker is a backing too: a letter printed in the middle of its own
       dot is the normal way to label a point, and the line arriving at that dot
       is behind the marker, not across the letter. Inscribe a box in the circle
       so only text genuinely inside the disc counts as covered. */
    if (op.kind === 'dot' && op.alpha >= 0.85) {
      const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2, r = Math.min(b.x1 - b.x0, b.y1 - b.y0) / 2;
      const k = r * 0.72;
      blocks.push({ b: { x0: cx - k, y0: cy - k, x1: cx + k, y1: cy + k }, seq: op.seq });
    }
    /* a dot or a block off the canvas is a coordinate error. Long lines are
       excluded on purpose: drawing a boundary right across and letting the
       canvas edge trim it is a normal idiom, and where such a line leaves its
       own plot the frame check below is the one that fires. */
    if (op.alpha >= VISIBLE && !op.clip && (op.kind === 'dot' || op.kind === 'rect')) {
      const outBy = Math.max(-b.x0, -b.y0, b.x1 - W, b.y1 - H);
      if (outBy > OFFCANVAS_PAINT && area(b) > 4) {
        add('offcanvas-paint', op.kind + ' reaches ' + r0(outBy) + 'px outside the ' + W + '×' + H
          + ' canvas [' + [b.x0, b.y0, b.x1, b.y1].map(r0).join(',') + ']');
      }
    }
  }

  for (const t of texts) {
    const core = inset(t.b, 0.1, 0.1);
    for (const bl of blocks) {
      if (bl.seq > t.op.seq) continue;
      if (bl.b.x0 <= core.x0 + 1 && bl.b.y0 <= core.y0 + 1 && bl.b.x1 >= core.x1 - 1 && bl.b.y1 >= core.y1 - 1) {
        if (bl.seq > t.backedBy) t.backedBy = bl.seq;
      }
    }
  }

  /* labels running off the edge */
  for (const t of texts) {
    const outBy = Math.max(-t.b.x0, -t.b.y0, t.b.x1 - W, t.b.y1 - H);
    if (outBy > OFFCANVAS_TEXT) {
      add('offcanvas-text', JSON.stringify(t.op.str.slice(0, 46)) + ' runs ' + r0(outBy) + 'px off the canvas');
    }
  }

  /* labels printed on labels */
  for (let i = 0; i < texts.length; i++) {
    for (let j = i + 1; j < texts.length; j++) {
      const A = texts[i], B = texts[j];
      if (A.op.str === B.op.str && Math.abs(A.b.x0 - B.b.x0) < 1 && Math.abs(A.b.y0 - B.b.y0) < 1) continue;
      if (B.backedBy > A.op.seq || A.backedBy > B.op.seq) continue;
      const a = inset(A.b, TEXT_INSET_X, TEXT_INSET_Y), b = inset(B.b, TEXT_INSET_X, TEXT_INSET_Y);
      const ov = overlapArea(a, b);
      if (!ov) continue;
      const frac = ov / Math.min(area(a), area(b));
      const i2 = intersect(a, b);
      /* Two labels side by side on the same line collide visibly long before
         either is 35% covered, so a narrow but full-height intrusion counts
         too — that is the commonest collision of the lot. */
      const sameLine = (i2.y1 - i2.y0) > 0.45 * Math.min(a.y1 - a.y0, b.y1 - b.y0);
      if (frac < OVERLAP_FRAC && !(sameLine && (i2.x1 - i2.x0) > 3)) continue;
      add('text-overlap', JSON.stringify(A.op.str.slice(0, 34)) + ' and ' + JSON.stringify(B.op.str.slice(0, 34))
        + (frac >= OVERLAP_FRAC ? ' overlap by ' + r0(100 * frac) + '%' : ' collide by ' + r0(i2.x1 - i2.x0) + 'px on the same line')
        + ' near (' + r0(a.x0) + ',' + r0(a.y0) + ')');
    }
  }

  /* a plot drawing outside its own frame, into whatever is next to it */
  for (const F of frames) {
    const innerF = inset(F, 0.02, 0.02);
    for (const op of ops) {
      if ((op.kind !== 'seg' && op.kind !== 'dot') || op.alpha < VISIBLE || op.clip || !op.vis) continue;
      const belongs = op.kind === 'seg'
        ? segHitsBox(op.a, op.b, innerF)
        : ((op.vis.x0 + op.vis.x1) / 2 >= innerF.x0 && (op.vis.x0 + op.vis.x1) / 2 <= innerF.x1
          && (op.vis.y0 + op.vis.y1) / 2 >= innerF.y0 && (op.vis.y0 + op.vis.y1) / 2 <= innerF.y1);
      if (!belongs) continue;
      const outBy = Math.max(F.x0 - op.vis.x0, F.y0 - op.vis.y0, op.vis.x1 - F.x1, op.vis.y1 - F.y1);
      if (outBy > FRAME_PAD) {
        add('outside-frame', op.kind + ' starting inside the plot box [' + [F.x0, F.y0, F.x1, F.y1].map(r0).join(',')
          + '] reaches ' + r0(outBy) + 'px beyond it — clip the plot or clamp the coordinates');
      }
    }
  }

  /* lines drawn through labels */
  for (const op of ops) {
    if (op.kind !== 'seg' || op.alpha < LINE_VISIBLE || (op.lw || 1) < LINE_MIN_W || !op.vis) continue;
    if (Math.hypot(op.b.x - op.a.x, op.b.y - op.a.y) < 8) continue;
    for (const t of texts) {
      if (t.backedBy > op.seq) continue;
      const box = inset(t.b, 0.16, 0.30);
      if (area(box) < 12) continue;
      if (op.clip) { const c = intersect(box, op.clip); if (c.x1 <= c.x0 || c.y1 <= c.y0) continue; }
      if (segHitsBox(op.a, op.b, box)) {
        add('line-over-text', 'a ' + (+(op.lw || 1).toFixed(1)) + 'px line crosses the label '
          + JSON.stringify(t.op.str.slice(0, 40)) + ' at (' + r0(box.x0) + ',' + r0(box.y0) + ')');
        break;
      }
    }
  }

  for (const k of Object.keys(counts)) {
    if (counts[k] > MAX_PER_KIND) found.push({ kind: k, detail: '… and ' + (counts[k] - MAX_PER_KIND) + ' more ' + k + ' in this same frame' });
  }
  return found;
}

module.exports = { checkFrame, intersect, area, inset, segHitsBox, overlapArea };
