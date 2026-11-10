/* Zero → AGI · Chapter 16 · Learning without neural networks
   DESIGN RULE: the reader cuts a scatter of card payments into boxes with their own hands before
   the words "decision tree" appear, and every later method is met the same way: drive it, watch a
   number react, then get the name.
   Interactives, in order: hand-grown tree (the opener); information-gain explorer on that same
   tree's first cut (the maths beat hangs off it); depth dial that overfits a tree; random forest
   with a tree-count slider and a two-forest disagreement score; gradient boosting on residuals,
   one stump at a time; k-nearest-neighbours with a draggable query; k-means where the reader
   drops the centres (with a broken start); straight line vs sigmoid with an outlier student.
   Every dataset comes from a seeded generator, so the numbers the prose quotes are reproducible. */
(function () {
  ZTA.registerChapter({
    id: '16-classical-ml',
    num: 16,
    part: 'VI',
    title: 'Learning without neural networks',
    tagline: 'Scissors, crowds, neighbours and centres of gravity: the machine learning that runs most of industry, with barely a neuron in it.',
    minutes: 55,
    objectives: [
      'grow a decision tree by hand and compute the entropy and information gain of a cut from its counts',
      'diagnose an overfitted tree, and explain how a random forest and gradient boosting each repair it',
      'predict what k-nearest-neighbours and k-means will do as k and the starting centres change',
      'say when a straight line is the wrong shape for a yes/no question and what logistic regression changes',
    ],
    takeaways: [
      '<b>A decision tree is a stack of yes/no questions.</b> Each cut is one question about one column, and every final box votes with its majority.',
      '<b>Cuts are chosen by information gain.</b> Entropy is the average surprise in a box, in bits; the machine takes the cut that lowers the size-weighted entropy most. Gini impurity is the same idea without the logarithm.',
      '<b>One deep tree memorises.</b> It scores 100% on its own data and far less on fresh data, and it redraws itself whenever the sample changes.',
      '<b>Forests average, boosting corrects.</b> A random forest grows many deep trees on resampled data and lets them vote. Gradient boosting adds small trees one at a time, each fitted to what is still wrong.',
      '<b>k-NN and k-means run on distance alone.</b> k-NN lets the k closest labelled examples vote. k-means needs no labels, and where it ends up depends on where its centres start.',
      '<b>Logistic regression is chapter 1\'s line pushed through chapter 11\'s sigmoid.</b> The output stays between 0 and 1, and an extreme example it already gets right cannot drag the boundary.',
    ],
    render(root, ctx) {
      const { h, p, section, callout, ul } = ctx;
      const C = ctx.colors;
      const FONT = '13px Inter, system-ui, sans-serif';
      const BOLD = '600 13px Inter, system-ui, sans-serif';
      const SMALL = '12px Inter, system-ui, sans-serif';
      const RED = C.danger, BLUE = C.accent;
      const pct = (v) => (v * 100).toFixed(1) + '%';

      /*CORE-START*/
      /* Seeded randomness: every dataset below is identical on every load, so the numbers the
         prose quotes are the numbers the reader sees. */
      function mulberry32(a) {
        return function () {
          a |= 0; a = a + 0x6D2B79F5 | 0;
          let t = Math.imul(a ^ a >>> 15, 1 | a);
          t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
          return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
      }
      function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }

      /* The fraud world. x = payment amount (£), y = miles from the cardholder's home, both
         0–100 and both landing on k + 0.5 so that a whole-number cut never sits on a point.
         Fraud lives at BOTH ends of the amount scale, as in chapter 1: tiny card-testing
         payments anywhere, and large payments far from home. NOISE of the labels break the rule. */
      const CUT_A = 20, CUT_B = 64, CUT_M = 42, NOISE = 0.1;
      const isFraudZone = (x, y) => x < CUT_A || (x > CUT_B && y > CUT_M);
      function makePayments(seed, n) {
        const r = mulberry32(seed), pts = [];
        for (let i = 0; i < n; i++) {
          const x = Math.floor(r() * 100) + 0.5, y = Math.floor(r() * 100) + 0.5;
          const flip = r() < NOISE;
          pts.push({ x, y, c: (isFraudZone(x, y) ? 1 : 0) ^ (flip ? 1 : 0) });
        }
        return pts;
      }
      const entropy = (a, b) => { const n = a + b; let e = 0; if (a && n) e -= a / n * Math.log2(a / n); if (b && n) e -= b / n * Math.log2(b / n); return e; };
      const gini = (a, b) => { const n = a + b; if (!n) return 0; const p = a / n; return 1 - p * p - (1 - p) * (1 - p); };
      const feat = (pt, f) => f === 0 ? pt.x : pt.y;
      /* best whole-number cut of the points `idx` on feature f: lowest weighted impurity */
      function bestCutOn(pts, idx, f, imp) {
        const s = idx.slice().sort((i, j) => feat(pts[i], f) - feat(pts[j], f));
        let tot1 = 0; for (const i of s) tot1 += pts[i].c;
        const n = s.length; let l1 = 0, best = null;
        for (let k = 0; k < n - 1; k++) {
          l1 += pts[s[k]].c;
          const a = feat(pts[s[k]], f), b = feat(pts[s[k + 1]], f);
          if (a === b) continue;
          const nl = k + 1, nr = n - nl;
          const w = (nl * imp(l1, nl - l1) + nr * imp(tot1 - l1, nr - (tot1 - l1))) / n;
          if (!best || w < best.w - 1e-12) best = { f, t: a + 0.5, w };
        }
        return best;
      }
      /* grow a tree greedily. opts: maxDepth, imp, rng (if given, each split may look at only
         one randomly chosen feature: the random-forest rule) */
      function grow(pts, idx, opts, depth) {
        depth = depth || 0;
        let n1 = 0; for (const i of idx) n1 += pts[i].c;
        const n0 = idx.length - n1;
        const node = { n1, n0, pred: n1 > n0 ? 1 : 0, depth };
        if (!n1 || !n0 || depth >= opts.maxDepth) return node;
        const imp = opts.imp || gini;
        let cands;
        if (opts.rng) { const f = opts.rng() < 0.5 ? 0 : 1; cands = [bestCutOn(pts, idx, f, imp) || bestCutOn(pts, idx, 1 - f, imp)]; }
        else cands = [bestCutOn(pts, idx, 0, imp), bestCutOn(pts, idx, 1, imp)];
        let best = null; for (const c of cands) if (c && (!best || c.w < best.w - 1e-12)) best = c;
        if (!best) return node;
        const L = [], R = [];
        for (const i of idx) (feat(pts[i], best.f) < best.t ? L : R).push(i);
        node.f = best.f; node.t = best.t;
        node.L = grow(pts, L, opts, depth + 1); node.R = grow(pts, R, opts, depth + 1);
        return node;
      }
      function predictTree(node, x, y) { while (node.L) node = (node.f === 0 ? x : y) < node.t ? node.L : node.R; return node.pred; }
      function countLeaves(node) { return node.L ? countLeaves(node.L) + countLeaves(node.R) : 1; }
      function treeDepth(node) { return node.L ? Math.max(treeDepth(node.L), treeDepth(node.R)) : node.depth; }
      function accTree(node, pts) { let k = 0; for (const q of pts) if (predictTree(node, q.x, q.y) === q.c) k++; return k / pts.length; }
      const range = (n) => Array.from({ length: n }, (_, i) => i);
      /* k nearest neighbours: indices of the k training points closest to (x, y) */
      function nearest(pts, x, y, k, skip) {
        const d = [];
        for (let i = 0; i < pts.length; i++) { if (i === skip) continue; const dx = pts[i].x - x, dy = pts[i].y - y; d.push([dx * dx + dy * dy, i]); }
        d.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        return d.slice(0, k).map(e => e[1]);
      }
      function knnVote(pts, x, y, k) { let v = 0; for (const i of nearest(pts, x, y, k)) v += pts[i].c; return v; }

      /* Boosting world: bike hires by hour of day. Two commuter peaks on a quiet night. */
      const bikeTruth = (hr) => 12 + 70 * Math.exp(-((hr - 8.5) ** 2) / 4.5) + 85 * Math.exp(-((hr - 17.5) ** 2) / 8) + 25 * Math.exp(-((hr - 13) ** 2) / 10);
      function makeBikes(seed, n, sd) {
        const r = mulberry32(seed), out = [];
        for (let i = 0; i < n; i++) { const x = Math.floor(r() * 96) / 4 + 0.125; out.push({ x, y: Math.max(0, bikeTruth(x) + gauss(r) * sd) }); }
        return out.sort((a, b) => a.x - b.x);
      }
      /* the one-cut tree ("stump") that best fits the residuals res[] at sorted positions */
      function bestStump(data, res) {
        const n = data.length; let tot = 0; for (const v of res) tot += v;
        let ls = 0, best = null;
        for (let k = 0; k < n - 1; k++) {
          ls += res[k];
          if (data[k].x === data[k + 1].x) continue;
          const nl = k + 1, nr = n - nl, score = ls * ls / nl + (tot - ls) * (tot - ls) / nr;
          if (!best || score > best.score + 1e-9) best = { score, t: (data[k].x + data[k + 1].x) / 2, lv: ls / nl, rv: (tot - ls) / nr };
        }
        return best;
      }
      const stumpOut = (s, x) => x < s.t ? s.lv : s.rv;

      /* k-means: one step = give every point to its nearest centre, then move each centre to
         the mean of its points. Returns how far the centres moved. */
      function kmAssign(pts, cs) { return pts.map(q => { let b = 0, bd = Infinity; cs.forEach((c, j) => { const d = (q.x - c.x) ** 2 + (q.y - c.y) ** 2; if (d < bd) { bd = d; b = j; } }); return b; }); }
      function kmStep(pts, cs) {
        const as = kmAssign(pts, cs); let moved = 0;
        cs.forEach((c, j) => {
          let sx = 0, sy = 0, n = 0; pts.forEach((q, i) => { if (as[i] === j) { sx += q.x; sy += q.y; n++; } });
          if (!n) return;
          moved += Math.hypot(sx / n - c.x, sy / n - c.y); c.x = sx / n; c.y = sy / n;
        });
        return moved;
      }
      function kmSpread(pts, cs) { if (!cs.length) return NaN; const as = kmAssign(pts, cs); let s = 0; pts.forEach((q, i) => { s += (q.x - cs[as[i]].x) ** 2 + (q.y - cs[as[i]].y) ** 2; }); return Math.sqrt(s / pts.length); }
      function makeBlobs(seed) {
        const r = mulberry32(seed), pts = [];
        const B = [[22, 30, 9, 40], [70, 72, 7, 30], [78, 26, 7, 30]];
        B.forEach(([cx, cy, sd, n], b) => { for (let i = 0; i < n; i++) pts.push({ x: cx + gauss(r) * sd, y: cy + gauss(r) * sd, b }); });
        return pts;
      }

      /* straight line by least squares, and logistic regression by Newton steps */
      function fitLine(xs, ys) {
        const n = xs.length; let sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
        const w = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        return { w, b: (sy - w * sx) / n };
      }
      const sigmoid = (z) => 1 / (1 + Math.exp(-z));
      function fitLogistic(xs, ys) {
        let w = 0, b = 0;
        for (let it = 0; it < 40; it++) {
          let g0 = 0, g1 = 0, h00 = 1e-6, h01 = 0, h11 = 1e-6;
          for (let i = 0; i < xs.length; i++) { const p = sigmoid(w * xs[i] + b), e = p - ys[i], s = p * (1 - p); g0 += e; g1 += e * xs[i]; h00 += s; h01 += s * xs[i]; h11 += s * xs[i] * xs[i]; }
          const det = h00 * h11 - h01 * h01; if (Math.abs(det) < 1e-12) break;
          b -= (h11 * g0 - h01 * g1) / det; w -= (h00 * g1 - h01 * g0) / det;
        }
        return { w, b };
      }
      function makeStudents(seed, n) {
        const r = mulberry32(seed), out = [];
        for (let i = 0; i < n; i++) { const x = Math.round((0.5 + r() * 9) * 2) / 2; out.push({ x, c: r() < sigmoid(1.3 * (x - 5)) ? 1 : 0 }); }
        return out.sort((a, b) => a.x - b.x);
      }
      /*CORE-END*/

      /* ---------- shared data: one fraud world, sampled several times ---------- */
      const HAND = makePayments(166, 40);                 /* the 40 payments the reader cuts by hand */
      /* Seeds chosen so that no sample holds two payments at the same spot with opposite labels and
         every tree is pure by depth 12; otherwise "memorise everything" would stop short of 100%. */
      const TRAIN_SEEDS = [205, 208, 201, 213, 209];
      const TRAIN = makePayments(TRAIN_SEEDS[0], 200);               /* the 200 the machine learns from */
      const FRESH = makePayments(9001, 1000);             /* 1,000 it never sees */

      /* ---------- shared drawing ---------- */
      /* data runs 0–100 on both axes; the plot is inset by PAD so a dot on the limit is not sliced
         by the clip. RX/RY are for box edges, which should reach the frame itself. UX/UY invert. */
      const PAD = 7;
      const PX = (pl, x) => pl.x + PAD + x / 100 * (pl.w - 2 * PAD);
      const PY = (pl, y) => pl.y + pl.h - PAD - y / 100 * (pl.h - 2 * PAD);
      const RX = (pl, x) => x <= 0 ? pl.x : x >= 100 ? pl.x + pl.w : PX(pl, x);
      const RY = (pl, y) => y <= 0 ? pl.y + pl.h : y >= 100 ? pl.y : PY(pl, y);
      const half = (v) => Math.round(v * 2) / 2;
      /* pointer position → data coordinates, or null when outside the frame */
      function toData(cv, pl, ev) {
        const q = cv.pos(ev);
        if (q.x < pl.x || q.x > pl.x + pl.w || q.y < pl.y || q.y > pl.y + pl.h) return null;
        return { x: ctx.clamp((q.x - pl.x - PAD) / (pl.w - 2 * PAD) * 100, 0, 100), y: ctx.clamp((pl.y + pl.h - PAD - q.y) / (pl.h - 2 * PAD) * 100, 0, 100) };
      }
      function frame(g, pl) { g.fillStyle = '#0f1520'; g.fillRect(pl.x, pl.y, pl.w, pl.h); g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(pl.x, pl.y, pl.w, pl.h); }
      function clipTo(g, pl, fn) { g.save(); g.beginPath(); g.rect(pl.x, pl.y, pl.w, pl.h); g.clip(); fn(); g.restore(); }
      /* axes and key for the payments scatter */
      function worldAxes(g, pl) {
        g.font = SMALL; g.fillStyle = C.muted; g.textAlign = 'center';
        for (let v = 0; v <= 100; v += 20) g.fillText(String(v), PX(pl, v), pl.y + pl.h + 15);
        g.fillText('payment amount, £', pl.x + pl.w / 2, pl.y + pl.h + 32);
        g.textAlign = 'right';
        for (let v = 20; v <= 100; v += 20) g.fillText(String(v), pl.x - 6, PY(pl, v) + 4);
        g.textAlign = 'left'; g.fillText('miles from home ↑', pl.x, pl.y - 9);
        g.textAlign = 'right';
        g.fillStyle = BLUE; g.fillText('● genuine', pl.x + pl.w, pl.y - 9);
        g.fillStyle = RED; g.fillText('● fraud', pl.x + pl.w - 76, pl.y - 9);
      }
      function drawPayments(g, pl, pts, r) {
        for (const q of pts) { g.beginPath(); g.arc(PX(pl, q.x), PY(pl, q.y), r, 0, Math.PI * 2); g.fillStyle = q.c ? RED : BLUE; g.fill(); g.lineWidth = 1; g.strokeStyle = '#0a0e16'; g.stroke(); }
      }
      const TINT = ['rgba(124,156,255,0.20)', 'rgba(251,113,133,0.24)'];
      /* fill the boxes of a tree, each tinted by its verdict */
      function paintTree(g, pl, node, x0, x1, y0, y1) {
        if (node.L) {
          if (node.f === 0) { paintTree(g, pl, node.L, x0, node.t, y0, y1); paintTree(g, pl, node.R, node.t, x1, y0, y1); }
          else { paintTree(g, pl, node.L, x0, x1, y0, node.t); paintTree(g, pl, node.R, x0, x1, node.t, y1); }
          return;
        }
        g.fillStyle = TINT[node.pred];
        g.fillRect(RX(pl, x0), RY(pl, y1), RX(pl, x1) - RX(pl, x0), RY(pl, y0) - RY(pl, y1));
      }
      /* n×n cells over the 0–100 square, runs of one colour merged into one rectangle */
      function paintCells(g, pl, n, colourAt) {
        /* edges snapped to half a logical pixel (a whole device pixel), so neighbouring
           translucent cells neither overlap nor leave a seam */
        const ex = (c) => half(RX(pl, c * 100 / n)), ey = (r) => half(RY(pl, r * 100 / n));
        for (let r = 0; r < n; r++) {
          let start = 0, cur = colourAt(0, r);
          for (let c = 1; c <= n; c++) {
            const col = c < n ? colourAt(c, r) : null;
            if (col === cur) continue;
            if (cur) { g.fillStyle = cur; g.fillRect(ex(start), ey(r + 1), ex(c) - ex(start), ey(r) - ey(r + 1)); }
            start = c; cur = col;
          }
        }
      }
      /* a small line chart: series = [{ys, color}], x positions spread evenly */
      function chart(g, pl, series, ymin, ymax, markAt, xs) {
        frame(g, pl);
        const n = series[0].ys.length;
        /* xs, if given, places each point at a fraction 0–1 of the width (for an uneven axis) */
        const xAt = (i) => pl.x + 6 + (xs ? xs[i] : n < 2 ? 0 : i / (n - 1)) * (pl.w - 12), yAt = (v) => pl.y + pl.h - 5 - (ctx.clamp(v, ymin, ymax) - ymin) / (ymax - ymin) * (pl.h - 10);
        clipTo(g, pl, () => {
          if (markAt != null) { g.strokeStyle = C.warn; g.setLineDash([4, 3]); g.lineWidth = 1.2; g.beginPath(); g.moveTo(xAt(markAt), pl.y); g.lineTo(xAt(markAt), pl.y + pl.h); g.stroke(); g.setLineDash([]); }
          for (const s of series) {
            g.strokeStyle = s.color; g.lineWidth = 2; g.beginPath();
            s.ys.forEach((v, i) => { if (i) g.lineTo(xAt(i), yAt(v)); else g.moveTo(xAt(i), yAt(v)); }); g.stroke();
            if (markAt != null && s.ys[markAt] != null) { g.beginPath(); g.arc(xAt(markAt), yAt(s.ys[markAt]), 3.5, 0, Math.PI * 2); g.fillStyle = s.color; g.fill(); }
          }
        });
        return { xAt, yAt };
      }

      /* ================================================================== */
      /* 1. Grow a tree by hand                                              */
      /* ================================================================== */
      function handTree() {
        const W = 720, H = 400, MAXCUTS = 6;
        const [cv, g] = ctx.canvas(W, H);
        cv.style.touchAction = 'none';
        const pl = { x: 44, y: 30, w: 420, h: 320 };
        const pts = HAND;
        const mk = (x0, x1, y0, y1, idx) => ({ x0, x1, y0, y1, idx });
        let rootN, hist, orient, cur, dirty = true;
        /* A click or tap HOLDS the yellow cut where it is. Without that, a mouse on its way down
           to the Cut button drags a horizontal cut to the bottom of the box as it leaves. */
        let held = false, down = false;
        const counts = (idx) => { let a = 0; for (const i of idx) a += pts[i].c; return [a, idx.length - a]; };
        const wrong = (idx) => { const [a, b] = counts(idx); return Math.min(a, b); };
        const leavesOf = (n, out) => { if (n.L) { leavesOf(n.L, out); leavesOf(n.R, out); } else out.push(n); return out; };
        const halves = (node, f, t) => { const L = [], R = []; for (const i of node.idx) (feat(pts[i], f) < t ? L : R).push(i); return [L, R]; };
        const totalWrong = () => leavesOf(rootN, []).reduce((s, n) => s + wrong(n.idx), 0);
        function reset() { rootN = mk(0, 100, 0, 100, range(pts.length)); hist = []; orient = 0; cur = { node: rootN, t: 50 }; held = false; dirty = true; }
        function cut(node, f, t) {
          const [L, R] = halves(node, f, t);
          node.f = f; node.t = t;
          node.L = f === 0 ? mk(node.x0, t, node.y0, node.y1, L) : mk(node.x0, node.x1, node.y0, t, L);
          node.R = f === 0 ? mk(t, node.x1, node.y0, node.y1, R) : mk(node.x0, node.x1, t, node.y1, R);
          hist.push(node);
        }
        function previewWrong() {
          if (!cur) return null;
          const [L, R] = halves(cur.node, orient, cur.t);
          return totalWrong() - wrong(cur.node.idx) + wrong(L) + wrong(R);
        }
        function aim(ev) {
          const d = toData(cv, pl, ev); if (!d) return false;
          const x = d.x, y = d.y;
          const node = leavesOf(rootN, []).find(n => x >= n.x0 && x <= n.x1 && y >= n.y0 && y <= n.y1);
          if (!node) return false;
          const lo = orient === 0 ? node.x0 : node.y0, hi = orient === 0 ? node.x1 : node.y1;
          if (hi - lo < 2) return false;
          cur = { node, t: ctx.clamp(Math.round(orient === 0 ? x : y), lo + 1, hi - 1) }; dirty = true;
          return true;
        }
        /* mouse: hovering aims until a click holds the cut. Touch and pen: a tap or a drag aims. */
        cv.addEventListener('pointerdown', (ev) => { down = true; if (aim(ev)) held = true; });
        cv.addEventListener('pointermove', (ev) => { if (down || (!held && ev.pointerType === 'mouse')) aim(ev); });
        const release = () => { down = false; };
        cv.addEventListener('pointerup', release); cv.addEventListener('pointercancel', release); cv.addEventListener('pointerleave', release);
        function nudge(by) {
          if (!cur) return;
          const n = cur.node, lo = orient === 0 ? n.x0 : n.y0, hi = orient === 0 ? n.x1 : n.y1;
          cur.t = ctx.clamp(cur.t + by, lo + 1, hi - 1); held = true; dirty = true;
        }
        /* the machine's version: three cuts, each time the single cut with the most information gain */
        function machine() {
          reset();
          for (let k = 0; k < 3; k++) {
            let best = null;
            for (const n of leavesOf(rootN, [])) {
              const [a, b] = counts(n.idx);
              for (const f of [0, 1]) {
                const c = bestCutOn(pts, n.idx, f, entropy);
                if (!c) continue;
                const gain = n.idx.length / pts.length * (entropy(a, b) - c.w);
                if (!best || gain > best.gain + 1e-12) best = { node: n, f, t: c.t, gain };
              }
            }
            if (!best) break;
            cut(best.node, best.f, best.t);
          }
          cur = null; held = false; dirty = true;
        }
        reset();

        function draw() {
          g.clearRect(0, 0, W, H);
          frame(g, pl);
          const ls = leavesOf(rootN, []);
          clipTo(g, pl, () => {
            ls.forEach((n) => {
              const [a, b] = counts(n.idx);
              g.fillStyle = TINT[a > b ? 1 : 0];
              g.fillRect(RX(pl, n.x0), RY(pl, n.y1), RX(pl, n.x1) - RX(pl, n.x0), RY(pl, n.y0) - RY(pl, n.y1));
            });
            g.strokeStyle = 'rgba(230,235,245,0.75)'; g.lineWidth = 1.5;
            for (const n of hist) {
              g.beginPath();
              if (n.f === 0) { g.moveTo(PX(pl, n.t), RY(pl, n.y0)); g.lineTo(PX(pl, n.t), RY(pl, n.y1)); }
              else { g.moveTo(RX(pl, n.x0), PY(pl, n.t)); g.lineTo(RX(pl, n.x1), PY(pl, n.t)); }
              g.stroke();
            }
            if (cur) {
              const n = cur.node;
              g.strokeStyle = C.warn; g.lineWidth = 2; g.setLineDash(held ? [] : [6, 4]); g.beginPath();
              if (orient === 0) { g.moveTo(PX(pl, cur.t), RY(pl, n.y0)); g.lineTo(PX(pl, cur.t), RY(pl, n.y1)); }
              else { g.moveTo(RX(pl, n.x0), PY(pl, cur.t)); g.lineTo(RX(pl, n.x1), PY(pl, cur.t)); }
              g.stroke(); g.setLineDash([]);
            }
            drawPayments(g, pl, pts, 5);
            ls.forEach((n, k) => {
              g.font = BOLD; g.textAlign = 'left'; g.fillStyle = 'rgba(230,235,245,0.9)';
              g.fillText(String.fromCharCode(65 + k), RX(pl, n.x0) + 7, RY(pl, n.y1) + 17);
            });
          });
          worldAxes(g, pl);
          /* the tree, written as the questions it asks */
          const tx = 484; let ty = 40;
          g.textAlign = 'left'; g.font = BOLD; g.fillStyle = C.text; g.fillText('Your tree, as questions', tx, ty); ty += 24;
          let letter = 0;
          (function walk(n, depth, lead) {
            const ix = tx + depth * 11;
            if (n.L) {
              g.font = SMALL; g.fillStyle = C.text;
              g.fillText(lead + (n.f === 0 ? 'amount under £' + n.t + '?' : 'under ' + n.t + ' miles?'), ix, ty); ty += 21;
              walk(n.L, depth + 1, 'yes → '); walk(n.R, depth + 1, 'no → ');
            } else {
              const [a, b] = counts(n.idx), name = String.fromCharCode(65 + letter++);
              g.font = SMALL; g.fillStyle = a > b ? RED : BLUE;
              g.fillText(lead + name + ': ' + a + ' fraud, ' + b + ' genuine', ix, ty); ty += 21;
            }
          })(rootN, 0, '');
          g.font = SMALL; g.fillStyle = C.muted;
          if (cur && !held) { g.fillStyle = C.warn; g.fillText('Click or tap to hold the yellow cut.', tx, H - 66); g.fillStyle = C.muted; }
          g.fillText('Red box: it says fraud.', tx, H - 44);
          g.fillText('Blue box: it says genuine.', tx, H - 26);
        }
        const ro = ctx.readout();
        function update() {
          const pw = previewWrong();
          ro.set({
            'cuts made': hist.length + (hist.length >= MAXCUTS ? ' (the limit)' : ''),
            'mistakes now': totalWrong() + ' of ' + pts.length,
            'yellow cut': cur ? (orient === 0 ? 'amount at £' + cur.t : cur.t + ' miles') : 'tap or point at a box',
            'mistakes if you cut here': pw == null ? '–' : pw,
          });
        }
        ctx.loop(() => { if (dirty) { draw(); update(); dirty = false; } });
        const controls = [
          ctx.button('Cut here', () => { if (cur && hist.length < MAXCUTS) { cut(cur.node, orient, cur.t); cur = null; held = false; dirty = true; } }, 'primary'),
          ctx.button('Nudge −1', () => nudge(-1)),
          ctx.button('Nudge +1', () => nudge(1)),
          ctx.button('Turn the cut', () => {
            orient = 1 - orient;
            if (cur) { const n = cur.node, lo = orient === 0 ? n.x0 : n.y0, hi = orient === 0 ? n.x1 : n.y1; cur = hi - lo < 2 ? null : { node: n, t: Math.round((lo + hi) / 2) }; }
            dirty = true;
          }),
          ctx.button('Undo', () => { const n = hist.pop(); if (n) { delete n.L; delete n.R; delete n.f; delete n.t; cur = null; held = false; dirty = true; } }),
          ctx.button('Start again', reset),
          ctx.button('Show the machine\'s three cuts', machine),
        ];
        return ctx.figure(cv, 'Forty card payments from a made-up bank: 16 fraud, 24 genuine. (A different bank from chapter 1\'s: the amounts are smaller, and here distance from home carries a signal.) Point at a box to aim the yellow cut and click to hold it there (the dashes turn solid); on a phone, tap or drag. The <b>Nudge</b> buttons move it one step at a time. Then press <b>Cut here</b>. Every box votes with its majority, and a tied box says genuine. The panel on the right is the same picture written as questions, which is the form a program would run.', controls, ro);
      }

      /* ================================================================== */
      /* 2. How good is a cut? Counts, entropy, gain                         */
      /* ================================================================== */
      function gainExplorer() {
        const W = 720, H = 430;
        const [cv, g] = ctx.canvas(W, H);
        const pl = { x: 44, y: 30, w: 400, h: 230 }, strip = { x: 44, y: 322, w: 400, h: 70 };
        const pts = HAND, all = range(pts.length);
        const S = { f: 0, t: 50, m: 'entropy', dirty: true };
        const imp = () => S.m === 'entropy' ? entropy : gini;
        function sides(f, t) { let l1 = 0, l0 = 0, r1 = 0, r0 = 0; for (const q of pts) { if (feat(q, f) < t) { if (q.c) l1++; else l0++; } else if (q.c) r1++; else r0++; } return { l1, l0, r1, r0 }; }
        function stats(f, t) {
          const s = sides(f, t), I = imp(), nl = s.l1 + s.l0, nr = s.r1 + s.r0, n = nl + nr;
          const before = I(s.l1 + s.r1, s.l0 + s.r0), hl = I(s.l1, s.l0), hr = I(s.r1, s.r0), after = (nl * hl + nr * hr) / n;
          return Object.assign(s, { nl, nr, n, before, hl, hr, after, gain: before - after });
        }
        function bar(x, y, w, a, b) {
          const n = a + b; g.fillStyle = '#1a2233'; g.fillRect(x, y, w, 10);
          if (!n) return;
          g.fillStyle = RED; g.fillRect(x, y, w * a / n, 10); g.fillStyle = BLUE; g.fillRect(x + w * a / n, y, w * b / n, 10);
        }
        function draw() {
          g.clearRect(0, 0, W, H);
          frame(g, pl);
          const st = stats(S.f, S.t), unit = S.m === 'entropy' ? ' bits' : '', nm = S.m === 'entropy' ? 'entropy' : 'Gini';
          clipTo(g, pl, () => {
            g.fillStyle = 'rgba(251,191,36,0.07)';
            if (S.f === 0) g.fillRect(pl.x, pl.y, PX(pl, S.t) - pl.x, pl.h); else g.fillRect(pl.x, PY(pl, S.t), pl.w, pl.y + pl.h - PY(pl, S.t));
            g.strokeStyle = C.warn; g.lineWidth = 2; g.beginPath();
            if (S.f === 0) { g.moveTo(PX(pl, S.t), pl.y); g.lineTo(PX(pl, S.t), pl.y + pl.h); } else { g.moveTo(pl.x, PY(pl, S.t)); g.lineTo(pl.x + pl.w, PY(pl, S.t)); }
            g.stroke();
            drawPayments(g, pl, pts, 4.5);
          });
          worldAxes(g, pl);
          /* gain at every possible cut */
          const gains = []; for (let t = 1; t <= 99; t++) gains.push(stats(S.f, t).gain);
          const top = Math.max(0.3, ...gains) * 1.08;
          frame(g, strip);
          const sx = (t) => PX(strip, t), sy = (v) => strip.y + strip.h - 3 - v / top * (strip.h - 8);
          clipTo(g, strip, () => {
            g.strokeStyle = C.green; g.lineWidth = 2; g.beginPath();
            gains.forEach((v, i) => { if (i) g.lineTo(sx(i + 1), sy(v)); else g.moveTo(sx(1), sy(v)); }); g.stroke();
            g.strokeStyle = C.warn; g.lineWidth = 1.5; g.beginPath(); g.moveTo(sx(S.t), strip.y); g.lineTo(sx(S.t), strip.y + strip.h); g.stroke();
            g.beginPath(); g.arc(sx(S.t), sy(st.gain), 4, 0, Math.PI * 2); g.fillStyle = C.warn; g.fill();
          });
          g.font = SMALL; g.textAlign = 'left'; g.fillStyle = C.green;
          g.fillText('gain from a cut at each position along ' + (S.f === 0 ? 'amount' : 'miles'), strip.x, strip.y - 8);
          g.fillStyle = C.muted; g.textAlign = 'center';
          for (let v = 0; v <= 100; v += 20) g.fillText(String(v), sx(v), strip.y + strip.h + 15);
          /* the three boxes, as counts */
          const tx = 466, bw = 236; let y = 40;
          const lname = S.f === 0 ? 'Left: amount under £' + S.t : 'Below: under ' + S.t + ' miles';
          const rname = S.f === 0 ? 'Right: £' + S.t + ' and over' : 'Above: ' + S.t + ' miles and over';
          function block(title, a, b, v) {
            g.textAlign = 'left'; g.font = BOLD; g.fillStyle = C.text; g.fillText(title, tx, y);
            bar(tx, y + 8, bw, a, b);
            g.font = SMALL; g.fillStyle = C.muted; g.fillText(a + ' fraud, ' + b + ' genuine', tx, y + 34);
            g.textAlign = 'right'; g.fillStyle = C.text; g.fillText(nm + ' ' + v.toFixed(3) + unit, tx + bw, y + 34);
            y += 62;
          }
          block('Before any cut', st.l1 + st.r1, st.l0 + st.r0, st.before);
          block(lname, st.l1, st.l0, st.hl);
          block(rname, st.r1, st.r0, st.hr);
          g.textAlign = 'left'; g.font = SMALL; g.fillStyle = C.muted;
          g.fillText('after the cut, weighted by size:', tx, y + 4);
          g.fillStyle = C.text;
          g.fillText(st.nl + '/' + st.n + ' × ' + st.hl.toFixed(3) + ' + ' + st.nr + '/' + st.n + ' × ' + st.hr.toFixed(3) + ' = ' + st.after.toFixed(3), tx, y + 24);
          g.fillStyle = C.muted; g.fillText('gain = before − after:', tx, y + 52);
          g.font = BOLD; g.fillStyle = C.green;
          g.fillText(st.before.toFixed(3) + ' − ' + st.after.toFixed(3) + ' = ' + st.gain.toFixed(3) + unit, tx, y + 72);
          ro.set({ cut: S.f === 0 ? 'amount at £' + S.t : S.t + ' miles', before: st.before.toFixed(3), after: st.after.toFixed(3), gain: st.gain.toFixed(3) + unit });
        }
        const ro = ctx.readout();
        ctx.loop(() => { if (S.dirty) { draw(); S.dirty = false; } });
        const sl = ctx.slider({ label: 'cut at', min: 1, max: 99, step: 1, value: 50, onChange: (v) => { S.t = v; S.dirty = true; } });
        const controls = [
          sl,
          ctx.select({ label: 'cut on', value: '0', options: [{ value: '0', label: 'payment amount' }, { value: '1', label: 'miles from home' }], onChange: (v) => { S.f = +v; S.dirty = true; } }),
          ctx.select({ label: 'measure', value: 'entropy', options: [{ value: 'entropy', label: 'entropy, in bits' }, { value: 'gini', label: 'Gini impurity' }], onChange: (v) => { S.m = v; S.dirty = true; } }),
          ctx.button('Jump to the best cut', () => { let bt = 1, bg = -1; for (let t = 1; t <= 99; t++) { const v = stats(S.f, t).gain; if (v > bg + 1e-12) { bg = v; bt = t; } } S.t = bt; sl.value = bt; S.dirty = true; }, 'primary'),
        ];
        return ctx.figure(cv, 'The same forty payments, and only the first cut. The three bars are the pile before the cut and the two boxes after it, red for fraud and blue for genuine. The green curve underneath scores every possible position for the cut at once; the yellow marker is where you are. A tree-growing program computes that curve for every column and takes the highest point.', controls, ro);
      }

      /* ================================================================== */
      /* 3. A tree allowed to go as deep as it likes                         */
      /* ================================================================== */
      function deepTree() {
        const W = 720, H = 372, MAXD = 12;
        const [cv, g] = ctx.canvas(W, H);
        const pl = { x: 44, y: 30, w: 400, h: 290 }, cp = { x: 500, y: 52, w: 204, h: 210 };
        const S = { depth: 3, sample: 0, pts: TRAIN, trees: [], tr: [], fr: [], dirty: true };
        function build() {
          S.pts = S.sample === 0 ? TRAIN : makePayments(TRAIN_SEEDS[S.sample], 200);
          S.trees = []; S.tr = []; S.fr = [];
          for (let d = 1; d <= MAXD; d++) { const t = grow(S.pts, range(S.pts.length), { maxDepth: d }); S.trees.push(t); S.tr.push(accTree(t, S.pts) * 100); S.fr.push(accTree(t, FRESH) * 100); }
          S.dirty = true;
        }
        build();
        function draw() {
          g.clearRect(0, 0, W, H);
          frame(g, pl);
          const t = S.trees[S.depth - 1];
          clipTo(g, pl, () => { paintTree(g, pl, t, 0, 100, 0, 100); drawPayments(g, pl, S.pts, 3); });
          worldAxes(g, pl);
          chart(g, cp, [{ ys: S.tr, color: BLUE }, { ys: S.fr, color: C.orange }], 50, 100, S.depth - 1);
          g.font = BOLD; g.textAlign = 'left'; g.fillStyle = C.text; g.fillText('Score at each depth', cp.x, cp.y - 28);
          g.font = SMALL; g.fillStyle = C.muted; g.fillText('100%', cp.x, cp.y - 8);
          g.fillText('50%', cp.x, cp.y + cp.h + 15);
          g.textAlign = 'right'; g.fillText('depth 1 → ' + MAXD, cp.x + cp.w, cp.y + cp.h + 15);
          g.textAlign = 'left';
          g.fillStyle = BLUE; g.fillText('● on its own 200 payments', cp.x, cp.y + cp.h + 40);
          g.fillStyle = C.orange; g.fillText('● on 1,000 fresh payments', cp.x, cp.y + cp.h + 58);
          ro.set({ 'depth allowed': S.depth, boxes: countLeaves(t), 'on its own 200': pct(S.tr[S.depth - 1] / 100), 'on 1,000 fresh': pct(S.fr[S.depth - 1] / 100), sample: (S.sample + 1) + ' of ' + TRAIN_SEEDS.length });
        }
        const ro = ctx.readout();
        ctx.loop(() => { if (S.dirty) { draw(); S.dirty = false; } });
        const sl = ctx.slider({ label: 'depth allowed', min: 1, max: MAXD, step: 1, value: 3, onChange: (v) => { S.depth = v; S.dirty = true; } });
        return ctx.figure(cv, 'A program now grows the tree, on 200 payments from the same made-up bank, where one label in ten breaks the pattern. Depth is how many questions it may ask in a row. Each box is tinted by its verdict. The chart tracks two scores: the blue one is on the 200 payments it grew from, the orange one on 1,000 it has never seen. The true pattern itself scores only 89.7% on those fresh payments, because their labels are noisy too. Treat that as the ceiling.', [
          sl,
          ctx.button('No limit: memorise everything', () => { S.depth = MAXD; sl.value = MAXD; S.dirty = true; }),
          ctx.button('Back to depth 3', () => { S.depth = 3; sl.value = 3; S.dirty = true; }),
          ctx.button('New training sample', () => { S.sample = (S.sample + 1) % TRAIN_SEEDS.length; build(); }),
        ], ro);
      }

      /* ================================================================== */
      /* 4. Random forest                                                    */
      /* ================================================================== */
      function forestFig() {
        const W = 720, H = 372, MAXT = 99, N = 100;
        const [cv, g] = ctx.canvas(W, H);
        const pl = { x: 44, y: 30, w: 400, h: 290 }, c1 = { x: 500, y: 52, w: 204, h: 96 }, c2 = { x: 500, y: 212, w: 204, h: 96 };
        function growForest(pts, seed) {
          const r = mulberry32(seed), trees = [];
          for (let t = 0; t < MAXT; t++) { const idx = []; for (let i = 0; i < pts.length; i++) idx.push(Math.floor(r() * pts.length)); trees.push(grow(pts, idx, { maxDepth: 99, rng: r })); }
          return trees;
        }
        const FA = growForest(TRAIN, 7), FB = growForest(makePayments(206, 200), 8);
        /* running vote totals on the fresh payments, for both forests, and on a grid for the picture */
        const nF = FRESH.length;
        const cumA = new Uint8Array(MAXT * nF), cumB = new Uint8Array(MAXT * nF), grid = new Uint8Array(MAXT * N * N);
        const single = [], accAt = [], disAt = [];
        for (let t = 0; t < MAXT; t++) {
          let ok = 0;
          for (let j = 0; j < nF; j++) {
            const a = predictTree(FA[t], FRESH[j].x, FRESH[j].y), b = predictTree(FB[t], FRESH[j].x, FRESH[j].y);
            if (a === FRESH[j].c) ok++;
            cumA[t * nF + j] = (t ? cumA[(t - 1) * nF + j] : 0) + a; cumB[t * nF + j] = (t ? cumB[(t - 1) * nF + j] : 0) + b;
          }
          single.push(ok / nF * 100);
          let acc = 0, dis = 0;
          for (let j = 0; j < nF; j++) { const pa = cumA[t * nF + j] * 2 > t + 1 ? 1 : 0, pb = cumB[t * nF + j] * 2 > t + 1 ? 1 : 0; if (pa === FRESH[j].c) acc++; if (pa !== pb) dis++; }
          accAt.push(acc / nF * 100); disAt.push(dis / nF * 100);
          for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) { const k = r * N + c; grid[t * N * N + k] = (t ? grid[(t - 1) * N * N + k] : 0) + predictTree(FA[t], c + 0.5, r + 0.5); }
        }
        const odd = (arr) => arr.filter((_, i) => i % 2 === 0);   /* forests of 1, 3, 5 … trees */
        const S = { T: 1, view: 'forest', dirty: true };
        function draw() {
          g.clearRect(0, 0, W, H);
          frame(g, pl);
          const T = S.T;
          clipTo(g, pl, () => {
            if (S.view === 'forest') {
              paintCells(g, pl, N, (c, r) => {
                const share = grid[(T - 1) * N * N + r * N + c] / T;          /* share of trees saying fraud */
                const lvl = Math.min(4, Math.floor(Math.abs(share - 0.5) * 10));
                return share > 0.5 ? 'rgba(251,113,133,' + (0.08 + lvl * 0.06) + ')' : 'rgba(124,156,255,' + (0.07 + lvl * 0.05) + ')';
              });
            } else paintTree(g, pl, FA[T - 1], 0, 100, 0, 100);
            drawPayments(g, pl, TRAIN, 3);
          });
          worldAxes(g, pl);
          const mark = (T - 1) / 2;
          chart(g, c1, [{ ys: odd(accAt), color: C.green }], 65, 95, mark);
          chart(g, c2, [{ ys: odd(disAt), color: C.warn }], 0, 30, mark);
          g.textAlign = 'left'; g.font = BOLD; g.fillStyle = C.green; g.fillText('Forest\'s score on fresh payments', c1.x, c1.y - 10);
          g.fillStyle = C.warn; g.fillText('Two forests disagree on', c2.x, c2.y - 10);
          g.font = SMALL; g.fillStyle = C.muted;
          g.fillText('65% to 95%', c1.x, c1.y + c1.h + 15); g.fillText('0% to 30%', c2.x, c2.y + c2.h + 15);
          g.textAlign = 'right';
          g.fillText('1 tree → ' + MAXT, c1.x + c1.w, c1.y + c1.h + 15); g.fillText('1 tree → ' + MAXT, c2.x + c2.w, c2.y + c2.h + 15);
          ro.set({ trees: T, 'newest tree alone': pct(single[T - 1] / 100), 'forest, on 1,000 fresh': pct(accAt[T - 1] / 100), 'two forests disagree on': pct(disAt[T - 1] / 100) });
        }
        const ro = ctx.readout();
        ctx.loop(() => { if (S.dirty) { draw(); S.dirty = false; } });
        return ctx.figure(cv, 'Every tree here is the memorising kind: no depth limit. Each one grows from its own random redraw of the same 200 payments, and at every cut it may look at only one column, picked by coin flip. The forest\'s verdict is the majority vote, and the tint is stronger where more trees agree. The yellow chart compares this forest with a second one grown the same way from a <i>different</i> 200 payments: the share of the 1,000 fresh payments on which the two give different verdicts.', [
          ctx.slider({ label: 'trees in the forest', min: 1, max: MAXT, step: 2, value: 1, onChange: (v) => { S.T = v; S.dirty = true; } }),
          ctx.select({ label: 'show', value: 'forest', options: [{ value: 'forest', label: 'the forest\'s vote' }, { value: 'tree', label: 'only the newest tree' }], onChange: (v) => { S.view = v; S.dirty = true; } }),
        ], ro);
      }

      /* ================================================================== */
      /* 5. Gradient boosting, one stump at a time                           */
      /* ================================================================== */
      function boostFig() {
        const W = 720, H = 420, CAP = 400, YMAX = 150, RMAX = 80;
        const [cv, g] = ctx.canvas(W, H);
        const top = { x: 46, y: 28, w: 424, h: 180 }, bot = { x: 46, y: 250, w: 424, h: 120 }, cp = { x: 512, y: 52, w: 194, h: 180 };
        const D = makeBikes(33, 60, 14), HOLD = makeBikes(533, 400, 14);
        const mean = D.reduce((s, d) => s + d.y, 0) / D.length;
        const rms = (pred, data) => Math.sqrt(pred.reduce((s, v, i) => s + (v - data[i].y) ** 2, 0) / pred.length);
        const S = { lr: 0.3, dirty: true };
        function reset() { S.pred = D.map(() => mean); S.hold = HOLD.map(() => mean); S.stumps = []; S.tr = [rms(S.pred, D)]; S.ho = [rms(S.hold, HOLD)]; S.next = bestStump(D, D.map((d, i) => d.y - S.pred[i])); S.dirty = true; }
        function add(n) {
          for (let k = 0; k < n && S.stumps.length < CAP; k++) {
            const s = S.next; S.stumps.push(s);
            S.pred = S.pred.map((v, i) => v + S.lr * stumpOut(s, D[i].x)); S.hold = S.hold.map((v, i) => v + S.lr * stumpOut(s, HOLD[i].x));
            S.tr.push(rms(S.pred, D)); S.ho.push(rms(S.hold, HOLD));
            S.next = bestStump(D, D.map((d, i) => d.y - S.pred[i]));
          }
          S.dirty = true;
        }
        reset();
        const modelAt = (x) => { let v = mean; for (const s of S.stumps) v += S.lr * stumpOut(s, x); return v; };
        const tx = (x) => top.x + x / 24 * top.w;
        function draw() {
          g.clearRect(0, 0, W, H);
          frame(g, top); frame(g, bot);
          const ty = (v) => top.y + top.h - 4 - ctx.clamp(v, 0, YMAX) / YMAX * (top.h - 8);
          const by = (v) => bot.y + bot.h / 2 - ctx.clamp(v, -RMAX, RMAX) / RMAX * (bot.h / 2 - 4);
          clipTo(g, top, () => {
            g.strokeStyle = C.muted; g.setLineDash([4, 3]); g.lineWidth = 1.3; g.beginPath();
            for (let i = 0; i <= 240; i++) { const x = i / 10; if (i) g.lineTo(tx(x), ty(bikeTruth(x))); else g.moveTo(tx(x), ty(bikeTruth(x))); }
            g.stroke(); g.setLineDash([]);
            for (const d of D) { g.beginPath(); g.arc(tx(d.x), ty(d.y), 3.2, 0, Math.PI * 2); g.fillStyle = BLUE; g.fill(); }
            g.strokeStyle = C.green; g.lineWidth = 2.2; g.beginPath();
            for (let i = 0; i <= 480; i++) { const x = i / 20, y = ty(modelAt(x)); if (i) g.lineTo(tx(x), y); else g.moveTo(tx(x), y); }
            g.stroke();
          });
          clipTo(g, bot, () => {
            g.strokeStyle = C.line; g.lineWidth = 1; g.beginPath(); g.moveTo(bot.x, by(0)); g.lineTo(bot.x + bot.w, by(0)); g.stroke();
            D.forEach((d, i) => { const r = d.y - S.pred[i]; g.strokeStyle = 'rgba(124,156,255,0.45)'; g.beginPath(); g.moveTo(tx(d.x), by(0)); g.lineTo(tx(d.x), by(r)); g.stroke(); g.beginPath(); g.arc(tx(d.x), by(r), 2.8, 0, Math.PI * 2); g.fillStyle = BLUE; g.fill(); });
            if (S.next) {
              const s = S.next; g.strokeStyle = C.warn; g.lineWidth = 2.2; g.beginPath();
              g.moveTo(tx(0), by(s.lv)); g.lineTo(tx(s.t), by(s.lv)); g.lineTo(tx(s.t), by(s.rv)); g.lineTo(tx(24), by(s.rv)); g.stroke();
            }
          });
          g.font = SMALL; g.textAlign = 'left';
          g.fillStyle = C.muted; g.fillText('bike hires per hour:', top.x, top.y - 9);
          g.fillStyle = BLUE; g.fillText('● 60 recorded hours', top.x + 118, top.y - 9);
          g.fillStyle = C.green; g.fillText('▬ the model so far', top.x + 252, top.y - 9);
          g.fillStyle = C.muted; g.fillText('what is still wrong at each hour (truth − model):', bot.x, bot.y - 9);
          g.fillStyle = C.warn; g.fillText('▬ next tree', bot.x + 300, bot.y - 9);
          g.fillStyle = C.muted; g.textAlign = 'center';
          for (let hr = 0; hr <= 24; hr += 4) g.fillText(hr + ':00', tx(hr), bot.y + bot.h + 15);
          g.fillText('hour of the day', top.x + top.w / 2, bot.y + bot.h + 33);
          g.textAlign = 'right'; g.fillText(String(YMAX), top.x - 6, top.y + 10); g.fillText('0', top.x - 6, top.y + top.h);
          g.fillText('+' + RMAX, bot.x - 6, bot.y + 10); g.fillText('−' + RMAX, bot.x - 6, bot.y + bot.h);
          /* error history */
          const n = S.tr.length;
          chart(g, cp, [{ ys: S.tr, color: BLUE }, { ys: S.ho, color: C.orange }], 0, 35, n - 1);
          g.font = BOLD; g.textAlign = 'left'; g.fillStyle = C.text; g.fillText('Typical miss, per tree', cp.x, cp.y - 28);
          g.font = SMALL; g.fillStyle = C.muted; g.fillText('35 hires', cp.x, cp.y - 8); g.fillText('0', cp.x, cp.y + cp.h + 15);
          g.textAlign = 'right'; g.fillText('0 trees → ' + (n - 1), cp.x + cp.w, cp.y + cp.h + 15);
          g.textAlign = 'left';
          g.fillStyle = BLUE; g.fillText('● on its own 60 hours', cp.x, cp.y + cp.h + 40);
          g.fillStyle = C.orange; g.fillText('● on 400 fresh hours', cp.x, cp.y + cp.h + 58);
          let bi = 0; for (let i = 1; i < n; i++) if (S.ho[i] < S.ho[bi]) bi = i;
          ro.set({ trees: (n - 1) + (n - 1 >= CAP ? ' (the limit)' : ''), 'miss, own 60 hours': S.tr[n - 1].toFixed(1), 'miss, 400 fresh hours': S.ho[n - 1].toFixed(1), 'lowest fresh miss': S.ho[bi].toFixed(1) + ' at ' + bi + ' trees' });
        }
        const ro = ctx.readout();
        ctx.loop(() => { if (S.dirty) { draw(); S.dirty = false; } });
        const sl = ctx.slider({ label: 'learning rate', min: 0.1, max: 1, step: 0.1, value: 0.3, digits: 1, onChange: (v) => { S.lr = v; reset(); } });
        return ctx.figure(cv, 'A different job: predict a number. Sixty recorded hours of bike hires from a made-up city, with two commuter peaks (grey dashes: the true pattern, which the model never sees). The model starts as the flat average. Each new tree is a single cut fitted to the lower panel, the gaps between the truth and the model so far, and only a fraction of it is added: the learning rate. "Typical miss" is the root-mean-square error, in hires per hour. The data is noisy enough that nothing can get a fresh-hours miss below about 14.', [
          sl,
          ctx.button('Add 1 tree', () => add(1), 'primary'),
          ctx.button('Add 10 trees', () => add(10)),
          ctx.button('Add 100 trees', () => add(100)),
          ctx.button('Start again', reset),
          ctx.button('Broken preset: rate 1.0, 400 trees', () => { S.lr = 1; sl.value = 1; reset(); add(CAP); }),
        ], ro);
      }

      /* ================================================================== */
      /* 6. k nearest neighbours                                             */
      /* ================================================================== */
      function knnFig() {
        const W = 720, H = 372, N = 50, n = TRAIN.length;
        const [cv, g] = ctx.canvas(W, H);
        cv.style.touchAction = 'none';
        const pl = { x: 44, y: 30, w: 400, h: 290 }, cp = { x: 500, y: 52, w: 204, h: 210 };
        /* for each place we will ever ask about, the running fraud count among its 1st, 2nd, 3rd …
           nearest training payments, so any k is a single lookup */
        function prefixVotes(qs) {
          const out = new Uint8Array(qs.length * n), key = new Float64Array(n);
          qs.forEach((q, j) => {
            for (let i = 0; i < n; i++) { const dx = TRAIN[i].x - q.x, dy = TRAIN[i].y - q.y; key[i] = Math.round((dx * dx + dy * dy) * 4) * 1024 + i; }
            key.sort();
            let v = 0; for (let r = 0; r < n; r++) { v += TRAIN[key[r] % 1024].c; out[j * n + r] = v; }
          });
          return out;
        }
        const cells = []; for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) cells.push({ x: c * 2 + 1, y: r * 2 + 1 });
        const vGrid = prefixVotes(cells), vTrain = prefixVotes(TRAIN), vFresh = prefixVotes(FRESH);
        const score = (votes, pts, k) => { let ok = 0; pts.forEach((q, j) => { if ((votes[j * n + k - 1] * 2 > k ? 1 : 0) === q.c) ok++; }); return ok / pts.length * 100; };
        const ks = []; for (let k = 1; k < n; k += 2) ks.push(k);
        const trC = ks.map(k => score(vTrain, TRAIN, k)), frC = ks.map(k => score(vFresh, FRESH, k));
        /* the slider steps through these, so the small values where everything happens are easy to hit */
        const KS = [1, 3, 5, 7, 9, 11, 15, 21, 31, 45, 65, 99, 149, 199];
        const xs = ks.map(k => Math.log(k) / Math.log(199));
        const S = { k: 5, q: { x: 80, y: 86 }, drag: false, dirty: true };
        function aim(ev) {
          const d = toData(cv, pl, ev); if (!d) return;
          S.q = { x: Math.round(d.x), y: Math.round(d.y) }; S.dirty = true;
        }
        cv.addEventListener('pointerdown', (ev) => { S.drag = true; aim(ev); });
        cv.addEventListener('pointermove', (ev) => { if (S.drag) aim(ev); });
        const stop = () => { S.drag = false; };
        cv.addEventListener('pointerup', stop); cv.addEventListener('pointerleave', stop); cv.addEventListener('pointercancel', stop);
        function draw() {
          g.clearRect(0, 0, W, H);
          frame(g, pl);
          const k = S.k, near = nearest(TRAIN, S.q.x, S.q.y, k);
          let v = 0; for (const i of near) v += TRAIN[i].c;
          const far = TRAIN[near[near.length - 1]], rad = Math.hypot(far.x - S.q.x, far.y - S.q.y);
          clipTo(g, pl, () => {
            paintCells(g, pl, N, (c, r) => TINT[vGrid[(r * N + c) * n + k - 1] * 2 > k ? 1 : 0]);
            drawPayments(g, pl, TRAIN, 3);
            const qx = PX(pl, S.q.x), qy = PY(pl, S.q.y);
            g.strokeStyle = 'rgba(230,235,245,0.6)'; g.lineWidth = 1;
            if (k <= 25) for (const i of near) { g.beginPath(); g.moveTo(qx, qy); g.lineTo(PX(pl, TRAIN[i].x), PY(pl, TRAIN[i].y)); g.stroke(); }
            g.strokeStyle = C.warn; g.setLineDash([5, 4]); g.lineWidth = 1.5; g.beginPath(); g.ellipse(qx, qy, rad / 100 * (pl.w - 2 * PAD), rad / 100 * (pl.h - 2 * PAD), 0, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
            g.beginPath(); g.arc(qx, qy, 7, 0, Math.PI * 2); g.fillStyle = '#fff'; g.fill(); g.lineWidth = 2.5; g.strokeStyle = v * 2 > k ? RED : BLUE; g.stroke();
          });
          worldAxes(g, pl);
          const mark = (k - 1) / 2;
          const ch = chart(g, cp, [{ ys: trC, color: BLUE }, { ys: frC, color: C.orange }], 50, 100, mark, xs);
          g.font = BOLD; g.textAlign = 'left'; g.fillStyle = C.text; g.fillText('Score at each k', cp.x, cp.y - 12);
          g.font = SMALL; g.fillStyle = C.muted; g.textAlign = 'right';
          g.fillText('100%', cp.x - 6, cp.y + 10); g.fillText('50%', cp.x - 6, cp.y + cp.h);
          /* k runs on a stretched axis: each tick is roughly three times the one before */
          g.textAlign = 'center'; g.strokeStyle = C.line; g.lineWidth = 1;
          for (const k of [1, 3, 9, 31, 99]) {
            const x = ch.xAt((k - 1) / 2);
            g.beginPath(); g.moveTo(x, cp.y + cp.h); g.lineTo(x, cp.y + cp.h + 4); g.stroke();
            g.fillText(String(k), x, cp.y + cp.h + 17);
          }
          g.textAlign = 'right'; g.fillText('199', cp.x + cp.w, cp.y + cp.h + 17);
          g.textAlign = 'left'; g.fillText('k, with the small values spread out', cp.x, cp.y + cp.h + 35);
          g.fillStyle = BLUE; g.fillText('● on its own 200 payments', cp.x, cp.y + cp.h + 58);
          g.fillStyle = C.orange; g.fillText('● on 1,000 fresh payments', cp.x, cp.y + cp.h + 76);
          ro.set({ 'new payment': '£' + S.q.x + ', ' + S.q.y + ' mi', ['its ' + k + ' nearest']: v + ' fraud, ' + (k - v) + ' genuine', verdict: v * 2 > k ? 'fraud' : 'genuine', 'own 200': pct(trC[mark] / 100), 'fresh 1,000': pct(frC[mark] / 100) });
        }
        const ro = ctx.readout();
        ctx.loop(() => { if (S.dirty) { draw(); S.dirty = false; } });
        return ctx.figure(cv, 'No cuts and no training. The white dot is a new payment; drag it anywhere. Its k closest neighbours among the 200 known payments vote, and the dashed ring reaches exactly as far as the k-th of them (it is drawn as an oval only because the plot is wider than it is tall). The tint shows the verdict the same vote would give at every other spot. The slider and the chart both spend most of their length on small k, because that is where the scores change. Both columns run 0 to 100 here, so distance treats them fairly; real columns must be put on comparable scales first, or the one with the biggest numbers decides everything.', [
          ctx.slider({ label: 'k: how many neighbours vote', min: 0, max: KS.length - 1, step: 1, value: 2, fmt: (i) => String(KS[i]), onChange: (i) => { S.k = KS[i]; S.dirty = true; } }),
        ], ro);
      }

      /* ================================================================== */
      /* 7. k-means                                                          */
      /* ================================================================== */
      function kmeansFig() {
        const W = 720, H = 372, MAXC = 6;
        const [cv, g] = ctx.canvas(W, H);
        const pl = { x: 44, y: 30, w: 400, h: 290 };
        const pts = makeBlobs(6);
        const PAL = [C.green, C.warn, C.pink, C.accent, C.orange, C.purple];
        const S = { cs: [], steps: 0, settled: false, trails: [], dirty: true };
        function clear() { S.cs = []; S.trails = []; S.steps = 0; S.settled = false; S.dirty = true; }
        function setStart(list) { clear(); S.cs = list.map(([x, y]) => ({ x, y })); S.trails = S.cs.map(c => [{ x: c.x, y: c.y }]); }
        function step() {
          if (!S.cs.length || S.settled) return false;
          const moved = kmStep(pts, S.cs);
          S.cs.forEach((c, j) => S.trails[j].push({ x: c.x, y: c.y }));
          if (moved < 1e-9) S.settled = true; else S.steps++;
          S.dirty = true; return !S.settled;
        }
        /* a click, not a pointerdown: a finger that starts a scroll on the plot must not drop a centre */
        cv.addEventListener('click', (ev) => {
          const d = toData(cv, pl, ev);
          if (!d || S.cs.length >= MAXC) return;
          S.cs.push({ x: Math.round(d.x), y: Math.round(d.y) }); S.trails.push([{ x: Math.round(d.x), y: Math.round(d.y) }]); S.settled = false; S.dirty = true;
        });
        function draw() {
          g.clearRect(0, 0, W, H);
          frame(g, pl);
          const as = S.cs.length ? kmAssign(pts, S.cs) : null;
          clipTo(g, pl, () => {
            pts.forEach((q, i) => { g.beginPath(); g.arc(PX(pl, ctx.clamp(q.x, 1, 99)), PY(pl, ctx.clamp(q.y, 1, 99)), 3.6, 0, Math.PI * 2); g.fillStyle = as ? PAL[as[i]] : C.muted; g.fill(); });
            S.cs.forEach((c, j) => {
              g.strokeStyle = PAL[j]; g.lineWidth = 1.5; g.setLineDash([3, 3]); g.beginPath();
              S.trails[j].forEach((t, k) => { if (k) g.lineTo(PX(pl, t.x), PY(pl, t.y)); else g.moveTo(PX(pl, t.x), PY(pl, t.y)); }); g.stroke(); g.setLineDash([]);
              const x = PX(pl, c.x), y = PY(pl, c.y);
              g.beginPath(); g.moveTo(x, y - 10); g.lineTo(x + 10, y); g.lineTo(x, y + 10); g.lineTo(x - 10, y); g.closePath();
              g.fillStyle = PAL[j]; g.fill(); g.lineWidth = 2.5; g.strokeStyle = '#fff'; g.stroke();
            });
          });
          g.font = SMALL; g.fillStyle = C.muted; g.textAlign = 'center';
          for (let v = 0; v <= 100; v += 20) g.fillText(String(v), PX(pl, v), pl.y + pl.h + 15);
          g.fillText('shop visits a year', pl.x + pl.w / 2, pl.y + pl.h + 32);
          g.textAlign = 'right'; for (let v = 20; v <= 100; v += 20) g.fillText(String(v), pl.x - 6, PY(pl, v) + 4);
          g.textAlign = 'left'; g.fillText('average basket, £ ↑', pl.x, pl.y - 9);
          g.textAlign = 'right'; g.fillText('100 customers, no labels', pl.x + pl.w, pl.y - 9);
          /* side panel: who owns what */
          const tx = 476; let y = 52;
          g.textAlign = 'left'; g.font = BOLD; g.fillStyle = C.text; g.fillText('Centres and their customers', tx, y); y += 26;
          g.font = SMALL;
          if (!S.cs.length) { g.fillStyle = C.muted; g.fillText('Tap or click the plot to drop one.', tx, y); }
          S.cs.forEach((c, j) => {
            let k = 0; as.forEach(a => { if (a === j) k++; });
            g.fillStyle = PAL[j]; g.fillText('◆ centre ' + (j + 1) + ': ' + k + ' customers', tx, y); y += 22;
          });
          g.fillStyle = C.muted;
          g.fillText('Each dot takes the colour of', tx, H - 62); g.fillText('its nearest centre. A step moves', tx, H - 44); g.fillText('each centre to its dots\' average.', tx, H - 26);
          ro.set({ centres: S.cs.length, 'steps taken': S.steps, 'typical distance to own centre': S.cs.length ? kmSpread(pts, S.cs).toFixed(1) : '–', state: !S.cs.length ? 'no centres yet' : S.settled ? 'settled: nothing moves' : 'not settled' });
        }
        const ro = ctx.readout();
        ctx.loop(() => { if (S.dirty) { draw(); S.dirty = false; } });
        return ctx.figure(cv, 'A hundred customers of a made-up shop, and nobody has labelled anything. Tap or click the plot to drop up to six centres. <b>Step</b> does the whole algorithm once: every customer joins the nearest centre, then every centre moves to the average position of its customers. "Typical distance" is the root-mean-square distance from a customer to their own centre, the number k-means is trying to shrink.', [
          ctx.button('Step', () => { step(); }, 'primary'),
          ctx.button('Run until settled', () => { let guard = 0; while (step() && guard++ < 200); }),
          ctx.button('Clear centres', clear),
          ctx.button('Broken preset: bad start', () => setStart([[14, 38], [30, 22], [74, 51]])),
          ctx.button('Spread-out start', () => setStart([[30, 50], [60, 60], [60, 30]])),
        ], ro);
      }

      /* ================================================================== */
      /* 8. A straight line and a sigmoid on the same yes/no data            */
      /* ================================================================== */
      function lineVsSigmoid() {
        const W = 720, H = 372, XMAX = 25, KEEN = 24, YLO = -0.45, YHI = 1.75;
        const [cv, g] = ctx.canvas(W, H);
        const pl = { x: 58, y: 28, w: 640, h: 262 };
        const ST = makeStudents(1, 24);
        const S = { keen: 10, q: 2, dirty: true };
        const X = (x) => pl.x + x / XMAX * pl.w, Y = (v) => pl.y + pl.h - (ctx.clamp(v, YLO, YHI) - YLO) / (YHI - YLO) * pl.h;
        function draw() {
          g.clearRect(0, 0, W, H);
          frame(g, pl);
          const xs = ST.map(s => s.x).concat([S.keen]), ys = ST.map(s => s.c).concat([1]);
          const L = fitLine(xs, ys), G = fitLogistic(xs, ys);
          const tl = (0.5 - L.b) / L.w, tg = -G.b / G.w;
          const wrongWith = (thr) => { let k = 0; xs.forEach((x, i) => { if ((x > thr ? 1 : 0) !== ys[i]) k++; }); return k; };
          clipTo(g, pl, () => {
            g.strokeStyle = C.line; g.lineWidth = 1;
            for (const v of [0, 0.5, 1]) { g.setLineDash(v === 0.5 ? [5, 4] : []); g.beginPath(); g.moveTo(pl.x, Y(v)); g.lineTo(pl.x + pl.w, Y(v)); g.stroke(); }
            g.setLineDash([]);
            g.strokeStyle = 'rgba(230,235,245,0.5)'; g.beginPath(); g.moveTo(X(S.q), pl.y); g.lineTo(X(S.q), pl.y + pl.h); g.stroke();
            g.strokeStyle = C.orange; g.lineWidth = 2.2; g.beginPath(); g.moveTo(X(0), Y(L.b)); g.lineTo(X(XMAX), Y(L.w * XMAX + L.b)); g.stroke();
            g.strokeStyle = C.green; g.beginPath();
            for (let i = 0; i <= XMAX * 10; i++) { const x = i / 10, y = Y(sigmoid(G.w * x + G.b)); if (i) g.lineTo(X(x), y); else g.moveTo(X(x), y); }
            g.stroke();
            g.setLineDash([3, 4]); g.lineWidth = 1.5;
            g.strokeStyle = C.orange; g.beginPath(); g.moveTo(X(tl), Y(0.5) - 34); g.lineTo(X(tl), Y(0.5) + 34); g.stroke();
            g.strokeStyle = C.green; g.beginPath(); g.moveTo(X(tg), Y(0.5) - 34); g.lineTo(X(tg), Y(0.5) + 34); g.stroke();
            g.setLineDash([]);
            const seen = {};
            ST.forEach((s) => { const key = s.x + '|' + s.c, k = seen[key] = (seen[key] || 0) + 1; g.beginPath(); g.arc(X(s.x), Y(s.c) + (s.c ? -1 : 1) * (k - 1) * 9, 4, 0, Math.PI * 2); g.fillStyle = s.c ? BLUE : RED; g.fill(); });
            g.beginPath(); g.arc(X(S.keen), Y(1), 6.5, 0, Math.PI * 2); g.fillStyle = C.warn; g.fill(); g.lineWidth = 2; g.strokeStyle = '#fff'; g.stroke();
          });
          g.font = SMALL; g.fillStyle = C.muted; g.textAlign = 'right';
          g.fillText('passed 1', pl.x - 6, Y(1) + 4); g.fillText('0.5', pl.x - 6, Y(0.5) + 4); g.fillText('failed 0', pl.x - 6, Y(0) + 4);
          g.textAlign = 'center';
          for (let v = 0; v <= XMAX; v += 4) g.fillText(String(v), X(v), pl.y + pl.h + 15);
          g.fillText('hours of revision', pl.x + pl.w / 2, pl.y + pl.h + 32);
          g.textAlign = 'left';
          g.fillStyle = C.orange; g.fillText('▬ straight line, least squares', pl.x, pl.y + pl.h + 54);
          g.fillStyle = C.green; g.fillText('▬ S-curve: the same kind of line, through a sigmoid', pl.x + 200, pl.y + pl.h + 54);
          g.fillStyle = C.warn; g.fillText('● the keen student', pl.x + 520, pl.y + pl.h + 54);
          g.fillStyle = C.muted; g.fillText('Short dashed marks: where each model crosses 0.5 and starts saying "pass". Thin white line: the new student.', pl.x, pl.y + pl.h + 74);
          const lq = L.w * S.q + L.b, gq = sigmoid(G.w * S.q + G.b);
          ro.set({
            'line says pass above': tl.toFixed(1) + ' h, wrong on ' + wrongWith(tl) + ' of 25',
            'S-curve says pass above': tg.toFixed(1) + ' h, wrong on ' + wrongWith(tg) + ' of 25',
            'line, for the keen student': (L.w * S.keen + L.b).toFixed(2),
            'new student: line': lq.toFixed(2), 'S-curve': gq.toFixed(2),
          });
        }
        const ro = ctx.readout();
        ctx.loop(() => { if (S.dirty) { draw(); S.dirty = false; } });
        const keenSl = ctx.slider({ label: 'the keen student\'s hours', min: 10, max: KEEN, step: 1, value: 10, onChange: (v) => { S.keen = v; S.dirty = true; } });
        return ctx.figure(cv, 'Twenty-five made-up students: hours of revision, and whether they passed (blue, drawn at 1) or failed (red, drawn at 0). Both models are refitted from scratch every time you move a slider. The orange one is chapter 1\'s <code class="inline">w·x + b</code> fitted by least squares. The green one feeds the same kind of sum through chapter 11\'s sigmoid. Each says "pass" where its output is above 0.5.', [
          keenSl,
          ctx.slider({ label: 'a new student\'s hours', min: 0, max: KEEN, step: 0.5, value: 2, digits: 1, onChange: (v) => { S.q = v; S.dirty = true; } }),
          ctx.button('Broken preset: a ' + KEEN + '-hour student', () => { S.keen = KEEN; keenSl.value = KEEN; S.dirty = true; }),
          ctx.button('Back to 10 hours', () => { S.keen = 10; keenSl.value = 10; S.dirty = true; }),
        ], ro);
      }

      /* ================================================================== */
      /* The chapter                                                         */
      /* ================================================================== */
      root.append(
        p(`Chapter 1 left a fraud problem unsolved. Fraud sat at both ends of the amount scale, and one straight line cannot slice both ends off a stick. So put the ruler down and pick up scissors. Your cuts must run straight up or straight across, and you may make several.`),
        callout('tryit', '🖐 Do this first, read after',
          `<b>1.</b> Before cutting, read <b>mistakes now</b>: 16 of 40. One box, mostly genuine, so it says "genuine" and misses every fraud.<br>
           <b>2.</b> Move the yellow line to <b>amount at £20</b>: point and click, or tap, and use the <b>Nudge</b> buttons if you land a pound or two off. Press <b>Cut here</b>. Mistakes: 8. The tiny payments on the left are nearly all fraud.<br>
           <b>3.</b> Aim inside the big right-hand box and cut again at <b>£64</b>. Mistakes: still 8. Apparently nothing gained.<br>
           <b>4.</b> Press <b>Turn the cut</b>, aim at the far-right box and cut at <b>38 miles</b>. Mistakes: 3.<br>
           <b>5.</b> Press <b>Start again</b> and try to beat 3 with three cuts of your own. Then press <b>Show the machine's three cuts</b>.`),
        handTree(),
        p(`You just grew a <em>decision tree</em>. The name comes from the panel on the right: one question at the top, branching into more questions, ending in boxes that each give a verdict. The final boxes are called <em>leaves</em>, and each cut is a <em>split</em>.`),
        p(`Look at what it did to chapter 1's impossible problem. Fraud at both ends of the amount scale took two cuts on one column. A third cut, on a second column, caught that large payments are only suspicious far from home. One line cannot say that. Three questions can.`),
        p(`The three payments still wrong are ones whose label breaks the pattern: a genuine payment deep in fraud country, or the reverse. Real data always has some. Remember them, because they are about to cause trouble.`),
        p(`One thing should be nagging. Your second cut changed nothing on the scoreboard, yet without it the third cut had nowhere to go. Counting mistakes is a poor guide to where to cut next. The machine uses a better one.`),
      );

      root.append(section('How the machine chooses where to cut',
        callout('tryit', '🖐 Try this — score a cut before you make it',
          `<b>1.</b> The slider starts at 50. Read <b>gain</b> under the picture: 0.030 bits. A cut down the middle tells you almost nothing.<br>
           <b>2.</b> Drag <b>cut at</b> down to 20. The left box now holds 9 fraud and 1 genuine, and the gain is 0.266 bits.<br>
           <b>3.</b> Press <b>Jump to the best cut</b>. Nothing moves: you are already on the peak of the green curve, and it is the cut you made by eye.<br>
           <b>4.</b> Change <b>cut on</b> to miles from home and press the jump button again. The best that column can offer is 0.069 bits, and look where: at 97 miles, shaving two payments off the top edge. That is a fluke of this sample. Distance tells you almost nothing until the amount cuts are in.<br>
           <b>5.</b> Go back to payment amount, change <b>measure</b> to Gini impurity and press the jump button once more. Different numbers (the gain reads 0.167), same cut: £20.`),
        gainExplorer(),

        /* maths beat: entropy and information gain, from the counts in the tree they just split */
        ctx.walkthrough([
          { say: 'With the cut at £20, the left box held 10 payments: <b>9 fraud and 1 genuine</b>.', math: '9 &divide; 10 = 0.9 fraud &nbsp;&nbsp;&nbsp; 1 &divide; 10 = 0.1 genuine', note: 'Everything that follows is built from counts like these and nothing else.' },
          { say: 'Pull one payment out of that box without looking: chapter 2\'s cross-entropy charged &minus;log p for an outcome given probability p, and read as surprise, with the log taken in base 2, that charge is measured in <b>bits</b>.', math: 'fraud: &minus;log<sub>2</sub> 0.9 = 0.15 bits &nbsp;&nbsp;&nbsp; genuine: &minus;log<sub>2</sub> 0.1 = 3.32 bits', note: 'Base 2 makes a fair coin flip cost exactly 1 bit. Chapter 2 used a different base, which is why its 50% shrug cost 0.69; chapter 8 counted in bits, as here.' },
          { say: 'You will get the small surprise nine times in ten and the big one once in ten, so take the average.', math: '0.9 &times; 0.15 + 0.1 &times; 3.32 = 0.47 bits', note: 'That average surprise is the box\'s <b>entropy</b>. The demo printed it as 0.469.' },
          { ask: 'A different box holds <b>10 fraud and 10 genuine</b>. What is its entropy?',
            options: ['0 bits', '0.5 bits', '1 bit'], answer: 2,
            explain: 'Each outcome has probability 0.5 and costs &minus;log<sub>2</sub> 0.5 = 1 bit, so the average is 1 bit. Half-and-half is the most mixed a two-way box can be. A box of one kind only scores 0.' },
          { say: 'The whole pile before any cut held 16 fraud and 24 genuine, which is close to a coin flip.', math: '0.4 &times; 1.32 + 0.6 &times; 0.74 = 0.97 bits' },
          { say: 'The right-hand box after the cut held 7 fraud and 23 genuine, and the same sum gives it 0.78 bits.', math: 'left box: 0.47 bits &nbsp;&nbsp;&nbsp; right box: 0.78 bits' },
          { say: 'Ten of the 40 payments went left and thirty went right, so weight each box by its share of the payments.', math: '10/40 &times; 0.47 + 30/40 &times; 0.78 = 0.70 bits', note: 'A big messy box should count for more than a small tidy one.' },
          { say: 'The <b>information gain</b> of the cut is the entropy before it minus the entropy after it.', math: '0.97 &minus; 0.70 = 0.27 bits', note: 'The demo prints 0.266, because it does not round along the way.' },
          { ask: 'A cut puts 20 payments on each side, and each side holds 8 fraud and 12 genuine. What is its gain?',
            options: ['0 bits', '0.5 bits', '0.97 bits'], answer: 0,
            explain: 'Both new boxes have exactly the mix the pile had before: 40% fraud. Same entropy before and after, so nothing was learned.' },
          { say: 'The green curve was this one calculation repeated at all 99 positions, and the machine takes the peak.' },
          { ask: 'Your second cut, at £64, split the right box (7 fraud, 23 genuine) into 1 fraud with 17 genuine and 6 fraud with 6 genuine. Mistakes stayed at 8. What happened to the entropy of those 30 payments?',
            options: ['It rose, because one new box is half-and-half', 'It fell from 0.78 to 0.59 bits', 'It stayed at 0.78 bits, like the mistakes'], answer: 1,
            explain: '18/30 &times; 0.31 + 12/30 &times; 1.00 = 0.59 bits. One box became almost pure and the mess was packed into a small one, ready for the third cut. Entropy sees progress that a count of mistakes misses.' },
        ], {
          title: 'From counts to information gain',
          recap: 'entropy is the average surprise in a box; a cut\'s gain is the entropy before it, minus the size-weighted entropy of the boxes it makes.',
        }),
        ctx.decoder([
          { sym: 'gain', name: 'information gain', says: 'How many bits of uncertainty the cut removed. Zero means the cut told you nothing.', points: 'the green number in the panel, and the height of the green curve.' },
          '=',
          { sym: 'H(before)', name: 'the entropy of the box before the cut', says: 'Average surprise, in bits, when you draw one payment from the uncut box. H is the usual letter for entropy.', points: 'the first bar: 16 fraud, 24 genuine, 0.971 bits.' },
          '&minus; [',
          { sym: 'n<sub>L</sub>/n', name: 'the share that went left', says: 'The number of payments in the left box, divided by the number there were before the cut.', points: '10/40 at a cut of £20.' },
          '&times;',
          { sym: 'H(left)', name: 'the entropy of the left box', says: 'The same average-surprise sum, using only the payments that landed on the left.', points: 'the second bar: 9 fraud, 1 genuine, 0.469 bits.' },
          '+',
          { sym: 'n<sub>R</sub>/n', name: 'the share that went right', says: 'The rest of the payments, as a fraction of the total. The two shares always add up to 1.', points: '30/40 at a cut of £20.' },
          '&times;',
          { sym: 'H(right)', name: 'the entropy of the right box', says: 'Average surprise among the payments that landed on the right.', points: 'the third bar: 7 fraud, 23 genuine, 0.784 bits.' },
          ']',
        ], {
          title: 'gain = H(before) − [ share left × H(left) + share right × H(right) ]',
          plain: 'the gain of a cut is how mixed the box was, minus how mixed the two new boxes are, each counted in proportion to its size.',
        }),

        p(`<em>Entropy</em> is the average surprise in a box, measured in bits. A box of one kind has none. A half-and-half box has one full bit. <em>Information gain</em> is how far a cut lowers the entropy, once each new box is weighted by how many payments landed in it.`),
        p(`That explains your second cut. Slicing at £64 left the mistake count at 8, but one side became almost purely genuine and the mess was packed into a small box that a third cut could clean. Entropy registers that as progress. A count of mistakes does not.`),
        p(`<em>Gini impurity</em> does the same job with cheaper arithmetic: one, minus the square of each share. For the uncut pile that is 1 − 0.4² − 0.6² = 0.48, the chance that two payments drawn at random (putting the first one back) carry different labels. You watched it choose the same cut. The two measures rarely disagree. Gini is the default in scikit-learn, the most widely used Python library for these methods, and it is slightly cheaper because it skips the logarithm.`),
        p(`The whole tree-growing program is now on the table. For every column and every position, compute the gain. Take the best. Repeat inside each new box. It never looks ahead and never revisits a cut, which is why the method is called <em>greedy</em>. It is also fast: no gradients, no learning rate, no epochs.`),
        callout('example', '🌍 A model you can follow with a finger',
          `A small decision tree is the one model in this course that fits on a sheet of paper and can be checked by a person, question by question. Triage charts, troubleshooting guides and loan pre-checks have this shape whether someone drew them or a program grew them from records.
           When a bank has to say why an application was refused, a path of yes/no answers is a reason. Ten million weights are not.`),
      ));

      root.append(section('A tree that knows its training data by heart',
        p(`Three cuts were enough for forty payments. Hand a program 200 and no limit, and it does not stop at three.`),
        callout('tryit', '🖐 Try this — let it memorise',
          `<b>1.</b> At depth 3 the tree has 8 boxes and scores 92.0% on its own 200 payments, 86.2% on fresh ones.<br>
           <b>2.</b> Press <b>No limit: memorise everything</b>. On its own payments: 100.0%. On fresh ones: 77.6%. It now has 41 boxes, many of them slivers fenced around a single dot.<br>
           <b>3.</b> Still at depth 12, press <b>New training sample</b> a few times. Same bank, a different 200 payments, and the slivers land somewhere new every time.<br>
           <b>4.</b> Press <b>Back to depth 3</b> and cycle through the samples again. The big shapes barely move.`),
        deepTree(),
        p(`This is chapter 3's overfitting, done with scissors. An unlimited tree keeps cutting until every box is pure, so every mislabelled payment gets a private sliver. Each sliver is a rule learned from one accident, and it is wrong for whichever fresh payment lands there next.`),
        p(`The sample button showed something worse than a low score. The deep tree is not wrong in a steady way. It is wrong differently for every sample it is given, which chapter 3 called variance.`),
        p(`One cure is the depth limit you just used, or its cousin <em>pruning</em>: grow the full tree, then remove the branches that do not earn their keep on held-out data. The other cure is stranger, and it needs nobody to pick the right depth.`),
      ));

      root.append(section('Many unreliable trees, one reliable vote',
        callout('tryit', '🖐 Try this — add trees and watch two numbers',
          `<b>1.</b> One tree. It scores 76.7% on fresh payments. A second one-tree forest, grown from a different 200 payments, gives a different verdict on 21.4% of them.<br>
           <b>2.</b> Change <b>show</b> to only the newest tree and drag the slider a little. Every tree is a different jagged mess.<br>
           <b>3.</b> Change <b>show</b> back to the forest's vote and drag <b>trees in the forest</b> to 99. Fresh score: 86.9%. Disagreement: 7.0%.<br>
           <b>4.</b> Stop at 25 on the way back down: the disagreement is already 8.2%. Most of the fall happens early.`),
        forestFig(),
        p(`No tree improved. Each one is still the memorising kind, and <b>newest tree alone</b> kept reading in the 70s and low 80s as you dragged. The vote is what improved, for the reason chapter 3's large batches gave a steadier arrow than single examples: errors that point in different directions cancel in an average, and what the trees agree on survives.`),
        p(`Cancelling only works if the trees are wrong in <i>different</i> places, so the method manufactures disagreement twice. Each tree learns from a <em>bootstrap sample</em>: 200 payments drawn from the 200 with replacement, so some appear twice and about a third are left out. And at each cut the tree may consider only a few columns picked at random.`),
        p(`Growing trees on bootstrap samples and letting them vote is called <em>bagging</em>. Add the random columns and you have a <em>random forest</em>. This demo has only two columns, so "a few" means one; for yes/no problems like this one, real forests commonly try about the square root of the number of columns at each cut (scikit-learn's default).`),
        p(`The yellow line is variance you can read off a chart. Built from different data, two single trees gave different verdicts on more than a fifth of all payments. Two forests of 99 differ on 7%. It never reaches zero, because the trees inside a forest share the same 200 payments and so share some mistakes. Adding trees does not make a forest overfit. They only stop helping.`),
        p(`Keep the size of the win in proportion. On this small two-column problem the forest's 86.9% only matches the depth-3 tree's 86.2%, and both sit just under the 89.7% ceiling. Nobody had to choose a depth to get it, though, and every tree in it is one you would never trust alone. On wide, messy tables, where no small tree is good enough, a forest usually beats any single tree.`),
        callout('history', '📜 Leo Breiman, three times',
          `In 1984 the statisticians Leo Breiman, Jerome Friedman, Richard Olshen and Charles Stone published <i>Classification and Regression Trees</i>, the book that set out the recipe you just ran, Gini impurity included.
           Ross Quinlan's ID3 algorithm, set out in his 1986 paper "Induction of Decision Trees", reached trees from the computer-science side and chose its cuts with the entropy Claude Shannon had defined in 1948.<br>
           Breiman then went after the weakness you saw at depth 12: bagging in 1996, random forests in 2001. A forest needs very little tuning and is hard to break, which made it a standard first model to try on a new table of data.`),
        callout('example', '🌍 A forest in the living room',
          `Microsoft's Kinect, the 2010 sensor that let an Xbox follow your body with no controller, labelled each pixel of its depth image as hand, elbow, knee and so on with a random forest (Shotton and colleagues, 2011).
           Trees suited the job because a verdict is a short run of comparisons, cheap enough to make for every pixel of every frame.`),
      ));

      root.append(section('Fix what is still wrong, one small tree at a time',
        p(`A forest grows its trees separately and averages them. There is a second way to combine trees, and it made its name by winning competitions.`),
        callout('tryit', '🖐 Try this — build a curve out of single cuts',
          `<b>1.</b> Before adding anything, the model is a flat line at the average and its typical miss is 30.5 hires on its own hours, 30.9 on fresh ones. The lower panel shows what is wrong at each hour, and the yellow step is the best single cut through those gaps.<br>
           <b>2.</b> Press <b>Add 1 tree</b> ten times, slowly. Each press adds a step to the green staircase, shrinks the gaps, and aims a new yellow tree at whatever is left. After ten: 21.0 and 21.2.<br>
           <b>3.</b> Press <b>Start again</b>, then <b>Add 100 trees</b>. Both commuter peaks are there: 12.0 on its own hours, 15.6 on fresh ones.<br>
           <b>4.</b> Press <b>Broken preset: rate 1.0, 400 trees</b>. Own hours: 6.5. Fresh hours: 18.7, and the readout says the lowest fresh miss was 15.4, back at 14 trees.<br>
           <b>5.</b> Drag <b>learning rate</b> down to 0.1, which starts the model again, and press the hundred-tree button four times. The same 400 trees, each added at a tenth of its size: 11.5 on its own hours, and a fresh miss of 15.6 that has not turned upwards yet.`),
        boostFig(),
        p(`Each tree had one job: predict what the model so far still gets wrong. Those leftover gaps are called <em>residuals</em>. Add the new tree's guess to the running total, work out the residuals again, repeat. Building a model as a sequence of corrections is called <em>boosting</em>.`),
        p(`Every tree here is as weak as a tree can be: one cut and two values, known as a <em>stump</em>. Alone it is nearly useless. A hundred of them, stacked, drew two rush hours. Production systems use slightly bigger trees, typically a handful of questions deep, and hundreds or thousands of them.`),
        p(`Unlike a forest, a boosted model <i>can</i> overfit by growing. At a rate of 1.0 every tree is added whole and the model races to fit its own 60 hours, noise included: the fresh miss bottomed out after 14 trees, then climbed to 18.7. So practitioners watch a held-out score and stop when it turns, which is chapter 3's early stopping.`),
        p(`The learning rate is chapter 1's η again: take only part of each correction. It sets the pace. At 0.1 the same 400 trees were still at their best. Notice what it did not do here: the lowest fresh miss was about 15.5 at every rate. A small rate needs more trees and forgives you for adding too many. On real data it usually ends a little more accurate as well, so the common practice is a small rate with many trees.`),
        p(`Why "gradient"? For squared error, the residual is exactly the direction each prediction should move to lower the loss. That is the negative gradient from chapter 2, taken with respect to the predictions instead of the weights. Change the loss and the trees fit that loss's gradient instead. <em>Gradient boosting</em> is gradient descent in which every step is a tree.`),
        callout('example', '🌍 XGBoost, LightGBM and the spreadsheet',
          `Jerome Friedman set out gradient boosting in a 2001 paper. In 2016 Tianqi Chen and Carlos Guestrin published XGBoost, a fast and careful implementation, and reported that 17 of the 29 winning solutions posted on Kaggle's blog during 2015 used it. Microsoft's LightGBM followed in 2017.<br>
           The family is still the first thing to try when data comes as rows and columns: customer records, transactions, sensor logs. A cut does not care whether a column is in pounds or pence. A column with no signal is never chosen. A sharp threshold such as "under £20" costs one question, where a smooth network has to bend itself around it.
           A 2022 study by Grinsztajn, Oyallon and Varoquaux ran the comparison on 45 medium-sized tables and found tree ensembles still ahead of deep networks. Chapter 17 is about making that choice.`),
      ));

      root.append(section('No training at all: ask the neighbours',
        callout('tryit', '🖐 Try this — move the question, then change who votes',
          `<b>1.</b> The white dot is a new payment of £80, made 86 miles from home. With k at 5 its nearest neighbours vote 4 fraud to 1 genuine: fraud.<br>
           <b>2.</b> Drag <b>k: how many neighbours vote</b> down to 1. The verdict flips to genuine, because the single nearest payment is one of the mislabelled ones. Own 200: 100.0%. Fresh 1,000: 78.1%. The tint has broken into islands around single dots.<br>
           <b>3.</b> Drag k all the way up to 199. The whole plot says genuine: 53.0% on its own 200, 59.5% on fresh payments.<br>
           <b>4.</b> Bring k back to 5, where the fresh score reads 85.4%. On the chart the orange line peaks about here, and the blue line has come down to meet it.<br>
           <b>5.</b> Now drag the white dot around the plot. The ring stretches where known payments are sparse and tightens where they are dense.`),
        knnFig(),
        p(`<em>k-nearest neighbours</em>, k-NN for short, has no training step at all. The model is the data, kept whole. To judge a new payment it measures the distance to every stored one, takes the k closest, and reports their majority.`),
        p(`k is the depth dial from the tree, turned the other way. At k = 1 every stored payment rules its own patch, mislabelled ones included: the same memorising as the unlimited tree, and nearly the same fresh score. At k = 199 almost everyone votes on every question, so the answer is always the commoner class, and you are left with the 59.5% that saying "genuine" every time would earn.`),
        p(`The costs arrive at the opposite end from a neural network's. Training is free. Every single prediction means searching the stored data, and "closest" only means something when the columns are on comparable scales. You have used the search half before: chapter 6's five nearest words by cosine similarity were a nearest-neighbour lookup over embeddings.`),
        callout('example', '🌍 "More like this"',
          `A "similar items" shelf is a nearest-neighbour lookup: turn every song, product or photo into a list of numbers, then return the stored ones closest to the one on screen.
           The vector database from chapter 6 does the same over embedded chunks of text, which is the retrieval half of RAG.`),
      ));

      root.append(section('No labels at all: find the groups',
        p(`Every method so far learned from examples that came with the answer attached. Take the answers away and one question is still askable: which of these belong together?`),
        callout('tryit', '🖐 Try this — drop the centres yourself, then get stuck',
          `<b>1.</b> Tap or click three places on the plot to drop three centres, then press <b>Step</b> until the readout says settled.<br>
           <b>2.</b> Press <b>Broken preset: bad start</b>, then <b>Run until settled</b>. Typical distance: 19.6. Two centres are sharing the left-hand crowd and one is stranded between the two crowds on the right.<br>
           <b>3.</b> Press <b>Step</b> again. Nothing moves. As far as the algorithm can tell, it has finished.<br>
           <b>4.</b> Press <b>Spread-out start</b> and run it to the end the same way. Typical distance: 11.3, one centre per crowd.<br>
           <b>5.</b> Press <b>Clear centres</b> and try two centres, then six. Six well-spread centres push the distance below 11.3, although there are only three crowds.`),
        kmeansFig(),
        p(`<em>k-means</em> is the first method here with no right answers to learn from, which chapter 3 called unsupervised learning. It knows only positions, and it repeats two moves: every point joins its nearest centre, then every centre moves to the middle of its points. Neither move can raise the typical distance, so it always settles.`),
        p(`Settling is not succeeding. After the bad start, every centre already sits at the average of the points nearest to it, so no move helps, although a far better arrangement exists. It is the local minimum from chapter 2, where the ball stopped in the wrong dip.`),
        p(`The standard defence is blunt: run it several times from different starts and keep the lowest distance. A smarter seeding rule called k-means++, which tends to pick starting centres far apart from one another, is scikit-learn's default.`),
        p(`Two cautions. You choose k, and the distance falls every time k rises (give each customer a private centre and it reaches zero), so the lowest number cannot choose k for you. And the algorithm will carve up data that has no groups in it at all. A <em>cluster</em> is a suggestion to go and look, never a finding by itself.`),
        callout('example', '🌍 Segments and palettes',
          `Marketing "segments" are often k-means centres with names attached afterwards: the weekly big-basket shoppers, the daily top-up shoppers.
           Saving a photo with only 16 colours can be the same algorithm run on pixels: every pixel is a point in red-green-blue space, the 16 centres become the palette, and each pixel is repainted as its centre.`),
      ));

      root.append(section('Chapter 1\'s line, used two ways',
        p(`One model in this chapter you have already built. It has been waiting since chapter 1.`),
        callout('tryit', '🖐 Try this — break a straight line with one good student',
          `<b>1.</b> To begin with the two models agree: each says pass above 5.5 hours, and each is wrong on 4 of 25.<br>
           <b>2.</b> Drag <b>a new student's hours</b> to 0. The line's answer is −0.25, a negative chance of passing. The S-curve says 0.01.<br>
           <b>3.</b> Press <b>Broken preset: a 24-hour student</b>. One diligent student, who passed, as anyone would expect. The line swings to reach them: its pass mark jumps to 7.1 hours and it is wrong on 8 of 25. The S-curve has not moved.<br>
           <b>4.</b> Read <b>line, for the keen student</b>: 1.45. As a probability, that is 145%.`),
        lineVsSigmoid(),
        p(`The orange model is <em>linear regression</em>: chapter 1's w·x + b, with the weight and bias chosen to make the squared misses as small as possible, a recipe called <em>least squares</em>. For predicting a number that rises steadily, such as price from floor area, it is the right first tool. It is solved in one calculation with no training loop, and its weight reads plainly: each extra hour adds this much.`),
        p(`For a yes/no answer it is the wrong shape. A line must keep rising, so it promises 145% to anyone keen enough. Worse, it is punished for a student it got right: the 24-hour pass sits far from the line, the squared miss is large, and the whole line tilts to shrink it. Four students near the boundary paid for that.`),
        p(`<em>Logistic regression</em> keeps the sum and changes the last step. It pushes w·x + b through chapter 11's sigmoid, so the output is a probability between 0 and 1, and it is scored with chapter 2's −log p. A confident correct answer now costs almost nothing, so the keen student pulls on nothing. You have trained one before: chapter 11's reward model was logistic regression on pairs.`),
        p(`It is also one neuron. Chapter 1's neuron said yes when w·x + b cleared zero; this one says how sure it is. Stack them in layers and you are back in chapter 2. Everything else in this chapter took the other road, with cuts and distances where a network has weights.`),
        callout('example', '🌍 The score behind a loan decision',
          `Logistic regression has been the standard tool of credit scoring for decades. In a classic scorecard each answer on the form adds or subtracts points, the points are summed, and the total maps to a probability of repayment.
           It survives largely because every point can be explained to a customer and to a regulator.`),
      ));

      root.append(section('Why this matters for modern AI',
        p(`It is tempting to file this chapter under history. A great deal of the machine learning running inside companies works on tables of rows and columns, and on tables the boosted trees you just built remain the method to beat. A language model drafting the fraud team's reports while a boosted tree scores the payments is an ordinary division of labour.`),
        p(`The same methods sit inside the LLM stack. Retrieval is nearest neighbours over embeddings. Teams run k-means on embeddings to see what a million support tickets are about, or to find near-duplicates in training data. Interpretability researchers fit a logistic regression to a model's internal activations, called a linear probe, to test whether a concept can be read from them.`),
        p(`And the ideas transfer unchanged. A tree's entropy and a language model's loss are built from the same thing: average surprise, −log p. The last layer of a language model choosing its next token is logistic regression widened to many classes, which is chapter 7's softmax, sitting on features that the earlier layers learned. Neural networks kept that final step and learned everything before it.`),
        callout('key', '🔑 Two ways to beat one unreliable model',
          `<b>Average</b> many models that are wrong in different places, and their errors cancel: the forest.<br>
           <b>Chain</b> many weak models so that each corrects what the last left wrong: boosting.<br>
           Both ideas outlive trees: any kind of model can be averaged or chained. Chapter 17 is about when to reach for which tool.`),
      ));

      root.append(
        ctx.quiz([
          { q: 'A tree scores 100% on the 200 payments it grew from and 79% on fresh ones. Regrown on a different 200 payments it draws quite different boxes. Which change goes after the cause?',
            options: ['Grow many such trees on resampled data and let them vote', 'Let the tree grow deeper so it can fix its remaining errors', 'Switch from Gini impurity to entropy when choosing the cuts', 'Report its score on the training payments, where it is reliable'], answer: 0,
            explain: 'A perfect training score, a poor fresh score and boxes that change with the sample are the marks of high variance. Averaging trees that err in different places cancels much of it; that is a random forest. A depth limit would also help. Going deeper is impossible (it is already pure) and Gini versus entropy rarely changes a cut.' },
          { q: 'A boosted model predicts 40 bike hires for 8:00 and the true figure is 70. The next tree outputs exactly the residual for that hour, and the learning rate is 0.5. What does the model predict for 8:00 after adding it?',
            options: ['70', '85', '40', '55'], answer: 3,
            explain: 'The residual is 70 − 40 = 30. Only half of the tree is added: 40 + 0.5 × 30 = 55. The next tree will be aimed at the 15 that remains.' },
          { q: 'A k-NN fraud model stores 200 payments, 94 of them fraud. With k = 1 it scores 100% on those 200 and 78% on fresh payments. To be safe, a colleague sets k to 199. What does the model do now?',
            options: ['Says "genuine" for every payment, whatever its amount or distance', 'Scores about 100% on fresh payments too, since every example now votes', 'Behaves much as it did at k = 1, because the nearest payment still decides', 'Keeps its 78%, because changing k changes the speed and not the verdicts'], answer: 0,
            explain: 'With 199 voters, almost the whole dataset votes on every question, so the commoner class wins everywhere: 105 genuine against 94 fraud. That is underfitting, the opposite failure from k = 1, where each stored payment (mislabelled ones included) rules its own patch. The useful k is in between, and you find it on held-out data.' },
          { q: 'You run k-means twice on the same customers with k = 3. One run settles at a typical distance of 19.6 and the other at 11.3. What happened, and what do you do?',
            options: ['The data must have changed between the runs, so collect it again and rerun', 'The runs began from different centres and one got stuck, so keep the 11.3 run', 'Three centres are too few, so raise k until the two runs agree with each other', 'k-means is random at every step, so average the two sets of centres together'], answer: 1,
            explain: 'k-means always settles, but where it settles depends on where the centres start. The 19.6 run is in a local minimum, like your bad start. The usual practice is several restarts, keeping the lowest distance.' },
          { q: 'A colleague fits a straight line to predict whether a customer cancels (1) or stays (0). It outputs 1.4 for some customers, and adding a few extreme but obvious cancellers changed its verdict on borderline customers. What is the fix?',
            options: ['Clip every output above 1 back down to 1 and keep the fitted line as it is', 'Delete the extreme customers from the data and refit the same straight line', 'Use logistic regression: the same weighted sum, passed through a sigmoid', 'Add more columns to the table so that the straight line is able to bend'], answer: 2,
            explain: 'This is the 24-hour student. Least squares punishes the line for being far from a point it classified correctly, so extreme cases tilt it. The sigmoid keeps outputs between 0 and 1 and charges almost nothing for confident correct answers, so the boundary stays put.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://r2d3.us/visual-intro-to-machine-learning-part-1/" target="_blank" rel="noopener">R2D3, "A Visual Introduction to Machine Learning"</a>: a scrolling, animated decision tree grown on house data. Ten minutes, and the best picture of this chapter's first half.`,
            `<a href="https://scikit-learn.org/stable/modules/tree.html" target="_blank" rel="noopener">scikit-learn user guide, "Decision Trees"</a>: the library most people reach for first, with the exact formulas for Gini and entropy and honest notes on where trees go wrong.`,
            `<a href="https://www.statlearning.com/" target="_blank" rel="noopener">James, Witten, Hastie &amp; Tibshirani, <i>An Introduction to Statistical Learning</i></a>: free to download, written for applied statisticians. Chapters 3 and 4 cover linear and logistic regression; chapter 8 covers trees, bagging, forests and boosting.`,
            `<a href="https://arxiv.org/abs/1603.02754" target="_blank" rel="noopener">Chen &amp; Guestrin (2016), "XGBoost: A Scalable Tree Boosting System"</a>: the paper behind the library. The first two sections are readable with what you now know.`,
            `<a href="https://arxiv.org/abs/2207.08815" target="_blank" rel="noopener">Grinsztajn, Oyallon &amp; Varoquaux (2022), "Why do tree-based models still outperform deep learning on typical tabular data?"</a>: the careful comparison, and a good bridge to chapter 17.`,
          ]),
        ),
      );
    },
  });
})();
