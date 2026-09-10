/* Zero → AGI · Chapter 01 · What does it mean for a machine to learn?
   Perceptron (Rosenblatt 1958), the perceptron learning rule, linear separability and XOR.
   Interactives: neuron calculator with flowing signals; click-to-place perceptron trainer with
   animated weight updates; nested AI ⊃ ML ⊃ DL diagram. Plain JS, no dependencies. */
(function () {
  ZTA.registerChapter({
    id: '01-what-is-learning',
    num: 1,
    part: 'I',
    title: 'What does it mean for a machine to learn?',
    tagline: 'A single artificial neuron from 1958 already contains the whole idea: weights, examples, mistakes, and small nudges.',
    render(root, ctx) {
      const { h, p, section, callout, ul, ol } = ctx;
      const C = ctx.colors;
      const FONT = '13px Inter, system-ui, sans-serif';
      const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
      /* toFixed, but: a value that rounds to zero never prints as "−0.00", non-finite values
         never print as "NaN" in the middle of a diagram, and the minus sign is the real U+2212
         used everywhere else on these canvases. */
      const fix = (v, d) => {
        if (!isFinite(v)) return v > 0 ? '∞' : v < 0 ? '−∞' : '—';
        const s = (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : v).toFixed(d);
        return s.charAt(0) === '-' ? '−' + s.slice(1) : s;
      };
      const f1 = (v) => fix(v, 1);
      const f2 = (v) => fix(v, 2);
      const sgn = (v) => (v < 0 ? '−' : '+');
      const big = (v) => (v >= 100000 ? Math.round(v / 1000) + 'k' : String(v));

      /* ------------------------------------------------------------------ */
      /* Interactive B: the neuron calculator                                 */
      /* ------------------------------------------------------------------ */
      function neuronCalculator() {
        const W = 720, H = 320;
        const [cv, g] = ctx.canvas(W, H);
        const S = { x: [3, 1, 2], w: [0.8, 1.5, 0.4], b: -2 };
        const names = ['currency symbols', 'sender unknown?', 'exclamation marks'];
        /* Every control moves on a 0.5 (inputs) or 0.1 (weights, bias) grid, so the arithmetic on
           screen is exact to two decimals. Snapping to 1e-6 stops binary-float dust from making a
           displayed "0.00" claim to be above zero, which would contradict the printed output. */
        const snap = (v) => Math.round(v * 1e6) / 1e6;
        // x2 answers a yes/no question, so it only takes the values 0 and 1; the other two are counts
        const xRange = [{ max: 8, step: 0.5 }, { max: 1, step: 1 }, { max: 8, step: 0.5 }];
        const xs = S.x.map((v, i) => ctx.slider({ label: 'x' + (i + 1) + ' · ' + names[i], min: 0, max: xRange[i].max, step: xRange[i].step, value: v, fmt: f1, onChange: (val) => { S.x[i] = val; } }));
        const ws = S.w.map((v, i) => ctx.slider({ label: 'w' + (i + 1) + ' (weight)', min: -2, max: 2, step: 0.1, value: v, fmt: f1, onChange: (val) => { S.w[i] = val; } }));
        const bs = ctx.slider({ label: 'b (bias)', min: -5, max: 5, step: 0.1, value: S.b, fmt: f1, onChange: (val) => { S.b = val; } });
        const ro = ctx.readout();
        const inX = 200, inY = [78, 158, 238], nX = 452, nY = 158, R = 44, biasX = 352, biasY = 40;
        const boxX = 528, boxW = 184, boxC = boxX + boxW / 2;
        function total() {
          const prods = S.w.map((w, i) => snap(w * S.x[i]));
          const sum = snap(prods.reduce((a, b) => a + b, 0) + S.b);
          return { prods, sum, out: sum > 0 ? 1 : 0 };
        }

        function node(x, y, r, fill, stroke) {
          g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fillStyle = fill; g.fill();
          g.lineWidth = 2; g.strokeStyle = stroke; g.stroke();
        }
        function edge(x0, y0, x1, y1, val, label, t, labelDy) {
          const mag = Math.abs(val);
          const col = val > 0 ? C.danger : val < 0 ? C.accent : C.muted;
          g.strokeStyle = col; g.globalAlpha = 0.35 + Math.min(0.65, mag * 0.25);
          g.lineWidth = 1 + Math.min(7, mag * 1.2);
          g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
          g.globalAlpha = 1;
          // flowing dots: speed grows with the size of the signal
          if (mag > 0.001) {
            const speed = 0.25 + Math.min(1.2, mag * 0.25);
            for (let k = 0; k < 3; k++) {
              const u = ((t * speed + k / 3) % 1 + 1) % 1;
              const px = x0 + (x1 - x0) * u, py = y0 + (y1 - y0) * u;
              g.beginPath(); g.arc(px, py, 3.5, 0, Math.PI * 2); g.fillStyle = col; g.fill();
            }
          }
          g.font = MONO; g.fillStyle = col; g.textAlign = 'center';
          const mx = x0 + (x1 - x0) * 0.5, my = y0 + (y1 - y0) * 0.5;
          g.fillText(label, mx, my + (labelDy || -9));
        }
        const show = (r) => ({ 'weighted sum + bias': f2(r.sum), output: r.out ? '1 (yes: spam)' : '0 (no: not spam)' });
        let roTimer = 0;
        function draw(dt, t) {
          g.clearRect(0, 0, W, H);
          const r = total(), prods = r.prods, sum = r.sum, out = r.out;
          // inputs
          for (let i = 0; i < 3; i++) {
            edge(inX + 18, inY[i], nX - R, nY, prods[i], f1(S.w[i]) + ' × ' + f1(S.x[i]) + ' = ' + f2(prods[i]), t, i === 2 ? 16 : -10);
          }
          edge(biasX, biasY + 14, nX - 16, nY - R + 6, S.b, 'b = ' + f1(S.b), t, -10);
          for (let i = 0; i < 3; i++) {
            node(inX, inY[i], 18, '#111827', C.text);
            g.fillStyle = C.text; g.font = MONO; g.textAlign = 'center'; g.fillText('x' + (i + 1), inX, inY[i] + 4);
            g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'right'; g.fillText(names[i] + ' = ' + f1(S.x[i]), inX - 26, inY[i] + 4);
          }
          node(biasX, biasY, 14, '#111827', C.muted);
          g.fillStyle = C.text; g.font = MONO; g.textAlign = 'center'; g.fillText('1', biasX, biasY + 4);
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'right'; g.fillText('bias input (always 1)', biasX - 24, biasY + 4);
          // the neuron
          node(nX, nY, R, '#0f1520', out ? C.green : C.line);
          g.fillStyle = C.text; g.font = 'bold 16px Inter, system-ui, sans-serif'; g.textAlign = 'center';
          g.fillText('Σ', nX, nY - 8);
          g.font = MONO; g.fillStyle = out ? C.danger : C.accent; g.fillText(f2(sum), nX, nY + 14);
          // threshold → output
          g.strokeStyle = C.muted; g.lineWidth = 2; g.beginPath(); g.moveTo(nX + R, nY); g.lineTo(boxX - 4, nY); g.stroke();
          g.beginPath(); g.moveTo(boxX - 4, nY); g.lineTo(boxX - 13, nY - 5); g.lineTo(boxX - 13, nY + 5); g.closePath(); g.fillStyle = C.muted; g.fill();
          g.fillStyle = '#111827'; g.strokeStyle = out ? C.green : C.line; g.lineWidth = 2;
          g.beginPath();
          if (g.roundRect) g.roundRect(boxX, nY - 34, boxW, 68, 10); else g.rect(boxX, nY - 34, boxW, 68);
          g.fill(); g.stroke();
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'center'; g.fillText('is the total > 0 ?', boxC, nY - 12);
          g.font = 'bold 15px Inter, system-ui, sans-serif'; g.fillStyle = out ? C.green : C.muted;
          g.fillText(out ? 'output 1 → SPAM' : 'output 0 → not spam', boxC, nY + 14);
          // formula line
          g.font = MONO; g.fillStyle = C.text; g.textAlign = 'center';
          const terms = S.w.map((w, i) => f1(w) + '×' + f1(S.x[i])).join(' + ');
          g.fillText(terms + ' ' + sgn(S.b) + ' ' + f1(Math.abs(S.b)) + ' = ' + f2(sum) + (out ? '  (above 0 → 1)' : '  (not above 0 → 0)'), W / 2, H - 14);
          roTimer += dt;
          if (roTimer > 0.15) { roTimer = 0; ro.set(show(r)); }
        }
        ro.set(show(total()));
        ctx.loop(draw);
        return ctx.figure(cv, 'A perceptron computing by hand. Edge thickness and dot speed show the size of each signal <b>weight × input</b>; red pushes toward "yes", blue toward "no".', [...xs, ...ws, bs], ro);
      }

      /* ------------------------------------------------------------------ */
      /* Interactive A: perceptron trainer                                    */
      /* ------------------------------------------------------------------ */
      function perceptronTrainer() {
        const W = 720, H = 400;
        const [cv, g] = ctx.canvas(W, H);
        cv.style.touchAction = 'none'; cv.style.cursor = 'crosshair';
        const plot = { x: 30, y: 20, w: 360, h: 360 };
        const RANGE = 1.2;
        const toPx = (x, y) => ({ x: plot.x + (x + RANGE) / (2 * RANGE) * plot.w, y: plot.y + (RANGE - y) / (2 * RANGE) * plot.h });
        const fromPx = (px, py) => ({ x: (px - plot.x) / plot.w * 2 * RANGE - RANGE, y: RANGE - (py - plot.y) / plot.h * 2 * RANGE });
        const S = { pts: [], w1: 0, w2: 0, b: 0, i: 0, errs: 0, epoch: 0, hist: [], training: false, converged: false, placing: 1, lr: 0.1, speed: 8, acc: 0, flash: 0, flashIdx: -1, msg: 'Press ▶ Train to start.', updates: 0 };

        function shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
        function resetWeights() {
          // reject a degenerate starting w: with ‖w‖ ≈ 0 the boundary line is numerically undefined
          S.w1 = 0; S.w2 = 0;
          for (let k = 0; k < 20 && S.w1 * S.w1 + S.w2 * S.w2 < 1e-4; k++) { S.w1 = ctx.rand(-0.5, 0.5); S.w2 = ctx.rand(-0.5, 0.5); }
          if (S.w1 * S.w1 + S.w2 * S.w2 < 1e-4) { S.w1 = 0.3; S.w2 = -0.2; }
          S.b = 0;
          S.i = 0; S.errs = 0; S.epoch = 0; S.hist = []; S.converged = false; S.updates = 0;
          S.flash = 0; S.flashIdx = -1; S.acc = 0;
          S.msg = 'Weights reset to small random values. Press ▶ Train.';
        }
        /* On non-separable data the rule never stops updating, so the weights random-walk forever.
           They cannot reach infinity at any realistic speed, but guard anyway: the perceptron's
           decision is sign(w·x + b), so dividing w1, w2 and b by the same positive number leaves
           every prediction and the drawn line exactly unchanged — it only keeps the printout legible. */
        function guardWeights() {
          if (!isFinite(S.w1) || !isFinite(S.w2) || !isFinite(S.b)) {
            S.training = false; resetWeights();
            S.msg = 'The numbers overflowed, so the weights were reset. Try a smaller learning rate.';
            return;
          }
          const n = Math.hypot(S.w1, S.w2);
          if (n > 1e3) { const k = 1e3 / n; S.w1 *= k; S.w2 *= k; S.b *= k; }
        }
        function loadPreset(name) {
          S.pts = [];
          if (name === 'separable') {
            for (let k = 0; k < 12; k++) {
              S.pts.push({ x: 0.45 + ctx.rand(-0.3, 0.3), y: 0.4 + ctx.rand(-0.3, 0.3), c: 1 });
              S.pts.push({ x: -0.45 + ctx.rand(-0.3, 0.3), y: -0.4 + ctx.rand(-0.3, 0.3), c: -1 });
            }
          } else if (name === 'xor') {
            const cs = [[0.55, 0.55, 1], [-0.55, -0.55, 1], [0.55, -0.55, -1], [-0.55, 0.55, -1]];
            for (const [cx, cy, c] of cs) for (let k = 0; k < 6; k++) S.pts.push({ x: cx + ctx.rand(-0.22, 0.22), y: cy + ctx.rand(-0.22, 0.22), c });
          }
          shuffle(S.pts);
          S.training = false; resetWeights(); updateBtns();
        }
        function predict(q) { return (S.w1 * q.x + S.w2 * q.y + S.b) > 0 ? 1 : -1; }
        function step() {
          if (!S.pts.length) { S.msg = 'Click on the plot to place some points first.'; S.training = false; return; }
          if (S.i >= S.pts.length) S.i = 0;
          const q = S.pts[S.i], pred = predict(q);
          if (pred !== q.c) {
            S.w1 += S.lr * q.c * q.x; S.w2 += S.lr * q.c * q.y; S.b += S.lr * q.c;
            guardWeights();
            S.errs++; S.updates++; S.flash = 1; S.flashIdx = S.i;
            S.msg = 'Point ' + (S.i + 1) + ': predicted ' + (pred > 0 ? 'red' : 'blue') + ', truth ' + (q.c > 0 ? 'red' : 'blue') + ' → nudge w ' + (q.c > 0 ? 'toward' : 'away from') + ' it (η = ' + f2(S.lr) + ').';
          } else {
            S.msg = 'Point ' + (S.i + 1) + ': correct. No change.';
          }
          S.i++;
          if (S.i >= S.pts.length) {
            S.epoch++; S.hist.push(S.errs); if (S.hist.length > 60) S.hist.shift();
            if (S.errs === 0) { S.converged = true; S.training = false; S.msg = 'Pass ' + S.epoch + ': zero mistakes. Converged — every point is on the right side.'; updateBtns(); }
            S.errs = 0; S.i = 0;
          }
        }
        // Sutherland–Hodgman clip of polygon by half-plane sign·(a·x + b·y + c) ≥ 0
        function clipHalf(poly, a, b, c, sign) {
          const out = [];
          for (let i = 0; i < poly.length; i++) {
            const P = poly[i], Q = poly[(i + 1) % poly.length];
            const fp = sign * (a * P.x + b * P.y + c), fq = sign * (a * Q.x + b * Q.y + c);
            if (fp >= 0) out.push(P);
            if ((fp >= 0) !== (fq >= 0)) { const t = fp / (fp - fq); out.push({ x: P.x + (Q.x - P.x) * t, y: P.y + (Q.y - P.y) * t }); }
          }
          return out;
        }
        function fillPoly(poly, color) {
          if (poly.length < 3) return;
          g.beginPath();
          poly.forEach((q, i) => { const s = toPx(q.x, q.y); if (i === 0) g.moveTo(s.x, s.y); else g.lineTo(s.x, s.y); });
          g.closePath(); g.fillStyle = color; g.fill();
        }
        /* Word-wrap with a hard line budget. Without the budget a long status message could run
           down through the legend and the sparkline underneath it. */
        function wrap(text, x, y, maxW, lh, maxLines) {
          const words = String(text).split(' '), lines = [];
          let line = '';
          for (const w of words) {
            const test = line ? line + ' ' + w : w;
            if (g.measureText(test).width > maxW && line) { lines.push(line); line = w; } else line = test;
          }
          if (line) lines.push(line);
          if (maxLines && lines.length > maxLines) {
            lines.length = maxLines;
            lines[maxLines - 1] = lines[maxLines - 1].replace(/\s*\S*$/, '') + '…';
          }
          lines.forEach((L, i) => g.fillText(L, x, y + i * lh));
          return y + lines.length * lh;
        }
        function draw(t) {
          g.clearRect(0, 0, W, H);
          // plot background + grid
          g.fillStyle = '#0f1520'; g.fillRect(plot.x, plot.y, plot.w, plot.h);
          g.strokeStyle = C.line; g.lineWidth = 1;
          for (let v = -1; v <= 1; v += 0.5) {
            const a = toPx(v, 0), b = toPx(0, v);
            g.globalAlpha = v === 0 ? 0.9 : 0.35;
            g.beginPath(); g.moveTo(a.x, plot.y); g.lineTo(a.x, plot.y + plot.h); g.stroke();
            g.beginPath(); g.moveTo(plot.x, b.y); g.lineTo(plot.x + plot.w, b.y); g.stroke();
          }
          g.globalAlpha = 1;
          const sq = [{ x: -RANGE, y: -RANGE }, { x: RANGE, y: -RANGE }, { x: RANGE, y: RANGE }, { x: -RANGE, y: RANGE }];
          const hasLine = S.w1 * S.w1 + S.w2 * S.w2 > 1e-6;
          if (hasLine) {
            fillPoly(clipHalf(sq, S.w1, S.w2, S.b, 1), 'rgba(251,113,133,0.13)');
            fillPoly(clipHalf(sq, S.w1, S.w2, S.b, -1), 'rgba(124,156,255,0.13)');
            /* The boundary line w1·x1 + w2·x2 + b = 0, found by intersecting it with the two edges
               of the visible square that it must cross. Always dividing by the LARGER of |w1|, |w2|
               keeps this well conditioned — the old form divided by ‖w‖², which blows up to
               astronomical pixel coordinates when the weights are small. */
            const seg = [];
            if (Math.abs(S.w2) >= Math.abs(S.w1)) {
              for (const x1 of [-RANGE, RANGE]) seg.push({ x: x1, y: -(S.w1 * x1 + S.b) / S.w2 });
            } else {
              for (const x2 of [-RANGE, RANGE]) seg.push({ x: -(S.w2 * x2 + S.b) / S.w1, y: x2 });
            }
            if (seg.every(q => isFinite(q.x) && isFinite(q.y) && Math.abs(q.x) < 1e4 && Math.abs(q.y) < 1e4)) {
              const a = toPx(seg[0].x, seg[0].y), b = toPx(seg[1].x, seg[1].y);
              g.save(); g.beginPath(); g.rect(plot.x, plot.y, plot.w, plot.h); g.clip();
              g.strokeStyle = C.text; g.lineWidth = 2.5; g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
              g.restore();
            }
          }
          // points
          let wrong = 0;
          S.pts.forEach((q, k) => {
            const s = toPx(q.x, q.y);
            const mis = hasLine ? predict(q) !== q.c : true;
            if (mis) wrong++;
            g.beginPath(); g.arc(s.x, s.y, 7, 0, Math.PI * 2); g.fillStyle = q.c > 0 ? C.danger : C.accent; g.fill();
            g.lineWidth = 1.5; g.strokeStyle = '#0a0e16'; g.stroke();
            if (mis) { g.beginPath(); g.arc(s.x, s.y, 11, 0, Math.PI * 2); g.strokeStyle = C.warn; g.lineWidth = 2; g.stroke(); }
            if (k === S.i && (S.training || S.updates > 0) && !S.converged) {
              g.beginPath(); g.arc(s.x, s.y, 14 + 2 * Math.sin(t * 6), 0, Math.PI * 2); g.strokeStyle = C.text; g.lineWidth = 1.5; g.globalAlpha = 0.8; g.stroke(); g.globalAlpha = 1;
            }
            if (k === S.flashIdx && S.flash > 0) {
              g.beginPath(); g.arc(s.x, s.y, 12 + (1 - S.flash) * 22, 0, Math.PI * 2); g.strokeStyle = C.warn; g.lineWidth = 3; g.globalAlpha = S.flash; g.stroke(); g.globalAlpha = 1;
            }
          });
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(plot.x, plot.y, plot.w, plot.h);
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          g.fillText('x1 →', plot.x + plot.w - 34, plot.y + plot.h - 6); g.fillText('x2 ↑', plot.x + 4, plot.y + 14);
          if (!S.pts.length) { g.fillStyle = C.text; g.textAlign = 'center'; g.fillText('Click here to place points', plot.x + plot.w / 2, plot.y + plot.h / 2); }

          // right panel. Every block below has a fixed budget so nothing can overrun the next one.
          const rx = 410, rw = 300;
          g.textAlign = 'left'; g.font = MONO; g.fillStyle = C.muted;
          g.fillText('on a mistake:  w ← w + η·y·x   (y = ±1)', rx, 34);
          g.fillStyle = C.text;
          g.fillText('w1 = ' + f2(S.w1) + '   w2 = ' + f2(S.w2) + '   b = ' + f2(S.b), rx, 56);
          g.fillStyle = C.muted;
          g.fillText('line: ' + f2(S.w1) + '·x1 ' + sgn(S.w2) + ' ' + f2(Math.abs(S.w2)) + '·x2 ' + sgn(S.b) + ' ' + f2(Math.abs(S.b)) + ' = 0', rx, 76);
          // `big` keeps this line inside the panel even after half an hour of continuous training
          g.fillText('pass ' + big(S.epoch) + ' · ' + big(S.updates) + ' updates · ' + wrong + '/' + S.pts.length + ' wrong', rx, 96);
          g.font = FONT; g.fillStyle = S.converged ? C.green : C.text;
          const y = wrap(S.msg, rx, 120, rw - 6, 18, 3);
          if (!S.converged && S.epoch >= 15 && S.hist.length >= 12 && S.hist.slice(-12).every(e => e > 0)) {
            g.fillStyle = C.warn; wrap('Not converging — the mistake count just bounces. No straight line can split these points.', rx, y + 6, rw - 6, 18, 2);
          }
          // legend
          const ly = 216;
          g.font = FONT;
          g.beginPath(); g.arc(rx + 6, ly, 6, 0, Math.PI * 2); g.fillStyle = C.danger; g.fill(); g.fillStyle = C.muted; g.fillText('class +1 ("yes")', rx + 18, ly + 4);
          g.beginPath(); g.arc(rx + 136, ly, 6, 0, Math.PI * 2); g.fillStyle = C.accent; g.fill(); g.fillStyle = C.muted; g.fillText('class −1 ("no")', rx + 148, ly + 4);
          g.beginPath(); g.arc(rx + 6, ly + 22, 8, 0, Math.PI * 2); g.strokeStyle = C.warn; g.lineWidth = 2; g.stroke(); g.fillStyle = C.muted; g.fillText('currently misclassified', rx + 18, ly + 26);
          g.beginPath(); g.arc(rx + 6, ly + 44, 8, 0, Math.PI * 2); g.strokeStyle = C.text; g.lineWidth = 1.5; g.stroke(); g.fillText('next example to check', rx + 18, ly + 48);
          // mistakes-per-pass sparkline
          const bx = rx, by = 276, bw = rw, bh = 100;
          g.fillStyle = '#0f1520'; g.fillRect(bx, by, bw, bh);
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(bx, by, bw, bh);
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          g.fillText(S.hist.length ? 'mistakes per pass (last ' + S.hist.length + ')' : 'mistakes per pass — no full pass yet', bx + 6, by + 14);
          if (S.hist.length) {
            const mx = Math.max(1, ...S.hist);
            const bwid = (bw - 12) / Math.max(20, S.hist.length);
            S.hist.forEach((e, k) => {
              // a perfect pass is the point of the whole demo, so give zero a visible green stub
              const hh = e === 0 ? 3 : (bh - 26) * e / mx;
              g.fillStyle = e === 0 ? C.green : C.warn;
              g.fillRect(bx + 6 + k * bwid, by + bh - 5 - hh, Math.max(1.5, bwid - 1), hh);
            });
            g.fillStyle = C.muted; g.textAlign = 'right'; g.fillText('max ' + mx, bx + bw - 6, by + 14); g.textAlign = 'left';
          }
        }
        const ro = ctx.readout();   // declared before the loop that reads it, not after
        let roTimer = 0;
        ctx.loop((dt, t) => {
          // at most 50 examples per frame, so the main thread is never blocked
          if (S.training) { S.acc += dt * S.speed; let guard = 0; while (S.acc >= 1 && S.training && guard++ < 50) { S.acc -= 1; step(); } }
          if (!S.training) S.acc = 0;
          S.flash = Math.max(0, S.flash - dt * 1.8);
          draw(t);
          roTimer += dt;
          if (roTimer > 0.12) {
            roTimer = 0;
            ro.set({ w1: f2(S.w1), w2: f2(S.w2), b: f2(S.b), pass: S.epoch, 'mistakes (last pass)': S.hist.length ? S.hist[S.hist.length - 1] : '–', status: S.converged ? 'converged ✓' : S.training ? 'training…' : 'paused' });
          }
        });
        cv.addEventListener('pointerdown', (ev) => {
          ev.preventDefault();
          const m = cv.pos(ev);
          if (m.x < plot.x || m.x > plot.x + plot.w || m.y < plot.y || m.y > plot.y + plot.h) return;
          for (let k = 0; k < S.pts.length; k++) {
            const s = toPx(S.pts[k].x, S.pts[k].y);
            if (Math.hypot(s.x - m.x, s.y - m.y) < 10) { S.pts.splice(k, 1); S.i = 0; S.errs = 0; S.converged = false; S.flashIdx = -1; S.msg = 'Point removed.'; return; }
          }
          if (S.pts.length >= 120) { S.msg = '120 points is plenty — click a point to remove one first.'; return; }
          const d = fromPx(m.x, m.y);
          S.pts.push({ x: d.x, y: d.y, c: S.placing });
          S.converged = false; S.msg = 'Added a ' + (S.placing > 0 ? 'red' : 'blue') + ' point. Press ▶ Train (click it again to remove it).';
        });

        const presetSel = ctx.select({ label: 'preset', options: [{ value: 'separable', label: 'Linearly separable' }, { value: 'xor', label: 'XOR (not separable)' }, { value: 'empty', label: 'Empty — draw your own' }], value: 'separable', onChange: loadPreset });
        const placeBtn = ctx.button('', () => { S.placing = -S.placing; updateBtns(); });
        const trainBtn = ctx.button('', () => { if (!S.pts.length) { S.msg = 'Place some points first.'; return; } S.training = !S.training; if (S.training) S.converged = false; updateBtns(); }, 'primary');
        const stepBtn = ctx.button('Step one example', () => { S.training = false; step(); updateBtns(); });
        const resetBtn = ctx.button('Reset weights', () => { S.training = false; resetWeights(); updateBtns(); });
        const clearBtn = ctx.button('Clear points', () => { S.pts = []; S.training = false; resetWeights(); S.msg = 'Cleared. Click on the plot to place points.'; updateBtns(); });
        const lrSl = ctx.slider({ label: 'learning rate η', min: 0.02, max: 1, step: 0.02, value: 0.1, fmt: f2, onChange: (v) => { S.lr = v; } });
        const spSl = ctx.slider({ label: 'examples per second', min: 1, max: 120, step: 1, value: 8, onChange: (v) => { S.speed = v; } });
        function updateBtns() {
          placeBtn.textContent = 'Placing: ' + (S.placing > 0 ? '● red (+1)' : '● blue (−1)');
          placeBtn.style.color = S.placing > 0 ? C.danger : C.accent;
          trainBtn.textContent = S.training ? '⏸ Pause' : '▶ Train perceptron';
        }
        loadPreset('separable');
        return ctx.figure(cv, 'The perceptron learning rule, one example at a time. The white line is <code class="inline">w1·x1 + w2·x2 + b = 0</code>; the red side predicts +1, the blue side −1. Yellow rings mark points currently on the wrong side. This plot labels the two classes +1 and −1, so the update rule reads <code class="inline">w ← w + η·y·x</code> on a mistake — the same algorithm as the 0/1 version above.', [presetSel, trainBtn, stepBtn, placeBtn, resetBtn, clearBtn, lrSl, spSl], ro);
      }

      /* ------------------------------------------------------------------ */
      /* Static diagram: AI ⊃ ML ⊃ DL ⊃ LLMs                                  */
      /* ------------------------------------------------------------------ */
      function nestedCircles() {
        const W = 720, H = 320;
        const [cv, g] = ctx.canvas(W, H);
        const rings = [
          { r: 145, cx: 200, cy: 160, color: C.muted, name: 'Artificial intelligence', year: 'term coined 1956', sub: 'Any technique that makes a machine do something that looks intelligent, including hand-written rules: expert systems, classic chess engines, route planners.' },
          { r: 105, cx: 225, cy: 175, color: C.accent, name: 'Machine learning', year: 'term coined 1959', sub: 'Programs whose behaviour is fitted to data instead of written by hand: the perceptron, credit scorecards, spam filters.' },
          { r: 68, cx: 250, cy: 190, color: C.green, name: 'Deep learning', year: 'took off 2012', sub: 'Machine learning with many-layered neural networks that learn their own features: image recognition, speech, translation.' },
          { r: 34, cx: 270, cy: 200, color: C.warn, name: 'Large language models', year: '2018 →', sub: 'Deep networks trained to predict the next word on most of the internet: Claude, GPT, Gemini.' },
        ];
        g.clearRect(0, 0, W, H);
        rings.forEach((r) => {
          g.beginPath(); g.arc(r.cx, r.cy, r.r, 0, Math.PI * 2);
          g.fillStyle = r.color; g.globalAlpha = 0.10; g.fill(); g.globalAlpha = 1;
          g.strokeStyle = r.color; g.lineWidth = 2; g.stroke();
        });
        const labelX = 395;
        rings.forEach((r, i) => {
          const ly = 34 + i * 72;
          // leader line from label to the top-right rim of the circle
          const ang = -Math.PI / 4 + i * 0.12;
          const ex = r.cx + Math.cos(ang) * r.r, ey = r.cy + Math.sin(ang) * r.r;
          g.strokeStyle = r.color; g.lineWidth = 1; g.globalAlpha = 0.7;
          g.beginPath(); g.moveTo(labelX - 8, ly + 4); g.lineTo(ex, ey); g.stroke(); g.globalAlpha = 1;
          g.beginPath(); g.arc(ex, ey, 3.5, 0, Math.PI * 2); g.fillStyle = r.color; g.fill();
          g.textAlign = 'left'; g.fillStyle = r.color; g.font = 'bold 14px Inter, system-ui, sans-serif';
          g.fillText(r.name, labelX, ly + 8);
          // measure the name in the font it was actually drawn in, before switching to MONO
          const nameW = g.measureText(r.name).width;
          g.fillStyle = C.muted; g.font = MONO; g.fillText(r.year, labelX + nameW + 12, ly + 8);
          g.font = '12px Inter, system-ui, sans-serif'; g.fillStyle = C.text; g.globalAlpha = 0.85;
          const words = r.sub.split(' '); let line = '', yy = ly + 26;
          for (const w of words) { const t = line ? line + ' ' + w : w; if (g.measureText(t).width > 310 && line) { g.fillText(line, labelX, yy); yy += 15; line = w; } else line = t; }
          if (line) g.fillText(line, labelX, yy);
          g.globalAlpha = 1;
        });
        g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'center';
        g.fillText('each circle sits inside the previous one', 200, H - 6);
        return ctx.figure(cv, 'The three terms the news uses interchangeably. Claude is an AI, built with machine learning, of the deep-learning kind.');
      }

      /* ================================================================== */
      /* Prose                                                              */
      /* ================================================================== */
      root.append(
        p(`Imagine it is 1998 and your boss asks you to build a spam filter. You start writing rules. If the subject line contains "FREE", block it. If the whole message is in capitals, block it. If it mentions a lottery you never entered, block it. Two weeks later the spammers write "F R E E", and your rulebook is out of date. You add rules. They adapt. You are in an arms race you will lose, because every rule is a fact you had to know in advance and write down by hand.`),
        p(`Now think about how a two-year-old learns the word "dog". Nobody hands the child a rulebook ("four legs, fur, barks; but not a cat, and not a wolf"). The child sees a labrador and hears "dog". She points at a cat and says "dog" and is gently corrected. She points at a poodle and is praised. After a few dozen examples and a few dozen mistakes she can recognise a dog she has never seen, of a breed she has never seen, from an angle she has never seen. The rules were never written down. They were <b>learned from examples and mistakes</b>.`),
        p(`That shift is what this chapter is about. Instead of writing the rules, we write a machine that <em>finds</em> the rules from labelled examples. So what, mechanically, does that machine look like? What does it actually do when it "learns"? The surprising answer is that the first such machine, built in 1958, already contains almost every idea we still use, including inside the model you are reading this on.`),

        section('Two ways to make a computer do something',
          p(`There are exactly two ways to get a computer to tell spam from real mail. The first is to write down the rules yourself. The second is to write a program that <em>finds</em> the rules from examples. The first is ordinary programming. The second is <em>machine learning</em>. Everything in this course, up to and including the model you chat with, is the second kind.`),
          ctx.table(['', 'Hand-written rules', 'Learned from examples'], [
            ['Who writes the logic', 'A human expert', 'The algorithm, from data'],
            ['What you need', 'Someone who can articulate every rule', 'Lots of examples with the right answer attached'],
            ['When the world changes', 'A human rewrites the rules', 'Retrain on fresh examples'],
            ['Where it fails', 'Rules too many or too subtle to write down (faces, speech, "is this a dog?")', 'Data that is scarce, biased, or mislabelled'],
          ]),
          p(`Notice what learning from examples requires: a supply of examples that come with the right answer attached (this email is spam, that one is not), and a procedure that turns the machine's mistakes into adjustments. Those two things, labelled examples and a way to adjust from mistakes, are the whole of it. Let's build the smallest possible machine that has both.`),
        ),

        section('The perceptron: a neuron you can compute on a napkin',
          p(`In 1958 Frank Rosenblatt, a psychologist at Cornell, built a machine he called the <em>perceptron</em>. It was inspired by a cartoon version of a brain cell: a neuron receives signals from many neighbours, some exciting it and some inhibiting it, and fires if the total excitement crosses a threshold. Rosenblatt's version is arithmetic you can do by hand:`),
          ol([
            `Describe the thing you are looking at with a few numbers: x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>.`,
            `Multiply each number by a <em>weight</em>: w<sub>1</sub>, w<sub>2</sub>, w<sub>3</sub>. A big positive weight means "this clue strongly suggests yes"; a negative weight means "this clue suggests no".`,
            `Add the products up, then add one more number, the <em>bias</em> b, which sets how easily the neuron says yes when there is no evidence either way.`,
            `If the total is above zero, output 1 ("yes"). Otherwise output 0 ("no").`,
          ]),
          ctx.code('output = 1  if  (w1·x1 + w2·x2 + w3·x3 + b) > 0\n         0  otherwise'),
          p(`That is the whole machine. Let's run it on an email. Describe every email with three numbers: how many currency symbols it contains, whether the sender is unknown (1) or in your address book (0), and how many exclamation marks it has. An email arrives with three dollar signs, from a stranger, with two exclamation marks: x = (3, 1, 2).`),
          p(`Say the weights are w = (0.8, 1.5, 0.4) and the bias is b = −2. The weighted sum is 0.8×3 + 1.5×1 + 0.4×2 = 2.4 + 1.5 + 0.8 = 4.7. Add the bias: 4.7 − 2 = 2.7. That is above zero, so the perceptron outputs 1: spam. Now a note from a friend with one exclamation mark, x = (0, 0, 1). The sum is 0 + 0 + 0.4 − 2 = −1.6. Below zero: not spam.`),
          p(`Read the weights as a statement of what matters. Being from a stranger counts most (1.5), dollar signs count a lot (0.8), exclamation marks count a little (0.4), and the bias of −2 says: unless the evidence adds up to more than 2, assume the email is fine. Every decision this machine makes is completely explained by those four numbers.`),
          callout('tryit', 'Try it: run the neuron yourself', `Drag the sliders. Watch the weighted sum inside the neuron and the flowing signals on each wire. <b>Try:</b> (1) Set x<sub>1</sub> and x<sub>2</sub> to 0, so the only evidence is exclamation marks from someone you know. Work out on paper how many it takes to trip the filter — 0.4 × x<sub>3</sub> has to beat the bias of 2 — then drag x<sub>3</sub> and check you were right. (2) Push w<sub>2</sub> negative: now a stranger counts <i>against</i> spam, and the wire turns blue. (3) Set all three inputs to 0 and move only the bias: with no evidence at all, the bias alone decides. <b>Notice:</b> the neuron never sees the email, only the three numbers someone chose to measure.`),
          neuronCalculator(),
        ),

        section('The words people use',
          p(`You have now seen every object the field talks about, so here are their names. Learn these six and most AI news becomes readable.`),
          ul([
            `<b>Model</b>: the machine that turns inputs into a prediction. Here it is the perceptron: a formula with adjustable knobs.`,
            `<b>Parameters</b>, also called <b>weights</b>: the knobs. w<sub>1</sub>, w<sub>2</sub>, w<sub>3</sub> and b. "Learning" means setting them. This model has four; a modern language model has hundreds of billions.`,
            `<b>Features</b>: the numbers you describe an input with, x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>. Someone has to decide what to measure. For decades this was the hard, human part of the job; deep learning's big trick, which you will meet in chapter 4, is learning the features too.`,
            `<b>Label</b>: the right answer attached to a training example. Spam or not spam. Dog or not dog.`,
            `<b>Prediction</b>: the model's output for an input. The perceptron's 1 or 0.`,
            `<b>Training data</b>: the pile of (features, label) pairs the model learns from. Its size and quality decide almost everything about how good the model gets.`,
          ]),
        ),

        section('How the perceptron learns',
          p(`Rosenblatt's real contribution was not the neuron. Weighted sums with a threshold had been described by Warren McCulloch and Walter Pitts in 1943. What was new was a procedure for setting the weights <em>automatically</em>, from examples, plus a proof that it works.`),
          p(`The procedure is almost embarrassingly simple. Show the perceptron one training example. If it gets the label right, do nothing. If it gets it wrong, nudge the weights a little: if it said no but the answer was yes, add a small multiple of the input to the weights, so that this input scores higher next time. If it said yes but the answer was no, subtract a small multiple of the input. Nudge the bias the same way. Move to the next example. Keep cycling through all the examples until you get through a whole pass without a single mistake.`),
          ctx.code('repeat until a whole pass makes no mistakes:\n    for each training example (x, y):    # y is the true label, 1 or 0\n        ŷ = 1 if (w·x + b) > 0 else 0    # ŷ ("y-hat") is the prediction\n        w ← w + η · (y − ŷ) · x          # η ("eta") is the learning rate, e.g. 0.1\n        b ← b + η · (y − ŷ)'),
          p(`Look at the term (y − ŷ). If the prediction is right it is 0 and nothing moves. If the answer was 1 and we said 0, it is +1 and the weights move <i>toward</i> the input. If the answer was 0 and we said 1, it is −1 and they move <i>away</i>. The <em>learning rate</em> η controls how big each nudge is. This is the ancestor of every training algorithm in use today: <b>compare the prediction to the truth, and move the parameters a little in the direction that would have made the mistake smaller.</b>`),
          p(`One notational wrinkle, because you will meet both versions and they look different. Written with labels 0 and 1, the update is η·(y − ŷ)·x, as above. Most textbooks instead call the two classes +1 and −1, and then the rule collapses to something even shorter: when you are right, do nothing; when you are wrong, w ← w + η·y·x. That is why the interactive below prints a shorter rule than the code you just read. It is the same algorithm wearing different labels.`),
          p(`Let's run it on our two emails, starting from all-zero weights with η = 0.1. The spam email (3, 1, 2) scores 0, which is not above zero, so we predict 0. Wrong. The weights become (0.3, 0.1, 0.2) and the bias 0.1. The friend's email (0, 0, 1) now scores 0.2 + 0.1 = 0.3: we predict spam. Wrong the other way. The weights become (0.3, 0.1, 0.1) and the bias 0. Second pass: the spam email scores 1.2, correct; the friend's email scores 0.1, still wrongly spam, so the weights become (0.3, 0.1, 0.0) and the bias −0.1. Third pass: 0.9 and −0.1. Both correct. Three mistakes, and the machine has written its own spam rule.`),
          callout('key', 'The convergence guarantee, and its catch', `In 1962 Albert Novikoff proved that if <em>any</em> setting of the weights can separate the examples perfectly, this procedure will find one after a finite number of mistakes, no matter where it starts. That "if" is the catch. The interactive below lets you feel exactly where it bites.`),
          callout('tryit', 'Try it: watch a perceptron learn', `<b>1.</b> Load the <b>Linearly separable</b> preset and press ▶ Train. The white ring shows which example is being checked; when it is on the wrong side it flashes yellow and the line jumps. Watch the mistakes-per-pass bars fall to a flat green zero: <em>converged</em>. <b>2.</b> Add your own points by clicking (toggle the colour with the "Placing" button, click a point again to delete it). Drop a red point deep in blue territory and train again — watch one stubborn example drag the whole line. <b>3.</b> Load <b>XOR</b>, push <i>examples per second</i> up to 120 and press ▶ Train. The mistake count never reaches zero; it bounces forever, because the line fixes one corner and breaks another. <b>4.</b> Now drag η from 0.02 to 1.0. The jumps get bigger but the story does not change — and if you started the weights from exactly zero it would not change <i>at all</i>, because the perceptron's answer depends only on the direction of w, never on its length.`),
          perceptronTrainer(),
        ),

        section('The wall: what one neuron cannot learn',
          p(`Look at what the perceptron actually draws. With two features, the weighted sum w<sub>1</sub>·x<sub>1</sub> + w<sub>2</sub>·x<sub>2</sub> + b = 0 is the equation of a straight line; the perceptron says yes on one side and no on the other. Learning just moves the line around. With three features it is a flat plane; with a thousand it is a "hyperplane", but it is always flat. A dataset that some straight line can split perfectly is called <em>linearly separable</em>, and that is exactly the class of problems a single neuron can learn.`),
          p(`Now consider the simplest problem that isn't: XOR, "exclusive or". Two inputs, and the answer is yes when exactly one of them is on. Yes at (0, 1) and (1, 0); no at (0, 0) and (1, 1). Draw those four points and try to separate the yeses from the noes with one straight line. You cannot; the yes points sit on opposite corners. The XOR preset above shows the learning rule thrashing forever: every fix breaks something else.`),
          p(`That sounds like a toy. It isn't. Almost every interesting question ("is this a face?", "is this sentence sarcastic?") has an XOR-shaped structure somewhere inside it: a clue that means yes in one context and no in another. A single straight cut cannot express "it depends".`),
          callout('history', '1969: Minsky, Papert, and the first AI winter', `Rosenblatt's perceptron made headlines in 1958; the <i>New York Times</i> reported the Navy's expectation that it would soon "walk, talk, see, write, reproduce itself and be conscious of its existence". In 1969 Marvin Minsky and Seymour Papert published <i>Perceptrons</i>, a careful mathematical book showing what a single-layer perceptron cannot compute, with XOR as the famous example. They noted that stacking layers might help, but nobody knew how to train stacked layers. Funding for neural-network research collapsed for over a decade, a period now called the first <em>AI winter</em>. The fix, back-propagation through multiple layers, is the subject of the next chapter. It took until 1986 to become widely known.`),
        ),

        section('Where single neurons still earn their keep',
          callout('example', 'Credit scoring', `When a bank decides whether to lend to you, a common tool is a <em>scorecard</em>: points for income, points for years at your address, points off for a recent missed payment, add them up, approve above a threshold. That is a perceptron with statistically fitted weights. The modern version, <em>logistic regression</em>, is the same weighted sum with a smooth curve instead of a hard threshold so it outputs a probability instead of a yes/no. It is still the most widely deployed machine-learning model in finance, partly because a regulator can read the weights and see why you were refused.`),
          callout('example', 'Spam filters', `The spam filters of the early 2000s, popularised by Paul Graham's 2002 essay "A Plan for Spam", counted words and combined the evidence with learned weights. Word counts as features, a flat boundary as the model, retraining as spammers adapted. Gmail's early filter worked this way; today's uses deep networks, but the features-weights-threshold skeleton is the same.`),
          callout('example', 'Early optical character recognition', `Rosenblatt's physical Mark I Perceptron (1960) used a 20×20 grid of photocells as its 400 features and learned to tell letters apart. Reading handwritten digits on bank cheques reliably needed multi-layer networks, which arrived in the 1990s (chapter 4); by the late 1990s such systems were reading a sizeable share of all cheques in the United States.`),
        ),

        section('AI, machine learning, deep learning: nested, not synonyms',
          p(`One more piece of vocabulary, because these three terms are used interchangeably in the news and they should not be. <b>Artificial intelligence</b> is the broad goal: machines doing things that would need intelligence if a person did them. A 1980s chess program with hand-written evaluation rules is AI without any learning. <b>Machine learning</b> is the subset where the behaviour is fitted to data rather than written by hand: the perceptron, credit scorecards, spam filters. <b>Deep learning</b> is the subset of machine learning that uses neural networks with many layers stacked on top of each other, so that the features themselves are learned rather than chosen by a person. Large language models are deep learning.`),
          nestedCircles(),
        ),

        section('Why this matters for modern AI',
          p(`It is tempting to think a chatbot has nothing to do with a four-weight spam detector. It has everything to do with it. Inside a large language model, the basic operation, repeated billions of times for every word it produces, is exactly the one you just did on a napkin: multiply inputs by weights, add them up, add a bias, pass the result through a squashing function. The features are no longer counted by hand; they are learned. The weights are not four; they are around 10<sup>11</sup> to 10<sup>12</sup>. The learning rule is not Rosenblatt's nudge but its smooth descendant, <em>gradient descent</em>, which we build in the next chapter. But "model", "parameters", "training data", "label" and "prediction" mean precisely what they meant on this page, and every headline about AI can be translated into those five words.`),
          p(`Keep one picture from this chapter: <b>learning is the act of turning mistakes into small adjustments of numbers.</b> Everything else is scale.`),
        ),

        ctx.quiz([
          { q: 'A perceptron has weights (2, −1) and bias −1. The input is (1, 2). What does it output?', options: ['1 ("yes"): the sum is positive', '0 ("no"): the sum is 2·1 + (−1)·2 − 1 = −1', 'It depends on the learning rate', 'It cannot decide without training data'], answer: 1, explain: 'Weighted sum: 2×1 + (−1)×2 = 0, plus the bias −1 gives −1. Not above zero, so the output is 0. The learning rate only matters while training, never when predicting.' },
          { q: 'Which of these is a <i>parameter</i> of the spam perceptron?', options: ['The number of training emails', 'The weight on "sender unknown"', 'The label "spam"', 'The number of exclamation marks in an email'], answer: 1, explain: 'Parameters are the numbers the model adjusts while learning: the weights and bias. The exclamation-mark count is a feature (an input); "spam" is a label; the dataset size is neither.' },
          { q: 'Why does the perceptron never converge on XOR?', options: ['There are too few training examples', 'The learning rate is too high', 'No single straight line separates the two classes', 'The bias starts at zero'], answer: 2, explain: 'A single neuron always draws a flat boundary. XOR puts the "yes" points on opposite corners, so no straight line can separate them, and the rule keeps fixing one mistake by creating another.' },
          { q: 'In the perceptron learning rule, what happens to the weights when the prediction is already correct?', options: ['They move toward the input', 'They move away from the input', 'Nothing changes', 'They are reset to zero'], answer: 2, explain: 'The update is η·(y − ŷ)·x. When the prediction matches the label, (y − ŷ) = 0, so the weights are untouched. Learning happens only on mistakes.' },
          { q: 'Which statement about the three nested terms is right?', options: ['"Deep learning" and "AI" mean the same thing', 'Deep learning is a subset of machine learning that uses many-layered neural networks', 'Machine learning is any program that contains if-statements', 'AI is a subset of machine learning'], answer: 1, explain: 'AI is the broad goal; machine learning is the subset that fits behaviour to data; deep learning is the subset of that which uses deep neural networks. Large language models sit in the innermost circle.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://www.youtube.com/watch?v=aircAruvnKk" target="_blank" rel="noopener">3Blue1Brown, "But what is a neural network?"</a>: the best 20-minute visual introduction; it starts exactly where this chapter ends.`,
            `<a href="https://karpathy.ai/zero-to-hero.html" target="_blank" rel="noopener">Andrej Karpathy, "Neural Networks: Zero to Hero"</a>: build everything in this course in Python, from a single gradient to a small GPT.`,
            `<a href="https://playground.tensorflow.org" target="_blank" rel="noopener">TensorFlow Playground</a>: a bigger cousin of the trainer above; try the XOR dataset with zero hidden layers and then with one.`,
            `<a href="https://doi.org/10.1037/h0042519" target="_blank" rel="noopener">Rosenblatt (1958), "The perceptron: a probabilistic model for information storage and organization in the brain"</a>: the original paper, in <i>Psychological Review</i>.`,
            `<a href="https://mitpress.mit.edu/9780262630221/perceptrons/" target="_blank" rel="noopener">Minsky &amp; Papert (1969), <i>Perceptrons</i></a>: the book that ended the first neural-network boom; the 1988 edition has a candid new preface.`,
          ]),
        ),
      );
    },
  });
})();
