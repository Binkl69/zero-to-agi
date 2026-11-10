/* Zero → AGI · Chapter 03 · The learning recipe
   DESIGN RULE: the reader overfits a curve with their own hands before the word "overfitting"
   appears, then spends a validation set and watches it lie to them. Every paragraph explains
   something they already did.
   Interactives, in order: polynomial curve-fit (train/held-out U-curve, the opener); three-pile
   split + the winner's curse from tuning against validation; animated early-stopping curve;
   optimizer race on a Rosenbrock valley; mini-batch gradient noise vs GPU cost; a free-label
   machine turning any sentence into self-supervised training pairs; a spam dial showing accuracy
   lying while precision and recall trade off. */
(function () {
  ZTA.registerChapter({
    id: '03-the-learning-recipe',
    num: 3,
    part: 'I',
    title: 'The recipe: data, loss, optimizer, generalization',
    tagline: 'Build an answer key with your own hands, watch it fail the real exam, then learn the discipline that stops every model doing the same thing.',
    minutes: 50,
    objectives: [
      'spot overfitting and underfitting from a training score and a held-out score',
      'explain why data is cut into three piles, and why a validation score you tuned against is inflated',
      'predict how much noise an average carries from the number of examples behind it (σ ÷ √n)',
      'read a confusion matrix and say when accuracy is hiding a useless model',
    ],
    takeaways: [
      '<b>The gap is the number that matters.</b> A model is judged by the distance between its training score and its held-out score, never by the training score alone.',
      '<b>Too rigid or too twitchy.</b> Underfitting misses the signal; overfitting fits the noise. Held-out error traces a U between them, and weight decay, dropout, early stopping and more data all push against the right-hand side.',
      '<b>Three piles.</b> Train on one, make choices against validation, score once on test. Picking the best of many validation scores picks the luckiest, so that score is spent.',
      '<b>Averages are noisy by σ ÷ √n.</b> Four times the examples halves the noise. It explains the fan of mini-batch gradients and the luck in a small validation set.',
      '<b>Momentum remembers, Adam rescales.</b> Momentum keeps most of its last step so one noisy batch barely dents it; Adam gives every weight a step size from its own gradient history. AdamW is the default.',
      '<b>Accuracy lies on unbalanced data.</b> Precision asks how much of what was flagged was right; recall asks how much of what was there got caught. Moving the threshold trades one for the other.',
    ],
    render(root, ctx) {
      const { h, p, section, sub, callout, ul, ol, cards } = ctx;
      const C = ctx.colors;
      const FONT = '13px Inter, system-ui, sans-serif';
      const MONO = '12px "JetBrains Mono", ui-monospace, monospace';

      /* ================================================================== */
      /* Interactive A: polynomial curve-fitting — overfitting vs underfitting */
      /* ================================================================== */
      function curveFit() {
        const W = 720, H = 300;
        const [cv, g] = ctx.canvas(W, H);
        const [ec, eg] = ctx.canvas(720, 150);
        const plot = { x: 46, y: 12, w: 630, h: 230 };
        const YLO = -2.0, YHI = 2.0;
        const trueF = (x) => Math.sin(2 * Math.PI * x * 1.5) * 0.55;
        const NOISE = 0.18, RIDGE = 1e-10, MAXD = 15;
        /* Legendre basis: same space of polynomials, vastly better conditioned. */
        function basis(u, d1) {
          const row = [1];
          if (d1 > 1) row.push(u);
          for (let k = 2; k < d1; k++) row.push(((2 * k - 1) * u * row[k - 1] - (k - 1) * row[k - 2]) / k);
          return row;
        }
        const toPx = (x, y) => ({ x: plot.x + x * plot.w, y: plot.y + (YHI - ctx.clamp(y, YLO, YHI)) / (YHI - YLO) * plot.h });

        function solve(A, b) {
          const n = b.length;
          const M = A.map((r) => r.slice()), y = b.slice();
          for (let c = 0; c < n; c++) {
            let piv = c, best = Math.abs(M[c][c]);
            for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > best) { best = Math.abs(M[r][c]); piv = r; }
            if (piv !== c) { const t = M[c]; M[c] = M[piv]; M[piv] = t; const ty = y[c]; y[c] = y[piv]; y[piv] = ty; }
            const d = M[c][c];
            if (Math.abs(d) < 1e-12) continue;
            for (let r = c + 1; r < n; r++) { const f = M[r][c] / d; if (!f) continue; for (let k = c; k < n; k++) M[r][k] -= f * M[c][k]; y[r] -= f * y[c]; }
          }
          const x = new Array(n).fill(0);
          for (let r = n - 1; r >= 0; r--) { let s = y[r]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = Math.abs(M[r][r]) < 1e-12 ? 0 : s / M[r][r]; }
          return x;
        }
        function fitDeg(idx, deg) {
          const d1 = deg + 1;
          const A = Array.from({ length: d1 }, () => new Array(d1).fill(0));
          const b = new Array(d1).fill(0);
          for (const i of idx) {
            const u = (S.xs[i] - 0.5) * 2;
            const row = basis(u, d1);
            for (let r = 0; r < d1; r++) { b[r] += row[r] * S.ys[i]; for (let c = 0; c < d1; c++) A[r][c] += row[r] * row[c]; }
          }
          for (let r = 0; r < d1; r++) A[r][r] += RIDGE;
          return solve(A, b);
        }
        function evalW(w, x) { const row = basis((x - 0.5) * 2, w.length); let s = 0; for (let k = 0; k < w.length; k++) s += w[k] * row[k]; return s; }
        function mse(w, idx) { let s = 0; for (const i of idx) { const e = evalW(w, S.xs[i]) - S.ys[i]; s += Math.min(16, e * e); } return s / Math.max(1, idx.length); }

        const S = { n: 40, deg: 3, xs: [], ys: [], train: [], held: [], w: [], curveTrain: [], curveVal: [], dirty: true };
        function recompute() {
          S.w = fitDeg(S.train, S.deg);
          S.curveTrain = []; S.curveVal = [];
          for (let d = 1; d <= MAXD; d++) { const w = fitDeg(S.train, d); S.curveTrain.push(mse(w, S.train)); S.curveVal.push(mse(w, S.held)); }
          S.dirty = true;
        }
        function makeData() {
          const xs = [], ys = [];
          for (let i = 0; i < S.n; i++) { const x = ctx.rand(0, 1); xs.push(x); ys.push(trueF(x) + ctx.randn() * NOISE); }
          S.xs = xs; S.ys = ys;
          const order = xs.map((_, i) => i);
          for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t; }
          const nTrain = Math.max(2, Math.round(order.length * 0.6));
          S.train = order.slice(0, nTrain); S.held = order.slice(nTrain);
          if (!S.held.length) S.held = [S.train.pop()];
          recompute();
        }
        makeData();

        function draw() {
          g.clearRect(0, 0, W, H);
          g.fillStyle = '#0f1520'; g.fillRect(plot.x, plot.y, plot.w, plot.h); g.strokeStyle = C.line; g.strokeRect(plot.x, plot.y, plot.w, plot.h);
          g.save(); g.beginPath(); g.rect(plot.x, plot.y, plot.w, plot.h); g.clip();
          g.strokeStyle = C.muted; g.setLineDash([4, 3]); g.lineWidth = 1.5; g.beginPath();
          for (let i = 0; i <= 200; i++) { const x = i / 200, s = toPx(x, trueF(x)); if (i === 0) g.moveTo(s.x, s.y); else g.lineTo(s.x, s.y); }
          g.stroke(); g.setLineDash([]);
          g.strokeStyle = C.green; g.lineWidth = 2.5; g.beginPath();
          for (let i = 0; i <= 240; i++) { const x = i / 240, yy = ctx.clamp(evalW(S.w, x), YLO - 6, YHI + 6), s = toPx(x, yy); if (i === 0) g.moveTo(s.x, s.y); else g.lineTo(s.x, s.y); }
          g.stroke();
          g.restore();
          for (const i of S.train) { const s = toPx(S.xs[i], S.ys[i]); g.beginPath(); g.arc(s.x, s.y, 4, 0, Math.PI * 2); g.fillStyle = C.accent; g.fill(); }
          for (const i of S.held) { const s = toPx(S.xs[i], S.ys[i]); g.beginPath(); g.arc(s.x, s.y, 4.4, 0, Math.PI * 2); g.fillStyle = C.orange; g.fill(); g.lineWidth = 1.2; g.strokeStyle = '#0a0e16'; g.stroke(); }
          /* two rows: the whole key on one line is wider than the canvas */
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          g.fillText('blue = training points   orange = held-out points', plot.x, H - 24);
          g.fillText('grey dashed = true hidden sine   green = fitted degree-' + S.deg + ' polynomial', plot.x, H - 6);
        }
        function drawErr() {
          const LW = 720, LH = 150, px = 44, py = 12, pw = 660, ph = 98;
          eg.clearRect(0, 0, LW, LH);
          eg.fillStyle = '#0f1520'; eg.fillRect(px, py, pw, ph); eg.strokeStyle = C.line; eg.strokeRect(px, py, pw, ph);
          const maxE = Math.max(0.25, ...S.curveTrain, ...S.curveVal);
          const xAt = (d) => px + (d - 1) / (MAXD - 1) * pw, yAt = (e) => py + ph - Math.min(1, e / maxE) * ph;
          eg.strokeStyle = C.accent; eg.lineWidth = 2; eg.beginPath();
          S.curveTrain.forEach((e, i) => { const x = xAt(i + 1), y = yAt(e); if (!i) eg.moveTo(x, y); else eg.lineTo(x, y); }); eg.stroke();
          eg.strokeStyle = C.danger; eg.lineWidth = 2; eg.beginPath();
          S.curveVal.forEach((e, i) => { const x = xAt(i + 1), y = yAt(e); if (!i) eg.moveTo(x, y); else eg.lineTo(x, y); }); eg.stroke();
          const mx = xAt(S.deg);
          eg.strokeStyle = C.warn; eg.setLineDash([4, 3]); eg.lineWidth = 1.3; eg.beginPath(); eg.moveTo(mx, py); eg.lineTo(mx, py + ph); eg.stroke(); eg.setLineDash([]);
          let best = 0; for (let i = 1; i < S.curveVal.length; i++) if (S.curveVal[i] < S.curveVal[best]) best = i;
          const bp = { x: xAt(best + 1), y: yAt(S.curveVal[best]) };
          eg.beginPath(); eg.arc(bp.x, bp.y, 5, 0, Math.PI * 2); eg.fillStyle = C.green; eg.fill();
          eg.font = MONO; eg.fillStyle = C.muted; eg.textAlign = 'left'; eg.fillText('degree 1 →→→ ' + MAXD, px, py + ph + 14);
          eg.font = FONT; eg.fillStyle = C.accent; eg.textAlign = 'left'; eg.fillText('● training error (always falls)', px + 150, py + ph + 14);
          eg.fillStyle = C.danger; eg.fillText('● held-out error (U-shaped)', px + 400, py + ph + 14);
          /* held off the right-hand border so the degree marker line cannot sit on it */
          eg.fillStyle = C.green; eg.textAlign = 'right'; eg.fillText('best degree = ' + (best + 1), px + pw - 10, 24);
        }
        const ro = ctx.readout();
        function updateRO() { ro.set({ degree: S.deg, points: S.n, 'train error': S.curveTrain[S.deg - 1] != null ? S.curveTrain[S.deg - 1].toFixed(3) : '–', 'held-out error': S.curveVal[S.deg - 1] != null ? S.curveVal[S.deg - 1].toFixed(3) : '–' }); }
        ctx.loop(() => { if (S.dirty) { draw(); drawErr(); updateRO(); S.dirty = false; } });
        const degSl = ctx.slider({ label: 'polynomial degree', min: 1, max: MAXD, step: 1, value: 3, onChange: (v) => { S.deg = v; recompute(); } });
        const nSl = ctx.slider({ label: 'number of points', min: 8, max: 50, step: 1, value: 40, onChange: (v) => { S.n = v; makeData(); } });
        const resampleBtn = ctx.button('New noise sample', () => makeData());
        const body = h('div', {}, cv, ec);
        return ctx.figure(body, 'Least-squares polynomial fit (normal equations, on a Legendre basis, with a nominal ridge term for numerical stability) to noisy samples of a hidden sine. Below: training error (blue) always falls as degree rises; held-out error (red) falls then rises — the U-shaped signature of the bias–variance trade-off.', [degSl, nSl, resampleBtn], ro);
      }

      /* ================================================================== */
      /* Interactive B: SGD vs momentum vs Adam on a curved valley           */
      /* ================================================================== */
      function optimizerRace() {
        const W = 720, H = 380;
        const [cv, g] = ctx.canvas(W, H);
        const GX0 = -2.2, GX1 = 2.2, GY0 = -1.5, GY1 = 4.2;
        const plot = { x: 10, y: 10, w: 480, h: 360 };
        const GRID = 90;
        const off = document.createElement('canvas'); off.width = GRID; off.height = GRID;
        const og = off.getContext('2d'); const img = og.createImageData(GRID, GRID);
        const BB = 5;
        const lossAt = (x, y) => (1 - x) * (1 - x) + BB * (y - x * x) * (y - x * x);
        const gradAt = (x, y) => [-2 * (1 - x) - 4 * BB * x * (y - x * x), 2 * BB * (y - x * x)];
        let maxLog = 1;
        {
          const vals = [];
          for (let gy = 0; gy < GRID; gy++) for (let gx = 0; gx < GRID; gx++) {
            const x = GX0 + (gx + 0.5) / GRID * (GX1 - GX0), y = GY1 - (gy + 0.5) / GRID * (GY1 - GY0);
            vals.push(Math.log(1 + lossAt(x, y)));
          }
          maxLog = Math.max(1, ...vals);
          const px = img.data; let k = 0;
          for (const v of vals) {
            const t = ctx.clamp(v / maxLog, 0, 1);
            const r = Math.round(12 + t * 235), gg = Math.round(16 + (1 - Math.abs(t - 0.55) * 1.4) * 55), b = Math.round(34 + (1 - t) * 55);
            px[k++] = ctx.clamp(r, 0, 255); px[k++] = ctx.clamp(gg, 0, 255); px[k++] = ctx.clamp(b, 0, 255); px[k++] = 255;
          }
          og.putImageData(img, 0, 0);
        }
        const toPx = (x, y) => ({ x: plot.x + (x - GX0) / (GX1 - GX0) * plot.w, y: plot.y + (GY1 - y) / (GY1 - GY0) * plot.h });
        const START = { x: -1.5, y: 2 };
        const OPT = [
          { name: 'SGD', color: C.muted, lr: 0.0035 },
          { name: 'Momentum', color: C.warn, lr: 0.0035, beta: 0.9 },
          { name: 'Adam', color: C.green, lr: 0.09, b1: 0.9, b2: 0.999, eps: 1e-8 },
        ];
        function fresh() { return { x: START.x, y: START.y, vx: 0, vy: 0, mx: 0, my: 0, vx2: 0, vy2: 0, t: 0, trail: [], step: 0 }; }
        let S = OPT.map(fresh);
        let playing = false, dirty = true;
        function stepOne(o, s) {
          let [gx, gy] = gradAt(s.x, s.y);
          gx = ctx.clamp(gx, -60, 60); gy = ctx.clamp(gy, -60, 60);
          if (o.name === 'SGD') { s.x -= o.lr * gx; s.y -= o.lr * gy; }
          else if (o.name === 'Momentum') { s.vx = o.beta * s.vx - o.lr * gx; s.vy = o.beta * s.vy - o.lr * gy; s.x += s.vx; s.y += s.vy; }
          else {
            s.t++;
            s.mx = o.b1 * s.mx + (1 - o.b1) * gx; s.my = o.b1 * s.my + (1 - o.b1) * gy;
            s.vx2 = o.b2 * s.vx2 + (1 - o.b2) * gx * gx; s.vy2 = o.b2 * s.vy2 + (1 - o.b2) * gy * gy;
            const mhx = s.mx / (1 - Math.pow(o.b1, s.t)), mhy = s.my / (1 - Math.pow(o.b1, s.t));
            const vhx = s.vx2 / (1 - Math.pow(o.b2, s.t)), vhy = s.vy2 / (1 - Math.pow(o.b2, s.t));
            s.x -= o.lr * mhx / (Math.sqrt(vhx) + o.eps); s.y -= o.lr * mhy / (Math.sqrt(vhy) + o.eps);
          }
          s.x = ctx.clamp(s.x, GX0 - 0.2, GX1 + 0.2); s.y = ctx.clamp(s.y, GY0 - 0.2, GY1 + 0.2);
          if (!isFinite(s.x) || !isFinite(s.y)) Object.assign(s, fresh());
          s.step++;
          /* Record movement, not time. A converged optimizer standing on the
             minimum used to keep pushing identical points and scroll its own
             descent path out of the buffer, so momentum's overshoot and Adam's
             glide vanished a few seconds after they arrived. */
          const last = s.trail[s.trail.length - 1];
          if (!last || Math.hypot(s.x - last.x, s.y - last.y) > 0.002) {
            s.trail.push({ x: s.x, y: s.y }); if (s.trail.length > 1500) s.trail.shift();
          }
        }
        function draw() {
          g.clearRect(0, 0, W, H);
          g.drawImage(off, plot.x, plot.y, plot.w, plot.h);
          g.strokeStyle = C.line; g.strokeRect(plot.x, plot.y, plot.w, plot.h);
          const mn = toPx(1, 1);
          g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(mn.x - 8, mn.y); g.lineTo(mn.x + 8, mn.y); g.moveTo(mn.x, mn.y - 8); g.lineTo(mn.x, mn.y + 8); g.stroke();
          g.fillStyle = '#fff'; g.font = FONT; g.textAlign = 'left'; g.fillText('minimum', mn.x + 10, mn.y - 8);
          /* The state is allowed 0.2 units past the plot range so a genuine
             overshoot stays visible as it leaves; clipping keeps the trail and
             its 7px marker from spilling into the legend beside the plot. */
          g.save(); g.beginPath(); g.rect(plot.x, plot.y, plot.w, plot.h); g.clip();
          OPT.forEach((o, oi) => {
            const s = S[oi];
            g.strokeStyle = o.color; g.lineWidth = 1.6; g.globalAlpha = 0.7; g.beginPath();
            s.trail.forEach((pt, i) => { const q = toPx(pt.x, pt.y); if (!i) g.moveTo(q.x, q.y); else g.lineTo(q.x, q.y); }); g.stroke(); g.globalAlpha = 1;
            const q = toPx(s.x, s.y);
            g.beginPath(); g.arc(q.x, q.y, 7, 0, Math.PI * 2); g.fillStyle = o.color; g.fill(); g.strokeStyle = '#0a0e16'; g.lineWidth = 1.5; g.stroke();
          });
          g.restore();
          const lx = plot.x + plot.w + 20;
          g.textAlign = 'left'; g.fillStyle = C.text; g.font = 'bold 13px Inter, system-ui, sans-serif'; g.fillText('same start, same valley', lx, 24);
          OPT.forEach((o, oi) => {
            const s = S[oi], y = 58 + oi * 96;
            g.beginPath(); g.arc(lx + 6, y, 6, 0, Math.PI * 2); g.fillStyle = o.color; g.fill();
            g.font = 'bold 13px Inter, system-ui, sans-serif'; g.fillStyle = C.text; g.fillText(o.name, lx + 20, y + 4);
            g.font = MONO; g.fillStyle = C.muted;
            g.fillText('loss  ' + lossAt(s.x, s.y).toFixed(3), lx, y + 22);
            g.fillText('steps ' + s.step, lx, y + 40);
            g.fillText('lr    ' + o.lr.toPrecision(2), lx, y + 58);
          });
        }
        ctx.loop(() => {
          if (playing) { for (let k = 0; k < 4; k++) OPT.forEach((o, oi) => stepOne(o, S[oi])); dirty = true; }
          if (dirty) { draw(); dirty = false; }
        });
        function reset() { S = OPT.map(fresh); dirty = true; }
        const sgdSl = ctx.slider({ label: 'SGD learning rate', min: 0.0005, max: 0.06, step: 0.0005, value: 0.0035, digits: 4, onChange: (v) => { OPT[0].lr = v; } });
        const momSl = ctx.slider({ label: 'Momentum learning rate', min: 0.0005, max: 0.08, step: 0.0005, value: 0.0035, digits: 4, onChange: (v) => { OPT[1].lr = v; } });
        const adamSl = ctx.slider({ label: 'Adam learning rate', min: 0.01, max: 0.4, step: 0.01, value: 0.09, digits: 2, onChange: (v) => { OPT[2].lr = v; } });
        const playBtn = ctx.button('▶ Start', () => { playing = !playing; playBtn.textContent = playing ? '⏸ Pause' : '▶ Start'; }, 'primary');
        const resetBtn = ctx.button('Reset', () => { reset(); playing = false; playBtn.textContent = '▶ Start'; });
        return ctx.figure(cv, 'Three optimizers descending the same curved valley (a tamed Rosenbrock surface; white cross marks the true minimum). SGD (grey) fights the curvature at every step and needs about 850 of them to reach the floor; momentum (yellow) builds speed down the slope and gets there in about 85, wobbling where it first turns into the valley; Adam (green) rescales every direction by its own gradient history and arrives in about 65. Adam\'s path is not straighter than SGD\'s — look closely and it ripples more near the cross. What it buys is stride: it does this on a learning rate twenty-five times SGD\'s, and tolerates a far wider range of them.', [sgdSl, momSl, adamSl, playBtn, resetBtn]);
      }

      /* ================================================================== */
      /* Interactive C: early stopping — training vs validation loss         */
      /* ================================================================== */
      function earlyStopping() {
        const W = 720, H = 300;
        const [cv, g] = ctx.canvas(W, H);
        const N = 60;
        const plot = { x: 46, y: 16, w: 640, h: 220 };
        const YMAX = 2.0;
        const S = { sizeIdx: 2, reg: 0.3, train: [], val: [], best: 0, epoch: 0, playing: true, speed: 7 };
        function gen() {
          const sizeFactor = [1.6, 1.0, 0.55, 0.3][S.sizeIdx - 1];
          const regFactor = 1 - S.reg * 0.85;
          const t0 = 8 + S.reg * 10 + (S.sizeIdx - 1) * 4;
          const k = 0.9 * sizeFactor * regFactor;
          const train = [], val = [];
          for (let t = 0; t < N; t++) {
            const shared = 0.10 + 0.9 * Math.exp(-t / 8);
            train.push(Math.max(0.02, shared + 0.015 * ctx.randn()));
            const over = k * Math.pow(Math.max(0, t - t0), 1.4) / 60;
            val.push(Math.max(0.02, shared + over + 0.02 * ctx.randn()));
          }
          let best = 0; for (let t = 1; t < N; t++) if (val[t] < val[best]) best = t;
          S.train = train; S.val = val; S.best = best; S.epoch = 0;
        }
        gen();
        const toPx = (e, v) => ({ x: plot.x + e / (N - 1) * plot.w, y: plot.y + (1 - Math.min(1, v / YMAX)) * plot.h });
        function draw() {
          g.clearRect(0, 0, W, H);
          g.fillStyle = '#0f1520'; g.fillRect(plot.x, plot.y, plot.w, plot.h); g.strokeStyle = C.line; g.strokeRect(plot.x, plot.y, plot.w, plot.h);
          const shown = Math.max(1, Math.min(N, Math.floor(S.epoch) + 1));
          if (shown > S.best) {
            const bx = toPx(S.best, 0).x;
            g.fillStyle = 'rgba(251,113,133,0.08)'; g.fillRect(bx, plot.y, plot.x + plot.w - bx, plot.h);
          }
          g.strokeStyle = C.accent; g.lineWidth = 2.2; g.beginPath();
          for (let e = 0; e < shown; e++) { const p = toPx(e, S.train[e]); if (!e) g.moveTo(p.x, p.y); else g.lineTo(p.x, p.y); } g.stroke();
          g.strokeStyle = C.danger; g.lineWidth = 2.2; g.beginPath();
          for (let e = 0; e < shown; e++) { const p = toPx(e, S.val[e]); if (!e) g.moveTo(p.x, p.y); else g.lineTo(p.x, p.y); } g.stroke();
          if (shown > S.best) {
            const p = toPx(S.best, S.val[S.best]);
            /* start the marker line below the caption band so it does not run
               straight through the "best epoch N" label */
            g.strokeStyle = C.green; g.setLineDash([4, 3]); g.lineWidth = 1.3; g.beginPath(); g.moveTo(p.x, plot.y + 24); g.lineTo(p.x, plot.y + plot.h); g.stroke(); g.setLineDash([]);
            g.beginPath(); g.arc(p.x, p.y, 5.5, 0, Math.PI * 2); g.fillStyle = C.green; g.fill();
            g.fillStyle = C.green; g.font = FONT; g.textAlign = 'center'; g.fillText('best epoch ' + S.best + ' — stop here', ctx.clamp(p.x, plot.x + 74, plot.x + plot.w - 92), plot.y + 14);
          }
          g.font = MONO; g.textAlign = 'left';
          g.fillStyle = C.accent; g.fillText('● training loss', plot.x, plot.y + plot.h + 16);
          g.fillStyle = C.danger; g.fillText('● validation loss', plot.x + 140, plot.y + plot.h + 16);
          g.fillStyle = C.text; g.textAlign = 'right'; g.fillText('epoch ' + Math.min(N - 1, Math.floor(S.epoch)) + ' / ' + (N - 1), plot.x + plot.w, plot.y + plot.h + 16);
        }
        ctx.loop((dt) => { if (S.playing) { S.epoch += dt * S.speed; if (S.epoch > N - 1) S.epoch = N - 1; } draw(); });
        const sizeSel = ctx.select({ label: 'training set size', options: [{ value: '1', label: '200 examples' }, { value: '2', label: '1,000 examples' }, { value: '3', label: '5,000 examples' }, { value: '4', label: '20,000 examples' }], value: '2', onChange: (v) => { S.sizeIdx = +v; gen(); } });
        const regSl = ctx.slider({ label: 'regularization strength', min: 0, max: 1, step: 0.1, value: 0.3, onChange: (v) => { S.reg = v; gen(); } });
        const playBtn = ctx.button('⏸ Pause', () => { S.playing = !S.playing; playBtn.textContent = S.playing ? '⏸ Pause' : '▶ Play'; }, 'primary');
        const replayBtn = ctx.button('↺ Replay', () => { gen(); S.playing = true; playBtn.textContent = '⏸ Pause'; });
        return ctx.figure(cv, 'A simulated training run. Training loss (blue) keeps falling — the model can always get a little better at scoring its own homework. Validation loss (red) falls, then turns upward once the model starts fitting noise instead of signal. The green dot is the epoch with the lowest validation loss: where early stopping would have saved the model.', [sizeSel, regSl, playBtn, replayBtn]);
      }

      /* wrap `text` into lines that fit inside maxW pixels of the given 2-D context */
      function wrapLines(gc, text, maxW) {
        const words = String(text).split(' '); const out = []; let line = '';
        for (const w of words) {
          const t = line ? line + ' ' + w : w;
          if (line && gc.measureText(t).width > maxW) { out.push(line); line = w; } else line = t;
        }
        if (line) out.push(line);
        return out;
      }
      function wrapText(gc, text, x, y, maxW, lh) {
        wrapLines(gc, text, maxW).forEach((ln, i) => gc.fillText(ln, x, y + i * lh));
      }

      /* small deterministic RNG so datasets are stable across frames */
      function rng(seed) {
        let a = seed >>> 0;
        return () => { a += 0x6D2B79F5; let t = a; t = Math.imul(t ^ (t >>> 15), 1 | t); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      }
      function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

      /* ================================================================== */
      /* Interactive D: three piles, and how tuning quietly burns one       */
      /* ================================================================== */
      function splitDeck() {
        const [cv, g] = ctx.canvas(720, 400);
        const TOTAL = 1000;
        let nTrain = 700, nVal = 150;
        const nTest = () => TOTAL - nTrain - nVal;
        /* each candidate hyperparameter setting has a hidden true quality;
           validation and test each see it through their own sampling noise */
        let tried = [];   // {trueQ, valObs, testObs}
        let best = [];    // running best-on-validation after each try
        /* Mark n examples, each right with probability q, and report the fraction
           right. A real score is a count out of n, so it can never pass 100% and
           its luck shrinks as 1/sqrt(n) without anyone coding that in. */
        function score(n, q) { let k = 0; for (let i = 0; i < n; i++) if (Math.random() < q) k++; return k / Math.max(1, n); }
        function reset() { tried = []; best = []; }
        function tryOne() {
          const trueQ = 0.72 + 0.14 * Math.random();
          tried.push({ trueQ, valObs: score(nVal, trueQ), testObs: score(nTest(), trueQ) });
          let bi = 0;
          for (let i = 1; i < tried.length; i++) if (tried[i].valObs > tried[bi].valObs) bi = i;
          best.push(tried[bi]);
        }

        const trSl = ctx.slider({ label: 'training examples', min: 400, max: 900, step: 10, value: 700, onChange: (v) => { nTrain = v; if (nTrain + nVal > TOTAL - 50) { nVal = TOTAL - 50 - nTrain; vaSl.value = nVal; } reset(); } });
        const vaSl = ctx.slider({ label: 'validation examples', min: 25, max: 400, step: 5, value: 150, onChange: (v) => { nVal = v; if (nTrain + nVal > TOTAL - 50) { nTrain = TOTAL - 50 - nVal; trSl.value = nTrain; } reset(); } });
        const oneBtn = ctx.button('Tune once', () => tryOne());
        const manyBtn = ctx.button('Tune 50 times', () => { for (let i = 0; i < 50; i++) tryOne(); }, 'primary');
        const resetBtn = ctx.button('Start over', reset);
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          /* ---- the three piles ---- */
          const BX = 40, BW = 640, BY = 30, BH = 34;
          const piles = [
            ['training', nTrain, C.accent, 'the optimizer sees this'],
            ['validation', nVal, C.warn, 'you tune against this'],
            ['test', nTest(), C.green, 'touched once, at the very end'],
          ];
          /* A row of labels, placed left to right, each sliding leftwards only as
             far as it must to stay on the canvas and off its neighbour. */
          function packRow(items) {
            let limit = 720 - 8;
            for (let i = items.length - 1; i >= 0; i--) {
              items[i].x = Math.max(BX, Math.min(items[i].x, limit - items[i].w));
              limit = items[i].x - 12;
            }
          }
          /* pass 1: the bars themselves, so no later bar can paint over a label */
          let x = BX;
          const laid = piles.map(([name, n, col, note]) => {
            const w = n / TOTAL * BW;
            g.fillStyle = col; g.globalAlpha = 0.75; g.fillRect(x, BY, w - 2, BH); g.globalAlpha = 1;
            const seat = { label: name + '  ' + n, col, note, x, w };
            x += w;
            return seat;
          });
          /* pass 2: name and count inside the bar when they fit there, and in the
             pile's own colour just above it when the bar is too narrow to hold them */
          g.font = 'bold ' + FONT;
          const above = [];
          laid.forEach(s => {
            const tw = g.measureText(s.label).width;
            if (tw + 16 <= s.w) { g.fillStyle = '#0a0e16'; g.fillText(s.label, s.x + 8, BY + 22); }
            else above.push({ text: s.label, col: s.col, x: s.x, w: tw });
          });
          packRow(above);
          above.forEach(a => { g.fillStyle = a.col; g.fillText(a.text, a.x, BY - 8); });
          /* pass 3: the one-line note under each pile wide enough to caption */
          g.font = MONO; g.fillStyle = C.muted;
          const notes = laid.filter(s => s.w > 110).map(s => ({ text: s.note, x: s.x + 8, w: g.measureText(s.note).width }));
          packRow(notes);
          notes.forEach(nt => g.fillText(nt.text, nt.x, BY + BH + 16));

          /* ---- the tuning chart ---- */
          const P = { x: 60, y: 110, w: 430, h: 230 };
          const MAXT = Math.max(60, best.length);
          const px = (i) => P.x + i / MAXT * P.w;
          const py = (q) => P.y + P.h - (q - 0.66) / 0.36 * P.h;
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
          g.font = MONO; g.fillStyle = C.muted;
          for (const q of [0.70, 0.80, 0.90, 1.00]) {
            g.beginPath(); g.moveTo(P.x, py(q)); g.lineTo(P.x + P.w, py(q)); g.stroke();
            g.fillText((q * 100).toFixed(0) + '%', P.x - 34, py(q) + 4);
          }
          g.fillText('hyperparameter settings tried', P.x + 110, P.y + P.h + 20);
          const series = [
            ['valObs', C.warn, 'what validation reports'],
            ['trueQ', C.text, 'the truth'],
            ['testObs', C.green, 'what the untouched test set reports'],
          ];
          series.forEach(([k, col]) => {
            if (best.length < 2) return;
            g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
            best.forEach((b, i) => { const y = py(ctx.clamp(b[k], 0.66, 1.0)); i ? g.lineTo(px(i), y) : g.moveTo(px(i), y); });
            g.stroke();
          });
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('score of the best-so-far setting', P.x, P.y - 14);

          /* ---- verdict panel ---- */
          const X = 520;
          g.font = FONT;
          series.forEach(([k, col, lab], i) => {
            g.fillStyle = col; g.fillRect(X, 118 + i * 46, 12, 3);
            g.fillStyle = C.muted;
            wrapLines(g, lab, 170).forEach((ln, j) => g.fillText(ln, X + 20, 122 + i * 46 + j * 15));
          });
          if (best.length) {
            const b = best[best.length - 1];
            const lie = (b.valObs - b.trueQ) * 100;
            g.font = 'bold 15px Inter, system-ui, sans-serif'; g.fillStyle = C.text;
            g.fillText('after ' + best.length + ' settings', X, 262);
            g.font = MONO;
            g.fillStyle = C.warn; g.fillText('validation says ' + (b.valObs * 100).toFixed(1) + '%', X, 284);
            g.fillStyle = C.text; g.fillText('truth is      ' + (b.trueQ * 100).toFixed(1) + '%', X, 302);
            g.fillStyle = C.green; g.fillText('test says     ' + (b.testObs * 100).toFixed(1) + '%', X, 320);
            /* the verdict can run to four lines, so it starts high enough that the
               last one still lands on the canvas */
            g.font = 'bold ' + FONT; g.fillStyle = lie > 2 ? C.danger : C.muted;
            wrapLines(g, lie > 2
              ? 'Validation has flattered you by ' + lie.toFixed(1) + ' points. The test set has not.'
              : 'Keep tuning and watch the yellow line pull away from the white one.', 178)
              .forEach((ln, j) => g.fillText(ln, X, 344 + j * 16));
            ro.set({ tried: best.length, validation: (b.valObs * 100).toFixed(1) + '%', truth: (b.trueQ * 100).toFixed(1) + '%', test: (b.testObs * 100).toFixed(1) + '%' });
          } else {
            g.font = FONT; g.fillStyle = C.muted;
            wrapText(g, 'Press "Tune 50 times". Each setting is scored on validation and the best one is kept.', X, 290, 175, 16);
            ro.set({ tried: 0, validation: '—', truth: '—', test: '—' });
          }
        });

        return ctx.figure(cv, 'A thousand examples, cut into three piles. Each "tune" tries one more hyperparameter setting, scores it on the validation set, and keeps whichever looks best so far. Because validation is a finite sample, its score is the true quality plus a bit of luck — and picking the maximum of many noisy scores systematically picks the lucky ones. The yellow line drifts above the white one. The test set was never used for choosing, so its luck is not selected for: the green line wobbles to either side of the truth instead of climbing away from it.', [trSl, vaSl, oneBtn, manyBtn, resetBtn], ro);
      }

      /* ================================================================== */
      /* Interactive E: mini-batches — noise, and why a GPU wants them big  */
      /* ================================================================== */
      function batchNoise() {
        const [cv, g] = ctx.canvas(720, 340);
        const N = 2000;
        let batch = 8;
        const data = [];
        (function build() {
          const r = rng(7);
          for (let i = 0; i < N; i++) data.push([1.0 + gauss(r) * 1.6, 0.55 + gauss(r) * 1.6]);
        })();
        const TRUE = data.reduce((a, d) => [a[0] + d[0] / N, a[1] + d[1] / N], [0, 0]);

        const bSl = ctx.slider({ label: 'mini-batch size', min: 1, max: 512, step: 1, value: 8, onChange: (v) => { batch = v; } });
        const presets = [1, 32, 512].map(b => ctx.button('batch ' + b, () => { batch = b; bSl.value = b; }));
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          /* ---- left: 40 sampled mini-batch gradients fanning around the true one ---- */
          const O = { x: 175, y: 195 }, SC = 52;
          const L = { x: 24, y: 40, w: 332, h: 290 };   // the left panel owns this box
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('where one mini-batch thinks downhill is', 30, 28);
          /* At batch 1 a single example can point anywhere and be several times
             longer than the true gradient, so the fan is clipped to its own panel:
             nothing here may paint over the cost chart on the right. */
          g.save(); g.beginPath(); g.rect(L.x, L.y, L.w, L.h); g.clip();
          g.strokeStyle = C.line; g.lineWidth = 1;
          g.beginPath(); g.moveTo(30, O.y); g.lineTo(320, O.y); g.stroke();
          g.beginPath(); g.moveTo(O.x, 45); g.lineTo(O.x, 320); g.stroke();
          let angSum = 0;
          const SAMPLES = 40;
          for (let s = 0; s < SAMPLES; s++) {
            let ax = 0, ay = 0;
            for (let i = 0; i < batch; i++) { const d = data[(Math.random() * N) | 0]; ax += d[0]; ay += d[1]; }
            ax /= batch; ay /= batch;
            const dot = ax * TRUE[0] + ay * TRUE[1];
            const cosang = dot / (Math.hypot(ax, ay) * Math.hypot(TRUE[0], TRUE[1]) || 1);
            angSum += Math.acos(ctx.clamp(cosang, -1, 1)) * 180 / Math.PI;
            g.strokeStyle = 'rgba(124,156,255,0.35)'; g.lineWidth = 1.5;
            g.beginPath(); g.moveTo(O.x, O.y); g.lineTo(O.x + ax * SC, O.y - ay * SC); g.stroke();
          }
          g.strokeStyle = C.text; g.lineWidth = 3;
          g.beginPath(); g.moveTo(O.x, O.y); g.lineTo(O.x + TRUE[0] * SC, O.y - TRUE[1] * SC); g.stroke();
          g.restore();
          /* the label belongs to the arrow but must stay clear of the cost chart,
             so it is drawn unclipped and on two short lines */
          const tipX = O.x + TRUE[0] * SC, tipY = O.y - TRUE[1] * SC;
          g.fillStyle = C.text; g.font = MONO;
          g.fillText('true gradient', tipX + 8, tipY - 20);
          g.fillText('(all ' + N.toLocaleString() + ' examples)', tipX + 8, tipY - 6);
          const avgAng = angSum / SAMPLES;
          g.font = 'bold 15px Inter, system-ui, sans-serif'; g.fillStyle = avgAng < 12 ? C.green : avgAng < 30 ? C.warn : C.danger;
          g.fillText('average error: ' + avgAng.toFixed(1) + '°', 30, 310);

          /* ---- right: cost per epoch, CPU vs GPU ---- */
          const P = { x: 400, y: 60, w: 280, h: 200 };
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('time for one pass over the data', P.x, 28);
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
          const bx = (b) => P.x + Math.log2(b) / 9 * P.w;
          /* CPU: work is the same however you slice it. GPU: fixed per-step overhead
             dominates at small batches, so bigger batches are nearly free until it saturates. */
          const cpu = () => 1.0;
          const gpu = (b) => 0.06 + 1.9 / b + (b > 256 ? (b - 256) / 256 * 0.10 : 0);
          const maxY = 1.3;
          const py = (v) => P.y + P.h - ctx.clamp(v, 0, maxY) / maxY * P.h;
          [[cpu, C.danger, 'CPU'], [gpu, C.green, 'GPU']].forEach(([fn, col, lab]) => {
            g.strokeStyle = col; g.lineWidth = 2.5; g.beginPath();
            for (let e = 0; e <= 9; e += 0.1) { const b = Math.pow(2, e); const X = bx(b), Y = py(fn(b)); e ? g.lineTo(X, Y) : g.moveTo(X, Y); }
            g.stroke();
            g.fillStyle = col; g.font = MONO; g.fillText(lab, P.x + P.w - 34, py(fn(512)) - 8);
          });
          g.setLineDash([4, 4]); g.strokeStyle = C.text; g.lineWidth = 1.5;
          g.beginPath(); g.moveTo(bx(batch), P.y); g.lineTo(bx(batch), P.y + P.h); g.stroke();
          g.setLineDash([]);
          g.font = MONO; g.fillStyle = C.muted;
          [1, 8, 64, 512].forEach(b => g.fillText(String(b), bx(b) - 6, P.y + P.h + 18));
          g.fillText('batch size (log scale)', P.x + 70, P.y + P.h + 36);
          ro.set({ batch, 'avg angle error': avgAng.toFixed(1) + '°', 'noise ∝ 1/√b': (1 / Math.sqrt(batch)).toFixed(3) });
        });

        return ctx.figure(cv, 'Left: forty different mini-batches, each drawn fresh from the same 2,000 examples, each pointing where <i>it</i> thinks downhill is. The white arrow is the true gradient over all the data. At batch 1 the fan is wild. Above a handful of examples the error shrinks with 1/√(batch size), so 4× the batch halves the noise. Right: a CPU does the same total work however you slice it, while a GPU has a fixed cost per step it can amortise over a whole batch at once — which is why batching is nearly free there, and one reason deep learning waited for graphics hardware. Both cost curves are drawn schematically, not measured: a real CPU gains a little from batching too, but nothing like this.', [bSl, ...presets], ro);
      }

      /* ================================================================== */
      /* Interactive F: where "unlabelled" training data gets its labels    */
      /* ================================================================== */
      function freeLabels() {
        let text = 'The cat sat on the mat because it was warm';
        let mode = 'next';
        const out = h('div', { class: 'figure-body' });
        const ta = ctx.textarea({ label: 'Type any sentence at all', value: text, onChange: (v) => { text = v; draw(); } });
        /* the highlight follows the mode, so the emphasised button is always the
           one whose table is on screen */
        const setMode = (m) => { mode = m; nextBtn.classList.toggle('primary', m === 'next'); maskBtn.classList.toggle('primary', m === 'mask'); draw(); };
        const nextBtn = ctx.button('predict the next word', () => setMode('next'), 'primary');
        const maskBtn = ctx.button('fill in the blank', () => setMode('mask'));
        const ro = ctx.readout();

        function draw() {
          const words = String(text).trim().split(/\s+/).filter(Boolean);
          out.innerHTML = '';
          const rows = [];
          if (mode === 'next') {
            for (let i = 1; i < words.length; i++) rows.push([words.slice(Math.max(0, i - 7), i).join(' '), words[i]]);
          } else {
            for (let i = 0; i < words.length; i++) {
              const c = words.slice();
              c[i] = '▁▁▁▁';
              rows.push([c.join(' '), words[i]]);
            }
          }
          const shown = rows.slice(0, 9);
          const tbl = h('table', { class: 'tbl' },
            h('thead', {}, h('tr', {},
              h('th', { html: mode === 'next' ? 'input the model sees' : 'input the model sees (one word hidden)' }),
              h('th', { html: 'free label — the answer, taken straight from the text' }))),
            h('tbody', {}, shown.map(([a, b]) => h('tr', {},
              h('td', { html: '<code class="inline">' + a.replace(/</g, '&lt;') + '</code>' }),
              h('td', { html: '<b style="color:#38d9a9">' + b.replace(/</g, '&lt;') + '</b>' })))));
          out.append(h('div', { class: 'table-wrap' }, tbl));
          if (rows.length > shown.length) out.append(h('p', { class: 'figure-caption', html: '…and ' + (rows.length - shown.length) + ' more from this one sentence.' }));
          ro.set({ words: words.length, 'training examples': rows.length, 'humans needed': 0 });
        }
        draw();
        return ctx.figure(out, 'No human labelled any of this. The label column was <i>already in the sentence</i> — it is just the word you covered up. That is the whole trick behind "pretrained on unlabelled text": ordinary writing is silently an infinite exercise book, and the model is trained on it with exactly the cross-entropy loss and backpropagation from chapter 2. Fifteen trillion tokens of text is roughly fifteen trillion of these rows.', [ta, nextBtn, maskBtn], ro);
      }

      /* ================================================================== */
      /* Interactive G: the spam dial — accuracy lies, precision and recall */
      /* ================================================================== */
      function spamDial() {
        const [cv, g] = ctx.canvas(720, 360);
        const N = 100;
        let rate = 20, thr = 0.5, mail = [];
        function build() {
          const r = rng(99);
          mail = [];
          const nSpam = Math.round(N * rate / 100);
          for (let i = 0; i < N; i++) {
            const isSpam = i < nSpam;
            mail.push({ spam: isSpam, score: ctx.clamp((isSpam ? 0.70 : 0.26) + gauss(r) * 0.155, 0.04, 0.96) });
          }
        }
        build();
        function counts() {
          let tp = 0, fp = 0, fn = 0, tn = 0;
          for (const m of mail) {
            const flagged = m.score >= thr;
            if (m.spam && flagged) tp++; else if (!m.spam && flagged) fp++;
            else if (m.spam && !flagged) fn++; else tn++;
          }
          return { tp, fp, fn, tn };
        }
        const rateSl = ctx.slider({ label: 'how much of the mail is really spam (%)', min: 5, max: 50, step: 1, value: 20, onChange: (v) => { rate = v; build(); } });
        const thrSl = ctx.slider({ label: 'flag as spam above score', min: 0.02, max: 0.98, step: 0.01, value: 0.5, digits: 2, onChange: (v) => { thr = v; } });
        /* 0.98 sits above every possible score and 0.02 below every one, so the two
           presets really do flag none and all of them. */
        const lazyBtn = ctx.button('Flag nothing ever', () => { thr = 0.98; thrSl.value = 0.98; });
        const paranoidBtn = ctx.button('Flag everything', () => { thr = 0.02; thrSl.value = 0.02; });
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          const { tp, fp, fn, tn } = counts();
          const acc = (tp + tn) / N;
          const prec = tp + fp ? tp / (tp + fp) : NaN;
          const rec = tp + fn ? tp / (tp + fn) : NaN;

          /* ---- histogram of scores ---- */
          /* two columns: the histogram owns everything left of COL, the matrix and
             the metric bars everything right of it. COL is far enough left that the
             longest metric note still ends inside the canvas. */
          const COL = 430;
          const P = { x: 40, y: 46, w: 370, h: 180 };
          const BINS = 26;
          const hs = Array.from({ length: BINS }, () => [0, 0]);
          mail.forEach(m => { const b = Math.min(BINS - 1, (m.score * BINS) | 0); hs[b][m.spam ? 1 : 0]++; });
          const peak = Math.max(1, ...hs.map(b => Math.max(b[0], b[1])));
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('every email, scored by the filter', P.x, 30);
          for (let b = 0; b < BINS; b++) {
            const bw = P.w / BINS;
            [[0, 'rgba(56,217,169,0.75)'], [1, 'rgba(251,113,133,0.8)']].forEach(([k, col]) => {
              const hgt = hs[b][k] / peak * (P.h / 2 - 6);
              g.fillStyle = col;
              if (k === 0) g.fillRect(P.x + b * bw + 1, P.y + P.h / 2 - hgt, bw - 2, hgt);
              else g.fillRect(P.x + b * bw + 1, P.y + P.h / 2 + 2, bw - 2, hgt);
            });
          }
          g.strokeStyle = C.line; g.lineWidth = 1;
          g.beginPath(); g.moveTo(P.x, P.y + P.h / 2); g.lineTo(P.x + P.w, P.y + P.h / 2); g.stroke();
          g.strokeStyle = C.warn; g.lineWidth = 2.5;
          g.beginPath(); g.moveTo(P.x + thr * P.w, P.y - 6); g.lineTo(P.x + thr * P.w, P.y + P.h + 6); g.stroke();
          /* These two label the halves of the histogram from inside the plot, and the
             threshold line sweeps across every x the reader can choose — so they are
             drawn after it, each on its own opaque plate. */
          g.font = MONO;
          const plate = (s, tx, ty, col) => {
            const tw = g.measureText(s).width;
            g.fillStyle = '#0a0e16'; g.fillRect(tx - 3, ty - 11, tw + 6, 15);
            g.fillStyle = col; g.fillText(s, tx, ty);
          };
          plate('real mail ↑', P.x + 4, P.y + 12, C.green);
          plate('spam ↓', P.x + 4, P.y + P.h - 4, C.danger);
          g.fillStyle = C.warn; g.font = MONO;
          /* the threshold line runs to the right edge of the plot, so its label is
             held inside the plot instead of sliding under the metric column */
          g.fillText('flag →', Math.min(P.x + thr * P.w + 6, P.x + P.w - g.measureText('flag →').width), P.y + P.h + 20);

          /* ---- confusion matrix ---- */
          const M = { x: COL, y: 56, c: 105, r: 48 };
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('really spam', M.x + 6, M.y - 20);
          g.fillText('really fine', M.x + M.c + 6, M.y - 20);
          const cells = [
            [tp, 'caught', C.green, 0, 0], [fp, 'false alarm', C.danger, 1, 0],
            [fn, 'missed', C.danger, 0, 1], [tn, 'left alone', C.green, 1, 1],
          ];
          cells.forEach(([v, lab, col, cx, cy]) => {
            const X = M.x + cx * M.c, Y = M.y + cy * M.r;
            g.strokeStyle = C.line; g.strokeRect(X, Y, M.c - 6, M.r - 6);
            g.font = 'bold 17px Inter, system-ui, sans-serif'; g.fillStyle = col;
            g.fillText(String(v), X + 8, Y + 22);
            g.font = MONO; g.fillStyle = C.muted; g.fillText(lab, X + 8, Y + 36);
          });

          /* ---- metric bars ---- */
          let y = 200;
          const BARW = 200;
          const bar = (lab, v, col, note) => {
            g.font = FONT; g.fillStyle = C.muted; g.fillText(lab, COL, y);
            g.fillStyle = C.line; g.fillRect(COL, y + 6, BARW, 10);
            if (!isNaN(v)) { g.fillStyle = col; g.fillRect(COL, y + 6, v * BARW, 10); }
            g.font = 'bold ' + FONT; g.fillStyle = col;
            g.fillText(isNaN(v) ? 'n/a' : (v * 100).toFixed(0) + '%', COL + BARW + 8, y + 15);
            if (note) { g.font = MONO; g.fillStyle = C.muted; g.fillText(note, COL, y + 30); }
            y += note ? 54 : 38;
          };
          bar('accuracy', acc, C.accent, 'fraction of all mail judged correctly');
          bar('precision', prec, C.warn, 'of what it flagged, how much was spam');
          bar('recall', rec, C.purple, 'of all the spam, how much it caught');

          /* ---- the trap ---- */
          g.font = 'bold ' + FONT;
          const trapped = tp === 0 && acc > 0.6;
          g.fillStyle = trapped ? C.danger : C.muted;
          wrapText(g, trapped
            ? 'Look at that: ' + (acc * 100).toFixed(0) + '% accuracy while catching zero spam. Accuracy on its own is a liar.'
            : 'Drag the yellow line and watch precision and recall trade against each other.',
            P.x, 264, 370, 17);
          g.font = MONO; g.fillStyle = C.muted;
          wrapText(g, 'Out of ' + N + ' emails, ' + (tp + fn) + ' are really spam. The filter flags ' + (tp + fp) + '; ' + tp + ' of those are right, ' + fp + ' good emails get caught, and ' + fn + ' spam slip through.',
            P.x, 310, 370, 16);
          ro.set({ accuracy: (acc * 100).toFixed(0) + '%', precision: isNaN(prec) ? 'n/a' : (prec * 100).toFixed(0) + '%', recall: isNaN(rec) ? 'n/a' : (rec * 100).toFixed(0) + '%' });
        });

        return ctx.figure(cv, 'One hundred emails. The filter gives each a spam score, and the yellow line is the only thing you actually control: flag everything above it. Move it right and you rarely bother a real email but let spam through (high precision, low recall). Move it left and you catch nearly everything while burying good mail in false alarms (high recall, low precision). There is no setting that maximises both — which one you want depends entirely on what each kind of mistake costs.', [rateSl, thrSl, lazyBtn, paranoidBtn], ro);
      }

      /* ================================================================== */
      /* The chapter: touch first, read second.                             */
      /* ================================================================== */
      root.append(
        callout('tryit', '🖐 Do this first — build an answer key, then watch it fail',
          `Blue dots are the questions you get to study. Orange dots are next year's exam, which you never see while fitting.<br>
           <b>1.</b> Set <b>degree</b> to 1. A straight line, wrong nearly everywhere — the model is too rigid.<br>
           <b>2.</b> Push <b>degree</b> to 15. The curve now bends to chase the blue dots, and <b>train error</b> in the readout is the lowest it has been at any degree.
           <b>Now read the number beside it</b>, the error on the orange dots, and look at the wild swings where there are no blue dots to pin the curve down.<br>
           <b>3.</b> Still at degree 15, press <b>New noise sample</b> a few times. The true curve never moves. Yours reshapes itself completely.<br>
           <b>4.</b> Find the degree with the lowest <i>held-out</i> error on the lower chart.`),
        curveFit(),
        p(`Degree 15 did not learn the curve. It learned the dots — including the random jitter in them, which will never repeat. It built an answer key.`),
      );

      root.append(section('The two ways to be wrong',
        p(`A friend crams for a driving-theory test by memorising last year's answer key, word for word. She scores 100% on that exact paper, then fails this year's badly. She never learned to drive. She learned an answer key, and that is what your degree-15 curve just did.`),
        p(`It has a name: <em>overfitting</em>. The model has enough flexibility to fit the noise as well as the signal, and noise never repeats, so it looks flawless on the data it studied and falls apart on anything new.`),
        p(`Degree 1 showed you the opposite failure. <em>Underfitting</em> is a model too rigid to capture even the signal — a straight line facing a curved reality — and it does badly on training data and new data alike. Both failures are fixable. Neither is visible if you only look at the training score.`),
        callout('key', '🔑 The most important number in machine learning',
          `Not the training score. <b>The gap between the training score and the held-out score.</b>
           A model that scores perfectly on what it studied and poorly on anything else is worthless, and only the gap reveals it.`),
        p(`Statisticians call the two failures bias and variance. <b>Bias</b> is being wrong the same way every time, whatever data you drew — a straight line will always miss a curve. <b>Variance</b> is instability: that reshaping you saw when you pressed <b>New noise sample</b> at degree 15.`),
        p(`Simple models are high-bias and low-variance; flexible ones are low-bias and high-variance. That is why the held-out error formed a <b>U</b>: too rigid on the left, too twitchy on the right, and the sweet spot somewhere in the middle.`),
        callout('example', '🌍 Overfitting in the wild',
          `A résumé-screening model trained on 200 successful hires from one company can hit 99% training accuracy by memorising which postcodes and college names happened to correlate with success in that one small batch — spurious correlations, not job skill.
           On the next batch of candidates it performs close to random, and it does so <i>confidently</i>.
           This is why serious teams report validation and test numbers and never training numbers, and why a vendor's "99.9% on our data" should always prompt the follow-up: measured how, on which held-out set?`),
      ));

      root.append(section('Three piles, and the one you must not spend',
        p(`So you hold data back. Cut it into three piles before you write a line of training code — and the reason there are three, not two, is subtler than it looks.`),
        callout('tryit', '🖐 Try this — spend the validation set and watch it lie to you',
          `<b>1.</b> Press <b>Tune 50 times</b>. Each press tries another hyperparameter setting and keeps whichever scored best on validation.<br>
           <b>2.</b> Watch the <b>yellow</b> line (what validation reports) pull away above the <b>white</b> line (the truth). The <b>green</b> line, the untouched test set, lands on either side of the truth and does not chase the yellow one.<br>
           <b>3.</b> Drag <b>validation examples</b> down to 25 and tune 50 times again. The lie gets much bigger.<br>
           <b>4.</b> Drag it up to 400. The lie shrinks. <b>Nothing about the model changed</b> — only how many examples you were judging it with.`),
        splitDeck(),
        p(`Nobody cheated there. Validation scores are the true quality plus a bit of sampling luck, and picking the best of fifty noisy scores reliably picks the luckiest one. Report that number as your result and you have quietly overfit the validation set.`),
        p(`Hence three piles. The <em>training set</em> is what the optimizer sees. The <em>validation set</em> is what you check while still making choices — layers, learning rate, when to stop — because those choices are themselves a kind of fitting. The <em>test set</em> is touched exactly once, at the very end.`),
        p(`A common split for 1,000 examples is 70/15/15. Train on the 700, tune against the 150, then score once on the untouched 150. That final number can be trusted. Re-use validation as your reported result after fifty rounds of tuning, and you have turned it into a second training set — which is precisely what the yellow line was showing you.`),
      ));

      root.append(section('Four levers against overfitting',
        p(`Four practical tools push back, and you will keep meeting them for the rest of the course.`),
        ul([
          `<b>Weight decay</b> — add a small penalty for large weights, so the optimizer keeps them small unless the data really insists. Small weights make smoother functions, and smooth functions memorise noise less easily. (The ridge term quietly stabilising the polynomial fit above is exactly this.)`,
          `<b>Dropout</b> — during training, randomly switch off a fraction of neurons on every pass, so the survivors cannot rely on one teammate always being present. The network is forced to learn redundant, robust features.`,
          `<b>Early stopping</b> — watch validation loss during training, and when it has clearly stopped improving, go back and keep the weights from the epoch where it was lowest. (The single lowest point, not the first uptick: validation loss is noisy and wobbles up and down on the way.)`,
          `<b>More and cleaner data</b> — the single most reliable fix. A model cannot memorise noise it has not seen enough of.`,
        ]),
        p(`All four do the same underlying thing: make it harder to fit noise without making it harder to fit signal. None are free — overdo weight decay or dropout and you are back to underfitting.`),
        callout('tryit', '🖐 Try this: catch the exact moment it turns',
          `It is already running. Training loss (blue) keeps falling — the model can always get better at its own homework. Validation loss (red) falls, then <b>turns upward</b>. That turn is memorisation beginning.<br>
           <b>1.</b> Find the green dot: the best validation epoch, and where early stopping would have saved you.<br>
           <b>2.</b> Set training set size to its smallest and regularization to zero. The turn arrives early and steeply.<br>
           <b>3.</b> Set size to its largest and regularization to maximum. The turn still arrives and the demo still plants its green dot — but look how little the red line rises after it. That is what "this model is not overfitting much" looks like.<br>
           <b>4.</b> Press <b>↺ Replay</b> a few times at fixed settings: the curve jitters, the story never changes.`),
        earlyStopping(),
        p(`Notice that the blue line never betrays you. It falls forever. If training loss were all you watched, you would ship the model from the far right of that chart — the most-trained one, and the worst one.`),
      ));

      root.append(section('Better ways downhill',
        p(`Chapter 2's gradient descent takes a step of size η downhill, every time. Two problems appear in real networks. Some directions of the landscape are steep and others nearly flat — a narrow curved valley, not a round bowl. And each of a million parameters would like its own step size, which no single global η can provide.`),
        callout('tryit', '🖐 Try this: three optimizers, one nasty valley',
          `Darker background is lower loss; the white cross is the true minimum. All three start together.<br>
           <b>1.</b> Press <b>▶ Start</b>. Plain SGD (grey) crawls — it fights the curvature at every step.<br>
           <b>2.</b> Momentum (yellow) builds speed down the slope and gets there about ten times sooner, wobbling at the turn into the valley.<br>
           <b>3.</b> Adam (green) takes a wigglier path, not a smoother one — but reaches the floor in a thirteenth of SGD's steps, and unlike the other two it survives every setting its slider offers.<br>
           <b>4.</b> Push the SGD or momentum learning rate up until its ball bounces off the walls. Then do the same to Adam — it tolerates a far wider range before misbehaving.`),
        optimizerRace(),
        p(`<em>Momentum</em> fixes the first problem by changing the metaphor. Instead of a marble that stops dead when you stop pushing, imagine a heavy ball: it keeps most of its previous velocity and adds the new gradient on top. Down a narrow valley it builds speed along the floor, while the side-to-side component flips sign each step and largely cancels itself out.`),
        callout('key', '🔑 Momentum, by hand',
          `v ← β·v − η·∇L, then position ← position + v, with β ≈ 0.9 (keep 90% of last step's velocity).<br>
           Say the ball is rolling left, v = −0.40, and one noisy batch now says "go right": its gradient is −0.30 (downhill is always against the gradient), with β = 0.9 and η = 0.1.<br>
           New velocity = 0.9 × (−0.40) − 0.1 × (−0.30) = −0.36 + 0.03 = <b>−0.33</b>. Still heading left, only dented.
           The ball remembers where it was heading and shrugs off one noisy gradient — which is exactly what you want when every gradient comes from a small random batch.`),
        p(`<em>Adam</em> fixes the second problem. For each parameter it tracks a running average of the gradient <i>and</i> of the gradient's squared size, then divides the first by the square root of the second. A parameter with consistently huge gradients gets shrunk to a sane step; one with tiny rare gradients gets amplified. Every parameter effectively gets a learning rate chosen from its own history.`),
        p(`A neat fact falls out of the algebra: on the very first update Adam's step is almost exactly η × sign(gradient), regardless of the gradient's actual size. Plain SGD's first step is η × gradient, so one freak batch throws it wildly off. That is why Adam tolerated the much wider slider range you just tested.`),
        p(`<em>AdamW</em> is Adam with one fix: weight decay applied directly to the weights rather than folded into the gradient. That decoupling turned out to matter, and AdamW — not plain Adam — trains the overwhelming majority of transformers today, including every language model in Part III.`),
        callout('history', '📜 Dropout (2012), Adam (2014), AdamW (2017)',
          `<em>Dropout</em> came from Geoffrey Hinton's group — described by Hinton and colleagues in 2012, then formalised by Nitish Srivastava, Hinton, Alex Krizhevsky, Ilya Sutskever and Ruslan Salakhutdinov in a 2014 JMLR paper.
           Hinton has said the idea came from how a bank foils fraud by shuffling which teller serves which customer, so no single employee's habits can be relied on to make a swindle work.<br>
           <em>Adam</em> was introduced by Diederik Kingma and Jimmy Ba in 2014, combining momentum with an earlier adaptive method called RMSProp.
           <em>AdamW</em> followed in 2017 from Ilya Loshchilov and Frank Hutter: a small algebraic fix that measurably improved final model quality, and is now the default in essentially every serious training codebase.`),
      ));

      root.append(section('Why training happens in batches',
        callout('tryit', '🖐 Try this',
          `<b>1.</b> Press <b>batch 1</b>. Forty mini-batches, forty wildly different opinions about which way is downhill.<br>
           <b>2.</b> Press <b>batch 32</b>, then <b>batch 512</b>. The fan collapses toward the white arrow.<br>
           <b>3.</b> Watch the right-hand chart as you do it: on a GPU, all that extra accuracy costs almost nothing.`),
        batchNoise(),

        /* maths beat: the average, its spread, and sigma / sqrt(n) — named after the fan */
        ctx.walkthrough([
          { say: 'Each thin blue line in that fan is an <b>average</b>: every example in one mini-batch voted on which way is downhill, and the votes were added up and divided by the batch size.', note: 'That is all a mini-batch gradient is.' },
          { say: 'The white arrow is the same average taken over all 2,000 examples, and it is what every blue line is trying to guess.', note: 'The value a noisy average is aiming at has a name: its <b>expectation</b>. You already know it as the long-run average.' },
          { say: 'A guess made from few examples is jumpy: at batch 32 the fan missed the white arrow by about 12°.', math: 'batch 32 &rarr; about 12° off' },
          { say: 'Then you pressed batch 512, sixteen times the examples, and the miss fell to about 3°: a quarter, not a sixteenth.', math: '&times;16 the examples &rarr; &divide;4 the error' },
          { ask: 'Sixteen times the data cut the error by four. If that pattern holds, how much data cuts the error by <b>ten</b>?',
            options: ['10 times as much', '20 times as much', '100 times as much'], answer: 2,
            explain: 'The error shrinks with the <b>square root</b> of the number of examples. √16 = 4, and √100 = 10.' },
          { say: 'The readout under the demo has been printing exactly this, as one divided by the square root of the batch size.', math: '1 &divide; √32 = 0.177 &nbsp;&nbsp;&nbsp; 1 &divide; √512 = 0.044' },
          { say: 'The 1 on top stands in for how much single examples disagree with each other, which statisticians measure with the <b>standard deviation</b>, written <b>&sigma;</b>.', note: 'Square it and you have the <b>variance</b>. In this demo each single example\'s vote has &sigma; = 1.6.' },
          { say: 'So the spread of an average of <b>n</b> things is &sigma; divided by the square root of n.', math: 'batch 64: &nbsp; 1.6 &divide; √64 = 1.6 &divide; 8 = 0.2' },
          { say: 'You met the same law in the three piles, because a validation score is an average too: 150 rights and wrongs, added up and divided by 150.', math: '0.4 &divide; √150 = 0.033 &rarr; about 3 points of luck', note: 'For a model that is right 80% of the time, the &sigma; of a single right-or-wrong is √(0.8 &times; 0.2) = 0.4.' },
          { ask: 'With only <b>25</b> validation examples, how much luck rides on each score?',
            options: ['About 8 points', 'About 3 points', 'About 1 point'], answer: 0,
            explain: '0.4 ÷ √25 = 0.4 ÷ 5 = 0.08. Eight points of luck on every score, and you kept the luckiest of fifty. That is why the yellow line ran away.' },
          { say: 'The training loss is one more average, of chapter 2\'s &minus;log p over every example in the batch, so it wobbles by the same law.', note: 'Which is why the curves in the early-stopping chart jitter from one replay to the next.' },
        ], {
          title: 'From a fan of arrows to σ ÷ √n',
          recap: 'an average of n noisy things misses by about &sigma; &divide; √n, so four times the data buys half the noise.',
        }),
        ctx.decoder([
          { sym: 'noise', name: 'how far a typical average lands from the truth', says: 'Statisticians call it the <em>standard error</em>. It is the standard deviation of the average itself, not of one example.', points: 'the width of the blue fan; the gap that opened between the yellow line and the white one.' },
          '=',
          { sym: '&sigma;', name: 'sigma', says: 'How much single examples disagree with each other: the standard deviation. The data sets it. You do not.', points: 'the wild fan at batch 1, where each line is one example.' },
          '÷',
          { sym: '√', name: 'the square root of', says: 'The reason the exchange rate is punishing. Four times the data gives twice the precision; a hundred times gives ten.', points: 'batch 32 to batch 512 taking 12° down to 3°.' },
          { sym: 'n', name: 'the number of things averaged', says: 'The only part you control.', points: 'the mini-batch size slider, and the validation examples slider before it.' },
        ], {
          title: 'noise  =  σ ÷ √n',
          plain: 'the noise in an average is the disagreement between single examples, divided by the square root of how many you averaged.',
        }),

        p(`Real training almost never computes the gradient from one example (that was chapter 2's playground, for clarity) nor from the whole dataset at once (far too slow to ever take a step). It averages over a <em>mini-batch</em> — commonly 32 to a few thousand — takes one step, then moves on.`),
        p(`That square root is a punishing exchange rate, which is why batches are not simply enormous: past a point you are paying four times the work <i>in each update</i> to halve a noise that was not hurting you much anyway.`),
        p(`A GPU's advantage was never doing one multiplication fast — it is doing thousands at the same instant. Feed it one example and most of the chip idles. Feed it 512 stacked into one matrix and the same multiply handles all of them for barely more time. `),
        p(`In this idealised picture bigger batches buy a CPU nothing at all — it does the same total work however you slice it, which is what the flat red line says. On a GPU they are nearly free, and that asymmetry is one quiet reason deep learning did not take off until graphics hardware was repurposed for it.`),
        callout('key', '🔑 Parameters vs hyperparameters',
          `<b>Parameters</b> are what the optimizer learns: weights and biases, everything adjusted by ∂L/∂w.<br>
           <b>Hyperparameters</b> are what a human sets before training and the optimizer never touches: learning rate, batch size, number of layers, dropout rate, the polynomial degree you dragged at the top of this chapter.<br>
           You <i>learn</i> parameters by gradient descent on the training set. You <i>tune</i> hyperparameters by watching validation — which is exactly why the validation set gets spent. There is no ∂L/∂(learning rate) to compute.`),
      ));

      root.append(section('Where the labels come from',
        p(`Every example so far arrived with a correct answer attached. That is only one of four ways a model can learn.`),
        cards([
          { title: 'Supervised', body: 'Every example is labelled by a human or an existing system: photo → "cat", house features → sale price. Precise, but labelling at scale is expensive.' },
          { title: 'Unsupervised', body: 'No labels at all. The model finds structure in the data itself — grouping similar customers, or compressing 1,000 numbers into 10 that still capture the pattern. Nobody defines "similar"; the data\'s own geometry does.' },
          { title: 'Self-supervised', body: 'The labels are manufactured from the data itself, for free: hide a word and predict it, hide part of an image and fill it in. No human labeller required.' },
          { title: 'Reinforcement learning', body: 'No answer table at all. An agent acts in an environment and gets a reward much later, and must work out which earlier action deserves the credit. Chapter 9 covers this properly.' },
        ]),
        callout('tryit', '🖐 Try this — manufacture training data out of thin air',
          `Type any sentence you like into the box, or paste a paragraph.<br>
           <b>Watch the counter:</b> every word you add creates another labelled training example, and the "humans needed" figure stays at zero.<br>
           Then press <b>fill in the blank</b> to see the other flavour of the same trick.`),
        freeLabels(),
        p(`This resolves a confusion worth clearing up. When you read that GPT-style models are "pretrained on the internet with no labels", that is <em>self-supervised</em> learning wearing a supervised engine underneath.`),
        p(`Every "predict the next word" example carries a free label — the actual next word — manufactured automatically from ordinary text, then trained with exactly the cross-entropy loss and backpropagation from chapter 2. It is what let language models train on essentially the whole internet, instead of waiting for a hand-labelled dataset that could never be big enough.`),
        p(`More data helps, but not all data is worth the same. Ten thousand carefully checked, genuinely diverse examples routinely beat a million scraped, duplicated, mislabelled ones — because the optimizer cannot tell a real pattern from a systematic error in the labels. It fits both with equal enthusiasm.`),
        callout('example', '🌍 Curation beats scale',
          `Early large language model training sets went through aggressive filtering and deduplication before a single weight was updated; Lee et al. (2022) found one 61-word sequence repeated over 60,000 times in C4, and showed that simply removing near-duplicate documents measurably improved the resulting model, because duplicated text taught it to overweight whatever happened to be copy-pasted often across the web.
           Later, smaller-but-carefully-curated model families showed the same lesson from the other direction: a modestly sized model trained on well-chosen data can beat a much larger model trained carelessly.
           "Bigger dataset" and "better dataset" are different axes, and the second is usually the cheaper one to pull.`),
      ));

      root.append(section('Grading the model',
        p(`"Evaluate" hides a real design choice: what exactly are you measuring? <em>Accuracy</em> — the fraction of predictions that are correct — is the obvious first metric and the most misleading one whenever the classes are unbalanced.`),
        callout('tryit', '🖐 Try this — build a 95%-accurate model that is completely useless',
          `<b>1.</b> Set <b>how much of the mail is really spam</b> to 5%, then press <b>Flag nothing ever</b>.<br>
           <b>Read the accuracy bar: 95%.</b> Now read the recall bar: zero. The filter catches no spam whatsoever and still scores 95%.<br>
           <b>2.</b> Press <b>Flag everything</b>. Recall is now perfect and precision is terrible.<br>
           <b>3.</b> Set spam back to 20% and drag the yellow line to about <b>0.48</b>: precision 73%, recall 95%. Now drag it to <b>0.62</b>: precision jumps to 94% and recall collapses to 75%. Past the easy gains, one cannot rise without the other falling.`),
        spamDial(),
        p(`Two sharper questions than accuracy. Of the emails it flagged, how many really were spam — that is <em>precision</em>. Of all the spam there was, how many did it catch — that is <em>recall</em>. The confusion matrix gives you both, and accuracy gives you neither.`),
        p(`A filter tuned for high recall catches nearly everything and annoys users with false alarms. One tuned for high precision rarely bothers a real email and lets more spam through. Which you want depends entirely on what each kind of mistake costs — and that is a product decision, not a modelling one.`),
        p(`Language models use a metric built straight from cross-entropy: <em>perplexity</em>, roughly e raised to the average per-token loss. A cross-entropy of 2.3 nats is a perplexity of about 10 — "about as unsure as if it had to guess uniformly among 10 equally likely next words". Lower is better, and a well-trained large model sits in the single digits on ordinary text.`),
      ));

      root.append(section('Why this matters for modern AI',
        p(`Every headline about a new frontier model is this chapter's recipe run at a scale a 2012 researcher would have called science fiction. Tens of terabytes of self-supervised text, filtered down from petabytes of raw crawl, standing in for hand-made labels. Cross-entropy for the loss. AdamW for the optimizer.`),
        p(`Hundreds of billions of parameters for the model, often a few trillion. Most frontier models whose design is public are now <em>sparse mixtures of experts</em>, so only a fraction of those parameters are used on any one token — which is why a single headline number tells you less than it used to. Chapter 10 takes that apart.`),
        p(`And a held-out benchmark — never the training data — standing in for the report card anyone trusts. The training-versus-validation gap you watched turn upward is the same gap researchers track at the scale of trillions of tokens.`),
        p(`A striking share of frontier-lab effort goes into data curation and evaluation: the first and last levers in this chapter. The regularizers change as you scale — a single pass over trillions of tokens leaves little to memorise, so frontier pretraining generally drops dropout altogether. The recipe does not change as it scales. Only its size does.`),
        callout('warning', '⚠️ The failure mode that survives at every scale',
          `Benchmark contamination is overfitting wearing a lab coat. If a test set leaked into the training data — and at internet scale, some of it always has — the reported score is a training score,
           and the model is grading its own homework in front of an audience. This is why frontier labs build fresh held-out evaluations, and why a benchmark number without a contamination check deserves the same follow-up as that vendor's "99.9% on our data".`),
        p(`One picture to keep: <b>the loss tells you how wrong you are, the optimizer tells you which way to move, and the validation set is the only honest judge of whether any of it generalises.</b>`),
      ));

      root.append(
        ctx.quiz([
          { q: 'A colleague shows you a model with training error 0.01 and held-out error 0.48, and wants to make it more flexible to push the training error lower still. What do you tell them?',
            options: ['Go ahead: a lower training error always carries over to new data', 'It is underfitting, so more flexibility is exactly what it needs', 'That gap says it is already fitting noise, and more flexibility will widen it', 'Held-out error is only noisier than training error, so ignore it'], answer: 2,
            explain: 'Numbers like these are what your degree-15 curve produced. Underfitting would show as two bad scores close together. A tiny training error beside a large held-out one is overfitting, and the cure is less flexibility, more data or regularization.' },
          { q: 'You try 50 settings, and the best one scores 90% on a 150-example validation set. What is the most honest forecast for that setting on fresh data?',
            options: ['Somewhat below 90%, because the best of fifty noisy scores is partly luck', 'About 90%, because the validation set was never trained on', 'Above 90%, because fresh data has not been tuned against', 'No forecast is possible until the model is retrained'], answer: 0,
            explain: 'This is the yellow line pulling away from the white one. Each score is true quality plus about 3 points of sampling luck (0.4 ÷ √150), and keeping the maximum selects for the luck. An untouched test set is how you find out how much.' },
          { q: 'A cancer-screening test is tuned so that it almost never misses a real case. As a result it sends many healthy people for follow-up scans. In this chapter\'s terms it has…',
            options: ['High precision and low recall', 'High precision and high recall', 'Low precision and low recall', 'High recall and low precision'], answer: 3,
            explain: 'Recall is the share of real cases caught: high. Precision is the share of flagged people who really have it: low. It is the spam dial with the yellow line dragged far to the left, and here that is the right trade, because a miss costs far more than a false alarm.' },
          { q: 'Your gradient at batch size 32 points about 12° away from the true one. Roughly what batch size brings that down to about 6°?',
            options: ['64', '128', '1,024', '16'], answer: 1,
            explain: 'Noise falls as σ ÷ √n, so halving it takes four times the examples: 32 × 4 = 128. Doubling to 64 only buys a factor of √2. That exchange rate is why batches are not simply made enormous.' },
          { q: 'A momentum optimizer keeps 90% of its last velocity. It is moving at −0.40 (leftwards) when one noisy batch adds a push of +0.03 to the right. What is its new velocity?',
            options: ['0.00: it stops dead', '+0.03: it turns and follows the newest gradient', '−0.33: still heading left, slightly slower', '−0.43: the push adds to its speed'], answer: 2,
            explain: '0.9 × (−0.40) + 0.03 = −0.36 + 0.03 = −0.33. Plain SGD would have followed the noisy batch. Momentum averages over recent steps, so a single bad gradient only dents it.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://developers.google.com/machine-learning/guides/rules-of-ml" target="_blank" rel="noopener">Google, "Rules of Machine Learning"</a>: 43 hard-won practical rules from people who ship models — the engineering discipline that surrounds everything in this chapter, once a model has to survive contact with a real product.`,
            `<a href="https://distill.pub/2017/momentum/" target="_blank" rel="noopener">"Why Momentum Really Works" (Distill)</a>: an interactive article that lets you feel the valley the optimizer race above is crossing.`,
            `<a href="https://arxiv.org/abs/1412.6980" target="_blank" rel="noopener">Kingma &amp; Ba (2014), "Adam: A Method for Stochastic Optimization"</a>: the original paper, unusually readable.`,
            `<a href="https://arxiv.org/abs/1711.05101" target="_blank" rel="noopener">Loshchilov &amp; Hutter (2017), "Decoupled Weight Decay Regularization"</a>: the one-line fix that made AdamW the default everywhere.`,
            `<a href="https://jmlr.org/papers/v15/srivastava14a.html" target="_blank" rel="noopener">Srivastava et al. (2014), "Dropout"</a>: the JMLR paper. Its own motivation reaches for sexual reproduction and small conspiracies rather than bank tellers; the teller story is one Hinton has told in talks and interviews.`,
          ]),
        ),
      );
    },
  });
})();
