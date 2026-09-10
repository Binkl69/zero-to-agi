/* Zero → AGI · Chapter 03 · The learning recipe
   The five-step loop every model follows: data, model+parameters, loss, optimizer, evaluation.
   Train/val/test splits, overfitting vs underfitting, bias-variance, regularization, optimizers
   (SGD -> momentum -> Adam/AdamW), batching & GPUs, hyperparameters vs parameters, the four kinds
   of learning, data quality vs quantity, and evaluation metrics (accuracy, precision/recall, perplexity).
   Interactives: polynomial curve-fit (train/held-out U-curve); optimizer race on a Rosenbrock valley;
   an animated early-stopping training-vs-validation curve. */
(function () {
  ZTA.registerChapter({
    id: '03-the-learning-recipe',
    num: 3,
    part: 'I',
    title: 'The recipe: data, loss, optimizer, generalization',
    tagline: 'Gather data, score the mistakes, walk downhill, and check your work on questions you never studied.',
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
        const NOISE = 0.18, RIDGE = 1e-4, MAXD = 15;
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
            const row = [1]; for (let k = 1; k < d1; k++) row.push(row[k - 1] * u);
            for (let r = 0; r < d1; r++) { b[r] += row[r] * S.ys[i]; for (let c = 0; c < d1; c++) A[r][c] += row[r] * row[c]; }
          }
          for (let r = 0; r < d1; r++) A[r][r] += RIDGE;
          return solve(A, b);
        }
        function evalW(w, x) { const u = (x - 0.5) * 2; let s = 0, pw = 1; for (let k = 0; k < w.length; k++) { s += w[k] * pw; pw *= u; } return s; }
        function mse(w, idx) { let s = 0; for (const i of idx) { const e = evalW(w, S.xs[i]) - S.ys[i]; s += Math.min(16, e * e); } return s / Math.max(1, idx.length); }

        const S = { n: 22, deg: 3, xs: [], ys: [], train: [], held: [], w: [], curveTrain: [], curveVal: [], dirty: true };
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
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          g.fillText('blue = training points   orange = held-out points   grey dashed = true hidden sine   green = fitted degree-' + S.deg + ' polynomial', plot.x, H - 4);
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
          eg.fillStyle = C.green; eg.textAlign = 'right'; eg.fillText('best degree = ' + (best + 1), px + pw, 24);
        }
        const ro = ctx.readout();
        function updateRO() { ro.set({ degree: S.deg, points: S.n, 'train error': S.curveTrain[S.deg - 1] != null ? S.curveTrain[S.deg - 1].toFixed(3) : '–', 'held-out error': S.curveVal[S.deg - 1] != null ? S.curveVal[S.deg - 1].toFixed(3) : '–' }); }
        ctx.loop(() => { if (S.dirty) { draw(); drawErr(); updateRO(); S.dirty = false; } });
        const degSl = ctx.slider({ label: 'polynomial degree', min: 1, max: MAXD, step: 1, value: 3, onChange: (v) => { S.deg = v; recompute(); } });
        const nSl = ctx.slider({ label: 'number of points', min: 8, max: 50, step: 1, value: 22, onChange: (v) => { S.n = v; makeData(); } });
        const resampleBtn = ctx.button('New noise sample', () => makeData());
        const body = h('div', {}, cv, ec);
        return ctx.figure(body, 'Least-squares polynomial fit (normal equations, solved with a tiny ridge term for numerical stability) to noisy samples of a hidden sine. Right: training error (blue) always falls as degree rises; held-out error (red) falls then rises — the U-shaped signature of the bias–variance trade-off.', [degSl, nSl, resampleBtn], ro);
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
          s.trail.push({ x: s.x, y: s.y }); if (s.trail.length > 500) s.trail.shift();
        }
        function draw() {
          g.clearRect(0, 0, W, H);
          g.drawImage(off, plot.x, plot.y, plot.w, plot.h);
          g.strokeStyle = C.line; g.strokeRect(plot.x, plot.y, plot.w, plot.h);
          const mn = toPx(1, 1);
          g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(mn.x - 8, mn.y); g.lineTo(mn.x + 8, mn.y); g.moveTo(mn.x, mn.y - 8); g.lineTo(mn.x, mn.y + 8); g.stroke();
          g.fillStyle = '#fff'; g.font = FONT; g.textAlign = 'left'; g.fillText('minimum', mn.x + 10, mn.y - 8);
          OPT.forEach((o, oi) => {
            const s = S[oi];
            g.strokeStyle = o.color; g.lineWidth = 1.6; g.globalAlpha = 0.7; g.beginPath();
            s.trail.forEach((pt, i) => { const q = toPx(pt.x, pt.y); if (!i) g.moveTo(q.x, q.y); else g.lineTo(q.x, q.y); }); g.stroke(); g.globalAlpha = 1;
            const q = toPx(s.x, s.y);
            g.beginPath(); g.arc(q.x, q.y, 7, 0, Math.PI * 2); g.fillStyle = o.color; g.fill(); g.strokeStyle = '#0a0e16'; g.lineWidth = 1.5; g.stroke();
          });
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
        const sgdSl = ctx.slider({ label: 'SGD learning rate', min: 0.0005, max: 0.03, step: 0.0005, value: 0.0035, digits: 4, onChange: (v) => { OPT[0].lr = v; } });
        const momSl = ctx.slider({ label: 'Momentum learning rate', min: 0.0005, max: 0.03, step: 0.0005, value: 0.0035, digits: 4, onChange: (v) => { OPT[1].lr = v; } });
        const adamSl = ctx.slider({ label: 'Adam learning rate', min: 0.01, max: 0.4, step: 0.01, value: 0.09, digits: 2, onChange: (v) => { OPT[2].lr = v; } });
        const playBtn = ctx.button('▶ Start', () => { playing = !playing; playBtn.textContent = playing ? '⏸ Pause' : '▶ Start'; }, 'primary');
        const resetBtn = ctx.button('Reset', () => { reset(); playing = false; playBtn.textContent = '▶ Start'; });
        return ctx.figure(cv, 'Three optimizers descending the same curved valley (a tamed Rosenbrock surface; white cross marks the true minimum). SGD (grey) fights the curvature every step; momentum (yellow) builds speed but overshoots the valley floor; Adam (green) rescales every direction by its own gradient history and glides down with far less zig-zag.', [sgdSl, momSl, adamSl, playBtn, resetBtn]);
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
            g.strokeStyle = C.green; g.setLineDash([4, 3]); g.lineWidth = 1.3; g.beginPath(); g.moveTo(p.x, plot.y); g.lineTo(p.x, plot.y + plot.h); g.stroke(); g.setLineDash([]);
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

      /* ================================================================== */
      /* Prose                                                              */
      /* ================================================================== */
      root.append(
        p(`A friend crams for a driving-theory test by memorising last year's answer key, word for word. She scores 100% on that exact test. This year's test has the same road rules but different questions, and she fails badly. She never learned to drive; she learned an answer key. Every machine-learning model is one bad decision away from doing exactly the same thing, and this chapter is the discipline that stops it.`),
        p(`Chapter 1 built a single neuron; chapter 2 stacked neurons into a network trained by backpropagation. Neither asked the question every real project lives or dies by: how do you know the model will work on data it has never seen? A model that scores perfectly on training data and falls apart on anything new is worthless in practice, and the gap between those two numbers is arguably the single most important number in machine learning.`),
        p(`This chapter is the recipe every model follows, from a two-line perceptron to a 500-billion-parameter language model: gather <b>data</b>, choose a <b>model</b> with adjustable <b>parameters</b>, define a <b>loss</b> that scores how wrong it is, pick an <b>optimizer</b> to nudge the parameters downhill, and <b>evaluate</b> on data the model never trained on. Get the recipe right and the model generalises to the real world. Get it wrong and you've built an expensive answer key.`),

        section('The recipe every model follows',
          p(`Strip away the specifics of image recognition, language models, game-playing agents and fraud detection, and almost every machine-learning project underneath is running the same five-step loop.`),
          ol([
            `<b>Data</b> — a pile of examples, each with inputs and (usually) a correct answer: photos with labels, sentences with the next word blanked out, board positions with the eventual winner. Garbage in the data becomes garbage the model confidently repeats.`,
            `<b>Model with parameters</b> — a function with knobs (weights) that can be turned. A single neuron has a handful of knobs; a frontier language model has hundreds of billions. More knobs, more shapes the function can bend into.`,
            `<b>Loss</b> — one number that says how wrong the current knob-settings are, by comparing predictions to correct answers. Chapter 2 covered mean-squared error and cross-entropy.`,
            `<b>Optimizer</b> — the rule that turns the loss's slope into a knob adjustment. Plain gradient descent is the simplest version; the second half of this chapter is about the better versions everyone actually uses.`,
            `<b>Evaluation</b> — checking the result on examples the optimizer never touched, because a model graded only on its own homework will always look better than it really is.`,
          ]),
          p(`Everything else in this course — convolutions, attention, transformers, reinforcement learning from human feedback — is really a smarter version of step 2. Steps 1, 3, 4 and 5 barely change. Learn this chapter well and you can read the methods section of almost any modern AI paper.`),
        ),

        section('Split your data before you touch a model',
          p(`Cut your labelled data into three separate piles before writing a line of training code. The <em>training set</em> is what the optimizer actually sees and adjusts weights against. The <em>validation set</em> is what you check progress against while still making decisions — how many layers, which learning rate, when to stop — because those choices are themselves a form of fitting, and repeatedly peeking at your final judge quietly cheats on it. The <em>test set</em> is touched exactly once, at the end, and its score is the only honest estimate of how the model will do in the world.`),
          p(`A common split for 1,000 examples is 70/15/15: 700 for training, 150 for validation, 150 for test. Train on the 700, tune dozens of choices against the 150 validation examples, then score once on the untouched 150 test examples — that final number can be trusted. Score on the training set instead, and the model is grading its own homework. Re-use the validation set as your reported result after fifty rounds of tuning against it, and validation has quietly become a second training set.`),
        ),

        callout('tryit', 'Try it: overfit a curve on purpose', `Below, some noisy points are drawn from a hidden sine wave you can't see directly. <b>1.</b> Set the polynomial degree to 1: a straight line, badly wrong almost everywhere — <em>underfitting</em>. <b>2.</b> Push the degree to 15: the curve now threads almost exactly through every blue training point, training error near zero — but watch the orange held-out points and the wild swings in between them. <b>3.</b> Find the degree with the lowest held-out error in the right-hand chart; it's usually 3–6, close to how curvy the true sine actually is. <b>4.</b> At degree 15, press "New noise sample" a few times and watch the fitted curve reshape itself completely, even though the true sine never moves — that instability is what statisticians call <em>variance</em>.`),
        curveFit(),

        section('Overfitting and underfitting: memorising vs understanding',
          p(`The exam-crammer from the opening paragraph has a name in machine learning: <em>overfitting</em>. The model has enough flexibility — enough parameters, high enough polynomial degree — to fit the training data's noise as well as its signal, and noise never repeats, so performance on new data suffers even while training performance looks flawless. Its mirror image is <em>underfitting</em>: a model too rigid, a straight line facing a curved reality, to capture even the signal, so it does poorly on training data and new data alike.`),
          p(`Statisticians call the two failure modes bias and variance. <b>Bias</b> is being systematically wrong the same way regardless of which training data you drew — a straight line will always miss a curve, however much data you feed it. <b>Variance</b> is being unstable: a small change in exactly which points you sampled swings the fitted model wildly, which is exactly what the degree-15 polynomial did the moment you resampled the noise. Simple models are high-bias, low-variance; flexible models are low-bias, high-variance. The sweet spot minimises both combined — why the held-out error above forms a U: too rigid on the left, too twitchy on the right, best in the middle.`),
        ),

        callout('example', 'Overfitting in the wild', `A résumé-screening model trained on 200 successful hires from one company can reach 99% training accuracy by memorising which postcodes and college names happened to correlate with success in that one small batch — spurious correlations, not job skill. On the next batch of candidates it performs close to random, and it does so confidently. This is why serious ML teams report validation and test numbers, never training numbers, and why a vendor's "99.9% on our data" should always prompt the follow-up: measured how, on which held-out set?`),

        section('Fighting overfitting: regularization',
          p(`Four practical levers push back against overfitting, and you'll meet all four again in every chapter from here on.`),
          ul([
            `<b>Weight decay</b> — add a small penalty to the loss for weights with large magnitude, so the optimizer keeps them small unless the data really insists otherwise. Small weights make smoother functions; smoother functions memorise noise less easily. (The ridge term quietly stabilising the polynomial fit above is exactly this.)`,
            `<b>Dropout</b> — during training, randomly switch off a fraction of neurons on every pass, so the survivors can't rely on one teammate always being present. The network is forced to learn redundant, robust features instead of memorising with a few overspecialised units.`,
            `<b>Early stopping</b> — watch the validation loss during training and stop the moment it starts climbing, even though training loss is still falling. Ship the model with the best validation score, not the most-trained one.`,
            `<b>More (and cleaner) data</b> — the single most reliable fix. A model can't memorise noise it hasn't seen enough of; more genuinely varied examples dilute any one quirk. It's why frontier labs now spend more effort curating data than tuning architecture.`,
          ]),
          p(`All four do the same underlying thing: make it harder for the model to fit noise without making it harder to fit signal. None are free — overdo weight decay or dropout and you're back to underfitting.`),
        ),

        callout('tryit', 'Try it: catch the exact moment of overfitting', `Press ▶ Play. Training loss (blue) keeps falling — the model can always get a little better at its own homework. Validation loss (red) falls too, then turns upward: that turn is the model starting to memorise instead of understand. <b>1.</b> Let it run and find the green dot, the best validation epoch. <b>2.</b> Set training set size to its smallest option and regularization to zero — the turn arrives early and steeply. <b>3.</b> Push training set size to its largest option and regularization to maximum — the turn arrives much later, if at all, inside 60 epochs. <b>4.</b> Press Replay a few times at the same settings: the exact curve jitters, but the story — training always improving, validation eventually not — repeats every time.`),
        earlyStopping(),

        section('Optimizers: from downhill walking to Adam',
          p(`Plain gradient descent (chapter 2) takes a step of size η in the downhill direction, every time. Two problems appear once real networks are involved: some directions of the loss landscape are steep and others nearly flat — picture a narrow, curved valley rather than a smooth round bowl — and every one of a million parameters would ideally like its own sensible step size, which no single global η can give them all at once.`),
          sub('Momentum: the heavy ball',
            p(`<em>Momentum</em> answers the first problem with a change of metaphor. Instead of a marble that stops dead the instant you stop pushing it, imagine a heavy ball rolling downhill: it keeps some of its previous velocity and adds the new gradient on top — v ← β·v − η·∇L, then position ← position + v, with β around 0.9 (keep 90% of last step's velocity). Down a narrow valley the ball builds speed along the valley floor, while the side-to-side component — which flips sign every step under plain gradient descent, causing zig-zag — partially cancels out over successive steps instead of whipping the ball sideways.`),
            p(`Worked example: suppose the previous velocity was v = −0.40 and the new gradient is 0.30, with β = 0.9 and η = 0.1. The new velocity is v = 0.9 × (−0.40) − 0.1 × 0.30 = <b>−0.39</b>, barely changed from before — the ball remembers where it was heading and shrugs off one noisy gradient.`),
          ),
          sub('Adam: a personal step size for every parameter',
            p(`<em>Adam</em> (Adaptive Moment Estimation) answers the second problem. For each parameter it keeps a running average of the gradient (direction, like momentum) and a running average of the gradient's <i>squared</i> size (how large or noisy that parameter's gradients tend to be), then divides the first by the square root of the second. A parameter with consistently huge gradients gets shrunk to a sane step; a parameter with tiny, rare gradients gets amplified to a useful one. Every parameter effectively gets its own learning rate, chosen automatically from its own history.`),
            p(`A neat fact falls out of the algebra: on the very first update, before the running averages build up, Adam's bias-correction makes the step almost exactly η × sign(gradient) — regardless of the gradient's actual size. Plain SGD's first step, by contrast, is η × gradient, so one unusually large gradient in a random batch throws the update wildly off course. That's one reason Adam tolerates a sloppier learning-rate choice than SGD — and why, below, the Adam ball keeps its footing across a much wider slider range than the other two.`),
          ),
          p(`<em>AdamW</em> is Adam with one fix: weight decay applied directly to the weights, decoupled from the adaptive gradient averaging, instead of folded into the gradient the way naive implementations did. That decoupling turned out to matter for how well the regularization actually regularizes, and AdamW — not plain Adam — is the optimizer behind the overwhelming majority of transformer training runs today, including the large language models in Part III of this course.`),
        ),

        callout('history', 'Adam (2014), dropout (2012/2014), AdamW (2017)', `<em>Adam</em> was introduced by Diederik Kingma and Jimmy Ba in 2014 ("Adam: A Method for Stochastic Optimization"), combining ideas from momentum with an earlier adaptive method called RMSProp. <em>Dropout</em> arrived a little earlier from Geoffrey Hinton's group — first described by Hinton and colleagues in 2012, then formalised by Nitish Srivastava, Hinton, Alex Krizhevsky, Ilya Sutskever and Ruslan Salakhutdinov in a 2014 JMLR paper — inspired, Hinton has said, by how a bank foils fraud by shuffling which teller serves which customer, so no single employee's habits can be relied on to make a swindle work. <em>AdamW</em> followed in 2017 from Ilya Loshchilov and Frank Hutter ("Decoupled Weight Decay Regularization"): a small algebraic fix that measurably improved final model quality, and is now the default optimizer in essentially every serious training codebase.`),

        callout('tryit', 'Try it: watch three optimizers cross the same valley', `The background is a curved valley loss surface (a tamed Rosenbrock function); darker is lower loss, and the white cross marks the true minimum. All three balls start from the same spot. <b>1.</b> Press ▶ Start and watch plain SGD (grey) crawl — it has to fight the valley's curvature at every step. <b>2.</b> Watch momentum (yellow) build up speed but overshoot the valley floor and wobble before settling. <b>3.</b> Watch Adam (green) glide down with far less zig-zag. <b>4.</b> Push the SGD or momentum learning-rate slider up until its ball starts bouncing off the walls; Adam tolerates a much wider range of its own learning rate before doing the same. <b>5.</b> Press Reset and try a small Adam learning rate: slower, but still steady, never wild.`),
        optimizerRace(),

        section('Batching, and why GPUs love it',
          p(`Real training almost never computes the gradient from one example at a time (that's what chapter 2's playground did for clarity), nor from the entire dataset at once (too slow to ever take a step). Instead it averages the gradient over a <em>mini-batch</em> — commonly 32 to a few thousand examples — takes one step, then moves to the next batch. A batch is small enough to give frequent, useful updates and large enough that its average gradient is a decent stand-in for the true one.`),
          p(`GPUs love batching because a GPU's advantage was never doing one multiplication fast — it's doing thousands of independent multiplications at the same instant. Feed it one example and most of the chip idles; feed it a batch of 512 stacked into one matrix, and the same matrix multiply that handled one example now handles all 512 for barely more time, because the chip was built to do exactly that in parallel. Bigger batches, up to a point, are almost free on a GPU and expensive on a CPU — one quiet reason deep learning didn't take off until GPUs were repurposed for it (chapter 4).`),
        ),

        section('Parameters vs hyperparameters',
          p(`One distinction is worth nailing down, because it comes up in every training run you'll ever read about. <b>Parameters</b> are the numbers the optimizer learns — weights, biases, everything adjusted by ∂L/∂w. <b>Hyperparameters</b> are the numbers a human (or a search script) sets before training starts and the optimizer never touches: learning rate, batch size, number of layers, dropout rate, the polynomial degree in the demo above. You tune hyperparameters by watching the validation set, as described earlier; you learn parameters by gradient descent on the training set. There is no ∂L/∂(learning rate) to compute — confusing the two is a classic beginner's mistake.`),
        ),

        section('Four ways a model can learn',
          p(`Every example in this course so far has come with a correct answer already attached — a label, a target value, a target class. That's only one of four fundamentally different ways a model can learn.`),
          cards([
            { title: 'Supervised', body: 'Every example is labelled by a human or an existing system: photo → "cat", house features → sale price. Precise, but labelling at scale is expensive.' },
            { title: 'Unsupervised', body: 'No labels at all — the model finds structure in the data itself, grouping similar customers into clusters or compressing 1,000 numbers into 10 that still capture most of the pattern. Nobody defines "similar"; the data\'s own geometry does.' },
            { title: 'Self-supervised', body: 'The labels are manufactured from the data itself, for free: hide a word and predict it, hide part of an image and fill it in. No human labeller required — unlimited raw text or images become an unlimited supply of exercises.' },
            { title: 'Reinforcement learning', body: "There's no answer table at all. An agent takes actions in an environment and gets a reward sometime later, and must work out which earlier action deserves the credit. Chapter 12 covers this in depth." },
          ]),
          p(`Here's the punchline that resolves a common confusion: when you read that GPT-style models are "pretrained on the internet with no labels", that's self-supervised learning wearing a supervised-learning engine underneath. Every "predict the next word" example carries a free label — the actual next word — manufactured automatically from ordinary text, then trained with exactly the cross-entropy loss and backpropagation from chapter 2. Self-supervised learning is what let language models train on essentially the whole internet instead of waiting for a hand-labelled dataset that could never be big enough.`),
        ),

        section('Data quality beats data quantity',
          p(`More data helps, but not all data is worth the same amount. Ten thousand carefully checked, genuinely diverse examples routinely beat a million scraped, duplicated, mislabeled ones, because the optimizer cannot tell a genuine pattern from a systematic error in the labels — it fits both with equal enthusiasm.`),
          callout('example', 'Curation beats scale', `Early large language model training sets went through aggressive filtering and deduplication before a single weight was updated; simply removing near-duplicate documents measurably improved the resulting model, because duplicated text taught the model to overweight whatever happened to be copy-pasted often across the web. Later, smaller-but-carefully-curated model families showed the same lesson from the other direction: a modestly sized model trained on well-chosen data can beat a much larger model trained carelessly. "Bigger dataset" and "better dataset" are different axes, and the second is usually the cheaper one to pull.`),
        ),

        section('Grading the model: evaluation metrics',
          p(`"Evaluate the model" hides a real design choice: what, exactly, are you measuring? <em>Accuracy</em> — the fraction of predictions that are correct — is the obvious first metric, and the most misleading one whenever the classes are unbalanced.`),
          p(`Say 1 in 20 emails is spam. A lazy model predicting "not spam" for everything scores 95% accuracy while catching zero spam — accuracy alone hides a useless model. Two sharper questions: of the emails flagged as spam, how many really were (<em>precision</em>)? Of all the actual spam, how many did it catch (<em>recall</em>)? Concretely: out of 100 emails, 20 are truly spam. The model flags 25; 18 of those really are spam, 7 good emails are wrongly flagged, and 2 real spam slip through. Precision = 18/25 = <b>72%</b>. Recall = 18/20 = <b>90%</b>. A filter tuned for high recall catches nearly everything but annoys users with false alarms; one tuned for high precision rarely bothers a real email but lets more spam through. Which matters depends on what a mistake costs.`),
          p(`Language models use a metric built directly from the cross-entropy loss: <em>perplexity</em>, roughly e raised to the average per-token loss. A cross-entropy of 2.3 nats gives a perplexity of about 10 — intuitively, "the model is about as unsure as if it had to guess uniformly among 10 equally likely next words." Lower is better; a well-trained large language model's perplexity on ordinary text sits in the single digits. Chapter 9 puts this number to proper use.`),
        ),

        section('Why this matters for modern AI',
          p(`Every headline about a new frontier model is this chapter's recipe, run at a scale a 2012 researcher would have called science fiction: petabytes of self-supervised data standing in for hand-made labels, hundreds of billions of parameters for the model, cross-entropy for the loss, AdamW for the optimizer, and a held-out benchmark — never the training data — standing in for the report card anyone trusts. The training-versus-validation gap you watched turn upward in the early-stopping demo is the same gap researchers track at the scale of trillions of tokens; a striking share of frontier-lab effort goes into data curation and regularization, exactly the levers in this chapter, because the recipe never changes — only its size.`),
          p(`One picture to keep: <b>the loss tells you how wrong you are, the optimizer tells you which way to move, and the validation set is the only honest judge of whether any of it generalises.</b>`),
        ),

        ctx.quiz([
          { q: 'You tune your model\'s hyperparameters by repeatedly checking performance against the same held-out set, then report that set\'s score as your final result. What\'s wrong?', options: ['Nothing — that\'s exactly what validation sets are for', 'You\'ve quietly turned your validation set into a second training set, so the reported score is optimistic', 'Hyperparameters should be tuned on the training set instead', 'The test set should have been used for tuning, not validation'], answer: 1, explain: 'Repeatedly optimizing decisions against one dataset lets you fit its quirks too, even unintentionally. A test set touched exactly once, at the very end, is the only honest final number.' },
          { q: 'A model gets 99% training accuracy and 61% validation accuracy. What does this pattern usually mean?', options: ['Underfitting — the model is too simple', 'Overfitting — the model memorised training-set noise instead of the general pattern', 'The optimizer\'s learning rate is far too low', 'The loss function was chosen incorrectly'], answer: 1, explain: 'A wide train/validation gap, with training performance near-perfect, is the classic signature of overfitting: high variance, not high bias.' },
          { q: 'In the momentum update v ← β·v − η·∇L, what does a larger β (closer to 1) do?', options: ['Makes the optimizer forget its previous direction faster', 'Makes the ball remember more of its previous velocity, smoothing out zig-zags across a narrow valley', 'Increases the learning rate directly', 'Turns momentum back into plain SGD'], answer: 1, explain: 'β is how much of the previous velocity survives each step. Close to 1, the ball keeps rolling in roughly its old direction and side-to-side gradients partially cancel out over time.' },
          { q: 'A spam filter flags very little as spam, but almost everything it does flag really is spam. That describes:', options: ['High recall, low precision', 'High precision, low recall', 'High accuracy, undefined precision', 'Overfitting'], answer: 1, explain: 'Flagging rarely but being right when it does is the signature of high precision (few false positives) at the cost of recall (missing spam it never flagged).' },
          { q: 'Which of these is a hyperparameter, not a parameter?', options: ['A specific weight connecting two neurons', 'A bias term learned during training', 'The learning rate used by the optimizer', 'The value the network predicts for one input'], answer: 2, explain: 'Weights and biases are learned by gradient descent — they are parameters. The learning rate is set by a human before training starts and the optimizer never adjusts it; that makes it a hyperparameter.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://arxiv.org/abs/1412.6980" target="_blank" rel="noopener">Kingma &amp; Ba (2014), "Adam: A Method for Stochastic Optimization"</a>: the original paper, short and readable.`,
            `<a href="https://jmlr.org/papers/v15/srivastava14a.html" target="_blank" rel="noopener">Srivastava, Hinton, Krizhevsky, Sutskever &amp; Salakhutdinov (2014), "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"</a>, JMLR.`,
            `<a href="https://arxiv.org/abs/1711.05101" target="_blank" rel="noopener">Loshchilov &amp; Hutter, "Decoupled Weight Decay Regularization"</a>: the AdamW paper.`,
            `<a href="https://distill.pub/2017/momentum/" target="_blank" rel="noopener">Distill.pub, "Why Momentum Really Works"</a>: an interactive visual explanation of the heavy-ball intuition.`,
            `<a href="https://developers.google.com/machine-learning/guides/rules-of-ml" target="_blank" rel="noopener">Google, "Rules of Machine Learning"</a>: hard-won, practical lessons on data quality, evaluation and when to bother with a fancier model at all.`,
          ]),
        ),
      );
    },
  });
})();
