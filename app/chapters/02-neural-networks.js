/* Zero → AGI · Chapter 02 · Neural networks and backpropagation
   Layers fix XOR; activation functions; loss; gradient descent; backprop = chain rule.
   Interactives: live MLP training playground (forward + backprop by hand, SGD, heatmap boundary,
   loss curve); 1-D gradient descent on a bumpy loss; backprop signal-flow animation. */
(function () {
  ZTA.registerChapter({
    id: '02-neural-networks',
    num: 2,
    part: 'I',
    title: 'Neural networks and backpropagation',
    tagline: 'Stack neurons, measure how wrong you are, and let the chain rule tell every weight which way to move.',
    render(root, ctx) {
      const { h, p, section, callout, ul, ol } = ctx;
      const C = ctx.colors;
      const FONT = '13px Inter, system-ui, sans-serif';
      const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
      const f2 = (v) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(2);
      const f3 = (v) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(3);
      const f4 = (v) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(4);
      const sig = (z) => 1 / (1 + Math.exp(-z));
      /* split `text` into lines that fit inside maxW pixels of the given 2-D context */
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
        const lines = wrapLines(gc, text, maxW);
        lines.forEach((ln, i) => gc.fillText(ln, x, y + i * lh));
        return lines.length;
      }

      /* ------------------------------------------------------------------ */
      /* Interactive A: the live training playground                          */
      /* ------------------------------------------------------------------ */
      function playground() {
        const W = 720, H = 380;
        const [cv, g] = ctx.canvas(W, H);
        const [lc, lg] = ctx.canvas(720, 130);
        const plot = { x: 10, y: 10, s: 360 };
        const GRID = 72;
        const off = document.createElement('canvas'); off.width = GRID; off.height = GRID;
        const og = off.getContext('2d'); const img = og.createImageData(GRID, GRID);
        const ACTS = {
          tanh: { f: Math.tanh, d: (z, a) => 1 - a * a },
          relu: { f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0) },
          sigmoid: { f: sig, d: (z, a) => a * (1 - a) },
        };
        const S = { ds: 'xor', N: 4, lr: Math.pow(10, -1.3), act: 'tanh', speed: 100, playing: false, showLines: true, data: [], epoch: 0, loss: NaN, acc: 0, hist: [], msg: '', W1: [], b1: [], W2: [], b2: 0, accLoss: 0, accCount: 0, accCorrect: 0, sampleIdx: 0, order: [], dirty: true, heatAge: 9, heatDone: false };
        const probe = { on: false, x: 0, y: 0 };
        const z = new Array(16).fill(0), a = new Array(16).fill(0);
        const clipG = (v) => (v > 5 ? 5 : v < -5 ? -5 : v);

        function makeData() {
          const d = [], n = 200;
          if (S.ds === 'xor') {
            for (let i = 0; i < n; i++) {
              let x = ctx.rand(-1, 1), y = ctx.rand(-1, 1);
              if (Math.abs(x) < 0.08) x += (x < 0 ? -0.08 : 0.08);
              if (Math.abs(y) < 0.08) y += (y < 0 ? -0.08 : 0.08);
              d.push({ x, y, l: x * y > 0 ? 1 : 0 });
            }
          } else if (S.ds === 'circles') {
            for (let i = 0; i < n; i++) {
              const inner = i % 2 === 0, r = inner ? ctx.rand(0, 0.42) : ctx.rand(0.62, 0.95), t = ctx.rand(0, Math.PI * 2);
              d.push({ x: r * Math.cos(t), y: r * Math.sin(t), l: inner ? 1 : 0 });
            }
          } else if (S.ds === 'spiral') {
            const m = n / 2;
            for (let k = 0; k < m; k++) for (let l = 0; l < 2; l++) {
              const r = 0.1 + 0.85 * k / m, t = 3 * Math.PI * k / m + l * Math.PI;
              d.push({ x: r * Math.cos(t) + ctx.randn() * 0.035, y: r * Math.sin(t) + ctx.randn() * 0.035, l });
            }
          } else {
            for (let i = 0; i < n; i++) {
              const l = i % 2, cx = l ? 0.5 : -0.5, cy = l ? 0.5 : -0.5;
              d.push({ x: cx + ctx.randn() * 0.28, y: cy + ctx.randn() * 0.28, l });
            }
          }
          for (const q of d) { q.x = ctx.clamp(q.x, -1, 1); q.y = ctx.clamp(q.y, -1, 1); }
          S.data = d;
        }
        function shuffleOrder() {
          const o = S.data.map((_, i) => i);
          for (let i = o.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [o[i], o[j]] = [o[j], o[i]]; }
          S.order = o;
        }
        function reinit(msg) {
          S.W1 = []; S.b1 = []; S.W2 = [];
          const s1 = S.act === 'relu' ? 1.0 : 0.9;
          for (let j = 0; j < S.N; j++) { S.W1.push([ctx.randn() * s1, ctx.randn() * s1]); S.b1.push(ctx.randn() * 0.3); S.W2.push(ctx.randn() * Math.sqrt(1 / S.N)); }
          S.b2 = 0; S.epoch = 0; S.hist = []; S.loss = NaN; S.acc = 0;
          S.accLoss = 0; S.accCount = 0; S.accCorrect = 0; S.sampleIdx = 0; shuffleOrder();
          S.msg = msg || ''; S.dirty = true; S.heatStale = true;
        }
        function forward(x, y) {
          const act = ACTS[S.act].f; let o = S.b2;
          for (let j = 0; j < S.N; j++) { const w = S.W1[j]; o += S.W2[j] * act(w[0] * x + w[1] * y + S.b1[j]); }
          return sig(o);
        }
        function endEpoch() {
          S.epoch++;
          S.loss = S.accLoss / Math.max(1, S.accCount); S.acc = S.accCorrect / Math.max(1, S.accCount);
          S.hist.push({ e: S.epoch, loss: S.loss, acc: S.acc });
          if (S.hist.length > 1200) S.hist = S.hist.filter((_, i) => i % 2 === 0);
          S.accLoss = 0; S.accCount = 0; S.accCorrect = 0; S.sampleIdx = 0; shuffleOrder();
          let bad = !isFinite(S.loss);
          for (let j = 0; j < S.N && !bad; j++) {
            S.W1[j][0] = ctx.clamp(S.W1[j][0], -20, 20); S.W1[j][1] = ctx.clamp(S.W1[j][1], -20, 20); S.b1[j] = ctx.clamp(S.b1[j], -20, 20); S.W2[j] = ctx.clamp(S.W2[j], -20, 20);
            if (!isFinite(S.W1[j][0] + S.W1[j][1] + S.b1[j] + S.W2[j])) bad = true;
          }
          S.b2 = ctx.clamp(S.b2, -20, 20);
          if (bad) reinit('Training diverged (loss became non-finite). Weights reset — try a lower learning rate.');
        }
        function trainSamples(k) {
          const n = S.data.length; if (!n) return;
          const act = ACTS[S.act].f, dact = ACTS[S.act].d, lr = S.lr, N = S.N;
          for (let s = 0; s < k; s++) {
            const d = S.data[S.order[S.sampleIdx++]];
            let o = S.b2;
            for (let j = 0; j < N; j++) { const w = S.W1[j]; z[j] = w[0] * d.x + w[1] * d.y + S.b1[j]; a[j] = act(z[j]); o += S.W2[j] * a[j]; }
            const pr = sig(o);
            S.accLoss += -(d.l ? Math.log(pr + 1e-9) : Math.log(1 - pr + 1e-9)); S.accCount++;
            if ((pr > 0.5) === (d.l === 1)) S.accCorrect++;
            const dO = pr - d.l;                      // ∂L/∂o for sigmoid + cross-entropy
            for (let j = 0; j < N; j++) {
              const dZ = dO * S.W2[j] * dact(z[j], a[j]); // chain rule back through the hidden unit
              S.W2[j] -= lr * clipG(dO * a[j]);
              S.W1[j][0] -= lr * clipG(dZ * d.x); S.W1[j][1] -= lr * clipG(dZ * d.y); S.b1[j] -= lr * clipG(dZ);
            }
            S.b2 -= lr * clipG(dO);
            if (S.sampleIdx >= n) endEpoch();
          }
          S.dirty = true; S.heatStale = true;
        }
        /* Recompute the 72×72 prediction image into the offscreen canvas. ~0.8 ms at N = 16, so it
           is throttled to 20 Hz while training (the blit below is free) and recomputed immediately
           whenever the reader changes something while paused. */
        function computeHeat() {
          const act = ACTS[S.act].f, N = S.N, W1 = S.W1, b1 = S.b1, W2 = S.W2, b2 = S.b2, px = img.data;
          let k = 0;
          for (let gy = 0; gy < GRID; gy++) {
            const yy = 1 - (gy + 0.5) / GRID * 2;
            for (let gx = 0; gx < GRID; gx++) {
              const xx = -1 + (gx + 0.5) / GRID * 2;
              let o = b2;
              for (let j = 0; j < N; j++) { const w = W1[j]; o += W2[j] * act(w[0] * xx + w[1] * yy + b1[j]); }
              let v = 2 / (1 + Math.exp(-o)) - 1;
              if (!(v >= -1 && v <= 1)) v = 0;          // NaN / ±Infinity guard
              const al = Math.abs(v) * 0.8;
              const r = v > 0 ? 251 : 124, gg = v > 0 ? 113 : 156, b = v > 0 ? 133 : 255;
              px[k++] = 15 + (r - 15) * al; px[k++] = 21 + (gg - 21) * al; px[k++] = 32 + (b - 32) * al; px[k++] = 255;
            }
          }
          og.putImageData(img, 0, 0);
          S.heatDone = true; S.heatStale = false; S.heatAge = 0;
        }
        function drawHeat() {
          if (!S.heatDone || (S.heatStale && (!S.playing || S.heatAge >= 0.05))) computeHeat();
          g.imageSmoothingEnabled = true;
          g.drawImage(off, plot.x, plot.y, plot.s, plot.s);
        }
        const toPx = (x, y) => ({ x: plot.x + (x + 1) / 2 * plot.s, y: plot.y + (1 - y) / 2 * plot.s });
        function draw() {
          g.clearRect(0, 0, W, H);
          drawHeat();
          if (S.showLines) {
            g.save(); g.beginPath(); g.rect(plot.x, plot.y, plot.s, plot.s); g.clip();
            g.strokeStyle = C.text; g.globalAlpha = 0.3; g.lineWidth = 1;
            for (let j = 0; j < S.N; j++) {
              const w0 = S.W1[j][0], w1 = S.W1[j][1], b = S.b1[j], n2 = w0 * w0 + w1 * w1;
              if (n2 < 1e-9) continue;
              const p0 = { x: -b * w0 / n2, y: -b * w1 / n2 }, dx = -w1 / Math.sqrt(n2), dy = w0 / Math.sqrt(n2);
              const A = toPx(p0.x - 4 * dx, p0.y - 4 * dy), B = toPx(p0.x + 4 * dx, p0.y + 4 * dy);
              g.beginPath(); g.moveTo(A.x, A.y); g.lineTo(B.x, B.y); g.stroke();
            }
            g.restore(); g.globalAlpha = 1;
          }
          for (const d of S.data) {
            const s = toPx(d.x, d.y);
            g.beginPath(); g.arc(s.x, s.y, 3.6, 0, Math.PI * 2); g.fillStyle = d.l ? C.danger : C.accent; g.fill();
            g.lineWidth = 1; g.strokeStyle = '#0a0e16'; g.stroke();
          }
          g.strokeStyle = C.line; g.strokeRect(plot.x, plot.y, plot.s, plot.s);
          // network diagram on the right
          const nx = [430, 560, 690], inY = [140, 240], outY = 190;
          const hidY = (j) => 28 + (j + 0.5) * (324 / S.N);
          let maxW = 0.5;                                   // no allocation, and non-finite-safe
          for (let j = 0; j < S.N; j++) {
            const m = Math.max(Math.abs(S.W1[j][0]), Math.abs(S.W1[j][1]), Math.abs(S.W2[j]));
            if (m > maxW && isFinite(m)) maxW = m;
          }
          const edgeStyle = (w) => { g.strokeStyle = w > 0 ? C.danger : C.accent; g.lineWidth = 0.6 + 4 * Math.min(1, Math.abs(w) / maxW); g.globalAlpha = 0.35 + 0.6 * Math.min(1, Math.abs(w) / maxW); };
          for (let j = 0; j < S.N; j++) {
            for (let i = 0; i < 2; i++) { edgeStyle(S.W1[j][i]); g.beginPath(); g.moveTo(nx[0], inY[i]); g.lineTo(nx[1], hidY(j)); g.stroke(); }
            edgeStyle(S.W2[j]); g.beginPath(); g.moveTo(nx[1], hidY(j)); g.lineTo(nx[2], outY); g.stroke();
          }
          g.globalAlpha = 1;
          const node = (x, y, r, label) => { g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fillStyle = '#111827'; g.fill(); g.strokeStyle = C.text; g.lineWidth = 1.5; g.stroke(); if (label) { g.fillStyle = C.text; g.font = MONO; g.textAlign = 'center'; g.fillText(label, x, y + 4); } };
          node(nx[0], inY[0], 13, 'x'); node(nx[0], inY[1], 13, 'y');
          const hr = Math.min(11, 150 / S.N);
          for (let j = 0; j < S.N; j++) node(nx[1], hidY(j), hr, null);
          node(nx[2], outY, 15, 'ŷ');
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'center';
          g.fillText('inputs', nx[0], 14); g.fillText('hidden (' + S.N + ', ' + S.act + ')', nx[1], 14); g.fillText('output', nx[2], 14);
          g.fillText('red edge = positive weight, blue = negative, thickness = size', 560, H - 8);
          if (!S.hist.length) { g.fillStyle = C.text; g.font = FONT; g.textAlign = 'center'; g.fillText('press ▶ Play to train', plot.x + plot.s / 2, plot.y + 22); }
          // probe: what does the network predict under the reader's finger?
          if (probe.on) {
            const s = toPx(probe.x, probe.y), pr = forward(probe.x, probe.y);
            g.save(); g.beginPath(); g.rect(plot.x, plot.y, plot.s, plot.s); g.clip();
            g.strokeStyle = C.text; g.globalAlpha = 0.55; g.lineWidth = 1;
            g.beginPath(); g.moveTo(s.x - 9, s.y); g.lineTo(s.x + 9, s.y); g.moveTo(s.x, s.y - 9); g.lineTo(s.x, s.y + 9); g.stroke();
            g.globalAlpha = 1;
            const txt = 'ŷ = ' + f2(pr) + '  →  ' + (pr > 0.5 ? 'class 1' : 'class 0');
            g.font = MONO; const bw = g.measureText(txt).width + 14;
            const bx = ctx.clamp(s.x + 12, plot.x + 2, plot.x + plot.s - bw - 2), by = ctx.clamp(s.y - 26, plot.y + 2, plot.y + plot.s - 22);
            g.fillStyle = 'rgba(10,14,22,0.88)'; g.fillRect(bx, by, bw, 20);
            g.strokeStyle = C.line; g.strokeRect(bx, by, bw, 20);
            g.fillStyle = pr > 0.5 ? C.danger : C.accent; g.textAlign = 'left'; g.fillText(txt, bx + 7, by + 14);
            g.restore();
          }
          // divergence / status message, inside the plot so it can never collide with the diagram
          if (S.msg) {
            g.font = FONT; g.textAlign = 'left';
            const lines = wrapLines(g, S.msg, plot.s - 20);
            const bh = lines.length * 15 + 10, by = plot.y + plot.s - bh - 6;
            g.fillStyle = 'rgba(10,14,22,0.9)'; g.fillRect(plot.x + 5, by, plot.s - 10, bh);
            g.strokeStyle = C.warn; g.globalAlpha = 0.6; g.strokeRect(plot.x + 5, by, plot.s - 10, bh); g.globalAlpha = 1;
            g.fillStyle = C.warn;
            lines.forEach((ln, i) => g.fillText(ln, plot.x + 12, by + 18 + i * 15));
          }
        }
        function drawLoss() {
          const LW = 720, LH = 130, px = 44, py = 12, pw = 620, ph = 96;
          lg.clearRect(0, 0, LW, LH);
          lg.fillStyle = '#0f1520'; lg.fillRect(px, py, pw, ph); lg.strokeStyle = C.line; lg.strokeRect(px, py, pw, ph);
          lg.font = MONO; lg.fillStyle = C.muted; lg.textAlign = 'right';
          lg.fillText('loss', px - 6, py + 12); lg.fillText('0', px - 6, py + ph);
          lg.textAlign = 'left'; lg.fillStyle = C.warn; lg.fillText('acc 100%', px + pw + 4, py + 12); lg.fillText('0%', px + pw + 4, py + ph);
          if (S.hist.length < 2) { lg.fillStyle = C.muted; lg.textAlign = 'center'; lg.font = FONT; lg.fillText('training loss (green) and accuracy (yellow) per epoch will appear here', LW / 2, py + ph / 2 + 4); return; }
          const maxL = Math.max(0.05, ...S.hist.map(q => q.loss));
          const n = S.hist.length, e0 = S.hist[0].e, e1 = S.hist[n - 1].e;
          const xAt = (e) => px + (e - e0) / Math.max(1, e1 - e0) * pw;
          lg.strokeStyle = C.warn; lg.lineWidth = 1.5; lg.setLineDash([4, 3]); lg.beginPath();
          S.hist.forEach((q, i) => { const x = xAt(q.e), y = py + ph - q.acc * ph; if (i === 0) lg.moveTo(x, y); else lg.lineTo(x, y); }); lg.stroke(); lg.setLineDash([]);
          lg.strokeStyle = C.green; lg.lineWidth = 2; lg.beginPath();
          S.hist.forEach((q, i) => { const x = xAt(q.e), y = py + ph - Math.min(1, q.loss / maxL) * ph; if (i === 0) lg.moveTo(x, y); else lg.lineTo(x, y); }); lg.stroke();
          lg.fillStyle = C.muted; lg.font = MONO; lg.textAlign = 'left'; lg.fillText('epoch ' + e1, px + 6, py + ph - 6);
          lg.textAlign = 'right'; lg.fillText('max ' + f3(maxL), px - 6, py + 26);
        }
        const ro = ctx.readout();
        function updateRO() { ro.set({ epoch: S.epoch, loss: isFinite(S.loss) ? f4(S.loss) : '–', accuracy: S.hist.length ? Math.round(S.acc * 100) + '%' : '–', parameters: 4 * S.N + 1, 'learning rate': S.lr.toPrecision(2) }); }
        ctx.loop((dt) => {
          S.heatAge += dt;
          if (S.playing) trainSamples(S.speed);
          if (S.playing || S.dirty) { draw(); drawLoss(); updateRO(); S.dirty = false; }
        });
        /* pointer probe — works with mouse and touch; never calls preventDefault, so a touch drag
           still scrolls the page instead of being swallowed by the canvas. */
        function movedProbe(e) {
          const q = cv.pos(e);
          const inside = q.x >= plot.x && q.x <= plot.x + plot.s && q.y >= plot.y && q.y <= plot.y + plot.s;
          if (!inside) { if (probe.on) { probe.on = false; S.dirty = true; } return; }
          probe.on = true;
          probe.x = (q.x - plot.x) / plot.s * 2 - 1;
          probe.y = 1 - (q.y - plot.y) / plot.s * 2;
          S.dirty = true;
        }
        const dropProbe = () => { if (probe.on) { probe.on = false; S.dirty = true; } };
        cv.addEventListener('pointerdown', movedProbe);
        cv.addEventListener('pointermove', movedProbe);
        cv.addEventListener('pointerup', movedProbe);
        cv.addEventListener('pointerleave', dropProbe);
        cv.addEventListener('pointercancel', dropProbe);
        const dsSel = ctx.select({ label: 'dataset', options: [{ value: 'xor', label: 'XOR' }, { value: 'circles', label: 'Two circles' }, { value: 'spiral', label: 'Spiral' }, { value: 'gauss', label: 'Two gaussians' }], value: 'xor', onChange: (v) => { S.ds = v; makeData(); reinit(); } });
        const nSl = ctx.slider({ label: 'hidden units', min: 2, max: 16, step: 1, value: 4, onChange: (v) => { S.N = v; reinit(); } });
        const lrSl = ctx.slider({ label: 'learning rate (log scale)', min: -3, max: 0, step: 0.1, value: -1.3, fmt: (v) => Math.pow(10, v).toPrecision(2), onChange: (v) => { S.lr = Math.pow(10, v); S.dirty = true; } });
        const actSel = ctx.select({ label: 'activation', options: ['tanh', 'relu', 'sigmoid'], value: 'tanh', onChange: (v) => { S.act = v; reinit(); } });
        const spSl = ctx.slider({ label: 'SGD steps per frame', min: 10, max: 400, step: 10, value: 100, onChange: (v) => { S.speed = v; } });
        const playBtn = ctx.button('▶ Play', () => { S.playing = !S.playing; playBtn.textContent = S.playing ? '⏸ Pause' : '▶ Play'; }, 'primary');
        const stepBtn = ctx.button('Step 1 epoch', () => { S.playing = false; playBtn.textContent = '▶ Play'; trainSamples(S.data.length); });
        const resetBtn = ctx.button('Reset weights', () => { reinit(); });
        const linesBtn = ctx.button('Hidden-unit lines: on', () => { S.showLines = !S.showLines; linesBtn.textContent = 'Hidden-unit lines: ' + (S.showLines ? 'on' : 'off'); S.dirty = true; });
        makeData(); reinit();
        const body = h('div', {}, cv, lc);
        return ctx.figure(body, 'A 2 → N → 1 network trained by plain stochastic gradient descent, implemented by hand in this page. The heatmap is the network\'s current prediction at every point (red = class 1, blue = class 0, dark = unsure) — point or tap anywhere on it to read the exact prediction there. Faint white lines are where each hidden unit\'s weighted sum is zero: the straight cuts the curved boundary is folded from.', [dsSel, nSl, actSel, lrSl, spSl, playBtn, stepBtn, resetBtn, linesBtn], ro);
      }

      /* ------------------------------------------------------------------ */
      /* Interactive B: gradient descent on a bumpy 1-D loss                  */
      /* ------------------------------------------------------------------ */
      function gradientDescent1D() {
        const W = 720, H = 312;
        const [cv, g] = ctx.canvas(W, H);
        const X0 = -6, X1 = 6;
        const f = (x) => 0.05 * x * x + 0.8 * Math.sin(1.3 * x) + 0.3 * Math.cos(2.7 * x);
        const df = (x) => 0.1 * x + 1.04 * Math.cos(1.3 * x) - 0.81 * Math.sin(2.7 * x);
        let ymin = Infinity, ymax = -Infinity, gmin = 0;
        for (let x = X0; x <= X1; x += 0.005) { const v = f(x); if (v < ymin) { ymin = v; gmin = x; } if (v > ymax) ymax = v; }
        const plot = { x: 40, y: 16, w: 660, h: 236 };
        const toPx = (x, y) => ({ x: plot.x + (x - X0) / (X1 - X0) * plot.w, y: plot.y + (ymax - y) / (ymax - ymin) * plot.h * 0.92 + plot.h * 0.04 });
        const S = { x: 4.5, start: 4.5, lr: 0.3, trail: [], steps: 0, running: false, acc: 0, lastJump: 0, msg: 'Press Step or ▶ Run.' };
        let runBtn = null;                                  // assigned below; reset() may run before you look
        const ro = ctx.readout();
        function reset() { S.x = S.start; S.trail = []; S.steps = 0; S.running = false; S.acc = 0; S.lastJump = 0; S.msg = 'Ball placed at x = ' + f2(S.start) + '.'; if (runBtn) runBtn.textContent = '▶ Run'; }
        function step() {
          const gx = df(S.x);
          let nx = S.x - S.lr * gx;
          nx = ctx.clamp(nx, X0, X1);
          S.trail.push({ from: S.x, to: nx }); if (S.trail.length > 60) S.trail.shift();
          S.lastJump = Math.abs(nx - S.x); S.x = nx; S.steps++;
          const slope = df(S.x);
          if (S.lastJump > 1.2) S.msg = 'Big leap (' + f2(S.lastJump) + '): the learning rate is so large the ball jumps across valleys instead of settling.';
          else if (Math.abs(slope) < 0.03) S.msg = Math.abs(S.x - gmin) < 0.3 ? 'Settled at the global minimum, the lowest point on the whole curve.' : 'Settled in a local minimum: the slope here is ≈ 0, so gradient descent stops, even though a lower valley exists at x ≈ ' + f2(gmin) + '.';
          else S.msg = 'Slope ' + f2(slope) + ' → next step of ' + (slope > 0 ? '−' : '+') + f2(Math.abs(S.lr * slope)) + ' (learning rate × slope).';
          updateRO();
        }
        function draw(t) {
          g.clearRect(0, 0, W, H);
          g.fillStyle = '#0f1520'; g.fillRect(plot.x, plot.y, plot.w, plot.h); g.strokeStyle = C.line; g.strokeRect(plot.x, plot.y, plot.w, plot.h);
          // the curve
          g.strokeStyle = C.accent; g.lineWidth = 2.5; g.beginPath();
          for (let i = 0; i <= 300; i++) { const x = X0 + (X1 - X0) * i / 300, s = toPx(x, f(x)); if (i === 0) g.moveTo(s.x, s.y); else g.lineTo(s.x, s.y); }
          g.stroke();
          // global minimum marker
          const gm = toPx(gmin, f(gmin));
          g.setLineDash([3, 4]); g.strokeStyle = C.green; g.lineWidth = 1; g.beginPath(); g.moveTo(gm.x, plot.y); g.lineTo(gm.x, plot.y + plot.h); g.stroke(); g.setLineDash([]);
          g.fillStyle = C.green; g.font = FONT; g.textAlign = 'center'; g.fillText('global minimum', gm.x, plot.y + 14);
          // trail
          S.trail.forEach((s, i) => {
            const al = (i + 1) / S.trail.length;
            const A = toPx(s.from, f(s.from)), B = toPx(s.to, f(s.to));
            g.strokeStyle = C.warn; g.globalAlpha = 0.25 + 0.6 * al; g.lineWidth = 1.5; g.beginPath(); g.moveTo(A.x, A.y); g.lineTo(B.x, B.y); g.stroke();
            g.beginPath(); g.arc(A.x, A.y, 3, 0, Math.PI * 2); g.fillStyle = C.warn; g.fill();
          });
          g.globalAlpha = 1;
          // tangent at the ball
          const slope = df(S.x), bx = S.x, by = f(bx);
          const dxT = 0.9, A = toPx(bx - dxT, by - slope * dxT), B = toPx(bx + dxT, by + slope * dxT);
          g.save(); g.beginPath(); g.rect(plot.x, plot.y, plot.w, plot.h); g.clip();
          g.strokeStyle = C.danger; g.lineWidth = 1.5; g.beginPath(); g.moveTo(A.x, A.y); g.lineTo(B.x, B.y); g.stroke();
          g.restore();
          // ball
          const P = toPx(bx, by);
          g.beginPath(); g.arc(P.x, P.y, 8, 0, Math.PI * 2); g.fillStyle = C.warn; g.fill(); g.strokeStyle = '#0a0e16'; g.lineWidth = 1.5; g.stroke();
          // proposed next step arrow
          const nx = ctx.clamp(bx - S.lr * slope, X0, X1), Q = toPx(nx, f(nx));
          g.strokeStyle = C.text; g.globalAlpha = 0.6; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(P.x, P.y); g.lineTo(Q.x, Q.y); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
          g.fillStyle = C.text; g.font = MONO; g.textAlign = 'left';
          g.fillText('x = ' + f2(bx) + '   loss = ' + f3(by) + '   slope = ' + f2(slope) + '   steps = ' + S.steps, plot.x + 8, plot.y + plot.h - 8);
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          wrapText(g, S.msg, plot.x, plot.y + plot.h + 20, plot.w, 15);   // wrapped: messages are long
          g.fillStyle = C.muted; g.font = MONO; g.textAlign = 'right'; g.fillText('red = slope under your feet · dashed = next step', plot.x + plot.w - 8, plot.y + plot.h - 8);
        }
        function updateRO() { ro.set({ x: f2(S.x), loss: f3(f(S.x)), slope: f2(df(S.x)), steps: S.steps, 'learning rate': f2(S.lr) }); }
        const lrSl = ctx.slider({ label: 'learning rate η', min: 0.01, max: 1.5, step: 0.01, value: 0.3, fmt: f2, onChange: (v) => { S.lr = v; updateRO(); } });
        const stSl = ctx.slider({ label: 'start position', min: -6, max: 6, step: 0.1, value: 4.5, fmt: (v) => (+v).toFixed(1), onChange: (v) => { S.start = v; reset(); updateRO(); } });
        const stepBtn = ctx.button('Step', () => { S.running = false; runBtn.textContent = '▶ Run'; step(); });
        runBtn = ctx.button('▶ Run', () => { S.running = !S.running; runBtn.textContent = S.running ? '⏸ Pause' : '▶ Run'; }, 'primary');
        const resetBtn = ctx.button('Reset', reset);
        updateRO();
        ctx.loop((dt, t) => { if (S.running) { S.acc += dt * 5; while (S.acc >= 1) { S.acc -= 1; step(); } } draw(t); });
        return ctx.figure(cv, 'Gradient descent in one dimension: feel the slope, step the other way, repeat. The curve is <code class="inline">0.05x² + 0.8·sin(1.3x) + 0.3·cos(2.7x)</code>, chosen for its bumps. Its four dips sit at x ≈ −5.74, −1.15, 0.96 and 3.44; only the second is the global minimum.', [lrSl, stSl, stepBtn, runBtn, resetBtn], ro);
      }

      /* ------------------------------------------------------------------ */
      /* Interactive C: backprop signal flow                                  */
      /* ------------------------------------------------------------------ */
      function backpropFlow() {
        const W = 720, H = 340;
        const [cv, g] = ctx.canvas(W, H);
        const x = [1.0, 0.5], W1 = [[0.6, -0.4], [0.3, 0.8]], b1 = [0.1, -0.5], W2 = [0.7, -0.5], b2 = 0.05, tgt = 1.0;
        const zz = [W1[0][0] * x[0] + W1[0][1] * x[1] + b1[0], W1[1][0] * x[0] + W1[1][1] * x[1] + b1[1]];
        const hh = zz.map(sig);
        const y = W2[0] * hh[0] + W2[1] * hh[1] + b2;
        const L = 0.5 * (y - tgt) * (y - tgt);
        const dy = y - tgt;
        const gW2 = [dy * hh[0], dy * hh[1]];
        const delta = [dy * W2[0] * hh[0] * (1 - hh[0]), dy * W2[1] * hh[1] * (1 - hh[1])];
        const gW1 = [[delta[0] * x[0], delta[0] * x[1]], [delta[1] * x[0], delta[1] * x[1]]];
        const P = { in: [[70, 100], [70, 250]], hid: [[345, 100], [345, 250]], out: [610, 175] };
        const T = { inputs: 0.6, f1: 2.3, hid: 2.9, f2: 4.3, out: 4.9, loss: 5.8, b2: 7.4, b1: 9.2, hold: 10.6 };
        const S = { t: 0, playing: true, speed: 1, dirty: true };
        const ease = (u) => Math.max(0, Math.min(1, u));
        function node(px, r, label, sub, subColor) {
          g.beginPath(); g.arc(px[0], px[1], r, 0, Math.PI * 2); g.fillStyle = '#111827'; g.fill(); g.strokeStyle = C.text; g.lineWidth = 1.5; g.stroke();
          g.fillStyle = C.text; g.font = MONO; g.textAlign = 'center'; g.fillText(label, px[0], px[1] + 4);
          if (sub) { g.fillStyle = subColor || C.muted; g.fillText(sub, px[0], px[1] + r + 16); }
        }
        /* One edge. Exactly one label is shown at a time — the forward product while the green wave
           is on or past it, the gradient once the red wave arrives — so labels can never overlap. */
        function edge(A, B, w, fwdU, bwdU, fwdLabel, bwdLabel, at) {
          g.strokeStyle = C.line; g.lineWidth = 2; g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke();
          const lx = A[0] + (B[0] - A[0]) * at, ly = A[1] + (B[1] - A[1]) * at;
          g.font = MONO; g.textAlign = 'center';
          if (fwdU > 0 && fwdU < 1) { const px = A[0] + (B[0] - A[0]) * fwdU, py = A[1] + (B[1] - A[1]) * fwdU; g.beginPath(); g.arc(px, py, 6, 0, Math.PI * 2); g.fillStyle = C.green; g.fill(); }
          if (bwdU > 0 && bwdU < 1) { const u = 1 - bwdU, px = A[0] + (B[0] - A[0]) * u, py = A[1] + (B[1] - A[1]) * u; g.beginPath(); g.arc(px, py, 6, 0, Math.PI * 2); g.fillStyle = C.danger; g.fill(); }
          if (bwdU >= 0.55) {
            g.fillStyle = C.muted; g.fillText('w = ' + f2(w), lx, ly - 7);
            g.fillStyle = C.danger; g.fillText(bwdLabel, lx, ly + 14);
          } else if (fwdU >= 0.55) {
            g.fillStyle = C.green; g.fillText(fwdLabel, lx, ly - 7);
          }
        }
        function draw() {
          const t = S.t;
          g.clearRect(0, 0, W, H);
          g.fillStyle = C.line; g.fillRect(0, 0, W, 3);
          g.fillStyle = C.accent; g.fillRect(0, 0, W * ctx.clamp(t / T.hold, 0, 1), 3);
          const phase = t < T.inputs ? 'Inputs arrive' : t < T.hid ? 'Forward pass: activations flow right' : t < T.out ? 'Forward pass: output' : t < T.loss ? 'Compare with the target → loss' : t < T.b1 ? 'Backward pass: gradients flow left' : 'Done: every weight now knows which way to move';
          g.fillStyle = t >= T.loss && t < T.b1 ? C.danger : C.green; g.font = 'bold 14px Inter, system-ui, sans-serif'; g.textAlign = 'left'; g.fillText(phase, 16, 24);
          g.fillStyle = C.muted; g.font = MONO; g.textAlign = 'left';
          g.fillText('fixed: hidden biases ' + f2(b1[0]) + ' / ' + f2(b1[1]) + ' · output bias ' + f2(b2), 16, 44);
          const f1u = ease((t - T.inputs) / (T.f1 - T.inputs)), f2u = ease((t - T.hid) / (T.f2 - T.hid));
          const b2u = ease((t - T.loss) / (T.b2 - T.loss)), b1u = ease((t - T.b2) / (T.b1 - T.b2));
          for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) {
            edge(P.in[i], P.hid[j], W1[j][i], f1u, b1u, f2(W1[j][i]) + '×' + f2(x[i]) + '=' + f2(W1[j][i] * x[i]), '∂L/∂w = ' + f3(gW1[j][i]), 0.35);
          }
          for (let j = 0; j < 2; j++) edge(P.hid[j], P.out, W2[j], f2u, b2u, f2(W2[j]) + '×' + f2(hh[j]) + '=' + f2(W2[j] * hh[j]), '∂L/∂w = ' + f3(gW2[j]), 0.5);
          // nodes
          const inAlpha = ease(t / T.inputs);
          g.globalAlpha = 0.25 + 0.75 * inAlpha;
          node(P.in[0], 20, 'x1', 'x1 = ' + f2(x[0])); node(P.in[1], 20, 'x2', 'x2 = ' + f2(x[1]));
          g.globalAlpha = 1;
          const hidShown = t >= T.f1, outShown = t >= T.f2, lossShown = t >= T.out, dyShown = t >= T.loss + 0.2, deltaShown = t >= T.b2;
          node(P.hid[0], 22, 'h1', hidShown ? 'σ(' + f2(zz[0]) + ') = ' + f3(hh[0]) : (t > T.inputs ? '…' : ''), C.green);
          node(P.hid[1], 22, 'h2', hidShown ? 'σ(' + f2(zz[1]) + ') = ' + f3(hh[1]) : (t > T.inputs ? '…' : ''), C.green);
          if (deltaShown) { g.fillStyle = C.danger; g.font = MONO; g.textAlign = 'center'; g.fillText('δ1 = ∂L/∂z1 = ' + f3(delta[0]), P.hid[0][0], P.hid[0][1] - 36); g.fillText('δ2 = ∂L/∂z2 = ' + f3(delta[1]), P.hid[1][0], P.hid[1][1] + 62); }
          node(P.out, 24, 'y', outShown ? 'y = ' + f3(y) : (t > T.hid ? '…' : ''), C.green);
          g.font = MONO; g.textAlign = 'center';
          g.fillStyle = C.muted; g.fillText('target t = ' + f2(tgt), P.out[0], 100);
          if (lossShown) {
            g.fillStyle = C.warn; g.font = 'bold 13px "JetBrains Mono", ui-monospace, monospace';
            g.fillText('L = ½(y − t)² = ' + f4(L), P.out[0], P.out[1] + 70);
          }
          if (dyShown) { g.fillStyle = C.danger; g.font = MONO; g.fillText('∂L/∂y = y − t = ' + f3(dy), P.out[0], P.out[1] + 92); }
          if (t >= T.b1) { g.fillStyle = C.text; g.font = FONT; g.textAlign = 'left'; g.fillText('Next: w ← w − η·∂L/∂w for every weight, then repeat with the next example.', 16, H - 8); }
        }
        const playBtn = ctx.button('⏸ Pause', () => { S.playing = !S.playing; S.dirty = true; playBtn.textContent = S.playing ? '⏸ Pause' : '▶ Play'; }, 'primary');
        const replayBtn = ctx.button('↺ Replay', () => { S.t = 0; S.playing = true; S.dirty = true; playBtn.textContent = '⏸ Pause'; });
        const spSl = ctx.slider({ label: 'speed', min: 0.25, max: 2, step: 0.25, value: 1, fmt: (v) => (+v).toFixed(2) + '×', onChange: (v) => { S.speed = v; } });
        /* drag anywhere across the picture to scrub the timeline (pointer events: mouse + touch).
           touchAction 'pan-y' keeps vertical page scrolling while claiming horizontal drags. */
        cv.style.touchAction = 'pan-y';
        let scrubbing = false;
        function scrub(e) {
          S.t = ctx.clamp(cv.pos(e).x / W, 0, 1) * T.hold;
          S.playing = false; S.dirty = true; playBtn.textContent = '▶ Play';
        }
        const endScrub = () => { scrubbing = false; };
        cv.addEventListener('pointerdown', (e) => { scrubbing = true; scrub(e); });
        cv.addEventListener('pointermove', (e) => { if (scrubbing) scrub(e); });
        cv.addEventListener('pointerup', endScrub);
        cv.addEventListener('pointercancel', endScrub);
        cv.addEventListener('pointerleave', endScrub);
        ctx.loop((dt) => {
          if (S.playing) { S.t += dt * S.speed; if (S.t > T.hold) S.t = 0; draw(); }
          else if (S.dirty) { draw(); S.dirty = false; }   // paused: don't redraw an unchanging frame
        });
        return ctx.figure(cv, 'One training step on a 2 → 2 → 1 network with sigmoid hidden units and a linear output. Green: the forward pass carries activations right. Red: the backward pass carries gradients left, reusing ∂L/∂y at every edge. Every number is computed live from the weights shown — drag left and right across the picture to scrub through the step.', [playBtn, replayBtn, spSl]);
      }

      /* ================================================================== */
      /* Prose                                                              */
      /* ================================================================== */
      root.append(
        p(`Your phone unlocks when it sees your face and stays locked for your sibling, who shares half your genes and most of your features. No straight line through pixel-space separates the two of you. Chapter 1 ended at a wall: a single neuron can only draw a straight boundary, and even the toy problem XOR is on the wrong side of it. Yet by 1989 a neural network was reading handwritten postcodes for the US Postal Service, and today the same family of machines writes code and reads X-rays.`),
        p(`Two ideas broke through the wall. The first is obvious in hindsight: use more than one neuron, and feed the outputs of some neurons into others. The second is the one that took until 1986 to become widely known and is still the engine of every modern model: a way to work out, for each of a million weights at once, which direction to nudge it. That method is called <em>backpropagation</em>.`),
        p(`This chapter builds both ideas from scratch, with numbers you can check by hand, and then lets you train a real network in your browser and watch its decision boundary bend.`),

        section('Stacking neurons: layers',
          p(`Take two perceptrons and point them at the same two inputs. Each draws its own straight line. Now feed their two outputs into a third perceptron. The third one never sees the raw inputs; it sees "which side of line A" and "which side of line B", and it can be told to say yes only for one particular combination. Its region is a strip between two lines, and a strip is not something one line can draw.`),
          p(`That is XOR solved. Hidden neuron A fires when x<sub>1</sub> + x<sub>2</sub> > 0.5 (at least one input on). Hidden neuron B fires when x<sub>1</sub> + x<sub>2</sub> > 1.5 (both on). The output neuron fires when A is on and B is off: exactly one input on. Three neurons in two layers, and the wall from 1969 is gone.`),
          p(`The middle neurons are called a <em>hidden layer</em>, because you never look at their outputs directly; they are the network's private vocabulary. A network with an input layer, one or more hidden layers and an output layer, where every neuron connects to every neuron in the next layer, is a <em>multi-layer perceptron</em> (MLP), also called a feed-forward network. With enough hidden neurons, one hidden layer can approximate any reasonable function to any accuracy you like; that is a theorem (Cybenko 1989, Hornik 1991). The theorem says nothing about how to <i>find</i> the weights. That is the hard part, and the rest of this chapter.`),
        ),

        section('Activation functions: why the squash matters',
          p(`After each neuron's weighted sum, a small non-linear function is applied before the value is passed on. This is the <em>activation function</em>, and the choice matters more than it looks.`),
          ul([
            `<b>Step</b> (the perceptron's threshold): output 0 or 1. Honest, but its slope is zero everywhere except at a single point, so it gives no hint about which way to move a weight. Useless for what follows.`,
            `<b>Sigmoid</b>, σ(z) = 1 / (1 + e<sup>−z</sup>): a smooth S-curve from 0 to 1, a "soft step". Its slope is defined everywhere, which is why it powered the 1986 revival. Downside: for large |z| the slope is nearly zero, so the learning signal fades when many sigmoids are stacked.`,
            `<b>tanh</b>: the same S-curve rescaled to run from −1 to 1. Centred on zero, which helps learning. The default in this chapter's playground.`,
            `<b>ReLU</b>, max(0, z): pass positive values through, zero out negatives. Absurdly simple, cheap, and its slope is exactly 1 for positive inputs, so signals do not fade through deep stacks. Since about 2011 it and its cousins are the default in deep networks.`,
          ]),
          p(`Why is a non-linearity needed at all? Because a stack of linear layers is still linear. If layer one computes h = W<sub>1</sub>·x and layer two computes y = W<sub>2</sub>·h, then y = (W<sub>2</sub>W<sub>1</sub>)·x: a single matrix, a single flat boundary, the 1969 wall again. The squash between layers is what gives stacking its power. Every curved boundary you will see in the playground is built from straight lines folded by activation functions.`),
        ),

        section('Loss: a number for how wrong you are',
          p(`The perceptron rule only knew right from wrong. To train many weights at once we need something finer: a single number that says <i>how</i> wrong the network is, that gets smaller as predictions improve, and that changes smoothly as the weights change. This is the <em>loss function</em>.`),
          ul([
            `<b>Mean squared error</b> (MSE): for each example, take (prediction − truth)², and average over the examples. Being off by 2 costs four times as much as being off by 1. Natural for predicting numbers: a house price, tomorrow's temperature.`,
            `<b>Cross-entropy</b>: when the network outputs a probability, the loss is −log(probability it gave to the correct answer). If it said 90% for the right class, the loss is −log(0.9) = 0.105. If it said 1%, the loss is 4.6. Confident mistakes are punished brutally, which is exactly what you want from a classifier. This is the loss that trains language models: −log(probability of the actual next word).`,
          ]),
          p(`Training now has a crisp definition: <b>find the weights that make the loss, averaged over the training data, as small as possible.</b> Picture the loss as a landscape over the space of all possible weight settings. Training is a search for the lowest valley.`),
          callout('key', 'The whole of training in one line', `Every model in this course — the four-neuron toy below, ResNet, GPT — is trained by the same loop: <b>predict, measure the error with a loss function, ask the chain rule which way each weight should move, move it a little, repeat.</b> Everything else (architecture, optimizers, data curation, RLHF) is a refinement of one of those five verbs.`),
        ),

        section('Gradient descent: downhill in the fog',
          p(`Imagine standing on a hillside in thick fog. You cannot see the valley, but you can feel the slope under your feet. Take a step in the steepest downhill direction, feel the slope again, step again. That is <em>gradient descent</em>. The <em>gradient</em> is the slope: for each weight, how much would the loss rise if I increased this weight a little? Move every weight a small amount the other way:`),
          ctx.code('w ← w − η · ∂L/∂w        # for every weight w, at every step'),
          p(`The learning rate η is your stride. Too small and you take a million steps to reach the valley. Too large and you leap over it, land higher up on the opposite slope, leap back, and the loss bounces or explodes. And on a bumpy landscape you can settle into the nearest dip, a <em>local minimum</em>, rather than the deepest one. Try all three below.`),
          callout('tryit', 'Try it: three ways to fail at walking downhill', `<b>1.</b> Start at 4.5 with η = 0.3 and press Step a few times. The ball settles in the dip near x ≈ 3.4 and stops: a <em>local minimum</em>. The slope is zero, so the rule has nothing to say, even though a deeper valley exists on the left. <b>2.</b> Set η to 1.2 and press ▶ Run: the ball ricochets between hillsides, sometimes landing higher than it started. <b>3.</b> Set η to 0.02: it crawls. <b>4.</b> Find a start position from which η = 0.3 reaches the global minimum. <b>Notice:</b> the red tangent is the only information the algorithm ever has.`),
          gradientDescent1D(),
          callout('warning', 'Where this picture lies to you', `A one-dimensional valley makes local minima look like the central danger of training, and for eighty years people assumed they were. In a network with a billion weights the landscape has a billion dimensions, and a point is only a local minimum if the curve bends <i>upward in every single one of them</i> — which is about as likely as a billion coin flips all coming up heads. Almost every flat spot is a <em>saddle point</em>: downhill in some directions, uphill in others, and gradient descent eventually slides off it. That is one reason enormous networks train far more reliably than the 1-D intuition predicts. Keep the ball-in-a-valley picture for the mechanics of a step, and distrust it about what the terrain is like.`),
        ),

        section('Backpropagation: the chain rule, layer by layer',
          p(`Gradient descent needs the slope of the loss with respect to every weight. For a weight deep inside a network, changing it changes a hidden activation, which changes the next layer, which changes the output, which changes the loss. Working out that slope separately for each of a million weights would be hopeless. <em>Backpropagation</em> is the observation that you can compute all of them in one sweep backwards through the network, reusing intermediate results, because of the chain rule from calculus: <b>the slope of a chain of functions is the product of the slopes of its links.</b>`),
          p(`Here is the whole thing on the smallest network that shows it. One input x, one hidden neuron h = σ(w<sub>1</sub>·x), one linear output y = w<sub>2</sub>·h, and a squared-error loss L = ½(y − t)². Take x = 1, target t = 1, and starting weights w<sub>1</sub> = 0.5, w<sub>2</sub> = −1.`),
          ol([
            `<b>Forward pass.</b> z = w<sub>1</sub>·x = 0.5. h = σ(0.5) = 0.6225. y = w<sub>2</sub>·h = −0.6225. Loss L = ½(−0.6225 − 1)² = ½ × 2.632 = <b>1.316</b>.`,
            `<b>How does L change with y?</b> ∂L/∂y = y − t = −1.6225. (Negative: making y bigger would reduce the loss.)`,
            `<b>Back one link, to w<sub>2</sub>.</b> Since y = w<sub>2</sub>·h, ∂y/∂w<sub>2</sub> = h. Chain rule: ∂L/∂w<sub>2</sub> = ∂L/∂y × ∂y/∂w<sub>2</sub> = −1.6225 × 0.6225 = <b>−1.010</b>.`,
            `<b>Back through the sigmoid, to w<sub>1</sub>.</b> ∂y/∂h = w<sub>2</sub> = −1. The sigmoid's slope is σ′(z) = h(1 − h) = 0.6225 × 0.3775 = 0.2350. And ∂z/∂w<sub>1</sub> = x = 1. Chain: ∂L/∂w<sub>1</sub> = (−1.6225) × (−1) × 0.2350 × 1 = <b>0.3813</b>.`,
            `<b>Step</b> with η = 0.1: w<sub>2</sub> ← −1 − 0.1 × (−1.010) = −0.899. w<sub>1</sub> ← 0.5 − 0.1 × 0.3813 = 0.4619.`,
            `<b>Check.</b> Forward again: h = σ(0.4619) = 0.6135, y = −0.899 × 0.6135 = −0.5515, L = ½(1.5515)² = <b>1.204</b>. The loss fell from 1.316 to 1.204. Repeat a few thousand times.`,
          ]),
          p(`Notice what happened in step 4: the quantity ∂L/∂y that we computed for the output was <i>reused</i> for the hidden weight, multiplied by the slopes of the links in between. In a network with a million weights, each layer's gradients are computed from the layer after it, in one backward sweep that costs about as much as the forward pass. That reuse is the entire trick, and the animation below shows it flowing.`),
          callout('tryit', 'Try it: watch the two passes', `Green dots are the forward pass: each edge multiplies its input by its weight and the products are summed at the next node. Then the loss is computed at the far right. Red dots are the backward pass: ∂L/∂y is computed once and pushed left, picking up a factor at every edge and node. <b>Notice:</b> every red number on the layer-1 edges contains the same ∂L/∂y = −0.789 as a factor; nothing is recomputed from scratch. Drag left and right across the picture to freeze the step mid-flight, and check one number against the recipe above with a calculator.`),
          backpropFlow(),
        ),

        section('The playground: train a network in your browser',
          p(`Everything above is implemented, by hand, in a few dozen lines of JavaScript underneath this figure: a forward pass, a backward pass, and a gradient step, repeated a few hundred times per animation frame. Nothing is pre-computed and nothing is faked; when the boundary wobbles, that is the gradient wobbling. Here is the whole of it, with the loops written out:`),
          ctx.code(`# one training step on one example (x, y) with label l ∈ {0, 1}
for j in hidden:
    z[j] = w1[j][0]*x + w1[j][1]*y + b1[j]      # hidden pre-activation
    a[j] = tanh(z[j])                           # hidden activation
o = sum(w2[j] * a[j] for j in hidden) + b2      # output pre-activation
p = 1 / (1 + exp(-o))                           # predicted probability of class 1
L = -log(p if l == 1 else 1 - p)                # cross-entropy loss

dL_do = p - l                # sigmoid + cross-entropy collapse to exactly this
dL_db2 = dL_do
for j in hidden:
    dL_dw2[j] = dL_do * a[j]
    dL_dz[j]  = dL_do * w2[j] * (1 - a[j]**2)   # tanh'(z) = 1 - tanh(z)^2
    dL_dw1[j] = (dL_dz[j] * x, dL_dz[j] * y)
    dL_db1[j] = dL_dz[j]

w -= lr * dL_dw              # same rule for every weight and every bias`),
          p(`Two details worth pausing on. First, <code class="inline">p − l</code>: when a sigmoid output meets a cross-entropy loss, all the messy derivatives cancel and the error signal is simply <i>predicted minus actual</i>. That is not a coincidence, and the same cancellation happens with softmax and cross-entropy in every language model. Second, notice that the second loop is shorter than the first: computing every gradient really does cost about the same as one forward pass.`),
          callout('tryit', 'Try it: the playground', `<b>1.</b> Start with <b>XOR</b>, 4 hidden units, tanh. Press ▶ Play. Watch the heatmap fold itself into four quadrants and the loss curve fall. <b>2.</b> Switch to <b>Two circles</b>. A round boundary from straight cuts: turn "Hidden-unit lines" on and count how many lines it takes. <b>3.</b> <b>Spiral</b> with 4 units: it cannot, and the loss plateaus. Push hidden units to 12–16 and let it run for a minute. <b>4.</b> Set the learning rate to 1.0: the loss thrashes or explodes (the page resets the weights if numbers blow up). Set it to 0.001: nothing seems to happen. <b>5.</b> Switch to <b>relu</b>: the boundary is made of straight segments with sharp corners. Sigmoid: slower, smoother. <b>6.</b> Press Reset a few times on the spiral with 6 units: different random starts give different solutions, some good, some stuck. That is the fog. <b>7.</b> Point at (or tap) any spot on the heatmap to read the network's exact prediction there — try a point right on the boundary and watch it sit near 0.50, the network's way of saying "no idea".`),
          playground(),
        ),

        section('Vocabulary you will hear constantly',
          ul([
            `<b>Epoch</b>: one full pass through the training data. The playground's counter is in epochs.`,
            `<b>Batch</b> (or mini-batch): the handful of examples whose gradients are averaged before each step. The playground uses a batch of one, pure <em>stochastic gradient descent</em> (SGD). Real training uses batches of hundreds to millions of examples, for reasons chapter 3 explains.`,
            `<b>Step</b> (or iteration): one weight update. Epochs × (examples ÷ batch size) = steps.`,
            `<b>Parameter count</b>: the number of learnable weights and biases. For a 2 → 16 → 1 network: 2×16 weights + 16 biases into the hidden layer, 16 weights + 1 bias into the output, total 65.`,
          ]),
          p(`Hold that number next to a modern language model, which is the same kind of object with around 10<sup>11</sup> to 10<sup>12</sup> weights, the same forward pass, and the same backward pass. The chain rule does not care how long the chain is.`),
        ),

        callout('history', '1986: rediscovered, then waiting 26 years for hardware', `The chain-rule method had been derived before (Seppo Linnainmaa in 1970 as a general technique, Paul Werbos in 1974 for neural networks) and sank without trace. In 1986 David Rumelhart, Geoffrey Hinton and Ronald Williams published "Learning representations by back-propagating errors" in <i>Nature</i>, showed it learning useful hidden representations, and this time the field noticed. It answered Minsky and Papert directly: multi-layer networks could be trained. What it could not overcome was 1986 hardware and 1986 data. Networks with a few thousand weights, trained on a few thousand examples, were matched or beaten by simpler methods through the 1990s and 2000s, and neural networks went out of fashion a second time. The same algorithm on a pair of GPUs (tens of times faster than the CPUs of the day on exactly the matrix multiplies training is made of) and a million labelled images produced the 2012 breakthrough of chapter 4. The idea was right for 26 years before the world caught up with it.`),

        callout('example', 'Where plain MLPs live today', `Every recommendation feed you scroll ends in an MLP that scores candidates. The feed-forward blocks inside a transformer, which hold roughly two-thirds of a large language model's weights, are exactly the 2-layer MLP from the playground, thousands of units wide. AlphaGo's value network head, credit-fraud detectors, and most predictions on spreadsheet-shaped data are MLPs. The architectures in Part II are ways of arranging MLPs, not replacements for them.`),

        section('Why this matters for modern AI',
          p(`When you read that a model was "trained on 15 trillion tokens", here is what physically happened: a forward pass predicted the next token, cross-entropy measured how wrong it was, backpropagation computed the gradient for every one of the 10<sup>11</sup>–10<sup>12</sup> weights, and an optimizer (chapter 3) nudged each one a tiny amount. Then again, a few million times, across thousands of GPUs, for months. There is no other mechanism. The network in the playground and Claude differ in size, in architecture (chapter 7), and in data. They do not differ in what "learning" means.`),
          p(`One picture to keep: <b>the loss is a landscape, the gradient is the slope under your feet, and backpropagation is how you feel the slope in a million directions at once.</b>`),
        ),

        ctx.quiz([
          { q: 'A network has three layers but no activation function between them. Why can it still not learn XOR?', options: ['Three layers is too few', 'Without a non-linearity the layers collapse into one linear map, which still draws a single flat boundary', 'The learning rate must be zero', 'XOR needs at least four inputs'], answer: 1, explain: 'W3·(W2·(W1·x)) = (W3W2W1)·x, a single matrix. Depth without non-linearity adds nothing; the squash between layers is what lets straight cuts be folded into curves.' },
          { q: 'You set the learning rate very high and the loss starts going <i>up</i>. What is happening?', options: ['The network has found the global minimum', 'Each step overshoots the valley and lands higher on the far slope', 'Backpropagation has stopped working', 'The data has changed'], answer: 1, explain: 'Gradient descent only knows the local slope. A stride longer than the valley is wide jumps across it; with each jump the loss can grow, until the numbers explode.' },
          { q: 'What is backpropagation, in one sentence?', options: ['A way to choose the learning rate', 'A special activation function', 'The chain rule applied backwards through the layers, so every weight\'s gradient is computed in one sweep', 'A method for collecting training data'], answer: 2, explain: 'Backprop reuses ∂L/∂(output) at every edge, multiplying by the slopes of the links in between. That makes the gradient for a million weights cost about as much as one forward pass.' },
          { q: 'A classifier gives the correct class a probability of 1%. Roughly what is its cross-entropy loss on that example, compared with giving it 90%?', options: ['About the same', 'About 0.1 versus 4.6: confident mistakes cost far more', 'Exactly ten times more', 'Zero, because it still ranked the class'], answer: 1, explain: '−log(0.9) ≈ 0.105 and −log(0.01) ≈ 4.6. Cross-entropy punishes confidently wrong predictions hardest, which drives the network to become well-calibrated.' },
          { q: 'How many parameters does a 2 → 8 → 1 network have?', options: ['11', '16', '33', '64'], answer: 2, explain: '2×8 weights + 8 biases = 24 into the hidden layer; 8 weights + 1 bias = 9 into the output. Total 33. The formula for 2 → N → 1 is 4N + 1.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://www.youtube.com/watch?v=Ilg3gGewQ5U" target="_blank" rel="noopener">3Blue1Brown, "What is backpropagation really doing?"</a> and the follow-up on the calculus: the visual version of the worked example above.`,
            `<a href="https://www.youtube.com/watch?v=VMj-3S1tku0" target="_blank" rel="noopener">Karpathy, "The spelled-out intro to neural networks and backpropagation: building micrograd"</a>: two and a half hours that build backprop from scratch in Python. Lab 02 of this course follows it.`,
            `<a href="https://playground.tensorflow.org" target="_blank" rel="noopener">TensorFlow Playground</a>: the playground above with more layers, more features, and regularisation.`,
            `<a href="https://www.nature.com/articles/323533a0" target="_blank" rel="noopener">Rumelhart, Hinton &amp; Williams (1986), "Learning representations by back-propagating errors"</a>: the four-page <i>Nature</i> paper.`,
            `<a href="http://neuralnetworksanddeeplearning.com/" target="_blank" rel="noopener">Michael Nielsen, <i>Neural Networks and Deep Learning</i></a>: a free online book whose chapter 2 is the clearest written derivation of backprop.`,
          ]),
        ),
      );
    },
  });
})();
