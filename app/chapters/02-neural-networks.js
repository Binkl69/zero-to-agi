/* Zero → AGI · Chapter 02 · Neural networks and backpropagation
   DESIGN RULE: the reader solves XOR with their own hands in the first ten seconds — the exact
   puzzle chapter 1 left them stuck on. Every paragraph explains something they just did.
   Interactives, in order: solve XOR by hand with two draggable lines + a combiner rule;
   stacked-layers-stay-linear lab (bendiness pinned at 0% without an activation); loss explorer
   (cross-entropy vs squared error, and the size of the shove each gives); 1-D gradient descent on
   a bumpy loss; backprop signal-flow animation; live MLP training playground; parameter-count
   scale chart from your toy to a frontier model. */
(function () {
  ZTA.registerChapter({
    id: '02-neural-networks',
    num: 2,
    part: 'I',
    title: 'Neural networks and backpropagation',
    tagline: 'The puzzle that beat you in chapter 1, solved with your own hands in ten seconds — then the one idea that lets a billion weights learn at once.',
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
          const hidY = (j) => 28 + (j + 0.5) * (300 / S.N);   // stays clear of the key on the last row
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
          g.fillText('red edge = positive weight, blue = negative', 560, H - 24);
          g.fillText('thickness = size of the weight', 560, H - 8);
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
          const LW = 720, LH = 130, px = 72, py = 12, pw = 580, ph = 96;   // margins fit "max 0.000" and "acc 100%"
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
      /* Measure a slope with subtraction and division. No calculus, and no  */
      /* lie either: this is the numerical derivative, and it is exactly what */
      /* gradient checking does in a real codebase (chapter 13's lab 02).     */
      /* ------------------------------------------------------------------ */
      function nudgeLab() {
        const [cv, g] = ctx.canvas(720, 400);
        let w = 2.6, eps = 0.8;
        /* a plain bowl, so the reader can check every number by eye */
        const L = (x) => 0.5 * (x - 1) * (x - 1) + 0.5;
        const trueSlope = (x) => (x - 1);          // for the tangent only; never shown as algebra

        /* w and the nudge are capped so that BOTH measured points always land inside the
           plot window below: a reader must never lose sight of the point being measured. */
        const wSl = ctx.slider({ label: 'the weight, w', min: -1.5, max: 3.5, step: 0.05, value: 2.6, digits: 2, onChange: (v) => { w = v; } });
        const eSl = ctx.slider({ label: 'how big a nudge', min: 0.005, max: 0.9, step: 0.005, value: 0.8, digits: 3, onChange: (v) => { eps = v; } });
        const tinyBtn = ctx.button('make the nudge tiny', () => { eps = 0.01; eSl.value = 0.01; }, 'primary');
        const bigBtn = ctx.button('make it big again', () => { eps = 0.8; eSl.value = 0.8; });
        const botBtn = ctx.button('go to the bottom', () => { w = 1; wSl.value = 1; });
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          const L0 = L(w), L1 = L(w + eps);
          const dL = L1 - L0;
          const measured = dL / eps;

          const P = { x: 60, y: 40, w: 380, h: 250 };
          const X0 = -1.8, X1 = 4.5, Y0 = 0, Y1 = 6.8;   // covers every reachable (w, w + nudge)
          const px = (x) => P.x + (x - X0) / (X1 - X0) * P.w;
          const py = (y) => P.y + P.h - (y - Y0) / (Y1 - Y0) * P.h;

          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('the weight, w  →', P.x + 120, P.y + P.h + 22);
          g.save(); g.translate(P.x - 40, P.y + P.h - 60); g.rotate(-Math.PI / 2);
          g.fillText('the loss, L  →', 0, 0); g.restore();

          /* everything that depends on w and the nudge stays inside the plot box. The clip is
             let out by one dot radius so a point sitting on an axis limit is not sliced in half. */
          g.save();
          g.beginPath(); g.rect(P.x - 6, P.y - 6, P.w + 12, P.h + 12); g.clip();

          /* the loss curve */
          g.strokeStyle = C.accent; g.lineWidth = 2.5; g.beginPath();
          for (let i = 0; i <= 140; i++) {
            const x = X0 + i / 140 * (X1 - X0);
            i ? g.lineTo(px(x), py(L(x))) : g.moveTo(px(x), py(L(x)));
          }
          g.stroke();

          /* the true tangent, faint, for comparison only */
          const m = trueSlope(w);
          g.strokeStyle = 'rgba(148,163,184,0.45)'; g.lineWidth = 1.5; g.setLineDash([4, 4]);
          g.beginPath();
          g.moveTo(px(w - 2.2), py(L0 + m * -2.2));
          g.lineTo(px(w + 2.2), py(L0 + m * 2.2));
          g.stroke(); g.setLineDash([]);

          /* the line through the two points you actually measured */
          g.strokeStyle = C.warn; g.lineWidth = 2.5;
          g.beginPath();
          g.moveTo(px(w - 1.6), py(L0 + measured * -1.6));
          g.lineTo(px(w + 2.4), py(L0 + measured * 2.4));
          g.stroke();

          /* the two points, and the two measurements between them */
          const ax = px(w), ay = py(L0), bx = px(w + eps), by = py(L1);
          g.strokeStyle = 'rgba(251,191,36,0.55)'; g.lineWidth = 1; g.setLineDash([2, 3]);
          g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, ay); g.lineTo(bx, by); g.stroke();
          g.setLineDash([]);

          [[ax, ay, C.text], [bx, by, C.warn]].forEach(([x, y, col]) => {
            g.beginPath(); g.arc(x, y, 6, 0, 7); g.fillStyle = col; g.fill();
            g.strokeStyle = '#0a0e16'; g.lineWidth = 2; g.stroke();
          });
          g.restore();

          /* the two measurements, kept inside the plot whatever the sliders say, and each on
             its own opaque chip so the yellow measurement line never runs through the number */
          g.font = MONO;
          const chip = (txt, x, y) => {
            const cw = g.measureText(txt).width;
            g.fillStyle = 'rgba(10,14,22,0.92)';
            g.fillRect(x - 4, y - 12, cw + 8, 17);
            g.fillStyle = C.warn; g.fillText(txt, x, y);
          };
          const tw = 'you moved w by ' + eps.toFixed(3), tl = 'L moved by ' + dL.toFixed(3);
          const wW = g.measureText(tw).width, wL = g.measureText(tl).width;
          chip(tw,
            ctx.clamp(Math.min(ax, bx) + 4, P.x + 6, P.x + P.w - wW - 6),
            ctx.clamp(ay + 16, P.y + 16, P.y + P.h - 6));
          chip(tl,
            bx + 8 + wL <= P.x + P.w - 6 ? bx + 8 : Math.max(P.x + 6, bx - 8 - wL),
            ctx.clamp((ay + by) / 2, P.y + 16, P.y + P.h - 6));

          /* ---- the arithmetic, spelled out ---- */
          const TX = 475;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('what you just measured', TX, 54);
          g.font = MONO; g.fillStyle = C.muted;
          let y = 82;
          const row = (lab, val, col) => {
            g.font = MONO; g.fillStyle = C.muted; g.fillText(lab, TX, y);
            g.font = 'bold ' + MONO; g.fillStyle = col || C.text; g.fillText(val, TX + 150, y);
            y += 20;
          };
          row('loss at w', L0.toFixed(3));
          row('loss at w + nudge', L1.toFixed(3));
          y += 4;
          row('so the loss moved', dL.toFixed(3), dL > 0 ? C.danger : C.green);
          row('and w moved', eps.toFixed(3), C.warn);
          y += 8;
          g.strokeStyle = C.line; g.beginPath(); g.moveTo(TX, y - 12); g.lineTo(TX + 205, y - 12); g.stroke();
          g.font = MONO; g.fillStyle = C.muted; g.fillText('divide one by the other', TX, y);
          y += 26;
          g.font = 'bold 24px Inter, system-ui, sans-serif';
          g.fillStyle = Math.abs(measured - m) < 0.05 ? C.green : C.warn;
          g.fillText(measured.toFixed(3), TX, y);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('that is the slope', TX + 90, y);

          /* how close the measurement is to the real tangent */
          y += 34;
          const err = Math.abs(measured - m);
          g.font = FONT; g.fillStyle = err < 0.02 ? C.green : err < 0.2 ? C.warn : C.danger;
          wrapText(g, err < 0.02
            ? 'The yellow line now sits on the grey one. Shrink the nudge and your measurement becomes the true slope.'
            : 'The yellow line is your measurement. The faint grey line is the true slope. Shrink the nudge to close the gap.',
            TX, y, 215, 16);

          /* the verdict, in words */
          g.font = 'bold ' + FONT;
          g.fillStyle = Math.abs(measured) < 0.05 ? C.green : measured > 0 ? C.danger : C.accent;
          wrapText(g, Math.abs(measured) < 0.05
            ? 'Slope ≈ 0. You are at the bottom. Nudging w either way makes things worse — there is nowhere better to go.'
            : measured > 0
              ? 'Slope is POSITIVE: raising w makes the loss worse. So to improve, go the other way — lower w.'
              : 'Slope is NEGATIVE: raising w makes the loss better. So to improve, raise w.',
            60, 330, 390, 18);

          ro.set({
            w: w.toFixed(2), 'nudge': eps.toFixed(3),
            'loss moved by': dL.toFixed(3),
            'slope = moved ÷ nudge': measured.toFixed(3),
          });
        });

        return ctx.figure(cv,
          'No calculus here — just subtraction and division. Move <b>w</b>, nudge it, and read how far the loss moved. Divide one by the other and you have the slope. The faint grey line is the true tangent: shrink the nudge and your measured yellow line lies right on top of it. That is genuinely all a derivative is, and this is not a simplification for teaching — it is <i>gradient checking</i>, which real engineers run to test whether their code computed the gradient correctly.',
          [wSl, eSl, tinyBtn, bigBtn, botBtn], ro);
      }

      /* ------------------------------------------------------------------ */
      /* Interactive B: gradient descent on a bumpy 1-D loss                  */
      /* ------------------------------------------------------------------ */
      function gradientDescent1D() {
        const W = 720, H = 338;      // room for the live numbers BELOW the plot, not on top of it
        const [cv, g] = ctx.canvas(W, H);
        const X0 = -6, X1 = 6;
        /* +1.1 keeps the whole curve above zero: a quantity called "loss" must never read negative */
        const f = (x) => 0.05 * x * x + 0.8 * Math.sin(1.3 * x) + 0.3 * Math.cos(2.7 * x) + 1.1;
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
          // everything that moves stays in the plot; the clip is let out by the ball's radius
          // so a ball parked on an axis limit is not sliced in half
          g.save(); g.beginPath(); g.rect(plot.x - 8, plot.y - 8, plot.w + 16, plot.h + 16); g.clip();
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
          g.restore();
          // key for the two guide lines: bottom-right of the plot, on an opaque chip so the
          // tangent cannot be drawn through it
          const key = 'red = slope under your feet · dashed = next step';
          g.font = MONO; g.textAlign = 'right';
          const kw = g.measureText(key).width, ky = plot.y + plot.h - 8;
          g.fillStyle = 'rgba(10,14,22,0.92)';
          g.fillRect(plot.x + plot.w - 12 - kw, ky - 13, kw + 8, 18);
          g.fillStyle = C.muted; g.fillText(key, plot.x + plot.w - 8, ky);
          // the live numbers go UNDER the plot, where nothing can be drawn over them and they
          // cannot collide with the key
          g.textAlign = 'left'; g.fillStyle = C.text; g.font = MONO;
          g.fillText('x = ' + f2(bx) + '   loss = ' + f3(by) + '   slope = ' + f2(slope) + '   steps = ' + S.steps, plot.x, plot.y + plot.h + 20);
          g.fillStyle = C.muted; g.font = FONT;
          wrapText(g, S.msg, plot.x, plot.y + plot.h + 42, plot.w, 15);   // wrapped: messages are long
        }
        function updateRO() { ro.set({ x: f2(S.x), loss: f3(f(S.x)), slope: f2(df(S.x)), steps: S.steps, 'learning rate': f2(S.lr) }); }
        const lrSl = ctx.slider({ label: 'learning rate η', min: 0.01, max: 1.5, step: 0.01, value: 0.3, fmt: f2, onChange: (v) => { S.lr = v; updateRO(); } });
        const stSl = ctx.slider({ label: 'start position', min: -6, max: 6, step: 0.1, value: 4.5, fmt: (v) => (+v).toFixed(1), onChange: (v) => { S.start = v; reset(); updateRO(); } });
        const stepBtn = ctx.button('Step', () => { S.running = false; runBtn.textContent = '▶ Run'; step(); });
        runBtn = ctx.button('▶ Run', () => { S.running = !S.running; runBtn.textContent = S.running ? '⏸ Pause' : '▶ Run'; }, 'primary');
        const resetBtn = ctx.button('Reset', reset);
        updateRO();
        ctx.loop((dt, t) => { if (S.running) { S.acc += dt * 5; while (S.acc >= 1) { S.acc -= 1; step(); } } draw(t); });
        return ctx.figure(cv, 'Gradient descent in one dimension: feel the slope, step the other way, repeat. The curve is <code class="inline">0.05x² + 0.8·sin(1.3x) + 0.3·cos(2.7x) + 1.1</code>, chosen for its bumps. Its four dips sit at x ≈ −5.74, −1.15, 0.96 and 3.44; only the second is the global minimum.', [lrSl, stSl, stepBtn, runBtn, resetBtn], ro);
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
        /* Every edge, described once. Exactly one label is shown at a time — the forward product
           while the green wave is on or past it, the gradient once the red wave arrives. */
        const EDGES = [];
        for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) {
          EDGES.push({ A: P.in[i], B: P.hid[j], rA: 20, rB: 22, at: 0.35, layer: 1, w: W1[j][i],
            fwd: f2(W1[j][i]) + '×' + f2(x[i]) + '=' + f2(W1[j][i] * x[i]),
            bwd: '∂L/∂w = ' + f3(gW1[j][i]) });
        }
        for (let j = 0; j < 2; j++) {
          EDGES.push({ A: P.hid[j], B: P.out, rA: 22, rB: 24, at: 0.5, layer: 2, w: W2[j],
            fwd: f2(W2[j]) + '×' + f2(hh[j]) + '=' + f2(W2[j] * hh[j]),
            bwd: '∂L/∂w = ' + f3(gW2[j]) });
        }
        /* A wire stops at each node's rim, so it can never run under a node's own label. */
        function wireEnds(e) {
          const ex = e.B[0] - e.A[0], ey = e.B[1] - e.A[1], en = Math.hypot(ex, ey) || 1;
          return [e.A[0] + ex / en * e.rA, e.A[1] + ey / en * e.rA,
            e.B[0] - ex / en * e.rB, e.B[1] - ey / en * e.rB];
        }
        /* Numbers sit ON their wire, so each gets an opaque chip: a sloping wire would otherwise
           be drawn straight through its own label. */
        function chipText(txt, cx, cy, col) {
          const cw = g.measureText(txt).width;
          g.fillStyle = 'rgba(10,14,22,0.92)';
          g.fillRect(cx - cw / 2 - 4, cy - 12, cw + 8, 17);
          g.fillStyle = col; g.fillText(txt, cx, cy);
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
          /* three passes, so nothing is ever painted across a number: every wire, then every
             label on its chip, then the travelling dots on top */
          g.strokeStyle = C.line; g.lineWidth = 2;
          for (const e of EDGES) { const q = wireEnds(e); g.beginPath(); g.moveTo(q[0], q[1]); g.lineTo(q[2], q[3]); g.stroke(); }
          g.font = MONO; g.textAlign = 'center';
          for (const e of EDGES) {
            const fu = e.layer === 1 ? f1u : f2u, bu = e.layer === 1 ? b1u : b2u;
            const lx = e.A[0] + (e.B[0] - e.A[0]) * e.at, ly = e.A[1] + (e.B[1] - e.A[1]) * e.at;
            if (bu >= 0.55) { chipText('w = ' + f2(e.w), lx, ly - 7, C.muted); chipText(e.bwd, lx, ly + 14, C.danger); }
            else if (fu >= 0.55) chipText(e.fwd, lx, ly - 7, C.green);
          }
          for (const e of EDGES) {
            const fu = e.layer === 1 ? f1u : f2u, bu = e.layer === 1 ? b1u : b2u, q = wireEnds(e);
            if (fu > 0 && fu < 1) { g.beginPath(); g.arc(q[0] + (q[2] - q[0]) * fu, q[1] + (q[3] - q[1]) * fu, 6, 0, Math.PI * 2); g.fillStyle = C.green; g.fill(); }
            if (bu > 0 && bu < 1) { const u = 1 - bu; g.beginPath(); g.arc(q[0] + (q[2] - q[0]) * u, q[1] + (q[3] - q[1]) * u, 6, 0, Math.PI * 2); g.fillStyle = C.danger; g.fill(); }
          }
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

      /* ------------------------------------------------------------------ */
      /* Interactive D: solve XOR by hand with two lines and a combiner      */
      /* ------------------------------------------------------------------ */
      function xorByHand() {
        const [cv, g] = ctx.canvas(720, 360);
        const PTS = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]; // x1, x2, target (XOR)
        /* two lines, each held by two draggable endpoints in unit coords */
        const lines = [
          { a: [0.05, 0.90], b: [0.90, 0.05], flip: false },   // fires when x1 + x2 > 0.95
          { a: [0.05, 0.30], b: [0.30, 0.05], flip: false },   // fires when x1 + x2 > 0.35
        ];
        const RULES = [
          { value: 'and',    label: 'A AND B',     fn: (A, B) => A && B },
          { value: 'or',     label: 'A OR B',      fn: (A, B) => A || B },
          { value: 'andnot', label: 'A AND NOT B', fn: (A, B) => A && !B },
          { value: 'notand', label: 'NOT A AND B', fn: (A, B) => !A && B },
        ];
        let rule = 'and';
        const PAD = 30, SQ = 300;           // plot square: 30..330
        const ux = (u) => PAD + u * SQ;
        const uy = (v) => PAD + SQ - v * SQ;
        const xu = (px) => (px - PAD) / SQ;
        const yu = (py) => (PAD + SQ - py) / SQ;

        function fires(L, x, y) {
          const s = (L.b[0] - L.a[0]) * (y - L.a[1]) - (L.b[1] - L.a[1]) * (x - L.a[0]);
          return L.flip ? s < 0 : s > 0;
        }
        /* the infinite line through two pixel points, trimmed to the plot square (Liang–Barsky).
           Both handles live inside the square, so the drawn line always starts and ends on the
           frame: it can never run across the truth table or off the canvas. */
        function boxSpan(ax, ay, bx, by) {
          const dx = bx - ax, dy = by - ay;
          if (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6) return null;
          let t0 = -1e6, t1 = 1e6;
          const p = [-dx, dx, -dy, dy];
          const q = [ax - PAD, PAD + SQ - ax, ay - PAD, PAD + SQ - ay];
          for (let i = 0; i < 4; i++) {
            if (p[i] === 0) { if (q[i] < 0) return null; continue; }
            const t = q[i] / p[i];
            if (p[i] < 0) { if (t > t1) return null; if (t > t0) t0 = t; }
            else { if (t < t0) return null; if (t < t1) t1 = t; }
          }
          return { x0: ax + t0 * dx, y0: ay + t0 * dy, x1: ax + t1 * dx, y1: ay + t1 * dy };
        }
        function out(x, y) {
          const f = RULES.find(r => r.value === rule).fn;
          return f(fires(lines[0], x, y), fires(lines[1], x, y)) ? 1 : 0;
        }
        function score() { return PTS.filter(([x, y, t]) => out(x, y) === t).length; }

        /* drag state */
        let drag = null;
        const handles = () => [
          { L: lines[0], k: 'a' }, { L: lines[0], k: 'b' },
          { L: lines[1], k: 'a' }, { L: lines[1], k: 'b' },
        ];
        function pick(p) {
          let best = null, bd = 20 * 20;
          for (const hh of handles()) {
            const dx = ux(hh.L[hh.k][0]) - p.x, dy = uy(hh.L[hh.k][1]) - p.y;
            const d = dx * dx + dy * dy;
            if (d < bd) { bd = d; best = hh; }
          }
          return best;
        }
        cv.addEventListener('pointerdown', (e) => {
          e.preventDefault();
          drag = pick(cv.pos(e));
          if (drag && cv.setPointerCapture) { try { cv.setPointerCapture(e.pointerId); } catch (err) {} }
        });
        cv.addEventListener('pointermove', (e) => {
          if (!drag) return;
          const p = cv.pos(e);
          drag.L[drag.k] = [ctx.clamp(xu(p.x), 0, 1), ctx.clamp(yu(p.y), 0, 1)];
        });
        const stop = () => { drag = null; };
        cv.addEventListener('pointerup', stop);
        cv.addEventListener('pointercancel', stop);
        cv.addEventListener('pointerleave', stop);

        /* built by hand rather than with ctx.select: the buttons below need to write the
           value back into the dropdown, and ctx.select exposes a getter only. */
        const ruleEl = h('select', {}, RULES.map(r => h('option', { value: r.value }, r.label)));
        ruleEl.value = 'and';
        ruleEl.addEventListener('change', () => { rule = ruleEl.value; });
        const setRule = (v) => { rule = v; ruleEl.value = v; };
        const ruleSel = h('div', { class: 'control' }, h('label', {}, 'Output neuron rule'), ruleEl);
        const flipA = ctx.button('Flip line A', () => { lines[0].flip = !lines[0].flip; });
        const flipB = ctx.button('Flip line B', () => { lines[1].flip = !lines[1].flip; });
        const solveBtn = ctx.button('Show me one answer', () => {
          drag = null;
          lines[0] = { a: [0.05, 0.45], b: [0.45, 0.05], flip: false };   // fires when x1 + x2 > 0.5
          lines[1] = { a: [0.55, 0.95], b: [0.95, 0.55], flip: false };   // fires when x1 + x2 > 1.5
          setRule('andnot');                                             // "exactly one switch on"
        }, 'primary');
        const resetBtn = ctx.button('Reset', () => {
          drag = null;
          lines[0] = { a: [0.05, 0.90], b: [0.90, 0.05], flip: false };
          lines[1] = { a: [0.05, 0.30], b: [0.30, 0.05], flip: false };
          setRule('and');
        });
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          /* shaded region where the network says 1 */
          const N = 50, cell = SQ / N;
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              const x = (i + 0.5) / N, y = (j + 0.5) / N;
              g.fillStyle = out(x, y) ? 'rgba(251,113,133,0.20)' : 'rgba(124,156,255,0.13)';
              g.fillRect(PAD + i * cell, PAD + SQ - (j + 1) * cell, cell + 0.6, cell + 0.6);
            }
          }
          g.strokeStyle = C.line; g.lineWidth = 1;
          g.strokeRect(PAD, PAD, SQ, SQ);

          /* the two lines — drawn only across the square they divide, and named in the
             margin at the edge they leave through, so no label sits under a line */
          [[lines[0], C.warn, 'A'], [lines[1], C.purple, 'B']].forEach(([L, col, name]) => {
            const span = boxSpan(ux(L.a[0]), uy(L.a[1]), ux(L.b[0]), uy(L.b[1]));
            g.save();
            g.beginPath(); g.rect(PAD - 9, PAD - 9, SQ + 18, SQ + 18); g.clip();  // +9 = handle radius
            if (span) {
              g.strokeStyle = col; g.lineWidth = 2.5;
              g.beginPath(); g.moveTo(span.x0, span.y0); g.lineTo(span.x1, span.y1); g.stroke();
            }
            g.fillStyle = col;
            [L.a, L.b].forEach(pt => {
              g.beginPath(); g.arc(ux(pt[0]), uy(pt[1]), 7, 0, 7); g.fill();
              g.strokeStyle = '#fff'; g.lineWidth = 2; g.stroke();
            });
            g.restore();
            if (span) {
              g.font = 'bold ' + FONT; g.fillStyle = col;
              const ex = span.x1, ey = span.y1;
              if (ey <= PAD + 0.5) g.fillText(name, ex - 4, PAD - 8);
              else if (ey >= PAD + SQ - 0.5) g.fillText(name, ex - 4, PAD + SQ + 22);
              else if (ex <= PAD + 0.5) g.fillText(name, PAD - 20, ey + 5);
              else g.fillText(name, PAD + SQ + 9, ey + 5);
            }
          });

          /* the four data points */
          PTS.forEach(([x, y, t]) => {
            const correct = out(x, y) === t;
            g.beginPath(); g.arc(ux(x), uy(y), 11, 0, 7);
            g.fillStyle = t ? C.danger : C.accent; g.fill();
            g.lineWidth = 3; g.strokeStyle = correct ? C.green : '#fff'; g.stroke();
            if (!correct) {
              g.strokeStyle = '#fff'; g.lineWidth = 2.5;
              g.beginPath();
              g.moveTo(ux(x) - 5, uy(y) - 5); g.lineTo(ux(x) + 5, uy(y) + 5);
              g.moveTo(ux(x) + 5, uy(y) - 5); g.lineTo(ux(x) - 5, uy(y) + 5);
              g.stroke();
            }
          });

          /* truth table on the right */
          let ty = 58;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('what each neuron says', 370, 36);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('x₁  x₂', 370, ty);
          g.fillText('A', 442, ty); g.fillText('B', 478, ty);
          g.fillText('out', 520, ty); g.fillText('want', 570, ty);
          ty += 8;
          g.strokeStyle = C.line; g.lineWidth = 1;
          g.beginPath(); g.moveTo(370, ty); g.lineTo(640, ty); g.stroke();
          ty += 22;
          PTS.forEach(([x, y, t]) => {
            const A = fires(lines[0], x, y), B = fires(lines[1], x, y), o = out(x, y);
            g.font = MONO;
            g.fillStyle = t ? C.danger : C.accent;
            g.fillText(' ' + x + '   ' + y, 370, ty);
            g.fillStyle = A ? C.warn : C.muted; g.fillText(A ? '1' : '0', 445, ty);
            g.fillStyle = B ? C.purple : C.muted; g.fillText(B ? '1' : '0', 481, ty);
            g.fillStyle = C.text; g.fillText(String(o), 526, ty);
            g.fillStyle = C.muted; g.fillText(String(t), 578, ty);
            g.fillStyle = o === t ? C.green : C.danger;
            g.font = 'bold ' + FONT;
            g.fillText(o === t ? '✓' : '✗', 616, ty);
            ty += 26;
          });
          const s = score();
          g.font = 'bold 22px Inter, system-ui, sans-serif';
          g.fillStyle = s === 4 ? C.green : C.text;
          g.fillText(s + ' / 4', 370, 250);
          g.font = FONT; g.fillStyle = s === 4 ? C.green : C.muted;
          wrapText(g, s === 4
            ? 'Solved. One straight line could never do this. Two lines and a combiner just did.'
            : 'Drag the four white-ringed handles to move lines A and B. Then try each output rule.',
            370, 276, 280, 17);
          ro.set({ score: s + '/4', rule: RULES.find(r => r.value === rule).label });
        });

        return ctx.figure(cv, 'The XOR puzzle that defeated you in chapter 1, now with two lines instead of one. Line A and line B are each a neuron: every point is on one side or the other, so each reports a plain 1 or 0 (the A and B columns). The output neuron never sees x₁ or x₂ — it only sees those two answers, and applies one simple rule to them. The shaded region is what the whole three-neuron network predicts.', [ruleSel, flipA, flipB, solveBtn, resetBtn], ro);
      }

      /* ------------------------------------------------------------------ */
      /* Interactive E: why a stack of linear layers stays linear            */
      /* ------------------------------------------------------------------ */
      function activationLab() {
        const [cv, g] = ctx.canvas(720, 380);
        const ACTS = {
          linear:  { label: 'linear (none)', f: (z) => z,                       col: '#fb7185' },
          sigmoid: { label: 'sigmoid',       f: (z) => 1 / (1 + Math.exp(-z)),  col: '#7c9cff' },
          tanh:    { label: 'tanh',          f: (z) => Math.tanh(z),            col: '#38d9a9' },
          relu:    { label: 'ReLU',          f: (z) => Math.max(0, z),          col: '#fbbf24' },
        };
        let act = 'tanh', depth = 3, probe = 0.8;
        let W = [];
        function reroll() {
          W = [];
          for (let i = 0; i < 6; i++) W.push({ w: ctx.rand(-2.2, 2.2), b: ctx.rand(-1.4, 1.4) });
        }
        reroll();
        /* pass x through `depth` layers of (weight, bias, activation) */
        function net(x) {
          let v = x;
          for (let i = 0; i < depth; i++) {
            v = W[i].w * v + W[i].b;
            if (i < depth - 1) v = ACTS[act].f(v);   // no squash on the final output
          }
          return v;
        }

        const actSel = ctx.select({
          label: 'Activation', value: 'tanh',
          options: Object.keys(ACTS).map(k => ({ value: k, label: ACTS[k].label })),
          onChange: (v) => { act = v; },
        });
        const depthSl = ctx.slider({ label: 'Layers', min: 2, max: 6, step: 1, value: 3, onChange: (v) => { depth = v; } });
        const probeSl = ctx.slider({ label: 'input z', min: -4, max: 4, step: 0.05, value: 0.8, digits: 2, onChange: (v) => { probe = v; } });
        const rollBtn = ctx.button('New random weights', reroll, 'primary');
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          const A = ACTS[act];

          /* ---- left: the activation function itself ---- */
          const L = { x: 40, y: 40, w: 270, h: 250 };
          const lx = (z) => L.x + (z + 4) / 8 * L.w;
          const ly = (a) => L.y + L.h - (a + 1.5) / 4.5 * L.h;
          g.strokeStyle = C.line; g.lineWidth = 1;
          g.beginPath(); g.moveTo(L.x, ly(0)); g.lineTo(L.x + L.w, ly(0)); g.stroke();
          g.beginPath(); g.moveTo(lx(0), L.y); g.lineTo(lx(0), L.y + L.h); g.stroke();
          g.strokeStyle = A.col; g.lineWidth = 2.5; g.beginPath();
          for (let i = 0; i <= 160; i++) {
            const z = -4 + i / 160 * 8, a = ctx.clamp(A.f(z), -1.6, 3.1);
            i ? g.lineTo(lx(z), ly(a)) : g.moveTo(lx(z), ly(a));
          }
          g.stroke();
          const pa = A.f(probe);
          g.setLineDash([3, 3]); g.strokeStyle = C.muted; g.lineWidth = 1;
          g.beginPath(); g.moveTo(lx(probe), L.y + L.h); g.lineTo(lx(probe), ly(ctx.clamp(pa, -1.6, 3.1))); g.stroke();
          g.setLineDash([]);
          g.fillStyle = A.col;
          g.beginPath(); g.arc(lx(probe), ly(ctx.clamp(pa, -1.6, 3.1)), 6, 0, 7); g.fill();
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('the squash: one neuron', L.x, 26);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('in ' + f2(probe) + '  →  out ' + f2(pa), L.x, L.y + L.h + 24);

          /* ---- right: what a whole stack can draw ---- */
          const R = { x: 390, y: 40, w: 290, h: 250 };
          let lo = Infinity, hi = -Infinity;
          const ys = [];
          for (let i = 0; i <= 220; i++) {
            const x = -4 + i / 220 * 8;
            let v = net(x);
            if (!isFinite(v)) v = 0;
            v = ctx.clamp(v, -60, 60);
            ys.push(v); if (v < lo) lo = v; if (v > hi) hi = v;
          }
          if (hi - lo < 1e-6) { lo -= 1; hi += 1; }
          const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
          const rx = (x) => R.x + (x + 4) / 8 * R.w;
          const ry = (v) => R.y + R.h - (v - lo) / (hi - lo) * R.h;
          g.strokeStyle = C.line; g.lineWidth = 1;
          g.strokeRect(R.x, R.y, R.w, R.h);
          if (lo < 0 && hi > 0) { g.beginPath(); g.moveTo(R.x, ry(0)); g.lineTo(R.x + R.w, ry(0)); g.stroke(); }
          g.strokeStyle = A.col; g.lineWidth = 2.5; g.beginPath();
          ys.forEach((v, i) => { const x = -4 + i / 220 * 8; i ? g.lineTo(rx(x), ry(v)) : g.moveTo(rx(x), ry(v)); });
          g.stroke();
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText(depth + ' layers stacked, end to end', R.x, 26);

          /* straightness test: max deviation from the straight line through the endpoints */
          let dev = 0;
          for (let i = 0; i <= 220; i++) {
            const t = i / 220, straight = ys[0] + (ys[220] - ys[0]) * t;
            dev = Math.max(dev, Math.abs(ys[i] - straight));
          }
          const rel = dev / Math.max(1e-9, hi - lo);
          const bent = rel > 0.01;
          g.font = 'bold 15px Inter, system-ui, sans-serif';
          g.fillStyle = bent ? C.green : C.danger;
          g.fillText(bent ? 'bent — this can fold' : 'perfectly straight — no matter what', R.x, R.y + R.h + 26);
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, bent
            ? 'Press "New random weights" all day: with a squash in the middle, the stack keeps producing shapes one line cannot.'
            : 'Press "New random weights" all day. Add layers. It stays a straight line. That is the 1969 wall, rebuilt.',
            R.x, R.y + R.h + 46, R.w, 17);
          ro.set({ activation: A.label, layers: depth, 'bendiness': (rel * 100).toFixed(1) + '%' });
        });

        return ctx.figure(cv, 'Left: the activation function on its own — the little squash applied after each neuron\'s weighted sum. Right: what you get when you chain several weight-and-bias layers together with that squash between them. "Bendiness" measures how far the output curve strays from a straight line. Choose <b>linear (none)</b> and it is pinned at 0.0% forever, however many layers you stack.', [actSel, depthSl, probeSl, rollBtn], ro);
      }

      /* ------------------------------------------------------------------ */
      /* Interactive F: what "wrong" should cost — squared error vs log loss */
      /* ------------------------------------------------------------------ */
      function lossLab() {
        const [cv, g] = ctx.canvas(720, 330);
        let p = 0.5, target = 1;
        const sq = (pp) => 0.5 * (pp - target) * (pp - target);
        const ce = (pp) => -(target * Math.log(Math.max(1e-9, pp)) + (1 - target) * Math.log(Math.max(1e-9, 1 - pp)));

        const pSl = ctx.slider({ label: 'model says P(correct class)', min: 0.01, max: 0.99, step: 0.01, value: 0.5, digits: 2, onChange: (v) => { p = v; } });
        const tSel = ctx.select({
          label: 'true answer', value: '1',
          options: [{ value: '1', label: 'class 1 (yes)' }, { value: '0', label: 'class 0 (no)' }],
          onChange: (v) => { target = +v; },
        });
        const confBtn = ctx.button('Confidently wrong (1%)', () => { p = 0.01; pSl.value = 0.01; });
        const hedgeBtn = ctx.button('Hedge (50%)', () => { p = 0.5; pSl.value = 0.5; });
        const rightBtn = ctx.button('Confidently right (99%)', () => { p = 0.99; pSl.value = 0.99; }, 'primary');
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          const P = { x: 55, y: 36, w: 380, h: 230 };
          const px = (v) => P.x + v * P.w;
          const py = (v) => P.y + P.h - ctx.clamp(v, 0, 5) / 5 * P.h;
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
          g.font = MONO; g.fillStyle = C.muted;
          for (let v = 0; v <= 5; v++) {
            g.beginPath(); g.moveTo(P.x, py(v)); g.lineTo(P.x + P.w, py(v)); g.stroke();
            g.fillText(String(v), P.x - 16, py(v) + 4);
          }
          ['0', '0.5', '1'].forEach((t, i) => g.fillText(t, px(i / 2) - 8, P.y + P.h + 18));

          /* the two loss curves */
          [[ce, C.danger, 'cross-entropy'], [sq, C.accent, 'squared error']].forEach(([fn, col]) => {
            g.strokeStyle = col; g.lineWidth = 2.5; g.beginPath();
            for (let i = 0; i <= 200; i++) {
              const v = 0.002 + i / 200 * 0.996;
              i ? g.lineTo(px(v), py(fn(v))) : g.moveTo(px(v), py(fn(v)));
            }
            g.stroke();
          });
          g.setLineDash([4, 4]); g.strokeStyle = C.text; g.lineWidth = 1.5;
          g.beginPath(); g.moveTo(px(p), P.y); g.lineTo(px(p), P.y + P.h); g.stroke();
          g.setLineDash([]);
          [[ce, C.danger], [sq, C.accent]].forEach(([fn, col]) => {
            g.fillStyle = col;
            g.beginPath(); g.arc(px(p), py(fn(p)), 6, 0, 7); g.fill();
          });
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('what one wrong answer costs you', P.x, 24);
          g.font = MONO; g.fillStyle = C.muted;
          g.save(); g.translate(20, P.y + P.h / 2 + 30); g.rotate(-Math.PI / 2);
          g.fillText('loss', 0, 0); g.restore();
          g.fillText('probability the model gave the right answer', P.x + 60, P.y + P.h + 36);

          /* right-hand numbers */
          const X = 470;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('this prediction', X, 24);
          const cev = ce(p), sqv = sq(p);
          const rows = [
            ['cross-entropy', cev.toFixed(3), C.danger],
            ['squared error', sqv.toFixed(3), C.accent],
          ];
          let y = 62;
          rows.forEach(([lab, val, col]) => {
            g.font = FONT; g.fillStyle = C.muted; g.fillText(lab, X, y);
            g.font = 'bold 20px Inter, system-ui, sans-serif'; g.fillStyle = col;
            g.fillText(val, X, y + 26);
            y += 62;
          });
          /* how hard each loss shoves the model */
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('how hard it shoves', X, y + 4);
          g.font = MONO; g.fillStyle = C.danger;
          g.fillText('cross-entropy: ' + (1 / Math.max(0.01, p)).toFixed(1) + '×', X, y + 26);
          g.fillStyle = C.accent;
          g.fillText('squared error: ' + Math.abs(p - target).toFixed(2) + '×', X, y + 46);
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, p < 0.15
            ? 'Badly wrong. Cross-entropy screams; squared error barely raises its voice.'
            : 'Drag toward 0.01 and watch the red curve run off the top of the chart.',
            X, y + 72, 220, 17);
          ro.set({ 'P(right)': p.toFixed(2), 'cross-entropy': cev.toFixed(3), 'squared err': sqv.toFixed(3) });
        });

        return ctx.figure(cv, 'Both curves are zero when the model is certain and right (far right) and rise as it gets things wrong. They part company at the left edge: squared error tops out at 0.5, while cross-entropy goes to infinity. "How hard it shoves" is the size of the gradient — the push the wrong weights receive. A model that is 1% sure of the right answer gets a 100× shove from cross-entropy and almost nothing from squared error.', [pSl, tSel, confBtn, hedgeBtn, rightBtn], ro);
      }

      /* ------------------------------------------------------------------ */
      /* Interactive G: counting weights, from toy to frontier               */
      /* ------------------------------------------------------------------ */
      function paramScale() {
        const [cv, g] = ctx.canvas(720, 320);
        let width = 8, depth = 1, inputs = 2;
        function params() {
          let total = 0, prev = inputs;
          for (let i = 0; i < depth; i++) { total += prev * width + width; prev = width; }
          total += prev * 1 + 1;
          return total;
        }
        const REF = [
          { name: 'your network', get: params, col: '#38d9a9' },
          { name: 'LeNet-5, 1998 (digits)', v: 60e3, col: '#7c9cff' },
          { name: 'AlexNet, 2012 (ImageNet)', v: 62e6, col: '#a78bfa' },
          { name: 'GPT-2, 2019', v: 1.5e9, col: '#fbbf24' },
          { name: 'GPT-3, 2020', v: 175e9, col: '#fb923c' },
          { name: 'frontier model, today', v: 1e12, col: '#fb7185' },
        ];
        const wSl = ctx.slider({ label: 'hidden units per layer', min: 2, max: 512, step: 1, value: 8, onChange: (v) => { width = v; } });
        const dSl = ctx.slider({ label: 'hidden layers', min: 1, max: 8, step: 1, value: 1, onChange: (v) => { depth = v; } });
        const iSl = ctx.slider({ label: 'inputs', min: 2, max: 784, step: 1, value: 2, onChange: (v) => { inputs = v; } });
        const mnistBtn = ctx.button('Match the 1989 postcode reader', () => {
          inputs = 256; iSl.value = 256; width = 30; wSl.value = 30; depth = 1; dSl.value = 1;
        }, 'primary');
        const ro = ctx.readout();
        const human = (n) => n >= 1e12 ? (n / 1e12).toFixed(1) + ' trillion'
          : n >= 1e9 ? (n / 1e9).toFixed(1) + ' billion'
          : n >= 1e6 ? (n / 1e6).toFixed(1) + ' million'
          : n >= 1e3 ? (n / 1e3).toFixed(1) + ' thousand'
          : String(Math.round(n));

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('every weight is one number the chain rule has to supply a gradient for', 40, 26);
          const X = 260, W = 340, top = 52, rowH = 40;   // W leaves room for the count beside a full bar
          REF.forEach((r, i) => {
            const v = r.get ? r.get() : r.v;
            const frac = Math.log10(Math.max(1, v)) / 12;      // log scale, 10^0 … 10^12
            const y = top + i * rowH;
            g.font = FONT; g.fillStyle = r.get ? C.green : C.muted;
            g.fillText(r.name, 40, y + 16);
            g.fillStyle = r.col;
            g.globalAlpha = r.get ? 1 : 0.65;
            g.fillRect(X, y + 4, Math.max(3, frac * W), 16);
            g.globalAlpha = 1;
            g.font = MONO; g.fillStyle = r.get ? C.green : C.muted;
            g.fillText(human(v), X + Math.max(3, frac * W) + 10, y + 17);
          });
          g.font = MONO; g.fillStyle = C.line;
          g.fillText('bar length is logarithmic: each equal step is 10× more weights', 40, top + REF.length * rowH + 18);
          ro.set({ shape: inputs + ' → ' + Array(depth).fill(width).join(' → ') + ' → 1', weights: human(params()) });
        });

        return ctx.figure(cv, 'A network\'s parameter count is just bookkeeping: every connection between two layers is one weight, plus one bias per neuron. Push the sliders and watch your toy climb the chart. The bars are logarithmic, so the gap between your network and a frontier model is far larger than it looks — but the training loop is character-for-character the same one running in the playground above.', [iSl, wSl, dSl, mnistBtn], ro);
      }

      /* ================================================================== */
      /* The chapter: touch first, read second.                             */
      /* ================================================================== */
      root.append(
        callout('tryit', '🖐 Do this first — you failed this exact puzzle in chapter 1',
          `Same four dots, same impossible task: <b>red dots on the red side, blue dots on the blue side</b>.
           But now you get <b>two</b> lines instead of one, and a third neuron that combines their answers.<br>
           <b>1.</b> Drag the four white-ringed handles to place lines A and B.
           <b>2.</b> Try each of the four output rules in the dropdown. <b>Get 4 / 4.</b><br>
           It is very doable. If you have been at it two minutes, press <b>Show me one answer</b> and study what it did.`),
        xorByHand(),
        p(`That was impossible twenty minutes ago. Nothing about the dots changed and no line got cleverer — you just stopped asking one neuron to do all the work.`),
      );

      root.append(section('What you just built',
        p(`Look at the A and B columns you were filling in. Those two neurons never answered the question. They answered <i>easier</i> questions — "is at least one switch on?" and "are both switches on?" — and the third neuron answered the real one using only their two replies.`),
        p(`The middle neurons are a <em>hidden layer</em>. Hidden because you never read their output directly; it is the network's private vocabulary, invented to make the final question easy. Stack an input layer, one or more hidden layers and an output layer, wire every neuron to every neuron in the next layer, and you have a <em>multi-layer perceptron</em> — an MLP, or feed-forward network.`),
        callout('key', '🔑 The idea that broke the 1969 wall',
          `One neuron draws one straight line. <b>Two neurons draw two lines, and a neuron above them can carve out the strip between them.</b>
           A strip is not something one line can make. Add more hidden neurons and you get regions of any shape you like.`),
        p(`How much shape? With enough hidden neurons, one hidden layer can approximate any reasonable function as closely as you want. That is a genuine theorem (Cybenko 1989, Hornik 1991). It is also almost useless on its own, because it says nothing about how to <i>find</i> the weights. Finding them is the rest of this chapter.`),
      ));

      root.append(section('The squash between layers is not decoration',
        callout('tryit', '🖐 Try this — break it on purpose',
          `Set <b>Activation</b> to <b>linear (none)</b> and press <b>New random weights</b> ten times. Drag <b>Layers</b> up to 6.<br>
           <b>Watch the "bendiness" number.</b> It stays at 0.0%, forever, whatever you do.
           Now switch to <b>tanh</b> or <b>ReLU</b> and roll again.`),
        activationLab(),
        p(`With no squash, six layers produced exactly what one layer produces: a straight line. That is not a quirk of the random numbers. It is arithmetic.`),
        p(`If layer one computes h = W<sub>1</sub>·x and layer two computes y = W<sub>2</sub>·h, then y = (W<sub>2</sub>W<sub>1</sub>)·x. Two matrices multiplied together are just a third matrix. A single matrix means a single flat boundary, which is the 1969 wall rebuilt out of a thousand neurons.`),
        p(`The little non-linear function applied after each weighted sum — the <em>activation function</em> — is what stops the collapse. It is the hinge. Every curved boundary you will ever see a network draw is made of straight cuts, folded at those hinges.`),
        callout('example', '🌍 Which hinge, in practice',
          `<b>Sigmoid</b> squashes to (0, 1) and reads as a probability, so it survives at the output of yes/no classifiers.
           <b>tanh</b> is the same shape centred on zero and trains better in hidden layers.
           <b>ReLU</b> — literally <code class="inline">max(0, z)</code> — looks too crude to work, and won:
           it is cheap, and its slope is exactly 1 for positive inputs, which keeps gradients from fading in deep stacks.
           Modern language models use polished relatives of ReLU called GELU and SwiGLU.`),
      ));

      root.append(section('Giving "wrong" a number',
        p(`The perceptron rule in chapter 1 only knew right from wrong. That is too coarse to train a million weights. We need a single number that says <i>how</i> wrong the network is — one that shrinks as predictions improve and changes smoothly when a weight moves. That is the <em>loss function</em>, and the choice of which one has consequences.`),
        callout('tryit', '🖐 Try this',
          `Press <b>Confidently wrong (1%)</b>. Read both numbers.<br>
           Cross-entropy charges about <b>4.6</b>. Squared error charges <b>0.49</b> — barely more than the <b>0.125</b> it charges for shrugging and saying 50%.<br>
           Now drag the slider slowly from right to left and watch the red curve leave the top of the chart.`),
        lossLab(),
        p(`Squared error treats a confident mistake as only slightly worse than a shrug. Cross-entropy treats it as a catastrophe, because it charges you <b>−log(probability you gave the right answer)</b>, and the log of a small number is enormous.`),
        p(`That is exactly the incentive you want. A model that says "99% sure" should be punished far harder for being wrong than one that admitted it was guessing. Cross-entropy is why language models end up roughly honest about their own uncertainty.`),
        callout('key', '🔑 The whole of training in one line',
          `Every model in this course — the toy below, ResNet, GPT — is trained by the same loop:
           <b>predict, measure the error with a loss function, ask the chain rule which way each weight should move, move it a little, repeat.</b>
           Everything else (architecture, optimizers, data curation, RLHF) is a refinement of one of those five verbs.`),
      ));

      root.append(section('Walking downhill in thick fog',
        p(`Training now has a crisp definition: <b>find the weights that make the average loss as small as possible.</b> Picture the loss as a landscape stretched over every possible setting of the weights. Training is a hunt for the lowest valley, done blindfolded.`),
        p(`You cannot see the valley. You can only feel the slope under your feet. So take a step in the steepest downhill direction, feel again, step again. That is <em>gradient descent</em>, and the <em>gradient</em> is that slope: for each weight, how much would the loss rise if I nudged this weight upward?`),
        callout('tryit', '🖐 Try this: three ways to fail at walking downhill',
          `<b>1.</b> Start at 4.5 with η = 0.3 and press Step a few times. The ball settles in the dip near x ≈ 3.4 and stops — a <em>local minimum</em>. The slope there is zero, so the rule has nothing left to say, even though a deeper valley sits to the left.<br>
           <b>2.</b> Set η to 1.2 and press ▶ Run: the ball ricochets between hillsides, sometimes landing higher than it started.<br>
           <b>3.</b> Set η to 0.02: it crawls.<br>
           <b>4.</b> Find a start position from which η = 0.3 reaches the deepest valley.<br>
           <b>Notice:</b> that red tangent line is the only information the algorithm ever has.`),
        gradientDescent1D(),

        /* maths beat 1, guided: measure a slope by hand, THEN meet the symbol. */
      ));

      root.append(section('The one piece of calculus this course needs',
        p(`That red tangent line has a name. Before the name, do it yourself — and you will not need any calculus, because you are going to <b>measure</b> the slope with subtraction and division.`),
        callout('tryit', '🖐 Try this — measure a slope with arithmetic you already have',
          `<b>1.</b> Change nothing yet. Read the right-hand column top to bottom: the loss at <b>w</b>, the loss a nudge later, how far it moved, how far you nudged. Divide. That is the slope.<br>
           <b>2.</b> Press <b>make the nudge tiny</b> and watch the yellow line settle onto the faint grey one. The grey line is the true slope; your measurement has become the same thing.<br>
           <b>3.</b> Drag <b>w</b> to the left of the dip and read the sentence at the bottom. Then drag to the right. <b>The sign of that number is the only thing gradient descent ever knows.</b><br>
           <b>4.</b> Press <b>go to the bottom</b>. The slope goes to zero and the sentence changes to say there is nowhere better to go.`),
        nudgeLab(),
        ctx.walkthrough([
          { say: 'You have a number you are allowed to change. Here it is called <b>w</b> — a weight.', note: 'A real model has billions of them. The idea does not change.' },
          { say: 'Changing it changes how wrong the model is. That wrongness is the <b>loss</b>, written <b>L</b>.', note: 'At w = 2.6 in the demo above, the loss reads 1.78.' },
          { say: 'You want to know one thing only: <b>should I make w bigger, or smaller?</b>', note: 'Not by how much yet. Just which direction.' },
          { say: 'So you tried it. You nudged w up a hair and watched what the loss did.', math: 'nudge w by 0.01 &rarr; the loss moved by 0.016' },
          { ask: 'The loss went <b>up</b> when you raised w. So what should you do to w?',
            options: ['Raise it further', 'Lower it', 'Leave it alone'], answer: 1,
            explain: 'Raising it made things worse, so the improvement is the other way. That one deduction is the whole of gradient descent.' },
          { say: 'Now put a number on <i>how strongly</i>: divide how far the loss moved by how far you nudged it.', math: '0.016 &divide; 0.01 = 1.6' },
          { say: 'That is the slope. A big number means this weight matters a lot right here. Near zero means it barely matters at all.', note: 'You have just computed a derivative, using division.' },
          { say: 'It has a symbol, and the symbol means exactly what you did. Nothing more.', math: '&part;L / &part;w &nbsp;=&nbsp; 1.6' },
          { say: 'Read it as "if I nudge w a hair, how much does L move?" The &part; is a curly <b>d</b> — curly for one reason only: L depends on many weights and we are moving one at a time.', note: 'If straight-d calculus once defeated you, this is not a harder version of it. It is the same idea with more things held still.' },
          { say: 'Do that for <i>every</i> weight at once and you get one number per weight. That list of numbers has a name: the <b>gradient</b>.', note: 'Gradient just means "all the slopes, together".' },
          { ask: 'A weight\'s slope comes out <b>negative</b>. Which way does gradient descent move that weight?',
            options: ['Down', 'Up', 'It depends on the learning rate'], answer: 1,
            explain: 'A negative slope means raising the weight LOWERS the loss — so raise it. This is exactly why the rule has a minus sign: you always move against the slope.' },
          { say: 'Which gives the rule. Take what you had, and step against the slope with a stride of size &eta;.', math: 'w &nbsp;&larr;&nbsp; w &nbsp;&minus;&nbsp; &eta; &middot; &part;L/&part;w' },
          { say: 'And you have met this before. Chapter 1\'s perceptron rule was <code class="inline">w &larr; w + &eta;(y &minus; &#375;)x</code>. <b>Same shape.</b>', note: 'Take what you had, add a step of size eta, in a direction worked out from how wrong you were. The perceptron rule IS gradient descent, on a cruder measure of wrongness.' },
          { say: 'Last thing, and it matters: <b>you will never compute one of these by hand.</b>', note: 'A tool called autograd does it for you, and in chapter 13 you build that tool yourself. What you need is to know what the number means when you see it — which, as of now, you do.' },
        ], {
          title: 'From "try it and see" to &part;L/&part;w',
          recap: 'nudge the weight, see which way the loss moved, divide to get the slope, then step the other way.',
        }),
      ));

      root.append(section('Three ways to fall over',
        p(`The learning rate η is your stride length, and all three failures above are stride failures. Too short and you need a million steps. Too long and you leap clean over the valley, land higher on the far slope, leap back, and the loss bounces or explodes.`),
        callout('warning', '⚠️ Where this picture lies to you',
          `A one-dimensional valley makes local minima look like the central danger of training, and for decades people assumed they were.
           In a network with a billion weights the landscape has a billion dimensions, and a point is only a local minimum if the curve bends
           <i>upward in every single one of them</i> — about as likely as a billion coin flips all coming up heads.
           Almost every flat spot is a <em>saddle point</em>: downhill in some directions, uphill in others, and gradient descent eventually slides off.
           That is one reason enormous networks train far more reliably than this 1-D intuition suggests.`),
      ));

      root.append(section('Backpropagation: one sweep, every gradient',
        p(`Gradient descent needs the slope of the loss with respect to every weight. For a weight buried deep in the network that looks hopeless: changing it changes a hidden activation, which changes the next layer, which changes the output, which changes the loss. Doing that calculation separately a million times is not a plan.`),
        p(`<em>Backpropagation</em> is the observation that you can get all of them in a single sweep backwards, reusing intermediate results, because of the chain rule: <b>the slope of a chain of functions is the product of the slopes of its links.</b>`),
        callout('tryit', '🖐 Try this: watch the two passes',
          `Green dots are the forward pass: each edge multiplies its input by its weight, and the products are summed at the next node. The loss is computed at the far right.<br>
           Red dots are the backward pass: ∂L/∂y is computed <b>once</b> and pushed left, picking up a factor at every edge and node.<br>
           <b>Notice:</b> every red number on the layer-1 edges contains that same ∂L/∂y = −0.789 as a factor. Nothing is recomputed from scratch.
           Drag left and right across the picture to freeze the step mid-flight.`),
        backpropFlow(),
        p(`That reuse is the entire trick. Each layer's gradients are built from the layer after it, so computing the gradient for a million weights costs about the same as one forward pass. Not a million times as much. About once.`),

        /* maths beat: the chain rule, named after they have watched it happen */
        p(`Those red numbers on the canvas are written in notation nobody has explained yet. Here it is, and it is one idea.`),
        ctx.decoder([
          { sym: '&part;L/&part;w<sub>1</sub>', name: 'how the loss depends on the first weight', says: 'The thing you actually want. It is buried deep — w₁ is nowhere near the loss.', points: 'the red number on a layer-1 edge.' },
          '=',
          { sym: '&part;L/&part;y', name: 'loss, given the output', says: 'How the loss responds to the final answer. Easy: it is right next to the loss.', points: 'the single red number at the far right, computed once.' },
          '×',
          { sym: '&part;y/&part;h', name: 'output, given the hidden value', says: 'How the output responds to the hidden neuron feeding it.', points: 'the factor picked up crossing one edge.' },
          '×',
          { sym: '&part;h/&part;w<sub>1</sub>', name: 'hidden value, given the weight', says: 'How that hidden neuron responds to the weight itself. Also easy: they are adjacent.', points: 'the factor picked up at the last hop.' },
        ], {
          title: '∂L/∂w₁  =  ∂L/∂y · ∂y/∂h · ∂h/∂w₁',
          hint: 'Click along the line, right to left — the same direction the red dots travel.',
          plain: 'The <em>chain rule</em>: to find how a far-away thing affects the loss, multiply the slopes of every link between them. Each individual link is easy. The chain is what makes it look hard.',
        }),
        callout('key', '🔑 Why this is cheap, in one observation',
          `Look at where <b>∂L/∂y</b> sits in that product: it is the <b>first factor in every weight's chain</b>, at every depth.<br>
           So you compute it <b>once</b>, push it leftward, and each layer multiplies it by one more local factor on the way past. Nothing is ever recomputed — which is exactly what you watched, and exactly why the whole backward sweep costs about what one forward pass costs.<br>
           <b>That is backpropagation.</b> Not a new kind of maths — the chain rule, plus the observation that you should work right to left so the shared part is only done once.`),
        p(`It is worth being blunt about the notation: <code class="inline">∂L/∂w</code> looks like the hard part and is the easy part. The genuinely clever move is the <b>ordering</b> — going backwards. Do the same multiplications left to right and you recompute the shared prefix for every single weight, and training a large model becomes impossible.`),

        callout('key', '🔑 Check one number by hand',
          `Smallest network that shows it: one input x, one hidden neuron h = σ(w<sub>1</sub>·x), one linear output y = w<sub>2</sub>·h, loss L = ½(y − t)².
           Take x = 1, t = 1, w<sub>1</sub> = 0.5, w<sub>2</sub> = −1.<br>
           Forward: h = σ(0.5) = 0.622, y = −0.622, L = ½(−1.622)² = 1.316.<br>
           Backward: ∂L/∂y = y − t = −1.622. Then ∂L/∂w<sub>2</sub> = ∂L/∂y · h = <b>−1.010</b>,
           and ∂L/∂w<sub>1</sub> = ∂L/∂y · w<sub>2</sub> · h(1−h) · x = −1.622 · −1 · 0.235 · 1 = <b>0.381</b>.<br>
           The same ∂L/∂y appears in both. That is the reuse, in four lines of arithmetic.`),
        p(`Note the signs. ∂L/∂w<sub>2</sub> is negative, so increasing w<sub>2</sub> <i>lowers</i> the loss and gradient descent will push it up. ∂L/∂w<sub>1</sub> is positive, so w<sub>1</sub> gets pushed down. Every weight is told which way to move and roughly how much, from one backward sweep.`),
        callout('history', '📜 1986: rediscovered, then a 26-year wait for hardware',
          `The method had been derived before — Seppo Linnainmaa in 1970 as a general technique, Paul Werbos in 1974 for neural networks — and sank without trace both times.
           In 1986 David Rumelhart, Geoffrey Hinton and Ronald Williams published "Learning representations by back-propagating errors" in <i>Nature</i>,
           showed it discovering useful hidden representations, and this time the field noticed. It answered Minsky and Papert directly: multi-layer networks <i>could</i> be trained.
           What it could not overcome was 1986 hardware and 1986 data. The algorithm was right and had to wait a quarter of a century for machines big enough to show it.`),
      ));

      root.append(section('Now train one for real',
        callout('tryit', '🖐 Try this: the playground',
          `<b>1.</b> Start with <b>XOR</b>, 4 hidden units, tanh. Press <b>▶ Play</b>. Watch the heatmap fold into four quadrants and the loss curve fall.<br>
           <b>2.</b> Switch to <b>Two circles</b>. Turn <b>Hidden-unit lines</b> on and count how many straight cuts it takes to fake a circle.<br>
           <b>3.</b> Try <b>Spiral</b> with 4 units. It cannot, and the loss flattens out. Push hidden units to 12–16 and let it run for a minute.<br>
           <b>4.</b> Set the learning rate to 1.0 — the loss thrashes or explodes. Set it to 0.001 and nothing visibly happens. Those are failures 2 and 3 from the fog, in a real network.`),
        playground(),
        p(`Everything in this chapter is implemented by hand underneath that figure, in a few dozen lines of plain JavaScript: a forward pass, a backward pass, a gradient step, repeated a few hundred times per animation frame. Nothing is pre-computed and nothing is faked. When the boundary wobbles, that is the gradient wobbling.`),
        ctx.code(
`// forward
for (let j = 0; j < N; j++) hid[j] = act(w1[j][0]*x + w1[j][1]*y + b1[j]);
let z = b2; for (let j = 0; j < N; j++) z += w2[j] * hid[j];
const pr = 1 / (1 + Math.exp(-z));          // predicted probability

// backward — note that dz is just (predicted − actual)
const dz = pr - label;
for (let j = 0; j < N; j++) {
  const dh = dz * w2[j] * dact(hid[j]);     // reuse dz, pick up two factors
  w2[j] -= lr * dz * hid[j];
  w1[j][0] -= lr * dh * x;
  w1[j][1] -= lr * dh * y;
  b1[j]    -= lr * dh;
}
b2 -= lr * dz;`),
        p(`Two details worth pausing on. First, <code class="inline">pr - label</code>: when a sigmoid output meets cross-entropy, every messy derivative cancels and the error signal is simply <i>predicted minus actual</i>. The same cancellation happens with softmax and cross-entropy inside every language model.`),
        p(`Second, look at what the backward half actually does. It is longer on the page, because it has to touch every weight — but there is no new forward work anywhere in it: each line is one multiply and one subtract, reusing numbers the forward pass already computed. That is why a full set of gradients costs only about twice what one prediction costs, and why training is affordable at all.`),
      ));

      root.append(section('Why this matters for modern AI',
        callout('tryit', '🖐 Try this',
          `Drag <b>hidden units</b> to 512 and <b>hidden layers</b> to 8 and watch your green bar crawl.
           Then press <b>Match the 1989 postcode reader</b> — the network that read US mail is smaller than the one you just built by dragging a slider.<br>
           Now look at how far the top bar still is, on a scale where every step is 10×.`),
        paramScale(),
        p(`Your browser toy has a few hundred weights. A frontier language model has ten to a thousand billion. It is the same kind of object: the same forward pass, the same backward pass, the same nudge. The chain rule does not care how long the chain is.`),
        p(`So when you read that a model was "trained on 15 trillion tokens", here is what physically happened. A forward pass predicted the next token. Cross-entropy measured how wrong it was. Backpropagation computed a gradient for every one of the billions of weights in one backward sweep. An optimizer nudged each one a tiny amount.`),
        p(`Then again. A few million times, across tens of thousands of GPUs, for months. There is no other mechanism and no second ingredient. The network in the playground and Claude differ in size, in architecture (chapter 7), and in the data they were shown. They do not differ in what "learning" means.`),
        callout('example', '🌍 Where plain MLPs live today',
          `Every recommendation feed you scroll ends in an MLP scoring candidates.
           The feed-forward blocks inside a transformer — which hold roughly two-thirds of a large language model's weights — are exactly the 2-layer MLP from the playground, thousands of units wide.
           AlphaGo's value head, card-fraud detectors, and most predictions on spreadsheet-shaped data are MLPs.
           The architectures in Part II are ways of <i>arranging</i> MLPs, not replacements for them.`),
        p(`One picture to carry into chapter 3: <b>the loss is a landscape, the gradient is the slope under your feet, and backpropagation is how you feel the slope in a billion directions at once.</b> What chapter 3 adds is everything that makes the walk actually work — batches, optimizers, and knowing when to stop.`),
      ));

      root.append(
        ctx.quiz([
          { q: 'In the first interactive, the output neuron scored 4/4 on XOR. What did neurons A and B actually contribute?', options: ['They each solved XOR and the output neuron voted', 'They answered two easier, linearly separable questions, and the output neuron combined those answers', 'They stored the four correct labels', 'They increased the learning rate'], answer: 1, explain: 'A fired for "at least one switch on", B for "both on". Neither is XOR. The output neuron only ever saw those two 0/1 answers, and "A and not B" is exactly XOR. That is what a hidden layer is for: inventing easier questions.' },
          { q: 'A network has six layers but no activation function between them. Why can it still not learn XOR?', options: ['Six layers is too few', 'Without a non-linearity the layers collapse into one linear map, which still draws a single flat boundary', 'The learning rate must be zero', 'XOR needs at least four inputs'], answer: 1, explain: 'W₃·(W₂·(W₁·x)) = (W₃W₂W₁)·x, a single matrix. That is why "bendiness" stayed pinned at 0.0% however many layers you stacked. Depth without a hinge adds nothing.' },
          { q: 'You set the learning rate very high and the loss starts going <i>up</i>. What is happening?', options: ['The network has found the global minimum', 'Each step overshoots the valley and lands higher on the far slope', 'Backpropagation has stopped working', 'The data has changed'], answer: 1, explain: 'Gradient descent only knows the local slope. A stride longer than the valley is wide jumps clean across it, and each jump can land higher, until the numbers blow up. That is failure 2 in the fog demo.' },
          { q: 'What is backpropagation, in one sentence?', options: ['A way to choose the learning rate', 'A special activation function', 'The chain rule applied backwards through the layers, so every weight\'s gradient is computed in one sweep', 'A method for collecting training data'], answer: 2, explain: 'Backprop computes ∂L/∂(output) once and reuses it at every edge, multiplying by the slopes of the links in between. That is why the gradient for a million weights costs about as much as one forward pass.' },
          { q: 'A classifier gives the correct class a probability of 1%. Roughly what is its cross-entropy loss, compared with giving it 90%?', options: ['About the same', 'About 0.1 versus 4.6: confident mistakes cost far more', 'Exactly ten times more', 'Zero, because it still ranked the class'], answer: 1, explain: '−log(0.9) ≈ 0.105 and −log(0.01) ≈ 4.6, the two numbers you read off the loss explorer. Cross-entropy punishes confidently wrong predictions hardest, which is what pushes a model toward being honest about its own uncertainty.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://www.youtube.com/watch?v=Ilg3gGewQ5U" target="_blank" rel="noopener">3Blue1Brown, "What is backpropagation really doing?"</a> and the follow-up on the calculus: the visual version of the worked example above.`,
            `<a href="https://www.youtube.com/watch?v=VMj-3S1tku0" target="_blank" rel="noopener">Karpathy, "The spelled-out intro to neural networks and backpropagation: building micrograd"</a>: two and a half hours that build backprop from scratch in Python. Lab 02 of this course follows it.`,
            `<a href="https://playground.tensorflow.org" target="_blank" rel="noopener">TensorFlow Playground</a>: the playground above with more layers, more input features, and regularisation.`,
            `<a href="https://www.nature.com/articles/323533a0" target="_blank" rel="noopener">Rumelhart, Hinton &amp; Williams (1986), "Learning representations by back-propagating errors"</a>: the four-page <i>Nature</i> paper that restarted the field.`,
            `<a href="http://neuralnetworksanddeeplearning.com/" target="_blank" rel="noopener">Michael Nielsen, <i>Neural Networks and Deep Learning</i></a>: a free online book whose chapter 2 is the clearest written derivation of backprop anywhere.`,
          ]),
        ),
      );
    },
  });
})();
