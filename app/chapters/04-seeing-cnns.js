/* Zero → AGI · Chapter 04 · Seeing: convolutional networks
   DESIGN RULE: the reader drags a shape one pixel and watches a fully-connected layer's input
   scramble while a filter's response merely moves. Every paragraph explains something they did.
   Interactives, in order: shift lab (fully-connected vs conv vs pooled, with change counters and
   the weight tally); paintable 16x16 convolution explorer with the multiply-add spelled out;
   2x2 max-pooling demo fed by the explorer; receptive-field grower (depth vs filter size vs
   pooling); stylised feature hierarchy (edges -> textures -> parts -> object); ViT patch
   tokenizer with the quadratic attention-cost bar. */
(function () {
  ZTA.registerChapter({
    id: '04-seeing-cnns',
    num: 4,
    part: 'II',
    title: 'Seeing: convolutional networks',
    tagline: 'Move a shape one pixel and watch an ordinary network fall apart — then meet the nine numbers that fix it, and still power how a model reads the image you upload.',
    minutes: 40,
    objectives: [
      'explain why a fully-connected layer cannot cope with a shape that moves, and what weight sharing does about it',
      'compute one cell of a feature map by hand, and read the Σ formula that describes it',
      'predict the receptive field of a stack of 3×3 layers, and say what pooling adds',
      'compute how many tokens an image becomes for a vision transformer, and what that does to attention cost',
    ],
    takeaways: [
      '<b>A filter is nine numbers.</b> Lay it on a 3×3 patch, multiply each number by the pixel under it, add up: one cell of the feature map. In symbols, feature[r, c] = Σ w[i, j] × x[r+i, c+j].',
      '<b>Weight sharing.</b> The same nine numbers are reused at every position, so the layer costs a handful of weights instead of millions and a pattern learned in one place is found everywhere.',
      '<b>Equivariant, then a little invariant.</b> Move the input and the feature map moves with it. Max pooling keeps the largest value in each 2×2 block, which shrinks the map and forgives small shifts.',
      '<b>Depth widens the view.</b> Each extra 3×3 layer adds two pixels to the receptive field, and pooling doubles the pace. Stacked layers tend to build edges into textures, parts and objects.',
      '<b>The bet outlived the operation.</b> A vision transformer cuts the image into patches and treats each as a token: 224 ÷ 16 = 14, so 196 tokens. Attention cost grows with the square of that count.',
    ],
    render(root, ctx) {
      const { h, p, section, callout, ul, ol } = ctx;
      const C = ctx.colors;
      const FONT = '13px Inter, system-ui, sans-serif';
      const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
      const f1 = (v) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(1);
      const f2 = (v) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(2);

      /* ================================================================== */
      /* Shared "lab": a paintable 16×16 grid convolved with a 3×3 kernel.    */
      /* Both the convolution explorer and the pooling demo read this state. */
      /* ================================================================== */
      const GRID = 16, K = 3, OUTW = GRID - K + 1, OUTH = GRID - K + 1; // 14×14 feature map

      function makeGrid(fill) { const g = []; for (let r = 0; r < GRID; r++) g.push(new Array(GRID).fill(fill)); return g; }
      const PRESET_GRIDS = {
        t: () => { const g = makeGrid(0); for (let c = 2; c < 14; c++) { g[2][c] = 1; g[3][c] = 1; } for (let r = 2; r < 14; r++) { g[r][7] = 1; g[r][8] = 1; } return g; },
        diag: () => { const g = makeGrid(0); for (let r = 0; r < GRID; r++) for (let c = 0; c < GRID; c++) if (Math.abs(r - c) <= 1) g[r][c] = 1; return g; },
        circle: () => { const g = makeGrid(0); for (let r = 0; r < GRID; r++) for (let c = 0; c < GRID; c++) { const d = Math.hypot(r - 7.5, c - 7.5); if (d >= 4.6 && d <= 6.2) g[r][c] = 1; } return g; },
      };
      const KERNELS = {
        vertical: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        horizontal: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        blur: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
        sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
      };
      const Lab = { grid: PRESET_GRIDS.t(), kernel: KERNELS.vertical.map((r) => r.slice()), feat: new Array(OUTW * OUTH).fill(null), done: 0 };

      function kernelNorm() { let s = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += Math.abs(Lab.kernel[i][j]); return Math.max(1e-6, s); }
      /* The pooling figure reads the finished map straight from the grid and the kernel, not from
         the explorer's step-by-step progress, so it is never fed a half-computed or emptied map. */
      function featAt(r, c) { if (r < 0 || c < 0 || r >= OUTH || c >= OUTW) return 0; return computeAt(r, c); }
      function computeAt(r, c) { let s = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += Lab.grid[r + i][c + j] * Lab.kernel[i][j]; return isFinite(s) ? s : 0; }
      function resetFeat() { Lab.feat = new Array(OUTW * OUTH).fill(null); Lab.done = 0; }
      function idxRC(i) { return { r: Math.floor(i / OUTW), c: i % OUTW }; }
      function stepFeat() { if (Lab.done >= OUTW * OUTH) resetFeat(); const { r, c } = idxRC(Lab.done); Lab.feat[Lab.done] = computeAt(r, c); Lab.done++; return { r, c }; }
      function fillFeat() { let guard = 0; while (Lab.done < OUTW * OUTH && guard++ < OUTW * OUTH + 4) stepFeat(); }
      fillFeat(); // pre-compute once so every figure has real data before anyone touches a control

      function wrapText(gc, text, x, y, maxW, lh) {
        const words = String(text).split(' '); let line = '', yy = y;
        for (const w of words) { const t = line ? line + ' ' + w : w; if (gc.measureText(t).width > maxW && line) { gc.fillText(line, x, yy); yy += lh; line = w; } else line = t; }
        if (line) gc.fillText(line, x, yy);
      }
      function heatColor(nv) { // nv already normalised to roughly [-1,1]
        const a = Math.min(1, Math.abs(nv));
        return nv >= 0 ? 'rgba(251,113,133,' + (0.12 + 0.75 * a) + ')' : 'rgba(124,156,255,' + (0.12 + 0.75 * a) + ')';
      }

      /* ------------------------------------------------------------------ */
      /* Interactive A: convolution explorer                                  */
      /* ------------------------------------------------------------------ */
      function convExplorer() {
        const W = 720, H = 400;
        const [cv, g] = ctx.canvas(W, H);
        const LP = { x: 24, y: 50, size: 240 }; LP.cell = LP.size / GRID;
        const RP = { x: 432, y: 50, size: 240 }; RP.cell = RP.size / OUTW;
        const SA = { playing: false, speed: 8, brush: 1, painting: false, cur: idxRC(Math.max(0, Lab.done - 1)) };
        const inputs = [];

        function inLP(m) { return m.x >= LP.x && m.x <= LP.x + LP.size && m.y >= LP.y && m.y <= LP.y + LP.size; }
        function paintAt(m) {
          const c = Math.floor((m.x - LP.x) / LP.cell), r = Math.floor((m.y - LP.y) / LP.cell);
          if (r >= 0 && r < GRID && c >= 0 && c < GRID) Lab.grid[r][c] = SA.brush;
        }
        cv.addEventListener('pointerdown', (ev) => {
          ev.preventDefault(); const m = cv.pos(ev);
          if (!inLP(m)) return;
          try { cv.setPointerCapture(ev.pointerId); } catch (e) {}
          SA.painting = true; resetFeat(); paintAt(m);
        });
        cv.addEventListener('pointermove', (ev) => { if (SA.painting) { const m = cv.pos(ev); if (inLP(m)) paintAt(m); } });
        const endPaint = () => { SA.painting = false; };
        cv.addEventListener('pointerup', endPaint); cv.addEventListener('pointercancel', endPaint); cv.addEventListener('mouseleave', endPaint);

        function syncInputs() { for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) inputs[i * 3 + j].value = Lab.kernel[i][j].toFixed(2); }
        function afterEdit() { resetFeat(); fillFeat(); SA.cur = idxRC(OUTW * OUTH - 1); }
        function loadPreset(name) { Lab.grid = PRESET_GRIDS[name](); afterEdit(); }
        function setKernel(name) { const kk = KERNELS[name]; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) Lab.kernel[i][j] = kk[i][j]; syncInputs(); afterEdit(); }
        function step() { SA.cur = stepFeat(); }

        function draw() {
          g.clearRect(0, 0, W, H);
          g.font = FONT; g.fillStyle = C.muted; g.textAlign = 'left';
          g.fillText('input — paint it, ' + GRID + '×' + GRID, LP.x, 40);
          g.fillText('feature map — ' + OUTW + '×' + OUTH, RP.x, 40);
          for (let r = 0; r < GRID; r++) for (let c = 0; c < GRID; c++) {
            g.fillStyle = Lab.grid[r][c] ? '#e6ebf5' : '#111827';
            g.fillRect(LP.x + c * LP.cell, LP.y + r * LP.cell, LP.cell - 1, LP.cell - 1);
          }
          g.strokeStyle = C.line; g.strokeRect(LP.x, LP.y, LP.size, LP.size);
          const norm = kernelNorm();
          for (let r = 0; r < OUTH; r++) for (let c = 0; c < OUTW; c++) {
            const raw = Lab.feat[r * OUTW + c];
            g.fillStyle = raw == null ? '#0d1320' : heatColor(raw / norm);
            g.fillRect(RP.x + c * RP.cell, RP.y + r * RP.cell, RP.cell - 0.6, RP.cell - 0.6);
          }
          g.strokeStyle = C.line; g.strokeRect(RP.x, RP.y, RP.size, RP.size);
          if (SA.cur) {
            const { r, c } = SA.cur;
            g.strokeStyle = C.warn; g.lineWidth = 2.5;
            g.strokeRect(LP.x + c * LP.cell, LP.y + r * LP.cell, LP.cell * 3, LP.cell * 3);
            g.strokeRect(RP.x + c * RP.cell, RP.y + r * RP.cell, RP.cell, RP.cell);
            const terms = [];
            for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) terms.push(f1(Lab.grid[r + i][c + j]) + '×' + f1(Lab.kernel[i][j]));
            const s = computeAt(r, c);
            g.fillStyle = C.text; g.font = MONO; g.textAlign = 'left';
            wrapText(g, terms.join(' + ') + '  =  ' + f2(s), LP.x, 314, 648, 15);
            g.fillStyle = C.warn; g.fillText('window at row ' + r + ', col ' + c + '  →  feature[' + r + ',' + c + '] = ' + f2(s), LP.x, 352);
          }
          g.fillStyle = C.muted; g.font = FONT;
          g.fillText(Lab.done + ' / ' + (OUTW * OUTH) + ' positions computed · red = filter excited, blue = filter inhibited, grey = not yet visited.', LP.x, H - 10);
        }
        let acc = 0;
        ctx.loop((dt) => { if (SA.playing) { acc += dt * SA.speed; while (acc >= 1) { acc -= 1; step(); } } draw(); });

        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
          const inp = h('input', { type: 'number', step: '0.5', value: Lab.kernel[i][j].toFixed(2), style: { width: '40px', textAlign: 'center', background: '#0f1520', color: C.text, border: '1px solid ' + C.line, borderRadius: '4px', padding: '3px 0' } });
          inp.addEventListener('input', () => { let v = parseFloat(inp.value); if (!isFinite(v)) v = 0; v = ctx.clamp(v, -9, 9); Lab.kernel[i][j] = v; afterEdit(); });
          inputs.push(inp);
        }
        const kgrid = h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(3, 42px)', gap: '3px' } }, inputs);
        const kernelWrap = h('div', { class: 'control' }, h('label', {}, 'filter (edit any cell)'), kgrid);
        const kernelSel = ctx.select({ label: 'filter preset', options: [{ value: 'vertical', label: 'Vertical edge' }, { value: 'horizontal', label: 'Horizontal edge' }, { value: 'blur', label: 'Blur' }, { value: 'sharpen', label: 'Sharpen' }], value: 'vertical', onChange: setKernel });
        const tBtn = ctx.button('Letter T', () => loadPreset('t'));
        const dBtn = ctx.button('Diagonal line', () => loadPreset('diag'));
        const cBtn = ctx.button('Circle', () => loadPreset('circle'));
        const clearBtn = ctx.button('Clear canvas', () => { Lab.grid = makeGrid(0); afterEdit(); });
        const brushBtn = ctx.button('Brush: draw', () => { SA.brush = SA.brush ? 0 : 1; brushBtn.textContent = 'Brush: ' + (SA.brush ? 'draw' : 'erase'); });
        const playBtn = ctx.button('▶ Play', () => { SA.playing = !SA.playing; playBtn.textContent = SA.playing ? '⏸ Pause' : '▶ Play'; }, 'primary');
        const stepBtn = ctx.button('Step', () => { SA.playing = false; playBtn.textContent = '▶ Play'; step(); });
        const replayBtn = ctx.button('↺ Replay from empty', () => { SA.playing = false; playBtn.textContent = '▶ Play'; resetFeat(); SA.cur = null; });
        const speedSl = ctx.slider({ label: 'positions / sec', min: 1, max: 40, step: 1, value: 8, onChange: (v) => { SA.speed = v; } });
        return ctx.figure(cv, 'Paint a shape on the left with mouse or finger. The orange square is the 3×3 filter\'s current position; its multiply-and-add is spelled out underneath. Pick a filter preset or edit the nine numbers directly.', [kernelSel, kernelWrap, tBtn, dBtn, cBtn, clearBtn, brushBtn, playBtn, stepBtn, replayBtn, speedSl]);
      }

      /* ------------------------------------------------------------------ */
      /* Interactive B: 2×2 max pooling on the feature map from A             */
      /* ------------------------------------------------------------------ */
      function poolingDemo() {
        const W = 720, H = 360;
        const [cv, g] = ctx.canvas(W, H);
        const PW = OUTW / 2, PH = OUTH / 2; // 7×7
        const LP = { x: 24, y: 50, size: 260 }; LP.cell = LP.size / OUTW;
        const RP = { x: 420, y: 50, size: 190 }; RP.cell = RP.size / PW;
        const SB = { playing: false, speed: 5, done: 0, cur: null, pooled: new Array(PW * PH).fill(null) };
        function idx2(i) { return { br: Math.floor(i / PW), bc: i % PW }; }
        function poolAt(br, bc) { const r0 = br * 2, c0 = bc * 2; return Math.max(featAt(r0, c0), featAt(r0, c0 + 1), featAt(r0 + 1, c0), featAt(r0 + 1, c0 + 1)); }
        function reset() { SB.done = 0; SB.cur = null; SB.pooled = new Array(PW * PH).fill(null); }
        function step() { if (SB.done >= PW * PH) reset(); const { br, bc } = idx2(SB.done); SB.pooled[SB.done] = poolAt(br, bc); SB.cur = { br, bc }; SB.done++; }
        function fillAll() { let guard = 0; while (SB.done < PW * PH && guard++ < PW * PH + 4) step(); }
        fillAll();

        let poolSig = null;
        function sourceSignature() { return Lab.kernel.join(',') + '|' + Lab.grid.join(''); }
        function draw() {
          /* the pooled panel is a snapshot; invalidate it when the feature map
             it was taken from has changed underneath */
          const sig = sourceSignature();
          if (sig !== poolSig) { poolSig = sig; reset(); fillAll(); }
          g.clearRect(0, 0, W, H);
          const norm = kernelNorm();
          g.font = FONT; g.fillStyle = C.muted; g.textAlign = 'left';
          g.fillText('feature map (input to pooling) — ' + OUTW + '×' + OUTH, LP.x, 40);
          g.fillText('after 2×2 max pooling — ' + PW + '×' + PH, RP.x, 40);
          for (let r = 0; r < OUTH; r++) for (let c = 0; c < OUTW; c++) { g.fillStyle = heatColor(featAt(r, c) / norm); g.fillRect(LP.x + c * LP.cell, LP.y + r * LP.cell, LP.cell - 0.6, LP.cell - 0.6); }
          g.strokeStyle = C.line; g.strokeRect(LP.x, LP.y, LP.size, LP.size);
          for (let r = 0; r < PH; r++) for (let c = 0; c < PW; c++) { const v = SB.pooled[r * PW + c]; g.fillStyle = v == null ? '#0d1320' : heatColor(v / norm); g.fillRect(RP.x + c * RP.cell, RP.y + r * RP.cell, RP.cell - 0.8, RP.cell - 0.8); }
          g.strokeStyle = C.line; g.strokeRect(RP.x, RP.y, RP.size, RP.size);
          if (SB.cur) {
            const { br, bc } = SB.cur;
            g.strokeStyle = C.warn; g.lineWidth = 2.5;
            g.strokeRect(LP.x + bc * 2 * LP.cell, LP.y + br * 2 * LP.cell, LP.cell * 2, LP.cell * 2);
            g.strokeRect(RP.x + bc * RP.cell, RP.y + br * RP.cell, RP.cell, RP.cell);
            const vals = [featAt(br * 2, bc * 2), featAt(br * 2, bc * 2 + 1), featAt(br * 2 + 1, bc * 2), featAt(br * 2 + 1, bc * 2 + 1)];
            g.fillStyle = C.text; g.font = MONO; g.textAlign = 'left';
            g.fillText('2×2 window: [' + vals.map(f2).join(', ') + ']  →  keep max = ' + f2(Math.max.apply(null, vals)), LP.x, 330);
          }
          g.fillStyle = C.muted; g.font = FONT;
          g.fillText(SB.done + ' / ' + (PW * PH) + ' pooled cells · each keeps the largest of its four values', LP.x, H - 10);
        }
        let acc = 0;
        ctx.loop((dt) => { if (SB.playing) { acc += dt * SB.speed; while (acc >= 1) { acc -= 1; step(); } } draw(); });
        const playBtn = ctx.button('▶ Play', () => { SB.playing = !SB.playing; playBtn.textContent = SB.playing ? '⏸ Pause' : '▶ Play'; }, 'primary');
        const stepBtn = ctx.button('Step', () => { SB.playing = false; playBtn.textContent = '▶ Play'; step(); });
        const replayBtn = ctx.button('↺ Replay', () => { SB.playing = false; playBtn.textContent = '▶ Play'; reset(); });
        const speedSl = ctx.slider({ label: 'blocks / sec', min: 1, max: 30, step: 1, value: 5, onChange: (v) => { SB.speed = v; } });
        return ctx.figure(cv, 'Fed directly by the feature map from the explorer above — repaint the canvas or change the filter up there and this figure updates too. Pooling shrinks the map by keeping only the largest value in each non-overlapping 2×2 block. Note <b>largest</b>, not loudest: there is no ReLU between the filter and the pool here, so the map carries negative responses and a strongly negative cell loses to a quiet positive one.', [playBtn, stepBtn, replayBtn, speedSl]);
      }

      /* ------------------------------------------------------------------ */
      /* Interactive C: stylised feature hierarchy (edges → … → object)      */
      /* ------------------------------------------------------------------ */
      function hierarchyDemo() {
        const W = 720, H = 324;
        const [cv, g] = ctx.canvas(W, H);
        const COLS = [
          { x: 110, label: 'Edges', sub: 'tiny oriented strokes' },
          { x: 290, label: 'Textures', sub: 'edges combined' },
          { x: 470, label: 'Parts', sub: 'recognisable pieces' },
          { x: 640, label: 'Object', sub: 'the whole thing' },
        ];
        const SC = { depth: 0, dir: 1, playing: true, speed: 0.5, target: 'face' };

        function stroke(cx, cy, s, angDeg) { const a = angDeg * Math.PI / 180; g.beginPath(); g.moveTo(cx - s * Math.cos(a), cy - s * Math.sin(a)); g.lineTo(cx + s * Math.cos(a), cy + s * Math.sin(a)); g.stroke(); }
        function drawEdges(cx, cy) { g.strokeStyle = C.accent; g.lineWidth = 2.5; [0, 30, 60, 90, 120, 150].forEach((a, i) => stroke(cx + (i % 3 - 1) * 30, cy + (Math.floor(i / 3) - 0.5) * 40, 11, a)); }
        function drawTextures(cx, cy) {
          g.strokeStyle = C.green; g.lineWidth = 2;
          g.beginPath(); g.moveTo(cx - 36, cy - 32); g.lineTo(cx - 8, cy - 32); g.lineTo(cx - 8, cy - 6); g.stroke();
          for (let i = 0; i < 4; i++) { g.beginPath(); g.moveTo(cx + 4 + i * 7, cy - 34); g.lineTo(cx + 4 + i * 7, cy - 8); g.stroke(); }
          g.strokeRect(cx - 36, cy + 4, 26, 26);
          g.beginPath(); g.moveTo(cx - 36, cy + 17); g.lineTo(cx - 10, cy + 17); g.moveTo(cx - 23, cy + 4); g.lineTo(cx - 23, cy + 30); g.stroke();
          g.beginPath(); for (let i = 0; i <= 18; i++) { const x = cx + 4 + i * 1.6, y = cy + 17 + Math.sin(i * 0.7) * 8; if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); } g.stroke();
        }
        function drawParts(cx, cy, target) {
          g.strokeStyle = C.warn; g.fillStyle = C.warn; g.lineWidth = 2;
          if (target === 'car') {
            g.beginPath(); g.arc(cx - 26, cy + 18, 14, 0, Math.PI * 2); g.stroke();
            for (let i = 0; i < 5; i++) { const a = i * Math.PI * 2 / 5; g.beginPath(); g.moveTo(cx - 26, cy + 18); g.lineTo(cx - 26 + Math.cos(a) * 12, cy + 18 + Math.sin(a) * 12); g.stroke(); }
            g.beginPath(); g.arc(cx + 30, cy - 18, 6, 0, Math.PI * 2); g.stroke();
            g.beginPath(); g.moveTo(cx + 2, cy - 30); g.lineTo(cx + 2, cy + 10); g.stroke();
            g.beginPath(); g.moveTo(cx - 24, cy - 28); g.lineTo(cx - 2, cy - 28); g.lineTo(cx - 8, cy - 8); g.lineTo(cx - 30, cy - 8); g.closePath(); g.stroke();
          } else if (target === 'cat') {
            g.beginPath(); g.moveTo(cx - 30, cy - 8); g.lineTo(cx - 18, cy - 38); g.lineTo(cx - 6, cy - 8); g.closePath(); g.stroke();
            g.beginPath(); g.ellipse(cx + 18, cy - 16, 8, 11, 0, 0, Math.PI * 2); g.stroke();
            g.beginPath(); g.moveTo(cx + 18, cy - 24); g.lineTo(cx + 18, cy - 8); g.stroke();
            for (let i = 0; i < 3; i++) { g.beginPath(); g.moveTo(cx - 2, cy + 8 + i * 6); g.lineTo(cx + 40, cy + 2 + i * 6); g.stroke(); }
            g.beginPath(); g.moveTo(cx - 8, cy + 16); g.lineTo(cx + 2, cy + 16); g.lineTo(cx - 3, cy + 24); g.closePath(); g.stroke();
          } else {
            g.beginPath(); g.ellipse(cx - 28, cy - 22, 13, 7, 0, 0, Math.PI * 2); g.stroke();
            g.beginPath(); g.arc(cx - 28, cy - 22, 3, 0, Math.PI * 2); g.fill();
            g.beginPath(); g.moveTo(cx + 6, cy - 32); g.quadraticCurveTo(cx + 22, cy - 40, cx + 38, cy - 32); g.stroke();
            g.beginPath(); g.moveTo(cx - 6, cy + 2); g.lineTo(cx + 2, cy + 18); g.lineTo(cx - 10, cy + 18); g.stroke();
            g.beginPath(); g.arc(cx + 18, cy + 22, 14, 0.15 * Math.PI, 0.85 * Math.PI); g.stroke();
          }
        }
        function drawObject(cx, cy, target) {
          g.strokeStyle = C.danger; g.fillStyle = C.danger; g.lineWidth = 2.4;
          if (target === 'car') {
            g.beginPath(); g.moveTo(cx - 52, cy + 20); g.lineTo(cx - 42, cy - 6); g.lineTo(cx - 16, cy - 22); g.lineTo(cx + 24, cy - 22); g.lineTo(cx + 44, cy - 6); g.lineTo(cx + 52, cy + 20); g.closePath(); g.stroke();
            g.beginPath(); g.arc(cx - 28, cy + 22, 12, 0, Math.PI * 2); g.stroke();
            g.beginPath(); g.arc(cx + 28, cy + 22, 12, 0, Math.PI * 2); g.stroke();
          } else if (target === 'cat') {
            g.beginPath(); g.arc(cx, cy + 6, 40, 0, Math.PI * 2); g.stroke();
            g.beginPath(); g.moveTo(cx - 38, cy - 16); g.lineTo(cx - 20, cy - 46); g.lineTo(cx - 6, cy - 18); g.closePath(); g.stroke();
            g.beginPath(); g.moveTo(cx + 38, cy - 16); g.lineTo(cx + 20, cy - 46); g.lineTo(cx + 6, cy - 18); g.closePath(); g.stroke();
            g.beginPath(); g.arc(cx - 14, cy - 2, 4, 0, Math.PI * 2); g.fill();
            g.beginPath(); g.arc(cx + 14, cy - 2, 4, 0, Math.PI * 2); g.fill();
          } else {
            g.beginPath(); g.arc(cx, cy, 48, 0, Math.PI * 2); g.stroke();
            g.beginPath(); g.arc(cx - 18, cy - 10, 5, 0, Math.PI * 2); g.fill();
            g.beginPath(); g.arc(cx + 18, cy - 10, 5, 0, Math.PI * 2); g.fill();
            g.beginPath(); g.arc(cx, cy + 8, 24, 0.1 * Math.PI, 0.9 * Math.PI); g.stroke();
          }
        }
        function colX(d) { const i0 = Math.max(0, Math.min(2, Math.floor(d))), i1 = i0 + 1, f = ctx.clamp(d - i0, 0, 1); return COLS[i0].x + (COLS[i1].x - COLS[i0].x) * f; }
        function draw() {
          g.clearRect(0, 0, W, H);
          g.strokeStyle = C.line; g.lineWidth = 1.5;
          for (let i = 0; i < 3; i++) {
            const x0 = COLS[i].x + 55, x1 = COLS[i + 1].x - 55, y = 150;
            g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke();
            g.beginPath(); g.moveTo(x1, y); g.lineTo(x1 - 8, y - 4); g.lineTo(x1 - 8, y + 4); g.closePath(); g.fillStyle = C.line; g.fill();
          }
          const mx = colX(SC.depth);
          g.fillStyle = 'rgba(251,191,36,0.10)'; g.fillRect(mx - 30, 56, 60, 156);
          g.strokeStyle = C.warn; g.setLineDash([4, 3]); g.lineWidth = 1.5;
          g.beginPath(); g.moveTo(mx, 56); g.lineTo(mx, 212); g.stroke(); g.setLineDash([]);
          COLS.forEach((cdef, i) => {
            const focus = Math.max(0, 1 - Math.abs(SC.depth - i));
            g.globalAlpha = 0.3 + 0.7 * focus;
            if (i === 0) drawEdges(cdef.x, 150); else if (i === 1) drawTextures(cdef.x, 150); else if (i === 2) drawParts(cdef.x, 150, SC.target); else drawObject(cdef.x, 150, SC.target);
            g.globalAlpha = 1;
            g.textAlign = 'center';
            g.fillStyle = focus > 0.4 ? C.text : C.muted; g.font = focus > 0.4 ? 'bold 13px Inter, system-ui, sans-serif' : FONT;
            g.fillText(cdef.label, cdef.x, 234);
            g.fillStyle = C.muted; g.font = FONT; g.fillText(cdef.sub, cdef.x, 252);
          });
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'center';
          wrapText(g, 'stylised illustration of what deeper layers tend to represent — not the network\'s real learned filters, which look far messier', W / 2, 284, 600, 16);
          g.textAlign = 'left';
        }
        ctx.loop((dt) => {
          if (SC.playing) { SC.depth += SC.dir * dt * SC.speed; if (SC.depth > 3) { SC.depth = 3; SC.dir = -1; } if (SC.depth < 0) { SC.depth = 0; SC.dir = 1; } depthSl.value = SC.depth; }
          draw();
        });
        const targetSel = ctx.select({ label: 'trained to recognise', options: [{ value: 'face', label: 'Face' }, { value: 'car', label: 'Car' }, { value: 'cat', label: 'Cat' }], value: 'face', onChange: (v) => { SC.target = v; } });
        const depthSl = ctx.slider({ label: 'depth', min: 0, max: 3, step: 0.02, value: 0, fmt: (v) => ['edges', 'textures', 'parts', 'object'][Math.round(v)], onChange: (v) => { SC.depth = v; SC.playing = false; playBtn.textContent = '▶ Play'; } });
        const playBtn = ctx.button('⏸ Pause', () => { SC.playing = !SC.playing; playBtn.textContent = SC.playing ? '⏸ Pause' : '▶ Play'; }, 'primary');
        const speedSl = ctx.slider({ label: 'sweep speed', min: 0.1, max: 2, step: 0.1, value: 0.5, fmt: (v) => (+v).toFixed(1) + '×', onChange: (v) => { SC.speed = v; } });
        return ctx.figure(cv, 'A cartoon, not a measurement: real networks are not this tidy, and nobody hand-designs the "eye" or "wheel" detector — it emerges from data. But the direction is real: early layers respond to simple local patterns, later layers respond to larger, more specific ones built out of the earlier layers\' outputs.', [targetSel, depthSl, playBtn, speedSl]);
      }

      /* ------------------------------------------------------------------ */
      /* Interactive D: drag the shape — what each architecture actually sees */
      /* ------------------------------------------------------------------ */
      function shiftLab() {
        const [cv, g] = ctx.canvas(720, 460);
        const N = 16;                       // 16×16 input
        const FW = N - 2;                   // 14×14 feature map (3×3 filter, stride 1)
        const PW = FW >> 1;                 // 7×7 after 2×2 max pool
        let ox = 5, oy = 5;                 // shape offset
        const HOME = { x: 5, y: 5 };
        const EDGE = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]];   // vertical-edge filter

        /* a small plus/cross shape, so an edge filter has something to find */
        function imageAt(dx, dy) {
          const im = [];
          for (let r = 0; r < N; r++) im.push(new Array(N).fill(0));
          for (let i = 0; i < 6; i++) {
            for (let t = 0; t < 2; t++) {
              const r1 = dy + 2, c1 = dx + i;
              if (r1 >= 0 && r1 + t < N && c1 >= 0 && c1 < N) im[r1 + t][c1] = 1;
              const r2 = dy + i, c2 = dx + 2;
              if (r2 >= 0 && r2 < N && c2 + t >= 0 && c2 + t < N) im[r2][c2 + t] = 1;
            }
          }
          return im;
        }
        function convolve(im) {
          const f = [];
          for (let r = 0; r < FW; r++) {
            f.push([]);
            for (let c = 0; c < FW; c++) {
              let s = 0;
              for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += im[r + i][c + j] * EDGE[i][j];
              f[r].push(s);
            }
          }
          return f;
        }
        function pool(f) {
          const o = [];
          for (let r = 0; r < PW; r++) {
            o.push([]);
            for (let c = 0; c < PW; c++) {
              o[r].push(Math.max(f[2 * r][2 * c], f[2 * r][2 * c + 1], f[2 * r + 1][2 * c], f[2 * r + 1][2 * c + 1]));
            }
          }
          return o;
        }
        /* fraction of entries that differ from the reference position */
        function changed(a, b) {
          let n = 0, tot = 0;
          for (let r = 0; r < a.length; r++) for (let c = 0; c < a[r].length; c++) { tot++; if (Math.abs(a[r][c] - b[r][c]) > 1e-9) n++; }
          return { n, tot, frac: tot ? n / tot : 0 };
        }
        /* how much of the response changed: sum|a-b| / (sum|a| + sum|b|), so 0% = identical and
           100% = nothing in common. Counting *entries* that differ would punish the pooled map
           purely for being four times smaller, which would hide the very effect pooling has. */
        function moved(a, b) {
          let d = 0, s = 0;
          for (let r = 0; r < a.length; r++) for (let c = 0; c < a[r].length; c++) {
            d += Math.abs(a[r][c] - b[r][c]); s += Math.abs(a[r][c]) + Math.abs(b[r][c]);
          }
          return { frac: s > 1e-9 ? d / s : 0 };
        }

        let drag = false;
        const CELL = 11, IX = 30, IY = 70;
        cv.addEventListener('pointerdown', (e) => { e.preventDefault(); const p = cv.pos(e); if (p.x > IX - 10 && p.x < IX + N * CELL + 10 && p.y > IY - 10 && p.y < IY + N * CELL + 10) drag = true; });
        cv.addEventListener('pointermove', (e) => {
          if (!drag) return;
          const p = cv.pos(e);
          ox = ctx.clamp(Math.round((p.x - IX) / CELL) - 3, 0, N - 6);
          oy = ctx.clamp(Math.round((p.y - IY) / CELL) - 3, 0, N - 6);
        });
        ['pointerup', 'pointercancel', 'pointerleave'].forEach(t => cv.addEventListener(t, () => { drag = false; }));

        const xSl = ctx.slider({ label: 'shift right', min: 0, max: N - 6, step: 1, value: 5, onChange: (v) => { ox = v; } });
        const ySl = ctx.slider({ label: 'shift down', min: 0, max: N - 6, step: 1, value: 5, onChange: (v) => { oy = v; } });
        const nudge = ctx.button('Nudge 1 pixel right', () => { ox = ctx.clamp(ox + 1, 0, N - 6); xSl.value = ox; }, 'primary');
        const home = ctx.button('Back to start', () => { ox = HOME.x; oy = HOME.y; xSl.value = ox; ySl.value = oy; });
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          const ref = imageAt(HOME.x, HOME.y), now = imageAt(ox, oy);
          const fRef = convolve(ref), fNow = convolve(now);
          const pRef = pool(fRef), pNow = pool(fNow);
          /* one metric across the row: the count and the normalised distance are not
             comparable, and printing 12/256 next to 39% told the reader the feature map
             had changed eight times more than the picture did */
          const cIn = changed(ref, now), mIn = moved(ref, now), cF = moved(fRef, fNow), cP = moved(pRef, pNow);

          /* ---- the image ---- */
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('drag the shape', IX, 40);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('16 × 16 pixels = 256 numbers', IX, 56);
          for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
            const on = now[r][c], was = ref[r][c];
            g.fillStyle = on ? C.text : '#131a27';
            g.fillRect(IX + c * CELL, IY + r * CELL, CELL - 1, CELL - 1);
            if (on !== was) {                       // mark every pixel the move altered
              g.strokeStyle = C.danger; g.lineWidth = 1;
              g.strokeRect(IX + c * CELL + 0.5, IY + r * CELL + 0.5, CELL - 2, CELL - 2);
            }
          }
          g.font = 'bold ' + FONT; g.fillStyle = cIn.frac > 0 ? C.danger : C.muted;
          g.fillText(cIn.n + ' of ' + cIn.tot + ' pixels changed', IX, IY + N * CELL + 22);
          g.font = MONO; g.fillStyle = C.muted;
          wrapText(g, 'red outline = a pixel whose value is different from the starting position', IX, IY + N * CELL + 42, 205, 15);

          /* ---- a fully-connected layer's view: one flat row ---- */
          const SX = 250, SY = 66;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('what a fully-connected layer sees', SX, 40);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('the same 256 numbers, flattened into one row', SX, 56);
          const SW = 440, cw = SW / N;
          for (let r = 0; r < N; r++) {
            for (let c = 0; c < N; c++) {
              const on = now[r][c], was = ref[r][c];
              g.fillStyle = on !== was ? C.danger : (on ? C.text : '#131a27');
              g.fillRect(SX + c * cw, SY + r * 5, cw - 0.8, 4);
            }
          }
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, 'Every row is 16 pixels of the image laid end to end. To this layer there is no "next to" — row 3 and row 4 are simply far-apart entries in a list, and it must learn the shape again for every position it could occupy.', SX, SY + N * 5 + 18, 440, 16);

          /* ---- feature map and pooled map: one column each, titles wrapped to width ---- */
          const MTITLE = 234, MSUB = 250, MY = 290;      // shared baselines for both panels
          const panel = (mat, X, cell, title, sub, subW, ch) => {
            g.font = 'bold ' + FONT; g.fillStyle = C.text;
            g.fillText(title, X, MTITLE);
            g.font = MONO; g.fillStyle = C.muted;
            wrapText(g, sub, X, MSUB, subW, 14);
            const peak = Math.max(1e-6, ...mat.flat().map(Math.abs));
            for (let r = 0; r < mat.length; r++) for (let c = 0; c < mat[r].length; c++) {
              g.fillStyle = heatColor(mat[r][c] / peak);
              g.fillRect(X + c * cell, MY + r * cell, cell - 1, cell - 1);
            }
            g.font = 'bold ' + FONT; g.fillStyle = ch.frac > 0.4 ? C.warn : C.green;
            g.fillText((ch.frac * 100).toFixed(0) + '% of this response changed', X, MY + mat.length * cell + 18);
          };
          panel(fNow, SX, 8, 'after one 3×3 filter', '14 × 14 feature map — 9 weights, reused everywhere', 170, cF);
          panel(pNow, SX + 200, 8, 'after 2×2 max pooling', '7 × 7 — keeps the strongest response nearby', 240, cP);

          /* ---- the weight tally: its own column under the image, clear of both maps ---- */
          const TX = IX;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('weights in the first layer', TX, 356);
          g.font = MONO;
          g.fillStyle = C.danger;
          g.fillText('fully connected: 150,528,000', TX, 376);
          g.fillStyle = C.muted;
          g.fillText('(a 224×224 colour photo into', TX, 392);
          g.fillText('1,000 hidden units)', TX, 408);
          g.fillStyle = C.green;
          g.fillText('one 3×3 filter: 27 + 1 bias', TX, 428);
          g.fillStyle = C.muted;
          g.fillText('(9 numbers per colour channel)', TX, 444);
          g.fillText('That is the whole layer.', TX, 460);

          const pct = (m) => (m.frac * 100).toFixed(0) + '%';
          ro.set({ shift: '(' + ox + ', ' + oy + ')', 'pixels differing': cIn.n + '/' + cIn.tot,
            'picture changed': pct(mIn), 'feature map changed': pct(cF), 'pooled changed': pct(cP) });
        });

        return ctx.figure(cv, 'The same shape, moved. Red marks every value that is different from where it started. A fully-connected layer has no notion that two pixels are neighbours — it sees a list of 256 unrelated numbers, so a one-pixel nudge rewrites twelve of those numbers and it must learn the shape afresh at every position. The filter has nine weights in total, reused at all 196 positions, and its response simply <i>moves with the shape</i>. Pooling then throws away some of that movement, which is where genuine position-blindness starts.', [xSl, ySl, nudge, home], ro);
      }

      /* ------------------------------------------------------------------ */
      /* Interactive E: how far back can one output unit see?                */
      /* ------------------------------------------------------------------ */
      function receptiveField() {
        const [cv, g] = ctx.canvas(720, 330);
        let layers = 1, ksize = 3, pools = 0;
        /* receptive field of one unit after `layers` k×k convs with `pools` 2×2 pools interleaved */
        function rf() {
          let r = 1, jump = 1;
          for (let i = 0; i < layers; i++) {
            r += (ksize - 1) * jump;              // the conv itself
            if (i < pools) { r += jump; jump *= 2; }  // then a 2x2 pool: widens by 1 step, halves the map
          }
          return r;
        }
        function paramsPerFilter() { return ksize * ksize; }
        const lSl = ctx.slider({ label: 'stacked conv layers', min: 1, max: 8, step: 1, value: 1, onChange: (v) => { layers = v; if (pools > v - 1) { pools = Math.max(0, v - 1); pSl.value = pools; } } });
        const kSl = ctx.slider({ label: 'filter size', min: 3, max: 7, step: 2, value: 3, onChange: (v) => { ksize = v; } });
        /* a pool needs a conv layer above it, so the slider snaps back to what the stack can hold */
        const pSl = ctx.slider({ label: '2×2 pools in between', min: 0, max: 7, step: 1, value: 0, onChange: (v) => { pools = Math.min(v, layers - 1); if (pools !== v) pSl.value = pools; } });
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          const R = rf();
          const IMG = 64, CELL = 4, X = 40, Y = 60;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('what one unit deep in the stack can actually see', X, 32);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('a 64 × 64 input image', X, 48);
          g.fillStyle = '#131a27';
          g.fillRect(X, Y, IMG * CELL, IMG * CELL);
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(X, Y, IMG * CELL, IMG * CELL);
          const side = Math.min(IMG, R) * CELL;
          const cx = X + IMG * CELL / 2, cy = Y + IMG * CELL / 2;
          g.fillStyle = 'rgba(56,217,169,0.30)';
          g.fillRect(cx - side / 2, cy - side / 2, side, side);
          g.strokeStyle = C.green; g.lineWidth = 2;
          g.strokeRect(cx - side / 2, cy - side / 2, side, side);
          g.fillStyle = C.text;
          g.fillRect(cx - CELL / 2, cy - CELL / 2, CELL, CELL);

          const TX = 340;
          g.font = 'bold 17px Inter, system-ui, sans-serif'; g.fillStyle = C.green;
          g.fillText('receptive field: ' + R + ' × ' + R + ' pixels', TX, 70);
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, 'The white dot is one single unit. The green square is every pixel of the original image that can influence it — its receptive field.', TX, 94, 330, 17);
          g.font = MONO; g.fillStyle = C.muted;
          const stackW = paramsPerFilter() * layers;         // one filter per layer, one channel
          g.fillText('weights: ' + paramsPerFilter() + ' per filter × ' + layers + (layers > 1 ? ' layers' : ' layer') + ' = ' + stackW, TX, 150);
          const cover = Math.min(IMG, R);                    // the image is only 64 x 64 - never claim more
          g.fillText(R >= IMG ? 'covering all ' + (IMG * IMG).toLocaleString() + ' pixels of the image'
            : 'covering ' + (cover * cover).toLocaleString() + ' pixels of the image', TX, 168);
          const ratio = (cover * cover) / stackW;
          g.font = 'bold ' + FONT; g.fillStyle = C.accent;
          g.fillText(ratio.toFixed(1) + '× as many pixels reached as weights spent', TX, 190);
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, R >= IMG
            ? 'This unit now sees the entire image, and it got there with only a handful of weights per layer. That is why depth, not filter size, is how vision networks grow their view.'
            : 'Add layers and watch the square grow. Adding a pool doubles how fast it grows, because everything above the pool is working on a half-size map. (Add a layer first: a pool needs one above it.)',
            TX, 214, 330, 17);
          ro.set({ layers, filter: ksize + '×' + ksize, pools, 'receptive field': R + '×' + R });
        });

        return ctx.figure(cv, 'One 3×3 filter sees a 3×3 patch. Stack a second on top and each of its units reads a 3×3 patch of <i>the first layer\'s outputs</i>, which between them covered 5×5 of the original image — so the view grows without the filters ever getting bigger. Pooling accelerates it: everything above a 2×2 pool is working on a half-size map, so each further step covers twice as much ground. The weight count shown is for one filter per layer on a single channel; real layers hold many filters, but the comparison between settings is the same. This is a large part of why vision networks are deep.', [lSl, kSl, pSl], ro);
      }

      /* ------------------------------------------------------------------ */
      /* Interactive F: how a modern multimodal model chops up your image    */
      /* ------------------------------------------------------------------ */
      function patchTokens() {
        /* real browser text runs wider than the headless estimate, so both prose
           blocks are given a full extra line of room and a wider column */
        const [cv, g] = ctx.canvas(720, 424);
        let side = 224, patch = 16;
        const SIZES = [112, 224, 336, 448, 672];
        const sSl = ctx.slider({ label: 'image size (pixels)', min: 0, max: 4, step: 1, value: 1, fmt: (v) => SIZES[v] + '²', onChange: (v) => { side = SIZES[v]; } });
        const pSl = ctx.slider({ label: 'patch size', min: 8, max: 56, step: 8, value: 16, onChange: (v) => { patch = v; } });
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          const per = Math.floor(side / patch);
          const tokens = per * per;
          const BOX = 280, X = 40, Y = 56;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('your image, cut into tokens', X, 32);
          /* a stand-in picture so the grid has something to cut up */
          for (let i = 0; i < BOX; i += 2) {
            for (let j = 0; j < BOX; j += 2) {
              const u = i / BOX, v = j / BOX;
              const d = Math.hypot(u - 0.45, v - 0.5);
              const val = d < 0.26 ? 0.75 - d : 0.18 + 0.12 * Math.sin(u * 14) * Math.cos(v * 11);
              g.fillStyle = 'rgba(124,156,255,' + ctx.clamp(val, 0.03, 0.9) + ')';
              g.fillRect(X + j, Y + i, 2, 2);
            }
          }
          const step = BOX / per;
          g.strokeStyle = 'rgba(230,235,245,0.45)'; g.lineWidth = per > 40 ? 0.3 : 1;
          for (let i = 0; i <= per; i++) {
            g.beginPath(); g.moveTo(X + i * step, Y); g.lineTo(X + i * step, Y + BOX); g.stroke();
            g.beginPath(); g.moveTo(X, Y + i * step); g.lineTo(X + BOX, Y + i * step); g.stroke();
          }
          g.strokeStyle = C.warn; g.lineWidth = 2;
          g.strokeRect(X, Y, step, step);
          g.font = MONO;
          const tagW = g.measureText('one token').width;
          g.fillStyle = 'rgba(10,14,22,0.8)';                       // so the label stays readable over the picture
          g.fillRect(X + step + 3, Y + step - 13, tagW + 6, 15);
          g.fillStyle = C.warn;
          g.fillText('one token', X + step + 6, Y + step - 2);

          const TX = 350, TW = 352;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('what the model is handed', TX, 32);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText(side + ' ÷ ' + patch + ' → ' + per + ' patches per side' + (per * patch === side ? '' : ' (' + (per * patch) + ' of ' + side + ' px used)'), TX, 58);
          const eq = per + ' × ' + per + ' = ';
          g.fillText(eq, TX, 80);
          const eqW = g.measureText(eq).width;                      // keep the big number clear of the sum
          g.font = 'bold 26px Inter, system-ui, sans-serif'; g.fillStyle = C.green;
          g.fillText(tokens.toLocaleString() + ' tokens', TX + eqW + 6, 82);
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, 'Each patch is flattened and pushed through one small layer into a vector — exactly the kind of vector a word becomes in chapter 6. From there one shared stack of attention layers handles both, with no image-specific machinery left downstream — a picture token and a word token are the same kind of object to it.', TX, 110, TW, 17);

          /* cost bar: attention is quadratic in the number of tokens */
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('attention cost grows with tokens²', TX, 226);
          const rel = tokens * tokens / (196 * 196);
          const w = ctx.clamp(Math.log10(Math.max(1, rel)) / 3, 0.02, 1) * TW;
          g.fillStyle = C.line; g.fillRect(TX, 236, TW, 14);
          g.fillStyle = rel > 20 ? C.danger : rel > 4 ? C.warn : C.green;
          g.fillRect(TX, 236, w, 14);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText(rel < 1 ? (1 / rel).toFixed(1) + '× cheaper than 224²/16' : rel.toFixed(1) + '× the cost of 224²/16', TX, 266);
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, 'Halve the patch size and you get four times the tokens and sixteen times the attention cost. Attention over those tokens grows with the SQUARE of their number, so resolution is not free inside the model — even though what you are billed grows with the pixel count itself. That is why models tile large images rather than shrinking the patch.', TX, 290, TW, 17);
          ro.set({ image: side + '²', patch: patch + '²', tokens });
        });

        return ctx.figure(cv, 'A Vision Transformer does not slide a filter at all. It cuts the image into a grid of fixed squares, turns each square into one vector, and hands the whole sequence to the same architecture that processes words. The classic setting — a 224×224 image in 16×16 patches — gives 196 tokens. The grid is the entire "architecture" here; everything else is learned by attention deciding which patches should look at which.', [sSl, pSl], ro);
      }

      /* ================================================================== */
      /* The chapter: touch first, read second.                             */
      /* ================================================================== */
      root.append(
        callout('tryit', '🖐 Do this first — move a shape one pixel and watch a network break',
          `<b>1.</b> Drag the white shape around the grid, or press <b>Nudge 1 pixel right</b>.<br>
           <b>2.</b> Read the counter under the image: a single pixel of movement rewrites <b>a dozen</b> of the 256 input values. Drag it right across the grid and it is dozens.<br>
           <b>3.</b> Look at the flattened row on the right — that is all a fully-connected network ever receives. Nothing in it says these pixels are neighbours.<br>
           <b>4.</b> Now look at the two heatmaps at the bottom. The shape moved; the pattern in them <i>moved with it</i> instead of scrambling.`),
        shiftLab(),
        p(`That is the problem and the fix in one picture. Nudge the shape one pixel and six of the twenty pixels that carried it land on different inputs of the fully-connected layer; drag it across the grid and eighteen of twenty do, so it would have to learn "cross" separately for every position it could occupy. The filter underneath just found the same thing in a new place, using the same nine numbers.`),
      );

      root.append(section('Why a plain network cannot see',
        p(`Show a child a photo of a golden retriever peeking out from behind a couch — one eye, half an ear, a stripe of fur — and they say "dog" before you finish the sentence. Chapter 2 gave you a network that can approximate any pattern you can define, so why not point it at a photograph?`),
        p(`You can. It is also a quiet disaster, and you just watched the first half of why.`),
        p(`Here is the second half. A modest photograph — 224 pixels wide, 224 tall, three colour channels — is <b>150,528</b> numbers. Flatten it and wire it to a modest 1,000 hidden units, and the first layer alone needs over <b>150 million weights</b>, before the network has looked at a single image.`),
        callout('key', '🔑 Three separate problems, worth seeing apart',
          `<b>1. Too many weights.</b> One fully-connected layer over one photo costs more parameters than most entire models.<br>
           <b>2. No notion of "nearby".</b> Flattening throws away the fact that two pixels touch. The network must rediscover geometry from scratch, from data.<br>
           <b>3. No notion of "the same thing, moved".</b> A pattern learned in the top-left teaches the network nothing about the same pattern in the bottom-right.<br>
           Convolution fixes all three with one move.`),
        p(`And it was needed badly enough that a version of it was already reading handwritten cheques for American banks by 1998 — fourteen years before deep learning became a headline.`),
      ));

      root.append(section('Convolution: a small filter that slides',
        p(`Stop connecting every input to every output. Instead, take one small grid of numbers — say 3×3, called a <em>filter</em> or <em>kernel</em> — lay it over a 3×3 patch of the image, multiply each filter number by the pixel underneath, and add the nine products. That single sum is one value of the output. Then slide one pixel over and do it again.`),
        p(`Do that at every position and you get a grid of sums called a <em>feature map</em>. It is not a picture any more. It is a map of "how strongly did this exact pattern show up, here?"`),
        callout('tryit', '🖐 Try this: feel the filter slide',
          `Start with the <b>Letter T</b> preset and the <b>Vertical edge</b> filter.<br>
           <b>1.</b> Press <b>Step</b> a few times and read the arithmetic under the grids — nine multiplications and an addition, nothing more.<br>
           <b>2.</b> Press <b>▶ Play</b>. The feature map lights up along the T's vertical stem and stays almost dark along the horizontal bar.<br>
           <b>3.</b> Switch to <b>Horizontal edge</b>. The opposite happens, from the same image.<br>
           <b>4.</b> Edit one of the nine numbers by hand. Every future feature map changes with it — <b>the filter is nothing but those nine numbers.</b><br>
           <b>5.</b> Paint your own shape, then press <b>▶ Play</b> again to run the filter over it.`),
        convExplorer(),

        /* maths beat: the nine products, then the sigma that abbreviates them */
        ctx.walkthrough([
          { say: 'Load <b>Letter T</b> with the <b>Vertical edge</b> filter and press <b>Step</b> once: the orange square sits in the top-left corner, covering nine pixels.', note: 'The yellow line under the grids reads: window at row 0, col 0 → feature[0,0] = 1.00.' },
          { say: 'Eight of those nine pixels are black, which is 0, so eight of the nine products are 0 whatever the filter says.', math: '0×−1 + 0×0 + 0×1 + 0×−1 + 0×0 + 0×1 + 0×−1 + 0×0 + …' },
          { say: 'The ninth pixel, bottom-right, is white (1) and the filter number lying on it is 1, so the whole sum is 1.', math: '… + 1×1 &nbsp;=&nbsp; 1' },
          { ask: 'Two steps on, at col 2, the bottom row of the window is three white pixels: 1, 1, 1. The bottom row of the filter is −1, 0, 1. What does that row add up to?',
            options: ['3', '0', '−1'], answer: 1,
            explain: '1×−1 + 1×0 + 1×1 = 0. Left and right cancel. The filter only answers when its left side and its right side see different things, which is what a vertical edge is.' },
          { say: 'To write that sum without listing nine products, give every cell an address: row <b>i</b> and column <b>j</b>, each counted 0, 1, 2 from the top-left, the way the demo counts.', note: 'So the bottom-right cell is i = 2, j = 2.' },
          { say: 'Call the pixel at that address <b>x[i, j]</b> and the filter number lying on top of it <b>w[i, j]</b>, so one product is w[i, j] × x[i, j].', math: 'w[2,2] × x[2,2] &nbsp;=&nbsp; 1 × 1' },
          { say: 'Adding up all of them has a symbol, <b>Σ</b>, a capital Greek S for "sum".', math: 'Σ &nbsp;w[i, j] × x[i, j]', note: 'Read it as: for every i and every j, multiply, then add the lot.' },
          { ask: 'i runs over 0, 1, 2 and so does j. How many products does that Σ add up?',
            options: ['3', '6', '9'], answer: 2,
            explain: 'Three rows times three columns: the nine cells of the filter. For a 7×7 filter the same Σ would add up 49.' },
          { say: 'Last piece: the window slides, so name its top-left corner by row <b>r</b> and column <b>c</b>, and the pixel under filter cell [i, j] becomes x[r+i, c+j].', note: 'r+i only means: start at row r, then go i rows further down.' },
          { say: 'That gives the whole feature map in one line, and it is the line the demo prints with the numbers filled in.', math: 'feature[r, c] &nbsp;=&nbsp; Σ &nbsp;w[i, j] × x[r+i, c+j]' },
          { ask: 'The window moves from feature[0, 0] to feature[6, 5]. Which part of that line changes?',
            options: ['The nine w values', 'The pixels x that the w values land on', 'Both'], answer: 1,
            explain: 'r and c appear only inside x. The w has no r or c in it at all, and that absence is the important part: nine numbers, the same at every position.' },
        ], {
          title: 'From nine multiplications to Σ',
          recap: 'lay the filter on the image, multiply each filter number by the pixel under it, add up the nine results, slide, repeat.',
        }),
        ctx.decoder([
          { sym: 'feature[r, c]', name: 'the feature map at row r, column c', says: 'One number of the output: how strongly the pattern showed up with the window\'s corner at row r, column c.', points: 'the orange-outlined cell in the right-hand grid.' },
          '=',
          { sym: 'Σ', name: 'the sum, over every i and j', says: 'Add up everything to its right, once for each of the nine filter cells.', points: 'the + signs in the long line under the grids.' },
          { sym: 'w[i, j]', name: 'the filter number in row i, column j', says: 'One of the nine numbers you edited. There is no r or c in it: it is the same wherever the window is.', points: 'the 3×3 block of number boxes.' },
          '×',
          { sym: 'x[r+i, c+j]', name: 'the pixel under that filter cell', says: 'Start at the window\'s corner (r, c), then step i rows down and j columns across.', points: 'one painted square inside the orange window.' },
        ], {
          title: 'feature[r, c]  =  Σ w[i, j] × x[r+i, c+j]',
          plain: 'to get one cell of the feature map, lay the filter on the image at that spot, multiply each filter number by the pixel under it, and add up the nine results. A real layer then adds one more learned number, the bias.',
        }),
        p(`A <em>stride</em> is how many pixels the filter moves each step — one, in the demo. <em>Padding</em> is whether you add a border of zeros so the output stays the same size, or let it shrink at the edges the way the 16×16 input became a 14×14 map.`),
        p(`Different filters find different things. A horizontal-edge kernel is the same idea rotated 90°. A blur kernel averages a patch instead of contrasting it. A sharpen kernel exaggerates a pixel against its neighbours. In a real network <b>nobody chooses these numbers</b> — backpropagation learns them, exactly as it learns any other weight.`),
        callout('key', '🔑 Weight sharing is the whole trick',
          `That filter has <b>nine numbers</b> plus a bias. Not nine per position — nine <i>total</i>, reused at all 196 positions.<br>
           This buys two things at once. <b>Far fewer parameters</b>: a convolutional layer with dozens of filters costs thousands of weights where a fully connected layer over the same image costs millions.<br>
           And <b>translation equivariance</b>: because the identical filter is laid over every patch, a pattern in the top-left produces the same response, merely relocated, as that pattern in the bottom-right.`),
        p(`Note the word: <em>equivariant</em>, not invariant. The feature map still moves when the input moves — you watched it move. It just moves in lock-step instead of demanding separate weights for every position. Not caring <i>where</i> a pattern was at all takes one more ingredient.`),
      ));

      root.append(section('Pooling: throwing away detail on purpose',
        p(`After a convolution produces a feature map, <em>pooling</em> shrinks it. The common form, <em>max pooling</em>, slides a 2×2 window across the map and keeps only the largest value in each window, discarding the other three. A 14×14 map becomes 7×7: a quarter of the numbers, and a quarter of the work for every layer downstream.`),
        callout('tryit', '🖐 Try this: watch resolution disappear on purpose',
          `<b>1.</b> In the explorer above, load the <b>Circle</b> preset with the <b>Sharpen</b> filter.<br>
           <b>2.</b> Come down here and press <b>Step</b> a few times. Read the four numbers in each 2×2 window — only the largest survives.<br>
           <b>3.</b> Go back up and paint over the circle. Both panels here follow every stroke.`),
        poolingDemo(),
        p(`Shrinking is only half the point. Because max pooling asks only "was this pattern present <i>somewhere</i> in this little window", it buys a small amount of genuine translation <em>invariance</em>. Shift the input one pixel and the strongest response in a 2×2 window is often still the strongest, unmoved in the pooled output.`),
        p(`That is the effect you can measure back in the first demo: move the <b>shift down</b> slider by one notch and the pooled map changes noticeably less than the feature map does. (Nudging sideways barely separates them — which is itself worth knowing: pooling buys a little tolerance, not a lot.) Stack a few rounds of convolve-then-pool and the network cares less and less about the exact pixel and more about whether something happened nearby.`),
      ));

      root.append(section('Depth is how a network widens its view',
        p(`One convolutional layer only ever sees a 3×3 window. Stack a second on top of the first layer's feature maps and each of its units — still using a 3×3 filter — is reading a 3×3 patch of <i>already-combined</i> edge responses, which corresponds to a larger patch of the original image. Its <em>receptive field</em> has grown.`),
        callout('tryit', '🖐 Try this: scrub the depth slider',
          `<b>1.</b> Drag <b>stacked conv layers</b> from 1 to 8 and watch the green square grow, while <b>weights per filter stays at 9</b>.<br>
           <b>2.</b> Now add <b>2×2 pools in between</b>. The square grows far faster, because every layer above a pool is working on a half-size map.<br>
           <b>3.</b> Take the pools back to 0 and <b>stacked conv layers</b> back to 1, then push <b>filter size</b> to 7. One 7×7 layer reaches 7×7 pixels for 49 weights.<br>
           <b>4.</b> Now set <b>filter size</b> to 3 and <b>stacked conv layers</b> to 3. The same 7×7 reach, for 27 weights. That trade is why the classic deep stacks — VGG, ResNet and their descendants — are built out of small filters rather than big ones. (Conv design has partly reversed it since 2022, once a cheaper kind of large filter caught on; chapter 7's transformers are the bigger reason the question moved on.)`),
        receptiveField(),
        p(`Layer by layer, the same trick — small filters, weight-shared, pooled down, stacked again — tends to build a hierarchy. Early layers respond to edges and simple contrasts. A little deeper, edges combine into textures and motifs: corners, stripes, curves. Deeper still, textures combine into parts — an eye, a wheel, an ear. Near the output, parts combine into whole objects.`),
        p(`Nobody designs that hierarchy. It falls out of training a deep stack of convolutions and poolings on enough labelled images, using exactly the backpropagation from chapter 2.`),
        callout('tryit', '🖐 Try this',
          `Slide <b>depth</b> from 0 to 3 — dragging it pauses the sweep — and watch the picture move from strokes, to textures, to parts, to a finished object.<br>
           Switch the target between <b>Face</b>, <b>Car</b> and <b>Cat</b>. The first two stages do not change at all — low-level structure is shared across almost everything you could photograph — while "parts" and "object" change completely.`),
        hierarchyDemo(),
        callout('history', '📜 LeNet-5 (1998) to AlexNet (2012): fourteen years in the wilderness',
          `Yann LeCun built the architecture in this chapter — convolution, pooling, layers stacked, trained end-to-end with backpropagation — as <em>LeNet-5</em> in 1998, and banks used it to read the handwritten numbers on cheques. It worked, and for over a decade it stayed mostly a curiosity: labelled datasets were small and GPUs for general computation barely existed.<br>
           That changed in 2012, when Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton entered <em>AlexNet</em> — a deeper descendant of LeNet-5, trained on two consumer GPUs with <em>ReLU</em> activations and <em>dropout</em> — into the <em>ImageNet challenge</em> (ILSVRC), a contest built on a 1,000-category slice of the ImageNet dataset, with 1.2 million training photos. Its top-5 error was about 15.3%; the best non-neural approach that year scored around 26.2%. That eleven-point gap is usually cited as the moment deep learning stopped being niche.<br>
           In 2015 Kaiming He and colleagues added <em>ResNet</em>'s skip connections, letting a layer's input jump straight to a later layer unchanged. That one trick fixed a problem where stacks past about 20 layers actually got <i>worse</i>, and let networks pass 100 layers and keep improving.`),
        callout('example', '🌍 Already in your pocket',
          `<b>Face unlock</b> turns your face into a compact numerical fingerprint and checks it against one stored on the device.
           <b>Medical imaging</b> tools flag likely tumours in X-rays, CT and MRI by the same edges-to-parts-to-whole pipeline, often matching specialists on narrow, well-defined tasks.
           <b>Self-driving perception</b> — where is the pedestrian, the lane line, the stop sign, right now — grew up on convolutional backbones, and many stacks still use one to extract features before a transformer reasons over them.
           <b>Photo search</b> works because a network already turned every photo into a description of its contents before you typed anything.`),
      ));

      root.append(section('Why this matters for modern AI',
        p(`Convolution's real contribution was never the 3×3 mechanics. It was the proof that <b>building in a good assumption about the data's structure</b> — nearby pixels relate to each other; a pattern means the same thing wherever it appears — beats forcing a generic network to rediscover that assumption from scratch.`),
        p(`That lesson outlived the operation itself. In 2020 the <em>Vision Transformer</em> showed you could skip the sliding filter entirely: cut the image into a grid of patches, treat each patch as a single token — the same kind of object chapter 6 builds for words — and hand the sequence to a Transformer.`),
        callout('tryit', '🖐 Try this — this is literally how a model reads an image you upload',
          `<b>1.</b> Leave it at <b>224²</b> with <b>16</b>-pixel patches: 196 tokens. That is the classic setting.<br>
           <b>2.</b> Drag <b>patch size</b> down to 8. Four times the tokens — and look at the attention-cost bar.<br>
           <b>3.</b> Push <b>image size</b> to 672². Now you know why uploading a high-resolution screenshot costs so much more than a thumbnail, and why models tile big images rather than shrinking the patch.`),
        patchTokens(),
        p(`This is how a modern multimodal model "sees" an image you upload. It is chopped into patches, each patch becomes the same kind of vector a word becomes, and the resulting mix of image-tokens and text-tokens is processed together by one shared architecture. One simplification to flag: most production models first run the patches through a vision network of their own and hand its output vectors to the language model. The grid and the token count work as shown; the step that turns a patch into a vector is deeper than one layer.`),
        p(`The tool changed — attention instead of a sliding kernel — but the bet did not. Images have local structure, which composes into parts, which composes into wholes. Convolution hard-coded that bet into the architecture. Transformers let the model learn it from data, which takes far more data and pays off at scale.`),
        p(`One picture to keep: <b>a convolution is a small pattern-detector that refuses to care where it is looking, and depth is how a network turns edges into objects.</b>`),
      ));

      root.append(
        ctx.quiz([
          { q: 'A fully-connected network has learned to spot a cross in the top-left corner of its 16×16 input. You show it the same cross in the bottom-right corner. What does it already know about this one?',
            options: ['Everything, because the pixel values are identical', 'Most of it, because neighbouring inputs share similar weights', 'About half, because the rows match even though the columns do not', 'Nothing, because those pixels feed weights that never saw a cross'], answer: 3,
            explain: 'Every input position has its own separate weights, and nothing ties the top-left set to the bottom-right set. That is what the flattened row in the first demo was showing: to this layer there is no "same thing, moved".' },
          { q: 'A convolutional layer has 32 filters, each 3×3, reading a one-channel image. Ignore biases. The image grows from 16×16 to 1,024×1,024. How many weights does the layer have now?',
            options: ['288, the same as before', '288 × 4,096, growing with the pixel count', 'About 30 million', '32 × 1,024'], answer: 0,
            explain: '32 filters × 9 numbers = 288, and the image size never enters the sum. In the formula, w[i, j] has no r or c in it. A bigger image means more positions to slide over, so more arithmetic, but not one extra weight.' },
          { q: 'You shift the input one pixel. The feature map shifts one pixel with it, while parts of the 2×2 max-pooled map do not change at all. Which description fits?',
            options: ['Both layers are translation invariant', 'Convolution is invariant and pooling is equivariant', 'Convolution is equivariant and pooling adds a little invariance', 'Both layers are translation equivariant and nothing more'], answer: 2,
            explain: 'Equivariant: the output moves when the input moves. Invariant: the output stays put. You measured it in the first demo, where moving the shape down one pixel changed the pooled map noticeably less than the feature map.' },
          { q: 'With no pooling, how many stacked 3×3 layers does one unit need before it can see an 11×11 patch of the original image?',
            options: ['3', '5', '11', '4'], answer: 1,
            explain: 'Each 3×3 layer widens the view by two pixels: 3, 5, 7, 9, 11. Five layers. Check it on the receptive-field slider, then add one pool and watch the same five layers reach further.' },
          { q: 'A vision transformer reads a 224×224 image in 16×16 patches: 196 tokens. You double the image side to 448 and keep the patch size. Roughly what happens?',
            options: ['2× the tokens and 2× the attention cost', '2× the tokens and 4× the attention cost', '4× the tokens and 16× the attention cost', '4× the tokens and 4× the attention cost'], answer: 2,
            explain: '448 ÷ 16 = 28 patches per side, and 28 × 28 = 784 tokens, four times 196. Attention cost grows with the square of the token count, so 4² = 16 times. The patch demo shows exactly that at 448².' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://poloclub.github.io/cnn-explainer/" target="_blank" rel="noopener">CNN Explainer</a>: an interactive, layer-by-layer walk through a real trained network on real photos. The natural next step after the explorer above.`,
            `<a href="https://www.youtube.com/watch?v=KuXjwB4LzSA" target="_blank" rel="noopener">3Blue1Brown, "But what is a convolution?"</a>: the operation itself, visually, beyond its use in neural networks.`,
            `<a href="https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf" target="_blank" rel="noopener">Krizhevsky, Sutskever &amp; Hinton (2012), "ImageNet Classification with Deep Convolutional Neural Networks"</a>: the AlexNet paper that started it.`,
            `<a href="https://arxiv.org/abs/1512.03385" target="_blank" rel="noopener">He et al. (2015), "Deep Residual Learning for Image Recognition"</a>: skip connections, and why 100-layer networks became trainable.`,
            `<a href="https://arxiv.org/abs/2010.11929" target="_blank" rel="noopener">Dosovitskiy et al. (2020), "An Image is Worth 16×16 Words"</a>: the Vision Transformer, and the patch grid you just dragged.`,
          ]),
        ),
      );
    },
  });
})();
