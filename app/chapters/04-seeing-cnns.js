/* Zero → AGI · Chapter 04 · Seeing: convolutional networks
   Why a flattened image defeats an MLP; convolution as a sliding filter; feature maps; weight
   sharing = fewer parameters + translation equivariance; pooling; stacking builds a hierarchy.
   LeNet-5 (1998), AlexNet (2012), ResNet (2015). Bridge to ViT / multimodal LLMs.
   Interactives: (a) paintable 16×16 convolution explorer with a live sliding window and the
   multiply-add spelled out; (b) 2×2 max-pooling demo fed by (a)'s feature map; (c) a stylised
   feature-hierarchy animation (edges → textures → parts → object). */
(function () {
  ZTA.registerChapter({
    id: '04-seeing-cnns',
    num: 4,
    part: 'II',
    title: 'Seeing: convolutional networks',
    tagline: 'A 3×3 filter that never learns where it is, only what it is looking for, and enough of them stacked turn pixels into objects.',
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
      function featAt(r, c) { if (r < 0 || c < 0 || r >= OUTH || c >= OUTW) return 0; const v = Lab.feat[r * OUTW + c]; return (v == null || !isFinite(v)) ? 0 : v; }
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

        function draw() {
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
          g.fillText(SB.done + ' / ' + (PW * PH) + ' pooled cells · each keeps the loudest of its four neighbours and throws the rest away.', LP.x, H - 10);
        }
        let acc = 0;
        ctx.loop((dt) => { if (SB.playing) { acc += dt * SB.speed; while (acc >= 1) { acc -= 1; step(); } } draw(); });
        const playBtn = ctx.button('▶ Play', () => { SB.playing = !SB.playing; playBtn.textContent = SB.playing ? '⏸ Pause' : '▶ Play'; }, 'primary');
        const stepBtn = ctx.button('Step', () => { SB.playing = false; playBtn.textContent = '▶ Play'; step(); });
        const replayBtn = ctx.button('↺ Replay', () => { SB.playing = false; playBtn.textContent = '▶ Play'; reset(); });
        const speedSl = ctx.slider({ label: 'blocks / sec', min: 1, max: 30, step: 1, value: 5, onChange: (v) => { SB.speed = v; } });
        return ctx.figure(cv, 'Fed directly by the feature map from the explorer above — repaint the canvas or change the filter up there and this figure updates too. Pooling shrinks the map by keeping only the strongest response in each non-overlapping 2×2 block.', [playBtn, stepBtn, replayBtn, speedSl]);
      }

      /* ------------------------------------------------------------------ */
      /* Interactive C: stylised feature hierarchy (edges → … → object)      */
      /* ------------------------------------------------------------------ */
      function hierarchyDemo() {
        const W = 720, H = 300;
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
          g.fillText('stylised illustration of what deeper layers tend to represent — not the network\'s real learned filters, which look far messier', W / 2, 282);
        }
        ctx.loop((dt) => {
          if (SC.playing) { SC.depth += SC.dir * dt * SC.speed; if (SC.depth > 3) { SC.depth = 3; SC.dir = -1; } if (SC.depth < 0) { SC.depth = 0; SC.dir = 1; } depthSl.value = SC.depth; }
          draw();
        });
        const targetSel = ctx.select({ label: 'trained to recognise', options: [{ value: 'face', label: 'Face' }, { value: 'car', label: 'Car' }, { value: 'cat', label: 'Cat' }], value: 'face', onChange: (v) => { SC.target = v; } });
        const depthSl = ctx.slider({ label: 'depth', min: 0, max: 3, step: 0.02, value: 0, fmt: (v) => ['edges', 'textures', 'parts', 'object'][Math.round(v)], onChange: (v) => { SC.depth = v; } });
        const playBtn = ctx.button('⏸ Pause', () => { SC.playing = !SC.playing; playBtn.textContent = SC.playing ? '⏸ Pause' : '▶ Play'; }, 'primary');
        const speedSl = ctx.slider({ label: 'sweep speed', min: 0.1, max: 2, step: 0.1, value: 0.5, fmt: (v) => (+v).toFixed(1) + '×', onChange: (v) => { SC.speed = v; } });
        return ctx.figure(cv, 'A cartoon, not a measurement: real networks are not this tidy, and nobody hand-designs the "eye" or "wheel" detector — it emerges from data. But the direction is real: early layers respond to simple local patterns, later layers respond to larger, more specific ones built out of the earlier layers\' outputs.', [targetSel, depthSl, playBtn, speedSl]);
      }

      /* ================================================================== */
      /* Prose                                                              */
      /* ================================================================== */
      root.append(
        p(`Show a child a photo of a golden retriever peeking out from behind a couch — one eye, half an ear, a stripe of fur — and they'll say "dog" before you finish the sentence. Chapter 2 gave you a network that can, with enough hidden units, approximate any pattern you can define. So why not point it at a photograph and let it learn to recognise dogs? You can. It is also a quiet disaster, and understanding exactly why is the whole point of this chapter.`),
        p(`A modest photograph — 224 pixels wide, 224 tall, three colour channels — is 224 × 224 × 3 = <b>150,528</b> numbers. Flatten it into one long row and feed it to the multi-layer perceptron from chapter 2, wired to a modest 1,000 hidden units, and the very first layer alone needs upward of 150 million weights, before the network has looked at a single image. Worse: nudge the dog two pixels to the left, and as far as that network is concerned, every one of those 150,528 inputs just changed at once. It has never been told that a pattern shifted is still the same pattern — it would have to relearn "dog" separately for every possible position in the photo.`),
        p(`Something else was needed, and it was needed badly enough that a version of it was already reading handwritten cheques for American banks by 1998 — fourteen years before deep learning became a headline.`),

        section('Why a plain network can\'t see',
          p(`Three separate problems, and it helps to see them apart.`),
          ul([
            `<b>Scale.</b> A fully connected layer wires every input pixel to every hidden unit. Parameters grow with pixels × units, so a modest image and a modest hidden layer already cost tens or hundreds of millions of weights, before training has taught the network anything.`,
            `<b>Lost geometry.</b> Pixel (5,5) is a neighbour of (5,6) and (6,5) on the page. Flatten the image into one long vector and neighbouring pixels can land hundreds of positions apart; the network is never told they were ever close. It has to rediscover "nearby pixels usually belong together" from scratch, from data, every time.`,
            `<b>No translation.</b> A cat is a cat whether it stands in the centre of the frame or the corner. An MLP has a completely separate set of weights connecting position (10,10) to a given hidden unit than the weights connecting position (200,200) to that same unit — nothing forces those two sets of weights to learn "the same thing". A detector trained only on centred cats can simply fail to fire when the same cat drifts to a corner.`,
          ]),
          p(`Convolutional networks, invented to fix exactly this, are built around one idea: stop connecting every input to every output, and instead reuse one small set of weights everywhere the input goes.`),
        ),

        section('Convolution: a small filter that slides',
          p(`A <em>convolution</em> is a small grid of numbers — say a 3×3 grid, called a <em>filter</em> or <em>kernel</em> — placed over a 3×3 patch of the image. Multiply each of the nine filter numbers by the pixel underneath it, add the nine products together, and that single sum becomes one value of the output. Then slide the filter one pixel over and do it again. And again, until it has visited every position in the image. The grid of sums this produces, one per position, is called a <em>feature map</em>: not a picture any more, but a map of "how strongly did this exact pattern show up, here?"`),
          p(`Work through the smallest concrete case. Suppose a 3×3 patch of an image is dark (value 0) on the left column and light (value 1) on the other two, a clean vertical edge:`),
          ol([
            `<b>The patch.</b> <code class="inline">[[0,1,1],[0,1,1],[0,1,1]]</code> — three identical rows, dark-light-light.`,
            `<b>The filter.</b> The classic vertical-edge kernel is <code class="inline">[[-1,0,1],[-1,0,1],[-1,0,1]]</code>: negative on the left, zero in the middle, positive on the right.`,
            `<b>Multiply, cell by cell.</b> Top row: 0×(−1)=0, 1×0=0, 1×1=1. The middle and bottom rows give the same three numbers, 0, 0, 1. Nine products in all: 0,0,1, 0,0,1, 0,0,1.`,
            `<b>Sum them: 0+0+1+0+0+1+0+0+1 = 3.</b> A strong positive number — the filter "lit up" because the patch was exactly the pattern it was built to detect: dark on the left, light on the right.`,
            `<b>Slide one pixel over.</b> A patch that is uniformly light (no edge in sight) gives products that mix positive and nothing to cancel toward, landing near zero. Repeat across the whole image and you get a map that glows bright wherever a left-to-right edge sits, and stays flat everywhere else.`,
          ]),
          p(`That is the entire mechanism. A <em>stride</em> is how many pixels the filter moves each step (1, in the example above); <em>padding</em> is whether you add a border of zeros so the output stays the same size as the input, or let it shrink a little at the edges the way our 16×16 input shrinks to a 14×14 feature map below. Different 3×3 filters detect different things: a horizontal-edge kernel is the same idea rotated 90°; a blur kernel averages a patch instead of contrasting it; a sharpen kernel exaggerates a pixel against its neighbours.`),
          callout('tryit', 'Try it: feel the filter slide', `Start with the <b>Letter T</b> preset and the <b>Vertical edge</b> filter. Press <b>Step</b> a few times and read the arithmetic under the grids — it is exactly the calculation above, just with real numbers from your canvas. Press <b>▶ Play</b> and watch the feature map light up along the T's vertical stem (a vertical edge) while the horizontal bar produces almost nothing. Switch to <b>Horizontal edge</b> and the opposite happens. Now edit one of the nine numbers by hand and watch every future feature map change with it — the filter is nothing but those nine numbers. Finally paint your own shape and see what each filter finds in it.`),
          convExplorer(),
        ),

        section('Weight sharing: the whole trick in one paragraph',
          callout('key', 'Weight sharing = fewer parameters + translation equivariance', `The filter above has <b>nine numbers</b> (plus a bias), full stop — not nine per position, nine <i>total</i>, reused at all 196 positions in the 16×16 canvas. Compare that to an MLP, which would need a separate weight for every pixel-to-output connection at every position. Reusing the same small filter everywhere buys two things at once. First, <b>far fewer parameters</b>: a whole convolutional layer with dozens of filters can cost thousands of weights where a fully connected layer over the same image would cost millions. Second, <b>translation equivariance</b>: because the identical filter is dotted against every patch, a pattern found in the top-left produces the same response, just relocated, as the identical pattern found in the bottom-right. Shift the input, and the feature map shifts with it — the network never has to relearn a pattern just because it moved.`),
          p(`"Equivariant", not "invariant": the feature map still moves when the input moves, it just moves in lock-step rather than requiring separate weights for every position. True invariance — not caring <i>where</i> a pattern was, only that it was there — comes from the next ingredient.`),
        ),

        section('Pooling: shrinking without losing the point',
          p(`After a convolution produces a feature map, <em>pooling</em> shrinks it. The most common form, <em>max pooling</em>, slides a small window (2×2 is typical) across the feature map and keeps only the largest value in each window, throwing the other three away. A 14×14 feature map becomes a 7×7 map: a quarter of the numbers, and a quarter of the work for every layer downstream.`),
          p(`Shrinking is only half the point. Because max pooling only asks "was this pattern present <i>somewhere</i> in this little window", it buys a small amount of genuine translation <em>invariance</em>: shift the input by one pixel, and the strongest response in a 2×2 window is often still the strongest response, unmoved in the pooled output. Stack a few rounds of convolution-then-pooling, and the network cares less and less about the exact pixel where something happened and more and more about whether it happened nearby.`),
          callout('tryit', 'Try it: watch resolution disappear on purpose', `Go back to the explorer above, load the <b>Circle</b> preset with the <b>Sharpen</b> filter, then press <b>▶ Play</b> there and switch down to the pooling demo below. Press <b>Step</b> a few times and read the 2×2 window's four numbers each time — notice only the largest survives. Press <b>▶ Play</b> here too, then go back and repaint the input canvas above; watch this figure's left panel update to match, then rebuild the pooled map from the new data.`),
          poolingDemo(),
        ),

        section('Stacking layers: a hierarchy of features',
          p(`One convolutional layer only sees a 3×3 window at a time. Stack a second convolutional layer on top of the first layer's feature maps, and each unit in that second layer, even though its own filter is still only 3×3, is now looking at a 3×3 patch of <i>already-combined</i> edge responses — which corresponds to a larger patch of the original image. Its <em>receptive field</em>, the region of the original picture that can influence it, has grown. Stack a third layer and it grows again.`),
          p(`This is why depth matters for vision specifically, not just in general. Layer by layer, the same trick — small filters, weight-shared, pooled down, stacked again — tends to build a hierarchy: early layers respond to edges and simple contrasts; a little deeper, edges combine into textures and small motifs (corners, stripes, curves); deeper still, textures combine into parts (an eye, a wheel, an ear); and near the output, parts combine into whole objects. Nobody designs this hierarchy by hand. It falls out of training a deep stack of convolutions and poolings on enough labelled images, exactly the way backpropagation from chapter 2 tunes any other network.`),
          callout('tryit', 'Try it: scrub through depth', `Slide <b>depth</b> from 0 to 3 and watch the illustration move from oriented strokes, to little textures, to recognisable parts, to a finished object. Switch the target between <b>Face</b>, <b>Car</b> and <b>Cat</b> — the early "edges" and "textures" stages barely change (low-level structure is shared across almost everything you'd ever photograph), but "parts" and "object" change completely. Press <b>▶ Play</b> to let it sweep on its own.`),
          hierarchyDemo(),
        ),

        callout('history', 'LeNet-5 (1998) to AlexNet (2012): fourteen years in the wilderness', `Yann LeCun built the architecture in this chapter — convolution, pooling, several layers stacked, trained end-to-end with backpropagation — as <em>LeNet-5</em> in 1998, and banks used it to read the handwritten numbers on cheques. It worked. It was also, for over a decade, mostly a curiosity: the labelled datasets were small and GPUs for general computation barely existed, so hand-engineered features plus simpler classifiers were competitive on most real problems. That changed in 2012, when Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton entered <em>AlexNet</em> — a deeper descendant of LeNet-5, trained on two consumer GPUs, using <em>ReLU</em> activations (chapter 2) instead of sigmoids and a technique called <em>dropout</em> to fight overfitting — into the ImageNet competition, a contest to classify 1.2 million photos into 1,000 categories. AlexNet's top-5 error rate was about 15.3%; the best non-neural approach that year scored around 26.2%. That eleven-point gap, on a benchmark the whole field was watching, is usually cited as the single moment deep learning stopped being a niche interest and became the default approach to vision — and, within a few years, to almost everything else. Three years later, in 2015, Kaiming He and colleagues introduced <em>ResNet</em>, adding "skip connections" that let a layer's input jump straight to a later layer, unchanged, alongside whatever that layer computed. That one trick solved a problem where very deep stacks (past about 20 layers) actually got <i>worse</i> at training, and let networks grow to over 100 layers and keep improving.`),

        callout('example', 'Where convolutional networks are already in your pocket', `<b>Face unlock</b> on a phone uses a CNN-derived model to turn your face into a compact numerical fingerprint and check it against the one stored on the device. <b>Medical imaging</b> tools flag likely tumours in X-rays, CT and MRI scans by the same edges-to-parts-to-whole pipeline, often matching or beating specialist radiologists on narrow, well-defined tasks. <b>Self-driving perception</b> stacks — the part of the system that has to answer "where is the pedestrian, the lane line, the stop sign, right now" — are built from convolutional backbones. <b>Photo search</b> ("find my beach photos") works because a CNN has already turned every photo into a description of its contents before you ever type a query.`),

        section('Why this matters for modern AI',
          p(`Convolution's real contribution wasn't the specific 3×3-filter mechanics — it was proving that <b>building in a good assumption about the data's structure</b> (nearby pixels relate to each other; a pattern means the same thing wherever it appears) beats forcing a generic network to rediscover that assumption from scratch. That lesson outlived the convolution operation itself. In 2020, the <em>Vision Transformer</em> (ViT) showed that you could cut an image into a grid of patches (16×16 pixels each is typical), treat each patch as a single "token" — the same kind of object chapter 6 will show you for words — and hand the whole sequence of patch-tokens to a Transformer (chapter 7), the architecture behind every modern large language model. No hand-built sliding filter at all; instead, the model learns, via attention, which patches should pay attention to which other patches.`),
          p(`This is exactly how a modern multimodal model like Claude "sees" an image you upload: it is chopped into patches, each patch is embedded into the same kind of vector a word gets turned into, and the resulting sequence of image-tokens and text-tokens is processed together by one shared architecture. The specific tool changed — attention instead of a sliding kernel — but the chapter's central bet, that images have local, then compositional, then global structure worth respecting, is exactly the bet these systems still make.`),
        ),

        ctx.quiz([
          { q: 'A 224×224 colour photo, flattened, feeds a fully connected hidden layer of 1,000 units. Roughly how many weights does just that first layer need?', options: ['About 1,000', 'About 150,000', 'About 150 million', 'About 150 billion'], answer: 2, explain: '224×224×3 = 150,528 inputs, each wired to all 1,000 hidden units: 150,528 × 1,000 ≈ 150 million weights, before the network has learned anything. That parameter explosion, not any inherent unfairness to images, is the first reason convolution exists.' },
          { q: 'A 3×3 convolution filter has 9 numbers. Why can the same 9 numbers be reused at every position in a 512×512 image, rather than needing a separate 9 for every position?', options: ['Because images are usually mostly background', 'Because weight sharing assumes a pattern means the same thing wherever it appears — the fewer-parameters and translation-equivariance idea', 'Because convolution only works on grayscale images', 'Because the filter is re-randomised at each position'], answer: 1, explain: 'That is weight sharing: the same small filter slides everywhere. It cuts parameters enormously and means a pattern learned in one part of the image is recognised anywhere else it appears, without retraining.' },
          { q: 'What does 2×2 max pooling do to a feature map?', options: ['Doubles its resolution by interpolating new pixels', 'Replaces each non-overlapping 2×2 block with its single largest value, shrinking the map to a quarter its size', 'Blurs the map by averaging every pixel with its neighbours', 'Deletes the feature map and starts the network over'], answer: 1, explain: 'Max pooling keeps only the strongest response in each small window. That shrinks the map (less computation downstream) and adds a bit of real translation invariance: a pattern shifted by a pixel or two often still wins its window.' },
          { q: 'What made AlexNet (2012) a turning point rather than just an incremental improvement over LeNet-5 (1998)?', options: ['It was the first network to use any convolution at all', 'It combined GPU training, ReLU activations and dropout on a much larger labelled dataset, cutting the ImageNet error rate roughly in half against the best non-neural competitor', 'It removed pooling layers entirely', 'It was trained without any labelled data'], answer: 1, explain: "AlexNet's architecture descended directly from LeNet-5. What changed was scale and hardware: GPUs, ReLU, dropout, and 1.2 million labelled images let a top-5 error of about 15.3% beat the best non-neural method's roughly 26.2%, a gap the whole field noticed." },
          { q: 'How does a Vision Transformer (ViT) process an image, in contrast to a CNN?', options: ['It runs the exact same 3×3 sliding filters, just with more of them', 'It cuts the image into patches, treats each patch as a token, and lets self-attention learn which patches relate to which — no sliding filter at all', 'It converts the image to text first and reads the text', 'It only works on black-and-white images'], answer: 1, explain: 'ViT borrows the Transformer architecture from language: image patches become tokens, exactly like words, and the model learns relationships between them via attention rather than a hand-fixed sliding kernel. Multimodal LLMs "see" images the same way.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://www.youtube.com/watch?v=KuXjwB4LzSA" target="_blank" rel="noopener">3Blue1Brown, "But what is a convolution?"</a> — the clearest visual walkthrough of the sliding-window arithmetic in this chapter.`,
            `<a href="https://poloclub.github.io/cnn-explainer/" target="_blank" rel="noopener">CNN Explainer</a> — an interactive, in-browser visualisation of a real trained CNN classifying images, layer by layer.`,
            `<a href="https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf" target="_blank" rel="noopener">Krizhevsky, Sutskever &amp; Hinton (2012), "ImageNet Classification with Deep Convolutional Neural Networks"</a> — the original AlexNet paper.`,
            `<a href="https://arxiv.org/abs/1512.03385" target="_blank" rel="noopener">He, Zhang, Ren &amp; Sun (2015), "Deep Residual Learning for Image Recognition"</a> — the ResNet paper that made 100+-layer networks trainable.`,
            `<a href="https://arxiv.org/abs/2010.11929" target="_blank" rel="noopener">Dosovitskiy et al. (2020), "An Image is Worth 16x16 Words"</a> — the Vision Transformer paper bridging this chapter to chapter 7.`,
          ]),
        ),
      );
    },
  });
})();
