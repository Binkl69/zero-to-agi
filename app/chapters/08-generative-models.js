/* Zero → AGI · Chapter 08 · Generative models: VAEs, GANs and diffusion
   DESIGN RULE: the reader spends the first thirty seconds failing to find a picture by guessing
   pixels, so the size of the problem is felt before it is described.
   Interactives, in order: needle-in-a-haystack random pixel search; plain autoencoder vs VAE
   latent space with clickable dead zones; GAN mode-collapse dynamics; diffusion on a 2-D cloud;
   classifier-free guidance trading prompt adherence against variety; latent-space walk. */
(function () {
  ZTA.registerChapter({
    id: '08-generative-models',
    num: 8,
    part: 'II',
    title: 'Creating: GANs, VAEs and diffusion',
    tagline: 'Three different tricks for the same magic act: learn a probability distribution well enough to draw brand-new samples from it.',
    render(root, ctx) {
      const { h, p, section, sub, callout, ul, ol } = ctx;
      const C = ctx.colors;
      const FONT = '13px Inter, system-ui, sans-serif';
      const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
      const f2 = (v) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(2);
      const f3 = (v) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(3);
      const f4 = (v) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(4);

      /* ================================================================== */
      /* Interactive A: diffusion on a 2-D point cloud (the centrepiece)      */
      /* ================================================================== */
      function diffusionCloud() {
        const W = 720, H = 440;
        const [cv, g] = ctx.canvas(W, H);

        /* ---- build a ~300-point smiley: outline ring + two eye blobs + a smile arc ---- */
        function makeSmiley() {
          const pts = [];
          const nOutline = 150;
          for (let i = 0; i < nOutline; i++) {
            const a = (i / nOutline) * Math.PI * 2;
            pts.push({ x: Math.cos(a), y: Math.sin(a), part: 'outline' });
          }
          const eyeCenters = [{ x: -0.35, y: 0.32 }, { x: 0.35, y: 0.32 }];
          for (const ec of eyeCenters) {
            for (let i = 0; i < 25; i++) {
              const r = 0.12 * Math.sqrt(ctx.rand(0, 1)), a = ctx.rand(0, Math.PI * 2);
              pts.push({ x: ec.x + r * Math.cos(a), y: ec.y + r * Math.sin(a), part: 'eye' });
            }
          }
          const mc = { x: 0, y: 0.05 }, mr = 0.55;
          for (let i = 0; i < 100; i++) {
            const a = (200 + 140 * i / 99) * Math.PI / 180; // a smile: bottom of a circle
            pts.push({ x: mc.x + mr * Math.cos(a), y: mc.y + mr * Math.sin(a), part: 'mouth' });
          }
          // centre and rescale so the cloud's average per-axis variance is ≈ 1 (what the schedule below assumes)
          let mx = 0, my = 0; for (const q of pts) { mx += q.x; my += q.y; } mx /= pts.length; my /= pts.length;
          for (const q of pts) { q.x -= mx; q.y -= my; }
          let vs = 0; for (const q of pts) vs += q.x * q.x + q.y * q.y;
          const s = 1 / Math.sqrt(Math.max(1e-6, vs / (pts.length * 2)));
          for (const q of pts) { q.x *= s; q.y *= s; }
          return pts;
        }
        let data = makeSmiley();
        let epsFixed = data.map(() => ({ x: ctx.randn(), y: ctx.randn() }));

        /* ---- variance-preserving schedule (the one DDPM uses), in continuous time t∈[0,1] ---- */
        const BMIN = 0.1, BMAX = 20;
        const Bint = (t) => BMIN * t + 0.5 * (BMAX - BMIN) * t * t;
        const abar = (t) => Math.exp(-Bint(t));

        let manualT = 0;
        let mode = 'forward'; // 'forward' (slider-driven) or 'reverse' (denoising animation)
        const R = { t: 1, k: 0, steps: 40, xr: [], playing: false, acc: 0 };
        const STEP_DUR = 0.05;

        function initReverse() {
          R.xr = data.map(() => ({ x: ctx.randn(), y: ctx.randn() }));
          R.t = 1; R.k = 0; R.acc = 0; R.playing = false;
        }

        /* Tweedie's formula for the empirical data distribution: the posterior mean of x0 given a
           noisy xt is the softmax-weighted average of every clean training point, weighted by the
           Gaussian likelihood of xt under "noise this point at the current level". */
        function tweedieDenoise(t) {
          const ab = abar(t), sq = Math.sqrt(ab), vr = Math.max(1e-5, 1 - ab);
          const out = new Array(R.xr.length);
          const logits = new Array(data.length);
          for (let i = 0; i < R.xr.length; i++) {
            const xt = R.xr[i];
            let mx = -Infinity;
            for (let j = 0; j < data.length; j++) {
              const dx = xt.x - sq * data[j].x, dy = xt.y - sq * data[j].y;
              const lg = -(dx * dx + dy * dy) / (2 * vr);
              logits[j] = lg; if (lg > mx) mx = lg;
            }
            let sum = 0, sx = 0, sy = 0;
            for (let j = 0; j < data.length; j++) { const w = Math.exp(logits[j] - mx); sum += w; sx += w * data[j].x; sy += w * data[j].y; }
            out[i] = { x: sx / sum, y: sy / sum };
          }
          return out;
        }

        function reverseStep() {
          if (R.k >= R.steps) return;
          const t = R.t;
          const x0hat = tweedieDenoise(t);
          const ab = abar(t), sq = Math.sqrt(ab), so = Math.sqrt(Math.max(1e-6, 1 - ab));
          const tNext = Math.max(0, t - 1 / R.steps);
          const abN = abar(tNext), sqN = Math.sqrt(abN), soN = Math.sqrt(Math.max(0, 1 - abN));
          let bad = false;
          for (let i = 0; i < R.xr.length; i++) {
            const xt = R.xr[i], x0 = x0hat[i];
            const ex = (xt.x - sq * x0.x) / so, ey = (xt.y - sq * x0.y) / so; // implied noise (DDIM)
            const nx = sqN * x0.x + soN * ex, ny = sqN * x0.y + soN * ey;
            if (!isFinite(nx) || !isFinite(ny)) { bad = true; break; }
            R.xr[i] = { x: ctx.clamp(nx, -10, 10), y: ctx.clamp(ny, -10, 10) };
          }
          if (bad) { initReverse(); return; }
          R.t = tNext; R.k++;
        }

        function currentPoints() {
          if (mode === 'reverse') return R.xr.map((p, i) => ({ x: p.x, y: p.y, part: data[i].part }));
          const t = manualT, ab = abar(t), sq = Math.sqrt(ab), so = Math.sqrt(Math.max(0, 1 - ab));
          return data.map((d, i) => ({ x: sq * d.x + so * epsFixed[i].x, y: sq * d.y + so * epsFixed[i].y, part: d.part }));
        }

        const viewR = 3.2;
        function draw() {
          g.clearRect(0, 0, W, H);
          g.fillStyle = '#0f1520'; g.fillRect(0, 0, W, H);
          const cx = W / 2, cy = H / 2 - 6, scale = Math.min(W, H - 40) / (2 * viewR) * 0.94;
          g.strokeStyle = C.line; g.strokeRect(cx - viewR * scale, cy - viewR * scale, viewR * scale * 2, viewR * scale * 2);
          const pts = currentPoints();
          /* at high t a Gaussian sample can land well outside the view window, so clip the
             cloud to the frame: nothing can paint over the caption line beneath it */
          g.save();
          g.beginPath(); g.rect(cx - viewR * scale, cy - viewR * scale, viewR * scale * 2, viewR * scale * 2); g.clip();
          for (const q of pts) {
            const sx = cx + q.x * scale, sy = cy - q.y * scale;
            g.beginPath(); g.arc(sx, sy, 2.6, 0, Math.PI * 2);
            g.fillStyle = q.part === 'outline' ? C.accent : q.part === 'eye' ? C.warn : C.pink;
            g.globalAlpha = 0.88; g.fill();
          }
          g.globalAlpha = 1;
          g.restore();
          const t = mode === 'reverse' ? R.t : manualT, ab = abar(t);
          g.fillStyle = C.text; g.font = FONT; g.textAlign = 'left';
          const lbl = mode === 'reverse' ? 'Reverse (learned denoising): ' : 'Forward (destroying with noise): ';
          g.fillText(lbl + 't=' + f2(t) + '   ᾱ(t)=' + f4(ab) + '   noise σ=' + f3(Math.sqrt(Math.max(0, 1 - ab))) + (mode === 'reverse' ? '   step ' + R.k + '/' + R.steps : ''), 10, H - 12);
        }

        const ro = ctx.readout();
        ctx.loop((dt) => {
          if (mode === 'reverse' && R.playing) {
            R.acc += dt;
            while (R.acc >= STEP_DUR && R.k < R.steps) { reverseStep(); R.acc -= STEP_DUR; tSlider.value = R.t; }
            if (R.k >= R.steps) { R.playing = false; playBtn.textContent = '▶ Play reverse'; }
          }
          draw();
          const t = mode === 'reverse' ? R.t : manualT;
          ro.set({ mode, t: f2(t), 'ᾱ(t)': f4(abar(t)), 'reverse step': R.k + '/' + R.steps });
        });

        const tSlider = ctx.slider({ label: 't (0 = clean data, 1 = pure noise)', min: 0, max: 1, step: 0.01, value: 0, fmt: f2, onChange: (v) => { manualT = v; mode = 'forward'; R.playing = false; playBtn.textContent = '▶ Play reverse'; } });
        const stepsSlider = ctx.slider({ label: 'reverse steps', min: 10, max: 150, step: 5, value: 40, onChange: (v) => { R.steps = v; if (mode === 'reverse') initReverse(); } });
        const playBtn = ctx.button('▶ Play reverse', () => {
          if (mode !== 'reverse' || R.k >= R.steps) { initReverse(); mode = 'reverse'; }
          R.playing = !R.playing; playBtn.textContent = R.playing ? '⏸ Pause' : '▶ Play reverse';
        }, 'primary');
        const stepBtn = ctx.button('Step once', () => {
          if (mode !== 'reverse' || R.k >= R.steps) { initReverse(); mode = 'reverse'; }
          R.playing = false; playBtn.textContent = '▶ Play reverse';
          reverseStep(); tSlider.value = R.t;
        });
        const regenBtn = ctx.button('Regenerate', () => {
          data = makeSmiley(); epsFixed = data.map(() => ({ x: ctx.randn(), y: ctx.randn() }));
          if (mode === 'reverse') initReverse();
        });
        return ctx.figure(cv,
          `A ~300-point smiley, destroyed and rebuilt with the exact variance-preserving schedule DDPM uses. <b>Drag the t slider</b> to dissolve the shape into Gaussian noise. <b>Press ▶ Play reverse</b> to start from fresh noise and watch it denoise back into a face, one small step at a time. Each reverse step guesses the clean point behind the noise as the softmax-weighted average of all 300 training points under a Gaussian kernel (Tweedie's formula) — in a real model, a neural network replaces this kernel estimator, but the arithmetic it is trained to approximate is exactly this.`,
          [tSlider, stepsSlider, playBtn, stepBtn, regenBtn], ro);
      }

      /* ================================================================== */
      /* Interactive B: a 1-D GAN — forger vs. an analytically optimal judge  */
      /* ================================================================== */
      function ganDynamics() {
        const W = 720, H = 320;
        const [cv, g] = ctx.canvas(W, H);
        const X0 = -6, X1 = 6;
        const gauss = (x, m, s) => { const z = (x - m) / s; return Math.exp(-0.5 * z * z) / (s * Math.sqrt(2 * Math.PI)); };
        const RM1 = -2.2, RS1 = 0.55, RM2 = 2.2, RS2 = 0.55;
        const pdata = (x) => 0.5 * gauss(x, RM1, RS1) + 0.5 * gauss(x, RM2, RS2);
        /* the optimal discriminator D*(x) = p_data / (p_data + p_gen). The two epsilons are
           kept symmetric so that far out in the tails, where both densities underflow to
           nothing, the judge reads a neutral 0.5 instead of diving to zero on rounding dust. */
        const dStar = (x, mu, sigma) => { const pd = pdata(x), pg = gauss(x, mu, sigma); return (pd + 1e-12) / (pd + pg + 2e-12); };

        // a static histogram of "real" samples, drawn once
        const bins = 44, hist = new Array(bins).fill(0);
        for (let i = 0; i < 2400; i++) {
          const m = Math.random() < 0.5 ? RM1 : RM2, s = Math.random() < 0.5 ? RS1 : RS2;
          const x = m + s * ctx.randn();
          const b = Math.floor((ctx.clamp(x, X0, X1) - X0) / (X1 - X0) * bins);
          hist[ctx.clamp(b, 0, bins - 1)]++;
        }
        const histMax = Math.max(...hist);

        function defaults() { return { mu: 0.6, sigma: 1.3, collapse: false, playing: false, iter: 0, speed: 1, lastMu: 0.6, lastSigma: 1.3 }; }
        let G = defaults();

        function objective(mu, sigma, zs) {
          let s = 0;
          for (const z of zs) {
            const x = mu + sigma * z;
            s += Math.log(dStar(x, mu, sigma) + 1e-9);
          }
          s /= zs.length;
          /* entropy bonus — a real anti-collapse trick. It has to be strong enough to actually
             beat the mode-seeking pull of the game, or both settings collapse and the toggle
             shows nothing: at this weight the generator reliably spreads to straddle both modes. */
          if (!G.collapse) s += 2.0 * Math.log(Math.max(sigma, 1e-3));
          return s;
        }

        function updateGen() {
          const zs = []; for (let i = 0; i < 48; i++) zs.push(ctx.randn());
          const hMu = 0.02, hS = Math.max(0.01, G.sigma * 0.03);
          const gMu = (objective(G.mu + hMu, G.sigma, zs) - objective(G.mu - hMu, G.sigma, zs)) / (2 * hMu);
          const gS = (objective(G.mu, G.sigma + hS, zs) - objective(G.mu, G.sigma - hS, zs)) / (2 * hS);
          const cgMu = ctx.clamp(gMu, -8, 8), cgS = ctx.clamp(gS, -8, 8);
          const lrMu = 0.55, lrS = 0.22;
          let nmu = G.mu + lrMu * cgMu, nsig = G.sigma + lrS * cgS;
          nsig = ctx.clamp(nsig, 0.12, 3.2); nmu = ctx.clamp(nmu, -4.5, 4.5);
          if (!isFinite(nmu) || !isFinite(nsig)) { G = defaults(); return; }
          G.mu = nmu; G.sigma = nsig; G.iter++;
        }

        const plot = { x: 46, y: 30, w: 660, h: 234 };
        const xAt = (x) => plot.x + (x - X0) / (X1 - X0) * plot.w;
        function draw() {
          g.clearRect(0, 0, W, H);
          g.fillStyle = '#0f1520'; g.fillRect(plot.x, plot.y, plot.w, plot.h); g.strokeStyle = C.line; g.strokeRect(plot.x, plot.y, plot.w, plot.h);
          g.save(); g.beginPath(); g.rect(plot.x, plot.y, plot.w, plot.h); g.clip();
          // real-data histogram
          const bw = plot.w / bins;
          g.fillStyle = 'rgba(124,156,255,0.35)';
          for (let i = 0; i < bins; i++) {
            const hgt = hist[i] / histMax * (plot.h * 0.62);
            g.fillRect(plot.x + i * bw, plot.y + plot.h - hgt, bw - 1, hgt);
          }
          // generator density curve + discriminator score curve
          let maxPg = 0; for (let i = 0; i <= 200; i++) maxPg = Math.max(maxPg, gauss(X0 + (X1 - X0) * i / 200, G.mu, G.sigma));
          g.beginPath(); g.strokeStyle = C.danger; g.lineWidth = 2.2;
          for (let i = 0; i <= 200; i++) {
            const x = X0 + (X1 - X0) * i / 200, pg = gauss(x, G.mu, G.sigma);
            const yy = plot.y + plot.h - (pg / (maxPg || 1)) * (plot.h * 0.62);
            i === 0 ? g.moveTo(xAt(x), yy) : g.lineTo(xAt(x), yy);
          }
          g.stroke();
          g.beginPath(); g.strokeStyle = C.warn; g.lineWidth = 1.8; g.setLineDash([5, 3]);
          for (let i = 0; i <= 200; i++) {
            const x = X0 + (X1 - X0) * i / 200, D = dStar(x, G.mu, G.sigma);
            const yy = plot.y + plot.h - D * plot.h * 0.9;
            i === 0 ? g.moveTo(xAt(x), yy) : g.lineTo(xAt(x), yy);
          }
          g.stroke(); g.setLineDash([]);
          // a handful of current generator samples along the baseline
          g.fillStyle = C.danger;
          for (let i = 0; i < 28; i++) { const x = G.mu + G.sigma * ctx.randn(); g.beginPath(); g.arc(xAt(ctx.clamp(x, X0 + 0.05, X1 - 0.05)), plot.y + plot.h - 4, 2.4, 0, Math.PI * 2); g.fill(); }
          g.restore();
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          g.fillText('blue bars = real data   red = generator’s density   dashed yellow = discriminator score D(x)', plot.x, plot.y - 10);
          g.fillStyle = C.text; g.font = MONO; g.textAlign = 'left';
          g.fillText('iter ' + G.iter + '   μ=' + f2(G.mu) + '   σ=' + f2(G.sigma) + (G.collapse ? '   [mode collapse: ON]' : ''), plot.x, plot.y + plot.h + 18);
        }
        const ro = ctx.readout();
        ctx.loop(() => {
          if (G.playing) for (let i = 0; i < G.speed; i++) updateGen();
          draw();
          ro.set({ iteration: G.iter, μ: f2(G.mu), σ: f2(G.sigma) });
        });
        const playBtn = ctx.button('▶ Train', () => { G.playing = !G.playing; playBtn.textContent = G.playing ? '⏸ Pause' : '▶ Train'; }, 'primary');
        const stepBtn = ctx.button('Step', () => updateGen());
        const resetBtn = ctx.button('Reset generator', () => { const c = G.collapse; G = defaults(); G.collapse = c; });
        const collapseBtn = ctx.button('Mode collapse: off', () => {
          G.collapse = !G.collapse; collapseBtn.textContent = 'Mode collapse: ' + (G.collapse ? 'ON' : 'off');
        });
        const spSl = ctx.slider({ label: 'steps per frame', min: 1, max: 8, step: 1, value: 1, onChange: (v) => { G.speed = v; } });
        return ctx.figure(cv,
          `The generator is a single movable Gaussian (red). The discriminator is not a trained network here — it is the exact optimal judge from Goodfellow's proof, D*(x) = p<sub>data</sub>(x) / (p<sub>data</sub>(x) + p<sub>gen</sub>(x)) (dashed yellow), so you are watching the generator's side of the game with a perfect opponent. Its mean and spread are updated every iteration by a genuine gradient step (computed by finite differences) on "fool the judge" — plus, with the mode-collapse toggle off, an explicit bonus for staying spread out, which no real GAN loss contains and which is here only so that the toggle has two distinguishable behaviours to show you. A single Gaussian can never truly cover two separate bumps at once, which is exactly why this toy is useful: watch it either spread out to straddle both modes, or — with the toggle on — shrink and lock onto just one, ignoring the other entirely.`,
          [playBtn, stepBtn, resetBtn, collapseBtn, spSl], ro);
      }

      /* ================================================================== */
      /* Interactive C: a 2-D latent pad, decoded into a procedural face      */
      /* ================================================================== */
      function latentWalk() {
        const PAD = 280, FACE = 280;
        const [cv1, g1] = ctx.canvas(PAD, PAD);
        const [cv2, g2] = ctx.canvas(FACE, FACE);
        const RNG = 1.5;
        let z = { x: 0.2, y: -0.2 };
        let A = { x: -1.1, y: -1.1 }, B = { x: 1.1, y: 1.1 };
        let dragging = false, auto = false;

        const padCenter = { x: PAD / 2, y: PAD / 2 }, padScale = (PAD * 0.42) / RNG;
        function toLatent(ev) {
          const pos = cv1.pos(ev);
          return { x: ctx.clamp((pos.x - padCenter.x) / padScale, -RNG, RNG), y: ctx.clamp(-(pos.y - padCenter.y) / padScale, -RNG, RNG) };
        }
        function drawPad() {
          g1.clearRect(0, 0, PAD, PAD);
          g1.fillStyle = '#0f1520'; g1.fillRect(0, 0, PAD, PAD);
          g1.strokeStyle = C.line; g1.lineWidth = 1;
          for (let k = -1.5; k <= 1.5; k += 0.5) {
            const px = padCenter.x + k * padScale, py = padCenter.y - k * padScale;
            g1.beginPath(); g1.moveTo(px, 0); g1.lineTo(px, PAD); g1.stroke();
            g1.beginPath(); g1.moveTo(0, py); g1.lineTo(PAD, py); g1.stroke();
          }
          g1.strokeStyle = C.muted; g1.strokeRect(0.5, 0.5, PAD - 1, PAD - 1);
          /* A and B can legitimately be dropped on the same spot, so give them different
             radii and put their labels on opposite sides: both stay readable when they coincide. */
          const mark = (pt, color, label, r, above) => {
            const px = padCenter.x + pt.x * padScale, py = padCenter.y - pt.y * padScale;
            g1.beginPath(); g1.arc(px, py, r, 0, Math.PI * 2); g1.strokeStyle = color; g1.lineWidth = 2; g1.stroke();
            g1.fillStyle = color; g1.font = FONT; g1.textAlign = 'center';
            const ly = above ? Math.max(py - r - 6, 12) : Math.min(py + r + 13, PAD - 6);
            g1.fillText(label, px, ly);
          };
          mark(A, C.green, 'A', 5, true); mark(B, C.orange, 'B', 9, false);
          const zx = padCenter.x + z.x * padScale, zy = padCenter.y - z.y * padScale;
          g1.beginPath(); g1.arc(zx, zy, 7, 0, Math.PI * 2); g1.fillStyle = C.accent; g1.fill(); g1.strokeStyle = '#0a0e16'; g1.lineWidth = 1.5; g1.stroke();
          g1.fillStyle = C.muted; g1.font = FONT; g1.textAlign = 'left';
          /* the rotated label reads upwards, so it is anchored low enough that its far end
             (roughly 32px of text) stays on the canvas instead of running off the top */
          g1.fillText('z1 →', 6, PAD - 6); g1.save(); g1.translate(15, PAD - 28); g1.rotate(-Math.PI / 2); g1.fillText('z2 →', 0, 0); g1.restore();
        }
        function drawFace() {
          g2.clearRect(0, 0, FACE, FACE);
          g2.fillStyle = '#0f1520'; g2.fillRect(0, 0, FACE, FACE);
          const z1 = ctx.clamp(z.x, -RNG, RNG), z2 = ctx.clamp(z.y, -RNG, RNG);
          const cx = FACE / 2, cy = FACE / 2 + 6;
          const width = 58 + 15 * z1;
          const eyeDist = 20 + 8 * z2;
          const eyeSize = ctx.clamp(7 + 2.4 * Math.tanh(z1), 2, 14);
          const browAngle = z1 * 10;
          const mouthCurve = Math.tanh(z2 * 1.1) * 22; // + = smile (bulges down), − = frown (arches up)
          const tilt = z1 * 4;
          g2.save(); g2.translate(cx, cy); g2.rotate(tilt * Math.PI / 180);
          g2.beginPath(); g2.ellipse(0, 0, Math.max(20, width), Math.max(20, width) * 1.08, 0, 0, Math.PI * 2);
          g2.fillStyle = '#111827'; g2.fill(); g2.strokeStyle = C.text; g2.lineWidth = 2; g2.stroke();
          [-1, 1].forEach((s) => { g2.beginPath(); g2.arc(s * eyeDist, -10, eyeSize, 0, Math.PI * 2); g2.fillStyle = C.text; g2.fill(); });
          [-1, 1].forEach((s) => {
            g2.save(); g2.translate(s * eyeDist, -10 - eyeSize - 9); g2.rotate(s * browAngle * Math.PI / 180);
            g2.strokeStyle = C.muted; g2.lineWidth = 2.5; g2.beginPath(); g2.moveTo(-9, 0); g2.lineTo(9, 0); g2.stroke(); g2.restore();
          });
          g2.beginPath(); g2.moveTo(-24, 30); g2.quadraticCurveTo(0, 30 + mouthCurve, 24, 30);
          g2.strokeStyle = C.accent; g2.lineWidth = 3; g2.stroke();
          g2.restore();
          g2.fillStyle = C.muted; g2.font = MONO; g2.textAlign = 'left';
          g2.fillText('z = (' + f2(z1) + ', ' + f2(z2) + ')', 8, FACE - 8);
        }
        cv1.addEventListener('pointerdown', (ev) => { dragging = true; auto = false; autoBtn.textContent = 'Auto walk: off'; z = toLatent(ev); try { cv1.setPointerCapture(ev.pointerId); } catch (e) {} });
        cv1.addEventListener('pointermove', (ev) => { if (!dragging) return; z = toLatent(ev); });
        const stop = () => { dragging = false; };
        cv1.addEventListener('pointerup', stop); cv1.addEventListener('pointercancel', stop);
        cv1.style.touchAction = 'none';

        const ro = ctx.readout();
        ctx.loop((dt, t) => {
          if (auto) { z = { x: RNG * 0.72 * Math.cos(t * 0.5), y: RNG * 0.72 * Math.sin(t * 0.5) }; interpSl.value = 0; }
          drawPad(); drawFace();
          ro.set({ z1: f2(z.x), z2: f2(z.y) });
        });
        const setABtn = ctx.button('Set A here', () => { A = { x: z.x, y: z.y }; });
        const setBBtn = ctx.button('Set B here', () => { B = { x: z.x, y: z.y }; });
        const interpSl = ctx.slider({ label: 'interpolate A → B', min: 0, max: 1, step: 0.01, value: 0, fmt: f2, onChange: (v) => { auto = false; autoBtn.textContent = 'Auto walk: off'; z = { x: ctx.lerp(A.x, B.x, v), y: ctx.lerp(A.y, B.y, v) }; } });
        const autoBtn = ctx.button('Auto walk: off', () => { auto = !auto; autoBtn.textContent = 'Auto walk: ' + (auto ? 'on' : 'off'); });
        const body = h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '16px', justifyContent: 'center', alignItems: 'flex-start' } },
          h('div', { style: { flex: '1 1 220px', maxWidth: '320px' } }, cv1),
          h('div', { style: { flex: '1 1 220px', maxWidth: '320px' } }, cv2));
        return ctx.figure(body,
          `Drag the blue dot on the left pad — that is a two-number latent code z. The face on the right is decoded from it by a handful of smooth formulas (face width, eye spacing, brow angle, smile curve), standing in for a decoder network. <b>Set A</b> and <b>Set B</b> at two spots you like, then drag the interpolation slider: the face morphs continuously between them, because nearby points in a well-trained latent space decode to nearby, similar outputs — never a jarring jump.`,
          [setABtn, setBBtn, interpSl, autoBtn], ro);
      }

      /* ================================================================== */

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

      /* ================================================================== */
      /* Interactive: try to find a picture by guessing pixels                */
      /* ================================================================== */
      function needleLab() {
        const [cv, g] = ctx.canvas(720, 340);
        const N = 8, LEVELS = 16;
        let grid = null, tries = 0, best = 0, auto = false, acc = 0;
        /* "realness": how strongly neighbouring pixels agree. Photographs are overwhelmingly
           smooth at this scale; random pixels are not. A crude proxy, but an honest one. */
        function realness(gr) {
          let same = 0, tot = 0;
          for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
            if (c < N - 1) { same += Math.abs(gr[r][c] - gr[r][c + 1]) <= 2 ? 1 : 0; tot++; }
            if (r < N - 1) { same += Math.abs(gr[r][c] - gr[r + 1][c]) <= 2 ? 1 : 0; tot++; }
          }
          return tot ? same / tot : 0;
        }
        function randomGrid() {
          const gr = [];
          for (let r = 0; r < N; r++) { gr.push([]); for (let c = 0; c < N; c++) gr[r].push((Math.random() * LEVELS) | 0); }
          return gr;
        }
        /* three genuine little "photographs" for comparison */
        const REAL = {
          smiley: () => {
            const gr = []; for (let r = 0; r < N; r++) { gr.push(new Array(N).fill(3)); }
            for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
              const d = Math.hypot(r - 3.5, c - 3.5);
              gr[r][c] = d < 3.6 ? 12 : 3;
            }
            gr[2][2] = 1; gr[2][5] = 1;
            gr[5][2] = 1; gr[5][3] = 1; gr[5][4] = 1; gr[5][5] = 1;
            return gr;
          },
          gradient: () => { const gr = []; for (let r = 0; r < N; r++) { gr.push([]); for (let c = 0; c < N; c++) gr[r].push(Math.round((r + c) / (2 * N - 2) * (LEVELS - 1))); } return gr; },
        };
        grid = randomGrid(); tries = 1; best = realness(grid);

        const drawBtn = ctx.button('Draw random pixels', () => { grid = randomGrid(); tries++; best = Math.max(best, realness(grid)); }, 'primary');
        const autoBtn = ctx.button('Keep trying', () => { auto = !auto; autoBtn.textContent = auto ? 'Stop' : 'Keep trying'; });
        const realBtn = ctx.button('Show me a real one', () => { grid = REAL.smiley(); });
        const gradBtn = ctx.button('…and another', () => { grid = REAL.gradient(); });
        const ro = ctx.readout();

        ctx.loop((dt) => {
          if (auto) { acc += dt; if (acc > 0.05) { acc = 0; grid = randomGrid(); tries++; best = Math.max(best, realness(grid)); } }
          g.clearRect(0, 0, cv.W, cv.H);
          const r0 = realness(grid);

          /* the image */
          const CELL = 26, X = 40, Y = 56;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('an 8 × 8 image, 16 shades of grey', X, 34);
          for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
            const v = grid[r][c] / (LEVELS - 1);
            g.fillStyle = 'rgb(' + Math.round(v * 235) + ',' + Math.round(v * 238) + ',' + Math.round(v * 245) + ')';
            g.fillRect(X + c * CELL, Y + r * CELL, CELL - 1, CELL - 1);
          }

          /* the scoreboard */
          const TX = 300;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('how picture-like is it?', TX, 34);
          g.fillStyle = C.line; g.fillRect(TX, 46, 340, 18);
          g.fillStyle = r0 > 0.8 ? C.green : r0 > 0.65 ? C.warn : C.danger;
          g.fillRect(TX, 46, r0 * 340, 18);
          g.font = 'bold 16px Inter, system-ui, sans-serif';
          g.fillStyle = r0 > 0.8 ? C.green : C.muted;
          g.fillText((r0 * 100).toFixed(1) + '%', TX + 348, 61);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('neighbouring pixels agreeing — real photos score high', TX, 82);

          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('images drawn at random:', TX, 118);
          g.font = 'bold 22px Inter, system-ui, sans-serif'; g.fillStyle = C.text;
          g.fillText(tries.toLocaleString(), TX + 210, 120);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('best random draw:', TX, 144);
          g.font = 'bold 22px Inter, system-ui, sans-serif'; g.fillStyle = best > 0.8 ? C.green : C.danger;
          g.fillText((best * 100).toFixed(1) + '%', TX + 210, 146);

          g.font = 'bold ' + FONT; g.fillStyle = C.warn;
          g.fillText('the size of the haystack', TX, 186);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('16 ^ 64  ≈  10 ^ 77 possible images', TX, 208);
          g.fillText('atoms in the observable universe ≈ 10 ^ 80', TX, 226);

          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, 'You will not find a photograph this way, and neither would every computer that has ever existed running until the sun burns out. The pictures are a vanishingly thin sliver inside that space. A generative model is a machine for landing inside the sliver on the first try.',
            40, 288, 640, 18);
          ro.set({ tries, 'this image': (r0 * 100).toFixed(1) + '%', 'best random draw': (best * 100).toFixed(1) + '%' });
        });

        return ctx.figure(cv,
          'The whole problem of generative modelling in one picture. Real images are points in a very high-dimensional space, and almost every point in that space is static — the real ones form a tiny, oddly-shaped cloud. Even at this absurdly small size, 8×8 pixels in 16 greys, there are about 10⁷⁷ possibilities and essentially none of them are pictures. Press <b>Show me a real one</b> to see what the score looks like when you land in the cloud. The rest of this chapter is three different ways of getting there deliberately.',
          [drawBtn, autoBtn, realBtn, gradBtn], ro);
      }

      /* ================================================================== */
      /* Interactive: why an autoencoder needs the "V"                       */
      /* ================================================================== */
      function vaeLab() {
        const W = 720, H = 432;
        const [cv, g] = ctx.canvas(W, H);
        let mode = 'ae', zx = 0.15, zy = 0.1;
        /* 40 training images, each encoded to a latent point. A plain autoencoder is free to
           park them anywhere it likes, so they land in far-apart islands with dead space in
           between. A VAE's KL term pulls every code toward one shared standard normal. */
        const rnd = (seed) => { let a = seed; return () => { a = (a * 1664525 + 1013904223) % 4294967296; return a / 4294967296; }; };
        const CODES = (() => {
          const r = rnd(11), out = [];
          const islands = [[-0.75, 0.6], [0.8, 0.55], [-0.6, -0.7], [0.7, -0.6]];
          for (let i = 0; i < 40; i++) {
            const is = islands[i % 4];
            const ae = { x: is[0] + (r() - 0.5) * 0.22, y: is[1] + (r() - 0.5) * 0.22 };
            const ang = r() * Math.PI * 2, rad = Math.sqrt(r()) * 0.85;
            const vae = { x: Math.cos(ang) * rad, y: Math.sin(ang) * rad };
            out.push({ ae, vae, kind: i % 4 });
          }
          return out;
        })();
        const pts = () => CODES.map(c => ({ p: mode === 'ae' ? c.ae : c.vae, kind: c.kind }));
        /* distance to the nearest training code decides whether the decoder has any idea */
        function nearest(x, y) {
          let d = Infinity, kind = 0;
          pts().forEach(o => { const dd = Math.hypot(o.p.x - x, o.p.y - y); if (dd < d) { d = dd; kind = o.kind; } });
          return { d, kind };
        }

        const modeLabel = () => (mode === 'ae' ? 'Switch to variational autoencoder' : 'Switch to plain autoencoder');
        const modeBtn = ctx.button(modeLabel(), () => {
          mode = mode === 'ae' ? 'vae' : 'ae';
          modeBtn.textContent = modeLabel();
        }, 'primary');
        const sampleBtn = ctx.button('Sample fresh randomness', () => {
          const ang = Math.random() * Math.PI * 2, rad = Math.sqrt(Math.random()) * 0.85;
          zx = Math.cos(ang) * rad; zy = Math.sin(ang) * rad;
        });
        const PAD = { x: 40, y: 56, s: 260 }, ZMAX = 0.92;
        const toPad = (x, y) => ({ px: PAD.x + (x + 1) / 2 * PAD.s, py: PAD.y + (1 - y) / 2 * PAD.s });
        const fromPad = (px, py) => ({ x: (px - PAD.x) / PAD.s * 2 - 1, y: 1 - (py - PAD.y) / PAD.s * 2 });
        let drag = false;
        const grab = (e) => {
          const q = cv.pos(e);
          if (q.x < PAD.x - 12 || q.x > PAD.x + PAD.s + 12 || q.y < PAD.y - 12 || q.y > PAD.y + PAD.s + 12) return false;
          const z = fromPad(q.x, q.y);
          /* inset by the ring's own radius so the marker never straddles the frame */
          zx = ctx.clamp(z.x, -ZMAX, ZMAX); zy = ctx.clamp(z.y, -ZMAX, ZMAX);
          return true;
        };
        cv.addEventListener('pointerdown', (e) => { e.preventDefault(); drag = grab(e); });
        cv.addEventListener('pointermove', (e) => { if (drag) grab(e); });
        ['pointerup', 'pointercancel', 'pointerleave'].forEach(t => cv.addEventListener(t, () => { drag = false; }));
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, W, H);
          const near = nearest(zx, zy);
          const dead = near.d > 0.22;

          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('latent space — where each image is filed', PAD.x, 34);
          g.strokeStyle = C.line; g.lineWidth = 1;
          g.strokeRect(PAD.x, PAD.y, PAD.s, PAD.s);
          /* shade the region the decoder has actually been trained to explain */
          g.save(); g.beginPath(); g.rect(PAD.x, PAD.y, PAD.s, PAD.s); g.clip();
          for (let i = 0; i < 34; i++) for (let j = 0; j < 34; j++) {
            const q = fromPad(PAD.x + (i + 0.5) * PAD.s / 34, PAD.y + (j + 0.5) * PAD.s / 34);
            const nd = nearest(q.x, q.y).d;
            if (nd < 0.22) {
              g.fillStyle = 'rgba(56,217,169,' + (0.22 * (1 - nd / 0.22)) + ')';
              g.fillRect(PAD.x + i * PAD.s / 34, PAD.y + j * PAD.s / 34, PAD.s / 34 + 0.6, PAD.s / 34 + 0.6);
            }
          }
          const COLS = [C.accent, C.warn, C.purple, C.pink];
          pts().forEach(o => {
            const q = toPad(o.p.x, o.p.y);
            g.fillStyle = COLS[o.kind];
            g.beginPath(); g.arc(q.px, q.py, 3.5, 0, 7); g.fill();
          });
          const zq = toPad(zx, zy);
          g.strokeStyle = dead ? C.danger : C.green; g.lineWidth = 2.5;
          g.beginPath(); g.arc(zq.px, zq.py, 9, 0, 7); g.stroke();
          g.restore();
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('drag the ring anywhere', PAD.x, PAD.y + PAD.s + 20);
          g.fillText('shaded = the decoder was trained here', PAD.x, PAD.y + PAD.s + 38);

          /* the decoded output */
          const OX = 370, OY = 56, OS = 170;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('what the decoder produces there', OX, 34);
          g.fillStyle = '#0d1320'; g.fillRect(OX, OY, OS, OS);
          g.save(); g.beginPath(); g.rect(OX, OY, OS, OS); g.clip();
          if (dead) {
            /* nothing was ever trained here, so the output is incoherent */
            for (let i = 0; i < 26; i++) for (let j = 0; j < 26; j++) {
              const v = Math.abs(Math.sin((i * 7.3 + zx * 31) * (j * 3.1 + zy * 17)));
              g.fillStyle = 'rgba(' + Math.round(v * 200) + ',' + Math.round((1 - v) * 160) + ',' + Math.round(v * 120 + 60) + ',0.85)';
              g.fillRect(OX + i * OS / 26, OY + j * OS / 26, OS / 26 + 0.6, OS / 26 + 0.6);
            }
          } else {
            /* a smooth shape whose parameters vary continuously with z */
            const cxp = OX + OS / 2, cyp = OY + OS / 2;
            const rr = 32 + zx * 20 + zy * 7;
            const squash = 1 + zy * 0.45;
            g.fillStyle = COLS[near.kind];
            g.globalAlpha = 0.85;
            g.beginPath();
            for (let a = 0; a <= 64; a++) {
              const th = a / 64 * Math.PI * 2;
              const wob = 1 + 0.18 * Math.sin(th * (3 + near.kind) + zx * 3);
              const px = cxp + Math.cos(th) * rr * wob, py = cyp + Math.sin(th) * rr * wob / squash;
              a ? g.lineTo(px, py) : g.moveTo(px, py);
            }
            g.closePath(); g.fill(); g.globalAlpha = 1;
          }
          g.restore();
          g.strokeStyle = C.line; g.strokeRect(OX, OY, OS, OS);

          g.font = 'bold 15px Inter, system-ui, sans-serif';
          g.fillStyle = dead ? C.danger : C.green;
          (dead ? ['dead space —', 'the decoder has never been here'] : ['a coherent output'])
            .forEach((ln, i) => g.fillText(ln, OX, OY + OS + 24 + i * 19));
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, mode === 'ae'
            ? 'A plain autoencoder is only ever asked to rebuild the images it was shown. Nothing requires the space between those codes to mean anything, so it fills with dead zones — and sampling randomness lands in one about five times in six.'
            : 'The VAE encodes each image to a small cloud rather than a point, and a second loss term pulls every cloud toward one shared standard normal. The islands merge, the gaps close, and fresh randomness now lands somewhere the decoder understands.',
            PAD.x, 382, W - 2 * PAD.x, 17);
          ro.set({ mode: mode === 'ae' ? 'plain autoencoder' : 'VAE', z: '(' + zx.toFixed(2) + ', ' + zy.toFixed(2) + ')', landed: dead ? 'dead space' : 'trained region' });
        });

        return ctx.figure(cv,
          'A schematic of the one idea that separates a VAE from a plain autoencoder. Both squeeze an image into a short latent code and expand it back. But a plain autoencoder may file those codes anywhere convenient, so they clump into far-apart islands with untrained dead space between them — and a generator needs to be able to feed in <i>fresh randomness</i>, which will land in the dead space almost every time. Press <b>Sample fresh randomness</b> repeatedly in each mode and count how often you get something coherent.',
          [modeBtn, sampleBtn], ro);
      }

      /* ================================================================== */
      /* Interactive: classifier-free guidance, and what it costs             */
      /* ================================================================== */
      function guidanceLab() {
        const [cv, g] = ctx.canvas(720, 330);
        let guide = 3.0, n = 60, seed = 5;
        const rnd = (s) => { let a = s; return () => { a = (a * 1664525 + 1013904223) % 4294967296; return a / 4294967296; }; };
        /* the data distribution is a wide ring of "all plausible images"; the prompt names a
           small region of it. Guidance pushes each sample from the ring toward that region. */
        const TARGET = { x: 0.45, y: 0.35 };
        function samples() {
          const r = rnd(seed), out = [];
          for (let i = 0; i < n; i++) {
            const ang = r() * Math.PI * 2, rad = 0.35 + r() * 0.55;
            const base = { x: Math.cos(ang) * rad, y: Math.sin(ang) * rad };
            const t = 1 - Math.exp(-guide / 5);           // 0 at no guidance, →1 as it climbs
            out.push({ x: base.x + (TARGET.x - base.x) * t, y: base.y + (TARGET.y - base.y) * t });
          }
          return out;
        }
        const gSl = ctx.slider({ label: 'guidance scale', min: 0, max: 15, step: 0.5, value: 3, digits: 1, onChange: (v) => { guide = v; } });
        const reseed = ctx.button('New batch', () => { seed = (seed * 7 + 13) % 99991; }, 'primary');
        const p0 = ctx.button('no guidance (0)', () => { guide = 0; gSl.value = 0; });
        const p7 = ctx.button('typical (7.5)', () => { guide = 7.5; gSl.value = 7.5; });
        const p15 = ctx.button('cranked (15)', () => { guide = 15; gSl.value = 15; });
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          const S = samples();
          const CX = 210, CY = 165, SC = 120;
          const sx = (x) => CX + x * SC, sy = (y) => CY - y * SC;

          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('one batch of generated images', 40, 28);
          /* the data distribution */
          g.strokeStyle = 'rgba(148,163,184,0.35)'; g.lineWidth = 1; g.setLineDash([4, 4]);
          [0.35, 0.9].forEach(r => { g.beginPath(); g.arc(CX, CY, r * SC, 0, 7); g.stroke(); });
          g.setLineDash([]);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('everything the model could make', CX - 90, CY - 0.9 * SC - 10);
          /* the prompt region */
          g.fillStyle = 'rgba(56,217,169,0.13)';
          g.beginPath(); g.arc(sx(TARGET.x), sy(TARGET.y), 26, 0, 7); g.fill();
          g.strokeStyle = C.green; g.lineWidth = 2;
          g.beginPath(); g.arc(sx(TARGET.x), sy(TARGET.y), 26, 0, 7); g.stroke();
          /* the samples — clipped so no batch can spill into the scoreboard column */
          g.save(); g.beginPath(); g.rect(0, 38, 400, 292); g.clip();
          let onPrompt = 0;
          S.forEach(s => {
            const hit = Math.hypot(s.x - TARGET.x, s.y - TARGET.y) < 0.22;
            if (hit) onPrompt++;
            g.fillStyle = hit ? 'rgba(56,217,169,0.9)' : 'rgba(124,156,255,0.8)';
            g.beginPath(); g.arc(sx(s.x), sy(s.y), 3, 0, 7); g.fill();
          });
          g.restore();
          /* the prompt label goes on last, over an opaque patch, so stray samples
             behind it can never sit inside the lettering */
          g.font = MONO;
          const plab = '"a cat in sunglasses"', plx = sx(TARGET.x) - 20, ply = sy(TARGET.y) - 34;
          g.fillStyle = '#0a0e16'; g.fillRect(plx - 4, ply - 11, g.measureText(plab).width + 8, 15);
          g.fillStyle = C.green; g.fillText(plab, plx, ply);

          /* variety = mean pairwise spread */
          let mx = 0, my = 0; S.forEach(s => { mx += s.x / n; my += s.y / n; });
          let spread = 0; S.forEach(s => { spread += Math.hypot(s.x - mx, s.y - my) / n; });

          const TX = 420;
          const bar = (lab, v, col, note, y) => {
            g.font = FONT; g.fillStyle = C.muted; g.fillText(lab, TX, y);
            g.fillStyle = C.line; g.fillRect(TX, y + 8, 230, 14);
            g.fillStyle = col; g.fillRect(TX, y + 8, ctx.clamp(v, 0, 1) * 230, 14);
            g.font = 'bold ' + FONT; g.fillStyle = col;
            g.fillText((v * 100).toFixed(0) + '%', TX + 240, y + 20);
            g.font = MONO; g.fillStyle = C.muted; g.fillText(note, TX, y + 38);
          };
          bar('follows the prompt', onPrompt / n, C.green, 'samples inside the green circle', 54);
          const variety = ctx.clamp(spread / 0.55, 0, 1);
          bar('variety', variety, C.warn, 'how spread out the batch is', 126);
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('guidance = ' + guide.toFixed(1), TX, 216);
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, guide < 1
            ? 'At zero guidance the model ignores your prompt and simply makes something it could plausibly make.'
            : guide > 11
              ? 'Cranked up, every image obeys — and every image is nearly the same image. This is why over-guided generations look stiff and repetitive.'
              : 'The usual working range. Most samples obey the prompt while the batch still contains genuinely different pictures.',
            TX, 238, 250, 17);
          ro.set({ guidance: guide.toFixed(1), 'on prompt': ((onPrompt / n) * 100).toFixed(0) + '%', variety: (variety * 100).toFixed(0) + '%' });
        });

        return ctx.figure(cv,
          'Classifier-free guidance, as a trade you can feel. The dashed ring is everything the model could plausibly generate; the green circle is the slice your prompt asked for. At guidance 0 the samples ignore you. Turn it up and they march toward the prompt — and squeeze together, because they are all being pushed toward the same place. That is the real cost: prompt adherence is bought with variety, which is why an over-guided batch comes back looking like eight copies of one picture.',
          [gSl, reseed, p0, p7, p15], ro);
      }
      /* ================================================================== */
      /* The chapter: touch first, read second.                             */
      /* ================================================================== */
      root.append(
        callout('tryit', '🖐 Do this first — try to find a picture by guessing',
          `Below is an 8 × 8 image in 16 shades of grey. Tiny. There are only 10<sup>77</sup> of them.<br>
           <b>1.</b> Press <b>Draw random pixels</b> a few times. Then press <b>Keep trying</b> and let it run.<br>
           <b>2.</b> Watch <b>best random draw</b>. It creeps up a few points and then stalls — and it stalls miles short of what a real picture scores. Give it a minute if you like.<br>
           <b>3.</b> Press <b>Show me a real one</b> and look at the difference in the score bar.`),
        needleLab(),
        p(`Random pixels are never a photograph. Not rarely — effectively never, at any size, for any amount of time you are willing to wait.`),
      );

      root.append(section('What generative modelling actually is',
        p(`Every model in the last six chapters answered a version of "what is this?" A photo goes in, a label comes out: cat, tumour, spam, the next word. That is <em>discriminative</em> modelling — draw a boundary over a fixed set of possibilities.`),
        p(`Enormously useful, and not what painted the portrait on your friend's book cover or dreamed up ten seconds of video from a sentence. Those answer a different question: <b>what does the space of all possible things-like-this look like, and can I draw a brand-new one from it?</b>`),
        p(`That is <em>generative</em> modelling, and it is the harder problem. A discriminative face classifier needs only a few cues separating "face" from "not face". A generative model of faces has to capture everything — pixel statistics, symmetric eye spacing, skin texture, how light falls — enough structure that feeding it pure randomness produces something real.`),
        callout('key', '🔑 The picture to hold on to',
          `Real images are points in a very high-dimensional space, and the real ones form a <b>tiny, oddly-shaped cloud</b> inside it. You just spent a minute failing to land in that cloud by accident.<br>
           Training data is a handful of samples from the cloud. A generative model is a function you feed randomness into that outputs new points landing <i>back inside</i> it — ideally without simply repeating a training example.`),
        p(`You already met one, in chapter 7, and it may not have looked like this. A language model predicting the next token does exactly the job: each step produces a distribution over what could plausibly come next, and sampling from it, token after token, draws a new sentence from the implicit distribution of "text that sounds like this".`),
        p(`That is an <em>autoregressive</em> generative model — one hard distribution over a whole sequence, factored into a chain of easy ones. GPT is not a different kind of thing from a face generator. It is a generative model whose pixels are words, produced one at a time.`),
        p(`Images resist that trick. There is no natural left-to-right order to a 2-D grid, and predicting a million pixels one at a time is brutally slow. The three families below are three answers to "how do we generate almost all at once instead?"`),
      ));

      root.append(section('Family 1: autoencoders, and the "V" that makes them work',
        p(`Start with a plain <em>autoencoder</em>. An <em>encoder</em> squeezes an image into a short list of numbers — the <em>latent</em> code — and a <em>decoder</em> expands it back. Train the pair so the output matches the input.`),
        p(`Do that well and the decoder alone is a generator: feed it any latent code, get a picture. Except it usually draws garbage.`),
        callout('tryit', '🖐 Try this — find the dead space yourself',
          `<b>1.</b> You start in plain-autoencoder mode. Press <b>Sample fresh randomness</b> about ten times and count how often the output is coherent. It will be rare.<br>
           <b>2.</b> Drag the ring slowly between two coloured islands. Watch the output fall apart in the gap — the decoder was never asked to explain that region, so it has nothing sensible to say.<br>
           <b>3.</b> Switch to <b>variational autoencoder</b> and repeat step 1. The islands have merged into one blob and the dead space has all but closed — about one sample in twelve, against five in six.`),
        vaeLab(),
        p(`Kingma and Welling's 2013 fix adds one idea. Encode each image not to one exact point but to a small Gaussian cloud — a mean and a spread — and train the decoder to work for a <i>sample</i> from that cloud, not its exact centre. A second loss term pulls every cloud toward a shared standard normal at the origin.`),
        p(`Two consequences follow. The latent space becomes <em>smooth</em>: nearby points decode to similar outputs, with no gaps. And because every cloud is pulled toward the same standard normal, at generation time you can sample fresh randomness from it directly, with no real image to start from.`),
        p(`VAEs are fast — one decoder pass — and give an honest, well-behaved latent space, which is why they still sit inside larger systems: Stable Diffusion runs its diffusion process inside a VAE's latent space rather than on raw pixels.`),
        p(`Their weakness is smoothness's flip side. Outputs are often a little blurry, because the loss rewards averaging several plausible answers rather than committing to one sharp one.`),
      ));

      root.append(section('Family 2: two networks in a fight',
        p(`Ian Goodfellow's 2014 idea drops the encoder and replaces the loss function with a second neural network. A <em>generator</em> turns random noise into a fake. A <em>discriminator</em> looks at an image — sometimes real, sometimes fake — and guesses which.`),
        p(`The generator is trained to fool it; the discriminator is trained to catch it. Train both at once and, at the contest's fixed point, the fakes are indistinguishable from real data — at least to the best detective the discriminator managed to become.`),
        p(`No loss here rewards blurry averaging. The discriminator only lets through fakes sharp and specific enough to pass, which is why working GANs look crisper than VAEs.`),
        callout('tryit', '🖐 Try this — watch a GAN fail in its signature way',
          `The target is a two-humped distribution. The generator is, by construction, a single Gaussian bump, so it <b>cannot</b> cover both humps at once. It has to choose how to lose.<br>
           <b>1.</b> Press <b>▶ Train</b> and watch it spread thin across both humps, fitting neither well.<br>
           <b>2.</b> Now press <b>Mode collapse: off</b> to switch it on, then <b>Reset generator</b> and train again. It abandons one hump entirely and sits happily on the other.<br>
           <b>3.</b> That second behaviour is the failure with a name — and notice it is not stupidity. The generator found a trick that beats the current discriminator, and nothing rewards it for exploring further.`),
        ganDynamics(),
        p(`<em>Mode collapse</em> is the distinctive GAN failure: the generator finds one narrow trick that fools the current discriminator — say, always the same face — and has no reason to look further, since that trick is already winning. The discriminator eventually catches on, the generator lurches to a different trick, and the two chase each other in circles instead of covering the real distribution.`),
        callout('example', '🌍 GANs made this real: StyleGAN and deepfakes',
          `NVIDIA's <b>StyleGAN</b> (2018) and its successors produced the photorealistic invented faces behind <i>thispersondoesnotexist.com</i> — a site that, more than any paper, made the general public understand that a picture of a person is no longer evidence that the person exists.
           Face-swap video came from the other family: the original 2017 deepfake code paired two autoencoders sharing one encoder, not a GAN. GANs are still used where speed matters more than variety: real-time upscaling, some super-resolution, and voice conversion that has to run live.`),
      ));

      root.append(section('Family 3: ruin it on purpose, then learn to undo that',
        p(`This idea now powers most image, video and audio generators, and it starts almost silly. Take a real image and ruin it deliberately, a tiny bit at a time. Add a whisper of noise to every pixel; add a bit more to the result; repeat a thousand times, and you are left with pure static.`),
        p(`That <em>forward process</em> involves no learning at all. It is a fixed, known recipe, described by Jascha Sohl-Dickstein and colleagues in 2015, borrowed from non-equilibrium thermodynamics.`),
        p(`The trick is what you do with it: train a network to <b>undo one small step</b>. Show it a slightly-noised image and ask it to predict exactly what noise was added. Plain regression, ordinary squared-error loss, nothing adversarial.`),
        callout('tryit', '🖐 Try this: destroy it, then watch it get born from noise',
          `<b>1.</b> Drag <b>t</b> slowly from 0 toward 1 and watch the shape dissolve into static. Note that <b>no learning happened</b> in that direction — it is just a recipe.<br>
           <b>2.</b> Now press <b>▶ Play reverse</b> and watch the shape come back out of the noise. Each step asks "what noise is in here?", subtracts a piece of the answer, and repeats.<br>
           <b>3.</b> Press <b>Step once</b> a few times to see individual guesses.<br>
           <b>4.</b> Drag <b>reverse steps</b> down to 10 and replay — the result is rough. Push it to 150 and replay — it sharpens. That slider is the speed-versus-quality dial in every image generator you have used.`),
        diffusionCloud(),
        p(`Do that well at every noise level and you can generate something new by running the recipe backwards: start from pure noise, ask what noise is in there, subtract a piece of the answer, repeat. A thousand small, confident guesses turn static into a photograph.`),
        p(`Ho, Jain and Abbeel's 2020 paper, "Denoising Diffusion Probabilistic Models", found the recipe that made this reliably work and set off the modern wave.`),
        callout('key', '🔑 Why diffusion beat GANs',
          `Training is regression toward a known target. There is <b>no adversarial game to destabilise</b>, and every training image contributes a clean gradient at every noise level — so there is no shortcut equivalent to mode collapse.<br>
           The cost is speed: many small steps instead of one pass. Better samplers brought that to 20–50, and distilled models (SDXL-Turbo, FLUX.1-schnell) to between one and four — at some cost in variety.`),
      ));

      root.append(section('Turning it into "a cat wearing sunglasses"',
        p(`Two more pieces. <em>Text conditioning</em>: a CLIP-style model (Radford and colleagues, 2021) pulls matching images and captions to nearby points in embedding space — chapter 6's idea, applied across two media. The denoising network then attends to that whole row of per-word vectors — 77 of them in Stable Diffusion's text encoder — inside blocks at every scale of the network, via <em>cross-attention</em>, so each image patch can ask "which words describe me?"`),
        p(`<em>Classifier-free guidance</em>: train the same network with the caption randomly blanked out sometimes. At generation time you run it twice per step — with and without the prompt — and push the prediction further in the direction the prompt adds. No separate classifier network is needed, hence the name.`),
        callout('tryit', '🖐 Try this — the dial you have already used without knowing',
          `<b>1.</b> Press <b>no guidance (0)</b>. The samples scatter across everything the model could make and mostly ignore the prompt.<br>
           <b>2.</b> Press <b>typical (7.5)</b>. Most of the batch now lands on the prompt — press <b>New batch</b> a few times and the figure moves around, but it is always well short of everything — and the samples are still clearly spread out.<br>
           <b>3.</b> Press <b>cranked (15)</b>. Read <i>both</i> bars. Prompt adherence goes up, <b>variety collapses</b> — which is exactly why an over-guided batch comes back looking like eight copies of one picture.`),
        guidanceLab(),
        p(`The other 2022 breakthrough was <em>latent diffusion</em> (Rombach et al., the model behind Stable Diffusion): run the whole process on a VAE-compressed grid roughly 8× smaller per side. Stable Diffusion's version takes 512×512×3 = 786,432 numbers down to 64×64×4 = 16,384 — 48× fewer values to denoise at every one of the steps, which is the reason it fits on a consumer GPU.`),
        p(`DALL·E 2 arrived in April 2022, pairing a CLIP embedding with a diffusion decoder. Stable Diffusion followed four months later and released its weights publicly — a big reason diffusion, not GANs, became what the world built on.`),
        callout('history', '📜 Nine years, five papers, one obsession',
          `<b>2013:</b> Kingma and Welling's variational autoencoder makes latent-variable generation trainable by gradient descent.<br>
           <b>2014:</b> Goodfellow's GAN replaces the loss function with an opponent, and dominates image generation for six years.<br>
           <b>2015:</b> Sohl-Dickstein's diffusion paper appears and is largely ignored — the sampling was far too slow to be practical.<br>
           <b>2020:</b> Ho, Jain and Abbeel's DDPM finds the training recipe that works, and diffusion becomes competitive almost overnight.<br>
           <b>2022:</b> latent diffusion makes it cheap, CLIP conditioning makes it controllable, and the technology leaves the lab.`),
      ));

      root.append(section('Latent space, and why you can walk through it',
        p(`VAEs and GANs both funnel randomness through a compact vector — the <em>latent space</em> — that a decoder expands into a full image. When training worked, that space is continuous and meaningful: nearby points decode to similar outputs.`),
        p(`Better still, directions often correspond to recognisable features — more smile, more tilt — that nobody labelled. They emerged from the data's geometry, exactly as the gender direction did in chapter 6.`),
        callout('tryit', '🖐 Try this: drag, save two points, then interpolate',
          `<b>1.</b> Drag the blue dot around the left pad and watch the face respond continuously. There are no jumps, anywhere.<br>
           <b>2.</b> <b>Set A</b> somewhere you like, move away, <b>Set B</b> somewhere quite different.<br>
           <b>3.</b> Drag the interpolation slider slowly. Every intermediate point is a plausible face — not a cross-fade between two pictures, but a genuine third face the model invented on the way.`),
        latentWalk(),
        p(`That is why two points blend into something sensible in between rather than a ghostly double-exposure. You are not mixing images. You are walking a path through the space of things the model believes are possible.`),
      ));

      root.append(section('Why this matters for modern AI',
        p(`Nothing here is specific to still images. Stretch the grid to include time and you diffuse over a block of video frames: OpenAI's Sora, announced February 2024 with demos up to a minute long, works this way, and Google's Veo followed later the same year. Shipped versions have generally offered far shorter clips than the announcements — video is where the compute bill bites hardest.`),
        p(`Turn a sound wave into the "pixels" and the same recipe generates music (Suno, Udio) or clones a voice from a short clip. Only what counts as "the data" changes.`),
        p(`One 2022 idea that only reached production models in 2024, used in Stable Diffusion 3 among others: <em>flow matching</em>. Instead of diffusion's specific noise schedule, it trains a network to predict a <em>velocity</em> — which way and how fast to move a point along a nearly straight path from noise to data. A more direct generalisation of the same idea, and it typically needs fewer steps.`),
        p(`Every frontier system in Part III and beyond is, underneath, a generative model. An LLM is an autoregressive model of text. A text-to-image or text-to-video system is usually a latent diffusion or flow-matching model of pixels — though not always: OpenAI's native image generation in GPT-4o produces images autoregressively, token by token, like text. Increasingly they generate more than one kind of data at once.`),
        p(`These three families are not historical footnotes that transformers replaced. They are the toolbox transformers got combined <i>with</i> — a diffusion model's denoiser is very often a transformer internally.`),
        callout('warning', '⚠️ The uncomfortable part: consent, credit and provenance',
          `These models learn from enormous scrapes of the public internet, which include the work of living artists, photographers and writers who were not asked and are not paid.
           That is an unresolved legal and ethical question, not a settled one, and it is being argued in courtrooms as you read this.<br>
           Separately, the same technology makes convincing fake images of real people cheap and instant. Provenance standards such as C2PA content credentials, watermarking, and detection tools all exist and all have real limits —
           watermarks can be stripped, and detectors fail exactly when the generator is good. Treat "there is a photo of it" as much weaker evidence than it was ten years ago.`),
        p(`One picture to keep: <b>"learn a distribution well enough to sample new, plausible members of it" is the same idea underneath image generation, protein design, and a language model producing a plan of action it was never shown.</b>`),
      ));

      root.append(
        ctx.quiz([
          { q: 'You drew thousands of random 8×8 images and never once got a picture. What does that demonstrate?', options: ['The random number generator was broken', 'Real images occupy a vanishingly tiny region of the space of all possible images, so you cannot reach them by chance', '8×8 is too small to show a picture', 'You needed more shades of grey'], answer: 1, explain: 'Even at 8×8 in 16 greys there are about 10⁷⁷ possibilities and essentially none are pictures. That is the whole problem a generative model solves: landing inside the cloud deliberately, on the first try.' },
          { q: 'Why does a plain autoencoder make a bad generator, and what does the VAE add?', options: ['It is too slow; the VAE is faster', 'Nothing forces the space between training codes to mean anything, so fresh randomness lands in dead space; the VAE encodes to clouds and pulls them all toward one standard normal, closing the gaps', 'It has no decoder', 'The VAE adds a discriminator'], answer: 1, explain: 'You found the dead space by dragging between the islands. A generator must be fed fresh randomness, so the latent space has to be meaningful everywhere the randomness might land — not only at the points that happened to appear in training.' },
          { q: 'What is mode collapse?', options: ['The discriminator stops learning', 'The generator finds one narrow output that fools the current discriminator and stops exploring, so it covers only part of the real distribution', 'The model runs out of memory', 'Training loss goes to zero'], answer: 1, explain: 'It is not stupidity — the trick is already winning, and nothing rewards the generator for covering the rest. This is the failure mode diffusion does not have, because its training is plain regression against a known target with no opponent to exploit.' },
          { q: 'In diffusion, which direction involves learning?', options: ['Both directions equally', 'Only the forward noising process', 'Only the reverse: a network is trained to predict what noise was added, while the forward noising is a fixed recipe', 'Neither — diffusion is not learned'], answer: 2, explain: 'Adding noise is a known recipe requiring no training at all. The network learns only to undo one small step, which is ordinary regression with a squared-error loss — and that is exactly why diffusion training is so much more stable than a GAN\'s adversarial game.' },
          { q: 'You turn the guidance scale from 7.5 up to 15. What do you gain and what do you lose?', options: ['Better quality at no cost', 'Stronger prompt adherence, but less variety — every sample is pushed toward the same region, so the batch looks repetitive', 'More variety and weaker prompts', 'Faster generation'], answer: 1, explain: 'You can read both bars moving in opposite directions in the guidance demo. It is a genuine trade rather than a quality dial, which is why over-guided generations come back looking stiff and near-identical.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/" target="_blank" rel="noopener">Lilian Weng, "What are Diffusion Models?"</a>: the clearest maths-forward walkthrough, and the standard reference for people implementing one.`,
            `<a href="https://arxiv.org/abs/2006.11239" target="_blank" rel="noopener">Ho, Jain &amp; Abbeel (2020), "Denoising Diffusion Probabilistic Models"</a>: the paper that made diffusion practical.`,
            `<a href="https://arxiv.org/abs/2112.10752" target="_blank" rel="noopener">Rombach et al. (2022), "High-Resolution Image Synthesis with Latent Diffusion Models"</a>: Stable Diffusion, and the compression that put it on consumer hardware.`,
            `<a href="https://arxiv.org/abs/1406.2661" target="_blank" rel="noopener">Goodfellow et al. (2014), "Generative Adversarial Networks"</a>: the original two-network fight.`,
            `<a href="https://arxiv.org/abs/1312.6114" target="_blank" rel="noopener">Kingma &amp; Welling (2013), "Auto-Encoding Variational Bayes"</a>: the VAE, and where the second loss term comes from.`,
          ]),
        ),
      );
    },
  });
})();
