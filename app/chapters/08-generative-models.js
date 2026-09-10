/* Zero → AGI · Chapter 08 · Generative models: GANs, VAEs and diffusion
   Discriminative vs generative; modelling a distribution; autoregression is generative too.
   VAEs (compress → latent → decode, smoothed by noise); GANs (forger vs detective, mode collapse);
   diffusion (destroy with noise, learn to reverse it; text conditioning; classifier-free guidance;
   why diffusion beat GANs; latent diffusion); video/audio as the same idea; flow matching.
   Interactives: (a) diffusion on a 2-D point cloud, forward VP-SDE schedule + genuine reverse
   denoising via Tweedie's formula on the empirical data distribution; (b) 1-D GAN dynamics with an
   analytically-optimal discriminator and a real (finite-difference) generator gradient, plus a
   mode-collapse toggle; (c) a 2-D latent pad decoded into a procedural face, with an interpolation
   slider between two saved points. */
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
          for (const q of pts) {
            const sx = cx + q.x * scale, sy = cy - q.y * scale;
            g.beginPath(); g.arc(sx, sy, 2.6, 0, Math.PI * 2);
            g.fillStyle = q.part === 'outline' ? C.accent : q.part === 'eye' ? C.warn : C.pink;
            g.globalAlpha = 0.88; g.fill();
          }
          g.globalAlpha = 1;
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
        const stepsSlider = ctx.slider({ label: 'reverse steps', min: 10, max: 150, step: 5, value: 40, onChange: (v) => { R.steps = v; } });
        const playBtn = ctx.button('▶ Play reverse', () => {
          if (mode !== 'reverse' || R.k >= R.steps) { initReverse(); mode = 'reverse'; }
          R.playing = !R.playing; playBtn.textContent = R.playing ? '⏸ Pause' : '▶ Play reverse';
        }, 'primary');
        const stepBtn = ctx.button('Step once', () => {
          if (mode !== 'reverse') { initReverse(); mode = 'reverse'; }
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
            const pd = pdata(x), pg = gauss(x, mu, sigma);
            const D = pd / (pd + pg + 1e-9);
            s += Math.log(D + 1e-9);
          }
          s /= zs.length;
          if (!G.collapse) s += 0.35 * Math.log(Math.max(sigma, 1e-3)); // entropy bonus: a real anti-collapse trick
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
          nsig = ctx.clamp(nsig, 0.12, 2.6); nmu = ctx.clamp(nmu, -4.5, 4.5);
          if (!isFinite(nmu) || !isFinite(nsig)) { G = defaults(); return; }
          G.mu = nmu; G.sigma = nsig; G.iter++;
        }

        const plot = { x: 46, y: 14, w: 660, h: 250 };
        const xAt = (x) => plot.x + (x - X0) / (X1 - X0) * plot.w;
        function draw() {
          g.clearRect(0, 0, W, H);
          g.fillStyle = '#0f1520'; g.fillRect(plot.x, plot.y, plot.w, plot.h); g.strokeStyle = C.line; g.strokeRect(plot.x, plot.y, plot.w, plot.h);
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
            const x = X0 + (X1 - X0) * i / 200, pd = pdata(x), pg = gauss(x, G.mu, G.sigma);
            const D = pd / (pd + pg + 1e-9), yy = plot.y + plot.h - D * plot.h * 0.9;
            i === 0 ? g.moveTo(xAt(x), yy) : g.lineTo(xAt(x), yy);
          }
          g.stroke(); g.setLineDash([]);
          // a handful of current generator samples along the baseline
          g.fillStyle = C.danger;
          for (let i = 0; i < 28; i++) { const x = G.mu + G.sigma * ctx.randn(); g.beginPath(); g.arc(xAt(ctx.clamp(x, X0, X1)), plot.y + plot.h - 4, 2.4, 0, Math.PI * 2); g.fill(); }
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          g.fillText('blue bars = real data   red = generator’s density   dashed yellow = discriminator score D(x)', plot.x, plot.y - 2);
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
          `The generator is a single movable Gaussian (red). The discriminator is not a trained network here — it is the exact optimal judge from Goodfellow's proof, D*(x) = p<sub>data</sub>(x) / (p<sub>data</sub>(x) + p<sub>gen</sub>(x)) (dashed yellow), so you are watching the generator's side of the game with a perfect opponent. Its mean and spread are updated every iteration by a genuine gradient step (computed by finite differences) on "fool the judge". A single Gaussian can never truly cover two separate bumps at once, which is exactly why this toy is useful: watch it either spread out to straddle both modes, or — with the toggle on — shrink and lock onto just one, ignoring the other entirely.`,
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
          const mark = (pt, color, label) => {
            const px = padCenter.x + pt.x * padScale, py = padCenter.y - pt.y * padScale;
            g1.beginPath(); g1.arc(px, py, 5, 0, Math.PI * 2); g1.strokeStyle = color; g1.lineWidth = 2; g1.stroke();
            g1.fillStyle = color; g1.font = FONT; g1.textAlign = 'center'; g1.fillText(label, px, py - 10);
          };
          mark(A, C.green, 'A'); mark(B, C.orange, 'B');
          const zx = padCenter.x + z.x * padScale, zy = padCenter.y - z.y * padScale;
          g1.beginPath(); g1.arc(zx, zy, 7, 0, Math.PI * 2); g1.fillStyle = C.accent; g1.fill(); g1.strokeStyle = '#0a0e16'; g1.lineWidth = 1.5; g1.stroke();
          g1.fillStyle = C.muted; g1.font = FONT; g1.textAlign = 'left';
          g1.fillText('z1 →', 6, PAD - 6); g1.save(); g1.translate(12, 16); g1.rotate(-Math.PI / 2); g1.fillText('z2 →', 0, 0); g1.restore();
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
      /* Prose                                                               */
      /* ================================================================== */
      root.append(
        p(`Every model in the last six chapters answered a version of the question "what is this?" A photo goes in, a label comes out: cat, tumour, spam, the next word. That is <em>discriminative</em> modelling — draw a boundary, or estimate a probability, over a fixed set of possibilities. It is enormously useful and it is not what produced the portrait Midjourney painted for your friend's book cover, the voice that reads your audiobook in a stranger's cadence, or the ten seconds of video Sora dreamed up from a sentence. Those came from a different question: not "what is this?" but "what does the space of all possible things-like-this look like, and can I draw a brand-new one from it?"`),
        p(`That is <em>generative</em> modelling, and it is the harder problem. A discriminative face classifier only needs a few cues that separate "face" from "not face". A generative model of faces has to capture everything — pixel statistics, symmetric eye spacing, skin texture, how light falls — enough structure that feeding it pure randomness produces something that looks real.`),
        p(`This chapter covers three ways people built that: <em>variational autoencoders</em> (2013), <em>generative adversarial networks</em> (2014), and <em>diffusion models</em> (2015, practical by 2020) — the technique behind essentially every image, video and voice generator since 2022. All three solve the same problem and disagree, sometimes fiercely, about how.`),

        section('Modelling a distribution, and why GPT already does it',
          p(`Stated precisely: real images (or sentences, or molecules) are points in a very high-dimensional space, and the real ones form a tiny, oddly-shaped cloud inside it — almost every arrangement of pixels is noise, not a photo. Training data is samples from that cloud. A generative model's job is a function you feed randomness into that outputs new points landing back inside the cloud, ideally without repeating a training example.`),
          p(`You already met a generative model, in the transformer chapter, and it may not have looked like one. A language model predicting the next token, over and over, does exactly this: each step outputs a probability distribution over "what could plausibly come next," and sampling from it, token after token, draws a new sentence from the implicit distribution of "text that sounds like this". That is an <em>autoregressive</em> generative model — one hard distribution over a whole sequence, factored into a chain of easy ones. GPT is not a different kind of thing from a face generator; it is a generative model whose "pixels" are words, produced one at a time.`),
          p(`Images resist that trick: there is no natural left-to-right order to a 2-D grid of pixels, and predicting a million pixels one at a time is brutally slow. VAEs, GANs and diffusion are three answers to "how do we generate almost all at once instead?"`),
        ),

        section('VAEs: compress, then decode from a smoothed-out map',
          p(`A plain autoencoder: an <em>encoder</em> squeezes an image into a short list of numbers (the <em>latent</em> code), a <em>decoder</em> expands it back, and you train the pair so the output matches the input. Do this well and the decoder alone is a generator — feed it any latent code, get a picture. Except it usually draws garbage: nothing promises that the space between codes seen in training means anything. There can be gaps the decoder was never asked to explain.`),
          p(`Diederik Kingma and Max Welling's 2013 fix, the <em>variational autoencoder</em> (VAE), adds one idea: encode each image not to one exact point but to a small Gaussian cloud (a mean and a spread), and train the decoder to work for a <em>sample</em> from that cloud, not its exact centre. A second loss term (KL divergence) pulls every cloud toward a shared standard normal at the origin. Two consequences follow. The latent space becomes <em>smooth</em> — nearby points decode to similar outputs, with no gaps, exactly what interactive (c) below lets you feel. And because every cloud is pulled toward the same standard normal, at generation time you can sample fresh randomness from it directly — no real image needed to start from.`),
          p(`VAEs are fast (one decoder pass) and give an honest, well-behaved latent space — which is why they still turn up inside larger systems: Stable Diffusion runs its diffusion process inside a VAE's latent space rather than on raw pixels. Their weakness is smoothness's flip side: outputs are often a little blurry, since the loss rewards averaging several plausible answers over committing to one sharp one.`),
        ),

        section('GANs: a forger and a detective, locked in a contest',
          p(`Ian Goodfellow's 2014 idea drops the encoder and replaces the loss function with a second neural network. A <em>generator</em> takes random noise and paints a fake. A <em>discriminator</em> looks at an image — sometimes real, sometimes the generator's fake — and guesses which. The generator is trained to fool it; the discriminator is trained to catch it. Train both at once and, at the contest's fixed point, the generator's fakes are indistinguishable from real data, at least to the best detective the discriminator became.`),
          p(`No loss here rewards blurry averaging — the discriminator only lets through fakes sharp and specific enough to pass, which is why working GANs look crisper than VAEs. When they fail, it is in a distinctive way called <em>mode collapse</em>: the generator finds one narrow trick that fools the current discriminator (say, always the same face) and has no reason to explore further, since that trick is already winning. The discriminator eventually catches it, the generator lurches to a different trick, and the two can chase each other in circles instead of covering the real distribution. Interactive (b) shows a toy version: a generator that is, by construction, a single Gaussian bump can never truly cover a two-humped target — watch it either spread thin across both humps or, with a toggle, collapse onto just one.`),
          callout('example', 'GANs made this real: StyleGAN and deepfakes', `NVIDIA's StyleGAN (2018) and StyleGAN2 (2019) generated faces convincing enough that "This Person Does Not Exist" fooled people for years — every face on it belongs to nobody. The same forger-vs-detective idea, pointed at footage of a real person, is the engine behind <em>deepfakes</em>: swap the identity being reproduced and you get face-swapped video good enough to cause real harm, which is why watermarking and detection (below) became urgent almost as soon as GANs got good.`),
        ),

        section('Diffusion: destroy carefully, then learn to undo it',
          p(`This idea now powers most image, video and audio generators, and it starts almost silly: take a real image and ruin it on purpose, a tiny bit at a time. Add a whisper of noise to every pixel; add a bit more to the result; repeat a thousand times, and you are left with pure static — every trace of the picture gone. This <em>forward process</em> involves no learning; it is a fixed, known recipe. Jascha Sohl-Dickstein and colleagues described it, borrowed from non-equilibrium thermodynamics, in 2015.`),
          p(`The trick is what you do with that recipe: train a network to <em>undo one small step</em>. Show it a slightly-noised image and ask it to predict exactly what noise was added — plain regression, an ordinary squared-error loss, nothing adversarial. Do that well at every noise level and you can generate a new image by running the recipe backwards: start from pure noise, ask "what noise is in here?", subtract a piece of the answer, repeat. A thousand small, confident guesses turn static into a photograph. Ho, Jain and Abbeel's 2020 paper, "Denoising Diffusion Probabilistic Models" (DDPM), found the recipe that made this reliably work and touched off the modern wave.`),
          callout('key', 'Worked example: guessing the clean point from the noisy one', `The reverse step needs one computation: given a noisy point x<sub>t</sub>, what clean point x<sub>0</sub> produced it? For a discrete set of training points, the exact answer (<em>Tweedie's formula</em>) is a weighted average of every training point, weighted by how likely a Gaussian bump centred there was to produce x<sub>t</sub>. Say the "dataset" is two points, x<sub>a</sub> = −2 and x<sub>b</sub> = +2, the noise level gives ᾱ = 0.5 (signal scaled by √0.5 ≈ 0.707, noise variance 0.5), and you observe x<sub>t</sub> = 1.3. Distance-squared to a noised x<sub>a</sub>: (1.3 + 1.414)² = 7.37. To a noised x<sub>b</sub>: (1.3 − 1.414)² = 0.013. Softmax those and the gap is so large x<sub>b</sub> gets almost all the weight: ≈0.06% on x<sub>a</sub>, ≈99.94% on x<sub>b</sub>, posterior mean ≈ 1.997. The model is nearly certain x<sub>t</sub> came from +2. Interactive (a) below runs exactly this, in two dimensions, over all 300 points of a smiley — a real model just replaces the lookup with a neural network's learned approximation of the same weighted average.`),
          p(`Two more pieces turn this into "a cat wearing sunglasses". <em>Text conditioning</em>: a CLIP-style model (Alec Radford and colleagues, OpenAI, 2021) pulls matching images and captions to nearby embedding-space points; the denoising network attends to that prompt vector at every layer via <em>cross-attention</em>, so each image patch can ask "which words describe me?" <em>Classifier-free guidance</em>: train the same network with the caption randomly blanked out sometimes, so at generation time you run it twice per step — with and without the prompt — and push the prediction further in the direction the prompt adds. Turn that guidance scale up for more literal prompt-following, at some cost to variety. No separate classifier network is needed, hence the name.`),
          p(`Why did diffusion overtake GANs by 2022? Training is regression toward a known target — no adversarial game to destabilise, and every training image contributes a clean gradient at every step, so there is no shortcut equivalent to mode collapse. The cost is speed (many small steps instead of one pass), though modern samplers need as few as 20–50. The other 2022 breakthrough, <em>latent diffusion</em> (Rombach et al., the model behind Stable Diffusion), runs the whole process on a VAE-compressed grid roughly 8× smaller per side — about 64× cheaper, and why it fits on a consumer GPU. DALL·E 2 (April 2022) paired a CLIP embedding with a diffusion decoder; Stable Diffusion followed four months later and released its weights publicly — a big reason diffusion, not GANs, became what the world built on.`),
          callout('tryit', 'Try it: destroy a smiley, then watch it get born from noise', `Drag the <b>t</b> slider from 0 to 1 and watch the smiley dissolve into a formless cloud — colours (outline, eyes, mouth) blur together as structure disappears. Then press <b>▶ Play reverse</b> and watch fresh random noise get rebuilt into a face using nothing but the weighted-average rule above, applied 40-plus times. Drag <b>reverse steps</b> down to 10: faster, rougher. Push it to 150: slower, cleaner — the exact trade-off every image generator's "steps" setting controls.`),
          diffusionCloud(),
        ),

        section('The same trick for video and sound, and a 2024–2025 refinement',
          p(`Nothing here is specific to still images. Stretch the grid to include time and you diffuse over a block of video frames: OpenAI's Sora (announced February 2024) generates up to roughly a minute this way; Google's Veo followed later the same year. Turn a sound wave into the "pixels" and the same recipe generates music (Suno, Udio) or clones a voice from a short clip. Only what counts as "the data" changes.`),
          p(`One 2024–2025 refinement worth knowing, used in systems including Stable Diffusion 3: <em>flow matching</em>. Instead of diffusion's specific noise schedule, it directly trains a network to predict a <em>velocity</em> — which way and how fast to move a point along a close-to-straight path from noise to data. A more direct generalisation of the same idea, and one that typically needs fewer steps to sample well.`),
        ),

        section('Latent space: the shared idea underneath VAEs and GANs',
          p(`VAEs and GANs both funnel randomness through a compact vector — the <em>latent space</em> — that a decoder or generator expands into a full image. When training worked, that space is continuous and meaningful: nearby points decode to similar outputs, and directions often correspond to recognisable features (more smile, more tilt) nobody labelled — they emerged from the data's geometry. That is why you can walk through latent space and watch one image morph into another, and why two points blend into something sensible in between rather than a mash-up.`),
          callout('tryit', 'Try it: drag, save two points, then interpolate', `Drag the dot on the latent pad and watch the face change continuously — no jumps, because the decode function is smooth in both inputs. Pick a face you like and press <b>Set A</b>; pick a different one and press <b>Set B</b>. Drag the <b>interpolate</b> slider from 0 to 1: a straight line through latent space gets decoded point by point, and the face morphs gradually rather than snapping. This is the demo every VAE and GAN paper since 2014 has used to argue "our latent space actually learned something."`),
          latentWalk(),
        ),

        callout('history', 'Six years, three papers, one obsession', `Kingma and Welling posted the VAE (December 2013) to make variational inference tractable with backpropagation — a technical motivation with a delightful side effect: a generator you could sample from. Goodfellow's GAN (NeurIPS 2014) was greeted as one of the most exciting ideas in machine learning that decade; quality and stability improved for years (StyleGAN, 2018, was arguably the high-water mark). Sohl-Dickstein's 2015 diffusion paper landed quietly — elegant but too slow and blurry to compete with GANs. It took Ho, Jain and Abbeel's 2020 DDPM paper, five years later, to show diffusion could match and then exceed GAN quality. The field pivoted hard: DALL·E 2 and Stable Diffusion both shipped in 2022, and by 2023 diffusion was the default for nearly every new image, video and audio generator.`),

        callout('example', 'Where you have already seen this', `Midjourney and Stable Diffusion turn prompts into illustrations for book covers and concept art. E-commerce tools generate clean product photos from a single snapshot, skipping a studio shoot. Further afield: RFdiffusion and similar tools (including David Baker's lab at the University of Washington) apply the same noise-and-denoise idea to 3-D atomic coordinates instead of pixels, generating entirely new protein structures — a genuinely new tool for drug and molecule design.`),

        callout('warning', 'The uncomfortable part: consent, credit and provenance', `Every model here is trained on enormous piles of real images, text and voices, usually scraped without asking the people who made them — why artists and Getty Images have sued image-generator makers, and why the law is still unsettled. The same technology that makes a birthday card in your art style can put words a real person never said into their mouth on video: non-consensual deepfakes are already used for fraud, harassment and disinformation, and realism is outrunning most people's instinct for what to trust. The partial technical response is <em>watermarking</em> (Google DeepMind's SynthID) and provenance standards like C2PA "Content Credentials" — neither foolproof, since watermarks can be stripped and industry adoption is still partial.`),

        section('Why this matters for modern AI',
          p(`Every frontier system in Part III and beyond is, underneath, a generative model: an LLM is an autoregressive model of text; a text-to-image or text-to-video system is a latent, diffusion- or flow-matching-based model of pixels; and systems increasingly generate more than one kind of data at once. These three families are not historical footnotes transformers replaced — they are the toolbox transformers got combined <i>with</i> (a diffusion model's denoiser is very often a transformer internally) to reach where things stand today. "Learn a distribution well enough to sample new, plausible members of it" is the same idea underneath image generation, protein design, and a language model producing a plan of action it was never explicitly shown.`),
        ),

        ctx.quiz([
          { q: 'What can a generative model do that a purely discriminative classifier cannot?', options: ['Compute a loss during training', 'Draw a decision boundary between two classes', 'Produce a brand-new example that plausibly belongs to the training distribution', 'Use backpropagation'], answer: 2, explain: 'A discriminative model only ever answers "which class is this existing input?" A generative model has learned enough about the whole distribution of the data to sample new points from it — that is the entire point of this chapter\'s three families.' },
          { q: 'In a VAE, why does the encoder produce a small cloud (mean and spread) around each input instead of one exact latent point?', options: ['It is faster to compute', 'It forces nearby latent points to decode to similar outputs, giving a smooth, gap-free latent space you can sample from', 'It increases the image resolution', 'It removes the need for a decoder'], answer: 1, explain: 'Training the decoder against jittered samples from that cloud, plus the KL term pulling every cloud toward a shared standard normal, is exactly what makes the latent space smooth and continuous enough to interpolate through, and samplable with fresh random noise at generation time.' },
          { q: 'What does "mode collapse" look like in the 1-D GAN demo?', options: ['The discriminator score becomes perfectly flat at 0.5 everywhere', 'The generator\'s single Gaussian shrinks and settles on just one of the two real bumps, ignoring the other entirely', 'Training loss reaches exactly zero and stops changing', 'The real-data histogram changes shape'], answer: 1, explain: 'A generator that has found one way to reliably fool the current discriminator has no built-in incentive to also cover the rest of the real distribution — it can lock onto a single mode and abandon the rest, which is precisely what the collapse toggle demonstrates.' },
          { q: 'In the point-cloud diffusion demo, how does the reverse process estimate the clean point behind a given noisy one?', options: ['It looks up the single nearest training point and copies it exactly', 'It takes a softmax-weighted average of every training point, weighted by how likely a Gaussian noise process centred there was to produce the observed noisy point', 'A separate discriminator network scores each candidate', 'It always returns the mean of the entire dataset'], answer: 1, explain: 'That weighted average is Tweedie\'s formula applied to the empirical data distribution — exactly what a real denoising network is trained to approximate, except with a learned function standing in for the explicit lookup over training points.' },
          { q: 'What does classifier-free guidance actually do at generation time?', options: ['Trains a separate image classifier to check the output afterward', 'Runs the denoising network twice per step (with and without the caption) and pushes the prediction further in the direction the caption adds', 'Skips several denoising steps to go faster', 'Increases the resolution of the output image'], answer: 1, explain: 'Because the same network was trained with the caption randomly blanked out, at sampling time you get both a conditioned and an unconditioned prediction "for free", and extrapolating away from the unconditional one steers generation more strongly toward the prompt — no extra classifier network required.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://arxiv.org/abs/1312.6114" target="_blank" rel="noopener">Kingma &amp; Welling (2013), "Auto-Encoding Variational Bayes"</a>: the original VAE paper.`,
            `<a href="https://arxiv.org/abs/1406.2661" target="_blank" rel="noopener">Goodfellow et al. (2014), "Generative Adversarial Networks"</a>: the four-page idea that launched a thousand follow-ups.`,
            `<a href="https://arxiv.org/abs/2006.11239" target="_blank" rel="noopener">Ho, Jain &amp; Abbeel (2020), "Denoising Diffusion Probabilistic Models"</a>: the paper that made diffusion practical.`,
            `<a href="https://arxiv.org/abs/2112.10752" target="_blank" rel="noopener">Rombach et al. (2022), "High-Resolution Image Synthesis with Latent Diffusion Models"</a>: the Stable Diffusion paper.`,
            `<a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/" target="_blank" rel="noopener">Lilian Weng, "What are Diffusion Models?"</a>: the clearest single write-up connecting the forward process, the training objective and sampling, with the full derivations this chapter simplified.`,
          ]),
        ),
      );
    },
  });
})();
