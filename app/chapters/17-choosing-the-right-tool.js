/* Zero → AGI · Chapter 17 · Choosing the right tool
   DESIGN RULE: the reader runs the head-to-head before reading a word of argument. A nine-knob
   logistic regression and a 161-knob neural net are really trained in the page, on seeded loan
   data, at whatever training-set size the reader picks; every number the prose quotes is printed
   by a demo in the state the try-it sends the reader to.
   The opener's first sample is the kindest of the twenty to the simple model, and the net there
   has no weight decay while logistic regression does; the prose says both, and the dropdown and
   the New-sample button let the reader check. Training is sliced across frames (see work()).
   Interactives, in order: the head-to-head and its learning curve (the opener); twenty training
   samples per model, to see spread and plateau (the bias–variance beat hangs off this one); a
   one-rule baseline the reader sets by hand against both trained models; one applicant scored by
   a model that can be read and one that cannot; a cost and latency calculator with a broken
   preset; an LLM-labels-then-distil demo where a real naive Bayes student is trained on a
   simulated teacher's labels; a decision guide scored by a points table the reader can see. */
(function () {
  ZTA.registerChapter({
    id: '17-choosing-the-right-tool',
    num: 17,
    part: 'VI',
    title: 'Choosing the right tool',
    tagline: 'Train a nine-knob model and a 161-knob neural net on the same loan data, and watch which one wins depend on how many rows you give them.',
    minutes: 50,
    objectives: [
      'predict, from the number of rows and the kind of data, whether a simple model or a neural net will score higher',
      'set a baseline before any model and say how much of the gain a model really added',
      'price a rule, a small model and an LLM call for a given volume and time limit',
      'pick a tool for a project from six facts about it, and say when to prototype with an LLM and distil',
    ],
    takeaways: [
      '<b>Few rows, few knobs.</b> With a few hundred rows a neural net left to itself fits the luck of the sample and loses to a simple model; restrained well, it only draws level. Flexibility pays when data is plentiful and the pattern has bends a simple model cannot draw.',
      '<b>Two reasons to miss.</b> Bias is the miss that more data never fixes; variance is the miss that changes with every sample and shrinks as rows arrive. Simple models trade a little of the first for much less of the second.',
      '<b>Beat the dumbest thing first.</b> Majority class, last value, one hand-written rule. A model is worth only what it adds on top of that.',
      '<b>If you must give reasons, use a model you can read.</b> A weighted sum states its reasons exactly. A probe of a black box is an estimate, and it changes from one customer to the next.',
      '<b>Multiply before you build.</b> Requests per day × cost per request, and the slowest step against the time allowed. Most wrong tool choices fail this sum.',
      '<b>LLMs earn their place on text, few labels and changing instructions.</b> When volume grows, let the LLM label a few thousand examples and train something small on them.',
    ],
    render(root, ctx) {
      const { h, p, section, callout, ul } = ctx;
      const C = ctx.colors;
      const FONT = '13px Inter, system-ui, sans-serif';
      const BOLD = 'bold 13px Inter, system-ui, sans-serif';
      const BIG = 'bold 22px Inter, system-ui, sans-serif';
      const MONO = '12px "JetBrains Mono", ui-monospace, monospace';

      /* ---------- seeded randomness: every quoted number is reproducible ---------- */
      function rng(seed) {
        let a = seed >>> 0;
        return () => { a += 0x6D2B79F5; let t = a; t = Math.imul(t ^ (t >>> 15), 1 | t); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      }
      function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
      const sig = (z) => 1 / (1 + Math.exp(-z));
      function wrapLines(gc, text, maxW) {
        const words = String(text).split(' '); const out = []; let line = '';
        for (const w of words) { const t = line ? line + ' ' + w : w; if (line && gc.measureText(t).width > maxW) { out.push(line); line = w; } else line = t; }
        if (line) out.push(line);
        return out;
      }
      const pct = (a) => (a * 100).toFixed(1) + '%';

      /* ---------- the loan table ----------
         Eight columns per applicant, stored as z-scores. Four carry signal, four are noise.
         The hidden truth is mostly a weighted sum, plus two bends a weighted sum cannot draw:
         debt matters more on a big loan, and risk is U-shaped in the number of cards. */
      const D = 8;
      const FEATS = [
        { name: 'years in current job', mu: 6, sd: 3, show: (v) => Math.max(0, v).toFixed(1) + ' years', dir: -1 },
        { name: 'existing debt', mu: 20, sd: 10, show: (v) => '£' + Math.max(0, v).toFixed(0) + 'k', dir: 1 },
        { name: 'loan size asked for', mu: 15, sd: 6, show: (v) => '£' + Math.max(1, v).toFixed(1) + 'k', dir: 1 },
        { name: 'credit cards held', mu: 3, sd: 1.5, show: (v) => Math.max(0, v).toFixed(1) + ' cards', dir: -1 },
        { name: 'day of month applied', mu: 15, sd: 6, show: (v) => 'day ' + Math.max(1, Math.min(31, v)).toFixed(0), dir: 1 },
        { name: 'letters in surname', mu: 7, sd: 2, show: (v) => Math.max(2, v).toFixed(0) + ' letters', dir: 1 },
        { name: 'hour of day applied', mu: 13, sd: 3, show: (v) => Math.max(0, Math.min(23, v)).toFixed(0) + ':00', dir: 1 },
        { name: 'house number', mu: 60, sd: 25, show: (v) => 'no. ' + Math.max(1, v).toFixed(0), dir: 1 },
      ];
      const trueLogit = (x) => 1.8 * (-1.1 * x[0] + 0.8 * x[1] + 0.5 * x[2] - 0.7 * x[3] + 0.7 * x[1] * x[2] + 0.4 * (x[3] * x[3] - 1) - 0.4);
      function makeData(n, seed) {
        const r = rng(seed), X = new Float64Array(n * D), y = new Uint8Array(n), x = new Array(D);
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < D; j++) { x[j] = gauss(r); X[i * D + j] = x[j]; }
          y[i] = r() < sig(trueLogit(x)) ? 1 : 0;
        }
        return { X, y, n };
      }
      /* logistic regression: 8 weights + 1 bias = 9 knobs. Full-batch Adam, light weight decay.
         Both trainers are generators that pause between epochs, so a slow phone can spread one
         training run over many animation frames instead of freezing for a third of a second. */
      function* trainLogit(d) {
        const w = new Float64Array(D + 1), m = new Float64Array(D + 1), v = new Float64Array(D + 1), g = new Float64Array(D + 1);
        for (let e = 1; e <= 200; e++) {
          g.fill(0);
          for (let i = 0; i < d.n; i++) {
            let z = w[D]; for (let j = 0; j < D; j++) z += w[j] * d.X[i * D + j];
            const err = sig(z) - d.y[i];
            for (let j = 0; j < D; j++) g[j] += err * d.X[i * D + j];
            g[D] += err;
          }
          for (let j = 0; j <= D; j++) {
            const gj = g[j] / d.n + (j < D ? 0.02 * w[j] : 0);
            m[j] = 0.9 * m[j] + 0.1 * gj; v[j] = 0.999 * v[j] + 0.001 * gj * gj;
            w[j] -= 0.1 * (m[j] / (1 - Math.pow(0.9, e))) / (Math.sqrt(v[j] / (1 - Math.pow(0.999, e))) + 1e-8);
          }
          yield;
        }
        return w;
      }
      const logitScore = (w, X, i) => { let z = w[D]; for (let j = 0; j < D; j++) z += w[j] * X[i * D + j]; return z; };
      /* neural net: 8 → 16 tanh → 1, which is 128 + 16 + 16 + 1 = 161 knobs.
         800 Adam steps on mini-batches of 64 (the whole set when it is smaller), the same
         recipe at every training-set size. wd is weight decay on the 144 weights (not the 17
         biases): 0 by default, or the 0.02 logistic regression always gets. */
      const HN = 16, OB = HN * D, OV = OB + HN, OC = OV + HN, NP = OC + 1;
      function* trainNet(d, seed, wd) {
        const r = rng(seed);
        const w = new Float64Array(NP), m = new Float64Array(NP), v = new Float64Array(NP), g = new Float64Array(NP), hid = new Float64Array(HN);
        for (let k = 0; k < OB; k++) w[k] = gauss(r) * Math.sqrt(1 / D);
        for (let k = 0; k < HN; k++) w[OV + k] = gauss(r) * Math.sqrt(1 / HN);
        const E = 800, B = Math.min(d.n, 64);
        for (let e = 1; e <= E; e++) {
          g.fill(0);
          const lr = 0.02 * (1 - 0.9 * e / E);
          for (let b = 0; b < B; b++) {
            const i = B === d.n ? b : Math.floor(r() * d.n), xo = i * D;
            let z = w[OC];
            for (let k = 0; k < HN; k++) {
              let a = w[OB + k]; const wo = k * D;
              for (let j = 0; j < D; j++) a += w[wo + j] * d.X[xo + j];
              a = Math.tanh(a); hid[k] = a; z += w[OV + k] * a;
            }
            const err = sig(z) - d.y[i]; g[OC] += err;
            for (let k = 0; k < HN; k++) {
              g[OV + k] += err * hid[k];
              const dh = err * w[OV + k] * (1 - hid[k] * hid[k]);
              g[OB + k] += dh; const wo = k * D;
              for (let j = 0; j < D; j++) g[wo + j] += dh * d.X[xo + j];
            }
          }
          const c1 = 1 - Math.pow(0.9, e), c2 = 1 - Math.pow(0.999, e);
          for (let k = 0; k < NP; k++) {
            const gk = g[k] / B + (wd && (k < OB || (k >= OV && k < OC)) ? wd * w[k] : 0);
            m[k] = 0.9 * m[k] + 0.1 * gk; v[k] = 0.999 * v[k] + 0.001 * gk * gk;
            w[k] -= lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + 1e-8);
          }
          if (e % 8 === 0) yield;
        }
        return w;
      }
      const netScore = (w, X, i) => {
        let z = w[OC];
        for (let k = 0; k < HN; k++) { let a = w[OB + k]; for (let j = 0; j < D; j++) a += w[k * D + j] * X[i * D + j]; z += w[OV + k] * Math.tanh(a); }
        return z;
      };
      function accuracy(score, w, d) { let c = 0; for (let i = 0; i < d.n; i++) if ((score(w, d.X, i) > 0 ? 1 : 0) === d.y[i]) c++; return c / d.n; }

      /* 4,000 applicants no model ever trains on. Every accuracy in the chapter is measured here. */
      const TEST = makeData(4000, 999);
      const LATE = TEST.y.reduce((a, b) => a + b, 0) / TEST.n;
      const BASE = Math.max(LATE, 1 - LATE);                       // always guess the commoner answer
      const CEIL = accuracy((w, X, i) => trueLogit(Array.from(X.subarray(i * D, i * D + D))), null, TEST);  // the hidden rule itself
      const SIZES = [30, 60, 100, 300, 1000, 3000];
      /* sample k is a fresh pool of 3,000 applicants; "n rows" means its first n */
      const pools = {}, fits = {}, jobs = {}, WD = 0.02;
      function* fitJob(k, n, wd) {
        const pool = pools[k] || (pools[k] = makeData(3000, 1006 + k));
        const d = { X: pool.X, y: pool.y, n };
        const lw = yield* trainLogit(d);
        const nw = yield* trainNet(d, 506 + k, wd);
        return { lw, nw, la: accuracy(logitScore, lw, TEST), na: accuracy(netScore, nw, TEST) };
      }
      const fitKey = (k, n, wd) => k + ':' + n + ':' + (wd || 0);
      const got = (k, n, wd) => fits[fitKey(k, n, wd)] || null;
      /* Spend at most `ms` on the fit for (sample k, n rows); return it once finished, else null.
         The headless checkers (which pin the canvas scale) get the whole fit in one go, so they
         test the trained picture and not the "training…" placeholder. */
      const SLICE = window.__ZTA_CANVAS_SCALE ? Infinity : 10;
      function work(k, n, wd, ms) {
        const key = fitKey(k, n, wd);
        if (fits[key]) return fits[key];
        const job = jobs[key] || (jobs[key] = fitJob(k, n, wd || 0));
        const t0 = performance.now();
        for (;;) {
          const r = job.next();
          if (r.done) { delete jobs[key]; return (fits[key] = r.value); }
          if (performance.now() - t0 >= ms) return null;
        }
      }
      const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
      const spread = (a) => { const mu = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - mu) * (y - mu), 0) / a.length); };

      /* ================================================================== */
      /* Interactive 1: the head-to-head                                    */
      /* ================================================================== */
      function headToHead() {
        const [cv, g] = ctx.canvas(720, 340);
        const P = { x: 52, y: 26, w: 400, h: 240 };
        const S = { si: 2, k: 1, wd: 0, seen: { 2: true }, dirty: true };
        const have = (i) => got(S.k, SIZES[i], S.wd);
        /* the next size still to train: the one being looked at first, then the rest in order */
        const pending = () => { if (!have(S.si)) return S.si; for (let i = 0; i < SIZES.length; i++) if (S.seen[i] && !have(i)) return i; return -1; };
        const xAt = (n) => P.x + 16 + (Math.log10(n) - Math.log10(30)) / 2 * (P.w - 32);
        const yAt = (a) => P.y + (0.90 - a) / 0.35 * P.h;
        const ro = ctx.readout();
        function draw() {
          g.clearRect(0, 0, cv.W, cv.H);
          g.fillStyle = '#0f1520'; g.fillRect(P.x, P.y, P.w, P.h); g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
          g.font = MONO; g.fillStyle = C.muted; g.textAlign = 'right';
          for (const a of [0.6, 0.7, 0.8, 0.9]) {
            g.strokeStyle = C.line; g.beginPath(); g.moveTo(P.x, yAt(a)); g.lineTo(P.x + P.w, yAt(a)); g.stroke();
            g.fillText((a * 100).toFixed(0) + '%', P.x - 6, yAt(a) + 4);
          }
          g.textAlign = 'center';
          SIZES.forEach((n, i) => { g.fillStyle = i === S.si ? C.text : C.muted; g.fillText(n.toLocaleString('en-GB'), xAt(n), P.y + P.h + 16); });
          g.fillStyle = C.muted; g.fillText('rows of training data (log scale)', P.x + P.w / 2, P.y + P.h + 34);
          g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
          g.fillStyle = 'rgba(124,156,255,0.10)'; g.fillRect(xAt(SIZES[S.si]) - 13, P.y, 26, P.h);
          g.setLineDash([5, 4]); g.lineWidth = 1.4;
          g.strokeStyle = C.warn; g.beginPath(); g.moveTo(P.x, yAt(CEIL)); g.lineTo(P.x + P.w, yAt(CEIL)); g.stroke();
          g.strokeStyle = C.muted; g.beginPath(); g.moveTo(P.x, yAt(BASE)); g.lineTo(P.x + P.w, yAt(BASE)); g.stroke();
          g.setLineDash([]);
          [['la', C.green], ['na', C.pink]].forEach(([key, col]) => {
            const pts = SIZES.map((n, i) => S.seen[i] && have(i) ? { x: xAt(n), y: yAt(have(i)[key]) } : null).filter(Boolean);
            g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
            pts.forEach((q, i) => { if (!i) g.moveTo(q.x, q.y); else g.lineTo(q.x, q.y); }); g.stroke();
            pts.forEach((q) => { g.beginPath(); g.arc(q.x, q.y, 5, 0, Math.PI * 2); g.fillStyle = col; g.fill(); g.strokeStyle = '#0a0e16'; g.lineWidth = 1.2; g.stroke(); });
          });
          g.restore();
          g.textAlign = 'left'; g.font = MONO;
          g.fillStyle = C.warn; g.fillText('- - the hidden rule itself: ' + pct(CEIL), P.x, P.y + P.h + 54);
          g.fillStyle = C.muted; g.fillText('- - always guess "on time": ' + pct(BASE), P.x + 232, P.y + P.h + 54);
          g.font = BOLD; g.fillStyle = C.text; g.fillText('accuracy on 4,000 applicants neither model has seen', P.x, 16);

          const f = have(S.si), X = 478;
          g.font = BOLD; g.fillStyle = C.text; g.fillText('trained on ' + SIZES[S.si].toLocaleString('en-GB') + ' rows', X, 44);
          g.font = FONT; g.fillStyle = C.green; g.fillText('● logistic regression', X, 80);
          g.font = MONO; g.fillStyle = C.muted; g.fillText('9 knobs', X + 160, 80);
          g.font = BIG; g.fillStyle = f ? C.green : C.muted; g.fillText(f ? pct(f.la) : 'training…', X, 108);
          g.font = FONT; g.fillStyle = C.pink; g.fillText('● neural net', X, 150);
          g.font = MONO; g.fillStyle = C.muted; g.fillText('161 knobs', X + 160, 150);
          g.font = BIG; g.fillStyle = f ? C.pink : C.muted; g.fillText(f ? pct(f.na) : 'training…', X, 178);
          const gap = f ? (f.la - f.na) * 100 : 0;
          if (f) {
            g.font = BOLD; g.fillStyle = Math.abs(gap) < 0.5 ? C.muted : gap > 0 ? C.green : C.pink;
            wrapLines(g, Math.abs(gap) < 0.5 ? 'Too close to call: under half a point apart.'
              : gap > 0 ? 'The simple model wins by ' + gap.toFixed(1) + ' points.'
              : 'The neural net wins by ' + (-gap).toFixed(1) + ' points.', 220).forEach((ln, i) => g.fillText(ln, X, 220 + i * 17));
          }
          g.font = FONT; g.fillStyle = C.muted;
          wrapLines(g, 'Sample ' + S.k + ' of 20. The net has ' + (S.wd ? 'the same weight decay as the simple model.' : 'no weight decay.'), 220).forEach((ln, i) => g.fillText(ln, X, 268 + i * 17));
          ro.set({ 'training rows': SIZES[S.si].toLocaleString('en-GB'), sample: S.k + ' of 20', 'logistic regression': f ? pct(f.la) : 'training…', 'neural net': f ? pct(f.na) : 'training…', 'simple minus net': f ? (gap >= 0 ? '+' : '') + gap.toFixed(1) + ' pts' : '—' });
        }
        ctx.loop(() => {
          const i = pending();
          if (i >= 0 && work(S.k, SIZES[i], S.wd, SLICE)) S.dirty = true;
          if (S.dirty) { draw(); S.dirty = false; }
        });
        const sl = ctx.slider({ label: 'training rows', min: 0, max: 5, step: 1, value: 2, fmt: (i) => SIZES[i].toLocaleString('en-GB'), onChange: (i) => { S.si = i; S.seen[i] = true; S.dirty = true; } });
        const allBtn = ctx.button('Train every size', () => { SIZES.forEach((_, i) => { S.seen[i] = true; }); S.dirty = true; }, 'primary');
        const newBtn = ctx.button('New sample of applicants', () => { S.k = S.k % 20 + 1; S.seen = {}; S.seen[S.si] = true; S.dirty = true; });
        const wdSel = ctx.select({ label: 'weight decay on the net', options: [{ value: '0', label: 'none' }, { value: '1', label: 'same as the simple model' }], value: '0', onChange: (v) => { S.wd = +v ? WD : 0; S.dirty = true; } });
        return ctx.figure(cv, 'Two models, really trained in this page each time you move the slider, on the same rows of a made-up loan book: eight columns per past applicant, and whether they paid late. Four columns matter and four are junk (house number, letters in the surname). Green is logistic regression: one weight per column plus a bias, 9 numbers. Pink is a neural net with one hidden layer of 16 neurons, 161 numbers, trained with one fixed recipe at every size. One thing is not even: logistic regression is trained with a light weight decay, chapter 3\'s restraint on large weights, and the net starts with none. The dropdown gives the net the same. The yellow line is the score of the hidden rule that generated the data; no model can beat it by more than a fluke, because some applicants pay late by plain chance.', [sl, wdSel, allBtn, newBtn], ro);
      }

      /* ================================================================== */
      /* Interactive 2: twenty samples per model — spread and plateau       */
      /* ================================================================== */
      function twentyDraws() {
        const [cv, g] = ctx.canvas(720, 300);
        const P = { x: 150, y: 40, w: 400, h: 200 };
        const S = { n: 300, done: 0, queue: 0, dirty: true };
        const xAt = (a) => P.x + 8 + (a - 0.60) / 0.30 * (P.w - 16);
        const ro = ctx.readout();
        function stats() {
          const la = [], na = []; let wins = 0;
          for (let k = 1; k <= S.done; k++) { const f = got(k, S.n, 0); la.push(f.la); na.push(f.na); if (f.la > f.na) wins++; }
          return { la, na, wins };
        }
        function draw() {
          g.clearRect(0, 0, cv.W, cv.H);
          const st = stats();
          g.textAlign = 'left'; g.font = BOLD; g.fillStyle = C.text;
          g.fillText('one dot = one model, trained on its own ' + S.n.toLocaleString('en-GB') + ' applicants', P.x, 22);
          g.fillStyle = '#0f1520'; g.fillRect(P.x, P.y, P.w, P.h); g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
          g.font = MONO; g.textAlign = 'center';
          for (let a = 0.60; a < 0.905; a += 0.05) {
            g.strokeStyle = C.line; g.beginPath(); g.moveTo(xAt(a), P.y); g.lineTo(xAt(a), P.y + P.h); g.stroke();
            g.fillStyle = C.muted; g.fillText((a * 100).toFixed(0) + '%', xAt(a), P.y + P.h + 16);
          }
          g.fillStyle = C.muted; g.fillText('accuracy on the 4,000 unseen applicants', P.x + P.w / 2, P.y + P.h + 34);
          g.textAlign = 'left'; g.fillStyle = C.warn; g.fillText('- - the hidden rule: ' + pct(CEIL), P.x, P.y + P.h + 52);
          const rows = [['logistic regression', '9 knobs', st.la, C.green, P.y + 55], ['neural net', '161 knobs', st.na, C.pink, P.y + 148]];
          g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
          g.setLineDash([5, 4]); g.strokeStyle = C.warn; g.lineWidth = 1.4; g.beginPath(); g.moveTo(xAt(CEIL), P.y); g.lineTo(xAt(CEIL), P.y + P.h); g.stroke(); g.setLineDash([]);
          rows.forEach(([, , arr, col, y]) => {
            arr.forEach((a, i) => { g.beginPath(); g.arc(xAt(ctx.clamp(a, 0.60, 0.90)), y + ((i * 7) % 11 - 5) * 5, 4.5, 0, Math.PI * 2); g.globalAlpha = 0.8; g.fillStyle = col; g.fill(); g.globalAlpha = 1; });
            if (arr.length > 1) {
              const mu = mean(arr), sd = spread(arr);
              g.strokeStyle = C.text; g.lineWidth = 2; g.beginPath(); g.moveTo(xAt(mu), y - 36); g.lineTo(xAt(mu), y + 36); g.stroke();
              g.lineWidth = 3; g.beginPath(); g.moveTo(xAt(mu - sd), y + 40); g.lineTo(xAt(mu + sd), y + 40); g.stroke();
            }
          });
          g.restore();
          rows.forEach(([name, knobs, arr, col, y]) => {
            g.textAlign = 'left'; g.font = BOLD; g.fillStyle = col; g.fillText(name, 12, y - 2);
            g.font = MONO; g.fillStyle = C.muted; g.fillText(knobs, 12, y + 16);
            if (arr.length > 1) {
              g.font = MONO; g.fillStyle = C.text; g.fillText('average ' + pct(mean(arr)), P.x + P.w + 14, y - 2);
              g.fillStyle = C.muted; g.fillText('spread ±' + (spread(arr) * 100).toFixed(1), P.x + P.w + 14, y + 16);
            }
          });
          if (!S.done) { g.font = FONT; g.fillStyle = C.muted; g.textAlign = 'center'; g.fillText('Press "Run 20 samples".', P.x + P.w / 2, P.y + P.h / 2 + 4); g.textAlign = 'left'; }
          const two = st.la.length > 1;
          ro.set({
            'rows each': S.n.toLocaleString('en-GB'), 'samples run': S.done + ' of 20',
            'logistic': two ? pct(mean(st.la)) + ' ±' + (spread(st.la) * 100).toFixed(1) : '—',
            'neural net': two ? pct(mean(st.na)) + ' ±' + (spread(st.na) * 100).toFixed(1) : '—',
            'simple model wins': S.done ? st.wins + ' of ' + S.done : '—',
          });
        }
        ctx.loop(() => {
          if (S.queue > 0 && work(S.done + 1, S.n, 0, SLICE)) { S.done++; S.queue--; S.dirty = true; }
          if (S.dirty) { draw(); S.dirty = false; }
        });
        const sel = ctx.select({ label: 'rows in each training sample', options: [30, 100, 300, 1000, 3000].map((n) => ({ value: String(n), label: n.toLocaleString('en-GB') + ' rows' })), value: '300', onChange: (v) => { S.n = +v; S.done = 0; S.queue = 0; S.dirty = true; } });
        const runBtn = ctx.button('Run 20 samples', () => { S.done = 0; S.queue = 20; S.dirty = true; }, 'primary');
        return ctx.figure(cv, 'The opening demo, repeated twenty times at one size. Each sample is a different set of past applicants drawn from the same world; both models are trained on it and marked on the same 4,000 unseen applicants. The white upright is the average of the twenty. The white bar under each cloud is its spread: one standard deviation either side of the average. The net has no weight decay here, whatever the dropdown in the opening demo says.', [sel, runBtn], ro);
      }

      /* ================================================================== */
      /* Interactive 3: one hand-set rule against two trained models         */
      /* ================================================================== */
      function baselineLadder() {
        const [cv, g] = ctx.canvas(720, 330);
        const S = { f: 0, thr: -0.5, dir: -1, dirty: true };
        let big = null;                                            // trained when the figure is first on screen
        const raw = (z) => FEATS[S.f].show(FEATS[S.f].mu + FEATS[S.f].sd * z);
        const ro = ctx.readout();
        function ruleAcc() {
          let c = 0;
          for (let i = 0; i < TEST.n; i++) { const z = TEST.X[i * D + S.f]; if (((S.dir > 0 ? z > S.thr : z < S.thr) ? 1 : 0) === TEST.y[i]) c++; }
          return c / TEST.n;
        }
        function draw() {
          g.clearRect(0, 0, cv.W, cv.H);
          if (!big) { g.textAlign = 'left'; g.font = BOLD; g.fillStyle = C.muted; g.fillText('Training both models on 3,000 rows…', 30, 26); ro.set({ 'your rule': '—', baseline: pct(BASE) }); return; }
          const P = { x: 30, y: 44, w: 340, h: 170 }, BINS = 24, ft = FEATS[S.f];
          const hs = Array.from({ length: BINS }, () => [0, 0]);
          const xz = (z) => P.x + (ctx.clamp(z, -3, 3) + 3) / 6 * P.w;
          for (let i = 0; i < TEST.n; i++) { const b = Math.min(BINS - 1, Math.max(0, Math.floor((TEST.X[i * D + S.f] + 3) / 6 * BINS))); hs[b][TEST.y[i]]++; }
          const peak = Math.max(1, ...hs.map((b) => Math.max(b[0], b[1])));
          g.textAlign = 'left'; g.font = BOLD; g.fillStyle = C.text; g.fillText('4,000 unseen applicants, by ' + ft.name, P.x, 26);
          g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
          const shadeX = S.dir > 0 ? xz(S.thr) : P.x, shadeW = S.dir > 0 ? P.x + P.w - xz(S.thr) : xz(S.thr) - P.x;
          g.fillStyle = 'rgba(251,191,36,0.10)'; g.fillRect(shadeX, P.y, shadeW, P.h);
          for (let b = 0; b < BINS; b++) {
            const bw = P.w / BINS, up = hs[b][0] / peak * (P.h / 2 - 6), dn = hs[b][1] / peak * (P.h / 2 - 6);
            g.fillStyle = 'rgba(56,217,169,0.75)'; g.fillRect(P.x + b * bw + 1, P.y + P.h / 2 - up, bw - 2, up);
            g.fillStyle = 'rgba(251,113,133,0.8)'; g.fillRect(P.x + b * bw + 1, P.y + P.h / 2 + 2, bw - 2, dn);
          }
          g.strokeStyle = C.warn; g.lineWidth = 2.5; g.beginPath(); g.moveTo(xz(S.thr), P.y); g.lineTo(xz(S.thr), P.y + P.h); g.stroke();
          g.restore();
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
          g.font = MONO;
          const plate = (s, tx, ty, col) => { const tw = g.measureText(s).width; g.fillStyle = '#0a0e16'; g.fillRect(tx - 3, ty - 11, tw + 6, 15); g.fillStyle = col; g.fillText(s, tx, ty); };
          plate('paid on time ↑', P.x + 5, P.y + 14, C.green);
          plate('paid late ↓', P.x + 5, P.y + P.h - 6, C.danger);
          const lab = S.dir > 0 ? 'refuse →' : '← refuse', lw = g.measureText(lab).width;
          g.fillStyle = C.warn; g.fillText(lab, ctx.clamp(S.dir > 0 ? xz(S.thr) + 4 : xz(S.thr) - lw - 4, P.x, P.x + P.w - lw), P.y + P.h + 18);

          const acc = ruleAcc(), COL = 400, BW = 230;
          const bars = [['always guess "on time"', BASE, C.muted], ['your one rule', acc, C.warn], ['logistic regression, 3,000 rows', big.la, C.green], ['neural net, 3,000 rows', big.na, C.pink]];
          bars.forEach(([name, a, col], i) => {
            const y = 40 + i * 52;
            g.font = FONT; g.fillStyle = C.muted; g.fillText(name, COL, y);
            g.fillStyle = C.line; g.fillRect(COL, y + 8, BW, 12);
            g.fillStyle = col; g.fillRect(COL, y + 8, ctx.clamp((a - 0.5) / 0.4, 0, 1) * BW, 12);
            g.font = BOLD; g.fillText(pct(a), COL + BW + 10, y + 19);
          });
          g.font = MONO; g.fillStyle = C.muted; g.fillText('bars run from 50% to 90%', COL, 40 + 4 * 52 - 6);
          const share = (acc - BASE) / (big.na - BASE);
          g.font = BOLD; g.fillStyle = acc <= BASE + 0.005 ? C.danger : C.text;
          const msg = acc < BASE - 0.005
            ? 'This rule is worse than refusing nobody: it turns away more people who would have paid on time than people who would not.'
            : acc <= BASE + 0.005
            ? 'This rule is no better than refusing nobody. Either the threshold is in the wrong place or the column carries no signal.'
            : 'Refuse when ' + ft.name + ' is ' + (S.dir > 0 ? 'above ' : 'below ') + raw(S.thr) + '. One line of code, and ' + (share * 100).toFixed(0) + '% of everything the neural net adds over the baseline.';
          wrapLines(g, msg, 660).forEach((ln, i) => g.fillText(ln, P.x, 268 + i * 18));
          ro.set({ 'your rule': pct(acc), baseline: pct(BASE), 'logistic': pct(big.la), 'neural net': pct(big.na), 'share of the net\'s gain': share > 0 ? (share * 100).toFixed(0) + '%' : 'none' });
        }
        ctx.loop(() => {
          if (!big && (big = work(1, 3000, 0, SLICE))) S.dirty = true;
          if (S.dirty) { draw(); S.dirty = false; }
        });
        const thrSl = ctx.slider({ label: 'refuse beyond', min: -2, max: 2, step: 0.1, value: -0.5, fmt: (z) => raw(z), onChange: (v) => { S.thr = v; S.dirty = true; } });
        const sel = ctx.select({ label: 'column the rule looks at', options: FEATS.map((f, i) => ({ value: String(i), label: f.name })), value: '0', onChange: (v) => { S.f = +v; S.dir = FEATS[S.f].dir; thrSl.value = S.thr; S.dirty = true; } });
        const flipBtn = ctx.button('Flip the rule', () => { S.dir = -S.dir; S.dirty = true; });
        return ctx.figure(cv, 'Left: every unseen applicant, sorted by one column; green bars paid on time, red bars paid late. Your rule refuses everyone on the shaded side of the yellow line and is marked on all 4,000. Right: the same marking for the laziest possible answer and for both trained models from the opening demo at 3,000 rows. (Setting one threshold against 4,000 examples barely flatters it. Against 40 it would, for the reason chapter 3 gave.)', [sel, thrSl, flipBtn], ro);
      }

      /* ================================================================== */
      /* Interactive 4: a decision you can read, and one you cannot          */
      /* ================================================================== */
      function readDecision() {
        const [cv, g] = ctx.canvas(720, 390);
        let big = null;                                            // shared with the figure above; trained once
        const DEF = [2, 24, 16, 3];
        const S = { v: DEF.slice(), dirty: true };
        const ro = ctx.readout();
        const zs = () => { const x = new Float64Array(D); for (let j = 0; j < 4; j++) x[j] = (S.v[j] - FEATS[j].mu) / FEATS[j].sd; return x; };
        const labels = [(v) => v + ' years in job', (v) => '£' + v + 'k of debt', (v) => '£' + v + 'k loan', (v) => v + ' credit cards'];
        const sgn = (v, d) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d == null ? 2 : d);
        function draw() {
          g.clearRect(0, 0, cv.W, cv.H);
          if (!big) { g.textAlign = 'left'; g.font = BOLD; g.fillStyle = C.muted; g.fillText('Training both models on 3,000 rows…', 16, 22); ro.set({ 'logistic: late risk': '—', 'net: late risk': '—' }); return; }
          const x = zs(), lw = big.lw;
          const lz = logitScore(lw, x, 0), nz = netScore(big.nw, x, 0);
          const x2 = x.slice(); x2[1] += 0.5;                      // £5k more debt is half a standard deviation
          const dL = logitScore(lw, x2, 0) - lz, dN = netScore(big.nw, x2, 0) - nz;
          /* ---- left: the whole logistic model ---- */
          g.textAlign = 'left'; g.font = BOLD; g.fillStyle = C.green; g.fillText('Logistic regression: this is the whole model', 16, 22);
          const ZX = 250, UNIT = 44;
          const rows = [['starting score', lw[D]]].concat([0, 1, 2, 3].map((j) => [labels[j](S.v[j]), lw[j] * x[j]]));
          rows.push(['the four junk columns', 0]);
          rows.forEach(([name, c], i) => {
            const y = 54 + i * 30;
            g.font = FONT; g.fillStyle = C.muted; g.fillText(name, 16, y);
            g.fillStyle = c > 0 ? C.danger : C.green;
            const len = ctx.clamp(c * UNIT, -84, 84);
            g.fillRect(len >= 0 ? ZX : ZX + len, y - 10, Math.abs(len), 12);
            g.font = MONO; g.textAlign = 'right'; g.fillStyle = C.text; g.fillText(sgn(c), 356, y); g.textAlign = 'left';
          });
          g.strokeStyle = C.muted; g.lineWidth = 1; g.beginPath(); g.moveTo(ZX, 38); g.lineTo(ZX, 54 + 5 * 30 + 6); g.stroke();
          g.strokeStyle = C.line; g.beginPath(); g.moveTo(16, 222); g.lineTo(356, 222); g.stroke();
          g.font = BOLD; g.fillStyle = C.text; g.fillText('add them up', 16, 244);
          g.font = MONO; g.textAlign = 'right'; g.fillText(sgn(lz), 356, 244); g.textAlign = 'left';
          g.font = FONT; g.fillStyle = C.muted; g.fillText('score above 0 means refuse', 16, 266);
          const verdict = (z, xx, y) => {
            g.font = BIG; g.fillStyle = z > 0 ? C.danger : C.green; g.fillText(z > 0 ? 'REFUSE' : 'APPROVE', xx, y);
            g.font = FONT; g.fillStyle = C.muted; g.fillText('chance of paying late: ' + (sig(z) * 100).toFixed(0) + '%', xx, y + 22);
          };
          verdict(lz, 16, 304);
          /* ---- right: the net ---- */
          const RX = 392;
          g.font = BOLD; g.fillStyle = C.pink; g.fillText('Neural net: 128 of its 161 weights', RX, 22);
          g.save(); g.beginPath(); g.rect(RX, 34, 312, 110); g.clip();
          for (let k = 0; k < HN; k++) for (let j = 0; j < D; j++) {
            g.fillStyle = '#0f1520'; g.fillRect(RX + k * 19.5, 34 + j * 13.5, 18, 12);
            g.fillStyle = ctx.heat(big.nw[k * D + j] / 1.5); g.fillRect(RX + k * 19.5, 34 + j * 13.5, 18, 12);
          }
          g.restore();
          g.font = FONT; g.fillStyle = C.muted; g.fillText('red = positive, blue = negative. The reason is in here.', RX, 164);
          verdict(nz, RX, 304);
          /* ---- the comparison that matters ---- */
          g.strokeStyle = C.line; g.beginPath(); g.moveTo(RX, 180); g.lineTo(704, 180); g.stroke();
          g.font = BOLD; g.fillStyle = C.text; g.fillText('What £5k more debt does to the score', RX, 202);
          g.font = FONT; g.fillStyle = C.green; g.fillText('logistic', RX, 226);
          g.font = MONO; g.fillStyle = C.text; g.fillText(sgn(dL) + '  for every applicant', RX + 70, 226);
          g.font = FONT; g.fillStyle = C.pink; g.fillText('neural net', RX, 250);
          g.font = MONO; g.fillStyle = dN < 0 ? C.warn : C.text; g.fillText(sgn(dN) + '  for this applicant', RX + 70, 250);
          g.font = BOLD; g.fillStyle = dN < 0 ? C.warn : C.muted;
          wrapLines(g, dN < 0 ? 'For this applicant the net says more debt makes them safer. Nothing on screen warned you the rule had turned over.'
            : (lz > 0) !== (nz > 0) ? 'The two models disagree about this person. Only one of them can show its working.'
            : 'Move the loan size and watch the net\'s number change while the logistic one stays put.', 680).forEach((ln, i) => g.fillText(ln, 16, 352 + i * 18));
          ro.set({ 'logistic: late risk': (sig(lz) * 100).toFixed(0) + '%', 'net: late risk': (sig(nz) * 100).toFixed(0) + '%', '£5k more debt, logistic': sgn(dL), '£5k more debt, net': sgn(dN) });
        }
        ctx.loop(() => {
          if (!big && (big = work(1, 3000, 0, SLICE))) S.dirty = true;
          if (S.dirty) { draw(); S.dirty = false; }
        });
        const mk = (j, label, min, max) => ctx.slider({ label, min, max, step: 1, value: DEF[j], onChange: (v) => { S.v[j] = v; S.dirty = true; } });
        const sls = [mk(0, 'years in current job', 0, 12), mk(1, 'existing debt (£k)', 0, 40), mk(2, 'loan size asked for (£k)', 3, 27), mk(3, 'credit cards held', 0, 6)];
        const reset = ctx.button('Reset applicant', () => { S.v = DEF.slice(); sls.forEach((s, j) => { s.value = DEF[j]; }); S.dirty = true; });
        return ctx.figure(cv, 'One applicant, scored by both models from the opening demo at 3,000 rows. Left: logistic regression multiplies each column by its one weight and adds the results. Red bars push toward refusal, green toward approval, and their sum is the score. Nothing is hidden. Right: the neural net\'s first-layer weights. Its score comes out of 161 numbers passed through 16 bent curves, so the only way to learn what debt does is to nudge debt and watch, which is what the bottom-right lines do. The junk columns are held at their average, so they add nothing here.', sls.concat([reset]), ro);
      }

      /* ================================================================== */
      /* Interactive 5: what it costs to answer                              */
      /* ================================================================== */
      function budgetCalc() {
        const [cv, g] = ctx.canvas(720, 330);
        const S = { req: 2000, llm: 0.003, budget: 30000, dirty: true };
        const ro = ctx.readout();
        const money = (d) => d >= 1e6 ? '$' + (d / 1e6).toFixed(d >= 1e7 ? 0 : 1) + 'M' : d >= 1000 ? '$' + Math.round(d).toLocaleString('en-GB') : d >= 10 ? '$' + d.toFixed(0) : d < 0.005 ? 'under 1¢' : '$' + d.toFixed(2);
        const ms = (t) => t >= 1000 ? (t / 1000).toFixed(t >= 10000 ? 0 : 1) + ' s' : t >= 1 ? t.toFixed(0) + ' ms' : t.toFixed(1) + ' ms';
        const count = (n) => n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + ' million' : Math.round(n).toLocaleString('en-GB');
        const sf2 = (v) => { const e = Math.pow(10, Math.floor(Math.log10(v)) - 1); return Math.round(v / e) * e; };
        function rowsNow() {
          return [
            { name: 'A hand-written rule', each: 0.0000001, slow: 0.1, col: C.muted },
            { name: 'A small trained model', each: 0.000005, slow: 5, col: C.green },
            { name: 'An LLM call for every request', each: S.llm, slow: 1500, col: C.pink },
            { name: 'Small model first, LLM for the hardest 5%', each: 0.000005 + 0.05 * S.llm, slow: 1500, col: C.accent, split: true },
          ];
        }
        function draw() {
          g.clearRect(0, 0, cv.W, cv.H);
          const rows = rowsNow(), BX = 250, BW = 250;
          g.textAlign = 'left'; g.font = BOLD; g.fillStyle = C.text;
          g.fillText(count(S.req) + ' requests a day, answer needed within ' + ms(S.budget), 16, 22);
          g.font = MONO; g.fillStyle = C.muted; g.fillText('bill per month (30 days); each grid line is ×10', BX, 44);
          g.textAlign = 'center';
          for (let e = 0; e <= 7; e++) {
            const x = BX + e / 7 * BW;
            g.strokeStyle = C.line; g.lineWidth = 1; g.beginPath(); g.moveTo(x, 52); g.lineTo(x, 286); g.stroke();
            if (e % 2 === 0) { g.fillStyle = C.muted; g.fillText(['$1', '', '$100', '', '$10k', '', '$1M', ''][e], x, 302); }
          }
          g.textAlign = 'left';
          rows.forEach((r, i) => {
            const y = 76 + i * 58, bill = r.each * S.req * 30;
            g.font = FONT; g.fillStyle = r.col;
            wrapLines(g, r.name, 220).forEach((ln, k) => g.fillText(ln, 16, y + k * 16));
            g.save(); g.beginPath(); g.rect(BX, 52, BW, 234); g.clip();
            g.fillStyle = r.col; g.globalAlpha = 0.85; g.fillRect(BX, y - 8, ctx.clamp(Math.log10(Math.max(1, bill)) / 7, 0.004, 1) * BW, 16); g.globalAlpha = 1;
            g.restore();
            g.font = BOLD; g.fillStyle = C.text; g.fillText(money(bill), BX + BW + 12, y + 4);
            const ok = r.slow <= S.budget;
            g.font = MONO; g.fillStyle = ok ? C.green : C.danger;
            const txt = r.split ? (ok ? '✓ ' + ms(r.slow) + ' at worst' : '✗ 1 answer in 20 is late')
              : ok ? '✓ ' + ms(r.slow) : '✗ ' + ms(r.slow) + ', ' + (r.slow / S.budget >= 10 ? Math.round(r.slow / S.budget) : (r.slow / S.budget).toFixed(1)) + '× too slow';
            g.fillText(txt, BX + BW + 12, y + 22);
          });
          const llmBill = S.llm * S.req * 30;
          g.font = BOLD; g.fillStyle = llmBill > 100000 || 1500 > S.budget ? C.danger : C.muted;
          wrapLines(g, 1500 > S.budget ? 'The LLM cannot answer in time at any price. A latency budget is a wall.'
            : llmBill > 100000 ? 'The LLM answers in time but its bill now has its own line in the company accounts.'
            : 'At this volume the LLM bill is small change. Engineer time costs more than any of these bars.', 690).forEach((ln, i) => g.fillText(ln, 16, 322 + i * 17));
          ro.set({ 'rule': money(rows[0].each * S.req * 30), 'small model': money(rows[1].each * S.req * 30), 'LLM every time': money(llmBill), 'small model + LLM for 5%': money(rows[3].each * S.req * 30) });
        }
        ctx.loop(() => { if (S.dirty) { draw(); S.dirty = false; } });
        const reqSl = ctx.slider({ label: 'requests per day', min: 2, max: 8, step: 0.05, value: Math.log10(2000), fmt: (e) => count(sf2(Math.pow(10, e))), onChange: (e) => { S.req = sf2(Math.pow(10, e)); S.dirty = true; } });
        const llmSl = ctx.slider({ label: 'price of one LLM call', min: 0.0005, max: 0.03, step: 0.0005, value: 0.003, fmt: (v) => '$' + (+v).toFixed(4), onChange: (v) => { S.llm = v; S.dirty = true; } });
        const budSl = ctx.slider({ label: 'time allowed per answer', min: 1, max: 4.5, step: 0.05, value: Math.log10(30000), fmt: (e) => ms(sf2(Math.pow(10, e))), onChange: (e) => { S.budget = sf2(Math.pow(10, e)); S.dirty = true; } });
        const preset = (label, req, bud, cls) => ctx.button(label, () => { S.req = req; S.budget = bud; S.llm = 0.003; reqSl.value = Math.log10(req); budSl.value = Math.log10(bud); llmSl.value = 0.003; S.dirty = true; }, cls);
        return ctx.figure(cv, 'Three ways to answer a request and one blend. The per-request figures are illustrative round numbers, chosen for their sizes relative to each other: a rule at $0.10 per million requests and 0.1 ms, a small trained model at $5 per million and 5 ms, and an LLM call taking 1.5 seconds at the price you set on the slider. The fourth row sends everything to the small model and passes on only the 5% it is least sure of. The bars are on a ×10 scale because the bills differ by factors of thousands.', [reqSl, llmSl, budSl, preset('Support inbox', 2000, 30000), preset('Loan desk', 500, 5000), preset('LLM on every keystroke', 50000000, 100, 'primary')], ro);
      }

      /* ================================================================== */
      /* Interactive 6: let the LLM label, then train something small        */
      /* ================================================================== */
      function distil() {
        const [cv, g] = ctx.canvas(720, 330);
        const NAMES = ['refund', 'delivery', 'login'];
        const OWN = [
          ['refund', 'charged', 'money', 'twice', 'payment', 'card', 'invoice', 'cancel', 'billing', 'overcharged', 'bank', 'fee'],
          ['parcel', 'late', 'tracking', 'courier', 'arrived', 'damaged', 'box', 'address', 'shipping', 'missing', 'delivery', 'driver'],
          ['password', 'login', 'locked', 'reset', 'account', 'email', 'code', 'sign', 'access', 'username', 'verify', 'app'],
        ];
        const SHARED = ['hello', 'please', 'help', 'my', 'the', 'is', 'not', 'still', 'order', 'need', 'thanks', 'again', 'urgent', 'with', 'have', 'was', 'this', 'it', 'yesterday', 'today', 'can', 'you', 'now', 'why', 'been', 'waiting', 'week', 'tried', 'nothing', 'happened'];
        const VOCAB = OWN[0].concat(OWN[1], OWN[2], SHARED), V = VOCAB.length, LEN = 7, TEACHER = 0.95;
        function tickets(n, seed) {
          const r = rng(seed), W = new Uint8Array(n * LEN), y = new Uint8Array(n), t = new Uint8Array(n);
          for (let i = 0; i < n; i++) {
            const k = Math.floor(r() * 3); y[i] = k;
            for (let j = 0; j < LEN; j++) {
              const u = r();
              W[i * LEN + j] = u < 0.42 ? k * 12 + Math.floor(r() * 12) : u < 0.50 ? ((k + 1 + Math.floor(r() * 2)) % 3) * 12 + Math.floor(r() * 12) : 36 + Math.floor(r() * SHARED.length);
            }
            t[i] = r() < TEACHER ? k : (k + 1 + Math.floor(r() * 2)) % 3;       // the simulated LLM's label
          }
          return { W, y, t, n };
        }
        const TRAIN = tickets(3000, 4242), HELD = tickets(2000, 777);
        /* naive Bayes on word counts, trained on the first n tickets with the teacher's labels */
        function student(n) {
          const cnt = [0, 1, 2].map(() => new Float64Array(V).fill(1)), tot = [V, V, V], prior = [1, 1, 1];
          for (let i = 0; i < n; i++) { const k = TRAIN.t[i]; prior[k]++; for (let j = 0; j < LEN; j++) { cnt[k][TRAIN.W[i * LEN + j]]++; tot[k]++; } }
          return (W, i) => {
            let best = 0, bs = -Infinity;
            for (let k = 0; k < 3; k++) { let s = Math.log(prior[k]); for (let j = 0; j < LEN; j++) s += Math.log(cnt[k][W[i * LEN + j]] / tot[k]); if (s > bs) { bs = s; best = k; } }
            return best;
          };
        }
        const NS = [5, 30, 100, 300, 1000, 3000];
        const models = NS.map(student);
        const accs = models.map((f) => { let c = 0; for (let i = 0; i < HELD.n; i++) if (f(HELD.W, i) === HELD.y[i]) c++; return c / HELD.n; });
        let tAcc = 0; for (let i = 0; i < HELD.n; i++) if (HELD.t[i] === HELD.y[i]) tAcc++; tAcc /= HELD.n;
        const S = { ni: 0, tk: 0, dirty: true };
        const ro = ctx.readout();
        const P = { x: 52, y: 30, w: 330, h: 190 };
        const xAt = (i) => P.x + 18 + i / 5 * (P.w - 36), yAt = (a) => P.y + (1.0 - a) / 0.70 * P.h;
        function draw() {
          g.clearRect(0, 0, cv.W, cv.H);
          g.textAlign = 'left'; g.font = BOLD; g.fillStyle = C.text; g.fillText('small model, marked on 2,000 unseen tickets', P.x, 18);
          g.fillStyle = '#0f1520'; g.fillRect(P.x, P.y, P.w, P.h); g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
          g.font = MONO; g.textAlign = 'right';
          for (const a of [0.4, 0.6, 0.8, 1.0]) { g.strokeStyle = C.line; g.beginPath(); g.moveTo(P.x, yAt(a)); g.lineTo(P.x + P.w, yAt(a)); g.stroke(); g.fillStyle = C.muted; g.fillText((a * 100).toFixed(0) + '%', P.x - 6, yAt(a) + 4); }
          g.textAlign = 'center';
          NS.forEach((n, i) => { g.fillStyle = i === S.ni ? C.text : C.muted; g.fillText(n.toLocaleString('en-GB'), xAt(i), P.y + P.h + 16); });
          g.fillStyle = C.muted; g.fillText('tickets the LLM labelled for it', P.x + P.w / 2, P.y + P.h + 34);
          g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
          g.fillStyle = 'rgba(124,156,255,0.10)'; g.fillRect(xAt(S.ni) - 14, P.y, 28, P.h);
          g.setLineDash([5, 4]); g.strokeStyle = C.pink; g.lineWidth = 1.4; g.beginPath(); g.moveTo(P.x, yAt(tAcc)); g.lineTo(P.x + P.w, yAt(tAcc)); g.stroke(); g.setLineDash([]);
          g.strokeStyle = C.green; g.lineWidth = 2; g.beginPath(); accs.forEach((a, i) => { if (!i) g.moveTo(xAt(i), yAt(a)); else g.lineTo(xAt(i), yAt(a)); }); g.stroke();
          accs.forEach((a, i) => { g.beginPath(); g.arc(xAt(i), yAt(a), i === S.ni ? 6.5 : 4.5, 0, Math.PI * 2); g.fillStyle = C.green; g.fill(); g.strokeStyle = '#0a0e16'; g.lineWidth = 1.2; g.stroke(); });
          g.restore();
          g.textAlign = 'left'; g.font = MONO; g.fillStyle = C.pink; g.fillText('- - the LLM itself: ' + pct(tAcc), P.x, P.y + P.h + 52);
          const X = 412, n = NS[S.ni], bill = n * 0.003;
          g.font = FONT; g.fillStyle = C.pink; g.fillText('LLM on every ticket', X, 44);
          g.font = BIG; g.fillText(pct(tAcc), X, 72);
          g.font = MONO; g.fillStyle = C.muted; g.fillText('$3,000 per million tickets', X + 110, 68);
          g.font = FONT; g.fillStyle = C.green; g.fillText('small model taught by ' + n.toLocaleString('en-GB') + ' LLM labels', X, 112);
          g.font = BIG; g.fillText(pct(accs[S.ni]), X, 140);
          g.font = MONO; g.fillStyle = C.muted; g.fillText('$5 per million tickets', X + 110, 136);
          g.fillText('one-off labelling bill: $' + bill.toFixed(2), X, 164);
          const i = S.tk, words = Array.from(HELD.W.subarray(i * LEN, i * LEN + LEN)).map((w) => VOCAB[w]).join(' ');
          const guess = models[S.ni](HELD.W, i);
          g.strokeStyle = C.line; g.beginPath(); g.moveTo(X, 180); g.lineTo(704, 180); g.stroke();
          g.font = FONT; g.fillStyle = C.muted; g.fillText('unseen ticket ' + (i + 1) + ':', X, 202);
          g.font = MONO; g.fillStyle = C.text; wrapLines(g, '"' + words + '"', 290).forEach((ln, k) => g.fillText(ln, X, 222 + k * 16));
          g.font = FONT; g.fillStyle = C.muted; g.fillText('really about: ' + NAMES[HELD.y[i]], X, 264);
          g.fillStyle = guess === HELD.y[i] ? C.green : C.danger; g.fillText('small model says: ' + NAMES[guess] + (guess === HELD.y[i] ? '  ✓' : '  ✗'), X, 284);
          g.font = BOLD; g.fillStyle = accs[S.ni] < 0.6 ? C.danger : C.text;
          wrapLines(g, accs[S.ni] < 0.6 ? 'Five labels is not a training set. The small model is wrong about half the time.'
            : 'The small model gives up ' + ((tAcc - accs[S.ni]) * 100).toFixed(1) + ' points and answers for one six-hundredth of the price.', 690).forEach((ln, k) => g.fillText(ln, 16, 318 + k * 16));
          ro.set({ 'LLM labels bought': n.toLocaleString('en-GB'), 'small model': pct(accs[S.ni]), 'LLM': pct(tAcc), 'labelling bill': '$' + bill.toFixed(2) });
        }
        ctx.loop(() => { if (S.dirty) { draw(); S.dirty = false; } });
        const sl = ctx.slider({ label: 'tickets labelled by the LLM', min: 0, max: 5, step: 1, value: 0, fmt: (i) => NS[i].toLocaleString('en-GB'), onChange: (i) => { S.ni = i; S.dirty = true; } });
        const nextBtn = ctx.button('Show another ticket', () => { S.tk = (S.tk + 1) % HELD.n; S.dirty = true; });
        /* jump to the next unseen ticket the small model, as currently trained, gets wrong */
        const missBtn = ctx.button('Show a mistake', () => {
          for (let s = 1; s <= HELD.n; s++) { const i = (S.tk + s) % HELD.n; if (models[S.ni](HELD.W, i) !== HELD.y[i]) { S.tk = i; break; } }
          S.dirty = true;
        });
        return ctx.figure(cv, 'Support tickets about one of three things: a refund, a delivery or a login. <b>The LLM here is simulated</b>: no language model runs in this page, and its place is taken by a labeller built to be right 95% of the time (94.3% on these 2,000 tickets). The small model is real. It is naive Bayes on word counts, trained in your browser on however many tickets you let the LLM label, mistakes included, then marked against the true answers. The tickets are seven words from a made-up vocabulary, so they read like telegrams. Prices are the illustrative ones from the calculator above.', [sl, nextBtn, missBtn], ro);
      }

      /* ================================================================== */
      /* Interactive 7: a decision guide scored by a points table            */
      /* ================================================================== */
      function guide() {
        const TOOLS = [
          { key: 'R', name: 'A hand-written rule', col: C.muted },
          { key: 'C', name: 'A classical model (chapter 16)', col: C.green },
          { key: 'N', name: 'A neural net you train', col: C.accent },
          { key: 'L', name: 'An LLM call', col: C.pink },
        ];
        const QS = [
          { label: 'What does one input look like?', opts: [
            ['a row in a table', { R: 1, C: 3, N: -1, L: -2 }, 'Rows of numbers and categories are where classical models are strongest.'],
            ['free text', { R: -1, C: 0, N: 1, L: 3 }, 'Free text is what an LLM was pretrained on.'],
            ['images or audio', { R: -3, C: -2, N: 3, L: 1 }, 'Pixels and sound need learned features, which is a neural net\'s trade.'],
          ] },
          { label: 'How many labelled examples do you have?', opts: [
            ['fewer than 100', { R: 2, C: -1, N: -3, L: 3 }, 'With under 100 labels there is nothing to train on; a rule or an instructed LLM needs none.'],
            ['hundreds to a few thousand', { R: 0, C: 2, N: -1, L: 1 }, 'A few hundred to a few thousand rows is where the simple model beat the net in the opening demo.'],
            ['tens of thousands or more', { R: -1, C: 1, N: 2, L: 0 }, 'With tens of thousands of labels a flexible model has enough data to pay for its knobs.'],
          ] },
          { label: 'Must each decision come with its reasons?', opts: [
            ['no', { R: 0, C: 0, N: 0, L: 0 }, ''],
            ['yes', { R: 2, C: 2, N: -2, L: -3 }, 'You owe reasons, so you need a model whose arithmetic you can show.'],
          ] },
          { label: 'How long may one answer take?', opts: [
            ['several seconds', { R: 0, C: 0, N: 0, L: 0 }, ''],
            ['about a second', { R: 0, C: 0, N: 0, L: -1 }, 'A one-second limit leaves an LLM call no slack.'],
            ['under 50 milliseconds', { R: 2, C: 2, N: 0, L: -4 }, 'Under 50 ms rules out a call to a large model whatever it costs.'],
          ] },
          { label: 'How many requests a day?', opts: [
            ['hundreds', { R: 0, C: 0, N: 0, L: 1 }, 'At hundreds a day, any per-call price is small change.'],
            ['around a hundred thousand', { R: 0, C: 0, N: 0, L: -1 }, 'At a hundred thousand a day the per-call price starts to show.'],
            ['tens of millions', { R: 1, C: 1, N: 0, L: -4 }, 'At tens of millions a day, a per-call price is multiplied into a headline cost.'],
          ] },
          { label: 'Will the instructions change often?', opts: [
            ['rarely', { R: 0, C: 0, N: 0, L: 0 }, ''],
            ['every few weeks', { R: 1, C: -1, N: -2, L: 2 }, 'Instructions that change every few weeks favour a tool you can re-instruct in a sentence instead of retraining.'],
          ] },
        ];
        const PRESETS = { 'Loan approvals': [0, 1, 1, 1, 0, 0], 'Support inbox': [1, 0, 0, 0, 0, 1], 'Keyboard autocomplete': [1, 2, 0, 2, 2, 0], 'Photo moderation': [2, 2, 0, 1, 1, 0] };
        let pick = PRESETS['Loan approvals'].slice();
        const out = h('div', {});
        const ro = ctx.readout();
        const qBox = h('div', { class: 'controls', style: { marginBottom: '12px' } });
        function buildQs() {
          qBox.innerHTML = '';
          QS.forEach((q, qi) => {
            const sel = h('select', {}, q.opts.map((o, oi) => h('option', { value: String(oi), selected: oi === pick[qi] }, o[0])));
            sel.addEventListener('change', () => { pick[qi] = +sel.value; draw(); });
            qBox.append(h('div', { class: 'control' }, h('label', {}, (qi + 1) + '. ' + q.label), sel));
          });
        }
        function draw() {
          const score = { R: 0, C: 0, N: 0, L: 0 }, why = [];
          QS.forEach((q, qi) => { const o = q.opts[pick[qi]]; for (const k in o[1]) score[k] += o[1][k]; if (o[2]) why.push({ text: o[2], pts: o[1] }); });
          const ranked = TOOLS.slice().sort((a, b) => score[b.key] - score[a.key]);
          const top = ranked[0], second = ranked[1];
          out.innerHTML = '';
          TOOLS.forEach((t) => {
            const s = score[t.key], w = Math.abs(s) / 10 * 50;
            out.append(h('div', { style: { display: 'flex', alignItems: 'center', gap: '10px', margin: '6px 0', fontSize: '14px' } },
              h('div', { style: { flex: '0 0 44%', color: t === top ? '#e6ebf5' : '#94a3b8', fontWeight: t === top ? '700' : '400' } }, t.name),
              h('div', { style: { flex: '1', position: 'relative', height: '14px', background: '#0f1520', borderRadius: '3px' } },
                h('div', { style: { position: 'absolute', left: '50%', top: '0', bottom: '0', width: '1px', background: '#94a3b8' } }),
                h('div', { style: { position: 'absolute', top: '0', bottom: '0', left: (s >= 0 ? 50 : 50 - w) + '%', width: w + '%', background: t.col, borderRadius: '2px' } })),
              h('div', { style: { flex: '0 0 34px', textAlign: 'right', fontWeight: '700', color: t.col } }, (s > 0 ? '+' : s < 0 ? '−' : '') + Math.abs(s))));
          });
          const notes = [];
          if (score[top.key] - score[second.key] <= 1) notes.push('<b>It is close.</b> ' + second.name + ' is within a point. Build both and let the held-out score decide.');
          if (top.key === 'L' && (pick[4] > 0 || pick[3] > 0)) notes.push('<b>Plan the exit.</b> Volume or the clock will squeeze this later: log every LLM answer now, and train a small model on those answers when it does.');
          if (top.key !== 'L' && top.key !== 'R' && pick[1] === 0) notes.push('<b>You have too few labels to train it yet.</b> Have an LLM or a person label a few thousand examples first.');
          if (pick[2] === 1 && top.key === 'C') notes.push('<b>Keep it readable.</b> Logistic regression or a shallow tree, not a forest of hundreds.');
          out.append(h('div', { class: 'callout key', style: { marginTop: '12px' } },
            h('div', { class: 'callout-title' }, 'Start with: ' + top.name + ' (' + (score[top.key] > 0 ? '+' : '') + score[top.key] + ')'),
            h('div', { html: '<b>First, whatever wins:</b> write down the dumbest baseline and its score.<br>' +
              (why.length ? why.map((x) => '• ' + x.text + ' <span style="color:#94a3b8">[' + TOOLS.map((t) => t.key + ' ' + (x.pts[t.key] > 0 ? '+' : x.pts[t.key] < 0 ? '−' : '') + Math.abs(x.pts[t.key])).join(', ') + ']</span>').join('<br>') : '• None of your answers pushes either way.') +
              (notes.length ? '<br>' + notes.join('<br>') : '') })));
          ro.set({ rule: score.R, 'classical model': score.C, 'neural net': score.N, 'LLM call': score.L, 'start with': top.name.replace(/^An? /, '') });
        }
        buildQs(); draw();
        const btns = Object.keys(PRESETS).map((name) => ctx.button(name, () => { pick = PRESETS[name].slice(); buildQs(); draw(); }));
        return ctx.figure(h('div', {}, qBox, out), 'Six facts about a project, turned into points for four kinds of tool. The bracket after each reason shows the points that answer gave: R for a rule, C for a classical model, N for a neural net you train yourself, L for an LLM call. The weights are this course\'s rule of thumb, written down so you can argue with them. A guide like this picks where to start. The held-out score picks what ships.', btns, ro);
      }

      /* ================================================================== */
      /* The chapter                                                         */
      /* ================================================================== */
      root.append(
        callout('tryit', '🖐 Do this first: run the head-to-head',
          `Two models learn to predict who will pay a loan late. Both have already trained on 100 past applicants.<br>
           <b>1.</b> Read the two big numbers. The 9-knob model is ahead by <b>4.9 points</b>.<br>
           <b>2.</b> Drag <b>training rows</b> to 300. Both improve; the simple model still leads, by 2.9.<br>
           <b>3.</b> Drag it to 3,000. The order has flipped: the neural net wins by 3.0.<br>
           <b>4.</b> Press <b>Train every size</b> and find where the pink line crosses the green one.`),
        headToHead(),
        p(`Nothing about either model changed as you dragged. The only thing that changed was how much data they were given. With 100 rows the model with 9 adjustable numbers beat the model with 161. With 3,000 rows it lost. On this sample the lines crossed between 300 and 1,000 rows.`),
        p(`One sample proves little, and this first one flatters the simple model. Drag back to 100 rows and press <b>New sample of applicants</b>: on sample 2 the net is ahead by 1.1. Keep pressing. Across the twenty samples the simple model leads in 18, by 2.3 points on average, not 4.9. At 30 rows it leads in only 13 and the gap swings by several points a press, because both models are mostly reporting the luck of the draw. The grey line is what you score by building no model at all.`),
        p(`The contest was not quite even, either. Set <b>weight decay on the net</b> to <b>same as the simple model</b> and press <b>Train every size</b>. At 100 rows the restrained net is now level with the simple model or a little ahead on most samples. At 3,000 its three-point win has gone: the restraint that stopped it chasing luck also flattened the bends it needed. So the rule is narrower than "simple wins". With few rows a well-set net and a simple model land in about the same place, and the simple model gets there with no dial to set. Set the dropdown back to <b>none</b> before reading on.`),
      );

      root.append(section('Why the smaller model won',
        p(`A table like this one, with a row per case and a column per fact, is called <em>tabular data</em>. Most business data has this shape: customers, transactions, patients, shipments. The chart you just filled in, score against amount of training data, is a <em>learning curve</em>, and reading one is the fastest way to tell whether more data or a different model is the fix.`),
        p(`One sample of applicants cannot tell you why the lines crossed. Twenty can.`),
        callout('tryit', '🖐 Try this: twenty samples each',
          `<b>1.</b> Leave the size at 300 rows and press <b>Run 20 samples</b>. Forty models train while you watch.<br>
           <b>2.</b> Compare the two clouds. Green sits further right, at an average of <b>80.7%</b>, and is tighter: spread ±0.6 against the net's ±1.1. The readout says the simple model won <b>20 of 20</b>.<br>
           <b>3.</b> Choose <b>30 rows</b> and run again. Both clouds smear across the chart, and the simple model wins only 13 of 20.<br>
           <b>4.</b> Choose <b>3,000 rows</b> and press it once more. The clouds have tightened to a narrow column. Green has stopped well short of the yellow line. Pink has gone past green and sits close to the line.`),
        twentyDraws(),

        /* maths beat: bias and variance, read off the two clouds */
        ctx.walkthrough([
          { say: 'The dashed yellow line, the hidden rule itself, scores 85.6%, and nothing trained on this data can beat it.', math: 'ceiling = 85.6%' },
          { say: 'At 3,000 rows logistic regression averaged 81.5% with a spread of only ±0.2, so it lands in the same place every time, and that place is 4.1 points short.', math: '85.6 &minus; 81.5 = 4.1 points short, every time' },
          { say: 'A miss that is the same whatever sample you draw has a name: <b>bias</b>.', note: 'It comes from the model\'s shape. A weighted sum cannot draw the two bends in the hidden rule, and no amount of data will teach it to.' },
          { say: 'The neural net at 3,000 rows averaged 84.3%, so its bias is far smaller.', math: '85.6 &minus; 84.3 = 1.3 points short' },
          { ask: 'Now go back to 300 rows, where logistic regression averaged 80.7%. Its total miss is 85.6 &minus; 80.7 = 4.9 points. If 4.1 of that is bias, how much is left over?',
            options: ['0.8 points', '4.1 points', '9.0 points'], answer: 0,
            explain: '4.9 − 4.1 = 0.8. That remainder is not built into the model. It comes from having only 300 rows to set the weights with.' },
          { say: 'That leftover is the price of <b>variance</b>: the model you get changes with the sample you happened to draw, and a model that wobbles lands, on average, further from the target.', note: 'The width of a cloud is the visible sign of the wobble; the leftover is what it costs. We are using the 3,000-row score as a stand-in for unlimited data. Logistic regression has all but flattened by then. The net is still climbing, so its true bias is smaller than the 1.3 above and its variance larger than the next step says.' },
          { say: 'Do the same sum for the neural net at 300 rows, where it averaged 78.1%.', math: '85.6 &minus; 78.1 = 7.5 short &nbsp;=&nbsp; 1.3 bias + 6.2 variance' },
          { ask: 'At 300 rows, which model has less bias, and which one won?',
            options: ['The net has less bias, and the net won', 'The net has less bias, but logistic regression won', 'Logistic regression has less bias, and it won'], answer: 1,
            explain: 'The net\'s bias is 1.3 against 4.1. But its variance is 6.2 against 0.8, so its total miss is 7.5 against 4.9. Low bias is worth nothing if variance eats it.' },
          { say: 'Variance is high when there are few rows per knob: 300 rows share out as 33 for each of logistic regression\'s 9 knobs, and fewer than 2 for each of the net\'s 161.', math: '300 &divide; 9 = 33 &nbsp;&nbsp;&nbsp; 300 &divide; 161 = 1.9', note: 'Rows per knob is a rough guide, not a law. Weight decay and chapter 3\'s other restraints let a model carry more knobs than its rows suggest, as the dropdown in the opening demo showed.' },
          { ask: 'You have 500 rows and the simple model is losing mostly to bias. What helps?',
            options: ['Collecting ten times more rows for the same model', 'A more flexible model, or better columns', 'Training the same model for longer'], answer: 1,
            explain: 'Bias is the part more data does not fix: logistic regression gained less than a point between 300 and 3,000 rows. You need a model or a column that can express the missing pattern.' },
          { say: 'So every model\'s miss is two misses added together, and the rows you have decide which one you can afford.', math: 'miss = bias + variance' },
        ], {
          title: 'From two clouds of dots to bias + variance',
          recap: 'a model misses by its bias, which data cannot fix, plus its variance, which shrinks as rows per knob grow.',
        }),
        ctx.decoder([
          { sym: 'miss', name: 'how far short of the best possible score', says: 'The gap between a model\'s score on unseen data and the score of the rule that generated the data.', points: 'the distance from a cloud to the dashed yellow line.' },
          '=',
          { sym: 'bias', name: 'bias', says: 'The miss that stays when data is unlimited. It belongs to the shape of the model.', points: 'the green cloud at 3,000 rows: tight, and still 4.1 points left of the yellow line.' },
          '+',
          { sym: 'variance', name: 'variance', says: 'The extra miss that comes from learning on one finite sample: the fitted model wobbles from sample to sample, and the wobble costs accuracy. It shrinks as rows per knob grow. Strictly, this is the cost of variance measured in accuracy points. A statistics text defines both words for squared error, where error = bias&sup2; + variance + noise, and calls the two parts used here approximation error and estimation error. The idea is the same.', points: 'how far a cloud\'s average sits below where that model ends up at 3,000 rows. The width of the cloud is the wobble you can see.' },
        ], {
          title: 'miss  =  bias + variance',
          plain: 'a model falls short partly because of what it cannot express and partly because of the sample it happened to learn from.',
        }),

        p(`Chapter 3 showed you this trade inside one model, as the U-curve you got by turning up a polynomial's degree. Here it decides between two models. The neural net can express almost anything, so it expresses the accidents of 300 applicants along with the pattern. The <em>bias–variance trade-off</em> says you cannot dodge both misses with a fixed amount of data. Weight decay is a dial on that trade: switched on in the opening demo, it cut the net's variance and raised its bias until the net behaved like the simple model.`),
        p(`Junk columns make it worse. Four of the eight columns here mean nothing, and a flexible model with few rows will find patterns in house numbers. Logistic regression is tempted too, but it has one knob per junk column to waste. The net has sixteen.`),
        callout('history', '📜 1992: the dilemma gets its name. 2022: the benchmark',
          `In 1992 Stuart Geman, Elie Bienenstock and René Doursat published "Neural Networks and the Bias/Variance Dilemma" in <i>Neural Computation</i>. It put neural nets into the language statisticians already used for every other curve-fitter, and argued that their flexibility was a cost as well as a strength.<br>
           Thirty years on, Léo Grinsztajn, Edouard Oyallon and Gaël Varoquaux tested the question on 45 real tables. Their 2022 paper found that tree-based models, the forests and boosted trees of chapter 16, still beat deep learning on medium-sized tabular data of around 10,000 rows, and that neural nets were hurt in particular by uninformative columns.`),
        p(`No tree model ran in the head-to-head. The net faced only the simplest classical model there is, and against a boosted forest, on a real table, its win at 3,000 rows would be far from assured. The win was also built in: the demo's hidden rule has two smooth bends, which suit a net. Had it been a plain weighted sum, the net would have had nothing to gain at any size. Real tables are often jagged in ways that suit trees.`),
      ));

      root.append(section('Beat the dumbest thing first',
        callout('tryit', '🖐 Try this: write a one-line model by hand',
          `The yellow line is a rule: refuse everyone on the shaded side.<br>
           <b>1.</b> It starts at "refuse anyone with under 4.5 years in their job" and scores <b>70.6%</b>. Drag <b>refuse beyond</b> and try to beat that. The best on offer is 70.9%.<br>
           <b>2.</b> Read the right-hand bars. Refusing nobody scores 60.8%. The neural net, with 3,000 rows and 161 weights, scores 84.5%.<br>
           <b>3.</b> Switch the column to <b>house number</b> and drag the threshold anywhere you like. Press <b>Flip the rule</b>. You will not get clear of the baseline.<br>
           <b>4.</b> Try <b>existing debt</b> and find its best threshold. It tops out near 64.5%: a real signal, and a weaker one than job tenure.`),
        baselineLadder(),
        p(`Your one line of code, with no training at all, collected about two fifths of everything the neural net achieved over doing nothing. That is common. The first sensible rule often captures a large share of the available gain, and each step up in machinery tends to buy less than the one before.`),
        p(`The grey bar has a name: a <em>baseline</em>, the score of the laziest method that could possibly work. For a yes/no question it is the <em>majority class</em>: always answer whichever is commoner. For a forecast it is the last value: tomorrow's sales will equal today's. For anything a person already does, it is the rule of thumb they use now.`),
        p(`A model's score means nothing until it stands next to a baseline. "84.5% accurate" sounds strong. "23.7 points better than refusing nobody, and 13.9 better than one line about job tenure" tells you what the model is worth, and whether it is worth maintaining.`),
        callout('example', '🌍 Baselines that are hard to beat',
          `Weather forecasters score themselves against <b>persistence</b>: the forecast that tomorrow will be the same as today. It is right often enough that a forecasting method has to prove it adds skill on top.<br>
           A shop's sales forecast is judged against "the same weekday last week". Chapter 3's spam dial showed the trap from the other side: when 95% of mail is fine, a filter that flags nothing is 95% accurate, so a 96% model has added one point, not ninety-six.`),
        p(`House number was the informative failure. No threshold rescues a column with no signal in it, and no model does either; a flexible one will only appear to on the training rows. Checking columns one at a time against the baseline, before any modelling, tells you where the signal lives and costs an afternoon.`),
      ));

      root.append(section('A decision you can read',
        callout('tryit', '🖐 Try this: explain a refusal',
          `This applicant has 2 years in their job, £24k of debt, and wants £16k. Both models refuse.<br>
           <b>1.</b> On the left, read the reason straight off the bars. The largest red one is the short time in the job, at <b>+1.53</b>; the debt adds +0.35. Drag <b>existing debt</b> down until the left verdict turns to APPROVE, at £8k. That is one answer to "what would I have to change?".<br>
           <b>2.</b> Press <b>Reset applicant</b>. Now look right and try to find the reason in the coloured grid.<br>
           <b>3.</b> Bottom right: £5k more debt adds <b>+0.43</b> to the logistic score for anyone. For the net it adds +1.04 for this person.<br>
           <b>4.</b> Drag <b>loan size asked for</b> down to 3. The net's number goes to −0.33: more debt now makes this applicant <i>safer</i>, and the net approves.`),
        readDecision(),
        p(`The left-hand model can be printed on a card: nine numbers, one weight per column and a starting score. Ask it why it refused and the answer is the model itself. Being able to follow a model's arithmetic from input to decision is called <em>interpretability</em>, and a weighted sum has it by construction. So does a small decision tree, which reads as a list of yes/no questions.`),
        p(`The right-hand model is a <em>black box</em>: every number inside is visible and none of them is a reason. The only way to learn what debt does is to nudge it and watch, and the answer you get is true for this applicant only. At a £3k loan it reversed. In this made-up world that reversal really is in the data, which is part of why the net scored higher. Whether a lender could stand behind it is another question, and with the net nobody would have known to ask.`),
        p(`Tools exist that automate the nudging and report which inputs mattered most for one prediction; SHAP and LIME are the best known. They are useful and they are estimates. They describe the black box from outside, one case at a time, and two of them can disagree.`),
        callout('example', '🌍 Who asks for the reason',
          `<b>The customer who was refused.</b> In the United States, the Equal Credit Opportunity Act's Regulation B requires a lender that turns down an application to state the specific, principal reasons for the refusal, or to supply them when the applicant asks. "The algorithm said no" does not meet that.<br>
           <b>The regulator or auditor</b>, who wants to know the model does not lean on something it must not, and wants the same answer next month.<br>
           <b>Your own team</b>, at three in the morning, working out why approvals halved after a data feed changed.`),
        p(`The price of readability in this demo was three points of accuracy at 3,000 rows. Sometimes that is worth paying and sometimes the law has already decided for you. In lending, insurance, hiring and medicine, start from the model you can read and make anything more complex earn its place.`),
      ));

      root.append(section('What it costs to answer',
        callout('tryit', '🖐 Try this: do the multiplication',
          `It opens on a support inbox: 2,000 requests a day, and half a minute is fine.<br>
           <b>1.</b> Read the LLM row: <b>$180</b> a month, in time. Nobody will argue about that bill.<br>
           <b>2.</b> Press <b>LLM on every keystroke</b>. Fifty million requests a day, each needed in 100 ms. The LLM row now reads <b>$4.5M</b> a month and 15× too slow.<br>
           <b>3.</b> Read the small-model row in the same state: $7,500, with time to spare.<br>
           <b>4.</b> Drag <b>time allowed per answer</b> up to 1.6 s. Both LLM rows turn green together. The blend is no faster on its hard cases; what it saves is money, at about a twentieth of the LLM's bill.`),
        budgetCalc(),
        p(`Two numbers decide most tool choices before accuracy gets a vote. The first is <em>latency</em>: the time from request to answer. A <em>latency budget</em> is the most the product can tolerate, and it behaves like a wall. A fraud check that takes two seconds to approve a card payment is a broken product, however accurate it is.`),
        p(`The second is the bill, which is one multiplication: requests per day × price per request × 30. A price that looks like nothing, a third of a cent, becomes $4.5 million a month at fifty million requests a day. The same price at 2,000 a day is $180. Volume turns a rounding error into a budget line, and it is the number people forget to ask for.`),
        p(`The fourth row is a common compromise called a <em>cascade</em>: a cheap model answers everything it is confident about and hands the rest to an expensive one. It cut the bill about twenty-fold here. It only helps the clock if a slow answer on the hard cases is acceptable.`),
        callout('warning', '⚠️ The prices are placeholders',
          `Every per-request figure in that calculator is an illustrative round number. Real prices depend on the model, the length of the prompt and the provider, and they change. Chapter 12 showed how an LLM bill is built up from tokens. What does not change is the shape: several factors of ten between a rule, a small model and a large one.`),
      ));

      root.append(section('When the LLM is the right tool',
        p(`Everything so far has gone against the big model, so here is its home ground. An LLM is the right first tool when the input is language, when you have a handful of examples instead of thousands, and when what you want can be said in a sentence that may change next week.`),
        p(`"Sort these emails into complaints, questions and praise" needs no training set at all. Put three examples in the prompt, the few-shot prompting of chapter 10, and it improves again. The model arrives already knowing what a complaint sounds like. A classical model would need thousands of labelled emails to learn that, and a new batch every time you added a category.`),
        p(`Then the product succeeds, volume climbs, and the calculator starts to hurt. There is a standard move for that moment.`),
        callout('tryit', '🖐 Try this: buy labels, then stop paying',
          `<b>1.</b> It opens with the LLM having labelled only 5 tickets. The small model trained on them scores <b>51.0%</b>, when guessing would get a third.<br>
           <b>2.</b> Drag <b>tickets labelled by the LLM</b> to 300. The small model reaches 90.8%, and the labels cost $0.90.<br>
           <b>3.</b> Drag to 3,000: <b>92.4%</b>, for a one-off $9.00. The LLM itself scores 94.3%.<br>
           <b>4.</b> Press <b>Show a mistake</b> a few times. The first is ticket 25. Read each one: could you have sorted it?`),
        distil(),
        p(`You used the expensive model as a teacher. It labelled a few thousand examples once, and a model small enough to run for almost nothing learned from those labels. It is a form of what chapter 12 called <em>distillation</em>. The small model kept most of the accuracy and dropped the bill by a factor of six hundred, and the labelling cost less than a sandwich.`),
        p(`It did not keep all of it, and here the simulation needs a warning. About 3% of these made-up tickets contain no word from their own topic, like ticket 25, and the stand-in teacher gets them right only because it was built to. Its 5% of wrong labels did no visible harm: scattered mistakes average out. With a real LLM the gap has a real cause. A word-counter ignores word order and meaning, and the LLM does not. That is the small model's bias.`),
        p(`Whether two points matter depends on what a wrong answer costs, which is why the cascade exists. Send the tickets the small model is unsure about to the teacher, and keep the cheap path for the rest.`),
        callout('example', '🌍 The pattern in practice',
          `<b>Prototype with the LLM.</b> A prompt and an afternoon tell you whether the task is possible at all and what "good" looks like.<br>
           <b>Log everything.</b> Every input and every LLM answer is a labelled example you did not have to pay a person for.<br>
           <b>Distil when the sums say so.</b> Train something small on the log, check it against a held-out set a person has labelled, and route only the hard cases upward.`),
      ));

      root.append(section('A guide you can drive',
        callout('tryit', '🖐 Try this: put your own project in',
          `<b>1.</b> It opens on <b>Loan approvals</b>: a table, a few thousand rows, reasons required. A classical model leads on <b>+7</b>, and an LLM call is on −4.<br>
           <b>2.</b> Press <b>Support inbox</b>. Free text, almost no labels, instructions that change: the LLM leads on +9.<br>
           <b>3.</b> Still on Support inbox, change question 5 to <b>tens of millions</b>. The LLM's score falls from +9 to +4, one point ahead of a hand-written rule. Read the new advice.<br>
           <b>4.</b> Now answer the six questions for something you work on.`),
        guide(),
        p(`The guide is itself the kind of model this chapter has been recommending: a points table you can read, argue with and change. It is also a baseline. If your instinct disagrees with it, find the answer that carries the disagreement and ask whether the points are wrong or your instinct is.`),
        p(`It leaves out two things that often settle the matter. One is what your team can run: a model nobody can maintain is the wrong tool whatever it scores. The other is the cost of a wrong answer. A misfiled ticket and a wrongly refused mortgage are not the same size of mistake, and the second deserves the slower, plainer, better-understood tool.`),
      ));

      root.append(section('Why this matters for modern AI work',
        p(`Chapters 7 to 12 could leave you believing every problem now ends in a transformer. A great deal of the machine learning running in production is still a rule, a regression or a forest, and the people who build with LLMs for a living spend much of their time deciding where <i>not</i> to put one.`),
        p(`A working system is usually a mixture. A rule rejects malformed input. A small classifier routes the request. An LLM handles the part that really is language, with a cheap model checking its output or taking over the easy cases. Each piece is chosen with the questions you just drove: what shape is the data, how much of it is labelled, who needs the reasons, how fast, how many.`),
        p(`The habits transfer directly to LLM work. A prompt needs a baseline, usually the simplest prompt you could write. A fine-tuned model has a learning curve, and it tells you whether to collect more examples. An agent has a latency budget and a cost per task. The next three chapters turn these into practice: where the data comes from, how to measure a system you can trust, and how to keep it alive after launch.`),
      ));

      root.append(section('Go deeper',
        ul([
          `<a href="https://arxiv.org/abs/2207.08815" target="_blank" rel="noopener">Grinsztajn, Oyallon &amp; Varoquaux (2022), "Why do tree-based models still outperform deep learning on typical tabular data?"</a> The 45-dataset benchmark from the history box.`,
          `<a href="https://developers.google.com/machine-learning/guides/rules-of-ml" target="_blank" rel="noopener">Martin Zinkevich, "Rules of Machine Learning"</a>. Google's engineering checklist. Rule 1 is not to be afraid to launch a product without machine learning.`,
          `<a href="https://arxiv.org/abs/1811.10154" target="_blank" rel="noopener">Cynthia Rudin (2019), "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead"</a>. The case for readable models, argued by someone who builds them.`,
          `<a href="https://arxiv.org/abs/1503.02531" target="_blank" rel="noopener">Hinton, Vinyals &amp; Dean (2015), "Distilling the Knowledge in a Neural Network"</a>. The paper that named the teacher-and-student trick.`,
          `<a href="https://scikit-learn.org/stable/machine_learning_map.html" target="_blank" rel="noopener">scikit-learn, "Choosing the right estimator"</a>. A flowchart version of this chapter's guide for classical models, from the library you would build them with.`,
        ]),
      ));

      root.append(ctx.quiz([
        { q: 'You have 40 example emails, the categories to sort them into change every month, about 300 arrive a day, and a reply within a few seconds is fine. What do you build first?',
          options: [
            'A neural net trained from scratch on the 40 examples',
            'Logistic regression on word counts, retrained each month',
            'An LLM with the categories written in the prompt, logging every answer',
            'Nothing yet: have several thousand emails labelled by hand, then train a classifier on them',
          ], answer: 2,
          explain: 'Free text, almost no labels, changing instructions, low volume and a relaxed clock: every fact points at the LLM. Forty examples cannot train anything, and 300 calls a day is small change. Logging the answers keeps the door open to distil later.' },
        { q: 'A vendor reports that their fraud model is 98.5% accurate. In your data, 1 transaction in 100 is fraud. What does the "never fraud" baseline score, and what follows?',
          options: [
            '99%, so on accuracy the model is worse than doing nothing',
            '1%, so the model is a huge improvement',
            '50%, so the model is roughly twice as good as tossing a coin for each transaction',
            '98.5%, so the model exactly matches the baseline',
          ], answer: 0,
          explain: 'Always answering "not fraud" is right for the 99 in 100 that are not fraud. A 98.5% model is half a point below that. Accuracy is the wrong measure here (chapter 3\'s precision and recall are the right ones), but the baseline is what exposed it.' },
        { q: 'A feature will be called 2 million times a day. One LLM call costs $0.002. What is the monthly bill (30 days) for calling the LLM every time?',
          options: ['$4,000', '$12,000', '$40,000', '$120,000'], answer: 3,
          explain: '2,000,000 × $0.002 = $4,000 a day, and 30 days of that is $120,000. $4,000 is the daily figure, which is the commonest slip.' },
        { q: 'On a table of 150 rows, a neural net scores 99% on its training rows and 71% on held-out rows. Logistic regression scores 78% and 76%. Which diagnosis fits?',
          options: [
            'The net is losing to bias: it is too rigid for this data',
            'The net is losing to variance: too many knobs for 150 rows',
            'Logistic regression is overfitting and its 76% cannot be trusted',
            'Both models are underfitting, so both need more training steps',
          ], answer: 1,
          explain: 'A 28-point gap between training and held-out scores is the signature of a model fitting the luck of its sample. 150 rows cannot pin down hundreds of weights. Logistic regression\'s two scores are close together, so its 76% is honest.' },
        { q: 'A logistic model refuses a loan when the score is above 0. For one applicant the contributions are: starting score −0.6, debt +1.2, loan size +0.3, years in job −0.4. Every £5k less debt lowers the score by 0.6. How much debt must go for approval?',
          options: ['None: the score is already below 0', 'Paying off £5k is enough', 'It takes £10k: £5k still leaves the score above 0', 'It cannot be worked out without probing the model'], answer: 1,
          explain: 'The score is −0.6 + 1.2 + 0.3 − 0.4 = +0.5. One £5k step takes off 0.6 and leaves −0.1, which is below the line. Being able to do this sum for the customer is what a readable model gives you.' },
      ]));
    },
  });
})();
