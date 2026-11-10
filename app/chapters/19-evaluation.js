/* Zero → AGI · Chapter 19 · Evaluation you can trust
   DESIGN RULE: the reader ships the wrong prompt on a 20-question eval before any statistics
   appear, and every later section is one more way a score can lie to them, met by hand first.
   Interactives, in order: draw-an-eval-set (the winner flips at small sizes, the opener);
   200 eval sets and the band they fall in (carries the maths beat: the standard error of a
   proportion); separate samples against the same items (paired comparison); a topic slice
   that reverses the overall winner; an A/B test with nothing to find that the reader can stop
   early; a regression suite where a patch fixes three cases and breaks two; a biased LLM
   judge unmasked by swapping the order; a support-bot dial that maximises a metric while the
   customers leave.
   Every number the prose quotes comes from a seeded generator, so it is the same on every load. */
(function () {
  ZTA.registerChapter({
    id: '19-evaluation',
    num: 19,
    part: 'VI',
    title: 'Evaluation you can trust',
    tagline: '"Did my change help?" is the question every ML job comes down to. Ship the wrong prompt once, by hand, and then learn the six ways a score lies.',
    minutes: 55,
    objectives: [
      'put a 95% interval on an eval score from its size, and say whether a gap between two versions is real',
      'design an eval set: what goes in it, how many items, which slices to report, and which part you never tune on',
      'run a comparison that can be trusted, offline on the same items and online without peeking',
      'spot a biased LLM judge, a regression hidden by a rising score, and a metric that is being gamed',
    ],
    takeaways: [
      '<b>A score is a sample.</b> Its standard error is √(p(1 − p) ÷ n), and 95 times in 100 the score lands within about two of those of the truth. For a score near 70% that is ±9 points at 100 items and ±3 at 1,000.',
      '<b>Same items for both versions.</b> Most questions are easy for both or hard for both, so pairing cancels that shared luck. In the demo it was worth four times the data; the more alike the two versions, the more it buys.',
      '<b>Look inside the average.</b> One number can rise while every slice falls, if the mix of questions changed. Report the slices you care about, each with its own count.',
      '<b>Offline is the rehearsal, online is the show.</b> An A/B test measures what users do. Fix its length in advance: stopping at the first good day manufactures wins.',
      '<b>Every fixed bug becomes a test.</b> A regression suite is read case by case, because a patch can fix three things, break two, and still raise the score.',
      '<b>A judge is a model, so evaluate it.</b> Swap the order, control the length, check it against human labels. And once any metric becomes the target, watch what it stopped measuring.',
    ],
    render(root, ctx) {
      const { h, p, section, callout, ul } = ctx;
      const C = ctx.colors;
      const FONT = '13px Inter, system-ui, sans-serif';
      const BOLD = 'bold 13px Inter, system-ui, sans-serif';
      const BIG = 'bold 15px Inter, system-ui, sans-serif';
      const MONO = '12px "JetBrains Mono", ui-monospace, monospace';

      /* mulberry32: every dataset in this chapter is drawn from it */
      function rng(seed) {
        let a = seed >>> 0;
        return () => { a += 0x6D2B79F5; let t = a; t = Math.imul(t ^ (t >>> 15), 1 | t); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      }
      function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
      const sig = (x) => 1 / (1 + Math.exp(-x));
      function wrapLines(gc, text, maxW) {
        const words = String(text).split(' '); const out = []; let line = '';
        for (const w of words) { const t = line ? line + ' ' + w : w; if (line && gc.measureText(t).width > maxW) { out.push(line); line = w; } else line = t; }
        if (line) out.push(line);
        return out;
      }
      const pts = (v, d) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d == null ? 1 : d);
      function panel(g, P) { g.fillStyle = '#0f1520'; g.fillRect(P.x, P.y, P.w, P.h); g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h); }

      /* ------------------------------------------------------------------
         The question pool shared by the first three demos: 5,000 customer
         questions, each with a hidden difficulty. Prompt A and prompt B each
         get every question right or wrong once, for good. Both find the same
         questions hard, which is what makes pairing work later. The counts
         are then nudged to exactly 3,500 and 3,700 right, so the truth is a
         clean 70.0% and 74.0%.
         ------------------------------------------------------------------ */
      const POOL = 5000;
      const poolA = [], poolB = [];
      (function () {
        const r = rng(20);
        for (let i = 0; i < POOL; i++) { const d = r(); poolA.push(r() < sig(20 * (0.70 - d)) ? 1 : 0); poolB.push(r() < sig(20 * (0.74 - d)) ? 1 : 0); }
        const fix = (arr, target) => {
          let s = arr.reduce((a, b) => a + b, 0);
          for (let i = 0; i < arr.length && s !== target; i++) {
            if (s > target && arr[i] === 1) { arr[i] = 0; s--; } else if (s < target && arr[i] === 0) { arr[i] = 1; s++; }
          }
        };
        fix(poolA, 3500); fix(poolB, 3700);
      })();
      const TRUE_A = 70.0, TRUE_B = 74.0;

      /* ================================================================== */
      /* 1. Draw an eval set: did my change help?                           */
      /* ================================================================== */
      function evalDraw() {
        const W = 720, H = 330;
        const [cv, g] = ctx.canvas(W, H);
        const SIZES = [20, 100, 1000];
        const S = { si: 0, draws: [], reveal: false, dirty: true };
        function drawSet(si, k) {
          const r = rng(357 + si * 100003 + k * 7919), n = SIZES[si];
          let a = 0, b = 0;
          for (let i = 0; i < n; i++) { const j = Math.floor(r() * POOL); a += poolA[j]; b += poolB[j]; }
          return { a: a / n * 100, b: b / n * 100 };
        }
        function more(k) { for (let i = 0; i < k; i++) S.draws.push(drawSet(S.si, S.draws.length + 1)); S.dirty = true; }
        function setSize(si) { S.si = si; S.draws = []; more(1); sizeBtns.forEach((b, i) => b.classList.toggle('primary', i === si)); }
        const ro = ctx.readout();
        const P = { x: 336, y: 50, w: 354, h: 200 };
        const RANGE = 40;
        function draw() {
          g.clearRect(0, 0, W, H);
          const n = SIZES[S.si], cur = S.draws[S.draws.length - 1];
          g.textAlign = 'left'; g.font = BOLD; g.fillStyle = C.text;
          g.fillText('eval set ' + S.draws.length + ': ' + n.toLocaleString() + ' random questions', 30, 26);
          /* the two bars */
          const base = 250, full = 180;
          [['A', cur.a, C.accent, 60, 'A · live prompt', TRUE_A], ['B', cur.b, C.orange, 170, 'B · your rewrite', TRUE_B]].forEach(([, v, col, x, name, truth]) => {
            g.fillStyle = '#0f1520'; g.fillRect(x, base - full, 80, full);
            g.fillStyle = col; g.fillRect(x, base - v / 100 * full, 80, v / 100 * full);
            g.font = BIG; g.textAlign = 'center'; g.fillStyle = C.text; g.fillText(v.toFixed(n >= 1000 ? 1 : 0) + '%', x + 40, base - v / 100 * full - 8);
            g.font = FONT; g.fillStyle = col; g.fillText(name, x + 40, base + 18);
            g.font = MONO; g.fillStyle = C.muted; g.fillText(S.reveal ? 'truly ' + truth.toFixed(1) + '%' : 'truly ?', x + 40, base + 36);
          });
          const d = cur.b - cur.a;
          g.textAlign = 'left'; g.font = BOLD;
          g.fillStyle = Math.abs(d) < 1e-9 ? C.muted : d > 0 ? C.orange : C.accent;
          g.fillText(Math.abs(d) < 1e-9 ? 'This set says: a dead heat' : 'This set says: ' + (d > 0 ? 'B' : 'A') + ' is better by ' + Math.abs(d).toFixed(n >= 1000 ? 1 : 0) + ' points', 30, 316);
          /* history of gaps */
          g.font = BOLD; g.fillStyle = C.text; g.fillText('B − A on every set drawn so far', P.x, 26);
          panel(g, P);
          const yAt = (v) => P.y + P.h / 2 - ctx.clamp(v, -RANGE, RANGE) / RANGE * (P.h / 2 - 6);
          g.font = MONO; g.textAlign = 'right';
          for (const t of [-40, -20, 0, 20, 40]) {
            g.strokeStyle = t === 0 ? C.muted : C.line; g.lineWidth = 1; g.beginPath(); g.moveTo(P.x, yAt(t)); g.lineTo(P.x + P.w, yAt(t)); g.stroke();
            g.fillStyle = C.muted; g.fillText(pts(t, 0), P.x - 6, yAt(t) + 4);
          }
          g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
          if (S.reveal) { g.strokeStyle = C.green; g.setLineDash([5, 4]); g.lineWidth = 1.5; g.beginPath(); g.moveTo(P.x, yAt(TRUE_B - TRUE_A)); g.lineTo(P.x + P.w, yAt(TRUE_B - TRUE_A)); g.stroke(); g.setLineDash([]); }
          const span = Math.max(60, S.draws.length);
          S.draws.forEach((dr, i) => {
            const dd = dr.b - dr.a, x = P.x + 8 + i / (span - 1) * (P.w - 16), y = yAt(dd);
            g.beginPath(); g.arc(x, y, i === S.draws.length - 1 ? 5 : 3.2, 0, Math.PI * 2);
            g.fillStyle = Math.abs(dd) < 1e-9 ? C.muted : dd > 0 ? C.orange : C.accent; g.fill();
            if (i === S.draws.length - 1) { g.strokeStyle = C.text; g.lineWidth = 1.5; g.stroke(); }
          });
          g.restore();
          g.textAlign = 'left'; g.font = MONO; g.fillStyle = C.muted;
          g.fillText('orange: B ahead   blue: A ahead   grey: tied', P.x, P.y + P.h + 20);
          g.fillStyle = S.reveal ? C.green : C.muted;
          g.fillText(S.reveal ? 'green dashed line: the true gap, +4.0 points' : 'the true gap is hidden', P.x, P.y + P.h + 40);
          let bw = 0, aw = 0, tie = 0;
          S.draws.forEach((dr) => { const dd = dr.b - dr.a; if (Math.abs(dd) < 1e-9) tie++; else if (dd > 0) bw++; else aw++; });
          ro.set({ 'eval size': n.toLocaleString(), 'sets drawn': S.draws.length, 'B ahead': bw, 'A ahead': aw, tied: tie });
        }
        const sizeBtns = ['20 items', '100 items', '1,000 items'].map((lab, i) => ctx.button(lab, () => setSize(i), i === 0 ? 'primary' : ''));
        const oneBtn = ctx.button('Draw a new eval set', () => more(1), 'primary');
        const manyBtn = ctx.button('Draw 50 more', () => more(50));
        const revBtn = ctx.button('Reveal the truth', () => { S.reveal = !S.reveal; revBtn.textContent = S.reveal ? 'Hide the truth' : 'Reveal the truth'; S.dirty = true; });
        more(1);
        ctx.loop(() => { if (S.dirty) { S.dirty = false; draw(); } });
        return ctx.figure(cv, 'Two versions of a support bot\'s prompt, each already marked right or wrong on all 5,000 questions in a pool. An eval set is a random handful of those questions, and both prompts answer the same handful. Left: the two scores on the handful just drawn. Right: the gap between them on every handful so far. Neither prompt changes between draws. The pool and the draws are simulated and seeded, so you see the same sets on every visit. The first set was hand-picked to sting: at 20 questions about one set in six puts A ahead, but by as much as 15 points only about one in a hundred. Every set after it is an ordinary draw.', [...sizeBtns, oneBtn, manyBtn, revBtn], ro);
      }

      /* ================================================================== */
      /* 2. 200 eval sets of one model: the band they fall in               */
      /* ================================================================== */
      function spreadLab() {
        const W = 720, H = 290;
        const [cv, g] = ctx.canvas(W, H);
        const SIZES = [20, 100, 1000];
        const S = { si: 1, p: 70, run: 0, scores: [], dirty: true };
        const REPS = 200;
        function sim() {
          const n = SIZES[S.si], q = S.p / 100;
          const r = rng(700001 + n * 13 + S.p * 101 + S.run * 7907);
          S.scores = [];
          for (let k = 0; k < REPS; k++) { let c = 0; for (let i = 0; i < n; i++) if (r() < q) c++; S.scores.push(c / n * 100); }
          S.dirty = true;
        }
        const ro = ctx.readout();
        const P = { x: 40, y: 44, w: 650, h: 176 };
        function draw() {
          g.clearRect(0, 0, W, H);
          const n = SIZES[S.si], q = S.p / 100;
          const se = Math.sqrt(q * (1 - q) / n) * 100;
          const mean = S.scores.reduce((a, b) => a + b, 0) / REPS;
          const sd = Math.sqrt(S.scores.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (REPS - 1));
          const lo = S.p - 2 * se, hi = S.p + 2 * se;
          const inside = S.scores.filter((v) => v >= lo - 1e-9 && v <= hi + 1e-9).length;
          const xAt = (v) => P.x + 5 + v / 100 * (P.w - 10);
          panel(g, P);
          g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
          g.fillStyle = 'rgba(56,217,169,0.13)'; g.fillRect(xAt(lo), P.y, xAt(hi) - xAt(lo), P.h);
          /* each bar is split by the same test the readout counts with, so the red you
             see is exactly the "outside" the readout reports */
          const bins = new Array(101).fill(0), binsOut = new Array(101).fill(0);
          S.scores.forEach((v) => { const k = ctx.clamp(Math.round(v), 0, 100); bins[k]++; if (!(v >= lo - 1e-9 && v <= hi + 1e-9)) binsOut[k]++; });
          const top = Math.max(...bins);
          bins.forEach((c, i) => {
            if (!c) return;
            const bh = c / top * (P.h - 14), oh = binsOut[i] / top * (P.h - 14);
            g.fillStyle = C.accent; g.fillRect(xAt(i) - 2.6, P.y + P.h - bh, 5.2, bh - oh);
            if (oh > 0) { g.fillStyle = C.danger; g.fillRect(xAt(i) - 2.6, P.y + P.h - oh, 5.2, oh); }
          });
          g.strokeStyle = C.green; g.lineWidth = 1.5; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(xAt(S.p), P.y); g.lineTo(xAt(S.p), P.y + P.h); g.stroke(); g.setLineDash([]);
          g.restore();
          g.font = BOLD; g.textAlign = 'left'; g.fillStyle = C.text;
          g.fillText('200 eval sets of ' + n.toLocaleString() + ' items, one model, one bar per score', P.x, 26);
          g.font = MONO; g.fillStyle = C.muted; g.textAlign = 'center';
          for (let t = 0; t <= 100; t += 20) g.fillText(t + '%', xAt(t), P.y + P.h + 16);
          g.font = FONT; g.fillStyle = C.green; g.textAlign = 'left';
          g.fillText('green line: the truth, ' + S.p + '%.  Shaded band: ' + lo.toFixed(1) + '% to ' + Math.min(100, hi).toFixed(1) + '%, predicted to hold about 95 in 100 scores.', P.x, P.y + P.h + 40);
          g.fillStyle = C.muted;
          g.fillText('blue: scores inside the band    red: scores that landed outside it', P.x, P.y + P.h + 60);
          ro.set({ 'eval size': n.toLocaleString(), 'measured spread': sd.toFixed(1) + ' points', 'predicted spread': se.toFixed(1) + ' points', 'inside the band': inside + ' of 200' });
        }
        const setSize = (si) => { S.si = si; sizeBtns.forEach((b, i) => b.classList.toggle('primary', i === si)); sim(); };
        const sizeBtns = ['20 items', '100 items', '1,000 items'].map((lab, i) => ctx.button(lab, () => setSize(i), i === 1 ? 'primary' : ''));
        const pSl = ctx.slider({ label: 'true accuracy', min: 50, max: 95, step: 5, value: 70, fmt: (v) => v + '%', onChange: (v) => { S.p = v; sim(); } });
        const again = ctx.button('Run 200 fresh eval sets', () => { S.run++; sim(); });
        sim();
        ctx.loop(() => { if (S.dirty) { S.dirty = false; draw(); } });
        return ctx.figure(cv, 'One model with a fixed true accuracy, scored on 200 different random eval sets of the same size. Each bar counts how many of those sets gave that score. "Measured spread" is the standard deviation of the 200 scores you are looking at. "Predicted spread" is worked out from the accuracy and the eval size alone, before a single set is drawn; the walkthrough below shows how. The "95 in 100" promise is a rule of thumb that assumes a symmetrical heap. It holds well in the middle of the range and slips where the heap is squashed against 100%: try 95% accuracy at 20 items.', [pSl, ...sizeBtns, again], ro);
      }

      /* ================================================================== */
      /* 3. Separate samples against the same items                         */
      /* ================================================================== */
      function pairedLab() {
        const W = 720, H = 362;
        const [cv, g] = ctx.canvas(W, H);
        const S = { n: 200, run: 0, un: [], pa: [], dirty: true };
        const REPS = 300;
        function sim() {
          const r = rng(500009 + S.n * 7 + S.run * 1009);
          S.un = []; S.pa = [];
          for (let k = 0; k < REPS; k++) {
            let a = 0, b = 0, a2 = 0, b2 = 0;
            for (let i = 0; i < S.n; i++) {
              const j = Math.floor(r() * POOL); a += poolA[j]; b += poolB[j];          // same items for both
              a2 += poolA[Math.floor(r() * POOL)]; b2 += poolB[Math.floor(r() * POOL)]; // a sample each
            }
            S.pa.push((b - a) / S.n * 100); S.un.push((b2 - a2) / S.n * 100);
          }
          S.dirty = true;
        }
        const stats = (arr) => {
          const m = arr.reduce((x, y) => x + y, 0) / arr.length;
          return { sd: Math.sqrt(arr.reduce((x, y) => x + (y - m) * (y - m), 0) / (arr.length - 1)), wrong: arr.filter((v) => v <= 1e-9).length };
        };
        const ro = ctx.readout();
        const X0 = 40, X1 = 700, LO = -20, HI = 28;
        const xAt = (v) => X0 + (v - LO) / (HI - LO) * (X1 - X0);
        function draw() {
          g.clearRect(0, 0, W, H);
          const su = stats(S.un), sp = stats(S.pa);
          const binsOf = (arr) => { const m = new Map(); arr.forEach((v) => { const k = Math.ceil(v / 2 - 1e-9); m.set(k, (m.get(k) || 0) + 1); }); return m; };
          const bu = binsOf(S.un), bp = binsOf(S.pa);
          const top = Math.max(...bu.values(), ...bp.values());
          const rows = [
            ['A and B each scored on their own ' + S.n + ' questions', bu, su, 58],
            ['A and B scored on the same ' + S.n + ' questions', bp, sp, 204],
          ];
          rows.forEach(([title, bins, st, y0]) => {
            const R = { x: X0, y: y0, w: X1 - X0, h: 96 };
            g.font = BOLD; g.textAlign = 'left'; g.fillStyle = C.text; g.fillText(title, X0, y0 - 24);
            g.font = MONO; g.fillStyle = C.muted;
            g.fillText('spread ' + st.sd.toFixed(1) + ' points · A ahead or level in ' + st.wrong + ' of 300', X0, y0 - 8);
            panel(g, R);
            g.save(); g.beginPath(); g.rect(R.x, R.y, R.w, R.h); g.clip();
            bins.forEach((c, k) => {
              const xl = xAt(k * 2 - 2), xr = xAt(k * 2), bh = c / top * (R.h - 8);
              g.fillStyle = k <= 0 ? C.danger : C.green; g.fillRect(xl + 1, R.y + R.h - bh, xr - xl - 2, bh);
            });
            g.strokeStyle = C.muted; g.lineWidth = 1; g.beginPath(); g.moveTo(xAt(0), R.y); g.lineTo(xAt(0), R.y + R.h); g.stroke();
            g.strokeStyle = C.text; g.setLineDash([5, 4]); g.lineWidth = 1.5; g.beginPath(); g.moveTo(xAt(4), R.y); g.lineTo(xAt(4), R.y + R.h); g.stroke(); g.setLineDash([]);
            g.restore();
          });
          g.font = MONO; g.fillStyle = C.muted; g.textAlign = 'center';
          for (const t of [-20, -10, 0, 10, 20]) g.fillText(pts(t, 0), xAt(t), 318);
          g.textAlign = 'left'; g.font = FONT;
          g.fillText('measured gap, B − A, in points.  White dashed line: the true gap, +4.', X0, 338);
          g.fillStyle = C.danger; g.fillText('Red bars: experiments that put A ahead or level, the wrong answer.', X0, 356);
          ro.set({ 'own samples': su.sd.toFixed(1) + ' pts spread, ' + su.wrong + ' wrong', 'same items': sp.sd.toFixed(1) + ' pts spread, ' + sp.wrong + ' wrong', 'pairing is worth': '×' + Math.pow(su.sd / sp.sd, 2).toFixed(1) + ' the data' });
        }
        const nSl = ctx.slider({ label: 'questions per eval set', min: 50, max: 1000, step: 50, value: 200, onChange: (v) => { S.n = v; sim(); } });
        const again = ctx.button('Run 300 fresh experiments', () => { S.run++; sim(); });
        sim();
        ctx.loop(() => { if (S.dirty) { S.dirty = false; draw(); } });
        return ctx.figure(cv, 'The same two prompts and the same pool as the opener, compared 300 times in two ways. Top: two separate random samples, one per prompt. Bottom: one random sample, answered by both. Each bar counts experiments that measured a gap in that 2-point range. "Pairing is worth" is the squared ratio of the two spreads: how many times more questions the top method needs to be as steady as the bottom one. "Wrong" counts every experiment that failed to put B ahead, ties included.', [nSl, again], ro);
      }

      /* ================================================================== */
      /* 4. Slices: the average that reverses                               */
      /* ================================================================== */
      function sliceLab() {
        const W = 720, H = 330;
        const [cv, g] = ctx.canvas(W, H);
        const N = 400;
        /* Failures spaced evenly through each list, so any first-k slice of it sits
           within one ticket of the stated rate and the slider cannot stumble on a
           lucky run. True rates: order status A 92% B 90%; billing disputes A 60% B 52%. */
        function outcomes(rate, phase) {
          const arr = [];
          for (let i = 0; i < N; i++) arr.push(Math.floor((i + 1) * rate + phase) - Math.floor(i * rate + phase));
          return arr;
        }
        const easyA = outcomes(0.92, 0.5), easyB = outcomes(0.90, 0.5), hardA = outcomes(0.60, 0.5), hardB = outcomes(0.52, 0.5);
        const sum = (arr, k) => { let s = 0; for (let i = 0; i < k; i++) s += arr[i]; return s; };
        const S = { share: 85, sliced: false, dirty: true };
        const ro = ctx.readout();
        function draw() {
          g.clearRect(0, 0, W, H);
          const kB = Math.round(S.share / 100 * N), hB = N - kB;
          const A = { easy: sum(easyA, 200) / 200 * 100, hard: sum(hardA, 200) / 200 * 100, all: (sum(easyA, 200) + sum(hardA, 200)) / N * 100 };
          const B = { easy: sum(easyB, kB) / kB * 100, hard: sum(hardB, hB) / hB * 100, all: (sum(easyB, kB) + sum(hardB, hB)) / N * 100 };
          g.font = BOLD; g.textAlign = 'left'; g.fillStyle = C.text;
          g.fillText(S.share === 50 ? 'A and B both scored on the same 400 tickets' : 'A scored on 400 of March\'s tickets, B on 400 of April\'s', 30, 26);
          const base = 236, full = 170;
          const groups = S.sliced
            ? [['Order status', A.easy, B.easy, 200, kB, 130], ['Billing disputes', A.hard, B.hard, 200, hB, 360], ['All 400 tickets', A.all, B.all, N, N, 590]]
            : [['All 400 tickets', A.all, B.all, N, N, 360]];
          groups.forEach(([name, a, b, na, nb, cx]) => {
            [[a, C.accent, cx - 70, 'A'], [b, C.orange, cx + 6, 'B']].forEach(([v, col, x, lab]) => {
              g.fillStyle = '#0f1520'; g.fillRect(x, base - full, 64, full);
              g.fillStyle = col; g.fillRect(x, base - v / 100 * full, 64, v / 100 * full);
              g.font = BIG; g.textAlign = 'center'; g.fillStyle = C.text; g.fillText(v.toFixed(1) + '%', x + 32, base - v / 100 * full - 7);
              g.font = BOLD; g.fillStyle = '#0a0e16'; g.fillText(lab, x + 32, base - 8);
            });
            g.font = BOLD; g.fillStyle = b > a ? C.orange : b < a ? C.accent : C.muted; g.textAlign = 'center';
            g.fillText(name + ': ' + (Math.abs(b - a) < 1e-9 ? 'level' : (b > a ? 'B' : 'A') + ' ahead by ' + Math.abs(b - a).toFixed(1)), cx, base + 20);
            g.font = MONO; g.fillStyle = C.muted;
            g.fillText('A ' + na + ' · B ' + nb + ' tickets', cx, base + 38);
          });
          const overallB = B.all > A.all, slicesA = A.easy > B.easy && A.hard > B.hard;
          g.textAlign = 'left'; g.font = BOLD;
          let msg, col;
          if (!S.sliced) { msg = overallB ? 'The dashboard says B is the better bot. Slice by topic before you believe it.' : 'On one number, A is ahead. Slice by topic to see where.'; col = C.muted; }
          else if (overallB && slicesA) { msg = 'B wins overall and loses on every topic. Its set simply held more easy tickets.'; col = C.danger; }
          else if (!overallB && slicesA) { msg = 'Same mix of tickets for both: A wins every topic and the overall number agrees.'; col = C.green; }
          else { msg = 'The overall number and the slices disagree on at least one topic.'; col = C.warn; }
          g.fillStyle = col;
          wrapLines(g, msg, 660).forEach((ln, i) => g.fillText(ln, 30, 300 + i * 17));
          const out = { 'A overall': A.all.toFixed(1) + '%', 'B overall': B.all.toFixed(1) + '%' };
          if (S.sliced) { out['order status A / B'] = A.easy.toFixed(1) + '% / ' + B.easy.toFixed(1) + '%'; out['billing A / B'] = A.hard.toFixed(1) + '% / ' + B.hard.toFixed(1) + '%'; }
          ro.set(out);
        }
        const sl = ctx.slider({ label: 'order-status share of B\'s eval set', min: 50, max: 90, step: 5, value: 85, fmt: (v) => v + '%', onChange: (v) => { S.share = v; S.dirty = true; } });
        const sliceBtn = ctx.button('Slice by topic', () => { S.sliced = !S.sliced; sliceBtn.textContent = S.sliced ? 'Back to one number' : 'Slice by topic'; S.dirty = true; }, 'primary');
        const sameBtn = ctx.button('Score both on the same tickets', () => { S.share = 50; sl.value = 50; S.dirty = true; });
        ctx.loop(() => { if (S.dirty) { S.dirty = false; draw(); } });
        return ctx.figure(cv, 'A broken comparison, on purpose. Bot A was scored on 400 tickets sampled in March, half of them order-status questions (easy) and half billing disputes (hard). Bot B was scored on 400 tickets from April, when a delivery delay flooded the queue with the easy kind. The slider sets how lopsided April was. Tickets and outcomes are simulated, at fixed rates.', [sl, sliceBtn, sameBtn], ro);
      }

      /* ================================================================== */
      /* 5. The A/B test you can stop early                                 */
      /* ================================================================== */
      function abLab() {
        const W = 720, H = 350;
        const [cv, g] = ctx.canvas(W, H);
        const DAYS = 30, PER = 500, BASE = 0.30;
        const SE = (d) => Math.sqrt(2 * BASE * (1 - BASE) / (PER * d));
        function run(seed, lift) {
          const r = rng(seed), out = []; let a = 0, b = 0;
          const pb = BASE + lift;
          for (let d = 1; d <= DAYS; d++) {
            a += Math.max(0, Math.round(PER * BASE + Math.sqrt(PER * BASE * (1 - BASE)) * gauss(r)));
            b += Math.max(0, Math.round(PER * pb + Math.sqrt(PER * pb * (1 - pb)) * gauss(r)));
            const n = PER * d, diff = (b - a) / n;
            out.push({ pa: a / n, pb: b / n, diff, win: diff > 2 * SE(d) });
          }
          return out;
        }
        const S = { mode: 'same', test: 1, day: 0, shipped: 0, data: null, batch: null, dirty: true };
        const seedOf = (mode, t) => (mode === 'same' ? 3689 : 53689) + t - 1;
        const liftOf = (mode) => (mode === 'same' ? 0 : 0.02);
        function fresh() { S.data = run(seedOf(S.mode, S.test), liftOf(S.mode)); S.day = 0; S.shipped = 0; S.dirty = true; }
        const ro = ctx.readout();
        const P = { x: 56, y: 36, w: 640, h: 204 };
        const RANGE = 6;
        function draw() {
          g.clearRect(0, 0, W, H);
          const xAt = (d) => P.x + d / DAYS * P.w, yAt = (v) => P.y + P.h / 2 - ctx.clamp(v, -RANGE, RANGE) / RANGE * (P.h / 2 - 4);
          g.font = BOLD; g.textAlign = 'left'; g.fillStyle = C.text;
          g.fillText('Test ' + S.test + ': thumbs-up rate of B minus A, in points, as the users arrive', P.x, 22);
          panel(g, P);
          g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
          g.fillStyle = 'rgba(148,163,184,0.16)'; g.beginPath();
          for (let d = 1; d <= DAYS; d++) { const y = yAt(2 * SE(d) * 100); d === 1 ? g.moveTo(xAt(d), y) : g.lineTo(xAt(d), y); }
          for (let d = DAYS; d >= 1; d--) g.lineTo(xAt(d), yAt(-2 * SE(d) * 100));
          g.closePath(); g.fill();
          g.strokeStyle = C.muted; g.lineWidth = 1; g.beginPath(); g.moveTo(P.x, yAt(0)); g.lineTo(P.x + P.w, yAt(0)); g.stroke();
          const upto = S.shipped ? DAYS : S.day;
          for (let d = 1; d <= upto; d++) {
            const cur = S.data[d - 1], ghost = S.shipped && d > S.shipped;
            if (d > 1) {
              g.strokeStyle = ghost ? 'rgba(230,235,245,0.25)' : C.text; g.lineWidth = 1.6;
              g.beginPath(); g.moveTo(xAt(d - 1), yAt(S.data[d - 2].diff * 100)); g.lineTo(xAt(d), yAt(cur.diff * 100)); g.stroke();
            }
            g.beginPath(); g.arc(xAt(d), yAt(cur.diff * 100), cur.win ? 4.5 : 3, 0, Math.PI * 2);
            g.fillStyle = ghost ? (cur.win ? 'rgba(56,217,169,0.4)' : 'rgba(230,235,245,0.3)') : cur.win ? C.green : C.text; g.fill();
          }
          if (S.shipped) { g.strokeStyle = C.warn; g.lineWidth = 1.5; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(xAt(S.shipped), P.y); g.lineTo(xAt(S.shipped), P.y + P.h); g.stroke(); g.setLineDash([]); }
          g.restore();
          g.font = MONO; g.fillStyle = C.muted; g.textAlign = 'right';
          for (const t of [-6, -3, 0, 3, 6]) g.fillText(pts(t, 0), P.x - 6, yAt(t) + 4);
          g.textAlign = 'center';
          for (const d of [5, 10, 15, 20, 25, 30]) g.fillText('day ' + d, xAt(d) - (d === 30 ? 16 : 0), P.y + P.h + 16);
          /* status */
          const cur = S.day ? S.data[S.day - 1] : null;
          let msg, col = C.muted;
          if (S.shipped) {
            const sh = S.data[S.shipped - 1], end = S.data[DAYS - 1];
            if (S.mode === 'same') { msg = 'You shipped B on day ' + S.shipped + ', ' + pts(sh.diff * 100) + ' points ahead. The truth: B is identical to A. By day 30 the gap was ' + pts(end.diff * 100) + (end.win ? ', still outside the band by luck.' : ', back inside the band.'); col = C.danger; }
            else { msg = 'You shipped B on day ' + S.shipped + '. The truth: B really is 2 points better. By day 30 the gap was ' + pts(end.diff * 100) + (sh.win ? '.' : ', though on the day you shipped it was still inside the band.'); col = sh.win ? C.green : C.warn; }
          } else if (!cur) { msg = 'Day 0. No users yet. The grey band is where chance alone keeps the gap about 95% of the time.'; }
          else if (cur.win) { msg = 'Day ' + S.day + ': B is ' + pts(cur.diff * 100) + ' points ahead and OUTSIDE the band. The dashboard would call that significant.'; col = C.green; }
          else if (S.day === DAYS) { msg = 'Day 30, the planned end: gap ' + pts(cur.diff * 100) + ' points, inside the band. No win to report.'; col = C.text; }
          else { msg = 'Day ' + S.day + ': gap ' + pts(cur.diff * 100) + ' points, inside the band.'; }
          g.textAlign = 'left'; g.font = BOLD; g.fillStyle = col;
          wrapLines(g, msg, 650).forEach((ln, i) => g.fillText(ln, P.x, 280 + i * 17));
          g.font = MONO; g.fillStyle = S.batch ? C.warn : C.muted;
          if (S.batch) {
            g.fillText('1,000 tests, ' + (S.batch.mode === 'same' ? 'B identical to A' : 'B truly 2 points better') + '.  Stop on the first day outside the band: ' + S.batch.peek + ' wins.', P.x, 322);
            g.fillText('Same 1,000 tests, looked at once, on day 30: ' + S.batch.end + ' wins.', P.x, 340);
          } else g.fillText('Press "Run 1,000 tests" to see how often each way of deciding declares a win.', P.x, 322);
          const o = { test: S.test, day: S.day, 'users per arm': (S.day * PER).toLocaleString() };
          if (cur) { o['A thumbs-up'] = (cur.pa * 100).toFixed(1) + '%'; o['B thumbs-up'] = (cur.pb * 100).toFixed(1) + '%'; o['gap'] = pts(cur.diff * 100); o['band'] = '±' + (2 * SE(S.day) * 100).toFixed(1); }
          ro.set(o);
        }
        const modeSel = ctx.select({ label: 'what is really true', options: [{ value: 'same', label: 'B is identical to A' }, { value: 'better', label: 'B is truly 2 points better' }], value: 'same', onChange: (v) => { S.mode = v; S.test = 1; S.batch = null; fresh(); } });
        const nextBtn = ctx.button('Next day', () => { if (!S.shipped && S.day < DAYS) { S.day++; S.dirty = true; } }, 'primary');
        const endBtn = ctx.button('Run to day 30', () => { if (!S.shipped) { S.day = DAYS; S.dirty = true; } });
        const shipBtn = ctx.button('Stop and ship B', () => { if (!S.shipped && S.day > 0) { S.shipped = S.day; S.dirty = true; } });
        const newBtn = ctx.button('New test', () => { S.test++; fresh(); });
        const batchBtn = ctx.button('Run 1,000 tests', () => {
          let peek = 0, end = 0;
          for (let t = 1; t <= 1000; t++) { const o = run(seedOf(S.mode, t), liftOf(S.mode)); if (o.some((x) => x.win)) peek++; if (o[DAYS - 1].win) end++; }
          S.batch = { peek, end, mode: S.mode }; S.dirty = true;
        });
        fresh();
        ctx.loop(() => { if (S.dirty) { S.dirty = false; draw(); } });
        return ctx.figure(cv, 'A simulated A/B test of a support bot. Each day 500 new users get version A and 500 get version B, and about 30% of them press thumbs-up. The line is the running gap. The grey band is two standard errors of that gap either side of zero, so it narrows as users accumulate. A green dot is a day when B sits above the band; only that side counts as a win, since nobody ships B for being behind. With the first setting B is a copy of A, so every green dot is a false alarm. Test 1 was hand-picked to cross the band early: most copies never do, as <b>New test</b> and <b>Run 1,000 tests</b> show. The band is worked out from the 30% base rate, and daily counts come from a seeded generator using a normal approximation.', [modeSel, nextBtn, endBtn, shipBtn, newBtn, batchBtn], ro);
      }

      /* ================================================================== */
      /* 6. The regression suite                                            */
      /* ================================================================== */
      function regressionLab() {
        const CASES = [
          ['Looks up an order by its number', 'orders', () => true],
          ['Walks through a password reset', 'account', () => true],
          ['Answers a yes/no question without five paragraphs', 'tone', (f) => f.r1],
          ['Gives the opening hours in the first line', 'tone', (f) => f.r1],
          ['Router setup: gives all six steps', 'how-to', (f) => !f.r1 || f.r3],
          ['Apologises first to an angry customer', 'tone', () => true],
          ['Does not promise a refund outside the 30-day policy', 'refunds', (f) => f.r2],
          ['Asks for the order number before discussing a charge', 'billing', () => true],
          ['"Where is my refund?" gets the refund\'s status', 'refunds', (f) => !f.r2 || f.r4],
          ['Replies in Spanish to a message in Spanish', 'language', () => true],
        ];
        const RULES = [
          ['r1', 'Rule 1: keep answers under three sentences'],
          ['r2', 'Rule 2: never discuss refunds, hand them to a human'],
          ['r3', 'Rule 3: step-by-step guides may run long'],
          ['r4', 'Rule 4: refund status questions are fine'],
        ];
        const F = { r1: false, r2: false, r3: false, r4: false };
        const NONE = { r1: false, r2: false, r3: false, r4: false };
        let open = false;
        const body = h('div', {});
        const ro = ctx.readout();
        const btns = RULES.map(([k, lab]) => ctx.button(lab, () => { F[k] = !F[k]; render(); }));
        function render() {
          btns.forEach((b, i) => b.classList.toggle('primary', F[RULES[i][0]]));
          const before = CASES.map((c) => c[2](NONE)), after = CASES.map((c) => c[2](F));
          const sb = before.filter(Boolean).length, sa = after.filter(Boolean).length;
          const fixed = after.filter((v, i) => v && !before[i]).length, broke = after.filter((v, i) => !v && before[i]).length;
          const on = RULES.filter(([k]) => F[k]).map(([k]) => k.slice(1));
          body.innerHTML = '';
          body.append(h('div', { style: { display: 'flex', gap: '18px', flexWrap: 'wrap', alignItems: 'baseline', marginBottom: '10px' } },
            h('div', { style: { fontSize: '1.5rem', fontWeight: '700' }, html: 'Suite score: ' + sb + ' / 10 &rarr; <span style="color:' + (sa > sb ? C.green : sa < sb ? C.danger : C.text) + '">' + sa + ' / 10</span>' }),
            h('div', { style: { color: 'var(--muted)' } }, 'last week\'s prompt → this build (' + (on.length ? 'rules ' + on.join(', ') + ' added' : 'no rules added yet') + ')')));
          if (!open) {
            body.append(h('p', { style: { color: 'var(--muted)', margin: '0' }, html: 'This is all a dashboard shows: one score, and whether it went up. Press <b>Show case-by-case diff</b> to see what happened underneath it.' }));
          } else {
            const tag = (ok) => '<span style="color:' + (ok ? C.green : C.danger) + '">' + (ok ? 'pass' : 'FAIL') + '</span>';
            body.append(h('div', { class: 'table-wrap' }, h('table', { class: 'tbl', style: { margin: '0' } },
              h('thead', {}, h('tr', {}, ['test case', 'last week', 'this build', 'change'].map((t) => h('th', {}, t)))),
              h('tbody', {}, CASES.map((c, i) => h('tr', {},
                h('td', { html: c[0] + ' <span style="color:var(--muted)">· ' + c[1] + '</span>' }),
                h('td', { html: tag(before[i]) }), h('td', { html: tag(after[i]) }),
                h('td', { html: after[i] && !before[i] ? '<b style="color:' + C.green + '">fixed</b>' : !after[i] && before[i] ? '<b style="color:' + C.danger + '">BROKE</b>' : '<span style="color:var(--muted)">same</span>' })))))));
          }
          ro.set({ score: sa + ' / 10', 'newly fixed': open ? fixed : '?', 'newly broken': open ? broke : '?', 'safe to ship': !open ? '?' : broke ? 'no: ' + broke + ' regression' + (broke > 1 ? 's' : '') : sa === 10 ? 'yes' : 'nothing broke, ' + (10 - sa) + ' still failing' });
        }
        const friday = ctx.button('Apply Friday\'s patch', () => { F.r1 = true; F.r2 = true; F.r3 = false; F.r4 = false; render(); });
        const diffBtn = ctx.button('Show case-by-case diff', () => { open = !open; diffBtn.textContent = open ? 'Hide case-by-case diff' : 'Show case-by-case diff'; render(); }, 'primary');
        const reset = ctx.button('Back to last week', () => { F.r1 = F.r2 = F.r3 = F.r4 = false; render(); });
        render();
        return ctx.figure(body, 'A ten-case regression suite for the support bot. Each rule is a sentence added to the system prompt, and each case passes or fails by a fixed, hand-written consequence of the rules that are switched on: this is a model of how prompt edits interact, with no language model behind it. Friday\'s patch adds rules 1 and 2 together.', [friday, ...btns, diffBtn, reset], ro);
      }

      /* ================================================================== */
      /* 7. The biased judge                                                */
      /* ================================================================== */
      function judgeLab() {
        const GOOD = 'No. Annual plans can be refunded within 30 days of purchase, and you are at day 45. I can switch off auto-renewal for you now.';
        const BAD = 'Thank you so much for reaching out, and I completely understand how important it is to feel confident about your purchase. Great news: you are absolutely eligible for a refund on your annual plan. Our refund policy is designed with customers in mind, and it covers annual subscriptions for a generous period after purchase. Here is how the process works. First, your request is reviewed by our billing team. Second, the unused portion of your plan is calculated on a pro-rata basis. Third, the refund is returned to your original payment method, typically within five to seven working days. In addition, you will keep full access to every premium feature until the refund has been completed. Please let me know if there is anything else at all that I can help you with today, and thank you again for being such a valued customer.';
        const wc = (s) => s.trim().split(/\s+/).length;
        /* [question, words in the correct answer, words in the wrong one, how plausible the wrong one sounds 0..1, correct answer first?] */
        const PAIRS = [
          ['Refund on an annual plan after 45 days?', wc(GOOD), wc(BAD), 0.5, false],
          ['Does the warranty cover water damage?', 25, 160, 0.3, false],
          ['How do I cancel before renewal?', 30, 200, 0.6, false],
          ['Which plan includes phone support?', 40, 45, 0.8, false],
          ['Reset two-factor after losing a phone', 60, 20, 0.2, true],
          ['Is my data stored in the EU?', 35, 38, 0.75, true],
          ['Export my records to a spreadsheet', 90, 30, 0.4, true],
          ['Delivery time to Canada?', 22, 120, 0.1, true],
        ];
        const LEN_BIAS = 0.5, POS_BIAS = 0.3;   // per 100 extra words; for sitting in slot 1
        const S = { swapped: false, trim: false, both: false };
        /* does the judge pick the correct answer, when the correct one sits first (or not)? */
        function picksCorrect(pr, correctFirst) {
          const cw = S.trim ? Math.min(pr[1], pr[2]) : pr[1], ww = S.trim ? Math.min(pr[1], pr[2]) : pr[2];
          return (1 - pr[3]) + LEN_BIAS * (cw - ww) / 100 + (correctFirst ? POS_BIAS : -POS_BIAS) > 0;
        }
        const body = h('div', {});
        const ro = ctx.readout();
        function render() {
          body.innerHTML = '';
          const first = (pr) => (S.swapped ? !pr[4] : pr[4]);
          const p0 = PAIRS[0], cf0 = first(p0);
          const cut = (s, n) => { const w = s.trim().split(/\s+/); return w.length <= n ? s : w.slice(0, n).join(' ') + ' …'; };
          const lim = S.trim ? Math.min(p0[1], p0[2]) : 1e9;
          const card = (slot, good) => h('div', { class: 'card', style: { margin: '0' } },
            h('h4', { html: 'Answer ' + slot + ' <span style="color:var(--muted);font-weight:400">· ' + Math.min(lim, good ? p0[1] : p0[2]) + ' words · human raters: <b style="color:' + (good ? C.green : C.danger) + '">' + (good ? 'correct' : 'wrong') + '</b></span>' }),
            h('p', { style: { margin: '0', fontSize: '.9rem' } }, cut(good ? GOOD : BAD, lim)));
          const pick0 = picksCorrect(p0, cf0);
          const slot0 = pick0 === cf0 ? 1 : 2;
          body.append(
            h('p', { style: { margin: '0 0 10px' }, html: '<b>Customer:</b> I bought the annual plan 45 days ago. Can I get a refund? <span style="color:var(--muted)">(Policy: annual plans are refundable for 30 days.)</span>' }),
            h('div', { class: 'grid-2', style: { margin: '0 0 10px' } }, card(1, cf0), card(2, !cf0)));
          let agree = 0, slot1 = 0, longer = 0, kept = 0, keptRight = 0;
          const rows = PAIRS.map((pr, i) => {
            const cf = first(pr), pick = picksCorrect(pr, cf), other = picksCorrect(pr, !cf);
            const slot = pick === cf ? 1 : 2;
            if (pick) agree++;
            if (slot === 1) slot1++;
            if ((pick ? pr[1] : pr[2]) > (pick ? pr[2] : pr[1])) longer++;
            const stable = pick === other;
            if (stable) { kept++; if (pick) keptRight++; }
            const cell = (good) => '<span style="color:' + (good ? C.green : C.danger) + '">' + (good ? 'correct' : 'wrong') + '</span> · ' + (S.trim ? Math.min(pr[1], pr[2]) : good ? pr[1] : pr[2]) + ' words';
            let verdict;
            if (S.both) verdict = stable ? 'same answer both times: <b style="color:' + (pick ? C.green : C.danger) + '">' + (pick ? 'right' : 'wrong') + '</b>' : '<span style="color:' + C.warn + '">changed its mind: send to a human</span>';
            else verdict = 'answer ' + slot + ': <b style="color:' + (pick ? C.green : C.danger) + '">' + (pick ? 'right' : 'wrong') + '</b>';
            return h('tr', {}, h('td', {}, (i + 1) + '. ' + pr[0]), h('td', { html: '1: ' + cell(cf) + '<br>2: ' + cell(!cf) }), h('td', { html: verdict }));
          });
          body.append(
            h('p', { style: { margin: '0 0 6px' }, html: '<b>Judge model, on this pair:</b> ' + (S.both
              ? (pick0 === picksCorrect(p0, !cf0) ? 'picks the same answer in both orders.' : 'picks a different answer when the order is swapped. Verdict discarded.')
              : '"Answer ' + slot0 + ' is better." <span style="color:' + (pick0 ? C.green : C.danger) + '">' + (pick0 ? 'The human raters agree.' : 'The human raters say the opposite.') + '</span>') }),
            h('div', { class: 'table-wrap' }, h('table', { class: 'tbl', style: { margin: '0', fontSize: '.85rem' } },
              h('thead', {}, h('tr', {}, ['pair', 'answers 1 and 2', S.both ? 'judge, asked in both orders' : 'judge prefers'].map((t) => h('th', {}, t)))),
              h('tbody', {}, rows))));
          if (S.both) ro.set({ 'verdicts that survived the swap': kept + ' of 8', 'survivors that are right': keptRight + ' of ' + kept, 'sent to a human': 8 - kept });
          else ro.set({ 'judge agrees with human raters': agree + ' of 8', 'picked answer 1': slot1 + ' of 8', 'picked the longer answer': S.trim ? 'lengths equal' : longer + ' of 8' });
          trimBtn.classList.toggle('primary', S.trim); bothBtn.classList.toggle('primary', S.both);
        }
        const swapBtn = ctx.button('Swap the order', () => { S.swapped = !S.swapped; render(); }, 'primary');
        const trimBtn = ctx.button('Trim both to the same length', () => { S.trim = !S.trim; render(); });
        const bothBtn = ctx.button('Ask in both orders', () => { S.both = !S.both; render(); });
        render();
        return ctx.figure(body, 'Eight pairs of answers from two versions of the bot. Human raters have already marked one answer in each pair correct. The "judge model" is a simulation with two biases wired in on top of a real but imperfect sense of quality: a bonus for whichever answer is shown first, and a bonus for every extra word. Only the first pair is shown in full; the table gives the word counts for the rest.', [swapBtn, trimBtn, bothBtn], ro);
      }

      /* ================================================================== */
      /* 8. Goodhart: the dial that maximises the dashboard                 */
      /* ================================================================== */
      function goodhartLab() {
        const W = 720, H = 330;
        const [cv, g] = ctx.canvas(W, H);
        const N = 2000;
        const T = [];
        (function () {
          const r = rng(1975);
          for (let i = 0; i < N; i++) {
            const d = r();
            const solved = r() < sig(14 * (0.55 - d));
            T.push({ d, solved, human: r() < 0.95, rates: r() < (solved ? 0.4 : 0.1), up: r() < (solved ? 0.9 : 0.2), back: r() < (solved ? 0.05 : 0.7), backH: r() < 0.08 });
          }
        })();
        function at(hh) {
          let kept = 0, raters = 0, ups = 0, repeats = 0, helped = 0;
          for (const t of T) {
            if (t.d * 100 <= hh) { kept++; if (t.rates) { raters++; if (t.up) ups++; } if (t.back) repeats++; if (t.solved) helped++; }
            else { if (t.backH) repeats++; if (t.human) helped++; }
          }
          return { defl: kept / N * 100, thumbs: raters ? ups / raters * 100 : NaN, norep: (1 - repeats / N) * 100, helped: helped / N * 100, human: N - kept, gaveUp: 0 };
        }
        const curve = []; for (let v = 0; v <= 100; v += 5) curve.push(at(v));
        const S = { dial: 30, reveal: false, dirty: true };
        const AGENT_COST = 4;   // illustrative pounds per ticket a human handles
        const ro = ctx.readout();
        const P = { x: 52, y: 20, w: 640, h: 220 };
        function draw() {
          g.clearRect(0, 0, W, H);
          panel(g, P);
          const xAt = (v) => P.x + 6 + v / 100 * (P.w - 12), yAt = (v) => P.y + P.h - 6 - v / 100 * (P.h - 12);
          g.font = MONO; g.fillStyle = C.muted; g.textAlign = 'right';
          for (const t of [0, 25, 50, 75, 100]) { g.fillText(t + '%', P.x - 6, yAt(t) + 4); g.strokeStyle = C.line; g.beginPath(); g.moveTo(P.x, yAt(t)); g.lineTo(P.x + P.w, yAt(t)); g.stroke(); }
          const series = [['defl', C.accent, 'deflection rate'], ['thumbs', C.warn, 'thumbs-up rate'], ['norep', C.purple, 'no repeat contact']];
          if (S.reveal) series.push(['helped', C.green, 'customers actually helped']);
          g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
          series.forEach(([k, col]) => {
            g.strokeStyle = col; g.lineWidth = k === 'helped' ? 3 : 2; g.beginPath(); let pen = false;
            curve.forEach((c, i) => { if (isNaN(c[k])) return; const x = xAt(i * 5), y = yAt(c[k]); pen ? g.lineTo(x, y) : g.moveTo(x, y); pen = true; });
            g.stroke();
          });
          g.strokeStyle = C.text; g.setLineDash([4, 4]); g.lineWidth = 1.2; g.beginPath(); g.moveTo(xAt(S.dial), P.y); g.lineTo(xAt(S.dial), P.y + P.h); g.stroke(); g.setLineDash([]);
          const cur = at(S.dial);
          series.forEach(([k, col]) => { if (isNaN(cur[k])) return; g.beginPath(); g.arc(xAt(S.dial), yAt(cur[k]), 4.5, 0, Math.PI * 2); g.fillStyle = col; g.fill(); });
          g.restore();
          g.textAlign = 'center'; g.fillStyle = C.muted; g.font = MONO;
          g.fillText('0%: hands everything to a human', xAt(0) + 104, P.y + P.h + 16);
          g.fillText('100%: never hands off', xAt(100) - 70, P.y + P.h + 16);
          g.textAlign = 'left'; g.font = FONT;
          let lx = P.x;
          series.forEach(([, col, lab]) => { g.fillStyle = col; g.fillRect(lx, 271, 14, 4); g.fillText(lab, lx + 20, 278); lx += g.measureText(lab).width + 44; });
          g.font = BOLD;
          const msg = !S.reveal ? 'Three dashboard metrics. What the customers went through is hidden.'
            : cur.helped < 70 ? 'Deflection ' + cur.defl.toFixed(0) + '%, and only ' + cur.helped.toFixed(0) + '% of customers got their problem solved.'
            : 'At this setting ' + cur.helped.toFixed(0) + '% of customers got their problem solved, with ' + cur.human.toLocaleString() + ' of 2,000 tickets going to humans.';
          g.fillStyle = !S.reveal ? C.muted : cur.helped < 70 ? C.danger : C.green;
          g.fillText(msg, P.x, 312);
          ro.set({ deflection: cur.defl.toFixed(1) + '%', 'thumbs-up': isNaN(cur.thumbs) ? 'no ratings' : cur.thumbs.toFixed(1) + '%', 'no repeat contact': cur.norep.toFixed(1) + '%', 'actually helped': S.reveal ? cur.helped.toFixed(1) + '%' : 'hidden', 'tickets to humans': cur.human.toLocaleString(), 'agent cost': '£' + (cur.human * AGENT_COST).toLocaleString() });
        }
        const sl = ctx.slider({ label: 'share of tickets the bot keeps for itself', min: 0, max: 100, step: 5, value: 30, fmt: (v) => v + '%', onChange: (v) => { S.dial = v; S.dirty = true; } });
        const maxBtn = ctx.button('Maximise deflection', () => { S.dial = 100; sl.value = 100; S.dirty = true; }, 'primary');
        const revBtn = ctx.button('Reveal who was actually helped', () => { S.reveal = !S.reveal; revBtn.textContent = S.reveal ? 'Hide who was actually helped' : 'Reveal who was actually helped'; S.dirty = true; });
        ctx.loop(() => { if (S.dirty) { S.dirty = false; draw(); } });
        return ctx.figure(cv, 'Two thousand simulated support tickets, from easy to hard. The dial sets how many the bot keeps: it hands the hardest ones to a human and answers the rest itself, even those it cannot solve. (A simplification: a real bot does not know exactly how hard each ticket is.) Human agents solve 95% of what reaches them, at an illustrative £4 a ticket. A customer the bot fails usually gives up without rating it, and often writes in again within the week. All rates are illustrative and seeded.', [sl, maxBtn, revBtn], ro);
      }

      /* ================================================================== */
      /* The chapter: touch first, read second.                             */
      /* ================================================================== */
      root.append(
        callout('tryit', '🖐 Do this first: decide whether your change helped',
          `You run a support bot. Prompt <b>A</b> is live. You have rewritten it as prompt <b>B</b>, and you test both on 20 real customer questions.<br>
           <b>1.</b> Read the two bars. This set puts A ahead by 15 points. Would you ship your rewrite?<br>
           <b>2.</b> Press <b>Draw a new eval set</b> nine times, deciding each time. After ten sets the readout says B ahead 5, A ahead 3, tied 2.<br>
           <b>3.</b> Press <b>Reveal the truth</b>: both prompts scored on all 5,000 questions.<br>
           <b>4.</b> Press <b>100 items</b>, then <b>Draw 50 more</b>. Do the same at <b>1,000 items</b>. Compare the <b>A ahead</b> counts.`),
        evalDraw(),
        p(`B is the better prompt, by 4 points, on every press. With 20 questions the eval could not see that: your first set put A ahead by 15 points (a rare one, chosen to make the point), and B won only five sets in ten, which is typical. Nothing about either prompt changed between presses. Only the questions did.`),
        p(`At 100 questions the wrong prompt still came out ahead in 4 of 51 sets, and level in one more. At 1,000 it never did. "Did my change help?" is the question a practitioner asks every working day, and the size of the eval set decides whether the answer means anything.`),
      );

      root.append(section('A score is a sample, and samples wobble',
        p(`An eval score is a measurement taken on a sample of the questions the world could ask. Draw a different sample and you get a different score. Before comparing two models, find out how far one model's score moves when nothing but the sample changes.`),
        callout('tryit', '🖐 Try this: one model, 200 eval sets',
          `<b>1.</b> The model here is right exactly 70% of the time. At <b>100 items</b>, read <b>measured spread</b> and <b>predicted spread</b> in the readout: both say 4.6 points.<br>
           <b>2.</b> Press <b>20 items</b>. The scores now run from 45% to 95% for the same model, and the measured spread is 10.6 points.<br>
           <b>3.</b> Press <b>1,000 items</b>: 1.3 points measured, 1.4 predicted. Fifty times the data of step 2 bought only seven or eight times less wobble.<br>
           <b>4.</b> Back at <b>100 items</b>, drag <b>true accuracy</b> to 95%. The band narrows to 90.6%–99.4%. Scores near 100% wobble less than scores near 50%.`),
        spreadLab(),

        /* maths beat: the standard error of a proportion, built on chapter 3's sigma / sqrt(n) */
        ctx.walkthrough([
          { say: 'Each bar in that chart is built from eval scores, and a score is an <b>average</b>: 100 answers, each worth 1 if right and 0 if wrong, added up and divided by 100.', note: 'Chapter 3 gave the law for every average: its noise is &sigma; &divide; √n, where &sigma; is how much single items disagree.' },
          { say: 'For right-or-wrong items you never have to measure &sigma;: multiply the share right by the share wrong and take the square root.', math: '√(0.70 &times; 0.30) = √0.21 = 0.458' },
          { say: 'Divide that by the square root of the number of items, and you have predicted the spread of the score before drawing a single eval set.', math: '0.458 &divide; √100 = 0.0458 &rarr; 4.6 points', note: 'The readout printed exactly this as "predicted spread", next to the 4.6 you measured.' },
          { ask: 'Same model, but only <b>20</b> items. √20 is about 4.47. What spread does the recipe predict?',
            options: ['About 2 points', 'About 5 points', 'About 10 points'], answer: 2,
            explain: '0.458 ÷ 4.47 = 0.102, which is 10.2 points. You measured 10.6 when you pressed <b>20 items</b>, close to it.' },
          { say: 'This spread has a name, the <b>standard error</b> of the score, and the whole recipe is usually written under one square root.', math: 'SE = √( p &times; (1 &minus; p) &divide; n ) &nbsp;&nbsp; with p = 0.70, n = 100 &rarr; 0.046' },
          { say: 'Two standard errors either side of the truth made the shaded band, and about 95 scores in every 100 fall inside it.', math: '70 &plusmn; 2 &times; 4.6 &nbsp;=&nbsp; 60.8% to 79.2%', note: 'In your 200 sets of 100 items, 194 landed inside: 97%. Press <b>Run 200 fresh eval sets</b> and the count moves around; over many runs it settles near 96 in 100 here. "95" is the round-number rule.' },
          { say: 'In real work you hold one score and no truth, so you turn the band around: your score, plus or minus two standard errors, is a <b>95% confidence interval</b> for the truth.', math: 'scored 74 of 100: &nbsp; √(0.74 &times; 0.26 &divide; 100) = 0.044 &nbsp;&rarr;&nbsp; 74 &plusmn; 8.8', note: 'So "74%" on 100 items means "somewhere from about 65% to 83%".' },
          { ask: 'A model scores <b>80%</b> on <b>400</b> items. What is its 95% interval?',
            options: ['76% to 84%', '78% to 82%', '72% to 88%'], answer: 0,
            explain: '√(0.8 × 0.2 ÷ 400) = 0.4 ÷ 20 = 0.02, or 2 points. Two of those either side of 80 is 76 to 84.' },
          { say: 'This quick recipe is an approximation, and it fails at the edges: a model that gets 20 of 20 right would be handed an interval of &plusmn;0, which nobody should believe.', math: '√(1.00 &times; 0.00 &divide; 20) = 0', note: 'Trust it when the set holds at least ten or so right answers and ten or so wrong ones. Outside that, statisticians use a corrected recipe, the Wilson interval.' },
          { say: 'Now reread the opener: at 20 items each score carried about &plusmn;20 points, and you were hunting for a gap of 4.', math: '2 &times; 10.2 = &plusmn;20.4 points &nbsp;&nbsp; against a gap of 4' },
          { ask: 'At 100 items the interval is about &plusmn;9 points. How many items bring it down to &plusmn;3?',
            options: ['300', '900', '2,700'], answer: 1,
            explain: 'Three times narrower costs 3 × 3 = nine times the items, because n sits under a square root. 100 × 9 = 900.' },
          { say: 'At 1,000 items the interval is &plusmn;2.9 points, which is finally smaller than the 4-point gap, though two models means two intervals.', math: '2 &times; √(0.21 &divide; 1000) = 2 &times; 0.0145 = 0.029', note: 'The next demo shows the trick that makes two models cheaper to compare than that suggests.' },
        ], {
          title: 'From 200 eval sets to the standard error of a score',
          recap: 'a score of p on n items has a standard error of √(p(1 &minus; p) &divide; n), and the score lands within about two of those of the truth, 95 times in 100.',
        }),
        ctx.decoder([
          { sym: 'SE', name: 'the standard error of the score', says: 'How far a typical eval score lands from the model\'s true accuracy, purely through which items happened to be drawn.', points: 'the "predicted spread" in the readout, and the width of the heap of bars.' },
          '= √(',
          { sym: 'p', name: 'the share right', says: 'The model\'s accuracy as a fraction: 0.70 for 70%. In practice you plug in the score you measured.', points: 'the <b>true accuracy</b> slider.' },
          '×',
          { sym: '(1 − p)', name: 'the share wrong', says: 'Multiplied by p, it measures how much single answers disagree. It peaks at 50/50 and shrinks toward 0% or 100%.', points: 'the band narrowing when you dragged true accuracy to 95%.' },
          '÷',
          { sym: 'n', name: 'the number of items', says: 'The size of the eval set, and the only part you choose. It sits under the square root, so four times the items halves the error.', points: 'the <b>20 items</b>, <b>100 items</b> and <b>1,000 items</b> buttons.' },
          ')',
        ], {
          title: 'SE  =  √( p × (1 − p) ÷ n )',
          plain: 'the standard error of a score is the square root of: share right, times share wrong, divided by the number of items.',
        }),

        p(`That interval is the first thing to compute on any eval, yours or a vendor's. The half-widths below are for a score near 70%; they are a little wider at 50% and narrower above 90%. The last column is a rough guide to the size of change such a set can show. Comparing two versions has its own arithmetic, which is the next section.`),
        ctx.table(['items in the eval set', '95% interval', 'what it can detect'], [
          ['20', '± 20 points', 'only a model that is plainly broken'],
          ['100', '± 9 points', 'large changes: a new model family'],
          ['400', '± 4.6 points', 'a solid prompt or fine-tuning win'],
          ['1,000', '± 2.9 points', 'modest improvements'],
          ['10,000', '± 0.9 points', 'the small gaps on public leaderboards'],
        ]),
        callout('example', '🌍 The leaderboard gap that is not there',
          `A model card says the new version scores 86% on a 164-problem coding benchmark and the old one 84%. Run the recipe: √(0.86 × 0.14 ÷ 164) is 2.7 points, so each score carries about ±5.4. The whole 2-point gap is three or four problems out of 164.
           The claim may still be true. This evidence cannot show it. (The scores are invented. The size is real: HumanEval, a much-quoted coding benchmark, has 164 problems, and several other well-known benchmarks hold only a few hundred.)`),
        callout('history', '📜 William Gosset, Guinness, 1908',
          `The problem of judging from a small sample was first cracked by a brewer. William Sealy Gosset worked at Guinness in Dublin, where a trial of a new barley or a batch of hops might yield only a handful of measurements.
           In 1908 he published "The Probable Error of a Mean" in the journal <i>Biometrika</i>, showing how much extra caution a small sample demands. Guinness did not let staff publish under their own names, so he signed it "Student". Student's t-test still carries the pseudonym, and every small eval set is his problem again.`),
      ));

      root.append(section('Ask both versions the same questions',
        p(`Two models, each scored on its own 1,000 questions, two intervals of ±3 points each: the gap between them is blurrier still, about ±4. There is a way to buy most of that precision back without labelling one more question.`),
        callout('tryit', '🖐 Try this: separate samples, or the same items?',
          `<b>1.</b> Both rows compare prompt A with prompt B 300 times, at 200 questions. Read the two <b>spread</b> figures: 4.6 points on separate samples, 2.2 on the same items.<br>
           <b>2.</b> Count the red: the top row names the wrong winner in 65 of 300 experiments, the bottom row in 13.<br>
           <b>3.</b> Drag <b>questions per eval set</b> to 800. The top row is now at 2.3 points of spread. Separate samples needed four times the questions to match what pairing gave you at 200.<br>
           <b>4.</b> Drag it down to 50: wrong 110 times on top, 68 below. Pairing helps. It cannot rescue an eval this small.`),
        pairedLab(),
        p(`Most questions are easy for both prompts or hard for both. Give each prompt its own sample and one of them may simply draw the easier questions: that luck lands in the gap. Give both the same sample and the luck cancels, because whatever made the set easy made it easy for both.`),
        p(`The opener did this from the start: both prompts answered every handful. That is why, at 100 questions, B came out ahead in 46 sets of 51 even though each score on its own carried ±9 points. The ±9 describes one score. The gap between two scores on the same questions is steadier than either.`),
        p(`This is a <em>paired comparison</em>. The only items that carry information are the ones where the two versions disagree, so a practitioner keeps the eval set fixed, runs every candidate on it, and reads the list of questions that flipped. It is the cheapest precision in this chapter. How much it buys depends on how alike the two versions are: this pool was built so that they disagree on about one question in ten, which makes pairing worth roughly four times the data. Two very different models share less luck and gain less.`),
      ));

      root.append(section('What goes in the set',
        p(`Size and pairing say how steady a score is. They say nothing about whether it measures the right thing. A thousand questions about order tracking tell you nothing about billing disputes, however tight the interval.`),
        ul([
          `<b>Real traffic first.</b> Sample from what users actually sent, in the proportions they sent it. An eval of questions the team imagined measures the team's imagination.`,
          `<b>The cases you fear, on purpose.</b> Random sampling under-represents the rare, expensive failures: the angry customer, the non-English message, the request that must be refused. Add them deliberately and report them separately.`,
          `<b>A written answer key.</b> For each item, record what a correct answer must contain, agreed by people who know the domain. An item nobody can mark is noise.`,
          `<b>Tags.</b> Topic, language, difficulty, source. The next demo shows what they are for.`,
        ]),
        p(`Then cut the set in two. One part is for daily work: you look at its failures, change the prompt, and rerun. Chapter 3 showed where that leads, because picking the best of many tries selects for luck. So keep a second part back, a <em>held-out set</em> that you never read item by item and never tune on, and score it only before a release.`),
        p(`For a language model there is a second way to spend a set: publish it. Chapter 11 showed scores inflating once benchmark questions leak into training data. Your own held-out questions, written in-house and kept off the public web, are the ones a model cannot have memorised.`),
        callout('tryit', '🖐 Try this: a winner that loses everywhere',
          `Bot A was scored on March's tickets, bot B on April's. This comparison is broken on purpose.<br>
           <b>1.</b> Read the two bars: A 76.0%, B 84.3%. Would you promote B?<br>
           <b>2.</b> Press <b>Slice by topic</b>. On order status A leads by 2.0 points. On billing disputes A leads by 8.3. B loses both and still wins overall.<br>
           <b>3.</b> Read the ticket counts under the bars: B's set held 340 easy tickets, A's 200.<br>
           <b>4.</b> Press <b>Score both on the same tickets</b>. B falls to 71.0% and the three verdicts agree.`),
        sliceLab(),
        p(`Nothing was miscounted. Order-status questions are easy for any bot, and B's set happened to be 85% order status. A change in the mix of questions moved the average more than the change in the bot did. Statisticians call this reversal Simpson's paradox, after Edward Simpson, who described it in 1951, though Karl Pearson and Udny Yule had noticed it half a century earlier.`),
        p(`A <em>slice</em> is the eval score on one tagged subgroup. Slices do two jobs. They catch a comparison that was never like for like, as here. And they catch the honest average that hides a failing group: B at 84% overall was answering barely half of billing disputes correctly, and those are the customers most likely to leave.`),
        callout('warning', '⚠️ A slice is a small eval set',
          `Sixty billing tickets, the size of B's billing slice above, carry an interval of about ±13 points. The demo's rates are fixed by hand, so its 8-point billing gap is exact; on real tickets a gap that size on a slice that small would prove nothing. Slice finely enough and some slice will always look terrible by chance. Report each slice with its item count, and grow the slices you care about until their intervals are worth reading.`),
      ));

      root.append(section('Offline is the rehearsal. Online is the show',
        p(`Everything so far is <em>offline evaluation</em>: a fixed set of questions, scored at your desk before any user is involved. It is fast, repeatable and safe. It also measures answers to last month's questions, marked by your own answer key, and users may want something else.`),
        p(`<em>Online evaluation</em> measures what real users do with the live product. Its standard instrument is the <em>A/B test</em>: split incoming users at random, give half the old version and half the new, and compare a behaviour such as thumbs-up rate. Random assignment is the cure for the March-and-April mistake above: neither arm can be handed the easier crowd on purpose or by habit. It does not pair anything. The two arms still differ by luck, and that luck is what the band in the next demo measures.`),
        callout('tryit', '🖐 Try this: an experiment with nothing to find',
          `In this test B is a <b>copy</b> of A. Any gap is pure chance. The test is planned to run 30 days.<br>
           <b>1.</b> Press <b>Next day</b> five times. On day 5 B is 3.3 points ahead, outside a band of ±2.6. Your manager wants to ship.<br>
           <b>2.</b> Press <b>Stop and ship B</b> and read what waiting would have shown: by day 30 the gap is −0.4, back inside the band.<br>
           <b>3.</b> Press <b>Run 1,000 tests</b>. Stopping at the first good day declares 134 wins out of 1,000 copies. Looking once on day 30 declares 21.<br>
           <b>4.</b> Switch <b>what is really true</b> to <b>B is truly 2 points better</b> and press <b>Run 1,000 tests</b>: waiting until day 30 still finds 970 of them.`),
        abLab(),
        p(`On any single day the band holds a no-difference gap about 95% of the time, and the misses split evenly: about 2.5% above it, where B "wins", and 2.5% below. Look on thirty days and you give chance thirty attempts to step above it, then stop the moment it does. That is <em>peeking</em>, and it turned a 2% false-win rate into 13% without anyone doing anything dishonest.`),
        p(`Peeking tempts because it catches real wins too: with B truly 2 points better, stopping at the first green dot flags 987 of the 1,000. It cannot tell those from the 134 false ones, and it exaggerates. Among those real winners the gap on the day they were stopped averaged 3.2 points, where the truth was 2.`),
        p(`The discipline is dull and it works. Before the test starts, write down the metric, the number of users and the end date. Look at the dashboard as often as you like for crashes. Decide once, at the end. Statisticians have built sequential tests that allow early stopping at an honest error rate. They pay for the extra looks with a wider band, and the most used designs make it much wider on the early days.`),
        callout('example', '🌍 You are in several A/B tests right now',
          `Search engines, shopping sites and streaming services run controlled experiments constantly; the button colour, ranking order or recommendation you saw today was probably one arm of one. AI products do the same with prompts and models.
           The thumbs under a chatbot's answer, the "regenerate" button and whether you copied the reply are all online signals that an offline eval cannot produce.`),
      ));

      root.append(section('The fix that breaks something else',
        callout('tryit', '🖐 Try this: ship Friday\'s patch',
          `Last week's bot rambled and promised a refund it should not have. A colleague wrote two new rules.<br>
           <b>1.</b> Press <b>Apply Friday's patch</b>. The suite score goes from 7 / 10 to 8 / 10. Would you ship?<br>
           <b>2.</b> Press <b>Show case-by-case diff</b>. Three cases were fixed and two that used to pass now say <b>BROKE</b>.<br>
           <b>3.</b> Press <b>Rule 3: step-by-step guides may run long</b>, then <b>Rule 4: refund status questions are fine</b>. Each repairs one broken case: 9 / 10, then 10 / 10.<br>
           <b>4.</b> Press <b>Back to last week</b>, then <b>Rule 1: keep answers under three sentences</b> on its own: 7 → 8 again, with the router guide broken underneath.`),
        regressionLab(),
        p(`A change that breaks something which used to work is a <em>regression</em>. The score hid two of them, because a score nets fixes against breakages and reports the difference. Read as a list, the patch told a bot to stay brief and thereby cut a six-step setup guide off at step three.`),
        p(`A <em>regression suite</em> is the defence: a set of cases that must keep passing, rerun on every change and read case by case. It grows by one rule of habit: every bug a user reports becomes a new case before it is fixed. Prompts need this more than ordinary code does, since one added sentence changes the model's behaviour on every input at once.`),
        p(`The suite also earns its keep when you change nothing. A hosted model gets upgraded by its provider, and behaviour your prompt relied on shifts. Teams that rerun their suite against each new model version find out in an afternoon. Teams without one find out from customers.`),
      ));

      root.append(section('Grading an answer that is written, not chosen',
        p(`"Right or wrong" was easy while answers were labels. A support reply is a paragraph, and there are a thousand acceptable ways to write it. Four ways to mark it, in rising order of cost and flexibility:`),
        ul([
          `<b>Exact match.</b> Compare with a reference string, or check that a required fact, number or function call is present. Cheap and perfectly repeatable. Blind to a correct answer in different words.`,
          `<b>A rubric.</b> A written checklist: states the policy correctly, answers the question asked, promises nothing we cannot do. A <em>rubric</em> turns "is this good?" into several narrower questions that different markers answer the same way.`,
          `<b>Human raters.</b> The reference standard, and slow and costly. Raters disagree with each other, so have two mark the same items and measure how often they match before trusting either.`,
          `<b>A model as the marker.</b> Hand the question, the answer and the rubric to a strong language model and ask for a verdict. This is <em>LLM-as-judge</em>: thousands of answers marked in minutes. The judge is itself a model, with habits of its own.`,
        ]),
        callout('tryit', '🖐 Try this: catch the judge out',
          `Human raters have marked the short answer correct and the long one wrong: the customer is past the refund window.<br>
           <b>1.</b> Read the judge's verdict on the pair. Then read the readout: it agrees with the humans on 4 of 8 pairs, and it picked answer 1 in 8 of 8.<br>
           <b>2.</b> Press <b>Swap the order</b>. The same two answers, and the judge now prefers the other one. Agreement rises to 6 of 8.<br>
           <b>3.</b> Press <b>Swap the order</b> again to go back, then <b>Trim both to the same length</b>: 7 of 8. Length was doing some of the judging.<br>
           <b>4.</b> Press the trim button again to switch it off, then <b>Ask in both orders</b>. Four verdicts survive the swap. One of the survivors is still wrong: find it in the table.`),
        judgeLab(),
        p(`A judge that picks answer 1 eight times out of eight is reading position. That is <em>position bias</em>, and swapping exposes it: any verdict that flips with the order was never a verdict. Asking in both orders and keeping only the stable answers is the standard repair.`),
        p(`Pair 3 survived the swap and was still wrong. Its bad answer ran to 200 words, and this judge rewards words. That is <em>verbosity bias</em>, a taste for the longer, more confident-sounding answer, and order-swapping cannot touch it. A rubric that names the facts required does better than asking which answer is "better".`),
        p(`A third habit has been measured in real judges: <em>self-preference</em>, a tendency to rate text from its own model family above other text of equal quality. So do not let a model mark its own homework against a competitor's. All three biases are documented in published studies of real LLM judges; the sizes in this demo are invented.`),
        callout('key', '🔑 Evaluate the evaluator',
          `Before trusting a judge, have humans label a few hundred items and measure how often the judge agrees with them, slice by slice. That agreement rate is the judge's own eval score, with its own interval. Recheck it whenever the judge model or its prompt changes.`),
      ));

      root.append(section('Choose the metric, then distrust it',
        callout('tryit', '🖐 Try this: hit the target',
          `Your team's bonus depends on <b>deflection rate</b>: the share of tickets closed without a human agent.<br>
           <b>1.</b> The bot starts by keeping 30% of tickets. Press <b>Maximise deflection</b>. Deflection reads 100.0% and thumbs-up 80.8%. The dashboard looks healthy.<br>
           <b>2.</b> One number moved the wrong way: <b>no repeat contact</b> fell from 92.3% to 65.5%.<br>
           <b>3.</b> Press <b>Reveal who was actually helped</b>: 54.9%.<br>
           <b>4.</b> Drag the dial back to 60%. Now 90.0% of customers are helped and deflection is 60.7%. The last 40 points of deflection were bought with customers.<br>
           <b>5.</b> Drag it to 0%, then to 45%. They help 95.7% and 95.9% of customers, but <b>agent cost</b> falls from £8,000 to £4,328. Up to there, deflection was pure saving.`),
        goodhartLab(),
        p(`The bot was never made better. It was made to stop asking for help. Deflection counted a customer who gave up as a success, and thumbs-up stayed respectable because people who give up rarely stay to press a button. Both metrics measured what was easy to count.`),
        p(`Deflection was never a bad thing to want. Below a dial of about 45% the bot takes easy tickets off the agents and customers lose nothing, so handing everything to humans is no answer either: it helps the same share at nearly twice the cost. The metric turned harmful only when it became the target and was pushed past the point where the bot stops solving.`),
        p(`Chapters 9 and 11 met this already: <em>Goodhart's law</em>, a measure that becomes a target stops being a good measure. In practice you defend against it in three ways. Pick the metric closest to what the user came for. Pair every target with a <b>guardrail metric</b> that must not get worse, as repeat contact was here. And read real transcripts every week, because no dashboard does that for you.`),
      ));

      root.append(section('Why this matters for LLM work',
        p(`Training a frontier model takes a lab. Building a product on one takes a prompt, some retrieval and a weekend. What separates products that keep improving from products that drift is whether each change is measured, and the measuring is the part nobody can do for you.`),
        p(`The working loop of an LLM engineer is this chapter in order. Collect real inputs and write the answer key. Compute the interval, so you know how big a change the set can see. Run both versions on the same items and read what flipped. Slice. Rerun the regression suite. Check the judge against people. Then test online, for a length fixed in advance.`),
        p(`Language models add three difficulties that classical models did not have. The same prompt can give different answers on different runs, so a single item's result is itself a sample. The model under your product can be replaced by its provider. And public benchmark questions may be in the training data, which is why your private, held-out set is worth more than any leaderboard.`),
        callout('key', '🔑 One sentence to keep',
          `A score is a claim about the future made from a sample of the past. Before you act on it, ask how big the sample was, whether both versions saw the same one, what the average is hiding, and who chose the metric.`),
      ));

      root.append(
        ctx.quiz([
          { q: 'Your new prompt scores 90% on a 100-item eval. What is the 95% interval on that score?',
            options: ['87% to 93%', '84% to 96%', '88% to 92%', '80% to 100%'], answer: 1,
            explain: '√(0.9 × 0.1 ÷ 100) = √0.0009 = 0.03, so one standard error is 3 points and two are 6. The truth is plausibly anywhere from 84% to 96%, which is the band you watched narrow as accuracy rose.' },
          { q: 'An A/B test was planned for 30 days. On day 6 the dashboard shows the new version "significantly ahead", and a teammate who has checked every morning wants to ship. What do you say?',
            options: ['Ship: a significant result is significant on whichever day it appears', 'Restart the test with fewer users so that it finishes sooner', 'The test is invalid, because the two groups contain different users', 'Keep it running to day 30: stopping at the first good day inflates false wins'], answer: 3,
            explain: 'This is the copy-of-A test you shipped on day 5. Each daily look is another chance for noise to cross the band, and in the demo that turned 21 false wins per 1,000 into 134. Different users in each arm is how an A/B test is meant to work; random assignment makes the arms comparable.' },
          { q: 'You have budget to label 500 questions and two prompt versions to compare. Which design gives the most trustworthy answer?',
            options: ['Run both versions on the same 500 questions', 'Give each version its own random 250, so neither is favoured', 'Score the old version on last month\'s logs and the new one on this month\'s', 'Use 20 carefully chosen questions and rerun them 25 times each'], answer: 0,
            explain: 'Pairing cancels the luck of which questions were drawn; in the demo it was worth about four times the data. Then read the questions where the two versions disagree. Separate samples add that luck back, different months change the mix of questions as in the slicing demo, and 20 questions rerun 25 times is still 20 questions.' },
          { q: 'An LLM judge compares pairs of answers and prefers "answer 1" in 9 of 10 pairs. You swap the order within every pair and it prefers "answer 1" in 8 of 10. What have you learned?',
            options: ['The judge is consistent, so its verdicts can be trusted', 'The judge has verbosity bias and prefers the longer answers', 'The judge has position bias: most of its verdicts follow the slot', 'The judge prefers answers written by its own model family'], answer: 2,
            explain: 'If the judge were reading quality, swapping would move its preference to "answer 2" for most pairs. Preferring slot 1 both times means the slot is deciding, and those verdicts should be discarded. Keep only verdicts that survive the swap, exactly as with <b>Ask in both orders</b>. Length and authorship did not change between the two runs, so they cannot explain it.' },
          { q: 'Your eval set has 900 order-status questions, 95% answered correctly, and 100 billing questions, 40% correct. What single score does the dashboard show, and what should you do about it?',
            options: ['67.5%: retrain, the model is failing overall', '89.5%: report billing as its own slice, because it is failing there', '89.5%: ship, since billing is too small a share to matter', '95%: the billing questions are outliers and should be dropped'], answer: 1,
            explain: '(900 × 0.95 + 100 × 0.40) ÷ 1,000 = (855 + 40) ÷ 1,000 = 89.5%. The average is dominated by the large, easy slice. 67.5% is the unweighted mean of the two rates, which is not what a dashboard computes. A slice at 40% is a failing product for those customers, whatever the headline says.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://arxiv.org/abs/2411.00640" target="_blank" rel="noopener">Evan Miller (2024), "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations"</a>: this chapter's maths beat applied to LLM evals in full, including paired differences and what to do when questions come in related groups.`,
            `<a href="https://arxiv.org/abs/2306.05685" target="_blank" rel="noopener">Zheng et al. (2023), "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"</a>: the study that measured position, verbosity and self-enhancement bias in LLM judges, and how well judges agree with people.`,
            `<a href="https://arxiv.org/abs/2404.13076" target="_blank" rel="noopener">Panickssery, Bowman &amp; Feng (2024), "LLM Evaluators Recognize and Favor Their Own Generations"</a>: evidence that self-preference is tied to a model recognising its own writing.`,
            `<a href="https://www.evanmiller.org/how-not-to-run-an-ab-test.html" target="_blank" rel="noopener">Evan Miller (2010), "How Not To Run an A/B Test"</a>: the short essay that taught a generation of engineers not to peek.`,
            `<a href="https://experimentguide.com/" target="_blank" rel="noopener">Kohavi, Tang &amp; Xu (2020), "Trustworthy Online Controlled Experiments"</a>: the standard book on A/B testing, written by people who ran experimentation at Microsoft, Google and LinkedIn.`,
          ]),
        ),
      );
    },
  });
})();
