/* Zero → AGI · Chapter 10 · Pretraining: teaching a model the whole internet
   One objective (next-token prediction), the data funnel, the compute bill, scaling laws,
   and what a base model actually is.
   Interactives: next-token prediction game (be the model, pay the cross-entropy);
   data-filtering funnel animation; scaling-law explorer (click the N–D plane, watch the
   loss and the money); training-run simulator with a scrubbable loss curve and samples. */
(function () {
  ZTA.registerChapter({
    id: '10-pretraining-llms',
    num: 10,
    part: 'III',
    title: 'Pretraining: teaching a model the whole internet',
    tagline: 'One objective, trillions of tokens, a hundred million dollars of electricity: how "guess the next word" becomes knowledge.',
    render(root, ctx) {
      const { h, p, section, sub, callout, ul } = ctx;
      const C = ctx.colors;
      const FONT = '11px Inter, system-ui, sans-serif';
      const FONTB = 'bold 11px Inter, system-ui, sans-serif';

      /* ================================================================== */
      /*  small formatters                                                  */
      /* ================================================================== */
      function fmtBig(x) {
        if (!isFinite(x)) return '—';
        const u = [['T', 1e12], ['B', 1e9], ['M', 1e6], ['k', 1e3]];
        for (const [s, v] of u) if (x >= v) return (x / v < 10 ? (x / v).toFixed(1) : Math.round(x / v)) + s;
        return x.toFixed(0);
      }
      function fmtSci(x) {
        if (!isFinite(x) || x <= 0) return '—';
        const e = Math.floor(Math.log10(x));
        return (x / Math.pow(10, e)).toFixed(1) + 'e' + e;
      }
      function fmtMoney(x) {
        if (!isFinite(x)) return '—';
        if (x >= 1e9) return '$' + (x / 1e9).toFixed(1) + 'B';
        if (x >= 1e6) return '$' + (x / 1e6).toFixed(1) + 'M';
        if (x >= 1e3) return '$' + (x / 1e3).toFixed(0) + 'k';
        return '$' + x.toFixed(0);
      }
      const esc = (t) => String(t).replace(/[&<>]/g, (c) => (c === '&' ? '&amp;' : c === '<' ? '&lt;' : '&gt;'));

      /* ================================================================== */
      /*  INTERACTIVE A — NEXT-TOKEN PREDICTION GAME                        */
      /* ================================================================== */
      function nextTokenGame() {
        const items = [
          { t: 'The capital of France is ___', o: [['Paris', 0.92], ['Lyon', 0.03], ['London', 0.02], ['Berlin', 0.03]], a: 0 },
          { t: 'Once upon a ___', o: [['time', 0.98], ['day', 0.01], ['night', 0.005], ['hill', 0.005]], a: 0 },
          { t: 'She opened the door and saw a ___', o: [['man', 0.40], ['letter', 0.30], ['cat', 0.22], ['dragon', 0.08]], a: 2 },
          { t: 'Water boils at 100 degrees ___', o: [['Celsius', 0.85], ['Fahrenheit', 0.08], ['Kelvin', 0.05], ['hotter', 0.02]], a: 0 },
          { t: 'def add(a, b):\n    return a + ___', o: [['b', 0.93], ['a', 0.04], ['1', 0.02], ['c', 0.01]], a: 0 },
          { t: 'The detective looked at the muddy boots and knew the killer was the ___', o: [['gardener', 0.45], ['butler', 0.30], ['wife', 0.15], ['dog', 0.10]], a: 0 },
          { t: 'Two plus two equals ___', o: [['four', 0.96], ['five', 0.02], ['three', 0.01], ['twenty-two', 0.01]], a: 0 },
          { t: 'He was born in 1990, so in 2020 he turned ___', o: [['thirty', 0.80], ['twenty', 0.10], ['forty', 0.06], ['thirty-one', 0.04]], a: 0 },
          { t: 'Bonjour, comment allez-___', o: [['vous', 0.90], ['tu', 0.06], ['il', 0.03], ['nous', 0.01]], a: 0 },
          { t: 'The stock market crashed, and investors were ___', o: [['worried', 0.35], ['panicking', 0.30], ['devastated', 0.30], ['thrilled', 0.05]], a: 2 },
          { t: 'The mitochondria is the powerhouse of the ___', o: [['cell', 0.95], ['body', 0.03], ['brain', 0.01], ['house', 0.01]], a: 0 },
          { t: "I'll have a coffee with milk and no ___", o: [['sugar', 0.88], ['cream', 0.06], ['ice', 0.04], ['salt', 0.02]], a: 0 },
          { t: 'The meeting is at 3 pm on ___', o: [['Zoom', 0.35], ['Monday', 0.25], ['Tuesday', 0.20], ['Friday', 0.20]], a: 3 },
          { t: 'Breaking: scientists discover new species of ___', o: [['frog', 0.35], ['beetle', 0.30], ['fish', 0.20], ['dinosaur', 0.15]], a: 1 },
          { t: 'To be or not to be, that is the ___', o: [['question', 0.97], ['answer', 0.01], ['point', 0.01], ['problem', 0.01]], a: 0 },
        ];
        /* picked[i] = index the reader chose for sentence i, or -1 if unanswered.
           Keeping it per-sentence means revisiting a sentence never double-counts. */
        const picked = new Array(items.length).fill(-1);
        let idx = 0;

        const counter = h('div', { class: 'muted', style: { fontFamily: 'var(--mono)', fontSize: '.78rem', marginBottom: '6px' } });
        const promptEl = h('pre', { class: 'code', style: { margin: '0 0 12px', fontSize: '1rem', whiteSpace: 'pre-wrap' } });
        const optsBox = h('div');
        const status = h('div', { class: 'muted', style: { fontSize: '.85rem', marginTop: '10px', minHeight: '3em' } });
        const readout = ctx.readout();

        function tally() {
          let done = 0, matched = 0, lossSum = 0;
          items.forEach((it, i) => {
            if (picked[i] < 0) return;
            done++;
            if (picked[i] === it.a) matched++;
            lossSum += -Math.log(Math.max(1e-9, it.o[it.a][1]));
          });
          const avg = done ? lossSum / done : 0;
          readout.set({
            answered: done + ' / ' + items.length,
            'you matched the text': done ? Math.round(100 * matched / done) + '%' : '—',
            "model's avg loss": done ? avg.toFixed(2) : '—',
            "model's perplexity": done ? Math.exp(avg).toFixed(1) : '—',
          });
        }

        function show() {
          const it = items[idx];
          counter.textContent = 'sentence ' + (idx + 1) + ' of ' + items.length;
          promptEl.textContent = it.t;
          optsBox.innerHTML = '';
          const rows = [];

          function reveal() {
            const chose = picked[idx];
            rows.forEach((r, j) => {
              const pj = it.o[j][1];
              r.bar.style.width = (100 * pj).toFixed(0) + '%';
              r.bar.style.background = j === it.a ? C.green : C.accent;
              r.info.textContent = 'p = ' + pj.toFixed(2) + ' · loss ' + (-Math.log(Math.max(1e-9, pj))).toFixed(2);
              r.row.classList.toggle('correct', j === it.a);
              r.row.classList.toggle('wrong', j === chose && chose !== it.a);
            });
            const pTrue = Math.max(1e-9, it.o[it.a][1]);
            const lossTrue = -Math.log(pTrue);
            status.innerHTML = (chose === it.a ? 'You matched the text. ' : 'The text actually continued with <b>' + esc(it.o[it.a][0]) + '</b>. ') +
              'The model put p = ' + pTrue.toFixed(2) + ' on it, so the loss at this one position is −ln(' + pTrue.toFixed(2) + ') = <b>' + lossTrue.toFixed(2) + '</b>' +
              (lossTrue > 1 ? ' — an expensive token. Gradient descent will push this probability up.' : ' — cheap. The model already expected this.');
          }

          it.o.forEach((opt, i) => {
            const bar = h('div', { style: { height: '8px', width: '0%', background: C.accent, borderRadius: '4px', transition: 'width .5s' } });
            const info = h('span', { class: 'muted', style: { fontFamily: 'var(--mono)', fontSize: '.78rem', minWidth: '140px', textAlign: 'right' } });
            const row = h('button', { class: 'quiz-opt', style: { display: 'flex', alignItems: 'center', gap: '12px' } },
              h('span', { style: { minWidth: '110px', fontWeight: 600 } }, opt[0]),
              h('div', { style: { flex: 1, background: 'var(--bg)', borderRadius: '4px', overflow: 'hidden' } }, bar),
              info);
            row.addEventListener('click', () => {
              if (picked[idx] >= 0) return;          // already answered: never re-count
              picked[idx] = i;
              reveal();
              tally();
            });
            rows.push({ row, bar, info });
            optsBox.append(row);
          });

          if (picked[idx] >= 0) reveal();
          else status.textContent = 'Which token comes next? Pick one, then look at what it cost.';
          tally();
        }

        const prevBtn = ctx.button('← Prev', () => { idx = (idx + items.length - 1) % items.length; show(); });
        const nextBtn = ctx.button('Next sentence →', () => { idx = (idx + 1) % items.length; show(); }, 'primary');
        const resetBtn = ctx.button('Restart', () => { picked.fill(-1); idx = 0; show(); });
        show();
        const body = h('div', {}, counter, promptEl, optsBox, status);
        return ctx.figure(body, 'Loss at one position = −ln p(true next token). Perplexity = e<sup>average loss</sup>. The distributions are hand-authored to imitate a well-trained model; a real vocabulary has ~100,000 options, not 4, so real losses are larger.', [prevBtn, nextBtn, resetBtn], readout);
      }

      /* ================================================================== */
      /*  INTERACTIVE B — DATA PIPELINE ANIMATION                           */
      /* ================================================================== */
      function dataPipeline() {
        const STG = [
          { name: 'raw crawl', tok: 100 }, { name: 'language ID', tok: 40 }, { name: 'dedup', tok: 15 },
          { name: 'quality filter', tok: 8 }, { name: 'toxicity / PII', tok: 6 },
        ];
        const [cv, g] = ctx.canvas(720, 300);
        const readout = ctx.readout();
        const gateX = (i) => 90 + i * 130;      // gates 1..4 sit at 220, 350, 480, 610
        const LANE_TOP = 58, LANE_BOT = 178;
        let parts = [], spawnAcc = 0, kept = 0, dropped = 0, running = true, roAcc = 1;

        function spawn() {
          const w = 8 + Math.random() * 8;
          parts.push({
            x: 18, y: LANE_TOP + Math.random() * (LANE_BOT - LANE_TOP - w * 1.3),
            vx: 90 + Math.random() * 50, vy: 0, stage: 0, dead: false, life: 1, w,
            warm: Math.random() < 0.5,
          });
        }
        function reset() { parts = []; spawnAcc = 0; kept = 0; dropped = 0; roAcc = 1; }

        ctx.loop((dt) => {
          if (running) {
            spawnAcc += dt * 18;
            while (spawnAcc > 1 && parts.length < 160) { spawn(); spawnAcc -= 1; }
            if (spawnAcc > 3) spawnAcc = 3;      // don't bank a burst while at the cap
            for (const q of parts) {
              if (q.dead) { q.vy += 300 * dt; q.y += q.vy * dt; q.x += q.vx * 0.3 * dt; q.life -= dt * 1.3; continue; }
              q.x += q.vx * dt;
              const gi = q.stage + 1;
              if (gi < STG.length && q.x >= gateX(gi)) {
                const keep = STG[gi].tok / STG[gi - 1].tok;    // 0.40, 0.375, 0.53, 0.75 → 6% overall
                if (Math.random() < keep) q.stage = gi;
                else { q.dead = true; q.vy = 20; dropped++; }
              }
              if (q.x > 702) { q.life = 0; kept++; }
            }
            parts = parts.filter((q) => q.life > 0);
          }
          /* ---- draw (whole canvas, every frame) ---- */
          g.clearRect(0, 0, cv.W, cv.H);
          g.fillStyle = C.bg; g.fillRect(0, 0, cv.W, cv.H);
          g.fillStyle = 'rgba(124,156,255,0.05)'; g.fillRect(0, 50, 720, 140);
          for (let i = 1; i < STG.length; i++) {
            const x = gateX(i);
            g.fillStyle = C.line; g.fillRect(x - 3, 40, 6, 160);
            g.fillStyle = C.warn; g.font = FONTB; g.textAlign = 'center'; g.textBaseline = 'top';
            g.fillText(STG[i].name, x, 206);
            g.fillStyle = C.text; g.font = FONT; g.fillText('→ ' + STG[i].tok + 'T', x, 222);
          }
          g.fillStyle = C.warn; g.font = FONTB; g.textAlign = 'left'; g.textBaseline = 'top';
          g.fillText(STG[0].name, 14, 206);
          g.fillStyle = C.text; g.font = FONT; g.fillText('~' + STG[0].tok + 'T tokens', 14, 222);
          g.fillStyle = C.green; g.font = FONTB; g.textAlign = 'right'; g.fillText('kept', 706, 206);
          g.fillStyle = C.text; g.font = FONT; g.fillText('≈ ' + STG[STG.length - 1].tok + 'T web', 706, 222);
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          g.fillText('+ mixed with code, books, papers, maths, dialogue and synthetic data → a ~15T-token training set (Llama 3 scale)', 14, 250);
          g.fillText('rejected pages fall out of the lane', 14, 268);
          for (const q of parts) {
            g.fillStyle = q.dead
              ? 'rgba(251,113,133,' + Math.max(0, q.life).toFixed(3) + ')'
              : (q.warm ? 'rgba(124,156,255,0.85)' : 'rgba(56,217,169,0.85)');
            g.fillRect(q.x, q.y, q.w, q.w * 1.3);
          }
          roAcc += dt;
          if (roAcc > 0.2) {                      // don't rebuild the readout DOM 60× a second
            roAcc = 0;
            const seen = kept + dropped;
            readout.set({
              'pages resolved': seen, kept, rejected: dropped,
              survival: seen ? (100 * kept / seen).toFixed(1) + '%' : '—',
              'in flight': parts.length,
            });
          }
        });

        const pauseBtn = ctx.button('❚❚ Pause', () => {
          running = !running;
          pauseBtn.textContent = running ? '❚❚ Pause' : '▶ Resume';
        });
        return ctx.figure(cv, 'Each gate keeps a fraction equal to the ratio of the token counts printed under it, so the survival rate converges on 6%. Real pipelines (FineWeb, Llama 3, Dolma) differ in order and thresholds, but the shape — keep a few per cent of the raw crawl — is universal.', [pauseBtn, ctx.button('Reset counters', reset)], readout);
      }

      /* ================================================================== */
      /*  INTERACTIVE C — SCALING-LAW EXPLORER                              */
      /* ================================================================== */
      function scalingExplorer() {
        const FITS = {
          published: { E: 1.69, A: 406.4, B: 410.7, alpha: 0.34, beta: 0.28, label: 'Chinchilla published fit (2022)' },
          refit: { E: 1.8172, A: 482.01, B: 2085.43, alpha: 0.3478, beta: 0.3658, label: 'Epoch AI replication refit (2024)' },
        };
        let fit = FITS.published;
        const st = { logN: 11, logD: 10.5, price: 2.5, mfu: 0.4 };
        const L = (N, D) => fit.E + fit.A / Math.pow(N, fit.alpha) + fit.B / Math.pow(D, fit.beta);
        /* Minimise L subject to 6ND = C.  Substituting D = K/N (K = C/6) and setting dL/dN = 0
           gives N* = (αA / βB)^(1/(α+β)) · K^(β/(α+β)),  D* = K / N*. */
        function optimal(Cc) {
          const K = Math.max(1, Cc / 6);
          const G = Math.pow(fit.alpha * fit.A / (fit.beta * fit.B), 1 / (fit.alpha + fit.beta));
          const N = G * Math.pow(K, fit.beta / (fit.alpha + fit.beta));
          return { N, D: K / N };
        }

        const [cv, g] = ctx.canvas(720, 380);
        const readout = ctx.readout();
        const P1 = { x: 52, y: 26, w: 296, h: 292, x0: 7, x1: 12, y0: 8, y1: 14 };
        const P2 = { x: 424, y: 26, w: 276, h: 292, x0: 15.5, x1: 27, y0: 1.5, y1: 6.5 };
        const X1 = (lN) => P1.x + (lN - P1.x0) / (P1.x1 - P1.x0) * P1.w;
        const Y1 = (lD) => P1.y + P1.h - (lD - P1.y0) / (P1.y1 - P1.y0) * P1.h;
        const X2 = (lc) => P2.x + (lc - P2.x0) / (P2.x1 - P2.x0) * P2.w;
        const Y2 = (l) => P2.y + P2.h - (ctx.clamp(l, P2.y0, P2.y1) - P2.y0) / (P2.y1 - P2.y0) * P2.h;

        const REFS = [
          { nm: 'GPT-3', N: 1.75e11, D: 3e11, ly: -7 },
          { nm: 'Chinchilla', N: 7e10, D: 1.4e12, ly: -7 },
          { nm: 'Llama 3 8B', N: 8e9, D: 1.5e13, ly: 14 },
          { nm: 'Llama 3 405B', N: 4.05e11, D: 1.5e13, ly: -7 },
        ];

        /* The loss field only changes when the constants change, so bake it once into an
           offscreen bitmap instead of running ~2,300 fillRects on every pointer move. */
        const HR = 150;
        const heat = document.createElement('canvas');
        heat.width = HR; heat.height = HR;
        const hg = heat.getContext('2d');
        let heatFor = null;
        function bandRGB(l) {
          const t = ctx.clamp((l - 1.7) / 3.3, 0, 1);          // 0 = best loss
          const band = Math.floor(t * 12) / 12;                 // quantised → visible iso-loss bands
          const a = 0.22 + 0.3 * band;
          const r = ctx.lerp(56, 251, band), gg = ctx.lerp(217, 113, band), b = ctx.lerp(169, 133, band);
          return [Math.round(10 + (r - 10) * a), Math.round(14 + (gg - 14) * a), Math.round(22 + (b - 22) * a)];
        }
        function buildHeat() {
          if (heatFor === fit) return;
          heatFor = fit;
          const img = hg.createImageData(HR, HR), px = img.data;
          let k = 0;
          for (let row = 0; row < HR; row++) {
            const lD = P1.y1 - (row + 0.5) / HR * (P1.y1 - P1.y0);   // row 0 = top = most tokens
            for (let col = 0; col < HR; col++) {
              const lN = P1.x0 + (col + 0.5) / HR * (P1.x1 - P1.x0);
              const c = bandRGB(L(Math.pow(10, lN), Math.pow(10, lD)));
              px[k++] = c[0]; px[k++] = c[1]; px[k++] = c[2]; px[k++] = 255;
            }
          }
          hg.putImageData(img, 0, 0);
        }

        function draw() {
          buildHeat();
          g.clearRect(0, 0, cv.W, cv.H);
          g.fillStyle = C.bg; g.fillRect(0, 0, cv.W, cv.H);
          g.imageSmoothingEnabled = false;
          g.drawImage(heat, P1.x, P1.y, P1.w, P1.h);
          g.imageSmoothingEnabled = true;

          /* ---- iso-compute diagonals: logN + logD = logC − log10(6) ---- */
          g.lineWidth = 1; g.font = FONT; g.textAlign = 'left'; g.textBaseline = 'bottom';
          for (let lc = 16; lc <= 26; lc += 2) {
            const k = lc - Math.log10(6);
            const lo = Math.max(P1.x0, k - P1.y1), hi = Math.min(P1.x1, k - P1.y0);
            if (hi <= lo) continue;                                  // this diagonal misses the panel
            g.strokeStyle = 'rgba(230,235,245,0.25)'; g.setLineDash([4, 4]);
            g.beginPath(); g.moveTo(X1(lo), Y1(k - lo)); g.lineTo(X1(hi), Y1(k - hi)); g.stroke();
            g.setLineDash([]);
            if (hi - lo > 0.6) {
              g.fillStyle = 'rgba(230,235,245,0.6)';
              g.fillText('1e' + lc, X1(lo) + 3, Y1(k - lo) - 3);
            }
          }

          /* ---- compute-optimal frontier (clipped to the panel) ---- */
          g.save();
          g.beginPath(); g.rect(P1.x, P1.y, P1.w, P1.h); g.clip();
          g.strokeStyle = '#ffffff'; g.lineWidth = 1.5; g.beginPath();
          let first = true;
          for (let lc = 14; lc <= 29; lc += 0.25) {
            const o = optimal(Math.pow(10, lc));
            const x = X1(Math.log10(o.N)), y = Y1(Math.log10(o.D));
            if (first) { g.moveTo(x, y); first = false; } else g.lineTo(x, y);
          }
          g.stroke();
          g.restore();

          /* ---- reference runs ---- */
          g.font = FONT; g.textBaseline = 'middle';
          for (const r of REFS) {
            const x = X1(Math.log10(r.N)), y = Y1(Math.log10(r.D));
            g.fillStyle = C.warn; g.beginPath(); g.arc(x, y, 3, 0, Math.PI * 2); g.fill();
            g.fillStyle = C.muted;
            if (x > P1.x + P1.w - 74) { g.textAlign = 'right'; g.fillText(r.nm, x - 6, y + r.ly); }
            else { g.textAlign = 'left'; g.fillText(r.nm, x + 6, y + r.ly); }
          }

          /* ---- the reader's point ---- */
          const N = Math.pow(10, st.logN), D = Math.pow(10, st.logD), Cc = 6 * N * D, loss = L(N, D);
          const ux = X1(st.logN), uy = Y1(st.logD);
          g.beginPath(); g.arc(ux, uy, 6, 0, Math.PI * 2);
          g.fillStyle = C.accent; g.fill(); g.strokeStyle = '#ffffff'; g.lineWidth = 2; g.stroke();

          /* ---- left axes ---- */
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P1.x, P1.y, P1.w, P1.h);
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'center'; g.textBaseline = 'top';
          for (let e = P1.x0; e <= P1.x1 + 1e-9; e += 1) g.fillText(fmtBig(Math.pow(10, e)), X1(e), P1.y + P1.h + 5);
          g.fillText('parameters N', P1.x + P1.w / 2, P1.y + P1.h + 20);
          g.textAlign = 'right'; g.textBaseline = 'middle';
          for (let e = P1.y0; e <= P1.y1 + 1e-9; e += 1) g.fillText(fmtBig(Math.pow(10, e)), P1.x - 5, Y1(e));
          g.save(); g.translate(12, P1.y + P1.h / 2); g.rotate(-Math.PI / 2);
          g.textAlign = 'center'; g.textBaseline = 'top'; g.fillText('training tokens D', 0, 0); g.restore();
          g.textAlign = 'left'; g.textBaseline = 'bottom'; g.fillStyle = C.muted;
          g.fillText('colour = loss · white line = compute-optimal', P1.x, P1.y - 6);

          /* ---- right panel: best achievable loss vs compute ---- */
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P2.x, P2.y, P2.w, P2.h);
          g.save();
          g.beginPath(); g.rect(P2.x, P2.y, P2.w, P2.h); g.clip();
          for (const r of REFS) {
            const x = X2(Math.log10(6 * r.N * r.D));
            g.strokeStyle = 'rgba(251,191,36,0.3)'; g.lineWidth = 1;
            g.beginPath(); g.moveTo(x, P2.y); g.lineTo(x, P2.y + P2.h); g.stroke();
          }
          g.strokeStyle = C.green; g.lineWidth = 2; g.beginPath(); first = true;
          for (let lc = P2.x0; lc <= P2.x1 + 1e-9; lc += 0.2) {
            const o = optimal(Math.pow(10, lc)), y = Y2(L(o.N, o.D));
            if (first) { g.moveTo(X2(lc), y); first = false; } else g.lineTo(X2(lc), y);
          }
          g.stroke();
          g.strokeStyle = 'rgba(148,163,184,0.5)'; g.lineWidth = 1; g.setLineDash([3, 3]);
          g.beginPath(); g.moveTo(P2.x, Y2(fit.E)); g.lineTo(P2.x + P2.w, Y2(fit.E)); g.stroke(); g.setLineDash([]);
          g.restore();
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left'; g.textBaseline = 'bottom';
          g.fillText('irreducible loss E = ' + fit.E, P2.x + 5, Y2(fit.E) - 3);

          const o = optimal(Cc), bestL = L(o.N, o.D);
          const px = ctx.clamp(X2(Math.log10(Cc)), P2.x, P2.x + P2.w), py = Y2(loss);
          if (loss - bestL > 0.02) {
            g.strokeStyle = C.danger; g.lineWidth = 1;
            g.beginPath(); g.moveTo(px, py); g.lineTo(px, Y2(bestL)); g.stroke();
            g.fillStyle = C.danger; g.textAlign = px > P2.x + P2.w - 90 ? 'right' : 'left'; g.textBaseline = 'middle';
            g.fillText('+' + (loss - bestL).toFixed(2) + ' loss wasted', px + (px > P2.x + P2.w - 90 ? -8 : 8), (py + Y2(bestL)) / 2);
          }
          g.beginPath(); g.arc(px, py, 6, 0, Math.PI * 2);
          g.fillStyle = C.accent; g.fill(); g.strokeStyle = '#ffffff'; g.lineWidth = 2; g.stroke();
          if (loss > P2.y1) {
            g.fillStyle = C.warn; g.font = FONT; g.textAlign = 'center'; g.textBaseline = 'top';
            g.fillText('↑ loss ' + loss.toFixed(2) + ' is off the top of this scale', P2.x + P2.w / 2, P2.y + 4);
          }
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'center'; g.textBaseline = 'top';
          for (let e = 16; e <= 26; e += 2) g.fillText('1e' + e, X2(e), P2.y + P2.h + 5);
          g.fillText('training compute C = 6·N·D (FLOPs)', P2.x + P2.w / 2, P2.y + P2.h + 20);
          g.textAlign = 'right'; g.textBaseline = 'middle';
          for (let l = 2; l <= 6; l += 1) g.fillText(l.toFixed(0), P2.x - 5, Y2(l));
          g.textAlign = 'left'; g.textBaseline = 'bottom';
          g.fillText('green = best loss reachable at each budget', P2.x, P2.y - 6);

          /* ---- numbers ---- */
          const gpuHours = Cc / (1e15 * st.mfu) / 3600;
          readout.set({
            N: fmtBig(N), D: fmtBig(D), 'tokens/param': (D / N).toFixed(0),
            C: fmtSci(Cc) + ' FLOPs', loss: loss.toFixed(3), perplexity: Math.exp(loss).toFixed(1),
            'optimal N*': fmtBig(o.N), 'optimal D*': fmtBig(o.D),
            'H100-hours': fmtBig(gpuHours), cost: fmtMoney(gpuHours * st.price),
            'days on 10k GPUs': (gpuHours / 10000 / 24).toFixed(1),
          });
        }

        const snap = (v, step) => Math.round(v / step) * step;
        const fmtLog = (v) => fmtBig(Math.pow(10, v));
        const sN = ctx.slider({ label: 'parameters N', min: P1.x0, max: P1.x1, step: 0.05, value: st.logN, fmt: fmtLog, onChange: (v) => { st.logN = v; draw(); } });
        const sD = ctx.slider({ label: 'training tokens D', min: P1.y0, max: P1.y1, step: 0.05, value: st.logD, fmt: fmtLog, onChange: (v) => { st.logD = v; draw(); } });
        const snapBtn = ctx.button('Snap to compute-optimal', () => {
          const o = optimal(6 * Math.pow(10, st.logN + st.logD));
          st.logN = ctx.clamp(snap(Math.log10(o.N), 0.05), P1.x0, P1.x1);
          st.logD = ctx.clamp(snap(Math.log10(o.D), 0.05), P1.y0, P1.y1);
          sN.value = st.logN; sD.value = st.logD; draw();
        }, 'primary');
        const llamaBtn = ctx.button('Jump to Llama 3 8B', () => {
          st.logN = snap(Math.log10(8e9), 0.05); st.logD = snap(Math.log10(1.5e13), 0.05);
          sN.value = st.logN; sD.value = st.logD; draw();
        });
        const priceSl = ctx.slider({ label: '$ per GPU-hour', min: 1, max: 6, step: 0.1, value: st.price, fmt: (v) => '$' + (+v).toFixed(1), onChange: (v) => { st.price = v; draw(); } });
        const mfuSl = ctx.slider({ label: 'MFU (utilisation)', min: 0.2, max: 0.6, step: 0.01, value: st.mfu, fmt: (v) => Math.round(v * 100) + '%', onChange: (v) => { st.mfu = v; draw(); } });
        const fitSel = ctx.select({
          label: 'loss-curve constants',
          options: [{ value: 'published', label: FITS.published.label }, { value: 'refit', label: FITS.refit.label }],
          value: 'published',
          onChange: (v) => { fit = FITS[v] || FITS.published; draw(); },
        });

        /* pointer (mouse + touch) picking on the N–D plane */
        cv.style.touchAction = 'none';
        cv.style.cursor = 'crosshair';
        let dragging = false;
        function pick(ev) {
          const q = cv.pos(ev);
          if (q.x < P1.x - 8 || q.x > P1.x + P1.w + 8 || q.y < P1.y - 8 || q.y > P1.y + P1.h + 8) return false;
          st.logN = ctx.clamp(snap(P1.x0 + (q.x - P1.x) / P1.w * (P1.x1 - P1.x0), 0.05), P1.x0, P1.x1);
          st.logD = ctx.clamp(snap(P1.y0 + (P1.y + P1.h - q.y) / P1.h * (P1.y1 - P1.y0), 0.05), P1.y0, P1.y1);
          sN.value = st.logN; sD.value = st.logD;
          draw();
          return true;
        }
        cv.addEventListener('pointerdown', (ev) => {
          if (!pick(ev)) return;
          dragging = true;
          if (ev.preventDefault) ev.preventDefault();
          try { cv.setPointerCapture(ev.pointerId); } catch (e) { /* not supported here */ }
        });
        cv.addEventListener('pointermove', (ev) => { if (dragging) pick(ev); });
        const endDrag = (ev) => {
          if (!dragging) return;
          dragging = false;
          try { cv.releasePointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
        };
        cv.addEventListener('pointerup', endDrag);
        cv.addEventListener('pointercancel', endDrag);

        draw();
        return ctx.figure(cv, 'Left: the (N, D) plane. Colour is the loss, the white line is the compute-optimal frontier, the dashed diagonals are lines of equal compute, and the blue dot is your run — click or drag anywhere on the plane to move it. L(N, D) = E + A/N<sup>α</sup> + B/D<sup>β</sup>. The defaults are the constants printed in the Chinchilla paper (E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28). Careful: that published fit implies about 80 tokens per parameter at 10<sup>23</sup> FLOPs, not 20 — the paper\'s headline "≈20 tokens per parameter" came from its other two estimation methods, and a 2024 replication that refit these constants recovers it (switch the dropdown and watch the white line move). Cost assumes one H100 at 10<sup>15</sup> FLOP/s dense bfloat16.', [sN, sD, snapBtn, llamaBtn, priceSl, mfuSl, fitSel], readout);
      }

      /* ================================================================== */
      /*  INTERACTIVE D — TRAINING-RUN SIMULATOR                            */
      /* ================================================================== */
      function trainingRun() {
        const PROMPT = 'The Eiffel Tower is located in';
        const STAGES = [
          { at: 6, name: '1M tokens · random characters', text: ' tqe,h  ni erlta oTs.h eeai fnr,t o ela  ihsnT rew oa tt.e hdrsi neoo lae' },
          { at: 8, name: '100M tokens · words, no grammar', text: ' the of the in and a to is the Paris the on of tower a the is, and the the in city of the France the and.' },
          { at: 9, name: '1B tokens · grammar, no sense', text: ' a large city that is the tower of the river. The tower is located in the city and the tower is the tower of the world in the year of the building.' },
          { at: 10, name: '10B tokens · fluent but unreliable', text: ' Paris, France, on the banks of the Seine. It was built in 1912 by the architect Alexandre Eiffel for the Paris Olympics and is 450 metres tall, making it the tallest building in Europe.' },
          { at: 12, name: '1T tokens · fluent and mostly right', text: ' Paris, France, on the Champ de Mars near the Seine. It was designed by Gustave Eiffel\'s engineering company and completed in 1889 for the World\'s Fair. At about 330 metres it was the tallest man-made structure in the world until 1930.' },
        ];
        /* how long each checkpoint's sample takes to type out (in decades of tokens) */
        STAGES.forEach((s, i) => { s.span = STAGES[i + 1] ? STAGES[i + 1].at - s.at : 0.3; });

        const N = 1e9, E = 1.69, A = 406.4, B = 410.7, al = 0.34, be = 0.28;
        const FLOOR = E + A / Math.pow(N, al);          // 2.044: the best a 1B model can ever do
        const lossAt = (D) => E + A / Math.pow(N, al) + B / Math.pow(D, be);
        const T0 = 6, T1 = 12.4, DT = 0.02;

        /* Precompute the whole run once, with reproducible correlated jitter, so that playing
           and scrubbing show exactly the same curve and no RNG runs inside the frame loop.
           (A 32-bit LCG via Math.imul — plain `seed * 1103515245` overflows 2^53 and degenerates.) */
        const curve = [];
        (function buildCurve() {
          let seed = 20240719 >>> 0;
          const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
          let jit = 0;
          for (let l = T0; l <= T1 + 1e-9; l += DT) {
            jit = jit * 0.72 + (rnd() - 0.5) * 0.28;
            const base = lossAt(Math.pow(10, l));
            curve.push([l, ctx.clamp(base + jit * Math.min(1, base / 3), FLOOR - 0.02, 11.4)]);
          }
        })();
        const idxAt = (l) => ctx.clamp(Math.round((l - T0) / DT), 0, curve.length - 1);

        const [cv, g] = ctx.canvas(720, 262);
        const sample = h('div', { style: { fontFamily: 'var(--mono)', fontSize: '.88rem', lineHeight: 1.6, padding: '12px 14px', borderTop: '1px solid var(--line)', minHeight: '104px', whiteSpace: 'pre-wrap' } });
        const readout = ctx.readout();
        const P = { x: 52, y: 22, w: 646, h: 196, x0: T0, x1: T1, y0: 1.5, y1: 11.5 };
        const X = (l) => P.x + (l - P.x0) / (P.x1 - P.x0) * P.w;
        const Y = (v) => P.y + P.h - (ctx.clamp(v, P.y0, P.y1) - P.y0) / (P.y1 - P.y0) * P.h;

        let logT = T0, playing = false, scrubbing = false, speed = 1, roAcc = 1, lastSample = '';

        const playBtn = ctx.button('▶ Play', () => {
          if (logT >= T1 - 1e-9) logT = T0;         // finished: rewind, then play
          setPlaying(!playing);
        }, 'primary');
        function setPlaying(v) { playing = v; playBtn.textContent = v ? '❚❚ Pause' : '▶ Play'; }
        const resetBtn = ctx.button('Reset', () => { logT = T0; setPlaying(false); lastSample = ''; roAcc = 1; });
        const speedSl = ctx.slider({ label: 'speed', min: 0.25, max: 3, step: 0.25, value: 1, fmt: (v) => (+v).toFixed(2) + '×', onChange: (v) => { speed = v; } });

        function stageFor(l) { let s = STAGES[0]; for (const q of STAGES) if (l >= q.at) s = q; return s; }

        function draw() {
          g.clearRect(0, 0, cv.W, cv.H);
          g.fillStyle = C.bg; g.fillRect(0, 0, cv.W, cv.H);
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
          g.font = FONT; g.textAlign = 'center'; g.textBaseline = 'top';
          for (let e = 6; e <= 12; e++) {
            g.strokeStyle = 'rgba(36,48,68,0.7)';
            g.beginPath(); g.moveTo(X(e), P.y); g.lineTo(X(e), P.y + P.h); g.stroke();
            g.fillStyle = C.muted; g.fillText(fmtBig(Math.pow(10, e)), X(e), P.y + P.h + 5);
          }
          g.fillStyle = C.muted; g.fillText('tokens seen (log scale — each gridline is 10× the last)', P.x + P.w / 2, P.y + P.h + 20);
          g.textAlign = 'right'; g.textBaseline = 'middle';
          for (let v = 2; v <= 11; v += 3) g.fillText(v.toFixed(0), P.x - 5, Y(v));
          g.save(); g.translate(12, P.y + P.h / 2); g.rotate(-Math.PI / 2);
          g.textAlign = 'center'; g.textBaseline = 'top'; g.fillText('loss (nats/token)', 0, 0); g.restore();

          /* the floor this model size can never beat */
          g.strokeStyle = 'rgba(148,163,184,0.55)'; g.setLineDash([3, 3]);
          g.beginPath(); g.moveTo(P.x, Y(FLOOR)); g.lineTo(P.x + P.w, Y(FLOOR)); g.stroke(); g.setLineDash([]);
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left'; g.textBaseline = 'bottom';
          g.fillText('floor for a 1B model: ' + FLOOR.toFixed(2), P.x + 5, Y(FLOOR) - 3);

          /* checkpoints */
          for (const s of STAGES) {
            g.strokeStyle = 'rgba(251,191,36,0.4)'; g.setLineDash([3, 3]);
            g.beginPath(); g.moveTo(X(s.at), P.y); g.lineTo(X(s.at), P.y + P.h); g.stroke(); g.setLineDash([]);
            g.fillStyle = C.warn; g.textAlign = 'center'; g.textBaseline = 'top';
            g.fillText('ckpt ' + fmtBig(Math.pow(10, s.at)), X(s.at), P.y + 3);
          }

          /* the run so far */
          const upto = idxAt(logT);
          g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
          if (upto > 0) {
            g.strokeStyle = C.accent; g.lineWidth = 2; g.beginPath();
            for (let i = 0; i <= upto; i++) { const q = curve[i]; if (i === 0) g.moveTo(X(q[0]), Y(q[1])); else g.lineTo(X(q[0]), Y(q[1])); }
            g.stroke();
          }
          const cur = curve[upto];
          g.beginPath(); g.arc(X(cur[0]), Y(cur[1]), 5, 0, Math.PI * 2);
          g.fillStyle = C.accent; g.fill(); g.strokeStyle = C.bg; g.lineWidth = 1.5; g.stroke();
          g.restore();

          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'right'; g.textBaseline = 'bottom';
          g.fillText('drag anywhere on the chart to scrub through the run', P.x + P.w - 5, P.y + P.h - 5);
        }

        function updateSample() {
          const s = stageFor(logT);
          const frac = ctx.clamp((logT - s.at) / s.span * 1.6, 0, 1);
          const n = Math.floor(s.text.length * frac);
          const html = '<span style="color:' + C.warn + '">' + esc(s.name) + '</span>\n' +
            '<span style="color:' + C.muted + '">' + esc(PROMPT) + '</span>' +
            '<span style="color:' + C.text + '">' + esc(s.text.slice(0, n)) + '</span>' +
            '<span style="color:' + C.accent + '">▌</span>';
          if (html !== lastSample) { sample.innerHTML = html; lastSample = html; }
        }

        /* pointer scrubbing — works with mouse and touch */
        cv.style.touchAction = 'none';
        cv.style.cursor = 'ew-resize';
        function scrubTo(ev) {
          const q = cv.pos(ev);
          logT = ctx.clamp(T0 + (q.x - P.x) / P.w * (T1 - T0), T0, T1);
        }
        cv.addEventListener('pointerdown', (ev) => {
          scrubbing = true; setPlaying(false);
          if (ev.preventDefault) ev.preventDefault();
          try { cv.setPointerCapture(ev.pointerId); } catch (e) { /* not supported here */ }
          scrubTo(ev);
        });
        cv.addEventListener('pointermove', (ev) => { if (scrubbing) scrubTo(ev); });
        const endScrub = (ev) => {
          if (!scrubbing) return;
          scrubbing = false;
          try { cv.releasePointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
        };
        cv.addEventListener('pointerup', endScrub);
        cv.addEventListener('pointercancel', endScrub);

        ctx.loop((dt) => {
          if (playing && !scrubbing) {
            logT += dt * 0.30 * speed;
            if (logT >= T1) { logT = T1; setPlaying(false); }
          }
          draw();
          updateSample();
          roAcc += dt;
          if (roAcc > 0.15) {
            roAcc = 0;
            const D = Math.pow(10, logT), loss = curve[idxAt(logT)][1];
            readout.set({
              'tokens seen': fmtBig(D), loss: loss.toFixed(2), perplexity: Math.exp(loss).toFixed(1),
              'FLOPs so far': fmtSci(6 * N * D), 'above the floor': (loss - FLOOR).toFixed(2),
            });
          }
        });

        const body = h('div', {}, cv, sample);
        const fig = ctx.figure(body, 'A 1-billion-parameter model\'s loss under the Chinchilla formula, with reproducible noise. The samples are hand-written archetypes of each stage, not real generations; in a real run the "fluent but unreliable" phase is where most of the apparent progress happens and where most of the hallucinations are born.', [playBtn, resetBtn, speedSl], readout);
        const fb = fig.querySelector('.figure-body');
        if (fb) fb.style.padding = '0';
        return fig;
      }

      /* ================================================================== */
      /*  PROSE                                                             */
      /* ================================================================== */
      root.append(
        p('Ask a modern language model to translate a Portuguese poem, debug a Rust program, explain the causes of the Thirty Years\' War and then write a limerick about them. It will do all four. Nobody wrote a translation module, a Rust module or a history module. Nobody labelled a single example of "good limerick". Every one of those abilities fell out of a single, almost stupid training task, applied to a slice of the internet so large that reading it aloud would take a hundred thousand years.'),
        p('That task is: <b>given some text, guess the next token</b>. This chapter is about why such a small objective produces such a large result, what it costs to run it at scale, and what the thing you get at the end actually is (spoiler: not yet an assistant). The transformer from chapter 7 is the machine; this chapter is the fuel and the factory.'),

        section('One objective: predict the next token',
          p('Take a document. Cut it at a random point. Show the model everything before the cut and ask for a probability distribution over what comes next: one number for every token in the vocabulary (about 100,000 of them), summing to 1. Compare that distribution with what actually came next. Penalise the model by how little probability it put on the truth. Nudge every weight to make that penalty smaller. Repeat, ten trillion times.'),
          p('Why does this teach anything beyond spelling? Because <b>predicting text well requires understanding what the text is about</b>. To predict the next word of "the boiling point of water at sea level is 100 degrees", you need a fact. To predict the next line of a Python function, you need to know what the function is for. To predict the word after "so the total cost is", you need to have added up the numbers. And, in the example Ilya Sutskever likes to give: to predict the name that follows "and the murderer was…" on the last page of a mystery novel, you need to have solved the murder. Grammar, facts, arithmetic, code, the structure of arguments, a rough model of how people and objects behave: all of it lowers the loss, so all of it gets learned, as a <em>side effect</em>.'),
          p('One detail makes this affordable. You might imagine the model being shown a prefix, guessing one token, and starting over. It is not. Because of the causal mask from chapter 7, a single forward pass over a 8,192-token document produces 8,192 predictions at once — at every position, a full distribution over what comes next, each one allowed to look only leftwards. So one pass yields 8,192 training signals instead of one. This trick has a name, <em>teacher forcing</em>: during training the model is always fed the <i>real</i> previous tokens, never its own guesses, which is why training parallelises and generation does not.'),
          ctx.code('# The whole of pretraining, in six lines of PyTorch.\n# ids: (batch, block_size + 1) token ids sliced out of one long packed stream of text\nfor ids in loader:\n    x, y = ids[:, :-1], ids[:, 1:]      # inputs, and the same text shifted left by one\n    logits = model(x)                   # (batch, block_size, vocab_size) — every position at once\n    loss = F.cross_entropy(             # mean of −log p(true token) over the whole batch\n        logits.flatten(0, 1),           # (batch * block_size, vocab_size)\n        y.flatten())                    # (batch * block_size,)\n    loss.backward()                     # one gradient for every one of the N parameters\n    optimizer.step(); optimizer.zero_grad(set_to_none=True)'),
          p('That is it. There is no other objective, no labels, no human in the loop. The data labels itself, which is why the internet works as a training set at all — this is what people mean by <em>self-supervised</em> learning.'),

          sub('Measuring it: cross-entropy and perplexity',
            p('The penalty is the <em>cross-entropy loss</em>: −log of the probability the model assigned to the correct token. If the model said 50% and was right, the loss is −log(0.5) ≈ 0.69. If it said 10%, the loss is 2.3. If it said 1%, 4.6. The loss is averaged over every token in the batch. A random guess over 100,000 tokens costs log(100,000) ≈ 11.5; a good model on typical web text averages roughly 2 per token (the exact number depends on the tokenizer and the data).'),
            p('There is a friendlier version of the same number. <em>Perplexity</em> = e<sup>loss</sup>, and it reads as "the model is, on average, as unsure as if it were choosing uniformly among this many options". Loss 2.3 means perplexity 10: the model behaves as though every token were a ten-way multiple-choice question. Loss 11.5 means perplexity 100,000: pure guessing. Every training curve you see in a paper is one of these two numbers going down.'),
            p('Here is the arithmetic in full, on one six-token sentence. The model reads <code class="inline">The cat sat on the mat</code> and makes five predictions — one at every position after the first:'),
            ctx.table(
              ['position', 'context the model can see', 'token that really came next', 'p the model gave it', 'loss = −ln p'],
              [
                ['1', '<code class="inline">The</code>', 'cat', '0.02', '3.91'],
                ['2', '<code class="inline">The cat</code>', 'sat', '0.10', '2.30'],
                ['3', '<code class="inline">The cat sat</code>', 'on', '0.60', '0.51'],
                ['4', '<code class="inline">The cat sat on</code>', 'the', '0.75', '0.29'],
                ['5', '<code class="inline">The cat sat on the</code>', 'mat', '0.15', '1.90'],
              ]),
            p('Add them: 3.91 + 2.30 + 0.51 + 0.29 + 1.90 = <b>8.91</b>. Divide by 5 predictions: the loss for this sentence is <b>1.78</b> nats per token, a perplexity of e<sup>1.78</sup> ≈ <b>5.9</b>. Notice where the cost lives. "on" after "The cat sat" was nearly free; "cat" after "The" was catastrophic, and rightly so — the model had no way to know, and a good model should not be confident there. Backpropagation now pushes up the probability of every one of those five true tokens, and pulls down everything else, a fraction of a percent at a time. That is one gradient step. A frontier run does a few million of them.'),
          ),
        ),

        callout('tryit', 'Try it: be the language model',
          'Fifteen sentences, each missing its last token. Pick the one you think comes next, then look at the "model\'s" distribution (hand-authored for this demo) and the loss −ln p for every option. <b>Notice two things.</b> When the true token is forced by the text (<i>"Once upon a ___"</i>) the loss is almost zero. When the text is genuinely ambiguous (<i>"The meeting is at 3 pm on ___"</i>) even a perfect model pays a large loss, because <b>the uncertainty is in the world, not in the model</b>. That irreducible share is why the loss curve later in this chapter flattens instead of reaching zero. Your running average loss and perplexity are at the bottom.'),
        nextTokenGame(),

        section('The data: ten trillion tokens, mostly thrown away',
          p('The raw material is the open web, and the standard starting point is <em>Common Crawl</em>, a non-profit that has been saving snapshots of billions of pages since 2008. A snapshot is a few hundred terabytes of HTML, and most of it is junk: navigation menus, cookie banners, SEO spam, the same Wikipedia article mirrored a thousand times, machine-translated product listings. The single biggest lever in pretraining is <b>what you do with the junk</b>.'),
          p('A modern pipeline runs the crawl through a series of gates. <b>Language identification</b> keeps the languages you want. <b>Deduplication</b> removes exact and near-duplicate documents, because a model that sees the same paragraph ten thousand times memorises it instead of learning from it. <b>Quality classifiers</b>, often small models trained to recognise "text that looks like a good textbook or a well-written article", score every page and drop the worst. <b>Toxicity and personal-data filters</b> drop the rest. Of every hundred tokens that go in, something like five to ten come out. Those survivors are then <em>mixed</em> with curated sources in deliberate proportions: books, academic papers, a lot of source code (GitHub), maths, and dialogue, because each one teaches something the web does not. Llama 3 was pretrained on about 15 trillion tokens after all of this; frontier models in 2025 are believed to use similar or larger amounts, increasingly supplemented with synthetic data written by earlier models.'),
          p('One more preprocessing step matters: the <em>tokenizer</em> itself is trained on a sample of this data (chapter 6 showed how byte-pair encoding works). Its vocabulary, typically 100,000 to 250,000 pieces, is frozen before pretraining begins, and it determines what "one token" even means for the rest of the model\'s life. This is why a model can be oddly bad at counting letters: it has never seen letters, only pieces.'),
        ),

        callout('tryit', 'Watch the funnel',
          'Each vertical bar is a filter, and each block is a page. Watch how few survive to the right-hand side, and read the running <b>survival</b> figure in the numbers below: it settles near 6%. <b>Try this:</b> pause it mid-flight and count how many blocks are left in the last lane compared with the first. The numbers on the gates are illustrative of a typical pipeline, not from any one lab.'),
        dataPipeline(),

        callout('example', 'Where you have already met a raw base model',
          'The grey "ghost text" that finishes your line in an IDE is very close to a naked next-token predictor: it is not answering you, it is continuing you. So was Gmail\'s Smart Compose, and so was the first GitHub Copilot, which was a code-pretrained GPT-3 variant with almost no assistant training on top — which is exactly why it was brilliant at completing a function and useless at being asked a question. Base models are still shipped deliberately: Llama, Qwen, Mistral and Gemma all publish a <code class="inline">-base</code> checkpoint alongside the <code class="inline">-instruct</code> one, because researchers want the autocomplete engine <i>before</i> anyone taught it manners.'),

        section('The compute: FLOPs, GPUs and a hundred million dollars',
          p('How much arithmetic does pretraining take? There is a rule of thumb that is accurate to within a factor of two for every transformer ever trained: <b>training FLOPs ≈ 6 × N × D</b>, where N is the number of parameters and D the number of tokens. Every parameter is used about twice per token in the forward pass (a multiply and an add) and about four times in the backward pass. Plug in GPT-3: 6 × 175 billion × 300 billion ≈ 3 × 10<sup>23</sup> floating-point operations. Llama 3 405B on 15 trillion tokens: about 3.8 × 10<sup>25</sup>. The largest 2025 runs are estimated at around 10<sup>26</sup>.'),
          p('A single H100 GPU does roughly 10<sup>15</sup> useful FLOPs per second on this kind of arithmetic, and in practice a training run only keeps it about 40% busy (the <em>model FLOPs utilisation</em>, MFU; the rest is lost to waiting for memory and for other GPUs). Work it through:'),
          ctx.code('N   = 4.05e11        # Llama 3 405B: parameters\nD   = 15.6e12        # training tokens\nC   = 6 * N * D      # = 3.8e25 FLOPs\n\ngpu = 1.0e15         # one H100, dense bfloat16, FLOP/s\nmfu = 0.40           # the fraction of peak a real run actually sustains\n\ngpu_seconds = C / (gpu * mfu)     # = 9.5e10\ngpu_hours   = gpu_seconds / 3600  # ≈ 26,000,000 H100-hours\ndays_on_16k = gpu_hours / 16000 / 24   # ≈ 69 days'),
          p('Meta reported roughly 16,000 H100s running for about two months, so the back-of-the-envelope lands within a few tens of per cent of the real thing. At two to three dollars per GPU-hour, that is tens of millions of dollars of compute for one run, before counting the failed experiments, the staff, or the data centre itself. Frontier runs at 10<sup>26</sup> are a few times that.'),
          p('The hardware has been on its own exponential. The A100 (2020) does about 3 × 10<sup>14</sup> FLOP/s in bfloat16; the H100 (2022) about 10<sup>15</sup>; the B200 (2024) roughly double that again, and each generation also has more memory and faster links between chips. Clusters went from a few thousand GPUs (GPT-3) to 16,000 (Llama 3) to 100,000 and beyond in 2024–2025. At that scale the interesting engineering is not the maths, it is getting a hundred thousand chips to behave like one.'),
          sub('Making 100,000 GPUs act as one',
            p('<b>Data parallelism.</b> Every GPU holds a full copy of the model and trains on a different slice of the batch. After each step, they average their gradients (an <i>all-reduce</i> over the network) so all copies stay identical. Simple, and the workhorse, but the model has to fit on one GPU, and at 405B parameters it does not.'),
            p('<b>Tensor parallelism.</b> Split each individual matrix multiplication across several GPUs: each holds a slice of every weight matrix and computes a slice of every activation, exchanging partial results constantly. This needs very fast links, so it is used within one server (eight GPUs sharing NVLink).'),
            p('<b>Pipeline parallelism.</b> Put layers 1–10 on one GPU, 11–20 on the next, and so on, and pass activations down the line like an assembly line. To stop GPUs idling while they wait for the previous stage, the batch is split into micro-batches that flow through the pipeline in sequence. Real runs combine all three: tensor parallel within a node, pipeline parallel across a few nodes, data parallel across the remaining thousands.'),
            p('Two more essentials. <b>Mixed precision</b>: weights are kept in 32-bit but the heavy arithmetic runs in 16-bit (bfloat16) or, increasingly, 8-bit floating point, which is faster and uses half the memory, with occasional loss-scaling tricks to keep small gradients from vanishing. <b>Checkpointing</b>: the full model state is saved to disk every hour or so, because with a hundred thousand GPUs <i>something</i> fails every few hours (Meta reported 466 job interruptions in 54 days of Llama 3 training, mostly GPU faults), and you do not want to lose a day of a fifty-day run.'),
          ),
        ),

        section('Scaling laws: the most expensive graph in the world',
          p('Given a compute budget, should you train a bigger model on less data or a smaller model on more? In January 2020, Kaplan and colleagues at OpenAI published <em>scaling laws</em>: loss falls as a smooth power law in parameters, data and compute, predictably, over many orders of magnitude. Their fit said parameters mattered more, so the field built ever-larger models: GPT-3 (175B) was trained on just 300 billion tokens.'),
          p('In March 2022, Hoffmann and colleagues at DeepMind redid the experiment more carefully (the <em>Chinchilla</em> paper) and found the earlier models had been badly under-trained. For a fixed budget, parameters and tokens should grow <b>together</b>, roughly 20 tokens per parameter. Their 70B model trained on 1.4 trillion tokens beat their own 280B Gopher and GPT-3, at a quarter of the size. Everyone recalibrated overnight.'),
          p('The shape of the law is worth staring at, because the interactive below is built from it. The loss of a model with N parameters trained on D tokens is fitted as <b>L(N, D) = E + A/N<sup>α</sup> + B/D<sup>β</sup></b>. Three terms, three stories. <b>E</b> is the entropy of the text itself — the part of the uncertainty that lives in the world (which day the meeting is on), which no model of any size can remove. <b>A/N<sup>α</sup></b> is what you lose for being too small to represent the pattern. <b>B/D<sup>β</sup></b> is what you lose for not having read enough. Both shrink as power laws, which on a log-log plot are straight lines — that is why the field can extrapolate at all.'),
          p('There is a twist. Chinchilla optimises <i>training</i> cost. But a model that is deployed to millions of users spends far more compute on <i>inference</i> than it ever did in training, and inference cost scales with parameters, not tokens. So it pays to go well past the Chinchilla point: train a <b>smaller</b> model for <b>longer</b> than is "optimal", eating a higher training bill to get a cheaper model to serve. Llama 3 trained its 8B model on 15 trillion tokens, nearly 2,000 tokens per parameter, a hundred times the Chinchilla ratio, and the loss was still going down.'),
          callout('history', 'Emergence: real, or a trick of the ruler?',
            'In 2022, a widely-read paper catalogued <i>emergent abilities</i>: skills such as multi-step arithmetic that appear to be absent in small models and then switch on abruptly at some scale, like a phase change. In 2023 another paper argued much of this is an artefact of the metric: if you score arithmetic as "all digits exactly right", a model that gets steadily better at each digit will look flat and then suddenly jump. Measured with smooth metrics, most "emergent" curves become gentle slopes. The honest summary in 2025: the underlying loss improves smoothly and predictably; what <i>we</i> care about (does it pass the exam?) can still change abruptly, because pass/fail is a threshold. Both camps are right about different things.'),
        ),

        callout('tryit', 'Try it: spend a hundred million dollars',
          '<b>1.</b> Drag the blue dot around the left panel (or use the two sliders). It is the (N, D) plane: parameters across, training tokens up, colour = the loss you would reach. <b>2.</b> The white line is the compute-optimal frontier and the dashed diagonals are lines of equal compute — every point on one diagonal costs the same. Slide along a diagonal and watch the loss get worse the further you stray from the white line. <b>3.</b> The right panel plots your run against the best loss any run of that budget could reach; the red bar is the loss you are throwing away. Press <b>Snap to compute-optimal</b> to land on the line. <b>4.</b> Press <b>Jump to Llama 3 8B</b>: it sits far above the white line, on purpose — the wasted training loss buys a model that is 50× cheaper to serve. <b>5.</b> Push the MFU slider from 60% down to 20% and watch the bill triple without the loss changing by a hair. That is the whole job of an infrastructure team.'),
        scalingExplorer(),

        section('What a training run looks like from the inside',
          p('Loss curves are boring to look at and thrilling to interpret. At the start, the model learns token frequencies (the word "the" is common) and the loss drops fast. Then it learns short-range structure: which tokens follow which, spelling, punctuation. Then grammar, then topic, then facts, then reasoning patterns, each stage costing ten times more tokens than the last for a smaller drop in loss. There are no visible boundaries; the samples just get better, on a log scale.'),
          p('The engineers watching that curve are mostly watching for it to <i>break</i>. A <em>loss spike</em> — the curve jumping upward for a few hundred steps — is the classic failure, usually a bad batch of data or a numerical overflow in a deep layer. The standard response is unglamorous and universal: roll back to the last checkpoint, skip the offending data, lower the learning rate, and carry on. Everything else is patience.'),
        ),

        callout('tryit', 'Watch a model learn to write',
          'Press ▶ Play, or just <b>drag along the chart</b> to scrub through the run by hand. The loss curve follows the Chinchilla formula for a 1-billion-parameter model; the samples at each checkpoint are hand-written to match what real models produce at those stages. <b>Notice three things.</b> The x-axis is logarithmic — each gridline is 10× more data, so the second half of the chart costs a thousand times more than the first. The jump from "grammatical nonsense" to "fluent but wrong" happens over one decade of data and buys almost no loss. And the grey dashed line is the floor: a 1B model cannot get below 2.04 no matter how much it reads. To go lower you have to make the model bigger, which is the whole argument of the previous interactive.'),
        trainingRun(),

        section('What you get at the end: a base model',
          p('After pretraining you have a <em>base model</em>, and it is important to be clear about what that is. It is not a chatbot. It is an <b>autocomplete engine for the internet</b>: give it text, and it produces the most plausible continuation. Give it a question, and it does not answer; it produces what typically follows a question on the web, which is often <i>another question</i>, or a list of related questions, or a forum reply beginning "same problem here, any updates?". Give it the start of a news article and it will write a convincing news article, including invented quotes and a plausible-looking URL at the bottom.'),
          p('You can coax a base model into being useful by making the desired output the most plausible continuation: write "Q: What is the capital of Peru? A:" and it will usually complete "Lima". Write three worked examples and it will do a fourth. This <em>few-shot prompting</em> was the headline of the GPT-3 paper. But it is fragile, it does not know when to stop, it will happily continue with "Q: What is the capital of Chile?" and answer that too, and it has no notion of being helpful, honest or safe. It simply is not trying to do anything except predict. Turning that into an assistant is a separate stage, <em>post-training</em>, and it is the whole of chapter 11.'),
          callout('key', 'Key idea',
            'Pretraining produces a model of <b>what text is like</b>. Everything an assistant model "knows" was learned here. Everything about how it <b>behaves</b> (answering rather than continuing, refusing, formatting, being honest about uncertainty) is added afterwards, on top of this.'),
        ),

        section('Two engineering trends: sparse experts and long contexts',
          p('<b>Mixture of experts (MoE).</b> In a standard transformer every token passes through every parameter. In an MoE, each feed-forward block is replaced by, say, 8 or 64 parallel "experts", and a tiny router network sends each token to only one or two of them. The model can have an enormous number of parameters (more stored knowledge) while each token only touches a fraction (less compute per token). GPT-4 was widely reported to use this design; Mixtral 8×7B (December 2023) made it mainstream in open models; DeepSeek-V3 (December 2024) has 671 billion parameters of which only 37 billion are active for any given token, which is why it could be trained for a reported few million dollars of GPU time. The cost is memory: all the experts must be loaded even though most sit idle for each token. Note what this does to the rule of thumb — for an MoE, the 6ND estimate uses the <i>active</i> parameter count, not the total.'),
          p('<b>Context length.</b> GPT-3 could see 2,048 tokens, about three pages. GPT-4 launched with 8k and 32k variants in 2023; 128k (a short novel) became standard by 2024; Gemini 1.5 offered a million tokens (several novels, or an hour of video). This took better positional encodings (RoPE and its extensions), attention variants that avoid the quadratic memory blow-up, and simply training on long documents. Chapter 12 covers why long contexts are expensive to <i>serve</i>, and why a model does not necessarily use all of what it can see.'),
        ),

        callout('history', 'The GPT line, in four steps',
          '<b>GPT-1 (June 2018)</b>: 117M parameters, trained on 7,000 unpublished books; showed pretraining-then-fine-tuning beats training from scratch. <b>GPT-2 (February 2019)</b>: 1.5B parameters, 40GB of web text; wrote paragraphs coherent enough that OpenAI initially withheld the full model. <b>GPT-3 (May 2020)</b>: 175B parameters, 300B tokens, ~3 × 10<sup>23</sup> FLOPs; showed few-shot prompting and made "scale" the strategy. <b>Chinchilla (March 2022)</b>: not a GPT, but it rewrote the recipe everyone used afterwards, including <b>Llama 3 (July 2024)</b>, whose 405B model on 15T tokens set the template for open frontier-class pretraining. Between GPT-3 and the 2025 frontier, training compute grew by roughly a factor of 300.'),

        section('Why this matters for modern AI',
          p('Three consequences follow from everything above. First, <b>knowledge has a cut-off</b>: whatever was not in the pretraining data is not in the model, which is why models need retrieval and tools (chapter 12) for anything recent. Second, <b>the loss is a proxy</b>: a model trained to imitate the internet imitates the internet\'s errors, biases, and confident nonsense too; that is why post-training exists and why hallucination is hard to remove. Third, <b>the economics are brutal and predictable</b>: because scaling laws work, labs can forecast what a 10× bigger run will achieve before spending the money, and that predictability is the reason the money keeps being spent.'),
          p('It also explains the shape of the industry. Pretraining is a capital expense that a handful of organisations can afford, and it happens once; everything after it — post-training, tools, agents, the product you actually use — is comparatively cheap and happens continuously. When you hear that a lab "released a new model", it is usually the second half that changed. One picture to keep: <b>pretraining fills the reservoir, and everything in Part III after this chapter is plumbing.</b>'),
        ),

        ctx.quiz([
          { q: 'A model assigns probability 0.05 to the token that actually came next. Its cross-entropy loss on that token is about…',
            options: ['0.05', '0.5', '3.0', '20'],
            answer: 2, explain: '−ln(0.05) ≈ 3.0. The loss is −log of the probability given to the truth; the matching perplexity would be e³ ≈ 20, i.e. as unsure as a 20-way guess.' },
          { q: 'Roughly how many FLOPs does it take to train a 10-billion-parameter model on 1 trillion tokens?',
            options: ['10^13', '6 × 10^22', '10^25', '6 × 10^10'],
            answer: 1, explain: '6 × N × D = 6 × 10^10 × 10^12 = 6 × 10^22. The factor of 6 is two operations per parameter per token forward, four backward.' },
          { q: 'What was the main finding of the Chinchilla paper (2022)?',
            options: ['Bigger models are always better', 'Earlier large models were under-trained; for a fixed compute budget, parameters and tokens should scale together (~20 tokens per parameter)', 'Transformers should be replaced by RNNs', 'Data quality does not matter'],
            answer: 1, explain: 'Chinchilla (70B, 1.4T tokens) beat Gopher (280B) using the same compute, by rebalancing the budget away from parameters and toward data.' },
          { q: 'Why did Meta train Llama 3 8B on 15 trillion tokens, far beyond the Chinchilla-optimal amount?',
            options: ['They made a mistake', 'Inference cost depends on model size, so a small model trained longer is cheaper to serve even if it costs more to train', 'Small models cannot be trained on less data', 'The scaling laws were wrong'],
            answer: 1, explain: 'Chinchilla optimises training cost only. When a model will be run billions of times, over-training a smaller model pays for itself in serving costs.' },
          { q: 'You give a base (pretrained-only) model the prompt "What is the capital of Peru?". What is it most likely to do?',
            options: ['Answer "Lima" and stop', 'Produce a plausible continuation of such text on the web, which might be more questions, a forum reply, or an answer that keeps going', 'Refuse because it lacks a system prompt', 'Search the internet'],
            answer: 1, explain: 'A base model is an autocomplete engine. It predicts what typically follows; it has not been trained to be an assistant, and it has no idea when to stop. That is post-training (chapter 11).' },
        ]),

        section('Go deeper', ul([
          '<a href="https://www.youtube.com/watch?v=7xTGNNLPyMI" target="_blank" rel="noopener">Andrej Karpathy, "Deep Dive into LLMs like ChatGPT" (2025, 3.5 h)</a> — the best single walkthrough of pretraining, tokenization, and what a base model is. The first hour covers this chapter.',
          '<a href="https://www.youtube.com/watch?v=zjkBMFhNj_g" target="_blank" rel="noopener">Andrej Karpathy, "Intro to Large Language Models" (2023, 1 h)</a> — the shorter version of the same story.',
          '<a href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noopener">Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)</a> and <a href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noopener">Kaplan et al., "Scaling Laws for Neural Language Models" (2020)</a> — the two papers the scaling-law explorer is built from.',
          '<a href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noopener">"The Llama 3 Herd of Models" (Meta, 2024)</a> — the most detailed public account of a frontier-scale pretraining run: data pipeline, parallelism, and every failure.',
          '<a href="https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1" target="_blank" rel="noopener">The FineWeb report (Hugging Face, 2024)</a> — how a 15-trillion-token web dataset is actually filtered, with an ablation for every step.',
          '<a href="https://lilianweng.github.io/posts/2021-09-25-train-large/" target="_blank" rel="noopener">Lilian Weng, "How to Train Really Large Models on Many GPUs"</a> — data, tensor and pipeline parallelism in detail.',
        ])),
      );
    },
  });
})();
