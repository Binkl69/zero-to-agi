/* Zero → AGI · Chapter 10 · Pretraining: how a base model is made
   DESIGN RULE: the reader does the model's only job — guess the next token — in the first ten
   seconds, and notices what they had to KNOW to do it well.
   Interactives, in order: next-token game (be the model); loss/perplexity lab on one worked
   sentence; the data funnel; the 6ND scaling and cost explorer; a training run with loss spikes;
   raw base model vs post-trained assistant; mixture-of-experts stored-vs-active parameters. */
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
          /* pages are clipped to the lane: a rejected one falls out of the bottom of it and is
             gone, instead of landing on the stage labels printed underneath */
          g.save();
          g.beginPath(); g.rect(0, 40, 720, 160); g.clip();
          for (const q of parts) {
            g.fillStyle = q.dead
              ? 'rgba(251,113,133,' + Math.max(0, q.life).toFixed(3) + ')'
              : (q.warm ? 'rgba(124,156,255,0.85)' : 'rgba(56,217,169,0.85)');
            g.fillRect(q.x, q.y, q.w, q.w * 1.3);
          }
          g.restore();
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
        const P1 = { x: 52, y: 26, w: 296, h: 292, x0: 7, x1: 12, y0: 8, y1: 15 };
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
            /* label the diagonal where it is already inside the panel — a diagonal that enters
               through the top edge would otherwise put its label in the panel's title row */
            const inset = 16 / (P1.h / (P1.y1 - P1.y0));        // 16px expressed in log-token units
            const ll = Math.max(lo, k - (P1.y1 - inset));
            if (hi - ll > 0.5) {
              g.fillStyle = 'rgba(230,235,245,0.6)';
              const lx0 = X1(ll), ly0 = Y1(k - ll) - 3;
              const w = g.measureText('1e' + lc).width;
              if (lx0 + 3 + w < P1.x + P1.w - 2) g.fillText('1e' + lc, lx0 + 3, ly0);
              else { g.textAlign = 'right'; g.fillText('1e' + lc, lx0 - 3, ly0); g.textAlign = 'left'; }
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

          /* ---- reference runs ----
             The frontier and the iso-compute diagonals sweep straight through this corner of the
             panel, so every name gets an opaque plate under it: the plate sits on the colour
             field (not on any data point) and keeps the name readable wherever the lines fall. */
          g.font = FONT; g.textBaseline = 'middle';
          for (const r of REFS) {
            const x = X1(Math.log10(r.N)), y = Y1(Math.log10(r.D));
            g.fillStyle = C.warn; g.beginPath(); g.arc(x, y, 3, 0, Math.PI * 2); g.fill();
            const w = g.measureText(r.nm).width;
            const lx0 = ctx.clamp(x > P1.x + P1.w - 74 ? x - 6 - w : x + 6, P1.x + 3, P1.x + P1.w - w - 3);
            const ly0 = ctx.clamp(y + r.ly, P1.y + 9, P1.y + P1.h - 9);
            g.fillStyle = C.bg; g.fillRect(lx0 - 3, ly0 - 7, w + 6, 14);
            g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText(r.nm, lx0, ly0);
          }

          /* ---- the reader's point ---- */
          const N = Math.pow(10, st.logN), D = Math.pow(10, st.logD), Cc = 6 * N * D, loss = L(N, D);
          /* keep the whole marker inside the panel at the extremes of the sliders, so it never
             sits half on top of the axis numbers */
          const ux = ctx.clamp(X1(st.logN), P1.x + 7, P1.x + P1.w - 7);
          const uy = ctx.clamp(Y1(st.logD), P1.y + 7, P1.y + P1.h - 7);
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
          const px = ctx.clamp(X2(Math.log10(Cc)), P2.x + 7, P2.x + P2.w - 7);
          const py = ctx.clamp(Y2(loss), P2.y + 7, P2.y + P2.h - 7);
          if (loss - bestL > 0.02) {
            g.strokeStyle = C.danger; g.lineWidth = 1;
            g.beginPath(); g.moveTo(px, py); g.lineTo(px, Y2(bestL)); g.stroke();
            const wasted = '+' + (loss - bestL).toFixed(2) + ' loss wasted';
            g.fillStyle = C.danger; g.textBaseline = 'middle';
            /* keep it to the right of the marker, where the green curve runs below it; only flip
               when the text would otherwise leave the canvas */
            const flip = px + 8 + g.measureText(wasted).width > cv.W - 6;
            g.textAlign = flip ? 'right' : 'left';
            /* when the optimum sits almost on the irreducible floor the midpoint of
               this segment lands on the 'irreducible loss E' label; keep clear of it */
            g.fillText(wasted, px + (flip ? -8 : 8), Math.min((py + Y2(bestL)) / 2, Y2(fit.E) - 22));
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
            N: fmtBig(N), D: fmtBig(D),
            'tokens/param': D / N < 0.01 ? (D / N).toExponential(1)
              : D / N < 10 ? (D / N).toFixed(2) : D / N < 1e4 ? (D / N).toFixed(0) : fmtBig(D / N),
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
          st.logN = Math.log10(8e9); st.logD = Math.log10(1.56e13);   /* exact, so the readout says 8B and 15.6T */
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
            /* one real spike, so the thing the callout sends the reader to look
               for exists: a sharp jump that recovers to trend on its own */
            const spike = 2.6 * Math.exp(-Math.pow((l - 10.64) / 0.045, 2));
            curve.push([l, ctx.clamp(base + jit * Math.min(1, base / 3) + spike, FLOOR - 0.02, 11.4)]);
          }
        })();
        const idxAt = (l) => ctx.clamp(Math.round((l - T0) / DT), 0, curve.length - 1);

        const [cv, g] = ctx.canvas(720, 280);
        const sample = h('div', { style: { fontFamily: 'var(--mono)', fontSize: '.88rem', lineHeight: 1.6, padding: '12px 14px', borderTop: '1px solid var(--line)', minHeight: '104px', whiteSpace: 'pre-wrap' } });
        const readout = ctx.readout();
        const P = { x: 52, y: 22, w: 646, h: 196, x0: T0, x1: T1, y0: 1.5, y1: 11.5 };
        /* inset the data range by a dot radius so the marker at either end of the run is not
           sliced in half by the clip that keeps the curve inside the plot box */
        const PAD = 7;
        const X = (l) => P.x + PAD + (l - P.x0) / (P.x1 - P.x0) * (P.w - 2 * PAD);
        const Y = (v) => P.y + P.h - PAD - (ctx.clamp(v, P.y0, P.y1) - P.y0) / (P.y1 - P.y0) * (P.h - 2 * PAD);

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

          /* checkpoints — the dashed line starts below the label row so it never runs through the
             label, and the label is kept inside the plot so it cannot land on the y-axis numbers */
          g.font = FONT; g.textAlign = 'center'; g.textBaseline = 'top';
          for (const s of STAGES) {
            const lab = 'ckpt ' + fmtBig(Math.pow(10, s.at));
            const half = g.measureText(lab).width / 2 + 3;
            g.strokeStyle = 'rgba(251,191,36,0.4)'; g.setLineDash([3, 3]);
            g.beginPath(); g.moveTo(X(s.at), P.y + 18); g.lineTo(X(s.at), P.y + P.h); g.stroke(); g.setLineDash([]);
            g.fillStyle = C.warn;
            g.fillText(lab, ctx.clamp(X(s.at), P.x + half, P.x + P.w - half), P.y + 3);
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

          /* the scrub hint lives under the axis, not inside the plot, where the floor line and the
             end of the loss curve would otherwise be drawn straight through it */
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'center'; g.textBaseline = 'top';
          g.fillText('drag anywhere on the chart to scrub through the run', P.x + P.w / 2, P.y + P.h + 38);
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
          logT = ctx.clamp(T0 + (q.x - P.x - PAD) / (P.w - 2 * PAD) * (T1 - T0), T0, T1);
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
      const MONOF = '12px "JetBrains Mono", ui-monospace, monospace';
      const UI = '13px Inter, system-ui, sans-serif';

      /* ================================================================== */
      /*  INTERACTIVE — what one number of loss actually means               */
      /* ================================================================== */
      function lossLab() {
        const [cv, g] = ctx.canvas(720, 350);
        /* the worked example from the chapter, with the model's confidence adjustable */
        const TOKENS = [
          { tok: 'cat', after: 'The', base: 0.020 },
          { tok: 'sat', after: 'The cat', base: 0.100 },
          { tok: 'on', after: 'The cat sat', base: 0.600 },
          { tok: 'the', after: 'The cat sat on', base: 0.750 },
          { tok: 'mat', after: 'The cat sat on the', base: 0.150 },
        ];
        let skill = 1.0;   // 0 = random guessing, 1 = as fitted, 2 = very strong
        const VOCAB = 100000;
        const pOf = (t) => ctx.clamp(Math.pow(t.base, 1 / Math.max(0.025, skill)), 1 / VOCAB, 0.999);

        const sSl = ctx.slider({ label: 'how good the model is', min: 0.025, max: 2.5, step: 0.025, value: 1, digits: 3, onChange: (v) => { skill = v; } });
        const rndBtn = ctx.button('an untrained model', () => { skill = 0.025; sSl.value = 0.025; });
        const midBtn = ctx.button('the worked example', () => { skill = 1; sSl.value = 1; }, 'primary');
        const goodBtn = ctx.button('a strong model', () => { skill = 2.2; sSl.value = 2.2; });
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          const losses = TOKENS.map(t => -Math.log(pOf(t)));
          const total = losses.reduce((a, b) => a + b, 0);
          const avg = total / TOKENS.length;
          const ppl = Math.exp(avg);

          g.font = 'bold ' + UI; g.fillStyle = C.text;
          g.fillText('five predictions across one sentence', 30, 26);
          g.font = MONOF; g.fillStyle = C.muted;
          g.fillText('context', 30, 48);
          g.fillText('true next', 235, 48);
          g.fillText('model said', 330, 48);
          g.fillText('cost  −log(p)', 440, 48);
          let y = 70;
          TOKENS.forEach((t, i) => {
            const pp = pOf(t), ls = losses[i];
            g.font = MONOF; g.fillStyle = C.muted;
            g.fillText('"' + t.after + '"', 30, y + 12);
            g.fillStyle = C.text; g.fillText(t.tok, 235, y + 12);
            g.fillStyle = pp > 0.4 ? C.green : pp > 0.08 ? C.warn : C.danger;
            g.fillText((pp * 100).toFixed(1) + '%', 330, y + 12);
            /* cost bar */
            g.fillStyle = C.line; g.fillRect(440, y + 2, 150, 13);
            g.fillStyle = ls < 1 ? C.green : ls < 3 ? C.warn : C.danger;
            g.fillRect(440, y + 2, ctx.clamp(ls / 12, 0, 1) * 150, 13);
            g.fillStyle = C.text; g.fillText(ls.toFixed(2), 600, y + 12);
            y += 26;
          });
          g.strokeStyle = C.line; g.lineWidth = 1;
          g.beginPath(); g.moveTo(30, y + 4); g.lineTo(660, y + 4); g.stroke();
          g.font = MONOF; g.fillStyle = C.muted;
          g.fillText('total ' + total.toFixed(2) + ' ÷ 5 predictions =', 30, y + 26);
          g.font = 'bold 18px Inter, system-ui, sans-serif'; g.fillStyle = C.accent;
          g.fillText(avg.toFixed(2) + ' nats per token', 260, y + 28);

          /* the loss → perplexity scale */
          const P = { x: 30, y: 262, w: 630, h: 16 };
          g.font = 'bold ' + UI; g.fillStyle = C.text;
          g.fillText('what that number means', P.x, P.y - 10);
          g.fillStyle = C.line; g.fillRect(P.x, P.y, P.w, P.h);
          const lx = (l) => P.x + ctx.clamp(l / 11.5, 0, 1) * P.w;
          g.fillStyle = 'rgba(56,217,169,0.35)'; g.fillRect(P.x, P.y, lx(2.5) - P.x, P.h);
          g.strokeStyle = C.accent; g.lineWidth = 3;
          g.beginPath(); g.moveTo(lx(avg), P.y - 5); g.lineTo(lx(avg), P.y + P.h + 5); g.stroke();
          g.font = MONOF; g.fillStyle = C.muted;
          g.fillText('0', P.x, P.y + 30);
          g.fillText('good models live here', P.x + 40, P.y + 30);
          /* right-align the end-of-scale label against the end of the scale itself, so it can
             never run past the canvas edge whatever the font measures */
          g.textAlign = 'right';
          g.fillText('11.5 = pure guessing over 100,000 tokens', lx(11.5), P.y + 30);
          g.textAlign = 'left';

          g.font = 'bold 16px Inter, system-ui, sans-serif'; g.fillStyle = C.text;
          const pplLine = 'perplexity = e^loss = ' + ppl.toFixed(1);
          g.fillText(pplLine, P.x, 322);
          /* start the gloss after the headline actually ends — a five-figure perplexity is much
             wider than the worked example's 5.9 and used to be written over */
          const nx = P.x + g.measureText(pplLine).width + 16;
          g.font = UI; g.fillStyle = C.muted;
          wrapText(g, '— as unsure as if every token were a ' + Math.round(ppl) + '-way multiple-choice question.', nx, 322, P.x + P.w - nx, 16);
          ro.set({ 'loss (nats/token)': avg.toFixed(2), perplexity: ppl.toFixed(1), 'like a multiple choice of': Math.round(ppl) });
        });

        return ctx.figure(cv,
          'The entire training signal, on one sentence. Notice where the cost lives: "on" after "The cat sat" is nearly free, while "cat" after "The" is punished hard — and rightly so, because the model had no way to know and should not have been confident. Drag the slider to an untrained model and the loss climbs toward 11.5, which is exactly log(100,000): the cost of pure guessing over the vocabulary. Every training curve you have ever seen in a paper is this number going down.',
          [sSl, rndBtn, midBtn, goodBtn], ro);
      }

      /* ================================================================== */
      /*  INTERACTIVE — what a base model actually does with your question   */
      /* ================================================================== */
      function baseVsChat() {
        const [cv, g] = ctx.canvas(720, 360);
        const CASES = [
          {
            prompt: 'What is the capital of Peru?',
            base: ['What is the capital of Bolivia?', 'What is the largest city in Peru?', 'What language do they speak in Peru?', '', 'Answers to these and more in our South America quiz pack — click here to download the PDF.'],
            chat: ['The capital of Peru is Lima.', '', 'It sits on the Pacific coast and is home to roughly a third of the country\'s population.'],
            why: 'On the web, a question is most often followed by more questions — a quiz, an FAQ, a forum thread. The base model is not refusing to answer. It is predicting correctly.',
          },
          {
            prompt: 'How do I fix a leaking tap?',
            base: ['Posted by mike_84 · 3 years ago', '', 'same problem here, any updates?', '', 'Re: How do I fix a leaking tap?', 'bump'],
            chat: ['Start by turning off the water supply under the sink.', '', 'Most drips come from a worn washer or O-ring, so the next step is to open the tap up and check those.'],
            why: 'Forum pages are full of this text. A model trained to continue the internet continues the forum thread, complete with the dead-end replies.',
          },
          {
            prompt: 'Q: What is the capital of Peru?\nA:',
            base: ['Lima', '', 'Q: What is the capital of Chile?', 'A: Santiago', '', 'Q: What is the capital of Ecuador?', 'A: Quito'],
            chat: ['The capital of Peru is Lima.'],
            why: 'This is few-shot prompting, and it works — you made the answer the most plausible continuation. But look what happens next: it has no idea when to stop, and cheerfully invents a whole quiz.',
          },
        ];
        let idx = 0, showChat = false, reveal = 0;

        const nextBtn = ctx.button('Next prompt →', () => {
          idx = (idx + 1) % CASES.length; reveal = 0;
          showChat = false; modeBtn.textContent = 'show: base model';
        });
        const modeBtn = ctx.button('show: base model', () => {
          showChat = !showChat; reveal = 0;
          modeBtn.textContent = 'show: ' + (showChat ? 'after post-training' : 'base model');
        }, 'primary');
        const ro = ctx.readout();

        ctx.loop((dt) => {
          reveal = Math.min(1, reveal + dt * 0.55);
          g.clearRect(0, 0, cv.W, cv.H);
          const c = CASES[idx];
          const lines = showChat ? c.chat : c.base;

          g.font = 'bold ' + UI; g.fillStyle = C.text;
          g.fillText('you type', 30, 28);
          g.fillStyle = 'rgba(124,156,255,0.12)';
          const ph = c.prompt.split('\n').length * 20 + 14;
          g.fillRect(30, 40, 640, ph);
          g.strokeStyle = C.accent; g.lineWidth = 1.5; g.strokeRect(30, 40, 640, ph);
          g.font = MONOF; g.fillStyle = C.accent;
          c.prompt.split('\n').forEach((ln, i) => g.fillText(ln, 42, 62 + i * 20));

          const Y0 = 40 + ph + 34;
          g.font = 'bold ' + UI; g.fillStyle = showChat ? C.green : C.warn;
          g.fillText(showChat ? 'what an assistant says (chapter 11)' : 'what the base model actually continues with', 30, Y0 - 12);
          g.fillStyle = showChat ? 'rgba(56,217,169,0.08)' : 'rgba(251,191,36,0.08)';
          g.fillRect(30, Y0, 640, 150);
          g.strokeStyle = showChat ? C.green : C.warn; g.lineWidth = 1.5;
          g.strokeRect(30, Y0, 640, 150);
          const shown = Math.ceil(lines.length * reveal);
          g.font = MONOF; g.fillStyle = C.text;
          lines.slice(0, shown).forEach((ln, i) => g.fillText(ln, 42, Y0 + 24 + i * 19));

          g.font = UI; g.fillStyle = C.muted;
          wrapText(g, showChat
            ? 'Post-training is what turns the left-hand behaviour into this. Nothing was added to the model\'s knowledge — only its sense of what it is for.'
            : c.why, 30, Y0 + 176, 640, 17);
          ro.set({ prompt: idx + 1 + ' of ' + CASES.length, showing: showChat ? 'after post-training' : 'raw base model' });
        });

        return ctx.figure(cv,
          'A base model is not a chatbot that needs polishing. It is an autocomplete engine for the internet, and it is doing its job perfectly — a question on the web really is most often followed by another question. The illustrated continuations here are written to be representative rather than sampled live, but the behaviour is real and is exactly what everyone who has loaded a raw base model has met. Everything that makes a model feel like an assistant is added in chapter 11.',
          [nextBtn, modeBtn], ro);
      }

      /* ================================================================== */
      /*  INTERACTIVE — mixture of experts: big brain, small bill per token  */
      /* ================================================================== */
      function moeLab() {
        const [cv, g] = ctx.canvas(720, 330);
        let experts = 8, topK = 2, expertSize = 7, dense = false, t = 0;
        const TOKENS = ['the', 'patient', 'presented', 'with', 'acute', 'chest', 'pain', 'and', 'ST', 'elevation'];
        const eSl = ctx.slider({ label: 'experts per block', min: 2, max: 64, step: 1, value: 8, onChange: (v) => { experts = v; if (topK > v) { topK = v; kSl.value = v; } } });
        const kSl = ctx.slider({ label: 'experts used per token', min: 1, max: 8, step: 1, value: 2, onChange: (v) => { topK = Math.min(v, experts); } });
        const zSl = ctx.slider({ label: 'billions of params per expert', min: 1, max: 20, step: 1, value: 7, onChange: (v) => { expertSize = v; } });
        const denseBtn = ctx.button('compare with a dense model', () => {
          dense = !dense;
          denseBtn.textContent = dense ? 'back to mixture of experts' : 'compare with a dense model';
        });
        const dsBtn = ctx.button('DeepSeek-V3-ish', () => { experts = 64; eSl.value = 64; topK = 4; kSl.value = 4; expertSize = 10; zSl.value = 10; }, 'primary');
        const ro = ctx.readout();

        ctx.loop((dt) => {
          t += dt;
          g.clearRect(0, 0, cv.W, cv.H);
          const total = experts * expertSize;
          const active = dense ? total : topK * expertSize;
          const tokIdx = Math.floor(t * 1.2) % TOKENS.length;
          /* a deterministic, stable "routing" so the picture does not flicker */
          const chosen = [];
          for (let i = 0; i < (dense ? experts : topK); i++) chosen.push((tokIdx * 7 + i * 3) % experts);

          g.font = 'bold ' + UI; g.fillStyle = C.text;
          g.fillText('one token arrives and the router picks where to send it', 30, 26);
          g.font = MONOF; g.fillStyle = C.accent;
          g.fillText('"' + TOKENS[tokIdx] + '"', 30, 52);

          const shown = Math.min(experts, 16);
          /* reserve room on the row for the "… and N more" tag before sizing the boxes, so the
             tag always fits on the canvas however many experts are hidden */
          const moreLbl = experts > shown ? '… and ' + (experts - shown) + ' more' : '';
          g.font = MONOF;
          const moreW = moreLbl ? g.measureText(moreLbl).width + 12 : 0;
          const BW = Math.min(38, (660 - moreW) / shown - 4);
          for (let i = 0; i < shown; i++) {
            const on = chosen.indexOf(i) >= 0;
            const x = 30 + i * (BW + 4);
            g.fillStyle = on ? 'rgba(56,217,169,0.55)' : '#141b28';
            g.fillRect(x, 74, BW, 60);
            g.strokeStyle = on ? C.green : C.line; g.lineWidth = on ? 2 : 1;
            g.strokeRect(x, 74, BW, 60);
            if (on) {
              g.strokeStyle = 'rgba(56,217,169,0.5)'; g.lineWidth = 1.5;
              g.beginPath(); g.moveTo(70, 58); g.lineTo(x + BW / 2, 74); g.stroke();
            }
          }
          g.font = MONOF; g.fillStyle = C.muted;
          if (moreLbl) g.fillText(moreLbl, 30 + shown * (BW + 4) + 6, 110);
          g.fillText(dense ? 'a dense model: every token goes through every parameter' : 'only the lit experts do any work for this token', 30, 152);

          /* the two bars */
          const BX = 30, BWD = 430;
          /* both bars share one scale, and the scale grows with the model, so the stored bar never
             saturates and the two bars always show the true stored : active ratio */
          const BMAX = Math.max(700, total);
          const bar = (lab, v, col, y, note) => {
            g.font = UI; g.fillStyle = C.muted; g.fillText(lab, BX, y);
            g.fillStyle = C.line; g.fillRect(BX, y + 8, BWD, 18);
            g.fillStyle = col; g.fillRect(BX, y + 8, ctx.clamp(v / BMAX, 0, 1) * BWD, 18);
            g.font = 'bold 16px Inter, system-ui, sans-serif'; g.fillStyle = col;
            g.fillText(v.toFixed(0) + 'B', BX + BWD + 14, y + 23);
            g.font = MONOF; g.fillStyle = C.muted; g.fillText(note, BX, y + 42);
          };
          bar('parameters stored (what it knows)', total, C.purple, 186, 'all of these sit in GPU memory, idle or not');
          bar('parameters used per token (what you pay)', active, C.green, 248, 'this is the number that goes into the 6ND estimate');

          g.font = 'bold ' + UI;
          g.fillStyle = dense ? C.warn : C.green;
          g.fillText(dense ? 'dense: stored and used are the same number'
            : (total / Math.max(1, active)).toFixed(1) + '× more knowledge than compute per token', BX, 318);
          ro.set({ experts, 'used per token': dense ? experts : topK, stored: total + 'B', active: active + 'B' });
        });

        return ctx.figure(cv,
          'In a standard dense transformer every token passes through every parameter. A mixture of experts replaces each feed-forward block with many parallel experts and a tiny router that sends each token to just a handful of them — two of eight in Mixtral, eight of 256 in DeepSeek-V3 — so the model can store far more knowledge than it spends compute on. DeepSeek-V3 has 671 billion parameters of which about 37 billion are active for any given token, which is why it could be trained for a reported few million dollars of GPU time. The cost is memory: every expert must be loaded even though most sit idle for each token.',
          [eSl, kSl, zSl, dsBtn, denseBtn], ro);
      }
      /* ================================================================== */
      /*  The chapter: touch first, read second.                            */
      /* ================================================================== */
      root.append(
        callout('tryit', '🖐 Do this first — be the language model',
          `You are about to do the <b>only</b> job a model is given during pretraining, on real sentences.<br>
           <b>1.</b> Guess the next word. Then reveal it and see what the model thought.<br>
           <b>2.</b> Do a few. Notice that some guesses are nearly free and others are impossible.<br>
           <b>3.</b> Now the important part: notice <b>what you had to know</b> to guess well. Not spelling — facts, arithmetic, what the sentence was about.<br>
           <b>4.</b> Press <b>Next sentence →</b> and keep going until that clicks.`),
        nextTokenGame(),
        p(`That is the entire training objective. There is no second task, no labels, no human in the loop.`),
      );

      root.append(section('Why guessing words teaches everything else',
        p(`Ask a modern model to translate a Portuguese poem, debug a Rust program, explain the Thirty Years' War and write a limerick about it. It does all four. Nobody wrote a translation module, a Rust module or a history module. Nobody labelled a single example of "good limerick".`),
        p(`All of it fell out of the task you just did, applied to a slice of the internet so large that reading it aloud would take a hundred thousand years.`),
        p(`It works because <b>predicting text well requires understanding what the text is about</b>. To finish "the boiling point of water at sea level is 100", you need a fact. To predict the next line of a Python function, you need to know what the function is for. To finish "so the total cost is", you need to have added the numbers up.`),
        callout('key', '🔑 The example worth remembering',
          `To predict the name that follows <i>"and the murderer was…"</i> on the last page of a mystery novel, <b>you need to have solved the murder.</b><br>
           Grammar, facts, arithmetic, code, the structure of an argument, a rough model of how people and objects behave: every one of them lowers the loss.
           So every one of them gets learned — as a <em>side effect</em> of guessing the next word.`),
        p(`One detail makes this affordable. You might imagine the model guessing one token and starting over. It is not. Because of the causal mask from chapter 7, a single forward pass over an 8,192-token document produces 8,192 predictions at once, each allowed to look only leftwards.`),
        p(`So one pass yields 8,192 training signals instead of one. This is <em>teacher forcing</em>: during training the model is always fed the <i>real</i> previous tokens, never its own guesses — which is exactly why training parallelises and generation does not.`),
        p(`The data labels itself. That is what people mean by <em>self-supervised</em> learning, and it is why the internet works as a training set at all.`),
      ));

      root.append(section('One number: how wrong, on average',
        p(`The penalty is the <em>cross-entropy loss</em> from chapter 2: −log of the probability the model gave the correct token. Said 50% and was right? Loss 0.69. Said 10%? 2.3. Said 1%? 4.6.`),
        callout('tryit', '🖐 Try this — watch a whole training run compressed into one slider',
          `<b>1.</b> Press <b>an untrained model</b>. The loss climbs toward <b>11.5</b>, which is exactly log(100,000) — the cost of pure guessing over the vocabulary.<br>
           <b>2.</b> Press <b>the worked example</b>, then <b>a strong model</b>. Watch which rows get cheaper and which stay expensive.<br>
           <b>3.</b> Look at the row for <b>"cat" after "The"</b>. It stays costly even for a strong model, and that is correct — nothing in "The" tells you a cat is coming.`),
        lossLab(),
        p(`There is a friendlier version of the same number. <em>Perplexity</em> = e<sup>loss</sup>, read as "the model is as unsure as if it were choosing uniformly among this many options". Loss 2.3 means perplexity 10: every token behaves like a ten-way multiple-choice question.`),
        p(`A random guess over a 100,000-token vocabulary costs about 11.5. A good model on typical web text averages roughly 2. Every training curve you have ever seen in a paper is one of those two numbers going down.`),
      ));

      root.append(section('The fuel: what actually goes in',
        p(`The raw material is the open web, and the standard starting point is <em>Common Crawl</em>, a non-profit that has been saving snapshots of billions of pages since 2008. One snapshot is a few hundred terabytes of HTML, and most of it is junk.`),
        p(`Navigation menus, cookie banners, SEO spam, the same Wikipedia article mirrored a thousand times, machine-translated product listings. <b>The single biggest lever in pretraining is what you do with the junk.</b>`),
        callout('tryit', '🖐 Try this: watch the funnel',
          `Watch the counters at each gate and keep an eye on the survival rate at the bottom.<br>
           <b>Of every hundred tokens that go in, something like five to ten come out.</b> The other ninety are not a rounding error — discarding them is most of the work.`),
        dataPipeline(),
        p(`<b>Language identification</b> keeps the languages you want. <b>Deduplication</b> removes near-duplicate documents, because a model that sees the same paragraph ten thousand times memorises it instead of learning from it.`),
        p(`<b>Quality classifiers</b> — small models trained to recognise text that looks like a good textbook or a well-written article — score every page and drop the worst. <b>Toxicity and personal-data filters</b> drop the rest.`),
        p(`The survivors are then <em>mixed</em> with curated sources in deliberate proportions: books, academic papers, a great deal of source code, maths, and dialogue, because each teaches something the open web does not. Llama 3 used about 15 trillion tokens after all of this.`),
        p(`One more step matters: the <em>tokenizer</em> is trained on a sample of this data and frozen before pretraining begins. Its vocabulary — typically 100,000 to 250,000 pieces — determines what "one token" means for the rest of the model's life. This is why a model can be oddly bad at counting letters. It has never seen letters, only pieces.`),
      ));

      root.append(section('The factory: what it costs',
        p(`There is a rule of thumb accurate to within a factor of two for every transformer ever trained: <b>training FLOPs ≈ 6 × N × D</b>, where N is parameters and D is tokens. Every parameter is used about twice per token going forwards and about four times coming back.`),
        p(`GPT-3: 6 × 175 billion × 300 billion ≈ 3 × 10<sup>23</sup> operations. Llama 3 405B on 15.6 trillion tokens: about 3.8 × 10<sup>25</sup>. The largest 2025 runs are estimated near 10<sup>26</sup>.`),
        p(`A single H100 does roughly 10<sup>15</sup> useful operations per second on this arithmetic, and a real run keeps it only about 40% busy — the <em>model FLOPs utilisation</em>, with the rest lost waiting for memory and for other GPUs.`),
        callout('tryit', '🖐 Try this: spend a hundred million dollars',
          `<b>1.</b> Press <b>Jump to Llama 3 8B</b> and read the cost and the wall-clock time.<br>
           <b>2.</b> Now drag <b>parameters N</b> up by 10× and watch the bill. Then put N back and drag <b>training tokens D</b> up by 10× instead. <b>The cost is the same</b> — 6ND does not care which one you grew.<br>
           <b>3.</b> Press <b>Snap to compute-optimal</b> and see where Chinchilla says the balance should sit.<br>
           <b>4.</b> Drop <b>MFU</b> from 40% to 20% and watch the wall-clock double. That number is pure engineering, and it is worth millions.`),
        scalingExplorer(),
        p(`Meta reported roughly 16,000 H100s running for about two months for Llama 3, so the back-of-the-envelope lands within a few tens of per cent of reality. At two to three dollars per GPU-hour that is tens of millions of dollars for one run — before the failed experiments, the staff, or the data centre.`),
        callout('key', '🔑 Making a hundred thousand chips behave like one',
          `<b>Data parallelism.</b> Every GPU holds a full copy of the model and trains on a different slice of the batch, then they average their gradients so all copies stay identical. The workhorse — but the model must fit on one GPU, and at 405B parameters it does not.<br>
           <b>Tensor parallelism.</b> Split each individual matrix multiply across several GPUs, each holding a slice of every weight matrix. Needs very fast links, so it stays within one server.<br>
           <b>Pipeline parallelism.</b> Layers 1–10 on one GPU, 11–20 on the next, activations passed down like an assembly line, with the batch split into micro-batches so nobody idles.<br>
           Real runs use all three at once: tensor within a node, pipeline across a few nodes, data parallel across the remaining thousands.`),
        p(`Two more essentials. <b>Mixed precision</b>: weights kept in 32-bit while the heavy arithmetic runs in 16-bit or increasingly 8-bit, which is faster and uses half the memory.`),
        p(`And <b>checkpointing</b>: the full model state is saved every hour or so, because with that many GPUs <i>something</i> fails every few hours. Meta reported 466 job interruptions in 54 days of Llama 3 training, mostly GPU faults. You do not want to lose a day of a fifty-day run.`),
      ));

      root.append(section('How big, on how much? The law that made this an industry',
        p(`Given a compute budget, should you train a bigger model on less data, or a smaller model on more? In January 2020 Kaplan and colleagues at OpenAI published <em>scaling laws</em>: loss falls as a smooth power law in parameters, data and compute, predictably, over many orders of magnitude.`),
        p(`Their fit said parameters mattered more, so the field built ever-larger models. GPT-3 had 175 billion parameters and was trained on just 300 billion tokens.`),
        p(`In March 2022 Hoffmann and colleagues at DeepMind redid the experiment more carefully — the <em>Chinchilla</em> paper — and found the earlier models had been badly under-trained. For a fixed budget, parameters and tokens should grow <b>together</b>, roughly 20 tokens per parameter.`),
        p(`Their 70B model trained on 1.4 trillion tokens beat their own 280B Gopher <i>and</i> GPT-3, at a quarter of the size. Everyone recalibrated overnight.`),
        callout('key', '🔑 Three terms, three stories',
          `The fitted law is <b>L(N, D) = E + A/N<sup>α</sup> + B/D<sup>β</sup></b>.<br>
           <b>E</b> is the entropy of the text itself — the uncertainty that lives in the world (which day the meeting is on), which no model of any size can remove.<br>
           <b>A/N<sup>α</sup></b> is what you lose for being too small to represent the pattern.<br>
           <b>B/D<sup>β</sup></b> is what you lose for not having read enough.<br>
           Both shrink as power laws, which are straight lines on a log-log plot. <b>That is why the field can extrapolate at all</b> — and why labs can forecast what a 10× bigger run will buy before spending the money.`),
        p(`There is a twist. Chinchilla optimises <i>training</i> cost. But a model deployed to millions of users spends far more compute on <i>inference</i> than it ever did on training, and inference cost scales with parameters, not tokens.`),
        p(`So it pays to go well past the Chinchilla point: train a <b>smaller</b> model for <b>longer</b> than is "optimal", eating a higher training bill to get a cheaper model to serve. Llama 3 trained its 8B model on 15 trillion tokens — nearly 2,000 tokens per parameter, a hundred times the Chinchilla ratio — and the loss was still going down.`),
      ));

      root.append(section('Watching it learn',
        callout('tryit', '🖐 Try this: watch a model learn to write',
          `<b>1.</b> Press <b>▶ Play</b> and read the samples as the loss falls. Early on it produces letter soup, then plausible-looking words, then grammar, then something with a topic.<br>
           <b>2.</b> Watch the <b>shape</b> of the curve, not the number. Each new ability costs ten to a hundred times more tokens than the last, for a smaller drop in loss — read the gaps between the checkpoint markers.<br>
           <b>3.</b> Look for the <b>loss spike</b> — the moment the curve jumps upward. That is the thing engineers actually sit and watch for.`),
        trainingRun(),
        p(`At the start the model learns token frequencies and the loss drops fast. Then short-range structure: spelling, punctuation, which tokens follow which. Then grammar, then topic, then facts, then reasoning patterns. There are no visible boundaries; the samples just get better, on a log scale.`),
        p(`A <em>loss spike</em> — the curve jumping upward for a few hundred steps — is the classic failure, usually a bad batch of data or a numerical overflow in a deep layer. The standard response is unglamorous and universal: roll back to the last checkpoint, skip the offending data, lower the learning rate, carry on. Everything else is patience.`),
      ));

      root.append(section('What you have at the end is not an assistant',
        p(`After pretraining you have a <em>base model</em>, and it is worth being precise about what that is. It is not a chatbot. It is an <b>autocomplete engine for the internet</b>.`),
        callout('tryit', '🖐 Try this — ask a raw base model a question',
          `<b>1.</b> Read what the base model does with "What is the capital of Peru?". It does not answer. It asks more questions.<br>
           <b>2.</b> Press <b>show: base model</b> to flip to the post-trained version and see the difference. <b>Nothing was added to its knowledge</b> — only its sense of what it is for.<br>
           <b>3.</b> Press <b>Next prompt →</b> twice to reach the <code class="inline">Q: … A:</code> example. That is few-shot prompting, and it works — right up until the model invents a whole quiz because nothing told it to stop.`),
        baseVsChat(),
        p(`Give a base model a question and it produces what typically follows a question on the web: often another question, or a list of related ones, or a forum reply beginning "same problem here, any updates?". Give it the start of a news article and it writes a convincing one, including invented quotes and a plausible-looking URL at the bottom.`),
        p(`You can coax it into being useful by making the desired output the most plausible continuation — write "Q: What is the capital of Peru? A:" and it will usually complete "Lima". Write three worked examples and it will do a fourth. This <em>few-shot prompting</em> was the headline of the GPT-3 paper.`),
        p(`But it is fragile, it does not know when to stop, and it has no notion of being helpful, honest or safe. It simply is not trying to do anything except predict. Turning that into an assistant is a separate stage — <em>post-training</em> — and it is the whole of chapter 11.`),
      ));

      root.append(section('Two things that changed the shape of models',
        p(`<b>Mixture of experts.</b> In a standard transformer every token passes through every parameter. In an MoE each feed-forward block becomes many parallel experts, and a tiny router sends each token to only one or two of them.`),
        callout('tryit', '🖐 Try this',
          `<b>1.</b> Press <b>DeepSeek-V3-ish</b>. Read the two bars: an enormous amount stored, a small fraction used per token.<br>
           <b>2.</b> Drag <b>experts used per token</b> up and watch the two bars close on each other. That slider is the whole trade-off.<br>
           <b>3.</b> Now press <b>compare with a dense model</b> — stored and used become the same number, and the bill per token jumps. (In dense mode the expert sliders have nothing left to do, which is the point.)`),
        moeLab(),
        p(`So the model can hold far more knowledge while each token touches only a fraction of it. GPT-4 was widely reported to use this design; Mixtral 8×7B made it mainstream in open models in December 2023; DeepSeek-V3 has 671 billion parameters of which about 37 billion are active per token, which is why it could be trained for a reported few million dollars of GPU time.`),
        p(`The cost is memory: every expert must be loaded even though most sit idle. And note what this does to the rule of thumb — for an MoE, the 6ND estimate uses the <i>active</i> parameter count, not the total.`),
        p(`<b>Context length.</b> GPT-3 could see 2,048 tokens, about three pages. GPT-4 launched with 8k and 32k variants in 2023; 128k — a short novel — became standard by 2024; Gemini 1.5 offered a million tokens.`),
        p(`That took better positional encodings (RoPE and its extensions), attention variants that avoid the quadratic memory blow-up, and simply training on long documents. Chapter 12 covers why long contexts are expensive to <i>serve</i>, and why a model does not necessarily use all of what it can see.`),
        callout('history', '📜 The GPT line, in four steps',
          `<b>GPT-1 (2018):</b> 117M parameters, ~5GB of books. Proof that pretrain-then-finetune works for language.<br>
           <b>GPT-2 (2019):</b> 1.5B parameters, 40GB of web text. Fluent enough that its staged release became a public argument about AI risk.<br>
           <b>GPT-3 (2020):</b> 175B parameters, 300B tokens. Few-shot prompting works without any finetuning at all — the paper that convinced the field scale was the road.<br>
           <b>GPT-4 (2023) onward:</b> details undisclosed, but widely believed to be a mixture of experts trained on trillions of tokens, with the real gains increasingly coming from post-training rather than pretraining.`),
      ));

      root.append(section('Why this matters for modern AI',
        p(`Three consequences follow from everything above.`),
        p(`First, <b>knowledge has a cut-off</b>. Whatever was not in the pretraining data is not in the model, which is why models need retrieval and tools — chapter 12 — for anything recent.`),
        p(`Second, <b>the loss is a proxy</b>. A model trained to imitate the internet imitates the internet's errors, biases and confident nonsense too. That is why post-training exists, and a large part of why hallucination is so hard to remove: fluent-but-wrong text is exactly what the objective asked for.`),
        p(`Third, <b>the economics are brutal and predictable</b>. Because scaling laws hold, labs can forecast what a 10× bigger run will achieve before spending the money. That predictability is the reason the money keeps being spent.`),
        p(`It also explains the shape of the industry. Pretraining is a capital expense a handful of organisations can afford, and it happens once. Everything after it — post-training, tools, agents, the product you actually use — is comparatively cheap and happens continuously.`),
        p(`When you hear that a lab "released a new model", it is usually the second half that changed. One picture to keep: <b>pretraining fills the reservoir, and everything in Part III after this chapter is plumbing.</b>`),
      ));

      root.append(ctx.quiz([
        { q: 'Why does "guess the next token" teach a model history, arithmetic and code?', options: ['Those subjects are labelled in the training data', 'Because predicting text well requires understanding what the text is about — every fact or skill that lowers the loss gets learned as a side effect', 'Because the tokenizer separates subjects', 'It does not; those abilities are added later'], answer: 1, explain: 'To finish "and the murderer was…" on the last page of a mystery, you have to have solved the murder. No part of the objective mentions history or arithmetic; they are simply useful for prediction, so gradient descent finds them.' },
        { q: 'A model assigns 1% probability to the token that actually came next. What is the loss, and why is the scale worth knowing?', options: ['0.01, which is small', '4.6 nats — and pure guessing over a 100,000-token vocabulary costs about 11.5, so 4.6 is bad but far from random', '100, because it was 100× wrong', '1, one per token'], answer: 1, explain: '−log(0.01) ≈ 4.6. The useful anchors are log(100,000) ≈ 11.5 for an untrained model and roughly 2 for a good one on web text. Perplexity = e^loss turns that into "as unsure as an N-way multiple choice".' },
        { q: 'Both GPT-3 (175B params, 300B tokens) and a hypothetical 17.5B model on 3T tokens cost the same to train. Why?', options: ['They do not — bigger models always cost more', 'Training compute is about 6ND, so trading parameters against tokens at a constant product leaves the bill unchanged', 'Because tokens are free', 'Because of mixed precision'], answer: 1, explain: 'You can check this on the cost explorer: raise N by 10x or raise D by 10x and the bill moves identically. Chinchilla\'s contribution was showing which side of that trade actually buys you a better model — roughly 20 tokens per parameter.' },
        { q: 'Why do labs train small models far past the "compute-optimal" point, as Llama 3 did with 15T tokens for an 8B model?', options: ['Because Chinchilla was wrong', 'Chinchilla minimises training cost, but a deployed model spends far more compute on inference, and inference cost scales with parameters — so a smaller, longer-trained model is cheaper to serve', 'To avoid overfitting', 'Because more tokens are always optimal'], answer: 1, explain: 'It is a deliberate trade: a higher training bill once, in exchange for a permanently cheaper model to run for millions of users. At nearly 2,000 tokens per parameter the loss was still falling.' },
        { q: 'You ask a raw base model "What is the capital of Peru?" and it replies with three more geography questions. What is going on?', options: ['The model does not know the answer', 'It is working perfectly — on the web a question is most often followed by more questions, and a base model predicts the most plausible continuation rather than answering', 'The tokenizer failed', 'It needs a longer context window'], answer: 1, explain: 'A base model is an autocomplete engine for the internet, not a chatbot. Few-shot prompting works by making the answer the most plausible continuation, but it is fragile and has no notion of when to stop. Post-training, in chapter 11, is what fixes this.' },
      ]));

      root.append(section('Go deeper',
        ul([
          '<a href="https://www.youtube.com/watch?v=zduSFxRajkE" target="_blank" rel="noopener">Karpathy, "Let\'s build the GPT Tokenizer"</a> and <a href="https://www.youtube.com/watch?v=zjkBMFhNj_g" target="_blank" rel="noopener">"Intro to Large Language Models"</a> — the best available explanations of what pretraining actually produces.',
          '<a href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noopener">Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models"</a> — the Chinchilla paper, and the 20-tokens-per-parameter result that recalibrated the field.',
          '<a href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noopener">Kaplan et al. (2020), "Scaling Laws for Neural Language Models"</a> — the original power laws.',
          '<a href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noopener">The Llama 3 Herd of Models (2024)</a> — an unusually candid engineering report, including the 466 job interruptions.',
          '<a href="https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1" target="_blank" rel="noopener">FineWeb: decanting the web for the finest text data at scale</a> — what the data funnel really looks like, with ablations.',
          '<a href="https://lilianweng.github.io/posts/2021-09-25-train-large/" target="_blank" rel="noopener">Lilian Weng, "How to Train Really Large Models on Many GPUs?"</a> — data, tensor and pipeline parallelism in proper detail.',
        ])));
    },
  });
})();
