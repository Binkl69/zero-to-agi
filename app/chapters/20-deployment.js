/* Zero → AGI · Chapter 20 · Deployment and life after launch
   DESIGN RULE: the reader breaks a serving system with their own hands before any of its
   vocabulary appears. Every number the prose quotes is printed by a demo, and every demo's data
   comes from a seeded generator, so the numbers are the same on every load.
   Interactives, in order: a queue simulator (arrival rate against capacity; the opener, and the
   subject of the maths beat ρ ÷ (1 − ρ)); a latency histogram where a fine average hides a bad
   tail; a batching dial trading time per request against GPUs; a bill with a cache, a router and
   a rate limit; an alert threshold trading false alarms against time to notice; a year of drift
   in which the reader buys retrains; a release shipped to everyone or to a canary; and a call
   to a flaky provider with a timeout, retries and a backup. */
(function () {
  ZTA.registerChapter({
    id: '20-deployment',
    num: 20,
    part: 'VI',
    title: 'Deployment and life after launch',
    tagline: 'A model that works on your laptop has passed the easy exam. Queue it, bill it, watch it go stale, ship a bad one, and learn what keeps it standing.',
    minutes: 60,
    objectives: [
      'predict how waiting time grows as a server gets busier, using ρ ÷ (1 − ρ)',
      'read p50, p95 and p99, and say when a healthy average is hiding a broken tail',
      'cut a serving bill with batching, caching, routing and limits, and name what each one costs you',
      'plan a release and its aftermath: a canary, an alert, a retraining trigger, a fallback',
    ],
    takeaways: [
      '<b>Busy servers queue.</b> With one server, the average wait is ρ ÷ (1 − ρ) jobs long. At 50% busy that is one job, at 90% nine, at 99% ninety-nine. Run with headroom.',
      '<b>Watch the tail.</b> An average can sit still while one request in fifty takes twenty times too long. Budgets and alerts belong on p95 and p99.',
      '<b>Every saving has a price.</b> Batching buys throughput with waiting, a cache buys speed with staleness, a router buys money with accuracy, a limit buys safety with refusals.',
      '<b>The world moves; the weights do not.</b> A frozen model decays as its inputs drift. Signals that need no labels, such as the share of inputs it flags, warn you weeks before the labels arrive.',
      '<b>Ship small, then widen.</b> Damage is share of traffic × extra error rate × minutes until someone notices. A canary shrinks the first term; a rollback switch shrinks the last.',
      '<b>Plan the failure path.</b> Timeouts, a few spaced retries, a backup answer and a route to a person decide what a user sees on the day the model is wrong, slow or down.',
    ],
    render(root, ctx) {
      const { h, p, section, callout, ul, cards } = ctx;
      const C = ctx.colors;
      const FONT = '13px Inter, system-ui, sans-serif';
      const BOLD = 'bold 13px Inter, system-ui, sans-serif';
      const PANEL = '#0f1520';

      /* Seeded generator: every demo below draws from its own copy, so nothing the prose
         quotes depends on Math.random. */
      function mulberry32(a) {
        return function () {
          a |= 0; a = (a + 0x6D2B79F5) | 0;
          let t = Math.imul(a ^ (a >>> 15), 1 | a);
          t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }
      function gaussFrom(r) { return () => { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }; }
      const num = (v) => Math.round(v).toLocaleString('en-GB');
      const money = (v) => '$' + num(v);
      const ms = (v) => (v < 10000 ? num(v) + ' ms' : (v / 1000).toFixed(1) + ' s');
      const secs = (v) => (v < 10 ? v.toFixed(1) : String(Math.round(v))) + ' s';
      function box(g, r) { g.fillStyle = PANEL; g.fillRect(r.x, r.y, r.w, r.h); g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(r.x, r.y, r.w, r.h); }
      function clipTo(g, r, fn) { g.save(); g.beginPath(); g.rect(r.x, r.y, r.w, r.h); g.clip(); fn(); g.restore(); }
      function text(g, s, x, y, color, align, font) { g.font = font || FONT; g.fillStyle = color || C.muted; g.textAlign = align || 'left'; g.fillText(s, x, y); }

      /* ================================================================== */
      /* 1. Queue simulator: arrival rate against capacity                    */
      /* ================================================================== */
      function queueSim() {
        const MU = 10, N = 400000, SEED = 2026, SHOW = 120, XMAX = 33, YMAX = 2000;
        const [cq, gq] = ctx.canvas(720, 200);
        const [cw, gw] = ctx.canvas(720, 204);
        const st = { lam: 5, c: 1 };
        const cache = {};
        /* Requests arrive at random moments (exponential gaps) and each takes a random time
           averaging 100 ms (exponential). A request waits for whichever server frees first. */
        function run(lam, c) {
          const key = lam + '|' + c;
          if (cache[key]) return cache[key];
          const r = mulberry32(SEED);
          const over = lam >= c * MU;
          const free = new Float64Array(c);
          const BIN = 0.005, hist = new Uint32Array(20001);
          const dt = 0.25, nb = SHOW / dt, diff = new Float32Array(nb + 2);
          let t = 0, sum = 0, n = 0, lastWait = 0;
          for (let i = 0; i < N; i++) {
            t += -Math.log(1 - r()) / lam;
            if (over && t > SHOW) break;
            let k = 0;
            for (let j = 1; j < c; j++) if (free[j] < free[k]) k = j;
            const start = Math.max(t, free[k]), w = start - t;
            free[k] = start + -Math.log(1 - r()) / MU;
            sum += w; n++; lastWait = w;
            hist[Math.min(20000, Math.floor(w / BIN))]++;
            if (t < SHOW) { const a = Math.ceil(t / dt), b = Math.min(nb, Math.ceil(start / dt)); if (b > a) { diff[a]++; diff[b]--; } }
          }
          const q = new Float32Array(nb);
          let acc = 0, maxQ = 0;
          for (let i = 0; i < nb; i++) { acc += diff[i]; q[i] = acc; if (acc > maxQ) maxQ = acc; }
          let cum = 0, p99 = 0;
          for (let b = 0; b <= 20000; b++) { cum += hist[b]; if (cum >= 0.99 * n) { p99 = (b + 1) * BIN * 1000; break; } }
          return (cache[key] = { over, mean: sum / n * 1000, p99, q, maxQ, lastWait: lastWait * 1000 });
        }
        /* long-run average wait for c servers (Erlang's formula); for c = 1 it is ρ/(1−ρ) × 100 ms */
        function theory(lam, c) {
          const a = lam / MU, rho = a / c;
          if (rho >= 1) return Infinity;
          let term = 1, s = 1;
          for (let k = 1; k < c; k++) { term *= a / k; s += term; }
          const last = term * a / c / (1 - rho);
          return last / (s + last) / (c * MU - lam) * 1000;
        }
        const ro = ctx.readout();
        function draw() {
          const R = run(st.lam, st.c), cap = st.c * MU;
          /* --- queue length over the first two minutes --- */
          const fq = { x: 50, y: 30, w: 650, h: 136 };
          gq.clearRect(0, 0, 720, 200);
          box(gq, fq);
          const ymax = Math.max(10, R.maxQ);
          text(gq, 'requests waiting in the queue, first two minutes', fq.x, 19, C.text, 'left', BOLD);
          text(gq, 'peak ' + num(R.maxQ), fq.x + fq.w, 19, R.over ? C.danger : C.muted, 'right');
          clipTo(gq, fq, () => {
            gq.beginPath(); gq.moveTo(fq.x, fq.y + fq.h);
            for (let i = 0; i < R.q.length; i++) {
              const x = fq.x + i / R.q.length * fq.w, x2 = fq.x + (i + 1) / R.q.length * fq.w, y = fq.y + fq.h - R.q[i] / ymax * (fq.h - 4);
              gq.lineTo(x, y); gq.lineTo(x2, y);
            }
            gq.lineTo(fq.x + fq.w, fq.y + fq.h); gq.closePath();
            gq.fillStyle = R.over ? 'rgba(251,113,133,0.55)' : 'rgba(124,156,255,0.6)'; gq.fill();
          });
          text(gq, '0', fq.x - 6, fq.y + fq.h, C.muted, 'right');
          text(gq, num(ymax), fq.x - 6, fq.y + 10, C.muted, 'right');
          for (let s = 0; s <= SHOW; s += 30) text(gq, s + ' s', fq.x + s / SHOW * fq.w, fq.y + fq.h + 17, C.muted, s === 0 ? 'left' : s === SHOW ? 'right' : 'center');
          /* --- average wait against arrival rate --- */
          const fw = { x: 50, y: 30, w: 650, h: 136 };
          gw.clearRect(0, 0, 720, 204);
          box(gw, fw);
          text(gw, 'average wait at every arrival rate, with ' + st.c + (st.c === 1 ? ' server' : ' servers'), fw.x, 19, C.text, 'left', BOLD);
          text(gw, 'yellow dashes = capacity', fw.x + fw.w, 19, C.warn, 'right');
          const X = (l) => fw.x + l / XMAX * fw.w, Y = (v) => fw.y + fw.h - Math.min(1, v / YMAX) * (fw.h - 8);
          clipTo(gw, fw, () => {
            gw.fillStyle = 'rgba(251,113,133,0.10)'; gw.fillRect(X(cap), fw.y, fw.x + fw.w - X(cap), fw.h);
            gw.strokeStyle = C.warn; gw.setLineDash([5, 4]); gw.lineWidth = 1.5; gw.beginPath(); gw.moveTo(X(cap), fw.y); gw.lineTo(X(cap), fw.y + fw.h); gw.stroke(); gw.setLineDash([]);
            gw.strokeStyle = C.accent; gw.lineWidth = 2.2; gw.beginPath();
            for (let i = 0; i <= 400; i++) {
              const l = 0.05 + (cap - 0.06) * i / 400, v = Math.min(YMAX * 3, theory(l, st.c));
              const y = fw.y + fw.h - v / YMAX * (fw.h - 8);
              if (!i) gw.moveTo(X(l), y); else gw.lineTo(X(l), y);
            }
            gw.stroke();
            const dy = R.over ? fw.y + 8 : Y(R.mean);
            gw.beginPath(); gw.arc(X(st.lam), dy, 6, 0, Math.PI * 2); gw.fillStyle = R.over ? C.danger : C.green; gw.fill(); gw.strokeStyle = '#0a0e16'; gw.lineWidth = 1.5; gw.stroke();
          });
          if (fw.x + fw.w - X(cap) > 240) text(gw, 'overload: the queue never drains', (X(cap) + fw.x + fw.w) / 2, fw.y + fw.h / 2 + 4, C.danger, 'center');
          text(gw, '0', fw.x - 6, fw.y + fw.h, C.muted, 'right');
          text(gw, '1 s', fw.x - 6, Y(1000) + 4, C.muted, 'right');
          text(gw, '2 s', fw.x - 6, Y(2000) + 6, C.muted, 'right');
          for (let l = 0; l <= 30; l += 10) text(gw, String(l), X(l), fw.y + fw.h + 17, C.muted, l === 0 ? 'left' : 'center');
          text(gw, 'across: requests arriving per second   blue = long-run average   dot = your simulation', fw.x, fw.y + fw.h + 34, C.muted, 'left');
          ro.set({
            'how busy (utilisation)': Math.round(st.lam / cap * 100) + '%',
            'average wait': R.over ? 'no limit (' + secs(R.lastWait / 1000) + ' after 2 min)' : ms(R.mean),
            'p99 wait': R.over ? 'no limit' : ms(R.p99),
            'longest queue': num(R.maxQ),
          });
        }
        const lamSl = ctx.slider({ label: 'requests arriving per second', min: 1, max: 32, step: 0.5, value: 5, fmt: (v) => String(v), onChange: (v) => { st.lam = v; draw(); } });
        const cSl = ctx.slider({ label: 'servers', min: 1, max: 3, step: 1, value: 1, onChange: (v) => { st.c = v; draw(); } });
        draw();
        return ctx.figure(h('div', {}, cq, cw),
          'A simulated service. Requests arrive at random moments; each takes a random time averaging 100 ms, so one server clears 10 a second flat out. Top: the queue during the first two minutes. The readout averages over 400,000 simulated requests; in overload there is no average to report, so it shows how long the latest arrival had waited after two minutes. Bottom: the long-run average wait at every arrival rate, with your simulation as the dot. The dice are fixed so the numbers match the text; another roll moves them a few per cent at moderate load and 10% or more close to capacity. Real services have less tidy randomness, but the same hockey stick.',
          [lamSl, cSl], ro);
      }

      /* ================================================================== */
      /* 2. The tail: a fine average hiding a bad p99                         */
      /* ================================================================== */
      function tailDemo() {
        const N = 10000, W = 720, H = 250;
        const [cv, g] = ctx.canvas(W, H);
        const r = mulberry32(7), gs = gaussFrom(r);
        const u = new Float32Array(N), zf = new Float32Array(N), zs = new Float32Array(N);
        for (let i = 0; i < N; i++) { u[i] = r(); zf[i] = gs(); zs[i] = gs(); }
        const st = { slow: 0, visit: 20 };
        const ro = ctx.readout();
        const fr = { x: 50, y: 30, w: 470, h: 176 };
        const LO = Math.log(50), HI = Math.log(20000), NB = 64;
        const X = (v) => fr.x + (Math.log(v) - LO) / (HI - LO) * fr.w;
        function draw() {
          const lat = new Float32Array(N);
          let sum = 0;
          for (let i = 0; i < N; i++) { lat[i] = u[i] < st.slow / 100 ? 4000 * Math.exp(0.3 * zs[i]) : 180 * Math.exp(0.35 * zf[i]); sum += lat[i]; }
          const sorted = Float32Array.from(lat).sort();
          const pct = (q) => sorted[Math.min(N - 1, Math.floor(q * N))];
          const mean = sum / N, p50 = pct(0.5), p95 = pct(0.95), p99 = pct(0.99);
          const visits = Math.floor(N / st.visit);
          let bad = 0;
          for (let v = 0; v < visits; v++) { let hit = false; for (let k = 0; k < st.visit; k++) if (lat[v * st.visit + k] > 1000) { hit = true; break; } if (hit) bad++; }
          const bins = new Array(NB).fill(0);
          for (let i = 0; i < N; i++) { const b = Math.floor((Math.log(lat[i]) - LO) / (HI - LO) * NB); if (b >= 0 && b < NB) bins[b]++; }
          const peak = Math.sqrt(Math.max(...bins));
          g.clearRect(0, 0, W, H);
          box(g, fr);
          text(g, 'how long 10,000 requests took', fr.x, 19, C.text, 'left', BOLD);
          clipTo(g, fr, () => {
            const bw = fr.w / NB;
            for (let b = 0; b < NB; b++) {
              if (!bins[b]) continue;
              const hh = Math.sqrt(bins[b]) / peak * (fr.h - 10);
              const mid = Math.exp(LO + (b + 0.5) / NB * (HI - LO));
              g.fillStyle = mid > 1000 ? C.danger : C.accent;
              g.fillRect(fr.x + b * bw + 0.5, fr.y + fr.h - hh, bw - 1, hh);
            }
            const mark = (v, col, dash) => { g.strokeStyle = col; g.lineWidth = 2; g.setLineDash(dash || []); g.beginPath(); g.moveTo(X(v), fr.y); g.lineTo(X(v), fr.y + fr.h); g.stroke(); g.setLineDash([]); };
            mark(p50, C.green); mark(p95, C.warn); mark(p99, C.orange); mark(mean, '#ffffff', [4, 4]);
          });
          [[100, '100 ms'], [1000, '1 s'], [10000, '10 s']].forEach(([v, s]) => text(g, s, X(v), fr.y + fr.h + 17, C.muted, 'center'));
          text(g, 'bars: blue under 1 s, red over. Heights on a square-root scale.', fr.x, fr.y + fr.h + 36, C.muted, 'left');
          const lx = 540;
          [['average', mean, '#ffffff'], ['p50 (median)', p50, C.green], ['p95', p95, C.warn], ['p99', p99, C.orange]].forEach(([name, v, col], i) => {
            const y = 56 + i * 40;
            g.fillStyle = col; g.fillRect(lx, y - 10, 4, 28);
            text(g, name, lx + 12, y, C.muted, 'left');
            text(g, ms(v), lx + 12, y + 18, C.text, 'left', BOLD);
          });
          ro.set({
            average: ms(mean), 'p50': ms(p50), 'p95': ms(p95), 'p99': ms(p99),
            ['visits of ' + st.visit + ' that hit a request over 1 s']: Math.round(bad / visits * 100) + '%',
          });
        }
        const slowSl = ctx.slider({ label: 'requests that take the slow path', min: 0, max: 10, step: 0.5, value: 0, fmt: (v) => v + '%', onChange: (v) => { st.slow = v; draw(); } });
        const visitSl = ctx.slider({ label: 'requests in one visit', min: 1, max: 40, step: 1, value: 20, onChange: (v) => { st.visit = v; draw(); } });
        const okBtn = ctx.button('Healthy day', () => { st.slow = 0; slowSl.value = 0; draw(); });
        const coldBtn = ctx.button('Cold cache: 2% slow', () => { st.slow = 2; slowSl.value = 2; draw(); }, 'primary');
        draw();
        return ctx.figure(cv,
          'Ten thousand seeded requests on a logarithmic time axis. Most take the fast path, around 180 ms. A chosen share take a slow path of about 4 seconds: a cache that missed, a long reply, a retry. The dashed white line is the average; the solid lines are the 50th, 95th and 99th percentiles. The last readout groups the requests into visits and counts the visits that met at least one request slower than a second.',
          [slowSl, visitSl, okBtn, coldBtn], ro);
      }

      /* ================================================================== */
      /* 3. Batching: throughput against time per request                     */
      /* ================================================================== */
      function batchLab() {
        const W = 720, H = 286, BUDGET = 300, PRICE = 2;
        const [cv, g] = ctx.canvas(W, H);
        const st = { b: 1, lam: 100 };
        const pass = (b) => 60 + 8 * b;                       /* ms for one GPU pass over a batch of b */
        const fill = (b, lam) => (b - 1) / (2 * lam) * 1000;   /* average ms a request waits for its batch to fill */
        const perGpu = (b) => b / pass(b) * 1000;              /* requests per second from one GPU */
        const gpus = (b, lam) => Math.ceil(lam / (0.7 * perGpu(b)));
        const ro = ctx.readout();
        const fa = { x: 60, y: 30, w: 640, h: 120 }, fb = { x: 60, y: 186, w: 640, h: 64 };
        const X = (b) => fa.x + 8 + (b - 1) / 63 * (fa.w - 16);
        function draw() {
          const lat = pass(st.b) + fill(st.b, st.lam), G = gpus(st.b, st.lam), ok = lat <= BUDGET;
          g.clearRect(0, 0, W, H);
          box(g, fa); box(g, fb);
          text(g, 'time per request', fa.x, 19, C.text, 'left', BOLD);
          text(g, 'blue = total   grey dashes = the GPU pass alone', fa.x + fa.w, 19, C.muted, 'right');
          const Y = (v) => fa.y + fa.h - v / 800 * fa.h;
          clipTo(g, fa, () => {
            g.strokeStyle = C.warn; g.setLineDash([5, 4]); g.lineWidth = 1.5; g.beginPath(); g.moveTo(fa.x, Y(BUDGET)); g.lineTo(fa.x + fa.w, Y(BUDGET)); g.stroke();
            g.strokeStyle = C.muted; g.setLineDash([3, 4]); g.beginPath(); for (let b = 1; b <= 64; b++) { if (b === 1) g.moveTo(X(b), Y(pass(b))); else g.lineTo(X(b), Y(pass(b))); } g.stroke(); g.setLineDash([]);
            g.strokeStyle = C.accent; g.lineWidth = 2.2; g.beginPath(); for (let b = 1; b <= 64; b++) { const y = Y(Math.min(1200, pass(b) + fill(b, st.lam))); if (b === 1) g.moveTo(X(b), y); else g.lineTo(X(b), y); } g.stroke();
            g.strokeStyle = ok ? C.green : C.danger; g.lineWidth = 1.2; g.beginPath(); g.moveTo(X(st.b), fa.y); g.lineTo(X(st.b), fa.y + fa.h); g.stroke();
            g.beginPath(); g.arc(X(st.b), Math.max(fa.y + 6, Y(lat)), 6, 0, Math.PI * 2); g.fillStyle = ok ? C.green : C.danger; g.fill();
          });
          text(g, '0', fa.x - 6, fa.y + fa.h, C.muted, 'right');
          text(g, '300 ms', fa.x - 6, Y(BUDGET) + 4, C.warn, 'right');
          text(g, '800 ms', fa.x - 6, fa.y + 10, C.muted, 'right');
          text(g, 'GPUs needed for this traffic', fb.x, fb.y - 9, C.text, 'left', BOLD);
          const gmax = gpus(1, st.lam);
          clipTo(g, fb, () => {
            const bw = (fb.w - 16) / 63;
            for (let b = 1; b <= 64; b++) {
              const hh = gpus(b, st.lam) / gmax * (fb.h - 6);
              g.fillStyle = b === st.b ? (ok ? C.green : C.danger) : 'rgba(148,163,184,0.45)';
              g.fillRect(X(b) - bw / 2 + 0.5, fb.y + fb.h - hh, bw - 1, hh);
            }
          });
          text(g, String(gmax), fb.x - 6, fb.y + 10, C.muted, 'right');
          text(g, '0', fb.x - 6, fb.y + fb.h, C.muted, 'right');
          [1, 16, 32, 48, 64].forEach((b) => text(g, b === 64 ? 'batch size 64' : String(b), X(b), fb.y + fb.h + 17, C.muted, b === 64 ? 'right' : 'center'));
          ro.set({
            'time per request': ms(lat),
            'budget of 300 ms': ok ? 'met' : 'missed',
            'one GPU serves': perGpu(st.b).toFixed(0) + ' / s',
            'GPUs needed': G,
            'bill per hour': money(G * PRICE),
          });
        }
        const bSl = ctx.slider({ label: 'batch size', min: 1, max: 64, step: 1, value: 1, onChange: (v) => { st.b = v; draw(); } });
        const lamSl = ctx.slider({ label: 'requests arriving per second', min: 10, max: 400, step: 10, value: 100, onChange: (v) => { st.lam = v; draw(); } });
        draw();
        return ctx.figure(cv,
          'An illustrative GPU: one pass over a batch takes 60 ms plus 8 ms per request in it, so a batch of 16 costs 188 ms, far less than sixteen separate passes. The server waits until the batch is full, which adds waiting time when traffic is thin. GPUs are counted so that each stays under 70% busy (the lesson of the queue above), at an illustrative $2 per GPU-hour. Simplified: queueing in front of the batcher is left out.',
          [bSl, lamSl], ro);
      }

      /* ================================================================== */
      /* 4. The bill: cache, router, rate limit                               */
      /* ================================================================== */
      function costLab() {
        const W = 720, H = 196, USERS = 5000, BIG = 0.02, SMALL = 0.002;
        const [cv, g] = ctx.canvas(W, H);
        const r = mulberry32(404);
        const perUser = new Int32Array(USERS);
        let TOTAL = 0;
        for (let i = 0; i < USERS; i++) { perUser[i] = Math.min(4000, Math.max(1, Math.floor(3.4 / Math.pow(1 - r(), 1 / 1.2)))); TOTAL += perUser[i]; }
        const st = { cache: 0, route: 0, cap: 0 };
        const ro = ctx.readout();
        function calc() {
          let kept = 0, hit = 0;
          for (let i = 0; i < USERS; i++) { if (st.cap && perUser[i] > st.cap) { kept += st.cap; hit++; } else kept += perUser[i]; }
          const refused = TOTAL - kept, c = st.cache / 100, rt = st.route / 100;
          const cached = kept * c, small = kept * (1 - c) * rt, big = kept * (1 - c) * (1 - rt);
          /* the router sends the easiest requests first; 60% of traffic is easy. Right-answer
             rates: large model 95% on anything, small model 93% on easy requests, 55% on hard. */
          const right = c * 0.95 + (1 - c) * ((1 - rt) * 0.95 + Math.min(rt, 0.6) * 0.93 + Math.max(0, rt - 0.6) * 0.55);
          return { refused, cached, small, big, hit, right, bill: small * SMALL + big * BIG, base: TOTAL * BIG };
        }
        function draw() {
          const R = calc();
          g.clearRect(0, 0, W, H);
          const bx = 20, bw = 680;
          text(g, 'where ' + num(TOTAL) + ' requests a day went', bx, 19, C.text, 'left', BOLD);
          const parts = [['refused', R.refused, C.muted], ['from cache', R.cached, C.green], ['small model', R.small, C.accent], ['large model', R.big, C.orange]];
          let x = bx;
          g.fillStyle = PANEL; g.fillRect(bx, 30, bw, 30);
          parts.forEach(([, v, col]) => { const w = v / TOTAL * bw; g.fillStyle = col; g.fillRect(x, 30, w, 30); x += w; });
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(bx, 30, bw, 30);
          parts.forEach(([name, v, col], i) => { const lx = bx + i * 172; g.fillStyle = col; g.fillRect(lx, 72, 11, 11); text(g, name + ' ' + num(v), lx + 17, 82, C.text, 'left'); });
          text(g, 'the bill, against ' + money(R.base) + ' a day with no controls', bx, 119, C.text, 'left', BOLD);
          g.fillStyle = PANEL; g.fillRect(bx, 130, bw, 30);
          const ws = R.small * SMALL / R.base * bw, wb = R.big * BIG / R.base * bw;
          g.fillStyle = C.accent; g.fillRect(bx, 130, ws, 30);
          g.fillStyle = C.orange; g.fillRect(bx + ws, 130, wb, 30);
          g.strokeStyle = C.line; g.strokeRect(bx, 130, bw, 30);
          g.fillStyle = C.accent; g.fillRect(bx, 172, 11, 11); text(g, 'small model ' + money(R.small * SMALL), bx + 17, 182, C.text, 'left');
          g.fillStyle = C.orange; g.fillRect(bx + 172, 172, 11, 11); text(g, 'large model ' + money(R.big * BIG), bx + 189, 182, C.text, 'left');
          ro.set({
            'bill per day': money(R.bill),
            saved: Math.round((1 - R.bill / R.base) * 100) + '%',
            'right answers': (R.right * 100).toFixed(1) + '%',
            'users who hit the limit': num(R.hit) + ' of ' + num(USERS),
            'requests refused': num(R.refused),
          });
        }
        const cacheSl = ctx.slider({ label: 'answers served from a cache', min: 0, max: 50, step: 5, value: 0, fmt: (v) => v + '%', onChange: (v) => { st.cache = v; draw(); } });
        const routeSl = ctx.slider({ label: 'sent to the small model', min: 0, max: 100, step: 5, value: 0, fmt: (v) => v + '%', onChange: (v) => { st.route = v; draw(); } });
        const capSel = ctx.select({ label: 'daily limit per user', value: '0', options: [{ value: '0', label: 'no limit' }, { value: '500', label: '500 requests' }, { value: '100', label: '100 requests' }, { value: '30', label: '30 requests' }], onChange: (v) => { st.cap = +v; draw(); } });
        draw();
        return ctx.figure(cv,
          'One day of traffic from 5,000 seeded users, a few of whom send thousands of requests. <b>Every price here is illustrative</b>: 2 cents a request on the large model, 0.2 cents on the small one, nothing for a cached answer. Six requests in ten are easy. The large model is right 95% of the time on anything; the small one is right 93% of the time on easy requests and 55% on hard ones. The router is assumed to send the easiest requests first, which real routers only approximate.',
          [cacheSl, routeSl, capSel], ro);
      }

      /* ================================================================== */
      /* 5. Alerts: false alarms against time to notice                       */
      /* ================================================================== */
      function alertLab() {
        const W = 720, H = 256, DAYS = 14, PERDAY = 288, NW = DAYS * PERDAY, REQ = 200, START = 10 * PERDAY + 108;
        const [cv, g] = ctx.canvas(W, H);
        const r = mulberry32(911);
        const pre = new Float64Array(NW + 1);
        for (let i = 0; i < NW; i++) { const pr = i >= START ? 0.04 : 0.02; let k = 0; for (let j = 0; j < REQ; j++) if (r() < pr) k++; pre[i + 1] = pre[i] + k; }
        const st = { thr: 3, k: 1 };
        const ro = ctx.readout();
        const fr = { x: 50, y: 48, w: 650, h: 150 };
        function draw() {
          const roll = new Float32Array(NW);
          for (let i = 0; i < NW; i++) { const a = Math.max(0, i + 1 - st.k); roll[i] = (pre[i + 1] - pre[a]) / (REQ * (i + 1 - a)) * 100; }
          const falseAt = [];
          let caught = -1;
          /* alarms are only judged once the window is full; a part-filled window is too jumpy */
          for (let i = st.k; i < NW; i++) {
            const up = roll[i] > st.thr && roll[i - 1] <= st.thr;
            if (i < START) { if (up) falseAt.push(i); } else if (caught < 0 && roll[i] > st.thr) caught = i;
          }
          const X = (i) => fr.x + i / NW * fr.w, Y = (v) => fr.y + fr.h - Math.min(10, v) / 10 * fr.h;
          g.clearRect(0, 0, W, H);
          box(g, fr);
          text(g, 'failed requests, two weeks', fr.x, 19, C.text, 'left', BOLD);
          text(g, 'yellow = alert line   red ticks = false alarms   green = fault caught', fr.x + fr.w, 19, C.muted, 'right');
          g.fillStyle = C.danger;
          const shown = falseAt.length > 400 ? falseAt.filter((_, i) => i % Math.ceil(falseAt.length / 400) === 0) : falseAt;
          shown.forEach((i) => g.fillRect(X(i) - 0.75, 30, 1.5, 12));
          if (caught >= 0) { g.fillStyle = C.green; g.fillRect(X(caught) - 2, 28, 4, 16); }
          clipTo(g, fr, () => {
            g.fillStyle = 'rgba(251,146,60,0.10)'; g.fillRect(X(START), fr.y, fr.x + fr.w - X(START), fr.h);
            g.strokeStyle = C.accent; g.lineWidth = 1; g.beginPath();
            const cols = Math.floor(fr.w);
            for (let c = 0; c < cols; c++) {
              const a = Math.max(st.k - 1, Math.floor(c / cols * NW)), b = Math.max(a + 1, Math.floor((c + 1) / cols * NW));
              let lo = 100, hi = 0;
              for (let i = a; i < b; i++) { if (roll[i] < lo) lo = roll[i]; if (roll[i] > hi) hi = roll[i]; }
              g.moveTo(fr.x + c + 0.5, Y(hi) - 0.5); g.lineTo(fr.x + c + 0.5, Y(lo) + 0.5);
            }
            g.stroke();
            g.strokeStyle = C.warn; g.lineWidth = 1.5; g.beginPath(); g.moveTo(fr.x, Y(st.thr)); g.lineTo(fr.x + fr.w, Y(st.thr)); g.stroke();
          });
          text(g, '0%', fr.x - 6, fr.y + fr.h, C.muted, 'right');
          text(g, '5%', fr.x - 6, Y(5) + 4, C.muted, 'right');
          text(g, '10%', fr.x - 6, fr.y + 10, C.muted, 'right');
          [1, 4, 7, 10, 13].forEach((d) => text(g, 'day ' + d, X((d - 1) * PERDAY), fr.y + fr.h + 17, C.muted, d === 1 ? 'left' : 'center'));
          text(g, 'the real fault starts here (day 11, 09:00) ▲', X(START) + 6, fr.y + fr.h + 38, C.orange, 'right');
          const mins = caught < 0 ? -1 : (caught - START + 1) * 5;
          ro.set({
            'false alarms in ten quiet days': num(falseAt.length),
            'real fault caught after': mins < 0 ? 'never' : mins < 120 ? mins + ' min' : (mins / 60).toFixed(1) + ' hours',
          });
        }
        const thrSl = ctx.slider({ label: 'page someone when failures pass', min: 2.5, max: 8, step: 0.25, value: 3, fmt: (v) => v.toFixed(2) + '%', onChange: (v) => { st.thr = v; draw(); } });
        const winSel = ctx.select({ label: 'averaged over the last', value: '1', options: [{ value: '1', label: '5 minutes' }, { value: '6', label: '30 minutes' }, { value: '24', label: '2 hours' }, { value: '144', label: '12 hours' }], onChange: (v) => { st.k = +v; draw(); } });
        draw();
        return ctx.figure(cv,
          'Two weeks of a seeded service handling 200 requests every five minutes. On a normal day about 2% of requests fail, and that rate wobbles from one five minutes to the next by luck alone. On day 11 a real fault doubles it to 4% and stays. The blue band is the failure rate averaged over the window you choose. A false alarm is each time the band rises across the alert line before the fault began.',
          [thrSl, winSel], ro);
      }

      /* ================================================================== */
      /* 6. Drift: the world moves under a frozen model                       */
      /* ================================================================== */
      function driftLab() {
        const WEEKS = 52, PER = 1000, SEP = 3.3, DRIFT = 0.08, JUMP = 1.0, JUMPWEEK = 30, RETRAIN = 5000, RATE = 1000, FRESH = 95;
        const [cv, g] = ctx.canvas(720, 244);
        const [cd, gd] = ctx.canvas(720, 132);
        const r = mulberry32(52), gs = gaussFrom(r);
        const Z = [];
        for (let w = 0; w <= WEEKS; w++) { const a = new Float32Array(2 * PER); for (let i = 0; i < 2 * PER; i++) a[i] = gs(); Z.push(a); }
        const wig = []; { let a = 0; for (let w = 0; w <= WEEKS; w++) { wig.push(a); a += 0.03 * gs(); } }
        const delta = (w) => DRIFT * w + (w >= JUMPWEEK ? JUMP : 0) + wig[w];
        const xAt = (w, i) => (i < PER ? 0 : SEP) + delta(w) + Z[w][i];   /* first half ordinary mail, second half spam */
        function fit(w) { let s = 0; for (let i = 0; i < 2 * PER; i++) s += xAt(w, i); return s / (2 * PER); }   /* midpoint of the two class means */
        function evalWeek(w, th) { let ok = 0, fl = 0; for (let i = 0; i < 2 * PER; i++) { const f = xAt(w, i) > th; if (f) fl++; if (f === (i >= PER)) ok++; } return [ok / (2 * PER) * 100, fl / (2 * PER) * 100]; }
        let S;
        function reset() { const th = fit(0), e = evalWeek(0, th); S = { week: 0, th, acc: [e[0]], flag: [e[1]], retrains: [], mistakes: 0 }; }
        function advance(n) {
          for (let k = 0; k < n && S.week < WEEKS; k++) {
            S.week++;
            const e = evalWeek(S.week, S.th);
            S.acc.push(e[0]); S.flag.push(e[1]);
            S.mistakes += Math.max(0, FRESH - e[0]) * RATE;
          }
        }
        function retrain() {
          if (S.week >= WEEKS || S.retrains[S.retrains.length - 1] === S.week) return;
          S.th = fit(S.week); S.retrains.push(S.week);
        }
        function policy(every) { reset(); for (let w = 1; w <= WEEKS; w++) { advance(1); if (every && w % every === 0) retrain(); } }
        reset();
        const ro = ctx.readout();
        const fr = { x: 50, y: 30, w: 650, h: 160 }, fd = { x: 50, y: 30, w: 650, h: 78 };
        function draw() {
          const X = (w) => fr.x + 6 + w / WEEKS * (fr.w - 12), Y = (v) => fr.y + fr.h - (Math.max(40, Math.min(100, v)) - 40) / 60 * fr.h;
          g.clearRect(0, 0, 720, 244);
          box(g, fr);
          text(g, 'one year of a frozen spam filter', fr.x, 19, C.text, 'left', BOLD);
          text(g, S.week >= WEEKS ? 'year over' : 'week ' + S.week + ' of 52', fr.x + fr.w, 19, C.muted, 'right');
          clipTo(g, fr, () => {
            g.strokeStyle = C.accent; g.setLineDash([4, 4]); g.lineWidth = 1.3;
            S.retrains.forEach((w) => { g.beginPath(); g.moveTo(X(w), fr.y); g.lineTo(X(w), fr.y + fr.h); g.stroke(); });
            g.setLineDash([]);
            const line = (arr, col) => { g.strokeStyle = col; g.lineWidth = 2.2; g.beginPath(); arr.forEach((v, w) => { if (!w) g.moveTo(X(w), Y(v)); else g.lineTo(X(w), Y(v)); }); g.stroke(); const n = arr.length - 1; g.beginPath(); g.arc(X(n), Y(arr[n]), 4.5, 0, Math.PI * 2); g.fillStyle = col; g.fill(); };
            line(S.flag, C.warn); line(S.acc, C.green);
          });
          text(g, '40%', fr.x - 6, fr.y + fr.h, C.muted, 'right');
          text(g, '70%', fr.x - 6, Y(70) + 4, C.muted, 'right');
          text(g, '100%', fr.x - 6, fr.y + 10, C.muted, 'right');
          [0, 13, 26, 39, 52].forEach((w) => text(g, 'week ' + w, X(w), fr.y + fr.h + 17, C.muted, w === 0 ? 'left' : w === 52 ? 'right' : 'center'));
          text(g, 'green = accuracy (needs labels)   yellow = share flagged as spam (free)   blue dashes = a retrain', fr.x, fr.y + fr.h + 38, C.muted, 'left');
          /* this week's mail */
          gd.clearRect(0, 0, 720, 132);
          box(gd, fd);
          text(gd, 'this week\'s mail by spam score', fd.x, 19, C.text, 'left', BOLD);
          text(gd, 'blue = ordinary   red = spam   white = the model\'s line', fd.x + fd.w, 19, C.muted, 'right');
          const LO = -4, HI = 12, NB = 80, XS = (v) => fd.x + (v - LO) / (HI - LO) * fd.w;
          const ha = new Array(NB).fill(0), hb = new Array(NB).fill(0);
          for (let i = 0; i < 2 * PER; i++) { const b = Math.floor((xAt(S.week, i) - LO) / (HI - LO) * NB); if (b >= 0 && b < NB) (i < PER ? ha : hb)[b]++; }
          clipTo(gd, fd, () => {
            const bw = fd.w / NB;
            for (let b = 0; b < NB; b++) {
              gd.fillStyle = 'rgba(124,156,255,0.7)'; gd.fillRect(fd.x + b * bw, fd.y + fd.h - ha[b] / 95 * fd.h, bw - 1, ha[b] / 95 * fd.h);
              gd.fillStyle = 'rgba(251,113,133,0.7)'; gd.fillRect(fd.x + b * bw, fd.y + fd.h - hb[b] / 95 * fd.h, bw - 1, hb[b] / 95 * fd.h);
            }
            gd.strokeStyle = '#ffffff'; gd.lineWidth = 2; gd.beginPath(); gd.moveTo(XS(S.th), fd.y); gd.lineTo(XS(S.th), fd.y + fd.h); gd.stroke();
          });
          text(gd, 'looks ordinary', fd.x, fd.y + fd.h + 17, C.muted, 'left');
          text(gd, 'looks like spam', fd.x + fd.w, fd.y + fd.h + 17, C.muted, 'right');
          const n = S.acc.length - 1;
          ro.set({
            week: S.week + ' of 52',
            'accuracy this week': S.acc[n].toFixed(1) + '%',
            'flagged as spam': S.flag[n].toFixed(1) + '%',
            retrains: S.retrains.length,
            'cost of mistakes': money(S.mistakes),
            'total spent': money(S.mistakes + S.retrains.length * RETRAIN),
          });
        }
        const weekBtn = ctx.button('Next week', () => { advance(1); draw(); });
        const monthBtn = ctx.button('Next 4 weeks', () => { advance(4); draw(); }, 'primary');
        const retrainBtn = ctx.button('Retrain now ($5,000)', () => { retrain(); draw(); });
        const restartBtn = ctx.button('Restart the year', () => { reset(); draw(); });
        const neverBtn = ctx.button('Never retrain', () => { policy(0); draw(); });
        const fourBtn = ctx.button('Retrain every 4 weeks', () => { policy(4); draw(); });
        draw();
        return ctx.figure(h('div', {}, cv, cd),
          'A spam filter trained once, at week 0, on 2,000 seeded emails a week. Each email has a single spam score; the model flags everything to the right of its white line. Week by week the mail itself changes: scores creep upward as language and products shift, and at one point in the year something changes overnight. Retraining refits the line to the latest week and takes effect the week after. Illustrative money: $5,000 a retrain, and $1,000 a week for every point of accuracy below a fresh model\'s 95%. Simplified to one score, and sped up: real drift happens across thousands of features at once, and few filters fall this far in a year.',
          [weekBtn, monthBtn, retrainBtn, restartBtn, neverBtn, fourBtn], ro);
      }

      /* ================================================================== */
      /* 7. Canary: ship to a few first                                       */
      /* ================================================================== */
      function canaryLab() {
        const W = 720, H = 250, MIN = 120, PERMIN = 1000, POLD = 0.02;
        const [cv, g] = ctx.canvas(W, H);
        const r = mulberry32(5);
        const U = new Float32Array(MIN * PERMIN);
        for (let i = 0; i < U.length; i++) U[i] = r();
        let histErr = 0;
        for (let i = 0; i < 60 * PERMIN; i++) if (r() < POLD) histErr++;
        const st = { share: 100, check: 30, pnew: 0.12 };
        const ro = ctx.readout();
        const fa = { x: 60, y: 30, w: 400, h: 176 }, fb = { x: 540, y: 30, w: 160, h: 176 };
        function damage(share, upto) {   /* cumulative extra wrong answers per minute */
          const k = Math.round(share * PERMIN / 100), out = new Float32Array(MIN + 1);
          let extra = 0, nErr = 0, oErr = 0;
          for (let m = 0; m < MIN; m++) {
            for (let j = 0; j < PERMIN; j++) {
              const v = U[m * PERMIN + j];
              if (j < k) { if (v < st.pnew) { if (m < upto) nErr++; if (v >= POLD) extra++; } } else if (v < POLD && m < upto) oErr++;
            }
            out[m + 1] = extra;
          }
          return { out, nErr, oErr, nN: k * upto, oN: (PERMIN - k) * upto };
        }
        function draw() {
          const mine = damage(st.share, st.check), all = st.share === 100 ? mine : damage(100, st.check);
          const ymax = all.out[MIN];
          const X = (m) => fa.x + m / MIN * fa.w, Y = (v) => fa.y + fa.h - v / ymax * (fa.h - 8);
          g.clearRect(0, 0, W, H);
          box(g, fa); box(g, fb);
          text(g, 'extra wrong answers caused by the release', fa.x, 19, C.text, 'left', BOLD);
          clipTo(g, fa, () => {
            const path = (arr, col, dash) => { g.strokeStyle = col; g.lineWidth = 2; g.setLineDash(dash || []); g.beginPath(); for (let m = 0; m <= MIN; m++) { if (!m) g.moveTo(X(m), Y(arr[m])); else g.lineTo(X(m), Y(arr[m])); } g.stroke(); g.setLineDash([]); };
            path(all.out, C.muted, [5, 4]);
            path(mine.out, C.danger);
            g.strokeStyle = '#ffffff'; g.lineWidth = 1.2; g.setLineDash([3, 4]); g.beginPath(); g.moveTo(X(st.check), fa.y); g.lineTo(X(st.check), fa.y + fa.h); g.stroke(); g.setLineDash([]);
            g.beginPath(); g.arc(X(st.check), Y(mine.out[st.check]), 5.5, 0, Math.PI * 2); g.fillStyle = C.danger; g.fill();
          });
          text(g, '0', fa.x - 6, fa.y + fa.h, C.muted, 'right');
          text(g, num(ymax), fa.x - 6, fa.y + 12, C.muted, 'right');
          [0, 30, 60, 90, 120].forEach((m) => text(g, m === 120 ? '120 min' : String(m), X(m), fa.y + fa.h + 17, C.muted, m === 0 ? 'left' : m === 120 ? 'right' : 'center'));
          text(g, 'red = your plan   grey dashes = everyone at once   white = you look', fa.x, fa.y + fa.h + 36, C.muted, 'left');
          /* dashboard */
          const oN = st.share === 100 ? 60 * PERMIN : mine.oN, oE = st.share === 100 ? histErr : mine.oErr;
          const po = oE / oN, pn = mine.nErr / mine.nN;
          const seO = Math.sqrt(po * (1 - po) / oN), seN = Math.sqrt(Math.max(pn, 0.001) * (1 - pn) / mine.nN);
          /* bars are three standard errors either side; the verdict is "clearly worse" only when the two bars do not touch */
          const BAR = 3, clear = pn - BAR * seN > po + BAR * seO;
          text(g, 'the dashboard then', fb.x, 19, C.text, 'left', BOLD);
          const YM = st.pnew > 0.05 ? 16 : 6, YB = (v) => fb.y + fb.h - Math.min(YM, v * 100) / YM * (fb.h - 8);
          clipTo(g, fb, () => {
            [[po, seO, fb.x + 45, C.muted], [pn, seN, fb.x + 115, clear ? C.danger : C.warn]].forEach(([pv, se, x, col]) => {
              g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); g.moveTo(x, YB(pv + BAR * se)); g.lineTo(x, YB(Math.max(0, pv - BAR * se))); g.moveTo(x - 8, YB(pv + BAR * se)); g.lineTo(x + 8, YB(pv + BAR * se)); g.moveTo(x - 8, YB(Math.max(0, pv - BAR * se))); g.lineTo(x + 8, YB(Math.max(0, pv - BAR * se))); g.stroke();
              g.beginPath(); g.arc(x, YB(pv), 5.5, 0, Math.PI * 2); g.fillStyle = col; g.fill();
            });
          });
          text(g, '0%', fb.x - 6, fb.y + fb.h, C.muted, 'right');
          text(g, YM + '%', fb.x - 6, fb.y + 12, C.muted, 'right');
          text(g, 'old model', fb.x + 45, fb.y + fb.h + 17, C.muted, 'center');
          text(g, 'new model', fb.x + 115, fb.y + fb.h + 17, C.muted, 'center');
          text(g, 'bars = the range luck allows', fb.x + fb.w, fb.y + fb.h + 36, C.muted, 'right');
          ro.set({
            'extra wrong answers by then': num(mine.out[st.check]),
            'old model wrong': (po * 100).toFixed(1) + '%',
            'new model wrong': (pn * 100).toFixed(1) + '% of ' + num(mine.nN),
            verdict: clear ? 'clearly worse: roll back' : 'inside the noise: cannot tell yet',
          });
        }
        const shareSl = ctx.slider({ label: 'traffic sent to the new model', min: 1, max: 100, step: 1, value: 100, fmt: (v) => v + '%', onChange: (v) => { st.share = v; draw(); } });
        const checkSl = ctx.slider({ label: 'minutes until someone looks', min: 5, max: 120, step: 5, value: 30, onChange: (v) => { st.check = v; draw(); } });
        const badSel = ctx.select({ label: 'the new model is', value: '0.12', options: [{ value: '0.12', label: 'badly broken: 12% wrong' }, { value: '0.03', label: 'subtly worse: 3% wrong' }], onChange: (v) => { st.pnew = +v; draw(); } });
        const allBtn = ctx.button('Ship to everyone', () => { st.share = 100; shareSl.value = 100; draw(); });
        const canBtn = ctx.button('Canary: 5% first', () => { st.share = 5; shareSl.value = 5; draw(); }, 'primary');
        draw();
        return ctx.figure(cv,
          'A seeded service taking 1,000 requests a minute. The old model gets 2% of them wrong; the release you are shipping is worse, and nobody knows yet. Left: wrong answers that would not have happened without the release, piling up until someone looks and rolls back. Right: what the dashboard shows at that moment, each dot with the range that luck alone could explain (three standard errors either side). The verdict says "clearly worse" only when the two ranges do not touch. A canary\'s old-model figure comes from the rest of the live traffic; when everyone is on the new model it comes from the hour before the release.',
          [shareSl, checkSl, badSel, allBtn, canBtn], ro);
      }

      /* ================================================================== */
      /* 8. Timeouts, retries and a backup                                    */
      /* ================================================================== */
      function fallbackLab() {
        const W = 720, H = 240, N = 10000, TRIES = 4;
        const [cv, g] = ctx.canvas(W, H);
        const r = mulberry32(88), gs = gaussFrom(r);
        const uf = new Float32Array(N * TRIES), uh = new Float32Array(N * TRIES), zl = new Float32Array(N * TRIES);
        for (let i = 0; i < N * TRIES; i++) { uf[i] = r(); uh[i] = r(); zl[i] = gs(); }
        const st = { scen: 'normal', timeout: 60, retries: 0, backup: 'error' };
        const ro = ctx.readout();
        function calc() {
          const fail = st.scen === 'normal' ? 0.03 : 0.6, hang = st.scen === 'normal' ? 0.02 : 0.05;
          let first = 0, later = 0, backup = 0, error = 0, calls = 0;
          const waits = new Float32Array(N);
          for (let i = 0; i < N; i++) {
            let t = 0, done = -1;
            for (let k = 0; k <= st.retries; k++) {
              const j = i * TRIES + k;
              calls++;
              if (uf[j] < fail) { t += 0.3; continue; }                      /* refused quickly */
              const took = uh[j] < hang ? 45 : 1.2 * Math.exp(0.4 * zl[j]);   /* a hung call answers after 45 s */
              if (took > st.timeout) { t += st.timeout; continue; }
              t += took; done = k; break;
            }
            if (done === 0) first++; else if (done > 0) later++; else if (st.backup === 'backup') { backup++; t += 0.4; } else error++;
            waits[i] = t;
          }
          waits.sort();
          return { first, later, backup, error, mult: calls / N, p50: waits[N / 2], p99: waits[Math.floor(N * 0.99)] };
        }
        function draw() {
          const R = calc(), bx = 20, bw = 680;
          g.clearRect(0, 0, W, H);
          text(g, 'what happened to 10,000 requests', bx, 19, C.text, 'left', BOLD);
          const parts = [['first try', R.first, C.green], ['after a retry', R.later, C.accent], ['backup answer', R.backup, C.warn], ['error shown', R.error, C.danger]];
          let x = bx;
          g.fillStyle = PANEL; g.fillRect(bx, 30, bw, 28);
          parts.forEach(([, v, col]) => { const w = v / N * bw; g.fillStyle = col; g.fillRect(x, 30, w, 28); x += w; });
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(bx, 30, bw, 28);
          parts.forEach(([name, v, col], i) => { const lx = bx + i * 172; g.fillStyle = col; g.fillRect(lx, 68, 11, 11); text(g, name + ' ' + num(v), lx + 17, 78, C.text, 'left'); });
          text(g, 'calls sent to the provider: ' + R.mult.toFixed(2) + ' for every request', bx, 113, C.text, 'left', BOLD);
          g.fillStyle = PANEL; g.fillRect(bx, 122, bw, 18);
          g.fillStyle = R.mult > 1.5 ? C.danger : C.accent; g.fillRect(bx, 122, R.mult / 4 * bw, 18);
          g.strokeStyle = C.line; g.strokeRect(bx, 122, bw, 18);
          g.strokeStyle = '#ffffff'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(bx + bw / 4, 119); g.lineTo(bx + bw / 4, 143); g.stroke();
          text(g, 'how long users waited: half were done within ' + secs(R.p50) + ', the slowest 1% waited ' + secs(R.p99) + ' or more', bx, 175, C.text, 'left', BOLD);
          const LX = (v) => bx + (Math.log10(Math.max(0.1, Math.min(200, v))) + 1) / 3.301 * bw;
          g.fillStyle = PANEL; g.fillRect(bx, 184, bw, 18);
          g.fillStyle = 'rgba(148,163,184,0.35)'; g.fillRect(LX(R.p50), 184, Math.max(2, LX(R.p99) - LX(R.p50)), 18);
          g.fillStyle = C.green; g.fillRect(LX(R.p50) - 2, 181, 4, 24);
          g.fillStyle = C.orange; g.fillRect(LX(R.p99) - 2, 181, 4, 24);
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(bx, 184, bw, 18);
          [[0.1, '0.1 s', 'left'], [1, '1 s', 'center'], [10, '10 s', 'center'], [100, '100 s', 'center']].forEach(([v, s, al]) => text(g, s, LX(v), 222, C.muted, al));
          text(g, 'green = p50   orange = p99', bx + bw, 236, C.muted, 'right');
          ro.set({
            'users shown an error': (R.error / N * 100).toFixed(1) + '%',
            'answered by the backup': (R.backup / N * 100).toFixed(1) + '%',
            'p99 wait': secs(R.p99),
            'calls per request': R.mult.toFixed(2),
          });
        }
        const scenSel = ctx.select({ label: 'the provider is having', value: 'normal', options: [{ value: 'normal', label: 'a normal day' }, { value: 'outage', label: 'an outage: 60% of calls fail' }], onChange: (v) => { st.scen = v; draw(); } });
        const toSl = ctx.slider({ label: 'give up on a call after', min: 1, max: 60, step: 1, value: 60, fmt: (v) => v + ' s', onChange: (v) => { st.timeout = v; draw(); } });
        const reSl = ctx.slider({ label: 'retries', min: 0, max: 3, step: 1, value: 0, onChange: (v) => { st.retries = v; draw(); } });
        const backSel = ctx.select({ label: 'if every try fails', value: 'error', options: [{ value: 'error', label: 'show the user an error' }, { value: 'backup', label: 'answer with a small backup model' }], onChange: (v) => { st.backup = v; draw(); } });
        draw();
        return ctx.figure(cv,
          'Ten thousand seeded requests to a model you rent from a provider. On a normal day 3% of calls are refused at once and 2% hang for 45 seconds before answering; the rest reply in about a second. During the outage 60% are refused and 5% hang. A retry here is sent immediately, which is the naive version: real clients wait a little longer before each new try. The white mark on the middle bar is one call per request.',
          [scenSel, toSl, reSl, backSel], ro);
      }

      /* ================================================================== */
      /* The chapter: touch first, read second.                               */
      /* ================================================================== */
      root.append(
        p(`Your model passed its evaluation. Now strangers will send it requests, at moments of their choosing, and expect an answer before they lose patience. Start by putting one server in front of them.`),
        callout('tryit', '🖐 Do this first: fill a server until it chokes',
          `One server, 100 ms of work per request on average. It can clear 10 a second.<br>
           <b>1.</b> It starts with 5 arriving every second: half its capacity. Read <b>average wait</b>: 99 ms, about one request's worth.<br>
           <b>2.</b> Drag <b>requests arriving per second</b> to 8, then 9, then 9.5. You added 5% more traffic on that last move. Look at what the wait did.<br>
           <b>3.</b> Drag it to 12. The server is now asked for more than it can do. The top chart climbs and never comes back down, and the wait has no limit.<br>
           <b>4.</b> Stay at 12 and set <b>servers</b> to 2. The average wait drops to 56 ms.`),
        queueSim(),
        p(`Nothing broke at 9.5. No bug, no crash, no slow code. The server was doing the same 100 ms of work per request as before. It was simply busy 95% of the time, and requests kept landing while it was in the middle of someone else's.`),
      );

      root.append(section('Why a busy server makes everyone wait',
        p(`The fraction of time a server spends working is its <em>utilisation</em>. Arrivals are lumpy: three requests land together, then nothing for a moment. A server with spare time absorbs a clump and catches up in the gap. A server with no spare time never catches up, so every clump adds to a queue that the next clump finds waiting.`),
        p(`You watched that turn into numbers. The rule behind them is short enough to carry in your head, and you can check every step of it against the readout above.`),
        ctx.walkthrough([
          { say: 'One request takes 100 ms of work on average, so one server working flat out clears 10 a second.', math: '1,000 ms &divide; 100 ms = 10 requests a second' },
          { say: 'With 5 arriving a second, the server is busy half the time and idle the other half.', math: '5 &divide; 10 = 0.5 busy &nbsp;&nbsp; 1 &minus; 0.5 = 0.5 idle', note: 'The demo printed this as <b>how busy: 50%</b>.' },
          { say: 'Divide the busy share by the idle share and you get 1: the average request waits one job\'s worth, 100 ms.', math: '0.5 &divide; 0.5 = 1 &nbsp;&rarr;&nbsp; 1 &times; 100 ms = 100 ms', note: 'The simulator printed 99 ms. Why this division is the right one takes a page of probability. Here you test it against the simulator instead.' },
          { say: 'At 8 a second the server is busy 0.8 of the time and idle 0.2, so the same division gives 4 jobs of waiting.', math: '0.8 &divide; 0.2 = 4 &nbsp;&rarr;&nbsp; 400 ms', note: 'The simulator printed 396 ms.' },
          { ask: 'At <b>9</b> a second the server is busy 0.9 of the time. Before you look back at the demo: what should the average wait be?',
            options: ['About 180 ms', 'About 450 ms', 'About 900 ms'], answer: 2,
            explain: '0.9 ÷ 0.1 = 9 jobs, and 9 × 100 ms = 900 ms. The simulator printed 903 ms.' },
          { say: 'So the whole rule is: busy share, divided by idle share, times the length of one job.', math: 'wait = busy &divide; idle &times; one job' },
          { say: 'Engineers write the busy share as the Greek letter <b>&rho;</b> (rho), which makes the idle share 1 &minus; &rho;.', math: 'wait = &rho; &divide; (1 &minus; &rho;) &times; one job' },
          { say: 'The idle share sits on the bottom of that division, so as it shrinks toward zero the wait has no ceiling.', math: '&rho; = 0.95 &rarr; 19 jobs &nbsp;&nbsp; &rho; = 0.99 &rarr; 99 jobs' },
          { ask: 'Traffic grows from 9 to 9.5 a second, about 5% more. What happens to the average wait?',
            options: ['It roughly doubles', 'It rises about 5%', 'It rises about 50%'], answer: 0,
            explain: '0.9 ÷ 0.1 = 9 jobs becomes 0.95 ÷ 0.05 = 19 jobs: 900 ms to 1,900 ms. The simulator printed 1,950 ms. It is off by more than before because a queue this jumpy needs far more than 400,000 requests to settle.' },
          { say: 'The exact formula belongs to one server with random arrivals and random job lengths; with more servers the numbers change and the hockey stick stays.', note: 'That is the blue curve under the demo, which you redrew when you set servers to 2.' },
          { ask: 'One server, 100 ms jobs. Your team promises an average wait of at most 100 ms. How busy may the server get?',
            options: ['90% busy', '50% busy', '99% busy'], answer: 1,
            explain: 'A wait of one job means ρ ÷ (1 − ρ) = 1, which happens at ρ = 0.5. Half the machine is the price of a short queue.' },
        ], {
          title: 'From a choking server to ρ ÷ (1 − ρ)',
          recap: 'the average wait is &rho; &divide; (1 &minus; &rho;) jobs long, so the last few points of utilisation cost more than all the others put together.',
        }),
        ctx.decoder([
          { sym: 'wait', name: 'the average time a request spends queueing', says: 'Time standing in line before any work starts on it. The work itself comes on top.', points: 'the <b>average wait</b> readout, and the height of the dot on the lower chart.' },
          '=',
          { sym: '&rho;', name: 'rho', says: 'Utilisation: the share of time the server is busy. Arrivals per second divided by what the server can clear per second.', points: 'the <b>how busy</b> readout: 5 arriving ÷ 10 possible = 50%.' },
          '÷',
          { sym: '(1 &minus; &rho;)', name: 'one minus rho', says: 'The idle share: the slack the server has for catching up after a clump of arrivals. Halve the slack and the wait more than doubles.', points: 'the shrinking gap between your dot and the yellow dashed line.' },
          '×',
          { sym: 'S', name: 'the service time', says: 'How long one job takes once the server starts on it. Faster work helps twice: each job in the queue is shorter, and the server is less busy, so ρ falls too.', points: 'the 100 ms each request needed.' },
        ], {
          title: 'wait  =  ρ ÷ (1 − ρ)  ×  S',
          plain: 'the average wait is the busy share over the idle share, times the length of one job.',
        }),
        p(`This is why well-run services look wasteful. A team that keeps its servers 60% busy is paying for the slack that absorbs a clump. Add servers when traffic climbs and remove them when it falls, and you have <em>autoscaling</em>, which is the same headroom rule applied by a program every few minutes.`),
        callout('history', '📜 1909: a telephone engineer works out how many lines a city needs',
          `Agner Krarup Erlang, a mathematician at the Copenhagen Telephone Company, published his first paper on telephone traffic in 1909, and in 1917 the formulas for how often a caller finds every line busy and how long they wait. Calls arrive at random and last a random time: the same problem as the server above. Engineers have sized exchanges, call centres and server fleets with his results ever since, and the unit of traffic load is named the erlang.`),
        callout('example', '🌍 You have stood in this queue',
          `A supermarket with every till staffed and a queue at each is running near 100% utilisation: one slow customer and the line reaches the freezers. A motorway that flows freely with room to spare can lock solid when a little more traffic joins, with no accident needed. A chat assistant that answers at once at 3 a.m. and makes you wait at 3 p.m. is the same curve.`),
      ));

      root.append(section('The average is the wrong number to watch',
        p(`The readout above carried a second figure, <b>p99 wait</b>, that was always several times the average. That gap matters more than it looks.`),
        callout('tryit', '🖐 Try this: break a service without moving its average much',
          `<b>1.</b> Read the four numbers on the right. On a healthy day they sit close together, from 179 ms to 409 ms.<br>
           <b>2.</b> Press <b>Cold cache: 2% slow</b>. One request in fifty now takes about four seconds.<br>
           <b>3.</b> The <b>average</b> moved from 191 ms to 277 ms. A dashboard showing only that would stay green. Now read <b>p99</b>.<br>
           <b>4.</b> Read the last number in the readout: the share of 20-request visits that met a slow request.<br>
           <b>5.</b> Drag <b>requests in one visit</b> up to 40 and watch that share climb.`),
        tailDemo(),
        p(`Sort every request from fastest to slowest and walk along the line. The time you find halfway is the 50th <em>percentile</em>, written p50, also called the median. Ninety-nine hundredths of the way along is <b>p99</b>: one request in a hundred is slower than that. The slow end of the line is the <em>tail</em>.`),
        p(`An average blends the tail into the crowd. Two percent of requests at four seconds added 86 ms to the average and multiplied p99 by ten, from 409 ms to 4,148 ms. The median barely moved, from 179 ms to 181 ms, because the typical request was untouched. Each number answers a different question, and "is anybody having a terrible time?" is answered only by the high percentiles.`),
        p(`The visit counter is the reason the tail is everyone's problem. Loading one screen of a real product can take twenty calls. If each has a 2% chance of being slow, the chance that all twenty are fast is 0.98 multiplied by itself twenty times, which is 0.67. A third of visits should hit the tail, and the demo's count came out at 36%.`),
        p(`So a promise about speed is written as a percentile: "p95 under 800 ms", say. That promise is a <em>latency budget</em>, and it gets spent like money. If the model takes 500 ms, then the network, the database lookup, the safety check and the queue must share the remaining 300.`),
        callout('example', '🌍 Why chat replies appear word by word',
          `A long reply can take many seconds to finish, and chapter 12 showed why: decode produces one token at a time. Products stream the tokens as they are made, so the wait you feel is the time to the <b>first</b> token, which is short. Teams that run chat products track that number and the tokens per second after it as two separate budgets.`),
      ));

      root.append(section('Batching: trading a little waiting for a lot of throughput',
        p(`A GPU is a wide machine. Chapter 3 used that to train on mini-batches; a server uses it to answer several people in one pass. How many it answers per second is its <em>throughput</em>.`),
        callout('tryit', '🖐 Try this: find the batch size that fits the budget',
          `Traffic is 100 requests a second, and your budget is 300 ms per request (the yellow line).<br>
           <b>1.</b> At <b>batch size</b> 1, read <b>GPUs needed</b> and <b>bill per hour</b>: 10 GPUs and $20.<br>
           <b>2.</b> Drag <b>batch size</b> up slowly. The bill collapses while the blue curve creeps toward the yellow line. Find the largest batch that still reads <b>met</b>.<br>
           <b>3.</b> Keep going to 64. Cheap, and far too slow.<br>
           <b>4.</b> Go back to batch size 16, then drag <b>requests arriving per second</b> down to 10. The same setting that was fine at lunchtime now misses the budget at night.`),
        batchLab(),
        p(`One pass over sixteen requests takes under three times as long as one pass over one (188 ms against 68), so the cost per request falls steeply. That is <em>batching</em>. The price is paid in time: every request waits for its batch to fill, then for the whole batch to finish. Throughput and latency pull against each other, and the batch size is where you choose.`),
        p(`The night-time failure is the instructive one. With ten requests a second, a batch of sixteen takes a second and a half to fill, and the first arrival sits there the whole time. Real servers use <b>dynamic batching</b>: take whatever has arrived within a few milliseconds and go. Language models add a twist called continuous batching, where a new request joins a batch that is already generating, between one token and the next.`),
      ));

      root.append(section('The bill, and three ways to shrink it',
        p(`Chapter 12 priced a single call: tokens in, tokens out. A product makes that call tens of thousands of times a day, and most of those calls are not equally hard, not equally new, and not equally distributed among users.`),
        callout('tryit', '🖐 Try this: get the bill under $600 a day without wrecking the answers',
          `With no controls, every request goes to the large model, and the day costs $1,674.<br>
           <b>1.</b> Drag <b>sent to the small model</b> to 60%. Easy questions now go to a model a tenth of the price. Read the bill and <b>right answers</b>.<br>
           <b>2.</b> Keep dragging to 100%. The bill is tiny. Read <b>right answers</b> again: you are now sending hard questions to a model that cannot do them.<br>
           <b>3.</b> Return to 60% and drag <b>answers served from a cache</b> to 30%.<br>
           <b>4.</b> Choose a <b>daily limit per user</b> of <b>100 requests</b>. Read how many users it touched and how much traffic it removed.`),
        costLab(),
        p(`Three different tools. A <em>response cache</em> stores an answer and hands it to the next person who asks the same thing: no model call, a few milliseconds, zero cost. It is blunter than chapter 12's prompt caching, which reuses work on a shared prefix. Its risk is staleness, and for anything personal, handing one user's answer to another.`),
        p(`A <em>router</em> looks at each request and picks a model for it: a small, cheap one for "what are your opening hours", the large one for the tangled complaint. You saw both its promise and its cliff. Up to the share of traffic that really is easy, it costs about a point of accuracy. Past that, each further saving is bought with wrong answers.`),
        p(`A <em>rate limit</em> caps what one user or one key may send in a period. In the demo, a limit of 100 requests a day touched 78 users out of 5,000 and removed 25,368 requests, nearly a third of all traffic. That lopsidedness is normal: a few heavy users, a runaway script or an attacker can be most of a bill. A limit is also what keeps one customer's loop from becoming every customer's queue.`),
        callout('warning', '⚠️ Savings are measured, not assumed',
          `Each of these changes what users receive. A router that sends 5% of hard questions to the small model looks free on the invoice and shows up as complaints. Before any of them ships, run the evaluation set from chapter 19 on evaluation through the cheaper path and compare.`),
      ));

      root.append(section('Monitoring: knowing before your users tell you',
        p(`A deployed model fails quietly. It does not crash; it returns a confident wrong answer with status "OK". So you collect numbers continuously and let a program wake someone when they move. The hard part is deciding what "moved" means.`),
        callout('tryit', '🖐 Try this: set an alert you could live with',
          `The alert starts at 3% on five-minute averages.<br>
           <b>1.</b> Read <b>false alarms in ten quiet days</b>: 287. Nobody answers a pager that cries wolf that often.<br>
           <b>2.</b> Drag <b>page someone when failures pass</b> up until the false alarms reach zero, at 6%. The real fault now takes 55 minutes to catch.<br>
           <b>3.</b> Bring the line down to 4% and choose <b>averaged over the last</b> <b>30 minutes</b>. Still no false alarms, and the fault is caught in 30 minutes.<br>
           <b>4.</b> Choose <b>2 hours</b> and bring the line down to 3%, where you started. Still quiet. The fault is caught in 50 minutes: a lower line than step 3, and slower.<br>
           <b>5.</b> Leave the line at 3% and choose <b>12 hours</b>. Quiet, and the fault burns for more than four hours before anyone is told.`),
        alertLab(),
        p(`Chapter 3's rule is at work here: the noise in an average shrinks with the square root of how many things you averaged. Five minutes of traffic is 200 requests, and luck alone swings its failure rate by a full point. Two hours is 4,800 requests and the swing is a fifth of that, so a line at 3% sits far above the noise and still well below the fault. That is why step 4 stayed quiet at the same 3% that paged 287 times in step 1.`),
        p(`A longer window buys calm and pays in delay, because a new fault must fill enough of the window to drag the average over the line. Every <em>alert</em> is a choice on that trade. Teams usually keep two: a fast, high line for disasters, and a slow, low one for the quiet rot.`),
        p(`What to log for every request: the input, the output, which model version answered, how long it took, what it cost, and what the user did next. The last item is the valuable one. Nobody labels production traffic as it arrives, so quality has to be read from behaviour.`),
        cards([
          { title: 'Signals that need no labels', body: 'Did the user rephrase and ask again? Copy the answer, or delete it? Click thumbs-down? Abandon the chat? Ask for a human? How often does the model refuse, or answer far longer or shorter than usual?' },
          { title: 'Signals that need labels', body: 'Was the answer right? Sample a few hundred real requests a week, have people grade them against a written guide, and you have a slow, trustworthy accuracy number to hold the fast ones against.' },
        ]),
      ));

      root.append(section('Drift: the world moves and the weights do not',
        p(`Training froze your model at a moment in time. Chapter 12 met one face of that: its knowledge stops on the day training did. There is a quieter one, and it affects every model ever deployed.`),
        callout('tryit', '🖐 Try this: run a spam filter for a year',
          `<b>1.</b> Press <b>Never retrain</b>. Watch the green accuracy line. By the end it is tossing a coin. Read <b>total spent</b>: $1,398,650.<br>
           <b>2.</b> Press <b>Retrain every 4 weeks</b>. Twelve retrains, a far better year: $116,250. Look at weeks 30 to 32, though. The schedule did not know something had happened.<br>
           <b>3.</b> Now beat it by hand. Press <b>Restart the year</b>, then step with <b>Next week</b> or <b>Next 4 weeks</b>, and press <b>Retrain now ($5,000)</b> when you judge it worth the money. A year under $100,000 is possible.<br>
           <b>4.</b> Play it once more watching only the <b>yellow</b> line. It needs no labels, and it tells you nearly everything.`),
        driftLab(),
        p(`The model never changed. Its line stayed exactly where training put it while the mail slid past underneath. That is <em>drift</em>: the data a model meets in production moving away from the data it was trained on. Spammers change their wording, a shop launches a product range, a new phone camera changes what photos look like, prices rise.`),
        p(`Drift comes at two speeds, and you met both. The slow creep rewards a regular schedule. The overnight jump at week 30 punishes one: a calendar retrains on the fourth Friday even when nothing has happened, and sits blind for three weeks when something does. A trigger tied to a monitored signal reacts to the world instead.`),
        p(`In the demo you could see accuracy every week. That was a luxury. Real labels arrive late: you learn that a loan defaulted months after approving it, and that an email was spam only if someone reports it. The yellow line is what you actually get in time.`),
        p(`The share of inputs a model flags, the spread of its confidence, the mix of topics coming in: all free and all immediate. Here the flagged share rose in step with the fall in accuracy. It will not catch every drift, because mail can change in ways that leave the flagged share alone, so the slow labelled sample keeps running beside it.`),
        p(`Retraining is not free either. It costs compute and people, and every retrained model is a new model that must pass evaluation and be released with the care of any other change. Which brings us to releasing.`),
      ));

      root.append(section('Releases, canaries and the rollback switch',
        p(`Sooner or later you will ship a model that is worse than the one it replaces. Your evaluation set missed something, or a preprocessing step changed, or the prompt lost a line. What matters is how many people meet it before you find out.`),
        callout('tryit', '🖐 Try this: ship the same bad model two ways',
          `The new model gets 12% of answers wrong, against 2% for the old one. Nobody will look at the dashboard for 30 minutes.<br>
           <b>1.</b> It starts on <b>Ship to everyone</b>. Read <b>extra wrong answers by then</b>: 3,029.<br>
           <b>2.</b> Press <b>Canary: 5% first</b>. Same model, same 30 minutes: 137. The dashboard on the right is just as damning.<br>
           <b>3.</b> Stay on the canary and switch <b>the new model is</b> to <b>subtly worse: 3% wrong</b>. Read the <b>verdict</b>. A small canary and a small regression: the bars overlap.<br>
           <b>4.</b> Drag <b>minutes until someone looks</b> up slowly. More data, narrower bars. The verdict turns at 100 minutes.`),
        canaryLab(),
        p(`A <em>canary release</em> sends a small share of traffic to the new version and compares it, live, against everyone else on the old one. The name comes from the canaries miners carried underground: the bird reacted to bad air before the miners did. The damage a bad release does is three things multiplied: share of traffic, extra error rate, and minutes until someone acts. The canary cut the first by twenty.`),
        p(`The subtle case shows the canary's cost. Less traffic means less evidence per minute, and a small difference hides in the noise for longer. So a canary is given a bake time: hours or a day at 5%, then 25%, then everyone, with a comparison at each step. Gross failures are caught in minutes; subtle ones are why the steps exist.`),
        p(`The third factor is shrunk by a <em>rollback</em>: one command that puts the previous version back. It only works if the previous model, its prompt and its configuration were kept, versioned together, and can be restored without a meeting. An <b>incident</b> run well reads: notice, roll back first, investigate second. Afterwards, write down what happened and add the case that slipped through to the evaluation set.`),
        callout('example', '🌍 You have been in a canary',
          `When an app shows a friend a new feature that you do not have yet, you are looking at a staged rollout. Phone operating systems, browsers and large websites release to a small percentage of users, watch crash and error rates for hours or days, and widen only if the numbers hold.`),
      ));

      root.append(section('When the model is wrong, slow or down',
        p(`Many products rent their model from a provider, over a network. That call will sometimes hang, fail, or return something you must not show a user. The design question is what happens next.`),
        callout('tryit', '🖐 Try this: survive a bad day at your provider',
          `<b>1.</b> A normal day, no protection. Read the readout: 3.0% of users see an error, and the <b>p99 wait</b> is 45 s.<br>
           <b>2.</b> Drag <b>give up on a call after</b> down to 10 s and set <b>retries</b> to 1. Errors nearly vanish, and the slowest 1% now wait about 11 s where they waited 45.<br>
           <b>3.</b> Now drag the timeout all the way down to 1 s. You are hanging up on healthy calls that needed a moment longer. Read <b>calls per request</b>. Put it back to 10 s.<br>
           <b>4.</b> Choose <b>an outage: 60% of calls fail</b> and set <b>retries</b> to 3. Fewer errors, but look at the middle bar: you are hammering a provider that is already on its knees.<br>
           <b>5.</b> Set <b>retries</b> back to 1 and choose <b>answer with a small backup model</b>.`),
        fallbackLab(),
        p(`A <em>timeout</em> is a decision to stop waiting. Without one, a hung call holds a user and a server slot for as long as it likes, and the slowest 1% of your users waited 45 seconds. Set it too tight and you cut off healthy calls, which you saw as errors and wasted work. It belongs just beyond the latency of a slow but normal reply.`),
        p(`A <em>retry</em> turns a rare failure into a rarer one: if 3% of calls fail, about one in a thousand fails twice. In an outage the same habit multiplies load on the system least able to bear it. Hence two rules: wait longer before each new try, with a random delay so clients do not retry in step, and stop after a few.`),
        p(`A <em>fallback</em> is the answer you give when the best one is unavailable: a smaller model, a cached reply, a plain "I can't do that right now, here is the help page". A worse answer delivered beats a perfect one that never arrives, provided the product is honest about which one the user got.`),
        p(`<em>Guardrails</em> are the checks around the model on a good day. Before the call: is this request in scope, is it an attack, does it contain data that must not leave the building? After it: is the output well-formed, does it leak anything, does it make a promise the company cannot keep? A failed check needs a designed path too.`),
        ul([
          `<b>A refusal path.</b> When the model declines or a check blocks the output, the user gets a clear message and a next step, never a blank box or a stack trace.`,
          `<b>Human escalation.</b> Low confidence, an angry user, a legal or medical question, a refund above a limit: hand the conversation to a person, with the transcript attached so nobody has to repeat themselves.`,
          `<b>Limits on agents.</b> Chapter 12's agent loop gets a ceiling on steps, spend and time, and anything irreversible (sending money, deleting data, emailing a customer) waits for a person to approve it.`,
        ]),
      ));

      root.append(section('Closing the loop: production feeds the next model',
        p(`Everything in this chapter produces data. The logs record what people really ask, which is never quite what the team imagined. The thumbs-downs, the rephrasings and the escalations mark where the model failed. The incident reports record failures nobody had thought to test for.`),
        p(`That stream flows back into the two chapters before this one. Failures sampled from production, labelled with care, become training data: chapter 18 on data. The cases that caused incidents become permanent entries in the test suite, so the same mistake cannot ship twice: chapter 19 on evaluation. The retrained model goes out through a canary, is watched by the same alerts, and starts producing logs of its own.`),
        p(`This cycle is the <em>feedback loop</em>, and teams that run it well improve faster than teams with better models and no loop. It needs care. Users only react to the answers they were shown, so the logs are biased toward what the current model already does. And real conversations contain names, addresses and secrets, so what is stored, for how long, and who may read it are decisions made on purpose, with the people whose data it is in mind.`),
        callout('key', '🔑 The system is the product',
          `The model is one box in a diagram that also contains a queue, a cache, a router, a set of checks, a dashboard, a pager, a rollback switch and a labelling team. Users experience the whole diagram. Most of what separates a demo from a product is in the boxes that are not the model.`),
      ));

      root.append(section('Why this matters for modern AI work',
        p(`Large language models make every section here sharper. A request's cost and duration depend on how many tokens come out, which nobody knows in advance, so latency tails are long by nature. One GPU serves many conversations at once, limited by the KV-cache memory that chapter 12 measured, so batching and utilisation decide the price of every token.`),
        p(`Drift takes new forms. The provider updates the model behind an API name and your carefully tuned prompt behaves differently on Tuesday. Users discover a new use for the product. The documents behind a retrieval system go out of date. None of this raises an error. A fixed evaluation set, rerun on a schedule and before every change, is how you notice.`),
        p(`Agents multiply it all. A task that takes forty model calls has forty chances to hit the tail, forty charges on the bill and forty places for a retry to fire. An agent with a 99% reliable step finishes a forty-step task cleanly two times in three. The engineering in this chapter is what makes long-running agents affordable and safe to leave alone.`),
      ));

      root.append(section('The whole road, from one line',
        p(`Chapter 1 began with a single straight line that you dragged by hand until it separated some dots. Then a rule that moved the line's numbers for you. That was learning: nudge the weights to shrink the error.`),
        p(`Everything since was that idea, extended. Stack the units and pass the blame backwards, and you have a neural network. Hold data back to check it honestly. Shape the network for images, for sequences, for meaning. Let every word look at every other word, and you have the transformer. Feed it a large share of what people have written, then teach it manners, tools and judgement, and you have the assistants of the last few chapters.`),
        p(`This part of the course was about the unglamorous remainder: the simpler models that still run most of industry, choosing between them, getting data worth learning from, measuring honestly, and keeping a system standing once real people lean on it. That work is most of the job, and it is where a careful person with good statistical sense is worth a great deal.`),
        p(`You can now read a model card, a paper's method section or an incident report and follow the argument. You can ask a vendor how the number was measured, and on what. You can open the labs, change the code and predict what will happen. There is a long way from here to owning a production system, and it is travelled by building things, which you are now equipped to start.`),
      ));

      root.append(
        ctx.quiz([
          { q: 'One server handles requests that take 200 ms each on average. Four requests arrive every second. Using this chapter\'s rule for one server, what is the average wait in the queue?',
            options: ['200 ms', '800 ms', '160 ms', '1,000 ms'], answer: 1,
            explain: 'The server can clear 5 a second, so it is 4 ÷ 5 = 0.8 busy. ρ ÷ (1 − ρ) = 0.8 ÷ 0.2 = 4 jobs, and 4 × 200 ms = 800 ms. The 200 ms of work comes on top of that.' },
          { q: 'A dashboard shows an average response time of 250 ms, well inside target. Support is flooded with "it hangs" complaints. p99 is 4 seconds, and loading a page makes 30 requests. Roughly what share of page loads meet at least one of those slowest-1% requests?',
            options: ['1%', '3%', '26%', '99%'], answer: 2,
            explain: 'The chance all 30 requests avoid the slowest 1% is 0.99 multiplied by itself 30 times, about 0.74. So about 26% of page loads hit the tail. The average hid it, as it did when you pressed Cold cache.' },
          { q: 'A release that gets 12% of answers wrong, against 2% before, goes to a 5% canary on a service taking 1,000 requests a minute. It is rolled back after 20 minutes. How many extra wrong answers did it cause, and how many would shipping to everyone have caused in the same time?',
            options: ['About 100, against about 2,000', 'About 120, against about 1,200', 'About 1,000, against about 2,000', 'About 5, against about 100'], answer: 0,
            explain: 'Damage is share × traffic × extra error rate × minutes. The canary: 0.05 × 1,000 × 0.10 × 20 = 100. Everyone: 1,000 × 0.10 × 20 = 2,000. Only 10 of the 12 points count, because 2% would have been wrong anyway.' },
          { q: 'Your fraud model\'s true labels arrive 60 days late. This week, with no release and no code change, the share of payments it flags rose from 4% to 9%. What is the best reading?',
            options: ['Nothing can be concluded until the labels arrive in 60 days', 'The model has got better at catching fraud, so leave it alone', 'Roll back to the previous model version to restore the old rate', 'The incoming payments have shifted; label a recent sample now'], answer: 3,
            explain: 'This is the yellow line in the drift demo: a signal that needs no labels moved while the model stayed frozen. There is no earlier version to blame, since nothing was released. The inputs changed, and a small freshly labelled sample tells you how much accuracy it cost and whether to retrain.' },
          { q: 'Your provider is in an outage and 60% of calls fail at once. You add three immediate retries and no backup. What happens?',
            options: ['Errors fall to zero, at the cost of slower answers', 'About 13% see an error, and the provider gets twice the calls', 'Errors stay at 60%, because retries cannot help in an outage', 'About 40% see an error, and the load on the provider is unchanged'], answer: 1,
            explain: 'A user sees an error only if all four tries fail: 0.6 × 0.6 × 0.6 × 0.6 = 0.13. But the tries add up to 1 + 0.6 + 0.36 + 0.216 = 2.2 calls per request, aimed at a system that is already failing. Spaced retries with a limit, plus a backup answer, is the better design.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://sre.google/sre-book/table-of-contents/" target="_blank" rel="noopener">Google, "Site Reliability Engineering"</a>: the free book on running services. The chapters on monitoring, service level objectives and postmortems are the long form of half this chapter.`,
            `<a href="https://research.google/pubs/pub40801/" target="_blank" rel="noopener">Dean &amp; Barroso (2013), "The Tail at Scale"</a>: why the slowest 1% of requests dominates large systems, and the tricks that tame it. Short and very readable.`,
            `<a href="https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html" target="_blank" rel="noopener">Sculley et al. (2015), "Hidden Technical Debt in Machine Learning Systems"</a>: the paper with the famous diagram in which the model is one small box among many large ones.`,
            `<a href="https://developers.google.com/machine-learning/guides/rules-of-ml" target="_blank" rel="noopener">Google, "Rules of Machine Learning"</a>: chapter 3 pointed here too. Read it again now; the rules about launching, monitoring and training–serving skew will mean more.`,
            `<a href="https://arxiv.org/abs/2209.09125" target="_blank" rel="noopener">Shankar et al. (2022), "Operationalizing Machine Learning: An Interview Study"</a>: eighteen machine learning engineers describe what deploying and maintaining models is like day to day.`,
          ]),
        ),
      );
    },
  });
})();
