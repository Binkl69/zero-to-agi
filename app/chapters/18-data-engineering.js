/* Zero → AGI · Chapter 18 · Data: where it comes from and how it goes wrong
   DESIGN RULE: the reader is hired to build an urgent-ticket classifier and does every data job
   by hand before it is named. They label twelve ambiguous tickets and are shown two colleagues
   who disagree with them; only then do "guideline", "adjudication" and "kappa" appear.
   Every dataset is generated from a seeded RNG so that every number the prose quotes is the
   number the demo prints on its opening draw. Where a lesson could be an accident of one draw
   (label noise on a small pile, the three cuts) the demo has a button that draws again.
   Interactives, in order: the labelling desk (opener); the agreement table that computes Cohen's
   kappa from the reader's own round; label noise against measured accuracy, with a dirty-test
   preset; a leak hunt with two planted leaky features; three ways to cut the piles, scored
   against next quarter; rare classes, resampling and the balanced-test trap; a near-duplicate
   threshold; a pipeline with a breakable stage and version fingerprints. */
(function () {
  ZTA.registerChapter({
    id: '18-data-engineering',
    num: 18,
    part: 'VI',
    title: 'Data: where it comes from and how it goes wrong',
    tagline: 'Label twelve support tickets, find out how far two colleagues disagree with you, then hunt down the leak behind a 98.6% score that falls to 34.3% on launch day.',
    minutes: 55,
    objectives: [
      'compute Cohen\'s kappa from a two-by-two table of labels and say what it corrects for',
      'find a leaky feature by asking whether its value exists at the moment of prediction',
      'choose between a random, a by-group and a by-time split for a given dataset',
      'predict what wrong labels do to a model and to the score you measure it with',
    ],
    takeaways: [
      '<b>A label is an opinion until there is a guideline.</b> Two careful people disagree on ambiguous items. The written rule, and a referee for the items it still leaves open, is most of the labelling job.',
      '<b>Kappa is agreement beyond chance.</b> κ = (observed − chance) ÷ (1 − chance). A labeller who clicks the same answer every time can agree with you often and still score zero.',
      '<b>Random label noise is survivable with lots of data. One-sided noise and small datasets are different.</b> And wrong labels in the <i>test</i> set make a good model look bad.',
      '<b>A leak is a feature you will not have at prediction time.</b> A score far better than expected is a reason to look for one before you celebrate.',
      '<b>Cut the piles the way the future will arrive.</b> Split by customer when customers repeat, by time when the world drifts, and remove duplicates before you split.',
      '<b>"Which data trained this model?" must have an answer.</b> Record a fingerprint for every stage, and keep secrets, unconsented personal data and anything you promised to delete out of the training set.',
    ],
    render(root, ctx) {
      const { h, p, section, callout, ul, cards } = ctx;
      const C = ctx.colors;
      const FONT = '13px Inter, system-ui, sans-serif';
      const BOLD = 'bold 13px Inter, system-ui, sans-serif';
      const MONO = '12px "JetBrains Mono", ui-monospace, monospace';

      /* ---------- seeded numerics shared by every demo ---------- */
      function rng(seed) {
        let a = seed >>> 0;
        return () => { a += 0x6D2B79F5; let t = a; t = Math.imul(t ^ (t >>> 15), 1 | t); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      }
      function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
      /* logistic regression by full-batch gradient descent from zero: deterministic */
      function fitLogit(X, y, wt, iters, l2) {
        const n = X.length, d = X[0].length, w = new Array(d + 1).fill(0);
        let tot = 0; for (let i = 0; i < n; i++) tot += wt ? wt[i] : 1;
        for (let it = 0; it < iters; it++) {
          const gr = new Array(d + 1).fill(0);
          for (let i = 0; i < n; i++) {
            let z = w[0]; const xi = X[i]; for (let k = 0; k < d; k++) z += w[k + 1] * xi[k];
            const e = (1 / (1 + Math.exp(-z)) - y[i]) * (wt ? wt[i] : 1);
            gr[0] += e; for (let k = 0; k < d; k++) gr[k + 1] += e * xi[k];
          }
          w[0] -= gr[0] / tot;
          for (let k = 0; k < d; k++) w[k + 1] -= gr[k + 1] / tot + (l2 || 0) * w[k + 1];
        }
        return w;
      }
      const score = (w, x) => { let z = w[0]; for (let k = 0; k < x.length; k++) z += w[k + 1] * x[k]; return z; };
      /* two overlapping clouds of tickets: class 1 (urgent) sits up and to the right */
      function blobs(seed, n, posRate) {
        const r = rng(seed), X = [], y = [], u = [];
        for (let i = 0; i < n; i++) { const c = r() < posRate ? 1 : 0, s = c ? 1 : -1; X.push([s + gauss(r), s + gauss(r)]); y.push(c); u.push(r()); }
        return { X, y, u };
      }
      const pc = (v) => (v * 100).toFixed(1) + '%';
      const chip = (text, col) => h('span', { style: { display: 'inline-block', padding: '1px 8px', borderRadius: '999px', border: '1px solid ' + col, color: col, fontSize: '.74rem', fontFamily: 'var(--mono)', marginRight: '5px', marginBottom: '3px', whiteSpace: 'nowrap' } }, text);
      const mark = (btn, on) => btn.classList.toggle('primary', !!on);

      /* ================================================================== */
      /* 1 · The labelling desk                                             */
      /* ================================================================== */
      /* b1/c1: what two simulated colleagues said with no guideline; b2/c2: with it. 1 = urgent. */
      const TICKETS = [
        { t: 'I have been charged twice for the same order. Please refund one of them.', b1: 1, c1: 1, b2: 1, c2: 1 },
        { t: 'Just wanted to say the new dashboard looks great!', b1: 0, c1: 0, b2: 0, c2: 0 },
        { t: 'THIS IS RIDICULOUS!!! The font on the settings page is far too small.', b1: 0, c1: 1, b2: 0, c2: 0 },
        { t: 'Our whole team is locked out and we have a client demo in one hour.', b1: 1, c1: 1, b2: 1, c2: 1 },
        { t: 'How do I export my data to a spreadsheet?', b1: 0, c1: 0, b2: 0, c2: 0 },
        { t: 'The app crashes every time I open it. I cannot do anything.', b1: 1, c1: 1, b2: 1, c2: 1 },
        { t: 'Would love a dark mode at some point. No rush.', b1: 0, c1: 0, b2: 0, c2: 0 },
        { t: 'I need my invoice corrected before our audit next month.', b1: 1, c1: 0, b2: 0, c2: 0 },
        { t: 'hi, the password reset email never arrives. tried 3 times. thanks', b1: 0, c1: 0, b2: 1, c2: 1 },
        { t: 'Someone I do not know has logged into my account.', b1: 1, c1: 1, b2: 1, c2: 1 },
        { t: 'Is there a student discount?', b1: 0, c1: 0, b2: 0, c2: 0 },
        { t: 'Card payments failed twice this morning, then worked on the third try.', b1: 0, c1: 1, b2: 0, c2: 1 },
      ];
      const N_T = TICKETS.length;
      const LAB = { round: 1, mine: { 1: new Array(N_T).fill(null), 2: new Array(N_T).fill(null) }, subs: [] };
      const labNotify = () => LAB.subs.forEach((f) => f());
      const word = (v) => (v ? 'urgent' : 'not urgent');

      function labelDesk() {
        const guide = h('div', { style: { display: 'none', border: '1px solid ' + C.green, borderRadius: '10px', padding: '10px 12px', marginBottom: '10px', fontSize: '.9rem' } },
          h('b', { style: { color: C.green } }, 'Guideline v1. '),
          'Urgent means the customer cannot use the product, is losing money, or has a security problem, right now. Tone is not urgency. A deadline more than two days away is not urgent.');
        const rows = TICKETS.map((tk, i) => {
          const bU = ctx.button('Urgent', () => setLab(i, 1));
          const bN = ctx.button('Not urgent', () => setLab(i, 0));
          const res = h('div', { style: { flex: '1 1 100%', minHeight: '1.5em', fontSize: '.8rem' } });
          const el = h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '6px 10px', alignItems: 'center', padding: '9px 0', borderTop: i ? '1px solid var(--line)' : 'none' } },
            h('div', { style: { flex: '1 1 250px', minWidth: '0', fontSize: '.93rem' } }, h('span', { style: { fontFamily: 'var(--mono)', color: C.muted } }, (i + 1) + '. '), tk.t),
            h('div', { style: { display: 'flex', gap: '6px', flex: '0 0 auto' } }, bU, bN),
            res);
          return { el, bU, bN, res };
        });
        const ro = ctx.readout();
        function setLab(i, v) { LAB.mine[LAB.round][i] = v; paint(); labNotify(); }
        function paint() {
          const R = LAB.round, mine = LAB.mine[R];
          guide.style.display = R === 2 ? '' : 'none';
          let done = 0, unan = 0, split = 0, yb = 0, yc = 0, bc = 0;
          rows.forEach((rw, i) => {
            const m = mine[i], b = TICKETS[i]['b' + R], c = TICKETS[i]['c' + R];
            mark(rw.bU, m === 1); mark(rw.bN, m === 0);
            rw.res.innerHTML = '';
            if (m === null) return;
            done++;
            if (m === b) yb++; if (m === c) yc++; if (b === c) bc++;
            const all = m === b && b === c;
            if (all) unan++; else split++;
            const maj = m + b + c >= 2 ? 1 : 0;
            rw.res.append(chip('you: ' + word(m), C.text), chip('B: ' + word(b), b === m ? C.muted : C.warn), chip('C: ' + word(c), c === m ? C.muted : C.warn),
              h('span', { style: { color: all ? C.green : C.warn } }, all ? 'all three agree' : 'split 2 to 1: majority says ' + word(maj) + ', goes to a referee'));
          });
          mark(r1, R === 1); mark(r2, R === 2);
          ro.set({ round: R === 1 ? '1 (no guideline)' : '2 (with guideline)', labelled: done + ' of ' + N_T, 'all three agree': unan, 'split 2 to 1': split,
            'you vs B': yb + ' of ' + done, 'you vs C': yc + ' of ' + done, 'B vs C': bc + ' of ' + done });
        }
        const r1 = ctx.button('Round 1: no guideline', () => { LAB.round = 1; paint(); labNotify(); });
        const r2 = ctx.button('Round 2: with the guideline', () => { LAB.round = 2; paint(); labNotify(); });
        const clr = ctx.button('Clear my labels', () => { LAB.mine[LAB.round] = new Array(N_T).fill(null); paint(); labNotify(); });
        paint();
        return ctx.figure(h('div', {}, guide, rows.map((r) => r.el)),
          'Twelve support tickets and one question: is it urgent? Labellers B and C are simulated, with fixed answers, and theirs stay hidden until you have committed to yours. In round 1 they work from instinct, as you do. In round 2 all three of you work from the same written rule. A ticket the three of you split on would go to a referee in a real operation; here the majority vote stands in.',
          [r1, r2, clr], ro);
      }

      /* ================================================================== */
      /* 2 · The agreement table: Cohen's kappa from the reader's own round  */
      /* ================================================================== */
      function kappaTable() {
        let pair = 'bc', R = 1;
        const body = h('div', {});
        const ro = ctx.readout();
        const NAMES = { bc: ['B', 'C'], yb: ['You', 'B'], yc: ['You', 'C'], lazy: ['B', 'Lazy'] };
        function labelsFor() {
          const A = [], B = [];
          for (let i = 0; i < N_T; i++) {
            const tk = TICKETS[i], me = LAB.mine[R][i];
            let a, b;
            if (pair === 'bc') { a = tk['b' + R]; b = tk['c' + R]; }
            else if (pair === 'lazy') { a = tk['b' + R]; b = 0; }
            else { a = me; b = pair === 'yb' ? tk['b' + R] : tk['c' + R]; }
            if (a === null || b === null) continue;
            A.push(a); B.push(b);
          }
          return [A, B];
        }
        const f2 = (v) => v.toFixed(2);
        function paint() {
          const [A, B] = labelsFor(), n = A.length, nm = NAMES[pair];
          body.innerHTML = '';
          if (!n) {
            body.append(h('div', { style: { color: C.warn, padding: '8px 0' } }, 'You have not labelled any tickets in round ' + R + ' yet. Scroll up, label some, and this table fills in.'));
            ro.set({ tickets: 0, 'observed agreement': '–', 'agreement by chance': '–', kappa: '–' });
            return;
          }
          let a = 0, b = 0, c = 0, d = 0;
          for (let i = 0; i < n; i++) { if (A[i] && B[i]) a++; else if (A[i] && !B[i]) b++; else if (!A[i] && B[i]) c++; else d++; }
          const po = (a + d) / n, pu = (a + b) / n * (a + c) / n, pn = (c + d) / n * (b + d) / n, pe = pu + pn;
          const k = pe >= 1 ? null : (po - pe) / (1 - pe);
          const cell = (txt, col, bold) => h('td', { style: { textAlign: 'center', color: col || C.text, fontWeight: bold ? '700' : '400', fontFamily: 'var(--mono)' } }, String(txt));
          const head = (txt) => h('th', { style: { textAlign: 'center' } }, txt);
          body.append(h('div', { class: 'table-wrap' }, h('table', { class: 'tbl', style: { margin: '0 0 10px' } },
            h('thead', {}, h('tr', {}, h('th', {}, ''), head(nm[1] + ': urgent'), head(nm[1] + ': not urgent'), head('total'))),
            h('tbody', {},
              h('tr', {}, h('td', {}, nm[0] + ': urgent'), cell(a, C.green, true), cell(b, C.warn), cell(a + b, C.muted)),
              h('tr', {}, h('td', {}, nm[0] + ': not urgent'), cell(c, C.warn), cell(d, C.green, true), cell(c + d, C.muted)),
              h('tr', {}, h('td', {}, 'total'), cell(a + c, C.muted), cell(b + d, C.muted), cell(n, C.muted))))));
          const line = (html) => h('div', { style: { fontFamily: 'var(--mono)', fontSize: '.82rem', margin: '5px 0', lineHeight: '1.5' }, html });
          body.append(
            line('<span style="color:' + C.green + '">agreed</span>: ' + a + ' + ' + d + ' = ' + (a + d) + ' of ' + n + ' &rarr; observed agreement <b>' + f2(po) + '</b>'),
            line('by chance, both say urgent: ' + (a + b) + '/' + n + ' &times; ' + (a + c) + '/' + n + ' = ' + f2(pu)),
            line('by chance, both say not urgent: ' + (c + d) + '/' + n + ' &times; ' + (b + d) + '/' + n + ' = ' + f2(pn)),
            line('agreement by chance: ' + f2(pu) + ' + ' + f2(pn) + ' = <b>' + f2(pe) + '</b>'),
            line(k === null ? 'kappa: undefined, because chance alone already gives 1.00'
              : 'kappa = (' + f2(po) + ' &minus; ' + f2(pe) + ') &divide; (1 &minus; ' + f2(pe) + ') = <b style="color:' + (k >= 0.6 ? C.green : k >= 0.4 ? C.warn : C.danger) + '">' + f2(k) + '</b>'));
          if ((pair === 'yb' || pair === 'yc') && n < N_T) body.append(h('div', { style: { color: C.muted, fontSize: '.8rem', marginTop: '6px' } }, 'Counted on the ' + n + ' ticket' + (n === 1 ? '' : 's') + ' you have labelled in round ' + R + '.'));
          ro.set({ tickets: n, 'observed agreement': f2(po), 'agreement by chance': f2(pe), kappa: k === null ? '–' : f2(k) });
        }
        LAB.subs.push(paint);
        const pairSel = ctx.select({ label: 'compare', value: 'bc', options: [
          { value: 'bc', label: 'B with C' }, { value: 'yb', label: 'You with B' }, { value: 'yc', label: 'You with C' },
          { value: 'lazy', label: 'B with Lazy, who clicks "not urgent" every time' }], onChange: (v) => { pair = v; paint(); } });
        const rSel = ctx.select({ label: 'labels from', value: '1', options: [{ value: '1', label: 'round 1 (no guideline)' }, { value: '2', label: 'round 2 (with the guideline)' }], onChange: (v) => { R = +v; paint(); } });
        paint();
        return ctx.figure(body,
          'Every ticket two labellers both judged lands in one of four cells. The green diagonal is where they agreed. The lines underneath redo the arithmetic live for whichever pair and round you choose, including your own labels from the desk above. The products are shown rounded to two places; the sums use the exact values.',
          [pairSel, rSel], ro);
      }

      /* ================================================================== */
      /* 3 · Label noise                                                    */
      /* ================================================================== */
      function noiseLab() {
        const W = 720, H = 350;
        const [cv, g] = ctx.canvas(W, H);
        const P = { x: 16, y: 30, w: 290, h: 290 }, Q = { x: 372, y: 30, w: 330, h: 190 };
        const pool = blobs(11, 4000, 0.5), test = blobs(7, 2000, 0.5);
        const RATES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45];
        const S = { n: 400, tr: 0, te: 0, mode: 'sym', salt: 0, dirty: true };
        const cache = {}, plans = {};
        /* Which labels go wrong, and in what order as the slider rises. The share is exact, so 5% of
           40 tickets is 2 tickets; "salt" picks a different set of victims for the same share. */
        function plan(n) {
          const key = n + ':' + S.salt;
          if (!plans[key]) {
            const r = rng(500 + 97 * S.salt + n), u = []; for (let i = 0; i < n; i++) u.push(r());
            const ord = u.map((_, i) => i).sort((a, b) => u[a] - u[b]);
            const sym = new Array(n), one = new Array(n).fill(Infinity); let k = 0;
            ord.forEach((i, rank) => { sym[i] = rank; if (pool.y[i] === 1) one[i] = k++; });
            plans[key] = { sym, one, nPos: k };
          }
          return plans[key];
        }
        const given = (i, rate, mode, n) => {
          const c = pool.y[i], pl = plan(n);
          return mode === 'sym' ? (pl.sym[i] < Math.round(rate / 100 * n) ? 1 - c : c) : (pl.one[i] < Math.round(rate / 100 * pl.nPos) ? 0 : c);
        };
        function model(n, mode, rate) {
          const key = n + mode + rate + ':' + S.salt;
          if (!cache[key]) {
            const X = pool.X.slice(0, n), y = []; for (let i = 0; i < n; i++) y.push(given(i, rate, mode, n));
            cache[key] = fitLogit(X, y, null, 200, 0);
          }
          return cache[key];
        }
        /* accuracy against the true test labels, and against test labels of which `te` per cent are flipped */
        function acc(w, te) {
          let ok = 0, okN = 0;
          for (let i = 0; i < test.X.length; i++) {
            const pr = score(w, test.X[i]) > 0 ? 1 : 0, tl = test.u[i] < te / 100 ? 1 - test.y[i] : test.y[i];
            if (pr === test.y[i]) ok++; if (pr === tl) okN++;
          }
          return [ok / test.X.length, okN / test.X.length];
        }
        const toP = (x, y) => [P.x + (x + 4) / 8 * P.w, P.y + (4 - y) / 8 * P.h];
        const ro = ctx.readout();
        function draw() {
          g.clearRect(0, 0, W, H);
          const w = model(S.n, S.mode, S.tr), shown = Math.min(S.n, 400);
          g.font = FONT; g.textAlign = 'left'; g.fillStyle = C.muted;
          g.fillText(S.n > 400 ? '400 of the ' + S.n.toLocaleString('en-GB') + ' training tickets' : 'the ' + S.n + ' training tickets', P.x, 20);
          g.fillStyle = '#0f1520'; g.fillRect(P.x, P.y, P.w, P.h); g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
          g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
          let wrong = 0;
          for (let i = 0; i < S.n; i++) if (given(i, S.tr, S.mode, S.n) !== pool.y[i]) wrong++;
          for (let i = 0; i < shown; i++) {
            const lab = given(i, S.tr, S.mode, S.n), q = toP(pool.X[i][0], pool.X[i][1]);
            g.beginPath(); g.arc(q[0], q[1], 2.6, 0, Math.PI * 2); g.fillStyle = lab ? C.orange : C.accent; g.globalAlpha = 0.85; g.fill(); g.globalAlpha = 1;
            if (lab !== pool.y[i]) { g.beginPath(); g.arc(q[0], q[1], 5, 0, Math.PI * 2); g.strokeStyle = '#fff'; g.lineWidth = 1.2; g.stroke(); }
          }
          /* the true dividing line, then the learned one */
          let a = toP(-4, 4), b = toP(4, -4);
          g.strokeStyle = C.muted; g.setLineDash([5, 4]); g.lineWidth = 1.5; g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); g.setLineDash([]);
          if (Math.abs(w[2]) > 1e-6) {
            a = toP(-4, -(w[0] + w[1] * -4) / w[2]); b = toP(4, -(w[0] + w[1] * 4) / w[2]);
            g.strokeStyle = C.green; g.lineWidth = 2.4; g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
          }
          g.restore();
          g.font = '12px Inter, system-ui, sans-serif'; g.fillStyle = C.muted; g.textAlign = 'left';
          g.fillText('orange = labelled urgent · blue = labelled not urgent · white ring = label is wrong', P.x, H - 8);

          /* chart: accuracy along the training-noise axis */
          g.font = FONT; g.fillStyle = C.muted; g.fillText('accuracy as the training labels get worse', Q.x, 20);
          g.fillStyle = '#0f1520'; g.fillRect(Q.x, Q.y, Q.w, Q.h); g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(Q.x, Q.y, Q.w, Q.h);
          const X = (rate) => Q.x + 10 + rate / 45 * (Q.w - 20), Y = (v) => Q.y + Q.h - 8 - (ctx.clamp(v, 0.4, 1) - 0.4) / 0.6 * (Q.h - 16);
          g.font = MONO; g.fillStyle = C.muted;
          g.textAlign = 'right'; [0.4, 0.6, 0.8, 1].forEach((v) => g.fillText(Math.round(v * 100) + '%', Q.x - 6, Y(v) + 4));
          g.textAlign = 'center'; [0, 15, 30, 45].forEach((v) => g.fillText(v + '%', X(v), Q.y + Q.h + 15));
          g.font = FONT; g.fillText('wrong labels in the training set', Q.x + Q.w / 2, Q.y + Q.h + 34);
          const pts = RATES.map((rt) => acc(model(S.n, S.mode, rt), S.te));
          g.save(); g.beginPath(); g.rect(Q.x, Q.y, Q.w, Q.h); g.clip();
          [[1, C.warn], [0, C.green]].forEach(([k, col]) => {
            g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
            pts.forEach((v, i) => { const x = X(RATES[i]), y = Y(v[k]); if (!i) g.moveTo(x, y); else g.lineTo(x, y); }); g.stroke();
          });
          const cur = acc(w, S.te);
          g.strokeStyle = C.text; g.setLineDash([3, 3]); g.lineWidth = 1; g.beginPath(); g.moveTo(X(S.tr), Q.y); g.lineTo(X(S.tr), Q.y + Q.h); g.stroke(); g.setLineDash([]);
          [[1, C.warn], [0, C.green]].forEach(([k, col]) => { g.beginPath(); g.arc(X(S.tr), Y(cur[k]), 5, 0, Math.PI * 2); g.fillStyle = col; g.fill(); });
          g.restore();
          g.textAlign = 'left'; g.font = FONT;
          g.fillStyle = C.green; g.fillText('● against the true labels: ' + pc(cur[0]), Q.x, Q.y + Q.h + 62);
          g.fillStyle = C.warn; g.fillText('● against the test labels you have: ' + pc(cur[1]), Q.x, Q.y + Q.h + 82);
          ro.set({ 'training tickets': S.n.toLocaleString('en-GB'), 'of which mislabelled': wrong.toLocaleString('en-GB'), 'true accuracy': pc(cur[0]), 'accuracy you would measure': pc(cur[1]) });
        }
        ctx.loop(() => { if (S.dirty) { S.dirty = false; draw(); } });
        const trSl = ctx.slider({ label: 'wrong labels in training', min: 0, max: 45, step: 5, value: 0, fmt: (v) => v + '%', onChange: (v) => { S.tr = v; S.dirty = true; } });
        const teSl = ctx.slider({ label: 'wrong labels in the test set', min: 0, max: 30, step: 5, value: 0, fmt: (v) => v + '%', onChange: (v) => { S.te = v; S.dirty = true; } });
        const nSel = ctx.select({ label: 'training tickets', value: '400', options: [{ value: '40', label: '40' }, { value: '400', label: '400' }, { value: '4000', label: '4,000' }], onChange: (v) => { S.n = +v; S.dirty = true; } });
        const mSel = ctx.select({ label: 'kind of mistake', value: 'sym', options: [{ value: 'sym', label: 'random, in both directions' }, { value: 'one', label: 'one-sided: urgent marked not urgent' }], onChange: (v) => { S.mode = v; S.dirty = true; } });
        const broken = ctx.button('Broken preset: clean model, dirty test', () => {
          S.n = 4000; S.mode = 'sym'; S.tr = 0; S.te = 20; trSl.value = 0; teSl.value = 20; S.dirty = true;
          const set = (c, i) => { const s = c.querySelector('select'); if (s) s.selectedIndex = i; };
          set(nSel, 2); set(mSel, 0);
        });
        const again = ctx.button('Reshuffle the mistakes', () => { S.salt++; S.dirty = true; });
        return ctx.figure(cv,
          'A toy version of your ticket model: each dot is a ticket placed by two numeric features, and the two kinds overlap, so the best possible line scores about 92%. Grey dashed is that best line; green is what logistic regression learned from the labels as given. The slider sets an exact share: at 5%, two of 40 labels are wrong. With one-sided mistakes it is the share of the urgent tickets that get marked not urgent. The test set is 2,000 separate tickets. Green on the chart scores the model against the truth; yellow scores it against test labels that are themselves partly wrong, which is the only number a real team ever sees.',
          [trSl, teSl, nSel, mSel, again, broken], ro);
      }

      /* ================================================================== */
      /* 4 · The leak hunt                                                  */
      /* ================================================================== */
      function leakHunt() {
        const FEATS = [
          { k: 'money', name: 'mentions_money', says: 'the message contains "refund", "charged" or "payment"' },
          { k: 'len', name: 'message_length', says: 'number of words in the message' },
          { k: 'paid', name: 'paid_plan', says: 'the customer is on a paid plan' },
          { k: 'exc', name: 'exclamation_marks', says: 'how many "!" the message has' },
          { k: 'row', name: 'row_number', says: 'position of the ticket in the exported file' },
          { k: 'tier2', name: 'sent_to_tier2', says: 'the ticket was passed to the senior support team' },
        ];
        function make(seed, n) {
          const r = rng(seed), out = [];
          for (let i = 0; i < n; i++) {
            const money = r() < 0.35 ? 1 : 0, paid = r() < 0.5 ? 1 : 0, len = gauss(r), exc = gauss(r);
            const z = -1.9 + 2.2 * money + 1.1 * paid + 0.7 * len + 0.25 * exc + gauss(r);
            out.push({ money, len, paid, exc, y: z > 0 ? 1 : 0, k: r(), t2: r() });
          }
          return out;
        }
        /* The export: ordinary tickets in arrival order, then the urgent ones appended from the
           escalations archive; and "sent to tier 2" is filled in by the human who triaged it.
           Next week's tickets arrive unsorted with that column still empty. */
        const exp = make(21, 1500), live = make(22, 1500);
        { const nN = exp.filter((r) => !r.y).length; let a = 0, b = 0;
          exp.forEach((r) => { r.row = r.y ? nN + (b++) : (a++); r.tier2 = r.y ? (r.t2 < 0.9 ? 1 : 0) : (r.t2 < 0.04 ? 1 : 0); });
          live.forEach((r, i) => { r.row = 1500 + i; r.tier2 = 0; }); }
        const tr = exp.filter((r) => r.k < 0.7), te = exp.filter((r) => r.k >= 0.7);
        const liveUrgent = live.filter((r) => r.y).length;
        const on = FEATS.map(() => true);
        let shipped = false;
        function run() {
          const idx = FEATS.map((_, i) => i).filter((i) => on[i]);
          if (!idx.length) return null;
          const mean = idx.map((i) => tr.reduce((s, r) => s + r[FEATS[i].k], 0) / tr.length);
          const sd = idx.map((i, j) => Math.sqrt(tr.reduce((s, r) => s + Math.pow(r[FEATS[i].k] - mean[j], 2), 0) / tr.length) || 1);
          const vec = (r) => idx.map((i, j) => (r[FEATS[i].k] - mean[j]) / sd[j]);
          const w = fitLogit(tr.map(vec), tr.map((r) => r.y), null, 300, 0.001);
          const hit = (set) => set.filter((r) => (score(w, vec(r)) > 0 ? 1 : 0) === r.y).length / set.length;
          const wts = FEATS.map(() => 0); idx.forEach((i, j) => { wts[i] = w[j + 1]; });
          return { test: hit(te), live: hit(live), flagged: live.filter((r) => score(w, vec(r)) > 0).length, caught: live.filter((r) => r.y && score(w, vec(r)) > 0).length / liveUrgent, wts };
        }
        const ro = ctx.readout();
        const big = (lab) => { const v = h('div', { style: { fontFamily: 'var(--mono)', fontSize: '1.7rem', fontWeight: '700' } }, '–'); const sub = h('div', { style: { fontSize: '.78rem', color: C.muted, minHeight: '2.4em' } }); return { v, sub, el: h('div', { style: { flex: '1 1 200px', border: '1px solid var(--line)', borderRadius: '10px', padding: '10px 12px' } }, h('div', { style: { fontSize: '.8rem', color: C.muted } }, lab), v, sub) }; };
        const bTest = big('accuracy on your test pile'), bLive = big('accuracy on next week\'s tickets');
        const rows = FEATS.map((f, i) => {
          const btn = ctx.button(f.name, () => { on[i] = !on[i]; shipped = false; paint(); });
          const bar = h('div', { style: { height: '10px', borderRadius: '5px', background: C.accent, width: '0%' } });
          const num = h('span', { style: { fontFamily: 'var(--mono)', fontSize: '.78rem', color: C.muted, flex: '0 0 44px', textAlign: 'right' } }, '');
          const el = h('div', { style: { display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '6px 10px', padding: '7px 0', borderTop: i ? '1px solid var(--line)' : 'none' } },
            h('div', { style: { flex: '0 0 172px' } }, btn),
            h('div', { style: { flex: '1 1 200px', fontSize: '.84rem', color: C.muted, minWidth: '0' } }, f.says),
            h('div', { style: { flex: '1 1 150px', display: 'flex', alignItems: 'center', gap: '8px' } }, h('div', { style: { flex: '1 1 auto', background: '#0a0e16', borderRadius: '5px', overflow: 'hidden' } }, bar), num));
          return { btn, bar, num, el };
        });
        function paint() {
          const r = run();
          rows.forEach((rw, i) => {
            mark(rw.btn, on[i]);
            rw.btn.style.opacity = on[i] ? '1' : '0.55';
            const wv = r ? Math.abs(r.wts[i]) : 0;
            rw.bar.style.width = Math.min(100, wv / 6.2 * 100).toFixed(1) + '%';
            rw.num.textContent = on[i] && r ? wv.toFixed(2) : 'off';
          });
          const nOn = on.filter(Boolean).length;
          if (!r) {
            bTest.v.textContent = '–'; bTest.sub.textContent = 'No features switched on: nothing to learn from.';
            bLive.v.textContent = '–'; bLive.sub.textContent = ''; bLive.v.style.color = C.text;
            ro.set({ 'features in use': 0, 'test accuracy': '–', 'next week': '–' });
            return;
          }
          bTest.v.textContent = pc(r.test); bTest.v.style.color = C.text;
          bTest.sub.textContent = te.length + ' tickets held back from the same export';
          if (shipped) {
            const gap = r.test - r.live;
            bLive.v.textContent = pc(r.live); bLive.v.style.color = gap > 0.05 ? C.danger : C.green;
            bLive.sub.textContent = 'flags ' + r.flagged.toLocaleString('en-GB') + ' of ' + live.length.toLocaleString('en-GB') + ' as urgent; catches ' + Math.round(r.caught * 100) + '% of the ' + liveUrgent + ' that are';
          } else { bLive.v.textContent = '?'; bLive.v.style.color = C.muted; bLive.sub.textContent = 'Press Ship it to find out.'; }
          ro.set({ 'features in use': nOn + ' of 6', 'test accuracy': pc(r.test), 'next week': shipped ? pc(r.live) : 'not shipped', 'gap': shipped ? ((r.test - r.live) * 100).toFixed(1) + ' points' : '–' });
        }
        const ship = ctx.button('Ship it', () => { shipped = true; paint(); }, 'primary');
        const all = ctx.button('Use all six again', () => { for (let i = 0; i < on.length; i++) on[i] = true; shipped = false; paint(); });
        paint();
        const body = h('div', {},
          h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '10px', marginBottom: '10px' } }, bTest.el, bLive.el),
          h('div', { style: { fontSize: '.78rem', color: C.muted, marginBottom: '2px' } }, 'Feature (press to switch on or off) · what it records · how hard the model leans on it'),
          rows.map((r) => r.el));
        return ctx.figure(body,
          'A logistic regression predicts "urgent" from up to six columns of a 1,500-ticket export: 1,057 to train on, 443 held back as the test pile. Next week is 1,500 fresh tickets of the same kind, scored as they arrive. Each bar is the size of the weight the model put on that column after scaling, so a long bar is a column it leans on. Two of the six columns carry the answer in a way that will not exist next week.',
          [ship, all], ro);
      }

      /* ================================================================== */
      /* 5 · Three ways to cut the piles                                    */
      /* ================================================================== */
      function splitLab() {
        const W = 720, H = 340;
        const [cv, g] = ctx.canvas(W, H);
        const G = 3000, DR = 0.1, EN = 0.15, JIT = 0.004;
        function gen(seed, fut) {
          const r = rng(seed), T = [];
          for (let c = 0; c < G; c++) {
            const cx = gauss(r), cy = gauss(r), e = EN * gauss(r), s = fut ? 13 : 1 + Math.floor(r() * 11);
            for (let m = s; m < s + 2 && (fut || m <= 12); m++) for (let k = 0; k < 2; k++) {
              const x = [cx + JIT * gauss(r), cy + JIT * gauss(r)];
              const y = (cx + e - DR * (m - 6.5)) > 0 ? 1 : 0;
              const flip = r() < 0.02;
              T.push({ g: c, m, k, x, y: flip ? 1 - y : y, u: r(), s });
            }
          }
          return T;
        }
        /* The memoriser: the label of the nearest stored ticket. `a` is sorted by x[0], so the search
           starts at the query's place in that order and walks outwards until no closer ticket is possible. */
        function nearest(a, x) {
          const n = a.length; let lo = 0, hi = n;
          while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid].x[0] < x[0]) lo = mid + 1; else hi = mid; }
          let best = Infinity, y = 0, i = lo - 1, j = lo;
          for (;;) {
            const di = i >= 0 ? x[0] - a[i].x[0] : Infinity, dj = j < n ? a[j].x[0] - x[0] : Infinity, dm = Math.min(di, dj);
            if (dm * dm >= best) break;
            const t = di <= dj ? a[i--] : a[j++], dx = t.x[0] - x[0], dy = t.x[1] - x[1], d = dx * dx + dy * dy;
            if (d < best) { best = d; y = t.y; }
          }
          return y;
        }
        const hit = (train, set) => { let ok = 0; for (const t of set) if (nearest(train, t.x) === t.y) ok++; return ok / Math.max(1, set.length); };
        const NAMES = { random: 'ticket by ticket, at random', customer: 'whole customers at a time', time: 'months 1 to 9 train, 10 to 12 test' };
        /* One dataset per draw. Role of every ticket under each cut: 0 train, 1 test, 2 set aside. */
        let nth = 0, DS = null;
        function build() {
          const D = gen(8 + 1000 * nth, false).sort((p, q) => p.x[0] - q.x[0]), F = gen(108 + 1000 * nth, true);
          const gu = []; { const r = rng(57 + 1000 * nth); for (let c = 0; c < G; c++) gu.push(r()); }
          const early = new Set(); D.forEach((t) => { if (t.m <= 9) early.add(t.g); });
          const role = {
            random: (t) => (t.u < 0.25 ? 1 : 0),
            customer: (t) => (gu[t.g] < 0.25 ? 1 : 0),
            time: (t) => (t.m <= 9 ? 0 : early.has(t.g) ? 2 : 1),
          };
          const res = {};
          Object.keys(role).forEach((k) => {
            const tr = D.filter((t) => role[k](t) === 0), te = D.filter((t) => role[k](t) === 1);
            res[k] = { test: hit(tr, te), fut: hit(tr, F), nTr: tr.length, nTe: te.length, nOut: D.length - tr.length - te.length };
          });
          /* forty customers, one row each, ordered by the month they first wrote in */
          const shown = [], seen = new Set();
          D.forEach((t) => { if (t.g < 40 && !seen.has(t.g)) { seen.add(t.g); shown.push({ g: t.g, s: t.s }); } });
          shown.sort((p, q) => p.s - q.s || p.g - q.g);
          const rowOf = {}; shown.forEach((c, i) => { rowOf[c.g] = i; });
          DS = { D: D.filter((t) => t.g < 40), role, res, rowOf };
        }
        let cut = 'random', dirty = true;
        const ro = ctx.readout();
        const GX = 20, GY = 44, CW = 34, RH = 6.4;
        function draw() {
          if (!DS) build();
          g.clearRect(0, 0, W, H);
          const r = DS.res[cut], role = DS.role[cut], rowOf = DS.rowOf, num = (v) => v.toLocaleString('en-GB');
          g.font = FONT; g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText('month', GX, 18);
          g.font = MONO; g.textAlign = 'center';
          for (let m = 1; m <= 12; m++) g.fillText(String(m), GX + (m - 0.5) * CW, 36);
          g.fillStyle = '#0f1520'; g.fillRect(GX, GY, 12 * CW, 40 * RH); g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(GX, GY, 12 * CW, 40 * RH);
          g.save(); g.beginPath(); g.rect(GX, GY, 12 * CW, 40 * RH); g.clip();
          DS.D.forEach((t) => {
            const row = rowOf[t.g]; if (row == null) return;
            const rl = role(t);
            g.fillStyle = rl === 0 ? C.accent : rl === 1 ? C.orange : '#3b4458';
            g.fillRect(GX + (t.m - 1) * CW + 2 + t.k * (CW / 2 - 1), GY + row * RH + 1, CW / 2 - 3, RH - 1.6);
          });
          g.restore();
          g.font = '12px Inter, system-ui, sans-serif'; g.textAlign = 'left'; g.fillStyle = C.muted;
          g.fillText('40 of the 3,000 customers, one row each · blue = train · orange = test · grey = set aside', GX, H - 12);
          /* right panel */
          const RX = 452, RW = 250;
          g.font = BOLD; g.fillStyle = C.text; g.fillText('cut: ' + NAMES[cut], RX, 56);
          const bar = (y, lab, v, col) => {
            g.font = FONT; g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText(lab, RX, y);
            g.fillStyle = '#0f1520'; g.fillRect(RX, y + 8, RW, 18);
            g.fillStyle = col; g.fillRect(RX, y + 8, RW * v, 18);
            g.font = BOLD; g.fillStyle = col; g.textAlign = 'right'; g.fillText(pc(v), RX + RW, y);
          };
          bar(96, 'what the test pile says', r.test, C.orange);
          bar(160, 'next quarter, new customers', r.fut, C.green);
          const gap = (r.test - r.fut) * 100;
          g.font = FONT; g.textAlign = 'left'; g.fillStyle = gap > 10 ? C.danger : gap > 5 ? C.warn : C.green;
          g.fillText('flattery: ' + gap.toFixed(1) + ' points', RX, 226);
          g.fillStyle = C.muted; g.font = MONO;
          g.fillText('train ' + num(r.nTr) + ' · test ' + num(r.nTe), RX, 256);
          g.fillText('set aside ' + num(r.nOut), RX, 274);
          ro.set({ dataset: 'no. ' + (nth + 1), cut: NAMES[cut], 'test pile says': pc(r.test), 'next quarter delivers': pc(r.fut), flattery: gap.toFixed(1) + ' points' });
          mark(bR, cut === 'random'); mark(bC, cut === 'customer'); mark(bT, cut === 'time');
        }
        ctx.loop(() => { if (dirty) { dirty = false; draw(); } });
        const bR = ctx.button('Cut at random', () => { cut = 'random'; dirty = true; });
        const bC = ctx.button('Cut by customer', () => { cut = 'customer'; dirty = true; });
        const bT = ctx.button('Cut by time', () => { cut = 'time'; dirty = true; });
        const bN = ctx.button('New data', () => { nth++; DS = null; dirty = true; });
        return ctx.figure(cv,
          '12,000 simulated tickets from 3,000 customers over twelve months; the grid shows 40 of the customers. Each customer writes four tickets across two months, and their tickets resemble each other. What counts as urgent also drifts a little every month. The model is the simplest one from chapter 16, a nearest-neighbour memoriser: it gives a new ticket the label of the most similar ticket it has stored. "Next quarter" is 12,000 tickets from 3,000 customers it has never seen, in months 13 and 14. The time cut also sets aside test-month tickets from customers who already appear in training.',
          [bR, bC, bT, bN], ro);
      }

      /* ================================================================== */
      /* 6 · Rare classes, resampling, and the balanced-test trap           */
      /* ================================================================== */
      function rareLab() {
        const W = 720, H = 330;
        const [cv, g] = ctx.canvas(W, H);
        const P = { x: 16, y: 30, w: 270, h: 270 };
        const RATE = { 2: 0.5, 5: 0.2, 20: 0.05, 100: 0.01 };
        const S = { one: 20, recipe: 'asis', mix: 'real', dirty: true };
        const cache = {};
        function data(one) {
          const k = 'd' + one;
          if (!cache[k]) cache[k] = { tr: blobs(31 + one, 4000, RATE[one]), real: blobs(32 + one, 20000, RATE[one]), bal: blobs(33, 2000, 0.5) };
          return cache[k];
        }
        function model(one, recipe) {
          const k = 'm' + one + recipe;
          if (cache[k]) return cache[k];
          const d = data(one).tr; let X = d.X, y = d.y, wt = null, nPos = 0;
          for (const v of y) nPos += v;
          const nNeg = y.length - nPos;
          if (recipe === 'down') {
            X = []; y = []; let kept = 0;
            for (let i = 0; i < d.y.length; i++) { if (d.y[i] === 1) { X.push(d.X[i]); y.push(1); } else if (kept < nPos) { X.push(d.X[i]); y.push(0); kept++; } }
          } else if (recipe === 'weight') { wt = d.y.map((v) => (v ? nNeg / Math.max(1, nPos) : 1)); }
          return (cache[k] = { w: fitLogit(X, y, wt, 200, 0), used: X.length, nPos });
        }
        const toP = (x, y) => [P.x + (x + 4) / 8 * P.w, P.y + (4 - y) / 8 * P.h];
        const ro = ctx.readout();
        function draw() {
          g.clearRect(0, 0, W, H);
          const d = data(S.one), m = model(S.one, S.recipe), set = S.mix === 'real' ? d.real : d.bal, w = m.w;
          let tp = 0, fp = 0, fn = 0, tn = 0;
          for (let i = 0; i < set.X.length; i++) { const pr = score(w, set.X[i]) > 0 ? 1 : 0; if (pr && set.y[i]) tp++; else if (pr) fp++; else if (set.y[i]) fn++; else tn++; }
          const n = set.X.length, rec = tp / Math.max(1, tp + fn), prec = tp + fp ? tp / (tp + fp) : 0, accu = (tp + tn) / n;
          g.font = FONT; g.textAlign = 'left'; g.fillStyle = C.muted;
          g.fillText('1,500 tickets from the test pile', P.x, 20);
          g.fillStyle = '#0f1520'; g.fillRect(P.x, P.y, P.w, P.h); g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
          g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
          for (const pass of [0, 1]) for (let i = 0; i < 1500; i++) {
            if (set.y[i] !== pass) continue;
            const q = toP(set.X[i][0], set.X[i][1]);
            g.beginPath(); g.arc(q[0], q[1], pass ? 3.2 : 2.2, 0, Math.PI * 2); g.fillStyle = pass ? C.orange : C.accent; g.globalAlpha = pass ? 1 : 0.6; g.fill(); g.globalAlpha = 1;
          }
          if (Math.abs(w[2]) > 1e-6) {
            const a = toP(-4, -(w[0] + w[1] * -4) / w[2]), b = toP(4, -(w[0] + w[1] * 4) / w[2]);
            g.strokeStyle = C.green; g.lineWidth = 2.4; g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
          }
          g.restore();
          g.font = '12px Inter, system-ui, sans-serif'; g.fillStyle = C.muted;
          g.fillText('orange = truly urgent · blue = not · everything above the green line gets flagged', P.x, H - 10);
          const RX = 330, RW = 372;
          const bar = (y, lab, v, col) => {
            g.font = FONT; g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText(lab, RX, y);
            g.fillStyle = '#0f1520'; g.fillRect(RX, y + 8, RW, 16);
            g.fillStyle = col; g.fillRect(RX, y + 8, RW * v, 16);
            g.font = BOLD; g.fillStyle = col; g.textAlign = 'right'; g.fillText(pc(v), RX + RW, y);
          };
          bar(52, 'urgent tickets caught (recall)', rec, C.orange);
          bar(112, 'alarms that were real (precision)', prec, C.green);
          bar(172, 'accuracy', accu, C.accent);
          g.font = MONO; g.fillStyle = C.muted; g.textAlign = 'left';
          g.fillText('trained on ' + m.used.toLocaleString('en-GB') + ' tickets, ' + m.nPos + ' of them urgent', RX, 232);
          g.fillText('test: ' + n.toLocaleString('en-GB') + ' tickets, ' + (tp + fn).toLocaleString('en-GB') + ' urgent, ' + (tp + fp).toLocaleString('en-GB') + ' flagged', RX, 252);
          g.font = FONT; g.fillStyle = S.mix === 'real' ? C.muted : C.warn;
          g.fillText(S.mix === 'real' ? 'test pile has the real-world mix' : 'test pile rebalanced to 50/50: not what production looks like', RX, 280);
          ro.set({ 'urgent caught': pc(rec), 'alarms that were real': pc(prec), accuracy: pc(accu), flagged: (tp + fp).toLocaleString('en-GB') + ' of ' + n.toLocaleString('en-GB') });
        }
        ctx.loop(() => { if (S.dirty) { S.dirty = false; draw(); } });
        const pick = (c, i) => { const s = c.querySelector('select'); if (s) s.selectedIndex = i; };
        const oSel = ctx.select({ label: 'how rare is urgent', value: '20', options: [{ value: '2', label: '1 ticket in 2' }, { value: '5', label: '1 ticket in 5' }, { value: '20', label: '1 ticket in 20' }, { value: '100', label: '1 ticket in 100' }], onChange: (v) => { S.one = +v; S.dirty = true; } });
        const rSel = ctx.select({ label: 'training recipe', value: 'asis', options: [{ value: 'asis', label: 'train on it as it comes' }, { value: 'down', label: 'throw away common tickets until 50/50' }, { value: 'weight', label: 'count each rare ticket extra' }], onChange: (v) => { S.recipe = v; S.dirty = true; } });
        const mSel = ctx.select({ label: 'test pile', value: 'real', options: [{ value: 'real', label: 'real-world mix' }, { value: 'bal', label: 'rebalanced to 50/50' }], onChange: (v) => { S.mix = v; S.dirty = true; } });
        const broken = ctx.button('Broken preset: balanced test pile', () => { S.one = 100; S.recipe = 'down'; S.mix = 'bal'; pick(oSel, 3); pick(rSel, 1); pick(mSel, 1); S.dirty = true; });
        return ctx.figure(cv,
          'The same two overlapping clouds as before, but now you choose how rare the urgent kind is. The training pile is always 4,000 tickets drawn at that rate; the real-world test pile is 20,000 at the same rate. The green line is where a logistic regression puts its cut-off under each training recipe. Precision and recall are the two questions from chapter 3\'s spam dial.',
          [oSel, rSel, mSel, broken], ro);
      }

      /* ================================================================== */
      /* 7 · Duplicates and near-duplicates                                 */
      /* ================================================================== */
      function dedupLab() {
        /* c = which real-world message this is a copy of (the hidden truth); pile: 0 train, 1 test */
        const ITEMS = [
          { t: 'My order #4471 has not arrived yet.', c: 'A', pile: 0 },
          { t: 'How do I reset my password?', c: 'B', pile: 0 },
          { t: 'I was charged twice, please refund me.', c: 'C', pile: 0 },
          { t: 'The app crashes when I upload a photo.', c: 'D', pile: 0 },
          { t: 'Can I change my delivery address?', c: 'E', pile: 0 },
          { t: 'My order #5120 still has not arrived.', c: 'A', pile: 0 },
          { t: 'my order #9302 has not arrived yet', c: 'A', pile: 1 },
          { t: 'How do I reset my password?', c: 'B', pile: 1 },
          { t: 'how can i reset my password', c: 'B', pile: 1 },
          { t: 'I was charged twice. Please refund me!!', c: 'C', pile: 1 },
          { t: 'The app crashes when I open settings.', c: 'F', pile: 1 },
          { t: 'Can I change the email on my account?', c: 'G', pile: 1 },
        ];
        let thr = 1, tidy = false;
        const toks = (s) => new Set((tidy ? s.toLowerCase().replace(/[^a-z ]/g, ' ') : s).split(/\s+/).filter(Boolean));
        function sim(a, b) {
          if (!tidy && thr >= 1) return a === b ? 1 : 0;            /* exact match means the same string */
          const A = toks(a), B = toks(b); let inter = 0; A.forEach((w) => { if (B.has(w)) inter++; });
          return inter / (A.size + B.size - inter);
        }
        const list = h('div', {});
        const ro = ctx.readout();
        const rowEls = ITEMS.map((it, i) => {
          const status = h('div', { style: { flex: '1 1 210px', fontSize: '.8rem', fontFamily: 'var(--mono)' } });
          const text = h('div', { style: { flex: '2 1 240px', fontSize: '.9rem', minWidth: '0' } }, it.t);
          const el = h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '4px 10px', alignItems: 'center', padding: '6px 0', borderTop: i ? '1px solid var(--line)' : 'none' } },
            h('div', { style: { flex: '0 0 74px' } }, chip((i + 1) + ' ' + (it.pile ? 'test' : 'train'), it.pile ? C.orange : C.accent)), text, status);
          list.append(el);
          return { status, text };
        });
        function paint() {
          const kept = [], verdict = [];
          ITEMS.forEach((it, i) => {
            let best = -1, bs = 0;
            for (const j of kept) { const s = sim(ITEMS[j].t, it.t); if (s > bs) { bs = s; best = j; } }
            if (best >= 0 && bs >= thr - 1e-9) verdict.push({ drop: true, of: best, s: bs }); else { kept.push(i); verdict.push({ drop: false }); }
          });
          let twins = 0, wrong = 0;
          ITEMS.forEach((it, i) => {
            const v = verdict[i], rw = rowEls[i];
            rw.status.innerHTML = '';
            rw.text.style.opacity = v.drop ? '0.45' : '1'; rw.text.style.textDecoration = v.drop ? 'line-through' : 'none';
            if (v.drop) {
              const bad = ITEMS[v.of].c !== it.c; if (bad) wrong++;
              rw.status.append(h('span', { style: { color: bad ? C.danger : C.muted } }, 'removed: ' + (v.s >= 0.9995 ? 'same as' : v.s.toFixed(2) + ' like') + ' no. ' + (v.of + 1) + (bad ? ' — a different problem!' : '')));
            } else if (it.pile === 1 && kept.some((j) => ITEMS[j].pile === 0 && ITEMS[j].c === it.c)) {
              twins++;
              rw.status.append(h('span', { style: { color: C.warn } }, 'kept, but has a twin in training'));
            } else rw.status.append(h('span', { style: { color: C.green } }, 'kept'));
          });
          mark(tBtn, tidy);
          ro.set({ 'tidy first': tidy ? 'on' : 'off', 'match threshold': thr.toFixed(2), 'tickets kept': kept.length + ' of ' + ITEMS.length, 'test tickets with a twin in training': twins, 'different tickets wrongly merged': wrong });
        }
        const sl = ctx.slider({ label: 'how alike counts as a duplicate', min: 0.3, max: 1, step: 0.05, value: 1, digits: 2, onChange: (v) => { thr = v; paint(); } });
        const tBtn = ctx.button('Tidy the text first', () => { tidy = !tidy; paint(); });
        paint();
        return ctx.figure(list,
          'Twelve tickets, six already dealt to the training pile and six to the test pile. Likeness is word overlap: the words two messages share, divided by all the words either uses (the Jaccard measure; real systems use faster approximations of the same idea). At 1.00 with tidying off, only character-for-character copies match. "Tidy the text first" lower-cases everything and strips punctuation and digits before comparing. Tickets are read top to bottom and each is compared with the ones already kept.',
          [sl, tBtn], ro);
      }

      /* ================================================================== */
      /* 8 · A pipeline with a breakable stage and version fingerprints     */
      /* ================================================================== */
      function pipelineLab() {
        /* The two scores per fault are ILLUSTRATIVE, chosen to match the direction and rough size
           of what the demos above measured. The fingerprints are real hashes of each stage's
           settings chained to the stage before it. */
        const ST = [
          { name: 'Collect', cfg: 'sources=web,phone', does: 'ticket logs from the website and the phone app',
            brk: 'A logging change silently drops every ticket from the phone app.', bcfg: 'sources=web', rep: 86, real: 71,
            who: 'Nobody. The test pile lost its phone tickets too, so the score does not move. Phone users find out.' },
          { name: 'Clean', cfg: 'dedup=0.70', does: 'strip signatures, remove near-duplicates',
            brk: 'The duplicate filter is switched off.', bcfg: 'dedup=off', rep: 93, real: 84,
            who: 'Nobody. The score went up, and nobody investigates good news.' },
          { name: 'Label', cfg: 'guideline=v3', does: 'two labellers per ticket, a referee for splits',
            brk: 'A new labelling vendor starts work without the guideline.', bcfg: 'guideline=none', rep: 74, real: 76,
            who: 'A team that tracks kappa sees it fall before any model is trained. A team that does not sees a worse model and no reason.' },
          { name: 'Split', cfg: 'split=customer+time', does: 'whole customers, newest months held out',
            brk: 'Someone replaces the split with a plain random shuffle.', bcfg: 'split=random', rep: 94, real: 85,
            who: 'Nobody, until launch. The model itself is fine; only its report card is wrong.' },
          { name: 'Train', cfg: 'file=tickets_v7', does: 'fit the model on the training pile',
            brk: 'The training job reads an old file, tickets_v3, by mistake.', bcfg: 'file=tickets_v3', rep: 79, real: 78,
            who: 'You, if you look: the score drops. Finding out why is the hard part.' },
          { name: 'Evaluate', cfg: 'test=audited', does: 'score once on a test pile whose labels were re-checked',
            brk: 'The test labels were never re-checked, and one in ten is wrong.', bcfg: 'test=raw', rep: 78, real: 85,
            who: 'Nobody. A good model looks bad, and may be thrown away for a worse one.' },
        ];
        const fnv = (s) => { let x = 0x811c9dc5; for (let i = 0; i < s.length; i++) { x ^= s.charCodeAt(i); x = Math.imul(x, 0x01000193); } return ('00000' + ((x >>> 0) % 0x1000000).toString(16)).slice(-6); };
        function prints(broken) { const out = []; let prev = ''; ST.forEach((s, i) => { prev = fnv(prev + '|' + s.name + '|' + (i === broken ? s.bcfg : s.cfg)); out.push(prev); }); return out; }
        const GOOD = prints(-1);
        let broken = -1, record = true;
        const flow = h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '8px' } });
        const note = h('div', { style: { marginTop: '12px', fontSize: '.9rem', lineHeight: '1.5' } });
        const ro = ctx.readout();
        const boxes = ST.map((s, i) => {
          const state = h('div', { style: { fontSize: '.74rem', fontWeight: '700', marginTop: '4px' } });
          const fp = h('div', { style: { fontFamily: 'var(--mono)', fontSize: '.74rem', marginTop: '4px' } });
          const el = h('div', { style: { flex: '1 1 118px', border: '1px solid var(--line)', borderRadius: '10px', padding: '8px 10px', minWidth: '0' } },
            h('div', { style: { fontWeight: '700' } }, (i + 1) + '. ' + s.name),
            h('div', { style: { fontSize: '.78rem', color: C.muted, minHeight: '2.6em' } }, s.does), state, fp);
          flow.append(el);
          return { el, state, fp };
        });
        function paint() {
          const now = prints(broken);
          let changed = 0;
          boxes.forEach((b, i) => {
            const st = broken < 0 || i < broken ? 0 : i === broken ? 2 : 1;
            const col = [C.green, C.warn, C.danger][st];
            b.el.style.borderColor = col;
            b.state.style.color = col; b.state.textContent = ['working', 'inherits the fault', 'BROKEN'][st];
            const diff = now[i] !== GOOD[i]; if (diff) changed++;
            b.fp.style.color = !record ? C.muted : diff ? C.danger : C.muted;
            b.fp.textContent = record ? 'fingerprint ' + now[i] + (diff ? ' ≠ ' + GOOD[i] : '') : 'fingerprint not recorded';
            mark(btns[i], i === broken);
          });
          const s = broken < 0 ? null : ST[broken];
          const rep = s ? s.rep : 86, real = s ? s.real : 85;
          note.innerHTML = '';
          if (!s) note.append(h('div', {}, h('b', { style: { color: C.green } }, 'All six stages working. '), 'The dashboard says 86% and customers get 85%. Break a stage to see what each number does.'));
          else {
            note.append(
              h('div', {}, h('b', { style: { color: C.danger } }, 'Fault: '), s.brk),
              h('div', {}, h('b', {}, 'Who notices: '), s.who),
              h('div', {}, h('b', {}, 'Tracing it: '), record
                ? 'Compare fingerprints with the last good run. The first one that differs is stage ' + (broken + 1) + ', ' + s.name + ' (' + s.cfg + ' became ' + s.bcfg + '). Start there.'
                : 'Nothing was recorded, so nothing says which stage changed. Somebody re-runs all six by hand and compares the outputs.'));
          }
          note.append(h('div', { style: { color: C.muted, fontSize: '.8rem', marginTop: '8px' } }, 'The two scores in this figure are scripted for illustration. Nothing is trained here. The fingerprints are computed.'));
          mark(vOn, record); mark(vOff, !record);
          ro.set({ 'dashboard says': rep + '%', 'customers get': real + '%', gap: (rep - real > 0 ? '+' : '') + (rep - real) + ' points', 'fingerprints changed': record ? changed + ' of 6' : 'unknown' });
        }
        const btns = ST.map((s, i) => ctx.button('Break ' + s.name, () => { broken = broken === i ? -1 : i; paint(); }));
        const mend = ctx.button('Mend everything', () => { broken = -1; paint(); });
        const vOn = ctx.button('Version records on', () => { record = true; paint(); });
        const vOff = ctx.button('Version records off', () => { record = false; paint(); });
        paint();
        return ctx.figure(h('div', {}, flow, note),
          'The ticket pipeline, start to finish. Each stage\'s fingerprint is a real hash of its own settings chained to the fingerprint of the stage before, so a change anywhere alters every fingerprint downstream of it. The two percentages are not computed. They were written by hand to follow the direction and rough size of what the earlier demos measured, and a real pipeline would have its own.',
          btns.concat([mend, vOn, vOff]), ro);
      }

      /* ================================================================== */
      /* The chapter: touch first, read second.                             */
      /* ================================================================== */
      root.append(
        p(`You have been hired to build a model that spots urgent support tickets. Before any model can exist, somebody has to decide what "urgent" means, one ticket at a time. Today that somebody is you.`),
        callout('tryit', '🖐 Do this first — label twelve tickets',
          `<b>1.</b> For each ticket press <b>Urgent</b> or <b>Not urgent</b>. Go on instinct. Two colleagues, B and C, have already done the same pile, and their answers appear beside yours as you commit.<br>
           <b>2.</b> When all twelve are done, read the bar under the tickets. <b>B vs C</b> says <b>9 of 12</b>: two careful people, working alone, disagreed on a quarter of the pile. See how you did against each of them.<br>
           <b>3.</b> Press <b>Round 2: with the guideline</b>, read the green rule, and label all twelve again. B vs C is now <b>11 of 12</b>, and both of them have changed their minds about ticket 9.<br>
           <b>4.</b> Look at ticket 12 in round 2. The rule is written down and B and C still split on it.`),
        labelDesk(),
        p(`Nobody in round 1 was careless. Ticket 3 shouts about a font, and C heard the shouting. Ticket 8 has a deadline a month away, and B heard the deadline. Ticket 9 is polite about being unable to log in, and both of them let it through as not urgent. Nothing told anyone whether "urgent" meant tone, deadlines or consequences.`),
        p(`A model trained on those labels would have learned a blur of two definitions. The written rule in round 2 is a <em>labelling guideline</em>, and writing it is most of the work of a labelling operation. Real ones run to many pages, grow a new clause each time labellers split, and carry worked examples of the hard cases.`),
        p(`Ticket 12 shows what a guideline cannot do. Payments failed and then worked: is that customer losing money right now? When labellers still split, the item goes to a senior person who decides, a step called <em>adjudication</em>, and the ruling usually becomes a new line in the guideline.`),
      );

      root.append(section('Where training data actually comes from',
        p(`Those twelve tickets came from somewhere, and so does every training set. There are only five places to get one, and each has its own way of going wrong. Most real datasets mix several.`),
        cards([
          { title: 'Logs', body: 'What your own product already records: tickets, clicks, searches, purchases. Plentiful and free. It only shows what happened under the old system, so anything the old system hid from users is missing.' },
          { title: 'Scraping', body: 'Copying public pages, as chapter 10\'s web crawl did. Enormous and messy: duplicates, spam and machine-written text, plus real questions of licence and consent.' },
          { title: 'Buying or licensing', body: 'Paying a publisher, a data broker or another company. Clean paperwork, but you inherit their collection choices without seeing them. Always ask how it was gathered and labelled.' },
          { title: 'Human labelling', body: 'Paying people to attach answers, as you just did. The only source of labels for a judgement nobody recorded. Slow, costly, and exactly as consistent as the guideline.' },
          { title: 'Synthetic', body: 'Asking a model or a simulator to write examples. Cheap and unlimited, but it can only reflect what the generator already believes, so it needs checking against real data.' },
          { title: 'And then, almost always', body: 'A mixture. Logs supply the tickets, people supply the labels, and a model drafts extra examples of the rare cases for a person to approve.' },
        ]),
        callout('example', '🌍 You have done this job before',
          `Every time you mark an email as spam, pick the squares containing traffic lights, or press thumbs-down on a chatbot answer, you are a labeller. Products you use collect those clicks as logs and use them as labels. In chapter 11 you played the same role on purpose, picking the better of two responses to train a reward model.`),
      ));

      root.append(section('How much do two labellers really agree?',
        p(`"B and C agreed on 9 of 12" sounds like a measurement. It is missing something, and a labeller who does no work at all will show you what.`),
        callout('tryit', '🖐 Try this — catch a labeller who is not reading',
          `<b>1.</b> The table opens on <b>B with C</b>, round 1. Read the last line: observed agreement 0.75, chance 0.50, kappa <b>0.50</b>.<br>
           <b>2.</b> Set <b>compare</b> to <b>B with Lazy</b>. Lazy clicks "not urgent" on everything without reading, and still agrees with B on 7 of 12: 0.58. Kappa is <b>0.00</b>.<br>
           <b>3.</b> Set <b>compare</b> to <b>You with B</b> to see the same arithmetic on your own labels.<br>
           <b>4.</b> Switch <b>labels from</b> to <b>round 2</b> and compare <b>B with C</b> again: kappa <b>0.83</b>.`),
        kappaTable(),

        /* maths beat: Cohen's kappa, counted before it is named */
        ctx.walkthrough([
          { say: 'In round 1, B and C gave the same label to 9 of the 12 tickets.', math: '9 &divide; 12 = 0.75', note: 'That is the green diagonal of the table: 4 both-urgent plus 5 both-not-urgent.' },
          { say: 'Before trusting 0.75, ask what two people would score if neither of them read a single ticket.' },
          { say: 'B called 5 of the 12 urgent and C called 6 of the 12 urgent, so if each scattered those labels blindly they would both land on "urgent" for about a fifth of the pile.', math: '5/12 &times; 6/12 = 0.21' },
          { say: 'The same blind scattering has them both landing on "not urgent" a bit more often.', math: '7/12 &times; 6/12 = 0.29' },
          { ask: 'So how much agreement does pure luck hand B and C before either of them reads anything?',
            options: ['0.06', '0.50', '0.75'], answer: 1,
            explain: 'Add the two ways of agreeing by luck: 0.21 + 0.29 = 0.50. Half of their agreement would have turned up anyway.' },
          { say: 'Luck provides 0.50, so the most anyone could add by actually reading is the rest.', math: '1 &minus; 0.50 = 0.50' },
          { say: 'B and C added 0.25 on top of luck.', math: '0.75 &minus; 0.50 = 0.25' },
          { say: 'Divide what they added by what was there to add, and you have the share of the possible beyond-luck agreement they reached.', math: '0.25 &divide; 0.50 = 0.50', note: 'This number is <b>Cohen\'s kappa</b>, written &kappa;. 1 is perfect agreement, 0 is no better than luck.' },
          { ask: 'Lazy agreed with B on 7 of 12, which is 0.58. Luck alone also gives that pair 0.58. What is their kappa?',
            options: ['0.58', '0.42', '0'], answer: 2,
            explain: '(0.58 − 0.58) ÷ (1 − 0.58) = 0. Lazy added nothing beyond luck, and kappa says so even though the raw agreement looked respectable.' },
          { ask: 'With the guideline, B and C agree on 11 of 12, which is 0.917, and luck still gives 0.50. What is kappa now?',
            options: ['0.83', '0.92', '0.42'], answer: 0,
            explain: '(0.917 − 0.50) ÷ (1 − 0.50) = 0.417 ÷ 0.50 = 0.83. The guideline moved kappa from 0.50 to 0.83 without replacing a single labeller.' },
        ], {
          title: 'From "9 of 12" to Cohen\'s kappa',
          recap: 'kappa is the agreement two labellers reached beyond luck, as a share of the agreement beyond luck that was available.',
        }),
        ctx.decoder([
          { sym: '&kappa;', name: 'kappa', says: 'Agreement beyond chance, on a scale where 0 is luck and 1 is perfect. It can go below 0 if two labellers agree less often than luck would.', points: 'the last line under the table: 0.50 for B with C in round 1, 0.00 for B with Lazy.' },
          '= (',
          { sym: 'p<sub>o</sub>', name: 'p observed', says: 'The share of items on which the two labellers gave the same label.', points: 'the green diagonal: 9 of 12, so 0.75.' },
          '&minus;',
          { sym: 'p<sub>e</sub>', name: 'p expected', says: 'The share they would agree on by luck, given how often each one uses each label. Multiply the two "urgent" rates, multiply the two "not urgent" rates, and add.', points: 'the two "by chance" lines: 0.21 + 0.29 = 0.50.' },
          ') &divide; ( 1 &minus;',
          { sym: 'p<sub>e</sub>', name: 'p expected', says: 'The same luck figure again. One minus it is the room left for real agreement.', points: 'the 1 − 0.50 in the last line.' },
          ')',
        ], {
          title: 'κ = (p<sub>o</sub> − p<sub>e</sub>) ÷ (1 − p<sub>e</sub>)',
          plain: 'kappa is observed agreement minus chance agreement, divided by one minus chance agreement.',
        }),

        p(`The general name for what you just measured is <em>inter-annotator agreement</em>, and <em>Cohen's kappa</em> is its most common yardstick for two labellers. A widely quoted rule of thumb, from Landis and Koch in 1977, calls 0.41 to 0.60 moderate, 0.61 to 0.80 substantial and anything above almost perfect. The bands are arbitrary. The habit of measuring is what matters.`),
        p(`Low agreement limits what you can measure. If two trained people only reach 0.50 on a task, a model scored against one of them is being marked by an examiner whom a second examiner would often overrule, and a higher score may only mean it copied that one person's habits. A low kappa says: fix the guideline before you buy another ten thousand labels.`),
        callout('history', '📜 Jacob Cohen, 1960',
          `Jacob Cohen, a psychologist at New York University, published "A coefficient of agreement for nominal scales" in 1960. Psychologists often needed two judges to sort the same cases into categories, and reported how often the judges matched. Cohen's paper argued that some of those matches would have happened by chance, and proposed a coefficient with the chance agreement taken out. The same statistic now referees medical diagnoses, content moderation and the datasets behind language models.`),
      ));

      root.append(section('What wrong labels do to a model',
        p(`Even with a guideline, some labels will be wrong. Labellers get tired, misread, or hit the wrong key. The question is how much of that a model can absorb, and the answer depends on three things you are about to change.`),
        callout('tryit', '🖐 Try this — poison the labels, then poison the exam',
          `<b>1.</b> Drag <b>wrong labels in training</b> to 30%. White rings appear on 120 of the 400 dots, yet the green line moves only a little: true accuracy goes from 91.3% to 86.4%.<br>
           <b>2.</b> Set <b>training tickets</b> to 40 and drag the slider up to 40%. Sixteen of the forty labels are wrong and it reads 70.3%. Now press <b>Reshuffle the mistakes</b> eight or ten times. The share of wrong labels stays the same and different tickets get hit. The score lands anywhere from about 50%, a coin toss, to about 90%.<br>
           <b>3.</b> Change the pile to 4,000 tickets and reshuffle again. With 1,600 wrong labels the score stays within a few points of the 92% best, whichever tickets are hit.<br>
           <b>4.</b> Still at 4,000 and 40%, set <b>kind of mistake</b> to <b>one-sided</b>. Only about 800 labels are wrong now, half as many, yet the line slides off to one side and accuracy drops to about 75%. Reshuffle: it stays there.<br>
           <b>5.</b> Press <b>Broken preset: clean model, dirty test</b>. The model is as good as it can be, 92.0%. The score you would measure is 75.8%.`),
        noiseLab(),
        p(`Wrong labels are called <em>label noise</em>. When the mistakes are random and go both ways they pull the line in both directions and largely cancel, provided there are enough examples for the cancelling to happen. Four thousand tickets shrugged off 40% noise every time you reshuffled. With forty, the result depended on which labels happened to be hit.`),
        p(`One-sided mistakes never cancel. If tired labellers miss urgent tickets but never invent them, every error pushes the same way and the model learns to miss them too, however much data you add. Half as many wrong labels did far more damage. A systematic error is more dangerous than a random one of the same size.`),
        p(`The broken preset is the trap that catches experienced teams. With one test label in five flipped, a model that is right 92% of the time agrees with the flawed answer key only 75.8% of the time. Nothing is wrong with the model. Clean the test pile first: it is small, and every other number depends on it.`),
        callout('example', '🌍 The most famous test set in computer vision has wrong answers',
          `In 2021 Northcutt, Athalye and Mueller checked the test sets of ten widely used benchmarks (images, text and audio) and estimated that at least 3.3% of their labels were wrong on average. In the ImageNet validation set, which a decade of image models were ranked on, they put the figure at about 6%. They also showed that correcting such errors can change which of two models comes out ahead.`),
      ));

      root.append(section('A score too good to be true',
        p(`A colleague has exported 1,500 labelled tickets with six columns each and trained a model on them. The test score is superb. Launch is next week.`),
        callout('tryit', '🖐 Try this — ship it, then find out why you should not have',
          `<b>1.</b> All six columns are on and the test pile says <b>98.6%</b>. Press <b>Ship it</b>.<br>
           <b>2.</b> Next week's accuracy: <b>34.3%</b>. Read the small print under it. The model flagged all 1,500 tickets as urgent.<br>
           <b>3.</b> Find the leak. Look at the bars, read what each column records, and ask of each: <i>would I know this value at the moment a ticket arrives?</i> Switch a suspect off and press <b>Ship it</b> again.<br>
           <b>4.</b> There are two leaks. With both off the two numbers nearly meet: 85.6% on the held-back tickets, 84.6% next week. That is an honest model.`),
        leakHunt(),
        p(`The two culprits are <b>row_number</b> and <b>sent_to_tier2</b>. Whoever built the file listed ordinary tickets first and pasted the urgent ones in from an escalations archive afterwards, so a high row number meant urgent. Next week's rows are all numbered higher still, and the model flags every one.`),
        p(`The second is subtler. A ticket is sent to the senior team <i>because</i> a human judged it urgent. The column is a record of the answer, written after the fact. On a ticket that has just arrived it is always empty. Switch off row_number alone and ship: the model that leans on this column catches a third of the urgent tickets.`),
        p(`Both are <em>data leakage</em>: information reaching the model in training that it will not have when it is used. Your test pile could not catch it, because the test pile came from the same export and carried the same leak. That is why the score was brilliant right up to launch.`),
        callout('key', '🔑 The one question that finds most leaks',
          `For every column, ask: <b>would this value exist, with this meaning, at the moment the prediction has to be made?</b> Identifiers, row order, timestamps and anything filled in by a later step of the process are the usual suspects. A score far above what the task should allow is a symptom to investigate.`),
        callout('example', '🌍 Leakage in published science',
          `Leakage is common enough to distort research. In a survey published in 2023, Sayash Kapoor and Arvind Narayanan of Princeton collected earlier reviews that had found leakage in 294 papers across 17 scientific fields using machine learning. In some of those cases the published results turned out to be wildly overoptimistic.`),
      ));

      root.append(section('Cutting the three piles properly',
        p(`Chapter 3 gave you three piles: train, validation and test. It did not say how to choose which example goes where, and the usual answer is at random. That is correct only when every example is independent of every other, and real data rarely is.`),
        callout('tryit', '🖐 Try this — three ways to cut the same data',
          `<b>1.</b> The demo opens on <b>Cut at random</b>. Orange test tickets are sprinkled everywhere, each next to a blue ticket from the same customer. The test pile says <b>93.0%</b>; next quarter delivers 71.4%.<br>
           <b>2.</b> Press <b>Cut by customer</b>. Whole rows turn orange. The test pile now says 80.5%, and the flattery falls from 21.6 points to 8.9.<br>
           <b>3.</b> Press <b>Cut by time</b>. Everything left of month 10 trains; new customers on the right are the test. The orange bar reads 75.5% against a green 68.5%.<br>
           <b>4.</b> Compare the two bars across all three cuts. The test score fell by more than 17 points. The green bar moved by 3.<br>
           <b>5.</b> Press <b>New data</b> for a fresh 12,000 tickets and go through the three cuts again. Every number shifts by a point or two. The order does not change.`),
        splitLab(),
        p(`The random cut put tickets from the same customer on both sides. A customer's tickets look alike, so the model only had to recognise the customer and copy the label. That is leakage again, entering through the split this time. Keeping every customer wholly on one side is a <em>group split</em>.`),
        p(`The by-customer score was still too rosy, because that cut trains on October to predict March. The meaning of "urgent" drifted month by month, and a model that has seen the future has an unfair view of it. Training on the past and testing on the most recent months is a <em>time split</em>.`),
        p(`Even the time split flattered the model by 7 points, because drift does not stop on launch day. No cut removes that. The rule is to split the way the future will arrive: new customers, later dates. Whatever will be new in production must be new in the test pile.`),
        p(`One thing did get worse. Under the time cut the green bar dropped 3 points, because that model never trained on months 10 to 12 and so was three months staler on launch day. The cut is for measuring. Once you trust the number, a real team retrains on every month it has before shipping.`),
      ));

      root.append(section('When the thing you want is rare',
        p(`In a real inbox, urgent tickets are the exception. Fraud, disease and equipment failure are rarer still. When one answer vastly outnumbers the other the data has <em>class imbalance</em>, and it bends both the model and the way you score it.`),
        callout('tryit', '🖐 Try this — make the rare thing rarer',
          `<b>1.</b> The demo opens at 1 ticket in 20. Accuracy is 97.6%, yet the model catches only 59.2% of the urgent tickets.<br>
           <b>2.</b> Set <b>how rare is urgent</b> to <b>1 ticket in 100</b>. Accuracy rises to 99.4% while the share caught falls to 39.2%.<br>
           <b>3.</b> Set <b>training recipe</b> to <b>throw away common tickets until 50/50</b>. It now catches 88.7%, but only 12.4% of its alarms are real.<br>
           <b>4.</b> Press <b>Broken preset: balanced test pile</b>. Same model, and now 92.3% of its alarms look real. Set <b>test pile</b> back to <b>real-world mix</b> to see the truth again.`),
        rareLab(),
        p(`Trained on data as it comes, the model learns that "not urgent" is nearly always right and sets a cautious cut-off. Rebalancing the training pile moves the green line. Throwing away common examples is <em>downsampling</em>; counting rare ones extra achieves much the same without discarding anything. Neither creates information. Both trade missed tickets for false alarms, the same trade you made by dragging the threshold on chapter 3's dial.`),
        p(`The broken preset is the mistake to remember. Rebalance the <i>training</i> pile if it helps. Never rebalance the <i>test</i> pile. A 50/50 exam told you nine alarms in ten were real, when in the live inbox it is one in eight. The test pile must look like production.`),
      ));

      root.append(section('The same ticket twice',
        p(`Customers paste the same complaint into three forms. Templates produce a thousand messages that differ by an order number. If one copy lands in training and another in the test pile, the model is examined on a question it has already seen the answer to.`),
        callout('tryit', '🖐 Try this — tune a duplicate filter',
          `<b>1.</b> As it opens, only exact copies count. One ticket is removed, and <b>3</b> test tickets still have a twin in training. Read them: a capital letter or an order number was enough to slip through.<br>
           <b>2.</b> Press <b>Tidy the text first</b>. Three are removed now, and 1 twin is left.<br>
           <b>3.</b> Drag the slider down to <b>0.70</b>. Five removed, no twins left, nothing wrongly merged.<br>
           <b>4.</b> Keep dragging, to <b>0.50</b>. The filter now deletes ticket 11, a crash in a different part of the app. At <b>0.40</b> it deletes a second one.`),
        dedupLab(),
        p(`Messages that are the same for every practical purpose but differ by a character are <em>near-duplicates</em>, and exact matching misses nearly all of them. Catching them takes a measure of likeness and a threshold, and the threshold has no perfect setting. Too strict leaves twins in. Too loose deletes real, different examples.`),
        p(`Order matters. Remove duplicates <i>before</i> you split, or a twin can sit on each side of the line. Chapter 10's filter funnel did this to the whole web for the same reason, and there duplicates did a second harm: text seen thousands of times gets memorised instead of learned from.`),
      ));

      root.append(section('A pipeline, and a name for every version',
        p(`Collect, clean, label, split, train, evaluate. You have now done each of these by hand. In a working team they are six programs run in a chain, usually by different people, and a fault in one flows silently into the rest.`),
        callout('tryit', '🖐 Try this — break one stage at a time',
          `<b>1.</b> The scores in this last figure are scripted, a summary of what the earlier demos measured. Press <b>Break Clean</b>. The dashboard goes <i>up</i>, from 86% to 93%, while customers get 84%. Read "who notices".<br>
           <b>2.</b> Press <b>Break Evaluate</b>. The opposite: the dashboard says 78% and customers get 85%.<br>
           <b>3.</b> Press <b>Break Label</b> and look at the fingerprints. Stages 3 to 6 all changed; stages 1 and 2 did not. The first changed fingerprint points at the culprit.<br>
           <b>4.</b> Press <b>Version records off</b> and read the "Tracing it" line again.`),
        pipelineLab(),
        p(`Count how many of the six faults lowered the dashboard number: three. Two raised it and one left it untouched. A data fault often makes the report card wrong while the pupil stays the same, or the reverse. The dashboard alone cannot tell you which.`),
        p(`The fingerprints are the cure for the tracing problem. Keeping an identifier for every version of the data, the labels, the guideline and the split, and writing down which ones each model was built from, is <em>data versioning</em>. With it, "what changed since the last good model?" is a comparison of two short lists.`),
        p(`Without it you cannot reproduce a result, cannot roll back, and cannot honour a request to remove someone's data, because you do not know which models contain it. "Which data trained this model?" must always have an answer. A short document describing a dataset's origin, contents, labelling and limits is called a <em>datasheet</em>, and writing one is a good place to start.`),
      ));

      root.append(section('What must not go into a training set',
        p(`A model can repeat its training data. In 2021 a team led by Nicholas Carlini got the public language model GPT-2 to print hundreds of passages word for word from its training text, among them people's names, phone numbers and email addresses. Treat anything you train on as something a stranger might one day be shown.`),
        callout('warning', '⚠️ Keep these out unless you can defend their presence',
          `<b>Secrets.</b> Passwords, access keys, card numbers. Support tickets and code are full of them, and they should be scrubbed before anything else happens.<br>
           <b>Personal data used beyond what people agreed to.</b> A ticket was sent to get help. Whether it may also train a model depends on what the customer was told and on the law that applies, which varies from country to country.<br>
           <b>Sensitive details.</b> Health, finances, children, precise location. Even with permission, ask whether the model needs them at all.<br>
           <b>Anything you promised to delete</b>, and anything whose licence does not cover this use.<br>
           <b>The test pile.</b> The one privacy-free item on the list, and the easiest to let in by accident.`),
        p(`This list is engineering habit, not legal advice, and it is not complete. Privacy law differs between countries and changes, so a real project should ask its own privacy or legal people before any personal data is collected for training. What holds everywhere is the engineering consequence: you can only remove or justify what you can find, so privacy and versioning are one piece of work.`),
      ));

      root.append(section('Why this matters for modern AI and LLM work',
        p(`Pretraining runs on trillions of tokens, and at that scale random noise washes out, as your 4,000-ticket model showed. Most applied work is not at that scale. Chapter 11's supervised fine-tuning used thousands of written demonstrations; many company fine-tunes use a few hundred.`),
        p(`That is the 40-ticket setting on the noise demo. In a set of 200 examples, ten bad ones are 5% of everything the model is shown, and if they share a fault, such as a wrong format or a sloppy tone, the fault is systematic. Fine-tuning data deserves to be read line by line by a person.`),
        p(`Evaluation sets for language models are smaller still, often a few hundred prompts, and every lesson here applies doubly. Graders need a guideline and a kappa. Test prompts must not appear in the fine-tuning data: chapter 11 called that contamination, and it is the leak hunt at web scale.`),
        p(`Synthetic data is now a routine source: a strong model writes examples for a weaker one to learn from. It inherits the generator's blind spots and its habits of phrasing, and it produces near-duplicates in bulk, so it needs the same deduplication, sampling by hand and versioning as anything collected from people.`),
        p(`One picture to keep: <b>the model is the last step of the work, and it can only be as good as the labels, as honest as the split, and as traceable as the data it came from.</b>`),
      ));

      root.append(
        ctx.quiz([
          { q: 'Two moderators each review the same 100 comments. Each marks 10 as abusive and 90 as fine, and they give the same verdict on 90 of the 100. What is their Cohen\'s kappa?',
            options: ['0.90', 'About 0.44', 'About 0.08', '0.82'], answer: 1,
            explain: 'Chance agreement is 0.9 × 0.9 + 0.1 × 0.1 = 0.82. Kappa is (0.90 − 0.82) ÷ (1 − 0.82) = 0.08 ÷ 0.18 ≈ 0.44. When one label dominates, most of a 90% match rate comes free, which is what Lazy showed you.' },
          { q: 'A model predicting which loans will default scores 99% on its test pile. Which of its input columns should you investigate first?',
            options: ['applicant_income', 'loan_amount', 'years_at_current_address', 'collections_agency_assigned'], answer: 3,
            explain: 'A collections agency is assigned after a borrower stops paying, so the column is a record of the outcome. It will be empty for every new applicant. Ask of each column whether its value exists when the prediction is made.' },
          { q: 'You have 5,000 chest X-rays from 500 patients, about ten each, and want a model for patients the hospital has not yet seen. How should the test pile be chosen?',
            options: ['Hold out every image from 100 of the patients', 'Shuffle all 5,000 images and hold out a random fifth', 'Hold out the two clearest images of every patient', 'Hold out images at random until each patient has lost two'], answer: 0,
            explain: 'Ten images of one person resemble each other far more than images of different people. Any cut that puts the same patient on both sides lets the model recognise the patient, the same flattery the random cut produced in the ticket demo. Only holding out whole patients tests it on people it has never seen.' },
          { q: 'A model is truly right on 95% of cases. Its test pile has two possible labels, and 10% of them have been flipped at random. Roughly what accuracy will you measure?',
            options: ['90%', '95%', '86%', '76%'], answer: 2,
            explain: 'It matches a correct label when it is right (0.95 × 0.90) and matches a flipped one when it is wrong (0.05 × 0.10): 0.855 + 0.005 = 0.86. This is the broken preset: a dirty test pile makes a good model look worse than it is.' },
          { q: 'One ticket in 100 is urgent. A model catches 90% of urgent tickets and wrongly flags 10% of the others. Of every 1,000 tickets, roughly what share of its alarms are real?',
            options: ['About 90%', 'About 8%', 'About 50%', 'About 1%'], answer: 1,
            explain: 'Of 1,000 tickets, 10 are urgent and 9 are caught. Of the other 990, 99 are flagged by mistake. That is 9 real alarms out of 108, about 8%. A 50/50 test pile would have reported 90% and hidden this.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://arxiv.org/abs/2103.14749" target="_blank" rel="noopener">Northcutt, Athalye &amp; Mueller (2021), "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks"</a>: the audit of ten benchmark test sets quoted above. The authors also put the mislabelled examples online to browse, at labelerrors.com.`,
            `<a href="https://arxiv.org/abs/2207.07048" target="_blank" rel="noopener">Kapoor &amp; Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science"</a>: the survey quoted above, with a taxonomy of the ways leakage happens and a template, the "model info sheet", for checking a study against each of them.`,
            `<a href="https://scikit-learn.org/stable/common_pitfalls.html" target="_blank" rel="noopener">scikit-learn, "Common pitfalls and recommended practices"</a>: the official guide's section on data leakage, with short code examples of the wrong and the right way to prepare data.`,
            `<a href="https://arxiv.org/abs/1803.09010" target="_blank" rel="noopener">Gebru et al., "Datasheets for Datasets"</a>: the proposal that every dataset should ship with a document saying where it came from and what it is fit for, including the list of questions to answer.`,
            `<a href="https://developers.google.com/machine-learning/guides/rules-of-ml" target="_blank" rel="noopener">Google, "Rules of Machine Learning"</a>: practical rules from people who ship models. A large share of them are about data and pipelines.`,
          ]),
        ),
      );
    },
  });
})();
