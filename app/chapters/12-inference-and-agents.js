/* Zero → AGI · Chapter 12 · Inference and agents: the model at work
   DESIGN RULE: the reader changes a model's personality with a slider in the first ten seconds,
   without touching a weight.
   Interactives, in order: sampling playground (temperature, top-k, top-p); KV-cache memory
   calculator; lost-in-the-middle recall curve; RAG retrieval with a deliberate miss; the
   think-act-observe agent loop; and a cost/latency lab with prompt caching. */
(function () {
  ZTA.registerChapter({
    id: '12-inference-and-agents',
    num: 12,
    part: 'III',
    title: 'Inference, tools and agents: how a model is actually used',
    tagline: 'A trained model is a frozen function. This is everything that happens between you pressing enter and an agent editing your codebase.',
    render(root, ctx) {
      const { h, p, section, sub, callout, ul, ol } = ctx;
      const C = ctx.colors;
      const FONT = '13px Inter, system-ui, sans-serif';
      const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
      const fmtGB = (v) => (v >= 100 ? Math.round(v) : v >= 10 ? v.toFixed(1) : v.toFixed(2)) + ' GB';
      const fmtInt = (v) => Math.round(v).toLocaleString('en-US');
      const clamp = ctx.clamp;

      /* ================================================================ */
      /* Interactive A: the sampling playground                            */
      /* ================================================================ */
      function samplingPlayground() {
        // A hand-authored next-token distribution for "The capital of Australia is".
        // p0 values sum to 1.000 exactly; logits are simply ln(p0), so a plain softmax at
        // temperature 1 with no top-k/top-p filtering reproduces them.
        const CANDS = [
          { tok: 'Canberra', p0: 0.520, col: C.green },
          { tok: 'Sydney', p0: 0.180, col: C.warn },
          { tok: 'a', p0: 0.080, col: C.muted },
          { tok: 'the', p0: 0.060, col: C.muted },
          { tok: 'located', p0: 0.050, col: C.muted },
          { tok: 'Melbourne', p0: 0.035, col: C.muted },
          { tok: 'home', p0: 0.020, col: C.muted },
          { tok: 'not', p0: 0.015, col: C.muted },
          { tok: 'Perth', p0: 0.012, col: C.muted },
          { tok: 'actually', p0: 0.010, col: C.muted },
          { tok: 'known', p0: 0.009, col: C.muted },
          { tok: 'Australia', p0: 0.009, col: C.muted },
        ];
        const N = CANDS.length;
        const LOGIT = CANDS.map((c) => Math.log(c.p0));
        const W = 720, H = 380;
        const [cv, g] = ctx.canvas(W, H);
        const S = { temp: 1.0, topK: N, topP: 1.0, probs: new Array(N).fill(1 / N), counts: new Array(N).fill(0), samples: 0 };

        function distribution() {
          let raw;
          if (S.temp <= 0.021) {
            // temperature 0: the limit of softmax(logit/T) as T→0 is a one-hot on the argmax.
            let best = 0; for (let i = 1; i < N; i++) if (LOGIT[i] > LOGIT[best]) best = i;
            raw = new Array(N).fill(0); raw[best] = 1;
          } else {
            const scaled = LOGIT.map((l) => l / S.temp);
            const m = Math.max(...scaled);
            const ex = scaled.map((v) => Math.exp(v - m));
            const sum = ex.reduce((a, b) => a + b, 0) || 1;
            raw = ex.map((v) => v / sum);
          }
          // top-k: keep the K highest-probability tokens.
          const order = raw.map((v, i) => i).sort((a, b) => raw[b] - raw[a]);
          const kept = new Set(order.slice(0, Math.max(1, Math.round(S.topK))));
          // top-p (nucleus): within the kept set, keep the smallest prefix (by probability,
          // highest first) whose cumulative mass reaches topP.
          let cum = 0; const nucleus = new Set();
          for (const i of order) {
            if (!kept.has(i)) continue;
            nucleus.add(i); cum += raw[i];
            if (cum >= S.topP - 1e-9) break;
          }
          const finalSet = nucleus.size ? nucleus : kept;
          let total = 0; const out = new Array(N).fill(0);
          for (const i of finalSet) { out[i] = raw[i]; total += raw[i]; }
          if (total <= 0) { out[order[0]] = 1; total = 1; }
          for (let i = 0; i < N; i++) out[i] /= total;
          return out;
        }

        function entropyBits(pr) {
          let e = 0; for (const v of pr) if (v > 0) e -= v * Math.log2(v);
          return e;
        }

        function draw() {
          g.clearRect(0, 0, W, H);
          g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
          const rowH = (H - 20) / N, x0 = 128, barMax = 440;
          g.font = FONT; g.textBaseline = 'middle';
          for (let i = 0; i < N; i++) {
            const y = 10 + i * rowH + rowH / 2;
            const pr = S.probs[i];
            const w = pr * barMax;
            g.fillStyle = '#0f1520'; g.fillRect(x0, y - rowH * 0.32, barMax, rowH * 0.64);
            g.fillStyle = CANDS[i].col; g.globalAlpha = pr > 0 ? 0.92 : 0.25; g.fillRect(x0, y - rowH * 0.32, Math.max(1, w), rowH * 0.64);
            g.globalAlpha = 1;
            if (S.samples > 0) {
              const cw = (S.counts[i] / S.samples) * barMax;
              g.strokeStyle = C.text; g.lineWidth = 2;
              g.beginPath(); g.moveTo(x0 + cw, y - rowH * 0.32); g.lineTo(x0 + cw, y + rowH * 0.32); g.stroke();
            }
            g.textAlign = 'right'; g.fillStyle = C.text; g.fillText(CANDS[i].tok, x0 - 10, y);
            g.textAlign = 'left'; g.fillStyle = C.muted;
            let label = (pr * 100).toFixed(1) + '%';
            if (S.samples > 0) label += '   (' + S.counts[i] + '/' + S.samples + ' sampled)';
            g.fillText(label, x0 + barMax + 10, y);
          }
          g.textAlign = 'left'; g.fillStyle = C.muted; g.font = MONO;
          g.fillText('white tick = share of the 20 samples that landed on this token', 10, H - 6);
        }

        function recompute() { S.probs = distribution(); S.counts = new Array(N).fill(0); S.samples = 0; updateRO(); draw(); }

        function sampleOnce() {
          const r = Math.random(); let c = 0;
          for (let i = 0; i < N; i++) { c += S.probs[i]; if (r <= c) return i; }
          return N - 1;
        }

        const ro = ctx.readout();
        function updateRO() {
          const pSyd = S.probs[1], pCan = S.probs[0];
          const kept = S.probs.filter((v) => v > 0).length;
          ro.set({
            'P(Canberra)': (pCan * 100).toFixed(1) + '%',
            'P(Sydney)': (pSyd * 100).toFixed(1) + '%',
            'tokens still possible': kept,
            'entropy': entropyBits(S.probs).toFixed(2) + ' bits',
          });
        }

        const tSl = ctx.slider({ label: 'temperature T', min: 0, max: 2, step: 0.05, value: 1, digits: 2, onChange: (v) => { S.temp = v; recompute(); } });
        const kSl = ctx.slider({ label: 'top-k', min: 1, max: N, step: 1, value: N, onChange: (v) => { S.topK = v; recompute(); } });
        const pSl = ctx.slider({ label: 'top-p (nucleus)', min: 0.05, max: 1, step: 0.05, value: 1, digits: 2, onChange: (v) => { S.topP = v; recompute(); } });
        const sampleBtn = ctx.button('Sample 20×', () => {
          for (let i = 0; i < 20; i++) S.counts[sampleOnce()]++;
          S.samples += 20; updateRO(); draw();
        }, 'primary');
        const clearBtn = ctx.button('Clear samples', () => { S.counts = new Array(N).fill(0); S.samples = 0; draw(); updateRO(); });

        recompute();
        return ctx.figure(cv, 'The model’s real next-token distribution after "The capital of Australia is" (hand-authored, but shaped like a real model’s: one clearly correct answer, one common popular-misconception runner-up, a long thin tail). Bars show the probability <i>after</i> temperature, top-k and top-p have been applied — that is what actually gets sampled from.', [tSl, kSl, pSl, sampleBtn, clearBtn], ro);
      }

      /* ================================================================ */
      /* Interactive B: KV-cache & inference-cost calculator               */
      /* ================================================================ */
      function kvCalculator() {
        const BYTES = { fp16: 2, int8: 1, int4: 0.5 };
        const H100_GB = 80;
        const PRESETS = {
          '7b': { layers: 32, dModel: 4096, label: '~7B dense (e.g. Llama-2-7B shape)' },
          '13b': { layers: 40, dModel: 5120, label: '~13B dense' },
          '70b': { layers: 80, dModel: 8192, label: '~70B dense' },
          'moe': { layers: 120, dModel: 12288, label: 'Frontier-scale MoE, ≈217B <i>active</i> params (est.)' },
        };
        const W = 720, H = 210;
        const [cv, g] = ctx.canvas(W, H);
        const S = { layers: 32, dModel: 4096, ctxExp: 13, batch: 1, prec: 'fp16' };

        function compute() {
          const ctxTok = Math.round(Math.pow(2, S.ctxExp));
          const bpv = BYTES[S.prec];
          // Standard transformer parameter-count approximation (ignores embeddings): 12 * L * d^2.
          const params = 12 * S.layers * S.dModel * S.dModel;
          const weightGB = (params * bpv) / (1024 ** 3);
          const kvPerSeqBytes = 2 * S.layers * S.dModel * ctxTok * bpv; // ×2 for Key and Value
          const kvTotalGB = (kvPerSeqBytes * S.batch) / (1024 ** 3);
          const totalGB = weightGB + kvTotalGB;
          const leftoverBytes = Math.max(0, H100_GB - weightGB) * (1024 ** 3);
          const maxBatchHere = kvPerSeqBytes > 0 ? Math.floor(leftoverBytes / kvPerSeqBytes) : 0;
          return { ctxTok, params, weightGB, kvTotalGB, totalGB, maxBatchHere, kvPerSeqBytes };
        }

        function draw(R) {
          g.clearRect(0, 0, W, H); g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
          const x0 = 16, x1 = W - 16, barY = 60, barH = 34;
          const maxShown = Math.max(H100_GB * 2, R.totalGB * 1.12);
          const px = (x1 - x0) / maxShown;
          g.fillStyle = '#0f1520'; g.fillRect(x0, barY, x1 - x0, barH);
          const wWeight = R.weightGB * px, wKv = R.kvTotalGB * px;
          g.fillStyle = C.accent; g.fillRect(x0, barY, Math.min(x1 - x0, wWeight), barH);
          g.fillStyle = C.orange; g.fillRect(x0 + Math.min(x1 - x0, wWeight), barY, Math.max(0, Math.min(x1 - x0 - wWeight, wKv)), barH);
          const tickX = x0 + H100_GB * px;
          g.strokeStyle = C.danger; g.lineWidth = 2; g.setLineDash([5, 4]);
          g.beginPath(); g.moveTo(tickX, barY - 14); g.lineTo(tickX, barY + barH + 14); g.stroke(); g.setLineDash([]);
          g.fillStyle = C.danger; g.font = FONT; g.textAlign = 'center';
          g.fillText('1× H100 — 80 GB', tickX, barY - 22);
          g.font = MONO; g.fillStyle = C.text; g.textAlign = 'left';
          g.fillText('weights: ' + fmtGB(R.weightGB), x0, barY + barH + 30);
          g.fillStyle = C.orange;
          g.fillText('KV cache (all requests): ' + fmtGB(R.kvTotalGB), x0, barY + barH + 48);
          g.fillStyle = C.muted; g.textAlign = 'right';
          g.fillText('total: ' + fmtGB(R.totalGB) + (R.totalGB > H100_GB ? '  (' + (R.totalGB / H100_GB).toFixed(1) + '× an H100)' : ''), x1, barY + barH + 30);
        }

        const ro = ctx.readout();
        function recompute() {
          const R = compute();
          draw(R);
          ro.set({
            'params (approx.)': (R.params / 1e9).toFixed(1) + 'B',
            'context': fmtInt(R.ctxTok) + ' tok',
            'KV / sequence': fmtGB(R.kvPerSeqBytes / (1024 ** 3)),
            'max concurrent requests @ this length on 1 H100': R.maxBatchHere <= 0 ? '0 — weights alone don’t fit' : fmtInt(R.maxBatchHere),
          });
        }

        const presetSel = ctx.select({
          label: 'model preset', value: '7b',
          options: [{ value: 'custom', label: 'Custom (use sliders)' }, ...Object.entries(PRESETS).map(([k, v]) => ({ value: k, label: v.label.replace(/<[^>]+>/g, '') }))],
          onChange: (v) => { const pr = PRESETS[v]; if (pr) { layersSl.value = pr.layers; dModelSl.value = pr.dModel; S.layers = pr.layers; S.dModel = pr.dModel; recompute(); } },
        });
        const layersSl = ctx.slider({ label: 'layers', min: 4, max: 140, step: 2, value: 32, onChange: (v) => { S.layers = v; recompute(); } });
        const dModelSl = ctx.slider({ label: 'd_model', min: 256, max: 16384, step: 128, value: 4096, onChange: (v) => { S.dModel = v; recompute(); } });
        const ctxSl = ctx.slider({ label: 'context length', min: 9, max: 18, step: 1, value: 13, fmt: (v) => fmtInt(Math.pow(2, v)) + ' tok', onChange: (v) => { S.ctxExp = v; recompute(); } });
        const batchSl = ctx.slider({ label: 'concurrent requests (batch)', min: 1, max: 64, step: 1, value: 1, onChange: (v) => { S.batch = v; recompute(); } });
        const precSel = ctx.select({ label: 'precision', value: 'fp16', options: [{ value: 'fp16', label: 'fp16 (2 bytes/weight)' }, { value: 'int8', label: 'int8 (1 byte/weight)' }, { value: 'int4', label: 'int4 (0.5 byte/weight)' }], onChange: (v) => { S.prec = v; recompute(); } });

        recompute();
        const costTable = ctx.table(
          ['Model class', 'Typical serving hardware', 'Tokens/s (single request)', '$ / million tokens'],
          [
            ['7B, int4 quantized', 'laptop CPU/GPU', '≈15–30', '$0 (local) / ≈$0.05–$0.15 hosted'],
            ['7B, fp16', '1 mid GPU or a slice of an H100', '≈80–150', '≈$0.10–$0.30'],
            ['70B, fp16', 'multi-GPU node', '≈25–45', '≈$0.50–$2'],
            ['Frontier MoE (trillion+ params, ~100–300B active)', 'multi-node datacenter cluster', '≈40–80 (higher for shorter, cheaper tiers)', '≈$1–$20+ depending on tier'],
          ]
        );
        return h('div', {},
          ctx.figure(cv, 'Live for the sliders above: the blue segment is model-weight memory, the orange segment is total KV-cache memory across every concurrent request, and the red dashed line marks one H100’s 80 GB. Longer context and more concurrent users both push the orange segment right — that is the entire cost story of serving a model.', [presetSel, layersSl, dModelSl, ctxSl, batchSl, precSel], ro),
          p('<b>Rough, order-of-magnitude only</b> — real throughput and pricing depend on the specific hardware, software stack, batching strategy and provider margin, and change every few months. Treat this table as "which order of magnitude", not a quote:'),
          costTable,
        );
      }

      /* ================================================================ */
      /* Interactive C: agent loop simulator                                */
      /* ================================================================ */
      function agentSimulator() {
        const WINDOW = 4000; // a small illustrative window so the bar visibly fills over 5 steps
        const BASE = 760; // tokens "already spent" on a system prompt + repo file listing
        const STEPS = [
          {
            thought: 'The test suite reports test_apply_discount failing. Let me open the test to see exactly what it expects.',
            call: { tool: 'read_file', args: { path: 'tests/test_pricing.py' } },
            result: 'def test_apply_discount():\n    price = apply_discount(100, 0.2)\n    assert price == 80',
            noTools: 'The test test_apply_discount is failing. I haven’t opened the file, but going by the name it’s probably in tests/test_pricing.py and probably checks that a 20% discount on 100 gives 80. I’ll assume that and move on.',
          },
          {
            thought: 'Now let’s look at the function it’s testing.',
            call: { tool: 'read_file', args: { path: 'src/pricing.py' } },
            result: 'def apply_discount(price, pct):\n    return price + price * pct',
            noTools: 'I’ll guess the implementation without reading it: a typical apply_discount probably does "return price * (1 - pct)". That looks fine to me, so the bug must be somewhere else — maybe the test itself is wrong.',
          },
          {
            thought: 'Found it: the discount is being <b>added</b>, not subtracted — price + price·pct makes the price go up. It should be price × (1 − pct). Applying the fix.',
            call: { tool: 'edit_file', args: { path: 'src/pricing.py', find: 'return price + price * pct', replace: 'return price * (1 - pct)' } },
            result: 'src/pricing.py updated (1 line changed).',
            noTools: 'Since I never saw the real code, I’ll instead "fix" the test’s expected value from 80 to 120 — without any way to know whether that’s actually correct.',
          },
          {
            thought: 'Let’s rerun the test to make sure the fix actually works, not just that it looks right.',
            call: { tool: 'run_tests', args: { path: 'tests/test_pricing.py' } },
            result: '1 passed in 0.03s',
            noTools: 'I have no way to run the tests without a tool. I’ll just state that it should pass now.',
          },
          {
            thought: 'All green. apply_discount now subtracts the discount instead of adding it. Done.',
            call: null,
            result: null,
            noTools: 'Done — though I never actually read the file or ran anything, so this is a guess dressed up as a report.',
          },
        ];
        const S = { shown: 0, tools: true };
        const trace = h('div', { style: { padding: '14px', display: 'flex', flexDirection: 'column', gap: '10px' } });
        const barOuter = h('div', { style: { height: '10px', borderRadius: '6px', background: '#0f1520', border: '1px solid #243044', overflow: 'hidden' } });
        const barFill = h('div', { style: { height: '100%', width: '0%', background: 'linear-gradient(90deg,#7c9cff,#38d9a9)' } });
        barOuter.append(barFill);
        const tokLabel = h('div', { class: 'muted', style: { fontFamily: 'var(--mono)', fontSize: '.8rem', marginTop: '4px' } });
        const pill = (text, color) => h('span', { class: 'pill', style: { color, borderColor: color } }, text);

        function approxTokens(s) { return Math.round(String(s).length / 4); }

        function render() {
          trace.innerHTML = '';
          let tokens = BASE;
          for (let i = 0; i < S.shown; i++) {
            const st = STEPS[i];
            if (S.tools) {
              const callTxt = st.call ? JSON.stringify(st.call) : '';
              tokens += approxTokens(st.thought) + approxTokens(callTxt) + approxTokens(st.result || '');
              const row = h('div', { style: { borderLeft: '3px solid ' + C.accent, paddingLeft: '10px' } },
                h('div', {}, pill('THOUGHT', C.accent), ' ', h('span', { html: st.thought })));
              if (st.call) row.append(h('div', { style: { marginTop: '4px' } }, pill('TOOL CALL', C.orange), ' ', h('code', { class: 'inline' }, JSON.stringify(st.call))));
              if (st.result) row.append(h('div', { style: { marginTop: '4px' } }, pill('RESULT', C.green), ' ', h('pre', { class: 'code', style: { margin: '4px 0 0' } }, h('code', {}, st.result))));
              trace.append(row);
            } else {
              tokens += approxTokens(st.noTools);
              trace.append(h('div', { style: { borderLeft: '3px solid ' + C.danger, paddingLeft: '10px' } },
                h('div', {}, pill('THOUGHT (no tools)', C.danger), ' ', h('span', { html: st.noTools })),
                h('div', { class: 'muted', style: { fontSize: '.82rem', marginTop: '4px' } }, '⚠ nothing here was read or executed — it is invented, and reads exactly as confidently as the real trace on the left.')));
            }
          }
          const pct = clamp((tokens / WINDOW) * 100, 0, 100);
          barFill.style.width = pct.toFixed(1) + '%';
          barFill.style.background = pct > 85 ? C.danger : pct > 55 ? C.warn : 'linear-gradient(90deg,#7c9cff,#38d9a9)';
          tokLabel.textContent = tokens + ' / ' + WINDOW + ' tokens in context (illustrative small window, ≈' + pct.toFixed(0) + '%)';
        }

        const stepBtn = ctx.button('Step ▶', () => { S.shown = Math.min(STEPS.length, S.shown + 1); render(); }, 'primary');
        const resetBtn = ctx.button('Reset', () => { S.shown = 0; render(); });
        const toggleBtn = ctx.button('Tools: ON (click to turn off)', function () {
          S.tools = !S.tools; S.shown = 0;
          this.textContent = S.tools ? 'Tools: ON (click to turn off)' : 'Tools: OFF — hallucinating (click to turn on)';
          render();
        });
        render();
        return ctx.figure(h('div', {}, trace, h('div', { style: { padding: '0 14px 14px' } }, barOuter, tokLabel)),
          'A scripted, deterministic trace — press Step to reveal each turn of the think → act → observe loop. With tools on, every claim about the code is backed by an actual read/run; with tools off, the model still talks like it checked, but every detail past step 1 is fabricated.',
          [stepBtn, resetBtn, toggleBtn]);
      }

      /* ================================================================ */
      /* Interactive D: tiny TF-IDF RAG demo                                 */
      /* ================================================================ */
      function ragDemo() {
        const DOCS = [
          { title: 'Vacation policy', text: 'Employees accrue 15 vacation days of paid time off per year, plus 10 public holidays. Unused vacation days roll over, up to 5 days, into the next calendar year.' },
          { title: 'Expense reports', text: 'Submit receipts within 30 days of purchase through the expense portal. Reimbursement for approved expense reports is processed within two pay cycles.' },
          { title: 'Remote work', text: 'Employees may work remotely up to 3 days per week with manager approval. Fully remote roles are marked separately in the HR system and are not limited to 3 days.' },
          { title: 'Parental leave', text: 'New parents receive 16 weeks of paid parental leave, usable within the first 12 months after a birth or adoption, regardless of remote or office work location.' },
          { title: 'Health benefits', text: 'The company covers 90 percent of premiums for the base medical plan. Dental and vision are optional add-ons chosen during open enrollment.' },
          { title: 'Equipment & onboarding', text: 'New hires receive a laptop and a one-time 500 dollar home-office equipment stipend during their first week of onboarding.' },
          { title: 'Performance reviews', text: 'Formal performance reviews happen twice a year, in March and September, with calibration meetings across teams before ratings are finalized.' },
          { title: 'Travel policy', text: 'Domestic flights should be booked economy class. International flights over 6 hours may be booked premium economy with director approval.' },
          { title: 'Security policy', text: 'All laptops must have full-disk encryption and an approved password manager installed within the first week of employment.' },
          { title: 'Learning budget', text: 'Each employee has an annual 1000 dollar learning and conference budget, approved by their manager and expensed like any other purchase.' },
        ];
        const Ndocs = DOCS.length;
        function tokenize(s) { return String(s).toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(Boolean); }
        const docTf = DOCS.map((d) => { const tf = {}; for (const t of tokenize(d.text)) tf[t] = (tf[t] || 0) + 1; return tf; });
        const df = {};
        docTf.forEach((tf) => { for (const t of Object.keys(tf)) df[t] = (df[t] || 0) + 1; });
        function idf(t) { return Math.log((Ndocs + 1) / ((df[t] || 0) + 1)) + 1; }
        function highlight(text, termSet) {
          return tokenize(text).length ? text.replace(/[a-zA-Z0-9]+/g, (w) => (termSet.has(w.toLowerCase()) ? '<mark style="background:rgba(251,191,36,.35);color:inherit;border-radius:3px;padding:0 2px;">' + w + '</mark>' : w)) : text;
        }

        const resultsEl = h('div', {});
        const promptEl = h('pre', { class: 'code' }, h('code', {}));
        const noteEl = h('div', { class: 'muted', style: { fontSize: '.85rem', margin: '6px 0 0' } });

        function runQuery(qtext) {
          const terms = Array.from(new Set(tokenize(qtext)));
          const termSet = new Set(terms);
          const scores = DOCS.map((d, i) => {
            let s = 0; for (const t of terms) s += (docTf[i][t] || 0) * idf(t);
            return { i, score: s };
          }).sort((a, b) => b.score - a.score);
          const top3 = scores.slice(0, 3);
          resultsEl.innerHTML = '';
          if (top3.every((r) => r.score === 0)) {
            noteEl.innerHTML = 'No document shares a single word with this question. A real embedding retriever would likely return three low-relevance documents here too — this is exactly the case where a model should say "I don’t know" instead of confidently blending unrelated context.';
          } else {
            noteEl.innerHTML = 'Highlighted words are query terms found in that document — that overlap (weighted by how rare each word is across the 10 documents) is the entire ranking signal.';
          }
          top3.forEach((r) => {
            const d = DOCS[r.i];
            resultsEl.append(h('div', { class: 'card', style: { marginBottom: '10px' } },
              h('h4', {}, d.title + '  ', h('span', { class: 'pill' }, 'score ' + r.score.toFixed(2))),
              h('p', { html: highlight(d.text, termSet) })));
          });
          const stuffed = '[SYSTEM] Answer using ONLY the context below. If it is not there, say you do not know.\n\n' +
            top3.map((r) => '[' + DOCS[r.i].title + '] ' + DOCS[r.i].text).join('\n\n') +
            '\n\n[USER] ' + qtext;
          promptEl.querySelector('code').textContent = stuffed;
        }

        const q = ctx.textarea({ label: 'Your question', value: 'How many vacation days do I get?', onChange: (v) => runQuery(v) });
        const b1 = ctx.button('Try: vacation days', () => { q.value = 'How many vacation days do I get?'; runQuery(q.value); });
        const b2 = ctx.button('Try: parental leave', () => { q.value = 'What is the parental leave policy?'; runQuery(q.value); });
        const b3 = ctx.button('Try: pets at the office (no match)', () => { q.value = 'Can I bring my dog to the office?'; runQuery(q.value); });
        runQuery(q.value);
        return ctx.figure(h('div', { style: { padding: '14px', display: 'flex', flexDirection: 'column', gap: '10px' } },
          resultsEl, h('div', {}, h('div', { class: 'muted', style: { fontSize: '.8rem', marginBottom: '4px' } }, 'What actually gets stuffed into the prompt:'), promptEl), noteEl),
          'A 10-document toy company handbook, ranked live by a plain TF-IDF score (term frequency × inverse document frequency, no length normalization — real systems refine this further, and most production RAG uses embeddings instead of word overlap, but the shape of the failure mode is the same).',
          [q, b1, b2, b3]);
      }


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

      /* ================================================================ */
      /* Interactive: lost in the middle                                   */
      /* ================================================================ */
      function lostInMiddle() {
        const [cv, g] = ctx.canvas(720, 330);
        let pos = 0.5, ctxLen = 30, model = 'typical';
        /* U-shaped recall: strong at the edges, sagging in the middle, and the sag deepens
           as the context grows. Shaped after the Liu et al. (2023) curves, not measured here. */
        const MODELS = { typical: { edge: 0.94, dip: 0.55 }, strong: { edge: 0.97, dip: 0.78 } };
        function recall(p, n) {
          const m = MODELS[model];
          const lenPenalty = ctx.clamp((n - 10) / 90, 0, 1);
          const floor = m.dip + (1 - lenPenalty) * (m.edge - m.dip) * 0.55;
          const u = Math.pow(Math.abs(p - 0.5) * 2, 1.7);       // 0 at middle, 1 at either edge
          return ctx.clamp(floor + (m.edge - floor) * u, 0, 1);
        }
        const pSl = ctx.slider({ label: 'where the answer is buried', min: 0, max: 1, step: 0.01, value: 0.5, digits: 2, onChange: (v) => { pos = v; } });
        const nSl = ctx.slider({ label: 'documents in the context', min: 5, max: 100, step: 1, value: 30, onChange: (v) => { ctxLen = v; } });
        const mBtn = ctx.button('a typical model', () => {
          model = model === 'typical' ? 'strong' : 'typical';
          mBtn.textContent = model === 'typical' ? 'a typical model' : 'a stronger model';
        }, 'primary');
        const startBtn = ctx.button('put it first', () => { pos = 0; pSl.value = 0; });
        const midBtn = ctx.button('put it in the middle', () => { pos = 0.5; pSl.value = 0.5; });
        const endBtn = ctx.button('put it last', () => { pos = 1; pSl.value = 1; });
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, 720, 330);
          /* the context, drawn as a row of documents */
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('the model\'s context: ' + ctxLen + ' documents, one of which has the answer', 34, 26);
          const BX = 34, BW = 650, BH = 38;
          const cw = BW / ctxLen;
          const hit = Math.min(ctxLen - 1, Math.round(pos * (ctxLen - 1)));
          for (let i = 0; i < ctxLen; i++) {
            g.fillStyle = i === hit ? C.green : '#161d2b';
            g.fillRect(BX + i * cw, 40, Math.max(1, cw - 1), BH);
          }
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(BX, 40, BW, BH);
          g.font = MONO; g.fillStyle = C.green;
          g.fillText('▲ the answer is here', ctx.clamp(BX + hit * cw - 40, BX, BX + BW - 130), 94);

          /* the recall curve */
          const P = { x: 60, y: 126, w: 400, h: 150 };
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('how often the model finds it', P.x, P.y - 10);
          g.strokeStyle = C.line; g.strokeRect(P.x, P.y, P.w, P.h);
          const px = (v) => P.x + v * P.w;
          const py = (v) => P.y + P.h - ctx.clamp((v - 0.4) / 0.62, 0, 1) * P.h;
          g.font = MONO; g.fillStyle = C.muted;
          [0.5, 0.75, 1].forEach(v => {
            g.beginPath(); g.moveTo(P.x, py(v)); g.lineTo(P.x + P.w, py(v)); g.stroke();
            g.fillText((v * 100).toFixed(0) + '%', P.x - 32, py(v) + 4);
          });
          g.strokeStyle = C.accent; g.lineWidth = 2.5; g.beginPath();
          for (let i = 0; i <= 60; i++) { const v = i / 60; i ? g.lineTo(px(v), py(recall(v, ctxLen))) : g.moveTo(px(v), py(recall(v, ctxLen))); }
          g.stroke();
          const r0 = recall(pos, ctxLen);
          g.fillStyle = r0 > 0.85 ? C.green : r0 > 0.65 ? C.warn : C.danger;
          g.beginPath(); g.arc(px(pos), py(r0), 7, 0, 7); g.fill();
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('start', P.x, P.y + P.h + 18);
          g.fillText('middle', P.x + P.w / 2 - 18, P.y + P.h + 18);
          g.fillText('end', P.x + P.w - 22, P.y + P.h + 18);

          const TX = 500;
          g.font = FONT; g.fillStyle = C.muted; g.fillText('found, at this position', TX, 150);
          g.font = 'bold 30px Inter, system-ui, sans-serif';
          g.fillStyle = r0 > 0.85 ? C.green : r0 > 0.65 ? C.warn : C.danger;
          g.fillText((r0 * 100).toFixed(0) + '%', TX, 186);
          const best = Math.max(recall(0, ctxLen), recall(1, ctxLen));
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('at the edges: ' + (best * 100).toFixed(0) + '%', TX, 210);
          g.font = FONT;
          g.fillStyle = (best - r0) > 0.1 ? C.danger : C.muted;
          wrapText(g, (best - r0) > 0.1
            ? 'A ' + ((best - r0) * 100).toFixed(0) + '-point penalty for nothing but where the text sat.'
            : 'Near an edge, where recall is reliable.', TX, 232, 190, 16);
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, 'Doubling the window does not double how much of it gets used. Mostly it means more middle to get lost in — which is why retrieval systems put the best match first or last, not wherever it happened to rank.',
            34, 300, 650, 17);
          ro.set({ position: pos < 0.15 ? 'near the start' : pos > 0.85 ? 'near the end' : 'buried in the middle', documents: ctxLen, 'found': (r0 * 100).toFixed(0) + '%' });
        });

        return ctx.figure(cv,
          'Liu and colleagues tested this in 2023: they placed the answer to a question at different points inside a long context and measured how often models found it. Accuracy was highest at the very start or the very end and sagged in the middle — a U-shaped curve, echoing how people recall a list best from its beginning and end. The curve here is shaped after their published results rather than measured, but the effect is real, robust, and the reason a huge advertised context window is not the same as a usable one.',
          [pSl, nSl, startBtn, midBtn, endBtn, mBtn], ro);
      }

      /* ================================================================ */
      /* Interactive: what a reply actually costs                          */
      /* ================================================================ */
      function costLab() {
        const [cv, g] = ctx.canvas(720, 340);
        let promptTok = 4000, outTok = 600, cacheHit = false, calls = 1000;
        const IN_PRICE = 3.0, OUT_PRICE = 15.0;      // $ per million tokens, illustrative
        const CACHE_READ = 0.1;                       // cached input costs ~10% of full price
        const PREFILL_TPS = 12000, DECODE_TPS = 70;   // tokens/sec, illustrative

        const pSl = ctx.slider({ label: 'prompt tokens', min: 200, max: 100000, step: 200, value: 4000, onChange: (v) => { promptTok = v; } });
        const oSl = ctx.slider({ label: 'output tokens', min: 50, max: 4000, step: 50, value: 600, onChange: (v) => { outTok = v; } });
        const cSl = ctx.slider({ label: 'calls per day', min: 1, max: 100000, step: 100, value: 1000, onChange: (v) => { calls = v; } });
        const cacheBtn = ctx.button('prompt caching: off', () => {
          cacheHit = !cacheHit;
          cacheBtn.textContent = 'prompt caching: ' + (cacheHit ? 'on' : 'off');
        }, 'primary');
        const agentBtn = ctx.button('an agent turn (big prompt, small reply)', () => {
          promptTok = 60000; pSl.value = 60000; outTok = 200; oSl.value = 200;
        });
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, 720, 340);
          const inCost = promptTok / 1e6 * IN_PRICE * (cacheHit ? CACHE_READ : 1);
          const outCost = outTok / 1e6 * OUT_PRICE;
          const per = inCost + outCost;
          const prefillS = promptTok / PREFILL_TPS * (cacheHit ? 0.05 : 1);
          const decodeS = outTok / DECODE_TPS;

          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('where the time goes, for one reply', 34, 26);
          const TOT = prefillS + decodeS, BW = 620;
          const pw = TOT ? prefillS / TOT * BW : 0;
          g.fillStyle = C.accent; g.fillRect(34, 40, pw, 30);
          g.fillStyle = C.purple; g.fillRect(34 + pw, 40, BW - pw, 30);
          g.font = MONO; g.fillStyle = '#0a0e16';
          if (pw > 80) g.fillText('prefill ' + prefillS.toFixed(2) + 's', 42, 60);
          if (BW - pw > 90) g.fillText('decode ' + decodeS.toFixed(2) + 's', 42 + pw, 60);
          g.fillStyle = C.muted;
          g.fillText('prefill reads the whole prompt in one parallel pass; decode emits one token at a time', 34, 88);
          g.font = 'bold 16px Inter, system-ui, sans-serif'; g.fillStyle = C.text;
          g.fillText('total ' + TOT.toFixed(2) + ' s  —  ' + (decodeS / Math.max(1e-6, TOT) * 100).toFixed(0) + '% of it spent one token at a time', 34, 112);

          /* cost breakdown */
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('what it costs', 34, 150);
          const bar = (lab, v, col, y, note) => {
            g.font = MONO; g.fillStyle = C.muted; g.fillText(lab, 34, y + 12);
            g.fillStyle = C.line; g.fillRect(200, y, 300, 16);
            g.fillStyle = col; g.fillRect(200, y, ctx.clamp(v / Math.max(per, 1e-9), 0, 1) * 300, 16);
            g.font = 'bold ' + MONO; g.fillStyle = col;
            g.fillText('$' + v.toFixed(5), 512, y + 13);
            if (note) { g.font = MONO; g.fillStyle = C.muted; g.fillText(note, 596, y + 13); }
            return y + 30;
          };
          let y = 162;
          y = bar('input tokens', inCost, C.accent, y, cacheHit ? '(cached)' : '');
          y = bar('output tokens', outCost, C.purple, y, '5× the rate');
          g.strokeStyle = C.line; g.beginPath(); g.moveTo(34, y + 2); g.lineTo(686, y + 2); g.stroke();
          g.font = MONO; g.fillStyle = C.muted; g.fillText('per reply', 34, y + 26);
          g.font = 'bold 18px Inter, system-ui, sans-serif'; g.fillStyle = C.text;
          g.fillText('$' + per.toFixed(4), 200, y + 28);
          g.font = MONO; g.fillStyle = C.muted; g.fillText('× ' + calls.toLocaleString() + ' calls/day', 300, y + 26);
          g.font = 'bold 20px Inter, system-ui, sans-serif'; g.fillStyle = C.green;
          g.fillText('$' + (per * calls).toFixed(2) + ' / day', 470, y + 28);

          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, cacheHit
            ? 'Prompt caching reuses the KV cache for a prefix you send repeatedly — a system prompt, a shared document — so the expensive prefill is paid once and later calls pay a fraction.'
            : 'Output tokens cost several times more than input tokens, because decode is sequential and prefill is not. That single asymmetry drives most of the cost of running an assistant.',
            34, y + 56, 650, 17);
          ro.set({ 'per reply': '$' + per.toFixed(4), 'per day': '$' + (per * calls).toFixed(2), latency: TOT.toFixed(2) + 's', caching: cacheHit ? 'on' : 'off' });
        });

        return ctx.figure(cv,
          'Illustrative rates, in the shape providers actually bill: a few dollars per million input tokens and several times that for output, because decode is sequential while prefill is one parallel pass. Press <b>an agent turn</b> to see the regime that surprises people — an agent resends its whole growing transcript every step, so its prompts are enormous and its replies tiny, and without prompt caching it pays full price for the same prefix over and over. Real prices change every few months; the shape does not.',
          [pSl, oSl, cSl, cacheBtn, agentBtn], ro);
      }

      /* ================================================================ */
      /* Layout                                                             */
      /* ================================================================ */
      root.append(
        callout('tryit', '🖐 Do this first — make the model say the wrong thing',
          `The bars are a model's next-word distribution after <b>"The capital of Australia is"</b>. Canberra is correct. Sydney is the answer most people would guess.<br>
           <b>1.</b> At the defaults, press <b>Sample 20×</b>. Canberra wins nearly every time.<br>
           <b>2.</b> Now raise <b>temperature</b> and sample again. Somewhere around 1.3 Sydney starts winning — <b>the model has not changed at all</b>, only how you are drawing from it.<br>
           <b>3.</b> Set temperature back and pull <b>top-k</b> down to 1. Now it is frozen on one answer forever, however high you push the temperature afterwards.<br>
           <b>4.</b> That is the entire difference between a model that feels creative and one that feels reliable, and it is a dial, not a retraining run.`),
        samplingPlayground(),
        p(`Nothing about the weights moved. You changed how a token is drawn from a fixed list of probabilities, and got a different personality.`),

        p(`Every previous chapter was about producing one thing: a set of weights. This chapter is about the other 99% of what you experience as "AI" — the moment those frozen weights are handed a question and have to produce an answer, in real time, on real hardware, for millions of people at once, sometimes wired up to run code with no one watching.`),
        p(`That handing-over is called <em>inference</em>, to distinguish it from training. Nothing learns; not one weight changes. And yet almost everything that makes one model feel different from another day to day — how random it feels, how it handles a huge codebase, how fast and cheap it is, whether it can use a calendar, whether it lies to you confidently — is decided here, not during training.`),
        p(`This chapter follows one query all the way through: from a list of probabilities over every possible next word, to a chatbot's reply, to an agent that reads a failing test, edits the file, reruns it, and reports back that it's fixed.`),

        section('From logits to a word: sampling',
          p(`At every step, a language model does one thing: given the tokens so far, it outputs one number for every token in its vocabulary (upwards of 100,000 of them). These raw numbers are <em>logits</em> — any real number, positive or negative, meaning nothing by themselves. <em>Softmax</em> turns them into something you can pick from: exponentiate every logit, then divide each by the sum of all the exponentials. The results are all positive and add to exactly 1 — a genuine probability distribution over "what word comes next."`),
          p(`<em>Temperature</em> is a dial applied before that: divide every logit by T before taking softmax. T = 1 leaves the raw distribution alone. T below 1 sharpens it, making the favourite token even more dominant; T above 1 flattens it, giving weaker candidates a real chance. T = 0 is the limit of that process — always the single highest-probability token, no randomness at all. That is <em>greedy decoding</em>.`),
          p(`Try three round logits: 2.0, 1.0 and 0.0. At T = 1, softmax gives them 66.5%, 24.5% and 9.0%. Halve the temperature (equivalent to doubling the logits, to 4, 2, 0) and the split sharpens to 86.7% / 11.7% / 1.6%. Double it (halving the logits to 1, 0.5, 0) and it flattens to 50.6% / 30.7% / 18.6%. Same three logits, three different personalities.`),
          p(`Two more controls trim the distribution before a token is drawn. <em>Top-k</em> keeps only the k highest-probability tokens and renormalizes; k = 1 is identical to greedy. <em>Top-p</em> (nucleus sampling) instead keeps the smallest set whose probabilities add to at least p — one token when the model is confident, dozens when it is unsure. Both exist to stop pure temperature sampling from occasionally drawing nonsense out of a long, low-probability tail.`),
          p(`One wrinkle: temperature 0 is <i>not</i> a guarantee of bit-identical output across machines. Floating-point addition is not associative — (a + b) + c can differ from a + (b + c) in its last decimal place — and a GPU sums each layer's millions of terms in whatever order its threads finish, which depends on batch size and hardware.`),
          p(` Usually invisible, but when two logits sit closer than that rounding error, the "top" token flips, and one flip early on cascades into a different reply. Greedy decoding is deterministic <i>for one program on one machine with one fixed batch</i> — not a property of the model itself.`),
        ),

        section('Serving a model: the KV cache and the price of a long context',
          p(`Generating a reply happens in two phases. <em>Prefill</em> reads your whole prompt at once — every token processed in one big, parallel matrix multiplication — so a long prompt is "digested" almost instantly. <em>Decode</em> is what follows: the reply is produced one token at a time, each new token depending on every token before it, so the same computation repeats, single-file, for as long as the reply runs. Decode, not prefill, is why a long answer visibly takes time.`),
          p(`Naively, producing token 500 would mean recomputing attention over all 499 tokens before it from scratch — and token 501 over 500, redoing nearly all of the same work every step.`),
          p(` The <em>KV cache</em> avoids that: every layer stores the Key and Value vectors it computes for each token, once, and reuses them for every later step; decode then only computes one new token's own Key/Value pair. That reuse is what makes generation feel roughly linear in speed — but the cache is memory, and it grows with every token kept around.`),
          p(`Its size is roughly <code class="inline">2 × layers × d<sub>model</sub> × tokens × batch × bytes-per-value</code> (2 for Key and Value). A 7B-shaped model — 32 layers, d<sub>model</sub> = 4096, fp16 — at 8,000 tokens: 2×32×4096×8000×1×2 ≈ 4.2 billion bytes, about <b>4 GB</b> of cache. Stretch to 128,000 tokens (16× longer) and the cache grows 16× too, to roughly <b>67 GB</b> — most of an H100's 80 GB, before the model's own ~14 GB of weights even load. Long context is a standing memory bill, multiplied by every concurrent conversation.`),
          p(`Several tricks push that bill down:`),
          ul([
            `<b>Batching</b> processes many users' decode steps together in one pass — efficient for the GPU, but it multiplies the KV cache by however many requests are in flight.`,
            `<b>Speculative decoding</b> (Leviathan et al. and Chen et al., both 2023) has a small, cheap "draft" model guess several tokens ahead; the large model checks the whole guess in one parallel pass and keeps whatever prefix was right. A good draft yields the large model's exact output two to three times faster.`,
            `<b>Quantization</b> rounds the weights themselves after training: fp16 (2 bytes/weight) can become int8 (1 byte) or int4 (0.5 bytes) — why a 7B model needing ~14 GB at fp16 fits under 4 GB at int4, comfortably on a laptop, at some cost in quality (usually minor at int8, noticeable but often acceptable at int4).`,
            `<b>Distillation</b> trains a smaller "student" model to mimic a larger "teacher's" outputs instead of shrinking one model's numbers; most cheap "mini"/"flash" tiers are distilled relatives of a bigger sibling.`,
            `<b>Mixture-of-experts (MoE)</b> serving (chapter 8) trades memory for speed the other way: it holds far more total parameters than a same-quality dense model but activates only a fraction per token — more memory to hold everything, less compute per token than its size suggests.`,
          ]),
          callout('tryit', 'Try it: watch a laptop-sized model become a datacenter model', `<b>1.</b> Leave the preset at 7B, fp16, and slide context from 512 tokens up to 200K+: watch the orange KV segment swallow the chart and blow past the red H100 line. <b>2.</b> Switch precision to int4 with the same settings: the whole bar shrinks by roughly 4×. <b>3.</b> Pick the 70B preset, and watch "max concurrent requests" in the readout collapse toward 0 or 1 at long context — that number is the entire reason API providers cap context and charge more for it. <b>4.</b> Push batch up to 64 at a long context on the MoE preset: this is the regime real inference clusters live in.`),
          kvCalculator(),
        ),

        section('Context windows and the "lost in the middle" effect',
          p(`The <em>context window</em> is the maximum number of tokens — prompt, reply, instructions, everything — the model attends to at once; anything outside it doesn't exist to the model. Frontier models in 2026 advertise windows from 128,000 tokens to well over a million, but a bigger window is not the same as using every part of it equally well.`),
          p(`Liu et al. tested this in 2023 ("Lost in the Middle"): they placed the answer to a question at different positions inside a long context and measured how often models found it. Accuracy was highest with the answer at the very start or end, and dropped noticeably in the middle — a U-shaped curve, echoing how humans recall a list best from its beginning and end.`),
          p(` Doubling the window doesn't double how much of it gets reliably used; mostly it means more middle to get lost in.`),
          callout('tryit', '🖐 Try this — bury the answer and watch recall fall',
            `<b>1.</b> Press <b>put it first</b>, then <b>put it last</b>. Both are found reliably.<br>
             <b>2.</b> Press <b>put it in the middle</b> and read the drop. <b>Nothing changed except where the text sat.</b><br>
             <b>3.</b> Now drag <b>documents in the context</b> from 5 up to 100 with the answer still in the middle, and watch the penalty deepen.<br>
             <b>4.</b> This is why a retrieval system puts its best match first or last, rather than wherever it happened to rank.`),
          lostInMiddle(),
        ),

        section('Prompting: instructions without touching a weight',
          p(`Everything fed to the model before it generates is the <em>prompt</em>, and shaping it changes behaviour enormously with no retraining. A <em>system prompt</em> sets persistent instructions and persona up front ("You are a concise, technical reviewer…"). <em>Few-shot prompting</em> shows two or three worked examples before asking for a new one — pattern-matching a format the model already knows.`),
          p(` <em>Chain-of-thought</em> prompting just asks it to reason step by step first; since each token is conditioned on everything written so far, those intermediate steps give the model more of its own reasoning to build the final answer on, which measurably helps multi-step problems.`),
        ),

        section('Retrieval-augmented generation: giving a frozen model fresh facts',
          p(`A model's knowledge is frozen at training time and can't cover this morning's news or your company's wiki. <em>Retrieval-augmented generation</em> (RAG) works around that with no retraining: turn documents into vectors (<em>embeddings</em>, chapter 6) that capture meaning, do the same to the question, retrieve whichever documents' vectors sit closest, and paste — "stuff" — their text into the prompt before answering. The model still only predicts the next token; it just has better material in front of it now.`),
          p(`RAG works well when an answer is a fact living, close to verbatim, in a handful of retrievable documents — a policy, a spec, a ticket. It fails when the retriever misses the right document because the question is worded too differently; when the true answer needs combining documents that weren't all retrieved together; or when retrieved text is noisy enough that the model blends it with its own, possibly wrong, prior beliefs. Retrieval quality, far more than model size, is the real bottleneck.`),
          callout('tryit', 'Try it: break the retriever on purpose', `<b>1.</b> Ask about vacation days with the default question and watch the top document win clearly, with "vacation" and "days" highlighted. <b>2.</b> Click "parental leave" and see a different document take over — the ranking really is reading the question. <b>3.</b> Click "pets at the office": no document in this ten-document handbook mentions pets, so every score collapses toward zero. Notice what gets stuffed into the prompt anyway — three irrelevant policies — and consider what a model that isn't instructed to admit uncertainty might do with that.`),
          ragDemo(),
        ),

        section('Tool use: letting the model ask for something real',
          p(`<em>Function calling</em> (tool use) lets a model do more than talk. Instead of a normal reply, it emits a structured request — typically JSON naming a tool and its arguments, e.g. <code class="inline">{"tool": "get_weather", "args": {"city": "Canberra"}}</code>. The model was trained to produce this shape when a tool would help; <b>it never executes anything itself</b>.`),
          p(` A surrounding <em>harness</em> parses the call, actually runs the function (hits an API, reads a file), and feeds the result back in as ordinary tokens in the context. The model keeps generating, now conditioned on real information it couldn't have had beforehand — that loop is the entire foundation of every agent.`),
        ),

        section('Agents: think, act, observe, repeat',
          p(`An <em>agent</em> is a model wrapped in a loop: <b>think</b> (reason about what to do next), <b>act</b> (emit a tool call), <b>observe</b> (read the result back into context), repeat until the task is done or a limit is hit.`),
          p(` Nothing about the model differs from chapter 11; the harness now keeps handing it the consequences of its own actions, turning a system that predicts one reply at a time into one that can edit sixteen files, run the tests, read the failure, and try again — unsupervised.`),
          p(`Coding agents (Claude Code, Cursor) run exactly this loop over a codebase: read, patch, test, read the output, repeat. <em>Computer use</em> runs the same loop over a whole screen — shown a screenshot, the model emits a click or keystroke as its "tool call" and observes the resulting screenshot, operating software that was never given an API. For a long time every framework wired models to tools its own bespoke way; <em>MCP, the Model Context Protocol</em> (below), standardises that connection.`),
          p(`Long runs fail in characteristic ways. Errors <em>compound</em>: a small misreading at step 3 becomes a wrong assumption at step 30, and the model rarely notices, trusting its own earlier reasoning almost as much as a fresh tool result.`),
          p(` Context <em>bloats</em>: every thought, call and result stays in the transcript, pushing the original instructions toward the "lost in the middle" zone. Both push toward real <em>memory</em> — a persistent store outside the window — and toward <em>multi-agent</em> designs, where a manager delegates narrow sub-tasks to fresh, small-context sub-agents instead of one context growing without bound.`),
          callout('tryit', 'Try it: pull the plug on tools', `Press <b>Step</b> to watch the agent work through fixing a bug, one turn at a time, with the token count climbing on a small illustrative context window. Then click <b>Tools: ON</b> to turn it off and press <b>Step</b> again: same task, same confident tone — but every claim about the file, the fix, and the test result past step one is simply invented. Nothing in the model's <i>language</i> tells you which mode you're in; only the presence of a real read/run does.`),
          agentSimulator(),
        ),

        callout('history', 'November 2024: Anthropic ships a common language for tools', `Before MCP, every AI product wired its model to its tools its own bespoke way — one integration per tool per product, rebuilt from scratch each time. In November 2024, Anthropic open-sourced the <em>Model Context Protocol</em>: an open specification for how a model-calling app discovers and calls external tools and data sources, regardless of model or client. A tool built to speak MCP once can be picked up by any MCP-aware agent, the way USB let one plug fit any device instead of a new port per manufacturer. Within about a year it was adopted well beyond Anthropic's own products, becoming shared plumbing for the whole agent ecosystem — including several of the connectors this course's own authoring tools use.`),

        callout('example', 'Where this is already running', `Perplexity and ChatGPT's browsing mode are RAG: search first, answer grounded in what was found, citations included. Siri and Alexa's weather answers are function calling: a structured request to a weather service, the result read back as words. Claude Code and Cursor run the full agent loop over a real repository — reading, editing, testing, repeating — and Anthropic's computer-use demos run the identical loop over a screenshot instead of a file tree, filling out forms and navigating apps with no dedicated API at all.`),

        section('Beyond text: multimodality and structured output',
          p(`The same next-token machinery generalises past words. An image is cut into a grid of small square <em>patches</em> (commonly 16×16 pixels), each embedded into a vector exactly like a word token — the model attends across patches and words identically, as one long mixed sequence.`),
          p(` Audio is typically chopped into short time windows and mapped onto a learned vocabulary of audio "tokens", or transcribed to text first. Same trick that made language models general in the first place: turn whatever the input is into a sequence of vectors, and let the transformer do what it already does.`),
          p(`<em>Structured output</em> forces a reply into a strict format — JSON matching a schema, say — not by asking politely but by constraining the sampler: at each step, only tokens that keep the output validly formed are allowed into the softmax at all. That is what lets an agent's tool calls, and a developer's API responses, be parsed programmatically instead of hoping the model got the punctuation right.`),
        ),

        section('The economics: tokens per second, dollars per million, and caching',
          p(`Every property above has a price. Providers bill per token — dollars per million input and output tokens, output usually costlier since decode is slower than prefill — and speed is quoted in tokens per second. <em>Prompt caching</em> reuses the KV cache for a prefix sent repeatedly (a system prompt, a shared document), so the expensive prefill is paid once and later calls reusing it pay only a fraction.`),
          p(` Quantization, speculative decoding, MoE and caching all exist for the same reason: inference runs trillions of times a day, and that is where the compute bill lands.`),
          callout('tryit', '🖐 Try this — price a product before you build it',
            `<b>1.</b> Read the time bar. Even on a 4,000-token prompt, most of the wall clock is <b>decode</b>, emitted one token at a time.<br>
             <b>2.</b> Press <b>an agent turn</b>: a 60,000-token prompt and a 200-token reply. An agent resends its whole growing transcript every single step, so this is the shape that actually dominates an agent's bill.<br>
             <b>3.</b> Now turn <b>prompt caching on</b> and watch both the cost and the prefill time collapse. That one feature is the difference between an agent being viable and not.<br>
             <b>4.</b> Set <b>calls per day</b> to something like your own product and read the daily figure.`),
          costLab(),
        ),

        section('Hallucination: confident, fluent, and sometimes false',
          p(`A next-token predictor has one job: given the tokens so far, produce a plausible next one. Nothing in that objective checks the result against reality — a fluent, confident sentence about a court case that never happened scores just as well, mechanically, as a true one, if equally plausible given the training data. <em>Hallucination</em> isn't a rare bug slipping through; it's the default behaviour of a system never taught the difference between "this sounds right" and "this is true" — reporting confidence in its fluency, not its correctness.`),
          p(`Three things reduce it, without eliminating it: grounding the model in retrieved or tool-provided facts (RAG, function calling) gives it real material to lean on instead of only trained-in memory; training it, via RLHF (chapter 11), to hedge or say "I don't know" shifts the incentive away from confident guessing; and asking it to cite sources or show its steps doesn't stop it being wrong, but makes the wrongness checkable.`),
        ),

        section('Why this matters for modern AI',
          p(`Training (chapters 1–11) builds the function once. This chapter is that function running, unchanged, trillions of times a day inside every product you touch. A fast reply is speculative decoding and prompt caching at work; a slow one on a huge file is decode-phase KV-cache pressure. An agent editing your repository unsupervised is just the think-act-observe loop above, running exactly as written — nothing more mysterious than a model, a harness, and a patient cycle of trying, checking, and trying again.`),
        ),

        ctx.quiz([
          { q: 'You lower the temperature from 1.5 to 0.2. What happens to the sampled distribution?', options: ['It becomes flatter and more random', 'It becomes sharper, concentrating almost all probability on the model’s top few tokens', 'The vocabulary size changes', 'Nothing — temperature only affects training'], answer: 1, explain: 'Dividing logits by a smaller T stretches the gaps between them before softmax, so the largest logit dominates even more. T→0 is the limit case: always the single top token (greedy).' },
          { q: 'What does the KV cache actually store, and why does it matter?', options: ['A copy of the prompt for logging purposes', 'Each layer’s Key and Value vectors for every token already processed, so decode never recomputes them from scratch', 'A backup of the model’s weights', 'The user’s previous conversations, for personalization'], answer: 1, explain: 'Without the cache, generating token 500 would mean redoing attention over all 499 earlier tokens from scratch. The cache reuses that work — at the cost of memory that scales with layers × tokens × batch.' },
          { q: 'Two servers run the exact same model at temperature 0. Why might they still occasionally produce different tokens?', options: ['Temperature 0 is not really deterministic in principle, by definition', 'Floating-point sums are computed in different orders on different hardware/batches, and tiny rounding differences can flip which logit is largest', 'The tokenizer changes randomly between runs', 'Temperature 0 disables the KV cache, causing drift'], answer: 1, explain: 'Floating-point addition is not associative; a GPU’s parallel summation order depends on hardware and batch composition. When two logits are separated by less than that rounding error, "the top token" can differ between runs.' },
          { q: 'Per Liu et al.’s "Lost in the Middle" (2023), where in a long context is a model least likely to make good use of a relevant fact?', options: ['At the very start', 'At the very end', 'Buried in the middle', 'Equally poor everywhere'], answer: 2, explain: 'Accuracy was highest for facts placed at the start or end of the context and dropped for facts placed in the middle — a U-shaped curve. A bigger context window doesn’t fix this; it adds more middle.' },
          { q: 'An agent claims it read a config file, found a bug, and fixed it — but "tools" were switched off. What is actually true here?', options: ['The claim is reliable because the model is trained to be honest', 'Nothing was read or executed; the model generated plausible-sounding text with no way to verify any of it', 'The KV cache substituted for the missing tool call', 'This can only happen with very small models'], answer: 1, explain: 'Tool use is what connects a model’s output to reality. Without it, "I read the file and fixed it" is exactly as fabricated as any other fluent, confident, false sentence — which is what hallucination is.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://www.anthropic.com/news/model-context-protocol" target="_blank" rel="noopener">Anthropic, "Introducing the Model Context Protocol"</a> (Nov 2024) — the announcement and spec for MCP.`,
            `<a href="https://arxiv.org/abs/2307.03172" target="_blank" rel="noopener">Liu et al., "Lost in the Middle: How Language Models Use Long Contexts"</a> (2023) — the paper behind the U-shaped curve above.`,
            `<a href="https://arxiv.org/abs/2211.17192" target="_blank" rel="noopener">Leviathan, Kalman & Matias, "Fast Inference from Transformers via Speculative Decoding"</a> (ICML 2023) — the draft-and-verify trick, explained properly.`,
            `<a href="https://arxiv.org/abs/2005.11401" target="_blank" rel="noopener">Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"</a> (2020) — the paper that named RAG.`,
            `<a href="https://github.com/ggerganov/llama.cpp" target="_blank" rel="noopener">llama.cpp</a> — the project that made running a quantized 7B (and much larger) model on an ordinary laptop routine.`,
          ]),
        ),
      );
    },
  });
})();
