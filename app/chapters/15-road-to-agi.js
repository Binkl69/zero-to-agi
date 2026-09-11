/* Zero → AGI · Chapter 15 · The road to AGI
   DESIGN RULE: the reader draws their own definition of AGI in the first thirty seconds and
   watches the verdict swing without a single fact about any model changing.
   Interactives, in order: the AGI definer; capability radar (what is done, partial, not close);
   the task-horizon doubling chart; the training-budget cost explorer; and a personal roadmap
   builder. */
(function () {
  ZTA.registerChapter({
    id: '15-road-to-agi',
    num: 15,
    part: 'V',
    title: 'What separates us from AGI, and how you could help build it',
    tagline: 'Fourteen chapters of mechanism, one honest reckoning: what today\'s models still can\'t do, and where a single person can actually push.',
    render(root, ctx) {
      const { h, p, section, sub, callout, ul, ol } = ctx;
      const C = ctx.colors;
      const FONT = '13px Inter, system-ui, sans-serif';
      const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
      const f1 = (v) => (+v).toFixed(1);
      const f2 = (v) => (+v).toFixed(2);

      /* ================================================================== */
      /* Interactive A: capability radar                                     */
      /* ================================================================== */
      function capabilityRadar() {
        const W = 720, H = 470;
        const [cv, g] = ctx.canvas(W, H);
        const cx = 230, cy = 235, maxR = 168;
        const AXES = [
          'Knowledge recall', 'Maths & logic', 'Coding', 'Long-horizon agency',
          'Continual learning', 'Sample efficiency', 'Physical world', 'Calibration', 'Memory', 'Energy efficiency',
        ];
        // editorial estimates, 0–10, explained in the caption and the panel below
        const V2020 = [5, 2, 2, 1, 1, 1, 1, 2, 1, 2];
        const V2026 = [9, 8, 8, 5, 2, 3, 3, 5, 4, 3];
        const VHUMAN = [6, 5, 6, 9, 9, 9, 9, 7, 8, 9];
        const GAP = [
          { name: 'Knowledge recall', gap: 'On raw facts, the gap has flipped: a 2026 frontier model has read far more than any single human ever will, and it shows on PhD-qualifying-exam questions (GPQA) across fields no one person masters all of. The residual problem isn\'t how much it knows — it\'s that it states wrong facts just as fluently as right ones.', close: 'Retrieval that grounds answers in checkable sources, and training that rewards "I don\'t know" as much as a correct answer, so confidence tracks truth.' },
          { name: 'Maths & logic', gap: 'Frontier reasoning models now solve International Mathematical Olympiad and Putnam-level problems that stump most humans, using long chains of search-like reasoning before answering. The gap that remains is genuinely novel proof: work far outside anything like the patterns seen in training.', close: 'More reinforcement learning on <em>verifiable</em> problems — ones with a checkable right answer — plus formal proof assistants like Lean that can confirm a step is actually valid, not just plausible-sounding.' },
          { name: 'Coding', gap: 'Models now write, run, debug and ship real pull requests across a multi-hour agentic session, holding a whole task in mind. Reliability still falls as the task lengthens: small mistakes compound over a long chain of edits, and there is often no ground truth to self-correct against beyond "does it compile".', close: 'Environments with fast, automatic feedback (tests that pass or fail, sandboxes that run the code), and training that specifically rewards recovering from an error rather than just avoiding one.' },
          { name: 'Long-horizon agency', gap: 'This is the sharpest gap on the chart. METR\'s 2025 measurements found that the length of task a frontier model can complete at 50% reliability was around 110 minutes for the best models of early 2025 — up enormously from seconds a few years earlier, but still far short of a week of autonomous work a competent employee could be trusted with.', close: 'Better internal state-tracking and planning, the ability to notice its own mistakes mid-task, and — per the chart below — a doubling trend that has to keep compounding for years, not just one clever trick.' },
          { name: 'Continual learning', gap: 'A model\'s weights are frozen the moment training ends. Whatever it "learns" during a conversation lives only in that conversation\'s context window; close the tab and it is gone. A junior employee who is corrected on Monday does not need to be corrected again on Friday. A deployed model does.', close: 'Safe ways to keep updating weights after deployment (without the model forgetting old skills or being poisoned by bad data), or architectures with an explicit, persistent, editable memory that is not just "more context".' },
          { name: 'Sample efficiency', gap: 'A frontier model sees trillions of words before it can reliably use a comma. A child learns a new word, often for life, from one or two encounters, because a young brain arrives with strong built-in priors about objects, causes and agents that a model has to discover the hard way from data.', close: 'Better inductive biases baked into the architecture, meta-learning (learning <i>how</i> to learn from few examples), and world models rich enough that a new fact needs less repetition to stick.' },
          { name: 'Physical world', gap: 'Ask a model trained mostly on text and images to predict what happens when you stack seven blocks unevenly, or how a rope will fall, and it is guessing from pictures of similar scenes, not from ever having pushed a block. Video-generation models are visibly better than they were in 2020, but they still routinely produce impossible physics.', close: 'Real interaction with the physical world — robotics, simulation with real physics engines, video pretraining at far larger scale — so the model\'s notion of "what happens next" is grounded in consequence, not just correlation between pixels.' },
          { name: 'Calibration', gap: 'Hallucination — a fluent, confident, wrong answer — is not solved in 2026. It is reduced: retrieval, verifier models and RLHF (chapter 11) have made top models noticeably better at saying "I\'m not sure" than their 2020 ancestors, which almost never did. It has not been made to go away.', close: 'Training objectives that directly reward calibrated uncertainty rather than confident-sounding text, and more systematic use of retrieval and verification before an answer is shown.' },
          { name: 'Memory', gap: 'Context windows have grown from a couple of thousand tokens in 2020 to hundreds of thousands or more in 2026 — but a bigger context is not the same thing as a life. Nothing persists by default between sessions, across identities, or across the model\'s own deployments the way your own memory persists across your whole life.', close: 'External memory stores that an agent reads and writes across sessions (an early, working version of this exists already in agent scaffolds), moving eventually toward memory that is native to the model rather than bolted on.' },
          { name: 'Energy efficiency', gap: 'Your brain runs on about 20 watts — a dim light bulb — and does everything a human does. Training a single frontier model can burn tens of megawatts for months, and every reply from a deployed model, multiplied by billions of queries a day, adds up to a serious slice of a data centre\'s power bill.', close: 'Sparser architectures (mixture-of-experts, chapter 7), better accelerators, and algorithmic efficiency gains — all real and ongoing, but silicon and biology are different substrates, and the gap may never fully close.' },
        ];
        const S = { hover: 0, mode: 'gap', show: { y2020: true, y2026: true, human: true } };
        function toXY(i, v) {
          const ang = -Math.PI / 2 + i * (2 * Math.PI / AXES.length);
          const r = ctx.clamp(v, 0, 10) / 10 * maxR;
          return { x: cx + r * Math.cos(ang), y: cy + r * Math.sin(ang) };
        }
        function axisAt(px, py) {
          const ang = Math.atan2(py - cy, px - cx) + Math.PI / 2;
          const norm = ((ang % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
          return Math.round(norm / (2 * Math.PI / AXES.length)) % AXES.length;
        }
        function poly(vals) { return vals.map((v, i) => toXY(i, v)); }
        function drawPoly(pts, color, alpha, fillA) {
          g.beginPath(); pts.forEach((pt, i) => i ? g.lineTo(pt.x, pt.y) : g.moveTo(pt.x, pt.y)); g.closePath();
          if (fillA) { g.fillStyle = color; g.globalAlpha = fillA; g.fill(); g.globalAlpha = 1; }
          g.strokeStyle = color; g.globalAlpha = alpha; g.lineWidth = 2; g.stroke(); g.globalAlpha = 1;
          pts.forEach((pt) => { g.beginPath(); g.arc(pt.x, pt.y, 3, 0, Math.PI * 2); g.fillStyle = color; g.fill(); });
        }
        function draw() {
          g.clearRect(0, 0, W, H);
          // rings
          for (let ring = 2; ring <= 10; ring += 2) {
            g.beginPath();
            for (let i = 0; i <= AXES.length; i++) { const pt = toXY(i % AXES.length, ring); i ? g.lineTo(pt.x, pt.y) : g.moveTo(pt.x, pt.y); }
            g.strokeStyle = C.line; g.lineWidth = 1; g.stroke();
          }
          // spokes + labels
          g.font = FONT;
          for (let i = 0; i < AXES.length; i++) {
            const edge = toXY(i, 10);
            g.strokeStyle = i === S.hover ? C.warn : C.line; g.lineWidth = i === S.hover ? 2 : 1;
            g.beginPath(); g.moveTo(cx, cy); g.lineTo(edge.x, edge.y); g.stroke();
            const lx = cx + (maxR + 34) * Math.cos(-Math.PI / 2 + i * (2 * Math.PI / AXES.length));
            const ly = cy + (maxR + 34) * Math.sin(-Math.PI / 2 + i * (2 * Math.PI / AXES.length));
            g.fillStyle = i === S.hover ? C.warn : C.text;
            g.textAlign = Math.cos(-Math.PI / 2 + i * (2 * Math.PI / AXES.length)) > 0.2 ? 'left' : Math.cos(-Math.PI / 2 + i * (2 * Math.PI / AXES.length)) < -0.2 ? 'right' : 'center';
            const words = AXES[i].split(' ');
            let ty = ly - (words.length > 1 ? 6 : 0);
            words.forEach((w) => { g.fillText(w, lx, ty); ty += 13; });
          }
          if (S.show.y2020) drawPoly(poly(V2020), C.muted, 0.9, 0.06);
          if (S.show.human) drawPoly(poly(VHUMAN), C.green, 0.9, 0.07);
          if (S.show.y2026) drawPoly(poly(V2026), C.accent, 0.95, 0.12);
          // legend
          const leg = [['2020 models', C.muted, 'y2020'], ['2026 frontier', C.accent, 'y2026'], ['Human expert', C.green, 'human']];
          let lx0 = 470, ly0 = 40;
          g.textAlign = 'left'; g.font = FONT;
          leg.forEach(([label, color, key]) => {
            g.fillStyle = color; g.globalAlpha = S.show[key] ? 1 : 0.3;
            g.fillRect(lx0, ly0 - 9, 12, 12);
            g.fillStyle = C.text; g.fillText(label + (S.show[key] ? '' : ' (hidden)'), lx0 + 18, ly0 + 1);
            g.globalAlpha = 1; ly0 += 22;
          });
          g.fillStyle = C.muted; g.font = FONT;
          wrapText(g, 'Editorial estimates on a 0–10 scale, for teaching — not a benchmark score. Hover or click an axis for what the gap means.', 470, 118, 220, 15);
        }
        function wrapText(gc, text, x, y, maxW, lh) {
          const words = text.split(' '); let line = '', yy = y;
          for (const w of words) { const t = line ? line + ' ' + w : w; if (gc.measureText(t).width > maxW && line) { gc.fillText(line, x, yy); yy += lh; line = w; } else line = t; }
          if (line) gc.fillText(line, x, yy);
        }
        const panel = h('div', { class: 'callout key', style: { marginTop: '2px' } });
        function renderPanel() {
          const info = GAP[S.hover];
          panel.innerHTML = '';
          panel.append(
            h('div', { class: 'callout-title' }, info.name + (S.mode === 'gap' ? ' — the gap' : ' — what would close it')),
            h('div', { html: S.mode === 'gap' ? info.gap : info.close }),
          );
        }
        function onMove(ev) {
          const pt = cv.pos(ev);
          const d = Math.hypot(pt.x - cx, pt.y - cy);
          if (d < 20 || d > maxR + 40) return;
          const i = axisAt(pt.x, pt.y);
          if (i !== S.hover) { S.hover = i; draw(); renderPanel(); }
        }
        cv.addEventListener('pointermove', onMove);
        cv.addEventListener('pointerdown', onMove);
        cv.addEventListener('mousemove', onMove);
        cv.addEventListener('click', onMove);
        draw(); renderPanel();
        const modeBtn = ctx.button('Show: the gap', () => { S.mode = S.mode === 'gap' ? 'close' : 'gap'; modeBtn.textContent = S.mode === 'gap' ? 'Show: the gap' : 'Show: what would close it'; renderPanel(); });
        const t2020 = ctx.button('Toggle 2020', () => { S.show.y2020 = !S.show.y2020; draw(); });
        const t2026 = ctx.button('Toggle 2026', () => { S.show.y2026 = !S.show.y2026; draw(); });
        const thuman = ctx.button('Toggle human', () => { S.show.human = !S.show.human; draw(); });
        const body = h('div', {}, cv, panel);
        return ctx.figure(body, 'Ten axes, three snapshots — all editorial estimates on a 0–10 scale for teaching, not a published benchmark. 2026 frontier models (blue) now reach past the human line on knowledge, maths and coding — but pull far inward on continual learning, sample efficiency, the physical world and energy use. That shape, not any single number, is the honest answer to "how close are we?"', [modeBtn, t2020, t2026, thuman]);
      }

      /* ================================================================== */
      /* Interactive B: cost-of-compute explorer                             */
      /* ================================================================== */
      function costExplorer() {
        const W = 720, H = 230;
        const [cv, g] = ctx.canvas(W, H);
        const REF = [
          { n: 1.5e8, label: 'GPT-2 (small)' },
          { n: 1.5e9, label: 'GPT-2 (XL)' },
          { n: 8e9, label: 'Llama-3-8B' },
          { n: 7e10, label: 'Llama-3-70B' },
          { n: 1.75e11, label: 'GPT-3 (175B)' },
          { n: 4.05e11, label: 'Llama-3-405B' },
        ];
        const S = { budget: -3, price: 2.5, mfu: 35 }; // budget slider is log10($)
        const plot = { x: 26, y: 24, w: 668, h: 140 };
        const logMin = 6.5, logMax = 12.3; // param count log10 range shown on axis
        const xAt = (n) => plot.x + (Math.log10(n) - logMin) / (logMax - logMin) * plot.w;
        function compute() {
          const budget = Math.pow(10, S.budget);
          const gpuHours = budget / S.price;
          const flopsPerGpuSec = 1e15; // ≈ one 2026-class accelerator, dense bf16 — a stated, round assumption
          const totalFlops = gpuHours * 3600 * flopsPerGpuSec * (S.mfu / 100);
          const N = Math.sqrt(totalFlops / 120); // Chinchilla: C = 6ND, D = 20N  ⇒  C = 120N²
          const D = 20 * N;
          return { budget, gpuHours, totalFlops, N, D };
        }
        function classOf(N) {
          if (N < 3e8) return 'well below GPT-2 (small) — a toy, useful for learning the pipeline';
          if (N < 3e9) return 'roughly GPT-2 (XL) class';
          if (N < 3e10) return 'roughly Llama-3-8B class';
          if (N < 1.2e11) return 'roughly Llama-3-70B class';
          if (N < 3e11) return 'roughly GPT-3 (175B) class';
          return 'at or beyond Llama-3-405B — frontier-lab territory';
        }
        const ro = ctx.readout();
        function draw() {
          const r = compute();
          g.clearRect(0, 0, W, H);
          g.fillStyle = '#0f1520'; g.fillRect(plot.x, plot.y, plot.w, plot.h); g.strokeStyle = C.line; g.strokeRect(plot.x, plot.y, plot.w, plot.h);
          // reference model ticks
          g.font = MONO; g.textAlign = 'center';
          REF.forEach((m) => {
            const x = xAt(m.n);
            g.strokeStyle = C.line; g.lineWidth = 1; g.beginPath(); g.moveTo(x, plot.y); g.lineTo(x, plot.y + plot.h); g.stroke();
            g.fillStyle = C.muted; g.save(); g.translate(x, plot.y + plot.h + 14); g.rotate(0); g.fillText(m.label, 0, 0); g.restore();
          });
          // your marker
          const nx = ctx.clamp(xAt(r.N), plot.x, plot.x + plot.w);
          g.strokeStyle = C.warn; g.lineWidth = 2.5; g.beginPath(); g.moveTo(nx, plot.y - 6); g.lineTo(nx, plot.y + plot.h); g.stroke();
          g.beginPath(); g.moveTo(nx - 7, plot.y - 6); g.lineTo(nx + 7, plot.y - 6); g.lineTo(nx, plot.y + 8); g.closePath(); g.fillStyle = C.warn; g.fill();
          g.fillStyle = C.warn; g.textAlign = 'left'; g.font = 'bold 12px "JetBrains Mono", monospace';
          g.fillText('your compute-optimal N', Math.min(nx + 10, plot.x + plot.w - 175), plot.y + 16);
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          g.fillText('parameters, log scale →', plot.x, plot.y - 10);
          ro.set({
            'budget': '$' + Math.round(r.budget).toLocaleString(),
            'GPU-hours': Math.round(r.gpuHours).toLocaleString(),
            'compute-optimal params (N)': r.N >= 1e9 ? f2(r.N / 1e9) + 'B' : f2(r.N / 1e6) + 'M',
            'training tokens (D ≈ 20N)': r.D >= 1e9 ? f1(r.D / 1e9) + 'B' : f1(r.D / 1e6) + 'M',
            'class': classOf(r.N),
          });
        }
        const bSl = ctx.slider({ label: 'budget (log scale, $)', min: 2, max: 9, step: 0.05, value: 3, fmt: (v) => '$' + Math.round(Math.pow(10, v)).toLocaleString(), onChange: (v) => { S.budget = v; draw(); } });
        const pSl = ctx.slider({ label: '$ / GPU-hour', min: 0.5, max: 8, step: 0.1, value: 2.5, fmt: (v) => '$' + f2(v), onChange: (v) => { S.price = v; draw(); } });
        const mSl = ctx.slider({ label: 'MFU (hardware utilisation, %)', min: 10, max: 60, step: 1, value: 35, fmt: (v) => Math.round(v) + '%', onChange: (v) => { S.mfu = v; draw(); } });
        draw();
        return ctx.figure(cv, 'Chinchilla\'s compute-optimal recipe (Hoffmann et al., 2022): for a training compute budget C, the best split is roughly N ≈ √(C/120) parameters and D ≈ 20N tokens. GPU throughput and $/GPU-hour are stated, round, editorial assumptions — real prices, chip generations and utilisation vary widely, and 2024–2026 labs often deliberately over-train smaller models on far more tokens than this ratio for cheaper inference. Treat every number here as order-of-magnitude, not a quote.', [bSl, pSl, mSl], ro);
      }

      /* ================================================================== */
      /* Interactive C: your roadmap builder                                 */
      /* ================================================================== */
      function roadmapBuilder() {
        const S = { bg: 'some', hrs: 8 };
        function tierWord(hrs) { return hrs < 6 ? 'a steady side project' : hrs <= 15 ? 'a serious part-time commitment' : 'close to a second job — you will move fast'; }
        function build() {
          const bg = S.bg, hrs = S.hrs, tier = tierWord(hrs);
          const lines = [];
          lines.push('YOUR 12-MONTH ROAD TO AGI-ADJACENT WORK');
          lines.push('Background: ' + BGLABEL[bg] + '   ·   Pace: ' + hrs + ' h/week (' + tier + ')');
          lines.push('');
          lines.push('MONTHS 1–3 — foundations, hands on the machinery');
          if (bg === 'none') lines.push('  • Learn Python basics alongside this course (any free intro is fine) while working #/ch/01-what-is-learning through #/ch/03-the-learning-recipe.');
          else lines.push('  • Work #/ch/01-what-is-learning through #/ch/03-the-learning-recipe; do every interactive, not just the reading.');
          lines.push('  • Lab: train the tiny networks in #/ch/13-build-it-yourself yourself, from scratch, no copy-paste.');
          lines.push('  • Milestone: explain backpropagation to a friend using the chain-rule worked example from #/ch/02-neural-networks, from memory.');
          if (hrs > 15) lines.push('  • Stretch: also read #/ch/04-seeing-cnns and #/ch/05-sequences-rnns this quarter, not next.');
          lines.push('');
          lines.push('MONTHS 4–6 — transformers and the training pipeline');
          lines.push('  • Work #/ch/06-embeddings, #/ch/07-transformers, #/ch/08-generative-models.');
          lines.push('  • Lab: reproduce a GPT-2-scale model from scratch — follow Karpathy\'s nanoGPT / llm.c and this course\'s #/ch/10-pretraining-llms side by side.');
          if (bg === 'eng' || bg === 'research') lines.push('  • Since you already write production code: also read the DeepSeek-V3 and Llama 3 technical reports (reading list below) and try to match one design choice against what you just built.');
          else lines.push('  • Don\'t worry about matching frontier numbers — the goal is a working, understood pipeline, at any scale that fits your GPU.');
          lines.push('  • Milestone: a model you trained yourself completes a next-token-prediction demo end to end, and you can point to the loss curve and say why it looks the way it does.');
          lines.push('');
          lines.push('MONTHS 7–9 — post-training, evaluation, and a public artefact');
          lines.push('  • Work #/ch/09-reinforcement-learning, #/ch/11-post-training, #/ch/12-inference-and-agents.');
          lines.push('  • Lab: fine-tune an open model (Unsloth or plain PyTorch + LoRA) on ONE narrow task until it beats a frontier model\'s API on that task specifically. Narrow beats general here.');
          lines.push('  • Start reading a paper a week from the reading list below, oldest to newest; write two sentences per paper in a public notebook.');
          lines.push('  • Milestone: publish something real — a fine-tuned model card, an eval result, a small write-up — under your own name.');
          lines.push('');
          lines.push('MONTHS 10–12 — contribute, then choose a direction');
          lines.push('  • Send a real pull request to an open project: llama.cpp, vLLM, Hugging Face transformers, lm-eval-harness, or nanoGPT. Start with docs or a small bug fix; the goal is one merged PR, not a rewrite.');
          lines.push('  • Read #/ch/14-timeline and this chapter again; re-take the quizzes below and see what changed.');
          if (bg === 'research') lines.push('  • You likely already have the base to apply for research-engineer or research-scientist roles, or a PhD; use the reading list to pick a specific open problem (interpretability, evals, RL environments) and go deep on it.');
          else if (bg === 'eng') lines.push('  • Milestone: decide research engineer (build the systems that make research possible: training infra, data, evals) vs research scientist (design the experiments) — both need everything above, they diverge from here.');
          else lines.push('  • Milestone: you now have a real, checkable body of work. Use it to apply for an ML-adjacent role, a fellowship (MATS, ARENA), or simply to keep going — the skills compound.');
          lines.push('');
          lines.push('SKILLS THAT MATTER MOST, IN ROUGH ORDER: PyTorch fluency, reading and reproducing a paper, data engineering (cleaning and deduplicating a training set is most of the job), evals (can you tell if a change helped?), writing (can you explain what you found?), then CUDA/Triton and distributed training once you outgrow a single GPU.');
          return lines.join('\n');
        }
        const BGLABEL = { none: 'no coding background yet', some: 'some coding (scripts, one class)', eng: 'a working software engineer', research: 'already doing ML research or engineering' };
        const ta = ctx.textarea({ label: 'Your plan (click inside, then select-all and copy)', value: build() });
        function refresh() { ta.value = build(); }
        const bgSel = ctx.select({ label: 'your background', options: [
          { value: 'none', label: 'No coding yet' }, { value: 'some', label: 'Some coding' }, { value: 'eng', label: 'Strong software engineer' }, { value: 'research', label: 'ML researcher / practitioner' },
        ], value: 'some', onChange: (v) => { S.bg = v; refresh(); } });
        const hrSl = ctx.slider({ label: 'hours per week', min: 2, max: 40, step: 1, value: 8, onChange: (v) => { S.hrs = v; refresh(); } });
        return ctx.figure(ta, 'A plan built entirely from your two answers above — it changes as you move the slider or switch background. <code class="inline">#/ch/…</code> refers to chapters in this course; the rest are real, freely available projects named in the reading list.', [bgSel, hrSl]);
      }

      /* ================================================================== */
      /* Interactive D: task-horizon chart                                   */
      /* ================================================================== */
      function taskHorizonChart() {
        const W = 720, H = 320;
        const [cv, g] = ctx.canvas(W, H);
        const DATA = [
          { year: 2019.5, min: 0.05, label: 'GPT-2' },
          { year: 2020.5, min: 0.2, label: 'GPT-3' },
          { year: 2022.7, min: 1.5, label: 'GPT-3.5' },
          { year: 2023.3, min: 6, label: 'GPT-4' },
          { year: 2024.5, min: 25, label: 'Claude 3.5 / GPT-4o class' },
          { year: 2025.1, min: 110, label: 'o3 / frontier reasoners' },
        ];
        const Y0 = 2019, DOUBLE_MONTHS = 7;
        const T0 = DATA[0].min / Math.pow(2, (DATA[0].year - Y0) * 12 / DOUBLE_MONTHS);
        function fit(year) { return T0 * Math.pow(2, (year - Y0) * 12 / DOUBLE_MONTHS); }
        const plot = { x: 46, y: 16, w: 640, h: 240 };
        const X0 = 2019, X1 = 2033;
        const YMIN = 0.02, YMAX = 4e5; // minutes: ~1 sec .. ~9 months
        const xAt = (yr) => plot.x + (yr - X0) / (X1 - X0) * plot.w;
        const yAt = (min) => plot.y + plot.h - (Math.log10(ctx.clamp(min, YMIN, YMAX)) - Math.log10(YMIN)) / (Math.log10(YMAX) - Math.log10(YMIN)) * plot.h;
        function humanLabel(min) {
          if (min < 1) return Math.round(min * 60) + ' sec';
          if (min < 90) return f1(min) + ' min';
          if (min < 60 * 24 * 2) return f1(min / 60) + ' hr';
          if (min < 60 * 24 * 60) return f1(min / (60 * 24)) + ' days';
          return f1(min / (60 * 24 * 30)) + ' months';
        }
        const S = { extrap: 2026 };
        const ro = ctx.readout();
        function draw() {
          g.clearRect(0, 0, W, H);
          g.fillStyle = '#0f1520'; g.fillRect(plot.x, plot.y, plot.w, plot.h); g.strokeStyle = C.line; g.strokeRect(plot.x, plot.y, plot.w, plot.h);
          // log gridlines
          g.font = MONO; g.textAlign = 'right';
          for (let p10 = Math.ceil(Math.log10(YMIN)); p10 <= Math.floor(Math.log10(YMAX)); p10++) {
            const v = Math.pow(10, p10), y = yAt(v);
            g.strokeStyle = C.line; g.lineWidth = 1; g.beginPath(); g.moveTo(plot.x, y); g.lineTo(plot.x + plot.w, y); g.stroke();
            g.fillStyle = C.muted; g.fillText(humanLabel(v), plot.x - 6, y + 4);
          }
          // year ticks
          g.textAlign = 'center';
          for (let yr = 2019; yr <= X1; yr += 2) { const x = xAt(yr); g.fillStyle = C.muted; g.fillText(String(yr), x, plot.y + plot.h + 16); }
          // fitted curve: solid through observed range, dashed into extrapolation
          g.lineWidth = 2.5;
          g.beginPath();
          for (let yr = X0; yr <= 2025.1; yr += 0.1) { const x = xAt(yr), y = yAt(fit(yr)); yr === X0 ? g.moveTo(x, y) : g.lineTo(x, y); }
          g.strokeStyle = C.accent; g.stroke();
          g.setLineDash([5, 4]); g.beginPath();
          for (let yr = 2025.1; yr <= X1; yr += 0.1) { const x = xAt(yr), y = yAt(fit(yr)); yr === 2025.1 ? g.moveTo(x, y) : g.lineTo(x, y); }
          g.strokeStyle = C.accent; g.globalAlpha = 0.6; g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
          // observed points
          DATA.forEach((d) => {
            const x = xAt(d.year), y = yAt(d.min);
            g.beginPath(); g.arc(x, y, 4, 0, Math.PI * 2); g.fillStyle = C.warn; g.fill(); g.strokeStyle = '#0a0e16'; g.lineWidth = 1; g.stroke();
          });
          // extrapolation marker
          const ex = ctx.clamp(S.extrap, X0, X1), exY = fit(ex);
          const mx = xAt(ex), my = yAt(exY);
          g.strokeStyle = C.danger; g.lineWidth = 1.5; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(mx, plot.y); g.lineTo(mx, plot.y + plot.h); g.stroke(); g.setLineDash([]);
          g.beginPath(); g.arc(mx, my, 6, 0, Math.PI * 2); g.fillStyle = C.danger; g.fill();
          g.fillStyle = C.text; g.font = FONT; g.textAlign = 'left';
          g.fillText('solid = fit to observed points · dashed = extrapolation · red = your year', plot.x + 4, plot.y + 14);
          ro.set({ year: f1(ex), 'projected 50%-task horizon': humanLabel(exY), 'doubling assumption': DOUBLE_MONTHS + ' months' });
        }
        const sl = ctx.slider({ label: 'extrapolate to year', min: 2019, max: 2033, step: 0.1, value: 2026, fmt: (v) => f1(v), onChange: (v) => { S.extrap = v; draw(); } });
        draw();
        return ctx.figure(cv, 'The length of task (measured in how long a skilled human takes) that a model completes with 50% reliability, on a log scale. Orange dots are approximate reconstructions of published results; METR (Kwa et al., 2025) reports frontier models\' 50%-horizon growing roughly 2× every 7 months since 2019, reaching about 110 minutes for early-2025 reasoning models — and notes the trend may have <em>accelerated</em> since 2024, so a straight exponential likely understates it. The blue line is that fit, not a guarantee: extrapolating any exponential this far is a bet, not a fact.', [sl], ro);
      }

      /* ================================================================== */
      /* Prose                                                              */
      /* ================================================================== */

      function wrapLines2(gc, text, maxW) {
        const words = String(text).split(' '); const out = []; let line = '';
        for (const w of words) {
          const t = line ? line + ' ' + w : w;
          if (line && gc.measureText(t).width > maxW) { out.push(line); line = w; } else line = t;
        }
        if (line) out.push(line);
        return out;
      }
      function wrapText2(gc, text, x, y, maxW, lh) {
        wrapLines2(gc, text, maxW).forEach((ln, i) => gc.fillText(ln, x, y + i * lh));
      }

      /* ================================================================== */
      /*  INTERACTIVE — draw your own finish line                            */
      /* ================================================================== */
      function agiDefiner() {
        const [cv, g] = ctx.canvas(720, 400);
        const FONT = '13px Inter, system-ui, sans-serif';
        const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
        /* met: 1 = clearly true of 2026 frontier models, 0.5 = partly/contested, 0 = not close */
        const CRITERIA = [
          { n: 'Hold a conversation indistinguishable from a person', met: 1, note: 'Turing, 1950. Widely considered passed, and quietly dropped as a target.' },
          { n: 'Score at expert level on hard exams across many fields', met: 1, note: 'Graduate-level science, law, medicine, competition maths.' },
          { n: 'Write working code for a non-trivial task', met: 1, note: 'Routine in 2026, and a large share of actual usage.' },
          { n: 'Work usefully across many unrelated domains', met: 1, note: 'Legg & Hutter, 2007: goals in a *wide* range of environments.' },
          { n: 'Stay coherent on a task lasting several hours', met: 0.5, note: 'Improving fast, but errors still compound over long agent runs.' },
          { n: 'Learn from experience without being retrained', met: 0, note: 'Weights freeze when training ends. Nothing in the chat survives it.' },
          { n: 'Know reliably what it does and does not know', met: 0.5, note: 'Calibration improved a lot and is still not dependable.' },
          { n: 'Make an original scientific discovery unaided', met: 0, note: 'Genuine assistance, yes. Unaided discovery, no.' },
          { n: 'Do most economically valuable work', met: 0, note: 'OpenAI\'s charter bar. Not close, on any honest reading.' },
          { n: 'Act in the physical world as competently as a person', met: 0, note: 'Robotics lags the language side by a wide margin.' },
        ];
        const picked = CRITERIA.map((c, i) => i < 4);
        const PRESETS = {
          turing: [0],
          legg: [0, 1, 2, 3],
          openai: [0, 1, 2, 3, 4, 6, 8],
          strict: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        };
        const setPreset = (k) => { picked.forEach((_, i) => { picked[i] = PRESETS[k].indexOf(i) >= 0; }); };
        const btns = CRITERIA.map((c, i) => ctx.button(String(i + 1), () => { picked[i] = !picked[i]; }));
        const pTuring = ctx.button('Turing (1950)', () => setPreset('turing'));
        const pLegg = ctx.button('Legg & Hutter (2007)', () => setPreset('legg'));
        const pOpenAI = ctx.button('"most economically valuable work"', () => setPreset('openai'), 'primary');
        const pStrict = ctx.button('everything on the list', () => setPreset('strict'));
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('tick what you think "AGI" has to mean — numbered buttons below toggle each row', 30, 24);

          let need = 0, have = 0;
          CRITERIA.forEach((c, i) => {
            const y = 40 + i * 30;
            const on = picked[i];
            if (on) { need++; have += c.met; }
            g.fillStyle = on ? 'rgba(124,156,255,0.10)' : 'transparent';
            g.fillRect(30, y, 655, 27);
            /* the tick box */
            g.strokeStyle = on ? C.accent : C.line; g.lineWidth = 1.5;
            g.strokeRect(34, y + 6, 15, 15);
            if (on) { g.fillStyle = C.accent; g.fillRect(37, y + 9, 9, 9); }
            g.font = MONO; g.fillStyle = C.line; g.fillText(String(i + 1), 56, y + 18);
            g.font = FONT; g.fillStyle = on ? C.text : '#55627a';
            g.fillText(c.n, 74, y + 18);
            /* status */
            const col = c.met === 1 ? C.green : c.met === 0.5 ? C.warn : C.danger;
            const lab = c.met === 1 ? 'done' : c.met === 0.5 ? 'partly' : 'not close';
            g.font = 'bold ' + MONO; g.fillStyle = on ? col : '#3a4558';
            g.fillText(lab, 470, y + 18);
            g.font = MONO; g.fillStyle = on ? C.muted : '#2f3949';
            wrapLines2(g, c.note, 160).slice(0, 1).forEach(ln => g.fillText(ln, 530, y + 18));
          });

          const frac = need ? have / need : 0;
          const Y = 352;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('by your definition, 2026 models are', 30, Y);
          g.fillStyle = C.line; g.fillRect(30, Y + 8, 360, 20);
          g.fillStyle = frac > 0.85 ? C.green : frac > 0.5 ? C.warn : C.danger;
          g.fillRect(30, Y + 8, frac * 360, 20);
          g.font = 'bold 20px Inter, system-ui, sans-serif';
          g.fillStyle = frac > 0.85 ? C.green : frac > 0.5 ? C.warn : C.danger;
          g.fillText((frac * 100).toFixed(0) + '%', 402, Y + 25);
          g.font = FONT; g.fillStyle = C.muted;
          wrapText2(g, need === 0
            ? 'Tick at least one thing. That is harder than it sounds, and it is the whole problem.'
            : frac > 0.95 ? 'By this definition it already arrived, and nobody held a ceremony.'
              : frac < 0.35 ? 'By this definition it is clearly not here, and the missing pieces are not small.'
                : 'By this definition it is genuinely arguable — which is why the public argument never resolves.',
            470, Y + 12, 215, 16);
          ro.set({ 'criteria you chose': need, 'already met': have.toFixed(1), 'your verdict': need === 0 ? '—' : (frac * 100).toFixed(0) + '%' });
        });

        return ctx.figure(cv,
          'Every row is a real definition someone has seriously proposed, and the status column is an honest reading of 2026 frontier models rather than a measurement. Move between the presets and watch the verdict swing from "already arrived" to "not close" without a single fact about any model changing. That is the actual state of the AGI debate: not a disagreement about capabilities, but about where to draw a line that was never a unit like a kilogram in the first place.',
          [...btns, pTuring, pLegg, pOpenAI, pStrict], ro);
      }

      root.append(
        callout('tryit', '🖐 Do this first — draw your own finish line',
          `Ten things people have seriously proposed as the definition of AGI. Tick whichever ones <b>you</b> think it has to mean.<br>
           <b>1.</b> Press <b>Turing (1950)</b>. By that definition it arrived some time ago and nobody held a ceremony.<br>
           <b>2.</b> Press <b>"most economically valuable work"</b> — OpenAI's own charter bar. Now it is clearly not here.<br>
           <b>3.</b> Press <b>everything on the list</b>. Not close.<br>
           <b>4.</b> <b>No fact about any model changed between those three clicks.</b> Only where you drew the line.`),
        agiDefiner(),
        p(`That is the actual state of the AGI debate. Not a disagreement about what models can do — those are measurable — but about where a finish line goes that was never a unit like a kilogram.`),

        p(`You have just spent fourteen chapters learning how a machine turns a pile of numbers into something that can hold a conversation, write code, and pass a bar exam. So here is the question you have actually been building toward: are we close to a machine that can do <i>anything</i> a smart human can do? And if we are not there yet, is there anything one person, reading this in 2026, could actually do about it?`),
        p(`Both deserve honest answers, not hype and not doom. The first: closer than most people in 2015 would have believed, and further than most 2026 headlines admit. The second: yes, more than at almost any point in this field's eighty-year history, because the tools, the papers, and the open models are, for the first time, sitting on your own laptop.`),
        p(`This chapter earns both answers. It starts with what people actually mean by "AGI" — a term older and slipperier than it sounds. Then it takes today's frontier models apart, axis by axis, to find exactly where the gaps still are. Then the roads people are betting on to close them, the walls that could stop any of those bets, and — since this is the last chapter — what you, specifically, can do next.`),

        section('What people have meant by "AGI"',
          p(`Alan Turing never used the phrase "artificial general intelligence." In 1950 he proposed something cleverer: instead of arguing about the word "think," ask whether a machine's typed answers could be told apart from a human's. That sidestep — judge behaviour, not some unmeasurable inner spark — is still the field's best trick, and it still bites us, because a system can imitate the behaviour of understanding without necessarily having the thing itself.`),
          p(`In 2007 Shane Legg and Marcus Hutter tried to pin the word down properly: intelligence is an agent's ability to achieve goals in a <em>wide</em> range of environments. That one word is doing all the work. A chess engine is extraordinary in one environment and useless in every other; generality, not raw skill, is the bar.`),
          p(`Organisations building toward that bar wrote their own versions of it. <a href="https://openai.com/charter/" target="_blank" rel="noopener">OpenAI's charter</a> defines AGI as "highly autonomous systems that outperform humans at most economically valuable work" — an economic bar, deliberately concrete. `),
          p(`Anthropic tends to avoid the term itself and instead writes about <em>transformative AI</em>: systems whose impact could rival the agricultural or industrial revolutions — a framing about consequences, not a skills checklist. And in 2023 Google DeepMind (Morris et al.) proposed <a href="https://arxiv.org/abs/2311.02462" target="_blank" rel="noopener">"Levels of AGI"</a>, grading systems on <i>depth</i> (how good, "emerging" to "superhuman") and <i>breadth</i> (how general), the way self-driving cars get graded 0 through 5 — so "is it AGI yet?" stops being one yes/no argument.`),
          p(`None of these agree on a finish line, and that is the honest point: "AGI" is not a unit like a kilogram, it is a moving target several serious people define differently, and you should be suspicious of anyone — in either direction — who claims certainty about when we cross it.`),
        ),

        callout('history', 'A test that keeps getting redefined', `Turing's 1950 paper predicted machines would pass his test by 2000. A version of it plausibly happened, quietly, sometime in the 2020s — and by then almost nobody treated it as the finish line, because a system could imitate conversation convincingly while still failing at planning, memory and reliability in ways a five-year-old would not. Each decade's definition of "real" intelligence has moved to whatever the current best machines still can't do. That isn't a failure of the field; it's a sign the goalposts were badly placed the first time, and every reframing since — Legg &amp; Hutter, the OpenAI charter, DeepMind's Levels of AGI — has been an attempt to place them better.`),

        section('What 2026 frontier models can already do',
          p(`Whatever you think "AGI" should mean, it's worth being precise about the checkable capabilities of 2026 frontier models — both the hype and the dismissal usually skip this part. `),
          p(`Today's best models answer PhD-qualifying-exam science questions (the GPQA benchmark) at a level beating most non-specialist PhDs outside their own field. They solve International Mathematical Olympiad and Putnam-competition problems most strong maths graduates cannot. Handed an open-ended coding task, they work autonomously for hours across dozens of files, run their own tests, and open a real pull request. They look at a screen, decide what to click, and operate real software (<em>computer use</em>). And they do all of this across text, images, audio and video in one system, not four bolted together.`),
          p(`That list would have sounded like science fiction to this field's own researchers in 2015. It is real, measured, and why the conversation about AGI stopped being purely academic around 2023.`),
        ),

        callout('example', 'Where this already shows up outside a lab', `A biology postdoc uses a frontier model as a genuinely useful second opinion on an experimental design question outside their subfield. A solo developer ships a small SaaS product where a coding agent wrote most of the backend overnight. A radiologist's second-read tool flags a missed finding on a chest X-ray. None of these is AGI. All of them were "obviously science fiction" answers to "what could a computer do" in a 2015 textbook.`),

        section('What they still cannot do',
          p(`Set those achievements next to what has barely moved, and the shape of the remaining problem comes into focus. Four gaps show up in nearly every serious researcher's list, worth naming precisely — "not AGI yet" isn't a feeling, it's these specific, checkable deficits.`),
          ul([
            `<b>Continual learning.</b> A model's weights are frozen the instant training ends. Anything it appears to "learn" mid-conversation lives only in that conversation's context and disappears the moment it ends. A human employee who is corrected once usually does not repeat the mistake; a deployed model, by default, will.`,
            `<b>Sample efficiency.</b> Training uses trillions of words. A toddler learns a new word for life from a couple of exposures, because a young brain arrives with strong priors about objects, causes, and other minds that a model currently has to reconstruct, laboriously, from raw text statistics.`,
            `<b>Long-horizon reliability.</b> A model that is right 98% of the time per step is wrong more often than not by step fifty, because errors compound. This single fact — not raw intelligence — is most of why an agent that nails a five-minute task can still fail a five-day one.`,
            `<b>Robust world models and physical common sense.</b> Ask what happens when an unevenly stacked tower of blocks tips over, and a system trained mostly on text and images is pattern-matching against similar-looking scenes, not simulating physics the way a body that has actually knocked things over learns to.`,
          ]),
          p(`Add to that list two problems that are shrinking but stubbornly not solved: <em>calibration</em> — a model still states a wrong fact exactly as fluently as a right one, so hallucination is reduced, not gone — and <em>energy</em>. Your brain runs on about 20 watts, a dim bulb, and does everything a human does with it. Training one frontier model burns tens of megawatts for months, and every one of the billions of daily replies from deployed models adds to that bill. Different substrate, wildly different economics.`),
          callout('tryit', 'Try it: read the shape of the gap, not just the size', `Hover or click each axis of the radar below. Notice that 2026 models (blue) already push <i>past</i> the human-expert line (green) on knowledge, maths and coding — the argument "models don't really understand anything" gets harder to make on those three axes specifically. Then look at the four axes on the left where blue collapses inward: that collapse, not the outward bulge, is where the honest uncertainty about AGI actually lives. Toggle "what would close it" for each axis and notice how different the fixes are — no single breakthrough closes all four.`),
          capabilityRadar(),
        ),

        section('The clock nobody agrees on: how fast is "long-horizon" moving?',
          p(`Of the four gaps above, long-horizon reliability is the one with the best public data behind it. In 2025 the AI safety research group METR asked a direct question: for a task of a given length (measured by how long a skilled human takes to do it), what is the longest task a given model can complete with 50% success? They call this the model's <em>time horizon</em>, and they have been tracking it since the GPT-2 era.`),
          p(`Their finding: the 50%-success time horizon of frontier models has been doubling roughly every seven months since 2019 — and the trend may have sped up since 2024. Early-2025 reasoning models reached about 110 minutes. `),
          p(`That single number is easy to misread in either direction. Read pessimistically, "110 minutes" sounds unimpressive next to a human workweek. Read as a trend, seven straight years of doubling every seven months is one of the fastest sustained capability curves in the history of any technology — and METR's own extrapolation is that, if the trend holds, tasks that take a skilled human a full month could be within reach within about five years of their 2025 measurement.`),
          p(`Notice the word <i>if</i>. Every exponential trend in this field's history — Moore's law, model scale, benchmark scores — has eventually bent, sometimes up and sometimes down, when it hit a wall nobody had priced in yet. Treat the extrapolation below as the trend's honest continuation, not a prophecy.`),
          callout('tryit', 'Try it: drag the horizon forward', `Start at 2026 and read the projected task length. Now drag to 2030: notice it jumps from hours to weeks, because a fixed <i>doubling time</i> compounds into an enormous absolute number surprisingly fast — that is what exponentials do, and it is the same math as chapter 14's compute curve. Then drag back to 2020 and compare the model's projected horizon there against what GPT-3 could actually do; a mismatch is the fit being a smooth idealisation of noisy, lumpy real progress.`),
          taskHorizonChart(),
        ),

        section('Candidate paths to close the gaps',
          p(`Nobody knows which of the following actually gets to AGI, whether it takes all of them together, or whether one more idea nobody has had yet is required. That uncertainty is the honest state of the field in 2026 — anyone who tells you otherwise is selling something. Here is what serious labs are actually betting compute on.`),
          ctx.cards([
            { title: 'Scale, plus RL on more environments', body: `Richard Sutton's 2019 "Bitter Lesson" observed that across seventy years of AI research, general methods leveraging more computation — search and learning — have repeatedly beaten hand-engineered domain knowledge. The bet: keep scaling, but replace "more text" with "more verifiable environments" — coding sandboxes, proof checkers, games — where RL (chapter 9) can push past what imitating human text alone teaches.` },
            { title: 'Test-time compute and search', body: `Instead of only making the model bigger, let it think longer per question: generate multiple reasoning paths, search over them, verify and select. This is most of what turned 2023's models into 2025's IMO-medal-level reasoners — trading inference cost for capability in a way pretraining scale alone can't.` },
            { title: 'Continual and online learning, plus memory', body: `Give a deployed model a safe way to keep updating — its weights, or an external memory it reads and writes across sessions — so it stops forgetting the moment the context window closes. This targets the continual-learning and memory gaps directly, and is one of the least solved items here.` },
            { title: 'World models and embodiment', body: `Train on video, simulation and eventually robotics, not just text, so physics and cause-and-effect are learned from consequences rather than described in words. The most expensive, slowest-moving path, because the physical world doesn't compress into a token stream the way text does.` },
            { title: 'New architectures: SSMs, hybrids, sparsity', body: `The transformer (chapter 7) isn't sacred. State-space models (Mamba and relatives, chapter 5) process long sequences more cheaply; mixture-of-experts sparsity (DeepSeek-V3 and others) lets a model activate far fewer parameters per token than it holds in total. Hybrids of all three are now standard in frontier models.` },
            { title: 'AI doing AI research', body: `The most speculative, most consequential bet: use current models to help design, debug and evaluate the next generation, automating part of the research loop itself. If it works, progress could compound faster than human researchers alone could sustain — which is exactly why the next section's safety questions aren't hypothetical.` },
          ]),
        ),

        section('The hard constraints nobody gets to skip',
          p(`Every path above runs into the same set of walls, regardless of which lab or which architecture is betting on it.`),
          sub('Compute and energy',
            p(`Frontier training runs are bottlenecked by how many advanced chips exist, which is bottlenecked by a handful of semiconductor fabs on Earth, which is bottlenecked by machines (extreme ultraviolet lithography) only one company currently makes. Every chip also needs power: frontier data centres now run in the hundreds of megawatts, competing for grid capacity with cities. "Just add more GPUs" is a real strategy and also a genuinely physical, multi-year bottleneck, not a software problem.`),
            callout('tryit', 'Try it: feel what a training budget actually buys', `Set the budget slider to $1,000 and note the "class" readout. Now move it to $100,000,000 and watch how many orders of magnitude of parameters that unlocks — and notice the caption's honest caveat that this ignores salaries, failed runs, and data costs, which for a frontier lab often cost as much again as the compute itself.`),
            costExplorer(),
          ),
          sub('Data walls',
            p(`Models have now read a meaningful fraction of all the well-formed public text humanity has ever written. That well isn't bottomless. Labs are responding with <em>synthetic data</em> (model-generated, filtered and verified) and a shift toward <em>verifiable environments</em> — coding with a test suite, maths with a checkable proof, games with a score — where more data can be generated on demand because correctness is checked automatically.`),
          ),
          sub('Evaluation saturation',
            p(`A benchmark stops being useful once every frontier model scores 95%+ on it — not because the underlying skill is solved, but because the test can no longer tell good models apart, and its questions have often leaked into later training data. Much of applied AI research in 2025–2026 is quietly about building benchmarks that haven't yet been gamed.`),
          ),
          sub('Alignment and safety: the failure mode that matters most',
            p(`This constraint is unlike the others: it's not a resource that runs out, it's a risk that grows <i>with</i> capability. The scenario researchers at every major lab actually worry about isn't "the model turns evil." It's more mundane and more serious: <b>capability outpacing our ability to verify what a system is actually doing and why.</b> A highly capable, somewhat unreliable model whose internal reasoning we can't fully inspect is a bad combination regardless of intent, because "we can't tell if this went wrong" scales badly with autonomy.`),
            p(`Three research responses exist today. <em>Interpretability</em> looks inside a model's weights and activations for what it's actually computing, rather than trusting its stated reasoning. <em>Scalable oversight</em> builds ways for humans (or weaker, trusted AI) to check the work of a system smarter or faster than they are. And labs now publish explicit commitments about what capability level triggers what precaution before deployment — Anthropic's <a href="https://www.anthropic.com/news/core-views-on-ai-safety" target="_blank" rel="noopener">Responsible Scaling Policy</a> is one lab's framework for this, not an industry standard.`),
          ),
          sub('Governance',
            p(`Export controls on advanced chips, reporting requirements for the largest runs, and international coordination attempts are all, in 2026, early, contested and unevenly enforced. The underlying problem is real regardless of any specific policy: the compute build-out is happening faster than most governments' ability to understand, let alone regulate, it.`),
          ),
        ),

        section('What one person can realistically do',
          p(`Given all of that: not "wait for a lab," but a real, tiered answer for someone at your own level of experience today.`),
          ul([
            `<b>In months:</b> finish this course, every lab included. Reproduce a GPT-2-scale model from scratch (chapter 13, plus Karpathy's nanoGPT/llm.c). Read one paper a week, oldest to newest, from the list below. This alone puts you ahead of most people who talk about AI professionally.`,
            `<b>In one to two years:</b> contribute to a real open-source project — llama.cpp, vLLM, Hugging Face's libraries, lm-eval-harness, Unsloth, nanoGPT — starting small. Publish an evaluation result or interpretability finding under your own name. Fine-tune a niche model that beats a frontier model's API on one specific, narrow task; narrow-and-real beats broad-and-vague this early.`,
            `<b>After that:</b> join an existing lab or team, or found one, with a public, checkable track record instead of a resume line. Decide between <em>research engineer</em> (build the training infrastructure, data pipelines and eval harnesses research runs on) and <em>research scientist</em> (design the experiments) — both need everything above; they diverge from here, not before.`,
          ]),
          p(`The skills that matter most, roughly in the order they pay off: PyTorch fluency; actually reproducing a paper, not just reading its abstract; data engineering (cleaning and deduplicating a dataset is more of the job than people expect); building evals (can you tell, cheaply and repeatably, whether a change helped?); clear writing; and, once you outgrow one GPU, distributed training and CUDA/Triton.`),
          callout('tryit', 'Try it: build your own plan', `Pick the background that matches you honestly and the hours per week you can actually sustain — not the hours you wish you had. Read the plan it generates. Then change one input and see what shifts: notice that the milestones change, but the underlying skill order (foundations, then a real training run, then a narrow public artefact, then a contribution) does not.`),
          roadmapBuilder(),
        ),

        section('Cost of compute, in round numbers (2026, approximate)',
          p(`Order-of-magnitude estimates for late 2026, not quotes — and stale within a year or two, so treat them that way.`),
          ctx.table(
            ['Budget', 'What it approximately buys in 2026'],
            [
              ['$100', 'A few dozen GPU-hours: LoRA-fine-tune a 1–7B open model on a niche task, or a few thousand frontier-API calls.'],
              ['$1,000', 'A few hundred GPU-hours: pretrain a genuine GPT-2-scale model from scratch, or heavily fine-tune a 7–13B model.'],
              ['$10,000', 'A small GPU cluster for days to weeks: a solid few-hundred-million-to-low-billion-parameter pretraining run.'],
              ['$1,000,000', 'Hundreds of thousands of GPU-hours: roughly comparable to reported costs for a 7–13B-class open model\'s pretraining, or large-scale RLHF post-training on an open base model.'],
              ['$100,000,000+', 'Tens of thousands of accelerators for months: frontier-class pretraining (hundreds of billions of parameters) — usually matched by an equally large spend on staff, data and runs that never ship.'],
            ],
          ),
        ),

        section('Why this matters for modern AI',
          p(`Every capability in this chapter — the PhD-level answers, the hours-long coding sessions, the 110-minute task horizon — was built from exactly the mechanisms in chapters 1 through 13: a loss function, gradient descent, backpropagation, a transformer's attention, scaled up, then shaped by RLHF and RL on verifiable tasks (chapters 9–11).`),
          p(` AGI, if it arrives, won't be a different kind of machine. It will be this same recipe, plus whichever candidate path above actually closes the remaining gaps — gaps that are specific, named, and in several cases already measured year over year, not vague hand-waving. That's the most useful thing this course can leave you with: not a prediction of when, but a precise enough map of <i>what is still missing</i> that you can watch it close, or fail to, with your own eyes.`),
          p(`And unlike almost any other transformative technology in history, the tools to work on that map — papers, open model weights, training code — sit on your computer right now, not locked in one company's basement. That's genuinely new, and it's why the roadmap above isn't wishful thinking.`),
        ),

        ctx.quiz([
          { q: 'According to OpenAI\'s charter, what does "AGI" mean?', options: ['A robot physically indistinguishable from a human', 'Highly autonomous systems that outperform humans at most economically valuable work', 'Any system that passes a five-minute text conversation test', 'A system with unlimited persistent memory'], answer: 1, explain: 'That is the charter\'s own working definition — an economic bar, not a behavioural or architectural one. Anthropic and DeepMind frame the same underlying question differently (transformative impact; graded levels of depth and breadth).' },
          { q: 'What did METR\'s 2025 study actually measure?', options: ['The dollar cost of training a frontier model', 'The length of task, measured by human completion time, that a model finishes with 50% reliability, tracked from 2019 onward', 'The number of parameters needed to match human Go-playing ability', 'Energy used per generated token'], answer: 1, explain: 'This is the "50%-task-completion time horizon," and METR found it doubling roughly every 7 months since 2019 — reaching about 110 minutes for early-2025 reasoning models, with the trend possibly accelerating since 2024.' },
          { q: 'Why can\'t a deployed 2026 model permanently learn from a single bad interaction, the way a new employee learns from one correction?', options: ['It is legally forbidden from learning', 'Its weights are frozen after training; anything it "learns" mid-conversation lives only in that context window and disappears when the conversation ends', 'It only processes images, not language', 'Its memory is unlimited, so nothing is ever lost, including mistakes'], answer: 1, explain: 'This is the continual-learning gap — one of the widest on the capability radar. In-context adaptation is real but temporary; nothing updates the underlying weights after deployment by default.' },
          { q: 'Richard Sutton\'s "Bitter Lesson" (2019) argues that across AI\'s history, what has consistently beaten hand-engineered domain knowledge?', options: ['Smaller, more efficient models', 'General methods that leverage more computation — search and learning — scaled up', 'Rule-based expert systems refined over decades', 'Better mathematical proofs of correctness'], answer: 1, explain: 'Sutton\'s observation, drawn from seventy years of AI research, is uncomfortable for domain experts but has held up repeatedly: methods that scale with compute keep beating methods that encode human insight by hand.' },
          { q: 'Per this chapter, what is the failure mode that safety researchers actually worry about most as models get more capable?', options: ['Models becoming too expensive to run for anyone but large labs', 'Capability outpacing our ability to verify and oversee what a model is actually doing — not the model "turning evil"', 'Running out of GPUs entirely by 2027', 'Benchmarks becoming too easy to saturate'], answer: 1, explain: 'The concrete concern is a capable, somewhat unreliable system whose internal reasoning we cannot fully inspect or verify — which is why interpretability and scalable oversight are active research areas, not solved problems.' },
        ]),

        section('Go deeper: the reading list, roughly in order',
          ol([
            `<a href="https://www.csee.umbc.edu/courses/471/papers/turing.pdf" target="_blank" rel="noopener">Turing, "Computing Machinery and Intelligence"</a> (1950) — the imitation game, and the first serious attempt to define the question at all.`,
            `<a href="https://arxiv.org/abs/0712.3329" target="_blank" rel="noopener">Legg &amp; Hutter, "Universal Intelligence: A Definition of Machine Intelligence"</a> (2007) — the formal definition behind "wide range of environments."`,
            `<a href="https://arxiv.org/abs/2311.02462" target="_blank" rel="noopener">Morris et al. (Google DeepMind), "Levels of AGI for Operationalizing Progress on the Path to AGI"</a> (2023) — a graded framework instead of a single yes/no line.`,
            `<a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener">Vaswani et al., "Attention Is All You Need"</a> (2017) — the architecture nearly everything since is built on.`,
            `<a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" target="_blank" rel="noopener">Radford et al., GPT-2, "Language Models are Unsupervised Multitask Learners"</a> (2019).`,
            `<a href="https://arxiv.org/abs/2005.14165" target="_blank" rel="noopener">Brown et al., GPT-3, "Language Models are Few-Shot Learners"</a> (2020).`,
            `<a href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noopener">Kaplan et al., "Scaling Laws for Neural Language Models"</a> (2020).`,
            `<a href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noopener">Hoffmann et al., "Training Compute-Optimal Large Language Models"</a> (Chinchilla, 2022) — the C = 6ND relationship used in the cost explorer above.`,
            `<a href="https://arxiv.org/abs/2203.02155" target="_blank" rel="noopener">Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback"</a> (InstructGPT, 2022).`,
            `<a href="https://arxiv.org/abs/2212.08073" target="_blank" rel="noopener">Bai et al., "Constitutional AI: Harmlessness from AI Feedback"</a> (2022).`,
            `<a href="https://arxiv.org/abs/2305.18290" target="_blank" rel="noopener">Rafailov et al., "Direct Preference Optimization"</a> (2023).`,
            `<a href="https://arxiv.org/abs/2303.12712" target="_blank" rel="noopener">Bubeck et al., "Sparks of Artificial General Intelligence: Early Experiments with GPT-4"</a> (2023).`,
            `<a href="http://www.incompleteideas.net/IncIdeas/BitterLesson.html" target="_blank" rel="noopener">Sutton, "The Bitter Lesson"</a> (2019) — one page, and worth reading twice.`,
            `<a href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noopener">Meta AI, "The Llama 3 Herd of Models"</a> (2024) — the most detailed public account of a frontier-scale training run.`,
            `<a href="https://arxiv.org/abs/2412.19437" target="_blank" rel="noopener">DeepSeek-AI, "DeepSeek-V3 Technical Report"</a> and <a href="https://arxiv.org/abs/2501.12948" target="_blank" rel="noopener">"DeepSeek-R1"</a> (2024–2025) — open-weight frontier training and RL-driven reasoning, at reported cost far below Western labs' equivalents.`,
            `<a href="https://www.anthropic.com/news/core-views-on-ai-safety" target="_blank" rel="noopener">Anthropic, "Core Views on AI Safety"</a> (2023) — one lab's public reasoning for why safety work is urgent, including the Responsible Scaling Policy framing.`,
            `<a href="https://situational-awareness.ai" target="_blank" rel="noopener">Aschenbrenner, "Situational Awareness"</a> (2024) — a widely-read, deliberately aggressive extrapolation; read it and the more cautious pieces on this list against each other.`,
            `<a href="https://arxiv.org/abs/2503.14499" target="_blank" rel="noopener">Kwa et al. (METR), "Measuring AI Ability to Complete Long Tasks"</a> (2025) — the task-horizon study behind the chart above.`,
            `<a href="https://www.youtube.com/@AndrejKarpathy" target="_blank" rel="noopener">Karpathy, "Neural Networks: Zero to Hero" and onward</a> — the videos this course's labs are built to follow, start to finish.`,
          ]),
        ),
      );
    },
  });
})();
