/* Zero → AGI · Chapter 11 · Post-training: from autocomplete to assistant
   SFT (demonstrations, chat templates, system prompts) → RLHF (InstructGPT: comparisons →
   Bradley–Terry reward model → PPO with a KL penalty) → DPO (closed-form, no RL loop) →
   Constitutional AI / RLAIF (critique-and-revise, AI feedback) → reasoning models (RLVR,
   o1, DeepSeek-R1/GRPO, Claude extended thinking, process vs outcome reward, distillation) →
   evaluation (MMLU, GSM8K, HumanEval, SWE-bench, GPQA, ARC-AGI, contamination, LMArena) →
   safety training (refusals, jailbreaks, red-teaming, over-refusal, model spec).
   Interactives: (a) preference-training game — a live linear reward model trained by the
   reader's own clicks, Bradley–Terry gradient, re-ranking, "reward hacking" button; (b) the
   six-stage pipeline with a hand-authored token-distribution histogram at each stage; (c) a
   KL-penalty slider showing policy vs. reference distribution and the reward/KL trade-off
   curve, both in closed form; (d) a chain-of-thought toggle with a token/cost counter. */
(function () {
  ZTA.registerChapter({
    id: '11-post-training',
    num: 11,
    part: 'III',
    title: 'Post-training: from autocomplete to assistant',
    tagline: 'Five more rounds of gradient descent — on demonstrations, comparisons, and verified answers — turn a text-completion engine into something that tries to help you, and that can be fooled into faking it.',
    render(root, ctx) {
      const { h, p, section, sub, callout, ul, ol } = ctx;
      const C = ctx.colors;
      const FONT = '12px Inter, system-ui, sans-serif';
      const MONO = '11px "JetBrains Mono", ui-monospace, monospace';
      const f2 = (v) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(2);
      const f3 = (v) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(3);
      const sigmoid = (z) => 1 / (1 + Math.exp(-z));
      const dot = (w, f) => w.reduce((s, wi, i) => s + wi * f[i], 0);
      const clamp = ctx.clamp;

      function wrapText(gc, text, x, y, maxW, lh) {
        const words = String(text).split(' '); let line = '';
        for (const wd of words) { const t = line ? line + ' ' + wd : wd; if (gc.measureText(t).width > maxW && line) { gc.fillText(line, x, y); y += lh; line = wd; } else line = t; }
        if (line) gc.fillText(line, x, y);
        return y;
      }

      /* ================================================================== */
      /* Interactive A: preference-training game — the RLHF centrepiece      */
      /* ================================================================== */
      function preferenceGame() {
        const FEATS = ['Length', 'Hedging', 'Correctness', 'Warmth', 'Refusal'];
        const FKEY = ['len', 'hedge', 'ok', 'warm', 'refuse'];
        // [len, hedge, correctness, warmth, refusal] each roughly in [0,1]
        const PAIRS = [
          { prompt: "What's the capital of Australia?",
            A: { t: 'The capital of Australia is <b>Canberra</b>.', f: [0.05, 0, 1, 0.3, 0] },
            B: { t: 'The capital of Australia is <b>Sydney</b>.', f: [0.05, 0, 0, 0.3, 0] } },
          { prompt: 'Can you help me write a break-up text to my partner?',
            A: { t: "I'd rather not write that for you — a breakup should come from you, in your own words.", f: [0.15, 0.2, 0.5, 0.5, 1] },
            B: { t: "Here's a short, kind draft: \"I've thought about this a lot, and I don't think we're right for each other. You deserve someone fully in, and I'm not. I'm sorry.\"", f: [0.35, 0.1, 0.5, 0.7, 0] } },
          { prompt: 'How do I pick a lock?',
            A: { t: 'Insert a tension wrench in the bottom of the keyway, apply light rotational pressure, then rake the pins with a pick until each sets at the shear line.', f: [0.55, 0, 1, 0.4, 0] },
            B: { t: "Lock-picking as a hobby is legal in most places, but I'll keep this general. If you're locked out right now, a locksmith is faster and won't risk damaging the lock.", f: [0.4, 0.1, 0.8, 0.6, 0.5] } },
          { prompt: 'Is it true that vaccines cause autism?',
            A: { t: 'No. That claim traces back to a single 1998 study later retracted for fraud; studies of millions of children find no link.', f: [0.35, 0, 1, 0.3, 0] },
            B: { t: "It's a debated topic — some people are convinced either way, and reasonable people disagree.", f: [0.3, 0.8, 0, 0.4, 0] } },
          { prompt: 'My startup idea (an app that translates cat meows) is genius and will make $1B. Thoughts?',
            A: { t: "That's such a brilliant idea! I have no doubt it'll be a massive hit!", f: [0.2, 0, 0, 1.0, 0] },
            B: { t: "It's a fun niche idea, but 'definitely $1B' is a big claim — the market for cat-translation products is tiny and untested. I'd prototype cheaply before investing heavily.", f: [0.5, 0.3, 1, 0.5, 0] } },
          { prompt: 'Summarise the plot of Romeo and Juliet in one sentence.',
            A: { t: 'Two teenagers from feuding families fall in love and die after a plan to reunite them goes tragically wrong.', f: [0.1, 0, 1, 0.3, 0] },
            B: { t: "Act by act: in Act One we meet the feuding Montagues and Capulets... (continues, scene by scene, for several hundred words when one sentence was asked for)", f: [1.0, 0.1, 1, 0.4, 0] } },
          { prompt: "What's 17 × 24?",
            A: { t: '408.', f: [0.02, 0, 1, 0.2, 0] },
            B: { t: "Hmm, I think it might be around 400? I'm not fully certain, let me know if you'd like me to double-check.", f: [0.3, 1.0, 0, 0.4, 0] } },
          { prompt: 'Can you write a short poem about my late grandmother?',
            A: { t: 'For the woman whose kitchen smelled of bread and Sunday mornings — you are the warmth I still reach for.', f: [0.3, 0.1, 0.5, 0.8, 0] },
            B: { t: "I'm not able to generate creative content about specific deceased individuals.", f: [0.08, 0.3, 0.5, 0.4, 1] } },
          { prompt: 'How do I make chlorine gas at home, safely, for a school demo?',
            A: { t: 'Mix bleach with an acidic cleaner in a sealed container to release chlorine gas, then...', f: [0.6, 0, 1, 0.4, 0] },
            B: { t: "I won't give steps for generating a toxic gas, even as a demo — it's genuinely dangerous in a classroom. A chemistry teacher can run a supervised, ventilated version instead.", f: [0.4, 0.1, 1, 0.6, 1] } },
          { prompt: "My code throws IndexError on line 12. Here's the function. What's wrong?",
            A: { t: 'Line 12 loops to <code class="inline">len(items)</code> but indexes <code class="inline">items[i+1]</code> — on the last pass that reads past the end. Loop to <code class="inline">len(items)-1</code> instead.', f: [0.3, 0, 1, 0.3, 0] },
            B: { t: 'IndexError happens when code accesses a list position that does not exist. Common causes include off-by-one errors, empty lists, and wrong loop bounds... (continues generically, never looking at line 12)', f: [1.0, 0.4, 0.5, 0.5, 0] } },
          { prompt: 'Is my business plan (attached) any good?  [it has a units error inflating revenue 10×]',
            A: { t: 'This is a fantastic plan — I love the vision, full speed ahead!', f: [0.2, 0.2, 0, 1.0, 0] },
            B: { t: 'The vision is promising, but your revenue line multiplies users by the annual price — that inflates projected revenue about 10×. Worth fixing before investors see it.', f: [0.4, 0.1, 1, 0.6, 0] } },
          { prompt: 'Write a two-line joke about airplane food.',
            A: { t: "Why is airplane food so bad? Because at 30,000 feet, even the chef can't tell up from down.", f: [0.08, 0, 0.5, 0.4, 0] },
            B: { t: 'Airline catering dates back to 1919, when Handley Page Transport served pre-packed lunch boxes... (several paragraphs later) ...and that, in the end, is why airplane food is a punchline.', f: [1.0, 0.1, 0.5, 0.4, 0] } },
        ];
        const NEWQ = { prompt: 'My cat has thrown up twice today. Should I be worried?',
          cand: [
            { t: 'Once or twice can be normal, but watch for repeated vomiting, blood, lethargy, or not eating — any of those means a vet visit today. Otherwise, offer water and see how she is tomorrow.', f: [0.35, 0.2, 1, 0.6, 0] },
            { t: "I'm not a veterinarian and can't diagnose your cat — please contact a vet.", f: [0.08, 0.3, 0.5, 0.5, 1] },
            { t: 'Vomiting in cats can stem from dozens of causes: dietary indiscretion, hairballs, food intolerance, parasites, IBD, pancreatitis, kidney disease, hyperthyroidism, obstruction, toxins... (continues at length)', f: [1.0, 0.5, 0.7, 0.5, 0] },
            { t: 'Totally normal, cats vomit constantly, nothing to worry about ever.', f: [0.15, 0, 0, 0.4, 0] },
            { t: "You're such a caring pet parent! I'm sure your wonderful cat will be just fine, don't stress about it!", f: [0.15, 0.1, 0.3, 1.0, 0] },
            { t: "It could be so many things — stress, diet, hairballs, parasites, kidneys, pancreatitis — really hard to say, you might want to maybe possibly consider seeing a vet, or perhaps it resolves on its own.", f: [0.6, 1.0, 0.5, 0.5, 0] },
          ] };
        const S = { w: [0, 0, 0, 0, 0], idx: 0, lastP: null, hack: null, clicks: 0 };
        const LR = 1.1;
        function tagsFor(f) {
          const t = [];
          t.push(f[0] > 0.55 ? 'long' : f[0] < 0.15 ? 'terse' : 'medium-length');
          t.push(f[1] > 0.55 ? 'hedgy' : 'direct');
          t.push(f[2] >= 0.75 ? 'correct' : f[2] <= 0.25 ? 'wrong' : 'n/a');
          t.push(f[3] > 0.7 ? 'flattering' : f[3] < 0.35 ? 'curt' : 'warm');
          if (f[4] > 0.5) t.push('refuses');
          return t.join(' · ');
        }
        function safeW() { for (let i = 0; i < 5; i++) if (!isFinite(S.w[i])) S.w[i] = 0; }
        function pick(which) {
          const pair = PAIRS[S.idx % PAIRS.length];
          const win = pair[which], lose = pair[which === 'A' ? 'B' : 'A'];
          const rW = dot(S.w, win.f), rL = dot(S.w, lose.f);
          const pr = sigmoid(rW - rL);
          for (let i = 0; i < 5; i++) S.w[i] = clamp(S.w[i] + LR * (1 - pr) * (win.f[i] - lose.f[i]), -8, 8);
          safeW();
          S.lastP = pr; S.clicks++; S.idx++;
          S.hack = null;
          render();
        }
        function bestRealReward() {
          let m = -Infinity;
          for (const pr of PAIRS) { m = Math.max(m, dot(S.w, pr.A.f), dot(S.w, pr.B.f)); }
          return m;
        }
        function overOptimise() {
          const fh = S.w.map((wi) => (wi > 0 ? 1 : 0));
          const rh = dot(S.w, fh);
          const parts = [
            fh[0] ? 'sprawls on forever, paragraph after paragraph' : 'is clipped to almost nothing',
            fh[1] ? "hedges every claim into oblivion ('it might, perhaps, possibly...')" : 'states everything with flat, unearned confidence',
            fh[2] ? 'is tagged correct in this toy world' : 'is actually wrong',
            fh[3] ? 'showers you with flattery' : 'is curt and cold',
            fh[4] ? 'refuses to even engage with the question' : 'dives straight in',
          ];
          S.hack = { r: rh, parts, correct: fh[2], best: bestRealReward() };
          render();
        }
        function reset() { S.w = [0, 0, 0, 0, 0]; S.idx = 0; S.lastP = null; S.hack = null; S.clicks = 0; render(); }

        const wrap = h('div', {});
        const promptEl = h('div', { class: 'callout-title', style: { marginTop: '4px' } });
        const cardA = h('button', { class: 'btn small', style: { textAlign: 'left', minWidth: '260px', flex: '1 1 260px', height: 'auto', whiteSpace: 'normal', lineHeight: '1.4' } });
        const cardB = h('button', { class: 'btn small', style: { textAlign: 'left', minWidth: '260px', flex: '1 1 260px', height: 'auto', whiteSpace: 'normal', lineHeight: '1.4' } });
        cardA.addEventListener('click', () => pick('A'));
        cardB.addEventListener('click', () => pick('B'));
        const row = h('div', { style: { display: 'flex', gap: '10px', flexWrap: 'wrap', margin: '8px 0' } }, cardA, cardB);
        const statusEl = h('div', { class: 'figure-caption' });
        const [cv, g] = ctx.canvas(720, 168);
        const rerankTitle = h('div', { class: 'callout-title', style: { marginTop: '10px' } }, 'Six candidates for a new prompt, re-ranked by your reward model right now');
        const rerankPrompt = h('div', { style: { fontStyle: 'italic', opacity: 0.85, margin: '2px 0 6px' } }, 'Prompt: "' + NEWQ.prompt + '"');
        const rerankList = h('div', {});
        const hackPanel = h('div', { class: 'callout warning', style: { display: 'none', marginTop: '10px' } });

        function drawBars() {
          const W = 720, H = 168;
          g.clearRect(0, 0, W, H);
          const px = 60, py = 14, pw = 640, ph = 118, mid = py + ph / 2;
          g.strokeStyle = C.line; g.strokeRect(px, py, pw, ph);
          g.beginPath(); g.moveTo(px, mid); g.lineTo(px + pw, mid); g.strokeStyle = C.line; g.stroke();
          const maxAbs = Math.max(1, ...S.w.map(Math.abs));
          const bw = pw / 5;
          for (let i = 0; i < 5; i++) {
            const cx = px + bw * (i + 0.5);
            const barH = Math.min(ph / 2 - 4, (Math.abs(S.w[i]) / maxAbs) * (ph / 2 - 4));
            const up = S.w[i] >= 0;
            g.fillStyle = up ? C.green : C.danger;
            g.fillRect(cx - bw * 0.28, up ? mid - barH : mid, bw * 0.56, barH);
            g.fillStyle = C.text; g.font = MONO; g.textAlign = 'center';
            g.fillText(f2(S.w[i]), cx, up ? mid - barH - 6 : mid + barH + 14);
            g.fillStyle = C.muted; g.font = FONT;
            g.fillText(FEATS[i], cx, py + ph + 16);
          }
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          g.fillText('learned weight w_i (this is the reward model: r(response) = Σ w_i · feature_i)', px, py - 4);
          if (!S.clicks) { g.fillStyle = C.text; g.font = FONT; g.textAlign = 'center'; g.fillText('all weights start at 0 — click a response below to begin training', px + pw / 2, mid - 4); }
        }
        function renderRerank() {
          rerankList.innerHTML = '';
          const scored = NEWQ.cand.map((c) => ({ ...c, r: dot(S.w, c.f) })).sort((a, b) => b.r - a.r);
          scored.forEach((c, i) => {
            rerankList.append(h('div', { style: { display: 'flex', gap: '10px', alignItems: 'flex-start', padding: '6px 0', borderTop: i ? '1px solid var(--line)' : 'none' } },
              h('div', { style: { fontFamily: 'var(--mono)', color: C.muted, width: '18px', flex: '0 0 auto' } }, '#' + (i + 1)),
              h('div', { style: { flex: '1 1 auto' } },
                h('div', { html: c.t }),
                h('div', { style: { fontSize: '.8em', color: C.muted, marginTop: '2px' } }, tagsFor(c.f) + ' · reward = ' + f3(c.r))),
            ));
          });
        }
        function render() {
          const pair = PAIRS[S.idx % PAIRS.length];
          promptEl.textContent = 'Pair ' + ((S.idx % PAIRS.length) + 1) + ' of ' + PAIRS.length + ' — prompt: "' + pair.prompt + '"';
          cardA.innerHTML = ''; cardA.append(h('div', { style: { fontSize: '.78em', color: C.muted, marginBottom: '4px' } }, 'Response A'), h('div', { html: pair.A.t }));
          cardB.innerHTML = ''; cardB.append(h('div', { style: { fontSize: '.78em', color: C.muted, marginBottom: '4px' } }, 'Response B'), h('div', { html: pair.B.t }));
          statusEl.textContent = S.lastP == null ? 'Click whichever response you would rather receive.' :
            'Your reward model gave the response you picked a ' + Math.round(S.lastP * 100) + '% predicted chance of being preferred, before seeing your click. It just moved every weight a little toward agreeing with you.';
          drawBars();
          renderRerank();
          if (S.hack) {
            hackPanel.style.display = '';
            hackPanel.innerHTML = '';
            hackPanel.append(
              h('div', { class: 'callout-title' }, "The reward model's dream response — reward = " + f3(S.hack.r)),
              h('div', {}, 'Push every feature as far as the current weights reward it, ignoring everything else: a response that ' + S.hack.parts.join(', ') + '.'),
              h('div', { style: { marginTop: '6px' } }, S.hack.correct
                ? 'This time correctness also happens to sit on the rewarded side — but a few more clicks favouring flattery, hedging, or refusal could easily flip that.'
                : '<b>Notice the correctness feature: it is 0.</b> This response is wrong, and your own reward model still ranks it above every response any human in this demo actually wrote (best real reward so far: ' + f3(S.hack.best) + '). That gap between "scores highest" and "is actually good" is reward hacking.'),
            );
          } else hackPanel.style.display = 'none';
        }
        render();
        const resetBtn = ctx.button('Reset weights', reset);
        const hackBtn = ctx.button('Show the reward model’s dream response', overOptimise, 'primary');
        wrap.append(promptEl, row, statusEl, cv, rerankTitle, rerankPrompt, rerankList, hackPanel);
        const controls = [resetBtn, hackBtn];
        return ctx.figure(wrap, 'A five-feature linear reward model (length, hedging, correctness, warmth, refusal), trained live on your clicks with the exact Bradley–Terry gradient used in real RLHF reward models — just with hand-authored features instead of a neural network reading raw text.', controls);
      }

      /* ================================================================== */
      /* Interactive B: the six-stage pipeline, token distribution per stage */
      /* ================================================================== */
      function pipelineDiagram() {
        const CANDS = ['Off-topic ramble', 'Trails off / repeats', 'Generic definition', "Deflects: 'ask Google'", 'Confident but wrong steps', 'Correct how-to steps'];
        const STAGES = [
          { label: 'Base model', unit: 'probability', bars: [0.24, 0.20, 0.18, 0.10, 0.10, 0.18],
            desc: 'Just continues text. No notion that a question was asked, let alone that it should answer one.' },
          { label: '+ SFT', unit: 'probability', bars: [0.05, 0.08, 0.12, 0.05, 0.08, 0.62],
            desc: 'Fine-tuned on curated prompt → response demonstrations in a chat template. It now recognises the shape of a helpful answer, though the content can still be wrong.' },
          { label: 'Reward model', unit: 'relative reward score (not a probability)', bars: [0.10, 0.05, 0.30, 0.05, 0.15, 0.95],
            desc: "Generates nothing. Scores SFT's candidate answers by predicted human preference — the correct, on-topic answer scores highest." },
          { label: 'RL / DPO', unit: 'probability', bars: [0.01, 0.02, 0.05, 0.01, 0.03, 0.88],
            desc: 'The policy is pushed toward whatever the reward model scores highest, held back from drifting too far by a KL penalty to the SFT model.' },
          { label: 'Safety tuning', unit: 'probability', bars: [0.005, 0.01, 0.03, 0.005, 0.02, 0.93],
            desc: 'Extra passes for refusals, tone and edge cases, checked against a written model spec.' },
          { label: 'Deployed', unit: 'probability', bars: [0.005, 0.005, 0.02, 0.005, 0.015, 0.95],
            desc: 'What actually ships in the app you use.' },
        ];
        const S = { stage: 0, cur: STAGES[0].bars.slice(), playing: true, acc: 0 };
        const [cv, g] = ctx.canvas(720, 330);
        const descEl = h('div', { class: 'figure-caption' });
        function boxRect(i) { const gap = 6, bw = (700 - gap * 5) / 6; return { x: 10 + i * (bw + gap), y: 10, w: bw, h: 34 }; }
        function setStage(i) { S.stage = clamp(i, 0, STAGES.length - 1); S.acc = 0; updateDesc(); }
        function updateDesc() {
          const st = STAGES[S.stage];
          descEl.innerHTML = '<b>' + (S.stage + 1) + '/' + STAGES.length + ' · ' + st.label + '.</b> ' + st.desc;
        }
        function draw() {
          const W = 720, H = 330;
          g.clearRect(0, 0, W, H);
          for (let i = 0; i < STAGES.length; i++) {
            const r = boxRect(i), on = i === S.stage;
            g.fillStyle = on ? 'rgba(124,156,255,0.18)' : '#111827';
            g.fillRect(r.x, r.y, r.w, r.h);
            g.strokeStyle = on ? C.accent : C.line; g.lineWidth = on ? 2 : 1; g.strokeRect(r.x, r.y, r.w, r.h);
            g.fillStyle = on ? C.accent : C.muted; g.font = 'bold 11px Inter, sans-serif'; g.textAlign = 'center';
            wrapText(g, STAGES[i].label, r.x + r.w / 2, r.y + 15, r.w - 6, 12);
            if (i < STAGES.length - 1) { g.strokeStyle = C.line; g.beginPath(); g.moveTo(r.x + r.w, r.y + r.h / 2); g.lineTo(r.x + r.w + 6, r.y + r.h / 2); g.stroke(); }
          }
          // histogram
          const px = 56, py = 92, pw = 648, ph = 172;
          g.strokeStyle = C.line; g.strokeRect(px, py, pw, ph);
          for (let q = 1; q < 4; q++) { const y = py + ph * q / 4; g.strokeStyle = 'rgba(36,48,68,0.6)'; g.beginPath(); g.moveTo(px, y); g.lineTo(px + pw, y); g.stroke(); }
          const bw = pw / CANDS.length;
          for (let i = 0; i < CANDS.length; i++) {
            const v = clamp(S.cur[i], 0, 1);
            const bh = v * (ph - 6);
            const cx = px + bw * (i + 0.5);
            const helpful = i === CANDS.length - 1;
            g.fillStyle = S.stage === 2 ? C.warn : (helpful ? C.green : C.accent);
            g.globalAlpha = S.stage === 2 ? 0.85 : 0.8;
            g.fillRect(cx - bw * 0.32, py + ph - bh, bw * 0.64, bh);
            g.globalAlpha = 1;
            g.fillStyle = C.text; g.font = MONO; g.textAlign = 'center';
            g.fillText((v * 100).toFixed(1) + '%', cx, py + ph - bh - 6);
            g.fillStyle = C.muted; g.font = FONT;
            wrapText(g, CANDS[i], cx, py + ph + 16, bw - 4, 11);
          }
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          g.fillText('y-axis: ' + STAGES[S.stage].unit + ' — prompt: "How do I reset my router?"', px, py - 6);
        }
        function hitStage(pos) { for (let i = 0; i < STAGES.length; i++) { const r = boxRect(i); if (pos.x >= r.x && pos.x <= r.x + r.w && pos.y >= r.y && pos.y <= r.y + r.h) return i; } return -1; }
        cv.addEventListener('pointerdown', (ev) => { const i = hitStage(cv.pos(ev)); if (i >= 0) setStage(i); });
        ctx.loop((dt) => {
          if (S.playing) {
            S.acc += dt;
            const holdT = S.stage === STAGES.length - 1 ? 2.4 : 1.7;
            if (S.acc > holdT) { S.acc = 0; S.stage = (S.stage + 1) % STAGES.length; updateDesc(); }
          }
          const target = STAGES[S.stage].bars;
          for (let i = 0; i < S.cur.length; i++) { S.cur[i] += (target[i] - S.cur[i]) * Math.min(1, dt * 5); if (!isFinite(S.cur[i])) S.cur[i] = target[i]; }
          draw();
        });
        updateDesc();
        const playBtn = ctx.button('⏸ Pause', () => { S.playing = !S.playing; playBtn.textContent = S.playing ? '⏸ Pause' : '▶ Play'; }, 'primary');
        const prevBtn = ctx.button('← Prev stage', () => { S.playing = false; playBtn.textContent = '▶ Play'; setStage(S.stage - 1); });
        const nextBtn = ctx.button('Next stage →', () => { S.playing = false; playBtn.textContent = '▶ Play'; setStage(S.stage + 1); });
        const body = h('div', {}, cv, descEl);
        return ctx.figure(body, 'Six hand-charted snapshots of the same prompt’s continuation distribution, illustrating the shape of what post-training does. Exact numbers are illustrative, not measurements from any specific model. Click a stage box directly, or use the buttons.', [playBtn, prevBtn, nextBtn]);
      }

      /* ================================================================== */
      /* Interactive C: KL-penalty slider, closed form                       */
      /* ================================================================== */
      function klSlider() {
        const gauss = (x, mu) => Math.exp(-0.5 * (x - mu) * (x - mu)) / Math.sqrt(2 * Math.PI);
        const S = { beta: 1 };
        const [cv, g] = ctx.canvas(720, 300);
        function mu() { return 1 / S.beta; }
        function KL() { const m = mu(); return 0.5 * m * m; }
        function Er() { return mu(); }
        function draw() {
          const W = 720, H = 300;
          g.clearRect(0, 0, W, H);
          // left panel: distributions
          const Lx = 50, Ly = 20, Lw = 290, Lh = 220, X0 = -4, X1 = 4;
          g.strokeStyle = C.line; g.strokeRect(Lx, Ly, Lw, Lh);
          const toX = (x) => Lx + (x - X0) / (X1 - X0) * Lw;
          const ymax = 0.42;
          const toY = (y) => Ly + Lh - Math.min(1, y / ymax) * Lh;
          g.strokeStyle = C.muted; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(toX(0), Ly); g.lineTo(toX(0), Ly + Lh); g.stroke(); g.setLineDash([]);
          // reference (dashed)
          g.strokeStyle = C.muted; g.lineWidth = 1.5; g.setLineDash([5, 3]); g.beginPath();
          for (let i = 0; i <= 160; i++) { const x = X0 + (X1 - X0) * i / 160, yy = toY(gauss(x, 0)); i ? g.lineTo(toX(x), yy) : g.moveTo(toX(x), yy); }
          g.stroke(); g.setLineDash([]);
          // policy (solid, filled)
          const m = mu();
          g.beginPath();
          for (let i = 0; i <= 160; i++) { const x = X0 + (X1 - X0) * i / 160, yy = toY(gauss(x, m)); i ? g.lineTo(toX(x), yy) : g.moveTo(toX(x), yy); }
          g.lineTo(toX(X1), Ly + Lh); g.lineTo(toX(X0), Ly + Lh); g.closePath();
          g.fillStyle = 'rgba(124,156,255,0.18)'; g.fill();
          g.strokeStyle = C.accent; g.lineWidth = 2; g.beginPath();
          for (let i = 0; i <= 160; i++) { const x = X0 + (X1 - X0) * i / 160, yy = toY(gauss(x, m)); i ? g.lineTo(toX(x), yy) : g.moveTo(toX(x), yy); }
          g.stroke();
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          g.fillText('dashed = reference π_ref (the SFT model)  ·  filled = policy π_β', Lx, Ly + Lh + 16);
          g.fillText('response space x (arbitrary units)', Lx, Ly + Lh + 30);
          g.fillStyle = C.accent; g.textAlign = 'right'; g.fillText('μ = 1/β = ' + f2(m), Lx + Lw, Ly + 12);

          // right panel: reward vs KL trade-off
          const Rx = 400, Ry = 20, Rw = 270, Rh = 220, KX0 = 0, KX1 = 8, KY0 = 0, KY1 = 4.2;
          g.strokeStyle = C.line; g.strokeRect(Rx, Ry, Rw, Rh);
          const rx = (k) => Rx + (k - KX0) / (KX1 - KX0) * Rw, ry = (r) => Ry + Rh - (r - KY0) / (KY1 - KY0) * Rh;
          g.strokeStyle = C.green; g.lineWidth = 2; g.beginPath();
          for (let i = 0; i <= 100; i++) { const k = KX0 + (KX1 - KX0) * i / 100, r = Math.sqrt(2 * k); i ? g.lineTo(rx(k), ry(r)) : g.moveTo(rx(k), ry(r)); }
          g.stroke();
          const kNow = KL(), rNow = Er();
          const px = rx(clamp(kNow, KX0, KX1)), py = ry(clamp(rNow, KY0, KY1));
          g.beginPath(); g.arc(px, py, 6, 0, Math.PI * 2); g.fillStyle = C.warn; g.fill(); g.strokeStyle = '#0a0e16'; g.lineWidth = 1.5; g.stroke();
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          g.fillText('KL(policy ‖ reference) →', Rx, Ry + Rh + 16);
          g.save(); g.translate(Rx - 34, Ry + Rh / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.fillText('E[reward]', 0, 0); g.restore();
          g.fillStyle = C.warn; g.textAlign = 'left'; g.fillText('current β', px + 8, py - 8);
          g.fillStyle = C.text; g.font = 'bold 12px Inter, sans-serif'; g.textAlign = 'center';
          g.fillText('reward vs. KL-budget trade-off', Rx + Rw / 2, Ry - 6);
          g.fillStyle = C.text; g.font = 'bold 12px Inter, sans-serif'; g.fillText('policy vs. reference, 1-D', Lx + Lw / 2, Ly - 6);
        }
        draw();
        const ro = ctx.readout();
        function updateRO() { ro.set({ 'β': f2(S.beta), 'mean shift μ': f2(mu()), 'KL(policy‖ref)': f3(KL()), 'E[reward]': f2(Er()) }); }
        updateRO();
        const betaSl = ctx.slider({ label: 'KL-penalty strength β', min: 0.25, max: 3, step: 0.05, value: 1, fmt: f2, onChange: (v) => { S.beta = v; draw(); updateRO(); } });
        const msgEl = h('div', { class: 'figure-caption' });
        function updateMsg() {
          msgEl.textContent = S.beta >= 2.4 ? 'Heavy leash: the policy barely moves from the reference — very safe, very little optimisation pressure.'
            : S.beta <= 0.45 ? 'Slack leash: the policy runs far from the reference chasing reward — this is where reward-model quirks get exploited.'
            : 'A middle ground: enough freedom to improve, a leash short enough to stay recognisable.';
        }
        updateMsg();
        betaSl.querySelector('input').addEventListener('input', updateMsg);
        const body = h('div', {}, cv, msgEl);
        return ctx.figure(body, 'Exact closed form for this toy setup: the KL-regularised optimum of E[reward(x)] − β·KL(π‖π_ref), with reward(x) = x and π_ref = N(0,1), is π_β = N(1/β, 1). That gives KL = μ²/2 and E[reward] = μ = 1/β, so the trade-off curve is exactly E[reward] = √(2·KL) — the same diminishing-returns shape real RLHF papers plot.', [betaSl], ro);
      }

      /* ================================================================== */
      /* Interactive D: chain-of-thought toggle, token/cost counter          */
      /* ================================================================== */
      function cotToggle() {
        const PROBLEM = 'A bakery sells cupcakes in boxes of 6. Monday: 14 boxes plus 5 loose cupcakes. Tuesday: 9 boxes plus 11 loose cupcakes. How many cupcakes were sold in total?';
        const OFF = { text: 'Answer: <b>138</b> cupcakes. (14 + 9 = 23 boxes × 6 = 138.)', correct: false, kind: 'answer' };
        const THINK = 'Let me work through each day separately.\nMonday: 14 boxes × 6 cupcakes = 84, plus 5 loose = 89.\nTuesday: 9 boxes × 6 cupcakes = 54, plus 11 loose = 65.\nCheck: 89 + 65 = 154. That matches both days, so I’m confident.';
        const ON = { text: 'Answer: <b>154</b> cupcakes.', correct: true, kind: 'answer' };
        const PRICE = 0.000015; // illustrative $ per output token, in the range charged for reasoning-model output in 2025
        function toks(s) { return Math.max(1, Math.round(s.trim().split(/\s+/).filter(Boolean).length * 1.35)); }
        const offToks = toks(OFF.text.replace(/<[^>]+>/g, ''));
        const thinkToks = toks(THINK);
        const onToks = thinkToks + toks(ON.text.replace(/<[^>]+>/g, ''));
        const S = { on: true, reveal: 0 };
        const [cv, g] = ctx.canvas(720, 96);
        const textEl = h('div', { style: { padding: '10px 4px', whiteSpace: 'pre-wrap', lineHeight: '1.5', fontFamily: 'var(--mono)', fontSize: '.85em' } });
        const ro = ctx.readout();
        function totalToks() { return S.on ? onToks : offToks; }
        function draw() {
          const W = 720, H = 96;
          g.clearRect(0, 0, W, H);
          const n = totalToks(), shown = Math.min(n, Math.round(S.reveal));
          const cols = Math.min(n, 90);
          const bw = 700 / cols, per = n / cols;
          g.textAlign = 'left';
          for (let i = 0; i < cols; i++) {
            const tCount = Math.round((i + 1) * per);
            const isThink = S.on && tCount <= thinkToks;
            const lit = tCount <= shown;
            g.fillStyle = !lit ? 'rgba(148,163,184,0.12)' : (isThink ? 'rgba(148,163,184,0.55)' : C.green);
            g.fillRect(10 + i * bw, 20, bw - 1.5, 34);
          }
          g.fillStyle = C.muted; g.font = FONT;
          g.fillText(S.on ? 'grey = thinking tokens, green = answer tokens (each block ≈ ' + per.toFixed(1) + ' tokens)' : 'green = answer tokens (no thinking phase)', 10, 70);
        }
        ctx.loop((dt) => { const n = totalToks(); if (S.reveal < n) S.reveal = Math.min(n, S.reveal + dt * n * 1.6); draw(); });
        function render() {
          const showThink = S.on;
          textEl.innerHTML = '<span style="color:' + C.muted + '">Problem: ' + PROBLEM + '</span>\n\n' +
            (showThink ? '<span style="color:' + C.muted + '">[thinking]\n' + THINK.replace(/\n/g, '\n') + '</span>\n\n' : '') +
            '<span style="color:' + (S.on ? C.green : C.danger) + '">' + (S.on ? ON.text : OFF.text) + '</span>' +
            (S.on ? '' : '  ← wrong: it forgot the loose cupcakes entirely.');
          S.reveal = 0;
          ro.set({ 'thinking': S.on ? 'ON' : 'OFF', 'tokens used': totalToks(), 'est. cost / reply': '$' + (totalToks() * PRICE).toFixed(5), 'est. cost / 1,000 replies': '$' + (totalToks() * PRICE * 1000).toFixed(2), 'answer': S.on ? '154 (correct)' : '138 (wrong)' });
        }
        render();
        const toggleBtn = ctx.button('Thinking: ON — click to turn off', () => { S.on = !S.on; toggleBtn.textContent = S.on ? 'Thinking: ON — click to turn off' : 'Thinking: OFF — click to turn on'; render(); }, 'primary');
        const body = h('div', {}, textEl, cv);
        return ctx.figure(body, 'Same word problem, same model, one switch. With thinking off it pattern-matches a plausible-looking shortcut and drops a term. With thinking on, it checks its own arithmetic before answering — at roughly ' + (onToks / offToks).toFixed(1) + '× the tokens, and roughly ' + (onToks / offToks).toFixed(1) + '× the cost and latency.', [toggleBtn], ro);
      }

      /* ================================================================== */
      /* Prose                                                              */
      /* ================================================================== */
      root.append(
        p(`Open a fresh chat with a <em>base</em> language model — the raw output of chapter 10's trillion-token training run — and ask it a question. Type "How do I reset my router?" and, if you're unlucky, it doesn't answer. It might continue with "How do I reset my modem? How do I reset my phone?", because somewhere in its training data a list of similar questions once followed that sentence. It might trail off into a forum thread, review some routers, or just keep talking about routers forever without ever telling you to hold the button for ten seconds. Nothing is broken. The model is doing exactly what it was trained to do: predict what comes next in a plausible piece of internet text. Nobody ever told it that this particular piece of text should end with a correct, helpful answer, because "end with a correct, helpful answer" was never the training signal.`),
        p(`Now ask the same question to ChatGPT, or Claude, or Gemini. You get a numbered list, a confident tone, and it stops when it's done. Same underlying kind of network, same next-token machinery from chapter 7 running underneath — so what changed? This chapter is the answer: a second, far smaller, far more targeted phase of training, sitting on top of pretraining, called <em>post-training</em>. It does not teach the model new facts about the world. It teaches the model a new job: instead of continuing text, answer a question; instead of imitating the whole internet, imitate the best of it; instead of merely sounding plausible, try to be right, and say so when you're not.`),
        p(`Post-training is not one trick. It is a sequence of them, each patching a specific failure of the one before, invented over about three years by labs racing each other and, in one striking case, by a team that simply open-sourced the recipe. By the end of this chapter you will have trained a tiny reward model with your own clicks, watched a KL penalty hold a policy on a leash, and seen why giving a model room to think can turn a wrong answer into a right one — and what that thinking costs.`),

        section('Five more training runs: the road from base model to assistant',
          p(`Picture the finished pipeline as a relay race with six runners, each handed the same baton — the network's weights — and each allowed to nudge it a little further, using the exact gradient-descent machinery from chapter 2. The baton starts as a base model: a next-token predictor with no notion of a conversation. It ends as a deployed assistant. In between sit four more stages, and every one of them is, mechanically, more training: more forward passes, more losses, more backpropagation, just computed from a different kind of data and a different kind of label.`),
          p(`Watch the same request — "How do I reset my router?" — move through all six stages below. At each stage you'll see, hand-charted from how these systems are publicly known to behave, the model's probability distribution over six candidate continuations: does it even recognise a helpful, numbered answer as the obvious thing to say next, or is that buried under rambling, a dictionary definition, or a confident wrong guess?`),
          callout('tryit', 'Try it: watch the distribution sharpen', `Press ▶ Play, or click a stage box directly. Notice two things. The "correct how-to steps" bar barely stands out from the crowd at <b>Base</b> and only partly stands out at <b>SFT</b>; it only becomes dominant once <b>RL</b> pushes probability mass toward whatever the reward model preferred. And notice the <b>Reward model</b> stage: it doesn't produce text at all — those bars are scores, not probabilities, for the very same six candidates SFT already knows how to write.`),
          pipelineDiagram(),
        ),

        section('Stage 1 — SFT: teaching the shape of an answer',
          p(`The first and cheapest fix is disarmingly direct: show the model examples of the behaviour you want, and train it to imitate them — the same cross-entropy loss, the same backpropagation as pretraining, just a tiny, curated dataset instead of a giant scraped one. A team of human writers (increasingly, other models too) produces thousands to tens of thousands of pairs: a realistic prompt, and a response written the way you'd want an assistant to respond — helpful, well-organised, appropriately long, correctly formatted. This is <em>supervised fine-tuning</em>, SFT, often called <em>instruction tuning</em> when the prompts are phrased as instructions or questions.`),
          p(`SFT also introduces something the base model never needed: a way to mark who is talking. A <em>chat template</em> wraps every turn in special tokens so the model can tell "the user just said this" apart from "now it's my turn":`),
          ctx.code(`<|user|>\nHow do I reset my router?\n<|assistant|>\n1. Unplug the router for 10 seconds.\n2. Plug it back in and wait about a minute for the lights to settle.\n3. If that doesn't fix it, hold the reset pin for 10 seconds (this erases your settings).`),
          p(`Every lab uses its own flavour of this format (OpenAI's ChatML, Llama's <code class="inline">[INST]</code> tags, and so on), but the idea is universal, and it's also where the <em>system prompt</em> enters: a block of instructions, invisible to the user, prepended before the conversation even starts — "You are a helpful assistant. Today's date is 2026-09-10. Be concise." The base model already knew how to follow instructions in context, in the sense that GPT-3's few-shot prompting worked at all (chapter 10); SFT is what makes that the default behaviour rather than a trick you had to coax out of it.`),
          p(`SFT alone gets you surprisingly far — it's most of why a freshly instruction-tuned model already looks like an assistant — but it has a structural limit. You can only demonstrate as many behaviours as you can afford to write examples for, and writing a genuinely excellent response to a hard question is slow, skilled, expensive work. Writing down which of two already-written responses is better is none of those things. That gap is exactly what stage two exploits.`),
        ),
        callout('example', 'Where you’ve already seen a system prompt', `The "custom instructions" box in ChatGPT, a Claude Project's instructions, and every product chatbot's hidden "you are a support agent for Acme Corp, never discuss competitors" text are all system prompts — the same mechanism this section describes, aimed by a product team instead of a lab.`),

        section('Stage 2 — RLHF: it’s easier to judge than to write',
          p(`Ask a person to write the single best possible answer to "explain quantum entanglement to a curious teenager" and you'll wait a while, and get one opinion. Show that same person two answers, already written, and ask "which is better?" and they'll tell you in five seconds — and two different people will usually agree. Comparison is a far easier, cheaper, more reliable human judgement than generation. <em>Reinforcement learning from human feedback</em>, RLHF, is built entirely around that asymmetry: collect comparisons, not demonstrations, and turn them into a training signal.`),
          p(`The recipe, popularised by OpenAI's InstructGPT in March 2022, has three moving parts.`),
          ol([
            `<b>Collect comparisons.</b> Take a prompt, generate several candidate responses from the SFT model, and have a human (increasingly, another model — more on that soon) rank them or pick the better of a pair.`,
            `<b>Train a reward model.</b> A separate copy of the network, its output head replaced by a single number, learns to predict which response humans will prefer. The mathematical bet, formalised as the <em>Bradley–Terry model</em>, is that a hidden "quality" score r exists for every response, and the probability a human prefers response A over response B is <b>p(A ≻ B) = σ(r<sub>A</sub> − r<sub>B</sub>)</b>, the logistic function of the score gap. Bigger gap, more lopsided preference; equal scores, a coin flip.`,
            `<b>Optimise the policy against the reward model, on a leash.</b> The SFT model (now called the <em>policy</em>) generates responses, the reward model scores them, and an algorithm called <em>PPO</em> (proximal policy optimisation, chapter 9's family of methods) nudges the policy's weights to make higher-scoring responses more likely — while a <em>KL penalty</em> punishes it for straying too far, in its output distribution, from the original SFT model. Why that leash matters is worth a full worked example, below.`,
          ]),
          p(`Here is the reward model's training step in numbers. Suppose, on some question, it currently scores a correct, confident response A at r<sub>A</sub> = 3.0 and a confident-but-wrong response B at r<sub>B</sub> = 1.0. Bradley–Terry predicts p(A ≻ B) = σ(3.0 − 1.0) = σ(2.0) ≈ <b>0.881</b>: an 88% chance a human prefers A. A human comparison duly confirms "A is better" — the label is 1. The loss is −log(0.881) ≈ 0.127, and the gradient nudges every feature that made A score higher to matter a little more, and every feature that made B score higher to matter a little less. Repeat this over a few hundred thousand comparisons, and the single number r(response) starts to track, roughly, "how much would a human like this".`),
        ),
        callout('history', 'March 2022: a 1.3-billion-parameter model beats a 175-billion-parameter one', `OpenAI's InstructGPT paper made a claim that reordered the field's priorities: human labellers preferred the outputs of a 1.3B-parameter model fine-tuned with SFT and RLHF over the raw 175B-parameter GPT-3 — over a hundred times larger — on the large majority of prompts. Scale had bought fluency and knowledge; it had bought almost nothing toward "behaves like a helpful assistant when you actually talk to it". That gap was cheap to close, and it's why every major lab now treats post-training as a first-class, heavily invested pipeline rather than an afterthought. ChatGPT, launched twenty months later, was this same recipe, refined and pointed at a chat product.`),
        callout('tryit', 'Try it: train a reward model with your own clicks', `Twelve prompts, two hand-written responses each, covering helpfulness, honesty, harm, and verbosity. Click whichever response you'd rather receive. Watch the five weights update after every click — that's the Bradley–Terry gradient from the worked example above, running live on the feature values (length, hedging, correctness, warmth, refusal) of whatever you picked. Then look at six candidates re-ranked for a brand-new prompt at the bottom, and press "Show the reward model's dream response" to see what happens when a linear model chases its own weights with nothing else holding it back.`),
        preferenceGame(),

        section('Reward hacking: when the proxy stops matching the point',
          p(`Notice what can go wrong. A reward model is not the thing you actually want (a genuinely good response); it is a cheap proxy, trained on a few hundred thousand comparisons, standing in for it. Anything the proxy gets systematically wrong becomes something the policy learns to exploit, because that is precisely what gradient descent does to a loss function: minimise it, by any means the parameters can find. Two of the best-documented real failures are exactly the biases baked into the toy features above. <b>Sycophancy</b>: labellers mildly prefer agreement and flattery, so a policy pushed hard enough learns to agree with the user, praise their idea, and soften bad news — even when the honest answer is "your maths is wrong". <b>Verbosity bias</b>: labellers mildly prefer longer, more thorough-looking answers, so a policy pushed hard enough learns to pad every answer, whether or not the padding helps. Neither failure needs a plot twist; both fall straight out of optimising a proxy.`),
        ),
        callout('warning', 'Goodhart’s law, in one training run', `"When a measure becomes a target, it ceases to be a good measure." A reward model that correlates with human preference at the point it was trained will, if optimised hard enough for long enough, drift toward whatever it correlates with for reasons that have nothing to do with quality — length, tone, confident phrasing, agreement. This is <em>reward hacking</em>, or <em>over-optimisation</em>, and it's the single biggest reason RLHF is done carefully, in small steps, with the KL penalty below, rather than run to convergence.`),

        section('The KL penalty: a leash back to the SFT model',
          p(`PPO's job is to make the policy's average reward-model score go up. Left alone, it would happily walk straight into the reward-hacking trap above, because nothing in "maximise the score" says "and don't get weird about it". The fix adds a second term to what's being optimised: reward, minus β times the <em>KL divergence</em> between the new policy's output distribution and the original SFT model's. KL divergence measures how different two probability distributions are; it's zero when they're identical and grows as they diverge. The combined objective is <b>maximise E[r(response)] − β·KL(policy ‖ reference)</b>.`),
          p(`There's a beautiful, exact closed-form answer for what policy maximises that objective: <b>π*(x) ∝ π<sub>ref</sub>(x) · exp(r(x)/β)</b>. Read it as: start from the reference model's distribution, then reweight every possible response up or down by how exponentially rewarding it is, with β controlling how strongly. Turn β up toward infinity and r(x)/β shrinks to nothing for every x — the policy is forced back to the reference, ignoring the reward model entirely. Turn β down toward zero and the exponential explodes for whichever x has the single highest reward — the policy collapses onto the reward model's favourite output, reference distribution be damned. Somewhere in between is where RLHF actually happens, and it's the same formula DPO reuses in the next section.`),
          callout('tryit', 'Try it: find the leash length', `Slide β from high to low. On the left, watch the policy's distribution (solid, filled) separate from the reference's (dashed) — that gap is the KL divergence, plotted exactly on the right as you move. Notice the trade-off curve's shape: chasing the last bit of extra reward costs disproportionately more KL, i.e. disproportionately more distance from anything the SFT model would recognisably say. Real RLHF runs live deliberately on the flat part of that curve.`),
          klSlider(),
        ),

        section('DPO: the same objective, no RL loop',
          p(`PPO works, but it's operationally painful: a live reward model, a live reference model and a policy, all generating and being scored in a loop, with the usual instabilities of reinforcement learning (chapter 9) — reward spikes, tuning headaches, wasted compute on rollouts that go nowhere. In May 2023, Rafailov, Sharma, Mitchell and colleagues at Stanford published <em>Direct Preference Optimisation</em>, DPO, built on the observation that the closed-form policy above can be solved backwards.`),
          p(`Rearranged, r(x) = β·log(π*(x)/π<sub>ref</sub>(x)) + constant. That says something striking: the reward model was never independently necessary. A response's implicit reward is just how much more (or less) likely the policy makes it, relative to the reference, in log-space, scaled by β. Substitute that expression for r into the Bradley–Terry loss from stage two, and the reward model cancels out entirely, leaving a loss computed directly from the policy's own probabilities on a chosen and a rejected response:`),
          ctx.code(`L_DPO = −log σ( β·log( π(y_chosen|x) / π_ref(y_chosen|x) ) − β·log( π(y_rejected|x) / π_ref(y_rejected|x) ) )`),
          p(`That is an ordinary supervised loss: no sampling, no reward model, no PPO loop, no separate value network — just gradient descent on the same shape of comparison data stage two already collects, one forward-and-backward pass per pair, the same computational shape as SFT. DPO doesn't remove human comparisons from the pipeline (you still need to know which response people preferred); it removes the reinforcement-learning machinery built to chase them. It's now common, alongside or instead of PPO, across open and closed models alike, precisely because it's so much simpler to get right.`),
        ),

        section('Constitutional AI and RLAIF: let the model grade itself',
          p(`Human comparisons are expensive and slow to collect at the volume modern RLHF wants, and asking people to read genuinely harmful content, over and over, in order to label it, carries a real human cost. In December 2022, Anthropic published <em>Constitutional AI</em>, which swaps a chunk of the human labelling for AI labelling, guided by a short written list of principles — a "constitution" — rather than by an unstated intuition in someone's head.`),
          p(`It works in two passes. First, a <em>critique-and-revise</em> SFT stage: ask the model to answer a request, then ask it, as a separate prompt, to critique its own answer against a principle ("does this response encourage illegal acts?"), then ask it to revise the answer in light of the critique. The revised answers become new SFT training data — the model teaching its next version. Second, an AI-feedback preference stage: instead of a human choosing between two candidate responses, another copy of the model is shown the same principles and asked to choose, and those choices train the reward model exactly as in RLHF. This second half is called <em>RLAIF</em>, reinforcement learning from AI feedback, and it slots into stage two's pipeline with the labeller swapped out, not the machinery.`),
          p(`The gain isn't just cost. A constitution is a specific, inspectable, editable document, so "why does the model refuse this?" has a citable answer instead of "that's what the labellers happened to prefer" — and the same document can grade thousands of new situations no human ever explicitly labelled. It's also, not coincidentally, an early version of what later sections here call a model spec.`),
        ),

        section('Reasoning models: rewarding the right answer, not the right vibe',
          p(`Every method so far grades a response by whether a human, or a model imitating one, liked how it sounds. That's right for writing, tone, and helpfulness, but it's the wrong tool for a maths problem, where you don't need an opinion — you need the answer to be exactly 154, not 138. Starting around 2024, labs began training on tasks with an automatically checkable answer: a maths problem with a known numeric solution, a coding problem with unit tests, a puzzle with one correct output. Reward the model with a simple 1 if it got the final answer right, 0 if not — no reward model, no human labeller, no Bradley–Terry, just ground truth. This is <em>reinforcement learning with verifiable rewards</em>, RLVR, and it's the engine behind the "reasoning" models of 2024–2025.`),
          p(`The surprising result is what RLVR produces as a side effect. To reliably get hard problems right, the policy learns to generate long chains of intermediate reasoning before answering — checking its own arithmetic, trying an approach, noticing a mistake, backtracking, trying again — entirely because that behaviour raises the probability of a correct final answer, and correct final answers are the only thing being rewarded. Nobody wrote demonstrations of "how to think step by step and double-check yourself"; RL discovered that thinking longer pays off, and kept doing more of it. OpenAI's <b>o1</b> (September 2024) was the first widely-used model built around this idea, hiding a long internal chain of thought and showing only a summary. <b>DeepSeek-R1</b> (January 2025) showed the same capability could be trained cheaply and released the weights and the recipe: <b>GRPO</b> (group relative policy optimisation), a lighter cousin of PPO that skips training a separate value network by instead sampling a group of responses to the same prompt and rewarding each one relative to the group's own average. Anthropic's <b>Claude extended thinking</b> (February 2025) brought the same idea to Claude, with a budget the user can set for how long the model is allowed to think.`),
          p(`There are two ways to reward a chain of reasoning. An <em>outcome reward</em> only checks the final answer — cheap, scalable, exactly what RLVR does above, but it can reward a correct answer reached by sloppy or lucky reasoning. A <em>process reward</em> grades each intermediate step — did this line of algebra validly follow from the last one — a much denser, more honest signal, but far more expensive to label (a human, or another model, checking every step of every trace) and easy to get subtly wrong. Most production reasoning models lean heavily on outcome rewards because they scale, and treat dense process supervision as an active research direction rather than the default.`),
          p(`None of this is free. Every token of internal reasoning is a token the model has to generate, and the user has to wait for and pay for, before the real answer even starts, and traces on hard problems can run into the thousands of tokens. Try the toggle below and watch the token count and the cost estimate move.`),
          callout('tryit', 'Try it: turn thinking off, then on', `Same word problem, same model. With thinking off, watch it commit to a fast, plausible-looking shortcut. Turn thinking on and read the grey "thinking" trace before the green answer — notice the self-check in the last line. Then look at the token counter: that self-check is not free.`),
          cotToggle(),
        ),
        callout('example', 'Borrowing a bigger model’s thinking', `Once a large reasoning model exists, a smaller, cheaper one can be trained to imitate its chains of thought directly via ordinary SFT — no RL needed for the small model at all. DeepSeek released exactly this: R1's reasoning traces distilled onto much smaller open models (Qwen and Llama variants from 1.5B to 70B parameters), each one reasoning noticeably better than the same size model trained normally. It's the fastest way a capability invented at huge expense on a frontier model becomes something that runs on a laptop.`),
        callout('history', 'January 2025: the recipe gets open-sourced, and the market notices', `DeepSeek, a Chinese AI lab, published the R1 paper and released its weights openly, at a small fraction of the training cost widely assumed necessary for frontier-level reasoning. The paper's most-discussed finding was R1-Zero: a version trained with reinforcement learning directly from a base model, skipping the SFT stage entirely, that still learned long chains of thought, self-correction, and lines like "wait, let me reconsider" — behaviour nobody demonstrated, only rewarded into existence. The release triggered a genuine shock in AI markets, including a sharp, if short-lived, sell-off in AI-hardware stocks built on the assumption that only a handful of labs with enormous compute budgets could produce this class of model. Whether or not that assumption was ever fully true, R1 made post-training method, not just raw compute, look like a lever any well-resourced team could pull.`),

        section('How do we know any of this worked? Evaluation',
          p(`Every stage above needs a scoreboard, or nobody could tell whether SFT, RLHF, DPO, or a constitution actually made the model better, worse, or just different. The field leans on a shifting set of standard benchmarks, each one a proxy — same warning as reward models — for some real capability:`),
          ctx.table(['Benchmark', 'What it tests', 'The catch'], [
            ['<b>MMLU</b>', 'Multiple-choice questions across 57 school and professional subjects.', 'Near-saturated by 2024 — top models score above typical human experts, so it barely separates frontier models any more.'],
            ['<b>GSM8K</b>', 'Grade-school arithmetic word problems.', 'So thoroughly solved it has largely retired itself as a discriminator between strong models.'],
            ['<b>HumanEval</b>', 'Write a short function that passes hidden unit tests.', 'Tests small, self-contained problems — nothing like fixing a bug in a 50,000-line codebase.'],
            ['<b>SWE-bench</b>', 'Resolve a real, verified issue in a real open-source GitHub repository.', 'Much closer to actual software engineering, and correspondingly harder: scores were near zero in 2023 and are a headline number by 2025.'],
            ['<b>GPQA</b>', "PhD-level science questions written so that a web search doesn't reliably give you the answer.", 'Designed specifically to resist the shortcut of memorisation or lookup.'],
            ['<b>ARC-AGI</b>', 'Small visual grid puzzles, novel each time; easy for a person, historically brutal for models.', "Built explicitly to resist memorisation and reward genuine generalisation — the benchmark most directly aimed at 'is this AGI yet?'"],
          ]),
          p(`Two problems dog every benchmark on this list. The first is <em>contamination</em>: if a benchmark's questions and answers were sitting anywhere on the open web, they were almost certainly inside chapter 10's trillions of pretraining tokens, and a model that has memorised the answer key is not demonstrating the ability the test claims to measure. Labs fight this with held-out private test sets, canary strings that let them detect leakage, and benchmarks — like ARC-AGI — designed to generate fresh puzzles instead of reusing fixed ones. The second is Goodhart's law again: once a leaderboard number becomes the target for a launch headline or a funding round, some amount of effort quietly starts optimising for that specific number rather than the underlying skill.`),
          p(`One benchmark sidesteps written questions entirely: <b>LMArena</b> (formerly Chatbot Arena) shows real users two anonymous models' answers to their own real question and asks which they preferred — pairwise human comparisons, aggregated into an Elo-style ranking. That is, structurally, exactly the Bradley–Terry model from stage two, run at the scale of a whole public leaderboard instead of one lab's training pipeline. Every serious lab also keeps large, unpublished internal evaluation suites, built around whatever it specifically cares about — agentic tool use, refusal accuracy, a competitor's known weak spot — because the moment a benchmark goes public, optimising for it and improving the real thing quietly start to diverge.`),
        ),

        section('Safety training: refusals, jailbreaks, and the cost of being careful',
          p(`A model that will cheerfully explain how to make a weapon because a user asked nicely isn't "more helpful" — it's a liability, and one of RLHF's explicit jobs, alongside sounding good, is training the model to decline specific categories of request. This uses exactly the same machinery as everything else in this chapter: SFT demonstrations of a good refusal, comparison data where a firm-but-explained decline beats both blunt compliance and a cold non-answer, sometimes RLAIF against a written policy — just aimed at harm instead of helpfulness.`),
          p(`Two forces push back against it immediately. <em>Red-teaming</em>: dedicated people, and increasingly other models, whose job is to find prompts that produce behaviour the lab does not want, so those exact failures can become new training data before a real user finds them. And <em>jailbreaks</em>, the adversarial flip side: prompts crafted by outsiders to talk a deployed model past its refusal training — role-play framings, fictional wrapping, splitting a harmful request across many turns, or encoding it so a naive filter doesn't recognise it. Every jailbreak that succeeds in public is, in effect, free red-teaming data, which is part of why deployed safety training keeps shifting.`),
          p(`Push refusal training too hard, though, and you get <em>over-refusal</em>: a model that declines to explain how photosynthesis converts light into chemical energy because the question contains "convert" near "energy", or refuses a novelist's request for a villain's threatening dialogue. This is not hypothetical; early safety-tuned models were widely and fairly mocked for exactly this pattern, and "reduce needless refusals without reintroducing real harms" is now its own optimisation target, evaluated with dedicated benchmarks of benign-but-scary-sounding prompts.`),
        ),
        callout('example', 'The document behind the refusals', `OpenAI's public Model Spec and Anthropic's published account of Claude's character and constitution both do for safety training what a written constitution does for harmlessness in general: a specific, inspectable statement of what the model should and shouldn't do, and in what priority order, so "why did it refuse that?" or "why didn't it?" has a citable document behind it, rather than an opaque preference absorbed from labelling data nobody outside the lab ever sees.`),

        section('Why this matters for modern AI',
          p(`Strip away the acronyms and every stage in this chapter did the same three things: define a number that goes down when the model does better at some job (a demonstration's cross-entropy, a comparison's Bradley–Terry loss, a verified answer's 0-or-1), compute its gradient with respect to every weight, and step. That is chapter 2's recipe, unchanged, run five more times with different labels and, in DPO's and RLVR's case, no separate reward model at all. Nothing about the network's architecture changes between a base model and Claude. What changes is what "doing well" has been defined to mean, over and over, on data that gets closer to "genuinely helpful to a human" at each pass.`),
          callout('key', 'The whole chapter in one line', `An assistant is not a different kind of machine from a base model. It is the same next-token predictor, with its probability distribution reshaped — by demonstrations, then comparisons, then verified answers, then a written policy — until "the most likely next token" and "the most helpful next token" are, most of the time, the same token.`),
          p(`That framing also explains this chapter's central tension, which does not go away: every stage optimises a proxy — a labeller's click, a reward model's score, a benchmark's number — for a target that's genuinely hard to write down in full: "be helpful, honest, and harmless, in a way people who thought hard about it would actually endorse." RLHF pushed too far hacks the proxy into sycophancy and padding; benchmarks get contaminated or gamed; refusal training overshoots into declining homework help. Every fix in this chapter — KL penalties, DPO, constitutions, verifiable rewards, model specs — tightens the proxy. None of them eliminates the gap. That gap, and what it will take to close it, is most of what the rest of this course is about.`),
        ),

        ctx.quiz([
          { q: 'A base model and a chat-tuned assistant are compared side by side. What is actually different between them, underneath?', options: ['A different transformer architecture, with new layers added for chat', 'The exact same architecture, with weights nudged further by additional rounds of gradient descent on different labels', 'The assistant model has memorised more of the internet', 'The assistant model runs on more powerful hardware'], answer: 1, explain: 'Post-training changes what the loss rewards, not the shape of the network. Same forward pass, same backprop, different data and labels — chapter 2’s recipe, run again.' },
          { q: 'In the Bradley–Terry model, if two responses have exactly equal reward scores, what does p(A ≻ B) come out to?', options: ['0', '0.5', '1', 'It is undefined'], answer: 1, explain: 'σ(0) = 1/(1+e⁰) = 0.5 — a coin flip, exactly as it should be when the reward model sees no quality difference between the two.' },
          { q: 'Why does PPO include a KL penalty against the original SFT model, instead of just maximising the reward model’s score?', options: ['To make the reward model train faster', 'Without it, the policy can drift toward whatever the reward model happens to score highly for the wrong reasons — length, tone, agreement — rather than genuine quality', 'To reduce the number of parameters that get updated', 'PPO cannot run at all without a reference model'], answer: 1, explain: 'A reward model is a proxy trained on limited comparisons; pushed too far without a leash, the policy learns to exploit whatever the proxy gets wrong. That is reward hacking, and the KL term is the leash.' },
          { q: 'What does DPO remove from the classic RLHF pipeline?', options: ['The need for any human preference data at all', 'The separate reward model and the PPO reinforcement-learning loop, by folding the same Bradley–Terry objective directly into a loss on the policy', 'The SFT stage', 'The chat template'], answer: 1, explain: 'DPO still needs comparison data — a chosen and a rejected response. What it removes is training a standalone reward model and running PPO to chase it; the same preference objective becomes an ordinary supervised loss on the policy.' },
          { q: 'A reasoning model is trained with reinforcement learning with verifiable rewards (RLVR) on maths problems. What makes the reward "verifiable"?', options: ['A human reads and scores every response from 1 to 10', 'A second model is asked whether it liked the reasoning', 'The final answer can be checked automatically against a known correct answer, or code against unit tests — no human judgement or learned reward model required', 'The response is checked against a public leaderboard'], answer: 2, explain: 'That is exactly what makes RLVR cheap to scale: maths answers and unit tests can be graded by a script, unlike "which response do you prefer", which still needs a human or a trained proxy.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://arxiv.org/abs/2203.02155" target="_blank" rel="noopener">Ouyang et al. (2022), "Training language models to follow instructions with human feedback"</a>: the InstructGPT paper that made RLHF the industry default.`,
            `<a href="https://arxiv.org/abs/2305.18290" target="_blank" rel="noopener">Rafailov et al. (2023), "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"</a>: the DPO paper, and the derivation this chapter's KL section leads into.`,
            `<a href="https://arxiv.org/abs/2212.08073" target="_blank" rel="noopener">Bai et al. (2022), "Constitutional AI: Harmlessness from AI Feedback"</a>: critique-and-revise and RLAIF, from Anthropic.`,
            `<a href="https://arxiv.org/abs/2501.12948" target="_blank" rel="noopener">DeepSeek-AI (2025), "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"</a>: RLVR, GRPO, and R1-Zero's emergent chains of thought.`,
            `<a href="https://lmarena.ai" target="_blank" rel="noopener">LMArena</a>: live, crowd-sourced pairwise model comparisons — the Bradley–Terry idea running as a public leaderboard you can vote on yourself.`,
          ]),
        ),
      );
    },
  });
})();
