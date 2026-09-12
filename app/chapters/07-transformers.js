/* Zero → AGI · Chapter 07 · Attention is all you need: the transformer
   DESIGN RULE: the reader watches the letters of "strawberry" disappear in the first ten seconds
   and sees exactly what a transformer receives. Every paragraph explains something they did.
   Interactives, in order: the strawberry/tokenization lab; live BPE tokenizer training;
   attention computed by hand on 2-D vectors; a rule-based attention visualiser with a causal
   mask toggle; positional encoding (order-blindness on/off, plus the sine-cosine pattern);
   the block diagram with a pulse tracing the residual stream; a parameter calculator that lands
   on GPT-2's 124M and GPT-3's 175B; and a KV-cache per-step work chart. */
(function () {
  /* ================================================================== */
  /* A tiny real byte-pair-encoding tokenizer, trained once at load time  */
  /* ================================================================== */
  const CORPUS_TEXT = `A language model reads text one token at a time and tries to predict the next
    token in the sequence. Early language models were built from simple word counts: given the
    previous words, count how often each next word appeared in a large collection of text, and
    predict whichever word was most common. This approach worked for short patterns but broke down
    quickly, because language has long range structure that a simple table of counts cannot capture.
    Neural language models replaced the counting table with a network of numbers that learns
    patterns from data instead of memorising them. The network reads a sequence of tokens, turns
    each token into a vector of numbers called an embedding, and passes those vectors through many
    layers of computation. Each layer transforms the vectors a little, mixing information between
    tokens and inside each token, until the final layer produces a prediction for the next token.
    Training such a network means showing it enormous amounts of text and adjusting its numbers,
    called parameters, so that its predictions get closer and closer to the actual next word every
    single time. A model with more parameters and more training text usually makes better
    predictions, which is why modern language models are trained on trillions of tokens using
    billions or even hundreds of billions of parameters. The tokens themselves are not always whole
    words. Modern tokenizers break rare or long words into smaller pieces called subword tokens, so
    that the model never encounters a word it has absolutely no representation for. Common words
    often remain a single token, while unusual words are split into several smaller tokens that the
    model has already seen before in other contexts. This tokenization step happens before the
    network ever sees the text, turning raw characters into a sequence of numbers the network can
    process. Once a model can predict the next token well, it can generate new text by repeatedly
    predicting one token, adding it to the sequence, and predicting again, one token after another,
    which is exactly how a language model writes a sentence, a paragraph, or an entire essay.
    Researchers keep training larger models on more text because, so far, larger models trained on
    more data reliably produce better predictions, better reasoning, and more useful behaviour across
    an enormous range of tasks, from answering questions to writing computer programs to translating
    between languages. Understanding how a token becomes a prediction, layer by layer, is the key to
    understanding how every modern language model actually works underneath its friendly
    conversational surface.`;

  function trainBPE(text, maxMerges) {
    const words = (text.toLowerCase().match(/[a-z]+/g) || []);
    const freq = new Map();
    for (const w of words) freq.set(w, (freq.get(w) || 0) + 1);
    const vocab = [];
    for (const [w, c] of freq) vocab.push({ symbols: w.split('').concat(['</w>']), count: c });
    const merges = [];
    for (let m = 0; m < maxMerges; m++) {
      const pairCounts = new Map();
      for (const entry of vocab) {
        const s = entry.symbols;
        for (let i = 0; i < s.length - 1; i++) {
          const key = s[i] + '' + s[i + 1];
          pairCounts.set(key, (pairCounts.get(key) || 0) + entry.count);
        }
      }
      let bestKey = null, bestCount = 0;
      for (const [k, c] of pairCounts) if (c > bestCount) { bestCount = c; bestKey = k; }
      if (!bestKey || bestCount < 2) break;
      const [a, b] = bestKey.split('');
      merges.push(a + '' + b);
      const merged = a + b;
      for (const entry of vocab) {
        const s = entry.symbols, next = [];
        let i = 0;
        while (i < s.length) {
          if (i < s.length - 1 && s[i] === a && s[i + 1] === b) { next.push(merged); i += 2; }
          else { next.push(s[i]); i++; }
        }
        entry.symbols = next;
      }
    }
    return merges;
  }
  const BPE_MERGES = trainBPE(CORPUS_TEXT, 200);
  /* the alphabet the merges start from: every character in the corpus, plus the end-of-word mark */
  const BPE_BASE_VOCAB = new Set((CORPUS_TEXT.toLowerCase().match(/[a-z]/g) || [])).size + 1;
  const BPE_RANK = new Map(BPE_MERGES.map((k, i) => [k, i]));
  const BPE_WORD_COUNT = (CORPUS_TEXT.toLowerCase().match(/[a-z]+/g) || []).length;

  /* limit = how many of the learned merge rules the reader has switched on; the tokenizer is
     the same algorithm either way, it just stops applying rules it has not reached yet. */
  function bpeEncodeWord(word, limit) {
    const cap = limit === undefined ? BPE_MERGES.length : limit;
    let symbols = word.split('').concat(['</w>']);
    if (symbols.length <= 2) return symbols;
    // eslint-disable-next-line no-constant-condition
    while (true) {
      let bestRank = Infinity, bestIdx = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const r = BPE_RANK.get(symbols[i] + '' + symbols[i + 1]);
        if (r !== undefined && r < cap && r < bestRank) { bestRank = r; bestIdx = i; }
      }
      if (bestIdx === -1) break;
      symbols = symbols.slice(0, bestIdx).concat([symbols[bestIdx] + symbols[bestIdx + 1]], symbols.slice(bestIdx + 2));
    }
    return symbols;
  }
  function bpeTokenizeText(text, limit) {
    const out = [];
    const re = /([a-zA-Z]+)|([0-9]+)|(\s+)|([^\sa-zA-Z0-9]+)/g;
    let m;
    while ((m = re.exec(text))) {
      if (m[1]) {
        const syms = bpeEncodeWord(m[1].toLowerCase(), limit);
        for (const s of syms) out.push(s.replace('</w>', '‿'));
      } else if (m[2]) {
        out.push(m[2]);
      } else if (m[3]) {
        out.push('·');
      } else if (m[4]) {
        for (const ch of m[4]) out.push(ch);
      }
    }
    return out;
  }

  /* ================================================================== */

  ZTA.registerChapter({
    id: '07-transformers',
    num: 7,
    part: 'II',
    title: 'Attention is all you need: the transformer',
    tagline: 'Watch the letters vanish, compute attention by hand, then count a real model\'s parameters on the back of an envelope — the architecture behind every model you can name.',
    render(root, ctx) {
      const { h, p, section, sub, callout, ul, ol } = ctx;
      const C = ctx.colors;
      const FONT = '13px Inter, system-ui, sans-serif';
      const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
      const f2 = (v) => (Math.abs(v) < 1e-9 ? '0.00' : v.toFixed(2));
      const f1 = (v) => (Math.abs(v) < 1e-9 ? '0.0' : v.toFixed(1));
      const pct = (v) => Math.round(v * 100) + '%';
      const softmax = (arr) => {
        let m = -Infinity;
        for (const v of arr) if (isFinite(v) && v > m) m = v;
        if (!isFinite(m)) m = 0;
        const ex = arr.map((v) => (isFinite(v) ? Math.exp(ctx.clamp(v - m, -40, 40)) : 0));
        const s = ex.reduce((a, b) => a + b, 0) || 1;
        return ex.map((v) => v / s);
      };
      function wrapText(g, text, x, y, maxW, lh) {
        const words = String(text).split(' ');
        let line = '';
        for (const w of words) {
          const t = line ? line + ' ' + w : w;
          if (g.measureText(t).width > maxW && line) { g.fillText(line, x, y); y += lh; line = w; }
          else line = t;
        }
        if (line) g.fillText(line, x, y);
        return y;
      }

      /* ================================================================== */
      /* Interactive (c): the tokenizer demo                                  */
      /* ================================================================== */
      function tokenizerDemo() {
        const box = h('div', {
          class: 'token-box',
          style: { minHeight: '64px', padding: '12px', background: '#0f1520', border: '1px solid ' + C.line, borderRadius: '10px', display: 'flex', flexWrap: 'wrap', gap: '5px', alignItems: 'center', lineHeight: '1.9' },
        });
        const mergeLine = h('div', {
          style: { marginTop: '10px', fontFamily: '"JetBrains Mono", ui-monospace, monospace', fontSize: '12px', color: C.muted },
        });
        const wrap = h('div', {}, box, mergeLine);
        const PALETTE = [C.accent, C.green, C.warn, C.danger, C.pink, C.purple, C.orange];
        const ro = ctx.readout();
        const S = { text: "Let's count: how many r's are in strawberry?", merges: BPE_MERGES.length };
        const showSym = (sym) => sym.replace('</w>', '\u203f');
        function render() {
          box.innerHTML = '';
          const clipped = S.text.length > 500;
          const text = S.text.slice(0, 500);
          const toks = text.length ? bpeTokenizeText(text, S.merges) : [];
          let ci = 0;
          for (const t of toks) {
            const isSpace = t === '·';
            const col = PALETTE[ci % PALETTE.length];
            const span = h('span', {
              style: {
                display: 'inline-block', background: isSpace ? 'transparent' : col + '2b',
                border: isSpace ? 'none' : '1px solid ' + col, color: isSpace ? C.muted : C.text,
                padding: isSpace ? '2px 3px' : '2px 7px', borderRadius: '6px',
                fontFamily: '"JetBrains Mono", ui-monospace, monospace', fontSize: '13px', whiteSpace: 'pre',
              },
            }, isSpace ? '␣' : t);
            box.append(span);
            if (!isSpace) ci++;
          }
          if (!toks.length) box.append(h('span', { style: { color: C.muted } }, 'Type something above.'));
          if (clipped) box.append(h('span', { style: { color: C.warn, fontSize: '12px' } }, ' (showing first 500 characters)'));
          const nChars = text.length;
          const rate = 0.000003; // illustrative $3 / million input tokens — not any specific real price
          if (S.merges === 0) {
            mergeLine.textContent = 'merge 0 of ' + BPE_MERGES.length + ' — nothing fused yet, so every token is a single character.';
          } else {
            const pair = BPE_MERGES[S.merges - 1].split('');
            mergeLine.textContent = 'merge ' + S.merges + ' of ' + BPE_MERGES.length + ' — newest rule: "'
              + showSym(pair[0]) + '" + "' + showSym(pair[1]) + '"  \u2192  "' + showSym(pair[0] + pair[1]) + '"';
          }
          ro.set({
            tokens: toks.length,
            characters: nChars,
            'chars / token': toks.length ? (nChars / toks.length).toFixed(2) : '–',
            'vocabulary': BPE_BASE_VOCAB + S.merges,
            'illustrative cost @ $3/M tok': '$' + (toks.length * rate).toFixed(6),
          });
        }
        const ta = ctx.textarea({ label: 'Type or paste text', value: S.text, onChange: (v) => { S.text = v; render(); } });
        const mSl = ctx.slider({
          label: 'merge rules learned', min: 0, max: BPE_MERGES.length, step: 1, value: BPE_MERGES.length,
          fmt: (v) => String(Math.round(v)), onChange: (v) => { S.merges = Math.round(v); render(); },
        });
        const b1 = ctx.button('Preset: “strawberry”', () => { S.text = 'strawberry'; ta.value = S.text; render(); });
        const b2 = ctx.button('Preset: a long rare word', () => { S.text = 'pneumonoultramicroscopicsilicovolcanoconiosis'; ta.value = S.text; render(); });
        const b3 = ctx.button('Preset: a normal sentence', () => { S.text = 'The transformer changed how machines read language forever.'; ta.value = S.text; render(); });
        render();
        return ctx.figure(wrap,
          `This is a real byte-pair-encoding (BPE) tokenizer, trained right now, in your browser, on a ${BPE_WORD_COUNT}-word built-in corpus about language models (not the internet) — it learned ${BPE_MERGES.length} merge rules by repeatedly fusing the most frequent adjacent pair of symbols. Each coloured chip is one token; “${'‿'}” marks the end of a word, the way real tokenizers mark word boundaries. Drag <b>merge rules learned</b> back to 0 and every token collapses to a single character; walk it forward and watch letter pairs, then endings, then whole words appear as single chips while the token count of the same sentence falls. Common words from its training text often survive as one piece; rare or unfamiliar ones fragment into smaller chunks. A production tokenizer (GPT-4's, Claude's) is the same algorithm trained on hundreds of billions of characters, so it recognises far more whole words — but any invented or rare-enough string still gets chopped up exactly like this.`,
          [ta, mSl, b1, b2, b3], ro);
      }

      /* ================================================================== */
      /* Interactive (b): compute attention by hand                          */
      /* ================================================================== */
      function attentionByHand() {
        const W = 720, H = 400;
        const [cv, g] = ctx.canvas(W, H);
        const S = {
          Q: [[1, 0], [0, 1], [1, 1]],
          K: [[1, 0], [0, 2], [1, -1]],
          V: [[1, 0], [0, 1], [2, 0]],
          qi: 0,
        };
        function compute() {
          const n = 3, d = 2, scale = Math.sqrt(d);
          const raw = [0, 0, 0], scaled = [0, 0, 0];
          for (let j = 0; j < n; j++) {
            raw[j] = S.Q[S.qi][0] * S.K[j][0] + S.Q[S.qi][1] * S.K[j][1];
            scaled[j] = raw[j] / scale;
          }
          const w = softmax(scaled);
          const out = [0, 0];
          for (let j = 0; j < n; j++) { out[0] += w[j] * S.V[j][0]; out[1] += w[j] * S.V[j][1]; }
          return { raw, scaled, w, out };
        }
        // left plot: shared 2-D plane for every Q, K, V arrow
        const PX = 20, PY = 40, PS = 260, ORG = { x: PX + PS / 2, y: PY + PS / 2 }, SC = PS / 6.4;
        function toPx(v) { return { x: ORG.x + v[0] * SC, y: ORG.y - v[1] * SC }; }
        function arrow(gc, to, color, dash, label, labelOff) {
          const p0 = ORG, p1 = toPx(to);
          gc.save(); gc.strokeStyle = color; gc.fillStyle = color; gc.lineWidth = 2; gc.setLineDash(dash || []);
          gc.beginPath(); gc.moveTo(p0.x, p0.y); gc.lineTo(p1.x, p1.y); gc.stroke();
          gc.setLineDash([]);
          const ang = Math.atan2(p1.y - p0.y, p1.x - p0.x);
          gc.beginPath(); gc.moveTo(p1.x, p1.y);
          gc.lineTo(p1.x - 8 * Math.cos(ang - 0.3), p1.y - 8 * Math.sin(ang - 0.3));
          gc.lineTo(p1.x - 8 * Math.cos(ang + 0.3), p1.y - 8 * Math.sin(ang + 0.3));
          gc.closePath(); gc.fill();
          gc.restore();
          return { x: p1.x + 6, y: p1.y + 4 + (labelOff || 0), text: label, color: color };
        }
        /* maxAbs is passed in, not guessed: the raw and the scaled panel share one scale so
           that ÷√2 visibly shortens every bar, and the scale grows with the data so a slider
           at its extreme cannot draw a bar taller than its frame. */
        function bars(gc, x0, y0, w2, h2, vals, labels, mode, maxAbs) {
          const bw = w2 / vals.length;
          const top = mode === 'unit' ? 1 : Math.max(1e-6, maxAbs);
          const zero = mode === 'unit' ? y0 + h2 - 2 : y0 + h2 / 2;
          const span = mode === 'unit' ? h2 - 8 : h2 / 2 - 4;
          gc.save();
          gc.strokeStyle = C.line; gc.strokeRect(x0, y0, w2, h2);
          if (mode !== 'unit') { gc.strokeStyle = C.muted; gc.globalAlpha = 0.5; gc.beginPath(); gc.moveTo(x0, zero); gc.lineTo(x0 + w2, zero); gc.stroke(); gc.globalAlpha = 1; }
          gc.beginPath(); gc.rect(x0, y0, w2, h2); gc.clip();
          vals.forEach((v, i) => {
            const cx = x0 + bw * (i + 0.5);
            const bh = ctx.clamp(Math.abs(v) / top, 0, 1) * span;
            const barY = v >= 0 ? zero - bh : zero;
            gc.fillStyle = v >= 0 ? C.accent : C.danger;
            gc.fillRect(cx - bw * 0.28, barY, bw * 0.56, Math.max(1, bh));
            /* the number rides inside a tall bar and sits just outside a short one, so it
               is always inside the frame and never lands on the key name below it */
            const inside = bh >= 18;
            gc.fillStyle = inside ? '#0a0e16' : C.text; gc.font = MONO; gc.textAlign = 'center';
            gc.fillText(v.toFixed(2), cx, v >= 0
              ? (inside ? barY + 14 : barY - 4)
              : (inside ? barY + bh - 5 : barY + bh + 12));
          });
          gc.restore();
          gc.save();
          gc.fillStyle = C.muted; gc.font = MONO; gc.textAlign = 'center';
          vals.forEach((v, i) => gc.fillText(labels[i], x0 + bw * (i + 0.5), y0 + h2 + 14));
          gc.restore();
        }
        function draw() {
          g.clearRect(0, 0, W, H);
          g.fillStyle = C.text; g.font = 'bold 13px Inter, system-ui, sans-serif'; g.textAlign = 'left';
          g.fillText('Q, K, V for all 3 tokens (shared plane)', PX, 20);
          g.strokeStyle = C.line; g.strokeRect(PX, PY, PS, PS);
          g.strokeStyle = C.line; g.globalAlpha = 0.4;
          g.beginPath(); g.moveTo(PX, ORG.y); g.lineTo(PX + PS, ORG.y); g.moveTo(ORG.x, PY); g.lineTo(ORG.x, PY + PS); g.stroke();
          g.globalAlpha = 1;
          const r = compute();
          const tips = [];
          g.save(); g.beginPath(); g.rect(PX, PY, PS, PS); g.clip();
          for (let t = 0; t < 3; t++) {
            tips.push(arrow(g, S.Q[t], C.accent, null, 'Q' + (t + 1), -8));
            tips.push(arrow(g, S.K[t], C.warn, [5, 3], 'K' + (t + 1), 4));
            tips.push(arrow(g, S.V[t], C.green, [1, 4], 'V' + (t + 1), 16));
          }
          tips.push(arrow(g, r.out, C.pink, null, 'output', 28));
          g.restore();
          /* names go on last, each nudged down until it sits clear of the ones already
             placed: three coincident vectors must still show three readable names */
          const placed = [];
          g.textAlign = 'left';
          tips.forEach((tp, idx) => {
            g.font = (idx === tips.length - 1 ? 'bold ' : '') + MONO;
            const tw = g.measureText(tp.text).width;
            const x = ctx.clamp(tp.x, PX + 3, PX + PS - tw - 3);
            let y = ctx.clamp(tp.y, PY + 12, PY + PS - 5);
            for (let guard = 0; guard < 24; guard++) {
              const clash = placed.some((q) => Math.abs(q.y - y) < 12 && x < q.x + q.w + 4 && q.x < x + tw + 4);
              if (!clash) break;
              y += 12;
              if (y > PY + PS - 5) y = PY + 12;
            }
            placed.push({ x, y, w: tw });
            g.fillStyle = tp.color; g.fillText(tp.text, x, y);
          });
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          wrapText(g, 'solid = Q · dashed = K · dotted = V · thick pink = the attention output for the selected query', PX, PY + PS + 18, PS, 13);

          const RX = 320, RW = W - RX - 16;
          g.fillStyle = C.text; g.font = 'bold 13px Inter, system-ui, sans-serif'; g.textAlign = 'left';
          g.fillText('query = token ' + (S.qi + 1) + '  —  scores against every key', RX, 20);
          const panel = (title, y0, vals, mode, maxAbs) => {
            g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
            g.fillText(title, RX, y0 - 8);
            bars(g, RX, y0, RW, 62, vals, ['K1', 'K2', 'K3'], mode, maxAbs);
          };
          const rawMax = Math.max(1, ...r.raw.map((v) => Math.abs(v)));
          panel('raw  QKᵀ', 48, r.raw, null, rawMax);
          panel('scaled  ÷ √2  (same scale as above)', 154, r.scaled, null, rawMax);
          panel('softmax weights (sum to 1)', 260, r.w, 'unit', 1);
          g.fillStyle = C.text; g.font = MONO; g.textAlign = 'left';
          wrapText(g, 'output = ' + r.w.map((w, i) => f2(w) + '·V' + (i + 1)).join(' + ') + ' = (' + f2(r.out[0]) + ', ' + f2(r.out[1]) + ')', RX, 362, RW, 15);
        }
        const ro = ctx.readout();
        function refresh() { draw(); const r = compute(); ro.set({ 'top key': 'K' + (r.w.indexOf(Math.max(...r.w)) + 1), weight: pct(Math.max(...r.w)) }); }
        const qSel = ctx.select({ label: 'inspect query', options: [{ value: '0', label: 'token 1' }, { value: '1', label: 'token 2' }, { value: '2', label: 'token 3' }], value: '0', onChange: (v) => { S.qi = +v; refresh(); } });
        const sliders = [];
        ['Q', 'K', 'V'].forEach((kind) => {
          for (let t = 0; t < 3; t++) {
            for (let dim = 0; dim < 2; dim++) {
              const lab = kind + (t + 1) + (dim === 0 ? '.x' : '.y');
              sliders.push(ctx.slider({
                label: lab, min: -3, max: 3, step: 0.1, value: S[kind][t][dim], fmt: f1,
                onChange: (v) => { S[kind][t][dim] = v; refresh(); },
              }));
            }
          }
        });
        refresh();
        return ctx.figure(cv,
          'Three tokens, each with its own 2-D Query, Key and Value vector — drag any slider and every number recomputes live. Raw score = Q·K (dot product); scale by ÷√d (d = 2 here) to stop scores from growing with dimension; softmax turns the three scores into weights that are positive and sum to 1; the output is that weighted average of the three Value vectors. The default numbers match the worked example in the text below — check a slider against it.',
          [qSel, ...sliders], ro);
      }

      /* ================================================================== */
      /* Interactive (a): the attention visualiser (centrepiece)             */
      /* ================================================================== */
      function attentionVisualiser() {
        const DEFAULT_SENTENCE = "The animal didn't cross the street because it was too tired";
        const PRON = new Set(['it', 'he', 'she', 'they', 'him', 'her', 'them', 'himself', 'herself', 'itself', 'themselves', 'i', 'we', 'you', 'who']);
        const DET = new Set(['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'their', 'our', 'its']);
        const CONJ = new Set(['because', 'although', 'though', 'but', 'and', 'or', 'since', 'if', 'when', 'while', 'so', 'yet', 'unless', 'after', 'before']);
        const AUX = new Set(["didn't", 'did', 'was', 'were', 'is', 'are', 'am', 'be', 'been', 'being', 'does', "doesn't", 'do', "don't", 'will', "won't", 'would', "wouldn't", 'could', "couldn't", 'should', "shouldn't", 'has', 'have', 'had', 'can', "can't", 'shall', 'must']);
        function classify(w) {
          if (PRON.has(w)) return 'pron';
          if (DET.has(w)) return 'det';
          if (CONJ.has(w)) return 'conj';
          if (AUX.has(w)) return 'verb';
          if (w.length <= 3) return 'func';
          return 'content';
        }
        const TAGCOL = { pron: C.pink, det: C.muted, conj: C.purple, verb: C.orange, func: C.muted, content: C.accent };
        const TAGNAME = { pron: 'pronoun', det: 'determiner', conj: 'conjunction', verb: 'verb/aux', func: 'function word', content: 'content word' };

        function analyse(text, causal, temp) {
          const raw = text.trim().length ? text.trim().split(/\s+/).slice(0, 16) : [];
          const n = raw.length;
          const clean = raw.map((t) => t.toLowerCase().replace(/[^a-z']/g, ''));
          const tags = clean.map(classify);
          let rankN = 0;
          const rank = tags.map((t) => (t === 'content' ? rankN++ : null));
          function rawScore(i, j) {
            let s = 0;
            if (tags[i] === 'pron' && tags[j] === 'content') s += 1.0 + Math.max(0, 2.2 - rank[j] * 1.0);
            if (tags[i] === 'det' && j === i + 1) s += 1.5;
            if (tags[i] === 'conj') s += tags[j] === 'content' ? 0.5 : 0.1;
            if (i === j) s += 0.3;
            s -= 0.08 * Math.abs(i - j);
            return s;
          }
          const weights = [];
          for (let i = 0; i < n; i++) {
            const logits = [];
            for (let j = 0; j < n; j++) logits.push(causal && j > i ? -Infinity : rawScore(i, j) / Math.max(0.05, temp));
            weights.push(softmax(logits));
          }
          return { tokens: raw, tags, weights, truncated: text.trim().split(/\s+/).length > 16 };
        }

        const W = 720, H = 560;
        const [cv, g] = ctx.canvas(W, H);
        const S = { text: DEFAULT_SENTENCE, causal: false, temp: 1, query: 0 };
        let chipBoxes = [], matrixGeo = null;
        function setup() {
          const a = analyse(S.text, S.causal, S.temp);
          const guess = a.tags.findIndex((t) => t === 'pron');
          S.query = ctx.clamp(guess >= 0 ? guess : Math.floor(a.tokens.length / 2), 0, Math.max(0, a.tokens.length - 1));
        }
        setup();

        function draw() {
          g.clearRect(0, 0, W, H);
          const a = analyse(S.text, S.causal, S.temp);
          const n = a.tokens.length;
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          g.fillText('Toy rule — hand-crafted to illustrate the mechanism, not a trained model\'s real weights.', 14, 16);
          if (!n) { g.fillStyle = C.muted; g.textAlign = 'center'; g.fillText('Type a sentence above.', W / 2, H / 2); chipBoxes = []; matrixGeo = null; return; }
          S.query = ctx.clamp(S.query, 0, n - 1);

          // layout token chips
          g.font = MONO;
          const gap = 8, rowY = 168, chipH = 26;
          let widths = a.tokens.map((t) => Math.max(20, g.measureText(t).width + 14));
          let total = widths.reduce((x, y) => x + y, 0) + gap * (n - 1);
          let scale = total > W - 32 ? (W - 32) / total : 1;
          let x = (W - Math.min(total, W - 32)) / 2;
          chipBoxes = [];
          for (let i = 0; i < n; i++) {
            const w = widths[i] * scale;
            chipBoxes.push({ x, y: rowY, w, h: chipH, i });
            x += w + gap * scale;
          }

          // arcs from selected query to every other token
          const wq = a.weights[S.query];
          const qBox = chipBoxes[S.query];
          for (let j = 0; j < n; j++) {
            if (j === S.query) continue;
            const wt = wq[j];
            if (wt < 0.01) continue;
            const b = chipBoxes[j];
            const p0 = { x: qBox.x + qBox.w / 2, y: rowY };
            const p1 = { x: b.x + b.w / 2, y: rowY };
            const midX = (p0.x + p1.x) / 2, apexY = rowY - 18 - wt * 110;
            g.beginPath(); g.moveTo(p0.x, p0.y); g.quadraticCurveTo(midX, apexY, p1.x, p1.y);
            g.strokeStyle = C.accent; g.globalAlpha = 0.15 + 0.8 * wt; g.lineWidth = 1 + 13 * wt; g.stroke();
            g.globalAlpha = 1;
            if (wt > 0.08) { g.fillStyle = C.text; g.font = MONO; g.textAlign = 'center'; g.fillText(pct(wt), midX, apexY - 4); }
          }

          // chips
          for (let i = 0; i < n; i++) {
            const b = chipBoxes[i];
            const isQ = i === S.query;
            g.fillStyle = '#111827';
            g.fillRect(b.x, b.y - chipH / 2, b.w, chipH);
            g.lineWidth = isQ ? 2.5 : 1.4;
            g.strokeStyle = isQ ? C.warn : TAGCOL[a.tags[i]];
            g.strokeRect(b.x, b.y - chipH / 2, b.w, chipH);
            g.fillStyle = C.text; g.font = MONO; g.textAlign = 'center';
            g.fillText(a.tokens[i], b.x + b.w / 2, b.y + 4);
            g.fillStyle = C.muted; g.font = '10px "JetBrains Mono", monospace';
            g.fillText(String(i + 1), b.x + b.w / 2, b.y - chipH / 2 - 4);
          }
          if (chipBoxes[S.query]) {
            const b = chipBoxes[S.query];
            g.fillStyle = C.warn; g.font = 'bold 11px Inter, system-ui, sans-serif'; g.textAlign = 'center';
            g.fillText('QUERY', b.x + b.w / 2, b.y + chipH / 2 + 16);
          }
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          /* clear of the QUERY tag that hangs below the selected chip */
          wrapText(g, 'border colour = toy part-of-speech guess: pink = pronoun, blue = content word, orange = verb/aux, purple = conjunction, grey = determiner/function word', 14, 213, W - 28, 13);
          if (a.truncated) { g.fillStyle = C.warn; g.fillText('(showing first 16 words)', 14, 243); }

          // matrix — its own heading first, then a cell size that keeps every row on the canvas
          g.fillStyle = C.text; g.font = 'bold 12px Inter, system-ui, sans-serif'; g.textAlign = 'left';
          g.fillText('Every query (row) × every key (column) — click a row to inspect it above.', 14, 258);
          const my0 = 286;
          const n2 = n, cell = ctx.clamp(Math.min(340 / n2, (H - my0 - 12) / n2, 30), 12, 30);
          const mx0 = (W - n2 * cell) / 2;
          matrixGeo = { x0: mx0, y0: my0, cell, n: n2 };
          for (let i = 0; i < n2; i++) {
            for (let j = 0; j < n2; j++) {
              const wt = a.weights[i][j];
              const cx = mx0 + j * cell, cy = my0 + i * cell;
              g.fillStyle = 'rgba(124,156,255,' + (0.06 + 0.9 * wt) + ')';
              g.fillRect(cx + 1, cy + 1, cell - 2, cell - 2);
              if (cell >= 20 && wt > 0.06) {
                g.fillStyle = wt > 0.5 ? '#0a0e16' : C.text; g.font = (cell >= 26 ? '10px' : '9px') + ' "JetBrains Mono", monospace'; g.textAlign = 'center';
                g.fillText(Math.round(wt * 100), cx + cell / 2, cy + cell / 2 + 3);
              }
            }
            if (i === S.query) { g.strokeStyle = C.warn; g.lineWidth = 2; g.strokeRect(mx0, my0 + i * cell, n2 * cell, cell); }
          }
          g.strokeStyle = C.line; g.strokeRect(mx0, my0, n2 * cell, n2 * cell);
          g.fillStyle = C.muted; g.font = '10px "JetBrains Mono", monospace'; g.textAlign = 'center';
          for (let k = 0; k < n2; k++) { g.fillText(String(k + 1), mx0 + k * cell + cell / 2, my0 - 4); g.fillText(String(k + 1), mx0 - 10, my0 + k * cell + cell / 2 + 3); }
        }

        function hit(ev) {
          const p = cv.pos(ev);
          for (const b of chipBoxes) if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y - b.h / 2 - 12 && p.y <= b.y + b.h / 2 + 12) return b.i;
          if (matrixGeo) {
            const { x0, y0, cell, n: n3 } = matrixGeo;
            if (p.x >= x0 && p.x <= x0 + n3 * cell && p.y >= y0 && p.y <= y0 + n3 * cell) return Math.floor((p.y - y0) / cell);
          }
          return -1;
        }
        cv.addEventListener('pointerdown', (ev) => { const i = hit(ev); if (i >= 0) { S.query = i; draw(); } });

        const ta = ctx.textarea({ label: 'Sentence (tokenised by whitespace)', value: S.text, onChange: (v) => { S.text = v; setup(); draw(); } });
        const causalBtn = ctx.button('Causal mask: off', () => { S.causal = !S.causal; causalBtn.textContent = 'Causal mask: ' + (S.causal ? 'on' : 'off'); draw(); });
        const tempSl = ctx.slider({ label: 'softmax temperature', min: 0.2, max: 2.5, step: 0.05, value: 1, fmt: (v) => v.toFixed(2), onChange: (v) => { S.temp = v; draw(); } });
        const resetBtn = ctx.button('Reset to default sentence', () => { S.text = DEFAULT_SENTENCE; ta.value = S.text; setup(); draw(); });
        draw();
        return ctx.figure(cv,
          'Click any token (or any matrix row) to make it the query and watch the arcs re-aim. Arc thickness and the "%" above it are that query\'s attention weight on the target token — illustrative weights from a hand-written rule (pronoun→earlier content-noun boost, determiner→next-noun boost, a same-token bias, and a mild preference for nearby words), not from a trained network. Turn on the causal mask to see a GPT-style decoder: every row can only look left, never right.',
          [ta, causalBtn, tempSl, resetBtn]);
      }

      /* ================================================================== */
      /* Interactive (d): the transformer block, animated                    */
      /* ================================================================== */
      function blockDiagram() {
        const W = 720, H = 380;
        const [cv, g] = ctx.canvas(W, H);
        const X0 = 64, X1 = 656, LY = 250, BOXY = 106, BOXH = 58;
        const S = { layers: 6, u: 0, playing: true, speed: 1 };
        const trail = [];
        function blockGeo(k, L) {
          const bw = (X1 - X0) / L, bx = X0 + (k + 0.5) * bw;
          return { bx, bw, attnX: bx - bw * 0.2, mlpX: bx + bw * 0.2 };
        }
        function phaseAt(u, L) {
          /* u is a clock that can arrive stale, backwards or wrapped; fold it into [0, L)
             so floor(u) can never index a block that is not on screen. */
          let uu = isFinite(u) ? u % L : 0;
          if (uu < 0) uu += L;
          const k = ctx.clamp(Math.floor(uu), 0, L - 1), t = uu - k;
          const { bx, bw, attnX, mlpX } = blockGeo(k, L);
          const lp = (a, b, u2) => a + (b - a) * ctx.clamp(u2, 0, 1);
          let x, y, label = '', box = null;
          if (t < 0.10) { x = lp(bx - bw * 0.5, attnX, t / 0.10); y = LY; }
          else if (t < 0.18) { x = attnX; y = lp(LY, BOXY + BOXH, (t - 0.10) / 0.08); }
          else if (t < 0.40) { x = attnX; y = BOXY + BOXH - 12; label = 'Attention — every token gathers information from every other token, all at once'; box = 'attn'; }
          else if (t < 0.48) { x = attnX; y = lp(BOXY + BOXH, LY, (t - 0.40) / 0.08); label = 'Add & Norm — attention’s output is added back onto the residual stream, then normalised'; }
          else if (t < 0.55) { x = lp(attnX, mlpX, (t - 0.48) / 0.07); y = LY; label = 'Add & Norm'; }
          else if (t < 0.63) { x = mlpX; y = lp(LY, BOXY + BOXH, (t - 0.55) / 0.08); }
          else if (t < 0.90) { x = mlpX; y = BOXY + BOXH - 12; label = 'MLP — the same small feed-forward network applied to each token independently'; box = 'mlp'; }
          else if (t < 0.98) { x = mlpX; y = lp(BOXY + BOXH, LY, (t - 0.90) / 0.08); label = 'Add & Norm — the MLP’s output is added back onto the residual stream, then normalised'; }
          else { x = lp(mlpX, bx + bw * 0.5, (t - 0.98) / 0.02); y = LY; label = 'Add & Norm'; }
          return { k, x, y, label, box };
        }
        function draw() {
          g.clearRect(0, 0, W, H);
          const L = ctx.clamp(Math.round(S.layers), 1, 12);
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          wrapText(g, 'The residual stream is a highway running left to right. Attention and the MLP are detours: they read from the highway and add their result back onto it (Add & Norm) before the next block.', 14, 20, W - 28, 14);
          // static faint block markers
          for (let k = 0; k < L; k++) {
            const { attnX, mlpX, bx, bw } = blockGeo(k, L);
            g.strokeStyle = C.line; g.globalAlpha = 0.7; g.lineWidth = 1;
            [attnX, mlpX].forEach((bxp) => {
              g.beginPath(); g.moveTo(bxp, LY); g.bezierCurveTo(bxp, BOXY + BOXH, bxp, BOXY + BOXH, bxp, BOXY + BOXH / 2); g.stroke();
            });
            g.globalAlpha = 1;
            g.fillStyle = C.muted; g.font = '10px "JetBrains Mono", monospace'; g.textAlign = 'center';
            g.fillText(String(k + 1), bx, LY + 46);
          }
          // residual line
          g.strokeStyle = C.accent; g.lineWidth = 3; g.beginPath(); g.moveTo(X0 - 30, LY); g.lineTo(X1 + 20, LY); g.stroke();
          g.fillStyle = C.accent; g.beginPath(); g.moveTo(X1 + 20, LY); g.lineTo(X1 + 10, LY - 5); g.lineTo(X1 + 10, LY + 5); g.closePath(); g.fill();
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          g.fillText('tokens + positions in', X0 - 30, LY + 20);
          g.textAlign = 'right'; g.fillText('→ next-token probabilities', X1 + 20, LY + 20);
          g.textAlign = 'center';
          const rsW = g.measureText('residual stream').width;
          g.fillStyle = '#0a0e16'; g.fillRect((X0 + X1) / 2 - rsW / 2 - 7, LY - 23, rsW + 14, 18);
          g.fillStyle = C.accent; g.fillText('residual stream', (X0 + X1) / 2, LY - 10);

          const ph = phaseAt(S.u, L);
          /* everything below is driven by the animation clock: keep it inside the diagram */
          g.save(); g.beginPath(); g.rect(24, BOXY - 10, W - 48, LY - BOXY + 40); g.clip();
          if (ph.box) {
            const geo = blockGeo(ph.k, L);
            const bxp = ph.box === 'attn' ? geo.attnX : geo.mlpX;
            const col = ph.box === 'attn' ? C.pink : C.orange, BW2 = 112;
            g.fillStyle = '#111827'; g.fillRect(bxp - BW2 / 2, BOXY, BW2, BOXH);
            g.strokeStyle = col; g.lineWidth = 2; g.strokeRect(bxp - BW2 / 2, BOXY, BW2, BOXH);
            g.fillStyle = col; g.textAlign = 'center';
            if (ph.box === 'attn') {
              g.font = 'bold 12px Inter, system-ui, sans-serif';
              g.fillText('Multi-Head', bxp, BOXY + 17);
              g.fillText('Attention', bxp, BOXY + 32);
            } else {
              g.font = 'bold 13px Inter, system-ui, sans-serif';
              g.fillText('MLP', bxp, BOXY + 25);
            }
          }
          trail.push({ x: ph.x, y: ph.y });
          if (trail.length > 16) trail.shift();
          trail.forEach((p, i) => {
            const al = (i + 1) / trail.length;
            g.beginPath(); g.arc(p.x, p.y, 3 + al * 4, 0, Math.PI * 2);
            g.fillStyle = 'rgba(251,113,133,' + (al * 0.8) + ')'; g.fill();
          });
          g.restore();
          if (ph.label) {
            g.fillStyle = C.warn; g.font = FONT; g.textAlign = 'center';
            wrapText(g, ph.label, W / 2, H - 34, W - 60, 14);
          }
          const layerNum = Math.min(L, ph.k + 1);
          g.fillStyle = C.text; g.font = MONO; g.textAlign = 'left';
          g.fillText('layer ' + layerNum + ' of ' + L, 14, H - 8);
        }
        const ro = ctx.readout();
        ctx.loop((dt) => {
          const L = ctx.clamp(Math.round(S.layers), 1, 12);
          if (S.playing) { S.u += Math.max(0, dt) * S.speed * 0.85; S.u = ((S.u % L) + L) % L; }
          draw();
          ro.set({ layers: L, 'params scale with': '12·d² per layer (next section)' });
        });
        const layerSl = ctx.slider({ label: 'layers', min: 1, max: 12, step: 1, value: 6, onChange: (v) => { S.layers = Math.round(v); S.u = 0; trail.length = 0; } });
        const playBtn = ctx.button('⏸ Pause', () => { S.playing = !S.playing; playBtn.textContent = S.playing ? '⏸ Pause' : '▶ Play'; }, 'primary');
        const speedSl = ctx.slider({ label: 'speed', min: 0.25, max: 2, step: 0.25, value: 1, fmt: (v) => v.toFixed(2) + '×', onChange: (v) => { S.speed = v; } });
        return ctx.figure(cv,
          'One pulse, one full pass through the stack. Depth means the SAME two-step recipe (attention, then MLP, each followed by Add & Norm) repeats layer after layer — GPT-2 small stacks it 12 times; the largest models today stack it a few dozen times — DeepSeek-V4-Pro uses 61 layers, and Qwen 3 dense models run from 28 to 64 — because what has grown fastest at the frontier is width and expert count, not depth. Attention itself is computed for every token in parallel; only the layer-by-layer stacking is sequential.',
          [layerSl, playBtn, speedSl], ro);
      }

      /* ================================================================== */
      /* Interactive: the strawberry problem, made visible                    */
      /* ================================================================== */
      function strawberryLab() {
        const [cv, g] = ctx.canvas(720, 330);
        /* a small stand-in vocabulary of common chunks; longest match wins, like a real BPE
           tokenizer's merge table. IDs are arbitrary but fixed, exactly as in a real one. */
        const CHUNKS = ['straw', 'berry', 'berries', 'straw', 'apple', 'rasp', 'blue', 'cran',
          'the', 'ing', 'tion', 'able', 'ness', 'pine', 'water', 'melon', 'hippo', 'potam',
          'us', 'er', 'ed', 'ly', 'un', 'pre', 'con', 'ph', 'ch', 'th', 'sh'];
        const ID = new Map();
        CHUNKS.forEach((c, i) => { if (!ID.has(c)) ID.set(c, 3000 + i * 137); });
        const idOf = (s) => ID.has(s) ? ID.get(s) : (s.charCodeAt(0) * 7 + 100);
        let word = 'strawberry', letter = 'r', hideLetters = false;

        function tokenize(w) {
          const out = []; let i = 0; const s = w.toLowerCase();
          while (i < s.length) {
            let best = '';
            for (const c of CHUNKS) if (c.length > best.length && s.startsWith(c, i)) best = c;
            if (best) { out.push(best); i += best.length; } else { out.push(s[i]); i += 1; }
          }
          return out;
        }

        const wIn = ctx.textarea({ label: 'a word', value: word, onChange: (v) => { word = v.trim().split(/\s+/)[0] || ''; } });
        const lIn = ctx.textarea({ label: 'letter to count', value: letter, onChange: (v) => { letter = (v || 'r').trim().slice(0, 1).toLowerCase(); } });
        const hideBtn = ctx.button('See it the way the model does', () => {
          hideLetters = !hideLetters;
          hideBtn.textContent = hideLetters ? 'Show me the letters again' : 'See it the way the model does';
        }, 'primary');
        const presets = ['strawberry', 'hippopotamus', 'the', 'unpredictable'].map(w =>
          ctx.button(w, () => { word = w; wIn.value = w; }));
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          const w = (word || '').toLowerCase();
          const toks = w ? tokenize(w) : [];
          const trueCount = w.split('').filter(c => c === letter).length;

          /* ---- what you see ---- */
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('what you see', 34, 30);
          const CW = 30;
          w.split('').forEach((c, i) => {
            const x = 34 + i * CW;
            const hit = c === letter;
            g.fillStyle = hit ? 'rgba(56,217,169,0.25)' : '#131a27';
            g.fillRect(x, 44, CW - 3, 32);
            g.font = 'bold 16px "JetBrains Mono", ui-monospace, monospace';
            g.fillStyle = hit ? C.green : C.text;
            g.fillText(c, x + 9, 66);
          });
          g.font = MONO; g.fillStyle = C.green;
          g.fillText(w.length + ' letters, ' + trueCount + ' × "' + letter + '" — you can just count them', 34, 96);

          /* ---- what the model sees ---- */
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('what the model receives', 34, 140);
          let x = 34;
          toks.forEach((t) => {
            const wdt = Math.max(58, t.length * 13 + 18);
            g.fillStyle = 'rgba(124,156,255,0.18)';
            g.fillRect(x, 154, wdt - 4, 46);
            g.strokeStyle = C.accent; g.lineWidth = 1.5;
            g.strokeRect(x, 154, wdt - 4, 46);
            g.font = 'bold ' + MONO; g.fillStyle = C.accent;
            g.fillText(String(idOf(t)), x + 8, 172);
            g.font = MONO;
            if (hideLetters) {
              g.fillStyle = '#2a3444';
              g.fillText('?'.repeat(t.length), x + 8, 191);
            } else {
              g.fillStyle = C.muted;
              g.fillText(t, x + 8, 191);
            }
            x += wdt + 6;
          });
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText(toks.length + ' token' + (toks.length === 1 ? '' : 's') + ' — that is the entire input to layer 1', 34, 222);

          /* ---- the verdict ---- */
          const single = toks.length === 1;
          g.font = 'bold 15px Inter, system-ui, sans-serif';
          g.fillStyle = single ? C.green : C.danger;
          wrapText(g, single
            ? 'One chunk. There is nothing to unpack, so questions about this word tend to go much better.'
            : 'The model gets ' + toks.length + ' opaque numbers. To answer "how many ' + letter + '", it must have learned, statistically, how each chunk happens to be spelled — a fact about spelling, one step removed from the meaning those numbers were built to carry.',
            34, 254, 650, 18);
          ro.set({ letters: w.length, tokens: toks.length, ['true count of "' + letter + '"']: trueCount });
        });

        return ctx.figure(cv,
          'Press <b>See it the way the model does</b> and the letters disappear, which is the honest picture: the first layer of a transformer receives a short row of numbers and nothing else. "strawberry" is not common enough to earn its own symbol, so it arrives as two chunks — GPT-4\'s tokenizer does roughly this. Counting letters then requires the model to recall how each chunk is spelled, which is not what the chunk-number was designed to carry. Try "the": one token, nothing hidden, and the problem evaporates.',
          [wIn, lIn, hideBtn, ...presets], ro);
      }

      /* ================================================================== */
      /* Interactive: attention is order-blind, and how position is restored  */
      /* ================================================================== */
      function positionLab() {
        /* real browser text runs wider than the headless estimate: the wave caption
           needs two more lines of room before the verdict line below it */
        const PW_H = 452;
        const [cv, g] = ctx.canvas(720, PW_H);
        const BASE = ['the', 'dog', 'bit', 'the', 'man'];
        let swapped = false, usePE = true, probe = 0;
        const words = () => swapped ? ['the', 'man', 'bit', 'the', 'dog'] : BASE;
        const D = 16;
        /* sinusoidal positional encoding, exactly as in the 2017 paper */
        function pe(pos) {
          const v = [];
          for (let i = 0; i < D; i++) {
            const k = Math.floor(i / 2);
            const ang = pos / Math.pow(10000, (2 * k) / D);
            v.push(i % 2 === 0 ? Math.sin(ang) : Math.cos(ang));
          }
          return v;
        }
        /* a crude content vector per word type, so identical words share one */
        const CONTENT = {
          the: [0.2, -0.1, 0.4], dog: [0.9, 0.3, -0.2], bit: [-0.3, 0.8, 0.1], man: [0.5, -0.6, 0.7],
        };
        function vecOf(w, pos) {
          const c = CONTENT[w] || [0, 0, 0];
          const base = [];
          for (let i = 0; i < D; i++) base.push(c[i % 3] * (1 - (i / D) * 0.5));
          return usePE ? base.map((x, i) => x + pe(pos)[i] * 0.9) : base;
        }
        function scores() {
          const ws = words();
          const vs = ws.map((w, i) => vecOf(w, i));
          return vs.map(q => {
            const raw = vs.map(k => q.reduce((s, x, i) => s + x * k[i], 0) / Math.sqrt(D));
            return softmax(raw);
          });
        }

        const swapBtn = ctx.button('Swap dog and man', () => { swapped = !swapped; }, 'primary');
        const peBtn = ctx.button('positional encoding: on', () => {
          usePE = !usePE;
          peBtn.textContent = 'positional encoding: ' + (usePE ? 'on' : 'off');
        });
        const probeSl = ctx.slider({ label: 'inspect position', min: 0, max: 4, step: 1, value: 0, onChange: (v) => { probe = v; } });
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, 720, PW_H);
          const ws = words();
          const S = scores();

          /* ---- attention grid ---- */
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('attention scores', 34, 28);
          const GX = 110, GY = 56, CELL = 42;
          ws.forEach((w, j) => {
            g.font = MONO; g.fillStyle = C.muted;
            g.fillText(w, GX + j * CELL + 4, GY - 8);
          });
          ws.forEach((w, i) => {
            g.font = MONO; g.fillStyle = i === probe ? C.warn : C.muted;
            g.fillText(w, 34, GY + i * CELL + 26);
            ws.forEach((_, j) => {
              const v = S[i][j];
              g.fillStyle = 'rgba(124,156,255,' + (0.06 + v * 1.6) + ')';
              g.fillRect(GX + j * CELL, GY + i * CELL, CELL - 2, CELL - 2);
              g.font = '10px "JetBrains Mono", ui-monospace, monospace';
              g.fillStyle = v > 0.35 ? '#0a0e16' : C.muted;
              g.fillText(v.toFixed(2), GX + j * CELL + 6, GY + i * CELL + 25);
            });
          });
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('each row sums to 1', GX, GY + ws.length * CELL + 16);

          /* ---- the positional encoding pattern itself ---- */
          const PX = 380, PY = 56, PW = 300, PH = 150;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('the position signal being added', PX, 28);
          const NPOS = 200;
          for (let pos = 0; pos < NPOS; pos++) {
            const v = pe(pos);
            for (let i = 0; i < D; i++) {
              g.fillStyle = ctx.heat(v[i]);
              g.fillRect(PX + pos * (PW / NPOS), PY + i * (PH / D), PW / NPOS - 0.4, PH / D - 0.4);
            }
          }
          g.strokeStyle = C.warn; g.lineWidth = 2;
          g.strokeRect(PX + probe * (PW / NPOS) - 1, PY - 2, PW / NPOS + 1, PH + 4);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('position  →', PX, PY + PH + 16);
          g.save(); g.translate(PX - 8, PY + PH - 10); g.rotate(-Math.PI / 2);
          g.fillText('dimension', 0, 0); g.restore();
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, 'Sine and cosine waves at different frequencies. Fast waves at the top separate neighbouring positions; the rows further down cycle so slowly they barely move across this window, which is how they mark roughly where in a long document you are — so a single vector encodes position at every scale at once, and nothing had to be learned.',
            PX, PY + PH + 40, 320, 16);

          /* ---- the verdict on order-blindness: full width, below both columns ---- */
          g.font = 'bold ' + FONT; g.fillStyle = usePE ? C.green : C.danger; g.textAlign = 'left';
          wrapText(g, usePE
            ? 'With positional encoding, swapping two words genuinely changes the scores — the model can tell the two sentences apart.'
            : 'Without it, swapping two words only shuffles the grid. The same numbers come back in a different order, because a dot product has no idea where either token sat.',
            34, 396, 652, 17);
          ro.set({ order: swapped ? 'man bit dog' : 'dog bit man', 'positional encoding': usePE ? 'on' : 'off' });
        });

        return ctx.figure(cv,
          'Attention is fundamentally a <i>set</i> operation: Q·Kᵀ compares content against content and has no idea where either token sat. Turn positional encoding off and swap two words — the same scores return, merely rearranged, so "the dog bit the man" and "the man bit the dog" are literally indistinguishable. Turn it back on and a position-dependent vector has been added to each embedding before layer 1, so identical words in different slots no longer produce identical keys. An RNN got order free by reading left to right; a transformer has to be told.',
          [swapBtn, peBtn, probeSl], ro);
      }

      /* ================================================================== */
      /* Interactive: count a real model's parameters on the back of a napkin */
      /* ================================================================== */
      function paramCalc() {
        const [cv, g] = ctx.canvas(720, 330);
        let d = 768, L = 12, V = 50257, nctx = 1024, tied = true;
        const dSl = ctx.slider({ label: 'width d', min: 128, max: 12288, step: 64, value: 768, onChange: (v) => { d = v; } });
        const lSl = ctx.slider({ label: 'layers L', min: 1, max: 128, step: 1, value: 12, onChange: (v) => { L = v; } });
        const vSl = ctx.slider({ label: 'vocabulary', min: 8000, max: 256000, step: 1000, value: 50257, onChange: (v) => { V = v; } });
        const gpt2 = ctx.button('GPT-2 small', () => { d = 768; dSl.value = 768; L = 12; lSl.value = 12; V = 50257; vSl.value = 50257; nctx = 1024; }, 'primary');
        const gpt3 = ctx.button('GPT-3', () => { d = 12288; dSl.value = 12288; L = 96; lSl.value = 96; V = 50257; vSl.value = 50257; nctx = 2048; });
        const ro = ctx.readout();
        const human = (n) => n >= 1e9 ? (n / 1e9).toFixed(1) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : (n / 1e3).toFixed(0) + 'K';

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          const attn = 4 * d * d;          // Q, K, V and the output projection
          const mlp = 8 * d * d;           // two matrices, hidden width 4d
          const per = attn + mlp;          // 12 d^2 per block
          const blocks = per * L;
          const emb = V * d;
          const pos = nctx * d;
          const total = blocks + emb + pos + (tied ? 0 : emb);

          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('where the parameters actually go', 34, 28);
          const rows = [
            ['attention (Q, K, V, out)  4d²', attn, C.accent, ' per block'],
            ['MLP (d→4d→d)              8d²', mlp, C.purple, ' per block'],
            ['× ' + L + ' blocks           12d²·L', blocks, C.green, ''],
            ['token embeddings       V·d', emb, C.warn, ''],
            ['position embeddings  n·d', pos, C.muted, ''],
          ];
          let y = 58;
          const BX = 330, BW = 250;
          const peak = Math.max(blocks, emb, 1);
          rows.forEach(([lab, v, col, note]) => {
            g.font = MONO; g.fillStyle = C.muted;
            g.fillText(lab, 34, y + 12);
            g.fillStyle = C.line; g.fillRect(BX, y, BW, 14);
            g.fillStyle = col; g.fillRect(BX, y, ctx.clamp(v / peak, 0, 1) * BW, 14);
            g.font = 'bold ' + MONO; g.fillStyle = col;
            g.fillText(human(v) + note, BX + BW + 10, y + 12);
            y += 30;
          });
          g.strokeStyle = C.line; g.lineWidth = 1;
          g.beginPath(); g.moveTo(34, y + 4); g.lineTo(680, y + 4); g.stroke();
          g.font = 'bold 22px Inter, system-ui, sans-serif'; g.fillStyle = C.green;
          g.fillText('total ≈ ' + human(total), 34, y + 36);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('params ≈ 12·d²·L + V·d + n·d' + (tied ? '   (output layer tied to the embeddings)' : ''), 34, y + 58);
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, 'The d² term is why doubling the width roughly quadruples the model while doubling the depth only doubles it — and why width is the expensive dial.',
            34, y + 82, 640, 17);
          ro.set({ d, L, vocab: V.toLocaleString(), total: human(total) });
        });

        return ctx.figure(cv,
          '"124 million parameters" is not a fact you have to look up — it is arithmetic. Each block spends 4d² on attention (the Q, K, V and output matrices) and 8d² on its MLP, which is 12d² per block; multiply by the number of blocks and add the embedding tables. Press <b>GPT-2 small</b> and the total lands on 124M, the published figure. Press <b>GPT-3</b> and it lands near 175B. Every model-size headline you will ever read is this formula with different numbers in it.',
          [dSl, lSl, vSl, gpt2, gpt3], ro);
      }

      /* ================================================================== */
      /* Interactive: why replies stream, and why long chats get slower       */
      /* ================================================================== */
      function kvCacheLab() {
        const KH = 380;
        const [cv, g] = ctx.canvas(720, KH);
        let prompt = 200, gen = 300, cached = true;
        const pSl = ctx.slider({ label: 'prompt length (tokens)', min: 0, max: 2000, step: 25, value: 200, onChange: (v) => { prompt = v; } });
        const gSl = ctx.slider({ label: 'tokens generated', min: 10, max: 1000, step: 10, value: 300, onChange: (v) => { gen = v; } });
        const cBtn = ctx.button('KV-cache: on', () => {
          cached = !cached;
          cBtn.textContent = 'KV-cache: ' + (cached ? 'on' : 'off');
        }, 'primary');
        const ro = ctx.readout();
        const human = (n) => n >= 1e9 ? (n / 1e9).toFixed(1) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e4 ? (n / 1e3).toFixed(0) + 'K' : Math.round(n).toLocaleString();

        ctx.loop(() => {
          g.clearRect(0, 0, 720, KH);
          /* Work to produce ONE more token when the sequence is already n long, counted in
             token-to-token comparisons — one unit is one query·key dot product and the value
             it drags along. With the cache: the new token's Q against n stored keys, so n.
             Without it: every token's K and V recomputed and full attention redone, so n².
             BOTH CURVES ARE IN THE SAME UNITS, so the gap you see on the axis is the real gap;
             per-step rather than cumulative, because per-step is what the reader feels. */
          const cachedAt = (n) => n;
          const uncachedAt = (n) => n * n;
          const cur = [], unc = [];
          for (let t = 0; t < gen; t++) { const n = prompt + t + 1; cur.push(cachedAt(n)); unc.push(uncachedAt(n)); }
          const peak = Math.max(unc[gen - 1], cur[gen - 1], 1);

          const P = { x: 62, y: 52, w: 418, h: 208 };
          g.font = 'bold ' + FONT; g.fillStyle = C.text; g.textAlign = 'left';
          g.fillText('work to produce each next token', P.x, 30);
          g.font = MONO; g.fillStyle = C.muted; g.textAlign = 'right';
          g.fillText('top of axis = ' + human(peak), P.x + P.w, 30);
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
          const px = (t) => P.x + t / Math.max(1, gen - 1) * P.w;
          /* log scale: on a linear axis normalised to the n^2 peak the cached
             curve is flat against the floor, which is the opposite of the point */
          const py = (v) => P.y + P.h - 2 - ctx.clamp(Math.log10(1 + Math.max(0, v)) / Math.log10(1 + peak), 0, 1) * (P.h - 4);
          const drawLine = (arr, col, wdt) => {
            g.strokeStyle = col; g.lineWidth = wdt; g.beginPath();
            arr.forEach((v, t) => { t ? g.lineTo(px(t), py(v)) : g.moveTo(px(t), py(v)); });
            g.stroke();
          };
          /* nothing data-driven may paint outside the plot frame */
          g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
          drawLine(unc, cached ? 'rgba(251,113,133,0.45)' : C.danger, cached ? 2 : 3);
          drawLine(cur, cached ? C.green : 'rgba(56,217,169,0.45)', cached ? 3 : 2);
          g.restore();

          g.font = MONO; g.fillStyle = C.muted; g.textAlign = 'center';
          g.fillText('tokens generated →', P.x + P.w / 2, P.y + P.h + 22);
          g.textAlign = 'left';
          g.save(); g.translate(P.x - 24, P.y + P.h - 20); g.rotate(-Math.PI / 2);
          g.fillText('work per token', 0, 0); g.restore();

          /* legend below the frame, where no curve can ever be drawn through it */
          const LY0 = P.y + P.h + 48;
          g.lineWidth = 3; g.strokeStyle = cached ? C.green : 'rgba(56,217,169,0.45)';
          g.beginPath(); g.moveTo(P.x, LY0 - 4); g.lineTo(P.x + 24, LY0 - 4); g.stroke();
          g.font = MONO; g.fillStyle = C.green;
          g.fillText('with cache: n — ' + human(cur[gen - 1]) + ' comparisons at the last step', P.x + 32, LY0);
          g.strokeStyle = cached ? 'rgba(251,113,133,0.45)' : C.danger;
          g.beginPath(); g.moveTo(P.x, LY0 + 16); g.lineTo(P.x + 24, LY0 + 16); g.stroke();
          g.fillStyle = C.danger;
          g.fillText('without: n² — ' + human(unc[gen - 1]) + ' comparisons at the last step', P.x + 32, LY0 + 20);

          const TX = 510;
          const finalN = prompt + gen;
          const saving = finalN;   // n^2 / n = n, the length of the sequence
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('per generated token', TX, 52);
          g.font = MONO; g.fillStyle = C.green;
          g.fillText('cached:   compute Q,K,V', TX, 76);
          g.fillText('          for 1 new token', TX, 92);
          g.fillStyle = C.danger;
          g.fillText('uncached: redo all', TX, 116);
          g.fillText('          ' + finalN + ' tokens', TX, 132);
          g.font = 'bold 20px Inter, system-ui, sans-serif';
          g.fillStyle = C.green;
          g.fillText(saving.toLocaleString() + '× less work', TX, 170);
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, 'A Key and Value never change once computed — only new tokens get added to the end. So store them.', TX, 194, 190, 16);
          g.font = 'bold ' + FONT; g.fillStyle = C.warn;
          g.fillText('the catch', TX, 268);
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, 'The cache grows with every token exchanged, so a long conversation costs more memory and slows down — even with the cache on.', TX, 288, 190, 16);
          ro.set({ 'context now': finalN.toLocaleString() + ' tokens', 'cache': cached ? 'on' : 'off', 'work saved': saving.toLocaleString() + '×' });
        });

        return ctx.figure(cv,
          'Generation is a loop: predict one token, append it, feed the whole sequence back in, predict again. Done naively, every step recomputes the Key and Value vectors for every earlier token — but those never change, so they are stored the first time and only the newest token is computed fresh. That is the KV-cache, and it is the single biggest reason a chat reply feels fast. Both curves are drawn in the same unit — one token-to-token comparison — on a logarithmic axis, because on a linear one the cached line sits flat against the floor: the gap between them is the whole story, and it is a factor of the context length. It also explains the thing you have felt: long conversations get slower and heavier, because the cache keeps growing with everything you have said.',
          [pSl, gSl, cBtn], ro);
      }
      /* ================================================================== */
      /* The chapter: touch first, read second.                             */
      /* ================================================================== */
      root.append(
        callout('tryit', '🖐 Do this first — find out why a model miscounts letters',
          `Ask most 2024-era chatbots how many r's are in "strawberry" and a surprising number say two.<br>
           <b>1.</b> The word is already loaded. Count the r's yourself: there are three, and it takes you a second.<br>
           <b>2.</b> Press <b>See it the way the model does</b>. The letters vanish and two bare numbers remain. <b>That is the entire input to layer 1.</b><br>
           <b>3.</b> Press <b>the</b>. One token, nothing hidden, no problem.<br>
           <b>4.</b> Try <b>hippopotamus</b>, then type a word of your own.`),
        strawberryLab(),
        p(`The model is not bad at spelling. A pocket calculator from 1975 could count characters perfectly. The model simply never receives the characters.`),
      );

      root.append(section('Tokens: how a model actually reads the page',
        p(`Before any attention happens, text has to become numbers. One number per word fails almost immediately — English has hundreds of thousands of words, new ones appear constantly, and "cats", "catlike" and "catastrophe" would each need an unconnected entry despite sharing letters.`),
        p(`One number per character fixes that — only about 100 symbols, nothing ever unrecognised — but a paragraph becomes a very long sequence of very uninformative units, and the model must rebuild "the" from t-h-e every single time.`),
        p(`Real tokenizers split the difference with <em>subword</em> tokens, built by <em>byte-pair encoding</em>. Start with individual characters, repeatedly find whichever adjacent pair appears most often in a huge pile of text, and glue it into a single new symbol. Repeat tens of thousands of times.`),
        p(`Early merges fuse common letter pairs ("t"+"h" → "th"). Later merges fuse whole words ("th"+"e" → "the"). Rare words never earn their own symbol, so they end up spelled out as two, three or five pieces stitched together from parts learned elsewhere.`),
        callout('tryit', '🖐 Try this: watch a tokenizer learn',
          `The demo below runs that exact algorithm, live, on a few hundred words of built-in text.<br>
           <b>1.</b> Step through the merges one at a time and read what each one fuses. The first few glue a single letter to the end-of-word mark ‿, or to another letter — the model is learning which pairs are common before it learns any words.<br>
           <b>2.</b> Keep going and watch whole words appear as single symbols.<br>
           <b>3.</b> Notice the vocabulary growing while the token count of the same text <b>falls</b>. That trade is the entire design.`),
        tokenizerDemo(),
        callout('key', '🔑 Why this costs you money',
          `Every API that serves a language model charges by the token, for what you send and what comes back, because a token is the model's actual unit of work — one forward pass per token produced.<br>
           A rough rule for English is about <b>four characters per token</b>, so a 500-word email is roughly 650–700 tokens.
           Code, dense mathematics, or a language whose script rarely appeared in training (many Southeast Asian and African languages) can cost <b>two to five times more tokens for the same content</b>, because the tokenizer never learned efficient chunks for it.`),
      ));

      root.append(section('The problem attention solves',
        p(`Chapter 5's recurrent networks read a sentence the way you read this one: left to right, one word at a time, carrying a running summary forward. That has two costs.`),
        p(`First, <b>speed</b>. Word 500 cannot be processed until words 1 to 499 have each taken their turn, so a GPU with thousands of idle cores sits mostly unused while the sequence trickles through.`),
        p(`Second, <b>memory</b>. Everything the network knows about word 1 must survive, compressed into one fixed-size vector, all the way to word 500 — and in practice it does not. That is chapter 5's <em>bottleneck problem</em>, and the vanishing gradients from the same chapter make it worse: the path the information has to travel is also the path the learning signal has to travel back along.`),
        p(`Attention removes both with one change: instead of relaying information down a chain, let every token query every other token directly. Word 500 looks straight at word 1 in a single step, with no relay and no fading.`),
        p(`And crucially every token can do this <i>at the same time</i>, because "look at everything" is one matrix multiplication — exactly what a GPU is built to do in parallel. That is the trade the 2017 paper made in its title.`),
        callout('key', '🔑 Query, Key, Value — the dating-app analogy',
          `You have a <em>type</em>: the sort of profile you are looking for. That is your <b>Query</b>.<br>
           Every other profile has a short headline summarising what they are about. That is their <b>Key</b>.<br>
           You compare your type against every headline and get a compatibility score for each person.<br>
           Then the twist that makes attention different from search: instead of picking your single best match, you take a <b>blend of everyone's full profile</b> — their <b>Value</b> — weighted by how well each scored. A perfect match dominates; a bad match still contributes a whisper.<br>
           Every token does this simultaneously: once as a query looking outward, once as a key and value being looked at by everyone else.`),
        p(`Concretely: every token's embedding is multiplied by three learned weight matrices to produce a Query <b>q</b>, a Key <b>k</b> and a Value <b>v</b>. For one query token, compute its compatibility with every token's key by a dot product, scale it down, then squash the row through <em>softmax</em> so the scores are positive and sum to exactly 1 — a genuine weighted average.`),
        callout('tryit', '🖐 Try this: compute attention by hand',
          `Three tiny 2-dimensional tokens, so you can check every digit with a calculator if you want to.<br>
           <b>1.</b> Step through the four stages: dot products, scale by √d, softmax, then blend the Values.<br>
           <b>2.</b> Drag a Query vector and watch which token it starts leaning on.<br>
           <b>3.</b> Watch the softmax row: it always sums to 1. Nothing is left over and nothing is negative.`),
        attentionByHand(),
        p(`Q·Kᵀ produces one raw score for every (query, key) pair at once — a full grid. Dividing by √d matters more than it looks: as the dimension grows, a plain dot product's typical size grows with it, and uncorrected the scores swing so wildly that softmax collapses into an all-or-nothing spike, which starves the gradient during training.`),
        p(`That four-step recipe — score, scale, softmax, blend — run for every token against every other token in one matrix multiplication, <b>is the entire mechanism</b>. Nothing else in a transformer is conceptually harder. Everything else is scale and repetition.`),
      ));

      root.append(section('What attention buys you, in language',
        p(`Toy numbers make the arithmetic checkable; the payoff is linguistic. A model can resolve a pronoun to the noun it stands for, directly, by attending across the whole sentence.`),
        p(`The classic test is a Winograd-style sentence: "The animal didn't cross the street because <b>it</b> was too tired." Every human instantly reads "it" as the animal, because a tired street makes no sense. Real trained attention heads learn exactly this kind of link from data.`),
        callout('tryit', '🖐 Try this: the attention visualiser',
          `<b>1.</b> Click the word <b>it</b> and watch where the weight goes.<br>
           <b>2.</b> Now swap the two nouns round: type <b>The street didn't bother the animal because it was too wide</b>. The link follows the word order, not the meaning — which is the honest limit of a hand-written rule, and exactly what a trained head learns to do better.<br>
           <b>3.</b> Toggle <b>causal mask</b> on and off. With it on, no token can attend to anything to its right, which is the constraint every GPT-style model trains under.`),
        callout('warning', '⚠️ Illustrative, not trained',
          `The visualiser below fakes one attention head with a few hand-written rules — a crude part-of-speech guess, a boost from pronouns back toward the sentence's first content word, a boost from "the" toward the following word, and a mild preference for nearby words.
           It shows you the <b>shape</b> of the behaviour without needing billions of trained weights. A real head is messier and was never told any of these rules.`),
        attentionVisualiser(),
        p(`One attention computation learns one <i>kind</i> of relationship — say, pronoun-to-noun. A sentence needs many at once: which adjective modifies which noun, which verb takes which object, which word agrees with which.`),
        p(`So a transformer runs attention several times in parallel, called <em>heads</em>, each with its own Q, K, V matrices, each free to specialise. A model with d = 768 and 12 heads gives each head 64 dimensions; the heads run simultaneously, and their outputs are concatenated back to 768 and passed through one more learned matrix that mixes their findings.`),
        p(`That is <em>multi-head attention</em>: not a bigger version of the mechanism, just several independent copies run side by side, each looking for something different.`),
      ));

      root.append(section('The blind spot: attention has no idea what order anything is in',
        p(`Here is a strange fact about Q·Kᵀ. It does not care about word order at all. Swap two tokens and the set of dot products between them is identical — attention is fundamentally a <i>set</i> operation.`),
        callout('tryit', '🖐 Try this — break the model\'s grasp of word order',
          `<b>1.</b> Turn <b>positional encoding off</b>, then press <b>Swap dog and man</b>. Read the grid carefully: the same numbers come back, merely rearranged. The two sentences are <b>literally indistinguishable</b>.<br>
           <b>2.</b> Turn positional encoding back on and swap again. Now the scores genuinely change.<br>
           <b>3.</b> Look at the right-hand pattern and drag <b>inspect position</b>. Fast waves at the top separate neighbouring positions; slow waves at the bottom separate distant ones.`),
        positionLab(),
        p(`That is a real problem, because "the dog bit the man" and "the man bit the dog" contain exactly the same words. An RNN got order for free by reading left to right. A transformer has to be told.`),
        p(`The fix is <em>positional encoding</em>: before the first layer, add a vector to each token's embedding that encodes its position, so token 1 and token 50 differ even when the word is identical.`),
        p(`The 2017 paper used the fixed pattern of sine and cosine waves you just dragged — no learning required, and the authors hoped it would let a model handle sequences longer than any it trained on. Measured later, it does not; that is why almost every model since 2023 uses RoPE instead. Many models instead learn a position vector per slot.`),
        p(`Most models from 2023 onward use <em>RoPE</em> (rotary position embedding), which rotates each Query and Key by an angle proportional to its position, so the dot product between two tokens naturally reflects their <i>relative</i> distance rather than absolute position. Worth knowing the name of; not worth deriving here.`),
      ));

      root.append(section('The block, and the highway running through it',
        p(`Attention only moves information <i>between</i> positions; it never processes the content of one position on its own. So every attention step is paired with a small ordinary <em>MLP</em> — chapter 2's network, applied to each token independently with the same weights — giving the model room to actually compute with what attention gathered.`),
        p(`Wrap each of the two in a <em>residual connection</em> — add the sublayer's output back onto its input rather than replacing it — plus a normalisation step that keeps activations in a stable range, and you have the standard block.`),
        callout('key', '🔑 The residual stream',
          `Think of <b>x</b>, the running vector at each token position, as a highway down the length of the network.
           Attention and the MLP are off-ramps: they read the highway, do their work, and merge their result back on. <b>The highway itself is never overwritten, only added to.</b><br>
           This is what lets networks stack dozens or hundreds of blocks without gradients vanishing on the way back — the same failure that limited pre-2015 networks, and the same fix as the LSTM's conveyor belt in chapter 5.
           A gradient can always flow straight back down the highway, with every sublayer offering an optional shortcut rather than a mandatory bottleneck.`),
        callout('tryit', '🖐 Try this: follow the pulse',
          `Watch one pulse make a full pass through the stack, and note that the <b>same two-step recipe</b> repeats layer after layer.<br>
           Drag the layer slider: GPT-2 small stacks it 12 times, and the largest 2026 models stack it over a hundred.<br>
           Attention is computed for every token in parallel; only the layer-by-layer stacking is sequential.`),
        blockDiagram(),
      ));

      root.append(section('Two halves, and why one of them won',
        p(`The original 2017 architecture was built for translation and had two halves. An <em>encoder</em> reads the whole source sentence with unrestricted attention — every token sees every other, including later ones — and builds a representation of what it means. A <em>decoder</em> then generates the translation one word at a time, attending both to its own previous output and, via cross-attention, to the encoder's representation.`),
        p(`Two influential 2018 models each kept one half. <b>BERT</b> is encoder-only: full bidirectional attention, trained by hiding random words and asking the model to fill them in. Excellent for understanding text — search, classification, the embeddings of chapter 6 — but never designed to generate long free-form prose.`),
        p(`<b>GPT</b> is decoder-only, and this is where <em>causal masking</em> comes in. A decoder predicting word 50 must not peek at the real word 50 sitting in the training example — that would make training trivial and useless, since at generation time word 50 does not exist yet. So every query is masked to attend only to its own position or earlier, exactly the toggle in the visualiser above.`),
        p(`GPT won as the shape for general assistants for one economic reason: next-token prediction on plain unlabelled text needs no translation pairs and no hand-labelled examples, so it can train on virtually the whole internet. And one decoder can be prompted to translate, summarise, code or chat — tasks that used to need separate systems.`),
        callout('history', '📜 Three papers, twenty months, 2017–2019',
          `<b>June 2017:</b> "Attention Is All You Need" (Vaswani et al., Google) introduces the transformer for machine translation, and beats the recurrent state of the art while training far faster.<br>
           <b>October 2018:</b> BERT (Devlin et al., Google) keeps the encoder, trains by masked-word prediction, and takes the top of nearly every language-understanding benchmark at once.<br>
           <b>February 2019:</b> GPT-2 (Radford et al., OpenAI) keeps the decoder, scales next-token prediction to 1.5 billion parameters, and produces text fluent enough that its staged release became a public argument about AI risk.<br>
           Everything since has largely been that third path, made much bigger.`),
        callout('example', '🌍 Where each half lives today',
          `<b>Encoder-only</b> models still quietly run an enormous amount of infrastructure: search ranking, document classification, spam and abuse detection, and the embedding models behind vector search in chapter 12. They are small, fast and cheap, and they never needed to generate a word.<br>
           <b>Decoder-only</b> models are what you talk to.<br>
           <b>Encoder–decoder</b> survives where there genuinely are two distinct sequences: translation systems, speech-to-text, and some summarisation models.`),
      ));

      root.append(section('Count the parameters yourself',
        p(`"124 million parameters" sounds like a fact you would have to look up. It is arithmetic, and doing it once demystifies every model-size headline you will ever read.`),
        callout('tryit', '🖐 Try this',
          `<b>1.</b> Press <b>GPT-2 small</b>: d = 768, 12 layers, 50,257 tokens. The total lands on <b>124M</b> — the published figure.<br>
           <b>2.</b> Press <b>GPT-3</b>: d = 12,288 and 96 layers. It lands near <b>175B</b>.<br>
           <b>3.</b> Now the lesson. Press <b>GPT-2 small</b>, then double <b>layers</b>: the <b>blocks</b> row doubles exactly, 85M to 170M, while the grand total goes 124.3M to 209.3M — the embedding tables do not grow with depth. Then put the layers back and double <b>width</b> instead: the blocks row roughly <b>quadruples</b>, because the per-block cost is 12<i>d</i>².`),
        paramCalc(),
        p(`Each block spends 4<i>d</i>² on attention — the Q, K and V matrices plus the output projection — and 8<i>d</i>² on its MLP, whose hidden layer is conventionally four times the width. That is 12<i>d</i>² per block.`),
        p(`Twelve blocks of 7.08M is about 85M. Add 38.6M of token embeddings and 0.8M of position embeddings and you get 124M. GPT-2 ties its output layer to the input embedding matrix, reusing the same numbers to turn the final vector back into probabilities, which is why there is no separate un-embedding line.`),
      ));

      root.append(section('Generating: why the reply arrives word by word',
        p(`Training sees a whole sentence at once and predicts every next-token in parallel. Using the model — <em>inference</em> — is different: it only knows the tokens generated so far, so it produces its reply <em>autoregressively</em>. Predict the most likely next token, append it, feed the now-longer sequence back in, predict again.`),
        p(`That loop, repeated hundreds or thousands of times, is why a chat response streams into view word by word. It is not a UI affectation. It is the actual order of computation.`),
        callout('tryit', '🖐 Try this — the optimisation you have already felt',
          `<b>1.</b> With the <b>KV-cache on</b>, the work curve is a straight line. Turn it <b>off</b> and watch it bend upward.<br>
           <b>2.</b> Drag <b>prompt length</b> to 2,000 and read the "work saved" figure. That is the difference between a reply that streams and one that crawls.<br>
           <b>3.</b> Now the part you have felt without knowing why: with the cache on, drag <b>tokens generated</b> up and watch the per-step cost keep climbing anyway.`),
        kvCacheLab(),
        p(`Done naively, the loop recomputes Key and Value vectors for every earlier token at every step — wasteful, since a token's Key and Value never change once computed. Only new tokens are ever added to the end.`),
        p(`The fix is a <em>KV-cache</em>: store every token's Key and Value the first time, and at each step compute Q, K and V for only the newest token, comparing its fresh Query against the whole cached history. It is the single biggest reason chat responses feel fast.`),
        p(`It is also why very long conversations get slower and use more memory: the cache keeps growing with every token you have exchanged, and it all has to sit in GPU memory at once.`),
      ));

      root.append(section('Why this matters for modern AI',
        p(`Two properties, working together, ended the architecture argument.`),
        p(`First, <b>parallel training</b>. Because attention over a whole sequence is one matrix multiplication rather than a step-by-step loop, an entire training example of thousands of tokens is processed in one shot, and thousands of examples across thousands of GPUs run simultaneously. An RNN's one-step-at-a-time nature made it structurally unable to use hardware that way, no matter how many GPUs you bought.`),
        p(`Second, <b>clean scaling</b>. Transformers reliably keep getting better as you add data, parameters and compute, in a smooth and predictable way — the subject of chapter 10's scaling laws — with no sign through years of scaling that the returns simply stop.`),
        p(`A parallelisable architecture that also scales predictably is exactly the combination that turns "bigger GPU budget" into "better model", which is the entire economic engine behind the last eight years of AI progress.`),
        p(`Every model you can name — Claude, the GPT family, Gemini, Llama — is this decoder-only recipe: tokens in, position information supplied somehow (added at the input, or rotated into Q and K inside every layer, as RoPE does), N copies of attention-then-MLP-with-residuals, a projection back to vocabulary-sized probabilities — usually the embedding matrix reused — generated one token at a time behind a KV-cache.`),
        p(`The differences between them are almost entirely differences of degree and detail covered later in this course — how many layers, how wide, what data, what fine-tuning — not differences in this skeleton. If you understand this page, you understand mechanically what happens between pressing enter and a reply appearing, for every major model in existence.`),
      ));

      root.append(
        ctx.quiz([
          { q: 'Why do language models miscount the letters in "strawberry"?', options: ['Arithmetic on letters is hard for computers', 'The word arrives as a couple of opaque chunk-numbers, so the model must recall how each chunk is spelled rather than simply looking at the letters', 'The model was never trained on the word', 'Tokenizers delete repeated letters'], answer: 1, explain: 'You saw it in the opening demo: press "See it the way the model does" and only the numbers remain. Spelling is one step removed from the meaning those numbers were built to carry. Try "the" — one token, no unpacking, no problem.' },
          { q: 'What does dividing by √d accomplish in the attention formula?', options: ['It normalises the output to length 1', 'It keeps raw dot products in a sane range as the dimension grows, so softmax does not collapse into an all-or-nothing spike and starve the gradient', 'It makes the computation faster', 'It is required for the causal mask'], answer: 1, explain: 'A dot product of longer vectors sums more terms, so its typical size grows like √d — which is exactly why the divisor is √d and not d. Uncorrected, the scores swing far enough that softmax saturates, gradients go to nearly zero, and training stalls.' },
          { q: 'You turned positional encoding off and swapped two words. What happened to the attention grid, and why?', options: ['It went blank', 'The same numbers came back, merely rearranged — because Q·Kᵀ compares content against content and has no idea where either token sat', 'The scores doubled', 'Nothing, because attention already tracks order'], answer: 1, explain: 'Attention is a set operation. That is why a position-dependent vector is added to every embedding before layer 1: without it, "the dog bit the man" and "the man bit the dog" are literally indistinguishable to the mechanism.' },
          { q: 'A transformer block has 12d² parameters. What does that imply about making a model bigger?', options: ['Depth and width cost the same', 'Doubling the width roughly quadruples the per-block cost, while doubling the depth only doubles it', 'Width is free', 'Parameter count does not depend on d'], answer: 1, explain: 'The per-block cost is quadratic in width and linear in the number of blocks, which you can verify on the calculator: press GPT-2 small, then double layers (total doubles), then double width instead (total roughly quadruples).' },
          { q: 'What is a KV-cache and what does it cost you?', options: ['It stores the model weights closer to the GPU; it costs nothing', 'It stores every token\'s Key and Value so they are not recomputed each step — making replies fast, but growing with the conversation, so long chats get slower and heavier', 'It caches common prompts so repeated questions are free', 'It compresses the context window'], answer: 1, explain: 'A token\'s Key and Value never change once computed, so storing them turns each generation step into a small increment instead of redoing all the past work. The cache lives in GPU memory and grows with every token exchanged, which is exactly why a very long conversation feels slower.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://jalammar.github.io/illustrated-transformer/" target="_blank" rel="noopener">Jay Alammar, "The Illustrated Transformer"</a>: the same architecture with a different set of pictures. The best second explanation there is.`,
            `<a href="https://www.youtube.com/watch?v=kCc8FmEb1nY" target="_blank" rel="noopener">Karpathy, "Let's build GPT: from scratch, in code, spelled out"</a>: two hours that build everything on this page in Python. Lab 06 of this course follows it.`,
            `<a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener">Vaswani et al. (2017), "Attention Is All You Need"</a>: the paper. Eleven pages, and section 3.2 is the formula you computed by hand.`,
            `<a href="https://arxiv.org/abs/1810.04805" target="_blank" rel="noopener">Devlin et al. (2018), "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"</a>: the encoder-only branch, and masked-language-model training.`,
            `<a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" target="_blank" rel="noopener">Radford et al. (2019), "Language Models are Unsupervised Multitask Learners"</a>: GPT-2, and the argument that one decoder can do every task.`,
          ]),
        ),
      );
    },
  });
})();
