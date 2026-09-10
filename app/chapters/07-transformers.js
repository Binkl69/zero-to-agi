/* Zero → AGI · Chapter 07 · Attention is all you need: the transformer
   Tokenisation (BPE toy tokenizer trained at load time); the parallelism problem attention solves;
   Query/Key/Value maths with a worked 3-token example; multi-head attention; positional encoding;
   the full block (attention → add&norm → MLP → add&norm) and the residual stream; encoder vs
   decoder and causal masking; GPT-2 small parameter arithmetic; autoregressive inference + KV-cache.
   Interactives: (a) attention visualiser on an editable sentence (toy hand-crafted rule), (b) compute
   attention by hand on 3 tokens with Q/K/V sliders, (c) a real BPE tokenizer trained on a small
   built-in corpus, (d) an animated transformer-block diagram with a pulse on the residual stream. */
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
  const BPE_RANK = new Map(BPE_MERGES.map((k, i) => [k, i]));
  const BPE_WORD_COUNT = new Set((CORPUS_TEXT.toLowerCase().match(/[a-z]+/g) || [])).size;

  function bpeEncodeWord(word) {
    let symbols = word.split('').concat(['</w>']);
    if (symbols.length <= 2) return symbols;
    // eslint-disable-next-line no-constant-condition
    while (true) {
      let bestRank = Infinity, bestIdx = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const r = BPE_RANK.get(symbols[i] + '' + symbols[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; bestIdx = i; }
      }
      if (bestIdx === -1) break;
      symbols = symbols.slice(0, bestIdx).concat([symbols[bestIdx] + symbols[bestIdx + 1]], symbols.slice(bestIdx + 2));
    }
    return symbols;
  }
  function bpeTokenizeText(text) {
    const out = [];
    const re = /([a-zA-Z]+)|([0-9]+)|(\s+)|([^\sa-zA-Z0-9]+)/g;
    let m;
    while ((m = re.exec(text))) {
      if (m[1]) {
        const syms = bpeEncodeWord(m[1].toLowerCase());
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
    tagline: 'Every token looks at every other token at once — no recurrence, no waiting in line — and that one idea rebuilt every model you use today.',
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
        const PALETTE = [C.accent, C.green, C.warn, C.danger, C.pink, C.purple, C.orange];
        const ro = ctx.readout();
        const S = { text: "Let's count: how many r's are in strawberry?" };
        function render() {
          box.innerHTML = '';
          const clipped = S.text.length > 500;
          const text = S.text.slice(0, 500);
          const toks = text.length ? bpeTokenizeText(text) : [];
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
          ro.set({
            tokens: toks.length,
            characters: nChars,
            'chars / token': toks.length ? (nChars / toks.length).toFixed(2) : '–',
            'illustrative cost @ $3/M tok': '$' + (toks.length * rate).toFixed(6),
            'merges learned': BPE_MERGES.length,
          });
        }
        const ta = ctx.textarea({ label: 'Type or paste text', value: S.text, onChange: (v) => { S.text = v; render(); } });
        const b1 = ctx.button('Preset: “strawberry”', () => { S.text = 'strawberry'; ta.value = S.text; render(); });
        const b2 = ctx.button('Preset: a long rare word', () => { S.text = 'pneumonoultramicroscopicsilicovolcanoconiosis'; ta.value = S.text; render(); });
        const b3 = ctx.button('Preset: a normal sentence', () => { S.text = 'The transformer changed how machines read language forever.'; ta.value = S.text; render(); });
        render();
        return ctx.figure(box,
          `This is a real byte-pair-encoding (BPE) tokenizer, trained right now, in your browser, on a ${BPE_WORD_COUNT}-word built-in corpus about language models (not the internet) — it learned ${BPE_MERGES.length} merge rules by repeatedly fusing the most frequent adjacent pair of symbols. Each coloured chip is one token; “${'‿'}” marks the end of a word, the way real tokenizers mark word boundaries. Common words from its training text often survive as one piece; rare or unfamiliar ones fragment into smaller chunks. A production tokenizer (GPT-4's, Claude's) is the same algorithm trained on hundreds of billions of characters, so it recognises far more whole words — but any invented or rare-enough string still gets chopped up exactly like this.`,
          [ta, b1, b2, b3], ro);
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
          gc.font = MONO; gc.textAlign = 'left';
          gc.fillText(label, p1.x + 6, p1.y + 4 + (labelOff || 0));
          gc.restore();
        }
        function bars(gc, x0, y0, w2, h2, vals, labels, mode) {
          gc.save();
          gc.strokeStyle = C.line; gc.strokeRect(x0, y0, w2, h2);
          const bw = w2 / vals.length;
          const maxAbs = mode === 'unit' ? 1 : 4;
          const zero = mode === 'unit' ? y0 + h2 : y0 + h2 / 2;
          if (mode !== 'unit') { gc.strokeStyle = C.muted; gc.globalAlpha = 0.5; gc.beginPath(); gc.moveTo(x0, zero); gc.lineTo(x0 + w2, zero); gc.stroke(); gc.globalAlpha = 1; }
          vals.forEach((v, i) => {
            const cx = x0 + bw * (i + 0.5);
            const bh = Math.min(mode === 'unit' ? h2 : h2 / 2, Math.abs(v) / maxAbs * (mode === 'unit' ? h2 : h2 / 2));
            const barY = v >= 0 ? zero - bh : zero;
            gc.fillStyle = v >= 0 ? C.accent : C.danger;
            gc.fillRect(cx - bw * 0.28, barY, bw * 0.56, Math.max(1, bh));
            gc.fillStyle = C.text; gc.font = MONO; gc.textAlign = 'center';
            gc.fillText(v.toFixed(2), cx, v >= 0 ? barY - 4 : barY + bh + 12);
            gc.fillStyle = C.muted; gc.fillText(labels[i], cx, y0 + h2 + 14);
          });
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
          for (let t = 0; t < 3; t++) {
            arrow(g, S.Q[t], C.accent, null, 'Q' + (t + 1), -6);
            arrow(g, S.K[t], C.warn, [5, 3], 'K' + (t + 1), 6);
            arrow(g, S.V[t], C.green, [1, 4], 'V' + (t + 1), 16);
          }
          const r = compute();
          arrow(g, r.out, C.pink, null, 'output', 26);
          g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
          wrapText(g, 'solid = Q · dashed = K · dotted = V · thick pink = the attention output for the selected query', PX, PY + PS + 18, PS, 13);

          const RX = 320, RW = W - RX - 16;
          g.fillStyle = C.text; g.font = 'bold 13px Inter, system-ui, sans-serif'; g.textAlign = 'left';
          g.fillText('query = token ' + (S.qi + 1) + '  —  scores against every key', RX, 20);
          bars(g, RX, 34, RW, 70, r.raw, ['K1', 'K2', 'K3']);
          g.fillStyle = C.muted; g.font = FONT; g.fillText('raw  QKᵀ', RX, 122);
          bars(g, RX, 138, RW, 70, r.scaled, ['K1', 'K2', 'K3']);
          g.fillText('scaled  ÷ √2', RX, 226);
          bars(g, RX, 242, RW, 70, r.w, ['K1', 'K2', 'K3'], 'unit');
          g.fillText('softmax weights (sum to 1)', RX, 330);
          g.fillStyle = C.text; g.font = MONO; g.textAlign = 'left';
          wrapText(g, 'output = ' + r.w.map((w, i) => f2(w) + '·V' + (i + 1)).join(' + ') + ' = (' + f2(r.out[0]) + ', ' + f2(r.out[1]) + ')', RX, 352, RW, 15);
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

        const W = 720, H = 500;
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
          wrapText(g, 'border colour = toy part-of-speech guess: pink = pronoun, blue = content word, orange = verb/aux, purple = conjunction, grey = determiner/function word', 14, 205, W - 28, 13);
          if (a.truncated) { g.fillStyle = C.warn; g.fillText('(showing first 16 words)', 14, 235); }

          // matrix
          const n2 = n, cell = ctx.clamp(Math.min(340 / n2, 30), 12, 30);
          const mx0 = (W - n2 * cell) / 2, my0 = 258;
          matrixGeo = { x0: mx0, y0: my0, cell, n: n2 };
          g.fillStyle = C.text; g.font = 'bold 12px Inter, system-ui, sans-serif'; g.textAlign = 'left';
          g.fillText('Every query (row) × every key (column) — click a row to inspect it above.', 14, my0 - 10);
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
        const X0 = 64, X1 = 656, LY = 250, BOXY = 108, BOXH = 46;
        const S = { layers: 6, u: 0, playing: true, speed: 1 };
        const trail = [];
        function blockGeo(k, L) {
          const bw = (X1 - X0) / L, bx = X0 + (k + 0.5) * bw;
          return { bx, bw, attnX: bx - bw * 0.2, mlpX: bx + bw * 0.2 };
        }
        function phaseAt(u, L) {
          const k = Math.min(L - 1, Math.floor(u)), t = u - k;
          const { bx, bw, attnX, mlpX } = blockGeo(k, L);
          const lp = (a, b, u2) => a + (b - a) * ctx.clamp(u2, 0, 1);
          let x, y, label = '', box = null;
          if (t < 0.10) { x = lp(bx - bw * 0.5, attnX, t / 0.10); y = LY; }
          else if (t < 0.18) { x = attnX; y = lp(LY, BOXY + BOXH, (t - 0.10) / 0.08); }
          else if (t < 0.40) { x = attnX; y = BOXY + BOXH / 2; label = 'Attention — every token gathers information from every other token, all at once'; box = 'attn'; }
          else if (t < 0.48) { x = attnX; y = lp(BOXY + BOXH, LY, (t - 0.40) / 0.08); label = 'Add & Norm — attention’s output is added back onto the residual stream, then normalised'; }
          else if (t < 0.55) { x = lp(attnX, mlpX, (t - 0.48) / 0.07); y = LY; label = 'Add & Norm'; }
          else if (t < 0.63) { x = mlpX; y = lp(LY, BOXY + BOXH, (t - 0.55) / 0.08); }
          else if (t < 0.90) { x = mlpX; y = BOXY + BOXH / 2; label = 'MLP — the same small feed-forward network applied to each token independently'; box = 'mlp'; }
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
          g.textAlign = 'center'; g.fillStyle = C.accent; g.fillText('residual stream', (X0 + X1) / 2, LY - 10);

          const ph = phaseAt(S.u, L);
          if (ph.box) {
            const geo = blockGeo(ph.k, L);
            const bxp = ph.box === 'attn' ? geo.attnX : geo.mlpX;
            g.strokeStyle = ph.box === 'attn' ? C.pink : C.orange; g.lineWidth = 2;
            g.strokeRect(bxp - 46, BOXY, 92, BOXH);
            g.fillStyle = '#111827'; g.fillRect(bxp - 46, BOXY, 92, BOXH);
            g.strokeRect(bxp - 46, BOXY, 92, BOXH);
            g.fillStyle = ph.box === 'attn' ? C.pink : C.orange; g.font = 'bold 12px Inter, system-ui, sans-serif'; g.textAlign = 'center';
            g.fillText(ph.box === 'attn' ? 'Multi-Head Attention' : 'MLP', bxp, BOXY + BOXH / 2 + 4);
          }
          trail.push({ x: ph.x, y: ph.y });
          if (trail.length > 16) trail.shift();
          trail.forEach((p, i) => {
            const al = (i + 1) / trail.length;
            g.beginPath(); g.arc(p.x, p.y, 3 + al * 4, 0, Math.PI * 2);
            g.fillStyle = 'rgba(251,113,133,' + (al * 0.8) + ')'; g.fill();
          });
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
          if (S.playing) { S.u += dt * S.speed * 0.85; if (S.u >= L) S.u -= L; }
          draw();
          ro.set({ layers: L, 'params scale with': '12·d² per layer (next section)' });
        });
        const layerSl = ctx.slider({ label: 'layers', min: 1, max: 12, step: 1, value: 6, onChange: (v) => { S.layers = Math.round(v); S.u = 0; trail.length = 0; } });
        const playBtn = ctx.button('⏸ Pause', () => { S.playing = !S.playing; playBtn.textContent = S.playing ? '⏸ Pause' : '▶ Play'; }, 'primary');
        const speedSl = ctx.slider({ label: 'speed', min: 0.25, max: 2, step: 0.25, value: 1, fmt: (v) => v.toFixed(2) + '×', onChange: (v) => { S.speed = v; } });
        return ctx.figure(cv,
          'One pulse, one full pass through the stack. Depth means the SAME two-step recipe (attention, then MLP, each followed by Add & Norm) repeats layer after layer — GPT-2 small stacks it 12 times; the largest 2026 models stack it over a hundred times. Attention itself is computed for every token in parallel; only the layer-by-layer stacking is sequential.',
          [layerSl, playBtn, speedSl], ro);
      }

      /* ================================================================== */
      /* Prose                                                                */
      /* ================================================================== */
      root.append(
        p(`Ask most 2024-era chatbots how many r's are in "strawberry" and a surprising number of them say two. Not because the model is bad at spelling in any human sense, and not because arithmetic on letters is hard for a machine — a pocket calculator from 1975 could count characters perfectly. It is because the model never sees the letters s-t-r-a-w-b-e-r-r-y one at a time. It sees a short row of two or three opaque chunks, each one a single number, and it was never shown a task that required unpacking those chunks back into individual characters. Understanding why requires understanding how text becomes numbers in the first place, and that turns out to be the smaller half of this chapter.`),
        p(`The bigger half is the architecture underneath every headline model of the last eight years: Claude, GPT, Gemini, Llama. In 2017 a small team at Google published a paper with an almost arrogant title, "Attention Is All You Need", and it was right. It threw away recurrence — the one-word-at-a-time processing that chapter 5's RNNs and LSTMs relied on — and replaced it with a single mechanism: every token looks directly at every other token, all at once, in parallel. No message has to travel step by step down a chain to get from word 1 to word 500. It can just look.`),
        p(`This chapter builds that mechanism from the ground up: the vectors it multiplies, the exact arithmetic, the block it lives inside, and the two ways models are built from it. By the end you will be able to compute a real transformer's parameter count on the back of an envelope.`),

        section('Tokens: how a model actually reads the page',
          p(`Before any attention happens, text has to become numbers. The crude approach is one number per word: give "cat" the number 4,821, give "dog" the number 917, and so on. That fails almost immediately — English has hundreds of thousands of words, new ones appear constantly, and "cats", "catlike" and "catastrophe" would each need their own unrouted entry despite sharing a root. The other crude approach is one number per character: only about 100 symbols needed, nothing is ever unrecognised, but a paragraph becomes a very long sequence of very uninformative units, and the model has to rebuild "the" from t-h-e every single time it appears.`),
          p(`Real tokenizers split the difference with <em>subword</em> tokens, built by an algorithm called <em>byte-pair encoding</em> (BPE). The idea, in one sentence: start with individual characters, then repeatedly find whichever adjacent pair appears most often anywhere in a huge pile of text and glue it into a single new symbol — repeat tens of thousands of times. Early merges fuse common letter pairs ("t"+"h" → "th"); later merges fuse whole common words ("th"+"e" → "the"); rare words never get their own dedicated symbol, so they end up spelled out as two, three or five subword pieces stitched together from parts learned elsewhere. The demo below trains this exact algorithm, live, on a few hundred words of built-in text, so you can watch it happen.`),
          p(`This is the real answer to the strawberry question. "Strawberry" is not a common enough word in most tokenizers' training data to earn its own single symbol, so it gets cut into pieces such as "straw" and "berry" (GPT-4's tokenizer does roughly this). The model's very first layer only ever receives those two chunk-numbers. To answer "how many r's", it would have to have learned, purely statistically, that the chunk spelled "berry" happens to contain two r's — a fact about the chunk's <i>spelling</i>, which is one step removed from the chunk's <i>meaning</i>, which is the only thing the chunk-number was ever designed to carry. Ask it to count letters in a word that tokenizes as a single familiar chunk, and it does much better, because now there is nothing to unpack.`),
          p(`Tokens matter for your wallet too. Every API that serves a language model charges by the token, for both what you send and what it sends back, because tokens are the model's actual unit of work — one forward pass through the network per token produced. A rough rule of thumb for English is about four characters per token, so a 500-word email is roughly 650–700 tokens. A codebase, dense mathematics, or a language whose script rarely appears in the training data (many Southeast Asian and African languages) can cost two to five times more tokens for the same content, because the tokenizer never learned efficient chunks for it.`),
          callout('tryit', 'Try it: watch a tokenizer learn', `Press the presets below. <b>“strawberry”</b>: watch it fragment into pieces that have nothing to do with the fruit — this toy tokenizer was trained on text about language models, not berries, so it never learned a chunk for "berry". <b>Long rare word</b>: watch a word nobody has ever needed a token for get cut almost to single letters. <b>A normal sentence</b>: notice how many common words already survive whole after only a couple hundred merges. Then type your own text — try your name, or a word in another language.`),
          tokenizerDemo(),
        ),

        section('The problem attention solves',
          p(`Chapter 5's recurrent networks read a sentence the way you read this one: left to right, one word at a time, carrying a running summary forward in a hidden state. That has two costs. First, speed: word 500 cannot be processed until words 1 through 499 have each taken their turn, so a GPU with thousands of idle cores sits mostly unused while the sequence trickles through step by step. Second, memory: everything the network knows about word 1 has to survive, compressed into one fixed-size vector, all the way to word 500 — and in practice it doesn't; distant information fades, which is exactly the vanishing-gradient problem from chapter 5.`),
          p(`Attention removes both costs with one change: instead of relaying information through a chain, let every token query every other token directly. Word 500 can look straight at word 1 in a single step, with no relay and no fading, and — crucially — every token can do this <i>at the same time</i>, because "look at everything" is just one matrix multiplication, and matrix multiplications are exactly what GPUs are built to do in parallel. This is the trade the 2017 paper made explicit in its title: attention alone, no recurrence, is all you need.`),
        ),

        section('Query, Key, Value: the mechanism itself',
          p(`Picture a dating app. You have a <em>type</em> — the sort of profile you're looking for. Call that your <em>Query</em>. Every other profile on the app has posted a short headline summarising what they're about — call that their <em>Key</em>. You compare your type against every headline and get a compatibility score for each person. Now here is the twist that makes attention different from a simple search: instead of picking your single best match, you take a <i>blend</i> of everyone's actual full profile — their <em>Value</em> — weighted by how well each one scored. A perfect match dominates the blend; a bad match still contributes a whisper. Every token in a sentence runs this exact process simultaneously, once as a query looking outward and once as a key/value being looked at by everyone else.`),
          p(`Concretely: every token's embedding is multiplied by three learned weight matrices to produce three vectors — a Query <b>q</b>, a Key <b>k</b>, and a Value <b>v</b>, each of some dimension <i>d</i>. For one query token, compute its raw compatibility with every token's key by a dot product, scale it down, then squash the whole row through <em>softmax</em> so the scores become positive and sum to exactly 1 — a genuine weighted average, nothing left over and nothing negative. The full formula for a whole sentence at once, stacking every token's q, k, v as rows of matrices Q, K, V:`),
          ctx.code('Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V'),
          p(`Q·Kᵀ produces one raw score for every (query, key) pair at once — a full grid. Dividing by √d matters more than it looks: as the vector dimension <i>d</i> grows, a plain dot product's typical size grows too (more terms being summed), and without correcting for it, scores would swing so wildly that softmax collapses into an almost all-or-nothing spike, which starves the gradient during training. Dividing by √d keeps the scores in a sane range regardless of dimension. Softmax then turns the row of scaled scores into weights; multiplying that weight row by V blends the Value vectors accordingly.`),
          p(`Here is the whole pipeline on three tiny 2-dimensional tokens, so you can check every digit. Token 1 has <b>q</b>=(1,0), token 2 has <b>k</b>=(0,2), and so on — full vectors below. Take the query from token 1: <b>q₁</b>=(1,0).`),
          ol([
            `<b>Raw scores</b> (q₁ · k<sub>j</sub>): against k₁=(1,0) → 1×1+0×0=<b>1</b>. Against k₂=(0,2) → 1×0+0×2=<b>0</b>. Against k₃=(1,−1) → 1×1+0×(−1)=<b>1</b>.`,
            `<b>Scale</b> by √d = √2 ≈ 1.414: scores become 0.707, 0, 0.707.`,
            `<b>Softmax</b>: e<sup>0.707</sup>≈2.028, e<sup>0</sup>=1, e<sup>0.707</sup>≈2.028; sum ≈ 5.056. Weights ≈ <b>0.401, 0.198, 0.401</b> — token 1 splits its attention almost evenly between tokens 1 and 3, and gives token 2 less, because its key pointed a different way.`,
            `<b>Weighted sum of V</b>: with v₁=(1,0), v₂=(0,1), v₃=(2,0): output = 0.401·(1,0) + 0.198·(0,1) + 0.401·(2,0) = (0.401+0.802, 0.198) ≈ <b>(1.20, 0.20)</b>.`,
          ]),
          p(`That four-step recipe, run for every token against every other token, computed in one shot by matrix multiplication, is the entire mechanism. Nothing else in a transformer is conceptually harder than this; everything else is scale and repetition.`),
          callout('tryit', 'Try it: compute attention by hand', `The numbers above are the interactive's default settings — confirm them first. Then drag <b>K2.y</b> up toward 3: token 2's key now points more like q₁ does, so watch its softmax weight rise and steal share from tokens 1 and 3. Switch the "inspect query" dropdown to token 2 or 3 and work out by eye which key it should favour before checking the bars. Try dragging a Value vector instead of a Key: notice the attention <i>weights</i> (which token matters) don't move at all, only the final blended <i>output</i> does — Value only supplies content, never relevance.`),
          attentionByHand(),
        ),

        section('Seeing it on a real sentence',
          p(`Toy numbers make the arithmetic checkable, but the payoff of attention is linguistic: a model can resolve a pronoun to the noun it stands for, directly, by attending across the whole sentence. The classic example is a Winograd-style sentence: "The animal didn't cross the street because it was too tired." Every human instantly reads "it" as the animal, not the street — a tired street makes no sense. Real trained attention heads learn to make exactly this kind of link from data. The visualiser below fakes one such head with a small set of hand-written rules (a crude part-of-speech guess, a boost from pronouns toward the nearest preceding noun-like word, a boost from "the" toward the word right after it, and a mild preference for nearby words) so you can see the shape of the behaviour without needing billions of trained weights.`),
          callout('warning', 'Illustrative, not trained', `The weights drawn below come from a hand-written scoring rule in this page's JavaScript, tuned so the classic "it → animal" example works. It is not a real attention head, has not seen any training data, and will do less sensible things on sentences it wasn't designed for. Real attention weights emerge from gradient descent over billions of examples (chapters 2–3); this is a cartoon of the <i>shape</i> of that behaviour, not a measurement of it.`),
          callout('tryit', 'Try it: the attention visualiser', `<b>1.</b> Click the word "it" (token 8). Watch a fat arc reach all the way back to "animal" — by far its heaviest weight — while "street", the other candidate noun, gets much less. <b>2.</b> Click "the" (token 1 or 5): it should point almost entirely at the noun immediately following it. <b>3.</b> Push the <b>temperature</b> slider high: every arc fades toward equal thickness — attention becomes almost uniform. Push it low: one arc dominates completely, almost a hard choice. <b>4.</b> Toggle <b>causal mask</b> on and click an early word like "The": its arcs to later words vanish — it is now only allowed to look left, exactly like GPT generating text one token at a time. <b>5.</b> Edit the sentence entirely and see how (or whether) the rule still behaves sensibly — this is where "illustrative, not trained" becomes obvious.`),
          attentionVisualiser(),
        ),

        section('Multi-head attention: several lenses at once',
          p(`One attention computation learns one <i>kind</i> of relationship — say, pronoun-to-noun. A sentence needs many kinds at once: which adjective modifies which noun, which verb takes which object, which word two positions back rhymes with this one. So a transformer doesn't run attention once per layer; it runs it several times in parallel, called <em>heads</em>, each with its own separate learned Q, K, V weight matrices, each free to specialise in a different pattern. A model with dimension <i>d</i> = 768 and 12 heads might give each head just 64 dimensions to work with (768 ÷ 12); the heads run independently and simultaneously, and their outputs are concatenated back into a 768-dimensional vector and passed through one more learned matrix that mixes the heads' findings together. This is <em>multi-head attention</em>: not a bigger version of the mechanism above, just several independent copies of it, run side by side, each looking for something different.`),
        ),

        section('Order does not come for free',
          p(`Here is a strange fact about the Q·Kᵀ formula: it does not care about word order at all. Swap two tokens' positions in the input and the set of dot products between them is identical — attention is fundamentally a <i>set</i> operation, blind to sequence. That is a real problem, because "the dog bit the man" and "the man bit the dog" contain exactly the same set of words. An RNN got order for free, because it read left to right by construction; a transformer has to be told explicitly.`),
          p(`The fix is <em>positional encoding</em>: before the first layer, add a vector to each token's embedding that encodes its position in the sequence, so token 1's embedding and token 50's embedding differ even if the underlying word is identical. The original 2017 paper used a fixed pattern of sine and cosine waves at different frequencies — no learning required, and it generalises to sequences longer than any seen in training. Many models instead just learn a position vector per slot, the same way they learn a vector per word. Most 2023-and-later models (Llama, and most open frontier models) use a newer trick called <em>RoPE</em> (rotary position embedding), which rotates each token's Query and Key vectors by an angle proportional to their position, so that the dot product between any two tokens naturally reflects their <i>relative</i> distance rather than their absolute position — a detail worth knowing the name of, not worth deriving here.`),
        ),

        section('The full block, and the residual stream',
          p(`Attention alone only moves information between token positions; it never processes the content of one position on its own. So every attention step is paired with a small ordinary <em>multi-layer perceptron</em> (chapter 2's MLP, applied to each token independently, same weights reused at every position) that gives the model room to actually compute with what attention gathered. Wrap each of the two with a <em>residual connection</em> — add the sublayer's output back onto its input rather than replacing it — and a normalisation step that keeps activations in a stable range, and you get the standard recipe:`),
          ctx.code('x  ← x + Attention(LayerNorm(x))\nx  ← x + MLP(LayerNorm(x))          # repeat this whole pair, N times'),
          p(`Think of <b>x</b>, the running vector at each token position, as a highway running the full length of the network: the <em>residual stream</em>. Attention and the MLP are off-ramps — they read the highway, do their work, and merge their result back on, but the highway itself is never overwritten, only added to. This additive design is what lets networks stack dozens or hundreds of these blocks without gradients vanishing on the way back during training (the same failure that limited how deep pre-2015 networks could usefully go): a gradient can always flow straight back down the highway, with every sublayer offering an optional shortcut rather than a mandatory bottleneck.`),
          callout('tryit', 'Try it: follow the pulse', `Watch one full pass through the stack. Notice the pulse always returns to the same highway line between detours — that's the residual add. Slide <b>layers</b> up to 12 (GPT-2 small's actual depth) and notice the diagram doesn't get more complicated, just longer: the identical two-step block repeats. Pause it mid-detour and read the caption to see exactly which of the four repeating operations (Attention, Add & Norm, MLP, Add & Norm) is active.`),
          blockDiagram(),
        ),

        section('Encoder, decoder, and why GPT never looks ahead',
          p(`The original 2017 architecture was built for machine translation and had two halves. An <em>encoder</em> reads the entire source sentence at once, with unrestricted attention — every token can look at every other token, including ones later in the sentence — and produces a rich representation of what the sentence means. A <em>decoder</em> then generates the translated sentence one word at a time, attending both to its own previous output and, via a separate cross-attention step, to the encoder's representation of the source.`),
          p(`Two influential 2018 models each kept only one half. <b>BERT</b> is encoder-only: full bidirectional attention, trained by hiding random words and asking the model to fill them in, which builds excellent representations for understanding text — search, classification, the embeddings of chapter 6 — but BERT was never designed to generate long free-form text. <b>GPT</b> is decoder-only, and this is where <em>causal masking</em> comes in: during training, a decoder predicting word 50 must not be allowed to peek at the real word 50 sitting right there in the training example — that would make training trivial and useless, since at actual generation time word 50 doesn't exist yet. So every query position is masked to only attend to keys at its own position or earlier (exactly the toggle in the visualiser above). GPT won out as the dominant shape for general-purpose assistants because next-token prediction on plain, unlabelled text needs no parallel translation pairs or hand-labelled examples — it can train on virtually the whole internet — and one decoder can be prompted to translate, summarise, code, or chat, tasks that used to need separate specialised systems.`),
        ),

        section('Counting the parameters: GPT-2 small, step by step',
          p(`"124 million parameters" sounds like a fact you'd have to look up. It's actually arithmetic you can do yourself, and doing it once demystifies every model-size headline you'll ever read. GPT-2 small uses embedding dimension <i>d</i> = 768, 12 stacked blocks, a vocabulary of 50,257 tokens, and a context window of 1,024 positions.`),
          ul([
            `<b>Token embeddings:</b> one learned <i>d</i>-dimensional vector per vocabulary entry — vocab × d = 50,257 × 768 ≈ <b>38.6M</b>.`,
            `<b>Position embeddings:</b> one learned vector per context slot — n_ctx × d = 1,024 × 768 ≈ <b>0.8M</b>.`,
            `<b>Per block, attention:</b> four d×d matrices (the Q, K, V and output projections) — 4d² = 4 × 768² ≈ <b>2.36M</b>.`,
            `<b>Per block, MLP:</b> two matrices expanding to 4d and back — 2 × (d × 4d) = 8d² = 8 × 768² ≈ <b>4.72M</b>.`,
            `<b>Per block total:</b> ≈ 12d² ≈ <b>7.08M</b> (LayerNorm adds a few thousand more — negligible at this scale).`,
          ]),
          p(`Twelve blocks: 12 × 7.08M ≈ 85M. Add the embeddings: 85M + 38.6M + 0.8M ≈ <b>124M</b> — and there's the number. (GPT-2 ties its output layer to the input embedding matrix — reusing the same 38.6M numbers to turn the final vector back into a probability over the vocabulary — which is why there's no separate "unembedding" line above.) The general shape, params ≈ V·d + n_ctx·d + 12d²·L, is the same formula every larger model scales up: GPT-3 pushed d to 12,288 and L to 96 and landed near 175 billion; the d² term is why doubling width roughly quadruples a model's size while doubling depth only doubles it.`),
        ),

        section('Why this design won',
          p(`Two properties, working together, ended the argument. First, <i>parallel training</i>: because attention over a whole sequence is one matrix multiplication rather than a step-by-step loop, an entire training example — thousands of tokens — is processed in one shot on a GPU, and thousands of examples across thousands of GPUs run simultaneously. An RNN's inherent one-step-at-a-time-ness made it structurally unable to use hardware this way, no matter how many GPUs you bought. Second, <i>clean scaling</i>: transformers reliably keep getting better as you feed them more data and more parameters and more compute, in a smooth, predictable way (the subject of chapter 11's scaling laws) — there was no sign, through years of scaling, of the returns simply stopping. A parallelisable architecture that also scales predictably is precisely the combination that turns "bigger GPU budget" into "better model", which is the entire economic engine behind the last eight years of AI progress.`),
        ),

        callout('history', 'Three papers, eighteen months, 2017–2019', `Ashish Vaswani and seven co-authors at Google published <i>"Attention Is All You Need"</i> in June 2017, introducing the full encoder–decoder transformer for machine translation — the title was a claim, and it held up. In June 2018, OpenAI's GPT-1 (Radford et al.) showed that a decoder-only transformer, pretrained to just predict the next word in ordinary text and then lightly fine-tuned, beat specialised systems across several language tasks at once, with 117M parameters. In October 2018, Google's BERT (Devlin et al.) took the encoder half instead, trained bidirectionally by masking random words, and reset the state of the art across nearly every language-understanding benchmark overnight — for several years "fine-tune a BERT" was the default recipe for any serious NLP product. In February 2019, OpenAI's GPT-2 scaled the decoder-only recipe to 1.5 billion parameters and showed it could perform tasks — translation, summarisation, question answering — it was never explicitly trained to do, just by predicting text well enough; OpenAI initially withheld the full model, citing misuse concerns, an early preview of the safety debates this course returns to in Part V.`),

        callout('example', 'Where each half lives today', `Decoder-only transformers write: every chat assistant you've used (Claude, ChatGPT, Gemini) generates its reply one token at a time with a causal mask, exactly as described above. Encoder-style bidirectional attention still quietly powers search-query understanding, spam and content classifiers, and the embedding models behind semantic search from chapter 6. Translation, once the transformer's original purpose, is now usually done by the same decoder-only recipe, simply prompted to translate rather than routed through a separate encoder.`),

        section('Generating text: one token at a time, and the KV-cache',
          p(`Training sees a whole sentence at once and predicts every next-token in parallel, but using the model — <em>inference</em> — is different: the model only knows the tokens generated so far, so it produces its reply <em>autoregressively</em>, one token at a time. Predict the most likely next token, append it to the sequence, feed the whole (now one-token-longer) sequence back in, predict again. This loop, repeated hundreds or thousands of times, is why a chat response streams into view word by word instead of appearing all at once — that is not a UI affectation, it is the actual order of computation.`),
          p(`Done naively, this loop would recompute Key and Value vectors for every earlier token from scratch at every single step — wasteful, since a token's Key and Value never change once computed; only new tokens ever get added to the end. The standard fix is a <em>KV-cache</em>: store every token's Key and Value vectors the first time they're computed, and at each new step only compute Q, K, V for the one newest token, comparing its fresh Query against the whole cached history. This turns each generation step from redoing all the past work into one small increment, and it's the single biggest reason chat responses feel fast — it's also why very long conversations get slower and use more memory, since the cache keeps growing with every token you've exchanged.`),
        ),

        section('Why this matters for modern AI',
          p(`Every model you can currently name — Claude, the GPT family, Gemini, Llama, and everything else described as "an LLM" in 2026 — is this exact decoder-only recipe: token and position embeddings in, N copies of attention-then-MLP-with-residuals, a final projection back to vocabulary-sized probabilities, generated one token at a time behind a KV-cache. The differences between them are almost entirely differences of degree and detail covered later in this course — how many layers, how wide, what data, what fine-tuning — not differences in this underlying skeleton. If you understand everything on this page, you understand, mechanically, what happens between you pressing enter and a reply appearing, for every major model in existence today.`),
        ),

        ctx.quiz([
          { q: 'Why can attention be computed in parallel across an entire sequence, while an RNN cannot?', options: ['Attention uses less memory', 'Every pairwise score is one matrix multiplication that doesn\'t depend on any other position\'s result being computed first, unlike an RNN\'s step-by-step hidden state', 'Attention only works on short sequences', 'GPUs cannot run RNNs at all'], answer: 1, explain: 'Q·Kᵀ computes every (query, key) score at once. An RNN\'s hidden state at step t requires step t−1\'s output first, forcing strictly sequential computation no matter how much hardware is available.' },
          { q: 'In scores = QKᵀ/√d, what does dividing by √d prevent?', options: ['Negative numbers', 'Scores growing large as the vector dimension d grows, which would push softmax toward an extreme, hard-to-train spike', 'The need for a Value vector', 'Ties between tokens'], answer: 1, explain: 'A dot product\'s typical magnitude scales with the number of terms summed, i.e. with d. Without the √d correction, larger models (bigger d) would see wildly larger raw scores and softmax would saturate, weakening the training signal.' },
          { q: 'What does softmax guarantee about one query\'s attention weights over all the keys?', options: ['They are all equal', 'They are each non-negative and sum to exactly 1, forming a weighted average over the Value vectors', 'Exactly one weight is 1 and the rest are 0', 'They sum to the vector dimension d'], answer: 1, explain: 'Softmax exponentiates and normalises, so every weight is positive and the row sums to 1 — a genuine weighted average of every token\'s Value, never a hard, all-or-nothing pick.' },
          { q: 'Why does GPT need a causal mask during training, while BERT does not?', options: ['BERT is a smaller model', 'GPT predicts the next token and must never see it or later tokens while training, since at real generation time they don\'t exist yet; BERT instead fills in masked words using full bidirectional context', 'Causal masking makes training faster, nothing more', 'BERT does not use attention'], answer: 1, explain: 'A decoder trained to predict word 50 while being allowed to look at word 50 would trivially cheat during training and then fail completely at real generation, when future words genuinely don\'t exist yet. BERT\'s fill-in-the-blank objective has no such constraint.' },
          { q: 'GPT-2 small has d = 768 and 12 layers. Roughly which two terms dominate its ≈124M parameters?', options: ['The LayerNorm parameters and the biases', 'The token embedding matrix (vocab × d ≈ 38.6M) and the twelve blocks\' matrices (≈12d² per block × 12 ≈ 85M)', 'The positional encodings alone', 'The softmax function\'s own parameters'], answer: 1, explain: 'params ≈ V·d + n_ctx·d + 12d²·L. With V=50,257, d=768, n_ctx=1,024, L=12: embeddings ≈ 39.4M and the stacked blocks ≈ 85M, summing to ≈124M; LayerNorm\'s few thousand parameters are negligible by comparison.' },
        ]),

        section('Go deeper',
          ul([
            `<a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener">Vaswani et al. (2017), "Attention Is All You Need"</a> — the original paper, all six pages of mechanism that started this chapter.`,
            `<a href="https://jalammar.github.io/illustrated-transformer/" target="_blank" rel="noopener">Jay Alammar, "The Illustrated Transformer"</a> — the diagrams that made this architecture click for a generation of engineers.`,
            `<a href="https://www.youtube.com/watch?v=kCc8FmEb1nY" target="_blank" rel="noopener">Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out"</a> — writes this entire chapter's mechanism as running Python, live.`,
            `<a href="https://arxiv.org/abs/1810.04805" target="_blank" rel="noopener">Devlin et al. (2018), "BERT: Pre-training of Deep Bidirectional Transformers"</a> — the encoder-only half of this story.`,
            `<a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" target="_blank" rel="noopener">Radford et al. (2019), "Language Models are Unsupervised Multitask Learners"</a> — the GPT-2 paper, decoder-only at 1.5B parameters.`,
          ]),
        ),
      );
    },
  });
})();
