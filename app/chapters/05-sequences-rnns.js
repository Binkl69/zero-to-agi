/* Chapter 05 — Remembering: sequences, RNNs and LSTMs */
(function () {
  const DEFAULT_TEXT = 'the cat sat on the mat. the dog sat on the log. the cat saw the dog and the dog saw the cat. the sun was warm and the sky was blue. the cat and the dog ran to the park. in the park there was a big tree and a small pond. the dog swam in the pond and the cat sat under the tree. when the sun went down they walked home. the cat ate fish and the dog ate meat. then the cat slept on the mat and the dog slept on the log. the next day the sun was warm again and the sky was blue again. the cat and the dog went back to the park to play by the pond and under the tree.';

  /* ---------- character n-gram model (lookup-table "memory") ---------- */
  function trainNgram(text, maxK) {
    const tables = [];
    for (let k = 0; k <= maxK; k++) {
      const t = new Map();
      for (let i = k; i < text.length; i++) {
        const key = text.slice(i - k, i), nxt = text[i];
        let e = t.get(key);
        if (!e) { e = { counts: new Map(), total: 0 }; t.set(key, e); }
        e.counts.set(nxt, (e.counts.get(nxt) || 0) + 1);
        e.total++;
      }
      tables.push(t);
    }
    return tables;
  }
  function predict(tables, prefix, k) {
    for (let kk = Math.min(k, prefix.length); kk >= 0; kk--) {
      const key = kk === 0 ? '' : prefix.slice(-kk);
      const e = tables[kk].get(key);
      if (e) {
        const dist = [...e.counts.entries()].map(([c, n]) => ({ c, p: n / e.total })).sort((a, b) => b.p - a.p);
        return { dist, usedK: kk };
      }
    }
    return { dist: [], usedK: -1 };
  }
  function sample(dist, temp) {
    if (!dist.length) return ' ';
    const T = Math.max(0.05, temp);
    const logits = dist.map(d => Math.log(Math.max(d.p, 1e-9)) / T);
    const m = Math.max(...logits);
    const w = logits.map(l => Math.exp(l - m));
    const s = w.reduce((a, b) => a + b, 0);
    let r = Math.random() * s;
    for (let i = 0; i < w.length; i++) { r -= w[i]; if (r <= 0) return dist[i].c; }
    return dist[w.length - 1].c;
  }
  const showChar = (c) => c === ' ' ? '␣' : c === '\n' ? '↵' : c;

  /* deterministic pseudo-random embedding for a word (for the unrolled-RNN toy) */
  function wordVec(word, dim) {
    let hsh = 2166136261;
    for (let i = 0; i < word.length; i++) { hsh ^= word.charCodeAt(i); hsh = Math.imul(hsh, 16777619) >>> 0; }
    const v = [];
    for (let i = 0; i < dim; i++) { const x = Math.sin((hsh % 10007) * 0.137 + i * 2.399) * 43758.5453; v.push((x - Math.floor(x)) * 2 - 1); }
    return v;
  }

  ZTA.registerChapter({
    id: '05-sequences-rnns',
    num: 5,
    part: 'II',
    title: 'Remembering: sequences, RNNs and LSTMs',
    tagline: 'How a network reads one word at a time and keeps a running summary — and why that summary kept forgetting.',
    render(root, ctx) {
      const { h, p, section, sub, callout, ul, colors } = ctx;

      /* ================= HOOK ================= */
      root.append(
        p(`Your phone finishes your sentences. You type "see you at the" and it offers <b>airport</b>, <b>office</b>, <b>party</b>. Somewhere behind that keyboard a model is doing something the networks of chapters 2–4 could not do: it is <b>reading</b>. It takes in a stream of words of any length, in order, and produces a guess about what comes next.`),
        p(`The image classifier of chapter 4 has a fixed-size input: 224 × 224 pixels, always. Feed it a sentence and it has nowhere to put the words. Sentences are five words long or fifty. "The dog bit the man" and "The man bit the dog" contain identical words and mean opposite things. And sometimes the word that matters is far behind you: "The trophy didn't fit in the suitcase because <b>it</b> was too big" — to know what <em>it</em> is, you must remember something from ten words ago.`),
        p(`This chapter answers one question: <b>how do you build a network that reads?</b> The first answer, the <em>recurrent neural network</em>, ruled language AI from about 2014 to 2017, powered Google Translate and Siri, and then lost to the transformer for reasons that will make the next two chapters click into place.`),
      );

      /* ================= WHY SEQUENCES ARE DIFFERENT ================= */
      root.append(section('Why sequences are a different kind of problem',
        p(`Text, speech, stock prices, DNA, a piano melody, your heartbeat: these are all <em>sequences</em>. Three things make them awkward for the networks we have met so far.`),
        ul([
          `<b>Variable length.</b> A network has a fixed number of input neurons. A sequence does not have a fixed number of items.`,
          `<b>Order matters.</b> Shuffle the pixels of a photo and it is ruined; shuffle the words of a sentence and it is ruined <i>differently</i> — the same words in a different order carry a different meaning, so position is information.`,
          `<b>Long-range dependencies.</b> The meaning of a word can depend on something that happened far back: a name mentioned three paragraphs ago, an opening bracket 200 characters earlier, a gene sequence 10,000 bases upstream.`,
        ]),
        p(`The oldest trick is to ignore all three: just look at the last few items. A <em>bigram</em> model predicts the next character from the previous one; a <em>trigram</em> from the previous two. It is nothing more than a table of counts. It works surprisingly well, and its failure mode is the whole reason recurrent networks were invented. Try it first, so the RNN makes sense afterwards.`),
        callout('tryit', 'Try it: a lookup-table language model',
          `The model below is trained instantly on the text in the box (edit it!). Type a prefix such as <code class="inline">the c</code> and watch the top-5 next-character probabilities. Move the context slider from <b>1</b> to <b>4</b>: with more context the predictions get sharper, but you will see "context unseen, backed off" more often — the table has never seen that exact combination. Then press <b>Generate</b>. At low temperature the text is repetitive; at high temperature it is chaos. Notice that even the best setting cannot keep a thought going for longer than its context window.`),
        buildNgramDemo(ctx),
        p(`What did the table just do? It stored a <b>memory</b> of what usually follows what — but a dumb memory: an exact-match lookup. If the model has seen <code class="inline">the c</code> followed by <code class="inline">a</code> a hundred times, it predicts <code class="inline">a</code>. If your context is one character longer than anything in the table, it knows nothing. A 10-character context over a 27-letter alphabet has 27<sup>10</sup> ≈ 200 trillion possible keys. No table can hold that, and no text is long enough to fill it. The recurrent network's whole idea is to replace the table with a <b>compressed, learned memory</b>.`),
      ));

      /* ================= THE RNN ================= */
      root.append(section('The recurrent network: one cell, applied again and again',
        p(`Here is the trick. Instead of a different neuron for every position in the sequence, use <b>one small network</b> — call it the <em>cell</em> — and apply it to the items one at a time. The cell has two inputs: the current item, and a vector of numbers it produced last time. That vector is the <em>hidden state</em>: a running summary of everything read so far. Each step, the cell reads the next item, updates the summary, and (optionally) emits a prediction.`),
        ctx.code(`h_t = tanh( W_h · h_{t-1}  +  W_x · x_t  +  b )      # new summary from old summary + new input
y_t = W_y · h_t                                     # prediction from the summary`),
        p(`Three weight matrices, reused at every step. Because the same weights are used everywhere, the network can handle a sequence of any length — it just keeps going. And because the summary is passed forward, in principle the word from ten steps ago can still influence the prediction now.`),
        sub('A worked example with a one-number memory',
          p(`Shrink everything to a single number to see it move. Let the hidden state be one number, the update rule be <code class="inline">h_t = tanh(0.8·h_{t−1} + 0.5·x_t)</code>, and feed the inputs <code class="inline">x = 1, 1, 0, 0, 0</code>:`),
          ctx.table(['step', 'x_t', '0.8·h_{t−1} + 0.5·x_t', 'h_t = tanh(·)'], [
            ['1', '1', '0 + 0.5 = 0.50', '0.46'],
            ['2', '1', '0.37 + 0.5 = 0.87', '0.70'],
            ['3', '0', '0.56 + 0 = 0.56', '0.51'],
            ['4', '0', '0.41 + 0 = 0.41', '0.39'],
            ['5', '0', '0.31 + 0 = 0.31', '0.30'],
          ]),
          p(`The memory of the two early "1" inputs is still there at step 5, but it is fading — multiplied by 0.8 (then squashed) every step. That fading is not a bug in this example; it is the central character flaw of the whole architecture, and the animation below lets you dial it up and down.`),
        ),
        sub('Unrolling in time',
          p(`It helps to draw the loop opened out: copy the cell once per time step, left to right, with the hidden state flowing along the arrows between copies. This <em>unrolled</em> picture is exactly a deep feed-forward network — as deep as the sequence is long — except that every layer shares the same weights. That picture is also how it is trained: run the sequence forward, compute the loss at the end (or at every step), and backpropagate through all the copies. Because the gradient flows backwards along the time axis, the method is called <em>backpropagation through time</em>.`),
        ),
        callout('tryit', 'Try it: watch a hidden state absorb a sentence',
          `Press <b>▶ Play</b> — or drag straight across the picture to scrub word by word. Each word enters the cell in turn and the six coloured slots of the hidden state are rewritten by exactly the rule in the table above, <code class="inline">h = tanh(decay·h + 0.5·x)</code>. The bars underneath show how much of each earlier word survives in the summary. Drag <b>memory decay</b> down to <b>0.5</b> and step to the end: the word "trophy" is down to 0.1% of its original strength (0.5<sup>10</sup>) — the model has no way left to know what <i>it</i> refers to. Push the decay to <b>1.0</b>: now nothing fades and every bar is full — but nothing is forgotten <i>selectively</i> either. All twelve words are now stirred into the same six numbers with equal weight, and there is no way to pull any single one back out. Real RNNs learn a decay somewhere in between, and it is never quite right for every word.`),
        buildUnrollDemo(ctx),
      ));

      /* ================= VANISHING GRADIENTS ================= */
      root.append(section('The flaw: whisper down a line of 100 people',
        p(`Training an RNN means asking: "the prediction at step 50 was wrong — which weights at step 1 should change?" The gradient has to travel backwards through all 49 cells in between. At each cell it gets multiplied by the same factor (roughly, the weight matrix times the slope of tanh). Multiply a number by 0.9 fifty times and you get 0.005. Multiply it by 1.1 fifty times and you get 117.`),
        p(`This is the party game where a sentence is whispered down a line of people: by the end, nothing of the original survives. Signals fade to zero — the <em>vanishing gradient</em> — so the network simply cannot learn that something far back mattered. Or, with slightly larger weights, the signal blows up — the <em>exploding gradient</em> — and training goes to NaN. Exploding gradients have a cheap fix (clip them to a maximum size, which every training loop now does). Vanishing gradients do not; and long-range dependencies are precisely what language is made of.`),
        callout('tryit', 'Try it: watch a gradient vanish or explode',
          `The chart below is the unrolled RNN seen from the gradient's point of view. The loss sits at the far left, at the last time step; the gradient travels rightwards into the past, multiplied by the same factor at every hop. Set the multiplier to <b>0.9</b> and the sequence length to <b>50</b>: the gradient reaching the first time step is 5×10<sup>−3</sup> — on the log scale (each gridline is 100× the one below) you can watch it fall off a cliff. Set the multiplier to <b>1.1</b>: it explodes past 10<sup>2</sup>. Now try <b>1.0</b> exactly — a flat line. Only a multiplier within a hair of one carries the signal intact, and that is the single trick the LSTM is built around; the green dashed line shows what a forget gate held at 0.99 buys you.`),
        buildGradDemo(ctx),
      ));

      /* ================= LSTM / GRU ================= */
      root.append(section('LSTM: a notebook with an eraser',
        p(`In 1997 Sepp Hochreiter and Jürgen Schmidhuber published a cell designed so that the gradient could travel far without shrinking: the <em>Long Short-Term Memory</em>. The idea is to give the cell a separate memory lane — the <em>cell state</em> — that runs straight through time with almost nothing multiplied into it, like a conveyor belt. Information is added to it or removed from it only when little learned switches called <em>gates</em> say so.`),
        p(`Think of a notebook. A plain RNN rewrites the whole page every step, so old notes get smudged. An LSTM has three gates, each a small sigmoid layer that outputs numbers between 0 and 1 for every slot in the notebook:`),
        ul([
          `<b>Forget gate</b> — the eraser. "Given what I just read, which old notes are now irrelevant?" A 0 wipes a slot, a 1 keeps it.`,
          `<b>Input gate</b> — the pen. "Which parts of the new word are worth writing down, and where?"`,
          `<b>Output gate</b> — the reading glasses. "Which notes do I need to look at right now to make this prediction?"`,
        ]),
        ctx.code(`f = σ(W_f·[h_{t-1}, x_t])          # forget gate   (0..1 per slot)
i = σ(W_i·[h_{t-1}, x_t])          # input gate
o = σ(W_o·[h_{t-1}, x_t])          # output gate
c̃ = tanh(W_c·[h_{t-1}, x_t])       # candidate notes
c_t = f * c_{t-1} + i * c̃          # erase some old notes, write some new ones
h_t = o * tanh(c_t)                # read out what is needed now`),
        p(`Look at the line for <code class="inline">c_t</code>. When the forget gate is 1 and the input gate is 0, the cell state is copied unchanged — a multiplier of exactly one, the thing you just found in the gradient demo. So a gradient can flow back through hundreds of steps along the belt. The network <i>learns</i> when to open and close the gates, which means it learns what to remember. In practice LSTMs handle dependencies of a few hundred tokens; plain RNNs manage perhaps ten.`),
        p(`The <em>GRU</em> (gated recurrent unit, Cho et al. 2014) is the same idea with two gates instead of three and no separate cell state. It is slightly cheaper and often just as good. Both were the workhorses of language AI until 2017.`),
      ));

      /* ================= SEQ2SEQ & ATTENTION ================= */
      root.append(section('Translation, the bottleneck, and the birth of attention',
        p(`If an RNN can read a sentence into a summary vector, another RNN can write a sentence <i>out</i> of that vector. Sutskever, Vinyals and Le showed this in 2014: an <em>encoder</em> LSTM reads English, its final hidden state is handed to a <em>decoder</em> LSTM, and the decoder emits French one word at a time, feeding each word back in as input for the next. The <em>sequence-to-sequence</em> model.`),
        p(`It worked, and it had an obvious weakness. The entire meaning of a 40-word sentence had to squeeze through a single fixed-size vector — a few hundred numbers. Translation quality fell off sharply for long sentences. This is the <em>bottleneck problem</em>: a whole paragraph forced through a keyhole. The tell is a trick from that paper: feeding the source sentence in <b>backwards</b> made translations markedly better, purely because it moved the first English words closer in time to the first French words the decoder had to produce. When reversing your input helps, your memory is the problem.`),
        p(`The fix arrived in parallel, from Bahdanau, Cho and Bengio in September 2014, building on Cho's encoder–decoder from that June. Keep every encoder hidden state, one per input word, instead of only the last. Then, each time the decoder writes a word, let it <b>look back</b> at all of them and compute a weighted average, with weights that depend on what it is trying to write right now. Writing "chat"? Weight "cat" heavily. The decoder learns where to look. They called this mechanism <em>attention</em>, and the weights, when plotted, showed the model aligning source and target words almost like a human translator would.`),
        callout('key', 'The bridge to chapter 7',
          `Attention was invented as a patch for RNNs. Three years later a team at Google asked: if attention is doing the important work, why keep the recurrence at all? Their answer was the transformer — attention and nothing else. Chapter 7 is about that. Everything there builds on the idea you just met: <b>a weighted look-back over all previous positions, with learned weights</b>.`),
      ));

      /* ================= REAL WORLD ================= */
      root.append(section('Where you have used an RNN',
        callout('example', 'Google Translate, November 2016',
          `Google replaced its phrase-based statistical translator with GNMT, an 8-layer LSTM encoder–decoder with attention, trained on hundreds of millions of sentence pairs. Overnight, translation errors dropped by roughly 60% on the language pairs Google measured, and users noticed that translations suddenly read like sentences instead of word salad. It was the biggest single quality jump in the product's history.`),
        callout('example', 'Voice recognition and the keyboard',
          `From about 2015 to 2019, the speech recognisers in Siri, Google Voice and Alexa were built on LSTMs reading audio frames (a sequence of tiny sound snapshots, 100 per second) and emitting characters. The autocomplete on your phone keyboard was for years a small LSTM running on-device; so was the "Smart Reply" that suggests "Sounds good!" under an email. RNNs also read heartbeat traces, predicted the next note in a melody, and generated fake Shakespeare one character at a time — Andrej Karpathy's 2015 essay on that last trick (see "Go deeper") convinced a generation of engineers that sequence models could learn structure nobody programmed.`),
      ));

      /* ================= HISTORY ================= */
      root.append(callout('history', '1986 → 1997 → 2014 → 2017',
        `Recurrent networks were trained with backpropagation through time from the mid-1980s (Rumelhart, Hinton and Williams, 1986; Elman's "Finding structure in time", 1990). Hochreiter's 1991 thesis diagnosed the vanishing-gradient problem precisely, and in <b>1997</b> he and Schmidhuber published the LSTM to solve it. The idea sat largely unused for a decade — the data and GPUs were not there. Then, from 2013, LSTMs began winning: handwriting recognition, speech (Graves, 2013), and in <b>2014</b> the seq2seq paper from Sutskever, Vinyals and Le, followed by Bahdanau's attention. By 2016 LSTMs were inside Google Translate. In <b>2017</b> the transformer arrived and within two years RNNs had all but vanished from language research.`));

      /* ================= WHY RNNs LOST ================= */
      root.append(section('Why RNNs lost — and why the idea came back',
        p(`Two reasons, both fatal. First, <b>speed</b>. An RNN must process token 1 before token 2 before token 3: the computation is inherently sequential. A GPU has thousands of cores that want to work at the same time, and a sequential algorithm leaves almost all of them idle. Training on a billion words took weeks. Second, <b>memory still faded</b>. LSTM gates stretched the usable context from ten tokens to a few hundred, but the summary was still a fixed-size vector overwritten at every step; a fact from 2,000 tokens back was gone.`),
        p(`The transformer fixed both at once. Every token attends to every other token directly, so nothing has to be squeezed through a summary — the "memory" is just all the previous tokens, kept. And because there is no step-by-step dependency during training, every position in the sequence is processed in parallel, which is exactly what a GPU wants. Chapter 7 builds it.`),
        p(`The story has a twist. Transformers have their own cost: attention over <i>n</i> tokens costs <i>n</i><sup>2</sup> work, so a million-token context is expensive. From 2023 researchers went back to recurrence with new tools. <em>State-space models</em> such as Mamba (Gu and Dao, December 2023) keep a compressed running state like an RNN but with learned, input-dependent gates that can be trained in parallel; <em>linear-attention</em> variants and hybrids (RWKV, Griffin, Jamba, and the hybrid layers in several 2024–2025 production models) mix a few attention layers with many recurrent ones. The hidden-state idea did not die; it was rebuilt with everything learned in between.`),
      ));

      /* ================= WHY IT MATTERS ================= */
      root.append(section('Why this matters for the models you use today',
        p(`Three ideas from this chapter are load-bearing in every modern language model. <b>Autoregression</b> — predict the next item, feed it back in, repeat — is exactly how ChatGPT or Claude write an answer: one token at a time, each conditioned on everything before. <b>Attention</b> was born here as a fix for RNN forgetfulness and became the whole architecture. And the <b>vanishing-gradient</b> lesson — that signals must be able to flow through a network unchanged — reappears in the transformer's residual connections and in the "highway" design of chapter 7. When you hear that a new model "has a state-space backbone" or "uses a hybrid of attention and recurrence", you now know precisely what trade-off is being made: fixed-size, fast memory versus complete, expensive memory.`),
      ));

      /* ================= QUIZ ================= */
      root.append(ctx.quiz([
        { q: 'What is the hidden state of an RNN?', options: ['A list of all the words seen so far', 'A fixed-size vector that summarises everything read so far, updated at every step', 'The weights of the network', 'The next word the network predicts'], answer: 1, explain: 'The hidden state is the running summary. It is a fixed-size vector, which is exactly why it can forget — the whole past must fit into it.' },
        { q: 'Why do gradients vanish in a plain RNN over long sequences?', options: ['Because the learning rate is too small', 'Because the gradient is multiplied by a similar factor at every time step, and a number below 1 raised to a large power is tiny', 'Because tanh has no gradient', 'Because the sequence is stored in a lookup table'], answer: 1, explain: 'Backpropagation through time multiplies the gradient by roughly the same factor per step. 0.9 to the power 50 is 0.005; the signal from early tokens simply does not arrive.' },
        { q: 'What is the purpose of the forget gate in an LSTM?', options: ['To delete the training data', 'To decide, per memory slot, how much of the old cell state to keep (near 1) or erase (near 0)', 'To speed up the GPU', 'To predict the next word'], answer: 1, explain: 'The forget gate is the eraser in the notebook analogy. When it outputs 1 the cell state is copied unchanged, which lets gradients travel far back without shrinking.' },
        { q: 'What was the "bottleneck" in the 2014 seq2seq translator, and what fixed it?', options: ['The GPU was too slow; bigger GPUs fixed it', 'The whole source sentence had to fit in one fixed-size vector; attention fixed it by letting the decoder look back at every encoder state', 'The vocabulary was too small; subword tokens fixed it', 'The decoder was too short; deeper decoders fixed it'], answer: 1, explain: 'Bahdanau et al. (2014) let the decoder compute a weighted average over all encoder hidden states, with learned weights. That mechanism is attention, and it became the core of the transformer.' },
        { q: 'Which of these is NOT a reason transformers replaced RNNs?', options: ['RNNs process tokens one after another, which leaves GPU cores idle', 'RNN memory is a fixed-size summary that fades over long contexts', 'Transformers let every token look directly at every other token', 'Transformers use far fewer parameters than RNNs'], answer: 3, explain: 'Transformers are usually much larger than the RNNs they replaced. They won on parallel training and on direct access to the whole context, not on parameter thrift.' },
      ]));

      /* ================= GO DEEPER ================= */
      root.append(section('Go deeper', ul([
        `<a href="https://karpathy.github.io/2015/05/21/rnn-effectiveness/" target="_blank">The Unreasonable Effectiveness of Recurrent Neural Networks</a> — Andrej Karpathy's 2015 essay; character-level RNNs writing Shakespeare, Wikipedia and Linux source code.`,
        `<a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/" target="_blank">Understanding LSTM Networks</a> — Chris Olah's illustrated walk through the gates. Still the clearest explanation there is.`,
        `<a href="https://www.bioinf.jku.at/publications/older/2604.pdf" target="_blank">Long Short-Term Memory</a> — Hochreiter &amp; Schmidhuber, 1997. The original paper.`,
        `<a href="https://arxiv.org/abs/1409.3215" target="_blank">Sequence to Sequence Learning with Neural Networks</a> (Sutskever et al., 2014) and <a href="https://arxiv.org/abs/1409.0473" target="_blank">Neural Machine Translation by Jointly Learning to Align and Translate</a> (Bahdanau et al., 2014) — seq2seq and the invention of attention.`,
        `<a href="https://arxiv.org/abs/2312.00752" target="_blank">Mamba: Linear-Time Sequence Modeling with Selective State Spaces</a> — Gu &amp; Dao, 2023. Recurrence returns.`,
      ])));
    },
  });

  /* ---------- Interactive A: character predictor (count-based n-gram) ---------- */
  function buildNgramDemo(ctx) {
    const W = 720, H = 250;
    const [cv, g] = ctx.canvas(W, H);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    const MAXK = 4;
    const S = { text: DEFAULT_TEXT, k: 3, prefix: 'the c', temp: 0.6, tables: null, dist: [], usedK: -1, options: 0 };
    const ro = ctx.readout();

    function retrain() { S.tables = trainNgram(S.text && S.text.length ? S.text : ' ', MAXK); recompute(); }
    function recompute() {
      if (!S.tables) return;
      const pref = S.prefix || '';
      const r = predict(S.tables, pref, S.k);
      S.dist = r.dist.slice(0, 5);
      S.usedK = r.usedK;
      S.options = r.dist.length;
      const o = {};
      o['distinct contexts stored at n=' + S.k] = S.tables[S.k] ? S.tables[S.k].size : 0;
      o['lookup'] = S.usedK === S.k ? 'exact match'
        : pref.length < S.k ? 'prefix shorter than n → used n=' + S.usedK
          : 'never seen → backed off to n=' + S.usedK;
      o['possible next characters'] = S.options;
      ro.set(o);
      draw();
    }
    function backoffNote() {
      if (S.usedK < 0) return '';
      if (S.usedK === S.k) return '';
      if ((S.prefix || '').length < S.k) return ',  prefix shorter than n → using n = ' + S.usedK;
      return ',  this context was never seen → backed off to n = ' + S.usedK;
    }
    function draw() {
      g.clearRect(0, 0, W, H);
      g.lineWidth = 1; g.globalAlpha = 1;
      const px = 26, py = 30, pw = W - 52, ph = 158;
      g.fillStyle = '#0f1520'; g.fillRect(px, py, pw, ph);
      g.strokeStyle = C.line; g.strokeRect(px, py, pw, ph);
      const shownPrefix = (S.prefix || '').slice(-S.k) || '∅';
      g.font = FONT; g.textAlign = 'left'; g.textBaseline = 'alphabetic'; g.fillStyle = C.muted;
      g.fillText('P(next char | "' + shownPrefix.split('').map(showChar).join('') + '")  —  n = ' + S.k + backoffNote(), px, py - 10);
      if (!S.dist.length) {
        g.fillStyle = C.muted; g.textAlign = 'center';
        g.fillText('type some training text below to begin', px + pw / 2, py + ph / 2);
        return;
      }
      const n = S.dist.length, maxP = Math.max(1e-6, ...S.dist.map(d => d.p));
      const slot = pw / n;
      S.dist.forEach((d, i) => {
        const cx = px + slot * (i + 0.5);
        const bh = Math.max(2, (d.p / maxP) * (ph - 48));
        const bw = Math.min(64, slot * 0.55);
        const by = py + ph - 26 - bh;
        g.fillStyle = i === 0 ? C.green : C.accent;
        g.fillRect(cx - bw / 2, by, bw, bh);
        g.fillStyle = C.text; g.font = MONO; g.textAlign = 'center';
        g.fillText(showChar(d.c), cx, py + ph - 8);
        g.fillText((d.p * 100).toFixed(1) + '%', cx, by - 6);
      });
    }

    const taWrap = ctx.textarea({ label: 'training text — edit me, the model retrains instantly', value: S.text, onChange: (v) => { S.text = v; retrain(); } });
    const prefIn = ctx.h('input', { type: 'text', value: S.prefix });
    prefIn.addEventListener('input', (e) => { S.prefix = (e.target && e.target.value != null) ? e.target.value : prefIn.value; recompute(); });
    const prefWrap = ctx.h('div', { class: 'control' }, ctx.h('label', {}, 'prefix typed so far'), prefIn);
    const kSl = ctx.slider({ label: 'context length n', min: 1, max: MAXK, step: 1, value: 3, onChange: (v) => { S.k = ctx.clamp(Math.round(v), 1, MAXK); recompute(); } });
    const tempSl = ctx.slider({ label: 'generation temperature', min: 0.1, max: 2, step: 0.05, value: 0.6, fmt: (v) => (+v).toFixed(2), onChange: (v) => { S.temp = ctx.clamp(v, 0.1, 2); } });
    const PLACEHOLDER = 'press "Generate 100 chars" …';
    const genOut = ctx.h('pre', { class: 'code' }, PLACEHOLDER);
    const genBtn = ctx.button('Generate 100 chars', () => {
      if (!S.tables) return;
      let seed = S.prefix || ' ', out = '';
      for (let i = 0; i < 100; i++) {
        const r = predict(S.tables, seed, S.k);
        const c = sample(r.dist, S.temp);
        out += c;
        seed = (seed + c).slice(-MAXK);          // only the last MAXK chars can ever be looked up
      }
      genOut.textContent = ((S.prefix || '') + out).split('').map(showChar).join('');
    }, 'primary');
    const resetBtn = ctx.button('↺ Reset', () => {
      S.text = DEFAULT_TEXT; S.k = 3; S.prefix = 'the c'; S.temp = 0.6;
      taWrap.value = DEFAULT_TEXT; prefIn.value = 'the c'; kSl.value = 3; tempSl.value = 0.6;
      genOut.textContent = PLACEHOLDER;
      retrain();
    });

    retrain();
    const body = ctx.h('div', {}, cv, genOut);
    return ctx.figure(body,
      `Bars are the model's top-5 guesses for the very next character, recomputed live as you type or edit the text — pure counting, no learning involved. Watch the readout: raising <b>n</b> multiplies the number of stored contexts and makes exact matches rarer, which is the combinatorial wall that kills lookup tables. An <em>RNN</em> throws the whole table away and replaces it with a small vector of numbers — a compressed, <b>learned</b> memory that plays the same role without storing every combination it has ever seen.`,
      [taWrap, prefWrap, kSl, tempSl, genBtn, resetBtn], ro);
  }

  /* ---------- Interactive B: unrolled RNN absorbing a sentence ---------- */
  function buildUnrollDemo(ctx) {
    const W = 720, H = 300;
    const [cv, g] = ctx.canvas(W, H);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    const DIM = 6;
    const W_IN = 0.5;                     // input weight — the same 0.5 as the worked example above
    const tokens = `the trophy didn't fit in the suitcase because it was too big`.split(' ');
    const vecs = tokens.map((w) => wordVec(w, DIM));
    const gap = Math.min(70, (W - 40) / tokens.length);
    const tx0 = 20;
    const f2 = (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);   // avoid printing "-0.00"

    const S = { t: -1, decay: 0.85, playing: false, h: new Array(DIM).fill(0), acc: 0, dirty: true };

    /* Replay the sentence from scratch up to step `upto`. Replaying (rather than mutating in place)
       keeps the state exactly consistent after the decay slider moves or the reader scrubs. */
    function replay(upto) {
      S.h = new Array(DIM).fill(0);
      for (let s = 0; s <= upto && s < tokens.length; s++) {
        const x = vecs[s];
        for (let d = 0; d < DIM; d++) S.h[d] = Math.tanh(S.decay * S.h[d] + W_IN * x[d]);
      }
      for (let d = 0; d < DIM; d++) if (!isFinite(S.h[d])) S.h[d] = 0;   // tanh cannot blow up, but never ship a NaN
      S.dirty = true;
    }
    function setStep(t) { S.t = ctx.clamp(Math.round(t), -1, tokens.length - 1); replay(S.t); }
    function pause() { S.playing = false; playBtn.textContent = '▶ Play'; }
    function reset() { pause(); S.acc = 0; setStep(-1); }
    function step() {
      if (S.t >= tokens.length - 1) { pause(); return; }
      setStep(S.t + 1);
    }
    /* fraction of word i's signal still present in the summary at step t: decay^(steps ago) */
    function weightOf(i) {
      if (S.t < 0 || i > S.t) return 0;
      return Math.pow(S.decay, S.t - i);
    }
    /* shorten a label until it fits inside maxW pixels (g.font must already be set) */
    function fitLabel(text, maxW) {
      if (g.measureText(text).width <= maxW) return text;
      let s = String(text);
      while (s.length > 1 && g.measureText(s + '…').width > maxW) s = s.slice(0, -1);
      return s + '…';
    }
    function draw() {
      g.clearRect(0, 0, W, H);
      g.lineWidth = 1; g.globalAlpha = 1;
      // token strip
      const ty = 26;
      g.font = MONO; g.textAlign = 'center';
      tokens.forEach((w, i) => {
        const cx = tx0 + gap * (i + 0.5);
        const active = i === S.t, done = i < S.t;
        g.fillStyle = active ? C.warn : done ? '#243044' : '#171f30';
        g.fillRect(cx - gap * 0.46, ty - 14, gap * 0.9, 24);
        g.fillStyle = active ? '#0a0e16' : done ? C.text : '#7b8aa8';
        g.fillText(fitLabel(w, gap * 0.84), cx, ty + 3);
      });
      // hidden state
      const hy = 92, cw = 50, cellGap = 56, hx0 = W / 2 - ((DIM - 1) * cellGap + cw) / 2;
      g.font = FONT; g.textAlign = 'center'; g.fillStyle = C.muted;
      g.fillText('hidden state h — ' + DIM + ' numbers, rewritten every step: h = tanh(decay·h + 0.5·x)', W / 2, hy - 14);
      for (let d = 0; d < DIM; d++) {
        const cx = hx0 + d * cellGap;
        g.fillStyle = '#111827'; g.fillRect(cx, hy, cw, 44);
        g.fillStyle = ctx.heat(S.h[d]); g.fillRect(cx, hy, cw, 44);
        g.strokeStyle = C.line; g.strokeRect(cx, hy, cw, 44);
        g.fillStyle = C.text; g.font = MONO; g.fillText(f2(S.h[d]), cx + cw / 2, hy + 26);
      }
      // fading trail
      const by0 = 176, bh = 74, barMax = bh - 16;
      g.font = FONT; g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('how much of each word is still in the summary  =  decay ^ (steps ago)', tx0, by0 - 12);
      g.strokeStyle = C.line;
      g.beginPath(); g.moveTo(tx0, by0 + bh + 0.5); g.lineTo(W - 20, by0 + bh + 0.5); g.stroke();
      tokens.forEach((w, i) => {
        const frac = weightOf(i);
        const cx = tx0 + gap * (i + 0.5);
        const barH = Math.max(1, frac * barMax);
        g.globalAlpha = i <= S.t ? 0.35 + 0.65 * frac : 0.12;
        g.fillStyle = i === S.t ? C.warn : C.accent;
        g.fillRect(cx - gap * 0.28, by0 + bh - barH, gap * 0.56, barH);
        g.globalAlpha = 1;
        g.font = MONO; g.textAlign = 'center';
        if (i <= S.t) { g.fillStyle = C.text; g.fillText(Math.round(frac * 100) + '%', cx, by0 + bh - barH - 4); }
        g.fillStyle = i <= S.t ? C.muted : '#7b8aa8';
        g.fillText(fitLabel(w, gap * 0.92), cx, by0 + bh + 14);
      });
      g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
      const status = S.t < 0 ? 'press ▶ Play — or drag anywhere on the canvas to scrub through the sentence'
        : S.t < tokens.length - 1 ? 'reading word ' + (S.t + 1) + ' of ' + tokens.length + ' — "' + tokens[S.t] + '"'
          : 'sentence finished — this six-number summary is all the model has left to answer "what does it refer to?"';
      g.fillText(status, tx0, H - 10);
    }

    const ro = ctx.readout();
    function updateRO() {
      let ss = 0; for (let d = 0; d < DIM; d++) ss += S.h[d] * S.h[d];
      ro.set({
        step: S.t < 0 ? '—' : (S.t + 1) + ' / ' + tokens.length,
        'word read': S.t < 0 ? '—' : '“' + tokens[S.t] + '”',
        '‖h‖': Math.sqrt(ss).toFixed(2),
        'word 1 (“the”) still weighs': S.t < 0 ? '—' : (weightOf(0) * 100).toFixed(1) + '%',
      });
    }

    const decaySl = ctx.slider({
      label: 'memory decay', min: 0.5, max: 1.0, step: 0.01, value: 0.85, fmt: (v) => (+v).toFixed(2),
      onChange: (v) => { S.decay = ctx.clamp(v, 0.5, 1); replay(S.t); },
    });
    const playBtn = ctx.button('▶ Play', () => {
      if (!S.playing && S.t >= tokens.length - 1) setStep(-1);      // pressing play at the end restarts
      S.playing = !S.playing; S.acc = 0;
      playBtn.textContent = S.playing ? '⏸ Pause' : '▶ Play';
    }, 'primary');
    const stepBtn = ctx.button('Step ▸', () => { pause(); if (S.t >= tokens.length - 1) setStep(-1); else step(); });
    const resetBtn = ctx.button('↺ Reset', reset);

    /* pointer scrubbing — works with mouse, pen and touch */
    let scrubbing = false;
    const scrubTo = (ev) => {
      const pos = cv.pos(ev);
      setStep(ctx.clamp(Math.floor((pos.x - tx0) / gap), 0, tokens.length - 1));
    };
    cv.addEventListener('pointerdown', (ev) => {
      if (ev.preventDefault) ev.preventDefault();
      scrubbing = true; pause();
      try { cv.setPointerCapture(ev.pointerId); } catch (e) {}
      scrubTo(ev);
    });
    cv.addEventListener('pointermove', (ev) => { if (scrubbing) scrubTo(ev); });
    const endScrub = () => { scrubbing = false; };
    cv.addEventListener('pointerup', endScrub); cv.addEventListener('pointercancel', endScrub);
    // safety net: if pointer capture failed and the release happens off-canvas, never stay stuck mid-scrub
    window.addEventListener('pointerup', endScrub); window.addEventListener('pointercancel', endScrub);
    ctx.onCleanup(() => { window.removeEventListener('pointerup', endScrub); window.removeEventListener('pointercancel', endScrub); });
    cv.style.touchAction = 'none'; cv.style.cursor = 'ew-resize';

    ctx.loop((dt) => {
      if (S.playing) { S.acc += dt; if (S.acc > 0.6) { S.acc = 0; step(); } }
      if (S.dirty) { draw(); updateRO(); S.dirty = false; }
    });
    reset();
    return ctx.figure(cv,
      `The worked example above, run with ${DIM} numbers instead of one: every word is squashed into the same ${DIM}-slot summary by <code class="inline">h = tanh(decay·h + 0.5·x)</code> (colour: blue = negative, red = positive, near-black = empty). The bars show how much of each word's signal survives — <b>decay<sup>steps ago</sup></b>. At decay 0.5 a word is 3% of itself five steps later; at 1.0 nothing fades at all, but then every word is stirred in with equal weight and none of them can be recovered separately. <b>Drag across the canvas</b> to scrub back and forth.`,
      [decaySl, playBtn, stepBtn, resetBtn], ro);
  }

  /* ---------- Interactive C: vanishing / exploding gradient chain ---------- */
  function buildGradDemo(ctx) {
    const W = 720, H = 300;
    const [cv, g] = ctx.canvas(W, H);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    const HIGHWAY = 0.99;                       // an LSTM forget gate sitting almost fully open
    const S = { n: 20, m: 0.9, highway: true };
    const ro = ctx.readout();

    /* magnitude of the gradient after i backward hops through the chain */
    const magAt = (i, mult) => Math.pow(mult, i);

    function draw() {
      g.clearRect(0, 0, W, H);
      g.lineWidth = 1; g.globalAlpha = 1; g.setLineDash([]);
      const px = 60, py = 40, pw = W - 90, ph = 196;
      g.fillStyle = '#0f1520'; g.fillRect(px, py, pw, ph);
      g.strokeStyle = C.line; g.strokeRect(px, py, pw, ph);
      const logMin = -8, logMax = 8;
      const yOf = (mag) => {
        const lm = ctx.clamp(Math.log10(Math.max(1e-300, mag)), logMin, logMax);
        return py + ph - (lm - logMin) / (logMax - logMin) * ph;
      };
      g.font = MONO; g.textAlign = 'right'; g.textBaseline = 'alphabetic';
      for (let e = logMin; e <= logMax; e += 2) {
        const y = yOf(Math.pow(10, e));
        g.strokeStyle = e === 0 ? C.text : C.line; g.globalAlpha = e === 0 ? 0.55 : 0.35;
        g.beginPath(); g.moveTo(px, y); g.lineTo(px + pw, y); g.stroke(); g.globalAlpha = 1;
        g.fillStyle = C.muted; g.fillText('1e' + e, px - 8, y + 4);
      }
      // N is the number of backward hops, so the chain has N+1 points: hop 0 (at the loss) … hop N.
      const n = Math.max(1, Math.round(S.n));
      const xAt = (i) => px + pw * i / n;

      if (S.highway) {                           // reference curve: the LSTM's cell-state conveyor belt
        g.beginPath();
        for (let i = 0; i <= n; i++) { const x = xAt(i), y = yOf(magAt(i, HIGHWAY)); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); }
        g.strokeStyle = C.green; g.lineWidth = 2; g.setLineDash([5, 4]); g.stroke(); g.setLineDash([]);
      }
      g.beginPath();
      for (let i = 0; i <= n; i++) { const x = xAt(i), y = yOf(magAt(i, S.m)); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); }
      g.strokeStyle = C.accent; g.lineWidth = 2; g.stroke();
      const dotStride = Math.max(1, Math.ceil((n + 1) / 40));   // never more than ~40 dots, whatever N is
      for (let i = 0; i <= n; i += dotStride) {
        const mag = magAt(i, S.m), x = xAt(i), y = yOf(mag);
        g.beginPath(); g.arc(x, y, 3.5, 0, Math.PI * 2);
        g.fillStyle = mag > 1e3 ? C.danger : mag < 1e-3 ? C.warn : C.green; g.fill();
      }
      g.lineWidth = 1;

      g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
      g.fillText('← the loss, at the last time step', px, py + ph + 20);
      g.textAlign = 'right';
      g.fillText('the first time step, ' + n + ' hops back →', px + pw, py + ph + 20);
      if (S.highway) {
        g.textAlign = 'left'; g.fillStyle = C.green; g.font = MONO;
        g.fillText('– – –  LSTM cell state, forget gate ≈ ' + HIGHWAY, px + 10, py + 16);
      }
      const final = magAt(n, S.m);
      g.textAlign = 'center'; g.font = 'bold 13px Inter, system-ui, sans-serif';
      g.fillStyle = final < 1e-3 ? C.warn : final > 1e3 ? C.danger : C.green;
      const tag = final < 1e-6 ? '  (vanished)' : final > 1e6 ? '  (exploded)' : '';
      g.fillText('gradient reaching the first time step ≈ ' + final.toExponential(2) + tag, px + pw / 2, py - 14);
    }
    function refresh() {
      draw();
      const final = magAt(Math.max(1, Math.round(S.n)), S.m);
      ro.set({
        'multiplier': S.m.toFixed(2),
        'steps back': Math.round(S.n),
        'multiplier ^ steps': final.toExponential(2),
        'verdict': final < 1e-3 ? 'vanished — step 1 cannot learn' : final > 1e3 ? 'exploded — clip the gradient' : 'signal survives',
      });
    }
    const nSl = ctx.slider({ label: 'sequence length N (time steps back)', min: 5, max: 50, step: 1, value: 20, onChange: (v) => { S.n = ctx.clamp(Math.round(v), 1, 50); refresh(); } });
    const mSl = ctx.slider({ label: 'per-step gradient multiplier', min: 0.5, max: 1.5, step: 0.01, value: 0.9, fmt: (v) => (+v).toFixed(2), onChange: (v) => { S.m = ctx.clamp(v, 0.5, 1.5); refresh(); } });
    const hwBtn = ctx.button('LSTM highway: on', () => {
      S.highway = !S.highway;
      hwBtn.textContent = 'LSTM highway: ' + (S.highway ? 'on' : 'off');
      draw();
    });
    const resetBtn = ctx.button('↺ Reset', () => {
      S.n = 20; S.m = 0.9; S.highway = true;
      nSl.value = 20; mSl.value = 0.9; hwBtn.textContent = 'LSTM highway: on';
      refresh();
    });
    refresh();
    return ctx.figure(cv,
      `The gradient is the same multiplier raised to a higher and higher power as it travels back in time (note the log scale: each gridline is 100× the one below). Below 1.0 it collapses toward zero — the <em>vanishing gradient</em>, so the first words of the sentence receive no correction at all; above 1.0 it rockets toward infinity — the <em>exploding gradient</em>, which is why every training loop clips it. The green dashed line is the LSTM's escape route: a forget gate held at 0.99 still delivers 60% of the signal after 50 steps.`,
      [nSl, mSl, hwBtn, resetBtn], ro);
  }
})();
