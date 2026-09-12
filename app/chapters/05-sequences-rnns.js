/* Zero → AGI · Chapter 05 · Remembering: sequences, RNNs and LSTMs
   DESIGN RULE: the reader trains a language model in the first ten seconds and hits its wall
   before any theory arrives. Every paragraph explains something they already did.
   Interactives, in order: character n-gram predictor (the lookup-table memory and its ceiling);
   order-and-length demo (identical bag-of-words for opposite meanings, plus fixed-slot
   truncation); unrolled RNN hidden state absorbing a sentence; gradient vanish/explode on a log
   scale; LSTM conveyor belt with forget/input gates; seq2seq bottleneck vs attention alignment. */
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
    tagline: 'Build a language model in two seconds, watch it hit a wall, then meet the running summary that got past it — and the forgetting that finally killed it.',
    render(root, ctx) {
      const { h, p, section, sub, callout, ul, colors } = ctx;

      /* ---------- open with the reader building a language model ---------- */
      root.append(
        callout('tryit', '🖐 Do this first — build a language model in two seconds, then break it',
          `The model below is trained <b>instantly</b> on the text in the box. It is nothing but a table of counts.<br>
           <b>1.</b> The box starts on the prefix <code class="inline">the </code> with one character of context. Read the guesses: seventeen candidates, and the best of them only 31.8%.<br>
           <b>2.</b> Move the <b>context</b> slider from 1 up to 4 and watch the candidate list shrink from seventeen to eight. More context, fewer things that can come next.<br>
           <b>2b.</b> Now type <code class="inline">og sa</code>. At context 1 there are twelve candidates and the best is 31%; at context 4 there are two and the best is 50%. That is what a longer memory buys.<br>
           <b>2c.</b> Now type <code class="inline">the qu</code>. Nothing in the text ever follows it, so the readout says <b>"never seen → backed off"</b> and the model falls all the way back to a single character of context. That is the wall.<br>
           <b>3.</b> Put <b>context</b> back to 1 and press <b>Generate</b> at temperature 0.1, then at 2.0: cold repeats <code class="inline">the the the</code> forever, hot is unreadable.<br>
           <b>3b.</b> Now set <b>context</b> to 4 and generate at 2.0 again. It stays in real words however hot you make it — not because it understands them, but because at four characters of context roughly two steps in three have only <i>one</i> possible next character. There is nothing left for temperature to choose between. That is memorisation wearing fluency as a costume.<br>
           <b>4.</b> Now find the ceiling: <b>at no setting can it hold a thought longer than its context window.</b> Edit the text box and watch it relearn instantly.`),
        buildNgramDemo(ctx),
        p(`You just built the oldest language model there is, and hit its wall. Everything in this chapter, and arguably everything in the rest of this course, is an attempt to get past that wall.`),
      );

      root.append(section('What the table did, and what it cannot do',
        p(`That table stored a <b>memory</b> of what usually follows what. But it is a dumb memory: an exact-match lookup. Seen <code class="inline">the c</code> followed by <code class="inline">a</code> a hundred times? Predict <code class="inline">a</code>. Given a context one character longer than anything in the table? It knows nothing at all.`),
        p(`The arithmetic is brutal. A 10-character context over a 27-letter alphabet has 27<sup>10</sup> ≈ <b>200 trillion</b> possible keys. No table can hold that, and no quantity of text could fill it. Worse, the table has no idea that <code class="inline">the ca</code> and <code class="inline">a ca</code> have anything in common — to a lookup key they are simply different strings.`),
        callout('key', '🔑 The one idea this chapter is chasing',
          `Replace the lookup table with a <b>compressed, learned memory</b>: a handful of numbers, updated as you read, that summarise everything so far.
           Not "which exact string did I see", but "what kind of thing am I in the middle of".`),
      ));

      root.append(section('Why chapter 4\'s network cannot just do this',
        p(`The obvious move is to reach for the tools you already have. It fails in three specific ways, and it is worth seeing them apart before meeting the fix.`),
        callout('tryit', '🖐 Try this — two sentences that mean opposite things',
          `<b>1.</b> Look at the two default sentences. They contain exactly the same words. Their word-count vectors are <b>identical</b>, so any model that ignores order must give both the same answer.<br>
           <b>2.</b> Edit either sentence and watch the verdict flip the moment the word <i>counts</i> differ — not the moment the meaning does.<br>
           <b>3.</b> Drag <b>input slots</b> down to 4 and watch <b>both</b> sentences lose their last word — the network never gets to see who did what to whom. Drag it to 20 and watch most of the input sit empty, wasted.`),
        buildOrderDemo(ctx),
        p(`Three problems, then. <b>Order carries meaning</b>, and a fixed input that is just a pile of word counts throws it away. <b>Length varies</b> — sentences are five words or fifty — while the classifier of chapter 4 demands exactly 224×224 pixels every time, so you must truncate or pad and waste.`),
        p(`And <b>the word that matters can be far behind you</b>. "The trophy didn't fit in the suitcase because <b>it</b> was too big" — to know what <em>it</em> refers to, you must reach back seven words. A fixed window either misses it or has to be enormous.`),
      ));

      root.append(section('The recurrent network: one cell, applied over and over',
        p(`Here is the trick. Instead of a different neuron for every position, use <b>one small network</b> — the <em>cell</em> — and apply it to the items one at a time.`),
        p(`The cell takes two inputs: the current item, and the vector of numbers it produced last time. That vector is the <em>hidden state</em>, a running summary of everything read so far. Each step it reads the next item, updates the summary, and optionally emits a prediction.`),
        callout('key', '🔑 Three weight matrices, reused at every step',
          `<code class="inline">h_t = tanh(W<sub>hh</sub>·h<sub>t−1</sub> + W<sub>xh</sub>·x<sub>t</sub> + b)</code>, then optionally <code class="inline">y_t = W<sub>hy</sub>·h<sub>t</sub></code>.<br>
           Because the <i>same</i> weights are used at every step, the network handles a sequence of any length — it just keeps going. This is weight sharing again, exactly as in chapter 4, but shared across <b>time</b> instead of across space.`),
        p(`Shrink it to one number to watch it move. Let the hidden state be a single value with the rule <code class="inline">h_t = tanh(0.8·h_{t−1} + 0.5·x_t)</code>, fed the inputs 1, 1, 0, 0, 0. After the two "1"s the state rises; then it decays — multiplied by 0.8 and squashed, every step.`),
        callout('tryit', '🖐 Try this: watch a hidden state absorb a sentence',
          `Press <b>▶ Play</b>, or drag straight across the picture to scrub word by word. Each word enters the cell in turn and the six coloured slots are rewritten by exactly that rule. The bars underneath show how much of each earlier word survives.<br>
           <b>1.</b> Drag <b>memory decay</b> to <b>0.5</b> and step to the end. The word "trophy" is down to 0.1% of its original strength (0.5<sup>10</sup>) — the model has no way left to know what <i>it</i> refers to.<br>
           <b>2.</b> Push decay to <b>1.0</b>. Now the decay term stops shrinking anything, so every bar reads 100% — but nothing is forgotten <i>selectively</i> either, and all twelve words are stirred into the same six numbers with equal weight, and none can be pulled back out.<br>
           <b>3.</b> Real RNNs learn a decay somewhere in between. <b>It is never right for every word.</b>`),
        buildUnrollDemo(ctx),
        p(`That fading is not an artefact of this toy. It is the central character flaw of the whole architecture, and the next demo shows exactly why it is so hard to escape.`),
        p(`It helps to draw the loop opened out: copy the cell once per time step, left to right, hidden state flowing along the arrows between copies. That <em>unrolled</em> picture is a deep feed-forward network — as deep as the sequence is long — except every layer shares the same weights. It is also how training works: run forward, compute the loss, and backpropagate through all the copies. Because the gradient travels backwards along the time axis, it is called <em>backpropagation through time</em>.`),
      ));

      root.append(section('The flaw: a whisper down a line of fifty people',
        p(`Training asks: "the prediction at step 50 was wrong — which weights at step 1 should change?" The gradient must travel backwards through all 49 cells in between, and at each one it is multiplied by roughly the same factor.`),
        p(`Multiply a number by 0.9 fifty times and you get 0.005. Multiply it by 1.1 fifty times and you get 117. There is no comfortable middle.`),
        callout('tryit', '🖐 Try this: watch a gradient vanish, then explode',
          `The chart is the unrolled network seen from the gradient's point of view: the loss sits at the far left, and the gradient travels rightwards into the past. Note the log scale — each gridline is 100× the one below.<br>
           <b>1.</b> Set the multiplier to <b>0.9</b> with length <b>50</b>. The gradient reaching the first step is 5×10<sup>−3</sup>. Watch it fall off a cliff.<br>
           <b>2.</b> Set it to <b>1.1</b>. It rockets past 10<sup>2</sup> — training goes to NaN.<br>
           <b>3.</b> Now try <b>1.0</b> exactly. A flat line.<br>
           <b>Only a multiplier within a hair of one carries the signal intact.</b> The green dashed line shows what a forget gate held at 0.99 buys you.`),
        buildGradDemo(ctx),
        p(`This is the party game where a sentence is whispered down a line of people: by the end nothing of the original survives. Signals fade to zero — the <em>vanishing gradient</em> — so the network cannot learn that something far back mattered.`),
        p(`Or, with slightly larger weights, the signal blows up — the <em>exploding gradient</em> — and training dies. Exploding has a cheap fix: clip the gradient to a maximum size, which every training loop now does. Vanishing has no such fix, and long-range dependencies are precisely what language is made of.`),
      ));

      root.append(section('The LSTM: a conveyor belt through time',
        p(`In 1997 Sepp Hochreiter and Jürgen Schmidhuber published a cell designed so the gradient could travel far without shrinking: the <em>Long Short-Term Memory</em>.`),
        p(`The idea is to give the cell a separate memory lane — the <em>cell state</em> — that runs straight through time with almost nothing multiplied into it, like a conveyor belt. Things are added to it or removed from it only when little learned switches called <em>gates</em> say so.`),
        p(`The 1997 cell had two gates — an <b>input gate</b> deciding what to write and an <b>output gate</b> deciding what to reveal — and a belt that could never be cleared: memories were added but never removed, so a cell running for a long time simply saturated. <b>Gers, Schmidhuber and Cummins</b> added the missing switch in a 2000 paper called <i>Learning to Forget</i>, and every LSTM since has had three gates.`),
        p(`Think of a notebook. A plain RNN rewrites the whole page every step, so old notes get smudged. The modern LSTM has three gates, each a small sigmoid layer producing a number between 0 and 1 for every slot: a <b>forget gate</b> deciding what to wipe, an <b>input gate</b> deciding what to write, and an <b>output gate</b> deciding what to reveal.`),
        callout('tryit', '🖐 Try this — hold a memory for thirty steps, then choose to drop it',
          `The value 1.0 is written into the belt at step 3. After that, <code class="inline">c<sub>t</sub> = forget × c<sub>t−1</sub> + input × new</code>.<br>
           <b>1.</b> Press <b>Remember perfectly</b>: forget = 1.00, input = 0. The line is <b>dead flat</b> across all thirty steps, and the gradient multiplier reads exactly 1.000. Nothing vanishes.<br>
           <b>2.</b> Drag <b>forget</b> to 0.90 — the LSTM behaving like a plain RNN. Watch it decay to almost nothing, and compare against the grey plain-RNN line.<br>
           <b>2b.</b> Put forget back to 1.00 and drag the <b>input gate</b> up. Later words now pile onto the belt and the flat line starts climbing and wobbling away from the value you wrote: holding a memory needs the write gate <i>shut</i>, not just the forget gate open.<br>
           <b>3.</b> Press <b>Forget on cue</b>: the gate stays at 1.00 and then slams shut at step 20. The memory is held perfectly, then <b>deliberately</b> dropped — which is the thing a plain RNN can never do.`),
        buildLSTMDemo(ctx),
        p(`Look at what forget = 1 and input = 0 does: the cell state is copied <b>unchanged</b>. A multiplier of exactly one — precisely the value you found in the gradient demo that neither vanishes nor explodes. So a gradient can flow back through hundreds of steps along the belt.`),
        p(`And crucially the network <i>learns</i> when to open and close those gates, which means it learns <b>what is worth remembering</b>. In practice LSTMs handle dependencies of a few hundred tokens; plain RNNs manage perhaps ten.`),
        p(`The <em>GRU</em> (Cho et al., 2014) is the same idea with two gates instead of three and no separate cell state. Slightly cheaper, often just as good. Both were the workhorses of language AI until 2017.`),
        callout('history', '📜 1986 → 1997 → 2014 → 2017',
          `Recurrent networks were trained with backpropagation through time from the mid-1980s (Rumelhart, Hinton and Williams, 1986; Elman's "Finding structure in time", 1990).
           Hochreiter's 1991 thesis diagnosed the vanishing-gradient problem precisely, and in <b>1997</b> he and Schmidhuber published the LSTM to solve it.
           The idea then sat largely unused for a decade — the data and the GPUs were not there.
           LSTMs began winning: three handwriting competitions in <b>2009</b> (Graves and colleagues, with bidirectional LSTMs), speech recognition in <b>2013</b> (Graves), and in <b>2014</b> the seq2seq paper from Sutskever, Vinyals and Le, followed by Bahdanau's attention.
           By 2016 LSTMs were inside Google Translate. In <b>2017</b> the transformer arrived, and within two years RNNs had all but vanished from language research.`),
      ));

      root.append(section('Reading one sentence, writing another',
        p(`If an RNN can read a sentence into a summary vector, another RNN can write a sentence <i>out</i> of that vector. Sutskever, Vinyals and Le showed this in 2014: an <em>encoder</em> reads English, its final hidden state is handed to a <em>decoder</em>, and the decoder emits French one word at a time, feeding each word back in as the next input. The <em>sequence-to-sequence</em> model.`),
        p(`It worked, and it had an obvious weakness. The entire meaning of a 40-word sentence had to squeeze through one fixed-size vector — 8,000 numbers in the 2014 seq2seq paper, and that was the whole sentence, however long it ran. Translation quality fell off sharply for long sentences — the <em>bottleneck problem</em>, a whole paragraph forced through a keyhole.`),
        callout('tryit', '🖐 Try this — watch the keyhole, then remove it',
          `<b>1.</b> Start in <b>bottleneck</b> mode and press <b>▶ Play</b>. Every source word is crushed into one vector, and the decoder writes from that alone. Drag the source-length slider up to 20 and watch the "numbers per source word" figure collapse — the picture keeps showing six words, but the arithmetic is what matters here.<br>
           <b>2.</b> Switch to <b>attention</b>. Now the decoder looks back at <b>every</b> source word each time it writes one, and the lines show which it is leaning on.<br>
           <b>3.</b> Step through the target words and watch the bright line track across the source — the model aligning the two languages, without ever being told how.`),
        buildAttentionDemo(ctx),
        p(`Here is the tell that the bottleneck was real: in the 2014 paper, feeding the source sentence in <b>backwards</b> made translations markedly better. Not because reversed English is easier, but because it moved the first English words closer in time to the first French words the decoder had to produce. <b>When reversing your input helps, your memory is the problem.</b>`),
        p(`The fix arrived from Bahdanau, Cho and Bengio in September 2014. Keep every encoder hidden state — one per input word — instead of only the last. Then, each time the decoder writes a word, let it <b>look back</b> at all of them and take a weighted average, with weights depending on what it is trying to write right now. Writing "chat"? Weight "cat" heavily.`),
        p(`They called this <em>attention</em>, and the weights, when plotted, showed the model aligning source and target words almost the way a human translator would. Nobody supplied those alignments. They fell out of training.`),
        callout('key', '🔑 The bridge to chapter 7',
          `Attention was invented as a <b>patch for RNNs</b>. Three years later a team at Google asked the obvious question: if attention is doing the important work, why keep the recurrence at all?<br>
           Their answer was the transformer — attention and nothing else. Everything in chapter 7 builds on the idea you just watched:
           <b>a weighted look-back over all previous positions, with learned weights.</b>`),
        callout('example', '🌍 Google Translate, November 2016',
          `Google replaced its phrase-based statistical translator with GNMT — an 8-layer LSTM encoder–decoder with attention, trained on hundreds of millions of sentence pairs.
           Overnight, translation errors dropped by roughly 60% on the language pairs Google measured, and users noticed that translations suddenly read like sentences instead of word salad.
           It was the biggest single quality jump in the product's history.`),
        callout('example', '🌍 Voice recognition and your keyboard',
          `From about 2015 to 2019 the speech recognisers in Siri, Google Voice and Alexa were LSTMs reading audio frames — tiny sound snapshots, 100 per second — and emitting characters.
           The autocomplete on your phone keyboard was for years a small LSTM running on-device, and so was the "Smart Reply" that suggests "Sounds good!" under an email.
           RNNs also read heartbeat traces, predicted the next note in a melody, and generated fake Shakespeare one character at a time — Andrej Karpathy's 2015 essay on that last trick convinced a generation of engineers that sequence models could learn structure nobody programmed.`),
      ));

      root.append(section('Why RNNs lost',
        p(`Two reasons, both fatal. First, <b>speed</b>. An RNN must finish token 1 before it can start token 2: the computation is inherently sequential. A GPU has thousands of cores that want to work simultaneously, and a sequential algorithm leaves almost all of them idle. Training on a billion words took weeks.`),
        p(`Second, <b>memory still faded</b>. Gates stretched the usable context from ten tokens to a few hundred, but the summary was still a fixed-size vector overwritten at every step. A fact from 2,000 tokens back was simply gone.`),
        p(`The transformer fixed both at once. Every token attends to every other token directly, so nothing is squeezed through a summary — the "memory" is just all the previous tokens, kept. And with no step-by-step dependency during training, every position is processed in parallel, which is exactly what a GPU wants.`),
        callout('warning', '⚠️ The twist: recurrence came back',
          `Transformers have their own cost — attention over <i>n</i> tokens takes <i>n</i><sup>2</sup> work, so a million-token context is expensive.
           From 2023 researchers went back to recurrence with better tools. <em>State-space models</em> such as Mamba (Gu and Dao, December 2023) keep a compressed running state like an RNN, but with learned, input-dependent gates that can be trained in parallel — fixing the exact two things that killed the RNN.
           <em>Linear-attention</em> variants and hybrids (RWKV, Griffin, Jamba, and hybrid layers in several 2024–2025 production models) mix a few attention layers with many recurrent ones.
           The hidden-state idea did not die. It was rebuilt with everything learned in between.`),
      ));

      root.append(section('Why this matters for modern AI',
        p(`Three ideas from this chapter are load-bearing in every model you use today.`),
        p(`<b>Autoregression</b> — predict the next item, feed it back in, repeat — is exactly how Claude or ChatGPT write an answer: one token at a time, each conditioned on everything before. You drove it yourself with the <b>Generate</b> button at the top of this chapter.`),
        p(`<b>Attention</b> was born here, as a patch for RNN forgetfulness, and went on to become the entire architecture. And the <b>vanishing-gradient lesson</b> — that signals must be able to flow through a network unchanged — reappears as the transformer's residual connections in chapter 7. The conveyor belt never went away; it just moved.`),
        p(`So when you hear that a new model "has a state-space backbone" or "uses a hybrid of attention and recurrence", you now know exactly what trade is being made: <b>fixed-size, fast, forgetful memory versus complete, expensive, perfect memory.</b>`),
        p(`One picture to keep: <b>an RNN is a network that reads one item at a time while carrying a running summary — and the entire history of this field is about that summary not being good enough.</b>`),
      ));

      root.append(ctx.quiz([
        { q: 'Your count table got sharper as you raised the context length, but started reporting "never seen → backed off". Why can you not simply use a context of 10 characters?', options: ['Longer contexts are slower to look up', 'A 10-character context over 27 letters has about 200 trillion possible keys — no table can hold it and no text can fill it', 'The alphabet is too small', 'Backing off is a bug in the implementation'], answer: 1, explain: '27^10 is roughly 200 trillion. That is the wall you hit by dragging the context slider, and the whole motivation for replacing an exact-match table with a compressed, learned summary.' },
        { q: 'A plain RNN multiplies its hidden state by roughly the same factor at every step. Why is that fatal for long sequences?', options: ['It makes the network too slow', 'Multiplied 50 times, a factor of 0.9 gives 0.005 and a factor of 1.1 gives 117 — the gradient either vanishes or explodes', 'It uses too much memory', 'The hidden state becomes negative'], answer: 1, explain: 'You saw both on the log-scale chart. Exploding gradients have a cheap fix (clipping). Vanishing ones do not, which is why the network simply cannot learn that something 50 steps back mattered.' },
        { q: 'What does an LSTM forget gate set to exactly 1.0, with the input gate at 0, actually do?', options: ['Erases the memory', 'Copies the cell state through unchanged, giving a gradient multiplier of exactly 1 — so nothing vanishes', 'Doubles the cell state each step', 'Randomises the memory to prevent overfitting'], answer: 1, explain: 'That is the conveyor belt, and the flat line you produced by pressing "Remember perfectly". A multiplier of exactly one is the only value that neither decays nor explodes, and the LSTM is built entirely around being able to hold it.' },
        { q: 'In the 2014 seq2seq paper, feeding the English sentence in backwards improved translations. What did that reveal?', options: ['That French and English have opposite word order', 'That the bottleneck was real — reversing moved the first source words closer in time to the first output words, so less had been forgotten', 'That the model was overfitting', 'That longer sentences are easier'], answer: 1, explain: 'It is a memory problem wearing a disguise. If shuffling the order in which information arrives helps, the model is forgetting the early part — which is exactly what attention fixed by keeping every encoder state instead of only the last.' },
        { q: 'Why did transformers replace RNNs?', options: ['They need less training data', 'RNNs must process tokens strictly in order, wasting a GPU, and their fixed-size summary still forgot; attention removes both limits at once', 'RNNs cannot be trained with backpropagation', 'Transformers are smaller'], answer: 1, explain: 'Speed and memory. Attention lets every token see every other token directly, so nothing is squeezed through a summary, and with no step-by-step dependency the whole sequence trains in parallel. The cost is that attention over n tokens takes n² work — which is why recurrence is now coming back in state-space models.' },
      ]));

      root.append(section('Go deeper',
        ul([
          `<a href="https://karpathy.github.io/2015/05/21/rnn-effectiveness/" target="_blank" rel="noopener">Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks"</a> — the 2015 essay that convinced a generation. Character-level RNNs writing Shakespeare, LaTeX and C code.`,
          `<a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/" target="_blank" rel="noopener">Chris Olah, "Understanding LSTM Networks"</a> — still the clearest diagram-led explanation of the gates you just dragged.`,
          `<a href="https://www.bioinf.jku.at/publications/older/2604.pdf" target="_blank" rel="noopener">Hochreiter &amp; Schmidhuber (1997), "Long Short-Term Memory"</a> — the original paper.`,
          `<a href="https://arxiv.org/abs/1409.3215" target="_blank" rel="noopener">Sutskever, Vinyals &amp; Le (2014), "Sequence to Sequence Learning"</a> and <a href="https://arxiv.org/abs/1409.0473" target="_blank" rel="noopener">Bahdanau, Cho &amp; Bengio (2014), "Neural Machine Translation by Jointly Learning to Align and Translate"</a> — seq2seq, and the invention of attention.`,
          `<a href="https://arxiv.org/abs/2312.00752" target="_blank" rel="noopener">Gu &amp; Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"</a> — recurrence returns, with the two fatal flaws fixed.`,
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
    /* opens at n = 1, which is where the try-it's first step starts reading */
    const S = { text: DEFAULT_TEXT, k: 1, prefix: 'the ', temp: 0.6, tables: null, dist: [], usedK: -1, options: 0 };
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
    const kSl = ctx.slider({ label: 'context length n', min: 1, max: MAXK, step: 1, value: 1, onChange: (v) => { S.k = ctx.clamp(Math.round(v), 1, MAXK); recompute(); } });
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
      S.text = DEFAULT_TEXT; S.k = 3; S.prefix = 'the '; S.temp = 0.6;
      taWrap.value = DEFAULT_TEXT; prefIn.value = 'the '; kSl.value = 3; tempSl.value = 0.6;
      genOut.textContent = PLACEHOLDER;
      retrain();
    });

    retrain();
    const body = ctx.h('div', {}, cv, genOut);
    return ctx.figure(body,
      `Bars are the model's top-5 guesses for the very next character, recomputed live as you type or edit the text — pure counting, no learning involved. Watch the readout: on this small text, raising <b>n</b> grows the table only slowly (24, 117, 196, 249 contexts) because it runs out of distinct snippets — and that is the wall from the other side. The number of contexts a language <em>could</em> need explodes as the alphabet is raised to the power of n, which is the combinatorial wall that kills lookup tables. An <em>RNN</em> throws the whole table away and replaces it with a small vector of numbers — a compressed, <b>learned</b> memory that plays the same role without storing every combination it has ever seen.`,
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
        if (i <= S.t) { g.fillStyle = C.text; g.fillText((frac * 100).toFixed(frac < 0.01 ? 1 : 0) + '%', cx, by0 + bh - barH - 4); }
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
      const tag = final < 1e-2 ? '  (vanished)' : final > 1e2 ? '  (exploded)' : '';
      g.fillText('gradient reaching the first time step ≈ ' + final.toExponential(2) + tag, px + pw / 2, py - 14);
    }
    function refresh() {
      draw();
      const final = magAt(Math.max(1, Math.round(S.n)), S.m);
      ro.set({
        'multiplier': S.m.toFixed(2),
        'steps back': Math.round(S.n),
        'multiplier ^ steps': final.toExponential(2),
        'verdict': final < 1e-2 ? 'vanished — step 1 cannot learn' : final > 1e2 ? 'exploded — clip the gradient' : 'signal survives',
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

  /* ---------- shared text helper for the figures below ---------- */
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

  /* ---------- Interactive B: order carries meaning; length varies ---------- */
  function buildOrderDemo(ctx) {
    const [cv, g] = ctx.canvas(720, 340);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    let a = 'the dog bit the man';
    let b = 'the man bit the dog';
    let slots = 8;

    const words = (s) => String(s).toLowerCase().trim().split(/\s+/).filter(Boolean);
    function bag(s) { const m = new Map(); for (const w of words(s)) m.set(w, (m.get(w) || 0) + 1); return m; }
    function sameBag(x, y) {
      if (x.size !== y.size) return false;
      for (const [k, v] of x) if (y.get(k) !== v) return false;
      return true;
    }

    const taA = ctx.textarea({ label: 'sentence A', value: a, onChange: (v) => { a = v; } });
    const taB = ctx.textarea({ label: 'sentence B', value: b, onChange: (v) => { b = v; } });
    const slotSl = ctx.slider({ label: 'input slots (fixed)', min: 3, max: 20, step: 1, value: 8, onChange: (v) => { slots = v; } });
    const ro = ctx.readout();

    ctx.loop(() => {
      g.clearRect(0, 0, cv.W, cv.H);
      const wa = words(a), wb = words(b);
      const ba = bag(a), bb = bag(b);
      const identical = sameBag(ba, bb) && wa.join(' ') !== wb.join(' ');

      /* ---- word-count vectors side by side ---- */
      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('what an order-blind model receives: just the counts', 30, 26);
      const vocab = [...new Set([...ba.keys(), ...bb.keys()])].sort();
      const colW = Math.min(66, 420 / Math.max(1, vocab.length));
      vocab.forEach((w, i) => {
        const x = 30 + i * colW;
        g.font = MONO; g.fillStyle = C.muted;
        g.fillText(w.slice(0, 7), x, 48);
        [[ba, 62, C.accent], [bb, 92, C.warn]].forEach(([bg, y, col]) => {
          const n = bg.get(w) || 0;
          g.fillStyle = n ? col : '#131a27';
          g.fillRect(x, y, colW - 6, 20);
          if (n) { g.fillStyle = '#0a0e16'; g.font = 'bold ' + MONO; g.fillText(String(n), x + (colW - 6) / 2 - 3, y + 14); }
        });
      });
      g.font = MONO; g.fillStyle = C.accent; g.fillText('A', 12, 76);
      g.fillStyle = C.warn; g.fillText('B', 12, 106);

      g.font = 'bold 15px Inter, system-ui, sans-serif';
      g.fillStyle = identical ? C.danger : C.green;
      g.fillText(identical
        ? 'Identical inputs — yet the sentences mean opposite things.'
        : (wa.join(' ') === wb.join(' ') ? 'The two sentences are the same.' : 'Different counts, so these two are distinguishable.'), 30, 136);
      if (identical) {
        g.font = FONT; g.fillStyle = C.muted;
        wrapText(g, 'Nothing downstream can tell these apart. Not a bigger network, not more training data — the information was destroyed before the model saw it.', 30, 156, 640, 17);
      }

      /* ---- the fixed-size input problem ---- */
      const Y = 210;
      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('and a fixed-size input has nowhere to put a sentence', 30, Y - 12);
      /* the grid has to leave room for the status label to its right, so the cells
         narrow as the slot count grows instead of pushing the label off the canvas */
      const CW = Math.min(30, 480 / slots), CH = 26;
      const CHARS = CW >= 26 ? 4 : 3;
      [[wa, Y, C.accent, 'A'], [wb, Y + 46, C.warn, 'B']].forEach(([ws, y, col, lab]) => {
        g.font = MONO; g.fillStyle = col; g.fillText(lab, 12, y + 18);
        for (let i = 0; i < slots; i++) {
          const x = 30 + i * CW;
          const w = ws[i];
          g.fillStyle = w ? col : '#131a27';
          g.globalAlpha = w ? 0.8 : 1; g.fillRect(x, y, CW - 3, CH); g.globalAlpha = 1;
          if (w) { g.fillStyle = '#0a0e16'; g.font = '10px "JetBrains Mono", ui-monospace, monospace'; g.fillText(w.slice(0, CHARS), x + 2, y + 17); }
        }
        const over = ws.length - slots;
        g.font = MONO;
        if (over > 0) { g.fillStyle = C.danger; g.fillText('+' + over + ' word' + (over > 1 ? 's' : '') + ' cut off', 30 + slots * CW + 8, y + 18); }
        else if (over < 0) { g.fillStyle = C.muted; g.fillText((-over) + ' slot' + (over < -1 ? 's' : '') + ' wasted', 30 + slots * CW + 8, y + 18); }
        else { g.fillStyle = C.green; g.fillText('exact fit — by luck', 30 + slots * CW + 8, y + 18); }
      });
      g.font = FONT; g.fillStyle = C.muted;
      wrapText(g, 'Pick the number of slots before you see the data and you are always either truncating real sentences or paying for empty ones.', 30, Y + 104, 500, 17);
      ro.set({ 'A words': wa.length, 'B words': wb.length, 'same counts': identical ? 'YES — indistinguishable' : 'no', slots });
    });

    return ctx.figure(cv,
      'Two sentences built from exactly the same words. Their word-count vectors — all an order-blind model ever receives — are byte-for-byte identical, so no amount of depth or data can separate "the dog bit the man" from "the man bit the dog". Underneath, the other half of the problem: a network with a fixed number of input slots must chop long sentences and waste slots on short ones. A network that reads has to solve both.',
      [taA, taB, slotSl], ro);
  }

  /* ---------- Interactive D: the LSTM conveyor belt ---------- */
  function buildLSTMDemo(ctx) {
    const [cv, g] = ctx.canvas(720, 340);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    const STEPS = 30, WRITE = 3;
    let forget = 1.0, input = 0.0, cutAt = -1;

    /* c_t = forget*c_{t-1} + input*x_t. The value 1.0 is written at step WRITE; after that a
       stream of *other* words keeps arriving, and the input gate decides whether any of it is
       allowed to scribble over the note. The plain-RNN line is h_t = tanh(0.8*h_{t-1} + x_t). */
    const other = (t) => 0.7 * Math.sin(t * 1.7);
    function belt() {
      const c = [0], rnn = [0], rnnNo = [0];
      for (let t = 1; t <= STEPS; t++) {
        const isWrite = t === WRITE;
        const x = isWrite ? 1 : other(t);
        const f = (cutAt >= 0 && t >= cutAt) ? 0 : forget;
        const i = isWrite ? 1 : input;                        // the write itself always lands
        c.push(f * c[t - 1] + i * x);
        rnn.push(Math.tanh(0.8 * rnn[t - 1] + x));
        rnnNo.push(Math.tanh(0.8 * rnnNo[t - 1] + (isWrite ? other(t) : x)));
      }
      /* the retained trace: the plain RNN's state minus the same run without
         the write, i.e. how much of the written value is still in there */
      return { c, rnn: rnn.map((v, t) => v - rnnNo[t]) };
    }

    const fSl = ctx.slider({ label: 'forget gate', min: 0, max: 1, step: 0.01, value: 1, digits: 2, onChange: (v) => { forget = v; cutAt = -1; } });
    const iSl = ctx.slider({ label: 'input gate (lets later words in)', min: 0, max: 0.5, step: 0.01, value: 0, digits: 2, onChange: (v) => { input = v; } });
    const holdBtn = ctx.button('Remember perfectly', () => { forget = 1; fSl.value = 1; input = 0; iSl.value = 0; cutAt = -1; }, 'primary');
    const decayBtn = ctx.button('Behave like a plain RNN', () => { forget = 0.9; fSl.value = 0.9; input = 0; iSl.value = 0; cutAt = -1; });
    const cueBtn = ctx.button('Forget on cue', () => { forget = 1; fSl.value = 1; input = 0; iSl.value = 0; cutAt = 20; });
    const ro = ctx.readout();

    ctx.loop(() => {
      g.clearRect(0, 0, cv.W, cv.H);
      const { c, rnn } = belt();
      const P = { x: 55, y: 46, w: 610, h: 200 };
      const px = (t) => P.x + t / STEPS * P.w;
      const top = Math.max(1.2, ...c.map(v => Math.abs(v))) * 1.05;
      const py = (v) => P.y + P.h - (ctx.clamp(v, -0.4, top) + 0.4) / (top + 0.4) * P.h;

      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('the cell state through time', P.x, 28);
      g.strokeStyle = C.line; g.lineWidth = 1;
      g.strokeRect(P.x, P.y, P.w, P.h);
      g.font = MONO; g.fillStyle = C.muted;
      [-0.25, 0, 0.25, 0.5, 0.75, 1].forEach(v => {
        g.beginPath(); g.moveTo(P.x, py(v)); g.lineTo(P.x + P.w, py(v)); g.stroke();
        g.fillText(v.toFixed(2), P.x - 34, py(v) + 4);
      });
      for (let t = 0; t <= STEPS; t += 5) g.fillText(String(t), px(t) - 5, P.y + P.h + 18);
      g.fillText('time step', P.x + P.w / 2 - 24, P.y + P.h + 36);

      /* the write event */
      g.setLineDash([3, 3]); g.strokeStyle = C.accent; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(px(WRITE), P.y); g.lineTo(px(WRITE), P.y + P.h); g.stroke();
      if (cutAt >= 0) {
        g.strokeStyle = C.danger;
        g.beginPath(); g.moveTo(px(cutAt), P.y); g.lineTo(px(cutAt), P.y + P.h); g.stroke();
      }
      g.setLineDash([]);
      /* plain RNN comparison, then the belt */
      g.strokeStyle = C.muted; g.lineWidth = 1.5; g.setLineDash([4, 3]);
      g.beginPath(); rnn.forEach((v, t) => { t ? g.lineTo(px(t), py(v)) : g.moveTo(px(t), py(v)); }); g.stroke();
      g.setLineDash([]);
      g.strokeStyle = C.green; g.lineWidth = 3;
      g.beginPath(); c.forEach((v, t) => { t ? g.lineTo(px(t), py(v)) : g.moveTo(px(t), py(v)); }); g.stroke();
      c.forEach((v, t) => { if (t % 3 === 0) { g.fillStyle = C.green; g.beginPath(); g.arc(px(t), py(v), 3, 0, 7); g.fill(); } });

      /* marker labels last, on their own plate: the cell-state trace autoscales
         and will otherwise be drawn straight through them */
      g.font = MONO;
      const plate = (txt, x, col) => {
        const w = g.measureText(txt).width;
        g.fillStyle = '#0a0e16'; g.fillRect(x + 3, P.y + 2, w + 6, 17);
        g.fillStyle = col; g.fillText(txt, x + 5, P.y + 14);
      };
      plate('write 1.0', px(WRITE), C.accent);
      if (cutAt >= 0) plate('gate shuts', px(cutAt), C.danger);

      /* Legend lives on the header row, outside the plot frame: inside the box both
         traces sweep the full height, so any in-plot key gets drawn straight through. */
      g.font = MONO;
      const KEY = [
        { col: C.muted, dash: [4, 3], lw: 1.5, t: 'plain RNN (decay 0.8)' },
        { col: C.green, dash: [], lw: 3, t: 'LSTM cell state' },
      ];
      const SW = 18, PAD = 6, GAP = 18;
      let kw = GAP * (KEY.length - 1);
      KEY.forEach((k) => { kw += SW + PAD + g.measureText(k.t).width; });
      let kx = P.x + P.w - kw;
      KEY.forEach((k) => {
        g.strokeStyle = k.col; g.lineWidth = k.lw; g.setLineDash(k.dash);
        g.beginPath(); g.moveTo(kx, 24); g.lineTo(kx + SW, 24); g.stroke();
        g.setLineDash([]);
        g.fillStyle = k.col; g.fillText(k.t, kx + SW + PAD, 28);
        kx += SW + PAD + g.measureText(k.t).width + GAP;
      });

      /* verdict */
      const left = c[STEPS];
      const eff = (cutAt >= 0 && STEPS >= cutAt) ? 0 : forget;
      g.font = 'bold 16px Inter, system-ui, sans-serif';
      g.fillStyle = left > 0.9 ? C.green : left > 0.2 ? C.warn : C.danger;
      g.fillText('after ' + STEPS + ' steps, the note reads ' + left.toFixed(3) + ' (it was written as 1.000)', P.x, 300);
      g.font = MONO; g.fillStyle = C.muted;
      g.fillText(cutAt >= 0
        ? 'gradient multiplier per step = the forget gate = ' + forget.toFixed(3) + ' until step ' + cutAt + ', then 0.000'
        : 'gradient multiplier per step = the forget gate = ' + eff.toFixed(3)
          + (eff === 1 ? '  ← nothing vanishes or explodes' : ''), P.x, 322);
      ro.set({ forget: forget.toFixed(2) + (cutAt >= 0 ? ' → 0 at step ' + cutAt : ''), input: input.toFixed(2), 'survives to step 30': (left * 100).toFixed(1) + '%', 'plain RNN': (rnn[STEPS] * 100).toFixed(1) + '%' });
    });

    return ctx.figure(cv,
      'The value 1.0 is written onto the belt at step 3, and after that <code class="inline">c<sub>t</sub> = forget × c<sub>t−1</sub> + input × new</code>. The forget gate is also exactly the gradient multiplier from the previous figure, which is the whole point: at 1.00 the memory is copied through untouched and a gradient can travel back thirty steps — or three hundred — without fading. Opening the input gate lets the stream of later words write onto the belt too, which is how a held memory gets corrupted even when nothing is forgetting. The dashed grey line shows how much of that written 1.0 is still present in a plain RNN fed the same stream — measured against an identical run in which the write never happened. It fades within a few steps. In a real LSTM these gates are not sliders; they are small sigmoid layers that <i>learn</i> when to open.',
      [fSl, iSl, holdBtn, decayBtn, cueBtn], ro);
  }

  /* ---------- Interactive E: the bottleneck, and attention removing it ---------- */
  function buildAttentionDemo(ctx) {
    const [cv, g] = ctx.canvas(720, 380);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    const SRC = ['the', 'cat', 'sat', 'on', 'the', 'mat'];
    const TGT = ['le', 'chat', 'était', 'assis', 'sur', 'le', 'tapis'];
    /* a plausible hand-written alignment, purely illustrative */
    const ALIGN = [
      [0.75, 0.10, 0.03, 0.04, 0.05, 0.03],
      [0.12, 0.74, 0.05, 0.03, 0.03, 0.03],
      [0.04, 0.10, 0.68, 0.08, 0.05, 0.05],
      [0.03, 0.06, 0.72, 0.09, 0.05, 0.05],
      [0.03, 0.04, 0.08, 0.72, 0.08, 0.05],
      [0.05, 0.03, 0.04, 0.10, 0.66, 0.12],
      [0.03, 0.04, 0.04, 0.06, 0.13, 0.70],
    ];
    let mode = 'bottleneck', tpos = 0, srcLen = 6, playing = false, acc = 0;

    const modeSel = ctx.h('select', {}, [
      ctx.h('option', { value: 'bottleneck' }, 'bottleneck (last state only)'),
      ctx.h('option', { value: 'attention' }, 'attention (look at all states)'),
    ]);
    modeSel.value = 'bottleneck';
    modeSel.addEventListener('change', () => { mode = modeSel.value; });
    const modeWrap = ctx.h('div', { class: 'control' }, ctx.h('label', {}, 'decoder can see'), modeSel);
    const lenSl = ctx.slider({ label: 'if the source were this many words', min: 6, max: 40, step: 1, value: 6, onChange: (v) => { srcLen = v; } });
    const stepBtn = ctx.button('Step', () => { tpos = (tpos + 1) % TGT.length; });
    const playBtn = ctx.button('▶ Play', () => { playing = !playing; playBtn.textContent = playing ? '⏸ Pause' : '▶ Play'; }, 'primary');
    const ro = ctx.readout();

    ctx.loop((dt) => {
      if (playing) { acc += dt; if (acc > 0.9) { acc = 0; tpos = (tpos + 1) % TGT.length; } }
      g.clearRect(0, 0, cv.W, cv.H);
      const SY = 60, TY = 300, X0 = 60, SW = 600;
      const sx = (i) => X0 + (i + 0.5) * (SW / SRC.length);
      const tx = (i) => X0 + (i + 0.5) * (SW / TGT.length);

      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('encoder reads English', X0, 30);
      g.fillText('decoder writes French', X0, TY + 54);

      /* connections */
      if (mode === 'bottleneck') {
        const KX = 360, KY = 180;
        SRC.forEach((w, i) => {
          g.strokeStyle = 'rgba(251,113,133,0.45)'; g.lineWidth = 1.5;
          g.beginPath(); g.moveTo(sx(i), SY + 14); g.lineTo(KX, KY); g.stroke();
        });
        TGT.forEach((w, j) => {
          g.strokeStyle = j === tpos ? C.warn : 'rgba(148,163,184,0.25)';
          g.lineWidth = j === tpos ? 2.5 : 1;
          g.beginPath(); g.moveTo(KX, KY); g.lineTo(tx(j), TY - 8); g.stroke();
        });
        g.fillStyle = C.danger;
        g.beginPath(); g.arc(KX, KY, 13, 0, 7); g.fill();
        g.fillStyle = '#0a0e16'; g.font = 'bold ' + MONO; g.fillText('256', KX - 11, KY + 4);
        g.font = 'bold ' + FONT; g.fillStyle = C.danger;
        g.fillText('everything must fit through here', KX + 24, KY - 6);
        g.font = MONO; g.fillStyle = C.muted;
        g.fillText((256 / srcLen).toFixed(1) + ' numbers per source word', KX + 24, KY + 14);
      } else {
        /* start the fan below the weight row so the thick line cannot slice
           through the very number it is illustrating */
        SRC.forEach((w, i) => {
          const a = ALIGN[tpos][i];
          g.strokeStyle = 'rgba(56,217,169,' + (0.08 + 0.85 * a) + ')';
          g.lineWidth = 0.6 + a * 7;
          g.beginPath(); g.moveTo(sx(i), SY + 40); g.lineTo(tx(tpos), TY - 8); g.stroke();
        });
        /* The fan sweeps the whole middle of the canvas, so this note gets an
           opaque card — painted after the lines, before the text — instead of
           being read through a 7px stroke. */
        const L1 = 'every source word stays available, all the time';
        const L2 = '256 numbers per source word, however long the sentence';
        g.font = 'bold ' + FONT; const w1 = g.measureText(L1).width;
        g.font = MONO; const w2 = g.measureText(L2).width;
        const cw = Math.max(w1, w2) + 24, cx = X0 - 12, cy = 158;
        g.fillStyle = '#0a0e16'; g.fillRect(cx, cy, cw, 50);
        g.fillStyle = 'rgba(56,217,169,0.55)'; g.fillRect(cx, cy, 3, 50);
        g.font = 'bold ' + FONT; g.fillStyle = C.green;
        g.fillText(L1, X0, cy + 20);
        g.font = MONO; g.fillStyle = C.muted;
        g.fillText(L2, X0, cy + 40);
      }

      /* words */
      SRC.forEach((w, i) => {
        const a = ALIGN[tpos][i];
        const hot = mode === 'attention' && a > 0.4;
        g.font = 'bold ' + FONT; g.fillStyle = hot ? C.green : C.text;
        const tw = g.measureText(w).width;
        if (hot) { g.fillStyle = 'rgba(56,217,169,0.18)'; g.fillRect(sx(i) - tw / 2 - 6, SY - 14, tw + 12, 22); g.fillStyle = C.green; }
        g.fillText(w, sx(i) - tw / 2, SY + 2);
        if (mode === 'attention') { g.font = MONO; g.fillStyle = C.muted; g.fillText(a.toFixed(2), sx(i) - 11, SY + 30); }
      });
      TGT.forEach((w, j) => {
        g.font = 'bold ' + FONT;
        const tw = g.measureText(w).width;
        g.fillStyle = j === tpos ? C.warn : (j < tpos ? C.muted : '#2a3444');
        g.fillText(w, tx(j) - tw / 2, TY + 16);
      });
      g.font = MONO; g.fillStyle = C.muted;
      g.fillText('writing word ' + (tpos + 1) + ' of ' + TGT.length, X0, TY + 36);

      ro.set({
        mode: mode === 'bottleneck' ? 'bottleneck' : 'attention',
        'writing': TGT[tpos],
        'info per source word': mode === 'bottleneck' ? (256 / srcLen).toFixed(1) + ' numbers' : '256 numbers',
      });
    });

    return ctx.figure(cv,
      'The same translation under both designs. In <b>bottleneck</b> mode the encoder crushes the whole sentence into one fixed vector and the decoder writes from that alone — so the longer the source, the less of each word survives, which is exactly why quality collapsed on long sentences. In <b>attention</b> mode every encoder state is kept and the decoder takes a fresh weighted average each time it writes a word; line thickness is that weight. The alignment shown here is illustrative, but the real thing looked much like it — and nobody supplied it, it fell out of training.',
      [modeWrap, lenSl, stepBtn, playBtn], ro);
  }
})();
