/* Chapter 6 — Meaning as geometry: embeddings */
(function () {
  const ZTA = window.ZTA;

  /* ---------------------------------------------------------------
     Hand-designed word vectors.
     Ten interpretable axes, so that famous analogies come out EXACTLY right.
     0 gender (male +1 / female -1)   1 royalty      2 animal     3 food
     4 vehicle                        5 isCity(+1)/isCountry(-1)
     6 regionA                        7 regionB      8 action     9 size
     --------------------------------------------------------------- */
  function V(o) {
    return [o.g || 0, o.roy || 0, o.ani || 0, o.food || 0, o.veh || 0,
            o.city || 0, o.rA || 0, o.rB || 0, o.act || 0, o.size || 0];
  }
  const WORDS = [
    // people & royalty — the classic analogy cluster
    { w: 'king',      v: V({ g: 1,  roy: 1,   size: 0.6 }),  c: 'people' },
    { w: 'queen',     v: V({ g: -1, roy: 1,   size: 0.6 }),  c: 'people' },
    { w: 'prince',    v: V({ g: 1,  roy: 0.7, size: 0.3 }),  c: 'people' },
    { w: 'princess',  v: V({ g: -1, roy: 0.7, size: 0.3 }),  c: 'people' },
    { w: 'man',       v: V({ g: 1,  ani: 0.3, size: 0.4 }),  c: 'people' },
    { w: 'woman',     v: V({ g: -1, ani: 0.3, size: 0.4 }),  c: 'people' },
    { w: 'boy',       v: V({ g: 1,  ani: 0.3, size: 0.15 }), c: 'people' },
    { w: 'girl',      v: V({ g: -1, ani: 0.3, size: 0.15 }), c: 'people' },
    { w: 'father',    v: V({ g: 1,  ani: 0.3, size: 0.5 }),  c: 'people' },
    { w: 'mother',    v: V({ g: -1, ani: 0.3, size: 0.5 }),  c: 'people' },
    // animals
    { w: 'dog',       v: V({ ani: 1, size: 0.30 }), c: 'animal' },
    { w: 'cat',       v: V({ ani: 1, size: 0.20 }), c: 'animal' },
    { w: 'horse',     v: V({ ani: 1, size: 0.70 }), c: 'animal' },
    { w: 'lion',      v: V({ ani: 1, size: 0.60 }), c: 'animal' },
    { w: 'mouse',     v: V({ ani: 1, size: 0.05 }), c: 'animal' },
    { w: 'elephant',  v: V({ ani: 1, size: 1.00 }), c: 'animal' },
    { w: 'bird',      v: V({ ani: 1, size: 0.12 }), c: 'animal' },
    { w: 'fish',      v: V({ ani: 1, size: 0.10 }), c: 'animal' },
    // food
    { w: 'apple',     v: V({ food: 1, size: 0.10 }), c: 'food' },
    { w: 'banana',    v: V({ food: 1, size: 0.12 }), c: 'food' },
    { w: 'bread',     v: V({ food: 1, size: 0.20 }), c: 'food' },
    { w: 'cheese',    v: V({ food: 1, size: 0.15 }), c: 'food' },
    { w: 'pizza',     v: V({ food: 1, size: 0.25 }), c: 'food' },
    { w: 'rice',      v: V({ food: 1, size: 0.05 }), c: 'food' },
    { w: 'soup',      v: V({ food: 1, size: 0.18 }), c: 'food' },
    { w: 'cake',      v: V({ food: 1, size: 0.22 }), c: 'food' },
    // vehicles
    { w: 'car',       v: V({ veh: 1, size: 0.40 }), c: 'vehicle' },
    { w: 'truck',     v: V({ veh: 1, size: 0.70 }), c: 'vehicle' },
    { w: 'bicycle',   v: V({ veh: 1, size: 0.15 }), c: 'vehicle' },
    { w: 'train',     v: V({ veh: 1, size: 0.90 }), c: 'vehicle' },
    { w: 'plane',     v: V({ veh: 1, size: 0.85 }), c: 'vehicle' },
    { w: 'boat',      v: V({ veh: 1, size: 0.55 }), c: 'vehicle' },
    // places — capitals and their countries
    { w: 'France',    v: V({ city: -1, rA: 1 }),  c: 'place' },
    { w: 'Paris',     v: V({ city: 1,  rA: 1 }),  c: 'place' },
    { w: 'Italy',     v: V({ city: -1, rB: 1 }),  c: 'place' },
    { w: 'Rome',      v: V({ city: 1,  rB: 1 }),  c: 'place' },
    { w: 'Japan',     v: V({ city: -1, rA: -1 }), c: 'place' },
    { w: 'Tokyo',     v: V({ city: 1,  rA: -1 }), c: 'place' },
    { w: 'Germany',   v: V({ city: -1, rB: -1 }), c: 'place' },
    { w: 'Berlin',    v: V({ city: 1,  rB: -1 }), c: 'place' },
    // verbs
    { w: 'run',       v: V({ act: 1, size: 0.5 }),  c: 'verb' },
    { w: 'walk',      v: V({ act: 1, size: 0.3 }),  c: 'verb' },
    { w: 'eat',       v: V({ act: 1, food: 0.4 }),  c: 'verb' },
    { w: 'sleep',     v: V({ act: 1, size: 0.1 }),  c: 'verb' },
  ];
  const CAT_COLOR = { people: '#f472b6', animal: '#38d9a9', food: '#fbbf24', vehicle: '#7c9cff', place: '#a78bfa', verb: '#fb923c' };
  const DIM = 10;

  /* vector maths */
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function norm(a) { return Math.sqrt(dot(a, a)); }
  function cosine(a, b) { const d = norm(a) * norm(b); return d < 1e-9 ? 0 : dot(a, b) / d; }
  function addv(a, b, s) { return a.map((x, i) => x + s * b[i]); }

  /* Project the 10-D vectors down to 2-D with real PCA (power iteration).
     This is exactly the trick used to draw the famous word2vec pictures. */
  function pca2(vecs) {
    const n = vecs.length, d = vecs[0].length;
    const mean = new Array(d).fill(0);
    for (const v of vecs) for (let i = 0; i < d; i++) mean[i] += v[i] / n;
    const X = vecs.map(v => v.map((x, i) => x - mean[i]));
    function topAxis(rows) {
      let v = new Array(d).fill(0).map((_, i) => Math.sin(i * 12.9898) * 43758.5453 % 1); // deterministic seed
      for (let it = 0; it < 60; it++) {
        const out = new Array(d).fill(0);
        for (const r of rows) { const s = dot(r, v); for (let i = 0; i < d; i++) out[i] += s * r[i]; }
        const m = norm(out); if (m < 1e-12) break;
        v = out.map(x => x / m);
      }
      return v;
    }
    const a1 = topAxis(X);
    const X2 = X.map(r => { const s = dot(r, a1); return r.map((x, i) => x - s * a1[i]); }); // deflate
    const a2 = topAxis(X2);
    return { mean, a1, a2 };
  }
  const PROJ = pca2(WORDS.map(w => w.v));
  function project(v) {
    const c = v.map((x, i) => x - PROJ.mean[i]);
    return { x: dot(c, PROJ.a1), y: dot(c, PROJ.a2) };
  }
  WORDS.forEach(w => { const p = project(w.v); w.px = p.x; w.py = p.y; });

  ZTA.registerChapter({
    id: '06-embeddings',
    num: 6,
    part: 'II',
    title: 'Meaning as geometry: embeddings',
    tagline: 'How a computer turns "king" into 300 numbers, and why that makes king minus man plus woman come out as queen.',

    render(root, ctx) {
      const h = ctx.h, C = ctx.colors;

      /* ============================ HOOK ============================ */
      root.append(
        ctx.p(`Type <b>"comfortable running shoes"</b> into a shopping site and it shows you trainers that never use the word "comfortable" anywhere on the page. Ask a music app for something like the song you just played and it finds a track by an artist you have never heard of, in a different decade, that somehow fits. Neither system understands English. Both are doing the same trick, and it is the single most reusable idea in modern AI.`),
        ctx.p(`The trick is this: <em>turn every thing into a point in space, and arrange the space so that similar things end up near each other</em>. Once meaning is geometry, the hard question "are these two things alike?" collapses into the easy question "how far apart are these two points?" Computers are extremely good at the second question.`),
        ctx.p(`Those points are called <em>embeddings</em>. Every model in this course from here on runs on them. The very first thing a language model does with your prompt is look up an embedding for each token; the very last thing it does is compare a vector against every word it knows. This chapter is about where those numbers come from and why they behave so strangely well.`),
      );

      /* ======================= THE NAIVE WAY ======================= */
      root.append(ctx.section('First attempt: give every word a number',
        ctx.p(`A computer cannot store the word "cat". It stores numbers. So the obvious first move is a lookup table: <code class="inline">aardvark = 1, apple = 2, … cat = 3312, … zebra = 50000</code>. This fails instantly and instructively. It claims that <code class="inline">cat</code> (3312) is nearly the same as <code class="inline">catalogue</code> (3313), and that <code class="inline">apple</code> is 3310 units away from <code class="inline">cat</code> but only 1 unit from <code class="inline">apply</code>. The numbers carry an order that has nothing to do with meaning.`),
        ctx.p(`The standard fix is <em>one-hot encoding</em>: give every word its own axis. With a 50,000-word vocabulary, "cat" becomes a list of 50,000 zeros with a single 1 in slot 3312. Now no word is accidentally close to another. But look at what that costs: <b>every pair of distinct words is now exactly equally far apart</b>. "cat" and "dog" are as unrelated as "cat" and "bureaucracy". We removed the false information and replaced it with no information.`),
        ctx.callout('key', '🔑 The key idea',
          `Both failures come from us <b>choosing</b> the numbers. The answer is to stop choosing and start <b>learning</b> them: let the numbers be parameters, and let a training task push them into a useful arrangement. Meaning is not something we encode. It is something that falls out of prediction.`),
      ));

      /* ==================== DISTRIBUTIONAL HYPOTHESIS ==================== */
      root.append(ctx.section('You shall know a word by the company it keeps',
        ctx.p(`In 1957 the linguist J. R. Firth wrote the sentence that this whole field rests on: you shall know a word by the company it keeps. It is a claim about how meaning works. If I show you a word you have never seen, in enough sentences, you will work out what it means from context alone:`),
        ctx.code(`I poured the tesgüino into a glass.\nEveryone drinks tesgüino at the festival.\nToo much tesgüino makes you sleepy.\nTesgüino is made from corn.`),
        ctx.p(`You now know roughly what tesgüino is: an alcoholic drink made from corn. Nobody defined it. You inferred it purely from the <em>distribution</em> of contexts it appears in. Notice also that you can tell it is more like "beer" than like "hammer", because beer appears in the same kinds of sentences. That is the <em>distributional hypothesis</em>: words that appear in similar contexts have similar meanings.`),
        ctx.p(`This turns a philosophical problem into an engineering one. We do not need to teach a machine what "beer" means. We need only give it a prediction task involving context, and force it to compress what it learns into a short list of numbers. Whatever arrangement of numbers makes the prediction easiest will necessarily place beer near wine, because they are interchangeable in text.`),
        ctx.callout('history', '📜 Where this came from',
          `In 2013 Tomáš Mikolov and colleagues at Google published <b>word2vec</b>, which made this practical at scale. The training task was almost insultingly simple: given a word, predict the words around it (skip-gram), or the reverse (CBOW). Trained on billions of words, it produced 300-number vectors whose geometry stunned people. Earlier work had similar ideas (latent semantic analysis in the late 1980s, Bengio's neural language model in 2003), but word2vec was fast enough to run on enormous corpora, and its results were vivid enough that the whole field noticed at once.`),
      ));

      /* ==================== INTERACTIVE 1: THE MAP ==================== */
      root.append(ctx.sub('The map of meaning',
        ctx.p(`Below is a small, hand-built embedding space: 44 words, each one a list of 10 numbers. The picture is a genuine <em>PCA projection</em> of those 10-dimensional vectors down to the 2 dimensions your screen has, computed in your browser when this page loaded. PCA finds the two directions along which the points are most spread out, so it keeps as much of the structure as a flat picture can hold.`),
        ctx.callout('tryit', '🖐 Try this',
          `<b>1.</b> Hover a word to see its five nearest neighbours by cosine similarity. Notice the clusters formed themselves out of the numbers; nobody drew the groups.<br>
           <b>2.</b> Drag to pan, scroll or pinch to zoom.<br>
           <b>3.</b> Now the famous part. In the arithmetic row, run <b>king − man + woman</b>. Watch the arrows. The answer is not looked up anywhere; it is the nearest word to a point computed by adding and subtracting vectors.<br>
           <b>4.</b> Try <b>Paris − France + Italy</b>. The same geometric move that turns a man into a woman turns a country into its capital.`),
      ));

      root.append(buildMap(ctx));

      root.append(
        ctx.p(`What is happening in that arithmetic is worth stating precisely, because it is easy to over-mystify. Nothing in the training process said "encode gender". But if the space is arranged so that context is predictable, then whatever distinguishes <i>king</i> from <i>queen</i> must also distinguish <i>man</i> from <i>woman</i>, because those pairs are swapped in the same kinds of sentences. The difference gets stored as a <b>direction</b>. Subtracting <i>man</i> and adding <i>woman</i> is a translation along that direction. Directions in the space turn out to mean things: there is a gender direction, a plural direction, a past-tense direction, a capital-city direction.`),
        ctx.callout('warning', '⚠️ An honest caveat',
          `The analogy result is real but often oversold. In published word2vec results the query vector's own inputs are excluded from the answer, and many analogies fail outside a curated set. The space here is hand-designed so the four preset analogies come out exactly; a trained space is messier. The <b>directions-carry-meaning</b> insight is solid. The <b>arithmetic always works</b> claim is not.`),
      );

      /* ==================== COSINE ==================== */
      root.append(ctx.section('Measuring closeness: cosine similarity',
        ctx.p(`"Near each other" needs a definition. The one nearly everyone uses is <em>cosine similarity</em>: the cosine of the angle between two vectors. It ignores length entirely and asks only whether two vectors point the same way. That matters because in text a common word gets a long vector and a rare word a short one, and we do not want frequency masquerading as meaning.`),
        ctx.code(`cos(a, b) = (a · b) / (|a| × |b|)\n\nworked example, in 2 dimensions:\n  a = [3, 4]      b = [4, 3]\n  a · b = 3×4 + 4×3            = 24\n  |a|   = sqrt(9 + 16)         = 5\n  |b|   = sqrt(16 + 9)         = 5\n  cos   = 24 / (5 × 5)         = 0.96      → 16 degrees apart, very similar\n\n  a = [3, 4]      c = [-4, 3]\n  a · c = 3×(-4) + 4×3         = 0\n  cos   = 0 / (5 × 5)          = 0.00      → 90 degrees, unrelated`),
        ctx.p(`The scale runs from <b>+1</b> (same direction, same meaning) through <b>0</b> (perpendicular, unrelated) to <b>−1</b> (opposite). In a real embedding space, "cat" and "dog" sit around 0.8, "cat" and "democracy" around 0.05. Drag the arrows below to feel it.`),
        ctx.callout('tryit', '🖐 Try this',
          `Drag either arrowhead. Watch how the cosine depends only on the <b>angle</b>: make one arrow twice as long and the number does not move. Then set them 90° apart and note the score of exactly zero. That is what "unrelated" means numerically.`),
      ));
      root.append(buildCosine(ctx));

      /* ==================== INTERACTIVE 3: LIVE TRAINING ==================== */
      root.append(ctx.section('Watching an embedding space assemble itself',
        ctx.p(`Everything so far used vectors I wrote by hand. Now let a model discover them. Below is a real <em>skip-gram</em> model, the word2vec training task, running in your browser: it takes a centre word, tries to predict the words around it, and adjusts each word's 2 numbers by gradient descent when it is wrong.`),
        ctx.p(`The training corpus is a set of template sentences. Crucially, <b>the model is never told which words are foods or vehicles</b>. It only ever sees which words appear near which. The clusters that appear are entirely a consequence of shared context.`),
        ctx.callout('tryit', '🖐 Try this',
          `Press <b>Train</b> and watch. The words begin scattered at random. Within a few thousand steps the fruits pull together, the animals pull together, the vehicles pull together, because each group is interchangeable in the corpus. Push the learning rate up to see the points thrash and overshoot; drop it low to see progress crawl. Press <b>Reset</b> and run it again: the clusters re-form, but in different positions and orientations. <b>The absolute coordinates are meaningless. Only the relative arrangement carries information.</b>`),
      ));
      root.append(buildWord2Vec(ctx));

      /* ==================== BEYOND WORDS ==================== */
      root.append(ctx.section('The idea generalises to absolutely everything',
        ctx.p(`Nothing in the recipe was about language. The recipe was: <b>define a prediction task, force the thing through a narrow layer of numbers, and the geometry of that layer becomes a similarity space.</b> Swap in any kind of object and it still works.`),
        ctx.cards([
          { title: 'Songs and videos', body: 'Predict what a listener plays next. Songs that get played in the same sessions land near each other. This is how "recommended for you" works at Spotify, Netflix and TikTok, usually alongside an embedding for <i>you</i> in the same space, so recommendation becomes a nearest-neighbour lookup.' },
          { title: 'Images', body: 'CLIP (2021) trained image and text encoders together so that a photo of a dog and the caption "a dog" land at the same point. One shared space for two modalities is what lets you search photos in plain English, and what lets an image generator understand a prompt.' },
          { title: 'Faces and voices', body: 'Face unlock does not store your face. It stores an embedding, and checks whether the new photo lands within a small distance of it. Same trick for speaker identification.' },
          { title: 'Proteins and molecules', body: 'Amino-acid sequences embedded by the same machinery power structure prediction and drug screening. AlphaFold 2 (2021) leaned heavily on representations learned from evolutionary context.' },
        ]),
        ctx.callout('example', '🌍 Where you have used this today',
          `<b>Semantic search.</b> Traditional search matches keywords; a page about "affordable laptops" will not match a query for "cheap notebooks". Embedding search converts both to vectors and compares angles, so it matches on meaning. This is the retrieval half of <b>RAG</b> (chapter 12): documents are embedded once and stored in a <em>vector database</em>; your question is embedded at query time, and the nearest documents are pasted into the model's context. Every "chat with your PDF" product is this and little else.`),
      ));

      /* ==================== INSIDE AN LLM ==================== */
      root.append(ctx.section('Where embeddings live inside a language model',
        ctx.p(`Open up any GPT-style model and the very first component is an <em>embedding table</em>: a big matrix with one row per token. For GPT-2 small that is 50,257 rows of 768 numbers, about 38.6 million parameters spent on nothing but "what does each token mean before we look at context?". Turning a token ID into a vector is not a computation at all; it is a row lookup.`),
        ctx.p(`Then something important happens that plain word2vec could never do. Word2vec gives "bank" exactly one vector, blending the riverbank and the financial sense into an unhappy average. A transformer's attention layers (chapter 7) update each token's vector using its neighbours, so by the middle layers "bank" in <i>"sat on the bank of the river"</i> has moved somewhere quite different from "bank" in <i>"the bank approved the loan"</i>. These are <em>contextual embeddings</em>, introduced at scale by ELMo and BERT in 2018, and they are why modern models handle ambiguity that word2vec could not.`),
        ctx.p(`At the other end, the model has to turn its final vector back into a word. It compares that vector against every row of a vocabulary matrix, producing one score per token; softmax turns the scores into probabilities. Many models <em>tie</em> this matrix to the input embedding table, reusing the same weights, which saves memory and usually helps. So a language model is bracketed by embeddings: a lookup on the way in, a comparison against the same table on the way out.`),
        ctx.callout('key', '🔑 Why this matters for everything after here',
          `Embeddings are the interface between the messy world and the maths. Text, images, audio, clicks, molecules: whatever you want a model to handle, the job is to get it into a vector space. Everything downstream, attention included, is operations on those vectors. If you understand that a model manipulates <b>positions in a meaning space</b>, the transformer in the next chapter is a much smaller step than it looks.`),
      ));

      /* ==================== BIAS ==================== */
      root.append(ctx.section('The space learns our prejudices too',
        ctx.p(`If directions in the space capture real regularities in text, they also capture the unpleasant ones. The 2016 paper <i>Man is to Computer Programmer as Woman is to Homemaker?</i> showed that word2vec vectors trained on Google News produced exactly that analogy. The same geometry that gives you king − man + woman = queen gives you doctor − man + woman = nurse.`),
        ctx.p(`This is not a bug in the algorithm. The algorithm faithfully recorded a statistical regularity in how people write. That is precisely why it matters: an embedding is a measurement of a corpus, and if you use it to screen CVs or rank job ads, you have automated and laundered the bias in the text. Debiasing techniques exist, most of them projecting out an identified direction, and none of them fully work. The durable lesson is that <b>a model's values are downstream of its data</b>, which is the same problem chapter 11 tackles with an enormous amount of deliberate human feedback.`),
      ));

      /* ==================== VISUALISING HIGH-D ==================== */
      root.append(ctx.section('A note on 300 dimensions',
        ctx.p(`Real embeddings have 300 to 4,096 dimensions and nobody can picture that. Three tools help. <em>PCA</em>, used in the map above, is a rigid rotation that keeps the directions of greatest spread; it preserves global structure and is fully reversible in spirit. <em>t-SNE</em> (2008) and <em>UMAP</em> (2018) are non-linear and optimise for keeping <b>neighbours</b> together, which produces gorgeous, cluster-rich pictures.`),
        ctx.callout('warning', '⚠️ Read t-SNE plots with suspicion',
          `In a t-SNE or UMAP picture the distances <b>between</b> clusters mean very little, cluster sizes mean very little, and changing the perplexity or seed changes the picture. They are excellent for spotting that structure exists, and unreliable for claims about how far apart two groups are. High-dimensional space is also just strange: in 300 dimensions almost every pair of random vectors is nearly perpendicular, which is exactly why there is room for so many distinct meanings.`),
      ));

      /* ==================== QUIZ ==================== */
      root.append(ctx.quiz([
        {
          q: 'Why is one-hot encoding a poor representation of word meaning?',
          options: [
            'It uses too much memory to store on modern hardware',
            'Every pair of distinct words is exactly equally far apart, so it encodes no similarity at all',
            'It cannot represent words that appear more than once in a sentence',
            'The numbers are too large for a neural network to process',
          ],
          answer: 1,
          explain: 'One-hot vectors are all mutually perpendicular. "cat" is exactly as close to "dog" as it is to "bureaucracy", so the representation carries no information about meaning. Memory is a secondary annoyance, not the core problem.',
        },
        {
          q: 'What does the distributional hypothesis claim?',
          options: [
            'Words are distributed evenly across a language',
            'Every word has exactly one meaning that can be looked up',
            'Words appearing in similar contexts tend to have similar meanings',
            'Meaning is distributed across the layers of a neural network',
          ],
          answer: 2,
          explain: 'Firth\'s 1957 formulation: "you shall know a word by the company it keeps." It is what makes learning meaning from raw text possible, because context is observable and meaning is not.',
        },
        {
          q: 'Cosine similarity ignores the length of the vectors. Why is that useful?',
          options: [
            'Because it makes the computation faster',
            'Because vector length mostly reflects things like word frequency, not meaning',
            'Because all embedding vectors have the same length anyway',
            'Because negative numbers cannot be handled otherwise',
          ],
          answer: 1,
          explain: 'A frequent word tends to acquire a longer vector. Cosine asks only about direction, so a rare word and a common word with the same meaning still score as similar.',
        },
        {
          q: 'In the live skip-gram demo, why do the fruits cluster together even though the model is never told what a fruit is?',
          options: [
            'The words are alphabetically adjacent in the vocabulary',
            'A separate classifier labels them before training starts',
            'They appear in the same contexts, so similar vectors make the prediction task easier',
            'They have similar spelling, which the model reads character by character',
          ],
          answer: 2,
          explain: 'The only signal is context. Because "apple", "banana" and "cherry" are interchangeable in the corpus, giving them similar vectors lowers the loss. Clustering is a consequence of the prediction objective, not a goal of it.',
        },
        {
          q: 'What can a transformer\'s contextual embeddings do that word2vec vectors cannot?',
          options: [
            'Represent more than 50,000 words',
            'Give the same word different vectors depending on the sentence it appears in',
            'Run without any training data',
            'Guarantee that analogy arithmetic always succeeds',
          ],
          answer: 1,
          explain: 'Word2vec assigns one fixed vector per word, so "bank" is an average of all its senses. Attention layers update each token\'s vector using its neighbours, so the river bank and the financial bank end up in different places.',
        },
      ]));

      /* ==================== GO DEEPER ==================== */
      root.append(ctx.section('Go deeper',
        ctx.ul([
          `<a href="https://arxiv.org/abs/1301.3781" target="_blank" rel="noopener">Efficient Estimation of Word Representations in Vector Space</a> — Mikolov et al., 2013. The original word2vec paper; short and very readable.`,
          `<a href="https://jalammar.github.io/illustrated-word2vec/" target="_blank" rel="noopener">The Illustrated Word2vec</a> — Jay Alammar. The best visual walkthrough of skip-gram and negative sampling anywhere.`,
          `<a href="https://arxiv.org/abs/2103.00020" target="_blank" rel="noopener">Learning Transferable Visual Models From Natural Language Supervision (CLIP)</a> — Radford et al., 2021. One shared embedding space for images and text.`,
          `<a href="https://arxiv.org/abs/1607.06520" target="_blank" rel="noopener">Man is to Computer Programmer as Woman is to Homemaker?</a> — Bolukbasi et al., 2016. Bias in embeddings, and the difficulty of removing it.`,
          `<a href="https://distill.pub/2016/misread-tsne/" target="_blank" rel="noopener">How to Use t-SNE Effectively</a> — Wattenberg et al., Distill. Interactive proof that t-SNE pictures can mislead you.`,
        ]),
      ));
    },
  });

  /* =================================================================
     INTERACTIVE 1 — the meaning map with vector arithmetic
     ================================================================= */
  function buildMap(ctx) {
    const h = ctx.h, C = ctx.colors;
    const W = 760, H = 440;
    const [cv, g] = ctx.canvas(W, H);

    // view transform
    let scale = 150, ox = W / 2, oy = H / 2;
    let hovered = null, dragging = false, lastX = 0, lastY = 0, moved = false;
    let result = null;      // {vec, px, py, best, a, b, c}
    let anim = 0;           // 0..1 animation progress for the arrows

    function toScreen(px, py) { return { x: ox + px * scale, y: oy - py * scale }; }

    const readout = ctx.readout();
    const PRESETS = [
      { label: 'king − man + woman', a: 'king', b: 'man', c: 'woman' },
      { label: 'Paris − France + Italy', a: 'Paris', b: 'France', c: 'Italy' },
      { label: 'prince − boy + girl', a: 'prince', b: 'boy', c: 'girl' },
      { label: 'Tokyo − Japan + Germany', a: 'Tokyo', b: 'Japan', c: 'Germany' },
    ];
    const names = WORDS.map(w => w.w).sort();
    const selA = ctx.select({ label: 'A', options: names, value: 'king', onChange: () => compute() });
    const selB = ctx.select({ label: '− B', options: names, value: 'man', onChange: () => compute() });
    const selC = ctx.select({ label: '+ C', options: names, value: 'woman', onChange: () => compute() });
    const presetSel = ctx.select({
      label: 'preset analogy',
      options: PRESETS.map((p, i) => ({ value: String(i), label: p.label })),
      value: '0',
      onChange: (v) => {
        const p = PRESETS[+v] || PRESETS[0];
        setSel(selA, p.a); setSel(selB, p.b); setSel(selC, p.c);
        compute();
      },
    });
    function setSel(wrap, val) {
      const s = wrap.querySelector ? wrap.querySelector('select') : null;
      if (s) { s.value = val; }
    }

    function lookup(name) { return WORDS.find(w => w.w === name) || WORDS[0]; }

    function compute() {
      const a = lookup(selA.value), b = lookup(selB.value), c = lookup(selC.value);
      const target = addv(addv(a.v, b.v, -1), c.v, 1);           // a - b + c
      let best = null, bestScore = -2;
      for (const w of WORDS) {
        if (w === a || w === b || w === c) continue;             // exclude the inputs, as word2vec papers do
        const s = cosine(target, w.v);
        if (s > bestScore) { bestScore = s; best = w; }
      }
      const p = project(target);
      result = { a, b, c, px: p.x, py: p.y, best, score: bestScore };
      anim = 0;
      readout.set({
        'query': a.w + ' − ' + b.w + ' + ' + c.w,
        'nearest word': best ? best.w : '—',
        'cosine': bestScore.toFixed(3),
      });
    }

    function nearestTo(word, k) {
      return WORDS.filter(w => w !== word)
        .map(w => ({ w, s: cosine(word.v, w.v) }))
        .sort((p, q) => q.s - p.s).slice(0, k);
    }

    /* pointer interaction: drag to pan, hover to inspect */
    cv.addEventListener('pointerdown', (e) => {
      dragging = true; moved = false;
      const p = cv.pos(e); lastX = p.x; lastY = p.y;
    });
    cv.addEventListener('pointermove', (e) => {
      const p = cv.pos(e);
      if (dragging) {
        ox += p.x - lastX; oy += p.y - lastY; lastX = p.x; lastY = p.y; moved = true;
        hovered = null;
        return;
      }
      // hover detection
      let found = null, bestD = 18;
      for (const w of WORDS) {
        const s = toScreen(w.px, w.py);
        const d = Math.hypot(s.x - p.x, s.y - p.y);
        if (d < bestD) { bestD = d; found = w; }
      }
      hovered = found;
    });
    const endDrag = () => { dragging = false; };
    cv.addEventListener('pointerup', endDrag);
    cv.addEventListener('pointercancel', endDrag);
    cv.addEventListener('pointerleave', () => { dragging = false; hovered = null; });
    cv.addEventListener('wheel', (e) => {
      if (e.preventDefault) e.preventDefault();
      const f = (e.deltaY || 0) > 0 ? 0.9 : 1.1;
      scale = Math.max(40, Math.min(600, scale * f));
    }, { passive: false });

    function draw() {
      g.clearRect(0, 0, W, H);
      g.fillStyle = C.bg; g.fillRect(0, 0, W, H);

      // faint axes
      g.strokeStyle = 'rgba(148,163,184,0.12)'; g.lineWidth = 1;
      g.beginPath(); g.moveTo(0, oy); g.lineTo(W, oy); g.moveTo(ox, 0); g.lineTo(ox, H); g.stroke();

      // arithmetic arrows
      if (result) {
        anim = Math.min(1, anim + 0.03);
        const pa = toScreen(result.a.px, result.a.py);
        const pb = toScreen(result.b.px, result.b.py);
        const pc = toScreen(result.c.px, result.c.py);
        const pr = toScreen(result.px, result.py);
        // stage 1: a -> (a-b)   stage 2: -> (a-b+c)
        const mid = { x: pa.x + (pb.x - pa.x) * 0 + (pa.x - pb.x) * 0, y: 0 };
        // draw as: from A, subtract B's offset, then add C's offset
        const origin = toScreen(0, 0);
        const stepB = { x: pa.x - (pb.x - origin.x), y: pa.y - (pb.y - origin.y) };
        const t1 = Math.min(1, anim * 2), t2 = Math.max(0, anim * 2 - 1);
        arrow(g, pa.x, pa.y, pa.x + (stepB.x - pa.x) * t1, pa.y + (stepB.y - pa.y) * t1, C.danger, '− ' + result.b.w);
        if (t2 > 0) arrow(g, stepB.x, stepB.y, stepB.x + (pr.x - stepB.x) * t2, stepB.y + (pr.y - stepB.y) * t2, C.green, '+ ' + result.c.w);
        if (anim >= 1 && result.best) {
          const pbest = toScreen(result.best.px, result.best.py);
          g.strokeStyle = C.warn; g.lineWidth = 2; g.setLineDash([4, 4]);
          g.beginPath(); g.arc(pbest.x, pbest.y, 16, 0, Math.PI * 2); g.stroke();
          g.setLineDash([]);
        }
        // the computed point
        g.fillStyle = C.warn;
        g.beginPath(); g.arc(pr.x, pr.y, 5, 0, Math.PI * 2); g.fill();
      }

      // hovered neighbour links
      if (hovered) {
        const hs = toScreen(hovered.px, hovered.py);
        for (const n of nearestTo(hovered, 5)) {
          const ns = toScreen(n.w.px, n.w.py);
          g.strokeStyle = 'rgba(56,217,169,' + Math.max(0.12, n.s * 0.55) + ')';
          g.lineWidth = 1 + Math.max(0, n.s) * 2;
          g.beginPath(); g.moveTo(hs.x, hs.y); g.lineTo(ns.x, ns.y); g.stroke();
        }
      }

      // words
      g.font = '12px Inter, system-ui, sans-serif';
      g.textAlign = 'center'; g.textBaseline = 'middle';
      for (const w of WORDS) {
        const s = toScreen(w.px, w.py);
        if (s.x < -60 || s.x > W + 60 || s.y < -30 || s.y > H + 30) continue;
        const isHot = hovered === w;
        const isRes = result && result.best === w && anim >= 1;
        const col = CAT_COLOR[w.c] || C.muted;
        g.fillStyle = col;
        g.beginPath(); g.arc(s.x, s.y, isHot || isRes ? 5 : 3, 0, Math.PI * 2); g.fill();
        g.fillStyle = isHot || isRes ? '#ffffff' : 'rgba(230,235,245,0.72)';
        g.font = (isHot || isRes ? '600 13px' : '12px') + ' Inter, system-ui, sans-serif';
        g.fillText(w.w, s.x, s.y - 12);
      }

      // hover panel
      if (hovered) {
        const list = nearestTo(hovered, 5);
        const bw = 190, bh = 22 + list.length * 17;
        let bx = 12, by = 12;
        g.fillStyle = 'rgba(15,21,32,0.94)'; g.strokeStyle = C.line; g.lineWidth = 1;
        g.beginPath(); g.rect(bx, by, bw, bh); g.fill(); g.stroke();
        g.textAlign = 'left'; g.textBaseline = 'top';
        g.fillStyle = '#fff'; g.font = '600 12px Inter, system-ui, sans-serif';
        g.fillText('nearest to "' + hovered.w + '"', bx + 10, by + 6);
        g.font = '11px JetBrains Mono, monospace';
        list.forEach((n, i) => {
          g.fillStyle = CAT_COLOR[n.w.c] || C.muted;
          g.fillText(n.w.w, bx + 10, by + 24 + i * 17);
          g.fillStyle = C.muted;
          g.fillText(n.s.toFixed(3), bx + 130, by + 24 + i * 17);
        });
        g.textAlign = 'center'; g.textBaseline = 'middle';
      }

      // legend
      g.textAlign = 'left'; g.textBaseline = 'middle';
      g.font = '11px Inter, system-ui, sans-serif';
      let lx = 12;
      for (const [k, col] of Object.entries(CAT_COLOR)) {
        g.fillStyle = col; g.beginPath(); g.arc(lx, H - 14, 4, 0, Math.PI * 2); g.fill();
        g.fillStyle = C.muted; g.fillText(k, lx + 9, H - 13);
        lx += 22 + g.measureText(k).width;
      }
      g.textAlign = 'center'; g.textBaseline = 'middle';
    }

    ctx.loop(draw);
    compute();

    return ctx.figure(cv,
      'A hand-built 10-dimensional embedding space, projected to 2-D with PCA computed in your browser. Colours mark categories for your benefit only; the model of this space has no idea they exist. The analogy result excludes the three input words, which is how the original word2vec evaluations were scored.',
      [presetSel, selA, selB, selC,
       ctx.button('Run arithmetic', () => compute(), 'primary'),
       ctx.button('Reset view', () => { scale = 150; ox = W / 2; oy = H / 2; })],
      readout);
  }

  function arrow(g, x1, y1, x2, y2, color, label) {
    if (!isFinite(x1) || !isFinite(x2)) return;
    g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 2;
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
    const a = Math.atan2(y2 - y1, x2 - x1), len = Math.hypot(x2 - x1, y2 - y1);
    if (len > 8) {
      g.beginPath();
      g.moveTo(x2, y2);
      g.lineTo(x2 - 9 * Math.cos(a - 0.4), y2 - 9 * Math.sin(a - 0.4));
      g.lineTo(x2 - 9 * Math.cos(a + 0.4), y2 - 9 * Math.sin(a + 0.4));
      g.closePath(); g.fill();
      if (label) {
        g.font = '600 11px Inter, system-ui, sans-serif';
        g.textAlign = 'center'; g.textBaseline = 'middle';
        g.fillText(label, (x1 + x2) / 2, (y1 + y2) / 2 - 10);
      }
    }
  }

  /* =================================================================
     INTERACTIVE 2 — cosine similarity with two draggable arrows
     ================================================================= */
  function buildCosine(ctx) {
    const C = ctx.colors;
    const W = 720, H = 320;
    const [cv, g] = ctx.canvas(W, H);
    const cx = W / 2, cy = H / 2, unit = 90;
    let a = { x: 1.6, y: 0.9 }, b = { x: 1.9, y: -0.4 };
    let drag = null;
    const readout = ctx.readout();

    function toS(v) { return { x: cx + v.x * unit, y: cy - v.y * unit }; }
    function fromS(p) { return { x: (p.x - cx) / unit, y: -(p.y - cy) / unit }; }

    cv.addEventListener('pointerdown', (e) => {
      const p = cv.pos(e);
      const da = Math.hypot(toS(a).x - p.x, toS(a).y - p.y);
      const db = Math.hypot(toS(b).x - p.x, toS(b).y - p.y);
      if (da < 28 && da <= db) drag = 'a'; else if (db < 28) drag = 'b'; else drag = null;
    });
    cv.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const v = fromS(cv.pos(e));
      const lim = (t) => Math.max(-2.6, Math.min(2.6, t));
      const nv = { x: lim(v.x), y: lim(v.y) };
      if (Math.hypot(nv.x, nv.y) < 0.15) return;   // never let a vector collapse to zero
      if (drag === 'a') a = nv; else b = nv;
    });
    const stop = () => { drag = null; };
    cv.addEventListener('pointerup', stop);
    cv.addEventListener('pointerleave', stop);
    cv.addEventListener('pointercancel', stop);

    function draw() {
      g.clearRect(0, 0, W, H);
      g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
      // grid
      g.strokeStyle = 'rgba(148,163,184,0.10)'; g.lineWidth = 1;
      for (let i = -3; i <= 3; i++) {
        g.beginPath(); g.moveTo(cx + i * unit, 0); g.lineTo(cx + i * unit, H); g.stroke();
        g.beginPath(); g.moveTo(0, cy + i * unit); g.lineTo(W, cy + i * unit); g.stroke();
      }
      g.strokeStyle = 'rgba(148,163,184,0.28)';
      g.beginPath(); g.moveTo(0, cy); g.lineTo(W, cy); g.moveTo(cx, 0); g.lineTo(cx, H); g.stroke();

      // angle wedge
      const angA = Math.atan2(a.y, a.x), angB = Math.atan2(b.y, b.x);
      g.fillStyle = 'rgba(124,156,255,0.16)';
      g.beginPath(); g.moveTo(cx, cy);
      g.arc(cx, cy, 46, -angA, -angB, angA < angB);
      g.closePath(); g.fill();

      const sa = toS(a), sb = toS(b);
      arrow(g, cx, cy, sa.x, sa.y, C.accent, '');
      arrow(g, cx, cy, sb.x, sb.y, C.green, '');
      g.fillStyle = C.accent; g.beginPath(); g.arc(sa.x, sa.y, 7, 0, Math.PI * 2); g.fill();
      g.fillStyle = C.green;  g.beginPath(); g.arc(sb.x, sb.y, 7, 0, Math.PI * 2); g.fill();
      g.font = '600 13px Inter, system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillStyle = '#0b0f17'; g.fillText('a', sa.x, sa.y); g.fillText('b', sb.x, sb.y);

      const cs = cosine([a.x, a.y], [b.x, b.y]);
      const deg = Math.acos(Math.max(-1, Math.min(1, cs))) * 180 / Math.PI;

      // panel
      g.textAlign = 'left'; g.textBaseline = 'top';
      g.font = '12px JetBrains Mono, monospace';
      const lines = [
        'a = [' + a.x.toFixed(2) + ', ' + a.y.toFixed(2) + ']   |a| = ' + Math.hypot(a.x, a.y).toFixed(2),
        'b = [' + b.x.toFixed(2) + ', ' + b.y.toFixed(2) + ']   |b| = ' + Math.hypot(b.x, b.y).toFixed(2),
        'a·b = ' + (a.x * b.x + a.y * b.y).toFixed(3),
        'cos = ' + cs.toFixed(3) + '   angle = ' + deg.toFixed(1) + '°',
      ];
      g.fillStyle = 'rgba(15,21,32,0.9)'; g.fillRect(12, 12, 260, 12 + lines.length * 18);
      lines.forEach((l, i) => { g.fillStyle = i === 3 ? C.warn : C.muted; g.fillText(l, 22, 20 + i * 18); });

      // verdict
      g.font = '600 14px Inter, system-ui, sans-serif'; g.textAlign = 'right';
      g.fillStyle = cs > 0.7 ? C.green : cs > 0.2 ? C.warn : cs > -0.2 ? C.muted : C.danger;
      g.fillText(cs > 0.7 ? 'very similar' : cs > 0.2 ? 'somewhat related' : cs > -0.2 ? 'unrelated' : 'opposite', W - 20, 20);
      g.textAlign = 'center'; g.textBaseline = 'middle';

      readout.set({ 'cosine': cs.toFixed(3), 'angle': deg.toFixed(1) + '°' });
    }
    ctx.loop(draw);

    return ctx.figure(cv,
      'Drag either arrowhead. Cosine similarity depends only on the angle between the vectors, never on their lengths, which is exactly why it is the standard measure for embeddings.',
      [ctx.button('Make them identical', () => { b = { x: a.x, y: a.y }; }),
       ctx.button('Make them perpendicular', () => { b = { x: -a.y, y: a.x }; }),
       ctx.button('Make them opposite', () => { b = { x: -a.x, y: -a.y }; })],
      readout);
  }

  /* =================================================================
     INTERACTIVE 3 — live skip-gram (word2vec) training in 2-D
     ================================================================= */
  function buildWord2Vec(ctx) {
    const C = ctx.colors;
    const W = 720, H = 400;
    const [cv, g] = ctx.canvas(W, H);

    // Toy corpus: template sentences that make groups interchangeable.
    const CORPUS = [
      'i ate the apple', 'i ate the banana', 'i ate the cherry', 'i ate the mango',
      'she ate the apple', 'she ate the banana', 'he ate the cherry', 'he ate the mango',
      'the apple was sweet', 'the banana was sweet', 'the cherry was sweet', 'the mango was sweet',
      'the dog ran fast', 'the cat ran fast', 'the horse ran fast', 'the wolf ran fast',
      'i saw a dog', 'i saw a cat', 'i saw a horse', 'i saw a wolf',
      'the dog was loud', 'the cat was loud', 'the horse was loud', 'the wolf was loud',
      'the car drove fast', 'the truck drove fast', 'the train drove fast', 'the bus drove fast',
      'i drove the car', 'i drove the truck', 'i drove the bus', 'she drove the truck',
      'the car was red', 'the truck was red', 'the train was red', 'the bus was red',
    ];
    // build vocab
    const tokens = CORPUS.map(s => s.split(' '));
    const vocab = [];
    const idx = {};
    for (const t of tokens) for (const w of t) if (!(w in idx)) { idx[w] = vocab.length; vocab.push(w); }
    const V = vocab.length;
    const GROUP = {};
    'apple banana cherry mango'.split(' ').forEach(w => GROUP[w] = 'fruit');
    'dog cat horse wolf'.split(' ').forEach(w => GROUP[w] = 'animal');
    'car truck train bus'.split(' ').forEach(w => GROUP[w] = 'vehicle');
    const GC = { fruit: C.warn, animal: C.green, vehicle: C.accent };

    // training pairs (skip-gram, window 2)
    const pairs = [];
    for (const t of tokens) {
      for (let i = 0; i < t.length; i++) {
        for (let j = Math.max(0, i - 2); j <= Math.min(t.length - 1, i + 2); j++) {
          if (i !== j) pairs.push([idx[t[i]], idx[t[j]]]);
        }
      }
    }

    let Win, Wout, step, seed;
    function rnd() { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; }
    function reset() {
      seed = 12345; step = 0;
      Win = []; Wout = [];
      for (let i = 0; i < V; i++) {
        Win.push([(rnd() - 0.5) * 1.0, (rnd() - 0.5) * 1.0]);
        Wout.push([(rnd() - 0.5) * 1.0, (rnd() - 0.5) * 1.0]);
      }
      lossEMA = null; curve.length = 0;
    }
    let lossEMA = null;
    const curve = [];
    reset();

    let running = false;
    const lrS = ctx.slider({ label: 'learning rate', min: 0.005, max: 0.25, step: 0.005, value: 0.06, digits: 3 });
    const readout = ctx.readout();

    function trainStep() {
      const lr = lrS.value;
      const [c, o] = pairs[Math.floor(rnd() * pairs.length)];
      const vc = Win[c];
      // softmax over the whole (tiny) vocabulary
      const scores = new Array(V);
      let mx = -Infinity;
      for (let k = 0; k < V; k++) { scores[k] = vc[0] * Wout[k][0] + vc[1] * Wout[k][1]; if (scores[k] > mx) mx = scores[k]; }
      let sum = 0;
      for (let k = 0; k < V; k++) { scores[k] = Math.exp(scores[k] - mx); sum += scores[k]; }
      const p = scores.map(s => s / sum);
      const loss = -Math.log(Math.max(1e-12, p[o]));
      lossEMA = lossEMA == null ? loss : lossEMA * 0.995 + loss * 0.005;
      // gradients
      const dvc = [0, 0];
      for (let k = 0; k < V; k++) {
        const gk = p[k] - (k === o ? 1 : 0);
        dvc[0] += gk * Wout[k][0]; dvc[1] += gk * Wout[k][1];
        Wout[k][0] -= lr * gk * vc[0];
        Wout[k][1] -= lr * gk * vc[1];
      }
      vc[0] -= lr * dvc[0]; vc[1] -= lr * dvc[1];
      // keep everything finite and bounded
      for (const M of [Win, Wout]) for (const v of M) {
        for (let d = 0; d < 2; d++) { if (!isFinite(v[d])) v[d] = (rnd() - 0.5) * 0.5; v[d] = Math.max(-8, Math.min(8, v[d])); }
      }
      step++;
    }

    function draw() {
      if (running) { for (let i = 0; i < 400; i++) trainStep(); if (step % 400 === 0) { curve.push(lossEMA || 0); if (curve.length > 240) curve.shift(); } }

      g.clearRect(0, 0, W, H);
      g.fillStyle = C.bg; g.fillRect(0, 0, W, H);

      // auto-fit the view to the embedding cloud
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const v of Win) { minX = Math.min(minX, v[0]); maxX = Math.max(maxX, v[0]); minY = Math.min(minY, v[1]); maxY = Math.max(maxY, v[1]); }
      const padX = Math.max(0.4, (maxX - minX) * 0.15), padY = Math.max(0.4, (maxY - minY) * 0.15);
      minX -= padX; maxX += padX; minY -= padY; maxY += padY;
      const plotW = W - 210;
      const sx = (x) => 20 + (x - minX) / Math.max(1e-6, maxX - minX) * (plotW - 40);
      const sy = (y) => H - 30 - (y - minY) / Math.max(1e-6, maxY - minY) * (H - 60);

      g.font = '11px Inter, system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
      for (let i = 0; i < V; i++) {
        const w = vocab[i], grp = GROUP[w];
        const x = sx(Win[i][0]), y = sy(Win[i][1]);
        if (!isFinite(x) || !isFinite(y)) continue;
        g.fillStyle = grp ? GC[grp] : 'rgba(148,163,184,0.55)';
        g.beginPath(); g.arc(x, y, grp ? 4 : 2.5, 0, Math.PI * 2); g.fill();
        g.font = (grp ? '600 12px' : '10px') + ' Inter, system-ui, sans-serif';
        g.fillStyle = grp ? GC[grp] : 'rgba(148,163,184,0.5)';
        g.fillText(w, x, y - 11);
      }

      // loss curve panel
      const px = W - 180, py = 24, pw = 158, ph = 96;
      g.fillStyle = 'rgba(15,21,32,0.92)'; g.strokeStyle = C.line; g.lineWidth = 1;
      g.beginPath(); g.rect(px, py, pw, ph); g.fill(); g.stroke();
      g.fillStyle = C.muted; g.font = '10px Inter, system-ui, sans-serif';
      g.textAlign = 'left'; g.textBaseline = 'top';
      g.fillText('loss', px + 8, py + 6);
      if (curve.length > 1) {
        const lo = Math.min.apply(null, curve), hi = Math.max.apply(null, curve);
        g.strokeStyle = C.green; g.lineWidth = 1.5; g.beginPath();
        curve.forEach((v, i) => {
          const X = px + 8 + i / Math.max(1, curve.length - 1) * (pw - 16);
          const Y = py + ph - 10 - (v - lo) / Math.max(1e-6, hi - lo) * (ph - 28);
          i ? g.lineTo(X, Y) : g.moveTo(X, Y);
        });
        g.stroke();
      }
      // legend
      g.font = '11px Inter, system-ui, sans-serif';
      let ly = py + ph + 14;
      for (const [k, col] of Object.entries(GC)) {
        g.fillStyle = col; g.beginPath(); g.arc(px + 12, ly, 4, 0, Math.PI * 2); g.fill();
        g.fillStyle = C.muted; g.fillText(k, px + 22, ly - 5);
        ly += 18;
      }
      g.fillStyle = 'rgba(148,163,184,0.5)'; g.beginPath(); g.arc(px + 12, ly, 3, 0, Math.PI * 2); g.fill();
      g.fillStyle = C.muted; g.fillText('other words', px + 22, ly - 5);
      g.textAlign = 'center'; g.textBaseline = 'middle';

      readout.set({
        'steps': step,
        'loss': lossEMA == null ? '—' : lossEMA.toFixed(3),
        'vocab': V,
        'parameters': V * 2 * 2,
        'state': running ? 'training' : 'paused',
      });
    }
    ctx.loop(draw);

    const playBtn = ctx.button('Train', () => { running = !running; playBtn.textContent = running ? 'Pause' : 'Train'; }, 'primary');

    return ctx.figure(cv,
      'A real skip-gram model with 2-dimensional embeddings, trained live by stochastic gradient descent on the template corpus. Only the coloured words belong to the three semantic groups; the model is never told that. Because the whole space is free to rotate and translate, each run settles into a different arrangement, and only the relative positions carry meaning.',
      [playBtn,
       ctx.button('Reset', () => { reset(); }),
       ctx.button('1000 steps', () => { for (let i = 0; i < 1000; i++) trainStep(); }),
       lrS],
      readout);
  }
})();
