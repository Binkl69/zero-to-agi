/* Zero → AGI · Chapter 06 · Meaning as geometry: embeddings
   DESIGN RULE: the reader scores two words three ways and watches the two obvious schemes fail
   before any theory arrives. Every paragraph explains something they already did.
   Interactives, in order: three-ways word comparison (ID vs one-hot vs learned); the
   distributional hypothesis as a guessing game over five contexts; PCA map with analogy
   arithmetic; cosine-similarity arrows; live skip-gram training; embedding-table parameter cost
   with weight tying; contextual embeddings pulling one word into two meanings. */
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
  /* Which words PCA is fitted to is not a detail — it decides what the picture
     can show. Fitted to all 36 words, the two directions of greatest spread are
     "what kind of thing is this" and "city or country", and gender lands in a
     third direction the screen does not have: king and queen then print on the
     same pixel and the analogy this chapter is built on becomes invisible. The
     famous word2vec figures are drawn from a handful of related words for
     exactly this reason, so the map lets the reader choose the family and
     refits. That choice is the lesson, not a workaround. */
  const FAMILIES = {
    people: { label: 'people & royalty', has: (w) => w.c === 'people' },
    place: { label: 'countries & capitals', has: (w) => w.c === 'place' },
    things: { label: 'animals, food & vehicles', has: (w) => w.c === 'animal' || w.c === 'food' || w.c === 'vehicle' },
    all: { label: 'all 36 words at once', has: () => true },
  };
  let PROJ = null;
  function project(v) {
    const c = v.map((x, i) => x - PROJ.mean[i]);
    return { x: dot(c, PROJ.a1), y: dot(c, PROJ.a2) };
  }
  function refit(fam) {
    const sub = WORDS.filter(FAMILIES[fam].has);
    PROJ = pca2(sub.map(w => w.v));
    /* scale so the fitted family fills a comparable area whichever one it is */
    let m = 0;
    for (const w of sub) { const p = project(w.v); m = Math.max(m, Math.abs(p.x), Math.abs(p.y)); }
    const k = m > 1e-9 ? 1 / m : 1;
    WORDS.forEach(w => { const p = project(w.v); w.px = p.x * k; w.py = p.y * k; });
  }
  refit('people');

  ZTA.registerChapter({
    id: '06-embeddings',
    num: 6,
    part: 'II',
    title: 'Meaning as geometry: embeddings',
    tagline: 'Score two words three ways, watch the obvious approaches fail, then see why turning meaning into geometry is the most reusable idea in modern AI.',

    render(root, ctx) {
      const h = ctx.h, C = ctx.colors;
      const p = ctx.p, section = ctx.section, callout = ctx.callout, ul = ctx.ul;

      /* ---------- open with the reader breaking the obvious approach ---------- */
      root.append(
        callout('tryit', '🖐 Do this first — measure how alike two words are, three different ways',
          `A computer cannot store the word "cat". It stores numbers. Below are the three ways of choosing those numbers, scored side by side.<br>
           <b>1.</b> Press <b>cat vs catalogue</b>. The numbering scheme calls them <b>almost identical</b>. They share four letters and nothing else.<br>
           <b>2.</b> Press <b>cat vs dog</b>. Now the numbering calls them miles apart, and one-hot calls them exactly as unrelated as <b>cat vs democracy</b>.<br>
           <b>3.</b> Work down the preset buttons and watch only the bottom row ever agree with your own judgement.`),
        buildThreeWays(ctx),
        p(`Two different failures there, and they fail in opposite directions.`),
      );

      root.append(section('Why both obvious ideas break',
        p(`Numbering the words alphabetically — <code class="inline">aardvark = 1, … cat = 3312, catalogue = 3313, … zebra = 50000</code> — smuggles in an order that has nothing to do with meaning. It asserts that <code class="inline">cat</code> is nearly <code class="inline">catalogue</code>, and that <code class="inline">apple</code> is 3,310 units from <code class="inline">cat</code> but 1 unit from <code class="inline">apply</code>. That is false information, confidently stated.`),
        p(`The standard fix is <em>one-hot encoding</em>: give every word its own axis. With a 50,000-word vocabulary, "cat" becomes 50,000 zeros with a single 1 in slot 3312. Now nothing is accidentally close to anything.`),
        p(`But look at the cost. <b>Every pair of distinct words is now exactly equally far apart.</b> "cat" and "dog" are as unrelated as "cat" and "bureaucracy". We removed the false information and replaced it with <i>no</i> information.`),
        callout('key', '🔑 The key idea',
          `Both failures come from <b>us choosing</b> the numbers. The answer is to stop choosing and start <b>learning</b> them:
           let the numbers be parameters, and let a training task push them into a useful arrangement.<br>
           Meaning is not something we encode. It is something that falls out of prediction.`),
      ));

      root.append(section('You already do this, and you do it from context alone',
        p(`In 1957 the linguist J. R. Firth wrote the sentence this whole field rests on: <i>you shall know a word by the company it keeps.</i>`),
        callout('tryit', '🖐 Try this — work out a word you have never seen',
          `<b>1.</b> Start with only the <b>first</b> context sentence switched on. Look at the bar chart: several candidates are plausible and the model has no idea.<br>
           <b>2.</b> Switch the sentences on one at a time. Watch the bars separate as the contexts pile up.<br>
           <b>3.</b> With all five on, one answer is clearly ahead — and <b>nobody ever defined the word.</b> You inferred it purely from the company it keeps, and so did the bar chart.`),
        buildTesguino(ctx),
        p(`That is the <em>distributional hypothesis</em>: words appearing in similar contexts have similar meanings. Notice you could also tell that tesgüino is more like <i>beer</i> than like <i>hammer</i>, because beer turns up in the same kinds of sentences.`),
        p(`This turns a philosophical problem into an engineering one. We do not need to teach a machine what "beer" means. We need only give it a prediction task involving context, and force it to compress what it learns into a short list of numbers.`),
        p(`Whatever arrangement of numbers makes that prediction easiest will necessarily put beer near wine — because they are interchangeable in text.`),
        callout('history', '📜 Where this came from',
          `In 2013 Tomáš Mikolov and colleagues at Google published <b>word2vec</b>, which made this practical at scale.
           The training task was almost insultingly simple: given a word, predict the words around it (skip-gram), or the reverse (CBOW).
           Trained on billions of words it produced 300-number vectors whose geometry stunned people.
           Earlier work had similar ideas — latent semantic analysis in the late 1980s, Bengio's neural language model in 2003 — but word2vec was fast enough, and the results legible enough, to change what everyone built next.`),
      ));

      root.append(section('The map, and the famous piece of arithmetic',
        p(`Below is a small hand-built embedding space: 36 words, each a list of 10 numbers. The picture is a genuine <em>PCA projection</em> of those 10-dimensional vectors down to the 2 your screen has, computed in your browser when this page loaded. PCA finds the two directions along which the points spread out most, so it keeps as much structure as a flat picture can hold.`),
        callout('tryit', '🖐 Try this',
          `<b>1.</b> Hover a word to see its five nearest neighbours by cosine similarity. The clusters formed themselves out of the numbers — nobody drew the groups.<br>
           <b>2.</b> Drag to pan, scroll or pinch to zoom.<br>
           <b>3.</b> Now the famous part. In the arithmetic row, run <b>king − man + woman</b>. Watch the arrows. The answer is not looked up anywhere; it is the nearest word to a point computed by adding and subtracting vectors.<br>
           <b>4.</b> Try the other presets, then build your own.`),
        buildMap(ctx),
        p(`What happens in that arithmetic is worth stating precisely, because it is easy to over-mystify. <b>Nothing in training said "encode gender."</b>`),
        p(`But if the space is arranged so that context is predictable, then whatever distinguishes <i>king</i> from <i>queen</i> must also distinguish <i>man</i> from <i>woman</i> — those pairs are swapped in the same kinds of sentences. The difference gets stored as a <b>direction</b>.`),
        p(`Subtracting <i>man</i> and adding <i>woman</i> is a translation along that direction. Directions in the space turn out to mean things: there is a gender direction, a plural direction, a past-tense direction, a capital-city direction.`),
        callout('warning', '⚠️ An honest caveat',
          `The analogy result is real but often oversold. In published word2vec results the query vector's own inputs are excluded from the answer, and many analogies fail outside a curated set.
           The space here is hand-designed so the preset analogies come out exactly; a trained space is messier.
           The <b>directions-carry-meaning</b> insight is solid. The <b>arithmetic always works</b> claim is not.`),
      ));

      root.append(section('Measuring "near"',
        p(`"Near each other" needs a definition. The one nearly everyone uses is <em>cosine similarity</em>: the cosine of the angle between two vectors. It ignores length entirely and asks only whether two vectors point the same way.`),
        p(`That matters because in text a common word gets a long vector and a rare word a short one, and we do not want frequency masquerading as meaning.`),
        callout('tryit', '🖐 Try this',
          `Drag either arrowhead. The cosine depends only on the <b>angle</b>: make one arrow twice as long and the number does not move.<br>
           Then set them 90° apart and note the score of exactly zero. That is what "unrelated" means numerically.`),
        buildCosine(ctx),
        p(`The scale runs from <b>+1</b> (same direction, same meaning) through <b>0</b> (perpendicular, unrelated) to <b>−1</b> (opposite). In a real embedding space "cat" and "dog" sit around 0.8, "cat" and "democracy" around 0.05.`),
      ));

      root.append(section('Now let a model find the numbers itself',
        p(`Everything so far used vectors written by hand. Below is a real <em>skip-gram</em> model — the word2vec training task — running in your browser. It takes a centre word, tries to predict the words around it, and adjusts each word's 2 numbers by gradient descent when it is wrong.`),
        callout('tryit', '🖐 Try this',
          `<b>1.</b> Press <b>Train</b>. The words start scattered at random. Within a few thousand steps the fruits pull together, the animals pull together, the vehicles pull together.<br>
           <b>2.</b> Push the learning rate up and watch the points thrash and overshoot; drop it low and watch progress crawl. Those are chapter 3's failures, in a real training run.<br>
           <b>3.</b> Press <b>Reset</b> and run again. The clusters re-form, <b>but in different positions and orientations</b> — the absolute coordinates mean nothing, only the relative geometry does.`),
        buildWord2Vec(ctx),
        p(`Crucially, <b>the model is never told which words are foods or vehicles.</b> It only ever sees which words appear near which. The clusters are entirely a consequence of shared context.`),
        p(`And nothing in that recipe was about language. The recipe was: <b>define a prediction task, force the thing through a narrow layer of numbers, and the geometry of that layer becomes a similarity space.</b> Swap in songs, products, molecules or users and it still works.`),
      ));

      root.append(section('How a language model actually uses this',
        p(`Open any GPT-style model and the very first component is an <em>embedding table</em>: a big matrix with one row per token. Turning a token ID into a vector is not a computation at all — it is a row lookup.`),
        callout('tryit', '🖐 Try this',
          `<b>1.</b> The defaults are GPT-2 small: 50,257 tokens × 768 numbers. Read the total — about <b>38.6 million</b> parameters spent purely on "what does each token mean <i>before</i> we look at context?"<br>
           <b>2.</b> Drag the dimension up to 4,096 and the vocabulary to 128,000, roughly a modern frontier model. The table alone passes half a billion.<br>
           <b>3.</b> Toggle <b>tie input and output</b> and watch the total halve. That is a real trick used in real models.`),
        buildEmbedTable(ctx),
        p(`At the other end the model must turn its final vector back into a word. It compares that vector against every row of a vocabulary matrix, producing one score per token, and softmax turns the scores into probabilities. Many models <em>tie</em> that matrix to the input table, reusing the same weights.`),
        p(`So a language model is bracketed by embeddings: a lookup on the way in, a comparison against the same table on the way out.`),
      ));

      root.append(section('The upgrade word2vec could never make',
        p(`Word2vec gives "bank" exactly one vector, blending the riverbank and the financial sense into an unhappy average. That is a hard ceiling: the word has one row in the table, so it has one meaning.`),
        callout('tryit', '🖐 Try this — watch one word become two',
          `<b>1.</b> At <b>layer 0</b> the two "bank" dots sit exactly on top of each other. They must: both are the same row of the same table, and nothing has looked at the sentence yet.<br>
           <b>2.</b> Drag <b>layer</b> upward. The dots separate, each drifting toward the company it keeps.<br>
           <b>3.</b> By the top layers one sits near <i>river</i> and <i>water</i>, the other near <i>loan</i> and <i>money</i>. Same word, same starting row, two different meanings — <b>resolved entirely by context</b>.`),
        buildContextBank(ctx),
        p(`A transformer's attention layers (chapter 7) update each token's vector using its neighbours, so by the middle layers "bank" in <i>"sat on the bank of the river"</i> has moved somewhere quite different from "bank" in <i>"the bank approved the loan"</i>.`),
        p(`These are <em>contextual embeddings</em>, introduced at scale by ELMo and BERT in 2018, and they are why modern models handle ambiguity that word2vec simply could not. It is also the cleanest one-line statement of what attention <i>does</i>: it moves each word's vector based on the words around it.`),
      ));

      root.append(section('The uncomfortable part',
        p(`If directions in the space capture real regularities in text, they also capture the unpleasant ones. The 2016 paper <i>Man is to Computer Programmer as Woman is to Homemaker?</i> showed that word2vec vectors trained on Google News produced exactly that analogy.`),
        p(`The same geometry that gives you king − man + woman = queen gives you doctor − man + woman = nurse. It is the identical mechanism; you cannot keep one and discard the other by adjusting the algorithm.`),
        p(`This is not a bug. The algorithm faithfully recorded a statistical regularity in how people write. That is precisely why it matters: an embedding is a <b>measurement of a corpus</b>, and if you use it to screen CVs or rank job ads, you have automated and laundered the bias in the text.`),
        p(`Debiasing techniques exist, most of them projecting out an identified direction, and none of them fully work — the information tends to survive in other directions. The durable lesson is that <b>a model's values are downstream of its data</b>, which is the same problem chapter 11 tackles with an enormous amount of deliberate human feedback.`),
      ));

      root.append(section('Why this matters for everything after here',
        callout('example', '🌍 Where you have used this today',
          `<b>Semantic search.</b> Keyword search will not match "cheap notebooks" to a page about "affordable laptops"; embedding search converts both to vectors and compares angles, so it matches on meaning.
           This is the retrieval half of <b>RAG</b> (chapter 12): documents are embedded once and stored in a <em>vector database</em>, your question is embedded at query time, and the nearest documents are pasted into the model's context.<br>
           <b>Recommendations.</b> Songs, films and products get embedded from who consumes them together, so "more like this" is a nearest-neighbour lookup.<br>
           <b>Deduplication, moderation, clustering.</b> Anywhere you need "are these two things basically the same?", it is a cosine away.`),
        p(`Real embeddings have 300 to 4,096 dimensions and nobody can picture that. Three tools help. <em>PCA</em>, used in the map above, is a rigid rotation keeping the directions of greatest spread, so it preserves global structure. <em>t-SNE</em> (2008) and <em>UMAP</em> (2018) are non-linear and optimise for keeping <b>neighbours</b> together, which produces gorgeous, cluster-rich pictures.`),
        callout('warning', '⚠️ Read t-SNE plots with suspicion',
          `In a t-SNE or UMAP picture, the distances <b>between</b> clusters mean very little, cluster sizes mean very little, and changing the perplexity or the random seed changes the picture.
           They are excellent for spotting that structure exists and unreliable for claims about how far apart two groups are.<br>
           High-dimensional space is also just strange: in 300 dimensions almost every pair of random vectors is nearly perpendicular, which is exactly why there is room for so many distinct meanings to sit without colliding.`),
        callout('key', '🔑 Why this matters for everything after here',
          `Embeddings are the interface between the messy world and the maths. Text, images, audio, clicks, molecules: whatever you want a model to handle, the job is to get it into a vector space.
           Everything downstream, attention included, is operations on those vectors.
           If you understand that a model manipulates <b>positions in a meaning space</b>, the transformer in the next chapter is a much smaller step than it looks.`),
        p(`One picture to keep: <b>an embedding turns "are these alike?" into "how far apart are these?", and every model from here on is moving points around in that space.</b>`),
      ));

      root.append(ctx.quiz([
        { q: 'Why is numbering words alphabetically (cat = 3312, catalogue = 3313) worse than useless?', options: ['It uses too much memory', 'It asserts a similarity structure that is real but wrong — cat ends up adjacent to catalogue and far from dog', 'Numbers cannot represent words', 'It only works for English'], answer: 1, explain: 'You saw it in the opening demo: the numbering scheme confidently reports cat and catalogue as near-identical. One-hot encoding fixes that by making every pair equally distant — which removes the false information and leaves none at all.' },
        { q: 'What is the distributional hypothesis?', options: ['Words are distributed evenly through a document', 'Words appearing in similar contexts have similar meanings', 'Every word needs its own dimension', 'Meaning is stored in a dictionary the model memorises'], answer: 1, explain: 'It is what let you work out "tesgüino" from five sentences without a definition. It also turns a philosophical question into an engineering one: give a model a context-prediction task, force it through a narrow layer, and similar words must end up near each other.' },
        { q: 'Why does cosine similarity ignore vector length?', options: ['Lengths are too expensive to compute', 'Because in text a common word gets a long vector and a rare one a short vector, and frequency should not masquerade as meaning', 'Because all embeddings have length 1', 'It does not — cosine uses length'], answer: 1, explain: 'Cosine asks only whether two vectors point the same way. You can verify it on the arrows: double one arrow\'s length and the score does not move.' },
        { q: 'In the contextual-embedding demo, why do the two "bank" dots sit exactly on top of each other at layer 0?', options: ['A bug in the visualisation', 'At layer 0 the word is just a row lookup from the embedding table — nothing has looked at the sentence yet, so both copies are identical', 'Because the two sentences are the same length', 'Because bank has only one meaning'], answer: 1, explain: 'That is the whole limitation of word2vec: one row per word, so one meaning per word. Attention is what moves each token\'s vector based on its neighbours, which is why the dots separate as you climb the layers.' },
        { q: 'word2vec trained on news text produced "man is to computer programmer as woman is to homemaker". What does that tell you?', options: ['The algorithm has a bug that debiasing fixes completely', 'The algorithm faithfully recorded a statistical regularity in how people write — an embedding is a measurement of its corpus', 'Embeddings should not be used for search', 'The training ran for too long'], answer: 1, explain: 'It is the same mechanism that produces king − man + woman = queen, so you cannot keep one and drop the other by changing the algorithm. Debiasing methods project out an identified direction and none fully work. A model\'s values are downstream of its data.' },
      ]));

      root.append(section('Go deeper',
        ul([
          `<a href="https://jalammar.github.io/illustrated-word2vec/" target="_blank" rel="noopener">Jay Alammar, "The Illustrated Word2vec"</a> — the clearest visual walkthrough of the training task you just ran.`,
          `<a href="https://arxiv.org/abs/1301.3781" target="_blank" rel="noopener">Mikolov et al. (2013), "Efficient Estimation of Word Representations in Vector Space"</a> — the original word2vec paper.`,
          `<a href="https://nlp.stanford.edu/projects/glove/" target="_blank" rel="noopener">GloVe (Pennington, Socher &amp; Manning, 2014)</a> — the other great early embedding method, built from co-occurrence counts rather than prediction.`,
          `<a href="https://arxiv.org/abs/1607.06520" target="_blank" rel="noopener">Bolukbasi et al. (2016), "Man is to Computer Programmer as Woman is to Homemaker?"</a> — the bias paper, including what debiasing can and cannot do.`,
          `<a href="https://distill.pub/2016/misread-tsne/" target="_blank" rel="noopener">"How to Use t-SNE Effectively" (Distill)</a> — interactive proof of why those beautiful cluster plots mislead.`,
        ])));
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
      { label: 'king − man + woman', a: 'king', b: 'man', c: 'woman', fam: 'people' },
      { label: 'Paris − France + Italy', a: 'Paris', b: 'France', c: 'Italy', fam: 'place' },
      { label: 'prince − boy + girl', a: 'prince', b: 'boy', c: 'girl', fam: 'people' },
      { label: 'Tokyo − Japan + Germany', a: 'Tokyo', b: 'Japan', c: 'Germany', fam: 'place' },
    ];
    let fam = 'people';
    const famSel = ctx.select({
      label: 'fit the map to',
      options: Object.keys(FAMILIES).map(k => ({ value: k, label: FAMILIES[k].label })),
      value: 'people',
      onChange: (v) => { fam = v; refit(fam); compute(); fitView(); },
    });
    const shown = () => WORDS.filter(w => FAMILIES[fam].has(w)
      || (result && (w === result.a || w === result.b || w === result.c || w === result.best)));
    /* Refitting changes the units of the whole map, so the view has to be
       refitted too or the cloud ends up squashed into a corner of the canvas.
       The analogy's own result point is included, so it can never land off the
       top of the picture. */
    function fitView() {
      const pts = shown().map(w => ({ x: w.px, y: w.py }));
      if (result) pts.push({ x: result.px, y: result.py });
      if (!pts.length) return;
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const p of pts) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
      const sxSpan = Math.max(1e-6, x1 - x0), sySpan = Math.max(1e-6, y1 - y0);
      scale = Math.max(40, Math.min(600, Math.min((W - 220) / sxSpan, (H - 150) / sySpan)));
      ox = W / 2 - (x0 + x1) / 2 * scale;
      oy = H / 2 + (y0 + y1) / 2 * scale;
    }
    const names = WORDS.map(w => w.w).sort();
    const recompute = () => { compute(); ensureVisible(); };
    const selA = ctx.select({ label: 'A', options: names, value: 'king', onChange: recompute });
    const selB = ctx.select({ label: '− B', options: names, value: 'man', onChange: recompute });
    const selC = ctx.select({ label: '+ C', options: names, value: 'woman', onChange: recompute });
    const presetSel = ctx.select({
      label: 'preset analogy',
      options: PRESETS.map((p, i) => ({ value: String(i), label: p.label })),
      value: '0',
      onChange: (v) => {
        const p = PRESETS[+v] || PRESETS[0];
        setSel(selA, p.a); setSel(selB, p.b); setSel(selC, p.c);
        if (p.fam && p.fam !== fam) { fam = p.fam; setSel(famSel, fam); refit(fam); }
        compute(); fitView();
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

    /* Changing A, B or C can put the result outside a view the reader panned to.
       Refit only when something actually fell off the picture, so deliberate
       panning and zooming survive. */
    function ensureVisible() {
      const pts = shown().map(w => toScreen(w.px, w.py));
      if (result) pts.push(toScreen(result.px, result.py));
      if (pts.some(p => p.x < 16 || p.x > W - 16 || p.y < 16 || p.y > H - 16)) fitView();
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
      for (const w of shown()) {
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

      const place = labelPlacer(W, H);
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
        arrow(g, pa.x, pa.y, pa.x + (stepB.x - pa.x) * t1, pa.y + (stepB.y - pa.y) * t1, C.danger, '− ' + result.b.w, place);
        if (t2 > 0) arrow(g, stepB.x, stepB.y, stepB.x + (pr.x - stepB.x) * t2, stepB.y + (pr.y - stepB.y) * t2, C.green, '+ ' + result.c.w, place);
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
          place.avoid(hs.x, hs.y, ns.x, ns.y);
        }
      }

      // words
      g.textAlign = 'center'; g.textBaseline = 'middle';
      const vis = shown().filter(w => { const s = toScreen(w.px, w.py); return s.x > -40 && s.x < W + 40 && s.y > -20 && s.y < H + 20; });
      /* dots first, so no label is ever painted under one */
      for (const w of vis) {
        const s = toScreen(w.px, w.py);
        const isHot = hovered === w, isRes = result && result.best === w && anim >= 1;
        g.fillStyle = CAT_COLOR[w.c] || C.muted;
        g.beginPath(); g.arc(s.x, s.y, isHot || isRes ? 5 : 3, 0, Math.PI * 2); g.fill();
      }
      /* then labels, the important ones first so they win the good positions */
      const order = vis.slice().sort((a, b) => rank(b) - rank(a));
      function rank(w) {
        if (hovered === w) return 4;
        if (result && (w === result.best || w === result.a || w === result.b || w === result.c)) return 3;
        return FAMILIES[fam].has(w) ? 1 : 0;
      }
      for (const w of order) {
        const s = toScreen(w.px, w.py);
        const isHot = hovered === w, isRes = result && result.best === w && anim >= 1;
        g.font = (isHot || isRes ? '600 13px' : '12px') + ' Inter, system-ui, sans-serif';
        place(g, w.w, s.x, s.y, (lx, ly, far) => {
          if (far) {
            g.strokeStyle = 'rgba(148,163,184,0.35)'; g.lineWidth = 1;
            g.beginPath(); g.moveTo(s.x, s.y); g.lineTo(lx, ly + (ly > s.y ? -7 : 7)); g.stroke();
          }
          g.fillStyle = isHot || isRes ? '#ffffff' : 'rgba(230,235,245,0.78)';
          g.fillText(w.w, lx, ly);
        });
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
    fitView();

    return ctx.figure(cv,
      'A hand-built 10-dimensional embedding space, projected to 2-D with PCA computed in your browser. <b>Change what the map is fitted to and watch the whole picture reorganise.</b> PCA keeps the two directions along which the chosen words spread out most and throws the other eight away, so fitting it to all 36 words buries gender entirely — king and queen land on the same pixel — while fitting it to the people puts gender on an axis and the classic analogy becomes something you can see. Nothing about the vectors changed. Colours mark categories for your benefit only; the space itself has no idea they exist. The analogy excludes the three input words, which is how the original word2vec evaluations were scored.',
      [famSel, presetSel, selA, selB, selC,
       ctx.button('Run arithmetic', () => recompute(), 'primary'),
       ctx.button('Reset view', () => fitView())],
      readout);
  }

  /* Place a label near its dot without ever printing it on a label already
     placed. The dot stays at the true position — only the annotation moves, and
     a leader line is drawn when it has to move far. Two words at the same point
     still look like one point, which is the honest picture. */
  function segCrossesBox(a, b, r) {
    let t0 = 0, t1 = 1;
    const dx = b.x - a.x, dy = b.y - a.y;
    const p = [-dx, dx, -dy, dy], q = [a.x - r.x0, r.x1 - a.x, a.y - r.y0, r.y1 - a.y];
    for (let i = 0; i < 4; i++) {
      if (p[i] === 0) { if (q[i] < 0) return false; continue; }
      const t = q[i] / p[i];
      if (p[i] < 0) { if (t > t1) return false; if (t > t0) t0 = t; }
      else { if (t < t0) return false; if (t < t1) t1 = t; }
    }
    return t1 > t0;
  }
  function labelPlacer(CW, CH) {
    const placed = [], lines = [];
    function place(g, text, x, y, draw) {
      const w = g.measureText(text).width, hh = 13;
      const OFFS = [[0, -12], [0, 15], [0, -25], [0, 28], [w / 2 + 10, 1], [-w / 2 - 10, 1],
      [0, -38], [0, 41], [w / 2 + 10, -14], [-w / 2 - 10, -14], [w / 2 + 10, 16], [-w / 2 - 10, 16],
      [w / 2 + 10, -28], [-w / 2 - 10, -28], [w / 2 + 10, 30], [-w / 2 - 10, 30]];
      for (const [dx, dy] of OFFS) {
        const b = { x0: x + dx - w / 2 - 1, y0: y + dy - hh / 2, x1: x + dx + w / 2 + 1, y1: y + dy + hh / 2 };
        if (CW && (b.x0 < 2 || b.x1 > CW - 2 || b.y0 < 2 || b.y1 > CH - 2)) continue;  /* stay on the canvas */
        if (placed.some(p => p.x0 < b.x1 && b.x0 < p.x1 && p.y0 < b.y1 && b.y0 < p.y1)) continue;
        if (lines.some(L => segCrossesBox(L[0], L[1], b))) continue;
        placed.push(b);
        draw(x + dx, y + dy, Math.hypot(dx, dy) > 20);
        return true;
      }
      return false;   /* nowhere free: better an unlabelled dot than a smear */
    }
    /* Lines registered here are avoided as well as other labels, so an arrow
       drawn across the map never ends up striking through a word. */
    place.avoid = (x1, y1, x2, y2) => { lines.push([{ x: x1, y: y1 }, { x: x2, y: y2 }]); };
    return place;
  }

  function arrow(g, x1, y1, x2, y2, color, label, place) {
    if (!isFinite(x1) || !isFinite(x2)) return;
    g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 2;
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
    if (place) place.avoid(x1, y1, x2, y2);
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
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        const put = (lx, ly) => { g.fillStyle = color; g.fillText(label, lx, ly); };
        /* Offset the label clear of its OWN arrow, perpendicular to it: the
           exact distance at which an axis-aligned box of this size stops
           touching a line at this angle. Straight up would work for a flat
           arrow and sit right on top of a steep one. */
        const lw = g.measureText(label).width;
        const d = Math.abs(Math.sin(a)) * (lw / 2) + Math.abs(Math.cos(a)) * 7 + 5;
        const up = Math.cos(a) > 0 ? -1 : 1;
        const ax = mx + -Math.sin(a) * d * up, ay = my + Math.cos(a) * d * up;
        if (place) place(g, label, ax, ay, put); else put(ax, ay);
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
      /* "Make them identical" puts b exactly on a, which is the point of that
         button — so say so, rather than stacking two letters on one pixel. */
      const coincide = Math.hypot(sa.x - sb.x, sa.y - sb.y) < 9;
      g.fillStyle = C.accent; g.beginPath(); g.arc(sa.x, sa.y, 7, 0, Math.PI * 2); g.fill();
      if (!coincide) { g.fillStyle = C.green; g.beginPath(); g.arc(sb.x, sb.y, 7, 0, Math.PI * 2); g.fill(); }
      g.font = '600 13px Inter, system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillStyle = '#0b0f17'; g.fillText('a', sa.x, sa.y);
      if (!coincide) g.fillText('b', sb.x, sb.y);
      else {
        g.font = '600 12px Inter, system-ui, sans-serif';
        g.fillStyle = C.green; g.textAlign = 'left';
        g.fillText('b is exactly on top of a', sa.x + 14, sa.y - 14);
        g.textAlign = 'center';
      }

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
      /* Words converge on top of each other as training succeeds — that is the
         point of the demo — so the dots stay where the maths puts them and only
         the labels step aside. The grouped words get first pick of the free
         positions, since they are the ones the reader is told to watch. */
      const pts = [];
      for (let i = 0; i < V; i++) {
        const x = sx(Win[i][0]), y = sy(Win[i][1]);
        if (!isFinite(x) || !isFinite(y)) continue;
        pts.push({ w: vocab[i], grp: GROUP[vocab[i]], x, y });
      }
      for (const p of pts) {
        g.fillStyle = p.grp ? GC[p.grp] : 'rgba(148,163,184,0.55)';
        g.beginPath(); g.arc(p.x, p.y, p.grp ? 4 : 2.5, 0, Math.PI * 2); g.fill();
      }
      const place = labelPlacer(W, H);
      for (const p of pts.slice().sort((a, b) => (b.grp ? 1 : 0) - (a.grp ? 1 : 0))) {
        g.font = (p.grp ? '600 12px' : '10px') + ' Inter, system-ui, sans-serif';
        place(g, p.w, p.x, p.y, (lx, ly, far) => {
          if (far) {
            g.strokeStyle = 'rgba(148,163,184,0.3)'; g.lineWidth = 1;
            g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(lx, ly + (ly > p.y ? -7 : 7)); g.stroke();
          }
          g.fillStyle = p.grp ? GC[p.grp] : 'rgba(148,163,184,0.5)';
          g.fillText(p.w, lx, ly);
        });
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

  /* ---------- shared canvas text helper ---------- */
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

  /* ---------- Interactive: three ways to score two words ---------- */
  function buildThreeWays(ctx) {
    const [cv, g] = ctx.canvas(720, 330);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    const byWord = new Map(WORDS.map(w => [w.w, w.v]));
    /* a few extra words so the alphabetical-neighbour failure is visible */
    const EXTRA = {
      catalogue: V({ size: 0.2 }),
      apply: V({ act: 0.8 }),
      democracy: V({ size: 0.5, act: 0.2 }),
    };
    const vecOf = (w) => byWord.get(w) || EXTRA[w];
    /* one alphabetical vocabulary, so an ID really is a position in a sorted word list */
    const VOCAB = [...new Set([...WORDS.map(w => w.w), ...Object.keys(EXTRA)])].sort();
    const idOf = (w) => VOCAB.indexOf(w) + 1;
    let a = 'cat', b = 'catalogue';

    const mkSel = (label, get, set) => {
      const sel = ctx.h('select', {}, VOCAB.map(w => ctx.h('option', { value: w }, w)));
      sel.value = get();
      sel.addEventListener('change', () => set(sel.value));
      const wrap = ctx.h('div', { class: 'control' }, ctx.h('label', {}, label), sel);
      wrap.sync = () => { sel.value = get(); };
      return wrap;
    };
    const selA = mkSel('word A', () => a, (v) => { a = v; });
    const selB = mkSel('word B', () => b, (v) => { b = v; });
    const preset = (x, y) => ctx.button(x + ' vs ' + y, () => { a = x; b = y; selA.sync(); selB.sync(); });
    const ro = ctx.readout();

    ctx.loop(() => {
      g.clearRect(0, 0, cv.W, cv.H);
      const ia = idOf(a), ib = idOf(b);
      const idGap = Math.abs(ia - ib);
      /* an ID scheme implies "close number = close meaning"; score it that way */
      const idScore = 1 - Math.min(1, idGap / VOCAB.length);
      const oneHot = a === b ? 1 : 0;           // distinct one-hot vectors are always perpendicular
      const emb = cosine(vecOf(a), vecOf(b));

      g.font = 'bold 15px Inter, system-ui, sans-serif'; g.fillStyle = C.text;
      g.fillText('how alike are "' + a + '" and "' + b + '"?', 34, 32);

      const rows = [
        {
          name: 'a number per word',
          detail: a + ' = ' + ia + ', ' + b + ' = ' + ib + '  →  ' + idGap + ' apart in the sorted list',
          v: idScore, col: C.danger,
          verdict: idGap <= 2 ? 'calls them nearly the same word' : 'calls them unrelated',
        },
        {
          name: 'one-hot (its own axis each)',
          detail: 'two different slots set to 1, so the vectors are perpendicular',
          v: oneHot, col: C.warn,
          verdict: a === b ? 'identical' : 'every distinct pair scores exactly 0',
        },
        {
          name: 'a learned embedding',
          detail: '10 numbers per word, cosine of the angle between them',
          v: emb, col: C.green,
          verdict: emb > 0.6 ? 'closely related' : emb > 0.25 ? 'somewhat related' : 'unrelated',
        },
      ];
      let y = 68;
      rows.forEach((r) => {
        g.font = 'bold ' + FONT; g.fillStyle = C.text;
        g.fillText(r.name, 34, y);
        g.font = MONO; g.fillStyle = C.muted;
        g.fillText(r.detail, 34, y + 18);
        const BX = 34, BW = 380, BY = y + 28;
        g.fillStyle = C.line; g.fillRect(BX, BY, BW, 14);
        const frac = ctx.clamp((r.v + 1) / 2, 0, 1);
        g.fillStyle = r.col;
        g.fillRect(BX + BW / 2, BY, (frac - 0.5) * BW, 14);
        g.strokeStyle = C.muted; g.lineWidth = 1;
        g.beginPath(); g.moveTo(BX + BW / 2, BY - 3); g.lineTo(BX + BW / 2, BY + 17); g.stroke();
        g.font = 'bold 15px Inter, system-ui, sans-serif'; g.fillStyle = r.col;
        g.fillText(r.v.toFixed(2), BX + BW + 14, BY + 13);
        g.font = FONT; g.fillStyle = C.muted;
        g.fillText(r.verdict, BX + BW + 62, BY + 13);
        y += 82;
      });
      g.font = MONO; g.fillStyle = C.muted;
      g.fillText('0', 34 + 190 - 4, 68 + 28 + 30);
      g.font = FONT;
      wrapText2(g, 'Only the bottom bar ever tracks what you actually mean by "alike". The top two are what you get when a human picks the numbers instead of letting a training task choose them.', 34, 312, 640, 17);
      ro.set({ 'sorted-list gap': idGap, 'one-hot cosine': oneHot.toFixed(2), 'embedding cosine': emb.toFixed(2) });
    });

    return ctx.figure(cv,
      'The same pair of words scored by all three schemes at once. The middle bar never moves off zero, because two distinct one-hot vectors are always exactly perpendicular — "cat" is as far from "dog" as it is from "democracy". The top bar moves, but for the wrong reason: it is measuring alphabetical adjacency. Only the learned embedding produces a number you would agree with.',
      [selA, selB, preset('cat', 'catalogue'), preset('cat', 'dog'), preset('cat', 'democracy'), preset('king', 'queen')], ro);
  }

  /* ---------- Interactive: the distributional hypothesis ---------- */
  function buildTesguino(ctx) {
    const [cv, g] = ctx.canvas(720, 390);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    /* each context sentence contributes a set of cue words */
    const CONTEXTS = [
      { text: 'A bottle of ▁▁▁ is on the table.', cues: ['bottle', 'table'] },
      { text: 'Everybody likes ▁▁▁.', cues: ['likes'] },
      { text: 'Do not have ▁▁▁ before you drive.', cues: ['drive', 'drink'] },
      { text: 'We make ▁▁▁ out of corn.', cues: ['corn', 'make'] },
      { text: '▁▁▁ makes you drunk.', cues: ['drunk', 'drink'] },
    ];
    /* which cues each candidate is genuinely seen with, in ordinary text */
    const CANDIDATES = [
      { w: 'beer', cues: ['bottle', 'table', 'likes', 'drive', 'drink', 'drunk'] },
      { w: 'wine', cues: ['bottle', 'table', 'likes', 'drive', 'drink', 'drunk'] },
      { w: 'corn', cues: ['table', 'likes', 'corn', 'make'] },
      { w: 'water', cues: ['bottle', 'table', 'likes', 'drink'] },
      { w: 'hammer', cues: ['table', 'make'] },
      { w: 'democracy', cues: ['likes'] },
    ];
    const on = CONTEXTS.map((_, i) => i === 0);
    const toggles = CONTEXTS.map((c, i) => ctx.button('sentence ' + (i + 1), () => { on[i] = !on[i]; }, i === 0 ? 'primary' : ''));
    const allBtn = ctx.button('Show all five', () => on.forEach((_, i) => { on[i] = true; }), 'primary');
    const noneBtn = ctx.button('Back to one', () => on.forEach((_, i) => { on[i] = (i === 0); }));
    const ro = ctx.readout();

    ctx.loop(() => {
      g.clearRect(0, 0, cv.W, cv.H);
      const liveCues = new Set();
      CONTEXTS.forEach((c, i) => { if (on[i]) c.cues.forEach(q => liveCues.add(q)); });

      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('the only evidence you get about "tesgüino"', 30, 26);
      let y = 52;
      CONTEXTS.forEach((c, i) => {
        g.font = MONO;
        g.fillStyle = on[i] ? C.text : '#2a3444';
        g.fillText((on[i] ? '●  ' : '○  ') + c.text, 30, y);
        y += 22;
      });

      /* score each candidate by how much of the live context it also occurs with */
      const scored = CANDIDATES.map(c => {
        let hit = 0;
        liveCues.forEach(q => { if (c.cues.indexOf(q) >= 0) hit++; });
        return { w: c.w, score: liveCues.size ? hit / liveCues.size : 0 };
      }).sort((x, z) => z.score - x.score);

      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('how well each candidate fits that company', 30, 188);
      const BX = 130, BW = 420;
      scored.forEach((s, i) => {
        const yy = 210 + i * 23;
        g.font = MONO; g.fillStyle = C.muted; g.fillText(s.w, 30, yy + 11);
        g.fillStyle = C.line; g.fillRect(BX, yy, BW, 15);
        g.fillStyle = i === 0 && s.score > scored[1].score ? C.green : C.accent;
        g.fillRect(BX, yy, s.score * BW, 15);
        g.fillStyle = C.muted; g.fillText((s.score * 100).toFixed(0) + '%', BX + BW + 10, yy + 12);
      });

      const tiedTop = scored.filter(s => s.score === scored[0].score).map(s => s.w);
      const nOn = on.filter(Boolean).length;
      const decided = nOn >= 4;
      g.font = 'bold ' + FONT;
      g.fillStyle = decided ? C.green : C.warn;
      /* two near-synonyms tying at the top is the right answer, not an unfinished one */
      wrapText2(g, nOn <= 1
        ? 'One sentence is not enough: ' + tiedTop.join(', ') + ' all fit equally well and the ranking means nothing yet.'
        : !decided
          ? 'Getting sharper, but ' + tiedTop.join(' and ') + ' still fit equally. Switch on more sentences.'
          : tiedTop.length > 1
            ? tiedTop.join(' and ') + ' lead together — and that is the correct answer, not a failure: they are near-synonyms, so no amount of context separates them. Both are far ahead of hammer and democracy.'
            : '"' + scored[0].w + '" fits the company best — inferred from context alone, with no definition anywhere.',
        30, 336, 660, 16);
      ro.set({ 'contexts shown': nOn, 'cue words': liveCues.size, 'best fit': scored[0].w });
    });

    return ctx.figure(cv,
      'A word you have never seen, and five sentences it appears in. Each sentence contributes cue words, and each candidate is scored by how much of that company it keeps in ordinary text. With one sentence the ranking is meaningless; by five, "beer" and "wine" separate from "hammer" and "democracy" — and note they do not separate from <i>each other</i>, which is exactly right, since tesgüino is a corn beer. Nothing here knows what any word means. It only knows what turns up nearby.',
      [...toggles, allBtn, noneBtn], ro);
  }

  /* ---------- Interactive: what the embedding table costs ---------- */
  function buildEmbedTable(ctx) {
    const [cv, g] = ctx.canvas(720, 300);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    let vocab = 50257, dim = 768, tied = false;
    const vSl = ctx.slider({ label: 'vocabulary (tokens)', min: 8000, max: 256000, step: 1000, value: 50257, onChange: (v) => { vocab = v; } });
    const dSl = ctx.slider({ label: 'embedding dimension', min: 64, max: 8192, step: 64, value: 768, onChange: (v) => { dim = v; } });
    const tieBtn = ctx.button('tie input and output', () => { tied = !tied; tieBtn.textContent = tied ? 'untie input and output' : 'tie input and output'; });
    const gpt2 = ctx.button('GPT-2 small', () => { vocab = 50257; vSl.value = 50257; dim = 768; dSl.value = 768; }, 'primary');
    const frontier = ctx.button('frontier-ish', () => { vocab = 128000; vSl.value = 128000; dim = 4096; dSl.value = 4096; });
    const ro = ctx.readout();
    const human = (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + ' billion' : n >= 1e6 ? (n / 1e6).toFixed(1) + ' million' : (n / 1e3).toFixed(0) + ' thousand';

    ctx.loop(() => {
      g.clearRect(0, 0, cv.W, cv.H);
      const one = vocab * dim;
      const total = tied ? one : one * 2;

      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('the embedding table: one row per token', 34, 28);

      /* a schematic of the matrix */
      const X = 34, Y = 52, W = 250, H = 150;
      g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(X, Y, W, H);
      const rows = 14;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < 18; c++) {
          const v = Math.sin(r * 1.7 + c * 0.9) * 0.5 + 0.5;
          g.fillStyle = 'rgba(124,156,255,' + (0.08 + v * 0.5) + ')';
          g.fillRect(X + 4 + c * ((W - 8) / 18), Y + 4 + r * ((H - 8) / rows), (W - 8) / 18 - 1, (H - 8) / rows - 1);
        }
      }
      g.font = MONO; g.fillStyle = C.muted;
      g.fillText('←  ' + dim.toLocaleString() + ' numbers  →', X + 40, Y + H + 18);
      g.save(); g.translate(X - 10, Y + H / 2 + 40); g.rotate(-Math.PI / 2);
      g.fillText(vocab.toLocaleString() + ' tokens', 0, 0); g.restore();

      const TX = 330;
      g.font = FONT; g.fillStyle = C.muted;
      g.fillText('input table', TX, 62);
      g.font = 'bold 20px Inter, system-ui, sans-serif'; g.fillStyle = C.accent;
      g.fillText(human(one), TX, 86);
      g.font = MONO; g.fillStyle = C.muted;
      g.fillText(vocab.toLocaleString() + ' × ' + dim.toLocaleString(), TX, 104);

      g.font = FONT; g.fillStyle = C.muted;
      g.fillText('output (un-embedding) table', TX, 140);
      g.font = 'bold 20px Inter, system-ui, sans-serif'; g.fillStyle = tied ? C.green : C.warn;
      g.fillText(tied ? 'reuses the same weights' : human(one), TX, 164);

      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('spent on meaning-before-context', TX, 208);
      g.font = 'bold 26px Inter, system-ui, sans-serif'; g.fillStyle = C.green;
      g.fillText(human(total), TX, 238);
      g.font = FONT; g.fillStyle = C.muted;
      wrapText2(g, 'And none of it is a computation. Turning a token ID into a vector is a row lookup — the model simply reads row number 3312.', TX, 262, 350, 17);
      ro.set({ vocab: vocab.toLocaleString(), dim, tied: tied ? 'yes' : 'no', parameters: human(total) });
    });

    return ctx.figure(cv,
      'GPT-2 small spends about 38.6 million parameters — 50,257 tokens × 768 numbers — on the input table alone, before a single layer of actual computation runs. Tying the input and output tables reuses one matrix for both the lookup on the way in and the comparison on the way out, halving that cost and usually helping quality slightly. It is the cheapest real optimisation in the whole architecture.',
      [vSl, dSl, gpt2, frontier, tieBtn], ro);
  }

  /* ---------- Interactive: one word, two meanings, resolved by context ---------- */
  function buildContextBank(ctx) {
    const [cv, g] = ctx.canvas(720, 396);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    const LAYERS = 12;
    let layer = 0, playing = false, acc = 0;
    /* fixed anchor words, and the two destinations "bank" drifts toward */
    const ANCHORS = [
      { w: 'river', x: -0.72, y: 0.46 }, { w: 'water', x: -0.60, y: 0.66 },
      { w: 'boat', x: -0.80, y: 0.18 },
      { w: 'loan', x: 0.70, y: 0.50 }, { w: 'money', x: 0.62, y: 0.70 },
      { w: 'interest', x: 0.82, y: 0.22 },
    ];
    const START = { x: 0.0, y: -0.55 };
    const DEST = [{ x: -0.66, y: 0.44 }, { x: 0.66, y: 0.48 }];
    const SENT = [
      'sat on the <b>bank</b> of the river',
      'the <b>bank</b> approved the loan',
    ];
    const ease = (t) => t * t * (3 - 2 * t);
    const posAt = (i, L) => {
      const t = ease(ctx.clamp(L / LAYERS, 0, 1));
      return { x: START.x + (DEST[i].x - START.x) * t, y: START.y + (DEST[i].y - START.y) * t };
    };

    const lSl = ctx.slider({ label: 'transformer layer', min: 0, max: LAYERS, step: 1, value: 0, onChange: (v) => { layer = v; } });
    const playBtn = ctx.button('▶ Climb the layers', () => { playing = !playing; playBtn.textContent = playing ? '⏸ Pause' : '▶ Climb the layers'; }, 'primary');
    const resetBtn = ctx.button('Back to layer 0', () => { layer = 0; lSl.value = 0; playing = false; playBtn.textContent = '▶ Climb the layers'; });
    const ro = ctx.readout();

    ctx.loop((dt) => {
      if (playing) { acc += dt; if (acc > 0.35) { acc = 0; layer = layer >= LAYERS ? 0 : layer + 1; lSl.value = layer; } }
      g.clearRect(0, 0, cv.W, cv.H);
      const CX = 360, CY = 190, S = 130;
      const sx = (x) => CX + x * S * 1.9;
      const sy = (y) => CY - y * S;

      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('the same word, in two sentences', 34, 26);
      SENT.forEach((s, i) => {
        g.font = MONO; g.fillStyle = i === 0 ? C.accent : C.warn;
        const plain = s.replace(/<[^>]+>/g, '');
        g.fillText((i === 0 ? '▲  ' : '■  ') + plain, 34, 48 + i * 20);
      });

      g.strokeStyle = C.line; g.lineWidth = 1;
      g.beginPath(); g.moveTo(sx(-1.05), sy(0)); g.lineTo(sx(1.05), sy(0)); g.stroke();

      ANCHORS.forEach(a => {
        g.fillStyle = 'rgba(148,163,184,0.5)';
        g.beginPath(); g.arc(sx(a.x), sy(a.y), 4, 0, 7); g.fill();
        g.font = MONO; g.fillStyle = C.muted;
        g.fillText(a.w, sx(a.x) - 12, sy(a.y) - 10);
      });

      /* the trails */
      [0, 1].forEach(i => {
        g.strokeStyle = i === 0 ? 'rgba(124,156,255,0.35)' : 'rgba(251,191,36,0.35)';
        g.lineWidth = 1.5; g.setLineDash([3, 3]); g.beginPath();
        for (let L = 0; L <= layer; L++) { const q = posAt(i, L); L ? g.lineTo(sx(q.x), sy(q.y)) : g.moveTo(sx(q.x), sy(q.y)); }
        g.stroke(); g.setLineDash([]);
      });
      [0, 1].forEach(i => {
        const q = posAt(i, layer);
        g.fillStyle = i === 0 ? C.accent : C.warn;
        if (i === 0) {
          g.beginPath(); g.moveTo(sx(q.x), sy(q.y) - 8); g.lineTo(sx(q.x) + 7, sy(q.y) + 5); g.lineTo(sx(q.x) - 7, sy(q.y) + 5); g.closePath(); g.fill();
        } else {
          g.fillRect(sx(q.x) - 6, sy(q.y) - 6, 12, 12);
        }
        g.font = 'bold ' + MONO; g.fillStyle = i === 0 ? C.accent : C.warn;
        g.fillText('bank', sx(q.x) + 12, sy(q.y) + 4);
      });

      const p0 = posAt(0, layer), p1 = posAt(1, layer);
      const sep = Math.hypot(p0.x - p1.x, p0.y - p1.y);
      g.font = 'bold 15px Inter, system-ui, sans-serif';
      g.fillStyle = layer === 0 ? C.danger : sep > 1.0 ? C.green : C.warn;
      g.fillText(layer === 0
        ? 'layer 0: identical. Same row of the same table.'
        : 'layer ' + layer + ': the two meanings are ' + sep.toFixed(2) + ' apart',
        34, 330);
      g.font = FONT; g.fillStyle = C.muted;
      wrapText2(g, layer === 0
        ? 'Before any attention runs, a word is just a lookup — so word2vec can never tell these two sentences apart.'
        : 'Each attention layer nudges every token\'s vector using its neighbours. "river" pulls one copy left; "loan" pulls the other right.',
        34, 350, 650, 16);
      ro.set({ layer, separation: sep.toFixed(2), 'sense A': layer > 6 ? 'riverbank' : 'undecided', 'sense B': layer > 6 ? 'finance' : 'undecided' });
    });

    return ctx.figure(cv,
      'A schematic of contextual embeddings, not a measurement of a specific model — but the behaviour it shows is real and was the key finding of ELMo and BERT in 2018. At layer 0 both copies of "bank" are the identical row of the embedding table, so they must coincide. Each attention layer then updates every token using the tokens around it, and the two copies drift apart toward the company each keeps. This is the single clearest statement of what attention does, and chapter 7 builds the mechanism.',
      [lSl, playBtn, resetBtn], ro);
  }
})();
