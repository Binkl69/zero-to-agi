/* Chapter 1 — What does it mean for a machine to learn?
   DESIGN RULE FOR THIS CHAPTER: the reader touches something in the first ten
   seconds. Text exists only to explain what they just saw with their own hands.
   Target: under 1,000 words of prose, six things to play with. */
(function () {
  const ZTA = window.ZTA;

  /* The four corners of the square. Every puzzle uses the same four dots and
     only changes which ones are "on". That is the whole point: same dots,
     wildly different difficulty. */
  const PTS = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const PUZZLES = {
    AND:  { labels: [0, 0, 0, 1], blurb: 'Light up ONLY when both switches are on.' },
    OR:   { labels: [0, 1, 1, 1], blurb: 'Light up when EITHER switch is on.' },
    XOR:  { labels: [0, 1, 1, 0], blurb: 'Light up when EXACTLY ONE switch is on.' },
  };

  ZTA.registerChapter({
    id: '01-what-is-learning',
    num: 1,
    part: 'I',
    title: 'What does it mean for a machine to learn?',
    tagline: 'Start by doing it yourself. Four dots, one ruler, three puzzles. The third one is impossible, and that fact shaped the next fifty years.',

    render(root, ctx) {
      const h = ctx.h;

      /* ---------- STRAIGHT INTO IT. No preamble. ---------- */
      root.append(
        ctx.callout('tryit', '🖐 Do this first, read after',
          `Four dots. <b>Red dots must end up on the red side of the line, blue dots on the blue side.</b>
           Drag the two white handles to move the line. Try to score 4 out of 4.<br>
           Start on <b>puzzle 1</b>. When you get it, switch to puzzle 2, then puzzle 3.`),
        buildLinePuzzle(ctx),
        ctx.p(`If you are reading this before playing, go back and play. It takes twenty seconds and the rest of the chapter will not land otherwise.`),
      );

      /* ---------- what just happened ---------- */
      root.append(ctx.section('What you just were',
        ctx.p(`You were a <em>neuron</em>. Not a metaphor for one, the actual thing. An artificial neuron has exactly one move available to it: <b>draw a straight line and call one side "yes" and the other side "no"</b>. That is its entire repertoire.`),
        ctx.p(`Puzzles 1 and 2 fell over easily. Puzzle 3 did not, and it is worth being precise about why: <b>it is not hard, it is impossible.</b> The two red dots sit in opposite corners with the blue dots in the other two, so any straight line you draw strands somebody on the wrong side. You cannot win, and neither can any machine that only draws one line.`),
        ctx.callout('key', '🔑 The one idea in this chapter',
          `One neuron equals one straight line. Some problems cannot be split with one straight line. That single sentence explains a machine built in 1958, why the field collapsed in 1969, and why the word "deep" appears in "deep learning".`),
      ));

      /* ---------- now the machine does it ---------- */
      root.append(ctx.section('Now let the machine find the line',
        ctx.p(`You moved the line by feel. A machine has no feel, so it uses a rule so simple you could do it on paper: <b>look at one dot; if you got it wrong, shove the line a little bit toward getting it right; repeat.</b> Nothing cleverer than that.`),
        ctx.callout('tryit', '🖐 Try this',
          `Press <b>Train</b> and watch the line stagger toward an answer on puzzle 1. Then switch to <b>puzzle 3</b> and press Train again. Leave it running. Watch the mistake counter at the bottom: it drops, rises, and never reaches zero. It is not thinking. It is stuck in a loop, redoing the same moves forever.`),
        buildTrainer(ctx),
        ctx.p(`That flailing is exactly what your own program printed in the terminal. The machine is not broken and it is not slow. It is looking for something that does not exist.`),
      ));

      /* ---------- proof ---------- */
      root.append(ctx.section('Proof that it is impossible, not just difficult',
        ctx.p(`Maybe the machine is just bad at searching? Settle it by trying <b>every line there is</b>. Below, the computer sweeps through thousands of lines at every angle and position, and tallies how many get all four dots right.`),
        ctx.callout('tryit', '🖐 Try this',
          `Run it on puzzle 1, then on puzzle 3. Compare the two counters at the end.`),
        buildBruteForce(ctx),
        ctx.p(`Thousands of lines work for puzzle 1. <b>Zero</b> work for puzzle 3. That is not the machine giving up early. There is no answer of that shape anywhere.`),
        ctx.callout('history', '📜 The book that froze the field',
          `In 1969 Marvin Minsky and Seymour Papert published <i>Perceptrons</i>, proving exactly this on paper. Frank Rosenblatt's perceptron, built in 1958 and breathlessly covered in the press, could never learn XOR. Funding dried up and neural-network research went cold for over a decade, a period now called the first <em>AI winter</em>. The irony is that the fix was already understood in principle. Nobody yet knew how to train it.`),
      ));

      /* ---------- the fix ---------- */
      root.append(ctx.section('The fix: stop using one line',
        ctx.p(`If one line cannot do it, use two. Two lines carve the square into a <b>stripe</b>, and the stripe can hold both red dots while excluding both blue ones.`),
        ctx.callout('tryit', '🖐 Try this',
          `Drag the slider from 1 line to 2 lines and watch the impossible puzzle become possible.`),
        buildTwoLines(ctx),
        ctx.p(`Each line is one neuron. To combine them you need a third neuron that watches the first two and answers "am I between them?". That stack is a <em>network</em>, and the middle row is a <em>hidden layer</em>. Add more layers and you can cut out any shape at all, which is where the "deep" in deep learning comes from.`),
        ctx.callout('key', '🔑 So why did this take until the 2010s?',
          `Nobody doubted more layers were more powerful. The problem was <b>training</b> them: with a hidden layer, it is no longer obvious which weight to blame for a mistake. The answer, <em>backpropagation</em>, is chapter 2. It needed the maths to be popularised in 1986, then twenty more years of faster chips and bigger datasets before it paid off.`),
      ));

      /* ---------- inside the neuron ---------- */
      root.append(ctx.section('What the neuron is actually doing with numbers',
        ctx.p(`"Draw a line" is the picture. Here is the arithmetic underneath it, and it is smaller than you would expect. Each input gets multiplied by a <em>weight</em>, the results are added up, a <em>bias</em> is added, and if the total clears zero the neuron fires.`),
        ctx.callout('tryit', '🖐 Try this',
          `Move the weight sliders and watch the line in the previous demos rotate. Learning <b>is</b> the search for these three numbers. There is nothing else in there.`),
        buildNeuronAnatomy(ctx),
        ctx.p(`Those three numbers are the neuron's <em>parameters</em>. Training means adjusting them until the answers come out right. A model you talk to today is this same arrangement with a few hundred billion parameters instead of three.`),
      ));

      /* ---------- rules vs learning ---------- */
      root.append(ctx.section('Why bother learning at all?',
        ctx.p(`You could just write the rules yourself. For decades that is what people did, and for some jobs it is still correct. The split is about whether you can <b>state</b> the rule.`),
        ctx.cards([
          { title: 'Write the rules by hand', body: 'You know the rule and can say it out loud. Tax calculations, traffic lights, chess legality. Precise, checkable, and it never surprises you. Useless when you cannot articulate the rule.' },
          { title: 'Learn from examples', body: 'You can recognise the answer but not describe how. Faces, spam, handwriting, whether a sentence sounds natural. You supply examples instead of instructions, and the machine finds the pattern.' },
        ]),
        ctx.p(`Nobody can write down the rule for "this photo contains a cat", yet you can label ten thousand photos in an afternoon. That trade, <b>examples instead of instructions</b>, is the entire bet of machine learning.`),
        ctx.callout('example', '🌍 Where single neurons still earn their keep',
          `This is not just history. A credit-scoring model, a hospital triage flag, or an A/B test winner is often one line drawn through data, because a single neuron is fast, cheap, and you can read off exactly why it decided what it decided. When regulators demand an explanation, one line beats a billion parameters.`),
      ));

      /* ---------- nesting ---------- */
      root.append(ctx.section('Three words people mix up',
        ctx.callout('tryit', '🖐 Try this', `Hover or tap each ring.`),
        buildNesting(ctx),
      ));

      /* ---------- quiz ---------- */
      root.append(ctx.quiz([
        {
          q: 'Why could you not solve puzzle 3 by dragging the line more carefully?',
          options: [
            'The handles were not sensitive enough',
            'The two red dots sit in opposite corners, so no straight line can separate them from the blue ones',
            'You needed to train it for more steps first',
            'The puzzle had a bug in it',
          ],
          answer: 1,
          explain: 'It is a geometric fact, not a matter of effort or precision. The brute-force demo tried thousands of lines and none of them worked.',
        },
        {
          q: 'What does a single artificial neuron actually compute?',
          options: [
            'It memorises every example it has seen',
            'It multiplies each input by a weight, adds them up with a bias, and fires if the total clears zero',
            'It compares the input to a database of known answers',
            'It simulates the chemistry of a biological brain cell',
          ],
          answer: 1,
          explain: 'Multiply, add, compare to a threshold. That is the whole operation, and geometrically it draws one straight boundary.',
        },
        {
          q: 'In the demo, what fixed the impossible puzzle?',
          options: [
            'A faster computer',
            'More training steps',
            'A second line, which needs a second neuron',
            'Removing one of the dots',
          ],
          answer: 2,
          explain: 'Two lines fence off a stripe that one line cannot. Stacking neurons like this is what "deep" means.',
        },
        {
          q: 'When is writing rules by hand still the better choice?',
          options: [
            'When you have millions of examples available',
            'When you can state the rule precisely and need to explain every decision',
            'Whenever the data contains numbers',
            'It never is; learned models are always better',
          ],
          answer: 1,
          explain: 'If you can articulate the rule, writing it is exact, cheap and auditable. Learning is for the cases where you recognise the answer but cannot describe the rule.',
        },
        {
          q: 'What caused the first AI winter?',
          options: [
            'Computers became too expensive to run',
            'A 1969 proof that a single-layer perceptron could not learn XOR, which drained confidence and funding',
            'Researchers ran out of data to train on',
            'A better technology replaced neural networks entirely',
          ],
          answer: 1,
          explain: 'Minsky and Papert made the limitation rigorous. The multi-layer fix existed in principle, but nobody could train it yet, so the field stalled for over a decade.',
        },
      ]));

      root.append(ctx.section('Go deeper',
        ctx.ul([
          `<a href="https://www.youtube.com/watch?v=aircAruvnKk" target="_blank" rel="noopener">3Blue1Brown, "But what is a neural network?"</a>: nineteen minutes, and the best visual introduction that exists. Watch it before chapter 2.`,
          `<a href="https://playground.tensorflow.org/" target="_blank" rel="noopener">TensorFlow Playground</a>: the same idea as the demos above, with more layers to play with. Set it to the spiral dataset and try to beat it.`,
          `<a href="https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon" target="_blank" rel="noopener">Cornell on Rosenblatt's perceptron</a>: the 1958 machine, the hype, and the backlash.`,
        ]),
      ));
    },
  });

  /* ==================================================================
     Shared drawing helpers for the square-with-four-dots demos
     ================================================================== */
  function makeBoard(cv, g, C, pad) {
    pad = pad || 54;
    const W = cv.W, H = cv.H;
    const size = Math.min(W, H) - pad * 2;
    const x0 = (W - size) / 2, y0 = (H - size) / 2;
    // data coords run -0.35 .. 1.35 so the dots sit inside with margin
    const LO = -0.35, HI = 1.35;
    const sx = (x) => x0 + (x - LO) / (HI - LO) * size;
    const sy = (y) => y0 + size - (y - LO) / (HI - LO) * size;
    const ix = (px) => LO + (px - x0) / size * (HI - LO);
    const iy = (py) => LO + (y0 + size - py) / size * (HI - LO);
    return { W, H, size, x0, y0, sx, sy, ix, iy };
  }

  function drawFrame(g, b, C) {
    g.clearRect(0, 0, b.W, b.H);
    g.fillStyle = C.bg; g.fillRect(0, 0, b.W, b.H);
    g.strokeStyle = 'rgba(148,163,184,0.18)'; g.lineWidth = 1;
    g.strokeRect(b.x0, b.y0, b.size, b.size);
  }

  function drawDots(g, b, labels, C, big) {
    for (let i = 0; i < PTS.length; i++) {
      const p = PTS[i], X = b.sx(p[0]), Y = b.sy(p[1]);
      g.fillStyle = labels[i] ? C.danger : C.accent;
      g.beginPath(); g.arc(X, Y, big ? 13 : 10, 0, Math.PI * 2); g.fill();
      g.strokeStyle = 'rgba(10,14,22,0.85)'; g.lineWidth = 2; g.stroke();
      g.fillStyle = '#0a0e16';
      g.font = '700 12px Inter, system-ui, sans-serif';
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(labels[i] ? '1' : '0', X, Y);
    }
    // axis hints
    g.fillStyle = C.muted; g.font = '11px Inter, system-ui, sans-serif';
    g.textAlign = 'center';
    g.fillText('switch A off', b.sx(0), b.sy(-0.28));
    g.fillText('switch A on', b.sx(1), b.sy(-0.28));
    g.save();
    g.translate(b.sx(-0.24), b.sy(0.5)); g.rotate(-Math.PI / 2);
    g.fillText('switch B  off → on', 0, 0);
    g.restore();
  }

  /* classify with the line through point A in direction (A→B); positive side = "1" */
  function classify(px, py, ax, ay, bx, by, flip) {
    const s = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    return (flip ? -s : s) > 0 ? 1 : 0;
  }

  /* shade the two half-planes */
  function shadeSides(g, b, ax, ay, bx, by, flip, C) {
    const step = 9;
    for (let X = b.x0; X < b.x0 + b.size; X += step) {
      for (let Y = b.y0; Y < b.y0 + b.size; Y += step) {
        const on = classify(b.ix(X + step / 2), b.iy(Y + step / 2), ax, ay, bx, by, flip);
        g.fillStyle = on ? 'rgba(251,113,133,0.13)' : 'rgba(124,156,255,0.13)';
        g.fillRect(X, Y, step, step);
      }
    }
  }

  function drawLineThrough(g, b, ax, ay, bx, by, color, width) {
    let dx = bx - ax, dy = by - ay;
    const len = Math.hypot(dx, dy) || 1;
    dx /= len; dy /= len;
    const far = 4;
    g.strokeStyle = color; g.lineWidth = width || 3;
    g.beginPath();
    g.moveTo(b.sx(ax - dx * far), b.sy(ay - dy * far));
    g.lineTo(b.sx(ax + dx * far), b.sy(ay + dy * far));
    g.stroke();
  }

  function scoreOf(labels, fn) {
    let n = 0;
    for (let i = 0; i < PTS.length; i++) if (fn(PTS[i][0], PTS[i][1]) === labels[i]) n++;
    return n;
  }

  /* ==================================================================
     1 — DRAG THE LINE YOURSELF
     ================================================================== */
  function buildLinePuzzle(ctx) {
    const C = ctx.colors;
    const [cv, g] = ctx.canvas(640, 480);
    const b = makeBoard(cv, g, C);
    let puzzle = 'AND', flip = false;
    let A = { x: 0.9, y: -0.2 }, B = { x: -0.2, y: 0.9 };
    let drag = null, solved = false, tries = 0, hintTimer = 0;
    const readout = ctx.readout();

    const sel = ctx.select({
      label: 'puzzle',
      options: [
        { value: 'AND', label: '1 · both switches on' },
        { value: 'OR', label: '2 · either switch on' },
        { value: 'XOR', label: '3 · exactly one switch on' },
      ],
      value: 'AND',
      onChange: (v) => { puzzle = v; solved = false; tries = 0; hintTimer = 0; },
    });

    function handleAt(p) {
      const da = Math.hypot(b.sx(A.x) - p.x, b.sy(A.y) - p.y);
      const db = Math.hypot(b.sx(B.x) - p.x, b.sy(B.y) - p.y);
      if (da < 30 && da <= db) return 'A';
      if (db < 30) return 'B';
      return null;
    }
    cv.addEventListener('pointerdown', (e) => { drag = handleAt(cv.pos(e)); if (drag) tries++; });
    cv.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const p = cv.pos(e);
      const t = { x: Math.max(-0.5, Math.min(1.5, b.ix(p.x))), y: Math.max(-0.5, Math.min(1.5, b.iy(p.y))) };
      const other = drag === 'A' ? B : A;
      if (Math.hypot(t.x - other.x, t.y - other.y) < 0.25) return;  // never collapse the line
      if (drag === 'A') A = t; else B = t;
    });
    const stop = () => { drag = null; };
    cv.addEventListener('pointerup', stop);
    cv.addEventListener('pointercancel', stop);
    cv.addEventListener('pointerleave', stop);

    ctx.loop((dt) => {
      const labels = PUZZLES[puzzle].labels;
      drawFrame(g, b, C);
      shadeSides(g, b, A.x, A.y, B.x, B.y, flip, C);
      drawLineThrough(g, b, A.x, A.y, B.x, B.y, C.text, 3);
      drawDots(g, b, labels, C, true);

      // handles
      for (const [hp, nm] of [[A, 'A'], [B, 'B']]) {
        g.fillStyle = '#fff';
        g.beginPath(); g.arc(b.sx(hp.x), b.sy(hp.y), 9, 0, Math.PI * 2); g.fill();
        g.strokeStyle = C.line; g.lineWidth = 2; g.stroke();
      }

      const score = scoreOf(labels, (x, y) => classify(x, y, A.x, A.y, B.x, B.y, flip));
      if (score === 4) solved = true;
      if (puzzle === 'XOR' && !solved) hintTimer += dt;

      // big score
      g.textAlign = 'left'; g.textBaseline = 'top';
      g.font = '700 26px Inter, system-ui, sans-serif';
      g.fillStyle = score === 4 ? C.green : C.text;
      g.fillText(score + ' / 4', 16, 14);
      g.font = '12px Inter, system-ui, sans-serif';
      g.fillStyle = C.muted;
      g.fillText(PUZZLES[puzzle].blurb, 16, 46);

      if (score === 4) {
        g.font = '700 15px Inter, system-ui, sans-serif'; g.fillStyle = C.green;
        g.textAlign = 'right';
        g.fillText('solved — try the next puzzle', b.W - 16, 18);
      } else if (puzzle === 'XOR' && hintTimer > 14) {
        g.font = '600 13px Inter, system-ui, sans-serif'; g.fillStyle = C.warn;
        g.textAlign = 'right';
        g.fillText('still stuck? that is the lesson. keep reading.', b.W - 16, 20);
      }
      g.textAlign = 'center'; g.textBaseline = 'middle';

      readout.set({ 'correct': score + ' of 4', 'puzzle': puzzle === 'AND' ? '1' : puzzle === 'OR' ? '2' : '3' });
    });

    return ctx.figure(cv,
      'Red dots want to be on the red side, blue on the blue side. Puzzles 1 and 2 are winnable. Puzzle 3 is not, and no amount of care will change that.',
      [sel,
       ctx.button('Swap which side is red', () => { flip = !flip; }),
       ctx.button('Reset line', () => { A = { x: 0.9, y: -0.2 }; B = { x: -0.2, y: 0.9 }; solved = false; hintTimer = 0; })],
      readout);
  }

  /* ==================================================================
     2 — WATCH THE MACHINE LEARN (the 1958 rule)
     ================================================================== */
  function buildTrainer(ctx) {
    const C = ctx.colors;
    const [cv, g] = ctx.canvas(640, 480);
    const b = makeBoard(cv, g, C);
    let puzzle = 'AND';
    let w1, w2, bias, cursor, pass, mistakes, lastMistakes, history, acc, running;

    function reset() {
      w1 = 0.4; w2 = -0.3; bias = 0.1;
      cursor = 0; pass = 0; mistakes = 0; lastMistakes = null; history = []; acc = 0;
    }
    reset();
    running = false;

    const LR = 0.12;
    function fire(x, y) { return (w1 * x + w2 * y + bias) > 0 ? 1 : 0; }

    function step() {
      const labels = PUZZLES[puzzle].labels;
      const p = PTS[cursor], target = labels[cursor];
      const out = fire(p[0], p[1]);
      const err = target - out;
      if (err !== 0) {
        mistakes++;
        w1 += LR * err * p[0];
        w2 += LR * err * p[1];
        bias += LR * err;
      }
      cursor++;
      if (cursor >= PTS.length) {
        cursor = 0; pass++;
        lastMistakes = mistakes;
        history.push(mistakes);
        if (history.length > 60) history.shift();
        mistakes = 0;
      }
    }

    const speed = ctx.slider({ label: 'speed', min: 1, max: 30, step: 1, value: 8 });
    const readout = ctx.readout();

    ctx.loop((dt) => {
      if (running) { acc += dt * speed.value; while (acc >= 1) { step(); acc -= 1; } }

      const labels = PUZZLES[puzzle].labels;
      drawFrame(g, b, C);
      // the line w1*x + w2*y + bias = 0 as two far-apart points
      const nx = w1, ny = w2, nn = Math.hypot(nx, ny) || 1e-6;
      const px = -bias * nx / (nn * nn), py = -bias * ny / (nn * nn);   // closest point to origin
      const dx = -ny / nn, dy = nx / nn;
      const Ax = px, Ay = py, Bx = px + dx, By = py + dy;
      const flip = classify(px + nx / nn * 0.1, py + ny / nn * 0.1, Ax, Ay, Bx, By, false) !== 1;
      shadeSides(g, b, Ax, Ay, Bx, By, flip, C);
      drawLineThrough(g, b, Ax, Ay, Bx, By, C.warn, 3);
      drawDots(g, b, labels, C, true);

      // highlight the dot currently being examined
      const cp = PTS[cursor];
      g.strokeStyle = C.green; g.lineWidth = 3;
      g.beginPath(); g.arc(b.sx(cp[0]), b.sy(cp[1]), 19, 0, Math.PI * 2); g.stroke();

      const score = scoreOf(labels, fire);
      g.textAlign = 'left'; g.textBaseline = 'top';
      g.font = '700 24px Inter, system-ui, sans-serif';
      g.fillStyle = score === 4 ? C.green : C.text;
      g.fillText(score + ' / 4', 16, 14);
      g.font = '12px Inter, system-ui, sans-serif'; g.fillStyle = C.muted;
      g.fillText('pass ' + pass + (lastMistakes == null ? '' : '  ·  mistakes last pass: ' + lastMistakes), 16, 44);

      // mistake history bars
      const bx0 = 16, by0 = b.H - 42;
      g.fillStyle = C.muted; g.font = '10px Inter, system-ui, sans-serif';
      g.fillText('mistakes per pass', bx0, by0 - 14);
      history.forEach((m, i) => {
        g.fillStyle = m === 0 ? C.green : C.danger;
        g.fillRect(bx0 + i * 8, by0 + 26 - m * 6.5, 6, Math.max(2, m * 6.5));
      });
      if (score === 4) {
        g.font = '700 14px Inter, system-ui, sans-serif'; g.fillStyle = C.green; g.textAlign = 'right';
        g.fillText('found a line and stopped', b.W - 16, 18);
      } else if (pass > 12) {
        g.font = '700 13px Inter, system-ui, sans-serif'; g.fillStyle = C.warn; g.textAlign = 'right';
        g.fillText('still going after ' + pass + ' passes…', b.W - 16, 18);
      }
      g.textAlign = 'center'; g.textBaseline = 'middle';

      readout.set({ 'correct': score + ' of 4', 'passes': pass, 'mistakes last pass': lastMistakes == null ? '—' : lastMistakes });
    });

    const playBtn = ctx.button('Train', () => {
      running = !running;
      playBtn.textContent = running ? 'Pause' : 'Train';
    }, 'primary');

    const sel = ctx.select({
      label: 'puzzle',
      options: [
        { value: 'AND', label: '1 · both switches on' },
        { value: 'OR', label: '2 · either switch on' },
        { value: 'XOR', label: '3 · exactly one switch on' },
      ],
      value: 'AND',
      onChange: (v) => { puzzle = v; reset(); },
    });

    return ctx.figure(cv,
      'The green ring shows which dot the machine is looking at right now. Every time it gets one wrong it nudges the line. On puzzles 1 and 2 the mistake bars hit zero and it stops. On puzzle 3 they never do.',
      [sel, playBtn, ctx.button('One step', () => step()), ctx.button('Reset', () => reset()), speed],
      readout);
  }

  /* ==================================================================
     3 — BRUTE FORCE: try every line there is
     ================================================================== */
  function buildBruteForce(ctx) {
    const C = ctx.colors;
    const [cv, g] = ctx.canvas(640, 460);
    const b = makeBoard(cv, g, C, 62);
    let puzzle = 'AND';
    let angleI, offI, tried, worked, running, done, bestShown, trail;
    const N_ANG = 90, N_OFF = 60;

    function reset() {
      angleI = 0; offI = 0; tried = 0; worked = 0; running = false; done = false; bestShown = null; trail = [];
    }
    reset();

    function lineFor(ai, oi) {
      const ang = ai / N_ANG * Math.PI;                 // 0..180 degrees covers every orientation
      const off = -1.2 + oi / (N_OFF - 1) * 2.4;        // distance from centre
      const nx = Math.cos(ang), ny = Math.sin(ang);
      const cxp = 0.5 + nx * off, cyp = 0.5 + ny * off; // point on the line
      return { ax: cxp, ay: cyp, bx: cxp - ny, by: cyp + nx };
    }

    function testOne() {
      const labels = PUZZLES[puzzle].labels;
      const L = lineFor(angleI, offI);
      for (const flip of [false, true]) {
        tried++;
        const s = scoreOf(labels, (x, y) => classify(x, y, L.ax, L.ay, L.bx, L.by, flip));
        if (s === 4) { worked++; if (!bestShown) bestShown = { L: L, flip: flip }; }
      }
      if (trail.length < 260 && angleI % 2 === 0) trail.push(L);
      offI++;
      if (offI >= N_OFF) { offI = 0; angleI++; }
      if (angleI >= N_ANG) { running = false; done = true; }
    }

    const readout = ctx.readout();
    ctx.loop(() => {
      if (running) for (let i = 0; i < 40; i++) { if (!running) break; testOne(); }

      const labels = PUZZLES[puzzle].labels;
      drawFrame(g, b, C);
      // faint trail of every line tried
      g.strokeStyle = 'rgba(148,163,184,0.11)'; g.lineWidth = 1;
      for (const L of trail) drawLineThrough(g, b, L.ax, L.ay, L.bx, L.by, 'rgba(148,163,184,0.11)', 1);
      // current line
      if (!done) {
        const L = lineFor(angleI, offI);
        drawLineThrough(g, b, L.ax, L.ay, L.bx, L.by, C.warn, 2);
      }
      // a winner, if one exists
      if (bestShown) {
        shadeSides(g, b, bestShown.L.ax, bestShown.L.ay, bestShown.L.bx, bestShown.L.by, bestShown.flip, C);
        drawLineThrough(g, b, bestShown.L.ax, bestShown.L.ay, bestShown.L.bx, bestShown.L.by, C.green, 3);
      }
      drawDots(g, b, labels, C, true);

      g.textAlign = 'left'; g.textBaseline = 'top';
      g.font = '12px Inter, system-ui, sans-serif'; g.fillStyle = C.muted;
      g.fillText('lines tried', 16, 14);
      g.font = '700 22px JetBrains Mono, monospace'; g.fillStyle = C.text;
      g.fillText(tried.toLocaleString(), 16, 30);
      g.font = '12px Inter, system-ui, sans-serif'; g.fillStyle = C.muted;
      g.fillText('lines that got all 4 right', 16, 62);
      g.font = '700 22px JetBrains Mono, monospace';
      g.fillStyle = worked > 0 ? C.green : C.danger;
      g.fillText(worked.toLocaleString(), 16, 78);
      if (done) {
        g.font = '700 14px Inter, system-ui, sans-serif';
        g.fillStyle = worked > 0 ? C.green : C.danger;
        g.textAlign = 'right';
        g.fillText(worked > 0 ? 'plenty of lines work here' : 'not one line works. ever.', b.W - 16, 18);
      }
      g.textAlign = 'center'; g.textBaseline = 'middle';
      readout.set({ 'tried': tried.toLocaleString(), 'worked': worked.toLocaleString(), 'status': done ? 'finished' : running ? 'sweeping' : 'ready' });
    });

    const runBtn = ctx.button('Try every line', () => {
      if (done) reset();
      running = !running;
      runBtn.textContent = running ? 'Pause' : (done ? 'Try every line' : 'Resume');
    }, 'primary');

    const sel = ctx.select({
      label: 'puzzle',
      options: [
        { value: 'AND', label: '1 · both switches on' },
        { value: 'OR', label: '2 · either switch on' },
        { value: 'XOR', label: '3 · exactly one switch on' },
      ],
      value: 'AND',
      onChange: (v) => { puzzle = v; reset(); runBtn.textContent = 'Try every line'; },
    });

    return ctx.figure(cv,
      'Every orientation and position, both ways round. On puzzle 1 the winner counter climbs into the thousands. On puzzle 3 it stays on zero from the first line to the last.',
      [sel, runBtn, ctx.button('Reset', () => { reset(); runBtn.textContent = 'Try every line'; })],
      readout);
  }

  /* ==================================================================
     4 — TWO LINES FIX IT
     ================================================================== */
  function buildTwoLines(ctx) {
    const C = ctx.colors;
    const [cv, g] = ctx.canvas(640, 460);
    const b = makeBoard(cv, g, C);
    const nLines = ctx.slider({ label: 'how many lines', min: 1, max: 2, step: 1, value: 1 });
    // line 1: x + y = 0.5   (above it means "at least one switch on")
    // line 2: x + y = 1.5   (below it means "not both on")
    const L1 = { ax: 0.5, ay: 0.0, bx: 0.0, by: 0.5 };
    const L2 = { ax: 1.5, ay: 0.0, bx: 0.0, by: 1.5 };
    const readout = ctx.readout();

    ctx.loop(() => {
      const labels = PUZZLES.XOR.labels;
      const two = nLines.value >= 2;
      drawFrame(g, b, C);

      const inside = (x, y) => {
        const above1 = (x + y) > 0.5;
        if (!two) return above1 ? 1 : 0;
        const below2 = (x + y) < 1.5;
        return (above1 && below2) ? 1 : 0;
      };

      const step = 9;
      for (let X = b.x0; X < b.x0 + b.size; X += step) {
        for (let Y = b.y0; Y < b.y0 + b.size; Y += step) {
          const on = inside(b.ix(X + step / 2), b.iy(Y + step / 2));
          g.fillStyle = on ? 'rgba(251,113,133,0.15)' : 'rgba(124,156,255,0.13)';
          g.fillRect(X, Y, step, step);
        }
      }
      drawLineThrough(g, b, L1.ax, L1.ay, L1.bx, L1.by, C.green, 3);
      if (two) drawLineThrough(g, b, L2.ax, L2.ay, L2.bx, L2.by, C.purple, 3);
      drawDots(g, b, labels, C, true);

      const score = scoreOf(labels, inside);
      g.textAlign = 'left'; g.textBaseline = 'top';
      g.font = '700 26px Inter, system-ui, sans-serif';
      g.fillStyle = score === 4 ? C.green : C.text;
      g.fillText(score + ' / 4', 16, 14);
      g.font = '12px Inter, system-ui, sans-serif'; g.fillStyle = C.muted;
      g.fillText(two ? 'two lines make a stripe' : 'one line, the best it can do', 16, 46);
      g.textAlign = 'right';
      g.font = '600 12px Inter, system-ui, sans-serif';
      g.fillStyle = C.green; g.fillText('line 1: at least one switch on', b.W - 16, 16);
      if (two) { g.fillStyle = C.purple; g.fillText('line 2: not both switches on', b.W - 16, 34); }
      g.textAlign = 'center'; g.textBaseline = 'middle';
      readout.set({ 'lines': two ? 2 : 1, 'correct': score + ' of 4' });
    });

    return ctx.figure(cv,
      'One line tops out at 3 of 4. Add a second and the red band between them holds exactly the two red dots. Each line is one neuron; a third neuron checks whether you are inside both.',
      [nLines],
      readout);
  }

  /* ==================================================================
     5 — INSIDE THE NEURON
     ================================================================== */
  function buildNeuronAnatomy(ctx) {
    const C = ctx.colors;
    const [cv, g] = ctx.canvas(700, 300);
    const x1 = ctx.slider({ label: 'input A', min: 0, max: 1, step: 1, value: 1 });
    const x2 = ctx.slider({ label: 'input B', min: 0, max: 1, step: 1, value: 1 });
    const w1 = ctx.slider({ label: 'weight on A', min: -1, max: 1, step: 0.05, value: 0.5, digits: 2 });
    const w2 = ctx.slider({ label: 'weight on B', min: -1, max: 1, step: 0.05, value: 0.5, digits: 2 });
    const bi = ctx.slider({ label: 'bias', min: -2, max: 1, step: 0.05, value: -0.75, digits: 2 });
    const readout = ctx.readout();

    ctx.loop(() => {
      g.clearRect(0, 0, cv.W, cv.H);
      g.fillStyle = C.bg; g.fillRect(0, 0, cv.W, cv.H);
      const a = x1.value, bb = x2.value, wa = w1.value, wb = w2.value, bias = bi.value;
      const sum = a * wa + bb * wb + bias;
      const out = sum > 0 ? 1 : 0;

      const cy = 150, nodeX = 430, outX = 610;
      const inY = [95, 205];
      const vals = [a, bb], ws = [wa, wb], names = ['A', 'B'];

      g.textAlign = 'center'; g.textBaseline = 'middle';
      for (let i = 0; i < 2; i++) {
        // input circle
        g.fillStyle = vals[i] ? C.accent : '#1b2436';
        g.beginPath(); g.arc(110, inY[i], 26, 0, Math.PI * 2); g.fill();
        g.strokeStyle = C.line; g.lineWidth = 2; g.stroke();
        g.fillStyle = vals[i] ? '#0a0e16' : C.muted;
        g.font = '700 18px Inter, system-ui, sans-serif';
        g.fillText(String(vals[i]), 110, inY[i]);
        g.fillStyle = C.muted; g.font = '12px Inter, system-ui, sans-serif';
        g.fillText('input ' + names[i], 110, inY[i] - 44);

        // wire, thickness and colour by weight
        const wgt = ws[i];
        g.strokeStyle = wgt >= 0 ? 'rgba(251,113,133,' + (0.25 + Math.abs(wgt) * 0.7) + ')'
                                 : 'rgba(124,156,255,' + (0.25 + Math.abs(wgt) * 0.7) + ')';
        g.lineWidth = 1 + Math.abs(wgt) * 9;
        g.beginPath(); g.moveTo(136, inY[i]); g.lineTo(nodeX - 46, cy); g.stroke();

        // weight label
        const mx = (136 + nodeX - 46) / 2, my = (inY[i] + cy) / 2;
        g.fillStyle = '#0f1520'; g.fillRect(mx - 30, my - 11, 60, 22);
        g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(mx - 30, my - 11, 60, 22);
        g.fillStyle = wgt >= 0 ? C.danger : C.accent;
        g.font = '600 12px JetBrains Mono, monospace';
        g.fillText((wgt >= 0 ? '+' : '') + wgt.toFixed(2), mx, my);
      }

      // the neuron
      g.fillStyle = '#131c2e'; g.strokeStyle = out ? C.green : C.line; g.lineWidth = 3;
      g.beginPath(); g.arc(nodeX, cy, 46, 0, Math.PI * 2); g.fill(); g.stroke();
      g.fillStyle = C.text; g.font = '600 12px Inter, system-ui, sans-serif';
      g.fillText('add it up', nodeX, cy - 14);
      g.font = '700 19px JetBrains Mono, monospace';
      g.fillStyle = sum > 0 ? C.green : C.danger;
      g.fillText((sum >= 0 ? '+' : '') + sum.toFixed(2), nodeX, cy + 12);
      g.fillStyle = C.muted; g.font = '11px Inter, system-ui, sans-serif';
      g.fillText('bias ' + (bias >= 0 ? '+' : '') + bias.toFixed(2), nodeX, cy + 66);

      // output
      g.strokeStyle = out ? C.green : 'rgba(148,163,184,0.4)'; g.lineWidth = out ? 5 : 2;
      g.beginPath(); g.moveTo(nodeX + 48, cy); g.lineTo(outX - 30, cy); g.stroke();
      g.fillStyle = out ? C.green : '#1b2436';
      g.beginPath(); g.arc(outX, cy, 28, 0, Math.PI * 2); g.fill();
      g.strokeStyle = C.line; g.lineWidth = 2; g.stroke();
      g.fillStyle = out ? '#0a0e16' : C.muted;
      g.font = '700 20px Inter, system-ui, sans-serif';
      g.fillText(String(out), outX, cy);
      g.fillStyle = C.muted; g.font = '12px Inter, system-ui, sans-serif';
      g.fillText(out ? 'FIRES' : 'silent', outX, cy - 46);

      // the sentence
      g.textAlign = 'left';
      g.font = '13px JetBrains Mono, monospace'; g.fillStyle = C.muted;
      g.fillText('(' + a + ' × ' + wa.toFixed(2) + ') + (' + bb + ' × ' + wb.toFixed(2) + ') + ' + bias.toFixed(2)
                 + '  =  ' + sum.toFixed(2) + '   ' + (sum > 0 ? '> 0, so fire' : '≤ 0, so stay quiet'), 24, cv.H - 26);
      g.textAlign = 'center';

      readout.set({ 'total': sum.toFixed(2), 'output': out });
    });

    return ctx.figure(cv,
      'Thicker wire means a bigger weight; red pushes toward firing and blue pushes against it. The bias is how much evidence the neuron demands before it fires at all. Three numbers, and that is the whole neuron.',
      [x1, x2, w1, w2, bi],
      readout);
  }

  /* ==================================================================
     6 — NESTED CIRCLES
     ================================================================== */
  function buildNesting(ctx) {
    const C = ctx.colors;
    const [cv, g] = ctx.canvas(700, 340);
    const RINGS = [
      { r: 158, color: C.accent, name: 'Artificial intelligence', since: 'the term was coined in 1956',
        body: 'Any machine doing something we would call clever. Includes hand-written rules with no learning at all, like a chess engine or a tax calculator.' },
      { r: 112, color: C.green, name: 'Machine learning', since: 'took hold from the 1980s',
        body: 'The subset that learns from examples instead of being told the rules. Includes plenty of methods with no neurons anywhere, like decision trees.' },
      { r: 62, color: C.pink, name: 'Deep learning', since: 'took over from 2012',
        body: 'Machine learning using many stacked layers of neurons. This is the part that produced image recognition, translation, and the model you are talking to.' },
    ];
    let hover = 2;
    const info = ctx.h('div', { style: { minHeight: '74px' } });

    // One handler, wired to both events. Never synthesise a fake event to
    // re-enter your own listener: it is easy to build one that re-triggers the
    // handler that created it.
    const pick = (e) => {
      const p = cv.pos(e);
      const d = Math.hypot(p.x - 210, p.y - 170);
      hover = d <= RINGS[2].r ? 2 : d <= RINGS[1].r ? 1 : d <= RINGS[0].r ? 0 : hover;
    };
    cv.addEventListener('pointermove', pick);
    cv.addEventListener('pointerdown', pick);

    function paintInfo() {
      const r = RINGS[hover];
      info.innerHTML = '';
      info.append(
        ctx.h('div', { style: { color: r.color, fontWeight: '700', fontSize: '1.05rem' } }, r.name),
        ctx.h('div', { style: { color: '#94a3b8', fontSize: '.8rem', marginBottom: '4px' } }, r.since),
        ctx.h('div', { html: r.body }));
    }

    ctx.loop(() => {
      g.clearRect(0, 0, cv.W, cv.H);
      g.fillStyle = C.bg; g.fillRect(0, 0, cv.W, cv.H);
      const cx = 210, cy = 170;
      for (let i = 0; i < RINGS.length; i++) {
        const r = RINGS[i], on = hover === i;
        g.fillStyle = on ? r.color + '2e' : r.color + '14';
        g.beginPath(); g.arc(cx, cy, r.r, 0, Math.PI * 2); g.fill();
        g.strokeStyle = r.color; g.lineWidth = on ? 3 : 1.5;
        g.stroke();
      }
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.font = '600 12px Inter, system-ui, sans-serif';
      g.fillStyle = RINGS[0].color; g.fillText('AI', cx, cy - 138);
      g.fillStyle = RINGS[1].color; g.fillText('machine learning', cx, cy - 94);
      g.fillStyle = RINGS[2].color; g.fillText('deep', cx, cy - 8); g.fillText('learning', cx, cy + 8);

      // side text
      g.textAlign = 'left';
      const r = RINGS[hover];
      g.fillStyle = r.color; g.font = '700 17px Inter, system-ui, sans-serif';
      g.fillText(r.name, 410, 96);
      g.fillStyle = C.muted; g.font = '12px Inter, system-ui, sans-serif';
      g.fillText(r.since, 410, 118);
      g.fillStyle = C.text; g.font = '13px Inter, system-ui, sans-serif';
      wrap(g, r.body, 410, 148, 260, 19);
      g.textAlign = 'center';
      paintInfo();
    });

    function wrap(g, text, x, y, maxW, lh) {
      const words = text.split(' ');
      let line = '', yy = y;
      for (const w of words) {
        const test = line ? line + ' ' + w : w;
        if (g.measureText(test).width > maxW && line) { g.fillText(line, x, yy); line = w; yy += lh; }
        else line = test;
      }
      if (line) g.fillText(line, x, yy);
    }

    return ctx.figure(cv,
      'Each one sits inside the last. Every deep-learning system is machine learning, and every machine-learning system is AI, but not the other way round.',
      null, null);
  }
})();
