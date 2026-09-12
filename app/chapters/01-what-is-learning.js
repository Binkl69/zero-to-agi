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

      /* ---------- the same move, on something that matters ---------- */
      root.append(ctx.section('Now do it where it counts',
        ctx.p(`Two switches and a light is a laboratory. Here is the identical move — one line, two handles — on decisions people actually pay for.`),
        ctx.callout('tryit', '🖐 Try this — and notice what you cannot do',
          `<b>1.</b> Start on <b>spam</b>. Drag until you are happy, then press <b>Show the best line there is</b> — the demo brute-forces every line there is and tells you the true ceiling.<br>
           <b>2.</b> That ceiling is <b>91.2%</b>, not 100%. Not because you drag badly: real data overlaps. Some real emails shout, some spam is polite.<br>
           <b>3.</b> Now move the line so you catch <i>every</i> spam, and watch how many real emails you condemn to the junk folder. <b>There is no setting that zeroes both counts.</b><br>
           <b>4.</b> Switch to <b>card fraud</b> and try properly. Its ceiling is <b>80.1%</b> — and guessing "genuine" every single time already scores 58.8%. Then read why.`),
        buildRealData(ctx),
        ctx.p(`Three lessons arrive at once, and none of them were visible in the toy.`),
        ctx.p(`<b>Perfect is a property of toys.</b> Four dots split 4 out of 4. A hundred real ones do not — spam tops out at 91.2%, the tumours at 92.5% — and the interesting number stops being "did I win" and becomes "how close to the ceiling am I".`),
        ctx.p(`<b>Not all mistakes are the same mistake.</b> Letting fraud through costs money. Freezing an honest customer's card at a petrol station costs a customer. The line you pick <i>is</i> that trade-off, and no amount of cleverness removes the choice — it is a business decision wearing a maths costume. Chapter 3 gives it names: precision and recall.`),
        ctx.p(`<b>And the fraud one really is beyond a line</b>, for exactly the reason puzzle 3 was. Fraud lives at <i>both</i> extremes: tiny "card testing" payments to check a stolen number still works, and one large cash-out. Genuine spending sits in the middle.`),
        ctx.p(`One line cannot cut both ends off a stick and leave the middle. That is XOR wearing a suit, and it costs real banks real money. Note that 80.1% means nothing until you know what guessing "genuine" every time already scores: 58.8%. Read together they say something real — 80.1% cuts the mistakes from 56 in 136 to 27 — and reading them together is the whole skill, because <b>and still mean the model has learned almost nothing.</b> Chapter 3 makes that trap explicit.`),
      ));

      /* ---------- maths beat 1: notation for the line they just dragged ---------- */
      root.append(ctx.section('What you have been doing, written down',
        ctx.p(`You have now dragged a line perhaps thirty times. There is a standard way to write down what you were adjusting, and you have already built every piece of it by hand.`),
        ctx.p(`Look at the panel on the right of the demo above, under <b>the line, as three numbers</b>. Those three numbers move as you drag. That is the whole formula, live.`),
        ctx.decoder([
          { sym: 'w<sub>1</sub>', name: 'w one', says: 'How much the <b>first</b> thing counts toward a yes. Big number, it matters a lot; near zero, the model has decided it is irrelevant; negative, it counts <i>against</i>.', points: 'the w₁ readout above. Rotate the line and watch it move.' },
          { sym: 'x<sub>1</sub>', name: 'x one', says: 'The <b>first</b> measurement of the thing in front of you — how many links this particular email has.', points: 'the horizontal position of one dot.' },
          '+',
          { sym: 'w<sub>2</sub>', name: 'w two', says: 'Same idea for the second thing. In the spam example, how much SHOUTING counts.', points: 'the w₂ readout above.' },
          { sym: 'x<sub>2</sub>', name: 'x two', says: 'The second measurement — how much of this email is capitals.', points: 'the vertical position of that same dot.' },
          '+',
          { sym: 'b', name: 'b, or "bias"', says: 'How suspicious the model is <b>before it looks at anything</b>. Drag the line further from the origin and this is what changes. It is the thumb on the scale.', points: 'the b readout above.' },
          '>',
          { sym: '0', name: 'zero', says: 'The finish line. Above it, the model says yes; below it, no. The line you were dragging is exactly the set of points where this sum comes out to zero — the border between the two verdicts.', points: 'the white line itself.' },
        ], {
          title: 'w₁x₁ + w₂x₂ + b > 0',
          hint: 'Click any symbol. Every one of them is something you have already moved with your hands.',
          plain: '"Weigh up each thing by how much it matters, add a standing level of suspicion, and if the total clears zero, say yes."',
        }),
        ctx.walkthrough([
          { say: 'Forget neurons for a moment. You are deciding whether to take an umbrella.', note: 'You do this without thinking. We are going to write down what you did.' },
          { say: 'You look at two things: <b>how dark the clouds are</b>, and <b>what day of the week it is</b>.' },
          { ask: 'Which of those should count toward "take an umbrella"?',
            options: ['Both equally', 'Cloud darkness a lot, the day not at all', 'The day of the week'], answer: 1,
            explain: 'Obviously the clouds. And "irrelevant" is a number here, not a shrug — it is zero.' },
          { say: 'So give each one a multiplier saying how much it counts. <b>That multiplier is the weight.</b>', math: 'clouds &times; 3 &nbsp;&nbsp;+&nbsp;&nbsp; day &times; 0' },
          { say: 'Weight 3 means it matters a lot. Weight 0 means the model has learned to ignore it. A <i>negative</i> weight means it counts <b>against</b>.', note: 'Nobody sets these by hand. Training is the search for them — that is what you watched the trainer doing.' },
          { say: 'Now the second number. Two people look at the same slightly-grey sky. One takes an umbrella, one does not.', note: 'Same evidence. Different answer.' },
          { ask: 'What is different between those two people?',
            options: ['They weigh the clouds differently', 'How much they mind getting wet, before looking outside at all', 'One of them is simply wrong'], answer: 1,
            explain: 'They start from different places. One is umbrella-inclined before any evidence arrives at all. That standing lean is the bias.' },
          { say: '<b>That is the bias.</b> Where you start, before any evidence. And it is <b>added</b> at the end — never multiplied by anything.', math: 'clouds &times; 3 &nbsp;+&nbsp; day &times; 0 &nbsp;<b>+</b>&nbsp; 2' },
          { say: 'Here is the proof that a bias has to exist. Later in this chapter there is a neuron with all three numbers on sliders and the line they describe drawn beside them. Drag its <b>bias</b> to exactly 0.', note: 'The line jumps to the bottom-left corner and is stuck there. The weights can spin it, but nothing can move it off that corner until the bias is non-zero.' },
          { say: 'With bias at zero the line can spin freely — but it is <b>stuck passing through the corner of the grid.</b> It can never move away from it.', note: 'The weights set the angle. The bias is the only thing that can slide the line across the page.' },
          { ask: 'So what does the bias do, geometrically?',
            options: ['Rotates the line', 'Slides the line away from the corner', 'Makes the line curve'], answer: 1,
            explain: 'Weights turn it, the bias moves it. Without a bias, every decision boundary in every model would be pinned to the corner — and almost nothing useful lives there.' },
          { say: 'These two words never change meaning again, in any chapter, at any size.', note: 'A frontier model holds somewhere between a few hundred billion and a few trillion of them — no frontier lab publishes the exact number any more. They are these ones, in unimaginable number, each still doing exactly this job.' },
        ], {
          title: 'Weight and bias, with an umbrella',
          recap: 'a weight is how much one thing counts. A bias is where you start before anything counts. Weights are multiplied; the bias is added.',
        }),
        ctx.callout('key', '🔑 "Is that really the whole field?"',
          `A fair question, and the honest answer is <b>yes — at the level of the mechanism.</b> Multiply each input by how much it counts, add them up, add a bias, compare to zero, and adjust when wrong. There is no secret extra ingredient waiting in chapter 12.<br>
           What is <i>not</i> simple is everything built on top: <b>how you wire billions of these together</b> (chapters 4–7), <b>what you feed them</b> (chapter 10), <b>what you point them at</b> (chapters 9 and 11), and <b>getting it to run at all</b> (chapter 12).<br>
           A brick is fired clay in a rectangle. That genuinely is the whole of a brick, and it tells you nothing whatever about how to build a cathedral.`),
        ctx.p(`One thing worth noticing: writing two symbols side by side means multiply them, so the whole left-hand side is "multiply each measurement by its importance, then add everything up". You will meet that pattern so often it gets its own name in chapter 7 — the <em>dot product</em> — and it is already the single most common operation in all of AI.`),
      ));

      /* ---------- what just happened ---------- */
      root.append(ctx.section('What you just were',
        ctx.p(`You were a <em>neuron</em>. Not a metaphor for one, the actual thing. An artificial neuron has exactly one move available to it: <b>draw a straight line and call one side "yes" and the other side "no"</b>. That is its entire repertoire.`),
        ctx.p(`Puzzles 1 and 2 fell over easily. Puzzle 3 did not, and it is worth being precise about why: <b>it is not hard, it is impossible.</b> The two red dots sit in opposite corners with the blue dots in the other two, so any straight line you draw strands somebody on the wrong side. You cannot win, and neither can any machine that only draws one line.`),
        ctx.callout('key', '🔑 The one idea in this chapter',
          `One neuron equals one straight line. Some problems cannot be split with one straight line. That single sentence explains a machine demonstrated in 1958, why neural networks were abandoned in 1969, and why the word "deep" appears in "deep learning".`),
      ));

      /* ---------- now the machine does it ---------- */
      root.append(ctx.section('Now let the machine find the line',
        ctx.p(`You moved the line by feel. A machine has no feel, so it uses a rule so simple you could do it on paper: <b>look at one dot; if you got it wrong, shove the line a little bit toward getting it right; repeat.</b> Nothing cleverer than that.`),
        ctx.callout('tryit', '🖐 Try this',
          `Press <b>Train</b> and watch the line stagger toward an answer on puzzle 1. Then switch to <b>puzzle 3</b> — it stops, so press <b>Train</b> again and leave it running. Watch the mistake counter at the bottom: it wobbles for a few passes and then locks solid at 4 mistakes out of 4. Every dot wrong, every pass, for as long as you care to watch. It is not thinking. It is stuck in a loop, redoing the same moves forever.`),
        buildTrainer(ctx),
        ctx.p(`That flailing is the machine doing exactly what it was told, forever: the same four corrections, undone and redone. It is not broken and it is not slow. It is looking for something that does not exist.`),
      ));

      /* ---------- maths beat 2: the rule the machine uses to learn ---------- */
      root.append(ctx.section('How the machine actually uses those numbers',
        ctx.p(`Earlier you wrote down how a neuron <b>decides</b>. That is only half of it, and it is the half that does not learn. A pocket calculator can decide. The other half is the rule that <b>changes</b> the numbers when the decision comes out wrong — and it is three lines long.`),
        ctx.p(`Watch the readout under the trainer while it runs. <b>w₁</b>, <b>w₂</b> and <b>b</b> are the three numbers from the previous section, and they are moving. This is what is moving them.`),
        ctx.walkthrough([
          { say: 'The machine looks at <b>one dot</b>. Just one, then the next, then the next.', note: 'The trainer above does this eight times a second at the default speed, and thirty at the top of the slider. Press One step to see a single one.' },
          { say: 'It makes a guess about that dot: <b>yes</b> or <b>no</b>, using the line where it currently sits.' },
          { say: 'Then it compares. The true answer is called <b>y</b>. Its guess is called <b>ŷ</b> — "y-hat". The hat means "this one is estimated".', note: 'You will see that hat on every prediction for the rest of the course.' },
          { say: 'Now subtract one from the other. There are only three possible answers, and that is the whole of it.', math: 'y &minus; &#375; &nbsp;=&nbsp; +1, &nbsp; 0, &nbsp; or &nbsp;&minus;1' },
          { ask: 'The guess was <b>right</b>. So y and ŷ are the same number. What is y &minus; ŷ?',
            options: ['+1', '0', '&minus;1'], answer: 1,
            explain: 'Zero. And here is the consequence: the whole correction gets multiplied by that zero, so NOTHING CHANGES. The machine only ever learns from mistakes.' },
          { say: 'When it <i>is</i> wrong, that bracket is +1 or &minus;1 — and the sign says which way the line needs to move.', note: '+1 means it said no and should have said yes. &minus;1 is the other way round.' },
          { say: 'So: take the weight you had, and move it a little in that direction.', math: 'w &nbsp;&larr;&nbsp; w &nbsp;+&nbsp; (a little, in that direction)' },
          { say: 'How little? That is set by one number, <b>&eta;</b> — "eta", the <b>learning rate</b>. Here it is fixed at 0.12.', note: 'Too big and the line thrashes about. Too small and it crawls. Chapter 3 is largely about getting this number right.' },
          { say: 'One last piece. The correction is also multiplied by the input itself, <b>x</b>.', note: 'So an input that pushed hard toward the wrong answer gets corrected hard, and an input near zero barely moves. Blame, in proportion to who caused it.' },
          { say: 'Put the four pieces together and you have the entire learning algorithm.', math: 'w &nbsp;&larr;&nbsp; w &nbsp;+&nbsp; &eta; (y &minus; &#375;) x' },
          { ask: 'Run the trainer on puzzle 1 until it settles. From then on the readout shows <b>0 — no change</b> at every step and the three numbers freeze. Why?',
            options: ['The demo has paused', 'It is getting those dots right, so the correction is zero', 'The learning rate is too small'], answer: 1,
            explain: 'Exactly. Right answers produce no learning at all. All the movement you see comes from the mistakes.' },
        ], {
          title: 'How the machine changes its mind, one piece at a time',
          recap: 'look at one example, subtract the guess from the truth, and if they differ, shove each weight a little in that direction.',
        }),
        ctx.p(`Here is that same rule as a reference you can come back to. Click any piece of it.`),
        ctx.decoder([
          { sym: 'w', name: 'w', says: 'One of the numbers the machine is allowed to change. There are three here: w₁, w₂ and b.', points: 'the w₁, w₂ and b readouts under the trainer.' },
          { sym: '←', name: 'becomes', says: 'Not "equals". <b>Replace the old value with this new one.</b> It is an instruction, carried out over and over — eight times a second in the demo you just ran, or thirty with the speed slider pushed up.', points: 'the numbers ticking over as it trains.' },
          { sym: 'w', name: 'w again', says: 'The value it had a moment ago. Every step builds on the last one; nothing starts fresh.' },
          '+',
          { sym: '&eta;', name: 'eta — the learning rate', says: 'How big a shove to give. In this demo it is fixed at <b>0.12</b>: small enough not to overshoot, big enough to get somewhere. Chapter 3 is largely about what happens when you get this number wrong.', points: 'nothing visible — it is the one number you cannot see moving, because it never moves.' },
          '(',
          { sym: 'y', name: 'y — the truth', says: 'What the answer <i>should</i> have been for the dot it is looking at right now. 1 or 0.', points: 'the colour of the dot.' },
          '&minus;',
          { sym: 'ŷ', name: 'y-hat — the guess', says: 'What the neuron actually said. The little hat means "estimated", and you will see it on every prediction in this course from here on.', points: 'which side of the line that dot currently falls on.' },
          ')',
          { sym: 'x', name: 'x — the input', says: 'The measurement itself. This is the clever part: the correction is <b>scaled by the input</b>, so an input that pushed hard toward the wrong answer gets corrected hard, and an input that was near zero barely moves at all.', points: 'the dot&rsquo;s position along one axis — its horizontal position when correcting w&#8321;, its vertical position when correcting w&#8322;. A dot at x&#8321; = 0 corrects w&#8321; not at all.' },
        ], {
          title: 'w ← w + η (y − ŷ) x',
          hint: 'Click any symbol. This is the entire learning algorithm — every one of these is something you watched happen.',
          plain: 'You got one wrong. Nudge each number a little, so that if the same example came round again you would be closer to right. Nudge the numbers attached to <b>big</b> inputs more, because those inputs did more of the damage.',
        }),
        ctx.callout('key', '🔑 The bit worth sitting with: (y − ŷ)',
          `That bracket can only be three things. <b>y − ŷ = 0</b> when the guess was right — and then the whole correction is zero, so <b>nothing changes at all</b>.
           <b>+1</b> when it said no and should have said yes. <b>−1</b> the other way.<br>
           So the machine <b>only ever learns from its mistakes</b>. When it is right it does not even pat itself on the back; it does nothing. You can watch this directly: on puzzles 1 and 2, once the line is found, the "last error" line in the readout shows <code class="inline">0 — no change</code> at every step and the numbers freeze.<br>
           The whole of modern AI is a more sophisticated answer to the question buried in that bracket: <b>how wrong were we, and which direction is less wrong?</b>`),
        ctx.p(`Three things in that rule carry forward without ever changing meaning, so they are worth naming now.`),
        ctx.ul([
          `<b>η, the learning rate.</b> Fixed at 0.12 here, and the single most fiddly number in all of machine learning. Chapter 2 lets you set it too high and watch a model explode.`,
          `<b>ŷ, the prediction.</b> The hat notation is universal. <em>y</em> is truth, <em>ŷ</em> is what the model said.`,
          `<b>(y − ŷ), the error.</b> Chapter 2 replaces this crude "am I wrong, yes or no" with a smooth number that says <i>how</i> wrong and <i>in which direction</i> — and that one upgrade is what makes it possible to train more than one layer.`,
        ]),
        ctx.callout('example', '🌍 This exact rule, in 1958 hardware',
          `Rosenblatt's Mark I Perceptron ran this update physically. The weights were <b>potentiometers</b> — little variable resistors — and the machine adjusted them with electric motors.
           When it got an answer wrong, motors turned the dials. <code class="inline">w ← w + η(y − ŷ)x</code> was not a line of code; it was a shaft rotating.<br>
           The formula you just decoded is old enough to have been implemented in brass and copper, and it is still, in a much-refined form, what is happening inside every model you use today.`),
      ));

      /* ---------- proof ---------- */
      root.append(ctx.section('Proof that it is impossible, not just difficult',
        ctx.p(`Maybe the machine is just bad at searching? Settle it by trying <b>every line there is</b>. Below, the computer sweeps through thousands of lines at every angle and position, and tallies how many get all four dots right.`),
        ctx.callout('tryit', '🖐 Try this',
          `Run it on puzzle 1, then on puzzle 3. Compare the two counters at the end.`),
        buildBruteForce(ctx),
        ctx.p(`Out of 10,800 lines tried, <b>414</b> work for puzzle 1 and <b>zero</b> work for puzzle 3. That is not the machine giving up early. There is no answer of that shape anywhere.`),
        ctx.callout('history', '📜 The book that froze the field',
          `In 1969 Marvin Minsky and Seymour Papert published <i>Perceptrons</i>, proving exactly this on paper. Frank Rosenblatt's perceptron — demonstrated to the press in 1958, then built as dedicated hardware — could never learn XOR. Funding for neural networks dried up and the approach went cold for over a decade. (The better-known <em>first AI winter</em> came a few years later, 1974–80, when funding collapsed across the whole of AI after the Lighthill report; the 1969 freeze was the connectionist one, and it lasted longer.) A period now called the first <em>AI winter</em>. The irony is that the fix was already understood in principle. Nobody yet knew how to train it.`),
      ));

      /* ---------- the fix ---------- */
      root.append(ctx.section('The fix: stop using one line',
        ctx.p(`If one line cannot do it, use two. Two lines carve the square into a <b>stripe</b>, and the stripe can hold both red dots while excluding both blue ones.`),
        ctx.callout('tryit', '🖐 Try this',
          `Drag the slider from 1 line to 2 lines and watch the impossible puzzle become possible.`),
        buildTwoLines(ctx),
        ctx.p(`Each line is one neuron. To combine them you need a third neuron that watches the first two and answers "am I between them?". That stack is a <em>network</em>, and the middle row is a <em>hidden layer</em>. Add more neurons to that middle row and you can cut out any shape at all. What extra <em>layers</em> buy is doing it efficiently — each layer hands the next an easier version of the question, instead of one enormous row brute-forcing the whole thing — and that is where the "deep" in deep learning comes from.`),
        ctx.p(`Which raises a fair objection: <b>there are only two lines on that picture. Where is the third one?</b>`),
        ctx.callout('tryit', '🖐 Try this — go and find the third line',
          `It is real, but it is not in that square — and a third line drawn <i>there</i> could not help anyway, because this chapter opened by proving no straight line in that square works.<br>
           <b>1.</b> Press <b>▶ Watch them move</b> and read the commentary as it goes.<br>
           <b>2.</b> Stop at the end and look at the dots. <b>One pair has landed on the same spot</b>, leaving three positions instead of four. Nothing was lost — A and B simply could not tell those two apart, so they ended up in the same place.<br>
           <b>3.</b> Now look at the green line. In <i>this</i> space, one straight line does the job.`),
        buildThirdNeuron(ctx),
        ctx.p(`So a hidden layer does not solve the problem. <b>It rearranges the problem until a straight line can solve it.</b> That is the most important sentence in this chapter, and it is what "deep" means: do it again, and again, each layer handing the next an easier version of the question.`),
        ctx.callout('key', '🔑 So why did this take until the 2010s?',
          `Nobody doubted more layers were more powerful. The problem was <b>training</b> them: with a hidden layer, it is no longer obvious which weight to blame for a mistake. The answer, <em>backpropagation</em>, is chapter 2. It needed the maths to be popularised in 1986, then another twenty-five years of faster chips and bigger datasets before it paid off.`),
      ));

      /* ---------- inside the neuron ---------- */
      root.append(ctx.section('What the neuron is actually doing with numbers',
        ctx.p(`You have now met both halves as formulas. Here they are as a machine you can take apart: the same weighted sum from the first maths beat, with the three numbers on sliders instead of on a readout.`),
        ctx.callout('tryit', '🖐 Try this',
          `Move the two weight sliders and watch the line in the panel underneath rotate; move the bias and watch it slide without turning. Set the bias to exactly 0 and it snaps to the corner — that is the proof from the walkthrough earlier, in your hands. Learning <b>is</b> the search for these three numbers. There is nothing else in there.`),
        buildNeuronAnatomy(ctx),
        ctx.p(`Those three numbers are the neuron's <em>parameters</em>. Training means adjusting them until the answers come out right. A model you talk to today is this same arrangement with somewhere between a few hundred billion and a few trillion parameters instead of three.`),
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
          q: 'Why did neural-network research stall for over a decade after 1969?',
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

  /* ------------------------------------------------------------------ */
  /* Interactive: the same line, on decisions that actually matter.      */
  /* Three real-shaped datasets. Two a line handles well. One it cannot, */
  /* for a reason that happens in actual businesses rather than in a     */
  /* logic puzzle.                                                       */
  /* ------------------------------------------------------------------ */
  function seeded(n) {
    let a = n >>> 0;
    return () => { a += 0x6D2B79F5; let t = a; t = Math.imul(t ^ (t >>> 15), 1 | t); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function gaussPair(r) {
    let u = 0, v = 0;
    while (u === 0) u = r(); while (v === 0) v = r();
    const m = Math.sqrt(-2 * Math.log(u));
    return [m * Math.cos(2 * Math.PI * v), m * Math.sin(2 * Math.PI * v)];
  }

  /* A count of links, a percentage, a size in millimetres and a price are all
     quantities that cannot go below zero, but a Gaussian happily produces a
     tumour of −6 mm. Every generated point is held inside the range its own
     axis advertises, so nothing is drawn off the end of the plot and no reader
     is shown a payment of −£9.63. Holding the points leaves every measured
     ceiling exactly where it was (91.2 / 92.5 / 80.1). The tumour figure reads
     92.5% rather than the older 91.7% because bestLine below now solves each
     angle exactly instead of sampling offsets on a grid that stepped past the
     best cut. Both numbers were remeasured, not assumed. */
  const hold = (v, hi) => Math.max(0, Math.min(hi, v));

  const DATASETS = {
    spam: {
      label: 'Is this email spam?',
      xName: 'links in the email', yName: 'SHOUTING (% capitals)',
      xShort: 'links', yShort: 'SHOUTING',
      xMax: 12, yMax: 60,
      pos: 'spam', neg: 'real email',
      build: () => {
        const r = seeded(21), pts = [];
        for (let i = 0; i < 70; i++) { const [a, b] = gaussPair(r); pts.push({ x: hold(2.2 + a * 1.9, 12), y: hold(12 + b * 8, 60), lab: 0 }); }
        for (let i = 0; i < 55; i++) { const [a, b] = gaussPair(r); pts.push({ x: hold(5.6 + a * 2.4, 12), y: hold(26 + b * 13, 60), lab: 1 }); }
        return pts;
      },
      note: 'A real filter uses hundreds of signals. Two is enough to see the shape of the problem.',
    },
    tumour: {
      label: 'Is this tumour malignant?',
      xName: 'size (mm)', yName: 'how irregular the edge is',
      xShort: 'size', yShort: 'edge',
      xMax: 40, yMax: 100,
      pos: 'malignant', neg: 'benign',
      build: () => {
        const r = seeded(77), pts = [];
        for (let i = 0; i < 62; i++) { const [a, b] = gaussPair(r); pts.push({ x: hold(13 + a * 4.0, 40), y: hold(32 + b * 13, 100), lab: 0 }); }
        for (let i = 0; i < 58; i++) { const [a, b] = gaussPair(r); pts.push({ x: hold(22 + a * 6.0, 40), y: hold(58 + b * 16, 100), lab: 1 }); }
        return pts;
      },
      note: 'This is close to the shape of the 1990s Wisconsin dataset, the problem a great many people learned this on.',
    },
    fraud: {
      label: 'Is this card payment fraud?',
      xName: 'amount (£)', yName: 'distance from home (km)',
      xShort: 'amount', yShort: 'distance',
      xMax: 900, yMax: 120,
      pos: 'fraud', neg: 'genuine',
      build: () => {
        const r = seeded(303), pts = [];
        /* The second axis is deliberately uninformative here: it overlaps across all three
           clusters, so it cannot rescue a line the way a tidy second feature would. The only
           real signal is amount, and for fraud that signal is bimodal. Measured ceiling for
           any straight line: 80.1%, against a 58.8% always-guess-genuine baseline. */
        for (let i = 0; i < 80; i++) { const [a, b] = gaussPair(r); pts.push({ x: hold(340 + a * 115, 900), y: hold(58 + b * 26, 120), lab: 0 }); }
        /* card testing: tiny amounts, checking a stolen number still works */
        for (let i = 0; i < 30; i++) { const [a, b] = gaussPair(r); pts.push({ x: hold(22 + a * 14, 900), y: hold(58 + b * 26, 120), lab: 1 }); }
        /* and the big-ticket cash-out at the other extreme */
        for (let i = 0; i < 26; i++) { const [a, b] = gaussPair(r); pts.push({ x: hold(775 + a * 70, 900), y: hold(58 + b * 26, 120), lab: 1 }); }
        return pts;
      },
      note: 'Fraud sits at BOTH extremes of amount. Distance from home does not separate them, so there is nothing for a line to grab.',
    },
  };

  function buildRealData(ctx) {
    const [cv, g] = ctx.canvas(720, 420);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    let key = 'spam';
    let pts = DATASETS[key].build();
    /* the line, held as two draggable endpoints in plot space (0..1 on each axis) */
    let A = { x: 0.12, y: 0.9 }, B = { x: 0.9, y: 0.12 };
    let flip = false, drag = null, showBest = false;

    const PAD = { l: 62, r: 300, t: 44, b: 58 };
    const PW = () => 720 - PAD.l - PAD.r;
    const PH = () => 420 - PAD.t - PAD.b;
    /* The plot box is clipped (see the loop), and the data runs right up to 0
       and to xMax, so the drawable range is inset by a dot radius: without it
       every point sitting on an axis limit would be sliced in half by the clip. */
    const R = 6;
    const sx = (u) => PAD.l + R + u * (PW() - 2 * R);
    const sy = (v) => PAD.t + R + (1 - v) * (PH() - 2 * R);
    const ux = (px) => (px - PAD.l - R) / (PW() - 2 * R);
    const uy = (py) => 1 - (py - PAD.t - R) / (PH() - 2 * R);
    /* handles stay inside the plot: far enough in that the 8px grab circle is
       drawn whole, and every line that separates anything still reachable */
    const HLO = 0.02, HHI = 0.98;
    const norm = (p) => ({ u: p.x / DATASETS[key].xMax, v: p.y / DATASETS[key].yMax });

    /* which side of the line a point falls on */
    function side(u, v) {
      const s = (B.x - A.x) * (v - A.y) - (B.y - A.y) * (u - A.x);
      return flip ? s < 0 : s > 0;
    }
    function stats() {
      let tp = 0, fp = 0, fn = 0, tn = 0;
      for (const p of pts) {
        const n = norm(p), said = side(n.u, n.v);
        if (p.lab === 1 && said) tp++;
        else if (p.lab === 0 && said) fp++;
        else if (p.lab === 1 && !said) fn++;
        else tn++;
      }
      return { tp, fp, fn, tn, acc: (tp + tn) / pts.length };
    }
    /* brute-force the best straight line, so "best possible" is measured, not asserted */
    /* The text promises this searches every line there is, so it has to. A grid
       over angle AND offset does not: it steps past the best offset and reports
       a ceiling a point or so under the truth. For a fixed angle the optimal
       cut can be found exactly — project every point onto the normal, sort, and
       the best threshold is one of the gaps — so only the angle is sampled, and
       finely. Both orientations are tried, as before. */
    function bestLine() {
      const P = pts.map(p => { const n = norm(p); return { s: 0, u: n.u, v: n.v, lab: p.lab }; });
      const N = P.length, ones = P.filter(p => p.lab === 1).length;
      let best = null;
      const keep = (ok, th, t, fl) => { if (!best || ok > best.ok) best = { ok, th, t, fl }; };
      for (let ang = 0; ang < 180; ang += 0.25) {
        const th = ang * Math.PI / 180, nx = Math.cos(th), ny = Math.sin(th);
        for (const p of P) p.s = p.u * nx + p.v * ny;
        const proj = P.slice().sort((a, b) => a.s - b.s);
        /* run = how many are right under "call it positive when the projection
           is above the threshold"; N − run is the same line, read the other way */
        let run = ones, t = proj[0].s - 0.01;
        keep(run, th, t, true); keep(N - run, th, t, false);
        for (let j = 0; j < proj.length; j++) {
          run += proj[j].lab === 1 ? -1 : 1;
          t = j + 1 < proj.length ? (proj[j].s + proj[j + 1].s) / 2 : proj[j].s + 0.01;
          keep(run, th, t, true); keep(N - run, th, t, false);
        }
      }
      /* turn the winning (angle, threshold) back into two points on the line,
         which is what the rest of the demo and the drag handles work with */
      const nx = Math.cos(best.th), ny = Math.sin(best.th);
      const fx = nx * best.t, fy = ny * best.t;
      const a = { x: fx + ny * 2, y: fy - nx * 2 };
      const b = { x: fx - ny * 2, y: fy + nx * 2 };
      return { ok: best.ok, a, b, fl: best.fl };
    }
    let cachedBest = null;
    const getBest = () => { if (!cachedBest) cachedBest = bestLine(); return cachedBest; };

    /* bestLine() describes its answer with two points two whole plot-widths
       apart, which is fine as geometry and useless as a pair of draggable
       handles — dropped straight in they land hundreds of pixels off the
       canvas. Trim the line to where it crosses the plot instead. The line
       through the trimmed pair is the same line, so the verdict is identical. */
    function trimToPlot(a, b) {
      const dx = b.x - a.x, dy = b.y - a.y;
      let t0 = -1e9, t1 = 1e9;
      const p = [-dx, dx, -dy, dy];
      const q = [a.x - HLO, HHI - a.x, a.y - HLO, HHI - a.y];
      for (let i = 0; i < 4; i++) {
        if (p[i] === 0) { if (q[i] < 0) return null; continue; }
        const t = q[i] / p[i];
        if (p[i] < 0) t0 = Math.max(t0, t); else t1 = Math.min(t1, t);
      }
      if (t1 <= t0) return null;
      return [{ x: a.x + dx * t0, y: a.y + dy * t0 }, { x: a.x + dx * t1, y: a.y + dy * t1 }];
    }

    function pick(pos) {
      const hits = [['A', A], ['B', B]];
      let found = null, bd = 22 * 22;
      for (const [k, pt] of hits) {
        const dx = sx(pt.x) - pos.x, dy = sy(pt.y) - pos.y, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; found = k; }
      }
      return found;
    }
    cv.addEventListener('pointerdown', (e) => { e.preventDefault(); drag = pick(cv.pos(e)); showBest = false; });
    cv.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const p = cv.pos(e);
      const t = { x: ctx.clamp(ux(p.x), HLO, HHI), y: ctx.clamp(uy(p.y), HLO, HHI) };
      if (drag === 'A') A = t; else B = t;
    });
    ['pointerup', 'pointercancel', 'pointerleave'].forEach(t => cv.addEventListener(t, () => { drag = null; }));

    const sel = ctx.h('select', {}, Object.keys(DATASETS).map(k => ctx.h('option', { value: k }, DATASETS[k].label)));
    sel.value = 'spam';
    sel.addEventListener('change', () => {
      key = sel.value; pts = DATASETS[key].build(); cachedBest = null; showBest = false;
      A = { x: 0.12, y: 0.9 }; B = { x: 0.9, y: 0.12 }; flip = false;
    });
    const selWrap = ctx.h('div', { class: 'control' }, ctx.h('label', {}, 'the decision'), sel);
    const flipBtn = ctx.button('Swap which side is which', () => { flip = !flip; showBest = false; });
    const bestBtn = ctx.button('Show the best line there is', () => {
      const bl = getBest();
      const seg = trimToPlot(bl.a, bl.b);
      if (seg) { A = seg[0]; B = seg[1]; } else { A = { x: HLO, y: HHI }; B = { x: HHI, y: HLO }; }
      flip = bl.fl; showBest = true;
    }, 'primary');
    const ro = ctx.readout();

    ctx.loop(() => {
      const D = DATASETS[key];
      g.clearRect(0, 0, cv.W, cv.H);
      const st = stats();

      /* shaded verdict regions */
      const STEP = 9;
      for (let px = PAD.l; px < PAD.l + PW(); px += STEP) {
        for (let py = PAD.t; py < PAD.t + PH(); py += STEP) {
          const said = side(ux(px + STEP / 2), uy(py + STEP / 2));
          g.fillStyle = said ? 'rgba(251,113,133,0.13)' : 'rgba(124,156,255,0.10)';
          g.fillRect(px, py, STEP, STEP);
        }
      }
      g.strokeStyle = C.line; g.lineWidth = 1;
      g.strokeRect(PAD.l, PAD.t, PW(), PH());

      /* axes */
      g.font = MONO; g.fillStyle = C.muted;
      g.fillText(D.xName + '  →', PAD.l, PAD.t + PH() + 34);
      g.save(); g.translate(PAD.l - 40, PAD.t + PH()); g.rotate(-Math.PI / 2);
      g.fillText(D.yName + '  →', 0, 0); g.restore();
      g.fillText('0', sx(0) - 4, PAD.t + PH() + 16);
      g.fillText(String(D.xMax), sx(1) - 14, PAD.t + PH() + 16);

      /* Everything that depends on the data or on where the reader has dragged
         is clipped to the plot box, so the boundary line cannot run off across
         the axis labels and the panel of numbers on the right. */
      g.save();
      g.beginPath(); g.rect(PAD.l, PAD.t, PW(), PH()); g.clip();

      /* the line */
      const dx = B.x - A.x, dy = B.y - A.y, n = Math.hypot(dx, dy) || 1;
      const ex = dx / n * 3, ey = dy / n * 3;
      g.strokeStyle = showBest ? C.green : C.text; g.lineWidth = 2.5;
      g.beginPath();
      g.moveTo(sx(A.x - ex), sy(A.y - ey));
      g.lineTo(sx(B.x + ex), sy(B.y + ey));
      g.stroke();
      [A, B].forEach(pt => {
        g.beginPath(); g.arc(sx(pt.x), sy(pt.y), 8, 0, 7);
        g.fillStyle = showBest ? C.green : '#fff'; g.fill();
        g.strokeStyle = '#0a0e16'; g.lineWidth = 2; g.stroke();
      });

      /* the data */
      pts.forEach(p => {
        const nn = norm(p), said = side(nn.u, nn.v), right = (said ? 1 : 0) === p.lab;
        g.beginPath(); g.arc(sx(nn.u), sy(nn.v), 4.2, 0, 7);
        g.fillStyle = p.lab === 1 ? C.danger : C.accent;
        g.globalAlpha = right ? 0.95 : 1; g.fill(); g.globalAlpha = 1;
        if (!right) { g.strokeStyle = C.warn; g.lineWidth = 2; g.stroke(); }
      });
      g.restore();

      /* --------- right-hand panel --------- */
      const TX = 720 - PAD.r + 22;
      g.font = 'bold 15px Inter, system-ui, sans-serif'; g.fillStyle = C.text;
      g.fillText(D.label, TX, 34);

      g.font = MONO; g.fillStyle = C.danger; g.fillText('● ' + D.pos, TX, 58);
      g.fillStyle = C.accent; g.fillText('● ' + D.neg, TX + 120, 58);
      g.fillStyle = C.warn; g.fillText('○ ringed = the line got it wrong', TX, 76);

      /* accuracy */
      g.font = FONT; g.fillStyle = C.muted; g.fillText('you are getting right', TX, 108);
      g.font = 'bold 30px Inter, system-ui, sans-serif';
      g.fillStyle = st.acc > 0.9 ? C.green : st.acc > 0.75 ? C.warn : C.danger;
      g.fillText((st.acc * 100).toFixed(1) + '%', TX, 142);
      g.font = MONO; g.fillStyle = C.muted;
      g.fillText((st.tp + st.tn) + ' of ' + pts.length + ' decisions', TX, 162);

      /* the two mistakes, which are not the same mistake */
      g.font = FONT; g.fillStyle = C.text; g.fillText('and the mistakes it makes', TX, 194);
      g.font = MONO;
      g.fillStyle = C.danger;
      g.fillText(st.fn + '  ' + D.pos + ' let through', TX, 216);
      g.fillStyle = C.warn;
      g.fillText(st.fp + '  ' + D.neg + ' wrongly flagged', TX, 234);

      /* live weights — this is what the maths beat below points at */
      const w1 = -(B.y - A.y), w2 = (B.x - A.x);
      const b0 = -(w1 * A.x + w2 * A.y);
      const sc = 1 / (Math.hypot(w1, w2) || 1);
      const fw = flip ? -1 : 1;
      g.font = 'bold ' + FONT; g.fillStyle = C.purple;
      g.fillText('the line, as three numbers', TX, 270);
      g.font = MONO; g.fillStyle = C.muted;
      /* the parentheticals name the axis, so they use each dataset's own short
         name — chopping the first word off yName turned "how irregular the edge
         is" into a label that just read "(how)" */
      g.fillText('w₁ = ' + (w1 * sc * fw).toFixed(2) + '   (' + D.xShort + ')', TX, 292);
      g.fillText('w₂ = ' + (w2 * sc * fw).toFixed(2) + '   (' + D.yShort + ')', TX, 310);
      g.fillText('b  = ' + (b0 * sc * fw).toFixed(2) + '   (suspicion by default)', TX, 328);

      const bestOk = getBest().ok / pts.length;
      g.font = FONT;
      g.fillStyle = st.acc >= bestOk - 0.001 ? C.green : C.muted;
      g.fillText('the best any straight line can do: ' + (bestOk * 100).toFixed(1) + '%', TX, 358);
      g.font = '12px Inter, system-ui, sans-serif'; g.fillStyle = C.muted;
      wrapText(g, D.note, TX, 380, 270, 15);

      ro.set({
        'getting right': (st.acc * 100).toFixed(1) + '%',
        [D.pos + ' missed']: st.fn,
        'false alarms': st.fp,
        'ceiling for one line': (bestOk * 100).toFixed(1) + '%',
      });
    });

    return ctx.figure(cv,
      'The same two handles, the same one line — but now every dot is a real decision with a cost attached. Two things change. There is no 4 out of 4 any more: real data overlaps, so <b>even the best possible line still gets some wrong</b>, and the demo brute-forces every line there is to tell you exactly where that ceiling sits. And the two kinds of mistake stop being interchangeable — letting fraud through and freezing an honest customer\'s card are very different failures, and the line you choose decides the balance between them.',
      [selWrap, flipBtn, bestBtn], ro);
  }


  /* ------------------------------------------------------------------ */
  /* The third neuron's line — which is real, but lives in a different   */
  /* space: the space of WHAT THE FIRST TWO SAID. Watching the four dots */
  /* move into that space is the whole reason depth works, and the       */
  /* chapter previously asserted this neuron without ever drawing it.    */
  /* ------------------------------------------------------------------ */
  function buildThirdNeuron(ctx) {
    const [cv, g] = ctx.canvas(720, 400);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    /* XOR: red when exactly one switch is on */
    const PTS = [
      { x: 0, y: 0, lab: 0 }, { x: 0, y: 1, lab: 1 },
      { x: 1, y: 0, lab: 1 }, { x: 1, y: 1, lab: 0 },
    ];
    /* neuron A fires when at least one switch is on; neuron B when both are */
    const A = (p) => (p.x + p.y > 0.5 ? 1 : 0);
    const B = (p) => (p.x + p.y > 1.5 ? 1 : 0);
    let t = 0, playing = false;

    const tSl = ctx.slider({ label: 'move into the new space', min: 0, max: 1, step: 0.01, value: 0, digits: 2, onChange: (v) => { t = v; stop(); } });
    const playBtn = ctx.button('▶ Watch them move', () => {
      playing = !playing;
      playBtn.textContent = playing ? '⏸ Pause' : '▶ Watch them move';
    }, 'primary');
    function stop() { playing = false; playBtn.textContent = '▶ Watch them move'; }
    const backBtn = ctx.button('Back to the start', () => { t = 0; tSl.value = 0; stop(); });
    const ro = ctx.readout();

    const ease = (u) => u * u * (3 - 2 * u);

    /* Plot geometry. Everything below stays inside PLOT, and the drawing is
       clipped to it as well, so no future edit here can spill into the text
       panel on the right. */
    const X0 = 56, Y0 = 58, SZ = 232;
    const TX = 348, TW = 348;
    /* the four dots sit at 0 and 1 exactly, so the data range is inset by a dot
       radius: without it the clip above would slice every corner dot in half */
    const PAD = 18, IN = SZ - PAD * 2;
    const sx = (v) => X0 + PAD + v * IN;
    const sy = (v) => Y0 + SZ - PAD - v * IN;

    ctx.loop((dt) => {
      if (playing) { t = Math.min(1, t + dt * 0.45); tSl.value = +t.toFixed(2); if (t >= 1) stop(); }
      g.clearRect(0, 0, cv.W, cv.H);
      const u = ease(t);
      const inOld = u < 0.5;

      /* ---- frame and axes ---- */
      g.strokeStyle = C.line; g.lineWidth = 1;
      g.strokeRect(X0, Y0, SZ, SZ);
      g.font = 'bold ' + FONT; g.textAlign = 'left'; g.textBaseline = 'alphabetic';
      g.fillStyle = inOld ? C.text : C.green;
      g.fillText(inOld ? 'the original square' : 'the space A and B built', X0, 34);

      g.font = MONO; g.fillStyle = C.muted;
      /* tick numbers sit just outside the box; the axis names sit a line below
         them, so the two can never collide however long the names get */
      g.textAlign = 'center';
      g.fillText('0', sx(0), Y0 + SZ + 16); g.fillText('1', sx(1), Y0 + SZ + 16);
      g.fillText(inOld ? 'switch 1' : 'what A said', X0 + SZ / 2, Y0 + SZ + 34);
      g.textAlign = 'right';
      g.fillText('0', X0 - 10, sy(0) + 4); g.fillText('1', X0 - 10, sy(1) + 4);
      g.save(); g.translate(X0 - 26, Y0 + SZ / 2); g.rotate(-Math.PI / 2);
      g.textAlign = 'center'; g.fillText(inOld ? 'switch 2' : 'what B said', 0, 0); g.restore();
      g.textAlign = 'left';

      /* ---- everything from here on is trapped inside the plot box ---- */
      g.save();
      g.beginPath(); g.rect(X0, Y0, SZ, SZ); g.clip();

      /* the two hidden neurons' lines, fading out as we leave their space.
         Each is drawn corner to corner WITHIN the unit square: A is the line
         x + y = 0.5, B is x + y = 1.5. */
      if (u < 0.78) {
        const fade = 1 - u / 0.78;
        [[0.5, C.warn, 'A', 0.30], [1.5, C.purple, 'B', 0.62]].forEach(([c, col, name, lx]) => {
          g.globalAlpha = fade * 0.9;
          g.strokeStyle = col; g.lineWidth = 2;
          g.beginPath();
          g.moveTo(sx(Math.max(0, c - 1)), sy(Math.min(1, c)));
          g.lineTo(sx(Math.min(1, c)), sy(Math.max(0, c - 1)));
          g.stroke();
          g.font = 'bold ' + MONO; g.fillStyle = col;
          g.fillText('neuron ' + name, sx(lx), sy(c - lx) - 8);
          g.globalAlpha = 1;
        });
      }

      /* the third neuron's line, fading in, in the new space: it separates
         "A said yes but B said no" from everything else, which is the line
         A − B = 0.5, from (0.5, 0) to (1, 0.5). */
      if (u > 0.55) {
        const fade = (u - 0.55) / 0.45;
        g.globalAlpha = fade;
        g.strokeStyle = C.green; g.lineWidth = 3;
        g.beginPath();
        g.moveTo(sx(0.5), sy(0));
        g.lineTo(sx(1), sy(0.5));
        g.stroke();
        g.font = 'bold ' + MONO; g.fillStyle = C.green;
        g.fillText('the third neuron', sx(0.04), sy(0.80));
        g.fillText('draws THIS line', sx(0.04), sy(0.80) + 16);
        g.globalAlpha = 1;
      }

      /* the four dots, travelling between the two spaces */
      const seen = {};
      PTS.forEach((p) => {
        const px = p.x + (A(p) - p.x) * u;
        const py = p.y + (B(p) - p.y) * u;
        const key = px.toFixed(2) + ',' + py.toFixed(2);
        const off = seen[key] ? 9 : 0;
        seen[key] = 1;
        g.beginPath(); g.arc(sx(px) + off, sy(py) - off, 10, 0, 7);
        g.fillStyle = p.lab ? C.danger : C.accent;
        g.fill();
        g.strokeStyle = '#0a0e16'; g.lineWidth = 2; g.stroke();
      });
      g.restore();

      /* ---- the running commentary, in its own column ---- */
      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('where is the third line?', TX, 34);
      g.font = FONT;
      let msg, colour = C.muted;
      if (u < 0.1) {
        msg = 'You are looking at the original square. Neurons A and B have drawn their two lines. No third line is visible — and that is exactly the complaint, because a third line drawn HERE could not help. Any straight line in this square fails.';
      } else if (u < 0.78) {
        msg = 'Watch the dots. They are not being shuffled for show: they are taking new coordinates. How far right is now "what A said". How far up is now "what B said". The two hidden neurons are building a new set of axes.';
      } else {
        msg = 'And there it is. One pair of dots has landed on the SAME SPOT, because A and B answered identically for both. That fold is what makes the problem easy: in this new space a single straight line separates red from blue, and the third neuron draws it.';
        colour = C.green;
      }
      g.fillStyle = colour;
      wrapText(g, msg, TX, 58, TW, 18);

      /* the truth table, filling in as we move */
      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('what each neuron says', TX, 232);
      g.font = MONO; g.fillStyle = C.muted;
      const COL = [TX, TX + 86, TX + 122, TX + 170, TX + 270];
      ['switches', 'A', 'B', 'third says', 'want'].forEach((s, i) => g.fillText(s, COL[i], 256));
      g.strokeStyle = C.line; g.lineWidth = 1;
      g.beginPath(); g.moveTo(TX, 264); g.lineTo(TX + 316, 264); g.stroke();
      let yy = 284;
      PTS.forEach((p) => {
        const a = A(p), b = B(p), out = (a - b > 0.5) ? 1 : 0;
        g.font = MONO;
        g.fillStyle = p.lab ? C.danger : C.accent; g.fillText(p.x + ',' + p.y, COL[0], yy);
        g.fillStyle = u > 0.2 ? C.warn : '#2a3444'; g.fillText(String(a), COL[1], yy);
        g.fillStyle = u > 0.2 ? C.purple : '#2a3444'; g.fillText(String(b), COL[2], yy);
        g.fillStyle = u > 0.8 ? C.green : '#2a3444'; g.fillText(u > 0.8 ? String(out) : '·', COL[3], yy);
        g.fillStyle = C.muted; g.fillText(String(p.lab), COL[4], yy);
        yy += 22;
      });
      if (u > 0.8) {
        g.font = 'bold ' + FONT; g.fillStyle = C.green;
        g.fillText('4 / 4 — with one straight line.', TX, yy + 16);
      }

      ro.set({
        'looking at': inOld ? 'the original square' : 'the space A and B built',
        'distinct positions': String(new Set(PTS.map(p => (p.x + (A(p) - p.x) * u).toFixed(2) + ',' + (p.y + (B(p) - p.y) * u).toFixed(2))).size),
      });
    });

    return ctx.figure(cv,
      'Your question was where the third line is, and the answer is that it exists but not in this picture — it lives in a <b>different space</b>. Slide across and watch the four dots take new coordinates: how far right becomes "what neuron A said", how far up becomes "what neuron B said". Two dots land on top of each other, because A and B could not tell them apart. That fold is the entire point of a hidden layer: <b>it does not solve the problem, it rearranges the problem until one straight line can.</b> Every deep network is this move, repeated.',
      [tSl, playBtn, backBtn], ro);
  }

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
    /* Axis hints live OUTSIDE the square, in the margin. Inside it they sat in
       the shaded region where the boundary line sweeps across them. */
    g.fillStyle = C.muted; g.font = '11px Inter, system-ui, sans-serif';
    g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText('switch A off', b.sx(0), b.y0 + b.size + 14);
    g.fillText('switch A on', b.sx(1), b.y0 + b.size + 14);
    g.save();
    g.translate(b.x0 - 22, b.sy(0.5)); g.rotate(-Math.PI / 2);
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

  /* The decision boundary is an infinite line, so it is drawn very long and then
     CLIPPED TO THE BOARD. Without the clip it runs clear across the canvas and
     slashes through the score, the legend, the axis labels and the mistake bars
     that live in the margins around the square. */
  function drawLineThrough(g, b, ax, ay, bx, by, color, width) {
    let dx = bx - ax, dy = by - ay;
    const len = Math.hypot(dx, dy) || 1;
    dx /= len; dy /= len;
    const far = 4;
    g.save();
    g.beginPath(); g.rect(b.x0, b.y0, b.size, b.size); g.clip();
    g.strokeStyle = color; g.lineWidth = width || 3;
    g.beginPath();
    g.moveTo(b.sx(ax - dx * far), b.sy(ay - dy * far));
    g.lineTo(b.sx(ax + dx * far), b.sy(ay + dy * far));
    g.stroke();
    g.restore();
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
    /* a wider margin than the other boards: the puzzle blurb is a full sentence
       and needs clear air above the square, or the line grazes its descenders */
    const b = makeBoard(cv, g, C, 62);
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
      /* handles stay inside the square — every line worth drawing crosses it,
         and a handle parked out in the margin would sit on the axis labels */
      const t = { x: Math.max(-0.3, Math.min(1.3, b.ix(p.x))), y: Math.max(-0.3, Math.min(1.3, b.iy(p.y))) };
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
      g.fillText(PUZZLES[puzzle].blurb, 16, 44);

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
    let lastErr = 0, lastDw1 = 0, lastDw2 = 0, lastDb = 0;   // the most recent correction, for the readout

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
      lastErr = err;
      lastDw1 = LR * err * p[0]; lastDw2 = LR * err * p[1]; lastDb = LR * err;
      if (err !== 0) {
        mistakes++;
        w1 += lastDw1;
        w2 += lastDw2;
        bias += lastDb;
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

      // mistake history bars, in the strip below the axis labels
      const bx0 = 16, by0 = b.H - 32;
      g.fillStyle = C.muted; g.font = '10px Inter, system-ui, sans-serif';
      g.fillText('mistakes per pass', bx0, by0 - 16);
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

      readout.set({
        'correct': score + ' of 4', 'passes': pass,
        'mistakes last pass': lastMistakes == null ? '—' : lastMistakes,
        'w₁': w1.toFixed(2), 'w₂': w2.toFixed(2), 'b': bias.toFixed(2),
        'last error (y − ŷ)': lastErr === 0 ? '0 — no change' : (lastErr > 0 ? '+1' : '−1'),
        'it just moved w₁ by': lastErr === 0 ? 'nothing' : lastDw1.toFixed(3),
      });
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
      onChange: (v) => { puzzle = v; reset(); running = false; playBtn.textContent = 'Train'; },
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

    /* how far the nearest dot sits from a line. lineFor gives a unit direction,
       so the cross product is already a perpendicular distance. */
    function marginOf(L) {
      let m = Infinity;
      for (const p of PTS) m = Math.min(m, Math.abs((L.bx - L.ax) * (p[1] - L.ay) - (L.by - L.ay) * (p[0] - L.ax)));
      return m;
    }

    function testOne() {
      const labels = PUZZLES[puzzle].labels;
      const L = lineFor(angleI, offI);
      for (const flip of [false, true]) {
        tried++;
        const s = scoreOf(labels, (x, y) => classify(x, y, L.ax, L.ay, L.bx, L.by, flip));
        if (s === 4) {
          worked++;
          /* show the winner with the most daylight around it, not whichever one
             the sweep reached first — the first is typically a line that grazes
             two dots, which reads as a mistake next to "plenty of lines work" */
          const m = marginOf(L);
          if (!bestShown || m > bestShown.m) bestShown = { L: L, flip: flip, m: m };
        }
      }
      if (trail.length < 260 && angleI % 2 === 0) trail.push(L);
      offI++;
      if (offI >= N_OFF) { offI = 0; angleI++; }
      if (angleI >= N_ANG) { running = false; done = true; }
    }

    const readout = ctx.readout();
    ctx.loop(() => {
      if (running) for (let i = 0; i < 40; i++) { if (!running) break; testOne(); }
      /* the sweep stops itself at the last line, so the button has to stop
         saying "Pause" without being pressed */
      if (done && runBtn.textContent === 'Pause') runBtn.textContent = 'Try every line';

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
      'Every orientation and position, both ways round. On puzzle 1 the winner counter climbs into the hundreds. On puzzle 3 it stays on zero from the first line to the last.',
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
    const [cv, g] = ctx.canvas(700, 470);
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
                 + '  =  ' + sum.toFixed(2) + '   ' + (sum > 0 ? '> 0, so fire' : '≤ 0, so stay quiet'), 24, 296);
      g.textAlign = 'center';

      /* ---- the same three numbers, drawn as the line they describe ---- */
      const PX = 40, PY = 330, PS = 124, PAD = 14, IN = PS - PAD * 2;
      const sx = (v) => PX + PAD + v * IN;
      const sy = (v) => PY + PS - PAD - v * IN;
      g.textAlign = 'left'; g.textBaseline = 'alphabetic';
      g.font = '600 13px Inter, system-ui, sans-serif'; g.fillStyle = C.text;
      g.fillText('the same three numbers, as a line', PX, PY - 14);
      g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(PX, PY, PS, PS);

      g.save();
      g.beginPath(); g.rect(PX, PY, PS, PS); g.clip();
      /* w·x + b = 0. Draw it long and let the clip trim it to the box. */
      const nn = Math.hypot(wa, wb);
      if (nn > 1e-6) {
        const cx0 = -bias * wa / (nn * nn), cy0 = -bias * wb / (nn * nn);
        const dx = -wb / nn, dy = wa / nn;
        /* shade the side that fires, so the bias reads as "which side wins" */
        g.fillStyle = 'rgba(56,217,169,0.10)';
        g.beginPath();
        g.moveTo(sx(cx0 - dx * 6), sy(cy0 - dy * 6));
        g.lineTo(sx(cx0 + dx * 6), sy(cy0 + dy * 6));
        g.lineTo(sx(cx0 + dx * 6 + wa * 6), sy(cy0 + dy * 6 + wb * 6));
        g.lineTo(sx(cx0 - dx * 6 + wa * 6), sy(cy0 - dy * 6 + wb * 6));
        g.closePath(); g.fill();
        g.strokeStyle = C.text; g.lineWidth = 2;
        g.beginPath();
        g.moveTo(sx(cx0 - dx * 6), sy(cy0 - dy * 6));
        g.lineTo(sx(cx0 + dx * 6), sy(cy0 + dy * 6));
        g.stroke();
      } else {
        g.font = '11px Inter, system-ui, sans-serif'; g.fillStyle = C.muted; g.textAlign = 'center';
        g.fillText('both weights are zero —', PX + PS / 2, PY + PS / 2 - 6);
        g.fillText('there is no line left', PX + PS / 2, PY + PS / 2 + 10);
        g.textAlign = 'left';
      }
      /* the input you have dialled in, on whichever side it lands */
      g.beginPath(); g.arc(sx(a), sy(bb), 7, 0, Math.PI * 2);
      g.fillStyle = out ? C.green : C.danger; g.fill();
      g.strokeStyle = '#0a0e16'; g.lineWidth = 2; g.stroke();
      g.restore();

      g.font = '11px JetBrains Mono, monospace'; g.fillStyle = C.muted; g.textAlign = 'center';
      g.fillText('A', sx(0.5), PY + PS + 15);
      g.save(); g.translate(PX - 9, PY + PS / 2); g.rotate(-Math.PI / 2);
      g.fillText('B', 0, 0); g.restore();

      g.textAlign = 'left';
      const TX2 = PX + PS + 26;
      g.font = '13px Inter, system-ui, sans-serif'; g.fillStyle = C.muted;
      /* canvas text is drawn literally — no markup here */
      wrapText(g, 'This is the line you dragged by hand earlier, except now you set it with the numbers instead. '
        + 'The two weights ROTATE it. The bias SLIDES it without turning it. The dot is the input you have dialled in, '
        + 'and it is green exactly when it sits on the shaded side.', TX2, PY + 16, 700 - TX2 - 24, 18);
      g.font = '600 13px Inter, system-ui, sans-serif';
      g.fillStyle = Math.abs(bias) < 1e-9 ? C.warn : C.muted;
      g.fillText(Math.abs(bias) < 1e-9
        ? 'bias = 0: the line is pinned through the bottom-left corner.'
        : 'Set bias to exactly 0 and watch the line jump to the corner.', TX2, PY + PS - 6);

      readout.set({ 'total': sum.toFixed(2), 'output': out });
    });

    return ctx.figure(cv,
      'Thicker wire means a bigger weight; red pushes toward firing and blue pushes against it. The bias is how much evidence the neuron demands before it fires at all. The panel underneath draws the <b>same three numbers as a line</b>: move the weights and it rotates, move the bias and it slides. Three numbers, and that is the whole neuron.',
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
