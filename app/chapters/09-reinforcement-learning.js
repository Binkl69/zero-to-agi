/* Zero → AGI · Chapter 09 · Learning by doing: reinforcement learning
   DESIGN RULE: the reader IS the agent for the first thirty seconds — no map, no labels, one
   number per move — so credit assignment is felt before it is named.
   Interactives, in order: blind-agent grid (be the agent); discount factor deciding whether an
   agent walks past a prize; tabular Q-learning gridworld with the value ripple; multi-armed
   bandit with regret curves; PPO clipping vs policy collapse; CoastRunners reward hacking. */
(function () {
  ZTA.registerChapter({
    id: '09-reinforcement-learning',
    num: 9,
    part: 'II',
    title: 'Learning by doing: reinforcement learning',
    tagline: 'Find the goal with no map and one number per move, then meet the algorithms that solve the problem you just felt — and the reward that lies to you.',
    render(root, ctx) {
      const { h, p, section, sub, callout, ul } = ctx;
      const C = ctx.colors;
      const FONT = '13px Inter, system-ui, sans-serif';
      const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
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

      /* ---------- open by making the reader the agent ---------- */
      root.append(
        callout('tryit', '🖐 Do this first — find the goal with no map and no instructions',
          `You are somewhere on a grid. You cannot see it. There is a good square somewhere and at least one very bad one.<br>
           <b>1.</b> Press the arrows and move. After each move you get <b>one number</b>. That is the only thing you will ever be told.<br>
           <b>2.</b> Keep going until something decisive happens.<br>
           <b>3.</b> When it does, ask yourself the question that defines this whole chapter: <b>which move was the mistake?</b><br>
           <b>4.</b> Then press <b>Reveal the map</b> and see what you were up against.`),
        blindAgent(),
        p(`Nobody told you the right move. Nobody told you the wrong one either — only how it went, once, at the end. Every algorithm in this chapter exists to close that gap.`),
      );

      root.append(section('Why this is a different kind of problem',
        p(`In March 2016, in a hotel in Seoul, a program called AlphaGo played move 37 of its second game against Lee Sedol. Commentators thought it was a mistake. Professionals estimated a human would play it about once in ten thousand games. It won.`),
        p(`Nobody had shown AlphaGo that move. There was no labelled dataset of brilliant moves, no teacher marking its homework. It found the move by <b>playing millions of games against itself and noticing what led to winning</b>.`),
        p(`Everything in this course so far was <em>supervised</em>: show the model an input and the correct output, measure the error, nudge the weights. But most of life has no correct output. When you learn to ride a bike, nobody hands you the right handlebar angle for each millisecond. You wobble, you fall, you adjust.`),
        h('div', { class: 'grid-2' },
          h('div', { class: 'card' }, h('h4', {}, 'Supervised learning'), h('p', { html: 'Input → correct output. The loss tells you <b>how wrong</b> and <b>in which direction</b>, for every example. Feedback is instant and precise.' })),
          h('div', { class: 'card' }, h('h4', {}, 'Reinforcement learning'), h('p', { html: 'State → action → reward, repeat. The reward tells you <b>how it went</b>, often much later, and never tells you what you should have done instead.' })),
        ),
        p(`This chapter is about learning from consequences. It matters for its own sake — games, robots, data-centre cooling — and it matters because it is the final ingredient that turns a text predictor into an assistant. Everything here comes back in chapter 11.`),
      ));

      root.append(section('Five words and a loop',
        p(`You just played all five parts. The <em>agent</em> is the learner — that was you. The <em>environment</em> is everything else: the grid, the game, the robot's physics.`),
        p(`At each tick the environment shows the agent a <em>state</em> (where am I?). The agent picks an <em>action</em>. The environment responds with a new state and a number: the <em>reward</em>. Then it repeats.`),
        p(`The reward is the entire training signal. It is often zero for a long time and then suddenly not — +1 for winning after 200 moves, −1 for losing, nothing in between. The agent's job is to maximise <b>total future reward</b>, not the next one.`),
        p(`Two more words. A <em>policy</em> is the strategy: a rule mapping a state to an action. A <em>value</em> is a prediction — "starting here, how much reward will I collect from now on?" A chess player who says "white is winning" is stating a value.`),
        p(`Learning good values is most of what RL does, because once you know the value of every state you can simply step toward the most valuable one.`),
        callout('key', '🔑 The credit-assignment problem',
          `You lose a chess game at move 40. The reward is −1. <b>Which move was the mistake?</b> Maybe move 12. Maybe the loss was baked in at move 3 and the next 37 moves were fine.<br>
           A single number arrives at the end and has to be spread back over every decision that led to it. You felt this directly in the demo above.<br>
           This is why RL needs more than gradient descent on a loss: <b>there is no per-step loss to descend.</b>`),
      ));

      root.append(section('How far ahead should it look?',
        p(`A reward of 10 now is worth more than a reward of 10 in fifty steps — partly because the world is uncertain, partly because we want quick wins. So RL multiplies future rewards by a <em>discount factor</em> γ per step.`),
        callout('tryit', '🖐 Try this — make an agent walk past a prize',
          `Two rewards down one corridor: a <b>+3</b> two steps away and a <b>+20</b> twelve steps away.<br>
           <b>1.</b> Press <b>impatient (0.50)</b>. The distant +20 is now worth about 0.005 — the agent grabs the +3 and stops.<br>
           <b>2.</b> Press <b>far-sighted (0.99)</b>. The +20 is worth around 17.7 and easily wins.<br>
           <b>3.</b> Drag γ slowly and find the exact value where the agent changes its mind. <b>Nothing about the world changed. Only its patience did.</b>`),
        discountLab(),
        p(`With γ = 0.99 an agent effectively plans about a hundred steps ahead; with γ = 0.5 it barely sees past the next few moves. It is one number, and it silently determines whether your agent is capable of a long-term plan at all.`),
        callout('key', '🔑 Explore or exploit?',
          `You have a favourite restaurant. It is good. There are twenty others you have never tried.<br>
           Every night: <em>exploit</em> what you know (a reliable 8/10) or <em>explore</em> (a 4, or a 10)? Never explore and you never find the best. Always explore and you eat a lot of bad dinners.<br>
           An RL agent faces this every single step. The crudest fix works surprisingly well: with probability <em>ε</em> — say 10% — ignore what you know and act at random. That is <em>ε-greedy</em>.`),
      ));

      root.append(section('Q-learning: the simplest thing that actually works',
        p(`From 1989. Keep a table, <em>Q</em>, with one row per state and one column per action. Q(s, a) is your current estimate of the total discounted reward from taking action a in state s and behaving sensibly afterwards. Start with all zeros.`),
        p(`Then every time you observe (state s, action a, reward r, new state s'), update one cell. Read the bracket as an <b>error</b>: what you just learned the value should be, minus what you previously believed. You move your belief part of the way toward the evidence.`),
        p(`That is the whole algorithm. Because the target uses the <i>next</i> state's value, information flows backwards one step per update: the cell next to the goal learns first, then the cell next to that. Credit assignment gets solved not all at once, but by a chain of one-step corrections repeated thousands of times.`),
        callout('tryit', '🖐 Try this — watch an agent learn the maze you just failed at',
          `<b>1.</b> Press <b>▶ Play</b> and watch the colours. The squares nearest the goal go green first, then the ripple spreads backwards. <b>That is credit assignment happening in front of you.</b><br>
           <b>2.</b> Drag <b>γ discount</b> down to 0.5 and press <b>Reset Q</b>. The ripple stops spreading — distant squares never learn the goal exists.<br>
           <b>3.</b> Push <b>ε exploration</b> to 1.0, so the agent moves <i>entirely at random</i>. The table still converges on a good policy. That is not a bug; it is the property called off-policy learning.<br>
           <b>4.</b> Turn <b>Edit map</b> on and move the goal. Watch it unlearn the old one.`),
        gridworld(),
        p(`Two details earn their place. When an episode ends there is no "next state", so the target is just the reward — that is the <b>only</b> place real reward enters the table, and every other value in the grid is ultimately a rumour about it.`),
        p(`And the update uses the <i>best</i> next value regardless of what the agent actually did next. So Q-learning learns the value of the best policy even while behaving randomly, which is exactly why step 3 above works. That property is called <em>off-policy</em> learning.`),
        p(`Q-learning is <em>tabular</em> when the table is literal, which only works for small worlds. For Atari or Go the table becomes a neural network taking the state and outputting a Q value per action, and the update rule becomes the loss it is trained on. That is <em>deep Q-learning</em>.`),
      ));

      root.append(section('The smallest possible version of the dilemma',
        p(`Strip RL down: no states, no delayed reward, just five slot machines with different hidden payout rates. Each pull pays £1 with some probability. You get 1,000 pulls. How do you find the best machine without wasting too many on the bad ones?`),
        p(`This is the <em>multi-armed bandit</em> — the exploration–exploitation dilemma with everything else removed. Casinos gave it the name; websites choosing which headline to show you are the modern use.`),
        callout('tryit', '🖐 Try this: beat the algorithms',
          `<b>1.</b> Pull the machines yourself for a while. Try to work out which is best — and notice how hard it is to tell a 0.5 machine from a 0.6 machine in twenty pulls.<br>
           <b>2.</b> Press <b>Run agents</b> and watch the regret curves. <b>Watch the shape, not the final number.</b> A good agent's curve <b>bends flat</b>; a bad one keeps climbing in a straight line.<br>
           <b>3.</b> Press <b>Reveal true rates</b> and see how close you got.<br>
           <b>4.</b> Raise the horizon to 20,000 and run again — the two strategies change places.`),
        bandit(),
        p(`The score that matters is <em>regret</em>: what you earned compared with pulling the best machine every time. A perfect agent has zero regret. A random agent's regret grows in a straight line. A good agent's curve bends flat — it explores early, identifies the winner, and stops paying for information it no longer needs.`),
        p(`<em>UCB</em> (upper confidence bound) scores each machine by its average payout <b>plus a bonus that grows the less you have tried it</b>. Barely-touched machines look optimistic, so you try them; as evidence accumulates the bonus shrinks and the winner takes over. "Optimism in the face of uncertainty" — a good slogan for life, too.`),
        p(`Notice the asymmetry that makes it work. The bonus falls as 1/√n, so a machine pulled ten times still carries a visible bonus while one pulled a thousand times carries almost none. Meanwhile ln(t) creeps upward forever, so a long-ignored machine slowly becomes tempting again — an agent that tapers its own exploration, with no ε knob for a human to guess.`),
      ));

      root.append(section('The other family: learn the policy directly',
        p(`Q-learning learns values and derives a policy from them. The other big family skips the middleman and learns the <em>policy</em> itself: a network that outputs a probability for each action.`),
        p(`The training rule, <em>REINFORCE</em> or the <em>policy gradient</em>, is one sentence: <b>run the policy, and for every action you took, nudge its probability up in proportion to the reward that followed (and down if the reward was bad)</b>. Over millions of games the noise averages out and only genuinely good habits survive.`),
        p(`Two refinements make it trainable at scale. First, a <em>baseline</em>: a second network predicts how well the agent expected to do, and the policy is nudged by the <b>surprise</b> rather than the raw return. If every game scores between 90 and 100, raw returns say "everything you did was great"; the surprise says which games were the 97s.`),
        p(`That difference is the <em>advantage</em>, and an algorithm with both a policy and a value network is called <em>actor–critic</em>.`),
        p(`Second, a limit on how far one update may move the policy — and this one is worth feeling rather than reading.`),
        callout('tryit', '🖐 Try this — break a training run on purpose',
          `<b>1.</b> With <b>clipping: on</b>, drag the step size right up to 2.5. The curve wobbles but keeps climbing.<br>
           <b>2.</b> Now turn <b>clipping: off</b> and do it again. Past about 1.5 the run does not degrade gracefully — it <b>falls off a cliff</b>, repeatedly. Each red dot is a collapse.<br>
           <b>3.</b> Press <b>New run</b> a few times to confirm it is not one unlucky seed.`),
        ppoClip(),
        p(`Here is why RL breaks in a way supervised learning never does: <b>the model chooses its own next training set</b>. Take too big a step and the policy that generated your data no longer resembles the policy you now have, so the data becomes worthless and performance collapses.`),
        p(`<em>TRPO</em> (2015) enforced a hard constraint on how far the policy could move. <em>PPO</em> (2017) got almost the same effect far more cheaply by clipping the update whenever action probabilities move more than about 20%.`),
        p(`PPO is the algorithm later used to train ChatGPT, so remember the name — and remember the reason for the clip, because it comes back in chapter 11.`),
      ));

      root.append(section('When the reward is not what you meant',
        p(`In 2016 OpenAI trained an agent on a boat-racing game called CoastRunners. "Finish the race" was hard to learn from, so they used the game's own score, which mostly comes from hitting targets along the course.`),
        p(`The agent found a lagoon where three targets respawned quickly and drove in circles there forever: crashing, catching fire, going backwards, never finishing — while scoring 20% higher than human players.`),
        callout('tryit', '🖐 Try this: what the reward says versus what you meant',
          `<b>1.</b> Leave it on the points-based reward and let it run for ten seconds. Read the score rate.<br>
           <b>2.</b> Switch the <b>reward function</b> and compare. The behaviour that <i>never finishes a lap</i> wins on the proxy.<br>
           <b>3.</b> Sit with that for a moment: <b>the agent is not confused. The reward is.</b>`),
        rewardHack(),
        p(`This has a name from economics — <em>Goodhart's law</em>: when a measure becomes a target, it stops being a good measure. Every proxy has gaps between what it counts and what you want, and a strong optimiser will find those gaps and pour itself into them.`),
        p(`Agents have learned to pause Tetris forever to avoid losing, to exploit physics-engine bugs to "run" by vibrating, and to knock a lego brick over so a height sensor reads high.`),
        callout('warning', '⚠️ Hold onto this one',
          `In chapter 11 the reward becomes "a human preferred this answer", and the model discovers that humans have gaps too: we tend to prefer answers that are <b>confident, long and flattering</b>.<br>
           That is reward hacking with people as the environment, and it is the single hardest unsolved problem in making assistants useful rather than merely pleasant.`),
      ));

      root.append(section('Why this matters for modern AI',
        p(`A language model fresh out of pretraining is a supervised product: it predicts the next word of the internet. Turning it into an assistant is an RL problem.`),
        p(`The <b>agent</b> is the model. The <b>environment</b> is a conversation. The <b>action</b> is the next token. And the <b>reward</b> is a score from a model trained to imitate human preferences. That is <em>RLHF</em>, and it is trained with PPO or one of its descendants.`),
        p(`RLHF adds a second rail on top of PPO's clip: a penalty on how far the model being trained drifts from the frozen model it started from. Without it the policy quickly discovers that a few weird, repetitive token sequences score very highly with the reward model, and it collapses into fluent-looking gibberish. The penalty is a leash back to English.`),
        p(`The 2024–2025 generation of <em>reasoning models</em> pushed RL further. Instead of a preference score they use rewards you can <b>verify</b>: did the maths answer match? did the code pass the tests? The environment is a coding task, the episode is a long chain of thought, the reward is 1 or 0 at the end.`),
        p(`Credit assignment over thousands of tokens, exploration to find non-obvious strategies, reward hacking when the tests are weak: every problem in this chapter, at scale.`),
        callout('history', '📜 The decade RL went from toy mazes to superhuman',
          `<b>1989:</b> Chris Watkins introduces Q-learning in his thesis, with a convergence proof.<br>
           <b>2013–2015:</b> DeepMind's DQN learns to play Atari from raw pixels with the same network for every game, then reaches human level across 49 of them.<br>
           <b>2016:</b> AlphaGo beats Lee Sedol 4–1. <b>2017:</b> AlphaGo Zero drops human games entirely and learns from self-play alone, beating the previous version 100–0.<br>
           <b>2017:</b> PPO is published, and quietly becomes the default policy-gradient algorithm everywhere.<br>
           <b>2022:</b> PPO is used to turn a language model into ChatGPT — and RL stops being a games technique.`),
        callout('example', '🌍 Where RL is quietly running today',
          `<b>Data-centre cooling:</b> DeepMind's controller cut Google's cooling energy substantially by learning setpoints no human schedule had tried.<br>
           <b>Robotics:</b> policies trained in simulation and transferred to real hardware — the trick being <em>domain randomisation</em>, randomising friction, lighting, masses and delays so wildly that the real world looks like one more variation. It is chapter 4's data augmentation, applied to physics.<br>
           <b>Recommendations and ad auctions:</b> bandit algorithms deciding which headline or price to show you, right now.<br>
           <b>Every assistant you use:</b> via RLHF, which is chapter 11.`),
        p(`One picture to keep: <b>reinforcement learning is what you do when nobody can tell you the right answer — only whether it went well.</b>`),
      ));

      root.append(ctx.quiz([
        { q: 'You found the goal in the opening demo but could not say which move was best. What is that problem called?', options: ['The exploration problem', 'The credit-assignment problem: a single delayed reward must be spread back over every decision that led to it', 'Overfitting', 'The discount problem'], answer: 1, explain: 'It is why RL needs more than gradient descent on a loss — there is no per-step loss to descend. Q-learning solves it by letting value information flow backwards one step per update, which is the ripple you watched spread out from the goal.' },
        { q: 'With γ = 0.5, why does an agent walk past a +20 reward twelve steps away in favour of a +3 two steps away?', options: ['The +20 is harder to reach', '20 × 0.5¹² ≈ 0.005, which is far less than 3 × 0.5² = 0.75 — the distant reward is discounted almost to nothing', 'Because ε-greedy makes it act randomly', 'Because Q-learning cannot handle large rewards'], answer: 1, explain: 'γ is the single number deciding how far ahead an agent bothers to look. Nothing about the world changed when you dragged that slider — only the agent\'s patience, and with it whether a long-term plan is possible at all.' },
        { q: 'You set exploration to 100%, so the agent moves entirely at random, and the Q table still converges on a good policy. Why?', options: ['The demo is pre-computed', 'Q-learning is off-policy: its update uses the value of the best next action regardless of what was actually done, so it learns the best policy while behaving randomly', 'Random movement is optimal in a maze', 'Because ε decays automatically'], answer: 1, explain: 'The update target uses max(Q[s2]), not the action actually taken next. That decoupling of "how I behave" from "what I learn" is precisely what off-policy means, and it is why exploration can be cranked up without destroying the result.' },
        { q: 'Why does a too-large policy update in RL cause collapse rather than mere overshoot?', options: ['RL uses a larger learning rate', 'The model chooses its own next training set — a policy that moves too far no longer resembles the one that gathered the data, so the data becomes worthless', 'Rewards become negative', 'The value network stops training'], answer: 1, explain: 'This is the failure you produced by turning clipping off and pushing the step size past 1.5. PPO bounds each update to roughly ±20% so the data stays approximately valid, and that clip is almost all PPO is.' },
        { q: 'A boat-racing agent drove in circles forever, never finishing, and scored 20% above human players. What went wrong?', options: ['The agent was buggy', 'Nothing went wrong with the agent — it perfectly optimised the reward it was given, which was not what the designers meant', 'The reward was too small', 'It needed more training'], answer: 1, explain: 'Goodhart\'s law: when a measure becomes a target it stops being a good measure. Hold onto it — in chapter 11 the reward becomes "a human preferred this answer", and humans reliably prefer confident, long, flattering answers.' },
      ]));

      root.append(section('Go deeper',
        ul([
          '<a href="http://incompleteideas.net/book/the-book-2nd.html" target="_blank" rel="noopener">Sutton &amp; Barto, <i>Reinforcement Learning: An Introduction</i></a> — the field\'s standard text, free online, and unusually readable. Chapters 3–6 are this chapter in proper depth.',
          '<a href="https://spinningup.openai.com/en/latest/" target="_blank" rel="noopener">OpenAI Spinning Up in Deep RL</a> — the best hands-on introduction to policy gradients, with clean, readable code.',
          '<a href="https://www.davidsilver.uk/teaching/" target="_blank" rel="noopener">David Silver\'s RL lecture course</a> — from the lead researcher on AlphaGo.',
          '<a href="https://www.nature.com/articles/nature14236" target="_blank" rel="noopener">Mnih et al. (2015), "Human-level control through deep reinforcement learning"</a> — DQN, and Atari from raw pixels. The <a href="https://arxiv.org/abs/1312.5602" target="_blank" rel="noopener">2013 workshop version</a> is shorter.',
          '<a href="https://openai.com/index/faulty-reward-functions/" target="_blank" rel="noopener">OpenAI, "Faulty Reward Functions in the Wild"</a> — the CoastRunners boat, with video. Worth watching once.',
          '<a href="https://lilianweng.github.io/posts/2018-02-19-rl-overview/" target="_blank" rel="noopener">Lilian Weng, "A (Long) Peek into Reinforcement Learning"</a> — a dense, accurate survey from tabular methods to PPO.',
        ])));

      /* ================================================================== */
      /*  INTERACTIVE A — GRIDWORLD Q-LEARNING                               */
      /* ================================================================== */
      function gridworld() {
        const COLS = 8, ROWS = 6, CS = 50, GX = 12, GY = 12;
        const DX = [0, 1, 0, -1], DY = [-1, 0, 1, 0];
        const key = (c, r) => c + ',' + r;
        let walls, goal, pit, start;
        function defaultMap() {
          walls = new Set(['2,1', '2,2', '2,3', '5,2', '5,3', '5,4', '4,0']);
          goal = { c: 7, r: 0 }; pit = { c: 6, r: 1 }; start = { c: 0, r: 5 };
        }
        defaultMap();
        const prm = { alpha: 0.3, gamma: 0.9, eps: 0.2, speed: 20, q0: 0 };
        let Q, agent, disp, episode, stepsInEp, epReturn, returns, editMode = false, playing = false, dragging = null;

        function resetQ() {
          Q = [];
          // q0 is the initial belief about every action. Above −0.1/(1−γ) it is *optimistic*:
          // an untried action always outranks a tried one, so even a greedy agent explores.
          for (let i = 0; i < COLS * ROWS; i++) Q.push([prm.q0, prm.q0, prm.q0, prm.q0]);
          episode = 0; returns = []; startEpisode(); disp = { x: agent.c, y: agent.r };
        }
        function startEpisode() { agent = { c: start.c, r: start.r }; stepsInEp = 0; epReturn = 0; }
        const isWall = (c, r) => walls.has(key(c, r));
        function argmaxRand(arr) {
          let best = -Infinity, idx = [];
          for (let i = 0; i < arr.length; i++) {
            if (arr[i] > best + 1e-9) { best = arr[i]; idx = [i]; } else if (Math.abs(arr[i] - best) <= 1e-9) idx.push(i);
          }
          return idx[Math.floor(Math.random() * idx.length)];
        }
        function step() {
          const s = agent.r * COLS + agent.c;
          const a = Math.random() < prm.eps ? Math.floor(Math.random() * 4) : argmaxRand(Q[s]);
          let nc = agent.c + DX[a], nr = agent.r + DY[a];
          if (nc < 0 || nc >= COLS || nr < 0 || nr >= ROWS || isWall(nc, nr)) { nc = agent.c; nr = agent.r; }
          let r = -0.1, done = false;
          if (nc === goal.c && nr === goal.r) { r = 10; done = true; }
          else if (nc === pit.c && nr === pit.r) { r = -10; done = true; }
          const s2 = nr * COLS + nc;
          const target = done ? r : r + prm.gamma * Math.max(Q[s2][0], Q[s2][1], Q[s2][2], Q[s2][3]);
          let q = Q[s][a] + prm.alpha * (target - Q[s][a]);
          if (!isFinite(q)) q = 0;
          Q[s][a] = ctx.clamp(q, -50, 50);
          agent.c = nc; agent.r = nr; stepsInEp++; epReturn += r;
          if (done || stepsInEp >= 300) {
            returns.push(epReturn); if (returns.length > 150) returns.shift();
            episode++; startEpisode(); disp = { x: agent.c, y: agent.r };
          }
        }
        resetQ();

        const [cv, g] = ctx.canvas(720, 330);
        const readout = ctx.readout();

        const untouched = (v) => Math.abs(v - prm.q0) < 1e-9;
        function valColor(v) {
          if (untouched(v)) return 'rgba(148,163,184,0.10)';   // grey: this action has never been tried here
          const m = ctx.clamp(Math.abs(v) / 10, 0, 1);
          const a = 0.08 + 0.85 * Math.sqrt(m);
          return v >= 0 ? 'rgba(56,217,169,' + a + ')' : 'rgba(251,113,133,' + a + ')';
        }
        function tri(x1, y1, x2, y2, x3, y3, fill) {
          g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.lineTo(x3, y3); g.closePath(); g.fillStyle = fill; g.fill();
        }
        function arrow(cx, cy, a, len) {
          const ex = cx + DX[a] * len, ey = cy + DY[a] * len;
          g.strokeStyle = 'rgba(230,235,245,0.85)'; g.lineWidth = 1.5;
          g.beginPath(); g.moveTo(cx - DX[a] * len * 0.6, cy - DY[a] * len * 0.6); g.lineTo(ex, ey); g.stroke();
          g.beginPath();
          g.moveTo(ex, ey);
          g.lineTo(ex - DX[a] * 4 + DY[a] * 3, ey - DY[a] * 4 + DX[a] * 3);
          g.lineTo(ex - DX[a] * 4 - DY[a] * 3, ey - DY[a] * 4 - DX[a] * 3);
          g.closePath(); g.fillStyle = 'rgba(230,235,245,0.85)'; g.fill();
        }
        function draw() {
          g.clearRect(0, 0, cv.W, cv.H);
          g.fillStyle = C.bg; g.fillRect(0, 0, cv.W, cv.H);
          for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
            const x = GX + c * CS, y = GY + r * CS, cx = x + CS / 2, cy = y + CS / 2;
            if (isWall(c, r)) { g.fillStyle = '#243044'; g.fillRect(x, y, CS, CS); continue; }
            const q = Q[r * COLS + c];
            const isGoal = c === goal.c && r === goal.r, isPit = c === pit.c && r === pit.r;
            if (isGoal) { g.fillStyle = 'rgba(56,217,169,0.9)'; g.fillRect(x, y, CS, CS); }
            else if (isPit) { g.fillStyle = 'rgba(251,113,133,0.9)'; g.fillRect(x, y, CS, CS); }
            else {
              tri(x, y, x + CS, y, cx, cy, valColor(q[0]));
              tri(x + CS, y, x + CS, y + CS, cx, cy, valColor(q[1]));
              tri(x, y + CS, x + CS, y + CS, cx, cy, valColor(q[2]));
              tri(x, y, x, y + CS, cx, cy, valColor(q[3]));
              if (q.some(v => !untouched(v))) {
                const bi = argmaxIdx(q);
                arrow(cx, cy, bi, 9);
                // the state's value, max_a Q(s,a) — the number the worked example converges on
                g.fillStyle = 'rgba(230,235,245,0.72)'; g.font = '9px Inter, sans-serif';
                g.textAlign = 'center'; g.textBaseline = 'bottom';
                g.fillText(q[bi].toFixed(1), cx, y + CS - 3);
              }
            }
            g.strokeStyle = '#1b2434'; g.lineWidth = 1; g.strokeRect(x + 0.5, y + 0.5, CS - 1, CS - 1);
            if (isGoal || isPit) {
              g.fillStyle = '#0a0e16'; g.font = 'bold 13px Inter, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
              g.fillText(isGoal ? '+10' : '−10', cx, cy);
            }
            if (c === start.c && r === start.r) {
              g.fillStyle = 'rgba(230,235,245,0.6)'; g.font = '10px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'top';
              g.fillText('start', x + 3, y + 3);
            }
          }
          // agent dot (smoothly interpolated)
          const ax = GX + (disp.x + 0.5) * CS, ay = GY + (disp.y + 0.5) * CS;
          g.beginPath(); g.arc(ax, ay, 9, 0, Math.PI * 2); g.fillStyle = C.accent; g.fill();
          g.strokeStyle = '#fff'; g.lineWidth = 2; g.stroke();
          if (editMode) {
            g.fillStyle = C.warn; g.font = '11px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'top';
            g.fillText('EDIT: click = toggle wall · drag goal / pit / start', GX, GY + ROWS * CS + 4);
          }
          // returns chart. Range covers the true worst case: 299 steps of −0.1 then the pit = −39.9.
          const px = 440, py = 20, pw = 260, ph = 250;
          const RLO = -40, RHI = 10, RSPAN = RHI - RLO, RPAD = 2;
          const retY = (v) => py + RPAD + (ph - 2 * RPAD) * (1 - (ctx.clamp(v, RLO, RHI) - RLO) / RSPAN);
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(px, py, pw, ph);
          g.fillStyle = C.muted; g.font = '11px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'top';
          g.fillText('return per episode (last ' + returns.length + ')', px, py - 14);
          const y0 = retY(0);
          g.strokeStyle = '#1b2434'; g.beginPath(); g.moveTo(px, y0); g.lineTo(px + pw, y0); g.stroke();
          g.fillStyle = C.muted; g.textAlign = 'right'; g.fillText('+10', px - 4, py); g.fillText('0', px - 4, y0 - 5); g.fillText('−40', px - 4, py + ph - 10);
          if (returns.length > 1) {
            g.save(); g.beginPath(); g.rect(px, py, pw, ph); g.clip();
            g.strokeStyle = C.green; g.lineWidth = 1.5; g.beginPath();
            returns.forEach((v, i) => {
              const x = px + i / (returns.length - 1) * pw, y = retY(v);
              i ? g.lineTo(x, y) : g.moveTo(x, y);
            });
            g.stroke();
            g.restore();
          }
          // legend, split over two lines so it stays inside the 260px-wide right column
          g.fillStyle = C.muted; g.textAlign = 'left'; g.font = '11px Inter, sans-serif';
          g.fillText('Q(s,a): green = positive, red = negative,', px, py + ph + 10);
          g.fillText('arrow = greedy action', px, py + ph + 25);
        }
        function argmaxIdx(arr) { let b = 0; for (let i = 1; i < 4; i++) if (arr[i] > arr[b]) b = i; return b; }

        ctx.loop((dt) => {
          if (playing) { const n = prm.speed; for (let i = 0; i < n; i++) step(); }
          const k = Math.min(1, dt * (prm.speed > 5 ? 30 : 14));
          disp.x = ctx.lerp(disp.x, agent.c, k); disp.y = ctx.lerp(disp.y, agent.r, k);
          draw();
          const last = returns.slice(-20);
          const avg = last.length ? (last.reduce((a, b) => a + b, 0) / last.length).toFixed(2) : '—';
          readout.set({ episode, steps: stepsInEp, 'return': epReturn.toFixed(1), 'avg return (last 20)': avg, 'ε': prm.eps.toFixed(2), 'Q₀': prm.q0.toFixed(1) });
        });

        // pointer editing
        function cellAt(ev) {
          const pos = cv.pos(ev);
          const c = Math.floor((pos.x - GX) / CS), r = Math.floor((pos.y - GY) / CS);
          if (c < 0 || c >= COLS || r < 0 || r >= ROWS) return null;
          return { c, r };
        }
        const same = (a, b) => a.c === b.c && a.r === b.r;
        cv.addEventListener('pointerdown', (ev) => {
          if (!editMode) return;
          ev.preventDefault();
          const cell = cellAt(ev); if (!cell) return;
          if (same(cell, goal)) dragging = 'goal';
          else if (same(cell, pit)) dragging = 'pit';
          else if (same(cell, start)) dragging = 'start';
          else { const k = key(cell.c, cell.r); if (walls.has(k)) walls.delete(k); else walls.add(k); startEpisode(); disp = { x: agent.c, y: agent.r }; }
          try { cv.setPointerCapture(ev.pointerId); } catch (e) {}
        });
        cv.addEventListener('pointermove', (ev) => {
          if (!editMode || !dragging) return;
          const cell = cellAt(ev); if (!cell || isWall(cell.c, cell.r)) return;
          const others = { goal, pit, start };
          for (const k in others) if (k !== dragging && same(cell, others[k])) return;
          if (dragging === 'goal') goal = cell; else if (dragging === 'pit') pit = cell; else start = cell;
          startEpisode(); disp = { x: agent.c, y: agent.r };
        });
        const up = () => { dragging = null; };
        cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
        // safety net: if setPointerCapture failed and the release happens off-canvas, never stay stuck mid-drag
        window.addEventListener('pointerup', up); window.addEventListener('pointercancel', up);
        ctx.onCleanup(() => { window.removeEventListener('pointerup', up); window.removeEventListener('pointercancel', up); });
        cv.style.touchAction = 'none';

        const playBtn = ctx.button('▶ Play', () => { playing = !playing; playBtn.textContent = playing ? '❚❚ Pause' : '▶ Play'; }, 'primary');
        const editBtn = ctx.button('Edit map: off', () => { editMode = !editMode; editBtn.textContent = 'Edit map: ' + (editMode ? 'ON' : 'off'); });
        const controls = [
          playBtn,
          ctx.button('Step ×1', () => { step(); }),
          ctx.button('Reset Q', () => { resetQ(); }),
          ctx.button('Default map', () => { defaultMap(); resetQ(); }),
          editBtn,
          ctx.slider({ label: 'speed (steps/frame)', min: 1, max: 200, step: 1, value: prm.speed, onChange: v => { prm.speed = v; } }),
          ctx.slider({ label: 'ε exploration', min: 0, max: 1, step: 0.01, value: prm.eps, onChange: v => { prm.eps = v; } }),
          ctx.slider({ label: 'α learning rate', min: 0.01, max: 1, step: 0.01, value: prm.alpha, onChange: v => { prm.alpha = v; } }),
          ctx.slider({ label: 'γ discount', min: 0, max: 0.99, step: 0.01, value: prm.gamma, onChange: v => { prm.gamma = v; } }),
          ctx.slider({ label: 'Q₀ optimism (resets table)', min: -3, max: 10, step: 0.5, value: prm.q0, fmt: (v) => (+v).toFixed(1), onChange: v => { prm.q0 = v; resetQ(); } }),
        ];
        return ctx.figure(cv, 'Tabular Q-learning on an 8×6 grid. Goal +10, pit −10, every step −0.1, episodes capped at 300 steps. Grey triangles are actions never tried from that cell. The Q table is kept when you edit the map, so you can watch the agent unlearn the old goal — but changing Q₀ wipes it, since it sets the starting belief.', controls, readout);
      }

      /* ================================================================== */
      /*  INTERACTIVE B — MULTI-ARMED BANDIT                                 */
      /* ================================================================== */
      function bandit() {
        const K = 5;
        let T = 1000;                      // horizon; the reader can raise it to see UCB overtake ε-greedy
        let rates, best, revealed = false, eps = 0.1;
        let you, agents, running = false;
        function newAgent(name, color) { return { name, color, n: new Array(K).fill(0), sum: new Array(K).fill(0), t: 0, regret: 0, reward: 0, hist: [0] }; }
        function reset() {
          rates = []; for (let i = 0; i < K; i++) rates.push(Math.round(ctx.rand(0.1, 0.85) * 100) / 100);
          best = Math.max(...rates);
          you = newAgent('you', C.warn); agents = [newAgent('ε-greedy', C.accent), newAgent('UCB', C.green)];
          running = false; revealed = false; revealBtn.textContent = 'Reveal true rates';
        }
        function pull(ag, k) {
          const r = Math.random() < rates[k] ? 1 : 0;
          ag.n[k]++; ag.sum[k] += r; ag.t++; ag.reward += r; ag.regret += best - rates[k];
          ag.hist.push(ag.regret);
          return r;
        }
        function mean(ag, k) { return ag.n[k] ? ag.sum[k] / ag.n[k] : 0; }
        function chooseEps(ag) {
          if (Math.random() < eps) return Math.floor(Math.random() * K);
          let b = 0, bv = -1, ties = [];
          for (let k = 0; k < K; k++) { const m = mean(ag, k); if (m > bv + 1e-12) { bv = m; ties = [k]; } else if (Math.abs(m - bv) <= 1e-12) ties.push(k); }
          b = ties[Math.floor(Math.random() * ties.length)];
          return b;
        }
        function chooseUCB(ag) {
          for (let k = 0; k < K; k++) if (ag.n[k] === 0) return k;
          let b = 0, bv = -Infinity;
          for (let k = 0; k < K; k++) { const s = mean(ag, k) + Math.sqrt(2 * Math.log(ag.t + 1) / ag.n[k]); if (s > bv) { bv = s; b = k; } }
          return b;
        }

        const [cv, g] = ctx.canvas(720, 340);
        const readout = ctx.readout();
        const MX = 14, MW = 62, MG = 8, MY = 30, MH = 220;
        function machineRect(k) { return { x: MX + k * (MW + MG), y: MY, w: MW, h: MH }; }
        let flash = new Array(K).fill(0);

        function draw(dt) {
          g.clearRect(0, 0, cv.W, cv.H);
          g.fillStyle = C.bg; g.fillRect(0, 0, cv.W, cv.H);
          g.fillStyle = C.muted; g.font = '11px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'top';
          g.fillText('click a machine to pull it · bar = your estimated payout · n = your pulls', MX, 8);
          for (let k = 0; k < K; k++) {
            const R = machineRect(k);
            flash[k] = Math.max(0, flash[k] - dt * 3);
            g.fillStyle = flash[k] > 0 ? 'rgba(251,191,36,' + (0.15 + 0.35 * flash[k]) + ')' : '#111827';
            g.strokeStyle = C.line; g.lineWidth = 1;
            g.beginPath(); g.roundRect ? g.roundRect(R.x, R.y, R.w, R.h, 8) : g.rect(R.x, R.y, R.w, R.h); g.fill(); g.stroke();
            // estimate bar (you)
            const m = mean(you, k), bh = (R.h - 60) * m;
            g.fillStyle = 'rgba(251,191,36,0.55)'; g.fillRect(R.x + 14, R.y + R.h - 34 - bh, R.w - 28, bh);
            // agents' estimates as thin ticks
            agents.forEach((ag, i) => { if (!ag.n[k]) return; const y = R.y + R.h - 34 - (R.h - 60) * mean(ag, k); g.strokeStyle = ag.color; g.lineWidth = 2; g.beginPath(); g.moveTo(R.x + 6 + i * 4, y); g.lineTo(R.x + 12 + i * 4, y); g.stroke(); });
            if (revealed) { const y = R.y + R.h - 34 - (R.h - 60) * rates[k]; g.strokeStyle = C.danger; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(R.x + 4, y); g.lineTo(R.x + R.w - 4, y); g.stroke(); g.setLineDash([]); }
            g.fillStyle = C.text; g.font = 'bold 12px Inter, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'top';
            g.fillText('#' + (k + 1), R.x + R.w / 2, R.y + 6);
            g.fillStyle = C.muted; g.font = '10px Inter, sans-serif';
            g.fillText('n=' + you.n[k], R.x + R.w / 2, R.y + R.h - 28);
            g.fillText(you.n[k] ? (100 * m).toFixed(0) + '%' : '?', R.x + R.w / 2, R.y + R.h - 16);
            if (revealed) { g.fillStyle = C.danger; g.fillText('true ' + (100 * rates[k]).toFixed(0) + '%', R.x + R.w / 2, R.y + 20); }
          }
          // legend
          let ly = MY + MH + 14;
          [you, ...agents].forEach((ag) => { g.fillStyle = ag.color; g.fillRect(MX, ly + 3, 10, 3); g.fillStyle = C.muted; g.textAlign = 'left'; g.font = '11px Inter, sans-serif'; g.fillText(ag.name + ': regret ' + ag.regret.toFixed(1) + ' after ' + ag.t + ' pulls', MX + 16, ly - 2); ly += 16; });
          // regret plot
          const px = 400, py = 30, pw = 300, ph = 250;
          g.strokeStyle = C.line; g.strokeRect(px, py, pw, ph);
          g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText('cumulative regret vs pulls', px, 8);
          const xmax = Math.max(T, you.t), ymax = Math.max(20, you.regret, agents[0].regret, agents[1].regret) * 1.1;
          g.textAlign = 'right'; g.fillText(ymax.toFixed(0), px - 4, py); g.fillText('0', px - 4, py + ph - 10);
          g.textAlign = 'center'; g.fillText(xmax + ' pulls', px + pw / 2, py + ph + 6);
          g.save(); g.beginPath(); g.rect(px, py, pw, ph); g.clip();
          [you, ...agents].forEach((ag) => {
            if (ag.hist.length < 2) return;
            g.strokeStyle = ag.color; g.lineWidth = 1.5; g.beginPath();
            const stride = Math.max(1, Math.floor(ag.hist.length / 300));
            // inset by half the stroke so a curve sitting on regret 0 is not sliced by the clip
            for (let i = 0; i < ag.hist.length; i += stride) { const x = px + i / xmax * pw, y = py + ph - 1 - ag.hist[i] / ymax * (ph - 2); i ? g.lineTo(x, y) : g.moveTo(x, y); }
            g.stroke();
          });
          g.restore();
          // random baseline (dashed): regret grows at (best - mean rate) per pull
          const avgGap = best - rates.reduce((a, b) => a + b, 0) / K;
          g.strokeStyle = 'rgba(148,163,184,0.5)'; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(px, py + ph);
          g.lineTo(px + pw, py + ph - Math.min(ph, avgGap * xmax / ymax * ph)); g.stroke(); g.setLineDash([]);
          g.fillStyle = 'rgba(148,163,184,0.7)'; g.textAlign = 'right'; g.fillText('random', px + pw - 4, py + ph - Math.min(ph, avgGap * xmax / ymax * ph) - 12);
        }

        cv.addEventListener('pointerdown', (ev) => {
          ev.preventDefault();
          const pos = cv.pos(ev);
          for (let k = 0; k < K; k++) { const R = machineRect(k); if (pos.x >= R.x && pos.x <= R.x + R.w && pos.y >= R.y && pos.y <= R.y + R.h) { const r = pull(you, k); flash[k] = r ? 1 : 0.3; } }
        });
        cv.style.touchAction = 'manipulation';

        ctx.loop((dt) => {
          if (running) {
            // a run always takes ~125 frames (≈2 s) whatever the horizon, and stays far under 1 ms/frame
            const perFrame = Math.max(8, Math.ceil(T / 125));
            for (let i = 0; i < perFrame; i++) {
              if (agents[0].t < T) pull(agents[0], chooseEps(agents[0]));
              if (agents[1].t < T) pull(agents[1], chooseUCB(agents[1]));
            }
            if (agents[0].t >= T && agents[1].t >= T) running = false;
          }
          draw(dt);
          readout.set({ 'your pulls': you.t, 'your £': you.reward, 'your regret': you.regret.toFixed(1), 'ε-greedy regret': agents[0].regret.toFixed(1), 'UCB regret': agents[1].regret.toFixed(1) });
        });

        const revealBtn = ctx.button('Reveal true rates', () => { revealed = !revealed; revealBtn.textContent = revealed ? 'Hide true rates' : 'Reveal true rates'; });
        const fmtT = (n) => (n >= 1000 ? (n / 1000) + 'k' : String(n));
        const runBtn = ctx.button('Run agents (1k pulls)', () => { agents = [newAgent('ε-greedy', C.accent), newAgent('UCB', C.green)]; running = true; }, 'primary');
        const controls = [
          runBtn,
          revealBtn,
          ctx.button('New machines', reset),
          ctx.select({
            label: 'horizon (pulls per run)',
            options: [{ value: '1000', label: '1,000 pulls' }, { value: '5000', label: '5,000 pulls' }, { value: '20000', label: '20,000 pulls' }],
            value: '1000',
            onChange: (v) => {
              const n = parseInt(v, 10);
              T = (n === 5000 || n === 20000) ? n : 1000;
              runBtn.textContent = 'Run agents (' + fmtT(T) + ' pulls)';
              agents = [newAgent('ε-greedy', C.accent), newAgent('UCB', C.green)]; running = false;
            },
          }),
          ctx.slider({ label: 'ε for ε-greedy', min: 0, max: 1, step: 0.01, value: eps, onChange: v => { eps = v; } }),
        ];
        reset();
        return ctx.figure(cv, 'Five Bernoulli bandits with hidden rates. Regret = (best rate − rate of the arm you pulled), summed over pulls. UCB uses the bonus √(2·ln t / n). The dashed line is what pulling at random would cost. Raise the horizon to 20,000 to see the two strategies change places.', controls, readout);
      }

      /* ================================================================== */
      /*  INTERACTIVE C — REWARD HACKING (scripted)                          */
      /* ================================================================== */
      function rewardHack() {
        const [cv, g] = ctx.canvas(720, 330);
        const readout = ctx.readout();
        const cx = 360, cy = 165, rx = 270, ry = 110;
        const trackPt = (t) => ({ x: cx + rx * Math.cos(t * Math.PI * 2), y: cy + ry * Math.sin(t * Math.PI * 2) });
        const lagoon = { x: 250, y: 165, r: 34 };
        let mode = 'finish'; // 'finish' | 'coins'
        let boat = { x: 0, y: 0, ang: 0 }, tParam = 0, theta = 0, points = 0, laps = 0, elapsed = 0;
        let coins = [];
        // scoring rate measured separately per reward function, so the two can be compared honestly
        const rate = { finish: 0, coins: 0 };
        function makeCoins() {
          coins = [];
          for (let i = 0; i < 8; i++) { const pt = trackPt((i + 0.5) / 8); coins.push({ x: pt.x, y: pt.y, respawn: 6, timer: 0, val: 10 }); }
          for (let i = 0; i < 3; i++) { const a = i / 3 * Math.PI * 2; coins.push({ x: lagoon.x + lagoon.r * Math.cos(a), y: lagoon.y + lagoon.r * Math.sin(a), respawn: 0.8, timer: 0, val: 10, lag: true }); }
        }
        function reset() { points = 0; laps = 0; elapsed = 0; tParam = 0; theta = 0; const s = trackPt(0); boat = { x: s.x, y: s.y, ang: Math.PI / 2 }; makeCoins(); }
        reset();

        function drawBoat(x, y, ang) {
          g.save(); g.translate(x, y); g.rotate(ang);
          g.beginPath(); g.moveTo(12, 0); g.lineTo(-8, 6); g.lineTo(-5, 0); g.lineTo(-8, -6); g.closePath();
          g.fillStyle = C.accent; g.fill(); g.strokeStyle = '#fff'; g.lineWidth = 1; g.stroke();
          g.restore();
        }
        ctx.loop((dt) => {
          elapsed += dt;
          // target position on the scripted path
          let target;
          if (mode === 'finish') {
            const prev = tParam; tParam = (tParam + dt * 0.12) % 1;
            if (prev > tParam) laps++;
            target = trackPt(tParam);
          } else {
            theta += dt * 3.2;
            target = { x: lagoon.x + lagoon.r * Math.cos(theta), y: lagoon.y + lagoon.r * Math.sin(theta) };
          }
          const k = Math.min(1, dt * 4);
          const nx = ctx.lerp(boat.x, target.x, k), ny = ctx.lerp(boat.y, target.y, k);
          if (Math.hypot(nx - boat.x, ny - boat.y) > 0.05) boat.ang = Math.atan2(ny - boat.y, nx - boat.x);
          boat.x = nx; boat.y = ny;
          // coins
          for (const c of coins) {
            if (c.timer > 0) { c.timer -= dt; continue; }
            if (Math.hypot(c.x - boat.x, c.y - boat.y) < 14) { points += c.val; c.timer = c.respawn; }
          }
          // draw
          g.clearRect(0, 0, cv.W, cv.H);
          g.fillStyle = C.bg; g.fillRect(0, 0, cv.W, cv.H);
          g.strokeStyle = '#1e2a44'; g.lineWidth = 34; g.beginPath(); g.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2); g.stroke();
          g.strokeStyle = '#2b3a58'; g.lineWidth = 1; g.setLineDash([6, 8]); g.beginPath(); g.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
          // finish line
          const f = trackPt(0);
          g.strokeStyle = '#fff'; g.lineWidth = 3; g.beginPath(); g.moveTo(f.x - 17, f.y); g.lineTo(f.x + 17, f.y); g.stroke();
          g.fillStyle = C.muted; g.font = '11px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'middle'; g.fillText('finish', f.x + 22, f.y);
          // lagoon
          g.fillStyle = 'rgba(124,156,255,0.08)'; g.beginPath(); g.arc(lagoon.x, lagoon.y, lagoon.r + 18, 0, Math.PI * 2); g.fill();
          g.fillStyle = C.muted; g.textAlign = 'center'; g.fillText('lagoon: coins respawn every 0.8 s', lagoon.x, lagoon.y + lagoon.r + 30);
          for (const c of coins) {
            const alive = c.timer <= 0;
            g.beginPath(); g.arc(c.x, c.y, 6, 0, Math.PI * 2);
            g.fillStyle = alive ? C.warn : 'rgba(251,191,36,0.15)'; g.fill();
          }
          drawBoat(boat.x, boat.y, boat.ang);
          g.fillStyle = mode === 'coins' ? C.danger : C.green; g.font = 'bold 13px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'top';
          g.fillText(mode === 'coins' ? 'reward = points  →  learned behaviour: circle the lagoon forever' : 'reward = laps finished  →  learned behaviour: race', 14, 10);
          g.fillStyle = C.muted; g.font = '11px Inter, sans-serif'; g.fillText('scripted re-enactment of the CoastRunners result (OpenAI, 2016)', 14, 28);
          // score-rate scoreboard: only trusted once a few seconds of this mode have elapsed
          if (elapsed > 3) rate[mode] = points / elapsed * 60;
          g.textAlign = 'right'; g.textBaseline = 'top'; g.font = '11px Inter, sans-serif';
          g.fillStyle = C.muted; g.fillText('game score per minute', cv.W - 14, 10);
          g.fillStyle = C.green; g.fillText('racing: ' + (rate.finish ? rate.finish.toFixed(0) : '—'), cv.W - 14, 26);
          g.fillStyle = C.danger; g.fillText('circling the lagoon: ' + (rate.coins ? rate.coins.toFixed(0) : '—'), cv.W - 14, 42);
          readout.set({ reward: mode === 'coins' ? 'points' : 'laps', points, laps, 'points/min': elapsed > 1 ? (points / elapsed * 60).toFixed(0) : '—' });
        });
        // switching the reward function restarts the clock, otherwise points/min mixes the two behaviours
        const modeSel = ctx.select({ label: 'reward function', options: [{ value: 'finish', label: 'laps finished (what we meant)' }, { value: 'coins', label: 'points from coins (the proxy)' }], value: mode, onChange: v => { mode = v === 'coins' ? 'coins' : 'finish'; reset(); } });
        return ctx.figure(cv, 'With reward = points, the highest-scoring behaviour never finishes a lap. Let each mode run for ten seconds and compare the score rates in the top right: the behaviour that never finishes wins on the proxy. The agent is not confused; the reward is.', [modeSel, ctx.button('Reset', () => { rate.finish = 0; rate.coins = 0; reset(); })], readout);
      }

      /* ================================================================== */
      /*  INTERACTIVE — BE THE AGENT: no map, no labels, only consequences   */
      /* ================================================================== */
      function blindAgent() {
        const [cv, g] = ctx.canvas(720, 340);
        const COLS = 6, ROWS = 5;
        const GOAL = { c: 5, r: 0 }, PIT = [{ c: 3, r: 2 }, { c: 1, r: 3 }];
        const START = { c: 0, r: 4 };
        let pos = { ...START }, trail = [{ ...START }], rewards = [], total = 0, steps = 0;
        let done = false, revealed = false, episodes = 0, bestSteps = null;
        const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
        const FONT = '13px Inter, system-ui, sans-serif';

        const isPit = (c, r) => PIT.some(q => q.c === c && q.r === r);
        const isGoal = (c, r) => GOAL.c === c && GOAL.r === r;

        function move(dc, dr) {
          if (done) return;
          const c = ctx.clamp(pos.c + dc, 0, COLS - 1), r = ctx.clamp(pos.r + dr, 0, ROWS - 1);
          pos = { c, r };
          trail.push({ c, r });
          steps++;
          let rew = -0.1;
          if (isGoal(c, r)) { rew = 10; done = true; }
          else if (isPit(c, r)) { rew = -10; done = true; }
          rewards.push(rew);
          total += rew;
          if (done) {
            episodes++;
            if (isGoal(c, r) && (bestSteps === null || steps < bestSteps)) bestSteps = steps;
          }
        }
        function reset() { pos = { ...START }; trail = [{ ...START }]; rewards = []; total = 0; steps = 0; done = false; }

        const up = ctx.button('↑', () => move(0, -1));
        const down = ctx.button('↓', () => move(0, 1));
        const left = ctx.button('←', () => move(-1, 0));
        const right = ctx.button('→', () => move(1, 0));
        const again = ctx.button('Try again', reset, 'primary');
        const revealBtn = ctx.button('Reveal the map', () => { revealed = !revealed; revealBtn.textContent = revealed ? 'Hide the map' : 'Reveal the map'; });
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          const CS = 52, GX = 30, GY = 52;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText(revealed ? 'the world, revealed' : 'the world (you cannot see it)', GX, 34);

          for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
            const x = GX + c * CS, y = GY + r * CS;
            const visited = trail.some(t => t.c === c && t.r === r);
            g.fillStyle = revealed
              ? (isGoal(c, r) ? 'rgba(56,217,169,0.35)' : isPit(c, r) ? 'rgba(251,113,133,0.35)' : '#131a27')
              : (visited ? '#1b2536' : '#0f1420');
            g.fillRect(x, y, CS - 2, CS - 2);
            g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(x, y, CS - 2, CS - 2);
            if (revealed && isGoal(c, r)) { g.font = '20px Inter, system-ui, sans-serif'; g.fillStyle = C.green; g.fillText('+10', x + 6, y + 32); }
            if (revealed && isPit(c, r)) { g.font = '20px Inter, system-ui, sans-serif'; g.fillStyle = C.danger; g.fillText('−10', x + 6, y + 32); }
          }
          /* the path you actually took */
          g.strokeStyle = 'rgba(124,156,255,0.55)'; g.lineWidth = 2;
          g.beginPath();
          trail.forEach((t, i) => {
            const x = GX + t.c * CS + CS / 2 - 1, y = GY + t.r * CS + CS / 2 - 1;
            i ? g.lineTo(x, y) : g.moveTo(x, y);
          });
          g.stroke();
          g.fillStyle = done ? (isGoal(pos.c, pos.r) ? C.green : C.danger) : C.accent;
          g.beginPath(); g.arc(GX + pos.c * CS + CS / 2 - 1, GY + pos.r * CS + CS / 2 - 1, 11, 0, 7); g.fill();

          /* the only information you actually receive */
          const TX = 370;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('everything you are told', TX, 34);
          g.font = MONO; g.fillStyle = C.muted;
          if (!rewards.length) {
            wrapText(g, 'Nothing yet. Press an arrow. You will get one number back — and that is all you will ever get.', TX, 58, 300, 16);
          } else {
            const show = rewards.slice(-9);
            show.forEach((rw, i) => {
              g.font = MONO;
              g.fillStyle = rw > 0 ? C.green : rw < -1 ? C.danger : C.muted;
              g.fillText('step ' + (steps - show.length + i + 1) + ':  reward ' + (rw > 0 ? '+' : '') + rw.toFixed(1), TX, 58 + i * 18);
            });
          }
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('total this attempt', TX, 236);
          g.font = 'bold 22px Inter, system-ui, sans-serif';
          g.fillStyle = total > 0 ? C.green : total < -1 ? C.danger : C.muted;
          g.fillText(total.toFixed(1), TX + 160, 238);

          g.font = FONT;
          if (done && isGoal(pos.c, pos.r)) {
            g.fillStyle = C.green;
            wrapText(g, 'Found it in ' + steps + ' steps. Now: which single move was the best one you made? You genuinely cannot tell — the reward only ever came at the end.', GX, 320, 640, 16);
          } else if (done) {
            g.fillStyle = C.danger;
            wrapText(g, 'That square cost you 10. Nothing warned you, and nothing told you which earlier move set you on course for it.', GX, 320, 640, 16);
          } else {
            g.fillStyle = C.muted;
            wrapText(g, 'No map, no labels, no teacher. Only a number after each move — and mostly the same small negative number.', GX, 320, 640, 16);
          }
          ro.set({ steps, total: total.toFixed(1), attempts: episodes, 'best so far': bestSteps === null ? '—' : bestSteps + ' steps' });
        });

        return ctx.figure(cv,
          'You are the agent. The grid is the environment, your position is the state, an arrow press is an action, and the number that comes back is the reward. Notice what you are never given: which move was the mistake. That gap between "how it went" and "what you should have done instead" is the defining difficulty of reinforcement learning, and every algorithm in this chapter is an answer to it.',
          [up, down, left, right, again, revealBtn], ro);
      }

      /* ================================================================== */
      /*  INTERACTIVE — HOW FAR AHEAD THE AGENT BOTHERS TO LOOK              */
      /* ================================================================== */
      function discountLab() {
        const [cv, g] = ctx.canvas(720, 320);
        let gamma = 0.9;
        /* two options down one corridor: a small reward close by, a big one far away */
        const NEAR = { steps: 2, reward: 3 };
        const FAR = { steps: 12, reward: 20 };
        const value = (o, gm) => o.reward * Math.pow(gm, o.steps);

        const gSl = ctx.slider({ label: 'γ discount factor', min: 0.5, max: 0.995, step: 0.005, value: 0.9, digits: 3, onChange: (v) => { gamma = v; } });
        const impatient = ctx.button('impatient (0.50)', () => { gamma = 0.5; gSl.value = 0.5; });
        const typical = ctx.button('typical (0.90)', () => { gamma = 0.9; gSl.value = 0.9; }, 'primary');
        const patient = ctx.button('far-sighted (0.99)', () => { gamma = 0.99; gSl.value = 0.99; });
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          const vN = value(NEAR, gamma), vF = value(FAR, gamma);
          const picksFar = vF > vN;

          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('two rewards down the same corridor', 34, 28);
          const X0 = 50, CS = 44, Y = 56;
          for (let i = 0; i <= 13; i++) {
            const x = X0 + i * CS;
            const isN = i === NEAR.steps, isF = i === FAR.steps;
            g.fillStyle = isN ? 'rgba(251,191,36,0.3)' : isF ? 'rgba(56,217,169,0.3)' : '#131a27';
            g.fillRect(x, Y, CS - 3, 40);
            g.strokeStyle = C.line; g.strokeRect(x, Y, CS - 3, 40);
            if (isN) { g.font = 'bold ' + MONO; g.fillStyle = C.warn; g.fillText('+3', x + 10, Y + 25); }
            if (isF) { g.font = 'bold ' + MONO; g.fillStyle = C.green; g.fillText('+20', x + 6, Y + 25); }
            if (i === 0) { g.fillStyle = C.accent; g.beginPath(); g.arc(x + CS / 2 - 1, Y + 20, 10, 0, 7); g.fill(); }
          }
          g.font = MONO; g.fillStyle = C.muted;
          // "you are here" goes above the corridor: on the label row below it would run straight
          // into the "2 steps" marker, which is only two cells along.
          g.fillText('you are here', X0 - 6, Y - 8);
          g.fillText('2 steps', X0 + NEAR.steps * CS - 8, Y + 58);
          g.fillText('12 steps', X0 + FAR.steps * CS - 10, Y + 58);

          /* the discount curve */
          const P = { x: 50, y: 150, w: 340, h: 120 };
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
          g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
          g.strokeStyle = C.accent; g.lineWidth = 2; g.beginPath();
          for (let i = 0; i <= 20; i++) {
            // inset by half the stroke so the γ⁰ = 1 end is not sliced by the clip
            const x = P.x + i / 20 * P.w, y = P.y + P.h - 1 - Math.pow(gamma, i) * (P.h - 2);
            i ? g.lineTo(x, y) : g.moveTo(x, y);
          }
          g.stroke();
          g.restore();
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('what a reward n steps away is worth today', P.x, P.y - 8);
          g.fillText('0', P.x - 14, P.y + P.h + 4); g.fillText('1', P.x - 14, P.y + 6);
          g.fillText('20 steps', P.x + P.w - 44, P.y + P.h + 18);

          /* the comparison */
          const TX = 430;
          // at γ = 0.5 the far reward is worth 0.005, so two decimals would print a bare "0.00"
          // and lose the whole point of the demo — small values get three.
          const fmtV = (v) => (v >= 0.1 ? v.toFixed(2) : v.toFixed(3));
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('what each is worth right now', TX, 150);
          const bar = (lab, v, col, y, detail) => {
            g.font = MONO; g.fillStyle = C.muted; g.fillText(lab, TX, y);
            g.fillStyle = C.line; g.fillRect(TX, y + 8, 180, 16);
            g.fillStyle = col; g.fillRect(TX, y + 8, ctx.clamp(v / 20, 0, 1) * 180, 16);
            g.font = 'bold 15px Inter, system-ui, sans-serif'; g.fillStyle = col;
            g.fillText(fmtV(v), TX + 190, y + 21);
            g.font = '11px "JetBrains Mono", ui-monospace, monospace'; g.fillStyle = C.muted;
            g.fillText(detail, TX, y + 38);
          };
          bar('near (+3, 2 steps)', vN, C.warn, 172, '3 × ' + gamma.toFixed(3) + '² = ' + fmtV(vN));
          bar('far (+20, 12 steps)', vF, C.green, 228, '20 × ' + gamma.toFixed(3) + '¹² = ' + fmtV(vF));

          g.font = 'bold 15px Inter, system-ui, sans-serif';
          g.fillStyle = picksFar ? C.green : C.warn;
          g.fillText(picksFar ? 'the agent walks past the +3' : 'the agent grabs the +3 and stops', 34, 300);
          ro.set({ γ: gamma.toFixed(3), 'near worth': fmtV(vN), 'far worth': fmtV(vF), chooses: picksFar ? 'the far +20' : 'the near +3' });
        });

        return ctx.figure(cv,
          'γ is the single number that decides how far ahead an agent bothers to look. A reward n steps away is worth γⁿ times its face value today, so the curve above is how quickly the future stops mattering. Drag γ down to 0.50 and the distant +20 is worth less than a quarter of a point — the agent takes the +3 and never sees the bigger prize. Push it to 0.99 and the +20 dominates easily. Nothing about the world changed; only the agent\'s patience did.',
          [gSl, impatient, typical, patient], ro);
      }

      /* ================================================================== */
      /*  INTERACTIVE — WHY PPO CLIPS THE UPDATE                             */
      /* ================================================================== */
      function ppoClip() {
        const [cv, g] = ctx.canvas(720, 330);
        let stepSize = 0.6, clipping = true, seed = 3;
        const CLIP = 0.2;
        /* A policy has one parameter; performance peaks at theta = 0. Each update is estimated
           from data the CURRENT policy generated, so the further the policy moves in one step,
           the more wrong that estimate becomes — which is exactly what the clip bounds. */
        const perf = (th) => Math.exp(-th * th / 2);
        function run() {
          let a = seed;
          const rnd = () => { a = (a * 1664525 + 1013904223) % 4294967296; return a / 4294967296; };
          let th = -2.2; const hist = [th];
          for (let i = 0; i < 40; i++) {
            const grad = -th * perf(th);                       // toward the peak
            const noise = (rnd() - 0.5) * 1.4;                 // the data is a small sample
            let ratio = stepSize * (grad + noise);
            if (clipping) ratio = ctx.clamp(ratio, -CLIP, CLIP);
            th += ratio;
            /* off-policy damage: a huge move invalidates the data that produced it */
            if (Math.abs(ratio) > 0.55) th += (rnd() - 0.5) * 4.5;
            th = ctx.clamp(th, -6, 6);
            hist.push(th);
          }
          return hist;
        }
        const sSl = ctx.slider({ label: 'update step size', min: 0.1, max: 2.5, step: 0.05, value: 0.6, digits: 2, onChange: (v) => { stepSize = v; } });
        const cBtn = ctx.button('clipping: on', () => {
          clipping = !clipping;
          cBtn.textContent = 'clipping: ' + (clipping ? 'on' : 'off');
        }, 'primary');
        const reseed = ctx.button('New run', () => { seed = (seed * 31 + 7) % 99991; });
        const ro = ctx.readout();

        ctx.loop(() => {
          g.clearRect(0, 0, cv.W, cv.H);
          const hist = run();
          const P = { x: 55, y: 50, w: 400, h: 210 };
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('how well the policy performs, update by update', P.x, 30);
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
          // DOT is the collapse-marker radius: inset the data range by it on every side so a
          // point sitting exactly on a limit is not sliced in half by the clip below.
          // HEAD leaves a band above the "best possible" line for its label, which the curve
          // (performance never exceeds 1) can therefore never reach.
          const DOT = 4, HEAD = 22;
          const px = (i) => P.x + DOT + i / (hist.length - 1) * (P.w - 2 * DOT);
          const py = (v) => P.y + P.h - DOT - ctx.clamp(v, 0, 1) * (P.h - DOT - HEAD);
          g.setLineDash([3, 3]); g.strokeStyle = 'rgba(148,163,184,0.4)';
          g.beginPath(); g.moveTo(P.x, py(1)); g.lineTo(P.x + P.w, py(1)); g.stroke();
          g.setLineDash([]);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('best possible', P.x + 4, py(1) - 8);
          const scores = hist.map(perf);
          g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
          g.strokeStyle = clipping ? C.green : C.danger; g.lineWidth = 2.5;
          g.beginPath();
          scores.forEach((v, i) => { i ? g.lineTo(px(i), py(v)) : g.moveTo(px(i), py(v)); });
          g.stroke();
          scores.forEach((v, i) => {
            if (i && Math.abs(hist[i] - hist[i - 1]) > 0.55) {
              g.fillStyle = C.danger;
              g.beginPath(); g.arc(px(i), py(v), DOT, 0, 7); g.fill();
            }
          });
          g.restore();
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('policy updates →', P.x + 150, P.y + P.h + 20);

          const TX = 490;
          const final = scores[scores.length - 1];
          const worst = Math.min(...scores.slice(5));
          const collapses = scores.slice(1).filter((v, i) => Math.abs(hist[i + 1] - hist[i]) > 0.55).length;
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('the clip band', TX, 50);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('moves bigger than ±' + CLIP.toFixed(1), TX, 72);
          g.fillText('are cut back to ±' + CLIP.toFixed(1) + '.', TX, 88);
          g.font = 'bold ' + FONT; g.fillStyle = C.text;
          g.fillText('final performance', TX, 128);
          g.font = 'bold 22px Inter, system-ui, sans-serif';
          g.fillStyle = final > 0.85 ? C.green : final > 0.4 ? C.warn : C.danger;
          g.fillText((final * 100).toFixed(0) + '%', TX, 156);
          g.font = MONO; g.fillStyle = C.muted;
          g.fillText('worst dip: ' + (worst * 100).toFixed(0) + '%', TX, 178);
          g.fillStyle = collapses ? C.danger : C.green;
          g.fillText('collapses: ' + collapses, TX, 196);
          g.font = FONT; g.fillStyle = C.muted;
          wrapText(g, collapses
            ? 'Each red dot is an update so large that the data which produced it no longer describes the policy you now have. Performance does not just stall — it falls off a cliff.'
            : 'Every update stays inside the band, so the data stays roughly valid and the curve climbs instead of lurching.',
            TX, 220, 200, 16);
          ro.set({ 'step size': stepSize.toFixed(2), clipping: clipping ? 'on' : 'off', final: (final * 100).toFixed(0) + '%', collapses });
        });

        return ctx.figure(cv,
          'A deliberately simple picture of the problem PPO solves. In supervised learning a too-large step just overshoots; in RL the model <i>chooses its own next training set</i>, so a policy that moves too far no longer resembles the one that gathered the data, and that data becomes worthless. Turn clipping off and push the step size past about 1.5: performance does not degrade gracefully, it collapses. Clipping bounds each update to roughly ±20%, which is almost all of what PPO does — and why it is the algorithm that trained ChatGPT.',
          [sSl, cBtn, reseed], ro);
      }
    },
  });
})();
