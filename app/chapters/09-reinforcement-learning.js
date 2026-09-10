/* Chapter 09 — Learning by doing: reinforcement learning */
(function () {
  ZTA.registerChapter({
    id: '09-reinforcement-learning',
    num: 9,
    part: 'II',
    title: 'Learning by doing: reinforcement learning',
    tagline: 'No answer key, only consequences: how a program learns to win games, walk, and (later) be a helpful assistant.',
    render(root, ctx) {
      const { h, p, section, sub, callout, ul } = ctx;
      const C = ctx.colors;

      /* ------------------------------------------------------------------ hook */
      root.append(
        p('In March 2016, in a hotel in Seoul, a program called AlphaGo played move 37 of its second game against Lee Sedol, one of the strongest Go players alive. Commentators thought it was a mistake. Professional players estimated a human would play it about one time in ten thousand. It won the game. Nobody had shown AlphaGo that move. There was no labelled dataset of "brilliant moves", no teacher marking its homework. It had discovered the move by <b>playing millions of games against itself and noticing what led to winning</b>.'),
        p('Everything in this course so far was <em>supervised</em>: you show the model an input and the correct output, it measures its error, it nudges its weights. But most of life has no correct output. When you learn to ride a bike, nobody hands you the right handlebar angle for each millisecond. You wobble, you fall, you adjust. The only signal is <b>how it went</b>.'),
        p('This chapter is about learning from consequences: <em>reinforcement learning</em>, or RL. It matters for its own sake (games, robots, data-centre cooling), and it matters because it is the final ingredient that turns a text predictor into an assistant. Keep that in mind: everything here will come back in chapter 11.'),
      );

      /* ------------------------------------------------------------------ mechanism */
      root.append(section('The vocabulary: five words and a loop',
        p('RL is a loop between two things. The <em>agent</em> is the learner: the program choosing what to do. The <em>environment</em> is everything else: the game, the robot\'s physics, the maze. At each tick, the environment shows the agent a <em>state</em> (where am I? what does the board look like?). The agent picks an <em>action</em>. The environment responds with a new state and a number: the <em>reward</em>. Then it repeats.'),
        p('The reward is the whole training signal. It is often zero for a long time and then suddenly not: +1 for winning a game after 200 moves, −1 for losing, 0 for every move in between. The agent\'s job is to choose actions that make the <b>total future reward</b> as large as possible. Not the next reward: the total, including the delayed ones.'),
        p('Two more words. A <em>policy</em> is the agent\'s strategy: a rule (in practice a neural network, or a table) that maps a state to an action, or to probabilities over actions. A <em>value</em> is a prediction: "starting from this state, how much reward am I going to collect from here on, if I follow my policy?" A chess player who looks at a position and says "white is winning" is stating a value. Learning good values is most of what RL does, because once you know the value of every state you can just step toward the most valuable one.'),
        h('div', { class: 'grid-2' },
          h('div', { class: 'card' }, h('h4', {}, 'Supervised learning'), h('p', { html: 'Input → correct output. The loss tells you <b>how wrong</b> and <b>in which direction</b> for every example. Feedback is instant and precise.' })),
          h('div', { class: 'card' }, h('h4', {}, 'Reinforcement learning'), h('p', { html: 'State → action → reward, repeat. The reward tells you <b>how it went</b>, often much later, and never tells you what you should have done instead.' })),
        ),
      ));

      root.append(section('Three problems that make RL hard',
        sub('1. Credit assignment: which of the 40 moves lost the game?',
          p('You play a game of chess, you lose at move 40. The reward is −1. Which move was the mistake? Maybe move 12. Maybe the loss was already baked in at move 3, and the next 37 moves were fine. The reward arrives at the end, a single number, and it has to be spread back over every decision that led to it. This is the <em>credit-assignment problem</em>, and it is the reason RL needs more than gradient descent on a loss: there is no per-step loss to descend.'),
        ),
        sub('2. Exploration versus exploitation: the restaurant problem',
          p('You have a favourite restaurant. It is good. There are twenty others on the street you have never tried. Every night you face the choice: <em>exploit</em> what you know (go to the favourite, get a reliable 8/10) or <em>explore</em> (try somewhere new, get a 4 or a 10). If you never explore, you never find the best restaurant. If you always explore, you eat a lot of bad dinners. An RL agent faces this every step, and the simplest fix is embarrassingly crude: with some small probability <em>ε</em> (epsilon, say 10%), ignore what you know and act at random. This is called <em>ε-greedy</em>. Smarter schemes give a bonus to actions you are <b>uncertain</b> about; you will meet one below.'),
        ),
        sub('3. Delayed reward and discounting',
          p('A reward of 10 now is worth more than a reward of 10 in fifty steps. Partly because the world is uncertain, partly because we want the agent to prefer quick wins. So RL multiplies future rewards by a <em>discount factor</em> γ (gamma) per step, typically 0.9 to 0.99. With γ = 0.9, a reward of 10 five steps away is worth 10 × 0.9<sup>5</sup> ≈ 5.9 today. With γ = 0.99 the agent effectively plans about a hundred steps ahead; with γ = 0.5 it is impatient and barely sees past the next few moves.'),
        ),
      ));

      root.append(section('Q-learning: a table of "how good is this action here?"',
        p('Here is the simplest RL algorithm that really works, from 1989. Keep a table, <em>Q</em>, with one row per state and one column per action. Q(s, a) is your current estimate of the total discounted reward you will get if you take action a in state s and behave sensibly afterwards. Start with all zeros. Then, every time you take a step and observe (state s, action a, reward r, new state s\'), update one cell:'),
        ctx.code('Q(s, a)  ←  Q(s, a)  +  α · [ r  +  γ · max_a\' Q(s\', a\')  −  Q(s, a) ]\n\n   α (alpha)   learning rate: how far to move toward the new estimate\n   γ (gamma)   discount factor\n   r + γ·max Q(s\',·)   the "target": reward now, plus the best you think you can do next'),
        p('Read the bracket as an <b>error</b>: what you just learned the value should be (reward now plus the best next value), minus what you previously believed. You move your belief part of the way toward the evidence. That is it. Because the target uses the <i>next</i> state\'s value, information flows backwards one step per update: the cell next to the goal learns first, then the cell next to that, and so on. This is how credit assignment gets solved: not all at once, but by a chain of one-step corrections repeated thousands of times.'),
        callout('example', 'A worked update',
          'Say the agent is one step from a goal it has already discovered. α = 0.5, γ = 0.9. Currently Q(s, right) = 0. It steps right, pays the step cost r = −0.1, and lands in a state whose best action has value 5.<br>' +
          'Target = −0.1 + 0.9 × 5 = <b>4.4</b>. Error = 4.4 − 0 = 4.4. New Q(s, right) = 0 + 0.5 × 4.4 = <b>2.2</b>.<br>' +
          'Do it again from the same place: target is still 4.4, error is 4.4 − 2.2 = 2.2, new value 3.3. Then 3.85, 4.12, … converging on 4.4. Every cell in the grid below is doing exactly this.'),
        p('Written out as code it is about ten lines, and the interactive below runs exactly this — no shortcuts, no pre-computed answers:'),
        ctx.code('# tabular Q-learning: the entire algorithm\nQ = zeros(n_states, n_actions)          # start believing nothing\n\nfor episode in range(10_000):\n    s, done = env.reset(), False\n    while not done:\n        # ε-greedy: roll a die, sometimes ignore what you know\n        a = randint(n_actions) if random() < eps else argmax(Q[s])\n\n        s2, r, done = env.step(a)\n\n        # the target: reward now + the best you believe you can do next\n        target = r if done else r + gamma * max(Q[s2])\n\n        Q[s][a] += alpha * (target - Q[s][a])   # move a fraction α toward it\n        s = s2'),
        p('Two details in that code earn their place. <code class="inline">target = r if done</code>: when the episode ends there is no "next state", so the target is just the reward — this is the only place real reward enters the table, and every other value in the grid is ultimately a rumour about it. And <code class="inline">argmax(Q[s])</code> inside the loop while the target uses <code class="inline">max(Q[s2])</code> regardless of what was actually done next: Q-learning learns the value of the <b>best</b> policy even while behaving randomly. That property has a name, <em>off-policy</em> learning, and it is why you can crank exploration to 100% below and the table still converges on a good policy.'),
        callout('key', 'Where you start the table is itself an exploration strategy',
          'The demo below starts every Q value at <b>0</b>, and every step costs −0.1. Put those two facts together: the moment the agent tries an action, that action\'s value is pushed <i>below</i> zero, while every action it has not tried is still sitting at zero. So "take the highest-valued action" quietly means "take the one you know least about". A purely greedy agent — ε = 0 — sweeps the whole grid on its own.<br><br>' +
          'This is <em>optimistic initialisation</em>, and there is an exact line where it stops working. Bumping into a wall returns you to the same cell, so that action\'s value converges on q = −0.1 + γq, which for γ = 0.9 is <b>−1</b>. Start the table above −1 and tried actions fall below untried ones: the agent explores. Start it below −1 and tried actions <i>rise</i> toward −1, above their untried neighbours, and a greedy agent locks onto the first thing it did and repeats it forever.<br><br>' +
          'The demo has a <b>Q₀</b> slider so you can cross that line yourself. It is the same idea you will meet again in the bandit section as UCB\'s optimism bonus: <b>make the unknown look attractive, and exploration takes care of itself.</b>'),
        p('Q-learning is <em>tabular</em> when the table is literal, which only works for small worlds. For Atari or Go the "table" becomes a neural network that takes the state and outputs a Q value per action; the update rule becomes the loss it is trained on. That is <em>deep Q-learning</em>, and we will get there. First, watch the table version learn.'),
      ));

      /* ------------------------------------------------------------------ interactive A: gridworld */
      root.append(callout('tryit', 'Try it: a Q-learning agent learns a maze in front of you',
        'Press <b>Play</b>. The dot is the agent. Every cell has four triangles, one per action (up/right/down/left), coloured by its Q value: green is good, red is bad, grey is "never tried from here". The small number is that cell\'s best value, max<sub>a</sub> Q(s, a). Watch the green spread <b>backwards from the goal</b>, one cell per episode at first, and watch the return chart climb from −30 to about +8.9 — which is the best return this maze allows: the shortest path is twelve moves, so eleven of them cost −0.1 and the twelfth lands on the +10.<br>' +
        'Then, in order:<br>' +
        '<b>(1) Set ε to 0.</b> The agent still solves the maze — faster, in fact. That is not a bug, and the reason is the next control.<br>' +
        '<b>(2) Now drag Q₀ down to −2</b> (this wipes the table) and leave ε at 0. The agent picks one action and repeats it forever: the return line flatlines at −30 and the grid stays grey. <b>Wind ε back up to 0.3</b> and watch it break out. This is the exploration failure, and you can only see it once the starting values are pessimistic — see the callout above for why.<br>' +
        '<b>(3) Put Q₀ back to 0 and set γ to 0.5.</b> The values collapse towards zero the further you get from the goal — the start cell is worth roughly 0.5<sup>11</sup> × 10 ≈ 0.005 — yet the arrows still point the right way. Discounting changes what the numbers <i>mean</i>, not necessarily what the policy <i>does</i>.<br>' +
        '<b>(4) Switch on Edit map</b> and drag the goal somewhere else <i>without resetting</i>. The old values die and new ones grow: you are watching the agent unlearn.<br>' +
        '<b>(5) Crank α to 1.</b> Each update throws away the old estimate entirely, so the values thrash with every unlucky episode.'));
      root.append(gridworld());

      /* ------------------------------------------------------------------ interactive B: bandit */
      root.append(section('Exploration, measured: the multi-armed bandit',
        p('Strip RL down to its smallest possible problem: no states, no delayed reward, just five slot machines with different, hidden payout rates. Each pull pays £1 with some probability. You get 1000 pulls. How do you find the best machine without wasting too many pulls on the bad ones? This is the <em>multi-armed bandit</em>, and it is the exploration–exploitation dilemma with everything else removed. Casinos are the origin of the name; websites choosing which headline to show you are the modern use.'),
        p('The score that matters is <em>regret</em>: how much you earned, compared with what you would have earned by pulling the best machine every time. A perfect agent has zero regret. A random agent\'s regret grows in a straight line. A good agent\'s regret curve <b>bends flat</b>: it explores early, identifies the winner, and stops paying for information it no longer needs. Watch the <i>shape</i> of the curve, not just where it ends — the shape is what tells you whether an agent is still paying for exploration it no longer needs.'),
        p('Two classic strategies. <em>ε-greedy</em> you already know. <em>UCB</em> (upper confidence bound) is cleverer: it scores each machine by its average payout <b>plus a bonus that grows with how rarely you have tried it</b>. Machines you have barely touched look optimistic, so you try them; as evidence piles up the bonus shrinks and the winner takes over. "Optimism in the face of uncertainty" is the slogan, and it is a good one for life too.'),
        ctx.code('score(k)  =  mean payout of k   +   sqrt( 2 · ln(t) / n(k) )\n              └─────┬─────┘       └────────┬────────┘\n                    │                      │\n         what you have actually      optimism bonus: large when you have\n         seen from machine k         pulled k rarely, shrinking like 1/√n\n                                     as evidence arrives\n\n   t     = total pulls so far        pull the machine with the highest score'),
        p('Notice the asymmetry that makes UCB work. The bonus falls as 1/√n(k), so a machine you have pulled ten times still carries a visible bonus, while one pulled a thousand times carries almost none. Meanwhile ln(t) creeps upward forever, so a machine you have ignored for a long time slowly becomes tempting again. The result is an agent that explores hard early, tapers off on its own schedule, and never needs an ε knob at all — one fewer number for a human to guess.'),
        callout('key', 'The honest comparison (and why the shape matters more than the score)',
          'Run the demo below at its default 1,000 pulls and <b>ε-greedy usually wins</b>: with ε = 0.1 it averages about 47 regret against UCB\'s 69. That is not a bug in UCB, and the demo is not rigged. It is the real result, and the reason is worth more than the scoreboard.<br><br>' +
          'ε-greedy spends a fixed <b>fraction</b> of every pull on exploration, forever. Once it has found the best machine, that spending is pure waste — but it is cheap waste, and over a short run the simplicity pays. Its regret therefore grows in a <b>straight line</b> with slope ε × (average gap), no matter how certain it becomes.<br><br>' +
          'UCB pays more up front to measure every machine properly, then tapers: its regret grows like <b>log t</b>. So the curves cross. Around 4,000 pulls in this setup they meet; by 20,000 ε-greedy has spent about 546 and UCB about 239, and the gap keeps widening. <b>Raise the horizon in the demo to 20k and watch it happen.</b><br><br>' +
          'And the knob is the real point: ε = 0.1 was <i>tuned</i>, by us, knowing the answer. Slide ε to 0.5 or 0.02 and ε-greedy falls apart. UCB has nothing to tune. In RL you almost never get to tune the exploration rate against the true answer, which is why "optimism in the face of uncertainty" beats "roll a die" everywhere that matters.'),
      ));
      root.append(callout('tryit', 'Try it: beat the algorithms',
        'Click machines to pull them yourself. Try to identify the best one in as few pulls as you can — then press <b>Reveal</b> and see how close you were.<br>' +
        'Then press <b>Run agents</b> and watch ε-greedy and UCB play the same machines. Things to do, in order:<br>' +
        '<b>(1)</b> At the default 1,000 pulls, note who ends lower — usually ε-greedy — then look at the <i>curves</i>: UCB\'s is already bending, ε-greedy\'s is a straight line.<br>' +
        '<b>(2)</b> Set the horizon to <b>20,000 pulls</b> and run again. The straight line keeps climbing at the same rate and UCB\'s flattening curve passes underneath it. That crossover is the whole argument.<br>' +
        '<b>(3)</b> Slide ε to <b>0.5</b> and run: ε-greedy now wastes half of every pull forever and its line steepens to nearly the random baseline. Slide ε to <b>0</b>: it often locks onto a mediocre machine on the strength of one lucky payout and never looks again. UCB\'s curve does not move in either case — it has no ε.<br>' +
        '<b>(4)</b> Press <b>New machines</b> a few times. Notice that when two machines are close in true rate, everyone\'s regret is small (it barely matters which you pick) and when one machine is far ahead, the bad strategies are punished hardest.'));
      root.append(bandit());

      /* ------------------------------------------------------------------ policy gradients */
      root.append(section('The other family: policy gradients',
        p('Q-learning learns values and derives a policy from them. The other big family skips the middleman and learns the <em>policy</em> directly: a network that outputs a probability for each action. The training rule, called <em>REINFORCE</em> or the <em>policy gradient</em>, is one sentence long: <b>run the policy, and for every action you took, nudge its probability up in proportion to the reward that followed it (and down if the reward was bad)</b>. Actions in winning games become more likely; actions in losing games less likely. Over millions of games the noise averages out and only the genuinely good habits survive.'),
        callout('example', 'A worked policy-gradient step',
          'Two actions, left and right. The network outputs two scores (<i>logits</i>), and a softmax turns them into probabilities. Both logits start at 0, so the policy is a coin flip: 50/50. The agent samples <b>right</b> and collects reward +1.<br>' +
          'The gradient of log π(right) with respect to the logits is <code class="inline">onehot(right) − π</code> = (0, 1) − (0.5, 0.5) = <b>(−0.5, +0.5)</b>. Multiply by the reward (+1) and a learning rate of 0.5, add it on, and the logits become (−0.25, +0.25).<br>' +
          'The new probability of going right is 1 / (1 + e<sup>−0.5</sup>) = <b>0.62</b>. One good outcome moved it from 50% to 62%. A reward of −1 would have moved it to 38% by exactly the same arithmetic. That is the whole algorithm: <b>the reward is a multiplier on the gradient of your own log-probability.</b>'),
        p('Run that over a full episode instead of one action and you get REINFORCE:'),
        ctx.code('# REINFORCE with a baseline\nfor episode in range(N):\n    states, actions, rewards = rollout(policy)     # play the game once\n\n    G, returns = 0, []                             # discounted return from each step onward\n    for r in reversed(rewards):\n        G = r + gamma * G\n        returns.insert(0, G)\n\n    # "was this better than I expected?" — value() is a second network\n    advantages = [G - value(s) for s, G in zip(states, returns)]\n\n    loss = -sum(log_prob(policy, s, a) * A\n                for s, a, A in zip(states, actions, advantages))\n    loss.backward(); optimizer.step()'),
        p('Two refinements turn that into something you can train at scale. The first is the <code class="inline">- value(s)</code> in the advantage line: a second network predicts how well the agent expected to do, and the policy is nudged by the <b>surprise</b> rather than the raw return. If every game scores between 90 and 100, raw returns say "everything you did was great"; the surprise says which games were the 97s. This is the <em>baseline</em>, the difference is the <em>advantage</em>, and an algorithm with both a policy and a value network is called <em>actor–critic</em>.'),
        p('The second is a limit on how far one update may move the policy. Take too big a step and the policy that generated your data no longer resembles the policy you now have, the data becomes worthless, and performance collapses in a way it never does in supervised learning — because in RL the model chooses its own next training set. <em>TRPO</em> (2015) enforced this with a hard constraint on the KL divergence between the old and new policy. <em>PPO</em> (2017) got almost the same effect far more cheaply by clipping the update whenever the action probabilities move more than about 20%. PPO is the algorithm that was later used to train ChatGPT, so remember the name — and remember the reason for the clip, because it comes back in chapter 11.'),
      ));

      /* ------------------------------------------------------------------ deep RL history */
      root.append(section('Deep RL: from Atari to Go',
        callout('history', 'The decade RL went from toy mazes to superhuman',
          '<b>2013–2015, DQN.</b> DeepMind trains a convolutional network to play 49 Atari games from raw pixels, using Q-learning with two tricks that made deep RL stable: <i>experience replay</i> (store past steps, train on random batches of them) and a slowly-updated <i>target network</i>. The arXiv paper appeared in December 2013; the <i>Nature</i> paper in February 2015. Same network, same hyperparameters, every game.<br><br>' +
          '<b>March 2016, AlphaGo</b> beats Lee Sedol 4–1. Go has ~10<sup>170</sup> positions; brute force is hopeless. AlphaGo combined a policy network (first trained on human games, then improved by <i>self-play</i> with policy gradients), a value network, and <i>Monte Carlo tree search</i> (MCTS): look ahead by playing out promising lines, guided by the networks.<br><br>' +
          '<b>December 2017, AlphaZero</b> drops the human games entirely. Starting from random play, it learns Go, chess and shogi from self-play alone, beating the strongest existing programs in each. The same recipe, three games. This was the moment "learn from scratch by playing yourself" stopped sounding like science fiction.<br><br>' +
          '<b>2018–2019, OpenAI Five and AlphaStar.</b> Real-time strategy games with hidden information, long horizons and enormous action spaces: Dota 2 (OpenAI Five, beat the world champions in April 2019, trained with PPO at colossal scale) and StarCraft II (AlphaStar, Grandmaster level, late 2019).<br><br>' +
          '<b>Robotics.</b> The same algorithms learned to make a robot hand solve a Rubik\'s cube (2019) and, more usefully, taught quadrupeds to walk over rough ground. The catch is that real robots are slow and break, so almost all training happens in simulation, which brings its own problem: see <i>sim-to-real</i> below.'),
        callout('example', 'Where RL is quietly running today',
          '<b>The thumbnail you just clicked.</b> Streaming services and news sites pick artwork and headlines with contextual bandits — the exact machinery of the demo above, one bandit per viewer, learning in hours instead of the weeks an A/B test would take. Netflix has published on choosing artwork this way.<br><br>' +
          '<b>Cooling bills.</b> DeepMind trained a controller on Google\'s data-centre sensor logs and reported cutting the energy used for cooling by up to 40% (2016); from 2018 the system was given direct control of the cooling plant, under human veto.<br><br>' +
          '<b>Chip layout.</b> Google published an RL method for chip floorplanning in <i>Nature</i> (2021) — the agent places memory blocks on the die as if playing a board game — and used its layouts in shipped TPU accelerators.<br><br>' +
          '<b>Fusion plasma.</b> DeepMind and EPFL trained a policy in simulation to hold a plasma in the shape they wanted inside a real tokamak, controlling nineteen magnetic coils at kilohertz rates (<i>Nature</i>, 2022). Sim-to-real, on a reactor.<br><br>' +
          '<b>The assistant you use.</b> Every major chat model has been through an RL stage. That is the last section of this chapter, and all of chapter 11.'),
      ));

      /* ------------------------------------------------------------------ reward hacking */
      root.append(section('Reward hacking: you get exactly what you asked for',
        p('In 2016 OpenAI trained an agent on a boat-racing game called CoastRunners. The obvious reward, "finish the race", was hard to learn from, so they used the game\'s own score, which mostly comes from hitting targets along the course. The agent found a lagoon where three targets respawned quickly, and drove in circles there forever: crashing, catching fire, going backwards, never finishing, while scoring 20% more than human players. It was not broken. It was <b>perfectly optimising the reward it was given</b>. The reward simply was not what its designers meant.'),
        p('This has a name from economics, <em>Goodhart\'s law</em>: when a measure becomes a target, it stops being a good measure. Every proxy has gaps between what it counts and what you want, and a strong optimiser will find those gaps and pour itself into them. Agents have learned to pause Tetris forever to avoid losing, to exploit physics-engine bugs to "run" by vibrating, and to knock a lego brick over so that its "height" sensor reads high. Hold onto this idea. In chapter 11 the reward will be "a human preferred this answer", and the model will discover that humans have gaps too: they tend to prefer confident, long, flattering answers. That is <b>reward hacking with people as the environment</b>.'),
      ));
      root.append(callout('tryit', 'Try it: what the reward says versus what you meant',
        'A tiny boat race. Leave it on <b>laps finished</b> for about ten seconds and watch the score rate settle in the top right. Then switch the reward to <b>points from coins</b> — the boat abandons the course, parks in the lagoon where three coins respawn every 0.8 seconds, and spins. Give it ten seconds and compare the two rates: the behaviour that never finishes a single lap scores roughly 60% more. <b>Notice</b> that nothing about the boat got worse. Only the number we chose to reward changed.<br>' +
        '<b>Honesty note:</b> this demo is a scripted re-enactment of the CoastRunners result, not a trained agent; the point is the shape of the failure, which is the same one you get from a real learner.'));
      root.append(rewardHack());

      root.append(section('Sim-to-real',
        p('A robot arm in simulation can practise a million grasps overnight; a real one manages a few thousand a day and needs a human nearby to reset the blocks. So you train in sim. But simulated friction, lighting and motor lag are never quite right, and policies that exploit those inaccuracies fall apart on real hardware. The fix that works surprisingly well is <em>domain randomisation</em>: randomise everything in the simulator (colours, masses, friction, camera position, delays) so wildly that the real world looks like just one more variation. A policy that copes with a thousand fake worlds tends to cope with the real one. It is the same trick as data augmentation in chapter 4, applied to physics.'),
      ));

      /* ------------------------------------------------------------------ why it matters */
      root.append(section('Why this matters for modern AI',
        p('A language model fresh out of pretraining (chapter 10) is a supervised product: it predicts the next word of the internet. Turning it into an assistant is an RL problem. The <b>agent</b> is the model. The <b>environment</b> is a conversation. The <b>action</b> is the next token (or the whole reply, depending on how you draw the boxes). And the <b>reward</b> is a score from a model trained to imitate human preferences. That is <em>RLHF</em>, reinforcement learning from human feedback, and it is trained with PPO or one of its descendants. RLHF then adds a second rail of its own, on top of PPO\'s clip: a penalty on the KL divergence between the model being trained and the frozen model it started from. Without it the policy quickly discovers that a few weird, repetitive token sequences score very highly with the reward model, and it collapses into fluent-looking gibberish. The penalty is a leash back to English.'),
        p('The 2024–2025 generation of <em>reasoning models</em> (OpenAI\'s o1, DeepSeek-R1, Claude with extended thinking) pushed RL further. Instead of a preference score, they use rewards you can <b>verify</b>: did the maths answer match? did the code pass the unit tests? The environment is a coding task; the episode is a long chain of thought; the reward is 1 or 0 at the end. Credit assignment over thousands of tokens, exploration to find non-obvious solution strategies, reward hacking when the tests are weak: every problem in this chapter, at scale. Chapter 11 goes through it in detail.'),
        callout('key', 'Key idea',
          'Supervised learning needs someone to write down the right answer. RL only needs someone to <b>recognise a good outcome</b>, which is far easier and far more dangerous: the agent will optimise your definition of "good", not your intention.'),
      ));

      /* ------------------------------------------------------------------ quiz */
      root.append(ctx.quiz([
        { q: 'In the Q-learning update, what does the "target" r + γ·max Q(s′,·) represent?',
          options: ['The reward the agent received on this step only', 'The agent\'s new evidence for how valuable the action was: reward now plus the best it believes it can do from the next state', 'The probability of choosing the action again', 'The number of steps remaining in the episode'],
          answer: 1, explain: 'The target is a one-step-better estimate. The update moves the stored Q value a fraction α of the way toward it, which is how value information flows backwards from the goal.' },
        { q: 'In the gridworld the table starts at Q₀ = 0 and every step costs −0.1. Why does even a greedy agent (ε = 0) explore the whole maze?',
          options: ['Because ε = 0 still leaves a small chance of a random move', 'Because trying an action pushes its value below 0, while untried actions are still sitting at 0 — so "greedy" means "try something new"', 'Because the discount factor injects noise into the choice', 'Because the +10 goal reward is visible from every cell'],
          answer: 1, explain: 'This is <b>optimistic initialisation</b>: a starting value the world cannot live up to makes ignorance look attractive, and exploration falls out for free. Drag the demo\'s Q₀ slider below −1 and the effect reverses — tried actions then rise <i>above</i> untried ones, the ε = 0 agent locks onto its first move and repeats it forever, and only turning ε up rescues it.' },
        { q: 'With γ = 0.9, a reward of +10 that is 10 steps away is worth about how much today?',
          options: ['10', '9', '3.5', '0.1'],
          answer: 2, explain: '10 × 0.9^10 ≈ 10 × 0.349 ≈ 3.5. Discounting makes distant rewards count for less, which is why cells far from the goal have smaller values.' },
        { q: 'The CoastRunners boat drove in circles collecting respawning targets instead of finishing the race. What went wrong?',
          options: ['The neural network was too small', 'The agent was optimising the reward it was given (score), which did not match what its designers wanted (finishing)', 'The learning rate was too high', 'The game had a bug in its physics'],
          answer: 1, explain: 'Classic reward hacking / Goodhart\'s law. The optimiser was working perfectly; the proxy reward had a gap in it, and the agent found the gap.' },
        { q: 'When RL is used to train a chatbot (RLHF), what plays the role of the environment?',
          options: ['The GPU cluster', 'The conversation, with a reward model scoring the replies', 'The tokenizer', 'The training dataset from the internet'],
          answer: 1, explain: 'The model is the agent, its replies are actions, the conversation is the environment, and a reward model trained on human preferences provides the reward. Chapter 11 builds this.' },
      ]));

      /* ------------------------------------------------------------------ go deeper */
      root.append(section('Go deeper', ul([
        '<a href="http://incompleteideas.net/book/the-book-2nd.html" target="_blank" rel="noopener">Sutton &amp; Barto, <i>Reinforcement Learning: An Introduction</i> (2nd ed., free PDF)</a> — the textbook; chapters 1–6 cover everything in this chapter.',
        '<a href="https://www.davidsilver.uk/teaching/" target="_blank" rel="noopener">David Silver\'s UCL RL lecture series</a> — ten lectures by AlphaGo\'s lead researcher. Lecture 1 is the best hour-long introduction that exists.',
        '<a href="https://arxiv.org/abs/1312.5602" target="_blank" rel="noopener">Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)</a> and the <a href="https://www.nature.com/articles/nature14236" target="_blank" rel="noopener">2015 Nature paper</a> — DQN.',
        '<a href="https://openai.com/index/faulty-reward-functions/" target="_blank" rel="noopener">OpenAI, "Faulty reward functions in the wild" (2016)</a> — the CoastRunners boat, with video.',
        '<a href="https://lilianweng.github.io/posts/2018-02-19-rl-overview/" target="_blank" rel="noopener">Lilian Weng, "A (Long) Peek into Reinforcement Learning"</a> — a dense, accurate survey from tabular methods to PPO.',
        '<a href="https://spinningup.openai.com/en/latest/" target="_blank" rel="noopener">OpenAI Spinning Up in Deep RL</a> — the best hands-on introduction to policy gradients, with clean code.',
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
          const RLO = -40, RHI = 10, RSPAN = RHI - RLO;
          const retY = (v) => py + ph * (1 - (ctx.clamp(v, RLO, RHI) - RLO) / RSPAN);
          g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(px, py, pw, ph);
          g.fillStyle = C.muted; g.font = '11px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'top';
          g.fillText('return per episode (last ' + returns.length + ')', px, py - 14);
          const y0 = retY(0);
          g.strokeStyle = '#1b2434'; g.beginPath(); g.moveTo(px, y0); g.lineTo(px + pw, y0); g.stroke();
          g.fillStyle = C.muted; g.textAlign = 'right'; g.fillText('+10', px - 4, py); g.fillText('0', px - 4, y0 - 5); g.fillText('−40', px - 4, py + ph - 10);
          if (returns.length > 1) {
            g.strokeStyle = C.green; g.lineWidth = 1.5; g.beginPath();
            returns.forEach((v, i) => {
              const x = px + i / (returns.length - 1) * pw, y = retY(v);
              i ? g.lineTo(x, y) : g.moveTo(x, y);
            });
            g.stroke();
          }
          g.fillStyle = C.muted; g.textAlign = 'left'; g.font = '11px Inter, sans-serif';
          g.fillText('Q(s,a): green = positive, red = negative, arrow = greedy action', px, py + ph + 10);
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
          [you, ...agents].forEach((ag) => {
            if (ag.hist.length < 2) return;
            g.strokeStyle = ag.color; g.lineWidth = 1.5; g.beginPath();
            const stride = Math.max(1, Math.floor(ag.hist.length / 300));
            for (let i = 0; i < ag.hist.length; i += stride) { const x = px + i / xmax * pw, y = py + ph - ag.hist[i] / ymax * ph; i ? g.lineTo(x, y) : g.moveTo(x, y); }
            g.stroke();
          });
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
    },
  });
})();
