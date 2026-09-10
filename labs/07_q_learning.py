"""
LAB 07 - REINFORCEMENT LEARNING: Q-LEARNING IN A GRIDWORLD
==========================================================

Run:   python labs/07_q_learning.py
       python labs/07_q_learning.py --gamma 0.5        (short-sighted agent)
       python labs/07_q_learning.py --epsilon 0.0      (never explores)
       python labs/07_q_learning.py --coin             (reward hacking!)
       python labs/07_q_learning.py --slip 0.2         (slippery floor)
Needs: nothing but Python.
Time:  ~1-2 s.

WHAT YOU WILL BUILD
-------------------
Supervised learning (Labs 01-06) needs someone to hand over the right answer
for every input. Reinforcement learning (RL) has no answers, only REWARDS:
the agent acts, the world responds with a number, and the agent must figure
out for itself which actions led to the good numbers. It is the paradigm
behind game-playing AIs, robot control, and - via RLHF (Lab 08) - the
fine-tuning of every chat model you have used.

The world: a 5x7 grid with walls (#), a goal (G, +10, episode ends), a pit
(P, -10, episode ends) and a small penalty (-0.1) for every step, so
dawdling costs. The agent starts at S and can move up/right/down/left.

The algorithm: tabular Q-learning (Watkins, 1989). Keep a table
Q[state][action] = "how much total future reward do I expect if I take this
action here and behave well afterwards?" Update it after every step with

    Q[s][a] <- Q[s][a] + alpha * ( r + gamma * max_a' Q[s'][a']  -  Q[s][a] )
                                   \\_______ target ________/    \\_ old _/

i.e. nudge the old guess toward "the reward I just got, plus the best I
think I can do from where I landed, discounted by gamma". That is the
Bellman equation used as a learning rule. Deep Q-Networks (Atari, 2013)
replace the table with a neural net; the update is the same.

Exploration: with probability epsilon the agent takes a RANDOM action
instead of the best-known one ("epsilon-greedy"). Without it, the agent
sticks with the first thing that worked and never finds better routes.

WHAT TO OBSERVE
---------------
1. The sparkline of episode reward: negative and noisy at first (falling in
   the pit, wandering), then climbing to ~+9 once the route is found.
2. The policy map: arrows that all flow toward G and away from P.
3. The value grid: V(s) = max_a Q[s][a] is highest next to the goal and
   fades with distance - each step away multiplies by gamma and costs 0.1.

WHAT TO CHANGE (3 experiments)
------------------------------
A. --gamma 0.5 (vs 0.9). The discount says how much tomorrow's reward is
   worth today. At 0.5, a reward 10 steps away is worth 10 * 0.5^10 = 0.01,
   less than the step penalties to get there: far-away cells learn ~0 and
   the arrows there point nowhere useful. At 0.99 the whole grid glows.
B. --epsilon 0.0 vs 0.3. With no exploration the agent may lock onto the
   first (possibly long) route it stumbled on. With too much, it keeps
   falling in the pit "by accident" even after it knows better - look at
   the reward curve staying noisy.
C. --coin : REWARD HACKING. A coin worth +2 appears next to the start and
   respawns every time it is collected. The designer's intent was "go to
   the goal", but the reward function now says "collect coins forever",
   and that is exactly what the agent learns: bounce between two cells for
   100 steps, never reaching G. Nothing is broken - the agent optimised the
   reward it was given. This is the toy version of every RL horror story
   (the boat racing game that spun in circles collecting power-ups, the
   chatbot that learned to please the reward model instead of the human).
   Try --coin --coin-reward 0.5 : the loop is no longer worth it and the
   agent goes to the goal. The line between "helpful" and "hacked" is one
   number in the reward function.

COMMON BUGS
-----------
* Using Q[s'][a'] for the action you WILL take (that's SARSA) instead of
  max_a' Q[s'][a'] (Q-learning). Both work; they learn different things
  under exploration - Q-learning learns the optimal policy even while
  behaving randomly ("off-policy").
* Forgetting that terminal states have no future: the target for a step
  into the goal/pit is just r, not r + gamma * max Q.
* Bootstrapping from an unvisited state whose Q is 0 looks "optimistic" or
  "pessimistic" depending on the reward scale. Here 0 is between -10 and
  +10, so it is roughly neutral; with all-negative rewards, 0-init makes
  the agent explore unvisited states (optimism in the face of uncertainty).
* alpha too high (1.0) with a slippery floor: the table thrashes on every
  unlucky step. Use a small alpha when the world is random.
"""

import sys
import random
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. THE WORLD
# ---------------------------------------------------------------------------
LAYOUT = [
    ". . . . . . .",
    ". # . # . # .",
    "S . . P . . G",
    ". # . # . # .",
    ". . . . . . .",
]
# The straight line from S to G runs through the pit. Detours exist above and
# below it (via the gaps at columns 2 and 4) and all the way around the edge.

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]      # up, right, down, left
ARROWS = ["^", ">", "v", "<"]


class GridWorld:
    def __init__(self, layout, step_penalty=-0.1, goal_reward=10.0, pit_reward=-10.0,
                 slip=0.0, coin=None, coin_reward=2.0, rng=None):
        self.grid = [row.split() for row in layout]
        self.n_rows, self.n_cols = len(self.grid), len(self.grid[0])
        self.step_penalty, self.goal_reward, self.pit_reward = step_penalty, goal_reward, pit_reward
        self.slip = slip                     # prob. that the chosen move is replaced by a random one
        self.coin, self.coin_reward = coin, coin_reward   # coin = (row, col) or None
        self.rng = rng or random.Random(0)
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r][c] == "S":
                    self.start = (r, c)

    def cell(self, s):
        return self.grid[s[0]][s[1]]

    def step(self, s, a):
        """Apply action a in state s. Returns (next_state, reward, done)."""
        if self.slip > 0 and self.rng.random() < self.slip:
            a = self.rng.randrange(4)         # slippery floor: wrong direction
        dr, dc = ACTIONS[a]
        r, c = s[0] + dr, s[1] + dc
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols) or self.grid[r][c] == "#":
            r, c = s                          # bump into wall/edge: stay put
        s2 = (r, c)
        reward = self.step_penalty            # every step costs a little
        done = False
        if self.cell(s2) == "G":
            reward += self.goal_reward; done = True
        elif self.cell(s2) == "P":
            reward += self.pit_reward; done = True
        elif self.coin is not None and s2 == self.coin:
            reward += self.coin_reward        # coin respawns instantly: the trap
        return s2, reward, done


# ---------------------------------------------------------------------------
# 2. Q-LEARNING
# ---------------------------------------------------------------------------
def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1, max_steps=100, seed=0):
    rng = random.Random(seed)
    Q = {}                                    # state -> [q_up, q_right, q_down, q_left]

    def q(s):
        if s not in Q:
            Q[s] = [0.0, 0.0, 0.0, 0.0]       # unvisited states start at 0
        return Q[s]

    rewards, reached_goal = [], []
    for ep in range(episodes):
        s = env.start
        total = 0.0
        done = False
        for _ in range(max_steps):
            # epsilon-greedy: mostly exploit the table, sometimes explore
            if rng.random() < epsilon:
                a = rng.randrange(4)
            else:
                qs = q(s)
                best = max(qs)
                a = rng.choice([i for i in range(4) if qs[i] == best])   # random tie-break
            s2, r, done = env.step(s, a)
            total += r
            # ---- THE UPDATE ----
            # target = what we just learned this action is worth:
            #   the reward, plus (if the episode continues) gamma x the best
            #   we believe we can do from the new state.
            target = r if done else r + gamma * max(q(s2))
            q(s)[a] += alpha * (target - q(s)[a])     # move the old estimate toward it
            s = s2
            if done:
                break
        rewards.append(total)
        reached_goal.append(done and env.cell(s) == "G")
    return Q, rewards, reached_goal


# ---------------------------------------------------------------------------
# 3. PRINTING: policy map, value grid, sparkline, greedy rollout
# ---------------------------------------------------------------------------
def print_policy(env, Q):
    print("  Policy (best action per cell):   S start   G goal   P pit   # wall   $ coin")
    for r in range(env.n_rows):
        row = []
        for c in range(env.n_cols):
            ch = env.grid[r][c]
            if ch in "#GP":
                row.append(ch)
            elif env.coin == (r, c):
                row.append("$")
            elif (r, c) in Q and max(Q[(r, c)]) != 0.0:
                qs = Q[(r, c)]
                row.append(ARROWS[qs.index(max(qs))])
            else:
                row.append("?" if ch != "S" else "S")   # ? = never learned anything here
        print("     " + " ".join(row))
    print()


def print_values(env, Q):
    print("  Value grid  V(s) = max_a Q[s][a]  (how good is it to stand here?):")
    for r in range(env.n_rows):
        cells = []
        for c in range(env.n_cols):
            ch = env.grid[r][c]
            if ch == "#":
                cells.append("  ####  ")
            elif ch == "G":
                cells.append("  GOAL  ")
            elif ch == "P":
                cells.append("  PIT   ")
            else:
                v = max(Q.get((r, c), [0.0]))
                cells.append(f"{v:+7.2f} ")
        print("    " + "".join(cells))
    print()


def sparkline(values, buckets=50):
    """Average `values` into `buckets` and draw them with 10 ASCII levels."""
    n = len(values)
    per = max(1, n // buckets)
    means = [sum(values[i:i + per]) / len(values[i:i + per]) for i in range(0, n, per)]
    lo, hi = min(means), max(means)
    chars = " .:-=+*#%@"
    line = "".join(chars[int((m - lo) / (hi - lo + 1e-9) * (len(chars) - 1))] for m in means)
    return f"  |{line}|   ({per} episodes per column, {lo:+.1f} .. {hi:+.1f})"


def greedy_rollout(env, Q, max_steps=30):
    """Follow the learned policy without exploration; return the visited cells."""
    s, path, done = env.start, [env.start], False
    saved_slip, env.slip = env.slip, 0.0     # deterministic for the demo
    for _ in range(max_steps):
        if s not in Q:
            break
        qs = Q[s]
        s, _, done = env.step(s, qs.index(max(qs)))
        path.append(s)
        if done:
            break
    env.slip = saved_slip
    return path, done


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.9, help="discount factor (try 0.5, 0.99)")
    ap.add_argument("--epsilon", type=float, default=0.1, help="exploration rate (try 0.0, 0.3)")
    ap.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    ap.add_argument("--slip", type=float, default=0.0, help="prob. of a random move (try 0.2)")
    ap.add_argument("--coin", action="store_true", help="add a respawning coin next to S")
    ap.add_argument("--coin-reward", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coin = (1, 0) if args.coin else None     # the cell directly above S
    env = GridWorld(LAYOUT, slip=args.slip, coin=coin, coin_reward=args.coin_reward,
                    rng=random.Random(args.seed))

    print("=" * 70)
    print(f"Q-learning: {args.episodes} episodes, gamma={args.gamma}, epsilon={args.epsilon}, "
          f"alpha={args.alpha}, slip={args.slip}" + (f", coin=+{args.coin_reward}" if coin else ""))
    print("=" * 70)
    print("  The world:")
    for r, row in enumerate(env.grid):
        shown = [("$" if coin == (r, c) else ch) for c, ch in enumerate(row)]
        print("     " + " ".join(shown))
    print()

    Q, rewards, reached = q_learning(env, episodes=args.episodes, alpha=args.alpha,
                                     gamma=args.gamma, epsilon=args.epsilon, seed=args.seed)

    print("  Episode reward over training (higher = better):")
    print(sparkline(rewards))
    last = max(1, args.episodes // 10)
    print(f"  first {last} episodes: mean reward {sum(rewards[:last]) / last:+.2f}, "
          f"reached goal {100 * sum(reached[:last]) / last:.0f}%")
    print(f"  last  {last} episodes: mean reward {sum(rewards[-last:]) / last:+.2f}, "
          f"reached goal {100 * sum(reached[-last:]) / last:.0f}%")
    print()
    print_policy(env, Q)
    print_values(env, Q)

    path, done = greedy_rollout(env, Q)
    end = env.cell(path[-1])
    outcome = ("reached G" if done and end == "G" else
               "fell in P" if done and end == "P" else
               f"STILL WANDERING after {len(path) - 1} steps (cut off)")
    print(f"  Greedy rollout from S ({len(path) - 1} steps): {outcome}")
    print("    " + " -> ".join(f"({r},{c})" for r, c in path[:16]) + (" -> ..." if len(path) > 16 else ""))
    if coin and not (done and end == "G"):
        print()
        print("  Reward hacking in action: the agent found that bouncing on the coin")
        print("  cell forever pays more than the goal, and that IS the optimal policy")
        print("  for the reward we wrote. The reward was wrong, not the agent.")
        print("  Try:  --coin --coin-reward 0.5   (loop no longer worth it)")
        print("        --coin --gamma 0.5         (short-sighted, but does it help?)")
