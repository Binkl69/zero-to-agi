"""
LAB 08 - Toy RLHF: preferences -> reward model -> policy (and why it goes wrong)
===============================================================================

WHAT YOU'LL BUILD
-----------------
The whole post-training pipeline from chapter 11, at a scale you can hold in
your head. Instead of a language model with billions of parameters emitting
billions of possible responses, our "policy" is a probability distribution over
EIGHT hand-written responses to a single prompt. Everything else - the reward
model, the Bradley-Terry preference loss, the KL penalty, PPO-style policy
optimisation, and DPO - is the real algorithm, unchanged.

You will build, in order:

  Step 1  An SFT-like starting policy (a uniform-ish distribution over responses).
  Step 2  Simulated human labellers who compare pairs of responses, noisily,
          according to a hidden "true" reward they never state out loud.
  Step 3  A reward model trained on those comparisons with the Bradley-Terry
          loss - and, crucially, a reward model that CANNOT SEE the one feature
          that actually matters (whether the answer is correct).
  Step 4  Policy optimisation against the reward model with a KL penalty back to
          the starting policy, swept over beta.
  Step 5  DPO, which reaches the same place without ever building a reward model.

WHAT TO OBSERVE
---------------
1. The reward model learns to score responses well ON AVERAGE while being
   badly wrong about one specific response. It has learned POLITENESS AS A
   PROXY FOR CORRECTNESS, because in the training comparisons the two happen
   to correlate.

2. With beta = 0 (no KL penalty), the policy collapses almost entirely onto
   response R7 - the confident, polite, thorough-sounding, WRONG answer. That
   is reward hacking. The optimiser did its job perfectly; the reward was wrong.

3. As beta rises, the KL penalty pulls the policy back toward its starting
   point. There is a sweet spot: enough optimisation to improve the true
   reward, not so much that it exploits the reward model's blind spot. This is
   the whole reason real RLHF keeps a KL term.

4. DPO shows the identical failure. The problem was never the RL algorithm.
   The problem is that the preference data underdetermines what we wanted.

WHAT TO CHANGE (three experiments)
----------------------------------
1. In `RM_FEATURES`, add "correct" to the list so the reward model CAN see
   correctness. Re-run. The hacking disappears. This is why so much of real
   alignment work is about giving the reward signal access to what matters
   (unit tests, verifiers, expert graders) rather than about better optimisers.

2. Turn `N_COMPARISONS` down to 30. Watch the learned weights get noisy and the
   ranking become unstable between runs. Preference data is expensive, and this
   is what "not enough of it" looks like.

3. Raise `PREF_NOISE` to 1.5 (very inconsistent labellers). The reward model
   degrades gracefully rather than catastrophically - averaging over many noisy
   comparisons is surprisingly robust, which is the empirical fact that makes
   RLHF work at all.

COMMON BUGS
-----------
* Forgetting the minus sign in the Bradley-Terry loss. If your learned weights
  come out as the exact negative of the true ones, this is why.
* Computing softmax without subtracting the max first -> overflow to inf/nan
  as logits grow. See `softmax()` below.
* Using KL(reference || policy) instead of KL(policy || reference). The second
  is the one that punishes the policy for putting mass where the reference put
  none, which is exactly the runaway behaviour we want to prevent.
* Letting log(0) into the KL. Clip probabilities first.
"""

import sys

try:  # Windows consoles default to cp1252 and choke on non-ASCII output
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# THE SETUP
#
# One prompt, eight candidate responses. Each response is described by five
# features. In a real system these features do not exist; the reward model is a
# neural network reading raw text. Making them explicit here lets us see
# exactly what the reward model latches onto.
# ---------------------------------------------------------------------------

PROMPT = "My laptop won't connect to the office wifi. What should I check?"

FEATURES = ["length", "polite", "correct", "hedging", "refusal"]

#                        len   pol   cor   hed   ref
RESPONSES = [
    ("R0  'Reboot it.'",
                        [0.05, 0.10, 1.00, 0.00, 0.0]),
    ("R1  Terse but right: check the band, forget network, re-auth",
                        [0.25, 0.20, 1.00, 0.00, 0.0]),
    ("R2  Friendly and right: same steps, explained, offers follow-up",
                        [0.60, 0.90, 1.00, 0.10, 0.0]),
    ("R3  Thorough and right: steps + why + how to check the 5GHz band",
                        [0.85, 0.80, 1.00, 0.05, 0.0]),
    ("R4  Waffle: 'it could be many things, hard to say' + no steps",
                        [0.55, 0.60, 0.00, 0.95, 0.0]),
    ("R5  Refuses: 'I can't help with IT support issues.'",
                        [0.10, 0.50, 0.00, 0.00, 1.0]),
    ("R6  Wrong and blunt: 'Your wifi card is dead, buy a new laptop.'",
                        [0.20, 0.05, 0.00, 0.00, 0.0]),
    ("R7  Wrong but GORGEOUS: long, warm, confident, totally invented steps",
                        [1.00, 1.00, 0.00, 0.00, 0.0]),
]
NAMES = [r[0] for r in RESPONSES]
X = np.array([r[1] for r in RESPONSES], dtype=float)   # (8 responses, 5 features)
N_RESP, N_FEAT = X.shape

# The hidden preferences of our simulated humans. They care overwhelmingly
# about being CORRECT, like a bit of warmth, dislike padding, hate hedging and
# hate being refused. Nobody ever writes this vector down - the labellers just
# click "A is better than B", and we have to recover it.
TRUE_W = np.array([-0.30, 0.60, 3.00, -1.20, -2.00])

TRUE_REWARD = X @ TRUE_W

# THE CRITICAL LINE OF THIS LAB.
# Our reward model reads only surface features. It has no access to ground
# truth about correctness - exactly like a real reward model, which sees text
# and has to guess whether the content is right.
RM_FEATURES = ["length", "polite", "hedging", "refusal"]     # <- experiment 1: add "correct"
RM_IDX = [FEATURES.index(f) for f in RM_FEATURES]
X_RM = X[:, RM_IDX]

N_COMPARISONS = 400      # <- experiment 2: try 30
PREF_NOISE = 0.6         # <- experiment 3: try 1.5. Higher = more inconsistent labellers.


def softmax(z):
    """Numerically stable softmax. Subtracting the max never changes the result
    but stops exp() from overflowing once logits grow during training."""
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def bar(value, lo, hi, width=28):
    """A little ASCII bar so the tables are readable at a glance."""
    if hi - lo < 1e-9:
        frac = 0.5
    else:
        frac = (value - lo) / (hi - lo)
    n = int(round(np.clip(frac, 0, 1) * width))
    return "#" * n + "." * (width - n)


def rule(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


# ===========================================================================
# STEP 1 - the starting policy (what SFT hands to the RL stage)
# ===========================================================================
rule("STEP 1  The starting policy (post-SFT, pre-RL)")

# Supervised fine-tuning has taught the model to produce plausible, on-topic
# answers, but it has no strong opinion about which is best. We represent the
# policy as logits over the eight responses; softmax turns them into
# probabilities. A real policy is a distribution over token sequences, which is
# the same object with an astronomically larger support.
theta_ref = np.zeros(N_RESP)
theta_ref[[2, 3, 4, 7]] += 0.35        # SFT slightly favours fluent-looking answers
pi_ref = softmax(theta_ref)

print(f"\nPrompt: {PROMPT}\n")
print(f"{'response':<62} {'p(SFT)':>7} {'true r':>8}")
print("-" * 78)
for i in range(N_RESP):
    print(f"{NAMES[i]:<62} {pi_ref[i]:>7.3f} {TRUE_REWARD[i]:>8.2f}")

best_true = int(np.argmax(TRUE_REWARD))
worst_true = int(np.argmin(TRUE_REWARD))
print(f"\nBy the hidden human preference, the best response is {NAMES[best_true].split()[0]} "
      f"and the worst is {NAMES[worst_true].split()[0]}.")
print(f"Expected true reward of the starting policy: {pi_ref @ TRUE_REWARD:.3f}")


# ===========================================================================
# STEP 2 - collect human preference comparisons
# ===========================================================================
rule("STEP 2  Collecting pairwise preferences from (simulated) humans")

# Why comparisons instead of scores? Because humans are bad at absolute ratings
# ("is this answer a 7 or an 8?") and good at relative ones ("this one is
# better"). Bradley-Terry is the standard model of that judgement: the
# probability a labeller prefers A over B is sigmoid of the reward difference.
# PREF_NOISE scales the difference; larger noise flattens the sigmoid toward a
# coin flip.
pairs = []
for _ in range(N_COMPARISONS):
    a, b = rng.choice(N_RESP, size=2, replace=False)
    p_a_wins = sigmoid((TRUE_REWARD[a] - TRUE_REWARD[b]) / PREF_NOISE)
    if rng.random() < p_a_wins:
        pairs.append((a, b))      # (winner, loser)
    else:
        pairs.append((b, a))

wins = np.zeros(N_RESP)
appear = np.zeros(N_RESP)
for w, l in pairs:
    wins[w] += 1
    appear[w] += 1
    appear[l] += 1
winrate = np.divide(wins, np.maximum(appear, 1))

print(f"\nCollected {len(pairs)} comparisons. Empirical win rates:\n")
print(f"{'response':<62} {'wins':>5} {'seen':>5} {'rate':>6}")
print("-" * 78)
for i in np.argsort(-winrate):
    print(f"{NAMES[i]:<62} {int(wins[i]):>5} {int(appear[i]):>5} {winrate[i]:>6.2f}")
print("\nNote what the labellers never told us: WHY they preferred what they did.")


# ===========================================================================
# STEP 3 - train the reward model with the Bradley-Terry loss
# ===========================================================================
rule("STEP 3  Training the reward model (Bradley-Terry loss)")

# The reward model is a linear function of the surface features:
#     r_hat(response) = w . x_rm(response)
# and we fit w by maximising the likelihood of the observed comparisons:
#     P(winner beats loser) = sigmoid(r_hat(winner) - r_hat(loser))
#     loss = -mean log sigmoid(r_hat(winner) - r_hat(loser))
# This is exactly the loss used to train real reward models; only the function
# class changes (a transformer with a scalar head instead of a dot product).

w_rm = np.zeros(len(RM_IDX))
lr, n_epochs = 0.5, 400
win_idx = np.array([p[0] for p in pairs])
lose_idx = np.array([p[1] for p in pairs])

print(f"\nreward model sees: {RM_FEATURES}")
print(f"reward model CANNOT see: {[f for f in FEATURES if f not in RM_FEATURES]}\n")
print(f"{'epoch':>6} {'BT loss':>10} {'pair accuracy':>15}")
print("-" * 78)

for epoch in range(n_epochs + 1):
    r_hat = X_RM @ w_rm
    diff = r_hat[win_idx] - r_hat[lose_idx]
    p_win = sigmoid(diff)
    loss = -np.mean(np.log(np.clip(p_win, 1e-12, 1.0)))

    # d(loss)/dw = -mean[ (1 - p_win) * (x_winner - x_loser) ]
    grad = -np.mean((1.0 - p_win)[:, None] * (X_RM[win_idx] - X_RM[lose_idx]), axis=0)
    # small weight decay keeps the weights finite when a feature is separable
    grad += 0.01 * w_rm

    if epoch % 100 == 0:
        print(f"{epoch:>6} {loss:>10.4f} {np.mean(diff > 0):>15.3f}")
    w_rm -= lr * grad

R_HAT = X_RM @ w_rm

print("\nLearned reward-model weights (on the features it can see):")
for f, val in zip(RM_FEATURES, w_rm):
    print(f"   {f:<10} {val:>+7.3f}   {bar(val, -2.5, 2.5)}")
print("\nThe hidden truth, for comparison:")
for f, val in zip(FEATURES, TRUE_W):
    seen = "" if f in RM_FEATURES else "   <- INVISIBLE to the reward model"
    print(f"   {f:<10} {val:>+7.3f}   {bar(val, -2.5, 2.5)}{seen}")

print(f"\n{'response':<62} {'r_hat':>7} {'true r':>8}")
print("-" * 78)
order = np.argsort(-R_HAT)
for i in order:
    flag = ""
    if R_HAT[i] >= R_HAT[order[0]] - 1e-9:
        flag = "  <- reward model's favourite"
    print(f"{NAMES[i]:<62} {R_HAT[i]:>7.2f} {TRUE_REWARD[i]:>8.2f}{flag}")

rm_best = int(np.argmax(R_HAT))
print(f"\nThe reward model ranks pairs correctly {np.mean((R_HAT[win_idx] - R_HAT[lose_idx]) > 0):.1%} "
      f"of the time - it looks like a good model.")
if rm_best != best_true:
    print(f"But its single favourite response is {NAMES[rm_best].split()[0]}, whose TRUE reward is "
          f"{TRUE_REWARD[rm_best]:.2f}, while the genuinely best response "
          f"{NAMES[best_true].split()[0]} scores {TRUE_REWARD[best_true]:.2f}.")
    print("Having learned politeness and length as proxies for correctness, the reward")
    print("model cannot tell a right answer from a beautifully-written wrong one.")


# ===========================================================================
# STEP 4 - optimise the policy against the reward model, with a KL penalty
# ===========================================================================
rule("STEP 4  Policy optimisation against the reward model (the RL step)")

# The objective real RLHF maximises:
#     J(pi) = E_{a ~ pi}[ r_hat(a) ]  -  beta * KL(pi || pi_ref)
#
# With only eight actions we can take the expectation exactly instead of
# sampling, which removes all the variance that makes PPO fiddly in practice.
# The gradient with respect to the logits works out beautifully:
#     f_a         = r_hat(a) - beta * (log pi_a - log pi_ref_a)
#     dJ/dtheta_j = pi_j * ( f_j - E_pi[f] )
# i.e. push up any action whose adjusted score beats the current average.

def optimise_policy(beta, steps=3000, lr=0.15):
    theta = theta_ref.copy()
    for _ in range(steps):
        pi = softmax(theta)
        pi_c = np.clip(pi, 1e-12, 1.0)
        f = R_HAT - beta * (np.log(pi_c) - np.log(np.clip(pi_ref, 1e-12, 1.0)))
        grad = pi * (f - pi @ f)
        theta += lr * grad          # ascent: we are MAXIMISING J
        theta = np.clip(theta, -60, 60)
    pi = softmax(theta)
    pi_c = np.clip(pi, 1e-12, 1.0)
    kl = float(np.sum(pi * (np.log(pi_c) - np.log(np.clip(pi_ref, 1e-12, 1.0)))))
    return pi, kl


print("\nSweeping the KL coefficient beta. 'RM reward' is what the optimiser sees;")
print("'TRUE reward' is what we actually wanted. Watch them come apart.\n")
# "p(correct)" is the total probability the policy puts on responses that are
# actually right. It is the clearest single number for watching alignment fail.
is_correct = X[:, FEATURES.index("correct")] > 0.5

print(f"{'beta':>6} {'KL':>7} {'RM reward':>11} {'TRUE reward':>13} {'p(correct)':>12}  {'p(R7, the trap)':>15}")
print("-" * 96)

results = []
for beta in [0.0, 0.05, 0.15, 0.4, 1.0, 3.0]:
    pi, kl = optimise_policy(beta)
    rm_r = float(pi @ R_HAT)
    true_r = float(pi @ TRUE_REWARD)
    p_ok = float(pi[is_correct].sum())
    results.append((beta, kl, rm_r, true_r, int(np.argmax(pi)), pi))
    print(f"{beta:>6.2f} {kl:>7.2f} {rm_r:>11.3f} {true_r:>13.3f} {p_ok:>12.3f}  {pi[7]:>15.3f}")

print(f"{'(SFT)':>6} {0.0:>7.2f} {float(pi_ref @ R_HAT):>11.3f} {float(pi_ref @ TRUE_REWARD):>13.3f} "
      f"{float(pi_ref[is_correct].sum()):>12.3f}  {pi_ref[7]:>15.3f}   <- where we started")

base_true = float(pi_ref @ TRUE_REWARD)
best_beta = max(results, key=lambda r: r[3])
print(f"\nStarting policy's true reward: {base_true:.3f}")
print(f"beta = 0.00 (pure reward maximisation) true reward: {results[0][3]:.3f}  "
      f"{'WORSE than where we started' if results[0][3] < base_true else ''}")
print(f"Best beta in this sweep: {best_beta[0]:.2f}, true reward {best_beta[3]:.3f}")

print("\nThis is the over-optimisation curve that every RLHF practitioner knows:")
print("reward-model score rises monotonically as beta falls, while the thing you")
print("actually care about rises, peaks, and then falls off a cliff. The KL term is")
print("not a regulariser for numerical convenience. It is the only thing stopping")
print("the optimiser from walking into the reward model's blind spot.")

print("\nFull distribution at beta = 0 (no KL penalty):")
pi0 = results[0][5]
for i in np.argsort(-pi0)[:4]:
    print(f"   {NAMES[i]:<62} p={pi0[i]:.3f}")


# ===========================================================================
# STEP 5 - DPO: the same result with no reward model at all
# ===========================================================================
rule("STEP 5  DPO - skipping the reward model entirely")

# Direct Preference Optimisation (Rafailov et al., 2023) observes that the
# optimal policy for the KL-regularised objective has a closed form, so you can
# rearrange it to express the reward in terms of the policy, and fit the policy
# DIRECTLY on the preference pairs:
#
#   loss = -log sigmoid( beta * [ (log pi_w - log pi_ref_w)
#                                -(log pi_l - log pi_ref_l) ] )
#
# No reward model, no sampling, no RL loop - just a supervised loss on pairs.
# Note the same beta appears, playing the same role.

def dpo(beta, steps=3000, lr=0.15):
    theta = theta_ref.copy()
    log_ref = np.log(np.clip(pi_ref, 1e-12, 1.0))
    for _ in range(steps):
        pi = softmax(theta)
        log_pi = np.log(np.clip(pi, 1e-12, 1.0))
        h = (log_pi - log_ref)                       # implicit reward, per response
        margin = beta * (h[win_idx] - h[lose_idx])
        s = sigmoid(-margin)                         # d/dmargin of -log sigmoid(margin)
        # accumulate dloss/dlog_pi over the batch
        g_log = np.zeros(N_RESP)
        np.add.at(g_log, win_idx, -beta * s)
        np.add.at(g_log, lose_idx, beta * s)
        g_log /= len(win_idx)
        # chain through softmax: dlog_pi_a/dtheta_j = delta_aj - pi_j
        grad = g_log - pi * np.sum(g_log)
        theta -= lr * grad                            # descent: we are MINIMISING the loss
        theta = np.clip(theta, -60, 60)
    return softmax(theta)


print(f"\n{'beta':>6} {'TRUE reward':>13}  {'top response':<44} {'p(top)':>7}")
print("-" * 78)
for beta in [0.05, 0.1, 0.3, 1.0]:
    pi = dpo(beta)
    top = int(np.argmax(pi))
    print(f"{beta:>6.2f} {float(pi @ TRUE_REWARD):>13.3f}  {NAMES[top][:44]:<44} {pi[top]:>7.3f}")

rule("WHAT THIS LAB SHOWS")
print("""
1. Comparisons are enough. From nothing but "A is better than B" we recovered a
   usable reward function. This is why RLHF scales: judging is far cheaper than
   demonstrating.

2. The reward model is a MEASUREMENT, and it inherits every blind spot of the
   data it was fitted to. Ours learned that polite, thorough-looking answers win,
   which was true on average and catastrophically false for one response.

3. Optimising hard against an imperfect measurement destroys the thing being
   measured. That is Goodhart's law, and it is the mechanism behind the
   sycophancy and padding you can still feel in deployed assistants.

4. The KL penalty buys back safety by refusing to travel far from a policy we
   already trust. Choosing beta is choosing how much you trust your reward model.

5. DPO removes the reward model but not the problem: it is fitted to the same
   preferences, so it inherits the same blind spot. The fix is never a better
   optimiser. It is a reward signal that can see what matters - which is exactly
   why 2024-2025 reasoning models moved to VERIFIABLE rewards (unit tests, proof
   checkers, exact-match answers) wherever they could.

Now go do experiment 1 in the docstring: add "correct" to RM_FEATURES and watch
the hacking vanish.
""")
