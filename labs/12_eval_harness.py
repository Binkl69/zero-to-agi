"""
LAB 12 - An evaluation harness (and six ways a score can lie to you)
====================================================================

Run:   python labs/12_eval_harness.py
       python labs/12_eval_harness.py --seed 1        (a different draw of everything)
Needs: numpy.
Time:  about two seconds.

WHAT YOU'LL BUILD
-----------------
"Model B scored 74%, model A scored 70%, so B is better" is the sentence this
lab teaches you to distrust. You build the harness that decides whether it is
true: an eval set, a scorer, error bars, a fair comparison, a regression
suite, a check on an automatic judge, and a check for leaked test items.

There is no real model here, and that is the point. The "models" are simulated
with a TRUE accuracy that we choose (A = 70%, B = 74%). Because we know the
right answer, we can see exactly when the harness gets it wrong. With a real
model you never have that luxury, which is why you need the habits.

  Step 1  Score two models on 1,000 items. Compare to the truth.
  Step 2  Bootstrap confidence intervals: how far off could a score be?
  Step 3  The winner flips. At 20 items the worse model wins or ties
          more than four times in ten; at 1,000, about once in a hundred.
  Step 4  Paired versus unpaired: the same data, analysed two ways,
          gives two different answers. One is correct.
  Step 5  A regression suite: the "fix" that raises the overall score
          while breaking a slice that used to work.
  Step 6  An LLM judge with position bias and length bias, and the
          order-swap test that exposes the first but not the second.
  Step 7  Contamination: leak test items into training and watch a 60%
          model score 90%.

WHAT TO OBSERVE
---------------
1. At n = 1,000 the two scores come out at 73.1% and 75.3%. The truth is 70%
   and 74%: model A got a lucky draw of 3.1 points. The 95% interval at
   n = 1,000 is about +-2.8 points. At n = 20 it is +-15 or more: a 20-item
   eval cannot tell 70% from 55% or 85%.

2. Cut the 1,000 items into 50 mini-evals of 20: B wins 26, the worse
   model A wins 16, and 8 are ties. Over 2,000 fresh eval sets of 20
   items, B is ahead only 56% of the time. With 1,000 items, 99%.

3. Being ahead is not the same as being clearly ahead. On our 1,000 items
   B leads by 2.2 points, and neither analysis can rule out "no difference":
   unpaired [-1.8, +6.0], paired [-0.8, +5.1]. The paired interval, from THE
   SAME answers, is 24% narrower, for free. Over 2,000 fresh eval sets the
   unpaired analysis detects the real 4-point gap 51% of the time and the
   paired one 72%. So even 1,000 items leave a 4-point gap undetected about
   three times in ten, and this default run happens to be one of those.

4. Version 2 of a model scores 4.4 points higher overall and is 21 points
   worse on unit conversions. The overall number hides it; the per-slice
   paired check catches it.

5. The simulated judge picks whichever answer it reads first 61% of the time
   (a fair judge: 50%), and reverses its verdict on 23% of pairs when the
   order is swapped. Counting only verdicts that survive the swap removes
   that. It does nothing about the length bias: on the pairs that survive,
   the judge still prefers the wordier model 65% of the time, when the
   terse one is truly the better on 58% of all pairs.

6. A model with a true accuracy of 60% scores 96.7% on a test set that is
   fully in its training data, and overtakes an honest 70% model once 30%
   of the test set has leaked.

WHAT TO CHANGE (5 experiments - predict the outcome BEFORE you run)
-------------------------------------------------------------------
Each is a flag, so you do not have to edit the file.

1. --seed 1, then 2, 3. Everything is redrawn. Predict which conclusions
   survive. At n = 1,000 B is ahead every time (by 4.3, 1.5 and 4.9
   points), but the paired interval only excludes zero for seeds 1 and 3.
   The first block of 20 items is a coin flip: over seeds 0-3 it says A,
   tie, B, A. The broken slice in Step 5 is caught every time. A result
   that holds for one seed is an anecdote.

2. --gap 2 (B is truly 72%, two points better instead of four). At
   n = 1,000 the paired analysis now detects the gap only 24% of the time
   (it was 72%), and B is ahead in 89% of eval sets instead of 99%. Even
   3,000 items detect it just 58% of the time. Halving the gap you want to
   see costs about four times the data, not twice.

3. --difficulty-spread 0 (every item equally hard, so the two models no
   longer succeed and fail on the SAME items; they agree on about 600 of
   1,000 instead of 768). Pairing stops helping: the two intervals are
   the same width and both detect the gap 52% of the time. Pairing only
   helps when the items are part of what makes the answers vary - which,
   on real evals, they nearly always are.

4. --position-bias 0. The order-swap test goes quiet (the first answer
   wins 50.0% of the time, 0.0% of verdicts reverse) - and the judge still
   prefers the wordy model on 62% of pairs. A clean order-swap test does
   not mean a clean judge. Then add --length-bias 0 as well: the wordy
   model's share drops to 44%, close to the truth (42%).

5. --memorised 0.75 (the model recalls only three in four leaked items).
   The fully-leaked score falls from 96.7% to 74.8%, and the leak needed
   to pass the honest 70% model rises from 30% to 80%. Contamination does
   not have to be total, or perfectly memorised, to change a ranking.

COMMON BUGS
-----------
* Resampling the two models' answers separately when they answered the same
  items. That throws away the pairing and widens the interval (Step 4).
* Bootstrapping with too few resamples. Below a few hundred the interval's
  own ends are noisy. 2,000 is cheap.
* Reporting only the overall score. Averages hide slices (Step 5).
* Treating "the judge agrees with itself" as "the judge is right" (Step 6).
* Tuning on the test set. Every time you look at the score and change
  something, a little of the test set leaks into your decisions (Step 7).
"""

import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):   # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

ap = argparse.ArgumentParser(description="An evaluation harness on simulated models.")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gap", type=float, default=4.0,
                help="true accuracy of B minus A, in points (default 4)")
ap.add_argument("--difficulty-spread", type=float, default=2.0,
                help="how much items differ in difficulty (0 = all the same)")
ap.add_argument("--position-bias", type=float, default=0.5,
                help="the judge's bonus for the answer shown first")
ap.add_argument("--length-bias", type=float, default=0.5,
                help="the judge's bonus per standard deviation of answer length")
ap.add_argument("--memorised", type=float, default=0.97,
                help="chance the contaminated model gets a leaked item right")
args = ap.parse_args()

SEED = args.seed
rng = np.random.default_rng(SEED)

TRUE_A = 0.70                       # model A's true accuracy
TRUE_B = TRUE_A + args.gap / 100    # model B's true accuracy
SPREAD = args.difficulty_spread
N_BOOT = 2000                       # bootstrap resamples
N_REPEATS = 2000                    # fresh eval sets when we measure "how often"


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# THE SIMULATED WORLD
#
# Every item has a hidden difficulty. Every model has a hidden skill. The
# chance a model gets an item right is sigmoid(skill - difficulty): easy items
# are answered by everyone, hard items by nobody, and the models differ in the
# middle. This matters for Step 4: two models answering the SAME items tend to
# succeed and fail together, exactly as real models do.
# ---------------------------------------------------------------------------

_GRID = np.linspace(-6, 6, 2001)                       # difficulties, in standard deviations
_GRID_W = np.exp(-0.5 * _GRID ** 2)
_GRID_W /= _GRID_W.sum()


def true_accuracy(skill, spread=None):
    """The accuracy this skill gives over the whole (infinite) population of items."""
    spread = SPREAD if spread is None else spread
    return float(np.sum(_GRID_W * sigmoid(skill - spread * _GRID)))


def skill_for(target, spread=None):
    """Find, by bisection, the skill whose true accuracy is `target`."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if true_accuracy(mid, spread) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


SKILL_A = skill_for(TRUE_A)
SKILL_B = skill_for(TRUE_B)


def draw_items(n, generator):
    return SPREAD * generator.standard_normal(n)


def answer(skill, difficulty, generator):
    """Run a simulated model over items. Returns an array of 1 (right) / 0 (wrong)."""
    return (generator.random(difficulty.shape) < sigmoid(skill - difficulty)).astype(float)


def pct(x):
    return f"{100 * x:5.1f}%"


# ---------------------------------------------------------------------------
# STEP 1: score two models
# ---------------------------------------------------------------------------
rule("STEP 1: Score two models on 1,000 items")

N = 1000
difficulty = draw_items(N, rng)
a = answer(SKILL_A, difficulty, rng)        # model A's 1,000 right/wrong marks
b = answer(SKILL_B, difficulty, rng)        # model B's, on the SAME items

print(f"\n  {'model':<8} {'true accuracy':>14} {'measured (n=1000)':>19} {'error':>8}")
print("  " + "-" * 52)
for name, truth, marks in [("A", TRUE_A, a), ("B", TRUE_B, b)]:
    print(f"  {name:<8} {pct(truth):>14} {pct(marks.mean()):>19} {100 * (marks.mean() - truth):>+7.1f}")
worst_error = max(abs(a.mean() - TRUE_A), abs(b.mean() - TRUE_B))
print(f"\n  A harness is a measuring instrument, and no reading is exact: with 1,000")
print(f"  items the worse of these two is off by {100 * worst_error:.1f} points. With a real model you")
print(f"  cannot see the 'true' column. The question is: how far off COULD a score be?")


# ---------------------------------------------------------------------------
# STEP 2: bootstrap confidence intervals
#
# We have one eval set and want to know how much the score would move if we
# had drawn a different one. The bootstrap fakes "a different one" by drawing
# n items from our own n items WITH REPLACEMENT (some twice, some never),
# rescoring, and repeating 2,000 times. The middle 95% of those rescored
# values is the confidence interval. No formula, no assumptions about bell
# curves, and it works for any metric you can compute.
# ---------------------------------------------------------------------------

def bootstrap_ci(marks, generator, n_boot=N_BOOT):
    n = len(marks)
    idx = generator.integers(0, n, size=(n_boot, n))      # each row: one resampled eval set
    means = marks[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


rule("STEP 2: Bootstrap confidence intervals - how far off could a score be?")
print(f"\n  Model A (true accuracy {pct(TRUE_A).strip()}) scored on the first n items:\n")
print(f"  {'n':>6} {'score':>8} {'95% interval':>20} {'half-width':>12}")
print("  " + "-" * 50)
half_widths = {}
for n in [20, 50, 100, 300, 1000]:
    lo, hi = bootstrap_ci(a[:n], rng)
    half_widths[n] = (hi - lo) / 2
    print(f"  {n:>6} {pct(a[:n].mean()):>8}     [{pct(lo)}, {pct(hi)}] {100 * (hi - lo) / 2:>9.1f} pts")
print(f"\n  The interval at n=20 is {half_widths[20] / half_widths[1000]:.1f} times wider than at n=1000. To halve an")
print(f"  error bar you need FOUR times the items: the width shrinks like 1/sqrt(n).")


# ---------------------------------------------------------------------------
# STEP 3: the winner flips
# ---------------------------------------------------------------------------
rule("STEP 3: The winner flips at small n")

print(f"\n  Our 1,000 items, cut into 50 separate mini-evals of 20 items each.")
print(f"  B is truly better by {args.gap:.0f} points. Who wins each mini-eval?\n")
blocks_a = a.reshape(50, 20).mean(axis=1)
blocks_b = b.reshape(50, 20).mean(axis=1)
wins_b = int(np.sum(blocks_b > blocks_a))
wins_a = int(np.sum(blocks_a > blocks_b))
ties = 50 - wins_a - wins_b
line = "".join("B" if y > x else ("A" if x > y else "=") for x, y in zip(blocks_a, blocks_b))
print(f"    {line}")
print(f"\n    B wins {wins_b} of 50, the worse model A wins {wins_a}, {ties} are ties.")
first = "B" if blocks_b[0] > blocks_a[0] else ("A" if blocks_a[0] > blocks_b[0] else "a tie")
print(f"    First 20 items alone: A {pct(blocks_a[0]).strip()}, B {pct(blocks_b[0]).strip()} -> {first}.")
print(f"    All 1,000 items:      A {pct(a.mean()).strip()}, B {pct(b.mean()).strip()} -> "
      f"{'B' if b.mean() > a.mean() else 'A'} by {100 * abs(b.mean() - a.mean()):.1f} points.")

print(f"\n  And over {N_REPEATS:,} completely fresh eval sets of each size:\n")
print(f"  {'n':>6} {'B ahead':>9} {'tie':>7} {'A ahead (wrong)':>17}")
print("  " + "-" * 43)
ahead = {}
for n in [20, 50, 100, 300, 1000, 3000]:
    d = draw_items((N_REPEATS, n), rng)
    ra = answer(SKILL_A, d, rng).mean(axis=1)
    rb = answer(SKILL_B, d, rng).mean(axis=1)
    ahead[n] = float(np.mean(rb > ra))
    print(f"  {n:>6} {pct(np.mean(rb > ra)):>9} {pct(np.mean(rb == ra)):>7} {pct(np.mean(ra > rb)):>17}")
print(f"\n  At n=20 the better model is ahead only {pct(ahead[20]).strip()} of the time. A leaderboard")
print(f"  built on 20 examples is mostly reporting which examples were picked.")


# ---------------------------------------------------------------------------
# STEP 4: paired versus unpaired
#
# Both models answered the same 1,000 items. Two ways to put an interval on
# the difference B - A:
#
#   UNPAIRED  pretend the two score lists are unrelated. Resample A's marks,
#             separately resample B's marks, subtract the means.
#   PAIRED    resample ITEMS. Each resampled item brings both models' marks
#             with it, so "this item was hard for everyone" cancels out.
#
# The paired interval is narrower whenever the models tend to get the same
# items right, and it is the correct one for this design.
# ---------------------------------------------------------------------------

def unpaired_diff_ci(x, y, generator, n_boot=N_BOOT):
    n = len(x)
    ix = generator.integers(0, n, size=(n_boot, n))
    iy = generator.integers(0, n, size=(n_boot, n))       # a DIFFERENT resample for y
    diffs = y[iy].mean(axis=1) - x[ix].mean(axis=1)
    return tuple(float(v) for v in np.percentile(diffs, [2.5, 97.5]))


def paired_diff_ci(x, y, generator, n_boot=N_BOOT):
    n = len(x)
    idx = generator.integers(0, n, size=(n_boot, n))      # ONE resample of items, used for both
    diffs = (y - x)[idx].mean(axis=1)
    return tuple(float(v) for v in np.percentile(diffs, [2.5, 97.5]))


rule("STEP 4: Paired versus unpaired comparison on the same items")

both = int(np.sum((a == 1) & (b == 1)))
neither = int(np.sum((a == 0) & (b == 0)))
only_a = int(np.sum((a == 1) & (b == 0)))
only_b = int(np.sum((a == 0) & (b == 1)))
print(f"\n  Item by item:  both right {both},  both wrong {neither},  only A right {only_a},  only B right {only_b}")
print(f"  The models agree on {both + neither} of {N} items. All the information about which is")
print(f"  better is in the {only_a + only_b} items where they differ ({only_b} - {only_a} = {only_b - only_a:+d} for B).")

diff = b.mean() - a.mean()
u_lo, u_hi = unpaired_diff_ci(a, b, rng)
p_lo, p_hi = paired_diff_ci(a, b, rng)


def verdict(lo, hi):
    return "excludes 0: B is better" if lo > 0 else ("excludes 0: A is better" if hi < 0 else "includes 0: cannot tell")


print(f"\n  B - A = {100 * diff:+.1f} points (true gap {args.gap:+.1f}).  95% interval for the difference:\n")
print(f"    unpaired  [{100 * u_lo:+5.1f}, {100 * u_hi:+5.1f}]   width {100 * (u_hi - u_lo):4.1f}   {verdict(u_lo, u_hi)}")
print(f"    paired    [{100 * p_lo:+5.1f}, {100 * p_hi:+5.1f}]   width {100 * (p_hi - p_lo):4.1f}   {verdict(p_lo, p_hi)}")
narrower = 100 * (1 - (p_hi - p_lo) / (u_hi - u_lo))
if narrower >= 8:
    print(f"\n  Same answers, same items. The paired interval is {narrower:.0f}% narrower.")
else:
    print(f"\n  Same answers, same items. The two intervals are about the same width.")
if p_lo <= 0 <= p_hi:
    print(f"  In this draw even 1,000 items did not settle it: zero is still inside the")
    print(f"  interval. That is not a bug. The table below says how often it happens.")

# One eval set is one draw. How often does each method detect the real gap?
# (For speed this uses the textbook formula for the interval instead of the
# bootstrap; on the eval set above the two agree to the first decimal place.)
print(f"\n  How often does each method detect the gap? {N_REPEATS:,} fresh eval sets per row:\n")
print(f"  {'n':>6} {'unpaired detects':>18} {'paired detects':>16}")
print("  " + "-" * 42)
power = {}
for n in [100, 300, 1000, 3000]:
    d = draw_items((N_REPEATS, n), rng)
    xa = answer(SKILL_A, d, rng)
    xb = answer(SKILL_B, d, rng)
    ma, mb = xa.mean(axis=1), xb.mean(axis=1)
    se_unpaired = np.sqrt(ma * (1 - ma) / n + mb * (1 - mb) / n)
    se_paired = (xb - xa).std(axis=1, ddof=1) / np.sqrt(n)
    det_u = float(np.mean((mb - ma) - 1.96 * se_unpaired > 0))
    det_p = float(np.mean((mb - ma) - 1.96 * se_paired > 0))
    power[n] = (det_u, det_p)
    print(f"  {n:>6} {pct(det_u):>18} {pct(det_p):>16}")
print(f"\n  \"Detects\" = the 95% interval for B - A lies wholly above zero.")
print(f"  At n=1000: unpaired {pct(power[1000][0]).strip()}, paired {pct(power[1000][1]).strip()}. Pairing is free statistical power;")
print(f"  running two models on different random samples of a benchmark wastes it.")


# ---------------------------------------------------------------------------
# STEP 5: a regression suite
#
# The eval set has four slices. Version 2 of a model was trained to fix its
# worst slice (dates). It did - and unit conversions got worse, because the
# fix taught it a shortcut that misfires there. The overall score went UP.
# ---------------------------------------------------------------------------
rule("STEP 5: A regression suite catches the fix that breaks something else")

SLICES = ["arithmetic", "dates", "units", "trivia"]
#               true accuracy per slice:   v1     v2
SLICE_TRUTH = {"arithmetic": (0.80, 0.80),
               "dates":      (0.45, 0.85),      # the fix
               "units":      (0.85, 0.62),      # the collateral damage
               "trivia":     (0.70, 0.70)}
PER_SLICE = 250

v1_marks, v2_marks = {}, {}
for s in SLICES:
    d = draw_items(PER_SLICE, rng)
    v1_marks[s] = answer(skill_for(SLICE_TRUTH[s][0]), d, rng)
    # v2 answers the same items. An item v1 got right, v2 usually gets right
    # too unless its skill on this slice dropped.
    v2_marks[s] = answer(skill_for(SLICE_TRUTH[s][1]), d, rng)

all_v1 = np.concatenate([v1_marks[s] for s in SLICES])
all_v2 = np.concatenate([v2_marks[s] for s in SLICES])
lo, hi = paired_diff_ci(all_v1, all_v2, rng)
print(f"\n  Overall:  v1 {pct(all_v1.mean()).strip()}  ->  v2 {pct(all_v2.mean()).strip()}   "
      f"change {100 * (all_v2.mean() - all_v1.mean()):+.1f} points, paired 95% interval [{100 * lo:+.1f}, {100 * hi:+.1f}]")
print(f"  On the overall number alone, v2 looks better and ships.\n")

print(f"  {'slice':<12} {'v1':>7} {'v2':>7} {'change':>8} {'paired 95% interval':>21} {'newly broken':>13}  verdict")
print("  " + "-" * 84)
regressions = []
for s in SLICES:
    x, y = v1_marks[s], v2_marks[s]
    lo, hi = paired_diff_ci(x, y, rng)
    broken = int(np.sum((x == 1) & (y == 0)))        # items that used to pass and now fail
    if hi < 0:
        status = "REGRESSION"
        regressions.append(s)
    elif lo > 0:
        status = "improved"
    else:
        status = "no clear change"
    print(f"  {s:<12} {pct(x.mean()):>7} {pct(y.mean()):>7} {100 * (y.mean() - x.mean()):>+7.1f} "
          f"     [{100 * lo:+6.1f}, {100 * hi:+6.1f}] {broken:>13}  {status}")
if regressions:
    worst = max(regressions, key=lambda s: v1_marks[s].mean() - v2_marks[s].mean())
    drop = 100 * (v1_marks[worst].mean() - v2_marks[worst].mean())
    print(f"\n  REGRESSION SUITE: FAIL - {', '.join(regressions)} got worse "
          f"({drop:.0f} points on {worst}). The overall score went {'up' if all_v2.mean() > all_v1.mean() else 'down'}.")
else:
    print(f"\n  REGRESSION SUITE: pass - no slice got clearly worse.")
print(f"  \"Newly broken\" counts items v1 answered correctly that v2 now gets wrong. Some")
print(f"  churn is normal in every slice; the paired interval says when it is more than churn.")
print(f"  (A 95% interval is wrong 1 time in 20, so a slice that did not truly change will")
print(f"  now and then be flagged too. Over seeds 0-29 that happens once.)")


# ---------------------------------------------------------------------------
# STEP 6: an LLM judge, and how to audit it
#
# For open-ended answers there is no answer key, so people ask another model
# to pick the better of two answers. That judge is itself a model with
# habits. Ours has two, both documented in real LLM judges:
#
#   position bias   a bonus for whichever answer is shown FIRST
#   length bias     a bonus for the LONGER answer
#
# Each answer has a true quality (which we know, because we made it up) and a
# length. The judge sees quality through some noise of its own, adds its
# biases, and picks the higher total.
# ---------------------------------------------------------------------------
rule("STEP 6: A biased LLM judge, and the order-swap test")

N_PAIRS = 2000
POSITION_BIAS = args.position_bias
LENGTH_BIAS = args.length_bias

# Two models answer the same 2,000 prompts. "Terse" is slightly BETTER and
# brief; "Wordy" is slightly worse and writes about twice as much.
q_terse = rng.standard_normal(N_PAIRS) + 0.10
q_wordy = rng.standard_normal(N_PAIRS) - 0.10
len_terse = rng.normal(60, 15, N_PAIRS)
len_wordy = rng.normal(120, 30, N_PAIRS)
LEN_MEAN, LEN_SD = 90.0, 38.0
# The judge's private misreading of each answer. It is fixed per answer: ask
# this judge the same question twice and it says the same thing (temperature 0).
see_terse = q_terse + 0.5 * rng.standard_normal(N_PAIRS)
see_wordy = q_wordy + 0.5 * rng.standard_normal(N_PAIRS)


def judge(seen_first, len_first, seen_second, len_second):
    """Returns True where the judge picks the answer shown first."""
    score_first = seen_first + LENGTH_BIAS * (len_first - LEN_MEAN) / LEN_SD + POSITION_BIAS
    score_second = seen_second + LENGTH_BIAS * (len_second - LEN_MEAN) / LEN_SD
    return score_first > score_second


truth_terse = float(np.mean(q_terse > q_wordy))
terse_first = judge(see_terse, len_terse, see_wordy, len_wordy)          # True = Terse wins
wordy_first = judge(see_wordy, len_wordy, see_terse, len_terse)          # True = Wordy wins

print(f"\n  Ground truth (which we only know because this is a simulation):")
print(f"    Terse gives the better answer on {pct(truth_terse).strip()} of prompts.\n")
print(f"  The judge's verdict, by the order the answers were shown in:")
print(f"    Terse shown first  -> judge prefers Terse {pct(terse_first.mean()).strip()}")
print(f"    Wordy shown first  -> judge prefers Terse {pct(1 - wordy_first.mean()).strip()}")
print(f"  Same answers. The verdict moved {100 * (terse_first.mean() - (1 - wordy_first.mean())):.1f} points because of the order alone.")

# THE ORDER-SWAP TEST: judge every pair twice, once in each order.
first_rate = (terse_first.mean() + wordy_first.mean()) / 2      # how often "first" wins
flipped = terse_first & wordy_first | ~terse_first & ~wordy_first   # the verdict followed the seat
print(f"\n  ORDER-SWAP TEST (judge every pair in both orders):")
print(f"    the answer shown first wins {pct(first_rate).strip()} of the time   (a fair judge: 50.0%)")
print(f"    the verdict reverses when the order is swapped on {pct(flipped.mean()).strip()} of pairs")
if first_rate > 0.52:
    print(f"    -> POSITION BIAS DETECTED. No answer key was needed to find it.")
else:
    print(f"    -> no position bias detected.")

# The standard repair: keep only verdicts that survive the swap.
consistent = ~flipped
terse_wins_consistent = float(np.mean(terse_first[consistent]))
print(f"\n  The repair: count a win only when both orders agree ({pct(consistent.mean()).strip()} of pairs).")
print(f"    judge prefers Terse on {pct(terse_wins_consistent).strip()} of those  ->  Wordy 'wins' {pct(1 - terse_wins_consistent).strip()}")
print(f"    ground truth: Terse is better on {pct(truth_terse).strip()}")
if 1 - terse_wins_consistent > 0.55:
    print(f"  The position bias is gone and the verdict is STILL wrong: the judge likes long")
    print(f"  answers, and swapping the order cannot see that. Each bias needs its own test.")

# The test for length bias needs an answer key (or a human): does the judge's
# accuracy depend on whether the better answer happens to be the longer one?
better_is_terse = q_terse > q_wordy
longer_is_better = ~better_is_terse          # Wordy is nearly always the longer one
acc_when_longer_better = float(np.mean((~terse_first & wordy_first)[longer_is_better]))
acc_when_shorter_better = float(np.mean((terse_first & ~wordy_first)[better_is_terse]))
print(f"\n  LENGTH TEST (needs ground truth from humans on a sample):")
print(f"    judge is right {pct(acc_when_longer_better).strip()} of the time when the better answer is the long one")
print(f"    judge is right {pct(acc_when_shorter_better).strip()} of the time when the better answer is the short one")
if acc_when_longer_better - acc_when_shorter_better > 0.10:
    print(f"    -> LENGTH BIAS DETECTED. An unbiased judge would score about the same on both.")
else:
    print(f"    -> no length bias detected.")


# ---------------------------------------------------------------------------
# STEP 7: contamination
#
# If test items were in the training data, the model can answer them from
# memory. Its score then measures recall of the test set, not ability.
# ---------------------------------------------------------------------------
rule("STEP 7: Contamination - leaked test items inflate the score")

TRUE_C = 0.60                 # the contaminated model's real ability
MEMORISED = args.memorised    # chance it reproduces an answer it saw in training
N_CONT = 5000
difficulty_c = draw_items(N_CONT, rng)
honest = answer(SKILL_A, difficulty_c, rng)             # model A again, never saw the test
fresh_c = answer(skill_for(TRUE_C), difficulty_c, rng)  # what C does on unseen items
recall_c = (rng.random(N_CONT) < MEMORISED).astype(float)   # what C does on memorised items
leak_order = rng.permutation(N_CONT)

print(f"\n  Model C has a TRUE accuracy of {pct(TRUE_C).strip()}. Model A (honest, {pct(TRUE_A).strip()}) never saw the test.")
print(f"  Leak a fraction of the {N_CONT:,} test items into C's training data and score both:\n")
print(f"  {'leaked':>8} {'C scores':>10} {'A scores':>10}   leaderboard says")
print("  " + "-" * 50)
crossover = None
for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]:
    leaked = np.zeros(N_CONT, dtype=bool)
    leaked[leak_order[:int(frac * N_CONT)]] = True
    c_marks = np.where(leaked, recall_c, fresh_c)
    says = "C is better  <- wrong" if c_marks.mean() > honest.mean() else "A is better"
    if crossover is None and c_marks.mean() > honest.mean():
        crossover = frac
    print(f"  {100 * frac:>7.0f}% {pct(c_marks.mean()):>10} {pct(honest.mean()):>10}   {says}")
    if frac == 1.0:
        full_leak = c_marks.mean()
print(f"\n  Fully leaked, the {pct(TRUE_C).strip()} model scores {pct(full_leak).strip()}.", end="")
if crossover is not None:
    print(f" It passes the honest model once {100 * crossover:.0f}% has leaked.")
else:
    print(f" It never passes the honest model.")

# The check: score the leaked and the never-seen items separately.
leaked = np.zeros(N_CONT, dtype=bool)
leaked[leak_order[:int(0.3 * N_CONT)]] = True
c_marks = np.where(leaked, recall_c, fresh_c)
print(f"\n  THE CHECK, at 30% leaked: split the test set by whether the item existed before")
print(f"  the model's training cutoff.")
print(f"    items that could have been seen:   C scores {pct(c_marks[leaked].mean()).strip()}")
print(f"    items written afterwards:          C scores {pct(c_marks[~leaked].mean()).strip()}   <- its real ability")
print(f"  A gap like that between old and new items is the signature of contamination.")
print(f"  It is why serious benchmarks keep a private, regularly refreshed test set.")

rule("WHAT THIS LAB SHOWS")
print("""
1. A score is a measurement with an error bar. Report the interval. If you only
   have 20 examples, the honest report is "we cannot tell yet".

2. Error bars shrink like 1/sqrt(n). Half the uncertainty costs four times the
   items; detecting a gap half as big costs four times the items.

3. Compare models on the SAME items and analyse the differences item by item.
   It is the cheapest statistical power you will ever get.

4. An average can go up while something you care about breaks. Keep slices,
   and keep a regression suite of things that used to work.

5. An automatic judge is a model with habits. Audit it before you trust it:
   swap the order, and check it against human labels on a sample.

6. A test set the model has seen measures memory. Hold some items back, write
   new ones, and be suspicious of any score that is too good.

Now try the five experiments at the top of this file. Each is one flag, and
each says what you should see. Start with:   python labs/12_eval_harness.py --seed 1
""")
