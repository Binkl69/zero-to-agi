"""
LAB 01 - THE PERCEPTRON (1958): THE SMALLEST THING THAT LEARNS
==============================================================

Run:   python labs/01_perceptron.py
Needs: nothing but Python. No numpy, no torch.
Time:  well under a second.

WHAT YOU WILL BUILD
-------------------
Frank Rosenblatt's perceptron: a single artificial neuron that looks at a
list of numbers (the "inputs"), multiplies each by a "weight", adds a "bias",
and fires (outputs 1) if the total is positive, otherwise stays quiet (0).

    output = 1  if  w1*x1 + w2*x2 + ... + b > 0   else  0

That is it. The whole of deep learning is millions of these stacked up.
The magic is not the neuron, it is the LEARNING RULE that tunes the weights
from examples, and the first such rule (1958) fits on one line:

    for every mistake:   w  <-  w + lr * (target - prediction) * x
                         b  <-  b + lr * (target - prediction)

Read it as: "if you said 0 but should have said 1, nudge the weights toward
this input so next time the total is bigger; if you said 1 but should have
said 0, nudge them away."

WHAT TO OBSERVE
---------------
1. AND and OR: the error count per epoch drops to 0 in a handful of epochs.
   The ASCII picture shows the learned straight line ("decision boundary")
   separating the 1s from the 0s.
2. XOR: the error count NEVER reaches 0, no matter how long you train.
   Look at the four points: no single straight line can put both 1s on one
   side and both 0s on the other. This is the famous limitation pointed out
   by Minsky & Papert (1969) that helped trigger the first "AI winter".
3. The fix: two layers. With a hidden layer of two neurons (one computing OR,
   one computing NAND) and an output neuron computing AND of those, XOR is
   solved. Depth lets you draw bent boundaries. This is why "deep" learning.

WHAT TO CHANGE (3 experiments)
------------------------------
A. Change LEARNING_RATE to 1.0 or 0.01. For the perceptron rule the learning
   rate only rescales the weights - it still converges (the perceptron
   convergence theorem). Notice that the final weights differ but the LINE
   they draw is the same shape.
B. Change the initial weights (see `Perceptron.__init__`) to something big
   like [5.0, -5.0]. It still converges, just needing more epochs.
   Count them.
C. Try to solve XOR by hand with a single perceptron: edit `w` and `b` in
   `xor_single_layer_attempt()` and re-run. You cannot. Then look at
   `two_layer_xor()` and change the hidden neurons: what happens if both
   hidden neurons compute OR?

COMMON BUGS (things that bite beginners)
----------------------------------------
* Forgetting the bias. Without `b` the line must pass through the origin
  (0,0), so even AND cannot be learned (AND needs the line to sit *between*
  (0,0) and (1,1)).
* Updating weights with `prediction - target` instead of `target - prediction`
  (the sign flips and the model runs away from the answer).
* Using `>=` vs `>` in the step function: both are fine as long as you are
  consistent, but they change which side ties fall on.
* Shuffling: the classic perceptron visits the examples in order every epoch.
  That is fine for 4 points but for real data you should shuffle each epoch.
"""

import sys
import random

# Windows consoles default to cp1252; make printing UTF-8 safe everywhere.
# (We still only print ASCII, so this is belt-and-braces.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

random.seed(8)  # fixed seed = the same run every time (try others: 1, 1958, ...)

LEARNING_RATE = 0.1
MAX_EPOCHS = 20


# ---------------------------------------------------------------------------
# 1. THE MODEL
# ---------------------------------------------------------------------------
class Perceptron:
    """A single neuron with a hard threshold (a 'step' activation)."""

    def __init__(self, n_inputs, learning_rate=LEARNING_RATE):
        # Small random weights. Zero would also work for the perceptron, but
        # random init is the habit you will need later, so start it now.
        self.w = [random.uniform(-0.5, 0.5) for _ in range(n_inputs)]
        self.b = random.uniform(-0.5, 0.5)
        self.lr = learning_rate

    def weighted_sum(self, x):
        # The "pre-activation": how strongly the evidence adds up.
        # zip pairs each weight with its input: w1*x1 + w2*x2 + ...
        return sum(wi * xi for wi, xi in zip(self.w, x)) + self.b

    def predict(self, x):
        # The step function: fire (1) if the evidence is positive, else 0.
        return 1 if self.weighted_sum(x) > 0 else 0

    def train_one_epoch(self, X, Y):
        """One pass over the data with the 1958 learning rule.
        Returns how many examples were misclassified during the pass."""
        errors = 0
        for x, target in zip(X, Y):
            prediction = self.predict(x)
            # (target - prediction) is 0 when we're right, +1 or -1 when wrong.
            # So the rule ONLY changes weights on mistakes - that's its charm.
            err = target - prediction
            if err != 0:
                errors += 1
                for i in range(len(self.w)):
                    self.w[i] += self.lr * err * x[i]
                self.b += self.lr * err
        return errors

    def fit(self, X, Y, max_epochs=MAX_EPOCHS, verbose=True):
        """Train until zero errors or max_epochs. Returns the error history."""
        history = []
        for epoch in range(1, max_epochs + 1):
            errors = self.train_one_epoch(X, Y)
            history.append(errors)
            if verbose:
                print(f"  epoch {epoch:2d}: {errors} error(s)   "
                      f"w = [{self.w[0]:+.3f}, {self.w[1]:+.3f}]  b = {self.b:+.3f}")
            if errors == 0:
                if verbose:
                    print(f"  -> converged after {epoch} epoch(s)")
                break
        return history


# ---------------------------------------------------------------------------
# 2. A LITTLE ASCII PLOTTER
# ---------------------------------------------------------------------------
def draw_decision_boundary(model, X, Y, size=10, lo=-0.5, hi=1.5):
    """Draw a size x size grid over the square [lo, hi] x [lo, hi].

      '.' = region the model calls 0        '+' = region the model calls 1
      '*' = the decision line itself (where w1*x1 + w2*x2 + b == 0)
      '0' / '1' = the training points with their true labels
    """
    step = (hi - lo) / (size - 1)
    rows = []
    for r in range(size):
        # Row 0 is the TOP of the picture, so it gets the LARGEST x2.
        x2 = hi - r * step
        row = []
        for c in range(size):
            x1 = lo + c * step
            row.append("+" if model.predict([x1, x2]) == 1 else ".")
        rows.append(row)

    # Mark the line: for each column find the x2 where the sum is exactly 0.
    #   w1*x1 + w2*x2 + b = 0   =>   x2 = -(w1*x1 + b) / w2
    # We walk along both axes so steep AND shallow lines come out unbroken.
    w1, w2 = model.w
    if abs(w2) > 1e-9:                      # for each column, solve for x2
        for c in range(size):
            x1 = lo + c * step
            x2_line = -(w1 * x1 + model.b) / w2
            r = round((hi - x2_line) / step)
            if 0 <= r < size:
                rows[r][c] = "*"
    if abs(w1) > 1e-9:                      # for each row, solve for x1
        for r in range(size):
            x2 = hi - r * step
            x1_line = -(w2 * x2 + model.b) / w1
            c = round((x1_line - lo) / step)
            if 0 <= c < size:
                rows[r][c] = "*"

    # Overlay the data points last so they are always visible.
    for (x1, x2), label in zip(X, Y):
        r = round((hi - x2) / step)
        c = round((x1 - lo) / step)
        rows[r][c] = str(label)

    print(f"      x2 = {hi:.1f}  ^")
    for r, row in enumerate(rows):
        print("                 | " + " ".join(row))
    print(f"      x2 = {lo:.1f}  +-" + "-" * (2 * size - 1) + f"> x1 from {lo:.1f} to {hi:.1f}")
    print("      legend: '.' predicts 0   '+' predicts 1   '*' the line   0/1 = data")


# ---------------------------------------------------------------------------
# 3. THE DATASETS: truth tables of logic gates
# ---------------------------------------------------------------------------
INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
AND_TARGETS = [0, 0, 0, 1]
OR_TARGETS = [0, 1, 1, 1]
XOR_TARGETS = [0, 1, 1, 0]  # "exactly one of them" - the troublemaker


def train_gate(name, targets):
    print("=" * 70)
    print(f"Training a perceptron on {name}")
    print("=" * 70)
    model = Perceptron(n_inputs=2)
    print(f"  start:      w = [{model.w[0]:+.3f}, {model.w[1]:+.3f}]  b = {model.b:+.3f}")
    history = model.fit(INPUTS, targets)
    print()
    print("  Truth table check:")
    for x, t in zip(INPUTS, targets):
        p = model.predict(x)
        mark = "ok" if p == t else "WRONG"
        print(f"    {x} -> predicted {p}, target {t}   {mark}")
    print()
    draw_decision_boundary(model, INPUTS, targets)
    print()
    return model, history


def xor_single_layer_attempt():
    print("=" * 70)
    print("Training a perceptron on XOR  (watch the error count: it never hits 0)")
    print("=" * 70)
    model = Perceptron(n_inputs=2)
    history = model.fit(INPUTS, XOR_TARGETS, max_epochs=20)
    print()
    if min(history) > 0:
        print(f"  After {len(history)} epochs the best it ever did was "
              f"{min(history)} error(s) in an epoch. It is cycling, not converging.")
    print("  Why? Look at the picture: the 1s sit on one diagonal and the 0s on the")
    print("  other. Any straight line leaves at least one point on the wrong side.")
    print()
    draw_decision_boundary(model, INPUTS, XOR_TARGETS)
    print()
    # Experiment C: try your own line. (Spoiler: there isn't one.)
    # model.w = [1.0, 1.0]; model.b = -0.5   # this is OR, gets (1,1) wrong
    return model, history


# ---------------------------------------------------------------------------
# 4. THE FIX: DEPTH. A two-layer network with hand-set weights solves XOR.
# ---------------------------------------------------------------------------
def step(z):
    return 1 if z > 0 else 0


def two_layer_xor():
    print("=" * 70)
    print("A 2-layer network with HAND-SET weights solves XOR")
    print("=" * 70)
    print("  Idea:  XOR(x1, x2) = AND( OR(x1, x2), NAND(x1, x2) )")
    print("         'at least one is on'  AND  'not both are on'")
    print()
    # Hidden layer: two neurons, each a perceptron looking at the raw inputs.
    # (Weights chosen by hand so the threshold falls in the right place.)
    hidden = [
        {"name": "OR  ", "w": [1.0, 1.0], "b": -0.5},   # fires if x1 + x2 > 0.5
        {"name": "NAND", "w": [-1.0, -1.0], "b": 1.5},  # fires if x1 + x2 < 1.5
    ]
    # Output neuron: an AND of the two hidden outputs.
    out = {"w": [1.0, 1.0], "b": -1.5}                  # fires if h1 + h2 > 1.5

    print("  x1 x2 | h1=OR h2=NAND | out | target")
    print("  ------+--------------+-----+-------")
    all_ok = True
    for x, t in zip(INPUTS, XOR_TARGETS):
        h = [step(sum(wi * xi for wi, xi in zip(n["w"], x)) + n["b"]) for n in hidden]
        y = step(sum(wi * hi for wi, hi in zip(out["w"], h)) + out["b"])
        all_ok &= (y == t)
        print(f"   {x[0]}  {x[1]} |   {h[0]}     {h[1]}     |  {y}  |   {t}   {'ok' if y == t else 'WRONG'}")
    print()
    print("  " + ("XOR solved. " if all_ok else "Something is off. ") +
          "Each hidden neuron draws ONE line; the output neuron combines the")
    print("  two half-planes into a stripe. Two lines can carve out what one cannot.")
    print()
    print("  In Lab 03 we will let a network LEARN weights like these by itself,")
    print("  which needs a smoother neuron (tanh instead of a hard step) and the")
    print("  chain rule (Lab 02) to know which way to nudge each weight.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_gate("AND", AND_TARGETS)
    train_gate("OR", OR_TARGETS)
    xor_single_layer_attempt()
    two_layer_xor()
