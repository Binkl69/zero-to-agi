"""
LAB 03 - A NEURAL NETWORK THAT LEARNS: NEURON -> LAYER -> MLP
=============================================================

Run:   python labs/03_mlp.py                (about 20-40 s on a laptop CPU)
       python labs/03_mlp.py --lr 10        (experiment: learning rate too high)
       python labs/03_mlp.py --hidden 1     (experiment: too few hidden units)
       python labs/03_mlp.py --steps 200 --hidden 16
Needs: labs/02_autograd.py (loaded by path - no install needed).
Time:  ~20-40 s. Pure Python is slow; that is the point of Lab 05/06 (numpy,
       torch). Here every one of the ~10,000 multiplications per step is a
       Python object so you can see all the moving parts.

WHAT YOU WILL BUILD
-------------------
Using the `Value` from Lab 02 as the only building block:

    Neuron : y = tanh( w . x + b )        (a perceptron with a smooth step)
    Layer  : a list of Neurons that all read the same input
    MLP    : Layers stacked, output of one is input of the next
             ("Multi-Layer Perceptron" - the 1986 answer to the 1969 XOR problem)

and the training loop that EVERY neural network uses, no exceptions:

    for step in range(N):
        loss = how wrong are we on the data?      # forward pass
        loss.backward()                           # gradients (Lab 02)
        for p in params: p.data -= lr * p.grad    # gradient descent step
        for p in params: p.grad = 0               # clean up

WHAT TO OBSERVE
---------------
1. XOR - the thing a single perceptron could NOT learn in Lab 01 - is learned
   in a couple of hundred steps by a 2-3-1 network. Nobody hand-set the
   weights this time.
2. "Two moons": two interleaved crescents. Watch the loss fall, then look at
   the ASCII decision boundary - it BENDS around the moons. Every bend is
   paid for by a hidden neuron.
3. The loss doesn't fall smoothly: it stalls, then drops. That's normal.

WHAT TO CHANGE (3 experiments)
------------------------------
A. --lr 3.0 : learning rate too high. The loss stops falling smoothly and
   BOUNCES (0.31 -> 0.47 -> 0.11 -> 0.28 ...): each step overshoots the
   valley and lands on the far wall. Then --lr 10 : the loss jumps to
   exactly 2.0 and freezes at 50% accuracy. Why 2.0? Every weight got so
   huge that every tanh is pinned at +1 or -1 ("saturated"); half the
   points are maximally wrong (error 2^2 = 4) and, worse, tanh's gradient
   at +-1 is ~0, so NOTHING can ever move again. In a network with an
   unbounded output the same overshoot shows up as loss -> inf/NaN.
   Then --lr 0.01: converges, but painfully slowly. Finding the biggest lr
   that is still stable is half of practical ML.
B. --hidden 1 or --hidden 2 : too few hidden units. One tanh unit can only
   draw one line-ish boundary, so it UNDERFITS: accuracy stalls ~90% and
   the picture shows a single gentle curve cutting through both moons.
   Then --hidden 32 and see the boundary get smoother/wigglier.
C. Change `tanh` to `relu` in `Neuron.__call__` (keep tanh on the OUTPUT
   neuron so it stays in [-1, 1]). ReLU nets train differently: faster
   early, and the boundary becomes piecewise-straight (look closely).

COMMON BUGS
-----------
* Forgetting to zero the gradients each step: they accumulate (Lab 02!) and
  the model spirals off. This is `optimizer.zero_grad()` in PyTorch.
* Stepping in the + direction: `p.data += lr * p.grad` climbs the loss.
* Initialising all weights to the same value: every neuron in a layer then
  computes the same thing, gets the same gradient, and stays identical
  forever ("symmetry"). Random init breaks the tie.
* Targets in {0,1} with a tanh output in [-1,1]: the network can never output
  exactly 1 or reach 0 comfortably. Use {-1,+1} for tanh (we do), or a
  sigmoid output for {0,1}.
"""

import sys
import math
import random
import argparse
import importlib.util
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# 0. IMPORT `Value` FROM LAB 02
# ---------------------------------------------------------------------------
# `import 02_autograd` is a syntax error (module names can't start with a
# digit), so we load the file by path. This is all importlib is doing: read
# the file, run it as a module, hand it back. Lab 02 guards its demo with
# `if __name__ == "__main__"`, so importing it runs no demo output.
_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("autograd", _here / "02_autograd.py")
autograd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(autograd)
Value = autograd.Value


# ---------------------------------------------------------------------------
# 1. THE MODEL: Neuron -> Layer -> MLP
# ---------------------------------------------------------------------------
class Neuron:
    """One unit: tanh(w . x + b). `activation=None` gives a plain linear unit."""

    def __init__(self, n_in, activation="tanh"):
        # Small random weights. Scaling by 1/sqrt(n_in) keeps the pre-activation
        # roughly unit-sized however many inputs there are (otherwise a neuron
        # with 100 inputs starts deep in tanh's flat, gradient-free region).
        scale = 1.0 / math.sqrt(n_in)
        self.w = [Value(random.uniform(-1, 1) * scale) for _ in range(n_in)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        # w . x + b, built from Value ops so it is differentiable.
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        if self.activation == "tanh":
            return act.tanh()
        if self.activation == "relu":
            return act.relu()
        return act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """n_out neurons that all read the same n_in inputs."""

    def __init__(self, n_in, n_out, activation="tanh"):
        self.neurons = [Neuron(n_in, activation) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]   # always a list, even for 1 neuron

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """sizes = [n_in, hidden1, hidden2, ..., n_out]. Hidden layers use tanh
    (or relu), the last layer uses tanh so outputs land in [-1, 1]."""

    def __init__(self, sizes, hidden_activation="tanh"):
        self.layers = []
        for i in range(len(sizes) - 1):
            last = (i == len(sizes) - 2)
            self.layers.append(Layer(sizes[i], sizes[i + 1],
                                     activation="tanh" if last else hidden_activation))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)         # output of one layer is input to the next
        return x[0] if len(x) == 1 else x   # unwrap a single output Value

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ---------------------------------------------------------------------------
# 2. DATA
# ---------------------------------------------------------------------------
XOR_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_Y = [-1.0, 1.0, 1.0, -1.0]      # -1/+1 instead of 0/1 because tanh output


def make_two_moons(n=100, noise=0.12, seed=3):
    """Two interleaving half-circles. Class +1 is the upper moon, -1 the lower.
    (This is what sklearn.datasets.make_moons draws; we write it ourselves.)"""
    rng = random.Random(seed)
    X, Y = [], []
    for i in range(n):
        t = rng.uniform(0, math.pi)         # angle along the half circle
        if i % 2 == 0:                      # upper moon, centred at (0, 0)
            x, y, label = math.cos(t), math.sin(t), 1.0
        else:                               # lower moon, shifted right & down
            x, y, label = 1.0 - math.cos(t), 0.5 - math.sin(t), -1.0
        X.append([x + rng.gauss(0, noise), y + rng.gauss(0, noise)])
        Y.append(label)
    return X, Y


# ---------------------------------------------------------------------------
# 3. TRAINING LOOP
# ---------------------------------------------------------------------------
def mse_loss(model, X, Y):
    """Mean squared error between predictions and targets, as a Value."""
    total = Value(0.0)
    for x, y in zip(X, Y):
        pred = model(x)
        total = total + (pred - y) ** 2
    return total * (1.0 / len(X))


def accuracy(model, X, Y):
    correct = sum(1 for x, y in zip(X, Y) if (model(x).data > 0) == (y > 0))
    return correct / len(X)


def train(model, X, Y, steps, lr, print_every, label):
    params = model.parameters()
    print(f"  [{label}] {len(params)} parameters, {len(X)} examples, lr={lr}, {steps} steps")
    history = []
    for step in range(steps + 1):
        # ---- forward: build the whole computation graph for this batch ----
        loss = mse_loss(model, X, Y)
        history.append(loss.data)

        # ---- diverged? stop early so the experiment is readable ------------
        if math.isnan(loss.data) or loss.data > 1e6:
            print(f"  step {step:4d}  loss = {loss.data}   <- DIVERGED (lr too high)")
            return history

        if step % print_every == 0 or step == steps:
            print(f"  step {step:4d}  loss = {loss.data:.4f}   train acc = {accuracy(model, X, Y):.0%}")

        if step == steps:
            break

        # ---- backward: gradients of loss w.r.t. every parameter ----------
        for p in params:
            p.grad = 0.0                # (bug #1 if you forget this)
        loss.backward()

        # ---- update: a step DOWNHILL ---------------------------------------
        for p in params:
            p.data -= lr * p.grad       # minus: descend, don't climb
    return history


# ---------------------------------------------------------------------------
# 4. ASCII DECISION BOUNDARY
# ---------------------------------------------------------------------------
def draw_boundary(model, X, Y, cols=40, rows=20, xlim=(-1.6, 2.6), ylim=(-1.1, 1.6)):
    """'#' = model says +1, '.' = model says -1; 'o' / 'x' are the data points."""
    x0, x1 = xlim
    y0, y1 = ylim
    grid = []
    for r in range(rows):
        y = y1 - (y1 - y0) * r / (rows - 1)          # top row = largest y
        row = []
        for c in range(cols):
            x = x0 + (x1 - x0) * c / (cols - 1)
            row.append("#" if model([x, y]).data > 0 else ".")
        grid.append(row)
    for (x, y), label in zip(X, Y):
        r = round((y1 - y) / (y1 - y0) * (rows - 1))
        c = round((x - x0) / (x1 - x0) * (cols - 1))
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = "o" if label > 0 else "x"
    print("   +" + "-" * cols + "+")
    for row in grid:
        print("   |" + "".join(row) + "|")
    print("   +" + "-" * cols + "+")
    print("    '#' predicted +1   '.' predicted -1   'o' true +1   'x' true -1")


def sparkline(values, width=60):
    """A one-line ASCII chart of the loss curve (plain ASCII, any console)."""
    if len(values) > width:                              # downsample
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    lo, hi = min(values), max(values)
    chars = " .:-=+*#%@"
    out = "".join(chars[int((v - lo) / (hi - lo + 1e-12) * (len(chars) - 1))] for v in values)
    return f"  loss curve [{lo:.3f} .. {hi:.3f}]: |{out}|"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=8, help="hidden units per layer (try 1, 2, 32)")
    ap.add_argument("--lr", type=float, default=0.2, help="learning rate (try 3.0 or 0.01)")
    ap.add_argument("--steps", type=int, default=250, help="gradient steps on two moons")
    ap.add_argument("--activation", default="tanh", choices=["tanh", "relu"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    # ---------------- XOR ----------------
    print("=" * 70)
    print("Part 1: learning XOR with a 2 -> 3 -> 1 network")
    print("=" * 70)
    xor_net = MLP([2, 3, 1])
    train(xor_net, XOR_X, XOR_Y, steps=300, lr=0.3, print_every=50, label="XOR")
    print()
    print("  Learned truth table:")
    for x, y in zip(XOR_X, XOR_Y):
        out = xor_net(x).data
        print(f"    {x} -> {out:+.3f}  (target {y:+.0f})  {'ok' if (out > 0) == (y > 0) else 'WRONG'}")
    print()
    draw_boundary(xor_net, XOR_X, XOR_Y, cols=30, rows=12, xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))
    print()

    # ---------------- TWO MOONS ----------------
    print("=" * 70)
    print(f"Part 2: two moons with a 2 -> {args.hidden} -> {args.hidden} -> 1 network"
          f" ({args.activation} hidden units)")
    print("=" * 70)
    X, Y = make_two_moons(n=100)
    net = MLP([2, args.hidden, args.hidden, 1], hidden_activation=args.activation)
    hist = train(net, X, Y, steps=args.steps, lr=args.lr, print_every=25, label="moons")
    print()
    print(sparkline(hist))
    print(f"  final train accuracy: {accuracy(net, X, Y):.0%}")
    print()
    draw_boundary(net, X, Y)
    print()
    print("Try:  --lr 10 (diverges)   --hidden 1 (underfits)   --activation relu")
