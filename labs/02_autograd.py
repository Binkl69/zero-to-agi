"""
LAB 02 - AUTOGRAD: A 150-LINE ENGINE THAT DIFFERENTIATES ANY FORMULA
====================================================================

Run:   python labs/02_autograd.py
Needs: nothing but Python (the `math` module).
Time:  under a second.

WHAT YOU WILL BUILD
-------------------
The heart of PyTorch/JAX/TensorFlow, in miniature: a class `Value` that wraps
a single number and REMEMBERS how it was computed. When you write

    L = (a * b + c) * d

each `*` and `+` creates a new Value that keeps pointers to its parents. That
is a graph. Calling `L.backward()` then walks the graph backwards, applying the
CHAIN RULE at every node, and leaves in every `.grad` the answer to:

    "if I nudged this number up by a tiny bit, how much would L change?"

Those numbers are the *gradients*. Training a neural network is nothing more
than: compute a loss, call backward(), nudge every weight a little in the
direction that lowers the loss, repeat a million times.

This design (a scalar-valued autograd engine) is the one Andrej Karpathy
popularised as "micrograd". Real frameworks do exactly the same thing with
tensors instead of single numbers, so the code is ~1000x faster - but the
*idea* is identical.

WHAT TO OBSERVE
---------------
1. The worked example prints every node with its value and gradient. Check a
   couple by hand: e.g. dL/dd should equal the value of (a*b + c).
2. The finite-difference check: for a random messy expression we compare our
   analytic gradient with the "poke it and see" numerical gradient
   (f(x+h) - f(x-h)) / 2h. They agree to ~1e-6. That check is how you debug
   ANY gradient code, including in PyTorch.
3. The graph printout: nodes indented under the node they feed into. A neural
   network is just a very big one of these.

WHAT TO CHANGE (3 experiments)
------------------------------
A. In `worked_example()` change the values of a, b, c, d and predict the six
   gradients on paper BEFORE running. (Rule: dL/dx = (how L depends on the
   node above x) * (how that node depends on x).)
B. Break the chain rule on purpose: in `__mul__` change `other.data * out.grad`
   to just `other.data` (forgetting to multiply by the upstream gradient).
   Run again - the finite-difference check will FAIL. This is the single most
   common bug when people implement backprop by hand.
C. Add a new operation, e.g. `sigmoid` = 1/(1+exp(-x)). You can either compose
   it from existing ops (it just works, no new backward needed) or write a
   dedicated `_backward` using the identity s' = s*(1-s). Verify with the
   finite-difference checker.

COMMON BUGS
-----------
* Using `=` instead of `+=` when accumulating gradients. If a node is used
  twice (e.g. `x * x`, or a weight shared by two neurons), both paths must
  ADD their contributions. Try `y = x*x` with `=`: you get x instead of 2x.
* Forgetting to zero gradients between backward() calls (frameworks make you
  call `optimizer.zero_grad()` for exactly this reason).
* Walking the graph in the wrong order. A node's gradient is only complete
  once EVERY node that uses it has passed its share back. That is what the
  topological sort guarantees.
* Numerical gradient with h too big (inaccurate) or too small (floating point
  noise). 1e-5 to 1e-6 is the sweet spot for float64.
"""

import sys
import math
import random

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. THE VALUE CLASS
# ---------------------------------------------------------------------------
class Value:
    """A number that remembers where it came from, so it can be differentiated.

    Attributes
    ----------
    data      : the actual float
    grad      : dOutput/dThis, filled in by backward(). Starts at 0.
    _backward : a tiny function that pushes this node's grad to its parents
    _prev     : the set of parent Values this one was computed from
    _op       : a string naming the operation ('+', '*', 'tanh', ...) - only
                for pretty-printing
    label     : optional human name, also only for printing
    """

    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label")

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None   # leaf nodes have nothing to push back
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        name = f"{self.label}=" if self.label else ""
        return f"Value({name}{self.data:.4f}, grad={self.grad:.4f})"

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _wrap(x):
        # Lets you write `v + 3` : plain numbers are promoted to Values.
        return x if isinstance(x, Value) else Value(x)

    # -- the arithmetic ops. Each one does TWO things: ---------------------
    #    (1) compute the forward value, and
    #    (2) define how gradient flows back to the inputs (the local
    #        derivative TIMES the gradient arriving from above = chain rule).
    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1. So each parent gets the whole
            # upstream gradient. NOTE the +=: gradients ACCUMULATE, because a
            # node may feed into several others.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a  -> "the OTHER one's value".
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        # Only constant exponents (x**2, x**-1, x**0.5). That covers division.
        assert isinstance(exponent, (int, float)), "only numeric exponents"
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            # d(x^n)/dx = n * x^(n-1)   (the power rule from school)
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # d(e^x)/dx = e^x, which we already computed: it's out.data
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        assert self.data > 0, "log needs a positive input"
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            # d(ln x)/dx = 1/x
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh x)/dx = 1 - tanh(x)^2. Squashes big inputs: near |x|>3
            # the derivative is ~0 and learning stalls ("saturation").
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # Gradient passes through unchanged if the input was positive,
            # and is blocked completely otherwise. (At exactly 0 we say 0.)
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # -- everything else is built from the ops above (free gradients!) ------
    def __neg__(self):            # -x
        return self * -1.0

    def __sub__(self, other):     # x - y  ==  x + (-y)
        return self + (-self._wrap(other))

    def __truediv__(self, other):  # x / y  ==  x * y^-1
        return self * (self._wrap(other) ** -1)

    # -- "reflected" versions so `3 * v`, `3 + v`, `3 - v`, `3 / v` work ----
    # Python calls these when the LEFT operand is a plain number.
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):     # 3 - v  ==  (-v) + 3
        return (-self) + other

    def __rtruediv__(self, other):  # 3 / v  ==  3 * v^-1
        return self._wrap(other) * (self ** -1)

    # -- BACKPROPAGATION ----------------------------------------------------
    def backward(self):
        """Fill .grad on every node this Value depends on."""
        # Step 1: order the nodes so that every node comes AFTER all the nodes
        # it depends on ("topological order"). Depth-first search does this.
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)   # appended only after all its parents
        build(self)

        # Step 2: dL/dL = 1 (the output changes by exactly as much as itself).
        self.grad = 1.0

        # Step 3: visit nodes from the output backwards. By the time we reach a
        # node, every node that consumed it has already added its share to
        # .grad, so the value is final and safe to push further back.
        for v in reversed(topo):
            v._backward()

    # Handy for experiments: forget all gradients.
    def zero_grad_graph(self):
        for v in _all_nodes(self):
            v.grad = 0.0


def _all_nodes(root):
    nodes, seen = [], set()

    def walk(v):
        if v not in seen:
            seen.add(v)
            nodes.append(v)
            for c in v._prev:
                walk(c)
    walk(root)
    return nodes


# ---------------------------------------------------------------------------
# 2. PRINT THE GRAPH AS INDENTED TEXT
# ---------------------------------------------------------------------------
def print_graph(v, indent=0, _seen=None):
    """Prints the computation graph as a tree: each node's children are
    indented underneath it. Shared nodes are printed once, then referenced."""
    if _seen is None:
        _seen = set()
    pad = "    " * indent
    name = v.label or "(unnamed)"
    op = f"  <- {v._op}" if v._op else "  (leaf)"
    if id(v) in _seen and v._prev:
        print(f"{pad}{name} = {v.data:.4f}  (already shown above)")
        return
    _seen.add(id(v))
    print(f"{pad}{name} = {v.data:.4f}   grad = {v.grad:.4f}{op}")
    # Children are stored in a set (unordered); sort by label for stable output.
    for child in sorted(v._prev, key=lambda c: (c.label, c.data)):
        print_graph(child, indent + 1, _seen)


# ---------------------------------------------------------------------------
# 3. THE FINITE-DIFFERENCE CHECK  (how to debug ANY gradient code)
# ---------------------------------------------------------------------------
def finite_difference_check(n_trials=5, h=1e-6, tol=1e-4, seed=0, verbose=True):
    """Build random expressions, compare backward() against numeric slopes.

    Numeric slope of f at x:  (f(x+h) - f(x-h)) / (2h)   ("central difference")
    If our _backward functions are right, the two agree to ~1e-7 or better.
    """
    rng = random.Random(seed)

    def expression(a, b, c):
        # A deliberately ugly formula that uses EVERY op we defined.
        # (Inputs are kept positive-ish where log needs it.)
        t = (a * b - c / 2.0 + 3.0) ** 2
        u = (t * 0.1).tanh() + (b - a).relu()
        w = (a.exp() * 0.05 + 1.0).log() / (c ** 2 + 1.0)
        return u * w + 2.0 - a ** 3 / 7.0 + (1.0 / (b + 5.0))

    all_ok = True
    for trial in range(n_trials):
        raw = [rng.uniform(-2.0, 2.0) for _ in range(3)]

        # analytic gradients via backward()
        vals = [Value(x, label=n) for x, n in zip(raw, "abc")]
        L = expression(*vals)
        L.backward()

        for i, name in enumerate("abc"):
            # numeric gradient: nudge ONE input, recompute the whole thing
            plus = list(raw); plus[i] += h
            minus = list(raw); minus[i] -= h
            f_plus = expression(*[Value(x) for x in plus]).data
            f_minus = expression(*[Value(x) for x in minus]).data
            numeric = (f_plus - f_minus) / (2 * h)
            analytic = vals[i].grad
            diff = abs(numeric - analytic)
            ok = diff < tol
            all_ok &= ok
            if verbose:
                print(f"  trial {trial} d/d{name}: analytic {analytic:+.6f}  "
                      f"numeric {numeric:+.6f}  |diff| {diff:.2e}  {'ok' if ok else 'MISMATCH'}")
    assert all_ok, "Analytic and numeric gradients disagree - a _backward is wrong!"
    return all_ok


# ---------------------------------------------------------------------------
# 4. THE WORKED EXAMPLE
# ---------------------------------------------------------------------------
def worked_example():
    print("=" * 70)
    print("Worked example:  L = (a*b + c) * d")
    print("=" * 70)
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    d = Value(-2.0, label="d")
    e = a * b;   e.label = "e"       # e = -6
    f = e + c;   f.label = "f"       # f =  4
    L = f * d;   L.label = "L"       # L = -8

    print(f"  forward:  e = a*b = {e.data:g},  f = e+c = {f.data:g},  L = f*d = {L.data:g}")
    print()
    L.backward()

    print("  Chain rule, one hop at a time (read bottom-up):")
    print("    dL/dL = 1                              (a thing changes as much as itself)")
    print(f"    dL/dd = f            = {d.grad:+g}          (L = f*d, so nudging d scales by f)")
    print(f"    dL/df = d            = {f.grad:+g}")
    print(f"    dL/dc = dL/df * df/dc = {f.grad:+g} * 1   = {c.grad:+g}")
    print(f"    dL/de = dL/df * df/de = {f.grad:+g} * 1   = {e.grad:+g}")
    print(f"    dL/da = dL/de * de/da = {e.grad:+g} * b = {e.grad:+g} * {b.data:g} = {a.grad:+g}")
    print(f"    dL/db = dL/de * de/db = {e.grad:+g} * a = {e.grad:+g} * {a.data:g} = {b.grad:+g}")
    print()
    print("  Meaning: increasing `a` by 0.01 should raise L by about 0.01*6 = 0.06.")
    L2 = ((a.data + 0.01) * b.data + c.data) * d.data
    print(f"  Check:   L(a=2.01) = {L2:.4f},  difference from L = {L2 - L.data:+.4f}")
    print()
    print("  The graph (each node's inputs indented beneath it):")
    print_graph(L, indent=2)
    print()

    # A second, more 'neural' example: one neuron with a tanh activation.
    print("-" * 70)
    print("A single neuron:  out = tanh(x1*w1 + x2*w2 + b)")
    print("-" * 70)
    x1, x2 = Value(2.0, label="x1"), Value(0.0, label="x2")
    w1, w2 = Value(-3.0, label="w1"), Value(1.0, label="w2")
    bias = Value(6.8813735870195432, label="b")   # chosen so the pre-activation is 0.8814
    n = x1 * w1 + x2 * w2 + bias; n.label = "n"
    out = n.tanh(); out.label = "out"
    out.backward()
    print(f"  n = {n.data:.4f}   out = tanh(n) = {out.data:.4f}")
    print(f"  d out/d n  = 1 - out^2 = {n.grad:.4f}")
    print(f"  d out/d w1 = d out/d n * x1 = {n.grad:.4f} * {x1.data:g} = {w1.grad:.4f}")
    print(f"  d out/d w2 = d out/d n * x2 = {n.grad:.4f} * {x2.data:g} = {w2.grad:.4f}"
          "   <- x2 is 0, so w2 gets NO learning signal from this example")
    print(f"  d out/d x1 = d out/d n * w1 = {n.grad:.4f} * {w1.data:g} = {x1.grad:.4f}")
    print()


def gradient_accumulation_demo():
    print("-" * 70)
    print("Why gradients must ACCUMULATE (+=): y = x * x")
    print("-" * 70)
    x = Value(3.0, label="x")
    y = x * x; y.label = "y"
    y.backward()
    print(f"  x = 3, y = x*x = {y.data:g}.  dy/dx should be 2x = 6.  We got: {x.grad:g}")
    print("  x appears TWICE as an input to the same '*' node. Each appearance")
    print("  contributes 3, and += adds them. With '=' you'd get 3 - silently wrong.")
    print()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    worked_example()
    gradient_accumulation_demo()

    print("=" * 70)
    print("Finite-difference check on a random expression using every op")
    print("=" * 70)
    finite_difference_check()
    print("  All analytic gradients match numeric ones to within 1e-4. Engine OK.")
    print()
    print("Next: Lab 03 stacks these Values into neurons and layers and trains them.")
