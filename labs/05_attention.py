"""
LAB 05 - ATTENTION AND THE TRANSFORMER BLOCK, IN NUMPY
======================================================

Run:   python labs/05_attention.py
Needs: numpy.
Time:  ~1 s.

WHAT YOU WILL BUILD
-------------------
Every piece of a GPT forward pass, as plain numpy functions with the shape of
every array written next to it:

    softmax                          turn scores into probabilities
    scaled_dot_product_attention     the 2017 formula: softmax(QK^T/sqrt(d)) V
    causal_mask                      "you may not look at the future"
    multi_head_attention             split channels into heads, attend, rejoin
    layer_norm, gelu, mlp            the other half of a block
    transformer_block                x + attn(ln(x)); x + mlp(ln(x))
    gpt_forward                      embeddings -> N blocks -> logits
    gpt2_param_count                 reproduces GPT-2 small's 124,439,808

THE ONE IDEA
------------
A token's vector needs information from OTHER tokens ("it" needs to know
what "it" refers to). Attention is a soft lookup table:

    every token publishes a KEY   ("here is what I contain")
    every token asks with a QUERY ("here is what I'm looking for")
    every token offers a VALUE    ("here is what you get if you pick me")

    score(i, j) = query_i . key_j            how relevant is token j to token i?
    weights     = softmax over j             turn scores into a distribution
    output_i    = sum_j weights[i, j] * value_j     a weighted mix of values

Q, K, V are just three linear projections of the same input. Everything
else - multiple heads, scaling by sqrt(d), the causal mask - is bookkeeping.

WHAT TO OBSERVE
---------------
1. The 3-token worked example prints the attention matrix. Rows sum to 1.
   With the causal mask the upper triangle is exactly 0: token 1 sees only
   itself, token 2 sees tokens 1-2, token 3 sees everything. That is what
   makes the model usable for generation - every position is trained as if
   the future did not exist.
2. Every shape is asserted. Read the reshapes in multi_head_attention slowly;
   they are where 90% of real transformer bugs live.
3. The parameter count: 124,439,808 exactly, broken down by part. Notice that
   the token embedding (38.6M) is a THIRD of the model, and the MLPs are
   two thirds of each block. Attention itself is the small part.

WHAT TO CHANGE (3 experiments)
------------------------------
A. In `worked_example()` change the query of token 3 to point at token 1
   (make it a copy of key_1) and watch the weights move. Then remove the
   1/sqrt(d) scaling with d=64 random vectors: the softmax saturates to a
   one-hot and gradients would vanish. That is why the sqrt(d) is there.
B. gpt2_param_count(n_layer=24, d=1024, n_head=16) -> GPT-2 medium (~355M).
   n_layer=48, d=1600, n_head=25 -> GPT-2 XL (~1.5B). Where did the
   parameters go as the model grew? (Hint: the embedding share shrinks.)
C. Set causal=False in gpt_forward. Attention now leaks the future: the
   model would trivially "predict" the next token by looking at it. This is
   the difference between GPT (decoder, causal) and BERT (encoder, not).

COMMON BUGS
-----------
* Softmax over the wrong axis. Weights must sum to 1 over the KEYS (last
  axis), so each query gets a distribution over what to attend to.
* Masking with 0 instead of -inf. Adding 0 does nothing; you need the
  masked scores to become exp(-inf) = 0 AFTER the softmax.
* Forgetting to subtract the max in softmax: exp(1000) overflows to inf.
* Transposing the wrong axes when splitting heads. (B, T, C) must become
  (B, n_head, T, head_dim) - heads and time swap places - so that the
  matmul batches over heads.
* LayerNorm over the batch/time axis instead of the channel axis.
"""

import sys
import math
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

np.set_printoptions(precision=3, suppress=True, linewidth=120)


# ---------------------------------------------------------------------------
# 1. SOFTMAX
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    """Turn arbitrary scores into probabilities along `axis`.
    Subtracting the max changes nothing mathematically (it cancels) but
    prevents exp() from overflowing on big inputs."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# 2. SCALED DOT-PRODUCT ATTENTION
# ---------------------------------------------------------------------------
def causal_mask(T):
    """(T, T) boolean, True where attention is ALLOWED: row i may look at
    columns j <= i (itself and the past). np.tril keeps the lower triangle."""
    return np.tril(np.ones((T, T), dtype=bool))


def scaled_dot_product_attention(Q, K, V, mask=None):
    """The formula from 'Attention Is All You Need' (2017).

    Q : (..., T_q, d_k)   queries   - one per output position
    K : (..., T_k, d_k)   keys      - one per input position
    V : (..., T_k, d_v)   values    - one per input position
    mask : (T_q, T_k) bool, True = may attend. None = attend to everything.

    Returns  out (..., T_q, d_v)  and  weights (..., T_q, T_k).
    The leading '...' can be (batch, heads) - numpy matmul batches over it.
    """
    d_k = Q.shape[-1]
    # Every query dotted with every key. swapaxes turns K (.., T_k, d_k) into
    # (.., d_k, T_k) so the matmul contracts over d_k and leaves (T_q, T_k).
    scores = Q @ K.swapaxes(-1, -2)                     # (..., T_q, T_k)
    # Why divide by sqrt(d_k)? A dot product of d_k random unit-variance
    # numbers has variance d_k. Without scaling, big d_k = huge scores =
    # softmax becomes a hard argmax and the gradients die.
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        # Put -inf-ish where attention is forbidden. After softmax that is 0.
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)                  # rows sum to 1
    out = weights @ V                                   # (..., T_q, d_v)
    return out, weights


# ---------------------------------------------------------------------------
# 3. MULTI-HEAD ATTENTION with every reshape spelled out
# ---------------------------------------------------------------------------
def multi_head_attention(x, p, n_head, causal=True):
    """x : (B, T, C)  batch of B sequences, T tokens each, C channels.
    p : dict with W_qkv (C, 3C), b_qkv (3C,), W_o (C, C), b_o (C,)
    Returns (B, T, C) and the weights (B, n_head, T, T)."""
    B, T, C = x.shape
    assert C % n_head == 0, "channels must split evenly across heads"
    hd = C // n_head                                    # head_dim

    # One matmul produces Q, K and V for all heads at once (that is what
    # GPT-2's 'c_attn' layer is): (B, T, C) @ (C, 3C) -> (B, T, 3C)
    qkv = x @ p["W_qkv"] + p["b_qkv"]
    q, k, v = np.split(qkv, 3, axis=-1)                 # 3 x (B, T, C)

    # Split the C channels into n_head groups of hd, then move the head axis
    # in front of T so that matmul treats (B, n_head) as batch dimensions
    # and each head attends independently over its own (T, hd) slice.
    #   (B, T, C) -> (B, T, n_head, hd) -> (B, n_head, T, hd)
    q = q.reshape(B, T, n_head, hd).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_head, hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_head, hd).transpose(0, 2, 1, 3)
    assert q.shape == (B, n_head, T, hd)

    mask = causal_mask(T) if causal else None           # broadcasts over (B, n_head)
    out, weights = scaled_dot_product_attention(q, k, v, mask)
    assert out.shape == (B, n_head, T, hd)
    assert weights.shape == (B, n_head, T, T)

    # Undo the split: put T back before heads, then flatten heads*hd = C.
    #   (B, n_head, T, hd) -> (B, T, n_head, hd) -> (B, T, C)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

    # Final projection lets the heads' outputs mix ('c_proj' in GPT-2).
    return out @ p["W_o"] + p["b_o"], weights


# ---------------------------------------------------------------------------
# 4. THE REST OF THE BLOCK: LayerNorm, GELU, MLP
# ---------------------------------------------------------------------------
def layer_norm(x, gamma, beta, eps=1e-5):
    """Normalise EACH token's C-vector to mean 0 / var 1, then let the model
    rescale (gamma) and shift (beta). Keeps activations in a sane range so
    depth doesn't blow them up. Note: statistics are over the LAST axis
    (channels), per token - never over the batch."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def gelu(x):
    """Gaussian Error Linear Unit: a smooth ReLU used by GPT-2/BERT.
    This is the tanh approximation GPT-2 uses."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def mlp(x, p):
    """Position-wise feed-forward: expand C -> 4C, nonlinearity, project back.
    Applied to every token independently - this is where 'facts' are stored."""
    h = gelu(x @ p["W_fc"] + p["b_fc"])                # (B, T, 4C)
    return h @ p["W_proj"] + p["b_proj"]               # (B, T, C)


def transformer_block(x, p, n_head, causal=True):
    """One GPT-2 block ('pre-norm' variant):
         x = x + attn( ln1(x) )
         x = x + mlp ( ln2(x) )
    The `x +` are RESIDUAL connections: each sub-layer only has to learn a
    small correction, and gradients have a straight highway back to the
    input. Without them, 12+ layers would not train."""
    a, _ = multi_head_attention(layer_norm(x, p["ln1_g"], p["ln1_b"]), p, n_head, causal)
    x = x + a
    x = x + mlp(layer_norm(x, p["ln2_g"], p["ln2_b"]), p)
    return x


def init_block_params(C, rng, std=0.02):
    """Random weights with GPT-2's shapes (std 0.02 like GPT-2's init)."""
    return {
        "ln1_g": np.ones(C), "ln1_b": np.zeros(C),
        "W_qkv": rng.normal(0, std, (C, 3 * C)), "b_qkv": np.zeros(3 * C),
        "W_o": rng.normal(0, std, (C, C)), "b_o": np.zeros(C),
        "ln2_g": np.ones(C), "ln2_b": np.zeros(C),
        "W_fc": rng.normal(0, std, (C, 4 * C)), "b_fc": np.zeros(4 * C),
        "W_proj": rng.normal(0, std, (4 * C, C)), "b_proj": np.zeros(C),
    }


# ---------------------------------------------------------------------------
# 5. A WHOLE (RANDOM) GPT FORWARD PASS
# ---------------------------------------------------------------------------
def gpt_forward(tokens, params, n_head, causal=True):
    """tokens : (B, T) integer ids   ->   logits (B, T, vocab)
    logits[b, t] scores every vocab entry as the NEXT token after position t."""
    B, T = tokens.shape
    wte, wpe = params["wte"], params["wpe"]            # (vocab, C), (ctx, C)
    assert T <= wpe.shape[0], "sequence longer than the context window"
    x = wte[tokens] + wpe[np.arange(T)]                # (B, T, C): what + where
    for block in params["blocks"]:
        x = transformer_block(x, block, n_head, causal)
    x = layer_norm(x, params["lnf_g"], params["lnf_b"])
    # Tied output head: reuse the token embedding matrix, transposed.
    # (B, T, C) @ (C, vocab) -> (B, T, vocab). No extra parameters.
    return x @ wte.T


# ---------------------------------------------------------------------------
# 6. PARAMETER COUNT: reproduce GPT-2 small exactly
# ---------------------------------------------------------------------------
def gpt2_param_count(n_layer=12, d=768, n_head=12, vocab=50257, ctx=1024):
    """Count parameters of a GPT-2-style model with tied embeddings.
    Returns (total, breakdown dict). n_head does not change the count -
    heads only split C, they do not add weights."""
    per_block = {
        "ln1 (gamma, beta)":        2 * d,
        "attn qkv  W + b":          d * 3 * d + 3 * d,
        "attn proj W + b":          d * d + d,
        "ln2 (gamma, beta)":        2 * d,
        "mlp fc    W + b":          d * 4 * d + 4 * d,
        "mlp proj  W + b":          4 * d * d + d,
    }
    block_total = sum(per_block.values())
    breakdown = {
        "token embedding wte (vocab x d)": vocab * d,
        "position embedding wpe (ctx x d)": ctx * d,
        f"{n_layer} blocks x {block_total:,} each": n_layer * block_total,
        "final layer norm": 2 * d,
        "output head (tied to wte)": 0,
    }
    total = sum(breakdown.values())
    return total, breakdown, per_block


# ---------------------------------------------------------------------------
# 7. WORKED EXAMPLE: 3 tokens, 4 dimensions, by hand
# ---------------------------------------------------------------------------
def worked_example():
    print("=" * 72)
    print("Worked example: 3 tokens, d = 4")
    print("=" * 72)
    names = ["the", "cat", "sat"]
    # Hand-picked Q/K/V so the story is readable. In a real model these come
    # from x @ W_q etc.; here we just say what each token asks for and offers.
    K = np.array([[1, 0, 0, 0],      # "the" : key points along axis 0
                  [0, 1, 0, 0],      # "cat" : axis 1
                  [0, 0, 1, 0]], float)  # "sat" : axis 2
    Q = np.array([[1, 0, 0, 0],      # "the" looks for itself
                  [2, 2, 0, 0],      # "cat" looks for 'the' and itself equally
                  [0, 3, 1, 0]], float)  # "sat" looks mostly for 'cat'
    V = np.array([[1, 0, 0, 0],      # what each token hands over if chosen
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]], float)

    print("  scores = Q K^T / sqrt(4):")
    print(Q @ K.T / 2.0)
    print()
    out, w = scaled_dot_product_attention(Q, K, V, mask=None)
    print("  attention weights WITHOUT causal mask (each row sums to 1):")
    for n, row in zip(names, w):
        print(f"    {n:>4} attends to  " + "  ".join(f"{m}:{p:.2f}" for m, p in zip(names, row)))
    print()
    out, w = scaled_dot_product_attention(Q, K, V, mask=causal_mask(3))
    print("  attention weights WITH causal mask (upper triangle forced to 0):")
    for n, row in zip(names, w):
        print(f"    {n:>4} attends to  " + "  ".join(f"{m}:{p:.2f}" for m, p in zip(names, row)))
    print()
    print("  output = weights @ V   (each row is a blend of the value vectors):")
    print(out)
    assert np.allclose(w.sum(-1), 1.0)
    assert np.allclose(w[np.triu_indices(3, k=1)], 0.0), "causal mask leaked!"
    print("  checks: rows sum to 1, future entries are exactly 0.  ok")
    print()
    print("  Read the last row: 'sat' asked mostly for 'cat', so its output is")
    print("  mostly cat's value. That is the whole mechanism - context, by lookup.")
    print()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    worked_example()

    # ---- full block + forward pass with random weights ---------------------
    print("=" * 72)
    print("A full transformer block and GPT forward pass (random weights)")
    print("=" * 72)
    B, T, C, n_head, vocab, ctx, n_layer = 2, 5, 16, 4, 50, 8, 2
    x = rng.normal(size=(B, T, C))
    p = init_block_params(C, rng)
    y = transformer_block(x, p, n_head)
    print(f"  block input  {x.shape} -> output {y.shape}   (same shape: residual stream)")
    _, w = multi_head_attention(x, p, n_head, causal=True)
    print(f"  attention weights per head: {w.shape} = (batch, head, query, key)")
    print(f"  head 0, sequence 0 (causal, so lower-triangular):")
    print(w[0, 0])

    params = {
        "wte": rng.normal(0, 0.02, (vocab, C)),
        "wpe": rng.normal(0, 0.02, (ctx, C)),
        "blocks": [init_block_params(C, rng) for _ in range(n_layer)],
        "lnf_g": np.ones(C), "lnf_b": np.zeros(C),
    }
    tokens = rng.integers(0, vocab, size=(B, T))
    logits = gpt_forward(tokens, params, n_head)
    probs = softmax(logits)
    print(f"  tokens {tokens.shape} -> logits {logits.shape} = (batch, time, vocab)")
    print(f"  next-token distribution at the last position of sequence 0 sums to "
          f"{probs[0, -1].sum():.3f}; max prob {probs[0, -1].max():.3f} "
          f"(~uniform 1/{vocab} = {1 / vocab:.3f}: random weights know nothing yet)")
    # Causality test: changing a FUTURE token must not change earlier logits.
    tokens2 = tokens.copy()
    tokens2[:, -1] = (tokens2[:, -1] + 1) % vocab
    logits2 = gpt_forward(tokens2, params, n_head)
    assert np.allclose(logits[:, :-1], logits2[:, :-1]), "future leaked into the past!"
    print("  causality check: editing the last token changed only the last logits.  ok")
    print()

    # ---- parameter count ---------------------------------------------------
    print("=" * 72)
    print("GPT-2 small parameter count (n_layer=12, d=768, n_head=12, vocab=50257, ctx=1024)")
    print("=" * 72)
    total, breakdown, per_block = gpt2_param_count()
    print("  inside ONE block:")
    for k, v in per_block.items():
        print(f"    {k:<24} {v:>12,}")
    print(f"    {'block total':<24} {sum(per_block.values()):>12,}")
    print("  whole model:")
    for k, v in breakdown.items():
        print(f"    {k:<36} {v:>12,}   ({100 * v / total:4.1f}%)")
    print(f"    {'TOTAL':<36} {total:>12,}")
    expected = 124_439_808
    assert total == expected, f"expected {expected:,}, got {total:,}"
    print(f"  matches the published 124,439,808 exactly ({total / 1e6:.1f}M).")
    print()
    print("  Take-aways: attention is only ~1/3 of each block (the MLP is 2/3);")
    print("  the vocabulary embedding alone is 31% of GPT-2 small. Bigger models")
    print("  grow d and n_layer, so the block share dominates and the embedding")
    print("  share shrinks. Try gpt2_param_count(n_layer=48, d=1600) for GPT-2 XL.")
    for name, kw in [("GPT-2 medium", dict(n_layer=24, d=1024, n_head=16)),
                     ("GPT-2 large", dict(n_layer=36, d=1280, n_head=20)),
                     ("GPT-2 XL", dict(n_layer=48, d=1600, n_head=25))]:
        t, _, _ = gpt2_param_count(**kw)
        print(f"    {name:<13} {kw}: {t / 1e6:8.1f}M")
