"""
LAB 06 - A TINY GPT THAT WRITES SHAKESPEARE (PyTorch)
=====================================================

Run:   python labs/06_tiny_gpt.py --quick          (~1-2 min on CPU: tiny model, 300 steps)
       python labs/06_tiny_gpt.py                  (~10-15 min on CPU, ~2 min on a GPU)
       python labs/06_tiny_gpt.py --sample --temperature 0.8
       python labs/06_tiny_gpt.py --sample --prompt "ROMEO:" --max-new-tokens 500
Needs: torch  (pip install torch  - the CPU-only build is fine for this lab)
Data:  labs/data/tinyshakespeare.txt (1.1 MB of Shakespeare, already in the repo)

WHAT YOU WILL BUILD
-------------------
The same architecture as Lab 05, now with real learned weights, in PyTorch
so backprop and the GPU are free. This is GPT-2's design in ~150 lines:

    tokens -> token embedding + position embedding
           -> N x [ x + attn(ln(x)) ; x + mlp(ln(x)) ]      (Lab 05's block)
           -> layer norm -> logits over the vocabulary (tied to the embedding)

The training objective is the one and only trick of every LLM: given the
characters so far, predict the next character. The loss is cross-entropy =
-log(probability the model gave the character that actually came next).

The tokenizer here is CHARACTER-level (65 symbols) instead of BPE (Lab 04) so
that the model can learn spelling from scratch in front of your eyes.

WHAT TO OBSERVE: THE LOSS LADDER
--------------------------------
Every 100 steps you get the loss; every 500 (100 in --quick) a 200-character
sample. Here is what each loss level *means* for a 65-symbol vocabulary:

    4.17  = ln(65). Uniform guessing. Output is random symbols: "xQ;k!zj"
    ~3.0  Learned letter frequencies: lots of e, t, a, spaces. Still noise.
    ~2.5  Learned pairs: pronounceable nonsense - "the wither sould hin"
    ~2.0  Real short words, line breaks, "NAME:" speaker headers appear
    ~1.7  Mostly real words, character names, sentence-ish rhythm. Expect
          the default run (0.8M params, 5000 steps) to land around 1.6-1.8.
    ~1.5  Grammar-ish phrases; nanoGPT's 10M-param model gets here at 5000
          steps on a GPU (try --n-layer 6 --d 384 --ctx 256 if you have one).
    ~1.0  Whole coherent lines. Needs a bigger model and/or more data.
    <0.9  You'd be memorising the corpus (overfitting) at this data size.

The val loss (held-out 10% of the text) is the honest number. When train
keeps falling but val rises, the model is memorising - stop.

WHAT TO CHANGE (3 experiments)
------------------------------
A. --temperature 0.3 vs 1.0 vs 1.5 in --sample mode. Temperature divides
   the logits before softmax: low = always pick the likeliest character
   (repetitive, "safe"), high = flatten the distribution (creative, then
   gibberish). Every chatbot's "creativity" slider is this one number.
B. --ctx 8 (default 128). With only 8 characters of context the model
   cannot remember who is speaking or close a quote it opened. Watch the
   samples lose long-range structure while the loss barely moves - loss
   is a local measure, coherence is a global one.
C. --n-layer 1 --d 32 (a minuscule model) vs --n-layer 6 --d 256 (if you
   have a GPU). Same data, same steps: capacity is what turns 2.0 into 1.5.
   Also try --dropout 0.0 with the big model and watch val loss turn up.

COMMON BUGS
-----------
* Off-by-one in the batch: y must be x shifted by exactly ONE character.
  If y == x the model learns to copy and the loss goes to ~0 instantly
  (a "too good to be true" loss is always a leak).
* Forgetting model.eval() / torch.no_grad() when evaluating: dropout stays
  on and the val loss is noisy; gradients are tracked and memory explodes.
* Feeding the model more than `ctx` tokens at generation time: the position
  embedding table has no row for position ctx+1 -> index error. generate()
  crops to the last ctx tokens for this reason.
* Mask applied AFTER softmax (it must be before, as -inf).
* Mismatched device: data on CPU, model on GPU -> "expected all tensors on
  the same device". Everything must be moved with .to(device).
* Learning rate: 1e-3 is right for AdamW at this size. 1e-2 diverges
  (loss climbs back above 3 and stays), 1e-4 crawls.

NOTE: this file was written on a machine without PyTorch installed and is
untested end-to-end here. It follows nanoGPT's shapes exactly and has been
read carefully for shape bugs; if you hit an error, the traceback plus the
shape comments below should make it a two-minute fix.
"""

import sys
import os
import math
import time
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
except ImportError:
    print("This lab needs PyTorch.  Install it with:\n\n    pip install torch\n\n"
          "The CPU-only build is fine (see https://pytorch.org/get-started/locally/\n"
          "for the exact command for your OS). Everything else in this course runs\n"
          "without it.")
    sys.exit(1)

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "tinyshakespeare.txt"
CKPT_DIR = HERE / "checkpoints"


# ---------------------------------------------------------------------------
# 1. THE MODEL  (mirrors Lab 05 function-for-function)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head, ctx, dropout):
        super().__init__()
        assert d % n_head == 0
        self.n_head = n_head
        self.c_attn = nn.Linear(d, 3 * d)       # one matmul -> Q, K, V for all heads
        self.c_proj = nn.Linear(d, d)           # mix the heads back together
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # (1, 1, ctx, ctx) lower-triangular boolean: True = may attend.
        # register_buffer: saved with the model, moved with .to(device), not trained.
        self.register_buffer("mask", torch.tril(torch.ones(ctx, ctx, dtype=torch.bool))
                             .view(1, 1, ctx, ctx))

    def forward(self, x):
        B, T, C = x.shape                                   # batch, time, channels
        hd = C // self.n_head                               # head dim
        q, k, v = self.c_attn(x).split(C, dim=2)            # 3 x (B, T, C)
        # (B, T, C) -> (B, T, nh, hd) -> (B, nh, T, hd): heads become a batch dim
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)     # (B, nh, T, T)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))  # no peeking
        att = F.softmax(att, dim=-1)                        # rows sum to 1
        att = self.attn_drop(att)
        y = att @ v                                         # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # back to (B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.c_fc = nn.Linear(d, 4 * d)
        self.c_proj = nn.Linear(4 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, d, n_head, ctx, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head, ctx, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = MLP(d, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))      # residual: tokens talk to each other
        x = x + self.mlp(self.ln2(x))       # residual: each token thinks alone
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, ctx, d, n_head, n_layer, dropout):
        super().__init__()
        self.ctx = ctx
        self.wte = nn.Embedding(vocab_size, d)     # what: token -> vector
        self.wpe = nn.Embedding(ctx, d)            # where: position -> vector
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, n_head, ctx, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        # Weight tying: the output projection IS the input embedding, transposed.
        # Saves vocab*d parameters and slightly improves small models.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2 trick: shrink the residual projections by 1/sqrt(2*n_layer) so
        # that summing 2*n_layer residual branches doesn't blow up activations.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. targets: (B, T) the NEXT token at each position.
        Returns logits (B, T, vocab) and, if targets given, the mean loss."""
        B, T = idx.shape
        assert T <= self.ctx, f"sequence of length {T} exceeds context {self.ctx}"
        pos = torch.arange(T, device=idx.device)                     # (T,)
        x = self.drop(self.wte(idx) + self.wpe(pos))                 # (B, T, d)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                     # (B, T, vocab)
        loss = None
        if targets is not None:
            # cross_entropy wants (N, vocab) vs (N,): flatten batch and time.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive sampling: predict, sample one token, append, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.ctx:]                 # crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)   # only the LAST position
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")     # keep only the top k
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# 2. DATA: character-level tokenizer + random batches
# ---------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        unknown = sorted(set(c for c in s if c not in self.stoi))
        if unknown:
            raise ValueError(f"characters not in the training vocabulary: {unknown!r} "
                             f"(the model only knows {len(self.chars)} symbols)")
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def load_data(device):
    if not DATA_PATH.exists():
        print(f"Missing {DATA_PATH}. Download it with:\n"
              "  curl -o labs/data/tinyshakespeare.txt "
              "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        sys.exit(1)
    text = DATA_PATH.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return tok, data[:n].to(device), data[n:].to(device)


def get_batch(data, batch_size, ctx):
    """Random windows of length ctx; targets are the same windows shifted by 1."""
    ix = torch.randint(len(data) - ctx - 1, (batch_size,))
    x = torch.stack([data[i:i + ctx] for i in ix])            # (B, T)
    y = torch.stack([data[i + 1:i + 1 + ctx] for i in ix])    # (B, T), one step ahead
    return x, y


@torch.no_grad()
def estimate_loss(model, data, batch_size, ctx, n_batches=20):
    model.eval()                        # dropout off
    losses = torch.zeros(n_batches)
    for i in range(n_batches):
        x, y = get_batch(data, batch_size, ctx)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_text(model, tok, device, n_chars=200, temperature=1.0, prompt="\n"):
    model.eval()
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, n_chars, temperature=temperature)
    model.train()
    return tok.decode(out[0].tolist())


# ---------------------------------------------------------------------------
# 3. TRAINING
# ---------------------------------------------------------------------------
def lr_at(step, args):
    """Linear warmup for the first 100 steps, then cosine decay to 10% of lr.
    Warmup stops Adam from taking huge steps while its statistics are cold;
    decay lets the model settle into a minimum instead of bouncing around it."""
    warmup = min(100, args.steps // 10)
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, args.steps - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def train(args, device):
    torch.manual_seed(args.seed)
    tok, train_data, val_data = load_data(device)
    print(f"  data: {len(train_data) + len(val_data):,} characters, "
          f"vocab of {len(tok.chars)} symbols, ln(vocab) = {math.log(len(tok.chars)):.2f}")

    model = GPT(len(tok.chars), args.ctx, args.d, args.n_head, args.n_layer, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {args.n_layer} layers, d={args.d}, {args.n_head} heads, ctx={args.ctx}, "
          f"{n_params / 1e6:.2f}M parameters, device={device}")
    print(f"  training for {args.steps} steps, batch {args.batch_size} x {args.ctx} tokens, lr {args.lr}")
    print()

    # AdamW: Adam (per-parameter adaptive step sizes) + decoupled weight decay.
    # The default optimiser of the entire LLM era.
    # Weight decay only on matrices (nanoGPT convention): decaying LayerNorm
    # gains and biases toward 0 hurts and buys nothing.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    t0 = time.time()
    for step in range(args.steps + 1):
        # ---- evaluate + report ----
        if step % args.eval_every == 0 or step == args.steps:
            train_loss = estimate_loss(model, train_data, args.batch_size, args.ctx)
            val_loss = estimate_loss(model, val_data, args.batch_size, args.ctx)
            elapsed = time.time() - t0
            per_step = elapsed / max(step, 1)
            eta = per_step * (args.steps - step) / 60
            print(f"step {step:5d} | train {train_loss:.3f} | val {val_loss:.3f} | "
                  f"lr {lr_at(step, args):.1e} | {per_step:.2f} s/step | eta {eta:.1f} min")
        if step % args.sample_every == 0 and step > 0 or step == args.steps:
            print("-" * 60)
            print(sample_text(model, tok, device, 200, temperature=1.0))
            print("-" * 60)
        if step == args.steps:
            break

        # ---- one optimisation step ----
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        x, y = get_batch(train_data, args.batch_size, args.ctx)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # tame rare huge gradients
        opt.step()

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "config": dict(vocab_size=len(tok.chars), ctx=args.ctx, d=args.d,
                       n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout),
        "chars": tok.chars,
    }
    torch.save(ckpt, args.ckpt)
    print(f"\nsaved checkpoint to {args.ckpt}")
    print(f"generate more with:  python labs/06_tiny_gpt.py --sample --ckpt {args.ckpt} --temperature 0.8")


def sample_from_checkpoint(args, device):
    if not Path(args.ckpt).exists():
        print(f"No checkpoint at {args.ckpt}. Train first: python labs/06_tiny_gpt.py --quick")
        sys.exit(1)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    model = GPT(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    tok = CharTokenizer("")
    tok.chars = ckpt["chars"]
    tok.stoi = {ch: i for i, ch in enumerate(tok.chars)}
    tok.itos = {i: ch for i, ch in enumerate(tok.chars)}
    torch.manual_seed(args.seed)
    print(f"  loaded {args.ckpt}: {cfg}")
    print(f"  temperature {args.temperature}, prompt {args.prompt!r}")
    print("-" * 60)
    print(sample_text(model, tok, device, args.max_new_tokens, args.temperature, args.prompt))
    print("-" * 60)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train or sample a tiny character-level GPT.")
    ap.add_argument("--quick", action="store_true", help="tiny model, 300 steps (~1-2 min CPU)")
    ap.add_argument("--sample", action="store_true", help="generate from a saved checkpoint")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n-layer", type=int, default=None)
    ap.add_argument("--d", type=int, default=None, help="embedding width")
    ap.add_argument("--n-head", type=int, default=None)
    ap.add_argument("--ctx", type=int, default=None, help="context length in characters")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--sample-every", type=int, default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    # Defaults depend on --quick; explicit flags always win.
    preset = (dict(steps=300, n_layer=2, d=64, n_head=2, ctx=64, batch_size=32, sample_every=100,
                   ckpt=str(CKPT_DIR / "tiny_gpt_quick.pt"))
              if args.quick else
              dict(steps=5000, n_layer=4, d=128, n_head=4, ctx=128, batch_size=32, sample_every=500,
                   ckpt=str(CKPT_DIR / "tiny_gpt.pt")))
    for k, v in preset.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.quick and args.lr == 1e-3:
        args.lr = 3e-3       # the tiny model can take bigger steps

    device = pick_device(args.device)
    if args.sample:
        sample_from_checkpoint(args, device)
    else:
        print("=" * 60)
        print("Tiny GPT on Tiny Shakespeare" + ("  [--quick]" if args.quick else ""))
        print("=" * 60)
        train(args, device)
