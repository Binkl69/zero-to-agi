"""
LAB 09 - Fine-tuning with LoRA: teach the tiny GPT a new style (PyTorch)
========================================================================

Run:   python labs/09_lora_finetune.py --quick     (about half a minute on a laptop CPU)
       python labs/09_lora_finetune.py             (about a minute; a better base model)
Needs: torch  (pip install torch  - the CPU-only build is fine)
Data:  labs/data/tinyshakespeare.txt   (what the model is pretrained on)
       labs/data/weather_reports.txt   (7 KB of plain modern weather
                                        forecasts, written for this lab)

WHAT YOU'LL BUILD
-----------------
Lab 06 trained a tiny GPT from nothing. Almost nobody does that for real.
What people actually do is take a model somebody else pretrained and
FINE-TUNE it: keep training, on their own data, until it writes the way they
need. This lab does that twice, to the same model, on the same data:

  FULL FINE-TUNING   every one of the model's 108,352 numbers may change
  LoRA               the model is frozen; 3,072 new numbers (2.8%) are
                     trained beside it

LoRA ("low-rank adaptation") is how most fine-tuning of open models is done
today, because the result is a file a few megabytes in size instead of a
second copy of a multi-gigabyte model. You implement it by hand, in the
short class `LoRALinear`: a frozen weight table W plus a trainable
correction (alpha / r) * B A, with B starting at zero.

The base model is lab 06's own code, imported from 06_tiny_gpt.py: the same
GPT class, tokenizer and batching. This lab pretrains it briefly on
Shakespeare itself, so it does not depend on you having run lab 06. If you
have, --base labs/checkpoints/tiny_gpt.pt starts from your own checkpoint.

  Step 1  Pretrain the base model; measure it on Shakespeare and on weather.
  Step 2  Add the adapters. Count what is trainable.
  Step 3  Fine-tune both ways for the same number of steps on the same
          batches. Watch the loss on held-out weather fall.
  Step 4  Measure Shakespeare again. Both have got worse at it: that is
          catastrophic forgetting. Then unplug the adapter.
  Step 5  Merge the adapter into W and check the outputs are identical.
  Step 6  Samples from all three models.

WHAT TO OBSERVE (numbers are from --quick; the last digit can differ on
your machine)
---------------
1. The base model scores 2.418 on held-out Shakespeare and 2.436 on
   held-out weather. (Lower is better; 4.17 is guessing. See lab 06's
   loss ladder.) Those are nearly the same: after 300 steps it knows
   letters and common words, not yet a style.

2. LoRA trains 3,072 numbers, full fine-tuning 108,352: 35 times fewer.
   Before any training the adapted model's outputs differ from the base
   model's by exactly 0.000000, because B is zero.

3. After 150 steps the held-out weather loss is 2.272 with LoRA and 2.055
   with full fine-tuning. Both learned; full fine-tuning learned more.
   LoRA's small size is a real constraint, not a free lunch.

4. Shakespeare got worse for both: +0.175 with LoRA, +0.400 with full
   fine-tuning. Nothing told the model to keep its old skill, so it
   did not.

5. Unplug the adapter and the Shakespeare loss is 2.418 again, exactly.
   Merge it instead and the outputs match the adapted model's to within
   0.000003 (floating-point rounding), in a model with no extra layers.

6. Without --quick (1,000 pretraining steps, 400 fine-tuning steps) the
   same story is larger and easier to see: weather 2.165 -> 1.849 (LoRA)
   and 1.570 (full); forgetting +0.279 against +0.942. The full model's
   loss on the bulletins it trained on is 1.175, far below the 1.570 it
   gets on unseen ones: it has started to memorise 5.8 KB of text. Its
   samples begin "The low the eard in the lours. The aron the eand and a
   storm", which is recognisably a forecast; the base model's begin
   "CKENTES II: Theat he land undee we and Thou son thee".

WHAT TO CHANGE (5 experiments - predict the outcome BEFORE you run)
-------------------------------------------------------------------
Add each flag to   python labs/09_lora_finetune.py --quick

1. --rank 1, then --rank 16. Trainable numbers: 768 (0.7%) and 12,288
   (11.3%). Held-out weather: 2.345 and 2.224 (rank 4 gave 2.272).
   Forgetting: +0.106 and +0.240 (rank 4: +0.175). More rank learns
   more and forgets more.

2. --rank 16 --targets all (adapters on the MLP layers too: 32,768
   numbers, 30% of the model). Weather 2.123, nearly level with full
   fine-tuning's 2.055 - and its forgetting is now level with full
   fine-tuning's too: +0.442 against +0.400. (With --seed 1 it is +0.400
   against +0.386; with --seed 2, +0.354 against +0.435. Sometimes a
   little more, sometimes a little less, never the 2.3 times less that
   rank 4 showed.) LoRA is not a shield against forgetting.

3. --full-lr 0.0001 (full fine-tuning, ten times more gently). It now
   learns about as much weather as LoRA (2.251 against 2.272) and
   forgets LESS than LoRA (+0.122 against +0.175; the same direction
   with --seed 1 and --seed 2). Put 2 and 3 together:
   forgetting follows how far you move the model, not which method moved
   it. What LoRA really gives you is in observation 5: the change is
   small, separate, and removable.

4. --b-init 0.1 (start B with random numbers, not zeros). Before a single
   step the outputs already differ from the base model's by 3.76 and the
   weather loss starts at 2.682, not 2.436: you are no longer fine-tuning
   the model you were given. It recovers by the end (2.274), but only
   because this task is easy; zero is the starting point that costs
   nothing.

5. --steps 600 (four times as long). Full fine-tuning reaches 1.776 on
   held-out weather but 1.510 on the bulletins it trained on, a gap of
   +0.266, and its Shakespeare loss climbs by +0.852. LoRA's gap stays
   at about zero (-0.008). With 5.8 KB of data, more steps mostly buy
   memorisation.

COMMON BUGS
-----------
* Forgetting to freeze the base weights. Then you are doing full
  fine-tuning with extra steps, and the "adapter" file is useless alone.
* Initialising both A and B to zero. The gradient of each is multiplied
  by the other, so nothing ever moves. One of them must start non-zero.
* Leaving out alpha / r when merging. The merged model then disagrees
  with the adapted one (Step 5 would print MISMATCH).
* Measuring only the new task. Forgetting is invisible unless you look.
* Comparing the two methods on different batches. Both runs here reset
  the random seed first, so they see exactly the same data.
"""

import sys
import os
import copy
import math
import time
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

if hasattr(sys.stdout, "reconfigure"):   # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("This lab needs PyTorch.  Install it with:\n\n    pip install torch\n\n"
          "The CPU-only build is fine (see https://pytorch.org/get-started/locally/\n"
          "for the exact command for your OS). Labs 10, 11 and 12 run without it.")
    sys.exit(1)

HERE = Path(__file__).resolve().parent
WEATHER_PATH = HERE / "data" / "weather_reports.txt"

# ---------------------------------------------------------------------------
# 0. REUSE LAB 06
#
# The model, the tokenizer and the batching are lab 06's, imported, not
# copied: the point of this lab is what you ADD to a model you already have.
# (A file whose name starts with a digit cannot be imported with a plain
# `import`, hence the three lines of importlib.)
# ---------------------------------------------------------------------------
sys.dont_write_bytecode = True          # do not leave a __pycache__ folder behind
_spec = importlib.util.spec_from_file_location("lab06", HERE / "06_tiny_gpt.py")
lab06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab06)


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. LoRA, BY HAND
#
# A Linear layer computes  y = x W^T + b,  where W is a big (out x in) table.
# Full fine-tuning changes every number in W. LoRA freezes W and learns a
# CHANGE to it, forced to be simple:
#
#       W_effective = W + (alpha / r) * B A
#
#   A is (r x in), B is (out x r), and r (the "rank") is small - 4 here.
#   So instead of out*in new numbers we learn r*(in + out).
#
#   B starts at ZERO, so B A = 0 and the model starts out exactly as it was.
#   A starts random, so that gradients can flow into B on the first step.
#   alpha / r is a fixed volume knob on the adapter.
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, b_init=0.0):
        super().__init__()
        self.base = base                                   # the original nn.Linear
        for p in self.base.parameters():
            p.requires_grad = False                        # FROZEN: never updated
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r) if b_init == 0
                              else torch.randn(base.out_features, r) * b_init)
        self.scale = alpha / r

    def forward(self, x):
        #       frozen path        low-rank path: x -> r numbers -> out numbers
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    def merged(self):
        """Fold the adapter into a plain Linear: W <- W + (alpha/r) B A."""
        out = nn.Linear(self.base.in_features, self.base.out_features)
        with torch.no_grad():
            out.weight.copy_(self.base.weight + self.scale * (self.B @ self.A))
            out.bias.copy_(self.base.bias)
        return out


def lora_sites(model, targets):
    """The Linear layers that get an adapter: (parent module, attribute name)."""
    sites = []
    for block in model.blocks:
        sites += [(block.attn, "c_attn"), (block.attn, "c_proj")]
        if targets == "all":
            sites += [(block.mlp, "c_fc"), (block.mlp, "c_proj")]
    return sites


def add_lora(model, r, alpha, targets, b_init=0.0):
    """Freeze the whole model, then wrap the chosen Linear layers."""
    for p in model.parameters():
        p.requires_grad = False
    for parent, name in lora_sites(model, targets):
        setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha, b_init))
    return model


def merge_lora(model, targets):
    """Replace every LoRALinear by one ordinary Linear with the adapter folded in."""
    for parent, name in lora_sites(model, targets):
        setattr(parent, name, getattr(parent, name).merged())
    return model


def remove_lora(model, targets):
    """Unplug the adapters: put the original frozen layers back."""
    for parent, name in lora_sites(model, targets):
        setattr(parent, name, getattr(parent, name).base)
    return model


def count(params):
    return sum(p.numel() for p in params)


# ---------------------------------------------------------------------------
# 2. DATA AND MEASUREMENT
# ---------------------------------------------------------------------------
@torch.no_grad()
def exact_loss(model, data, ctx, stride):
    """Average next-character loss over fixed windows of `data`. No randomness:
    the same model and data always give the same number, so two models can be
    compared to three decimal places."""
    model.eval()
    starts = range(0, len(data) - ctx - 1, stride)
    x = torch.stack([data[i:i + ctx] for i in starts])
    y = torch.stack([data[i + 1:i + 1 + ctx] for i in starts])
    total = 0.0
    for i in range(0, len(x), 256):
        _, loss = model(x[i:i + 256], y[i:i + 256])
        total += loss.item() * len(x[i:i + 256])
    model.train()
    return total / len(x)


def train(model, params, data, steps, lr, batch_size, ctx, seed, report=None, every=0):
    """Lab 06's training loop, cut down. `params` is the list of tensors the
    optimiser is allowed to change - that one argument is the whole
    difference between full fine-tuning and LoRA."""
    torch.manual_seed(seed)                     # same seed -> same batches for both methods
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    schedule = SimpleNamespace(steps=steps, lr=lr)
    history = []
    for step in range(steps + 1):
        if report is not None and every and (step % every == 0 or step == steps):
            history.append((step, report(model)))
        if step == steps:
            break
        for g in opt.param_groups:
            g["lr"] = lab06.lr_at(step, schedule)
        x, y = lab06.get_batch(data, batch_size, ctx)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    return history


def sample(model, tok, device, seed, n=150, prompt="\n"):
    torch.manual_seed(seed)
    text = lab06.sample_text(model, tok, device, n, temperature=0.8, prompt=prompt)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tune lab 06's tiny GPT with LoRA and with full fine-tuning.")
    ap.add_argument("--quick", action="store_true", help="fewer steps")
    ap.add_argument("--rank", type=int, default=4, help="r, the rank of the adapter (default 4)")
    ap.add_argument("--alpha", type=float, default=None, help="adapter scale is alpha/r (default 2*r)")
    ap.add_argument("--targets", choices=["attn", "all"], default="attn",
                    help="which layers get an adapter: attention only, or attention + MLP")
    ap.add_argument("--steps", type=int, default=None, help="fine-tuning steps, each method")
    ap.add_argument("--pretrain-steps", type=int, default=None)
    ap.add_argument("--lora-lr", type=float, default=1e-2)
    ap.add_argument("--full-lr", type=float, default=1e-3)
    ap.add_argument("--b-init", type=float, default=0.0, help="std of B's starting values (LoRA uses 0)")
    ap.add_argument("--base", default=None, help="start from a lab 06 checkpoint instead of pretraining")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    if args.steps is None:
        args.steps = 150 if args.quick else 400
    if args.pretrain_steps is None:
        args.pretrain_steps = 300 if args.quick else 1000
    if args.alpha is None:
        args.alpha = 2.0 * args.rank
    BATCH, CTX = 32, 64

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    t_start = time.time()

    # ---- data ----
    tok, shake_train, shake_val = lab06.load_data(device)
    weather = WEATHER_PATH.read_text(encoding="utf-8")
    bulletins = weather.strip().split("\n\n")
    held_out = [b for i, b in enumerate(bulletins) if i % 6 == 5]          # every sixth bulletin
    kept = [b for i, b in enumerate(bulletins) if i % 6 != 5]
    weather_train = torch.tensor(tok.encode("\n\n".join(kept) + "\n"), dtype=torch.long, device=device)
    weather_val = torch.tensor(tok.encode("\n\n".join(held_out) + "\n"), dtype=torch.long, device=device)
    shake_probe = shake_val[:20000]             # a fixed slice of held-out Shakespeare

    def on_weather(m):
        return exact_loss(m, weather_val, CTX, stride=8)

    def on_shakespeare(m):
        return exact_loss(m, shake_probe, CTX, stride=64)

    # =======================================================================
    rule("STEP 1: The base model - lab 06's tiny GPT, pretrained on Shakespeare")
    if args.base:
        ckpt = torch.load(args.base, map_location=device, weights_only=True)
        if ckpt["chars"] != tok.chars:
            print("That checkpoint was trained on a different alphabet.")
            sys.exit(1)
        base = lab06.GPT(**ckpt["config"]).to(device)
        base.load_state_dict(ckpt["model_state"])
        CTX = ckpt["config"]["ctx"]
        print(f"\n  loaded {args.base}")
    else:
        torch.manual_seed(args.seed)
        base = lab06.GPT(len(tok.chars), CTX, 64, 2, 2, 0.1).to(device)
        print(f"\n  pretraining for {args.pretrain_steps} steps on {len(shake_train):,} characters of Shakespeare ...")
        train(base, list(base.parameters()), shake_train, args.pretrain_steps, 3e-3, BATCH, CTX, args.seed)
    n_total = count(base.parameters())
    base_shake, base_weather = on_shakespeare(base), on_weather(base)
    print(f"  model: {len(base.blocks)} layers, {n_total:,} parameters  ({time.time() - t_start:.0f} s)")
    print(f"\n  new text: {len(weather):,} characters of weather forecasts in {len(bulletins)} bulletins;")
    print(f"  {len(kept)} to train on, {len(held_out)} held out to measure on. A taste:")
    print(f"    \"{' '.join(kept[0].split())[15:83]}...\"")
    print(f"\n  loss of the base model   on held-out Shakespeare: {base_shake:.3f}")
    print(f"                           on held-out weather:     {base_weather:.3f}")
    if base_weather - base_shake > 0.05:
        print(f"  It has never seen a weather forecast, and it shows.")
    else:
        print(f"  Nearly the same: a base model this weak has learned letters and common words,")
        print(f"  not yet a style, so one kind of English is about as hard for it as another.")
        print(f"  (Without --quick the base model is better and the two numbers pull apart.)")

    # =======================================================================
    rule("STEP 2: Add LoRA adapters and count what is trainable")
    lora = add_lora(copy.deepcopy(base), args.rank, args.alpha, args.targets, args.b_init)
    lora_params = [p for p in lora.parameters() if p.requires_grad]
    n_lora = count(lora_params)
    full = copy.deepcopy(base)
    full_params = list(full.parameters())
    sites = lora_sites(base, args.targets)
    print(f"\n  rank r = {args.rank}, alpha = {args.alpha:g} (scale alpha/r = {args.alpha / args.rank:g}), "
          f"adapters on {len(sites)} layers ({'attention only' if args.targets == 'attn' else 'attention and MLP'})\n")
    one = lora.blocks[0].attn.c_attn
    print(f"  one adapted layer, blocks[0].attn.c_attn:")
    print(f"    W  frozen     {tuple(one.base.weight.shape)!s:<10} = {one.base.weight.numel():>6,} numbers")
    print(f"    A  trainable  {tuple(one.A.shape)!s:<10} = {one.A.numel():>6,} numbers")
    print(f"    B  trainable  {tuple(one.B.shape)!s:<10} = {one.B.numel():>6,} numbers, all zero to start\n")
    print(f"  {'':<22} {'trainable numbers':>18} {'share of the model':>20}")
    print(f"  {'full fine-tuning':<22} {n_total:>18,} {'100.0%':>20}")
    print(f"  {'LoRA':<22} {n_lora:>18,} {100 * n_lora / n_total:>19.1f}%")
    print(f"\n  LoRA trains {n_total / n_lora:.0f} times fewer numbers.")

    probe_x = shake_probe[:CTX].unsqueeze(0)
    base.eval(), lora.eval()
    with torch.no_grad():
        gap0 = (lora(probe_x)[0] - base(probe_x)[0]).abs().max().item()
    base.train(), lora.train()
    print(f"  Before any training, largest difference between the adapted model's outputs")
    print(f"  and the base model's: {gap0:.6f}" + ("  (B is zero, so the adapter is silent)" if gap0 == 0 else
                                                  "  (B was NOT zero: the model is already disturbed)"))

    # =======================================================================
    rule(f"STEP 3: Fine-tune on weather, {args.steps} steps each way")
    every = max(1, args.steps // 5)
    t0 = time.time()
    hist_lora = train(lora, lora_params, weather_train, args.steps, args.lora_lr, BATCH, CTX, args.seed + 1,
                      report=on_weather, every=every)
    t_lora = time.time() - t0
    t0 = time.time()
    hist_full = train(full, full_params, weather_train, args.steps, args.full_lr, BATCH, CTX, args.seed + 1,
                      report=on_weather, every=every)
    t_full = time.time() - t0
    print(f"\n  Loss on the HELD-OUT weather bulletins (lower is better). Same batches, same")
    print(f"  number of steps; learning rate {args.lora_lr:g} for LoRA, {args.full_lr:g} for full fine-tuning.\n")
    print(f"  {'step':>6} {'LoRA':>10} {'full fine-tune':>16}")
    print("  " + "-" * 34)
    for (step, a), (_, b) in zip(hist_lora, hist_full):
        print(f"  {step:>6} {a:>10.3f} {b:>16.3f}")
    lora_weather, full_weather = hist_lora[-1][1], hist_full[-1][1]
    lora_train_loss = exact_loss(lora, weather_train, CTX, stride=16)
    full_train_loss = exact_loss(full, weather_train, CTX, stride=16)
    print(f"\n  Held-out weather: {base_weather:.3f} -> {lora_weather:.3f} (LoRA), {full_weather:.3f} (full fine-tuning).")
    better = "Full fine-tuning" if full_weather < lora_weather else "LoRA"
    print(f"  {better} ends lower, by {abs(full_weather - lora_weather):.3f}.")
    gap_l, gap_f = lora_weather - lora_train_loss, full_weather - full_train_loss
    print(f"\n  And on the bulletins they TRAINED on:   LoRA {lora_train_loss:.3f}   full {full_train_loss:.3f}")
    print(f"  Gap between seen and unseen bulletins:  LoRA {gap_l:+.3f}   full {gap_f:+.3f}")
    if gap_f > 0.15 and gap_f > 2 * max(gap_l, 0.0):
        print(f"  The full model is much better on text it has seen than on text it has not: with")
        print(f"  {n_total:,} free numbers and only {len(weather_train):,} characters to learn from, it has begun to")
        print(f"  MEMORISE the bulletins. LoRA's {n_lora:,} numbers leave it far less room to do that.")
    print(f"  (Time: LoRA {t_lora:.0f} s, full {t_full:.0f} s. LoRA still runs the whole model forwards and")
    print(f"  backwards, so on a model this small it saves little or no time. Its saving")
    print(f"  is in what you have to store and ship.)")

    # =======================================================================
    rule("STEP 4: Catastrophic forgetting - what happened to the Shakespeare?")
    lora_shake, full_shake = on_shakespeare(lora), on_shakespeare(full)
    print(f"\n  Loss on held-out Shakespeare, the thing the model USED to be good at:\n")
    print(f"  {'':<26} {'Shakespeare':>12} {'change':>9} {'weather':>10}")
    print("  " + "-" * 60)
    print(f"  {'base model':<26} {base_shake:>12.3f} {'':>9} {base_weather:>10.3f}")
    print(f"  {'after LoRA':<26} {lora_shake:>12.3f} {lora_shake - base_shake:>+9.3f} {lora_weather:>10.3f}")
    print(f"  {'after full fine-tuning':<26} {full_shake:>12.3f} {full_shake - base_shake:>+9.3f} {full_weather:>10.3f}")
    forgot_l, forgot_f = lora_shake - base_shake, full_shake - base_shake
    print(f"\n  FORGETTING: LoRA {forgot_l:+.3f}, full fine-tuning {forgot_f:+.3f}.")
    if forgot_l > 0 and forgot_f > 0:
        if forgot_f > forgot_l:
            print(f"  Both forgot. Full fine-tuning forgot {forgot_f / forgot_l:.1f} times as much.")
        else:
            print(f"  Both forgot. In this run LoRA forgot MORE than full fine-tuning did.")
        print(f"  Nobody told either model to stop writing Shakespeare. Training only ever says")
        print(f"  \"be better at THIS\"; whatever is not in the new data is free to be overwritten.")
        print(f"\n  Do not read that table as \"LoRA protects old knowledge\". Look at the weather")
        print(f"  column: the method that learned more is the one that forgot more. Whether")
        print(f"  LoRA is gentler, or simply moved the model less, is a question you can test:")
        print(f"  experiments 2 and 3 at the top of this file do, and the answer is not flattering.")
    else:
        print(f"  At least one of them did NOT get worse at Shakespeare. That happens when the")
        print(f"  base model had barely learned Shakespeare to begin with: there was little to")
        print(f"  forget, and learning any English at all helps a little. Pretrain for longer")
        print(f"  (--pretrain-steps 300 or more) and the forgetting appears.")

    # ---- the property that is actually special about LoRA ----
    unplugged = remove_lora(copy.deepcopy(lora), args.targets)
    unplugged_shake = on_shakespeare(unplugged)
    print(f"\n  Now unplug the adapter (throw away A and B, keep the frozen W):")
    print(f"    Shakespeare loss {unplugged_shake:.3f}  -  the base model's was {base_shake:.3f}")
    if abs(unplugged_shake - base_shake) < 1e-9:
        print(f"  Exactly the base model again. LoRA's forgetting lives entirely in the adapter:")
        print(f"  {n_lora:,} numbers you can remove, swap for another adapter, or keep one of per")
        print(f"  task. The fully fine-tuned model can only be undone by keeping a second copy")
        print(f"  of all {n_total:,}.")

    # =======================================================================
    rule("STEP 5: Merge the adapter into W - same outputs, no extra layers")
    merged = merge_lora(copy.deepcopy(lora), args.targets)
    x = torch.cat([shake_probe[:CTX].unsqueeze(0), weather_val[:CTX].unsqueeze(0)])
    lora.eval(), merged.eval()
    with torch.no_grad():
        out_lora, out_merged = lora(x)[0], merged(x)[0]
    lora.train()
    gap = (out_lora - out_merged).abs().max().item()
    same_choice = (out_lora.argmax(-1) == out_merged.argmax(-1)).float().mean().item()
    print(f"\n  W_merged = W + (alpha/r) * B A, computed once, for each of the {len(sites)} adapted layers.")
    print(f"    parameters, adapted model:  {count(lora.parameters()):,}   (base {n_total:,} + adapter {n_lora:,})")
    print(f"    parameters, merged model:   {count(merged.parameters()):,}   (the same shape as the base model)")
    print(f"    largest difference between their outputs: {gap:.1e}")
    print(f"    (the outputs themselves are around {out_lora.abs().mean().item():.1f} in size)")
    print(f"    same most-likely next character at {100 * same_choice:.0f}% of {out_lora.shape[0] * out_lora.shape[1]} positions")
    print(f"    weather loss: adapted {on_weather(lora):.4f}, merged {on_weather(merged):.4f}")
    if gap < 1e-4:
        print(f"  MERGE CHECK: OK. Identical up to floating-point rounding. A merged LoRA model costs")
        print(f"  nothing extra to run: the adapter was only ever a compact way to STORE a change.")
    else:
        print(f"  MERGE CHECK: MISMATCH - the merged model does not match the adapted one.")

    # =======================================================================
    rule("STEP 6: What they write now")
    print(f"\n  The same starting point and the same random seed for all three.\n")
    for name, m in [("base", base), ("LoRA", lora), ("full", full)]:
        text = sample(m, tok, device, args.seed)
        print(f"  {name:<5} {text[:70]}")
        print(f"  {'':<5} {text[70:140]}")
    print(f"\n  A model this small, trained for seconds, does not write English. Read for")
    print(f"  texture: does a line look more like a play (NAMES in capitals, thee and thou)")
    print(f"  or like a forecast (the, low, rain, degrees)? Where you cannot tell by eye,")
    print(f"  that is normal at this size, and it is why Steps 3 and 4 measure instead.")

    rule("WHAT THIS LAB SHOWS")
    print(f"""
1. Fine-tuning is just more training, on different data. The model has no
   idea it is being "adapted".

2. LoRA trains a small low-rank CHANGE to frozen weights: here {n_lora:,} numbers
   in place of {n_total:,}. With B = 0 at the start, it begins as exactly the
   base model.

3. Fewer trainable numbers is a real constraint, not a free lunch. Compare the
   two columns in Step 3 and decide for yourself what it cost here.

4. Fine-tuning forgets. Measure the OLD task as well as the new one, every
   time. LoRA's forgetting is confined to an adapter that can be unplugged;
   full fine-tuning's is written into every weight.

5. A merged adapter is an ordinary model. That is why a 50 MB LoRA file can
   stand in for a 5 GB fine-tuned checkpoint.

Total time: {time.time() - t_start:.0f} seconds.
Now try the experiments at the top of this file. Each is one flag, and each
says what you should see. Start with:   python labs/09_lora_finetune.py --quick --rank 1
""")
