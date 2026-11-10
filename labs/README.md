# The labs

Reading about a neural network and building one are different experiences, and only
the second one sticks. These eight labs build the real machinery, small enough to
read in one sitting and slow enough to watch.

Every lab is a single standalone file with no framework, heavy teaching comments, and
a docstring at the top that tells you **what you'll build**, **what to observe**,
**what to change** (four or five experiments, each with the outcome you should get, so
you can predict first and check afterwards), and **common bugs**. Run it, read it, break it.

## Setup

```bash
python -m venv .venv
```

Activate it — on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Then install the dependencies and run the first lab from the repository root:

```bash
pip install -r requirements.txt
```

```bash
python labs/01_perceptron.py
```

Python 3.10 or newer. Labs 01–04 and 07 are pure Python and need nothing installed at
all; 05 and 08 need numpy; only lab 06 needs PyTorch. If the PyTorch download is
enormous (on Linux it can be several gigabytes of GPU libraries), see Troubleshooting
for the small CPU-only install.

The commands here are written from the repository root, but every lab finds its own
files, so `cd labs` followed by `python 01_perceptron.py` works just as well.

## The ladder

Do them in order. Each one assumes the previous one.

### 01 · `01_perceptron.py` — the 1958 machine · pure Python · instant

Frank Rosenblatt's perceptron: inputs times weights, add a bias, threshold. You
implement the original learning rule, watch it converge on AND and OR while printing
the decision boundary as ASCII art, and then watch it fail forever on XOR — the failure
that helped freeze the field for a decade. The lab ends by hand-wiring a two-layer
network that solves XOR, which is the entire argument for depth in about fifteen lines.

**Teaches:** weights, bias, a learning rule, linear separability, why depth matters.

### 02 · `02_autograd.py` — build what PyTorch does · pure Python · instant

A scalar autograd engine in the spirit of Karpathy's micrograd. You write a `Value`
class that records every operation into a graph, then a `backward()` that topologically
sorts the graph and applies the chain rule. It includes a finite-difference gradient
check that asserts your analytic gradients match numerical ones, which is the single
most useful debugging habit in machine learning.

**Teaches:** computational graphs, the chain rule as code, why `loss.backward()` is not
magic. This is the most important lab in the set.

### 03 · `03_mlp.py` — a real network, trained by your own engine · pure Python · ~1 min

Neuron, Layer and MLP classes built on the `Value` class you just wrote. It trains on
XOR and on a two-moons dataset you generate yourself, prints the loss as it falls, and
renders the learned decision boundary as ASCII. It is slow precisely because it is
scalar: every single number gets its own graph node. Feeling that slowness is the
motivation for tensors and GPUs. `python labs/03_mlp.py --quick` stops the moons after
40 steps and finishes in about ten seconds, if you only want to see it run.

**Teaches:** forward and backward through layers, SGD, why the industry uses batched
tensor maths.

### 04 · `04_tokenizer_bpe.py` — how text becomes numbers · pure Python · under 1 s

Byte-pair encoding from scratch, over raw bytes, the same family of algorithm behind
GPT tokenizers. You train merges on a built-in paragraph, watch the first twenty merges
appear, encode and decode with a round-trip assertion, and measure the compression
ratio (2.64 bytes per token on this small text; real tokenizers reach about 4). It also
explains, concretely, why a model struggles to count the r's in "strawberry": it never
sees the letters.

**Teaches:** vocabularies, subwords, why token counts drive your API bill.

### 05 · `05_attention.py` — the transformer's engine · numpy · under 1 s

Scaled dot-product attention, causal masking, and multi-head attention with every
reshape commented, plus a worked three-token example with the attention matrix printed
and a short demonstration of what the division by √d is for. Then a complete
transformer block forward pass in numpy — LayerNorm, attention, a GELU MLP, residuals —
and a parameter counter that reproduces GPT-2 small's 124,439,808 parameters exactly,
with the breakdown.

**Teaches:** what `attention(Q, K, V)` actually computes, shapes, where the parameters
in a real model live.

### 06 · `06_tiny_gpt.py` — train a language model · PyTorch · ~20 s quick, ~30 min full on CPU

A minimal GPT: token and position embeddings, N blocks of causal self-attention and
MLP, a tied output head. It trains a character-level model on
`labs/data/tinyshakespeare.txt` and prints a sample every 500 steps, so you literally
watch the output go from noise to letter-frequency soup to words to almost-grammatical
Elizabethan dialogue.

```bash
python labs/06_tiny_gpt.py --quick
```

```bash
python labs/06_tiny_gpt.py
```

```bash
python labs/06_tiny_gpt.py --sample --temperature 0.8
```

Start with `--quick`: a model of 0.11M parameters, 300 steps, loss from 4.20 down to
about 2.40, and samples that have reached word-shaped nonsense. `--sample` then
generates from whichever checkpoint you have trained (the full one if it exists,
otherwise the quick one). All five experiments in the lab use `--quick`, so none of
them needs the long run.

The comments tell you what each loss level means: about 4.2 is uniform over the 65
characters, 3.0 means it has learned letter frequencies, 2.0 means words, 1.5 means
grammar is appearing. The full run (0.8M parameters, 5,000 steps) ends at a validation
loss of about 1.6.

**Teaches:** the actual architecture of a GPT, a real training loop, sampling and
temperature, checkpoints.

### 07 · `07_q_learning.py` — learning from consequences · pure Python · instant

Tabular Q-learning in a gridworld with walls, a goal and a pit. It prints the policy as
an arrow map and the value function as a grid, plus an ASCII reward curve over
episodes. One experiment (`--coin`) adds a respawning coin and shows the agent learning
to farm it forever instead of finishing — reward hacking, in about twenty lines. The
next one shrinks the coin until farming it pays less than finishing, and the agent farms
it anyway, because it never explores far enough to find out.

**Teaches:** states, actions, rewards, discounting, exploration versus exploitation,
and why reward design is hard.

### 08 · `08_toy_rlhf.py` — how an assistant gets its manners · numpy · ~1 s

The full post-training pipeline at toy scale. Simulated humans compare pairs of
responses; you fit a reward model with the Bradley-Terry loss; you optimise a policy
against that reward model with a KL penalty; then you do the same thing with DPO and no
reward model at all.

The reward model deliberately cannot see whether an answer is *correct*, only surface
features like politeness and length — exactly the situation a real reward model is in.
It learns politeness and length as proxies for correctness, and with the KL penalty
switched off the policy collapses onto a confident, beautifully written, completely
wrong answer. The probability of a correct response falls from 0.40 to 0.002. DPO, given
the same view of the responses, falls into the same trap with nearly the same numbers.
Then you switch one line, give the reward model access to correctness, and watch the
failure disappear. The result holds for every random seed tried (40 of them), and one
of the experiments asks you to check that yourself.

**Teaches:** preference learning, reward models, the KL penalty, over-optimisation and
Goodhart's law, DPO, and why 2024–2025 reasoning models moved to verifiable rewards.

## Where to go after these

You now have the concepts. The next step is scale, and it is more accessible than it
looks.

- **[nanoGPT](https://github.com/karpathy/nanoGPT)** — the grown-up version of lab 06.
  Reproducing GPT-2 (124M) on OpenWebText costs a few hundred dollars of rented GPU
  time, or you can train a small model on your own data for the price of a coffee.
- **[llm.c](https://github.com/karpathy/llm.c)** — the same model in raw C and CUDA.
  Read it when you want to know where the FLOPs actually go.
- **[Hugging Face TRL](https://github.com/huggingface/trl)** — production implementations
  of SFT, reward modelling, PPO, DPO and GRPO. Lab 08 is the toy; this is the tool.
- **[Unsloth](https://github.com/unslothai/unsloth)** — fine-tune an open model with
  LoRA on a single consumer GPU, often in under an hour.
- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** —
  build and run evals. Writing a good eval is one of the highest-leverage things a
  newcomer can contribute.

Chapter 15 of the course lays out a twelve-month plan built around these.

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`** — only lab 06 needs it. On Windows
and macOS `pip install torch` gets a suitable build. On Linux that command downloads
several gigabytes of GPU libraries; unless you have an NVIDIA card, use the CPU build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If pip picks a wheel that fails, use the exact command from
[pytorch.org](https://pytorch.org/get-started/locally/).

**Garbled characters or a `UnicodeEncodeError` on Windows** — every lab calls
`sys.stdout.reconfigure(encoding="utf-8")` at the top, which fixes this on Python 3.7+.
If your console still mangles output, run `chcp 65001` before the script, or use
Windows Terminal.

**Lab 03 feels frozen** — it is not, it is just scalar autograd doing hundreds of
thousands of tiny operations. It prints a line every 25 steps, a few seconds apart, and
finishes in about a minute. That slowness is the lesson. `--quick` cuts it to ten
seconds.

**Lab 06 is very slow** — use `--quick` first; it should take well under a minute. The
full run is the only long thing in the course. It auto-detects CUDA and Apple Silicon
MPS; pass `--device cpu` to force CPU. On CPU it uses half your logical cores, which
is usually fastest and leaves the machine usable.

**Different numbers than the comments describe** — every lab uses a fixed seed, so two
runs on your machine print the same thing. The last digits can still differ between
machines and between numpy or PyTorch versions. Trends matter, exact digits do not.

## Checking all the labs at once

```bash
bash scripts/labs-check.sh
```

runs every lab in its quick mode (about half a minute in total) and fails
if one crashes or stops printing the line that carries its lesson. The same script runs
on every pull request. On Windows it needs Git Bash or WSL; the labs themselves do not.
