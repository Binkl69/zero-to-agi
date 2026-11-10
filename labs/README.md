# The labs

Reading about a neural network and building one are different experiences, and only
the second one sticks. These twelve labs build the real machinery, small enough to
read in one sitting and slow enough to watch. The first eight are about how a model
works. The last four are about using one: fine-tuning it, giving it documents, giving
it tools, and finding out whether any of that actually helped.

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

Python 3.10 or newer. Labs 01–04, 07 and 11 are pure Python and need nothing installed
at all; 05, 08, 10 and 12 need numpy; only labs 06 and 09 need PyTorch. Nothing needs
an API key or an internet connection. If the PyTorch download is
enormous (on Linux it can be several gigabytes of GPU libraries), see Troubleshooting
for the small CPU-only install.

The commands here are written from the repository root, but every lab finds its own
files, so `cd labs` followed by `python 01_perceptron.py` works just as well.

## The ladder

Do them in order. Each one assumes the previous one.

| Lab | What you build | Needs | Time |
| --- | --- | --- | --- |
| 01 `01_perceptron.py` | the 1958 perceptron, and why it fails on XOR | nothing | instant |
| 02 `02_autograd.py` | the engine behind `loss.backward()` | nothing | instant |
| 03 `03_mlp.py` | a network trained by that engine | nothing | ~1 min, `--quick` ~10 s |
| 04 `04_tokenizer_bpe.py` | a byte-pair tokenizer | nothing | under 1 s |
| 05 `05_attention.py` | attention and a transformer block | numpy | under 1 s |
| 06 `06_tiny_gpt.py` | a GPT that writes Shakespeare | PyTorch | `--quick` ~20 s, full ~30 min |
| 07 `07_q_learning.py` | an agent that learns from reward | nothing | instant |
| 08 `08_toy_rlhf.py` | preferences, a reward model, RLHF and DPO | numpy | ~1 s |
| 09 `09_lora_finetune.py` | LoRA and full fine-tuning of lab 06's GPT | PyTorch | `--quick` ~30 s, full ~1 min |
| 10 `10_rag.py` | retrieval over a handbook, measured | numpy | under 1 s |
| 11 `11_agent_tools.py` | an agent loop with tools, retries and guards | nothing | ~7 s |
| 12 `12_eval_harness.py` | an evaluation harness with error bars | numpy | ~2 s |

Labs 09–12 use no real language model and no network. Where a real system would
call an LLM, the lab uses a small stand-in, says so, and marks the one function you
would replace.

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

## From understanding to doing

The first eight labs explain the machine. These four are the things people actually
do with one. Each prints measured numbers, and each lists experiments with the result
you should get.

### 09 · `09_lora_finetune.py` — fine-tune a model you already have · PyTorch · ~30 s quick, ~1 min full

Takes lab 06's tiny GPT (it imports that file rather than copying it), pretrains it
briefly on Shakespeare, and then teaches it a second style: 7 KB of plain modern
weather forecasts in `labs/data/weather_reports.txt`. It does this twice, on the same
batches: once by full fine-tuning, and once with LoRA written by hand — a frozen
weight table plus a small trainable correction `(alpha / r) · B A`, with `B` starting
at zero.

```bash
python labs/09_lora_finetune.py --quick
```

```bash
python labs/09_lora_finetune.py
```

With `--quick` you should see: LoRA trains 3,072 numbers against 108,352 for full
fine-tuning (2.8%); loss on held-out weather falls from 2.44 to 2.27 with LoRA and to
2.06 with full fine-tuning; loss on held-out Shakespeare rises by 0.18 with LoRA and
by 0.40 with full fine-tuning (catastrophic forgetting, measured); unplugging the
adapter restores the base model exactly; and merging the adapter into the weights
gives the same outputs to within floating-point rounding. Without `--quick` the base
model is better and the fully fine-tuned model's samples become recognisably
forecast-shaped ("The low the eard in the lours ... a storm").

**What to try:** `--rank 1` and `--rank 16` (more rank learns more and forgets more);
`--rank 16 --targets all`, where LoRA forgets as much as full fine-tuning did;
`--full-lr 0.0001`, where full fine-tuning forgets *less* than LoRA. Together those
two say something the usual summary of LoRA gets wrong: it does not protect old
knowledge, it packages the change so that it is small and removable.

**Teaches:** fine-tuning, low-rank adapters, trainable versus frozen parameters,
catastrophic forgetting, merging, and why you always re-measure the old task.

### 10 · `10_rag.py` — give a model documents it never saw · numpy · under 1 s

Retrieval-augmented generation with the generation left honestly empty. You chunk a
small handbook (`labs/data/library_handbook.txt`, an invented library's rules), build
TF-IDF vectors and cosine similarity by hand, and retrieve the best chunks for 32
questions whose answers are known — so "retrieval worked" is a number. The answer step
is a stand-in that picks the best-matching sentence; the lab says plainly that a real
system hands the retrieved chunks to an LLM at that point, and marks the function.

```bash
python labs/10_rag.py
```

You should see: 35 chunks of 40 words; 16 of 20 questions answered by the top 3
chunks when the question uses the handbook's own words, and 0 of 12 when it was
deliberately written in other words ("bicycle" where the handbook says "cycle");
a chunk-size table in which 5-word chunks and 320-word chunks both do badly on a
fixed word budget (25% and 19%) and cutting by section does best (69%); a table for
k in which the hit-rate rises while the share of relevant text handed over falls
from 41% to 4%; and a stand-in answer step that gets 13 of the 16 questions whose
passage was retrieved, and none of the other 16.

**What to try:** `--chunk-words 8` (hits fall from 16 to 7 of 20, mostly because
answers are cut in two); `--overlap 20` (16 becomes 19); `--no-idf` (16 becomes 13);
`--by-section --expand` (retrieval reaches 32 of 32, and the answer step, at 25 of 32,
is now the weak part); and `--ask "your own question"`.

**Teaches:** chunking, TF-IDF, cosine similarity, top-k retrieval, hit-rate, why
keyword search misses synonyms and what embeddings are for, and why most RAG failures
are retrieval failures.

### 11 · `11_agent_tools.py` — a model that acts · pure Python · ~7 s

An agent loop with real tools: a calculator that parses arithmetic instead of running
code, a search over a small catalogue, a note store, and one dangerous tool that
emails a note. The loop parses and validates every tool call, turns failures into
messages the model can read, retries tools that time out, enforces a step budget, and
prints a transcript. The "model" is a rule-based planner of about sixty lines, so the
lab is free, offline and exactly repeatable; the file marks `policy()` as the one
function a real agent replaces with an LLM call.

```bash
python labs/11_agent_tools.py
```

You should see: a quote worked out in 5 model calls and 4 tool calls (124.20); eight
malformed replies refused, each with a reason; a search that fails 30% of the time
succeeding 47% of the time with no retries and 99% with three; a table in which a 5%
slip per step leaves a 10-step task succeeding 58% of the time and a 50-step task 8%
(0.95^50 is 7.7%); and a prompt injection — a catalogue entry that contains an
instruction — which a naive loop obeys, emailing a private note to a stranger while
still returning the right answer, and which a guarded loop ignores.

**What to try:** `--error-rate 0.01` (the 50-step task goes from 8% to 61%);
`--retries 0`; `--budget 4` (the agent runs out of steps one short of answering);
`--guard gate` (the model is fooled, the loop blocks the email, and the task fails:
the gate turned a theft into a failure, not into a success); `--flaky 0.6`.

**Teaches:** the agent loop, tool schemas, parsing and validation, retries, step
budgets, compounding error, prompt injection, and least privilege.

### 12 · `12_eval_harness.py` — find out whether it worked · numpy · ~2 s

An evaluation harness run on simulated models whose true accuracy we choose (70% and
74%), so you can see exactly when the harness is wrong. It scores them, puts bootstrap
confidence intervals on the scores, compares them paired and unpaired, runs a
regression suite over slices, audits a simulated LLM judge, and shows what leaked test
items do to a score.

```bash
python labs/12_eval_harness.py
```

You should see: scores of 73.1% and 75.3% on 1,000 items (the truth is 70% and 74%);
an interval of about ±15 points at 20 items and ±2.8 at 1,000; the better model ahead
in only 56% of 20-item evals but 99% of 1,000-item ones; a paired interval 24%
narrower than the unpaired one on the same answers — and, in this default draw,
neither interval excluding zero, because 1,000 items detect a 4-point gap only 72% of
the time (paired) or 51% (unpaired); a "fix" that raises the overall score by 4.4
points while making one slice 21 points worse; a judge that picks the first answer
61% of the time and prefers a wordier, slightly worse model on 65% of pairs; and a 60%
model scoring 96.7% on a fully leaked test set.

**What to try:** `--seed 1`, `2`, `3` (which conclusions survive?); `--gap 2` (half
the gap needs about four times the data); `--difficulty-spread 0` (pairing stops
helping); `--position-bias 0` (the order-swap test goes quiet and the length bias is
still there); `--memorised 0.75`.

**Teaches:** scores as measurements, bootstrap intervals, sample size, paired
comparison, slices and regression suites, judge bias, contamination.

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
  LoRA on a single consumer GPU, often in under an hour. Lab 09 is the toy; this is
  the tool.
- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** —
  build and run evals. Writing a good eval is one of the highest-leverage things a
  newcomer can contribute, and lab 12 is the habits it needs.

Chapter 15 of the course lays out a twelve-month plan built around these.

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`** — only labs 06 and 09 need it. On Windows
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

runs every lab in its quick mode (under a minute in total) and fails
if one crashes or stops printing the line that carries its lesson. The same script runs
on every pull request. On Windows it needs Git Bash or WSL; the labs themselves do not.
