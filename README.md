# Zero → AGI

**An interactive, visual course on how AI actually works** — from a single artificial neuron in 1958 to
the frontier models of 2026, how they are trained, how to build your own, and what still separates us
from AGI.

Built for one specific reader: a smart adult who feels the field ran away from them in the last five
years and wants to genuinely *understand* it, not just read headlines. The course assumes no maths
beyond high-school and no machine-learning background. It is built on the belief that experience and
exposure are the best teacher, so every chapter has things you can poke at: live-training networks,
draggable points, sliders, games, and a language model that trains in your browser tab.

## Read it now

**→ [binkl69.github.io/zero-to-agi](https://binkl69.github.io/zero-to-agi/)**

Or run it locally. No build step, no dependencies. Open `index.html` in any modern browser, or serve
the folder:

```bash
python -m http.server 8000
```

then visit http://localhost:8000.

Progress (completed chapters, learning-path milestones) is saved in your browser's local storage.

## What's inside

| Part | Chapters | You will be able to… |
|---|---|---|
| **I · Foundations** | 1 What does it mean for a machine to learn? · 2 Neural networks & backprop · 3 The recipe: data, loss, optimizer, generalization | explain a loss function and gradient descent, and watch a network learn XOR |
| **II · The Model Zoo** | 4 CNNs · 5 RNNs & LSTMs · 6 Embeddings · 7 Transformers & attention · 8 GANs, VAEs & diffusion · 9 Reinforcement learning | say why each architecture exists and what problem it fixed |
| **III · How We Teach Models** | 10 Pretraining · 11 Post-training (SFT, RLHF, DPO, Constitutional AI, reasoning RL) · 12 Inference, tools & agents | draw the whole pipeline from raw web text to an assistant like Claude |
| **IV · Build It Yourself** | 13 From one neuron to a tiny GPT (+ the Python labs) | train real models on your own machine |
| **V · The Frontier** | 14 80 years of AI in one timeline · 15 What separates us from AGI, and how you could help build it | judge the AGI debate on the merits and have a concrete personal roadmap |

Plus a searchable **glossary** of every term used, and a **learning path** page with milestones you
can tick off.

## The Python labs (`labs/`)

Reading is not enough. The labs build the real thing, small:

| Lab | Builds | Needs |
|---|---|---|
| `01_perceptron.py` | Rosenblatt's 1958 perceptron; watch it fail on XOR | Python only |
| `02_autograd.py` | a scalar autograd engine (what PyTorch does under the hood), gradient-checked | Python only |
| `03_mlp.py` | a multi-layer network trained with *your* autograd | Python only |
| `04_tokenizer_bpe.py` | byte-pair encoding, the tokenizer family behind GPT models | Python only |
| `05_attention.py` | scaled dot-product & multi-head attention, a full transformer block, GPT-2's parameter count | numpy |
| `06_tiny_gpt.py` | a character-level GPT trained on Shakespeare | PyTorch |
| `07_q_learning.py` | a Q-learning agent in a gridworld, plus reward hacking | Python only |
| `08_toy_rlhf.py` | a reward model trained from preferences and a policy optimised against it (RLHF, then DPO) | numpy |

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python labs/01_perceptron.py
```

See [labs/README.md](labs/README.md) for the full guide, expected runtimes and what to try next.

## Suggested pace

Roughly one chapter per sitting, labs interleaved as chapter 13 and the learning-path page suggest.
Three to six weeks at a few hours a week gets through the whole thing with real understanding.

## Repository layout

```
index.html              the app shell (loads every chapter as a plain <script>)
app/css/style.css       shared styles
app/js/core.js          chapter registry + helper toolkit (canvas, sliders, loops, quizzes)
app/js/main.js          router, sidebar, home / learning path / glossary pages
app/js/glossary.js      170-term glossary
app/chapters/NN-*.js    one self-contained file per chapter
labs/                   Python labs
scripts/smoke.js        headless test harness for the chapters
docs/CHAPTER_CONTRACT.md  how chapters are written (useful if you want to add one)
```

## Testing

The chapters are plain JavaScript with no test framework, so there is a small harness
that renders every chapter into a fake DOM, ticks its animation loops, clicks every
button, drags every slider, and fails on any runtime error, missing quiz, blank canvas
or thin prose:

```bash
node scripts/smoke.js
```

Add a chapter id to test just one, for example `node scripts/smoke.js 07-transformers`.

## Contributing / extending

Chapters are independent files that register themselves; see `docs/CHAPTER_CONTRACT.md`. Corrections
to facts and dates are especially welcome — the field moves fast and this snapshot is from September 2026.

## License

MIT. The Shakespeare text used by lab 06 is public domain.
