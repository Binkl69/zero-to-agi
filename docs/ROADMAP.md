# Roadmap: from "informed non-practitioner" to a road that does not stop at 4

The 15 chapters as of v16 land a committed reader at roughly **Level 4** on this ladder:

| Level | | |
|---|---|---|
| 1 | Consumer — uses AI, no model of how it works | |
| 2 | Informed user — knows prompting, knows it hallucinates | |
| 3 | Conversant — follows technical discussion, evaluates claims | reading only |
| 4 | Technical generalist — understands the mechanism end to end, reads and modifies the code | **reading + labs** |
| 5 | Junior practitioner — fine-tunes and ships models for real tasks | the goal of D + E |
| 6 | Practitioner — owns ML systems in production | reachable, not from a course alone |
| 7 | Researcher — advances the field | years of original work; no course does this |

**The design goal is not to deliver Level 7.** It is that nothing in the course *caps* the
reader: no simplification that must later be unlearned, no hand-waving where the real thing is
within reach, and every chapter honest about where the frontier actually is. A course can end at
Level 4 with a wall, or end at Level 4 with a clearly marked road to 6. This is the second.

## Known gaps being closed

| | Workstream | Status |
|---|---|---|
| A | Chapter 1 enrichment — real decisions, not only toy switches | in progress |
| B | Maths spine woven through all existing chapters | in progress |
| C | Classical ML (2 chapters) | to do |
| D | Production reality (3 chapters) | to do |
| E | More labs — fine-tuning, RAG, agents, eval harness | to do |

---

## B. The maths spine

### The rule

**Notation after the intuition, never before.** This is the `Jargon after the experience` rule in
`CHAPTER_CONTRACT.md`, applied to symbols. A symbol may only appear after the reader has already
done the thing it describes, and its job is to *name that experience*, not to introduce it.

- **No standalone maths chapter.** A maths block is where a reader quits.
- **One maths beat per chapter**, 3–5 minutes, attached to something their hands already did.
- **Maximum size:** one interactive plus one short callout. If it needs more, it is in the wrong
  chapter or it is not earning its place.
- Every beat must be reachable by a reader who has done the preceding chapters and nothing else.

### The delivery mechanism: the symbol decoder

`ctx.decoder(...)` renders a formula where every symbol is clickable. Clicking a symbol
highlights the part of the demo the reader has just used that the symbol refers to, in plain
words. Reading notation without flinching is the concrete, checkable skill that separates
"I have heard of attention" from "I can read the paper".

### Reader profile this is calibrated for

Failed calculus repeatedly; aced practical and applied statistics a decade ago. This is a
common and highly informative profile:

- The calculus modern AI needs is **two ideas** — a derivative is a slope, and the chain rule
  means multiply the slopes along the chain. Multivariable/vector calculus (the usual Calc 3
  wall: surface integrals, divergence, curl) is essentially absent from this material.
- **Most of the maths here is statistics**: cross-entropy, softmax, KL divergence, perplexity,
  distributions, sampling, power laws, and Bradley–Terry (which *is* logistic regression).
  Lead with the statistical framing wherever both are available.

### Where each beat goes

Sequenced so each depends only on earlier ones.

| Ch | Beat | Names something already done |
|---|---|---|
| 01 | coordinates; a line as `w·x + b`; what a weight and a bias are | the line they dragged |
| 02 | derivative = slope; the chain rule = multiply along the chain | the fog demo; the reused ∂L/∂y |
| 03 | expectation and average; variance; why `−log p` | the U-curve; the loss explorer |
| 04 | summation notation `Σ`; indices over a grid | the nine filter cells they multiplied |
| 05 | recurrence; exponents and decay (`0.9^50`) | the LSTM belt; the vanishing gradient |
| 06 | vectors, dot product, norm, cosine | the arrows they dragged |
| 07 | matrix multiply; softmax; why `÷√d` | attention computed by hand |
| 08 | probability distributions; sampling; KL divergence | the needle-in-a-haystack; guidance |
| 09 | expectation over trajectories; geometric series (discounting) | the γ corridor |
| 10 | logs, power laws, and reading a log–log plot | the scaling explorer |
| 11 | the sigmoid; Bradley–Terry as logistic regression | the reward model they trained by clicking |
| 12 | growth rates and why `n²` hurts | the KV-cache and attention-cost charts |
| 13 | (labs carry their own notation) | — |
| 14 | exponential vs linear growth, on a log axis | the compute curve |
| 15 | doubling times and what they do and do not imply | the task-horizon chart |

---

## C. Classical ML

Two chapters. The course currently goes perceptron → neural networks → transformers, which
leaves a reader unable to recognise the majority of ML actually running in industry.

- **C1 — Learning without neural networks.** Decision trees, random forests, gradient boosting
  (XGBoost/LightGBM), k-NN, k-means, linear and logistic regression. Entropy and information
  gain as the maths beat.
- **C2 — Choosing the right tool.** When a neural network is the wrong answer: tabular data,
  small datasets, interpretability requirements, latency and cost budgets, regulated settings
  where you must explain a decision. Includes a head-to-head the reader runs themselves.

## D. Production reality

Three chapters. This is the main Level 4 → 5 lever.

- **D1 — Data engineering.** Where training data actually comes from, labelling operations,
  pipelines, versioning, leakage in practice.
- **D2 — Evaluation you can trust.** Designing an eval set, offline vs online, A/B tests,
  regression suites, LLM-as-judge and its failure modes.
- **D3 — Deployment and life after launch.** Serving, latency budgets, cost control,
  monitoring, drift, incidents, rollback, and the feedback loop back into D1.

## E. Labs

Beyond the existing eight: fine-tuning with LoRA on data the reader collects, a RAG system over
their own documents, an agent with real tools, and an evaluation harness. This is where skill
accrues; the chapters only make it legible.
