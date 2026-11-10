# Reading and Practice

The companion to [`COMPETENCY_ROADMAP.md`](COMPETENCY_ROADMAP.md). That document says how many hours
and what they are for. This one says what happens inside them: what to read, how fast, what to
practise, and what gets handed in.

---

## 1. Why you feel slow

You are reading *Superintelligence* and feeling slow. The diagnosis matters more than the
reassurance, so here it is straight.

**You are not slow. You picked a hard book with poor returns for the milestone you are on.**

*Superintelligence* (2014) is dense analytic philosophy — long sentences, abstract arguments, heavy
qualification on every claim. It is slow for everyone; people who say otherwise skimmed it. It also
predates the entire transformer era. It is a careful argument about a hypothetical, written before
GPT-1 existed. It will not help you explain AI to your leadership, will not help you build anything,
and its central framing is now a minority position among working practitioners.

It is a serious book and worth reading one day. It is not worth blocking Milestone 1 on.

**Recommendation: park it.** Move it to a Sunday background thread — twenty minutes a week, no
target date, finish it in Phase 2 or don't. If you want that territory properly, read
**Brian Christian, *The Alignment Problem*** instead: the same ground, far more readable, and it
contains the RLHF history that connects directly to chapter 11 of the course.

And there is a deeper miscalibration underneath the feeling, which is worth fixing permanently:

> **You are measuring pages per hour. The field measures what you can do afterwards.**

Nobody has ever been hired, promoted, or consulted because they finished a book. Reading is an input
with a notoriously weak conversion rate, and "I read it front to back at a steady pace" is the
lowest-yield way to consume anything in a fast-moving field. The rest of this document is built
around that.

---

## 2. Three streams, three speeds

Most people read everything at one speed, which is why they feel slow at the hard things and learn
nothing from the easy ones. Split your reading into three streams and run each at its own pace.

| Stream | What it is | Purpose | Speed | Peaks in |
|---|---|---|---|---|
| **Mechanism** | The course, then papers | Understand how it actually works | Slow, deliberate, with a pen | Phase 1, then papers in 3 |
| **Craft** | Docs, cookbooks, other people's code | Be able to build the thing | Not "read" — *run* | Phase 2 |
| **Judgment** | Books, essays, policy | Language, framing, taste, what to refuse | Fast, extractive, ruthless | Phase 1, tapering |

The important consequence, stated plainly because it is counterintuitive:

> **Books are the worst hours-to-value ratio in AI right now — except for judgment, where they are
> the best.**

A technical AI book is roughly two years stale the day it prints. For learning to build, docs and
code beat books by a wide margin. But for *language, framing, and knowing which ideas are bad* —
which is the entire content of Milestone 1, educating leadership — books are unmatched, because
that is the part of the field that moves slowly.

So: books front-load into Phase 1 and mostly stop. Papers and documentation take over in Phase 2
and never leave. If you are still working through books in Phase 3, something has gone wrong.

---

## 3. The extraction protocol

### Books: the three-hour rule

Never read a non-fiction AI book front-to-back at page-one pace. Instead:

1. **Introduction, conclusion, and every chapter's first and last page.** Thirty minutes. You now
   know the argument.
2. **Decide which chapters actually earn full reading.** Usually two to four. The rest you have
   already extracted.
3. **Read those properly**, with a pen. Two hours.
4. **One page of notes, in your own words, maximum.** If you cannot fill the page, you did not read
   it. If you need more than a page, you are transcribing rather than thinking.

**Time-box: three hours to a book's usable content.** If it genuinely needs more, it is a reference
work, not a read — keep it on the shelf and go to it with questions.

The test of a finished book is not that you finished it. It is: **can you use a sentence from it in
a leadership conversation next week?** If not, the three hours were the correct budget and you
already spent enough.

### Papers: three passes

Keshav's method, and it is the single highest-leverage reading skill in this plan.

| Pass | Time | What you read | What you decide |
|---|---|---|---|
| 1 | 5–10 min | Title, abstract, intro, section headings, conclusion, figures | Whether there is a pass 2 |
| 2 | ~1 hour | Figures, tables, results. Skip the maths. | Whether there is a pass 3 |
| 3 | 4+ hours | Everything, reimplementing the core idea as you go | — |

**Most papers stop at pass 1, and that is correct, not lazy.** Over two years perhaps forty papers
get pass 2 and **no more than six get pass 3.** Choosing which six is a real skill and the whole
point of the method.

Here is the thing to notice about feeling slow: **you have been giving pass-3 effort to pass-1
material.** Reading every sentence of everything with equal care is not thoroughness, it is an
absence of triage, and it is exhausting precisely because it wastes most of the effort.

### Every stream: the four-tier shelf

Sort everything on arrival. Most of what crosses your desk is tier 3 or 4 and the mistake is
treating it as tier 1.

- **Read** — properly, with notes. Rare.
- **Skim** — the three-hour rule, or pass 1.
- **Reference** — do not read. Go to it with a question, later.
- **Skip** — say so out loud. Deciding not to read something is a decision, not a failure.

---

## 4. The reading list

Tiered. `[R]` read properly · `[S]` skim under the three-hour rule · `[X]` reference only.

### Phase 1 · Weeks 1–13 — judgment and mechanism

**Primary: the course.** Chapters 1–15 and labs 01–08. This is the mechanism stream and it is the
bulk of the phase. Everything below is the judgment stream, at roughly 3 hours a week.

| | Book | Why, for you specifically |
|---|---|---|
| `[R]` | **Narayanan & Kapoor — *AI Snake Oil*** (2024) | **Read this first.** Its core subject is predictive AI failing at exactly the tasks HR wants it for: hiring, performance prediction, attrition. It hands you the Phase 1 kill test directly. Probably the highest-value book on this list for your goal. |
| `[R]` | **Mollick — *Co-Intelligence*** (2024) | The best available book for "educate leadership." Practical, organisational, fast. Read it with your own function in mind and it half-writes your week 4–6 explainer. |
| `[S]` | **Suleyman — *The Coming Wave*** (2023) | Geopolitical and policy framing. Skim it. You want the vocabulary for national-strategy conversations, not the argument. |
| `[S]` | **Christian — *The Alignment Problem*** (2020) | Readable narrative history of alignment, including RLHF's origins. Pairs directly with chapter 11. This is your *Superintelligence* substitute. |
| `[X]` | **Bostrom — *Superintelligence*** (2014) | Parked. Sunday background thread, twenty minutes, no deadline. |

**Papers — pass 1 and 2 only, three of them:**
*Attention Is All You Need* (2017) · *InstructGPT* (2022) · one paper on bias in automated résumé
screening. Do not attempt pass 3 in Phase 1; the labs are doing that job better.

**Running sources** (30 min/week, not more): Ethan Mollick's *One Useful Thing* · Simon Willison's
blog — the single best source for "what actually happened this week" · Anthropic's engineering blog
and *Core Views on AI Safety*.

### Phase 2 · Weeks 14–52 — craft

Books nearly stop here. Your primary reading becomes documentation and other people's code, and
most of it happens with your hands on the keyboard rather than in a chair. Budget roughly
**1.5 hours a week** (the Thursday block) for deliberate reading; the rest is absorbed into building.

| | Source | Notes |
|---|---|---|
| `[R]` | **Huyen — *AI Engineering*** (O'Reilly, 2025) | The book for this phase. Building applications on foundation models: prompting, RAG, fine-tuning, agents, dataset engineering, evaluation. ~532pp — read it across the phase, one chapter per project, not in one run. |
| `[X]` | **Huyen — *Designing Machine Learning Systems*** (2022) | Reference for the production-systems half: pipelines, monitoring, drift. Go to it when P2.4 forces you to. |
| `[R]` | **Karpathy — *Neural Networks: Zero to Hero*** + **nanoGPT** | Not a book, and the most important entry here. It is the bridge from the course's labs to real code. Do it, don't watch it. |
| `[R]` | **Anthropic docs and cookbook** | Tool use, structured outputs, prompt caching, agent patterns, evals. This is a craft-stream primary source, not background. |
| `[X]` | **Géron — *Hands-On Machine Learning*** | Reference only. Do the chapters a project forces on you. Reading it cover to cover is a three-month detour. |

**Papers — pass 2, roughly one a week:** RAG · LoRA · DPO · Constitutional AI · Chinchilla
(compute-optimal scaling) · *Judging LLM-as-a-Judge* (MT-Bench) · ReAct and Toolformer.
**Pass 3 on exactly two of them**, chosen for whichever project needs them most.

### Phase 3 · Weeks 53–104 — frontier

Papers only, plus the two textbooks of your specialty. Roughly **3 hours a week** across the Monday
and Thursday blocks.

| | Source | Notes |
|---|---|---|
| `[R]` | **Barocas, Hardt & Narayanan — *Fairness and Machine Learning*** (free at fairmlbook.org) | **The textbook of your specialty.** Read it properly, over the phase. If you read one thing in Phase 3, this. |
| `[R]` | Kleinberg, Mullainathan & Raghavan (2016); Chouldechova (2017) | The fairness impossibility results. Short. See §6 for why these two papers are worth more to you than almost anything else. |
| `[X]` | FAccT and NeurIPS proceedings; targeted arXiv alerts | Track your specialty, not the field. Follow perhaps eight researchers, not eight hundred. |

**FAccT** (ACM Conference on Fairness, Accountability and Transparency) is the venue for your
specialty and a realistic submission target — considerably more reachable for your profile than
NeurIPS, and much better aimed. Read two years of its proceedings in Phase 3, and treat a workshop
submission as the week-90 stretch goal.

---

## 5. Weekly drills

Reading is an input. These convert it. Each fits inside a block you already have — most are 20–40
minutes — and each one has a failure mode that tells you something.

**1 · The explain-it drill** *(weekly, Phase 1–2, 30 min)*
Pick one concept from the week. Write 200 words explaining it with **no analogy at all**. Then
write 50 words with exactly one analogy. Post both.
*Why:* the no-analogy version is the Phase 1 metric. If you cannot write it, you have understood the
analogy and not the thing — which is the specific failure mode Milestone 1's anti-metric names.

**2 · The prediction drill** *(before every lab and every experiment, 10 min)*
Before running anything, write down what you expect to happen. Numbers where possible. Then run it.
*Why:* the gap between your prediction and the result *is* your model of the system, made visible.
It is the cheapest diagnostic in the plan and almost nobody does it. Keep them all; the shrinking
gap over two years is your most honest progress record.

**3 · The ablation drill** *(weekly, Phase 1–2, 40 min)*
Break it on purpose. Multiply the learning rate by ten. Remove the positional encoding. Set
temperature to 0, then 2. Delete half the training data. Predict first, then observe.
*Why:* you learn what a component does by removing it, not by reading about it. This is also
literally how research is done, so the habit transfers.

**4 · The red-team drill** *(weekly from Phase 2, 40 min)*
Attack your own system. Adversarial CVs. Prompt injection hidden in a policy document. Out-of-scope
questions asked confidently. Identical candidates with swapped names.
*Why:* a skill in Phase 2 and a research method in Phase 3 — the last of those examples is your
benchmark's core test case, discovered by drilling.

**5 · The teaching drill** *(monthly, Phase 2–3, one Saturday block)*
Write a section of the course. The course's own [`ROADMAP.md`](ROADMAP.md) lists its gaps — C1–C2
(classical ML), D1–D3 (data engineering, evaluation, deployment), and workstream E (labs on
fine-tuning, RAG, agents, eval harnesses).
*Why:* **that gap list is your assignment list.** You will be learning exactly those five things in
Phase 2, in that order, and teaching is the cheapest available proof of understanding. Your practice
output improves the course, and by Phase 3 you are the author of the material you learned from —
which is itself a credential.

---

## 6. Your moat: what no AI engineer knows

This section is short and it is the most valuable page in either document.

Everything in §4 is available to anyone. The reading below is what makes you *irreplaceable* rather
than merely competent, and it is reading you are uniquely positioned to absorb quickly because you
already have the domain context that makes it legible.

### Selection validity — and the revision nobody in AI noticed

Schmidt & Hunter (1998), *The validity and utility of selection methods in personnel psychology*,
was the canonical hierarchy of what predicts job performance for twenty-four years. Then Sackett,
Zhang, Berry & Lievens (2022) showed that the range-restriction correction at the heart of that
method had been **systematically overcorrecting**, and revised the estimates down substantially —
general mental ability from roughly r = .51 to r = .31, with most predictors falling .10 to .20.

Read both. Here is why it matters more than it looks:

1. **The human baseline your AI system must beat is weaker and far more contested than anyone in AI
   realises.** A model that predicts job performance at r = .35 sounds unimpressive right up until
   you know what the revised human-interview baseline actually is.
2. **Most people building résumé screeners will cite the 1998 numbers**, because those are the
   numbers in every HR textbook and every vendor deck. They will be wrong in public. You will not.
3. It is a live demonstration of the thing your whole specialty is about: a measurement correction
   quietly invalidated a field's headline numbers for two decades. That is the argument for
   assurance, and you can make it with a real example instead of a hypothetical.

No amount of ML skill produces this knowledge. It took you a fortnight's reading and a career's
context.

### The fairness impossibility results

Kleinberg, Mullainathan & Raghavan (2016) and Chouldechova (2017), independently, proved that the
reasonable definitions of fairness **cannot all be satisfied at once** except in degenerate cases.
Calibration, equal false-positive rates, and equal false-negative rates are mutually incompatible
when base rates differ.

Both papers are short. Give them pass 3.

Why this is the highest-value reading in the plan: it means **every regulation about fair AI has
implicitly chosen a fairness definition, and almost nobody who writes such regulations knows they
made that choice.** Being the person in the room who can say "your rule as drafted requires equal
false-negative rates, which means it forbids calibration, which means here is what it will do in
practice" is worth more than any other single sentence available to you. That is not consulting.
That is being consulted.

### The regulatory template

- **NYC Local Law 144** — bias audits for automated employment decision tools, in force since July
  2023. The closest existing template for what a UAE regime would plausibly require and for what
  your Phase 3 benchmark should measure. Read the law and two published audits.
- **EU AI Act**, employment high-risk provisions. The reference framework everyone else copies.
- **The UAE framework** — verify at source during Phase 1, as the roadmap says. Do not rely on
  secondary summaries, including this repository's.

---

## 7. Phase assignments

Each phase hands in artifacts and then **defends them to a person**. The defence is not optional and
it is not a formality — unexamined work reliably turns out to be weaker than it felt, and finding
that out from a reviewer in week 30 is cheap where finding it out from a regulator in week 90 is not.

| Phase | Hand in | Defended to | Pass bar |
|---|---|---|---|
| 1 | The 6-slide explainer · the HR AI Opportunity Map · the draft use policy · two written kill cases | Your leadership, live, 45 min | One decision demonstrably changed |
| 2 | P2.1–P2.4, each with its eval set, numbers, and failure modes · four course sections written | A working ML engineer, paid if necessary | Their findings are about trade-offs, not fundamentals |
| 3 | The benchmark, public · six write-ups · one training run documented end to end · one distillation result | A researcher in your specialty; then a policy audience | Treated as a peer on your specialty |

Two notes on the bars. The Phase 2 bar is deliberately about the *reviewer's findings* rather than
their verdict, because "it's good" is unfalsifiable and "you've misunderstood how retrieval scoring
works" is information. And **pay for that review if you have to.** A few hundred dollars for two
hours of a competent engineer's attention at week 45 is the highest-return spend in this plan, and
the failure mode it prevents — a year of confident work built on one wrong assumption — is the most
expensive one available to you.

---

## 8. Where the reading hours actually sit

The roadmap's blocks, with the streams assigned. This is the version you execute on a Monday
morning without deciding anything.

**Phase 1 — 10 h/week**

| Day | h | Block |
|---|---|---|
| Mon | 1.5 | Course chapter — do the demos, don't just read them |
| Tue | 1.5 | Course chapter or lab |
| Wed | 1.5 | Judgment stream — the books, three-hour rule |
| Thu | 1.5 | Judgment stream, or the explain-it drill |
| Fri | — | *(off in Phase 1 — build the habit before loading it)* |
| Sat | 3.0 | Lab block · prediction drill first, ablation drill after |
| Sun | 1.0 | Write up · update metrics · plan next week's blocks |

**Phase 2 — 12 h/week**

| Day | h | Block |
|---|---|---|
| Mon | 1.5 | Craft stream — one new technique, from docs |
| Tue | 1.5 | Build |
| Wed | 1.5 | Build |
| Thu | 1.5 | One paper at pass 2, or a chapter of *AI Engineering* |
| Fri | 2.0 | The hard thing |
| Sat | 3.0 | Build block — ends in a commit · red-team drill |
| Sun | 1.0 | Write up · metrics · plan |

**Phase 3 — 12 h/week + 4 sprint weeks**

| Day | h | Block |
|---|---|---|
| Mon | 1.5 | Specialty papers, pass 2–3 |
| Tue | 1.5 | Training / experiment |
| Wed | 1.5 | Training / experiment |
| Thu | 1.5 | Benchmark and artifact work |
| Fri | 2.0 | The hard thing — usually the writing |
| Sat | 3.0 | Build / train block |
| Sun | 1.0 | Publish · network · plan |

Total deliberate reading: about **3 hours a week in Phase 1**, dropping to **1.5 in Phase 2** as
craft absorbs it, back to **3 in Phase 3** as papers take over. Across two years that is roughly
**250 hours of reading** out of 1,190 — a fifth. If reading is taking more than that, the
extraction protocol is not being applied, and the fix is triage rather than speed.
