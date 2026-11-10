"""
LAB 10 - Retrieval-augmented generation, without the generation
===============================================================

Run:   python labs/10_rag.py
       python labs/10_rag.py --chunk-words 8        (experiment 1, and so on)
Needs: numpy.
Data:  labs/data/library_handbook.txt (a 7 KB handbook for an invented
       library, written for this lab)
Time:  under a second.

WHAT YOU'LL BUILD
-----------------
A language model only knows what was in its training data. It has never read
your company's handbook, your notes, or yesterday's news. Retrieval-augmented
generation (RAG) is the standard fix, and it is simpler than its name:

    1. CHUNK     cut your documents into pieces
    2. INDEX     turn every piece into a vector of numbers
    3. RETRIEVE  turn the question into a vector the same way, and fetch
                 the k pieces whose vectors point in the most similar direction
    4. GENERATE  paste those pieces into the prompt and let the LLM answer
                 from them

This lab builds steps 1 to 3 for real, by hand, and MEASURES them: 32
questions about the handbook, each with a known answer, so "retrieval worked"
is a number and not a feeling.

THE HONEST PART: there is no LLM in this file, so step 4 is a stand-in. It
picks the single sentence from the retrieved chunks that best matches the
question and prints it. A real system hands the retrieved chunks to an LLM at
exactly that point (the function is `generate()`, and it is marked). The
lesson survives the substitution, because of a fact every RAG builder learns
the hard way: if the right passage was not retrieved, no model, however
good, can answer from it. Most RAG failures are retrieval failures.

The vectors here are TF-IDF: one number per word in the vocabulary, large
when a word is frequent in this chunk and rare elsewhere. Real systems
usually use learned "embedding" vectors instead (the wte table of lab 06,
grown up). TF-IDF needs no model and no download, it is still a strong
baseline, and its characteristic blind spot - it only matches words, not
meanings - is one of the things this lab measures.

WHAT TO OBSERVE
---------------
1. The handbook becomes 35 chunks of 40 words and a table of 35 x 552
   numbers, nearly all zero. That table is the whole "vector database".

2. Hit-rate with the 3 best chunks: 16 of 20 (80%) for questions that use
   the handbook's own words, and 0 of 12 for questions deliberately written
   in other words. Overall 50%.

3. Chunk size, on a fixed budget of 120 retrieved words: 25% at 5-word
   chunks, 50% at 40, 56% at 80, 19% at 320. Cutting at the handbook's own
   section breaks beats every fixed size: 69%.

4. k: the right chunk is in first place for only 65% of the own-words
   questions, in the top 3 for 80%, in the top 10 for 95%. But at k=10 only
   7% of what you hand over is relevant, against 41% at k=1.

5. "Is there anywhere to lock up my bicycle?" shares no words at all with
   "Cycle racks for 20 bikes are beside the east entrance". Similarity
   0.000. A hand-written synonym list lifts the other-words group from 0%
   to 67%; the lab says plainly why that is not a real solution.

6. The stand-in answer step gets 13 of 32 right: 13 of the 16 where the
   right passage was retrieved, and 0 of the 16 where it was not.

WHAT TO CHANGE (5 experiments - predict the outcome BEFORE you run)
-------------------------------------------------------------------
Each is a flag, so you do not have to edit the file.

1. --chunk-words 8. Own-words hits fall from 16 of 20 to 7. The list of
   misses shows why: for 7 of the 13, the evidence phrase itself has been
   cut in two by a chunk boundary, so no ranking could have found it.

2. --overlap 20 (each 40-word chunk shares 20 words with the next). Own-
   words hits rise from 16 to 19 of 20 and nothing is cut in two any more.
   The price: 68 chunks to store and search instead of 35. Overlap is the
   standard patch for boundary cuts.

3. --no-idf (count words, but do not down-weight common ones). Own-words
   hits fall from 16 to 13 of 20: "the", "is" and "how" now count as much
   as "lift" or "printing", so chunks win by sharing filler.

4. --expand, then --by-section --expand. The synonym list alone lifts the
   other-words group from 0 to 8 of 12. With section chunks as well,
   retrieval is perfect: 32 of 32. The answer step still gets only 25 of
   32, because picking one sentence is a poor substitute for reading. At
   that point retrieval is solved and the missing piece is exactly the
   LLM this lab does not have.

5. Ask your own: --ask "Is there anywhere to lock up my bikes?" (the
   handbook's word instead of "bicycle"). The right chunk appears, but
   only third: two chunks that merely share "there", "up", "is" and "to"
   outrank it. Now --ask "cycle racks": first place. Short keyword
   queries suit a word-matching index better than polite sentences do,
   which is why people learned to talk to search engines in keywords.

COMMON BUGS
-----------
* Scoring "retrieval worked" by reading three answers. Build the eval set.
* Comparing chunk sizes at the same k. Three 300-word chunks are not the
  same amount of text as three 30-word chunks; compare on a word budget.
* Forgetting to normalise the vectors: long chunks then win every search.
* Building the question vector with a different vocabulary or different
  IDF weights from the index. Both sides must be measured the same way.
* Blaming the model for an answer whose passage was never retrieved.
"""

import sys
import re
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):   # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "library_handbook.txt"

ap = argparse.ArgumentParser(description="Retrieval over a small handbook with hand-built TF-IDF.")
ap.add_argument("--chunk-words", type=int, default=40, help="words per chunk (default 40)")
ap.add_argument("--overlap", type=int, default=0, help="words shared by neighbouring chunks (default 0)")
ap.add_argument("--k", type=int, default=3, help="chunks retrieved per question (default 3)")
ap.add_argument("--no-idf", action="store_true", help="count words only; do not down-weight common ones")
ap.add_argument("--expand", action="store_true", help="add hand-written synonyms to every question")
ap.add_argument("--by-section", action="store_true", help="one chunk per section instead of fixed sizes")
ap.add_argument("--ask", default=None, help="ask your own question and see what is retrieved")
args = ap.parse_args()

CHUNK_WORDS = args.chunk_words
OVERLAP = args.overlap
K = args.k
USE_IDF = not args.no_idf
BUDGET = 120            # words of retrieved text we can "afford" to hand to the LLM


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# THE EVAL SET
#
# Each question comes with EVIDENCE: a short phrase that appears exactly once
# in the handbook, in the passage that answers the question. Retrieval "hits"
# if one of the retrieved chunks contains that phrase.
#
# The first 20 questions use the handbook's own words. The last 12 ask for the
# same kind of fact in DIFFERENT words, the way a real person does when they
# have not read the document: "bicycle" where the handbook says "cycle",
# "laptop" where it says "wireless internet".
# ---------------------------------------------------------------------------

SAME_WORDS = [
    ("What time does the library close on Saturdays?",          "close at 4 pm"),
    ("How many books can an adult member have on loan?",        "up to 12 books"),
    ("How long is the loan period for a book?",                 "is 3 weeks"),
    ("How much does it cost to borrow a film?",                 "1 pound each"),
    ("What is the fine per day for an overdue item?",           "20p per day"),
    ("How much does a replacement card cost?",                  "costs 2 pounds"),
    ("How long will a reserved item be held at the desk?",      "for 7 days"),
    ("How many public computers are there?",                    "14 public computers"),
    ("What is the wireless network called?",                    "FernhillGuest"),
    ("How much does colour printing cost?",                     "40p a page"),
    ("How many people does a study room seat?",                 "seats 6 people"),
    ("When is story time for children?",                        "Wednesday at 10:30 am"),
    ("When does the reading group meet?",                       "first Monday of each month"),
    ("Which bus routes stop outside the library?",              "routes 7 and 22"),
    ("How many spaces does the car park have?",                 "30 spaces"),
    ("When is the local history room open?",                    "Tuesdays and Thursdays from 1 pm"),
    ("How long is lost property kept at the main desk?",        "for 28 days"),
    ("What is the phone number of the enquiry desk?",           "01632 960 114"),
    ("How old must volunteers be to apply?",                    "aged 16 or over"),
    ("How many bags of donated books can I bring?",             "up to 2 bags"),
]

OTHER_WORDS = [
    ("What is the penalty for handing a novel in late?",        "20p per day"),
    ("Can I take my dog in with me?",                           "other pets must wait outside"),
    ("Is there anywhere to lock up my bicycle?",                "Cycle racks"),
    ("How do I get online with my own laptop?",                 "FernhillGuest"),
    ("Where can I buy a snack?",                                "tea, coffee and cake"),
    ("I am too unwell to visit. Can somebody drop novels round?", "ask for Margit"),
    ("Do you have old photos of the town?",                     "going back to 1840"),
    ("What happens if I rip a page?",                           "count as damage"),
    ("Can I sign up if I only commute here for my job?",        "lives, works or studies"),
    ("Somebody was shouting and swearing. Can they be thrown out?", "leave for the day"),
    ("Are there sessions for toddlers?",                        "Wednesday at 10:30 am"),
    ("My purse was mislaid yesterday. Where will it be?",       "kept at the main desk"),
]

# Experiment 5 (--expand): a hand-written synonym list. For every word on the
# left that appears in a question, the words on the right are added to it.
# This is a crude, manual stand-in for what learned embeddings do on their own.
SYNONYMS = {
    "penalty": "fine charge", "late": "overdue due", "novel": "book item", "novels": "books",
    "dog": "pets animals", "bicycle": "cycle bikes", "online": "wireless internet network",
    "laptop": "wireless internet", "snack": "food cake cafe", "unwell": "illness",
    "drop": "brought delivery", "photos": "photographs", "town": "district", "rip": "torn damage",
    "sign": "join membership", "job": "works", "shouting": "abusive disturbs", "thrown": "leave ban",
    "toddlers": "children under", "purse": "wallets", "mislaid": "left lost property",
}


# ---------------------------------------------------------------------------
# STEP 1: CHUNK
# ---------------------------------------------------------------------------

def chunk_by_words(text, size, overlap=0):
    """Cut the text into pieces of `size` words. With overlap, each piece
    starts `size - overlap` words after the previous one, so a sentence cut
    by one boundary is whole in the neighbouring piece."""
    words = text.split()
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def chunk_by_paragraph(text):
    """Cut where the author did: one chunk per section (heading + paragraph)."""
    parts = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, heading = [], ""
    for p in parts:
        if p.isupper():                 # a heading line: glue it to what follows
            heading = p
        else:
            chunks.append((heading + " " + p).strip())
            heading = ""
    return chunks


# ---------------------------------------------------------------------------
# STEP 2: INDEX  (TF-IDF, by hand)
#
# tokens     lower-case words; nothing cleverer. "Books" and "book" are
#            different tokens, "bicycle" and "cycle" are unrelated.
# TF         how many times the word occurs in this chunk
# IDF        log(number of chunks / number of chunks containing the word) + 1
#            "the" is in nearly every chunk -> small. "lift" is in one -> large.
# vector     TF x IDF for every word in the vocabulary, then scaled to
#            length 1 so that long chunks do not win just by being long.
# ---------------------------------------------------------------------------

def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class Index:
    def __init__(self, chunks, use_idf=True):
        self.chunks = chunks
        vocab = sorted({t for c in chunks for t in tokens(c)})
        self.col = {word: j for j, word in enumerate(vocab)}
        tf = np.zeros((len(chunks), len(vocab)))
        for i, c in enumerate(chunks):
            for t in tokens(c):
                tf[i, self.col[t]] += 1
        df = (tf > 0).sum(axis=0)                           # chunks containing each word
        self.idf = np.log(len(chunks) / df) + 1 if use_idf else np.ones(len(vocab))
        self.matrix = self._unit(tf * self.idf)             # (n_chunks, n_vocab)

    @staticmethod
    def _unit(m):
        norm = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(norm == 0, 1, norm)

    def vector(self, text):
        v = np.zeros(len(self.col))
        for t in tokens(text):
            if t in self.col:                               # words the handbook never uses are dropped
                v[self.col[t]] += 1
        return self._unit(v * self.idf)

    def search(self, question, k):
        """STEP 3: RETRIEVE. Cosine similarity = dot product of unit vectors.
        Returns [(score, chunk_number), ...], best first."""
        scores = self.matrix @ self.vector(question)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(float(scores[i]), int(i)) for i in order]


def expand(question):
    extra = [SYNONYMS[t] for t in tokens(question) if t in SYNONYMS]
    return question + " " + " ".join(extra) if extra else question


def prepare(question):
    return expand(question) if args.expand else question


def contains(chunk, evidence):
    return evidence.lower() in chunk.lower()


def hit_rate(index, questions, k):
    return float(np.mean([any(contains(index.chunks[i], ev) for _, i in index.search(prepare(q), k))
                          for q, ev in questions]))


def pct(x):
    return f"{100 * x:.0f}%"


def wrap(text, width):
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    return lines + [line]


# ---------------------------------------------------------------------------
if not DATA_PATH.exists():
    print(f"Missing {DATA_PATH}\nIt ships with the course; restore it from the repository.")
    sys.exit(1)
TEXT = DATA_PATH.read_text(encoding="utf-8")
FLAT = " ".join(TEXT.split())
for _, evidence in SAME_WORDS + OTHER_WORDS:
    assert FLAT.lower().count(evidence.lower()) == 1, f"evidence must appear exactly once: {evidence!r}"
ALL = SAME_WORDS + OTHER_WORDS
N_WORDS = len(TEXT.split())

rule("STEP 1 and 2: Chunk the handbook and index it")
chunks = chunk_by_paragraph(TEXT) if args.by_section else chunk_by_words(TEXT, CHUNK_WORDS, OVERLAP)
index = Index(chunks, USE_IDF)
SIZES = [len(c.split()) for c in chunks]
CHUNK_DESC = "one-section" if args.by_section else f"{CHUNK_WORDS}-word"
print(f"\n  handbook: {len(TEXT):,} characters, {N_WORDS:,} words")
if args.by_section:
    print(f"  chunks:   {len(chunks)} pieces, one per section, {min(SIZES)} to {max(SIZES)} words each")
else:
    print(f"  chunks:   {len(chunks)} pieces of {CHUNK_WORDS} words"
          + (f", each sharing {OVERLAP} words with the next" if OVERLAP else ""))
print(f"  index:    a table of {index.matrix.shape[0]} rows x {index.matrix.shape[1]} columns "
      f"(one column per distinct word)")
row = index.matrix[0]
top = np.argsort(-row)[:6]
words_by_col = {j: w for w, j in index.col.items()}
print(f"\n  Chunk 0 is: \"{chunks[0][:66]}...\"")
print(f"  Its vector is zero in {int((row == 0).sum())} of {len(row)} columns. The largest entries:")
print("    " + "   ".join(f"{words_by_col[j]} {row[j]:.2f}" for j in top))
common = "the"
rare = "lift"
if USE_IDF and common in index.col and rare in index.col:
    print(f"  IDF weights: \"{common}\" {index.idf[index.col[common]]:.2f} (it is everywhere), "
          f"\"{rare}\" {index.idf[index.col[rare]]:.2f} (it is in one chunk).")


def show_retrieval(question, evidence=None, k=3):
    for rank, (score, i) in enumerate(index.search(question, k), start=1):
        if score == 0:
            print(f"  #{rank}  nothing else shares a single word with the question")
            break
        shared = sorted(set(tokens(question)) & set(tokens(chunks[i])), key=lambda w: (-index.idf[index.col[w]], w))
        mark = "  <- contains the evidence" if evidence and contains(chunks[i], evidence) else ""
        print(f"  #{rank}  similarity {score:.3f}  chunk {i}{mark}")
        print(f"      \"{chunks[i][:68]}...\"")
        print(f"      shared words, rarest first: {', '.join(shared[:8]) if shared else 'none'}")


if args.ask:
    rule("YOUR QUESTION")
    question = prepare(args.ask)
    print(f"\n  Question: {args.ask}")
    if question != args.ask:
        print(f"  After synonym expansion: {question}")
    print()
    show_retrieval(question, k=K)
    if index.search(question, 1)[0][0] == 0:
        print(f"\n  Every similarity is zero: the handbook contains none of your words, so the")
        print(f"  ranking above is arbitrary. Try the handbook's own vocabulary.")
    print(f"\n  (No evidence phrase is known for your question, so nothing is scored. Open")
    print(f"  labs/data/library_handbook.txt and judge the result yourself.)")
    sys.exit(0)

rule("STEP 3: Retrieve - one question, step by step")
q, ev = SAME_WORDS[1]
print(f"\n  Question: {q}")
print(f"  Evidence we are hoping to fetch: \"{ev}\"\n")
show_retrieval(prepare(q), ev)


# ---------------------------------------------------------------------------
# MEASURE IT
# ---------------------------------------------------------------------------
rule(f"MEASURE: retrieval hit-rate at k={K} on 32 questions with known answers")

same_hits = hit_rate(index, SAME_WORDS, K)
other_hits = hit_rate(index, OTHER_WORDS, K)
print(f"\n  A hit = one of the top {K} chunks contains the evidence phrase.\n")
print(f"    questions in the handbook's own words     {same_hits * len(SAME_WORDS):>2.0f} of {len(SAME_WORDS)}   {pct(same_hits):>4}")
print(f"    questions in a visitor's own words        {other_hits * len(OTHER_WORDS):>2.0f} of {len(OTHER_WORDS)}   {pct(other_hits):>4}")
print(f"    all                                       {hit_rate(index, ALL, K) * len(ALL):>2.0f} of {len(ALL)}   {pct(hit_rate(index, ALL, K)):>4}")
if other_hits == 0:
    print(f"\n  0 of 12 is not an accident and not typical: those 12 were written on purpose")
    print(f"  to avoid the handbook's vocabulary. Real visitors land somewhere between the")
    print(f"  two groups. Failure mode 3 below takes one of them apart.")
print(f"\n  Misses among the first group:")
missed = [(q, ev) for q, ev in SAME_WORDS
          if not any(contains(chunks[i], ev) for _, i in index.search(prepare(q), K))]
for q, ev in missed:
    whole = any(contains(c, ev) for c in chunks)
    why = "retrieved the wrong chunks" if whole else "the evidence is CUT IN TWO by a chunk boundary"
    print(f"    {q:<52} {why}")
if not missed:
    print(f"    none")


# ---------------------------------------------------------------------------
# FAILURE MODE 1: chunk size
# ---------------------------------------------------------------------------
rule("FAILURE MODE 1: Chunks too small, chunks too big")


def hit_within_budget(index, questions, budget):
    """Hand over chunks in rank order until `budget` words are used up.
    (A chunk that does not fit is cut off at the budget.)"""
    hits = 0
    for q, ev in questions:
        left, got = budget, []
        for _, i in index.search(prepare(q), len(index.chunks)):
            words = index.chunks[i].split()
            got.append(" ".join(words[:left]))
            left -= len(words)
            if left <= 0:
                break
        hits += any(contains(g, ev) for g in got)
    return hits / len(questions)


print(f"\n  Same handbook, same 32 questions, different chunk sizes.")
print(f"  'top {K}' hands over the {K} best chunks however long they are.")
print(f"  'budget' hands over the best chunks until {BUDGET} words are used: a fair fight,")
print(f"  because every word you put in an LLM's prompt costs money and attention.\n")
print(f"  {'chunk size':>12} {'chunks':>7} {'hit, top ' + str(K):>11} {'words handed':>13} {'hit, ' + str(BUDGET) + '-word budget':>22} {'evidence cut':>13}")
print("  " + "-" * 83)
sweep = {}
for size in [5, 10, 20, 40, 80, 160, 320, N_WORDS]:
    cs = chunk_by_words(TEXT, size, min(OVERLAP, size - 1))
    ix = Index(cs, USE_IDF)
    cut = sum(not any(contains(c, ev) for c in cs) for _, ev in ALL)
    sweep[size] = (hit_rate(ix, ALL, K), hit_within_budget(ix, ALL, BUDGET))
    label = f"{size} words" if size != N_WORDS else "whole text"
    print(f"  {label:>12} {len(cs):>7} {pct(sweep[size][0]):>11} {min(K, len(cs)) * size:>13,} "
          f"{pct(sweep[size][1]):>22} {cut:>10} of {len(ALL)}")
cs = chunk_by_paragraph(TEXT)
ix = Index(cs, USE_IDF)
avg = int(np.mean([len(c.split()) for c in cs]))
para = (hit_rate(ix, ALL, K), hit_within_budget(ix, ALL, BUDGET))
print(f"  {'by section':>12} {len(cs):>7} {pct(para[0]):>11} {'about ' + str(K * avg):>13} {pct(para[1]):>22} {0:>10} of {len(ALL)}")
best_budget = max(sweep, key=lambda s: sweep[s][1])
print(f"\n  Too small ({pct(sweep[5][1])} at 5 words): a chunk holds the answer or the words of the")
print(f"  question, rarely both, and many answers are cut in half at a boundary.")
print(f"  Too big ({pct(sweep[320][1])} at 320 words, within the budget): the chunk is about everything,")
print(f"  so its vector matches nothing in particular. 'Whole text' scores 100% on")
print(f"  top-{K} by handing over all {N_WORDS:,} words, which is not retrieval at all.")
print(f"  Best fixed size within the budget: {best_budget} words ({pct(sweep[best_budget][1])}). "
      f"Cutting by section: {pct(para[1])}.")


# ---------------------------------------------------------------------------
# FAILURE MODE 2: k
# ---------------------------------------------------------------------------
rule("FAILURE MODE 2: k too small, k too large")

print(f"\n  {CHUNK_DESC} chunks. Retrieve the top k and see what the LLM would be handed.")
print(f"  Hit-rate is shown for the 20 own-words questions and for all 32.\n")
print(f"  {'k':>4} {'hit, own words':>15} {'hit, all 32':>12} {'words handed':>13} {'chunks that are relevant':>26}")
print("  " + "-" * 74)
k_sweep = {}
for k in [1, 2, 3, 5, 10, 20]:
    hits, relevant, handed = [], [], []
    for q, ev in ALL:
        found = index.search(prepare(q), k)
        got = [contains(chunks[i], ev) for _, i in found]
        hits.append(any(got))
        relevant.append(np.mean(got))
        handed.append(sum(SIZES[i] for _, i in found))
    k_sweep[k] = (float(np.mean(hits[:len(SAME_WORDS)])), float(np.mean(relevant)), float(np.mean(handed)),
                  float(np.mean(hits)))
    print(f"  {k:>4} {pct(k_sweep[k][0]):>15} {pct(k_sweep[k][3]):>12} {k_sweep[k][2]:>13,.0f} {pct(k_sweep[k][1]):>26}")
print(f"\n  k=1 misses {pct(1 - k_sweep[1][0])} of even the own-words answers: the right chunk is often")
print(f"  second or third. k=20 finds {pct(k_sweep[20][0])} of them but hands over {k_sweep[20][2]:,.0f} words, of which")
print(f"  {pct(1 - k_sweep[20][1])} are about something else. This lab cannot measure what that noise does")
print(f"  to an LLM's answer (there is no LLM here). In real systems it costs money on")
print(f"  every question, slows the reply, and gives the model more irrelevant text to be")
print(f"  misled by.")


# ---------------------------------------------------------------------------
# FAILURE MODE 3: the synonym miss
# ---------------------------------------------------------------------------
rule("FAILURE MODE 3: The synonym miss - right meaning, wrong words")

q, ev = OTHER_WORDS[2]
target = next((i for i, c in enumerate(chunks) if contains(c, ev)), None)
print(f"\n  Question: {q}")
if target is None:
    print(f"  With chunks this small the evidence (\"{ev}\") is cut in two, so there is no")
    print(f"  single chunk to find. The handbook's sentence is:")
    print(f"    \"Cycle racks for 20 bikes are beside the east entrance.\"")
else:
    ranking = [i for _, i in index.search(q, len(chunks))]
    scores = dict((i, s) for s, i in index.search(q, len(chunks)))
    shared = sorted(set(tokens(q)) & set(tokens(chunks[target])))
    print(f"  The passage that answers it (chunk {target}):")
    print(f"    \"...{chunks[target][chunks[target].lower().find(ev.lower()):][:72]}...\"")
    print(f"  Words they share: {', '.join(shared) if shared else 'none'}")
    print(f"  Similarity {scores[target]:.3f}. It is ranked {ranking.index(target) + 1} of {len(chunks)}"
          + (" (level with every other chunk that shares no words)." if scores[target] == 0 else "."))
print(f"\n  The handbook says \"cycle\" and \"bikes\". The visitor said \"bicycle\". To a person")
print(f"  those are the same thing. To a word-matching index they have nothing in")
print(f"  common, and no value of k short of 'everything' reliably fixes that.")

with_syn = float(np.mean([any(contains(chunks[i], ev) for _, i in index.search(expand(q), K))
                          for q, ev in OTHER_WORDS]))
without = float(np.mean([any(contains(chunks[i], ev) for _, i in index.search(q, K))
                         for q, ev in OTHER_WORDS]))
print(f"\n  The 12 visitor's-own-words questions, top {K}:")
print(f"    plain TF-IDF                                  {pct(without):>4}")
print(f"    with a hand-written synonym list (--expand)   {pct(with_syn):>4}")
print(f"\n  The synonym list works because it was written by someone who had already read")
print(f"  both the questions and the handbook. That does not scale past a demo. What")
print(f"  real systems do instead is use EMBEDDINGS: vectors produced by a trained")
print(f"  model, in which \"bicycle\" and \"cycle\" land close together because the model")
print(f"  saw them used alike. That needs a model, so it is not in this offline lab;")
print(f"  everything else in this file stays exactly the same when you swap it in.")


# ---------------------------------------------------------------------------
# STEP 4: "GENERATE"
#
# ===========================================================================
#   THIS IS WHERE THE LLM GOES. A real system does:
#
#       prompt = ("Answer the question using only the context below.\n\n"
#                 + "\n\n".join(retrieved_chunks) + "\n\nQuestion: " + question)
#       return call_your_llm(prompt)
#
#   We have no LLM, so generate() below does the crudest possible thing: it
#   returns the one SENTENCE from the retrieved chunks most similar to the
#   question. It cannot rephrase, combine two sentences, or say "I don't know".
# ===========================================================================

def generate(question, retrieved_chunks, index):
    sentences = [s.strip() for c in retrieved_chunks for s in re.split(r"(?<=[.?!;])\s+", c) if s.strip()]
    qv = index.vector(question)
    return max(sentences, key=lambda s: float(index.vector(s) @ qv))


rule("STEP 4: \"Generate\" - an honest stand-in for the LLM")
print(f"\n  There is no language model in this lab. In place of one, we print the single")
print(f"  retrieved sentence that best matches the question. A real system hands the")
print(f"  retrieved chunks to an LLM here and lets it write the answer.\n")
answered = []
for q, ev in ALL:
    got = [chunks[i] for _, i in index.search(prepare(q), K)]
    sentence = generate(prepare(q), got, index)
    answered.append((q, ev, sentence, any(contains(c, ev) for c in got), contains(sentence, ev)))
for q, ev, sentence, retrieved, right in [answered[0], answered[8], answered[22]]:
    print(f"  Q: {q}")
    for n, line in enumerate(wrap(sentence, 70)):
        print(f"  {'A:' if n == 0 else '  '} {line}")
    print(f"     [{'the right passage was retrieved' if retrieved else 'the right passage was NOT retrieved'}; "
          f"the sentence {'contains' if right else 'does not contain'} the answer]\n")
n_retrieved = sum(a[3] for a in answered)
n_right = sum(a[4] for a in answered)
n_right_given = sum(a[4] for a in answered if a[3])
print(f"  Over all 32 questions:")
print(f"    right passage retrieved                        {n_retrieved} of 32")
print(f"    extracted sentence contains the answer         {n_right} of 32")
print(f"    ... when the right passage was retrieved       {n_right_given} of {n_retrieved}")
print(f"    ... when it was not                            {n_right - n_right_given} of {32 - n_retrieved}")
print(f"\n  The last line is zero by construction (you cannot copy an answer out of a")
print(f"  passage that was never fetched) and it is the point of the lab. When retrieval")
print(f"  misses, the answer step has nothing to work with: it still returns a fluent,")
print(f"  confident, irrelevant sentence. An LLM in the same position does the same or,")
print(f"  worse, fills the gap from memory. Fix retrieval first, and measure it")
print(f"  separately from generation.")
if n_right_given < n_retrieved:
    print(f"  The line above it is the other half: in {n_retrieved - n_right_given} of {n_retrieved} cases the passage WAS there and")
    print(f"  our one-sentence picker still chose the wrong sentence. Reading a passage and")
    print(f"  writing the answer is the part an LLM is genuinely good at.")

rule("WHAT THIS LAB SHOWS")
print("""
1. RAG is search plus a prompt. The search half can be built, measured and
   debugged without any model at all, and it is where most systems fail.

2. Build the eval set first: questions with known evidence. Without it you
   are tuning chunk size and k by reading a few answers and hoping.

3. Chunk size is a real trade-off. Too small splits answers from their
   context; too big blurs every chunk into "about everything". Cutting where
   the author cut (sections, paragraphs) is usually the best first try.

4. k trades misses for noise. Measure hit-rate and the amount of irrelevant
   text together, never one alone.

5. Word matching cannot see meaning. That is the job of embeddings, and
   embeddings have their own blind spots (exact codes, names and numbers),
   which is why production systems usually combine the two.

6. If the passage was not retrieved, the model cannot use it.

Now try the experiments at the top of this file. Each is one flag, and each
says what you should see. Start with:   python labs/10_rag.py --chunk-words 8
""")
