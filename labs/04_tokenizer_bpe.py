"""
LAB 04 - BYTE-PAIR ENCODING: HOW TEXT BECOMES NUMBERS
=====================================================

Run:   python labs/04_tokenizer_bpe.py
       python labs/04_tokenizer_bpe.py --vocab-size 300 --show "your own words here"
Needs: nothing but Python.
Time:  ~1 s.

WHAT YOU WILL BUILD
-------------------
A language model is a machine that predicts the next TOKEN, so before any
model can read, text must be chopped into tokens and each token given an
integer id. The chopping scheme used by GPT-2/3/4, Llama, Claude and friends
is Byte-Pair Encoding (BPE), and it is embarrassingly simple:

    1. Start with the 256 possible BYTES as the vocabulary (so ANY text in
       any language or emoji is representable - no "unknown" token ever).
    2. Look at the training text as a sequence of byte-ids. Find the pair of
       adjacent ids that occurs most often. Give that pair a brand-new id.
       Replace every occurrence of the pair with the new id.
    3. Repeat step 2 until the vocabulary is as big as you want.

That is the whole training algorithm. "th" is learned first because it is
everywhere, then "the", then " the" (note the leading space - GPT-style
tokenizers glue the space onto the following word), then "ing", "ed", ...
After 50,000 merges (GPT-2) common words are single tokens and rare words
are 2-4 pieces.

Encoding new text = replay the merges in the order they were learned.
Decoding = look up each id's bytes, concatenate, decode as UTF-8.

WHAT TO OBSERVE
---------------
1. The first 20 merges. They are ALL boring English glue: "th", "e ", "in",
   " the", "ing". The tokenizer has no idea what a word is; it just counts.
2. How "strawberry" and "unbelievably" get split. With our tiny vocabulary
   they shatter into several pieces; with GPT-4's 100k vocabulary
   "strawberry" is [" str", "aw", "berry"] - still not letters!
3. The compression ratio: about 3-4 bytes per token on English. This is why
   model context windows are quoted in tokens, and why a "128k context" is
   roughly a 300-page book.

THE "HOW MANY R's IN STRAWBERRY" FAILURE
---------------------------------------
Why did big models famously answer "2"? Because the model never sees the
letters. It sees the token ids for [" str", "aw", "berry"]. To count r's it
would have to have MEMORISED, for each of those tokens, how the token is
spelled - there is no spelling information inside an id. Tokenisation is
also why models are weak at reversing strings, rhyming, arithmetic on long
numbers (digits get grouped unpredictably: "1234567" -> "123","456","7"),
and why the same word tokenises differently with/without a leading space or
capital letter (" Hello" vs "Hello" vs "hello" are three unrelated ids).

WHAT TO CHANGE (3 experiments)
------------------------------
A. --vocab-size 260 vs 400 vs 800. Watch the compression ratio and the token
   pieces for "unbelievably". More merges = fewer tokens per word, but a
   bigger embedding table for the model (Lab 05 counts that cost).
B. Set PRETOKENIZE = False. Now merges may cross word boundaries and you get
   tokens like "of the" or "s and". GPT-2 added the regex pre-split
   precisely to stop this (and to keep punctuation separate from words).
C. Train on a different text (paste anything into TRAINING_TEXT): code,
   French, a chat log. The merges adapt instantly - a tokenizer is a
   compression codec fitted to its training data, nothing more.

COMMON BUGS
-----------
* Merging pairs in the wrong order at encode time. Merges must be applied
  in the order they were learned (lowest id first), otherwise you produce
  token sequences the model never saw in training.
* Splitting a multi-byte UTF-8 character across tokens, then decoding a
  single token: it is not valid UTF-8 on its own. That is why decode()
  uses errors="replace" and why you should decode whole sequences.
* Counting pairs across the boundary between two pre-tokenized chunks.
  Merges must stay INSIDE a chunk (that's the entire point of pre-splitting).
* Off-by-one on the vocabulary: 256 byte tokens + N merges = vocab of 256+N,
  so --vocab-size 300 means 44 merges, not 300.
"""

import sys
import re
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# 0. PRE-TOKENIZATION (the GPT-2 trick, simplified)
# ---------------------------------------------------------------------------
# Before merging, GPT-2 splits text into chunks with a regular expression so
# that merges never cross a word boundary and punctuation stays separate.
# The real pattern uses Unicode classes (\p{L}) and needs the `regex` package;
# this ASCII-flavoured version with the stdlib `re` keeps the spirit:
#   " ?[A-Za-z]+"        a word, with its optional leading space attached
#   " ?[0-9]+"           a run of digits
#   " ?[^\sA-Za-z0-9]+"  a run of punctuation/symbols
#   "\s+"                leftover whitespace (newlines etc.)
PRETOKENIZE = True
PRETOKEN_PATTERN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


def pretokenize(text):
    if not PRETOKENIZE:
        return [text]
    return PRETOKEN_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# 1. TRAINING
# ---------------------------------------------------------------------------
def count_pairs(chunks, counts=None):
    """How often does each adjacent pair (id_i, id_i+1) occur, across chunks?"""
    counts = {} if counts is None else counts
    for ids in chunks:
        for a, b in zip(ids, ids[1:]):       # zip against the shifted list
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge_pair(ids, pair, new_id):
    """Return a copy of `ids` with every occurrence of `pair` replaced by new_id."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2                            # skip both halves of the pair
        else:
            out.append(ids[i])
            i += 1
    return out


def train(text, vocab_size, verbose=False):
    """Learn BPE merges from `text`.

    Returns
      merges : dict  (id_a, id_b) -> new_id, in the ORDER they were learned
               (dict preserves insertion order; the new_ids also increase)
      vocab  : dict  id -> bytes, for every id including the 256 base bytes
    """
    assert vocab_size >= 256, "vocab must at least hold the 256 raw bytes"
    n_merges = vocab_size - 256

    # Every chunk becomes a list of byte values 0..255. That's our starting
    # "alphabet": no characters, no words, just bytes.
    chunks = [list(chunk.encode("utf-8")) for chunk in pretokenize(text)]

    merges = {}
    vocab = {i: bytes([i]) for i in range(256)}
    for i in range(n_merges):
        counts = count_pairs(chunks)
        if not counts:
            break                              # text fully merged (tiny inputs)
        pair = max(counts, key=counts.get)     # the most frequent adjacent pair
        if counts[pair] < 2:
            break                              # nothing repeats any more: stop
        new_id = 256 + i
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        chunks = [merge_pair(ids, pair, new_id) for ids in chunks]
        if verbose:
            print(f"  merge {i + 1:3d}: {pair} -> {new_id}  {vocab[new_id]!r}  (x{counts[pair]})")
    return merges, vocab


# ---------------------------------------------------------------------------
# 2. ENCODE / DECODE
# ---------------------------------------------------------------------------
def encode_chunk(ids, merges):
    """Apply the learned merges to one chunk, earliest-learned merge first."""
    while len(ids) >= 2:
        pairs = set(zip(ids, ids[1:]))
        # Of all pairs present, pick the one that was learned EARLIEST
        # (smallest new_id). If none of them is a known merge, we're done.
        pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge_pair(ids, pair, merges[pair])
    return ids


def encode(text, merges):
    ids = []
    for chunk in pretokenize(text):
        ids.extend(encode_chunk(list(chunk.encode("utf-8")), merges))
    return ids


def decode(ids, vocab):
    raw = b"".join(vocab[i] for i in ids)
    # A token may end mid-character (UTF-8 chars are 1-4 bytes). Decoding the
    # WHOLE sequence at once is fine; decoding a single token might not be.
    return raw.decode("utf-8", errors="replace")


def show_pieces(text, merges, vocab):
    """Print how a string is chopped, e.g.  str|aw|berry  (5 tokens)."""
    ids = encode(text, merges)
    pieces = [vocab[i].decode("utf-8", errors="replace") for i in ids]
    print(f"  {text!r:<45} -> {'|'.join(pieces)}   ({len(ids)} tokens, "
          f"{len(text.encode('utf-8'))} bytes)")
    return ids


# ---------------------------------------------------------------------------
# 3. TRAINING TEXT (~3 KB of ordinary English, written for this lab)
# ---------------------------------------------------------------------------
TRAINING_TEXT = """
The first machines that could read did not read at all. They counted. A
punched card held a number, the number stood for a letter, and the letter
meant nothing to the machine that sorted it. For decades this was the whole
story: text went in as codes, came out as codes, and the meaning stayed
with the people on either side. The machine was a very fast clerk with no
idea what the forms were for.

The change came slowly and then all at once. Researchers noticed that if you
count which words tend to follow which other words in a large enough pile of
text, the counts themselves start to look like knowledge. The word after
"strong" is often "coffee" and rarely "tea"; the word after "the capital of
France is" is almost always "Paris". Nobody typed those facts in. They were
sitting in the statistics of ordinary writing, waiting for a machine patient
enough to tally them. Counting, it turned out, was reading after all, or at
least the beginning of it.

But counting words has a problem: there are too many of them, and new ones
appear every day. A model that only knows the words it has seen is helpless
in front of a typo, a name, a piece of code, or a word in another language.
So the modern trick is to count pieces of words instead. Start from the raw
bytes, which can spell anything, and merge the most common pairs over and
over until the pieces are big enough to be useful but small enough to cover
everything. The pieces are called tokens, and everything a language model
knows, it knows in tokens. It has never seen a letter in its life.

This has strange consequences. Ask such a model to spell a word backwards
and it may stumble, because it never saw the letters in the first place;
it saw two or three tokens and has to remember how each of them is spelled.
Ask it how many times a letter appears in a word and it may confidently give
the wrong answer for the same reason. Ask it to add two long numbers and the
digits arrive in awkward clumps that make the arithmetic harder than it
needs to be. None of these are failures of intelligence exactly. They are
failures of eyesight, of the peculiar glasses the model was made to wear.

And yet the same glasses are what make the whole thing work. Tokens compress
text by a factor of three or four, which means a model can attend to three
or four times more meaning in the same window. They give every word, in every
language, a fixed and reliable set of pieces. They let the model learn that
"unbelievably" is built from "un", "believ", and "ably", and reuse what it
knows about each piece the first time it meets "unbelievable". The counting
machine grew up, and this is what it learned to see.

The lesson for the rest of this course is simple. Whenever a model does
something baffling with text, ask first what it actually saw. The answer is
never letters and never quite words. It is a list of integers, chosen by a
counting procedure that ran once, long ago, on a pile of text the model will
never read again.
"""


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=512, help="256 bytes + N merges")
    ap.add_argument("--show", default=None, help="an extra string to tokenize")
    args = ap.parse_args()

    text = TRAINING_TEXT.strip()
    n_bytes = len(text.encode("utf-8"))
    print("=" * 72)
    print(f"Training BPE on {n_bytes} bytes of text, target vocab = {args.vocab_size} "
          f"({args.vocab_size - 256} merges)")
    print("=" * 72)
    merges, vocab = train(text, args.vocab_size)
    print(f"  learned {len(merges)} merges; vocab size = {len(vocab)}")
    print()

    print("First 20 merges (the pair that was most frequent at that moment):")
    for i, (pair, new_id) in enumerate(list(merges.items())[:20]):
        left, right = vocab[pair[0]], vocab[pair[1]]
        print(f"  {i + 1:2d}. {left!r:>10} + {right!r:<10} -> id {new_id}  {vocab[new_id]!r}")
    print("  (note: ' the' with a leading space is a different token from 'the')")
    print()

    print("How words get chopped (pieces separated by |):")
    show_pieces("strawberry", merges, vocab)
    show_pieces("unbelievably", merges, vocab)
    show_pieces("The counting machine learned to read.", merges, vocab)
    show_pieces("Strawberry", merges, vocab)          # capital: different pieces!
    show_pieces(" strawberry", merges, vocab)         # leading space: different again
    show_pieces("1234567", merges, vocab)
    if args.show:
        show_pieces(args.show, merges, vocab)
    print()

    print("Compression on the training text itself:")
    ids = encode(text, merges)
    print(f"  {n_bytes} bytes -> {len(ids)} tokens  =  {n_bytes / len(ids):.2f} bytes per token")
    print(f"  (GPT-2/GPT-4 on English: roughly 4 bytes per token, i.e. ~0.75 words per token)")
    print()

    print("Round trip: decode(encode(text)) == text ?")
    assert decode(ids, vocab) == text, "round trip failed!"
    print("  yes - BPE is lossless. Tokens are compression, not understanding.")

    # A string the tokenizer has never seen still works, because the base
    # vocabulary is all 256 bytes:
    novel = "Zebras quiz jazz: 42% xylophone! éè — \U0001F600"
    assert decode(encode(novel, merges), vocab) == novel
    print(f"  also survives unseen characters, accents and emoji: {len(encode(novel, merges))} "
          f"tokens for a {len(novel.encode('utf-8'))}-byte string (rare stuff = more tokens)")
    print()
    print("Why 'how many r's in strawberry' goes wrong: the model gets the ids for")
    print("the pieces above, never the letters. Counting letters means recalling the")
    print("spelling of each piece from memory - the id itself carries no spelling.")
