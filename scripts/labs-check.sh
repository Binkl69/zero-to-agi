#!/usr/bin/env bash
# Runs every Python lab in its quick mode and fails if one crashes, writes to
# stderr, or stops printing the lines its own docstring promises.
#
# The labs are the part of the course a learner runs with their own hands, and
# for a long time nothing ran them at all: lab 06 shipped with a note saying it
# had never been executed, and lab 08's headline result held for one seed in
# two. A lab that exits 0 can still be wrong, so each one is also checked for
# the sentence that carries its lesson.
#
#   bash scripts/labs-check.sh              all twelve labs (needs numpy + torch)
#   LABS_SKIP_TORCH=1 bash scripts/labs-check.sh    skip labs 06 and 09, loudly
#   PYTHON=python3.12 bash scripts/labs-check.sh    pick the interpreter
#
# Every lab is started from an empty temp directory, not the repo root, so a
# lab that quietly depends on the current directory fails here.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABS="$ROOT/labs"
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
# a hung lab should fail, not stall CI; macOS has no `timeout`, so it is optional
LIMIT=""
command -v timeout >/dev/null 2>&1 && LIMIT="timeout 600"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

failures=0
total_start=$(date +%s)

# run NAME PYTHON_FLAGS "lab.py args..." "must contain" ["must contain" ...]
# A pattern starting with "!" must NOT appear; one starting with "~" is a regex.
# Patterns avoid the last digits of trained numbers on purpose: those may differ
# between platforms and numpy versions, the lesson may not.
run() {
  local name="$1" strict="$2" cmd="$3"; shift 3
  local out="$WORK/out.txt" err="$WORK/err.txt" start end code ok=1
  start=$(date +%s.%N)
  # shellcheck disable=SC2086
  (cd "$WORK" && $LIMIT "$PY" $strict "$LABS"/$cmd >"$out" 2>"$err")
  code=$?
  end=$(date +%s.%N)
  local secs
  secs=$(awk -v a="$start" -v b="$end" 'BEGIN { printf "%.1f", b - a }')

  local why=()
  [ "$code" -ne 0 ] && { ok=0; why+=("exit code $code"); }
  if [ -s "$err" ]; then ok=0; why+=("wrote to stderr: $(head -c 300 "$err")"); fi
  for want in "$@"; do
    if [ "${want:0:1}" = "!" ]; then
      grep -qF -- "${want:1}" "$out" && { ok=0; why+=("printed forbidden text: ${want:1}"); }
    elif [ "${want:0:1}" = "~" ]; then
      grep -qE -- "${want:1}" "$out" || { ok=0; why+=("no line matches: ${want:1}"); }
    else
      grep -qF -- "$want" "$out" || { ok=0; why+=("missing line: $want"); }
    fi
  done

  if [ "$ok" -eq 1 ]; then
    printf '  ok    %6ss  %s\n' "$secs" "$name"
  else
    printf '  FAIL  %6ss  %s\n' "$secs" "$name"
    for w in "${why[@]}"; do printf '          %s\n' "$w"; done
    printf '          last lines of output:\n'
    tail -n 5 "$out" | sed 's/^/            /'
    failures=$((failures + 1))
  fi
}

echo "labs-check: $("$PY" --version 2>&1), labs in $LABS"

# -W error turns any Python warning into a crash for the labs we fully control.
W="-W error"

run "01 perceptron" "$W" "01_perceptron.py" \
  "Training a perceptron on AND" "-> converged after" "gets 1 of 4 points right" \
  "XOR solved." "!WRONG" "!Something is off"

run "02 autograd" "$W" "02_autograd.py" \
  "dL/da = dL/de * de/da = -2 * b = -2 * -3 = +6" "dy/dx should be 2x = 6.  We got: 6" \
  "Engine OK." "!MISMATCH"

run "03 mlp --quick" "$W" "03_mlp.py --quick" \
  "~step  300  loss = 0\.00[0-9]+   train acc = 100%" "~final train accuracy: (9[0-9]|100)%" "!WRONG" "!DIVERGED"

run "03 mlp --seed 14 (the stuck seed of experiment D)" "$W" "03_mlp.py --seed 14 --steps 1" \
  "~step  300  loss = 0\.5[0-9]+   train acc = 50%" "WRONG"

run "04 tokenizer" "$W" "04_tokenizer_bpe.py" \
  "learned 256 merges; vocab size = 512" "-> id 262  b' the'" \
  "st|r|aw|b|er|r|y" "yes - BPE is lossless"

run "04 tokenizer --vocab-size 800 (runs out of pairs)" "$W" "04_tokenizer_bpe.py --vocab-size 800" \
  "stopped early" "unbeliev|ably" "yes - BPE is lossless"

run "05 attention" "$W" "05_attention.py" \
  "checks: rows sum to 1, future entries are exactly 0.  ok" \
  "causality check: editing the last token changed only the last logits.  ok" \
  "matches the published 124,439,808 exactly"

run "07 q-learning" "$W" "07_q_learning.py" \
  "Greedy rollout from S (10 steps): reached G"

run "07 q-learning --coin (reward hacking)" "$W" "07_q_learning.py --coin" \
  "reached goal 0%" "STILL WANDERING" "Reward hacking in action"

run "07 q-learning --coin --coin-reward 0.1 (coin not worth it)" "$W" "07_q_learning.py --coin --coin-reward 0.1" \
  "Greedy rollout from S (10 steps): reached G"

run "08 toy rlhf" "$W" "08_toy_rlhf.py" \
  "~^R7 .*<- reward model's favourite" \
  "WORSE than where we started" \
  "The same curve as Step 4" "WHAT THIS LAB SHOWS"

# Labs 10-12 print measured rates. The patterns pin the conclusion (who wins,
# what is detected) and leave room in the digits, which are seeded but may
# move in the last place between numpy versions.
run "10 rag" "$W" "10_rag.py" \
  "~questions in the handbook's own words +1[5-7] of 20" \
  "~questions in a visitor's own words +0 of 12" \
  "Words they share: none" "~Cutting by section: (6[5-9]|7[0-9])%" \
  "There is no language model in this lab" "~when it was not +0 of 16"

run "10 rag --by-section --expand (retrieval solved, answer step is not)" "$W" "10_rag.py --by-section --expand" \
  "~all +32 of 32" "~extracted sentence contains the answer +2[0-9] of 32"

run "10 rag --chunk-words 8 (answers cut in two)" "$W" "10_rag.py --chunk-words 8" \
  "the evidence is CUT IN TWO by a chunk boundary" "~questions in the handbook's own words +[5-9] of 20"

# 300 simulated runs per number instead of 2,000: enough to tell 8% from 60%
run "11 agent tools (--trials 300)" "$W" "11_agent_tools.py --trials 300" \
  "DONE in 5 model calls and 4 tool calls." "ANSWER The total is 124.20" \
  "8 of 9 replies refused with a reason" "Retrying (1 of 3)" \
  "~^ +50 +7\.7% +1?[0-9]\.[0-9]% +(9[5-9]|100)\.[0-9]%" \
  "INJECTION SUCCEEDED. The user asked for a quote" "INJECTION IGNORED." "!Traceback"

run "11 agent tools --guard gate (fooled, but blocked)" "$W" "11_agent_tools.py --guard gate --trials 20" \
  "BLOCKED: this task is not permitted to use email_note" "INJECTION BLOCKED BY THE GATE."

run "11 agent tools --budget 4 (runs out of steps)" "$W" "11_agent_tools.py --budget 4 --trials 20" \
  "STOPPED: step budget exhausted after 4 model calls."

run "12 eval harness" "$W" "12_eval_harness.py" \
  "~^ +20 +5[0-9]\.[0-9]% +1[0-9]\.[0-9]% +(2[0-9]|3[0-5])\.[0-9]%" \
  "~^ +1000 +9[7-9]\.[0-9]% " "~The paired interval is [12][0-9]% narrower" \
  "REGRESSION SUITE: FAIL - units got worse" "POSITION BIAS DETECTED" "LENGTH BIAS DETECTED" \
  "C is better  <- wrong" "WHAT THIS LAB SHOWS"

run "12 eval harness --position-bias 0 (swap test quiet, length bias remains)" "$W" "12_eval_harness.py --position-bias 0" \
  "no position bias detected" "LENGTH BIAS DETECTED"

if "$PY" -c "import torch" >/dev/null 2>&1; then
  # 150 steps is enough to get from 4.17 (uniform) to under 2.7; the full
  # --quick preset is 300 steps and gets to about 2.3.
  CK="$WORK/quick.pt"
  run "06 tiny gpt --quick (150 steps)" "" "06_tiny_gpt.py --quick --steps 150 --device cpu --ckpt $CK" \
    "vocab of 65 symbols, ln(vocab) = 4.17" "step     0 | train 4." "step   150 | train" "saved checkpoint"
  # The lesson of the lab is that the loss falls; a run that stays at 4.17 exits 0 too.
  val=$(grep -F "step   150 |" "$WORK/out.txt" | sed -E 's/.*val ([0-9.]+).*/\1/')
  if awk -v v="${val:-9}" 'BEGIN { exit !(v < 2.9) }'; then
    printf '  ok             06 val loss after 150 steps is %s (< 2.9)\n' "$val"
  else
    printf '  FAIL           06 val loss after 150 steps is %s, expected under 2.9\n' "${val:-missing}"
    failures=$((failures + 1))
  fi
  run "06 tiny gpt --sample (reload the checkpoint)" "" \
    "06_tiny_gpt.py --sample --device cpu --ckpt $CK --max-new-tokens 60 --prompt ROMEO:" \
    "loaded" "ROMEO:"

  # Lab 09 imports lab 06's model. 120 pretraining steps and 60 fine-tuning
  # steps per method keep this to a few seconds; --quick alone is 300 and 150.
  run "09 lora fine-tune --quick (120 + 60 steps)" "" \
    "09_lora_finetune.py --quick --pretrain-steps 120 --steps 60 --device cpu" \
    "(B is zero, so the adapter is silent)" "~^  LoRA +3,072 +2\.8%" \
    "Exactly the base model again." "MERGE CHECK: OK" "WHAT THIS LAB SHOWS"
  # The lesson is in three inequalities, so check them as numbers: both methods
  # learn the weather, and both get worse at Shakespeare.
  line=$(grep -F "Held-out weather:" "$WORK/out.txt")
  forgot=$(grep -F "FORGETTING:" "$WORK/out.txt")
  if echo "$line" | sed -E 's/[^0-9.]+/ /g' | awk '{ exit !($2 < $1 && $3 < $1) }' &&
     echo "$forgot" | grep -qE 'LoRA \+0\.[0-9]+, full fine-tuning \+[0-9]+\.[0-9]+'; then
    printf '  ok             09 both methods learned the weather and both forgot Shakespeare\n'
  else
    printf '  FAIL           09 expected weather loss to fall and Shakespeare loss to rise for both methods\n'
    printf '          %s\n          %s\n' "${line:-missing}" "${forgot:-missing}"
    failures=$((failures + 1))
  fi
elif [ "${LABS_SKIP_TORCH:-0}" = "1" ]; then
  echo "  SKIP           06 tiny gpt: torch is not installed and LABS_SKIP_TORCH=1. LAB 06 WAS NOT RUN."
  echo "  SKIP           09 lora fine-tune: torch is not installed and LABS_SKIP_TORCH=1. LAB 09 WAS NOT RUN."
else
  echo "  FAIL           06 tiny gpt: torch is not installed (pip install torch, or set LABS_SKIP_TORCH=1 to skip)"
  echo "  FAIL           09 lora fine-tune: torch is not installed (pip install torch, or set LABS_SKIP_TORCH=1 to skip)"
  failures=$((failures + 2))
fi

echo "labs-check: $(( $(date +%s) - total_start ))s total"
if [ "$failures" -ne 0 ]; then
  echo "labs-check: $failures FAILED"
  exit 1
fi
echo "labs-check: all labs passed"
