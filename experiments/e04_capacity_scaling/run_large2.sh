#!/usr/bin/env bash
# Push the lever that worked: large lifted blind 84.6 -> 85.0 by fixing the big-error tail.
# Three follow-ups:
#   - the missing large objective (wkl)
#   - AraBERT-large on RAW text: a controlled raw-vs-d3tok pair at large scale, and a
#     differently preprocessed member should also decorrelate
#   - XLM-R-large (550M): more capacity, different pretraining and tokenizer
# Greedy selection drops whatever does not help, so extra candidates are cheap.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}
LG=aubmindlab/bert-large-arabertv2
XLMR=FacebookAI/xlm-roberta-large

run () { # tag model variant objective [bs] [lr] [epochs]
  local tag=$1 model=$2 var=$3 obj=$4 bs=${5:-24} lr=${6:-1.5e-5} ep=${7:-6}
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag (done)"; return; }
  while pgrep -f 'slra_st.train' >/dev/null; do sleep 15; done
  echo ">>> $(date +%H:%M:%S) $tag ($obj/$var bs=$bs)"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant "$var" --objective "$obj" \
      --seed 42 --max_len 160 --bs "$bs" --lr "$lr" --epochs "$ep" --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) done $tag (exit $?)"
}

run arabertv2_large_wkl   $LG   d3tok wkl  24 1.5e-5 6
echo "=== L2_A DONE $(date +%H:%M:%S) ==="
run arabertv2_large_reg_raw  $LG raw reg  24 1.5e-5 6
run arabertv2_large_corn_raw $LG raw corn 24 1.5e-5 6
echo "=== L2_B DONE $(date +%H:%M:%S) ==="
run xlmr_large_reg   $XLMR raw reg  16 1e-5 6
run xlmr_large_corn  $XLMR raw corn 16 1e-5 6
echo "=== L2_C DONE $(date +%H:%M:%S) ==="
echo "=== LARGE2 DONE $(date +%H:%M:%S) ==="
