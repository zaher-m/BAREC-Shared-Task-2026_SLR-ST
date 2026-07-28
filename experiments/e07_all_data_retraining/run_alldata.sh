#!/usr/bin/env bash
# The data lever: retrain the 10 V6 members on TRAIN+DEV (+13% data), holding out TEST for
# early stopping and calibration. Blind is never touched. Two effects wanted at once: more
# data on proven members, and a family trained on different data should have decorrelated
# errors. Base members first (fast), large ones last.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}
AR=aubmindlab/bert-base-arabertv2
LG=aubmindlab/bert-large-arabertv2
AE=aubmindlab/araelectra-base-discriminator
AM=NAMAA-Space/AraModernBert-Base-V1.0
ARB=UBC-NLP/ARBERTv2

run () { # tag model variant objective [bs] [lr] [epochs]
  local tag=$1 model=$2 var=$3 obj=$4 bs=${5:-32} lr=${6:-2e-5} ep=${7:-8}
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag (done)"; return; }
  while pgrep -f 'slra_st.train' >/dev/null; do sleep 15; done
  echo ">>> $(date +%H:%M:%S) $tag ($obj/$var bs=$bs) ALLDATA"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant "$var" --objective "$obj" \
      --alldata --seed 42 --max_len 160 --bs "$bs" --lr "$lr" --epochs "$ep" --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) done $tag (exit $?)"
}

# ---- base members (fast) ----
run arabertv2_corn_ad    $AR  d3tok corn
run arabertv2_emd_ad     $AR  d3tok emd
run arabertv2_soft_ad    $AR  d3tok soft
run arabertv2_wkl_ad     $AR  d3tok wkl
run araelectra_corn_ad   $AE  d3tok corn
run aramodern_reg_ad     $AM  raw   reg
run arbertv2_corn_ad     $ARB d3tok corn
echo "=== AD_BASE DONE $(date +%H:%M:%S) ==="
# ---- large members (slow) ----
run arabertv2_large_corn_ad $LG d3tok corn 24 1.5e-5 6
run arabertv2_large_reg_ad  $LG d3tok reg  24 1.5e-5 6
run arabertv2_large_soft_ad $LG d3tok soft 24 1.5e-5 6
echo "=== ALLDATA DONE $(date +%H:%M:%S) ==="
