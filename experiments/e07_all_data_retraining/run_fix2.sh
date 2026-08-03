#!/usr/bin/env bash
# Relaunch after a host reboot killed the previous run at epoch 1 (it was healthy: 83.554).
# Epochs 6 -> 4 to bank the result sooner and risk less on another reboot; epoch 1 was already
# 83.55 and these models usually peak at epoch 2-3.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}

run () { # tag model variant objective [bs] [lr] [epochs]
  local tag=$1 model=$2 var=$3 obj=$4 bs=${5:-32} lr=${6:-2e-5} ep=${7:-8}
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag"; return; }
  while pgrep -f -- "--tag ${tag} " >/dev/null; do sleep 15; done
  echo ">>> $(date +%H:%M:%S) $tag ($obj/$var lr=$lr ep=$ep)"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant "$var" --objective "$obj" \
      --alldata --seed 42 --max_len 160 --bs "$bs" --lr "$lr" --epochs "$ep" --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) done $tag (exit $?)"
}
run arabertv2_large_reg_ad2 aubmindlab/bert-large-arabertv2 d3tok reg 24 8e-6 4
echo "=== FIX2 DONE $(date +%H:%M:%S) ==="
run marbert_corn_ad    UBC-NLP/MARBERTv2                        d3tok corn 32 2e-5 6
run camelbert_corn_ad  CAMeL-Lab/bert-base-arabic-camelbert-mix  d3tok corn 32 2e-5 6
echo "=== POOL DONE $(date +%H:%M:%S) ==="
