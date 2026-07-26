#!/usr/bin/env bash
# The pivot after seeds came out neutral: spend the GPU on decorrelated members, not replicas.
# Replicas cannot help because the level-12 tail is systematic.
#   Phase A: backbone x objective combos not trained yet (fast, base size).
#   Phase B: AraBERT-large, which may read the hard cases differently instead of the same way.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}

MB=UBC-NLP/MARBERTv2
CB=CAMeL-Lab/bert-base-arabic-camelbert-mix
AE=aubmindlab/araelectra-base-discriminator
AM=NAMAA-Space/AraModernBert-Base-V1.0
ARB=UBC-NLP/ARBERTv2
LG=aubmindlab/bert-large-arabertv2

run () { # tag model variant objective seed [bs] [lr] [epochs]
  local tag=$1 model=$2 var=$3 obj=$4 seed=$5 bs=${6:-32} lr=${7:-2e-5} ep=${8:-8}
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag (done)"; return; }
  while pgrep -f 'slra_st.train' >/dev/null; do sleep 15; done
  echo ">>> $(date +%H:%M:%S) $tag ($obj/$var bs=$bs)"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant "$var" --objective "$obj" \
      --seed "$seed" --max_len 160 --bs "$bs" --lr "$lr" --epochs "$ep" --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) done $tag (exit $?)"
}

# ===== PHASE A: cross-backbone new objective combinations =====
run marbert_emd_d3      $MB  d3tok emd  42
run camelbert_emd_d3    $CB  d3tok emd  42
run camelbert_wkl_d3    $CB  d3tok wkl  42
run araelectra_emd_d3   $AE  d3tok emd  42
run araelectra_wkl_d3   $AE  d3tok wkl  42
run arbertv2_corn_d3    $ARB d3tok corn 42
run arbertv2_soft_d3    $ARB d3tok soft 42
run aramodern_soft_raw  $AM  raw   soft 42
echo "=== PHASE_A DONE $(date +%H:%M:%S) ==="

# ===== PHASE B: AraBERT-large =====
run arabertv2_large_reg  $LG d3tok reg  42 24 1.5e-5 6
run arabertv2_large_corn $LG d3tok corn 42 24 1.5e-5 6
run arabertv2_large_soft $LG d3tok soft 42 24 1.5e-5 6
run arabertv2_large_emd  $LG d3tok emd  42 24 1.5e-5 6
echo "=== CAMPAIGN2 DONE $(date +%H:%M:%S) ==="
