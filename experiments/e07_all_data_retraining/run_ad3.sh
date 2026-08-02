#!/usr/bin/env bash
# Same pool expansion, but sharing the GPU: the guard waits only on this script's own tag
# instead of on any trainer, so it runs alongside a sibling project at roughly 50/50.
# Why keep adding AD members: the AD curve is still steep (8 -> 10 members gave +0.23 on the
# blend), so more AD material is the most likely way past 85.3.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}
AR=aubmindlab/bert-base-arabertv2
LG=aubmindlab/bert-large-arabertv2
MB=UBC-NLP/MARBERTv2
CB=CAMeL-Lab/bert-base-arabic-camelbert-mix
AE=aubmindlab/araelectra-base-discriminator
AM=NAMAA-Space/AraModernBert-Base-V1.0

run () { # tag model variant objective [bs] [lr] [epochs]
  local tag=$1 model=$2 var=$3 obj=$4 bs=${5:-32} lr=${6:-2e-5} ep=${7:-8}
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag (done)"; return; }
  while pgrep -f -- "--tag ${tag} " >/dev/null; do sleep 15; done   # only this script's own tag
  echo ">>> $(date +%H:%M:%S) $tag ($obj/$var bs=$bs) ALLDATA"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant "$var" --objective "$obj" \
      --alldata --seed 42 --max_len 160 --bs "$bs" --lr "$lr" --epochs "$ep" --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) done $tag (exit $?)"
}

run arabertv2_reg_ad     $AR d3tok reg      # reg was the objective missing from the _ad set
run marbert_corn_ad      $MB d3tok corn
run camelbert_corn_ad    $CB d3tok corn
run araelectra_soft_ad   $AE d3tok soft
run marbert_soft_ad      $MB d3tok soft
run aramodern_corn_ad    $AM raw   corn
echo "=== AD3_BASE DONE $(date +%H:%M:%S) ==="
run arabertv2_large_emd_ad $LG d3tok emd 24 1.5e-5 6
run arabertv2_large_wkl_ad $LG d3tok wkl 24 1.5e-5 6
echo "=== AD3 DONE $(date +%H:%M:%S) ==="
