#!/usr/bin/env bash
# Diversity campaign, ~24 models: seed replicas of the core first, then new backbone/objective
# combos, then AraBERT-large. Phases ordered so the pool can be re-ensembled after each one.
#
# Phase 1 answered the question and the rest was dropped: seed replicas added about nothing,
# because the leftover error is a systematic level-12 bias, not variance that averaging can
# cancel. Phases 2-5 were replaced by run_campaign2.sh, which spent the same GPU hours on
# members that are actually decorrelated.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}

AR=aubmindlab/bert-base-arabertv2
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
  echo ">>> $(date +%H:%M:%S) $tag ($obj/$var seed=$seed bs=$bs)"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant "$var" --objective "$obj" \
      --seed "$seed" --max_len 160 --bs "$bs" --lr "$lr" --epochs "$ep" --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) done $tag (exit $?)"
}

# ===== PHASE 1: AraBERTv2 seed diversity (seed 101) =====
run arabertv2_reg_s2   $AR d3tok reg  101
run arabertv2_soft_s2  $AR d3tok soft 101
run arabertv2_corn_s2  $AR d3tok corn 101
run arabertv2_emd_s2   $AR d3tok emd  101
run arabertv2_wkl_s2   $AR d3tok wkl  101
echo "=== PHASE1 DONE $(date +%H:%M:%S) ==="

# ===== PHASE 2: a second seed (202) + raw-variant diversity =====
run arabertv2_reg_s3   $AR d3tok reg  202
run arabertv2_soft_s3  $AR d3tok soft 202
run arabertv2_corn_s3  $AR d3tok corn 202
run arabertv2_emd_s3   $AR d3tok emd  202
run arabertv2_reg_raw2 $AR raw   reg  101
echo "=== PHASE2 DONE $(date +%H:%M:%S) ==="

# ===== PHASE 3: cross-backbone new objectives + seeds =====
run marbert_reg_s2     $MB d3tok reg  101
run marbert_emd_d3     $MB d3tok emd  42
run marbert_corn_s2    $MB d3tok corn 101
run camelbert_soft_s2  $CB d3tok soft 101
run camelbert_emd_d3   $CB d3tok emd  42
run camelbert_wkl_d3   $CB d3tok wkl  42
run araelectra_soft_s2 $AE d3tok soft 101
run araelectra_emd_d3  $AE d3tok emd  42
echo "=== PHASE3 DONE $(date +%H:%M:%S) ==="

# ===== PHASE 4: more backbone diversity =====
run aramodern_corn_s2  $AM  raw   corn 101
run aramodern_soft_raw $AM  raw   soft 42
run arbertv2_corn_d3   $ARB d3tok corn 42
echo "=== PHASE4 DONE $(date +%H:%M:%S) ==="

# ===== PHASE 5: AraBERT-large =====
run arabertv2_large_reg  $LG d3tok reg  42 24 1.5e-5 6
run arabertv2_large_corn $LG d3tok corn 42 24 1.5e-5 6
run arabertv2_large_soft $LG d3tok soft 42 24 1.5e-5 6
echo "=== CAMPAIGN DONE $(date +%H:%M:%S) ==="
