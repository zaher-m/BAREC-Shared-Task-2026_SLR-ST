#!/usr/bin/env bash
# Fix the one real defect in the pool: arabertv2_large_reg_ad collapsed to a constant output
# (loss stuck near 6.5, dev QWK 0.000 from epoch 0). bert-large with a scalar regression head
# diverges at lr 1.5e-5. Retrain at half the learning rate, then add the next AD base models.
# Threshold tuning is done at this point (blind swings of 0.1-0.2 are noise), so member
# quality is the only lever left.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}

run () { # tag model variant objective [bs] [lr] [epochs]
  local tag=$1 model=$2 var=$3 obj=$4 bs=${5:-32} lr=${6:-2e-5} ep=${7:-8}
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag (done)"; return; }
  while pgrep -f -- "--tag ${tag} " >/dev/null; do sleep 15; done
  echo ">>> $(date +%H:%M:%S) $tag ($obj/$var bs=$bs lr=$lr) ALLDATA"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant "$var" --objective "$obj" \
      --alldata --seed 42 --max_len 160 --bs "$bs" --lr "$lr" --epochs "$ep" --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) done $tag (exit $?)"
}

# the fix: lr 1.5e-5 -> 8e-6 for the large regression head
run arabertv2_large_reg_ad2 aubmindlab/bert-large-arabertv2 d3tok reg 24 8e-6 6
echo "=== FIX DONE $(date +%H:%M:%S) ==="
# deepen the AD pool with the next base models (~40 min each)
run marbert_corn_ad     UBC-NLP/MARBERTv2                         d3tok corn
run camelbert_corn_ad   CAMeL-Lab/bert-base-arabic-camelbert-mix   d3tok corn
run araelectra_soft_ad  aubmindlab/araelectra-base-discriminator   d3tok soft
echo "=== FIX+POOL DONE $(date +%H:%M:%S) ==="
