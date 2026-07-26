#!/usr/bin/env bash
# Objective diversity: add soft-label (SORD), weighted-kappa-loss and squared-EMD heads on top
# of reg/corn. Bet is that five loss geometries on one strong backbone decorrelate better than
# five different backbones at the same single-model quality.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}
while pgrep -f 'slra_st.train' >/dev/null; do sleep 15; done

run () { # tag model objective
  local tag=$1 model=$2 obj=$3
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag"; return; }
  echo ">>> $(date +%H:%M) $tag ($obj)"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant d3tok --objective "$obj" \
      --max_len 160 --bs 32 --lr 2e-5 --epochs 7 --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M) done $tag (exit $?)"
}
run arabertv2_soft_d3   aubmindlab/bert-base-arabertv2            soft
run arabertv2_wkl_d3    aubmindlab/bert-base-arabertv2            wkl
run marbert_soft_d3     UBC-NLP/MARBERTv2                         soft
run camelbert_soft_d3   CAMeL-Lab/bert-base-arabic-camelbert-mix   soft
run araelectra_soft_d3  aubmindlab/araelectra-base-discriminator  soft
run arabertv2_emd_d3    aubmindlab/bert-base-arabertv2            emd
run marbert_wkl_d3      UBC-NLP/MARBERTv2                         wkl
echo "=== OBJ DONE $(date +%H:%M) ==="
