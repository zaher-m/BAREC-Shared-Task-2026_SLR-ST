#!/usr/bin/env bash
# AraBERT-large with the dynamic-padding trainer, at bs 24 instead of 12.
# AraBERTv2 dominates the ensemble across all five objectives, so a bigger AraBERT is a better
# bet than yet another base-size backbone.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}
MODEL=aubmindlab/bert-large-arabertv2

run () { # tag objective
  local tag=$1 obj=$2
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag (done)"; return; }
  while pgrep -f 'slra_st.train' >/dev/null; do sleep 15; done
  echo ">>> $(date +%H:%M:%S) start $tag ($obj)"
  $PY -m slra_st.train --model "$MODEL" --tag "$tag" --variant d3tok --objective "$obj" \
      --max_len 160 --bs 24 --lr 1.5e-5 --epochs 6 --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) done $tag (exit $?)"
}
run arabertv2_large_reg   reg
run arabertv2_large_corn  corn
run arabertv2_large_soft  soft
run arabertv2_large_emd   emd
echo "=== LARGE-LOCAL DONE $(date +%H:%M:%S) ==="
