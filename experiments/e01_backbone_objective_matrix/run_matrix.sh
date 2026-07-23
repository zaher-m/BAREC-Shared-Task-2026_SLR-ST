#!/usr/bin/env bash
# First backbone matrix: 3 base backbones x {reg, corn}, with class upsampling.
# One model at a time. The GPU is shared, and serial turned out to be faster than parallel.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}

# marbert_reg_d3 was launched by hand before this script; wait for it to release the GPU
while pgrep -f 'slra_st.train --model UBC-NLP/MARBERTv2 --tag marbert_reg_d3' >/dev/null; do sleep 20; done

run () { # tag model objective
  local tag=$1 model=$2 obj=$3
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag (done)"; return; }
  echo ">>> $(date +%H:%M) training $tag ($model / $obj)"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant d3tok --objective "$obj" \
      --max_len 160 --bs 48 --lr 2e-5 --epochs 7 --patience 2 --upsample \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M) finished $tag (exit $?)"
}

run camelbert_reg_d3  CAMeL-Lab/bert-base-arabic-camelbert-mix  reg
run aramodern_reg_d3  NAMAA-Space/AraModernBert-Base-V1.0       reg
run marbert_corn_d3   UBC-NLP/MARBERTv2                         corn
run aramodern_corn_d3 NAMAA-Space/AraModernBert-Base-V1.0       corn
echo "=== ALL MATRIX RUNS DONE $(date +%H:%M) ==="
