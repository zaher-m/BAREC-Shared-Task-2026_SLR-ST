#!/usr/bin/env bash
# Second matrix pass. No upsampling this time (the A/B was mixed, so keep it simple), and
# per-backbone tokenization: AraModernBERT gets raw text, its tokenizer likes unsegmented
# input. Ordered by expected value so the pool can be ensembled as it fills up.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}

while pgrep -f 'slra_st.train' >/dev/null; do sleep 15; done

run () { # tag model objective variant
  local tag=$1 model=$2 obj=$3 var=$4
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag (done)"; return; }
  echo ">>> $(date +%H:%M) $tag ($model / $obj / $var)"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant "$var" --objective "$obj" \
      --max_len 160 --bs 48 --lr 2e-5 --epochs 7 --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M) done $tag (exit $?)"
}

run arabertv2_reg_d3   aubmindlab/bert-base-arabertv2            reg  d3tok
run marbert_reg_d3n    UBC-NLP/MARBERTv2                         reg  d3tok
run camelbert_reg_d3n  CAMeL-Lab/bert-base-arabic-camelbert-mix  reg  d3tok
run aramodern_reg_raw  NAMAA-Space/AraModernBert-Base-V1.0       reg  raw
run arabertv2_corn_d3  aubmindlab/bert-base-arabertv2            corn d3tok
echo "=== MATRIX2 DONE $(date +%H:%M) ==="
