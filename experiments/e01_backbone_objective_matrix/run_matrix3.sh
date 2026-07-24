#!/usr/bin/env bash
# Third matrix pass: add AraELECTRA and ARBERTv2, plus CORN heads for the backbones that only
# had regression. CORN gives better Acc/MAE, which is where we trail the accuracy-heavy teams.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}

while pgrep -f 'slra_st.train' >/dev/null; do sleep 15; done
while ! grep -q 'DL_DONE' artifacts/logs/dl_v3.log 2>/dev/null; do sleep 10; done   # wait for the downloads

run () { # tag model objective variant
  local tag=$1 model=$2 obj=$3 var=$4
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag (done)"; return; }
  echo ">>> $(date +%H:%M) $tag ($model / $obj / $var)"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant "$var" --objective "$obj" \
      --max_len 160 --bs 32 --lr 2e-5 --epochs 7 --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M) done $tag (exit $?)"
}

run araelectra_reg_d3   aubmindlab/araelectra-base-discriminator  reg  d3tok
run arbertv2_reg_d3     UBC-NLP/ARBERTv2                          reg  d3tok
run araelectra_corn_d3  aubmindlab/araelectra-base-discriminator  corn d3tok
run marbert_corn_d3     UBC-NLP/MARBERTv2                         corn d3tok
run camelbert_corn_d3   CAMeL-Lab/bert-base-arabic-camelbert-mix   corn d3tok
run aramodern_corn_raw  NAMAA-Space/AraModernBert-Base-V1.0       corn raw
echo "=== MATRIX3 DONE $(date +%H:%M) ==="
