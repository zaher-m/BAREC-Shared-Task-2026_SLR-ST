#!/usr/bin/env bash
# Sequential replacement for run_par.sh, with dynamic per-batch padding in the trainer.
# One model at a time gets the whole GPU. Padding 17-token sentences up to 160 was the real
# waste: removing it took epochs from ~300s to ~213s.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}

run () { # tag model objective
  local tag=$1 model=$2 obj=$3
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag (done)"; return; }
  while pgrep -f 'slra_st.train' >/dev/null; do sleep 15; done
  echo ">>> $(date +%H:%M:%S) start $tag ($obj)"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant d3tok --objective "$obj" \
      --max_len 160 --bs 32 --lr 2e-5 --epochs 7 --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) done $tag (exit $?)"
}
run arabertv2_soft_d3   aubmindlab/bert-base-arabertv2            soft
run arabertv2_wkl_d3    aubmindlab/bert-base-arabertv2            wkl
run marbert_soft_d3     UBC-NLP/MARBERTv2                         soft
run camelbert_soft_d3   CAMeL-Lab/bert-base-arabic-camelbert-mix   soft
run araelectra_soft_d3  aubmindlab/araelectra-base-discriminator  soft
run arabertv2_emd_d3    aubmindlab/bert-base-arabertv2            emd
run marbert_wkl_d3      UBC-NLP/MARBERTv2                         wkl
echo "=== SEQ OBJ DONE $(date +%H:%M:%S) ==="
