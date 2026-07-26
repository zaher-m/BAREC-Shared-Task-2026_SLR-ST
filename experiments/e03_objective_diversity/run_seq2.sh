#!/usr/bin/env bash
# Continuation at bs 64 (was 32). At these sequence lengths the GPU is kernel-bound, so bigger
# batches mean bigger matmuls and half the steps. lr 2e-5 -> 2.5e-5 for the larger batch.
# Waits for the in-flight bs32 run first so nothing stacks.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}

run () { # tag model objective
  local tag=$1 model=$2 obj=$3
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag (done)"; return; }
  while pgrep -f 'slra_st.train' >/dev/null; do sleep 15; done
  echo ">>> $(date +%H:%M:%S) start $tag ($obj) bs64"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant d3tok --objective "$obj" \
      --max_len 160 --bs 64 --lr 2.5e-5 --epochs 8 --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) done $tag (exit $?)"
}
run marbert_soft_d3     UBC-NLP/MARBERTv2                         soft
run camelbert_soft_d3   CAMeL-Lab/bert-base-arabic-camelbert-mix   soft
run araelectra_soft_d3  aubmindlab/araelectra-base-discriminator  soft
run arabertv2_emd_d3    aubmindlab/bert-base-arabertv2            emd
run marbert_wkl_d3      UBC-NLP/MARBERTv2                         wkl
echo "=== SEQ2 OBJ DONE $(date +%H:%M:%S) ==="
