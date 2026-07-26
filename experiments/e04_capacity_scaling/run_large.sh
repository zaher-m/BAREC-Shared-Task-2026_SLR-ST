#!/usr/bin/env bash
# Capacity, first attempt: AraBERT-large (~2.7x the base parameter count). error_analysis.py
# had shown the leftover loss is a systematic under-prediction tail at level 12, which is a
# bias, so more capacity can help where seed replicas cannot.
# Downloads the checkpoint first so the campaign does not wait on it.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}
export HF_TOKEN=${HF_TOKEN:-}   # gated downloads need a token in the environment

echo ">>> downloading bert-large-arabertv2 $(date +%H:%M)"
$PY - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("aubmindlab/bert-large-arabertv2",
    allow_patterns=['*.json','*.txt','pytorch_model.bin','*.safetensors','vocab*','tokenizer*'])
print("downloaded", p)
PY

run () { # tag model objective
  local tag=$1 model=$2 obj=$3
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag"; return; }
  echo ">>> $(date +%H:%M) $tag ($obj)"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant d3tok --objective "$obj" \
      --max_len 160 --bs 12 --lr 1.5e-5 --epochs 6 --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M) done $tag (exit $?)"
}
run arabertv2_large_reg   aubmindlab/bert-large-arabertv2  reg
run arabertv2_large_corn  aubmindlab/bert-large-arabertv2  corn
echo "=== LARGE DONE $(date +%H:%M) ==="
