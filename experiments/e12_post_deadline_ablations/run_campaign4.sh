#!/usr/bin/env bash
# Night run, reordered ahead of run_campaign3.sh's remaining phases. The one untested source of
# new information is capacity plus a different pretraining corpus, i.e. ALLaM-7B (Arabic LLM,
# ~52x the large encoder) with LoRA on all labeled data.
# Pre-registered rule: it only joins the system if its blend gain clears the +-0.1-0.2 QWK
# noise floor. Then the emd 5-fold family, then blind inference for both.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}

run () { # tag model objective extra...
  local tag=$1 model=$2 obj=$3; shift 3
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag"; return; }
  while pgrep -f 'slra_st.train' >/dev/null; do sleep 30; done
  echo ">>> $(date +%H:%M:%S) $tag"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant d3tok --objective "$obj" \
      --seed 42 --max_len 160 "$@" > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) done $tag (exit $?)"
}

infer_batch () { # name member1 member2 ...
  local name=$1; shift
  local members=""; for m in "$@"; do
    [ -f "artifacts/scores/members/scores_${m}.npz" ] && members="$members\"$m\","
  done
  [ -z "$members" ] && return
  $PY - <<PY
import json
from slra_st.calibration import DEFAULT_THRESHOLDS
m = [${members%,}]
json.dump({"members": m, "weights": [1]*len(m), "thresholds": DEFAULT_THRESHOLDS},
          open("configs/ensembles/research/${name}.json", "w"), indent=2)
PY
  echo ">>> $(date +%H:%M:%S) blind inference: $name"
  $PY -m slra_st.infer --input data/proc_d3tok_blind.parquet --id-col ID --text-col text \
      --pre-d3tok --ensemble "research/${name}" \
      --out "artifacts/predictions/research/blind_${name}" \
      > "artifacts/logs/infer_${name}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) inference $name (exit $?)"
}

run allam7b_soft_ad ALLaM-AI/ALLaM-7B-Instruct-preview soft --alldata --lora --bs 16 --lr 1e-4 --epochs 2 --patience 2
infer_batch allam allam7b_soft_ad
echo "=== ALLAM DONE $(date +%H:%M:%S) ==="
for f in 0 1 2 3 4; do run arabertv2_emd_kf$f aubmindlab/bert-base-arabertv2 emd --fold $f --bs 32 --lr 2e-5 --epochs 5 --patience 2; done
infer_batch kfemd arabertv2_emd_kf0 arabertv2_emd_kf1 arabertv2_emd_kf2 arabertv2_emd_kf3 arabertv2_emd_kf4
echo "=== CAMPAIGN4 DONE $(date +%H:%M:%S) ==="
