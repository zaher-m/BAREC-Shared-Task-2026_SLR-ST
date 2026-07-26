#!/usr/bin/env bash
# Throughput test: train several models at once on the one GPU, since VRAM is free anyway.
# ABANDONED after ~43 min. With three co-tenants an epoch took 1273s vs 298s solo, so 4.2x
# slower per job and no net gain. The GPU is compute-bound, free VRAM does not help.
# Replaced by run_seq.sh.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}   # several trainers on one host oversubscribe the CPU
MAXJOBS=${MAXJOBS:-3}

# tag|model|objective. arabertv2_soft_d3 is left out on purpose, it is already training
JOBS=(
  "arabertv2_wkl_d3|aubmindlab/bert-base-arabertv2|wkl"
  "marbert_soft_d3|UBC-NLP/MARBERTv2|soft"
  "camelbert_soft_d3|CAMeL-Lab/bert-base-arabic-camelbert-mix|soft"
  "araelectra_soft_d3|aubmindlab/araelectra-base-discriminator|soft"
  "arabertv2_emd_d3|aubmindlab/bert-base-arabertv2|emd"
  "marbert_wkl_d3|UBC-NLP/MARBERTv2|wkl"
)

run_one () {
  local tag=$1 model=$2 obj=$3
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag (already done)"; return; }
  pgrep -f -- "--tag $tag " >/dev/null && { echo "skip $tag (already running)"; return; }
  echo ">>> $(date +%H:%M:%S) start $tag ($obj)"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant d3tok --objective "$obj" \
      --max_len 160 --bs 32 --lr 2e-5 --epochs 7 --patience 2 \
      > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) done $tag (exit $?)"
}

running=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r tag model obj <<< "$spec"
  run_one "$tag" "$model" "$obj" &
  running=$((running+1))
  if [ "$running" -ge "$MAXJOBS" ]; then
    wait -n
    running=$((running-1))
  fi
  sleep 20   # stagger model loading so the HF cache and disk are not hit all at once
done
wait
echo "=== PAR DONE $(date +%H:%M:%S) ==="
