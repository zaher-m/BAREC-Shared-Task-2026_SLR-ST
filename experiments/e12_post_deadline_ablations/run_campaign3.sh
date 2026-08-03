#!/usr/bin/env bash
# Post-deadline campaign. These runs answer paper questions, they are not meant to move the
# leaderboard.
#
#  P) PSEUDO-LABEL adaptation: train on all labeled data plus the blind sentences carrying the
#     submission 10's labels. How much of a 21-model ensemble fits in ONE base model if
#     the ensemble teaches it?
#  K) 5-FOLD by document over all 69k labeled sentences. The 2-partition blend (train-only +
#     all-data) was worth +0.31, so does partition diversity keep paying at 5?
#
# Each family ends with a blind inference. Those use the default threshold grid, i.e. they are
# uncalibrated on purpose: what we want is the score matrix, not the labels.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}
AR=aubmindlab/bert-base-arabertv2
LG=aubmindlab/bert-large-arabertv2

# ---- step 0: one-time d3tok pass over blind, labelled by the 85.4 submission ----
if [ ! -f data/proc_d3tok_blind.parquet ]; then
  echo ">>> $(date +%H:%M:%S) preprocessing blind (d3tok, one-time)"
  $PY - <<'PY'
import pandas as pd
from slra_st.preprocess import build_d3tok
b = pd.read_parquet("data/blind_sent.parquet")
lab = pd.read_csv("submissions/submitted/pred_priorQWK_acc")   # the 85.4 submission
assert list(b["Sentence ID"]) == list(lab["Sentence ID"])
tfn = build_d3tok()
out = pd.DataFrame({"ID": b["Sentence ID"].astype(str),
                    "text": [tfn(t) for t in b["Sentence"].astype(str)],
                    "label19": lab["Prediction"].astype(int)})
out.to_parquet("data/proc_d3tok_blind.parquet")
print("wrote data/proc_d3tok_blind.parquet", len(out))
PY
fi

run () { # tag model objective extra...
  local tag=$1 model=$2 obj=$3; shift 3
  [ -f "artifacts/scores/members/scores_${tag}.npz" ] && { echo "skip $tag"; return; }
  while pgrep -f -- "--tag ${tag} " >/dev/null; do sleep 15; done
  echo ">>> $(date +%H:%M:%S) $tag ($obj $*)"
  $PY -m slra_st.train --model "$model" --tag "$tag" --variant d3tok --objective "$obj" \
      --seed 42 --max_len 160 "$@" > "artifacts/logs/train_${tag}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) done $tag (exit $?)"
}

infer_batch () { # name member1 member2 ...
  local name=$1; shift
  local members=""; for m in "$@"; do
    [ -f "artifacts/scores/members/scores_${m}.npz" ] && members="$members\"$m\","
  done
  [ -z "$members" ] && { echo "no members for $name"; return; }
  $PY - <<PY
import json
from slra_st.calibration import DEFAULT_THRESHOLDS
m = [${members%,}]
json.dump({"members": m, "weights": [1]*len(m), "thresholds": DEFAULT_THRESHOLDS},
          open("configs/ensembles/research/${name}.json", "w"), indent=2)
print("infer set ${name}:", m)
PY
  echo ">>> $(date +%H:%M:%S) blind inference: $name"
  $PY -m slra_st.infer --input data/proc_d3tok_blind.parquet --id-col ID --text-col text \
      --pre-d3tok --ensemble "research/${name}" \
      --out "artifacts/predictions/research/blind_${name}" \
      > "artifacts/logs/infer_${name}.log" 2>&1
  echo "<<< $(date +%H:%M:%S) inference $name (exit $?)"
}

# ===== P: pseudo-label domain adaptation (two diverse strong configs) =====
run arabertv2_emd_ps  $AR emd  --alldata --pseudo data/proc_d3tok_blind.parquet --bs 32 --lr 2e-5 --epochs 5 --patience 2
run arabertv2_soft_ps $AR soft --alldata --pseudo data/proc_d3tok_blind.parquet --bs 32 --lr 2e-5 --epochs 5 --patience 2
infer_batch psfam arabertv2_emd_ps arabertv2_soft_ps
echo "=== PSEUDO FAMILY DONE $(date +%H:%M:%S) ==="

# ===== K: 5-fold document-partitioned over all labeled data =====
for f in 0 1 2 3 4; do run arabertv2_emd_kf$f  $AR emd  --fold $f --bs 32 --lr 2e-5 --epochs 5 --patience 2; done
infer_batch kfemd arabertv2_emd_kf0 arabertv2_emd_kf1 arabertv2_emd_kf2 arabertv2_emd_kf3 arabertv2_emd_kf4
echo "=== KFOLD-EMD DONE $(date +%H:%M:%S) ==="
for f in 0 1 2 3 4; do run arabertv2_soft_kf$f $AR soft --fold $f --bs 32 --lr 2e-5 --epochs 5 --patience 2; done
infer_batch kfsoft arabertv2_soft_kf0 arabertv2_soft_kf1 arabertv2_soft_kf2 arabertv2_soft_kf3 arabertv2_soft_kf4
echo "=== KFOLD-SOFT DONE $(date +%H:%M:%S) ==="
for f in 0 1 2 3 4; do run arabertv2_large_soft_kf$f $LG soft --fold $f --bs 24 --lr 8e-6 --epochs 4 --patience 2; done
infer_batch kflarge arabertv2_large_soft_kf0 arabertv2_large_soft_kf1 arabertv2_large_soft_kf2 arabertv2_large_soft_kf3 arabertv2_large_soft_kf4
echo "=== CAMPAIGN3 DONE $(date +%H:%M:%S) ==="
