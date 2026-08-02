#!/usr/bin/env bash
# Deadline sequence, meant to run unattended:
#  1. wait for the last _ad model, or a hard cutoff
#  2. stop this project's training so inference gets the GPU (kills match this project's tags
#     only, so a sibling project sharing the GPU is left alone)
#  3. validate the enlarged blend, continue only if it beats the shipped one
#  4. run blind inference (bs 256, caches member scores) and check the submission file
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=${PY:-.venv/bin/python}
HERE=experiments/e09_endgame_gate
CUTOFF=${CUTOFF:-2215}          # HHMM: stop waiting for training past this

echo "=== endgame start $(date +%H:%M) (cutoff $CUTOFF) ==="
while [ ! -f artifacts/scores/members/scores_arabertv2_reg_ad.npz ]; do
  now=$(date +%H%M)
  [ "$now" -ge "$CUTOFF" ] && { echo "cutoff reached at $now, proceeding without reg_ad"; break; }
  sleep 60
done
[ -f artifacts/scores/members/scores_arabertv2_reg_ad.npz ] && echo "reg_ad ready $(date +%H:%M)"

pkill -f 'run_ad3.sh' 2>/dev/null
for t in arabertv2_reg_ad marbert_corn_ad camelbert_corn_ad araelectra_soft_ad \
         marbert_soft_ad aramodern_corn_ad arabertv2_large_emd_ad arabertv2_large_wkl_ad; do
  pkill -9 -f -- "--tag ${t} " 2>/dev/null
done
sleep 5
echo "this project's training stopped $(date +%H:%M); GPU freed for inference"

echo "=== validate enlarged blend $(date +%H:%M) ==="
$PY "$HERE/endgame.py" 2>&1 | tail -12

if [ -f configs/ensembles/final_v8.json ]; then
  echo "=== blind inference on final_v8 $(date +%H:%M) ==="
  $PY -m slra_st.infer --input data/blind_sent.parquet --id-col "Sentence ID" \
      --text-col "Sentence" --ensemble final_v8 \
      --out submissions/submitted/prediction_final > artifacts/logs/infer_final.log 2>&1
  echo "inference exit=$? $(date +%H:%M)"
  grep -E 'cached|wrote' artifacts/logs/infer_final.log | tail -2
  echo "=== validate submission ==="
  $PY -m slra_st.submission submissions/submitted/prediction_final \
      --previous submissions/submitted/prediction_blend
else
  echo "=== no improvement found; V7 (prediction_blend.zip) remains final ==="
fi
echo "=== ENDGAME DONE $(date +%H:%M) ==="
