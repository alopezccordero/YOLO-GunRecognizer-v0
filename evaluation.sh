#!/bin/bash
# Runs all evaluations for each model and writes ONLY the summary tables to
# results.txt (Ultralytics scan/speed/warning logs are filtered out).
set -o pipefail

OUT=results.txt
CHAMP=runs/gun_detector/yolo26s_1280_diverse_2_downsample/weights/best.pt
V2=runs/gun_detector/yolo26s_1280_v2/weights/best.pt
V3=runs/gun_detector/yolo26s_1280_v3-2/weights/best.pt

{
  echo "# Evaluation run: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "NORMAL EVALUATION"
  for w in "$CHAMP" "$V2" "$V3"; do python eval_slices.py  --weights "$w"; done
  echo "HARD NEGATIVE EVALUATIONS"
  for w in "$CHAMP" "$V2" "$V3"; do python eval_hard_neg.py --weights "$w"; done
  echo "UGR EVALUATIONS"
  for w in "$CHAMP" "$V2" "$V3"; do python eval_ugr.py      --weights "$w"; done
} 2>&1 | awk '
  /^# Evaluation run:/ { print; next }               # keep the timestamp header
  /^====/              { inblock = !inblock; print; next }   # toggle on each ==== fence
  inblock              { print; next }               # keep everything inside a block
  /EVALUATION/         { print }                      # keep the section headers
' | tee "$OUT"
