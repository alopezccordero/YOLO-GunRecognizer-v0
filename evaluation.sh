#!/bin/bash
CHAMP=runs/gun_detector/yolo26s_1280_diverse_2_downsample/weights/best.pt
V2=runs/gun_detector/yolo26s_1280_v2/weights/best.pt
V3=runs/gun_detector/yolo26s_1280_v3/weights/best.pt

echo "===== PER-SLICE (youtube recall etc.) ====="
for w in "$CHAMP" "$V2" "$V3"; do python eval_slices.py --weights "$w"; done

echo "===== HARD-NEGATIVE FP RATE ====="
for w in "$CHAMP" "$V2" "$V3"; do python eval_hard_neg.py --weights "$w"; done

echo "===== UGR BENCHMARK ====="
for w in "$CHAMP" "$V2" "$V3"; do python eval_ugr.py --weights "$w"; done