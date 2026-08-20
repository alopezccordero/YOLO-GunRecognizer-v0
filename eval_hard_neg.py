###  python eval_hard_neg.py --weights runs/gun_detector/yolo26s_1280_v3/weights/best.pt

import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMGS = ROOT / "benchmarks" / "hard_neg" / "images"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--src", default=str(IMGS), help="Folder of no-gun images.")
    args = ap.parse_args()

    model = YOLO(args.weights)
    maxconf = []
    for r in model.predict(source=args.src, conf=0.01, imgsz=args.imgsz,
                           stream=True, verbose=False):
        cs = r.boxes.conf.tolist() if (r.boxes is not None and len(r.boxes)) else []
        maxconf.append(max(cs) if cs else 0.0)

    n = len(maxconf)
    print("\n" + "=" * 60)
    print(f"Hard-negative FP rate  weights={Path(args.weights).parent.parent.name}")
    print(f"held-out no-gun images: {n}   imgsz={args.imgsz}")
    print("-" * 60)
    print("  conf thresh   FP images   FP rate")
    for t in [0.05, 0.10, 0.20, 0.25, 0.30, 0.50]:
        fp = sum(1 for c in maxconf if c >= t)
        print(f"     >= {t:<5}     {fp:4d}/{n}     {100*fp/n:5.1f}%")
    print("=" * 60)
    print("Lower FP rate = fewer false alarms on non-gun scenes = better precision.")


if __name__ == "__main__":
    main()
