"""Evaluate a trained gun detector on separate slices of the dataset so you get
honest, defensible numbers instead of one blended metric.

Slices (per split):
  - openimages : catalog/scene photos (filenames WITHOUT the cctv_ prefix)
  - cctv       : all merged CCTV frames (cctv_*)
  - youtube    : the in-the-wild youtube source only (cctv_youtube*)

The 'test' slices use videos the model never trained on -> the numbers to
headline. The val 'openimages' slice is directly comparable to the earlier
baseline / yolo26s_960_aug runs (same 240 images).

Usage:
  python eval_slices.py --weights runs/gun_detector/yolo26s_960_aug_2/weights/best.pt
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent
DEST = REPO_ROOT / "gun_dataset"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# (slice name, split folder, filename filter)
SLICES = [
    ("val   / all",        "val",  lambda n: True),
    ("val   / openimages", "val",  lambda n: not n.startswith("cctv_")),
    ("val   / cctv",       "val",  lambda n: n.startswith("cctv_")),
    ("test  / all",        "test", lambda n: True),
    ("test  / openimages", "test", lambda n: not n.startswith("cctv_")),
    ("test  / cctv",       "test", lambda n: n.startswith("cctv_")),
    ("test  / youtube",    "test", lambda n: n.startswith("cctv_youtube")),
]


def list_images(split: str, keep) -> list[str]:
    d = DEST / "images" / split
    return [str(p.resolve()) for p in sorted(d.iterdir())
            if p.suffix.lower() in IMG_EXTS and keep(p.name)]


def recall_prec_at_conf(m, conf):
    """Recall & precision at a FIXED confidence threshold for class 0.

    b.mp / b.mr are reported at the max-F1 point on the PR curve, NOT at a
    chosen conf. To see deployment behaviour at `conf` we read the
    recall/precision-vs-confidence curves that val() computes over the full
    [0,1] threshold sweep and interpolate at `conf`. (The old confusion-matrix
    read returned zeros because the matrix isn't populated with plots=False.)
    Returns (R, P), either of which may be None if the curves aren't available.
    """
    b = m.box
    # Primary: internal curve arrays (px = confidence axis, *_curve per class).
    try:
        px = np.asarray(b.px, dtype=float)            # (1000,) confidences 0..1
        rc = np.asarray(b.r_curve, dtype=float)       # (nc, 1000) recall
        pc = np.asarray(b.p_curve, dtype=float)       # (nc, 1000) precision
        r = rc[0] if rc.ndim == 2 else rc
        p = pc[0] if pc.ndim == 2 else pc
        if px.size and r.size == px.size:
            return float(np.interp(conf, px, r)), float(np.interp(conf, px, p))
    except Exception:
        pass
    # Fallback: public curves_results, matched by axis labels.
    R = P = None
    try:
        for x, y, xlabel, ylabel in b.curves_results:
            if "confidence" not in str(xlabel).lower():
                continue
            yv = np.asarray(y, dtype=float)
            yv = yv[0] if yv.ndim == 2 else yv
            val = float(np.interp(conf, np.asarray(x, dtype=float), yv))
            if str(ylabel).lower().startswith("recall"):
                R = val
            elif str(ylabel).lower().startswith("precision"):
                P = val
    except Exception:
        pass
    return R, P


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="Low conf for mAP / max-F1 P/R integration (default 0.001).")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--augment", action="store_true",
                    help="Enable test-time augmentation (TTA).")
    ap.add_argument("--dep-conf", type=float, default=0.20,
                    help="Deployment conf threshold for the R@conf / P@conf columns.")
    args = ap.parse_args()

    model = YOLO(args.weights)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, split, keep in SLICES:
            imgs = list_images(split, keep)
            if not imgs:
                rows.append((name, 0, None))
                continue
            list_txt = Path(tmp) / f"{name.replace('/', '_').replace(' ', '')}.txt"
            list_txt.write_text("\n".join(imgs) + "\n")
            data_yaml = Path(tmp) / f"{list_txt.stem}.yaml"
            data_yaml.write_text(yaml.safe_dump({
                "path": str(DEST.resolve()),
                "train": str(list_txt),   # unused, present for safety
                "val": str(list_txt),
                "names": {0: "gun"},
            }))
            # Single low-conf pass: gives mAP + max-F1 P/R AND the full
            # recall/precision-vs-confidence curves used for R@conf / P@conf.
            m = model.val(data=str(data_yaml), split="val", imgsz=args.imgsz,
                          conf=args.conf, iou=args.iou, augment=args.augment,
                          workers=0, plots=False, verbose=False)
            b = m.box
            r_dep, p_dep = recall_prec_at_conf(m, args.dep_conf)
            rows.append((name, len(imgs),
                         (b.mp, b.mr, b.map50, b.map, r_dep, p_dep)))

    tta = " +TTA" if args.augment else ""
    print("\n" + "=" * 88)
    print(f"weights={Path(args.weights).parent.parent.name}  imgsz={args.imgsz}"
          f"  dep_conf={args.dep_conf}{tta}")
    print(f"{'slice':22s} {'imgs':>5s}  {'P':>6s} {'R':>6s} "
          f"{'mAP50':>6s} {'mAP50-95':>8s}  {'R@'+str(args.dep_conf):>7s} "
          f"{'P@'+str(args.dep_conf):>7s}")
    print("-" * 88)
    for name, n, met in rows:
        if met is None:
            print(f"{name:22s} {n:5d}   (no images)")
        else:
            p, r, m50, m, rd, pd = met
            rd_s = f"{rd:7.3f}" if rd is not None else "   n/a "
            pd_s = f"{pd:7.3f}" if pd is not None else "   n/a "
            print(f"{name:22s} {n:5d}  {p:6.3f} {r:6.3f} {m50:6.3f} {m:8.3f}"
                  f"  {rd_s} {pd_s}")
    print("=" * 88)
    print("P/R/mAP50/mAP50-95 = PR-curve metrics (P/R at max-F1 point).")
    print(f"R@{args.dep_conf}/P@{args.dep_conf} = recall/precision at the deployment "
          f"conf threshold (interpolated from the R/P-confidence curve).")
    print("Headline number: 'test / youtube' = never-seen, in-the-wild.")


if __name__ == "__main__":
    main()
