"""Evaluate a trained gun detector on the DEDUPED UGR handgun benchmark.
Uses benchmarks/ugr_pistol/ugr_clean.txt (UGR images with no overlap vs train).

Usage:
  python eval_ugr.py --weights runs/gun_detector/yolo26s_1280_diverse_2_downsample/weights/best.pt
"""
from __future__ import annotations
import argparse
import tempfile
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
UGR = ROOT / "benchmarks" / "ugr_pistol"


def recall_prec_at_conf(m, conf):
    """R/P at a fixed conf, interpolated from the R/P-confidence curves."""
    b = m.box
    try:
        px = np.asarray(b.px, float)
        rc = np.asarray(b.r_curve, float)
        pc = np.asarray(b.p_curve, float)
        r = rc[0] if rc.ndim == 2 else rc
        p = pc[0] if pc.ndim == 2 else pc
        if px.size and r.size == px.size:
            return float(np.interp(conf, px, r)), float(np.interp(conf, px, p))
    except Exception:
        pass
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--dep-conf", type=float, default=0.20)
    ap.add_argument("--list", default=str(UGR / "ugr_clean.txt"),
                    help="Image list to eval (default: deduped ugr_clean.txt).")
    args = ap.parse_args()

    model = YOLO(args.weights)
    with tempfile.TemporaryDirectory() as tmp:
        list_abs = str(Path(args.list).resolve())   # absolute so it isn't re-based under `path`
        data_yaml = Path(tmp) / "bench.yaml"
        data_yaml.write_text(yaml.safe_dump({
            "path": str(Path(args.list).resolve().parent),
            "train": list_abs,
            "val": list_abs,
            "names": {0: "gun"},
        }))
        m = model.val(data=str(data_yaml), split="val", imgsz=args.imgsz,
                      conf=args.conf, iou=args.iou, augment=args.augment,
                      workers=0, plots=False, verbose=False)
    b = m.box
    rd, pd = recall_prec_at_conf(m, args.dep_conf)
    rd_s = f"{rd:.3f}" if rd is not None else "n/a"
    pd_s = f"{pd:.3f}" if pd is not None else "n/a"

    n_imgs = sum(1 for _ in open(args.list)) if Path(args.list).exists() else "?"
    tta = " +TTA" if args.augment else ""
    print("\n" + "=" * 72)
    print(f"{Path(args.list).stem} benchmark (deduped vs train)  imgsz={args.imgsz}{tta}")
    print(f"weights = {Path(args.weights).parent.parent.name}   images = {n_imgs}")
    print("-" * 72)
    print(f"  P        {b.mp:.3f}")
    print(f"  R        {b.mr:.3f}   (max-F1)")
    print(f"  mAP50    {b.map50:.3f}")
    print(f"  mAP50-95 {b.map:.3f}")
    print(f"  R@{args.dep_conf}   {rd_s}    P@{args.dep_conf} {pd_s}")
    print("=" * 72)


if __name__ == "__main__":
    main()
