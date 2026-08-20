"""Build a CLEAN held-out hard-negative FP benchmark.

Benchmark source = security-footage-analysis (a DIFFERENT source than the
people-fkg4e negatives now used for TRAINING). Every candidate is checked with a
perceptual hash (dHash) against the WHOLE training set, so no image the model has
seen -- including near-duplicate video frames -- leaks into the benchmark.

The previous version excluded only by exact filename, which missed same-video
frames and INFLATED the reported false-positive rate. No labels needed (all
images are no-gun).

  python carve_hardneg_eval.py                       # 400 clean security-footage images
  python carve_hardneg_eval.py --src _roboflow_dl/<other> --n 400
"""
import argparse, random, shutil
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "gun_dataset" / "images" / "train"
OUT = ROOT / "benchmarks" / "hard_neg" / "images"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def dhash(path, size=8):
    try:
        im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return None
    px = list(im.getdata()); b = 0
    for r in range(size):
        row = r * (size + 1)
        for c in range(size):
            b = (b << 1) | (1 if px[row + c] > px[row + c + 1] else 0)
    return b


def ham(a, b):
    return bin(a ^ b).count("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="_roboflow_dl/security-footage-analysis-v1",
                    help="Benchmark source (must differ from the TRAIN negatives' source).")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--thresh", type=int, default=8,
                    help="dHash distance: <= this to any train image is an overlap -> dropped.")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    src = ROOT / args.src
    OUT.mkdir(parents=True, exist_ok=True)

    print("hashing train set (this is the leakage guard)...")
    train_h = [h for p in TRAIN.iterdir() if p.suffix.lower() in IMG_EXTS
               for h in [dhash(p)] if h is not None]
    print(f"train hashed: {len(train_h)}")

    cands = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS]
    random.seed(args.seed)
    random.shuffle(cands)
    print(f"candidates in {src.name}: {len(cands)} -- picking {args.n} that do NOT overlap train...")

    picked, dropped = 0, 0
    for p in cands:
        if picked >= args.n:
            break
        h = dhash(p)
        if h is None:
            continue
        if any(ham(h, t) <= args.thresh for t in train_h):   # near-duplicate of a training image
            dropped += 1
            continue
        shutil.copy2(p, OUT / p.name)
        picked += 1

    print(f"\ndropped as train-overlap (dHash <= {args.thresh}): {dropped}")
    print(f"CLEAN held-out FP benchmark: {picked} images -> {OUT}")
    if picked < args.n:
        print("WARNING: fewer than requested were clean; the source overlaps train heavily.")


if __name__ == "__main__":
    main()
