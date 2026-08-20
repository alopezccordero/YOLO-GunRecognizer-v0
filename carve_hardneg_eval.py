"""Build a HELD-OUT hard-negative eval set: security-footage images the model did
NOT train on (excludes the hn_ files already in train). No labels needed (all
no-gun). Used by eval_hard_neg.py to measure the false-positive rate honestly."""
import random, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "_roboflow_dl" / "people-fkg4e-v1"
TRAIN = ROOT / "gun_dataset" / "images" / "train"
OUT = ROOT / "benchmarks" / "hard_neg" / "images"
OUT.mkdir(parents=True, exist_ok=True)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
N, SEED = 400, 123      # different seed than the training sample (42)

# original filenames already used for training (hn_<name> in train)
trained = {p.name[3:] for p in TRAIN.iterdir() if p.name.startswith("hn_")}
print(f"already-trained hard negatives to exclude: {len(trained)}")

pool = [p for p in SRC.rglob("*")
        if p.suffix.lower() in IMG_EXTS and p.name not in trained]
random.seed(SEED)
picked = random.sample(pool, min(N, len(pool)))
for p in picked:
    shutil.copy2(p, OUT / p.name)

print(f"pool (unused negatives): {len(pool)}   carved held-out eval set: {len(picked)}")
print(f"-> {OUT}")
