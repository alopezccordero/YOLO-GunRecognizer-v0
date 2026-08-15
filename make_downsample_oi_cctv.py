#experiment 4 - downsample cctv and open images by 1/2
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "gun_dataset"
TRAIN_IMG = DEST / "images" / "train"
CCTV_FRAC, OI_FRAC, SEED = 1/3, 1/2, 42
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def kind(name):
    if name.startswith("cctv_"):         return "cctv"
    if name.startswith("r46_"):          return "r46"
    if name.startswith("pistol-csvic_"): return "pistol"   # dropped
    return "oi"

imgs = [p for p in sorted(TRAIN_IMG.iterdir()) if p.suffix.lower() in IMG_EXTS]
buckets = {"cctv": [], "r46": [], "oi": [], "pistol": []}
for p in imgs:
    buckets[kind(p.name)].append(p)

random.seed(SEED)
keep_cctv = random.sample(buckets["cctv"], round(len(buckets["cctv"]) * CCTV_FRAC))
keep_oi   = random.sample(buckets["oi"],   round(len(buckets["oi"])   * OI_FRAC))
kept = sorted(buckets["r46"] + keep_cctv + keep_oi, key=lambda p: p.name)

list_path = DEST / "train_downsample_oi_cctv.txt"
list_path.write_text("\n".join(p.resolve().as_posix() for p in kept) + "\n")
yaml_path = DEST / "dataset_downsample_oi_cctv.yaml"
yaml_path.write_text(
    f"path: {DEST.resolve().as_posix()}\n"
    f"train: {list_path.resolve().as_posix()}\n"
    f"val: images/val\ntest: images/test\nnames:\n  0: gun\n")

print(f"oi {len(buckets['oi'])}->{len(keep_oi)}  cctv {len(buckets['cctv'])}->{len(keep_cctv)}  "
            f"r46 {len(buckets['r46'])}  (pistol dropped {len(buckets['pistol'])})")
print(f"NEW TRAIN TOTAL: {len(kept)}  -> {yaml_path.name}")