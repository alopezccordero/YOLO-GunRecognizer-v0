import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "gun_dataset"
TRAIN_IMG = DEST / "images" / "train"
STAGED_FRAC, SEED = 1/3, 42
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def kind(name):
    if name.startswith("cctv-gun-detector_"): return "diverse"   # NEW — keep ALL
    if name.startswith("cctv_"):              return "staged"    # gmd/mgd — downsample
    if name.startswith("r46_"):               return "r46"       # keep
    if name.startswith("pistol-csvic_"):      return "pistol"    # exclude
    return "oi"                                                   # keep

buckets = {}
for p in sorted(TRAIN_IMG.iterdir()):
    if p.suffix.lower() in IMG_EXTS:
        buckets.setdefault(kind(p.name), []).append(p)

random.seed(SEED)
staged = buckets.get("staged", [])
keep_staged = random.sample(staged, round(len(staged) * STAGED_FRAC))
kept = sorted(buckets.get("oi", []) + buckets.get("r46", []) +
              buckets.get("diverse", []) + keep_staged, key=lambda p: p.name)

list_path = DEST / "train_v2.txt"
list_path.write_text("\n".join(p.resolve().as_posix() for p in kept) + "\n")
(DEST / "dataset_v2.yaml").write_text(
    f"path: {DEST.resolve().as_posix()}\ntrain: {list_path.resolve().as_posix()}\n"
    f"val: images/val\ntest: images/test\nnames:\n  0: gun\n")

print({k: len(v) for k, v in buckets.items()})
print(f"kept staged (1/3): {len(keep_staged)}   NEW TRAIN TOTAL: {len(kept)}")