import random
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DEST = ROOT / "gun_dataset"
TRAIN_IMG = DEST / "images" / "train"
KEEP_FRAC = 1/3
SEED = 42
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

imgs = [p for p in sorted(TRAIN_IMG.iterdir()) if p.suffix.lower() in IMG_EXTS]
cctv = [p for p in imgs if p.name.startswith("cctv_")]
other = [p for p in imgs if not p.name.startswith("cctv_")
         and not p.name.startswith("pistol-csvic_")]

random.seed(SEED)
keep_cctv = random.sample(cctv, round(len(cctv) * KEEP_FRAC))
kept = sorted(other + keep_cctv, key=lambda p: p.name)

list_path = DEST / "train_downsample_cctv.txt"
list_path.write_text("\n".join(p.resolve().as_posix() for p in kept) + "\n")

yaml_path = DEST / "dataset_downsample_cctv.yaml"
yaml_path.write_text(
    f"path: {DEST.resolve().as_posix()}\n"
    f"train: {list_path.resolve().as_posix()}\n"
    f"val: images/val\n"
    f"test: images/test\n"
    f"names:\n  0: gun\n"
)

print(f"train imgs on disk: {len(imgs)}  (cctv {len(cctv)}, other {len(other)})")
print(f"kept cctv (1/3): {len(keep_cctv)}  ->  NEW TRAIN TOTAL: {len(kept)}")
print(f"yaml: {yaml_path}")