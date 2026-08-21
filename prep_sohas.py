"""Build a SOHAS benchmark for the single-class 'gun' detector.

SOHAS (Perez-Hernandez et al. 2020) has 6 classes; class 0 = pistol. We keep the
pistol boxes as class 0 ('gun') and DROP the rest (smartphone/knife/purse/bill/
card) -> those images become no-gun backgrounds. So the benchmark measures BOTH
recall/precision on real pistols AND false positives on gun-like handled objects
(the smartphone-in-hand case) in one set. Deduped vs the training set (dHash).

Output: benchmarks/sohas/{images,labels}/ + sohas_clean.txt (eval with:
  python eval_ugr.py --list benchmarks/sohas/sohas_clean.txt)
"""
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOHAS = (ROOT / "benchmarks" / "OD-WeaponDetection" /
         "Weapons and similar handled objects" /
         "Sohas_weapon-Detection-YOLOv5" / "obj_train_data")
TRAIN = ROOT / "gun_dataset" / "images" / "train"
OUT = ROOT / "benchmarks" / "sohas"
OUT_IMG, OUT_LBL = OUT / "images", OUT / "labels"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
THRESH = 8

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
def ham(a, b): return bin(a ^ b).count("1")

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

print("hashing train (dedup guard)...")
train_h = [h for p in TRAIN.iterdir() if p.suffix.lower() in IMG_EXTS
           for h in [dhash(p)] if h is not None]
print(f"train hashed: {len(train_h)}")

lbl_files = list((SOHAS / "labels" / "train").glob("*.txt")) + \
            list((SOHAS / "labels" / "test").glob("*.txt"))
print(f"SOHAS label files: {len(lbl_files)}")

kept_paths, pistol_imgs, bg_imgs, dropped, miss = [], 0, 0, 0, 0
for lf in lbl_files:
    sub = lf.parent.name  # train / test
    img = next((SOHAS / "images" / sub / (lf.stem + e) for e in IMG_EXTS
                if (SOHAS / "images" / sub / (lf.stem + e)).exists()), None)
    if img is None:
        miss += 1; continue
    h = dhash(img)
    if h is not None and any(ham(h, t) <= THRESH for t in train_h):
        dropped += 1; continue
    gun_lines = [ln for ln in lf.read_text().splitlines()
                 if ln.split() and ln.split()[0] == "0"]   # class 0 == pistol == gun
    dst_img = OUT_IMG / img.name
    shutil.copy2(img, dst_img)
    (OUT_LBL / (lf.stem + ".txt")).write_text("\n".join(gun_lines) + ("\n" if gun_lines else ""))
    kept_paths.append(str(dst_img.resolve()))
    if gun_lines: pistol_imgs += 1
    else:         bg_imgs += 1

(OUT / "sohas_clean.txt").write_text("\n".join(kept_paths) + "\n")
(OUT / "sohas.yaml").write_text(
    f"path: {OUT.resolve().as_posix()}\ntrain: {(OUT/'sohas_clean.txt').resolve().as_posix()}\n"
    f"val: {(OUT/'sohas_clean.txt').resolve().as_posix()}\nnames:\n  0: gun\n")

print(f"\ndropped as train-overlap (dHash<={THRESH}): {dropped}   missing img: {miss}")
print(f"KEPT {len(kept_paths)} images:  {pistol_imgs} with pistol (positives), {bg_imgs} no-gun (FP test)")
print(f"-> {OUT/'sohas_clean.txt'}  |  eval: python eval_ugr.py --list benchmarks/sohas/sohas_clean.txt")
