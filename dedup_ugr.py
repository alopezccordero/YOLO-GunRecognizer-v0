"""Dedup UGR benchmark vs gun_dataset TRAIN (dHash). Writes ugr_clean.txt = UGR
images with NO near-duplicate in train (so the benchmark is a clean test of the
champion). Read-only except the list file."""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
UGR = ROOT / "benchmarks" / "ugr_pistol" / "images"
TRAIN = ROOT / "gun_dataset" / "images" / "train"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
THRESH = 8

def dhash(path, size=8):
    try:
        img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return None
    px = list(img.getdata()); bits = 0
    for r in range(size):
        row = r * (size + 1)
        for c in range(size):
            bits = (bits << 1) | (1 if px[row + c] > px[row + c + 1] else 0)
    return bits

def ham(a, b): return bin(a ^ b).count("1")

print("hashing train...")
train_h = []
for p in TRAIN.iterdir():
    if p.suffix.lower() in IMG_EXTS:
        h = dhash(p)
        if h is not None: train_h.append(h)
print(f"train hashed: {len(train_h)}")

print("hashing UGR + comparing...")
ugr = [p for p in sorted(UGR.iterdir()) if p.suffix.lower() in IMG_EXTS]
clean, overlap = [], 0
examples = []
for p in ugr:
    h = dhash(p)
    if h is None:
        continue
    bd = min(ham(h, th) for th in train_h)
    if bd <= THRESH:
        overlap += 1
        if len(examples) < 10:
            examples.append((bd, p.name))
    else:
        clean.append(str(p.resolve()))

out = ROOT / "benchmarks" / "ugr_pistol" / "ugr_clean.txt"
out.write_text("\n".join(clean) + "\n")
print(f"\nUGR total: {len(ugr)}")
print(f"overlap with train (<= {THRESH}): {overlap}")
print(f"clean benchmark images: {len(clean)}  -> {out.name}")
for d, n in examples:
    print(f"   dropped d={d}  {n}")
