##to verify if there is leakage. DHASH is used
#each image becomes a 64 bit figerprint, two images are near duplicate if their fingerplits differ in <= 8bits
#duplicates score 0

from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "_roboflow_dl" / "cctv-gun-detector-v1" # source from roboflow_dl
DEST = ROOT / "gun_dataset"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
THRESH = 8

def dhash(path, size=8):
    try:
        img = Image.open(path).convert("L").resize((size+1, size), Image.LANCZOS)
    except:
        return None
    px = list(img.getdata()); bits = 0
    for r in range(size):
        row = r * (size + 1)
        for c in range(size): 
            bits = (bits << 1) | (1 if px[row + c] > px[row + c + 1] else 0)
    return bits

def ham(a, b): return bin(a ^ b).count("1")

held = []
for split in ["test", "val"]:
    d = DEST / "images" / split
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            h = dhash(p)
            if h is not None:
                held.append((f"{split}/{p.name}", h))
print(f"held-out images fingerprinted: {len(held)}") #how many images were held

matches, checked = [], 0
for sp in ["train", "valid", "test"]:
    img_dir = SRC / sp / "images"
    if not img_dir.is_dir(): continue
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS: continue
        h = dhash(p)
        if h is None: continue
        checked += 1
        best_name, best_d = None, 999
        for name, hh in held:
            d = ham(h, hh)
            if d < best_d:
                best_d, best_name = d, name
        if best_d <= THRESH:
            matches.append((best_d, f"{sp}/{p.name}", best_name))

print(f"source frames checked: {checked}")
print(f"\n=== near_duplicates (Hamming <= {THRESH}): {len(matches)} ===")


for d, src, held_name, in sorted(matches):
    print(f"    dist={d:2d} SRC {src}   <->    HELD {held_name}")

if not matches:
    print(" none - dataset is clean vs test/val, save to merge all-to-train")

# Generate a shell script to delete duplicate source images
DELETE_SCRIPT = ROOT / "delete_roboflow_duplicates.sh"

with open(DELETE_SCRIPT, "w", newline="\n") as f:
    f.write("#!/bin/bash\n")
    f.write("set -e\n\n")
    f.write("# Auto-generated script to remove near-duplicate images\n")
    f.write(f"# dHash threshold: {THRESH}\n")
    f.write(f"# Number of duplicates: {len(matches)}\n\n")

    for d, src, held_name in sorted(matches):
        # src looks like: train/image.jpg
        sp, filename = src.split("/", 1)

        image_path = SRC / sp / "images" / filename

        # Roboflow YOLO label with same stem
        label_path = SRC / sp / "labels" / f"{Path(filename).stem}.txt"

        f.write(
            f'# dist={d} | {src} <-> {held_name}\n'
        )
        f.write(
            f'rm -f "{image_path}"\n'
        )
        f.write(
            f'rm -f "{label_path}"\n\n'
        )

print(f"\nDeletion script written to:")
print(DELETE_SCRIPT)
print("\nReview it before running it.")