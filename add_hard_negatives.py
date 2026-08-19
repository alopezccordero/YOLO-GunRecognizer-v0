import argparse, random, shutil
from pathlib import Path
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "gun_dataset"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
FIREARM = {"gun","guns","firearm","pistol","handgun","revolver","rifle","shotgun",
           "weapon","smg","carbine"}

def is_gun(name): return any(k in name.strip().lower() for k in FIREARM)

def dhash(path, size=8):
    try: im = Image.open(path).convert("L").resize((size+1, size), Image.LANCZOS)
    except Exception: return None
    px = list(im.getdata()); bits = 0
    for r in range(size):
        row = r*(size+1)
        for c in range(size):
            bits = (bits<<1) | (1 if px[row+c] > px[row+c+1] else 0)
    return bits
def ham(a,b): return bin(a^b).count("1")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--prefix", default="hn_")
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedup", action="store_true", help="skip test duplicates")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dy = next(src.rglob("data.yaml"), None)
    gun_ids = set()
    if dy:
        names = yaml.safe_load(dy.read_text()).get("names", [])
        names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        gun_ids = {i for i, n in enumerate(names) if is_gun(str(n))}
        print(f"classes: {names}   gun class ids (excluded): {sorted(gun_ids)}")

    # candidate images = those with NO gun box
    cands = []
    for img in src.rglob("*"):
        if img.suffix.lower() not in IMG_EXTS: continue
        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
        has_gun = False
        if lbl.exists() and gun_ids:
            for line in lbl.read_text().splitlines():
                p = line.split()
                if p and p[0].isdigit() and int(p[0]) in gun_ids:
                    has_gun = True; break
        if not has_gun:
            cands.append(img)
    print(f"gun-free candidates: {len(cands)} / picking {min(args.max, len(cands))}")

    random.seed(args.seed)
    picked = random.sample(cands, min(args.max, len(cands)))

    test_h = []
    if args.dedup:
        for p in (DEST/"images"/"test").iterdir():
            if p.suffix.lower() in IMG_EXTS:
                h = dhash(p)
                if h is not None: test_h.append(h)

    out_img = DEST/"images"/"train"; out_lbl = DEST/"labels"/"train"
    added = dupe = 0
    for p in picked:
        if args.dedup:
            h = dhash(p)
            if h is not None and any(ham(h, th) <= 8 for th in test_h):
                dupe += 1; continue
        if not args.dry_run:
            shutil.copy2(p, out_img / f"{args.prefix}{p.name}")
            (out_lbl / f"{args.prefix}{p.stem}.txt").write_text("")   # empty = background
        added += 1
    tag = "[DRY RUN] would add" if args.dry_run else "Added"
    print(f"{tag} {added} hard negatives; skipped {dupe} test-dupes")

if __name__ == "__main__":
    main()