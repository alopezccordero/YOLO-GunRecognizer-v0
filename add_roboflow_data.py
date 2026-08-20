from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

import yaml
from roboflow import Roboflow


REPO_ROOT = Path(__file__).resolve().parent
DEST = REPO_ROOT / "gun_dataset"

# Class names (from the Roboflow data.yaml) that count as a firearm. Matching
# is case-insensitive and substring-based, so "pistol" also matches "Pistol"
# and "hand_pistol". Extend with --firearm-classes if a dataset uses odd names.
FIREARM_KEYWORDS = {
    "gun", "guns", "firearm", "pistol", "handgun", "revolver", "rifle",
    "shotgun", "smg", "submachine", "machine_gun", "machinegun", "sniper",
    "carbine", "weapon",
}

# Roboflow split dir -> our split dir
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def is_firearm(name: str, extra: set[str]) -> bool:
    n = name.strip().lower()
    if n in extra:
        return True
    return any(kw in n for kw in FIREARM_KEYWORDS)


def load_class_names(data_yaml: Path) -> list[str]:
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):  # {0: 'a', 1: 'b'} -> ordered list
        return [names[i] for i in sorted(names)]
    return list(names)


def rewrite_label(src: Path, keep_ids: set[int]) -> list[str]:
    """Return YOLO label lines for firearm boxes only, class id forced to 0."""
    out: list[str] = []
    for line in src.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        if int(parts[0]) in keep_ids:
            out.append(" ".join(["0", *parts[1:]]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--fmt", default="yolov8",
                    help="Roboflow export format (yolov8 / yolov11).")
    ap.add_argument("--prefix", default=None,
                    help="Filename prefix to avoid collisions "
                         "(default: '<project>_').")
    ap.add_argument("--firearm-classes", nargs="*", default=[],
                    help="Extra exact class names to treat as firearms.")
    ap.add_argument("--all-to-train", action="store_true",
                    help="every source split into train")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be merged, copy nothing.")
    ap.add_argument("--force-download", action="store_true",
                    help="Re-download even if a local copy already exists.")
    args = ap.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY in the environment first.")
    if not DEST.exists():
        raise SystemExit(f"Destination {DEST} not found. Build the base "
                         f"dataset first (prepare_dataset.py).")

    prefix = args.prefix or f"{args.project}_"
    extra = {c.strip().lower() for c in args.firearm_classes}

    # Download into a stable folder inside the project (NOT a throwaway temp
    # dir). The Roboflow SDK caches by version and will skip re-downloading if
    # it thinks the version already exists; a persistent location means the
    # files survive between runs so we can still find them (and inspect them).
    dl_dir = REPO_ROOT / "_roboflow_dl" / f"{args.project}-v{args.version}"
    dl_dir.mkdir(parents=True, exist_ok=True)

    # Reuse a prior download if present; only hit the API when needed (or when
    # --force-download is passed). Avoids re-pulling thousands of images.
    candidates = list(dl_dir.rglob("data.yaml"))
    if candidates and not args.force_download:
        print(f"Reusing existing download in {dl_dir} "
              f"(pass --force-download to re-fetch).")
    else:
        print(f"Downloading {args.workspace}/{args.project}:v{args.version} "
              f"into {dl_dir} ...")
        rf = Roboflow(api_key=api_key)
        ds = (rf.workspace(args.workspace)
                .project(args.project)
                .version(args.version)
                .download(args.fmt, location=str(dl_dir), overwrite=True))
        candidates = list(dl_dir.rglob("data.yaml"))
        if not candidates and Path(ds.location).exists():
            candidates = list(Path(ds.location).rglob("data.yaml"))
    if not candidates:
        contents = sorted(p.name for p in dl_dir.iterdir()) or ["(empty)"]
        raise SystemExit(
            f"data.yaml not found. SDK reported location: {ds.location}\n"
            f"Contents of {dl_dir}: {contents}\n"
            f"The download likely failed (auth/format/version). Verify the "
            f"workspace/project/version and that the '{args.fmt}' export "
            f"exists for this version on Roboflow.")
    data_yaml = candidates[0]
    root = data_yaml.parent
    print(f"Dataset root: {root}")

    names = load_class_names(data_yaml)
    keep_ids = {i for i, n in enumerate(names) if is_firearm(n, extra)}
    firearm_names = [names[i] for i in sorted(keep_ids)]
    dropped_names = [n for i, n in enumerate(names) if i not in keep_ids]
    print(f"Classes: {names}")
    print(f"  kept as gun : {firearm_names}")
    print(f"  dropped     : {dropped_names or '(none)'}")
    if not keep_ids:
        raise SystemExit("No firearm classes matched. Pass the correct "
                         "names via --firearm-classes.")

    totals = {"images": 0, "boxes": 0, "skipped": 0}
    split_map = {k: "train" for k in SPLIT_MAP} if args.all_to_train else SPLIT_MAP

    for rf_split, our_split in split_map.items():
        lbl_dir = root / rf_split / "labels"
        img_dir = root / rf_split / "images"
        if not lbl_dir.is_dir():
            continue

        out_img = DEST / "images" / our_split
        out_lbl = DEST / "labels" / our_split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        added = boxes = skipped = 0
        for lbl in sorted(lbl_dir.glob("*.txt")):
            img = next((img_dir / (lbl.stem + e) for e in IMG_EXTS
                        if (img_dir / (lbl.stem + e)).exists()), None)
            if img is None:
                continue
            lines = rewrite_label(lbl, keep_ids)
            if not lines:  # no firearm in this image -> skip it
                skipped += 1
                continue
            if not args.dry_run:
                (out_lbl / f"{prefix}{lbl.name}").write_text(
                    "\n".join(lines) + "\n")
                shutil.copy2(img, out_img / f"{prefix}{img.name}")
            added += 1
            boxes += len(lines)

        print(f"  {our_split:5s}: +{added} images, {boxes} boxes, "
              f"{skipped} skipped (no firearm)")
        totals["images"] += added
        totals["boxes"] += boxes
        totals["skipped"] += skipped

    tag = "[DRY RUN] would add" if args.dry_run else "Added"
    print(f"\n{tag} {totals['images']} images / {totals['boxes']} boxes "
          f"(skipped {totals['skipped']} firearm-free images).")
    print(f"Dataset root: {DEST}")


if __name__ == "__main__":
    main()
