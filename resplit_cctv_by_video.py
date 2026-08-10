"""Re-split the CCTV (cctv_*) frames in gun_dataset/ by VIDEO, so that all
frames from one source video live in a single split (train/val/test).

Consecutive video frames are near-identical; a random per-frame split leaks
near-duplicates across train and val/test and inflates metrics. Assigning whole
videos to one split each removes that leakage. Open Images (non-cctv) files are
left untouched -- those are independent photos with no such correlation.

Dry-run by default. Pass --apply to actually move files.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent
DEST = REPO_ROOT / "gun_dataset"
SPLITS = ["train", "val", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Whole-video -> split assignment (~80/10/10 by frame count). Every distinct
# cctv video id must appear exactly once here.
VIDEO_SPLIT = {
    # train (big sources kept here)
    "mgd_custom": "train",
    "mock_attack": "train",
    "mgd_varying": "train",
    "gmd_2": "train",
    "gmd_3": "train",
    # val
    "gmd_1": "val",
    "gmd_6": "val",
    "gmd_7": "val",
    # test (includes the in-the-wild youtube source)
    "youtube": "test",
    "gmd_4": "test",
    "gmd_5": "test",
}


def video_id(name: str) -> str | None:
    """Extract the source-video id from a cctv_ filename, else None."""
    if not name.startswith("cctv_"):
        return None
    s = name[len("cctv_"):]
    return re.split(r"_frame\d+", s, maxsplit=1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Actually move files (default: dry-run).")
    args = ap.parse_args()

    # Gather every cctv image and its current split.
    items = []  # (video_id, current_split, image_path)
    seen_videos: set[str] = set()
    for sp in SPLITS:
        img_dir = DEST / "images" / sp
        if not img_dir.is_dir():
            continue
        for p in img_dir.iterdir():
            vid = video_id(p.name)
            if vid is None:
                continue
            seen_videos.add(vid)
            items.append((vid, sp, p))

    unknown = seen_videos - set(VIDEO_SPLIT)
    if unknown:
        raise SystemExit(f"Unmapped cctv videos: {sorted(unknown)}. "
                         f"Add them to VIDEO_SPLIT.")

    moves = defaultdict(int)          # (from,to) -> count
    final = defaultdict(int)          # target split -> count
    moved = 0
    for vid, cur, img in items:
        tgt = VIDEO_SPLIT[vid]
        final[tgt] += 1
        if tgt == cur:
            continue
        moves[(cur, tgt)] += 1
        moved += 1
        if not args.apply:
            continue
        lbl = DEST / "labels" / cur / (img.stem + ".txt")
        (DEST / "images" / tgt).mkdir(parents=True, exist_ok=True)
        (DEST / "labels" / tgt).mkdir(parents=True, exist_ok=True)
        shutil.move(str(img), str(DEST / "images" / tgt / img.name))
        if lbl.exists():
            shutil.move(str(lbl), str(DEST / "labels" / tgt / lbl.name))

    mode = "APPLIED" if args.apply else "DRY-RUN (no files moved)"
    print(f"=== {mode} ===")
    print(f"cctv frames total: {len(items)}, videos: {len(seen_videos)}")
    print("\nmoves:")
    for (a, b), c in sorted(moves.items()):
        print(f"  {a:5s} -> {b:5s}: {c}")
    print(f"  (unchanged: {len(items) - moved})")
    print("\ncctv frames per split after re-split:")
    for sp in SPLITS:
        print(f"  {sp:5s}: {final[sp]}")
    if not args.apply:
        print("\nRe-run with --apply to move the files.")


if __name__ == "__main__":
    main()
