"""Convert UGR WeaponS (Pascal VOC) -> YOLO single-class 'gun'.
Reads WeaponS/ (images) + WeaponS_bbox/ (xml); writes images/ + labels/ + ugr.yaml.
Read-only on source."""
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "benchmarks" / "ugr_pistol"
IMG_SRC = SRC / "WeaponS"
XML_SRC = SRC / "WeaponS_bbox"
OUT_IMG = SRC / "images"
OUT_LBL = SRC / "labels"
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)
EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp")

done = miss = boxes = nobox = 0
for xf in sorted(XML_SRC.glob("*.xml")):
    if xf.name.startswith("._"):
        continue
    img = next((IMG_SRC / (xf.stem + e) for e in EXTS if (IMG_SRC / (xf.stem + e)).exists()), None)
    if img is None:
        miss += 1
        continue
    r = ET.parse(xf).getroot()
    W = float(r.findtext("size/width") or 0)
    H = float(r.findtext("size/height") or 0)
    if not W or not H:
        from PIL import Image
        W, H = Image.open(img).size
    lines = []
    for obj in r.findall("object"):
        b = obj.find("bndbox")
        if b is None:
            continue
        x1, y1 = float(b.findtext("xmin")), float(b.findtext("ymin"))
        x2, y2 = float(b.findtext("xmax")), float(b.findtext("ymax"))
        cx, cy = ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
        bw, bh = min(bw, 1), min(bh, 1)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    if not lines:
        nobox += 1
        continue
    shutil.copy2(img, OUT_IMG / img.name)
    (OUT_LBL / (xf.stem + ".txt")).write_text("\n".join(lines) + "\n")
    done += 1
    boxes += len(lines)

(SRC / "ugr.yaml").write_text(
    f"path: {SRC.resolve().as_posix()}\ntrain: images\nval: images\nnames:\n  0: gun\n")
print(f"converted {done} imgs / {boxes} boxes; missing_img {miss}; no_box {nobox}")
print(f"yaml: {SRC / 'ugr.yaml'}")
