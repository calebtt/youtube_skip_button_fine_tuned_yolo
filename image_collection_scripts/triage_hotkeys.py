import argparse
import shutil
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder containing images to triage (non-recursive)")
    ap.add_argument("--out", required=True, help="Output folder (will create triage/ad and triage/not_ad)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out_ad = out / "triage" / "ad"
    out_not = out / "triage" / "not_ad"
    out_ad.mkdir(parents=True, exist_ok=True)
    out_not.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in sorted(src.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not imgs:
        print(f"No images found in: {src}")
        return

    print("Keys: A=ad, N=not_ad, S=skip, Q=quit")
    i = 0
    while i < len(imgs):
        p = imgs[i]
        im = cv2.imread(str(p))
        if im is None:
            i += 1
            continue

        cv2.imshow("triage", im)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            i += 1
            continue
        elif key in (ord('a'), ord('A')):
            shutil.move(str(p), str(out_ad / p.name))
        elif key in (ord('n'), ord('N')):
            shutil.move(str(p), str(out_not / p.name))
        else:
            continue

        i += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
