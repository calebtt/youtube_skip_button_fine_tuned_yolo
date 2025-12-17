from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_positive_label(txt_path: Path) -> bool:
    if not txt_path.exists():
        return False
    s = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    return len(s) > 0


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    # simulate compression artifacts; keep RGB
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def add_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return img
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def augment_one(img: Image.Image, rng: random.Random) -> Image.Image:
    # Ensure RGB
    img = img.convert("RGB")

    # Brightness/contrast/saturation
    if rng.random() < 0.95:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.65, 1.45))
    if rng.random() < 0.95:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.65, 1.45))
    if rng.random() < 0.90:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.55, 1.60))

    # Gamma (implemented via brightness + contrast-ish approximation)
    # (PIL doesn't have direct gamma; this is a lightweight stand-in)
    if rng.random() < 0.70:
        gamma = rng.uniform(0.80, 1.25)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip(arr ** (1.0 / gamma), 0, 1)
        img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")

    # Slight blur (simulates motion/compression/resize softness)
    if rng.random() < 0.60:
        radius = rng.uniform(0.0, 1.2)
        if radius > 0.05:
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Noise (simulates codec + sensor)
    if rng.random() < 0.60:
        sigma = rng.uniform(0.0, 6.0)  # pixels in 0..255 space
        img = add_gaussian_noise(img, sigma)

    # JPEG artifacts (very useful for screen captures)
    if rng.random() < 0.75:
        q = rng.randint(35, 95)
        img = jpeg_roundtrip(img, q)

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="train/images", help="Input images folder")
    ap.add_argument("--labels", default="train/labels", help="Input labels folder (YOLO txt)")
    ap.add_argument("--out-images", default="train/images", help="Output images folder")
    ap.add_argument("--out-labels", default="train/labels", help="Output labels folder")
    ap.add_argument("--copies", type=int, default=15, help="Synthetic copies per positive image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="syn", help="Filename prefix for synthetic images")
    args = ap.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out_images = Path(args.out_images)
    out_labels = Path(args.out_labels)

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    img_paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    positives = []
    for img_path in img_paths:
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if is_positive_label(lbl_path):
            positives.append((img_path, lbl_path))

    if not positives:
        print(f"No positives found. Checked labels in: {labels_dir}")
        return

    created = 0
    for img_path, lbl_path in positives:
        base_lbl = lbl_path.read_text(encoding="utf-8", errors="ignore")

        with Image.open(img_path) as im:
            for i in range(args.copies):
                aug = augment_one(im, rng)

                # make unique name
                tag = f"{args.prefix}_{img_path.stem}_{i:03d}_{rng.randrange(10**9):09d}"
                out_img_path = out_images / (tag + ".jpg")   # save as jpg (good for artifact variety)
                out_lbl_path = out_labels / (tag + ".txt")

                if out_img_path.exists() or out_lbl_path.exists():
                    continue

                aug.save(out_img_path, quality=rng.randint(70, 95), optimize=True)
                out_lbl_path.write_text(base_lbl, encoding="utf-8")
                created += 1

    print(f"Positives found: {len(positives)}")
    print(f"Created synthetic positives: {created}")
    print(f"Output images: {out_images}")
    print(f"Output labels: {out_labels}")


if __name__ == "__main__":
    main()
