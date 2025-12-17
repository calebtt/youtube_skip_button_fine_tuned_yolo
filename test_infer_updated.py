from __future__ import annotations

import argparse
import random
import time
import shutil
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def iter_images(root: Path):
    if root.is_file() and root.suffix.lower() in IMG_EXTS:
        yield root
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def chunked(lst, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def try_cuda_cleanup():
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Batched YOLO inference + prints paths for detected images.")
    ap.add_argument("--weights", required=True, help="Path to weights (e.g. runs/detect/train2/weights/best.pt)")
    ap.add_argument("--source", required=True, help="Folder (recursive) or single image file")
    ap.add_argument("--conf", type=float, default=0.60)
    ap.add_argument("--iou", type=float, default=0.70)
    ap.add_argument("--imgsz", type=int, default=1088)  # stride-safe for max stride 32
    ap.add_argument("--max-det", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1, help="Inference chunk size (lower if OOM)")
    ap.add_argument("--device", default="0", help='e.g. "0" for GPU0, "cpu" for CPU')
    ap.add_argument("--half", action="store_true", help="Use FP16 on GPU")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of images")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before limiting")
    ap.add_argument("--print-conf", action="store_true", help="Print max confidence alongside each detected image path")
    ap.add_argument("--detected-dir", default="detected", help="Folder to copy images with detections into")
    args = ap.parse_args()

    weights = Path(args.weights)
    src = Path(args.source)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    paths = list(iter_images(src))
    if not paths:
        print(f"No images found under: {src}")
        return

    if args.shuffle:
        random.shuffle(paths)
    if args.limit is not None:
        paths = paths[: max(1, int(args.limit))]

    print("\n--- Batched inference ---")
    print(f"Weights: {weights}")
    print(f"Source:  {src}")
    print(f"Images:  {len(paths)}")
    print(f"conf={args.conf} iou={args.iou} imgsz={args.imgsz} max_det={args.max_det} batch={args.batch} device={args.device} half={args.half}")

    try_cuda_cleanup()
    model = YOLO(str(weights))

    t0 = time.time()
    n_with_det = 0
    total_det = 0
    conf_sum = 0.0
    conf_count = 0

    detected_dir = Path(args.detected_dir)
    detected_dir.mkdir(parents=True, exist_ok=True)

    # Print detected images as we go
    for batch_paths in chunked(paths, max(1, int(args.batch))):
        results = model.predict(
            source=[str(p) for p in batch_paths],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            stream=True,
            verbose=False,
            device=args.device,
            half=args.half,
            batch=max(1, int(args.batch)),
        )

        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue

            n = len(boxes)
            if n > 0:
                n_with_det += 1
                total_det += n

                # Print the image path for any detection
                img_path = getattr(r, "path", None)
                if img_path is None:
                    # fallback (rare)
                    img_path = "<unknown>"

                if args.print_conf:
                    try:
                        max_conf = float(boxes.conf.max().detach().cpu().item())
                        print(f"DETECTED {max_conf:.3f} {img_path}")
                    except Exception:
                        print(f"DETECTED {img_path}")
                else:
                    print(f"DETECTED {img_path}")

                # Copy detected image into --detected-dir
                try:
                    img_p = Path(str(img_path))
                    if img_p.exists():
                        if src.is_dir():
                            try:
                                rel = img_p.resolve().relative_to(src.resolve())
                                dst = detected_dir / rel
                            except Exception:
                                dst = detected_dir / img_p.name
                        else:
                            dst = detected_dir / img_p.name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(img_p, dst)
                except Exception as e:
                    print(f"WARN: failed to copy detected image '{img_path}' to '{detected_dir}': {e}")

                # Stats
                try:
                    confs = boxes.conf.detach().cpu().tolist()
                    conf_sum += sum(confs)
                    conf_count += len(confs)
                except Exception:
                    pass

    dt = time.time() - t0
    det_rate = n_with_det / len(paths)
    det_per_100 = det_rate * 100.0
    avg_conf = (conf_sum / conf_count) if conf_count else 0.0

    print("\nResults:")
    print(f"  Images w/ >=1 detection: {n_with_det}/{len(paths)} ({det_per_100:.2f} per 100 images)")
    print(f"  Total detections:        {total_det}")
    print(f"  Avg detection conf:      {avg_conf:.3f}")
    print(f"  Time:                   {dt:.2f}s ({dt/len(paths):.4f}s/image)\n")


if __name__ == "__main__":
    main()
