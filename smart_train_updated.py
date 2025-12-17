from __future__ import annotations

import argparse
import sys
from multiprocessing import freeze_support
from pathlib import Path
import time

from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def find_latest_best_pt(runs_root: Path = Path("runs")) -> Path | None:
    """
    Find the most recently modified Ultralytics 'best.pt' under runs/**/weights/best.pt.
    Returns None if no such file exists.
    """
    if not runs_root.exists():
        return None

    best_candidates = list(runs_root.rglob("weights/best.pt"))
    if not best_candidates:
        return None

    # Pick newest by modification time
    best_candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return best_candidates[0]


def choose_start_model(base_model: str, resume_best_mode: str) -> str:
    """
    Decide which weights to start training from:
      - base_model (default)
      - latest discovered best.pt (if mode is yes/ask and user confirms)

    resume_best_mode: 'ask' | 'yes' | 'no'
    """
    latest_best = find_latest_best_pt()
    if latest_best is None or not latest_best.exists():
        return base_model

    if resume_best_mode == "no":
        return base_model

    if resume_best_mode == "yes":
        print(f"Using previous best weights: {latest_best}")
        return str(latest_best)

    # ask (interactive)
    if not sys.stdin.isatty():
        # Non-interactive shell (e.g., CI) -> do not prompt
        return base_model

    resp = input(f"Found previous best weights at '{latest_best}'. Resume from this best? [y/N]: ").strip().lower()
    if resp in {"y", "yes"}:
        print(f"Resuming from previous best weights: {latest_best}")
        return str(latest_best)

    return base_model


def iter_images(root: Path):
    if root.is_file() and root.suffix.lower() in IMG_EXTS:
        yield root
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def run_test_inference(weights_path: Path, test_root: Path, conf: float, iou: float, imgsz: int, max_det: int):
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")
    if not test_root.exists():
        raise FileNotFoundError(f"test folder not found: {test_root}")

    paths = list(iter_images(test_root))
    if not paths:
        print(f"No images found in: {test_root}")
        return

    print(f"\n--- Test inference ---")
    print(f"Weights: {weights_path}")
    print(f"Test images: {len(paths)} from {test_root}")
    print(f"conf={conf} iou={iou} imgsz={imgsz} max_det={max_det}")

    model = YOLO(str(weights_path))

    t0 = time.time()
    n_with_det = 0
    total_det = 0
    conf_sum = 0.0
    conf_count = 0

    # stream=True avoids holding all results in memory
    for r in model.predict(
        source=[str(p) for p in paths],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        stream=True,
        verbose=False,
    ):
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        n = len(boxes)
        if n > 0:
            n_with_det += 1
            total_det += n
            # Average confidence of detections
            try:
                confs = boxes.conf.detach().cpu().tolist()
                conf_sum += sum(confs)
                conf_count += len(confs)
            except Exception:
                pass

    dt = time.time() - t0
    det_rate = (n_with_det / len(paths)) if paths else 0.0
    det_per_100 = det_rate * 100.0
    avg_conf = (conf_sum / conf_count) if conf_count else 0.0

    print(f"\nResults:")
    print(f"  Images with >=1 detection: {n_with_det}/{len(paths)} ({det_per_100:.2f} per 100 images)")
    print(f"  Total detections:          {total_det}")
    print(f"  Avg detection confidence:  {avg_conf:.3f}")
    print(f"  Time: {dt:.2f}s  ({(dt/len(paths)):.4f}s/image)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--save-period", type=int, default=5)
    ap.add_argument("--model", default="yolov11n.pt", help="Base model or checkpoint to start training from")
    ap.add_argument(
        "--resume-best",
        choices=["ask", "yes", "no"],
        default="ask",
        help="When training, optionally resume from the most recent runs/**/weights/best.pt (ask|yes|no)",
    )
    ap.add_argument("--workers", type=int, default=8)

    # Optional test pass (usually point this to negatives-only screenshots)
    ap.add_argument("--test", default=None, help="Folder of images to run inference on after training")
    ap.add_argument("--test-conf", type=float, default=0.60)
    ap.add_argument("--test-iou", type=float, default=0.70)
    ap.add_argument("--test-max-det", type=int, default=20)

    # If you ever want “just test” without training
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--weights", default=None, help="Weights to test when using --skip-train (defaults to runs/*/best.pt if omitted)")
    args = ap.parse_args()

    run_dir = None
    best_pt = None

    if not args.skip_train:
        start_model = choose_start_model(args.model, args.resume_best)
        model = YOLO(start_model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            save_period=args.save_period,
            workers=args.workers,
            plots=False,
            verbose=True,
        )

        save_dir = getattr(results, "save_dir", None)
        if save_dir is None:
            save_dir = getattr(getattr(model, "trainer", None), "save_dir", None) or "runs/detect/train"
        run_dir = Path(save_dir)
        best_pt = run_dir / "weights" / "best.pt"

        print(f"\nRun directory: {run_dir}")
        print(f"best.pt: {best_pt}  (exists={best_pt.exists()})")

    # Test inference using best.pt (or provided weights)
    if args.test:
        if args.skip_train:
            if args.weights:
                best_pt = Path(args.weights)
            else:
                raise ValueError("When using --skip-train, pass --weights <path-to-best.pt>")

        run_test_inference(
            weights_path=best_pt,
            test_root=Path(args.test),
            conf=args.test_conf,
            iou=args.test_iou,
            imgsz=args.imgsz,
            max_det=args.test_max_det,
        )


if __name__ == "__main__":
    freeze_support()
    main()
