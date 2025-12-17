#!/usr/bin/env python3
"""
YOLO GUI tester (generic, Ultralytics)

Purpose
- Select a fine-tuned YOLO .pt model.
- Select one or more images (or a folder).
- Run detection, display counts, and save annotated outputs to a chosen folder.

Dependencies
  pip install -U ultralytics pillow
"""

import threading
import queue
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from PIL import Image, ImageTk


@dataclass
class DetSummary:
    image_path: str
    num_dets: int
    classes: str
    saved_path: Optional[str]
    best_conf: Optional[float]


def _shorten(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return "…" + s[-(n - 1):]


def _safe_import_ultralytics():
    try:
        from ultralytics import YOLO  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Model Tester (Ultralytics)")
        self.geometry("1200x780")

        ok, err = _safe_import_ultralytics()
        if not ok:
            messagebox.showerror(
                "Missing dependency",
                "Could not import ultralytics.\n\n"
                "Install:\n  pip install -U ultralytics pillow\n\n"
                f"Import error:\n{err}",
            )
            self.destroy()
            return

        from ultralytics import YOLO
        self._YOLO = YOLO

        self.model = None
        self.model_lock = threading.Lock()

        self.work_q: "queue.Queue[tuple[str, dict]]" = queue.Queue()
        self.result_q: "queue.Queue[tuple[str, object]]" = queue.Queue()

        self.image_paths: List[str] = []
        self.current_preview_path: Optional[str] = None
        self.current_photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._start_worker()
        self.after(100, self._poll_results)

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Button(top, text="Select Model (.pt)…", command=self._select_model).pack(side=tk.LEFT)
        self.model_path_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.model_path_var, width=60).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="Load Model", command=self._load_model).pack(side=tk.LEFT, padx=8)

        self.status_var = tk.StringVar(value="Model not loaded")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.RIGHT)

        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))

        ttk.Button(mid, text="Select Images…", command=self._select_images).pack(side=tk.LEFT)
        ttk.Button(mid, text="Select Folder…", command=self._select_folder).pack(side=tk.LEFT, padx=8)

        ttk.Label(mid, text="Conf:").pack(side=tk.LEFT, padx=(16, 6))
        self.conf_var = tk.DoubleVar(value=0.25)
        ttk.Entry(mid, textvariable=self.conf_var, width=6).pack(side=tk.LEFT)

        ttk.Label(mid, text="IOU:").pack(side=tk.LEFT, padx=(12, 6))
        self.iou_var = tk.DoubleVar(value=0.45)
        ttk.Entry(mid, textvariable=self.iou_var, width=6).pack(side=tk.LEFT)

        ttk.Label(mid, text="ImgSz:").pack(side=tk.LEFT, padx=(12, 6))
        self.imgsz_var = tk.IntVar(value=640)
        ttk.Entry(mid, textvariable=self.imgsz_var, width=6).pack(side=tk.LEFT)

        ttk.Button(mid, text="Output Folder…", command=self._select_output).pack(side=tk.RIGHT)
        self.out_dir_var = tk.StringVar(value=str(Path.cwd() / "detected"))
        ttk.Entry(mid, textvariable=self.out_dir_var, width=50).pack(side=tk.RIGHT, padx=8)

        runrow = ttk.Frame(self)
        runrow.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(runrow, text="Run Detection + Save Annotated", command=self._run).pack(side=tk.RIGHT)

        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        lf = ttk.LabelFrame(left, text="Selected Images")
        lf.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.img_list = tk.Listbox(lf, height=10)
        self.img_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        self.img_list.bind("<<ListboxSelect>>", self._on_select_image)

        sb = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=self.img_list.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=8)
        self.img_list.configure(yscrollcommand=sb.set)

        rf = ttk.LabelFrame(left, text="Detection Summaries")
        rf.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(10, 0))

        cols = ("image", "detections", "classes", "best_conf", "saved")
        self.tree = ttk.Treeview(rf, columns=cols, show="headings", height=12)
        heading_names = {"image": "image", "detections": "detections", "classes": "classes", "best_conf": "best_conf", "saved": "saved"}
        for c, w in zip(cols, (280, 90, 350, 110, 300)):
            self.tree.heading(c, text=heading_names.get(c, c))
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)

        rsb = ttk.Scrollbar(rf, orient=tk.VERTICAL, command=self.tree.yview)
        rsb.pack(side=tk.RIGHT, fill=tk.Y, pady=8)
        self.tree.configure(yscrollcommand=rsb.set)

        pf = ttk.LabelFrame(right, text="Preview")
        pf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.preview_canvas = tk.Canvas(pf, width=420, height=420, bg="#111")
        self.preview_canvas.pack(padx=10, pady=10)

        ttk.Button(pf, text="Open Full Image…", command=self._open_full).pack(padx=10, pady=(0, 10), anchor="w")
        ttk.Button(pf, text="Open Output Folder…", command=self._open_output_folder).pack(padx=10, pady=(0, 10), anchor="w")

    def _start_worker(self):
        t = threading.Thread(target=self._worker_loop, daemon=True)
        t.start()

    def _worker_loop(self):
        while True:
            job_type, payload = self.work_q.get()
            try:
                if job_type == "load_model":
                    model_path = payload["model_path"]
                    self.result_q.put(("status", f"Loading model: {model_path}"))
                    m = self._YOLO(model_path)
                    with self.model_lock:
                        self.model = m
                    self.result_q.put(("status", f"Model loaded: {model_path}"))
                elif job_type == "run":
                    self._run_job(payload)
            except Exception as e:
                self.result_q.put(("error", str(e)))
            finally:
                self.work_q.task_done()

    def _select_model(self):
        p = filedialog.askopenfilename(title="Select YOLO model", filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if p:
            self.model_path_var.set(p)

    def _load_model(self):
        p = self.model_path_var.get().strip()
        if not p:
            messagebox.showwarning("Model required", "Select a .pt model file first.")
            return
        if not Path(p).exists():
            messagebox.showerror("Not found", f"Model file not found:\n{p}")
            return
        self.work_q.put(("load_model", {"model_path": p}))

    def _select_images(self):
        paths = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
        )
        if not paths:
            return
        self._set_images(list(paths))

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Select folder of images")
        if not folder:
            return
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        paths = [str(p) for p in Path(folder).glob("*") if p.suffix.lower() in exts]
        if not paths:
            messagebox.showinfo("No images", "No images found in that folder.")
            return
        self._set_images(paths)

    def _set_images(self, paths: List[str]):
        self.image_paths = paths
        self.img_list.delete(0, tk.END)
        for p in self.image_paths:
            self.img_list.insert(tk.END, p)
        self.status_var.set(f"{len(self.image_paths)} images selected")
        self.img_list.selection_clear(0, tk.END)
        self.img_list.selection_set(0)
        self._on_select_image()

    def _select_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.out_dir_var.set(folder)

    def _on_select_image(self, *_):
        sel = self.img_list.curselection()
        if not sel:
            return
        path = self.image_paths[sel[0]]
        self.current_preview_path = path
        try:
            img = Image.open(path).convert("RGB")
            canvas_w = int(self.preview_canvas["width"])
            canvas_h = int(self.preview_canvas["height"])
            img.thumbnail((canvas_w, canvas_h))
            self.current_photo = ImageTk.PhotoImage(img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.current_photo)
        except Exception as e:
            messagebox.showerror("Image load failed", f"Failed to open image:\n{path}\n\n{e}")

    def _open_full(self):
        if not self.current_preview_path:
            return
        try:
            os.startfile(self.current_preview_path)  # type: ignore[attr-defined]
        except Exception:
            messagebox.showinfo("Path", self.current_preview_path)

    def _open_output_folder(self):
        out_dir = self.out_dir_var.get().strip()
        if not out_dir:
            return
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            os.startfile(out_dir)  # type: ignore[attr-defined]
        except Exception:
            messagebox.showinfo("Output folder", out_dir)

    def _ensure_model(self) -> bool:
        with self.model_lock:
            return self.model is not None

    def _run(self):
        if not self.image_paths:
            messagebox.showwarning("No images", "Select one or more images first.")
            return
        if not self._ensure_model():
            self._load_model()

        out_dir = Path(self.out_dir_var.get().strip() or "detected")
        out_dir.mkdir(parents=True, exist_ok=True)

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.work_q.put(("run", {
            "paths": self.image_paths,
            "out_dir": str(out_dir),
            "conf": float(self.conf_var.get()),
            "iou": float(self.iou_var.get()),
            "imgsz": int(self.imgsz_var.get()),
        }))

    def _run_job(self, payload: dict):
        paths: List[str] = payload["paths"]
        out_dir = Path(payload["out_dir"])
        conf = float(payload["conf"])
        iou = float(payload["iou"])
        imgsz = int(payload["imgsz"])

        import time
        for _ in range(600):
            with self.model_lock:
                if self.model is not None:
                    break
            time.sleep(0.1)

        with self.model_lock:
            model = self.model
        if model is None:
            raise RuntimeError("Model not loaded. Click 'Load Model' and try again.")

        total = len(paths)
        for i, p in enumerate(paths, start=1):
            self.result_q.put(("status", f"Detecting {i}/{total}: {p}"))
            results = model.predict(source=p, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
            r0 = results[0]

            num = 0
            class_summary = ""
            best_conf = None
            saved_path = None

            try:
                num = 0 if r0.boxes is None else len(r0.boxes)
                if r0.boxes is not None and hasattr(r0.boxes, "conf") and r0.boxes.conf is not None:
                    try:
                        confs = [float(x) for x in r0.boxes.conf.tolist()]
                        if confs:
                            best_conf = max(confs)
                    except Exception:
                        best_conf = None
                class_counts = {}
                if r0.boxes is not None and hasattr(r0.boxes, "cls"):
                    for cls_idx in r0.boxes.cls.tolist():
                        cls_i = int(cls_idx)
                        name = r0.names.get(cls_i, str(cls_i))
                        class_counts[name] = class_counts.get(name, 0) + 1
                class_summary = ", ".join(f"{k}:{v}" for k, v in sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))) if class_counts else "(none)"
            except Exception:
                class_summary = "(could not summarize)"

            # Save annotated output ONLY if there was at least one detection
            if num > 0:
                try:
                    annotated = r0.plot()  # numpy array BGR
                    annotated = annotated[..., ::-1]  # BGR->RGB
                    img = Image.fromarray(annotated.astype("uint8"))
                    out_name = Path(p).stem + "_det.jpg"
                    saved_path = str(out_dir / out_name)
                    img.save(saved_path, quality=92)
                except Exception as e:
                    saved_path = None
                    self.result_q.put(("status", f"Warning: failed to save annotated image for {p}: {e}"))
            else:
                saved_path = None

            self.result_q.put(("summary", DetSummary(p, num, class_summary, saved_path, best_conf)))

        self.result_q.put(("status", f"Done. Processed {total} images. Saved annotated images (detections only) to: {out_dir}"))

    def _poll_results(self):
        try:
            while True:
                kind, obj = self.result_q.get_nowait()
                if kind == "status":
                    self.status_var.set(str(obj))
                elif kind == "error":
                    messagebox.showerror("Error", str(obj))
                    self.status_var.set("Error (see popup)")
                elif kind == "summary":
                    s: DetSummary = obj  # type: ignore[assignment]
                    self.tree.insert("", tk.END, values=(
                        _shorten(s.image_path, 45),
                        str(s.num_dets),
                        _shorten(s.classes, 65),
                        "" if s.best_conf is None else f"{s.best_conf:.3f}",
                        "" if not s.saved_path else _shorten(s.saved_path, 55),
                    ))
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_results)


if __name__ == "__main__":
    app = App()
    app.mainloop()
