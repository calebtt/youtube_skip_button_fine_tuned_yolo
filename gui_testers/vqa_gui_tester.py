#!/usr/bin/env python3
"""
VQA GUI tester (generic)

Purpose
- Select one or more images.
- Ask 1-2 visual questions per image using a VQA model.
- Shows results in a list and previews the selected image.
- Optional heuristic "region hint" overlay (e.g., bottom-right crop) to help with UI-button style questions.

Note
- This app is intentionally generic: you type the questions you care about.

Dependencies
  pip install -U pillow transformers torch

Default VQA model
- Uses a Hugging Face VQA pipeline. You can change MODEL_ID below.
"""

import threading
import queue
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image, ImageTk, ImageDraw

# ---------- Config ----------
MODEL_ID = "dandelin/vilt-b32-finetuned-vqa"
BOTTOM_RIGHT_CROP_FRACTION = 0.33
# ---------------------------


@dataclass
class VqaResult:
    image_path: str
    question: str
    answer: str
    score: Optional[float]
    used_crop: bool


def _shorten(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return "…" + s[-(n - 1):]


def _safe_import_transformers():
    try:
        from transformers import pipeline  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VQA Image Tester (Generic)")
        self.geometry("1100x720")

        ok, err = _safe_import_transformers()
        if not ok:
            messagebox.showerror(
                "Missing dependency",
                "Could not import transformers.\n\n"
                "Install:\n  pip install -U transformers torch pillow\n\n"
                f"Import error:\n{err}",
            )
            self.destroy()
            return

        from transformers import pipeline
        self._pipeline_cls = pipeline

        self.vqa = None
        self.vqa_lock = threading.Lock()

        self.work_q: "queue.Queue[Tuple[str, dict]]" = queue.Queue()
        self.result_q: "queue.Queue[Tuple[str, object]]" = queue.Queue()

        self.image_paths: List[str] = []
        self.current_image: Optional[Image.Image] = None
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.current_preview_path: Optional[str] = None

        self._build_ui()
        self._start_worker()
        self.after(100, self._poll_results)

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Button(top, text="Select Images…", command=self._select_images).pack(side=tk.LEFT)

        ttk.Label(top, text="Model ID:").pack(side=tk.LEFT, padx=(12, 6))
        self.model_var = tk.StringVar(value=MODEL_ID)
        ttk.Entry(top, textvariable=self.model_var, width=45).pack(side=tk.LEFT)

        ttk.Button(top, text="Load Model", command=self._load_model).pack(side=tk.LEFT, padx=8)

        self.status_var = tk.StringVar(value="Model not loaded")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.RIGHT)

        qf = ttk.LabelFrame(self, text="Questions (type what you want to test)")
        qf.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))

        qrow = ttk.Frame(qf)
        qrow.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(qrow, text="Q1:").grid(row=0, column=0, sticky="w")
        self.q1_var = tk.StringVar(value="Is there a video player visible?")
        ttk.Entry(qrow, textvariable=self.q1_var, width=90).grid(row=0, column=1, padx=6, sticky="we")

        ttk.Label(qrow, text="Q2:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.q2_var = tk.StringVar(value="Is there a button visible in the bottom-right corner?")
        ttk.Entry(qrow, textvariable=self.q2_var, width=90).grid(row=1, column=1, padx=6, pady=(6, 0), sticky="we")

        qrow.columnconfigure(1, weight=1)

        opt = ttk.Frame(qf)
        opt.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.use_crop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opt,
            text="Also ask Q2 on bottom-right crop (helps with UI button checks)",
            variable=self.use_crop_var,
        ).pack(side=tk.LEFT)

        ttk.Button(opt, text="Run on Selected Images", command=self._run).pack(side=tk.RIGHT)

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

        rf = ttk.LabelFrame(left, text="Results")
        rf.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(10, 0))

        cols = ("image", "question", "answer", "score", "crop")
        self.tree = ttk.Treeview(rf, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (240, 420, 150, 80, 60)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)

        rsb = ttk.Scrollbar(rf, orient=tk.VERTICAL, command=self.tree.yview)
        rsb.pack(side=tk.RIGHT, fill=tk.Y, pady=8)
        self.tree.configure(yscrollcommand=rsb.set)

        pf = ttk.LabelFrame(right, text="Preview")
        pf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.preview_canvas = tk.Canvas(pf, width=420, height=420, bg="#111")
        self.preview_canvas.pack(padx=10, pady=10)

        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            pf, text="Show bottom-right hint box overlay", variable=self.overlay_var, command=self._refresh_preview
        ).pack(padx=10, pady=(0, 10), anchor="w")

        ttk.Button(pf, text="Open Full Image…", command=self._open_full).pack(padx=10, pady=(0, 10), anchor="w")

    def _start_worker(self):
        t = threading.Thread(target=self._worker_loop, daemon=True)
        t.start()

    def _worker_loop(self):
        while True:
            job_type, payload = self.work_q.get()
            try:
                if job_type == "load_model":
                    model_id = payload["model_id"]
                    self.result_q.put(("status", f"Loading model: {model_id}"))
                    vqa = self._pipeline_cls("visual-question-answering", model=model_id)
                    with self.vqa_lock:
                        self.vqa = vqa
                    self.result_q.put(("status", f"Model loaded: {model_id}"))
                elif job_type == "run":
                    self._run_job(payload)
            except Exception as e:
                self.result_q.put(("error", str(e)))
            finally:
                self.work_q.task_done()

    def _load_model(self):
        model_id = self.model_var.get().strip()
        if not model_id:
            messagebox.showwarning("Model ID required", "Enter a Hugging Face model id (or path).")
            return
        self.work_q.put(("load_model", {"model_id": model_id}))

    def _select_images(self):
        paths = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
        )
        if not paths:
            return
        self.image_paths = list(paths)
        self.img_list.delete(0, tk.END)
        for p in self.image_paths:
            self.img_list.insert(tk.END, p)
        self.status_var.set(f"{len(self.image_paths)} images selected")
        self.img_list.selection_clear(0, tk.END)
        self.img_list.selection_set(0)
        self._on_select_image()

    def _on_select_image(self, *_):
        sel = self.img_list.curselection()
        if not sel:
            return
        idx = sel[0]
        path = self.image_paths[idx]
        self.current_preview_path = path
        try:
            img = Image.open(path).convert("RGB")
            self.current_image = img
            self._refresh_preview()
        except Exception as e:
            messagebox.showerror("Image load failed", f"Failed to open image:\n{path}\n\n{e}")

    def _refresh_preview(self):
        if self.current_image is None:
            return
        img = self.current_image.copy()

        if self.overlay_var.get():
            w, h = img.size
            bw = int(w * BOTTOM_RIGHT_CROP_FRACTION)
            bh = int(h * BOTTOM_RIGHT_CROP_FRACTION)
            x0, y0 = w - bw, h - bh
            draw = ImageDraw.Draw(img)
            draw.rectangle([x0, y0, w - 1, h - 1], outline="red", width=max(2, w // 250))

        canvas_w = int(self.preview_canvas["width"])
        canvas_h = int(self.preview_canvas["height"])
        img.thumbnail((canvas_w, canvas_h))
        self.current_photo = ImageTk.PhotoImage(img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.current_photo)

    def _open_full(self):
        if not self.current_preview_path:
            return
        import os
        try:
            os.startfile(self.current_preview_path)  # type: ignore[attr-defined]
        except Exception:
            messagebox.showinfo("Path", self.current_preview_path)

    def _ensure_model(self) -> bool:
        with self.vqa_lock:
            return self.vqa is not None

    def _run(self):
        if not self.image_paths:
            messagebox.showwarning("No images", "Select one or more images first.")
            return
        if not self._ensure_model():
            self._load_model()

        q1 = self.q1_var.get().strip()
        q2 = self.q2_var.get().strip()
        if not q1 and not q2:
            messagebox.showwarning("No questions", "Enter at least one question.")
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.work_q.put(("run", {"paths": self.image_paths, "q1": q1, "q2": q2, "use_crop": self.use_crop_var.get()}))

    def _run_job(self, payload: dict):
        paths: List[str] = payload["paths"]
        q1: str = payload["q1"]
        q2: str = payload["q2"]
        use_crop: bool = payload["use_crop"]

        import time
        for _ in range(600):
            with self.vqa_lock:
                if self.vqa is not None:
                    break
            time.sleep(0.1)
        with self.vqa_lock:
            vqa = self.vqa
        if vqa is None:
            raise RuntimeError("Model not loaded. Click 'Load Model' and try again.")

        total = len(paths)
        done = 0
        for p in paths:
            done += 1
            self.result_q.put(("status", f"Running {done}/{total}: {p}"))
            img = Image.open(p).convert("RGB")

            if q1:
                out = vqa(image=img, question=q1, top_k=1)
                if isinstance(out, list):
                    out = out[0] if out else {"answer": "", "score": None}
                self.result_q.put(("result", VqaResult(p, q1, str(out.get("answer", "")), out.get("score", None), False)))

            if q2:
                out = vqa(image=img, question=q2, top_k=1)
                if isinstance(out, list):
                    out = out[0] if out else {"answer": "", "score": None}
                self.result_q.put(("result", VqaResult(p, q2, str(out.get("answer", "")), out.get("score", None), False)))

                if use_crop:
                    w, h = img.size
                    bw = int(w * BOTTOM_RIGHT_CROP_FRACTION)
                    bh = int(h * BOTTOM_RIGHT_CROP_FRACTION)
                    crop = img.crop((w - bw, h - bh, w, h))
                    out2 = vqa(image=crop, question=q2, top_k=1)
                    if isinstance(out2, list):
                        out2 = out2[0] if out2 else {"answer": "", "score": None}
                    self.result_q.put(("result", VqaResult(p, q2 + " (bottom-right crop)", str(out2.get("answer", "")), out2.get("score", None), True)))

        self.result_q.put(("status", f"Done. Processed {total} images."))

    def _poll_results(self):
        try:
            while True:
                kind, obj = self.result_q.get_nowait()
                if kind == "status":
                    self.status_var.set(str(obj))
                elif kind == "error":
                    messagebox.showerror("Error", str(obj))
                    self.status_var.set("Error (see popup)")
                elif kind == "result":
                    r: VqaResult = obj  # type: ignore[assignment]
                    self.tree.insert("", tk.END, values=(
                        _shorten(r.image_path, 40),
                        _shorten(r.question, 70),
                        r.answer,
                        "" if r.score is None else f"{float(r.score):.3f}",
                        "yes" if r.used_crop else "",
                    ))
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_results)


if __name__ == "__main__":
    app = App()
    app.mainloop()
