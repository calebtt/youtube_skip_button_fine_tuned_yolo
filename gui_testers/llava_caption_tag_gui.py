#!/usr/bin/env python3
"""
LLaVA GUI tester (generic image descriptions + tag matching)

Fixes the error:
  image features and image tokens do not match: tokens 0, features ...

Why it happens:
- LLaVA-style models require an image placeholder token in the prompt.
- The generic transformers pipeline("image-to-text") can end up calling the model
  with empty/ignored text, producing 0 image tokens while still passing image features.

What this script does instead:
- Uses AutoProcessor + model.generate() and (when available) the model's chat template
  so the image token is always present.

Dependencies:
  pip install -U pillow transformers torch
"""

import threading
import queue
import re
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image, ImageTk

DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEFAULT_PROMPT = "Describe this image in detail. Include any visible UI elements, buttons, and on-screen text."


@dataclass
class CaptionResult:
    image_path: str
    caption: str
    matched_tags: List[str]


def _safe_import_transformers():
    try:
        import transformers  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def _match_tags(text: str, tags: List[str]) -> List[str]:
    hits = []
    for t in tags:
        if re.search(re.escape(t), text, flags=re.IGNORECASE):
            hits.append(t)
    return hits


def _pick_device_and_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _strip_to_answer(text: str) -> str:
    t = text.strip()
    for marker in ["ASSISTANT:", "Assistant:", "assistant:", "### Assistant:", "###assistant:"]:
        if marker in t:
            t = t.split(marker)[-1].strip()
    return t


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LLaVA Description + Tag Search (Generic)")
        self.geometry("1180x760")

        ok, err = _safe_import_transformers()
        if not ok:
            messagebox.showerror(
                "Missing dependency",
                f"""Could not import transformers.

Install:
  pip install -U transformers torch pillow

Import error:
{err}""",
            )
            self.destroy()
            return

        self.processor = None
        self.model = None
        self.model_lock = threading.Lock()

        self.work_q = queue.Queue()
        self.result_q = queue.Queue()

        self.image_paths: List[str] = []
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
        self.model_var = tk.StringVar(value=DEFAULT_MODEL_ID)
        ttk.Entry(top, textvariable=self.model_var, width=50).pack(side=tk.LEFT)

        ttk.Button(top, text="Load Model", command=self._load_model).pack(side=tk.LEFT, padx=8)

        self.status_var = tk.StringVar(value="Model not loaded")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.RIGHT)

        mid = ttk.LabelFrame(self, text="Prompt + Tags")
        mid.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(mid, text="Prompt:").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))
        self.prompt_var = tk.StringVar(value=DEFAULT_PROMPT)
        ttk.Entry(mid, textvariable=self.prompt_var, width=120).grid(row=0, column=1, sticky="we", padx=8, pady=(8, 0))

        ttk.Label(mid, text="Tags (comma-separated):").grid(row=1, column=0, sticky="w", padx=8, pady=(8, 8))
        self.tags_var = tk.StringVar(value="button, text, menu, player")
        ttk.Entry(mid, textvariable=self.tags_var, width=120).grid(row=1, column=1, sticky="we", padx=8, pady=(8, 8))

        mid.columnconfigure(1, weight=1)

        runrow = ttk.Frame(self)
        runrow.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(runrow, text="Generate + Search Tags", command=self._run).pack(side=tk.RIGHT)

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

        rf = ttk.LabelFrame(left, text="Captions + Matches")
        rf.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(10, 0))

        self.text = tk.Text(rf, wrap=tk.WORD)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        self.text.tag_configure("hit", underline=True)

        rsb = ttk.Scrollbar(rf, orient=tk.VERTICAL, command=self.text.yview)
        rsb.pack(side=tk.RIGHT, fill=tk.Y, pady=8)
        self.text.configure(yscrollcommand=rsb.set)

        pf = ttk.LabelFrame(right, text="Preview")
        pf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.preview_canvas = tk.Canvas(pf, width=420, height=420, bg="#111")
        self.preview_canvas.pack(padx=10, pady=10)

        ttk.Button(pf, text="Open Full Image…", command=self._open_full).pack(padx=10, pady=(0, 10), anchor="w")

    def _start_worker(self):
        t = threading.Thread(target=self._worker_loop, daemon=True)
        t.start()

    def _worker_loop(self):
        while True:
            job_type, payload = self.work_q.get()
            try:
                if job_type == "load_model":
                    self._load_model_job(payload)
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

    def _load_model_job(self, payload: dict):
        model_id = payload["model_id"]
        self.result_q.put(("status", f"Loading model: {model_id}"))

        device, dtype = _pick_device_and_dtype()

        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)

        model = None
        first_err = None
        try:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if device != "cpu" else None,
            )
        except Exception as e:
            first_err = e
            model = None

        if model is None:
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map="auto" if device != "cpu" else None,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model '{model_id}'.\n\nFirst error:\n{first_err}\n\nSecond error:\n{e}"
                )

        model.eval()

        with self.model_lock:
            self.processor = processor
            self.model = model

        self.result_q.put(("status", f"Model loaded: {model_id} ({device})"))

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
        import os
        try:
            os.startfile(self.current_preview_path)  # type: ignore[attr-defined]
        except Exception:
            messagebox.showinfo("Path", self.current_preview_path)

    def _ensure_model(self) -> bool:
        with self.model_lock:
            return self.model is not None and self.processor is not None

    def _run(self):
        if not self.image_paths:
            messagebox.showwarning("No images", "Select one or more images first.")
            return
        if not self._ensure_model():
            self._load_model()

        prompt = self.prompt_var.get().strip()
        tags = [t.strip() for t in self.tags_var.get().split(",") if t.strip()]
        if not prompt:
            messagebox.showwarning("Prompt required", "Enter a prompt.")
            return

        self.text.delete("1.0", tk.END)
        self.work_q.put(("run", {"paths": self.image_paths, "prompt": prompt, "tags": tags}))

    def _run_job(self, payload: dict):
        paths: List[str] = payload["paths"]
        user_prompt: str = payload["prompt"]
        tags: List[str] = payload["tags"]

        import time
        for _ in range(900):
            with self.model_lock:
                if self.model is not None and self.processor is not None:
                    break
            time.sleep(0.1)

        with self.model_lock:
            model = self.model
            processor = self.processor
        if model is None or processor is None:
            raise RuntimeError("Model not loaded. Click 'Load Model' and try again.")

        import torch

        total = len(paths)
        for i, p in enumerate(paths, start=1):
            self.result_q.put(("status", f"Generating {i}/{total}: {p}"))
            img = Image.open(p).convert("RGB")

            # Chat template ensures the required image token(s) are present.
            if hasattr(processor, "apply_chat_template"):
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}
                ]
                templ = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(images=img, text=templ, return_tensors="pt")
            else:
                # Fallback: many LLaVA-style models accept "<image>" in the prompt.
                templ = f"USER: <image>\n{user_prompt}\nASSISTANT:"
                inputs = processor(images=img, text=templ, return_tensors="pt")

            # Move tensors to model device when possible.
            try:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            except Exception:
                pass

            with torch.inference_mode():
                gen_ids = model.generate(**inputs, max_new_tokens=180, do_sample=False)

            if hasattr(processor, "batch_decode"):
                decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            else:
                decoded = processor.decode(gen_ids[0], skip_special_tokens=True)  # type: ignore[attr-defined]

            caption = _strip_to_answer(decoded)
            matched = _match_tags(caption, tags)
            self.result_q.put(("caption", CaptionResult(p, caption, matched)))

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
                elif kind == "caption":
                    self._append_caption(obj)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_results)

    def _append_caption(self, r: CaptionResult):
        header = f"\n=== {r.image_path} ===\n"
        self.text.insert(tk.END, header)
        start_idx = self.text.index(tk.END)
        self.text.insert(tk.END, r.caption.strip() + "\n")

        caption_start = start_idx
        caption_end = self.text.index(tk.END)
        caption_text = self.text.get(caption_start, caption_end)

        for tag in r.matched_tags:
            for m in re.finditer(re.escape(tag), caption_text, flags=re.IGNORECASE):
                s = f"{caption_start}+{m.start()}c"
                e = f"{caption_start}+{m.end()}c"
                self.text.tag_add("hit", s, e)

        if r.matched_tags:
            self.text.insert(tk.END, f"Matched tags: {', '.join(r.matched_tags)}\n")
        else:
            self.text.insert(tk.END, "Matched tags: (none)\n")


if __name__ == "__main__":
    app = App()
    app.mainloop()
