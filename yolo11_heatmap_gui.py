import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import torch
import mss
from PIL import Image, ImageTk, ImageDraw, ImageFont

from ultralytics import YOLO


# ------------------- CONFIG -------------------
CAPTURE_DIR = "captures_heatmap"
os.makedirs(CAPTURE_DIR, exist_ok=True)

CONFIG_FILE = "last_model_path_heatmap.txt"

CANVAS_W = 960
CANVAS_H = 540

TARGET_HEIGHT = 1080  # scale screenshot down to at most this height for inference/display
DEFAULT_IMGSZ = 1280  # square letterbox size for CAM forward pass
# ---------------------------------------------


@dataclass
class Det:
    xyxy: tuple  # (x1, y1, x2, y2) in "img" coordinates (the scaled image we run inference on)
    cls_id: int
    conf: float


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def letterbox_square(pil_img: Image.Image, new_size: int, fill=114):
    """
    Letterbox an image into a square (new_size x new_size), preserving aspect ratio.
    Returns: (padded_img, meta)
      meta: dict with keys: scale, pad_x, pad_y, new_w, new_h, orig_w, orig_h
    """
    img = pil_img.convert("RGB")
    orig_w, orig_h = img.size
    scale = min(new_size / orig_w, new_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    pad_x = (new_size - new_w) // 2
    pad_y = (new_size - new_h) // 2

    canvas = Image.new("RGB", (new_size, new_size), (fill, fill, fill))
    canvas.paste(resized, (pad_x, pad_y))

    meta = dict(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        new_w=new_w,
        new_h=new_h,
        orig_w=orig_w,
        orig_h=orig_h,
        size=new_size,
    )
    return canvas, meta


def unletterbox_cam(cam_sq: np.ndarray, meta: dict) -> np.ndarray:
    """
    cam_sq: (S, S) float array aligned to letterboxed square
    Convert back to original image size by removing padding and resizing.
    """
    S = meta["size"]
    pad_x, pad_y = meta["pad_x"], meta["pad_y"]
    new_w, new_h = meta["new_w"], meta["new_h"]
    orig_w, orig_h = meta["orig_w"], meta["orig_h"]

    cam_sq = cam_sq[:S, :S]
    # Crop out padding region
    cam_crop = cam_sq[pad_y: pad_y + new_h, pad_x: pad_x + new_w]
    cam_img = Image.fromarray((cam_crop * 255.0).astype(np.uint8), mode="L")
    cam_img = cam_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    cam = np.asarray(cam_img).astype(np.float32) / 255.0
    return cam


def eigen_cam_from_activation(act: torch.Tensor) -> np.ndarray:
    """
    act: torch tensor (1, C, h, w)
    Returns CAM: (h, w) float in [0,1]
    Eigen-CAM style: first principal component projection of activations.
    """
    # (C, h, w)
    a = act[0].detach().float().cpu().numpy()
    C, h, w = a.shape
    X = a.reshape(C, -1).T  # (h*w, C)

    # center
    X = X - X.mean(axis=0, keepdims=True)

    # SVD -> principal component in feature space
    # X = U S Vt, Vt[0] is first PC direction (C,)
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        pc = Vt[0]  # (C,)
    except Exception:
        # fallback: use norm across channels if SVD fails
        cam = np.linalg.norm(a, axis=0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-9)
        return cam.astype(np.float32)

    proj = X @ pc  # (h*w,)
    cam = proj.reshape(h, w)
    cam = np.maximum(cam, 0.0)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-9)
    return cam.astype(np.float32)


def colorize_cam(cam: np.ndarray) -> Image.Image:
    """
    cam: (H,W) float [0,1]
    Returns RGB heatmap image (PIL).
    Uses matplotlib colormap if available; otherwise simple red-yellow ramp.
    """
    cam = np.clip(cam, 0.0, 1.0)
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("inferno")
        rgba = cmap(cam)  # (H,W,4)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")
    except Exception:
        # fallback: simple ramp (red/yellow)
        r = (cam * 255).astype(np.uint8)
        g = (np.sqrt(cam) * 255).astype(np.uint8)
        b = (np.zeros_like(r)).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
        return Image.fromarray(rgb, mode="RGB")


def alpha_blend(base: Image.Image, overlay: Image.Image, alpha: float) -> Image.Image:
    alpha = float(clamp(alpha, 0.0, 1.0))
    if alpha <= 0:
        return base.copy()
    if alpha >= 1:
        return overlay.copy()
    base = base.convert("RGB")
    overlay = overlay.convert("RGB")
    return Image.blend(base, overlay, alpha)


def get_torch_model(yolo_obj) -> torch.nn.Module:
    """
    Return the *graph-aware* Ultralytics model (DetectionModel/SegModel/etc),
    not the inner .model Sequential.
    """
    m = getattr(yolo_obj, "model", None)  # this is typically a torch.nn.Module already
    if isinstance(m, torch.nn.Module):
        return m

    # fallback: sometimes wrapped differently
    inner = getattr(m, "model", None)
    if isinstance(inner, torch.nn.Module):
        return inner

    raise RuntimeError(f"Could not locate a callable torch model on YOLO object. type(yolo_obj.model)={type(m)}")


def list_last_conv2d_modules(torch_model: torch.nn.Module, keep_last=40):
    convs = []
    for name, module in torch_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            convs.append((name, module))
    if len(convs) > keep_last:
        convs = convs[-keep_last:]
    return convs


class HeatmapGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO Heatmap Explorer (Eigen-CAM) — click-to-target + class filter")
        self.root.geometry("1100x900")
        self.root.resizable(True, True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: YOLO | None = None
        self.torch_model: torch.nn.Module | None = None
        self.model_path = ""

        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    self.model_path = f.read().strip()
            except Exception:
                pass

        # MSS capture (keep on Tk thread like your working GUI)
        self.sct = mss.mss()
        self.monitor_index = tk.IntVar(value=1)
        self.monitor = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]

        # State
        self.running = False
        self.next_capture_time = 0.0
        self.last_scaled_img: Image.Image | None = None  # scaled-to-<=1080 image used for inference/display
        self.last_dets: list[Det] = []
        self.selected_det_index: int | None = None

        # UI variables
        self.confidence = tk.DoubleVar(value=0.40)
        self.interval = tk.DoubleVar(value=2.0)
        self.save_all = tk.BooleanVar(value=True)

        self.class_filter_enabled = tk.BooleanVar(value=False)
        self.class_id_var = tk.StringVar(value="")  # user-typed class id
        self.class_combo_var = tk.StringVar(value="(none)")

        self.cam_mode = tk.StringVar(value="Mask to selected box")  # Global / Mask / Crop
        self.cam_alpha = tk.DoubleVar(value=0.50)
        self.cam_imgsz = tk.IntVar(value=DEFAULT_IMGSZ)
        self.cam_pad_pct = tk.DoubleVar(value=0.25)

        self.layer_combo_var = tk.StringVar(value="(load model)")
        self.conv_candidates: list[tuple[str, torch.nn.Module]] = []
        self.hook_handle = None
        self.last_activation: torch.Tensor | None = None

        # ---------------- UI ----------------
        title = tk.Label(root, text="YOLO Heatmap Explorer (Eigen-CAM)", font=("Segoe UI", 20, "bold"))
        title.pack(pady=10)

        top = ttk.Frame(root)
        top.pack(fill=tk.X, padx=12, pady=6)

        ttk.Button(top, text="Load .pt Model", command=self.load_model).pack(side=tk.LEFT, padx=6)
        self.model_status = ttk.Label(top, text="No model loaded", foreground="red")
        self.model_status.pack(side=tk.LEFT, padx=10)

        ttk.Label(top, text=f"Device: {self.device}").pack(side=tk.LEFT, padx=10)

        # Monitor selector
        ttk.Label(top, text="Monitor:").pack(side=tk.LEFT, padx=(20, 6))
        self.monitor_combo = ttk.Combobox(top, width=32, state="readonly")
        self.monitor_combo.pack(side=tk.LEFT, padx=6)
        self._populate_monitors()
        self.monitor_combo.bind("<<ComboboxSelected>>", self.on_monitor_changed)

        # Controls row 2
        row2 = ttk.Frame(root)
        row2.pack(fill=tk.X, padx=12, pady=6)

        ttk.Button(row2, text="Capture Once", command=self.capture_once).pack(side=tk.LEFT, padx=6)

        self.btn_toggle = ttk.Button(row2, text="Start Monitoring", command=self.toggle_monitoring)
        self.btn_toggle.pack(side=tk.LEFT, padx=6)

        ttk.Label(row2, text="Interval (s):").pack(side=tk.LEFT, padx=(12, 6))
        ttk.Entry(row2, textvariable=self.interval, width=6).pack(side=tk.LEFT, padx=6)

        ttk.Checkbutton(row2, text="Save captures", variable=self.save_all).pack(side=tk.LEFT, padx=(12, 6))

        ttk.Label(row2, text="Conf:").pack(side=tk.LEFT, padx=(12, 6))
        ttk.Scale(row2, from_=0.0, to=1.0, variable=self.confidence, orient=tk.HORIZONTAL, length=140).pack(side=tk.LEFT, padx=6)
        self.conf_label = ttk.Label(row2, text=f"{self.confidence.get():.2f}")
        self.conf_label.pack(side=tk.LEFT, padx=6)
        self.confidence.trace_add("write", lambda *_: self.conf_label.config(text=f"{self.confidence.get():.2f}"))

        # Class filter row
        row3 = ttk.LabelFrame(root, text="Class Targeting / Filtering")
        row3.pack(fill=tk.X, padx=12, pady=8)

        ttk.Checkbutton(row3, text="Enable class filter (only show/target this class)", variable=self.class_filter_enabled,
                        command=self.recompute_from_last).pack(side=tk.LEFT, padx=6)

        ttk.Label(row3, text="Type Class ID:").pack(side=tk.LEFT, padx=(12, 6))
        id_entry = ttk.Entry(row3, textvariable=self.class_id_var, width=6)
        id_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(row3, text="Apply ID", command=self.apply_class_id).pack(side=tk.LEFT, padx=6)

        ttk.Label(row3, text="Or pick:").pack(side=tk.LEFT, padx=(12, 6))
        self.class_combo = ttk.Combobox(row3, width=32, state="readonly", textvariable=self.class_combo_var)
        self.class_combo.pack(side=tk.LEFT, padx=6)
        self.class_combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_class_combo())

        ttk.Button(row3, text="Clear selection", command=self.clear_selection).pack(side=tk.RIGHT, padx=6)

        # CAM settings row
        row4 = ttk.LabelFrame(root, text="Heatmap Settings (Eigen-CAM)")
        row4.pack(fill=tk.X, padx=12, pady=8)

        ttk.Label(row4, text="Mode:").pack(side=tk.LEFT, padx=(6, 6))
        ttk.Combobox(row4, width=20, state="readonly", textvariable=self.cam_mode,
                     values=["Global", "Mask to selected box", "Crop to selected box"]).pack(side=tk.LEFT, padx=6)

        ttk.Label(row4, text="Alpha:").pack(side=tk.LEFT, padx=(12, 6))
        ttk.Scale(row4, from_=0.0, to=1.0, variable=self.cam_alpha, orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=6)

        ttk.Label(row4, text="CAM imgsz:").pack(side=tk.LEFT, padx=(12, 6))
        ttk.Entry(row4, textvariable=self.cam_imgsz, width=7).pack(side=tk.LEFT, padx=6)

        ttk.Label(row4, text="Box pad %:").pack(side=tk.LEFT, padx=(12, 6))
        ttk.Entry(row4, textvariable=self.cam_pad_pct, width=7).pack(side=tk.LEFT, padx=6)

        ttk.Label(row4, text="Layer:").pack(side=tk.LEFT, padx=(12, 6))
        self.layer_combo = ttk.Combobox(row4, width=40, state="readonly", textvariable=self.layer_combo_var)
        self.layer_combo.pack(side=tk.LEFT, padx=6)
        self.layer_combo.bind("<<ComboboxSelected>>", lambda _e: self._install_hook_for_selected_layer())

        ttk.Button(row4, text="Recompute heatmap", command=self.recompute_from_last).pack(side=tk.RIGHT, padx=6)

        # Tabs for images
        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        tab_det = ttk.Frame(nb)
        tab_cam = ttk.Frame(nb)
        nb.add(tab_det, text="Detections")
        nb.add(tab_cam, text="Heatmap")

        self.det_canvas = tk.Canvas(tab_det, width=CANVAS_W, height=CANVAS_H, bg="#0d0d0d", highlightthickness=2, highlightbackground="#333")
        self.det_canvas.pack(pady=8)
        self.det_canvas.bind("<Button-1>", self.on_canvas_click)

        self.cam_canvas = tk.Canvas(tab_cam, width=CANVAS_W, height=CANVAS_H, bg="#0d0d0d", highlightthickness=2, highlightbackground="#333")
        self.cam_canvas.pack(pady=8)

        # Copyable log (like your GUI, but with explicit copy buttons)
        log_frame = ttk.Frame(root)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        btns = ttk.Frame(log_frame)
        btns.pack(fill=tk.X)

        ttk.Button(btns, text="Copy Log", command=self.copy_log).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=6)

        self.log = tk.Text(log_frame, height=10, state="disabled", font=("Consolas", 10),
                           bg="#1e1e1e", fg="#00ff00", insertbackground="#00ff00")
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.config(yscrollcommand=scrollbar.set)

        # Photo refs
        self.det_photo = None
        self.cam_photo = None

        # display mapping for click->image coords
        self._disp_meta = None  # dict with disp_w/disp_h/offset_x/offset_y/scale_x/scale_y

        self.log_msg("Ready. Load a model, Capture Once, then click a box to target heatmap.")

    # ---------- logging ----------
    def log_msg(self, msg: str):
        self.log.configure(state="normal")
        t = datetime.now().strftime("%H:%M:%S")
        self.log.insert("end", f"[{t}] {msg}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def log_exc(self, where: str, e: Exception):
        tb = traceback.format_exc()
        self.log_msg(f"ERROR in {where}: {repr(e)}")
        self.log_msg(tb)

    def copy_log(self):
        try:
            txt = self.log.get("1.0", "end-1c")
            self.root.clipboard_clear()
            self.root.clipboard_append(txt)
            self.log_msg("Log copied to clipboard.")
        except Exception as e:
            self.log_exc("copy_log", e)

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ---------- monitors ----------
    def _populate_monitors(self):
        items = []
        for i, m in enumerate(self.sct.monitors):
            if i == 0:
                continue
            items.append(f"{i}: {m['width']}x{m['height']} @ ({m['left']},{m['top']})")
        if not items:
            items = ["1: (single monitor)"]
        self.monitor_combo["values"] = items
        # best-effort select #1
        self.monitor_combo.current(0)

    def on_monitor_changed(self, _evt=None):
        try:
            s = self.monitor_combo.get().split(":")[0].strip()
            idx = int(s)
            self.monitor_index.set(idx)
            self.monitor = self.sct.monitors[idx]
            self.log_msg(f"Monitor set to {idx}: {self.monitor['width']}x{self.monitor['height']}")
        except Exception as e:
            self.log_exc("on_monitor_changed", e)

    # ---------- model ----------
    def load_model(self):
        try:
            initial_dir = os.path.dirname(self.model_path) or "."
            initial_file = os.path.basename(self.model_path) if self.model_path else ""
            model_path = filedialog.askopenfilename(
                initialdir=initial_dir,
                initialfile=initial_file,
                title="Select YOLO Model File",
                filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")],
            )
            if not model_path:
                return

            self.model = YOLO(model_path)
            self.torch_model = get_torch_model(self.model)
            self.torch_model.eval().to(self.device)

            self.model_path = model_path
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                f.write(model_path)

            self.model_status.config(text=f"Model: {os.path.basename(model_path)}", foreground="green")
            self.log_msg(f"Model loaded: {model_path}")

            # Populate class dropdown from model.names
            names = getattr(self.model, "names", None)
            if names is None:
                # sometimes names are on results, but usually model has them
                names = {}
            if isinstance(names, dict):
                items = ["(none)"] + [f"{i}: {names[i]}" for i in sorted(names.keys())]
            else:
                # list-like
                items = ["(none)"] + [f"{i}: {n}" for i, n in enumerate(names)]
            self.class_combo["values"] = items
            self.class_combo_var.set("(none)")

            # Populate conv layer list
            self.conv_candidates = list_last_conv2d_modules(self.torch_model, keep_last=40)
            if self.conv_candidates:
                layer_names = [name for name, _m in self.conv_candidates]
                self.layer_combo["values"] = layer_names
                self.layer_combo_var.set(layer_names[-1])
                self._install_hook_for_selected_layer()
                self.log_msg(f"Loaded {len(layer_names)} conv layers (showing last {len(layer_names)}). Default = last conv.")
            else:
                self.layer_combo["values"] = ["(no conv layers found?)"]
                self.layer_combo_var.set("(no conv layers found?)")

        except Exception as e:
            self.model_status.config(text="Model load failed", foreground="red")
            self.log_exc("load_model", e)

    def _install_hook_for_selected_layer(self):
        try:
            if self.hook_handle is not None:
                try:
                    self.hook_handle.remove()
                except Exception:
                    pass
                self.hook_handle = None

            if not self.torch_model or not self.conv_candidates:
                return

            target_name = self.layer_combo_var.get().strip()
            target_mod = None
            for name, mod in self.conv_candidates:
                if name == target_name:
                    target_mod = mod
                    break
            if target_mod is None:
                self.log_msg(f"Could not find layer '{target_name}' to hook.")
                return

            def hook_fn(_m, _inp, out):
                # out: (B,C,H,W)
                if isinstance(out, torch.Tensor) and out.ndim == 4:
                    self.last_activation = out

            self.hook_handle = target_mod.register_forward_hook(hook_fn)
            self.log_msg(f"Hooked layer for CAM: {target_name}")
        except Exception as e:
            self.log_exc("_install_hook_for_selected_layer", e)

    # ---------- class filter ----------
    def apply_class_id(self):
        self.class_combo_var.set("(none)")
        self.selected_det_index = None
        self.recompute_from_last()

    def apply_class_combo(self):
        s = self.class_combo_var.get()
        if s and s != "(none)":
            # update text entry
            try:
                cls_id = int(s.split(":")[0].strip())
                self.class_id_var.set(str(cls_id))
            except Exception:
                pass
        self.selected_det_index = None
        self.recompute_from_last()

    def _get_target_class_id(self):
        if not self.class_filter_enabled.get():
            return None
        s = self.class_id_var.get().strip()
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None

    # ---------- monitoring ----------
    def toggle_monitoring(self):
        if not self.running:
            if self.model is None:
                self.log_msg("Load a model first!")
                return
            self.running = True
            self.btn_toggle.config(text="Stop Monitoring")
            self.log_msg("Monitoring started.")
            self.next_capture_time = time.time()
            self.root.after(50, self._loop)
        else:
            self.running = False
            self.btn_toggle.config(text="Start Monitoring")
            self.log_msg("Monitoring stopped.")

    def _loop(self):
        if not self.running:
            return
        try:
            now = time.time()
            iv = float(self.interval.get())
            iv = max(0.2, iv)
            if now >= self.next_capture_time:
                self.next_capture_time = now + iv
                self.capture_and_process()
        except Exception as e:
            self.log_exc("_loop", e)
        finally:
            self.root.after(50, self._loop)

    def capture_once(self):
        if self.model is None:
            self.log_msg("Load a model first!")
            return
        self.capture_and_process()

    # ---------- main work ----------
    def capture_and_process(self):
        try:
            # 1) Full-res capture (same MSS pattern as your working GUI) :contentReference[oaicite:4]{index=4}
            raw = self.sct.grab(self.monitor)
            full_img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

            # 2) Scale down to <=1080p height (like your GUI) :contentReference[oaicite:5]{index=5}
            scale = TARGET_HEIGHT / full_img.height
            if scale < 1:
                new_size = (int(full_img.width * scale), TARGET_HEIGHT)
                img = full_img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                img = full_img.copy()
                scale = 1.0

            self.last_scaled_img = img

            # 3) Optionally save
            if self.save_all.get():
                fn = os.path.join(CAPTURE_DIR, f"capture_{now_ts()}.png")
                full_img.save(fn)
                self.log_msg(f"Saved capture: {fn}")

            # 4) Run detection with Ultralytics (boxes.cls/boxes.conf/boxes.xyxy)
            conf = float(self.confidence.get())
            # We keep imgsz for predict separate from CAM imgsz.
            results = self.model(img, conf=conf, imgsz=DEFAULT_IMGSZ, device=self.device, verbose=False)[0]

            dets: list[Det] = []
            boxes = getattr(results, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                cls = boxes.cls.detach().cpu().numpy().astype(int)
                cf = boxes.conf.detach().cpu().numpy()
                for i in range(len(cls)):
                    dets.append(Det(tuple(xyxy[i].tolist()), int(cls[i]), float(cf[i])))

            # 5) Apply class filter
            target_cls = self._get_target_class_id()
            if target_cls is not None:
                dets = [d for d in dets if d.cls_id == target_cls]

            self.last_dets = dets
            if dets:
                self.log_msg(f"Detections: {len(dets)}")
            else:
                self.log_msg("Detections: 0")

            # 6) Render detections tab
            self._render_detections(img)

            # 7) Render heatmap tab
            self._render_heatmap(img)

        except Exception as e:
            self.log_exc("capture_and_process", e)

    def _render_detections(self, img: Image.Image):
        try:
            disp = img.copy()
            draw = ImageDraw.Draw(disp)

            # basic font fallback
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except Exception:
                font = ImageFont.load_default()

            names = getattr(self.model, "names", {}) if self.model else {}

            for i, d in enumerate(self.last_dets):
                x1, y1, x2, y2 = map(int, d.xyxy)
                label = f"{d.cls_id}:{names.get(d.cls_id, 'cls')} {d.conf:.2f}"
                # highlight selected
                is_sel = (self.selected_det_index == i)
                w = 5 if is_sel else 3
                color = "#00FFFF" if is_sel else "#FF0000"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=w)
                draw.text((x1 + 4, y1 + 4), label, fill=color, font=font)

            # Fit to canvas while preserving aspect ratio, centered (like your GUI)
            thumb = disp.copy()
            thumb.thumbnail((CANVAS_W, CANVAS_H), Image.Resampling.LANCZOS)
            tw, th = thumb.size
            off_x = (CANVAS_W - tw) // 2
            off_y = (CANVAS_H - th) // 2

            self.det_photo = ImageTk.PhotoImage(thumb)
            self.det_canvas.delete("all")
            self.det_canvas.create_image(off_x, off_y, image=self.det_photo, anchor="nw")

            # store mapping for click
            self._disp_meta = dict(
                disp_w=tw,
                disp_h=th,
                off_x=off_x,
                off_y=off_y,
                scale_x=tw / img.size[0],
                scale_y=th / img.size[1],
            )

            # Status line
            status = f"{len(self.last_dets)} detections (click box to target)" if self.last_dets else "No detections"
            self.det_canvas.create_text(CANVAS_W // 2, 20, text=status, fill="#00FF00" if self.last_dets else "#888888",
                                        font=("Segoe UI", 14, "bold"))
        except Exception as e:
            self.log_exc("_render_detections", e)

    def _render_heatmap(self, img: Image.Image):
        try:
            if self.model is None or self.torch_model is None:
                return

            # If layer hook not installed, can't CAM
            if self.hook_handle is None:
                self.log_msg("CAM: no hooked layer; pick a conv layer after loading model.")
                return

            imgsz = int(self.cam_imgsz.get())
            imgsz = int(clamp(imgsz, 256, 2048))

            # 1) Letterbox to square for CAM forward pass
            lb, meta = letterbox_square(img, imgsz, fill=114)
            x = np.asarray(lb).astype(np.float32) / 255.0  # (H,W,3)
            x = np.transpose(x, (2, 0, 1))  # (3,H,W)
            x = torch.from_numpy(x).unsqueeze(0).to(self.device)

            # 2) Forward pass to populate self.last_activation
            self.last_activation = None
            with torch.no_grad():
                _ = self.torch_model(x)

            act = self.last_activation
            if act is None:
                self.log_msg("CAM: activation was not captured (hook layer may not run). Try another layer.")
                return

            # 3) Eigen-CAM on activation
            cam_small = eigen_cam_from_activation(act)  # (h,w)
            cam_small_img = Image.fromarray((cam_small * 255).astype(np.uint8), mode="L")
            cam_sq_img = cam_small_img.resize((imgsz, imgsz), Image.Resampling.BILINEAR)
            cam_sq = np.asarray(cam_sq_img).astype(np.float32) / 255.0

            # 4) Map CAM back to original image size
            cam = unletterbox_cam(cam_sq, meta)  # (orig_h, orig_w) float

            # 5) If user selected a box and mode wants it: mask/crop/renorm
            mode = self.cam_mode.get()
            if self.selected_det_index is not None and 0 <= self.selected_det_index < len(self.last_dets):
                d = self.last_dets[self.selected_det_index]
                x1, y1, x2, y2 = map(int, d.xyxy)

                pad_pct = float(self.cam_pad_pct.get())
                pad_pct = float(clamp(pad_pct, 0.0, 1.0))

                bw = x2 - x1
                bh = y2 - y1
                px = int(round(bw * pad_pct))
                py = int(round(bh * pad_pct))

                X1 = clamp(x1 - px, 0, img.size[0] - 1)
                Y1 = clamp(y1 - py, 0, img.size[1] - 1)
                X2 = clamp(x2 + px, 1, img.size[0])
                Y2 = clamp(y2 + py, 1, img.size[1])

                if mode == "Mask to selected box":
                    masked = np.zeros_like(cam)
                    masked[Y1:Y2, X1:X2] = cam[Y1:Y2, X1:X2]
                    # renormalize within box
                    box = masked[Y1:Y2, X1:X2]
                    if box.size > 0:
                        bmin, bmax = float(box.min()), float(box.max())
                        if bmax > bmin:
                            masked[Y1:Y2, X1:X2] = (box - bmin) / (bmax - bmin + 1e-9)
                    cam = masked

                elif mode == "Crop to selected box":
                    # crop CAM + image, then later we paste it back (so the tab still shows full image)
                    crop_cam = cam[Y1:Y2, X1:X2]
                    if crop_cam.size > 0:
                        cmin, cmax = float(crop_cam.min()), float(crop_cam.max())
                        if cmax > cmin:
                            crop_cam = (crop_cam - cmin) / (cmax - cmin + 1e-9)
                        cam2 = np.zeros_like(cam)
                        cam2[Y1:Y2, X1:X2] = crop_cam
                        cam = cam2

            # 6) Colorize + overlay
            heat = colorize_cam(cam)
            alpha = float(self.cam_alpha.get())
            blended = alpha_blend(img, heat, alpha)

            # draw selection bbox (for clarity)
            if self.selected_det_index is not None and 0 <= self.selected_det_index < len(self.last_dets):
                d = self.last_dets[self.selected_det_index]
                x1, y1, x2, y2 = map(int, d.xyxy)
                dd = ImageDraw.Draw(blended)
                dd.rectangle([x1, y1, x2, y2], outline="#00FFFF", width=4)

            # 7) Fit to canvas
            thumb = blended.copy()
            thumb.thumbnail((CANVAS_W, CANVAS_H), Image.Resampling.LANCZOS)
            tw, th = thumb.size
            off_x = (CANVAS_W - tw) // 2
            off_y = (CANVAS_H - th) // 2

            self.cam_photo = ImageTk.PhotoImage(thumb)
            self.cam_canvas.delete("all")
            self.cam_canvas.create_image(off_x, off_y, image=self.cam_photo, anchor="nw")
            self.cam_canvas.create_text(CANVAS_W // 2, 20, text=f"Heatmap ({mode}) — click a box in Detections tab",
                                        fill="#00FF00", font=("Segoe UI", 14, "bold"))
        except Exception as e:
            self.log_exc("_render_heatmap", e)

    # ---------- clicking / targeting ----------
    def on_canvas_click(self, event):
        try:
            if not self.last_dets or not self.last_scaled_img or not self._disp_meta:
                return

            m = self._disp_meta
            x_disp = event.x - m["off_x"]
            y_disp = event.y - m["off_y"]
            if x_disp < 0 or y_disp < 0 or x_disp >= m["disp_w"] or y_disp >= m["disp_h"]:
                return

            # map to image coords
            x_img = x_disp / (m["scale_x"] + 1e-9)
            y_img = y_disp / (m["scale_y"] + 1e-9)

            # pick box containing click, else nearest center
            best_i = None
            best_dist = 1e18
            for i, d in enumerate(self.last_dets):
                x1, y1, x2, y2 = d.xyxy
                if x1 <= x_img <= x2 and y1 <= y_img <= y2:
                    best_i = i
                    best_dist = 0
                    break
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                dist = (cx - x_img) ** 2 + (cy - y_img) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_i = i

            if best_i is None:
                return

            self.selected_det_index = best_i
            d = self.last_dets[best_i]
            self.log_msg(f"Selected box #{best_i} cls={d.cls_id} conf={d.conf:.2f}")

            # If class filter is enabled, auto-set the class ID to the clicked detection
            if self.class_filter_enabled.get():
                self.class_id_var.set(str(d.cls_id))

            # redraw + recompute CAM without recapturing
            self._render_detections(self.last_scaled_img)
            self._render_heatmap(self.last_scaled_img)

        except Exception as e:
            self.log_exc("on_canvas_click", e)

    def clear_selection(self):
        self.selected_det_index = None
        self.recompute_from_last()

    def recompute_from_last(self):
        try:
            if self.last_scaled_img is None:
                return
            self._render_detections(self.last_scaled_img)
            self._render_heatmap(self.last_scaled_img)
        except Exception as e:
            self.log_exc("recompute_from_last", e)


if __name__ == "__main__":
    root = tk.Tk()
    app = HeatmapGUI(root)
    root.mainloop()
