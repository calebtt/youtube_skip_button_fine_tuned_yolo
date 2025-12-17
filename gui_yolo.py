import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
import torch
from ultralytics import YOLO
import mss
import pyautogui
import time
import os
from datetime import datetime

# ------------------- CONFIG -------------------
CAPTURE_DIR = "captures"
TARGET_HEIGHT = 1080
CAPTURE_INTERVAL = 4.0   # seconds between captures
os.makedirs(CAPTURE_DIR, exist_ok=True)
CONFIG_FILE = "last_model_path.txt"
# ---------------------------------------------

class AdSkipperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Ad Skipper – 4s Interval")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)

        self.model = None
        self.model_path = ""
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                self.model_path = f.read().strip()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.running = False
        self.next_capture_time = 0
        self.last_click = 0
        self.confidence = tk.DoubleVar(value=0.4)
        self.var_monitor = tk.BooleanVar(value=False)
        self.var_skip = tk.BooleanVar(value=False)
        self.var_save_all = tk.BooleanVar(value=False)

        # Title
        tk.Label(root, text="YouTube Ad Skipper", font=("Segoe UI", 22, "bold")).pack(pady=15)

        # Controls
        ctrl = ttk.Frame(root)
        ctrl.pack(pady=10)

        self.load_button = ttk.Button(ctrl, text="Load Model", command=self.load_model)
        self.load_button.pack(side=tk.LEFT, padx=25)

        self.model_status = ttk.Label(ctrl, text="No model loaded", foreground="red", font=("Segoe UI", 12, "bold"))
        self.model_status.pack(side=tk.LEFT, padx=25)

        # New line for other controls
        ctrl2 = ttk.Frame(root)
        ctrl2.pack(pady=10)

        self.monitor_check = ttk.Checkbutton(ctrl2, text="Enable Monitoring", variable=self.var_monitor, command=self.toggle)
        self.monitor_check.pack(side=tk.LEFT, padx=25)
        self.monitor_check.state(['disabled'])

        ttk.Checkbutton(ctrl2, text="Enable Auto-Skip", variable=self.var_skip).pack(side=tk.LEFT, padx=25)

        ttk.Checkbutton(ctrl2, text="Save ALL captures (unchecked = detections only)", variable=self.var_save_all).pack(side=tk.LEFT, padx=25)

        self.status = ttk.Label(ctrl2, text="Stopped", foreground="red", font=("Segoe UI", 12, "bold"))
        self.status.pack(side=tk.LEFT, padx=25)

        # Confidence threshold control
        ctrl3 = ttk.Frame(root)
        ctrl3.pack(pady=10)

        ttk.Label(ctrl3, text="Confidence Threshold:").pack(side=tk.LEFT, padx=25)
        self.conf_label = ttk.Label(ctrl3, text=f"{self.confidence.get():.2f}")
        self.conf_label.pack(side=tk.LEFT, padx=5)
        ttk.Scale(ctrl3, from_=0.0, to=1.0, variable=self.confidence, orient=tk.HORIZONTAL, length=150, command=self.update_conf_label).pack(side=tk.LEFT, padx=25)

        # Image canvas
        self.canvas = tk.Canvas(root, width=960, height=540, bg="#0d0d0d", highlightthickness=2, highlightbackground="#333")
        self.canvas.pack(pady=15)

        # Large log box with scrollbar
        log_frame = ttk.Frame(root)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        self.log = tk.Text(log_frame, height=10, state="disabled", font=("Consolas", 10), bg="#1e1e1e", fg="#00ff00", insertbackground="#00ff00")
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.config(yscrollcommand=scrollbar.set)

        self.photo = None

    def load_model(self):
        initial_dir = os.path.dirname(self.model_path) or "."
        initial_file = os.path.basename(self.model_path)
        model_path = filedialog.askopenfilename(initialdir=initial_dir, initialfile=initial_file, title="Select YOLO Model File", filetypes=[("PyTorch Model", "*.pt")])

        if model_path:
            self.model = YOLO(model_path)
            with open(CONFIG_FILE, 'w') as f:
                f.write(model_path)
            self.model_path = model_path
            self.model_status.config(text=f"Model: {os.path.basename(model_path)}", foreground="green")
            print(f"Model loaded on {self.device}")
            self.monitor_check.state(['!disabled'])
            self.log_msg(f"Model loaded: {model_path}")

    def update_conf_label(self, value):
        self.conf_label.config(text=f"{float(value):.2f}")

    def log_msg(self, msg):
        self.log.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.insert("end", f"[{timestamp}] {msg}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def toggle(self):
        if self.var_monitor.get():
            if self.model is None:
                self.log_msg("Load a model first!")
                self.var_monitor.set(False)
                return
            if not self.running:
                self.running = True
                self.status.config(text="RUNNING – 4s interval", foreground="#00ff00")
                self.log_msg("Monitoring STARTED – capturing every 4 seconds")
                self.next_capture_time = time.time()
                self.root.after(100, self.loop)
        else:
            self.running = False
            self.status.config(text="Stopped", foreground="red")
            self.log_msg("Monitoring STOPPED")

    def loop(self):
        if not self.running:
            return

        now = time.time()
        if now >= self.next_capture_time:
            self.next_capture_time = now + CAPTURE_INTERVAL
            self.capture_and_process()
        self.root.after(100, self.loop)

    def capture_and_process(self):
        # Full-res capture
        raw = self.sct.grab(self.monitor)
        full_img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

        # Scale to 1080p max height
        scale = TARGET_HEIGHT / full_img.height
        if scale < 1:
            new_size = (int(full_img.width * scale), TARGET_HEIGHT)
            img = full_img.resize(new_size, Image.Resampling.LANCZOS)
        else:
            img = full_img.copy()
            scale = 1.0

        # Detect
        conf_thresh = self.confidence.get()
        results = self.model(img, conf=conf_thresh, imgsz=1280, device=self.device, verbose=False)[0]
        detected = results.boxes is not None and len(results.boxes) > 0

        # Draw boxes
        draw = ImageDraw.Draw(img)
        max_conf = 0.0
        max_conf_index = -1
        if detected:
            conf_list = results.boxes.conf.tolist()
            max_conf = max(conf_list)
            max_conf_index = conf_list.index(max_conf)
            for i in range(len(results.boxes)):
                box = results.boxes.xyxy[i].tolist()
                conf = conf_list[i]
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle([x1, y1, x2, y2], outline="#FF0000", width=7)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                draw.ellipse([cx-15, cy-15, cx+15, cy+15], fill="#FF0000", outline="#FFFFFF", width=3)

                # Log detection position (full res)
                real_cx = int(cx / scale)
                real_cy = int(cy / scale)
                self.log_msg(f"DETECTED AD at ({real_cx}, {real_cy}) with confidence {conf:.2f}")

        # Auto-skip if enabled
        if detected and self.var_skip.get() and max_conf_index != -1:
            if time.time() - self.last_click > 3.0:
                box = results.boxes.xyxy[max_conf_index].tolist()
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                real_cx = int(cx / scale)
                real_cy = int(cy / scale)
                pyautogui.click(real_cx, real_cy)
                self.last_click = time.time()
                self.log_msg(f"SKIPPED AD → clicked at ({real_cx}, {real_cy}) with confidence {max_conf:.2f}")

        # Save logic
        should_save = self.var_save_all.get() or detected
        if should_save:
            if detected:
                prefix = f"detection_{max_conf:.2f}"
            else:
                prefix = "capture"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{ts}.png"
            full_img.save(os.path.join(CAPTURE_DIR, filename))  # always save full-res
            self.log_msg(f"Saved → {filename}")

        # Update GUI display (only once per capture)
        display = img.copy()
        display.thumbnail((960, 540), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(480, 270, image=self.photo)

        # Status update
        status_text = "Button DETECTED!" if detected else "No button"
        self.canvas.create_text(480, 30, text=status_text, fill="#00FF00" if detected else "#888888",
                                font=("Segoe UI", 16, "bold"))

if __name__ == "__main__":
    root = tk.Tk()
    app = AdSkipperGUI(root)
    root.mainloop()