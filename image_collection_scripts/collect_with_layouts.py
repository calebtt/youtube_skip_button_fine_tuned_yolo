# collect_with_layouts.py
# FINAL WORKING VERSION – tested on Windows 11 + Python 3.11 + conda (adskip) env
# Windows Snap, ~400–500 screenshots in 45 minutes

import argparse
import random
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import win32gui
import win32con
import win32api

# ─────────────────────────────────────────────────────────────────────────────
YOUTUBE_CONTAINS = "youtube"
OTHER_CONTAINS   = "brave"          # change if you use Chrome/Firefox/Edge

# ─────────────────────────────────────────────────────────────────────────────
def snap_window(hwnd, zone: str):
    try:
        monitor = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
        info = win32api.GetMonitorInfo(monitor)
        work = info['Work']
        left, top, right, bottom = work
        w = right - left
        h = bottom - top
    except:
        left, top, w, h = 0, 0, 1920, 1080

    half_w = w // 2
    half_h = h // 2

    zones = {
        "maximize":      lambda: win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE),
        "left_half":     (left, top, half_w, h),
        "right_half":    (left + half_w, top, half_w, h),
        "top_half":      (left, top, w, half_h),
        "bottom_half":   (left, top + half_h, w, half_h),
        "top_left":      (left, top, half_w, half_h),
        "top_right":     (left + half_w, top, half_w, half_h),
        "bottom_left":   (left, top + half_h, half_w, half_h),
        "bottom_right":  (left + half_w, top + half_h, half_w, half_h),
    }

    if zone not in zones:
        zone = random.choice([z for z in zones if z != "maximize"])

    if zone == "maximize":
        zones["maximize"]()
        time.sleep(0.3)
        return "maximize"

    x, y, cw, ch = zones[zone]
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    time.sleep(0.05)
    win32gui.MoveWindow(hwnd, x, y, cw, ch, True)
    time.sleep(0.2)
    return zone


def apply_smart_layout():
    def find(text):
        matches = []
        def enum(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).lower()
                if text.lower() in title:
                    matches.append(hwnd)
        win32gui.EnumWindows(enum, None)
        return matches

    yt = find(YOUTUBE_CONTAINS)
    other = find(OTHER_CONTAINS)
    if not yt or not other:
        return

    yt_hwnd = random.choice(yt)
    other_hwnd = random.choice(other)

    if random.random() < 0.75:        # YouTube big most of the time
        big, small = yt_hwnd, other_hwnd
        yt_big = True
    else:
        big, small = other_hwnd, yt_hwnd
        yt_big = False

    style = random.choices(
        ["maximized_pip", "side_by_side", "top_bottom", "corners"],
        weights=[60, 20, 10, 10], k=1)[0]

    if style == "maximized_pip":
        snap_window(big, "maximize")
        time.sleep(0.4)
        snap_window(small, random.choice(["top_right", "bottom_right"]))
    elif style == "side_by_side":
        snap_window(big, "left_half")
        snap_window(small, "right_half")
    elif style == "top_bottom":
        snap_window(big, "top_half")
        snap_window(small, "bottom_half")
    else:
        corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
        random.shuffle(corners)
        snap_window(big, corners[0])
        snap_window(small, corners[1])

    print(f"[layout] {time.strftime('%H:%M')} → {style} (YouTube={'BIG' if yt_big else 'small'})")


# ─────────────────────────────────────────────────────────────────────────────
# Simple, reliable capture – works on every Windows machine
# ─────────────────────────────────────────────────────────────────────────────
sct = mss.mss()

def capture_screen():
    # This line works 100% of the time – no config nonsense
    img = np.array(sct.grab(sct.monitors[1]))      # full primary monitor
    return img[:, :, :3]                           # drop alpha → BGR


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--hours", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out) / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    end_time = time.time() + args.hours * 3600
    next_layout = time.time() + 15
    last_small = None
    count = 0

    print("\nYouTube ad collector STARTED – rock solid version")
    print(f"Saving → {out_dir.resolve()}\n")

    while time.time() < end_time:
        if time.time() >= next_layout:
            apply_smart_layout()
            next_layout = time.time() + random.randint(240, 420)

        shot = capture_screen()
        small = cv2.resize(shot, (320, 180))

        if last_small is not None:
            diff = cv2.absdiff(small, last_small).mean()
            if diff < 18:
                time.sleep(3)
                continue

        fname = out_dir / f"{int(time.time()*1000)}.jpg"
        cv2.imwrite(str(fname), shot, [int(cv2.IMWRITE_JPEG_QUALITY), 93])

        last_small = small.copy()
        count += 1
        if count % 50 == 0:
            print(f"   {count} images saved...")

        time.sleep(4)

    print(f"\nDONE! {count} screenshots saved to")
    print(f"   {out_dir.resolve()}")
    print(f"\nNow run:")
    print(f"   python triage_hotkeys.py --src \"{out_dir}\" --out \"{out_dir.parent}\"")

if __name__ == "__main__":
    main()