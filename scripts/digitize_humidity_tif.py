#!/usr/bin/env python3
"""Digitize a blue humidity trace from a scanned chart TIFF/PNG."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract humidity (%) time series from chart image."
    )
    parser.add_argument("input_image", type=Path, help="Path to TIF/PNG/JPG chart")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/humidity_trace.csv"),
        help="Output CSV for extracted line",
    )
    parser.add_argument(
        "--output-overlay",
        type=Path,
        default=Path("output/humidity_trace_overlay.png"),
        help="Output image with detected trace",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="",
        help="Optional YYYY-MM-DD date for timestamp construction",
    )
    parser.add_argument(
        "--start-hour",
        type=float,
        default=8.0,
        help="Left edge hour label of the chart (default: 8)",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=24.0,
        help="Horizontal span represented by chart in hours (default: 24)",
    )
    parser.add_argument(
        "--step-px",
        type=int,
        default=1,
        help="Output sampling step in pixels (default: 1, no x-sampling loss)",
    )
    parser.add_argument("--x0", type=int, default=-1, help="Manual ROI x0")
    parser.add_argument("--x1", type=int, default=-1, help="Manual ROI x1")
    parser.add_argument("--y0", type=int, default=-1, help="Manual ROI y0 (0%)")
    parser.add_argument("--y1", type=int, default=-1, help="Manual ROI y1 (100%)")
    return parser.parse_args()


def detect_roi(image_bgr: np.ndarray) -> tuple[int, int, int, int]:
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (5, 40, 70), (35, 255, 255))
    contours, _ = cv2.findContours(orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return int(0.03 * w), int(0.99 * w), int(0.075 * h), int(0.985 * h)

    contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(contour)
    if bw < int(0.6 * w) or bh < int(0.5 * h):
        return int(0.03 * w), int(0.99 * w), int(0.075 * h), int(0.985 * h)

    x0 = max(0, x + 8)
    x1 = min(w - 1, x + bw - 8)
    y0 = max(0, y + int(0.07 * bh))
    y1 = min(h - 1, y + bh - 8)
    return x0, x1, y0, y1


def parse_date(date_str: str) -> Optional[dt.date]:
    if not date_str:
        return None
    return dt.date.fromisoformat(date_str)


def smooth_series(y_vals: np.ndarray, window: int = 11) -> np.ndarray:
    if window < 3 or len(y_vals) < window:
        return y_vals
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y_vals, kernel, mode="same")


def main() -> None:
    args = parse_args()
    try:
        pil_img = Image.open(args.input_image)
        if "A" in pil_img.getbands():
            rgba = pil_img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            pil_img = Image.alpha_composite(bg, rgba).convert("RGB")
        else:
            pil_img = pil_img.convert("RGB")
        image_rgb = np.array(pil_img)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise SystemExit(f"Cannot read image: {args.input_image} ({exc})")

    auto_x0, auto_x1, auto_y0, auto_y1 = detect_roi(image_bgr)
    x0 = auto_x0 if args.x0 < 0 else args.x0
    x1 = auto_x1 if args.x1 < 0 else args.x1
    y0 = auto_y0 if args.y0 < 0 else args.y0
    y1 = auto_y1 if args.y1 < 0 else args.y1

    x0, x1 = max(0, x0), min(image_bgr.shape[1] - 1, x1)
    y0, y1 = max(0, y0), min(image_bgr.shape[0] - 1, y1)
    if (x1 - x0) > 20:
        x0 += 4
        x1 -= 4
    if x1 <= x0 or y1 <= y0:
        raise SystemExit("Invalid ROI coordinates.")

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv, (5, 40, 70), (35, 255, 255))
    orange_ratio = float(np.mean(orange_mask[y0 : y1 + 1, x0 : x1 + 1] > 0))

    # Ink can be blue pen or dark gray/black pencil depending on scanned sheet.
    blue_mask = cv2.inRange(hsv, (85, 25, 15), (155, 255, 230))
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (179, 90, 130))
    ink_mask = cv2.bitwise_or(blue_mask, dark_mask)
    roi_mask = ink_mask[y0 : y1 + 1, x0 : x1 + 1]
    roi_mask = cv2.dilate(roi_mask, np.ones((1, 7), np.uint8), iterations=1)

    # Keep only connected components that span meaningful horizontal distance.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi_mask, connectivity=8)
    width = x1 - x0 + 1
    keep = np.zeros(n_labels, dtype=bool)
    min_span = max(20, int(0.015 * width))
    for li in range(1, n_labels):
        span_w = stats[li, cv2.CC_STAT_WIDTH]
        area = stats[li, cv2.CC_STAT_AREA]
        if span_w >= min_span and area >= 12:
            keep[li] = True
    if keep.any():
        roi_mask = np.where(keep[labels], 255, 0).astype(np.uint8)

    x_idx = np.arange(width)
    y_trace = np.full(width, np.nan, dtype=float)
    max_jump_px = 30.0
    last_y: Optional[float] = None
    for xi in x_idx:
        ys = np.where(roi_mask[:, xi] > 0)[0]
        if ys.size > 0:
            if last_y is None:
                chosen = float(np.median(ys))
            else:
                chosen = float(ys[np.argmin(np.abs(ys - last_y))])
                if abs(chosen - last_y) > max_jump_px:
                    chosen = np.nan
            y_trace[xi] = chosen
            if not np.isnan(chosen):
                last_y = chosen

    valid = np.where(~np.isnan(y_trace))[0]
    coverage_ratio = float(valid.size / max(1, width))
    if valid.size < 30:
        raise SystemExit(
            "Too few trace pixels found. Try manual ROI (--x0 --x1 --y0 --y1)."
        )
    if coverage_ratio < 0.08:
        raise SystemExit(
            "Trace coverage too low (<8%). Try manual ROI or better scan contrast."
        )

    y_trace = np.interp(x_idx, valid, y_trace[valid])
    y_trace = smooth_series(y_trace, window=11)

    sample_step = max(1, args.step_px)
    sampled = x_idx[::sample_step]
    sampled_y = y_trace[::sample_step]

    humidity = np.clip((sampled_y / (y1 - y0)) * 100.0, 0.0, 100.0)
    elapsed_hours = (sampled / max(1, width - 1)) * args.duration_hours
    hour_of_day = (args.start_hour + elapsed_hours) % 24.0

    base_date = parse_date(args.date)
    base_dt = None
    if base_date:
        base_dt = dt.datetime.combine(base_date, dt.time()) + dt.timedelta(
            hours=args.start_hour
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["x_px", "y_px", "humidity_pct", "hour_of_day", "elapsed_hour", "timestamp"]
        )
        for xi, yi, hum, hod, eh in zip(sampled, sampled_y, humidity, hour_of_day, elapsed_hours):
            timestamp = ""
            if base_dt is not None:
                timestamp = (base_dt + dt.timedelta(hours=float(eh))).isoformat(
                    timespec="minutes"
                )
            writer.writerow(
                [int(x0 + xi), round(float(y0 + yi), 2), round(float(hum), 2), round(float(hod), 4), round(float(eh), 4), timestamp]
            )

    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (40, 255, 40), 2)
    pts = np.array(
        [[int(x0 + xi), int(y0 + yi)] for xi, yi in zip(sampled, sampled_y)],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
    args.output_overlay.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_overlay), overlay)

    print(
        f"ROI x0={x0} x1={x1} y0={y0} y1={y1}; "
        f"orange_ratio={orange_ratio:.3f}; coverage_ratio={coverage_ratio:.3f}; "
        f"wrote {len(sampled)} samples to {args.output_csv}"
    )
    print(f"Overlay written to {args.output_overlay}")


if __name__ == "__main__":
    main()
