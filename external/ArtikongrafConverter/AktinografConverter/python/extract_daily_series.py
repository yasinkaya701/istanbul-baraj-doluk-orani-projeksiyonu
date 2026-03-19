#!/usr/bin/env python3
import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from chart_models import ChartModel, get_models


@dataclass
class ExtractionResult:
    file_path: Path
    kind: str
    points: List[Tuple[float, float, float, str, str, float]]  # x, y, hour_float, hour_label, hour_precise, value
    alignment_cc: float
    valid_coverage: float
    trace_strength: float
    trace_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract radiation curve data from chart images.")
    parser.add_argument("--input-dir", default="data", help="Input folder containing chart images.")
    parser.add_argument("--output-dir", default="output", help="Output folder for csv files.")
    parser.add_argument("--diff-dir", default="diff", help="Output folder for side-by-side comparison PNGs.")
    parser.add_argument(
        "--clean-first",
        action="store_true",
        help="Delete all previous files under output-dir and diff-dir before processing.",
    )
    parser.add_argument(
        "--radiation-spike-margin-int",
        type=int,
        default=0,
        help=(
            "Radiation spike detector margin (int). "
            "No global smoothing is applied. "
            "Only obvious ink spikes are bridged. "
            "Larger margin makes detector less aggressive."
        ),
    )
    parser.add_argument("--debug-dir", default=None, help="Optional folder to write debug images.")
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.JPG", "*.JPEG", "*.PNG", "*.TIF", "*.TIFF"],
    )
    return parser.parse_args()


def load_and_align_to_template(
    image_bgr: np.ndarray,
    template_bgr: np.ndarray,
) -> Tuple[np.ndarray, float]:
    tpl_h, tpl_w = template_bgr.shape[:2]
    in_h, in_w = image_bgr.shape[:2]

    if (in_w, in_h) != (tpl_w, tpl_h):
        image_resized = cv2.resize(image_bgr, (tpl_w, tpl_h), interpolation=cv2.INTER_CUBIC)
    else:
        image_resized = image_bgr.copy()

    scale = min(1400.0 / tpl_w, 1.0)
    sw, sh = int(tpl_w * scale), int(tpl_h * scale)
    tpl_small = cv2.resize(template_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    img_small = cv2.resize(image_resized, (sw, sh), interpolation=cv2.INTER_AREA)

    tpl_gray = cv2.cvtColor(tpl_small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    tpl_gray = cv2.GaussianBlur(tpl_gray, (5, 5), 0)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    warp = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)

    try:
        cc, warp_small = cv2.findTransformECC(
            tpl_gray,
            img_gray,
            warp,
            cv2.MOTION_HOMOGRAPHY,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
    except cv2.error:
        # Fallback: sadece resize + no-warp.
        return image_resized, 0.0

    s = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
    s_inv = np.array([[1.0 / scale, 0, 0], [0, 1.0 / scale, 0], [0, 0, 1]], dtype=np.float32)
    warp_full = s_inv @ warp_small @ s

    aligned = cv2.warpPerspective(
        image_resized,
        warp_full,
        (tpl_w, tpl_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, float(cc)


def build_plot_roi(model: ChartModel, height: int, width: int) -> np.ndarray:
    xs = np.arange(width, dtype=np.float64)
    top = model.top_y(xs) - 12.0
    bottom = model.bottom_y(xs) + 12.0

    y_mid = 0.5 * (np.median(top) + np.median(bottom))
    left = float(model.left_hour_x(np.array([y_mid]))[0]) - model.hour_step_px * 0.7
    right = left + model.hour_step_px * (model.hour_slots + 1.4)
    left_i = int(max(0, np.floor(left)))
    right_i = int(min(width - 1, np.ceil(right)))

    ys = np.arange(height, dtype=np.float64)[:, None]
    roi = (ys >= top[None, :]) & (ys <= bottom[None, :])
    x_mask = np.zeros(width, dtype=bool)
    x_mask[left_i : right_i + 1] = True
    roi &= x_mask[None, :]
    return roi


def _hue_dist(h: np.ndarray, center: int) -> np.ndarray:
    d = np.abs(h.astype(np.int16) - int(center))
    return np.minimum(d, 180 - d).astype(np.float32)


def _dominant_pen_hue(h: np.ndarray, s: np.ndarray, v: np.ndarray, roi: np.ndarray, grid: np.ndarray, fallback: int) -> int:
    cand = roi & (~grid) & (s >= 16) & (v <= 245)
    if not np.any(cand):
        return fallback

    hue_vals = h[cand].astype(np.int32)
    weights = (s[cand].astype(np.float32) + 1.0)
    hist = np.bincount(hue_vals, weights=weights, minlength=180).astype(np.float32)
    if hist.max() <= 0:
        return fallback

    # 1D circular smoothing
    ext = np.concatenate([hist[-4:], hist, hist[:4]])
    kernel = np.ones(9, dtype=np.float32) / 9.0
    smooth = np.convolve(ext, kernel, mode="same")[4:-4]
    return int(np.argmax(smooth))


def detect_trace_score(aligned_bgr: np.ndarray, model: ChartModel) -> np.ndarray:
    hsv = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = h.astype(np.uint8)
    s_f = s.astype(np.float32)
    v_f = v.astype(np.float32)

    roi = build_plot_roi(model, aligned_bgr.shape[0], aligned_bgr.shape[1])
    grid = (h >= 4) & (h <= 35) & (s >= 14) & (v <= 240)

    peak_h = _dominant_pen_hue(h, s, v, roi, grid, fallback=145)
    dist = _hue_dist(h, peak_h)
    sigma = 12.0
    hue_score = np.exp(-(dist * dist) / (2.0 * sigma * sigma))
    sat_score = np.clip((s_f - 10.0) / 120.0, 0.0, 1.0)
    dark_score = np.clip((235.0 - v_f) / 190.0, 0.0, 1.0)
    purple_fallback = ((h >= 118) & (h <= 175) & (s >= 12) & (v <= 245)).astype(np.float32)
    color_score = hue_score * (0.65 * sat_score + 0.35 * dark_score)
    color_score = np.maximum(color_score, 0.55 * purple_fallback)

    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    ksize = 13
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((ksize, ksize), np.uint8))
    blackhat = cv2.GaussianBlur(blackhat.astype(np.float32) / 255.0, (0, 0), 1.1)

    score = 0.78 * color_score + 0.22 * blackhat

    score = score.astype(np.float32)
    score[grid] *= 0.08
    score[~roi] = 0.0
    score = cv2.GaussianBlur(score, (0, 0), 0.8)
    return np.clip(score, 0.0, 1.0)


def score_to_trace_image(score: np.ndarray) -> np.ndarray:
    return np.clip(score * 255.0, 0, 255).astype(np.uint8)


def trace_map_from_score_image(score_img: np.ndarray, model: ChartModel) -> np.ndarray:
    base = score_img.astype(np.float32) / 255.0
    nonzero = score_img[score_img > 0]
    if nonzero.size == 0:
        return base

    thr = int(np.clip(np.percentile(nonzero, 58.0), 20, 190))
    open_kernel = np.ones((3, 3), dtype=np.uint8)
    close_kernel = np.ones((5, 5), dtype=np.uint8)
    min_score = 8

    binary = (score_img >= thr).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    support = cv2.GaussianBlur(binary.astype(np.float32) / 255.0, (0, 0), 0.7)
    trace_map = base * np.clip(0.45 + 0.55 * support, 0.0, 1.0)
    trace_map[score_img < min_score] = 0.0
    return trace_map.astype(np.float32)


def extract_path_from_trace_score_image(
    score_img: np.ndarray,
    model: ChartModel,
    x_min: int,
    x_max: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    trace_map = trace_map_from_score_image(score_img, model)
    return trace_y_from_score(trace_map, model, x_min, x_max)


def _pick_column_candidates(col_scores: np.ndarray, y_offset: int, max_candidates: int, min_sep: int) -> Tuple[np.ndarray, np.ndarray]:
    if col_scores.size == 0:
        return np.array([float(y_offset)]), np.array([0.0], dtype=np.float32)

    k = min(max_candidates * 4, col_scores.size)
    top_idx = np.argpartition(col_scores, -k)[-k:]
    top_idx = top_idx[np.argsort(col_scores[top_idx])[::-1]]

    selected: List[int] = []
    for i in top_idx:
        ii = int(i)
        if all(abs(ii - s) >= min_sep for s in selected):
            selected.append(ii)
            if len(selected) >= max_candidates:
                break

    if not selected:
        selected = [int(np.argmax(col_scores))]

    idx = np.array(selected, dtype=np.int32)
    ys = y_offset + idx.astype(np.float64)
    ss = col_scores[idx].astype(np.float32)
    return ys, ss


def _median_filter_1d(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr.copy()
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    out = np.empty_like(arr)
    for i in range(arr.size):
        out[i] = np.median(padded[i : i + k])
    return out


def _wrap_hours_1_to_24(hours: np.ndarray) -> np.ndarray:
    # Converts any continuous hour value into [1, 24] range.
    return np.mod(hours - 1.0, 24.0) + 1.0


def _enforce_monotonic_hours(hours: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    if hours.size <= 1:
        return hours
    out = hours.copy()
    for i in range(1, out.size):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out


def _bridge_obvious_spikes(
    values: np.ndarray,
    hours: np.ndarray,
    active: np.ndarray,
    margin_int: int,
) -> np.ndarray:
    if values.size <= 5:
        return values

    out = values.copy()
    hmono = _enforce_monotonic_hours(hours, eps=1e-6)
    dt = np.maximum(np.diff(hmono), 1e-4)
    rate = np.diff(out) / dt

    # margin_int grows -> detector less aggressive.
    scale = 1.0 if margin_int <= 0 else max(0.6, min(2.2, float(margin_int) / 20.0))
    up_rate_thr = 12.0 * scale
    down_rate_thr = -1.5 * scale
    min_lift = 0.28 * scale
    jump_in_thr = 0.34 * scale
    jump_out_thr = 0.24 * scale
    max_block_hours = 0.9
    min_block_hours = 0.03

    n = out.size
    used = np.zeros(n, dtype=bool)
    i = 0
    while i < n - 2:
        # Candidate upward edge entering an active region.
        if rate[i] <= up_rate_thr or (not active[i + 1]) or ((out[i + 1] - out[i]) <= (0.22 * scale)):
            i += 1
            continue

        s = i + 1
        best_e = None
        j = s
        while j < n - 1:
            if (hmono[j] - hmono[s]) > max_block_hours:
                break
            if not active[j]:
                break
            if rate[j] < down_rate_thr or ((out[j] - out[j + 1]) > (0.03 * scale)):
                best_e = j
                break
            j += 1

        if best_e is None:
            i += 1
            continue

        e = int(best_e)
        if e - s + 1 < 2:
            i += 1
            continue
        if (hmono[e] - hmono[s]) < min_block_hours:
            i += 1
            continue
        if s <= 0 or e >= n - 1:
            i += 1
            continue
        if np.any(used[s : e + 1]):
            i += 1
            continue

        l0 = max(0, s - 10)
        r1 = min(n, e + 1)
        r0 = min(n, e + 9)
        left_ctx = out[l0:s]
        right_ctx = out[r1:r0]
        if right_ctx.size < 3:
            right_ctx = out[e + 1 : r0]
        if left_ctx.size < 3 or right_ctx.size < 3:
            i += 1
            continue

        left_med = float(np.median(left_ctx))
        right_med = float(np.median(right_ctx))
        left = float(out[s - 1])
        right = float(out[e + 1])
        block = out[s : e + 1]
        block_med = float(np.median(block))
        block_max = float(np.max(block))
        block_std = float(np.std(block))
        lift = block_med - 0.5 * (left_med + right_med)
        jump_in = block_max - left_med
        jump_out = block_max - right_med
        flat_like = block_std < 0.08
        sharp_two_edges = jump_in > jump_in_thr and jump_out > jump_out_thr

        # Only repair very explicit anomaly blocks.
        if lift > min_lift and (flat_like or sharp_two_edges):
            dur_h = max(float(hmono[e + 1] - hmono[s - 1]), 1e-4)
            max_delta = 1.0 * dur_h  # ~0.5 per 30 min
            right = float(np.clip(right, left - max_delta, left + max_delta))
            interp = np.linspace(left, right, (e - s + 3), dtype=np.float64)[1:-1]
            out[s : e + 1] = interp
            used[s : e + 1] = True
            i = e + 1
            continue

        i += 1

    return np.clip(out, 0.0, 2.0)


def _true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    split = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([0], split + 1))
    ends = np.concatenate((split, [idx.size - 1]))
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def _bridge_flat_ink_plateaus(values: np.ndarray, hours: np.ndarray, active: np.ndarray) -> np.ndarray:
    # Strict anomaly repair: only very high islands above local baseline are bridged.
    if values.size <= 8:
        return values

    out = values.copy()
    hmono = _enforce_monotonic_hours(hours, eps=1e-6)
    baseline = _median_filter_1d(out, 31)
    resid = out - baseline

    mask = active & (resid > 0.55)
    support = np.convolve(mask.astype(np.int32), np.ones(5, dtype=np.int32), mode="same")
    mask = active & (support >= 2)

    for s, e in _true_runs(mask):
        if s <= 0 or e >= out.size - 2:
            continue
        block_len = e - s + 1
        if block_len < 2 or block_len > 90:
            continue
        if (hmono[e] - hmono[s]) > 1.0:
            continue

        left_ctx = out[max(0, s - 8) : s]
        right_ctx = out[e + 1 : min(out.size, e + 9)]
        if left_ctx.size < 3 or right_ctx.size < 3:
            continue

        left_med = float(np.median(left_ctx))
        right_med = float(np.median(right_ctx))
        block_max = float(np.max(out[s : e + 1]))
        if (block_max - max(left_med, right_med)) < 0.5:
            continue

        left_anchor = float(out[s - 1])
        right_anchor = float(out[e + 1])
        dur_h = max(float(hmono[e + 1] - hmono[s - 1]), 1e-4)
        max_delta = 1.2 * dur_h
        right_anchor = float(np.clip(right_anchor, left_anchor - max_delta, left_anchor + max_delta))

        interp = np.linspace(left_anchor, right_anchor, block_len + 2, dtype=np.float64)[1:-1]
        out[s : e + 1] = interp

    return np.clip(out, 0.0, 2.0)


def _bridge_large_steps(values: np.ndarray, hours: np.ndarray, active: np.ndarray) -> np.ndarray:
    # Target only extreme minute-level jumps, then bridge between neighboring stable parts.
    if values.size <= 4:
        return values

    out = values.copy()
    hmono = _enforce_monotonic_hours(hours, eps=1e-6)
    n = out.size
    max_step = 0.30
    return_tol = 0.25
    max_window_h = 1.0

    i = 1
    while i < n:
        if not (active[i - 1] and active[i]):
            i += 1
            continue

        step = out[i] - out[i - 1]
        if step <= max_step:
            i += 1
            continue

        left = float(out[i - 1])
        j = i
        found = False
        while j < n and active[j] and (hmono[j] - hmono[i]) <= max_window_h:
            if out[j] <= (left + return_tol):
                found = True
                break
            j += 1

        if found and j > i:
            right = float(out[j])
            interp = np.linspace(left, right, (j - i + 2), dtype=np.float64)[1:-1]
            out[i:j] = interp
            i = j
            continue

        # No quick return found: convert the sudden jump into a physically plausible ramp
        # for the next 1 hour window.
        horizon = i
        while horizon < n - 1 and active[horizon + 1] and (hmono[horizon + 1] - hmono[i]) <= max_window_h:
            horizon += 1
        if horizon > i:
            dur_h = max(float(hmono[horizon] - hmono[i - 1]), 1e-4)
            right_allowed = left + 1.0 * dur_h  # ~0.5 per 30 min
            right = float(min(out[horizon], right_allowed))
            interp = np.linspace(left, right, (horizon - i + 3), dtype=np.float64)[1:-1]
            out[i : horizon + 1] = interp
            i = horizon + 1
            continue

        # If no return before inactive boundary, bridge spike tail to a low end anchor.
        end = i
        while end < n - 1 and active[end + 1]:
            end += 1
        if (hmono[end] - hmono[i]) <= max_window_h:
            right = float(min(out[end], left + 0.15))
            interp = np.linspace(left, right, (end - i + 3), dtype=np.float64)[1:-1]
            out[i : end + 1] = interp
            i = end + 1
            continue

        i += 1

    return np.clip(out, 0.0, 2.0)


def postprocess_radiation_values(hours_cont: np.ndarray, values: np.ndarray, spike_margin_int: int) -> np.ndarray:
    if values.size == 0:
        return values

    vals = np.clip(values.astype(np.float64), 0.0, 2.0)
    hours = hours_cont.astype(np.float64)
    hours_wrapped = _wrap_hours_1_to_24(hours)
    active_window = (hours_wrapped >= 4.0) & (hours_wrapped < 20.0)
    vals[~active_window] = 0.0
    vals = _bridge_obvious_spikes(vals, hours, active_window, spike_margin_int)
    vals = _bridge_large_steps(vals, hours, active_window)
    vals[~active_window] = 0.0
    return vals


def trace_y_from_score(score: np.ndarray, model: ChartModel, x_min: int, x_max: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    h, w = score.shape[:2]
    x_min = max(0, x_min)
    x_max = min(w - 1, x_max)
    xs = np.arange(x_min, x_max + 1, dtype=np.int32)

    top = model.top_y(xs.astype(np.float64)) - 8.0
    bottom = model.bottom_y(xs.astype(np.float64)) + 8.0

    max_candidates = 8
    smooth_w = 0.055
    min_sep = 3

    cand_y: List[np.ndarray] = []
    cand_s: List[np.ndarray] = []

    for i, x in enumerate(xs):
        y0 = max(0, int(np.floor(top[i])))
        y1 = min(h - 1, int(np.ceil(bottom[i])))
        if y1 < y0:
            y0, y1 = 0, h - 1

        col = score[y0 : y1 + 1, x]
        ys, ss = _pick_column_candidates(col, y0, max_candidates=max_candidates, min_sep=min_sep)
        cand_y.append(ys)
        cand_s.append(ss)

    if not cand_y:
        return xs.astype(np.float64), np.array([], dtype=np.float64), 0.0, 0.0

    prev_y = cand_y[0]
    prev_cost = (1.0 - cand_s[0]) * 3.0
    back_ptr: List[np.ndarray] = []

    for i in range(1, len(xs)):
        cur_y = cand_y[i]
        cur_s = cand_s[i]
        dy = np.abs(cur_y[:, None] - prev_y[None, :])
        transition = prev_cost[None, :] + smooth_w * np.minimum(dy, 28.0) + 0.015 * np.maximum(dy - 28.0, 0.0)
        bp = np.argmin(transition, axis=1)
        cur_cost = (1.0 - cur_s) * 3.0 + transition[np.arange(cur_y.size), bp]
        back_ptr.append(bp)
        prev_y = cur_y
        prev_cost = cur_cost

    idx = int(np.argmin(prev_cost))
    y_path = np.empty(len(xs), dtype=np.float64)
    s_path = np.empty(len(xs), dtype=np.float32)
    y_path[-1] = cand_y[-1][idx]
    s_path[-1] = cand_s[-1][idx]

    for i in range(len(xs) - 2, -1, -1):
        idx = int(back_ptr[i][idx])
        y_path[i] = cand_y[i][idx]
        s_path[i] = cand_s[i][idx]

    y_path = _median_filter_1d(y_path, 5)
    y_path = np.convolve(y_path, np.ones(5, dtype=np.float64) / 5.0, mode="same")

    coverage = float((s_path > 0.10).mean())
    strength = float(s_path.mean())
    return xs.astype(np.float64), y_path, coverage, strength


def wrap_hour(hour: float) -> float:
    h = float(hour)
    while h > 24.0:
        h -= 24.0
    while h <= 0.0:
        h += 24.0
    return h


def format_minute_hour(hour_cont: float) -> str:
    h = float(_wrap_hours_1_to_24(np.array([hour_cont], dtype=np.float64))[0])
    hh = int(np.floor(h + 1e-9))
    mm = int(round((h - hh) * 60.0))
    if mm == 60:
        mm = 0
        hh += 1
    while hh > 24:
        hh -= 24
    while hh <= 0:
        hh += 24
    return f"{hh}.{mm:02d}"


def _wrap_hour_label(hour_cont: float) -> str:
    h = int(round(hour_cont))
    while h > 24:
        h -= 24
    while h <= 0:
        h += 24
    return str(h)


def _draw_chart_from_points(
    canvas: np.ndarray,
    points: List[Tuple[float, float, float, str, str, float]],
    model: ChartModel,
) -> np.ndarray:
    h, w = canvas.shape[:2]
    m_left, m_right, m_top, m_bottom = 90, 40, 40, 70
    x0, y0 = m_left, h - m_bottom
    x1, y1 = w - m_right, m_top

    cv2.rectangle(canvas, (x0, y1), (x1, y0), (220, 220, 220), 1)

    x_min = float(model.hour_start)
    x_max = float(model.hour_start + model.hour_slots)
    y_min = float(model.value_min)
    y_max = float(model.value_max)

    def x_to_px(xv: float) -> int:
        t = 0.0 if x_max == x_min else (xv - x_min) / (x_max - x_min)
        return int(round(x0 + t * (x1 - x0)))

    def y_to_px(yv: float) -> int:
        t = 0.0 if y_max == y_min else (yv - y_min) / (y_max - y_min)
        return int(round(y0 - t * (y0 - y1)))

    # X ticks
    x_tick_step = 2 if (x_max - x_min) > 18 else 1
    tick = int(np.floor(x_min))
    while tick <= int(np.ceil(x_max)):
        px = x_to_px(float(tick))
        cv2.line(canvas, (px, y0), (px, y0 + 5), (150, 150, 150), 1)
        cv2.putText(
            canvas,
            _wrap_hour_label(float(tick)),
            (px - 12, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        tick += x_tick_step

    # Y ticks
    y_step = 0.2
    y_tick = y_min
    while y_tick <= y_max + 1e-9:
        py = y_to_px(y_tick)
        cv2.line(canvas, (x0 - 5, py), (x0, py), (150, 150, 150), 1)
        label = f"{y_tick:.1f}"
        cv2.putText(
            canvas,
            label,
            (x0 - 45, py + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        y_tick += y_step

    # Data line
    if points:
        arr = np.array([[p[2], p[5]] for p in points], dtype=np.float64)
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        poly = np.array([[x_to_px(float(r[0])), y_to_px(float(r[1]))] for r in arr], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], isClosed=False, color=(170, 40, 180), thickness=2)

    cv2.putText(
        canvas,
        "Reconstructed Series",
        (x0, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Hour",
        ((x0 + x1) // 2 - 20, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    y_name = "Radiation"
    cv2.putText(
        canvas,
        y_name,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    return canvas


def resample_to_minute_radiation(
    points: List[Tuple[float, float, float, str, str, float]],
    spike_margin_int: int,
) -> List[Tuple[float, float, float, str, str, float]]:
    if not points:
        return []

    arr = np.array([[p[0], p[1], p[2], p[5]] for p in points], dtype=np.float64)
    order = np.argsort(arr[:, 2])
    arr = arr[order]

    x_src = arr[:, 0]
    y_src = arr[:, 1]
    h_src = _enforce_monotonic_hours(arr[:, 2], eps=1e-6)
    v_src = np.clip(arr[:, 3], 0.0, 2.0)

    # Day timeline: 20.00 -> 19.59 (exactly 1440 minute points).
    h_dst = 20.0 + (np.arange(24 * 60, dtype=np.float64) / 60.0)
    x_dst = np.interp(h_dst, h_src, x_src, left=x_src[0], right=x_src[-1])
    y_dst = np.interp(h_dst, h_src, y_src, left=y_src[0], right=y_src[-1])
    v_dst = np.interp(h_dst, h_src, v_src, left=0.0, right=0.0)

    v_dst = postprocess_radiation_values(h_dst, v_dst, spike_margin_int)
    v_dst = np.clip(v_dst, 0.0, 2.0)

    out: List[Tuple[float, float, float, str, str, float]] = []
    for x, y, h, val in zip(x_dst, y_dst, h_dst, v_dst):
        h_wrap = float(_wrap_hours_1_to_24(np.array([h], dtype=np.float64))[0])
        label = str(int(np.floor(h_wrap + 1e-9)))
        out.append((float(x), float(y), float(h), label, format_minute_hour(float(h)), float(val)))
    return out


def save_diff_panel(
    src_path: Path,
    points: List[Tuple[float, float, float, str, str, float]],
    model: ChartModel,
    out_path: Path,
) -> None:
    src = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if src is None:
        return

    h, w = src.shape[:2]
    right = np.full((h, w, 3), 250, dtype=np.uint8)
    right = _draw_chart_from_points(right, points, model)

    panel = np.full((h, w * 2 + 8, 3), 235, dtype=np.uint8)
    panel[:, :w] = src
    panel[:, w + 8 :] = right
    cv2.line(panel, (w + 4, 0), (w + 4, h - 1), (180, 180, 180), 1)

    cv2.putText(
        panel,
        f"{src_path.name} | kind={model.name}",
        (14, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def extract_points_for_model(
    aligned_bgr: np.ndarray,
    model: ChartModel,
    radiation_spike_margin_int: int,
) -> Tuple[List[Tuple[float, float, float, str, str, float]], float, float, np.ndarray]:
    h, w = aligned_bgr.shape[:2]
    y_mid = h * 0.5
    left = float(model.left_hour_x(np.array([y_mid]))[0]) - model.hour_step_px * 0.6
    right = left + model.hour_step_px * (model.hour_slots + 1.2)
    x_min = int(max(0, np.floor(left)))
    x_max = int(min(w - 1, np.ceil(right)))

    score = detect_trace_score(aligned_bgr, model)
    score_img = score_to_trace_image(score)
    xs, ys, coverage, strength = extract_path_from_trace_score_image(score_img, model, x_min, x_max)
    if ys.size < 50:
        return [], coverage, strength, score_img

    xs_v = xs
    ys_v = ys
    raw_values = model.value_from_xy(xs_v, ys_v)
    hours_cont = model.hour_continuous(xs_v, ys_v, raw_values)
    if model.name != "radiation":
        raise RuntimeError("This extractor is configured for radiation only.")

    hours_cont = _enforce_monotonic_hours(hours_cont)
    values = np.clip(raw_values.astype(np.float64), 0.0, 2.0)
    slots_rounded = np.round(hours_cont - model.hour_start).astype(int)
    slots_rounded = np.clip(slots_rounded, 0, model.hour_slots)
    labels = [model.hour_label_from_slot(int(s)) for s in slots_rounded]
    precise = [format_minute_hour(float(h)) for h in hours_cont]

    points = []
    for x, y, h_cont, lbl, h_prec, val in zip(xs_v, ys_v, hours_cont, labels, precise, values):
        points.append((float(x), float(y), float(h_cont), lbl, h_prec, float(val)))
    points = resample_to_minute_radiation(points, radiation_spike_margin_int)
    return points, coverage, strength, score_img


def write_csv(path: Path, rows: List[Dict[str, object]], headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def save_debug_images(
    debug_dir: Path,
    image_path: Path,
    aligned: np.ndarray,
    model: ChartModel,
    score_img: np.ndarray,
    points: List[Tuple[float, float, float, str, str, float]],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    overlay = aligned.copy()
    if points:
        poly = np.array([[int(round(p[0])), int(round(p[1]))] for p in points], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [poly], isClosed=False, color=(0, 255, 0), thickness=2)

    cv2.imwrite(str(debug_dir / f"{stem}_aligned.png"), aligned)
    cv2.imwrite(str(debug_dir / f"{stem}_trace_score.png"), score_img)
    cv2.imwrite(str(debug_dir / f"{stem}_overlay.png"), overlay)


def process_file(
    file_path: Path,
    models: Dict[str, ChartModel],
    debug_dir: Optional[Path],
    radiation_spike_margin_int: int,
) -> ExtractionResult:
    image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {file_path}")

    kind = "radiation"
    model = models[kind]

    template = cv2.imread(str(model.template_path), cv2.IMREAD_COLOR)
    if template is None:
        raise RuntimeError(f"Template not found: {model.template_path}")

    aligned, cc = load_and_align_to_template(image, template)
    points, coverage, strength, score_img = extract_points_for_model(aligned, model, radiation_spike_margin_int)

    if debug_dir is not None:
        save_debug_images(debug_dir, file_path, aligned, model, score_img, points)

    return ExtractionResult(
        file_path=file_path,
        kind=kind,
        points=points,
        alignment_cc=cc,
        valid_coverage=coverage,
        trace_strength=strength,
        trace_source="trace_score_png",
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    input_dir = (project_root / args.input_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    diff_dir = (project_root / args.diff_dir).resolve()
    debug_dir = Path(args.debug_dir).resolve() if args.debug_dir else None

    if args.clean_first:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if diff_dir.exists():
            shutil.rmtree(diff_dir)

    models = get_models(project_root)

    files: List[Path] = []
    for ext in args.extensions:
        files.extend(sorted(input_dir.glob(ext)))
    files = sorted(set(files))

    if not files:
        raise RuntimeError(f"No files found in: {input_dir}")

    all_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for fp in files:
        try:
            result = process_file(fp, models, debug_dir, args.radiation_spike_margin_int)
            model = models[result.kind]

            # Side-by-side diff image: left=input, right=reconstructed graph
            save_diff_panel(fp, result.points, model, diff_dir / f"{fp.stem}_diff.png")

            image_rows: List[Dict[str, object]] = []
            for x, y, hour_float, hour_label, hour_precise, value in result.points:
                row = {
                    "source_file": fp.name,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "hour_precise": hour_precise,
                    "radiation": round(value, 4),
                }
                image_rows.append(row)
                all_rows.append(row)

            if image_rows:
                headers = list(image_rows[0].keys())
                write_csv(output_dir / f"{fp.stem}_series.csv", image_rows, headers)

            summary_rows.append(
                {
                    "source_file": fp.name,
                    "kind": result.kind,
                    "points": len(result.points),
                    "alignment_cc": round(result.alignment_cc, 6),
                    "valid_coverage": round(result.valid_coverage, 6),
                    "trace_strength": round(result.trace_strength, 6),
                    "trace_source": result.trace_source,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as e:
            summary_rows.append(
                {
                    "source_file": fp.name,
                    "kind": "",
                    "points": 0,
                    "alignment_cc": 0.0,
                    "valid_coverage": 0.0,
                    "trace_strength": 0.0,
                    "trace_source": "",
                    "status": "error",
                    "error": str(e),
                }
            )

    if all_rows:
        headers = list(all_rows[0].keys())
        write_csv(output_dir / "all_series.csv", all_rows, headers)

    ok_count = sum(1 for r in summary_rows if r.get("status") == "ok")
    err_count = sum(1 for r in summary_rows if r.get("status") == "error")
    print(f"Processed {len(files)} image(s): ok={ok_count}, error={err_count}")
    print(f"Clean first: {args.clean_first}")
    print(
        "Radiation spike margin int: "
        f"{args.radiation_spike_margin_int} "
        "(no global smoothing; only obvious spike-bridging, larger=less aggressive)"
    )
    print(f"Output: {output_dir}")
    print(f"Diff: {diff_dir}")


if __name__ == "__main__":
    main()
