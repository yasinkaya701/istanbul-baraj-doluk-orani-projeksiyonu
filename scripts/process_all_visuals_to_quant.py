#!/usr/bin/env python3
"""Process all chart images and build lossless + quant-ready observation tables.

Pipeline:
1) Scan all image files under graph root.
2) Extract a robust trace-like signal from each image.
3) Convert signal to daily numeric observation by inferred variable.
4) Merge with numeric observations (prefer numeric on same variable/day).
5) Save bronze/silver/gold layers:
   - bronze: per-image processing manifest
   - silver: lossless merged observations (numeric + all visual extractions)
   - gold: model-ready (temp/humidity/pressure/precip) with day-level numeric priority
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


TR_MAP = str.maketrans("cgiosuiCGIOSUI", "cgiosuiCGIOSUI")
TR_CHARS = str.maketrans(
    {
        "c": "c",
        "g": "g",
        "i": "i",
        "o": "o",
        "s": "s",
        "u": "u",
        "C": "C",
        "G": "G",
        "I": "I",
        "O": "O",
        "S": "S",
        "U": "U",
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "Ç": "C",
        "Ğ": "G",
        "İ": "I",
        "Ö": "O",
        "Ş": "S",
        "Ü": "U",
    }
)

MONTH_MAP = {
    "ocak": 1,
    "january": 1,
    "jan": 1,
    "subat": 2,
    "sub": 2,
    "february": 2,
    "feb": 2,
    "mart": 3,
    "march": 3,
    "mar": 3,
    "nisan": 4,
    "april": 4,
    "apr": 4,
    "mayis": 5,
    "may": 5,
    "haziran": 6,
    "june": 6,
    "jun": 6,
    "temmuz": 7,
    "july": 7,
    "jul": 7,
    "agustos": 8,
    "august": 8,
    "aug": 8,
    "eylul": 9,
    "september": 9,
    "sep": 9,
    "ekim": 10,
    "october": 10,
    "oct": 10,
    "kasim": 11,
    "november": 11,
    "nov": 11,
    "aralik": 12,
    "december": 12,
    "dec": 12,
}

VAR_RANGES = {
    "humidity": (0.0, 100.0),
    "temp": (-30.0, 50.0),
    "pressure": (980.0, 1045.0),
    "precip": (0.0, 80.0),
    "wind_speed": (0.0, 35.0),
    "wind_dir": (0.0, 360.0),
    "solar": (0.0, 14.0),
    "unknown": (0.0, 1.0),
}

MODEL_VARS = {"temp", "humidity", "pressure", "precip"}


@dataclass
class ExtractResult:
    ok: bool
    message: str
    y_norm_median: float
    y_norm_std: float
    coverage: float
    quality: float
    width_px: int
    height_px: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process all graph images and build lossless + quant inputs.")
    p.add_argument(
        "--graph-root",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/DATA/Graf Kağıtları Tarama "),
        help="Root directory of graph-paper images.",
    )
    p.add_argument(
        "--numeric-parquet",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/sample/observations_numeric.parquet"),
        help="Numeric observation parquet path (from ingest_numeric_and_plot.py).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/quant_all_visuals_input"),
        help="Output directory.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap for debug (0 means all).",
    )
    p.add_argument(
        "--image-exts",
        type=str,
        default=".tif,.tiff,.png,.jpg,.jpeg,.bmp,.webp",
        help="Comma-separated image extensions to include.",
    )
    return p.parse_args()


def norm_text(x: Any) -> str:
    s = str(x).strip().lower().translate(TR_CHARS)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def infer_variable(path: Path) -> str:
    t = norm_text(str(path))
    if any(k in t for k in ["nem", "humidity", "relative humidity", " rh "]):
        return "humidity"
    if any(k in t for k in ["sicaklik", "sicak", "temperature", "temp", "termogram"]):
        return "temp"
    if any(k in t for k in ["basinc", "pressure", "hpa"]):
        return "pressure"
    if any(k in t for k in ["yagis", "precip", "rain"]):
        return "precip"
    if "ruzgar" in t and any(k in t for k in ["hiz", "speed"]):
        return "wind_speed"
    if "ruzgar" in t and any(k in t for k in ["yon", "direction"]):
        return "wind_dir"
    if any(k in t for k in ["aktinograf", "helyograf", "gunes", "sun", "radiation"]):
        return "solar"
    return "unknown"


def infer_month(path: Path) -> int | None:
    t = norm_text(str(path))
    for k, mm in MONTH_MAP.items():
        if re.search(rf"\b{re.escape(k)}\b", t):
            return mm
    return None


def parse_year(path: Path) -> int | None:
    m = re.search(r"(18|19|20)\d{2}", str(path))
    if not m:
        return None
    return int(m.group(0))


def parse_day(path: Path) -> int | None:
    name = path.stem
    m = re.search(r"[-_](\d{1,3})$", name)
    if not m:
        return None
    d = int(m.group(1))
    if 1 <= d <= 31:
        return d
    return None


def infer_date(path: Path, fallback_seq_day: int) -> tuple[pd.Timestamp | None, str]:
    y = parse_year(path)
    m = infer_month(path)
    d = parse_day(path)

    if y is not None and m is not None and d is not None:
        try:
            return pd.Timestamp(year=y, month=m, day=d), "exact"
        except ValueError:
            pass
    if y is not None and m is not None:
        # Keep month fixed, assign sequential day with month bounds.
        d2 = int(np.clip(fallback_seq_day, 1, 31))
        while d2 >= 1:
            try:
                return pd.Timestamp(year=y, month=m, day=d2), "estimated_day"
            except ValueError:
                d2 -= 1
        return pd.Timestamp(year=y, month=m, day=1), "estimated_day"
    if y is not None:
        # No month/day info: spread by day-of-year to avoid timestamp collapse.
        doy = int(((max(1, fallback_seq_day) - 1) % 365) + 1)
        ts = pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=doy - 1)
        return ts, "estimated_doy"
    return None, "missing"


def load_image_rgb(path: Path) -> np.ndarray:
    pil_img = Image.open(path)
    if "A" in pil_img.getbands():
        rgba = pil_img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(bg, rgba).convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img)


def resize_rgb(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    pil = Image.fromarray(img)
    pil = pil.resize((int(new_w), int(new_h)), resample=Image.Resampling.BILINEAR)
    return np.array(pil)


def rgb_to_gray01(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return np.asarray(arr, dtype=np.float32) / 255.0
    gray = (0.299 * arr[:, :, 0]) + (0.587 * arr[:, :, 1]) + (0.114 * arr[:, :, 2])
    return np.clip(gray / 255.0, 0.0, 1.0)


def rgb_to_hsv01(img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(img).convert("HSV")
    hsv = np.array(pil).astype(np.float32)
    h = hsv[:, :, 0] / 255.0
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    return np.stack([h, s, v], axis=2)


def box_blur2d(arr: np.ndarray, k: int = 5) -> np.ndarray:
    k = int(max(1, k))
    if k == 1:
        return arr
    if arr.ndim != 2:
        return arr
    ker = np.ones(k, dtype=np.float32) / float(k)
    tmp = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), axis=1, arr=arr)
    out = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), axis=0, arr=tmp)
    return out.astype(np.float32)


def crop_core(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = int(max(0, round(0.03 * w)))
    x1 = int(min(w, round(0.97 * w)))
    y0 = int(max(0, round(0.08 * h)))
    y1 = int(min(h, round(0.97 * h)))
    c = img[y0:y1, x0:x1]
    if c.size == 0:
        return img
    return c


def extract_signal(path: Path) -> ExtractResult:
    try:
        rgb = load_image_rgb(path)
    except Exception as exc:
        return ExtractResult(False, f"read_error:{exc}", np.nan, np.nan, 0.0, 0.0, 0, 0)

    core = crop_core(rgb)
    h, w = core.shape[:2]
    if h < 20 or w < 20:
        return ExtractResult(False, "image_too_small", np.nan, np.nan, 0.0, 0.0, w, h)

    # Keep extraction cheap enough to scale to hundreds of archival images.
    max_dim = 420
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        core = resize_rgb(core, new_w=int(w * scale), new_h=int(h * scale))
        h, w = core.shape[:2]

    if cv2 is not None:
        hsv = cv2.cvtColor(core, cv2.COLOR_RGB2HSV).astype(np.float32)
        gray = cv2.cvtColor(core, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        hch = hsv[:, :, 0] / 179.0
        sch = hsv[:, :, 1] / 255.0
        vch = hsv[:, :, 2] / 255.0
    else:
        hsv01 = rgb_to_hsv01(core)
        gray = rgb_to_gray01(core)
        hch = hsv01[:, :, 0]
        sch = hsv01[:, :, 1]
        vch = hsv01[:, :, 2]

    darkness = 1.0 - gray
    orange = ((hch >= 0.02) & (hch <= 0.14) & (sch > 0.20) & (vch > 0.15)).astype(np.float32)
    score = (0.95 * darkness) + (0.55 * sch) - (0.80 * orange)
    score = np.clip(score, 0.0, 1.0)
    if cv2 is not None:
        score = cv2.GaussianBlur(score, (3, 3), 0)
    else:
        score = box_blur2d(score, k=3)

    ncols = min(w, 120)
    x_idx = np.linspace(0, w - 1, num=ncols, dtype=int)
    seed_cols = x_idx[: min(12, len(x_idx))]
    y_seed = int(np.argmax(np.mean(score[:, seed_cols], axis=1)))

    jump = max(4, int(round(h * 0.08)))
    thr = float(np.quantile(score, 0.65))
    thr_low = max(0.05, 0.35 * thr)

    ys: list[int] = []
    strong: list[float] = []
    last = y_seed
    for xi in x_idx:
        y0 = max(0, last - jump)
        y1 = min(h, last + jump + 1)
        local = score[y0:y1, xi]
        if local.size == 0:
            ys.append(last)
            strong.append(0.0)
            continue
        li = int(np.argmax(local))
        best = float(local[li])
        y_best = y0 + li
        if best < thr_low:
            y_best = int(np.argmax(score[:, xi]))
            best = float(score[y_best, xi])
        ys.append(y_best)
        strong.append(best)
        last = y_best

    y_arr = np.asarray(ys, dtype=float)
    if len(y_arr) >= 5:
        kernel = np.ones(5, dtype=float) / 5.0
        y_arr = np.convolve(y_arr, kernel, mode="same")

    y_norm = 1.0 - (y_arr / max(1.0, h - 1.0))
    y_norm = np.clip(y_norm, 0.0, 1.0)

    coverage = float(np.mean(np.asarray(strong) >= thr_low))
    y_std = float(np.nanstd(y_norm))
    y_med = float(np.nanmedian(y_norm))
    quality = float(np.clip((0.70 * coverage) + (0.30 * min(1.0, y_std * 4.0)), 0.0, 1.0))

    if not np.isfinite(y_med):
        return ExtractResult(False, "nan_signal", np.nan, np.nan, coverage, quality, w, h)
    if coverage < 0.10:
        return ExtractResult(False, "low_coverage", y_med, y_std, coverage, quality, w, h)

    return ExtractResult(True, "ok", y_med, y_std, coverage, quality, w, h)


def map_value(y_norm: float, variable: str) -> float:
    lo, hi = VAR_RANGES.get(variable, VAR_RANGES["unknown"])
    return float(lo + (hi - lo) * float(y_norm))


def qc_flag(variable: str, value: float) -> str:
    if pd.isna(value):
        return "missing"
    if variable == "humidity" and not (0.0 <= value <= 100.0):
        return "range_fail"
    if variable == "temp" and not (-60.0 <= value <= 70.0):
        return "range_fail"
    if variable == "pressure" and not (850.0 <= value <= 1100.0):
        return "range_fail"
    if variable == "precip" and value < 0.0:
        return "range_fail"
    return "ok"


def canonical_variable(x: Any) -> str:
    t = norm_text(x)
    if any(k in t for k in ["humidity", "nem", "rh"]):
        return "humidity"
    if any(k in t for k in ["temp", "sicak", "temperature", "termogram"]):
        return "temp"
    if any(k in t for k in ["pressure", "basinc", "hpa"]):
        return "pressure"
    if any(k in t for k in ["precip", "rain", "yagis"]):
        return "precip"
    return t if t else "unknown"


def load_numeric(numeric_parquet: Path) -> pd.DataFrame:
    raw = pd.read_parquet(numeric_parquet)
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["timestamp"], errors="coerce"),
            "variable": raw["variable"].map(canonical_variable),
            "value": pd.to_numeric(raw["value"], errors="coerce"),
            "qc_flag": raw.get("qc_flag", "ok").astype(str) if "qc_flag" in raw.columns else "ok",
            "source_kind": raw.get("source_kind", "numeric"),
            "source_file": raw.get("source_file", "").astype(str) if "source_file" in raw.columns else "",
            "method": raw.get("method", "numeric_ingest").astype(str) if "method" in raw.columns else "numeric_ingest",
            "confidence": pd.to_numeric(raw.get("confidence", 0.95), errors="coerce").fillna(0.95),
        }
    )
    out = out.dropna(subset=["timestamp", "variable", "value"])
    out["qc_flag"] = [qc_flag(v, x) if str(q).strip() == "" else q for v, x, q in zip(out["variable"], out["value"], out["qc_flag"])]
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {x.strip().lower() for x in str(args.image_exts).split(",") if x.strip()}
    files = sorted([p for p in args.graph_root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No image files found under: {args.graph_root} for extensions={sorted(exts)}")

    folder_order: dict[str, int] = {}
    process_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []

    for p in files:
        k = str(p.parent)
        folder_order[k] = folder_order.get(k, 0) + 1
        seq_day = folder_order[k]

        variable = infer_variable(p)
        date_ts, date_quality = infer_date(p, fallback_seq_day=seq_day)
        ext = extract_signal(p)
        val = map_value(ext.y_norm_median, variable) if np.isfinite(ext.y_norm_median) else np.nan

        process_rows.append(
            {
                "file": str(p),
                "variable_inferred": variable,
                "timestamp_date": str(date_ts.date()) if date_ts is not None else "",
                "date_quality": date_quality,
                "extract_ok": bool(ext.ok),
                "extract_message": ext.message,
                "coverage": ext.coverage,
                "quality": ext.quality,
                "y_norm_median": ext.y_norm_median,
                "y_norm_std": ext.y_norm_std,
                "value_mapped": val,
                "width_px": ext.width_px,
                "height_px": ext.height_px,
            }
        )

        # Keep all visuals in measurement table, even if variable is not in model set.
        if date_ts is None or not np.isfinite(val):
            continue
        ts = date_ts + pd.Timedelta(hours=12)
        visual_rows.append(
            {
                "timestamp": ts,
                "variable": variable,
                "value": float(val),
                "qc_flag": qc_flag(variable, float(val)),
                "source_kind": "graph_paper_digitized",
                "source_file": str(p),
                "method": "image_trace_daily_median",
                "confidence": float(np.clip(0.30 + 0.60 * ext.quality, 0.05, 0.95)),
                "date_quality": date_quality,
                "extract_ok": bool(ext.ok),
            }
        )

    report_df = pd.DataFrame(process_rows)
    visual_all = pd.DataFrame(visual_rows)
    if not visual_all.empty:
        visual_all = visual_all.sort_values(["timestamp", "variable"]).reset_index(drop=True)

    visual_model = visual_all[visual_all["variable"].isin(sorted(MODEL_VARS))].copy() if not visual_all.empty else pd.DataFrame()

    if not args.numeric_parquet.exists():
        raise SystemExit(
            f"Numeric parquet not found: {args.numeric_parquet}. Run scripts/ingest_numeric_and_plot.py first."
        )
    numeric_all = load_numeric(args.numeric_parquet)
    numeric_model = numeric_all[numeric_all["variable"].isin(sorted(MODEL_VARS))].copy()

    if not visual_model.empty:
        num_day = numeric_model.assign(day=numeric_model["timestamp"].dt.floor("D"))[["variable", "day"]].drop_duplicates()
        vis_day = visual_model.assign(day=visual_model["timestamp"].dt.floor("D"))
        vis_merge = vis_day.merge(num_day.assign(has_num=1), on=["variable", "day"], how="left")
        visual_keep = vis_merge[vis_merge["has_num"].isna()].drop(columns=["day", "has_num"])
    else:
        visual_keep = pd.DataFrame(columns=numeric_model.columns)

    # Silver (lossless): keep all numeric + all extracted visual rows.
    silver = pd.concat(
        [
            numeric_all[["timestamp", "variable", "value", "qc_flag", "source_kind", "source_file", "method", "confidence"]],
            visual_all[["timestamp", "variable", "value", "qc_flag", "source_kind", "source_file", "method", "confidence"]],
        ],
        ignore_index=True,
    )
    silver = silver.dropna(subset=["timestamp", "variable", "value"]).sort_values(["timestamp", "variable"]).reset_index(drop=True)

    # Gold (model-ready): keep model vars and prefer numeric on same day-variable.
    gold = pd.concat(
        [
            numeric_model[["timestamp", "variable", "value", "qc_flag", "source_kind", "source_file", "method", "confidence"]],
            visual_keep[["timestamp", "variable", "value", "qc_flag", "source_kind", "source_file", "method", "confidence"]],
        ],
        ignore_index=True,
    )
    gold = gold.dropna(subset=["timestamp", "variable", "value"]).sort_values(["timestamp", "variable"]).reset_index(drop=True)

    # Save outputs.
    report_csv = out_dir / "visual_process_report.csv"
    report_pq = out_dir / "visual_process_report.parquet"
    visual_all_csv = out_dir / "visual_measurements_all.csv"
    visual_all_pq = out_dir / "visual_measurements_all.parquet"
    visual_model_csv = out_dir / "visual_model_observations.csv"
    visual_model_pq = out_dir / "visual_model_observations.parquet"
    silver_csv = out_dir / "observations_lossless_silver.csv"
    silver_pq = out_dir / "observations_lossless_silver.parquet"
    gold_csv = out_dir / "observations_with_all_visuals_for_quant.csv"
    gold_pq = out_dir / "observations_with_all_visuals_for_quant.parquet"

    report_df.to_csv(report_csv, index=False)
    report_df.to_parquet(report_pq, index=False)
    visual_all.to_csv(visual_all_csv, index=False)
    visual_all.to_parquet(visual_all_pq, index=False)
    visual_model.to_csv(visual_model_csv, index=False)
    visual_model.to_parquet(visual_model_pq, index=False)
    silver.to_csv(silver_csv, index=False)
    silver.to_parquet(silver_pq, index=False)
    gold.to_csv(gold_csv, index=False)
    gold.to_parquet(gold_pq, index=False)

    summary = {
        "graph_root": str(args.graph_root),
        "image_files_total": int(len(files)),
        "image_extensions": sorted(exts),
        "visual_rows_all": int(len(visual_all)),
        "visual_rows_model_vars": int(len(visual_model)),
        "numeric_rows_all": int(len(numeric_all)),
        "numeric_rows_model_vars": int(len(numeric_model)),
        "silver_rows_all": int(len(silver)),
        "gold_rows_model_vars": int(len(gold)),
        "extract_success_count": int(report_df["extract_ok"].sum()) if not report_df.empty else 0,
        "extract_fail_count": int((~report_df["extract_ok"]).sum()) if not report_df.empty else 0,
        "outputs": {
            "visual_process_report_csv": str(report_csv),
            "visual_measurements_all_csv": str(visual_all_csv),
            "visual_model_observations_csv": str(visual_model_csv),
            "silver_observations_csv": str(silver_csv),
            "silver_observations_parquet": str(silver_pq),
            "gold_observations_csv": str(gold_csv),
            "gold_observations_parquet": str(gold_pq),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
