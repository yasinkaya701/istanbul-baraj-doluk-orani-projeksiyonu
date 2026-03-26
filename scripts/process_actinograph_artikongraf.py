#!/usr/bin/env python3
"""
Process actinograph daily charts using ArtikongrafConverter (radiation model).
Outputs a daily radiation report compatible with build_complete_solar_dataset.py
and (optionally) per-file minute series CSVs.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import cv2
from PIL import Image

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[1]
ARTI_ROOT = ROOT / "external" / "ArtikongrafConverter" / "AktinografConverter"
ARTI_PY = ARTI_ROOT / "python"

if str(ARTI_PY) not in sys.path:
    sys.path.insert(0, str(ARTI_PY))

import extract_daily_series as eds  # noqa: E402
from chart_models import get_models  # noqa: E402

# ---- Date inference ----
MONTH_MAP = {
    "ocak": 1, "jan": 1,
    "subat": 2, "feb": 2,
    "mart": 3, "mar": 3,
    "nisan": 4, "apr": 4,
    "mayis": 5, "may": 5,
    "haziran": 6, "jun": 6,
    "temmuz": 7, "jul": 7,
    "agustos": 8, "aug": 8,
    "eylul": 9, "sep": 9,
    "ekim": 10, "oct": 10,
    "kasim": 11, "nov": 11,
    "aralik": 12, "dec": 12,
}

TR_CHARS = str.maketrans({
    "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
    "Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U",
})


def norm_text(x: str) -> str:
    s = str(x).strip().lower().translate(TR_CHARS)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def infer_date(path: Path) -> datetime:
    t = norm_text(str(path))
    m_year = re.search(r"(19|20)\d{2}", str(path))
    year = int(m_year.group(0)) if m_year else 2000

    month = 1
    for k, mm in MONTH_MAP.items():
        if re.search(rf"\b{re.escape(k)}\b", t):
            month = mm
            break

    name = path.stem
    m_day = re.search(r"[-_](\d{1,2})$", name)
    day = 1
    if m_day:
        d_val = int(m_day.group(1))
        if 1 <= d_val <= 31:
            day = d_val

    try:
        return datetime(year, month, day, 12, 0, 0)
    except Exception:
        return datetime(year, month, 1, 12, 0, 0)


# ---- Worker globals ----
_MODEL = None
_TEMPLATE = None
_SPIKE = 0
_SERIES_DIR = None
_RATE_UNIT = "cal_cm2_min"
_MIN_COVERAGE = 0.0
_MIN_STRENGTH = 0.0
_LOWQ_DEBUG_DIR = None


def _init_worker(
    spike_margin_int: int,
    series_dir: str | None,
    rate_unit: str,
    min_coverage: float,
    min_strength: float,
    lowq_debug_dir: str | None,
) -> None:
    global _MODEL, _TEMPLATE, _SPIKE, _SERIES_DIR, _RATE_UNIT, _MIN_COVERAGE, _MIN_STRENGTH, _LOWQ_DEBUG_DIR
    models = get_models(ARTI_ROOT)
    _MODEL = models["radiation"]
    _TEMPLATE = cv2.imread(str(_MODEL.template_path), cv2.IMREAD_COLOR)
    if _TEMPLATE is None:
        raise RuntimeError(f"Template not found: {_MODEL.template_path}")
    _SPIKE = int(spike_margin_int)
    _SERIES_DIR = Path(series_dir) if series_dir else None
    _RATE_UNIT = rate_unit
    _MIN_COVERAGE = float(min_coverage)
    _MIN_STRENGTH = float(min_strength)
    _LOWQ_DEBUG_DIR = Path(lowq_debug_dir) if lowq_debug_dir else None


def _load_image_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _points_to_rows(points, source_file: str):
    rows = []
    for x, y, _hcont, _hlbl, hprec, val in points:
        rows.append({
            "source_file": source_file,
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "hour_precise": str(hprec),
            "radiation": round(float(val), 4),
        })
    return rows


def _write_series_csv(out_path: Path, rows) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def _compute_daily_total(points) -> tuple[float, float]:
    # We integrate the rate curve over time (minute resolution, 1440 points).
    # Default assumes values are in cal/cm2/min.
    vals = np.array([p[5] for p in points], dtype=float)
    if _RATE_UNIT == "cal_cm2_hour":
        # Convert per-hour rate to per-minute before summing.
        total_cal_cm2 = float(np.nansum(vals) / 60.0)
    else:
        total_cal_cm2 = float(np.nansum(vals))
    total_mj_m2 = total_cal_cm2 * 0.04184
    return total_cal_cm2, total_mj_m2


def _process_one(path_str: str) -> dict:
    path = Path(path_str)
    try:
        img_bgr = _load_image_bgr(path)
        aligned, cc = eds.load_and_align_to_template(img_bgr, _TEMPLATE)
        # ECC can collapse to a white frame on low-contrast sheets.
        # Fallback to direct resize when alignment looks invalid.
        if aligned is None or aligned.mean() > 254.5 or aligned.std() < 1.0 or cc < 0.5:
            aligned = cv2.resize(
                img_bgr,
                (_TEMPLATE.shape[1], _TEMPLATE.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            cc = 0.0
        points, coverage, strength, score_img = eds.extract_points_for_model(aligned, _MODEL, _SPIKE)
        if not points:
            return {
                "source_file": path.name,
                "status": "skipped",
                "error": "no_points",
            }

        quality_pass = (coverage >= _MIN_COVERAGE) and (strength >= _MIN_STRENGTH)
        if not quality_pass:
            if _LOWQ_DEBUG_DIR is not None:
                eds.save_debug_images(_LOWQ_DEBUG_DIR, path, aligned, _MODEL, score_img, points)
            return {
                "source_file": path.name,
                "year": infer_date(path).year,
                "month": infer_date(path).month,
                "alignment_cc": round(float(cc), 6),
                "valid_coverage": round(float(coverage), 6),
                "trace_strength": round(float(strength), 6),
                "rate_unit": _RATE_UNIT,
                "status": "low_quality",
                "error": "quality_gate_failed",
            }

        if _SERIES_DIR is not None:
            rows = _points_to_rows(points, path.name)
            out_csv = _SERIES_DIR / f"{path.stem}_series.csv"
            _write_series_csv(out_csv, rows)

        dt = infer_date(path)
        total_cal_cm2, total_mj_m2 = _compute_daily_total(points)
        return {
            "date": dt.strftime("%Y-%m-%d"),
            "daily_total_cal_cm2": round(total_cal_cm2, 4),
            "daily_total_mj_m2": round(total_mj_m2, 4),
            "source_file": path.name,
            "year": dt.year,
            "month": dt.month,
            "data_source": "real_extracted",
            "alignment_cc": round(float(cc), 6),
            "valid_coverage": round(float(coverage), 6),
            "trace_strength": round(float(strength), 6),
            "rate_unit": _RATE_UNIT,
            "status": "ok",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "source_file": path.name,
            "status": "error",
            "error": str(e),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Process actinograph daily charts using ArtikongrafConverter")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "new data" / "Aktinograph-GÜNLÜK-2",
        help="Root folder containing actinograph images",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=ROOT / "output" / "universal_datasets" / "daily_solar_radiation_report.csv",
        help="Output daily radiation report CSV",
    )
    parser.add_argument(
        "--series-dir",
        type=Path,
        default=None,
        help="Optional folder to write per-file minute series CSVs",
    )
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers")
    parser.add_argument("--spike-margin", type=int, default=0, help="Radiation spike margin int")
    parser.add_argument("--min-coverage", type=float, default=0.60, help="Quality gate: minimum coverage")
    parser.add_argument("--min-strength", type=float, default=0.15, help="Quality gate: minimum trace strength")
    parser.add_argument(
        "--lowq-debug-dir",
        type=Path,
        default=None,
        help="Optional folder to save debug overlays for low-quality sheets",
    )
    parser.add_argument(
        "--rate-unit",
        choices=["cal_cm2_min", "cal_cm2_hour"],
        default="cal_cm2_min",
        help="Unit of y-axis rate on chart.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of files to process")
    args = parser.parse_args()

    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    all_files = [p for p in args.input_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    all_files = sorted(all_files)
    if args.limit and args.limit > 0:
        all_files = all_files[: args.limit]

    if not all_files:
        raise SystemExit(f"No image files found under {args.input_root}")

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    if args.series_dir is not None:
        args.series_dir.mkdir(parents=True, exist_ok=True)

    results = []
    ok = 0
    skip = 0
    err = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(
            args.spike_margin,
            str(args.series_dir) if args.series_dir else None,
            args.rate_unit,
            args.min_coverage,
            args.min_strength,
            str(args.lowq_debug_dir) if args.lowq_debug_dir else None,
        ),
    ) as ex:
        futures = {ex.submit(_process_one, str(p)): p for p in all_files}
        for i, fut in enumerate(as_completed(futures)):
            res = fut.result()
            results.append(res)
            st = res.get("status")
            if st == "ok":
                ok += 1
            elif st == "skipped":
                skip += 1
            else:
                err += 1
            if (i + 1) % 200 == 0:
                print(f"[{i+1}/{len(all_files)}] ok={ok} skip={skip} err={err}")

    # Build daily report
    ok_rows = [r for r in results if r.get("status") == "ok"]
    if ok_rows:
        df = pd.DataFrame(ok_rows)
        df = df.sort_values(["date", "source_file"]).reset_index(drop=True)
        # If multiple files per date, average totals and keep first source_file
        daily = (
            df.groupby("date", as_index=False)
            .agg(
                daily_total_cal_cm2=("daily_total_cal_cm2", "mean"),
                daily_total_mj_m2=("daily_total_mj_m2", "mean"),
                source_file=("source_file", "first"),
                year=("year", "first"),
                month=("month", "first"),
                data_source=("data_source", "first"),
                rate_unit=("rate_unit", "first"),
                alignment_cc=("alignment_cc", "mean"),
                valid_coverage=("valid_coverage", "mean"),
                trace_strength=("trace_strength", "mean"),
            )
        )
        daily.to_csv(args.out_report, index=False)
        print(f"\nSaved daily report: {args.out_report}")
        print(f"Days: {len(daily)} | Date range: {daily['date'].min()} -> {daily['date'].max()}")
    else:
        print("No valid rows produced.")

    # Save error log if needed
    err_rows = [r for r in results if r.get("status") != "ok"]
    if err_rows:
        err_path = args.out_report.with_name(args.out_report.stem + "_errors.csv")
        pd.DataFrame(err_rows).to_csv(err_path, index=False)
        print(f"Errors saved: {err_path}")


if __name__ == "__main__":
    main()
