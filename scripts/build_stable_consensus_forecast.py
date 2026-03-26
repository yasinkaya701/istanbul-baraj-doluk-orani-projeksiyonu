#!/usr/bin/env python3
"""Build stable consensus forecast for temp/humidity from model suite outputs.

This script creates a robust monthly consensus forecast by:
- selecting candidate model forecasts per variable (temp/humidity),
- reading model-quality metrics from index/leaderboard files,
- applying stability penalties (jump + volatility + low coverage),
- blending with weighted median and weighted quantiles,
- applying light smoothing to reduce jagged forecast paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_MODELS = [
    "quant",
    "prophet",
    "strong",
    "analog",
    "prophet_ultra",
    "best_meta",
    "walkforward",
    "literature",
]

TARGET_VARIABLES = ["temp", "humidity"]
DEFAULT_UNITS = {"temp": "C", "humidity": "%"}
RANGES = {"temp": (-60.0, 60.0), "humidity": (0.0, 100.0)}


@dataclass
class Candidate:
    model: str
    variable: str
    forecast_csv: Path
    metric_value: float
    metric_key: str
    selection_reason: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build stable consensus forecast from run directory")
    p.add_argument("--run-dir", type=Path, required=True, help="Model suite output directory")
    p.add_argument("--future-start", type=int, default=2026)
    p.add_argument("--future-end", type=int, default=-1, help="<=0 means infer from run-dir")
    p.add_argument("--output-model-dir", type=str, default="stable_consensus")
    p.add_argument("--min-models", type=int, default=2, help="Minimum candidate count to build consensus")
    return p.parse_args()


def _to_float(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(x):
        return float("nan")
    return x


def _freq_rank(name: str) -> int:
    n = str(name).lower()
    if "_monthly_" in n or "_ms_" in n or n == "ms" or "monthly" in n:
        return 0
    if "_daily_" in n or "_d_" in n or n == "d" or "daily" in n:
        return 1
    if "_hourly_" in n or "_h_" in n or n == "h" or "hourly" in n:
        return 2
    if "_yearly_" in n or "_ys_" in n or n == "ys" or n == "y" or "yearly" in n:
        return 3
    return 9


def _extract_year_hint(name: str) -> int:
    m = re.search(r"_to_(\d{4})(?:\D|$)", name)
    if m:
        return int(m.group(1))
    m2 = re.search(r"_(\d{4})_(\d{4})(?:\D|$)", name)
    if m2:
        return int(m2.group(2))
    return -1


def _extract_year(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    m = re.search(r"(\d{4})", text)
    if m:
        y = int(m.group(1))
        if 1800 <= y <= 2300:
            return y
    return None


def infer_target_year(run_dir: Path) -> int:
    years: list[int] = []
    for p in run_dir.rglob("*.csv"):
        y = _extract_year_hint(p.name)
        if y >= 1900:
            years.append(y)
    if not years:
        return 2035
    return max(years)


def detect_date_col(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except OSError:
        return None
    cols = [str(c).strip() for c in header]
    for c in ("ds", "timestamp", "date", "datetime"):
        if c in cols:
            return c
    return cols[0] if cols else None


def year_bounds(path: Path, date_col: str) -> tuple[int | None, int | None]:
    lo: int | None = None
    hi: int | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                y = _extract_year(row.get(date_col))
                if y is None:
                    continue
                if lo is None or y < lo:
                    lo = y
                if hi is None or y > hi:
                    hi = y
    except OSError:
        return None, None
    return lo, hi


def _metric_from_row(row: dict[str, Any]) -> tuple[float, str] | None:
    # Lower is better.
    priority = [
        "score",
        "cv_rmse",
        "rmse_cv",
        "rmse",
        "best_cv_rmse",
        "holdout_rmse",
        "mae_cv",
        "cv_mae",
        "mae",
        "smape_cv",
        "smape",
        "mape",
    ]
    for k in priority:
        if k not in row:
            continue
        v = _to_float(row.get(k))
        if math.isfinite(v):
            return v, k
    # Rank is weak evidence; keep as last-resort with heavy downweight.
    if "rank" in row:
        r = _to_float(row.get("rank"))
        if math.isfinite(r):
            return 25.0 + r, "rank_penalized"
    return None


def _normalize_path(raw: str, model_dir: Path) -> Path | None:
    txt = str(raw or "").strip()
    if not txt or txt.startswith("generated://"):
        return None
    p = Path(txt)
    if p.exists():
        return p.resolve()
    p2 = (Path.cwd() / txt).resolve()
    if p2.exists():
        return p2
    p3 = (model_dir / txt).resolve()
    if p3.exists():
        return p3
    return None


def _metric_sources(model_dir: Path) -> list[Path]:
    out: list[Path] = []
    for pat in ("*index*.csv", "*metrics*.csv"):
        out.extend(sorted(model_dir.glob(pat)))
    lb = model_dir / "leaderboards"
    if lb.exists():
        out.extend(sorted(lb.glob("*.csv")))
    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq


def collect_metric_map(model_dir: Path, variable: str) -> tuple[dict[Path, tuple[float, str]], tuple[float, str] | None]:
    var = variable.lower()
    direct: dict[Path, tuple[float, str]] = {}
    generic_best: tuple[float, str] | None = None
    for src in _metric_sources(model_dir):
        try:
            with src.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                headers = [h.strip() for h in (reader.fieldnames or []) if h]
                if not headers:
                    continue
                has_var_col = "variable" in headers
                src_match = var in src.name.lower()
                for row in reader:
                    row_var = str(row.get("variable", "")).strip().lower()
                    if has_var_col and row_var and row_var != var:
                        continue
                    m = _metric_from_row(row)
                    if m is None:
                        continue
                    mv, mk = m
                    desc = f"{src.name}:{mk}={mv:.6g}"
                    p = _normalize_path(str(row.get("forecast_csv", "")), model_dir=model_dir)
                    if p is not None and var in p.name.lower():
                        cur = direct.get(p)
                        if cur is None or mv < cur[0]:
                            direct[p] = (mv, desc)
                    if has_var_col or src_match:
                        if generic_best is None or mv < generic_best[0]:
                            generic_best = (mv, desc)
        except OSError:
            continue
    return direct, generic_best


def pick_forecast_for_model(
    run_dir: Path,
    model: str,
    variable: str,
    future_start: int,
) -> Candidate | None:
    model_dir = run_dir / model
    fdir = model_dir / "forecasts"
    if not fdir.exists():
        return None
    files = [p for p in fdir.glob("*.csv") if variable in p.name.lower()]
    if not files:
        return None

    metric_direct, metric_generic = collect_metric_map(model_dir=model_dir, variable=variable)
    ranked: list[tuple[tuple[int, float, int, int, str], Candidate]] = []
    for fp in files:
        dcol = detect_date_col(fp)
        if not dcol:
            continue
        _, y_max = year_bounds(fp, dcol)
        if y_max is not None and y_max < future_start:
            continue
        rp = fp.resolve()
        metric = metric_direct.get(rp, metric_generic)
        if metric is None:
            has_metric = 1
            mv = float("inf")
            mk = "none"
            reason = "fallback:name+frequency"
        else:
            has_metric = 0
            mv = float(metric[0])
            mk = metric[1].split(":")[-1].split("=")[0]
            reason = f"metric:{metric[1]}"
        candidate = Candidate(
            model=model,
            variable=variable,
            forecast_csv=rp,
            metric_value=mv,
            metric_key=mk,
            selection_reason=reason,
        )
        score = (has_metric, mv, _freq_rank(fp.name), -_extract_year_hint(fp.name), fp.name)
        ranked.append((score, candidate))

    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1]


def parse_is_forecast(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=bool)
    txt = s.astype(str).str.lower().str.strip()
    out = txt.map({"true": True, "false": False, "1": True, "0": False})
    return out.fillna(False).astype(bool)


def load_monthly_forecast(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    dcol = None
    for c in ("ds", "timestamp", "date", "datetime"):
        if c in d.columns:
            dcol = c
            break
    if dcol is None:
        dcol = d.columns[0]
    d["ds"] = pd.to_datetime(d[dcol], errors="coerce")
    d = d.dropna(subset=["ds"]).copy()
    d["yhat"] = pd.to_numeric(d.get("yhat"), errors="coerce")
    d["actual"] = pd.to_numeric(d.get("actual"), errors="coerce")
    d["yhat_lower"] = pd.to_numeric(d.get("yhat_lower"), errors="coerce")
    d["yhat_upper"] = pd.to_numeric(d.get("yhat_upper"), errors="coerce")
    if "is_forecast" in d.columns:
        d["is_forecast"] = parse_is_forecast(d["is_forecast"])
    else:
        d["is_forecast"] = False
    d["month_ds"] = d["ds"].dt.to_period("M").dt.to_timestamp()
    # If a month has both historical and forecast records (edge cases), forecast wins.
    agg = (
        d.groupby("month_ds", as_index=False)
        .agg(
            yhat=("yhat", "mean"),
            actual=("actual", "mean"),
            yhat_lower=("yhat_lower", "mean"),
            yhat_upper=("yhat_upper", "mean"),
            is_forecast=("is_forecast", "max"),
        )
        .rename(columns={"month_ds": "ds"})
        .sort_values("ds")
    )
    return agg


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    wsum = float(np.sum(w))
    if wsum <= 0:
        return float(np.median(v))
    cdf = np.cumsum(w) / wsum
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = max(0, min(idx, len(v) - 1))
    return float(v[idx])


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    return weighted_quantile(values, weights, 0.5)


def median3_smooth(arr: np.ndarray, alpha: float = 0.32, passes: int = 2) -> np.ndarray:
    y = arr.astype(float).copy()
    n = len(y)
    if n < 3:
        return y
    for _ in range(max(1, int(passes))):
        m = y.copy()
        for i in range(1, n - 1):
            med = float(np.median([y[i - 1], y[i], y[i + 1]]))
            m[i] = (1.0 - alpha) * y[i] + alpha * med
        y = m
    return y


def ewma_smooth(arr: np.ndarray, alpha: float = 0.22) -> np.ndarray:
    y = arr.astype(float).copy()
    if len(y) < 2:
        return y
    out = y.copy()
    for i in range(1, len(y)):
        out[i] = alpha * out[i] + (1.0 - alpha) * out[i - 1]
    return out


def stability_stats(hist: np.ndarray, fut: np.ndarray) -> tuple[float, float, float]:
    if len(hist) < 6 or len(fut) < 3:
        return 1.0, 1.0, 0.0
    hdiff = np.diff(hist[-60:]) if len(hist) >= 60 else np.diff(hist)
    fdiff = np.diff(fut)
    hstd = float(np.nanstd(hdiff)) if len(hdiff) else 1e-6
    fstd = float(np.nanstd(fdiff)) if len(fdiff) else 0.0
    vol_ratio = fstd / max(hstd, 1e-6)
    hq = float(np.nanpercentile(hist, 90) - np.nanpercentile(hist, 10))
    jump = abs(float(fut[0]) - float(hist[-1])) / max(hq, 1e-6)
    penalty = 1.0 / (1.0 + 0.45 * max(0.0, vol_ratio - 2.0) + 0.30 * max(0.0, jump - 2.0))
    penalty = float(np.clip(penalty, 0.10, 1.0))
    return penalty, vol_ratio, jump


def build_reference_actual_map(loaded: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    # Preferred reference: explicit historical actual values.
    actual_parts: list[pd.DataFrame] = []
    for _, f in loaded.items():
        h = f[f["is_forecast"] == False][["ds", "actual"]].copy()
        h["actual"] = pd.to_numeric(h["actual"], errors="coerce")
        h = h.dropna(subset=["ds", "actual"])
        if not h.empty:
            actual_parts.append(h)

    if actual_parts:
        a = pd.concat(actual_parts, ignore_index=True)
        ref = a.groupby("ds", as_index=False)["actual"].median().rename(columns={"actual": "ref_actual"})
        return ref.sort_values("ds"), "explicit_actual_median"

    # Fallback reference: historical yhat median across models.
    y_parts: list[pd.DataFrame] = []
    for _, f in loaded.items():
        h = f[f["is_forecast"] == False][["ds", "yhat"]].copy()
        h["yhat"] = pd.to_numeric(h["yhat"], errors="coerce")
        h = h.dropna(subset=["ds", "yhat"]).rename(columns={"yhat": "ref_actual"})
        if not h.empty:
            y_parts.append(h)

    if y_parts:
        y = pd.concat(y_parts, ignore_index=True)
        ref = y.groupby("ds", as_index=False)["ref_actual"].median()
        return ref.sort_values("ds"), "historical_yhat_median"

    return pd.DataFrame(columns=["ds", "ref_actual"]), "none"


def build_variable_consensus(
    run_dir: Path,
    variable: str,
    future_start: int,
    future_end: int,
    min_models: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected: list[Candidate] = []
    loaded: dict[str, pd.DataFrame] = {}
    for model in BASE_MODELS:
        c = pick_forecast_for_model(run_dir=run_dir, model=model, variable=variable, future_start=future_start)
        if c is None:
            continue
        try:
            f = load_monthly_forecast(c.forecast_csv)
        except Exception:
            continue
        fut_eval = f[(f["is_forecast"]) & (f["ds"].dt.year >= future_start) & (f["ds"].dt.year <= future_end)].copy()
        fut_all = f[(f["is_forecast"]) & (f["ds"].dt.year <= future_end)].copy()
        if fut_eval.empty or fut_all.empty:
            continue
        selected.append(c)
        loaded[model] = f

    if len(selected) == 0:
        raise RuntimeError(f"{variable}: no candidate forecasts found")
    if len(selected) < int(min_models):
        raise RuntimeError(f"{variable}: insufficient candidates ({len(selected)} < {int(min_models)})")

    ref_actual_df, ref_source = build_reference_actual_map(loaded)
    ref_scale = np.nan
    if not ref_actual_df.empty:
        rv = ref_actual_df["ref_actual"].to_numpy(dtype=float)
        if len(rv) >= 6:
            ref_scale = float(np.nanpercentile(rv, 90) - np.nanpercentile(rv, 10))
            if not math.isfinite(ref_scale) or ref_scale <= 1e-9:
                ref_scale = float(np.nanstd(rv))
        elif len(rv) >= 2:
            ref_scale = float(np.nanstd(rv))
        if not math.isfinite(ref_scale) or ref_scale <= 1e-9:
            ref_scale = 1.0

    expected_months = max(1, (future_end - future_start + 1) * 12)

    rows: list[dict[str, Any]] = []
    for c in selected:
        f = loaded[c.model]
        hist = f[f["is_forecast"] == False]["yhat"].dropna().to_numpy(dtype=float)
        fut = f[(f["is_forecast"]) & (f["ds"].dt.year >= future_start) & (f["ds"].dt.year <= future_end)]["yhat"].dropna().to_numpy(dtype=float)
        if not len(fut):
            continue
        coverage = float(len(fut) / expected_months)
        coverage = float(np.clip(coverage, 0.0, 1.0))
        metric = c.metric_value if math.isfinite(c.metric_value) and c.metric_value > 0 else float("nan")
        if math.isfinite(metric):
            base_w = 1.0 / max(metric, 1e-6)
        else:
            base_w = 0.15
        stab_w, vol_ratio, jump = stability_stats(hist=hist, fut=fut)
        cov_w = 0.50 + 0.50 * coverage
        # Historical-fit factor against reference actual map.
        hdf = f[f["is_forecast"] == False][["ds", "yhat"]].copy()
        hdf["yhat"] = pd.to_numeric(hdf["yhat"], errors="coerce")
        if ref_actual_df.empty:
            jh = pd.DataFrame(columns=["ds", "yhat", "ref_actual"])
        else:
            jh = hdf.dropna(subset=["ds", "yhat"]).merge(ref_actual_df, on="ds", how="inner")
        hist_overlap_n = int(len(jh))
        surrogate_rmse = float("nan")
        surrogate_mae = float("nan")
        identity_like = False
        if hist_overlap_n > 0:
            err = jh["yhat"].to_numpy(dtype=float) - jh["ref_actual"].to_numpy(dtype=float)
            surrogate_rmse = float(np.sqrt(np.mean(err * err)))
            surrogate_mae = float(np.mean(np.abs(err)))
            if hist_overlap_n >= 24 and surrogate_rmse <= 1e-8:
                identity_like = True

        if math.isfinite(surrogate_rmse):
            hist_fit_w = 1.0 / (1.0 + surrogate_rmse / max(ref_scale, 1e-6))
            hist_fit_w = float(np.clip(hist_fit_w, 0.30, 1.0))
        else:
            hist_fit_w = 0.85
        if hist_overlap_n < 12:
            hist_fit_w *= 0.92
        if identity_like:
            # Backfill-like perfect historical fit should not dominate weights.
            hist_fit_w *= 0.78

        raw_w = base_w * stab_w * cov_w * hist_fit_w
        rows.append(
            {
                "model": c.model,
                "variable": variable,
                "forecast_csv": str(c.forecast_csv),
                "metric_value": metric,
                "metric_key": c.metric_key,
                "selection_reason": c.selection_reason,
                "coverage_ratio": coverage,
                "vol_ratio": vol_ratio,
                "jump_score": jump,
                "stability_weight": stab_w,
                "coverage_weight": cov_w,
                "hist_overlap_n": hist_overlap_n,
                "surrogate_rmse": surrogate_rmse,
                "surrogate_mae": surrogate_mae,
                "hist_fit_weight": hist_fit_w,
                "reference_source": ref_source,
                "identity_like": identity_like,
                "weight_raw": raw_w,
            }
        )

    wd = pd.DataFrame(rows).sort_values("weight_raw", ascending=False)
    if wd.empty:
        raise RuntimeError(f"{variable}: no usable model weights")

    # Second-pass robustness: reduce weights for candidates that deviate from
    # preliminary consensus over target future window.
    prelim_sum = float(wd["weight_raw"].sum())
    if math.isfinite(prelim_sum) and prelim_sum > 0:
        prelim_w = {str(r["model"]): float(r["weight_raw"]) / prelim_sum for _, r in wd.iterrows()}
    else:
        prelim_w = {str(r["model"]): 1.0 / len(wd) for _, r in wd.iterrows()}

    eval_frames: dict[str, pd.DataFrame] = {}
    for model in wd["model"].astype(str).tolist():
        f = loaded[model]
        ff = f[(f["is_forecast"]) & (f["ds"].dt.year >= future_start) & (f["ds"].dt.year <= future_end)][["ds", "yhat"]].copy()
        ff = ff.dropna(subset=["ds", "yhat"]).sort_values("ds")
        eval_frames[model] = ff

    all_eval_ds = sorted(set(pd.concat([ef["ds"] for ef in eval_frames.values()]).astype("datetime64[ns]").tolist()))
    center_rows: list[tuple[pd.Timestamp, float]] = []
    for ds in all_eval_ds:
        vals: list[float] = []
        wts: list[float] = []
        for model, ef in eval_frames.items():
            rr = ef[ef["ds"] == ds]
            if rr.empty:
                continue
            v = _to_float(rr["yhat"].iloc[0])
            if not math.isfinite(v):
                continue
            vals.append(v)
            wts.append(float(prelim_w.get(model, 0.0)))
        if not vals:
            continue
        center_rows.append((pd.Timestamp(ds), weighted_median(np.asarray(vals, dtype=float), np.asarray(wts, dtype=float))))

    center_df = pd.DataFrame(center_rows, columns=["ds", "center_yhat"])
    if center_df.empty:
        wd["disagreement_dev_ratio"] = np.nan
        wd["disagreement_weight"] = 1.0
    else:
        cvals = center_df["center_yhat"].to_numpy(dtype=float)
        cspan = float(np.nanpercentile(cvals, 90) - np.nanpercentile(cvals, 10))
        cspan = max(cspan, 1e-6)
        dev_ratio_map: dict[str, float] = {}
        dw_map: dict[str, float] = {}
        for model, ef in eval_frames.items():
            j = ef.merge(center_df, on="ds", how="inner")
            if j.empty:
                dev_ratio = float("nan")
                d_w = 0.80
            else:
                dev = np.abs(j["yhat"].to_numpy(dtype=float) - j["center_yhat"].to_numpy(dtype=float))
                dev_ratio = float(np.nanmedian(dev) / cspan) if len(dev) else float("nan")
                if not math.isfinite(dev_ratio):
                    d_w = 0.80
                else:
                    d_w = 1.0 / (1.0 + 1.8 * max(0.0, dev_ratio - 0.15))
                    d_w = float(np.clip(d_w, 0.25, 1.0))
            dev_ratio_map[model] = dev_ratio
            dw_map[model] = d_w

        wd["disagreement_dev_ratio"] = wd["model"].astype(str).map(dev_ratio_map)
        wd["disagreement_weight"] = wd["model"].astype(str).map(dw_map).fillna(0.80)
        wd["weight_raw"] = wd["weight_raw"] * wd["disagreement_weight"]
        wd = wd.sort_values("weight_raw", ascending=False)

    wsum = float(wd["weight_raw"].sum())
    if not math.isfinite(wsum) or wsum <= 0:
        wd["weight"] = 1.0 / len(wd)
    else:
        wd["weight"] = wd["weight_raw"] / wsum
    # Avoid single-model domination for stability.
    wd["weight"] = wd["weight"].clip(upper=0.60)
    wd["weight"] = wd["weight"] / float(wd["weight"].sum())

    model_weights = {str(r["model"]): float(r["weight"]) for _, r in wd.iterrows()}
    model_frames: dict[str, pd.DataFrame] = {}
    for model in model_weights:
        f = loaded[model]
        ff = f[(f["is_forecast"]) & (f["ds"].dt.year <= future_end)][["ds", "yhat"]].copy()
        ff = ff.dropna(subset=["ds", "yhat"]).sort_values("ds")
        model_frames[model] = ff

    all_ds = sorted(set(pd.concat([mf["ds"] for mf in model_frames.values()]).astype("datetime64[ns]").tolist()))
    fut_rows: list[dict[str, Any]] = []
    for ds in all_ds:
        vals: list[float] = []
        wts: list[float] = []
        for model, mf in model_frames.items():
            rr = mf[mf["ds"] == ds]
            if rr.empty:
                continue
            v = _to_float(rr["yhat"].iloc[0])
            if not math.isfinite(v):
                continue
            vals.append(v)
            wts.append(model_weights[model])
        if not vals:
            continue
        vv = np.asarray(vals, dtype=float)
        ww = np.asarray(wts, dtype=float)
        y = weighted_median(vv, ww)
        lo = weighted_quantile(vv, ww, 0.20)
        hi = weighted_quantile(vv, ww, 0.80)
        fut_rows.append({"ds": pd.Timestamp(ds), "yhat": y, "yhat_lower": lo, "yhat_upper": hi, "is_forecast": True})

    fut_df = pd.DataFrame(fut_rows).sort_values("ds")
    if fut_df.empty:
        raise RuntimeError(f"{variable}: empty consensus forecast")

    ys = fut_df["yhat"].to_numpy(dtype=float)
    ys2 = median3_smooth(ys, alpha=0.34, passes=2)
    ys3 = ewma_smooth(ys2, alpha=0.22)
    # Keep long-run level stable after smoothing.
    shift = float(np.nanmean(ys) - np.nanmean(ys3))
    ys3 = ys3 + shift
    lo_b, hi_b = RANGES[variable]
    ys3 = np.clip(ys3, lo_b, hi_b)
    fut_df["yhat"] = ys3
    fut_df["yhat_lower"] = np.minimum(fut_df["yhat_lower"].to_numpy(dtype=float), ys3)
    fut_df["yhat_upper"] = np.maximum(fut_df["yhat_upper"].to_numpy(dtype=float), ys3)

    # Historical monthly actual anchor: prefer file with most non-null actuals.
    hist_pick_model = None
    hist_pick_count = -1
    for model, f in loaded.items():
        cnt = int(f[(f["is_forecast"] == False) & (f["actual"].notna())].shape[0])
        if cnt > hist_pick_count:
            hist_pick_count = cnt
            hist_pick_model = model
    if hist_pick_model is None:
        hist_pick_model = list(loaded.keys())[0]
    hist_src = loaded[hist_pick_model]
    hist_df = hist_src[hist_src["is_forecast"] == False][["ds", "actual", "yhat"]].copy()
    hist_df["actual"] = hist_df["actual"].where(hist_df["actual"].notna(), hist_df["yhat"])
    hist_df["yhat"] = hist_df["actual"]
    hist_df["yhat_lower"] = np.nan
    hist_df["yhat_upper"] = np.nan
    hist_df["is_forecast"] = False
    hist_df = hist_df.dropna(subset=["ds", "yhat"]).sort_values("ds")

    out = pd.concat([hist_df, fut_df], ignore_index=True, sort=False)
    out = out.drop_duplicates(subset=["ds"], keep="last").sort_values("ds")
    out["variable"] = variable
    out["unit"] = DEFAULT_UNITS[variable]
    out["frequency"] = "MS"
    out["model_strategy"] = "stable_consensus_weighted_median_v1"
    out["actual"] = out["actual"].where(~out["is_forecast"], np.nan)
    out["ds"] = pd.to_datetime(out["ds"]).dt.strftime("%Y-%m-%d")

    return out[
        [
            "ds",
            "actual",
            "yhat",
            "yhat_lower",
            "yhat_upper",
            "is_forecast",
            "variable",
            "unit",
            "frequency",
            "model_strategy",
        ]
    ].copy(), wd


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"run-dir missing: {run_dir}")

    future_end = int(args.future_end)
    if future_end <= 0:
        future_end = infer_target_year(run_dir)
    future_start = int(args.future_start)
    if future_start > future_end:
        future_start = max(1900, future_end - 1)

    model_dir = run_dir / str(args.output_model_dir)
    fc_dir = model_dir / "forecasts"
    lb_dir = model_dir / "leaderboards"
    ch_dir = model_dir / "charts"
    for d in (fc_dir, lb_dir, ch_dir):
        d.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    ok_count = 0
    for variable in TARGET_VARIABLES:
        try:
            out_df, weights_df = build_variable_consensus(
                run_dir=run_dir,
                variable=variable,
                future_start=future_start,
                future_end=future_end,
                min_models=int(args.min_models),
            )
            fc_csv = fc_dir / f"{variable}_monthly_stable_consensus_to_{future_end}.csv"
            out_df.to_csv(fc_csv, index=False)

            w_csv = lb_dir / f"{variable}_stable_consensus_weights_to_{future_end}.csv"
            weights_df.to_csv(w_csv, index=False)

            # Quick chart
            d = out_df.copy()
            d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
            fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
            hist = d[d["is_forecast"] == False]
            fut = d[d["is_forecast"] == True]
            if not hist.empty:
                ax.plot(hist["ds"], hist["yhat"], color="#4b5563", linewidth=1.4, label="historical")
            if not fut.empty:
                ax.plot(fut["ds"], fut["yhat"], color="#2563eb", linewidth=2.0, label="stable_consensus")
                if fut["yhat_lower"].notna().any() and fut["yhat_upper"].notna().any():
                    ax.fill_between(
                        fut["ds"],
                        fut["yhat_lower"].astype(float),
                        fut["yhat_upper"].astype(float),
                        color="#93c5fd",
                        alpha=0.25,
                        label="p20-p80 band",
                    )
            ax.set_title(f"{variable} stable consensus to {future_end}")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            png = ch_dir / f"{variable}_monthly_stable_consensus_to_{future_end}.png"
            fig.savefig(png, dpi=170)
            plt.close(fig)

            index_rows.append(
                {
                    "variable": variable,
                    "frequency": "MS",
                    "target_year": future_end,
                    "status": "ok",
                    "message": "",
                    "model_strategy": "stable_consensus_weighted_median_v1",
                    "forecast_csv": str(fc_csv),
                    "weights_csv": str(w_csv),
                    "chart_png": str(png),
                    "future_start": future_start,
                    "future_end": future_end,
                    "n_models": int(len(weights_df)),
                    "weights_json": json.dumps(
                        {str(r["model"]): float(r["weight"]) for _, r in weights_df.iterrows()},
                        ensure_ascii=False,
                    ),
                    "weighted_metric_proxy": float(
                        np.nansum(weights_df["weight"].to_numpy(dtype=float) * weights_df["metric_value"].to_numpy(dtype=float))
                    ),
                    "score": float(
                        np.nansum(weights_df["weight"].to_numpy(dtype=float) * weights_df["metric_value"].to_numpy(dtype=float))
                    ),
                }
            )
            ok_count += 1
        except Exception as exc:  # noqa: BLE001
            index_rows.append(
                {
                    "variable": variable,
                    "frequency": "MS",
                    "target_year": future_end,
                    "status": "skipped",
                    "message": str(exc),
                    "model_strategy": "stable_consensus_weighted_median_v1",
                    "forecast_csv": "",
                    "weights_csv": "",
                    "chart_png": "",
                    "future_start": future_start,
                    "future_end": future_end,
                    "n_models": 0,
                    "weights_json": "{}",
                    "weighted_metric_proxy": float("nan"),
                    "score": float("nan"),
                }
            )

    idx = pd.DataFrame(index_rows)
    idx_csv = model_dir / f"stable_consensus_index_to_{future_end}.csv"
    idx.to_csv(idx_csv, index=False)
    print(f"Wrote: {idx_csv}")
    for r in index_rows:
        if r.get("status") == "ok":
            print(f"Wrote: {r['forecast_csv']}")
            print(f"Wrote: {r['weights_csv']}")
            print(f"Wrote: {r['chart_png']}")
        else:
            print(f"Skipped {r.get('variable')}: {r.get('message')}")
    if ok_count == 0:
        print("No variable could be produced for stable consensus; index created with skipped rows.")


if __name__ == "__main__":
    main()
