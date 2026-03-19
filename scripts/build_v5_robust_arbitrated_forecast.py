#!/usr/bin/env python3
"""Build v5 robust arbitrated forecast.

Improvements over v4:
- Wider candidate pool (quant/strong/prophet + best_meta_v41).
- Observation-overlap error metrics for every candidate.
- Coverage-aware penalties and interval-quality checks.
- Optional top-2 weighted blend candidate per variable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class Candidate:
    variable: str
    model_name: str
    source: str
    metric_name: str
    metric_value: float
    forecast_csv: Path


RANGES = {
    "temp": (-60.0, 60.0),
    "humidity": (0.0, 100.0),
    "pressure": (850.0, 1100.0),
    "precip": (0.0, 500.0),
}

END_BOUNDS = {
    "temp": (-20.0, 22.0),
    "humidity": (60.0, 96.0),
    "pressure": (930.0, 1035.0),
    "precip": (1.0, 220.0),
}

DELTA_ABS_MAX = {
    "temp": 11.0,
    "humidity": 10.0,
    "pressure": 22.0,
    "precip": 130.0,
}

MIN_OVERLAP = {
    "MS": 12,
    "YS": 6,
    "D": 30,
}

OBS_AGG = {
    "temp": "mean",
    "humidity": "mean",
    "pressure": "mean",
    "precip": "sum",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v5 robust arbitrated forecast")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/data_factory/run_20260306_000419"),
    )
    p.add_argument(
        "--obs-parquet",
        type=Path,
        default=None,
        help="Long-format observations parquet; default uses calibrated recent regime.",
    )
    p.add_argument(
        "--blend-threshold",
        type=float,
        default=0.22,
        help="If top-2 candidate score gap ratio <= threshold, create weighted blend candidate.",
    )
    return p.parse_args()


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def confidence_from_score(
    score_total: float,
    penalty: float,
    overlap_coverage: float,
    overlap_points: float,
    note: str,
) -> tuple[float, str]:
    s = safe_float(score_total, default=np.inf)
    p = safe_float(penalty, default=0.0)
    c = safe_float(overlap_coverage, default=0.0)
    n = safe_float(overlap_points, default=0.0)
    note_s = str(note or "")
    if not np.isfinite(s):
        return 0.0, "D"
    conf = 92.0 - (20.0 * s) - (2.0 * p)
    conf += 10.0 * max(0.0, min(c, 1.0) - 0.5)
    if n < 24:
        conf -= 12.0
    if "overlap_identity" in note_s:
        conf -= 16.0
    if "no_interval" in note_s:
        conf -= 6.0
    conf = float(np.clip(conf, 0.0, 95.0))
    if conf >= 85.0:
        grade = "A"
    elif conf >= 70.0:
        grade = "B"
    elif conf >= 55.0:
        grade = "C"
    else:
        grade = "D"
    return conf, grade


def load_obs(obs_path: Path) -> pd.DataFrame:
    d = pd.read_parquet(obs_path)
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["timestamp", "variable", "value"])
    return d[["timestamp", "variable", "value"]].copy()


def infer_freq(ds: pd.Series) -> str:
    s = pd.to_datetime(ds, errors="coerce").dropna().sort_values()
    if len(s) < 3:
        return "MS"
    diff_days = s.diff().dt.total_seconds().dropna() / (24 * 3600)
    if diff_days.empty:
        return "MS"
    med = float(np.nanmedian(diff_days))
    if med <= 2.0:
        return "D"
    if med <= 45.0:
        return "MS"
    return "YS"


def agg_observations(obs: pd.DataFrame, variable: str, freq: str) -> pd.DataFrame:
    d = obs[obs["variable"] == variable].copy()
    if d.empty:
        return pd.DataFrame(columns=["ds", "actual"])

    agg = OBS_AGG.get(variable, "mean")
    if freq == "D":
        d["ds"] = d["timestamp"].dt.floor("D")
    elif freq == "YS":
        d["ds"] = d["timestamp"].dt.to_period("Y").dt.to_timestamp()
    else:
        d["ds"] = d["timestamp"].dt.to_period("M").dt.to_timestamp()

    g = d.groupby("ds", as_index=False)["value"]
    if agg == "sum":
        out = g.sum().rename(columns={"value": "actual"})
    else:
        out = g.mean().rename(columns={"value": "actual"})
    out["actual"] = pd.to_numeric(out["actual"], errors="coerce")
    return out.dropna(subset=["ds", "actual"]).sort_values("ds")


def load_fc(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["ds"] = pd.to_datetime(d.get("ds"), errors="coerce")
    d = d.dropna(subset=["ds"]).sort_values("ds")
    for c in ["yhat", "yhat_lower", "yhat_upper", "actual"]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "is_forecast" not in d.columns:
        d["is_forecast"] = False
    d["is_forecast"] = d["is_forecast"].astype(bool)
    return d


def overlap_metrics(fc: pd.DataFrame, obs: pd.DataFrame, variable: str) -> dict[str, Any]:
    hist = fc[fc["is_forecast"] == False].copy()
    actual_from_file = bool(hist["actual"].notna().any()) if "actual" in hist.columns else False
    if hist.empty:
        return {
            "overlap_points": 0,
            "overlap_coverage": 0.0,
            "overlap_rmse": np.nan,
            "overlap_mae": np.nan,
            "rolling_rmse_med": np.nan,
            "rolling_rmse_q90": np.nan,
            "naive_skill": np.nan,
            "identity_fit_ratio": np.nan,
            "actual_from_file": actual_from_file,
            "freq": infer_freq(fc["ds"]),
        }

    freq = str(hist["frequency"].dropna().iloc[0]) if "frequency" in hist.columns and hist["frequency"].notna().any() else infer_freq(hist["ds"])
    freq = "YS" if freq.startswith("Y") else ("D" if freq.startswith("D") else "MS")

    obs_agg = agg_observations(obs, variable=variable, freq=freq)
    h = hist[["ds", "yhat", "actual"]].copy()

    if h["actual"].notna().any():
        h_obs = h.copy().rename(columns={"actual": "actual_obs"})
    else:
        h_obs = h.merge(obs_agg.rename(columns={"actual": "actual_obs"}), on="ds", how="left")

    eval_df = h_obs.dropna(subset=["yhat", "actual_obs"]).sort_values("ds")
    n_hist = int(len(h))
    n_eval = int(len(eval_df))
    coverage = float(n_eval / n_hist) if n_hist else 0.0

    if n_eval == 0:
        return {
            "overlap_points": 0,
            "overlap_coverage": coverage,
            "overlap_rmse": np.nan,
            "overlap_mae": np.nan,
            "rolling_rmse_med": np.nan,
            "rolling_rmse_q90": np.nan,
            "naive_skill": np.nan,
            "identity_fit_ratio": np.nan,
            "actual_from_file": actual_from_file,
            "freq": freq,
        }

    err = eval_df["yhat"].to_numpy(dtype=float) - eval_df["actual_obs"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    act_std = float(np.nanstd(eval_df["actual_obs"].to_numpy(dtype=float)))
    tol = max(1e-8, 1e-3 * max(act_std, 1e-6))
    identity_fit_ratio = float(np.mean(np.abs(err) <= tol))

    # Rolling RMSE to avoid one-window bias.
    vals_y = eval_df["yhat"].to_numpy(dtype=float)
    vals_a = eval_df["actual_obs"].to_numpy(dtype=float)
    if freq == "YS":
        w = min(max(4, n_eval // 2), n_eval)
    elif freq == "D":
        w = min(max(30, n_eval // 6), n_eval)
    else:
        w = min(max(12, n_eval // 4), n_eval)
    rolling = []
    if n_eval >= w and w >= 2:
        for i in range(0, n_eval - w + 1):
            e = vals_y[i : i + w] - vals_a[i : i + w]
            rolling.append(float(np.sqrt(np.mean(e * e))))
    rolling_med = float(np.nanmedian(rolling)) if rolling else np.nan
    rolling_q90 = float(np.nanquantile(rolling, 0.9)) if rolling else np.nan

    naive = eval_df["actual_obs"].shift(1)
    ndf = eval_df.assign(naive=naive).dropna(subset=["naive", "actual_obs"])
    if len(ndf) >= 3:
        e_naive = ndf["naive"].to_numpy(dtype=float) - ndf["actual_obs"].to_numpy(dtype=float)
        rmse_naive = float(np.sqrt(np.mean(e_naive * e_naive)))
        if rmse_naive > 1e-9:
            skill = float(1.0 - (rmse / rmse_naive))
        else:
            skill = np.nan
    else:
        skill = np.nan

    return {
        "overlap_points": n_eval,
        "overlap_coverage": coverage,
        "overlap_rmse": rmse,
        "overlap_mae": mae,
        "rolling_rmse_med": rolling_med,
        "rolling_rmse_q90": rolling_q90,
        "naive_skill": skill,
        "identity_fit_ratio": identity_fit_ratio,
        "actual_from_file": actual_from_file,
        "freq": freq,
    }


def add_quant_candidates(run_dir: Path, out: list[Candidate]) -> None:
    idx = run_dir / "calibrated_quant" / "quant_index_to_2035.csv"
    if not idx.exists():
        return
    d = pd.read_csv(idx)
    for _, r in d.iterrows():
        var = str(r.get("variable", "")).strip()
        if not var:
            continue
        out.append(
            Candidate(
                variable=var,
                model_name="quant_calibrated",
                source="calibrated_quant",
                metric_name="cv_rmse",
                metric_value=safe_float(r.get("cv_rmse")),
                forecast_csv=Path(str(r.get("forecast_csv", ""))),
            )
        )


def add_strong_candidates(run_dir: Path, out: list[Candidate]) -> None:
    strong_sets = [
        ("calibrated_strong", run_dir / "calibrated_strong" / "strong_ensemble_index_to_2035.csv"),
        ("calibrated_strong_hp", run_dir / "calibrated_strong_hp" / "strong_ensemble_index_to_2035.csv"),
        ("calibrated_strong_t", run_dir / "calibrated_strong_t" / "strong_ensemble_index_to_2035.csv"),
    ]
    for source_name, idx in strong_sets:
        if not idx.exists():
            continue
        d = pd.read_csv(idx)
        for _, r in d.iterrows():
            var = str(r.get("variable", "")).strip()
            if not var:
                continue
            out.append(
                Candidate(
                    variable=var,
                    model_name=f"strong_{source_name}",
                    source=source_name,
                    metric_name="best_cv_rmse",
                    metric_value=safe_float(r.get("best_cv_rmse")),
                    forecast_csv=Path(str(r.get("forecast_csv", ""))),
                )
            )


def add_prophet_candidates(run_dir: Path, out: list[Candidate]) -> None:
    prophet_sets = [
        ("calibrated_prophet", run_dir / "calibrated_prophet" / "prophet_index_to_2035.csv"),
        ("calibrated_prophet_hp", run_dir / "calibrated_prophet_hp" / "prophet_index_to_2035.csv"),
        ("calibrated_prophet_p", run_dir / "calibrated_prophet_p" / "prophet_index_to_2035.csv"),
    ]
    for source_name, idx in prophet_sets:
        if not idx.exists():
            continue
        d = pd.read_csv(idx)
        for _, r in d.iterrows():
            var = str(r.get("variable", "")).strip()
            if not var:
                continue
            out.append(
                Candidate(
                    variable=var,
                    model_name=f"prophet_{source_name}",
                    source=source_name,
                    metric_name="holdout_rmse",
                    metric_value=safe_float(r.get("holdout_rmse")),
                    forecast_csv=Path(str(r.get("forecast_csv", ""))),
                )
            )


def add_best_meta_candidates(run_dir: Path, out: list[Candidate]) -> None:
    base = run_dir / "best_meta"
    fdir = base / "forecasts"
    if not fdir.exists():
        return

    rmse_proxy_map: dict[str, float] = {}
    rdir = base / "reports"
    if rdir.exists():
        for rp in rdir.glob("*_best_meta_report_to_2035.json"):
            try:
                o = json.loads(rp.read_text(encoding="utf-8"))
                var = str(o.get("variable", "")).strip()
                rmse_proxy_map[var] = safe_float(o.get("rmse_proxy"))
            except Exception:
                continue

    for fp in fdir.glob("*_best_meta_to_2035.csv"):
        try:
            d = pd.read_csv(fp, nrows=2)
            var = str(d.get("variable", pd.Series([""])).dropna().astype(str).iloc[0]).strip()
        except Exception:
            continue
        if not var:
            continue
        out.append(
            Candidate(
                variable=var,
                model_name="best_meta_local",
                source="best_meta",
                metric_name="rmse_proxy",
                metric_value=safe_float(rmse_proxy_map.get(var)),
                forecast_csv=fp,
            )
        )


def _interval_penalty(fut: pd.DataFrame, spread: float) -> tuple[float, list[str]]:
    p = 0.0
    notes: list[str] = []
    lo = pd.to_numeric(fut["yhat_lower"], errors="coerce") if "yhat_lower" in fut.columns else pd.Series(dtype=float)
    hi = pd.to_numeric(fut["yhat_upper"], errors="coerce") if "yhat_upper" in fut.columns else pd.Series(dtype=float)
    if lo.notna().any() and hi.notna().any():
        width = (hi - lo).dropna()
        if len(width):
            w_med = float(np.nanmedian(width))
            wr = w_med / max(spread, 1e-6)
            if wr > 4.8:
                p += 4.0
                notes.append("interval_too_wide")
            elif wr < 0.03:
                p += 1.5
                notes.append("interval_too_narrow")
    else:
        p += 1.0
        notes.append("no_interval")
    return p, notes


def candidate_score(c: Candidate, obs: pd.DataFrame) -> dict[str, Any]:
    if not c.forecast_csv.exists():
        return {
            "ok": False,
            "score_total": np.inf,
            "penalty": 1000.0,
            "note": "forecast_missing",
        }

    d = load_fc(c.forecast_csv)
    hist = d[d["is_forecast"] == False].copy()
    fut = d[d["is_forecast"] == True].copy()

    if fut.empty:
        return {
            "ok": False,
            "score_total": np.inf,
            "penalty": 1000.0,
            "note": "forecast_empty",
        }

    ov = overlap_metrics(fc=d, obs=obs, variable=c.variable)

    y_hist = pd.to_numeric(hist["yhat"], errors="coerce").dropna().to_numpy(dtype=float)
    y_fut = pd.to_numeric(fut["yhat"], errors="coerce").dropna().to_numpy(dtype=float)
    y0 = float(y_fut[0]) if len(y_fut) else np.nan
    y1 = float(y_fut[-1]) if len(y_fut) else np.nan

    # Build a robust spread from overlap actuals if available, else model history.
    if ov["overlap_points"] > 0:
        freq = ov["freq"]
        obs_agg = agg_observations(obs, variable=c.variable, freq=freq)
        hh = hist[["ds", "yhat", "actual"]].copy()
        if hh["actual"].notna().any():
            actual_series = hh["actual"].dropna().to_numpy(dtype=float)
        else:
            merged = hh.merge(obs_agg.rename(columns={"actual": "actual_obs"}), on="ds", how="left")
            actual_series = merged["actual_obs"].dropna().to_numpy(dtype=float)
    else:
        actual_series = np.array([], dtype=float)

    if len(actual_series) >= 5:
        spread = float(np.nanpercentile(actual_series, 90) - np.nanpercentile(actual_series, 10))
    elif len(y_hist) >= 5:
        spread = float(np.nanpercentile(y_hist, 90) - np.nanpercentile(y_hist, 10))
    else:
        spread = float(np.nanstd(y_hist)) if len(y_hist) else 1.0
    spread = max(spread, 1e-6)

    metric_norm = np.nan
    if np.isfinite(c.metric_value):
        metric_norm = float(c.metric_value / spread)

    overlap_norm = np.nan
    if np.isfinite(ov["overlap_rmse"]):
        overlap_norm = float(ov["overlap_rmse"] / spread)

    rolling_norm = np.nan
    if np.isfinite(ov["rolling_rmse_med"]):
        rolling_norm = float(ov["rolling_rmse_med"] / spread)

    # If history part is effectively copied from actual values, do not trust overlap-based errors.
    identity_ratio = safe_float(ov.get("identity_fit_ratio"), np.nan)
    actual_from_file = bool(ov.get("actual_from_file", False))
    if actual_from_file and np.isfinite(identity_ratio) and identity_ratio >= 0.95 and np.isfinite(c.metric_value):
        overlap_norm = np.nan
        rolling_norm = np.nan

    # Weighted base score, robust to missing metric columns.
    comp_vals = []
    comp_w = []
    if np.isfinite(metric_norm):
        comp_vals.append(metric_norm)
        comp_w.append(0.45)
    if np.isfinite(overlap_norm):
        comp_vals.append(overlap_norm)
        comp_w.append(0.35)
    if np.isfinite(rolling_norm):
        comp_vals.append(rolling_norm)
        comp_w.append(0.20)
    if comp_w:
        base = float(np.average(comp_vals, weights=comp_w))
    else:
        base = np.inf

    penalty = 0.0
    notes: list[str] = []

    lo, hi = RANGES.get(c.variable, (-np.inf, np.inf))
    if not np.isfinite(y0) or not np.isfinite(y1):
        penalty += 100.0
        notes.append("nan_forecast")

    if len(y_fut):
        if np.nanmin(y_fut) < lo or np.nanmax(y_fut) > hi:
            penalty += 30.0
            notes.append("range_violation")

    e_lo, e_hi = END_BOUNDS.get(c.variable, (-np.inf, np.inf))
    if np.isfinite(y1) and (y1 < e_lo or y1 > e_hi):
        penalty += 10.0
        notes.append("end_bound")

    if len(y_hist) and len(y_fut):
        last = float(y_hist[-1])
        jump = abs(y0 - last) / spread
        if jump > 4.0:
            penalty += 8.0
            notes.append("jump")
        dmax = DELTA_ABS_MAX.get(c.variable)
        if dmax is not None and abs(y1 - y0) > dmax:
            penalty += 6.0
            notes.append("delta_excess")

    if len(y_hist) >= 12 and len(y_fut) >= 12:
        hist_vol = float(np.nanstd(np.diff(y_hist[-24:])))
        fut_vol = float(np.nanstd(np.diff(y_fut[:24])))
        if hist_vol > 1e-9 and fut_vol / hist_vol > 6.0:
            penalty += 6.0
            notes.append("volatile")

    if len(y_hist) >= 24 and len(y_fut) >= 24:
        h_ref = float(np.nanmedian(y_hist[-60:])) if len(y_hist) >= 60 else float(np.nanmedian(y_hist))
        f_ref = float(np.nanmedian(y_fut))
        if np.isfinite(h_ref) and abs(h_ref) > 1e-9 and np.isfinite(f_ref):
            ratio = f_ref / h_ref
            if c.variable == "pressure" and (ratio < 0.95 or ratio > 1.05):
                penalty += 8.0
                notes.append("level_shift")
            elif c.variable == "humidity" and (ratio < 0.84 or ratio > 1.16):
                penalty += 6.0
                notes.append("level_shift")
            elif c.variable == "temp" and (ratio < 0.60 or ratio > 1.65):
                penalty += 6.0
                notes.append("level_shift")
            elif c.variable == "precip" and (ratio < 0.45 or ratio > 2.10):
                penalty += 8.0
                notes.append("level_shift")

    if len(y_hist) >= 36 and len(y_fut) >= 36:
        hh = y_hist[-120:] if len(y_hist) >= 120 else y_hist
        ff = y_fut
        h_mid = len(hh) // 2
        f_mid = len(ff) // 2
        h1 = float(np.nanmedian(hh[:h_mid]))
        h2 = float(np.nanmedian(hh[h_mid:]))
        f1 = float(np.nanmedian(ff[:f_mid]))
        f2 = float(np.nanmedian(ff[f_mid:]))
        drift_hist = abs(h2 - h1)
        drift_fut = abs(f2 - f1)
        drift_floor = {"temp": 1.5, "humidity": 2.5, "pressure": 3.0, "precip": 15.0}.get(c.variable, 1.0)
        if np.isfinite(drift_fut) and np.isfinite(drift_hist):
            if drift_fut > max(drift_floor, 3.0 * drift_hist):
                penalty += 7.0
                notes.append("trend_drift")

    int_pen, int_notes = _interval_penalty(fut=fut, spread=spread)
    penalty += int_pen
    notes.extend(int_notes)

    # Coverage penalty: even good-fit candidates are downweighted if history overlap is sparse.
    cov = safe_float(ov["overlap_coverage"], 0.0)
    if cov < 0.90:
        penalty += (0.90 - cov) * 10.0
        notes.append("low_overlap_coverage")

    min_overlap = MIN_OVERLAP.get(str(ov["freq"]), 12)
    if safe_float(ov["overlap_points"], 0.0) < float(min_overlap):
        penalty += 3.0
        notes.append("low_overlap_points")

    if np.isfinite(ov["naive_skill"]) and ov["naive_skill"] < -0.2:
        penalty += 3.0
        notes.append("below_naive")

    if actual_from_file and np.isfinite(identity_ratio) and identity_ratio >= 0.95:
        penalty += 4.0
        notes.append("overlap_identity")

    score_total = base + penalty
    return {
        "ok": np.isfinite(score_total),
        "score_base": base,
        "penalty": penalty,
        "score_total": score_total,
        "note": ";".join(sorted(set(notes))),
        "forecast_start": str(pd.to_datetime(fut["ds"].iloc[0]).date()),
        "forecast_end": str(pd.to_datetime(fut["ds"].iloc[-1]).date()),
        "forecast_start_yhat": y0,
        "forecast_end_yhat_2035": y1,
        "frequency": str(ov["freq"]),
        "overlap_points": int(ov["overlap_points"]),
        "overlap_coverage": float(ov["overlap_coverage"]),
        "overlap_rmse": safe_float(ov["overlap_rmse"]),
        "overlap_mae": safe_float(ov["overlap_mae"]),
        "rolling_rmse_med": safe_float(ov["rolling_rmse_med"]),
        "rolling_rmse_q90": safe_float(ov["rolling_rmse_q90"]),
        "naive_skill": safe_float(ov["naive_skill"]),
        "identity_fit_ratio": safe_float(ov["identity_fit_ratio"]),
        "actual_from_file": actual_from_file,
    }


def make_weighted_blend(
    var: str,
    top2: pd.DataFrame,
    blend_dir: Path,
) -> tuple[Candidate, Path]:
    top2 = top2.sort_values("score_total").reset_index(drop=True)
    r1 = top2.iloc[0]
    r2 = top2.iloc[1]

    s1 = max(safe_float(r1["score_total"], 1e-6), 1e-6)
    s2 = max(safe_float(r2["score_total"], 1e-6), 1e-6)
    w1 = 1.0 / s1
    w2 = 1.0 / s2
    ww = w1 + w2
    w1 /= ww
    w2 /= ww

    d1 = load_fc(Path(str(r1["forecast_csv"])))
    d2 = load_fc(Path(str(r2["forecast_csv"])))

    keys = ["ds", "is_forecast"]
    a = d1[keys + ["yhat", "yhat_lower", "yhat_upper", "actual"]].copy().rename(
        columns={
            "yhat": "yhat_1",
            "yhat_lower": "yhat_lower_1",
            "yhat_upper": "yhat_upper_1",
            "actual": "actual_1",
        }
    )
    b = d2[keys + ["yhat", "yhat_lower", "yhat_upper", "actual"]].copy().rename(
        columns={
            "yhat": "yhat_2",
            "yhat_lower": "yhat_lower_2",
            "yhat_upper": "yhat_upper_2",
            "actual": "actual_2",
        }
    )

    m = a.merge(b, on=["ds", "is_forecast"], how="outer").sort_values("ds")
    for c in ["yhat_1", "yhat_2", "yhat_lower_1", "yhat_lower_2", "yhat_upper_1", "yhat_upper_2", "actual_1", "actual_2"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")

    y1 = m["yhat_1"]
    y2 = m["yhat_2"]
    m["yhat"] = np.where(y1.notna() & y2.notna(), w1 * y1 + w2 * y2, np.where(y1.notna(), y1, y2))

    lo1, lo2 = m["yhat_lower_1"], m["yhat_lower_2"]
    hi1, hi2 = m["yhat_upper_1"], m["yhat_upper_2"]
    m["yhat_lower"] = np.where(lo1.notna() & lo2.notna(), w1 * lo1 + w2 * lo2, np.where(lo1.notna(), lo1, lo2))
    m["yhat_upper"] = np.where(hi1.notna() & hi2.notna(), w1 * hi1 + w2 * hi2, np.where(hi1.notna(), hi1, hi2))

    ac1, ac2 = m["actual_1"], m["actual_2"]
    m["actual"] = np.where(ac1.notna(), ac1, ac2)

    m["variable"] = var
    m["model_strategy"] = f"blend[{r1['model_name']}:{w1:.3f}+{r2['model_name']}:{w2:.3f}]"

    out = m[["ds", "actual", "yhat", "yhat_lower", "yhat_upper", "is_forecast", "variable", "model_strategy"]].copy()
    out["ds"] = pd.to_datetime(out["ds"]).dt.strftime("%Y-%m-%d")

    blend_dir.mkdir(parents=True, exist_ok=True)
    fp = blend_dir / f"{var}_v5_top2_blend_to_2035.csv"
    out.to_csv(fp, index=False)

    metric_blend = np.nan
    mv1 = safe_float(r1["metric_value"])
    mv2 = safe_float(r2["metric_value"])
    if np.isfinite(mv1) and np.isfinite(mv2):
        metric_blend = w1 * mv1 + w2 * mv2
    elif np.isfinite(mv1):
        metric_blend = mv1
    elif np.isfinite(mv2):
        metric_blend = mv2

    c = Candidate(
        variable=var,
        model_name=f"blend_v5_{r1['model_name']}__{r2['model_name']}",
        source="v5_internal_blend",
        metric_name="blend_metric",
        metric_value=metric_blend,
        forecast_csv=fp,
    )
    return c, fp


def collect_candidates(run_dir: Path) -> list[Candidate]:
    out: list[Candidate] = []
    add_quant_candidates(run_dir, out)
    add_strong_candidates(run_dir, out)
    add_prophet_candidates(run_dir, out)
    add_best_meta_candidates(run_dir, out)

    dedup: dict[tuple[str, str], Candidate] = {}
    for c in out:
        key = (c.variable, str(c.forecast_csv))
        if key not in dedup:
            dedup[key] = c
    return list(dedup.values())


def render_dashboard(winners: pd.DataFrame, out_png: Path) -> None:
    variables = sorted(winners["variable"].dropna().astype(str).unique().tolist())
    n_vars = max(1, len(variables))
    ncols = 2 if n_vars > 1 else 1
    nrows = int(np.ceil(n_vars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8 * nrows))
    axes = axes.ravel()

    for ax, var in zip(axes, variables):
        w = winners[winners["variable"] == var]
        if w.empty:
            ax.set_axis_off()
            continue
        row = w.iloc[0]
        d = load_fc(Path(str(row["forecast_csv"])))
        hist = d[d["is_forecast"] == False].copy()
        fut = d[d["is_forecast"] == True].copy()

        ax.plot(pd.to_datetime(hist["ds"]), pd.to_numeric(hist["yhat"], errors="coerce"), color="#4c78a8", lw=1.2, label="History")
        ax.plot(pd.to_datetime(fut["ds"]), pd.to_numeric(fut["yhat"], errors="coerce"), color="#f58518", lw=1.5, label="Forecast")

        lo = pd.to_numeric(fut["yhat_lower"], errors="coerce")
        hi = pd.to_numeric(fut["yhat_upper"], errors="coerce")
        if lo.notna().any() and hi.notna().any():
            ax.fill_between(pd.to_datetime(fut["ds"]), lo, hi, color="#f58518", alpha=0.18, label="Interval")

        ax.set_title(
            f"{var} | {row['model_name']} | 2035={safe_float(row['forecast_end_yhat_2035']):.3f} | "
            f"score={safe_float(row['score_total']):.2f} | conf={safe_float(row['confidence_score']):.1f} ({row['confidence_grade']})"
        )
        ax.grid(alpha=0.25)
        note = str(row.get("note", "")).strip()
        if note:
            ax.text(0.01, 0.04, f"note: {note}", transform=ax.transAxes, fontsize=8, color="#a33")

    for ax in axes[len(variables) :]:
        ax.set_axis_off()

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Final v5: Robust Arbitration + Overlap Validation", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def render_confidence_panel(winners: pd.DataFrame, out_png: Path) -> None:
    cfig, cax = plt.subplots(figsize=(10, 5))
    w2 = winners.sort_values("variable").copy()
    bars = cax.bar(w2["variable"], w2["confidence_score"], color="#4c78a8", alpha=0.88)

    for i, (_, r) in enumerate(w2.iterrows()):
        cax.text(
            i,
            safe_float(r["confidence_score"]) + 1.0,
            f"{safe_float(r['confidence_score']):.1f}\n{r['confidence_grade']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for b in bars:
        b.set_linewidth(0.8)
        b.set_edgecolor("#2f4b7c")

    cax.set_ylim(0, 105)
    cax.set_ylabel("Confidence Score")
    cax.set_title("v5 Winner Confidence")
    cax.grid(axis="y", alpha=0.25)
    cfig.tight_layout()
    cfig.savefig(out_png, dpi=160)
    plt.close(cfig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    out_dir = run_dir / "quant" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_path = args.obs_parquet
    if obs_path is None:
        obs_path = run_dir / "calibrated" / "observations_calibrated_recent_regime.parquet"
    obs = load_obs(obs_path)

    candidates = collect_candidates(run_dir)

    base_rows: list[dict[str, Any]] = []
    for c in candidates:
        sc = candidate_score(c, obs=obs)
        base_rows.append(
            {
                "variable": c.variable,
                "model_name": c.model_name,
                "source": c.source,
                "metric_name": c.metric_name,
                "metric_value": c.metric_value,
                "forecast_csv": str(c.forecast_csv),
                **sc,
            }
        )

    scored = pd.DataFrame(base_rows)

    # Add blend candidates when top-2 are close for each variable.
    blend_dir = run_dir / "quant" / "forecasts"
    blend_rows: list[dict[str, Any]] = []
    for var in sorted(scored["variable"].dropna().unique()):
        pool = scored[(scored["variable"] == var) & (scored["ok"] == True)].sort_values("score_total").reset_index(drop=True)
        if len(pool) < 2:
            continue
        s1 = safe_float(pool.loc[0, "score_total"], np.inf)
        s2 = safe_float(pool.loc[1, "score_total"], np.inf)
        if not np.isfinite(s1) or not np.isfinite(s2):
            continue
        gap_ratio = (s2 - s1) / max(s1, 1e-6)
        if gap_ratio > args.blend_threshold:
            continue

        blend_cand, _ = make_weighted_blend(var=var, top2=pool.iloc[:2], blend_dir=blend_dir)
        scb = candidate_score(blend_cand, obs=obs)
        blend_rows.append(
            {
                "variable": blend_cand.variable,
                "model_name": blend_cand.model_name,
                "source": blend_cand.source,
                "metric_name": blend_cand.metric_name,
                "metric_value": blend_cand.metric_value,
                "forecast_csv": str(blend_cand.forecast_csv),
                **scb,
                "blend_gap_ratio": gap_ratio,
            }
        )

    if blend_rows:
        scored = pd.concat([scored, pd.DataFrame(blend_rows)], ignore_index=True, sort=False)

    cand_csv = out_dir / "v5_candidate_scores.csv"
    scored.sort_values(["variable", "score_total", "penalty"]).to_csv(cand_csv, index=False)

    sorted_ok = scored[scored["ok"] == True].sort_values(
        ["variable", "score_total", "penalty", "metric_value"],
        ascending=[True, True, True, True],
    )
    winners = sorted_ok.drop_duplicates(subset=["variable"], keep="first").reset_index(drop=True)

    conf_vals = winners.apply(
        lambda r: confidence_from_score(
            r.get("score_total"),
            r.get("penalty"),
            r.get("overlap_coverage"),
            r.get("overlap_points"),
            r.get("note", ""),
        ),
        axis=1,
    )
    winners["confidence_score"] = [x[0] for x in conf_vals]
    winners["confidence_grade"] = [x[1] for x in conf_vals]

    winners_csv = out_dir / "v5_final_arbitrated_ozet.csv"
    winners.to_csv(winners_csv, index=False)

    dash_png = out_dir / "v5_final_arbitrated_dashboard.png"
    render_dashboard(winners=winners, out_png=dash_png)

    conf_png = out_dir / "v5_confidence_panel.png"
    render_confidence_panel(winners=winners, out_png=conf_png)

    md_lines = [
        "# Final v5 Arbitration Summary",
        "",
        f"- Candidate table: `{cand_csv}`",
        f"- Final winners: `{winners_csv}`",
        f"- Dashboard: `{dash_png}`",
        f"- Confidence panel: `{conf_png}`",
        f"- Observation parquet: `{obs_path}`",
        "",
    ]

    for _, r in winners.sort_values("variable").iterrows():
        md_lines.append(
            f"- {r['variable']}: {r['model_name']} | 2035={safe_float(r['forecast_end_yhat_2035']):.3f} | "
            f"score={safe_float(r['score_total']):.3f} | overlap_rmse={safe_float(r.get('overlap_rmse')):.3f} | "
            f"coverage={safe_float(r.get('overlap_coverage')):.2f} | confidence={safe_float(r['confidence_score']):.1f} ({r['confidence_grade']})"
        )

    md_path = out_dir / "v5_final_arbitrated_yorum.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "obs_parquet": str(obs_path),
        "candidate_count": int(len(scored)),
        "winner_count": int(len(winners)),
        "blend_candidate_count": int(len(blend_rows)),
        "outputs": {
            "candidate_scores_csv": str(cand_csv),
            "final_winners_csv": str(winners_csv),
            "dashboard_png": str(dash_png),
            "confidence_panel_png": str(conf_png),
            "comment_md": str(md_path),
        },
    }
    summary_json = out_dir / "v5_final_arbitrated_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
