#!/usr/bin/env python3
"""Build v6 stable consensus forecast from v5 candidates.

Goal:
- Improve stability and robustness by combining top validated models.
- Apply post-blend calibration: continuity, trend guard, bias correction,
  seasonal amplitude guard, and interval recalibration.
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

OBS_AGG = {
    "temp": "mean",
    "humidity": "mean",
    "pressure": "mean",
    "precip": "sum",
}


@dataclass
class Member:
    model_name: str
    forecast_csv: Path
    score_total: float
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v6 stable consensus forecast")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/data_factory/run_20260306_000419"),
    )
    p.add_argument(
        "--obs-parquet",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum members per variable for consensus.",
    )
    return p.parse_args()


def sf(x: Any, d: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return d


def infer_freq(ds: pd.Series) -> str:
    s = pd.to_datetime(ds, errors="coerce").dropna().sort_values()
    if len(s) < 3:
        return "MS"
    diff_days = s.diff().dt.total_seconds().dropna() / (24 * 3600)
    if diff_days.empty:
        return "MS"
    med = float(np.nanmedian(diff_days))
    if med <= 2:
        return "D"
    if med <= 45:
        return "MS"
    return "YS"


def load_obs(obs_path: Path) -> pd.DataFrame:
    d = pd.read_parquet(obs_path)
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["timestamp", "variable", "value"])
    return d[["timestamp", "variable", "value"]].copy()


def agg_obs(obs: pd.DataFrame, variable: str, freq: str) -> pd.DataFrame:
    d = obs[obs["variable"] == variable].copy()
    if d.empty:
        return pd.DataFrame(columns=["ds", "actual"])

    if freq == "YS":
        d["ds"] = d["timestamp"].dt.to_period("Y").dt.to_timestamp()
    elif freq == "D":
        d["ds"] = d["timestamp"].dt.floor("D")
    else:
        d["ds"] = d["timestamp"].dt.to_period("M").dt.to_timestamp()

    agg = OBS_AGG.get(variable, "mean")
    g = d.groupby("ds", as_index=False)["value"]
    if agg == "sum":
        out = g.sum().rename(columns={"value": "actual"})
    else:
        out = g.mean().rename(columns={"value": "actual"})
    return out.sort_values("ds")


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


def choose_members(cands: pd.DataFrame, variable: str, top_k: int) -> list[Member]:
    g = cands[(cands["variable"] == variable) & (cands["ok"] == True)].copy()
    if g.empty:
        return []

    # Hard rejects.
    bad = g["note"].fillna("").str.contains("range_violation|forecast_missing|forecast_empty", regex=True)
    g = g[~bad].copy()
    if g.empty:
        return []

    g = g.sort_values("score_total")
    g_all = g.copy()
    best = sf(g["score_total"].iloc[0], np.inf)
    if np.isfinite(best):
        g = g[g["score_total"] <= best * 2.7].copy()

    g = g.head(max(1, top_k)).copy()

    # For sparse or unstable variables, keep a secondary candidate for consensus diversity.
    if variable in {"temp", "precip"} and len(g) < 2 and len(g_all) >= 2:
        fallback_pool = g_all.iloc[1:].copy()
        severe = fallback_pool["note"].fillna("").str.contains("range_violation|forecast_missing|forecast_empty|end_bound|delta_excess", regex=True)
        fallback_good = fallback_pool[~severe]
        add_row = fallback_good.iloc[0] if not fallback_good.empty else fallback_pool.iloc[0]
        if str(add_row.get("model_name", "")) not in set(g["model_name"].astype(str)):
            g = pd.concat([g, add_row.to_frame().T], ignore_index=True)

    out = []
    for _, r in g.iterrows():
        out.append(
            Member(
                model_name=str(r["model_name"]),
                forecast_csv=Path(str(r["forecast_csv"])),
                score_total=sf(r["score_total"], np.inf),
                note=str(r.get("note", "") or ""),
            )
        )
    return out


def member_weights(members: list[Member]) -> dict[str, float]:
    if not members:
        return {}

    raw = []
    names = []
    for m in members:
        w = 1.0 / max(m.score_total, 1e-6)
        n = m.note
        if "overlap_identity" in n:
            w *= 0.55
        if "low_overlap_coverage" in n:
            w *= 0.60
        if "no_interval" in n:
            w *= 0.85
        if "end_bound" in n:
            w *= 0.80
        names.append(m.model_name)
        raw.append(max(w, 1e-9))

    raw = np.array(raw, dtype=float)
    raw = raw / raw.sum()

    # Keep at least minor diversity when >=2 members.
    if len(raw) >= 2:
        raw = np.clip(raw, 0.15, 0.80)
        raw = raw / raw.sum()

    return {n: float(w) for n, w in zip(names, raw)}


def _fit_linear_calibrator(pred: np.ndarray, actual: np.ndarray, variable: str) -> dict[str, Any]:
    x = np.asarray(pred, dtype=float)
    y = np.asarray(actual, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 8:
        return {"usable": False, "a": 0.0, "b": 1.0, "log_space": False}

    if variable == "precip":
        x_t = np.log1p(np.clip(x, 0.0, None))
        y_t = np.log1p(np.clip(y, 0.0, None))
        b_lo, b_hi = 0.35, 1.80
        log_space = True
    elif variable == "pressure":
        x_t = x
        y_t = y
        b_lo, b_hi = 0.80, 1.20
        log_space = False
    elif variable == "humidity":
        x_t = x
        y_t = y
        b_lo, b_hi = 0.60, 1.40
        log_space = False
    else:
        x_t = x
        y_t = y
        b_lo, b_hi = 0.50, 1.50
        log_space = False

    if np.nanstd(x_t) < 1e-9 or len(np.unique(np.round(x_t, 6))) < 3:
        return {"usable": False, "a": 0.0, "b": 1.0, "log_space": log_space}

    try:
        b, a = np.polyfit(x_t, y_t, 1)
    except Exception:
        return {"usable": False, "a": 0.0, "b": 1.0, "log_space": log_space}

    b = float(np.clip(b, b_lo, b_hi))
    a = float(a)
    return {"usable": True, "a": a, "b": b, "log_space": log_space}


def _apply_linear_calibrator(values: pd.Series | np.ndarray, cal: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not bool(cal.get("usable", False)):
        return arr
    a = sf(cal.get("a"), 0.0)
    b = sf(cal.get("b"), 1.0)
    if bool(cal.get("log_space", False)):
        z = np.log1p(np.clip(arr, 0.0, None))
        return np.expm1(a + b * z)
    return a + b * arr


def tune_member_weights(
    members: list[Member],
    obs: pd.DataFrame,
    variable: str,
    default_weights: dict[str, float],
) -> dict[str, float]:
    if len(members) < 2:
        return default_weights

    # Tune first two members; keep any remaining members with reduced fixed share.
    m1, m2 = members[0], members[1]
    w_def_1 = sf(default_weights.get(m1.model_name), 0.7)
    w_def_2 = sf(default_weights.get(m2.model_name), 0.3)
    rem_names = [m.model_name for m in members[2:]]
    rem_sum = float(np.sum([sf(default_weights.get(n), 0.0) for n in rem_names]))
    rem_target = min(0.20, max(0.0, rem_sum))
    lead_total = 1.0 - rem_target

    # Build merged dataframe once.
    wm = {m1.model_name: w_def_1, m2.model_name: w_def_2}
    for n in rem_names:
        wm[n] = sf(default_weights.get(n), 0.0)
    s = np.sum(list(wm.values()))
    if s <= 0:
        return default_weights
    wm = {k: float(v / s) for k, v in wm.items()}
    merged = merge_members(members, wm)
    freq = infer_freq(merged["ds"])

    hist = merged[merged["is_forecast"] == False].copy()
    fut = merged[merged["is_forecast"] == True].copy()
    if hist.empty or fut.empty:
        return default_weights

    obs_agg = agg_obs(obs, variable=variable, freq=freq)

    # Build candidate columns for first two members on history.
    d1 = load_fc(m1.forecast_csv)[["ds", "is_forecast", "yhat"]].rename(columns={"yhat": "y1"})
    d2 = load_fc(m2.forecast_csv)[["ds", "is_forecast", "yhat"]].rename(columns={"yhat": "y2"})
    h = d1.merge(d2, on=["ds", "is_forecast"], how="inner")
    h = h[h["is_forecast"] == False].copy()
    h["ds"] = pd.to_datetime(h["ds"], errors="coerce")
    h["y1"] = pd.to_numeric(h["y1"], errors="coerce")
    h["y2"] = pd.to_numeric(h["y2"], errors="coerce")
    h = h.dropna(subset=["ds", "y1", "y2"])
    if h.empty:
        return default_weights
    h = h.merge(obs_agg.rename(columns={"actual": "actual_obs"}), on="ds", how="left")
    h = h.dropna(subset=["actual_obs"]).sort_values("ds")
    if len(h) < 12:
        return default_weights

    # Future volatility references for objective.
    h_ref = hist["yhat"].dropna().to_numpy(dtype=float)
    f_ref = fut["yhat"].dropna().to_numpy(dtype=float)
    spread = float(np.nanpercentile(h["actual_obs"], 90) - np.nanpercentile(h["actual_obs"], 10))
    spread = max(spread, 1e-6)
    hv_ref = float(np.nanstd(np.diff(h_ref[-min(60, len(h_ref)):]))) if len(h_ref) >= 3 else np.nan
    fv_ref = float(np.nanstd(np.diff(f_ref[:min(60, len(f_ref))]))) if len(f_ref) >= 3 else np.nan
    jump_ref = abs(float(f_ref[0]) - float(h_ref[-1])) / spread if len(h_ref) and len(f_ref) else 0.0

    # Keep secondary member limited if it has suspicious note.
    max_w2 = 0.45
    if "overlap_identity" in (m2.note or ""):
        max_w2 = 0.35
    if "end_bound" in (m2.note or ""):
        max_w2 = min(max_w2, 0.30)

    best_obj = np.inf
    best_w1 = lead_total * max(0.55, min(0.95, w_def_1 / max(w_def_1 + w_def_2, 1e-6)))

    for w1_share in np.linspace(0.55, 0.95, 9):
        w1 = lead_total * w1_share
        w2 = lead_total - w1
        if w2 > max_w2:
            continue
        pred = w1 * h["y1"].to_numpy(dtype=float) + w2 * h["y2"].to_numpy(dtype=float)
        act = h["actual_obs"].to_numpy(dtype=float)
        if len(pred) < 12:
            continue

        split = int(max(8, np.floor(len(pred) * 0.8)))
        split = min(split, len(pred) - 2)
        if split < 6:
            continue
        tr_p, va_p = pred[:split], pred[split:]
        tr_a, va_a = act[:split], act[split:]
        cal = _fit_linear_calibrator(tr_p, tr_a, variable=variable)
        va_pred = _apply_linear_calibrator(va_p, cal)
        rmse = float(np.sqrt(np.mean((va_pred - va_a) ** 2)))

        # Stability terms from baseline merged series.
        obj = rmse / spread
        if np.isfinite(hv_ref) and hv_ref > 1e-9 and np.isfinite(fv_ref):
            obj += 0.12 * max((fv_ref / hv_ref) - 3.0, 0.0)
        obj += 0.08 * max(jump_ref - 3.0, 0.0)

        if obj < best_obj:
            best_obj = obj
            best_w1 = w1

    best_w2 = lead_total - best_w1
    out = {}
    out[m1.model_name] = float(best_w1)
    out[m2.model_name] = float(best_w2)
    for n in rem_names:
        out[n] = float(rem_target * sf(default_weights.get(n), 0.0) / max(rem_sum, 1e-9)) if rem_target > 0 else 0.0

    s2 = np.sum(list(out.values()))
    if s2 <= 0:
        return default_weights
    out = {k: float(v / s2) for k, v in out.items()}
    return out


def merge_members(members: list[Member], weights: dict[str, float]) -> pd.DataFrame:
    base = None
    for m in members:
        d = load_fc(m.forecast_csv)
        d = d[["ds", "is_forecast", "yhat", "yhat_lower", "yhat_upper", "actual"]].copy()
        d = d.rename(
            columns={
                "yhat": f"yhat__{m.model_name}",
                "yhat_lower": f"lo__{m.model_name}",
                "yhat_upper": f"hi__{m.model_name}",
                "actual": f"actual__{m.model_name}",
            }
        )
        if base is None:
            base = d
        else:
            base = base.merge(d, on=["ds", "is_forecast"], how="outer")

    assert base is not None
    base = base.sort_values("ds").reset_index(drop=True)

    y_cols = [c for c in base.columns if c.startswith("yhat__")]
    lo_cols = [c for c in base.columns if c.startswith("lo__")]
    hi_cols = [c for c in base.columns if c.startswith("hi__")]
    ac_cols = [c for c in base.columns if c.startswith("actual__")]

    for c in y_cols + lo_cols + hi_cols + ac_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    def weighted_row(row: pd.Series, pref: str) -> float:
        vals = []
        ws = []
        for m in members:
            c = f"{pref}__{m.model_name}"
            v = sf(row.get(c), np.nan)
            if np.isfinite(v):
                vals.append(v)
                ws.append(weights.get(m.model_name, 0.0))
        if not ws or sum(ws) <= 0:
            return np.nan
        ws = np.array(ws, dtype=float)
        ws = ws / ws.sum()
        vals = np.array(vals, dtype=float)
        return float(np.sum(ws * vals))

    out = base[["ds", "is_forecast"]].copy()
    out["yhat"] = base.apply(lambda r: weighted_row(r, "yhat"), axis=1)
    out["yhat_lower"] = base.apply(lambda r: weighted_row(r, "lo"), axis=1)
    out["yhat_upper"] = base.apply(lambda r: weighted_row(r, "hi"), axis=1)

    # Pick first available actual (all should be same where present).
    out["actual"] = np.nan
    for c in ac_cols:
        out["actual"] = out["actual"].where(out["actual"].notna(), base[c])

    return out


def _trend_strength(y: np.ndarray) -> float:
    if len(y) < 6:
        return 0.0
    mid = len(y) // 2
    a = float(np.nanmedian(y[:mid]))
    b = float(np.nanmedian(y[mid:]))
    return b - a


def stabilize_future(df: pd.DataFrame, variable: str, freq: str) -> pd.DataFrame:
    d = df.copy()
    hist = d[d["is_forecast"] == False].copy()
    fut = d[d["is_forecast"] == True].copy()
    if fut.empty:
        return d

    y_hist = pd.to_numeric(hist["yhat"], errors="coerce").dropna().to_numpy(dtype=float)
    y_fut = pd.to_numeric(fut["yhat"], errors="coerce").to_numpy(dtype=float)

    if len(y_hist):
        last = float(y_hist[-1])
        # Continuity relaxation: do not allow large jump at forecast start.
        jump = y_fut[0] - last
        n = len(y_fut)
        decay = np.exp(-np.arange(n) / 3.0)
        y_fut = y_fut - jump * decay

    # Seasonal anomaly smoothing for monthly series.
    if freq == "MS" and len(y_hist) >= 12 and len(y_fut) >= 6:
        h_idx = pd.to_datetime(hist["ds"], errors="coerce")
        f_idx = pd.to_datetime(fut["ds"], errors="coerce")
        h_df = pd.DataFrame({"ds": h_idx, "y": y_hist[: len(h_idx)]}).dropna()
        f_df = pd.DataFrame({"ds": f_idx, "y": y_fut[: len(f_idx)]}).dropna()
        if not h_df.empty and not f_df.empty:
            h_df["m"] = h_df["ds"].dt.month
            climo = h_df.groupby("m")["y"].median().to_dict()
            clim = f_df["ds"].dt.month.map(climo).astype(float).to_numpy()
            if np.isfinite(clim).sum() >= len(clim) // 2:
                miss = ~np.isfinite(clim)
                if miss.any():
                    clim[miss] = float(np.nanmedian(y_hist))
                anom = y_fut - clim
                alpha_map = {"temp": 0.20, "humidity": 0.15, "precip": 0.12}
                alpha = alpha_map.get(variable, 0.18)
                anom_sm = pd.Series(anom).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=float)
                y_fut = clim + (0.70 * anom_sm + 0.30 * anom)

    # Trend guard.
    if len(y_hist) >= 10 and len(y_fut) >= 10:
        hh = y_hist[-120:] if len(y_hist) >= 120 else y_hist
        drift_hist = abs(_trend_strength(hh))
        drift_fut = abs(_trend_strength(y_fut))
        floor = {"temp": 1.5, "humidity": 2.5, "pressure": 3.0, "precip": 20.0}.get(variable, 1.0)
        allowed = max(floor, 2.2 * drift_hist)
        if drift_fut > allowed and drift_fut > 1e-9:
            scale = allowed / drift_fut
            y0 = y_fut[0]
            y_fut = y0 + (y_fut - y0) * scale

    # Variable-specific long-horizon guards.
    if len(y_fut) >= 2:
        y0 = float(y_fut[0])
        y1 = float(y_fut[-1])
        n = len(y_fut)
        if variable == "temp":
            span_years = max(1.0, n / 12.0) if freq == "MS" else max(1.0, n)
            min_end = y0 + 0.01 * span_years
            if y1 < min_end:
                lift = np.linspace(0.0, min_end - y1, n)
                y_fut = y_fut + lift
        elif variable == "precip":
            min_ratio = 0.60
            if y0 > 1e-9 and y1 < min_ratio * y0:
                target_end = min_ratio * y0
                lift = np.linspace(0.0, target_end - y1, n)
                y_fut = y_fut + lift
        elif variable == "humidity" and len(y_hist) >= 24:
            p95 = float(np.nanquantile(y_hist, 0.95))
            max_end = min(96.0, p95 + 2.0)
            if y1 > max_end:
                drop = np.linspace(0.0, y1 - max_end, n)
                y_fut = y_fut - drop

    # Seasonal amplitude guard for monthly series.
    if freq == "MS" and len(y_hist) >= 24 and len(y_fut) >= 24:
        h_idx = pd.to_datetime(hist["ds"], errors="coerce")
        h = pd.DataFrame({"ds": h_idx, "y": y_hist[: len(h_idx)]}).dropna()
        if not h.empty:
            h["m"] = h["ds"].dt.month
            h_seas = h.groupby("m")["y"].median()
            h_amp = float(np.nanstd(h_seas.values)) if len(h_seas) >= 6 else 0.0

            f_idx = pd.to_datetime(fut["ds"], errors="coerce")
            f = pd.DataFrame({"ds": f_idx, "y": y_fut[: len(f_idx)]}).dropna()
            f["m"] = f["ds"].dt.month
            f_seas = f.groupby("m")["y"].median()
            f_amp = float(np.nanstd(f_seas.values)) if len(f_seas) >= 6 else 0.0

            if h_amp > 1e-9 and f_amp > 1.45 * h_amp:
                scale = (1.45 * h_amp) / f_amp
                f_mean = float(np.nanmean(y_fut))
                y_fut = f_mean + (y_fut - f_mean) * scale

    # Physical clipping.
    lo, hi = RANGES.get(variable, (-np.inf, np.inf))
    y_fut = np.clip(y_fut, lo, hi)

    e_lo, e_hi = END_BOUNDS.get(variable, (-np.inf, np.inf))
    if len(y_fut):
        y_fut[-1] = float(np.clip(y_fut[-1], e_lo, e_hi))

    d.loc[d["is_forecast"] == True, "yhat"] = y_fut
    return d


def calibrate_bias_and_intervals(df: pd.DataFrame, obs: pd.DataFrame, variable: str, freq: str) -> tuple[pd.DataFrame, dict[str, float]]:
    d = df.copy()
    hist = d[d["is_forecast"] == False].copy()
    fut = d[d["is_forecast"] == True].copy()

    obs_agg = agg_obs(obs, variable=variable, freq=freq)

    def build_hist_eval(frame: pd.DataFrame) -> pd.DataFrame:
        hh = frame[frame["is_forecast"] == False][["ds", "yhat", "actual"]].copy()
        # Prefer real observations for calibration; file-level "actual" may be synthetic in some pipelines.
        hh_obs = hh.drop(columns=["actual"]).merge(obs_agg.rename(columns={"actual": "actual_obs"}), on="ds", how="left")
        if hh_obs["actual_obs"].notna().sum() == 0 and hh["actual"].notna().any():
            # Fallback only when no observation overlap exists at all.
            hh_obs["actual_obs"] = hh["actual"].to_numpy()
        hh = hh_obs
        hh["ds"] = pd.to_datetime(hh["ds"], errors="coerce")
        hh["yhat"] = pd.to_numeric(hh["yhat"], errors="coerce")
        hh["actual_obs"] = pd.to_numeric(hh["actual_obs"], errors="coerce")
        return hh.dropna(subset=["ds", "yhat", "actual_obs"]).sort_values("ds")

    h_eval = build_hist_eval(d)

    if h_eval.empty:
        return d, {"bias": np.nan, "rmse": np.nan, "mae": np.nan}

    pred_hist = h_eval["yhat"].to_numpy(dtype=float)
    act_hist = h_eval["actual_obs"].to_numpy(dtype=float)

    # Linear calibration (slope + intercept), fallback to pure bias when unstable.
    cal = _fit_linear_calibrator(pred_hist, act_hist, variable=variable)
    y_all = pd.to_numeric(d["yhat"], errors="coerce")
    y_cal = _apply_linear_calibrator(y_all, cal)
    d["yhat"] = y_cal

    # Precipitation: select the best monthly calibration mode by time holdout.
    if variable == "precip" and freq == "MS":
        h_local = build_hist_eval(d)
        if len(h_local) >= 60:
            h_local["m"] = h_local["ds"].dt.month
            h_local = h_local.sort_values("ds").reset_index(drop=True)

            def nearest_month(month: int, keys: list[int]) -> int | None:
                if not keys:
                    return None
                def cyc_dist(a: int, b: int) -> int:
                    d0 = abs(a - b)
                    return min(d0, 12 - d0)
                return sorted(keys, key=lambda kk: cyc_dist(month, kk))[0]

            def fit_month_bias(train: pd.DataFrame) -> dict[int, float]:
                out: dict[int, float] = {}
                for mm, g in train.groupby("m"):
                    e = pd.to_numeric(g["yhat"], errors="coerce").to_numpy(dtype=float) - pd.to_numeric(g["actual_obs"], errors="coerce").to_numpy(dtype=float)
                    e = e[np.isfinite(e)]
                    if len(e) < 4:
                        continue
                    b_m = float(np.nanmedian(e))
                    shrink = float(len(e) / (len(e) + 12.0))
                    out[int(mm)] = shrink * b_m
                return out

            def apply_month_bias(vals: np.ndarray, months: np.ndarray, params: dict[int, float]) -> np.ndarray:
                out = vals.copy()
                keys = sorted(params.keys())
                if not keys:
                    return out
                for i in range(len(out)):
                    if not np.isfinite(out[i]) or not np.isfinite(months[i]):
                        continue
                    mm = int(months[i])
                    if mm not in params:
                        nm = nearest_month(mm, keys)
                        if nm is None:
                            continue
                        corr = params[nm]
                    else:
                        corr = params[mm]
                    out[i] = out[i] - corr
                return out

            def fit_month_linear(train: pd.DataFrame) -> dict[int, tuple[float, float]]:
                out: dict[int, tuple[float, float]] = {}
                for mm, g in train.groupby("m"):
                    x = pd.to_numeric(g["yhat"], errors="coerce").to_numpy(dtype=float)
                    y = pd.to_numeric(g["actual_obs"], errors="coerce").to_numpy(dtype=float)
                    mask = np.isfinite(x) & np.isfinite(y)
                    x = x[mask]
                    y = y[mask]
                    if len(x) < 8 or np.nanstd(x) < 1e-9:
                        continue
                    try:
                        b_m, a_m = np.polyfit(x, y, 1)
                    except Exception:
                        continue
                    b_m = float(np.clip(b_m, 0.20, 2.20))
                    a_m = float(a_m)
                    shrink = float(len(x) / (len(x) + 14.0))
                    b_m = 1.0 + shrink * (b_m - 1.0)
                    a_m = shrink * a_m
                    out[int(mm)] = (a_m, b_m)
                return out

            def apply_month_linear(vals: np.ndarray, months: np.ndarray, params: dict[int, tuple[float, float]]) -> np.ndarray:
                out = vals.copy()
                keys = sorted(params.keys())
                if not keys:
                    return out
                for i in range(len(out)):
                    if not np.isfinite(out[i]) or not np.isfinite(months[i]):
                        continue
                    mm = int(months[i])
                    if mm not in params:
                        nm = nearest_month(mm, keys)
                        if nm is None:
                            continue
                        a_m, b_m = params[nm]
                    else:
                        a_m, b_m = params[mm]
                    out[i] = a_m + b_m * out[i]
                return out

            split = int(max(36, np.floor(len(h_local) * 0.80)))
            split = min(split, len(h_local) - 8)
            split = max(split, 24)
            tr = h_local.iloc[:split].copy()
            va = h_local.iloc[split:].copy()

            modes: list[tuple[str, float, Any]] = []

            # none
            p_none = va["yhat"].to_numpy(dtype=float)
            a_val = va["actual_obs"].to_numpy(dtype=float)
            rmse_none = float(np.sqrt(np.mean((p_none - a_val) ** 2))) if len(va) else np.inf
            modes.append(("none", rmse_none, None))

            # month_bias
            mb = fit_month_bias(tr)
            if mb:
                p_mb = apply_month_bias(
                    va["yhat"].to_numpy(dtype=float),
                    va["m"].to_numpy(dtype=float),
                    mb,
                )
                rmse_mb = float(np.sqrt(np.mean((p_mb - a_val) ** 2))) if len(va) else np.inf
                modes.append(("month_bias", rmse_mb, mb))

            # month_linear
            ml = fit_month_linear(tr)
            if ml:
                p_ml = apply_month_linear(
                    va["yhat"].to_numpy(dtype=float),
                    va["m"].to_numpy(dtype=float),
                    ml,
                )
                rmse_ml = float(np.sqrt(np.mean((p_ml - a_val) ** 2))) if len(va) else np.inf
                modes.append(("month_linear", rmse_ml, ml))

            best_mode, _, _ = sorted(modes, key=lambda x: x[1])[0]

            # Refit chosen mode on full history and apply to all rows.
            ds_all = pd.to_datetime(d["ds"], errors="coerce")
            months_all = ds_all.dt.month.to_numpy(dtype=float)
            y_all2 = pd.to_numeric(d["yhat"], errors="coerce").to_numpy(dtype=float)

            if best_mode == "month_bias":
                mb_full = fit_month_bias(h_local)
                y_all2 = apply_month_bias(y_all2, months_all, mb_full)
            elif best_mode == "month_linear":
                ml_full = fit_month_linear(h_local)
                y_all2 = apply_month_linear(y_all2, months_all, ml_full)

            d["yhat"] = y_all2

    h_eval2 = build_hist_eval(d)
    pred_hist_cal = h_eval2["yhat"].to_numpy(dtype=float)
    act_hist2 = h_eval2["actual_obs"].to_numpy(dtype=float)
    err_centered = pred_hist_cal - act_hist2
    bias = float(np.nanmedian(err_centered))

    # Final median bias centering.
    d["yhat"] = pd.to_numeric(d["yhat"], errors="coerce") - bias
    err_centered = err_centered - bias

    rmse = float(np.sqrt(np.mean(err_centered * err_centered)))
    mae = float(np.mean(np.abs(err_centered)))

    q10 = float(np.nanquantile(err_centered, 0.10)) if len(err_centered) >= 8 else -1.28 * rmse
    q90 = float(np.nanquantile(err_centered, 0.90)) if len(err_centered) >= 8 else 1.28 * rmse

    fut_mask = d["is_forecast"] == True
    y = pd.to_numeric(d.loc[fut_mask, "yhat"], errors="coerce")

    lo = pd.to_numeric(d.loc[fut_mask, "yhat_lower"], errors="coerce")
    hi = pd.to_numeric(d.loc[fut_mask, "yhat_upper"], errors="coerce")

    # If intervals missing or degenerate, rebuild from residual quantiles.
    rebuild = (not lo.notna().any()) or (not hi.notna().any())
    if not rebuild:
        width = (hi - lo).dropna()
        rebuild = width.empty or float(np.nanmedian(width)) <= 0

    if rebuild:
        d.loc[fut_mask, "yhat_lower"] = y + q10
        d.loc[fut_mask, "yhat_upper"] = y + q90
    else:
        # Keep shape but re-center around corrected yhat.
        half = (hi - lo) / 2.0
        half = half.fillna(float(np.nanmedian(half.dropna())) if half.notna().any() else max(0.2, rmse))
        d.loc[fut_mask, "yhat_lower"] = y - half
        d.loc[fut_mask, "yhat_upper"] = y + half

    # Precipitation-specific guard for months with no observed history (e.g., missing Nov/Dec).
    if variable == "precip" and freq == "MS":
        hist_for_climo = h_eval2.copy()
        hist_for_climo["m"] = hist_for_climo["ds"].dt.month
        climo = hist_for_climo.groupby("m")["actual_obs"].median().to_dict()
        present_months = set(int(k) for k in climo.keys())

        def nearest_anchor(month: int) -> float:
            if not climo:
                return np.nan
            months = list(climo.keys())
            months = [int(m) for m in months]

            def cyc_dist(a: int, b: int) -> int:
                d = abs(a - b)
                return min(d, 12 - d)

            ms = sorted(months, key=lambda mm: cyc_dist(month, mm))
            vals = [float(climo[m]) for m in ms[:2] if np.isfinite(float(climo[m]))]
            if not vals:
                return np.nan
            return float(np.mean(vals))

        fut_idx = d.index[fut_mask]
        y_arr = pd.to_numeric(d.loc[fut_mask, "yhat"], errors="coerce").to_numpy(dtype=float)
        half_arr = (
            (pd.to_numeric(d.loc[fut_mask, "yhat_upper"], errors="coerce") - pd.to_numeric(d.loc[fut_mask, "yhat_lower"], errors="coerce"))
            / 2.0
        ).to_numpy(dtype=float)
        default_half = float(np.nanmedian(half_arr[np.isfinite(half_arr)])) if np.isfinite(half_arr).any() else max(0.2, rmse)
        half_arr = np.where(np.isfinite(half_arr), half_arr, default_half)

        ds_f = pd.to_datetime(d.loc[fut_mask, "ds"], errors="coerce")
        for i in range(len(y_arr)):
            if not np.isfinite(y_arr[i]) or pd.isna(ds_f.iloc[i]):
                continue
            mm = int(ds_f.iloc[i].month)
            anch = nearest_anchor(mm)
            if not np.isfinite(anch):
                continue
            if mm not in present_months:
                # Pull toward plausible seasonal anchor if that month was never observed.
                y_arr[i] = 0.55 * y_arr[i] + 0.45 * anch
                lo_a = 0.35 * anch
                hi_a = 2.80 * anch
            else:
                lo_a = 0.20 * anch
                hi_a = 3.00 * anch
            y_arr[i] = float(np.clip(y_arr[i], lo_a, hi_a))

        d.loc[fut_mask, "yhat"] = y_arr
        d.loc[fut_mask, "yhat_lower"] = y_arr - half_arr
        d.loc[fut_mask, "yhat_upper"] = y_arr + half_arr

    # Keep future yhat within physical bounds.
    r_lo, r_hi = RANGES.get(variable, (-np.inf, np.inf))
    d.loc[fut_mask, "yhat"] = pd.to_numeric(d.loc[fut_mask, "yhat"], errors="coerce").clip(lower=r_lo, upper=r_hi)

    # Hard clip intervals.
    d.loc[fut_mask, "yhat_lower"] = pd.to_numeric(d.loc[fut_mask, "yhat_lower"], errors="coerce").clip(lower=r_lo, upper=r_hi)
    d.loc[fut_mask, "yhat_upper"] = pd.to_numeric(d.loc[fut_mask, "yhat_upper"], errors="coerce").clip(lower=r_lo, upper=r_hi)

    # Ensure ordering.
    lo2 = pd.to_numeric(d.loc[fut_mask, "yhat_lower"], errors="coerce")
    hi2 = pd.to_numeric(d.loc[fut_mask, "yhat_upper"], errors="coerce")
    y2 = pd.to_numeric(d.loc[fut_mask, "yhat"], errors="coerce")
    lo_fix = np.minimum(lo2, np.minimum(hi2, y2))
    hi_fix = np.maximum(hi2, np.maximum(lo2, y2))
    d.loc[fut_mask, "yhat_lower"] = lo_fix
    d.loc[fut_mask, "yhat_upper"] = hi_fix

    return d, {"bias": bias, "rmse": rmse, "mae": mae}


def confidence(score: float, rmse: float, members: int, note: str) -> tuple[float, str]:
    s = sf(score, np.inf)
    r = sf(rmse, np.inf)
    n = max(int(members), 1)
    if not np.isfinite(s):
        return 0.0, "D"
    c = 90.0 - 18.0 * s
    if np.isfinite(r):
        c -= min(18.0, 0.9 * r)
    if n >= 2:
        c += 4.0
    if "no_interval" in (note or ""):
        c -= 4.0
    c = float(np.clip(c, 0.0, 95.0))
    if c >= 85:
        g = "A"
    elif c >= 70:
        g = "B"
    elif c >= 55:
        g = "C"
    else:
        g = "D"
    return c, g


def render_dashboard(rows: pd.DataFrame, out_png: Path) -> None:
    variables = sorted(rows["variable"].dropna().astype(str).unique().tolist())
    n_vars = max(1, len(variables))
    ncols = 2 if n_vars > 1 else 1
    nrows = int(np.ceil(n_vars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8 * nrows))
    axes = axes.ravel()

    for ax, var in zip(axes, variables):
        rr = rows[rows["variable"] == var]
        if rr.empty:
            ax.set_axis_off()
            continue
        r = rr.iloc[0]
        d = load_fc(Path(str(r["forecast_csv"])))
        h = d[d["is_forecast"] == False]
        f = d[d["is_forecast"] == True]

        ax.plot(pd.to_datetime(h["ds"]), pd.to_numeric(h["yhat"], errors="coerce"), color="#4c78a8", lw=1.2, label="History")
        ax.plot(pd.to_datetime(f["ds"]), pd.to_numeric(f["yhat"], errors="coerce"), color="#f58518", lw=1.5, label="Forecast")

        lo = pd.to_numeric(f["yhat_lower"], errors="coerce")
        hi = pd.to_numeric(f["yhat_upper"], errors="coerce")
        if lo.notna().any() and hi.notna().any():
            ax.fill_between(pd.to_datetime(f["ds"]), lo, hi, color="#f58518", alpha=0.18, label="Interval")

        ax.set_title(
            f"{var} | 2035={sf(r['forecast_end_yhat_2035']):.3f} | score={sf(r['consensus_score']):.3f} | conf={sf(r['confidence_score']):.1f} ({r['confidence_grade']})"
        )
        ax.grid(alpha=0.25)
        ax.text(0.01, 0.03, str(r.get("members", "")), transform=ax.transAxes, fontsize=8, color="#333")

    for ax in axes[len(variables) :]:
        ax.set_axis_off()

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("v6 Stable Consensus Forecast", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    reports_dir = run_dir / "quant" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    fc_dir = run_dir / "quant" / "forecasts"
    fc_dir.mkdir(parents=True, exist_ok=True)

    v5_cands = reports_dir / "v5_candidate_scores.csv"
    if not v5_cands.exists():
        raise FileNotFoundError(f"Missing candidate table: {v5_cands}")

    cands = pd.read_csv(v5_cands)

    obs_path = args.obs_parquet
    if obs_path is None:
        obs_path = run_dir / "calibrated" / "observations_calibrated_recent_regime.parquet"
    obs = load_obs(obs_path)

    out_rows: list[dict[str, Any]] = []
    vars_order = sorted(cands["variable"].dropna().astype(str).unique().tolist())

    for var in vars_order:
        members = choose_members(cands, variable=var, top_k=args.top_k)
        if not members:
            continue

        w = member_weights(members)
        w = tune_member_weights(members=members, obs=obs, variable=var, default_weights=w)
        merged = merge_members(members, w)

        freq = infer_freq(merged["ds"])
        merged = stabilize_future(merged, variable=var, freq=freq)
        merged, fit = calibrate_bias_and_intervals(merged, obs=obs, variable=var, freq=freq)

        merged["variable"] = var
        merged["frequency"] = freq
        merged["model_strategy"] = "v6_stable_consensus"

        out_fc = fc_dir / f"{var}_{freq.lower()}_v6_stable_consensus_to_2035.csv"
        save_df = merged.copy()
        save_df["ds"] = pd.to_datetime(save_df["ds"], errors="coerce").dt.strftime("%Y-%m-%d")
        save_df.to_csv(out_fc, index=False)

        fut = merged[merged["is_forecast"] == True].sort_values("ds")
        y_end = sf(fut["yhat"].iloc[-1]) if not fut.empty else np.nan

        base_member_score = float(np.sum([w[m.model_name] * m.score_total for m in members]))
        obs_agg = agg_obs(obs, variable=var, freq=freq)
        spread = np.nan
        if not obs_agg.empty:
            spread = float(np.nanpercentile(obs_agg["actual"], 90) - np.nanpercentile(obs_agg["actual"], 10))
        spread = max(sf(spread, np.nan), 1e-6)
        rmse_norm = sf(fit["rmse"], np.nan) / spread if np.isfinite(sf(fit["rmse"], np.nan)) else np.nan

        hist_y = pd.to_numeric(merged[merged["is_forecast"] == False]["yhat"], errors="coerce").dropna().to_numpy(dtype=float)
        fut_y = pd.to_numeric(merged[merged["is_forecast"] == True]["yhat"], errors="coerce").dropna().to_numpy(dtype=float)
        hv = float(np.nanstd(np.diff(hist_y[-min(60, len(hist_y)):]))) if len(hist_y) >= 3 else np.nan
        fv = float(np.nanstd(np.diff(fut_y[:min(60, len(fut_y))]))) if len(fut_y) >= 3 else np.nan
        vol_pen = 0.0
        if np.isfinite(hv) and hv > 1e-9 and np.isfinite(fv):
            vol_pen = 0.12 * max((fv / hv) - 3.0, 0.0)
        jump_pen = 0.0
        if len(hist_y) and len(fut_y):
            jump = abs(float(fut_y[0]) - float(hist_y[-1])) / spread
            jump_pen = 0.08 * max(jump - 3.0, 0.0)

        if np.isfinite(rmse_norm):
            score = float(rmse_norm + vol_pen + jump_pen)
        else:
            score = base_member_score
        members_txt = ", ".join([f"{m.model_name}:{w[m.model_name]:.3f}" for m in members])
        note = ";".join(sorted(set([m.note for m in members if m.note])))
        conf, grade = confidence(score=score, rmse=fit["rmse"], members=len(members), note=note)

        out_rows.append(
            {
                "variable": var,
                "forecast_csv": str(out_fc),
                "frequency": freq,
                "members": members_txt,
                "member_count": len(members),
                "consensus_score": score,
                "base_member_score": base_member_score,
                "fit_rmse": fit["rmse"],
                "fit_mae": fit["mae"],
                "fit_bias": fit["bias"],
                "note": note,
                "forecast_end_yhat_2035": y_end,
                "confidence_score": conf,
                "confidence_grade": grade,
            }
        )

    out = pd.DataFrame(out_rows)
    out_csv = reports_dir / "v6_stable_consensus_ozet.csv"
    out.to_csv(out_csv, index=False)

    dash_png = reports_dir / "v6_stable_consensus_dashboard.png"
    if not out.empty:
        render_dashboard(out, dash_png)

    # Compare with v5 winners.
    v5_w = reports_dir / "v5_final_arbitrated_ozet.csv"
    cmp_csv = reports_dir / "v6_vs_v5_comparison.csv"
    cmp_md = reports_dir / "v6_vs_v5_comparison.md"
    if v5_w.exists() and not out.empty:
        v5 = pd.read_csv(v5_w)
        c = v5[["variable", "model_name", "forecast_end_yhat_2035", "score_total"]].rename(
            columns={
                "model_name": "v5_model",
                "forecast_end_yhat_2035": "v5_2035",
                "score_total": "v5_score",
            }
        )
        c = c.merge(
            out[["variable", "forecast_end_yhat_2035", "consensus_score", "confidence_score", "confidence_grade", "members"]].rename(
                columns={
                    "forecast_end_yhat_2035": "v6_2035",
                    "consensus_score": "v6_score",
                }
            ),
            on="variable",
            how="outer",
        )
        c["projection_delta_v6_minus_v5"] = c["v6_2035"] - c["v5_2035"]
        c["score_delta_v6_minus_v5"] = c["v6_score"] - c["v5_score"]
        c.to_csv(cmp_csv, index=False)

        lines = ["# v6 vs v5 Comparison", ""]
        for _, r in c.sort_values("variable").iterrows():
            lines.append(
                f"- {r['variable']}: v5={r['v5_model']} ({sf(r['v5_score']):.3f}) -> v6 consensus ({sf(r['v6_score']):.3f}), "
                f"2035 delta={sf(r['projection_delta_v6_minus_v5']):.3f}, conf={sf(r['confidence_score']):.1f} ({r['confidence_grade']})"
            )
        cmp_md.write_text("\n".join(lines), encoding="utf-8")

    md = reports_dir / "v6_stable_consensus_yorum.md"
    lines = [
        "# v6 Stable Consensus Summary",
        "",
        f"- Candidate source: `{v5_cands}`",
        f"- Observation source: `{obs_path}`",
        f"- Output summary: `{out_csv}`",
        f"- Dashboard: `{dash_png}`",
        "",
    ]
    if not out.empty:
        for _, r in out.sort_values("variable").iterrows():
            lines.append(
                f"- {r['variable']}: 2035={sf(r['forecast_end_yhat_2035']):.3f} | score={sf(r['consensus_score']):.3f} | "
                f"fit_rmse={sf(r['fit_rmse']):.3f} | conf={sf(r['confidence_score']):.1f} ({r['confidence_grade']})"
            )
    md.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "obs_parquet": str(obs_path),
        "variable_count": int(len(out)),
        "outputs": {
            "summary_csv": str(out_csv),
            "dashboard_png": str(dash_png),
            "comparison_csv": str(cmp_csv),
            "comparison_md": str(cmp_md),
            "comment_md": str(md),
        },
    }
    out_json = reports_dir / "v6_stable_consensus_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
