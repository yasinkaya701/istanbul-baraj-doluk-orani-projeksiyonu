#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TR_MAP = str.maketrans(
    {
        "ı": "i",
        "İ": "I",
        "ğ": "g",
        "Ğ": "G",
        "ş": "s",
        "Ş": "S",
        "ö": "o",
        "Ö": "O",
        "ç": "c",
        "Ç": "C",
        "ü": "u",
        "Ü": "U",
    }
)


def normalize_token(x: Any) -> str:
    s = str(x or "").translate(TR_MAP).strip().lower()
    for old, new in [("/", "_"), ("-", "_"), (" ", "_")]:
        s = s.replace(old, new)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def canonical_variable_name(x: Any) -> str:
    t = normalize_token(x)
    aliases = {
        "nem": "humidity",
        "humidity": "humidity",
        "rh": "humidity",
        "yagis": "precip",
        "yais": "precip",
        "rain": "precip",
        "precip": "precip",
        "sicaklik": "temp",
        "sicak": "temp",
        "temp": "temp",
        "temperature": "temp",
        "basinc": "pressure",
        "pressure": "pressure",
    }
    if t in aliases:
        return aliases[t]
    if "humid" in t or "nem" in t:
        return "humidity"
    if "precip" in t or "rain" in t or "yagis" in t:
        return "precip"
    if "temp" in t or "sicak" in t:
        return "temp"
    if "press" in t or "basinc" in t:
        return "pressure"
    return t or "target"


def read_table_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suf in {".xlsx", ".xls", ".ods"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported observations extension: {path.suffix}")


def normalize_observations_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "variable", "value"])

    cols = list(df.columns)
    ts_col = None
    for c in ("timestamp", "ds", "date", "datetime"):
        if c in cols:
            ts_col = c
            break
    if ts_col is None:
        return pd.DataFrame(columns=["timestamp", "variable", "value"])

    if "variable" in cols and "value" in cols:
        out = df[[ts_col, "variable", "value"]].copy()
        out = out.rename(columns={ts_col: "timestamp"})
    else:
        wide_cols = [c for c in cols if c != ts_col]
        if not wide_cols:
            return pd.DataFrame(columns=["timestamp", "variable", "value"])
        out = df[[ts_col] + wide_cols].copy().melt(
            id_vars=[ts_col],
            var_name="variable",
            value_name="value",
        )
        out = out.rename(columns={ts_col: "timestamp"})

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["variable"] = out["variable"].map(canonical_variable_name)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["timestamp", "variable", "value"]).copy()
    return out[["timestamp", "variable", "value"]].sort_values("timestamp").reset_index(drop=True)


def load_observations_df(path_text: str | None) -> tuple[pd.DataFrame, str]:
    if not path_text:
        return pd.DataFrame(columns=["timestamp", "variable", "value"]), ""
    p = Path(path_text).expanduser().resolve()
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "variable", "value"]), ""
    try:
        raw = read_table_any(p)
        obs = normalize_observations_df(raw)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "variable", "value"]), ""
    return obs, str(p)


def monthly_history_from_observations(obs_df: pd.DataFrame, variable: str) -> pd.DataFrame:
    if obs_df is None or obs_df.empty:
        return pd.DataFrame(columns=["ds", "actual"])
    var = canonical_variable_name(variable)
    d = obs_df[obs_df["variable"] == var].copy()
    if d.empty:
        return pd.DataFrame(columns=["ds", "actual"])
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if d.empty:
        return pd.DataFrame(columns=["ds", "actual"])
    d["ds"] = d["timestamp"].dt.to_period("M").dt.to_timestamp()
    if var == "precip":
        out = d.groupby("ds", as_index=False)["value"].sum()
    else:
        out = d.groupby("ds", as_index=False)["value"].mean()
    out = out.rename(columns={"value": "actual"}).dropna(subset=["ds", "actual"]).sort_values("ds")
    return out[["ds", "actual"]].reset_index(drop=True)


def parse_csv_list(text: str | None) -> list[str]:
    if text is None:
        return []
    out = [x.strip() for x in str(text).split(",") if x.strip()]
    return out


def continuity_metrics(ds: pd.Series | pd.DatetimeIndex, y: np.ndarray) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=float)
    td = pd.to_datetime(ds, errors="coerce")
    if isinstance(td, pd.Series):
        t_arr = td.values
    else:
        t_arr = np.asarray(td)
    ok = np.isfinite(y_arr) & pd.notna(t_arr)
    yy = y_arr[ok]
    tt = t_arr[ok]
    if yy.size <= 1:
        return {
            "jump_q95": 0.0,
            "jump_max": 0.0,
            "ann_jump_q95": 0.0,
            "ann_jump_max": 0.0,
            "amp_p90_p10": 0.0,
            "mean": float(np.nanmean(yy)) if yy.size else 0.0,
        }
    dd = np.abs(np.diff(yy))
    jump_q95 = float(np.quantile(dd, 0.95)) if dd.size else 0.0
    jump_max = float(np.max(dd)) if dd.size else 0.0
    ann_jump_q95 = 0.0
    ann_jump_max = 0.0
    try:
        ann = pd.DataFrame({"year": pd.DatetimeIndex(tt).year.astype(float), "y": yy}).dropna(subset=["year", "y"])
        ann = ann.groupby("year", as_index=False)["y"].mean().dropna()
        ad = np.abs(np.diff(ann["y"].values.astype(float)))
        ann_jump_q95 = float(np.quantile(ad, 0.95)) if ad.size else 0.0
        ann_jump_max = float(np.max(ad)) if ad.size else 0.0
    except Exception:
        ann_jump_q95 = 0.0
        ann_jump_max = 0.0
    amp = float(np.quantile(yy, 0.90) - np.quantile(yy, 0.10)) if yy.size else 0.0
    return {
        "jump_q95": float(jump_q95),
        "jump_max": float(jump_max),
        "ann_jump_q95": float(ann_jump_q95),
        "ann_jump_max": float(ann_jump_max),
        "amp_p90_p10": float(max(0.0, amp)),
        "mean": float(np.nanmean(yy)) if yy.size else 0.0,
    }


def median3_smooth(y: np.ndarray, alpha: float, passes: int) -> np.ndarray:
    yy = np.asarray(y, dtype=float).copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    if yy.size < 3 or a <= 1e-9:
        return yy
    pcount = int(max(1, passes))
    for _ in range(pcount):
        prev = yy.copy()
        for i in range(1, len(yy) - 1):
            c0, c1, c2 = prev[i - 1], prev[i], prev[i + 1]
            if not (np.isfinite(c0) and np.isfinite(c1) and np.isfinite(c2)):
                continue
            med = float(np.median([c0, c1, c2]))
            yy[i] = (1.0 - a) * c1 + a * med
    return yy


def plot_forecast(df: pd.DataFrame, variable: str, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
    d = d.sort_values("ds")
    h = d[d["is_forecast"] == False].copy()
    f = d[d["is_forecast"] == True].copy()
    plt.figure(figsize=(11, 5))
    if not h.empty:
        plt.plot(h["ds"], h["yhat"], label="history", color="#4c72b0", linewidth=1.5)
    if not f.empty:
        plt.plot(f["ds"], f["yhat"], label="forecast", color="#dd8452", linewidth=2.0)
        if {"yhat_lower", "yhat_upper"}.issubset(f.columns):
            lo = pd.to_numeric(f["yhat_lower"], errors="coerce")
            hi = pd.to_numeric(f["yhat_upper"], errors="coerce")
            ok = lo.notna() & hi.notna()
            if bool(ok.any()):
                plt.fill_between(f.loc[ok, "ds"], lo.loc[ok], hi.loc[ok], color="#dd8452", alpha=0.18, linewidth=0)
    plt.title(f"{variable} monthly forecast")
    plt.xlabel("date")
    plt.ylabel(variable)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=165)
    plt.close()


def profile_weights(name: str) -> tuple[dict[str, float], dict[str, float]]:
    ann_s = {"temp": 1.4, "humidity": 0.35, "precip": 0.65}
    ann_c = {"temp": 1.8, "humidity": 1.2, "precip": 0.9}
    if name == "smooth":
        ann = ann_s
    elif name == "consistency":
        ann = ann_c
    else:
        ann = {k: 0.5 * (ann_s[k] + ann_c[k]) for k in ann_s}
    max_w = {"temp": 0.03, "humidity": 0.02, "precip": 0.01}
    return ann, max_w


def objective_of(metrics: dict[str, float], variable: str, ann_w: dict[str, float], max_w: dict[str, float]) -> float:
    return (
        float(metrics.get("jump_q95", 0.0))
        + float(ann_w.get(variable, 0.8)) * float(metrics.get("ann_jump_q95", 0.0))
        + float(max_w.get(variable, 0.02)) * float(metrics.get("jump_max", 0.0))
    )


def build_accuracy_context(history_df: pd.DataFrame, variable: str) -> dict[str, Any]:
    if history_df is None or history_df.empty:
        return {"enabled": False, "reason": "empty_history"}
    h = history_df.copy()
    h["ds"] = pd.to_datetime(h["ds"], errors="coerce")
    h["actual"] = pd.to_numeric(h["actual"], errors="coerce")
    h = h.dropna(subset=["ds", "actual"]).sort_values("ds")
    if len(h) < 24:
        return {"enabled": False, "reason": "insufficient_history"}

    max_recent = 120
    h_recent = h.iloc[max(0, len(h) - max_recent) :].copy()
    y = h_recent["actual"].values.astype(float)
    m = h_recent["ds"].dt.month.values.astype(int)

    month_mean = h_recent.groupby(h_recent["ds"].dt.month)["actual"].mean().to_dict()
    level = float(np.nanmean(y))
    spread = float(max(np.nanstd(y), 1e-6))
    amp = float(np.quantile(y, 0.90) - np.quantile(y, 0.10)) if len(y) >= 10 else float(np.nanmax(y) - np.nanmin(y))
    scale = float(max(spread, 0.10 * max(amp, 1e-6), 1e-6))

    ann = h_recent.groupby(h_recent["ds"].dt.year)["actual"].mean().reset_index(drop=False)
    if len(ann) >= 3:
        xx = ann["ds"].values.astype(float)
        yy = ann["actual"].values.astype(float)
        x0 = float(xx[0])
        den = float(np.dot(xx - x0, xx - x0)) + 1e-12
        slope = float(np.dot(xx - x0, yy - yy.mean()) / den)
    else:
        slope = 0.0
    slope_scale = float(max(abs(slope), scale / 8.0, 1e-4))

    return {
        "enabled": True,
        "reason": "ok",
        "variable": str(variable),
        "level_recent": float(level),
        "scale": float(scale),
        "month_mean_recent": {int(k): float(v) for k, v in month_mean.items()},
        "trend_slope_recent": float(slope),
        "trend_slope_scale": float(slope_scale),
        "recent_points": int(len(h_recent)),
    }


def accuracy_penalty_from_context(
    ds_forecast: pd.Series | pd.DatetimeIndex,
    y_forecast: np.ndarray,
    ctx: dict[str, Any],
    variable: str,
) -> tuple[float, dict[str, float]]:
    if not ctx or not bool(ctx.get("enabled", False)):
        return 0.0, {"enabled": 0.0}

    td = pd.to_datetime(ds_forecast, errors="coerce")
    if isinstance(td, pd.Series):
        dsv = td.values
    else:
        dsv = np.asarray(td)
    yy = np.asarray(y_forecast, dtype=float)
    ok = np.isfinite(yy) & pd.notna(dsv)
    if int(ok.sum()) < 12:
        return 0.0, {"enabled": 0.0}
    dsv = dsv[ok]
    yy = yy[ok]

    f = pd.DataFrame({"ds": pd.to_datetime(dsv), "y": yy}).sort_values("ds")
    f_early = f.iloc[: min(len(f), 60)].copy()
    if f_early.empty:
        return 0.0, {"enabled": 0.0}

    level_fc = float(np.nanmean(f_early["y"].values.astype(float)))
    level_recent = float(ctx.get("level_recent", level_fc))
    scale = float(max(ctx.get("scale", 1.0), 1e-6))
    level_pen = abs(level_fc - level_recent) / scale

    mon_recent = dict(ctx.get("month_mean_recent", {}))
    mon_fc = f_early.groupby(f_early["ds"].dt.month)["y"].mean().to_dict()
    month_errs: list[float] = []
    for m in range(1, 13):
        if m in mon_recent and m in mon_fc:
            month_errs.append((float(mon_fc[m]) - float(mon_recent[m])) / scale)
    month_pen = float(np.mean(np.square(month_errs))) if month_errs else 0.0

    ann = f_early.groupby(f_early["ds"].dt.year)["y"].mean().reset_index(drop=False)
    if len(ann) >= 3:
        xx = ann["ds"].values.astype(float)
        yy_ann = ann["y"].values.astype(float)
        x0 = float(xx[0])
        den = float(np.dot(xx - x0, xx - x0)) + 1e-12
        slope_fc = float(np.dot(xx - x0, yy_ann - yy_ann.mean()) / den)
    else:
        slope_fc = 0.0
    slope_recent = float(ctx.get("trend_slope_recent", 0.0))
    slope_scale = float(max(ctx.get("trend_slope_scale", 1.0), 1e-6))
    trend_pen = abs(slope_fc - slope_recent) / slope_scale

    var = str(variable)
    if var == "temp":
        w_month, w_trend, w_level = 0.45, 0.40, 0.15
    elif var == "humidity":
        w_month, w_trend, w_level = 0.50, 0.20, 0.30
    elif var == "precip":
        w_month, w_trend, w_level = 0.35, 0.15, 0.50
    else:
        w_month, w_trend, w_level = 0.40, 0.30, 0.30

    total = float(w_month * month_pen + w_trend * trend_pen + w_level * level_pen)
    detail = {
        "enabled": 1.0,
        "month_pen": float(month_pen),
        "trend_pen": float(trend_pen),
        "level_pen": float(level_pen),
        "total_pen": float(total),
    }
    return total, detail


def alpha_candidates(variable: str, temp_alpha_max: float = 0.50) -> list[float]:
    if variable == "temp":
        base = [0.04, 0.08, 0.12, 0.16, 0.22, 0.28, 0.32, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        tmax = float(np.clip(temp_alpha_max, 0.04, 0.95))
        out = [a for a in base if a <= tmax + 1e-12]
        return out or [0.04]
    if variable == "humidity":
        return [0.0025, 0.005, 0.01, 0.015, 0.02, 0.04, 0.06, 0.08, 0.12]
    if variable == "precip":
        return [0.0025, 0.005, 0.01, 0.015, 0.03, 0.06, 0.09, 0.12, 0.14, 0.18]
    return [0.04, 0.08, 0.12]


@dataclass
class BlendResult:
    variable: str
    chosen_runs: list[str]
    weights: list[float]
    metrics_before_smoothing: dict[str, float]
    metrics_after_smoothing: dict[str, float]
    objective_before_smoothing: float
    objective_after_smoothing: float
    accuracy_penalty_before: float
    accuracy_penalty_after: float
    accuracy_penalty_detail_before: dict[str, float]
    accuracy_penalty_detail_after: dict[str, float]
    smoothing_alpha: float
    smoothing_passes: int
    frame: pd.DataFrame


def optimize_var_blend(
    variable: str,
    forecast_frames: dict[str, pd.DataFrame],
    ann_w: dict[str, float],
    max_w: dict[str, float],
    weight_step: float,
    apply_smoothing: bool,
    temp_alpha_max: float,
    smooth_pass_min: int,
    smooth_pass_max: int,
    enable_triple_blend: bool,
    triple_step: float,
    accuracy_context: dict[str, Any] | None,
    accuracy_penalty_weight: float,
) -> BlendResult:
    runs = sorted(forecast_frames.keys())
    if len(runs) == 0:
        raise ValueError(f"No forecast frames for variable={variable}")
    if len(runs) == 1:
        only = runs[0]
        f = forecast_frames[only].copy()
        y = pd.to_numeric(f["yhat"], errors="coerce").values.astype(float)
        m = continuity_metrics(pd.to_datetime(f["ds"]), y)
        ap, apd = accuracy_penalty_from_context(
            ds_forecast=pd.to_datetime(f["ds"]),
            y_forecast=y,
            ctx=(accuracy_context or {}),
            variable=variable,
        )
        base_obj = objective_of(m, variable, ann_w, max_w)
        obj_total = float(base_obj + float(max(0.0, accuracy_penalty_weight)) * float(ap))
        return BlendResult(
            variable=variable,
            chosen_runs=[only],
            weights=[1.0],
            metrics_before_smoothing=m,
            metrics_after_smoothing=m,
            objective_before_smoothing=obj_total,
            objective_after_smoothing=obj_total,
            accuracy_penalty_before=float(ap),
            accuracy_penalty_after=float(ap),
            accuracy_penalty_detail_before=dict(apd),
            accuracy_penalty_detail_after=dict(apd),
            smoothing_alpha=0.0,
            smoothing_passes=0,
            frame=f,
        )

    ds_ref = pd.to_datetime(forecast_frames[runs[0]]["ds"]).reset_index(drop=True)
    for r in runs[1:]:
        ds_cur = pd.to_datetime(forecast_frames[r]["ds"]).reset_index(drop=True)
        if len(ds_ref) != len(ds_cur) or not bool((ds_ref.values == ds_cur.values).all()):
            raise ValueError(f"Forecast dates mismatch for variable={variable}: {runs[0]} vs {r}")

    cols = ["yhat", "yhat_lower", "yhat_upper"]
    mats: dict[str, dict[str, np.ndarray]] = {}
    for r in runs:
        ff = forecast_frames[r]
        mats[r] = {c: pd.to_numeric(ff[c], errors="coerce").values.astype(float) for c in cols}

    step = float(np.clip(weight_step, 0.01, 0.5))
    grid = np.arange(0.0, 1.0 + 0.5 * step, step)
    tri_step = float(np.clip(triple_step, 0.02, 0.5))
    tri_grid = np.arange(0.0, 1.0 + 0.5 * tri_step, tri_step)

    def build_by_weights(r1: str, r2: str, w1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w1c = float(np.clip(w1, 0.0, 1.0))
        w2 = 1.0 - w1c
        y = w1c * mats[r1]["yhat"] + w2 * mats[r2]["yhat"]
        lo = w1c * mats[r1]["yhat_lower"] + w2 * mats[r2]["yhat_lower"]
        hi = w1c * mats[r1]["yhat_upper"] + w2 * mats[r2]["yhat_upper"]
        lo = np.minimum(lo, y)
        hi = np.maximum(hi, y)
        return y, lo, hi

    best: dict[str, Any] | None = None

    pen_w = float(max(0.0, accuracy_penalty_weight))

    for i, r1 in enumerate(runs):
        for r2 in runs[i + 1 :]:
            for w1 in grid:
                y, lo, hi = build_by_weights(r1, r2, float(w1))
                m = continuity_metrics(ds_ref, y)
                base_obj = objective_of(m, variable, ann_w, max_w)
                ap, apd = accuracy_penalty_from_context(ds_ref, y, (accuracy_context or {}), variable=variable)
                obj = float(base_obj + pen_w * ap)
                rec = {
                    "runs": [r1, r2],
                    "weights": [float(w1), float(1.0 - w1)],
                    "y": y,
                    "lo": lo,
                    "hi": hi,
                    "metrics": m,
                    "base_obj": float(base_obj),
                    "accuracy_penalty": float(ap),
                    "accuracy_penalty_detail": dict(apd),
                    "obj": float(obj),
                }
                if best is None or float(obj) < float(best["obj"]) - 1e-12:
                    best = rec

    if enable_triple_blend and len(runs) >= 3:
        for i, r1 in enumerate(runs):
            for j, r2 in enumerate(runs[i + 1 :], start=i + 1):
                for r3 in runs[j + 1 :]:
                    for w1 in tri_grid:
                        w1c = float(np.clip(w1, 0.0, 1.0))
                        rem = 1.0 - w1c
                        if rem < -1e-12:
                            continue
                        for w2 in np.arange(0.0, rem + 0.5 * tri_step, tri_step):
                            w2c = float(np.clip(w2, 0.0, rem))
                            w3c = float(max(0.0, 1.0 - w1c - w2c))
                            y = w1c * mats[r1]["yhat"] + w2c * mats[r2]["yhat"] + w3c * mats[r3]["yhat"]
                            lo = w1c * mats[r1]["yhat_lower"] + w2c * mats[r2]["yhat_lower"] + w3c * mats[r3]["yhat_lower"]
                            hi = w1c * mats[r1]["yhat_upper"] + w2c * mats[r2]["yhat_upper"] + w3c * mats[r3]["yhat_upper"]
                            lo = np.minimum(lo, y)
                            hi = np.maximum(hi, y)
                            m = continuity_metrics(ds_ref, y)
                            base_obj = objective_of(m, variable, ann_w, max_w)
                            ap, apd = accuracy_penalty_from_context(ds_ref, y, (accuracy_context or {}), variable=variable)
                            obj = float(base_obj + pen_w * ap)
                            rec = {
                                "runs": [r1, r2, r3],
                                "weights": [float(w1c), float(w2c), float(w3c)],
                                "y": y,
                                "lo": lo,
                                "hi": hi,
                                "metrics": m,
                                "base_obj": float(base_obj),
                                "accuracy_penalty": float(ap),
                                "accuracy_penalty_detail": dict(apd),
                                "obj": float(obj),
                            }
                            if best is None or float(obj) < float(best["obj"]) - 1e-12:
                                best = rec

    assert best is not None
    y0 = best["y"].copy()
    lo0 = best["lo"].copy()
    hi0 = best["hi"].copy()
    m0 = dict(best["metrics"])
    obj0 = float(best["obj"])
    acc0 = float(best.get("accuracy_penalty", 0.0))
    acc0d = dict(best.get("accuracy_penalty_detail", {}))

    if apply_smoothing:
        base_amp = float(max(m0.get("amp_p90_p10", 0.0), 1e-6))
        base_mean = float(m0.get("mean", 0.0))
        best_obj = float(obj0)
        best_y = y0.copy()
        best_alpha = 0.0
        best_passes = 0
        best_m = dict(m0)
        best_acc = float(acc0)
        best_accd = dict(acc0d)
        pmin = int(max(1, smooth_pass_min))
        pmax = int(max(pmin, smooth_pass_max))
        for a in alpha_candidates(variable, temp_alpha_max=temp_alpha_max):
            for pcount in range(pmin, pmax + 1):
                ys = median3_smooth(y0, alpha=float(a), passes=int(pcount))
                mm = continuity_metrics(ds_ref, ys)
                ok = (
                    float(mm.get("jump_q95", float("inf"))) <= float(m0.get("jump_q95", 0.0)) * 1.02 + 1e-9
                    and float(mm.get("jump_max", float("inf"))) <= float(m0.get("jump_max", 0.0)) * 1.05 + 1e-9
                    and float(mm.get("amp_p90_p10", 0.0)) >= 0.75 * base_amp
                    and abs(float(mm.get("mean", 0.0)) - base_mean)
                    <= max(0.02 * abs(base_mean), 0.6 if variable == "temp" else (1.2 if variable == "humidity" else 2.0))
                )
                if not ok:
                    continue
                oo_base = objective_of(mm, variable, ann_w, max_w)
                ap, apd = accuracy_penalty_from_context(ds_ref, ys, (accuracy_context or {}), variable=variable)
                oo = float(oo_base + pen_w * ap)
                if oo < best_obj - 1e-9:
                    best_obj = float(oo)
                    best_y = ys
                    best_alpha = float(a)
                    best_passes = int(pcount)
                    best_m = dict(mm)
                    best_acc = float(ap)
                    best_accd = dict(apd)
        if best_alpha > 0.0:
            shift = best_y - y0
            y1 = best_y
            lo1 = lo0 + shift
            hi1 = hi0 + shift
            lo1 = np.minimum(lo1, y1)
            hi1 = np.maximum(hi1, y1)
            y0, lo0, hi0 = y1, lo1, hi1
            m1 = best_m
            obj1 = best_obj
            sm_alpha = best_alpha
            sm_passes = best_passes
            acc1 = best_acc
            acc1d = best_accd
        else:
            m1 = m0
            obj1 = obj0
            sm_alpha = 0.0
            sm_passes = 0
            acc1 = acc0
            acc1d = acc0d
    else:
        m1 = m0
        obj1 = obj0
        sm_alpha = 0.0
        sm_passes = 0
        acc1 = acc0
        acc1d = acc0d

    out = forecast_frames[best["runs"][0]].copy()
    out["yhat"] = y0
    out["yhat_lower"] = lo0
    out["yhat_upper"] = hi0

    return BlendResult(
        variable=variable,
        chosen_runs=list(best["runs"]),
        weights=list(best["weights"]),
        metrics_before_smoothing=m0,
        metrics_after_smoothing=m1,
        objective_before_smoothing=float(obj0),
        objective_after_smoothing=float(obj1),
        accuracy_penalty_before=float(acc0),
        accuracy_penalty_after=float(acc1),
        accuracy_penalty_detail_before=dict(acc0d),
        accuracy_penalty_detail_after=dict(acc1d),
        smoothing_alpha=float(sm_alpha),
        smoothing_passes=int(sm_passes),
        frame=out,
    )


def run(args: argparse.Namespace) -> None:
    run_dirs = [Path(x) for x in parse_csv_list(args.run_dirs)]
    if not run_dirs:
        raise ValueError("--run-dirs is required")
    labels = parse_csv_list(args.run_labels)
    if labels and len(labels) != len(run_dirs):
        raise ValueError("--run-labels length must match --run-dirs")
    if not labels:
        labels = [p.name for p in run_dirs]

    label_to_dir = {labels[i]: run_dirs[i] for i in range(len(run_dirs))}

    base_label = str(args.base_label) if args.base_label else labels[0]
    if base_label not in label_to_dir:
        raise ValueError(f"base label not found: {base_label}")
    base_dir = label_to_dir[base_label]

    variables = parse_csv_list(args.variables) or ["temp", "humidity", "precip"]
    ann_w, max_w = profile_weights(str(args.objective_profile))
    obs_df, obs_path = load_observations_df(args.observations)

    out_dir = Path(args.output_dir).resolve()
    fc_dir = out_dir / "forecasts"
    ch_dir = out_dir / "charts"
    rp_dir = out_dir / "reports"
    for d in [out_dir, fc_dir, ch_dir, rp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    blend_results: dict[str, BlendResult] = {}
    index_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for var in variables:
        base_fp = base_dir / "forecasts" / f"{var}_monthly_best_meta_to_2035.csv"
        if not base_fp.exists():
            continue
        base_df = pd.read_csv(base_fp)
        base_df["ds"] = pd.to_datetime(base_df["ds"], errors="coerce")
        base_fc = base_df[base_df["is_forecast"] == True].copy().sort_values("ds").reset_index(drop=True)
        if base_fc.empty:
            continue

        hist_ctx_source = "base_history"
        hist_ctx = build_accuracy_context(
            history_df=base_df[base_df["is_forecast"] == False][["ds", "actual"]].copy(),
            variable=var,
        )
        if not obs_df.empty:
            obs_hist = monthly_history_from_observations(obs_df=obs_df, variable=var)
            obs_ctx = build_accuracy_context(history_df=obs_hist, variable=var)
            if bool(obs_ctx.get("enabled", False)):
                hist_ctx = obs_ctx
                hist_ctx_source = "observations"

        pool: dict[str, pd.DataFrame] = {}
        for label, rd in label_to_dir.items():
            fp = rd / "forecasts" / f"{var}_monthly_best_meta_to_2035.csv"
            if not fp.exists():
                continue
            dd = pd.read_csv(fp)
            dd["ds"] = pd.to_datetime(dd["ds"], errors="coerce")
            ff = dd[dd["is_forecast"] == True].copy().sort_values("ds").reset_index(drop=True)
            if len(ff) != len(base_fc):
                continue
            if not bool((ff["ds"].values == base_fc["ds"].values).all()):
                continue
            pool[label] = ff[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

        if not pool:
            continue

        br = optimize_var_blend(
            variable=var,
            forecast_frames=pool,
            ann_w=ann_w,
            max_w=max_w,
            weight_step=float(args.weight_step),
            apply_smoothing=bool(args.apply_smoothing),
            temp_alpha_max=float(args.temp_alpha_max),
            smooth_pass_min=int(args.smooth_pass_min),
            smooth_pass_max=int(args.smooth_pass_max),
            enable_triple_blend=bool(args.enable_triple_blend),
            triple_step=float(args.triple_step),
            accuracy_context=hist_ctx,
            accuracy_penalty_weight=float(args.accuracy_penalty_weight),
        )
        blend_results[var] = br

        out_df = base_df.copy().sort_values("ds").reset_index(drop=True)
        mask = out_df["is_forecast"] == True
        out_df.loc[mask, "yhat"] = br.frame["yhat"].values
        out_df.loc[mask, "yhat_lower"] = br.frame["yhat_lower"].values
        out_df.loc[mask, "yhat_upper"] = br.frame["yhat_upper"].values
        out_df["model_strategy"] = "postprocess_meta_blend_optimizer"

        fc_csv = fc_dir / f"{var}_monthly_best_meta_to_2035.csv"
        out_df.to_csv(fc_csv, index=False)
        chart_png = ch_dir / f"{var}_monthly_best_meta_to_2035.png"
        plot_forecast(out_df, variable=var, out_png=chart_png)

        rep = {
            "variable": var,
            "chosen_runs": br.chosen_runs,
            "weights": br.weights,
            "objective_profile": str(args.objective_profile),
            "accuracy_penalty_weight": float(args.accuracy_penalty_weight),
            "accuracy_context_source": hist_ctx_source,
            "observations_path": obs_path,
            "objective_before_smoothing": float(br.objective_before_smoothing),
            "objective_after_smoothing": float(br.objective_after_smoothing),
            "accuracy_penalty_before": float(br.accuracy_penalty_before),
            "accuracy_penalty_after": float(br.accuracy_penalty_after),
            "accuracy_penalty_detail_before": br.accuracy_penalty_detail_before,
            "accuracy_penalty_detail_after": br.accuracy_penalty_detail_after,
            "accuracy_context": hist_ctx,
            "smoothing_alpha": float(br.smoothing_alpha),
            "smoothing_passes": int(br.smoothing_passes),
            "metrics_before_smoothing": br.metrics_before_smoothing,
            "metrics_after_smoothing": br.metrics_after_smoothing,
            "forecast_csv": str(fc_csv),
            "chart_png": str(chart_png),
        }
        rep_json = rp_dir / f"{var}_meta_blend_report.json"
        rep_json.write_text(json.dumps(rep, indent=2), encoding="utf-8")

        index_rows.append(
            {
                "variable": var,
                "frequency": "MS",
                "target_year": int(args.target_year),
                "history_points": int((out_df["is_forecast"] == False).sum()),
                "forecast_steps": int((out_df["is_forecast"] == True).sum()),
                "model_strategy": "postprocess_meta_blend_optimizer",
                "forecast_csv": str(fc_csv),
                "chart_png": str(chart_png),
                "report_json": str(rep_json),
                "blend_runs": "|".join(br.chosen_runs),
                "blend_weights": "|".join(f"{w:.6f}" for w in br.weights),
                "smoothing_alpha": float(br.smoothing_alpha),
                "accuracy_context_source": hist_ctx_source,
                "accuracy_penalty_after": float(br.accuracy_penalty_after),
            }
        )
        metrics_rows.append(
            {
                "variable": var,
                "jump_q95": float(br.metrics_after_smoothing["jump_q95"]),
                "jump_max": float(br.metrics_after_smoothing["jump_max"]),
                "ann_jump_q95": float(br.metrics_after_smoothing["ann_jump_q95"]),
            }
        )

    if not index_rows:
        raise RuntimeError("No variable blended. Check run dirs and forecast files.")

    idx_df = pd.DataFrame(index_rows).sort_values("variable")
    idx_df.to_csv(out_dir / "best_meta_index_to_2035.csv", index=False)

    met_df = pd.DataFrame(metrics_rows).sort_values("variable")
    met_df.to_csv(out_dir / "metrics_detail.csv", index=False)

    ann_s = {"temp": 1.4, "humidity": 0.35, "precip": 0.65}
    ann_c = {"temp": 1.8, "humidity": 1.2, "precip": 0.9}
    for profile_name, ann_map in [("smooth", ann_s), ("consistency", ann_c), ("selected", ann_w)]:
        total = 0.0
        for _, r in met_df.iterrows():
            var = str(r["variable"])
            total += (
                float(r["jump_q95"])
                + float(ann_map.get(var, 0.8)) * float(r["ann_jump_q95"])
                + float(max_w.get(var, 0.02)) * float(r["jump_max"])
            )
        score_rows.append({"profile": profile_name, "score": float(total)})
    pd.DataFrame(score_rows).to_csv(out_dir / "metrics_score.csv", index=False)

    print("Meta blend optimization completed.")
    print(f"Output: {out_dir}")
    print(f"Index: {out_dir / 'best_meta_index_to_2035.csv'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Post-process and optimize meta forecast blend across run outputs.")
    p.add_argument("--run-dirs", type=str, required=True, help="Comma-separated run output directories.")
    p.add_argument("--run-labels", type=str, default="", help="Comma-separated labels aligned with --run-dirs.")
    p.add_argument("--base-label", type=str, default="", help="Label to use as base history/structure source.")
    p.add_argument("--observations", type=str, default="", help="Optional observations table (csv/parquet/xlsx/ods).")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--variables", type=str, default="temp,humidity,precip")
    p.add_argument("--objective-profile", type=str, default="balanced", choices=["smooth", "consistency", "balanced"])
    p.add_argument("--weight-step", type=float, default=0.05)
    p.add_argument("--apply-smoothing", action="store_true", default=False)
    p.add_argument("--temp-alpha-max", type=float, default=0.50)
    p.add_argument("--smooth-pass-min", type=int, default=2)
    p.add_argument("--smooth-pass-max", type=int, default=6)
    p.add_argument("--enable-triple-blend", action="store_true", default=False)
    p.add_argument("--triple-step", type=float, default=0.10)
    p.add_argument("--accuracy-penalty-weight", type=float, default=0.20)
    p.add_argument("--target-year", type=int, default=2035)
    return p


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
