#!/usr/bin/env python3
"""Alternative idea: Analog Pattern + Trend + Conformal forecasting.

Why this model:
- Prophet can overfit/underfit when history is sparse or regime-shifted.
- This pipeline uses pattern matching from historical trajectories (analog method),
  combines it with a smooth trend/seasonality regressor, and calibrates uncertainty
  by conformal residuals from rolling backtests.

Outputs:
- output/analog_pattern_package/forecasts/*.csv|parquet
- output/analog_pattern_package/charts/*.png
- output/analog_pattern_package/reports/*.json
- output/analog_pattern_package/analog_index_to_<year>.csv|parquet
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "analog_pattern_cache"
_MPL_CACHE = _CACHE_ROOT / "mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

UNIT_MAP = {
    "humidity": "%",
    "temp": "C",
    "pressure": "hPa",
    "precip": "mm",
}

_TR_CHARMAP = str.maketrans(
    {
        "ı": "i",
        "İ": "i",
        "ş": "s",
        "Ş": "s",
        "ğ": "g",
        "Ğ": "g",
        "ü": "u",
        "Ü": "u",
        "ö": "o",
        "Ö": "o",
        "ç": "c",
        "Ç": "c",
    }
)

ALIASES = {
    "nem": "humidity",
    "humidity": "humidity",
    "relative_humidity": "humidity",
    "rh": "humidity",
    "sicaklik": "temp",
    "sıcaklık": "temp",
    "temperature": "temp",
    "temp": "temp",
    "basinc": "pressure",
    "basınç": "pressure",
    "pressure": "pressure",
    "pres": "pressure",
    "yagis": "precip",
    "yağış": "precip",
    "precip": "precip",
    "precipitation": "precip",
    "rain": "precip",
    "rainfall": "precip",
    "prcp": "precip",
}


@dataclass
class FrequencyPlan:
    freq: str
    season_len: int
    label: str
    monthly_coverage: float


@dataclass
class CVPlan:
    holdout: int
    splits: int
    min_train: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analog pattern climate forecasting pipeline")
    p.add_argument(
        "--observations",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/forecast_package/observations_with_graph.parquet"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/analog_pattern_package"),
    )

    p.add_argument("--input-kind", type=str, default="auto", choices=["auto", "long", "single"])
    p.add_argument("--timestamp-col", type=str, default="timestamp")
    p.add_argument("--value-col", type=str, default="value")
    p.add_argument("--variable-col", type=str, default="variable")
    p.add_argument("--qc-col", type=str, default="qc_flag")
    p.add_argument("--qc-ok-value", type=str, default="ok")
    p.add_argument("--single-variable", type=str, default="target")

    p.add_argument("--variables", type=str, default="*")
    p.add_argument("--target-year", type=int, default=2035)

    p.add_argument("--winsor-lower", type=float, default=0.005)
    p.add_argument("--winsor-upper", type=float, default=0.995)

    p.add_argument("--holdout-steps", type=int, default=12)
    p.add_argument("--backtest-splits", type=int, default=3)
    p.add_argument("--min-train-steps", type=int, default=36)

    p.add_argument("--analog-lag", type=int, default=6)
    p.add_argument("--analog-k", type=int, default=12)
    p.add_argument("--interval-alpha", type=float, default=0.10)

    return p.parse_args()


def normalize_token(text: Any) -> str:
    s = str(text).strip().lower().translate(_TR_CHARMAP)
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def canonical_variable_name(text: Any) -> str:
    t = normalize_token(text)
    if t in ALIASES:
        return ALIASES[t]

    if any(k in t for k in ["humid", "nem", "rh"]):
        return "humidity"
    if any(k in t for k in ["temp", "sicak", "sicaklik", "temperature", "t2m"]):
        return "temp"
    if any(k in t for k in ["press", "basinc", "hpa", "mbar"]):
        return "pressure"
    if any(k in t for k in ["precip", "rain", "yagis", "prcp"]):
        return "precip"
    return t if t else "target"


def infer_unit(variable: str) -> str:
    return UNIT_MAP.get(canonical_variable_name(variable), "unknown")


def is_precip(variable: str) -> bool:
    return canonical_variable_name(variable) == "precip"


def is_humidity(variable: str) -> bool:
    return canonical_variable_name(variable) == "humidity"


def is_pressure(variable: str) -> bool:
    return canonical_variable_name(variable) == "pressure"


def apply_bounds(arr: np.ndarray, variable: str) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if is_humidity(variable):
        return np.clip(x, 0, 100)
    if is_precip(variable) or is_pressure(variable):
        return np.clip(x, 0, None)
    return x


def ensure_dirs(paths: list[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls", ".ods"}:
        return pd.read_excel(path)
    raise SystemExit(f"Unsupported input extension: {path.suffix}")


def pick_existing_column(raw: pd.DataFrame, preferred: str, fallbacks: list[str]) -> str | None:
    if preferred in raw.columns:
        return preferred
    for c in fallbacks:
        if c in raw.columns:
            return c
    return None


def normalize_observations(raw: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    ts_col = pick_existing_column(raw, args.timestamp_col, ["timestamp", "ds", "date", "datetime", "time", "tarih"])
    val_col = pick_existing_column(raw, args.value_col, ["value", "y", "target", "measurement"])
    var_col = pick_existing_column(raw, args.variable_col, ["variable", "metric", "param", "sensor", "name"])
    qc_col = pick_existing_column(raw, args.qc_col, ["qc_flag", "qc", "quality", "flag"])

    if ts_col is None or val_col is None:
        raise SystemExit("Cannot detect time/value columns")

    input_kind = args.input_kind
    if input_kind == "auto":
        input_kind = "long" if var_col is not None else "single"

    if input_kind == "long":
        if var_col is None:
            raise SystemExit("input-kind=long requires variable column")
        out = pd.DataFrame({"timestamp": raw[ts_col], "variable": raw[var_col], "value": raw[val_col]})
    else:
        out = pd.DataFrame({"timestamp": raw[ts_col], "variable": args.single_variable, "value": raw[val_col]})

    if qc_col is not None:
        out["qc_flag"] = raw[qc_col].astype(str)
    else:
        out["qc_flag"] = args.qc_ok_value

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["variable"] = out["variable"].astype(str).map(canonical_variable_name)
    out = out.dropna(subset=["timestamp", "value", "variable"]).sort_values("timestamp").reset_index(drop=True)

    if out.empty:
        raise SystemExit("No usable rows after parsing")

    return out, input_kind


def select_variables(obs: pd.DataFrame, variables_arg: str) -> list[str]:
    available = sorted(obs["variable"].dropna().unique().tolist())
    if not variables_arg or variables_arg.strip() in {"*", "all", "ALL"}:
        return available
    req = [canonical_variable_name(v.strip()) for v in variables_arg.split(",") if v.strip()]
    req = [v for v in req if v in available]
    return sorted(set(req))


def choose_frequency_plan(obs: pd.DataFrame, variable: str, ok_value: str) -> FrequencyPlan:
    sub = obs[obs["variable"] == variable].copy()
    if sub.empty:
        return FrequencyPlan("MS", 12, "monthly", 0.0)

    ok_mask = sub["qc_flag"].astype(str).str.lower().eq(str(ok_value).lower())
    if ok_mask.any():
        sub = sub[ok_mask]

    sub = sub.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if sub.empty:
        return FrequencyPlan("MS", 12, "monthly", 0.0)

    raw = sub.groupby("timestamp")["value"].mean()

    m_cov = raw.resample("MS").count()
    m_obs = int((m_cov > 0).sum())
    m_total = max(1, len(m_cov))
    coverage = m_obs / m_total

    y_cov = raw.resample("YS").count()
    y_obs = int((y_cov > 0).sum())

    if coverage < 0.45 and y_obs >= 15:
        return FrequencyPlan("YS", 1, "yearly", float(coverage))
    return FrequencyPlan("MS", 12, "monthly", float(coverage))


def aggregate_series(obs: pd.DataFrame, variable: str, plan: FrequencyPlan, ok_value: str) -> pd.Series:
    sub = obs[obs["variable"] == variable].copy()
    if sub.empty:
        return pd.Series(dtype=float)

    ok_mask = sub["qc_flag"].astype(str).str.lower().eq(str(ok_value).lower())
    if ok_mask.any():
        sub = sub[ok_mask]

    sub = sub.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if sub.empty:
        return pd.Series(dtype=float)

    raw = sub.groupby("timestamp")["value"].mean()

    if plan.freq == "YS":
        if is_precip(variable):
            s = raw.resample("YS").sum(min_count=1).fillna(0.0)
        else:
            s = raw.resample("YS").mean().interpolate("time").ffill().bfill()
        return s.astype(float)

    if is_precip(variable):
        s = raw.resample("MS").sum(min_count=1).fillna(0.0)
    else:
        s = raw.resample("MS").mean().interpolate("time").ffill().bfill()
    return s.astype(float)


def winsorize(s: pd.Series, ql: float, qu: float) -> pd.Series:
    if s.empty:
        return s
    lo = float(np.clip(ql, 0.0, 0.49))
    hi = float(np.clip(qu, 0.51, 1.0))
    x = s.astype(float).copy()
    x = x.clip(lower=float(x.quantile(lo)), upper=float(x.quantile(hi)))
    x = x.interpolate("time").ffill().bfill()
    return x


def make_design(ds: pd.Series, freq: str, anchor: pd.Timestamp) -> np.ndarray:
    d = pd.to_datetime(ds)
    t = (d - anchor).dt.days.astype(float).values
    scale = max(1.0, float(np.max(np.abs(t))))
    t = t / scale

    cols = [t, t**2]

    if freq == "MS":
        m = d.dt.month.astype(float).values
        q = d.dt.quarter.astype(float).values
        cols += [
            np.sin(2 * np.pi * m / 12.0),
            np.cos(2 * np.pi * m / 12.0),
            np.sin(2 * np.pi * q / 4.0),
            np.cos(2 * np.pi * q / 4.0),
        ]
    else:
        y = (d.dt.year - anchor.year).astype(float).values
        cols += [
            np.sin(2 * np.pi * y / 11.0),
            np.cos(2 * np.pi * y / 11.0),
        ]

    return np.vstack(cols).T


def fit_base_ridge(train_ds: pd.Series, train_y: np.ndarray, freq: str, alpha: float = 1.0) -> tuple[Ridge, pd.Timestamp]:
    anchor = pd.Timestamp(train_ds.min())
    X = make_design(train_ds, freq=freq, anchor=anchor)
    model = Ridge(alpha=float(alpha), random_state=42)
    model.fit(X, np.asarray(train_y, dtype=float))
    return model, anchor


def predict_base(model: Ridge, anchor: pd.Timestamp, ds: pd.Series, freq: str) -> np.ndarray:
    X = make_design(ds, freq=freq, anchor=anchor)
    return model.predict(X).astype(float)


def monthly_climatology(train_ds: pd.Series, train_y: np.ndarray, freq: str) -> dict[int, float]:
    d = pd.to_datetime(train_ds)
    y = np.asarray(train_y, dtype=float)
    out = {}

    if freq == "MS":
        for m in range(1, 13):
            vals = y[d.dt.month.values == m]
            out[m] = float(np.mean(vals)) if len(vals) else float(np.mean(y))
    else:
        out[1] = float(np.mean(y))

    return out


def to_anomaly(ds: pd.Series, y: np.ndarray, freq: str, clim: dict[int, float]) -> np.ndarray:
    d = pd.to_datetime(ds)
    y = np.asarray(y, dtype=float)
    if freq == "MS":
        base = np.array([clim.get(int(m), float(np.mean(y))) for m in d.dt.month.values], dtype=float)
    else:
        base = np.repeat(clim.get(1, float(np.mean(y))), len(y))
    return y - base


def analog_predict(
    train_ds: pd.Series,
    train_y: np.ndarray,
    future_ds: pd.Series,
    freq: str,
    lag: int,
    k: int,
) -> np.ndarray:
    train_ds = pd.to_datetime(train_ds)
    future_ds = pd.to_datetime(future_ds)
    train_y = np.asarray(train_y, dtype=float)

    if len(train_y) < max(8, lag + 3):
        return np.repeat(float(np.mean(train_y)), len(future_ds))

    clim = monthly_climatology(train_ds, train_y, freq=freq)
    anom_hist = to_anomaly(train_ds, train_y, freq=freq, clim=clim)

    pred_vals = []
    pred_anom = []

    hist_anom_ext = list(anom_hist)

    for fds in future_ds:
        if len(hist_anom_ext) < lag + 2:
            pa = float(np.mean(hist_anom_ext)) if hist_anom_ext else 0.0
            pred_anom.append(pa)
            month = int(fds.month) if freq == "MS" else 1
            pred_vals.append(pa + clim.get(month, float(np.mean(train_y))))
            hist_anom_ext.append(pa)
            continue

        context = np.asarray(hist_anom_ext[-lag:], dtype=float)

        cand_next = []
        cand_dist = []
        max_i = len(anom_hist) - 1

        for i in range(lag, max_i):
            wnd = anom_hist[i - lag : i]
            nxt = anom_hist[i]
            if len(wnd) != lag:
                continue

            if freq == "MS":
                nxt_month = int(train_ds.iloc[i].month)
                if nxt_month != int(fds.month):
                    continue

            d = float(np.sqrt(np.mean((wnd - context) ** 2)))
            cand_next.append(float(nxt))
            cand_dist.append(d)

        if not cand_next:
            pa = float(np.mean(hist_anom_ext[-lag:]))
        else:
            order = np.argsort(cand_dist)[: max(1, int(k))]
            dsel = np.asarray([cand_dist[j] for j in order], dtype=float)
            vsel = np.asarray([cand_next[j] for j in order], dtype=float)
            w = 1.0 / (dsel + 1e-6)
            w = w / np.sum(w)
            pa = float(np.sum(w * vsel))

        pred_anom.append(pa)
        month = int(fds.month) if freq == "MS" else 1
        val = pa + clim.get(month, float(np.mean(train_y)))
        pred_vals.append(float(val))
        hist_anom_ext.append(pa)

    return np.asarray(pred_vals, dtype=float)


def split_points(n: int, holdout: int, splits: int, min_train: int) -> list[int]:
    out = []
    for i in range(splits, 0, -1):
        cut = n - i * holdout
        if min_train <= cut < n:
            out.append(cut)
    return sorted(set(out))


def make_cv_plan(n: int, freq: str, season_len: int, args: argparse.Namespace) -> CVPlan:
    if freq == "YS":
        hold = min(max(1, int(args.holdout_steps)), max(1, n // 5))
        hold = min(hold, 3)
        min_floor = max(8, season_len + 5)
    else:
        hold = min(max(1, int(args.holdout_steps)), max(1, n // 4))
        min_floor = max(24, season_len * 2)

    max_min_train = max(2, n - hold)
    min_train = min(max_min_train, max(min_floor, int(round(n * 0.60))))
    if min_train >= n:
        min_train = max(2, n - hold)

    splits = max(1, int(args.backtest_splits))
    while splits > 1 and not split_points(n, hold, splits, min_train):
        splits -= 1
    if not split_points(n, hold, splits, min_train):
        splits = 1

    return CVPlan(holdout=hold, splits=splits, min_train=min_train)


def metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    e = yt - yp
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    bias = float(np.mean(e))

    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    ratio = np.zeros_like(denom, dtype=float)
    mask = denom > 1e-9
    ratio[mask] = np.abs(e[mask]) / denom[mask]
    smape = float(np.mean(ratio) * 100.0)

    return {"mae": mae, "rmse": rmse, "bias": bias, "smape": smape}


def make_future_index(last_ds: pd.Timestamp, periods: int, freq: str) -> pd.DatetimeIndex:
    if periods <= 0:
        return pd.DatetimeIndex([])
    if freq == "YS":
        return pd.date_range(last_ds + pd.offsets.YearBegin(1), periods=periods, freq="YS")
    return pd.date_range(last_ds + pd.offsets.MonthBegin(1), periods=periods, freq="MS")


def horizon_steps(last_ds: pd.Timestamp, target_year: int, freq: str) -> int:
    if freq == "YS":
        return max(0, target_year - last_ds.year)
    target_ds = pd.Timestamp(year=target_year, month=12, day=1)
    h = int((target_ds.year - last_ds.year) * 12 + (target_ds.month - last_ds.month))
    return max(0, h)


def rolling_cv_scores(ds: pd.Series, y: np.ndarray, freq: str, season_len: int, cv: CVPlan, analog_lag: int, analog_k: int) -> tuple[dict[str, float], list[float]]:
    cuts = split_points(len(y), holdout=cv.holdout, splits=cv.splits, min_train=cv.min_train)
    if not cuts:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "smape": np.nan,
            "rmse_std": np.nan,
            "n_folds": 0,
            "w_base": np.nan,
            "w_analog": np.nan,
        }, []

    fold_rmse = []
    fold_mae = []
    fold_bias = []
    fold_smape = []
    fold_abs_err = []

    for cut in cuts:
        tr_ds = pd.to_datetime(ds.iloc[:cut]).reset_index(drop=True)
        te_ds = pd.to_datetime(ds.iloc[cut : cut + cv.holdout]).reset_index(drop=True)
        tr_y = np.asarray(y[:cut], dtype=float)
        te_y = np.asarray(y[cut : cut + cv.holdout], dtype=float)

        if len(te_y) == 0:
            continue

        # base
        base_model, anchor = fit_base_ridge(tr_ds, tr_y, freq=freq, alpha=1.0)
        base_pred = predict_base(base_model, anchor=anchor, ds=te_ds, freq=freq)

        # analog
        analog_pred = analog_predict(
            train_ds=tr_ds,
            train_y=tr_y,
            future_ds=te_ds,
            freq=freq,
            lag=int(analog_lag),
            k=int(analog_k),
        )

        # dynamic blend by inner fit quality
        m_base = metric_pack(te_y, base_pred)
        m_analog = metric_pack(te_y, analog_pred)

        if np.isfinite(m_base["rmse"]) and np.isfinite(m_analog["rmse"]):
            ib = 1.0 / max(1e-6, m_base["rmse"])
            ia = 1.0 / max(1e-6, m_analog["rmse"])
            s = ib + ia
            wb = ib / s
            wa = ia / s
        else:
            wb = 0.6
            wa = 0.4

        pred = wb * base_pred + wa * analog_pred
        met = metric_pack(te_y, pred)

        fold_rmse.append(met["rmse"])
        fold_mae.append(met["mae"])
        fold_bias.append(met["bias"])
        fold_smape.append(met["smape"])
        fold_abs_err.extend(np.abs(te_y - pred).tolist())

    if not fold_rmse:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "smape": np.nan,
            "rmse_std": np.nan,
            "n_folds": 0,
            "w_base": np.nan,
            "w_analog": np.nan,
        }, []

    # choose global blend from fold means
    rb = np.mean(fold_rmse)
    ib = 1.0 / max(1e-6, rb)
    ia = 1.0 / max(1e-6, rb * 1.03)
    s = ib + ia
    w_base = ib / s
    w_analog = ia / s

    return {
        "rmse": float(np.mean(fold_rmse)),
        "mae": float(np.mean(fold_mae)),
        "bias": float(np.mean(fold_bias)),
        "smape": float(np.mean(fold_smape)),
        "rmse_std": float(np.std(fold_rmse)),
        "n_folds": int(len(fold_rmse)),
        "w_base": float(w_base),
        "w_analog": float(w_analog),
    }, fold_abs_err


def build_final_forecast(ds: pd.Series, y: np.ndarray, future_ds: pd.Series, freq: str, analog_lag: int, analog_k: int, w_base: float, w_analog: float) -> np.ndarray:
    model, anchor = fit_base_ridge(ds, y, freq=freq, alpha=1.0)
    base_pred = predict_base(model, anchor=anchor, ds=future_ds, freq=freq)
    analog_pred = analog_predict(
        train_ds=ds,
        train_y=y,
        future_ds=future_ds,
        freq=freq,
        lag=int(analog_lag),
        k=int(analog_k),
    )
    return w_base * base_pred + w_analog * analog_pred


def conformal_band(abs_errors: list[float], alpha: float) -> float:
    if not abs_errors:
        return float("nan")
    q = float(np.clip(1.0 - alpha, 0.50, 0.99))
    return float(np.quantile(np.asarray(abs_errors, dtype=float), q))


def plot_series(out_df: pd.DataFrame, last_ds: pd.Timestamp, variable: str, plan: FrequencyPlan, chart_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    hist = out_df[out_df["ds"] <= last_ds]
    fc = out_df[out_df["ds"] > last_ds]

    ax.plot(hist["ds"], hist["yhat"], color="#1f77b4", linewidth=1.4, label="historical")
    if not fc.empty:
        ax.plot(fc["ds"], fc["yhat"], color="#d62728", linewidth=2.0, label="analog+trend forecast")
        if fc["yhat_lower"].notna().any() and fc["yhat_upper"].notna().any():
            ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], color="#d62728", alpha=0.15, label="conformal band")

    ax.axvline(last_ds, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_title(f"Analog Pattern Forecast - {variable} ({plan.label})")
    ax.set_xlabel("date")
    ax.set_ylabel(f"value ({infer_unit(variable)})")
    ax.grid(alpha=0.24)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    raw = read_table(args.observations)
    obs, input_kind = normalize_observations(raw, args)
    vars_use = select_variables(obs, args.variables)

    if not vars_use:
        raise SystemExit("No variables selected")

    out = args.output_dir
    fc_dir = out / "forecasts"
    ch_dir = out / "charts"
    rep_dir = out / "reports"
    ensure_dirs([out, fc_dir, ch_dir, rep_dir])

    print(f"Input: {args.observations} | kind={input_kind} | rows={len(obs)} | variables={','.join(vars_use)}")

    index_rows = []

    for variable in vars_use:
        plan = choose_frequency_plan(obs, variable=variable, ok_value=args.qc_ok_value)
        s = aggregate_series(obs, variable=variable, plan=plan, ok_value=args.qc_ok_value)
        if s.empty:
            continue

        s = winsorize(s, ql=args.winsor_lower, qu=args.winsor_upper)

        ds = pd.to_datetime(s.index.to_series()).reset_index(drop=True)
        y = s.values.astype(float)

        last_ds = pd.Timestamp(ds.iloc[-1])
        horizon = horizon_steps(last_ds, target_year=args.target_year, freq=plan.freq)

        cv = make_cv_plan(len(y), freq=plan.freq, season_len=plan.season_len, args=args)

        scores, abs_err = rolling_cv_scores(
            ds=ds,
            y=y,
            freq=plan.freq,
            season_len=plan.season_len,
            cv=cv,
            analog_lag=int(args.analog_lag),
            analog_k=int(args.analog_k),
        )

        w_base = float(scores["w_base"]) if np.isfinite(scores["w_base"]) else 0.6
        w_analog = float(scores["w_analog"]) if np.isfinite(scores["w_analog"]) else 0.4

        fidx = make_future_index(last_ds, periods=horizon, freq=plan.freq)
        fds = pd.Series(fidx)

        y_fc = build_final_forecast(
            ds=ds,
            y=y,
            future_ds=fds,
            freq=plan.freq,
            analog_lag=int(args.analog_lag),
            analog_k=int(args.analog_k),
            w_base=w_base,
            w_analog=w_analog,
        )

        # bias correction from CV residual mean
        bias_corr = float(scores["bias"]) if np.isfinite(scores["bias"]) else 0.0
        y_fc = y_fc + bias_corr

        # conformal interval
        q_abs = conformal_band(abs_err, alpha=float(args.interval_alpha))
        if not np.isfinite(q_abs):
            q_abs = float(np.nanstd(y) * 0.1)

        lo = y_fc - q_abs
        hi = y_fc + q_abs

        y_hist = apply_bounds(y, variable)
        y_fc = apply_bounds(y_fc, variable)
        lo = apply_bounds(lo, variable)
        hi = apply_bounds(hi, variable)

        hist_df = pd.DataFrame(
            {
                "ds": ds,
                "yhat": y_hist,
                "yhat_lower": np.nan,
                "yhat_upper": np.nan,
                "is_forecast": False,
            }
        )

        fc_df = pd.DataFrame(
            {
                "ds": fds,
                "yhat": y_fc,
                "yhat_lower": lo,
                "yhat_upper": hi,
                "is_forecast": True,
            }
        )

        out_df = pd.concat([hist_df, fc_df], ignore_index=True)
        out_df["variable"] = variable
        out_df["unit"] = infer_unit(variable)
        out_df["frequency"] = plan.freq
        out_df["model_strategy"] = "analog_pattern_trend"

        freq_tag = "yearly" if plan.freq == "YS" else "monthly"

        fc_csv = fc_dir / f"{variable}_{freq_tag}_analog_to_{args.target_year}.csv"
        fc_pq = fc_dir / f"{variable}_{freq_tag}_analog_to_{args.target_year}.parquet"
        out_df.to_csv(fc_csv, index=False)
        out_df.to_parquet(fc_pq, index=False)

        chart_png = ch_dir / f"{variable}_{freq_tag}_analog_to_{args.target_year}.png"
        plot_series(out_df=out_df, last_ds=last_ds, variable=variable, plan=plan, chart_path=chart_png)

        rep = {
            "variable": variable,
            "frequency": plan.freq,
            "frequency_label": plan.label,
            "monthly_coverage": float(plan.monthly_coverage),
            "cv_holdout_steps": int(cv.holdout),
            "cv_splits": int(cv.splits),
            "cv_min_train_steps": int(cv.min_train),
            "rmse": float(scores["rmse"]) if np.isfinite(scores["rmse"]) else None,
            "mae": float(scores["mae"]) if np.isfinite(scores["mae"]) else None,
            "bias": float(scores["bias"]) if np.isfinite(scores["bias"]) else None,
            "smape": float(scores["smape"]) if np.isfinite(scores["smape"]) else None,
            "rmse_std": float(scores["rmse_std"]) if np.isfinite(scores["rmse_std"]) else None,
            "w_base": float(w_base),
            "w_analog": float(w_analog),
            "bias_correction": float(bias_corr),
            "conformal_q_abs": float(q_abs),
            "forecast_csv": str(fc_csv),
            "forecast_parquet": str(fc_pq),
            "chart_png": str(chart_png),
        }
        rep_json = rep_dir / f"{variable}_{freq_tag}_analog_report_to_{args.target_year}.json"
        rep_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

        index_rows.append(
            {
                "variable": variable,
                "frequency": plan.freq,
                "frequency_label": plan.label,
                "monthly_coverage": plan.monthly_coverage,
                "target_year": args.target_year,
                "cv_holdout_steps": cv.holdout,
                "cv_splits": cv.splits,
                "cv_min_train_steps": cv.min_train,
                "rmse": scores["rmse"],
                "mae": scores["mae"],
                "bias": scores["bias"],
                "smape": scores["smape"],
                "rmse_std": scores["rmse_std"],
                "w_base": w_base,
                "w_analog": w_analog,
                "bias_correction": bias_corr,
                "conformal_q_abs": q_abs,
                "model_strategy": "analog_pattern_trend",
                "forecast_csv": str(fc_csv),
                "forecast_parquet": str(fc_pq),
                "chart_png": str(chart_png),
                "report_json": str(rep_json),
                "input_kind": input_kind,
                "input_path": str(args.observations),
            }
        )

    idx = pd.DataFrame(index_rows).sort_values("variable") if index_rows else pd.DataFrame()
    idx_csv = out / f"analog_index_to_{args.target_year}.csv"
    idx_pq = out / f"analog_index_to_{args.target_year}.parquet"
    idx.to_csv(idx_csv, index=False)
    idx.to_parquet(idx_pq, index=False)

    print("Analog pattern pipeline completed.")
    print(f"Index: {idx_csv}")
    if not idx.empty:
        print(idx[["variable", "frequency", "rmse", "w_base", "w_analog", "forecast_csv"]].to_string(index=False))


if __name__ == "__main__":
    main()
