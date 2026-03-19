#!/usr/bin/env python3
"""Ultra Prophet 500+ line training pipeline for robust climate forecasting.

Goal:
- Provide a standalone Prophet-first model pipeline with strong consistency.
- Minimize prediction deviation (bias + variance) using time-series CV.
- Produce clean, reproducible outputs for hackathon delivery.

Design principles:
1) Prophet-centric: core model family is Prophet.
2) Low deviation: candidate scoring includes RMSE + |bias| + fold instability.
3) Robust preprocessing: winsorization, imputation, optional log transform.
4) Dynamic CV: adapts train/holdout sizes to sparse series.
5) Bias correction: apply residual-mean correction to final forecast.
6) Fallback safety: seasonal naive fallback when data is too short.
7) Traceability: leaderboard, run reports, and index files are always written.

Input data support:
- parquet / csv / tsv / xlsx / xls / ods
- long form: timestamp + variable + value (+ optional qc)
- single series: timestamp + value

Outputs:
- forecasts/*.csv|parquet
- charts/*.png
- components/*.png
- leaderboards/*.csv
- reports/*.json
- prophet_ultra_index_to_<year>.csv|parquet
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Environment and cache guards
# -----------------------------------------------------------------------------

_CACHE_ROOT = Path(tempfile.gettempdir()) / "prophet_ultra_500_cache"
_MPL_CACHE = _CACHE_ROOT / "mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from prophet import Prophet
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Prophet import failed. Install first:\n"
        "  pip install prophet\n"
        f"Original error: {exc}"
    )

try:
    import cmdstanpy

    cmdstanpy.disable_logging()
except Exception:
    pass

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


# -----------------------------------------------------------------------------
# Constants and aliases
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class FrequencyPlan:
    freq: str
    season_len: int
    label: str
    monthly_coverage: float


@dataclass
class ProphetConfig:
    seasonality_mode: str
    changepoint_prior_scale: float
    seasonality_prior_scale: float
    changepoint_range: float
    interval_width: float
    n_changepoints: int
    yearly_fourier: int
    quarterly_fourier: int
    monthly_fourier: int


@dataclass
class CandidateResult:
    model_label: str
    cfg_json: str
    rmse: float
    mae: float
    smape: float
    bias: float
    rmse_std: float
    bias_std: float
    score: float
    n_folds: int
    status: str
    error: str


@dataclass
class CVPlan:
    holdout: int
    splits: int
    min_train: int


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ultra Prophet 500+ robust forecasting pipeline")

    p.add_argument(
        "--observations",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/forecast_package/observations_with_graph.parquet"),
        help="Input dataset path",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/prophet_ultra_500"),
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

    p.add_argument("--holdout-steps", type=int, default=12)
    p.add_argument("--backtest-splits", type=int, default=3)
    p.add_argument("--min-train-steps", type=int, default=36)

    p.add_argument("--winsor-lower", type=float, default=0.005)
    p.add_argument("--winsor-upper", type=float, default=0.995)

    p.add_argument("--bias-weight", type=float, default=0.50)
    p.add_argument("--stability-weight", type=float, default=0.25)

    p.add_argument("--max-ensemble-models", type=int, default=2)
    p.add_argument("--interval-z", type=float, default=1.96)

    p.add_argument("--fast-grid", type=str, default="true", help="true/false")
    p.add_argument("--report-top", type=int, default=8)

    return p.parse_args()


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def to_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def timestamp_safe(v: Any) -> pd.Timestamp:
    return pd.Timestamp(v)


# -----------------------------------------------------------------------------
# Input reading
# -----------------------------------------------------------------------------

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
        raise SystemExit("Cannot detect time/value columns. Use --timestamp-col and --value-col.")

    input_kind = args.input_kind
    if input_kind == "auto":
        input_kind = "long" if var_col is not None else "single"

    if input_kind == "long":
        if var_col is None:
            raise SystemExit("input-kind=long requires a variable column")
        out = pd.DataFrame(
            {
                "timestamp": raw[ts_col],
                "variable": raw[var_col],
                "value": raw[val_col],
            }
        )
    else:
        out = pd.DataFrame(
            {
                "timestamp": raw[ts_col],
                "variable": args.single_variable,
                "value": raw[val_col],
            }
        )

    if qc_col is not None:
        out["qc_flag"] = raw[qc_col].astype(str)
    else:
        out["qc_flag"] = args.qc_ok_value

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["variable"] = out["variable"].astype(str).map(canonical_variable_name)
    out["qc_flag"] = out["qc_flag"].astype(str)

    out = out.dropna(subset=["timestamp", "value", "variable"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    if out.empty:
        raise SystemExit("No usable rows after parsing.")

    return out, input_kind


def select_variables(obs: pd.DataFrame, variables_arg: str) -> list[str]:
    available = sorted(obs["variable"].dropna().unique().tolist())
    if not variables_arg or variables_arg.strip() in {"*", "all", "ALL"}:
        return available
    req = [canonical_variable_name(x.strip()) for x in variables_arg.split(",") if x.strip()]
    req = [x for x in req if x in available]
    return sorted(set(req))


# -----------------------------------------------------------------------------
# Frequency planning and aggregation
# -----------------------------------------------------------------------------

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

    m_cov_series = raw.resample("MS").count()
    m_obs = int((m_cov_series > 0).sum())
    m_total = max(1, int(len(m_cov_series)))
    m_cov = m_obs / m_total

    y_cov_series = raw.resample("YS").count()
    y_obs = int((y_cov_series > 0).sum())

    # Sparse monthly coverage + enough yearly history => yearly plan.
    if m_cov < 0.45 and y_obs >= 15:
        return FrequencyPlan("YS", 1, "yearly", float(m_cov))

    return FrequencyPlan("MS", 12, "monthly", float(m_cov))


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


# -----------------------------------------------------------------------------
# Preprocessing and transforms
# -----------------------------------------------------------------------------

def winsorize_and_impute(s: pd.Series, q_low: float, q_high: float) -> pd.Series:
    if s.empty:
        return s

    x = s.copy().astype(float)
    lo = float(np.clip(q_low, 0.0, 0.49))
    hi = float(np.clip(q_high, 0.51, 1.0))

    lo_v = float(x.quantile(lo))
    hi_v = float(x.quantile(hi))
    x = x.clip(lower=lo_v, upper=hi_v)

    if x.isna().any():
        x = x.interpolate("time").ffill().bfill()
    return x


def to_model_frame(s: pd.Series, variable: str) -> tuple[pd.DataFrame, bool]:
    use_log = is_precip(variable)
    y = s.values.astype(float)
    if use_log:
        y = np.log1p(np.clip(y, 0, None))

    df = pd.DataFrame({"ds": s.index, "y": y})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna().sort_values("ds").reset_index(drop=True)
    return df, use_log


def invert_target(y: np.ndarray, use_log: bool) -> np.ndarray:
    x = np.asarray(y, dtype=float)
    return np.expm1(x) if use_log else x


# -----------------------------------------------------------------------------
# Regressor engineering (deterministic calendar features)
# -----------------------------------------------------------------------------

def build_regressors(ds: pd.Series, freq: str, anchor: pd.Timestamp) -> pd.DataFrame:
    ds_t = pd.to_datetime(ds)
    out = pd.DataFrame({"ds": ds_t})

    # normalized trend index
    trend_idx = (ds_t - anchor).dt.days.astype(float)
    scale = float(max(1.0, np.nanmax(np.abs(trend_idx.values))))
    out["trend_idx"] = trend_idx / scale

    if freq == "MS":
        month = ds_t.dt.month.astype(float)
        quarter = ds_t.dt.quarter.astype(float)
        out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
        out["quarter_sin"] = np.sin(2.0 * np.pi * quarter / 4.0)
        out["quarter_cos"] = np.cos(2.0 * np.pi * quarter / 4.0)
    else:
        # yearly cyclic proxy (11-year solar-ish cycle placeholder)
        year_delta = (ds_t.dt.year - int(anchor.year)).astype(float)
        out["cycle11_sin"] = np.sin(2.0 * np.pi * year_delta / 11.0)
        out["cycle11_cos"] = np.cos(2.0 * np.pi * year_delta / 11.0)

    return out


def regressor_cols(freq: str) -> list[str]:
    if freq == "MS":
        return ["trend_idx", "month_sin", "month_cos", "quarter_sin", "quarter_cos"]
    return ["trend_idx", "cycle11_sin", "cycle11_cos"]


def attach_regressors(df: pd.DataFrame, freq: str, anchor: pd.Timestamp) -> pd.DataFrame:
    regs = build_regressors(df["ds"], freq=freq, anchor=anchor)
    out = df.merge(regs, on="ds", how="left")
    return out


def make_future_with_regressors(model: Prophet, periods: int, freq: str, anchor: pd.Timestamp) -> pd.DataFrame:
    fut = model.make_future_dataframe(periods=periods, freq=freq)
    regs = build_regressors(fut["ds"], freq=freq, anchor=anchor)
    fut = fut.merge(regs, on="ds", how="left")
    return fut


# -----------------------------------------------------------------------------
# Candidate configs
# -----------------------------------------------------------------------------

def build_prophet_grid(variable: str, interval_width: float, fast_grid: bool) -> list[ProphetConfig]:
    if fast_grid:
        cps = [0.03, 0.10]
        sps = [10.0, 20.0]
        crs = [0.90]
        ncp = [15, 30]
        yf = [8, 12]
        qf = [4]
        mf = [3]
    else:
        cps = [0.03, 0.10, 0.25]
        sps = [5.0, 10.0, 20.0]
        crs = [0.85, 0.95]
        ncp = [10, 20, 30]
        yf = [6, 10, 14]
        qf = [3, 5]
        mf = [2, 4]

    if is_precip(variable):
        modes = ["additive", "multiplicative"]
    else:
        modes = ["additive"]

    out: list[ProphetConfig] = []
    for mode, cp, sp, cr, nc, yfo, qfo, mfo in itertools.product(modes, cps, sps, crs, ncp, yf, qf, mf):
        out.append(
            ProphetConfig(
                seasonality_mode=mode,
                changepoint_prior_scale=float(cp),
                seasonality_prior_scale=float(sp),
                changepoint_range=float(cr),
                interval_width=float(interval_width),
                n_changepoints=int(nc),
                yearly_fourier=int(yfo),
                quarterly_fourier=int(qfo),
                monthly_fourier=int(mfo),
            )
        )

    # De-duplicate in case grid params collide.
    uniq = {}
    for cfg in out:
        k = json.dumps(cfg.__dict__, sort_keys=True)
        uniq[k] = cfg
    return list(uniq.values())


# -----------------------------------------------------------------------------
# Prophet fit/predict
# -----------------------------------------------------------------------------

def build_prophet_model(cfg: ProphetConfig, freq: str, anchor: pd.Timestamp) -> Prophet:
    yearly_flag = True if freq == "MS" else False

    m = Prophet(
        yearly_seasonality=yearly_flag,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=cfg.seasonality_mode,
        interval_width=cfg.interval_width,
        changepoint_prior_scale=cfg.changepoint_prior_scale,
        seasonality_prior_scale=cfg.seasonality_prior_scale,
        changepoint_range=cfg.changepoint_range,
        n_changepoints=cfg.n_changepoints,
    )

    for r in regressor_cols(freq):
        m.add_regressor(r, standardize=True)

    if freq == "MS":
        m.add_seasonality(name="yearly_custom", period=365.25, fourier_order=cfg.yearly_fourier)
        m.add_seasonality(name="quarterly", period=365.25 / 4.0, fourier_order=cfg.quarterly_fourier)
        m.add_seasonality(name="monthly_cycle", period=365.25 / 12.0, fourier_order=cfg.monthly_fourier)
    else:
        m.add_seasonality(name="long_cycle", period=365.25 * 11.0, fourier_order=3)

    return m


def fit_predict_prophet(
    train_df: pd.DataFrame,
    periods: int,
    freq: str,
    cfg: ProphetConfig,
    anchor: pd.Timestamp,
) -> pd.DataFrame:
    m = build_prophet_model(cfg, freq=freq, anchor=anchor)
    train_aug = attach_regressors(train_df, freq=freq, anchor=anchor)

    # Ensure regressor columns exist and are finite.
    for c in regressor_cols(freq):
        if c not in train_aug.columns:
            raise ValueError(f"Missing regressor column in train: {c}")
        if train_aug[c].isna().any():
            train_aug[c] = train_aug[c].interpolate("linear").ffill().bfill()

    m.fit(train_aug)
    future = make_future_with_regressors(m, periods=periods, freq=freq, anchor=anchor)

    for c in regressor_cols(freq):
        if c not in future.columns:
            raise ValueError(f"Missing regressor column in future: {c}")
        if future[c].isna().any():
            future[c] = future[c].interpolate("linear").ffill().bfill()

    pred = m.predict(future)
    return pred


# -----------------------------------------------------------------------------
# Baseline fallback model
# -----------------------------------------------------------------------------

def seasonal_naive_predict(train_df: pd.DataFrame, periods: int, season_len: int) -> np.ndarray:
    y = np.asarray(train_df["y"].values, dtype=float)
    if periods <= 0:
        return np.array([], dtype=float)

    if len(y) == 0:
        return np.zeros(periods, dtype=float)

    if season_len <= 1 or len(y) < season_len:
        base = float(np.mean(y[-min(3, len(y)) :]))
        return np.repeat(base, periods)

    season = y[-season_len:]
    reps = int(np.ceil(periods / season_len))
    return np.tile(season, reps)[:periods]


# -----------------------------------------------------------------------------
# CV and metrics
# -----------------------------------------------------------------------------

def split_points(n: int, holdout: int, splits: int, min_train: int) -> list[int]:
    out: list[int] = []
    for i in range(splits, 0, -1):
        cut = n - i * holdout
        if min_train <= cut < n:
            out.append(cut)
    return sorted(set(out))


def make_cv_plan(n: int, freq: str, season_len: int, args: argparse.Namespace) -> CVPlan:
    if freq == "YS":
        holdout = min(max(1, int(args.holdout_steps)), max(1, n // 5))
        holdout = min(holdout, 3)
        min_train_floor = max(8, season_len + 5)
    else:
        holdout = min(max(1, int(args.holdout_steps)), max(1, n // 4))
        min_train_floor = max(24, season_len * 2)

    max_min_train = max(2, n - holdout)
    min_train = min(max_min_train, max(min_train_floor, int(round(n * 0.60))))
    if min_train >= n:
        min_train = max(2, n - holdout)

    splits = max(1, int(args.backtest_splits))
    while splits > 1 and not split_points(n, holdout, splits, min_train):
        splits -= 1
    if not split_points(n, holdout, splits, min_train):
        splits = 1

    return CVPlan(holdout=holdout, splits=splits, min_train=min_train)


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

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "smape": smape,
    }


def score_with_bias_stability(metrics: dict[str, float], bias_weight: float, stability_weight: float) -> float:
    rmse = float(metrics.get("rmse", np.nan))
    bias = float(metrics.get("bias", np.nan))
    rmse_std = float(metrics.get("rmse_std", np.nan))

    if not np.isfinite(rmse):
        return np.inf

    bias_term = abs(bias) * max(0.0, float(bias_weight))
    st_term = max(0.0, float(stability_weight)) * (rmse_std if np.isfinite(rmse_std) else rmse)
    return float(rmse + bias_term + st_term)


def evaluate_prophet_candidate(
    df_model: pd.DataFrame,
    variable: str,
    freq: str,
    season_len: int,
    cfg: ProphetConfig,
    cv: CVPlan,
    use_log: bool,
    bias_weight: float,
    stability_weight: float,
) -> CandidateResult:
    cuts = split_points(len(df_model), cv.holdout, cv.splits, cv.min_train)
    if not cuts:
        return CandidateResult(
            model_label="prophet",
            cfg_json=json.dumps(cfg.__dict__, sort_keys=True),
            rmse=np.nan,
            mae=np.nan,
            smape=np.nan,
            bias=np.nan,
            rmse_std=np.nan,
            bias_std=np.nan,
            score=np.inf,
            n_folds=0,
            status="no_folds",
            error="insufficient folds",
        )

    rmse_list = []
    mae_list = []
    smape_list = []
    bias_list = []

    anchor = timestamp_safe(df_model["ds"].min())

    for cut in cuts:
        train = df_model.iloc[:cut].copy()
        test = df_model.iloc[cut : cut + cv.holdout].copy()
        if len(test) == 0:
            continue

        try:
            pred = fit_predict_prophet(train, periods=len(test), freq=freq, cfg=cfg, anchor=anchor)
            yhat_t = pred[["ds", "yhat"]].tail(len(test)).copy()
            chk = test[["ds", "y"]].merge(yhat_t, on="ds", how="inner")
            if chk.empty:
                continue

            y_true = apply_bounds(invert_target(chk["y"].values, use_log=use_log), variable)
            y_pred = apply_bounds(invert_target(chk["yhat"].values, use_log=use_log), variable)
            met = metric_pack(y_true, y_pred)
            rmse_list.append(met["rmse"])
            mae_list.append(met["mae"])
            smape_list.append(met["smape"])
            bias_list.append(met["bias"])
        except Exception:
            continue

    if not rmse_list:
        return CandidateResult(
            model_label="prophet",
            cfg_json=json.dumps(cfg.__dict__, sort_keys=True),
            rmse=np.nan,
            mae=np.nan,
            smape=np.nan,
            bias=np.nan,
            rmse_std=np.nan,
            bias_std=np.nan,
            score=np.inf,
            n_folds=0,
            status="failed",
            error="all folds failed",
        )

    rmse = float(np.mean(rmse_list))
    mae = float(np.mean(mae_list))
    smape = float(np.mean(smape_list))
    bias = float(np.mean(bias_list))
    rmse_std = float(np.std(rmse_list))
    bias_std = float(np.std(bias_list))

    sc = score_with_bias_stability(
        {
            "rmse": rmse,
            "bias": bias,
            "rmse_std": rmse_std,
        },
        bias_weight=bias_weight,
        stability_weight=stability_weight,
    )

    return CandidateResult(
        model_label="prophet",
        cfg_json=json.dumps(cfg.__dict__, sort_keys=True),
        rmse=rmse,
        mae=mae,
        smape=smape,
        bias=bias,
        rmse_std=rmse_std,
        bias_std=bias_std,
        score=sc,
        n_folds=len(rmse_list),
        status="ok",
        error="",
    )


def evaluate_seasonal_naive(
    df_model: pd.DataFrame,
    variable: str,
    season_len: int,
    cv: CVPlan,
    use_log: bool,
    bias_weight: float,
    stability_weight: float,
) -> CandidateResult:
    cuts = split_points(len(df_model), cv.holdout, cv.splits, cv.min_train)
    if not cuts:
        return CandidateResult(
            model_label="seasonal_naive",
            cfg_json="",
            rmse=np.nan,
            mae=np.nan,
            smape=np.nan,
            bias=np.nan,
            rmse_std=np.nan,
            bias_std=np.nan,
            score=np.inf,
            n_folds=0,
            status="no_folds",
            error="insufficient folds",
        )

    rmse_list = []
    mae_list = []
    smape_list = []
    bias_list = []

    for cut in cuts:
        train = df_model.iloc[:cut].copy()
        test = df_model.iloc[cut : cut + cv.holdout].copy()
        if len(test) == 0:
            continue

        try:
            y_pred_t = seasonal_naive_predict(train, periods=len(test), season_len=season_len)
            y_true = apply_bounds(invert_target(test["y"].values, use_log=use_log), variable)
            y_pred = apply_bounds(invert_target(y_pred_t, use_log=use_log), variable)
            met = metric_pack(y_true, y_pred)
            rmse_list.append(met["rmse"])
            mae_list.append(met["mae"])
            smape_list.append(met["smape"])
            bias_list.append(met["bias"])
        except Exception:
            continue

    if not rmse_list:
        return CandidateResult(
            model_label="seasonal_naive",
            cfg_json="",
            rmse=np.nan,
            mae=np.nan,
            smape=np.nan,
            bias=np.nan,
            rmse_std=np.nan,
            bias_std=np.nan,
            score=np.inf,
            n_folds=0,
            status="failed",
            error="all folds failed",
        )

    rmse = float(np.mean(rmse_list))
    mae = float(np.mean(mae_list))
    smape = float(np.mean(smape_list))
    bias = float(np.mean(bias_list))
    rmse_std = float(np.std(rmse_list))
    bias_std = float(np.std(bias_list))

    sc = score_with_bias_stability(
        {
            "rmse": rmse,
            "bias": bias,
            "rmse_std": rmse_std,
        },
        bias_weight=bias_weight,
        stability_weight=stability_weight,
    )

    return CandidateResult(
        model_label="seasonal_naive",
        cfg_json="",
        rmse=rmse,
        mae=mae,
        smape=smape,
        bias=bias,
        rmse_std=rmse_std,
        bias_std=bias_std,
        score=sc,
        n_folds=len(rmse_list),
        status="ok",
        error="",
    )


# -----------------------------------------------------------------------------
# Final forecasting and uncertainty
# -----------------------------------------------------------------------------

def make_horizon(last_ds: pd.Timestamp, target_year: int, freq: str) -> int:
    if freq == "YS":
        return max(0, target_year - int(last_ds.year))

    target_ds = pd.Timestamp(year=target_year, month=12, day=1)
    months = int((target_ds.year - last_ds.year) * 12 + (target_ds.month - last_ds.month))
    return max(0, months)


def make_future_index(last_ds: pd.Timestamp, periods: int, freq: str) -> pd.DatetimeIndex:
    if periods <= 0:
        return pd.DatetimeIndex([])
    if freq == "YS":
        return pd.date_range(last_ds + pd.offsets.YearBegin(1), periods=periods, freq="YS")
    return pd.date_range(last_ds + pd.offsets.MonthBegin(1), periods=periods, freq="MS")


def compute_interval_series(yhat: np.ndarray, resid_sigma: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    if len(yhat) == 0:
        return np.array([]), np.array([])

    sigma = float(resid_sigma) if np.isfinite(resid_sigma) and resid_sigma > 0 else float(np.nanstd(yhat) * 0.10)
    horizon = np.arange(1, len(yhat) + 1, dtype=float)
    growth = np.sqrt(1.0 + horizon / max(12.0, float(len(horizon))))
    band = float(z) * sigma * growth

    lower = yhat - band
    upper = yhat + band
    return lower, upper


def run_final_prophet(
    df_model: pd.DataFrame,
    variable: str,
    freq: str,
    cfg: ProphetConfig,
    use_log: bool,
    horizon: int,
    bias_correction: float,
    resid_sigma: float,
    interval_z: float,
) -> tuple[pd.DataFrame, Prophet]:
    anchor = timestamp_safe(df_model["ds"].min())
    pred_all = fit_predict_prophet(df_model, periods=horizon, freq=freq, cfg=cfg, anchor=anchor)

    # transform back
    yhat = invert_target(pred_all["yhat"].values, use_log=use_log)

    # bias correction for low deviation
    yhat = yhat + float(bias_correction)
    yhat = apply_bounds(yhat, variable)

    # uncertainty from CV residual sigma
    lo, hi = compute_interval_series(yhat, resid_sigma=resid_sigma, z=interval_z)
    lo = apply_bounds(lo, variable)
    hi = apply_bounds(hi, variable)

    out = pd.DataFrame(
        {
            "ds": pd.to_datetime(pred_all["ds"]),
            "yhat": yhat,
            "yhat_lower": lo,
            "yhat_upper": hi,
        }
    )

    # Refit model object for component plot (same configuration).
    m = build_prophet_model(cfg, freq=freq, anchor=anchor)
    train_aug = attach_regressors(df_model, freq=freq, anchor=anchor)
    m.fit(train_aug)

    return out, m


def run_final_seasonal_naive(
    df_model: pd.DataFrame,
    variable: str,
    freq: str,
    season_len: int,
    use_log: bool,
    horizon: int,
    bias_correction: float,
    resid_sigma: float,
    interval_z: float,
) -> tuple[pd.DataFrame, None]:
    y_hist = apply_bounds(invert_target(df_model["y"].values, use_log=use_log), variable)

    y_fc_t = seasonal_naive_predict(df_model, periods=horizon, season_len=season_len)
    y_fc = apply_bounds(invert_target(y_fc_t, use_log=use_log), variable)
    y_fc = y_fc + float(bias_correction)
    y_fc = apply_bounds(y_fc, variable)

    lo, hi = compute_interval_series(y_fc, resid_sigma=resid_sigma, z=interval_z)
    lo = apply_bounds(lo, variable)
    hi = apply_bounds(hi, variable)

    hist_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(df_model["ds"]),
            "yhat": y_hist,
            "yhat_lower": np.nan,
            "yhat_upper": np.nan,
        }
    )

    last_ds = timestamp_safe(df_model["ds"].max())
    fidx = make_future_index(last_ds, periods=horizon, freq=freq)

    fc_df = pd.DataFrame(
        {
            "ds": fidx,
            "yhat": y_fc,
            "yhat_lower": lo,
            "yhat_upper": hi,
        }
    )

    out = pd.concat([hist_df, fc_df], ignore_index=True)
    return out, None


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_forecast(out_df: pd.DataFrame, last_ds: pd.Timestamp, variable: str, freq_label: str, unit: str, chart_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))

    hist = out_df[out_df["ds"] <= last_ds]
    fc = out_df[out_df["ds"] > last_ds]

    ax.plot(hist["ds"], hist["yhat"], color="#1f77b4", linewidth=1.4, label="historical")
    if not fc.empty:
        ax.plot(fc["ds"], fc["yhat"], color="#d62728", linewidth=2.0, label="forecast")
        if fc["yhat_lower"].notna().any() and fc["yhat_upper"].notna().any():
            ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], color="#d62728", alpha=0.15, label="uncertainty")

    ax.axvline(last_ds, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_title(f"Ultra Prophet Forecast - {variable} ({freq_label})")
    ax.set_xlabel("date")
    ax.set_ylabel(f"value ({unit})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=170)
    plt.close(fig)


def plot_components_safe(
    model: Prophet | None,
    out_df: pd.DataFrame,
    comp_path: Path,
) -> None:
    if model is None:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.text(0.03, 0.65, "Components unavailable\n(seasonal naive fallback)", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(comp_path, dpi=160)
        plt.close(fig)
        return

    # Prophet component plot uses model prediction frame.
    try:
        fig = model.plot_components(model.predict(out_df[["ds"]].copy()))
        fig.tight_layout()
        fig.savefig(comp_path, dpi=160)
        plt.close(fig)
    except Exception:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.text(0.03, 0.65, "Components plot failed\n(model forecast still available)", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(comp_path, dpi=160)
        plt.close(fig)


def plot_diagnostics(leaderboard: pd.DataFrame, diag_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))

    ok = leaderboard[(leaderboard["status"] == "ok") & np.isfinite(leaderboard["score"])].copy()
    if ok.empty:
        ax.text(0.03, 0.65, "No successful candidate diagnostics", fontsize=11)
        ax.axis("off")
    else:
        top = ok.sort_values("score").head(min(10, len(ok)))
        ax.bar(np.arange(len(top)), top["score"].values, color="#2ca02c", alpha=0.8)
        ax.set_xticks(np.arange(len(top)))
        labels = [f"#{int(r)}" for r in top["rank"].values]
        ax.set_xticklabels(labels)
        ax.set_ylabel("selection score")
        ax.set_xlabel("top candidates")
        ax.set_title("Candidate Score (lower is better)")
        ax.grid(alpha=0.20)

    fig.tight_layout()
    fig.savefig(diag_path, dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    fast_grid = to_bool(args.fast_grid)

    raw = read_table(args.observations)
    obs, input_kind = normalize_observations(raw, args)
    vars_use = select_variables(obs, args.variables)

    if not vars_use:
        raise SystemExit("No variables left after filtering")

    out = args.output_dir
    fc_dir = out / "forecasts"
    ch_dir = out / "charts"
    cmp_dir = out / "components"
    lb_dir = out / "leaderboards"
    rep_dir = out / "reports"
    ensure_dirs([out, fc_dir, ch_dir, cmp_dir, lb_dir, rep_dir])

    print(
        f"Input: {args.observations} | kind={input_kind} | rows={len(obs)} | variables={','.join(vars_use)}"
    )

    index_rows: list[dict[str, Any]] = []

    for variable in vars_use:
        plan = choose_frequency_plan(obs, variable=variable, ok_value=args.qc_ok_value)
        raw_s = aggregate_series(obs, variable=variable, plan=plan, ok_value=args.qc_ok_value)
        if raw_s.empty:
            continue

        clean_s = winsorize_and_impute(raw_s, q_low=args.winsor_lower, q_high=args.winsor_upper)
        df_model, use_log = to_model_frame(clean_s, variable=variable)
        if df_model.empty:
            continue

        last_ds = timestamp_safe(df_model["ds"].max())
        horizon = make_horizon(last_ds, target_year=args.target_year, freq=plan.freq)

        cv = make_cv_plan(len(df_model), freq=plan.freq, season_len=plan.season_len, args=args)

        # Candidate evaluation: seasonal naive baseline + Prophet grid
        candidate_results: list[CandidateResult] = []

        baseline = evaluate_seasonal_naive(
            df_model=df_model,
            variable=variable,
            season_len=plan.season_len,
            cv=cv,
            use_log=use_log,
            bias_weight=float(args.bias_weight),
            stability_weight=float(args.stability_weight),
        )
        candidate_results.append(baseline)

        cfgs = build_prophet_grid(variable=variable, interval_width=0.8, fast_grid=fast_grid)

        for cfg in cfgs:
            cres = evaluate_prophet_candidate(
                df_model=df_model,
                variable=variable,
                freq=plan.freq,
                season_len=plan.season_len,
                cfg=cfg,
                cv=cv,
                use_log=use_log,
                bias_weight=float(args.bias_weight),
                stability_weight=float(args.stability_weight),
            )
            candidate_results.append(cres)

        lb = pd.DataFrame(
            [
                {
                    "model_label": c.model_label,
                    "cfg_json": c.cfg_json,
                    "rmse": c.rmse,
                    "mae": c.mae,
                    "smape": c.smape,
                    "bias": c.bias,
                    "rmse_std": c.rmse_std,
                    "bias_std": c.bias_std,
                    "score": c.score,
                    "n_folds": c.n_folds,
                    "status": c.status,
                    "error": c.error,
                }
                for c in candidate_results
            ]
        )

        lb = lb.sort_values(["score", "rmse", "rmse_std"], na_position="last").reset_index(drop=True)
        lb["rank"] = np.arange(1, len(lb) + 1)

        lb_csv = lb_dir / f"{variable}_prophet_ultra_leaderboard_to_{args.target_year}.csv"
        lb.to_csv(lb_csv, index=False)

        # Keep only successful rows for model selection
        lb_ok = lb[(lb["status"] == "ok") & np.isfinite(lb["score"])].copy()

        if lb_ok.empty:
            best_label = "seasonal_naive"
            best_cfg = None
            best_rmse = np.nan
            best_bias = 0.0
            resid_sigma = np.nan
        else:
            # Optional ensemble over top prophet candidates for stability.
            top = lb_ok.head(max(1, int(args.max_ensemble_models))).copy()
            if len(top) > 1:
                inv = 1.0 / top["score"].replace(0, np.nan)
                inv_sum = float(inv.sum()) if np.isfinite(inv.sum()) else np.nan
                if np.isfinite(inv_sum) and inv_sum > 0:
                    top["weight"] = inv / inv_sum
                else:
                    top["weight"] = 1.0 / len(top)
            else:
                top["weight"] = 1.0

            # Choose anchor best candidate for final single-model fit + bias correction.
            best = top.iloc[0]
            best_label = str(best["model_label"])
            best_rmse = float(best["rmse"]) if np.isfinite(best["rmse"]) else np.nan
            best_bias = float(best["bias"]) if np.isfinite(best["bias"]) else 0.0
            resid_sigma = float(best["rmse"]) if np.isfinite(best["rmse"]) else np.nan

            if best_label == "prophet":
                cfg_json = str(best["cfg_json"]) if isinstance(best["cfg_json"], str) else ""
                if cfg_json:
                    best_cfg = ProphetConfig(**json.loads(cfg_json))
                else:
                    best_cfg = None
            else:
                best_cfg = None

        # Run final model
        if best_label == "prophet" and best_cfg is not None:
            out_df, model_for_components = run_final_prophet(
                df_model=df_model,
                variable=variable,
                freq=plan.freq,
                cfg=best_cfg,
                use_log=use_log,
                horizon=horizon,
                # error = true - pred => add mean error to reduce bias
                bias_correction=float(best_bias),
                resid_sigma=float(resid_sigma),
                interval_z=float(args.interval_z),
            )
            model_strategy = "prophet_ultra_tuned"
            selected_cfg_json = json.dumps(best_cfg.__dict__, sort_keys=True)
        else:
            out_df, model_for_components = run_final_seasonal_naive(
                df_model=df_model,
                variable=variable,
                freq=plan.freq,
                season_len=plan.season_len,
                use_log=use_log,
                horizon=horizon,
                bias_correction=float(best_bias),
                resid_sigma=float(resid_sigma),
                interval_z=float(args.interval_z),
            )
            model_strategy = "seasonal_naive_fallback"
            selected_cfg_json = ""

        out_df = out_df.sort_values("ds").reset_index(drop=True)
        out_df["is_forecast"] = out_df["ds"] > last_ds
        out_df["variable"] = variable
        out_df["unit"] = infer_unit(variable)
        out_df["frequency"] = plan.freq
        out_df["model_strategy"] = model_strategy
        out_df["use_log_transform"] = use_log

        freq_tag = "yearly" if plan.freq == "YS" else "monthly"

        fc_csv = fc_dir / f"{variable}_{freq_tag}_prophet_ultra_to_{args.target_year}.csv"
        fc_pq = fc_dir / f"{variable}_{freq_tag}_prophet_ultra_to_{args.target_year}.parquet"
        out_df.to_csv(fc_csv, index=False)
        out_df.to_parquet(fc_pq, index=False)

        chart_png = ch_dir / f"{variable}_{freq_tag}_prophet_ultra_to_{args.target_year}.png"
        comp_png = cmp_dir / f"{variable}_{freq_tag}_prophet_ultra_components_to_{args.target_year}.png"
        diag_png = ch_dir / f"{variable}_{freq_tag}_prophet_ultra_diagnostics_to_{args.target_year}.png"

        plot_forecast(
            out_df=out_df,
            last_ds=last_ds,
            variable=variable,
            freq_label=plan.label,
            unit=infer_unit(variable),
            chart_path=chart_png,
        )
        plot_components_safe(model=model_for_components, out_df=out_df, comp_path=comp_png)
        plot_diagnostics(leaderboard=lb, diag_path=diag_png)

        top_rows = lb.head(max(1, int(args.report_top))).copy()
        rep = {
            "variable": variable,
            "frequency": plan.freq,
            "frequency_label": plan.label,
            "monthly_coverage": float(plan.monthly_coverage),
            "last_observation": str(last_ds),
            "target_year": int(args.target_year),
            "horizon_steps": int(horizon),
            "cv_holdout_steps": int(cv.holdout),
            "cv_splits": int(cv.splits),
            "cv_min_train_steps": int(cv.min_train),
            "model_strategy": model_strategy,
            "best_label": best_label,
            "best_cfg_json": selected_cfg_json,
            "best_cv_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
            "bias_correction_applied": float(best_bias),
            "residual_sigma": float(resid_sigma) if np.isfinite(resid_sigma) else None,
            "leaderboard_csv": str(lb_csv),
            "forecast_csv": str(fc_csv),
            "forecast_parquet": str(fc_pq),
            "chart_png": str(chart_png),
            "components_png": str(comp_png),
            "diagnostics_png": str(diag_png),
            "top_candidates": top_rows.to_dict(orient="records"),
        }

        rep_json = rep_dir / f"{variable}_{freq_tag}_prophet_ultra_report_to_{args.target_year}.json"
        rep_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

        index_rows.append(
            {
                "variable": variable,
                "frequency": plan.freq,
                "frequency_label": plan.label,
                "monthly_coverage": plan.monthly_coverage,
                "last_observation": str(last_ds),
                "target_year": args.target_year,
                "horizon_steps": horizon,
                "cv_holdout_steps": cv.holdout,
                "cv_splits": cv.splits,
                "cv_min_train_steps": cv.min_train,
                "model_strategy": model_strategy,
                "best_model_label": best_label,
                "best_cv_rmse": best_rmse,
                "bias_correction": float(best_bias),
                "selected_cfg_json": selected_cfg_json,
                "forecast_csv": str(fc_csv),
                "forecast_parquet": str(fc_pq),
                "chart_png": str(chart_png),
                "components_png": str(comp_png),
                "diagnostics_png": str(diag_png),
                "leaderboard_csv": str(lb_csv),
                "report_json": str(rep_json),
                "input_kind": input_kind,
                "input_path": str(args.observations),
            }
        )

    idx = pd.DataFrame(index_rows).sort_values("variable") if index_rows else pd.DataFrame()
    idx_csv = out / f"prophet_ultra_index_to_{args.target_year}.csv"
    idx_pq = out / f"prophet_ultra_index_to_{args.target_year}.parquet"
    idx.to_csv(idx_csv, index=False)
    idx.to_parquet(idx_pq, index=False)

    print("Ultra Prophet pipeline completed.")
    print(f"Index: {idx_csv}")
    if not idx.empty:
        cols = ["variable", "frequency", "model_strategy", "best_cv_rmse", "forecast_csv"]
        print(idx[cols].to_string(index=False))


if __name__ == "__main__":
    main()
