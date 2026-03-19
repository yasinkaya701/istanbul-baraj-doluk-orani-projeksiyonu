#!/usr/bin/env python3
"""Strong and consistent climate forecasting pipeline.

Hybrid ensemble over multiple model families:
- Prophet
- ETS (Exponential Smoothing)
- SARIMA
- Lag-feature Ridge regression
- Seasonal Naive baseline

The pipeline selects robust models with rolling time-series CV,
then builds a weighted ensemble using both error and stability.
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

# Avoid cache permission warnings in restricted runtimes.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "strong_model_cache"
_MPL_CACHE = _CACHE_ROOT / "mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

UNIT_MAP = {"humidity": "%", "temp": "C", "pressure": "hPa", "precip": "mm"}

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
class ProphetConfig:
    seasonality_mode: str
    changepoint_prior_scale: float
    seasonality_prior_scale: float
    changepoint_range: float
    interval_width: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train strong and consistent hybrid ensemble for climate forecasting")
    p.add_argument(
        "--observations",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/forecast_package/observations_with_graph.parquet"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/strong_ensemble_package"),
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
    p.add_argument("--backtest-splits", type=int, default=4)
    p.add_argument("--min-train-steps", type=int, default=36)
    p.add_argument("--max-ensemble-models", type=int, default=3)
    p.add_argument("--interval-z", type=float, default=1.96)
    p.add_argument("--winsor-quantile", type=float, default=0.995)
    p.add_argument("--print-top", type=int, default=6)
    return p.parse_args()


def normalize_token(text: object) -> str:
    s = str(text).strip().lower().translate(_TR_CHARMAP)
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def canonical_variable_name(text: object) -> str:
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


def to_bool_series_ok(x: pd.Series, ok_value: str) -> pd.Series:
    return x.astype(str).str.lower().eq(str(ok_value).lower())


def apply_bounds(arr: np.ndarray, variable: str) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if is_humidity(variable):
        return np.clip(x, 0, 100)
    if is_precip(variable) or is_pressure(variable):
        return np.clip(x, 0, None)
    return x


def invert_transform(y: np.ndarray, use_log: bool) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.expm1(y) if use_log else y


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv"}:
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
        raise SystemExit("Cannot detect time/value columns. Provide --timestamp-col and --value-col.")

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
        out["qc_flag"] = raw[qc_col]
    else:
        out["qc_flag"] = args.qc_ok_value

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["variable"] = out["variable"].astype(str).map(canonical_variable_name)
    out["qc_flag"] = out["qc_flag"].astype(str)
    out = out.dropna(subset=["timestamp", "value", "variable"]).sort_values("timestamp").reset_index(drop=True)
    if out.empty:
        raise SystemExit("No usable rows after parsing")
    return out, input_kind


def choose_base_series(obs: pd.DataFrame, variable: str, ok_value: str) -> tuple[pd.Series, str, int, float]:
    sub = obs[obs["variable"] == variable].copy()
    if sub.empty:
        return pd.Series(dtype=float), "MS", 12, 0.0

    ok_mask = to_bool_series_ok(sub["qc_flag"], ok_value)
    if ok_mask.any():
        sub = sub[ok_mask]
    if sub.empty:
        return pd.Series(dtype=float), "MS", 12, 0.0

    sub = sub.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if sub.empty:
        return pd.Series(dtype=float), "MS", 12, 0.0

    raw = sub.groupby("timestamp")["value"].mean()

    # Monthly raw coverage check
    obs_m = raw.resample("MS").count()
    observed_months = int((obs_m > 0).sum())
    full_months = int(len(obs_m)) if len(obs_m) else 1
    monthly_coverage = observed_months / max(1, full_months)

    # If monthly coverage is sparse but yearly has enough history, switch to yearly.
    obs_y = raw.resample("YS").count()
    observed_years = int((obs_y > 0).sum())

    use_yearly = monthly_coverage < 0.45 and observed_years >= 15

    if use_yearly:
        if is_precip(variable):
            s = raw.resample("YS").sum(min_count=1)
            s = s.fillna(0.0)
        else:
            s = raw.resample("YS").mean()
            s = s.interpolate("time").ffill().bfill()
        return s.astype(float), "YS", 1, monthly_coverage

    if is_precip(variable):
        s = raw.resample("MS").sum(min_count=1)
        s = s.fillna(0.0)
    else:
        s = raw.resample("MS").mean()
        s = s.interpolate("time").ffill().bfill()
    return s.astype(float), "MS", 12, monthly_coverage


def preprocess_series(series: pd.Series, variable: str, winsor_q: float) -> tuple[pd.DataFrame, bool]:
    s = series.copy().astype(float)
    q = float(np.clip(winsor_q, 0.90, 1.0))
    hi = float(s.quantile(q)) if len(s) else np.nan
    if np.isfinite(hi):
        s = s.clip(upper=hi)

    use_log = is_precip(variable)
    if use_log:
        y = np.log1p(np.clip(s.values, 0, None))
    else:
        y = s.values.astype(float)

    out = pd.DataFrame({"ds": s.index, "y": y})
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out = out.dropna().sort_values("ds").reset_index(drop=True)
    return out, use_log


def split_points(n: int, holdout: int, splits: int, min_train: int) -> list[int]:
    out = []
    for i in range(splits, 0, -1):
        cut = n - i * holdout
        if min_train <= cut < n:
            out.append(cut)
    return sorted(set(out))


def metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = np.zeros_like(denom, dtype=float)
    mask = denom > 1e-9
    ratio[mask] = np.abs(err[mask]) / denom[mask]
    smape = float(np.mean(ratio) * 100.0)
    return {"mae": mae, "rmse": rmse, "smape": smape}


def make_future_index(last_ds: pd.Timestamp, periods: int, freq: str) -> pd.DatetimeIndex:
    if periods <= 0:
        return pd.DatetimeIndex([])
    if freq == "YS":
        return pd.date_range(last_ds + pd.offsets.YearBegin(1), periods=periods, freq="YS")
    return pd.date_range(last_ds + pd.offsets.MonthBegin(1), periods=periods, freq="MS")


def prophet_configs(variable: str) -> list[ProphetConfig]:
    if is_precip(variable):
        return [
            ProphetConfig("multiplicative", 0.10, 10.0, 0.90, 0.80),
            ProphetConfig("multiplicative", 0.25, 20.0, 0.90, 0.80),
            ProphetConfig("additive", 0.10, 10.0, 0.90, 0.80),
        ]
    return [
        ProphetConfig("additive", 0.03, 10.0, 0.85, 0.80),
        ProphetConfig("additive", 0.10, 20.0, 0.90, 0.80),
    ]


def predict_prophet(train_df: pd.DataFrame, periods: int, freq: str, cfg: ProphetConfig) -> np.ndarray:
    yearly = freq == "MS"
    m = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=cfg.seasonality_mode,
        interval_width=cfg.interval_width,
        changepoint_prior_scale=cfg.changepoint_prior_scale,
        seasonality_prior_scale=cfg.seasonality_prior_scale,
        changepoint_range=cfg.changepoint_range,
    )
    if freq == "MS":
        m.add_seasonality(name="quarterly", period=365.25 / 4.0, fourier_order=5)

    m.fit(train_df)
    fut = m.make_future_dataframe(periods=periods, freq=freq)
    pred = m.predict(fut).tail(periods)
    return pred["yhat"].values.astype(float)


def predict_seasonal_naive(train_df: pd.DataFrame, periods: int, season_len: int) -> np.ndarray:
    y = train_df["y"].values.astype(float)
    if len(y) == 0 or periods <= 0:
        return np.array([], dtype=float)

    if season_len <= 1 or len(y) < season_len:
        base = float(np.mean(y[-min(len(y), 3) :]))
        return np.repeat(base, periods)

    season = y[-season_len:]
    reps = int(np.ceil(periods / season_len))
    return np.tile(season, reps)[:periods].astype(float)


def _fit_ets(y: np.ndarray, season_len: int, seasonal_mode: str | None, damped: bool) -> ExponentialSmoothing:
    if seasonal_mode is None or season_len <= 1 or len(y) < (2 * season_len + 2):
        model = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=damped,
            seasonal=None,
            initialization_method="estimated",
        )
    else:
        model = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=damped,
            seasonal=seasonal_mode,
            seasonal_periods=season_len,
            initialization_method="estimated",
        )
    return model.fit(optimized=True, use_brute=False)


def predict_ets(train_df: pd.DataFrame, periods: int, season_len: int, mode: str) -> np.ndarray:
    y = train_df["y"].values.astype(float)
    if len(y) < 8:
        raise ValueError("ETS requires more data")

    if mode == "add":
        fit = _fit_ets(y, season_len=season_len, seasonal_mode="add", damped=False)
    elif mode == "add_damped":
        fit = _fit_ets(y, season_len=season_len, seasonal_mode="add", damped=True)
    elif mode == "mul":
        if np.min(y) <= 0:
            raise ValueError("mul ETS requires positive series")
        fit = _fit_ets(y, season_len=season_len, seasonal_mode="mul", damped=False)
    else:
        raise ValueError(f"unknown ETS mode {mode}")

    return fit.forecast(periods).astype(float)


def sarima_candidates(season_len: int) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    if season_len <= 1:
        return [
            ((1, 1, 1), (0, 0, 0, 0)),
            ((2, 1, 0), (0, 0, 0, 0)),
            ((0, 1, 1), (0, 0, 0, 0)),
            ((1, 0, 1), (0, 0, 0, 0)),
        ]

    return [
        ((1, 0, 1), (0, 1, 1, season_len)),
        ((1, 1, 1), (0, 1, 1, season_len)),
        ((2, 1, 2), (1, 0, 1, season_len)),
        ((0, 1, 1), (0, 1, 1, season_len)),
    ]


def predict_sarima(train_df: pd.DataFrame, periods: int, season_len: int) -> np.ndarray:
    y = train_df["y"].values.astype(float)
    if len(y) < 12:
        raise ValueError("SARIMA requires more data")

    best_aic = np.inf
    best_res = None
    for order, sorder in sarima_candidates(season_len):
        try:
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=sorder,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            aic = float(res.aic)
            if np.isfinite(aic) and aic < best_aic:
                best_aic = aic
                best_res = res
        except Exception:
            continue

    if best_res is None:
        raise ValueError("SARIMA fit failed")

    f = best_res.get_forecast(periods)
    return np.asarray(f.predicted_mean, dtype=float)


def lag_indices(season_len: int) -> list[int]:
    base = [1, 2, 3, 6]
    if season_len > 1:
        base.append(season_len)
    out = sorted(set(x for x in base if x > 0))
    return out


def build_lagged_xy(train_df: pd.DataFrame, season_len: int) -> tuple[np.ndarray, np.ndarray]:
    y = train_df["y"].values.astype(float)
    ds = pd.to_datetime(train_df["ds"])
    lags = lag_indices(season_len)
    max_lag = max(lags)

    X = []
    target = []
    for i in range(max_lag, len(y)):
        lag_vals = [y[i - l] for l in lags]
        roll3 = float(np.mean(y[max(0, i - 3) : i]))
        if ds.iloc[i].month:
            m = ds.iloc[i].month
            sin_m = math.sin(2 * math.pi * m / 12.0)
            cos_m = math.cos(2 * math.pi * m / 12.0)
        else:
            sin_m = 0.0
            cos_m = 1.0
        X.append(lag_vals + [roll3, sin_m, cos_m])
        target.append(y[i])

    if not X:
        return np.zeros((0, len(lags) + 3)), np.array([])
    return np.asarray(X, dtype=float), np.asarray(target, dtype=float)


def predict_lag_ridge(train_df: pd.DataFrame, periods: int, season_len: int, freq: str) -> np.ndarray:
    X, y = build_lagged_xy(train_df, season_len=season_len)
    if len(y) < 24:
        raise ValueError("lag ridge needs more lagged samples")

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y)

    hist_y = train_df["y"].values.astype(float).tolist()
    lags = lag_indices(season_len)
    future_idx = make_future_index(pd.Timestamp(train_df["ds"].iloc[-1]), periods, freq)
    preds = []
    for ds in future_idx:
        lag_vals = [hist_y[-l] for l in lags]
        roll3 = float(np.mean(hist_y[-3:])) if len(hist_y) >= 3 else float(np.mean(hist_y))
        m = ds.month
        sin_m = math.sin(2 * math.pi * m / 12.0)
        cos_m = math.cos(2 * math.pi * m / 12.0)
        feat = np.asarray(lag_vals + [roll3, sin_m, cos_m], dtype=float).reshape(1, -1)
        pred = float(model.predict(feat)[0])
        preds.append(pred)
        hist_y.append(pred)

    return np.asarray(preds, dtype=float)


def predict_by_model(
    model_id: str,
    train_df: pd.DataFrame,
    periods: int,
    freq: str,
    season_len: int,
    prophet_cfg: ProphetConfig | None,
) -> np.ndarray:
    if periods <= 0:
        return np.array([], dtype=float)

    if model_id == "seasonal_naive":
        return predict_seasonal_naive(train_df, periods, season_len)
    if model_id == "prophet":
        if prophet_cfg is None:
            raise ValueError("prophet cfg missing")
        return predict_prophet(train_df, periods, freq, prophet_cfg)
    if model_id == "ets_add":
        return predict_ets(train_df, periods, season_len, mode="add")
    if model_id == "ets_add_damped":
        return predict_ets(train_df, periods, season_len, mode="add_damped")
    if model_id == "ets_mul":
        return predict_ets(train_df, periods, season_len, mode="mul")
    if model_id == "sarima":
        return predict_sarima(train_df, periods, season_len)
    if model_id == "lag_ridge":
        return predict_lag_ridge(train_df, periods, season_len, freq)
    raise ValueError(f"unknown model_id: {model_id}")


def cv_scores_for_model(
    model_id: str,
    df_model: pd.DataFrame,
    variable: str,
    freq: str,
    season_len: int,
    holdout: int,
    splits: int,
    min_train: int,
    use_log: bool,
    prophet_cfg: ProphetConfig | None,
) -> dict[str, object]:
    cuts = split_points(len(df_model), holdout=holdout, splits=splits, min_train=min_train)
    if not cuts:
        return {
            "model_id": model_id,
            "mae": np.nan,
            "rmse": np.nan,
            "smape": np.nan,
            "rmse_std": np.nan,
            "n_folds": 0,
            "status": "no_folds",
            "error": "insufficient history for requested CV",
        }

    mae_list = []
    rmse_list = []
    smape_list = []

    for cut in cuts:
        train = df_model.iloc[:cut].copy()
        test = df_model.iloc[cut : cut + holdout].copy()
        if len(test) == 0:
            continue

        try:
            pred_t = predict_by_model(
                model_id=model_id,
                train_df=train,
                periods=len(test),
                freq=freq,
                season_len=season_len,
                prophet_cfg=prophet_cfg,
            )
            y_true = apply_bounds(invert_transform(test["y"].values, use_log=use_log), variable)
            y_pred = apply_bounds(invert_transform(pred_t, use_log=use_log), variable)
            m = metric_pack(y_true, y_pred)
            mae_list.append(m["mae"])
            rmse_list.append(m["rmse"])
            smape_list.append(m["smape"])
        except Exception:
            continue

    if not rmse_list:
        return {
            "model_id": model_id,
            "mae": np.nan,
            "rmse": np.nan,
            "smape": np.nan,
            "rmse_std": np.nan,
            "n_folds": 0,
            "status": "failed",
            "error": "all folds failed",
        }

    return {
        "model_id": model_id,
        "mae": float(np.mean(mae_list)),
        "rmse": float(np.mean(rmse_list)),
        "smape": float(np.mean(smape_list)),
        "rmse_std": float(np.std(rmse_list)),
        "n_folds": int(len(rmse_list)),
        "status": "ok",
        "error": "",
    }


def ensemble_weights(lb_ok: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if lb_ok.empty:
        return lb_ok

    top = lb_ok.sort_values(["rmse", "rmse_std", "mae"]).head(max(1, top_k)).copy()
    # Stability-aware score: lower rmse and lower rmse_std => higher score.
    top["stability_penalty"] = 1.0 + (top["rmse_std"] / top["rmse"].replace(0, np.nan)).fillna(0.0)
    top["inv_score"] = 1.0 / (top["rmse"] * top["stability_penalty"]).replace(0, np.nan)
    s = top["inv_score"].sum()
    if not np.isfinite(s) or s <= 0:
        top["weight"] = 1.0 / len(top)
    else:
        top["weight"] = top["inv_score"] / s
    return top


def requested_variables(obs: pd.DataFrame, variables_arg: str) -> list[str]:
    available = sorted(obs["variable"].dropna().unique().tolist())
    if not variables_arg or variables_arg.strip() in {"*", "all", "ALL"}:
        return available
    req = [canonical_variable_name(v.strip()) for v in variables_arg.split(",") if v.strip()]
    req = [v for v in req if v in available]
    return sorted(set(req))


def candidate_model_rows(variable: str) -> list[dict[str, object]]:
    rows = [
        {"model_id": "seasonal_naive", "prophet_cfg": None, "model_label": "seasonal_naive"},
        {"model_id": "ets_add", "prophet_cfg": None, "model_label": "ets_add"},
        {"model_id": "ets_add_damped", "prophet_cfg": None, "model_label": "ets_add_damped"},
        {"model_id": "sarima", "prophet_cfg": None, "model_label": "sarima"},
        {"model_id": "lag_ridge", "prophet_cfg": None, "model_label": "lag_ridge"},
    ]

    if is_precip(variable):
        rows.append({"model_id": "ets_mul", "prophet_cfg": None, "model_label": "ets_mul"})

    for i, cfg in enumerate(prophet_configs(variable), start=1):
        rows.append({"model_id": "prophet", "prophet_cfg": cfg, "model_label": f"prophet_cfg_{i}"})

    return rows


def make_history_frame(df_model: pd.DataFrame, variable: str, use_log: bool) -> pd.DataFrame:
    y_hist = apply_bounds(invert_transform(df_model["y"].values, use_log=use_log), variable)
    out = pd.DataFrame(
        {
            "ds": pd.to_datetime(df_model["ds"].values),
            "yhat": y_hist,
            "yhat_lower": np.nan,
            "yhat_upper": np.nan,
            "is_forecast": False,
        }
    )
    return out


def combine_forecasts(
    pred_map: dict[str, np.ndarray],
    weight_map: dict[str, float],
    residual_sigma_map: dict[str, float],
    interval_z: float,
    variable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_ids = [m for m in pred_map if m in weight_map]
    if not model_ids:
        return np.array([]), np.array([]), np.array([])

    w = np.array([weight_map[m] for m in model_ids], dtype=float)
    p = np.vstack([pred_map[m] for m in model_ids])
    yhat = np.sum(w[:, None] * p, axis=0)

    # Uncertainty from weighted residual sigma with slight horizon growth.
    base_sigma = float(np.sqrt(np.sum((w * np.array([residual_sigma_map.get(m, np.nan) for m in model_ids])) ** 2)))
    if not np.isfinite(base_sigma) or base_sigma <= 0:
        base_sigma = float(np.nanstd(yhat) * 0.1)

    horizon = np.arange(1, len(yhat) + 1, dtype=float)
    growth = np.sqrt(1.0 + horizon / max(12.0, float(len(horizon))))
    sigma = base_sigma * growth

    lower = yhat - interval_z * sigma
    upper = yhat + interval_z * sigma

    yhat = apply_bounds(yhat, variable)
    lower = apply_bounds(lower, variable)
    upper = apply_bounds(upper, variable)
    return yhat, lower, upper


def main() -> None:
    args = parse_args()

    raw = read_table(args.observations)
    obs, detected_kind = normalize_observations(raw, args)
    vars_use = requested_variables(obs, args.variables)
    if not vars_use:
        raise SystemExit("No variables selected after filter")

    out = args.output_dir
    fc_dir = out / "forecasts"
    ch_dir = out / "charts"
    lb_dir = out / "leaderboards"
    rep_dir = out / "reports"
    for d in [out, fc_dir, ch_dir, lb_dir, rep_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(
        f"Input: {args.observations} | kind={detected_kind} | rows={len(obs)} | variables={','.join(vars_use)}"
    )

    index_rows: list[dict[str, object]] = []

    for variable in vars_use:
        series, freq, season_len, monthly_coverage = choose_base_series(obs, variable, ok_value=args.qc_ok_value)
        if series.empty:
            continue

        df_model, use_log = preprocess_series(series, variable=variable, winsor_q=args.winsor_quantile)
        if df_model.empty:
            continue

        last_ds = pd.Timestamp(df_model["ds"].max())
        target_ds = pd.Timestamp(year=args.target_year, month=12, day=1)
        if freq == "YS":
            target_ds = pd.Timestamp(year=args.target_year, month=1, day=1)
            horizon = max(0, args.target_year - last_ds.year)
        else:
            horizon = int((target_ds.year - last_ds.year) * 12 + (target_ds.month - last_ds.month))
            horizon = max(0, horizon)

        n_steps = len(df_model)
        if freq == "YS":
            holdout_eff = min(max(1, int(args.holdout_steps)), max(1, n_steps // 5))
            holdout_eff = min(holdout_eff, 3)
            min_train_floor = max(8, season_len + 5)
        else:
            holdout_eff = min(max(1, int(args.holdout_steps)), max(1, n_steps // 4))
            min_train_floor = max(24, season_len * 2)

        max_min_train = max(2, n_steps - holdout_eff)
        min_train_eff = min(max_min_train, max(min_train_floor, int(round(n_steps * 0.60))))
        if min_train_eff >= n_steps:
            min_train_eff = max(2, n_steps - holdout_eff)

        splits_eff = max(1, int(args.backtest_splits))
        while splits_eff > 1 and not split_points(n_steps, holdout_eff, splits_eff, min_train_eff):
            splits_eff -= 1
        if not split_points(n_steps, holdout_eff, splits_eff, min_train_eff):
            splits_eff = 1

        candidates = candidate_model_rows(variable)
        lb_rows = []
        for c in candidates:
            score = cv_scores_for_model(
                model_id=c["model_id"],
                df_model=df_model,
                variable=variable,
                freq=freq,
                season_len=season_len,
                holdout=holdout_eff,
                splits=splits_eff,
                min_train=min_train_eff,
                use_log=use_log,
                prophet_cfg=c["prophet_cfg"],
            )
            row = {
                "variable": variable,
                "frequency": freq,
                "monthly_coverage": monthly_coverage,
                "model_label": c["model_label"],
                "model_id": c["model_id"],
                "prophet_cfg": json.dumps(c["prophet_cfg"].__dict__) if c["prophet_cfg"] is not None else "",
            }
            row.update(score)
            lb_rows.append(row)

        lb = pd.DataFrame(lb_rows)
        lb = lb.sort_values(["rmse", "rmse_std", "mae"], na_position="last").reset_index(drop=True)
        lb["rank"] = np.arange(1, len(lb) + 1)
        lb_csv = lb_dir / f"{variable}_leaderboard_to_{args.target_year}.csv"
        lb.to_csv(lb_csv, index=False)

        lb_ok = lb[(lb["status"] == "ok") & np.isfinite(lb["rmse"])].copy()
        if lb_ok.empty:
            # Hard fallback if all advanced models fail.
            selected = pd.DataFrame(
                [{"model_label": "seasonal_naive", "model_id": "seasonal_naive", "weight": 1.0, "rmse": np.nan, "rmse_std": np.nan}]
            )
        else:
            selected = ensemble_weights(lb_ok, top_k=int(args.max_ensemble_models))

        weight_map = dict(zip(selected["model_label"], selected["weight"]))

        # Train selected models on full data and forecast.
        pred_map = {}
        resid_sigma = {}
        used_models = []
        for _, r in selected.iterrows():
            model_label = str(r["model_label"])
            model_id = str(r["model_id"])
            if model_label.startswith("prophet_cfg_"):
                cfg_json = lb_ok[lb_ok["model_label"] == model_label]["prophet_cfg"].head(1)
                if len(cfg_json) == 0 or not cfg_json.iloc[0]:
                    continue
                cfg_dict = json.loads(cfg_json.iloc[0])
                cfg = ProphetConfig(**cfg_dict)
            else:
                cfg = None

            try:
                yhat_t = predict_by_model(
                    model_id=model_id,
                    train_df=df_model,
                    periods=horizon,
                    freq=freq,
                    season_len=season_len,
                    prophet_cfg=cfg,
                )
                yhat = apply_bounds(invert_transform(yhat_t, use_log=use_log), variable)
                pred_map[model_label] = yhat
                rmse_val = lb_ok[lb_ok["model_label"] == model_label]["rmse"].head(1)
                if len(rmse_val) == 1 and np.isfinite(rmse_val.iloc[0]):
                    resid_sigma[model_label] = float(rmse_val.iloc[0])
                else:
                    resid_sigma[model_label] = float(np.nanstd(yhat)) if len(yhat) else np.nan
                used_models.append(model_label)
            except Exception:
                continue

        if not pred_map:
            # final fallback
            yhat_t = predict_seasonal_naive(df_model, periods=horizon, season_len=season_len)
            yhat = apply_bounds(invert_transform(yhat_t, use_log=use_log), variable)
            pred_map = {"seasonal_naive": yhat}
            weight_map = {"seasonal_naive": 1.0}
            resid_sigma = {"seasonal_naive": float(np.nanstd(yhat)) if len(yhat) else np.nan}
            used_models = ["seasonal_naive"]

        y_ens, y_lo, y_hi = combine_forecasts(
            pred_map=pred_map,
            weight_map=weight_map,
            residual_sigma_map=resid_sigma,
            interval_z=float(args.interval_z),
            variable=variable,
        )

        hist = make_history_frame(df_model, variable=variable, use_log=use_log)
        f_idx = make_future_index(last_ds, periods=horizon, freq=freq)
        fc = pd.DataFrame(
            {
                "ds": f_idx,
                "yhat": y_ens,
                "yhat_lower": y_lo,
                "yhat_upper": y_hi,
                "is_forecast": True,
            }
        )

        out_df = pd.concat([hist, fc], ignore_index=True)
        out_df["variable"] = variable
        out_df["unit"] = infer_unit(variable)
        out_df["frequency"] = freq
        out_df["selected_models"] = ",".join(used_models)

        fc_csv = fc_dir / f"{variable}_{'yearly' if freq=='YS' else 'monthly'}_strong_ensemble_to_{args.target_year}.csv"
        fc_pq = fc_dir / f"{variable}_{'yearly' if freq=='YS' else 'monthly'}_strong_ensemble_to_{args.target_year}.parquet"
        out_df.to_csv(fc_csv, index=False)
        out_df.to_parquet(fc_pq, index=False)

        # Chart
        fig, ax = plt.subplots(figsize=(12, 4.8))
        h = out_df[out_df["is_forecast"] == False]
        f = out_df[out_df["is_forecast"] == True]
        ax.plot(h["ds"], h["yhat"], color="#1f77b4", linewidth=1.4, label="historical")
        if not f.empty:
            ax.plot(f["ds"], f["yhat"], color="#d62728", linewidth=2.0, label="strong ensemble")
            if f["yhat_lower"].notna().any() and f["yhat_upper"].notna().any():
                ax.fill_between(f["ds"], f["yhat_lower"], f["yhat_upper"], color="#d62728", alpha=0.15, label="uncertainty")
        ax.axvline(last_ds, color="#555555", linestyle="--", linewidth=1.0)
        ax.set_title(f"Strong Consistent Forecast - {variable} ({'yearly' if freq=='YS' else 'monthly'})")
        ax.set_xlabel("date")
        ax.set_ylabel(f"value ({infer_unit(variable)})")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        chart_png = ch_dir / f"{variable}_{'yearly' if freq=='YS' else 'monthly'}_strong_ensemble_to_{args.target_year}.png"
        fig.savefig(chart_png, dpi=170)
        plt.close(fig)

        top_rows = lb.head(max(1, int(args.print_top))).copy()
        rep = {
            "variable": variable,
            "frequency": freq,
            "monthly_coverage": float(monthly_coverage),
            "last_observation": str(last_ds),
            "horizon_steps": int(horizon),
            "cv_holdout_steps": int(holdout_eff),
            "cv_splits": int(splits_eff),
            "cv_min_train_steps": int(min_train_eff),
            "selected_models": used_models,
            "weights": {k: float(v) for k, v in weight_map.items()},
            "top_leaderboard": top_rows.to_dict(orient="records"),
            "leaderboard_csv": str(lb_csv),
            "forecast_csv": str(fc_csv),
            "chart_png": str(chart_png),
        }
        rep_json = rep_dir / f"{variable}_report_to_{args.target_year}.json"
        rep_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

        best_rmse = float(lb_ok["rmse"].min()) if not lb_ok.empty else np.nan
        index_rows.append(
            {
                "variable": variable,
                "frequency": freq,
                "monthly_coverage": monthly_coverage,
                "last_observation": str(last_ds),
                "target_year": args.target_year,
                "horizon_steps": horizon,
                "cv_holdout_steps": holdout_eff,
                "cv_splits": splits_eff,
                "cv_min_train_steps": min_train_eff,
                "best_cv_rmse": best_rmse,
                "ensemble_models": ",".join(used_models),
                "ensemble_weights_json": json.dumps({k: float(v) for k, v in weight_map.items()}),
                "forecast_csv": str(fc_csv),
                "forecast_parquet": str(fc_pq),
                "leaderboard_csv": str(lb_csv),
                "chart_png": str(chart_png),
                "report_json": str(rep_json),
                "input_kind": detected_kind,
                "input_path": str(args.observations),
            }
        )

    idx = pd.DataFrame(index_rows).sort_values("variable") if index_rows else pd.DataFrame()
    idx_csv = out / f"strong_ensemble_index_to_{args.target_year}.csv"
    idx_pq = out / f"strong_ensemble_index_to_{args.target_year}.parquet"
    idx.to_csv(idx_csv, index=False)
    idx.to_parquet(idx_pq, index=False)

    print("Strong ensemble pipeline completed.")
    print(f"Index: {idx_csv}")
    if not idx.empty:
        print(idx[["variable", "frequency", "best_cv_rmse", "ensemble_models"]].to_string(index=False))


if __name__ == "__main__":
    main()
