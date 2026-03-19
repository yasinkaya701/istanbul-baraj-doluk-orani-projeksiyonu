#!/usr/bin/env python3
"""Ultra monthly forecasting pipeline (finance-style + model zoo).

Highlights:
- Candidate model zoo:
  - ta_ridge, ta_robust (technical-indicator regressors)
  - theta, ets_add, ets_damped
  - sarima_111x111, sarima_210x110, sarima_011x011
  - seasonal_naive
- Rolling-origin backtest and non-negative stacking (NNLS)
- Conformal-style calibrated prediction bands (step-wise)
- Bridge to split year (e.g. 2020), then forecast to target year (e.g. 2025)
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import nnls
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

VARIABLES = ["humidity", "temp", "pressure", "precip"]
UNITS = {"humidity": "%", "temp": "C", "pressure": "unknown", "precip": "mm"}
LAGS = [1, 2, 3, 6, 12, 24]


@dataclass
class CandidateEval:
    name: str
    mae: float
    rmse: float
    preds: np.ndarray
    y_true: np.ndarray
    step_idx: np.ndarray


@dataclass
class CandidateFitted:
    name: str
    model: Any
    spec: dict[str, Any]


@dataclass
class EnsemblePack:
    candidates: list[CandidateFitted]
    weights: np.ndarray
    cv_mae: float
    residuals: np.ndarray
    residuals_by_step: dict[int, np.ndarray]
    leaderboard: pd.DataFrame


@dataclass
class TARidgeModel:
    cols: list[str]
    mu: np.ndarray
    sigma: np.ndarray
    beta: np.ndarray


@dataclass
class TARobustModel:
    cols: list[str]
    mu: np.ndarray
    sigma: np.ndarray
    beta: np.ndarray


@dataclass
class PatternKNNModel:
    window: int
    k: int
    X: np.ndarray
    next_delta: np.ndarray
    next_month: np.ndarray
    delta_scale: float


@dataclass
class PatternAnalogModel:
    window: int
    top_k: int


class Transform:
    def forward(self, s: pd.Series) -> pd.Series:
        raise NotImplementedError

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentityTransform(Transform):
    def forward(self, s: pd.Series) -> pd.Series:
        return s.astype(float)

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=float)


class Log1pTransform(Transform):
    def forward(self, s: pd.Series) -> pd.Series:
        x = np.asarray(s.values, dtype=float)
        x = np.clip(x, 0, None)
        return pd.Series(np.log1p(x), index=s.index)

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return np.expm1(np.asarray(arr, dtype=float))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultra climate monthly model-zoo forecast.")
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/forecast_package/observations_with_graph.parquet"),
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/forecast_package/charts"),
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/forecast_package/strong_model_datasets"),
    )
    parser.add_argument("--split-year", type=int, default=2020)
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument("--horizon-months", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--max-models", type=int, default=4)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--cv-horizon", type=int, default=6)
    parser.add_argument(
        "--analog-alpha",
        type=float,
        default=0.20,
        help="Blend ratio from historical analog-outcome path into final forecast (0.0-1.0).",
    )
    parser.add_argument(
        "--force-models",
        type=str,
        default="",
        help="Comma-separated model names to blend (e.g. sarima_auto_1,pattern_knn12).",
    )
    parser.add_argument(
        "--irregularity",
        type=float,
        default=0.0,
        help="Adds short-term irregularity (0.0-1.0). Keep 0.0 for max forecast accuracy.",
    )
    return parser.parse_args()


def monthly_series(obs: pd.DataFrame, variable: str) -> tuple[pd.Series, pd.Timestamp]:
    sub = obs[(obs["variable"] == variable) & (obs["qc_flag"] == "ok")].copy()
    if sub.empty:
        return pd.Series(dtype=float), pd.Timestamp("1970-01-01")

    sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["timestamp", "value"]).sort_values("timestamp")

    raw = sub.groupby("timestamp")["value"].mean()
    if variable == "precip":
        m = raw.resample("MS").sum(min_count=1)
    else:
        m = raw.resample("MS").mean()

    observed = m.notna()
    last_obs = observed[observed].index.max()

    if variable == "precip":
        m = m.fillna(0.0)
    else:
        m = m.interpolate("time").ffill().bfill()

    return m.astype(float), pd.Timestamp(last_obs)


def pick_transform(variable: str) -> Transform:
    if variable == "precip":
        return Log1pTransform()
    return IdentityTransform()


def _mean(arr: np.ndarray) -> float:
    return float(np.nanmean(arr)) if len(arr) else 0.0


def _ema(values: np.ndarray, span: int) -> float:
    if len(values) == 0:
        return 0.0
    return float(pd.Series(values).ewm(span=span, adjust=False).mean().iloc[-1])


def _rsi(values: np.ndarray, period: int = 14) -> float:
    if len(values) < 2:
        return 50.0
    win = values[-(period + 1) :] if len(values) > period else values
    d = np.diff(win)
    if len(d) == 0:
        return 50.0
    gains = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    ag = float(np.mean(gains))
    al = float(np.mean(losses))
    if al <= 1e-12:
        return 100.0 if ag > 0 else 50.0
    rs = ag / al
    return float(100.0 - (100.0 / (1.0 + rs)))


def autocorr_lag(x: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(x) <= lag + 2:
        return np.nan
    a = x[:-lag]
    b = x[lag:]
    sa = float(np.nanstd(a))
    sb = float(np.nanstd(b))
    if sa < 1e-9 or sb < 1e-9:
        return np.nan
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else np.nan


def pattern_signature(series: pd.Series) -> dict[str, float]:
    x = np.asarray(series.values, dtype=float)
    if len(x) < 24:
        return {
            "n_points": float(len(x)),
            "dominant_lag": np.nan,
            "dominant_lag_acf": np.nan,
            "seasonal_strength": np.nan,
            "trend_strength": np.nan,
            "change_points": np.nan,
        }

    # Dominant lag by autocorrelation on lags 2..24.
    lags = list(range(2, 25))
    acfs = [autocorr_lag(x, lag) for lag in lags]
    acf_arr = np.asarray(acfs, dtype=float)
    if np.isfinite(acf_arr).any():
        j = int(np.nanargmax(np.abs(acf_arr)))
        dom_lag = float(lags[j])
        dom_acf = float(acf_arr[j])
    else:
        dom_lag = np.nan
        dom_acf = np.nan

    # Simple additive decomposition proxies.
    s = pd.Series(x)
    trend = s.rolling(12, min_periods=1, center=True).mean()
    detr = s - trend
    month = pd.Series(series.index.month, index=series.index)
    seas_profile = detr.groupby(month).mean()
    seas = month.map(seas_profile).astype(float)
    resid = s - trend - seas

    def safe_nanvar(arr: np.ndarray) -> float:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) < 2:
            return 0.0
        return float(np.var(a))

    var_x = safe_nanvar(x)
    var_res = safe_nanvar(resid.values)
    var_detr = safe_nanvar((s - trend).values)
    if var_x > 1e-12:
        seasonal_strength = float(max(0.0, min(1.0, 1.0 - var_res / var_x)))
        trend_strength = float(max(0.0, min(1.0, 1.0 - var_detr / var_x)))
    else:
        seasonal_strength = 0.0
        trend_strength = 0.0

    # Regime/change-point proxy from robust differenced z-score.
    d = np.diff(x)
    med = float(np.nanmedian(d))
    mad = float(np.nanmedian(np.abs(d - med))) + 1e-9
    rz = np.abs((d - med) / (1.4826 * mad))
    change_points = float(np.sum(rz > 3.5))

    return {
        "n_points": float(len(x)),
        "dominant_lag": dom_lag,
        "dominant_lag_acf": dom_acf,
        "seasonal_strength": seasonal_strength,
        "trend_strength": trend_strength,
        "change_points": change_points,
    }


def feature_dict(history: np.ndarray, ts: pd.Timestamp, pos: int) -> dict[str, float]:
    h = np.asarray(history, dtype=float)
    hmean = _mean(h)

    def lag(k: int) -> float:
        return float(h[-k]) if len(h) >= k else np.nan

    feats: dict[str, float] = {
        "trend": float(pos),
        "trend_log": float(np.log1p(max(1, pos))),
        "month_sin": float(np.sin(2.0 * np.pi * ts.month / 12.0)),
        "month_cos": float(np.cos(2.0 * np.pi * ts.month / 12.0)),
    }

    for k in LAGS:
        feats[f"lag_{k}"] = lag(k)

    l1, l2, l3, l12 = feats["lag_1"], feats["lag_2"], feats["lag_3"], feats["lag_12"]

    def ret(a: float, b: float) -> float:
        if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-9:
            return 0.0
        return float(a / b - 1.0)

    feats["ret_1"] = ret(l1, l2)
    feats["ret_3"] = ret(l1, l3)
    feats["ret_12"] = ret(l1, l12)
    feats["mom_1"] = float(l1 - l2) if np.isfinite(l1) and np.isfinite(l2) else 0.0
    feats["mom_3"] = float(l1 - l3) if np.isfinite(l1) and np.isfinite(l3) else 0.0
    feats["mom_12"] = float(l1 - l12) if np.isfinite(l1) and np.isfinite(l12) else 0.0

    w3 = h[-3:] if len(h) else np.array([hmean])
    w6 = h[-6:] if len(h) else np.array([hmean])
    w12 = h[-12:] if len(h) else np.array([hmean])
    w20 = h[-20:] if len(h) else np.array([hmean])

    feats["ma_3"] = _mean(w3)
    feats["ma_6"] = _mean(w6)
    feats["ma_12"] = _mean(w12)
    feats["ma_20"] = _mean(w20)
    feats["std_3"] = float(np.nanstd(w3))
    feats["std_6"] = float(np.nanstd(w6))
    feats["std_12"] = float(np.nanstd(w12))

    ema12 = _ema(h, 12)
    ema26 = _ema(h, 26)
    feats["ema_12"] = ema12
    feats["ema_26"] = ema26
    feats["macd"] = ema12 - ema26

    if len(h):
        s = pd.Series(h)
        macd_s = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
        signal = float(macd_s.ewm(span=9, adjust=False).mean().iloc[-1])
    else:
        signal = 0.0
    feats["macd_signal"] = signal
    feats["macd_hist"] = feats["macd"] - signal

    std20 = float(np.nanstd(w20))
    feats["boll_z"] = float((l1 - feats["ma_20"]) / (2 * std20)) if np.isfinite(l1) and std20 > 1e-9 else 0.0
    feats["rsi_14"] = _rsi(h)
    feats["vol_14"] = float(np.nanmean(np.abs(np.diff(h[-15:])))) if len(h) > 1 else 0.0

    for k, v in list(feats.items()):
        if not np.isfinite(v):
            feats[k] = hmean if k.startswith("lag_") else 0.0

    return feats


def build_supervised(series: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    vals = series.values.astype(float)
    idx = series.index
    start = min(max(24, len(series) // 4), max(8, len(series) - 1))

    X_rows: list[dict[str, float]] = []
    ys: list[float] = []

    for i in range(start, len(series)):
        feats = feature_dict(vals[:i], pd.Timestamp(idx[i]), i)
        y = float(vals[i])
        if np.isfinite(y):
            X_rows.append(feats)
            ys.append(y)

    if not X_rows:
        return pd.DataFrame(), np.array([], dtype=float)
    return pd.DataFrame(X_rows), np.asarray(ys, dtype=float)


def _standardize(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = A.mean(axis=0)
    sigma = A.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    Z = (A - mu) / sigma
    return Z, mu, sigma


def fit_ta_ridge(series: pd.Series) -> TARidgeModel | None:
    X, y = build_supervised(series)
    if len(X) < 12:
        return None
    A = X.values.astype(float)
    Z, mu, sigma = _standardize(A)
    X1 = np.column_stack([np.ones(len(Z)), Z])

    alpha = 2.5
    I = np.eye(X1.shape[1])
    I[0, 0] = 0.0

    try:
        beta = np.linalg.solve(X1.T @ X1 + alpha * I, X1.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X1, y, rcond=None)[0]

    return TARidgeModel(cols=list(X.columns), mu=mu, sigma=sigma, beta=beta)


def fit_ta_robust(series: pd.Series) -> TARobustModel | None:
    X, y = build_supervised(series)
    if len(X) < 14:
        return None
    A = X.values.astype(float)
    Z, mu, sigma = _standardize(A)
    Xc = sm.add_constant(Z, has_constant="add")

    try:
        res = sm.RLM(y, Xc, M=sm.robust.norms.HuberT()).fit(maxiter=250)
    except Exception:
        return None

    return TARobustModel(cols=list(X.columns), mu=mu, sigma=sigma, beta=np.asarray(res.params, dtype=float))


def fit_theta(series: pd.Series) -> Any | None:
    if len(series) < 18:
        return None
    try:
        return ThetaModel(series, period=12).fit(use_mle=True)
    except Exception:
        return None


def fit_ets_add(series: pd.Series) -> Any | None:
    if len(series) < 18:
        return None
    try:
        mdl = ExponentialSmoothing(
            series,
            trend="add",
            damped_trend=False,
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        )
        return mdl.fit(optimized=True, use_brute=True)
    except Exception:
        return None


def fit_ets_damped(series: pd.Series) -> Any | None:
    if len(series) < 18:
        return None
    try:
        mdl = ExponentialSmoothing(
            series,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        )
        return mdl.fit(optimized=True, use_brute=True)
    except Exception:
        return None


def fit_sarima_generic(series: pd.Series, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]) -> Any | None:
    if len(series) < 24:
        return None
    try:
        mdl = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return mdl.fit(disp=False)
    except Exception:
        return None


def fit_sarima_111x111(series: pd.Series) -> Any | None:
    return fit_sarima_generic(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))


def fit_sarima_210x110(series: pd.Series) -> Any | None:
    return fit_sarima_generic(series, order=(2, 1, 0), seasonal_order=(1, 1, 0, 12))


def fit_sarima_011x011(series: pd.Series) -> Any | None:
    return fit_sarima_generic(series, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))


def fit_seasonal_naive(series: pd.Series) -> dict[str, Any] | None:
    if len(series) == 0:
        return None
    vals = np.asarray(series.values, dtype=float)
    if len(vals) < 12:
        profile = np.repeat(_mean(vals), 12)
    else:
        profile = np.array([np.nanmean(vals[i::12]) for i in range(12)], dtype=float)
        profile = np.where(np.isnan(profile), _mean(vals), profile)
    return {"profile": profile}


def fit_pattern_knn(series: pd.Series, window: int, k: int) -> PatternKNNModel | None:
    y = np.asarray(series.values, dtype=float)
    idx = series.index
    if len(y) < window + 18:
        return None

    X_rows: list[np.ndarray] = []
    d_rows: list[float] = []
    m_rows: list[int] = []

    for t in range(window, len(y)):
        win = y[t - window : t]
        mu = float(np.nanmean(win))
        sd = float(np.nanstd(win))
        if not np.isfinite(sd) or sd < 1e-9:
            sd = 1.0
        z = (win - mu) / sd
        d = float(y[t] - y[t - 1])
        if np.isfinite(d) and np.isfinite(z).all():
            X_rows.append(z.astype(float))
            d_rows.append(d)
            m_rows.append(int(pd.Timestamp(idx[t]).month))

    if len(X_rows) < max(24, k * 4):
        return None

    d_arr = np.asarray(d_rows, dtype=float)
    d_scale = float(np.nanstd(d_arr))
    if not np.isfinite(d_scale) or d_scale <= 1e-9:
        d_scale = float(np.nanmean(np.abs(d_arr))) if len(d_arr) else 1.0
    if not np.isfinite(d_scale) or d_scale <= 1e-9:
        d_scale = 1.0

    return PatternKNNModel(
        window=window,
        k=k,
        X=np.vstack(X_rows),
        next_delta=d_arr,
        next_month=np.asarray(m_rows, dtype=int),
        delta_scale=d_scale,
    )


def fit_pattern_knn12(series: pd.Series) -> PatternKNNModel | None:
    return fit_pattern_knn(series, window=12, k=7)


def fit_pattern_knn24(series: pd.Series) -> PatternKNNModel | None:
    return fit_pattern_knn(series, window=24, k=9)


def fit_pattern_analog(series: pd.Series, window: int, top_k: int) -> PatternAnalogModel | None:
    if len(series) < window + 18:
        return None
    return PatternAnalogModel(window=window, top_k=top_k)


def fit_pattern_analog24(series: pd.Series) -> PatternAnalogModel | None:
    return fit_pattern_analog(series, window=24, top_k=8)


def fit_pattern_analog36(series: pd.Series) -> PatternAnalogModel | None:
    return fit_pattern_analog(series, window=36, top_k=10)


def _zscore(v: np.ndarray) -> np.ndarray:
    mu = float(np.nanmean(v))
    sd = float(np.nanstd(v))
    if not np.isfinite(sd) or sd < 1e-9:
        sd = 1.0
    return (v - mu) / sd


def analog_match_paths(
    history: pd.Series,
    future_index: pd.DatetimeIndex,
    window: int,
    top_k: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Find similar historical windows and return weighted future path + case report."""
    h = history.copy().astype(float)
    y = np.asarray(h.values, dtype=float)
    idx = h.index
    horizon = len(future_index)

    if horizon == 0 or len(y) < window + horizon + 2:
        return np.array([], dtype=float), pd.DataFrame()

    ref = y[-window:]
    z_ref = _zscore(ref)
    target_month = int(pd.Timestamp(future_index[0]).month)

    cases: list[dict[str, Any]] = []
    paths: list[np.ndarray] = []
    dists: list[float] = []

    # Candidate i means future starts at i and window ends at i-1.
    max_i = len(y) - horizon
    for i in range(window, max_i + 1):
        # Seasonal alignment: same start month as target forecast start month.
        if int(pd.Timestamp(idx[i]).month) != target_month:
            continue

        win = y[i - window : i]
        z_win = _zscore(win)
        dist = float(np.sqrt(np.mean((z_ref - z_win) ** 2)))
        if not np.isfinite(dist):
            continue

        base = float(y[i - 1])
        fut = y[i : i + horizon]
        delta = fut - base
        path = float(y[-1]) + delta

        paths.append(path)
        dists.append(dist)

        c = {
            "distance": dist,
            "window_start": str(pd.Timestamp(idx[i - window])),
            "window_end": str(pd.Timestamp(idx[i - 1])),
            "future_start": str(pd.Timestamp(idx[i])),
            "future_end": str(pd.Timestamp(idx[i + horizon - 1])),
            "delta_1": float(delta[0]),
            "delta_3": float(delta[2] if horizon >= 3 else delta[-1]),
            "delta_6": float(delta[5] if horizon >= 6 else delta[-1]),
            "delta_last": float(delta[-1]),
        }
        cases.append(c)

    # If no month-aligned matches, relax alignment.
    if not paths:
        for i in range(window, max_i + 1):
            win = y[i - window : i]
            z_win = _zscore(win)
            dist = float(np.sqrt(np.mean((z_ref - z_win) ** 2)))
            if not np.isfinite(dist):
                continue
            base = float(y[i - 1])
            fut = y[i : i + horizon]
            delta = fut - base
            path = float(y[-1]) + delta
            paths.append(path)
            dists.append(dist)
            cases.append(
                {
                    "distance": dist,
                    "window_start": str(pd.Timestamp(idx[i - window])),
                    "window_end": str(pd.Timestamp(idx[i - 1])),
                    "future_start": str(pd.Timestamp(idx[i])),
                    "future_end": str(pd.Timestamp(idx[i + horizon - 1])),
                    "delta_1": float(delta[0]),
                    "delta_3": float(delta[2] if horizon >= 3 else delta[-1]),
                    "delta_6": float(delta[5] if horizon >= 6 else delta[-1]),
                    "delta_last": float(delta[-1]),
                }
            )

    if not paths:
        return np.array([], dtype=float), pd.DataFrame()

    order = np.argsort(np.asarray(dists))
    sel = order[: min(top_k, len(order))]
    P = np.vstack([paths[i] for i in sel])
    d = np.asarray([dists[i] for i in sel], dtype=float)

    scale = float(np.nanmedian(d)) + 1e-9
    w = np.exp(-d / scale)
    if not np.isfinite(w).all() or w.sum() <= 1e-12:
        w = np.ones_like(d)
    w = w / w.sum()

    yhat = (w[:, None] * P).sum(axis=0)

    case_df = pd.DataFrame([cases[i] for i in sel]).copy()
    case_df.insert(0, "rank", np.arange(1, len(case_df) + 1))
    case_df["weight"] = w
    return yhat.astype(float), case_df


def predict_ta(model: TARidgeModel | TARobustModel, history: pd.Series, future_index: pd.DatetimeIndex, start_pos: int) -> np.ndarray:
    h = history.copy().astype(float)
    preds: list[float] = []

    for step, ts in enumerate(future_index):
        feats = feature_dict(h.values, pd.Timestamp(ts), start_pos + step)
        x = np.array([feats[c] for c in model.cols], dtype=float)
        z = (x - model.mu) / model.sigma
        x1 = np.concatenate([[1.0], z])
        yhat = float(np.dot(x1, model.beta))
        preds.append(yhat)
        h.loc[ts] = yhat

    return np.asarray(preds, dtype=float)


def predict_ts(model: Any, future_index: pd.DatetimeIndex) -> np.ndarray:
    if len(future_index) == 0:
        return np.array([], dtype=float)
    fc = model.forecast(steps=len(future_index))
    arr = np.asarray(fc, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def predict_seasonal_naive(model: dict[str, Any], future_index: pd.DatetimeIndex) -> np.ndarray:
    profile = np.asarray(model["profile"], dtype=float)
    if len(profile) < 12:
        profile = np.pad(profile, (0, 12 - len(profile)), constant_values=float(_mean(profile)))
    return np.array([profile[(ts.month - 1) % 12] for ts in future_index], dtype=float)


def predict_pattern_knn(model: PatternKNNModel, history: pd.Series, future_index: pd.DatetimeIndex) -> np.ndarray:
    h = history.copy().astype(float)
    preds: list[float] = []

    for ts in future_index:
        if len(h) < model.window:
            yhat = float(h.iloc[-1]) if len(h) else 0.0
            preds.append(yhat)
            h.loc[ts] = yhat
            continue

        win = np.asarray(h.values[-model.window :], dtype=float)
        mu = float(np.nanmean(win))
        sd = float(np.nanstd(win))
        if not np.isfinite(sd) or sd < 1e-9:
            sd = 1.0
        z = (win - mu) / sd

        month = int(pd.Timestamp(ts).month)
        mask = model.next_month == month
        X = model.X[mask] if mask.any() else model.X
        d = model.next_delta[mask] if mask.any() else model.next_delta
        if len(X) == 0:
            X = model.X
            d = model.next_delta

        dist = np.linalg.norm(X - z[None, :], axis=1)
        k = int(min(model.k, len(dist)))
        sel = np.argpartition(dist, k - 1)[:k] if k > 0 else np.arange(len(dist))
        d_sel = d[sel]
        dist_sel = dist[sel]

        scale = float(np.nanmedian(dist_sel)) + 1e-6
        w = np.exp(-dist_sel / scale)
        if not np.isfinite(w).all() or w.sum() <= 1e-12:
            w = np.ones_like(dist_sel)
        w = w / w.sum()

        delta = float(np.dot(w, d_sel))
        delta = float(np.clip(delta, -3.0 * model.delta_scale, 3.0 * model.delta_scale))
        yhat = float(h.iloc[-1] + delta)

        preds.append(yhat)
        h.loc[ts] = yhat

    return np.asarray(preds, dtype=float)


def predict_pattern_analog(model: PatternAnalogModel, history: pd.Series, future_index: pd.DatetimeIndex) -> np.ndarray:
    yhat, _cases = analog_match_paths(
        history=history,
        future_index=future_index,
        window=model.window,
        top_k=model.top_k,
    )
    if len(yhat) == len(future_index):
        return yhat
    # fallback
    naive = fit_seasonal_naive(history)
    if naive is None:
        return np.zeros(len(future_index), dtype=float)
    return predict_seasonal_naive(naive, future_index)


def discover_auto_sarima_orders(series_t: pd.Series, max_orders: int = 2) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int], float]]:
    """Quick AIC search to enrich candidate set with the best SARIMA orders."""
    if len(series_t) < 28:
        return []

    grid = [
        ((1, 1, 1), (1, 1, 1, 12)),
        ((1, 1, 1), (0, 1, 1, 12)),
        ((0, 1, 2), (0, 1, 1, 12)),
        ((1, 1, 2), (0, 1, 1, 12)),
        ((2, 1, 1), (0, 1, 1, 12)),
        ((0, 1, 1), (0, 1, 1, 12)),
        ((2, 1, 0), (1, 1, 0, 12)),
    ]

    scored: list[tuple[tuple[int, int, int], tuple[int, int, int, int], float]] = []
    for order, seasonal in grid:
        fit = fit_sarima_generic(series_t, order=order, seasonal_order=seasonal)
        if fit is None:
            continue
        aic = getattr(fit, "aic", np.nan)
        if np.isfinite(aic):
            scored.append((order, seasonal, float(aic)))

    scored = sorted(scored, key=lambda x: x[2])
    out: list[tuple[tuple[int, int, int], tuple[int, int, int, int], float]] = []
    seen: set[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = set()
    for order, seasonal, aic in scored:
        key = (order, seasonal)
        if key in seen:
            continue
        seen.add(key)
        out.append((order, seasonal, aic))
        if len(out) >= max_orders:
            break
    return out


def candidate_specs_for_series(series_raw: pd.Series, transform: Transform) -> dict[str, dict[str, Any]]:
    specs = {
        "ta_ridge": {"fit": fit_ta_ridge, "kind": "ta"},
        "ta_robust": {"fit": fit_ta_robust, "kind": "ta"},
        "pattern_knn12": {"fit": fit_pattern_knn12, "kind": "pattern"},
        "pattern_knn24": {"fit": fit_pattern_knn24, "kind": "pattern"},
        "pattern_analog24": {"fit": fit_pattern_analog24, "kind": "analog"},
        "pattern_analog36": {"fit": fit_pattern_analog36, "kind": "analog"},
        "theta": {"fit": fit_theta, "kind": "ts"},
        "ets_add": {"fit": fit_ets_add, "kind": "ts"},
        "ets_damped": {"fit": fit_ets_damped, "kind": "ts"},
        "sarima_111x111": {"fit": fit_sarima_111x111, "kind": "ts"},
        "sarima_210x110": {"fit": fit_sarima_210x110, "kind": "ts"},
        "sarima_011x011": {"fit": fit_sarima_011x011, "kind": "ts"},
        "seasonal_naive": {"fit": fit_seasonal_naive, "kind": "naive"},
    }

    series_t = transform.forward(series_raw)
    auto_orders = discover_auto_sarima_orders(series_t, max_orders=2)
    for i, (order, seasonal, _aic) in enumerate(auto_orders, start=1):
        name = f"sarima_auto_{i}"

        def _fit(s: pd.Series, order=order, seasonal=seasonal):
            return fit_sarima_generic(s, order=order, seasonal_order=seasonal)

        specs[name] = {"fit": _fit, "kind": "ts"}

    return specs


def predict_with_spec(spec: dict[str, Any], model: Any, history: pd.Series, future_index: pd.DatetimeIndex, start_pos: int) -> np.ndarray:
    kind = spec["kind"]
    if kind == "ta":
        return predict_ta(model, history=history, future_index=future_index, start_pos=start_pos)
    if kind == "pattern":
        return predict_pattern_knn(model, history=history, future_index=future_index)
    if kind == "analog":
        return predict_pattern_analog(model, history=history, future_index=future_index)
    if kind == "naive":
        return predict_seasonal_naive(model, future_index=future_index)
    return predict_ts(model, future_index=future_index)


def rolling_cutpoints(n: int, folds: int, horizon: int, min_train: int) -> list[int]:
    out = []
    for i in range(folds, 0, -1):
        cut = n - i * horizon
        if min_train <= cut < n:
            out.append(cut)
    return sorted(set(out))


def evaluate_candidate(
    series_raw: pd.Series,
    transform: Transform,
    name: str,
    spec: dict[str, Any],
    cuts: list[int],
    horizon: int,
) -> CandidateEval | None:
    preds_all: list[float] = []
    y_all: list[float] = []
    step_all: list[int] = []

    for cut in cuts:
        train_raw = series_raw.iloc[:cut]
        valid_raw = series_raw.iloc[cut : cut + horizon]
        if len(valid_raw) == 0:
            continue

        train_t = transform.forward(train_raw)
        model = spec["fit"](train_t)
        if model is None:
            return None

        pred_t = predict_with_spec(spec, model, history=train_t, future_index=valid_raw.index, start_pos=len(train_t))
        if len(pred_t) != len(valid_raw) or not np.isfinite(pred_t).all():
            return None

        pred = transform.inverse(pred_t)
        if not np.isfinite(pred).all():
            return None

        err = valid_raw.values.astype(float) - pred.astype(float)
        preds_all.extend(pred.tolist())
        y_all.extend(valid_raw.values.astype(float).tolist())
        step_all.extend(list(range(1, len(valid_raw) + 1)))

    if not preds_all:
        return None

    p = np.asarray(preds_all, dtype=float)
    y = np.asarray(y_all, dtype=float)
    e = y - p
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    return CandidateEval(name=name, mae=mae, rmse=rmse, preds=p, y_true=y, step_idx=np.asarray(step_all, dtype=int))


def fit_final_candidate(series_raw: pd.Series, transform: Transform, name: str, spec: dict[str, Any]) -> CandidateFitted | None:
    series_t = transform.forward(series_raw)
    model = spec["fit"](series_t)
    if model is None:
        return None
    return CandidateFitted(name=name, model=model, spec=spec)


def stack_weights(pred_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    w, _ = nnls(pred_matrix, y_true)
    if w.sum() <= 1e-12:
        w = np.ones(pred_matrix.shape[1], dtype=float)
    return w / w.sum()


def build_ensemble(
    series_raw: pd.Series,
    transform: Transform,
    max_models: int,
    cv_folds: int,
    cv_horizon: int,
    force_models: list[str] | None = None,
) -> EnsemblePack | None:
    specs = candidate_specs_for_series(series_raw=series_raw, transform=transform)
    n = len(series_raw)
    if n < 12:
        return None

    horizon = min(cv_horizon, max(2, n // 8))
    cuts = rolling_cutpoints(n=n, folds=cv_folds, horizon=horizon, min_train=max(18, horizon * 2))
    if not cuts:
        return None

    evals: list[CandidateEval] = []
    for name, spec in specs.items():
        ev = evaluate_candidate(series_raw=series_raw, transform=transform, name=name, spec=spec, cuts=cuts, horizon=horizon)
        if ev is not None and np.isfinite(ev.mae):
            evals.append(ev)

    if not evals:
        return None

    evals = sorted(evals, key=lambda x: x.mae)
    forced = [m.strip() for m in (force_models or []) if m and m.strip()]
    if forced:
        keep_evals = [ev for ev in evals if ev.name in forced]
        if not keep_evals:
            keep_evals = evals[:1]
    elif max_models <= 1:
        keep_evals = evals[:1]
    else:
        keep_evals = evals[: max(1, min(max_models * 2, len(evals)))]

    y_true = keep_evals[0].y_true
    if any(len(ev.y_true) != len(y_true) for ev in keep_evals):
        keep_evals = evals[: max(1, min(max_models, len(evals)))]

    if forced:
        inv = np.array([1.0 / max(ev.mae, 1e-9) for ev in keep_evals], dtype=float)
        w = inv / inv.sum()
        selected = keep_evals
    elif max_models <= 1:
        selected = [keep_evals[0]]
        w = np.array([1.0], dtype=float)
    else:
        P = np.column_stack([ev.preds for ev in keep_evals])
        w_full = stack_weights(P, keep_evals[0].y_true)

        # Keep strongest positive-weight models up to max_models.
        order = np.argsort(-w_full)
        sel_idx = [i for i in order if w_full[i] > 1e-8][: max_models]
        if not sel_idx:
            sel_idx = list(range(min(max_models, len(keep_evals))))

        selected = [keep_evals[i] for i in sel_idx]
        w = w_full[sel_idx]
        w = w / w.sum()

    final_candidates: list[CandidateFitted] = []
    final_names: list[str] = []
    final_w: list[float] = []

    for i, ev in enumerate(selected):
        cf = fit_final_candidate(series_raw=series_raw, transform=transform, name=ev.name, spec=specs[ev.name])
        if cf is None:
            continue
        final_candidates.append(cf)
        final_names.append(ev.name)
        final_w.append(float(w[i]))

    if not final_candidates:
        return None

    wf = np.asarray(final_w, dtype=float)
    wf = wf / wf.sum()

    # Ensemble residuals on CV set (for interval calibration).
    Psel = np.column_stack([selected[i].preds for i in range(len(selected))])
    ens_pred = (wf[: Psel.shape[1]][None, :] * Psel).sum(axis=1)
    residuals = selected[0].y_true - ens_pred

    step_idx = selected[0].step_idx
    residuals_by_step: dict[int, np.ndarray] = {}
    for k in np.unique(step_idx):
        residuals_by_step[int(k)] = residuals[step_idx == k]

    cv_mae = float(np.mean(np.abs(residuals)))
    # update candidate names in same order as weights/candidates
    for i, c in enumerate(final_candidates):
        c.name = final_names[i]

    weight_map = {name: float(wf[i]) for i, name in enumerate(final_names)}
    leaderboard = pd.DataFrame(
        [
            {
                "model": ev.name,
                "mae": float(ev.mae),
                "rmse": float(ev.rmse),
                "selected": ev.name in weight_map,
                "weight": float(weight_map.get(ev.name, 0.0)),
            }
            for ev in evals
        ]
    ).sort_values(["mae", "rmse"], ignore_index=True)

    return EnsemblePack(
        candidates=final_candidates,
        weights=wf,
        cv_mae=cv_mae,
        residuals=residuals,
        residuals_by_step=residuals_by_step,
        leaderboard=leaderboard,
    )


def refit_ensemble_from_template(template: EnsemblePack | None, series_raw: pd.Series, transform: Transform) -> EnsemblePack | None:
    """Refit selected best-model set on new history, preserving model identities and weights."""
    if template is None:
        return None

    series_t = transform.forward(series_raw)
    candidates: list[CandidateFitted] = []
    weights: list[float] = []

    for i, cf in enumerate(template.candidates):
        model = cf.spec["fit"](series_t)
        if model is None:
            continue
        candidates.append(CandidateFitted(name=cf.name, model=model, spec=cf.spec))
        weights.append(float(template.weights[i]))

    if not candidates:
        return None

    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return EnsemblePack(
        candidates=candidates,
        weights=w,
        cv_mae=template.cv_mae,
        residuals=template.residuals,
        residuals_by_step=template.residuals_by_step,
        leaderboard=template.leaderboard.copy(),
    )


def interval_radius(residuals: np.ndarray, residuals_by_step: dict[int, np.ndarray], steps: int, confidence: float) -> np.ndarray:
    if steps <= 0:
        return np.array([], dtype=float)

    c = float(np.clip(confidence, 0.55, 0.95))
    if residuals.size == 0:
        return np.zeros(steps, dtype=float)

    base = np.abs(np.asarray(residuals, dtype=float))
    cap = float(np.quantile(base, 0.90))
    base = np.clip(base, None, cap)
    q_base = float(np.quantile(base, c))

    out = np.zeros(steps, dtype=float)
    max_step = max(residuals_by_step.keys()) if residuals_by_step else 1

    for k in range(1, steps + 1):
        arr = residuals_by_step.get(k)
        if arr is None:
            arr = residuals_by_step.get(max_step)
        if arr is None or len(arr) < 5:
            q = q_base
        else:
            aa = np.abs(np.asarray(arr, dtype=float))
            cc = float(np.quantile(aa, 0.90))
            aa = np.clip(aa, None, cc)
            q = float(np.quantile(aa, c))

        # Slight widening only.
        out[k - 1] = q * (1.0 + 0.008 * np.sqrt(max(0.0, k - 1.0)))

    return out


def apply_bounds(values: np.ndarray, variable: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    if variable == "humidity":
        arr = np.clip(arr, 0, 100)
    elif variable in {"precip", "pressure"}:
        arr = np.clip(arr, 0, None)
    return arr


def tighten_band(yhat: np.ndarray, low: np.ndarray, high: np.ndarray, variable: str, history: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if len(yhat) == 0:
        return low, high

    rad = (np.asarray(high, dtype=float) - np.asarray(low, dtype=float)) / 2.0
    y = np.asarray(yhat, dtype=float)

    h = np.asarray(history.values, dtype=float)
    hist_mad = float(np.nanmedian(np.abs(np.diff(h)))) if len(h) > 1 else float(np.nanstd(h))
    if not np.isfinite(hist_mad):
        hist_mad = 0.0

    if variable == "precip":
        cap = np.minimum(0.08 * np.abs(y) + 2.0, 0.18 * hist_mad + 4.0)
    elif variable == "humidity":
        cap = np.minimum(0.06 * np.abs(y) + 0.7, 0.22 * hist_mad + 0.7)
    elif variable == "pressure":
        cap = np.minimum(0.015 * np.abs(y) + 0.2, 0.20 * hist_mad + 0.2)
    else:
        cap = np.minimum(0.09 * np.abs(y) + 0.8, 0.30 * hist_mad + 0.8)

    cap = np.where(np.isfinite(cap), cap, 0.0)
    rad = np.minimum(rad, cap)
    return y - rad, y + rad


def add_realistic_irregularity(
    base_pred: np.ndarray,
    history: pd.Series,
    future_index: pd.DatetimeIndex,
    variable: str,
    strength: float,
) -> np.ndarray:
    """Inject controlled variability so forecast lines are not unnaturally smooth."""
    y = np.asarray(base_pred, dtype=float).copy()
    if len(y) == 0:
        return y

    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 1e-9:
        return y

    h = np.asarray(history.values, dtype=float)
    if len(h) < 8:
        return y

    hist = pd.Series(h, index=history.index)
    # Month-level variability profile from history.
    month_std = hist.groupby(hist.index.month).std().to_dict()
    # Robust volatility from first differences.
    d = np.diff(h)
    d_vol = float(np.nanmedian(np.abs(d))) if len(d) else float(np.nanstd(h))
    if not np.isfinite(d_vol):
        d_vol = 0.0

    # Deterministic seed for repeatability.
    seed_map = {"humidity": 101, "temp": 102, "pressure": 103, "precip": 104}
    rng = np.random.default_rng(seed_map.get(variable, 999))

    eps = np.zeros(len(y), dtype=float)
    ar = 0.45 if variable != "precip" else 0.30

    for i, ts in enumerate(future_index):
        ms = float(month_std.get(ts.month, np.nan))
        if not np.isfinite(ms) or ms <= 1e-9:
            ms = d_vol
        local_scale = 0.35 * ms + 0.05 * abs(float(y[i])) + 0.15 * d_vol
        z = float(rng.standard_t(df=5))
        shock = s * local_scale * z
        eps[i] = (ar * eps[i - 1] if i > 0 else 0.0) + shock

    # Keep each variable physically meaningful.
    if variable == "precip":
        y = y + np.maximum(eps, -0.70 * y)
        y = np.clip(y, 0, None)
    elif variable == "humidity":
        y = np.clip(y + eps, 0, 100)
    elif variable == "pressure":
        y = np.clip(y + eps, 0, None)
    else:
        y = y + eps

    return y


def forecast_ensemble(
    ens: EnsemblePack | None,
    series_raw: pd.Series,
    transform: Transform,
    future_index: pd.DatetimeIndex,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, float]:
    steps = len(future_index)
    if steps == 0:
        e = np.array([], dtype=float)
        return e, e, e, "none", "", np.nan

    if ens is None or len(series_raw) < 12:
        # fallback seasonal naive
        model = fit_seasonal_naive(series_raw)
        y = predict_seasonal_naive(model, future_index) if model is not None else np.zeros(steps)
        spread = float(np.nanstd(series_raw.values)) if len(series_raw) > 1 else 0.0
        rad = np.repeat(spread * 0.25, steps)
        return y, y - rad, y + rad, "seasonal_naive", "seasonal_naive", np.nan

    series_t = transform.forward(series_raw)

    pred_list = []
    model_names = []
    for cf in ens.candidates:
        pred_t = predict_with_spec(cf.spec, cf.model, history=series_t, future_index=future_index, start_pos=len(series_t))
        pred = transform.inverse(pred_t)
        pred_list.append(pred)
        model_names.append(cf.name)

    mat = np.vstack(pred_list)
    w = ens.weights
    if len(w) != mat.shape[0]:
        w = np.repeat(1.0 / mat.shape[0], mat.shape[0])
    yhat = (w[:, None] * mat).sum(axis=0)

    rad = interval_radius(ens.residuals, ens.residuals_by_step, steps=steps, confidence=confidence)
    low = yhat - rad
    high = yhat + rad

    weight_desc = ",".join([f"{n}:{ww:.3f}" for n, ww in zip(model_names, w)])
    return yhat, low, high, "+".join(model_names), weight_desc, ens.cv_mae


def build_pipeline_for_variable(
    variable: str,
    series: pd.Series,
    last_obs: pd.Timestamp,
    split_year: int,
    target_year: int,
    horizon_months: int,
    confidence: float,
    max_models: int,
    cv_folds: int,
    cv_horizon: int,
    irregularity: float,
    force_models: list[str] | None,
    analog_alpha: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_end = pd.Timestamp(year=split_year, month=12, day=1)
    target_end = pd.Timestamp(year=target_year, month=12, day=1)

    transform = pick_transform(variable)
    observed = series.loc[:last_obs].copy().sort_index()

    bridge_index = (
        pd.date_range(last_obs + pd.offsets.MonthBegin(1), split_end, freq="MS")
        if last_obs < split_end
        else pd.DatetimeIndex([])
    )

    ens_bridge = build_ensemble(
        observed,
        transform=transform,
        max_models=max_models,
        cv_folds=cv_folds,
        cv_horizon=cv_horizon,
        force_models=force_models,
    )
    b_pred, _, _, b_model, b_weights, b_cv_mae = forecast_ensemble(
        ens=ens_bridge,
        series_raw=observed,
        transform=transform,
        future_index=bridge_index,
        confidence=confidence,
    )
    b_pred = add_realistic_irregularity(
        base_pred=b_pred,
        history=observed,
        future_index=bridge_index,
        variable=variable,
        strength=irregularity * 0.65,
    )
    b_pred = apply_bounds(b_pred, variable)

    extended = observed.copy()
    if len(bridge_index) > 0:
        extended = pd.concat([extended, pd.Series(b_pred, index=bridge_index)])

    if target_end <= split_end:
        future_index = pd.DatetimeIndex([])
    else:
        ff = pd.date_range(split_end + pd.offsets.MonthBegin(1), target_end, freq="MS")
        future_index = ff[:horizon_months] if horizon_months and horizon_months > 0 else ff

    # Keep future model identity anchored to best model(s) selected on true observed period.
    ens_future = refit_ensemble_from_template(ens_bridge, series_raw=extended, transform=transform)
    if ens_future is None:
        ens_future = build_ensemble(
            extended,
            transform=transform,
            max_models=max_models,
            cv_folds=cv_folds,
            cv_horizon=cv_horizon,
            force_models=force_models,
        )
    f_pred, f_low, f_high, f_model, f_weights, f_cv_mae = forecast_ensemble(
        ens=ens_future,
        series_raw=extended,
        transform=transform,
        future_index=future_index,
        confidence=confidence,
    )

    analog_pred, analog_cases = analog_match_paths(
        history=extended,
        future_index=future_index,
        window=24,
        top_k=8,
    )
    analog_alpha_used = 0.0
    a = float(np.clip(analog_alpha, 0.0, 1.0))
    if len(analog_pred) == len(f_pred) and a > 1e-9:
        f_pred = (1.0 - a) * np.asarray(f_pred, dtype=float) + a * np.asarray(analog_pred, dtype=float)
        analog_alpha_used = a

    # Re-center around a realistic, non-overly-smooth path.
    f_pred = add_realistic_irregularity(
        base_pred=f_pred,
        history=extended,
        future_index=future_index,
        variable=variable,
        strength=irregularity,
    )
    half_band = (np.asarray(f_high, dtype=float) - np.asarray(f_low, dtype=float)) / 2.0
    f_low = f_pred - half_band
    f_high = f_pred + half_band

    f_low, f_high = tighten_band(yhat=f_pred, low=f_low, high=f_high, variable=variable, history=extended)

    f_pred = apply_bounds(f_pred, variable)
    f_low = apply_bounds(f_low, variable)
    f_high = apply_bounds(f_high, variable)

    obs_df = pd.DataFrame(
        {
            "timestamp": observed.index,
            "value": observed.values,
            "low": np.nan,
            "high": np.nan,
            "segment": "observed",
            "is_forecast": False,
        }
    )
    bridge_df = pd.DataFrame(
        {
            "timestamp": bridge_index,
            "value": b_pred,
            "low": np.nan,
            "high": np.nan,
            "segment": "bridge_to_split",
            "is_forecast": True,
        }
    )
    future_df = pd.DataFrame(
        {
            "timestamp": future_index,
            "value": f_pred,
            "low": f_low,
            "high": f_high,
            "segment": "forecast_after_split",
            "is_forecast": True,
        }
    )

    full = pd.concat([obs_df, bridge_df, future_df], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    meta = {
        "split_end": split_end,
        "bridge_model": b_model,
        "bridge_weights": b_weights,
        "future_model": f_model,
        "future_weights": f_weights,
        "cv_mae_bridge": b_cv_mae,
        "cv_mae_future": f_cv_mae,
        "leaderboard_bridge": ens_bridge.leaderboard if ens_bridge is not None else pd.DataFrame(),
        "leaderboard_future": ens_future.leaderboard if ens_future is not None else pd.DataFrame(),
        "analog_cases": analog_cases,
        "analog_alpha_used": analog_alpha_used,
    }
    return full, meta


def plot_variable(df: pd.DataFrame, variable: str, split_end: pd.Timestamp, out_path: Path, confidence: float) -> None:
    obs = df[df["segment"] == "observed"]
    bridge = df[df["segment"] == "bridge_to_split"]
    future = df[df["segment"] == "forecast_after_split"]

    fig, ax = plt.subplots(figsize=(12.7, 4.9))
    ax.plot(obs["timestamp"], obs["value"], color="#1f77b4", linewidth=1.5, label="Tarihsel")

    if not bridge.empty:
        ax.plot(
            bridge["timestamp"],
            bridge["value"],
            color="#ff7f0e",
            linewidth=1.8,
            linestyle="--",
            label="Kopru (2020)",
        )

    if not future.empty:
        ax.plot(
            future["timestamp"],
            future["value"],
            color="#d62728",
            linewidth=2.0,
            label="Model-zoo tahmin",
        )
        if future["low"].notna().any() and future["high"].notna().any():
            ax.fill_between(
                future["timestamp"],
                future["low"],
                future["high"],
                color="#d62728",
                alpha=0.15,
                label=f"Guven bandi ({int(confidence * 100)}%)",
            )

    ax.axvline(split_end, color="#555555", linestyle=":", linewidth=1.1)
    ax.text(split_end, ax.get_ylim()[1], " 2020 split", fontsize=8, va="top", ha="left", color="#444444")
    ax.set_title(f"Aylik Gelismis Tahmin ({variable})")
    ax.set_xlabel("Tarih")
    ax.set_ylabel(f"Deger ({UNITS.get(variable, 'unit')})")
    ax.grid(alpha=0.24)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    obs = pd.read_parquet(args.observations)
    force_models = [m.strip() for m in args.force_models.split(",") if m.strip()]

    args.datasets_dir.mkdir(parents=True, exist_ok=True)
    args.charts_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_dir = args.datasets_dir / "leaderboards"
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    analog_dir = args.datasets_dir / "analog_cases"
    analog_dir.mkdir(parents=True, exist_ok=True)

    # Replace old chart set.
    for old in args.charts_dir.glob("*.png"):
        old.unlink()

    rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []

    for var in VARIABLES:
        series, last_obs = monthly_series(obs, var)
        if series.empty:
            continue

        sig = pattern_signature(series)
        pattern_rows.append(
            {
                "variable": var,
                **sig,
                "series_start": str(series.index.min()),
                "series_end": str(series.index.max()),
            }
        )

        out_df, meta = build_pipeline_for_variable(
            variable=var,
            series=series,
            last_obs=last_obs,
            split_year=args.split_year,
            target_year=args.target_year,
            horizon_months=args.horizon_months,
            confidence=args.confidence,
            max_models=args.max_models,
            cv_folds=args.cv_folds,
            cv_horizon=args.cv_horizon,
            irregularity=args.irregularity,
            force_models=force_models,
            analog_alpha=args.analog_alpha,
        )
        if out_df.empty:
            continue

        out_df["variable"] = var
        out_df["frequency"] = "monthly"
        out_df["unit"] = UNITS.get(var, "unknown")
        out_df["confidence_level"] = args.confidence

        stem = f"{var}_monthly_strong_split{args.split_year}_to{args.target_year}"
        csv_p = args.datasets_dir / f"{stem}.csv"
        pq_p = args.datasets_dir / f"{stem}.parquet"
        out_df.to_csv(csv_p, index=False)
        out_df.to_parquet(pq_p, index=False)

        chart_p = args.charts_dir / f"{stem}.png"
        plot_variable(out_df, var, pd.Timestamp(meta["split_end"]), chart_p, args.confidence)

        lb_bridge = meta.get("leaderboard_bridge", pd.DataFrame())
        lb_future = meta.get("leaderboard_future", pd.DataFrame())
        lb_bridge_csv = leaderboard_dir / f"{var}_bridge_leaderboard_split{args.split_year}_to{args.target_year}.csv"
        lb_future_csv = leaderboard_dir / f"{var}_future_leaderboard_split{args.split_year}_to{args.target_year}.csv"
        analog_csv = analog_dir / f"{var}_analog_cases_split{args.split_year}_to{args.target_year}.csv"
        if isinstance(lb_bridge, pd.DataFrame) and not lb_bridge.empty:
            lb_bridge.to_csv(lb_bridge_csv, index=False)
        else:
            pd.DataFrame(columns=["model", "mae", "rmse", "selected", "weight"]).to_csv(lb_bridge_csv, index=False)
        if isinstance(lb_future, pd.DataFrame) and not lb_future.empty:
            lb_future.to_csv(lb_future_csv, index=False)
        else:
            pd.DataFrame(columns=["model", "mae", "rmse", "selected", "weight"]).to_csv(lb_future_csv, index=False)
        analog_cases = meta.get("analog_cases", pd.DataFrame())
        if isinstance(analog_cases, pd.DataFrame) and not analog_cases.empty:
            analog_cases.to_csv(analog_csv, index=False)
        else:
            pd.DataFrame(columns=["rank", "distance", "window_start", "window_end", "future_start", "future_end", "delta_1", "delta_3", "delta_6", "delta_last", "weight"]).to_csv(analog_csv, index=False)

        rows.append(
            {
                "variable": var,
                "last_observation": str(last_obs),
                "split_year": args.split_year,
                "target_year": args.target_year,
                "confidence_level": args.confidence,
                "bridge_model": meta.get("bridge_model", ""),
                "bridge_weights": meta.get("bridge_weights", ""),
                "future_model": meta.get("future_model", ""),
                "future_weights": meta.get("future_weights", ""),
                "cv_mae_bridge": meta.get("cv_mae_bridge", np.nan),
                "cv_mae_future": meta.get("cv_mae_future", np.nan),
                "irregularity": args.irregularity,
                "analog_alpha": args.analog_alpha,
                "analog_alpha_used": meta.get("analog_alpha_used", 0.0),
                "force_models": ",".join(force_models),
                "rows_total": len(out_df),
                "bridge_rows": int((out_df["segment"] == "bridge_to_split").sum()),
                "forecast_rows": int((out_df["segment"] == "forecast_after_split").sum()),
                "dataset_csv": str(csv_p),
                "dataset_parquet": str(pq_p),
                "chart_png": str(chart_p),
                "leaderboard_bridge_csv": str(lb_bridge_csv),
                "leaderboard_future_csv": str(lb_future_csv),
                "analog_cases_csv": str(analog_csv),
            }
        )

    idx = pd.DataFrame(rows)
    if not idx.empty:
        idx = idx.sort_values("variable")

    idx_csv = args.datasets_dir / f"strong_model_index_split{args.split_year}_to{args.target_year}.csv"
    idx_pq = args.datasets_dir / f"strong_model_index_split{args.split_year}_to{args.target_year}.parquet"
    idx.to_csv(idx_csv, index=False)
    idx.to_parquet(idx_pq, index=False)

    patt = pd.DataFrame(pattern_rows).sort_values("variable")
    patt_csv = args.datasets_dir / f"pattern_report_split{args.split_year}_to{args.target_year}.csv"
    patt_pq = args.datasets_dir / f"pattern_report_split{args.split_year}_to{args.target_year}.parquet"
    patt.to_csv(patt_csv, index=False)
    patt.to_parquet(patt_pq, index=False)

    print("Ultra model-zoo charts generated and old charts replaced.")
    print(f"Charts dir: {args.charts_dir}")
    print(f"Datasets dir: {args.datasets_dir}")
    print(f"Pattern report: {patt_csv}")
    if not idx.empty:
        print(
            idx[
                [
                    "variable",
                    "confidence_level",
                    "bridge_model",
                    "future_model",
                    "cv_mae_bridge",
                    "cv_mae_future",
                    "bridge_rows",
                    "forecast_rows",
                    "chart_png",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
