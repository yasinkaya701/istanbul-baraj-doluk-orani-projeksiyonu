#!/usr/bin/env python3
"""Decision-support forecast for Istanbul dam occupancy.

Outputs:
- Forecasts for each dam + overall mean
- Best strategy per series (best single model vs top-k ensemble)
- Conformal intervals (month-adaptive)
- Risk probabilities: P(y < 40%), P(y < 30%)
- Next-12-month risk ranking table
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_istanbul_dam_decision"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DEFAULT_RESOURCE_ID = "af0b3902-cfd9-4096-85f7-e2c3017e4f21"
DEFAULT_API = "https://data.ibb.gov.tr/api/3/action/datastore_search"


@dataclass
class ModelStat:
    model: str
    rmse: float
    mae: float
    smape: float
    n_points: int
    weight: float
    rmse_recent: float
    rmse_std: float
    bias_abs: float
    score: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Istanbul dams decision support forecast")
    p.add_argument("--resource-id", default=DEFAULT_RESOURCE_ID)
    p.add_argument("--api-url", default=DEFAULT_API)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision"),
    )
    p.add_argument("--horizon-months", type=int, default=60)
    p.add_argument("--season-len", type=int, default=12)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--cv-test-months", type=int, default=12)
    p.add_argument("--min-train-months", type=int, default=72)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--ensemble-max-models", type=int, default=3)
    p.add_argument(
        "--ensemble-tie-margin",
        type=float,
        default=0.01,
        help="Ensemble score'u tek model score'una bu oranda yakin ise ensemble secilir.",
    )
    p.add_argument(
        "--ensemble-shrink",
        type=float,
        default=0.00,
        help="Ensemble agirliklarini esit-agirlik karisimina ceken shrink orani [0,1].",
    )
    p.add_argument(
        "--enable-stacked-ensemble",
        dest="enable_stacked_ensemble",
        action="store_true",
        default=True,
        help="Ridge tabanli stacked ensemble adayi uret.",
    )
    p.add_argument(
        "--disable-stacked-ensemble",
        dest="enable_stacked_ensemble",
        action="store_false",
        help="Ridge tabanli stacked ensemble adayini kapat.",
    )
    p.add_argument(
        "--stack-l2",
        type=float,
        default=1.0,
        help="Stacked ensemble ridge regularizasyon katsayisi.",
    )
    p.add_argument(
        "--stack-blend-invscore",
        type=float,
        default=0.35,
        help="Stacked agirliklari inverse-score agirliklariyla karistirma orani [0,1].",
    )
    p.add_argument(
        "--stack-min-weight",
        type=float,
        default=0.0,
        help="Bu esigin altindaki stacked agirliklari sifira cek (0 ise kapali).",
    )
    p.add_argument(
        "--include-ets-damped",
        action="store_true",
        help="Damped ETS modelini aday havuza dahil et.",
    )
    p.add_argument(
        "--auto-tune-selection",
        dest="auto_tune_selection",
        action="store_true",
        default=False,
        help="CV ustunde seri-bazli model secim parametresi ayari yap.",
    )
    p.add_argument(
        "--no-auto-tune-selection",
        dest="auto_tune_selection",
        action="store_false",
        help="Otomatik model secim tuning adimini kapat.",
    )
    p.add_argument(
        "--tune-stability-lambda",
        type=float,
        default=0.10,
        help="Otomatik secimde objective = recent_rmse + lambda * split_std.",
    )
    p.add_argument(
        "--recent-split-weight",
        type=float,
        default=0.80,
        help="Gecekmis split'leri asagi agirliklamak icin [0,1] arasi katsayi. 1.0 esit agirlik.",
    )
    p.add_argument(
        "--stability-penalty",
        type=float,
        default=0.10,
        help="Split bazli hata oynakligina ceza katsayisi.",
    )
    p.add_argument(
        "--bias-penalty",
        type=float,
        default=0.00,
        help="Mutlak bias (ortalama residual) ceza katsayisi.",
    )
    p.add_argument(
        "--horizon-damping-start",
        type=int,
        default=18,
        help="Bu aydan sonra tahmin seasonal-naive'a kontrollu yaklastirilir.",
    )
    p.add_argument(
        "--horizon-damping-strength",
        type=float,
        default=0.35,
        help="Ufuk sonunda seasonal-naive'a maksimum blend orani [0,1].",
    )
    p.add_argument(
        "--interval-smoothing",
        type=float,
        default=0.35,
        help="Aylik interval quantile degerleri icin komsu ay smoothing orani [0,1].",
    )
    p.add_argument(
        "--lead-bias-correction-strength",
        type=float,
        default=0.00,
        help="CV lead-time bias duzeltmesi icin shrink orani [0,1].",
    )
    p.add_argument(
        "--lead-bias-month-strength",
        type=float,
        default=0.00,
        help="Takvim ayi bazli bias duzeltme katkisi [0,1].",
    )
    p.add_argument(
        "--lead-bias-max-abs",
        type=float,
        default=0.08,
        help="Bias duzeltmesinin mutlak ust siniri (oran puani).",
    )
    p.add_argument(
        "--lead-bias-min-samples",
        type=int,
        default=3,
        help="Lead/ay bazli bias tahmini icin minimum ornek sayisi.",
    )
    p.add_argument(
        "--seasonal-floor-threshold",
        type=float,
        default=1.01,
        help="Uzun ufukta yhat, seasonal-naive'dan margin kadar asagida bu orandan fazla ise seasonal floor uygula (0-1, >1=kapali).",
    )
    p.add_argument(
        "--seasonal-floor-margin",
        type=float,
        default=0.0,
        help="Seasonal floor tetiklenirken kullanilan minimum fark (oran puani).",
    )
    p.add_argument(
        "--seasonal-floor-min-horizon",
        type=int,
        default=24,
        help="Seasonal floor kontrolunun devreye girmesi icin minimum horizon uzunlugu (ay).",
    )
    p.add_argument("--plot-series", default="overall_mean")
    return p.parse_args()


def fetch_records(api_url: str, resource_id: str, limit: int = 5000) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode({"resource_id": resource_id, "limit": str(limit), "offset": str(offset)})
        with urllib.request.urlopen(f"{api_url}?{query}", timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("success"):
            raise RuntimeError(f"API returned success=false at offset={offset}")
        result = payload["result"]
        chunk = result.get("records", [])
        records.extend(chunk)
        total = int(result.get("total", len(records)))
        if len(records) >= total or not chunk:
            break
        offset += len(chunk)
    return records


def to_numeric(x: Any) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    return float(s.replace(",", "."))


def build_daily_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if "Tarih" not in df.columns:
        raise ValueError("Expected 'Tarih' column in records.")
    dam_cols = [c for c in df.columns if c not in {"_id", "Tarih"}]

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["Tarih"], errors="coerce")
    for c in dam_cols:
        out[c] = df[c].map(to_numeric)
        # Mixed scales exist in the source: ratio [0,1] and percent [0,100].
        mask_percent_like = out[c] > 1.2
        out.loc[mask_percent_like, c] = out.loc[mask_percent_like, c] / 100.0
        out[c] = out[c].clip(lower=0.0, upper=1.0)
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out["overall_mean"] = out[dam_cols].mean(axis=1, skipna=True)
    return out


def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    monthly = daily.copy()
    monthly["ds"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
    value_cols = [c for c in monthly.columns if c not in {"date", "ds"}]
    monthly = monthly.groupby("ds", as_index=False)[value_cols].mean()
    return monthly.sort_values("ds").reset_index(drop=True)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.abs(y_true) + np.abs(y_pred)
    den = np.where(den == 0, np.nan, den)
    return float(np.nanmean(2.0 * np.abs(y_pred - y_true) / den) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def split_rmse_stats(df: pd.DataFrame, recent_split_weight: float) -> tuple[float, float]:
    if df.empty:
        return float("nan"), float("nan")
    rows = []
    for split, g in df.groupby("split"):
        y_true = g["actual"].to_numpy(dtype=float)
        y_pred = g["yhat"].to_numpy(dtype=float)
        rows.append({"split": int(split), "rmse": rmse(y_true, y_pred)})
    tmp = pd.DataFrame(rows).sort_values("split")
    if tmp.empty:
        return float("nan"), float("nan")
    max_split = int(tmp["split"].max())
    # Most-recent split gets highest weight; older splits are geometrically downweighted.
    w = np.power(float(np.clip(recent_split_weight, 0.0, 1.0)), max_split - tmp["split"].to_numpy(dtype=float))
    wsum = float(np.sum(w))
    if wsum <= 0:
        wrmse = float(np.nanmean(tmp["rmse"].to_numpy(dtype=float)))
    else:
        wrmse = float(np.sum(w * tmp["rmse"].to_numpy(dtype=float)) / wsum)
    return wrmse, float(np.nanstd(tmp["rmse"].to_numpy(dtype=float)))


def clip_ratio(v: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(v, dtype=float), 0.0, 1.0)


def seasonal_naive_forecast(train: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    y = np.asarray(train, dtype=float)
    if y.size == 0:
        return np.full(horizon, np.nan, dtype=float)
    if y.size < season_len:
        return np.repeat(y[-1], horizon)
    base = y[-season_len:]
    rep = int(math.ceil(horizon / season_len))
    return np.tile(base, rep)[:horizon]


def ets_forecast(train: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    y = np.asarray(train, dtype=float)
    try:
        if y.size >= max(2 * season_len, 24):
            fit = ExponentialSmoothing(
                y,
                trend="add",
                seasonal="add",
                seasonal_periods=season_len,
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=False)
        elif y.size >= 8:
            fit = ExponentialSmoothing(y, trend="add", initialization_method="estimated").fit(
                optimized=True,
                use_brute=False,
            )
        else:
            return np.repeat(y[-1], horizon)
        return np.asarray(fit.forecast(horizon), dtype=float)
    except Exception:
        return seasonal_naive_forecast(y, horizon=horizon, season_len=season_len)


def ets_damped_forecast(train: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    y = np.asarray(train, dtype=float)
    try:
        if y.size >= max(2 * season_len, 24):
            fit = ExponentialSmoothing(
                y,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=season_len,
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=False)
        elif y.size >= 10:
            fit = ExponentialSmoothing(
                y,
                trend="add",
                damped_trend=True,
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=False)
        else:
            return np.repeat(y[-1], horizon)
        return np.asarray(fit.forecast(horizon), dtype=float)
    except Exception:
        return ets_forecast(y, horizon=horizon, season_len=season_len)


def sarima_forecast(train: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    y = np.asarray(train, dtype=float)
    if y.size < max(3 * season_len, 36):
        return np.full(horizon, np.nan, dtype=float)
    orders = [(1, 1, 1), (2, 1, 1), (1, 0, 1)]
    sorders = [(1, 0, 1, season_len), (0, 1, 1, season_len), (1, 1, 0, season_len)]
    best_aic = float("inf")
    best_fit = None
    for order in orders:
        for sorder in sorders:
            try:
                fit = SARIMAX(
                    y,
                    order=order,
                    seasonal_order=sorder,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best_aic = float(fit.aic)
                    best_fit = fit
            except Exception:
                continue
    if best_fit is None:
        return np.full(horizon, np.nan, dtype=float)
    return np.asarray(best_fit.get_forecast(steps=horizon).predicted_mean, dtype=float)


def model_forecasts(train: np.ndarray, horizon: int, season_len: int, include_ets_damped: bool) -> dict[str, np.ndarray]:
    preds = {
        "seasonal_naive": seasonal_naive_forecast(train, horizon=horizon, season_len=season_len),
        "ets": ets_forecast(train, horizon=horizon, season_len=season_len),
        "sarima": sarima_forecast(train, horizon=horizon, season_len=season_len),
    }
    if include_ets_damped:
        preds["ets_damped"] = ets_damped_forecast(train, horizon=horizon, season_len=season_len)
    return {k: clip_ratio(v) for k, v in preds.items()}


def rolling_backtest(
    series: pd.Series,
    season_len: int,
    cv_splits: int,
    cv_test_months: int,
    min_train_months: int,
    include_ets_damped: bool,
) -> pd.DataFrame:
    y = series.dropna().astype(float).copy()
    n = len(y)
    rows: list[dict[str, Any]] = []
    for split in range(cv_splits):
        test_end = n - (cv_splits - split - 1) * cv_test_months
        test_start = test_end - cv_test_months
        train_end = test_start
        if train_end < min_train_months:
            continue
        train = y.iloc[:train_end].to_numpy(dtype=float)
        test = y.iloc[test_start:test_end]
        preds = model_forecasts(train, horizon=len(test), season_len=season_len, include_ets_damped=include_ets_damped)
        for model_name, pred in preds.items():
            for lead_idx, (ds, actual, yhat) in enumerate(
                zip(test.index, test.to_numpy(dtype=float), pred, strict=False), start=1
            ):
                rows.append(
                    {
                        "split": split + 1,
                        "ds": pd.Timestamp(ds),
                        "lead_month": int(lead_idx),
                        "model": model_name,
                        "actual": float(actual),
                        "yhat": float(yhat),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["split", "ds", "lead_month", "model", "actual", "yhat", "abs_err", "residual"])
    out["residual"] = out["actual"] - out["yhat"]
    out["abs_err"] = np.abs(out["residual"])
    return out.sort_values(["split", "model", "ds"]).reset_index(drop=True)


def model_score(rmse_recent: float, rmse_std: float, bias_abs: float, stability_penalty: float, bias_penalty: float) -> float:
    if not np.isfinite(rmse_recent):
        return float("inf")
    rel_var = float(rmse_std / max(rmse_recent, 1e-8)) if np.isfinite(rmse_std) else 0.0
    score = rmse_recent * (1.0 + max(0.0, float(stability_penalty)) * max(0.0, rel_var))
    score += max(0.0, float(bias_penalty)) * max(0.0, float(bias_abs))
    return float(score)


def model_stats(
    cv_df: pd.DataFrame,
    recent_split_weight: float,
    stability_penalty: float,
    bias_penalty: float,
) -> tuple[list[ModelStat], dict[str, float]]:
    if cv_df.empty:
        return [], {}
    stats: list[ModelStat] = []
    for model_name, g in cv_df.groupby("model"):
        y_true = g["actual"].to_numpy(dtype=float)
        y_pred = g["yhat"].to_numpy(dtype=float)
        rmse_recent, rmse_std = split_rmse_stats(g, recent_split_weight=recent_split_weight)
        bias_abs = float(np.abs(g["residual"].mean()))
        score = model_score(
            rmse_recent=rmse_recent if np.isfinite(rmse_recent) else rmse(y_true, y_pred),
            rmse_std=rmse_std if np.isfinite(rmse_std) else 0.0,
            bias_abs=bias_abs,
            stability_penalty=stability_penalty,
            bias_penalty=bias_penalty,
        )
        st = ModelStat(
            model=model_name,
            rmse=rmse(y_true, y_pred),
            mae=float(np.mean(np.abs(y_true - y_pred))),
            smape=smape(y_true, y_pred),
            n_points=len(g),
            weight=0.0,
            rmse_recent=float(rmse_recent if np.isfinite(rmse_recent) else rmse(y_true, y_pred)),
            rmse_std=float(rmse_std if np.isfinite(rmse_std) else 0.0),
            bias_abs=bias_abs,
            score=score,
        )
        stats.append(st)
    stats = sorted(stats, key=lambda s: (s.score, s.rmse))
    inv = np.array([1.0 / max(s.score, 1e-8) for s in stats], dtype=float)
    inv = inv / inv.sum()
    for i, s in enumerate(stats):
        s.weight = float(inv[i])
    return stats, {s.model: s.weight for s in stats}


def oof_ensemble(
    cv_df: pd.DataFrame,
    stats: list[ModelStat],
    top_k: int,
    ensemble_shrink: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if cv_df.empty or not stats:
        return pd.DataFrame(columns=["split", "ds", "actual", "yhat", "abs_err", "residual"]), {}
    selected = stats[: max(1, top_k)]
    raw_weights = {s.model: s.weight for s in selected}
    norm = sum(raw_weights.values())
    weights = {k: v / max(norm, 1e-8) for k, v in raw_weights.items()}
    if weights:
        # Stabilize ensemble weights by shrinking toward equal weights.
        shrink = float(np.clip(ensemble_shrink, 0.0, 1.0))
        uniform = 1.0 / len(weights)
        weights = {k: (1.0 - shrink) * float(v) + shrink * uniform for k, v in weights.items()}
        s = sum(weights.values())
        weights = {k: float(v) / max(float(s), 1e-8) for k, v in weights.items()}

    out = oof_ensemble_from_weights(cv_df=cv_df, weights=weights)
    if out.empty:
        return out, {}
    return out, weights


def oof_ensemble_from_weights(cv_df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if cv_df.empty or not weights:
        return pd.DataFrame(columns=["split", "ds", "actual", "yhat", "abs_err", "residual"])
    idx_cols = ["split", "ds", "actual"]
    if "lead_month" in cv_df.columns:
        idx_cols = ["split", "lead_month", "ds", "actual"]
    piv = cv_df.pivot_table(index=idx_cols, columns="model", values="yhat").reset_index()
    models = [m for m in weights if m in piv.columns]
    if not models:
        return pd.DataFrame(columns=["split", "ds", "actual", "yhat", "abs_err", "residual"])
    pred = np.zeros(len(piv), dtype=float)
    wsum = 0.0
    for m in models:
        w = float(weights[m])
        pred += w * piv[m].to_numpy(dtype=float)
        wsum += w
    base_cols = ["split", "ds", "actual"]
    if "lead_month" in piv.columns:
        base_cols = ["split", "lead_month", "ds", "actual"]
    out = piv[base_cols].copy()
    out["yhat"] = clip_ratio(pred / max(wsum, 1e-8))
    out["residual"] = out["actual"] - out["yhat"]
    out["abs_err"] = np.abs(out["residual"])
    return out


def stacked_weights_ridge(
    cv_df: pd.DataFrame,
    inv_weights: dict[str, float],
    l2: float,
    blend_invscore: float,
    min_weight: float,
) -> dict[str, float]:
    if cv_df.empty or not inv_weights:
        return {}
    piv = cv_df.pivot_table(index=["split", "ds", "actual"], columns="model", values="yhat").reset_index()
    models = [m for m in inv_weights if m in piv.columns]
    if not models:
        return {}

    x = piv[models].to_numpy(dtype=float)
    y = piv["actual"].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    if len(y) < max(8, len(models) + 2):
        return {m: float(inv_weights[m]) for m in models}

    lam = max(0.0, float(l2))
    a = x.T @ x + lam * np.eye(len(models), dtype=float)
    b = x.T @ y
    try:
        w = np.linalg.solve(a, b)
    except Exception:
        w = np.linalg.lstsq(a, b, rcond=None)[0]
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    if float(np.sum(w)) <= 1e-12:
        w = np.array([float(inv_weights[m]) for m in models], dtype=float)
    w = w / max(float(np.sum(w)), 1e-12)

    if min_weight > 0.0:
        thr = float(max(0.0, min_weight))
        w = np.where(w >= thr, w, 0.0)
        if float(np.sum(w)) <= 1e-12:
            w = np.array([float(inv_weights[m]) for m in models], dtype=float)
        w = w / max(float(np.sum(w)), 1e-12)

    inv = np.array([float(inv_weights[m]) for m in models], dtype=float)
    inv = inv / max(float(np.sum(inv)), 1e-12)
    blend = float(np.clip(blend_invscore, 0.0, 1.0))
    w = (1.0 - blend) * w + blend * inv
    w = w / max(float(np.sum(w)), 1e-12)
    return {m: float(v) for m, v in zip(models, w, strict=False)}


def ensemble_oof_score(
    ens_oof: pd.DataFrame,
    recent_split_weight: float,
    stability_penalty: float,
    bias_penalty: float,
) -> tuple[float, float]:
    if ens_oof.empty:
        return float("inf"), float("inf")
    ens_rmse = rmse(ens_oof["actual"].to_numpy(dtype=float), ens_oof["yhat"].to_numpy(dtype=float))
    ens_rmse_recent, ens_rmse_std = split_rmse_stats(ens_oof, recent_split_weight=recent_split_weight)
    ens_bias_abs = float(np.abs(ens_oof["residual"].mean()))
    ens_score = model_score(
        rmse_recent=ens_rmse_recent if np.isfinite(ens_rmse_recent) else ens_rmse,
        rmse_std=ens_rmse_std if np.isfinite(ens_rmse_std) else 0.0,
        bias_abs=ens_bias_abs,
        stability_penalty=stability_penalty,
        bias_penalty=bias_penalty,
    )
    return float(ens_rmse), float(ens_score)


def choose_strategy(
    stats: list[ModelStat],
    ens_oof: pd.DataFrame,
    recent_split_weight: float,
    stability_penalty: float,
    bias_penalty: float,
    ensemble_tie_margin: float = 0.03,
) -> tuple[str, float, float]:
    if not stats:
        return "none", float("nan"), float("nan")
    best_single_model = stats[0].model
    best_single_rmse = stats[0].rmse
    best_single_score = stats[0].score
    if ens_oof.empty:
        return best_single_model, best_single_rmse, best_single_score
    ens_rmse = rmse(ens_oof["actual"].to_numpy(dtype=float), ens_oof["yhat"].to_numpy(dtype=float))
    ens_rmse_recent, ens_rmse_std = split_rmse_stats(ens_oof, recent_split_weight=recent_split_weight)
    ens_bias_abs = float(np.abs(ens_oof["residual"].mean()))
    ens_score = model_score(
        rmse_recent=ens_rmse_recent if np.isfinite(ens_rmse_recent) else ens_rmse,
        rmse_std=ens_rmse_std if np.isfinite(ens_rmse_std) else 0.0,
        bias_abs=ens_bias_abs,
        stability_penalty=stability_penalty,
        bias_penalty=bias_penalty,
    )
    if ens_score <= best_single_score * (1.0 + max(0.0, float(ensemble_tie_margin))):
        return "ensemble_topk", ens_rmse, ens_score
    return best_single_model, best_single_rmse, best_single_score


def evaluate_strategy_combo(
    cv_df: pd.DataFrame,
    ensemble_max_models: int,
    ensemble_shrink: float,
    enable_stacked_ensemble: bool,
    stack_l2: float,
    stack_blend_invscore: float,
    stack_min_weight: float,
    recent_split_weight: float,
    stability_penalty: float,
    bias_penalty: float,
    ensemble_tie_margin: float,
    tune_stability_lambda: float,
) -> dict[str, Any]:
    stats, _ = model_stats(
        cv_df,
        recent_split_weight=recent_split_weight,
        stability_penalty=stability_penalty,
        bias_penalty=bias_penalty,
    )
    ens_oof, ens_weights = oof_ensemble(
        cv_df,
        stats=stats,
        top_k=ensemble_max_models,
        ensemble_shrink=ensemble_shrink,
    )

    best_single_model = stats[0].model if stats else "none"
    best_single_rmse = float(stats[0].rmse) if stats else float("inf")
    best_single_score = float(stats[0].score) if stats else float("inf")

    strategy = best_single_model
    strategy_rmse = best_single_rmse
    strategy_score = best_single_score
    selected_oof = cv_df[cv_df["model"] == best_single_model].copy() if best_single_model != "none" else pd.DataFrame()
    selected_weights: dict[str, float] = {}

    topk_rmse, topk_score = ensemble_oof_score(
        ens_oof=ens_oof,
        recent_split_weight=recent_split_weight,
        stability_penalty=stability_penalty,
        bias_penalty=bias_penalty,
    )
    if np.isfinite(topk_score) and topk_score <= best_single_score * (1.0 + max(0.0, float(ensemble_tie_margin))):
        strategy = "ensemble_topk"
        strategy_rmse = topk_rmse
        strategy_score = topk_score
        selected_oof = ens_oof.copy()
        selected_weights = dict(ens_weights)

    if enable_stacked_ensemble and ens_weights:
        stack_weights = stacked_weights_ridge(
            cv_df=cv_df,
            inv_weights=ens_weights,
            l2=stack_l2,
            blend_invscore=stack_blend_invscore,
            min_weight=stack_min_weight,
        )
        stack_oof = oof_ensemble_from_weights(cv_df=cv_df, weights=stack_weights)
        stack_rmse, stack_score = ensemble_oof_score(
            ens_oof=stack_oof,
            recent_split_weight=recent_split_weight,
            stability_penalty=stability_penalty,
            bias_penalty=bias_penalty,
        )
        score_cap = best_single_score * (1.0 + max(0.0, float(ensemble_tie_margin)))
        if np.isfinite(stack_score) and stack_score <= score_cap:
            if stack_score < strategy_score - 1e-12 or (
                abs(stack_score - strategy_score) <= 1e-12 and stack_rmse < strategy_rmse
            ):
                strategy = "ensemble_stacked"
                strategy_rmse = stack_rmse
                strategy_score = stack_score
                selected_oof = stack_oof.copy()
                selected_weights = dict(stack_weights)

    sel = selected_oof.copy()
    sel_recent, sel_std = split_rmse_stats(sel, recent_split_weight=recent_split_weight)
    if not np.isfinite(sel_recent):
        sel_recent = float(strategy_rmse if np.isfinite(strategy_rmse) else np.inf)
    if not np.isfinite(sel_std):
        sel_std = 0.0
    objective = float(sel_recent + max(0.0, float(tune_stability_lambda)) * sel_std)
    return {
        "stats": stats,
        "ens_oof": selected_oof,
        "ens_weights": selected_weights,
        "strategy": strategy,
        "strategy_rmse": float(strategy_rmse),
        "strategy_score": float(strategy_score),
        "selected_rmse_recent": float(sel_recent),
        "selected_rmse_std": float(sel_std),
        "objective": objective,
        "recent_split_weight": float(recent_split_weight),
        "stability_penalty": float(stability_penalty),
        "bias_penalty": float(bias_penalty),
        "ensemble_tie_margin": float(ensemble_tie_margin),
        "ensemble_shrink": float(ensemble_shrink),
        "stack_l2": float(stack_l2),
        "stack_blend_invscore": float(stack_blend_invscore),
        "stack_min_weight": float(stack_min_weight),
        "enable_stacked_ensemble": bool(enable_stacked_ensemble),
    }


def smooth_monthwise_quantiles(q_map: dict[int, float], q_global: float, smooth: float) -> dict[int, float]:
    if not q_map:
        return {}
    out: dict[int, float] = {}
    s = float(np.clip(smooth, 0.0, 1.0))
    for m in range(1, 13):
        curr = float(q_map.get(m, q_global))
        prev_m = 12 if m == 1 else m - 1
        next_m = 1 if m == 12 else m + 1
        prev_v = float(q_map.get(prev_m, q_global))
        next_v = float(q_map.get(next_m, q_global))
        neigh = 0.5 * (prev_v + next_v)
        out[m] = float((1.0 - s) * curr + s * neigh)
    return out


def apply_horizon_damping(
    yhat: np.ndarray,
    seasonal_naive_ref: np.ndarray,
    start_month: int,
    max_strength: float,
) -> np.ndarray:
    arr = np.asarray(yhat, dtype=float).copy()
    ref = np.asarray(seasonal_naive_ref, dtype=float)
    n = len(arr)
    if n == 0:
        return arr
    start = int(max(1, start_month))
    strength = float(np.clip(max_strength, 0.0, 1.0))
    if strength <= 0.0 or start >= n:
        return clip_ratio(arr)
    for i in range(start - 1, n):
        frac = (i - (start - 1)) / max(1, (n - start))
        lam = strength * frac
        arr[i] = (1.0 - lam) * arr[i] + lam * ref[i]
    return clip_ratio(arr)


def apply_seasonal_floor_guard(
    yhat: np.ndarray,
    seasonal_naive_ref: np.ndarray,
    threshold_ratio: float,
    margin: float,
    min_horizon: int,
) -> tuple[np.ndarray, bool, float]:
    arr = np.asarray(yhat, dtype=float).copy()
    ref = np.asarray(seasonal_naive_ref, dtype=float)
    n = len(arr)
    thr = float(threshold_ratio)
    if n == 0 or n < int(max(1, min_horizon)) or thr <= 0.0 or thr > 1.0:
        return clip_ratio(arr), False, 0.0

    m = float(max(0.0, margin))
    trigger_mask = arr < (ref - m)
    trigger_ratio = float(np.mean(trigger_mask)) if len(trigger_mask) else 0.0
    if trigger_ratio >= thr:
        arr = np.maximum(arr, ref)
        return clip_ratio(arr), True, trigger_ratio
    return clip_ratio(arr), False, trigger_ratio


def smooth_series(arr: np.ndarray, smooth: float) -> np.ndarray:
    x = np.asarray(arr, dtype=float).copy()
    if x.size <= 1:
        return x
    s = float(np.clip(smooth, 0.0, 1.0))
    out = x.copy()
    for i in range(x.size):
        prev_i = max(i - 1, 0)
        next_i = min(i + 1, x.size - 1)
        neigh = 0.5 * (x[prev_i] + x[next_i])
        out[i] = (1.0 - s) * x[i] + s * neigh
    return out


def build_lead_month_bias(
    errors_df: pd.DataFrame,
    horizon_months: int,
    min_samples: int,
    lead_strength: float,
    month_strength: float,
    max_abs: float,
) -> tuple[np.ndarray, dict[int, float]]:
    if errors_df.empty:
        return np.zeros(horizon_months, dtype=float), {}
    tmp = errors_df.copy()
    if "lead_month" not in tmp.columns:
        tmp["lead_month"] = 1
    tmp["lead_month"] = pd.to_numeric(tmp["lead_month"], errors="coerce").fillna(1).astype(int)
    tmp["month"] = pd.to_datetime(tmp["ds"]).dt.month
    tmp["residual"] = pd.to_numeric(tmp["residual"], errors="coerce")
    tmp = tmp.dropna(subset=["residual"])
    if tmp.empty:
        return np.zeros(horizon_months, dtype=float), {}

    global_bias = float(tmp["residual"].mean())
    lead_bias: dict[int, float] = {}
    for lead, g in tmp.groupby("lead_month"):
        arr = g["residual"].to_numpy(dtype=float)
        if len(arr) >= max(1, int(min_samples)):
            lead_bias[int(lead)] = float(np.median(arr))
        else:
            lead_bias[int(lead)] = global_bias

    month_bias: dict[int, float] = {}
    for month, g in tmp.groupby("month"):
        arr = g["residual"].to_numpy(dtype=float)
        if len(arr) >= max(1, int(min_samples)):
            month_bias[int(month)] = float(np.median(arr))
        else:
            month_bias[int(month)] = global_bias
    for m in range(1, 13):
        month_bias.setdefault(m, global_bias)

    # Build horizon-length lead correction and smooth it to avoid jagged paths.
    lead_curve = []
    last_known = global_bias
    for h in range(1, max(1, int(horizon_months)) + 1):
        if h in lead_bias:
            last_known = lead_bias[h]
        lead_curve.append(last_known)
    lead_curve_arr = smooth_series(np.asarray(lead_curve, dtype=float), smooth=0.35)
    lead_curve_arr = np.clip(lead_curve_arr, -float(max_abs), float(max_abs))
    lead_curve_arr *= float(np.clip(lead_strength, 0.0, 1.0))

    month_bias = {
        int(k): float(np.clip(v, -float(max_abs), float(max_abs)) * float(np.clip(month_strength, 0.0, 1.0)))
        for k, v in month_bias.items()
    }
    return lead_curve_arr, month_bias


def monthwise_q(errors_df: pd.DataFrame, alpha: float) -> tuple[dict[int, float], float]:
    if errors_df.empty:
        return {}, float("nan")
    q_global = float(np.nanquantile(errors_df["abs_err"].to_numpy(dtype=float), min(max(1.0 - alpha, 0.5), 0.999)))
    q_map: dict[int, float] = {}
    errors_df = errors_df.copy()
    errors_df["month"] = pd.to_datetime(errors_df["ds"]).dt.month
    for month, g in errors_df.groupby("month"):
        arr = g["abs_err"].dropna().to_numpy(dtype=float)
        if len(arr) >= 5:
            q_map[int(month)] = float(np.nanquantile(arr, min(max(1.0 - alpha, 0.5), 0.999)))
        else:
            q_map[int(month)] = q_global
    return q_map, q_global


def residual_pool_by_month(errors_df: pd.DataFrame) -> tuple[dict[int, np.ndarray], np.ndarray]:
    if errors_df.empty:
        return {}, np.array([], dtype=float)
    tmp = errors_df.copy()
    tmp["month"] = pd.to_datetime(tmp["ds"]).dt.month
    by_month: dict[int, np.ndarray] = {}
    for month, g in tmp.groupby("month"):
        by_month[int(month)] = g["residual"].dropna().to_numpy(dtype=float)
    global_res = tmp["residual"].dropna().to_numpy(dtype=float)
    return by_month, global_res


def forecast_series(
    ds: pd.Series,
    series: pd.Series,
    horizon_months: int,
    season_len: int,
    cv_splits: int,
    cv_test_months: int,
    min_train_months: int,
    alpha: float,
    ensemble_max_models: int,
    ensemble_tie_margin: float,
    ensemble_shrink: float,
    enable_stacked_ensemble: bool,
    stack_l2: float,
    stack_blend_invscore: float,
    stack_min_weight: float,
    include_ets_damped: bool,
    auto_tune_selection: bool,
    tune_stability_lambda: float,
    recent_split_weight: float,
    stability_penalty: float,
    bias_penalty: float,
    horizon_damping_start: int,
    horizon_damping_strength: float,
    interval_smoothing: float,
    lead_bias_correction_strength: float,
    lead_bias_month_strength: float,
    lead_bias_max_abs: float,
    lead_bias_min_samples: int,
    seasonal_floor_threshold: float,
    seasonal_floor_margin: float,
    seasonal_floor_min_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = pd.Series(series.to_numpy(dtype=float), index=pd.to_datetime(ds)).dropna().sort_index()
    cv_df = rolling_backtest(
        series=ts,
        season_len=season_len,
        cv_splits=cv_splits,
        cv_test_months=cv_test_months,
        min_train_months=min_train_months,
        include_ets_damped=include_ets_damped,
    )
    combo_rows: list[dict[str, float]] = [
        {
            "recent_split_weight": float(recent_split_weight),
            "stability_penalty": float(stability_penalty),
            "bias_penalty": float(bias_penalty),
            "ensemble_tie_margin": float(ensemble_tie_margin),
            "ensemble_shrink": float(ensemble_shrink),
            "stack_l2": float(stack_l2),
            "stack_blend_invscore": float(stack_blend_invscore),
            "stack_min_weight": float(stack_min_weight),
        }
    ]
    if auto_tune_selection:
        for rw in [0.80, 0.90, 1.00]:
            for sp in [0.00, 0.10, 0.20]:
                for bp in [0.00, 0.05]:
                    for tm in [0.00, 0.01, 0.02]:
                        for es in [0.05, 0.15, 0.25]:
                            combo_rows.append(
                                {
                                    "recent_split_weight": float(rw),
                                    "stability_penalty": float(sp),
                                    "bias_penalty": float(bp),
                                    "ensemble_tie_margin": float(tm),
                                    "ensemble_shrink": float(es),
                                    "stack_l2": float(stack_l2),
                                    "stack_blend_invscore": float(stack_blend_invscore),
                                    "stack_min_weight": float(stack_min_weight),
                                }
                            )
    # De-duplicate while preserving order.
    uniq: list[dict[str, float]] = []
    seen: set[tuple[float, float, float, float, float, float, float, float]] = set()
    for c in combo_rows:
        key = (
            round(c["recent_split_weight"], 6),
            round(c["stability_penalty"], 6),
            round(c["bias_penalty"], 6),
            round(c["ensemble_tie_margin"], 6),
            round(c["ensemble_shrink"], 6),
            round(c["stack_l2"], 6),
            round(c["stack_blend_invscore"], 6),
            round(c["stack_min_weight"], 6),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    best_eval: dict[str, Any] | None = None
    for c in uniq:
        ev = evaluate_strategy_combo(
            cv_df=cv_df,
            ensemble_max_models=ensemble_max_models,
            ensemble_shrink=c["ensemble_shrink"],
            enable_stacked_ensemble=enable_stacked_ensemble,
            stack_l2=c["stack_l2"],
            stack_blend_invscore=c["stack_blend_invscore"],
            stack_min_weight=c["stack_min_weight"],
            recent_split_weight=c["recent_split_weight"],
            stability_penalty=c["stability_penalty"],
            bias_penalty=c["bias_penalty"],
            ensemble_tie_margin=c["ensemble_tie_margin"],
            tune_stability_lambda=tune_stability_lambda,
        )
        if best_eval is None:
            best_eval = ev
            continue
        if ev["objective"] < best_eval["objective"] - 1e-12:
            best_eval = ev
        elif abs(ev["objective"] - best_eval["objective"]) <= 1e-12 and ev["strategy_rmse"] < best_eval["strategy_rmse"]:
            best_eval = ev

    assert best_eval is not None
    stats = best_eval["stats"]
    ens_oof = best_eval["ens_oof"]
    ens_weights = best_eval["ens_weights"]
    strategy = str(best_eval["strategy"])
    strategy_rmse = float(best_eval["strategy_rmse"])
    strategy_score = float(best_eval["strategy_score"])
    used_recent_split_weight = float(best_eval["recent_split_weight"])
    used_stability_penalty = float(best_eval["stability_penalty"])
    used_bias_penalty = float(best_eval["bias_penalty"])
    used_ensemble_tie_margin = float(best_eval["ensemble_tie_margin"])
    used_ensemble_shrink = float(best_eval["ensemble_shrink"])
    used_stack_l2 = float(best_eval["stack_l2"])
    used_stack_blend_invscore = float(best_eval["stack_blend_invscore"])
    used_stack_min_weight = float(best_eval["stack_min_weight"])
    used_enable_stacked_ensemble = bool(best_eval["enable_stacked_ensemble"])

    final_preds = model_forecasts(
        ts.to_numpy(dtype=float),
        horizon=horizon_months,
        season_len=season_len,
        include_ets_damped=include_ets_damped,
    )
    future_ds = pd.date_range(ts.index.max() + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")
    out = pd.DataFrame({"ds": future_ds})
    for model_name, pred in final_preds.items():
        out[f"pred_{model_name}"] = pred

    if strategy.startswith("ensemble_") and ens_weights:
        yhat = np.zeros(len(out), dtype=float)
        wsum = 0.0
        for m, w in ens_weights.items():
            col = f"pred_{m}"
            if col in out.columns:
                yhat += float(w) * out[col].to_numpy(dtype=float)
                wsum += float(w)
        out["yhat"] = clip_ratio(yhat / max(wsum, 1e-8))
        error_base = ens_oof.copy()
    else:
        fallback_model = stats[0].model if stats else "seasonal_naive"
        use_model = strategy if f"pred_{strategy}" in out.columns else fallback_model
        out["yhat"] = clip_ratio(out.get(f"pred_{use_model}", out.filter(like="pred_").mean(axis=1)).to_numpy(dtype=float))
        error_base = cv_df[cv_df["model"] == use_model].copy()

    out["yhat"] = apply_horizon_damping(
        yhat=out["yhat"].to_numpy(dtype=float),
        seasonal_naive_ref=out["pred_seasonal_naive"].to_numpy(dtype=float),
        start_month=horizon_damping_start,
        max_strength=horizon_damping_strength,
    )

    if not error_base.empty:
        lead_curve, month_bias = build_lead_month_bias(
            errors_df=error_base,
            horizon_months=len(out),
            min_samples=lead_bias_min_samples,
            lead_strength=lead_bias_correction_strength,
            month_strength=lead_bias_month_strength,
            max_abs=lead_bias_max_abs,
        )
        corr = []
        for i, ds_val in enumerate(out["ds"], start=1):
            month = int(pd.Timestamp(ds_val).month)
            lead_corr = float(lead_curve[min(i - 1, len(lead_curve) - 1)]) if len(lead_curve) else 0.0
            month_corr = float(month_bias.get(month, 0.0))
            corr.append(lead_corr + month_corr)
        out["lead_bias_correction"] = np.asarray(corr, dtype=float)
        out["yhat"] = clip_ratio(out["yhat"].to_numpy(dtype=float) + out["lead_bias_correction"].to_numpy(dtype=float))
    else:
        out["lead_bias_correction"] = 0.0

    y_guarded, floor_applied, floor_ratio = apply_seasonal_floor_guard(
        yhat=out["yhat"].to_numpy(dtype=float),
        seasonal_naive_ref=out["pred_seasonal_naive"].to_numpy(dtype=float),
        threshold_ratio=float(seasonal_floor_threshold),
        margin=float(seasonal_floor_margin),
        min_horizon=int(seasonal_floor_min_horizon),
    )
    out["yhat"] = y_guarded
    out["seasonal_floor_applied"] = int(floor_applied)
    out["seasonal_floor_trigger_ratio"] = float(floor_ratio)
    if floor_applied:
        strategy = f"{strategy}+seasonal_floor"

    if error_base.empty:
        q_val = 0.05
        out["yhat_lower"] = clip_ratio(out["yhat"].to_numpy(dtype=float) - q_val)
        out["yhat_upper"] = clip_ratio(out["yhat"].to_numpy(dtype=float) + q_val)
        out["interval_q_abs"] = q_val
        out["prob_below_40"] = (out["yhat"] < 0.40).astype(float)
        out["prob_below_30"] = (out["yhat"] < 0.30).astype(float)
    else:
        q_map, q_global = monthwise_q(error_base, alpha=alpha)
        q_map = smooth_monthwise_quantiles(q_map=q_map, q_global=q_global, smooth=interval_smoothing)
        pool_by_month, global_pool = residual_pool_by_month(error_base)

        qs = []
        p40 = []
        p30 = []
        for _, row in out.iterrows():
            month = int(pd.Timestamp(row["ds"]).month)
            yhat = float(row["yhat"])
            q = q_map.get(month, q_global)
            qs.append(q)

            sample = pool_by_month.get(month)
            if sample is None or len(sample) < 5:
                sample = global_pool
            if sample is None or len(sample) == 0:
                p40.append(float(yhat < 0.40))
                p30.append(float(yhat < 0.30))
            else:
                p40.append(float(np.mean((yhat + sample) < 0.40)))
                p30.append(float(np.mean((yhat + sample) < 0.30)))

        out["interval_q_abs"] = qs
        out["yhat_lower"] = clip_ratio(out["yhat"].to_numpy(dtype=float) - out["interval_q_abs"].to_numpy(dtype=float))
        out["yhat_upper"] = clip_ratio(out["yhat"].to_numpy(dtype=float) + out["interval_q_abs"].to_numpy(dtype=float))
        out["prob_below_40"] = p40
        out["prob_below_30"] = p30

    out["strategy"] = strategy
    out["strategy_rmse"] = strategy_rmse
    out["strategy_score"] = strategy_score
    out["selection_recent_split_weight"] = used_recent_split_weight
    out["selection_stability_penalty"] = used_stability_penalty
    out["selection_bias_penalty"] = used_bias_penalty
    out["selection_ensemble_tie_margin"] = used_ensemble_tie_margin
    out["selection_ensemble_shrink"] = used_ensemble_shrink
    out["selection_enable_stacked_ensemble"] = int(used_enable_stacked_ensemble)
    out["selection_stack_l2"] = used_stack_l2
    out["selection_stack_blend_invscore"] = used_stack_blend_invscore
    out["selection_stack_min_weight"] = used_stack_min_weight
    out["selection_lead_bias_correction_strength"] = float(lead_bias_correction_strength)
    out["selection_lead_bias_month_strength"] = float(lead_bias_month_strength)
    out["selection_lead_bias_max_abs"] = float(lead_bias_max_abs)
    out["selection_lead_bias_min_samples"] = int(lead_bias_min_samples)
    out["selection_seasonal_floor_threshold"] = float(seasonal_floor_threshold)
    out["selection_seasonal_floor_margin"] = float(seasonal_floor_margin)
    out["selection_seasonal_floor_min_horizon"] = int(seasonal_floor_min_horizon)

    stat_rows = [
        {
            "model": s.model,
            "rmse": s.rmse,
            "mae": s.mae,
            "smape": s.smape,
            "n_points": s.n_points,
            "weight": s.weight,
            "rmse_recent": s.rmse_recent,
            "rmse_split_std": s.rmse_std,
            "bias_abs": s.bias_abs,
            "score_total": s.score,
            "used_recent_split_weight": used_recent_split_weight,
            "used_stability_penalty": used_stability_penalty,
            "used_bias_penalty": used_bias_penalty,
            "used_ensemble_tie_margin": used_ensemble_tie_margin,
            "used_ensemble_shrink": used_ensemble_shrink,
            "used_enable_stacked_ensemble": int(used_enable_stacked_ensemble),
            "used_stack_l2": used_stack_l2,
            "used_stack_blend_invscore": used_stack_blend_invscore,
            "used_stack_min_weight": used_stack_min_weight,
            "used_lead_bias_correction_strength": float(lead_bias_correction_strength),
            "used_lead_bias_month_strength": float(lead_bias_month_strength),
            "used_lead_bias_max_abs": float(lead_bias_max_abs),
            "used_lead_bias_min_samples": int(lead_bias_min_samples),
            "used_seasonal_floor_threshold": float(seasonal_floor_threshold),
            "used_seasonal_floor_margin": float(seasonal_floor_margin),
            "used_seasonal_floor_min_horizon": int(seasonal_floor_min_horizon),
        }
        for s in stats
    ]
    if not ens_oof.empty:
        ens_rmse_recent, ens_rmse_std = split_rmse_stats(ens_oof, recent_split_weight=used_recent_split_weight)
        ens_bias_abs = float(np.abs(ens_oof["residual"].mean()))
        ens_score = model_score(
            rmse_recent=ens_rmse_recent if np.isfinite(ens_rmse_recent) else rmse(
                ens_oof["actual"].to_numpy(dtype=float), ens_oof["yhat"].to_numpy(dtype=float)
            ),
            rmse_std=ens_rmse_std if np.isfinite(ens_rmse_std) else 0.0,
            bias_abs=ens_bias_abs,
            stability_penalty=used_stability_penalty,
            bias_penalty=used_bias_penalty,
        )
        stat_rows.append(
            {
                "model": "ensemble_topk",
                "rmse": rmse(ens_oof["actual"].to_numpy(dtype=float), ens_oof["yhat"].to_numpy(dtype=float)),
                "mae": float(np.mean(np.abs(ens_oof["actual"].to_numpy(dtype=float) - ens_oof["yhat"].to_numpy(dtype=float)))),
                "smape": smape(ens_oof["actual"].to_numpy(dtype=float), ens_oof["yhat"].to_numpy(dtype=float)),
                "n_points": len(ens_oof),
                "weight": np.nan,
                "rmse_recent": ens_rmse_recent,
                "rmse_split_std": ens_rmse_std,
                "bias_abs": ens_bias_abs,
                "score_total": ens_score,
                "used_recent_split_weight": used_recent_split_weight,
                "used_stability_penalty": used_stability_penalty,
                "used_bias_penalty": used_bias_penalty,
                "used_ensemble_tie_margin": used_ensemble_tie_margin,
                "used_ensemble_shrink": used_ensemble_shrink,
                "used_enable_stacked_ensemble": int(used_enable_stacked_ensemble),
                "used_stack_l2": used_stack_l2,
                "used_stack_blend_invscore": used_stack_blend_invscore,
                "used_stack_min_weight": used_stack_min_weight,
                "used_lead_bias_correction_strength": float(lead_bias_correction_strength),
                "used_lead_bias_month_strength": float(lead_bias_month_strength),
                "used_lead_bias_max_abs": float(lead_bias_max_abs),
                "used_lead_bias_min_samples": int(lead_bias_min_samples),
                "used_seasonal_floor_threshold": float(seasonal_floor_threshold),
                "used_seasonal_floor_margin": float(seasonal_floor_margin),
                "used_seasonal_floor_min_horizon": int(seasonal_floor_min_horizon),
            }
        )
        if strategy == "ensemble_stacked":
            stat_rows.append(
                {
                    "model": "ensemble_stacked",
                    "rmse": rmse(ens_oof["actual"].to_numpy(dtype=float), ens_oof["yhat"].to_numpy(dtype=float)),
                    "mae": float(
                        np.mean(np.abs(ens_oof["actual"].to_numpy(dtype=float) - ens_oof["yhat"].to_numpy(dtype=float)))
                    ),
                    "smape": smape(ens_oof["actual"].to_numpy(dtype=float), ens_oof["yhat"].to_numpy(dtype=float)),
                    "n_points": len(ens_oof),
                    "weight": np.nan,
                    "rmse_recent": ens_rmse_recent,
                    "rmse_split_std": ens_rmse_std,
                    "bias_abs": ens_bias_abs,
                    "score_total": ens_score,
                    "used_recent_split_weight": used_recent_split_weight,
                    "used_stability_penalty": used_stability_penalty,
                    "used_bias_penalty": used_bias_penalty,
                    "used_ensemble_tie_margin": used_ensemble_tie_margin,
                    "used_ensemble_shrink": used_ensemble_shrink,
                    "used_enable_stacked_ensemble": int(used_enable_stacked_ensemble),
                    "used_stack_l2": used_stack_l2,
                    "used_stack_blend_invscore": used_stack_blend_invscore,
                    "used_stack_min_weight": used_stack_min_weight,
                    "used_lead_bias_correction_strength": float(lead_bias_correction_strength),
                    "used_lead_bias_month_strength": float(lead_bias_month_strength),
                    "used_lead_bias_max_abs": float(lead_bias_max_abs),
                    "used_lead_bias_min_samples": int(lead_bias_min_samples),
                    "used_seasonal_floor_threshold": float(seasonal_floor_threshold),
                    "used_seasonal_floor_margin": float(seasonal_floor_margin),
                    "used_seasonal_floor_min_horizon": int(seasonal_floor_min_horizon),
                }
            )
    stats_df = pd.DataFrame(stat_rows)
    return out, stats_df, cv_df


def save_plot(monthly: pd.DataFrame, fc: pd.DataFrame, series_name: str, out_png: Path) -> None:
    hist = monthly[["ds", series_name]].dropna().copy()
    plt.figure(figsize=(11, 5.5))
    plt.plot(hist["ds"], hist[series_name] * 100.0, color="#1f77b4", linewidth=1.8, label="Gerceklesen")
    plt.plot(fc["ds"], fc["yhat"] * 100.0, color="#d62728", linestyle="--", linewidth=1.8, label="Tahmin")
    plt.fill_between(fc["ds"], fc["yhat_lower"] * 100.0, fc["yhat_upper"] * 100.0, color="#d62728", alpha=0.15, label="90% aralik")
    plt.ylabel("Doluluk (%)")
    plt.xlabel("Tarih")
    plt.title(f"Istanbul Dam Occupancy ({series_name})")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def build_next12_risk_summary(forecast_all: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp("2026-03-01")
    end = pd.Timestamp("2027-02-01")
    tmp = forecast_all[(forecast_all["ds"] >= start) & (forecast_all["ds"] <= end)].copy()
    if tmp.empty:
        return pd.DataFrame()
    rows = []
    for series_name, g in tmp.groupby("series"):
        i_min = g["yhat"].idxmin()
        rows.append(
            {
                "series": series_name,
                "strategy": str(g["strategy"].iloc[0]),
                "months_lt40": int((g["yhat"] < 0.40).sum()),
                "months_lt30": int((g["yhat"] < 0.30).sum()),
                "mean_prob_below_40_pct": float(g["prob_below_40"].mean() * 100.0),
                "mean_prob_below_30_pct": float(g["prob_below_30"].mean() * 100.0),
                "mean_yhat_pct": float(g["yhat"].mean() * 100.0),
                "worst_month": str(pd.Timestamp(g.loc[i_min, "ds"]).date()),
                "worst_yhat_pct": float(g["yhat"].min() * 100.0),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["months_lt40", "months_lt30", "mean_prob_below_40_pct", "worst_yhat_pct"],
        ascending=[False, False, False, True],
    )
    return out.reset_index(drop=True)


def write_summary_markdown(out_dir: Path, risk_df: pd.DataFrame, summary: dict[str, Any]) -> None:
    lines = []
    lines.append("# Istanbul Baraj Tahmin Karar Destegi")
    lines.append("")
    lines.append(f"- Veri donemi: `{summary['monthly_start']}` - `{summary['monthly_end']}`")
    lines.append(f"- Tahmin donemi: `{summary['forecast_start']}` - `{summary['forecast_end']}`")
    lines.append("- Not: Son gozlem 2024-02 oldugu icin 2026 sonrasi degerler model projeksiyonudur.")
    lines.append("")
    lines.append("## 2026-03 to 2027-02 Risk Siralamasi")
    lines.append("")
    if risk_df.empty:
        lines.append("Risk tablosu olusturulamadi.")
    else:
        lines.append(
            risk_df[
                [
                    "series",
                    "strategy",
                    "months_lt40",
                    "months_lt30",
                    "mean_prob_below_40_pct",
                    "mean_prob_below_30_pct",
                    "mean_yhat_pct",
                    "worst_month",
                    "worst_yhat_pct",
                ]
            ].to_markdown(index=False, floatfmt=".1f")
        )
    (out_dir / "KARAR_DESTEK_OZETI.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    records = fetch_records(api_url=args.api_url, resource_id=args.resource_id)
    daily = build_daily_frame(records)
    monthly = aggregate_monthly(daily)
    monthly.to_csv(out / "istanbul_dam_monthly_history.csv", index=False)

    series_cols = [c for c in monthly.columns if c != "ds"]
    all_fc = []
    all_stats = []
    all_cv = []

    for series_name in series_cols:
        fc, stats, cv = forecast_series(
            ds=monthly["ds"],
            series=monthly[series_name],
            horizon_months=args.horizon_months,
            season_len=args.season_len,
            cv_splits=args.cv_splits,
            cv_test_months=args.cv_test_months,
            min_train_months=args.min_train_months,
            alpha=args.alpha,
            ensemble_max_models=args.ensemble_max_models,
            ensemble_tie_margin=args.ensemble_tie_margin,
            ensemble_shrink=args.ensemble_shrink,
            enable_stacked_ensemble=args.enable_stacked_ensemble,
            stack_l2=args.stack_l2,
            stack_blend_invscore=args.stack_blend_invscore,
            stack_min_weight=args.stack_min_weight,
            include_ets_damped=args.include_ets_damped,
            auto_tune_selection=args.auto_tune_selection,
            tune_stability_lambda=args.tune_stability_lambda,
            recent_split_weight=args.recent_split_weight,
            stability_penalty=args.stability_penalty,
            bias_penalty=args.bias_penalty,
            horizon_damping_start=args.horizon_damping_start,
            horizon_damping_strength=args.horizon_damping_strength,
            interval_smoothing=args.interval_smoothing,
            lead_bias_correction_strength=args.lead_bias_correction_strength,
            lead_bias_month_strength=args.lead_bias_month_strength,
            lead_bias_max_abs=args.lead_bias_max_abs,
            lead_bias_min_samples=args.lead_bias_min_samples,
            seasonal_floor_threshold=args.seasonal_floor_threshold,
            seasonal_floor_margin=args.seasonal_floor_margin,
            seasonal_floor_min_horizon=args.seasonal_floor_min_horizon,
        )
        fc["series"] = series_name
        stats["series"] = series_name
        cv["series"] = series_name
        all_fc.append(fc)
        all_stats.append(stats)
        all_cv.append(cv)

    forecast_all = pd.concat(all_fc, ignore_index=True)
    stats_all = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    cv_all = pd.concat(all_cv, ignore_index=True) if all_cv else pd.DataFrame()

    forecast_all.to_csv(out / "istanbul_dam_forecasts_decision.csv", index=False)
    stats_all.to_csv(out / "istanbul_dam_cv_metrics_decision.csv", index=False)
    cv_all.to_csv(out / "istanbul_dam_cv_predictions_decision.csv", index=False)

    risk_df = build_next12_risk_summary(forecast_all)
    risk_df.to_csv(out / "risk_summary_2026_03_to_2027_02.csv", index=False)

    plot_series = args.plot_series if args.plot_series in series_cols else "overall_mean"
    fc_plot = forecast_all[forecast_all["series"] == plot_series].copy()
    save_plot(monthly=monthly, fc=fc_plot, series_name=plot_series, out_png=out / f"forecast_{plot_series}.png")

    strategy_table = (
        forecast_all.groupby("series", as_index=False)
        .agg(
            strategy=("strategy", "first"),
            strategy_rmse=("strategy_rmse", "first"),
            strategy_score=("strategy_score", "first"),
            selection_recent_split_weight=("selection_recent_split_weight", "first"),
            selection_stability_penalty=("selection_stability_penalty", "first"),
            selection_bias_penalty=("selection_bias_penalty", "first"),
            selection_ensemble_tie_margin=("selection_ensemble_tie_margin", "first"),
            selection_ensemble_shrink=("selection_ensemble_shrink", "first"),
            selection_enable_stacked_ensemble=("selection_enable_stacked_ensemble", "first"),
            selection_stack_l2=("selection_stack_l2", "first"),
            selection_stack_blend_invscore=("selection_stack_blend_invscore", "first"),
            selection_stack_min_weight=("selection_stack_min_weight", "first"),
            selection_lead_bias_correction_strength=("selection_lead_bias_correction_strength", "first"),
            selection_lead_bias_month_strength=("selection_lead_bias_month_strength", "first"),
            selection_lead_bias_max_abs=("selection_lead_bias_max_abs", "first"),
            selection_lead_bias_min_samples=("selection_lead_bias_min_samples", "first"),
            selection_seasonal_floor_threshold=("selection_seasonal_floor_threshold", "first"),
            selection_seasonal_floor_margin=("selection_seasonal_floor_margin", "first"),
            selection_seasonal_floor_min_horizon=("selection_seasonal_floor_min_horizon", "first"),
        )
        .sort_values("strategy_score")
    )
    strategy_table.to_csv(out / "strategy_summary.csv", index=False)

    summary = {
        "resource_id": args.resource_id,
        "record_count_daily": int(len(daily)),
        "record_count_monthly": int(len(monthly)),
        "monthly_start": str(monthly["ds"].min().date()),
        "monthly_end": str(monthly["ds"].max().date()),
        "forecast_horizon_months": int(args.horizon_months),
        "forecast_start": str(fc_plot["ds"].min().date()) if not fc_plot.empty else None,
        "forecast_end": str(fc_plot["ds"].max().date()) if not fc_plot.empty else None,
        "alpha": float(args.alpha),
        "ensemble_max_models": int(args.ensemble_max_models),
        "ensemble_tie_margin": float(args.ensemble_tie_margin),
        "ensemble_shrink": float(args.ensemble_shrink),
        "enable_stacked_ensemble": bool(args.enable_stacked_ensemble),
        "stack_l2": float(args.stack_l2),
        "stack_blend_invscore": float(args.stack_blend_invscore),
        "stack_min_weight": float(args.stack_min_weight),
        "include_ets_damped": bool(args.include_ets_damped),
        "auto_tune_selection": bool(args.auto_tune_selection),
        "tune_stability_lambda": float(args.tune_stability_lambda),
        "recent_split_weight": float(args.recent_split_weight),
        "stability_penalty": float(args.stability_penalty),
        "bias_penalty": float(args.bias_penalty),
        "horizon_damping_start": int(args.horizon_damping_start),
        "horizon_damping_strength": float(args.horizon_damping_strength),
        "interval_smoothing": float(args.interval_smoothing),
        "lead_bias_correction_strength": float(args.lead_bias_correction_strength),
        "lead_bias_month_strength": float(args.lead_bias_month_strength),
        "lead_bias_max_abs": float(args.lead_bias_max_abs),
        "lead_bias_min_samples": int(args.lead_bias_min_samples),
        "seasonal_floor_threshold": float(args.seasonal_floor_threshold),
        "seasonal_floor_margin": float(args.seasonal_floor_margin),
        "seasonal_floor_min_horizon": int(args.seasonal_floor_min_horizon),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_markdown(out, risk_df=risk_df, summary=summary)

    print(f"Saved outputs to: {out}")
    print("Top risk rows (2026-03 to 2027-02):")
    if not risk_df.empty:
        print(risk_df.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
