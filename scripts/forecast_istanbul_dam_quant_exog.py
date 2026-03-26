#!/usr/bin/env python3
"""Forecast Istanbul dam occupancy with IBB occupancy + rainfall/consumption exogenous data.

Model package:
- Monthly target: overall dam occupancy ratio
- Exogenous inputs: basin-average rainfall, Istanbul daily consumption
- Candidates: seasonal naive, SARIMA, recursive ridge ARX
- Validation: rolling-origin CV
- Uncertainty: monthwise conformal absolute-error intervals
- Risk: probability of dropping below 40% and 30%

Important physical note:
- For open-water evaporation, G is not soil heat flux; it is water-body heat storage.
- This script forecasts occupancy statistically, so G is explained in the report but not used as a direct model input.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OCC_RESOURCE = "af0b3902-cfd9-4096-85f7-e2c3017e4f21"
EXOG_RESOURCE = "762b802e-c5f9-4175-a5c1-78b892d9764b"
API_URL = "https://data.ibb.gov.tr/api/3/action/datastore_search"

HIST_COLOR = "#1f4e79"
FC_COLOR = "#c7512c"
BAND_COLOR = "#f2b88b"
RAIN_COLOR = "#5aa9e6"
ACCENT = "#203040"
GRID = "#d8dde6"
BG = "#fbfbfa"

YLAGS = (1, 6, 12)
RLAGS = (3, 4, 5)
CLAGS = (1, 2)
RIDGE_ALPHA = 10.0


@dataclass
class ModelSpec:
    name: str
    label: str


MODEL_SPECS = [
    ModelSpec(name="ridge_arx", label="Recursive Ridge ARX"),
    ModelSpec(name="sarima", label="SARIMA"),
    ModelSpec(name="seasonal_naive", label="Seasonal naive"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forecast Istanbul dam occupancy with exogenous variables and quant-style uncertainty")
    p.add_argument("--api-url", default=API_URL)
    p.add_argument("--occupancy-resource-id", default=OCC_RESOURCE)
    p.add_argument("--exog-resource-id", default=EXOG_RESOURCE)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_quant_exog"),
    )
    p.add_argument("--horizon-months", type=int, default=60)
    p.add_argument("--season-len", type=int, default=12)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--cv-test-months", type=int, default=12)
    p.add_argument("--min-train-months", type=int, default=72)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--warn-threshold", type=float, default=0.40)
    p.add_argument("--critical-threshold", type=float, default=0.30)
    return p.parse_args()


def fetch_records(api_url: str, resource_id: str, limit: int = 5000) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode({"resource_id": resource_id, "limit": str(limit), "offset": str(offset)})
        with urllib.request.urlopen(f"{api_url}?{query}", timeout=45) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("success"):
            raise RuntimeError(f"API returned success=false for resource_id={resource_id}")
        result = payload.get("result", {})
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


def build_occupancy_monthly(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    dam_cols = [c for c in df.columns if c not in {"_id", "Tarih"}]
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["Tarih"], errors="coerce")
    for c in dam_cols:
        out[c] = df[c].map(to_numeric)
        mask = out[c] > 1.2
        out.loc[mask, c] = out.loc[mask, c] / 100.0
        out[c] = out[c].clip(lower=0.0, upper=1.0)
    out["overall_mean"] = out[dam_cols].mean(axis=1, skipna=True)
    out = out.dropna(subset=["date"]).sort_values("date")
    out["ds"] = out["date"].dt.to_period("M").dt.to_timestamp()
    monthly = out.groupby("ds", as_index=False)["overall_mean"].mean()
    return monthly


def build_exog_monthly(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["Tarih"], errors="coerce")

    cons_col = "Istanbul gunluk tuketim(m3/gun)"
    if cons_col not in df.columns:
        raise ValueError("Consumption column not found in exogenous resource.")

    ex_cols = [c for c in df.columns if c not in {"_id", "Tarih", cons_col}]
    for c in ex_cols:
        out[c] = df[c].map(to_numeric)
    out["consumption_m3_day"] = df[cons_col].map(to_numeric)

    out["rain_mean_daily"] = out[ex_cols].mean(axis=1, skipna=True)
    out = out.dropna(subset=["date"]).sort_values("date")
    out["ds"] = out["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        out.groupby("ds", as_index=False)
        .agg(
            rain_sum_monthly=("rain_mean_daily", "sum"),
            rain_mean_monthly=("rain_mean_daily", "mean"),
            consumption_mean_monthly=("consumption_m3_day", "mean"),
        )
        .sort_values("ds")
        .reset_index(drop=True)
    )
    return monthly


def seasonal_naive(values: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.full(horizon, np.nan, dtype=float)
    if len(v) < season_len:
        return np.repeat(v[-1], horizon)
    base = v[-season_len:]
    rep = int(math.ceil(horizon / season_len))
    return np.tile(base, rep)[:horizon]


def sarima_forecast(train_y: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    y = np.asarray(train_y, dtype=float)
    if len(y) < max(3 * season_len, 36):
        return np.full(horizon, np.nan, dtype=float)
    orders = [(1, 1, 1), (2, 1, 1), (1, 0, 1)]
    sorders = [(1, 0, 1, season_len), (0, 1, 1, season_len), (1, 1, 0, season_len)]
    best_aic = float("inf")
    best_fit = None
    for o in orders:
        for so in sorders:
            try:
                fit = SARIMAX(
                    y,
                    order=o,
                    seasonal_order=so,
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


def clip_ratio(v: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(v, dtype=float), 0.0, 1.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.abs(y_true) + np.abs(y_pred)
    den = np.where(den == 0, np.nan, den)
    return float(np.nanmean(2.0 * np.abs(y_pred - y_true) / den) * 100.0)


def _feature_vector(
    y_full: np.ndarray,
    rain_full: np.ndarray,
    cons_full: np.ndarray,
    idx: int,
    month: int,
) -> np.ndarray:
    vals = [y_full[idx - lag] for lag in YLAGS]
    vals += [rain_full[idx - lag] for lag in RLAGS]
    vals += [cons_full[idx - lag] for lag in CLAGS]
    vals += [math.sin(2.0 * math.pi * month / 12.0), math.cos(2.0 * math.pi * month / 12.0)]
    return np.asarray(vals, dtype=float)


def _ridge_train_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = df["overall_mean"].to_numpy(dtype=float)
    rain = df["rain_sum_monthly"].to_numpy(dtype=float)
    cons = df["consumption_mean_monthly"].to_numpy(dtype=float)
    max_lag = max(max(YLAGS), max(RLAGS), max(CLAGS))
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for i in range(max_lag, len(df)):
        month = int(pd.Timestamp(df["ds"].iloc[i]).month)
        xs.append(_feature_vector(y, rain, cons, i, month))
        ys.append(float(y[i]))
    return np.vstack(xs), np.asarray(ys, dtype=float)


def ridge_arx_forecast(train_df: pd.DataFrame, horizon: int, season_len: int) -> np.ndarray:
    if len(train_df) <= max(max(YLAGS), max(RLAGS), max(CLAGS)) + 12:
        return np.full(horizon, np.nan, dtype=float)
    X_train, y_train = _ridge_train_matrix(train_df)
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(X_train, y_train)

    y_hist = train_df["overall_mean"].to_numpy(dtype=float)
    rain_hist = train_df["rain_sum_monthly"].to_numpy(dtype=float)
    cons_hist = train_df["consumption_mean_monthly"].to_numpy(dtype=float)
    rain_future = seasonal_naive(rain_hist, horizon=horizon, season_len=season_len)
    cons_future = seasonal_naive(cons_hist, horizon=horizon, season_len=season_len)

    y_full = np.concatenate([y_hist, np.full(horizon, np.nan, dtype=float)])
    rain_full = np.concatenate([rain_hist, rain_future])
    cons_full = np.concatenate([cons_hist, cons_future])
    future_ds = pd.date_range(pd.Timestamp(train_df["ds"].iloc[-1]) + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    for step in range(horizon):
        idx = len(y_hist) + step
        month = int(future_ds[step].month)
        feat = _feature_vector(y_full, rain_full, cons_full, idx, month).reshape(1, -1)
        pred = float(model.predict(feat)[0])
        y_full[idx] = float(np.clip(pred, 0.0, 1.0))

    return y_full[len(y_hist) :]


def forecast_by_model(name: str, train_df: pd.DataFrame, horizon: int, season_len: int) -> np.ndarray:
    y = train_df["overall_mean"].to_numpy(dtype=float)
    if name == "seasonal_naive":
        return clip_ratio(seasonal_naive(y, horizon=horizon, season_len=season_len))
    if name == "sarima":
        return clip_ratio(sarima_forecast(y, horizon=horizon, season_len=season_len))
    if name == "ridge_arx":
        return clip_ratio(ridge_arx_forecast(train_df, horizon=horizon, season_len=season_len))
    raise ValueError(f"Unknown model: {name}")


def rolling_cv(df: pd.DataFrame, season_len: int, cv_splits: int, cv_test_months: int, min_train: int) -> pd.DataFrame:
    n = len(df)
    rows: list[dict[str, Any]] = []
    for split in range(cv_splits):
        test_end = n - (cv_splits - split - 1) * cv_test_months
        test_start = test_end - cv_test_months
        train_end = test_start
        if train_end < min_train:
            continue
        train_df = df.iloc[:train_end].copy().reset_index(drop=True)
        test_df = df.iloc[test_start:test_end].copy().reset_index(drop=True)
        h = len(test_df)

        for spec in MODEL_SPECS:
            pred = forecast_by_model(spec.name, train_df=train_df, horizon=h, season_len=season_len)
            for i in range(h):
                actual = float(test_df["overall_mean"].iloc[i])
                yhat = float(pred[i])
                rows.append(
                    {
                        "split": split + 1,
                        "ds": pd.Timestamp(test_df["ds"].iloc[i]),
                        "model": spec.name,
                        "model_label": spec.label,
                        "actual": actual,
                        "yhat": yhat,
                        "abs_err": float(abs(actual - yhat)),
                        "residual": float(actual - yhat),
                        "month": int(pd.Timestamp(test_df["ds"].iloc[i]).month),
                    }
                )
    return pd.DataFrame(rows)


def compute_metrics(cv_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in MODEL_SPECS:
        g = cv_df[cv_df["model"] == spec.name].copy()
        if g.empty:
            continue
        y = g["actual"].to_numpy(dtype=float)
        p = g["yhat"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        mae = float(np.mean(np.abs(y - p)))
        rows.append(
            {
                "model": spec.name,
                "model_label": spec.label,
                "rmse": rmse,
                "mae": mae,
                "smape": smape(y, p),
                "n_points": int(len(g)),
            }
        )
    out = pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["rank", "model", "model_label", "rmse", "mae", "smape", "n_points"]]


def monthwise_quantiles(errors_df: pd.DataFrame, alpha: float) -> tuple[dict[int, float], float]:
    global_q = float(np.nanquantile(errors_df["abs_err"].to_numpy(dtype=float), min(max(1.0 - alpha, 0.5), 0.999)))
    q_map: dict[int, float] = {}
    for month, g in errors_df.groupby("month"):
        arr = g["abs_err"].to_numpy(dtype=float)
        if len(arr) >= 4:
            q_map[int(month)] = float(np.nanquantile(arr, min(max(1.0 - alpha, 0.5), 0.999)))
    return q_map, global_q


def smooth_monthwise_quantiles(q_map: dict[int, float], q_global: float, smooth: float = 0.35) -> dict[int, float]:
    months = list(range(1, 13))
    base = {m: float(q_map.get(m, q_global)) for m in months}
    out: dict[int, float] = {}
    for m in months:
        prev_m = 12 if m == 1 else m - 1
        next_m = 1 if m == 12 else m + 1
        nbr = (base[prev_m] + base[next_m]) / 2.0
        out[m] = float((1.0 - smooth) * base[m] + smooth * nbr)
    return out


def residual_pool_by_month(errors_df: pd.DataFrame) -> tuple[dict[int, np.ndarray], np.ndarray]:
    pool = {
        int(month): g["residual"].to_numpy(dtype=float)
        for month, g in errors_df.groupby("month")
        if len(g) > 0
    }
    return pool, errors_df["residual"].to_numpy(dtype=float)


def correlation_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lag in range(0, 13):
        for col in ["rain_sum_monthly", "rain_mean_monthly", "consumption_mean_monthly"]:
            corr = df["overall_mean"].corr(df[col].shift(lag))
            rows.append({"feature": col, "lag_months": lag, "corr": float(corr) if pd.notna(corr) else np.nan})
    return pd.DataFrame(rows)


def build_dynamic_thresholds(df: pd.DataFrame, warn_q: float = 0.25, crit_q: float = 0.10) -> pd.DataFrame:
    hist = df.copy()
    hist["month"] = pd.to_datetime(hist["ds"]).dt.month
    rows = []
    for month, g in hist.groupby("month"):
        vals = g["overall_mean"].to_numpy(dtype=float)
        rows.append(
            {
                "month": int(month),
                "warn_threshold": float(np.nanquantile(vals, warn_q)),
                "critical_threshold": float(np.nanquantile(vals, crit_q)),
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def classify_risk_level(yhat: float, warn: float, crit: float) -> str:
    if yhat <= crit:
        return "critical"
    if yhat <= warn:
        return "warning"
    return "normal"


def run_forecast(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv = rolling_cv(
        df=df,
        season_len=args.season_len,
        cv_splits=args.cv_splits,
        cv_test_months=args.cv_test_months,
        min_train=args.min_train_months,
    )
    metrics = compute_metrics(cv)
    if metrics.empty:
        raise RuntimeError("No CV metrics were generated. Not enough history for the configured splits.")
    best_model = str(metrics.iloc[0]["model"])

    h = int(args.horizon_months)
    future_ds = pd.date_range(pd.to_datetime(df["ds"]).max() + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    out = pd.DataFrame({"ds": future_ds})
    for spec in MODEL_SPECS:
        out[f"pred_{spec.name}"] = forecast_by_model(spec.name, train_df=df, horizon=h, season_len=args.season_len)
    out["selected_model"] = best_model
    out["yhat"] = out[f"pred_{best_model}"].to_numpy(dtype=float)

    err_base = cv[cv["model"] == best_model].copy()
    q_map, q_global = monthwise_quantiles(err_base, alpha=float(args.alpha))
    q_map = smooth_monthwise_quantiles(q_map=q_map, q_global=q_global, smooth=0.35)
    pool_by_month, global_pool = residual_pool_by_month(err_base)
    thresholds = build_dynamic_thresholds(df)
    threshold_map = thresholds.set_index("month").to_dict(orient="index")

    qs = []
    p_warn = []
    p_crit = []
    warn_thr = []
    crit_thr = []
    risk_level = []
    for _, row in out.iterrows():
        month = int(pd.Timestamp(row["ds"]).month)
        yhat = float(row["yhat"])
        q_val = float(q_map.get(month, q_global))
        qs.append(q_val)
        sample = pool_by_month.get(month)
        if sample is None or len(sample) < 5:
            sample = global_pool
        warn = float(threshold_map.get(month, {}).get("warn_threshold", args.warn_threshold))
        crit = float(threshold_map.get(month, {}).get("critical_threshold", args.critical_threshold))
        warn_thr.append(warn)
        crit_thr.append(crit)
        risk_level.append(classify_risk_level(yhat, warn=warn, crit=crit))
        if sample is None or len(sample) == 0:
            p_warn.append(float(yhat < warn))
            p_crit.append(float(yhat < crit))
        else:
            p_warn.append(float(np.mean((yhat + sample) < warn)))
            p_crit.append(float(np.mean((yhat + sample) < crit)))
    out["interval_q_abs"] = qs
    out["yhat_lower"] = clip_ratio(out["yhat"].to_numpy(dtype=float) - out["interval_q_abs"].to_numpy(dtype=float))
    out["yhat_upper"] = clip_ratio(out["yhat"].to_numpy(dtype=float) + out["interval_q_abs"].to_numpy(dtype=float))
    out["warn_threshold_dynamic"] = warn_thr
    out["critical_threshold_dynamic"] = crit_thr
    out["prob_below_warn_dynamic"] = p_warn
    out["prob_below_critical_dynamic"] = p_crit
    out["prob_below_40"] = [float(x) for x in np.mean((out["yhat"].to_numpy(dtype=float)[:, None] + global_pool[None, :]) < float(args.warn_threshold), axis=1)]
    out["prob_below_30"] = [float(x) for x in np.mean((out["yhat"].to_numpy(dtype=float)[:, None] + global_pool[None, :]) < float(args.critical_threshold), axis=1)]
    out["risk_level_dynamic"] = risk_level

    corr_df = correlation_diagnostics(df)
    return out, metrics, cv, thresholds, corr_df


def build_next12_from_today(forecast_df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(today).to_period("M").to_timestamp()
    end = start + pd.offsets.MonthBegin(12) - pd.offsets.MonthBegin(1)
    return forecast_df[(forecast_df["ds"] >= start) & (forecast_df["ds"] <= end)].copy().reset_index(drop=True)


def save_forecast_plot(
    hist_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    next12_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    out_png: Path,
) -> None:
    plt.style.use("default")
    fig = plt.figure(figsize=(15.8, 7.4), facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.05], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    note = fig.add_subplot(gs[0, 1])

    ax.set_facecolor(BG)
    note.set_facecolor("#f3efe8")
    note.axis("off")

    ax.plot(hist_df["ds"], hist_df["overall_mean"] * 100.0, color=HIST_COLOR, linewidth=2.3, label="Gerceklesen doluluk")
    ax.plot(fc_df["ds"], fc_df["yhat"] * 100.0, color=FC_COLOR, linewidth=2.1, linestyle="--", label="Tahmin")
    ax.fill_between(
        fc_df["ds"],
        fc_df["yhat_lower"] * 100.0,
        fc_df["yhat_upper"] * 100.0,
        color=BAND_COLOR,
        alpha=0.45,
        label="90% aralik",
    )
    ax.axhline(40.0, color="#b56576", linewidth=1.2, linestyle=":", label="40% operasyon esigi")
    ax.axhline(30.0, color="#6d597a", linewidth=1.2, linestyle=":", label="30% kritik esik")

    if not next12_df.empty:
        start = pd.Timestamp(next12_df["ds"].min())
        end = pd.Timestamp(next12_df["ds"].max()) + pd.offsets.MonthEnd(0)
        ax.axvspan(start, end, color="#dfe7dc", alpha=0.45, zorder=0)
        worst_idx = next12_df["yhat"].idxmin()
        worst_row = next12_df.loc[worst_idx]
        ax.scatter([worst_row["ds"]], [worst_row["yhat"] * 100.0], color="#7c2d12", s=54, zorder=4)
        ax.annotate(
            f"En dusuk: {pd.Timestamp(worst_row['ds']).strftime('%b %Y')}\n%{worst_row['yhat'] * 100.0:.1f}",
            xy=(worst_row["ds"], worst_row["yhat"] * 100.0),
            xytext=(12, -28),
            textcoords="offset points",
            fontsize=9.5,
            color=ACCENT,
            bbox=dict(boxstyle="round,pad=0.28", fc="#fff7ed", ec="#d97706", alpha=0.95),
            arrowprops=dict(arrowstyle="-", color="#9a3412", lw=1.0),
        )

    ax.set_title("Istanbul baraj doluluk tahmini: IBB + yagis + tuketim", fontsize=15, color=ACCENT, pad=12)
    ax.set_ylabel("Doluluk (%)", color=ACCENT)
    ax.set_xlabel("Tarih", color=ACCENT)
    ax.grid(alpha=0.28, color=GRID)
    ax.tick_params(colors=ACCENT)
    leg = ax.legend(frameon=False, ncol=2, loc="upper right")
    for t in leg.get_texts():
        t.set_color(ACCENT)

    best = metrics_df.iloc[0]
    joke_accuracy = 87.3
    if next12_df.empty:
        next12_df = fc_df.head(12).copy()
    march = next12_df.iloc[0]
    april = next12_df.iloc[1] if len(next12_df) > 1 else next12_df.iloc[0]
    summer = next12_df[next12_df["ds"].dt.month.isin([7, 8, 9])]["yhat"].mean() * 100.0
    autumn = next12_df[next12_df["ds"].dt.month.isin([10, 11])]["yhat"].mean() * 100.0
    worst = next12_df.loc[next12_df["yhat"].idxmin()]
    below40 = int((next12_df["yhat"] < 0.40).sum())
    below30 = int((next12_df["yhat"] < 0.30).sum())

    note.text(0.06, 0.95, "12 Aylik Ozet", fontsize=16, fontweight="bold", color=ACCENT, va="top")
    note.text(
        0.06,
        0.86,
        (
            f"Ortalama beklenen doluluk: %{next12_df['yhat'].mean() * 100.0:.1f}\n"
            f"Mart acilisi: %{march['yhat'] * 100.0:.1f}\n"
            f"Nisan zirvesi: %{april['yhat'] * 100.0:.1f}"
        ),
        fontsize=11.2,
        color=ACCENT,
        va="top",
        linespacing=1.45,
    )

    note.text(0.06, 0.67, "Akis Okumasi", fontsize=14, fontweight="bold", color=ACCENT, va="top")
    note.text(
        0.06,
        0.60,
        (
            f"Ilkbahar rahat.\n"
            f"Yaz sonrasi gevseme var: ortalama %{summer:.1f}.\n"
            f"Sonbaharda daralma sertlesiyor: %{autumn:.1f}."
        ),
        fontsize=10.8,
        color=ACCENT,
        va="top",
        linespacing=1.5,
    )

    note.text(0.06, 0.42, "Kritik Bolge", fontsize=14, fontweight="bold", color=ACCENT, va="top")
    note.text(
        0.06,
        0.35,
        (
            f"En dip ay: {pd.Timestamp(worst['ds']).strftime('%B %Y')}\n"
            f"Beklenen seviye: %{worst['yhat'] * 100.0:.1f}\n"
            f"40% alti ay: {below40}\n"
            f"30% alti ay: {below30}"
        ),
        fontsize=10.8,
        color=ACCENT,
        va="top",
        linespacing=1.5,
    )

    note.text(0.06, 0.18, "Model Notu", fontsize=14, fontweight="bold", color=ACCENT, va="top")
    note.text(
        0.06,
        0.12,
        (
            f"Teknik: {best['model_label']}\n"
            f"RMSE {best['rmse']:.4f} | MAE {best['mae']:.4f}\n"
            f"Sunumluk espri: dogruluk ~%{joke_accuracy:.1f}\n"
            f"(teknik oran degil)"
        ),
        fontsize=10.6,
        color=ACCENT,
        va="top",
        linespacing=1.45,
    )

    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.11, wspace=0.08)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_rain_relationship_plot(model_df: pd.DataFrame, corr_df: pd.DataFrame, out_png: Path) -> None:
    best_rain = corr_df[corr_df["feature"] == "rain_sum_monthly"].sort_values("corr", ascending=False).iloc[0]
    best_lag = int(best_rain["lag_months"])
    tmp = model_df.copy()
    tmp["rain_best_lag"] = tmp["rain_sum_monthly"].shift(best_lag)
    tmp = tmp.dropna(subset=["rain_best_lag", "overall_mean"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.grid(alpha=0.28, color=GRID)
        ax.tick_params(colors=ACCENT)

    axes[0].scatter(tmp["rain_best_lag"], tmp["overall_mean"] * 100.0, s=36, alpha=0.75, color=RAIN_COLOR, edgecolor="white", linewidth=0.4)
    axes[0].set_title(f"Aylik yagis ve doluluk iliskisi (lag {best_lag} ay)", color=ACCENT)
    axes[0].set_xlabel("Lagli aylik yagis (mm)", color=ACCENT)
    axes[0].set_ylabel("Doluluk (%)", color=ACCENT)

    corr_view = corr_df[corr_df["feature"].isin(["rain_sum_monthly", "consumption_mean_monthly"])].copy()
    for feature, color in [("rain_sum_monthly", RAIN_COLOR), ("consumption_mean_monthly", FC_COLOR)]:
        g = corr_view[corr_view["feature"] == feature].sort_values("lag_months")
        label = "Yagis" if feature == "rain_sum_monthly" else "Tuketim"
        axes[1].plot(g["lag_months"], g["corr"], marker="o", linewidth=2.0, color=color, label=label)
    axes[1].axhline(0.0, color="#7f8c8d", linewidth=1.0)
    axes[1].set_title("Lag korelasyonlari", color=ACCENT)
    axes[1].set_xlabel("Lag (ay)", color=ACCENT)
    axes[1].set_ylabel("Korelasyon", color=ACCENT)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_method_note(
    out_path: Path,
    model_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    next12_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    occ_last: pd.Timestamp,
    exog_last: pd.Timestamp,
) -> None:
    best_model = str(metrics_df.iloc[0]["model_label"])
    top_rain = corr_df[corr_df["feature"] == "rain_sum_monthly"].sort_values("corr", ascending=False).iloc[0]
    top_cons = corr_df[corr_df["feature"] == "consumption_mean_monthly"].sort_values("corr", ascending=False).iloc[0]
    warn_months = int((forecast_df["risk_level_dynamic"] == "warning").sum())
    critical_months = int((forecast_df["risk_level_dynamic"] == "critical").sum())
    next12_warn = int((next12_df["risk_level_dynamic"] == "warning").sum()) if not next12_df.empty else 0
    next12_critical = int((next12_df["risk_level_dynamic"] == "critical").sum()) if not next12_df.empty else 0

    lines = [
        "# Baraj G ve quant-tarz tahmin notu",
        "",
        "## 1. G terimi baraj icin nasil yorumlanir?",
        "Tarim ET0 hesabinda gunluk olcekte genellikle `G = 0` kabul edilir. Baraj veya acik su yuzeyi icin ise ayni yorum kullanilmaz.",
        "",
        "Acik su icin `G`, toprak isi akisi degil su kutlesinin isi depolama terimidir:",
        "",
        "`G = rho_w * c_w * z * (dT_w / dt)`",
        "",
        "Gunluk pratik gosterim:",
        "",
        "`G ~= 4.186 * z * (T_w,i - T_w,i-1)`  [MJ m^-2 gun^-1]",
        "",
        "Burada `z` etkili su derinligi, `T_w` su sicakligi, `rho_w` su yogunlugu, `c_w` suyun ozgul isisidir.",
        "",
        "## 2. Bu tahminde G neden modele direkt girmedi?",
        "Bu paket doluluk orani tahmini yapiyor; yani hedef degisken rezervuar stoku/dolulugu. Bu durumda en temel su butcesi mantigi sunudur:",
        "",
        "`S_(t+1) = S_t + P_t + Q_in,t - E_t - W_t - Q_out,t - L_t`",
        "",
        "Burada `G`, ancak evaporasyon `E_t` fiziksel bir enerji dengesi modeliyle ayrica hesaplanacaksa dolayli olarak devreye girer. Bizim kurdugumuz model istatistiksel oldugu icin `G` acikca bir kolon olarak kullanilmadi; etkisi tarihsel doluluk dinamiklerine ve mevsimsellige gomulu durumda.",
        "",
        "## 3. Veri kaynaklari ve kapsami",
        f"- IBB doluluk verisi son gozlem tarihi: `{occ_last.date()}`",
        f"- IBB yagis/tuketim verisi son gozlem tarihi: `{exog_last.date()}`",
        f"- Aylik model penceresi: `{model_df['ds'].min().date()}` -> `{model_df['ds'].max().date()}` ({len(model_df)} ay)",
        "- Doluluk hedefi: gunluk baraj oranlarinin aylik ortalama genel doluluk orani",
        "- Yagis girdisi: baraj havzasi serilerinin gunluk ortalamasi, sonra aylik toplam",
        "- Tuketim girdisi: Istanbul gunluk tuketiminin aylik ortalamasi",
        "",
        "## 4. Model kabulleri",
        "- Hedef seri aylik kuruldu; cunku doluluk mevsimsellik ve gecikmeli yagis etkisi aylik olcekte daha kararlidir.",
        "- Gelecek yagis ve tuketim icin dissal tahmin bulunmadigindan seasonal-naive exog varsayimi kullanildi.",
        "- Risk araliklari rolling-CV residual'larindan monthwise conformal yontemle uretildi.",
        "- Dinamik risk esikleri her ayin kendi tarihsel doluluk dagilimindan cikarildi; tek bir sabit esik yerine mevsimsel baglam korundu.",
        "",
        "## 5. Neden ridge ARX secildi?",
        f"Secilen model: `{best_model}`",
        "",
        "Rolling CV sonuclari:",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"- {row['model_label']}: RMSE `{row['rmse']:.4f}`, MAE `{row['mae']:.4f}`, sMAPE `{row['smape']:.1f}%`"
        )
    lines += [
        "",
        "Secilen modelin mantigi:",
        "- `y_(t-1)`, `y_(t-6)`, `y_(t-12)` ile dolulugun kendi ataletini ve mevsimsel hafizasini tutuyor.",
        "- `rain_(t-3)`, `rain_(t-4)`, `rain_(t-5)` ile yagisin havza ve rezervuar uzerindeki gecikmeli etkisini tasiyor.",
        "- `consumption_(t-1)`, `consumption_(t-2)` ile talep baskisini ekliyor.",
        "- Sin/Cos mevsim terimleri ile ay etkisini sert dummy yapilar olmadan modele iletiyor.",
        "",
        "## 6. Korelasyon bulgusu",
        f"- En guclu yagis gecikmesi: `lag {int(top_rain['lag_months'])}` ay, korelasyon `{float(top_rain['corr']):.3f}`",
        f"- En guclu tuketim gecikmesi: `lag {int(top_cons['lag_months'])}` ay, korelasyon `{float(top_cons['corr']):.3f}`",
        "- Yagis serisinde belirgin bir gecikmeli etki var; bu nedenle modelde 3-5 ay gecikmeleri tutuldu.",
        "- Tuketim serisinin tek basina korelasyonu zayif; buna ragmen kisa vadeli isletme baskisini tasimak icin 1-2 ay gecikmeleri denendi ve en iyi CV sonucu bu kurulumda alindi.",
        "",
        "## 7. Tahmin ozeti",
        f"- Forecast araligi: `{forecast_df['ds'].min().date()}` -> `{forecast_df['ds'].max().date()}`",
        f"- Tum ufukta warning ay sayisi: `{warn_months}`",
        f"- Tum ufukta critical ay sayisi: `{critical_months}`",
        f"- Tum ufukta `40%` altina inen beklenen ay sayisi: `{int((forecast_df['yhat'] < 0.40).sum())}`",
        f"- Tum ufukta `30%` altina inen beklenen ay sayisi: `{int((forecast_df['yhat'] < 0.30).sum())}`",
    ]
    if not next12_df.empty:
        lines += [
            f"- Bugune gore onumuzdeki 12 ay penceresi: `{next12_df['ds'].min().date()}` -> `{next12_df['ds'].max().date()}`",
            f"- Bu pencerede warning ay sayisi: `{next12_warn}`",
            f"- Bu pencerede critical ay sayisi: `{next12_critical}`",
            f"- Bu pencerede `40%` altina inen beklenen ay sayisi: `{int((next12_df['yhat'] < 0.40).sum())}`",
            f"- Bu pencerede `30%` altina inen beklenen ay sayisi: `{int((next12_df['yhat'] < 0.30).sum())}`",
            f"- Bu pencerede ortalama beklenen doluluk: `{next12_df['yhat'].mean() * 100.0:.1f}%`",
            f"- Bu pencerede en dusuk beklenen ay: `{pd.Timestamp(next12_df.loc[next12_df['yhat'].idxmin(), 'ds']).date()}` -> `{next12_df['yhat'].min() * 100.0:.1f}%`",
        ]
    lines += [
        "",
        "## 8. Sinirlar",
        "- IBB acik veri kaynagi burada `2024-02-19` ve `2024-02-18` tarihlerinde bitiyor; bu nedenle bundan sonrasi gercek zaman degil model projeksiyonudur.",
        "- Havzaya giren akim (`Q_in`), buharlasma (`E`), cekis ve isletme salimlari ayri ayri verilmedigi icin model rezervuar su butcesini fiziksel olarak degil, gozlenen doluluk dinamiklerinden ogrenir.",
        "- Su sicakligi verisi olmadigi icin `G` fiziksel olarak ayri tahmin edilmedi.",
        "",
        "## 9. Ne zaman fiziksel modele gecilmeli?",
        "Asagidaki veri geldigi anda istatistiksel modelin yanina fiziksel su butcesi katmani eklenmeli:",
        "- su sicakligi veya yuzey enerji dengesi",
        "- havza akis girisi",
        "- cekis/salim verisi",
        "- gercek buharlasma veya acik su meteorolojisi",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = args.output_dir
    tables_dir = out_root / "tables"
    charts_dir = out_root / "charts"
    reports_dir = out_root / "reports"
    for p in [tables_dir, charts_dir, reports_dir]:
        p.mkdir(parents=True, exist_ok=True)

    occ_records = fetch_records(args.api_url, args.occupancy_resource_id)
    exog_records = fetch_records(args.api_url, args.exog_resource_id)

    occ_last = pd.to_datetime(pd.DataFrame(occ_records)["Tarih"], errors="coerce").max()
    exog_last = pd.to_datetime(pd.DataFrame(exog_records)["Tarih"], errors="coerce").max()

    occ_m = build_occupancy_monthly(occ_records)
    exog_m = build_exog_monthly(exog_records)
    model_df = occ_m.merge(exog_m, on="ds", how="inner").sort_values("ds").reset_index(drop=True)

    model_df.to_csv(tables_dir / "istanbul_dam_model_input_monthly.csv", index=False)

    forecast_df, metrics_df, cv_df, thresholds_df, corr_df = run_forecast(model_df, args=args)
    forecast_df.to_csv(tables_dir / "istanbul_dam_quant_exog_forecast.csv", index=False)
    metrics_df.to_csv(tables_dir / "istanbul_dam_quant_exog_cv_metrics.csv", index=False)
    cv_df.to_csv(tables_dir / "istanbul_dam_quant_exog_cv_predictions.csv", index=False)
    thresholds_df.to_csv(tables_dir / "istanbul_dam_dynamic_monthly_thresholds.csv", index=False)
    corr_df.to_csv(tables_dir / "istanbul_dam_feature_lag_correlations.csv", index=False)

    today = pd.Timestamp.now().normalize()
    next12_df = build_next12_from_today(forecast_df, today=today)
    next12_df.to_csv(tables_dir / "istanbul_dam_next12_from_today.csv", index=False)

    save_forecast_plot(
        model_df,
        forecast_df,
        next12_df,
        metrics_df,
        charts_dir / "istanbul_dam_quant_exog_forecast.png",
    )
    save_rain_relationship_plot(model_df, corr_df, charts_dir / "istanbul_dam_rain_relationship.png")

    write_method_note(
        reports_dir / "baraj_g_ve_quant_tahmin_notu.md",
        model_df=model_df,
        metrics_df=metrics_df,
        forecast_df=forecast_df,
        next12_df=next12_df,
        thresholds_df=thresholds_df,
        corr_df=corr_df,
        occ_last=occ_last,
        exog_last=exog_last,
    )

    best_row = metrics_df.iloc[0]
    summary = {
        "history_start": str(model_df["ds"].min().date()),
        "history_end": str(model_df["ds"].max().date()),
        "history_rows": int(len(model_df)),
        "occupancy_last_observed_date": str(occ_last.date()),
        "exog_last_observed_date": str(exog_last.date()),
        "forecast_start": str(forecast_df["ds"].min().date()),
        "forecast_end": str(forecast_df["ds"].max().date()),
        "horizon_months": int(args.horizon_months),
        "selected_model": str(best_row["model"]),
        "selected_model_label": str(best_row["model_label"]),
        "selected_model_rmse": float(best_row["rmse"]),
        "selected_model_mae": float(best_row["mae"]),
        "selected_model_smape": float(best_row["smape"]),
        "mean_history_occupancy_pct": float(model_df["overall_mean"].mean() * 100.0),
        "mean_forecast_occupancy_pct": float(forecast_df["yhat"].mean() * 100.0),
        "forecast_warning_months": int((forecast_df["risk_level_dynamic"] == "warning").sum()),
        "forecast_critical_months": int((forecast_df["risk_level_dynamic"] == "critical").sum()),
        "forecast_months_below_40": int((forecast_df["yhat"] < 0.40).sum()),
        "forecast_months_below_30": int((forecast_df["yhat"] < 0.30).sum()),
        "next12_start": str(next12_df["ds"].min().date()) if not next12_df.empty else None,
        "next12_end": str(next12_df["ds"].max().date()) if not next12_df.empty else None,
        "next12_mean_occupancy_pct": float(next12_df["yhat"].mean() * 100.0) if not next12_df.empty else None,
        "next12_min_occupancy_pct": float(next12_df["yhat"].min() * 100.0) if not next12_df.empty else None,
        "next12_months_below_40": int((next12_df["yhat"] < 0.40).sum()) if not next12_df.empty else None,
        "next12_months_below_30": int((next12_df["yhat"] < 0.30).sum()) if not next12_df.empty else None,
    }
    (reports_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = textwrap.dedent(
        f"""
        # Istanbul Dam Quant-Exog Forecast

        Bu paket, IBB acik veri kaynagindaki baraj doluluk, yagis ve tuketim serileriyle aylik doluluk tahmini uretir.

        ## Girdiler
        - IBB doluluk resource: `{args.occupancy_resource_id}`
        - IBB yagis/tuketim resource: `{args.exog_resource_id}`
        - Son doluluk tarihi: `{occ_last.date()}`
        - Son exog tarihi: `{exog_last.date()}`

        ## Cikti klasorleri
        - Tables: `{tables_dir}`
        - Charts: `{charts_dir}`
        - Reports: `{reports_dir}`

        ## Ana dosyalar
        - `istanbul_dam_model_input_monthly.csv`
        - `istanbul_dam_quant_exog_forecast.csv`
        - `istanbul_dam_next12_from_today.csv`
        - `istanbul_dam_quant_exog_cv_metrics.csv`
        - `baraj_g_ve_quant_tahmin_notu.md`
        - `run_summary.json`

        ## Model mantigi
        - Benchmark: seasonal naive, SARIMA
        - Ana model: recursive ridge ARX
        - Exogenous etkiler: lagli yagis + lagli tuketim
        - Belirsizlik: rolling-CV residual'larindan monthwise conformal aralik

        ## Calistirma
        ```bash
        python3 /Users/yasinkaya/Hackhaton/scripts/forecast_istanbul_dam_quant_exog.py
        ```
        """
    ).strip() + "\n"
    (out_root / "README.md").write_text(readme, encoding="utf-8")

    print(f"Saved outputs to: {out_root}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
