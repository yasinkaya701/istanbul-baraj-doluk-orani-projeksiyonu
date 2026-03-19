#!/usr/bin/env python3
"""Daily 1-step walk-forward holdout evaluation for Istanbul overall dam occupancy.

Evaluation setup:
- Train end: 2020-12-31
- Test: 2021-01-01+
- Target: overall_mean (daily mean of all dams)

This script is for short-horizon operational accuracy (1-day ahead), not
for long-horizon monthly projection quality.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_istanbul_dam_daily_holdout"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import ElasticNet, Ridge

    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


DEFAULT_RESOURCE_ID = "af0b3902-cfd9-4096-85f7-e2c3017e4f21"
DEFAULT_API = "https://data.ibb.gov.tr/api/3/action/datastore_search"


@dataclass
class EvalResult:
    model: str
    rmse: float
    mae: float
    smape: float
    approx_accuracy_from_smape_pct: float
    n_test_points: int


@dataclass
class BiasSelection:
    enabled: bool
    selected_scheme: str
    selected_shrink_k: float
    selected_momentum_c: float
    selected_cv_smape: float
    cv_window_days: int
    cv_start: str
    cv_step_days: int
    cv_recency_weight: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily walk-forward holdout eval: train<=2020, test>=2021")
    p.add_argument("--history-csv", type=Path, default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/istanbul_dam_daily_history.csv"))
    p.add_argument("--api-url", default=DEFAULT_API)
    p.add_argument("--resource-id", default=DEFAULT_RESOURCE_ID)
    p.add_argument("--output-dir", type=Path, default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/holdout_2020_daily_walkforward"))
    p.add_argument("--train-end", default="2020-12-31")
    p.add_argument("--interpolate-missing-days", action="store_true", default=True)
    p.add_argument("--skip-ml", action="store_true", help="Only evaluate persistence model.")
    p.add_argument("--enable-bias-correction", action="store_true", default=True)
    p.add_argument("--bias-schemes", default="month,month_ten,week,month_dow")
    p.add_argument("--bias-shrink-grid", default="0,2,5,10,20,30,40,60")
    p.add_argument("--bias-momentum-grid", default="0,0.1,0.2,0.3,0.35,0.4")
    p.add_argument("--bias-cv-start", default="2019-01-01")
    p.add_argument("--bias-cv-window-days", type=int, default=180)
    p.add_argument("--bias-cv-step-days", type=int, default=182)
    p.add_argument("--bias-cv-recency-weight", type=float, default=0.90)
    p.add_argument("--bootstrap-samples", type=int, default=1200)
    p.add_argument("--bootstrap-block-days", type=int, default=14)
    p.add_argument("--bootstrap-seed", type=int, default=42)
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


def build_daily_history(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if "Tarih" not in df.columns:
        raise ValueError("Expected 'Tarih' column in records.")
    dam_cols = [c for c in df.columns if c not in {"_id", "Tarih"}]
    if not dam_cols:
        raise ValueError("No dam columns found in source.")

    out = pd.DataFrame()
    out["ds"] = pd.to_datetime(df["Tarih"], errors="coerce")
    for c in dam_cols:
        out[c] = df[c].map(to_numeric)
        percent_like = out[c] > 1.2
        out.loc[percent_like, c] = out.loc[percent_like, c] / 100.0
        out[c] = out[c].clip(lower=0.0, upper=1.0)
    out["overall_mean"] = out[dam_cols].mean(axis=1, skipna=True)
    out = out.dropna(subset=["ds", "overall_mean"]).sort_values("ds").drop_duplicates(subset=["ds"])
    return out[["ds", "overall_mean"]].reset_index(drop=True)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.abs(y_true) + np.abs(y_pred)
    den = np.where(den == 0, np.nan, den)
    return float(np.nanmean(2.0 * np.abs(y_pred - y_true) / den) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy().sort_values("ds").reset_index(drop=True)
    for lag in [1, 2, 3, 7, 14, 30, 365]:
        out[f"y_lag{lag}"] = out["overall_mean"].shift(lag)
    out["dow_sin"] = np.sin(2.0 * np.pi * out["ds"].dt.dayofweek / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * out["ds"].dt.dayofweek / 7.0)
    out["month_sin"] = np.sin(2.0 * np.pi * out["ds"].dt.month / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out["ds"].dt.month / 12.0)
    out["trend"] = np.arange(len(out), dtype=float)
    return out


def make_bias_key(ds: pd.Series, scheme: str) -> pd.Series:
    if scheme == "month":
        return ds.dt.month.astype(str)
    if scheme == "month_ten":
        ten = ((ds.dt.day - 1) // 10 + 1).astype(str)
        return ds.dt.month.astype(str) + "-" + ten
    if scheme == "week":
        return ds.dt.isocalendar().week.astype(str)
    if scheme == "month_dow":
        return ds.dt.month.astype(str) + "-" + ds.dt.dayofweek.astype(str)
    raise ValueError(f"Unsupported bias scheme: {scheme}")


def fit_bias_map(
    train_df: pd.DataFrame,
    scheme: str,
    shrink_k: float,
) -> tuple[dict[str, float], float]:
    tmp = train_df[["ds", "overall_mean", "y_lag1"]].dropna().copy()
    tmp["residual"] = tmp["overall_mean"] - tmp["y_lag1"]
    global_med = float(tmp["residual"].median()) if not tmp.empty else 0.0
    if tmp.empty:
        return {}, global_med
    tmp["key"] = make_bias_key(tmp["ds"], scheme=scheme)
    g = tmp.groupby("key", as_index=False)["residual"].agg(median="median", count="count")
    k = float(max(0.0, shrink_k))
    g["bias"] = (g["count"] * g["median"] + k * global_med) / (g["count"] + k)
    return dict(zip(g["key"], g["bias"])), global_med


def predict_persistence_bias(
    frame: pd.DataFrame,
    scheme: str,
    bias_map: dict[str, float],
    fallback_bias: float,
) -> np.ndarray:
    tmp = frame[["ds", "y_lag1"]].dropna().copy()
    keys = make_bias_key(tmp["ds"], scheme=scheme)
    bias = keys.map(lambda x: bias_map.get(x, fallback_bias)).to_numpy(dtype=float)
    pred = np.clip(tmp["y_lag1"].to_numpy(dtype=float) + bias, 0.0, 1.0)
    return pred


def predict_persistence_bias_momentum(
    frame: pd.DataFrame,
    scheme: str,
    bias_map: dict[str, float],
    fallback_bias: float,
    momentum_c: float,
) -> np.ndarray:
    tmp = frame[["ds", "y_lag1", "y_lag2"]].dropna().copy()
    keys = make_bias_key(tmp["ds"], scheme=scheme)
    bias = keys.map(lambda x: bias_map.get(x, fallback_bias)).to_numpy(dtype=float)
    m = tmp["y_lag1"].to_numpy(dtype=float) - tmp["y_lag2"].to_numpy(dtype=float)
    pred = np.clip(tmp["y_lag1"].to_numpy(dtype=float) + bias + float(momentum_c) * m, 0.0, 1.0)
    return pred


def select_bias_scheme_cv(
    train_df: pd.DataFrame,
    schemes: list[str],
    shrink_grid: list[float],
    momentum_grid: list[float],
    cv_start: pd.Timestamp,
    cv_window_days: int,
    cv_step_days: int,
    cv_recency_weight: float,
) -> BiasSelection:
    base = train_df[["ds", "overall_mean", "y_lag1", "y_lag2"]].dropna().copy()
    if base.empty:
        return BiasSelection(
            enabled=False,
            selected_scheme="month",
            selected_shrink_k=0.0,
            selected_momentum_c=0.0,
            selected_cv_smape=float("nan"),
            cv_window_days=int(cv_window_days),
            cv_start=str(cv_start.date()),
            cv_step_days=int(cv_step_days),
            cv_recency_weight=float(cv_recency_weight),
        )

    fold_starts: list[pd.Timestamp] = []
    t = pd.Timestamp(cv_start)
    max_t = pd.Timestamp(base["ds"].max())
    while t <= max_t:
        fold_starts.append(t)
        t = t + pd.Timedelta(days=int(max(1, cv_step_days)))

    best_obj = float("inf")
    best_scheme = schemes[0] if schemes else "month"
    best_k = 0.0
    best_c = 0.0

    for scheme in schemes:
        for k in shrink_grid:
            for c in (momentum_grid or [0.0]):
                fold_vals: list[float] = []
                fold_idx: list[int] = []
                for i, start in enumerate(fold_starts):
                    tr = base[base["ds"] < start].copy()
                    te = base[(base["ds"] >= start) & (base["ds"] < start + pd.Timedelta(days=int(max(1, cv_window_days))))].copy()
                    if len(tr) < 365 or len(te) < 30:
                        continue
                    mp, glob = fit_bias_map(tr, scheme=scheme, shrink_k=float(k))
                    p = predict_persistence_bias_momentum(
                        te,
                        scheme=scheme,
                        bias_map=mp,
                        fallback_bias=glob,
                        momentum_c=float(c),
                    )
                    y = te["overall_mean"].to_numpy(dtype=float)
                    y = y[-len(p) :]
                    fold_vals.append(smape(y, p))
                    fold_idx.append(i)

                if not fold_vals:
                    continue
                idx_arr = np.asarray(fold_idx, dtype=float)
                max_idx = float(np.max(idx_arr))
                rec = float(np.clip(cv_recency_weight, 0.0, 1.0))
                w = np.power(rec, max_idx - idx_arr)
                obj = float(np.sum(w * np.asarray(fold_vals, dtype=float)) / max(float(np.sum(w)), 1e-12))
                if obj < best_obj:
                    best_obj = obj
                    best_scheme = scheme
                    best_k = float(k)
                    best_c = float(c)

    if not np.isfinite(best_obj):
        return BiasSelection(
            enabled=False,
            selected_scheme="month",
            selected_shrink_k=0.0,
            selected_momentum_c=0.0,
            selected_cv_smape=float("nan"),
            cv_window_days=int(cv_window_days),
            cv_start=str(cv_start.date()),
            cv_step_days=int(cv_step_days),
            cv_recency_weight=float(cv_recency_weight),
        )
    return BiasSelection(
        enabled=True,
        selected_scheme=str(best_scheme),
        selected_shrink_k=float(best_k),
        selected_momentum_c=float(best_c),
        selected_cv_smape=float(best_obj),
        cv_window_days=int(cv_window_days),
        cv_start=str(cv_start.date()),
        cv_step_days=int(cv_step_days),
        cv_recency_weight=float(cv_recency_weight),
    )


def block_bootstrap_indices(
    n: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    b = int(max(1, min(block_size, n)))
    n_blocks = int(np.ceil(n / b))
    starts = rng.integers(0, max(1, n - b + 1), size=n_blocks)
    out = []
    for s in starts:
        out.extend(range(int(s), int(s) + b))
    return np.asarray(out[:n], dtype=int)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_samples: int,
    block_size: int,
    seed: int,
) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    n = len(y)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(max(1, n_samples))):
        idx = block_bootstrap_indices(n=n, block_size=block_size, rng=rng)
        vals.append(float(metric_fn(y[idx], p[idx])))
    arr = np.asarray(vals, dtype=float)
    return float(np.nanquantile(arr, 0.025)), float(np.nanmedian(arr)), float(np.nanquantile(arr, 0.975))


def bootstrap_delta_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn,
    n_samples: int,
    block_size: int,
    seed: int,
) -> tuple[float, float, float, float]:
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(y_pred_a, dtype=float)
    b = np.asarray(y_pred_b, dtype=float)
    n = len(y)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(max(1, n_samples))):
        idx = block_bootstrap_indices(n=n, block_size=block_size, rng=rng)
        va = float(metric_fn(y[idx], a[idx]))
        vb = float(metric_fn(y[idx], b[idx]))
        vals.append(vb - va)
    arr = np.asarray(vals, dtype=float)
    lo = float(np.nanquantile(arr, 0.025))
    med = float(np.nanmedian(arr))
    hi = float(np.nanquantile(arr, 0.975))
    prob_a_better = float(np.mean(arr > 0.0))
    return lo, med, hi, prob_a_better


def evaluate_persistence(test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tmp = test_df[["ds", "overall_mean", "y_lag1"]].dropna().copy()
    ds = pd.to_datetime(tmp["ds"]).to_numpy()
    y_true = tmp["overall_mean"].to_numpy(dtype=float)
    y_pred = np.clip(tmp["y_lag1"].to_numpy(dtype=float), 0.0, 1.0)
    return ds, y_true, y_pred


def evaluate_walkforward_ml(
    feat_df: pd.DataFrame,
    train_end: pd.Timestamp,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat_cols = [c for c in feat_df.columns if c.startswith("y_lag") or c.endswith("_sin") or c.endswith("_cos") or c == "trend"]
    test_days = feat_df.loc[feat_df["ds"] > train_end, "ds"].to_list()
    preds: list[float] = []
    actuals: list[float] = []
    dss: list[pd.Timestamp] = []

    for day in test_days:
        hist = feat_df[feat_df["ds"] < day][["overall_mean", *feat_cols]].dropna()
        row = feat_df[feat_df["ds"] == day][["overall_mean", *feat_cols]].dropna()
        if row.empty or len(hist) < 100:
            continue

        x_train = hist[feat_cols].to_numpy(dtype=float)
        y_train = hist["overall_mean"].to_numpy(dtype=float)
        x_test = row[feat_cols].to_numpy(dtype=float)
        y_test = row["overall_mean"].to_numpy(dtype=float)

        if model_name == "ridge":
            model = Ridge(alpha=2.0)
        elif model_name == "enet":
            model = ElasticNet(alpha=0.0002, l1_ratio=0.2, max_iter=50000)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model.fit(x_train, y_train)
        pred = float(np.clip(model.predict(x_test)[0], 0.0, 1.0))
        preds.append(pred)
        actuals.append(float(y_test[0]))
        dss.append(pd.Timestamp(day))

    return np.asarray(dss), np.asarray(actuals, dtype=float), np.asarray(preds, dtype=float)


def to_result(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> EvalResult:
    s = smape(y_true, y_pred)
    return EvalResult(
        model=model_name,
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        smape=s,
        approx_accuracy_from_smape_pct=float(100.0 - s),
        n_test_points=int(len(y_true)),
    )


def save_plot(train: pd.DataFrame, test: pd.DataFrame, pred_best: pd.DataFrame, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12.0, 5.8))
    plt.plot(train["ds"], train["overall_mean"] * 100.0, color="#1d4ed8", linewidth=1.2, label="Egitim (<=2020)")
    plt.plot(test["ds"], test["overall_mean"] * 100.0, color="#111827", linewidth=1.0, alpha=0.75, label="Gercek (2021+)")
    plt.plot(pred_best["ds"], pred_best["yhat"] * 100.0, color="#dc2626", linewidth=1.2, label="En iyi model (1-gun)")
    plt.axvline(pd.Timestamp("2021-01-01"), color="#6b7280", linestyle=":", linewidth=1.0)
    plt.text(pd.Timestamp("2021-01-01"), 99, "  Test baslangici", fontsize=9, color="#374151", va="top")
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.xlabel("Tarih")
    plt.ylabel("Doluluk (%)")
    plt.title("Istanbul Baraj Doluluk - Gunluk 1-Adim Holdout Testi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    daily: pd.DataFrame
    if args.history_csv.exists():
        daily = pd.read_csv(args.history_csv, parse_dates=["ds"])
        if "overall_mean" not in daily.columns:
            raise SystemExit(f"Missing overall_mean column in {args.history_csv}")
        daily = daily[["ds", "overall_mean"]].dropna().sort_values("ds").drop_duplicates(subset=["ds"])
    else:
        records = fetch_records(api_url=args.api_url, resource_id=args.resource_id)
        daily = build_daily_history(records)
        daily.to_csv(args.history_csv, index=False)

    if args.interpolate_missing_days:
        idx = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
        daily = daily.set_index("ds").reindex(idx)
        daily["overall_mean"] = daily["overall_mean"].interpolate(limit_direction="both")
        daily = daily.clip(lower=0.0, upper=1.0)
        daily.index.name = "ds"
        daily = daily.reset_index()

    train_end = pd.Timestamp(args.train_end)
    train = daily[daily["ds"] <= train_end][["ds", "overall_mean"]].copy()
    test = daily[daily["ds"] > train_end][["ds", "overall_mean"]].copy()
    if train.empty or test.empty:
        raise SystemExit("Train/test split is empty. Check --train-end or data coverage.")

    feat = add_features(daily)

    model_preds: dict[str, pd.DataFrame] = {}
    results: list[EvalResult] = []
    bias_sel = BiasSelection(
        enabled=False,
        selected_scheme="month",
        selected_shrink_k=0.0,
        selected_momentum_c=0.0,
        selected_cv_smape=float("nan"),
        cv_window_days=int(args.bias_cv_window_days),
        cv_start=str(pd.Timestamp(args.bias_cv_start).date()),
        cv_step_days=int(args.bias_cv_step_days),
        cv_recency_weight=float(args.bias_cv_recency_weight),
    )

    ds_p, y_p, p_p = evaluate_persistence(feat[feat["ds"] > train_end])
    pred_p = pd.DataFrame({"ds": pd.to_datetime(ds_p), "actual": y_p, "yhat": p_p, "model": "persistence_lag1"})
    model_preds["persistence_lag1"] = pred_p
    results.append(to_result("persistence_lag1", y_p, p_p))

    if bool(args.enable_bias_correction):
        schemes = [s.strip() for s in str(args.bias_schemes).split(",") if s.strip()]
        shrink_grid = [float(x.strip()) for x in str(args.bias_shrink_grid).split(",") if x.strip()]
        momentum_grid = [float(x.strip()) for x in str(args.bias_momentum_grid).split(",") if x.strip()]
        bias_sel = select_bias_scheme_cv(
            train_df=feat[feat["ds"] <= train_end][["ds", "overall_mean", "y_lag1", "y_lag2"]].copy(),
            schemes=schemes or ["month"],
            shrink_grid=shrink_grid or [0.0],
            momentum_grid=momentum_grid or [0.0],
            cv_start=pd.Timestamp(args.bias_cv_start),
            cv_window_days=int(args.bias_cv_window_days),
            cv_step_days=int(args.bias_cv_step_days),
            cv_recency_weight=float(args.bias_cv_recency_weight),
        )
        if bias_sel.enabled:
            tr_for_bias = feat[feat["ds"] <= train_end][["ds", "overall_mean", "y_lag1", "y_lag2"]].copy()
            te_for_bias = feat[feat["ds"] > train_end][["ds", "overall_mean", "y_lag1", "y_lag2"]].dropna().copy()
            mp, glob = fit_bias_map(
                train_df=tr_for_bias,
                scheme=bias_sel.selected_scheme,
                shrink_k=float(bias_sel.selected_shrink_k),
            )
            p_b = predict_persistence_bias_momentum(
                frame=te_for_bias,
                scheme=bias_sel.selected_scheme,
                bias_map=mp,
                fallback_bias=glob,
                momentum_c=float(bias_sel.selected_momentum_c),
            )
            y_b = te_for_bias["overall_mean"].to_numpy(dtype=float)
            ds_b = te_for_bias["ds"].to_numpy()
            pred_b = pd.DataFrame(
                {
                    "ds": pd.to_datetime(ds_b),
                    "actual": y_b,
                    "yhat": p_b,
                    "model": "persistence_bias_cv",
                }
            )
            model_preds["persistence_bias_cv"] = pred_b
            results.append(to_result("persistence_bias_cv", y_b, p_b))

    if SKLEARN_OK and not args.skip_ml:
        for name in ["ridge", "enet"]:
            ds_m, y_m, p_m = evaluate_walkforward_ml(feat_df=feat, train_end=train_end, model_name=name)
            pred_m = pd.DataFrame({"ds": pd.to_datetime(ds_m), "actual": y_m, "yhat": p_m, "model": name})
            if not pred_m.empty:
                model_preds[name] = pred_m
                results.append(to_result(name, y_m, p_m))

    if not results:
        raise SystemExit("No models produced valid test predictions.")

    results_sorted = sorted(results, key=lambda r: (r.smape, r.rmse))
    best = results_sorted[0]
    pred_best = model_preds[best.model].copy().sort_values("ds").reset_index(drop=True)

    all_pred = pd.concat([v for v in model_preds.values()], ignore_index=True).sort_values(["model", "ds"]).reset_index(drop=True)
    all_pred.to_csv(args.output_dir / "overall_mean_daily_holdout_predictions_2021_plus.csv", index=False)

    metrics_json = {
        "target_series": "overall_mean",
        "frequency": "daily",
        "forecast_mode": "one_step_walkforward",
        "train_end": str(train_end.date()),
        "test_start": str(test["ds"].min().date()),
        "test_end": str(test["ds"].max().date()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "models": [
            {
                "model": r.model,
                "rmse": r.rmse,
                "mae": r.mae,
                "smape": r.smape,
                "approx_accuracy_from_smape_pct": r.approx_accuracy_from_smape_pct,
                "n_test_points": r.n_test_points,
            }
            for r in results_sorted
        ],
        "best_model": {
            "model": best.model,
            "rmse": best.rmse,
            "mae": best.mae,
            "smape": best.smape,
            "approx_accuracy_from_smape_pct": best.approx_accuracy_from_smape_pct,
            "n_test_points": best.n_test_points,
        },
        "bias_correction_selection": {
            "enabled": bool(bias_sel.enabled),
            "selected_scheme": str(bias_sel.selected_scheme),
            "selected_shrink_k": float(bias_sel.selected_shrink_k),
            "selected_momentum_c": float(bias_sel.selected_momentum_c),
            "selected_cv_smape": float(bias_sel.selected_cv_smape) if np.isfinite(bias_sel.selected_cv_smape) else None,
            "cv_window_days": int(bias_sel.cv_window_days),
            "cv_start": str(bias_sel.cv_start),
            "cv_step_days": int(bias_sel.cv_step_days),
            "cv_recency_weight": float(bias_sel.cv_recency_weight),
        },
        "note": "Bu sonuc 1-gun ileri walk-forward operasyonel dogrulugudur; aylik uzun-ufuk projeksiyonla birebir karsilastirilmamali.",
    }

    base_pred = model_preds.get("persistence_lag1")
    if base_pred is not None and not base_pred.empty and not pred_best.empty:
        ab = (
            pred_best[["ds", "actual", "yhat"]]
            .rename(columns={"yhat": "yhat_best"})
            .merge(
                base_pred[["ds", "yhat"]].rename(columns={"yhat": "yhat_base"}),
                on="ds",
                how="inner",
            )
            .dropna()
            .reset_index(drop=True)
        )
        if not ab.empty:
            y = ab["actual"].to_numpy(dtype=float)
            p_best = ab["yhat_best"].to_numpy(dtype=float)
            p_base = ab["yhat_base"].to_numpy(dtype=float)
            smape_best_ci = bootstrap_ci(
                y_true=y,
                y_pred=p_best,
                metric_fn=smape,
                n_samples=int(args.bootstrap_samples),
                block_size=int(args.bootstrap_block_days),
                seed=int(args.bootstrap_seed),
            )
            smape_base_ci = bootstrap_ci(
                y_true=y,
                y_pred=p_base,
                metric_fn=smape,
                n_samples=int(args.bootstrap_samples),
                block_size=int(args.bootstrap_block_days),
                seed=int(args.bootstrap_seed) + 1,
            )
            delta_ci = bootstrap_delta_ci(
                y_true=y,
                y_pred_a=p_best,
                y_pred_b=p_base,
                metric_fn=smape,
                n_samples=int(args.bootstrap_samples),
                block_size=int(args.bootstrap_block_days),
                seed=int(args.bootstrap_seed) + 2,
            )
            mae_best_ci = bootstrap_ci(
                y_true=y,
                y_pred=p_best,
                metric_fn=mae,
                n_samples=int(args.bootstrap_samples),
                block_size=int(args.bootstrap_block_days),
                seed=int(args.bootstrap_seed) + 3,
            )
            mae_base_ci = bootstrap_ci(
                y_true=y,
                y_pred=p_base,
                metric_fn=mae,
                n_samples=int(args.bootstrap_samples),
                block_size=int(args.bootstrap_block_days),
                seed=int(args.bootstrap_seed) + 4,
            )
            mae_delta_ci = bootstrap_delta_ci(
                y_true=y,
                y_pred_a=p_best,
                y_pred_b=p_base,
                metric_fn=mae,
                n_samples=int(args.bootstrap_samples),
                block_size=int(args.bootstrap_block_days),
                seed=int(args.bootstrap_seed) + 5,
            )
            metrics_json["statistical_validation"] = {
                "baseline_model": "persistence_lag1",
                "comparison_model": str(best.model),
                "bootstrap_samples": int(args.bootstrap_samples),
                "bootstrap_block_days": int(args.bootstrap_block_days),
                "smape_ci95_comparison_model": {
                    "low": float(smape_best_ci[0]),
                    "median": float(smape_best_ci[1]),
                    "high": float(smape_best_ci[2]),
                },
                "smape_ci95_baseline_model": {
                    "low": float(smape_base_ci[0]),
                    "median": float(smape_base_ci[1]),
                    "high": float(smape_base_ci[2]),
                },
                "smape_improvement_vs_baseline_ci95": {
                    "low": float(delta_ci[0]),
                    "median": float(delta_ci[1]),
                    "high": float(delta_ci[2]),
                },
                "prob_comparison_better_than_baseline": float(delta_ci[3]),
                "mae_ci95_comparison_model": {
                    "low": float(mae_best_ci[0]),
                    "median": float(mae_best_ci[1]),
                    "high": float(mae_best_ci[2]),
                },
                "mae_ci95_baseline_model": {
                    "low": float(mae_base_ci[0]),
                    "median": float(mae_base_ci[1]),
                    "high": float(mae_base_ci[2]),
                },
                "mae_improvement_vs_baseline_ci95": {
                    "low": float(mae_delta_ci[0]),
                    "median": float(mae_delta_ci[1]),
                    "high": float(mae_delta_ci[2]),
                },
                "prob_mae_comparison_better_than_baseline": float(mae_delta_ci[3]),
            }
    (args.output_dir / "overall_mean_daily_holdout_metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Istanbul Baraj Gunluk Holdout (Train<=2020, Test>=2021)")
    lines.append("")
    lines.append(f"- Egitim sonu: `{metrics_json['train_end']}`")
    lines.append(f"- Test donemi: `{metrics_json['test_start']}` -> `{metrics_json['test_end']}`")
    lines.append(f"- En iyi model: `{best.model}`")
    lines.append(f"- sMAPE: `{best.smape:.3f}`")
    lines.append(f"- Yaklasik dogruluk (100-sMAPE): `%{best.approx_accuracy_from_smape_pct:.2f}`")
    lines.append(
        f"- Bias duzeltme secimi (yalniz train CV): "
        f"`{metrics_json['bias_correction_selection']['selected_scheme']}`, "
        f"`k={metrics_json['bias_correction_selection']['selected_shrink_k']}`, "
        f"`momentum_c={metrics_json['bias_correction_selection']['selected_momentum_c']}`"
    )
    if "statistical_validation" in metrics_json:
        sv = metrics_json["statistical_validation"]
        lines.append(
            f"- Bootstrap guven: `P({sv['comparison_model']} < {sv['baseline_model']})`"
            f" = `{sv['prob_comparison_better_than_baseline']:.3f}`"
        )
        lines.append(
            f"- sMAPE iyilesme %95 GA (baseline - model): "
            f"`[{sv['smape_improvement_vs_baseline_ci95']['low']:.4f}, "
            f"{sv['smape_improvement_vs_baseline_ci95']['high']:.4f}]`"
        )
        lines.append(
            f"- MAE iyilesme %95 GA (baseline - model): "
            f"`[{sv['mae_improvement_vs_baseline_ci95']['low']:.6f}, "
            f"{sv['mae_improvement_vs_baseline_ci95']['high']:.6f}]`"
        )
    lines.append("")
    lines.append("## Tum Modeller")
    lines.append("")
    tbl = pd.DataFrame(
        [
            {
                "model": r.model,
                "rmse": r.rmse,
                "mae": r.mae,
                "smape": r.smape,
                "accuracy_pct": r.approx_accuracy_from_smape_pct,
                "n_test_points": r.n_test_points,
            }
            for r in results_sorted
        ]
    )
    lines.append(tbl.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("Not: Bu test, her gun bir sonraki gunu tahmin eden walk-forward kurulumdur.")
    (args.output_dir / "overall_mean_daily_holdout_report.md").write_text("\n".join(lines), encoding="utf-8")

    save_plot(train=train, test=test, pred_best=pred_best, out_png=args.output_dir / "overall_mean_daily_holdout_plot.png")

    print(args.output_dir / "overall_mean_daily_holdout_metrics.json")
    print(args.output_dir / "overall_mean_daily_holdout_predictions_2021_plus.csv")
    print(args.output_dir / "overall_mean_daily_holdout_plot.png")


if __name__ == "__main__":
    main()
