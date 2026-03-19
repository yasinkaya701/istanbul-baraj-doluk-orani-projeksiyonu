#!/usr/bin/env python3
"""Retrain daily Istanbul overall dam model on latest available data and forecast forward."""

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

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_istanbul_dam_daily_latest_retrain"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RESOURCE_ID = "af0b3902-cfd9-4096-85f7-e2c3017e4f21"
DEFAULT_API = "https://data.ibb.gov.tr/api/3/action/datastore_search"


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
    p = argparse.ArgumentParser(description="Retrain daily model with latest Istanbul dam data and forecast next days")
    p.add_argument("--history-csv", type=Path, default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/istanbul_dam_daily_history.csv"))
    p.add_argument("--api-url", default=DEFAULT_API)
    p.add_argument("--resource-id", default=DEFAULT_RESOURCE_ID)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/daily_latest_retrain"),
    )
    p.add_argument("--refresh-from-api", action="store_true", default=True)
    p.add_argument("--forecast-days", type=int, default=120)
    p.add_argument("--alpha", type=float, default=0.10, help="Interval tail level. alpha=0.1 -> approx 90% interval.")
    p.add_argument("--interpolate-missing-days", action="store_true", default=True)
    p.add_argument("--bias-schemes", default="month,month_ten,week,month_dow")
    p.add_argument("--bias-shrink-grid", default="0,2,5,10,20,30,40,60")
    p.add_argument("--bias-momentum-grid", default="0,0.1,0.2,0.3,0.35,0.4")
    p.add_argument("--bias-cv-start", default="2019-01-01")
    p.add_argument("--bias-cv-window-days", type=int, default=180)
    p.add_argument("--bias-cv-step-days", type=int, default=182)
    p.add_argument("--bias-cv-recency-weight", type=float, default=0.90)
    return p.parse_args()


def to_numeric(x: Any) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    return float(s.replace(",", "."))


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


def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy().sort_values("ds").reset_index(drop=True)
    out["y_lag1"] = out["overall_mean"].shift(1)
    out["y_lag2"] = out["overall_mean"].shift(2)
    return out


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.abs(y_true) + np.abs(y_pred)
    den = np.where(den == 0, np.nan, den)
    return float(np.nanmean(2.0 * np.abs(y_pred - y_true) / den) * 100.0)


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


def fit_bias_map(train_df: pd.DataFrame, scheme: str, shrink_k: float) -> tuple[dict[str, float], float]:
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


def predict_bias_momentum_frame(
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
            for c in momentum_grid:
                fold_vals: list[float] = []
                fold_idx: list[int] = []
                for i, start in enumerate(fold_starts):
                    tr = base[base["ds"] < start].copy()
                    te = base[(base["ds"] >= start) & (base["ds"] < start + pd.Timedelta(days=int(max(1, cv_window_days))))].copy()
                    if len(tr) < 365 or len(te) < 30:
                        continue
                    mp, glob = fit_bias_map(tr, scheme=scheme, shrink_k=float(k))
                    p = predict_bias_momentum_frame(te, scheme=scheme, bias_map=mp, fallback_bias=glob, momentum_c=float(c))
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


def recursive_forecast(
    last_ds: pd.Timestamp,
    last_y1: float,
    last_y2: float,
    horizon_days: int,
    scheme: str,
    bias_map: dict[str, float],
    fallback_bias: float,
    momentum_c: float,
) -> pd.DataFrame:
    ds_list = []
    yhat_list = []
    y1 = float(last_y1)
    y2 = float(last_y2)
    for i in range(1, int(max(1, horizon_days)) + 1):
        ds = pd.Timestamp(last_ds) + pd.Timedelta(days=i)
        tmp = pd.DataFrame({"ds": [ds]})
        key = str(make_bias_key(tmp["ds"], scheme=scheme).iloc[0])
        bias = float(bias_map.get(key, fallback_bias))
        yhat = float(np.clip(y1 + bias + float(momentum_c) * (y1 - y2), 0.0, 1.0))
        ds_list.append(ds)
        yhat_list.append(yhat)
        y2 = y1
        y1 = yhat
    return pd.DataFrame({"ds": ds_list, "yhat": yhat_list})


def save_plot(hist: pd.DataFrame, fc: pd.DataFrame, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12.0, 5.8))
    tail = hist.tail(365).copy()
    plt.plot(tail["ds"], tail["overall_mean"] * 100.0, color="#1d4ed8", linewidth=1.5, label="Gercek (son 365 gun)")
    plt.plot(fc["ds"], fc["yhat"] * 100.0, color="#dc2626", linewidth=1.6, linestyle="--", label="Gunluk tahmin")
    if "yhat_lower" in fc.columns and "yhat_upper" in fc.columns:
        plt.fill_between(fc["ds"], fc["yhat_lower"] * 100.0, fc["yhat_upper"] * 100.0, color="#dc2626", alpha=0.15, label="Tahmin araligi")
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.xlabel("Tarih")
    plt.ylabel("Doluluk (%)")
    plt.title("Istanbul Baraj Doluluk - Latest Retrain Gunluk Tahmin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.refresh_from_api):
        records = fetch_records(api_url=args.api_url, resource_id=args.resource_id)
        daily = build_daily_history(records)
        args.history_csv.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(args.history_csv, index=False)
    else:
        if not args.history_csv.exists():
            raise SystemExit(f"history csv not found: {args.history_csv}")
        daily = pd.read_csv(args.history_csv, parse_dates=["ds"])
        daily = daily[["ds", "overall_mean"]].dropna().sort_values("ds").drop_duplicates(subset=["ds"])

    if bool(args.interpolate_missing_days):
        idx = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
        daily = daily.set_index("ds").reindex(idx)
        daily["overall_mean"] = daily["overall_mean"].interpolate(limit_direction="both")
        daily = daily.clip(lower=0.0, upper=1.0)
        daily.index.name = "ds"
        daily = daily.reset_index()

    feat = add_features(daily)
    base = feat[["ds", "overall_mean", "y_lag1", "y_lag2"]].dropna().copy()
    if base.empty:
        raise SystemExit("Not enough rows after lag feature generation.")

    schemes = [s.strip() for s in str(args.bias_schemes).split(",") if s.strip()]
    shrink_grid = [float(x.strip()) for x in str(args.bias_shrink_grid).split(",") if x.strip()]
    momentum_grid = [float(x.strip()) for x in str(args.bias_momentum_grid).split(",") if x.strip()]
    sel = select_bias_scheme_cv(
        train_df=base,
        schemes=schemes or ["month"],
        shrink_grid=shrink_grid or [0.0],
        momentum_grid=momentum_grid or [0.0],
        cv_start=pd.Timestamp(args.bias_cv_start),
        cv_window_days=int(args.bias_cv_window_days),
        cv_step_days=int(args.bias_cv_step_days),
        cv_recency_weight=float(args.bias_cv_recency_weight),
    )
    if not sel.enabled:
        raise SystemExit("Bias model selection failed.")

    bias_map, global_bias = fit_bias_map(base, scheme=sel.selected_scheme, shrink_k=sel.selected_shrink_k)

    ins = base.copy()
    ins["yhat"] = predict_bias_momentum_frame(
        ins,
        scheme=sel.selected_scheme,
        bias_map=bias_map,
        fallback_bias=global_bias,
        momentum_c=sel.selected_momentum_c,
    )
    ins["residual"] = ins["overall_mean"] - ins["yhat"]
    q = float(np.nanquantile(np.abs(ins["residual"].to_numpy(dtype=float)), min(max(1.0 - float(args.alpha), 0.5), 0.999)))

    last_ds = pd.Timestamp(daily["ds"].max())
    last_vals = daily["overall_mean"].to_numpy(dtype=float)
    if len(last_vals) < 2:
        raise SystemExit("Need at least 2 daily observations for momentum forecast.")
    fc = recursive_forecast(
        last_ds=last_ds,
        last_y1=float(last_vals[-1]),
        last_y2=float(last_vals[-2]),
        horizon_days=int(args.forecast_days),
        scheme=sel.selected_scheme,
        bias_map=bias_map,
        fallback_bias=global_bias,
        momentum_c=sel.selected_momentum_c,
    )
    fc["yhat_lower"] = np.clip(fc["yhat"].to_numpy(dtype=float) - q, 0.0, 1.0)
    fc["yhat_upper"] = np.clip(fc["yhat"].to_numpy(dtype=float) + q, 0.0, 1.0)
    fc["model"] = "persistence_bias_cv_latest"
    fc["selected_scheme"] = sel.selected_scheme
    fc["selected_shrink_k"] = sel.selected_shrink_k
    fc["selected_momentum_c"] = sel.selected_momentum_c
    fc["interval_abs_q"] = q
    fc.to_csv(args.output_dir / "overall_mean_daily_latest_forecast.csv", index=False)

    ins[["ds", "overall_mean", "yhat", "residual"]].to_csv(args.output_dir / "overall_mean_daily_in_sample_fit.csv", index=False)
    save_plot(daily, fc, args.output_dir / "overall_mean_daily_latest_forecast.png")

    summary = {
        "history_source": "ibb_api" if bool(args.refresh_from_api) else "history_csv",
        "history_rows": int(len(daily)),
        "history_start": str(pd.Timestamp(daily["ds"].min()).date()),
        "history_end": str(pd.Timestamp(daily["ds"].max()).date()),
        "selected_scheme": str(sel.selected_scheme),
        "selected_shrink_k": float(sel.selected_shrink_k),
        "selected_momentum_c": float(sel.selected_momentum_c),
        "selected_cv_smape": float(sel.selected_cv_smape),
        "cv_start": str(sel.cv_start),
        "cv_window_days": int(sel.cv_window_days),
        "cv_step_days": int(sel.cv_step_days),
        "cv_recency_weight": float(sel.cv_recency_weight),
        "in_sample_smape": float(smape(ins["overall_mean"].to_numpy(dtype=float), ins["yhat"].to_numpy(dtype=float))),
        "forecast_days": int(args.forecast_days),
        "forecast_start": str(pd.Timestamp(fc["ds"].min()).date()),
        "forecast_end": str(pd.Timestamp(fc["ds"].max()).date()),
        "interval_alpha": float(args.alpha),
        "interval_abs_q": float(q),
        "note": "Veri kaynagi su anda 2024-02-19 tarihine kadar guncel gorunuyor.",
    }
    (args.output_dir / "latest_retrain_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Istanbul Baraj Gunluk Latest Retrain")
    lines.append("")
    lines.append(f"- Veri donemi: `{summary['history_start']}` -> `{summary['history_end']}`")
    lines.append(f"- Secilen model: `persistence_bias_cv_latest`")
    lines.append(f"- Bias anahtari: `{summary['selected_scheme']}`")
    lines.append(f"- Shrink k: `{summary['selected_shrink_k']}`")
    lines.append(f"- Momentum katsayisi: `{summary['selected_momentum_c']}`")
    lines.append(f"- CV objective (sMAPE): `{summary['selected_cv_smape']:.4f}`")
    lines.append(f"- In-sample sMAPE: `{summary['in_sample_smape']:.4f}`")
    lines.append(f"- Tahmin donemi: `{summary['forecast_start']}` -> `{summary['forecast_end']}`")
    lines.append(f"- Not: {summary['note']}")
    (args.output_dir / "latest_retrain_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(args.output_dir / "latest_retrain_summary.json")
    print(args.output_dir / "overall_mean_daily_latest_forecast.csv")
    print(args.output_dir / "overall_mean_daily_latest_forecast.png")


if __name__ == "__main__":
    main()
