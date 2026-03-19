#!/usr/bin/env python3
"""Production-grade Prophet climate forecasting pipeline.

This script is designed to be reusable for different data layouts:
- Long observations table: timestamp + variable + value (+ optional qc)
- Single-series table: timestamp + value
- Input files: parquet/csv/tsv/xlsx/xls/ods

Outputs:
- forecasts/*.csv|parquet
- charts/*.png
- components/*.png
- leaderboards/*.csv
- prophet_index_to_<year>.csv|parquet

Examples:
  python3 scripts/prophet_climate_forecast.py \
    --observations output/forecast_package/observations_with_graph.parquet \
    --input-kind auto \
    --target-year 2035 \
    --auto-tune true

  python3 scripts/prophet_climate_forecast.py \
    --observations output/1987_hourly_temp.csv \
    --input-kind single \
    --timestamp-col timestamp \
    --value-col temp_c \
    --single-variable temp \
    --variables temp \
    --target-year 2030
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Reduce matplotlib/font cache warnings in restricted environments.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "prophet_climate_cache"
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
        "Prophet import failed. Install in your runtime:\n"
        "  pip install prophet\n"
        f"Original error: {exc}"
    )

# Keep runtime logs concise for hackathon usage.
try:
    import cmdstanpy

    cmdstanpy.disable_logging()
except Exception:
    pass
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


DEFAULT_VARIABLES = ["humidity", "temp", "pressure", "precip"]
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
    p = argparse.ArgumentParser(description="Reusable Prophet climate forecasting pipeline")
    p.add_argument(
        "--observations",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/forecast_package/observations_with_graph.parquet"),
        help="Input dataset path (parquet/csv/tsv/xlsx/xls/ods)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/prophet_package"),
    )
    p.add_argument(
        "--input-kind",
        type=str,
        default="auto",
        choices=["auto", "long", "single"],
        help="auto: detect format, long: timestamp+variable+value, single: timestamp+value",
    )
    p.add_argument("--timestamp-col", type=str, default="timestamp")
    p.add_argument("--value-col", type=str, default="value")
    p.add_argument("--variable-col", type=str, default="variable")
    p.add_argument("--qc-col", type=str, default="qc_flag")
    p.add_argument("--qc-ok-value", type=str, default="ok")
    p.add_argument("--single-variable", type=str, default="target")

    p.add_argument("--target-year", type=int, default=2035)
    p.add_argument("--interval-width", type=float, default=0.8)
    p.add_argument("--holdout-months", type=int, default=12)
    p.add_argument("--backtest-splits", type=int, default=2)
    p.add_argument("--min-train-months", type=int, default=36)
    p.add_argument("--auto-tune", type=str, default="true", help="true/false")
    p.add_argument("--variables", type=str, default="*", help="Comma list or * for all")
    p.add_argument(
        "--winsor-quantile",
        type=float,
        default=0.995,
        help="Upper winsorization quantile for outlier control (0.90-1.0)",
    )
    return p.parse_args()


def to_bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def is_humidity_like(variable: str) -> bool:
    return canonical_variable_name(variable) == "humidity"


def is_precip_like(variable: str) -> bool:
    return canonical_variable_name(variable) == "precip"


def is_pressure_like(variable: str) -> bool:
    return canonical_variable_name(variable) == "pressure"


def infer_unit(variable: str) -> str:
    canon = canonical_variable_name(variable)
    return UNIT_MAP.get(canon, "unknown")


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
    ts_col = pick_existing_column(
        raw,
        args.timestamp_col,
        ["timestamp", "ds", "date", "datetime", "time", "tarih"],
    )
    val_col = pick_existing_column(raw, args.value_col, ["value", "y", "target", "measurement"])
    var_col = pick_existing_column(raw, args.variable_col, ["variable", "metric", "param", "sensor", "name"])
    qc_col = pick_existing_column(raw, args.qc_col, ["qc_flag", "qc", "quality", "flag"])

    if ts_col is None or val_col is None:
        raise SystemExit(
            "Cannot detect time/value columns. Provide --timestamp-col and --value-col explicitly."
        )

    input_kind = args.input_kind
    if input_kind == "auto":
        input_kind = "long" if var_col is not None else "single"

    if input_kind == "long":
        if var_col is None:
            raise SystemExit("input-kind=long requires a variable column. Set --variable-col.")
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

    out = out.dropna(subset=["timestamp", "value", "variable"]).sort_values("timestamp")
    out = out.reset_index(drop=True)

    if out.empty:
        raise SystemExit("No usable rows after parsing input table.")

    return out, input_kind


def aggregate_monthly(obs: pd.DataFrame, variable: str, qc_ok_value: str = "ok") -> pd.Series:
    sub = obs[obs["variable"] == variable].copy()
    if sub.empty:
        return pd.Series(dtype=float)

    if "qc_flag" in sub.columns:
        ok_mask = sub["qc_flag"].str.lower().eq(str(qc_ok_value).lower())
        if ok_mask.any():
            sub = sub[ok_mask]

    sub = sub.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if sub.empty:
        return pd.Series(dtype=float)

    raw = sub.groupby("timestamp")["value"].mean()
    if is_precip_like(variable):
        s = raw.resample("MS").sum(min_count=1)
        s = s.fillna(0.0)
    else:
        s = raw.resample("MS").mean()
        s = s.interpolate("time").ffill().bfill()

    return s.astype(float)


def apply_bounds(arr: np.ndarray, variable: str) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if is_humidity_like(variable):
        return np.clip(x, 0, 100)
    if is_precip_like(variable) or is_pressure_like(variable):
        return np.clip(x, 0, None)
    return x


def preprocess_series(series: pd.Series, variable: str, winsor_q: float) -> tuple[pd.DataFrame, bool]:
    s = series.copy().astype(float)

    q = float(np.clip(winsor_q, 0.90, 1.0))
    hi = float(s.quantile(q)) if len(s) else np.nan
    if np.isfinite(hi):
        s = s.clip(upper=hi)

    use_log = is_precip_like(variable)
    if use_log:
        y = np.log1p(np.clip(s.values, 0, None))
    else:
        y = s.values.astype(float)

    df = pd.DataFrame({"ds": s.index, "y": y})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna().sort_values("ds").reset_index(drop=True)
    return df, use_log


def invert_target(yhat: np.ndarray, use_log: bool) -> np.ndarray:
    return np.expm1(np.asarray(yhat, dtype=float)) if use_log else np.asarray(yhat, dtype=float)


def prophet_model(cfg: ProphetConfig) -> Prophet:
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=cfg.seasonality_mode,
        interval_width=cfg.interval_width,
        changepoint_prior_scale=cfg.changepoint_prior_scale,
        seasonality_prior_scale=cfg.seasonality_prior_scale,
        changepoint_range=cfg.changepoint_range,
    )
    # Monthly climate series often has sub-annual rhythm.
    m.add_seasonality(name="quarterly", period=365.25 / 4.0, fourier_order=5)
    return m


def default_config(variable: str, interval_width: float) -> ProphetConfig:
    mode = "multiplicative" if is_precip_like(variable) else "additive"
    return ProphetConfig(
        seasonality_mode=mode,
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10.0,
        changepoint_range=0.9,
        interval_width=interval_width,
    )


def config_grid(variable: str, interval_width: float) -> list[ProphetConfig]:
    modes = ["additive", "multiplicative"] if is_precip_like(variable) else ["additive"]
    cps = [0.03, 0.10, 0.25]
    sps = [5.0, 10.0, 20.0]
    cpr = [0.85, 0.95]
    out: list[ProphetConfig] = []
    for mode, cp, sp, cr in itertools.product(modes, cps, sps, cpr):
        out.append(
            ProphetConfig(
                seasonality_mode=mode,
                changepoint_prior_scale=float(cp),
                seasonality_prior_scale=float(sp),
                changepoint_range=float(cr),
                interval_width=interval_width,
            )
        )
    return out


def split_points(n: int, holdout: int, splits: int, min_train: int) -> list[int]:
    points: list[int] = []
    for i in range(splits, 0, -1):
        cut = n - i * holdout
        if min_train <= cut < n:
            points.append(cut)
    return sorted(set(points))


def metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    e = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    yt = np.asarray(y_true, dtype=float)
    mask = np.abs(yt) > 1e-9
    if mask.any():
        mape = float(np.mean(np.abs(e[mask] / yt[mask])) * 100)
    else:
        mape = np.nan
    return {"mae": mae, "rmse": rmse, "mape": mape}


def backtest_score(
    df_model: pd.DataFrame,
    cfg: ProphetConfig,
    holdout: int,
    splits: int,
    min_train: int,
    use_log: bool,
) -> dict[str, float]:
    n = len(df_model)
    cuts = split_points(n=n, holdout=holdout, splits=splits, min_train=min_train)
    if not cuts:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "n_folds": 0}

    maes: list[float] = []
    rmses: list[float] = []
    mapes: list[float] = []

    for cut in cuts:
        train = df_model.iloc[:cut].copy()
        test = df_model.iloc[cut : cut + holdout].copy()
        if len(test) == 0:
            continue

        try:
            m = prophet_model(cfg)
            m.fit(train)
            fut = m.make_future_dataframe(periods=len(test), freq="MS")
            pred = m.predict(fut)[["ds", "yhat"]].tail(len(test))
            chk = test[["ds", "y"]].merge(pred, on="ds", how="inner")
            if chk.empty:
                continue

            y_true = invert_target(chk["y"].values, use_log=use_log)
            y_pred = invert_target(chk["yhat"].values, use_log=use_log)
            mp = metric_pack(y_true, y_pred)
            maes.append(mp["mae"])
            rmses.append(mp["rmse"])
            mapes.append(mp["mape"])
        except Exception:
            continue

    if not maes:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "n_folds": 0}

    return {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "mape": float(np.nanmean(mapes)) if len(mapes) else np.nan,
        "n_folds": float(len(maes)),
    }


def fit_and_forecast(
    df_model: pd.DataFrame,
    cfg: ProphetConfig,
    months: int,
    variable: str,
    use_log: bool,
) -> tuple[Prophet, pd.DataFrame]:
    m = prophet_model(cfg)
    m.fit(df_model)
    future = m.make_future_dataframe(periods=months, freq="MS")
    pred = m.predict(future)

    pred["yhat"] = invert_target(pred["yhat"].values, use_log=use_log)
    pred["yhat_lower"] = invert_target(pred["yhat_lower"].values, use_log=use_log)
    pred["yhat_upper"] = invert_target(pred["yhat_upper"].values, use_log=use_log)

    pred["yhat"] = apply_bounds(pred["yhat"].values, variable)
    pred["yhat_lower"] = apply_bounds(pred["yhat_lower"].values, variable)
    pred["yhat_upper"] = apply_bounds(pred["yhat_upper"].values, variable)

    return m, pred


def seasonal_naive_forecast(df_model: pd.DataFrame, months: int, variable: str, use_log: bool) -> pd.DataFrame:
    y_hist = invert_target(df_model["y"].values, use_log=use_log)
    ds_hist = pd.to_datetime(df_model["ds"].values)
    hist = pd.DataFrame({"ds": ds_hist, "yhat": y_hist})

    last_ds = pd.Timestamp(hist["ds"].max())
    fidx = pd.date_range(last_ds + pd.offsets.MonthBegin(1), periods=months, freq="MS")

    if len(hist) >= 12:
        profile = np.array(
            [np.nanmean(hist[hist["ds"].dt.month == m]["yhat"].values) for m in range(1, 13)],
            dtype=float,
        )
        mval = float(np.nanmean(hist["yhat"].values))
        profile = np.where(np.isnan(profile), mval, profile)
    else:
        profile = np.repeat(float(np.nanmean(hist["yhat"].values)), 12)

    fvals = np.array([profile[d.month - 1] for d in fidx], dtype=float)
    fvals = apply_bounds(fvals, variable)

    out_hist = pd.DataFrame(
        {
            "ds": hist["ds"],
            "yhat": hist["yhat"],
            "yhat_lower": np.nan,
            "yhat_upper": np.nan,
        }
    )
    out_fc = pd.DataFrame({"ds": fidx, "yhat": fvals, "yhat_lower": np.nan, "yhat_upper": np.nan})
    return pd.concat([out_hist, out_fc], ignore_index=True)


def requested_variables(obs: pd.DataFrame, variables_arg: str) -> list[str]:
    available = sorted(obs["variable"].dropna().unique().tolist())
    if not variables_arg or variables_arg.strip() in {"*", "all", "ALL"}:
        return available

    req = [canonical_variable_name(v.strip()) for v in variables_arg.split(",") if v.strip()]
    req = [v for v in req if v in available]
    return sorted(set(req))


def main() -> None:
    args = parse_args()
    auto_tune = to_bool(args.auto_tune)

    raw = read_table(args.observations)
    obs, detected_kind = normalize_observations(raw, args)
    vars_use = requested_variables(obs, args.variables)
    if not vars_use:
        raise SystemExit(
            "No variables left after filtering. Check --variables or input variable names."
        )

    out = args.output_dir
    fc_dir = out / "forecasts"
    ch_dir = out / "charts"
    cmp_dir = out / "components"
    lb_dir = out / "leaderboards"
    for d in [out, fc_dir, ch_dir, cmp_dir, lb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(
        f"Input: {args.observations} | detected_kind={detected_kind} | "
        f"rows={len(obs)} | variables={','.join(vars_use)}"
    )

    index_rows: list[dict[str, object]] = []

    for var in vars_use:
        s = aggregate_monthly(obs, var, qc_ok_value=args.qc_ok_value)
        if s.empty:
            continue

        df_model, use_log = preprocess_series(s, variable=var, winsor_q=args.winsor_quantile)
        if df_model.empty:
            continue

        last_ds = pd.Timestamp(df_model["ds"].max())
        target_ds = pd.Timestamp(year=args.target_year, month=12, day=1)
        months = int((target_ds.year - last_ds.year) * 12 + (target_ds.month - last_ds.month))
        months = max(0, months)

        if auto_tune:
            cfgs = config_grid(var, interval_width=args.interval_width)
        else:
            cfgs = [default_config(var, interval_width=args.interval_width)]

        lb_rows: list[dict[str, object]] = []
        for i, cfg in enumerate(cfgs, start=1):
            score = backtest_score(
                df_model=df_model,
                cfg=cfg,
                holdout=int(args.holdout_months),
                splits=int(args.backtest_splits),
                min_train=int(args.min_train_months),
                use_log=use_log,
            )
            lb_rows.append(
                {
                    "rank_seed": i,
                    "seasonality_mode": cfg.seasonality_mode,
                    "changepoint_prior_scale": cfg.changepoint_prior_scale,
                    "seasonality_prior_scale": cfg.seasonality_prior_scale,
                    "changepoint_range": cfg.changepoint_range,
                    "interval_width": cfg.interval_width,
                    "mae": score["mae"],
                    "rmse": score["rmse"],
                    "mape": score["mape"],
                    "n_folds": score["n_folds"],
                }
            )

        lb = pd.DataFrame(lb_rows)
        lb = lb.sort_values(["rmse", "mae", "mape"], na_position="last").reset_index(drop=True)
        lb["rank"] = np.arange(1, len(lb) + 1)
        lb_csv = lb_dir / f"{var}_prophet_leaderboard_to_{args.target_year}.csv"
        lb.to_csv(lb_csv, index=False)

        if len(df_model) < int(args.min_train_months):
            strategy = "seasonal_naive_fallback"
            pred = seasonal_naive_forecast(df_model, months=months, variable=var, use_log=use_log)
            best_cfg = default_config(var, interval_width=args.interval_width)
            best_score = {"mae": np.nan, "rmse": np.nan, "mape": np.nan}
            m = None
        else:
            best = lb.iloc[0]
            best_cfg = ProphetConfig(
                seasonality_mode=str(best["seasonality_mode"]),
                changepoint_prior_scale=float(best["changepoint_prior_scale"]),
                seasonality_prior_scale=float(best["seasonality_prior_scale"]),
                changepoint_range=float(best["changepoint_range"]),
                interval_width=float(best["interval_width"]),
            )
            best_score = {
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "mape": float(best["mape"]),
            }
            strategy = "prophet_tuned" if auto_tune else "prophet_default"
            m, pred = fit_and_forecast(df_model, cfg=best_cfg, months=months, variable=var, use_log=use_log)

        pred["is_forecast"] = pred["ds"] > last_ds
        pred["variable"] = var
        pred["unit"] = infer_unit(var)
        pred["model_strategy"] = strategy
        pred["use_log_transform"] = use_log

        csv_p = fc_dir / f"{var}_monthly_prophet_to_{args.target_year}.csv"
        pq_p = fc_dir / f"{var}_monthly_prophet_to_{args.target_year}.parquet"
        pred.to_csv(csv_p, index=False)
        pred.to_parquet(pq_p, index=False)

        fig, ax = plt.subplots(figsize=(12, 4.8))
        hist = pred[pred["is_forecast"] == False]
        fc = pred[pred["is_forecast"] == True]
        ax.plot(hist["ds"], hist["yhat"], color="#1f77b4", linewidth=1.4, label="historical")
        if not fc.empty:
            ax.plot(fc["ds"], fc["yhat"], color="#d62728", linewidth=1.9, label="forecast")
            if fc["yhat_lower"].notna().any() and fc["yhat_upper"].notna().any():
                ax.fill_between(
                    fc["ds"],
                    fc["yhat_lower"],
                    fc["yhat_upper"],
                    color="#d62728",
                    alpha=0.15,
                    label="uncertainty",
                )
        ax.axvline(last_ds, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_title(f"Prophet monthly forecast - {var}")
        ax.set_xlabel("date")
        ax.set_ylabel(f"value ({infer_unit(var)})")
        ax.grid(alpha=0.24)
        ax.legend(loc="best")
        fig.tight_layout()
        chart_p = ch_dir / f"{var}_monthly_prophet_to_{args.target_year}.png"
        fig.savefig(chart_p, dpi=160)
        plt.close(fig)

        if m is not None:
            fig2 = m.plot_components(pred)
            fig2.tight_layout()
            comp_p = cmp_dir / f"{var}_monthly_prophet_components_to_{args.target_year}.png"
            fig2.savefig(comp_p, dpi=160)
            plt.close(fig2)
        else:
            comp_p = cmp_dir / f"{var}_monthly_prophet_components_to_{args.target_year}.png"
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            ax3.text(0.02, 0.6, "Components unavailable\n(fallback model used)", fontsize=11)
            ax3.axis("off")
            fig3.tight_layout()
            fig3.savefig(comp_p, dpi=160)
            plt.close(fig3)

        index_rows.append(
            {
                "variable": var,
                "last_observation": str(last_ds),
                "target_year": args.target_year,
                "horizon_months": months,
                "model_strategy": strategy,
                "use_log_transform": use_log,
                "best_seasonality_mode": best_cfg.seasonality_mode,
                "best_changepoint_prior_scale": best_cfg.changepoint_prior_scale,
                "best_seasonality_prior_scale": best_cfg.seasonality_prior_scale,
                "best_changepoint_range": best_cfg.changepoint_range,
                "holdout_mae": best_score["mae"],
                "holdout_rmse": best_score["rmse"],
                "holdout_mape": best_score["mape"],
                "forecast_csv": str(csv_p),
                "forecast_parquet": str(pq_p),
                "chart_png": str(chart_p),
                "components_png": str(comp_p),
                "leaderboard_csv": str(lb_csv),
                "input_kind": detected_kind,
                "input_path": str(args.observations),
            }
        )

    idx = pd.DataFrame(index_rows).sort_values("variable") if index_rows else pd.DataFrame()
    idx_csv = out / f"prophet_index_to_{args.target_year}.csv"
    idx_pq = out / f"prophet_index_to_{args.target_year}.parquet"
    idx.to_csv(idx_csv, index=False)
    idx.to_parquet(idx_pq, index=False)

    print("Prophet pipeline completed.")
    print(f"Index: {idx_csv}")
    if not idx.empty:
        print(idx[["variable", "model_strategy", "holdout_rmse", "forecast_csv"]].to_string(index=False))


if __name__ == "__main__":
    main()
