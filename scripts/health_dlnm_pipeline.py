#!/usr/bin/env python3
"""DLNM-like heat-health pipeline with cross-basis features.

This script builds a distributed-lag nonlinear exposure representation from
temperature/humidity time series and optionally fits a Poisson GLM when health
outcomes are provided.

Design goals:
- Use explicit cross-basis construction (exposure basis x lag basis)
- Keep assumptions transparent in outputs
- Support "prepare-only" mode when outcome data is unavailable
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DLNM-like climate-health modeling pipeline")
    p.add_argument("--temp-csv", type=Path, required=True, help="Temperature CSV")
    p.add_argument("--humidity-csv", type=Path, required=True, help="Humidity CSV")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")

    p.add_argument("--date-col", default="ds", help="Date column")
    p.add_argument("--temp-col", default="yhat", help="Temperature value column")
    p.add_argument("--humidity-col", default="yhat", help="Humidity value column")

    p.add_argument("--outcome-csv", type=Path, default=None, help="Optional outcome CSV")
    p.add_argument("--outcome-date-col", default="ds", help="Outcome date column")
    p.add_argument("--outcome-col", default="deaths", help="Outcome count column")
    p.add_argument("--population-col", default=None, help="Optional population column (offset)")

    p.add_argument("--exposure", choices=["heat_index", "temp"], default="heat_index")
    p.add_argument("--threshold-mode", choices=["absolute", "quantile"], default="quantile")
    p.add_argument("--threshold-value-c", type=float, default=26.0)
    p.add_argument("--threshold-quantile", type=float, default=0.90)
    p.add_argument("--center-quantile", type=float, default=0.50)
    p.add_argument("--max-lag", type=int, default=21)
    p.add_argument("--exposure-basis-df", type=int, default=4)
    p.add_argument("--lag-basis-df", type=int, default=4)
    p.add_argument("--seasonal-df", type=int, default=8)
    p.add_argument("--trend-df", type=int, default=6)
    p.add_argument("--min-rows-for-fit", type=int, default=180)

    p.add_argument(
        "--resample-daily-mode",
        choices=["auto", "never", "always"],
        default="auto",
        help="Resample climate series to daily frequency",
    )
    p.add_argument(
        "--climate-resample-method",
        choices=["linear", "ffill"],
        default="linear",
        help="Resample interpolation method for climate series",
    )

    return p.parse_args()


def _pick_numeric_series(df: pd.DataFrame, requested_col: str, fallbacks: list[str]) -> pd.Series:
    candidates = [requested_col] + fallbacks
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s
    raise ValueError(f"No usable numeric column found. Tried: {candidates}")


def load_series(path: Path, date_col: str, value_col: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"{path}: missing date column '{date_col}'")
    out = pd.DataFrame()
    out["ds"] = pd.to_datetime(df[date_col], errors="coerce")
    out[f"{prefix}_value"] = _pick_numeric_series(df, value_col, ["actual", "value", "y"])
    out = out.dropna(subset=["ds", f"{prefix}_value"]).copy()
    out = out.groupby("ds", as_index=False)[f"{prefix}_value"].mean().sort_values("ds")
    return out


def c_to_f(temp_c: np.ndarray) -> np.ndarray:
    return (temp_c * 9.0 / 5.0) + 32.0


def f_to_c(temp_f: np.ndarray) -> np.ndarray:
    return (temp_f - 32.0) * 5.0 / 9.0


def heat_index_c(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    t_f = c_to_f(temp_c)
    r = np.clip(rh_pct, 0.0, 100.0)

    hi_simple = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (r * 0.094))
    hi_reg = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * r
        - 0.22475541 * t_f * r
        - 6.83783e-3 * (t_f**2)
        - 5.481717e-2 * (r**2)
        + 1.22874e-3 * (t_f**2) * r
        + 8.5282e-4 * t_f * (r**2)
        - 1.99e-6 * (t_f**2) * (r**2)
    )

    use_simple = ((hi_simple + t_f) / 2.0) < 80.0
    hi_f = np.where(use_simple, hi_simple, hi_reg)

    low_humidity_adj_mask = (r < 13.0) & (t_f >= 80.0) & (t_f <= 112.0)
    low_humidity_core = np.clip((17.0 - np.abs(t_f - 95.0)) / 17.0, 0.0, None)
    low_humidity_adj = ((13.0 - r) / 4.0) * np.sqrt(low_humidity_core)
    hi_f = np.where(low_humidity_adj_mask, hi_f - low_humidity_adj, hi_f)

    high_humidity_adj_mask = (r > 85.0) & (t_f >= 80.0) & (t_f <= 87.0)
    high_humidity_adj = ((r - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
    hi_f = np.where(high_humidity_adj_mask, hi_f + high_humidity_adj, hi_f)

    return f_to_c(hi_f)


def maybe_resample_daily(df: pd.DataFrame, mode: str, method: str) -> pd.DataFrame:
    d = df.sort_values("ds").copy()
    if len(d) < 2:
        return d
    median_step_days = float(np.median(np.diff(d["ds"].values).astype("timedelta64[D]").astype(float)))

    do_resample = mode == "always" or (mode == "auto" and median_step_days > 1.5)
    if not do_resample:
        return d

    d = d.set_index("ds").sort_index()
    all_days = pd.date_range(d.index.min(), d.index.max(), freq="D")
    d = d.reindex(all_days)
    d.index.name = "ds"
    if method == "linear":
        d = d.interpolate(method="time", limit_direction="both")
    else:
        d = d.ffill().bfill()
    return d.reset_index()


def spline_basis(values: np.ndarray, df: int, include_intercept: bool, var_name: str) -> tuple[np.ndarray, list[str]]:
    formula = f"bs({var_name}, df={int(df)}, degree=3, include_intercept={str(bool(include_intercept))}) - 1"
    mat = dmatrix(formula, {var_name: values}, return_type="dataframe")
    return np.asarray(mat, dtype=float), list(mat.columns)


def build_crossbasis(
    exposure: np.ndarray,
    max_lag: int,
    exposure_basis_df: int,
    lag_basis_df: int,
    center_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    bx_all, _ = spline_basis(exposure, df=exposure_basis_df, include_intercept=False, var_name="x")
    bx_ref, _ = spline_basis(np.array([center_value], dtype=float), df=exposure_basis_df, include_intercept=False, var_name="x")
    bx_ref = bx_ref[0]

    lag_idx = np.arange(0, int(max_lag) + 1, dtype=float)
    bl, _ = spline_basis(lag_idx, df=lag_basis_df, include_intercept=True, var_name="lag")

    kx = bx_all.shape[1]
    kl = bl.shape[1]
    n = len(exposure)
    out_n = n - int(max_lag)
    if out_n <= 0:
        raise ValueError("Not enough rows for requested max_lag")

    cb = np.zeros((out_n, kx * kl), dtype=float)
    for t in range(int(max_lag), n):
        row = np.zeros((kx, kl), dtype=float)
        for l in range(0, int(max_lag) + 1):
            bx = bx_all[t - l] - bx_ref
            row += np.outer(bx, bl[l])
        cb[t - int(max_lag)] = row.reshape(-1)
    return cb, bx_all, bl, kx


def build_confounders(ds: pd.Series, seasonal_df: int, trend_df: int) -> pd.DataFrame:
    out = pd.DataFrame(index=ds.index)
    out["dow"] = ds.dt.dayofweek.astype(int)
    dow_dummies = pd.get_dummies(out["dow"], prefix="dow", drop_first=True)

    doy = ds.dt.dayofyear.astype(float)
    season_mat = dmatrix(
        f"bs(doy, df={int(seasonal_df)}, degree=3, include_intercept=False) - 1",
        {"doy": doy},
        return_type="dataframe",
    )
    season_mat.columns = [f"season_{i}" for i in range(season_mat.shape[1])]

    t = np.arange(len(ds), dtype=float)
    trend_mat = dmatrix(
        f"bs(t, df={int(trend_df)}, degree=3, include_intercept=False) - 1",
        {"t": t},
        return_type="dataframe",
    )
    trend_mat.columns = [f"trend_{i}" for i in range(trend_mat.shape[1])]

    return pd.concat([dow_dummies, season_mat, trend_mat], axis=1)


def compute_rr_surface(
    beta_cb: np.ndarray,
    exposure_grid: np.ndarray,
    center_value: float,
    max_lag: int,
    exposure_basis_df: int,
    bl: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lag_values = np.arange(0, int(max_lag) + 1, dtype=int)
    kx = int(len(beta_cb) / bl.shape[1])
    kl = bl.shape[1]
    if kx * kl != len(beta_cb):
        raise ValueError("Cross-basis coefficient length mismatch")

    records: list[dict[str, Any]] = []
    curve_records: list[dict[str, Any]] = []

    bx_ref, _ = spline_basis(np.array([center_value], dtype=float), df=exposure_basis_df, include_intercept=False, var_name="x")
    bx_ref = bx_ref[0]

    for x in exposure_grid:
        bx, _ = spline_basis(np.array([float(x)], dtype=float), df=exposure_basis_df, include_intercept=False, var_name="x")
        bx = bx[0] - bx_ref
        lag_log_rr = []
        for l in lag_values:
            vec = np.outer(bx, bl[l]).reshape(-1)
            log_rr = float(np.dot(vec, beta_cb))
            rr = float(np.exp(log_rr))
            lag_log_rr.append(log_rr)
            records.append({"exposure": float(x), "lag": int(l), "log_rr": log_rr, "rr": rr})

        cumulative_log_rr = float(np.sum(lag_log_rr))
        curve_records.append(
            {
                "exposure": float(x),
                "cumulative_log_rr": cumulative_log_rr,
                "cumulative_rr": float(np.exp(cumulative_log_rr)),
            }
        )

    return pd.DataFrame(records), pd.DataFrame(curve_records)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    temp = load_series(args.temp_csv, args.date_col, args.temp_col, "temp")
    hum = load_series(args.humidity_csv, args.date_col, args.humidity_col, "humidity")
    climate = temp.merge(hum, on="ds", how="inner").sort_values("ds")
    if climate.empty:
        raise SystemExit("No overlap between temp and humidity series.")

    climate = maybe_resample_daily(climate, mode=args.resample_daily_mode, method=args.climate_resample_method)

    climate["temp_c"] = climate["temp_value"].astype(float)
    climate["humidity_pct"] = climate["humidity_value"].astype(float).clip(0.0, 100.0)
    climate["heat_index_c"] = heat_index_c(climate["temp_c"].to_numpy(), climate["humidity_pct"].to_numpy())

    if args.exposure == "heat_index":
        climate["exposure_raw"] = climate["heat_index_c"]
    else:
        climate["exposure_raw"] = climate["temp_c"]

    if args.threshold_mode == "absolute":
        threshold_c = float(args.threshold_value_c)
    else:
        q = float(np.clip(args.threshold_quantile, 0.01, 0.99))
        threshold_c = float(climate["exposure_raw"].quantile(q))

    climate["exposure_excess_c"] = (climate["exposure_raw"] - threshold_c).clip(lower=0.0)
    center_value = float(climate["exposure_raw"].quantile(float(np.clip(args.center_quantile, 0.01, 0.99))))

    climate_out = args.output_dir / "dlnm_climate_prepared.csv"
    climate.to_csv(climate_out, index=False)

    cb, _, bl, kx = build_crossbasis(
        exposure=climate["exposure_excess_c"].to_numpy(),
        max_lag=int(args.max_lag),
        exposure_basis_df=int(args.exposure_basis_df),
        lag_basis_df=int(args.lag_basis_df),
        center_value=0.0,  # exposure_excess is already threshold-centered
    )
    kl = bl.shape[1]
    cb_cols = [f"cb_{i}" for i in range(cb.shape[1])]

    model_ds = climate["ds"].iloc[int(args.max_lag) :].reset_index(drop=True)
    cb_df = pd.DataFrame(cb, columns=cb_cols)
    base_df = pd.DataFrame({"ds": model_ds})
    design_df = pd.concat([base_df, cb_df], axis=1)
    conf_df = build_confounders(model_ds, seasonal_df=int(args.seasonal_df), trend_df=int(args.trend_df)).reset_index(drop=True)
    design_df = pd.concat([design_df, conf_df], axis=1)

    design_path = args.output_dir / "dlnm_design_matrix.csv"
    design_df.to_csv(design_path, index=False)

    metadata = {
        "inputs": {
            "temp_csv": str(args.temp_csv),
            "humidity_csv": str(args.humidity_csv),
            "outcome_csv": str(args.outcome_csv) if args.outcome_csv else None,
            "exposure": args.exposure,
            "threshold_mode": args.threshold_mode,
            "threshold_value_c": args.threshold_value_c,
            "threshold_quantile": args.threshold_quantile,
            "center_quantile": args.center_quantile,
            "max_lag": args.max_lag,
            "exposure_basis_df": args.exposure_basis_df,
            "lag_basis_df": args.lag_basis_df,
            "seasonal_df": args.seasonal_df,
            "trend_df": args.trend_df,
        },
        "resolved": {
            "threshold_c": threshold_c,
            "center_value_c": center_value,
            "n_climate_rows": int(len(climate)),
            "n_design_rows": int(len(design_df)),
            "kx": int(kx),
            "kl": int(kl),
            "crossbasis_features": int(cb.shape[1]),
        },
    }

    if args.outcome_csv is None:
        metadata["mode"] = "prepare_only"
        metadata_path = args.output_dir / "dlnm_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote: {climate_out}")
        print(f"Wrote: {design_path}")
        print(f"Wrote: {metadata_path}")
        return

    out_df = pd.read_csv(args.outcome_csv)
    if args.outcome_date_col not in out_df.columns or args.outcome_col not in out_df.columns:
        raise SystemExit("Outcome CSV missing required columns.")

    outcomes = pd.DataFrame()
    outcomes["ds"] = pd.to_datetime(out_df[args.outcome_date_col], errors="coerce")
    outcomes["y"] = pd.to_numeric(out_df[args.outcome_col], errors="coerce")
    if args.population_col and args.population_col in out_df.columns:
        outcomes["population"] = pd.to_numeric(out_df[args.population_col], errors="coerce")
    outcomes = outcomes.dropna(subset=["ds", "y"]).sort_values("ds")
    outcomes = outcomes.groupby("ds", as_index=False).agg({"y": "sum", **({"population": "mean"} if "population" in outcomes.columns else {})})

    fit_df = design_df.merge(outcomes, on="ds", how="inner").dropna(subset=["y"]).copy()
    if len(fit_df) < int(args.min_rows_for_fit):
        raise SystemExit(f"Not enough rows for model fit after merge: {len(fit_df)}")

    x_cols = [c for c in fit_df.columns if c.startswith("cb_") or c.startswith("dow_") or c.startswith("season_") or c.startswith("trend_")]
    X = fit_df[x_cols].astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = fit_df["y"].astype(float)

    offset = None
    if "population" in fit_df.columns:
        pop = fit_df["population"].clip(lower=1.0)
        offset = np.log(pop)

    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    result = model.fit(cov_type="HC0")

    coef_df = pd.DataFrame(
        {
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "z": result.tvalues.values,
            "p_value": result.pvalues.values,
        }
    )
    coef_df["ci_low"] = coef_df["coef"] - 1.96 * coef_df["std_err"]
    coef_df["ci_high"] = coef_df["coef"] + 1.96 * coef_df["std_err"]

    cb_terms = [c for c in coef_df["term"] if c.startswith("cb_")]
    beta_cb = coef_df.set_index("term").loc[cb_terms, "coef"].to_numpy(dtype=float)

    exp_low = float(climate["exposure_excess_c"].quantile(0.01))
    exp_high = float(climate["exposure_excess_c"].quantile(0.99))
    exposure_grid = np.linspace(exp_low, exp_high, 50)
    rr_surface, rr_curve = compute_rr_surface(
        beta_cb=beta_cb,
        exposure_grid=exposure_grid,
        center_value=0.0,
        max_lag=int(args.max_lag),
        exposure_basis_df=int(args.exposure_basis_df),
        bl=bl,
    )

    coef_path = args.output_dir / "dlnm_glm_coefficients.csv"
    summary_txt = args.output_dir / "dlnm_glm_summary.txt"
    fit_df_path = args.output_dir / "dlnm_fit_dataset.csv"
    rr_surface_path = args.output_dir / "dlnm_rr_surface.csv"
    rr_curve_path = args.output_dir / "dlnm_cumulative_rr_curve.csv"
    metadata_path = args.output_dir / "dlnm_metadata.json"

    coef_df.to_csv(coef_path, index=False)
    fit_df.to_csv(fit_df_path, index=False)
    rr_surface.to_csv(rr_surface_path, index=False)
    rr_curve.to_csv(rr_curve_path, index=False)
    summary_txt.write_text(str(result.summary()), encoding="utf-8")

    metadata["mode"] = "fit"
    metadata["fit"] = {
        "n_fit_rows": int(len(fit_df)),
        "aic": float(result.aic),
        "deviance": float(result.deviance),
        "pearson_chi2": float(result.pearson_chi2),
        "df_model": float(result.df_model),
        "df_resid": float(result.df_resid),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "# DLNM-Like Model Report",
        "",
        f"- Rows fitted: {len(fit_df)}",
        f"- AIC: {result.aic:.3f}",
        f"- Deviance: {result.deviance:.3f}",
        f"- Threshold (C): {threshold_c:.3f}",
        f"- Exposure: {args.exposure}",
        f"- Max lag: {args.max_lag}",
        "",
        "## Notes",
        "",
        "- This is a DLNM-like cross-basis GLM implementation.",
        "- Causal interpretation requires domain-appropriate confounders and validated outcome data.",
    ]
    (args.output_dir / "dlnm_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {climate_out}")
    print(f"Wrote: {design_path}")
    print(f"Wrote: {fit_df_path}")
    print(f"Wrote: {coef_path}")
    print(f"Wrote: {summary_txt}")
    print(f"Wrote: {rr_surface_path}")
    print(f"Wrote: {rr_curve_path}")
    print(f"Wrote: {metadata_path}")


if __name__ == "__main__":
    main()

