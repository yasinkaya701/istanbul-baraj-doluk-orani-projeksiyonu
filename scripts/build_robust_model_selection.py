#!/usr/bin/env python3
"""Build stability-aware model selection on top of model_suite outputs.

This script scans a model suite run directory, collects comparable accuracy/stability
signals from model index files, and selects one best model per variable.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TR_MAP = str.maketrans(
    {
        "ı": "i",
        "İ": "I",
        "ğ": "g",
        "Ğ": "G",
        "ş": "s",
        "Ş": "S",
        "ö": "o",
        "Ö": "O",
        "ç": "c",
        "Ç": "C",
        "ü": "u",
        "Ü": "U",
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build robust model selection for model suite run")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/robust_selection",
    )
    p.add_argument(
        "--coverage-floor",
        type=float,
        default=0.85,
        help="Coverage below this threshold receives a penalty.",
    )
    p.add_argument(
        "--freq-mismatch-penalty",
        type=float,
        default=0.35,
        help="Penalty for candidates outside preferred frequency for variable.",
    )
    return p.parse_args()


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def normalize_token(x: Any) -> str:
    s = str(x or "").translate(TR_MAP).strip().lower()
    for old, new in [("/", "_"), ("-", "_"), (" ", "_")]:
        s = s.replace(old, new)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def canonical_variable_name(x: Any) -> str:
    t = normalize_token(x)
    aliases = {
        "nem": "humidity",
        "humidity": "humidity",
        "rh": "humidity",
        "yagis": "precip",
        "yais": "precip",
        "rain": "precip",
        "precip": "precip",
        "sicaklik": "temp",
        "sicak": "temp",
        "temp": "temp",
        "temperature": "temp",
        "basinc": "pressure",
        "pressure": "pressure",
    }
    if t in aliases:
        return aliases[t]
    if "humid" in t or "nem" in t:
        return "humidity"
    if "precip" in t or "rain" in t or "yagis" in t:
        return "precip"
    if "temp" in t or "sicak" in t:
        return "temp"
    if "press" in t or "basinc" in t:
        return "pressure"
    return t or "target"


def normalize_frequency(x: Any) -> str:
    t = normalize_token(x).upper()
    if t in {"MS", "M", "MONTHLY", "AYLIK"}:
        return "MS"
    if t in {"YS", "Y", "A", "AS", "YEARLY", "YILLIK"}:
        return "YS"
    if t in {"W", "WEEKLY", "HAFTALIK"}:
        return "W"
    if t in {"D", "DAILY", "GUNLUK"}:
        return "D"
    if t in {"H", "HOURLY", "SAATLIK"}:
        return "H"
    return t if t else "NA"


def infer_frequency_from_dates(ds: pd.Series) -> str:
    ts = pd.to_datetime(ds, errors="coerce").dropna().sort_values().drop_duplicates()
    if len(ts) < 3:
        return "NA"
    diff_days = ts.diff().dt.total_seconds().dropna().to_numpy(dtype=float) / 86400.0
    if len(diff_days) == 0:
        return "NA"
    med = float(np.nanmedian(diff_days))
    if med <= 0.1:
        return "H"
    if med <= 2.0:
        return "D"
    if med <= 10.0:
        return "W"
    if med <= 45.0:
        return "MS"
    return "YS"


def resolve_path(path_text: Any, run_dir: Path, repo_root: Path) -> Path | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    raw = Path(text)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append((run_dir / raw).resolve())
    candidates.append((repo_root / raw).resolve())
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def rel_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists() and path.is_file():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def read_table_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suf in {".xlsx", ".xls", ".ods"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported observations extension: {path.suffix}")


def normalize_observations_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "variable", "value"])

    cols = list(df.columns)
    ts_col = None
    for c in ("timestamp", "ds", "date", "datetime"):
        if c in cols:
            ts_col = c
            break
    if ts_col is None:
        return pd.DataFrame(columns=["timestamp", "variable", "value"])

    if "variable" in cols and "value" in cols:
        out = df[[ts_col, "variable", "value"]].copy()
        out = out.rename(columns={ts_col: "timestamp"})
    else:
        wide_cols = [c for c in cols if c != ts_col]
        if not wide_cols:
            return pd.DataFrame(columns=["timestamp", "variable", "value"])
        out = df[[ts_col] + wide_cols].copy().melt(
            id_vars=[ts_col],
            var_name="variable",
            value_name="value",
        )
        out = out.rename(columns={ts_col: "timestamp"})

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["variable"] = out["variable"].map(canonical_variable_name)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["timestamp", "variable", "value"]).copy()
    return out[["timestamp", "variable", "value"]].sort_values("timestamp").reset_index(drop=True)


def load_observation_context(run_dir: Path, repo_root: Path) -> dict[str, Any]:
    summary_path = run_dir / "model_suite_summary.json"
    obs_path = None
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    for key in ("observations_used", "observations_original"):
        cand = resolve_path(summary.get(key), run_dir=run_dir, repo_root=repo_root)
        if cand is not None and cand.exists():
            obs_path = cand
            break

    if obs_path is None:
        return {
            "available": False,
            "path": "",
            "obs": pd.DataFrame(columns=["timestamp", "variable", "value"]),
            "cache": {},
        }
    try:
        raw = read_table_any(obs_path)
        obs = normalize_observations_df(raw)
    except Exception:
        obs = pd.DataFrame(columns=["timestamp", "variable", "value"])
    return {
        "available": not obs.empty,
        "path": str(obs_path),
        "obs": obs,
        "cache": {},
    }


def get_observation_series(ctx: dict[str, Any], variable: str, frequency: str) -> pd.DataFrame:
    var = canonical_variable_name(variable)
    freq = normalize_frequency(frequency)
    key = (var, freq)
    cache = ctx.get("cache", {})
    if key in cache:
        return cache[key]
    obs = ctx.get("obs", pd.DataFrame())
    if obs.empty:
        out = pd.DataFrame(columns=["ds", "actual"])
        cache[key] = out
        return out

    d = obs[obs["variable"] == var].copy()
    if d.empty:
        out = pd.DataFrame(columns=["ds", "actual"])
        cache[key] = out
        return out

    if freq == "H":
        d["ds"] = d["timestamp"].dt.floor("h")
    elif freq == "D":
        d["ds"] = d["timestamp"].dt.floor("D")
    elif freq == "W":
        d["ds"] = d["timestamp"].dt.to_period("W").dt.start_time
    elif freq == "YS":
        d["ds"] = d["timestamp"].dt.to_period("Y").dt.to_timestamp()
    else:
        d["ds"] = d["timestamp"].dt.to_period("M").dt.to_timestamp()

    agg_kind = "sum" if var == "precip" else "mean"
    grp = d.groupby("ds", as_index=False)["value"]
    if agg_kind == "sum":
        out = grp.sum().rename(columns={"value": "actual"})
    else:
        out = grp.mean().rename(columns={"value": "actual"})
    out["actual"] = pd.to_numeric(out["actual"], errors="coerce")
    out = out.dropna(subset=["ds", "actual"]).sort_values("ds").reset_index(drop=True)
    cache[key] = out
    return out


def parse_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False).astype(bool)
    txt = s.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y", "on"}
    false_set = {"0", "false", "f", "no", "n", "off"}
    mapped = txt.map(lambda x: True if x in true_set else False if x in false_set else np.nan)
    if mapped.isna().all():
        return pd.Series(False, index=s.index)
    return mapped.fillna(False).astype(bool)


def infer_forecast_freq(path: Path) -> str:
    try:
        d = pd.read_csv(path, nrows=600)
    except Exception:
        return "NA"
    date_col = None
    for c in ("ds", "timestamp", "date"):
        if c in d.columns:
            date_col = c
            break
    if date_col is None:
        return "NA"
    return infer_frequency_from_dates(d[date_col])


def overlap_metrics_from_forecast(
    path: Path,
    variable: str,
    frequency: str,
    obs_ctx: dict[str, Any],
) -> dict[str, float]:
    try:
        d = pd.read_csv(path)
    except Exception:
        return {
            "rmse": float("nan"),
            "bias_abs": float("nan"),
            "rmse_std": float("nan"),
            "coverage": float("nan"),
            "n_eval": float("nan"),
            "n_hist": float("nan"),
            "skill_naive": float("nan"),
        }

    date_col = None
    for c in ("ds", "timestamp", "date"):
        if c in d.columns:
            date_col = c
            break
    if date_col is None or "yhat" not in d.columns:
        return {
            "rmse": float("nan"),
            "bias_abs": float("nan"),
            "rmse_std": float("nan"),
            "coverage": float("nan"),
            "n_eval": float("nan"),
            "n_hist": float("nan"),
            "skill_naive": float("nan"),
        }

    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).copy()
    d = d.rename(columns={date_col: "ds"})
    d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
    d = d.dropna(subset=["ds"]).copy()
    d["yhat"] = pd.to_numeric(d["yhat"], errors="coerce")
    if "actual" in d.columns:
        d["actual"] = pd.to_numeric(d["actual"], errors="coerce")
    else:
        d["actual"] = np.nan

    has_is_forecast = "is_forecast" in d.columns
    if has_is_forecast:
        d["is_forecast"] = parse_bool_series(d["is_forecast"])
    elif d["actual"].notna().any():
        d["is_forecast"] = d["actual"].isna()
    else:
        d["is_forecast"] = False

    var = canonical_variable_name(variable)
    freq = normalize_frequency(frequency)
    if freq == "NA":
        freq = infer_frequency_from_dates(d["ds"])
    obs_series = get_observation_series(obs_ctx, variable=var, frequency=freq)

    d["actual_obs"] = d["actual"]
    if not obs_series.empty:
        d = d.merge(obs_series.rename(columns={"actual": "actual_from_obs"}), on="ds", how="left")
        d["actual_obs"] = d["actual_obs"].where(d["actual_obs"].notna(), d["actual_from_obs"])
    else:
        d["actual_from_obs"] = np.nan

    if has_is_forecast:
        hist_mask = ~d["is_forecast"]
    elif d["actual"].notna().any():
        hist_mask = d["actual"].notna()
    elif not obs_series.empty:
        obs_max = obs_series["ds"].max()
        hist_mask = d["ds"] <= obs_max
    else:
        hist_mask = pd.Series(False, index=d.index)

    hist_total = int(hist_mask.sum())
    hist = d[hist_mask & d["yhat"].notna() & d["actual_obs"].notna()].copy()
    n_eval = int(len(hist))
    coverage = float(n_eval / hist_total) if hist_total > 0 else float("nan")
    if n_eval == 0:
        return {
            "rmse": float("nan"),
            "bias_abs": float("nan"),
            "rmse_std": float("nan"),
            "coverage": coverage,
            "n_eval": 0.0,
            "n_hist": float(hist_total),
            "skill_naive": float("nan"),
        }

    hist = hist.sort_values("ds")
    err = hist["yhat"].to_numpy(dtype=float) - hist["actual_obs"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean(err * err)))
    bias_abs = float(abs(np.mean(err)))
    rmse_std = float(np.std(err))

    naive = hist["actual_obs"].shift(1)
    naive_df = hist.assign(naive=naive).dropna(subset=["naive", "actual_obs"])
    if len(naive_df) >= 3:
        en = naive_df["naive"].to_numpy(dtype=float) - naive_df["actual_obs"].to_numpy(dtype=float)
        rmse_naive = float(np.sqrt(np.mean(en * en)))
        skill_naive = float(1.0 - (rmse / rmse_naive)) if rmse_naive > 1e-9 else float("nan")
    else:
        skill_naive = float("nan")

    return {
        "rmse": rmse,
        "bias_abs": bias_abs,
        "rmse_std": rmse_std,
        "coverage": coverage,
        "n_eval": float(n_eval),
        "n_hist": float(hist_total),
        "skill_naive": float(skill_naive),
    }


def make_row(
    *,
    variable: str,
    frequency: str,
    model_key: str,
    model_label: str,
    forecast_csv: Path | None,
    metric_source: str,
    rmse: float,
    bias_abs: float = float("nan"),
    rmse_std: float = float("nan"),
    coverage: float = float("nan"),
    ensemble_members: float = float("nan"),
    n_eval: float = float("nan"),
    n_hist: float = float("nan"),
    skill_naive: float = float("nan"),
) -> dict[str, Any]:
    return {
        "variable": canonical_variable_name(variable),
        "frequency": normalize_frequency(frequency),
        "model_key": str(model_key),
        "model_label": str(model_label),
        "forecast_csv": str(forecast_csv) if forecast_csv is not None else "",
        "metric_source": str(metric_source),
        "rmse": safe_float(rmse),
        "bias_abs": safe_float(bias_abs),
        "rmse_std": safe_float(rmse_std),
        "coverage": safe_float(coverage),
        "ensemble_members": safe_float(ensemble_members),
        "n_eval": safe_float(n_eval),
        "n_hist": safe_float(n_hist),
        "skill_naive": safe_float(skill_naive),
    }


def collect_quant(run_dir: Path, repo_root: Path, obs_ctx: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    d = read_csv_if_exists(run_dir / "quant" / "quant_index_to_{}.csv".format(detect_target_year(run_dir)))
    if d.empty:
        # fallback: any quant index
        files = sorted((run_dir / "quant").glob("quant_index_to_*.csv"))
        if files:
            d = read_csv_if_exists(files[-1])
    if d.empty:
        return out
    for _, r in d.iterrows():
        fc = resolve_path(r.get("forecast_csv"), run_dir, repo_root)
        overlap = overlap_metrics_from_forecast(
            fc,
            variable=r.get("variable"),
            frequency=r.get("frequency"),
            obs_ctx=obs_ctx,
        ) if fc is not None else {}
        overlap_rmse = safe_float(overlap.get("rmse"))
        use_overlap = np.isfinite(overlap_rmse) and overlap_rmse > 0
        out.append(
            make_row(
                variable=r.get("variable"),
                frequency=r.get("frequency"),
                model_key="quant",
                model_label=r.get("model_strategy", "quant"),
                forecast_csv=fc,
                metric_source=("quant_overlap" if use_overlap else "quant_index"),
                rmse=(overlap_rmse if use_overlap else safe_float(r.get("cv_rmse"))),
                bias_abs=(safe_float(overlap.get("bias_abs")) if use_overlap else abs(safe_float(r.get("cv_bias")))),
                rmse_std=(safe_float(overlap.get("rmse_std")) if use_overlap else safe_float(r.get("cv_rmse_std"))),
                coverage=(safe_float(overlap.get("coverage")) if use_overlap else safe_float(r.get("monthly_coverage"))),
                n_eval=safe_float(overlap.get("n_eval")),
                n_hist=safe_float(overlap.get("n_hist")),
                skill_naive=safe_float(overlap.get("skill_naive")),
            )
        )
    return out


def collect_prophet(run_dir: Path, repo_root: Path, obs_ctx: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    files = sorted((run_dir / "prophet").glob("prophet_index_to_*.csv"))
    if not files:
        return out
    d = read_csv_if_exists(files[-1])
    if d.empty:
        return out
    for _, r in d.iterrows():
        fc = resolve_path(r.get("forecast_csv"), run_dir, repo_root)
        freq = "NA"
        if fc is not None:
            freq = infer_forecast_freq(fc)
        overlap = overlap_metrics_from_forecast(
            fc,
            variable=r.get("variable"),
            frequency=freq,
            obs_ctx=obs_ctx,
        ) if fc is not None else {}
        overlap_rmse = safe_float(overlap.get("rmse"))
        use_overlap = np.isfinite(overlap_rmse) and overlap_rmse > 0
        out.append(
            make_row(
                variable=r.get("variable"),
                frequency=freq,
                model_key="prophet",
                model_label=r.get("model_strategy", "prophet"),
                forecast_csv=fc,
                metric_source=("prophet_overlap" if use_overlap else "prophet_index"),
                rmse=(overlap_rmse if use_overlap else safe_float(r.get("holdout_rmse"))),
                bias_abs=safe_float(overlap.get("bias_abs")),
                rmse_std=safe_float(overlap.get("rmse_std")),
                coverage=safe_float(overlap.get("coverage")),
                n_eval=safe_float(overlap.get("n_eval")),
                n_hist=safe_float(overlap.get("n_hist")),
                skill_naive=safe_float(overlap.get("skill_naive")),
            )
        )
    return out


def collect_strong(run_dir: Path, repo_root: Path, obs_ctx: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    files = sorted((run_dir / "strong").glob("strong_ensemble_index_to_*.csv"))
    if not files:
        return out
    d = read_csv_if_exists(files[-1])
    if d.empty:
        return out
    for _, r in d.iterrows():
        fc = resolve_path(r.get("forecast_csv"), run_dir, repo_root)
        overlap = overlap_metrics_from_forecast(
            fc,
            variable=r.get("variable"),
            frequency=r.get("frequency"),
            obs_ctx=obs_ctx,
        ) if fc is not None else {}
        overlap_rmse = safe_float(overlap.get("rmse"))
        use_overlap = np.isfinite(overlap_rmse) and overlap_rmse > 0
        out.append(
            make_row(
                variable=r.get("variable"),
                frequency=r.get("frequency"),
                model_key="strong",
                model_label=r.get("ensemble_models", "strong"),
                forecast_csv=fc,
                metric_source=("strong_overlap" if use_overlap else "strong_index"),
                rmse=(overlap_rmse if use_overlap else safe_float(r.get("best_cv_rmse"))),
                bias_abs=safe_float(overlap.get("bias_abs")),
                rmse_std=safe_float(overlap.get("rmse_std")),
                coverage=(safe_float(overlap.get("coverage")) if use_overlap else safe_float(r.get("monthly_coverage"))),
                n_eval=safe_float(overlap.get("n_eval")),
                n_hist=safe_float(overlap.get("n_hist")),
                skill_naive=safe_float(overlap.get("skill_naive")),
            )
        )
    return out


def collect_analog(run_dir: Path, repo_root: Path, obs_ctx: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    files = sorted((run_dir / "analog").glob("analog_index_to_*.csv"))
    if not files:
        return out
    d = read_csv_if_exists(files[-1])
    if d.empty:
        return out
    for _, r in d.iterrows():
        fc = resolve_path(r.get("forecast_csv"), run_dir, repo_root)
        overlap = overlap_metrics_from_forecast(
            fc,
            variable=r.get("variable"),
            frequency=r.get("frequency"),
            obs_ctx=obs_ctx,
        ) if fc is not None else {}
        overlap_rmse = safe_float(overlap.get("rmse"))
        use_overlap = np.isfinite(overlap_rmse) and overlap_rmse > 0
        out.append(
            make_row(
                variable=r.get("variable"),
                frequency=r.get("frequency"),
                model_key="analog",
                model_label=r.get("model_strategy", "analog"),
                forecast_csv=fc,
                metric_source=("analog_overlap" if use_overlap else "analog_index"),
                rmse=(overlap_rmse if use_overlap else safe_float(r.get("rmse"))),
                bias_abs=(safe_float(overlap.get("bias_abs")) if use_overlap else abs(safe_float(r.get("bias")))),
                rmse_std=(safe_float(overlap.get("rmse_std")) if use_overlap else safe_float(r.get("rmse_std"))),
                coverage=(safe_float(overlap.get("coverage")) if use_overlap else safe_float(r.get("monthly_coverage"))),
                n_eval=safe_float(overlap.get("n_eval")),
                n_hist=safe_float(overlap.get("n_hist")),
                skill_naive=safe_float(overlap.get("skill_naive")),
            )
        )
    return out


def collect_prophet_ultra(run_dir: Path, repo_root: Path, obs_ctx: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    files = sorted((run_dir / "prophet_ultra").glob("prophet_ultra_index_to_*.csv"))
    if not files:
        return out
    d = read_csv_if_exists(files[-1])
    if d.empty:
        return out
    for _, r in d.iterrows():
        fc = resolve_path(r.get("forecast_csv"), run_dir, repo_root)
        overlap = overlap_metrics_from_forecast(
            fc,
            variable=r.get("variable"),
            frequency=r.get("frequency"),
            obs_ctx=obs_ctx,
        ) if fc is not None else {}
        overlap_rmse = safe_float(overlap.get("rmse"))
        use_overlap = np.isfinite(overlap_rmse) and overlap_rmse > 0
        out.append(
            make_row(
                variable=r.get("variable"),
                frequency=r.get("frequency"),
                model_key="prophet_ultra",
                model_label=r.get("model_strategy", "prophet_ultra"),
                forecast_csv=fc,
                metric_source=("prophet_ultra_overlap" if use_overlap else "prophet_ultra_index"),
                rmse=(overlap_rmse if use_overlap else safe_float(r.get("best_cv_rmse"))),
                bias_abs=(safe_float(overlap.get("bias_abs")) if use_overlap else abs(safe_float(r.get("bias_correction")))),
                rmse_std=safe_float(overlap.get("rmse_std")),
                coverage=(safe_float(overlap.get("coverage")) if use_overlap else safe_float(r.get("monthly_coverage"))),
                n_eval=safe_float(overlap.get("n_eval")),
                n_hist=safe_float(overlap.get("n_hist")),
                skill_naive=safe_float(overlap.get("skill_naive")),
            )
        )
    return out


def collect_walkforward(run_dir: Path, repo_root: Path, obs_ctx: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    files = sorted((run_dir / "walkforward").glob("walkforward_index_*.csv"))
    if not files:
        return out
    d = read_csv_if_exists(files[-1])
    if d.empty:
        return out
    for _, r in d.iterrows():
        fc = resolve_path(r.get("forecast_csv"), run_dir, repo_root)
        metrics = overlap_metrics_from_forecast(
            fc,
            variable=r.get("variable"),
            frequency=r.get("frequency"),
            obs_ctx=obs_ctx,
        ) if fc is not None else {}
        out.append(
            make_row(
                variable=r.get("variable"),
                frequency=r.get("frequency"),
                model_key="walkforward",
                model_label=r.get("model_strategy", "walkforward"),
                forecast_csv=fc,
                metric_source="walkforward_overlap",
                rmse=safe_float(metrics.get("rmse")),
                bias_abs=safe_float(metrics.get("bias_abs")),
                rmse_std=safe_float(metrics.get("rmse_std")),
                coverage=safe_float(metrics.get("coverage")),
                n_eval=safe_float(metrics.get("n_eval")),
                n_hist=safe_float(metrics.get("n_hist")),
                skill_naive=safe_float(metrics.get("skill_naive")),
            )
        )
    return out


def collect_best_meta(run_dir: Path, repo_root: Path, obs_ctx: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    files = sorted((run_dir / "best_meta").glob("best_meta_index_to_*.csv"))
    if not files:
        return out
    d = read_csv_if_exists(files[-1])
    if d.empty:
        return out
    for _, r in d.iterrows():
        fc = resolve_path(r.get("forecast_csv"), run_dir, repo_root)
        metrics = overlap_metrics_from_forecast(
            fc,
            variable=r.get("variable"),
            frequency=r.get("frequency"),
            obs_ctx=obs_ctx,
        ) if fc is not None else {}
        out.append(
            make_row(
                variable=r.get("variable"),
                frequency=r.get("frequency"),
                model_key="best_meta",
                model_label=r.get("model_strategy", "best_meta"),
                forecast_csv=fc,
                metric_source="best_meta_overlap",
                rmse=safe_float(metrics.get("rmse")),
                bias_abs=safe_float(metrics.get("bias_abs")),
                rmse_std=safe_float(metrics.get("rmse_std")),
                coverage=safe_float(metrics.get("coverage")),
                n_eval=safe_float(metrics.get("n_eval")),
                n_hist=safe_float(metrics.get("n_hist")),
                skill_naive=safe_float(metrics.get("skill_naive")),
            )
        )
    return out


def collect_literature(run_dir: Path, repo_root: Path, obs_ctx: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    lb_dir = run_dir / "literature" / "leaderboards"
    if not lb_dir.is_dir():
        return out
    for lb_path in sorted(lb_dir.glob("*_literature_cv_metrics.csv")):
        d = read_csv_if_exists(lb_path)
        if d.empty or "rmse_cv" not in d.columns:
            continue
        d = d.copy()
        d["rmse_cv"] = pd.to_numeric(d["rmse_cv"], errors="coerce")
        d = d.dropna(subset=["rmse_cv"])
        if d.empty:
            continue
        best = d.sort_values("rmse_cv").iloc[0]
        var = canonical_variable_name(best.get("variable"))
        freq = normalize_frequency(best.get("frequency"))
        fc = None
        fc_candidates = sorted((run_dir / "literature" / "forecasts").glob(f"{var}_{freq.lower()}_literature_forecast_to_*.csv"))
        if not fc_candidates:
            fc_candidates = sorted((run_dir / "literature" / "forecasts").glob(f"{var}_*_literature_forecast_to_*.csv"))
        if fc_candidates:
            fc = resolve_path(fc_candidates[-1], run_dir, repo_root)
        overlap = overlap_metrics_from_forecast(
            fc,
            variable=var,
            frequency=freq,
            obs_ctx=obs_ctx,
        ) if fc is not None else {}
        overlap_rmse = safe_float(overlap.get("rmse"))
        use_overlap = np.isfinite(overlap_rmse) and overlap_rmse > 0
        out.append(
            make_row(
                variable=var,
                frequency=freq,
                model_key="literature",
                model_label=f"literature_{best.get('model', 'ensemble')}",
                forecast_csv=fc,
                metric_source=("literature_overlap" if use_overlap else "literature_leaderboard"),
                rmse=(overlap_rmse if use_overlap else safe_float(best.get("rmse_cv"))),
                bias_abs=safe_float(overlap.get("bias_abs")),
                rmse_std=safe_float(overlap.get("rmse_std")),
                coverage=safe_float(overlap.get("coverage")),
                n_eval=safe_float(overlap.get("n_eval")),
                n_hist=safe_float(overlap.get("n_hist")),
                skill_naive=safe_float(overlap.get("skill_naive")),
            )
        )
    return out


def collect_stable_consensus(run_dir: Path, repo_root: Path, obs_ctx: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    model_dir = run_dir / "stable_consensus"
    files = sorted(model_dir.glob("stable_consensus_index_to_*.csv"))
    if not files:
        return out
    d = read_csv_if_exists(files[-1])
    if d.empty:
        return out
    for _, r in d.iterrows():
        fc = resolve_path(r.get("forecast_csv"), run_dir, repo_root)
        overlap = overlap_metrics_from_forecast(
            fc,
            variable=r.get("variable"),
            frequency=r.get("frequency", "MS"),
            obs_ctx=obs_ctx,
        ) if fc is not None else {}
        overlap_rmse = safe_float(overlap.get("rmse"))
        use_overlap = np.isfinite(overlap_rmse) and overlap_rmse > 0
        out.append(
            make_row(
                variable=r.get("variable"),
                frequency=r.get("frequency", "MS"),
                model_key="stable_consensus",
                model_label=r.get("model_strategy", "stable_consensus"),
                forecast_csv=fc,
                metric_source=("stable_consensus_overlap" if use_overlap else "stable_consensus_index"),
                rmse=(overlap_rmse if use_overlap else safe_float(r.get("weighted_metric_proxy"))),
                bias_abs=safe_float(overlap.get("bias_abs")),
                rmse_std=safe_float(overlap.get("rmse_std")),
                coverage=safe_float(overlap.get("coverage")),
                ensemble_members=safe_float(r.get("n_models")),
                n_eval=safe_float(overlap.get("n_eval")),
                n_hist=safe_float(overlap.get("n_hist")),
                skill_naive=safe_float(overlap.get("skill_naive")),
            )
        )
    return out


def detect_target_year(run_dir: Path) -> int:
    summary = run_dir / "model_suite_summary.json"
    if summary.exists():
        try:
            d = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            d = {}
        results = d.get("results", [])
        for row in results:
            if str(row.get("name", "")) == "quant":
                cmd = str(row.get("command", ""))
                parts = cmd.split()
                for i, tok in enumerate(parts):
                    if tok == "--target-year" and i + 1 < len(parts):
                        try:
                            return int(parts[i + 1])
                        except Exception:
                            pass
    for p in sorted((run_dir / "quant").glob("quant_index_to_*.csv")):
        try:
            return int(p.stem.split("_to_")[-1])
        except Exception:
            continue
    return 2035


def choose_preferred_frequency(df: pd.DataFrame) -> str:
    freq_rank = {"MS": 5, "YS": 4, "W": 3, "D": 2, "H": 1}
    tmp = df[np.isfinite(df["rmse"].to_numpy(dtype=float))].copy()
    if tmp.empty:
        return "NA"
    scores = {}
    for freq, g in tmp.groupby("frequency"):
        scores[str(freq)] = int(len(g)) * 10 + int(freq_rank.get(str(freq), 0))
    return sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)[0][0]


def add_scoring(df: pd.DataFrame, coverage_floor: float, freq_mismatch_penalty: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df

    out_frames = []
    selected_rows = []

    for variable, g in df.groupby("variable", dropna=False):
        g2 = g.copy().reset_index(drop=True)
        preferred_freq = choose_preferred_frequency(g2)

        pref = g2[g2["frequency"] == preferred_freq].copy()
        pref_rmse = pref["rmse"].to_numpy(dtype=float)
        pref_rmse = pref_rmse[np.isfinite(pref_rmse) & (pref_rmse > 0)]
        med_rmse = float(np.nanmedian(pref_rmse)) if len(pref_rmse) else float("nan")
        if not np.isfinite(med_rmse) or med_rmse <= 0:
            all_rmse = g2["rmse"].to_numpy(dtype=float)
            all_rmse = all_rmse[np.isfinite(all_rmse) & (all_rmse > 0)]
            med_rmse = float(np.nanmedian(all_rmse)) if len(all_rmse) else float("nan")

        pref_bias = np.abs(pref["bias_abs"].to_numpy(dtype=float))
        pref_bias = pref_bias[np.isfinite(pref_bias) & (pref_bias > 0)]
        med_bias = float(np.nanmedian(pref_bias)) if len(pref_bias) else float("nan")

        pref_std = pref["rmse_std"].to_numpy(dtype=float)
        pref_std = pref_std[np.isfinite(pref_std) & (pref_std > 0)]
        med_std = float(np.nanmedian(pref_std)) if len(pref_std) else float("nan")

        scores = []
        for _, r in g2.iterrows():
            rmse = safe_float(r.get("rmse"))
            if not np.isfinite(rmse) or rmse <= 0:
                scores.append(
                    {
                        "score_rmse": float("inf"),
                        "score_bias": float("inf"),
                        "score_std": float("inf"),
                        "score_coverage": float("inf"),
                        "score_freq": float("inf"),
                        "score_fallback": float("inf"),
                        "score_total": float("inf"),
                        "preferred_frequency": preferred_freq,
                    }
                )
                continue

            rmse_norm = rmse / med_rmse if np.isfinite(med_rmse) and med_rmse > 0 else rmse

            bias = abs(safe_float(r.get("bias_abs")))
            has_bias = np.isfinite(bias) and bias > 0
            if np.isfinite(bias) and bias > 0:
                denom = med_bias if np.isfinite(med_bias) and med_bias > 0 else max(bias, 1.0)
                score_bias = 0.18 * (bias / denom)
            else:
                score_bias = 0.0

            std = safe_float(r.get("rmse_std"))
            has_std = np.isfinite(std) and std > 0
            if np.isfinite(std) and std > 0:
                denom = med_std if np.isfinite(med_std) and med_std > 0 else max(std, 1.0)
                score_std = 0.15 * (std / denom)
            else:
                score_std = 0.0

            cov = safe_float(r.get("coverage"))
            has_cov = np.isfinite(cov)
            score_cov = max(0.0, float(coverage_floor) - cov) * 0.8 if has_cov else 0.0
            score_cov_missing = 0.14 if not has_cov else 0.0
            score_bias_missing = 0.08 if not has_bias else 0.0
            score_std_missing = 0.06 if not has_std else 0.0
            score_freq = 0.0 if str(r.get("frequency")) == preferred_freq else float(freq_mismatch_penalty)
            label = str(r.get("model_label", "")).lower()
            score_fallback = 0.20 if "seasonal_naive" in label else 0.0
            model_key = str(r.get("model_key", "")).strip().lower()
            members = safe_float(r.get("ensemble_members"))
            n_eval = safe_float(r.get("n_eval"))
            skill = safe_float(r.get("skill_naive"))
            score_weak_consensus = (
                0.22
                if model_key == "stable_consensus" and np.isfinite(members) and members <= 1.0
                else 0.0
            )
            if np.isfinite(n_eval):
                if n_eval < 12:
                    score_low_evidence = 0.28
                elif n_eval < 24:
                    score_low_evidence = 0.14
                else:
                    score_low_evidence = 0.0
            else:
                score_low_evidence = 0.10
            if np.isfinite(skill):
                score_skill = (-0.12 * min(skill, 1.0)) if skill >= 0 else (0.16 * min(abs(skill), 1.5))
            else:
                score_skill = 0.0

            total = (
                rmse_norm
                + score_bias
                + score_std
                + score_cov
                + score_cov_missing
                + score_bias_missing
                + score_std_missing
                + score_freq
                + score_fallback
                + score_weak_consensus
                + score_low_evidence
                + score_skill
            )
            scores.append(
                {
                    "score_rmse": float(rmse_norm),
                    "score_bias": float(score_bias),
                    "score_std": float(score_std),
                    "score_coverage": float(score_cov),
                    "score_cov_missing": float(score_cov_missing),
                    "score_bias_missing": float(score_bias_missing),
                    "score_std_missing": float(score_std_missing),
                    "score_freq": float(score_freq),
                    "score_fallback": float(score_fallback),
                    "score_weak_consensus": float(score_weak_consensus),
                    "score_low_evidence": float(score_low_evidence),
                    "score_skill": float(score_skill),
                    "score_total": float(total),
                    "preferred_frequency": preferred_freq,
                }
            )

        s_df = pd.DataFrame(scores)
        g2 = pd.concat([g2, s_df], axis=1)
        g2["selected"] = False

        finite = g2[np.isfinite(g2["score_total"].to_numpy(dtype=float))].copy()
        if not finite.empty:
            best = finite.sort_values("score_total").iloc[0]
            g2.loc[g2.index == best.name, "selected"] = True
            ordered = finite.sort_values("score_total").reset_index(drop=True)
            n_eval_best = safe_float(best.get("n_eval"))
            skill_best = safe_float(best.get("skill_naive"))
            if len(ordered) >= 2:
                best_s = float(ordered.loc[0, "score_total"])
                second_s = float(ordered.loc[1, "score_total"])
                gap_ratio = max(0.0, second_s - best_s) / max(second_s, 1e-9)
            else:
                gap_ratio = 0.0
            cov_best = safe_float(best.get("coverage"), default=0.70)
            conf = 55.0 + 30.0 * min(gap_ratio, 1.0) + 10.0 * min(max(cov_best, 0.0), 1.0)
            if len(ordered) < 2:
                conf -= 15.0
            missing_pen = (
                safe_float(best.get("score_cov_missing"), 0.0)
                + safe_float(best.get("score_bias_missing"), 0.0)
                + safe_float(best.get("score_std_missing"), 0.0)
            )
            conf -= 40.0 * missing_pen
            if np.isfinite(n_eval_best):
                if n_eval_best < 12:
                    conf -= 20.0
                elif n_eval_best < 24:
                    conf -= 10.0
            else:
                conf -= 8.0
            if np.isfinite(skill_best):
                conf += 12.0 * max(-0.5, min(skill_best, 1.0))
            conf -= 5.0 * (safe_float(best.get("score_bias"), 0.0) + safe_float(best.get("score_std"), 0.0))
            conf = float(np.clip(conf, 0.0, 95.0))
            if conf >= 85.0:
                grade = "A"
            elif conf >= 70.0:
                grade = "B"
            elif conf >= 55.0:
                grade = "C"
            else:
                grade = "D"
            g2.loc[g2.index == best.name, "confidence"] = conf
            g2.loc[g2.index == best.name, "confidence_grade"] = grade
            g2.loc[g2.index == best.name, "candidate_count_variable"] = int(len(ordered))
            selected_rows.append(g2.loc[g2.index == best.name].copy())

        out_frames.append(g2)

    scored = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    return scored, selected


def load_forecast_standardized(path: Path, variable: str, frequency: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    date_col = None
    for c in ("ds", "timestamp", "date"):
        if c in d.columns:
            date_col = c
            break
    if date_col is None or "yhat" not in d.columns:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["ds"] = pd.to_datetime(d[date_col], errors="coerce")
    out["yhat"] = pd.to_numeric(d.get("yhat"), errors="coerce")
    out["yhat_lower"] = pd.to_numeric(d.get("yhat_lower"), errors="coerce")
    out["yhat_upper"] = pd.to_numeric(d.get("yhat_upper"), errors="coerce")
    if "actual" in d.columns:
        out["actual"] = pd.to_numeric(d.get("actual"), errors="coerce")
    else:
        out["actual"] = np.nan
    if "is_forecast" in d.columns:
        out["is_forecast"] = d["is_forecast"].astype(bool)
    else:
        out["is_forecast"] = out["actual"].isna()
    if "variable" in d.columns:
        out["variable"] = d["variable"].map(canonical_variable_name).fillna(canonical_variable_name(variable))
    else:
        out["variable"] = canonical_variable_name(variable)
    if "frequency" in d.columns:
        out["frequency"] = d["frequency"].map(normalize_frequency).fillna(normalize_frequency(frequency))
    else:
        out["frequency"] = normalize_frequency(frequency)
    out = out.dropna(subset=["ds", "yhat"]).sort_values("ds").reset_index(drop=True)
    return out


def build_outputs(selected: pd.DataFrame, out_dir: Path, repo_root: Path) -> dict[str, Any]:
    out_fc_dir = out_dir / "forecasts"
    out_fc_dir.mkdir(parents=True, exist_ok=True)

    all_parts = []
    for _, r in selected.iterrows():
        fc_text = str(r.get("forecast_csv", "")).strip()
        if not fc_text:
            continue
        fc = Path(fc_text)
        if not fc.exists():
            continue
        var = canonical_variable_name(r.get("variable"))
        freq = normalize_frequency(r.get("frequency"))
        d = load_forecast_standardized(fc, variable=var, frequency=freq)
        if d.empty:
            continue
        d["selected_model_key"] = str(r.get("model_key", ""))
        d["selected_model_label"] = str(r.get("model_label", ""))
        d["score_total"] = safe_float(r.get("score_total"))
        out_csv = out_fc_dir / f"{var}_{freq.lower()}_robust_selected.csv"
        d.to_csv(out_csv, index=False)
        d_future = d[d["is_forecast"] == True].copy()
        if not d_future.empty:
            d_future.to_csv(out_fc_dir / f"{var}_{freq.lower()}_robust_selected_future.csv", index=False)
        all_parts.append(d)

    outputs: dict[str, Any] = {}
    if all_parts:
        combo = pd.concat(all_parts, ignore_index=True).sort_values(["variable", "ds"])
        combo_csv = out_dir / "robust_selected_forecasts.csv"
        combo.to_csv(combo_csv, index=False)
        combo_future = combo[combo["is_forecast"] == True].copy()
        combo_future_csv = out_dir / "robust_selected_future_forecasts.csv"
        combo_future.to_csv(combo_future_csv, index=False)
        outputs["combined_forecasts_csv"] = rel_to_repo(combo_csv, repo_root)
        outputs["combined_future_forecasts_csv"] = rel_to_repo(combo_future_csv, repo_root)
    return outputs


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run-dir not found: {run_dir}")
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (args.output_dir.resolve() if args.output_dir is not None else (run_dir / "robust_selection").resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_ctx = load_observation_context(run_dir=run_dir, repo_root=repo_root)

    rows: list[dict[str, Any]] = []
    rows.extend(collect_quant(run_dir, repo_root, obs_ctx))
    rows.extend(collect_prophet(run_dir, repo_root, obs_ctx))
    rows.extend(collect_strong(run_dir, repo_root, obs_ctx))
    rows.extend(collect_analog(run_dir, repo_root, obs_ctx))
    rows.extend(collect_prophet_ultra(run_dir, repo_root, obs_ctx))
    rows.extend(collect_walkforward(run_dir, repo_root, obs_ctx))
    rows.extend(collect_best_meta(run_dir, repo_root, obs_ctx))
    rows.extend(collect_literature(run_dir, repo_root, obs_ctx))
    rows.extend(collect_stable_consensus(run_dir, repo_root, obs_ctx))

    cand = pd.DataFrame(rows)
    if cand.empty:
        raise SystemExit("No candidate metrics found in run directory.")
    cand["variable"] = cand["variable"].map(canonical_variable_name)
    cand["frequency"] = cand["frequency"].map(normalize_frequency)
    for c in ("rmse", "bias_abs", "rmse_std", "coverage"):
        cand[c] = pd.to_numeric(cand[c], errors="coerce")

    scored, selected = add_scoring(
        cand,
        coverage_floor=float(args.coverage_floor),
        freq_mismatch_penalty=float(args.freq_mismatch_penalty),
    )
    if selected.empty:
        raise SystemExit("Could not select any robust model (all scores invalid).")

    candidates_csv = out_dir / "robust_model_candidates.csv"
    selected_csv = out_dir / "robust_model_selected.csv"
    scored.to_csv(candidates_csv, index=False)
    selected.to_csv(selected_csv, index=False)

    export_outputs = build_outputs(selected, out_dir, repo_root)

    selected_items = []
    for _, r in selected.sort_values("variable").iterrows():
        selected_items.append(
            {
                "variable": str(r.get("variable", "")),
                "frequency": str(r.get("frequency", "")),
                "preferred_frequency": str(r.get("preferred_frequency", "")),
                "model_key": str(r.get("model_key", "")),
                "model_label": str(r.get("model_label", "")),
                "metric_source": str(r.get("metric_source", "")),
                "rmse": safe_float(r.get("rmse")),
                "bias_abs": safe_float(r.get("bias_abs")),
                "rmse_std": safe_float(r.get("rmse_std")),
                "coverage": safe_float(r.get("coverage")),
                "ensemble_members": safe_float(r.get("ensemble_members")),
                "n_eval": safe_float(r.get("n_eval")),
                "n_hist": safe_float(r.get("n_hist")),
                "skill_naive": safe_float(r.get("skill_naive")),
                "score_total": safe_float(r.get("score_total")),
                "confidence": safe_float(r.get("confidence")),
                "confidence_grade": str(r.get("confidence_grade", "")),
                "forecast_csv": rel_to_repo(Path(str(r.get("forecast_csv"))), repo_root)
                if str(r.get("forecast_csv", "")).strip()
                else "",
            }
        )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": rel_to_repo(run_dir, repo_root),
        "candidates_count": int(len(scored)),
        "selected_count": int(len(selected)),
        "variables_selected": sorted({str(x) for x in selected["variable"].tolist()}),
        "selected_models": selected_items,
        "outputs": {
            "candidates_csv": rel_to_repo(candidates_csv, repo_root),
            "selected_csv": rel_to_repo(selected_csv, repo_root),
            **export_outputs,
        },
    }

    summary_json = out_dir / "robust_model_selection_summary.json"
    summary_md = out_dir / "robust_model_selection_summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Robust Model Selection Summary",
        "",
        f"- Run dir: `{summary['run_dir']}`",
        f"- Candidate count: `{summary['candidates_count']}`",
        f"- Selected count: `{summary['selected_count']}`",
        "",
        "## Selected Models",
    ]
    for item in selected_items:
        lines.append(
            "- `{variable}` ({frequency}) -> `{model_key}` | rmse={rmse:.4f} | score={score:.4f} | conf={conf:.1f} ({grade})".format(
                variable=item["variable"],
                frequency=item["frequency"],
                model_key=item["model_key"],
                rmse=float(item["rmse"]) if np.isfinite(item["rmse"]) else float("nan"),
                score=float(item["score_total"]) if np.isfinite(item["score_total"]) else float("nan"),
                conf=float(item["confidence"]) if np.isfinite(item["confidence"]) else float("nan"),
                grade=item["confidence_grade"] or "-",
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
