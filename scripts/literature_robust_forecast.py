#!/usr/bin/env python3
"""Literature-backed robust climate forecasting pipeline.

Model design (from literature-guided best practices):
- Leakage-safe rolling-origin backtesting (expanding window)
- Multiple model families (seasonal naive + linear + tree ensembles)
- Forecast combination by CV quality (M4-style ensemble intuition)
- Scale-free metrics (MASE + sMAPE) for fair model ranking
- Split-conformal prediction intervals for calibrated uncertainty
- Optional teleconnection exogenous features (ONI/NAO) when reachable
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Restrict thread usage for sandboxed/macOS SHM stability.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
_CACHE_ROOT = Path(tempfile.gettempdir()) / "lit_robust_cache"
_MPL_CACHE = _CACHE_ROOT / "mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ONI_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"
NAO_ASCII_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"

TR_CHARMAP = str.maketrans(
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

UNIT_MAP = {"humidity": "%", "temp": "C", "pressure": "hPa", "precip": "mm"}

@dataclass
class ModelCVStats:
    model: str
    mae: float
    rmse: float
    smape: float
    mase: float
    n_splits: int
    score: float
    weight: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Literature-backed robust climate forecast")
    p.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="Input table (csv/parquet/xlsx/ods) with time series data.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/literature_robust_forecast"),
    )
    p.add_argument("--variables", type=str, default="*")
    p.add_argument("--target-year", type=int, default=2035)
    p.add_argument("--input-kind", type=str, default="auto", choices=["auto", "long", "single", "wide"])
    p.add_argument("--timestamp-col", type=str, default="timestamp")
    p.add_argument("--value-col", type=str, default="value")
    p.add_argument("--variable-col", type=str, default="variable")
    p.add_argument("--single-variable", type=str, default="target")
    p.add_argument("--max-lag", type=int, default=24)
    p.add_argument("--cv-splits", type=int, default=6)
    p.add_argument("--min-train-points", type=int, default=60)
    p.add_argument("--alpha", type=float, default=0.1, help="Conformal alpha. 0.1 -> 90% interval.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-teleconnections", type=str, default="true")
    p.add_argument("--max-forecast-steps", type=int, default=5000)
    return p.parse_args()


def to_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def normalize_token(text: Any) -> str:
    s = str(text).strip().lower().translate(TR_CHARMAP)
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


def unit_for(variable: str) -> str:
    return UNIT_MAP.get(canonical_variable_name(variable), "unknown")


def apply_bounds(x: np.ndarray, variable: str) -> np.ndarray:
    var = canonical_variable_name(variable)
    y = np.asarray(x, dtype=float)
    if var == "humidity":
        return np.clip(y, 0.0, 100.0)
    if var == "temp":
        return np.clip(y, -60.0, 65.0)
    if var in {"precip", "pressure"}:
        return np.clip(y, 0.0, None)
    return y


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suf in {".xlsx", ".xls", ".ods"}:
        return pd.read_excel(path)
    raise SystemExit(f"Unsupported extension: {path.suffix}")


def pick_col(df: pd.DataFrame, preferred: str, fallbacks: list[str]) -> str | None:
    if preferred in df.columns:
        return preferred
    for c in fallbacks:
        if c in df.columns:
            return c
    return None


def normalize_observations(raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    ts_col = pick_col(raw, args.timestamp_col, ["timestamp", "ds", "date", "datetime", "time", "tarih"])
    val_col = pick_col(raw, args.value_col, ["value", "y", "target", "measurement"])
    var_col = pick_col(raw, args.variable_col, ["variable", "metric", "param", "sensor", "name"])
    if ts_col is None:
        raise SystemExit("Timestamp column not found.")

    kind = args.input_kind
    if kind == "auto":
        if var_col is not None and val_col is not None:
            kind = "long"
        elif val_col is not None:
            kind = "single"
        else:
            kind = "wide"

    if kind == "long":
        if val_col is None or var_col is None:
            raise SystemExit("Long format icin value + variable column gerekli.")
        df = raw[[ts_col, var_col, val_col]].copy()
        df.columns = ["timestamp", "variable", "value"]
    elif kind == "single":
        if val_col is None:
            raise SystemExit("Single format icin value column gerekli.")
        df = raw[[ts_col, val_col]].copy()
        df.columns = ["timestamp", "value"]
        df["variable"] = str(args.single_variable)
    else:
        # Wide: timestamp + many numeric columns.
        wide = raw.copy()
        keep = [c for c in wide.columns if c != ts_col]
        num_cols = [c for c in keep if pd.to_numeric(wide[c], errors="coerce").notna().mean() > 0.7]
        if not num_cols:
            raise SystemExit("Wide format: numeric variable columns bulunamadi.")
        df = wide[[ts_col] + num_cols].melt(id_vars=[ts_col], var_name="variable", value_name="value")
        df.columns = ["timestamp", "variable", "value"]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["variable"] = df["variable"].astype(str).map(canonical_variable_name)
    df = df.dropna(subset=["timestamp", "value", "variable"]).copy()
    df = df.groupby(["timestamp", "variable"], as_index=False)["value"].mean()
    df = df.sort_values(["variable", "timestamp"]).reset_index(drop=True)
    return df


def infer_freq(index: pd.DatetimeIndex) -> str:
    if len(index) < 3:
        return "MS"
    f = pd.infer_freq(index)
    if f:
        if f.startswith("W"):
            return "W-MON"
        if f.startswith("Q"):
            return "QS"
        if f.startswith("A") or f.startswith("Y"):
            return "YS"
        if f.startswith("M"):
            return "MS"
        if f.startswith("H"):
            return "h"
        if f.startswith("D"):
            return "D"
        return f
    d = np.median(np.diff(index.values).astype("timedelta64[h]").astype(float))
    if d <= 2:
        return "h"
    if d <= 36:
        return "D"
    if d <= 24 * 10:
        return "W-MON"
    if d <= 24 * 45:
        return "MS"
    if d <= 24 * 130:
        return "QS"
    return "YS"


def season_length(freq: str) -> int:
    ff = str(freq).lower()
    if ff.startswith("h"):
        return 24
    if ff.startswith("d"):
        return 7
    if ff.startswith("w"):
        return 52
    if ff.startswith("m"):
        return 12
    if ff.startswith("q"):
        return 4
    if ff.startswith("y") or ff.startswith("a"):
        return 1
    return 12


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(200.0 * np.mean(np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


def mase(y_true: np.ndarray, y_pred: np.ndarray, train: np.ndarray, m: int) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    train = np.asarray(train, dtype=float)
    if len(train) <= m:
        m = 1
    denom = np.mean(np.abs(train[m:] - train[:-m])) if len(train) > m else np.nan
    if not np.isfinite(denom) or denom <= 0:
        return float(np.mean(np.abs(y_true - y_pred)))
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def monthly_key(ts: pd.Timestamp) -> str:
    return f"{int(ts.year):04d}-{int(ts.month):02d}"


def fetch_oni_nao() -> dict[str, dict[str, float]]:
    out = {"oni": {}, "nao": {}}
    try:
        with urllib.request.urlopen(ONI_URL, timeout=18) as r:
            html = r.read().decode("utf-8", errors="ignore")
        try:
            tables = pd.read_html(io.StringIO(html), flavor="lxml")
        except Exception:
            tables = pd.read_html(io.StringIO(html))
        target = None
        for t in tables:
            tt = t.copy()
            if not tt.empty and str(tt.iloc[0, 0]).strip().lower() == "year":
                tt.columns = tt.iloc[0]
                tt = tt.iloc[1:].copy()
            cols = [str(c).strip().upper() for c in tt.columns]
            if "YEAR" in cols and "DJF" in cols and "NDJ" in cols:
                target = tt.copy()
                target.columns = [str(c).strip().upper() for c in target.columns]
                break
        if target is not None:
            season_to_month = {
                "DJF": 1,
                "JFM": 2,
                "FMA": 3,
                "MAM": 4,
                "AMJ": 5,
                "MJJ": 6,
                "JJA": 7,
                "JAS": 8,
                "ASO": 9,
                "SON": 10,
                "OND": 11,
                "NDJ": 12,
            }
            for _, row in target.iterrows():
                yy = int(pd.to_numeric(row.get("YEAR"), errors="coerce"))
                for s, m in season_to_month.items():
                    v = pd.to_numeric(row.get(s), errors="coerce")
                    if np.isfinite(v):
                        out["oni"][f"{yy:04d}-{m:02d}"] = float(v)
    except Exception:
        pass

    try:
        with urllib.request.urlopen(NAO_ASCII_URL, timeout=18) as r:
            txt = r.read().decode("utf-8", errors="ignore")
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            y = pd.to_numeric(parts[0], errors="coerce")
            m = pd.to_numeric(parts[1], errors="coerce")
            v = pd.to_numeric(parts[2], errors="coerce")
            if np.isfinite(y) and np.isfinite(m) and np.isfinite(v):
                out["nao"][f"{int(y):04d}-{int(m):02d}"] = float(v)
    except Exception:
        pass
    return out


def build_supervised(
    y: pd.Series,
    max_lag: int,
    seasonal_m: int,
    tele: dict[str, dict[str, float]] | None,
) -> pd.DataFrame:
    df = pd.DataFrame({"y": y.astype(float)})
    lags = sorted({1, 2, 3, 6, 12, 24, max_lag})
    lags = [l for l in lags if l <= max_lag]
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for w in [3, 6, 12]:
        if w <= max_lag:
            base = df["y"].shift(1)
            df[f"roll_mean_{w}"] = base.rolling(w).mean()
            df[f"roll_std_{w}"] = base.rolling(w).std()
    ts = pd.DatetimeIndex(df.index)
    df["month_sin"] = np.sin(2 * np.pi * ts.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * ts.month / 12.0)
    df["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7.0)
    df["trend"] = np.linspace(0.0, 1.0, len(df), endpoint=True)
    if seasonal_m > 1:
        df[f"seasonal_lag_{seasonal_m}"] = df["y"].shift(seasonal_m)
    if tele is not None and (tele.get("oni") or tele.get("nao")):
        keys = [monthly_key(pd.Timestamp(x)) for x in ts]
        oni_vals = np.array([tele["oni"].get(k, np.nan) for k in keys], dtype=float)
        nao_vals = np.array([tele["nao"].get(k, np.nan) for k in keys], dtype=float)
        min_non_nan = max(3, int(0.05 * len(df)))
        if np.isfinite(oni_vals).sum() >= min_non_nan:
            s_oni = pd.Series(oni_vals, index=df.index).interpolate().ffill().bfill().fillna(0.0)
            df["oni"] = s_oni.astype(float)
        if np.isfinite(nao_vals).sum() >= min_non_nan:
            s_nao = pd.Series(nao_vals, index=df.index).interpolate().ffill().bfill().fillna(0.0)
            df["nao"] = s_nao.astype(float)
    return df


def feature_row_for_future(
    ts: pd.Timestamp,
    history: list[float],
    max_lag: int,
    seasonal_m: int,
    tele: dict[str, dict[str, float]] | None,
    trend_pos: float,
) -> dict[str, float]:
    row: dict[str, float] = {}
    lags = sorted({1, 2, 3, 6, 12, 24, max_lag})
    lags = [l for l in lags if l <= max_lag]
    for lag in lags:
        row[f"lag_{lag}"] = history[-lag] if len(history) >= lag else np.nan
    for w in [3, 6, 12]:
        if w <= max_lag:
            if len(history) >= w:
                vals = np.array(history[-w:], dtype=float)
                row[f"roll_mean_{w}"] = float(np.mean(vals))
                row[f"roll_std_{w}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            else:
                row[f"roll_mean_{w}"] = np.nan
                row[f"roll_std_{w}"] = np.nan
    row["month_sin"] = math.sin(2 * math.pi * ts.month / 12.0)
    row["month_cos"] = math.cos(2 * math.pi * ts.month / 12.0)
    row["doy_sin"] = math.sin(2 * math.pi * ts.dayofyear / 365.25)
    row["doy_cos"] = math.cos(2 * math.pi * ts.dayofyear / 365.25)
    row["dow_sin"] = math.sin(2 * math.pi * ts.dayofweek / 7.0)
    row["dow_cos"] = math.cos(2 * math.pi * ts.dayofweek / 7.0)
    row["trend"] = float(trend_pos)
    if seasonal_m > 1:
        row[f"seasonal_lag_{seasonal_m}"] = history[-seasonal_m] if len(history) >= seasonal_m else np.nan
    if tele is not None and (tele.get("oni") or tele.get("nao")):
        k = monthly_key(ts)
        row["oni"] = float(tele["oni"].get(k, 0.0)) if tele.get("oni") else 0.0
        row["nao"] = float(tele["nao"].get(k, 0.0)) if tele.get("nao") else 0.0
    return row


def model_pool(seed: int) -> dict[str, Any]:
    return {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "gbrt": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=500,
                        learning_rate=0.03,
                        max_depth=3,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "huber": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    HuberRegressor(epsilon=1.35, alpha=0.0003, max_iter=3000),
                ),
            ]
        ),
    }


def rolling_train_end_points(n_obs: int, min_train: int, test_size: int, n_splits: int) -> list[int]:
    max_train = n_obs - test_size
    if max_train <= min_train:
        return []
    pts = np.linspace(min_train, max_train, num=max(2, n_splits), dtype=int).tolist()
    out = sorted(set(int(x) for x in pts if min_train <= x <= max_train))
    return out


def recursive_forecast_model(
    model: Any,
    feature_cols: list[str],
    history: list[float],
    future_index: pd.DatetimeIndex,
    max_lag: int,
    seasonal_m: int,
    tele: dict[str, dict[str, float]] | None,
) -> np.ndarray:
    preds: list[float] = []
    hist = list(history)
    base_n = len(history)
    for i, ts in enumerate(future_index, start=1):
        row = feature_row_for_future(
            ts=pd.Timestamp(ts),
            history=hist,
            max_lag=max_lag,
            seasonal_m=seasonal_m,
            tele=tele,
            trend_pos=(base_n + i) / (base_n + len(future_index) + 1),
        )
        x = np.array([[row.get(c, np.nan) for c in feature_cols]], dtype=float)
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        hist.append(yhat)
    return np.array(preds, dtype=float)


def seasonal_naive_forecast(history: list[float], horizon: int, m: int) -> np.ndarray:
    if not history:
        return np.zeros(horizon, dtype=float)
    if m <= 1 or len(history) < m:
        return np.full(horizon, history[-1], dtype=float)
    out = []
    for h in range(horizon):
        out.append(history[-m + (h % m)])
    return np.array(out, dtype=float)


def evaluate_variable(
    var: str,
    series: pd.Series,
    freq: str,
    args: argparse.Namespace,
    tele: dict[str, dict[str, float]] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    y = series.sort_index().astype(float)
    m = season_length(freq)
    sup = build_supervised(y=y, max_lag=args.max_lag, seasonal_m=m, tele=tele).dropna().copy()
    if len(sup) < max(args.min_train_points, args.max_lag + 20):
        raise ValueError(f"{var}: yeterli veri yok (usable={len(sup)}).")

    feature_cols = [c for c in sup.columns if c != "y"]
    models = model_pool(seed=args.seed)
    test_size = max(4, m)
    n_obs = len(y)
    min_train = max(args.min_train_points, args.max_lag + m + 5)
    split_points = rolling_train_end_points(n_obs=n_obs, min_train=min_train, test_size=test_size, n_splits=args.cv_splits)
    if not split_points:
        raise ValueError(f"{var}: rolling split kurulamadı.")

    perf_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    per_model_errors: dict[str, list[dict[str, float]]] = {"seasonal_naive": []}
    for k in models:
        per_model_errors[k] = []

    for split_id, train_end in enumerate(split_points, start=1):
        train_end_ts = y.index[train_end - 1]
        test_idx = y.index[train_end : train_end + test_size]
        if len(test_idx) < test_size:
            continue

        train_sup = sup.loc[sup.index <= train_end_ts]
        if len(train_sup) < min_train:
            continue

        # Seasonal naive baseline.
        naive_pred = seasonal_naive_forecast(history=y.iloc[:train_end].tolist(), horizon=test_size, m=m)
        actual = y.loc[test_idx].to_numpy(dtype=float)
        per_model_errors["seasonal_naive"].append({"actual": actual, "pred": naive_pred, "train": y.iloc[:train_end].to_numpy()})

        split_model_preds: dict[str, np.ndarray] = {"seasonal_naive": naive_pred}
        for name, mdl in models.items():
            mdl.fit(train_sup[feature_cols].to_numpy(dtype=float), train_sup["y"].to_numpy(dtype=float))
            pred = recursive_forecast_model(
                model=mdl,
                feature_cols=feature_cols,
                history=y.iloc[:train_end].tolist(),
                future_index=pd.DatetimeIndex(test_idx),
                max_lag=args.max_lag,
                seasonal_m=m,
                tele=tele,
            )
            split_model_preds[name] = pred
            per_model_errors[name].append({"actual": actual, "pred": pred, "train": y.iloc[:train_end].to_numpy()})

        for ts, yy in zip(test_idx, actual, strict=True):
            rec: dict[str, Any] = {"split": split_id, "timestamp": ts, "actual": yy}
            for nm, pp in split_model_preds.items():
                rec[f"pred_{nm}"] = float(pp[list(test_idx).index(ts)])
            oof_rows.append(rec)

    # Aggregate model CV metrics and derive weights.
    cv_stats: list[ModelCVStats] = []
    for model_name, chunks in per_model_errors.items():
        if not chunks:
            continue
        mae_list, rmse_list, smape_list, mase_list = [], [], [], []
        for ch in chunks:
            a = np.asarray(ch["actual"], dtype=float)
            p = np.asarray(ch["pred"], dtype=float)
            tr = np.asarray(ch["train"], dtype=float)
            mae_list.append(float(np.mean(np.abs(a - p))))
            rmse_list.append(float(np.sqrt(np.mean((a - p) ** 2))))
            smape_list.append(smape(a, p))
            mase_list.append(mase(a, p, tr, m))
        mae_v = float(np.mean(mae_list))
        rmse_v = float(np.mean(rmse_list))
        smape_v = float(np.mean(smape_list))
        mase_v = float(np.mean(mase_list))
        score = 0.5 * smape_v + 0.5 * mase_v
        cv_stats.append(
            ModelCVStats(
                model=model_name,
                mae=mae_v,
                rmse=rmse_v,
                smape=smape_v,
                mase=mase_v,
                n_splits=len(chunks),
                score=score,
                weight=0.0,
            )
        )

    if not cv_stats:
        raise ValueError(f"{var}: CV metrikleri üretilemedi.")

    inv = np.array([1.0 / max(s.score, 1e-6) for s in cv_stats], dtype=float)
    inv = inv / inv.sum()
    for i, s in enumerate(cv_stats):
        s.weight = float(inv[i])

    weights = {s.model: s.weight for s in cv_stats}

    # Train final models on all available data and produce forecast.
    full_train = sup.copy()
    trained_models: dict[str, Any] = {}
    for name, mdl in models.items():
        mdl.fit(full_train[feature_cols].to_numpy(dtype=float), full_train["y"].to_numpy(dtype=float))
        trained_models[name] = mdl

    last_ts = y.index.max()
    future_idx = pd.date_range(
        start=last_ts + pd.tseries.frequencies.to_offset(freq),
        end=pd.Timestamp(f"{args.target_year}-12-31"),
        freq=freq,
    )
    if len(future_idx) == 0:
        raise ValueError(f"{var}: target-year icin horizon olusmadi.")
    if len(future_idx) > int(args.max_forecast_steps):
        future_idx = future_idx[: int(args.max_forecast_steps)]

    history_vals = y.tolist()
    model_forecasts: dict[str, np.ndarray] = {}
    model_forecasts["seasonal_naive"] = seasonal_naive_forecast(history_vals, len(future_idx), m)
    for name, mdl in trained_models.items():
        model_forecasts[name] = recursive_forecast_model(
            model=mdl,
            feature_cols=feature_cols,
            history=history_vals,
            future_index=future_idx,
            max_lag=args.max_lag,
            seasonal_m=m,
            tele=tele,
        )

    ens = np.zeros(len(future_idx), dtype=float)
    for k, pred in model_forecasts.items():
        w = weights.get(k, 0.0)
        ens += w * pred
    ens = apply_bounds(ens, var)

    # OOF conformal calibration.
    oof = pd.DataFrame(oof_rows).sort_values("timestamp")
    if not oof.empty:
        oof_ens = np.zeros(len(oof), dtype=float)
        for k, w in weights.items():
            c = f"pred_{k}"
            if c in oof.columns:
                oof_ens += w * oof[c].to_numpy(dtype=float)
        abs_resid = np.abs(oof["actual"].to_numpy(dtype=float) - oof_ens)
        q = float(np.quantile(abs_resid, 1 - args.alpha))
        if not np.isfinite(q):
            q = float(np.nanstd(abs_resid)) if len(abs_resid) else 0.0
    else:
        q = float(np.nanstd(y.to_numpy(dtype=float)))
    lower = apply_bounds(ens - q, var)
    upper = apply_bounds(ens + q, var)

    forecast_df = pd.DataFrame(
        {
            "timestamp": future_idx,
            "variable": var,
            "frequency": freq,
            "model": "literature_ensemble",
            "yhat": ens,
            "yhat_lower": lower,
            "yhat_upper": upper,
            "interval_alpha": float(args.alpha),
            "interval_method": "split_conformal_abs_residual",
            "unit": unit_for(var),
        }
    )

    cv_df = pd.DataFrame(
        [
            {
                "variable": var,
                "frequency": freq,
                "model": s.model,
                "mae_cv": s.mae,
                "rmse_cv": s.rmse,
                "smape_cv": s.smape,
                "mase_cv": s.mase,
                "score": s.score,
                "weight": s.weight,
                "n_splits": s.n_splits,
            }
            for s in sorted(cv_stats, key=lambda x: x.score)
        ]
    )
    oof_out = oof.copy()
    if not oof_out.empty:
        oof_ens = np.zeros(len(oof_out), dtype=float)
        for k, w in weights.items():
            c = f"pred_{k}"
            if c in oof_out.columns:
                oof_ens += w * oof_out[c].to_numpy(dtype=float)
        oof_out["pred_ensemble"] = oof_ens
        oof_out["abs_error_ensemble"] = np.abs(oof_out["actual"] - oof_out["pred_ensemble"])
        oof_out["variable"] = var
        oof_out["frequency"] = freq

    meta = {
        "variable": var,
        "frequency": freq,
        "n_points": int(len(y)),
        "n_future": int(len(future_idx)),
        "season_length": int(m),
        "conformal_q": float(q),
        "weights": weights,
    }
    return forecast_df, cv_df, oof_out, meta


def ensure_dirs(out: Path) -> dict[str, Path]:
    d = {
        "forecasts": out / "forecasts",
        "leaderboards": out / "leaderboards",
        "reports": out / "reports",
        "charts": out / "charts",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def write_literature_note(path: Path) -> None:
    lines = [
        "# Literature-Backed Method Notes",
        "",
        "Bu pipeline aşağıdaki literatür ilkelerini uygular:",
        "- Rolling-origin zaman serisi doğrulama (leakage güvenli)",
        "- sMAPE ve MASE ile ölçek-bağımsız model kıyası",
        "- Çoklu model kombinasyonu (ensemble weighting)",
        "- Split-conformal ile kalibre belirsizlik aralığı",
        "- ENSO/NAO dışsal sürücülerinin (varsa) özellik olarak eklenmesi",
        "",
        "## Kaynaklar",
        "- Bergmeir C, Hyndman RJ, Koo B (2018): https://doi.org/10.1016/j.csda.2018.03.006",
        "- Hyndman RJ, Koehler AB (2006): https://doi.org/10.1016/j.ijforecast.2006.03.001",
        "- Makridakis S et al. M4 (2020): https://doi.org/10.1016/j.ijforecast.2019.04.014",
        "- Romano Y et al. CQR (2019): https://arxiv.org/abs/1905.03222",
        "- WeatherBench (2020): https://arxiv.org/abs/2002.00469",
        "- GraphCast (2023): https://arxiv.org/abs/2212.12794",
        "- NOAA CPC ONI: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php",
        "- NOAA CPC NAO index: https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii",
        "",
        "Not: Bu script nedensellik değil, tahmin performansı ve kalibre belirsizlik üretir.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir
    dirs = ensure_dirs(out)

    raw = read_table(args.observations)
    obs = normalize_observations(raw, args=args)

    selected_vars = None
    if str(args.variables).strip() != "*":
        selected_vars = {canonical_variable_name(v) for v in str(args.variables).split(",") if v.strip()}
        obs = obs[obs["variable"].isin(selected_vars)].copy()
    if obs.empty:
        raise SystemExit("İşlenecek veri bulunamadı (variables filtresini kontrol et).")

    tele = fetch_oni_nao() if to_bool(args.use_teleconnections) else None
    all_forecasts = []
    all_cv = []
    all_oof = []
    meta_rows = []
    skipped_vars = []

    for var, g in obs.groupby("variable"):
        s = (
            g.sort_values("timestamp")
            .set_index("timestamp")["value"]
            .astype(float)
            .groupby(level=0)
            .mean()
        )
        freq = infer_freq(pd.DatetimeIndex(s.index))
        # Regularize to frequency grid and interpolate.
        s = s.asfreq(freq)
        if s.isna().mean() > 0.6:
            # Too sparse -> fallback monthly aggregation.
            s = g.set_index("timestamp")["value"].resample("MS").mean()
            freq = "MS"
        s = s.interpolate(method="time").ffill().bfill()

        try:
            fc, cv, oof, meta = evaluate_variable(
                var=var,
                series=s,
                freq=freq,
                args=args,
                tele=tele,
            )
        except ValueError as exc:
            skipped_vars.append({"variable": var, "reason": str(exc), "frequency": freq})
            continue
        all_forecasts.append(fc)
        all_cv.append(cv)
        if not oof.empty:
            all_oof.append(oof)
        meta_rows.append(meta)

        fc_csv = dirs["forecasts"] / f"{var}_{freq.lower()}_literature_forecast_to_{args.target_year}.csv"
        fc_parquet = dirs["forecasts"] / f"{var}_{freq.lower()}_literature_forecast_to_{args.target_year}.parquet"
        fc.to_csv(fc_csv, index=False)
        fc.to_parquet(fc_parquet, index=False)

        cv_csv = dirs["leaderboards"] / f"{var}_{freq.lower()}_literature_cv_metrics.csv"
        cv.to_csv(cv_csv, index=False)

        plt.figure(figsize=(12, 5))
        hist_tail = s.iloc[max(0, len(s) - 180) :]
        plt.plot(hist_tail.index, hist_tail.values, label="history", lw=1.4, color="#1f3b73")
        plt.plot(fc["timestamp"], fc["yhat"], label="forecast", lw=1.8, color="#cc4f1b")
        plt.fill_between(
            fc["timestamp"],
            fc["yhat_lower"],
            fc["yhat_upper"],
            color="#cc4f1b",
            alpha=0.18,
            label=f"{int((1-args.alpha)*100)}% interval",
        )
        plt.title(f"{var} | literature_ensemble | freq={freq}")
        plt.xlabel("tarih")
        plt.ylabel(f"deger ({unit_for(var)})")
        plt.legend(loc="best")
        plt.tight_layout()
        fig_path = dirs["charts"] / f"{var}_{freq.lower()}_literature_forecast.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

    if not all_forecasts:
        raise SystemExit("Hiçbir değişken literature modeli için yeterli veri sağlamadı.")

    forecast_all = pd.concat(all_forecasts, ignore_index=True)
    cv_all = pd.concat(all_cv, ignore_index=True) if all_cv else pd.DataFrame()
    oof_all = pd.concat(all_oof, ignore_index=True) if all_oof else pd.DataFrame()
    meta_df = pd.DataFrame(meta_rows)

    idx_csv = out / f"literature_forecast_index_to_{args.target_year}.csv"
    idx_parquet = out / f"literature_forecast_index_to_{args.target_year}.parquet"
    forecast_all.to_csv(idx_csv, index=False)
    forecast_all.to_parquet(idx_parquet, index=False)
    if not cv_all.empty:
        cv_all.to_csv(out / "literature_cv_metrics_all.csv", index=False)
    if not oof_all.empty:
        oof_all.to_csv(out / "literature_oof_predictions.csv", index=False)
    meta_df.to_csv(out / "literature_run_meta.csv", index=False)
    (out / "literature_run_meta.json").write_text(json.dumps(meta_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_literature_note(dirs["reports"] / "literature_method_notes.md")
    if skipped_vars:
        skipped_df = pd.DataFrame(skipped_vars)
        skipped_df.to_csv(dirs["reports"] / "literature_skipped_variables.csv", index=False)
        (dirs["reports"] / "literature_skipped_variables.json").write_text(
            json.dumps(skipped_vars, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Wrote: {idx_csv}")
    print(f"Wrote: {idx_parquet}")
    print(f"Wrote: {out / 'literature_cv_metrics_all.csv'}")
    print(f"Wrote: {out / 'literature_run_meta.json'}")
    if skipped_vars:
        print(f"Skipped variables: {len(skipped_vars)} (details in reports/literature_skipped_variables.*)")
    print("Tamamlandı: literature_robust_forecast")


if __name__ == "__main__":
    main()
