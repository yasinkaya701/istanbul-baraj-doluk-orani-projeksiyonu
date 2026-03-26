#!/usr/bin/env python3
"""Quant-style climate forecasting pipeline.

Core idea (finance-inspired):
- Mean model: deterministic trend/seasonality (Ridge regression)
- Regime model: Markov Switching on residuals (2-state by default)
- Volatility model: EGARCH (if available) else EWMA, with regime variance blend
- Uncertainty calibration: rolling CV + conformal quantiles
- Anomaly engine: standardized residual spikes + tail threshold

This is intentionally robust to sparse, mixed-frequency climate archives.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "quant_climate_cache"
_MPL_CACHE = _CACHE_ROOT / "mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from climate_scenario_adjustment import (
    climate_delta_series,
    climate_strategy_suffix,
    from_args as climate_cfg_from_args,
    with_series_baseline as climate_cfg_with_series_baseline,
)
try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

UNIT_MAP = {
    "humidity": "%",
    "temp": "C",
    "pressure": "hPa",
    "precip": "mm",
    "et0": "mm",
}

VAR_TR_MAP = {
    "humidity": "Nem",
    "temp": "Sıcaklık",
    "pressure": "Basınç",
    "precip": "Yağış",
    "et0": "ET0",
}

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
    "et0": "et0",
    "eto": "et0",
    "reference_et": "et0",
    "reference_et0": "et0",
    "evapotranspiration": "et0",
}

GLOBAL_EVENT_CATALOG = [
    {
        "start": "1982-04-01",
        "end": "1983-07-31",
        "title": "Güçlü El Nino (1982-83)",
        "tags": ["temp", "precip", "humidity", "pressure"],
        "source": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php",
    },
    {
        "start": "1988-06-01",
        "end": "1989-06-30",
        "title": "Güçlü La Nina (1988-89)",
        "tags": ["temp", "precip", "humidity", "pressure"],
        "source": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php",
    },
    {
        "start": "1991-06-01",
        "end": "1993-12-31",
        "title": "Pinatubo sonrası aerosol soğuma dönemi",
        "tags": ["temp", "pressure"],
        "source": "https://www.nasa.gov/mission_pages/noaa-n/climate/climate_weather.html",
    },
    {
        "start": "1997-05-01",
        "end": "1998-06-30",
        "title": "Süper El Nino (1997-98)",
        "tags": ["temp", "precip", "humidity", "pressure"],
        "source": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php",
    },
    {
        "start": "1999-07-01",
        "end": "2001-03-31",
        "title": "Çok yıllı La Nina (1999-2001)",
        "tags": ["temp", "precip", "humidity", "pressure"],
        "source": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php",
    },
    {
        "start": "2003-06-01",
        "end": "2003-09-30",
        "title": "Avrupa sıcak hava dalgası (2003 yazı)",
        "tags": ["temp", "humidity", "pressure"],
        "source": "https://public.wmo.int/en/media/news/wmo-statement-state-of-global-climate",
    },
    {
        "start": "2010-06-01",
        "end": "2010-09-30",
        "title": "Rusya sıcak dalgası / Pakistan selleri (2010)",
        "tags": ["temp", "humidity", "precip", "pressure"],
        "source": "https://earthobservatory.nasa.gov/images/44815/russian-fires-and-pakistan-floods-linked",
    },
    {
        "start": "2015-05-01",
        "end": "2016-06-30",
        "title": "Güçlü El Nino (2015-16)",
        "tags": ["temp", "precip", "humidity", "pressure"],
        "source": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php",
    },
    {
        "start": "2018-06-01",
        "end": "2018-08-31",
        "title": "Kuzey yarımküre sıcaklık ve aşırı yağış uçları (2018)",
        "tags": ["temp", "precip", "humidity"],
        "source": "https://wmo.int/media/news/severe-impacts-extreme-weather-and-climate-events-continue-2018",
    },
    {
        "start": "2020-08-01",
        "end": "2022-12-31",
        "title": "Üç yıllık La Nina (2020-2022)",
        "tags": ["temp", "precip", "humidity", "pressure"],
        "source": "https://public.wmo.int/en/media/news/wmo-update-la-nina-expected-continue",
    },
    {
        "start": "2023-06-01",
        "end": "2024-06-30",
        "title": "Güçlü El Nino + küresel sıcak anomali (2023-2024)",
        "tags": ["temp", "precip", "humidity", "pressure"],
        "source": "https://public.wmo.int/en/media/press-release/state-of-global-climate-2023",
    },
]

TURKEY_EVENT_CATALOG = [
    {
        "start": "1993-01-01",
        "end": "1994-12-31",
        "title": "Marmara su stresi/kuraklık dönemi (1993-1994)",
        "region": "Marmara",
        "tags": ["precip", "humidity", "temp", "pressure"],
        "source": "https://www.mgm.gov.tr/veridegerlendirme/iklim-raporlari.aspx",
    },
    {
        "start": "2007-01-01",
        "end": "2008-12-31",
        "title": "Türkiye genelinde kuraklık dönemi (2007-2008)",
        "region": "Türkiye",
        "tags": ["precip", "humidity", "temp"],
        "source": "https://www.mgm.gov.tr/veridegerlendirme/iklim-raporlari.aspx",
    },
    {
        "start": "2009-09-01",
        "end": "2009-09-30",
        "title": "İstanbul-Marmara aşırı yağış/sel olayı (2009)",
        "region": "Marmara",
        "tags": ["precip", "humidity", "pressure"],
        "source": "https://www.afad.gov.tr/afetler",
    },
    {
        "start": "2010-08-01",
        "end": "2010-10-31",
        "title": "Marmara kuvvetli yağış ve sel dönemi (2010)",
        "region": "Marmara",
        "tags": ["precip", "humidity", "pressure"],
        "source": "https://www.afad.gov.tr/afetler",
    },
    {
        "start": "2014-01-01",
        "end": "2014-12-31",
        "title": "Türkiye kurak ve sıcak yıl sinyali (2014)",
        "region": "Türkiye",
        "tags": ["precip", "humidity", "temp"],
        "source": "https://www.mgm.gov.tr/veridegerlendirme/iklim-raporlari.aspx",
    },
    {
        "start": "2017-07-01",
        "end": "2017-07-31",
        "title": "Marmara'da şiddetli dolu/konvektif yağış olayı (2017)",
        "region": "Marmara",
        "tags": ["precip", "humidity", "pressure", "temp"],
        "source": "https://www.afad.gov.tr/afetler",
    },
    {
        "start": "2021-07-01",
        "end": "2021-08-31",
        "title": "Akdeniz sıcak dalga/yangın ve Batı Karadeniz sel dönemi (2021)",
        "region": "Akdeniz-Karadeniz",
        "tags": ["temp", "precip", "humidity", "pressure"],
        "source": "https://wmo.int/media/news/deadly-floods-and-wildfires-heatwaves-and-drought",
    },
]


@dataclass
class FrequencyPlan:
    freq: str
    season_len: int
    label: str
    monthly_coverage: float


@dataclass
class CVPlan:
    holdout: int
    splits: int
    min_train: int


@dataclass
class RegimePack:
    success: bool
    k_regimes: int
    transition: np.ndarray
    means: np.ndarray
    vars_: np.ndarray
    last_probs: np.ndarray
    smoothed_probs: np.ndarray
    aic: float
    note: str


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quant regime climate forecasting pipeline")
    p.add_argument(
        "--observations",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/forecast_package/observations_with_graph.parquet"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/quant_climate_package"),
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

    p.add_argument("--winsor-lower", type=float, default=0.005)
    p.add_argument("--winsor-upper", type=float, default=0.995)

    p.add_argument("--holdout-steps", type=int, default=12)
    p.add_argument("--backtest-splits", type=int, default=3)
    p.add_argument("--min-train-steps", type=int, default=36)

    p.add_argument("--ewma-lambda", type=float, default=0.94)
    p.add_argument("--vol-model", type=str, default="auto", choices=["auto", "egarch", "ewma"])
    p.add_argument("--egarch-p", type=int, default=1)
    p.add_argument("--egarch-o", type=int, default=1)
    p.add_argument("--egarch-q", type=int, default=1)
    p.add_argument("--egarch-dist", type=str, default="t", choices=["normal", "t"])
    p.add_argument("--interval-alpha", type=float, default=0.10)

    p.add_argument("--regime-k", type=int, default=2)
    p.add_argument("--regime-maxiter", type=int, default=200)

    p.add_argument("--anomaly-z", type=float, default=2.5)
    p.add_argument("--anomaly-top", type=int, default=15)
    p.add_argument(
        "--climate-scenario",
        type=str,
        default="ssp245",
        help="İklim senaryosu: none, ssp126, ssp245, ssp370, ssp585",
    )
    p.add_argument(
        "--climate-baseline-year",
        type=float,
        default=float("nan"),
        help="Senaryo düzeltmesi için baz yıl; NaN ise seri son gözlem yılı otomatik alınır.",
    )
    p.add_argument(
        "--climate-temp-rate",
        type=float,
        default=float("nan"),
        help="Sıcaklık trend override (C/yıl). NaN ise senaryo varsayılanı.",
    )
    p.add_argument(
        "--humidity-per-temp-c",
        type=float,
        default=-2.0,
        help="Nem düzeltme katsayısı (yüzde puan / C).",
    )
    p.add_argument(
        "--climate-adjustment-method",
        type=str,
        default="pathway",
        help="Düzeltme metodu: pathway (IPCC AR6 SSP eğrisi) veya linear.",
    )
    p.add_argument(
        "--disable-climate-adjustment",
        action="store_true",
        help="Senaryo katsayısı düzeltmesini kapat.",
    )

    return p.parse_args()


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def normalize_token(text: Any) -> str:
    s = str(text).strip().lower().translate(_TR_CHARMAP)
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


def is_precip(variable: str) -> bool:
    return canonical_variable_name(variable) == "precip"


def is_humidity(variable: str) -> bool:
    return canonical_variable_name(variable) == "humidity"


def is_pressure(variable: str) -> bool:
    return canonical_variable_name(variable) == "pressure"


def is_et0(variable: str) -> bool:
    return canonical_variable_name(variable) == "et0"


def infer_unit(variable: str) -> str:
    return UNIT_MAP.get(canonical_variable_name(variable), "unknown")


def variable_tr(variable: str) -> str:
    return VAR_TR_MAP.get(canonical_variable_name(variable), str(variable))


def frequency_tr(freq: str) -> str:
    f = str(freq).upper()
    if f.startswith("Y"):
        return "yıllık"
    if f.startswith("W"):
        return "haftalık"
    if f.startswith("D"):
        return "günlük"
    return "aylık"


def is_yearly_freq(freq: str) -> bool:
    return str(freq).upper().startswith("Y")


def is_monthly_freq(freq: str) -> bool:
    f = str(freq).upper()
    return f == "MS" or f.startswith("M")


def is_weekly_freq(freq: str) -> bool:
    return str(freq).upper().startswith("W")


def is_daily_freq(freq: str) -> bool:
    return str(freq).upper().startswith("D")


def apply_bounds(arr: np.ndarray, variable: str) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if is_humidity(variable):
        return np.clip(x, 0, 100)
    if is_precip(variable) or is_pressure(variable) or is_et0(variable):
        return np.clip(x, 0, None)
    return x


def ensure_dirs(paths: list[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Data ingestion
# -----------------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
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
        raise SystemExit("Cannot detect time/value columns")

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
        out["qc_flag"] = raw[qc_col].astype(str)
    else:
        out["qc_flag"] = args.qc_ok_value

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["variable"] = out["variable"].astype(str).map(canonical_variable_name)

    out = out.dropna(subset=["timestamp", "value", "variable"]).sort_values("timestamp").reset_index(drop=True)
    if out.empty:
        raise SystemExit("No usable rows after parsing")
    return out, input_kind


def select_variables(obs: pd.DataFrame, variables_arg: str) -> list[str]:
    available = sorted(obs["variable"].dropna().unique().tolist())
    if not variables_arg or variables_arg.strip() in {"*", "all", "ALL"}:
        return available
    req = [canonical_variable_name(v.strip()) for v in variables_arg.split(",") if v.strip()]
    req = [v for v in req if v in available]
    return sorted(set(req))


# -----------------------------------------------------------------------------
# Frequency + aggregation
# -----------------------------------------------------------------------------

def choose_frequency_plan(obs: pd.DataFrame, variable: str, ok_value: str) -> FrequencyPlan:
    sub = obs[obs["variable"] == variable].copy()
    if sub.empty:
        return FrequencyPlan("MS", 12, "monthly", 0.0)

    ok_mask = sub["qc_flag"].astype(str).str.lower().eq(str(ok_value).lower())
    if ok_mask.any():
        sub = sub[ok_mask]

    sub = sub.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if sub.empty:
        return FrequencyPlan("MS", 12, "monthly", 0.0)

    raw = sub.groupby("timestamp")["value"].mean()

    m_cov = raw.resample("MS").count()
    m_obs = int((m_cov > 0).sum())
    m_total = max(1, len(m_cov))
    coverage = m_obs / m_total

    y_cov = raw.resample("YS").count()
    y_obs = int((y_cov > 0).sum())

    if coverage < 0.45 and y_obs >= 15:
        return FrequencyPlan("YS", 1, "yearly", float(coverage))
    return FrequencyPlan("MS", 12, "monthly", float(coverage))


def aggregate_series(obs: pd.DataFrame, variable: str, plan: FrequencyPlan, ok_value: str) -> pd.Series:
    sub = obs[obs["variable"] == variable].copy()
    if sub.empty:
        return pd.Series(dtype=float)

    ok_mask = sub["qc_flag"].astype(str).str.lower().eq(str(ok_value).lower())
    if ok_mask.any():
        sub = sub[ok_mask]

    sub = sub.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if sub.empty:
        return pd.Series(dtype=float)

    raw = sub.groupby("timestamp")["value"].mean()

    if plan.freq == "YS":
        if is_precip(variable):
            s = raw.resample("YS").sum(min_count=1).fillna(0.0)
        else:
            s = raw.resample("YS").mean().interpolate("time").ffill().bfill()
        return s.astype(float)

    if is_precip(variable):
        s = raw.resample("MS").sum(min_count=1).fillna(0.0)
    else:
        s = raw.resample("MS").mean().interpolate("time").ffill().bfill()
    return s.astype(float)


def winsorize(s: pd.Series, ql: float, qu: float) -> pd.Series:
    if s.empty:
        return s
    lo = float(np.clip(ql, 0.0, 0.49))
    hi = float(np.clip(qu, 0.51, 1.0))
    x = s.astype(float).copy()
    x = x.clip(lower=float(x.quantile(lo)), upper=float(x.quantile(hi)))
    x = x.interpolate("time").ffill().bfill()
    return x


# -----------------------------------------------------------------------------
# Features and base mean model
# -----------------------------------------------------------------------------

def make_design(ds: pd.Series, freq: str, anchor: pd.Timestamp) -> np.ndarray:
    d = pd.to_datetime(ds)
    if len(d) == 0:
        if is_monthly_freq(freq):
            return np.empty((0, 6), dtype=float)
        if is_weekly_freq(freq):
            return np.empty((0, 6), dtype=float)
        if is_daily_freq(freq):
            return np.empty((0, 8), dtype=float)
        return np.empty((0, 4), dtype=float)

    t = (d - anchor).dt.days.astype(float).values
    scale = max(1.0, float(np.max(np.abs(t))))
    t = t / scale

    cols = [t, t**2]

    if is_monthly_freq(freq):
        m = d.dt.month.astype(float).values
        q = d.dt.quarter.astype(float).values
        cols += [
            np.sin(2 * np.pi * m / 12.0),
            np.cos(2 * np.pi * m / 12.0),
            np.sin(2 * np.pi * q / 4.0),
            np.cos(2 * np.pi * q / 4.0),
        ]
    elif is_weekly_freq(freq):
        w = d.dt.isocalendar().week.astype(float).values
        m = d.dt.month.astype(float).values
        cols += [
            np.sin(2 * np.pi * w / 52.0),
            np.cos(2 * np.pi * w / 52.0),
            np.sin(2 * np.pi * m / 12.0),
            np.cos(2 * np.pi * m / 12.0),
        ]
    elif is_daily_freq(freq):
        doy = d.dt.dayofyear.astype(float).values
        dow = d.dt.dayofweek.astype(float).values
        m = d.dt.month.astype(float).values
        cols += [
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            np.sin(2 * np.pi * m / 12.0),
            np.cos(2 * np.pi * m / 12.0),
        ]
    else:
        y = (d.dt.year - anchor.year).astype(float).values
        cols += [
            np.sin(2 * np.pi * y / 11.0),
            np.cos(2 * np.pi * y / 11.0),
        ]

    return np.vstack(cols).T


def _recency_sample_weights(n: int, freq: str, min_w: float = 0.15) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    if is_monthly_freq(freq):
        half_life = float(np.clip(n * 0.35, 18.0, 120.0))
    elif is_weekly_freq(freq):
        half_life = float(np.clip(n * 0.30, 52.0, 260.0))
    elif is_daily_freq(freq):
        half_life = float(np.clip(n * 0.22, 180.0, 730.0))
    else:
        half_life = float(np.clip(n * 0.45, 4.0, 14.0))
    age = (n - 1 - np.arange(n, dtype=float))
    w = np.power(0.5, age / max(half_life, 1.0))
    return np.clip(w, float(min_w), 1.0)


def _base_alpha_grid(freq: str) -> list[float]:
    if is_monthly_freq(freq):
        return [0.05, 0.10, 0.30, 1.0, 2.0, 3.0]
    if is_weekly_freq(freq):
        return [0.10, 0.30, 1.0, 3.0, 8.0]
    if is_daily_freq(freq):
        return [0.30, 1.0, 3.0, 10.0, 20.0]
    return [0.03, 0.10, 0.30, 1.0, 2.0, 3.0]


def fit_base_ridge(ds: pd.Series, y: np.ndarray, freq: str, alpha: float = 1.0) -> tuple[Ridge, pd.Timestamp, float]:
    anchor = pd.Timestamp(pd.to_datetime(ds).min())
    X = make_design(ds, freq=freq, anchor=anchor)
    yv = np.asarray(y, dtype=float)
    n = len(yv)
    sw_recency = _recency_sample_weights(n, freq=freq, min_w=0.18)
    sw_uniform = np.ones(n, dtype=float)

    if is_monthly_freq(freq):
        min_tune_n = 36
        val_min = 8
    elif is_weekly_freq(freq):
        min_tune_n = 80
        val_min = 10
    elif is_daily_freq(freq):
        min_tune_n = 240
        val_min = 14
    else:
        min_tune_n = 16
        val_min = 4

    alpha_grid = _base_alpha_grid(freq=freq)
    alpha_best = float(alpha)
    final_sw = sw_uniform

    if n >= min_tune_n and len(np.unique(np.round(yv, 6))) >= 8:
        val_len = int(np.clip(round(n * 0.20), val_min, max(val_min, n // 3)))
        split = n - val_len
        if split >= max(8, val_min):
            X_tr, y_tr = X[:split], yv[:split]
            X_va, y_va = X[split:], yv[split:]
            best_score = np.inf
            for sw_mode in (sw_uniform, sw_recency):
                w_tr = sw_mode[:split]
                w_va = sw_mode[split:]
                for a in alpha_grid:
                    try:
                        m = Ridge(alpha=float(a), random_state=42)
                        m.fit(X_tr, y_tr, sample_weight=w_tr)
                        p = m.predict(X_va).astype(float)
                        score = float(np.sqrt(np.average((y_va - p) ** 2, weights=w_va)))
                        if np.isfinite(score) and score < best_score:
                            best_score = score
                            alpha_best = float(a)
                            final_sw = sw_mode
                    except Exception:
                        continue

    model = Ridge(alpha=float(alpha_best), random_state=42)
    model.fit(X, yv, sample_weight=final_sw)
    return model, anchor, float(alpha_best)


def predict_base(model: Ridge, anchor: pd.Timestamp, ds: pd.Series, freq: str) -> np.ndarray:
    if len(pd.to_datetime(ds)) == 0:
        return np.array([], dtype=float)
    X = make_design(ds, freq=freq, anchor=anchor)
    return model.predict(X).astype(float)


def seasonal_terms(ts: pd.Timestamp, freq: str) -> np.ndarray:
    t = pd.Timestamp(ts)
    if is_monthly_freq(freq):
        m = float(t.month)
        q = float(((t.month - 1) // 3) + 1)
        return np.array(
            [
                np.sin(2 * np.pi * m / 12.0),
                np.cos(2 * np.pi * m / 12.0),
                np.sin(2 * np.pi * q / 4.0),
                np.cos(2 * np.pi * q / 4.0),
            ],
            dtype=float,
        )
    if is_weekly_freq(freq):
        w = float(pd.Timestamp(t).isocalendar().week)
        m = float(t.month)
        return np.array(
            [
                np.sin(2 * np.pi * w / 52.0),
                np.cos(2 * np.pi * w / 52.0),
                np.sin(2 * np.pi * m / 12.0),
                np.cos(2 * np.pi * m / 12.0),
            ],
            dtype=float,
        )
    if is_daily_freq(freq):
        doy = float(t.dayofyear)
        dow = float(t.dayofweek)
        m = float(t.month)
        return np.array(
            [
                np.sin(2 * np.pi * doy / 365.25),
                np.cos(2 * np.pi * doy / 365.25),
                np.sin(2 * np.pi * dow / 7.0),
                np.cos(2 * np.pi * dow / 7.0),
                np.sin(2 * np.pi * m / 12.0),
                np.cos(2 * np.pi * m / 12.0),
            ],
            dtype=float,
        )
    y = float(t.year)
    return np.array([np.sin(2 * np.pi * y / 11.0), np.cos(2 * np.pi * y / 11.0)], dtype=float)


def choose_ar_lags(n: int, freq: str) -> int:
    if is_monthly_freq(freq):
        return int(np.clip(n // 6, 4, 18))
    if is_weekly_freq(freq):
        return int(np.clip(n // 10, 8, 52))
    if is_daily_freq(freq):
        return int(np.clip(n // 25, 14, 90))
    return int(np.clip(n // 8, 2, 6))


def fit_ar_residual_model(ds: pd.Series, resid: np.ndarray, freq: str) -> tuple[Ridge | None, int, np.ndarray, float]:
    d = pd.to_datetime(ds).reset_index(drop=True)
    r = np.asarray(resid, dtype=float)
    n = len(r)
    if n < 16:
        return None, 0, np.zeros(n, dtype=float), 0.0

    lags = choose_ar_lags(n, freq)
    if n <= lags + 8:
        return None, lags, np.zeros(n, dtype=float), 0.0

    X = []
    yt = []
    idx = []
    for t in range(lags, n):
        lag_block = r[t - lags : t][::-1]
        x = np.concatenate([lag_block, seasonal_terms(d.iloc[t], freq=freq)])
        X.append(x)
        yt.append(r[t])
        idx.append(t)

    if not X:
        return None, lags, np.zeros(n, dtype=float), 0.0

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(yt, dtype=float)
    idx_arr = np.asarray(idx, dtype=int)
    row_w_recency = _recency_sample_weights(n, freq=freq, min_w=0.20)[idx_arr]
    row_w_uniform = np.ones(len(idx_arr), dtype=float)

    split = int(np.clip(round(len(y_arr) * 0.82), 12, max(12, len(y_arr) - 6)))
    X_tr, y_tr = X_arr[:split], y_arr[:split]
    X_va, y_va = X_arr[split:], y_arr[split:]

    alpha_grid = [0.5, 1.0, 2.0, 4.0]
    alpha_best = 2.0
    best_rmse = np.inf
    final_w = row_w_uniform
    if len(X_va):
        for row_w in (row_w_uniform, row_w_recency):
            w_tr = row_w[:split]
            for a in alpha_grid:
                try:
                    m = Ridge(alpha=float(a), random_state=42)
                    m.fit(X_tr, y_tr, sample_weight=w_tr)
                    p = m.predict(X_va).astype(float)
                    rmse = float(np.sqrt(np.mean((y_va - p) ** 2)))
                    if np.isfinite(rmse) and rmse < best_rmse:
                        best_rmse = rmse
                        alpha_best = float(a)
                        final_w = row_w
                except Exception:
                    continue

    model = Ridge(alpha=float(alpha_best), random_state=42)
    model.fit(X_arr, y_arr, sample_weight=final_w)

    hist_pred = np.zeros(n, dtype=float)
    for t in range(lags, n):
        lag_block = r[t - lags : t][::-1]
        x = np.concatenate([lag_block, seasonal_terms(d.iloc[t], freq=freq)])
        hist_pred[t] = float(model.predict(x.reshape(1, -1))[0])

    return model, lags, hist_pred, float(alpha_best)


def predict_ar_future(
    model: Ridge | None,
    lags: int,
    resid_hist: np.ndarray,
    future_ds: pd.Series,
    freq: str,
) -> np.ndarray:
    fds = pd.to_datetime(future_ds).reset_index(drop=True)
    if model is None or lags <= 0 or len(fds) == 0:
        return np.zeros(len(fds), dtype=float)

    state = list(np.asarray(resid_hist, dtype=float))
    if len(state) < lags:
        return np.zeros(len(fds), dtype=float)

    out = np.zeros(len(fds), dtype=float)
    for i, ts in enumerate(fds):
        lag_block = np.asarray(state[-lags:][::-1], dtype=float)
        x = np.concatenate([lag_block, seasonal_terms(ts, freq=freq)])
        val = float(model.predict(x.reshape(1, -1))[0])
        out[i] = val
        state.append(val)
    return out


def choose_boost_lags(n: int, freq: str) -> int:
    if is_monthly_freq(freq):
        return int(np.clip(n // 10, 6, 18))
    if is_weekly_freq(freq):
        return int(np.clip(n // 16, 8, 40))
    if is_daily_freq(freq):
        return int(np.clip(n // 30, 14, 70))
    return int(np.clip(n // 12, 3, 6))


def fit_boost_residual_model(
    ds: pd.Series, resid: np.ndarray, freq: str
) -> tuple[GradientBoostingRegressor | None, int, np.ndarray, float, float, str]:
    d = pd.to_datetime(ds).reset_index(drop=True)
    r = np.asarray(resid, dtype=float)
    n = len(r)
    if is_monthly_freq(freq):
        min_len = 48
    elif is_weekly_freq(freq):
        min_len = 120
    elif is_daily_freq(freq):
        min_len = 540
    else:
        min_len = 16
    if n < min_len:
        return None, 0, np.zeros(n, dtype=float), 0.0, 0.0, "none"

    lags = choose_boost_lags(n, freq)
    if n <= lags + 10:
        return None, lags, np.zeros(n, dtype=float), 0.0, 0.0, "none"

    X = []
    yv = []
    idx = []
    for t in range(lags, n):
        lag_block = r[t - lags : t][::-1]
        x = np.concatenate([lag_block, seasonal_terms(d.iloc[t], freq=freq)])
        X.append(x)
        yv.append(r[t])
        idx.append(t)

    X = np.asarray(X, dtype=float)
    yv = np.asarray(yv, dtype=float)
    idx_arr = np.asarray(idx, dtype=int)
    if len(yv) < 18:
        return None, lags, np.zeros(n, dtype=float), 0.0, 0.0, "none"

    split = int(round(len(yv) * 0.8))
    split = int(np.clip(split, 12, len(yv) - 6))
    X_tr, y_tr = X[:split], yv[:split]
    X_va, y_va = X[split:], yv[split:]

    row_w = _recency_sample_weights(n, freq=freq, min_w=0.18)[idx_arr]
    w_tr = row_w[:split]
    w_va = row_w[split:] if len(y_va) else np.array([], dtype=float)

    base_n_estimators = 600 if is_monthly_freq(freq) else (400 if is_weekly_freq(freq) else (350 if is_daily_freq(freq) else 300))
    base_lr = 0.025 if is_monthly_freq(freq) else (0.03 if is_weekly_freq(freq) else (0.03 if is_daily_freq(freq) else 0.04))
    base_depth = 3 if is_monthly_freq(freq) else 2
    cfgs = [
        {"name": "gbr_huber_base", "n_estimators": base_n_estimators, "learning_rate": base_lr, "max_depth": base_depth, "subsample": 0.80},
        {"name": "gbr_huber_smooth", "n_estimators": int(base_n_estimators * 1.15), "learning_rate": base_lr * 0.75, "max_depth": max(2, base_depth), "subsample": 0.85},
        {"name": "gbr_huber_fast", "n_estimators": int(base_n_estimators * 0.75), "learning_rate": base_lr * 1.40, "max_depth": max(2, base_depth - 1), "subsample": 0.75},
    ]
    mae_0 = float(np.average(np.abs(y_va), weights=w_va)) if len(y_va) else 0.0
    best_model = None
    best_gain = -np.inf
    best_cfg = "none"
    if len(y_va):
        for cfg in cfgs:
            try:
                m = GradientBoostingRegressor(
                    loss="huber",
                    n_estimators=int(cfg["n_estimators"]),
                    learning_rate=float(cfg["learning_rate"]),
                    max_depth=int(cfg["max_depth"]),
                    min_samples_leaf=3,
                    subsample=float(cfg["subsample"]),
                    random_state=42,
                )
                m.fit(X_tr, y_tr, sample_weight=w_tr)
                pred_va = m.predict(X_va).astype(float)
                mae_m = float(np.average(np.abs(y_va - pred_va), weights=w_va))
                gain = 1.0 - (mae_m / max(mae_0, 1e-9))
                if np.isfinite(gain) and gain > best_gain:
                    best_gain = float(gain)
                    best_model = m
                    best_cfg = str(cfg["name"])
            except Exception:
                continue
    else:
        best_gain = 0.0

    # Guardrail: if booster does not improve out-of-sample MAE, disable it.
    if best_model is None or (not np.isfinite(best_gain)) or (best_gain <= 0.05):
        return None, lags, np.zeros(n, dtype=float), float(max(best_gain, 0.0)), 0.0, "none"

    weight = float(np.clip(0.20 + best_gain * 1.8, 0.20, 0.70))

    hist_pred = np.zeros(n, dtype=float)
    hist_pred[lags:] = best_model.predict(X).astype(float)
    return best_model, lags, hist_pred, float(best_gain), float(weight), best_cfg


def predict_boost_future(
    model: GradientBoostingRegressor | None,
    lags: int,
    resid_hist: np.ndarray,
    future_ds: pd.Series,
    freq: str,
) -> np.ndarray:
    fds = pd.to_datetime(future_ds).reset_index(drop=True)
    if model is None or lags <= 0 or len(fds) == 0:
        return np.zeros(len(fds), dtype=float)

    state = list(np.asarray(resid_hist, dtype=float))
    if len(state) < lags:
        return np.zeros(len(fds), dtype=float)

    out = np.zeros(len(fds), dtype=float)
    for i, ts in enumerate(fds):
        lag_block = np.asarray(state[-lags:][::-1], dtype=float)
        x = np.concatenate([lag_block, seasonal_terms(ts, freq=freq)])
        val = float(model.predict(x.reshape(1, -1))[0])
        out[i] = val
        state.append(val)
    return out


def calibrate_component_blend(
    resid: np.ndarray,
    reg_hist_mu: np.ndarray,
    ar_hist_mu: np.ndarray,
    boost_hist_mu: np.ndarray,
    freq: str,
    default_weights: np.ndarray,
) -> tuple[np.ndarray, float, bool, int]:
    r = np.asarray(resid, dtype=float)
    X = np.column_stack(
        [
            np.asarray(reg_hist_mu, dtype=float),
            np.asarray(ar_hist_mu, dtype=float),
            np.asarray(boost_hist_mu, dtype=float),
        ]
    )
    n = len(r)
    if n < 18 or X.shape[0] != n:
        return np.asarray(default_weights, dtype=float), 0.0, False, 0

    if is_monthly_freq(freq):
        eval_n = int(np.clip(n // 3, 18, 120))
    elif is_weekly_freq(freq):
        eval_n = int(np.clip(n // 4, 20, 180))
    elif is_daily_freq(freq):
        eval_n = int(np.clip(n // 5, 24, 360))
    else:
        eval_n = int(np.clip(n // 2, 6, 24))
    start = max(0, n - eval_n)

    y_cal = r[start:]
    X_cal = X[start:]
    w = _recency_sample_weights(len(y_cal), freq=freq, min_w=0.25)
    sqrt_w = np.sqrt(np.asarray(w, dtype=float))
    Xw = X_cal * sqrt_w.reshape(-1, 1)
    yw = y_cal * sqrt_w

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0].astype(float)
    except Exception:
        return np.asarray(default_weights, dtype=float), 0.0, False, int(len(y_cal))

    beta = np.where(np.isfinite(beta), beta, 0.0)
    beta[0] = float(np.clip(beta[0], 0.70, 1.30))
    beta[1] = float(np.clip(beta[1], 0.00, 1.20))
    beta[2] = float(np.clip(beta[2], 0.00, 0.85))

    pred_def = X_cal @ np.asarray(default_weights, dtype=float)
    pred_new = X_cal @ beta
    mae_def = float(np.average(np.abs(y_cal - pred_def), weights=w))
    mae_new = float(np.average(np.abs(y_cal - pred_new), weights=w))
    gain = 1.0 - (mae_new / max(mae_def, 1e-9))

    if (not np.isfinite(gain)) or gain <= 0.15:
        return np.asarray(default_weights, dtype=float), float(max(gain, 0.0)), False, int(len(y_cal))
    return beta, float(gain), True, int(len(y_cal))


def dynamic_analog_weight(resid_core: np.ndarray, freq: str, base_weight: float) -> tuple[float, float]:
    r = np.asarray(resid_core, dtype=float)
    n = len(r)
    if n < 10:
        return float(base_weight), 0.0

    def _corr_at_lag(arr: np.ndarray, lag: int) -> float:
        if lag <= 0 or len(arr) <= lag + 4:
            return float("nan")
        a = arr[lag:]
        b = arr[:-lag]
        sa = float(np.std(a))
        sb = float(np.std(b))
        if sa <= 1e-9 or sb <= 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    cands = []
    lag1 = _corr_at_lag(r, 1)
    if np.isfinite(lag1):
        cands.append(lag1)
    if is_monthly_freq(freq):
        lag_season = _corr_at_lag(r, 12)
    elif is_weekly_freq(freq):
        lag_season = _corr_at_lag(r, 52)
    elif is_daily_freq(freq):
        lag_season = _corr_at_lag(r, 365)
    else:
        lag_season = _corr_at_lag(r, 2)
    if np.isfinite(lag_season):
        cands.append(lag_season)

    strength = float(np.clip(max(cands) if cands else 0.0, 0.0, 0.95))
    if is_monthly_freq(freq):
        w = base_weight * (0.58 + 0.90 * strength)
        lo, hi = 0.08, 0.32
    elif is_weekly_freq(freq):
        w = base_weight * (0.62 + 0.85 * strength)
        lo, hi = 0.06, 0.27
    elif is_daily_freq(freq):
        w = base_weight * (0.66 + 0.80 * strength)
        lo, hi = 0.05, 0.22
    else:
        w = base_weight * (0.64 + 0.82 * strength)
        lo, hi = 0.05, 0.22
    return float(np.clip(w, lo, hi)), strength


def analog_pattern_projection(resid: np.ndarray, steps: int, freq: str) -> tuple[np.ndarray, bool, int, int]:
    r = np.asarray(resid, dtype=float)
    n = len(r)
    if steps <= 0:
        return np.array([]), False, 0, 0

    if is_monthly_freq(freq):
        window = int(np.clip(n // 8, 6, 18))
    elif is_weekly_freq(freq):
        window = int(np.clip(n // 18, 12, 64))
    elif is_daily_freq(freq):
        window = int(np.clip(n // 30, 21, 120))
    else:
        window = int(np.clip(n // 10, 3, 6))

    if n <= window + 8:
        return np.zeros(steps, dtype=float), False, window, 0

    target = r[-window:]
    cands = []
    for end in range(window, n - 1):
        seg = r[end - window : end]
        dist = float(np.sqrt(np.mean((seg - target) ** 2)))
        cands.append((dist, end))

    if not cands:
        return np.zeros(steps, dtype=float), False, window, 0

    cands.sort(key=lambda x: x[0])
    k = int(np.clip(len(cands) // 10, 3, 10))
    top = cands[:k]

    dvals = np.array([d for d, _ in top], dtype=float)
    w = 1.0 / (dvals + 1e-6)
    w = w / max(float(np.sum(w)), 1e-9)

    fc = np.zeros(steps, dtype=float)
    for h in range(1, steps + 1):
        vals = []
        ww = []
        for wi, (_, end) in zip(w, top):
            idx = end + h - 1
            if idx < n:
                vals.append(float(r[idx]))
                ww.append(float(wi))
        if vals:
            ww_arr = np.asarray(ww, dtype=float)
            ww_arr = ww_arr / max(float(np.sum(ww_arr)), 1e-9)
            fc[h - 1] = float(np.dot(np.asarray(vals, dtype=float), ww_arr))
        elif h > 1:
            fc[h - 1] = fc[h - 2] * 0.96

    if is_monthly_freq(freq):
        decay_scale = 60.0
    elif is_weekly_freq(freq):
        decay_scale = 220.0
    elif is_daily_freq(freq):
        decay_scale = 900.0
    else:
        decay_scale = 18.0
    decay = np.exp(-np.arange(steps, dtype=float) / decay_scale)
    fc = fc * decay
    return fc, True, window, k


def local_trend_projection(y: np.ndarray, steps: int, freq: str) -> tuple[np.ndarray, float]:
    arr = np.asarray(y, dtype=float)
    n = len(arr)
    if steps <= 0 or n < 8:
        return np.zeros(max(0, steps), dtype=float), 0.0

    if is_monthly_freq(freq):
        if n < 120:
            return np.zeros(steps, dtype=float), 0.0
        win = int(np.clip(n // 3, 12, 36))
        decay_scale = 72.0
        slope_mul = 0.18
        drift_mul = 0.65
    elif is_weekly_freq(freq):
        if n < 200:
            return np.zeros(steps, dtype=float), 0.0
        win = int(np.clip(n // 4, 26, 104))
        decay_scale = 280.0
        slope_mul = 0.20
        drift_mul = 0.70
    elif is_daily_freq(freq):
        if n < 720:
            return np.zeros(steps, dtype=float), 0.0
        win = int(np.clip(n // 5, 120, 720))
        decay_scale = 1200.0
        slope_mul = 0.16
        drift_mul = 0.62
    else:
        if n < 30:
            return np.zeros(steps, dtype=float), 0.0
        win = int(np.clip(n // 3, 6, 12))
        decay_scale = 24.0
        slope_mul = 0.25
        drift_mul = 0.75

    seg = arr[-win:]
    x = np.arange(len(seg), dtype=float)
    try:
        slope = float(np.polyfit(x, seg, 1)[0])
    except Exception:
        slope = 0.0

    d = np.diff(seg)
    slope_cap = max(float(np.nanstd(d)) * slope_mul, 1e-6)
    slope = float(np.clip(slope, -slope_cap, slope_cap))

    h = np.arange(1, steps + 1, dtype=float)
    drift = slope * h * np.exp(-h / decay_scale) * drift_mul
    return drift, slope


# -----------------------------------------------------------------------------
# Regime + volatility
# -----------------------------------------------------------------------------

def safe_transition_from_params(params: np.ndarray, k: int) -> np.ndarray:
    # params include transition logits; robust fallback if parsing fails
    P = np.full((k, k), 1.0 / k, dtype=float)
    if k == 2 and len(params) >= 2:
        # Approximation for 2-state MarkovRegression logits:
        # p[0->0], p[1->0] often represented with transformed params.
        try:
            p00 = 1.0 / (1.0 + np.exp(-float(params[0])))
            p10 = 1.0 / (1.0 + np.exp(-float(params[1])))
            P = np.array([[p00, 1 - p00], [p10, 1 - p10]], dtype=float)
        except Exception:
            pass
    return P


def fit_markov_residuals(resid: np.ndarray, k_regimes: int, maxiter: int) -> RegimePack:
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]

    if len(r) < 20:
        return RegimePack(
            success=False,
            k_regimes=1,
            transition=np.array([[1.0]]),
            means=np.array([0.0]),
            vars_=np.array([float(np.var(r)) if len(r) else 1.0]),
            last_probs=np.array([1.0]),
            smoothed_probs=np.ones((len(r), 1), dtype=float),
            aic=np.nan,
            note="short_series",
        )

    k = max(2, int(k_regimes))

    try:
        model = MarkovRegression(r, k_regimes=k, trend="c", switching_variance=True)
        res = model.fit(disp=False, maxiter=int(maxiter))

        smoothed = np.asarray(res.smoothed_marginal_probabilities)
        if smoothed.ndim == 1:
            smoothed = smoothed.reshape(-1, 1)

        if smoothed.shape[1] != k:
            k = smoothed.shape[1]

        # regime means and variances from expected assignment
        means = []
        vars_ = []
        for j in range(k):
            w = smoothed[:, j]
            wsum = float(np.sum(w))
            if wsum <= 1e-9:
                means.append(float(np.mean(r)))
                vars_.append(float(np.var(r)))
            else:
                mu = float(np.sum(w * r) / wsum)
                vv = float(np.sum(w * (r - mu) ** 2) / wsum)
                means.append(mu)
                vars_.append(max(vv, 1e-9))

        P = safe_transition_from_params(np.asarray(res.params), k)
        p_last = smoothed[-1].astype(float)
        s = float(np.sum(p_last))
        if s > 0:
            p_last = p_last / s
        else:
            p_last = np.repeat(1.0 / k, k)

        return RegimePack(
            success=True,
            k_regimes=k,
            transition=P,
            means=np.asarray(means, dtype=float),
            vars_=np.asarray(vars_, dtype=float),
            last_probs=p_last,
            smoothed_probs=smoothed,
            aic=float(res.aic),
            note="ok",
        )
    except Exception as exc:
        v = float(np.var(r)) if len(r) else 1.0
        return RegimePack(
            success=False,
            k_regimes=1,
            transition=np.array([[1.0]]),
            means=np.array([0.0]),
            vars_=np.array([max(v, 1e-9)]),
            last_probs=np.array([1.0]),
            smoothed_probs=np.ones((len(r), 1), dtype=float),
            aic=np.nan,
            note=f"markov_failed:{type(exc).__name__}",
        )


def regime_expected_path(pack: RegimePack, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if steps <= 0:
        return np.array([]), np.array([]), np.empty((0, pack.k_regimes))

    p = np.asarray(pack.last_probs, dtype=float)
    P = np.asarray(pack.transition, dtype=float)
    mu = np.asarray(pack.means, dtype=float)
    vv = np.asarray(pack.vars_, dtype=float)

    probs = []
    exp_mu = []
    exp_var = []

    for _ in range(steps):
        p = p @ P
        s = float(np.sum(p))
        if s > 0:
            p = p / s
        probs.append(p.copy())

        m = float(np.sum(p * mu))
        v = float(np.sum(p * vv))
        exp_mu.append(m)
        exp_var.append(max(v, 1e-9))

    return np.asarray(exp_mu), np.asarray(exp_var), np.asarray(probs)


def ewma_variance(resid: np.ndarray, lam: float) -> np.ndarray:
    r = np.asarray(resid, dtype=float)
    r = np.where(np.isfinite(r), r, 0.0)
    if len(r) == 0:
        return np.array([])

    lam = float(np.clip(lam, 0.80, 0.999))
    var = np.zeros(len(r), dtype=float)
    var[0] = max(float(np.var(r)), 1e-9)
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1.0 - lam) * (r[t - 1] ** 2)
        var[t] = max(var[t], 1e-9)
    return var


def ewma_forecast_var(last_var: float, unc_var: float, steps: int, lam: float) -> np.ndarray:
    if steps <= 0:
        return np.array([])
    lam = float(np.clip(lam, 0.80, 0.999))
    out = np.zeros(steps, dtype=float)
    v0 = max(float(last_var), 1e-9)
    u = max(float(unc_var), 1e-9)

    for h in range(1, steps + 1):
        out[h - 1] = u + (lam**h) * (v0 - u)
        out[h - 1] = max(out[h - 1], 1e-9)
    return out


def egarch_variance(
    resid: np.ndarray,
    steps: int,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "t",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (hist_var, fc_var, note). Raises on failure."""
    if not ARCH_AVAILABLE:
        raise RuntimeError("arch_not_available")

    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 40:
        raise RuntimeError("egarch_short_series")

    scale = float(np.nanstd(r))
    if not np.isfinite(scale) or scale <= 1e-9:
        raise RuntimeError("egarch_low_scale")

    # Stabilize optimizer: scale residuals to percent-like range.
    r_scaled = (r / scale) * 100.0
    vol_dist = "t" if dist == "t" else "normal"

    am = arch_model(
        r_scaled,
        mean="Zero",
        vol="EGARCH",
        p=max(1, int(p)),
        o=max(0, int(o)),
        q=max(1, int(q)),
        dist=vol_dist,
        rescale=False,
    )
    res = am.fit(disp="off", show_warning=False)

    cond_vol_scaled = np.asarray(res.conditional_volatility, dtype=float)
    hist_var = (cond_vol_scaled * (scale / 100.0)) ** 2
    hist_var = np.maximum(hist_var, 1e-9)

    if steps <= 0:
        return hist_var, np.array([]), f"egarch_{p}{o}{q}_{vol_dist}"

    try:
        f = res.forecast(horizon=int(steps), reindex=False)
    except Exception:
        # EGARCH often requires simulation for multi-step forecasts.
        f = res.forecast(
            horizon=int(steps),
            method="simulation",
            simulations=500,
            random_state=42,
            reindex=False,
        )
    fc_var_scaled = np.asarray(f.variance.values[-1], dtype=float)
    fc_var = fc_var_scaled * ((scale / 100.0) ** 2)
    fc_var = np.maximum(fc_var, 1e-9)

    if len(fc_var) != int(steps):
        # Conservative fallback if library output shape differs.
        fc_var = np.repeat(float(fc_var[-1]) if len(fc_var) else float(np.var(r)), int(steps))
        fc_var = np.maximum(fc_var, 1e-9)

    return hist_var, fc_var, f"egarch_{p}{o}{q}_{vol_dist}"


def compute_volatility_path(
    resid_centered: np.ndarray,
    steps: int,
    vol_model: str,
    ewma_lam: float,
    egarch_p: int,
    egarch_o: int,
    egarch_q: int,
    egarch_dist: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (hist_var, fc_var, vol_model_used)."""
    vm = str(vol_model).lower()

    if vm in {"auto", "egarch"}:
        try:
            h, f, note = egarch_variance(
                resid=resid_centered,
                steps=steps,
                p=egarch_p,
                o=egarch_o,
                q=egarch_q,
                dist=egarch_dist,
            )
            return h, f, note
        except Exception as exc:
            if vm == "egarch":
                # Forced egarch request falls back to ewma for continuity.
                fallback_note = f"ewma_fallback_from_egarch:{type(exc).__name__}"
            else:
                fallback_note = f"ewma_fallback_from_auto:{type(exc).__name__}"
        else:
            fallback_note = "ewma"
    else:
        fallback_note = "ewma"

    # EWMA fallback (or explicit selection)
    ew_hist_var = ewma_variance(resid_centered, lam=ewma_lam)
    if len(ew_hist_var) == 0:
        ew_hist_var = np.repeat(max(float(np.var(resid_centered)), 1e-9), max(1, len(resid_centered)))

    unc_var = max(float(np.var(resid_centered)), 1e-9)
    ew_fc_var = ewma_forecast_var(last_var=float(ew_hist_var[-1]), unc_var=unc_var, steps=steps, lam=ewma_lam)
    return ew_hist_var, ew_fc_var, fallback_note


# -----------------------------------------------------------------------------
# CV and metrics
# -----------------------------------------------------------------------------

def split_points(n: int, holdout: int, splits: int, min_train: int) -> list[int]:
    out = []
    for i in range(splits, 0, -1):
        cut = n - i * holdout
        if min_train <= cut < n:
            out.append(cut)
    return sorted(set(out))


def make_cv_plan(n: int, freq: str, season_len: int, args: argparse.Namespace) -> CVPlan:
    if freq == "YS":
        hold = min(max(1, int(args.holdout_steps)), max(1, n // 5))
        hold = min(hold, 3)
        min_floor = max(8, season_len + 5)
    else:
        hold = min(max(1, int(args.holdout_steps)), max(1, n // 4))
        min_floor = max(24, season_len * 2)

    max_min_train = max(2, n - hold)
    min_train = min(max_min_train, max(min_floor, int(round(n * 0.60))))
    if min_train >= n:
        min_train = max(2, n - hold)

    splits = max(1, int(args.backtest_splits))
    while splits > 1 and not split_points(n, hold, splits, min_train):
        splits -= 1
    if not split_points(n, hold, splits, min_train):
        splits = 1

    return CVPlan(holdout=hold, splits=splits, min_train=min_train)


def metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    e = yt - yp

    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    bias = float(np.mean(e))

    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    ratio = np.zeros_like(denom, dtype=float)
    mask = denom > 1e-9
    ratio[mask] = np.abs(e[mask]) / denom[mask]
    smape = float(np.mean(ratio) * 100.0)

    return {"mae": mae, "rmse": rmse, "bias": bias, "smape": smape}


# -----------------------------------------------------------------------------
# Forecast engine
# -----------------------------------------------------------------------------

def make_future_index(last_ds: pd.Timestamp, periods: int, freq: str) -> pd.DatetimeIndex:
    if periods <= 0:
        return pd.DatetimeIndex([])
    if freq == "YS":
        return pd.date_range(last_ds + pd.offsets.YearBegin(1), periods=periods, freq="YS")
    return pd.date_range(last_ds + pd.offsets.MonthBegin(1), periods=periods, freq="MS")


def horizon_steps(last_ds: pd.Timestamp, target_year: int, freq: str) -> int:
    if freq == "YS":
        return max(0, target_year - last_ds.year)
    target_ds = pd.Timestamp(year=target_year, month=12, day=1)
    h = int((target_ds.year - last_ds.year) * 12 + (target_ds.month - last_ds.month))
    return max(0, h)


def fit_forecast_quant(
    ds: pd.Series,
    y: np.ndarray,
    future_ds: pd.Series,
    freq: str,
    regime_k: int,
    regime_maxiter: int,
    vol_model: str,
    ewma_lam: float,
    egarch_p: int,
    egarch_o: int,
    egarch_q: int,
    egarch_dist: str,
    variable: str,
) -> dict[str, Any]:
    ds = pd.to_datetime(ds).reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    fds = pd.to_datetime(future_ds).reset_index(drop=True)

    # Base mean (recency-weighted ridge with adaptive alpha)
    base_model, anchor, base_alpha = fit_base_ridge(ds, y, freq=freq, alpha=1.0)
    base_hist = predict_base(base_model, anchor=anchor, ds=ds, freq=freq)
    base_fc = predict_base(base_model, anchor=anchor, ds=fds, freq=freq)

    resid = y - base_hist

    # Regime model on residual
    pack = fit_markov_residuals(resid, k_regimes=regime_k, maxiter=regime_maxiter)

    # Historical expected regime mean from smoothed probs
    if pack.smoothed_probs.shape[0] == len(resid):
        reg_hist_mu = np.sum(pack.smoothed_probs * pack.means.reshape(1, -1), axis=1)
        reg_hist_var = np.sum(pack.smoothed_probs * pack.vars_.reshape(1, -1), axis=1)
    else:
        reg_hist_mu = np.zeros(len(resid), dtype=float)
        reg_hist_var = np.repeat(float(np.mean(pack.vars_)), len(resid))

    # Future regime expected path
    reg_fc_mu, reg_fc_var, fc_probs = regime_expected_path(pack, steps=len(fds))

    # Residual AR learner: captures local pattern memory beyond regime means.
    ar_input = resid - reg_hist_mu
    ar_model, ar_lags, ar_hist_mu, ar_alpha = fit_ar_residual_model(ds=ds, resid=ar_input, freq=freq)
    ar_fc_mu = predict_ar_future(model=ar_model, lags=ar_lags, resid_hist=ar_input, future_ds=fds, freq=freq)

    # Nonlinear booster on residuals (if validation gain is positive).
    boost_input = resid - reg_hist_mu - ar_hist_mu
    boost_model, boost_lags, boost_hist_mu_raw, boost_gain, boost_weight, boost_cfg = fit_boost_residual_model(
        ds=ds, resid=boost_input, freq=freq
    )
    boost_fc_mu_raw = predict_boost_future(model=boost_model, lags=boost_lags, resid_hist=boost_input, future_ds=fds, freq=freq)

    # Calibrate regime/AR/boost blend on recent residuals.
    default_blend = np.array([1.0, 1.0, float(boost_weight)], dtype=float)
    blend_w, blend_gain, blend_applied, blend_eval_n = calibrate_component_blend(
        resid=resid,
        reg_hist_mu=reg_hist_mu,
        ar_hist_mu=ar_hist_mu,
        boost_hist_mu=boost_hist_mu_raw,
        freq=freq,
        default_weights=default_blend,
    )
    reg_w = float(blend_w[0])
    ar_w = float(blend_w[1])
    boost_w = float(blend_w[2])
    reg_hist_adj = reg_hist_mu * reg_w
    ar_hist_adj = ar_hist_mu * ar_w
    boost_hist_adj = boost_hist_mu_raw * boost_w
    reg_fc_adj = reg_fc_mu * reg_w
    ar_fc_adj = ar_fc_mu * ar_w
    boost_fc_adj = boost_fc_mu_raw * boost_w

    # Analog projection from nearest historical windows (pattern memory).
    resid_core = resid - reg_hist_adj - ar_hist_adj - boost_hist_adj
    analog_fc_mu, analog_used, analog_window, analog_neighbors = analog_pattern_projection(
        resid=resid_core,
        steps=len(fds),
        freq=freq,
    )
    if is_monthly_freq(freq):
        base_analog_weight = 0.26
    elif is_weekly_freq(freq):
        base_analog_weight = 0.20
    elif is_daily_freq(freq):
        base_analog_weight = 0.16
    else:
        base_analog_weight = 0.18
    analog_weight, analog_strength = dynamic_analog_weight(resid_core=resid_core, freq=freq, base_weight=base_analog_weight)
    analog_fc_adj = analog_fc_mu * analog_weight

    trend_fc_raw, trend_slope = local_trend_projection(y=y, steps=len(fds), freq=freq)
    tail_win = int(np.clip(len(y) // 4, 8, 96))
    diff_scale = float(np.nanstd(np.diff(y[-tail_win:]))) if len(y) > 2 else 0.0
    if not np.isfinite(diff_scale) or diff_scale <= 1e-9:
        trend_weight = 0.0
    else:
        slope_strength = float(abs(trend_slope) / max(diff_scale, 1e-6))
        trend_weight = float(np.clip((slope_strength - 0.08) / 0.85, 0.0, 1.0))
    trend_fc_adj = trend_fc_raw * trend_weight

    # Combined mean path
    yhat_hist = base_hist + reg_hist_adj + ar_hist_adj + boost_hist_adj
    yhat_fc = base_fc + reg_fc_adj + ar_fc_adj + boost_fc_adj + analog_fc_adj + trend_fc_adj

    # Volatility path (EGARCH preferred when available).
    vol_hist_var, vol_fc_var, vol_model_used = compute_volatility_path(
        resid_centered=resid_core,
        steps=len(fds),
        vol_model=vol_model,
        ewma_lam=ewma_lam,
        egarch_p=egarch_p,
        egarch_o=egarch_o,
        egarch_q=egarch_q,
        egarch_dist=egarch_dist,
    )

    var_hist = np.maximum(vol_hist_var + reg_hist_var, 1e-9)
    var_fc = np.maximum(vol_fc_var + reg_fc_var, 1e-9)

    # Bounds safety
    yhat_hist = apply_bounds(yhat_hist, variable)
    yhat_fc = apply_bounds(yhat_fc, variable)

    return {
        "base_model": base_model,
        "anchor": anchor,
        "resid": resid,
        "pack": pack,
        "yhat_hist": yhat_hist,
        "yhat_fc": yhat_fc,
        "var_hist": var_hist,
        "var_fc": var_fc,
        "fc_probs": fc_probs,
        "aic": pack.aic,
        "note": pack.note,
        "base_alpha": float(base_alpha),
        "vol_model_used": vol_model_used,
        "ar_model_used": bool(ar_model is not None),
        "ar_lags": int(ar_lags),
        "ar_alpha": float(ar_alpha),
        "ar_hist_mu": ar_hist_adj,
        "ar_fc_mu": ar_fc_adj,
        "boost_model_used": bool(boost_model is not None),
        "boost_lags": int(boost_lags),
        "boost_gain": float(boost_gain),
        "boost_weight": float(boost_w),
        "boost_weight_suggested": float(boost_weight),
        "boost_cfg": str(boost_cfg),
        "blend_reg_weight": float(reg_w),
        "blend_ar_weight": float(ar_w),
        "blend_boost_weight": float(boost_w),
        "blend_gain": float(blend_gain),
        "blend_applied": bool(blend_applied),
        "blend_eval_n": int(blend_eval_n),
        "boost_hist_mu": boost_hist_adj,
        "boost_fc_mu": boost_fc_adj,
        "boost_hist_mu_raw": boost_hist_mu_raw,
        "boost_fc_mu_raw": boost_fc_mu_raw,
        "analog_used": bool(analog_used),
        "analog_window": int(analog_window),
        "analog_neighbors": int(analog_neighbors),
        "analog_weight": float(analog_weight),
        "analog_strength": float(analog_strength),
        "analog_fc_mu": analog_fc_adj,
        "trend_slope": float(trend_slope),
        "trend_weight": float(trend_weight),
        "trend_fc_adj": trend_fc_adj,
    }


def rolling_cv(
    ds: pd.Series,
    y: np.ndarray,
    freq: str,
    season_len: int,
    cv: CVPlan,
    args: argparse.Namespace,
    variable: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    cuts = split_points(len(y), holdout=cv.holdout, splits=cv.splits, min_train=cv.min_train)
    if not cuts:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "smape": np.nan,
            "rmse_std": np.nan,
            "n_folds": 0,
        }, pd.DataFrame(columns=["ds", "err", "abs_err", "sigma_pred", "nc_score"])

    fold_rmse = []
    fold_mae = []
    fold_bias = []
    fold_smape = []
    cal_rows = []

    for cut in cuts:
        tr_ds = pd.to_datetime(ds.iloc[:cut]).reset_index(drop=True)
        te_ds = pd.to_datetime(ds.iloc[cut : cut + cv.holdout]).reset_index(drop=True)
        tr_y = np.asarray(y[:cut], dtype=float)
        te_y = np.asarray(y[cut : cut + cv.holdout], dtype=float)

        if len(te_y) == 0:
            continue

        fit = fit_forecast_quant(
            ds=tr_ds,
            y=tr_y,
            future_ds=te_ds,
            freq=freq,
            regime_k=int(args.regime_k),
            regime_maxiter=int(args.regime_maxiter),
            vol_model=str(args.vol_model),
            ewma_lam=float(args.ewma_lambda),
            egarch_p=int(args.egarch_p),
            egarch_o=int(args.egarch_o),
            egarch_q=int(args.egarch_q),
            egarch_dist=str(args.egarch_dist),
            variable=variable,
        )

        pred = fit["yhat_fc"]
        met = metric_pack(te_y, pred)

        fold_rmse.append(met["rmse"])
        fold_mae.append(met["mae"])
        fold_bias.append(met["bias"])
        fold_smape.append(met["smape"])

        sigma = np.sqrt(np.maximum(np.asarray(fit["var_fc"], dtype=float), 1e-9))
        abs_e = np.abs(te_y - pred)
        if len(sigma):
            sigma_floor = max(float(np.quantile(sigma, 0.20)), 1e-6)
            sigma_safe = np.maximum(sigma, sigma_floor)
        else:
            sigma_safe = np.repeat(max(float(np.std(tr_y)), 1e-6), len(abs_e))
            sigma = sigma_safe.copy()
        nc = abs_e / sigma_safe

        for i in range(len(abs_e)):
            err = float(te_y[i] - pred[i])
            cal_rows.append(
                {
                    "ds": pd.Timestamp(te_ds.iloc[i]),
                    "err": err,
                    "abs_err": float(abs_e[i]),
                    "sigma_pred": float(sigma[i] if i < len(sigma) else np.nan),
                    "nc_score": float(nc[i]),
                }
            )

    if not fold_rmse:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "smape": np.nan,
            "rmse_std": np.nan,
            "n_folds": 0,
        }, pd.DataFrame(columns=["ds", "err", "abs_err", "sigma_pred", "nc_score"])

    return {
        "rmse": float(np.mean(fold_rmse)),
        "mae": float(np.mean(fold_mae)),
        "bias": float(np.mean(fold_bias)),
        "smape": float(np.mean(fold_smape)),
        "rmse_std": float(np.std(fold_rmse)),
        "n_folds": int(len(fold_rmse)),
    }, pd.DataFrame(cal_rows)


# -----------------------------------------------------------------------------
# Anomaly and plotting
# -----------------------------------------------------------------------------

def detect_anomalies(ds: pd.Series, y_true: np.ndarray, yhat: np.ndarray, var_hist: np.ndarray, z_thr: float, top_n: int, variable: str) -> pd.DataFrame:
    ds = pd.to_datetime(ds)
    e = np.asarray(y_true, dtype=float) - np.asarray(yhat, dtype=float)
    sigma = np.sqrt(np.maximum(np.asarray(var_hist, dtype=float), 1e-9))
    z = e / sigma
    abs_e = np.abs(e)

    q_tail = float(np.quantile(abs_e, 0.99)) if len(abs_e) else np.nan
    med = float(np.nanmedian(e)) if len(e) else 0.0
    mad = float(np.nanmedian(np.abs(e - med))) if len(e) else 0.0
    robust_scale = max(1.4826 * mad, 1e-6)
    robust_z = (e - med) / robust_scale

    flag_sigma = (np.abs(z) >= float(z_thr))
    flag_robust = (np.abs(robust_z) >= max(2.6, float(z_thr)))
    flag_tail = np.zeros_like(flag_sigma, dtype=bool)
    if np.isfinite(q_tail):
        flag_tail = abs_e >= q_tail

    iforest_score = np.repeat(np.nan, len(e)).astype(float)
    flag_iforest = np.zeros(len(e), dtype=bool)
    if len(e) >= 40:
        d1 = np.r_[0.0, np.diff(e)]
        X_iso = np.column_stack([e, z, robust_z, d1])
        iso = IsolationForest(
            n_estimators=300,
            contamination=min(0.12, max(0.03, float(top_n) / max(len(e), 1))),
            random_state=42,
        )
        pred = iso.fit_predict(X_iso)
        score = -iso.score_samples(X_iso)
        iforest_score = score.astype(float)
        q_if = float(np.quantile(iforest_score, 0.92))
        flag_iforest = (pred == -1) | (iforest_score >= q_if)

    flag = flag_sigma | flag_robust | flag_tail | flag_iforest

    out = pd.DataFrame(
        {
            "ds": ds,
            "variable": variable,
            "actual": y_true,
            "expected": yhat,
            "residual": e,
            "sigma": sigma,
            "zscore": z,
            "robust_zscore": robust_z,
            "abs_residual": abs_e,
            "iforest_score": iforest_score,
            "flag_sigma": flag_sigma,
            "flag_robust": flag_robust,
            "flag_tail": flag_tail,
            "flag_iforest": flag_iforest,
            "is_anomaly": flag,
        }
    )

    anom_type = []
    for r in out.itertuples(index=False):
        if abs(float(r.zscore)) >= max(3.0, float(z_thr) + 0.5):
            anom_type.append("şiddetli sapma")
        elif bool(r.flag_iforest) and bool(r.flag_robust):
            anom_type.append("örüntü kırılması")
        elif bool(r.flag_tail):
            anom_type.append("kuyruk olayı")
        elif bool(r.flag_sigma):
            anom_type.append("rejim dışı sapma")
        else:
            anom_type.append("zayıf anomali")
    out["anomaly_type_tr"] = anom_type

    anom_only = out[out["is_anomaly"]].copy()
    if anom_only.empty:
        anom_only = out.copy()

    anom_only = anom_only.sort_values("abs_residual", ascending=False).reset_index(drop=True)
    anom_only = anom_only.head(max(1, int(top_n)))
    anom_only["anomaly_rank"] = np.arange(1, len(anom_only) + 1, dtype=int)
    anom_only = anom_only.sort_values("ds").reset_index(drop=True)
    return anom_only


def conformal_q(abs_errors: list[float], alpha: float) -> float:
    if not abs_errors:
        return float("nan")
    q = float(np.clip(1.0 - alpha, 0.50, 0.99))
    return float(np.quantile(np.asarray(abs_errors, dtype=float), q))


def seasonal_bias_vector(
    cal_df: pd.DataFrame,
    future_ds: pd.Series,
    freq: str,
    global_bias: float,
) -> np.ndarray:
    if len(future_ds) == 0:
        return np.array([], dtype=float)
    out = np.repeat(float(global_bias), len(future_ds)).astype(float)
    if cal_df.empty or "err" not in cal_df.columns:
        return out
    if not is_monthly_freq(freq):
        return out

    c = cal_df.dropna(subset=["ds", "err"]).copy()
    if c.empty:
        return out
    c["month"] = pd.to_datetime(c["ds"]).dt.month
    g = c.groupby("month")["err"].agg(["mean", "count"]).reset_index()
    month_bias: dict[int, float] = {}
    for r in g.itertuples(index=False):
        m = int(r.month)
        mean_m = float(r.mean)
        n = int(r.count)
        # Shrink monthly bias toward global to avoid overfit on small folds.
        w = float(np.clip(n / (n + 3.0), 0.0, 1.0))
        month_bias[m] = float(global_bias + w * (mean_m - global_bias))

    months = pd.to_datetime(future_ds).dt.month.astype(int).values
    return np.array([month_bias.get(int(m), float(global_bias)) for m in months], dtype=float)


def match_global_events(ts: pd.Timestamp, variable: str) -> list[dict[str, str]]:
    t = pd.Timestamp(ts)
    var = canonical_variable_name(variable)
    hits: list[dict[str, str]] = []
    for ev in GLOBAL_EVENT_CATALOG:
        if var not in ev["tags"]:
            continue
        s = pd.Timestamp(ev["start"])
        e = pd.Timestamp(ev["end"])
        if s <= t <= e:
            hits.append({"title": ev["title"], "source": ev["source"]})
    return hits


def match_turkey_events(ts: pd.Timestamp, variable: str) -> list[dict[str, str]]:
    t = pd.Timestamp(ts)
    var = canonical_variable_name(variable)
    hits: list[dict[str, str]] = []
    for ev in TURKEY_EVENT_CATALOG:
        if var not in ev["tags"]:
            continue
        s = pd.Timestamp(ev["start"])
        e = pd.Timestamp(ev["end"])
        if s <= t <= e:
            hits.append(
                {
                    "title": ev["title"],
                    "region": ev.get("region", "Türkiye"),
                    "source": ev["source"],
                }
            )
    return hits


def infer_turkey_pattern_hint(ts: pd.Timestamp, variable: str, residual: float, zscore: float) -> str:
    t = pd.Timestamp(ts)
    m = int(t.month)
    var = canonical_variable_name(variable)
    z = abs(float(zscore))
    r = float(residual)

    # Kış aylarında NAO benzeri sinyal yorumu (heuristic).
    if m in {12, 1, 2} and z >= 1.2:
        if var == "temp" and r < 0:
            return "NAO-negatif benzeri kış paterni (daha soğuk koşullar)"
        if var == "temp" and r > 0:
            return "NAO-pozitif benzeri kış paterni (daha ılık koşullar)"
        if var == "precip" and r > 0:
            return "NAO-negatif benzeri patern (Marmara/Doğu Akdeniz yağış artışı)"
        if var == "precip" and r < 0:
            return "NAO-pozitif benzeri patern (yağış baskılanması)"
        if var == "pressure" and r > 0:
            return "NAO-pozitif benzeri patern (yüksek basınç eğilimi)"
        if var == "pressure" and r < 0:
            return "NAO-negatif benzeri patern (düşük basınç/siklonik eğilim)"

    # Yaz aylarında Doğu Akdeniz blokajı benzeri sinyal.
    if m in {6, 7, 8} and z >= 1.2:
        if var == "pressure" and r > 0:
            return "Doğu Akdeniz blokajı benzeri yüksek basınç sinyali"
        if var == "temp" and r > 0:
            return "Blokaj/sıcak dalga benzeri yaz paterni"
        if var == "precip" and r < 0:
            return "Doğu Akdeniz blokajı benzeri yağış azalışı"

    # Sonbahar yağışları için Akdeniz siklonik aktivite ipucu.
    if m in {9, 10, 11} and var == "precip" and r > 0 and z >= 1.2:
        return "Akdeniz siklonik aktivite artışı benzeri sonbahar sinyali"

    return ""


def _primary_cause_label(variable: str, residual: float, zscore: float) -> str:
    var = canonical_variable_name(variable)
    pos = residual >= 0
    az = abs(float(zscore))
    if var == "precip":
        if pos:
            return "aşırı yağış/sağanak sinyali" if az >= 2.8 else "yağış rejimi yukarı sapma"
        return "kuraklık veya yağış açığı sinyali"
    if var == "temp":
        return "sıcaklık yukarı sapma (sıcak dalga benzeri)" if pos else "sıcaklık aşağı sapma (soğuk dalga benzeri)"
    if var == "humidity":
        return "nemlilik yukarı sapma" if pos else "nemlilik aşağı sapma"
    if var == "pressure":
        return "yüksek basınç rejimi" if pos else "düşük basınç/siklonik rejim"
    return "istatistiksel anomali"


def enrich_anomalies_with_causes(
    anom_df: pd.DataFrame,
    ds_hist: pd.Series,
    y_hist: np.ndarray,
    yhat_hist: np.ndarray,
    variable: str,
    plan: FrequencyPlan,
) -> pd.DataFrame:
    if anom_df.empty:
        return anom_df

    hist = pd.DataFrame({"ds": pd.to_datetime(ds_hist), "actual": np.asarray(y_hist, dtype=float)})
    hist = hist.sort_values("ds").reset_index(drop=True)
    hist["resid"] = hist["actual"] - np.asarray(yhat_hist, dtype=float)
    hist["diff"] = hist["actual"].diff()

    diff_std = float(np.nanstd(hist["diff"].values)) if len(hist) > 3 else 0.0

    month_mu: dict[int, float] = {}
    if plan.freq == "MS":
        grp = hist.groupby(hist["ds"].dt.month)["actual"].mean()
        month_mu = {int(k): float(v) for k, v in grp.items()}

    hist_map = {pd.Timestamp(r.ds): float(r.actual) for r in hist.itertuples(index=False)}
    out = anom_df.copy()
    out["global_event_match"] = ""
    out["global_event_source"] = ""
    out["local_event_match"] = ""
    out["local_event_source"] = ""
    out["local_pattern_hint"] = ""
    out["cause_primary"] = ""
    out["cause_details_tr"] = ""
    out["month_climatology_delta"] = np.nan
    out["delta_prev"] = np.nan
    out["delta_yoy"] = np.nan
    out["cause_confidence"] = ""

    for i, row in out.iterrows():
        ts = pd.Timestamp(row["ds"])
        residual = float(row["residual"])
        z = float(row["zscore"])
        rz = float(row["robust_zscore"]) if "robust_zscore" in row else z
        primary = _primary_cause_label(variable=variable, residual=residual, zscore=z)
        a_type = str(row["anomaly_type_tr"]) if "anomaly_type_tr" in row else "anomali"

        prev_val = hist_map.get(ts - pd.offsets.MonthBegin(1), np.nan) if plan.freq == "MS" else hist_map.get(ts - pd.offsets.YearBegin(1), np.nan)
        curr_val = float(row["actual"])
        delta_prev = curr_val - float(prev_val) if np.isfinite(prev_val) else np.nan

        if plan.freq == "MS":
            prev_y = hist_map.get(ts - pd.DateOffset(years=1), np.nan)
            delta_yoy = curr_val - float(prev_y) if np.isfinite(prev_y) else np.nan
            clim = month_mu.get(int(ts.month), np.nan)
            clim_delta = curr_val - float(clim) if np.isfinite(clim) else np.nan
        else:
            delta_yoy = np.nan
            clim_delta = np.nan

        details = [f"anomali_tipi={a_type}", f"z={z:.2f}", f"robust_z={rz:.2f}", f"rezidü={residual:.2f}"]
        if np.isfinite(delta_prev):
            details.append(f"önceki_dönem_farkı={delta_prev:.2f}")
        if np.isfinite(delta_yoy):
            details.append(f"yıllık_fark={delta_yoy:.2f}")
        if np.isfinite(clim_delta):
            details.append(f"mevsim_normali_farkı={clim_delta:.2f}")

        if diff_std > 1e-9 and np.isfinite(delta_prev) and abs(delta_prev) > 1.5 * diff_std:
            details.append("ani rejim geçişi sinyali")
        if abs(z) >= 3.0:
            details.append("kuyruk olay şiddeti yüksek")
        elif abs(z) >= 2.2:
            details.append("kuyruk olay şiddeti orta")
        else:
            details.append("istatistiksel sapma")
        if bool(row.get("flag_iforest", False)):
            details.append("çok değişkenli örüntü kırılması (iforest)")
        if bool(row.get("flag_robust", False)) and abs(rz) >= 3.0:
            details.append("robust z ile güçlü sapma doğrulandı")

        local_hint = infer_turkey_pattern_hint(ts=ts, variable=variable, residual=residual, zscore=z)
        if local_hint:
            out.at[i, "local_pattern_hint"] = local_hint
            details.append(f"yerel_patern_ipucu={local_hint}")

        ev_hits = match_global_events(ts=ts, variable=variable)
        if ev_hits:
            out.at[i, "global_event_match"] = " | ".join([x["title"] for x in ev_hits])
            out.at[i, "global_event_source"] = " | ".join([x["source"] for x in ev_hits])
            details.append("global olay penceresi eşleşmesi var")
        else:
            out.at[i, "global_event_match"] = "doğrudan eşleşme yok"
            out.at[i, "global_event_source"] = ""

        local_hits = match_turkey_events(ts=ts, variable=variable)
        if local_hits:
            out.at[i, "local_event_match"] = " | ".join([f"{x['region']}: {x['title']}" for x in local_hits])
            out.at[i, "local_event_source"] = " | ".join([x["source"] for x in local_hits])
            details.append("Türkiye-odaklı olay penceresi eşleşmesi var")
        else:
            out.at[i, "local_event_match"] = "doğrudan eşleşme yok"
            out.at[i, "local_event_source"] = ""

        out.at[i, "cause_primary"] = primary
        out.at[i, "cause_details_tr"] = "; ".join(details)
        out.at[i, "month_climatology_delta"] = clim_delta
        out.at[i, "delta_prev"] = delta_prev
        out.at[i, "delta_yoy"] = delta_yoy
        out.at[i, "cause_confidence"] = "yüksek" if abs(z) >= 3.0 else ("orta" if abs(z) >= 2.0 else "düşük")

    return out


def build_global_anomaly_context_report(all_anoms: pd.DataFrame, report_md: Path, top_csv: Path) -> None:
    if all_anoms.empty:
        report_md.write_text("# Anomali Bağlam Raporu\n\nAnomali tespit edilmedi.\n", encoding="utf-8")
        pd.DataFrame().to_csv(top_csv, index=False)
        return

    top = all_anoms.copy()
    top["abs_z"] = top["zscore"].astype(float).abs()
    max_abs_res = float(np.nanmax(np.abs(top["residual"].astype(float).values))) if len(top) else 1.0
    max_if = float(np.nanmax(np.nan_to_num(top.get("iforest_score", 0.0)))) if len(top) else 1.0
    max_abs_res = max(max_abs_res, 1e-9)
    max_if = max(max_if, 1e-9)
    top["severity_score"] = (
        0.62 * top["abs_z"]
        + 0.28 * (np.abs(top["residual"].astype(float)) / max_abs_res)
        + 0.10 * (np.nan_to_num(top.get("iforest_score", 0.0)) / max_if)
    )
    top = top.sort_values("severity_score", ascending=False).head(20).reset_index(drop=True)
    keep = [
        "ds",
        "variable",
        "actual",
        "expected",
        "residual",
        "zscore",
        "robust_zscore",
        "anomaly_type_tr",
        "cause_primary",
        "cause_details_tr",
        "local_pattern_hint",
        "local_event_match",
        "local_event_source",
        "global_event_match",
        "global_event_source",
        "cause_confidence",
        "severity_score",
        "anomalies_csv",
    ]
    cols = [c for c in keep if c in top.columns]
    top[cols].to_csv(top_csv, index=False)

    lines = []
    lines.append("# Anomali Tarihleri ve Muhtemel Sebepler")
    lines.append("")
    lines.append("Not: Bu eşleme nedensellik kanıtı değildir; istatistiksel anomali + tarihsel bağlam uyumudur.")
    lines.append("")
    for i, r in top.iterrows():
        ds = pd.Timestamp(r["ds"]).strftime("%Y-%m")
        vtr = variable_tr(str(r["variable"]))
        lines.append(f"## {i+1}) {ds} | {vtr} | z={float(r['zscore']):.2f}")
        if "anomaly_type_tr" in r:
            lines.append(f"- Anomali tipi: {r.get('anomaly_type_tr', '')}")
        lines.append(f"- Muhtemel ana sebep: {r.get('cause_primary', '')}")
        lines.append(f"- Güven seviyesi: {r.get('cause_confidence', '')}")
        lines.append(f"- Teknik açıklama: {r.get('cause_details_tr', '')}")
        lh = str(r.get("local_pattern_hint", ""))
        if lh:
            lines.append(f"- Yerel patern ipucu: {lh}")
        lm = str(r.get("local_event_match", ""))
        ls = str(r.get("local_event_source", ""))
        if lm and lm != "doğrudan eşleşme yok":
            lines.append(f"- Türkiye-odaklı olay eşleşmesi: {lm}")
            if ls:
                lines.append(f"- Yerel kaynak: {ls}")
        else:
            lines.append("- Türkiye-odaklı olay eşleşmesi: doğrudan eşleşme bulunamadı")
        gm = str(r.get("global_event_match", ""))
        gs = str(r.get("global_event_source", ""))
        if gm and gm != "doğrudan eşleşme yok":
            lines.append(f"- Global olay eşleşmesi: {gm}")
            if gs:
                lines.append(f"- Kaynak: {gs}")
        else:
            lines.append("- Global olay eşleşmesi: doğrudan eşleşme bulunamadı")
        lines.append("")

    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def plot_forecast(out_df: pd.DataFrame, anom_df: pd.DataFrame, last_ds: pd.Timestamp, variable: str, plan: FrequencyPlan, chart_path: Path) -> None:
    fig, (ax, ax_note) = plt.subplots(
        1,
        2,
        figsize=(15.8, 5.2),
        gridspec_kw={"width_ratios": [3.8, 1.35], "wspace": 0.06},
    )
    hist = out_df[out_df["ds"] <= last_ds]
    fc = out_df[out_df["ds"] > last_ds]

    ax.plot(hist["ds"], hist["yhat"], color="#1f77b4", linewidth=1.4, label="Geçmiş uyum")
    if "actual" in hist.columns:
        ax.plot(hist["ds"], hist["actual"], color="#17becf", linewidth=1.0, alpha=0.55, label="Gözlem")

    if not fc.empty:
        ax.plot(fc["ds"], fc["yhat"], color="#d62728", linewidth=2.0, label="Projeksiyon")
        if fc["yhat_lower"].notna().any() and fc["yhat_upper"].notna().any():
            ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], color="#d62728", alpha=0.14, label="Güven bandı")

    # General trend line for readability (slow-moving envelope).
    trend_src = pd.concat(
        [
            pd.DataFrame({"ds": hist["ds"], "v": hist["actual"]}),
            pd.DataFrame({"ds": fc["ds"], "v": fc["yhat"]}),
        ],
        ignore_index=True,
    )
    trend_src = trend_src.dropna(subset=["ds", "v"]).sort_values("ds")
    if not trend_src.empty:
        win = 24 if plan.freq == "MS" else 5
        min_p = max(3, win // 3)
        trend = trend_src["v"].rolling(win, min_periods=min_p).mean()
        trend = trend.interpolate(limit_direction="both").ffill().bfill()
        ax.plot(trend_src["ds"], trend, color="#111111", linewidth=1.8, linestyle="--", alpha=0.9, label="Genel trend")

    if not anom_df.empty:
        ax.scatter(anom_df["ds"], anom_df["actual"], color="#ff7f0e", s=36, label="Anomaliler", zorder=3)

    ax.axvline(last_ds, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_title(f"Kuant Rejim Tahmini - {variable_tr(variable)} ({frequency_tr(plan.freq)})")
    ax.set_xlabel("Tarih")
    ax.set_ylabel(f"Değer ({infer_unit(variable)})")
    ax.grid(alpha=0.24)
    ax.legend(loc="best")

    # Sağ panel: anomali sebeplerini kısa not halinde göster.
    ax_note.axis("off")
    note_lines: list[str] = ["Anomali Nedenleri", ""]
    if anom_df.empty:
        note_lines.append("Kayıtlı anomali yok.")
    else:
        show = anom_df.copy().sort_values("abs_residual", ascending=False).head(6)
        for _, r in show.iterrows():
            ds = pd.Timestamp(r["ds"]).strftime("%Y-%m")
            primary = str(r.get("cause_primary", "")).strip()
            local = str(r.get("local_event_match", "")).strip()
            global_m = str(r.get("global_event_match", "")).strip()

            lead = f"{ds}: {primary}" if primary else f"{ds}: anomali"
            note_lines.append(lead)
            if local and local != "doğrudan eşleşme yok":
                note_lines.append(f" Yerel: {local.split('|')[0].strip()}")
            if global_m and global_m != "doğrudan eşleşme yok":
                note_lines.append(f" Global: {global_m.split('|')[0].strip()}")
            detail = str(r.get("anomaly_type_tr", "")).strip()
            if detail:
                note_lines.append(f" Tip: {detail}")
            note_lines.append("")

    wrapped_lines: list[str] = []
    for ln in note_lines:
        if ln.strip():
            wrapped_lines.extend(textwrap.wrap(ln, width=40))
        else:
            wrapped_lines.append("")
    note_text = "\n".join(wrapped_lines).strip()
    ax_note.text(
        0.02,
        0.98,
        note_text,
        ha="left",
        va="top",
        fontsize=9.6,
        family="monospace",
        bbox={"facecolor": "#f4f4f4", "alpha": 0.9, "edgecolor": "#999999", "boxstyle": "round,pad=0.5"},
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.12, wspace=0.06)
    fig.savefig(chart_path, dpi=170)
    plt.close(fig)


def plot_regime_probs(ds_fc: pd.Series, fc_probs: np.ndarray, prob_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    if fc_probs.size == 0:
        ax.text(0.03, 0.65, "Rejim olasılığı üretilemedi", fontsize=11)
        ax.axis("off")
    else:
        for j in range(fc_probs.shape[1]):
            ax.plot(ds_fc, fc_probs[:, j], linewidth=1.6, label=f"rejim_{j}")
        ax.set_ylim(0, 1)
        ax.set_title("Tahmin Rejim Olasılıkları")
        ax.set_xlabel("Tarih")
        ax.set_ylabel("Olasılık")
        ax.grid(alpha=0.22)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(prob_path, dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    climate_cfg = climate_cfg_from_args(args)

    raw = read_table(args.observations)
    obs, input_kind = normalize_observations(raw, args)
    vars_use = select_variables(obs, args.variables)

    if not vars_use:
        raise SystemExit("No variables selected")

    out = args.output_dir
    fc_dir = out / "forecasts"
    ch_dir = out / "charts"
    an_dir = out / "anomalies"
    rep_dir = out / "reports"
    ensure_dirs([out, fc_dir, ch_dir, an_dir, rep_dir])

    print(f"Girdi: {args.observations} | tür={input_kind} | satır={len(obs)} | değişkenler={','.join(vars_use)}")

    index_rows = []
    all_anomaly_rows: list[pd.DataFrame] = []

    for variable in vars_use:
        plan = choose_frequency_plan(obs, variable=variable, ok_value=args.qc_ok_value)
        s = aggregate_series(obs, variable=variable, plan=plan, ok_value=args.qc_ok_value)
        if s.empty:
            continue

        s = winsorize(s, ql=args.winsor_lower, qu=args.winsor_upper)

        ds = pd.Series(pd.to_datetime(s.index))
        y = s.values.astype(float)

        last_ds = pd.Timestamp(ds.iloc[-1])
        climate_cfg_var = climate_cfg_with_series_baseline(climate_cfg, last_ds)
        horizon = horizon_steps(last_ds, target_year=args.target_year, freq=plan.freq)
        fidx = make_future_index(last_ds, periods=horizon, freq=plan.freq)
        fds = pd.Series(fidx)

        cv = make_cv_plan(len(y), freq=plan.freq, season_len=plan.season_len, args=args)

        scores, cal_df = rolling_cv(
            ds=ds,
            y=y,
            freq=plan.freq,
            season_len=plan.season_len,
            cv=cv,
            args=args,
            variable=variable,
        )

        fit = fit_forecast_quant(
            ds=ds,
            y=y,
            future_ds=fds,
            freq=plan.freq,
            regime_k=int(args.regime_k),
            regime_maxiter=int(args.regime_maxiter),
            vol_model=str(args.vol_model),
            ewma_lam=float(args.ewma_lambda),
            egarch_p=int(args.egarch_p),
            egarch_o=int(args.egarch_o),
            egarch_q=int(args.egarch_q),
            egarch_dist=str(args.egarch_dist),
            variable=variable,
        )

        yhat_hist = fit["yhat_hist"].copy()
        yhat_fc = fit["yhat_fc"].copy()

        # Bias correction from CV:
        # - global mean error
        # - monthly seasonal correction (for monthly frequencies)
        bias_corr_global = float(scores["bias"]) if np.isfinite(scores["bias"]) else 0.0
        bias_corr_vec = seasonal_bias_vector(
            cal_df=cal_df,
            future_ds=fds,
            freq=plan.freq,
            global_bias=bias_corr_global,
        )
        yhat_fc = yhat_fc + bias_corr_vec

        # Optional scenario-based climate drift for future horizon.
        climate_delta_fc = climate_delta_series(fds, variable=variable, cfg=climate_cfg_var)
        if len(climate_delta_fc):
            yhat_fc = yhat_fc + climate_delta_fc
        yhat_fc = apply_bounds(yhat_fc, variable)

        # Adaptive conformal band:
        # 1) normalized score from CV (|err| / sigma_pred),
        # 2) seasonal absolute-error reference,
        # 3) robust cap from historical residual spread.
        alpha = float(args.interval_alpha)
        if is_precip(variable):
            alpha_eff = max(alpha, 0.24)
        elif plan.freq == "YS":
            alpha_eff = max(alpha, 0.18)
        else:
            alpha_eff = alpha

        q_abs = conformal_q(cal_df["abs_err"].dropna().tolist(), alpha=alpha_eff) if not cal_df.empty else float("nan")
        if not np.isfinite(q_abs):
            q_abs = float(np.quantile(np.abs(fit["resid"]), 0.80)) if len(fit["resid"]) else float(np.nanstd(y) * 0.1)

        q_norm = conformal_q(cal_df["nc_score"].dropna().tolist(), alpha=alpha_eff) if not cal_df.empty else float("nan")
        if not np.isfinite(q_norm):
            sigma_hist_safe = np.sqrt(np.maximum(np.asarray(fit["var_hist"], dtype=float), 1e-9))
            score_hist = np.abs(np.asarray(fit["resid"], dtype=float)) / np.maximum(sigma_hist_safe, 1e-6)
            q_norm = float(np.quantile(score_hist, np.clip(1.0 - alpha_eff, 0.50, 0.99))) if len(score_hist) else 1.64
        if is_precip(variable):
            q_norm = float(np.clip(q_norm, 0.50, 1.90))
        else:
            q_norm = float(np.clip(q_norm, 0.70, 2.80))

        sigma_fc = np.sqrt(np.maximum(fit["var_fc"], 1e-9))
        sigma_hist = np.sqrt(np.maximum(np.asarray(fit["var_hist"], dtype=float), 1e-9))
        if len(sigma_fc) and len(sigma_hist):
            lo_sig = max(float(np.quantile(sigma_hist, 0.15)) * 0.85, 1e-6)
            hi_sig = max(float(np.quantile(sigma_hist, 0.90)) * 1.30, lo_sig * 1.10)
            sigma_use = np.clip(sigma_fc, lo_sig, hi_sig)
        else:
            sigma_use = sigma_fc

        band_sigma = q_norm * sigma_use if len(sigma_use) else np.array([])

        seasonal_band = np.repeat(float(q_abs), len(fds)).astype(float)
        if plan.freq == "MS" and len(fds) and not cal_df.empty:
            cal_tmp = cal_df.copy()
            cal_tmp["month"] = pd.to_datetime(cal_tmp["ds"]).dt.month
            q_target = float(np.clip(1.0 - alpha_eff, 0.55, 0.98))
            per_month = {}
            for m in range(1, 13):
                vals = cal_tmp.loc[cal_tmp["month"] == m, "abs_err"].dropna().values
                if len(vals) >= max(3, int(cv.splits)):
                    per_month[m] = float(np.quantile(vals, q_target))
                else:
                    per_month[m] = float(q_abs)
            seasonal_band = np.array([per_month[int(pd.Timestamp(x).month)] for x in fds], dtype=float)

        hist_abs = np.abs(np.asarray(fit["resid"], dtype=float))
        if len(hist_abs):
            hist_q90 = float(np.quantile(hist_abs, 0.90))
            hist_q95 = float(np.quantile(hist_abs, 0.95))
        else:
            hist_q90 = float(q_abs)
            hist_q95 = float(q_abs)
        if is_precip(variable):
            band_cap = max(hist_q90 * 1.00, q_abs * 1.02, 1e-6)
            band_floor = max(q_abs * 0.08, 1e-6)
            w_sigma = 0.45
        else:
            band_cap = max(hist_q95 * 1.18, q_abs * 1.10, 1e-6)
            band_floor = max(q_abs * 0.10, 1e-6)
            w_sigma = 0.70

        blend = (w_sigma * band_sigma + (1.0 - w_sigma) * seasonal_band) if len(band_sigma) else seasonal_band
        band = np.clip(blend, band_floor, band_cap)

        lo_fc = apply_bounds(yhat_fc - band, variable)
        hi_fc = apply_bounds(yhat_fc + band, variable)

        hist_df = pd.DataFrame(
            {
                "ds": ds,
                "actual": y,
                "yhat": yhat_hist,
                "yhat_lower": np.nan,
                "yhat_upper": np.nan,
                "is_forecast": False,
                "climate_delta": 0.0,
            }
        )
        fc_df = pd.DataFrame(
            {
                "ds": fds,
                "actual": np.nan,
                "yhat": yhat_fc,
                "yhat_lower": lo_fc,
                "yhat_upper": hi_fc,
                "is_forecast": True,
                "climate_delta": climate_delta_fc,
            }
        )

        out_df = pd.concat([hist_df, fc_df], ignore_index=True)
        out_df["variable"] = variable
        out_df["unit"] = infer_unit(variable)
        out_df["frequency"] = plan.freq
        base_strategy = f"quant_markov_ar_boostblend_analog_{fit['vol_model_used']}_{fit['boost_cfg']}"
        out_df["model_strategy"] = f"{base_strategy}{climate_strategy_suffix(variable, climate_cfg_var)}"

        freq_tag = "yearly" if plan.freq == "YS" else "monthly"
        fc_csv = fc_dir / f"{variable}_{freq_tag}_quant_to_{args.target_year}.csv"
        fc_pq = fc_dir / f"{variable}_{freq_tag}_quant_to_{args.target_year}.parquet"
        out_df.to_csv(fc_csv, index=False)
        out_df.to_parquet(fc_pq, index=False)

        anom = detect_anomalies(
            ds=ds,
            y_true=y,
            yhat=yhat_hist,
            var_hist=fit["var_hist"],
            z_thr=float(args.anomaly_z),
            top_n=int(args.anomaly_top),
            variable=variable,
        )
        anom = enrich_anomalies_with_causes(
            anom_df=anom,
            ds_hist=ds,
            y_hist=y,
            yhat_hist=yhat_hist,
            variable=variable,
            plan=plan,
        )
        an_csv = an_dir / f"{variable}_{freq_tag}_anomalies_to_{args.target_year}.csv"
        anom["anomalies_csv"] = str(an_csv)
        anom.to_csv(an_csv, index=False)
        if not anom.empty:
            all_anomaly_rows.append(anom.copy())

        chart_png = ch_dir / f"{variable}_{freq_tag}_quant_to_{args.target_year}.png"
        prob_png = ch_dir / f"{variable}_{freq_tag}_regime_probs_to_{args.target_year}.png"

        plot_forecast(out_df=out_df, anom_df=anom, last_ds=last_ds, variable=variable, plan=plan, chart_path=chart_png)
        plot_regime_probs(ds_fc=fds, fc_probs=fit["fc_probs"], prob_path=prob_png)

        rep = {
            "variable": variable,
            "frequency": plan.freq,
            "frequency_label": plan.label,
            "climate_adjustment_enabled": bool(climate_cfg_var.enabled),
            "climate_scenario": str(climate_cfg_var.scenario),
            "climate_adjustment_method": str(climate_cfg_var.method),
            "climate_baseline_year": float(climate_cfg_var.baseline_year),
            "climate_temp_rate_c_per_year": float(climate_cfg_var.temp_rate_c_per_year),
            "humidity_per_temp_c": float(climate_cfg_var.humidity_per_temp_c),
            "monthly_coverage": float(plan.monthly_coverage),
            "cv_holdout_steps": int(cv.holdout),
            "cv_splits": int(cv.splits),
            "cv_min_train_steps": int(cv.min_train),
            "cv_rmse": float(scores["rmse"]) if np.isfinite(scores["rmse"]) else None,
            "cv_mae": float(scores["mae"]) if np.isfinite(scores["mae"]) else None,
            "cv_bias": float(scores["bias"]) if np.isfinite(scores["bias"]) else None,
            "cv_smape": float(scores["smape"]) if np.isfinite(scores["smape"]) else None,
            "cv_rmse_std": float(scores["rmse_std"]) if np.isfinite(scores["rmse_std"]) else None,
            "regime_success": bool(fit["pack"].success),
            "regime_k": int(fit["pack"].k_regimes),
            "regime_means": [float(x) for x in np.asarray(fit["pack"].means).ravel()],
            "regime_vars": [float(x) for x in np.asarray(fit["pack"].vars_).ravel()],
            "regime_aic": float(fit["aic"]) if np.isfinite(fit["aic"]) else None,
            "regime_note": fit["note"],
            "vol_model_requested": str(args.vol_model),
            "vol_model_used": str(fit["vol_model_used"]),
            "base_alpha": float(fit["base_alpha"]),
            "ar_model_used": bool(fit["ar_model_used"]),
            "ar_lags": int(fit["ar_lags"]),
            "ar_alpha": float(fit["ar_alpha"]),
            "boost_model_used": bool(fit["boost_model_used"]),
            "boost_lags": int(fit["boost_lags"]),
            "boost_cfg": str(fit["boost_cfg"]),
            "boost_gain": float(fit["boost_gain"]),
            "boost_weight": float(fit["boost_weight"]),
            "boost_weight_suggested": float(fit["boost_weight_suggested"]),
            "blend_reg_weight": float(fit["blend_reg_weight"]),
            "blend_ar_weight": float(fit["blend_ar_weight"]),
            "blend_boost_weight": float(fit["blend_boost_weight"]),
            "blend_gain": float(fit["blend_gain"]),
            "blend_applied": bool(fit["blend_applied"]),
            "blend_eval_n": int(fit["blend_eval_n"]),
            "analog_used": bool(fit["analog_used"]),
            "analog_window": int(fit["analog_window"]),
            "analog_neighbors": int(fit["analog_neighbors"]),
            "analog_weight": float(fit["analog_weight"]),
            "analog_strength": float(fit["analog_strength"]),
            "ewma_lambda": float(args.ewma_lambda),
            "egarch_p": int(args.egarch_p),
            "egarch_o": int(args.egarch_o),
            "egarch_q": int(args.egarch_q),
            "egarch_dist": str(args.egarch_dist),
            "bias_correction_global": float(bias_corr_global),
            "bias_correction_mode": "seasonal_monthly" if is_monthly_freq(plan.freq) else "global_only",
            "bias_correction_vector_mean": float(np.nanmean(bias_corr_vec)) if len(bias_corr_vec) else 0.0,
            "interval_alpha_requested": float(alpha),
            "interval_alpha_used": float(alpha_eff),
            "conformal_q_abs": float(q_abs),
            "conformal_q_norm": float(q_norm),
            "band_cap": float(band_cap),
            "band_floor": float(band_floor),
            "trend_slope": float(fit["trend_slope"]),
            "trend_weight": float(fit["trend_weight"]),
            "anomaly_count": int(len(anom)),
            "forecast_csv": str(fc_csv),
            "forecast_parquet": str(fc_pq),
            "anomalies_csv": str(an_csv),
            "chart_png": str(chart_png),
            "regime_probs_png": str(prob_png),
        }
        rep_json = rep_dir / f"{variable}_{freq_tag}_quant_report_to_{args.target_year}.json"
        rep_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

        index_rows.append(
            {
                "variable": variable,
                "frequency": plan.freq,
                "frequency_label": plan.label,
                "monthly_coverage": plan.monthly_coverage,
                "target_year": args.target_year,
                "cv_holdout_steps": cv.holdout,
                "cv_splits": cv.splits,
                "cv_min_train_steps": cv.min_train,
                "cv_rmse": scores["rmse"],
                "cv_mae": scores["mae"],
                "cv_bias": scores["bias"],
                "cv_smape": scores["smape"],
                "cv_rmse_std": scores["rmse_std"],
                "regime_success": fit["pack"].success,
                "regime_k": fit["pack"].k_regimes,
                "regime_aic": fit["aic"],
                "model_strategy": f"{base_strategy}{climate_strategy_suffix(variable, climate_cfg_var)}",
                "climate_adjustment_enabled": bool(climate_cfg_var.enabled),
                "climate_scenario": str(climate_cfg_var.scenario),
                "climate_adjustment_method": str(climate_cfg_var.method),
                "climate_baseline_year": float(climate_cfg_var.baseline_year),
                "climate_temp_rate_c_per_year": float(climate_cfg_var.temp_rate_c_per_year),
                "vol_model_requested": str(args.vol_model),
                "vol_model_used": str(fit["vol_model_used"]),
                "base_alpha": float(fit["base_alpha"]),
                "ar_model_used": bool(fit["ar_model_used"]),
                "ar_lags": int(fit["ar_lags"]),
                "ar_alpha": float(fit["ar_alpha"]),
                "boost_model_used": bool(fit["boost_model_used"]),
                "boost_lags": int(fit["boost_lags"]),
                "boost_cfg": str(fit["boost_cfg"]),
                "boost_gain": float(fit["boost_gain"]),
                "boost_weight": float(fit["boost_weight"]),
                "boost_weight_suggested": float(fit["boost_weight_suggested"]),
                "blend_reg_weight": float(fit["blend_reg_weight"]),
                "blend_ar_weight": float(fit["blend_ar_weight"]),
                "blend_boost_weight": float(fit["blend_boost_weight"]),
                "blend_gain": float(fit["blend_gain"]),
                "blend_applied": bool(fit["blend_applied"]),
                "blend_eval_n": int(fit["blend_eval_n"]),
                "analog_used": bool(fit["analog_used"]),
                "analog_window": int(fit["analog_window"]),
                "analog_neighbors": int(fit["analog_neighbors"]),
                "analog_weight": float(fit["analog_weight"]),
                "analog_strength": float(fit["analog_strength"]),
                "interval_alpha_used": float(alpha_eff),
                "trend_slope": float(fit["trend_slope"]),
                "trend_weight": float(fit["trend_weight"]),
                "forecast_csv": str(fc_csv),
                "forecast_parquet": str(fc_pq),
                "anomalies_csv": str(an_csv),
                "chart_png": str(chart_png),
                "regime_probs_png": str(prob_png),
                "report_json": str(rep_json),
                "input_kind": input_kind,
                "input_path": str(args.observations),
            }
        )

    idx = pd.DataFrame(index_rows).sort_values("variable") if index_rows else pd.DataFrame()
    idx_csv = out / f"quant_index_to_{args.target_year}.csv"
    idx_pq = out / f"quant_index_to_{args.target_year}.parquet"
    idx.to_csv(idx_csv, index=False)
    idx.to_parquet(idx_pq, index=False)

    all_anom = pd.concat(all_anomaly_rows, ignore_index=True) if all_anomaly_rows else pd.DataFrame()
    top_anom_csv = rep_dir / "top_anomalies_global_context_input.csv"
    world_md = rep_dir / "world_events_for_anomalies.md"
    build_global_anomaly_context_report(all_anoms=all_anom, report_md=world_md, top_csv=top_anom_csv)

    print("Kuant rejim hattı tamamlandı.")
    print(f"İndeks: {idx_csv}")
    if not idx.empty:
        cols = ["variable", "frequency", "cv_rmse", "cv_bias", "forecast_csv", "anomalies_csv"]
        print(idx[cols].to_string(index=False))


if __name__ == "__main__":
    main()
