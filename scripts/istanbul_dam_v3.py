#!/usr/bin/env python3
"""
İstanbul Baraj Doluluk Tahmini — v3 (Üstün Mimari)
====================================================
Bu sürüm, mevcut pipeline'ımıza uyarlanmıştır (dosya yolları + veri şeması).

Kurulum (opsiyonel gelişmiş özellikler için):
    pip install lightgbm xgboost optuna shap statsmodels

Kullanım:
    python scripts/istanbul_dam_v3.py
    python scripts/istanbul_dam_v3.py --tune --n-trials 80
    python scripts/istanbul_dam_v3.py --end-date 2040-12-01
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from scipy.special import logit, expit   # logit / sigmoid (inverse logit)
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from statsmodels.tsa.seasonal import STL
    HAS_STL = True
except Exception:
    HAS_STL = False

warnings.filterwarnings("ignore")

ROOT = Path("/Users/yasinkaya/Hackhaton")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("dam_v3")

EPS = 1e-4   # logit sınırlarından uzak durma payı


# ─────────────────────────────────────────────────────────────────────────────
# 1. Yardımcı — metrikler
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    mask = yt > 1.0
    if mask.sum() == 0:
        return np.nan
    return float(mean_absolute_percentage_error(yt[mask], yp[mask]) * 100)

def pearson_r(y_true, y_pred):
    if len(y_true) < 3:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Logit dönüşümü (bounded target için)
# ─────────────────────────────────────────────────────────────────────────────

def to_logit(fill_pct: np.ndarray) -> np.ndarray:
    """fill_pct ∈ [0,100]  →  logit(fill_pct/100) ∈ (-∞,+∞)"""
    p = np.clip(fill_pct / 100.0, EPS, 1.0 - EPS)
    return logit(p)

def from_logit(z: np.ndarray) -> np.ndarray:
    """logit uzayı → fill_pct ∈ [0,100]"""
    return expit(z) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fizik katmanı
# ─────────────────────────────────────────────────────────────────────────────

def compute_api(rain: pd.Series, k: float = 0.92) -> pd.Series:
    """
    Antecedent Precipitation Index — toprak nemi belleği.
    API(t) = k * API(t-1) + rain(t)
    k=0.92 aylık resolüsyon için hidroloji literatüründe standarttır.
    """
    api = np.zeros(len(rain))
    r = rain.fillna(0).values
    for i in range(1, len(r)):
        api[i] = k * api[i - 1] + r[i]
    return pd.Series(api, index=rain.index, name="api")


def compute_snow_proxy(t_mean: pd.Series, rain: pd.Series,
                       t_acc: float = 3.0, t_melt: float = 5.0,
                       melt_rate: float = 15.0) -> pd.Series:
    """
    Degree-day kar modeli.
    T < t_acc  → yağış kar olarak birikir
    T > t_melt → kar erir (melt_rate mm/°C/ay)
    İstanbul havzaları için: Uludağ etkisi, Trakya kar kapsamı.
    """
    snow = np.zeros(len(t_mean))
    t = t_mean.fillna(method="ffill").fillna(0).values
    r = rain.fillna(0).values
    for i in range(1, len(t)):
        accumulation = r[i] if t[i] < t_acc else 0.0
        melt = max(0.0, (t[i] - t_melt) * melt_rate) if t[i] > t_melt else 0.0
        snow[i] = max(0.0, snow[i - 1] + accumulation - melt)
    return pd.Series(snow, index=t_mean.index, name="snow_proxy_mm")


def compute_bucket_model(rain: pd.Series, et0: pd.Series,
                         capacity: float = 200.0,
                         demand_proxy_mm: float = 8.0) -> pd.Series:
    """
    Basit su dengesi bucket modeli.
    storage(t) = clip(storage(t-1) + rain(t) - ET0(t) - demand, 0, capacity)
    Bu; barajın toplam havzasını değil, toprağın yüzey nemi rezervini temsil eder.
    İdeal olarak havza alanı ve ölçeklendirilmiş gerçek depolama kapasitesiyle
    kalibre edilmeli — burada göreceli bir indeks olarak kullanılır.
    """
    storage = np.zeros(len(rain))
    r = rain.fillna(0).values
    e = et0.fillna(0).values
    storage[0] = capacity / 2.0
    for i in range(1, len(r)):
        inflow  = r[i]
        outflow = e[i] + demand_proxy_mm
        storage[i] = float(np.clip(storage[i - 1] + inflow - outflow, 0, capacity))
    return pd.Series(storage / capacity * 100.0, index=rain.index, name="bucket_pct")


def compute_stl_features(series: pd.Series, period: int = 12) -> pd.DataFrame:
    """
    STL ile trend ve residual bileşenleri.
    Sezon bileşeni özellik olarak kullanılmaz çünkü zaten trigonometrik
    kodlamayla temsil ediliyor.
    """
    if not HAS_STL or series.dropna().shape[0] < period * 2:
        n = len(series)
        return pd.DataFrame({
            "stl_trend":   np.zeros(n),
            "stl_resid":   np.zeros(n),
        }, index=series.index)
    try:
        stl = STL(series.fillna(series.median()), period=period, robust=True)
        res = stl.fit()
        return pd.DataFrame({
            "stl_trend": res.trend.values,
            "stl_resid": res.resid.values,
        }, index=series.index)
    except Exception as e:
        log.warning(f"STL başarısız: {e}")
        n = len(series)
        return pd.DataFrame({"stl_trend": np.zeros(n), "stl_resid": np.zeros(n)},
                             index=series.index)


def compute_spi(rain: pd.Series, window: int) -> pd.Series:
    """Standardised Precipitation Index benzeri: (rain - mean) / std"""
    rm = rain.rolling(window, min_periods=max(2, window // 2)).mean()
    rs = rain.rolling(window, min_periods=max(2, window // 2)).std().replace(0, np.nan)
    return ((rain - rm) / rs).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Klimatolojik taban çizgisi ve anomali hedefi
# ─────────────────────────────────────────────────────────────────────────────

def compute_climatology(fill_pct: pd.Series, months: pd.Series) -> pd.Series:
    """Her ay için gözlem ortalaması (sadece gerçek veriden)."""
    valid = fill_pct.dropna()
    valid_months = months.loc[valid.index]
    clim = valid.groupby(valid_months).mean()
    return clim  # Series indexed by month 1..12


def fill_to_anomaly(fill_pct: pd.Series, months: pd.Series,
                    climatology: pd.Series) -> pd.Series:
    """fill_pct → anomali = fill_pct - klimatoloji(ay)"""
    clim_vals = months.map(climatology)
    return fill_pct - clim_vals


def anomaly_to_fill(anomaly: pd.Series, months: pd.Series,
                    climatology: pd.Series) -> pd.Series:
    """Anomali → fill_pct = anomali + klimatoloji(ay)"""
    clim_vals = months.map(climatology)
    return (anomaly + clim_vals).clip(0, 100)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Özellik mühendisliği (tam)
# ─────────────────────────────────────────────────────────────────────────────

ALL_FEATURE_COLS = [
    # Ham iklim
    "rain_mm", "et0_mm_month", "t_mean_c", "rh_mean_pct", "pressure_kpa",
    "vpd_kpa_mean", "climate_balance_mm",
    # Fizik özellikleri (YENİ)
    "api", "snow_proxy_mm", "bucket_pct",
    # STL (YENİ)
    "stl_trend", "stl_resid",
    # Kuraklık indeksleri
    "spi_3", "spi_6", "spi_12",
    # Kümülatif denge
    "cum_balance_6", "cum_balance_12",
    # Delta
    "rain_delta", "et0_delta", "balance_delta", "t_delta", "rh_delta",
    # Hareketli ortalama — iklim
    "rain_ma3", "rain_ma6", "rain_ma12",
    "et0_ma3", "et0_ma6",
    "balance_ma3", "balance_ma6", "balance_ma12",
    # Mevsimsellik
    "month_sin", "month_cos", "quarter_sin", "quarter_cos",
    # Otoregresif — doluluk
    "lag1_fill", "lag2_fill", "lag3_fill", "lag6_fill", "lag12_fill",
    "fill_ma3", "fill_ma6",
    # Anomali lağları
    "lag1_anomaly", "lag3_anomaly", "lag12_anomaly",
    # Logit hedef lağları
    "lag1_logit", "lag12_logit",
]


def build_features(df: pd.DataFrame, climatology: pd.Series | None = None) -> pd.DataFrame:
    """
    Tüm özellikleri hesaplar. df sıralı olmalı (date artan).
    climatology: ay→ortalama doluluk Series; None ise anomali hesaplanmaz.
    """
    d = df.copy().sort_values("date").reset_index(drop=True)

    # VPD
    if "vpd_kpa_mean" not in d.columns or d["vpd_kpa_mean"].isna().all():
        es = 0.6108 * np.exp((17.27 * d["t_mean_c"]) / (d["t_mean_c"] + 237.3))
        d["vpd_kpa_mean"] = es - es * (d["rh_mean_pct"] / 100.0)
    else:
        d["vpd_kpa_mean"] = d["vpd_kpa_mean"].fillna(
            0.6108 * np.exp((17.27 * d["t_mean_c"]) / (d["t_mean_c"] + 237.3))
            * (1 - d["rh_mean_pct"] / 100.0)
        )

    d["climate_balance_mm"] = d["rain_mm"] - d["et0_mm_month"]
    d["month"]   = d["date"].dt.month
    d["quarter"] = d["date"].dt.quarter

    # ── Fizik özellikleri
    d["api"]          = compute_api(d["rain_mm"]).values
    d["snow_proxy_mm"]= compute_snow_proxy(d["t_mean_c"], d["rain_mm"]).values
    d["bucket_pct"]   = compute_bucket_model(d["rain_mm"], d["et0_mm_month"]).values

    # ── Kuraklık indeksleri
    d["spi_3"]  = compute_spi(d["rain_mm"], 3).values
    d["spi_6"]  = compute_spi(d["rain_mm"], 6).values
    d["spi_12"] = compute_spi(d["rain_mm"], 12).values

    # ── Kümülatif denge
    d["cum_balance_6"]  = d["climate_balance_mm"].rolling(6,  min_periods=1).sum()
    d["cum_balance_12"] = d["climate_balance_mm"].rolling(12, min_periods=1).sum()

    # ── Delta
    d["rain_delta"]    = d["rain_mm"].diff(1).fillna(0)
    d["et0_delta"]     = d["et0_mm_month"].diff(1).fillna(0)
    d["balance_delta"] = d["climate_balance_mm"].diff(1).fillna(0)
    d["t_delta"]       = d["t_mean_c"].diff(1).fillna(0)
    d["rh_delta"]      = d["rh_mean_pct"].diff(1).fillna(0)

    # ── Hareketli ortalama — iklim
    d["rain_ma3"]     = d["rain_mm"].rolling(3,  min_periods=1).mean()
    d["rain_ma6"]     = d["rain_mm"].rolling(6,  min_periods=1).mean()
    d["rain_ma12"]    = d["rain_mm"].rolling(12, min_periods=1).mean()
    d["et0_ma3"]      = d["et0_mm_month"].rolling(3,  min_periods=1).mean()
    d["et0_ma6"]      = d["et0_mm_month"].rolling(6,  min_periods=1).mean()
    d["balance_ma3"]  = d["climate_balance_mm"].rolling(3,  min_periods=1).mean()
    d["balance_ma6"]  = d["climate_balance_mm"].rolling(6,  min_periods=1).mean()
    d["balance_ma12"] = d["climate_balance_mm"].rolling(12, min_periods=1).mean()

    # ── Mevsimsellik
    d["month_sin"]   = np.sin(2 * np.pi * d["month"]   / 12)
    d["month_cos"]   = np.cos(2 * np.pi * d["month"]   / 12)
    d["quarter_sin"] = np.sin(2 * np.pi * d["quarter"] / 4)
    d["quarter_cos"] = np.cos(2 * np.pi * d["quarter"] / 4)

    # ── fill_pct yoksa klimatoloji ile başlat
    if "weighted_total_fill" in d.columns and "fill_pct" not in d.columns:
        d["fill_pct"] = d["weighted_total_fill"] * 100.0

    # ── Otoregresif özellikler (geçmiş fill'den)
    fp = d.get("fill_pct", pd.Series(np.nan, index=d.index))
    d["lag1_fill"]  = fp.shift(1)
    d["lag2_fill"]  = fp.shift(2)
    d["lag3_fill"]  = fp.shift(3)
    d["lag6_fill"]  = fp.shift(6)
    d["lag12_fill"] = fp.shift(12)
    d["fill_ma3"]   = fp.shift(1).rolling(3, min_periods=1).mean()
    d["fill_ma6"]   = fp.shift(1).rolling(6, min_periods=1).mean()

    # ── Anomali lağları
    if climatology is not None:
        anom = fill_to_anomaly(fp, d["month"], climatology)
        d["lag1_anomaly"]  = anom.shift(1)
        d["lag3_anomaly"]  = anom.shift(3)
        d["lag12_anomaly"] = anom.shift(12)
        # Logit lağları
        lgt = to_logit(fp.fillna(fp.median()).values)
        lgt_s = pd.Series(lgt, index=d.index)
        d["lag1_logit"]  = lgt_s.shift(1)
        d["lag12_logit"] = lgt_s.shift(12)
    else:
        for c in ["lag1_anomaly","lag3_anomaly","lag12_anomaly","lag1_logit","lag12_logit"]:
            d[c] = 0.0

    # ── STL (sadece geçmişte bilinen fill_pct üzerinde)
    if "fill_pct" in d.columns and not d["fill_pct"].isna().all():
        stl_feats = compute_stl_features(d["fill_pct"])
        d["stl_trend"] = stl_feats["stl_trend"].values
        d["stl_resid"]  = stl_feats["stl_resid"].values
    else:
        d["stl_trend"] = 0.0
        d["stl_resid"]  = 0.0

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 6. Veri yükleme
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge(panel_path: Path, climate_path: Path | None = None) -> pd.DataFrame:
    log.info(f"Panel yükleniyor: {panel_path.name}")
    panel = pd.read_csv(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    if "weighted_total_fill" in panel.columns:
        panel["fill_pct"] = panel["weighted_total_fill"] * 100.0

    if climate_path and climate_path.exists():
        log.info(f"İklim projeksiyonu: {climate_path.name}")
        clim = pd.read_csv(climate_path)
        clim["date"] = pd.to_datetime(clim["date"])
        if "precip_mm_month" in clim.columns:
            clim = clim.rename(columns={"precip_mm_month": "rain_mm"})
        future = clim[clim["date"].dt.year > panel["date"].dt.year.max()].copy()

        # Aylık klimatoloji (panel verisinden)
        panel["_month"] = panel["date"].dt.month
        clim_means = {}
        for col in ["rain_mm", "et0_mm_month", "t_mean_c", "rh_mean_pct", "pressure_kpa"]:
            if col in panel.columns:
                clim_means[col] = panel.groupby("_month")[col].mean()

        full_idx = pd.date_range(panel["date"].min(),
                                 pd.Timestamp("2040-12-01"), freq="MS")
        base = pd.DataFrame({"date": full_idx})
        base = base.merge(panel.drop(columns=["_month"], errors="ignore"),
                          on="date", how="left")

        future_cols = [c for c in ["rain_mm","et0_mm_month","t_mean_c","rh_mean_pct"]
                       if c in future.columns]
        base = base.merge(future[["date"] + future_cols], on="date", how="left",
                          suffixes=("", "_clim"))
        for col in future_cols:
            cc = f"{col}_clim"
            if cc in base.columns:
                base[col] = base[col].where(base[cc].isna(), base[cc])
                base.drop(columns=[cc], inplace=True)

        base["_month"] = base["date"].dt.month
        for col, monthly in clim_means.items():
            base[col] = base[col].fillna(base["_month"].map(monthly))
        base.drop(columns=["_month"], inplace=True)
        panel = base

    # Global ortalama fallback (fill_pct ve weighted_total_fill hariç)
    num_cols = panel.select_dtypes(include=np.number).columns
    exclude = {"fill_pct", "weighted_total_fill"}
    fill_cols = [c for c in num_cols if c not in exclude]
    panel[fill_cols] = panel[fill_cols].fillna(panel[fill_cols].mean())
    panel = panel.sort_values("date").reset_index(drop=True)
    log.info(f"Veri: {len(panel)} satır | "
             f"{panel['date'].min().date()} – {panel['date'].max().date()}")
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# 7. Model kataloğu
# ─────────────────────────────────────────────────────────────────────────────

def make_lgb_dart(params: dict | None = None) -> "lgb.LGBMRegressor":
    """LightGBM DART — en iyi regularizasyon."""
    defaults = dict(
        boosting_type="dart",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        drop_rate=0.1,        # DART: %10 ağaç silinir her iterasyonda
        skip_drop=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if params:
        defaults.update(params)
    return lgb.LGBMRegressor(**defaults)


def make_lgb_quantile(alpha: float) -> "lgb.LGBMRegressor":
    return lgb.LGBMRegressor(
        objective="quantile", alpha=alpha,
        boosting_type="gbdt",
        n_estimators=600, learning_rate=0.05,
        num_leaves=31, max_depth=6,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, n_jobs=-1, verbose=-1,
    )


def make_xgb(params: dict | None = None) -> "xgb.XGBRegressor":
    defaults = dict(
        n_estimators=600, learning_rate=0.04,
        max_depth=5, subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.5,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    if params:
        defaults.update(params)
    return xgb.XGBRegressor(**defaults)


def make_etr() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=400, max_features=0.6,
        min_samples_leaf=3, random_state=42, n_jobs=-1,
    )


def build_model_catalog(lgb_params: dict | None = None,
                        xgb_params: dict | None = None) -> dict:
    models = {}
    if HAS_LGB:
        models["lgb_dart"] = make_lgb_dart(lgb_params)
        log.info("  LightGBM DART eklendi")
    if HAS_XGB:
        models["xgb"] = make_xgb(xgb_params)
        log.info("  XGBoost eklendi")
    models["etr"] = make_etr()
    log.info("  Extra Trees eklendi")

    if len(models) >= 2:
        estimators = list(models.items())
        meta = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        models["stack"] = StackingRegressor(
            estimators=estimators,
            final_estimator=meta,
            cv=5, n_jobs=-1, passthrough=False,
        )
        log.info("  Stacking Ensemble eklendi")
    return models


# ─────────────────────────────────────────────────────────────────────────────
# 8. Purged Walk-Forward CV (embargo ile)
# ─────────────────────────────────────────────────────────────────────────────

def purged_walk_forward_cv(
    df: pd.DataFrame,
    models: dict,
    feat_cols: list,
    target_col: str = "anomaly_logit",   # logit uzayındaki anomali
    n_test_years: int = 5,
    embargo_months: int = 6,             # test öncesi bu kadar ay eğitimden çıkar
    min_train_months: int = 60,
) -> pd.DataFrame:
    """
    Purged Walk-Forward CV.
    Her test yılı için:
    - Eğitim: test_start - embargo_months öncesine kadar
    - Test: ilgili yıl
    - Sonuç: logit uzayından fill_pct'ye dönüştürülmüş RMSE
    """
    log.info(f"Purged WF-CV: {n_test_years} test yılı, {embargo_months} ay embargo")

    hist = df[df["fill_pct"].notna()].copy()
    hist = hist.dropna(subset=feat_cols + ["fill_pct"])
    if target_col not in hist.columns:
        log.warning(f"  Hedef sütun '{target_col}' yok, CV atlanıyor.")
        return pd.DataFrame()

    years = sorted(hist["date"].dt.year.unique())
    test_years = years[-n_test_years:]
    rows = []

    for test_year in test_years:
        test_start = pd.Timestamp(f"{test_year}-01-01")
        embargo_end = test_start - pd.DateOffset(months=embargo_months)

        train_df = hist[hist["date"] < embargo_end]
        test_df  = hist[hist["date"].dt.year == test_year]

        if len(train_df) < min_train_months or test_df.empty:
            log.warning(f"  {test_year}: yetersiz veri, atlandı.")
            continue

        X_tr = train_df[feat_cols].values
        y_tr = train_df[target_col].values
        X_te = test_df[feat_cols].values
        y_fill_te = test_df["fill_pct"].values

        for name, model in models.items():
            if name == "stack":
                continue   # Stacker CV'de çok yavaş
            t0 = time.time()
            try:
                model.fit(X_tr, y_tr)
                y_pred_logit = model.predict(X_te)
                # Geri dönüşüm: logit anomali → fill_pct
                clim_test = test_df["clim_fill"].values if "clim_fill" in test_df.columns else 0
                y_pred_fill = np.clip(from_logit(y_pred_logit) + clim_test, 0, 100)
                r = rmse(y_fill_te, y_pred_fill)
                m = mape(pd.Series(y_fill_te), pd.Series(y_pred_fill))
                p = pearson_r(y_fill_te, y_pred_fill)
                elapsed = time.time() - t0
                rows.append({
                    "model": name, "test_year": test_year,
                    "rmse_pp": round(r, 3), "mape_pct": round(m, 3),
                    "pearson_r": round(p, 3), "n_train": len(X_tr),
                    "elapsed_s": round(elapsed, 2),
                })
                log.info(f"  {test_year} │ {name:10s} │ RMSE={r:.2f}pp  "
                         f"MAPE={m:.1f}%  r={p:.3f}  ({elapsed:.1f}s)")
            except Exception as e:
                log.error(f"  {test_year} │ {name}: HATA — {e}")

    cv_df = pd.DataFrame(rows)
    if not cv_df.empty:
        summary = cv_df.groupby("model")[["rmse_pp","mape_pct","pearson_r"]].mean().round(3)
        log.info(f"\nCV Özeti:\n{summary.to_string()}\n")
    return cv_df


# ─────────────────────────────────────────────────────────────────────────────
# 9. Konformal Prediction (coverage-guaranteed bantlar)
# ─────────────────────────────────────────────────────────────────────────────

class SplitConformalPredictor:
    """
    Split Conformal Prediction.
    Kalibre seti residual'larından coverage-guaranteed bantlar hesaplar.
    Quantile regression'ın varsayımsal bantlarından üstün:
    verilen coverage (örn. %90) gerçekten %90 kapsıyor.
    """
    def __init__(self, coverage: float = 0.90):
        self.coverage = coverage
        self.q_hat: float = 0.0

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        residuals = np.abs(y_true - y_pred)
        alpha = 1 - self.coverage
        n = len(residuals)
        # Conformal quantile: ceil((n+1)(1-alpha))/n
        level = np.ceil((n + 1) * (1 - alpha)) / n
        level = min(level, 1.0)
        self.q_hat = float(np.quantile(residuals, level))
        log.info(f"  Konformal q̂ ({self.coverage*100:.0f}% coverage): "
                 f"±{self.q_hat:.2f} pp")

    def predict_interval(self, y_pred: np.ndarray):
        lower = np.clip(y_pred - self.q_hat, 0, 100)
        upper = np.clip(y_pred + self.q_hat, 0, 100)
        return lower, upper


# ─────────────────────────────────────────────────────────────────────────────
# 10. Horizon-aware dinamik blending
# ─────────────────────────────────────────────────────────────────────────────

def horizon_alpha(h: int,
                  h_ml_max: int = 24,
                  h_clim_start: int = 120,
                  alpha_ml: float = 0.95,
                  alpha_clim: float = 0.60) -> float:
    """
    h = forecast horizon (ay cinsinden, 1-indexed).
    α = ML'in ağırlığı; (1-α) = klimatolojinin ağırlığı.
    h ≤ 24      → α = alpha_ml (ML güvenilir)
    24 < h ≤ 120 → doğrusal geçiş
    h > 120     → α = alpha_clim (klimatoloji ağırlıklı, ama hâlâ ML baskın)
    """
    if h <= h_ml_max:
        return alpha_ml
    elif h >= h_clim_start:
        return alpha_clim
    else:
        t = (h - h_ml_max) / (h_clim_start - h_ml_max)
        return alpha_ml + t * (alpha_clim - alpha_ml)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Otoregresif projeksiyon (v3 — tam)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_v3(
    df: pd.DataFrame,
    model,
    climatology: pd.Series,
    conformal: SplitConformalPredictor,
    q10_model,
    q90_model,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    feat_cols: list,
    use_logit_target: bool = True,
    use_anomaly_target: bool = True,
    stochastic_residuals: bool = True,
    residual_scale: float = 0.9,
    rw_decay: float = 0.85,
    rw_scale: float = 0.9,
    seed: int = 42,
) -> pd.DataFrame:
    """
    İyileştirilmiş otoregresif projeksiyon:
    - Logit uzayında tahmin → fill_pct'ye geri dönüşüm
    - Anomali modeli → klimatoloji taban + ML sapması
    - Horizon-aware blending
    - Konformal belirsizlik bantları
    - Güvenli buffer tabanlı lag güncelleme
    """
    data = df.copy().sort_values("date").reset_index(drop=True)

    # İndex haritalama
    date_to_idx = {d: i for i, d in enumerate(data["date"])}
    start_i = date_to_idx.get(start_date)
    end_i   = date_to_idx.get(end_date)
    if start_i is None or end_i is None:
        log.error("start_date veya end_date veri setinde yok!")
        data["fill_sim"] = data.get("fill_pct", pd.Series(np.nan))
        data["fill_lo"]  = data["fill_sim"]
        data["fill_hi"]  = data["fill_sim"]
        return data

    # Çalışma buffer'ları
    fill_buf   = data["fill_pct"].values.copy().astype(float)  # gerçek + simüle
    result_sim = np.full(len(data), np.nan)
    result_lo  = np.full(len(data), np.nan)
    result_hi  = np.full(len(data), np.nan)

    # Geçmiş: gerçek değerleri direkt koy
    for i in range(start_i):
        result_sim[i] = fill_buf[i]
        result_lo[i]  = fill_buf[i]
        result_hi[i]  = fill_buf[i]

    def buf_lag(i, lag):
        pos = i - lag
        if pos < 0 or np.isnan(fill_buf[pos]):
            # Geri git, ilk geçerli değeri bul
            for k in range(pos - 1, -1, -1):
                if k >= 0 and not np.isnan(fill_buf[k]):
                    return float(fill_buf[k])
            return float(np.nanmean(fill_buf[:i])) if i > 0 else 50.0
        return float(fill_buf[pos])

    def buf_ma(i, window):
        vals = [fill_buf[i - k] for k in range(1, window + 1) if i - k >= 0
                and not np.isnan(fill_buf[i - k])]
        return float(np.mean(vals)) if vals else 50.0

    rng = np.random.default_rng(seed)
    rw_state = 0.0  # düşük frekanslı rastgele yürüyüş (daha doğal dalga)
    for year in range(start_date.year, end_date.year + 1):
        train_end = pd.Timestamp(f"{year - 1}-12-01")
        train_df = data[(data["date"] <= train_end) & data["fill_pct"].notna()].copy()
        train_df = train_df.dropna(subset=feat_cols + ["fill_pct"])
        if train_df.empty:
            continue

        X_tr = train_df[feat_cols].values

        # Hedef: logit anomali
        if use_anomaly_target and use_logit_target:
            anom_tr = fill_to_anomaly(train_df["fill_pct"],
                                       train_df["date"].dt.month, climatology)
            y_tr = to_logit(np.clip(anom_tr + 50.0, EPS * 100, (1 - EPS) * 100))
        elif use_anomaly_target:
            y_tr = fill_to_anomaly(train_df["fill_pct"],
                                    train_df["date"].dt.month, climatology).values
        elif use_logit_target:
            y_tr = to_logit(train_df["fill_pct"].values)
        else:
            y_tr = train_df["fill_pct"].values

        try:
            model.fit(X_tr, y_tr)
            q10_model.fit(X_tr, y_tr)
            q90_model.fit(X_tr, y_tr)
        except Exception as e:
            log.error(f"  {year}: eğitim hatası — {e}")
            continue

        # residual dağılımı (ay bazında) — düzenliliği kırmak için
        month_resid = {m: [] for m in range(1, 13)}
        if stochastic_residuals:
            try:
                pred_tr = model.predict(X_tr)
                clim_tr = train_df["date"].dt.month.map(climatology).values
                if use_anomaly_target and use_logit_target:
                    pred_fill = np.clip(from_logit(pred_tr) - 50.0 + clim_tr, 0, 100)
                elif use_logit_target:
                    pred_fill = np.clip(from_logit(pred_tr), 0, 100)
                else:
                    pred_fill = np.clip(pred_tr, 0, 100)
                resid = train_df["fill_pct"].values - pred_fill
                for m, r in zip(train_df["date"].dt.month.values, resid):
                    month_resid[int(m)].append(float(r))
            except Exception as e:
                log.warning(f"  Residual hesaplanamadı ({year}): {e}")

        year_mask = ((data["date"] >= pd.Timestamp(f"{year}-01-01")) &
                     (data["date"] <= min(pd.Timestamp(f"{year}-12-01"), end_date)) &
                     (data["date"] >= start_date))
        idxs = data.index[year_mask].tolist()

        for h, i in enumerate(idxs, 1):
            row = data.loc[i].copy()
            month = int(row["month"]) if "month" in row else row["date"].month

            # Otoregresif özellikleri güncelle
            for lag, col in [(1,"lag1_fill"),(2,"lag2_fill"),(3,"lag3_fill"),
                              (6,"lag6_fill"),(12,"lag12_fill")]:
                row[col] = buf_lag(i, lag)
            row["fill_ma3"] = buf_ma(i, 3)
            row["fill_ma6"] = buf_ma(i, 6)

            # Anomali lag'ları
            for lag, col in [(1,"lag1_anomaly"),(3,"lag3_anomaly"),(12,"lag12_anomaly")]:
                lf = buf_lag(i, lag)
                clim_m = climatology.get(
                    data.loc[i - lag, "date"].month if i - lag >= 0 else month, lf
                )
                row[col] = lf - clim_m

            # Logit lag'ları
            row["lag1_logit"]  = float(to_logit(np.clip(buf_lag(i, 1),  1, 99)))
            row["lag12_logit"] = float(to_logit(np.clip(buf_lag(i, 12), 1, 99)))

            # STL güncelleme (yaklaşık — gerçek veri olmadan STL yenilemek mümkün değil)
            row["stl_trend"] = float(np.nanmean([buf_lag(i, k) for k in range(1, 13)]))
            row["stl_resid"]  = buf_lag(i, 1) - row["stl_trend"]

            # NaN koruması
            for fc in feat_cols:
                if pd.isna(row.get(fc, np.nan)):
                    row[fc] = float(X_tr[:, feat_cols.index(fc)].mean())

            X_pred = row[feat_cols].values.reshape(1, -1)

            try:
                pred_raw    = float(model.predict(X_pred)[0])
                pred_q10    = float(q10_model.predict(X_pred)[0])
                pred_q90    = float(q90_model.predict(X_pred)[0])
            except Exception as e:
                log.warning(f"  Tahmin hatası i={i}: {e}")
                pred_raw = pred_q10 = pred_q90 = to_logit(50.0) if use_logit_target else 50.0

            # Geri dönüşüm → fill_pct
            clim_val = float(climatology.get(month, 50.0))
            if use_anomaly_target and use_logit_target:
                # logit anomali → anomali → fill
                anom_pred    = from_logit(pred_raw)    - 50.0
                anom_q10     = from_logit(pred_q10)   - 50.0
                anom_q90     = from_logit(pred_q90)   - 50.0
                fill_pred    = float(np.clip(clim_val + anom_pred, 0, 100))
                fill_q10_raw = float(np.clip(clim_val + anom_q10,  0, 100))
                fill_q90_raw = float(np.clip(clim_val + anom_q90,  0, 100))
            elif use_logit_target:
                fill_pred    = float(np.clip(from_logit(pred_raw), 0, 100))
                fill_q10_raw = float(np.clip(from_logit(pred_q10), 0, 100))
                fill_q90_raw = float(np.clip(from_logit(pred_q90), 0, 100))
            else:
                fill_pred    = float(np.clip(pred_raw, 0, 100))
                fill_q10_raw = float(np.clip(pred_q10, 0, 100))
                fill_q90_raw = float(np.clip(pred_q90, 0, 100))

            # Horizon-aware blending: uzak gelecekte klimatolojiye yaklaş
            α = horizon_alpha(h)
            fill_blended = α * fill_pred + (1 - α) * clim_val

            # Stochastic residuals (ay bazında) — simülasyonu aşırı düzenlilikten çıkar
            if stochastic_residuals:
                pool = month_resid.get(month, [])
                if pool:
                    noise = float(rng.choice(pool)) * residual_scale
                else:
                    noise = float(rng.normal(0.0, 2.0)) * residual_scale
                fill_blended = fill_blended + noise

            # düşük frekanslı sapma (rejim değişimi hissi)
            rw_state = rw_decay * rw_state + float(rng.normal(0.0, 1.0))
            fill_blended = fill_blended + rw_state * rw_scale
            fill_blended = float(np.clip(fill_blended, 0, 100))

            # Konformal bantlar
            lo_conf, hi_conf = conformal.predict_interval(np.array([fill_blended]))
            fill_lo = float(min(lo_conf[0], fill_q10_raw))
            fill_hi = float(max(hi_conf[0], fill_q90_raw))
            fill_lo = float(np.clip(fill_lo, 0, 100))
            fill_hi = float(np.clip(fill_hi, 0, 100))

            result_sim[i] = fill_blended
            result_lo[i]  = fill_lo
            result_hi[i]  = fill_hi
            fill_buf[i]   = fill_blended

    out = data[["date"]].copy()
    out["fill_pct"] = data.get("fill_pct", pd.Series(np.nan, index=data.index))
    out["fill_sim"] = result_sim
    out["fill_lo"]  = result_lo
    out["fill_hi"]  = result_hi
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 12. Optuna
# ─────────────────────────────────────────────────────────────────────────────

def optuna_tune(X: np.ndarray, y: np.ndarray, n_trials: int = 60,
                model_type: str = "lgb") -> dict:
    if not HAS_OPTUNA:
        return {}
    log.info(f"Optuna başlıyor: {n_trials} deneme ({model_type})")

    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        if model_type == "lgb" and HAS_LGB:
            params = dict(
                boosting_type="dart",
                n_estimators=trial.suggest_int("n_estimators", 200, 800),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 63),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 30),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                drop_rate=trial.suggest_float("drop_rate", 0.05, 0.3),
                random_state=42, n_jobs=-1, verbose=-1,
            )
            m = lgb.LGBMRegressor(**params)
        elif model_type == "xgb" and HAS_XGB:
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 800),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                random_state=42, n_jobs=-1, verbosity=0,
            )
            m = xgb.XGBRegressor(**params)
        else:
            return 999.0

        scores = []
        for tr_idx, val_idx in tscv.split(X):
            m.fit(X[tr_idx], y[tr_idx])
            scores.append(rmse(y[val_idx], m.predict(X[val_idx])))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"Optuna tamamlandı — en iyi CV RMSE: {study.best_value:.4f}")
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# 13. SHAP
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(model, X: np.ndarray, feat_cols: list,
                      out_dir: Path, tag: str = "best"):
    if not HAS_SHAP:
        return
    log.info(f"SHAP hesaplanıyor ({tag})…")
    try:
        try:
            exp = shap.TreeExplainer(model)
            sv  = exp.shap_values(X)
        except Exception:
            exp = shap.LinearExplainer(model, X)
            sv  = exp.shap_values(X)

        mean_abs = np.abs(sv).mean(axis=0)
        imp = pd.Series(mean_abs, index=feat_cols).sort_values()
        top = imp.tail(20)

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ["#185FA5" if v > imp.median() else "#85B7EB" for v in top.values]
        ax.barh(top.index, top.values, color=colors, height=0.7)
        ax.set_xlabel("Ortalama |SHAP| (logit-anomali uzayı)", fontsize=10)
        ax.set_title(f"SHAP Özellik Önemi — {tag}", fontsize=11)
        ax.axvline(top.values.mean(), color="#888", lw=0.7, ls="--")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        fig.savefig(out_dir / f"shap_{tag}.png", dpi=150)
        plt.close(fig)
        log.info(f"  SHAP kaydedildi: shap_{tag}.png")

        top5 = imp.tail(5)[::-1]
        log.info(f"  Top 5 özellik: {list(top5.index)}")
    except Exception as e:
        log.warning(f"  SHAP başarısız: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Görselleştirme
# ─────────────────────────────────────────────────────────────────────────────

def plot_full(out: pd.DataFrame, model_name: str, last_obs: pd.Timestamp,
              cv_summary: dict | None, fig_dir: Path, climatology: pd.Series):
    hist_m = out["date"] <= last_obs
    fut_m  = out["date"] >  last_obs

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, height_ratios=[2.2, 1],
                             hspace=0.40, wspace=0.30)
    ax_main  = fig.add_subplot(gs[0, :])
    ax_zoom  = fig.add_subplot(gs[1, 0])
    ax_yr    = fig.add_subplot(gs[1, 1])
    ax_clim  = fig.add_subplot(gs[1, 2])

    # ── Ana grafik
    ax_main.plot(out["date"][hist_m], out["fill_pct"][hist_m],
                 color="#111", lw=1.3, label="Gözlenen", zorder=4)
    ax_main.plot(out["date"][fut_m],  out["fill_sim"][fut_m],
                 color="#D55E00", lw=1.8, label=f"Tahmin — {model_name}", zorder=5)
    ax_main.fill_between(
        out["date"][fut_m],
        out["fill_lo"][fut_m].clip(0,100),
        out["fill_hi"][fut_m].clip(0,100),
        color="#D55E00", alpha=0.15, lw=0, label="Konformal %90 bandı",
    )
    ax_main.axvline(last_obs, color="#888", lw=0.8, ls="--", alpha=0.7)
    ax_main.axhline(30, color="#E24B4A", lw=0.7, ls=":", alpha=0.55)
    ax_main.text(out["date"].iloc[0], 31.5, "Kritik eşik (%30)",
                 fontsize=8, color="#E24B4A", alpha=0.7)
    ax_main.set_ylim(0, 108)
    ax_main.set_ylabel("Doluluk (%)", fontsize=10)
    ax_main.set_title(
        f"İstanbul Baraj Doluluk — {model_name.upper()} │ "
        f"Logit+Anomali+Fizik+Konformal",
        fontsize=11,
    )
    ax_main.legend(loc="upper right", fontsize=9)
    ax_main.grid(True, color="#EEEEEE", lw=0.5)
    ax_main.spines[["top","right"]].set_visible(False)

    if cv_summary:
        mt = (f"CV (purged): RMSE={cv_summary.get('rmse_pp','?'):.2f}pp  "
              f"MAPE={cv_summary.get('mape_pct','?'):.1f}%  "
              f"Pearson={cv_summary.get('pearson_r','?'):.3f}")
        ax_main.text(0.01, 0.02, mt, transform=ax_main.transAxes,
                     fontsize=8.5, color="#555",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#F9F9F9",
                               ec="#CCCCCC", alpha=0.9))

    # ── Zoom: son 4 yıl
    cutoff = last_obs - pd.DateOffset(years=4)
    zm = (out["date"] >= cutoff) & hist_m
    ax_zoom.plot(out["date"][zm], out["fill_pct"][zm], "#111", lw=1.2)
    ax_zoom.set_title("Son 4 yıl (gözlenen)", fontsize=9)
    ax_zoom.set_ylabel("Doluluk (%)")
    ax_zoom.grid(True, color="#EEE", lw=0.5)
    ax_zoom.spines[["top","right"]].set_visible(False)

    # ── Yıllık ortalama çubuk
    def yr_mean(g):
        if (g["date"] > last_obs).any():
            return g["fill_sim"].mean()
        return g["fill_pct"].mean()
    yearly = out.groupby(out["date"].dt.year).apply(yr_mean).dropna()
    cols_yr = ["#185FA5" if y <= last_obs.year else "#D55E00" for y in yearly.index]
    ax_yr.bar(yearly.index, yearly.values, color=cols_yr, alpha=0.85, width=0.8)
    ax_yr.axhline(30, color="#E24B4A", lw=0.7, ls=":", alpha=0.5)
    ax_yr.set_title("Yıllık ortalama doluluk", fontsize=9)
    ax_yr.set_ylim(0, 100)
    ax_yr.spines[["top","right"]].set_visible(False)
    ax_yr.grid(True, axis="y", color="#EEE", lw=0.5)
    ax_yr.legend(handles=[
        Patch(color="#185FA5", alpha=0.85, label="Gözlenen"),
        Patch(color="#D55E00", alpha=0.85, label="Tahmin"),
    ], fontsize=8)

    # ── Klimatoloji (aylık ortalama) çubuk
    months_tr = ["Oca","Şub","Mar","Nis","May","Haz","Tem","Ağu","Eyl","Eki","Kas","Ara"]
    ax_clim.bar(range(1,13), [climatology.get(m, 0) for m in range(1,13)],
                color="#0F6E56", alpha=0.7, width=0.7)
    ax_clim.set_xticks(range(1,13))
    ax_clim.set_xticklabels(months_tr, fontsize=7)
    ax_clim.set_title("Aylık klimatoloji (taban çizgisi)", fontsize=9)
    ax_clim.set_ylabel("Ort. doluluk (%)")
    ax_clim.spines[["top","right"]].set_visible(False)
    ax_clim.grid(True, axis="y", color="#EEE", lw=0.5)

    plt.savefig(fig_dir / f"{model_name}_v3_projection.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Grafik: {model_name}_v3_projection.png")


def plot_cv_comparison(cv_df: pd.DataFrame, fig_dir: Path):
    if cv_df.empty:
        return
    summary = cv_df.groupby("model")["rmse_pp"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#185FA5" if i == 0 else "#85B7EB" for i in range(len(summary))]
    bars = ax.barh(summary.index, summary.values, color=colors, alpha=0.9)
    ax.bar_label(bars, fmt="%.2f pp", padding=4, fontsize=9)
    ax.set_xlabel("Ort. CV RMSE (puan)", fontsize=10)
    ax.set_title("Model Karşılaştırması — Purged Walk-Forward CV", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, axis="x", color="#EEE", lw=0.5)
    plt.tight_layout()
    fig.savefig(fig_dir / "model_comparison_cv.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 15. Ana pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="İstanbul Baraj v3")
    p.add_argument("--panel",
        default="output/newdata_feature_store/tables/"
                "istanbul_dam_driver_panel_2000_2026_extended.csv")
    p.add_argument("--climate",
        default="output/scientific_climate_projection_2026_2040/"
                "climate_projection_2010_2040_monthly.csv")
    p.add_argument("--out",       default="output/istanbul_v3")
    p.add_argument("--tune",      action="store_true")
    p.add_argument("--n-trials",  type=int, default=60)
    p.add_argument("--end-date",  default="2040-12-01")
    p.add_argument("--cv-folds",  type=int, default=5)
    p.add_argument("--coverage",  type=float, default=0.90,
                   help="Konformal coverage (örn. 0.90 = %90)")
    p.add_argument("--no-shap",   action="store_true")
    args = p.parse_args()

    out_dir = ROOT / args.out
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-8s │ %(message)s"))
    log.addHandler(fh)

    log.info("=" * 65)
    log.info("İstanbul Baraj Doluluk v3 — başladı")
    log.info(f"  LightGBM: {HAS_LGB}  XGBoost: {HAS_XGB}  "
             f"Optuna: {HAS_OPTUNA}  SHAP: {HAS_SHAP}  STL: {HAS_STL}")
    log.info("=" * 65)

    # ── 1. Veri yükle
    clim_path = ROOT / args.climate
    raw = load_and_merge(ROOT / args.panel,
                         clim_path if clim_path.exists() else None)

    last_obs   = raw[raw["fill_pct"].notna()]["date"].max()
    start_date = (pd.Timestamp(last_obs) + pd.DateOffset(months=1)).normalize()
    end_date   = pd.Timestamp(args.end_date)
    log.info(f"Son gözlem: {last_obs.date()} | "
             f"Projeksiyon: {start_date.date()} → {end_date.date()}")

    # ── 2. Klimatoloji hesapla (sadece geçmiş veriden)
    hist_raw = raw[raw["fill_pct"].notna()].copy()
    climatology = compute_climatology(hist_raw["fill_pct"],
                                       hist_raw["date"].dt.month)
    log.info(f"Klimatoloji (aylık ort.):\n"
             f"  {dict(climatology.round(1))}")
    raw["clim_fill"] = raw["date"].dt.month.map(climatology)

    # ── 3. Özellik mühendisliği
    log.info("Özellik mühendisliği…")
    df = build_features(raw, climatology)

    # Anomali ve logit-anomali hedefleri
    df["anomaly"] = fill_to_anomaly(df["fill_pct"], df["month"], climatology)
    df["anomaly_logit"] = np.where(
        df["fill_pct"].notna(),
        to_logit(np.clip(df["anomaly"] + 50.0, EPS * 100, (1 - EPS) * 100)),
        np.nan,
    )

    # Kullanılabilir özellikler
    feat_cols = [f for f in ALL_FEATURE_COLS if f in df.columns]
    log.info(f"Özellik sayısı: {len(feat_cols)}")

    # ── 4. Optuna (opsiyonel)
    lgb_params, xgb_params = {}, {}
    if args.tune:
        hist_df = df[df["fill_pct"].notna()].dropna(
            subset=feat_cols + ["anomaly_logit"])
        X_all = hist_df[feat_cols].values
        y_all = hist_df["anomaly_logit"].values
        if HAS_LGB:
            lgb_params = optuna_tune(X_all, y_all, args.n_trials, "lgb")
        if HAS_XGB:
            xgb_params = optuna_tune(X_all, y_all, args.n_trials, "xgb")

    # ── 5. Modeller
    log.info("Modeller kuruluyor…")
    models = build_model_catalog(lgb_params, xgb_params)
    if HAS_LGB:
        q10_model = make_lgb_quantile(0.10)
        q90_model = make_lgb_quantile(0.90)
    else:
        from sklearn.linear_model import QuantileRegressor
        q10_model = make_pipeline(StandardScaler(),
                                  QuantileRegressor(quantile=0.10, solver="highs"))
        q90_model = make_pipeline(StandardScaler(),
                                  QuantileRegressor(quantile=0.90, solver="highs"))

    # ── 6. Walk-Forward CV
    log.info("\nPurged Walk-Forward CV başlıyor…")
    cv_df = purged_walk_forward_cv(
        df, {k: v for k, v in models.items() if k != "stack"},
        feat_cols, target_col="anomaly_logit",
        n_test_years=args.cv_folds,
    )
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)
    plot_cv_comparison(cv_df, fig_dir)

    # En iyi model
    if not cv_df.empty:
        best_name = cv_df.groupby("model")["rmse_pp"].mean().idxmin()
        log.info(f"\n★  EN İYİ MODEL: {best_name.upper()}  ★")
    else:
        best_name = "lgb_dart" if HAS_LGB else "etr"

    # ── 7. Konformal kalibrasyonu
    log.info("\nKonformal kalibrasyon…")
    hist_df = df[df["fill_pct"].notna()].dropna(subset=feat_cols + ["anomaly_logit"])
    cal_n   = max(24, len(hist_df) // 5)          # son %20 kalibrasyon seti
    train_df_cal = hist_df.iloc[:-cal_n]
    cal_df       = hist_df.iloc[-cal_n:]

    best_model = models[best_name]
    X_tr_cal = train_df_cal[feat_cols].values
    y_tr_cal = train_df_cal["anomaly_logit"].values
    best_model.fit(X_tr_cal, y_tr_cal)

    X_cal = cal_df[feat_cols].values
    y_cal = cal_df["anomaly_logit"].values
    pred_cal_logit = best_model.predict(X_cal)

    # Geri dönüşüm
    clim_cal = cal_df["date"].dt.month.map(climatology).values
    pred_cal_fill = np.clip(from_logit(pred_cal_logit) - 50.0 + clim_cal, 0, 100)
    y_cal_fill    = cal_df["fill_pct"].values

    conformal = SplitConformalPredictor(coverage=args.coverage)
    conformal.calibrate(y_cal_fill, pred_cal_fill)

    # ── 8. Projeksiyon — her model
    log.info("\nProjeksiyonlar hesaplanıyor…")
    all_outs = []

    # Final eğitim: tüm geçmiş veri
    X_fin = hist_df[feat_cols].values
    y_fin = hist_df["anomaly_logit"].values

    for name, model in models.items():
        log.info(f"\n── {name.upper()} ──")
        t0 = time.time()
        try:
            model.fit(X_fin, y_fin)
        except Exception as e:
            log.error(f"  Final eğitim hatası: {e}")
            continue

        # Q10/Q90 final eğitim
        q10_model.fit(X_fin, y_fin)
        q90_model.fit(X_fin, y_fin)
        log.info(f"  Eğitim: {time.time()-t0:.1f}s")

        # SHAP (sadece en iyi model)
        if name == best_name and not args.no_shap:
            run_shap_analysis(model, X_fin, feat_cols, fig_dir, name)

        out = simulate_v3(
            df, model, climatology, conformal,
            q10_model, q90_model,
            start_date, end_date, feat_cols,
        )
        out["model"] = name
        out.to_csv(out_dir / f"projection_{name}.csv", index=False)
        all_outs.append(out)

        cv_sum = None
        if not cv_df.empty:
            m_cv = cv_df[cv_df["model"] == name]
            if not m_cv.empty:
                cv_sum = m_cv[["rmse_pp","mape_pct","pearson_r"]].mean().to_dict()
        plot_full(out, name, last_obs, cv_sum, fig_dir, climatology)

    # ── 9. Ensemble median
    if len(all_outs) >= 2:
        log.info("\n── Ensemble median ──")
        pivot_sim = pd.concat(all_outs).pivot_table(
            index="date", columns="model", values="fill_sim")
        pivot_lo  = pd.concat(all_outs).pivot_table(
            index="date", columns="model", values="fill_lo")
        pivot_hi  = pd.concat(all_outs).pivot_table(
            index="date", columns="model", values="fill_hi")

        ens = all_outs[0][["date","fill_pct"]].copy().set_index("date")
        ens["fill_sim"] = pivot_sim.median(axis=1)
        ens["fill_lo"]  = pivot_lo.min(axis=1)
        ens["fill_hi"]  = pivot_hi.max(axis=1)
        ens = ens.reset_index()
        ens["model"] = "ensemble_median"
        ens.to_csv(out_dir / "projection_ensemble_median.csv", index=False)
        plot_full(ens, "ensemble_median", last_obs, None, fig_dir, climatology)
        all_outs.append(ens)

    # ── 10. Tüm modeller birleşik
    if all_outs:
        pd.concat(all_outs).to_csv(out_dir / "projection_all.csv", index=False)

    # ── Özet
    log.info("\n" + "=" * 65)
    log.info("TAMAMLANDI")
    log.info(f"Çıktılar → {out_dir}")
    if not cv_df.empty:
        best_cv = cv_df[cv_df["model"]==best_name][["rmse_pp","mape_pct","pearson_r"]].mean()
        log.info(f"En iyi model: {best_name.upper()}")
        log.info(f"  RMSE={best_cv['rmse_pp']:.2f}pp  "
                 f"MAPE={best_cv['mape_pct']:.1f}%  "
                 f"Pearson={best_cv['pearson_r']:.3f}")
    log.info(f"Konformal bant genişliği: ±{conformal.q_hat:.2f} pp ({args.coverage*100:.0f}% coverage)")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
