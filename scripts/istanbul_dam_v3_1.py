#!/usr/bin/env python3
"""
İstanbul Baraj Doluluk Tahmini — v3.1 (Tekrar Sorunu Giderildi)
================================================================
v3'ten farklar:
  ✓ Drifting (kayan) klimatoloji — 2040'a doğru sistematik düşüş
  ✓ AR(1) korelasyonlu gürültü — bootstrap tekrarı ortadan kalktı
  ✓ Tarihsel OLS + Mann-Kendall trend çıkarımı
  ✓ İklim değişikliği parametreleri (IPCC SSP2-4.5 İstanbul)
  ✓ Belirsizlik ufukla büyür (√h ölçekli AR(1))
  ✓ Trend vurgulu 4-panel grafik

Kurulum:
    pip install lightgbm xgboost optuna shap statsmodels scipy

Kullanım:
    python istanbul_dam_v3_1.py
    python istanbul_dam_v3_1.py --tune --n-trials 80
    python istanbul_dam_v3_1.py --temp-rise 0.055 --precip-drop 0.005
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

from scipy.special import logit, expit
from scipy import stats
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

try:
    import lightgbm as lgb; HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import xgboost as xgb; HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

try:
    import shap; HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from statsmodels.tsa.seasonal import STL; HAS_STL = True
except Exception:
    HAS_STL = False

warnings.filterwarnings("ignore")
ROOT = Path("/Users/yasinkaya/Hackhaton")
EPS  = 1e-4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("dam_v3_1")


# ══════════════════════════════════════════════════════════════════════════════
# § 1  Metrikler
# ══════════════════════════════════════════════════════════════════════════════

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    mask = yt > 1.0
    return float(mean_absolute_percentage_error(yt[mask], yp[mask]) * 100) \
           if mask.sum() else float("nan")

def pearson_r(y_true, y_pred):
    return float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) >= 3 else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# § 2  Logit dönüşümü
# ══════════════════════════════════════════════════════════════════════════════

def to_logit(x):
    return logit(np.clip(np.asarray(x, float) / 100.0, EPS, 1 - EPS))

def from_logit(z):
    return expit(np.asarray(z, float)) * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# § 3  Fizik katmanı
# ══════════════════════════════════════════════════════════════════════════════

def compute_api(rain: pd.Series, k: float = 0.92) -> pd.Series:
    api = np.zeros(len(rain)); r = rain.fillna(0).values
    for i in range(1, len(r)):
        api[i] = k * api[i-1] + r[i]
    return pd.Series(api, index=rain.index)

def compute_snow_proxy(t: pd.Series, rain: pd.Series,
                        t_acc=3.0, t_melt=5.0, melt_rate=15.0) -> pd.Series:
    snow = np.zeros(len(t))
    tv = t.ffill().fillna(0).values; rv = rain.fillna(0).values
    for i in range(1, len(tv)):
        acc  = rv[i] if tv[i] < t_acc else 0.0
        melt = max(0.0, (tv[i] - t_melt) * melt_rate) if tv[i] > t_melt else 0.0
        snow[i] = max(0.0, snow[i-1] + acc - melt)
    return pd.Series(snow, index=t.index)

def compute_bucket(rain: pd.Series, et0: pd.Series,
                    cap=200.0, demand=8.0) -> pd.Series:
    s = np.zeros(len(rain)); rv = rain.fillna(0).values; ev = et0.fillna(0).values
    s[0] = cap / 2
    for i in range(1, len(rv)):
        s[i] = float(np.clip(s[i-1] + rv[i] - ev[i] - demand, 0, cap))
    return pd.Series(s / cap * 100.0, index=rain.index)

def compute_stl(series: pd.Series, period=12) -> pd.DataFrame:
    n = len(series)
    blank = pd.DataFrame({"stl_trend": np.zeros(n), "stl_resid": np.zeros(n)},
                          index=series.index)
    if not HAS_STL or series.dropna().shape[0] < period * 2:
        return blank
    try:
        res = STL(series.fillna(series.median()), period=period, robust=True).fit()
        return pd.DataFrame({"stl_trend": res.trend.values,
                              "stl_resid": res.resid.values}, index=series.index)
    except Exception as e:
        log.warning(f"STL: {e}"); return blank

def spi(rain: pd.Series, w: int) -> pd.Series:
    rm = rain.rolling(w, min_periods=max(2, w//2)).mean()
    rs = rain.rolling(w, min_periods=max(2, w//2)).std().replace(0, np.nan)
    return ((rain - rm) / rs).fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# § 4  Klimatoloji — statik + kayan
# ══════════════════════════════════════════════════════════════════════════════

def compute_static_climatology(fill: pd.Series, months: pd.Series) -> pd.Series:
    valid = fill.dropna()
    return valid.groupby(months.loc[valid.index]).mean()   # idx 1..12

def fill_to_anomaly(fill: pd.Series, months: pd.Series, clim: pd.Series) -> pd.Series:
    return fill - months.map(clim)


# ─────────────────────────────────── TREND ───────────────────────────────────

def extract_trend(df: pd.DataFrame, fill_col="fill_pct") -> dict:
    """OLS + Mann-Kendall trend analizi (yıllık ortalamalar üzerinde)."""
    hist = df[df[fill_col].notna()].copy()
    hist["year"] = hist["date"].dt.year
    annual = hist.groupby("year")[fill_col].mean().dropna()

    if len(annual) < 5:
        return {"slope": 0.0, "se": 1.0, "p": 1.0, "annual": annual}

    x, y = annual.index.values.astype(float), annual.values
    slope, _, _, p, se = stats.linregress(x, y)

    # Mann-Kendall
    n = len(y); s = sum(np.sign(y[j] - y[i])
                        for i in range(n-1) for j in range(i+1, n))
    var_s = n*(n-1)*(2*n+5)/18
    z_mk  = (s - np.sign(s)) / np.sqrt(var_s) if var_s else 0
    p_mk  = 2*(1 - stats.norm.cdf(abs(z_mk)))

    log.info(f"Tarihsel trend: {slope:+.3f} pp/yıl  "
             f"p_OLS={p:.3f}  z_MK={z_mk:+.2f}  p_MK={p_mk:.3f}")
    log.info(f"  {annual.index.min()}–{annual.index.max()} toplam ≈ "
             f"{slope*(annual.index.max()-annual.index.min()):+.1f} pp")

    return {"slope": float(slope), "se": float(se), "p": float(p),
            "z_mk": float(z_mk), "p_mk": float(p_mk), "annual": annual}


def compute_drifting_climatology(
    static_clim: pd.Series,
    base_year:   int,
    trend:       dict,
    temp_rise_per_yr:   float = 0.040,   # °C/yıl (IPCC SSP2-4.5)
    precip_drop_per_yr: float = 0.003,   # oran (pozitif → azalır)
    temp_fill_sens:     float = -0.35,   # pp/°C
    damping:            float = 0.75,    # 0→sabit, 1→tam trend
    max_drift:          float = -35.0,   # 2040 için maksimum pp düşüş
) -> dict[int, pd.Series]:
    """
    Her gelecek yılı için ay bazında kayan klimatoloji.
    """
    tau = 20.0
    slope = trend.get("slope", 0.0)
    result = {}

    for yr in range(base_year, 2041):
        dt = float(yr - base_year)
        trend_c  = slope * dt * (1 - np.exp(-dt / tau)) * damping
        temp_c   = temp_fill_sens * temp_rise_per_yr * dt
        total_sh = float(np.clip(trend_c + temp_c, max_drift, 8.0))

        adj = {}
        for m in range(1, 13):
            base_v   = float(static_clim.get(m, 50.0))
            precip_c = -precip_drop_per_yr * dt * base_v * 0.4
            adj[m]   = float(np.clip(base_v + total_sh + precip_c, 2.0, 98.0))
        result[yr] = pd.Series(adj)

    return result


# ─────────────────────────────── AR(1) istatistik ────────────────────────────

def estimate_ar1(df: pd.DataFrame, model, feat_cols: list,
                  fill_col="fill_pct") -> dict:
    """Geçmiş residual'lardan AR(1) φ ve aylık σ tahmin et."""
    hist = df[df[fill_col].notna()].dropna(subset=feat_cols + [fill_col]).copy()
    if len(hist) < 24:
        return {"phi": 0.65, "sigma": {m: 3.0 for m in range(1, 13)}}
    try:
        model.fit(hist[feat_cols].values, hist[fill_col].values)
        resid = hist[fill_col].values - model.predict(hist[feat_cols].values)
    except Exception:
        return {"phi": 0.65, "sigma": {m: 3.0 for m in range(1, 13)}}

    phi = float(np.clip(np.corrcoef(resid[1:], resid[:-1])[0,1], 0, 0.95))
    months = hist["date"].dt.month.values
    sigma  = {m: float(np.std(resid[months == m])) if (months == m).sum() > 2
              else 3.0 for m in range(1, 13)}

    log.info(f"AR(1): φ={phi:.3f}  σ_ortalama={np.mean(list(sigma.values())):.2f} pp")
    return {"phi": phi, "sigma": sigma}


# ══════════════════════════════════════════════════════════════════════════════
# § 5  Özellik mühendisliği
# ══════════════════════════════════════════════════════════════════════════════

ALL_FEATS = [
    "rain_mm","et0_mm_month","t_mean_c","rh_mean_pct","pressure_kpa",
    "vpd_kpa_mean","climate_balance_mm",
    "api","snow_proxy_mm","bucket_pct",
    "stl_trend","stl_resid",
    "spi_3","spi_6","spi_12",
    "cum_balance_6","cum_balance_12",
    "rain_delta","et0_delta","balance_delta","t_delta","rh_delta",
    "rain_ma3","rain_ma6","rain_ma12",
    "et0_ma3","et0_ma6",
    "balance_ma3","balance_ma6","balance_ma12",
    "month_sin","month_cos","quarter_sin","quarter_cos",
    "lag1_fill","lag2_fill","lag3_fill","lag6_fill","lag12_fill",
    "fill_ma3","fill_ma6",
    "lag1_anomaly","lag3_anomaly","lag12_anomaly",
    "lag1_logit","lag12_logit",
]

def build_features(df: pd.DataFrame, clim: pd.Series | None = None) -> pd.DataFrame:
    d = df.copy().sort_values("date").reset_index(drop=True)

    # VPD
    es = 0.6108 * np.exp(17.27 * d["t_mean_c"] / (d["t_mean_c"] + 237.3))
    vpd_calc = es * (1 - d["rh_mean_pct"] / 100)
    if "vpd_kpa_mean" not in d.columns or d["vpd_kpa_mean"].isna().all():
        d["vpd_kpa_mean"] = vpd_calc
    else:
        d["vpd_kpa_mean"] = d["vpd_kpa_mean"].fillna(vpd_calc)

    d["climate_balance_mm"] = d["rain_mm"] - d["et0_mm_month"]
    d["month"]   = d["date"].dt.month
    d["quarter"] = d["date"].dt.quarter

    d["api"]           = compute_api(d["rain_mm"]).values
    d["snow_proxy_mm"] = compute_snow_proxy(d["t_mean_c"], d["rain_mm"]).values
    d["bucket_pct"]    = compute_bucket(d["rain_mm"], d["et0_mm_month"]).values

    for w, n in [(3,"spi_3"),(6,"spi_6"),(12,"spi_12")]:
        d[n] = spi(d["rain_mm"], w).values
    d["cum_balance_6"]  = d["climate_balance_mm"].rolling(6,  min_periods=1).sum()
    d["cum_balance_12"] = d["climate_balance_mm"].rolling(12, min_periods=1).sum()

    for src, dst in [("rain_mm","rain"),("et0_mm_month","et0"),
                      ("climate_balance_mm","balance"),
                      ("t_mean_c","t"),("rh_mean_pct","rh")]:
        d[f"{dst}_delta"] = d[src].diff(1).fillna(0)

    for w, n in [(3,"rain_ma3"),(6,"rain_ma6"),(12,"rain_ma12")]:
        d[n] = d["rain_mm"].rolling(w, min_periods=1).mean()
    for w, n in [(3,"et0_ma3"),(6,"et0_ma6")]:
        d[n] = d["et0_mm_month"].rolling(w, min_periods=1).mean()
    for w, n in [(3,"balance_ma3"),(6,"balance_ma6"),(12,"balance_ma12")]:
        d[n] = d["climate_balance_mm"].rolling(w, min_periods=1).mean()

    d["month_sin"]   = np.sin(2 * np.pi * d["month"]   / 12)
    d["month_cos"]   = np.cos(2 * np.pi * d["month"]   / 12)
    d["quarter_sin"] = np.sin(2 * np.pi * d["quarter"] / 4)
    d["quarter_cos"] = np.cos(2 * np.pi * d["quarter"] / 4)

    if "weighted_total_fill" in d.columns and "fill_pct" not in d.columns:
        d["fill_pct"] = d["weighted_total_fill"] * 100.0

    fp = d.get("fill_pct", pd.Series(np.nan, index=d.index))
    for lag, n in [(1,"lag1_fill"),(2,"lag2_fill"),(3,"lag3_fill"),
                    (6,"lag6_fill"),(12,"lag12_fill")]:
        d[n] = fp.shift(lag)
    d["fill_ma3"] = fp.shift(1).rolling(3, min_periods=1).mean()
    d["fill_ma6"] = fp.shift(1).rolling(6, min_periods=1).mean()

    if clim is not None:
        anom = fill_to_anomaly(fp, d["month"], clim)
        for lag, n in [(1,"lag1_anomaly"),(3,"lag3_anomaly"),(12,"lag12_anomaly")]:
            d[n] = anom.shift(lag)
        lgt = pd.Series(to_logit(fp.fillna(fp.median()).values), index=d.index)
        d["lag1_logit"]  = lgt.shift(1)
        d["lag12_logit"] = lgt.shift(12)
    else:
        for c in ["lag1_anomaly","lag3_anomaly","lag12_anomaly",
                  "lag1_logit","lag12_logit"]:
            d[c] = 0.0

    stl_f = compute_stl(fp if "fill_pct" in d.columns else pd.Series(np.nan))
    d["stl_trend"] = stl_f["stl_trend"].values
    d["stl_resid"]  = stl_f["stl_resid"].values

    return d


# ══════════════════════════════════════════════════════════════════════════════
# § 6  Veri yükleme
# ══════════════════════════════════════════════════════════════════════════════

def load_data(panel_path: Path, climate_path: Path | None = None) -> pd.DataFrame:
    log.info(f"Panel: {panel_path.name}")
    panel = pd.read_csv(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    if "weighted_total_fill" in panel.columns:
        panel["fill_pct"] = panel["weighted_total_fill"] * 100.0

    if climate_path and climate_path.exists():
        log.info(f"İklim proj.: {climate_path.name}")
        clim = pd.read_csv(climate_path)
        clim["date"] = pd.to_datetime(clim["date"])
        if "precip_mm_month" in clim.columns:
            clim = clim.rename(columns={"precip_mm_month": "rain_mm"})
        future = clim[clim["date"].dt.year > panel["date"].dt.year.max()]

        panel["_m"] = panel["date"].dt.month
        clim_means = {c: panel.groupby("_m")[c].mean()
                      for c in ["rain_mm","et0_mm_month","t_mean_c",
                                 "rh_mean_pct","pressure_kpa"] if c in panel.columns}

        full = pd.DataFrame({"date": pd.date_range(
                    panel["date"].min(), "2040-12-01", freq="MS")})
        full = full.merge(panel.drop(columns=["_m"], errors="ignore"),
                          on="date", how="left")
        fcols = [c for c in ["rain_mm","et0_mm_month","t_mean_c","rh_mean_pct"]
                 if c in future.columns]
        full = full.merge(future[["date"]+fcols], on="date", how="left",
                          suffixes=("","_c"))
        for c in fcols:
            cc = f"{c}_c"
            if cc in full.columns:
                full[c] = full[c].where(full[cc].isna(), full[cc])
                full.drop(columns=[cc], inplace=True)
        full["_m"] = full["date"].dt.month
        for c, monthly in clim_means.items():
            full[c] = full[c].fillna(full["_m"].map(monthly))
        full.drop(columns=["_m"], inplace=True)
        panel = full

    # fill_pct ve weighted_total_fill'i koruyarak diğer NaN'ları doldur
    num = panel.select_dtypes(include=np.number).columns
    excl = {"fill_pct", "weighted_total_fill"}
    fill_these = [c for c in num if c not in excl]
    panel[fill_these] = panel[fill_these].fillna(panel[fill_these].mean())
    panel = panel.sort_values("date").reset_index(drop=True)
    log.info(f"{len(panel)} satır | {panel['date'].min().date()} – "
             f"{panel['date'].max().date()}")
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# § 7  Model kataloğu
# ══════════════════════════════════════════════════════════════════════════════

def make_lgb_dart(p=None):
    d = dict(boosting_type="dart", n_estimators=600, learning_rate=0.05,
             num_leaves=31, max_depth=6, min_child_samples=10,
             subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1,
             reg_lambda=1.0, drop_rate=0.1, skip_drop=0.5,
             random_state=42, n_jobs=-1, verbose=-1)
    if p: d.update(p)
    return lgb.LGBMRegressor(**d)

def make_lgb_q(alpha):
    return lgb.LGBMRegressor(objective="quantile", alpha=alpha,
        boosting_type="gbdt", n_estimators=600, learning_rate=0.05,
        num_leaves=31, max_depth=6, subsample=0.85, colsample_bytree=0.85,
        random_state=42, n_jobs=-1, verbose=-1)

def make_xgb(p=None):
    d = dict(n_estimators=600, learning_rate=0.04, max_depth=5,
             subsample=0.85, colsample_bytree=0.85,
             reg_alpha=0.1, reg_lambda=1.5,
             random_state=42, n_jobs=-1, verbosity=0)
    if p: d.update(p)
    return xgb.XGBRegressor(**d)

def make_etr():
    return ExtraTreesRegressor(n_estimators=400, max_features=0.6,
                                min_samples_leaf=3, random_state=42, n_jobs=-1)

def build_catalog(lgb_p=None, xgb_p=None):
    m = {}
    if HAS_LGB: m["lgb_dart"] = make_lgb_dart(lgb_p); log.info("  LGB DART")
    if HAS_XGB: m["xgb"]      = make_xgb(xgb_p);      log.info("  XGBoost")
    m["etr"] = make_etr();                              log.info("  ETR")
    if len(m) >= 2:
        m["stack"] = StackingRegressor(
            estimators=list(m.items()),
            final_estimator=make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            cv=5, n_jobs=-1)
        log.info("  Stacking Ensemble")
    return m


# ══════════════════════════════════════════════════════════════════════════════
# § 8  Purged Walk-Forward CV
# ══════════════════════════════════════════════════════════════════════════════

def purged_cv(df, models, feat_cols, target_col="anomaly_logit",
              n_folds=5, embargo=6, min_train=60):
    log.info(f"Purged CV: {n_folds} fold, {embargo} ay embargo")
    hist = df[df["fill_pct"].notna()].dropna(subset=feat_cols+["fill_pct"])
    if target_col not in hist.columns:
        log.warning("Hedef sütun yok, CV atlanıyor.")
        return pd.DataFrame()

    years = sorted(hist["date"].dt.year.unique())[-n_folds:]
    rows  = []
    for yr in years:
        ts  = pd.Timestamp(f"{yr}-01-01")
        ee  = ts - pd.DateOffset(months=embargo)
        tr  = hist[hist["date"] < ee]
        te  = hist[hist["date"].dt.year == yr]
        if len(tr) < min_train or te.empty: continue

        for name, model in models.items():
            if name == "stack": continue
            try:
                model.fit(tr[feat_cols].values, tr[target_col].values)
                pred_l = model.predict(te[feat_cols].values)
                clim_t = te["clim_fill"].values if "clim_fill" in te.columns else 0
                pred_f = np.clip(from_logit(pred_l) + clim_t, 0, 100)
                rows.append({"model": name, "test_year": yr,
                              "rmse_pp": round(rmse(te["fill_pct"].values, pred_f), 3),
                              "mape_pct": round(mape(pd.Series(te["fill_pct"].values),
                                                     pd.Series(pred_f)), 3),
                              "pearson_r": round(pearson_r(te["fill_pct"].values,
                                                           pred_f), 3)})
                log.info(f"  {yr} │ {name:10s} │ RMSE={rows[-1]['rmse_pp']:.2f}")
            except Exception as e:
                log.error(f"  {yr} │ {name}: {e}")

    cv = pd.DataFrame(rows)
    if not cv.empty:
        log.info(f"\n{cv.groupby('model')[['rmse_pp','mape_pct','pearson_r']].mean().round(3).to_string()}\n")
    return cv


# ══════════════════════════════════════════════════════════════════════════════
# § 9  Split Conformal Prediction
# ══════════════════════════════════════════════════════════════════════════════

class SplitConformal:
    def __init__(self, coverage=0.90):
        self.coverage = coverage; self.q_hat = 0.0

    def calibrate(self, y_true, y_pred):
        r = np.abs(y_true - y_pred); alpha = 1 - self.coverage
        n = len(r)
        lv = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
        self.q_hat = float(np.quantile(r, lv))
        log.info(f"  Konformal ±{self.q_hat:.2f} pp ({self.coverage*100:.0f}%)")

    def interval(self, y_pred):
        return (np.clip(y_pred - self.q_hat, 0, 100),
                np.clip(y_pred + self.q_hat, 0, 100))


# ══════════════════════════════════════════════════════════════════════════════
# § 10  Horizon alpha
# ══════════════════════════════════════════════════════════════════════════════

def h_alpha(h, h1=24, h2=120, a1=0.95, a2=0.55):
    if h <= h1: return a1
    if h >= h2: return a2
    return a1 + (h - h1) / (h2 - h1) * (a2 - a1)


# ══════════════════════════════════════════════════════════════════════════════
# § 11  Projeksiyon (v3.1 — drifting clim + AR(1))
# ══════════════════════════════════════════════════════════════════════════════

def simulate(
    df:           pd.DataFrame,
    model,
    static_clim:  pd.Series,
    drift_clim:   dict[int, pd.Series],
    conformal:    SplitConformal,
    q10:          object,
    q90:          object,
    start:        pd.Timestamp,
    end:          pd.Timestamp,
    feat_cols:    list,
    ar1:          dict,                  # {"phi": float, "sigma": dict}
    use_logit:    bool = True,
    use_anomaly:  bool = True,
    seed:         int  = 42,
) -> pd.DataFrame:
    """
    Bilimsel projeksiyon:
    ├─ Her yıl model yeniden eğitilir
    ├─ Drifting klimatoloji taban çizgisi
    ├─ AR(1) korelasyonlu gürültü (tekrar yok)
    └─ Horizon-aware blending (yakın→ML, uzak→klimatoloji)
    """
    data  = df.copy().sort_values("date").reset_index(drop=True)
    d2i   = {d: i for i, d in enumerate(data["date"])}
    si, ei = d2i.get(start), d2i.get(end)
    if si is None or ei is None:
        log.error("start/end date bulunamadı"); return data

    buf = data["fill_pct"].values.copy().astype(float)
    sim = np.full(len(data), np.nan)
    lo  = np.full(len(data), np.nan)
    hi  = np.full(len(data), np.nan)

    for i in range(si):
        v = buf[i]; sim[i] = lo[i] = hi[i] = v

    def blag(i, lag):
        p = i - lag
        if p < 0: return float(np.nanmean(buf[:max(1,i)]))
        v = buf[p]
        if np.isnan(v):
            for k in range(p-1, max(-1,p-24), -1):
                if k >= 0 and not np.isnan(buf[k]): return float(buf[k])
            return float(np.nanmean(buf[:max(1,i)]))
        return float(v)

    def bma(i, w):
        v = [buf[i-k] for k in range(1,w+1) if i-k>=0 and not np.isnan(buf[i-k])]
        return float(np.mean(v)) if v else 50.0

    rng       = np.random.default_rng(seed)
    phi       = ar1.get("phi", 0.65)
    sig_mo    = ar1.get("sigma", {m: 3.0 for m in range(1,13)})
    ar1_state = 0.0
    h_total   = 0

    for year in range(start.year, end.year + 1):
        yr_clim = drift_clim.get(year,
                    drift_clim.get(max(drift_clim.keys())))

        train_end = pd.Timestamp(f"{year-1}-12-01")
        tr = data[(data["date"] <= train_end) &
                   data["fill_pct"].notna()].dropna(subset=feat_cols+["fill_pct"])
        if tr.empty: continue

        Xtr = tr[feat_cols].values

        if use_anomaly and use_logit:
            anom = fill_to_anomaly(tr["fill_pct"], tr["date"].dt.month, static_clim)
            ytr  = to_logit(np.clip(anom + 50.0, EPS*100, (1-EPS)*100))
        elif use_logit:
            ytr = to_logit(tr["fill_pct"].values)
        else:
            ytr = tr["fill_pct"].values

        try:
            model.fit(Xtr, ytr); q10.fit(Xtr, ytr); q90.fit(Xtr, ytr)
        except Exception as e:
            log.error(f"  {year} eğitim: {e}"); continue

        mask = ((data["date"] >= pd.Timestamp(f"{year}-01-01")) &
                (data["date"] <= min(pd.Timestamp(f"{year}-12-01"), end)) &
                (data["date"] >= start))
        idxs = data.index[mask].tolist()

        for i in idxs:
            h_total += 1
            mo = int(data.loc[i, "month"]) if "month" in data.columns \
                 else int(data.loc[i, "date"].month)

            row = data.loc[i].copy()
            for lag, c in [(1,"lag1_fill"),(2,"lag2_fill"),(3,"lag3_fill"),
                            (6,"lag6_fill"),(12,"lag12_fill")]:
                row[c] = blag(i, lag)
            row["fill_ma3"] = bma(i, 3); row["fill_ma6"] = bma(i, 6)

            for lag, c in [(1,"lag1_anomaly"),(3,"lag3_anomaly"),(12,"lag12_anomaly")]:
                lf  = blag(i, lag)
                lmo = int(data.loc[i-lag,"month"]) if i-lag >= 0 else mo
                row[c] = lf - float(static_clim.get(lmo, 50.0))

            row["lag1_logit"]  = float(to_logit(np.clip(blag(i,  1), 1, 99)))
            row["lag12_logit"] = float(to_logit(np.clip(blag(i, 12), 1, 99)))
            row["stl_trend"]   = float(np.nanmean([blag(i,k) for k in range(1,13)]))
            row["stl_resid"]   = blag(i,1) - row["stl_trend"]

            for fc in feat_cols:
                if pd.isna(row.get(fc, np.nan)):
                    row[fc] = float(Xtr[:, feat_cols.index(fc)].mean())

            Xp = row[feat_cols].values.reshape(1,-1)

            try:
                pr  = float(model.predict(Xp)[0])
                pq10= float(q10.predict(Xp)[0])
                pq90= float(q90.predict(Xp)[0])
            except Exception:
                pr = pq10 = pq90 = (to_logit(50.0) if use_logit else 50.0)

            # ── Geri dönüşüm (statik klimatolojiye göre)
            sc = float(static_clim.get(mo, 50.0))
            dc = float(yr_clim.get(mo, sc))         # kayan taban

            if use_anomaly and use_logit:
                fml   = float(np.clip(from_logit(pr)   - 50.0 + sc, 0, 100))
                fq10r = float(np.clip(from_logit(pq10) - 50.0 + sc, 0, 100))
                fq90r = float(np.clip(from_logit(pq90) - 50.0 + sc, 0, 100))
            elif use_logit:
                fml   = float(np.clip(from_logit(pr),   0, 100))
                fq10r = float(np.clip(from_logit(pq10), 0, 100))
                fq90r = float(np.clip(from_logit(pq90), 0, 100))
            else:
                fml   = float(np.clip(pr,   0, 100))
                fq10r = float(np.clip(pq10, 0, 100))
                fq90r = float(np.clip(pq90, 0, 100))

            # ── Horizon-aware blending + klim kayması
            alpha    = h_alpha(h_total)
            clim_sh  = dc - sc                          # kayan taban farkı (negatif)
            blended  = alpha * (fml + clim_sh) + (1-alpha) * dc

            # ── AR(1) gürültü (tekrar eden bootstrap değil)
            sigma     = sig_mo.get(mo, 3.0)
            ar1_state = phi * ar1_state + sigma * float(rng.normal(0,1))
            # Belirsizlik ufukla √h orantılı büyür (ama sınırlı)
            ns = float(np.clip(np.sqrt(h_total / 12.0) * 0.40, 0, 1.8))
            blended   = float(np.clip(blended + ar1_state * ns, 0, 100))

            # ── Belirsizlik bantları
            lc, hc = conformal.interval(np.array([blended]))
            flo = float(np.clip(min(lc[0], fq10r), 0, 100))
            fhi = float(np.clip(max(hc[0], fq90r), 0, 100))

            sim[i] = blended; lo[i] = flo; hi[i] = fhi; buf[i] = blended

    out = data[["date"]].copy()
    out["fill_pct"] = data.get("fill_pct", pd.Series(np.nan, index=data.index))
    out["fill_sim"] = sim; out["fill_lo"] = lo; out["fill_hi"] = hi
    return out


# ══════════════════════════════════════════════════════════════════════════════
# § 12  Optuna
# ══════════════════════════════════════════════════════════════════════════════

def optuna_tune(X, y, n_trials=60, mtype="lgb"):
    if not HAS_OPTUNA: return {}
    log.info(f"Optuna: {n_trials} deneme ({mtype})")
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    def obj(trial):
        if mtype == "lgb" and HAS_LGB:
            m = lgb.LGBMRegressor(
                boosting_type="dart",
                n_estimators   = trial.suggest_int("n_estimators", 200, 800),
                learning_rate  = trial.suggest_float("lr", 0.01, 0.1, log=True),
                num_leaves     = trial.suggest_int("leaves", 15, 63),
                max_depth      = trial.suggest_int("depth", 3, 8),
                min_child_samples = trial.suggest_int("mcs", 5, 30),
                subsample      = trial.suggest_float("sub", 0.6, 1.0),
                colsample_bytree = trial.suggest_float("col", 0.6, 1.0),
                reg_alpha      = trial.suggest_float("ra", 1e-3, 10, log=True),
                reg_lambda     = trial.suggest_float("rl", 1e-3, 10, log=True),
                drop_rate      = trial.suggest_float("dr", 0.05, 0.3),
                random_state=42, n_jobs=-1, verbose=-1)
        elif mtype == "xgb" and HAS_XGB:
            m = xgb.XGBRegressor(
                n_estimators   = trial.suggest_int("n_estimators", 200, 800),
                learning_rate  = trial.suggest_float("lr", 0.01, 0.1, log=True),
                max_depth      = trial.suggest_int("depth", 3, 8),
                subsample      = trial.suggest_float("sub", 0.6, 1.0),
                colsample_bytree = trial.suggest_float("col", 0.6, 1.0),
                reg_alpha      = trial.suggest_float("ra", 1e-3, 10, log=True),
                reg_lambda     = trial.suggest_float("rl", 1e-3, 10, log=True),
                random_state=42, n_jobs=-1, verbosity=0)
        else: return 999.0
        sc = []
        for tri, vli in tscv.split(X):
            m.fit(X[tri], y[tri]); sc.append(rmse(y[vli], m.predict(X[vli])))
        return float(np.mean(sc))

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    log.info(f"Optuna: best CV RMSE = {study.best_value:.4f}")
    return study.best_params


# ══════════════════════════════════════════════════════════════════════════════
# § 13  SHAP
# ══════════════════════════════════════════════════════════════════════════════

def run_shap(model, X, feat_cols, out_dir, tag):
    if not HAS_SHAP: return
    log.info(f"SHAP ({tag})…")
    try:
        try:   exp = shap.TreeExplainer(model); sv = exp.shap_values(X)
        except:exp = shap.LinearExplainer(model, X); sv = exp.shap_values(X)
        imp = pd.Series(np.abs(sv).mean(0), index=feat_cols).sort_values()
        top = imp.tail(20)
        fig, ax = plt.subplots(figsize=(9,6))
        colors = ["#185FA5" if v > imp.median() else "#85B7EB" for v in top.values]
        ax.barh(top.index, top.values, color=colors, height=0.7)
        ax.set_xlabel("Ort. |SHAP| (logit-anomali uzayı)"); ax.set_title(f"SHAP — {tag}")
        ax.axvline(top.values.mean(), color="#888", lw=0.7, ls="--")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); fig.savefig(out_dir/f"shap_{tag}.png", dpi=150); plt.close(fig)
        log.info(f"  Top 5: {list(imp.tail(5)[::-1].index)}")
    except Exception as e:
        log.warning(f"  SHAP başarısız: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# § 14  Görselleştirme — 4 panel, trend vurgulu
# ══════════════════════════════════════════════════════════════════════════════

def plot_full(out, drift_clim, last_obs, hist_trend,
              model_name, cv_sum, fig_dir, static_clim):
    hm = out["date"] <= last_obs
    fm = out["date"] >  last_obs

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 4, height_ratios=[2.5, 1],
                             hspace=0.42, wspace=0.32)
    ax1 = fig.add_subplot(gs[0,:])    # ana projeksiyon
    ax2 = fig.add_subplot(gs[1, 0])   # kayan klimatoloji
    ax3 = fig.add_subplot(gs[1,1:3])  # yıllık ort. + OLS trend
    ax4 = fig.add_subplot(gs[1, 3])   # belirsizlik büyümesi

    # ── Ana
    ax1.plot(out["date"][hm], out["fill_pct"][hm], "#111", lw=1.3,
             label="Gözlenen", zorder=4)
    ax1.plot(out["date"][fm], out["fill_sim"][fm], "#D55E00", lw=2.0,
             label=f"Tahmin — {model_name}", zorder=5)
    ax1.fill_between(out["date"][fm],
                      out["fill_lo"][fm].clip(0,100),
                      out["fill_hi"][fm].clip(0,100),
                      color="#D55E00", alpha=0.13, lw=0, label="Konformal %90")

    # Kayan klimatoloji orta çizgisi (Temmuz — en düşük sezon)
    drift_yrs = [y for y in sorted(drift_clim) if y >= last_obs.year]
    ax1.plot([pd.Timestamp(f"{y}-07-01") for y in drift_yrs],
             [drift_clim[y].get(7, 50.0) for y in drift_yrs],
             color="#0F6E56", lw=1.1, ls="--", alpha=0.65,
             label="Kayan klimatoloji (Tem.)")

    ax1.axvline(last_obs, color="#888", lw=0.8, ls="--", alpha=0.7)
    ax1.axhline(30, color="#E24B4A", lw=0.7, ls=":", alpha=0.5)
    ax1.text(out["date"].iloc[0], 31.5, "Kritik eşik (%30)",
             fontsize=8, color="#E24B4A", alpha=0.7)
    ax1.set_ylim(0, 108); ax1.set_ylabel("Doluluk (%)", fontsize=10)
    sl = hist_trend.get("slope", 0)
    ax1.set_title(f"İstanbul Baraj — {model_name.upper()}  │  "
                   f"Tarihsel trend: {sl:+.3f} pp/yıl  │  "
                   f"Kayan klim. + AR(1) gürültü + Konformal bant",
                   fontsize=10.5)
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.grid(True, color="#EEE", lw=0.5)
    ax1.spines[["top","right"]].set_visible(False)
    if cv_sum:
        ax1.text(0.01, 0.02,
                 f"CV: RMSE={cv_sum.get('rmse_pp','?'):.2f}pp  "
                 f"MAPE={cv_sum.get('mape_pct','?'):.1f}%  "
                 f"Pearson={cv_sum.get('pearson_r','?'):.3f}",
                 transform=ax1.transAxes, fontsize=8.5, color="#555",
                 bbox=dict(boxstyle="round,pad=0.3",fc="#F9F9F9",ec="#CCC",alpha=0.9))

    # ── Kayan klimatoloji (3 ay)
    for mo, col, lbl in [(3,"#185FA5","Mart"),(7,"#E24B4A","Temmuz"),(11,"#0F6E56","Kasım")]:
        ax2.plot(drift_yrs, [drift_clim[y].get(mo, 50) for y in drift_yrs],
                 color=col, lw=1.3, label=lbl, marker="o", markersize=2.5)
    ax2.set_title("Kayan klimatoloji / ay", fontsize=9)
    ax2.set_ylabel("Taban doluluk (%)")
    ax2.legend(fontsize=7); ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(True, color="#EEE", lw=0.5)

    # ── Yıllık ortalama + OLS trend
    def ymean(g):
        return g["fill_sim"].mean() if (g["date"]>last_obs).any() else g["fill_pct"].mean()
    yr = out.groupby(out["date"].dt.year).apply(ymean).dropna()
    cols_yr = ["#185FA5" if y <= last_obs.year else "#D55E00" for y in yr.index]
    ax3.bar(yr.index, yr.values, color=cols_yr, alpha=0.8, width=0.8)
    xf = yr.index.values.astype(float); yf = yr.values
    if len(xf) > 3:
        ms, bs, *_ = stats.linregress(xf, yf)
        ax3.plot(xf, ms*xf+bs, "#333", lw=1.1, ls="--", alpha=0.7,
                 label=f"OLS trend {ms:+.2f} pp/yıl")
        ax3.legend(fontsize=7)
    ax3.axhline(30, color="#E24B4A", lw=0.7, ls=":", alpha=0.5)
    ax3.set_title("Yıllık ort. doluluk + trend", fontsize=9)
    ax3.set_ylim(0, 100); ax3.set_ylabel("Doluluk (%)")
    ax3.spines[["top","right"]].set_visible(False)
    ax3.grid(True, axis="y", color="#EEE", lw=0.5)
    ax3.legend(handles=[
        Patch(color="#185FA5", alpha=0.8, label="Gözlenen"),
        Patch(color="#D55E00", alpha=0.8, label="Tahmin"),
    ] + (ax3.get_legend_handles_labels()[0][-1:] if ax3.get_legend() else []),
    fontsize=7)

    # ── Belirsizlik genişliği
    fut = out[fm].copy(); fut["unc"] = (fut["fill_hi"]-fut["fill_lo"]).clip(0,100)
    ax4.plot(range(1, len(fut)+1), fut["unc"].values, "#185FA5", lw=1.2)
    ax4.set_xlabel("Ufuk (ay)"); ax4.set_ylabel("Bant genişliği (pp)")
    ax4.set_title("Belirsizlik büyümesi", fontsize=9)
    ax4.spines[["top","right"]].set_visible(False)
    ax4.grid(True, color="#EEE", lw=0.5)

    plt.savefig(fig_dir/f"{model_name}_v31_projection.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Grafik: {model_name}_v31_projection.png")


def plot_cv_bar(cv_df, fig_dir):
    if cv_df.empty: return
    s = cv_df.groupby("model")["rmse_pp"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8,4))
    colors  = ["#185FA5" if i==0 else "#85B7EB" for i in range(len(s))]
    bars    = ax.barh(s.index, s.values, color=colors, alpha=0.9)
    ax.bar_label(bars, fmt="%.2f pp", padding=4, fontsize=9)
    ax.set_xlabel("Ort. CV RMSE (pp)"); ax.set_title("Model Karşılaştırması — Purged WF-CV")
    ax.spines[["top","right"]].set_visible(False); ax.grid(True, axis="x", color="#EEE", lw=0.5)
    plt.tight_layout(); fig.savefig(fig_dir/"model_cv.png", dpi=150); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# § 15  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="İstanbul Baraj v3.1")
    p.add_argument("--panel",   default="output/newdata_feature_store/tables/"
                                        "istanbul_dam_driver_panel_2000_2026_extended.csv")
    p.add_argument("--climate", default="output/scientific_climate_projection_2026_2040/"
                                        "climate_projection_2010_2040_monthly.csv")
    p.add_argument("--out",     default="output/istanbul_v3_1")
    p.add_argument("--tune",    action="store_true")
    p.add_argument("--n-trials",type=int, default=60)
    p.add_argument("--end-date",default="2040-12-01")
    p.add_argument("--cv-folds",type=int, default=5)
    p.add_argument("--coverage",type=float, default=0.90)
    p.add_argument("--no-shap", action="store_true")
    # İklim parametreleri (hassas ayar)
    p.add_argument("--temp-rise",   type=float, default=0.040,
                   help="°C/yıl sıcaklık artışı (IPCC SSP2-4.5 varsayılan: 0.040)")
    p.add_argument("--precip-drop", type=float, default=0.003,
                   help="oran/yıl yağış azalması (İstanbul: 0.003)")
    p.add_argument("--trend-damping", type=float, default=0.75,
                   help="Trend extrapolasyon dampingi (0–1)")
    args = p.parse_args()

    out_dir = ROOT / args.out; fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(exist_ok=True)

    fh = logging.FileHandler(out_dir/"run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-8s │ %(message)s"))
    log.addHandler(fh)

    log.info("="*70)
    log.info("İstanbul Baraj v3.1 — Drifting klim + AR(1) gürültü")
    log.info(f"  LGB:{HAS_LGB}  XGB:{HAS_XGB}  Optuna:{HAS_OPTUNA}  "
             f"SHAP:{HAS_SHAP}  STL:{HAS_STL}")
    log.info(f"  temp_rise={args.temp_rise} °C/yr  "
             f"precip_drop={args.precip_drop}/yr  "
             f"damping={args.trend_damping}")
    log.info("="*70)

    # ── 1. Veri
    cp  = ROOT / args.climate
    raw = load_data(ROOT / args.panel, cp if cp.exists() else None)
    last_obs   = raw[raw["fill_pct"].notna()]["date"].max()
    start_date = (pd.Timestamp(last_obs) + pd.DateOffset(months=1)).normalize()
    end_date   = pd.Timestamp(args.end_date)
    log.info(f"Son gözlem: {last_obs.date()} | "
             f"Projeksiyon: {start_date.date()} → {end_date.date()}")

    # ── 2. Statik klimatoloji
    hist_raw    = raw[raw["fill_pct"].notna()].copy()
    static_clim = compute_static_climatology(hist_raw["fill_pct"],
                                              hist_raw["date"].dt.month)
    log.info(f"Statik klimatoloji: {dict(static_clim.round(1))}")
    raw["clim_fill"] = raw["date"].dt.month.map(static_clim)

    # ── 3. Tarihsel trend + kayan klimatoloji
    hist_trend  = extract_trend(raw)
    drift_clim  = compute_drifting_climatology(
        static_clim, last_obs.year, hist_trend,
        temp_rise_per_yr   = args.temp_rise,
        precip_drop_per_yr = args.precip_drop,
        damping            = args.trend_damping,
    )
    log.info(f"Kayan klim. Temmuz: "
             f"2025={drift_clim.get(2025,{}).get(7,0):.1f}  "
             f"2030={drift_clim.get(2030,{}).get(7,0):.1f}  "
             f"2040={drift_clim.get(2040,{}).get(7,0):.1f}")

    # ── 4. Özellik mühendisliği
    log.info("Özellik mühendisliği…")
    df = build_features(raw, static_clim)
    df["anomaly"] = fill_to_anomaly(df["fill_pct"], df["month"], static_clim)
    df["anomaly_logit"] = np.where(
        df["fill_pct"].notna(),
        to_logit(np.clip(df["anomaly"] + 50.0, EPS*100, (1-EPS)*100)),
        np.nan)

    feat_cols = [f for f in ALL_FEATS if f in df.columns]
    log.info(f"Özellik sayısı: {len(feat_cols)}")

    # ── 5. Optuna
    lgb_p = xgb_p = {}
    if args.tune:
        hdf = df[df["fill_pct"].notna()].dropna(subset=feat_cols+["anomaly_logit"])
        Xa, ya = hdf[feat_cols].values, hdf["anomaly_logit"].values
        if HAS_LGB: lgb_p = optuna_tune(Xa, ya, args.n_trials, "lgb")
        if HAS_XGB: xgb_p = optuna_tune(Xa, ya, args.n_trials, "xgb")

    # ── 6. Modeller
    log.info("Modeller…")
    models = build_catalog(lgb_p, xgb_p)
    if HAS_LGB:
        q10_m = make_lgb_q(0.10); q90_m = make_lgb_q(0.90)
    else:
        q10_m = make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.10, solver="highs"))
        q90_m = make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.90, solver="highs"))

    # ── 7. Walk-Forward CV
    log.info("\nPurged WF-CV…")
    cv_df = purged_cv(df, {k:v for k,v in models.items() if k!="stack"},
                       feat_cols, n_folds=args.cv_folds)
    cv_df.to_csv(out_dir/"cv_results.csv", index=False)
    plot_cv_bar(cv_df, fig_dir)

    best_name = (cv_df.groupby("model")["rmse_pp"].mean().idxmin()
                 if not cv_df.empty else ("lgb_dart" if HAS_LGB else "etr"))
    log.info(f"\n★  EN İYİ MODEL: {best_name.upper()}  ★")

    # ── 8. AR(1) istatistikleri
    log.info("\nAR(1) parametreleri…")
    hdf   = df[df["fill_pct"].notna()].dropna(subset=feat_cols+["anomaly_logit"])
    ar1_p = estimate_ar1(df, models[best_name], feat_cols)

    # ── 9. Konformal kalibrasyon
    log.info("\nKonformal kalibrasyon…")
    cal_n  = max(24, len(hdf) // 5)
    tr_cal, cal_df = hdf.iloc[:-cal_n], hdf.iloc[-cal_n:]
    bm = models[best_name]
    bm.fit(tr_cal[feat_cols].values, tr_cal["anomaly_logit"].values)
    pl = bm.predict(cal_df[feat_cols].values)
    clim_c  = cal_df["date"].dt.month.map(static_clim).values
    pred_cf = np.clip(from_logit(pl) - 50.0 + clim_c, 0, 100)
    conf    = SplitConformal(args.coverage)
    conf.calibrate(cal_df["fill_pct"].values, pred_cf)

    # ── 10. Projeksiyon
    log.info("\nProjeksiyonlar…")
    Xfin = hdf[feat_cols].values; yfin = hdf["anomaly_logit"].values
    all_outs = []

    for name, model in models.items():
        log.info(f"\n── {name.upper()} ──")
        t0 = time.time()
        try:   model.fit(Xfin, yfin)
        except Exception as e: log.error(f"  Eğitim: {e}"); continue
        q10_m.fit(Xfin, yfin); q90_m.fit(Xfin, yfin)
        log.info(f"  Eğitim: {time.time()-t0:.1f}s")

        if name == best_name and not args.no_shap:
            run_shap(model, Xfin, feat_cols, fig_dir, name)

        out = simulate(df, model, static_clim, drift_clim,
                       conf, q10_m, q90_m,
                       start_date, end_date, feat_cols, ar1_p)
        out["model"] = name
        out.to_csv(out_dir/f"projection_{name}.csv", index=False)
        all_outs.append(out)

        cv_s = None
        if not cv_df.empty:
            m_cv = cv_df[cv_df["model"]==name]
            if not m_cv.empty:
                cv_s = m_cv[["rmse_pp","mape_pct","pearson_r"]].mean().to_dict()
        plot_full(out, drift_clim, last_obs, hist_trend, name, cv_s, fig_dir, static_clim)

    # ── 11. Ensemble median
    if len(all_outs) >= 2:
        log.info("\n── Ensemble median ──")
        ps = pd.concat(all_outs).pivot_table(index="date",columns="model",values="fill_sim")
        pl = pd.concat(all_outs).pivot_table(index="date",columns="model",values="fill_lo")
        ph = pd.concat(all_outs).pivot_table(index="date",columns="model",values="fill_hi")
        ens = all_outs[0][["date","fill_pct"]].set_index("date")
        ens["fill_sim"] = ps.median(axis=1)
        ens["fill_lo"]  = pl.min(axis=1)
        ens["fill_hi"]  = ph.max(axis=1)
        ens = ens.reset_index(); ens["model"] = "ensemble_median"
        ens.to_csv(out_dir/"projection_ensemble.csv", index=False)
        plot_full(ens, drift_clim, last_obs, hist_trend,
                  "ensemble_median", None, fig_dir, static_clim)
        all_outs.append(ens)

    if all_outs:
        pd.concat(all_outs).to_csv(out_dir/"projection_all.csv", index=False)

    # ── Özet
    log.info("\n"+"="*70)
    log.info("TAMAMLANDI"); log.info(f"Çıktılar → {out_dir}")
    if not cv_df.empty:
        bc = cv_df[cv_df["model"]==best_name][["rmse_pp","mape_pct","pearson_r"]].mean()
        log.info(f"En iyi: {best_name.upper()}  "
                 f"RMSE={bc['rmse_pp']:.2f}pp  MAPE={bc['mape_pct']:.1f}%  "
                 f"r={bc['pearson_r']:.3f}")
    log.info(f"Konformal: ±{conf.q_hat:.2f} pp ({args.coverage*100:.0f}%)")
    if all_outs:
        ens_df = all_outs[-1] if all_outs[-1]["model"].iloc[0] == "ensemble_median" \
                 else all_outs[0]
        last_sim = ens_df.dropna(subset=["fill_sim"])
        if not last_sim.empty:
            v2040 = last_sim.tail(12)["fill_sim"].mean()
            log.info(f"2040 yıllık ort. tahmin: {v2040:.1f}%  "
                     f"(statik klim. ort.: {static_clim.mean():.1f}%)")
    log.info("="*70)


if __name__ == "__main__":
    main()
