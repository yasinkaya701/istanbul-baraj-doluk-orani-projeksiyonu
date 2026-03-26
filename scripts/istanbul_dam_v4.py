#!/usr/bin/env python3
"""
İstanbul Baraj Doluluk Tahmini — v4 (Su Dengesi + Δfill)
================================================================
Bu sürüm: Δfill hedefi + fiziksel su dengesi + gerçek tüketim.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
import re

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
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING); HAS_OPTUNA = True
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
try:
    import requests; HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

warnings.filterwarnings("ignore")
ROOT = Path("/Users/yasinkaya/Hackhaton")
EPS  = 1e-4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("dam_v4")


def rmse(yt, yp): return float(np.sqrt(mean_squared_error(yt, yp)))
def mape(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp); m = yt > 1
    return float(mean_absolute_percentage_error(yt[m], yp[m])*100) if m.sum() else float("nan")
def pearson_r(yt, yp):
    return float(np.corrcoef(yt, yp)[0,1]) if len(yt) >= 3 else float("nan")


def to_logit(x): return logit(np.clip(np.asarray(x, float) / 100.0, EPS, 1 - EPS))
def from_logit(z): return expit(np.asarray(z, float)) * 100.0


def load_ibb_fill(path_csv: Path, path_xlsx: Path) -> pd.DataFrame:
    df = None
    if path_csv.exists():
        try:
            raw = pd.read_csv(path_csv, sep=None, engine="python", encoding="utf-8-sig")
            date_col = next((c for c in raw.columns
                             if any(k in c.lower() for k in ["tarih","date","yıl"])), None)
            fill_col = next((c for c in raw.columns
                             if any(k in c.lower() for k in ["doluluk","dolu","fill","oran","%"])), None)
            if date_col and fill_col:
                raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
                raw[fill_col] = pd.to_numeric(raw[fill_col], errors="coerce")
                raw = raw.dropna(subset=[date_col, fill_col])
                if raw[fill_col].median() < 2:
                    raw[fill_col] = raw[fill_col] * 100
                raw = raw.rename(columns={date_col: "date", fill_col: "fill_pct"})
                raw["date"] = raw["date"].dt.to_period("M").dt.to_timestamp()
                df = raw.groupby("date")["fill_pct"].mean().reset_index()
                log.info(f"  IBB fill CSV: {len(df)} aylık kayıt")
        except Exception as e:
            log.warning(f"  IBB fill CSV: {e}")
    if df is None and path_xlsx.exists():
        try:
            xdf = pd.read_excel(path_xlsx, engine="openpyxl")
            date_col = next((c for c in xdf.columns
                             if any(k in str(c).lower() for k in ["tarih","date"])), None)
            fill_col = next((c for c in xdf.columns
                             if any(k in str(c).lower() for k in ["doluluk","dolu","fill","oran","%"])), None)
            if date_col and fill_col:
                xdf[date_col] = pd.to_datetime(xdf[date_col], dayfirst=True, errors="coerce")
                xdf[fill_col] = pd.to_numeric(xdf[fill_col], errors="coerce")
                xdf = xdf.dropna(subset=[date_col, fill_col])
                if xdf[fill_col].median() < 2:
                    xdf[fill_col] = xdf[fill_col] * 100
                xdf = xdf.rename(columns={date_col: "date", fill_col: "fill_pct"})
                xdf["date"] = xdf["date"].dt.to_period("M").dt.to_timestamp()
                df = xdf.groupby("date")["fill_pct"].mean().reset_index()
                log.info(f"  IBB fill XLSX: {len(df)} aylık kayıt")
        except Exception as e:
            log.warning(f"  IBB fill XLSX: {e}")
    return df if df is not None else pd.DataFrame()


def load_ibb_consumption(path_xlsx: Path) -> pd.DataFrame:
    if not path_xlsx.exists():
        log.warning(f"  Tüketim dosyası yok: {path_xlsx.name}")
        return pd.DataFrame()
    try:
        raw = pd.read_excel(path_xlsx, engine="openpyxl")
        date_col = next((c for c in raw.columns
                         if any(k in str(c).lower() for k in ["tarih","date"])), None)
        cons_col = next((c for c in raw.columns
                         if "günlük" in str(c).lower() and "tüketim" in str(c).lower()), None)
        if not cons_col:
            cons_col = next((c for c in raw.columns
                             if any(k in str(c).lower() for k in ["tüketim","tuketim","icme","içme","su"])), None)
        if not (date_col and cons_col):
            log.warning("  Tüketim: uygun kolon bulunamadı")
            return pd.DataFrame()
        raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
        raw[cons_col] = pd.to_numeric(raw[cons_col], errors="coerce")
        raw = raw.dropna(subset=[date_col, cons_col])
        raw = raw.rename(columns={date_col:"date", cons_col:"consumption_m3"})
        raw["date"] = raw["date"].dt.to_period("M").dt.to_timestamp()
        out = raw.groupby("date")["consumption_m3"].sum().reset_index()
        log.info(f"  Tüketim: {len(out)} aylık kayıt")
        return out
    except Exception as e:
        log.warning(f"  Tüketim XLSX: {e}")
        return pd.DataFrame()


def load_ibb_catchment_rain(path_xlsx: Path) -> pd.DataFrame:
    if not path_xlsx.exists():
        log.warning(f"  Havza yağış dosyası yok: {path_xlsx.name}")
        return pd.DataFrame()
    try:
        raw = pd.read_excel(path_xlsx, engine="openpyxl")
        date_col = next((c for c in raw.columns
                         if any(k in str(c).lower() for k in ["tarih","date","yıl"])), None)
        rain_col = next((c for c in raw.columns
                         if any(k in str(c).lower() for k in ["yağış","yagis","rain","precip","mm"])), None)
        if not (date_col and rain_col):
            num_cols = raw.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                rain_col = num_cols[0]
        if not date_col:
            return pd.DataFrame()
        raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
        raw[rain_col] = pd.to_numeric(raw[rain_col], errors="coerce")
        raw = raw.dropna(subset=[date_col, rain_col])
        raw = raw.rename(columns={date_col:"date", rain_col:"catchment_rain_mm"})
        raw["date"] = raw["date"].dt.to_period("M").dt.to_timestamp()
        out = raw.groupby("date")["catchment_rain_mm"].sum().reset_index()
        log.info(f"  Havza yağış: {len(out)} aylık kayıt")
        return out
    except Exception as e:
        log.warning(f"  Havza yağış XLSX: {e}")
        return pd.DataFrame()


ISTANBUL_DAM_CAPACITY_MCM = 918.0
CATCHMENT_AREA_KM2       = 1000.0
RUNOFF_COEFFICIENT       = 0.35
OPEN_WATER_EVAP_MM_M     = 1.2


def compute_water_balance(
    rain_mm: pd.Series,
    et0_mm: pd.Series,
    consumption_m3: pd.Series,
    capacity_mcm: float = ISTANBUL_DAM_CAPACITY_MCM,
    runoff_coef: float = RUNOFF_COEFFICIENT,
    catchment_km2: float = CATCHMENT_AREA_KM2,
    demand_proxy_m3: float = 3.0e6,
) -> pd.Series:
    catchment_m2 = catchment_km2 * 1e6
    reservoir_m2 = 45.0e6
    capacity_m3  = capacity_mcm * 1e6
    inflow  = rain_mm.values * runoff_coef * (catchment_m2 / 1e3)  # mm * m2 /1000
    evap    = et0_mm.values * OPEN_WATER_EVAP_MM_M * (reservoir_m2 / 1e3)
    demand  = consumption_m3.values if len(consumption_m3) == len(rain_mm) \
              and consumption_m3.sum() > 0 else np.full(len(rain_mm), demand_proxy_m3)
    delta_m3 = inflow - evap - demand
    return pd.Series(delta_m3 / capacity_m3 * 100.0, index=rain_mm.index, name="wb_delta_pp")


def extract_trend(df: pd.DataFrame, col="fill_pct") -> dict:
    hist = df[df[col].notna()].copy()
    hist["year"] = hist["date"].dt.year
    annual = hist.groupby("year")[col].mean().dropna()
    if len(annual) < 5:
        return {"slope": 0.0, "se": 1.0, "p": 1.0, "annual": annual}
    x, y = annual.index.values.astype(float), annual.values
    slope, _, _, p, se = stats.linregress(x, y)
    n = len(y)
    s = sum(np.sign(y[j]-y[i]) for i in range(n-1) for j in range(i+1,n))
    vs = n*(n-1)*(2*n+5)/18
    zmk = (s-np.sign(s))/np.sqrt(vs) if vs else 0
    pmk = 2*(1-stats.norm.cdf(abs(zmk)))
    log.info(f"Trend: {slope:+.3f} pp/yr  p_OLS={p:.3f}  z_MK={zmk:+.2f}  p_MK={pmk:.3f}")
    return {"slope": float(slope), "se": float(se), "p": float(p),
            "z_mk": float(zmk), "p_mk": float(pmk), "annual": annual}


def compute_drifting_clim(
    static_clim: pd.Series,
    base_year: int,
    trend: dict,
    temp_rise: float = 0.040,
    precip_drop: float = 0.003,
    temp_sens: float = -0.35,
    damping: float = 0.70,
    max_drift: float = -35.0,
    mode: str = "linear",   # "linear" | "exp"
    tau: float = 60.0,      # exp modunda zaman sabiti (yıl)
) -> dict[int, pd.Series]:
    slope = trend.get("slope", 0.0)
    out = {}
    for yr in range(base_year, 2042):
        dt = float(yr - base_year)
        if mode == "exp":
            tc = slope * dt * (1 - np.exp(-dt / tau)) * damping
        else:
            # Linear drift: 2030 sonrası “ani yavaşlama” yaşamaz
            tc = slope * dt * damping
        tp = temp_sens * temp_rise * dt
        sh = float(np.clip(tc + tp, max_drift, 8.0))
        out[yr] = pd.Series({m: float(np.clip(
            float(static_clim.get(m,50)) + sh - precip_drop*dt*float(static_clim.get(m,50))*0.4,
            2, 98)) for m in range(1,13)})
    return out


def _api(r, k=0.92):
    a = np.zeros(len(r)); rv = r.fillna(0).values
    for i in range(1,len(rv)): a[i] = k*a[i-1]+rv[i]
    return pd.Series(a, index=r.index)
def _snow(t, r, ta=3, tm=5, mr=15):
    s = np.zeros(len(t)); tv=t.ffill().fillna(0).values; rv=r.fillna(0).values
    for i in range(1,len(tv)):
        acc = rv[i] if tv[i]<ta else 0
        mlt = max(0,(tv[i]-tm)*mr) if tv[i]>tm else 0
        s[i] = max(0,s[i-1]+acc-mlt)
    return pd.Series(s, index=t.index)
def _bucket(r, e, cap=200, dem=8):
    s = np.zeros(len(r)); rv=r.fillna(0).values; ev=e.fillna(0).values
    s[0]=cap/2
    for i in range(1,len(rv)):
        s[i]=float(np.clip(s[i-1]+rv[i]-ev[i]-dem,0,cap))
    return pd.Series(s/cap*100, index=r.index)
def _spi(r, w):
    rm=r.rolling(w,min_periods=max(2,w//2)).mean()
    rs=r.rolling(w,min_periods=max(2,w//2)).std().replace(0,np.nan)
    return ((r-rm)/rs).fillna(0)
def _stl(s, period=12):
    n=len(s); blank=pd.DataFrame({"stl_trend":np.zeros(n),"stl_resid":np.zeros(n)},index=s.index)
    if not HAS_STL or s.dropna().shape[0]<period*2: return blank
    try:
        res=STL(s.fillna(s.median()),period=period,robust=True).fit()
        return pd.DataFrame({"stl_trend":res.trend.values,"stl_resid":res.resid.values},index=s.index)
    except: return blank


ALL_FEATS = [
    "rain_mm","et0_mm_month","t_mean_c","rh_mean_pct","vpd_kpa_mean",
    "climate_balance_mm",
    "catchment_rain_mm",
    "consumption_norm",
    "wb_delta_pp",
    "api","snow_proxy_mm","bucket_pct",
    "stl_trend","stl_resid",
    "spi_3","spi_6","spi_12",
    "cum_balance_6","cum_balance_12",
    "month_sin","month_cos","quarter_sin","quarter_cos",
    "rain_delta","t_delta","rh_delta",
    "rain_ma3","rain_ma6","rain_ma12",
    "balance_ma3","balance_ma6","balance_ma12",
    "lag1_fill","lag3_fill","lag6_fill","lag12_fill",
    "lag1_delta","lag2_delta","lag3_delta","lag6_delta","lag12_delta",
    "delta_ma3","delta_ma6",
    "wb_lag1","wb_lag3",
]


def build_features(df: pd.DataFrame,
                   consumption: pd.DataFrame | None = None,
                   catchment_rain: pd.DataFrame | None = None) -> pd.DataFrame:
    d = df.copy().sort_values("date").reset_index(drop=True)
    if "weighted_total_fill" in d.columns and "fill_pct" not in d.columns:
        d["fill_pct"] = d["weighted_total_fill"] * 100.0

    if consumption is not None and not consumption.empty:
        d = d.merge(consumption, on="date", how="left")
        med = d["consumption_m3"].median()
        d["consumption_m3"] = d["consumption_m3"].fillna(med)
        d["consumption_norm"] = d["consumption_m3"] / (med + 1e-9)
    else:
        d["consumption_m3"]   = 0.0
        d["consumption_norm"] = 1.0

    if catchment_rain is not None and not catchment_rain.empty:
        d = d.merge(catchment_rain, on="date", how="left")
        d["catchment_rain_mm"] = d["catchment_rain_mm"].fillna(d["rain_mm"])
    else:
        d["catchment_rain_mm"] = d.get("rain_mm", pd.Series(50.0, index=d.index))

    es = 0.6108*np.exp(17.27*d["t_mean_c"]/(d["t_mean_c"]+237.3))
    vpd_calc = es*(1-d["rh_mean_pct"]/100)
    if "vpd_kpa_mean" not in d.columns or d["vpd_kpa_mean"].isna().all():
        d["vpd_kpa_mean"] = vpd_calc
    else:
        d["vpd_kpa_mean"] = d["vpd_kpa_mean"].fillna(vpd_calc)

    d["climate_balance_mm"] = d["rain_mm"]-d["et0_mm_month"]
    d["month"]   = d["date"].dt.month
    d["quarter"] = d["date"].dt.quarter

    d["api"]           = _api(d["catchment_rain_mm"]).values
    d["snow_proxy_mm"] = _snow(d["t_mean_c"], d["catchment_rain_mm"]).values
    d["bucket_pct"]    = _bucket(d["catchment_rain_mm"], d["et0_mm_month"]).values

    d["wb_delta_pp"] = compute_water_balance(
        d["catchment_rain_mm"], d["et0_mm_month"], d["consumption_m3"]
    ).values

    for w, n in [(3,"spi_3"),(6,"spi_6"),(12,"spi_12")]:
        d[n] = _spi(d["catchment_rain_mm"], w).values
    d["cum_balance_6"]  = d["climate_balance_mm"].rolling(6, min_periods=1).sum()
    d["cum_balance_12"] = d["climate_balance_mm"].rolling(12, min_periods=1).sum()

    d["rain_delta"] = d["rain_mm"].diff(1).fillna(0)
    d["t_delta"]    = d["t_mean_c"].diff(1).fillna(0)
    d["rh_delta"]   = d["rh_mean_pct"].diff(1).fillna(0)

    for w, n in [(3,"rain_ma3"),(6,"rain_ma6"),(12,"rain_ma12")]:
        d[n] = d["rain_mm"].rolling(w,min_periods=1).mean()
    for w, n in [(3,"balance_ma3"),(6,"balance_ma6"),(12,"balance_ma12")]:
        d[n] = d["climate_balance_mm"].rolling(w,min_periods=1).mean()

    d["month_sin"]   = np.sin(2*np.pi*d["month"]/12)
    d["month_cos"]   = np.cos(2*np.pi*d["month"]/12)
    d["quarter_sin"] = np.sin(2*np.pi*d["quarter"]/4)
    d["quarter_cos"] = np.cos(2*np.pi*d["quarter"]/4)

    fp = d.get("fill_pct", pd.Series(np.nan, index=d.index))
    d["delta_fill"] = fp.diff(1)
    for lag, n in [(1,"lag1_fill"),(3,"lag3_fill"),(6,"lag6_fill"),(12,"lag12_fill")]:
        d[n] = fp.shift(lag)
    for lag, n in [(1,"lag1_delta"),(2,"lag2_delta"),(3,"lag3_delta"),
                    (6,"lag6_delta"),(12,"lag12_delta")]:
        d[n] = d["delta_fill"].shift(lag)
    d["delta_ma3"] = d["delta_fill"].shift(1).rolling(3,min_periods=1).mean()
    d["delta_ma6"] = d["delta_fill"].shift(1).rolling(6,min_periods=1).mean()

    d["wb_lag1"] = d["wb_delta_pp"].shift(1)
    d["wb_lag3"] = d["wb_delta_pp"].shift(3)

    stlf = _stl(fp)
    d["stl_trend"] = stlf["stl_trend"].values
    d["stl_resid"]  = stlf["stl_resid"].values

    return d


def static_climatology(fill: pd.Series, months: pd.Series) -> pd.Series:
    valid = fill.dropna()
    return valid.groupby(months.loc[valid.index]).mean()
def climatology_delta(delta: pd.Series, months: pd.Series) -> pd.Series:
    valid = delta.dropna()
    return valid.groupby(months.loc[valid.index]).mean()


def estimate_ar1(df, model, feat_cols, target_col="delta_fill"):
    hist = df[df[target_col].notna()].dropna(subset=feat_cols+[target_col]).copy()
    if len(hist) < 24:
        return {
            "phi": 0.55,
            "sigma": {m: 2.0 for m in range(1,13)},
            "sigma_hist": {m: 2.0 for m in range(1,13)},
        }
    try:
        model.fit(hist[feat_cols].values, hist[target_col].values)
        resid = hist[target_col].values - model.predict(hist[feat_cols].values)
    except Exception:
        return {
            "phi": 0.55,
            "sigma": {m: 2.0 for m in range(1,13)},
            "sigma_hist": {m: 2.0 for m in range(1,13)},
        }
    phi = float(np.clip(np.corrcoef(resid[1:],resid[:-1])[0,1], 0, 0.92))
    months = hist["date"].dt.month.values
    sigma  = {m: float(np.std(resid[months==m])) if (months==m).sum()>2 else 2.0
              for m in range(1,13)}
    sigma_hist = {
        m: float(np.std(hist.loc[months == m, target_col].values))
        if (months == m).sum() > 2 else 2.0
        for m in range(1, 13)
    }
    log.info(
        f"AR(1): φ={phi:.3f}  σ_resid_ort={np.mean(list(sigma.values())):.2f}  "
        f"σ_hist_ort={np.mean(list(sigma_hist.values())):.2f}"
    )
    return {"phi": phi, "sigma": sigma, "sigma_hist": sigma_hist}


class SplitConformal:
    def __init__(self, cov=0.90): self.cov=cov; self.q=0.0
    def calibrate(self, yt, yp):
        r=np.abs(yt-yp); a=1-self.cov; n=len(r)
        lv=min(np.ceil((n+1)*(1-a))/n,1.0)
        self.q=float(np.quantile(r,lv))
        log.info(f"  Konformal ±{self.q:.2f} pp ({self.cov*100:.0f}%)")
    def interval(self, yp):
        return np.clip(yp-self.q,0,100), np.clip(yp+self.q,0,100)


def make_models(lgb_p=None, xgb_p=None):
    m = {}
    if HAS_LGB:
        d = dict(boosting_type="dart",n_estimators=600,learning_rate=0.05,
                 num_leaves=31,max_depth=6,min_child_samples=10,
                 subsample=0.85,colsample_bytree=0.85,reg_alpha=0.1,
                 reg_lambda=1.0,drop_rate=0.1,random_state=42,n_jobs=-1,verbose=-1)
        if lgb_p: d.update(lgb_p)
        m["lgb_dart"] = lgb.LGBMRegressor(**d); log.info("  LGB DART")
        qbase = dict(
            boosting_type="gbdt",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=10,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            objective="quantile",
        )
        m["lgb_q10"]  = lgb.LGBMRegressor(**{**qbase, "alpha": 0.10})
        m["lgb_q90"]  = lgb.LGBMRegressor(**{**qbase, "alpha": 0.90})
    if HAS_XGB:
        d2 = dict(n_estimators=600,learning_rate=0.04,max_depth=5,
                  subsample=0.85,colsample_bytree=0.85,reg_alpha=0.1,
                  reg_lambda=1.5,random_state=42,n_jobs=-1,verbosity=0)
        if xgb_p: d2.update(xgb_p)
        m["xgb"] = xgb.XGBRegressor(**d2); log.info("  XGBoost")
    etr = ExtraTreesRegressor(n_estimators=400,max_features=0.6,
                               min_samples_leaf=3,random_state=42,n_jobs=-1)
    m["etr"] = etr; log.info("  ETR")
    base = [(k,v) for k,v in m.items() if k not in ("lgb_q10","lgb_q90")]
    if len(base) >= 2:
        m["stack"] = StackingRegressor(
            estimators=base,
            final_estimator=make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            cv=5, n_jobs=-1)
        log.info("  Stack")
    return m


def purged_cv(df, models, feat_cols, target="delta_fill",
              n_folds=5, embargo=6, min_train=60):
    log.info(f"Purged CV ({target}): {n_folds} fold, {embargo} ay embargo")
    hist = df[df[target].notna()].dropna(subset=feat_cols+[target])
    years = sorted(hist["date"].dt.year.unique())[-n_folds:]
    rows  = []
    for yr in years:
        ts = pd.Timestamp(f"{yr}-01-01")
        ee = ts - pd.DateOffset(months=embargo)
        tr = hist[hist["date"]<ee]
        te = hist[hist["date"].dt.year==yr]
        if len(tr)<min_train or te.empty: continue
        for name, model in models.items():
            if name in ("lgb_q10","lgb_q90","stack"): continue
            try:
                model.fit(tr[feat_cols].values, tr[target].values)
                pred_d = model.predict(te[feat_cols].values)
                fill_start = float(tr["fill_pct"].iloc[-1]) if "fill_pct" in tr.columns \
                             and not tr["fill_pct"].isna().all() else 50.0
                pred_f = np.clip(fill_start + np.cumsum(pred_d), 0, 100)
                true_f = te["fill_pct"].values if "fill_pct" in te.columns else pred_f
                r = rmse(true_f, pred_f)
                rows.append({"model":name,"test_year":yr,
                             "rmse_pp":round(r,3),
                             "mape_pct":round(mape(pd.Series(true_f),pd.Series(pred_f)),3),
                             "pearson_r":round(pearson_r(true_f,pred_f),3)})
                log.info(f"  {yr}│{name:10s}│RMSE={r:.2f}pp")
            except Exception as e:
                log.error(f"  {yr}│{name}: {e}")
    cv = pd.DataFrame(rows)
    if not cv.empty:
        log.info(f"\n{cv.groupby('model')[['rmse_pp','mape_pct','pearson_r']].mean().round(3).to_string()}\n")
    return cv


def h_alpha(h, h1=18, h2=96, a1=0.95, a2=0.20):
    if h<=h1: return a1
    if h>=h2: return a2
    return a1+(h-h1)/(h2-h1)*(a2-a1)


def simulate_v4(
    df: pd.DataFrame,
    model, q10, q90,
    static_clim: pd.Series,
    drift_clim: dict[int, pd.Series],
    clim_delta: pd.Series,
    conformal: SplitConformal,
    ar1: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    feat_cols: list,
    seed: int = 42,
    sigma_floor: float = 0.6,
    sigma_hist_scale: float = 0.25,
    op_k: float = 0.04,
) -> pd.DataFrame:
    data  = df.copy().sort_values("date").reset_index(drop=True)
    d2i   = {d:i for i,d in enumerate(data["date"])}
    si, ei = d2i.get(start), d2i.get(end)
    if si is None or ei is None:
        log.error("start/end bulunamadı"); return data

    buf  = data["fill_pct"].values.copy().astype(float)
    wb_buf = data.get("wb_delta_pp", pd.Series(0.0, index=data.index)).values.copy().astype(float)
    dbuf = np.zeros(len(data))
    fp = data["fill_pct"].values.astype(float)
    for i in range(1, si):
        dbuf[i] = fp[i] - fp[i-1] if not (np.isnan(fp[i]) or np.isnan(fp[i-1])) else 0.0

    sim = np.full(len(data), np.nan)
    lo  = np.full(len(data), np.nan)
    hi  = np.full(len(data), np.nan)
    for i in range(si):
        sim[i]=lo[i]=hi[i]=float(buf[i]) if not np.isnan(buf[i]) else np.nan

    rng = np.random.default_rng(seed)
    phi   = ar1.get("phi", 0.55)
    sigmo = ar1.get("sigma", {m:2.0 for m in range(1,13)})
    sigma_hist = ar1.get("sigma_hist", {m:2.0 for m in range(1,13)})
    ar1s  = 0.0
    h_tot = 0

    def blag_fill(i, lag):
        p = i-lag
        if p<0: return float(np.nanmean(buf[:max(1,i)]))
        v=buf[p]; return v if not np.isnan(v) else float(np.nanmean(buf[max(0,p-12):p+1]))
    def blag_delta(i, lag):
        p = i-lag
        if p<0 or p>=len(dbuf): return 0.0
        return float(dbuf[p])
    def bma_delta(i, w):
        vs=[dbuf[i-k] for k in range(1,w+1) if i-k>=0]
        return float(np.mean(vs)) if vs else 0.0

    for year in range(start.year, end.year+1):
        yr_clim = drift_clim.get(year, drift_clim.get(max(drift_clim)))
        tr_end  = pd.Timestamp(f"{year-1}-12-01")
        tr = data[(data["date"]<=tr_end) & data["fill_pct"].notna()].dropna(subset=feat_cols+["delta_fill"])
        if tr.empty: continue
        Xtr = tr[feat_cols].values; ytr = tr["delta_fill"].values
        try:
            model.fit(Xtr,ytr); q10.fit(Xtr,ytr); q90.fit(Xtr,ytr)
        except Exception as e:
            log.error(f"  {year}: {e}"); continue

        mask = ((data["date"]>=pd.Timestamp(f"{year}-01-01")) &
                (data["date"]<=min(pd.Timestamp(f"{year}-12-01"),end)) &
                (data["date"]>=start))
        idxs = data.index[mask].tolist()

        for i in idxs:
            h_tot += 1
            mo = int(data.loc[i,"month"]) if "month" in data.columns \
                 else int(data.loc[i,"date"].month)
            row = data.loc[i].copy()
            for lag,c in [(1,"lag1_fill"),(3,"lag3_fill"),(6,"lag6_fill"),(12,"lag12_fill")]:
                row[c]=blag_fill(i,lag)
            for lag,c in [(1,"lag1_delta"),(2,"lag2_delta"),(3,"lag3_delta"),
                           (6,"lag6_delta"),(12,"lag12_delta")]:
                row[c]=blag_delta(i,lag)
            row["delta_ma3"]=bma_delta(i,3); row["delta_ma6"]=bma_delta(i,6)
            row["wb_lag1"] = wb_buf[i-1] if i-1 >= 0 else 0.0
            row["wb_lag3"] = wb_buf[i-3] if i-3 >= 0 else 0.0
            row["stl_trend"]=float(np.nanmean([blag_fill(i,k) for k in range(1,13)]))
            row["stl_resid"]=blag_fill(i,1)-row["stl_trend"]
            for fc in feat_cols:
                if pd.isna(row.get(fc,np.nan)):
                    row[fc]=float(Xtr[:,feat_cols.index(fc)].mean())
            Xp = row[feat_cols].values.reshape(1,-1)
            try:
                d_ml  = float(model.predict(Xp)[0])
                d_q10 = float(q10.predict(Xp)[0])
                d_q90 = float(q90.predict(Xp)[0])
            except Exception:
                d_ml = d_q10 = d_q90 = float(clim_delta.get(mo,0))
            wb = float(row.get("wb_delta_pp", clim_delta.get(mo,0)))
            alpha   = h_alpha(h_tot)
            d_blend = alpha*d_ml + (1-alpha)*(wb + float(clim_delta.get(mo,0)))/2.0
            prev_fill = blag_fill(i,1)
            dc = float(yr_clim.get(mo, float(static_clim.get(mo,50))))
            # Operasyonel geri çekim: doluluğu drifting klimatolojiye yumuşakça yaklaştır
            d_blend += op_k * (dc - prev_fill)
            sigma_base = sigmo.get(mo, 2.0)
            sigma_hist_m = sigma_hist.get(mo, 2.0)
            sigma = max(sigma_base, sigma_floor, sigma_hist_m * sigma_hist_scale)
            ar1s    = phi*ar1s + sigma*float(rng.normal(0,1))
            ns      = float(np.clip(np.sqrt(h_tot/12)*0.30, 0, 1.5))
            d_blend = d_blend + ar1s*ns
            fill_new  = float(np.clip(prev_fill + d_blend, 0, 100))
            gravity = (1-alpha)*0.08
            fill_new = fill_new*(1-gravity) + dc*gravity
            fill_new = float(np.clip(fill_new,0,100))
            lo_fill = float(np.clip(prev_fill+d_q10-abs(ar1s)*ns, 0, 100))
            hi_fill = float(np.clip(prev_fill+d_q90+abs(ar1s)*ns, 0, 100))
            lc,hc = conformal.interval(np.array([fill_new]))
            flo = float(np.clip(min(lc[0],lo_fill),0,100))
            fhi = float(np.clip(max(hc[0],hi_fill),0,100))
            sim[i]=fill_new; lo[i]=flo; hi[i]=fhi
            buf[i]=fill_new; dbuf[i]=fill_new-prev_fill

    out = data[["date"]].copy()
    out["fill_pct"]=data.get("fill_pct",pd.Series(np.nan,index=data.index))
    out["fill_sim"]=sim; out["fill_lo"]=lo; out["fill_hi"]=hi
    return out


def plot_v4(out, drift_clim, last_obs, hist_trend,
            model_name, cv_sum, fig_dir, static_clim):
    hm=out["date"]<=last_obs; fm=out["date"]>last_obs
    fig=plt.figure(figsize=(17,11))
    gs=gridspec.GridSpec(2,4,height_ratios=[2.5,1],hspace=0.42,wspace=0.32)
    ax1=fig.add_subplot(gs[0,:]); ax2=fig.add_subplot(gs[1,0])
    ax3=fig.add_subplot(gs[1,1:3]); ax4=fig.add_subplot(gs[1,3])

    ax1.plot(out["date"][hm],out["fill_pct"][hm],"#111",lw=1.3,label="Gözlenen",zorder=4)
    ax1.plot(out["date"][fm],out["fill_sim"][fm],"#D55E00",lw=2.0,
             label=f"Tahmin — {model_name}",zorder=5)
    ax1.fill_between(out["date"][fm],
                      out["fill_lo"][fm].clip(0,100),out["fill_hi"][fm].clip(0,100),
                      color="#D55E00",alpha=0.13,lw=0,label="Konformal %90")
    dys=[y for y in sorted(drift_clim) if y>=last_obs.year]
    ax1.plot([pd.Timestamp(f"{y}-07-01") for y in dys],
             [drift_clim[y].get(7,50) for y in dys],
             "#0F6E56",lw=1.0,ls="--",alpha=0.6,label="Kayan klim. (Tem.)")
    ax1.axvline(last_obs,color="#888",lw=0.8,ls="--",alpha=0.7)
    ax1.axhline(30,color="#E24B4A",lw=0.7,ls=":",alpha=0.5)
    ax1.text(out["date"].iloc[0],31.5,"Kritik eşik (%30)",fontsize=8,color="#E24B4A",alpha=0.7)
    ax1.set_ylim(0,108); ax1.set_ylabel("Doluluk (%)",fontsize=10)
    sl=hist_trend.get("slope",0)
    ax1.set_title(f"İstanbul Baraj v4 — {model_name.upper()}  │  "
                   f"Trend:{sl:+.2f}pp/yr  │  Δfill+WB+Tüketim+AR(1)",fontsize=10.5)
    ax1.legend(fontsize=8.5,loc="upper right"); ax1.grid(True,color="#EEE",lw=0.5)
    ax1.spines[["top","right"]].set_visible(False)
    if cv_sum:
        ax1.text(0.01,0.02,
                 f"CV: RMSE={cv_sum.get('rmse_pp','?'):.2f}pp  "
                 f"MAPE={cv_sum.get('mape_pct','?'):.1f}%  r={cv_sum.get('pearson_r','?'):.3f}",
                 transform=ax1.transAxes,fontsize=8.5,color="#555",
                 bbox=dict(boxstyle="round,pad=0.3",fc="#F9F9F9",ec="#CCC",alpha=0.9))

    for mo,col,lbl in [(3,"#185FA5","Mart"),(7,"#E24B4A","Temmuz"),(11,"#0F6E56","Kasım")]:
        ax2.plot(dys,[drift_clim[y].get(mo,50) for y in dys],color=col,lw=1.3,
                 label=lbl,marker="o",markersize=2.5)
    ax2.set_title("Kayan klimatoloji / ay",fontsize=9)
    ax2.set_ylabel("Taban (%)")
    ax2.legend(fontsize=7); ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(True,color="#EEE",lw=0.5)

    def ym(g):
        return g["fill_sim"].mean() if (g["date"]>last_obs).any() else g["fill_pct"].mean()
    yr=out.groupby(out["date"].dt.year).apply(ym).dropna()
    cyr=["#185FA5" if y<=last_obs.year else "#D55E00" for y in yr.index]
    ax3.bar(yr.index,yr.values,color=cyr,alpha=0.8,width=0.8)
    x,y=yr.index.values.astype(float),yr.values
    if len(x)>3:
        ms,bs,*_=stats.linregress(x,y)
        ax3.plot(x,ms*x+bs,"#333",lw=1.1,ls="--",alpha=0.7,
                 label=f"OLS {ms:+.2f} pp/yr"); ax3.legend(fontsize=7)
    ax3.axhline(30,color="#E24B4A",lw=0.7,ls=":",alpha=0.5)
    ax3.set_title("Yıllık ort. + trend",fontsize=9); ax3.set_ylim(0,100)
    ax3.spines[["top","right"]].set_visible(False); ax3.grid(True,axis="y",color="#EEE",lw=0.5)
    ax3.legend(handles=[Patch(color="#185FA5",alpha=0.8,label="Gözlenen"),
                         Patch(color="#D55E00",alpha=0.8,label="Tahmin")]+
               (ax3.get_legend_handles_labels()[0][-1:] if ax3.get_legend() else []),fontsize=7)

    fut=out[fm].copy(); fut["unc"]=(fut["fill_hi"]-fut["fill_lo"]).clip(0,100)
    ax4.plot(range(1,len(fut)+1),fut["unc"].values,"#185FA5",lw=1.2)
    ax4.set_xlabel("Ufuk (ay)"); ax4.set_ylabel("Bant (pp)")
    ax4.set_title("Belirsizlik büyümesi",fontsize=9)
    ax4.spines[["top","right"]].set_visible(False); ax4.grid(True,color="#EEE",lw=0.5)

    plt.savefig(fig_dir/f"{model_name}_v4.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Grafik: {model_name}_v4.png")


def plot_cv_bar(cv_df, fig_dir):
    if cv_df.empty: return
    s=cv_df.groupby("model")["rmse_pp"].mean().sort_values()
    fig,ax=plt.subplots(figsize=(8,4))
    cols=["#185FA5" if i==0 else "#85B7EB" for i in range(len(s))]
    bars=ax.barh(s.index,s.values,color=cols,alpha=0.9)
    ax.bar_label(bars,fmt="%.2f pp",padding=4,fontsize=9)
    ax.set_xlabel("Ort. CV RMSE (pp)"); ax.set_title("Model CV Karşılaştırması")
    ax.spines[["top","right"]].set_visible(False); ax.grid(True,axis="x",color="#EEE",lw=0.5)
    plt.tight_layout(); fig.savefig(fig_dir/"model_cv.png",dpi=150); plt.close(fig)


def fetch_iski_month_end() -> pd.DataFrame:
    """
    İSKİ ay sonu doluluk (son 12 ay).
    API: https://iskiapi.iski.istanbul/api/iski/baraj/sonBirYildakiAySonlariDoluluk/v2
    Gerekli token, iski.istanbul Nuxt JS içinden okunur.
    Döndürür: date (MS), fill_obs
    """
    if not HAS_REQUESTS:
        log.warning("requests yok, İSKİ verisi alınamadı.")
        return pd.DataFrame()
    try:
        js = requests.get("https://iski.istanbul/_nuxt/34ba1a6.js", timeout=20).text
        m = re.search(r'NUXT_ENV_AUTH_TOKEN:\"([0-9a-f]+)\"', js)
        if not m:
            log.warning("İSKİ token bulunamadı.")
            return pd.DataFrame()
        token = m.group(1)
        headers = {"Authorization": f"Bearer {token}"}
        url = "https://iskiapi.iski.istanbul/api/iski/baraj/sonBirYildakiAySonlariDoluluk/v2"
        r = requests.get(url, headers=headers, timeout=20)
        data = r.json().get("data", [])
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["tarih"], dayfirst=True, errors="coerce") \
                         .dt.to_period("M").dt.to_timestamp()
        df["fill_obs"] = pd.to_numeric(df["oran"], errors="coerce")
        df = df[["date", "fill_obs"]].dropna()
        return df
    except Exception as e:
        log.warning(f"İSKİ veri çekme hatası: {e}")
        return pd.DataFrame()


def compare_iski(proj_df: pd.DataFrame,
                 obs_df: pd.DataFrame,
                 out_path: Path) -> tuple[dict | None, pd.DataFrame]:
    """
    İSKİ ay sonu ile projeksiyon kıyaslar.
    Döndürür: (metrikler, birleşik df)
    """
    m = proj_df.merge(obs_df, on="date", how="inner")
    if m.empty:
        return None, m
    m[["date", "fill_obs", "fill_sim"]].to_csv(out_path, index=False)
    metrics = {
        "rmse": rmse(m["fill_obs"].values, m["fill_sim"].values),
        "mape": mape(m["fill_obs"].values, m["fill_sim"].values),
        "pearson_r": pearson_r(m["fill_obs"].values, m["fill_sim"].values),
        "n": len(m),
    }
    return metrics, m


def apply_bias_correction(proj_df: pd.DataFrame,
                          obs_df: pd.DataFrame,
                          w_month: float = 0.60) -> pd.DataFrame | None:
    """
    Ay bazlı (mevsimsel) bias düzeltmesi.
    w_month: aylık bias'a verilen ağırlık (global bias ile shrink).
    """
    m = proj_df.merge(obs_df, on="date", how="inner")
    if m.empty or len(m) < 6:
        return None
    err = m["fill_sim"] - m["fill_obs"]
    global_bias = float(err.mean())
    month_bias = err.groupby(m["date"].dt.month).mean().to_dict()

    def bias_for_month(mo: int) -> float:
        mb = float(month_bias.get(mo, global_bias))
        return w_month * mb + (1.0 - w_month) * global_bias

    out = proj_df.copy()
    out["bias_adj"] = out["date"].dt.month.map(bias_for_month)
    for c in ["fill_sim", "fill_lo", "fill_hi"]:
        out[c] = (out[c] - out["bias_adj"]).clip(0, 100)
    return out


def plot_iski_compare(orig_df: pd.DataFrame,
                      corr_df: pd.DataFrame | None,
                      fig_dir: Path,
                      tag: str = "compare_iski_last12m",
                      orig_label: str = "v4 (orijinal)",
                      corr_label: str = "v4 (düzeltilmiş)",
                      obs_label: str = "İSKİ gözlem"):
    if orig_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(orig_df["date"], orig_df["fill_obs"], label=obs_label,
            color="#111", lw=1.8)
    ax.plot(orig_df["date"], orig_df["fill_sim"], label=orig_label,
            color="#D55E00", lw=1.5, alpha=0.6)
    if corr_df is not None and not corr_df.empty:
        ax.plot(corr_df["date"], corr_df["fill_sim"], label=corr_label,
                color="#0F6E56", lw=2.0)
    ax.set_title("v4 vs İSKİ — Son 12 Ay (Ay sonu)")
    ax.set_ylabel("Doluluk (%)")
    ax.grid(True, color="#EEE", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    plt.savefig(fig_dir / f"{tag}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def compute_month_end_delta(daily_path: Path) -> pd.Series | None:
    """
    Günlük doluluk verisinden ay sonu - aylık ortalama farkını çıkarır.
    Döndürür: ay(1..12) -> delta (pp)
    """
    if not daily_path.exists():
        log.warning(f"Ay sonu düzeltme: günlük dosya yok → {daily_path.name}")
        return None
    try:
        raw = pd.read_excel(daily_path)
        if "Tarih" not in raw.columns:
            # olası tarih kolonları
            date_col = next((c for c in raw.columns
                             if "tarih" in str(c).lower()), None)
        else:
            date_col = "Tarih"
        if not date_col:
            return None
        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
        dam_cols = [c for c in raw.columns if c != date_col]
        # tüm dam kolonlarını sayısala çevir
        for c in dam_cols:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw = raw.dropna(subset=[date_col])
        # günlük ortalama (damlar arası)
        daily = raw[dam_cols].mean(axis=1)
        if daily.median() < 2:
            daily = daily * 100.0
        df = pd.DataFrame({"date": raw[date_col], "daily_mean": daily})
        df["month"] = df["date"].dt.month
        df["ym"] = df["date"].dt.to_period("M")
        # aylık ortalama
        m_mean = df.groupby("ym")["daily_mean"].mean()
        # ay sonu (son gözlem)
        idx = df.groupby("ym")["date"].idxmax()
        m_end = df.loc[idx].set_index("ym")["daily_mean"]
        delta = (m_end - m_mean).dropna()
        month_delta = delta.groupby(delta.index.month).mean()
        log.info(f"Ay sonu düzeltme (ort.): {month_delta.round(2).to_dict()}")
        return month_delta
    except Exception as e:
        log.warning(f"Ay sonu delta hesaplanamadı: {e}")
        return None


def estimate_monthly_mean_from_month_end(iski_df: pd.DataFrame,
                                         month_delta: pd.Series) -> pd.DataFrame:
    """
    İSKİ ay sonu doluluk -> aylık ortalama tahmini.
    fill_mean ≈ month_end - delta_month
    """
    if iski_df.empty:
        return iski_df
    out = iski_df.copy()
    out["month"] = out["date"].dt.month
    out["fill_obs"] = out["fill_obs"] - out["month"].map(month_delta).fillna(0.0)
    out["fill_obs"] = out["fill_obs"].clip(0, 100)
    return out[["date", "fill_obs"]]


def apply_iski_monthly_to_panel(panel: pd.DataFrame,
                                iski_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Panel'deki eksik fill_pct değerlerini İSKİ aylık gözlemle tamamlar.
    """
    if iski_monthly.empty:
        return panel
    out = panel.copy()
    m = out.merge(iski_monthly.rename(columns={"fill_obs": "fill_iski"}),
                  on="date", how="left")
    before = out["fill_pct"].notna().sum()
    out["fill_pct"] = out["fill_pct"].where(out["fill_pct"].notna(), m["fill_iski"])
    after = out["fill_pct"].notna().sum()
    added = int(after - before)
    log.info(f"İSKİ aylık gözlem eklendi: {added} ay")
    return out


def apply_month_end_adjustment(proj_df: pd.DataFrame,
                               month_delta: pd.Series) -> pd.DataFrame:
    out = proj_df.copy()
    out["month"] = out["date"].dt.month
    out["me_delta"] = out["month"].map(month_delta).fillna(0.0)
    for c in ["fill_sim", "fill_lo", "fill_hi"]:
        out[c] = (out[c] + out["me_delta"]).clip(0, 100)
    return out


def load_all_data(panel_path: Path, climate_path: Path | None,
                  ibb_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info(f"Panel: {panel_path.name}")
    panel = pd.read_csv(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    if "weighted_total_fill" in panel.columns:
        panel["fill_pct"] = panel["weighted_total_fill"] * 100.0

    if climate_path and climate_path.exists():
        log.info(f"İklim: {climate_path.name}")
        clim=pd.read_csv(climate_path); clim["date"]=pd.to_datetime(clim["date"])
        if "precip_mm_month" in clim.columns:
            clim=clim.rename(columns={"precip_mm_month":"rain_mm"})
        fut=clim[clim["date"].dt.year>panel["date"].dt.year.max()]
        panel["_m"]=panel["date"].dt.month
        cmeans={c:panel.groupby("_m")[c].mean()
                for c in ["rain_mm","et0_mm_month","t_mean_c","rh_mean_pct","pressure_kpa"]
                if c in panel.columns}
        full=pd.DataFrame({"date":pd.date_range(panel["date"].min(),"2040-12-01",freq="MS")})
        full=full.merge(panel.drop(columns=["_m"],errors="ignore"),on="date",how="left")
        fc=[c for c in ["rain_mm","et0_mm_month","t_mean_c","rh_mean_pct"] if c in fut.columns]
        full=full.merge(fut[["date"]+fc],on="date",how="left",suffixes=("","_c"))
        for c in fc:
            cc=f"{c}_c"
            if cc in full.columns:
                full[c]=full[c].where(full[cc].isna(),full[cc]); full.drop(columns=[cc],inplace=True)
        full["_m"]=full["date"].dt.month
        for c,mo in cmeans.items():
            full[c]=full[c].fillna(full["_m"].map(mo))
        full.drop(columns=["_m"],inplace=True); panel=full

    num=panel.select_dtypes(include=np.number).columns
    excl={"fill_pct","weighted_total_fill"}
    fill_these=[c for c in num if c not in excl]
    panel[fill_these]=panel[fill_these].fillna(panel[fill_these].mean())
    panel=panel.sort_values("date").reset_index(drop=True)
    log.info(f"Panel: {len(panel)} satır")

    consumption = load_ibb_consumption(
        ibb_dir / "İstanbul_Barajlarına_Düşen_Yağış_Ve_Günlük_Tüketim_Verileri_762b802e-c5f9-4175-a5c1-78b892d9764b.xlsx")
    catchment   = load_ibb_catchment_rain(
        ibb_dir / "İstanbul_Barajlarına_Düşen_Yağış_Ve_Günlük_Tüketim_Verileri_762b802e-c5f9-4175-a5c1-78b892d9764b.xlsx")

    return panel, consumption, catchment


def run_shap(model, X, feat_cols, out_dir, tag):
    if not HAS_SHAP: return
    try:
        try:   exp=shap.TreeExplainer(model); sv=exp.shap_values(X)
        except:exp=shap.LinearExplainer(model,X); sv=exp.shap_values(X)
        imp=pd.Series(np.abs(sv).mean(0),index=feat_cols).sort_values()
        top=imp.tail(20)
        fig,ax=plt.subplots(figsize=(9,6))
        colors=["#185FA5" if v>imp.median() else "#85B7EB" for v in top.values]
        ax.barh(top.index,top.values,color=colors,height=0.7)
        ax.set_xlabel("Ort. |SHAP| (Δfill pp uzayı)"); ax.set_title(f"SHAP — {tag}")
        ax.axvline(top.values.mean(),color="#888",lw=0.7,ls="--")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); fig.savefig(out_dir/f"shap_{tag}.png",dpi=150); plt.close(fig)
        log.info(f"  Top5: {list(imp.tail(5)[::-1].index)}")
    except Exception as e:
        log.warning(f"  SHAP: {e}")


def main():
    p=argparse.ArgumentParser(description="İstanbul Baraj v4")
    p.add_argument("--panel",
        default="output/newdata_feature_store/tables/istanbul_dam_driver_panel_2000_2026_extended.csv")
    p.add_argument("--climate",
        default="output/scientific_climate_projection_2026_2040/climate_projection_2010_2040_monthly.csv")
    p.add_argument("--ibb-dir",  default="external/raw/ibb")
    p.add_argument("--out",      default="output/istanbul_v4")
    p.add_argument("--end-date", default="2040-12-01")
    p.add_argument("--cv-folds", type=int,   default=5)
    p.add_argument("--coverage", type=float, default=0.90)
    p.add_argument("--tune",     action="store_true")
    p.add_argument("--n-trials", type=int,   default=60)
    p.add_argument("--no-shap",  action="store_true")
    p.add_argument("--scenarios",action="store_true")
    p.add_argument("--temp-rise",   type=float, default=0.040)
    p.add_argument("--precip-drop", type=float, default=0.003)
    p.add_argument("--damping",     type=float, default=0.70)
    p.add_argument("--drift-mode",  type=str, default="linear",
                   choices=["linear","exp"],
                   help="Kayan klimatoloji trend modu (linear önerilir).")
    p.add_argument("--drift-tau",   type=float, default=60.0,
                   help="Exp modunda zaman sabiti (yıl).")
    p.add_argument("--sigma-floor", type=float, default=0.6,
                   help="AR(1) gürültü alt sınırı (pp). Çok düzenli çıktıyı kırar.")
    p.add_argument("--sigma-hist-scale", type=float, default=0.25,
                   help="Aylık Δfill std katkı ölçeği (0–1).")
    p.add_argument("--op-k", type=float, default=0.04,
                   help="Operasyonel geri çekim katsayısı (aylık).")
    p.add_argument("--no-bias-correct", action="store_true",
                   help="İSKİ ay sonu bias düzeltmesini kapat.")
    p.add_argument("--bias-month-weight", type=float, default=0.60,
                   help="Aylık bias ağırlığı (0–1).")
    p.add_argument("--monthend-adjust", action="store_true",
                   help="Ay sonu düzeltmesini uygula (varsayılan kapalı).")
    p.add_argument("--use-iski-augment", action="store_true",
                   help="İSKİ 2025–2026 aylarını eğitime ekle (varsayılan kapalı).")
    args=p.parse_args()

    out_dir=ROOT/args.out; fig_dir=out_dir/"figures"
    out_dir.mkdir(parents=True,exist_ok=True); fig_dir.mkdir(exist_ok=True)
    fh=logging.FileHandler(out_dir/"run.log",encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-8s │ %(message)s"))
    log.addHandler(fh)

    log.info("="*70)
    log.info("İstanbul Baraj v4 — Δfill+WB+Tüketim+AR(1)+KayanKlim")
    log.info(f"  LGB:{HAS_LGB} XGB:{HAS_XGB} Optuna:{HAS_OPTUNA} SHAP:{HAS_SHAP}")
    log.info(f"  sigma_floor={args.sigma_floor}  sigma_hist_scale={args.sigma_hist_scale}")
    log.info("="*70)

    ibb_dir=ROOT/args.ibb_dir
    cp=ROOT/args.climate
    panel, consumption, catchment = load_all_data(
        ROOT/args.panel, cp if cp.exists() else None, ibb_dir)

    # ── İSKİ ay sonu verisini hazırla (karşılaştırma için)
    daily_path = ROOT / args.ibb_dir / (
        "İstanbul_Barajları_Günlük_Doluluk_Oranları_"
        "af0b3902-cfd9-4096-85f7-e2c3017e4f21.xlsx"
    )
    month_delta = compute_month_end_delta(daily_path)
    iski_month_end = fetch_iski_month_end()
    iski_monthly = None
    if month_delta is not None and not iski_month_end.empty:
        iski_monthly = estimate_monthly_mean_from_month_end(iski_month_end, month_delta)
    elif not iski_month_end.empty:
        # ay sonu verisi var ama delta yok → doğrudan kullan (yaklaşık)
        iski_monthly = iski_month_end.copy()
    # İstenirse eğitime ekle
    if args.use_iski_augment and iski_monthly is not None and not iski_monthly.empty:
        panel = apply_iski_monthly_to_panel(panel, iski_monthly)

    last_obs   = panel[panel["fill_pct"].notna()]["date"].max()
    start_date = (pd.Timestamp(last_obs)+pd.DateOffset(months=1)).normalize()
    end_date   = pd.Timestamp(args.end_date)
    log.info(f"Son gözlem: {last_obs.date()} | {start_date.date()} → {end_date.date()}")

    hist_raw    = panel[panel["fill_pct"].notna()].copy()
    sclim       = static_climatology(hist_raw["fill_pct"], hist_raw["date"].dt.month)
    panel["clim_fill"]=panel["date"].dt.month.map(sclim)
    log.info(f"Statik klim: {dict(sclim.round(1))}")

    hist_trend = extract_trend(panel)
    drift_clim = compute_drifting_clim(
        sclim, last_obs.year, hist_trend,
        args.temp_rise, args.precip_drop,
        damping=args.damping,
        mode=args.drift_mode, tau=args.drift_tau)
    y0 = last_obs.year
    log.info(f"Kayan klim Tem: {y0}={drift_clim.get(y0,{}).get(7,0):.1f} "
             f"2030={drift_clim.get(2030,{}).get(7,0):.1f} "
             f"2040={drift_clim.get(2040,{}).get(7,0):.1f} "
             f"(mode={args.drift_mode})")

    log.info("Özellik mühendisliği (Δfill hedefli)…")
    df = build_features(panel, consumption, catchment)
    feat_cols = [f for f in ALL_FEATS if f in df.columns]
    log.info(f"Özellik: {len(feat_cols)}")

    cld = climatology_delta(df["delta_fill"], df["month"])
    log.info(f"Δfill klimatolojisi: {dict(cld.round(2))}")

    lgb_p=xgb_p={}
    if args.tune:
        hdf=df[df["delta_fill"].notna()].dropna(subset=feat_cols+["delta_fill"])
        Xa,ya=hdf[feat_cols].values,hdf["delta_fill"].values
        if HAS_LGB: lgb_p=optuna_tune(Xa,ya,args.n_trials,"lgb")
        if HAS_XGB: xgb_p=optuna_tune(Xa,ya,args.n_trials,"xgb")

    log.info("Modeller…")
    all_m = make_models(lgb_p, xgb_p)
    main_m = {k:v for k,v in all_m.items() if k not in ("lgb_q10","lgb_q90")}
    q10 = all_m.get("lgb_q10",
           make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.10,solver="highs")))
    q90 = all_m.get("lgb_q90",
           make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.90,solver="highs")))

    log.info("\nPurged CV…")
    cv_df = purged_cv(df, main_m, feat_cols, n_folds=args.cv_folds)
    cv_df.to_csv(out_dir/"cv_results.csv",index=False)
    plot_cv_bar(cv_df, fig_dir)
    best_name = (cv_df.groupby("model")["rmse_pp"].mean().idxmin()
                 if not cv_df.empty else ("lgb_dart" if HAS_LGB else "etr"))
    log.info(f"★ EN İYİ: {best_name.upper()}")

    hdf = df[df["delta_fill"].notna()].dropna(subset=feat_cols+["delta_fill"])
    ar1 = estimate_ar1(df, main_m[best_name], feat_cols)
    cal_n   = max(24, len(hdf)//5)
    tr_cal  = hdf.iloc[:-cal_n]; cal_df = hdf.iloc[-cal_n:]
    bm = main_m[best_name]
    bm.fit(tr_cal[feat_cols].values, tr_cal["delta_fill"].values)
    pred_d = bm.predict(cal_df[feat_cols].values)
    fill_start = float(tr_cal["fill_pct"].iloc[-1]) if not tr_cal["fill_pct"].isna().all() else 50.0
    pred_f = np.clip(fill_start + np.cumsum(pred_d), 0, 100)
    true_f = cal_df["fill_pct"].values
    conf = SplitConformal(args.coverage); conf.calibrate(true_f, pred_f)

    log.info("\nProjeksiyonlar…")
    Xfin=hdf[feat_cols].values; yfin=hdf["delta_fill"].values
    all_outs=[]
    ens_out=None

    scenario_params = [
        ("baz",      args.temp_rise,   args.precip_drop, args.damping),
    ] if not args.scenarios else [
        ("iyimser",  0.025, 0.001, 0.40),
        ("baz",      args.temp_rise, args.precip_drop, args.damping),
        ("kotumser", 0.055, 0.006, 1.00),
    ]

    for name, model in main_m.items():
        log.info(f"\n── {name.upper()} ──")
        t0=time.time()
        try:   model.fit(Xfin,yfin)
        except Exception as e: log.error(f"  Eğitim: {e}"); continue
        q10.fit(Xfin,yfin); q90.fit(Xfin,yfin)
        log.info(f"  Eğitim: {time.time()-t0:.1f}s")

        if name==best_name and not args.no_shap:
            run_shap(model,Xfin,feat_cols,fig_dir,name)

        for sc_name, tr, pd_drop, damp in scenario_params:
            sc_drift = compute_drifting_clim(
                sclim, last_obs.year, hist_trend,
                tr, pd_drop, damping=damp,
                mode=args.drift_mode, tau=args.drift_tau)
            out = simulate_v4(df, model, q10, q90, sclim, sc_drift,
                               cld, conf, ar1, start_date, end_date, feat_cols,
                               sigma_floor=args.sigma_floor,
                               sigma_hist_scale=args.sigma_hist_scale,
                               op_k=args.op_k)
            sc_tag = f"{name}_{sc_name}" if len(scenario_params)>1 else name
            out["model"]=sc_tag
            out.to_csv(out_dir/f"projection_{sc_tag}.csv",index=False)
            all_outs.append(out)
            cv_s=None
            if not cv_df.empty:
                mc=cv_df[cv_df["model"]==name]
                if not mc.empty: cv_s=mc[["rmse_pp","mape_pct","pearson_r"]].mean().to_dict()
            plot_v4(out, sc_drift, last_obs, hist_trend, sc_tag, cv_s, fig_dir, sclim)

    if len(all_outs)>=2:
        log.info("\\n── Ensemble ──")
        ps=pd.concat(all_outs).pivot_table(index="date",columns="model",values="fill_sim")
        pl=pd.concat(all_outs).pivot_table(index="date",columns="model",values="fill_lo")
        ph=pd.concat(all_outs).pivot_table(index="date",columns="model",values="fill_hi")
        ens=all_outs[0][["date","fill_pct"]].set_index("date")
        ens["fill_sim"]=ps.median(axis=1); ens["fill_lo"]=pl.min(axis=1)
        ens["fill_hi"]=ph.max(axis=1); ens=ens.reset_index(); ens["model"]="ensemble"
        ens.to_csv(out_dir/"projection_ensemble.csv",index=False)
        plot_v4(ens,drift_clim,last_obs,hist_trend,"ensemble",None,fig_dir,sclim)
        all_outs.append(ens)
        ens_out = ens

    if ens_out is None and all_outs:
        ens_out = all_outs[0]

    # ── İSKİ ay sonu kıyas (+ opsiyonel bias düzeltme)
    if ens_out is not None:
        # Ay sonu düzeltme (opsiyonel)
        ens_me = None
        if args.monthend_adjust:
            daily_path = ROOT / args.ibb_dir / (
                "İstanbul_Barajları_Günlük_Doluluk_Oranları_"
                "af0b3902-cfd9-4096-85f7-e2c3017e4f21.xlsx"
            )
            md = compute_month_end_delta(daily_path)
            if md is not None:
                ens_me = apply_month_end_adjustment(ens_out, md)
                ens_me["model"] = (
                    f"{ens_out['model'].iloc[0]}_monthend"
                    if "model" in ens_out.columns else "ensemble_monthend"
                )
                ens_me.to_csv(out_dir/f"projection_{ens_me['model'].iloc[0]}.csv", index=False)
                plot_v4(ens_me, drift_clim, last_obs, hist_trend,
                        ens_me["model"].iloc[0], None, fig_dir, sclim)

        # İSKİ kıyas için: aylık ortalama (aylık model için) ve ay sonu (ay-sonu düzeltme için)
        iski_obs_monthly = iski_monthly if iski_monthly is not None else fetch_iski_month_end()
        iski_obs_monthend = fetch_iski_month_end()
        if iski_obs_monthly is not None and not iski_obs_monthly.empty:
            # orijinal kıyas
            metrics_orig, cmp_orig = compare_iski(
                ens_out, iski_obs_monthly, out_dir/"comparison_iski_last12m.csv")

            # ay sonu düzeltmeli kıyas
            metrics_me = None
            cmp_me = None
            if ens_me is not None and iski_obs_monthend is not None and not iski_obs_monthend.empty:
                metrics_me, cmp_me = compare_iski(
                    ens_me, iski_obs_monthend, out_dir/"comparison_iski_last12m_monthend.csv")

            metrics_corr = None
            cmp_corr = None
            if not args.no_bias_correct:
                base_for_bias = ens_me if ens_me is not None else ens_out
                ens_bc = apply_bias_correction(base_for_bias, iski_obs_monthly,
                                               w_month=args.bias_month_weight)
                if ens_bc is not None:
                    bc_name = (
                        f"{base_for_bias['model'].iloc[0]}_biascorr"
                        if "model" in base_for_bias.columns else "ensemble_biascorr"
                    )
                    ens_bc["model"] = bc_name
                    ens_bc.to_csv(out_dir/f"projection_{bc_name}.csv", index=False)
                    plot_v4(ens_bc, drift_clim, last_obs, hist_trend,
                            bc_name, None, fig_dir, sclim)
                    metrics_corr, cmp_corr = compare_iski(
                        ens_bc, iski_obs_monthly, out_dir/"comparison_iski_last12m_biascorr.csv")
                    if metrics_corr:
                        log.info(
                            f"İSKİ kıyas (bias düzeltmeli): RMSE={metrics_corr['rmse']:.2f}pp  "
                            f"MAPE={metrics_corr['mape']:.1f}%  r={metrics_corr['pearson_r']:.3f}")

            # kıyas grafiği: orijinal (+ varsa bias)
            plot_iski_compare(
                cmp_orig, cmp_corr, fig_dir,
                tag="compare_iski_last12m",
                orig_label="v4 (aylık tahmin)",
                corr_label="v4 (bias düzeltme)",
                obs_label="İSKİ (aylık)"
            )
            if cmp_me is not None and not cmp_me.empty:
                plot_iski_compare(
                    cmp_me, None, fig_dir,
                    tag="compare_iski_last12m_monthend",
                    orig_label="v4 (ay-sonu tahmin)",
                    obs_label="İSKİ (ay-sonu)"
                )

            # Metrik özeti
            rows = []
            if metrics_orig:
                rows.append({"model":"v4_original", **metrics_orig})
            if metrics_me:
                rows.append({"model":"v4_monthend", **metrics_me})
            if metrics_corr:
                rows.append({"model":"v4_biascorr", **metrics_corr})
            if rows:
                pd.DataFrame(rows).to_csv(
                    out_dir/"comparison_iski_last12m_metrics.csv", index=False)
        else:
            log.info("İSKİ ay sonu verisi alınamadı; kıyas atlandı.")

    if all_outs:
        pd.concat(all_outs).to_csv(out_dir/"projection_all.csv",index=False)

    log.info("\\n"+"="*70); log.info("TAMAMLANDI"); log.info(f"Çıktılar → {out_dir}")
    if not cv_df.empty:
        bc=cv_df[cv_df["model"]==best_name][["rmse_pp","mape_pct","pearson_r"]].mean()
        log.info(f"En iyi: {best_name.upper()}  RMSE={bc['rmse_pp']:.2f}pp  "
                 f"MAPE={bc['mape_pct']:.1f}%  r={bc['pearson_r']:.3f}")
    log.info(f"Konformal: ±{conf.q:.2f}pp ({args.coverage*100:.0f}%)")
    if all_outs:
        last=next((o for o in reversed(all_outs) if o["model"].iloc[0]=="ensemble"),all_outs[0])
        v40=last.dropna(subset=["fill_sim"]).tail(12)["fill_sim"].mean()
        log.info(f"2040 yıllık ort. tahmin: {v40:.1f}%  "
                 f"(statik klim ort.: {sclim.mean():.1f}%)")
    log.info("="*70)


if __name__ == "__main__":
    main()
