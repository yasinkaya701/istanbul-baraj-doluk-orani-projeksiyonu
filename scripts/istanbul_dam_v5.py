#!/usr/bin/env python3
"""
İstanbul Baraj Doluluk Tahmini — v5 (Hibrit Fizik + ML Artık)
=============================================================
Neden önceki versiyonlar doluluk artırdı?
─────────────────────────────────────────
• fill_pct (seviye) veya Δfill (değişim) modellendi.
• 2020-2024 İstanbul ortalamanın üzerinde yağış aldı (La Niña etkileri).
• ML bu 4 yılı "normal" olarak öğrendi → uzak gelecekte pozitif sürükleme.
• Tüketim artışı (~%1.5/yıl, nüfus 2000-2024'te 8M→15M) modele girmedi.

v5 çözümü:
─────────────────────────────────────────
1. GR4J-tipi FIZIKSEL SU DENGESİ modeli:
   • SCS-CN akış (antecedent moisture'a duyarlı)
   • Gerçek tüketim + Gompertz büyüme projeksiyonu
   • Rezervuar buharlaşması (ET0 × Kpan × yüzey alanı)
   fill_fiziksel(t) = clip(fill(t-1) + [Q_in - Q_out - E] / V_max × 100, 0, 100)

2. ML sadece ARTIĞI öğrenir:
   artık = fill_gözlenen - fill_fiziksel
   Bu artık:
   • Sıfır ortalama (fizik trendi taşıdığı için)
   • ±5-8 pp → fill ±20-30 pp'ye kıyasla öğrenmesi çok kolay
   • Uzun vadeli sürükleme matematiksel olarak imkânsız

3. 100 yıllık Kandilli verisi → uzun dönem yağış trendi
   • 1911-2023 veri → 1980 sonrası kırılma noktası
   • Çok daha güvenilir trend tahmini (24 yıl yerine 50+ yıl)

4. Tüketim büyüme modeli (Gompertz):
   • IBB gerçek tüketim verisinden kalibre edilir
   • Nüfus platoya ulaştıkça (2035-2040) talep büyümesi yavaşlar
   • Per-capita verimlilik artışı da modele dahildir

5. SSP senaryo fanı (3 kol):
   • SSP1-2.6 (iyimser), SSP2-4.5 (baz), SSP5-8.5 (kötümser)
   • Fizik modeli senaryo değerleriyle çalışır

Kurulum:
    pip install lightgbm xgboost optuna shap statsmodels openpyxl scipy

Kullanım:
    python istanbul_dam_v5.py
    python istanbul_dam_v5.py --scenarios        # 3 SSP senaryosu
    python istanbul_dam_v5.py --tune             # Optuna optimizasyonu
    python istanbul_dam_v5.py --calibrate-cn     # CN parametresini kalibre et
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
from scipy import stats, optimize
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

try:
    import lightgbm as lgb; HAS_LGB = True
except Exception: HAS_LGB = False
try:
    import xgboost as xgb; HAS_XGB = True
except Exception: HAS_XGB = False
try:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING); HAS_OPTUNA = True
except Exception: HAS_OPTUNA = False
try:
    import shap; HAS_SHAP = True
except Exception: HAS_SHAP = False
try:
    from statsmodels.tsa.seasonal import STL; HAS_STL = True
except Exception: HAS_STL = False

warnings.filterwarnings("ignore")
ROOT = Path("/Users/yasinkaya/Hackhaton")
EPS  = 1e-6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("dam_v5")

# İstanbul baraj sistemi sabitleri (kalibrasyon ile değiştirilebilir)
DAM_CAPACITY_MCM   = 918.0    # Milyon m³ toplam kapasite (2024)
CATCHMENT_KM2      = 4985.0   # Toplam havza km² (7 büyük baraj)
RESERVOIR_AREA_KM2 = 42.0     # Ortalama su yüzeyi km²
CN_BASE            = 72.0     # SCS eğri numarası (AMC-II, karma arazi)
KPAN               = 0.75     # Buharlaşma tavası katsayısı


# ══════════════════════════════════════════════════════════════════════════════
# § 1  Metrikler
# ══════════════════════════════════════════════════════════════════════════════

def rmse(yt, yp): return float(np.sqrt(mean_squared_error(yt, yp)))
def mape(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp); m = yt > 1
    return float(mean_absolute_percentage_error(yt[m], yp[m])*100) if m.sum() else float("nan")
def pearson_r(yt, yp):
    return float(np.corrcoef(yt, yp)[0,1]) if len(yt) >= 3 else float("nan")
def nse(yt, yp):
    """Nash-Sutcliffe Efficiency — hidroloji standardı."""
    yt = np.asarray(yt); yp = np.asarray(yp)
    return float(1 - np.sum((yt-yp)**2) / np.sum((yt-np.mean(yt))**2))


# ══════════════════════════════════════════════════════════════════════════════
# § 2  SCS-CN Yüzey Akışı Modeli
# ══════════════════════════════════════════════════════════════════════════════

def cn_amc_adjust(cn_ii: float, p5_mm: float) -> float:
    """
    AMC (Antecedent Moisture Condition) sınıfına göre CN düzeltmesi.
    p5_mm: son 5 günlük yağış (mm)
    AMC-I  (<12.7 mm): kuru toprak, düşük akış
    AMC-II (12.7-27.9): normal koşullar
    AMC-III (>27.9 mm): nemli toprak, yüksek akış

    SCS TR-55 formülasyonu.
    """
    if p5_mm < 12.7:
        # AMC-I: CN_I
        return cn_ii * 4.2 / (10 - 0.058 * cn_ii)
    elif p5_mm > 27.9:
        # AMC-III: CN_III
        return cn_ii * 23.0 / (10 + 0.13 * cn_ii)
    else:
        return cn_ii


def scs_cn_runoff(rain_mm: float, cn: float) -> float:
    """
    SCS-CN yüzey akışı (mm).
    Q = (P - 0.2S)² / (P + 0.8S)  P > 0.2S için, diğer durumlarda 0
    S = 25400/CN - 254 (mm)
    """
    if rain_mm <= 0 or cn <= 0:
        return 0.0
    s = 25400.0 / cn - 254.0
    ia = 0.2 * s  # başlangıç kayıpları (initial abstraction)
    if rain_mm <= ia:
        return 0.0
    return (rain_mm - ia)**2 / (rain_mm + 0.8*s)


def compute_monthly_runoff(
    rain_mm:    pd.Series,
    cn_base:    float = CN_BASE,
    catchment:  float = CATCHMENT_KM2,
) -> pd.Series:
    """
    Aylık havza akışını MCM olarak hesapla.
    5 günlük antecedent yağış → AMC → CN → SCS runoff.
    """
    r = rain_mm.fillna(0).values
    runoff_mm = np.zeros(len(r))
    p5 = np.zeros(len(r))

    for i in range(len(r)):
        # 5 günlük (aylık çözünürlükte ≈ önceki ay)
        p5[i] = r[i-1] if i > 0 else 0.0
        cn = cn_amc_adjust(cn_base, p5[i])
        runoff_mm[i] = scs_cn_runoff(r[i], cn)

    # mm → MCM: runoff_mm × catchment_km² × 1e6 m²/km² × 0.001 m/mm × 1e-6 MCM/m³
    runoff_mcm = runoff_mm * catchment * 1e6 * 0.001 * 1e-6
    return pd.Series(runoff_mcm, index=rain_mm.index, name="runoff_mcm")


# ══════════════════════════════════════════════════════════════════════════════
# § 3  Tüketim büyüme modeli (Gompertz)
# ══════════════════════════════════════════════════════════════════════════════

def fit_consumption_trend(
    consumption: pd.Series,   # aylık tüketim (m³)
    years:       pd.Series,   # yıl (float)
) -> dict:
    """
    Gompertz büyüme eğrisi: C(t) = K × exp(-exp(-r(t-t0)))
    K = asimptotik kapasite (plato)
    r = büyüme hızı
    t0 = dönüm noktası yılı

    Neden Gompertz?
    • İstanbul nüfusu platoya yaklaşıyor (tahmin: 2035-2040)
    • Per-capita tüketim verimlilikle azalıyor
    • Lineer/üstel büyüme 2040'ta gerçekçi olmayan değerler verir
    """
    if len(consumption) < 10:
        log.warning("  Tüketim modeli: yetersiz veri, doğrusal kullanılıyor")
        m, b = np.polyfit(years.values, consumption.values, 1)
        return {"type": "linear", "slope": m, "intercept": b, "r2": 0.0}

    try:
        # Gompertz fit
        def gompertz(t, K, r, t0):
            return K * np.exp(-np.exp(-r * (t - t0)))

        p0 = [consumption.max() * 1.5, 0.1, float(years.mean())]
        bounds = ([consumption.max(), 0.01, years.min()],
                  [consumption.max() * 5, 1.0, years.max() + 20])
        popt, _ = optimize.curve_fit(
            gompertz, years.values, consumption.values,
            p0=p0, bounds=bounds, maxfev=5000)

        K, r, t0 = popt
        pred = gompertz(years.values, *popt)
        ss_res = np.sum((consumption.values - pred)**2)
        ss_tot = np.sum((consumption.values - consumption.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        log.info(f"  Gompertz tüketim: K={K:.2e} m³  r={r:.3f}  t0={t0:.1f}  R²={r2:.3f}")
        return {"type": "gompertz", "K": K, "r": r, "t0": t0, "r2": r2}

    except Exception as e:
        log.warning(f"  Gompertz fit başarısız ({e}), doğrusal kullanılıyor")
        m, b, *_ = stats.linregress(years.values, consumption.values)
        return {"type": "linear", "slope": m, "intercept": b, "r2": 0.0}


def predict_consumption(model: dict, year: float) -> float:
    """Yıl için aylık tüketim tahmin et (m³)."""
    if model["type"] == "gompertz":
        K, r, t0 = model["K"], model["r"], model["t0"]
        return float(K * np.exp(-np.exp(-r * (year - t0))))
    else:
        return float(model["slope"] * year + model["intercept"])


# ══════════════════════════════════════════════════════════════════════════════
# § 4  Fiziksel Su Dengesi Modeli (GR4J-tipi)
# ══════════════════════════════════════════════════════════════════════════════

class PhysicalWaterBalance:
    """
    GR4J-ilhamlı aylık rezervuar su dengesi modeli.

    Durum değişkeni: fill_pct ∈ [0, 100]

    Adım:
      1. SCS-CN ile yüzey akışı (MCM)
      2. Üretim deposu: toprak nemi doygunluğu (GR4J S1)
      3. Aktif depolama → baraj rezervuarı
      4. Tüketim çıkışı (gerçek IBB verisi veya Gompertz projeksiyonu)
      5. Rezervuar buharlaşması (ET0 × Kpan × Alan)
      6. fill(t) = clip(fill(t-1) + net_inflow/capacity × 100, 0, 100)

    Parametreler (kalibrasyon ile bulunabilir):
      cn_base: SCS eğri numarası (55-85 arası, İstanbul için ~72)
      x1:      üretim deposu kapasitesi (mm) — toprak nemi tutma
      routing: yönlendirme katsayısı (akışın ne kadarı barajlara ulaşır)
    """
    def __init__(self,
                 cn_base:     float = CN_BASE,
                 x1:          float = 200.0,   # üretim deposu (mm)
                 routing:     float = 0.85,    # akışın barajlara ulaşma oranı
                 capacity_mcm:float = DAM_CAPACITY_MCM,
                 catchment_km2:float= CATCHMENT_KM2,
                 reservoir_km2:float= RESERVOIR_AREA_KM2,
                 kpan:        float = KPAN):
        self.cn_base      = cn_base
        self.x1           = x1
        self.routing      = routing
        self.cap          = capacity_mcm
        self.catch        = catchment_km2
        self.res_area_m2  = reservoir_km2 * 1e6
        self.kpan         = kpan
        self.soil_moisture = x1 * 0.5   # başlangıç toprak nemi (mm)

    def _production_store(self, rain_mm: float) -> tuple[float, float]:
        """
        GR4J üretim deposu.
        Toprak dolmadan runoff olmaz; aşım runoff'a geçer.
        Döndürür: (actual_runoff_mm, new_soil_moisture)
        """
        s = self.soil_moisture
        x1 = self.x1
        # Tanh formülasyonu (GR4J'den)
        tanh_val = np.tanh(rain_mm / x1) if x1 > 0 else 0
        ps = x1 * (1 - (s/x1)**2) * tanh_val / (1 + (s/x1) * tanh_val) \
             if s < x1 else 0.0
        new_s = min(x1, s + ps)
        excess = rain_mm - ps   # toprağa gitmeyen kısım → SCS ile akışa geçer
        return max(0.0, excess), new_s

    def step(self,
             rain_mm:       float,
             et0_mm:        float,
             consumption_mcm:float,
             fill_prev_pct: float) -> tuple[float, dict]:
        """
        Bir aylık adım.
        Döndürür: (fill_new_pct, diagnostics_dict)
        """
        # 1. Üretim deposu (toprak nemi)
        excess_mm, self.soil_moisture = self._production_store(rain_mm)

        # 2. SCS-CN akışı (aşım miktarından)
        p5 = rain_mm  # aylık çözünürlükte önceki ayı proxy olarak kullan
        cn = cn_amc_adjust(self.cn_base, p5)
        runoff_mm = scs_cn_runoff(excess_mm, cn) if excess_mm > 0 else 0.0

        # 3. Havza → rezervuar (mm → MCM, yönlendirme katsayısı ile)
        inflow_mcm = runoff_mm * self.catch * 1e6 * 0.001 * 1e-6 * self.routing

        # 4. Rezervuar buharlaşması (mm → MCM)
        evap_mm  = et0_mm * self.kpan
        evap_mcm = evap_mm * self.res_area_m2 * 0.001 * 1e-6

        # 5. Net değişim
        net_mcm = inflow_mcm - consumption_mcm - evap_mcm

        # 6. fill güncelle
        fill_new = float(np.clip(
            fill_prev_pct + net_mcm / self.cap * 100.0, 0.0, 100.0))

        diag = {
            "runoff_mm": runoff_mm, "inflow_mcm": inflow_mcm,
            "evap_mcm": evap_mcm, "consumption_mcm": consumption_mcm,
            "net_mcm": net_mcm, "cn": cn,
        }
        return fill_new, diag

    def run(self,
            rain: pd.Series,
            et0:  pd.Series,
            consumption_mcm: pd.Series,
            fill_init: float = 50.0) -> pd.DataFrame:
        """Tüm dönemi çalıştır, sonuçları döndür."""
        fills = [fill_init]
        diags = []
        for i in range(len(rain)):
            f, d = self.step(
                rain.iloc[i], et0.iloc[i],
                consumption_mcm.iloc[i], fills[-1])
            fills.append(f)
            diags.append(d)
        out = pd.DataFrame(diags, index=rain.index)
        out["fill_physical"] = fills[1:]
        return out

    def calibrate(self,
                  rain: pd.Series,
                  et0:  pd.Series,
                  consumption_mcm: pd.Series,
                  fill_obs: pd.Series,
                  fill_init: float = 50.0) -> dict:
        """
        Nelder-Mead optimizasyonu ile cn_base, x1, routing kalibre et.
        Hedef: Nash-Sutcliffe Efficiency maksimize et (= RMSE minimize et).
        """
        log.info("  Fizik modeli kalibre ediliyor (Nelder-Mead)…")
        valid = fill_obs.dropna()
        if len(valid) < 24:
            log.warning("  Yetersiz gözlem, kalibrasyon atlanıyor.")
            return {"cn_base": self.cn_base, "x1": self.x1, "routing": self.routing}

        def objective(params):
            cn, x1, rout = params
            # Daha gerçekçi dar aralıklar (İstanbul havzaları için)
            # CN: 68–78, x1: 120–300 mm, routing: 0.70–0.90
            if cn < 68 or cn > 78 or x1 < 120 or x1 > 300 or rout < 0.70 or rout > 0.90:
                return 1e6
            wb = PhysicalWaterBalance(cn, x1, rout,
                                       self.cap, self.catch,
                                       RESERVOIR_AREA_KM2, self.kpan)
            try:
                res = wb.run(rain, et0, consumption_mcm, fill_init)
                obs = fill_obs.loc[res.index].dropna()
                pred = res.loc[obs.index, "fill_physical"]
                return -nse(obs.values, pred.values)
            except Exception:
                return 1e6

        x0 = [self.cn_base, self.x1, self.routing]
        result = optimize.minimize(objective, x0, method="Nelder-Mead",
                                   options={"maxiter": 500, "xatol": 0.5, "fatol": 0.01})

        self.cn_base, self.x1, self.routing = result.x
        best_nse = -result.fun
        log.info(f"  Kalibrasyon: CN={self.cn_base:.1f}  x1={self.x1:.1f}  "
                 f"routing={self.routing:.3f}  NSE={best_nse:.3f}")
        return {"cn_base": self.cn_base, "x1": self.x1,
                "routing": self.routing, "nse": best_nse}


# ══════════════════════════════════════════════════════════════════════════════
# § 5  100 Yıllık Trend Analizi
# ══════════════════════════════════════════════════════════════════════════════

def load_kandilli_precipitation(base: Path) -> pd.DataFrame:
    """
    Kandilli 1911-2023 yıllık yağış verisi.
    Desteklenen formatlar:
    - Wide format (kolonlar yıl, satırlar ay)
    - Long format (year, rain_mm)
    """
    candidates = [
        base / "new data/Veriler_H-3/Yaßçü/1911-2023.xlsx",
        base / "new data/Veriler_H-3/Yaßçü/Yaßçü 1911-Hepsi.xlsx",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_excel(path, engine="openpyxl")

            # 1) Wide format: kolonlar yıl (int veya string)
            year_cols = []
            for c in df.columns:
                try:
                    y = int(str(c))
                    if 1900 <= y <= 2030:
                        year_cols.append(c)
                except Exception:
                    continue
            if year_cols:
                wide = df[year_cols].apply(pd.to_numeric, errors="coerce")
                annual = wide.sum(axis=0, skipna=True)
                out = pd.DataFrame({"year": [int(str(c)) for c in year_cols],
                                    "rain_mm": annual.values})
                out = out.dropna()
                out = out.sort_values("year").reset_index(drop=True)
                log.info(f"  Kandilli yağış (wide): {len(out)} yıl "
                         f"({out['year'].min()}–{out['year'].max()})")
                return out

            # 2) Long format: year, rain_mm
            num = df.select_dtypes(include=np.number).columns.tolist()
            if len(num) >= 2:
                out = df.rename(columns={num[0]: "year", num[1]: "rain_mm"})
                out = out[["year","rain_mm"]].dropna()
                out["year"] = out["year"].astype(int)
                out = out[(out["year"] >= 1900) & (out["year"] <= 2030)]
                log.info(f"  Kandilli yağış (long): {len(out)} yıl "
                         f"({out['year'].min()}–{out['year'].max()})")
                return out.sort_values("year").reset_index(drop=True)
        except Exception as e:
            log.warning(f"  Kandilli {path.name}: {e}")
    log.warning("  Kandilli verisi yüklenemedi, boş döndürülüyor.")
    return pd.DataFrame(columns=["year","rain_mm"])


def analyse_long_term_trend(df_kandilli: pd.DataFrame) -> dict:
    """
    100 yıllık yağış trendini analiz et.
    Kırılma noktası tespiti (Pettitt testi) + iki dönem OLS.
    """
    if df_kandilli.empty or len(df_kandilli) < 20:
        log.warning("  Uzun dönem trend: yetersiz veri.")
        return {"slope_full": 0.0, "slope_recent": 0.0,
                "breakpoint": 1980, "p_value": 1.0}

    x = df_kandilli["year"].values.astype(float)
    y = df_kandilli["rain_mm"].values

    # Tam dönem OLS
    s_full, b_full, r_full, p_full, se_full = stats.linregress(x, y)

    # Pettitt-benzeri kırılma noktası: en büyük Mann-Whitney U değişimi
    n = len(y)
    u_vals = np.zeros(n)
    for k in range(1, n):
        u_vals[k] = u_vals[k-1] + np.sum(np.sign(y[k] - y[:k]))
    bp_idx = int(np.argmax(np.abs(u_vals)))
    bp_year = int(x[bp_idx]) if bp_idx > 0 else 1980

    # Son dönem (kırılma sonrası) OLS
    mask_r = x >= bp_year
    if mask_r.sum() >= 10:
        s_rec, *_ = stats.linregress(x[mask_r], y[mask_r])
    else:
        s_rec = s_full

    log.info(f"  Uzun dönem trend: {s_full:+.2f} mm/yr (tam dönem)  "
             f"{s_rec:+.2f} mm/yr ({bp_year}+)  p={p_full:.3f}")
    log.info(f"  Kırılma noktası: {bp_year}")

    return {
        "slope_full":   float(s_full),
        "slope_recent": float(s_rec),
        "breakpoint":   bp_year,
        "p_value":      float(p_full),
        "mean_rain":    float(y.mean()),
        "mean_recent":  float(y[mask_r].mean()) if mask_r.sum() > 0 else float(y.mean()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 6  Senaryo iklim projeksiyonları
# ══════════════════════════════════════════════════════════════════════════════

SSP_SCENARIOS = {
    "ssp126": {   # İyimser
        "label":        "SSP1-2.6 (İyimser)",
        "color":        "#185FA5",
        "temp_per_yr":  0.020,   # °C/yr (2025'ten)
        "precip_delta": -0.0015, # oran/yr (−0.15%/yr)
        "demand_growth":0.008,   # yıllık talep artışı (%0.8)
    },
    "ssp245": {   # Baz
        "label":        "SSP2-4.5 (Baz)",
        "color":        "#D55E00",
        "temp_per_yr":  0.040,
        "precip_delta": -0.003,
        "demand_growth":0.012,
    },
    "ssp585": {   # Kötümser
        "label":        "SSP5-8.5 (Kötümser)",
        "color":        "#E24B4A",
        "temp_per_yr":  0.065,
        "precip_delta": -0.006,
        "demand_growth":0.015,
    },
}

def apply_scenario_to_row(row: pd.Series, base_year: int,
                           scenario: dict, cons_model: dict) -> pd.Series:
    """Bir aya ait iklim değerlerini senaryo ile değiştir."""
    row = row.copy()
    dt = float(row["date"].year - base_year)
    if dt <= 0:
        return row

    # Yağış değişimi (oransal)
    precip_factor = 1.0 + scenario["precip_delta"] * dt
    precip_factor = max(0.5, precip_factor)   # %50'nin altına düşme
    row["rain_mm"] = float(row["rain_mm"]) * precip_factor
    if "catchment_rain_mm" in row.index:
        row["catchment_rain_mm"] = float(row.get("catchment_rain_mm",
                                                   row["rain_mm"])) * precip_factor

    # Sıcaklık artışı → ET0 artışı (~2% / °C Penman-Monteith)
    delta_t = scenario["temp_per_yr"] * dt
    row["t_mean_c"] = float(row["t_mean_c"]) + delta_t
    row["et0_mm_month"] = float(row["et0_mm_month"]) * (1 + 0.02 * delta_t)

    # Tüketim projeksiyonu
    year_float = float(row["date"].year) + (row["date"].month - 1) / 12.0
    row["consumption_mcm"] = predict_consumption(cons_model, year_float) * 1e-6 \
                              * (1 + scenario["demand_growth"] * dt)

    return row


# ══════════════════════════════════════════════════════════════════════════════
# § 7  Özellik mühendisliği — artık hedefli
# ══════════════════════════════════════════════════════════════════════════════

ALL_FEATS = [
    # Ham iklim
    "rain_mm", "et0_mm_month", "t_mean_c", "rh_mean_pct",
    "vpd_kpa_mean", "climate_balance_mm",
    # Havza
    "catchment_rain_mm", "api",
    # Fizik model çıktıları (artık için kritik özellikler)
    "wb_runoff_mm", "wb_inflow_mcm", "wb_evap_mcm",
    "wb_net_mcm", "wb_cn",
    # Kuraklık indeksleri
    "spi_3", "spi_6", "spi_12", "pdsi_proxy",
    # Kümülatif denge
    "cum_balance_6", "cum_balance_12",
    # Sezon
    "month_sin", "month_cos", "quarter_sin", "quarter_cos",
    # Delta iklim
    "rain_delta", "t_delta",
    # MA iklim
    "rain_ma3", "rain_ma6", "rain_ma12",
    "balance_ma3", "balance_ma6",
    # Fill seviyesi (artık bağlamı için)
    "lag1_fill", "lag3_fill", "lag12_fill",
    # Artık lag'ları (ML artığının kendi belleği)
    "lag1_resid", "lag2_resid", "lag3_resid", "lag6_resid", "lag12_resid",
    "resid_ma3", "resid_ma6",
    # Fiziksel tahmin lag'ı
    "lag1_wb", "lag3_wb",
    # STL
    "stl_trend", "stl_resid_fill",
]

def _api(r, k=0.92):
    a = np.zeros(len(r)); rv = r.fillna(0).values
    for i in range(1,len(rv)): a[i] = k*a[i-1]+rv[i]
    return pd.Series(a, index=r.index)

def _spi(r, w):
    rm = r.rolling(w, min_periods=max(2,w//2)).mean()
    rs = r.rolling(w, min_periods=max(2,w//2)).std().replace(0,np.nan)
    return ((r-rm)/rs).fillna(0)

def _pdsi_proxy(rain, et0, window=12):
    """
    Palmer Drought Severity Index (basitleştirilmiş).
    PDSI ≈ (yağış - ET0) kümülatif anomalisi / standart sapma
    """
    balance = rain - et0
    rm = balance.rolling(window, min_periods=window//2).mean()
    rs = balance.rolling(window, min_periods=window//2).std().replace(0, np.nan)
    return ((balance - rm) / rs).fillna(0)

def _stl(s, period=12):
    n = len(s); blank = pd.DataFrame({"stl_trend":np.zeros(n),"stl_resid_fill":np.zeros(n)},index=s.index)
    if not HAS_STL or s.dropna().shape[0] < period*2: return blank
    try:
        res = STL(s.fillna(s.median()), period=period, robust=True).fit()
        return pd.DataFrame({"stl_trend":res.trend.values,"stl_resid_fill":res.resid.values},index=s.index)
    except: return blank

def build_features(df: pd.DataFrame,
                   wb_results: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Özellik mühendisliği.
    wb_results: fizik modelinden fill_physical, runoff_mm, vb.
    Hedef: artık = fill_obs - fill_physical
    """
    d = df.copy().sort_values("date").reset_index(drop=True)

    if "weighted_total_fill" in d.columns and "fill_pct" not in d.columns:
        d["fill_pct"] = d["weighted_total_fill"] * 100.0

    # VPD
    es = 0.6108 * np.exp(17.27 * d["t_mean_c"] / (d["t_mean_c"] + 237.3))
    vpd_calc = es * (1 - d["rh_mean_pct"] / 100)
    d["vpd_kpa_mean"] = d.get("vpd_kpa_mean", vpd_calc).fillna(vpd_calc)
    d["climate_balance_mm"] = d["rain_mm"] - d["et0_mm_month"]
    d["month"]   = d["date"].dt.month
    d["quarter"] = d["date"].dt.quarter

    # Havza yağışı
    if "catchment_rain_mm" not in d.columns:
        d["catchment_rain_mm"] = d["rain_mm"]
    d["catchment_rain_mm"] = d["catchment_rain_mm"].fillna(d["rain_mm"])

    # API
    d["api"] = _api(d["catchment_rain_mm"]).values

    # Kuraklık
    for w, n in [(3,"spi_3"),(6,"spi_6"),(12,"spi_12")]:
        d[n] = _spi(d["catchment_rain_mm"], w).values
    d["pdsi_proxy"]    = _pdsi_proxy(d["catchment_rain_mm"], d["et0_mm_month"]).values
    d["cum_balance_6"] = d["climate_balance_mm"].rolling(6, min_periods=1).sum()
    d["cum_balance_12"]= d["climate_balance_mm"].rolling(12, min_periods=1).sum()

    # Delta, MA
    d["rain_delta"] = d["rain_mm"].diff(1).fillna(0)
    d["t_delta"]    = d["t_mean_c"].diff(1).fillna(0)
    for w, n in [(3,"rain_ma3"),(6,"rain_ma6"),(12,"rain_ma12")]:
        d[n] = d["rain_mm"].rolling(w, min_periods=1).mean()
    for w, n in [(3,"balance_ma3"),(6,"balance_ma6")]:
        d[n] = d["climate_balance_mm"].rolling(w, min_periods=1).mean()

    # Mevsim
    d["month_sin"]   = np.sin(2*np.pi*d["month"]/12)
    d["month_cos"]   = np.cos(2*np.pi*d["month"]/12)
    d["quarter_sin"] = np.sin(2*np.pi*d["quarter"]/4)
    d["quarter_cos"] = np.cos(2*np.pi*d["quarter"]/4)

    # Fill lag'ları
    fp = d.get("fill_pct", pd.Series(np.nan, index=d.index))
    for lag, n in [(1,"lag1_fill"),(3,"lag3_fill"),(12,"lag12_fill")]:
        d[n] = fp.shift(lag)

    # STL
    stlf = _stl(fp)
    d["stl_trend"]      = stlf["stl_trend"].values
    d["stl_resid_fill"] = stlf["stl_resid_fill"].values

    # Fizik modeli çıktılarını ekle
    if wb_results is not None:
        for col in ["fill_physical","wb_runoff_mm","wb_inflow_mcm",
                    "wb_evap_mcm","wb_net_mcm","wb_cn"]:
            if col in wb_results.columns:
                d = d.merge(wb_results[["date",col]].rename(
                    columns={col:col}), on="date", how="left")

        # Artık (ML'in öğreneceği şey)
        if "fill_physical" in d.columns and "fill_pct" in d.columns:
            d["residual"] = d["fill_pct"] - d["fill_physical"]
        else:
            d["residual"] = np.nan

        # Artık lag'ları
        resid = d.get("residual", pd.Series(np.nan, index=d.index))
        for lag, n in [(1,"lag1_resid"),(2,"lag2_resid"),(3,"lag3_resid"),
                        (6,"lag6_resid"),(12,"lag12_resid")]:
            d[n] = resid.shift(lag)
        d["resid_ma3"] = resid.shift(1).rolling(3, min_periods=1).mean()
        d["resid_ma6"] = resid.shift(1).rolling(6, min_periods=1).mean()

        # Fizik tahmin lag'ları
        if "fill_physical" in d.columns:
            wp = d["fill_physical"]
            d["lag1_wb"] = wp.shift(1)
            d["lag3_wb"] = wp.shift(3)
        else:
            d["lag1_wb"] = d["lag1_fill"]
            d["lag3_wb"] = d["lag3_fill"]
    else:
        # Fizik yok: geçici sıfırlar
        for c in ["fill_physical","wb_runoff_mm","wb_inflow_mcm","wb_evap_mcm",
                  "wb_net_mcm","wb_cn","residual",
                  "lag1_resid","lag2_resid","lag3_resid","lag6_resid","lag12_resid",
                  "resid_ma3","resid_ma6","lag1_wb","lag3_wb"]:
            d[c] = 0.0

    return d


# ══════════════════════════════════════════════════════════════════════════════
# § 8  Veri yükleme
# ══════════════════════════════════════════════════════════════════════════════

def _read_ibb_xlsx(path: Path, date_kws: list, val_kws: list) -> pd.DataFrame:
    """IBB Excel dosyasından date+value çift yükle."""
    if not path.exists():
        log.warning(f"  {path.name} bulunamadı")
        return pd.DataFrame()
    try:
        raw = pd.read_excel(path, engine="openpyxl")
        dc = next((c for c in raw.columns
                   if any(k in str(c).lower() for k in date_kws)), None)
        vc = next((c for c in raw.columns
                   if any(k in str(c).lower() for k in val_kws)), None)
        if not dc:
            log.warning(f"  {path.name}: tarih kolonu bulunamadı")
            return pd.DataFrame()
        if not vc:
            num_cols = raw.select_dtypes(include=np.number).columns
            vc = num_cols[0] if len(num_cols) else None
        if not vc:
            return pd.DataFrame()
        raw[dc] = pd.to_datetime(raw[dc], dayfirst=True, errors="coerce")
        raw[vc] = pd.to_numeric(raw[vc], errors="coerce")
        raw = raw.dropna(subset=[dc, vc])
        raw = raw.rename(columns={dc:"date", vc:"value"})
        raw["date"] = raw["date"].dt.to_period("M").dt.to_timestamp()
        return raw.groupby("date")["value"].mean().reset_index()
    except Exception as e:
        log.warning(f"  {path.name}: {e}")
        return pd.DataFrame()

def load_ibb_daily_consumption_and_rain(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Günlük İBB dosyasından:
      - İstanbul günlük tüketim (m³/gün) → aylık toplam m³
      - Baraj yağışı (çok kolon) → aylık toplam mm (kolon ortalaması)
    """
    if not path.exists():
        log.warning(f"  {path.name} bulunamadı")
        return pd.DataFrame(), pd.DataFrame()
    try:
        raw = pd.read_excel(path, engine="openpyxl")
        date_col = next((c for c in raw.columns if "tarih" in str(c).lower()), None)
        cons_col = next((c for c in raw.columns
                         if "tüketim" in str(c).lower() or "tuketim" in str(c).lower()), None)
        if not date_col:
            log.warning(f"  {path.name}: tarih kolonu bulunamadı")
            return pd.DataFrame(), pd.DataFrame()
        raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")

        # Tüketim
        cons_df = pd.DataFrame()
        if cons_col:
            raw[cons_col] = pd.to_numeric(raw[cons_col], errors="coerce")
            cons = raw.dropna(subset=[date_col, cons_col]).copy()
            cons = cons.rename(columns={date_col:"date", cons_col:"consumption_m3"})
            cons["date"] = cons["date"].dt.to_period("M").dt.to_timestamp()
            cons_df = cons.groupby("date")["consumption_m3"].sum().reset_index()

        # Yağış: date + consumption dışında kalan sayısal kolonların ortalaması
        num_cols = raw.select_dtypes(include=np.number).columns.tolist()
        rain_cols = [c for c in num_cols if c not in [cons_col]]
        rain_df = pd.DataFrame()
        if rain_cols:
            tmp = raw[[date_col] + rain_cols].copy()
            tmp["rain_mm"] = tmp[rain_cols].mean(axis=1, skipna=True)
            tmp = tmp.dropna(subset=[date_col, "rain_mm"])
            tmp = tmp.rename(columns={date_col:"date"})
            tmp["date"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
            rain_df = tmp.groupby("date")["rain_mm"].sum().reset_index()

        return cons_df, rain_df
    except Exception as e:
        log.warning(f"  {path.name}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def load_all_data(panel_path: Path, climate_path: Path | None,
                  ibb_dir: Path) -> tuple:
    """
    Döndürür: (panel_df, consumption_monthly_m3, catchment_rain_monthly_mm)
    """
    log.info(f"Panel: {panel_path.name}")
    panel = pd.read_csv(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    if "weighted_total_fill" in panel.columns:
        panel["fill_pct"] = panel["weighted_total_fill"] * 100.0

    # İklim projeksiyonu ekle
    if climate_path and climate_path.exists():
        clim = pd.read_csv(climate_path)
        clim["date"] = pd.to_datetime(clim["date"])
        if "precip_mm_month" in clim.columns:
            clim = clim.rename(columns={"precip_mm_month":"rain_mm"})
        fut = clim[clim["date"].dt.year > panel["date"].dt.year.max()]
        panel["_m"] = panel["date"].dt.month
        cmeans = {c: panel.groupby("_m")[c].mean()
                  for c in ["rain_mm","et0_mm_month","t_mean_c","rh_mean_pct","pressure_kpa"]
                  if c in panel.columns}
        full = pd.DataFrame({"date": pd.date_range(
                                panel["date"].min(),"2040-12-01",freq="MS")})
        full = full.merge(panel.drop(columns=["_m"],errors="ignore"),on="date",how="left")
        fc = [c for c in ["rain_mm","et0_mm_month","t_mean_c","rh_mean_pct"] if c in fut.columns]
        full = full.merge(fut[["date"]+fc], on="date", how="left", suffixes=("","_c"))
        for c in fc:
            cc = f"{c}_c"
            if cc in full.columns:
                full[c] = full[c].where(full[cc].isna(), full[cc])
                full.drop(columns=[cc], inplace=True)
        full["_m"] = full["date"].dt.month
        for c, mo in cmeans.items():
            full[c] = full[c].fillna(full["_m"].map(mo))
        full.drop(columns=["_m"], inplace=True)
        panel = full

    excl = {"fill_pct","weighted_total_fill"}
    num = [c for c in panel.select_dtypes(include=np.number).columns if c not in excl]
    panel[num] = panel[num].fillna(panel[num].mean())
    panel = panel.sort_values("date").reset_index(drop=True)
    log.info(f"Panel: {len(panel)} satır | {panel['date'].min().date()} – {panel['date'].max().date()}")

    # IBB günlük dosyadan tüketim + yağış
    cons, cr = load_ibb_daily_consumption_and_rain(
        ibb_dir/"İstanbul_Barajlarına_Düşen_Yağış_Ve_Günlük_Tüketim_Verileri_762b802e-c5f9-4175-a5c1-78b892d9764b.xlsx"
    )
    if not cons.empty:
        log.info(f"  Tüketim: {len(cons)} ay")
    if not cr.empty:
        cr = cr.rename(columns={"rain_mm":"catchment_rain_mm"})
        log.info(f"  Havza yağış: {len(cr)} ay")

    return panel, cons, cr


# ══════════════════════════════════════════════════════════════════════════════
# § 9  Model kataloğu
# ══════════════════════════════════════════════════════════════════════════════

def make_models(lgb_p=None, xgb_p=None):
    """Artık tahmin edecek ML modelleri."""
    m = {}
    if HAS_LGB:
        d = dict(boosting_type="dart", n_estimators=500, learning_rate=0.04,
                 num_leaves=31, max_depth=5, min_child_samples=10,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5,
                 reg_lambda=2.0, drop_rate=0.15, random_state=42,
                 n_jobs=-1, verbose=-1)
        if lgb_p: d.update(lgb_p)
        m["lgb_dart"] = lgb.LGBMRegressor(**d)
        m["lgb_q10"]  = lgb.LGBMRegressor(objective="quantile", alpha=0.10,
                                            boosting_type="gbdt", n_estimators=400,
                                            learning_rate=0.04, num_leaves=31,
                                            subsample=0.8, random_state=42,
                                            n_jobs=-1, verbose=-1)
        m["lgb_q90"]  = lgb.LGBMRegressor(objective="quantile", alpha=0.90,
                                            boosting_type="gbdt", n_estimators=400,
                                            learning_rate=0.04, num_leaves=31,
                                            subsample=0.8, random_state=42,
                                            n_jobs=-1, verbose=-1)
        log.info("  LGB DART + Q10/Q90")
    if HAS_XGB:
        d2 = dict(n_estimators=500, learning_rate=0.04, max_depth=4,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.5, reg_lambda=2.0,
                  random_state=42, n_jobs=-1, verbosity=0)
        if xgb_p: d2.update(xgb_p)
        m["xgb"] = xgb.XGBRegressor(**d2)
        log.info("  XGBoost")
    m["etr"] = ExtraTreesRegressor(n_estimators=300, max_features=0.6,
                                    min_samples_leaf=4, random_state=42, n_jobs=-1)
    log.info("  ETR")
    # Stacking
    base = [(k,v) for k,v in m.items() if k not in ("lgb_q10","lgb_q90")]
    if len(base) >= 2:
        m["stack"] = StackingRegressor(
            estimators=base,
            final_estimator=make_pipeline(StandardScaler(), Ridge(alpha=20.0)),
            cv=5, n_jobs=1)
        log.info("  Stacking")
    return m


# ══════════════════════════════════════════════════════════════════════════════
# § 10  Purged Walk-Forward CV
# ══════════════════════════════════════════════════════════════════════════════

def purged_cv(df, models, feat_cols, target="residual",
              n_folds=5, embargo=6, min_train=60):
    log.info(f"Purged CV (hedef={target}): {n_folds} fold, {embargo} ay embargo")
    hist = df[df[target].notna()].dropna(subset=feat_cols+[target])
    years = sorted(hist["date"].dt.year.unique())[-n_folds:]
    rows  = []
    for yr in years:
        ts = pd.Timestamp(f"{yr}-01-01")
        ee = ts - pd.DateOffset(months=embargo)
        tr = hist[hist["date"] < ee]
        te = hist[hist["date"].dt.year == yr]
        if len(tr) < min_train or te.empty: continue
        for name, model in models.items():
            if name in ("lgb_q10","lgb_q90","stack"): continue
            try:
                model.fit(tr[feat_cols].values, tr[target].values)
                pred_r = model.predict(te[feat_cols].values)
                # artık + fizik = fill tahmini
                if "fill_physical" in te.columns:
                    pred_f = np.clip(te["fill_physical"].values + pred_r, 0, 100)
                else:
                    pred_f = pred_r
                true_f = te["fill_pct"].values if "fill_pct" in te.columns else pred_f
                r = rmse(true_f, pred_f)
                rows.append({"model":name,"test_year":yr,
                             "rmse_pp":round(r,3),
                             "nse":round(nse(true_f,pred_f),3),
                             "pearson_r":round(pearson_r(true_f,pred_f),3)})
                log.info(f"  {yr}│{name:10s}│RMSE={r:.2f}pp  NSE={rows[-1]['nse']:.3f}")
            except Exception as e:
                log.error(f"  {yr}│{name}: {e}")
    cv = pd.DataFrame(rows)
    if not cv.empty:
        log.info(f"\n{cv.groupby('model')[['rmse_pp','nse','pearson_r']].mean().round(3).to_string()}\n")
    return cv


# ══════════════════════════════════════════════════════════════════════════════
# § 11  Split Conformal + AR(1)
# ══════════════════════════════════════════════════════════════════════════════

class SplitConformal:
    def __init__(self, cov=0.90): self.cov=cov; self.q=0.0
    def calibrate(self, yt, yp):
        r=np.abs(yt-yp); n=len(r); a=1-self.cov
        self.q=float(np.quantile(r, min(np.ceil((n+1)*(1-a))/n, 1.0)))
        log.info(f"  Konformal ±{self.q:.2f} pp ({self.cov*100:.0f}%)")
    def interval(self, yp):
        return np.clip(yp-self.q, -50, 50), np.clip(yp+self.q, -50, 50)


def estimate_ar1_resid(df, model, feat_cols, target="residual") -> dict:
    hist = df[df[target].notna()].dropna(subset=feat_cols+[target]).copy()
    if len(hist) < 24:
        return {"phi": 0.50, "sigma": {m: 2.5 for m in range(1,13)}}
    try:
        model.fit(hist[feat_cols].values, hist[target].values)
        resid = hist[target].values - model.predict(hist[feat_cols].values)
    except Exception:
        resid = hist[target].values
    phi = float(np.clip(np.corrcoef(resid[1:], resid[:-1])[0,1], 0, 0.90))
    months = hist["date"].dt.month.values
    sigma  = {m: float(np.std(resid[months==m])) if (months==m).sum()>2 else 2.5
              for m in range(1,13)}
    log.info(f"  AR(1) artık: φ={phi:.3f}  σ_ort={np.mean(list(sigma.values())):.2f} pp")
    return {"phi": phi, "sigma": sigma}


# ══════════════════════════════════════════════════════════════════════════════
# § 12  Projeksiyon — hibrit fizik + ML artık
# ══════════════════════════════════════════════════════════════════════════════

def simulate_v5(
    df:           pd.DataFrame,
    wb_model:     PhysicalWaterBalance,
    ml_model,     q10, q90,
    conformal:    SplitConformal,
    ar1:          dict,
    cons_model:   dict,
    start:        pd.Timestamp,
    end:          pd.Timestamp,
    feat_cols:    list,
    scenario:     dict,
    base_year:    int,
    seed:         int = 42,
) -> pd.DataFrame:
    """
    Hibrit projeksiyon:
      fill(t) = clip(fill_fiziksel(t) + artık_ML(t), 0, 100)

    fill_fiziksel: GR4J-tipi su dengesi (senaryo iklimi ile)
    artık_ML:      ML düzeltme terimi (sıfır ortalama, sınırlı aralık)

    Neden sürükleme olmaz:
    • Fizik modeli tüketim artışı ve yağış azalmasını zaten taşır
    • ML yalnızca fizik modelinin hata ettiği küçük kısmı düzeltir
    • AR(1) gürültü residual uzayındadır → kümülatif drift ±15 pp ile sınırlı
    """
    data = df.copy().sort_values("date").reset_index(drop=True)
    d2i  = {d:i for i,d in enumerate(data["date"])}
    si   = d2i.get(start)
    ei   = d2i.get(end)
    if si is None or ei is None:
        log.error("start/end bulunamadı"); return data

    fill_buf   = data["fill_pct"].values.copy().astype(float)
    resid_buf  = np.zeros(len(data))
    wb_buf     = data.get("fill_physical", pd.Series(np.nan, index=data.index)).values.copy().astype(float)

    sim = np.full(len(data), np.nan)
    lo  = np.full(len(data), np.nan)
    hi  = np.full(len(data), np.nan)

    # Geçmiş değerleri kopyala
    for i in range(si):
        v = fill_buf[i]
        sim[i] = lo[i] = hi[i] = v if not np.isnan(v) else np.nan

    rng     = np.random.default_rng(seed)
    phi     = ar1.get("phi", 0.50)
    sigmo   = ar1.get("sigma", {m: 2.5 for m in range(1,13)})
    ar1_s   = 0.0
    h_total = 0

    def blag(buf, i, lag, default=50.0):
        p = i - lag
        if p < 0: return default
        v = buf[p]
        return float(v) if not np.isnan(v) else default

    for year in range(start.year, end.year + 1):
        # Eğitim: bu yıldan öncesi (6 ay embargo ile)
        embargo_date = pd.Timestamp(f"{year}-01-01") - pd.DateOffset(months=6)
        tr = data[(data["date"] < embargo_date) & data["fill_pct"].notna()
                  ].dropna(subset=feat_cols + ["residual"])
        if tr.empty:
            log.warning(f"  {year}: eğitim verisi yok.")
            continue

        try:
            ml_model.fit(tr[feat_cols].values, tr["residual"].values)
            q10.fit(tr[feat_cols].values, tr["residual"].values)
            q90.fit(tr[feat_cols].values, tr["residual"].values)
        except Exception as e:
            log.error(f"  {year}: eğitim hatası {e}"); continue

        # Bu yılın ayları
        year_mask = ((data["date"] >= pd.Timestamp(f"{year}-01-01")) &
                     (data["date"] <= min(pd.Timestamp(f"{year}-12-01"), end)) &
                     (data["date"] >= start))
        idxs = data.index[year_mask].tolist()

        for i in idxs:
            h_total += 1
            mo = int(data.loc[i,"month"]) if "month" in data.columns \
                 else int(data.loc[i,"date"].month)

            # Senaryo ile iklim satırını güncelle
            row = apply_scenario_to_row(data.loc[i].copy(), base_year,
                                         scenario, cons_model)

            # Fizik modeli adımı
            prev_fill = blag(fill_buf, i, 1, default=50.0)
            wb_model.soil_moisture = blag(
                wb_buf, i, 1, default=wb_model.x1 * 0.5)  # yaklaşık

            cons_mcm = float(row.get("consumption_mcm",
                                      predict_consumption(cons_model,
                                                          float(row["date"].year)) * 1e-6))
            fill_phys, diag = wb_model.step(
                float(row["rain_mm"]),
                float(row["et0_mm_month"]),
                cons_mcm,
                prev_fill)

            # Özellik satırını güncelle (lag'lar simüle değerlerden)
            row["lag1_fill"]    = blag(fill_buf, i, 1)
            row["lag3_fill"]    = blag(fill_buf, i, 3)
            row["lag12_fill"]   = blag(fill_buf, i, 12)
            row["lag1_resid"]   = blag(resid_buf, i, 1, 0.0)
            row["lag2_resid"]   = blag(resid_buf, i, 2, 0.0)
            row["lag3_resid"]   = blag(resid_buf, i, 3, 0.0)
            row["lag6_resid"]   = blag(resid_buf, i, 6, 0.0)
            row["lag12_resid"]  = blag(resid_buf, i, 12, 0.0)
            row["resid_ma3"]    = np.mean([blag(resid_buf,i,k,0) for k in range(1,4)])
            row["resid_ma6"]    = np.mean([blag(resid_buf,i,k,0) for k in range(1,7)])
            row["lag1_wb"]      = blag(wb_buf, i, 1, fill_phys)
            row["lag3_wb"]      = blag(wb_buf, i, 3, fill_phys)
            row["fill_physical"]= fill_phys
            row["wb_runoff_mm"] = diag["runoff_mm"]
            row["wb_inflow_mcm"]= diag["inflow_mcm"]
            row["wb_evap_mcm"]  = diag["evap_mcm"]
            row["wb_net_mcm"]   = diag["net_mcm"]
            row["wb_cn"]        = diag["cn"]
            row["stl_trend"]    = blag(fill_buf, i, 1, fill_phys)
            row["stl_resid_fill"]= blag(resid_buf, i, 1, 0.0)

            for fc in feat_cols:
                if pd.isna(row.get(fc, np.nan)):
                    col_idx = feat_cols.index(fc)
                    row[fc] = float(tr[feat_cols].values[:, col_idx].mean())

            Xp = row[feat_cols].values.reshape(1,-1)

            # ML artık tahmini
            try:
                r_ml  = float(ml_model.predict(Xp)[0])
                r_q10 = float(q10.predict(Xp)[0])
                r_q90 = float(q90.predict(Xp)[0])
            except Exception:
                r_ml = r_q10 = r_q90 = 0.0

            # AR(1) gürültü (artık uzayında — küçük!)
            sigma  = sigmo.get(mo, 2.5)
            ar1_s  = phi * ar1_s + sigma * float(rng.normal(0, 1))
            ns     = float(np.clip(np.sqrt(h_total/12) * 0.20, 0, 1.0))
            r_ml   = r_ml + ar1_s * ns

            # Artık sınırlaması: ML modeli ne kadar yanılabilir?
            # Uzak ufukta artık katkısını azalt (fizik daha güvenilir)
            resid_weight = float(np.clip(1.0 - h_total / 200.0, 0.15, 1.0))
            r_ml  *= resid_weight
            r_q10 *= resid_weight
            r_q90 *= resid_weight

            # Nihai fill
            fill_final = float(np.clip(fill_phys + r_ml, 0, 100))

            # Belirsizlik bantları
            lo_r, hi_r = conformal.interval(np.array([r_ml]))
            fill_lo = float(np.clip(fill_phys + lo_r[0] + r_q10, 0, 100))
            fill_hi = float(np.clip(fill_phys + hi_r[0] + r_q90, 0, 100))
            # Sıralama garantisi
            fill_lo = min(fill_lo, fill_final)
            fill_hi = max(fill_hi, fill_final)

            sim[i]         = fill_final
            lo[i]          = fill_lo
            hi[i]          = fill_hi
            fill_buf[i]    = fill_final
            resid_buf[i]   = r_ml
            wb_buf[i]      = fill_phys

    out = data[["date"]].copy()
    out["fill_pct"]      = data.get("fill_pct", pd.Series(np.nan, index=data.index))
    out["fill_physical"] = wb_buf
    out["fill_sim"]      = sim
    out["fill_lo"]       = lo
    out["fill_hi"]       = hi
    out["scenario"]      = scenario.get("label","?")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# § 13  Görselleştirme
# ══════════════════════════════════════════════════════════════════════════════

SC_COLORS = {
    "SSP1-2.6 (İyimser)":  "#185FA5",
    "SSP2-4.5 (Baz)":      "#D55E00",
    "SSP5-8.5 (Kötümser)": "#E24B4A",
    "default":             "#D55E00",
}

def plot_v5_single(out: pd.DataFrame, last_obs: pd.Timestamp,
                   hist_trend: dict, cv_sum: dict | None,
                   fig_dir: Path, tag: str):
    hm = out["date"] <= last_obs; fm = out["date"] > last_obs
    fig = plt.figure(figsize=(17,11))
    gs  = gridspec.GridSpec(2,4,height_ratios=[2.5,1],hspace=0.42,wspace=0.32)
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1:3])
    ax4 = fig.add_subplot(gs[1,3])

    sc_lbl  = out["scenario"].iloc[0] if "scenario" in out.columns else "default"
    sc_col  = SC_COLORS.get(sc_lbl, "#D55E00")

    ax1.plot(out["date"][hm], out["fill_pct"][hm], "#111", lw=1.3,
             label="Gözlenen", zorder=4)
    ax1.plot(out["date"][fm], out["fill_sim"][fm], sc_col, lw=2.0,
             label=f"Hibrit tahmin — {sc_lbl}", zorder=5)
    if "fill_physical" in out.columns:
        ax1.plot(out["date"][fm], out["fill_physical"][fm], sc_col,
                 lw=1.0, ls=":", alpha=0.5, label="Sadece fizik (WB)")
    ax1.fill_between(out["date"][fm],
                      out["fill_lo"][fm].clip(0,100),
                      out["fill_hi"][fm].clip(0,100),
                      color=sc_col, alpha=0.12, lw=0, label="Konformal %90")
    ax1.axvline(last_obs, color="#888", lw=0.8, ls="--", alpha=0.7)
    ax1.axhline(30, color="#E24B4A", lw=0.8, ls=":", alpha=0.55)
    ax1.text(out["date"].iloc[0], 31.5, "Kritik eşik (%30)",
             fontsize=8, color="#E24B4A", alpha=0.7)
    ax1.set_ylim(0, 108); ax1.set_ylabel("Doluluk (%)", fontsize=10)
    sl = hist_trend.get("slope_recent", hist_trend.get("slope_full", 0))
    ax1.set_title(
        f"İstanbul Baraj v5 Hibrit — {tag}\n"
        f"Kandilli trend: {sl:+.2f} mm/yr | GR4J-tipi WB + ML artık | {sc_lbl}",
        fontsize=10.5)
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.grid(True, color="#EEE", lw=0.5)
    ax1.spines[["top","right"]].set_visible(False)
    if cv_sum:
        ax1.text(0.01, 0.02,
                 f"CV: RMSE={cv_sum.get('rmse_pp','?'):.2f}pp  "
                 f"NSE={cv_sum.get('nse','?'):.3f}  r={cv_sum.get('pearson_r','?'):.3f}",
                 transform=ax1.transAxes, fontsize=8.5, color="#555",
                 bbox=dict(boxstyle="round,pad=0.3",fc="#F9F9F9",ec="#CCC",alpha=0.9))

    # Fizik vs gözlenen (backtesting)
    if "fill_physical" in out.columns:
        obs = out[hm & out["fill_pct"].notna()]
        ax2.scatter(obs["fill_physical"], obs["fill_pct"],
                    alpha=0.5, s=8, color="#185FA5")
        mn = min(obs["fill_physical"].min(), obs["fill_pct"].min())
        mx = max(obs["fill_physical"].max(), obs["fill_pct"].max())
        ax2.plot([mn,mx],[mn,mx],"#E24B4A",lw=0.8,ls="--",alpha=0.7)
        wb_r = pearson_r(obs["fill_physical"].values, obs["fill_pct"].values)
        ax2.set_title(f"Fizik vs Gözlenen (r={wb_r:.3f})", fontsize=9)
        ax2.set_xlabel("Fizik modeli (pp)")
        ax2.set_ylabel("Gözlenen (pp)")
        ax2.spines[["top","right"]].set_visible(False)
        ax2.grid(True, color="#EEE", lw=0.5)

    # Yıllık ortalama + OLS trend
    def ym(g):
        return g["fill_sim"].mean() if (g["date"]>last_obs).any() else g["fill_pct"].mean()
    yr = out.groupby(out["date"].dt.year).apply(ym).dropna()
    cyr = ["#185FA5" if y<=last_obs.year else sc_col for y in yr.index]
    ax3.bar(yr.index, yr.values, color=cyr, alpha=0.8, width=0.8)
    x, y = yr.index.values.astype(float), yr.values
    if len(x) > 3:
        ms, bs, *_ = stats.linregress(x, y)
        ax3.plot(x, ms*x+bs, "#333", lw=1.1, ls="--", alpha=0.7,
                 label=f"OLS {ms:+.2f} pp/yr"); ax3.legend(fontsize=7)
    ax3.axhline(30, color="#E24B4A", lw=0.7, ls=":", alpha=0.5)
    ax3.set_title("Yıllık ort. doluluk + trend", fontsize=9)
    ax3.set_ylim(0,100); ax3.set_ylabel("Doluluk (%)")
    ax3.spines[["top","right"]].set_visible(False)
    ax3.grid(True, axis="y", color="#EEE", lw=0.5)
    ax3.legend(handles=[
        Patch(color="#185FA5",alpha=0.8,label="Gözlenen"),
        Patch(color=sc_col,  alpha=0.8,label="Tahmin"),
    ]+(ax3.get_legend_handles_labels()[0][-1:] if ax3.get_legend() else []),fontsize=7)

    # Belirsizlik genişliği
    fut = out[fm].copy(); fut["unc"] = (fut["fill_hi"]-fut["fill_lo"]).clip(0,100)
    ax4.plot(range(1,len(fut)+1), fut["unc"].values, sc_col, lw=1.2)
    ax4.set_xlabel("Ufuk (ay)"); ax4.set_ylabel("Bant (pp)")
    ax4.set_title("Belirsizlik büyümesi", fontsize=9)
    ax4.spines[["top","right"]].set_visible(False)
    ax4.grid(True, color="#EEE", lw=0.5)

    plt.savefig(fig_dir/f"{tag}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Grafik: {tag}.png")


def plot_scenario_fan(all_outs: list[pd.DataFrame], last_obs: pd.Timestamp,
                       fig_dir: Path):
    """Tüm senaryoları tek grafikte göster."""
    fig, ax = plt.subplots(figsize=(15,7))
    hm = all_outs[0]["date"] <= last_obs
    ax.plot(all_outs[0]["date"][hm], all_outs[0]["fill_pct"][hm],
            "#111", lw=1.3, label="Gözlenen", zorder=5)
    for out in all_outs:
        sc_lbl = out["scenario"].iloc[0] if "scenario" in out.columns else "?"
        col    = SC_COLORS.get(sc_lbl, "#888")
        fm = out["date"] > last_obs
        ax.plot(out["date"][fm], out["fill_sim"][fm], col, lw=1.8,
                label=sc_lbl, zorder=4)
        ax.fill_between(out["date"][fm],
                         out["fill_lo"][fm].clip(0,100),
                         out["fill_hi"][fm].clip(0,100),
                         color=col, alpha=0.08, lw=0)
    ax.axvline(last_obs, color="#888", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(30, color="#E24B4A", lw=0.8, ls=":", alpha=0.55)
    ax.text(all_outs[0]["date"].iloc[0], 31.5, "Kritik eşik (%30)",
            fontsize=8, color="#E24B4A", alpha=0.7)
    ax.set_ylim(0, 108); ax.set_ylabel("Doluluk (%)", fontsize=11)
    ax.set_title("İstanbul Baraj Doluluk — SSP Senaryo Yelpazesi (v5 Hibrit)",
                 fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, color="#EEE", lw=0.5)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(fig_dir/"scenario_fan.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("  Senaryo yelpazesi: scenario_fan.png")


def plot_cv_bar(cv_df, fig_dir):
    if cv_df.empty: return
    s = cv_df.groupby("model")["rmse_pp"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8,4))
    cols = ["#185FA5" if i==0 else "#85B7EB" for i in range(len(s))]
    bars = ax.barh(s.index, s.values, color=cols, alpha=0.9)
    ax.bar_label(bars, fmt="%.2f pp", padding=4, fontsize=9)
    ax.set_xlabel("Ort. CV RMSE (pp)")
    ax.set_title("Model Karşılaştırması — Purged WF-CV (artık uzayı)")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, axis="x", color="#EEE", lw=0.5)
    plt.tight_layout(); fig.savefig(fig_dir/"model_cv.png", dpi=150); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# § 14  Optuna
# ══════════════════════════════════════════════════════════════════════════════

def optuna_tune(X, y, n_trials=60, mtype="lgb"):
    if not HAS_OPTUNA: return {}
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    def obj(trial):
        if mtype=="lgb" and HAS_LGB:
            m = lgb.LGBMRegressor(
                boosting_type="dart",
                n_estimators   = trial.suggest_int("ne",200,700),
                learning_rate  = trial.suggest_float("lr",0.01,0.08,log=True),
                num_leaves     = trial.suggest_int("nl",15,50),
                max_depth      = trial.suggest_int("md",3,7),
                min_child_samples = trial.suggest_int("mcs",8,40),
                subsample      = trial.suggest_float("sub",0.6,1.0),
                colsample_bytree=trial.suggest_float("col",0.6,1.0),
                reg_alpha      = trial.suggest_float("ra",0.1,20,log=True),
                reg_lambda     = trial.suggest_float("rl",0.1,20,log=True),
                drop_rate      = trial.suggest_float("dr",0.05,0.30),
                random_state=42, n_jobs=-1, verbose=-1)
        elif mtype=="xgb" and HAS_XGB:
            m = xgb.XGBRegressor(
                n_estimators   = trial.suggest_int("ne",200,700),
                learning_rate  = trial.suggest_float("lr",0.01,0.08,log=True),
                max_depth      = trial.suggest_int("md",3,7),
                subsample      = trial.suggest_float("sub",0.6,1.0),
                colsample_bytree=trial.suggest_float("col",0.6,1.0),
                reg_alpha      = trial.suggest_float("ra",0.1,20,log=True),
                reg_lambda     = trial.suggest_float("rl",0.1,20,log=True),
                random_state=42, n_jobs=-1, verbosity=0)
        else: return 999.0
        sc = []
        for ti, vi in tscv.split(X):
            m.fit(X[ti], y[ti]); sc.append(rmse(y[vi], m.predict(X[vi])))
        return float(np.mean(sc))
    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    log.info(f"Optuna best={study.best_value:.4f}"); return study.best_params


# ══════════════════════════════════════════════════════════════════════════════
# § 15  SHAP
# ══════════════════════════════════════════════════════════════════════════════

def run_shap(model, X, feat_cols, out_dir, tag):
    if not HAS_SHAP: return
    try:
        try:   exp=shap.TreeExplainer(model); sv=exp.shap_values(X)
        except:exp=shap.LinearExplainer(model,X); sv=exp.shap_values(X)
        imp = pd.Series(np.abs(sv).mean(0), index=feat_cols).sort_values()
        top = imp.tail(20)
        fig, ax = plt.subplots(figsize=(9,6))
        colors = ["#185FA5" if v>imp.median() else "#85B7EB" for v in top.values]
        ax.barh(top.index, top.values, color=colors, height=0.7)
        ax.set_xlabel("Ort. |SHAP| (artık pp uzayı)")
        ax.set_title(f"SHAP — {tag}")
        ax.axvline(top.values.mean(), color="#888", lw=0.7, ls="--")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); fig.savefig(out_dir/f"shap_{tag}.png", dpi=150); plt.close(fig)
        log.info(f"  Top5: {list(imp.tail(5)[::-1].index)}")
    except Exception as e: log.warning(f"  SHAP: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# § 16  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="İstanbul Baraj v5 — Hibrit Fizik+ML")
    p.add_argument("--panel",
        default="output/newdata_feature_store/tables/istanbul_dam_driver_panel_2000_2026_extended.csv")
    p.add_argument("--climate",
        default="output/scientific_climate_projection_2026_2040/climate_projection_2010_2040_monthly.csv")
    p.add_argument("--ibb-dir",   default="external/raw/ibb")
    p.add_argument("--out",       default="output/istanbul_v5")
    p.add_argument("--end-date",  default="2040-12-01")
    p.add_argument("--cv-folds",  type=int,   default=5)
    p.add_argument("--coverage",  type=float, default=0.90)
    p.add_argument("--tune",      action="store_true")
    p.add_argument("--n-trials",  type=int,   default=60)
    p.add_argument("--no-shap",   action="store_true")
    p.add_argument("--calibrate-cn", action="store_true",
                   help="CN/x1/routing fizik parametrelerini kalibre et")
    p.add_argument("--scenarios", action="store_true",
                   help="SSP1-2.6 / SSP2-4.5 / SSP5-8.5 üç senaryo çalıştır")
    args = p.parse_args()

    out_dir = ROOT / args.out; fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(out_dir/"run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-8s │ %(message)s"))
    log.addHandler(fh)

    log.info("="*70)
    log.info("İstanbul Baraj v5 — GR4J-tipi WB + ML Artık + SSP Senaryolar")
    log.info(f"  LGB:{HAS_LGB}  XGB:{HAS_XGB}  Optuna:{HAS_OPTUNA}  SHAP:{HAS_SHAP}")
    log.info("="*70)

    # ── 1. Veri
    ibb_dir = ROOT / args.ibb_dir
    cp = ROOT / args.climate
    panel, cons_raw, catchment_rain = load_all_data(
        ROOT / args.panel, cp if cp.exists() else None, ibb_dir)

    last_obs   = panel[panel["fill_pct"].notna()]["date"].max()
    start_date = (pd.Timestamp(last_obs) + pd.DateOffset(months=1)).normalize()
    end_date   = pd.Timestamp(args.end_date)
    base_year  = last_obs.year
    log.info(f"Son gözlem: {last_obs.date()} | {start_date.date()} → {end_date.date()}")

    # ── 2. Kandilli uzun dönem trend
    log.info("\n100 Yıllık Trend Analizi…")
    df_kand = load_kandilli_precipitation(ROOT)
    hist_trend = analyse_long_term_trend(df_kand)

    # ── 3. Havza yağışını panele ekle
    if not catchment_rain.empty:
        panel = panel.merge(catchment_rain, on="date", how="left")
        panel["catchment_rain_mm"] = panel["catchment_rain_mm"].fillna(panel["rain_mm"])
    else:
        panel["catchment_rain_mm"] = panel["rain_mm"]

    # ── 4. Tüketim modeli
    log.info("\nTüketim büyüme modeli…")
    if not cons_raw.empty:
        panel = panel.merge(cons_raw, on="date", how="left")
        hist_cons = panel[panel["consumption_m3"].notna()].copy()
        cons_years = hist_cons["date"].dt.year + (hist_cons["date"].dt.month-1)/12
        cons_model = fit_consumption_trend(hist_cons["consumption_m3"],
                                            cons_years)
        # MCM dönüşümü
        panel["consumption_mcm"] = panel["consumption_m3"].fillna(
            panel["date"].apply(lambda d: predict_consumption(
                cons_model, d.year + (d.month-1)/12))) * 1e-6
    else:
        log.warning("  Tüketim verisi yok — varsayılan kullanılıyor (3 Mm³/ay)")
        panel["consumption_m3"]  = 3.0e6
        panel["consumption_mcm"] = 3.0
        cons_model = {"type": "linear", "slope": 0.05e6, "intercept": 3.0e6}

    # ── 5. Fiziksel su dengesi modeli
    log.info("\nFizik modeli hazırlanıyor…")
    wb = PhysicalWaterBalance()

    if args.calibrate_cn:
        log.info("  Kalibrasyon başlıyor…")
        hist = panel[panel["fill_pct"].notna()].dropna(subset=["fill_pct"])
        cal_params = wb.calibrate(
            hist["catchment_rain_mm"], hist["et0_mm_month"],
            hist["consumption_mcm"], hist["fill_pct"],
            fill_init=float(hist["fill_pct"].iloc[0]))
        log.info(f"  Kalibre parametreler: {cal_params}")
    else:
        log.info(f"  Varsayılan parametreler: CN={wb.cn_base}  x1={wb.x1}  routing={wb.routing}")

    # Tarihsel + gelecek için fizik çalıştır
    log.info("  Fizik modeli çalıştırılıyor (2000–2040)…")
    fill_init = float(panel["fill_pct"].dropna().iloc[0]) if not panel["fill_pct"].isna().all() else 50.0
    wb_hist = wb.run(panel["catchment_rain_mm"].fillna(0),
                     panel["et0_mm_month"].fillna(0),
                     panel["consumption_mcm"].fillna(3.0),
                     fill_init=fill_init)
    wb_hist["date"] = panel["date"].values

    # Fizik kalite metrikleri
    obs_mask = panel["fill_pct"].notna()
    if obs_mask.sum() > 0:
        obs_fill  = panel.loc[obs_mask, "fill_pct"].values
        phys_fill = wb_hist.loc[obs_mask, "fill_physical"].values
        r_wb = rmse(obs_fill, phys_fill)
        n_wb = nse(obs_fill, phys_fill)
        log.info(f"  Fizik model RMSE={r_wb:.2f}pp  NSE={n_wb:.3f}")

    # ── 6. Özellik mühendisliği
    log.info("\nÖzellik mühendisliği (artık hedefli)…")
    df = build_features(panel, wb_hist)
    feat_cols = [f for f in ALL_FEATS if f in df.columns]
    log.info(f"  Özellik sayısı: {len(feat_cols)}")

    # ── 7. Optuna
    lgb_p = xgb_p = {}
    if args.tune:
        hdf = df[df["residual"].notna()].dropna(subset=feat_cols+["residual"])
        Xa, ya = hdf[feat_cols].values, hdf["residual"].values
        if HAS_LGB: lgb_p = optuna_tune(Xa, ya, args.n_trials, "lgb")
        if HAS_XGB: xgb_p = optuna_tune(Xa, ya, args.n_trials, "xgb")

    # ── 8. Modeller
    log.info("\nML modelleri (artık üzerinde)…")
    all_m = make_models(lgb_p, xgb_p)
    main_m = {k:v for k,v in all_m.items() if k not in ("lgb_q10","lgb_q90")}
    q10 = all_m.get("lgb_q10",
           make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.10, solver="highs")))
    q90 = all_m.get("lgb_q90",
           make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.90, solver="highs")))

    # ── 9. Walk-Forward CV
    log.info("\nPurged CV…")
    cv_df = purged_cv(df, main_m, feat_cols, n_folds=args.cv_folds)
    cv_df.to_csv(out_dir/"cv_results.csv", index=False)
    plot_cv_bar(cv_df, fig_dir)

    best_name = (cv_df.groupby("model")["rmse_pp"].mean().idxmin()
                 if not cv_df.empty else ("lgb_dart" if HAS_LGB else "etr"))
    log.info(f"★ EN İYİ MODEL (artık RMSE): {best_name.upper()}")

    # ── 10. Konformal kalibrasyon (artık uzayında)
    log.info("\nKonformal kalibrasyon…")
    hdf    = df[df["residual"].notna()].dropna(subset=feat_cols+["residual"])
    cal_n  = max(24, len(hdf)//5)
    tr_cal = hdf.iloc[:-cal_n]; cal_df = hdf.iloc[-cal_n:]
    bm = main_m[best_name]
    bm.fit(tr_cal[feat_cols].values, tr_cal["residual"].values)
    pred_r = bm.predict(cal_df[feat_cols].values)
    conf   = SplitConformal(args.coverage)
    conf.calibrate(cal_df["residual"].values, pred_r)

    # AR(1)
    ar1 = estimate_ar1_resid(df, main_m[best_name], feat_cols)

    # ── 11. SHAP (en iyi model)
    if not args.no_shap:
        bm.fit(hdf[feat_cols].values, hdf["residual"].values)
        run_shap(bm, hdf[feat_cols].values, feat_cols, fig_dir, best_name)

    # ── 12. Proyeksiyonlar
    log.info("\nProjeksiyonlar…")
    Xfin = hdf[feat_cols].values; yfin = hdf["residual"].values

    scenarios_to_run = (
        list(SSP_SCENARIOS.values()) if args.scenarios
        else [SSP_SCENARIOS["ssp245"]]
    )

    all_outs = []

    for name, model in main_m.items():
        log.info(f"\n── {name.upper()} ──")
        t0 = time.time()
        try: model.fit(Xfin, yfin)
        except Exception as e: log.error(f"  Eğitim: {e}"); continue
        q10.fit(Xfin, yfin); q90.fit(Xfin, yfin)
        log.info(f"  Eğitim: {time.time()-t0:.1f}s")

        for sc in scenarios_to_run:
            log.info(f"  Senaryo: {sc['label']}")
            out = simulate_v5(
                df, wb, model, q10, q90,
                conf, ar1, cons_model,
                start_date, end_date, feat_cols,
                sc, base_year)
            tag = f"{name}_{sc['label'].replace(' ','_').replace('-','').replace('(','').replace(')','')}"
            out["model"] = name
            out.to_csv(out_dir/f"projection_{tag}.csv", index=False)
            all_outs.append(out)

            cv_s = None
            if not cv_df.empty:
                mc = cv_df[cv_df["model"]==name]
                if not mc.empty:
                    cv_s = mc[["rmse_pp","nse","pearson_r"]].mean().to_dict()
            plot_v5_single(out, last_obs, hist_trend, cv_s, fig_dir, tag)

    # ── 13. Senaryo yelpazesi grafiği
    if args.scenarios and len(all_outs) >= 3:
        # Baz model, 3 senaryo
        baz_outs = [o for o in all_outs if o["model"].iloc[0] == best_name]
        if len(baz_outs) >= 3:
            plot_scenario_fan(baz_outs, last_obs, fig_dir)

    # ── 14. Birleşik CSV
    if all_outs:
        pd.concat(all_outs).to_csv(out_dir/"projection_all.csv", index=False)

    # ── Özet
    log.info("\n"+"="*70)
    log.info("TAMAMLANDI")
    log.info(f"Çıktılar → {out_dir}")
    log.info(f"Kandilli trend: {hist_trend.get('slope_recent',0):+.2f} mm/yr "
             f"({hist_trend.get('breakpoint',1980)}+)")
    if not cons_raw.empty:
        log.info(f"Tüketim modeli: {cons_model.get('type','?')}")
    log.info(f"Fizik modeli NSE={n_wb:.3f}  RMSE={r_wb:.2f}pp (tarihsel)")
    if not cv_df.empty:
        bc = cv_df[cv_df["model"]==best_name][["rmse_pp","nse","pearson_r"]].mean()
        log.info(f"En iyi ML artık: {best_name.upper()}  "
                 f"RMSE={bc['rmse_pp']:.2f}pp  NSE={bc['nse']:.3f}")
    log.info(f"Konformal bant: ±{conf.q:.2f}pp ({args.coverage*100:.0f}%)")
    if all_outs:
        baz_outs = [o for o in all_outs
                    if "SSP2-4.5" in str(o.get("scenario",[""])) or
                       o["model"].iloc[0]==best_name]
        if baz_outs:
            v40 = baz_outs[0].dropna(subset=["fill_sim"]).tail(12)["fill_sim"].mean()
            log.info(f"2040 baz senaryo ort. tahmini: {v40:.1f}%")
    log.info("="*70)


if __name__ == "__main__":
    main()
