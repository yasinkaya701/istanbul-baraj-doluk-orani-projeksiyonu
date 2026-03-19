#!/usr/bin/env python3
"""
et_ml.py - Makine ogrenmesi ile ET0 tahmini: Random Forest + Gradient Boosting + opsiyonel LSTM
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from compute_et0_fao56 import calc_ra_mj_m2_day

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
SIGMA_MJ_K4_M2_DAY = 4.903e-9

# LSTM icin opsiyonel TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
    from tensorflow.keras.models import Sequential

    TF_MEVCUT = True
except Exception:
    TF_MEVCUT = False
    print("  Bilgi: TensorFlow bulunamadi, sadece RF/GB calisacak.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ET0 ML model egitimi (RF/GB/LSTM)")
    p.add_argument("--year", type=int, default=1987, help="Analiz yili")
    p.add_argument(
        "--data-source",
        type=str,
        choices=["et0_csv", "wide"],
        default="et0_csv",
        help="Egitim verisi kaynagi: et0_csv (hazir ET0) veya wide (genis meteoroloji)",
    )
    p.add_argument(
        "--et0-csv",
        type=Path,
        default=Path("output/spreadsheet/et0_inputs_completed_1987.csv"),
        help="Gunluk ET0 giris CSV",
    )
    p.add_argument(
        "--water-balance-csv",
        type=Path,
        default=Path("output/spreadsheet/water_balance_partial_1987.csv"),
        help="Aylik yagis/su butcesi CSV",
    )
    p.add_argument(
        "--wide-csv",
        type=Path,
        default=Path("output/spreadsheet/meteoroloji_model_egitim_wide_genisletilmis_filled.csv"),
        help="Genis meteoroloji CSV (ds,temp,humidity,pressure,solar,wind_speed,precip)",
    )
    p.add_argument("--latitude", type=float, default=41.01, help="Enlem (Ra hesabi)")
    p.add_argument("--elevation-m", type=float, default=39.0, help="Rakim (basinc fallback)")
    p.add_argument("--run-lstm", action="store_true", help="TensorFlow varsa LSTM de calistir")
    p.add_argument("--lstm-window", type=int, default=30, help="LSTM gecmis pencere boyu (gun)")
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Cikti dosya son eki (varsayilan: yil veya wide_y1_y2)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/spreadsheet"),
        help="Cikti klasoru",
    )
    return p.parse_args()


def _saturation_vp(temp_c: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def _daily_precip_from_monthly(index: pd.DatetimeIndex, wb_csv: Path) -> pd.Series:
    s = pd.Series(0.0, index=index, dtype=float)
    if not wb_csv.exists():
        return s
    wb = pd.read_csv(wb_csv)
    if "month" not in wb.columns or "precip_obs_mm" not in wb.columns:
        return s
    wb = wb.copy()
    wb["month"] = wb["month"].astype(str)
    wb["precip_obs_mm"] = pd.to_numeric(wb["precip_obs_mm"], errors="coerce").fillna(0.0)
    month_map = dict(zip(wb["month"], wb["precip_obs_mm"]))
    months = index.to_period("M").astype(str)
    days = index.days_in_month.astype(int)
    vals = np.array([month_map.get(m, 0.0) for m in months], dtype=float)
    return pd.Series(np.where(days > 0, vals / days, 0.0), index=index, dtype=float)


def _compute_et0_pm(d: pd.DataFrame, elevation_m: float) -> pd.Series:
    """FAO-56 Penman-Monteith (gunluk) ET0."""
    tmax = d["Tmax"].to_numpy(dtype=float)
    tmin = d["Tmin"].to_numpy(dtype=float)
    tmean = d["Tmean"].to_numpy(dtype=float)
    rh = d["rh_mean"].to_numpy(dtype=float)
    u2 = d["U2"].to_numpy(dtype=float)
    rs = d["Rs"].to_numpy(dtype=float)
    ra = d["Ra"].to_numpy(dtype=float)
    p_kpa = d["p_kpa"].to_numpy(dtype=float)

    rso = (0.75 + 2.0e-5 * elevation_m) * ra
    rs = np.minimum(np.maximum(rs, 0.0), rso)
    rns = 0.77 * rs

    es_tmax = _saturation_vp(tmax)
    es_tmin = _saturation_vp(tmin)
    es = 0.5 * (es_tmax + es_tmin)
    ea = (np.clip(rh, 1.0, 100.0) / 100.0) * es

    delta = 4098.0 * (0.6108 * np.exp((17.27 * tmean) / (tmean + 237.3))) / ((tmean + 237.3) ** 2)
    gamma = 0.000665 * p_kpa

    tmax_k = tmax + 273.16
    tmin_k = tmin + 273.16
    rs_rso = np.where(rso > 0, rs / rso, np.nan)
    rs_rso = np.clip(rs_rso, 0.0, 1.0)
    rnl = (
        SIGMA_MJ_K4_M2_DAY
        * ((tmax_k**4 + tmin_k**4) / 2.0)
        * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0)))
        * (1.35 * rs_rso - 0.35)
    )
    rn = rns - rnl

    num = 0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)
    den = delta + gamma * (1.0 + 0.34 * u2)
    et0 = np.where(den > 0, num / den, np.nan)
    et0 = np.clip(et0, 0.0, None)
    return pd.Series(et0, index=d.index)


def _hazirla_df_et0(et0_csv: Path, wb_csv: Path, year: int, latitude: float, elevation_m: float) -> pd.DataFrame:
    raw = pd.read_csv(et0_csv)
    needed = {
        "date",
        "t_mean_c",
        "t_min_c",
        "t_max_c",
        "rh_mean_pct",
        "u2_m_s",
        "rs_mj_m2_day",
        "et0_completed_mm_day",
    }
    miss = needed - set(raw.columns)
    if miss:
        raise ValueError(f"{et0_csv} eksik kolonlar: {sorted(miss)}")

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).copy()
    raw = raw[raw["date"].dt.year == year].sort_values("date").reset_index(drop=True)
    if raw.empty:
        raise ValueError(f"{year} icin satir yok: {et0_csv}")

    d = pd.DataFrame({"date": raw["date"]})
    d["Tmax"] = pd.to_numeric(raw["t_max_c"], errors="coerce")
    d["Tmin"] = pd.to_numeric(raw["t_min_c"], errors="coerce")
    d["Tmean"] = pd.to_numeric(raw["t_mean_c"], errors="coerce")
    d["U2"] = pd.to_numeric(raw["u2_m_s"], errors="coerce")
    d["Rs"] = pd.to_numeric(raw["rs_mj_m2_day"], errors="coerce")
    d["ET0"] = pd.to_numeric(raw["et0_completed_mm_day"], errors="coerce")
    d["P"] = _daily_precip_from_monthly(pd.DatetimeIndex(d["date"]), wb_csv).to_numpy(dtype=float)

    # Basinc fallback
    if "p_kpa" in raw.columns:
        d["p_kpa"] = pd.to_numeric(raw["p_kpa"], errors="coerce")
    else:
        d["p_kpa"] = np.nan
    p_fallback = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    d["p_kpa"] = d["p_kpa"].fillna(p_fallback)

    # Ra / es / ea / delta / gamma / Rn
    doy = d["date"].dt.dayofyear.astype(int)
    d["Ra"] = calc_ra_mj_m2_day(doy, latitude).to_numpy(dtype=float)
    d["es"] = 0.5 * (_saturation_vp(d["Tmax"].to_numpy(dtype=float)) + _saturation_vp(d["Tmin"].to_numpy(dtype=float)))
    rh = pd.to_numeric(raw["rh_mean_pct"], errors="coerce").fillna(60.0).to_numpy(dtype=float)
    d["rh_mean"] = rh
    d["ea"] = (rh / 100.0) * d["es"].to_numpy(dtype=float)
    tmean = d["Tmean"].to_numpy(dtype=float)
    d["delta"] = 4098.0 * (0.6108 * np.exp((17.27 * tmean) / (tmean + 237.3))) / ((tmean + 237.3) ** 2)
    d["gamma"] = 0.000665 * d["p_kpa"]
    d["Rn"] = 0.77 * d["Rs"]
    # EF proxy: RH yuksek -> EF yuksek, ruzgar yuksek -> EF dusuk
    d["EF"] = np.clip(0.15 + 0.008 * rh - 0.015 * d["U2"].fillna(2.0).to_numpy(dtype=float), 0.15, 0.90)

    d = d.set_index("date").sort_index()
    d = d.interpolate(method="time").ffill().bfill()
    return d


def _hazirla_df_wide(wide_csv: Path, latitude: float, elevation_m: float) -> pd.DataFrame:
    raw = pd.read_csv(wide_csv)
    need = {"ds", "temp", "humidity", "pressure", "solar", "wind_speed", "precip"}
    miss = need - set(raw.columns)
    if miss:
        raise ValueError(f"{wide_csv} eksik kolonlar: {sorted(miss)}")

    raw["ds"] = pd.to_datetime(raw["ds"], errors="coerce")
    raw = raw.dropna(subset=["ds"]).copy()
    raw = raw.sort_values("ds")
    if raw.empty:
        raise ValueError(f"Bos dataset: {wide_csv}")

    daily = (
        raw.set_index("ds")
        .resample("D")
        .agg(
            Tmean=("temp", "mean"),
            Tmax=("temp", "max"),
            Tmin=("temp", "min"),
            rh_mean=("humidity", "mean"),
            U2=("wind_speed", "mean"),
            Rs=("solar", "mean"),
            P=("precip", "sum"),
            p_raw=("pressure", "mean"),
        )
        .dropna(subset=["Tmean", "Tmax", "Tmin"])
    )

    # Unit heuristics
    p_med = float(pd.to_numeric(daily["p_raw"], errors="coerce").median())
    if p_med > 200.0:
        daily["p_kpa"] = pd.to_numeric(daily["p_raw"], errors="coerce") / 10.0
    else:
        daily["p_kpa"] = pd.to_numeric(daily["p_raw"], errors="coerce")

    u90 = float(pd.to_numeric(daily["U2"], errors="coerce").quantile(0.90))
    if u90 > 15.0:
        daily["U2"] = pd.to_numeric(daily["U2"], errors="coerce") / 3.6  # km/h -> m/s
    else:
        daily["U2"] = pd.to_numeric(daily["U2"], errors="coerce")

    rs_med = float(pd.to_numeric(daily["Rs"], errors="coerce").median())
    if rs_med > 50.0:
        daily["Rs"] = pd.to_numeric(daily["Rs"], errors="coerce") * 0.0864  # W/m2 mean -> MJ/m2/day
    elif rs_med < 1.5:
        daily["Rs"] = pd.to_numeric(daily["Rs"], errors="coerce") * 3.6  # kWh/m2/day -> MJ/m2/day
    else:
        daily["Rs"] = pd.to_numeric(daily["Rs"], errors="coerce")

    p_fallback = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    daily["p_kpa"] = daily["p_kpa"].fillna(p_fallback)
    daily["rh_mean"] = pd.to_numeric(daily["rh_mean"], errors="coerce").clip(lower=1.0, upper=100.0).fillna(60.0)
    daily["U2"] = pd.to_numeric(daily["U2"], errors="coerce").clip(lower=0.1, upper=20.0).fillna(2.0)
    daily["Rs"] = pd.to_numeric(daily["Rs"], errors="coerce").clip(lower=0.1).fillna(method="ffill").fillna(method="bfill")
    daily["P"] = pd.to_numeric(daily["P"], errors="coerce").fillna(0.0)

    daily["Ra"] = calc_ra_mj_m2_day(daily.index.dayofyear.astype(int), latitude).to_numpy(dtype=float)
    daily["es"] = 0.5 * (_saturation_vp(daily["Tmax"].to_numpy(dtype=float)) + _saturation_vp(daily["Tmin"].to_numpy(dtype=float)))
    daily["ea"] = (daily["rh_mean"].to_numpy(dtype=float) / 100.0) * daily["es"].to_numpy(dtype=float)
    tmean = daily["Tmean"].to_numpy(dtype=float)
    daily["delta"] = 4098.0 * (0.6108 * np.exp((17.27 * tmean) / (tmean + 237.3))) / ((tmean + 237.3) ** 2)
    daily["gamma"] = 0.000665 * daily["p_kpa"]
    daily["Rn"] = 0.77 * daily["Rs"]
    daily["EF"] = np.clip(0.15 + 0.008 * daily["rh_mean"] - 0.015 * daily["U2"], 0.15, 0.90)
    daily["ET0"] = _compute_et0_pm(daily, elevation_m)

    daily = daily.sort_index()
    daily = daily.interpolate(method="time").ffill().bfill()
    daily = daily[daily["ET0"].notna()].copy()
    return daily


def ozellik_uret(df: pd.DataFrame) -> pd.DataFrame:
    """ET0 tahmini icin ozellik seti."""
    d = df.copy()

    d["doy"] = d.index.dayofyear
    d["ay"] = d.index.month
    d["yil"] = d.index.year
    d["hafta"] = d.index.isocalendar().week.astype(int)

    d["sin_doy"] = np.sin(2 * np.pi * d["doy"] / 365)
    d["cos_doy"] = np.cos(2 * np.pi * d["doy"] / 365)

    d["Trange"] = d["Tmax"] - d["Tmin"]
    d["VPD"] = d["es"] - d["ea"]
    d["Tmax_sq"] = d["Tmax"] ** 2
    d["Rs_Ra"] = (d["Rs"] / d["Ra"].clip(0.1)).clip(0, 1)

    for lag in [1, 2, 3, 7, 14]:
        d[f"ET0_lag{lag}"] = d["ET0"].shift(lag)
        d[f"Tmax_lag{lag}"] = d["Tmax"].shift(lag)

    for pencere in [7, 14, 30]:
        d[f"ET0_ma{pencere}"] = d["ET0"].rolling(pencere).mean()
        d[f"Tmax_ma{pencere}"] = d["Tmax"].rolling(pencere).mean()
        d[f"P_ma{pencere}"] = d["P"].rolling(pencere).sum()

    d["T_anom"] = d["Tmean"] - d.groupby("doy")["Tmean"].transform("mean")
    d["P_anom"] = d["P"] - d.groupby("doy")["P"].transform("mean")

    yil_min = float(d["yil"].min())
    yil_max = float(d["yil"].max())
    if yil_max > yil_min:
        d["trend"] = (d["yil"] - yil_min) / (yil_max - yil_min)
    else:
        d["trend"] = 0.0

    return d.dropna()


OZELLIKLER = [
    "Tmax",
    "Tmin",
    "Tmean",
    "Trange",
    "Rs",
    "Rs_Ra",
    "Rn",
    "VPD",
    "es",
    "ea",
    "delta",
    "gamma",
    "U2",
    "EF",
    "sin_doy",
    "cos_doy",
    "doy",
    "ay",
    "ET0_lag1",
    "ET0_lag2",
    "ET0_lag3",
    "ET0_lag7",
    "Tmax_lag1",
    "Tmax_lag7",
    "ET0_ma7",
    "ET0_ma14",
    "ET0_ma30",
    "Tmax_ma7",
    "P_ma30",
    "T_anom",
    "P_anom",
    "trend",
]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return {"rmse": rmse, "mae": mae, "r2": r2, "bias": bias}


def rf_model_egit(df: pd.DataFrame, out_dir: Path) -> dict[str, dict]:
    """
    Random Forest ve Gradient Boosting egitimi.
    """
    print("\nRandom Forest / Gradient Boosting egitimi basliyor...")

    d = ozellik_uret(df)
    mevcut_ozellikler = [f for f in OZELLIKLER if f in d.columns]

    X = d[mevcut_ozellikler].to_numpy(dtype=float)
    y = d["ET0"].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    split = int(len(X_sc) * 0.80)
    if split < 60 or (len(X_sc) - split) < 20:
        raise ValueError("Modelleme icin yeterli veri yok. Daha uzun seri gerekiyor.")
    X_tr, X_te = X_sc[:split], X_sc[split:]
    y_tr, y_te = y[:split], y[split:]
    idx_te = d.index[split:]

    modeller = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
    }

    sonuclar: dict[str, dict] = {}
    tscv = TimeSeriesSplit(n_splits=4)
    for ad, model in modeller.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        m = _metrics(y_te, pred)

        cv_rmse = []
        cv_mae = []
        cv_r2 = []
        for cv_tr, cv_va in tscv.split(X_tr):
            model_cv = model.__class__(**model.get_params())
            model_cv.fit(X_tr[cv_tr], y_tr[cv_tr])
            p_cv = model_cv.predict(X_tr[cv_va])
            m_cv = _metrics(y_tr[cv_va], p_cv)
            cv_rmse.append(m_cv["rmse"])
            cv_mae.append(m_cv["mae"])
            cv_r2.append(m_cv["r2"])

        print(
            f"  {ad:20s} | RMSE={m['rmse']:.3f} | MAE={m['mae']:.3f} | "
            f"R2={m['r2']:.4f} | Bias={m['bias']:.3f}"
        )

        sonuclar[ad] = {
            "model": model,
            "scaler": scaler,
            "pred": pred,
            "gercek": y_te,
            "idx": idx_te,
            "rmse": m["rmse"],
            "mae": m["mae"],
            "r2": m["r2"],
            "bias": m["bias"],
            "cv_rmse_mean": float(np.mean(cv_rmse)),
            "cv_mae_mean": float(np.mean(cv_mae)),
            "cv_r2_mean": float(np.mean(cv_r2)),
            "ozellikler": mevcut_ozellikler,
        }

    _ozellik_onem_grafik(sonuclar["Random Forest"], out_dir / "rf_ozellik_onem.png")
    _performans_grafik(sonuclar, idx_te, out_dir / "ml_performans.png")
    return sonuclar


def _ozellik_onem_grafik(sn: dict, out_png: Path) -> None:
    model = sn["model"]
    onem = pd.Series(model.feature_importances_, index=sn["ozellikler"]).sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(onem.index, onem.values, color=plt.cm.RdYlGn(onem.values / max(onem.max(), 1e-9)))
    ax.set_xlabel("Ozellik onemi (Gini)")
    ax.set_title("Random Forest - En onemli 20 ozellik")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {out_png}")


def _performans_grafik(sonuclar: dict[str, dict], idx_te: pd.DatetimeIndex, out_png: Path) -> None:
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle("Makine Ogrenmesi ET0 Tahmin Performansi", fontsize=14, fontweight="bold")

    renkler = {"Random Forest": "#2196F3", "Gradient Boosting": "#FF5722"}

    ax1 = fig.add_subplot(gs[0, :])
    gercek_son = pd.Series(next(iter(sonuclar.values()))["gercek"], index=idx_te).resample("W").mean()
    ax1.plot(gercek_son.index, gercek_son.values, "k-", linewidth=1.2, label="Gozlem", zorder=5)
    for ad, sn in sonuclar.items():
        pred_son = pd.Series(sn["pred"], index=idx_te).resample("W").mean()
        ax1.plot(pred_son.index, pred_son.values, "--", color=renkler[ad], linewidth=1.2, label=f"{ad} (R2={sn['r2']:.3f})", alpha=0.85)
    ax1.set_ylabel("ET0 (mm/gun)")
    ax1.set_title("Test donemi tahmin vs gozlem (haftalik ort.)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    sn = sonuclar["Random Forest"]
    ax2.scatter(sn["gercek"], sn["pred"], alpha=0.15, s=8, color="#2196F3")
    lo, hi = min(sn["gercek"].min(), sn["pred"].min()), max(sn["gercek"].max(), sn["pred"].max())
    ax2.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    ax2.set_xlabel("Gozlem (mm/gun)")
    ax2.set_ylabel("Tahmin (mm/gun)")
    ax2.set_title(f"Random Forest | R2={sn['r2']:.4f} | RMSE={sn['rmse']:.3f}")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    sn = sonuclar["Gradient Boosting"]
    ax3.scatter(sn["gercek"], sn["pred"], alpha=0.15, s=8, color="#FF5722")
    ax3.plot([lo, hi], [lo, hi], "b--", linewidth=1.5)
    ax3.set_xlabel("Gozlem (mm/gun)")
    ax3.set_ylabel("Tahmin (mm/gun)")
    ax3.set_title(f"Gradient Boosting | R2={sn['r2']:.4f} | RMSE={sn['rmse']:.3f}")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {out_png}")


def lstm_model_egit(df: pd.DataFrame, out_dir: Path, pencere: int = 30) -> dict | None:
    """
    Cok degiskenli LSTM: pencere gunluk gecmis -> ertesi gun ET0 tahmini.
    """
    if not TF_MEVCUT:
        print("  Uyari: TensorFlow yok, LSTM atlandi.")
        return None

    print(f"\nLSTM egitimi basliyor (pencere={pencere} gun)...")

    d = ozellik_uret(df)
    temel_ozellikler = ["Tmax", "Tmin", "Rs", "VPD", "U2", "sin_doy", "cos_doy", "ET0_lag1", "ET0_ma7", "trend"]
    cols = [c for c in temel_ozellikler if c in d.columns] + ["ET0"]
    if len(cols) < 5:
        print("  Uyari: LSTM icin yeterli ozellik yok, atlandi.")
        return None

    arr = d[cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    arr_sc = scaler.fit_transform(arr)

    X_seq, y_seq = [], []
    for i in range(pencere, len(arr_sc)):
        X_seq.append(arr_sc[i - pencere : i, :-1])
        y_seq.append(arr_sc[i, -1])
    X_seq = np.asarray(X_seq, dtype=float)
    y_seq = np.asarray(y_seq, dtype=float)
    if len(X_seq) < 120:
        print("  Uyari: LSTM icin yeterli sequence yok, atlandi.")
        return None

    split = int(len(X_seq) * 0.80)
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]

    model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=(pencere, X_seq.shape[2])),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=7, verbose=0),
    ]
    history = model.fit(
        X_tr,
        y_tr,
        validation_split=0.15,
        epochs=120,
        batch_size=64,
        callbacks=callbacks,
        verbose=0,
    )

    pred_sc = model.predict(X_te, verbose=0).flatten()

    et0_idx = len(cols) - 1
    et0_std = scaler.scale_[et0_idx]
    et0_mean = scaler.mean_[et0_idx]
    pred_mm = pred_sc * et0_std + et0_mean
    gercek_mm = y_te * et0_std + et0_mean

    m = _metrics(gercek_mm, pred_mm)
    print(f"  LSTM | RMSE={m['rmse']:.3f} | MAE={m['mae']:.3f} | R2={m['r2']:.4f} | Bias={m['bias']:.3f}")

    _lstm_egri(history, out_dir / "lstm_egri.png")
    idx_te = d.index[split + pencere : split + pencere + len(pred_mm)]
    return {
        "model": model,
        "scaler": scaler,
        "pred": pred_mm,
        "gercek": gercek_mm,
        "idx": idx_te,
        "rmse": m["rmse"],
        "mae": m["mae"],
        "r2": m["r2"],
        "bias": m["bias"],
        "history": history,
    }


def _lstm_egri(history, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"], label="Egitim loss")
    ax.plot(history.history["val_loss"], label="Validasyon loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("LSTM ogrenme egrisi")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {out_png}")


def _save_outputs(
    sonuclar_rf: dict[str, dict],
    sonuc_lstm: dict | None,
    suffix: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    metrics_rows = []
    pred_rows = []

    for ad, sn in sonuclar_rf.items():
        metrics_rows.append(
            {
                "model": ad,
                "rmse": sn["rmse"],
                "mae": sn["mae"],
                "r2": sn["r2"],
                "bias": sn["bias"],
                "cv_rmse_mean": sn["cv_rmse_mean"],
                "cv_mae_mean": sn["cv_mae_mean"],
                "cv_r2_mean": sn["cv_r2_mean"],
            }
        )
        for dt, yt, yp in zip(sn["idx"], sn["gercek"], sn["pred"]):
            pred_rows.append(
                {
                    "date": pd.Timestamp(dt).date().isoformat(),
                    "model": ad,
                    "y_true_et0_mm_day": float(yt),
                    "y_pred_et0_mm_day": float(yp),
                    "abs_error_mm_day": float(abs(yp - yt)),
                }
            )

    if sonuc_lstm is not None:
        metrics_rows.append(
            {
                "model": "LSTM",
                "rmse": sonuc_lstm["rmse"],
                "mae": sonuc_lstm["mae"],
                "r2": sonuc_lstm["r2"],
                "bias": sonuc_lstm["bias"],
                "cv_rmse_mean": np.nan,
                "cv_mae_mean": np.nan,
                "cv_r2_mean": np.nan,
            }
        )
        for dt, yt, yp in zip(sonuc_lstm["idx"], sonuc_lstm["gercek"], sonuc_lstm["pred"]):
            pred_rows.append(
                {
                    "date": pd.Timestamp(dt).date().isoformat(),
                    "model": "LSTM",
                    "y_true_et0_mm_day": float(yt),
                    "y_pred_et0_mm_day": float(yp),
                    "abs_error_mm_day": float(abs(yp - yt)),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse").reset_index(drop=True)
    pred_df = pd.DataFrame(pred_rows).sort_values(["model", "date"]).reset_index(drop=True)

    out_metrics = out_dir / f"et0_ml_metrics_{suffix}.csv"
    out_preds = out_dir / f"et0_ml_predictions_{suffix}.csv"
    metrics_df.to_csv(out_metrics, index=False)
    pred_df.to_csv(out_preds, index=False)
    return out_metrics, out_preds


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_source == "wide":
        df = _hazirla_df_wide(args.wide_csv, args.latitude, args.elevation_m)
        min_y = int(df.index.min().year)
        max_y = int(df.index.max().year)
        suffix = args.tag or f"wide_{min_y}_{max_y}"
        print(f"Veri kaynagi: wide ({min_y}-{max_y}), satir={len(df)}")
    else:
        df = _hazirla_df_et0(args.et0_csv, args.water_balance_csv, args.year, args.latitude, args.elevation_m)
        suffix = args.tag or str(args.year)
        print(f"Veri kaynagi: et0_csv ({args.year}), satir={len(df)}")

    sonuclar_rf = rf_model_egit(df, args.out_dir)

    sonuc_lstm = None
    if args.run_lstm:
        sonuc_lstm = lstm_model_egit(df, args.out_dir, pencere=args.lstm_window)
    else:
        print("\nLSTM devre disi (bayrak verilmedi).")

    out_metrics, out_preds = _save_outputs(sonuclar_rf, sonuc_lstm, suffix, args.out_dir)
    print("\nML ciktilari:")
    print(f"- {out_metrics}")
    print(f"- {out_preds}")


if __name__ == "__main__":
    main()
