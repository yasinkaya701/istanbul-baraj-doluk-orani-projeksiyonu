#!/usr/bin/env python3
"""
et_analiz.py - Kuraklik izleme, enerji bilancosu ve CMIP6 benzeri projeksiyon

Bu script, mevcut ET0 ve aylik su butcesi ciktilarindan gunluk analiz dataframe'i
olusturur; sonra:
  1) kuruklik_analizi
  2) enerji_bilancoso
  3) cmip6_projeksiyon
uretimi yapar.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kuraklik + enerji + projeksiyon analizi")
    p.add_argument("--year", type=int, default=1987, help="Analiz yili")
    p.add_argument(
        "--et0-csv",
        type=Path,
        default=Path("output/spreadsheet/et0_inputs_completed_1987.csv"),
        help="Gunluk ET0 girdi CSV",
    )
    p.add_argument(
        "--water-balance-csv",
        type=Path,
        default=Path("output/spreadsheet/water_balance_partial_1987.csv"),
        help="Aylik yagis/su butcesi CSV",
    )
    p.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/spreadsheet"),
        help="Cikti klasoru",
    )
    return p.parse_args()


def _safe_savgol(values: np.ndarray, max_window: int = 11, poly: int = 2) -> np.ndarray:
    n = len(values)
    if n < 5:
        return values
    w = min(max_window, n if n % 2 == 1 else n - 1)
    if w < 5:
        return values
    return savgol_filter(values, window_length=w, polyorder=min(poly, w - 2))


def _load_daily_input(et0_csv: Path, wb_csv: Path, year: int) -> pd.DataFrame:
    et0 = pd.read_csv(et0_csv)
    req = {"date", "et0_completed_mm_day", "t_mean_c", "rh_mean_pct", "u2_m_s", "rs_mj_m2_day"}
    miss = req - set(et0.columns)
    if miss:
        raise ValueError(f"{et0_csv} eksik kolonlar: {sorted(miss)}")

    et0["date"] = pd.to_datetime(et0["date"], errors="coerce")
    et0 = et0.dropna(subset=["date"]).copy()
    et0 = et0[et0["date"].dt.year == year].sort_values("date").reset_index(drop=True)
    if et0.empty:
        raise ValueError(f"{year} icin ET0 satiri bulunamadi: {et0_csv}")

    base = et0[["date", "et0_completed_mm_day", "t_mean_c", "rh_mean_pct", "u2_m_s", "rs_mj_m2_day"]].copy()
    base["days_in_month"] = base["date"].dt.days_in_month.astype(int)
    base["month"] = base["date"].dt.to_period("M").astype(str)

    if wb_csv.exists():
        wb = pd.read_csv(wb_csv)
        need = {"month", "precip_obs_mm"}
        miss2 = need - set(wb.columns)
        if miss2:
            raise ValueError(f"{wb_csv} eksik kolonlar: {sorted(miss2)}")
        wb = wb[["month", "precip_obs_mm"]].copy()
        wb["precip_obs_mm"] = pd.to_numeric(wb["precip_obs_mm"], errors="coerce").fillna(0.0)
        base = base.merge(wb, on="month", how="left")
    else:
        base["precip_obs_mm"] = 0.0

    base["precip_obs_mm"] = base["precip_obs_mm"].fillna(0.0)
    # Gunluk yagis proxy: aylik toplam / ay gun sayisi
    base["P"] = np.where(base["days_in_month"] > 0, base["precip_obs_mm"] / base["days_in_month"], 0.0)
    base["ET0"] = pd.to_numeric(base["et0_completed_mm_day"], errors="coerce").fillna(0.0)
    base["Tmean"] = pd.to_numeric(base["t_mean_c"], errors="coerce").fillna(method="ffill").fillna(method="bfill")

    rs = pd.to_numeric(base["rs_mj_m2_day"], errors="coerce")
    rs = rs.fillna(rs.median())
    rh = pd.to_numeric(base["rh_mean_pct"], errors="coerce").fillna(60.0)
    u2 = pd.to_numeric(base["u2_m_s"], errors="coerce").fillna(2.0)

    # Enerji terimleri proxy (Rn, LE, H, G_flux, EF)
    # Rn: net kisadalga yaklasimi, EF: nem+ruzgar tabanli proxy
    base["Rn"] = 0.77 * rs
    ef = np.clip(0.15 + 0.008 * rh - 0.015 * u2, 0.15, 0.90)
    base["EF"] = ef
    base["LE"] = base["Rn"] * base["EF"]
    base["H"] = base["Rn"] - base["LE"]
    base["G_flux"] = 0.10 * base["Rn"]

    out = base[["date", "P", "ET0", "Tmean", "Rn", "LE", "H", "G_flux", "EF"]].copy()
    out = out.set_index("date").sort_index()
    return out


def spi_hesapla(yagis: pd.Series, pencere: int = 12) -> pd.Series:
    """Standart Yagis Indeksi (log-normal standardizasyonu)."""
    kayan = yagis.rolling(pencere, min_periods=pencere).sum()
    spi = pd.Series(np.nan, index=yagis.index, dtype=float)
    for ay in range(1, 13):
        maske = kayan.index.month == ay
        deger = kayan[maske].dropna()
        if len(deger) < 15:
            continue
        log_d = np.log(deger.clip(lower=0.01))
        mu, sigma = float(log_d.mean()), float(log_d.std())
        if not np.isfinite(sigma) or sigma <= 1e-12:
            continue
        hedef = kayan[maske & kayan.notna()].clip(lower=0.01)
        spi.loc[hedef.index] = (np.log(hedef) - mu) / sigma
    return spi


def kuruklik_analizi(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SPI, SPEI, su stresi trendi ve donemsel kuraklik olaylari."""
    print("\nKuraklik analizi...")

    aylik = df.resample("ME").agg({"P": "sum", "ET0": "sum", "Tmean": "mean", "Rn": "mean", "LE": "mean"})
    aylik["SSI"] = aylik["P"] - aylik["ET0"]
    aylik["SPI3"] = spi_hesapla(aylik["P"], 3)
    aylik["SPI12"] = spi_hesapla(aylik["P"], 12)

    # SPEI benzeri: su stresi serisini pozitif banda kaydirip standardize et.
    aylik["SPEI12"] = spi_hesapla(aylik["SSI"].clip(lower=-500) + 600, 12)
    if aylik["SPEI12"].std() and np.isfinite(aylik["SPEI12"].std()):
        aylik["SPEI12"] = (aylik["SPEI12"] - aylik["SPEI12"].mean()) / aylik["SPEI12"].std()

    yillik = aylik.resample("YE").agg({"P": "sum", "ET0": "sum", "SSI": "sum", "SPI12": "mean", "SPEI12": "mean"})
    yillik["yil"] = yillik.index.year

    print("\n  Mann-Kendall trend testi:")
    for kol in ["P", "ET0", "SSI"]:
        seri = yillik[kol].dropna()
        if len(seri) < 4:
            print(f"    {kol:6s}: yetersiz veri")
            continue
        tau, pval = stats.kendalltau(range(len(seri)), seri)
        yon = "Artis" if tau > 0 else "Azalis"
        sig = "anlamli" if pval < 0.05 else "anlamsiz"
        print(f"    {kol:6s}: {yon:6s} | tau={tau:+.3f} | p={pval:.4f} ({sig})")

    _kuruklik_grafik(aylik, yillik, out_dir / "kuruklik_analizi.png")
    return aylik, yillik


def _kuruklik_grafik(aylik: pd.DataFrame, yillik: pd.DataFrame, out_png: Path) -> None:
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle("Kuraklik Izleme Analizi", fontsize=15, fontweight="bold")

    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(yillik.index, yillik["P"], alpha=0.4, color="#1565C0", label="Yagis (P)")
    ax.plot(yillik.index, yillik["ET0"], color="#C62828", linewidth=1.5, label="ET0")
    for kol, renk in [("P", "#1565C0"), ("ET0", "#C62828")]:
        seri = yillik[kol].dropna()
        if len(seri) > 1:
            z = np.polyfit(range(len(seri)), seri, 1)
            ax.plot(seri.index, np.poly1d(z)(range(len(seri))), "--", color=renk, linewidth=1.2, alpha=0.8)
    ax.set_ylabel("mm/yil")
    ax.set_title("Yillik Yagis ve ET0 (kesikli = trend)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    renkler = ["#C62828" if v < 0 else "#1565C0" for v in yillik["SSI"]]
    ax.bar(yillik.index, yillik["SSI"], color=renkler, alpha=0.75, width=300)
    ax.axhline(0, color="black", linewidth=0.8)
    ssi_ma = yillik["SSI"].rolling(10, center=True).mean()
    ax.plot(yillik.index, ssi_ma, "k-", linewidth=2, label="10-yil ort.")
    ax.set_ylabel("P - ET0 (mm/yil)")
    ax.set_title("Su Stresi Indeksi")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 0])
    spi = aylik["SPI12"].dropna()
    ax.fill_between(spi.index, spi, 0, where=spi < 0, color="#C62828", alpha=0.55, label="Kurak")
    ax.fill_between(spi.index, spi, 0, where=spi >= 0, color="#1565C0", alpha=0.35, label="Nemli")
    ax.axhline(-1, color="orange", linestyle="--", linewidth=0.9, label="Orta kurak")
    ax.axhline(-2, color="orange", linestyle=":", linewidth=0.9, label="Asiri kurak")
    ax.set_ylabel("SPI-12")
    ax.set_title("Standart Yagis Indeksi (12 ay)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    spei = aylik["SPEI12"].dropna()
    ax.fill_between(spei.index, spei, 0, where=spei < 0, color="#BF360C", alpha=0.55, label="Kurak")
    ax.fill_between(spei.index, spei, 0, where=spei >= 0, color="#0277BD", alpha=0.35, label="Nemli")
    ax.axhline(-1, color="orange", linestyle="--", linewidth=0.9)
    ax.axhline(-2, color="red", linestyle=":", linewidth=0.9)
    ax.set_ylabel("SPEI-12")
    ax.set_title("Standart P-ET0 Indeksi (12 ay)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {out_png}")


def enerji_bilancoso(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Rn = LE + H + G yaklasik dagilimi | EF trendi."""
    print("\nEnerji bilancosu analizi...")

    aylik = df[["Rn", "LE", "H", "G_flux", "EF"]].resample("ME").mean()
    mevsim = df[["Rn", "LE", "H", "G_flux"]].groupby(df.index.month).mean()
    dekad = df.groupby((df.index.year // 10) * 10)[["Rn", "LE", "H"]].mean()
    yillik_ef = df["EF"].resample("YE").mean()
    yillik_le = df["LE"].resample("YE").sum()

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
    fig.suptitle("Enerji Bilancosu Analizi", fontsize=15, fontweight="bold")
    aylar = ["Oca", "Sub", "Mar", "Nis", "May", "Haz", "Tem", "Agu", "Eyl", "Eki", "Kas", "Ara"]

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(12)
    g = 0.22
    ax.bar(x - g, mevsim["Rn"], g * 1.8, label="Rn", color="#F57C00", alpha=0.85)
    ax.bar(x, mevsim["LE"], g * 1.8, label="LE (Lat.)", color="#1976D2", alpha=0.85)
    ax.bar(x + g, mevsim["H"], g * 1.8, label="H (Duy.)", color="#C62828", alpha=0.85)
    ax.bar(x + g * 2, mevsim["G_flux"], g * 1.8, label="G (Toprak)", color="#558B2F", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(aylar, fontsize=7)
    ax.set_ylabel("MJ/m2/gun")
    ax.set_title("Mevsimsel Enerji Dagilimi")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(yillik_le.index, yillik_le.values, color="#1976D2", linewidth=1, alpha=0.6)
    if len(yillik_le) > 1:
        z = np.polyfit(range(len(yillik_le)), yillik_le.values, 1)
        ax.plot(yillik_le.index, np.poly1d(z)(range(len(yillik_le))), "r--", linewidth=2, label=f"Trend: {z[0]:+.1f} MJ/yil2")
    ma = yillik_le.rolling(10, center=True).mean()
    ax.plot(ma.index, ma.values, "b-", linewidth=2, label="10-yil ort.")
    ax.set_ylabel("LE (MJ/m2/yil)")
    ax.set_title("Yillik Latent Heat Trendi")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(yillik_ef.index, yillik_ef.values, color="#2E7D32", linewidth=1, alpha=0.6)
    ma_ef = yillik_ef.rolling(10, center=True).mean()
    ax.plot(ma_ef.index, ma_ef.values, color="#1B5E20", linewidth=2, label="10-yil ort.")
    ax.fill_between(
        yillik_ef.index,
        yillik_ef.values,
        0.5,
        where=yillik_ef.values < 0.5,
        alpha=0.2,
        color="red",
        label="EF < 0.5 (su stresi)",
    )
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("EF = LE/Rn")
    ax.set_title("Evaporatif Fraksiyon - Su Stresi Gostergesi")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    x_d = np.arange(len(dekad))
    g2 = 0.25
    ax.bar(x_d - g2, dekad["Rn"], g2 * 1.8, label="Rn", color="#F57C00", alpha=0.85)
    ax.bar(x_d, dekad["LE"], g2 * 1.8, label="LE", color="#1976D2", alpha=0.85)
    ax.bar(x_d + g2, dekad["H"], g2 * 1.8, label="H", color="#C62828", alpha=0.85)
    ax.set_xticks(x_d)
    ax.set_xticklabels([f"{y}s" for y in dekad.index], fontsize=9)
    ax.set_ylabel("MJ/m2/gun")
    ax.set_title("Dekad Bazli Enerji Ortalamalari")
    ax.legend()
    ax.grid(alpha=0.3)

    out_png = out_dir / "enerji_bilancoso.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {out_png}")
    return aylik


def cmip6_projeksiyon(yillik: pd.DataFrame, out_dir: Path, seed: int = 42) -> pd.DataFrame:
    """SSP1-2.6, SSP2-4.5, SSP5-8.5 icin ET0/P projeksiyonu."""
    print("\nCMIP6 ET0 projeksiyonu...")
    rng = np.random.default_rng(seed)

    gecmis = yillik[["ET0", "P"]].dropna().copy()
    gecmis["yil"] = gecmis.index.year
    baz_et = float(gecmis["ET0"].mean()) if not gecmis.empty else 900.0
    baz_p = float(gecmis["P"].mean()) if not gecmis.empty else 700.0
    baz_ssi = float((gecmis["P"] - gecmis["ET0"]).mean()) if not gecmis.empty else (baz_p - baz_et)

    if len(gecmis) >= 15:
        m_et, b_et, r, p_val, se = stats.linregress(gecmis["yil"], gecmis["ET0"])
        m_p, b_p, *_ = stats.linregress(gecmis["yil"], gecmis["P"])
        print(f"  ET0 trend: {m_et:+.2f} mm/yil | R2={r**2:.3f} | p={p_val:.4f}")
    else:
        # Kisa seride trendi bazdan baslat; degiskenligi yillik oynakliktan al.
        m_et, b_et = 0.0, baz_et
        m_p, b_p = 0.0, baz_p
        se = float(max(10.0, gecmis["ET0"].std() if len(gecmis) > 1 else 20.0))
        r, p_val = 0.0, 1.0
        print("  Uyari: gecmis yillik seri kisa, trend fallback modu kullanildi.")
        print(f"  ET0 trend (fallback): {m_et:+.2f} mm/yil | R2={r**2:.3f} | p={p_val:.4f}")

    senaryolar = {
        "SSP1-2.6": {"renk": "#43A047", "dT_2100": 1.5, "dP_2100": -0.05, "etiket": "Dusuk emisyon"},
        "SSP2-4.5": {"renk": "#FB8C00", "dT_2100": 2.5, "dP_2100": -0.10, "etiket": "Orta emisyon"},
        "SSP5-8.5": {"renk": "#E53935", "dT_2100": 4.5, "dP_2100": -0.20, "etiket": "Yuksek emisyon"},
    }
    gelecek_yillar = np.arange(2026, 2101)

    proj_dict: dict[str, dict[str, np.ndarray]] = {}
    for ssp, bilgi in senaryolar.items():
        et_proj, p_proj = [], []
        for yil in gelecek_yillar:
            t_frac = (yil - 2026) / 74.0
            dT = bilgi["dT_2100"] * t_frac
            dP_oran = bilgi["dP_2100"] * t_frac
            gurultu = float(rng.normal(0, se * 8))
            et_val = (m_et * yil + b_et) + baz_et * 0.038 * dT + gurultu
            p_val2 = (m_p * yil + b_p) * (1 + dP_oran) + float(rng.normal(0, 30))
            et_proj.append(et_val)
            p_proj.append(p_val2)
        proj_dict[ssp] = {"ET0": np.array(et_proj), "P": np.array(p_proj)}

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
    fig.suptitle("CMIP6 Senaryolari: ET0 ve Su Dengesi Projeksiyonu (2026-2100)", fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, :])
    gec_ma = gecmis["ET0"].rolling(5, center=True).mean()
    ax1.plot(gecmis.index, gecmis["ET0"], color="#9E9E9E", linewidth=0.6, alpha=0.5)
    ax1.plot(gec_ma.index, gec_ma.values, "k-", linewidth=1.5, label="Gozlem")
    for ssp, bilgi in senaryolar.items():
        sm = _safe_savgol(proj_dict[ssp]["ET0"])
        ax1.plot(gelecek_yillar, sm, color=bilgi["renk"], linewidth=2.2, label=f"{ssp} - {bilgi['etiket']}")
        ax1.fill_between(gelecek_yillar, sm - se * 15, sm + se * 15, color=bilgi["renk"], alpha=0.1)
    ax1.axvline(2026, color="gray", linestyle=":", linewidth=1.5)
    ax1.set_ylabel("Yillik ET0 (mm/yil)")
    ax1.set_title("Referans ET0 Projeksiyonu")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(gecmis.index, gecmis["P"], color="#9E9E9E", linewidth=0.6, alpha=0.5)
    p_ma = gecmis["P"].rolling(5, center=True).mean().dropna()
    ax2.plot(p_ma.index, p_ma.values, "k-", linewidth=1.5, label="Gozlem")
    for ssp, bilgi in senaryolar.items():
        sm = _safe_savgol(proj_dict[ssp]["P"])
        ax2.plot(gelecek_yillar, sm, color=bilgi["renk"], linewidth=1.8, label=ssp)
    ax2.axvline(2026, color="gray", linestyle=":", linewidth=1.2)
    ax2.set_ylabel("Yillik Yagis (mm)")
    ax2.set_title("Yagis Projeksiyonu")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    for ssp, bilgi in senaryolar.items():
        ssi = proj_dict[ssp]["P"] - proj_dict[ssp]["ET0"]
        anom = ssi - baz_ssi
        sm = _safe_savgol(anom)
        ax3.plot(gelecek_yillar, sm, color=bilgi["renk"], linewidth=1.8, label=ssp)
        ax3.fill_between(gelecek_yillar, sm, 0, where=sm < 0, color=bilgi["renk"], alpha=0.08)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_ylabel("P - ET0 Anomalisi (mm/yil)")
    ax3.set_title("Su Stresi Degisimi (2026 baza gore)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[2, :])
    ssps = list(senaryolar.keys())
    ref = m_et * 2026 + b_et
    et2050, et2100 = [], []
    for ssp in ssps:
        idx50 = int(np.where(gelecek_yillar == 2050)[0][0])
        idx00 = int(np.where(gelecek_yillar == 2100)[0][0])
        et2050.append(float(proj_dict[ssp]["ET0"][idx50] - ref))
        et2100.append(float(proj_dict[ssp]["ET0"][idx00] - ref))
    x = np.arange(len(ssps))
    g = 0.3
    b1 = ax4.bar(x - g / 2, et2050, g, label="2050 anomalisi", color=[senaryolar[s]["renk"] for s in ssps], alpha=0.6, hatch="//")
    b2 = ax4.bar(x + g / 2, et2100, g, label="2100 anomalisi", color=[senaryolar[s]["renk"] for s in ssps], alpha=0.9)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(ssps)
    ax4.set_ylabel("ET0 Artisi (mm/yil, 2026 bazi)")
    ax4.set_title("Senaryo Bazli ET0 Artis Miktari")
    ax4.legend()
    ax4.grid(alpha=0.3)
    for rect in [*b1, *b2]:
        h = rect.get_height()
        ax4.annotate(
            f"{h:+.0f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    out_png = out_dir / "cmip6_projeksiyon.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {out_png}")

    rows = []
    for ssp in ssps:
        etv = proj_dict[ssp]["ET0"]
        pv = proj_dict[ssp]["P"]
        ssiv = pv - etv
        anom = ssiv - baz_ssi
        for yil, e0, pp, ssi, aa in zip(gelecek_yillar, etv, pv, ssiv, anom):
            rows.append({"year": int(yil), "scenario": ssp, "et0_mm_year": float(e0), "p_mm_year": float(pp), "ssi_mm_year": float(ssi), "ssi_anomaly_mm_year": float(aa)})
    proj_df = pd.DataFrame(rows)
    return proj_df


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_daily_input(args.et0_csv, args.water_balance_csv, args.year)
    aylik_k, yillik_k = kuruklik_analizi(df, args.out_dir)
    aylik_e = enerji_bilancoso(df, args.out_dir)
    proj = cmip6_projeksiyon(yillik_k, args.out_dir, seed=args.seed)

    aylik_k.to_csv(args.out_dir / f"kuruklik_aylik_{args.year}.csv")
    yillik_k.to_csv(args.out_dir / f"kuruklik_yillik_{args.year}.csv")
    aylik_e.to_csv(args.out_dir / f"enerji_aylik_{args.year}.csv")
    proj.to_csv(args.out_dir / f"cmip6_projeksiyon_{args.year}_to_2100.csv", index=False)
    print("\nCSV ciktilari kaydedildi.")


if __name__ == "__main__":
    main()
