#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ET0 formula explainer image.")
    parser.add_argument("--summary-json", type=Path, default=Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_real_radiation/reports/tarim_et0_real_radiation_summary.json"))
    parser.add_argument("--out-png", type=Path, default=Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_real_radiation/charts/tarim_et0_formula_explained.png"))
    parser.add_argument("--label", type=str, default="Tarimsal", help="Context label (e.g., Baraj / Tarimsal).")
    return parser.parse_args()

def add_text(fig: plt.Figure, x: float, y: float, text: str, size: float, weight: str = "normal") -> None:
    fig.text(x, y, text, ha="left", va="top", fontsize=size, fontweight=weight, color="#111111", family="DejaVu Sans")

def add_bullet(fig: plt.Figure, x: float, y: float, text: str, size: float = 15.5) -> None:
    fig.text(x, y, f"•  {text}", ha="left", va="top", fontsize=size, color="#111111", family="DejaVu Sans")

def main() -> None:
    args = parse_args()
    label = args.label.strip()
    lower_label = label.lower()
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    cov = summary["coverage"]
    hist = summary["historical_stats"]
    rad = summary["radiation_input"]
    fig = plt.figure(figsize=(16, 14.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = 0.95
    add_text(fig, 0.06, y, f"{label} ET0 Modeli", 27, "bold")
    y -= 0.055
    add_text(fig, 0.06, y, "Bu calismada FAO-56 Penman-Monteith yaklasimini kullandik. Amac, sicaklik, nem, radyasyon ve buhar basinci acigi bilgisini birlestirerek gunluk referans evapotranspirasyonu hesaplamakti.", 17)

    y -= 0.09
    add_text(fig, 0.06, y, "1. Kullandigimiz temel formul", 21, "bold")
    y -= 0.045
    add_text(fig, 0.09, y, "FAO-56 Penman-Monteith:", 17, "bold")
    formula = r"$ET_0 = \frac{0.408\,\Delta\,(R_n-G) + \gamma\,\frac{900}{T+273}\,u_2\,(e_s-e_a)}{\Delta + \gamma\,(1+0.34u_2)}$"
    fig.text(0.50, y - 0.045, formula, ha="center", va="top", fontsize=31, color="#111111", family="DejaVu Serif")

    y -= 0.16
    add_bullet(fig, 0.11, y, r"$R_n$: net radyasyon")
    y -= 0.036
    add_bullet(fig, 0.11, y, r"$G$: toprak isi akisi")
    y -= 0.036
    add_bullet(fig, 0.11, y, r"$T$: ortalama sicaklik")
    y -= 0.036
    add_bullet(fig, 0.11, y, r"$u_2$: 2 m ruzgar hizi")
    y -= 0.036
    add_bullet(fig, 0.11, y, r"$e_s-e_a$: buhar basinci acigi")
    y -= 0.036
    add_bullet(fig, 0.11, y, r"$\Delta$: doygun buhar basinci egrisinin egimi")
    y -= 0.036
    add_bullet(fig, 0.11, y, r"$\gamma$: psikrometrik sabit")

    y -= 0.07
    add_text(fig, 0.06, y, "2. Bu calismada kullandigimiz sadelestirmeler", 21, "bold")
    y -= 0.048
    fig.text(0.11, y, r"$T = \frac{T_{\max}+T_{\min}}{2}$", ha="left", va="top", fontsize=21, color="#111111", family="DejaVu Serif")
    add_text(fig, 0.28, y, "Gunluk seride en tutarli ortalama sicaklik tanimi oldugu icin.", 15.5)
    y -= 0.058
    fig.text(0.11, y, r"$\Delta = f(T)$", ha="left", va="top", fontsize=21, color="#111111", family="DejaVu Serif")
    add_text(fig, 0.28, y, "Delta dogrudan sicaklik degildir; sicakliktan fiziksel denklemle turetilir.", 15.5)
    y -= 0.058
    fig.text(0.11, y, r"$G = 0$", ha="left", va="top", fontsize=21, color="#111111", family="DejaVu Serif")
    add_text(fig, 0.28, y, "Gunluk ET0 hesabinda standart ve savunulabilir kabul oldugu icin.", 15.5)
    y -= 0.058
    fig.text(0.11, y, r"$u_2 = 2.0\ \mathrm{m\,s^{-1}}$", ha="left", va="top", fontsize=21, color="#111111", family="DejaVu Serif")
    add_text(fig, 0.28, y, "Uzun donem kesintisiz ruzgar serisi olmadigi icin sabit fallback kullandik.", 15.5)
    y -= 0.058
    fig.text(0.11, y, r"$R_s$: dogrudan radyasyon dosyasindan", ha="left", va="top", fontsize=18.5, color="#111111", family="DejaVu Serif")
    add_text(fig, 0.38, y, "Tahmini radyasyon yerine veri temelli girdi kullanmak istedigimiz icin.", 15.5)
    y -= 0.058
    fig.text(0.11, y, r"$\mathrm{coverage\ fraction} \geq 0.80$", ha="left", va="top", fontsize=18.5, color="#111111", family="DejaVu Serif")
    add_text(fig, 0.40, y, "Eksik aylarin trendi ve forecasti yapay olarak bozmasini engellemek icin.", 15.5)

    y -= 0.085
    add_text(fig, 0.06, y, "3. Delta neden tek gunluk deger alindi?", 21, "bold")
    y -= 0.045
    add_bullet(fig, 0.09, y, "Delta sicakliga baglidir; sicaklik gun icinde degistigi icin Delta da degisir.")
    y -= 0.04
    add_bullet(fig, 0.09, y, "Ogleden sonra sicaklik maksimuma yaklastigi icin Delta genelde en yuksek olur.")
    y -= 0.04
    add_bullet(fig, 0.09, y, "Ama bu paket gunluk FAO-56 kurdugu icin T = (Tmax + Tmin)/2 uzerinden tek bir gunluk Delta kullandik.")
    y -= 0.04
    add_bullet(fig, 0.09, y, "Neden: mevcut operasyonel seri 3747 gun, 120 ay ve 10 tam yil olarak gunluk olcekte kuruldu.")
    y -= 0.04
    add_bullet(fig, 0.09, y, "Saatlik sicaklik, nem, ruzgar ve radyasyon birlikte varsa saatlik ET0 hesaplanabilir; bu daha ayrintili bir katmandir.")

    y -= 0.075
    add_text(fig, 0.06, y, "4. 5 yillik ortalama ne ise yarar?", 21, "bold")
    y -= 0.045
    add_bullet(fig, 0.09, y, "Yillik seride kisa donem oynakligi azaltir ve ana yonu gorunur kilar.")
    y -= 0.04
    add_bullet(fig, 0.09, y, "Aylik seride 60 aylik hareketli ortalama olarak kullanildiginda uzun donem egilimi gosterir.")
    y -= 0.04
    add_bullet(fig, 0.09, y, "Tek tek sicrama yerine kalici desen degisimine odaklanmayi saglar.")

    y -= 0.075
    add_text(fig, 0.06, y, "5. Grafik olcekleri ve sayisal baglam", 21, "bold")
    y -= 0.045
    add_bullet(fig, 0.09, y, "Aylik grafik: 1995-2004 arasindaki aylik ET0 serisi")
    y -= 0.04
    add_bullet(fig, 0.09, y, "Yillik grafik: her yilin toplam ET0 degeri ve 5 yillik hareketli ortalama")
    y -= 0.04
    add_bullet(fig, 0.09, y, "10 yillik ozet: 1995-2004 penceresinin ortalama aylik deseni")
    y -= 0.04
    add_bullet(fig, 0.09, y, f"Ortalama yillik ET0: {hist['et0_mm_year_mean']:.1f} mm/yil | trend: {hist['trend_mm_per_decade']:+.1f} mm/10 yil")
    y -= 0.04
    add_bullet(fig, 0.09, y, f"Radyasyon gunleri: real_extracted={rad['real_extracted_days']} | synthetic={rad['synthetic_days']}")

    y -= 0.075
    if "baraj" in lower_label:
        footer = "Model penceresi: {} - {}    |    Sonraki katman: Acik su buharlasma katsayisi (K)".format(
            cov["model_start"], cov["model_end"]
        )
    else:
        footer = "Model penceresi: {} - {}    |    Sonraki katman: ETc = Kc x ET0".format(
            cov["model_start"], cov["model_end"]
        )
    add_text(fig, 0.06, y, footer, 17, "bold")

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote: {args.out_png}")

if __name__ == "__main__":
    main()
