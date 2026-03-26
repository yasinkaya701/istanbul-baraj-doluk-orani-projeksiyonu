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
    add_text(fig, 0.06, y, "Bu çalışmada FAO-56 Penman-Monteith yaklaşımını kullandık. Amaç, sıcaklık, nem, radyasyon ve buhar basıncı açığı bilgisini birleştirerek günlük referans evapotranspirasyonu hesaplamaktı.", 17)

    y -= 0.09
    add_text(fig, 0.06, y, "1. Kullandığımız temel formül", 21, "bold")
    y -= 0.045
    add_text(fig, 0.09, y, "FAO-56 Penman-Monteith:", 17, "bold")
    formula = r"$ET_0 = \frac{0.408\,\Delta\,(R_n-G) + \gamma\,\frac{900}{T+273}\,u_2\,(e_s-e_a)}{\Delta + \gamma\,(1+0.34u_2)}$"
    fig.text(0.50, y - 0.045, formula, ha="center", va="top", fontsize=31, color="#111111", family="DejaVu Serif")

    y -= 0.16
    add_bullet(fig, 0.11, y, r"$R_n$: net radyasyon")
    y -= 0.036
    add_bullet(fig, 0.11, y, r"$G$: toprak ısı akısı")
    y -= 0.036
    add_bullet(fig, 0.11, y, r"$T$: ortalama sıcaklık")
    y -= 0.036
    add_bullet(fig, 0.11, y, r"$u_2$: 2 m rüzgar hızı")
    y -= 0.036
    add_bullet(fig, 0.11, y, r"$e_s-e_a$: buhar basıncı açığı")
    y -= 0.036
    add_bullet(fig, 0.11, y, r"$\Delta$: doygun buhar basıncı eğrisinin eğimi")
    y -= 0.036
    add_bullet(fig, 0.11, y, r"$\gamma$: psikrometrik sabit")

    y -= 0.07
    add_text(fig, 0.06, y, "2. Bu çalışmada kullandığımız sadeleştirmeler", 21, "bold")
    y -= 0.048
    fig.text(0.11, y, r"$T = \frac{T_{\max}+T_{\min}}{2}$", ha="left", va="top", fontsize=21, color="#111111", family="DejaVu Serif")
    add_text(fig, 0.28, y, "Günlük seride en tutarlı ortalama sıcaklık tanımı olduğu için.", 15.5)
    y -= 0.058
    fig.text(0.11, y, r"$\Delta = f(T)$", ha="left", va="top", fontsize=21, color="#111111", family="DejaVu Serif")
    add_text(fig, 0.28, y, "Delta doğrudan sıcaklık değildir; sıcaklıktan fiziksel denklemle türetilir.", 15.5)
    y -= 0.058
    fig.text(0.11, y, r"$G = 0$", ha="left", va="top", fontsize=21, color="#111111", family="DejaVu Serif")
    add_text(fig, 0.28, y, "Günlük ET0 hesabında standart ve savunulabilir kabul olduğu için.", 15.5)
    y -= 0.058
    fig.text(0.11, y, r"$u_2 = 2.0\ \mathrm{m\,s^{-1}}$", ha="left", va="top", fontsize=21, color="#111111", family="DejaVu Serif")
    add_text(fig, 0.28, y, "Uzun dönem kesintisiz rüzgar serisi olmadığı için sabit fallback kullandık.", 15.5)
    y -= 0.058
    fig.text(0.11, y, r"$R_s$: dogrudan radyasyon dosyasindan", ha="left", va="top", fontsize=18.5, color="#111111", family="DejaVu Serif")
    add_text(fig, 0.38, y, "Tahmini radyasyon yerine veri temelli girdi kullanmak istediğimiz için.", 15.5)
    y -= 0.058
    fig.text(0.11, y, r"$\mathrm{coverage\ fraction} \geq 0.80$", ha="left", va="top", fontsize=18.5, color="#111111", family="DejaVu Serif")
    add_text(fig, 0.40, y, "Eksik ayların trendi ve forecasti yapay olarak bozmasını engellemek için.", 15.5)

    y -= 0.085
    add_text(fig, 0.06, y, "3. Delta neden tek günlük değer alındı?", 21, "bold")
    y -= 0.045
    add_bullet(fig, 0.09, y, "Delta sıcaklığa bağlıdır; sıcaklık gün içinde değiştiği için Delta da değişir.")
    y -= 0.04
    add_bullet(fig, 0.09, y, "Öğleden sonra sıcaklık maksimuma yaklaştığı için Delta genelde en yüksek olur.")
    y -= 0.04
    add_bullet(fig, 0.09, y, "Ama bu paket günlük FAO-56 kurduğu için T = (Tmax + Tmin)/2 üzerinden tek bir günlük Delta kullandık.")
    y -= 0.04
    add_bullet(fig, 0.09, y, "Neden: mevcut operasyonel seri 3747 gün, 120 ay ve 10 tam yıl olarak günlük ölçekte kuruldu.")
    y -= 0.04
    add_bullet(fig, 0.09, y, "Saatlik sıcaklık, nem, rüzgar ve radyasyon birlikte varsa saatlik ET0 hesaplanabilir; bu daha ayrıntılı bir katmandır.")

    y -= 0.075
    add_text(fig, 0.06, y, "4. 5 yıllık ortalama ne işe yarar?", 21, "bold")
    y -= 0.045
    add_bullet(fig, 0.09, y, "Yıllık seride kısa dönem oynaklığı azaltır ve ana yönü görünür kılar.")
    y -= 0.04
    add_bullet(fig, 0.09, y, "Aylık seride 60 aylık hareketli ortalama olarak kullanıldığında uzun dönem eğilimi gösterir.")
    y -= 0.04
    add_bullet(fig, 0.09, y, "Tek tek sıçrama yerine kalıcı desen değişimine odaklanmayı sağlar.")

    y -= 0.075
    add_text(fig, 0.06, y, "5. Grafik ölçekleri ve sayısal bağlam", 21, "bold")
    y -= 0.045
    add_bullet(fig, 0.09, y, "Aylık grafik: 1995-2004 arasındaki aylık ET0 serisi")
    y -= 0.04
    add_bullet(fig, 0.09, y, "Yıllık grafik: her yılın toplam ET0 değeri ve 5 yıllık hareketli ortalama")
    y -= 0.04
    add_bullet(fig, 0.09, y, "10 yıllık özet: 1995-2004 penceresinin ortalama aylık deseni")
    y -= 0.04
    add_bullet(fig, 0.09, y, f"Ortalama yıllık ET0: {hist['et0_mm_year_mean']:.1f} mm/yıl | trend: {hist['trend_mm_per_decade']:+.1f} mm/10 yıl")
    y -= 0.04
    add_bullet(fig, 0.09, y, f"Radyasyon günleri: real_extracted={rad['real_extracted_days']} | synthetic={rad['synthetic_days']}")

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
