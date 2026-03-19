#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


ROOT = Path("/Users/yasinkaya/Hackhaton")
OUT_DIR = ROOT / "output" / "report" / "figures_tr"
FONT_PATH = Path("/opt/anaconda3/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")


def configure_matplotlib() -> None:
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 160


def load_csv(rel_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel_path)


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_last_14_days() -> None:
    df = load_csv("output/iski_baraj_api_snapshot/tables/son_14_gun_toplam_doluluk.csv")
    df["tarih"] = pd.to_datetime(df["tarih"])

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(df["tarih"], df["dolulukOrani"], color="#0f766e", linewidth=2.8, marker="o", markersize=4)
    ax.fill_between(df["tarih"], df["dolulukOrani"], color="#99f6e4", alpha=0.35)
    ax.set_title("Son 14 Günde İstanbul Toplam Baraj Doluluğu", fontsize=15, weight="bold")
    ax.set_ylabel("Doluluk (%)")
    ax.set_xlabel("Tarih")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    first = df.iloc[0]
    last = df.iloc[-1]
    ax.annotate(
        f"Başlangıç: %{first['dolulukOrani']:.2f}",
        xy=(first["tarih"], first["dolulukOrani"]),
        xytext=(10, -18),
        textcoords="offset points",
        fontsize=9,
    )
    ax.annotate(
        f"Son değer: %{last['dolulukOrani']:.2f}",
        xy=(last["tarih"], last["dolulukOrani"]),
        xytext=(-85, 12),
        textcoords="offset points",
        fontsize=9,
    )
    fig.autofmt_xdate()
    save(fig, "son_14_gun_doluluk_tr.png")


def fig_weighted_vs_mean() -> None:
    df = load_csv("output/istanbul_dam_deep_research/weighted_total_vs_mean.csv")
    df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(df["date"], df["weighted_total_fill"] * 100, color="#0f766e", linewidth=2.3, label="Kapasite ağırlıklı toplam")
    ax.plot(df["date"], df["overall_mean"] * 100, color="#7c3aed", linewidth=1.8, alpha=0.85, label="Eşit ağırlıklı ortalama")
    ax.set_title("İki Toplam Doluluk Tanımının Karşılaştırması", fontsize=15, weight="bold")
    ax.set_ylabel("Doluluk (%)")
    ax.set_xlabel("Tarih")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "agirlikli_vs_esit_ortalama_tr.png")


def fig_official_supply_vs_proxy() -> None:
    df = load_csv("output/newdata_feature_store/tables/official_supply_vs_model_consumption_monthly.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["city_supply_m3_month_official", "model_consumption_m3_month"]).copy()

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(
        df["date"],
        df["city_supply_m3_month_official"] / 1_000_000,
        color="#0f766e",
        linewidth=2.2,
        label="Resmî şehir suyu (milyon m³/ay)",
    )
    ax.plot(
        df["date"],
        df["model_consumption_m3_month"] / 1_000_000,
        color="#1d4ed8",
        linewidth=2.0,
        label="Model tüketim vekili (milyon m³/ay)",
    )
    if "recorded_water_m3_official" in df.columns:
        sub = df.dropna(subset=["recorded_water_m3_official"])
        if not sub.empty:
            ax.plot(
                sub["date"],
                sub["recorded_water_m3_official"] / 1_000_000,
                color="#b45309",
                linewidth=1.6,
                alpha=0.9,
                label="Resmî kayda alınan su",
            )

    ax.set_title("Resmî Şehir Suyu ile Model Tüketim Vekilinin Karşılaştırması", fontsize=15, weight="bold")
    ax.set_ylabel("Hacim (milyon m³/ay)")
    ax.set_xlabel("Tarih")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "resmi_arz_ve_tuketim_tr.png")


def fig_model_comparison() -> None:
    df = load_csv("output/newdata_feature_store/tables/deepened_feature_model_metrics.csv")
    label_map = {
        "plus_temp_humidity": "Sıcaklık ve nem ekli model",
        "deep_all": "Tüm ek değişkenler",
        "baseline_full": "Temel model",
        "plus_vpd_balance": "VPD ve su dengesi ekli",
    }
    df["label"] = df["model"].map(label_map).fillna(df["model"])
    colors = ["#0f766e", "#0ea5a4", "#1d4ed8", "#b45309"]

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    bars = ax.bar(df["label"], df["rmse_pp"], color=colors[: len(df)], width=0.64)
    ax.set_title("Model Karşılaştırması: Tahmin Hatası (RMSE)", fontsize=15, weight="bold")
    ax.set_ylabel("RMSE (yüzde puan)")
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    for bar, value in zip(bars, df["rmse_pp"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    save(fig, "model_karsilastirma_tr.png")


def fig_scenario_effects() -> None:
    df = load_csv("output/istanbul_dam_deep_research/scenario_summary.csv")
    selected = {
        "rain_plus10_3m": "Yağış +%10",
        "et0_plus10_3m": "ET0 +%10",
        "cons_plus10_3m": "Tüketim +%10",
        "restriction_minus15_3m": "Talep -%15 kısıt",
        "hot_dry_high_demand": "Sıcak-kurak-yüksek talep",
    }
    rows = []
    for key, label in selected.items():
        sub = df[df["scenario"] == key].set_index("horizon_month")
        rows.append({"Senaryo": label, "3. ay": sub.loc[3, "delta_pp"], "6. ay": sub.loc[6, "delta_pp"]})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    x = range(len(plot_df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], plot_df["3. ay"], width=width, color="#1d4ed8", label="3. ay etkisi")
    ax.bar([i + width / 2 for i in x], plot_df["6. ay"], width=width, color="#0f766e", label="6. ay etkisi")
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_title("Senaryo Deneyleri: Doluluk Üzerindeki Etki", fontsize=15, weight="bold")
    ax.set_ylabel("Etkisi (yüzde puan)")
    ax.set_xlabel("Senaryo")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["Senaryo"], rotation=12, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "senaryo_etkileri_tr.png")


def fig_source_status() -> None:
    df = load_csv("output/model_useful_data_bundle/tables/istanbul_source_current_context.csv")
    df = df.sort_values("dolulukOrani", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    bars = ax.barh(df["baslikAdi"], df["dolulukOrani"], color="#0f766e")
    ax.set_title("Baraj Bazında Güncel Doluluk Durumu", fontsize=15, weight="bold")
    ax.set_xlabel("Doluluk (%)")
    ax.set_ylabel("Kaynak")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, storage in zip(bars, df["max_storage_million_m3"]):
        ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2, f"{storage:.1f} m m³", va="center", fontsize=8.5)
    ax.text(
        0.99,
        0.01,
        "Sağdaki küçük etiketler: azami depolama kapasitesi",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#475569",
    )
    save(fig, "kaynak_bazli_guncel_durum_tr.png")


def main() -> None:
    configure_matplotlib()
    fig_last_14_days()
    fig_weighted_vs_mean()
    fig_official_supply_vs_proxy()
    fig_model_comparison()
    fig_scenario_effects()
    fig_source_status()
    print(OUT_DIR)


if __name__ == "__main__":
    main()
