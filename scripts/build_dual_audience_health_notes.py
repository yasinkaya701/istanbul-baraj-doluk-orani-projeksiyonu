#!/usr/bin/env python3
"""Build dual-audience health notes (executive + technical) and a decision visual."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build executive and technical notes from health outputs.")
    p.add_argument("--root-dir", type=Path, default=Path("output/health_impact"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--date-label", type=str, default=str(date.today()))
    p.add_argument("--model-summary-csv", type=Path, default=None)
    p.add_argument("--traffic-csv", type=Path, default=None)
    p.add_argument("--disease-csv", type=Path, default=None)
    return p.parse_args()


def resolve_model_summary(root: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    for p in [
        root / "model_comparison_summary_stable_calibrated.csv",
        root / "model_comparison_summary_duzenlenmis_run.csv",
    ]:
        if p.exists():
            return p
    return root / "model_comparison_summary.csv"


def resolve_traffic_csv(root: Path, override: Path | None, date_label: str) -> Path | None:
    if override is not None:
        return override
    dated = root / f"literatur_uyum_trafik_isigi_{date_label}.csv"
    if dated.exists():
        return dated
    matches = sorted(root.glob("literatur_uyum_trafik_isigi_*.csv"))
    if not matches:
        return None
    return matches[-1]


def resolve_disease_csv(root: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    return root / "health_all_disease_summary.csv"


def pick_row(model_df: pd.DataFrame, needle: str) -> pd.Series:
    labels = model_df["model"].astype(str).str.lower()
    d = model_df[labels == needle.lower()]
    if d.empty:
        d = model_df[labels.str.contains(needle.lower(), regex=False)]
    if d.empty:
        raise ValueError(f"Cannot find model row for: {needle}")
    return d.iloc[0]


def top_groups(disease_df: pd.DataFrame, model_name: str, n: int = 3) -> list[str]:
    labels = disease_df["model"].astype(str).str.lower()
    d = disease_df[labels.str.contains(model_name.lower(), regex=False)].copy()
    if d.empty:
        return []
    d = d.sort_values("direct_signal_score", ascending=False).head(n)
    return [str(x) for x in d["disease_group_tr"].tolist()]


def traffic_pct(traffic_df: pd.DataFrame | None, model_needle: str, label: str) -> float:
    if traffic_df is None:
        return float("nan")
    d = traffic_df[traffic_df["model"].astype(str).str.lower().str.contains(model_needle.lower(), regex=False)]
    d = d[d["uyum_etiketi"] == label]
    if d.empty:
        return 0.0
    return float(d["oran_yuzde"].iloc[0])


def safe_read_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_executive_md(
    out_path: Path,
    date_label: str,
    strong: pd.Series,
    quant: pd.Series,
    strong_traffic: dict[str, float],
    quant_traffic: dict[str, float],
) -> str:
    strong_pct = (float(strong["future_rr_mean"]) - 1.0) * 100.0
    quant_pct = (float(quant["future_rr_mean"]) - 1.0) * 100.0
    one_liner = (
        f"2026-2035 projeksiyonunda strong modelde isi-kaynakli risk +%{strong_pct:.1f} "
        f"(RR {float(strong['future_rr_mean']):.3f}), quant modelde +%{quant_pct:.1f} "
        f"(RR {float(quant['future_rr_mean']):.3f}); kapasite planini strong, rutin izlemeyi quant tabaninda yonetin."
    )

    lines = [
        f"# Yonetici Tek Cumle Ozet ({date_label})",
        "",
        f"- {one_liner}",
        "",
        "## 3 Kisa Not",
        (
            f"- Esik ustu ay orani: strong %{float(strong['future_threshold_exceed_share']) * 100:.1f}, "
            f"quant %{float(quant['future_threshold_exceed_share']) * 100:.1f}."
        ),
        (
            f"- OOD artis payi: strong %{float(strong['delta_ood_share']) * 100:.1f}, "
            f"quant %{float(quant['delta_ood_share']) * 100:.1f}."
        ),
        (
            f"- Literatur trafik ozeti: strong yesil/sari/kirmizi = "
            f"%{strong_traffic['yesil']:.1f}/%{strong_traffic['sari']:.1f}/%{strong_traffic['kirmizi']:.1f}, "
            f"quant yesil/sari/kirmizi = "
            f"%{quant_traffic['yesil']:.1f}/%{quant_traffic['sari']:.1f}/%{quant_traffic['kirmizi']:.1f}."
        ),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return one_liner


def build_technical_md(
    out_path: Path,
    date_label: str,
    strong: pd.Series,
    quant: pd.Series,
    strong_traffic: dict[str, float],
    quant_traffic: dict[str, float],
    top3: list[str],
) -> None:
    lines = [
        f"# Teknik Ekip Kisa Aksiyon Ozeti ({date_label})",
        "",
        "## Durum",
        f"- Strong future RR: {float(strong['future_rr_mean']):.4f} | delta RR: {float(strong['delta_rr_mean']):+.4f}",
        f"- Quant future RR: {float(quant['future_rr_mean']):.4f} | delta RR: {float(quant['delta_rr_mean']):+.4f}",
        (
            f"- Literatur trafik: strong yesil/sari/kirmizi %{strong_traffic['yesil']:.1f}/"
            f"%{strong_traffic['sari']:.1f}/%{strong_traffic['kirmizi']:.1f}; "
            f"quant yesil/sari/kirmizi %{quant_traffic['yesil']:.1f}/"
            f"%{quant_traffic['sari']:.1f}/%{quant_traffic['kirmizi']:.1f}."
        ),
        "",
        "## Hedefli Teknik Is Listesi (ilk sprint)",
        "1. Strong senaryoda RR buyuklugunu surukleyen aylari ayri et ve etki dagilimini yaz.",
        "2. UV + PM2.5/O3 + polen proxy katmanlari icin veri boru hatti ac.",
        "3. DLNM kalibrasyonunu gunluk saglik sonlanimlari ile tekrar kos.",
        "4. Risk esiklerini ilce bazli karar kurallarina cevir (erken uyari tetikleyicisi).",
        "5. Model secimi icin iki katmanli politika uygula: kapasite=strong, izleme=quant.",
        "6. Aylik literatur uyum raporunu otomatik yenile.",
        "",
        "## Onceleyecegin Hastalik Gruplari (strong top-3)",
    ]
    if top3:
        lines.extend([f"- {item}" for item in top3])
    else:
        lines.append("- Top grup bilgisi bulunamadi.")
    lines.extend(["", "## Not", "- Cilt kanseri icin UV katmani olmadan sayisal artis yuzdesi verilmemeli.", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_csv(
    out_path: Path,
    date_label: str,
    strong: pd.Series,
    quant: pd.Series,
    strong_traffic: dict[str, float],
    quant_traffic: dict[str, float],
    one_liner: str,
) -> None:
    rec = {
        "date": date_label,
        "strong_future_rr": float(strong["future_rr_mean"]),
        "quant_future_rr": float(quant["future_rr_mean"]),
        "strong_delta_rr": float(strong["delta_rr_mean"]),
        "quant_delta_rr": float(quant["delta_rr_mean"]),
        "strong_threshold_share_pct": float(strong["future_threshold_exceed_share"]) * 100.0,
        "quant_threshold_share_pct": float(quant["future_threshold_exceed_share"]) * 100.0,
        "strong_ood_share_pct": float(strong["delta_ood_share"]) * 100.0,
        "quant_ood_share_pct": float(quant["delta_ood_share"]) * 100.0,
        "strong_yesil_pct": strong_traffic["yesil"],
        "strong_sari_pct": strong_traffic["sari"],
        "strong_kirmizi_pct": strong_traffic["kirmizi"],
        "quant_yesil_pct": quant_traffic["yesil"],
        "quant_sari_pct": quant_traffic["sari"],
        "quant_kirmizi_pct": quant_traffic["kirmizi"],
        "recommendation": one_liner,
    }
    pd.DataFrame([rec]).to_csv(out_path, index=False)


def build_visual(
    out_path: Path,
    date_label: str,
    strong: pd.Series,
    quant: pd.Series,
    strong_traffic: dict[str, float],
    quant_traffic: dict[str, float],
) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

    labels = ["Strong", "Quant"]
    rr_vals = [float(strong["future_rr_mean"]), float(quant["future_rr_mean"])]
    rr_colors = ["#e4572e", "#2d7ff9"]
    axes[0].bar(labels, rr_vals, color=rr_colors)
    axes[0].axhline(1.0, color="#666666", linestyle="--", linewidth=1)
    axes[0].set_title("Gelecek RR Karsilastirma")
    axes[0].set_ylabel("Future RR")
    axes[0].set_ylim(0.98, max(rr_vals) + 0.05)
    for i, v in enumerate(rr_vals):
        axes[0].text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=11)

    colors = {"yesil": "#2ca02c", "sari": "#f1c40f", "kirmizi": "#e74c3c"}
    ylabels = ["Strong", "Quant"]
    y_pos = [1, 0]
    yesil = [strong_traffic["yesil"], quant_traffic["yesil"]]
    sari = [strong_traffic["sari"], quant_traffic["sari"]]
    kirmizi = [strong_traffic["kirmizi"], quant_traffic["kirmizi"]]

    axes[1].barh(y_pos, yesil, color=colors["yesil"], label="yesil")
    axes[1].barh(y_pos, sari, left=yesil, color=colors["sari"], label="sari")
    left_k = [a + b for a, b in zip(yesil, sari)]
    axes[1].barh(y_pos, kirmizi, left=left_k, color=colors["kirmizi"], label="kirmizi")
    axes[1].set_yticks(y_pos, ylabels)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Senaryo orani (%)")
    axes[1].set_title("Literatur Uyum Trafik Ozeti")
    axes[1].legend(loc="lower right", frameon=False)

    fig.suptitle(f"Karar Ozeti | {date_label}", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Karar kurali: kapasite=strong, izleme tabani=quant.",
        ha="center",
        fontsize=11,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root_dir
    out_dir = args.output_dir or root
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_summary(root, args.model_summary_csv)
    traffic_path = resolve_traffic_csv(root, args.traffic_csv, args.date_label)
    disease_path = resolve_disease_csv(root, args.disease_csv)

    model_df = pd.read_csv(model_path)
    traffic_df = safe_read_csv(traffic_path)
    disease_df = pd.read_csv(disease_path)

    strong = pick_row(model_df, "strong")
    quant = pick_row(model_df, "quant")

    strong_traffic = {
        "yesil": traffic_pct(traffic_df, "strong", "yesil_uyumlu"),
        "sari": traffic_pct(traffic_df, "strong", "sari_sinirda"),
        "kirmizi": traffic_pct(traffic_df, "strong", "kirmizi_celiski_riski"),
    }
    quant_traffic = {
        "yesil": traffic_pct(traffic_df, "quant", "yesil_uyumlu"),
        "sari": traffic_pct(traffic_df, "quant", "sari_sinirda"),
        "kirmizi": traffic_pct(traffic_df, "quant", "kirmizi_celiski_riski"),
    }

    top3 = top_groups(disease_df, "strong", 3)

    exec_md = out_dir / f"yonetici_tek_cumle_ozet_{args.date_label}.md"
    tech_md = out_dir / f"teknik_ekip_aksiyon_ozeti_{args.date_label}.md"
    out_csv = out_dir / f"karar_mesaji_ozet_{args.date_label}.csv"
    out_png = out_dir / f"karar_mesaji_gorsel_{args.date_label}.png"

    one_liner = build_executive_md(exec_md, args.date_label, strong, quant, strong_traffic, quant_traffic)
    build_technical_md(tech_md, args.date_label, strong, quant, strong_traffic, quant_traffic, top3)
    build_csv(out_csv, args.date_label, strong, quant, strong_traffic, quant_traffic, one_liner)
    build_visual(out_png, args.date_label, strong, quant, strong_traffic, quant_traffic)

    print(f"Wrote: {exec_md}")
    print(f"Wrote: {tech_md}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
