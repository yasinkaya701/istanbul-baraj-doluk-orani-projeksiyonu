#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def target_label(target_id: str) -> str:
    m = {
        "01_humidity_precip_hexbin": "Nem-Yagis",
        "02_temp_humidity_seasonal_box": "Sicaklik-Nem",
        "03_pressure_precip_era_spearman": "Basinc-Yagis(era)",
        "04_mgm_monthly_pattern": "Aylik Patern",
        "05_lag_correlation": "Lag Korelasyon",
    }
    return m.get(target_id, target_id)


def type_is_realistic(target_type: str, ref_type: str) -> bool:
    allow = {
        "hexbin": {"hexbin", "scatter"},
        "box_violin": {"box", "violin"},
        "era_bar": {"bar"},
        "monthly_line": {"line"},
        "lag_corr": {"lag_corr"},
    }
    return str(ref_type) in allow.get(str(target_type), {str(ref_type)})


def confidence_label(score: float, win: float, margin: float, agreement: bool, type_ok: bool) -> str:
    if agreement and type_ok and score >= 0.80 and win >= 0.75 and margin >= 0.65:
        return "cok_yuksek"
    if agreement and type_ok and score >= 0.72 and win >= 0.70 and margin >= 0.60:
        return "yuksek"
    if agreement and type_ok and score >= 0.66 and win >= 0.60 and margin >= 0.45:
        return "orta"
    return "dusuk"


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    out_dir = base / "output" / "analysis" / "internet_graph_similarity"

    v3 = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v3_best.csv")
    v4 = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v4_best.csv")
    v5 = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v5_best.csv")
    dec = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v5_decision_pack.csv")

    df = (
        v5.merge(v4[["target_id", "ref_id"]].rename(columns={"ref_id": "ref_v4"}), on="target_id", how="left")
        .merge(v3[["target_id", "ref_id"]].rename(columns={"ref_id": "ref_v3"}), on="target_id", how="left")
        .merge(dec[["target_id", "decision_mode", "decision_confidence"]], on="target_id", how="left")
        .copy()
    )

    df["model_agreement"] = (
        (df["ref_id"].astype(str) == df["ref_v4"].astype(str))
        & (df["ref_id"].astype(str) == df["ref_v3"].astype(str))
    )
    df["type_realistic"] = [
        type_is_realistic(t, r) for t, r in zip(df["target_type"].astype(str), df["ref_type"].astype(str))
    ]

    df["realism_score"] = (
        0.48 * df["consensus_v5_score"].astype(float)
        + 0.27 * df["robust_win_prob"].astype(float)
        + 0.15 * np.clip(df["robust_margin_prob"].astype(float), 0.0, 1.0)
        + 0.10 * (1.0 - df["robust_top1_entropy"].astype(float))
    )

    df["confidence_level"] = [
        confidence_label(float(s), float(w), float(m), bool(a), bool(t))
        for s, w, m, a, t in zip(
            df["consensus_v5_score"],
            df["robust_win_prob"],
            df["robust_margin_prob"],
            df["model_agreement"],
            df["type_realistic"],
        )
    ]

    df["realistic_correct"] = [
        bool(a and t and s >= 0.66 and w >= 0.60 and m >= 0.45)
        for a, t, s, w, m in zip(
            df["model_agreement"],
            df["type_realistic"],
            df["consensus_v5_score"],
            df["robust_win_prob"],
            df["robust_margin_prob"],
        )
    ]

    df["target_label"] = df["target_id"].map(target_label).fillna(df["target_id"])

    out_cols = [
        "target_id",
        "target_label",
        "target_type",
        "ref_id",
        "ref_type",
        "title",
        "page_url",
        "consensus_v5_score",
        "robust_win_prob",
        "robust_margin_prob",
        "robust_top1_entropy",
        "realism_score",
        "model_agreement",
        "type_realistic",
        "confidence_level",
        "realistic_correct",
        "decision_mode",
        "decision_confidence",
    ]
    out_df = df[out_cols].sort_values("target_id").reset_index(drop=True)

    csv_out = out_dir / "internet_graph_similarity_realistic_correct_correlations.csv"
    out_df.to_csv(csv_out, index=False, float_format="%.6f")

    # visualization
    x = np.arange(len(out_df))
    labels = out_df["target_label"].astype(str).tolist()
    score = out_df["realism_score"].to_numpy(dtype=float)
    corr = out_df["realistic_correct"].astype(bool).to_numpy()
    colors = ["#1d4ed8" if c else "#9ca3af" for c in corr]

    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    bars = ax.bar(x, score, color=colors, alpha=0.92)
    ax.axhline(0.66, color="#ef4444", linestyle="--", linewidth=1.2, label="minimum realism")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Realism Score")
    ax.set_title("Gercekci ve Dogru Korelasyon Filtre Sonucu")
    ax.grid(axis="y", alpha=0.25)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.014, f"{score[i]:.3f}", ha="center", fontsize=8)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    png_out = out_dir / "internet_graph_similarity_realistic_correct_correlations.png"
    fig.savefig(png_out, dpi=180)
    plt.close(fig)

    # report
    lines = []
    lines.append("# Gercekci ve Dogru Korelasyon Ozeti (TR)")
    lines.append("")
    lines.append("- Kriterler: model anlasmasi (v3=v4=v5), tip uyumu, robust olasilik ve marj esikleri.")
    lines.append("- `realistic_correct=true` icin minimum: score>=0.66, win>=0.60, margin>=0.45")
    lines.append("")
    lines.append("| Hedef | Secilen Ref | v5 Skor | Win | Marj | Realism | Guven | Gercekci/Dogru |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|")
    for _, r in out_df.iterrows():
        lines.append(
            f"| {r['target_id']} | {r['ref_id']} ({r['ref_type']}) | {float(r['consensus_v5_score']):.3f} | {float(r['robust_win_prob']):.3f} | {float(r['robust_margin_prob']):.3f} | {float(r['realism_score']):.3f} | {r['confidence_level']} | {bool(r['realistic_correct'])} |"
        )

    lines.append("")
    lines.append("## Cikti Dosyalari")
    lines.append("")
    lines.append(f"- `{csv_out}`")
    lines.append(f"- `{png_out}`")
    md_out = out_dir / "internet_graph_similarity_realistic_correct_correlations_report_tr.md"
    md_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {csv_out}")
    print(f"Saved: {png_out}")
    print(f"Saved: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
