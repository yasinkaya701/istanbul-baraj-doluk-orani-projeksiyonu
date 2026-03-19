#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_family_set(text: str) -> set[str]:
    if text is None:
        return set()
    parts = [x.strip() for x in str(text).split(",") if str(x).strip()]
    return set(parts)


def target_label(target_id: str) -> str:
    m = {
        "01_humidity_precip_hexbin": "Nem-Yagis",
        "02_temp_humidity_seasonal_box": "Sicaklik-Nem",
        "03_pressure_precip_era_spearman": "Basinc-Yagis(era)",
        "04_mgm_monthly_pattern": "Aylik Patern",
        "05_lag_correlation": "Lag Korelasyon",
    }
    return m.get(target_id, target_id)


def recommendation_label(consensus: float, weight_stability: float, aug_stability: float, family_ok: float) -> str:
    robust = min(weight_stability, aug_stability)
    if consensus >= 0.75 and robust >= 0.65 and family_ok >= 1.0:
        return "onerilir"
    if consensus >= 0.64 and robust >= 0.45:
        return "uygun"
    if consensus >= 0.54:
        return "izlenmeli"
    return "zayif"


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    out_dir = base / "output" / "analysis" / "internet_graph_similarity"

    scores = pd.read_csv(out_dir / "internet_graph_similarity_scores.csv")
    sensitivity_ref = pd.read_csv(out_dir / "internet_graph_similarity_weight_sensitivity_ref.csv")
    aug_ref = pd.read_csv(out_dir / "internet_graph_similarity_augmentation_robustness_ref.csv")
    family_rec = pd.read_csv(out_dir / "internet_graph_similarity_family_recommendation.csv")

    df = scores.merge(
        sensitivity_ref[["target_id", "ref_id", "win_freq"]],
        on=["target_id", "ref_id"],
        how="left",
    )
    df["win_freq"] = df["win_freq"].fillna(0.0).astype(float)
    df = df.merge(
        aug_ref[["target_id", "ref_id", "aug_win_freq"]],
        on=["target_id", "ref_id"],
        how="left",
    )
    df["aug_win_freq"] = df["aug_win_freq"].fillna(0.0).astype(float)

    fam_map = {str(r["target_id"]): parse_family_set(str(r["preferred_family"])) for _, r in family_rec.iterrows()}
    df["family_match"] = [
        1.0 if str(rt) in fam_map.get(str(t), set()) else 0.0
        for t, rt in zip(df["target_id"].astype(str), df["ref_type"].astype(str))
    ]

    # Consensus v2 score:
    # - 0.33 hybrid quality
    # - 0.25 weight-stability (weight sensitivity)
    # - 0.20 perturbation-stability
    # - 0.12 semantic compatibility
    # - 0.10 family alignment
    df["consensus_score"] = (
        0.33 * df["hybrid_score"].astype(float)
        + 0.25 * df["win_freq"].astype(float)
        + 0.20 * df["aug_win_freq"].astype(float)
        + 0.12 * df["semantic_compat"].astype(float)
        + 0.10 * df["family_match"].astype(float)
    )
    df["consensus_label"] = [
        recommendation_label(c, s, a, f)
        for c, s, a, f in zip(df["consensus_score"], df["win_freq"], df["aug_win_freq"], df["family_match"])
    ]

    # Full table
    full_path = out_dir / "internet_graph_similarity_consensus_v2_scores.csv"
    df.sort_values(["target_id", "consensus_score"], ascending=[True, False]).to_csv(full_path, index=False, float_format="%.6f")

    # Top-3 per target
    top_rows = []
    best_rows = []
    for target_id, g in df.groupby("target_id", sort=False):
        g = g.sort_values("consensus_score", ascending=False).reset_index(drop=True)
        for rank, (_, r) in enumerate(g.head(3).iterrows(), start=1):
            row = r.to_dict()
            row["rank"] = rank
            top_rows.append(row)
        first = g.iloc[0]
        second = g.iloc[1] if len(g) > 1 else g.iloc[0]
        best_rows.append(
            {
                "target_id": first["target_id"],
                "target_type": first["target_type"],
                "ref_id": first["ref_id"],
                "ref_type": first["ref_type"],
                "title": first["title"],
                "page_url": first["page_url"],
                "consensus_score": float(first["consensus_score"]),
                "consensus_label": first["consensus_label"],
                "hybrid_score": float(first["hybrid_score"]),
                "win_freq": float(first["win_freq"]),
                "aug_win_freq": float(first["aug_win_freq"]),
                "semantic_compat": float(first["semantic_compat"]),
                "family_match": float(first["family_match"]),
                "score_gap_to_second": float(first["consensus_score"] - second["consensus_score"]),
            }
        )

    top3 = pd.DataFrame(top_rows)
    top3_path = out_dir / "internet_graph_similarity_consensus_v2_top3.csv"
    top3.to_csv(top3_path, index=False, float_format="%.6f")

    best = pd.DataFrame(best_rows).sort_values("target_id").reset_index(drop=True)
    best_path = out_dir / "internet_graph_similarity_consensus_v2_best.csv"
    best.to_csv(best_path, index=False, float_format="%.6f")

    # Visual summary
    x = np.arange(len(best))
    labels = [target_label(x) for x in best["target_id"].astype(str)]
    consensus = best["consensus_score"].to_numpy(dtype=float)
    weight_stability = best["win_freq"].to_numpy(dtype=float)
    aug_stability = best["aug_win_freq"].to_numpy(dtype=float)
    hybrid = best["hybrid_score"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2.1, 1.4]})

    ax = axes[0]
    bars = ax.bar(x, consensus, color=["#2563eb", "#16a34a", "#ea580c", "#0891b2", "#7c3aed"])
    ax.set_ylim(0, 1)
    ax.set_title("Konsensus Skoru (Final Oneri)")
    ax.set_ylabel("Consensus")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{consensus[i]:.3f}", ha="center", fontsize=9)
        ax.text(i, 0.03, str(best.iloc[i]["ref_id"]), ha="center", fontsize=8)

    ax2 = axes[1]
    w = 0.26
    ax2.bar(x - w, hybrid, width=w, color="#334155", label="Hybrid")
    ax2.bar(x, weight_stability, width=w, color="#0f766e", label="Weight Stability")
    ax2.bar(x + w, aug_stability, width=w, color="#0369a1", label="Perturbation Stability")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Bilesen Skor")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    viz_path = out_dir / "internet_graph_similarity_consensus_v2_summary.png"
    fig.savefig(viz_path, dpi=180)
    plt.close(fig)

    # Markdown report
    lines: list[str] = []
    lines.append("# Konsensus Oneri Raporu v2 (TR)")
    lines.append("")
    lines.append("- Bu rapor skor, agirlik-stabilitesi, perturbasyon-stabilitesi ve aile uyumunu tek puanda birlestirir.")
    lines.append("- Formula: `0.33*hybrid + 0.25*win_freq + 0.20*aug_win_freq + 0.12*semantic + 0.10*family_match`")
    lines.append("")
    lines.append("| Hedef | Onerilen Ref | Consensus | Etiket | Hybrid | Agirlik Stabilite | Perturbasyon Stabilite | Semantik | Aile Uyumu |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for _, r in best.iterrows():
        lines.append(
            f"| {r['target_id']} | {r['ref_id']} ({r['ref_type']}) | {float(r['consensus_score']):.3f} | {r['consensus_label']} | "
            f"{float(r['hybrid_score']):.3f} | {float(r['win_freq']):.3f} | {float(r['aug_win_freq']):.3f} | "
            f"{float(r['semantic_compat']):.2f} | {float(r['family_match']):.0f} |"
        )
    lines.append("")
    lines.append("## Cikti Dosyalari")
    lines.append("")
    lines.append(f"- `{full_path}`")
    lines.append(f"- `{top3_path}`")
    lines.append(f"- `{best_path}`")
    lines.append(f"- `{viz_path}`")
    md_path = out_dir / "internet_graph_similarity_consensus_v2_report_tr.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {full_path}")
    print(f"Saved: {top3_path}")
    print(f"Saved: {best_path}")
    print(f"Saved: {viz_path}")
    print(f"Saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
