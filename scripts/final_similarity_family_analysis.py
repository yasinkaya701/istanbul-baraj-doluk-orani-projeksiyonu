#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def preferred_ref_types(target_type: str) -> set[str]:
    mapping = {
        "hexbin": {"hexbin", "scatter"},
        "box_violin": {"box", "violin"},
        "era_bar": {"bar"},
        "monthly_line": {"line"},
        "lag_corr": {"lag_corr"},
    }
    return mapping.get(target_type, {"hexbin", "scatter", "box", "violin", "bar", "line", "lag_corr"})


def family_confidence_label(gap_outside: float, top3_mean: float) -> str:
    if gap_outside >= 0.10 and top3_mean >= 0.60:
        return "yuksek"
    if gap_outside >= 0.05 and top3_mean >= 0.54:
        return "orta_yuksek"
    if gap_outside >= 0.02:
        return "orta"
    return "dusuk"


def target_label(target_id: str) -> str:
    m = {
        "01_humidity_precip_hexbin": "Nem-Yagis",
        "02_temp_humidity_seasonal_box": "Sicaklik-Nem",
        "03_pressure_precip_era_spearman": "Basinc-Yagis(era)",
        "04_mgm_monthly_pattern": "Aylik Patern",
        "05_lag_correlation": "Lag Korelasyon",
    }
    return m.get(target_id, target_id)


def make_gap_plot(df: pd.DataFrame, out_png: Path) -> None:
    x = np.arange(len(df))
    family_best = df["family_best_score"].to_numpy(dtype=float)
    outside_best = df["outside_best_score"].to_numpy(dtype=float)
    labels = [target_label(x) for x in df["target_id"].astype(str)]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    w = 0.35
    ax.bar(x - w / 2, family_best, width=w, color="#0ea5e9", label="Onerilen aile en iyi skor")
    ax.bar(x + w / 2, outside_best, width=w, color="#94a3b8", label="Aile disi en iyi skor")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hibrit Skor")
    ax.set_title("Aile Bazli Karsilastirma (Final)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    for i in range(len(df)):
        g = float(df.iloc[i]["family_gap_outside"])
        ax.text(i, min(0.98, family_best[i] + 0.02), f"gap={g:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def make_heatmap(score_matrix: pd.DataFrame, out_png: Path) -> None:
    fam_cols = list(score_matrix.columns)
    arr = score_matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    im = ax.imshow(arr, aspect="auto", cmap="YlGnBu", vmin=0.3, vmax=0.9)
    ax.set_title("Hedefe Gore Grafik Ailesi Skor Matrisi (max)")
    ax.set_xlabel("Grafik Ailesi")
    ax.set_ylabel("Hedef")
    ax.set_xticks(np.arange(len(fam_cols)))
    ax.set_xticklabels(fam_cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(score_matrix.index)))
    ax.set_yticklabels([target_label(x) for x in score_matrix.index.astype(str)])

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    out_dir = base / "output" / "analysis" / "internet_graph_similarity"
    scores_path = out_dir / "internet_graph_similarity_scores.csv"
    scores = pd.read_csv(scores_path)

    rec_rows = []
    for target_id, g in scores.groupby("target_id", sort=False):
        g = g.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        target_type = str(g.iloc[0]["target_type"])
        pref = preferred_ref_types(target_type)

        g_family = g[g["ref_type"].astype(str).isin(pref)].reset_index(drop=True)
        g_outside = g[~g["ref_type"].astype(str).isin(pref)].reset_index(drop=True)
        if len(g_family) == 0:
            g_family = g.copy()
        if len(g_outside) == 0:
            g_outside = g.copy()

        best_fam = g_family.iloc[0]
        top3_mean = float(g_family.head(3)["hybrid_score"].mean())
        outside_best = float(g_outside.iloc[0]["hybrid_score"])
        gap_outside = float(best_fam["hybrid_score"] - outside_best)
        conf = family_confidence_label(gap_outside, top3_mean)

        rec_rows.append(
            {
                "target_id": target_id,
                "target_type": target_type,
                "preferred_family": ",".join(sorted(pref)),
                "recommended_ref_id": best_fam["ref_id"],
                "recommended_ref_type": best_fam["ref_type"],
                "recommended_title": best_fam["title"],
                "recommended_page_url": best_fam["page_url"],
                "family_best_score": float(best_fam["hybrid_score"]),
                "family_top3_mean_score": top3_mean,
                "outside_best_score": outside_best,
                "family_gap_outside": gap_outside,
                "family_confidence": conf,
            }
        )

    rec = pd.DataFrame(rec_rows).sort_values("target_id").reset_index(drop=True)
    rec_path = out_dir / "internet_graph_similarity_family_recommendation.csv"
    rec.to_csv(rec_path, index=False, float_format="%.6f")

    # Family score matrix (max score per family for each target)
    families = ["hexbin", "scatter", "box", "violin", "bar", "line", "lag_corr"]
    piv = (
        scores.pivot_table(index="target_id", columns="ref_type", values="hybrid_score", aggfunc="max")
        .reindex(columns=families)
        .fillna(0.0)
        .sort_index()
    )
    matrix_path = out_dir / "internet_graph_similarity_family_score_matrix.csv"
    piv.to_csv(matrix_path, float_format="%.6f")

    # Visual outputs
    gap_png = out_dir / "final_family_gap_ozet.png"
    heat_png = out_dir / "final_family_score_heatmap.png"
    make_gap_plot(rec, gap_png)
    make_heatmap(piv, heat_png)

    # Markdown summary
    lines: list[str] = []
    lines.append("# Final Aile Bazli Oneri Raporu (TR)")
    lines.append("")
    lines.append("- Bu rapor internet benzerlik skorlarini aile bazinda toparlar.")
    lines.append("- Amaç: `hexbin/scatter` gibi yakin tipleri tek ailede degerlendirip daha mantikli karar vermek.")
    lines.append("")
    lines.append("| Hedef | Onerilen Aile | En Iyi Ref | Aile Skoru | Aile Disi En Iyi | Gap | Guven |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for _, r in rec.iterrows():
        lines.append(
            f"| {r['target_id']} | {r['preferred_family']} | {r['recommended_ref_id']} ({r['recommended_ref_type']}) | "
            f"{float(r['family_best_score']):.3f} | {float(r['outside_best_score']):.3f} | "
            f"{float(r['family_gap_outside']):.3f} | {r['family_confidence']} |"
        )
    lines.append("")
    lines.append("## Cikti Dosyalari")
    lines.append("")
    lines.append(f"- `{rec_path}`")
    lines.append(f"- `{matrix_path}`")
    lines.append(f"- `{gap_png}`")
    lines.append(f"- `{heat_png}`")
    lines.append("")
    lines.append("## Kaynaklar")
    lines.append("")
    for u in sorted(rec["recommended_page_url"].astype(str).unique()):
        lines.append(f"- {u}")
    md_path = out_dir / "final_family_recommendation_report_tr.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {rec_path}")
    print(f"Saved: {matrix_path}")
    print(f"Saved: {gap_png}")
    print(f"Saved: {heat_png}")
    print(f"Saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
