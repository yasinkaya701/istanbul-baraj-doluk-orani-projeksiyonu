#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

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


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    out_dir = base / "output" / "analysis" / "internet_graph_similarity"

    scores = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v2_scores.csv")
    top3 = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v2_top3.csv")

    df = top3.merge(
        scores[
            [
                "target_id",
                "ref_id",
                "consensus_score",
                "hybrid_score",
                "win_freq",
                "aug_win_freq",
                "semantic_compat",
                "family_match",
                "consensus_label",
                "ref_type",
                "title",
                "page_url",
            ]
        ],
        on=["target_id", "ref_id", "consensus_score", "hybrid_score", "win_freq", "aug_win_freq", "semantic_compat", "family_match", "consensus_label", "ref_type", "title", "page_url"],
        how="left",
    )

    df["c_hybrid"] = 0.33 * df["hybrid_score"].astype(float)
    df["c_weight_stab"] = 0.25 * df["win_freq"].astype(float)
    df["c_aug_stab"] = 0.20 * df["aug_win_freq"].astype(float)
    df["c_semantic"] = 0.12 * df["semantic_compat"].astype(float)
    df["c_family"] = 0.10 * df["family_match"].astype(float)

    explain_csv = out_dir / "internet_graph_similarity_consensus_v2_explainability.csv"
    df.to_csv(explain_csv, index=False, float_format="%.6f")

    # Per-target stacked bars for top3
    pair_dir = out_dir / "consensus_v2_explainability"
    pair_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    comp_cols = ["c_hybrid", "c_weight_stab", "c_aug_stab", "c_semantic", "c_family"]
    comp_names = ["Hybrid", "Agirlik", "Perturbasyon", "Semantik", "Aile"]
    colors = ["#334155", "#0f766e", "#0369a1", "#f59e0b", "#7c3aed"]

    for target_id, g in df.groupby("target_id", sort=False):
        g = g.sort_values("rank").head(3).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        x = np.arange(len(g))
        bottom = np.zeros(len(g))
        for c, n, col in zip(comp_cols, comp_names, colors):
            vals = g[c].to_numpy(dtype=float)
            ax.bar(x, vals, bottom=bottom, label=n, color=col)
            bottom += vals
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"r{int(r)}:{rid}" for r, rid in zip(g["rank"], g["ref_id"])], rotation=10, ha="right")
        ax.set_ylabel("Consensus Katkı")
        ax.set_title(f"{target_label(str(target_id))} - Top3 Konsensus Katki Dagilimi")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, ncol=3)
        for i, sc in enumerate(g["consensus_score"].to_numpy(dtype=float)):
            ax.text(i, sc + 0.015, f"{sc:.3f}", ha="center", fontsize=8)
        fig.tight_layout()
        out_png = pair_dir / f"{target_id}_top3_consensus_components.png"
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        image_paths.append(out_png)

    # Summary sheet
    if image_paths:
        cols = 2
        rows = int(np.ceil(len(image_paths) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4.4))
        axes_arr = np.array(axes).reshape(-1)
        for i, ax in enumerate(axes_arr):
            if i >= len(image_paths):
                ax.axis("off")
                continue
            img = plt.imread(str(image_paths[i]))
            ax.imshow(img)
            ax.set_title(image_paths[i].stem, fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        sheet = pair_dir / "consensus_v2_explainability_sheet.png"
        fig.savefig(sheet, dpi=170)
        plt.close(fig)
    else:
        sheet = pair_dir / "consensus_v2_explainability_sheet.png"

    # Top1 component heatmap
    top1 = df[df["rank"] == 1].copy().sort_values("target_id").reset_index(drop=True)
    heat_arr = top1[comp_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    im = ax.imshow(heat_arr, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=0.35)
    ax.set_title("Top1 Konsensus Bilesen Katkilari")
    ax.set_xticks(np.arange(len(comp_names)))
    ax.set_xticklabels(comp_names)
    ax.set_yticks(np.arange(len(top1)))
    ax.set_yticklabels([target_label(x) for x in top1["target_id"].astype(str)])
    for i in range(heat_arr.shape[0]):
        for j in range(heat_arr.shape[1]):
            ax.text(j, i, f"{heat_arr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    heat_png = out_dir / "internet_graph_similarity_consensus_v2_component_heatmap.png"
    fig.savefig(heat_png, dpi=180)
    plt.close(fig)

    # Markdown report
    lines: list[str] = []
    lines.append("# Konsensus v2 Aciklanabilirlik Raporu (TR)")
    lines.append("")
    lines.append("- Bu rapor Top-3 adaylarin konsensus skorunu bilesenlerine ayirir.")
    lines.append("- Bilesenler: Hybrid, Agirlik stabilitesi, Perturbasyon stabilitesi, Semantik uyum, Aile uyumu.")
    lines.append("")
    lines.append("| Hedef | Rank | Ref | Consensus | Hybrid Katki | Agirlik Katki | Perturbasyon Katki | Semantik Katki | Aile Katki |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for _, r in df.sort_values(["target_id", "rank"]).iterrows():
        lines.append(
            f"| {r['target_id']} | {int(r['rank'])} | {r['ref_id']} ({r['ref_type']}) | {float(r['consensus_score']):.3f} | "
            f"{float(r['c_hybrid']):.3f} | {float(r['c_weight_stab']):.3f} | {float(r['c_aug_stab']):.3f} | "
            f"{float(r['c_semantic']):.3f} | {float(r['c_family']):.3f} |"
        )
    lines.append("")
    lines.append("## Cikti Dosyalari")
    lines.append("")
    lines.append(f"- `{explain_csv}`")
    lines.append(f"- `{pair_dir}`")
    lines.append(f"- `{sheet}`")
    lines.append(f"- `{heat_png}`")
    md_out = out_dir / "internet_graph_similarity_consensus_v2_explainability_report_tr.md"
    md_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {explain_csv}")
    print(f"Saved: {sheet}")
    print(f"Saved: {heat_png}")
    print(f"Saved: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
