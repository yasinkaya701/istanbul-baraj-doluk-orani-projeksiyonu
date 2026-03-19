#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def bin3(values: pd.Series, q1: float, q2: float) -> pd.Series:
    v = values.astype(float)
    out = np.where(v < q1, "Dusuk", np.where(v < q2, "Orta", "Yuksek"))
    return pd.Series(out, index=values.index)


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

    v5 = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v5_scores.csv")
    best = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v5_best.csv")

    df = v5.copy()
    df["uyum_pct"] = df["consensus_v5_score"].astype(float) * 100.0
    df["win_pct"] = df["robust_win_prob"].astype(float) * 100.0

    q_u1, q_u2 = df["uyum_pct"].quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
    q_w1, q_w2 = df["win_pct"].quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()

    df["uyum_seviye"] = bin3(df["uyum_pct"], q_u1, q_u2)
    df["win_seviye"] = bin3(df["win_pct"], q_w1, q_w2)

    levels = ["Dusuk", "Orta", "Yuksek"]
    ctab = pd.crosstab(df["uyum_seviye"], df["win_seviye"]).reindex(index=levels, columns=levels, fill_value=0)

    total_n = int(ctab.to_numpy().sum())
    global_pct = (ctab.astype(float) / max(total_n, 1)) * 100.0
    row_pct = ctab.astype(float).div(ctab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0

    annot = np.empty(ctab.shape, dtype=object)
    for i in range(ctab.shape[0]):
        for j in range(ctab.shape[1]):
            n = int(ctab.iloc[i, j])
            g = float(global_pct.iloc[i, j])
            r = float(row_pct.iloc[i, j])
            annot[i, j] = f"n={n}\nGenel %{g:.1f}\nSatir %{r:.1f}"

    fig, ax = plt.subplots(figsize=(8.2, 6.7))
    sns.heatmap(
        global_pct,
        annot=annot,
        fmt="",
        cmap="YlGnBu",
        linewidths=0.6,
        cbar_kws={"label": "Hucre Genel Yuzde (%)"},
        ax=ax,
        vmin=0,
        vmax=max(5.0, float(global_pct.to_numpy().max())),
    )
    ax.set_title(
        "Sifirdan 3x3 Uyum Matrisi\n"
        f"Uyum tercil: <{q_u1:.2f}, <{q_u2:.2f}, >= {q_u2:.2f} | "
        f"Win tercil: <{q_w1:.2f}, <{q_w2:.2f}, >= {q_w2:.2f}",
        fontsize=11,
    )
    ax.set_xlabel("Kazanma Olasiligi Seviyesi")
    ax.set_ylabel("Uyum Skoru Seviyesi")
    fig.tight_layout()

    matrix_png = out_dir / "internet_graph_similarity_3x3_from_scratch_percent_matrix.png"
    fig.savefig(matrix_png, dpi=180)
    plt.close(fig)

    counts_csv = out_dir / "internet_graph_similarity_3x3_from_scratch_counts.csv"
    global_csv = out_dir / "internet_graph_similarity_3x3_from_scratch_global_percent.csv"
    row_csv = out_dir / "internet_graph_similarity_3x3_from_scratch_row_percent.csv"
    ctab.to_csv(counts_csv)
    global_pct.to_csv(global_csv, float_format="%.6f")
    row_pct.to_csv(row_csv, float_format="%.6f")

    top = best.copy()
    top["target_label"] = top["target_id"].map(target_label)
    top["uyum_pct"] = top["consensus_v5_score"].astype(float) * 100.0
    top["win_pct"] = top["robust_win_prob"].astype(float) * 100.0
    top["uyum_seviye"] = bin3(top["uyum_pct"], q_u1, q_u2)
    top["win_seviye"] = bin3(top["win_pct"], q_w1, q_w2)
    top_out = top[
        [
            "target_id",
            "target_label",
            "ref_id",
            "ref_type",
            "title",
            "page_url",
            "uyum_pct",
            "win_pct",
            "uyum_seviye",
            "win_seviye",
        ]
    ].sort_values("target_id")
    top_csv = out_dir / "internet_graph_similarity_3x3_from_scratch_top1_percent.csv"
    top_out.to_csv(top_csv, index=False, float_format="%.6f")

    md_lines: list[str] = []
    md_lines.append("# Sifirdan 3x3 Yuzde Analizi (TR)")
    md_lines.append("")
    md_lines.append("- Tum adaylar (n=295) uzerinden hesaplandi.")
    md_lines.append("- Hucre metni: adet + genel yuzde + satir ici yuzde.")
    md_lines.append("")
    md_lines.append("## Tercil Esikleri")
    md_lines.append("")
    md_lines.append(f"- Uyum (consensus_v5_score*100): q33={q_u1:.3f}, q66={q_u2:.3f}")
    md_lines.append(f"- Win (robust_win_prob*100): q33={q_w1:.3f}, q66={q_w2:.3f}")
    md_lines.append("")
    md_lines.append("## Top1 Uyum Yuzdeleri")
    md_lines.append("")
    md_lines.append("| Hedef | Ref | Uyum % | Win % | Uyum Seviye | Win Seviye |")
    md_lines.append("|---|---|---:|---:|---|---|")
    for _, r in top_out.iterrows():
        md_lines.append(
            f"| {r['target_id']} | {r['ref_id']} ({r['ref_type']}) | {float(r['uyum_pct']):.2f} | {float(r['win_pct']):.2f} | {r['uyum_seviye']} | {r['win_seviye']} |"
        )

    md_lines.append("")
    md_lines.append("## Cikti")
    md_lines.append(f"- `{matrix_png}`")
    md_lines.append(f"- `{counts_csv}`")
    md_lines.append(f"- `{global_csv}`")
    md_lines.append(f"- `{row_csv}`")
    md_lines.append(f"- `{top_csv}`")

    md_out = out_dir / "internet_graph_similarity_3x3_from_scratch_report_tr.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved: {matrix_png}")
    print(f"Saved: {counts_csv}")
    print(f"Saved: {global_csv}")
    print(f"Saved: {row_csv}")
    print(f"Saved: {top_csv}")
    print(f"Saved: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
