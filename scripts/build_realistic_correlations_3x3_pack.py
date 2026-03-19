#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


def target_label(target_id: str) -> str:
    m = {
        "01_humidity_precip_hexbin": "Nem-Yagis",
        "02_temp_humidity_seasonal_box": "Sicaklik-Nem",
        "03_pressure_precip_era_spearman": "Basinc-Yagis(era)",
        "04_mgm_monthly_pattern": "Aylik Patern",
        "05_lag_correlation": "Lag Korelasyon",
    }
    return m.get(target_id, target_id)


def level3(v: float, lo: float, hi: float) -> str:
    if v < lo:
        return "Dusuk"
    if v < hi:
        return "Orta"
    return "Yuksek"


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    out_dir = base / "output" / "analysis" / "internet_graph_similarity"

    real = pd.read_csv(out_dir / "internet_graph_similarity_realistic_correct_correlations.csv")
    top3 = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v5_top3.csv")
    pair_dir = out_dir / "pairs_top3"

    # 3x3 matrix (realistic set)
    real["score_lvl"] = [level3(float(x), 0.66, 0.75) for x in real["consensus_v5_score"]]
    real["win_lvl"] = [level3(float(x), 0.60, 0.75) for x in real["robust_win_prob"]]

    ctab = pd.crosstab(real["score_lvl"], real["win_lvl"]).reindex(
        index=["Dusuk", "Orta", "Yuksek"], columns=["Dusuk", "Orta", "Yuksek"], fill_value=0
    )

    total_n = int(ctab.to_numpy().sum())
    ctab_pct = (ctab.astype(float) / max(total_n, 1)) * 100.0
    annot = np.empty(ctab.shape, dtype=object)
    for i in range(ctab.shape[0]):
        for j in range(ctab.shape[1]):
            n = int(ctab.iloc[i, j])
            p = float(ctab_pct.iloc[i, j])
            annot[i, j] = f"{n}\\n%{p:.1f}"

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    sns.heatmap(
        ctab_pct,
        annot=annot,
        fmt="",
        cmap="YlGnBu",
        cbar=True,
        cbar_kws={"label": "Hucre Yuzdesi (%)"},
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("3x3 Korelasyon Matrisi (Skor x Robust Win)\\nHucre: adet + yuzde")
    ax.set_xlabel("Robust Win Seviyesi")
    ax.set_ylabel("v5 Skor Seviyesi")
    fig.tight_layout()
    matrix_png = out_dir / "internet_graph_similarity_realistic_correct_3x3_matrix.png"
    fig.savefig(matrix_png, dpi=180)
    plt.close(fig)

    ctab_counts_csv = out_dir / "internet_graph_similarity_realistic_correct_3x3_matrix_counts.csv"
    ctab_pct_csv = out_dir / "internet_graph_similarity_realistic_correct_3x3_matrix_percent.csv"
    ctab.to_csv(ctab_counts_csv)
    ctab_pct.to_csv(ctab_pct_csv, float_format="%.4f")

    # 3x3 best visual sheet (top 9 rows from v5_top3)
    cand = top3.sort_values("consensus_v5_score", ascending=False).reset_index(drop=True)

    rows = []
    for _, r in cand.iterrows():
        t = str(r["target_id"])
        rank = int(r["rank"])
        ref = str(r["ref_id"])
        exact = pair_dir / f"{t}_r{rank}_{ref}.png"
        if exact.exists():
            img = exact
        else:
            alts = sorted(pair_dir.glob(f"{t}_r{rank}_{ref}*.png"))
            if not alts:
                continue
            img = alts[0]
        rows.append(
            {
                "target_id": t,
                "target_label": target_label(t),
                "rank": rank,
                "ref_id": ref,
                "score": float(r["consensus_v5_score"]),
                "fit_pct": float(r["consensus_v5_score"]) * 100.0,
                "win_pct": float(r.get("robust_win_prob", np.nan)) * 100.0,
                "image_path": str(img),
                "page_url": r["page_url"],
            }
        )
        if len(rows) >= 9:
            break

    sel = pd.DataFrame(rows)
    sel_csv = out_dir / "internet_graph_similarity_realistic_correct_3x3_selection.csv"
    sel.to_csv(sel_csv, index=False, float_format="%.6f")

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    axes_arr = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes_arr):
        if i >= len(sel):
            ax.axis("off")
            continue
        rr = sel.iloc[i]
        img = Image.open(rr["image_path"])
        ax.imshow(img)
        ax.set_title(
            f"{rr['target_label']} | r{int(rr['rank'])} | {rr['ref_id']}\nuyum=%{float(rr['fit_pct']):.1f} | win=%{float(rr['win_pct']):.1f}",
            fontsize=9,
        )
        ax.axis("off")
    fig.tight_layout()
    sheet_png = out_dir / "internet_graph_similarity_realistic_correct_3x3_sheet.png"
    fig.savefig(sheet_png, dpi=170)
    plt.close(fig)

    # report
    md = []
    md.append("# 3x3 Paket (TR)")
    md.append("")
    md.append("- 3x3 Korelasyon Matrisi: v5 skor seviyesi x robust win seviyesi")
    md.append("- Matris hucrelerinde `adet + hucre yuzdesi` verilir.")
    md.append("- 3x3 Gorsel Sheet: consensus_v5_top3 havuzundan en iyi 9 eslesme, her hucrede `uyum %` ve `win %` yazilir.")
    md.append("")
    md.append("## Cikti")
    md.append(f"- `{matrix_png}`")
    md.append(f"- `{ctab_counts_csv}`")
    md.append(f"- `{ctab_pct_csv}`")
    md.append(f"- `{sheet_png}`")
    md.append(f"- `{sel_csv}`")
    md_out = out_dir / "internet_graph_similarity_realistic_correct_3x3_report_tr.md"
    md_out.write_text("\n".join(md), encoding="utf-8")

    print(f"Saved: {matrix_png}")
    print(f"Saved: {sheet_png}")
    print(f"Saved: {sel_csv}")
    print(f"Saved: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
