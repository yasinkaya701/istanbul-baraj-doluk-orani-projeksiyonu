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


def pass_9lvl(score: float, win: float, margin: float, entropy: float, agreement: bool, type_ok: bool) -> bool:
    # 9/10 seviye: daha sert ve muhafazakar kriter
    return bool(
        agreement
        and type_ok
        and score >= 0.75
        and win >= 0.75
        and margin >= 0.70
        and entropy <= 0.30
    )


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    out_dir = base / "output" / "analysis" / "internet_graph_similarity"

    df = pd.read_csv(out_dir / "internet_graph_similarity_realistic_correct_correlations.csv")

    df["score"] = df["consensus_v5_score"].astype(float)
    df["win"] = df["robust_win_prob"].astype(float)
    df["margin"] = df["robust_margin_prob"].astype(float)
    df["entropy"] = df["robust_top1_entropy"].astype(float)
    df["agree"] = df["model_agreement"].astype(bool)
    df["type_ok"] = df["type_realistic"].astype(bool)

    df["pass_9lvl"] = [
        pass_9lvl(s, w, m, e, a, t)
        for s, w, m, e, a, t in zip(df["score"], df["win"], df["margin"], df["entropy"], df["agree"], df["type_ok"])
    ]

    # 9-luk skoru: 4 temel metrik + 2 doğrulama bonusu
    df["score_9lvl"] = (
        0.35 * df["score"]
        + 0.25 * df["win"]
        + 0.20 * np.clip(df["margin"], 0.0, 1.0)
        + 0.10 * (1.0 - np.clip(df["entropy"], 0.0, 1.0))
        + 0.05 * df["agree"].astype(float)
        + 0.05 * df["type_ok"].astype(float)
    )

    df["target_label"] = df["target_id"].map(target_label).fillna(df["target_id"])

    out = df[
        [
            "target_id",
            "target_label",
            "ref_id",
            "ref_type",
            "title",
            "page_url",
            "score",
            "win",
            "margin",
            "entropy",
            "score_9lvl",
            "pass_9lvl",
        ]
    ].sort_values("score_9lvl", ascending=False)

    csv_out = out_dir / "internet_graph_similarity_realistic_correct_correlations_9lvl.csv"
    out.to_csv(csv_out, index=False, float_format="%.6f")

    # visualization
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    x = np.arange(len(out))
    colors = ["#1d4ed8" if bool(v) else "#9ca3af" for v in out["pass_9lvl"].tolist()]
    ax.bar(x, out["score_9lvl"].to_numpy(dtype=float), color=colors)
    ax.axhline(0.75, color="#dc2626", linestyle="--", linewidth=1.2, label="9-luk taban")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(out["target_label"].tolist(), rotation=0)
    ax.set_ylabel("9-luk Skor")
    ax.set_title("Gercekci/Dogru Korelasyonlar - 9/10 Seviye")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    for i, v in enumerate(out["score_9lvl"].to_numpy(dtype=float)):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    png_out = out_dir / "internet_graph_similarity_realistic_correct_correlations_9lvl.png"
    fig.savefig(png_out, dpi=180)
    plt.close(fig)

    # report
    passed = out[out["pass_9lvl"] == True]
    lines = []
    lines.append("# Gercekci/Dogru Korelasyonlar - 9/10 Seviye (TR)")
    lines.append("")
    lines.append("- 9-luk esikler: score>=0.75, win>=0.75, margin>=0.70, entropy<=0.30, model_agreement=True, type_realistic=True")
    lines.append("")
    lines.append("| Hedef | Ref | score | win | margin | entropy | 9-luk skor | Gecti mi? |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for _, r in out.iterrows():
        lines.append(
            f"| {r['target_id']} | {r['ref_id']} ({r['ref_type']}) | {float(r['score']):.3f} | {float(r['win']):.3f} | {float(r['margin']):.3f} | {float(r['entropy']):.3f} | {float(r['score_9lvl']):.3f} | {bool(r['pass_9lvl'])} |"
        )

    lines.append("")
    lines.append(f"- 9-luktan gecen hedef sayisi: **{len(passed)}/{len(out)}**")
    lines.append("")
    lines.append("## Cikti")
    lines.append(f"- `{csv_out}`")
    lines.append(f"- `{png_out}`")

    md_out = out_dir / "internet_graph_similarity_realistic_correct_correlations_9lvl_report_tr.md"
    md_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {csv_out}")
    print(f"Saved: {png_out}")
    print(f"Saved: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
