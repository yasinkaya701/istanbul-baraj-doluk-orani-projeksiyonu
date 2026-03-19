#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

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


def policy_mode(label: str, gap: float, aug: float) -> str:
    if str(label) == "onerilir" and gap >= 0.22 and aug >= 0.65:
        return "single_primary"
    return "dual_primary_backup"


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    out_dir = base / "output" / "analysis" / "internet_graph_similarity"

    top3 = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v2_top3.csv")
    best = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v2_best.csv")

    rows = []
    json_rows = []
    for target_id, g in top3.groupby("target_id", sort=False):
        g = g.sort_values("rank").reset_index(drop=True)
        p = g.iloc[0]
        b = g.iloc[1] if len(g) > 1 else g.iloc[0]
        gap = float(p["consensus_score"] - b["consensus_score"])
        mode = policy_mode(str(p["consensus_label"]), gap, float(p["aug_win_freq"]))

        confidence = "yuksek" if mode == "single_primary" else ("orta" if gap >= 0.12 else "dusuk")

        rule_text = (
            "Primary kullan; skor farki ve robustluk yeterli."
            if mode == "single_primary"
            else "Primary zayiflarsa (stabilite/uyum) backup'a gec."
        )

        row = {
            "target_id": target_id,
            "target_label": target_label(str(target_id)),
            "decision_mode": mode,
            "decision_confidence": confidence,
            "primary_ref_id": p["ref_id"],
            "primary_ref_type": p["ref_type"],
            "primary_consensus": float(p["consensus_score"]),
            "primary_hybrid": float(p["hybrid_score"]),
            "primary_weight_stability": float(p["win_freq"]),
            "primary_perturbation_stability": float(p["aug_win_freq"]),
            "backup_ref_id": b["ref_id"],
            "backup_ref_type": b["ref_type"],
            "backup_consensus": float(b["consensus_score"]),
            "gap_primary_backup": gap,
            "rule_text": rule_text,
            "primary_page_url": p["page_url"],
            "backup_page_url": b["page_url"],
        }
        rows.append(row)

        json_rows.append(
            {
                "target_id": target_id,
                "target_label": target_label(str(target_id)),
                "mode": mode,
                "confidence": confidence,
                "primary": {
                    "ref_id": p["ref_id"],
                    "ref_type": p["ref_type"],
                    "consensus": float(p["consensus_score"]),
                    "hybrid": float(p["hybrid_score"]),
                    "weight_stability": float(p["win_freq"]),
                    "perturbation_stability": float(p["aug_win_freq"]),
                    "page_url": p["page_url"],
                },
                "backup": {
                    "ref_id": b["ref_id"],
                    "ref_type": b["ref_type"],
                    "consensus": float(b["consensus_score"]),
                    "page_url": b["page_url"],
                },
                "gap_primary_backup": gap,
                "rule_text": rule_text,
            }
        )

    dec = pd.DataFrame(rows).sort_values("target_id").reset_index(drop=True)
    csv_out = out_dir / "internet_graph_similarity_consensus_v2_decision_pack.csv"
    dec.to_csv(csv_out, index=False, float_format="%.6f")

    json_out = out_dir / "internet_graph_similarity_consensus_v2_decision_pack.json"
    json_out.write_text(json.dumps(json_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # visualization: primary vs backup + mode
    x = np.arange(len(dec))
    labels = dec["target_label"].astype(str).tolist()
    pvals = dec["primary_consensus"].to_numpy(dtype=float)
    bvals = dec["backup_consensus"].to_numpy(dtype=float)
    modes = dec["decision_mode"].astype(str).tolist()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2.1, 1.3]})
    ax = axes[0]
    w = 0.36
    ax.bar(x - w / 2, pvals, width=w, color="#2563eb", label="Primary consensus")
    ax.bar(x + w / 2, bvals, width=w, color="#94a3b8", label="Backup consensus")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Consensus")
    ax.set_title("Karar Paketi: Primary vs Backup")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    for i in range(len(dec)):
        gap = float(dec.iloc[i]["gap_primary_backup"])
        ax.text(i, max(pvals[i], bvals[i]) + 0.015, f"gap={gap:.3f}", ha="center", fontsize=8)

    ax2 = axes[1]
    conf_map = {"dusuk": 1, "orta": 2, "yuksek": 3}
    conf_vals = dec["decision_confidence"].map(conf_map).fillna(1).to_numpy(dtype=float)
    mode_vals = np.array([1.0 if m == "single_primary" else 0.0 for m in modes], dtype=float)
    ax2.bar(x - 0.18, conf_vals, width=0.36, color="#0f766e", label="Karar guveni")
    ax2.bar(x + 0.18, mode_vals, width=0.36, color="#7c3aed", label="Mod (single=1, dual=0)")
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(["dual", "single", "orta", "yuksek"])
    ax2.set_ylim(-0.1, 3.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    viz_out = out_dir / "internet_graph_similarity_consensus_v2_decision_pack.png"
    fig.savefig(viz_out, dpi=180)
    plt.close(fig)

    # markdown report
    lines = []
    lines.append("# Konsensus v2 Karar Paketi (TR)")
    lines.append("")
    lines.append("- Cikti, her hedef icin primary + backup onerisi ve kural metni uretir.")
    lines.append("- Modlar: `single_primary` veya `dual_primary_backup`")
    lines.append("")
    lines.append("| Hedef | Mod | Guven | Primary | Backup | Primary Skor | Backup Skor | Gap |")
    lines.append("|---|---|---|---|---|---:|---:|---:|")
    for _, r in dec.iterrows():
        lines.append(
            f"| {r['target_id']} | {r['decision_mode']} | {r['decision_confidence']} | "
            f"{r['primary_ref_id']} ({r['primary_ref_type']}) | {r['backup_ref_id']} ({r['backup_ref_type']}) | "
            f"{float(r['primary_consensus']):.3f} | {float(r['backup_consensus']):.3f} | {float(r['gap_primary_backup']):.3f} |"
        )
    lines.append("")
    lines.append("## Cikti Dosyalari")
    lines.append("")
    lines.append(f"- `{csv_out}`")
    lines.append(f"- `{json_out}`")
    lines.append(f"- `{viz_out}`")
    md_out = out_dir / "internet_graph_similarity_consensus_v2_decision_pack_report_tr.md"
    md_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {csv_out}")
    print(f"Saved: {json_out}")
    print(f"Saved: {viz_out}")
    print(f"Saved: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
