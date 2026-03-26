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


def recommendation_label(score: float, win_prob: float, ci: float, margin: float, entropy: float) -> str:
    if score >= 0.78 and win_prob >= 0.68 and ci <= 0.11 and margin >= 0.16 and entropy <= 0.30:
        return "onerilir"
    if score >= 0.67 and win_prob >= 0.45 and ci <= 0.16 and margin >= 0.06:
        return "uygun"
    if score >= 0.56:
        return "izlenmeli"
    return "zayif"


def stability_label(win_prob: float, ci: float, entropy: float) -> str:
    if win_prob >= 0.78 and ci <= 0.10 and entropy <= 0.25:
        return "cok_stabil"
    if win_prob >= 0.60 and ci <= 0.14:
        return "stabil"
    if win_prob >= 0.42:
        return "orta"
    return "oynak"


def policy_mode(label: str, gap: float, win_prob: float, backup_win_prob: float, ci: float, entropy: float) -> str:
    if (
        str(label) == "onerilir"
        and gap >= 0.12
        and win_prob >= 0.70
        and backup_win_prob <= 0.28
        and ci <= 0.13
        and entropy <= 0.30
    ):
        return "single_primary"
    return "dual_primary_backup"


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    out_dir = base / "output" / "analysis" / "internet_graph_similarity"

    v4 = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v4_scores.csv")

    need_cols = [
        "target_id",
        "target_type",
        "ref_id",
        "ref_type",
        "title",
        "page_url",
        "consensus_v4_score",
        "consensus_v4_base",
        "p_top1",
        "margin_prob",
        "mc_ci_width",
        "top1_entropy",
    ]
    missing = [c for c in need_cols if c not in v4.columns]
    if missing:
        raise RuntimeError(f"Missing columns in v4 scores: {missing}")

    df = v4[need_cols].copy()
    df["margin_prob"] = np.clip(df["margin_prob"].astype(float), 0.0, 1.0)
    df["p_top1"] = np.clip(df["p_top1"].astype(float), 0.0, 1.0)
    df["mc_ci_width"] = np.clip(df["mc_ci_width"].astype(float), 0.0, 1.0)
    df["top1_entropy"] = np.clip(df["top1_entropy"].astype(float), 0.0, 1.0)

    # Robust stability simulation:
    # - random coefficient drift (Dirichlet around base weights)
    # - candidate dropout stress
    # - additive noise scaled by target entropy
    rng = np.random.default_rng(42)
    draws = 5000
    base_w = np.array([0.52, 0.30, 0.10, 0.08], dtype=float)
    alpha = np.array([6.0, 4.0, 2.5, 1.5], dtype=float)

    sim_rows = []
    sim_summary_rows = []

    for target_id, g in df.groupby("target_id", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)

        c1 = g["consensus_v4_score"].to_numpy(dtype=float)
        c2 = g["p_top1"].to_numpy(dtype=float)
        c3 = 1.0 - g["mc_ci_width"].to_numpy(dtype=float)
        c4 = g["margin_prob"].to_numpy(dtype=float)
        ent = float(g["top1_entropy"].iloc[0])
        sigma = 0.015 + 0.065 * ent

        win_counts = np.zeros(n, dtype=float)
        second_counts = np.zeros(n, dtype=float)

        for _ in range(draws):
            w = rng.dirichlet(alpha)
            # Keep around 78% candidates each draw to stress reference-set changes harder.
            keep = rng.random(n) < 0.78
            if int(np.sum(keep)) < 2:
                keep[np.argsort(c1)[-2:]] = True

            noise = rng.normal(0.0, sigma, size=n)
            s = (w[0] * c1 + w[1] * c2 + w[2] * c3 + w[3] * c4) + noise
            s[~keep] = -1e9

            order = np.argsort(s)[::-1]
            win_counts[order[0]] += 1.0
            second_counts[order[1]] += 1.0

        win_prob = win_counts / float(draws)
        second_prob = second_counts / float(draws)
        p_norm = win_prob / np.maximum(win_prob.sum(), 1e-12)
        p_pos = p_norm[p_norm > 0]
        ent_draw = -float(np.sum(p_pos * np.log(p_pos))) / float(np.log(max(n, 2)))

        for i in range(n):
            max_other = float(np.max(np.delete(win_prob, i))) if n > 1 else float(win_prob[i])
            sim_rows.append(
                {
                    "target_id": target_id,
                    "ref_id": g.iloc[i]["ref_id"],
                    "robust_win_prob": float(win_prob[i]),
                    "robust_second_prob": float(second_prob[i]),
                    "robust_margin_prob": float(win_prob[i] - max_other),
                    "sim_sigma": sigma,
                    "draws": draws,
                }
            )

        sim_summary_rows.append(
            {
                "target_id": target_id,
                "robust_top1_entropy": ent_draw,
                "sim_sigma": sigma,
                "draws": draws,
            }
        )

    sim_df = pd.DataFrame(sim_rows)
    sim_sum = pd.DataFrame(sim_summary_rows)

    out = df.merge(sim_df, on=["target_id", "ref_id"], how="left").merge(sim_sum, on="target_id", how="left")

    # v5 score: increase stability effect; keep quality anchor from v4.
    out["consensus_v5_score"] = (
        0.53 * out["consensus_v4_score"].astype(float)
        + 0.24 * out["robust_win_prob"].astype(float)
        + 0.10 * (1.0 - out["mc_ci_width"].astype(float))
        + 0.05 * np.clip(out["robust_margin_prob"].astype(float), 0.0, 1.0)
        + 0.08 * (1.0 - out["robust_top1_entropy"].astype(float))
    )

    out["consensus_v5_label"] = [
        recommendation_label(float(s), float(w), float(ci), float(m), float(e))
        for s, w, ci, m, e in zip(
            out["consensus_v5_score"],
            out["robust_win_prob"],
            out["mc_ci_width"],
            out["robust_margin_prob"],
            out["robust_top1_entropy"],
        )
    ]
    out["stability_class"] = [
        stability_label(float(w), float(ci), float(e))
        for w, ci, e in zip(out["robust_win_prob"], out["mc_ci_width"], out["robust_top1_entropy"])
    ]

    full_path = out_dir / "internet_graph_similarity_consensus_v5_scores.csv"
    out.sort_values(["target_id", "consensus_v5_score"], ascending=[True, False]).to_csv(
        full_path, index=False, float_format="%.6f"
    )

    sim_path = out_dir / "internet_graph_similarity_consensus_v5_robust_simulation.csv"
    sim_df.to_csv(sim_path, index=False, float_format="%.6f")

    top_rows = []
    best_rows = []
    dec_rows = []

    for target_id, g in out.groupby("target_id", sort=False):
        g = g.sort_values("consensus_v5_score", ascending=False).reset_index(drop=True)

        for rank, (_, r) in enumerate(g.head(3).iterrows(), start=1):
            row = r.to_dict()
            row["rank"] = rank
            top_rows.append(row)

        p = g.iloc[0]
        b = g.iloc[1] if len(g) > 1 else g.iloc[0]
        gap = float(p["consensus_v5_score"] - b["consensus_v5_score"])
        mode = policy_mode(
            str(p["consensus_v5_label"]),
            gap,
            float(p["robust_win_prob"]),
            float(b["robust_win_prob"]),
            float(p["mc_ci_width"]),
            float(p["robust_top1_entropy"]),
        )
        conf = "yuksek" if mode == "single_primary" else ("orta" if gap >= 0.10 else "dusuk")

        best_rows.append(
            {
                "target_id": p["target_id"],
                "target_type": p["target_type"],
                "ref_id": p["ref_id"],
                "ref_type": p["ref_type"],
                "title": p["title"],
                "page_url": p["page_url"],
                "consensus_v5_score": float(p["consensus_v5_score"]),
                "consensus_v5_label": p["consensus_v5_label"],
                "stability_class": p["stability_class"],
                "robust_win_prob": float(p["robust_win_prob"]),
                "robust_margin_prob": float(p["robust_margin_prob"]),
                "mc_ci_width": float(p["mc_ci_width"]),
                "robust_top1_entropy": float(p["robust_top1_entropy"]),
                "score_gap_to_second": gap,
            }
        )

        dec_rows.append(
            {
                "target_id": target_id,
                "target_label": target_label(str(target_id)),
                "decision_mode": mode,
                "decision_confidence": conf,
                "primary_ref_id": p["ref_id"],
                "primary_ref_type": p["ref_type"],
                "primary_score": float(p["consensus_v5_score"]),
                "primary_win_prob": float(p["robust_win_prob"]),
                "primary_margin_prob": float(p["robust_margin_prob"]),
                "primary_stability": p["stability_class"],
                "backup_ref_id": b["ref_id"],
                "backup_ref_type": b["ref_type"],
                "backup_score": float(b["consensus_v5_score"]),
                "backup_win_prob": float(b["robust_win_prob"]),
                "gap_primary_backup": gap,
                "rule_text": (
                    "Primary kullan; stabilite ve marj yeterli."
                    if mode == "single_primary"
                    else "Primary + backup birlikte izle; olasilik marji sinirda."
                ),
                "primary_page_url": p["page_url"],
                "backup_page_url": b["page_url"],
            }
        )

    top3 = pd.DataFrame(top_rows)
    best = pd.DataFrame(best_rows).sort_values("target_id").reset_index(drop=True)
    dec = pd.DataFrame(dec_rows).sort_values("target_id").reset_index(drop=True)

    top3_path = out_dir / "internet_graph_similarity_consensus_v5_top3.csv"
    best_path = out_dir / "internet_graph_similarity_consensus_v5_best.csv"
    dec_path = out_dir / "internet_graph_similarity_consensus_v5_decision_pack.csv"

    top3.to_csv(top3_path, index=False, float_format="%.6f")
    best.to_csv(best_path, index=False, float_format="%.6f")
    dec.to_csv(dec_path, index=False, float_format="%.6f")

    # Visual 1: v5 summary
    x = np.arange(len(best))
    labels = [target_label(str(t)) for t in best["target_id"].astype(str)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2.0, 1.3]})
    ax = axes[0]
    ax.bar(x, best["consensus_v5_score"], color="#1d4ed8", alpha=0.92, label="v5 skor")
    ax.plot(x, best["robust_win_prob"], color="#dc2626", marker="o", linewidth=2, label="robust kazanma olasiligi")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Konsensus v5: Sertlestirilmis Stabil Tahmin")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    ax2 = axes[1]
    w = 0.36
    ax2.bar(x - w / 2, best["robust_margin_prob"], width=w, color="#0f766e", label="robust marj")
    ax2.bar(x + w / 2, best["mc_ci_width"], width=w, color="#475569", label="belirsizlik")
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    viz1 = out_dir / "internet_graph_similarity_consensus_v5_summary.png"
    fig.savefig(viz1, dpi=180)
    plt.close(fig)

    # Visual 2: v4 vs v5 compare
    v4_best = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v4_best.csv")
    cmp = v4_best[["target_id", "consensus_v4_score"]].merge(
        best[["target_id", "consensus_v5_score", "robust_win_prob"]], on="target_id", how="outer"
    ).sort_values("target_id")

    x = np.arange(len(cmp))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ww = 0.33
    ax.bar(x - ww, cmp["consensus_v4_score"], width=ww, color="#94a3b8", label="v4")
    ax.bar(x, cmp["consensus_v5_score"], width=ww, color="#1d4ed8", label="v5")
    ax.bar(x + ww, cmp["robust_win_prob"], width=ww, color="#dc2626", label="v5 robust_win")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([target_label(str(t)) for t in cmp["target_id"].astype(str)])
    ax.set_title("v4-v5 Skor ve Robust Kazanma Karsilastirmasi")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    viz2 = out_dir / "internet_graph_similarity_consensus_v4_v5_compare.png"
    fig.savefig(viz2, dpi=180)
    plt.close(fig)

    # report
    lines = []
    lines.append("# Konsensus v5 Sertlestirilmis Stabil Tahmin Raporu (TR)")
    lines.append("")
    lines.append("- Yontem: agirlik belirsizligi + aday dusurme stresi + entropi-olcekli gurultu simulasyonu.")
    lines.append("- Amaç: tek modele/asiri guvene bagli secimleri azaltmak.")
    lines.append("")
    lines.append("## Top-1 Sonuclar")
    lines.append("")
    lines.append("| Hedef | Oneri | v5 Skor | Etiket | Stabilite | Win Olasilik | Marj | Belirsizlik | Entropi |")
    lines.append("|---|---|---:|---|---|---:|---:|---:|---:|")
    for _, r in best.iterrows():
        lines.append(
            f"| {r['target_id']} | {r['ref_id']} ({r['ref_type']}) | {float(r['consensus_v5_score']):.3f} | {r['consensus_v5_label']} | {r['stability_class']} | {float(r['robust_win_prob']):.3f} | {float(r['robust_margin_prob']):.3f} | {float(r['mc_ci_width']):.3f} | {float(r['robust_top1_entropy']):.3f} |"
        )

    lines.append("")
    lines.append("## Cikti Dosyalari")
    lines.append("")
    lines.append(f"- `{full_path}`")
    lines.append(f"- `{sim_path}`")
    lines.append(f"- `{top3_path}`")
    lines.append(f"- `{best_path}`")
    lines.append(f"- `{dec_path}`")
    lines.append(f"- `{viz1}`")
    lines.append(f"- `{viz2}`")

    md_path = out_dir / "internet_graph_similarity_consensus_v5_report_tr.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {full_path}")
    print(f"Saved: {sim_path}")
    print(f"Saved: {top3_path}")
    print(f"Saved: {best_path}")
    print(f"Saved: {dec_path}")
    print(f"Saved: {viz1}")
    print(f"Saved: {viz2}")
    print(f"Saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
