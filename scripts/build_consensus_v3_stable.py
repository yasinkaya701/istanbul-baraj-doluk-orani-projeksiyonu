#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_family_set(text: str) -> set[str]:
    if text is None:
        return set()
    return {x.strip() for x in str(text).split(",") if x.strip()}


def target_label(target_id: str) -> str:
    m = {
        "01_humidity_precip_hexbin": "Nem-Yagis",
        "02_temp_humidity_seasonal_box": "Sicaklik-Nem",
        "03_pressure_precip_era_spearman": "Basinc-Yagis(era)",
        "04_mgm_monthly_pattern": "Aylik Patern",
        "05_lag_correlation": "Lag Korelasyon",
    }
    return m.get(target_id, target_id)


def wilson_lower_bound(p: np.ndarray, n: np.ndarray, z: float = 1.96) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)
    out = p.copy()
    valid = n > 0
    if not np.any(valid):
        return out
    pv = p[valid]
    nv = n[valid]
    z2 = z * z
    denom = 1.0 + z2 / nv
    center = pv + z2 / (2.0 * nv)
    radius = z * np.sqrt((pv * (1.0 - pv) + z2 / (4.0 * nv)) / nv)
    out[valid] = (center - radius) / denom
    return np.clip(out, 0.0, 1.0)


def recommendation_label(score: float, p_top1: float, robust_min: float) -> str:
    if score >= 0.76 and p_top1 >= 0.75 and robust_min >= 0.55:
        return "onerilir"
    if score >= 0.66 and p_top1 >= 0.50 and robust_min >= 0.40:
        return "uygun"
    if score >= 0.56:
        return "izlenmeli"
    return "zayif"


def stability_label(p_top1: float, ci_width: float) -> str:
    if p_top1 >= 0.85 and ci_width <= 0.10:
        return "cok_stabil"
    if p_top1 >= 0.65 and ci_width <= 0.14:
        return "stabil"
    if p_top1 >= 0.45:
        return "orta"
    return "oynak"


def policy_mode(label: str, gap: float, p_top1: float, ci_width: float) -> str:
    if str(label) == "onerilir" and gap >= 0.16 and p_top1 >= 0.70 and ci_width <= 0.14:
        return "single_primary"
    return "dual_primary_backup"


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    out_dir = base / "output" / "analysis" / "internet_graph_similarity"

    scores = pd.read_csv(out_dir / "internet_graph_similarity_scores.csv")
    wref = pd.read_csv(out_dir / "internet_graph_similarity_weight_sensitivity_ref.csv")
    aref = pd.read_csv(out_dir / "internet_graph_similarity_augmentation_robustness_ref.csv")
    fam = pd.read_csv(out_dir / "internet_graph_similarity_family_recommendation.csv")
    arch_rec = pd.read_csv(out_dir / "internet_graph_similarity_archetype_target_recommendation.csv")
    arch_ref = pd.read_csv(out_dir / "internet_graph_similarity_archetype_ref_clusters.csv")

    # Base table
    df = scores.copy()
    df = df.merge(wref[["target_id", "ref_id", "win_count", "win_freq"]], on=["target_id", "ref_id"], how="left")
    df = df.merge(
        aref[["target_id", "ref_id", "aug_win_count", "aug_win_freq"]], on=["target_id", "ref_id"], how="left"
    )
    df["win_count"] = df["win_count"].fillna(0.0).astype(float)
    df["win_freq"] = df["win_freq"].fillna(0.0).astype(float)
    df["aug_win_count"] = df["aug_win_count"].fillna(0.0).astype(float)
    df["aug_win_freq"] = df["aug_win_freq"].fillna(0.0).astype(float)

    # Family match
    fam_map: Dict[str, set[str]] = {str(r["target_id"]): parse_family_set(str(r["preferred_family"])) for _, r in fam.iterrows()}
    df["family_match"] = [
        1.0 if str(rt) in fam_map.get(str(t), set()) else 0.0
        for t, rt in zip(df["target_id"].astype(str), df["ref_type"].astype(str))
    ]

    # Archetype match
    tgt_cluster = arch_rec[["target_id", "best_cluster_id"]].rename(columns={"best_cluster_id": "target_best_cluster"})
    ref_cluster = arch_ref[["ref_id", "cluster_id"]].rename(columns={"cluster_id": "ref_cluster"})
    df = df.merge(tgt_cluster, on="target_id", how="left")
    df = df.merge(ref_cluster, on="ref_id", how="left")
    df["archetype_match"] = (
        (df["target_best_cluster"].fillna(-1).astype(int) == df["ref_cluster"].fillna(-2).astype(int)).astype(float)
    )

    # Total draw counts per target
    win_total = wref.groupby("target_id", as_index=False)["win_count"].sum().rename(columns={"win_count": "n_win_total"})
    aug_total = aref.groupby("target_id", as_index=False)["aug_win_count"].sum().rename(columns={"aug_win_count": "n_aug_total"})
    df = df.merge(win_total, on="target_id", how="left")
    df = df.merge(aug_total, on="target_id", how="left")
    df["n_win_total"] = df["n_win_total"].fillna(0.0).astype(float)
    df["n_aug_total"] = df["n_aug_total"].fillna(0.0).astype(float)

    # Conservative stability estimates
    df["win_lb"] = wilson_lower_bound(df["win_freq"].to_numpy(), df["n_win_total"].to_numpy())
    df["aug_lb"] = wilson_lower_bound(df["aug_win_freq"].to_numpy(), df["n_aug_total"].to_numpy())
    df["consistency"] = 1.0 - (df["win_freq"] - df["aug_win_freq"]).abs()

    # Consensus v3 base score (stability-aware)
    df["consensus_v3_base"] = (
        0.30 * df["hybrid_score"].astype(float)
        + 0.21 * df["win_lb"].astype(float)
        + 0.21 * df["aug_lb"].astype(float)
        + 0.10 * df["semantic_compat"].astype(float)
        + 0.10 * df["family_match"].astype(float)
        + 0.05 * df["archetype_match"].astype(float)
        + 0.03 * df["consistency"].astype(float)
    )

    # Monte Carlo rank stability
    rng = np.random.default_rng(42)
    draws = 3000
    mc_rows = []
    enriched = []

    for target_id, g in df.groupby("target_id", sort=False):
        g = g.reset_index(drop=True)
        m = len(g)

        alpha_w = g["win_count"].to_numpy(dtype=float) + 1.0
        beta_w = np.maximum(g["n_win_total"].to_numpy(dtype=float) - g["win_count"].to_numpy(dtype=float), 0.0) + 1.0
        alpha_a = g["aug_win_count"].to_numpy(dtype=float) + 1.0
        beta_a = np.maximum(g["n_aug_total"].to_numpy(dtype=float) - g["aug_win_count"].to_numpy(dtype=float), 0.0) + 1.0

        w_draw = rng.beta(alpha_w, beta_w, size=(draws, m))
        a_draw = rng.beta(alpha_a, beta_a, size=(draws, m))
        h_mu = g["hybrid_score"].to_numpy(dtype=float)
        h_draw = np.clip(rng.normal(loc=h_mu, scale=0.03, size=(draws, m)), 0.0, 1.0)

        sem = g["semantic_compat"].to_numpy(dtype=float)[None, :]
        famm = g["family_match"].to_numpy(dtype=float)[None, :]
        arch = g["archetype_match"].to_numpy(dtype=float)[None, :]
        cons_draw = 1.0 - np.abs(w_draw - a_draw)

        s_draw = (
            0.30 * h_draw
            + 0.21 * w_draw
            + 0.21 * a_draw
            + 0.10 * sem
            + 0.10 * famm
            + 0.05 * arch
            + 0.03 * cons_draw
        )

        winners = np.argmax(s_draw, axis=1)
        for i in range(m):
            p_top1 = float(np.mean(winners == i))
            q05, q95 = np.quantile(s_draw[:, i], [0.05, 0.95])
            mc_rows.append(
                {
                    "target_id": target_id,
                    "ref_id": g.iloc[i]["ref_id"],
                    "p_top1": p_top1,
                    "mc_score_mean": float(np.mean(s_draw[:, i])),
                    "mc_score_std": float(np.std(s_draw[:, i])),
                    "mc_q05": float(q05),
                    "mc_q95": float(q95),
                    "mc_ci_width": float(q95 - q05),
                }
            )

    mc_df = pd.DataFrame(mc_rows)
    df = df.merge(mc_df, on=["target_id", "ref_id"], how="left")

    # Final calibrated score
    df["consensus_v3_score"] = 0.72 * df["consensus_v3_base"].astype(float) + 0.28 * df["p_top1"].fillna(0.0).astype(float)
    df["robust_min"] = np.minimum(df["win_lb"], df["aug_lb"])
    df["consensus_v3_label"] = [
        recommendation_label(float(s), float(p), float(r))
        for s, p, r in zip(df["consensus_v3_score"], df["p_top1"], df["robust_min"])
    ]
    df["stability_class"] = [
        stability_label(float(p), float(w))
        for p, w in zip(df["p_top1"].fillna(0.0), df["mc_ci_width"].fillna(1.0))
    ]

    # Save full
    full_path = out_dir / "internet_graph_similarity_consensus_v3_scores.csv"
    df.sort_values(["target_id", "consensus_v3_score"], ascending=[True, False]).to_csv(
        full_path, index=False, float_format="%.6f"
    )

    # Top3 and best
    top_rows = []
    best_rows = []
    dec_rows = []
    for target_id, g in df.groupby("target_id", sort=False):
        g = g.sort_values("consensus_v3_score", ascending=False).reset_index(drop=True)
        for rank, (_, r) in enumerate(g.head(3).iterrows(), start=1):
            row = r.to_dict()
            row["rank"] = rank
            top_rows.append(row)

        p = g.iloc[0]
        b = g.iloc[1] if len(g) > 1 else g.iloc[0]
        gap = float(p["consensus_v3_score"] - b["consensus_v3_score"])
        mode = policy_mode(str(p["consensus_v3_label"]), gap, float(p["p_top1"]), float(p["mc_ci_width"]))
        conf = "yuksek" if mode == "single_primary" else ("orta" if gap >= 0.10 else "dusuk")

        best_rows.append(
            {
                "target_id": p["target_id"],
                "target_type": p["target_type"],
                "ref_id": p["ref_id"],
                "ref_type": p["ref_type"],
                "title": p["title"],
                "page_url": p["page_url"],
                "consensus_v3_score": float(p["consensus_v3_score"]),
                "consensus_v3_label": p["consensus_v3_label"],
                "stability_class": p["stability_class"],
                "hybrid_score": float(p["hybrid_score"]),
                "win_lb": float(p["win_lb"]),
                "aug_lb": float(p["aug_lb"]),
                "p_top1": float(p["p_top1"]),
                "mc_ci_width": float(p["mc_ci_width"]),
                "score_gap_to_second": gap,
            }
        )

        rule = (
            "Primary kullan; skor farki ve stabilite yuksek."
            if mode == "single_primary"
            else "Primary ve backup birlikte takip et; oynaklik/gap sinirda."
        )

        dec_rows.append(
            {
                "target_id": target_id,
                "target_label": target_label(str(target_id)),
                "decision_mode": mode,
                "decision_confidence": conf,
                "primary_ref_id": p["ref_id"],
                "primary_ref_type": p["ref_type"],
                "primary_score": float(p["consensus_v3_score"]),
                "primary_p_top1": float(p["p_top1"]),
                "primary_stability": p["stability_class"],
                "backup_ref_id": b["ref_id"],
                "backup_ref_type": b["ref_type"],
                "backup_score": float(b["consensus_v3_score"]),
                "gap_primary_backup": gap,
                "rule_text": rule,
                "primary_page_url": p["page_url"],
                "backup_page_url": b["page_url"],
            }
        )

    top3 = pd.DataFrame(top_rows)
    best = pd.DataFrame(best_rows).sort_values("target_id").reset_index(drop=True)
    dec = pd.DataFrame(dec_rows).sort_values("target_id").reset_index(drop=True)

    top3_path = out_dir / "internet_graph_similarity_consensus_v3_top3.csv"
    best_path = out_dir / "internet_graph_similarity_consensus_v3_best.csv"
    dec_path = out_dir / "internet_graph_similarity_consensus_v3_decision_pack.csv"
    mc_path = out_dir / "internet_graph_similarity_consensus_v3_monte_carlo.csv"

    top3.to_csv(top3_path, index=False, float_format="%.6f")
    best.to_csv(best_path, index=False, float_format="%.6f")
    dec.to_csv(dec_path, index=False, float_format="%.6f")
    mc_df.to_csv(mc_path, index=False, float_format="%.6f")

    # Visual 1: score + p_top1
    x = np.arange(len(best))
    labels = [target_label(str(t)) for t in best["target_id"].astype(str)]
    svals = best["consensus_v3_score"].to_numpy(dtype=float)
    pvals = best["p_top1"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2.0, 1.3]})
    ax = axes[0]
    ax.bar(x, svals, color="#2563eb", alpha=0.9, label="Consensus v3")
    ax.plot(x, pvals, color="#dc2626", marker="o", linewidth=2.0, label="Top1 olasiligi")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Skor / Olasilik")
    ax.set_title("Konsensus v3: Stabilite ve Tahmin Guveni")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    for i in range(len(best)):
        ax.text(i, svals[i] + 0.015, f"{svals[i]:.3f}", ha="center", fontsize=8)

    ax2 = axes[1]
    ci = best["mc_ci_width"].to_numpy(dtype=float)
    ax2.bar(x, ci, color="#0f766e", alpha=0.9)
    ax2.set_ylim(0, max(0.2, float(np.max(ci)) + 0.03))
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Belirsizlik (q95-q05)")
    ax2.set_title("Top1 Belirsizlik Genisligi (Dusuk Daha Iyi)")
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    viz1 = out_dir / "internet_graph_similarity_consensus_v3_summary.png"
    fig.savefig(viz1, dpi=180)
    plt.close(fig)

    # Visual 2: top3 uncertainty bands
    fig, axes = plt.subplots(len(best), 1, figsize=(12, 2.2 * len(best)))
    if len(best) == 1:
        axes = [axes]
    for ax, (tid, g) in zip(axes, top3.groupby("target_id", sort=False)):
        g = g.sort_values("rank")
        y = np.arange(len(g))
        mean = g["mc_score_mean"].to_numpy(dtype=float)
        lo = g["mc_q05"].to_numpy(dtype=float)
        hi = g["mc_q95"].to_numpy(dtype=float)
        ax.hlines(y, lo, hi, color="#64748b", linewidth=3)
        ax.plot(mean, y, "o", color="#2563eb")
        ax.set_yticks(y)
        ax.set_yticklabels([f"r{int(r)}:{rid}" for r, rid in zip(g["rank"], g["ref_id"])], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_title(target_label(str(tid)), fontsize=10)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    viz2 = out_dir / "internet_graph_similarity_consensus_v3_uncertainty_top3.png"
    fig.savefig(viz2, dpi=180)
    plt.close(fig)

    # Markdown report
    lines: list[str] = []
    lines.append("# Konsensus v3 Stabilite ve Tahmin Dogrulugu Raporu (TR)")
    lines.append("")
    lines.append("- Amaç: tahmini daha stabil yapmak ve hatali tekil secimleri azaltmak.")
    lines.append("- Yontem: Wilson alt siniri + Monte Carlo Top1 olasiligi + arsetip uyumu.")
    lines.append("- Skor: `0.72*base + 0.28*p_top1`")
    lines.append("- Base: `0.30*hybrid + 0.21*win_lb + 0.21*aug_lb + 0.10*semantic + 0.10*family + 0.05*archetype + 0.03*consistency`")
    lines.append("")
    lines.append("## Top-1 Sonuclar")
    lines.append("")
    lines.append("| Hedef | Oneri | Score | Etiket | Stabilite | Top1 Olasilik | Belirsizlik | Gap(1-2) |")
    lines.append("|---|---|---:|---|---|---:|---:|---:|")
    for _, r in best.iterrows():
        lines.append(
            f"| {r['target_id']} | {r['ref_id']} ({r['ref_type']}) | {float(r['consensus_v3_score']):.3f} | {r['consensus_v3_label']} | {r['stability_class']} | {float(r['p_top1']):.3f} | {float(r['mc_ci_width']):.3f} | {float(r['score_gap_to_second']):.3f} |"
        )

    lines.append("")
    lines.append("## Cikti Dosyalari")
    lines.append("")
    lines.append(f"- `{full_path}`")
    lines.append(f"- `{top3_path}`")
    lines.append(f"- `{best_path}`")
    lines.append(f"- `{dec_path}`")
    lines.append(f"- `{mc_path}`")
    lines.append(f"- `{viz1}`")
    lines.append(f"- `{viz2}`")

    md_path = out_dir / "internet_graph_similarity_consensus_v3_report_tr.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {full_path}")
    print(f"Saved: {top3_path}")
    print(f"Saved: {best_path}")
    print(f"Saved: {dec_path}")
    print(f"Saved: {mc_path}")
    print(f"Saved: {viz1}")
    print(f"Saved: {viz2}")
    print(f"Saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
