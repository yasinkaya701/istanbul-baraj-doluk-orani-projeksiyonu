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


def bayes_rate(count: np.ndarray, total: np.ndarray, p0: np.ndarray, kappa: float) -> np.ndarray:
    return (count + kappa * p0) / np.maximum(total + kappa, 1e-9)


def recommendation_label(score: float, p_top1: float, ci_width: float, margin_prob: float) -> str:
    if score >= 0.78 and p_top1 >= 0.70 and ci_width <= 0.12 and margin_prob >= 0.35:
        return "onerilir"
    if score >= 0.68 and p_top1 >= 0.50 and ci_width <= 0.16 and margin_prob >= 0.20:
        return "uygun"
    if score >= 0.57:
        return "izlenmeli"
    return "zayif"


def stability_label(p_top1: float, ci_width: float, margin_prob: float) -> str:
    if p_top1 >= 0.82 and ci_width <= 0.10 and margin_prob >= 0.45:
        return "cok_stabil"
    if p_top1 >= 0.65 and ci_width <= 0.14 and margin_prob >= 0.25:
        return "stabil"
    if p_top1 >= 0.45:
        return "orta"
    return "oynak"


def policy_mode(label: str, gap: float, p_top1: float, ci_width: float, margin_prob: float) -> str:
    if str(label) == "onerilir" and gap >= 0.14 and p_top1 >= 0.68 and ci_width <= 0.13 and margin_prob >= 0.30:
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
    v3_best = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v3_best.csv")

    df = scores.copy()
    df = df.merge(wref[["target_id", "ref_id", "win_count", "win_freq"]], on=["target_id", "ref_id"], how="left")
    df = df.merge(
        aref[["target_id", "ref_id", "aug_win_count", "aug_win_freq"]], on=["target_id", "ref_id"], how="left"
    )
    for c in ["win_count", "win_freq", "aug_win_count", "aug_win_freq"]:
        df[c] = df[c].fillna(0.0).astype(float)

    # Family / archetype matches
    fam_map: Dict[str, set[str]] = {str(r["target_id"]): parse_family_set(str(r["preferred_family"])) for _, r in fam.iterrows()}
    df["family_match"] = [
        1.0 if str(rt) in fam_map.get(str(t), set()) else 0.0
        for t, rt in zip(df["target_id"].astype(str), df["ref_type"].astype(str))
    ]

    tgt_cluster = arch_rec[["target_id", "best_cluster_id"]].rename(columns={"best_cluster_id": "target_best_cluster"})
    ref_cluster = arch_ref[["ref_id", "cluster_id"]].rename(columns={"cluster_id": "ref_cluster"})
    df = df.merge(tgt_cluster, on="target_id", how="left")
    df = df.merge(ref_cluster, on="ref_id", how="left")
    df["archetype_match"] = (
        (df["target_best_cluster"].fillna(-1).astype(int) == df["ref_cluster"].fillna(-2).astype(int)).astype(float)
    )

    # Totals
    win_total = wref.groupby("target_id", as_index=False)["win_count"].sum().rename(columns={"win_count": "n_win_total"})
    aug_total = aref.groupby("target_id", as_index=False)["aug_win_count"].sum().rename(columns={"aug_win_count": "n_aug_total"})
    df = df.merge(win_total, on="target_id", how="left")
    df = df.merge(aug_total, on="target_id", how="left")
    df["n_win_total"] = df["n_win_total"].fillna(0.0).astype(float)
    df["n_aug_total"] = df["n_aug_total"].fillna(0.0).astype(float)

    # Bayesian shrinkage prior: p0 = 1/m for each target
    m_per_target = df.groupby("target_id")["ref_id"].transform("nunique").astype(float)
    p0 = (1.0 / np.maximum(m_per_target, 1.0)).to_numpy(dtype=float)
    kappa = 60.0

    df["win_bayes"] = bayes_rate(df["win_count"].to_numpy(dtype=float), df["n_win_total"].to_numpy(dtype=float), p0, kappa)
    df["aug_bayes"] = bayes_rate(df["aug_win_count"].to_numpy(dtype=float), df["n_aug_total"].to_numpy(dtype=float), p0, kappa)
    df["consistency"] = 1.0 - np.abs(df["win_bayes"] - df["aug_bayes"])

    # Deterministic base score (calibrated for stability)
    df["consensus_v4_base"] = (
        0.27 * df["hybrid_score"].astype(float)
        + 0.23 * df["win_bayes"].astype(float)
        + 0.23 * df["aug_bayes"].astype(float)
        + 0.10 * df["semantic_compat"].astype(float)
        + 0.09 * df["family_match"].astype(float)
        + 0.05 * df["archetype_match"].astype(float)
        + 0.03 * df["consistency"].astype(float)
    )

    # Monte Carlo with posterior uncertainty + mild metric noise
    rng = np.random.default_rng(42)
    draws = 3500
    mc_rows = []
    p2_rows = []

    for target_id, g in df.groupby("target_id", sort=False):
        g = g.reset_index(drop=True)
        m = len(g)
        p0_t = np.repeat(1.0 / max(m, 1), m)

        aw = g["win_count"].to_numpy(dtype=float) + kappa * p0_t + 1.0
        bw = np.maximum(g["n_win_total"].to_numpy(dtype=float) - g["win_count"].to_numpy(dtype=float), 0.0) + kappa * (1.0 - p0_t) + 1.0
        aa = g["aug_win_count"].to_numpy(dtype=float) + kappa * p0_t + 1.0
        ba = np.maximum(g["n_aug_total"].to_numpy(dtype=float) - g["aug_win_count"].to_numpy(dtype=float), 0.0) + kappa * (1.0 - p0_t) + 1.0

        w_draw = rng.beta(aw, bw, size=(draws, m))
        a_draw = rng.beta(aa, ba, size=(draws, m))

        h_mu = g["hybrid_score"].to_numpy(dtype=float)
        sem_mu = g["semantic_compat"].to_numpy(dtype=float)
        h_draw = np.clip(rng.normal(loc=h_mu, scale=0.028, size=(draws, m)), 0.0, 1.0)
        s_draw = np.clip(rng.normal(loc=sem_mu, scale=0.015, size=(draws, m)), 0.0, 1.0)

        famm = g["family_match"].to_numpy(dtype=float)[None, :]
        arch = g["archetype_match"].to_numpy(dtype=float)[None, :]
        cons_draw = 1.0 - np.abs(w_draw - a_draw)

        score_draw = (
            0.27 * h_draw
            + 0.23 * w_draw
            + 0.23 * a_draw
            + 0.10 * s_draw
            + 0.09 * famm
            + 0.05 * arch
            + 0.03 * cons_draw
        )

        order = np.argsort(score_draw, axis=1)[:, ::-1]
        top1 = order[:, 0]
        top2 = order[:, 1] if m > 1 else order[:, 0]
        # Calibrated support probability via tempered softmax (less overconfident than winner frequency).
        temp = 0.10
        logits = score_draw / temp
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        prob_draw = exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)
        p_top1_vec = prob_draw.mean(axis=0)
        p_top2_vec = np.array(
            [
                float(np.max(np.delete(p_top1_vec, i))) if m > 1 else float(p_top1_vec[i])
                for i in range(m)
            ],
            dtype=float,
        )

        for i in range(m):
            p_top1 = float(p_top1_vec[i])
            p_top2 = float(p_top2_vec[i])
            mean = float(np.mean(score_draw[:, i]))
            std = float(np.std(score_draw[:, i]))
            q05, q95 = np.quantile(score_draw[:, i], [0.05, 0.95])
            mc_rows.append(
                {
                    "target_id": target_id,
                    "ref_id": g.iloc[i]["ref_id"],
                    "p_top1": p_top1,
                    "p_top2": p_top2,
                    "margin_prob": p_top1 - p_top2,
                    "mc_score_mean": mean,
                    "mc_score_std": std,
                    "mc_q05": float(q05),
                    "mc_q95": float(q95),
                    "mc_ci_width": float(q95 - q05),
                }
            )

        # target-level uncertainty concentration
        p = p_top1_vec / np.maximum(p_top1_vec.sum(), 1e-12)
        p_pos = p[p > 0]
        entropy = -float(np.sum(p_pos * np.log(p_pos))) / float(np.log(max(m, 2)))
        p2_rows.append({"target_id": target_id, "top1_entropy": entropy})

    mc_df = pd.DataFrame(mc_rows)
    ent_df = pd.DataFrame(p2_rows)

    df = df.merge(mc_df, on=["target_id", "ref_id"], how="left")
    df = df.merge(ent_df, on="target_id", how="left")

    # Final calibrated score: stability + uncertainty aware
    df["uncertainty_bonus"] = 1.0 - df["mc_ci_width"].fillna(1.0).astype(float)
    df["consensus_v4_score"] = (
        0.58 * df["consensus_v4_base"].astype(float)
        + 0.22 * df["p_top1"].fillna(0.0).astype(float)
        + 0.12 * df["uncertainty_bonus"].astype(float)
        + 0.08 * np.clip(df["margin_prob"].fillna(0.0).astype(float), -1.0, 1.0)
    )

    df["consensus_v4_label"] = [
        recommendation_label(float(s), float(p), float(w), float(m))
        for s, p, w, m in zip(df["consensus_v4_score"], df["p_top1"], df["mc_ci_width"], df["margin_prob"])
    ]
    df["stability_class"] = [
        stability_label(float(p), float(w), float(m))
        for p, w, m in zip(df["p_top1"], df["mc_ci_width"], df["margin_prob"])
    ]

    full_path = out_dir / "internet_graph_similarity_consensus_v4_scores.csv"
    df.sort_values(["target_id", "consensus_v4_score"], ascending=[True, False]).to_csv(
        full_path, index=False, float_format="%.6f"
    )

    top_rows = []
    best_rows = []
    dec_rows = []

    for target_id, g in df.groupby("target_id", sort=False):
        g = g.sort_values("consensus_v4_score", ascending=False).reset_index(drop=True)
        for rank, (_, r) in enumerate(g.head(3).iterrows(), start=1):
            row = r.to_dict()
            row["rank"] = rank
            top_rows.append(row)

        p = g.iloc[0]
        b = g.iloc[1] if len(g) > 1 else g.iloc[0]
        gap = float(p["consensus_v4_score"] - b["consensus_v4_score"])
        mode = policy_mode(
            str(p["consensus_v4_label"]),
            gap,
            float(p["p_top1"]),
            float(p["mc_ci_width"]),
            float(p["margin_prob"]),
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
                "consensus_v4_score": float(p["consensus_v4_score"]),
                "consensus_v4_label": p["consensus_v4_label"],
                "stability_class": p["stability_class"],
                "p_top1": float(p["p_top1"]),
                "margin_prob": float(p["margin_prob"]),
                "mc_ci_width": float(p["mc_ci_width"]),
                "top1_entropy": float(p["top1_entropy"]),
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
                "primary_score": float(p["consensus_v4_score"]),
                "primary_p_top1": float(p["p_top1"]),
                "primary_margin_prob": float(p["margin_prob"]),
                "primary_stability": p["stability_class"],
                "backup_ref_id": b["ref_id"],
                "backup_ref_type": b["ref_type"],
                "backup_score": float(b["consensus_v4_score"]),
                "gap_primary_backup": gap,
                "rule_text": (
                    "Primary kullan; skor farki + olasilik marji yuksek."
                    if mode == "single_primary"
                    else "Primary ve backup birlikte izle; marj veya belirsizlik sinirda."
                ),
                "primary_page_url": p["page_url"],
                "backup_page_url": b["page_url"],
            }
        )

    top3 = pd.DataFrame(top_rows)
    best = pd.DataFrame(best_rows).sort_values("target_id").reset_index(drop=True)
    dec = pd.DataFrame(dec_rows).sort_values("target_id").reset_index(drop=True)

    top3_path = out_dir / "internet_graph_similarity_consensus_v4_top3.csv"
    best_path = out_dir / "internet_graph_similarity_consensus_v4_best.csv"
    dec_path = out_dir / "internet_graph_similarity_consensus_v4_decision_pack.csv"
    mc_path = out_dir / "internet_graph_similarity_consensus_v4_monte_carlo.csv"

    top3.to_csv(top3_path, index=False, float_format="%.6f")
    best.to_csv(best_path, index=False, float_format="%.6f")
    dec.to_csv(dec_path, index=False, float_format="%.6f")
    mc_df.to_csv(mc_path, index=False, float_format="%.6f")

    # Visual 1: v4 summary
    x = np.arange(len(best))
    labels = [target_label(str(t)) for t in best["target_id"].astype(str)]
    s = best["consensus_v4_score"].to_numpy(dtype=float)
    p = best["p_top1"].to_numpy(dtype=float)
    m = best["margin_prob"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2.0, 1.3]})
    ax = axes[0]
    ax.bar(x, s, color="#2563eb", alpha=0.92, label="Consensus v4")
    ax.plot(x, p, color="#dc2626", marker="o", linewidth=2, label="Top1 olasiligi")
    ax.plot(x, np.clip(m, 0.0, 1.0), color="#0f766e", marker="s", linewidth=2, label="Marj olasiligi")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Skor / Olasilik")
    ax.set_title("Konsensus v4: Kalibre Stabil Tahmin")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    ax2 = axes[1]
    ci = best["mc_ci_width"].to_numpy(dtype=float)
    ent = best["top1_entropy"].to_numpy(dtype=float)
    w = 0.36
    ax2.bar(x - w / 2, ci, width=w, color="#334155", label="Belirsizlik (CI genisligi)")
    ax2.bar(x + w / 2, ent, width=w, color="#7c3aed", label="Top1 entropi")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max(0.35, float(max(np.max(ci), np.max(ent))) + 0.03))
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    viz1 = out_dir / "internet_graph_similarity_consensus_v4_summary.png"
    fig.savefig(viz1, dpi=180)
    plt.close(fig)

    # Visual 2: v3 vs v4 score compare
    cmp = v3_best[["target_id", "consensus_v3_score"]].merge(
        best[["target_id", "consensus_v4_score"]], on="target_id", how="outer"
    ).sort_values("target_id")
    x = np.arange(len(cmp))
    labels_cmp = [target_label(str(t)) for t in cmp["target_id"].astype(str)]

    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    ww = 0.36
    ax.bar(x - ww / 2, cmp["consensus_v3_score"], width=ww, color="#94a3b8", label="v3")
    ax.bar(x + ww / 2, cmp["consensus_v4_score"], width=ww, color="#2563eb", label="v4")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_cmp)
    ax.set_ylabel("Skor")
    ax.set_title("Konsensus v3 vs v4")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    viz2 = out_dir / "internet_graph_similarity_consensus_v3_v4_compare.png"
    fig.savefig(viz2, dpi=180)
    plt.close(fig)

    # report
    lines = []
    lines.append("# Konsensus v4 Kalibre Stabil Tahmin Raporu (TR)")
    lines.append("")
    lines.append("- Hedef: stabiliteyi korurken asiri-guveni azaltmak ve tahmini daha guvenilir hale getirmek.")
    lines.append("- Yenilik: Bayesci shrinkage (`win_bayes/aug_bayes`) + Monte Carlo marj olasiligi (`margin_prob`) + entropi kontrolu.")
    lines.append("")
    lines.append("## Top-1 Sonuclar")
    lines.append("")
    lines.append("| Hedef | Oneri | v4 Skor | Etiket | Stabilite | Top1 Olasilik | Marj | Belirsizlik | Entropi |")
    lines.append("|---|---|---:|---|---|---:|---:|---:|---:|")
    for _, r in best.iterrows():
        lines.append(
            f"| {r['target_id']} | {r['ref_id']} ({r['ref_type']}) | {float(r['consensus_v4_score']):.3f} | {r['consensus_v4_label']} | {r['stability_class']} | {float(r['p_top1']):.3f} | {float(r['margin_prob']):.3f} | {float(r['mc_ci_width']):.3f} | {float(r['top1_entropy']):.3f} |"
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

    md_path = out_dir / "internet_graph_similarity_consensus_v4_report_tr.md"
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
