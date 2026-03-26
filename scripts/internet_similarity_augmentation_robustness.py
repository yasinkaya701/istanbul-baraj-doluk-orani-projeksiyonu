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
    scores = pd.read_csv(out_dir / "internet_graph_similarity_scores.csv")

    # Metric-perturbation robustness:
    # image-level augmentations are approximated as bounded random perturbations
    # over pairwise feature metrics.
    n_aug = 4000
    rng = np.random.default_rng(20260306)

    df = scores.copy().reset_index(drop=True)
    targets = list(df["target_id"].drop_duplicates())

    wins_ref = {(t, r): 0 for t in targets for r in df.loc[df["target_id"] == t, "ref_id"].unique()}
    wins_type = {(t, rt): 0 for t in targets for rt in df.loc[df["target_id"] == t, "ref_type"].unique()}

    base_edge_iou = df["edge_iou"].astype(float).to_numpy()
    base_edge_corr = df["edge_corr"].astype(float).to_numpy()
    base_hist_corr = df["hist_corr"].astype(float).to_numpy()
    base_ahash = df["ahash_sim"].astype(float).to_numpy()
    base_ssim = df["ssim"].astype(float).to_numpy()
    base_hog = df["hog_corr"].astype(float).to_numpy()
    sem = df["semantic_compat"].astype(float).to_numpy()
    pen = df["compat_penalty"].astype(float).to_numpy()

    target_idx = {t: np.where(df["target_id"].astype(str).to_numpy() == t)[0] for t in targets}

    for _ in range(n_aug):
        # Perturbation scales tuned to mimic mild style/quality changes
        edge_iou = np.clip(base_edge_iou + rng.normal(0.0, 0.020, size=len(df)), 0.0, 1.0)
        edge_corr = np.clip(base_edge_corr + rng.normal(0.0, 0.060, size=len(df)), -1.0, 1.0)
        hist_corr = np.clip(base_hist_corr + rng.normal(0.0, 0.050, size=len(df)), -1.0, 1.0)
        ahash = np.clip(base_ahash + rng.normal(0.0, 0.045, size=len(df)), 0.0, 1.0)
        ssim = np.clip(base_ssim + rng.normal(0.0, 0.050, size=len(df)), 0.0, 1.0)
        hog = np.clip(base_hog + rng.normal(0.0, 0.070, size=len(df)), -1.0, 1.0)

        ec_n = (edge_corr + 1.0) / 2.0
        hc_n = (hist_corr + 1.0) / 2.0
        hg_n = (hog + 1.0) / 2.0

        visual = 0.22 * edge_iou + 0.16 * ec_n + 0.14 * hc_n + 0.10 * ahash + 0.18 * ssim + 0.20 * hg_n
        hybrid = (0.68 * visual + 0.32 * sem) * pen

        for t in targets:
            idx = target_idx[t]
            sub = hybrid[idx]
            j = int(np.argmax(sub))
            gidx = int(idx[j])
            ref_id = str(df.iloc[gidx]["ref_id"])
            ref_type = str(df.iloc[gidx]["ref_type"])
            wins_ref[(t, ref_id)] += 1
            wins_type[(t, ref_type)] += 1

    ref_rows = []
    for (t, r), c in wins_ref.items():
        ref_rows.append({"target_id": t, "ref_id": r, "aug_win_count": c, "aug_win_freq": c / n_aug})
    ref_df = pd.DataFrame(ref_rows).sort_values(["target_id", "aug_win_freq"], ascending=[True, False]).reset_index(drop=True)
    ref_out = out_dir / "internet_graph_similarity_augmentation_robustness_ref.csv"
    ref_df.to_csv(ref_out, index=False, float_format="%.6f")

    type_rows = []
    for (t, rt), c in wins_type.items():
        type_rows.append({"target_id": t, "ref_type": rt, "aug_type_win_count": c, "aug_type_win_freq": c / n_aug})
    type_df = pd.DataFrame(type_rows).sort_values(["target_id", "aug_type_win_freq"], ascending=[True, False]).reset_index(drop=True)
    type_out = out_dir / "internet_graph_similarity_augmentation_robustness_type.csv"
    type_df.to_csv(type_out, index=False, float_format="%.6f")

    top_rows = []
    for t in targets:
        br = ref_df[ref_df["target_id"] == t].iloc[0]
        bt = type_df[type_df["target_id"] == t].iloc[0]
        top_rows.append(
            {
                "target_id": t,
                "best_aug_ref": br["ref_id"],
                "best_aug_ref_freq": br["aug_win_freq"],
                "best_aug_type": bt["ref_type"],
                "best_aug_type_freq": bt["aug_type_win_freq"],
                "n_perturbations": n_aug,
            }
        )
    top_df = pd.DataFrame(top_rows).sort_values("target_id").reset_index(drop=True)
    top_out = out_dir / "internet_graph_similarity_augmentation_robustness_top.csv"
    top_df.to_csv(top_out, index=False, float_format="%.6f")

    # visualization
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 9.5))
    axes_arr = np.array(axes).reshape(-1)
    for i, t in enumerate(targets):
        ax = axes_arr[i]
        g = ref_df[ref_df["target_id"] == t].head(3)
        vals = g["aug_win_freq"].to_numpy(dtype=float)
        ax.bar(np.arange(len(g)), vals, color=["#0284c7", "#0ea5e9", "#7dd3fc"])
        ax.set_ylim(0, 1)
        ax.set_title(target_label(t), fontsize=10)
        ax.set_xticks(np.arange(len(g)))
        ax.set_xticklabels(g["ref_id"].astype(str).to_list(), rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for j, v in enumerate(vals):
            ax.text(j, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for k in range(len(targets), len(axes_arr)):
        axes_arr[k].axis("off")
    fig.suptitle("Perturbasyon Robustlugu: Top-3 Kazanan Frekanslari", fontsize=12)
    fig.tight_layout()
    viz_out = out_dir / "internet_graph_similarity_augmentation_robustness_top3.png"
    fig.savefig(viz_out, dpi=180)
    plt.close(fig)

    lines = []
    lines.append("# Perturbasyon Robustluk Raporu (TR)")
    lines.append("")
    lines.append(f"- Simulasyon sayisi: {n_aug}")
    lines.append("- Not: Bu analiz image-level augmentasyon yerine metrik-perturbasyon yaklasimi ile hizli robustluk testi yapar.")
    lines.append("")
    lines.append("| Hedef | En Stabil Ref | Ref Frekans | En Stabil Tip | Tip Frekans |")
    lines.append("|---|---|---:|---|---:|")
    for _, r in top_df.iterrows():
        lines.append(
            f"| {r['target_id']} | {r['best_aug_ref']} | {float(r['best_aug_ref_freq']):.3f} | "
            f"{r['best_aug_type']} | {float(r['best_aug_type_freq']):.3f} |"
        )
    lines.append("")
    lines.append("## Cikti Dosyalari")
    lines.append("")
    lines.append(f"- `{ref_out}`")
    lines.append(f"- `{type_out}`")
    lines.append(f"- `{top_out}`")
    lines.append(f"- `{viz_out}`")
    md_out = out_dir / "internet_graph_similarity_augmentation_robustness_report_tr.md"
    md_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {ref_out}")
    print(f"Saved: {type_out}")
    print(f"Saved: {top_out}")
    print(f"Saved: {viz_out}")
    print(f"Saved: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
