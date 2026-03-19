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


def confidence_from_freq(freq: float) -> str:
    if freq >= 0.70:
        return "yuksek"
    if freq >= 0.50:
        return "orta_yuksek"
    if freq >= 0.30:
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


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    out_dir = base / "output" / "analysis" / "internet_graph_similarity"
    scores = pd.read_csv(out_dir / "internet_graph_similarity_scores.csv")

    # Deterministic reproducibility
    rng = np.random.default_rng(20260306)
    n_iter = 2000

    # Base visual component proportions used in the main pipeline
    base_w = np.array([0.22, 0.16, 0.14, 0.10, 0.18, 0.20], dtype=float)
    alpha = base_w * 50.0  # concentrates samples around base weights

    # Precompute normalized components
    df = scores.copy()
    df["edge_corr_n"] = (df["edge_corr"].astype(float) + 1.0) / 2.0
    df["hist_corr_n"] = (df["hist_corr"].astype(float) + 1.0) / 2.0
    df["hog_corr_n"] = (df["hog_corr"].astype(float) + 1.0) / 2.0

    targets = list(df["target_id"].drop_duplicates())
    wins_ref = {(t, r): 0 for t in targets for r in df.loc[df["target_id"] == t, "ref_id"].unique()}
    wins_type = {(t, rt): 0 for t in targets for rt in df.loc[df["target_id"] == t, "ref_type"].unique()}
    wins_family_ref = {(t, r): 0 for t in targets for r in df.loc[df["target_id"] == t, "ref_id"].unique()}

    for _ in range(n_iter):
        w = rng.dirichlet(alpha)
        mix = float(rng.uniform(0.60, 0.75))  # visual weight

        visual = (
            w[0] * df["edge_iou"].astype(float)
            + w[1] * df["edge_corr_n"].astype(float)
            + w[2] * df["hist_corr_n"].astype(float)
            + w[3] * df["ahash_sim"].astype(float)
            + w[4] * df["ssim"].astype(float)
            + w[5] * df["hog_corr_n"].astype(float)
        )
        hybrid = (mix * visual + (1.0 - mix) * df["semantic_compat"].astype(float)) * df["compat_penalty"].astype(float)
        df["hybrid_dyn"] = hybrid

        for t in targets:
            g = df[df["target_id"] == t].sort_values("hybrid_dyn", ascending=False).reset_index(drop=True)
            win = g.iloc[0]
            wins_ref[(t, str(win["ref_id"]))] += 1
            wins_type[(t, str(win["ref_type"]))] += 1

            pref = preferred_ref_types(str(win["target_type"]))
            gf = g[g["ref_type"].astype(str).isin(pref)].reset_index(drop=True)
            if len(gf) == 0:
                gf = g
            fwin = gf.iloc[0]
            wins_family_ref[(t, str(fwin["ref_id"]))] += 1

    rows_ref = []
    for (t, r), c in wins_ref.items():
        rows_ref.append({"target_id": t, "ref_id": r, "win_count": c, "win_freq": c / n_iter})
    ref_df = pd.DataFrame(rows_ref).sort_values(["target_id", "win_freq"], ascending=[True, False]).reset_index(drop=True)
    ref_out = out_dir / "internet_graph_similarity_weight_sensitivity_ref.csv"
    ref_df.to_csv(ref_out, index=False, float_format="%.6f")

    rows_type = []
    for (t, rt), c in wins_type.items():
        rows_type.append({"target_id": t, "ref_type": rt, "win_count": c, "win_freq": c / n_iter})
    type_df = pd.DataFrame(rows_type).sort_values(["target_id", "win_freq"], ascending=[True, False]).reset_index(drop=True)
    type_out = out_dir / "internet_graph_similarity_weight_sensitivity_type.csv"
    type_df.to_csv(type_out, index=False, float_format="%.6f")

    rows_family = []
    for (t, r), c in wins_family_ref.items():
        rows_family.append({"target_id": t, "ref_id": r, "family_win_count": c, "family_win_freq": c / n_iter})
    fam_df = pd.DataFrame(rows_family).sort_values(["target_id", "family_win_freq"], ascending=[True, False]).reset_index(drop=True)
    fam_out = out_dir / "internet_graph_similarity_weight_sensitivity_family_ref.csv"
    fam_df.to_csv(fam_out, index=False, float_format="%.6f")

    # Top summary
    top_rows = []
    for t in targets:
        best_ref = ref_df[ref_df["target_id"] == t].iloc[0]
        best_type = type_df[type_df["target_id"] == t].iloc[0]
        best_family = fam_df[fam_df["target_id"] == t].iloc[0]
        top_rows.append(
            {
                "target_id": t,
                "best_ref": best_ref["ref_id"],
                "best_ref_freq": best_ref["win_freq"],
                "best_ref_confidence": confidence_from_freq(float(best_ref["win_freq"])),
                "best_type": best_type["ref_type"],
                "best_type_freq": best_type["win_freq"],
                "best_type_confidence": confidence_from_freq(float(best_type["win_freq"])),
                "best_family_ref": best_family["ref_id"],
                "best_family_ref_freq": best_family["family_win_freq"],
                "best_family_ref_confidence": confidence_from_freq(float(best_family["family_win_freq"])),
            }
        )
    top_df = pd.DataFrame(top_rows).sort_values("target_id").reset_index(drop=True)
    top_out = out_dir / "internet_graph_similarity_weight_sensitivity_top.csv"
    top_df.to_csv(top_out, index=False, float_format="%.6f")

    # Plot: top-3 winner frequencies per target
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 9.5))
    axes_arr = np.array(axes).reshape(-1)
    for i, t in enumerate(targets):
        ax = axes_arr[i]
        g = ref_df[ref_df["target_id"] == t].head(3).copy()
        ax.bar(np.arange(len(g)), g["win_freq"].astype(float).to_numpy(), color=["#2563eb", "#0ea5e9", "#93c5fd"])
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(len(g)))
        ax.set_xticklabels(g["ref_id"].astype(str).to_list(), rotation=20, ha="right", fontsize=8)
        ax.set_title(target_label(t), fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        for j, v in enumerate(g["win_freq"].astype(float).to_numpy()):
            ax.text(j, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for k in range(len(targets), len(axes_arr)):
        axes_arr[k].axis("off")
    fig.suptitle("Ağırlık Duyarlılığı: Top-3 Kazanan Frekansları", fontsize=12)
    fig.tight_layout()
    sens_png = out_dir / "internet_graph_similarity_weight_sensitivity_top3.png"
    fig.savefig(sens_png, dpi=180)
    plt.close(fig)

    # Markdown report
    lines: list[str] = []
    lines.append("# Agirlik Duyarlilik Raporu (TR)")
    lines.append("")
    lines.append(f"- Simulasyon sayisi: {n_iter}")
    lines.append("- Her simulasyonda gorsel agirliklar rastgele (Dirichlet), hibrit karisim agirligi rastgele secildi.")
    lines.append("")
    lines.append("| Hedef | En Stabil Ref | Ref Frekans | Ref Guven | En Stabil Tip | Tip Frekans | Tip Guven |")
    lines.append("|---|---|---:|---|---|---:|---|")
    for _, r in top_df.iterrows():
        lines.append(
            f"| {r['target_id']} | {r['best_ref']} | {float(r['best_ref_freq']):.3f} | {r['best_ref_confidence']} | "
            f"{r['best_type']} | {float(r['best_type_freq']):.3f} | {r['best_type_confidence']} |"
        )
    lines.append("")
    lines.append("## Cikti Dosyalari")
    lines.append("")
    lines.append(f"- `{ref_out}`")
    lines.append(f"- `{type_out}`")
    lines.append(f"- `{fam_out}`")
    lines.append(f"- `{top_out}`")
    lines.append(f"- `{sens_png}`")
    md_out = out_dir / "internet_graph_similarity_weight_sensitivity_report_tr.md"
    md_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {ref_out}")
    print(f"Saved: {type_out}")
    print(f"Saved: {fam_out}")
    print(f"Saved: {top_out}")
    print(f"Saved: {sens_png}")
    print(f"Saved: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
