#!/usr/bin/env python3
"""Visualize all-disease x etiology climate-health outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot all-disease etiology outputs")
    p.add_argument("--root-dir", type=Path, default=Path("output/health_impact"))
    p.add_argument("--output-dir", type=Path, default=Path("output/health_impact/figures"))
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_direct_scores(summary_df: pd.DataFrame, out_path: Path) -> None:
    s = summary_df.copy()
    s = s.sort_values(["model", "direct_signal_score"], ascending=[True, False])

    strong = s[s["model"] == "strong"].head(10).copy()
    quant = s[s["model"] == "quant"].head(10).copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    for ax, df, title, color in [
        (axes[0], strong, "Strong: En Yuksek 10 Hastalik Grubu", "#d95f02"),
        (axes[1], quant, "Quant: En Yuksek 10 Hastalik Grubu", "#1b9e77"),
    ]:
        y = np.arange(len(df))
        ax.barh(y, df["direct_signal_score"], color=color, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(df["disease_group_tr"])
        ax.invert_yaxis()
        ax.set_xlabel("Dogrudan Risk Sinyal Skoru")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        for i, v in enumerate(df["direct_signal_score"]):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)

    fig.suptitle("Tum Hastalik Gruplari: Dogrudan Iklim-Risk Sinyalleri", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_etiology_coverage(matrix_df: pd.DataFrame, out_path: Path) -> None:
    m = matrix_df.copy()
    piv = (
        m.groupby(["disease_group_tr", "etiology_tr"])["quantifiable_with_current_data"]
        .mean()
        .reset_index()
        .pivot(index="disease_group_tr", columns="etiology_tr", values="quantifiable_with_current_data")
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    data = piv.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_title("Etiyoloji Bazinda Olculebilirlik (1=dogrudan hesaplanabiliyor)")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            txt = "1" if data[i, j] >= 0.5 else "0"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Olculebilirlik")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    summary_df = pd.read_csv(args.root_dir / "health_all_disease_summary.csv")
    matrix_df = pd.read_csv(args.root_dir / "health_all_disease_etiology_matrix.csv")

    out1 = args.output_dir / "fig06_all_disease_direct_scores.png"
    out2 = args.output_dir / "fig07_all_disease_etiology_coverage.png"

    plot_direct_scores(summary_df, out1)
    plot_etiology_coverage(matrix_df, out2)

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")


if __name__ == "__main__":
    main()
