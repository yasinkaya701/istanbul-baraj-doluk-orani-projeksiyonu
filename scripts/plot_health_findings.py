#!/usr/bin/env python3
"""Create human-readable visualization pack for health impact findings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot health impact findings")
    p.add_argument("--root-dir", type=Path, default=Path("output/health_impact"))
    p.add_argument("--output-dir", type=Path, default=Path("output/health_impact/figures"))
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = pd.read_csv(root / "model_comparison_summary.csv")
    strong = pd.read_csv(root / "strong" / "sensitivity" / "sensitivity_summary.csv")
    quant = pd.read_csv(root / "quant" / "sensitivity" / "sensitivity_summary.csv")
    lit = pd.read_csv(root / "literatur_tutarlilik_kontrolu_2026-03-05.csv")
    return model, strong, quant, lit


def clean_ok(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns:
        return df[df["status"] == "ok"].copy()
    return df.copy()


def plot_model_overview(model: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    m = model.copy()
    m = m.sort_values("model")
    names = m["model"].tolist()

    axes[0].bar(names, m["future_rr_mean"], color=["#1f77b4", "#ff7f0e"])
    axes[0].set_title("Gelecek Ortalama RR")
    axes[0].set_ylim(bottom=1.0)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(names, m["future_af_mean"], color=["#1f77b4", "#ff7f0e"])
    axes[1].set_title("Gelecek Ortalama AF")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].bar(names, m["delta_ood_share"], color=["#1f77b4", "#ff7f0e"])
    axes[2].set_title("Delta OOD Share")
    axes[2].set_ylim(bottom=0.0)
    axes[2].grid(axis="y", alpha=0.2)

    fig.suptitle("Model Ozet Gorunumu", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_rr_distribution(strong_ok: pd.DataFrame, quant_ok: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    s_vals = strong_ok["future_rr_mean"].to_numpy()
    q_vals = quant_ok["future_rr_mean"].to_numpy()

    axes[0].boxplot([s_vals, q_vals], tick_labels=["strong", "quant"], showmeans=True)
    axes[0].set_title("Future RR Dagilimi")
    axes[0].set_ylim(bottom=min(1.0, q_vals.min() * 0.999))
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].hist(s_vals, bins=20, alpha=0.7, label="strong", color="#1f77b4")
    axes[1].hist(q_vals, bins=20, alpha=0.7, label="quant", color="#ff7f0e")
    axes[1].set_title("Future RR Histogram")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()

    fig.suptitle("Belirsizlik Dagilimi", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scenario_ranges(strong_ok: pd.DataFrame, quant_ok: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    def range_vals(df: pd.DataFrame, label: str, color: str, y: float) -> None:
        mn = float(df["future_rr_mean"].min())
        mx = float(df["future_rr_mean"].max())
        md = float(df["future_rr_mean"].median())
        ax.hlines(y, mn, mx, color=color, linewidth=6, alpha=0.7)
        ax.plot(md, y, "o", color="black", markersize=6)
        ax.text(mx, y + 0.03, f"{label}: {mn:.3f} - {mx:.3f}", fontsize=9, color=color)

    range_vals(strong_ok, "strong", "#1f77b4", 1.0)
    range_vals(quant_ok, "quant", "#ff7f0e", 0.65)

    ax.set_title("Senaryo Araliklari (Future RR)")
    ax.set_xlabel("Future RR")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_alignment(strong_ok: pd.DataFrame, quant_ok: pd.DataFrame, lit: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    align = pd.DataFrame(
        {
            "model": ["strong", "quant"],
            "delta_rr_positive_share": [
                float((strong_ok["delta_rr_mean"] > 0).mean()),
                float((quant_ok["delta_rr_mean"] > 0).mean()),
            ],
        }
    )

    axes[0].bar(align["model"], align["delta_rr_positive_share"], color=["#1f77b4", "#ff7f0e"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Delta RR > 0 Senaryo Orani")
    axes[0].grid(axis="y", alpha=0.2)

    verdict_counts = lit["verdict"].value_counts()
    labels = verdict_counts.index.tolist()
    sizes = verdict_counts.values
    colors = ["#2ca02c" if str(x).lower() == "uyumlu" else "#d62728" for x in labels]
    axes[1].pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90, colors=colors)
    axes[1].set_title("Literatur Tutarlilik Ozeti")

    fig.suptitle("Literaturle Uyum Kontrolu", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_epi_adaptation_heatmap(strong_ok: pd.DataFrame, quant_ok: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    def draw(df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
        piv = (
            df.groupby(["epi_mode", "adaptation_mode"])["future_rr_mean"]
            .mean()
            .reset_index()
            .pivot(index="epi_mode", columns="adaptation_mode", values="future_rr_mean")
        )
        piv = piv.reindex(columns=sorted(piv.columns))
        data = piv.to_numpy(dtype=float)
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels([str(x).replace("meta_urban_", "").replace("_yang2024", "") for x in piv.index])

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=8, color="black")
        return im

    im1 = draw(strong_ok, axes[0], "Strong: Epi x Adaptation (Mean Future RR)")
    im2 = draw(quant_ok, axes[1], "Quant: Epi x Adaptation (Mean Future RR)")

    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Future RR")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_readme(out_dir: Path, files: list[Path]) -> None:
    lines = [
        "# Gorsel Paket",
        "",
        "Bu klasor bulgularin insanlar icin daha okunur grafik ozetini icerir.",
        "",
        "## Dosyalar",
    ]
    for f in files:
        lines.append(f"- {f.name}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    model, strong, quant, lit = load_inputs(args.root_dir)
    strong_ok = clean_ok(strong)
    quant_ok = clean_ok(quant)

    out1 = args.output_dir / "fig01_model_overview.png"
    out2 = args.output_dir / "fig02_rr_distribution.png"
    out3 = args.output_dir / "fig03_scenario_ranges.png"
    out4 = args.output_dir / "fig04_literature_alignment.png"
    out5 = args.output_dir / "fig05_epi_adaptation_heatmap.png"

    plot_model_overview(model, out1)
    plot_rr_distribution(strong_ok, quant_ok, out2)
    plot_scenario_ranges(strong_ok, quant_ok, out3)
    plot_alignment(strong_ok, quant_ok, lit, out4)
    plot_epi_adaptation_heatmap(strong_ok, quant_ok, out5)

    write_readme(args.output_dir, [out1, out2, out3, out4, out5])

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    print(f"Wrote: {out3}")
    print(f"Wrote: {out4}")
    print(f"Wrote: {out5}")
    print(f"Wrote: {args.output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
