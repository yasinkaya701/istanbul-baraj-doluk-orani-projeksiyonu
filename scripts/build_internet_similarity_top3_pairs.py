#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def target_map(base: Path) -> dict[str, Path]:
    return {
        "01_humidity_precip_hexbin": base / "output" / "analysis" / "nonlinear_viz" / "01_humidity_precip_hexbin.png",
        "02_temp_humidity_seasonal_box": base / "output" / "analysis" / "nonlinear_viz" / "02_temp_humidity_seasonal_box.png",
        "03_pressure_precip_era_spearman": base / "output" / "analysis" / "nonlinear_viz" / "03_pressure_precip_era_spearman.png",
        "04_mgm_monthly_pattern": base / "output" / "analysis" / "nonlinear_viz" / "04_mgm_monthly_pattern.png",
        "05_lag_correlation": base / "output" / "analysis" / "graph_similarity" / "targets" / "t5_lag_correlation.png",
    }


def render_pair(target_path: Path, ref_path: Path, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    axes[0].imshow(Image.open(target_path))
    axes[0].set_title("Hedef")
    axes[0].axis("off")
    axes[1].imshow(Image.open(ref_path))
    axes[1].set_title("Internet Ref")
    axes[1].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def build_sheet(images: list[Path], out_path: Path, cols: int = 3) -> None:
    n = len(images)
    if n == 0:
        return
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2))
    axes_arr = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes_arr):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(Image.open(images[i]))
        ax.set_title(images[i].stem, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    sim_dir = base / "output" / "analysis" / "internet_graph_similarity"
    top = pd.read_csv(sim_dir / "internet_graph_similarity_top5.csv")
    tmap = target_map(base)

    out_dir = sim_dir / "pairs_top3"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs: list[Path] = []
    for target_id, g in top.groupby("target_id", sort=False):
        g3 = g.sort_values("rank").head(3)
        tpath = tmap.get(str(target_id))
        if tpath is None or not tpath.exists():
            continue
        this_target: list[Path] = []
        for _, r in g3.iterrows():
            rpath = Path(str(r["image_path"]))
            if not rpath.exists():
                continue
            out = out_dir / f"{target_id}_r{int(r['rank'])}_{r['ref_id']}.png"
            title = f"{target_id} | rank={int(r['rank'])} | ref={r['ref_id']} | hibrit={float(r['hybrid_score']):.3f}"
            render_pair(tpath, rpath, out, title)
            this_target.append(out)
            all_pairs.append(out)
        build_sheet(this_target, out_dir / f"{target_id}_top3_sheet.png", cols=3)

    build_sheet(all_pairs, out_dir / "all_targets_top3_sheet.png", cols=3)
    print(f"Saved top3 pair visuals in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
