#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def load_gray(path: Path, size=(256, 256)) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size, Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32)


def ahash_bits(arr: np.ndarray) -> np.ndarray:
    small = Image.fromarray(arr.astype(np.uint8)).resize((8, 8), Image.Resampling.BICUBIC)
    a = np.asarray(small, dtype=np.float32)
    med = np.median(a)
    return (a > med).astype(np.uint8).flatten()


def phash_sim(a: np.ndarray, b: np.ndarray) -> float:
    ha = ahash_bits(a)
    hb = ahash_bits(b)
    return 1.0 - np.mean(ha != hb)


def hist_corr(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    ha, _ = np.histogram(a.ravel(), bins=bins, range=(0, 255), density=True)
    hb, _ = np.histogram(b.ravel(), bins=bins, range=(0, 255), density=True)
    ha = ha - ha.mean()
    hb = hb - hb.mean()
    den = np.linalg.norm(ha) * np.linalg.norm(hb)
    if den == 0:
        return 0.0
    return float(np.clip(np.dot(ha, hb) / den, -1, 1))


def pixel_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.ravel()
    y = b.ravel()
    x = x - x.mean()
    y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den == 0:
        return 0.0
    return float(np.clip(np.dot(x, y) / den, -1, 1))


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    outdir = base / "output" / "analysis" / "graph_similarity"
    target_dir = outdir / "targets"
    sample_dir = base / "output" / "graph_samples_one_per_type"

    outdir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    targets = sorted(target_dir.glob("*.png"))
    samples = sorted(sample_dir.glob("*.png"))

    if not targets:
        raise RuntimeError(f"No target images found in {target_dir}")
    if not samples:
        raise RuntimeError(f"No sample images found in {sample_dir}")

    cache = {}
    for p in targets + samples:
        cache[p] = load_gray(p)

    rows = []
    for t in targets:
        a = cache[t]
        for s in samples:
            b = cache[s]
            hs = hist_corr(a, b)
            ps = phash_sim(a, b)
            pc = pixel_corr(a, b)
            score = 0.45 * ((pc + 1) / 2) + 0.35 * ((hs + 1) / 2) + 0.20 * ps
            rows.append(
                {
                    "target": t.name,
                    "sample": s.name,
                    "combined_score": score,
                    "pixel_corr": pc,
                    "hist_corr": hs,
                    "phash_sim": ps,
                }
            )

    sim_df = pd.DataFrame(rows).sort_values(["target", "combined_score"], ascending=[True, False])
    sim_path = outdir / "graph_similarity_scores.csv"
    sim_df.to_csv(sim_path, index=False, float_format="%.6f")

    best = []
    for target, g in sim_df.groupby("target"):
        for rank, (_, r) in enumerate(g.head(5).iterrows(), start=1):
            best.append(
                {
                    "target": target,
                    "rank": rank,
                    "sample": r["sample"],
                    "combined_score": r["combined_score"],
                    "pixel_corr": r["pixel_corr"],
                    "hist_corr": r["hist_corr"],
                    "phash_sim": r["phash_sim"],
                }
            )
    best_df = pd.DataFrame(best)
    best_path = outdir / "graph_similarity_top_matches.csv"
    best_df.to_csv(best_path, index=False, float_format="%.6f")

    sem_map = {
        "t1_humidity_precip_scatter.png": ["correlation_heatmap.png", "diagnostics.png", "monthly_chart.png"],
        "t2_temp_humidity_scatter.png": ["correlation_heatmap.png", "annual_compare.png", "monthly_chart.png"],
        "t3_monthly_anomaly_lines.png": ["monthly_chart.png", "annual_trends.png", "walkforward.png"],
        "t4_pressure_precip_era_bar.png": ["regime_probs.png", "annual_compare.png", "diagnostics.png"],
        "t5_lag_correlation.png": ["diagnostics.png", "regime_probs.png", "model_components.png"],
    }

    lines = []
    lines.append("# Grafik Benzerlik Arastirmasi")
    lines.append("")
    lines.append("- Karsilastirma: hedef grafikler vs `output/graph_samples_one_per_type/*.png`")
    lines.append("- Puan: `0.45*pixel_corr + 0.35*hist_corr + 0.20*pHash` (hepsi normalize)")
    lines.append("")
    for target, g in best_df.groupby("target"):
        lines.append(f"## {target}")
        lines.append("")
        lines.append("| # | Benzer ornek | Puan | pixel_corr |")
        lines.append("|---|---|---:|---:|")
        for _, r in g.sort_values("rank").iterrows():
            lines.append(f"| {int(r['rank'])} | {r['sample']} | {r['combined_score']:.3f} | {r['pixel_corr']:.3f} |")
        lines.append("")

    lines.append("## Semantik Benzerlik (kullanim amaci)")
    lines.append("")
    for key, vals in sem_map.items():
        pretty = ", ".join([f"`{x}`" for x in vals])
        lines.append(f"- `{key}` -> {pretty}")

    report_path = outdir / "graph_similarity_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {sim_path}")
    print(f"Saved: {best_path}")
    print(f"Saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
