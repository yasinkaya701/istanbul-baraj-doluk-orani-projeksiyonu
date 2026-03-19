#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def trim_whitespace(gray: np.ndarray, white_thr: int = 245) -> np.ndarray:
    mask = gray < white_thr
    if not mask.any():
        return gray
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return gray[y0:y1, x0:x1]


def load_and_prep(path: Path, size=(256, 256)) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    arr = trim_whitespace(arr, white_thr=245)
    arr_img = Image.fromarray(arr.astype(np.uint8)).resize(size, Image.Resampling.BICUBIC)
    return np.asarray(arr_img, dtype=np.float32)


def ahash_bits(arr: np.ndarray) -> np.ndarray:
    small = Image.fromarray(arr.astype(np.uint8)).resize((8, 8), Image.Resampling.BICUBIC)
    a = np.asarray(small, dtype=np.float32)
    med = np.median(a)
    return (a > med).astype(np.uint8).flatten()


def ahash_sim(a: np.ndarray, b: np.ndarray) -> float:
    ha = ahash_bits(a)
    hb = ahash_bits(b)
    return 1.0 - np.mean(ha != hb)


def gradient_map(arr: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:] = arr[:, 1:] - arr[:, :-1]
    gy[1:, :] = arr[1:, :] - arr[:-1, :]
    g = np.sqrt(gx * gx + gy * gy)
    return g


def edge_binary(arr: np.ndarray, q: float = 0.85) -> np.ndarray:
    g = gradient_map(arr)
    thr = np.quantile(g, q)
    return (g >= thr).astype(np.uint8)


def edge_iou(a_bin: np.ndarray, b_bin: np.ndarray) -> float:
    inter = np.logical_and(a_bin == 1, b_bin == 1).sum()
    union = np.logical_or(a_bin == 1, b_bin == 1).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def edge_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.ravel().astype(np.float32)
    y = b.ravel().astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den == 0:
        return 0.0
    return float(np.clip(np.dot(x, y) / den, -1, 1))


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    outdir = base / "output" / "analysis" / "graph_similarity"
    target_dir = outdir / "targets"
    sample_dir = base / "output" / "graph_samples_one_per_type"

    targets = sorted(target_dir.glob("*.png"))
    samples = sorted(sample_dir.glob("*.png"))
    if not targets or not samples:
        raise RuntimeError("Targets or samples not found.")

    cache = {}
    edge_cache = {}
    for p in targets + samples:
        arr = load_and_prep(p)
        cache[p] = arr
        edge_cache[p] = edge_binary(arr, q=0.85)

    rows = []
    for t in targets:
        a = cache[t]
        a_edge = edge_cache[t]
        for s in samples:
            b = cache[s]
            b_edge = edge_cache[s]
            iou = edge_iou(a_edge, b_edge)
            ec = edge_corr(gradient_map(a), gradient_map(b))
            ah = ahash_sim(a, b)
            score = 0.50 * iou + 0.30 * ((ec + 1) / 2) + 0.20 * ah
            rows.append(
                {
                    "target": t.name,
                    "sample": s.name,
                    "edge_score": score,
                    "edge_iou": iou,
                    "edge_corr": ec,
                    "ahash_sim": ah,
                }
            )

    df = pd.DataFrame(rows).sort_values(["target", "edge_score"], ascending=[True, False])
    df.to_csv(outdir / "graph_similarity_edge_scores.csv", index=False, float_format="%.6f")

    top = []
    for target, g in df.groupby("target"):
        g2 = g.head(5)
        for rank, (_, r) in enumerate(g2.iterrows(), start=1):
            top.append(
                {
                    "target": target,
                    "rank": rank,
                    "sample": r["sample"],
                    "edge_score": r["edge_score"],
                    "edge_iou": r["edge_iou"],
                    "edge_corr": r["edge_corr"],
                    "ahash_sim": r["ahash_sim"],
                }
            )
    top_df = pd.DataFrame(top)
    top_df.to_csv(outdir / "graph_similarity_edge_top_matches.csv", index=False, float_format="%.6f")

    md = []
    md.append("# Grafik Benzerlik Arastirmasi (Edge-Weighted)")
    md.append("")
    md.append("- Yontem: beyaz bosluk kirpma + gradient edge karsilastirma")
    md.append("- Skor: `0.50*edge_iou + 0.30*edge_corr + 0.20*ahash`")
    md.append("")
    for target, g in top_df.groupby("target"):
        md.append(f"## {target}")
        md.append("")
        md.append("| # | Benzer ornek | Edge skor | IoU |")
        md.append("|---|---|---:|---:|")
        for _, r in g.sort_values("rank").iterrows():
            md.append(f"| {int(r['rank'])} | {r['sample']} | {r['edge_score']:.3f} | {r['edge_iou']:.3f} |")
        md.append("")
    (outdir / "graph_similarity_edge_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Saved: {outdir / 'graph_similarity_edge_scores.csv'}")
    print(f"Saved: {outdir / 'graph_similarity_edge_top_matches.csv'}")
    print(f"Saved: {outdir / 'graph_similarity_edge_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
