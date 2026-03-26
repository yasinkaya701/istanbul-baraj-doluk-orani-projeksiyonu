#!/usr/bin/env python3
from __future__ import annotations

from datetime import date
from pathlib import Path
import html
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    from skimage.metrics import structural_similarity as sk_ssim  # type: ignore
except Exception:  # pragma: no cover
    sk_ssim = None

try:
    from skimage.feature import hog as sk_hog  # type: ignore
except Exception:  # pragma: no cover
    sk_hog = None


def trim_whitespace(gray: np.ndarray, white_thr: int = 245) -> np.ndarray:
    mask = gray < white_thr
    if not mask.any():
        return gray
    ys, xs = np.where(mask)
    return gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def load_prepped(path: Path, size: tuple[int, int] = (320, 320)) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    arr = trim_whitespace(arr, white_thr=245)
    img = Image.fromarray(arr.astype(np.uint8)).resize(size, Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def ahash_bits(arr: np.ndarray) -> np.ndarray:
    small = Image.fromarray(arr.astype(np.uint8)).resize((8, 8), Image.Resampling.BICUBIC)
    a = np.asarray(small, dtype=np.float32)
    med = np.median(a)
    return (a > med).astype(np.uint8).flatten()


def ahash_sim(a: np.ndarray, b: np.ndarray) -> float:
    ha = ahash_bits(a)
    hb = ahash_bits(b)
    return float(1.0 - np.mean(ha != hb))


def hist_corr(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    ha, _ = np.histogram(a.ravel(), bins=bins, range=(0, 255), density=True)
    hb, _ = np.histogram(b.ravel(), bins=bins, range=(0, 255), density=True)
    ha = ha - ha.mean()
    hb = hb - hb.mean()
    den = np.linalg.norm(ha) * np.linalg.norm(hb)
    if den == 0:
        return 0.0
    return float(np.clip(np.dot(ha, hb) / den, -1, 1))


def gradient_map(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:] = a[:, 1:] - a[:, :-1]
    gy[1:, :] = a[1:, :] - a[:-1, :]
    return np.sqrt(gx * gx + gy * gy)


def edge_binary(arr: np.ndarray, q: float = 0.86) -> np.ndarray:
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
    x = gradient_map(a).ravel().astype(np.float32)
    y = gradient_map(b).ravel().astype(np.float32)
    x = x - x.mean()
    y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den == 0:
        return 0.0
    return float(np.clip(np.dot(x, y) / den, -1, 1))


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    if sk_ssim is None:
        return float(((edge_corr(a, b) + 1.0) / 2.0))
    v = float(sk_ssim(a, b, data_range=255))
    return float(np.clip(v, 0.0, 1.0))


def hog_corr(a: np.ndarray, b: np.ndarray) -> float:
    if sk_hog is None:
        return 0.0
    fa = sk_hog(a, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    fb = sk_hog(b, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    fa = fa - fa.mean()
    fb = fb - fb.mean()
    den = np.linalg.norm(fa) * np.linalg.norm(fb)
    if den == 0:
        return 0.0
    return float(np.clip(np.dot(fa, fb) / den, -1.0, 1.0))


def semantic_compat(target_type: str, ref_type: str) -> float:
    table = {
        "hexbin": {"hexbin": 1.00, "scatter": 0.90, "line": 0.45, "bar": 0.35, "box": 0.40, "violin": 0.43, "lag_corr": 0.28},
        "box_violin": {"box": 1.00, "violin": 0.98, "bar": 0.58, "line": 0.54, "scatter": 0.48, "hexbin": 0.44, "lag_corr": 0.32},
        "era_bar": {"bar": 1.00, "line": 0.76, "box": 0.52, "violin": 0.49, "scatter": 0.45, "hexbin": 0.39, "lag_corr": 0.66},
        "monthly_line": {"line": 1.00, "bar": 0.80, "scatter": 0.62, "box": 0.50, "violin": 0.44, "hexbin": 0.39, "lag_corr": 0.50},
        "lag_corr": {"lag_corr": 1.00, "bar": 0.72, "line": 0.66, "scatter": 0.44, "box": 0.35, "violin": 0.30, "hexbin": 0.24},
    }
    return float(table.get(target_type, {}).get(ref_type, 0.38))


def compatibility_penalty(compat: float) -> float:
    if compat >= 0.75:
        return 1.00
    if compat >= 0.55:
        return 0.93
    return 0.86


def confidence_label(top1: float, gap: float) -> str:
    if top1 >= 0.65 and gap >= 0.07:
        return "yuksek"
    if top1 >= 0.58 and gap >= 0.04:
        return "orta_yuksek"
    if top1 >= 0.52 and gap >= 0.02:
        return "orta"
    return "dusuk"


def type_confidence_label(gap_other_type: float) -> str:
    if gap_other_type >= 0.08:
        return "yuksek"
    if gap_other_type >= 0.04:
        return "orta"
    return "dusuk"


def preferred_ref_types(target_type: str) -> set[str]:
    mapping = {
        "hexbin": {"hexbin", "scatter"},
        "box_violin": {"box", "violin"},
        "era_bar": {"bar"},
        "monthly_line": {"line"},
        "lag_corr": {"lag_corr"},
    }
    return mapping.get(target_type, {"line", "bar", "scatter", "box", "violin", "hexbin", "lag_corr"})


def target_catalog(base: Path) -> list[dict[str, str]]:
    return [
        {
            "target_id": "01_humidity_precip_hexbin",
            "target_type": "hexbin",
            "path": str(base / "output" / "analysis" / "nonlinear_viz" / "01_humidity_precip_hexbin.png"),
        },
        {
            "target_id": "02_temp_humidity_seasonal_box",
            "target_type": "box_violin",
            "path": str(base / "output" / "analysis" / "nonlinear_viz" / "02_temp_humidity_seasonal_box.png"),
        },
        {
            "target_id": "03_pressure_precip_era_spearman",
            "target_type": "era_bar",
            "path": str(base / "output" / "analysis" / "nonlinear_viz" / "03_pressure_precip_era_spearman.png"),
        },
        {
            "target_id": "04_mgm_monthly_pattern",
            "target_type": "monthly_line",
            "path": str(base / "output" / "analysis" / "nonlinear_viz" / "04_mgm_monthly_pattern.png"),
        },
        {
            "target_id": "05_lag_correlation",
            "target_type": "lag_corr",
            "path": str(base / "output" / "analysis" / "graph_similarity" / "targets" / "t5_lag_correlation.png"),
        },
    ]


def save_pair_figure(target_path: Path, ref_path: Path, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a = Image.open(target_path)
    b = Image.open(ref_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].imshow(a)
    axes[0].set_title("Hedef")
    axes[0].axis("off")
    axes[1].imshow(b)
    axes[1].set_title("Internet Ref")
    axes[1].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_pair_summary(pair_files: list[Path], out_path: Path) -> None:
    if not pair_files:
        return
    n = len(pair_files)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.8 * rows))
    axes_arr = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes_arr):
        if i >= n:
            ax.axis("off")
            continue
        img = Image.open(pair_files[i])
        ax.imshow(img)
        ax.set_title(pair_files[i].stem, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_html_dashboard(summary: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for _, r in summary.iterrows():
        img_rel = f"pairs/{r['pair_file']}"
        row = (
            "<tr>"
            f"<td>{html.escape(str(r['target_id']))}</td>"
            f"<td>{html.escape(str(r['ref_id']))}</td>"
            f"<td>{html.escape(str(r['target_type']))} ↔ {html.escape(str(r['ref_type']))}</td>"
            f"<td>{float(r['hybrid_score']):.3f}</td>"
            f"<td>{float(r['score_gap_to_second']):.3f}</td>"
            f"<td>{html.escape(str(r['confidence']))} / {html.escape(str(r['type_confidence']))}</td>"
            f"<td><a href=\"{html.escape(str(r['page_url']))}\">kaynak</a></td>"
            f"<td><img src=\"{html.escape(img_rel)}\" style=\"max-width:360px;border:1px solid #ddd\"/></td>"
            "</tr>"
        )
        rows.append(row)

    html_doc = f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8"/>
  <title>Internet Grafik Benzerlik Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 20px; }}
    h1 {{ margin: 0 0 8px 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f7f7f7; text-align: left; }}
  </style>
</head>
<body>
  <h1>Internet Grafik Benzerlik Dashboard</h1>
  <p>Tarih: {date.today().isoformat()} | Lineer regresyon yok.</p>
  <table>
    <thead>
      <tr>
        <th>Hedef</th><th>Ref</th><th>Tip Uyumu</th><th>Hibrit</th><th>Gap</th><th>Guven (tam/tip)</th><th>Kaynak</th><th>Onizleme</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    ref_dir = base / "output" / "analysis" / "internet_refs"
    ref_img_dir = ref_dir / "images"
    manifest_path = ref_dir / "internet_chart_manifest.tsv"

    out_dir = base / "output" / "analysis" / "internet_graph_similarity"
    pair_dir = out_dir / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep="\t")
    manifest["image_path"] = manifest["local_file"].map(lambda x: str(ref_img_dir / str(x)))
    manifest["exists"] = manifest["image_path"].map(lambda x: Path(x).exists())
    manifest = manifest[manifest["exists"]].copy()
    if manifest.empty:
        raise RuntimeError(f"No internet reference images found in {ref_img_dir}")

    tcat = pd.DataFrame(target_catalog(base))
    tcat["exists"] = tcat["path"].map(lambda x: Path(x).exists())
    tcat = tcat[tcat["exists"]].copy()
    if tcat.empty:
        raise RuntimeError("No target images found for similarity.")

    cache: dict[Path, np.ndarray] = {}
    edge_cache: dict[Path, np.ndarray] = {}
    for p in [Path(x) for x in list(manifest["image_path"]) + list(tcat["path"])]:
        arr = load_prepped(p)
        cache[p] = arr
        edge_cache[p] = edge_binary(arr, q=0.86)

    rows = []
    for _, tr in tcat.iterrows():
        tp = Path(tr["path"])
        ta = cache[tp]
        te = edge_cache[tp]
        for _, rr in manifest.iterrows():
            rp = Path(rr["image_path"])
            ra = cache[rp]
            re = edge_cache[rp]

            iou = edge_iou(te, re)
            ec = edge_corr(ta, ra)
            hc = hist_corr(ta, ra)
            ah = ahash_sim(ta, ra)
            ss = ssim_score(ta, ra)
            hg = hog_corr(ta, ra)

            ec_n = (ec + 1.0) / 2.0
            hc_n = (hc + 1.0) / 2.0
            hg_n = (hg + 1.0) / 2.0

            visual = 0.22 * iou + 0.16 * ec_n + 0.14 * hc_n + 0.10 * ah + 0.18 * ss + 0.20 * hg_n
            compat = semantic_compat(str(tr["target_type"]), str(rr["chart_type"]))
            penalty = compatibility_penalty(compat)
            hybrid = (0.68 * visual + 0.32 * compat) * penalty

            rows.append(
                {
                    "target_id": tr["target_id"],
                    "target_type": tr["target_type"],
                    "ref_id": rr["ref_id"],
                    "provider": rr["provider"],
                    "ref_type": rr["chart_type"],
                    "title": rr["title"],
                    "page_url": rr["page_url"],
                    "image_url": rr["image_url"],
                    "image_path": rr["image_path"],
                    "edge_iou": iou,
                    "edge_corr": ec,
                    "hist_corr": hc,
                    "ahash_sim": ah,
                    "ssim": ss,
                    "hog_corr": hg,
                    "visual_score": visual,
                    "semantic_compat": compat,
                    "compat_penalty": penalty,
                    "hybrid_score": hybrid,
                }
            )

    scores = pd.DataFrame(rows).sort_values(["target_id", "hybrid_score"], ascending=[True, False]).reset_index(drop=True)
    scores_path = out_dir / "internet_graph_similarity_scores.csv"
    scores.to_csv(scores_path, index=False, float_format="%.6f")

    top_rows = []
    summary_rows = []
    family_rows = []
    for target_id, g in scores.groupby("target_id", sort=False):
        g2 = g.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        for rank, (_, r) in enumerate(g2.head(5).iterrows(), start=1):
            d = r.to_dict()
            d["rank"] = rank
            top_rows.append(d)

        first = g2.iloc[0]
        second = g2.iloc[1] if len(g2) > 1 else g2.iloc[0]
        gap = float(first["hybrid_score"] - second["hybrid_score"])
        conf = confidence_label(float(first["hybrid_score"]), gap)
        others = g2[g2["ref_type"].astype(str) != str(first["ref_type"])]
        best_other_type = float(others["hybrid_score"].max()) if len(others) > 0 else float(second["hybrid_score"])
        gap_other_type = float(first["hybrid_score"] - best_other_type)
        conf_type = type_confidence_label(gap_other_type)
        summary_rows.append(
            {
                "target_id": first["target_id"],
                "target_type": first["target_type"],
                "ref_id": first["ref_id"],
                "provider": first["provider"],
                "ref_type": first["ref_type"],
                "title": first["title"],
                "hybrid_score": first["hybrid_score"],
                "visual_score": first["visual_score"],
                "semantic_compat": first["semantic_compat"],
                "compat_penalty": first["compat_penalty"],
                "score_second_best": float(second["hybrid_score"]),
                "score_gap_to_second": gap,
                "confidence": conf,
                "score_best_other_type": best_other_type,
                "score_gap_to_other_type": gap_other_type,
                "type_confidence": conf_type,
                "page_url": first["page_url"],
                "image_url": first["image_url"],
                "image_path": first["image_path"],
            }
        )

        pref_types = preferred_ref_types(str(first["target_type"]))
        gfam = g2[g2["ref_type"].astype(str).isin(pref_types)].reset_index(drop=True)
        if len(gfam) == 0:
            gfam = g2.copy()
        fam_first = gfam.iloc[0]
        fam_second = gfam.iloc[1] if len(gfam) > 1 else fam_first
        fam_gap = float(fam_first["hybrid_score"] - fam_second["hybrid_score"])
        fam_conf = confidence_label(float(fam_first["hybrid_score"]), fam_gap)
        family_rows.append(
            {
                "target_id": fam_first["target_id"],
                "target_type": fam_first["target_type"],
                "preferred_ref_types": ",".join(sorted(pref_types)),
                "ref_id": fam_first["ref_id"],
                "provider": fam_first["provider"],
                "ref_type": fam_first["ref_type"],
                "title": fam_first["title"],
                "hybrid_score": fam_first["hybrid_score"],
                "visual_score": fam_first["visual_score"],
                "semantic_compat": fam_first["semantic_compat"],
                "score_second_in_family": float(fam_second["hybrid_score"]),
                "score_gap_in_family": fam_gap,
                "family_confidence": fam_conf,
                "page_url": fam_first["page_url"],
                "image_url": fam_first["image_url"],
                "image_path": fam_first["image_path"],
            }
        )

    top = pd.DataFrame(top_rows)
    top_path = out_dir / "internet_graph_similarity_top5.csv"
    top.to_csv(top_path, index=False, float_format="%.6f")

    summary = pd.DataFrame(summary_rows).sort_values("target_id").reset_index(drop=True)
    summary_path = out_dir / "internet_graph_similarity_best_match.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    family_summary = pd.DataFrame(family_rows).sort_values("target_id").reset_index(drop=True)
    family_summary_path = out_dir / "internet_graph_similarity_family_best_match.csv"
    family_summary.to_csv(family_summary_path, index=False, float_format="%.6f")

    conf_path = out_dir / "internet_graph_similarity_confidence.csv"
    summary.loc[
        :,
        [
            "target_id",
            "ref_id",
            "hybrid_score",
            "score_second_best",
            "score_gap_to_second",
            "confidence",
            "score_best_other_type",
            "score_gap_to_other_type",
            "type_confidence",
        ],
    ].to_csv(
        conf_path, index=False, float_format="%.6f"
    )

    pair_files: list[Path] = []
    pair_names: list[str] = []
    for _, r in summary.iterrows():
        tpath = Path(tcat.loc[tcat["target_id"] == r["target_id"], "path"].iloc[0])
        rpath = Path(r["image_path"])
        out_pair = pair_dir / f"{r['target_id']}_vs_{r['ref_id']}.png"
        title = (
            f"{r['target_id']} vs {r['ref_id']} | hibrit={float(r['hybrid_score']):.3f} "
            f"| gap={float(r['score_gap_to_second']):.3f} | guven={r['confidence']}/{r['type_confidence']}"
        )
        save_pair_figure(tpath, rpath, out_pair, title)
        pair_files.append(out_pair)
        pair_names.append(out_pair.name)

    summary["pair_file"] = pair_names
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    summary_img = pair_dir / "internet_similarity_pair_summary.png"
    build_pair_summary(pair_files, summary_img)

    family_pair_dir = out_dir / "pairs_family"
    family_pair_dir.mkdir(parents=True, exist_ok=True)
    family_pair_files: list[Path] = []
    family_pair_names: list[str] = []
    for _, r in family_summary.iterrows():
        tpath = Path(tcat.loc[tcat["target_id"] == r["target_id"], "path"].iloc[0])
        rpath = Path(r["image_path"])
        out_pair = family_pair_dir / f"{r['target_id']}_family_vs_{r['ref_id']}.png"
        title = (
            f"{r['target_id']} family-best vs {r['ref_id']} | hibrit={float(r['hybrid_score']):.3f} "
            f"| fam_gap={float(r['score_gap_in_family']):.3f} | fam_guven={r['family_confidence']}"
        )
        save_pair_figure(tpath, rpath, out_pair, title)
        family_pair_files.append(out_pair)
        family_pair_names.append(out_pair.name)
    family_summary["pair_file"] = family_pair_names
    family_summary.to_csv(family_summary_path, index=False, float_format="%.6f")
    family_summary_img = family_pair_dir / "internet_similarity_family_pair_summary.png"
    build_pair_summary(family_pair_files, family_summary_img)

    dashboard_path = out_dir / "internet_graph_similarity_dashboard.html"
    build_html_dashboard(summary, dashboard_path)

    md = []
    md.append("# Internet Tabanli Grafik Benzerlik Raporu")
    md.append("")
    md.append(f"- Tarih: {date.today().isoformat()}")
    md.append("- Not: Lineer regresyon kullanilmadi.")
    md.append("- Gorsel skor: edge IoU + edge corr + hist corr + aHash + SSIM + HOG")
    md.append("- Hibrit skor: `(0.68*visual + 0.32*semantic) * compat_penalty`")
    md.append("")
    md.append("## En Iyi Eslesmeler (Top-1)")
    md.append("")
    md.append("| Hedef | En Benzer Internet Grafigi | Tur Uyumu | Hibrit | Gap | Guven (tam/tip) | Kaynak |")
    md.append("|---|---|---|---:|---:|---|---|")
    for _, r in summary.iterrows():
        md.append(
            f"| {r['target_id']} | {r['title']} (`{r['ref_id']}`) | {r['target_type']}↔{r['ref_type']} | "
            f"{float(r['hybrid_score']):.3f} | {float(r['score_gap_to_second']):.3f} | {r['confidence']}/{r['type_confidence']} | {r['page_url']} |"
        )
    md.append("")
    md.append("## Aile-Odakli En Iyi Eslesmeler (Onerilen)")
    md.append("")
    md.append("| Hedef | Aile | En Iyi Ref | Hibrit | Aile Ici Gap | Aile Guven | Kaynak |")
    md.append("|---|---|---|---:|---:|---|---|")
    for _, r in family_summary.iterrows():
        md.append(
            f"| {r['target_id']} | {r['preferred_ref_types']} | {r['title']} (`{r['ref_id']}`) | "
            f"{float(r['hybrid_score']):.3f} | {float(r['score_gap_in_family']):.3f} | {r['family_confidence']} | {r['page_url']} |"
        )
    md.append("")
    md.append("## Hedef Bazli Top-5")
    md.append("")
    for target_id, g in top.groupby("target_id", sort=False):
        md.append(f"### {target_id}")
        md.append("")
        md.append("| # | Ref | Tur | Hibrit | Visual | Semantik | Penalty | SSIM | HOG corr |")
        md.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for _, r in g.sort_values("rank").iterrows():
            md.append(
                f"| {int(r['rank'])} | {r['ref_id']} | {r['ref_type']} | {float(r['hybrid_score']):.3f} | "
                f"{float(r['visual_score']):.3f} | {float(r['semantic_compat']):.2f} | {float(r['compat_penalty']):.2f} | "
                f"{float(r['ssim']):.3f} | {float(r['hog_corr']):.3f} |"
            )
        md.append("")
    md.append("## Cikti Dosyalari")
    md.append("")
    md.append(f"- Skorlar: `{scores_path}`")
    md.append(f"- Top-5: `{top_path}`")
    md.append(f"- Top-1 ozet: `{summary_path}`")
    md.append(f"- Aile-odakli ozet: `{family_summary_path}`")
    md.append(f"- Guven tablosu: `{conf_path}`")
    md.append(f"- HTML dashboard: `{dashboard_path}`")
    md.append(f"- Cift gorseller: `{pair_dir}`")
    md.append(f"- Aile cift gorseller: `{family_pair_dir}`")

    report_path = out_dir / "internet_graph_similarity_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Saved: {scores_path}")
    print(f"Saved: {top_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {family_summary_path}")
    print(f"Saved: {conf_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {dashboard_path}")
    print(f"Saved: {summary_img}")
    print(f"Saved: {family_summary_img}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
