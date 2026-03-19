#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


TARGET_LABELS: Dict[str, str] = {
    "01_humidity_precip_hexbin": "Nem-Yagis",
    "02_temp_humidity_seasonal_box": "Sicaklik-Nem",
    "03_pressure_precip_era_spearman": "Basinc-Yagis(era)",
    "04_mgm_monthly_pattern": "Aylik Patern",
    "05_lag_correlation": "Lag Korelasyon",
}

TARGET_COLORS: Dict[str, str] = {
    "01_humidity_precip_hexbin": "#1d4ed8",
    "02_temp_humidity_seasonal_box": "#0f766e",
    "03_pressure_precip_era_spearman": "#b45309",
    "04_mgm_monthly_pattern": "#7c3aed",
    "05_lag_correlation": "#be123c",
}


def safe_read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)


def chart_feature(gray: np.ndarray) -> np.ndarray:
    # Edge + orientation + intensity descriptors for chart morphology.
    edges = cv2.Canny(gray, 80, 180)
    edge = edges.astype(np.float32) / 255.0
    edge_density = np.array([edge.mean()], dtype=np.float32)

    row_prof = edge.mean(axis=1)
    col_prof = edge.mean(axis=0)
    row_b = row_prof.reshape(16, 16).mean(axis=1)
    col_b = col_prof.reshape(16, 16).mean(axis=1)

    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel().astype(np.float32)
    hist /= np.maximum(hist.sum(), 1e-9)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    bins = np.linspace(0, 2 * np.pi, 9)
    ori_hist, _ = np.histogram(ang, bins=bins, weights=mag)
    ori_hist = ori_hist.astype(np.float32)
    ori_hist /= np.maximum(ori_hist.sum(), 1e-9)

    return np.concatenate([edge_density, row_b, col_b, hist, ori_hist], axis=0)


def choose_cluster_count(x: np.ndarray, max_k: int = 8) -> int:
    n = x.shape[0]
    if n < 6:
        return 2
    k_values = list(range(3, min(max_k, n - 1) + 1))
    best_k = 3
    best_score = -1.0
    for k in k_values:
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(x, labels)
        if s > best_score:
            best_score = s
            best_k = k
    return best_k


def reassign_small_clusters(x: np.ndarray, labels: np.ndarray, min_size: int = 2) -> np.ndarray:
    labels = labels.astype(int).copy()
    counts = pd.Series(labels).value_counts().to_dict()
    stable_clusters = [cid for cid, n in counts.items() if int(n) >= int(min_size)]
    if not stable_clusters:
        return labels

    centroids: Dict[int, np.ndarray] = {}
    for cid in stable_clusters:
        centroids[int(cid)] = x[labels == int(cid)].mean(axis=0)

    for i, cid in enumerate(labels.tolist()):
        if int(counts.get(int(cid), 0)) >= int(min_size):
            continue
        v = x[i : i + 1]
        best_c = stable_clusters[0]
        best_s = -1.0
        for scid in stable_clusters:
            s = float(cosine_similarity(v, centroids[int(scid)][None, :])[0, 0])
            if s > best_s:
                best_s = s
                best_c = int(scid)
        labels[i] = int(best_c)

    # Normalize to contiguous cluster ids.
    uniq = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(uniq)}
    labels = np.array([remap[int(v)] for v in labels.tolist()], dtype=int)
    return labels


def build_feature_matrix(ref_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    valid_rows: List[int] = []

    for i, r in ref_df.iterrows():
        p = Path(str(r["image_path"]))
        if not p.exists():
            continue
        g = safe_read_gray(p)
        feats.append(chart_feature(g))
        valid_rows.append(i)

    if not feats:
        raise RuntimeError("No reference images found for archetype analysis.")

    feat_arr = np.vstack(feats)
    feat_arr = (feat_arr - feat_arr.mean(axis=0, keepdims=True)) / (feat_arr.std(axis=0, keepdims=True) + 1e-9)
    sim = cosine_similarity(feat_arr)
    sim = np.clip(sim, -1.0, 1.0)
    return np.array(valid_rows, dtype=int), feat_arr, sim


def cluster_label_from_types(types: pd.Series, cid: int) -> str:
    shares = types.value_counts(normalize=True)
    top = shares.index.tolist()[:2]
    if len(top) >= 2 and shares.iloc[1] >= 0.22:
        return f"K{cid}_{top[0]}+{top[1]}"
    return f"K{cid}_{top[0]}"


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    out_dir = base / "output" / "analysis" / "internet_graph_similarity"

    scores = pd.read_csv(out_dir / "internet_graph_similarity_scores.csv")
    consensus_v2 = pd.read_csv(out_dir / "internet_graph_similarity_consensus_v2_scores.csv")

    ref_df = (
        scores[["ref_id", "title", "provider", "ref_type", "page_url", "image_path"]]
        .drop_duplicates("ref_id")
        .reset_index(drop=True)
    )

    valid_idx, feat_arr, sim = build_feature_matrix(ref_df)
    ref_valid = ref_df.iloc[valid_idx].reset_index(drop=True)

    # Cluster directly on morphology features.
    k = choose_cluster_count(feat_arr, max_k=8)
    cluster = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(feat_arr)
    cluster = reassign_small_clusters(feat_arr, cluster, min_size=2)
    ref_valid["cluster_id"] = cluster.astype(int)
    final_k = int(ref_valid["cluster_id"].nunique())

    cluster_label_map: Dict[int, str] = {}
    for cid, g in ref_valid.groupby("cluster_id"):
        cluster_label_map[int(cid)] = cluster_label_from_types(g["ref_type"], int(cid))
    ref_valid["cluster_label"] = ref_valid["cluster_id"].map(cluster_label_map)

    # Intra-cluster similarity and size.
    sizes = ref_valid["cluster_id"].value_counts().to_dict()
    intra_vals = []
    for i, row in ref_valid.iterrows():
        cid = int(row["cluster_id"])
        idxs = np.where(ref_valid["cluster_id"].to_numpy(dtype=int) == cid)[0]
        if len(idxs) <= 1:
            intra_vals.append(1.0)
        else:
            vals = sim[i, idxs]
            intra_vals.append(float(np.mean(vals[vals < 0.999999])))
    ref_valid["cluster_size"] = ref_valid["cluster_id"].map(lambda x: int(sizes.get(int(x), 1)))
    ref_valid["intra_cluster_similarity"] = intra_vals

    # Pairwise table.
    pair_rows = []
    for i in range(len(ref_valid)):
        for j in range(i + 1, len(ref_valid)):
            pair_rows.append(
                {
                    "ref_id_a": ref_valid.iloc[i]["ref_id"],
                    "ref_id_b": ref_valid.iloc[j]["ref_id"],
                    "sim": float(sim[i, j]),
                    "same_cluster": int(ref_valid.iloc[i]["cluster_id"] == ref_valid.iloc[j]["cluster_id"]),
                }
            )
    pair_df = pd.DataFrame(pair_rows).sort_values("sim", ascending=False).reset_index(drop=True)

    # Target -> cluster affinity with robust score (prevents tiny-cluster overfit).
    score_merge = consensus_v2.merge(
        ref_valid[["ref_id", "cluster_id", "cluster_label", "cluster_size"]], on="ref_id", how="inner"
    )

    aff = (
        score_merge.groupby(["target_id", "cluster_id", "cluster_label"], as_index=False)
        .agg(
            cluster_mean=("consensus_score", "mean"),
            cluster_median=("consensus_score", "median"),
            cluster_best=("consensus_score", "max"),
            n_refs=("ref_id", "nunique"),
        )
        .sort_values(["target_id", "cluster_mean"], ascending=[True, False])
    )

    max_refs = max(int(aff["n_refs"].max()), 1)
    aff["size_norm"] = np.log1p(aff["n_refs"].astype(float)) / np.log1p(float(max_refs))
    aff["cluster_affinity"] = (
        0.50 * aff["cluster_mean"]
        + 0.35 * aff["cluster_best"]
        + 0.10 * aff["cluster_median"]
        + 0.05 * aff["size_norm"]
    )
    aff["target_label"] = aff["target_id"].map(TARGET_LABELS).fillna(aff["target_id"])

    best_cluster = aff.sort_values(["target_id", "cluster_affinity"], ascending=[True, False]).groupby(
        "target_id", as_index=False
    ).first()

    rec_rows = []
    for _, r in best_cluster.iterrows():
        t = r["target_id"]
        cid = int(r["cluster_id"])
        cand = score_merge[(score_merge["target_id"] == t) & (score_merge["cluster_id"] == cid)].copy()
        cand = cand.sort_values("consensus_score", ascending=False).head(3)
        rec_rows.append(
            {
                "target_id": t,
                "target_label": TARGET_LABELS.get(str(t), str(t)),
                "best_cluster_id": cid,
                "best_cluster_label": r["cluster_label"],
                "cluster_affinity": float(r["cluster_affinity"]),
                "cluster_mean": float(r["cluster_mean"]),
                "cluster_best": float(r["cluster_best"]),
                "cluster_size": int(r["n_refs"]),
                "cluster_best_ref": str(cand.iloc[0]["ref_id"]) if len(cand) else "",
                "cluster_best_score": float(cand.iloc[0]["consensus_score"]) if len(cand) else np.nan,
                "top3_refs_in_cluster": ", ".join(cand["ref_id"].astype(str).tolist()),
            }
        )
    rec_df = pd.DataFrame(rec_rows).sort_values("target_id").reset_index(drop=True)

    # Save tables.
    ref_out = out_dir / "internet_graph_similarity_archetype_ref_clusters.csv"
    pair_out = out_dir / "internet_graph_similarity_archetype_pairwise_similarity.csv"
    aff_out = out_dir / "internet_graph_similarity_archetype_target_cluster_affinity.csv"
    rec_out = out_dir / "internet_graph_similarity_archetype_target_recommendation.csv"

    ref_valid.to_csv(ref_out, index=False, float_format="%.6f")
    pair_df.to_csv(pair_out, index=False, float_format="%.6f")
    aff.to_csv(aff_out, index=False, float_format="%.6f")
    rec_df.to_csv(rec_out, index=False, float_format="%.6f")

    # Visual 1: Reference similarity heatmap (cluster sorted).
    order = ref_valid.sort_values(["cluster_id", "ref_type", "ref_id"]).index.to_numpy()
    sim_ord = sim[np.ix_(order, order)]
    labels = ref_valid.loc[order, "ref_id"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(12.5, 10))
    sns.heatmap(sim_ord, cmap="YlGnBu", vmin=0.0, vmax=1.0, ax=ax, cbar_kws={"label": "Benzesim"})
    ax.set_title("Internet Referans Grafikleri Arasi Benzesim Isi Haritasi")
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, rotation=0, fontsize=7)
    fig.tight_layout()
    heat_out = out_dir / "internet_graph_similarity_archetype_ref_heatmap.png"
    fig.savefig(heat_out, dpi=190)
    plt.close(fig)

    # Visual 2: Archetype network graph.
    g = nx.Graph()
    for _, r in ref_valid.iterrows():
        g.add_node(r["ref_id"], cluster=int(r["cluster_id"]), ref_type=str(r["ref_type"]))

    threshold = float(np.quantile(pair_df["sim"].to_numpy(dtype=float), 0.90)) if len(pair_df) else 0.7
    for _, r in pair_df.iterrows():
        if float(r["sim"]) >= threshold:
            g.add_edge(str(r["ref_id_a"]), str(r["ref_id_b"]), weight=float(r["sim"]))

    if g.number_of_edges() == 0:
        for _, r in pair_df.head(20).iterrows():
            g.add_edge(str(r["ref_id_a"]), str(r["ref_id_b"]), weight=float(r["sim"]))

    pos = nx.spring_layout(g, seed=42, k=0.85, iterations=220)

    unique_clusters = sorted(ref_valid["cluster_id"].unique().tolist())
    palette = sns.color_palette("Set2", n_colors=max(3, len(unique_clusters)))
    c2color = {cid: palette[i % len(palette)] for i, cid in enumerate(unique_clusters)}

    fig, ax = plt.subplots(figsize=(12, 8))
    edge_w = [1.0 + 2.2 * (d.get("weight", threshold) - threshold + 0.05) for _, _, d in g.edges(data=True)]
    nx.draw_networkx_edges(g, pos, alpha=0.25, width=edge_w, edge_color="#475569", ax=ax)

    for cid in unique_clusters:
        nodes = [n for n, d in g.nodes(data=True) if int(d.get("cluster", -1)) == int(cid)]
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=nodes,
            node_color=[c2color[cid]],
            node_size=460,
            edgecolors="#0f172a",
            linewidths=0.6,
            alpha=0.92,
            ax=ax,
        )

    nx.draw_networkx_labels(g, pos, font_size=7.3, font_color="#0f172a", ax=ax)
    ax.set_title("Internet Grafik Arsetip Agi (Yuksek Benzesim Kenarlari)")
    ax.axis("off")

    legend_handles = [
        Patch(facecolor=c2color[cid], edgecolor="#0f172a", label=cluster_label_map[cid]) for cid in unique_clusters
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    net_out = out_dir / "internet_graph_similarity_archetype_network.png"
    fig.savefig(net_out, dpi=190)
    plt.close(fig)

    # Visual 3: Target-cluster robust affinity heatmap.
    pivot = (
        aff.pivot_table(index="target_label", columns="cluster_label", values="cluster_affinity", aggfunc="mean")
        .fillna(0.0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.heatmap(pivot, cmap="magma", vmin=0.0, vmax=1.0, annot=True, fmt=".2f", linewidths=0.4, ax=ax)
    ax.set_title("Hedef Grafik - Arsetip Kume Robust Yakinlik Isi Haritasi")
    ax.set_xlabel("Arsetip Kumeleri")
    ax.set_ylabel("Hedef Grafikler")
    fig.tight_layout()
    taff_out = out_dir / "internet_graph_similarity_archetype_target_affinity_heatmap.png"
    fig.savefig(taff_out, dpi=190)
    plt.close(fig)

    # Visual 4: 2D embedding for references + target anchors.
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(feat_arr)
    ref_valid["x"] = xy[:, 0]
    ref_valid["y"] = xy[:, 1]

    fig, ax = plt.subplots(figsize=(11.8, 7.6))
    for cid in unique_clusters:
        gref = ref_valid[ref_valid["cluster_id"] == cid]
        ax.scatter(
            gref["x"],
            gref["y"],
            s=90,
            alpha=0.82,
            c=[c2color[cid]],
            edgecolors="#111827",
            linewidths=0.4,
            label=cluster_label_map[cid],
        )
        for _, rr in gref.iterrows():
            ax.text(float(rr["x"]) + 0.02, float(rr["y"]) + 0.02, str(rr["ref_id"]), fontsize=6.7, alpha=0.88)

    top3 = (
        consensus_v2.sort_values(["target_id", "consensus_score"], ascending=[True, False])
        .groupby("target_id", as_index=False)
        .head(3)
    )

    for tid, tg in top3.groupby("target_id"):
        gg = tg.merge(ref_valid[["ref_id", "x", "y"]], on="ref_id", how="inner")
        if gg.empty:
            continue
        w = gg["consensus_score"].to_numpy(dtype=float)
        w = w / np.maximum(w.sum(), 1e-9)
        tx = float(np.sum(gg["x"].to_numpy(dtype=float) * w))
        ty = float(np.sum(gg["y"].to_numpy(dtype=float) * w))
        c = TARGET_COLORS.get(str(tid), "#111827")
        ax.scatter([tx], [ty], s=420, marker="*", c=[c], edgecolors="#111827", linewidths=0.8, zorder=6)
        ax.text(tx + 0.04, ty + 0.04, TARGET_LABELS.get(str(tid), str(tid)), fontsize=9, weight="bold", color=c)

    ax.set_title("Internet Grafik Arsetip Gomumu (PCA) + Hedef Capa Noktalari")
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    emb_out = out_dir / "internet_graph_similarity_archetype_embedding_targets.png"
    fig.savefig(emb_out, dpi=190)
    plt.close(fig)

    # Visual 5: Top pairwise similar internet charts.
    top_pairs = pair_df.head(12).copy()
    top_pairs["pair"] = top_pairs["ref_id_a"].astype(str) + " ~ " + top_pairs["ref_id_b"].astype(str)
    top_pairs = top_pairs.sort_values("sim", ascending=True)

    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    colors = np.where(top_pairs["same_cluster"].to_numpy(dtype=int) == 1, "#2563eb", "#94a3b8")
    ax.barh(top_pairs["pair"], top_pairs["sim"], color=colors)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Benzesim")
    ax.set_title("Top-12 Internet Grafik Cifti (Morfolojik Benzesim)")
    ax.grid(axis="x", alpha=0.25)
    for i, v in enumerate(top_pairs["sim"].to_numpy(dtype=float)):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    top_pairs_out = out_dir / "internet_graph_similarity_archetype_top_pairs.png"
    fig.savefig(top_pairs_out, dpi=190)
    plt.close(fig)

    # Markdown report.
    lines: List[str] = []
    lines.append("# Internet Grafik Arsetip Haritasi Raporu (TR)")
    lines.append("")
    lines.append("- Bu adim, internet referans grafiklerini morfolojik benzesime gore otomatik kumeledi.")
    lines.append(f"- Baslangic kume sayisi (silhouette): **{k}**")
    lines.append(f"- Min-boyut duzeltmesi sonrasi efektif kume sayisi: **{final_k}**")
    lines.append("- Kume secimi robust yakinlik ile yapildi: `0.50*mean + 0.35*best + 0.10*median + 0.05*size_norm`.")
    lines.append("- Bu formul, kucuk ama tesadufi yuksek ortalamali kumelerin asiri one cikmasini engeller.")
    lines.append("")
    lines.append("## Hedef Bazli Arsetip Ozet")
    lines.append("")
    lines.append("| Hedef | En Iyi Kume | Robust Kume Skoru | Kume Boyutu | Kume Icindeki Top3 |")
    lines.append("|---|---|---:|---:|---|")
    for _, r in rec_df.iterrows():
        lines.append(
            f"| {r['target_label']} | {r['best_cluster_label']} | {float(r['cluster_affinity']):.3f} | {int(r['cluster_size'])} | {r['top3_refs_in_cluster']} |"
        )

    lines.append("")
    lines.append("## Cikti Dosyalari")
    lines.append("")
    lines.append(f"- `{ref_out}`")
    lines.append(f"- `{pair_out}`")
    lines.append(f"- `{aff_out}`")
    lines.append(f"- `{rec_out}`")
    lines.append(f"- `{heat_out}`")
    lines.append(f"- `{net_out}`")
    lines.append(f"- `{taff_out}`")
    lines.append(f"- `{emb_out}`")
    lines.append(f"- `{top_pairs_out}`")

    md_out = out_dir / "internet_graph_similarity_archetype_report_tr.md"
    md_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {ref_out}")
    print(f"Saved: {pair_out}")
    print(f"Saved: {aff_out}")
    print(f"Saved: {rec_out}")
    print(f"Saved: {heat_out}")
    print(f"Saved: {net_out}")
    print(f"Saved: {taff_out}")
    print(f"Saved: {emb_out}")
    print(f"Saved: {top_pairs_out}")
    print(f"Saved: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
