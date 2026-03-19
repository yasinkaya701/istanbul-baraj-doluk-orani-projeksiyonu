#!/usr/bin/env python3
"""Scientific validation layer for internet-enriched extreme events.

Outputs:
- bilimsel_dogrulama_istatistikleri.csv
- tum_asiri_olaylar_bilimsel_filtreli.csv
- bilimsel_dogrulama_raporu.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Asiri olaylar icin bilimsel dogrulama ve kalite raporu.")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/tum_asiri_olaylar_internet_nedenleri.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perm-iter", type=int, default=3000)
    p.add_argument("--boot-iter", type=int, default=3000)
    return p.parse_args()


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1]."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    cmp = a[:, None] - b[None, :]
    gt = np.sum(cmp > 0)
    lt = np.sum(cmp < 0)
    return float((gt - lt) / (len(a) * len(b)))


def permutation_pvalue(
    active: np.ndarray,
    inactive: np.ndarray,
    rng: np.random.Generator,
    n_iter: int,
) -> float:
    if len(active) == 0 or len(inactive) == 0:
        return np.nan
    obs = abs(float(active.mean() - inactive.mean()))
    allv = np.concatenate([active, inactive])
    n_active = len(active)
    ge = 0
    for _ in range(n_iter):
        perm = rng.permutation(allv)
        a = perm[:n_active]
        b = perm[n_active:]
        stat = abs(float(a.mean() - b.mean()))
        if stat >= obs:
            ge += 1
    return float((ge + 1) / (n_iter + 1))


def bootstrap_ci_mean_diff(
    active: np.ndarray,
    inactive: np.ndarray,
    rng: np.random.Generator,
    n_iter: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if len(active) == 0 or len(inactive) == 0:
        return np.nan, np.nan
    diffs = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        a = rng.choice(active, size=len(active), replace=True)
        b = rng.choice(inactive, size=len(inactive), replace=True)
        diffs[i] = float(a.mean() - b.mean())
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return lo, hi


def add_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["enso_active"] = ~d["enso_phase"].astype(str).isin(["unknown", "enso_neutral"])
    d["nao_active"] = ~d["nao_phase"].astype(str).isin(["unknown", "nao_neutral"])
    d["known_window_active"] = d["known_window_title"].fillna("").astype(str).str.len().gt(0)
    d["quant_active"] = d["quant_cause_primary"].fillna("").astype(str).str.len().gt(0)
    d["source_count"] = (
        d["internet_source_links"]
        .fillna("")
        .astype(str)
        .apply(lambda x: len([p for p in x.split(" ; ") if p.strip()]))
    )
    d["driver_evidence_count"] = (
        d["enso_active"].astype(int)
        + d["nao_active"].astype(int)
        + d["known_window_active"].astype(int)
        + d["quant_active"].astype(int)
    )
    d["scientific_score"] = d["driver_evidence_count"] + (d["source_count"] >= 3).astype(int)

    def tier(score: int) -> str:
        if score >= 4:
            return "A"
        if score == 3:
            return "B"
        if score == 2:
            return "C"
        return "D"

    d["scientific_tier"] = d["scientific_score"].astype(int).map(tier)
    return d


def run_tests(df: pd.DataFrame, rng: np.random.Generator, n_perm: int, n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tests = [
        ("enso_active", "ENSO_nonneutral_vs_neutral"),
        ("nao_active", "NAO_nonneutral_vs_neutral"),
        ("known_window_active", "known_window_hit_vs_nohit"),
        ("quant_active", "quant_context_hit_vs_nohit"),
    ]

    for var, g in df.groupby("variable"):
        for col, test_name in tests:
            active = g[g[col]]["peak_severity_score"].to_numpy(dtype=float)
            inactive = g[~g[col]]["peak_severity_score"].to_numpy(dtype=float)
            if len(active) < 8 or len(inactive) < 8:
                rows.append(
                    {
                        "variable": var,
                        "test_name": test_name,
                        "n_active": len(active),
                        "n_inactive": len(inactive),
                        "mean_active": np.nan,
                        "mean_inactive": np.nan,
                        "mean_diff_active_minus_inactive": np.nan,
                        "cliffs_delta": np.nan,
                        "p_value_perm": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "significant_0_05": False,
                        "yorum_tr": "orneklem yetersiz",
                    }
                )
                continue

            mean_active = float(np.mean(active))
            mean_inactive = float(np.mean(inactive))
            mean_diff = mean_active - mean_inactive
            cdelta = cliffs_delta(active, inactive)
            pval = permutation_pvalue(active, inactive, rng=rng, n_iter=n_perm)
            ci_low, ci_high = bootstrap_ci_mean_diff(active, inactive, rng=rng, n_iter=n_boot)
            sig = bool(np.isfinite(pval) and pval < 0.05)

            if not np.isfinite(cdelta):
                strength = "belirsiz"
            else:
                ad = abs(cdelta)
                if ad < 0.147:
                    strength = "cok_kucuk"
                elif ad < 0.33:
                    strength = "kucuk"
                elif ad < 0.474:
                    strength = "orta"
                else:
                    strength = "buyuk"

            direction = "arti" if mean_diff > 0 else "eksi"
            yorum = (
                f"aktif grubun siddeti {'daha yuksek' if mean_diff > 0 else 'daha dusuk'}; "
                f"etki={strength}, yon={direction}"
            )

            rows.append(
                {
                    "variable": var,
                    "test_name": test_name,
                    "n_active": len(active),
                    "n_inactive": len(inactive),
                    "mean_active": mean_active,
                    "mean_inactive": mean_inactive,
                    "mean_diff_active_minus_inactive": mean_diff,
                    "cliffs_delta": cdelta,
                    "p_value_perm": pval,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "significant_0_05": sig,
                    "yorum_tr": yorum,
                }
            )

    out = pd.DataFrame(rows).sort_values(["variable", "test_name"]).reset_index(drop=True)
    out = add_fdr_bh(out, p_col="p_value_perm", q_col="p_fdr_bh")
    out["significant_fdr_0_05"] = out["p_fdr_bh"] < 0.05
    out["ci_excludes_zero"] = (out["ci95_low"] > 0) | (out["ci95_high"] < 0)
    out["strong_significant"] = out["significant_fdr_0_05"] & out["ci_excludes_zero"]
    return out


def add_fdr_bh(df: pd.DataFrame, p_col: str, q_col: str) -> pd.DataFrame:
    out = df.copy()
    p = pd.to_numeric(out[p_col], errors="coerce")
    mask = p.notna()
    q = pd.Series(np.nan, index=out.index, dtype=float)
    if int(mask.sum()) == 0:
        out[q_col] = q
        return out

    pvals = p[mask].to_numpy(dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = len(ranked)
    bh = ranked * m / (np.arange(1, m + 1))
    # Monotonicity adjustment from largest to smallest.
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)

    back = np.empty_like(bh)
    back[order] = bh
    q.loc[mask] = back
    out[q_col] = q
    return out


def write_report(
    out_path: Path,
    df: pd.DataFrame,
    stats: pd.DataFrame,
    filtered: pd.DataFrame,
) -> None:
    by_var = (
        df.groupby("variable", as_index=False)
        .agg(
            olay_sayisi=("event_id", "size"),
            ort_siddet=("peak_severity_score", "mean"),
            ort_kanit=("driver_evidence_count", "mean"),
            ort_kaynak=("source_count", "mean"),
            a_tier=("scientific_tier", lambda s: int((s == "A").sum())),
            b_tier=("scientific_tier", lambda s: int((s == "B").sum())),
            c_tier=("scientific_tier", lambda s: int((s == "C").sum())),
            d_tier=("scientific_tier", lambda s: int((s == "D").sum())),
        )
        .sort_values("olay_sayisi", ascending=False)
    )

    top = df.sort_values("peak_severity_score", ascending=False).head(40)
    sig_raw = stats[stats["significant_0_05"] == True].copy()  # noqa: E712
    sig_fdr = stats[stats["significant_fdr_0_05"] == True].copy()  # noqa: E712
    sig_strong = stats[stats["strong_significant"] == True].copy()  # noqa: E712

    lines: list[str] = []
    lines.append("# Bilimsel Dogrulama Raporu")
    lines.append("")
    lines.append(f"- Toplam olay: **{len(df)}**")
    lines.append(f"- Bilimsel filtreyi gecen olay: **{len(filtered)}**")
    lines.append(
        "- Not: Bu rapor istatistiksel iliskiyi olcer; tek basina kesin nedensellik kaniti degildir."
    )
    lines.append("")
    lines.append("## Veri Kalitesi")
    lines.append(f"- Tekrar eden event_id: {int(df['event_id'].duplicated().sum())}")
    lines.append(f"- Eksik peak_severity_score: {int(df['peak_severity_score'].isna().sum())}")
    lines.append(
        f"- Kaynaksiz olay (source_count=0): {int((df['source_count'] == 0).sum())}"
    )
    lines.append("")
    lines.append("## Degisken Bazli Ozet")
    for _, r in by_var.iterrows():
        lines.append(
            f"- {r['variable']}: olay={int(r['olay_sayisi'])}, ort_siddet={float(r['ort_siddet']):.3f}, "
            f"ort_kanit={float(r['ort_kanit']):.2f}, ort_kaynak={float(r['ort_kaynak']):.2f}, "
            f"tier(A/B/C/D)=({int(r['a_tier'])}/{int(r['b_tier'])}/{int(r['c_tier'])}/{int(r['d_tier'])})"
        )
    lines.append("")
    lines.append("## Anlamli Testler (Ham p<0.05)")
    if sig_raw.empty:
        lines.append("- Ham p<0.05 sonuc bulunmadi.")
    else:
        for _, r in sig_raw.iterrows():
            lines.append(
                f"- {r['variable']} | {r['test_name']}: diff={float(r['mean_diff_active_minus_inactive']):.3f}, "
                f"cliff={float(r['cliffs_delta']):.3f}, p={float(r['p_value_perm']):.4f}, "
                f"CI95=[{float(r['ci95_low']):.3f}, {float(r['ci95_high']):.3f}]"
            )
    lines.append("")
    lines.append("## Anlamli Testler (FDR duzeltmeli q<0.05)")
    if sig_fdr.empty:
        lines.append("- FDR duzeltmesi sonrasi anlamli sonuc bulunmadi.")
    else:
        for _, r in sig_fdr.iterrows():
            lines.append(
                f"- {r['variable']} | {r['test_name']}: q={float(r['p_fdr_bh']):.4f}, "
                f"p={float(r['p_value_perm']):.4f}, diff={float(r['mean_diff_active_minus_inactive']):.3f}, "
                f"CI95=[{float(r['ci95_low']):.3f}, {float(r['ci95_high']):.3f}]"
            )
    lines.append("")
    lines.append("## Guclu Anlamlilik (FDR q<0.05 ve CI 0'i kesmiyor)")
    if sig_strong.empty:
        lines.append("- Guclu anlamli test bulunmadi.")
    else:
        for _, r in sig_strong.iterrows():
            lines.append(
                f"- {r['variable']} | {r['test_name']}: q={float(r['p_fdr_bh']):.4f}, "
                f"diff={float(r['mean_diff_active_minus_inactive']):.3f}, "
                f"CI95=[{float(r['ci95_low']):.3f}, {float(r['ci95_high']):.3f}]"
            )
    lines.append("")
    lines.append("## En Siddetli 40 Olay (Bilimsel Skor ile)")
    lines.append("")
    lines.append("|Event|Degisken|Siddet|Tier|Skor|Kanit|ENSO|NAO|")
    lines.append("|---|---|---:|---:|---:|---:|---|---|")
    for _, r in top.iterrows():
        lines.append(
            f"|{r['event_id']}|{r['variable']}|{float(r['peak_severity_score']):.2f}|{r['scientific_tier']}|"
            f"{int(r['scientific_score'])}|{int(r['driver_evidence_count'])}|{r['enso_phase']}|{r['nao_phase']}|"
        )
    lines.append("")
    lines.append("## Bilimsel Filtre Kurali")
    lines.append("- Kosul: `scientific_score >= 3` ve `internet_confidence` yuksek/cok_yuksek")
    lines.append("- Bu filtre, coklu kanit ve kaynak yogunlugunu bir arada ister.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv).copy()
    df["variable"] = df["variable"].astype(str).str.lower().str.strip()
    df["peak_severity_score"] = pd.to_numeric(df["peak_severity_score"], errors="coerce")
    df = df.dropna(subset=["peak_severity_score"]).copy()

    df = add_evidence_columns(df)

    rng = np.random.default_rng(args.seed)
    stats = run_tests(df, rng=rng, n_perm=args.perm_iter, n_boot=args.boot_iter)

    filtered = df[
        (df["scientific_score"] >= 3)
        & (df["internet_confidence"].astype(str).isin(["yuksek", "cok_yuksek"]))
    ].copy()
    filtered = filtered.sort_values(
        ["scientific_score", "peak_severity_score"], ascending=[False, False]
    ).reset_index(drop=True)

    stats_csv = args.output_dir / "bilimsel_dogrulama_istatistikleri.csv"
    filtered_csv = args.output_dir / "tum_asiri_olaylar_bilimsel_filtreli.csv"
    report_md = args.output_dir / "bilimsel_dogrulama_raporu.md"

    stats.to_csv(stats_csv, index=False)
    filtered.to_csv(filtered_csv, index=False)
    write_report(report_md, df=df, stats=stats, filtered=filtered)

    print(f"Wrote: {stats_csv}")
    print(f"Wrote: {filtered_csv}")
    print(f"Wrote: {report_md}")
    print(
        filtered[["variable", "scientific_tier"]]
        .value_counts()
        .rename("count")
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
