#!/usr/bin/env python3
"""Build compact story visuals for Istanbul dam scenario-risk outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Istanbul dam story visual package")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision"),
    )
    p.add_argument("--window-start", default="2026-03-01")
    p.add_argument("--window-end", default="2027-02-01")
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--threshold-high-risk-months", type=int, default=4)
    p.add_argument("--threshold-medium-risk-months", type=int, default=2)
    p.add_argument("--threshold-high-prob40", type=float, default=60.0)
    p.add_argument("--threshold-medium-prob40", type=float, default=35.0)
    return p.parse_args()


def risk_level(row: pd.Series, args: argparse.Namespace) -> str:
    if (
        int(row["months_lt40"]) >= args.threshold_high_risk_months
        or float(row["mean_prob_below_40_pct"]) >= args.threshold_high_prob40
    ):
        return "high"
    if (
        int(row["months_lt40"]) >= args.threshold_medium_risk_months
        or float(row["mean_prob_below_40_pct"]) >= args.threshold_medium_prob40
    ):
        return "medium"
    return "low"


def scenario_group_name(s: str) -> str:
    if s == "baseline":
        return "baseline"
    if s.startswith("dry_"):
        return "dry"
    if s.startswith("wet_"):
        return "wet"
    return "other"


def scenario_sort_key(s: str) -> tuple[int, int, str]:
    if s == "baseline":
        return (0, 0, s)
    if s.startswith("dry_"):
        label = s.split("_", 1)[1]
        ord_map = {"mild": 1, "base": 2, "stress": 2, "severe": 3, "extreme": 4}
        return (1, ord_map.get(label, 9), s)
    if s.startswith("wet_"):
        label = s.split("_", 1)[1]
        ord_map = {"mild": 1, "base": 2, "relief": 2, "severe": 3, "extreme": 4}
        return (2, ord_map.get(label, 9), s)
    return (3, 99, s)


def scenario_colors() -> dict[str, str]:
    return {
        "baseline": "#334155",
        "dry": "#b91c1c",
        "wet": "#0e7490",
        "other": "#6b7280",
    }


def build_story_figure(
    weights_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    out_png: Path,
    top_n: int,
    args: argparse.Namespace,
) -> None:
    colors = scenario_colors()

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Panel 1: scenario weights
    w = weights_df.copy()
    w["group"] = w["scenario"].map(scenario_group_name)
    w = w.sort_values("scenario", key=lambda s: s.map(scenario_sort_key))
    ax1.bar(w["scenario"], w["weight"] * 100.0, color=[colors.get(g, "#6b7280") for g in w["group"]], alpha=0.9)
    ax1.set_title("Senaryo Agirliklari (tarihsel anomali tabanli)")
    ax1.set_ylabel("Agirlik (%)")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(axis="y", alpha=0.25)

    # Panel 2: overall_mean scenario frontier
    ov = risk_df[risk_df["series"] == "overall_mean"].copy()
    ov["group"] = ov["scenario"].map(scenario_group_name)
    ov = ov.sort_values("scenario", key=lambda s: s.map(scenario_sort_key))
    ax2.scatter(
        ov["mean_yhat_pct"],
        ov["mean_prob_below_40_pct"],
        s=90,
        c=[colors.get(g, "#6b7280") for g in ov["group"]],
        alpha=0.9,
    )
    for _, r in ov.iterrows():
        ax2.text(float(r["mean_yhat_pct"]) + 0.3, float(r["mean_prob_below_40_pct"]) + 0.3, str(r["scenario"]), fontsize=8)
    ax2.set_title("overall_mean Senaryo Risk-Egri")
    ax2.set_xlabel("Ortalama tahmin (%)")
    ax2.set_ylabel("Ort. P(<40) (%)")
    ax2.grid(alpha=0.25)

    # Panel 3: top expected risk series
    g3 = expected_df.copy().sort_values("expected_prob_below_40_pct", ascending=False).head(max(1, int(top_n)))
    g3 = g3.sort_values("expected_prob_below_40_pct")
    ax3.barh(g3["series"], g3["expected_prob_below_40_pct"], color="#b45309", alpha=0.9, label="Beklenen P(<40)")
    ax3.plot(g3["prob_high_risk"] * 100.0, g3["series"], "o-", color="#b91c1c", label="P(high)")
    ax3.set_xlim(0, 100)
    ax3.set_title("Beklenen Riskte En Kritik Seriler")
    ax3.set_xlabel("Risk (%)")
    ax3.grid(axis="x", alpha=0.25)
    ax3.legend(loc="lower right", fontsize=8)

    # Panel 4: risk count by scenario
    rr = risk_df.copy()
    rr["risk_level"] = rr.apply(lambda r: risk_level(r, args), axis=1)
    rows = []
    for scen, g in rr.groupby("scenario"):
        rows.append(
            {
                "scenario": scen,
                "high": int((g["risk_level"] == "high").sum()),
                "medium": int((g["risk_level"] == "medium").sum()),
                "low": int((g["risk_level"] == "low").sum()),
            }
        )
    cdf = pd.DataFrame(rows).sort_values("scenario", key=lambda s: s.map(scenario_sort_key))
    x = np.arange(len(cdf))
    ax4.bar(x, cdf["low"], label="low", color="#15803d")
    ax4.bar(x, cdf["medium"], bottom=cdf["low"], label="medium", color="#b45309")
    ax4.bar(x, cdf["high"], bottom=cdf["low"] + cdf["medium"], label="high", color="#b91c1c")
    ax4.set_xticks(x)
    ax4.set_xticklabels(cdf["scenario"], rotation=30, ha="right")
    ax4.set_title("Senaryo Bazli Risk Dagilimi (seri sayisi)")
    ax4.set_ylabel("Seri adedi")
    ax4.grid(axis="y", alpha=0.25)
    ax4.legend(loc="upper right", fontsize=8)

    fig.suptitle("Istanbul Baraj Tahmin Hikaye Paketi", fontsize=16, weight="bold")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def build_overall_timeline(
    scenario_fc_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    dynamic_fc_df: pd.DataFrame | None,
    out_png: Path,
    out_csv: Path,
    window_start: str,
    window_end: str,
) -> pd.DataFrame:
    colors = scenario_colors()
    tmp = scenario_fc_df.copy()
    tmp["ds"] = pd.to_datetime(tmp["ds"])
    tmp = tmp[(tmp["ds"] >= pd.Timestamp(window_start)) & (tmp["ds"] <= pd.Timestamp(window_end))]
    tmp = tmp[tmp["series"] == "overall_mean"].copy()
    if tmp.empty:
        return pd.DataFrame()

    scen_order = scenario_order(tmp["scenario"].astype(str).unique().tolist())
    w = weights_df[["scenario", "weight"]].copy() if not weights_df.empty else pd.DataFrame(columns=["scenario", "weight"])
    if w.empty:
        w = pd.DataFrame({"scenario": scen_order, "weight": [1.0 / max(1, len(scen_order))] * len(scen_order)})
    w["weight"] = w["weight"] / max(float(w["weight"].sum()), 1e-9)

    merged = tmp.merge(w, on="scenario", how="left")
    merged["weight"] = merged["weight"].fillna(0.0)
    merged["weighted_yhat"] = merged["scenario_yhat"] * merged["weight"]
    expected = (
        merged.groupby("ds", as_index=False)["weighted_yhat"].sum().rename(columns={"weighted_yhat": "expected_yhat"})
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    for scen in scen_order:
        g = tmp[tmp["scenario"] == scen].sort_values("ds")
        if g.empty:
            continue
        grp = scenario_group_name(scen)
        alpha = 1.0 if scen == "baseline" else 0.7
        lw = 2.4 if scen == "baseline" else 1.8
        ax.plot(
            g["ds"],
            g["scenario_yhat"] * 100.0,
            label=scen,
            color=colors.get(grp, "#6b7280"),
            linewidth=lw,
            alpha=alpha,
        )

    ax.plot(
        expected["ds"],
        expected["expected_yhat"] * 100.0,
        color="#111827",
        linewidth=2.8,
        linestyle="--",
        label="weighted_expected",
    )

    if dynamic_fc_df is not None and not dynamic_fc_df.empty:
        dd = dynamic_fc_df.copy()
        dd["ds"] = pd.to_datetime(dd["ds"])
        dd = dd[(dd["ds"] >= pd.Timestamp(window_start)) & (dd["ds"] <= pd.Timestamp(window_end))]
        dd = dd[(dd["series"] == "overall_mean") & (dd["scenario"] == "baseline")].sort_values("ds")
        if not dd.empty:
            ax.plot(dd["ds"], dd["warning_threshold"] * 100.0, color="#b45309", linewidth=1.6, linestyle=":", label="warning_thr")
            ax.plot(
                dd["ds"],
                dd["critical_threshold"] * 100.0,
                color="#991b1b",
                linewidth=1.6,
                linestyle=":",
                label="critical_thr",
            )

    ax.set_title("overall_mean Aylik Senaryo Yolu + Agirlikli Beklenti")
    ax.set_ylabel("Doluluk (%)")
    ax.set_xlabel("Ay")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    expected["expected_yhat_pct"] = expected["expected_yhat"] * 100.0
    expected.to_csv(out_csv, index=False)
    return expected


def build_expected_gap_heatmap(
    dynamic_fc_df: pd.DataFrame | None,
    weights_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    out_png: Path,
    out_csv: Path,
    window_start: str,
    window_end: str,
    top_n: int,
) -> pd.DataFrame:
    if dynamic_fc_df is None or dynamic_fc_df.empty:
        return pd.DataFrame()
    tmp = dynamic_fc_df.copy()
    tmp["ds"] = pd.to_datetime(tmp["ds"])
    tmp = tmp[(tmp["ds"] >= pd.Timestamp(window_start)) & (tmp["ds"] <= pd.Timestamp(window_end))]
    if tmp.empty:
        return pd.DataFrame()

    w = weights_df[["scenario", "weight"]].copy() if not weights_df.empty else pd.DataFrame(columns=["scenario", "weight"])
    if w.empty:
        scenarios = scenario_order(tmp["scenario"].astype(str).unique().tolist())
        w = pd.DataFrame({"scenario": scenarios, "weight": [1.0 / max(1, len(scenarios))] * len(scenarios)})
    w["weight"] = w["weight"] / max(float(w["weight"].sum()), 1e-9)

    m = tmp.merge(w, on="scenario", how="left")
    m["weight"] = m["weight"].fillna(0.0)
    m["weighted_gap"] = m["scenario_gap_to_warning"] * m["weight"] * 100.0
    exp = m.groupby(["series", "ds"], as_index=False)["weighted_gap"].sum()
    exp = exp.rename(columns={"weighted_gap": "expected_gap_to_warning_pct"})

    top_series = (
        expected_df.sort_values("expected_prob_below_40_pct", ascending=False)["series"].head(max(1, int(top_n))).tolist()
        if not expected_df.empty
        else []
    )
    if not top_series:
        top_series = sorted(exp["series"].unique().tolist())[: max(1, int(top_n))]

    exp = exp[exp["series"].isin(top_series)].copy()
    if exp.empty:
        return pd.DataFrame()
    exp["series"] = pd.Categorical(exp["series"], categories=top_series, ordered=True)
    exp = exp.sort_values(["series", "ds"])

    pivot = exp.pivot(index="series", columns="ds", values="expected_gap_to_warning_pct")
    z = pivot.values
    zmax = float(np.nanmax(np.abs(z))) if np.isfinite(np.nanmax(np.abs(z))) else 1.0
    zmax = max(zmax, 1.0)

    fig, ax = plt.subplots(figsize=(14, max(4.8, 0.6 * len(pivot.index) + 2.5)))
    im = ax.imshow(z, aspect="auto", interpolation="nearest", cmap="RdYlGn", vmin=-zmax, vmax=zmax)
    xlabels = [pd.Timestamp(c).strftime("%Y-%m") for c in pivot.columns]
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    ax.set_title("Beklenen Uyari Esigi Gapi (agirlikli, puan)")
    ax.set_xlabel("Ay")
    ax.set_ylabel("Seri")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Gap to warning (puan)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    exp.to_csv(out_csv, index=False)
    return exp


def scenario_order(values: list[str]) -> list[str]:
    return sorted({str(v) for v in values}, key=scenario_sort_key)


def build_story_markdown(
    weights_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    overall_timeline_df: pd.DataFrame,
    expected_gap_df: pd.DataFrame,
    out_md: Path,
    window_start: str,
    window_end: str,
) -> None:
    lines = []
    lines.append("# Hikaye Ozeti")
    lines.append("")
    lines.append("## Senaryo Agirliklari")
    lines.append("")
    for _, r in weights_df.sort_values("weight", ascending=False).iterrows():
        lines.append(f"- {r['scenario']}: {float(r['weight'])*100.0:.1f}%")
    lines.append("")

    lines.append("## Beklenen Riskte En Kritik 5 Seri")
    lines.append("")
    top = expected_df.sort_values("expected_prob_below_40_pct", ascending=False).head(5)
    for _, r in top.iterrows():
        lines.append(
            "- "
            f"{r['series']}: Beklenen P(<40)={float(r['expected_prob_below_40_pct']):.1f}%, "
            f"P(high)={float(r['prob_high_risk'])*100.0:.1f}%, "
            f"beklenen ort. doluluk={float(r['expected_mean_yhat_pct']):.1f}%."
        )
    lines.append("")

    ov = risk_df[risk_df["series"] == "overall_mean"].sort_values("scenario", key=lambda s: s.map(scenario_sort_key))
    lines.append("## overall_mean Senaryo Ozeti")
    lines.append("")
    for _, r in ov.iterrows():
        lines.append(
            "- "
            f"{r['scenario']}: ortalama={float(r['mean_yhat_pct']):.1f}%, "
            f"ort. P(<40)={float(r['mean_prob_below_40_pct']):.1f}%, "
            f"<40 ay={int(r['months_lt40'])}."
        )

    if overall_timeline_df is not None and not overall_timeline_df.empty:
        g = overall_timeline_df.sort_values("ds")
        min_row = g.loc[g["expected_yhat_pct"].idxmin()]
        max_row = g.loc[g["expected_yhat_pct"].idxmax()]
        lines.append("")
        lines.append(f"## Agirlikli Aylik Beklenti ({window_start} - {window_end})")
        lines.append("")
        lines.append(
            "- "
            f"En dusuk beklenen sistem dolulugu: {float(min_row['expected_yhat_pct']):.1f}% "
            f"({pd.Timestamp(min_row['ds']).strftime('%Y-%m')})."
        )
        lines.append(
            "- "
            f"En yuksek beklenen sistem dolulugu: {float(max_row['expected_yhat_pct']):.1f}% "
            f"({pd.Timestamp(max_row['ds']).strftime('%Y-%m')})."
        )

    if expected_gap_df is not None and not expected_gap_df.empty:
        lines.append("")
        lines.append("## Dinamik Uyari Esigi Gap Ozeti")
        lines.append("")
        gg = expected_gap_df.copy()
        gg["series"] = gg["series"].astype(str)
        neg = gg.groupby("series", as_index=False)["expected_gap_to_warning_pct"].min()
        neg = neg.sort_values("expected_gap_to_warning_pct").head(5)
        for _, r in neg.iterrows():
            lines.append(
                "- "
                f"{r['series']}: en kotu agirlikli gap {float(r['expected_gap_to_warning_pct']):.1f} puan."
            )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    w_path = args.input_dir / "scenario_weights.csv"
    e_path = args.input_dir / "expected_risk_summary.csv"
    r_path = args.input_dir / "scenario_risk_summary.csv"
    sfc_path = args.input_dir / "scenario_forecasts.csv"
    dyn_path = args.input_dir / "scenario_dynamic_threshold_forecasts.csv"
    if not all(p.exists() for p in [w_path, e_path, r_path, sfc_path]):
        raise SystemExit("Required files missing. Run build_istanbul_dam_weighted_scenario_risk.py first.")

    w = pd.read_csv(w_path)
    e = pd.read_csv(e_path)
    r = pd.read_csv(r_path)
    sfc = pd.read_csv(sfc_path, parse_dates=["ds"])
    dyn = pd.read_csv(dyn_path, parse_dates=["ds"]) if dyn_path.exists() else None

    out_png = args.output_dir / "story_visual_pack.png"
    out_md = args.output_dir / "HIKAYE_OZETI.md"
    out_csv = args.output_dir / "story_key_metrics.csv"
    out_timeline_png = args.output_dir / "story_overall_timeline_weighted.png"
    out_timeline_csv = args.output_dir / "story_overall_timeline_weighted.csv"
    out_gap_png = args.output_dir / "story_expected_gap_heatmap.png"
    out_gap_csv = args.output_dir / "story_expected_gap_heatmap.csv"

    build_story_figure(weights_df=w, expected_df=e, risk_df=r, out_png=out_png, top_n=args.top_n, args=args)
    overall_timeline = build_overall_timeline(
        scenario_fc_df=sfc,
        weights_df=w,
        dynamic_fc_df=dyn,
        out_png=out_timeline_png,
        out_csv=out_timeline_csv,
        window_start=args.window_start,
        window_end=args.window_end,
    )
    gap_df = build_expected_gap_heatmap(
        dynamic_fc_df=dyn,
        weights_df=w,
        expected_df=e,
        out_png=out_gap_png,
        out_csv=out_gap_csv,
        window_start=args.window_start,
        window_end=args.window_end,
        top_n=args.top_n,
    )
    build_story_markdown(
        weights_df=w,
        expected_df=e,
        risk_df=r,
        overall_timeline_df=overall_timeline,
        expected_gap_df=gap_df,
        out_md=out_md,
        window_start=args.window_start,
        window_end=args.window_end,
    )

    key = e.sort_values("expected_prob_below_40_pct", ascending=False).copy()
    key.to_csv(out_csv, index=False)

    print(out_png)
    print(out_timeline_png)
    print(out_gap_png)
    print(out_md)
    print(out_csv)
    print(out_timeline_csv)
    if not gap_df.empty:
        print(out_gap_csv)


if __name__ == "__main__":
    main()
