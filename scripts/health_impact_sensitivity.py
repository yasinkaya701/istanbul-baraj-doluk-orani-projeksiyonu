#!/usr/bin/env python3
"""Run scenario sensitivity for health_impact_analysis.py.

This script executes a parameter grid over epidemiology and adaptation choices
and aggregates key outputs into compact comparison tables.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensitivity analysis for health impact projections")
    p.add_argument("--temp-csv", type=Path, required=True, help="Temperature input CSV")
    p.add_argument("--humidity-csv", type=Path, required=True, help="Humidity input CSV")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory for sensitivity package")
    p.add_argument("--base-script", type=Path, default=Path("scripts/health_impact_analysis.py"))
    p.add_argument("--analysis-scope", choices=["all", "forecast_only", "historical_only"], default="all")
    p.add_argument("--baseline-start", type=int, default=1991)
    p.add_argument("--baseline-end", type=int, default=2020)
    p.add_argument("--future-start", type=int, default=2026)
    p.add_argument("--future-end", type=int, default=2035)
    p.add_argument(
        "--epi-modes",
        default="meta_urban_mortality_yang2024,meta_urban_morbidity_yang2024,meta_urban_heatwave_mortality_yang2024",
        help="Comma-separated epi modes",
    )
    p.add_argument(
        "--epi-ci-bounds",
        default="point,lower,upper",
        help="Comma-separated ci bound choices for epi presets: point,lower,upper",
    )
    p.add_argument(
        "--threshold-quantiles",
        default="0.80,0.85,0.90",
        help="Comma-separated baseline quantiles",
    )
    p.add_argument(
        "--adaptation-modes",
        default="none,moderate,strong",
        help="Comma-separated adaptation modes",
    )
    p.add_argument(
        "--humidity-interaction-modes",
        default="true",
        help="Comma-separated bool flags: true,false",
    )
    return p.parse_args()


def parse_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    out = []
    for x in parse_list(raw):
        out.append(float(x))
    return out


def parse_bool_list(raw: str) -> list[bool]:
    out = []
    for x in parse_list(raw):
        v = x.lower()
        if v in {"1", "true", "yes", "y"}:
            out.append(True)
        elif v in {"0", "false", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean flag: {x}")
    return out


def scenario_id(epi_mode: str, q: float, adaptation_mode: str, hum_int: bool) -> str:
    q_text = str(q).replace(".", "p")
    return f"epi_{epi_mode}__q_{q_text}__adapt_{adaptation_mode}__humint_{str(hum_int).lower()}"


def build_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Sensitivity Analysis\n\nNo scenarios were completed.\n"

    worst = df.sort_values("future_rr_mean", ascending=False).head(10)
    best = df.sort_values("future_rr_mean", ascending=True).head(10)

    lines = [
        "# Sensitivity Analysis Summary",
        "",
        f"- Scenarios: {len(df)}",
        f"- Future RR mean range: {df['future_rr_mean'].min():.4f} - {df['future_rr_mean'].max():.4f}",
        f"- Future AF mean range: {df['future_af_mean'].min():.4f} - {df['future_af_mean'].max():.4f}",
        f"- Delta RR mean range: {df['delta_rr_mean'].min():.4f} - {df['delta_rr_mean'].max():.4f}",
        "",
        "## Top 10 Highest Future Risk",
        "",
    ]
    for _, r in worst.iterrows():
        lines.append(
            f"- {r['scenario_id']}: future_rr={r['future_rr_mean']:.4f}, future_af={r['future_af_mean']:.4f}, "
            f"epi={r['epi_mode']}, q={r['threshold_quantile']}, adapt={r['adaptation_mode']}, hum_int={r['humidity_interaction']}"
        )

    lines += ["", "## Top 10 Lowest Future Risk", ""]
    for _, r in best.iterrows():
        lines.append(
            f"- {r['scenario_id']}: future_rr={r['future_rr_mean']:.4f}, future_af={r['future_af_mean']:.4f}, "
            f"epi={r['epi_mode']}, q={r['threshold_quantile']}, adapt={r['adaptation_mode']}, hum_int={r['humidity_interaction']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    epi_modes = parse_list(args.epi_modes)
    epi_ci_bounds = parse_list(args.epi_ci_bounds)
    threshold_quantiles = parse_float_list(args.threshold_quantiles)
    adaptation_modes = parse_list(args.adaptation_modes)
    hum_int_modes = parse_bool_list(args.humidity_interaction_modes)

    records: list[dict] = []

    for epi_mode in epi_modes:
        for epi_ci_bound in epi_ci_bounds:
            for q in threshold_quantiles:
                for adaptation_mode in adaptation_modes:
                    for hum_int in hum_int_modes:
                        sid = (
                            scenario_id(epi_mode, q, adaptation_mode, hum_int)
                            + f"__ci_{str(epi_ci_bound).lower()}"
                        )
                        run_dir = runs_dir / sid
                        run_dir.mkdir(parents=True, exist_ok=True)

                        cmd = [
                            "python3",
                            str(args.base_script),
                            "--temp-csv",
                            str(args.temp_csv),
                            "--humidity-csv",
                            str(args.humidity_csv),
                            "--output-dir",
                            str(run_dir),
                            "--analysis-scope",
                            str(args.analysis_scope),
                            "--baseline-start",
                            str(args.baseline_start),
                            "--baseline-end",
                            str(args.baseline_end),
                            "--future-start",
                            str(args.future_start),
                            "--future-end",
                            str(args.future_end),
                            "--epi-mode",
                            str(epi_mode),
                            "--epi-ci-bound",
                            str(epi_ci_bound),
                            "--risk-threshold-mode",
                            "baseline_quantile",
                            "--risk-threshold-quantile",
                            str(q),
                            "--adaptation-mode",
                            str(adaptation_mode),
                        ]
                        if hum_int:
                            cmd.append("--enable-humidity-interaction")
                        else:
                            cmd.append("--no-enable-humidity-interaction")

                        cp = subprocess.run(cmd, capture_output=True, text=True)
                        if cp.returncode != 0:
                            records.append(
                                {
                                    "scenario_id": sid,
                                    "status": "failed",
                                    "stderr": cp.stderr[-2000:],
                                }
                            )
                            continue

                        summary_path = run_dir / "health_impact_summary.json"
                        if not summary_path.exists():
                            records.append({"scenario_id": sid, "status": "failed", "stderr": "missing summary json"})
                            continue

                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        b = summary.get("baseline", {})
                        f = summary.get("future", {})
                        d = summary.get("delta", {})

                        records.append(
                            {
                                "scenario_id": sid,
                                "status": "ok",
                                "epi_mode": epi_mode,
                                "epi_ci_bound": epi_ci_bound,
                                "threshold_quantile": q,
                                "adaptation_mode": adaptation_mode,
                                "humidity_interaction": hum_int,
                                "risk_beta_effective": summary.get("inputs", {}).get("risk_beta_per_c_effective"),
                                "risk_threshold_c": summary.get("inputs", {}).get("risk_threshold_c"),
                                "baseline_rr_mean": b.get("mean_proxy_relative_risk"),
                                "future_rr_mean": f.get("mean_proxy_relative_risk"),
                                "delta_rr_mean": d.get("mean_proxy_relative_risk_delta"),
                                "baseline_af_mean": b.get("mean_attributable_fraction"),
                                "future_af_mean": f.get("mean_attributable_fraction"),
                                "delta_af_mean": d.get("mean_attributable_fraction_delta"),
                                "future_threshold_exceed_share": f.get("threshold_exceed_month_share"),
                                "future_wet_hot_share": f.get("wet_hot_share"),
                                "future_dry_hot_share": f.get("dry_hot_share"),
                                "future_adaptation_factor_mean": f.get("mean_adaptation_factor"),
                                "future_out_of_distribution_share": f.get("out_of_distribution_share"),
                            }
                        )

    df = pd.DataFrame(records)
    summary_csv = args.output_dir / "sensitivity_summary.csv"
    df.to_csv(summary_csv, index=False)

    ok = df[df["status"] == "ok"].copy() if not df.empty and "status" in df.columns else pd.DataFrame()
    if not ok.empty:
        ranked = ok.sort_values("future_rr_mean", ascending=False)
        ranked_csv = args.output_dir / "sensitivity_ranked_future_risk.csv"
        ranked.to_csv(ranked_csv, index=False)
    else:
        ranked_csv = args.output_dir / "sensitivity_ranked_future_risk.csv"
        pd.DataFrame().to_csv(ranked_csv, index=False)

    md_path = args.output_dir / "sensitivity_summary.md"
    md_path.write_text(build_markdown(ok), encoding="utf-8")

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {ranked_csv}")
    print(f"Wrote: {md_path}")
    if not ok.empty:
        print(
            "Future RR mean range:",
            f"{ok['future_rr_mean'].min():.4f}",
            "to",
            f"{ok['future_rr_mean'].max():.4f}",
        )


if __name__ == "__main__":
    main()
