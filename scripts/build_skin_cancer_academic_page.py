#!/usr/bin/env python3
"""Build an academically expanded and presentation-ready skin-cancer page.

Inputs:
- output/yearly_skin_cancer_evidence/annual_climate_and_skin_cancer_proxy.csv
- output/yearly_skin_cancer_evidence/trend_proof_table.csv
- output/yearly_skin_cancer_evidence/historical_correlation_matrix.csv
- output/yearly_skin_cancer_evidence/run_meta.json

Outputs:
- academic_literature_table.csv
- literature_alignment_checks.csv
- figures/academic_projection_strip.png
- akademik_sunum_sayfasi.html
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build skin-cancer academic presentation page")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/yearly_skin_cancer_evidence"),
        help="Directory containing yearly skin-cancer evidence outputs",
    )
    p.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Optional output html path (default: <input-dir>/akademik_sunum_sayfasi.html)",
    )
    return p.parse_args()


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    return v


def _fmt(x: Any, digits: int = 3, default: str = "-") -> str:
    v = _safe_float(x)
    if not np.isfinite(v):
        return default
    return f"{v:.{digits}f}"


def _fmt_pct(x: Any, digits: int = 1, default: str = "-") -> str:
    v = _safe_float(x)
    if not np.isfinite(v):
        return default
    return f"{v:.{digits}f}%"


def _find_row(df: pd.DataFrame, year: int) -> pd.Series:
    out = df.loc[df["year"] == int(year)]
    if out.empty:
        raise SystemExit(f"Year not found in annual table: {year}")
    return out.iloc[0]


def _trend_row(trend: pd.DataFrame, variable: str) -> pd.Series:
    out = trend.loc[trend["variable"].astype(str) == str(variable)]
    if out.empty:
        raise SystemExit(f"Trend variable not found: {variable}")
    return out.iloc[0]


def _load_required(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return path


def build_literature_table() -> pd.DataFrame:
    rows = [
        {
            "evidence_id": "WHO-QA-UV-SKIN",
            "category": "Public health guidance",
            "exposure": "UV radiation",
            "outcome": "Skin cancer",
            "quantitative_finding": "Main environmental cause of skin cancers; 10% ozone reduction can add ~300,000 non-melanoma cases/year globally.",
            "expectation_for_model": "Higher effective UV should increase skin-cancer risk pressure.",
            "source_url": "https://www.who.int/news-room/questions-and-answers/item/radiation-ultraviolet-(uv)-radiation-and-skin-cancer",
        },
        {
            "evidence_id": "WHO-FS-UV",
            "category": "Public health burden",
            "exposure": "Solar UV",
            "outcome": "Global burden",
            "quantitative_finding": "Estimated annual burden is 2-3 million non-melanoma and ~132,000 melanoma cases worldwide.",
            "expectation_for_model": "UV-linked burden is materially relevant and not negligible.",
            "source_url": "https://www.who.int/news-room/fact-sheets/detail/ultraviolet-radiation",
        },
        {
            "evidence_id": "WHO-ILO-2023",
            "category": "Occupational epidemiology",
            "exposure": "Chronic occupational solar UV",
            "outcome": "Non-melanoma skin cancer mortality",
            "quantitative_finding": "About 1 in 3 non-melanoma skin-cancer deaths attributable to occupational sun exposure; pooled risk increase ~60% among outdoor workers.",
            "expectation_for_model": "Sustained UV exposure should be treated as strong long-run risk amplifier.",
            "source_url": "https://www.who.int/news/item/08-11-2023-working-under-the-sun-causes-1-in-3-deaths-from-non-melanoma-skin-cancer--say-who-and-ilo",
        },
        {
            "evidence_id": "IARC-2025-UV",
            "category": "Cancer causation",
            "exposure": "UV radiation",
            "outcome": "Melanoma attribution",
            "quantitative_finding": "IARC communication reports that >80% of melanomas in fair-skinned populations are attributable to UV radiation (e.g., UK estimate ~83%).",
            "expectation_for_model": "Directionality should remain UV-driven for melanoma-related pressure.",
            "source_url": "https://www.iarc.who.int/wp-content/uploads/2025/02/wcd-2025-skin-cancer-infographic-EN.pdf",
        },
        {
            "evidence_id": "PMID-21054335",
            "category": "Meta-analysis (occupational UV)",
            "exposure": "Occupational UV",
            "outcome": "Cutaneous SCC",
            "quantitative_finding": "Systematic review/meta-analysis: pooled OR 1.77 (95% CI 1.40-2.22) for SCC.",
            "expectation_for_model": "Higher cumulative UV exposure should map to higher risk (positive sign).",
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/21054335/",
        },
        {
            "evidence_id": "PMID-23033409",
            "category": "Meta-analysis (artificial UV)",
            "exposure": "Indoor tanning",
            "outcome": "NMSC (SCC/BCC)",
            "quantitative_finding": "Summary RR: SCC 1.67 (1.29-2.17), BCC 1.29 (1.08-1.53).",
            "expectation_for_model": "UV-related exposure increments should not reduce risk.",
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/23033409/",
        },
        {
            "evidence_id": "PMID-34885049",
            "category": "Meta-analysis (indoor tanning)",
            "exposure": "Indoor tanning (dose + early age)",
            "outcome": "Melanoma and NMSC",
            "quantitative_finding": "RR melanoma 1.27, NMSC 1.40; early-onset melanoma RR 1.75.",
            "expectation_for_model": "UV exposure effect should be monotonic and stronger in high/early exposure contexts.",
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/34885049/",
        },
        {
            "evidence_id": "NASA-POWER-API",
            "category": "Data provenance",
            "exposure": "ALLSKY_SFC_SW_DWN and CLOUD_AMT",
            "outcome": "Climate driver inputs",
            "quantitative_finding": "NASA POWER API provides monthly irradiance and cloud amount used to derive yearly effective UV proxy.",
            "expectation_for_model": "Solar and cloud are physically grounded drivers for UV-related exposure pressure.",
            "source_url": "https://power.larc.nasa.gov/api/temporal/monthly/point",
        },
    ]
    return pd.DataFrame(rows)


def build_alignment_checks(
    annual: pd.DataFrame,
    trend: pd.DataFrame,
    corr: pd.DataFrame,
) -> pd.DataFrame:
    hist = annual.loc[~annual["is_projected"]].copy()
    hist = hist.sort_values("year").reset_index(drop=True)

    def corr_val(a: str, b: str) -> float:
        if a not in corr.index or b not in corr.columns:
            return float("nan")
        return _safe_float(corr.loc[a, b])

    solar_uv_corr = corr_val("solar_kwh_m2_day", "effective_uv_kwh_m2_day")
    cloud_uv_corr = corr_val("cloud_pct", "effective_uv_kwh_m2_day")
    uv_case_corr = corr_val("effective_uv_kwh_m2_day", "cases_per10k_mid")

    uv_tr = _trend_row(trend, "effective_uv_kwh_m2_day")
    case_tr = _trend_row(trend, "cases_per10k_mid")
    cloud_tr = _trend_row(trend, "cloud_pct")
    solar_tr = _trend_row(trend, "solar_kwh_m2_day")

    hist_end = int(hist["year"].max())
    proj = annual.loc[annual["is_projected"]].copy()
    proj_end = int(proj["year"].max()) if not proj.empty else hist_end
    case_hist = _safe_float(_find_row(annual, hist_end)["cases_per10k_mid"])
    case_proj = _safe_float(_find_row(annual, proj_end)["cases_per10k_mid"])

    rr_beta_mid = np.nan
    if "rr_mid" in annual.columns and "effective_uv_delta_pct" in annual.columns:
        rr_beta_mid = _safe_float(np.log(_safe_float(annual["rr_mid"].iloc[-1])) / _safe_float(annual["effective_uv_delta_pct"].iloc[-1]))

    checks = [
        {
            "check_name": "Solar-UV sign",
            "evidence_type": "data-driven",
            "literature_expectation": "Solar up -> effective UV up (positive relation)",
            "observed_metric": "corr(solar, effective_uv)",
            "observed_value": solar_uv_corr,
            "pass_rule": "> 0",
            "pass_flag": bool(np.isfinite(solar_uv_corr) and solar_uv_corr > 0),
            "notes": "Physical consistency check between irradiance and UV proxy.",
        },
        {
            "check_name": "Cloud-UV sign",
            "evidence_type": "data-driven",
            "literature_expectation": "Cloud up -> effective UV down (negative relation)",
            "observed_metric": "corr(cloud, effective_uv)",
            "observed_value": cloud_uv_corr,
            "pass_rule": "< 0",
            "pass_flag": bool(np.isfinite(cloud_uv_corr) and cloud_uv_corr < 0),
            "notes": "Cloud attenuation should suppress surface UV exposure.",
        },
        {
            "check_name": "Historical UV trend",
            "evidence_type": "data-driven",
            "literature_expectation": "If solar rises and cloud drops, UV pressure should not decrease",
            "observed_metric": "OLS slope effective_uv + p-value",
            "observed_value": _safe_float(uv_tr["slope_per_year"]),
            "pass_rule": "slope > 0 and p < 0.05",
            "pass_flag": bool(
                np.isfinite(_safe_float(uv_tr["slope_per_year"]))
                and _safe_float(uv_tr["slope_per_year"]) > 0
                and _safe_float(uv_tr["p_value"]) < 0.05
            ),
            "notes": f"p={_fmt(uv_tr['p_value'], digits=4)}",
        },
        {
            "check_name": "UV-risk direction",
            "evidence_type": "assumption-driven",
            "literature_expectation": "Higher UV exposure should increase skin-cancer risk",
            "observed_metric": "corr(effective_uv, cases_per10k_mid)",
            "observed_value": uv_case_corr,
            "pass_rule": "> 0",
            "pass_flag": bool(np.isfinite(uv_case_corr) and uv_case_corr > 0),
            "notes": "Proxy risk is UV-linked by design; sign is expected and should stay positive.",
        },
        {
            "check_name": "Projection direction",
            "evidence_type": "model-output",
            "literature_expectation": "With positive UV pressure, near-term risk should not decline",
            "observed_metric": f"cases_per10k_mid {hist_end}->{proj_end}",
            "observed_value": case_proj - case_hist,
            "pass_rule": ">= 0",
            "pass_flag": bool(np.isfinite(case_proj) and np.isfinite(case_hist) and (case_proj - case_hist) >= 0),
            "notes": f"{_fmt(case_hist, 4)} -> {_fmt(case_proj, 4)}",
        },
        {
            "check_name": "Driver coherence",
            "evidence_type": "data-driven",
            "literature_expectation": "Solar trend positive and cloud trend negative together reinforce UV pressure",
            "observed_metric": "sign(solar slope), sign(cloud slope)",
            "observed_value": _safe_float(_safe_float(solar_tr["slope_per_year"]) - _safe_float(cloud_tr["slope_per_year"])),
            "pass_rule": "solar slope > 0 and cloud slope < 0",
            "pass_flag": bool(
                _safe_float(solar_tr["slope_per_year"]) > 0 and _safe_float(cloud_tr["slope_per_year"]) < 0
            ),
            "notes": (
                f"solar slope={_fmt(solar_tr['slope_per_year'], 4)}, "
                f"cloud slope={_fmt(cloud_tr['slope_per_year'], 4)}"
            ),
        },
        {
            "check_name": "Cases trend significance",
            "evidence_type": "model-output",
            "literature_expectation": "Risk trend should be statistically detectable in historical window",
            "observed_metric": "OLS slope cases_per10k_mid + p-value",
            "observed_value": _safe_float(case_tr["slope_per_year"]),
            "pass_rule": "slope > 0 and p < 0.05",
            "pass_flag": bool(
                np.isfinite(_safe_float(case_tr["slope_per_year"]))
                and _safe_float(case_tr["slope_per_year"]) > 0
                and _safe_float(case_tr["p_value"]) < 0.05
            ),
            "notes": f"p={_fmt(case_tr['p_value'], digits=4)}",
        },
    ]

    out = pd.DataFrame(checks)
    out["alignment_label"] = np.where(out["pass_flag"], "uyumlu", "gozden_gecir")
    return out


def plot_projection_strip(annual: pd.DataFrame, out_png: Path) -> None:
    d = annual.sort_values("year").copy()
    years = d["year"].to_numpy(dtype=int)
    is_proj = d["is_projected"].to_numpy(dtype=bool)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax1 = plt.subplots(figsize=(13.5, 6.5))
    ax2 = ax1.twinx()

    ax1.plot(years, d["effective_uv_kwh_m2_day"], color="#f59e0b", linewidth=2.5, label="Etkili UV proxy")
    ax2.plot(years, d["cases_per10k_mid"], color="#dc2626", linewidth=2.5, label="Cilt kanseri proxy (10k)")
    ax2.fill_between(
        years,
        d["cases_per10k_low"].to_numpy(dtype=float),
        d["cases_per10k_high"].to_numpy(dtype=float),
        color="#fecaca",
        alpha=0.45,
        label="Belirsizlik bandi",
    )

    if np.any(is_proj):
        x0 = int(years[is_proj].min()) - 0.5
        x1 = int(years[is_proj].max()) + 0.5
        ax1.axvspan(x0, x1, color="#f5f5f4", alpha=0.85, zorder=0)
        ax1.axvline(x0, color="#6b7280", linestyle="--", linewidth=1.1)

    ax1.set_title("Yillik Gelisim: Etkili UV ve 10.000 Kiside Cilt Kanseri Proxy")
    ax1.set_xlabel("Yil")
    ax1.set_ylabel("Etkili UV (kWh/m2/gun)", color="#92400e")
    ax2.set_ylabel("Vaka / 10.000", color="#991b1b")
    ax1.grid(alpha=0.24)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _table_html(df: pd.DataFrame, classes: str = "", max_rows: int | None = None) -> str:
    view = df.copy()
    if max_rows is not None:
        view = view.head(int(max_rows))
    cols = list(view.columns)
    lines = [f'<table class="{html.escape(classes)}">', "<thead><tr>"]
    for c in cols:
        lines.append(f"<th>{html.escape(str(c))}</th>")
    lines.append("</tr></thead><tbody>")
    for _, row in view.iterrows():
        lines.append("<tr>")
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if np.isfinite(v):
                    txt = f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}"
                else:
                    txt = "-"
            else:
                txt = str(v)
            lines.append(f"<td>{html.escape(txt)}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "".join(lines)


def build_html(
    annual: pd.DataFrame,
    trend: pd.DataFrame,
    literature: pd.DataFrame,
    checks: pd.DataFrame,
    assumptions: dict[str, Any],
    inputs: dict[str, Any],
    out_html: Path,
) -> None:
    hist = annual.loc[~annual["is_projected"]].copy().sort_values("year")
    proj = annual.loc[annual["is_projected"]].copy().sort_values("year")

    y_start = int(hist["year"].min())
    y_hist_end = int(hist["year"].max())
    y_proj_end = int(proj["year"].max()) if not proj.empty else y_hist_end

    row_hist = _find_row(annual, y_hist_end)
    row_proj = _find_row(annual, y_proj_end)

    c_hist = _safe_float(row_hist["cases_per10k_mid"])
    c_proj = _safe_float(row_proj["cases_per10k_mid"])
    c_delta_pct = ((c_proj - c_hist) / c_hist * 100.0) if np.isfinite(c_hist) and abs(c_hist) > 1e-12 else np.nan

    uv_hist = _safe_float(row_hist["effective_uv_kwh_m2_day"])
    uv_proj = _safe_float(row_proj["effective_uv_kwh_m2_day"])
    uv_delta_pct = ((uv_proj - uv_hist) / uv_hist * 100.0) if np.isfinite(uv_hist) and abs(uv_hist) > 1e-12 else np.nan

    tr_uv = _trend_row(trend, "effective_uv_kwh_m2_day")
    tr_case = _trend_row(trend, "cases_per10k_mid")

    fit_window_years = int(assumptions.get("projection_fit_window_years", 20))
    fit_start_effective = int(assumptions.get("projection_fit_start_effective", y_start))
    projection_damping = float(assumptions.get("projection_damping", 1.0))
    uv_lag_years = int(assumptions.get("uv_lag_years", 1))
    growth_cap = float(assumptions.get("projected_case_growth_cap_pct", 99.0))
    growth_floor = float(assumptions.get("projected_case_growth_floor_pct", -99.0))
    noise_enable = bool(assumptions.get("projection_noise_enable", False))
    noise_strength = float(assumptions.get("projection_noise_strength", 0.0))
    noise_ar1 = float(assumptions.get("projection_noise_ar1", 0.0))
    align_cases_with_solar = bool(assumptions.get("align_cases_with_solar", False))
    solar_parallel_weight = float(assumptions.get("solar_parallel_weight", 0.0))
    solar_parallel_eps_pct = float(assumptions.get("solar_parallel_eps_pct", 0.0))
    uv_parallel_weight = float(assumptions.get("uv_parallel_weight", 0.0))
    case_growth_inertia = float(assumptions.get("case_growth_inertia", 0.0))
    growth_guardrail_decay = float(assumptions.get("growth_guardrail_decay", 0.0))

    checks = checks.copy()
    checks["durum"] = np.where(checks["pass_flag"], "UYUMLU", "GOZDEN GECIR")
    checks["pass_flag"] = checks["pass_flag"].astype(bool)
    aligned_count = int(checks["pass_flag"].sum())
    total_checks = int(len(checks))

    proj_table = annual.loc[annual["year"] >= y_hist_end, ["year", "effective_uv_kwh_m2_day", "cases_per10k_low", "cases_per10k_mid", "cases_per10k_high"]].copy()
    proj_table = proj_table.rename(
        columns={
            "year": "Yil",
            "effective_uv_kwh_m2_day": "EtkiliUV_kWh_m2_gun",
            "cases_per10k_low": "Vaka10k_Dusuk",
            "cases_per10k_mid": "Vaka10k_Merkez",
            "cases_per10k_high": "Vaka10k_Yuksek",
        }
    )

    lit_table = literature[["evidence_id", "category", "exposure", "outcome", "quantitative_finding", "source_url"]].copy()
    lit_table = lit_table.rename(
        columns={
            "evidence_id": "Kanit_ID",
            "category": "Tur",
            "exposure": "Maruziyet",
            "outcome": "Sonuc",
            "quantitative_finding": "Ana_Bulgu",
            "source_url": "Kaynak",
        }
    )

    checks_table = checks[
        [
            "check_name",
            "evidence_type",
            "literature_expectation",
            "observed_metric",
            "observed_value",
            "pass_rule",
            "durum",
            "notes",
        ]
    ].copy()
    checks_table = checks_table.rename(
        columns={
            "check_name": "Kontrol",
            "evidence_type": "Tip",
            "literature_expectation": "Literatur_Beklentisi",
            "observed_metric": "Gozlenen_Metrik",
            "observed_value": "Deger",
            "pass_rule": "Kural",
            "durum": "Durum",
            "notes": "Not",
        }
    )

    relative = lambda p: p.name if p.parent == out_html.parent else p.relative_to(out_html.parent)  # noqa: E731
    fig_climate = relative((out_html.parent / "figures" / "yearly_climate_drivers.png").resolve())
    fig_cases = relative((out_html.parent / "figures" / "skin_cancer_per10k_trend.png").resolve())
    fig_evidence = relative((out_html.parent / "figures" / "evidence_dashboard.png").resolve())
    fig_strip = relative((out_html.parent / "figures" / "academic_projection_strip.png").resolve())

    refs_html = []
    for _, row in literature.iterrows():
        refs_html.append(
            "<li>"
            f"<b>{html.escape(str(row['evidence_id']))}</b>: "
            f"{html.escape(str(row['quantitative_finding']))} "
            f"<a href=\"{html.escape(str(row['source_url']))}\" target=\"_blank\" rel=\"noopener\">link</a>"
            "</li>"
        )

    data_refs: list[str] = []
    data_map = [
        ("station_table1", "Rasathane/Otomatik Istasyon (CR800 Table1)"),
        ("temp_local", "Yerel sicaklik tarihsel tablo"),
        ("humidity_local", "Yerel nem tarihsel tablo"),
        ("precip_local", "Yerel yagis tarihsel tablo"),
    ]
    for k, label in data_map:
        p = str(inputs.get(k, "")).strip()
        if not p:
            continue
        data_refs.append(
            f"<li><b>{html.escape(label)}</b>: <code>{html.escape(p)}</code></li>"
        )
    data_refs.append(
        "<li><b>NASA POWER API</b>: "
        "<a href=\"https://power.larc.nasa.gov/api/temporal/monthly/point\" target=\"_blank\" rel=\"noopener\">"
        "https://power.larc.nasa.gov/api/temporal/monthly/point</a></li>"
    )

    def _arr(series: pd.Series) -> list[float | None]:
        out: list[float | None] = []
        for x in series.to_list():
            v = _safe_float(x)
            out.append(v if np.isfinite(v) else None)
        return out

    plot_payload = {
        "year": [int(y) for y in annual["year"].to_list()],
        "uv": _arr(annual["effective_uv_kwh_m2_day"]),
        "cases_low": _arr(annual["cases_per10k_low"]),
        "cases_mid": _arr(annual["cases_per10k_mid"]),
        "cases_high": _arr(annual["cases_per10k_high"]),
        "is_projected": [bool(x) for x in annual["is_projected"].to_list()],
        "history_end": int(y_hist_end),
    }
    plot_payload_json = json.dumps(plot_payload, ensure_ascii=True)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    html_text = f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cilt Kanseri - Akademik Sunum Sayfasi</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #f8fbf8;
      --ink: #102026;
      --ink-soft: #47616b;
      --card: #ffffff;
      --line: #d3dee4;
      --ok: #15803d;
      --warn: #b45309;
      --hero1: #d8efe4;
      --hero2: #f8dfc8;
      --accent: #1d4ed8;
      --shadow: 0 14px 34px rgba(16,32,38,0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Source Sans 3", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 700px at 0% 0%, var(--hero1) 0%, transparent 64%),
        radial-gradient(900px 560px at 100% 0%, var(--hero2) 0%, transparent 60%),
        var(--bg);
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px 14px 38px;
    }}
    .hero {{
      background: rgba(255,255,255,0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 18px 18px 14px;
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0;
      font-family: "Space Grotesk", sans-serif;
      font-size: clamp(1.35rem, 3vw, 2.0rem);
    }}
    .sub {{
      color: var(--ink-soft);
      margin-top: 6px;
      font-size: 0.96rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow);
      padding: 12px;
    }}
    .span-12 {{ grid-column: span 12; }}
    .span-6 {{ grid-column: span 6; }}
    .span-4 {{ grid-column: span 4; }}
    h2 {{
      margin: 2px 2px 10px;
      font-family: "Space Grotesk", sans-serif;
      font-size: 1.02rem;
    }}
    .kpi {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .k {{
      border: 1px dashed var(--line);
      border-radius: 12px;
      padding: 8px 10px;
      background: #fbfdff;
    }}
    .k .l {{ color: var(--ink-soft); font-size: 0.78rem; }}
    .k .v {{ font-family: "Space Grotesk", sans-serif; font-size: 1.1rem; font-weight: 700; margin-top: 4px; }}
    .pill {{
      display: inline-block;
      margin-top: 8px;
      border-radius: 999px;
      padding: 6px 11px;
      font-size: 0.80rem;
      border: 1px solid #bfd5e2;
      background: #eaf2fb;
      color: #123a5c;
    }}
    .term-list {{
      margin: 0;
      padding-left: 18px;
      line-height: 1.45;
    }}
    .term-list li {{ margin-bottom: 8px; }}
    .img-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    figure {{
      margin: 0;
      border: 1px solid #e5ecef;
      border-radius: 12px;
      overflow: hidden;
      background: #fff;
    }}
    figure img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    figure figcaption {{
      font-size: 0.80rem;
      color: var(--ink-soft);
      padding: 8px 10px;
      border-top: 1px solid #edf2f5;
      background: #fbfdff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.86rem;
    }}
    th, td {{
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid #e4edf0;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #fff;
      z-index: 1;
      color: var(--ink-soft);
      text-transform: uppercase;
      font-size: 0.74rem;
      letter-spacing: 0.03em;
    }}
    .table-wrap {{
      max-height: 370px;
      overflow: auto;
      border: 1px solid #e4edf0;
      border-radius: 12px;
    }}
    .refs {{
      margin: 4px 0 0 18px;
      padding: 0;
      line-height: 1.35;
    }}
    .refs li {{ margin: 0 0 8px 0; }}
    .foot {{
      color: var(--ink-soft);
      font-size: 0.83rem;
      margin-top: 8px;
      line-height: 1.4;
    }}
    .reality-strip {{
      display: grid;
      grid-template-columns: repeat(10, minmax(0, 1fr));
      gap: 10px;
    }}
    .rs {{
      border: 1px solid #d7e2e8;
      border-radius: 12px;
      background: #f6fbff;
      padding: 8px 10px;
    }}
    .rs .t {{
      color: var(--ink-soft);
      font-size: 0.76rem;
    }}
    .rs .v {{
      font-family: "Space Grotesk", sans-serif;
      font-weight: 700;
      margin-top: 3px;
      font-size: 0.98rem;
    }}
    #interactive-traj {{
      width: 100%;
      height: 430px;
      border: 1px solid #e6edf1;
      border-radius: 12px;
      overflow: hidden;
    }}
    @media (max-width: 960px) {{
      .span-6, .span-4 {{ grid-column: span 12; }}
      .kpi {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .img-grid {{ grid-template-columns: 1fr; }}
      .reality-strip {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      #interactive-traj {{ height: 360px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Cilt Kanseri ve Iklim Suruculeri: Akademik Genisletilmis Sunum</h1>
      <div class="sub">
        Donem: {y_start}-{y_hist_end} tarihsel, {y_hist_end + 1}-{y_proj_end} tek-yontem projeksiyon.
        Uretim zamani: {generated_at}
      </div>
      <div class="pill">Model turu: Tek yontem + gercekcilik katmani (lag, damping, buyume siniri)</div>
    </section>

    <section class="card span-12">
      <h2>Yonetici Ozeti</h2>
      <div class="kpi">
        <div class="k"><div class="l">{y_hist_end} merkez vaka (10k)</div><div class="v">{_fmt(c_hist, 4)}</div></div>
        <div class="k"><div class="l">{y_proj_end} merkez vaka (10k)</div><div class="v">{_fmt(c_proj, 4)}</div></div>
        <div class="k"><div class="l">{y_hist_end}->{y_proj_end} vaka degisimi</div><div class="v">{_fmt_pct(c_delta_pct, 1)}</div></div>
        <div class="k"><div class="l">Literatur uyumu</div><div class="v">{aligned_count}/{total_checks}</div></div>
      </div>
      <div class="kpi" style="margin-top:10px;">
        <div class="k"><div class="l">Etkili UV degisimi ({y_hist_end}->{y_proj_end})</div><div class="v">{_fmt_pct(uv_delta_pct, 1)}</div></div>
        <div class="k"><div class="l">UV trend p-degeri</div><div class="v">{_fmt(tr_uv['p_value'], 4)}</div></div>
        <div class="k"><div class="l">Vaka trend p-degeri</div><div class="v">{_fmt(tr_case['p_value'], 4)}</div></div>
        <div class="k"><div class="l">UV trend yillik egim</div><div class="v">{_fmt(tr_uv['slope_per_year'], 4)}</div></div>
      </div>
    </section>

    <section class="card span-6">
      <h2>Terimler (Sade Dil)</h2>
      <ul class="term-list">
        <li><b>Etkili UV proxy:</b> Gunes enerjisinin, bulut etkisi cikarildiktan sonra cilde ulasma baskisini temsil eden gostergesi.</li>
        <li><b>Vaka / 10.000:</b> 10.000 kisilik bir nufusta bir yilda beklenen yeni vaka sayisi senaryosu.</li>
        <li><b>Belirsizlik bandi:</b> Tek bir sayi yerine dusuk-merkez-yuksek aralik vererek model belirsizligini gosteren bolge.</li>
        <li><b>p-degeri:</b> Trendin tesadufen gorulme olasiligi; 0.05 altinda ise istatistiksel kanit daha gucludur.</li>
        <li><b>Uyum kontrolu:</b> Modelin yonunun, WHO/IARC/PubMed literaturundeki yonle celisip celismedigini test eder.</li>
      </ul>
    </section>

    <section class="card span-6">
      <h2>Yontem Ozeti</h2>
      <ul class="term-list">
        <li>Yerel veri: sicaklik, nem, yagis yillik serileri; yeni otomatik istasyon verisi son yillari gunceller.</li>
        <li>Internet veri: NASA POWER API ile aylik gunes ve bulut verileri alinip yilliga cevrildi.</li>
        <li>Etkili UV proxy = <code>solar_kwh_m2_day x (1 - cloud_pct/100)</code>.</li>
        <li>2026-2035 icin tek model: son yillara agirlikli Theil-Sen trend uzatimi (dayanikli egim).</li>
        <li>Risk senaryosu WHO/IARC yon bilgisi ve acik varsayimlarla 10.000 kisiye normalize edildi.</li>
      </ul>
      <div class="foot">
        Not: Bu sayfa klinik tani araci degildir; toplum duzeyi risk-baski senaryosu sunar.
      </div>
    </section>

    <section class="card span-12">
      <h2>Gercekcilik Guvenceleri</h2>
      <div class="reality-strip">
        <div class="rs"><div class="t">Fit penceresi</div><div class="v">Son {fit_window_years} yil ({fit_start_effective}+)</div></div>
        <div class="rs"><div class="t">Projeksiyon damping</div><div class="v">{projection_damping:.2f}</div></div>
        <div class="rs"><div class="t">UV gecikme penceresi</div><div class="v">{uv_lag_years} yil</div></div>
        <div class="rs"><div class="t">Gurultu aktif</div><div class="v">{"Evet" if noise_enable else "Hayir"}</div></div>
        <div class="rs"><div class="t">Gurultu gucu</div><div class="v">{noise_strength:.2f}</div></div>
        <div class="rs"><div class="t">Gurultu AR(1)</div><div class="v">{noise_ar1:.2f}</div></div>
        <div class="rs"><div class="t">Gunes paralellik</div><div class="v">{"Evet" if align_cases_with_solar else "Hayir"}</div></div>
        <div class="rs"><div class="t">Paralellik agirlik</div><div class="v">{solar_parallel_weight:.2f}</div></div>
        <div class="rs"><div class="t">Paralellik eps</div><div class="v">{solar_parallel_eps_pct:.3f}%</div></div>
        <div class="rs"><div class="t">UV agirlik</div><div class="v">{uv_parallel_weight:.2f}</div></div>
        <div class="rs"><div class="t">Buyume ataleti</div><div class="v">{case_growth_inertia:.2f}</div></div>
        <div class="rs"><div class="t">Guardrail decay</div><div class="v">{growth_guardrail_decay:.3f}</div></div>
        <div class="rs"><div class="t">Yillik artis tavani</div><div class="v">{growth_cap:.2f}%</div></div>
        <div class="rs"><div class="t">Yillik dusus tabani</div><div class="v">{growth_floor:.2f}%</div></div>
      </div>
      <div class="foot">
        Amac: uzun donem uzatmalarda asiri hizli artislari engellemek ve tahmini daha muhafazakar bir patikaya cekmek.
      </div>
    </section>

    <section class="card span-12">
      <h2>Interaktif Gelisim Grafiği</h2>
      <div id="interactive-traj"></div>
      <div class="foot">
        Turuncu alan belirsizlik bandini, kirmizi cizgi 10.000 kiside merkez senaryoyu gosterir. Dikey cizgi tarihsel/projeksiyon ayrimidir.
      </div>
    </section>

    <section class="card span-12">
      <h2>Gorsel Bulgular</h2>
      <div class="img-grid">
        <figure>
          <img src="{html.escape(str(fig_strip))}" alt="Yillik UV ve cilt kanseri proxy gelisimi">
          <figcaption>Yillik UV baskisi ile 10.000 kiside vaka senaryosunun birlikte gelisimi</figcaption>
        </figure>
        <figure>
          <img src="{html.escape(str(fig_climate))}" alt="Yillik iklim suruculeri">
          <figcaption>Gunes, bulut, yagis, nem ve sicaklik suruculerinin yillik patikasi</figcaption>
        </figure>
        <figure>
          <img src="{html.escape(str(fig_cases))}" alt="10.000 kiside cilt kanseri trendi">
          <figcaption>Merkez senaryo ve belirsizlik bandi (10.000 kisi bazinda)</figcaption>
        </figure>
        <figure>
          <img src="{html.escape(str(fig_evidence))}" alt="Istatistiksel kanit panosu">
          <figcaption>Trend gucu ve tarihsel korelasyon panosu</figcaption>
        </figure>
      </div>
    </section>

    <section class="card span-12">
      <h2>Literatur Kanit Tablosu (Akademik Genisletilmis)</h2>
      <div class="table-wrap">{_table_html(lit_table, classes="lit-table")}</div>
    </section>

    <section class="card span-12">
      <h2>Model-Literatur Celiski Kontrolu</h2>
      <div class="table-wrap">{_table_html(checks_table, classes="check-table")}</div>
      <div class="foot">
        Yorum: {aligned_count}/{total_checks} kontrol "UYUMLU". "GOZDEN GECIR" satirlari model varsayimi veya veri siniri kaynakli olabilir ve bir sonraki iterasyonda iyilestirilmelidir.
      </div>
    </section>

    <section class="card span-12">
      <h2>Yillara Gore Gelisim (Ozet Tablo)</h2>
      <div class="table-wrap">{_table_html(proj_table, classes="proj-table")}</div>
    </section>

    <section class="card span-12">
      <h2>Kaynaklar</h2>
      <ol class="refs">
        {"".join(refs_html)}
      </ol>
      <div class="foot">
        Sinirlar: UV-B dogrudan olculmedi, etkili UV proxy kullanildi. Risk sayilari nedensel kesinlik degil, literatur-uyumlu senaryo hesaplaridir.
      </div>
    </section>

    <section class="card span-12">
      <h2>Kaynakca (Veri + Akademik)</h2>
      <h3 style="margin:2px 0 8px; font-size:0.96rem; color:#334155;">Veri Kaynaklari</h3>
      <ul class="refs">
        {"".join(data_refs)}
      </ul>
      <h3 style="margin:12px 0 8px; font-size:0.96rem; color:#334155;">Akademik ve Kurumsal Kaynaklar</h3>
      <ol class="refs">
        {"".join(refs_html)}
      </ol>
      <div class="foot">
        Not: Veri kaynaklari bu calismada kullanilan dosya/API girdilerini; akademik kaynaklar ise model yonu ve etki varsayimlarinin dayanaklarini gosterir.
      </div>
    </section>
  </div>
  <script>
    (function() {{
      const p = {plot_payload_json};
      const years = p.year;
      const bandUpper = {{
        x: years,
        y: p.cases_high,
        mode: "lines",
        line: {{width: 0}},
        hoverinfo: "skip",
        showlegend: false
      }};
      const bandLower = {{
        x: years,
        y: p.cases_low,
        mode: "lines",
        line: {{width: 0}},
        fill: "tonexty",
        fillcolor: "rgba(239, 68, 68, 0.20)",
        name: "Belirsizlik bandi",
        hovertemplate: "Yil %{{x}}<br>Alt: %{{y:.4f}}<extra></extra>"
      }};
      const midCases = {{
        x: years,
        y: p.cases_mid,
        mode: "lines+markers",
        name: "Vaka/10k (merkez)",
        line: {{color: "#dc2626", width: 3}},
        marker: {{size: 5}},
        hovertemplate: "Yil %{{x}}<br>Merkez: %{{y:.4f}}<extra></extra>"
      }};
      const uvTrace = {{
        x: years,
        y: p.uv,
        mode: "lines",
        name: "Etkili UV",
        yaxis: "y2",
        line: {{color: "#f59e0b", width: 2.3, dash: "dot"}},
        hovertemplate: "Yil %{{x}}<br>Etkili UV: %{{y:.4f}}<extra></extra>"
      }};
      const layout = {{
        margin: {{l: 50, r: 52, t: 18, b: 38}},
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "#ffffff",
        legend: {{orientation: "h", x: 0, y: 1.12}},
        xaxis: {{
          title: "Yil",
          showgrid: true,
          gridcolor: "#edf2f7",
          zeroline: false
        }},
        yaxis: {{
          title: "Vaka / 10.000",
          showgrid: true,
          gridcolor: "#edf2f7",
          zeroline: false
        }},
        yaxis2: {{
          title: "Etkili UV (kWh/m2/gun)",
          overlaying: "y",
          side: "right",
          showgrid: false,
          zeroline: false
        }},
        shapes: [{{
          type: "line",
          x0: p.history_end + 0.5,
          x1: p.history_end + 0.5,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: {{color: "#64748b", width: 1.4, dash: "dash"}}
        }}],
        annotations: [{{
          x: p.history_end + 0.6,
          y: 1.04,
          yref: "paper",
          text: "Projeksiyon baslangici",
          showarrow: false,
          font: {{size: 11, color: "#475569"}}
        }}]
      }};
      Plotly.newPlot("interactive-traj", [bandUpper, bandLower, midCases, uvTrace], layout, {{
        responsive: true,
        displayModeBar: false
      }});
    }})();
  </script>
</body>
</html>
"""
    out_html.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.input_dir.resolve()
    out_html = args.output_html.resolve() if args.output_html else (root / "akademik_sunum_sayfasi.html")

    annual_csv = _load_required(root / "annual_climate_and_skin_cancer_proxy.csv")
    trend_csv = _load_required(root / "trend_proof_table.csv")
    corr_csv = _load_required(root / "historical_correlation_matrix.csv")
    meta_json = _load_required(root / "run_meta.json")

    annual = pd.read_csv(annual_csv).copy()
    trend = pd.read_csv(trend_csv).copy()
    corr = pd.read_csv(corr_csv, index_col=0).copy()
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    assumptions = dict(meta.get("assumptions", {}))
    inputs = dict(meta.get("inputs", {}))

    annual["year"] = pd.to_numeric(annual["year"], errors="coerce").astype("Int64")
    annual = annual.dropna(subset=["year"]).copy()
    annual["year"] = annual["year"].astype(int)

    if "is_projected" not in annual.columns:
        raise SystemExit("annual table missing is_projected column")
    annual["is_projected"] = annual["is_projected"].astype(str).str.lower().map({"true": True, "false": False})
    if annual["is_projected"].isna().any():
        annual["is_projected"] = annual["year"] > int(annual["year"].max())  # defensive fallback
    annual = annual.sort_values("year").reset_index(drop=True)

    literature = build_literature_table()
    checks = build_alignment_checks(annual=annual, trend=trend, corr=corr)

    lit_csv = root / "academic_literature_table.csv"
    checks_csv = root / "literature_alignment_checks.csv"
    fig_strip = root / "figures" / "academic_projection_strip.png"

    literature.to_csv(lit_csv, index=False)
    checks.to_csv(checks_csv, index=False)
    plot_projection_strip(annual=annual, out_png=fig_strip)
    build_html(
        annual=annual,
        trend=trend,
        literature=literature,
        checks=checks,
        assumptions=assumptions,
        inputs=inputs,
        out_html=out_html,
    )

    print(f"Wrote: {lit_csv}")
    print(f"Wrote: {checks_csv}")
    print(f"Wrote: {fig_strip}")
    print(f"Wrote: {out_html}")


if __name__ == "__main__":
    main()
