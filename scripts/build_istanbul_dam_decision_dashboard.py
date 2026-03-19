#!/usr/bin/env python3
"""Build a self-contained HTML dashboard for Istanbul dam decision outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Istanbul dam decision dashboard")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision"),
    )
    p.add_argument(
        "--output-html",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/dashboard.html"),
    )
    return p.parse_args()


def to_records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


def build_html(payload_json: str) -> str:
    return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Istanbul Baraj Karar Destek Paneli</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg1: #f3f7f0;
      --bg2: #e5edf3;
      --ink: #102230;
      --ink-soft: #3f5566;
      --card: #ffffff;
      --accent: #0f766e;
      --accent2: #c2410c;
      --line: #c6d2dc;
      --good: #15803d;
      --warn: #b45309;
      --risk: #b91c1c;
      --shadow: 0 16px 40px rgba(16, 34, 48, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 700px at -10% -10%, #d7ece6 0%, transparent 65%),
        radial-gradient(1000px 600px at 110% 0%, #f5d9be 0%, transparent 60%),
        linear-gradient(130deg, var(--bg1), var(--bg2));
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 16px 28px;
    }}
    .hero {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0;
      font-family: "Space Grotesk", sans-serif;
      font-size: clamp(1.4rem, 3.4vw, 2.2rem);
      letter-spacing: 0.01em;
    }}
    .meta {{
      color: var(--ink-soft);
      font-size: 0.92rem;
      margin-top: 6px;
    }}
    .badge {{
      background: #e1f2ef;
      color: #0a4a45;
      border: 1px solid #b9dfd9;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 0.84rem;
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(12, 1fr);
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow);
      padding: 12px;
    }}
    .card h2 {{
      margin: 4px 4px 10px;
      font-size: 1rem;
      font-family: "Space Grotesk", sans-serif;
      letter-spacing: 0.01em;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }}
    select {{
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      background: #fff;
      color: var(--ink);
      min-width: 220px;
    }}
    .kpi {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(4, minmax(100px, 1fr));
      margin-top: 8px;
    }}
    .k {{
      border: 1px dashed var(--line);
      border-radius: 12px;
      padding: 8px 10px;
      background: #fbfdff;
    }}
    .k .l {{
      color: var(--ink-soft);
      font-size: 0.78rem;
    }}
    .k .v {{
      margin-top: 3px;
      font-size: 1.05rem;
      font-weight: 600;
      font-family: "Space Grotesk", sans-serif;
    }}
    #chart-main, #chart-prob {{
      width: 100%;
      height: 380px;
    }}
    #chart-prob {{
      height: 300px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    th, td {{
      text-align: left;
      padding: 8px 8px;
      border-bottom: 1px solid #e4ecf2;
    }}
    th {{
      color: var(--ink-soft);
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      position: sticky;
      top: 0;
      background: #fff;
      z-index: 1;
    }}
    .risk-high {{ color: var(--risk); font-weight: 600; }}
    .risk-mid {{ color: var(--warn); font-weight: 600; }}
    .risk-low {{ color: var(--good); font-weight: 600; }}
    .table-wrap {{
      max-height: 420px;
      overflow: auto;
      border: 1px solid #e4ecf2;
      border-radius: 12px;
    }}
    .span-12 {{ grid-column: span 12; }}
    .span-8 {{ grid-column: span 8; }}
    .span-4 {{ grid-column: span 4; }}
    @media (max-width: 920px) {{
      .span-8, .span-4 {{ grid-column: span 12; }}
      .kpi {{ grid-template-columns: repeat(2, minmax(100px, 1fr)); }}
      #chart-main {{ height: 340px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <h1>Istanbul Baraj Karar Destek Paneli</h1>
        <div class="meta" id="meta-line">Veri yukleniyor...</div>
      </div>
      <div class="badge">2026-03 -> 2027-02 risk odagi</div>
    </div>

    <div class="grid">
      <section class="card span-8">
        <h2>Seri Bazli Tahmin ve Belirsizlik</h2>
        <div class="controls">
          <label for="seriesSel">Seri:</label>
          <select id="seriesSel"></select>
          <span id="strategyText" class="meta"></span>
        </div>
        <div id="chart-main"></div>
        <div class="kpi">
          <div class="k"><div class="l">2026-03..2027-02 Ortalama</div><div class="v" id="k_mean">-</div></div>
          <div class="k"><div class="l">Minimum</div><div class="v" id="k_min">-</div></div>
          <div class="k"><div class="l">%40 Alti Ortalama Olasilik</div><div class="v" id="k_p40">-</div></div>
          <div class="k"><div class="l">%30 Alti Ortalama Olasilik</div><div class="v" id="k_p30">-</div></div>
        </div>
      </section>

      <section class="card span-4">
        <h2>Risk Olasiligi (Aylik)</h2>
        <div id="chart-prob"></div>
      </section>

      <section class="card span-12">
        <h2>Risk Siralamasi (2026-03 -> 2027-02)</h2>
        <div class="table-wrap">
          <table id="risk-table">
            <thead>
              <tr>
                <th>Seri</th>
                <th>Strateji</th>
                <th>&lt;40 Ay</th>
                <th>&lt;30 Ay</th>
                <th>Ort. P(&lt;40)</th>
                <th>Ort. P(&lt;30)</th>
                <th>Ort. Tahmin</th>
                <th>En Kotu Ay</th>
                <th>En Kotu Tahmin</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </section>
    </div>
  </div>

  <script>
    const payload = {payload_json};
    const history = payload.history;
    const forecast = payload.forecast;
    const risk = payload.risk;
    const strategies = payload.strategies;
    const summary = payload.summary || {{}};

    const sel = document.getElementById("seriesSel");
    const strategyText = document.getElementById("strategyText");

    const seriesList = [...new Set(forecast.map(d => d.series))].sort((a, b) => a.localeCompare(b));
    const preferred = "overall_mean";
    seriesList.forEach(s => {{
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s;
      sel.appendChild(opt);
    }});
    if (seriesList.includes(preferred)) sel.value = preferred;

    const strategyMap = Object.fromEntries(strategies.map(r => [r.series, r]));
    const histBySeries = Object.fromEntries(seriesList.map(s => [s, history.map(h => ({{ds:h.ds, y:h[s]}})).filter(x => x.y != null)]));
    const fcBySeries = Object.fromEntries(seriesList.map(s => [s, forecast.filter(f => f.series === s)]));

    function pct(v) {{
      return (v * 100).toFixed(1) + "%";
    }}

    function renderRiskTable() {{
      const tbody = document.querySelector("#risk-table tbody");
      tbody.innerHTML = "";
      risk.forEach(r => {{
        const tr = document.createElement("tr");
        const cls = r.months_lt40 >= 4 ? "risk-high" : (r.months_lt40 >= 2 ? "risk-mid" : "risk-low");
        tr.innerHTML = `
          <td>${{r.series}}</td>
          <td>${{r.strategy}}</td>
          <td class="${{cls}}">${{r.months_lt40}}</td>
          <td>${{r.months_lt30}}</td>
          <td>${{r.mean_prob_below_40_pct.toFixed(1)}}%</td>
          <td>${{r.mean_prob_below_30_pct.toFixed(1)}}%</td>
          <td>${{r.mean_yhat_pct.toFixed(1)}}%</td>
          <td>${{r.worst_month}}</td>
          <td>${{r.worst_yhat_pct.toFixed(1)}}%</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function computeKpi(fcRows) {{
      const n12 = fcRows.filter(r => r.ds >= "2026-03-01" && r.ds <= "2027-02-01");
      if (!n12.length) return null;
      const meanY = n12.reduce((a,b)=>a+b.yhat,0)/n12.length;
      const minY = Math.min(...n12.map(x=>x.yhat));
      const p40 = n12.reduce((a,b)=>a+b.prob_below_40,0)/n12.length;
      const p30 = n12.reduce((a,b)=>a+b.prob_below_30,0)/n12.length;
      return {{meanY, minY, p40, p30}};
    }}

    function renderSeries(series) {{
      const h = histBySeries[series] || [];
      const f = fcBySeries[series] || [];
      const st = strategyMap[series];
      strategyText.textContent = st ? `Strateji: ${{st.strategy}} | CV RMSE: ${{Number(st.strategy_rmse).toFixed(4)}}` : "";

      const trHist = {{
        x: h.map(d => d.ds),
        y: h.map(d => d.y * 100),
        mode: "lines",
        name: "Gerceklesen",
        line: {{color: "#1f77b4", width: 2}}
      }};
      const trFc = {{
        x: f.map(d => d.ds),
        y: f.map(d => d.yhat * 100),
        mode: "lines",
        name: "Tahmin",
        line: {{color: "#c2410c", width: 2, dash: "dash"}}
      }};
      const trUpper = {{
        x: f.map(d => d.ds),
        y: f.map(d => d.yhat_upper * 100),
        mode: "lines",
        line: {{color: "rgba(194,65,12,0.0)"}},
        showlegend: false,
        hoverinfo: "skip"
      }};
      const trLower = {{
        x: f.map(d => d.ds),
        y: f.map(d => d.yhat_lower * 100),
        mode: "lines",
        fill: "tonexty",
        fillcolor: "rgba(194,65,12,0.18)",
        line: {{color: "rgba(194,65,12,0.0)"}},
        name: "90% aralik",
        hoverinfo: "skip"
      }};

      Plotly.newPlot("chart-main", [trHist, trFc, trUpper, trLower], {{
        margin: {{l: 52, r: 18, t: 8, b: 45}},
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: {{gridcolor: "#dde6ee"}},
        yaxis: {{title: "Doluluk (%)", gridcolor: "#dde6ee", range: [0, 100]}},
        legend: {{orientation: "h", y: 1.08}}
      }}, {{displayModeBar: false, responsive: true}});

      const p40 = {{
        x: f.map(d => d.ds),
        y: f.map(d => d.prob_below_40 * 100),
        type: "bar",
        name: "P(<40%)",
        marker: {{color: "#b45309"}}
      }};
      const p30 = {{
        x: f.map(d => d.ds),
        y: f.map(d => d.prob_below_30 * 100),
        type: "bar",
        name: "P(<30%)",
        marker: {{color: "#b91c1c"}}
      }};
      Plotly.newPlot("chart-prob", [p40, p30], {{
        barmode: "group",
        margin: {{l: 52, r: 18, t: 8, b: 45}},
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: {{gridcolor: "#e4ecf2"}},
        yaxis: {{title: "Olasilik (%)", range: [0,100], gridcolor: "#e4ecf2"}},
        legend: {{orientation: "h", y: 1.1}}
      }}, {{displayModeBar: false, responsive: true}});

      const k = computeKpi(f);
      if (k) {{
        document.getElementById("k_mean").textContent = (k.meanY*100/100).toFixed(1) + "%";
        document.getElementById("k_min").textContent = (k.minY*100/100).toFixed(1) + "%";
        document.getElementById("k_p40").textContent = (k.p40*100/100).toFixed(1) + "%";
        document.getElementById("k_p30").textContent = (k.p30*100/100).toFixed(1) + "%";
      }}
    }}

    document.getElementById("meta-line").textContent =
      `Veri: ${{summary.monthly_start || "?"}} -> ${{summary.monthly_end || "?"}} | Tahmin: ${{summary.forecast_start || "?"}} -> ${{summary.forecast_end || "?"}}`;

    renderRiskTable();
    renderSeries(sel.value);
    sel.addEventListener("change", () => renderSeries(sel.value));
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    in_dir = args.input_dir

    history_path = in_dir / "istanbul_dam_monthly_history.csv"
    forecast_path = in_dir / "istanbul_dam_forecasts_decision.csv"
    risk_path = in_dir / "risk_summary_2026_03_to_2027_02.csv"
    strategy_path = in_dir / "strategy_summary.csv"
    summary_path = in_dir / "run_summary.json"

    if not history_path.exists() or not forecast_path.exists() or not risk_path.exists() or not strategy_path.exists():
        raise SystemExit("Missing required decision output files. Run forecast_istanbul_dam_decision_support.py first.")

    history = pd.read_csv(history_path, parse_dates=["ds"])
    forecast = pd.read_csv(forecast_path, parse_dates=["ds"])
    risk = pd.read_csv(risk_path)
    strategies = pd.read_csv(strategy_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    payload = {
        "history": to_records(history),
        "forecast": to_records(forecast),
        "risk": to_records(risk),
        "strategies": to_records(strategies),
        "summary": summary,
    }
    html = build_html(json.dumps(payload, ensure_ascii=False))
    args.output_html.write_text(html, encoding="utf-8")
    print(args.output_html)


if __name__ == "__main__":
    main()

