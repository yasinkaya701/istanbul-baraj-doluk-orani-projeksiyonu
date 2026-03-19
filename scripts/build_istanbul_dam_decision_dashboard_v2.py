#!/usr/bin/env python3
"""Build scenario-aware decision dashboard (v2) for Istanbul dams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Istanbul decision dashboard v2 (scenario aware)")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision"),
    )
    p.add_argument(
        "--output-html",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/dashboard_v2.html"),
    )
    return p.parse_args()


def to_records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


def build_html(payload_json: str) -> str:
    return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Istanbul Baraj Karar Destek Paneli v2</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg1: #f7faf5;
      --bg2: #e7eef5;
      --ink: #0f2233;
      --ink-soft: #456071;
      --card: #ffffff;
      --line: #cbd7e2;
      --good: #15803d;
      --warn: #b45309;
      --risk: #b91c1c;
      --shadow: 0 14px 36px rgba(15,34,51,0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 700px at 0% -5%, #d9ece6 0%, transparent 64%),
        radial-gradient(900px 540px at 100% 0%, #f2ddc6 0%, transparent 62%),
        linear-gradient(130deg, var(--bg1), var(--bg2));
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 26px 16px 30px;
    }}
    .top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 16px;
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0;
      font-family: "Space Grotesk", sans-serif;
      font-size: clamp(1.4rem, 3.1vw, 2.1rem);
    }}
    .meta {{
      color: var(--ink-soft);
      font-size: 0.92rem;
      margin-top: 6px;
    }}
    .badge {{
      border: 1px solid #b5ddd7;
      background: #e5f4f1;
      color: #0c4c45;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 0.84rem;
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 14px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow);
      padding: 12px;
    }}
    .span-8 {{ grid-column: span 8; }}
    .span-4 {{ grid-column: span 4; }}
    .span-12 {{ grid-column: span 12; }}
    h2 {{
      margin: 4px 4px 10px;
      font-size: 1rem;
      font-family: "Space Grotesk", sans-serif;
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
      min-width: 200px;
    }}
    #chart-main {{ width: 100%; height: 390px; }}
    #chart-prob {{ width: 100%; height: 300px; }}
    .kpi {{
      display: grid;
      grid-template-columns: repeat(4, minmax(100px,1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .k {{
      background: #fbfdff;
      border: 1px dashed var(--line);
      border-radius: 12px;
      padding: 8px 10px;
    }}
    .k .l {{ font-size: 0.78rem; color: var(--ink-soft); }}
    .k .v {{ font-size: 1.05rem; font-weight: 600; margin-top: 3px; font-family: "Space Grotesk", sans-serif; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    th, td {{
      text-align: left;
      padding: 8px;
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
    .table-wrap {{
      border: 1px solid #e4ecf2;
      border-radius: 12px;
      max-height: 430px;
      overflow: auto;
    }}
    .risk-high {{ color: var(--risk); font-weight: 600; }}
    .risk-mid {{ color: var(--warn); font-weight: 600; }}
    .risk-low {{ color: var(--good); font-weight: 600; }}
    @media (max-width: 960px) {{
      .span-8, .span-4 {{ grid-column: span 12; }}
      .kpi {{ grid-template-columns: repeat(2, minmax(100px,1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div>
        <h1>Istanbul Baraj Karar Destek Paneli v2</h1>
        <div class="meta" id="metaLine">Veri yukleniyor...</div>
      </div>
      <div class="badge">Senaryo: baz + dry/wet siddet seviyeleri</div>
    </div>

    <div class="grid">
      <section class="card span-8">
        <h2>Seri Tahmini (Senaryo Secimli)</h2>
        <div class="controls">
          <label for="scenarioSel">Senaryo:</label>
          <select id="scenarioSel"></select>
          <label for="riskSel">Risk Filtresi:</label>
          <select id="riskSel"></select>
          <label for="seriesSel">Seri:</label>
          <select id="seriesSel"></select>
          <span class="meta" id="strategyText"></span>
        </div>
        <div id="chart-main"></div>
        <div class="kpi">
          <div class="k"><div class="l">2026-03..2027-02 Ortalama</div><div class="v" id="k_mean">-</div></div>
          <div class="k"><div class="l">Minimum</div><div class="v" id="k_min">-</div></div>
          <div class="k"><div class="l">%40 Alti Ort. Olasilik</div><div class="v" id="k_p40">-</div></div>
          <div class="k"><div class="l">%30 Alti Ort. Olasilik</div><div class="v" id="k_p30">-</div></div>
        </div>
      </section>

      <section class="card span-4">
        <h2>Aylik Risk Olasiliklari</h2>
        <div id="chart-prob"></div>
      </section>

      <section class="card span-12">
        <h2>Risk Siralamasi (Secili Senaryo)</h2>
        <div class="table-wrap">
          <table id="riskTable">
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
    const history = payload.history || [];
    const scenarios = payload.scenarios || [];
    const risk = payload.scenarioRisk || [];
    const strategies = payload.strategies || [];
    const summary = payload.summary || {{}};

    const scenarioSel = document.getElementById("scenarioSel");
    const riskSel = document.getElementById("riskSel");
    const seriesSel = document.getElementById("seriesSel");
    const strategyText = document.getElementById("strategyText");

    const scenarioList = [...new Set(scenarios.map(r => r.scenario))];
    function scenarioRank(s) {{
      if (s === "baseline") return [0, 0, s];
      if (s.startsWith("dry_")) {{
        const label = s.split("_")[1] || "";
        const ord = {{mild:1, base:2, stress:2, severe:3, extreme:4}}[label] ?? 9;
        return [1, ord, s];
      }}
      if (s.startsWith("wet_")) {{
        const label = s.split("_")[1] || "";
        const ord = {{mild:1, base:2, relief:2, severe:3, extreme:4}}[label] ?? 9;
        return [2, ord, s];
      }}
      return [3, 99, s];
    }}
    scenarioList.sort((a,b) => {{
      const ra = scenarioRank(a), rb = scenarioRank(b);
      if (ra[0] !== rb[0]) return ra[0] - rb[0];
      if (ra[1] !== rb[1]) return ra[1] - rb[1];
      return String(ra[2]).localeCompare(String(rb[2]));
    }});
    scenarioList.forEach(s => {{
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s;
      scenarioSel.appendChild(opt);
    }});
    if (scenarioList.includes("baseline")) scenarioSel.value = "baseline";

    const riskModes = [
      {{value: "all", label: "all"}},
      {{value: "high", label: "high"}},
      {{value: "medium", label: "medium"}},
      {{value: "low", label: "low"}}
    ];
    riskModes.forEach(r => {{
      const opt = document.createElement("option");
      opt.value = r.value;
      opt.textContent = r.label;
      riskSel.appendChild(opt);
    }});
    riskSel.value = "all";

    const seriesList = [...new Set(scenarios.map(r => r.series))].sort((a,b)=>a.localeCompare(b));

    const strategyMap = Object.fromEntries(strategies.map(r => [r.series, r]));
    const histBySeries = Object.fromEntries(seriesList.map(s => [s, history.map(h => ({{ds:h.ds, y:h[s]}})).filter(v => v.y != null)]));

    function riskLevel(r) {{
      if (r.months_lt40 >= 4) return "high";
      if (r.months_lt40 >= 2) return "medium";
      return "low";
    }}

    function riskClass(r) {{
      const lvl = riskLevel(r);
      if (lvl === "high") return "risk-high";
      if (lvl === "medium") return "risk-mid";
      return "risk-low";
    }}

    function filteredRiskRows(scenario) {{
      const mode = riskSel.value || "all";
      const rows = risk.filter(r => r.scenario === scenario);
      if (mode === "all") return rows;
      return rows.filter(r => riskLevel(r) === mode);
    }}

    function syncSeriesOptions() {{
      const scenario = scenarioSel.value;
      const rows = filteredRiskRows(scenario);
      const allowed = rows.map(r => r.series);
      const list = (allowed.length ? [...new Set(allowed)] : [...seriesList]).sort((a,b)=>a.localeCompare(b));
      const prev = seriesSel.value;
      seriesSel.innerHTML = "";
      list.forEach(s => {{
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = s;
        seriesSel.appendChild(opt);
      }});
      if (list.includes(prev)) {{
        seriesSel.value = prev;
      }} else if (list.includes("overall_mean")) {{
        seriesSel.value = "overall_mean";
      }} else if (list.length) {{
        seriesSel.value = list[0];
      }}
    }}

    function renderRiskTable(scenario) {{
      const rows = filteredRiskRows(scenario);
      const tbody = document.querySelector("#riskTable tbody");
      tbody.innerHTML = "";
      rows.forEach(r => {{
        const tr = document.createElement("tr");
        const cls = riskClass(r);
        tr.innerHTML = `
          <td>${{r.series}}</td>
          <td>${{r.strategy}}</td>
          <td class="${{cls}}">${{r.months_lt40}}</td>
          <td>${{r.months_lt30}}</td>
          <td>${{Number(r.mean_prob_below_40_pct).toFixed(1)}}%</td>
          <td>${{Number(r.mean_prob_below_30_pct).toFixed(1)}}%</td>
          <td>${{Number(r.mean_yhat_pct).toFixed(1)}}%</td>
          <td>${{r.worst_month}}</td>
          <td>${{Number(r.worst_yhat_pct).toFixed(1)}}%</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function currentRows() {{
      const scenario = scenarioSel.value;
      const series = seriesSel.value;
      return scenarios.filter(r => r.scenario === scenario && r.series === series);
    }}

    function renderCharts() {{
      const scenario = scenarioSel.value;
      const series = seriesSel.value;
      const h = histBySeries[series] || [];
      const f = currentRows();

      const st = strategyMap[series];
      strategyText.textContent = st ? `Strateji: ${{st.strategy}} | CV RMSE: ${{Number(st.strategy_rmse).toFixed(4)}}` : "";

      Plotly.newPlot("chart-main", [
        {{
          x: h.map(d => d.ds),
          y: h.map(d => d.y * 100),
          mode: "lines",
          name: "Gerceklesen",
          line: {{color: "#1f77b4", width: 2}}
        }},
        {{
          x: f.map(d => d.ds),
          y: f.map(d => d.scenario_yhat * 100),
          mode: "lines",
          name: `Tahmin (${{scenario}})`,
          line: {{color: "#c2410c", width: 2, dash: "dash"}}
        }},
        {{
          x: f.map(d => d.ds),
          y: f.map(d => d.scenario_yhat_upper * 100),
          mode: "lines",
          line: {{color: "rgba(194,65,12,0.0)"}},
          showlegend: false,
          hoverinfo: "skip"
        }},
        {{
          x: f.map(d => d.ds),
          y: f.map(d => d.scenario_yhat_lower * 100),
          mode: "lines",
          fill: "tonexty",
          fillcolor: "rgba(194,65,12,0.18)",
          line: {{color: "rgba(194,65,12,0.0)"}},
          name: "90% aralik",
          hoverinfo: "skip"
        }}
      ], {{
        margin: {{l: 52, r: 18, t: 8, b: 45}},
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: {{gridcolor: "#dde6ee"}},
        yaxis: {{title: "Doluluk (%)", range: [0, 100], gridcolor: "#dde6ee"}},
        legend: {{orientation: "h", y: 1.08}}
      }}, {{displayModeBar: false, responsive: true}});

      Plotly.newPlot("chart-prob", [
        {{
          x: f.map(d => d.ds),
          y: f.map(d => d.scenario_prob_below_40 * 100),
          type: "bar",
          name: "P(<40%)",
          marker: {{color: "#b45309"}}
        }},
        {{
          x: f.map(d => d.ds),
          y: f.map(d => d.scenario_prob_below_30 * 100),
          type: "bar",
          name: "P(<30%)",
          marker: {{color: "#b91c1c"}}
        }}
      ], {{
        barmode: "group",
        margin: {{l: 52, r: 18, t: 8, b: 45}},
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: {{gridcolor: "#e4ecf2"}},
        yaxis: {{title: "Olasilik (%)", range: [0, 100], gridcolor: "#e4ecf2"}},
        legend: {{orientation: "h", y: 1.1}}
      }}, {{displayModeBar: false, responsive: true}});

      const n12 = f.filter(r => r.ds >= "2026-03-01" && r.ds <= "2027-02-01");
      if (n12.length) {{
        const mean = n12.reduce((a,b) => a + b.scenario_yhat, 0) / n12.length;
        const min = Math.min(...n12.map(x => x.scenario_yhat));
        const p40 = n12.reduce((a,b)=>a+b.scenario_prob_below_40,0) / n12.length;
        const p30 = n12.reduce((a,b)=>a+b.scenario_prob_below_30,0) / n12.length;
        document.getElementById("k_mean").textContent = (mean*100).toFixed(1) + "%";
        document.getElementById("k_min").textContent = (min*100).toFixed(1) + "%";
        document.getElementById("k_p40").textContent = (p40*100).toFixed(1) + "%";
        document.getElementById("k_p30").textContent = (p30*100).toFixed(1) + "%";
      }}
    }}

    function refresh() {{
      syncSeriesOptions();
      renderRiskTable(scenarioSel.value);
      renderCharts();
    }}

    document.getElementById("metaLine").textContent =
      `Veri: ${{summary.monthly_start || "?"}} -> ${{summary.monthly_end || "?"}} | Tahmin: ${{summary.forecast_start || "?"}} -> ${{summary.forecast_end || "?"}}`;

    scenarioSel.addEventListener("change", refresh);
    riskSel.addEventListener("change", refresh);
    seriesSel.addEventListener("change", refresh);
    syncSeriesOptions();
    refresh();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    in_dir = args.input_dir

    history_path = in_dir / "istanbul_dam_monthly_history.csv"
    scenario_fc_path = in_dir / "scenario_forecasts.csv"
    scenario_risk_path = in_dir / "scenario_risk_summary.csv"
    strategy_path = in_dir / "strategy_summary.csv"
    summary_path = in_dir / "run_summary.json"
    scenario_summary_path = in_dir / "scenario_summary.json"

    if not history_path.exists() or not scenario_fc_path.exists() or not scenario_risk_path.exists() or not strategy_path.exists():
        raise SystemExit("Missing scenario files. Run build_istanbul_dam_scenarios.py first.")

    history = pd.read_csv(history_path, parse_dates=["ds"])
    scenario_fc = pd.read_csv(scenario_fc_path, parse_dates=["ds"])
    scenario_risk = pd.read_csv(scenario_risk_path)
    strategies = pd.read_csv(strategy_path)

    summary = {}
    if summary_path.exists():
        summary.update(json.loads(summary_path.read_text(encoding="utf-8")))
    if scenario_summary_path.exists():
        summary.update(json.loads(scenario_summary_path.read_text(encoding="utf-8")))

    payload = {
        "history": to_records(history),
        "scenarios": to_records(scenario_fc),
        "scenarioRisk": to_records(scenario_risk),
        "strategies": to_records(strategies),
        "summary": summary,
    }
    args.output_html.write_text(build_html(json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
    print(args.output_html)


if __name__ == "__main__":
    main()
