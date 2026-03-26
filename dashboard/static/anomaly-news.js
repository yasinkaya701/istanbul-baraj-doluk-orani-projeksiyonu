const state = {
  runs: [],
  payload: null,
  variable: "all",
  tier: "all",
  matchBucket: "all",
  newsOnly: false,
  metric: "pointSeverity",
  selectedDate: null,
  chartRange: null,
};

const el = {
  runSelect: document.getElementById("runSelect"),
  metricSelect: document.getElementById("metricSelect"),
  newsOnlyToggle: document.getElementById("newsOnlyToggle"),
  variableChips: document.getElementById("variableChips"),
  tierChips: document.getElementById("tierChips"),
  matchChips: document.getElementById("matchChips"),
  statsGrid: document.getElementById("statsGrid"),
  latestMeta: document.getElementById("latestMeta"),
  latestClimateCards: document.getElementById("latestClimateCards"),
  variableBoards: document.getElementById("variableBoards"),
  coverageGrid: document.getElementById("coverageGrid"),
  coverageMeta: document.getElementById("coverageMeta"),
  chartPlot: document.getElementById("chartPlot"),
  chartWrap: document.getElementById("chartWrap"),
  chartTitle: document.getElementById("chartTitle"),
  chartMeta: document.getElementById("chartMeta"),
  integrityNote: document.getElementById("integrityNote"),
  detailTitle: document.getElementById("detailTitle"),
  detailDate: document.getElementById("detailDate"),
  detailSummary: document.getElementById("detailSummary"),
  detailBadges: document.getElementById("detailBadges"),
  climateGrid: document.getElementById("climateGrid"),
  newsCards: document.getElementById("newsCards"),
  eventCards: document.getElementById("eventCards"),
  dayTableBody: document.getElementById("dayTableBody"),
  tableMeta: document.getElementById("tableMeta"),
  refreshBtn: document.getElementById("refreshBtn"),
};

const VARIABLE_LABELS = {
  precip: "Yagis",
  humidity: "Nem",
  temp: "Sicaklik",
  pressure: "Basinc",
};

const VARIABLE_COLORS = {
  precip: "#2b6e98",
  humidity: "#2d8a7a",
  temp: "#d46b3f",
  pressure: "#55607c",
};

const BUCKET_LABELS = {
  direct: "Dogrudan",
  analog: "Benzer olay",
  archive: "Arsiv baglami",
  none: "Eslesme yok",
};

const BUCKET_COLORS = {
  direct: "#f0b23f",
  analog: "#1f5666",
  archive: "#7d6952",
  none: "#b9afa1",
};

const CLIMATE_LABELS = {
  t_mean_c: "Ort. Sicaklik",
  t_max_c: "Maks. Sicaklik",
  t_min_c: "Min. Sicaklik",
  rh_mean_pct: "Ort. Nem",
  vpd_kpa: "VPD",
  es_minus_ea_kpa: "Es-Ea",
};

const VARIABLE_ORDER = ["temp", "humidity", "pressure", "precip"];
const BUCKET_ORDER = ["direct", "analog", "archive", "none"];
const STRICT_REAL_ONLY = true;

function clearNode(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function formatDate(dateText) {
  if (!dateText) return "-";
  return new Date(`${String(dateText).slice(0, 10)}T00:00:00`).toLocaleDateString("tr-TR", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Number(value).toLocaleString("tr-TR", {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

function shortText(value, max = 78) {
  const text = String(value || "").trim();
  if (!text) return "-";
  return text.length <= max ? text : `${text.slice(0, max - 3)}...`;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function basenamePath(value) {
  const text = String(value || "").trim();
  if (!text) return "";
  const parts = text.split("/");
  return parts[parts.length - 1] || text;
}

function asFiniteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function metricLabel(metricKey) {
  const item = state.payload?.metrics?.find((entry) => entry.key === metricKey);
  return item?.label || metricKey;
}

function metricValue(day, metricKey) {
  if (!day) return null;
  if (metricKey in day) return asFiniteNumber(day[metricKey]);
  if (day.climate && metricKey in day.climate) return asFiniteNumber(day.climate[metricKey]);
  return null;
}

function realNewsItems(day) {
  const items = Array.isArray(day?.news) ? day.news : [];
  if (!STRICT_REAL_ONLY) return items;
  return items.filter((item) => {
    const bucket = String(item?.bucket || "").toLowerCase();
    const kind = String(item?.kind || "").toLowerCase();
    return bucket === "direct" || kind.includes("exact");
  });
}

function hasRealNews(day) {
  return realNewsItems(day).length > 0;
}

function displayBucketForDay(day) {
  if (!STRICT_REAL_ONLY) return String(day?.matchBucket || "none");
  return hasRealNews(day) ? "direct" : "none";
}

function displayBucketLabelForDay(day) {
  const bucket = displayBucketForDay(day);
  return BUCKET_LABELS[bucket] || bucket;
}

function bestRealNewsScore(day) {
  const items = realNewsItems(day);
  for (const item of items) {
    const score = asFiniteNumber(item?.score);
    if (score !== null) return score;
  }
  return null;
}

function quantile(values, q) {
  const arr = (values || [])
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v))
    .sort((a, b) => a - b);
  if (!arr.length) return null;
  const qq = Math.max(0, Math.min(1, Number(q)));
  const pos = (arr.length - 1) * qq;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return arr[lo];
  const weight = pos - lo;
  return arr[lo] * (1 - weight) + arr[hi] * weight;
}

function robustBounds(values, qLo = 0.02, qHi = 0.98) {
  const lo = quantile(values, qLo);
  const hi = quantile(values, qHi);
  let min = Number.isFinite(lo) ? lo : Math.min(...values);
  let max = Number.isFinite(hi) ? hi : Math.max(...values);
  if (!(max > min)) {
    min -= 1;
    max += 1;
  }
  return { min, max };
}

function orderedValues(values, order) {
  const set = new Set(values);
  const ordered = order.filter((value) => set.has(value));
  values.forEach((value) => {
    if (!ordered.includes(value)) ordered.push(value);
  });
  return ordered;
}

function availableVariables() {
  const values = Object.keys(state.payload?.stats?.variables || {});
  return orderedValues(values, VARIABLE_ORDER);
}

function filteredDays() {
  const days = state.payload?.days || [];
  return days.filter((day) => {
    if (state.variable !== "all" && day.topVariable !== state.variable) return false;
    if (state.tier !== "all" && day.scientificTier !== state.tier) return false;
    if (state.matchBucket !== "all" && displayBucketForDay(day) !== state.matchBucket) return false;
    if (state.newsOnly && !hasRealNews(day)) return false;
    return metricValue(day, state.metric) !== null;
  });
}

function selectedDay() {
  const days = filteredDays();
  return days.find((day) => day.date === state.selectedDate) || days[days.length - 1] || null;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

async function loadRuns() {
  const payload = await fetchJson("/api/anomaly-news-runs");
  state.runs = payload.runs || [];
}

async function loadRun(runId = "") {
  const query = runId ? `?id=${encodeURIComponent(runId)}` : "";
  state.payload = await fetchJson(`/api/anomaly-news${query}`);
  state.runs = state.payload.runs || state.runs;
  state.chartRange = null;
  if (!state.payload.metrics?.some((item) => item.key === state.metric)) {
    state.metric = state.payload.metrics?.[0]?.key || "eventSeverity";
  }
  if (!selectedDay()) state.selectedDate = state.payload.days?.[state.payload.days.length - 1]?.date || null;
  renderAll();
}

function renderRunSelect() {
  clearNode(el.runSelect);
  state.runs.forEach((run) => {
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent = `${run.label} | ${run.day_count} gun`;
    if (state.payload?.run?.id === run.id) option.selected = true;
    el.runSelect.appendChild(option);
  });
}

function renderMetricSelect() {
  clearNode(el.metricSelect);
  (state.payload?.metrics || []).forEach((item) => {
    const option = document.createElement("option");
    option.value = item.key;
    option.textContent = item.label;
    if (item.key === state.metric) option.selected = true;
    el.metricSelect.appendChild(option);
  });
}

function renderChipGroup(container, values, activeValue, onClick, labelFn = (value) => value) {
  clearNode(container);

  const allChip = document.createElement("button");
  allChip.type = "button";
  allChip.className = `chip ${activeValue === "all" ? "active" : ""}`;
  allChip.textContent = "Tum";
  allChip.addEventListener("click", () => onClick("all"));
  container.appendChild(allChip);

  values.forEach((value) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `chip ${activeValue === value ? "active" : ""}`;
    chip.textContent = labelFn(value);
    chip.addEventListener("click", () => onClick(value));
    container.appendChild(chip);
  });
}

function renderStats() {
  clearNode(el.statsGrid);
  const days = filteredDays();
  const directCount = days.filter((day) => hasRealNews(day)).length;
  const emptyCount = days.length - directCount;
  const cards = [
    {
      label: "Filtreli gun",
      value: days.length,
      note: `${state.payload?.stats?.totalDays || 0} toplam gun`,
    },
    {
      label: "Dogrudan gercek haber",
      value: directCount,
      note: "Ayni gun / ayni event haber kaydi",
    },
    {
      label: "Haber yok",
      value: emptyCount,
      note: "Dogrudan haber kaydi bulunamadi",
    },
    {
      label: "Maks event siddeti",
      value: formatNumber(days.reduce((mx, day) => Math.max(mx, Number(day.eventSeverity || 0)), 0), 2),
      note: "Filtredeki en yuksek event siddeti",
    },
    {
      label: "Secili metrik",
      value: metricLabel(state.metric),
      note: "Grafik Y ekseni",
    },
  ];

  cards.forEach((item) => {
    const card = document.createElement("article");
    card.innerHTML = `
      <p class="stat-label">${item.label}</p>
      <p class="stat-value">${item.value}</p>
      <p class="stat-note">${item.note}</p>
    `;
    el.statsGrid.appendChild(card);
  });
}

function buildMiniSvg(variable, days) {
  const width = 260;
  const height = 88;
  const pad = 8;
  const values = days
    .map((day) => day.eventSeverity)
    .filter((value) => value !== null && value !== undefined);
  if (!values.length) return `<svg class="mini-svg" viewBox="0 0 ${width} ${height}"></svg>`;

  const bounds = robustBounds(values, 0.05, 0.95);
  const safeMin = bounds.min;
  const safeMax = bounds.max;
  const points = days
    .map((day, index) => {
      const value = Math.max(safeMin, Math.min(safeMax, day.eventSeverity ?? safeMin));
      const x = pad + (index / Math.max(days.length - 1, 1)) * (width - pad * 2);
      const y = pad + (1 - (value - safeMin) / (safeMax - safeMin || 1)) * (height - pad * 2);
      return `${x},${y}`;
    })
    .join(" ");
  return `
    <svg class="mini-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <rect x="0" y="0" width="${width}" height="${height}" rx="14" fill="rgba(31,86,102,0.03)"></rect>
      <polyline fill="none" stroke="${VARIABLE_COLORS[variable] || "#666"}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="${points}"></polyline>
    </svg>
  `;
}

function renderLatestClimate() {
  const latest = state.payload?.latestClimate || {};
  el.latestMeta.textContent = latest.date ? `${formatDate(latest.date)} | ${latest.source || "-"}` : "Veri yok";
  clearNode(el.latestClimateCards);

  const cards = [
    {
      label: "Ort. Sicaklik",
      value: `${formatNumber(latest.t_mean_c, 2)} C`,
      note: `Min ${formatNumber(latest.t_min_c, 2)} | Max ${formatNumber(latest.t_max_c, 2)}`,
    },
    {
      label: "Ort. Nem",
      value: `${formatNumber(latest.rh_mean_pct, 1)} %`,
      note: `Kaynak ${latest.sourceHumidity || latest.source || "-"}`,
    },
    {
      label: "VPD",
      value: `${formatNumber(latest.vpd_kpa, 3)} kPa`,
      note: `Es-Ea ${formatNumber(latest.es_minus_ea_kpa, 3)} kPa`,
    },
    {
      label: "Gozlem",
      value: formatNumber(latest.autoObsCount, 0),
      note: `${latest.sourceTemp || "-"} | ${latest.eaFormula || "-"}`,
    },
  ];

  cards.forEach((item) => {
    const card = document.createElement("article");
    card.className = "live-card";
    card.innerHTML = `
      <p class="metric-label">${item.label}</p>
      <p class="metric-value">${item.value}</p>
      <p class="metric-note">${item.note}</p>
    `;
    el.latestClimateCards.appendChild(card);
  });
}

function renderVariableBoards() {
  clearNode(el.variableBoards);
  const allDays = state.payload?.days || [];
  availableVariables().forEach((variable) => {
    const days = allDays.filter((day) => day.topVariable === variable);
    const newsDays = days.filter((day) => hasRealNews(day)).length;
    const directDays = newsDays;
    const coverage = days.length ? (newsDays / days.length) * 100 : 0;
    const latestDay = days[days.length - 1];
    const sevVals = days.map((day) => Number(day.eventSeverity)).filter((value) => Number.isFinite(value));
    const p95Severity = sevVals.length ? quantile(sevVals, 0.95) : 0;

    const card = document.createElement("article");
    card.className = `variable-board ${state.variable === variable ? "active" : ""}`;
    card.innerHTML = `
      <div class="variable-board-head">
        <h3>${VARIABLE_LABELS[variable] || variable}</h3>
        <span class="muted-mini">${days.length} gun</span>
      </div>
      ${buildMiniSvg(variable, days)}
      <div class="mini-progress"><span style="width:${coverage.toFixed(1)}%"></span></div>
      <div class="mini-stat-row">
        <span>Haberli ${newsDays}</span>
        <span>Dogrudan ${directDays}</span>
      </div>
      <div class="mini-stat-row">
        <span>P95 siddet ${formatNumber(p95Severity, 2)}</span>
        <span>${latestDay ? formatDate(latestDay.date) : "-"}</span>
      </div>
    `;
    card.addEventListener("click", () => {
      state.variable = state.variable === variable ? "all" : variable;
      state.selectedDate = filteredDays()[filteredDays().length - 1]?.date || null;
      renderAll();
    });
    el.variableBoards.appendChild(card);
  });
}

function renderCoverage() {
  clearNode(el.coverageGrid);
  const days = filteredDays();
  const total = days.length || 1;
  el.coverageMeta.textContent = `${days.length} gunluk filtre`;

  const bucketsToRender = STRICT_REAL_ONLY ? ["direct", "none"] : BUCKET_ORDER;
  bucketsToRender.forEach((bucket) => {
    const count =
      bucket === "none"
        ? days.filter((day) => !hasRealNews(day)).length
        : days.filter((day) => displayBucketForDay(day) === bucket).length;
    const pct = Math.round((count / total) * 100);
    const card = document.createElement("article");
    card.className = `coverage-card ${bucket}`;
    card.innerHTML = `
      <div class="coverage-top">
        <p class="metric-label">${BUCKET_LABELS[bucket]}</p>
        <p class="coverage-value">${count}</p>
      </div>
      <p class="coverage-note">%${pct} pay</p>
      <span class="coverage-bar" style="width:${pct}%"></span>
    `;
    el.coverageGrid.appendChild(card);
  });
}

function laneVariables(points) {
  const vars = orderedValues(
    [...new Set(points.map((day) => day.topVariable).filter(Boolean))],
    VARIABLE_ORDER
  );
  if (state.variable !== "all") return vars.filter((value) => value === state.variable);
  return vars;
}

function bubbleSize(day) {
  const sev = Math.max(0, Number(day.eventSeverity || day.pointSeverity || 0));
  return Math.max(7, Math.min(20, 7 + Math.log1p(sev) * 3.2));
}

function buildHoverHtml(day) {
  const directNews = realNewsItems(day);
  const lines = [
    `<b>${escapeHtml(formatDate(day.date))}</b>`,
    `${escapeHtml(VARIABLE_LABELS[day.topVariable] || day.topVariable || "-")} | ${escapeHtml(displayBucketLabelForDay(day))} | ${escapeHtml(day.scientificTier || "-")} tier`,
    `${escapeHtml(metricLabel(state.metric))}: ${escapeHtml(formatNumber(metricValue(day, state.metric), 2))}`,
    `Event siddeti: ${escapeHtml(formatNumber(day.eventSeverity, 2))} | Nokta siddeti: ${escapeHtml(formatNumber(day.pointSeverity, 2))}`,
  ];

  if (directNews.length) {
    directNews.slice(0, 2).forEach((item) => {
      const meta = [
        item.kindLabel || item.bucketLabel || "-",
        item.source || "-",
        item.date ? formatDate(item.date) : "-",
      ]
        .filter(Boolean)
        .join(" | ");
      lines.push(`<br><span>${escapeHtml(meta)}</span>`);
      lines.push(escapeHtml(shortText(item.headline, 110)));
    });
  } else {
    lines.push("<br><span>Bu gun icin haber kaydi yok.</span>");
  }

  return lines.join("<br>");
}

function buildLineSeries(days) {
  const xs = [];
  const ys = [];
  let previousDateMs = null;
  const maxGapMs = 370 * 24 * 60 * 60 * 1000;

  days.forEach((day) => {
    const value = metricValue(day, state.metric);
    if (value === null) return;
    const currentDateMs = new Date(`${day.date}T00:00:00`).getTime();
    if (previousDateMs !== null && currentDateMs - previousDateMs > maxGapMs) {
      xs.push(null);
      ys.push(null);
    }
    xs.push(day.date);
    ys.push(value);
    previousDateMs = currentDateMs;
  });

  return { xs, ys };
}

function plotAxisRef(prefix, index) {
  return index === 0 ? prefix : `${prefix}${index + 1}`;
}

function plotAxisLayoutKey(prefix, index) {
  return `${prefix}axis${index === 0 ? "" : index + 1}`;
}

function renderIntegrityNote(points, lanes) {
  const directCount = points.filter((day) => hasRealNews(day)).length;
  const noNewsCount = points.length - directCount;
  const sources = [state.payload?.newsSummaryPath, state.payload?.newsEnrichedPath]
    .map((value) => basenamePath(value))
    .filter(Boolean)
    .join(" | ");

  const parts = [
    `Grafik secili run icindeki ${points.length} gercek anomaly-day kaydini ${lanes.length} lane uzerinde cizer.`,
    "Hover ve detay kartlarinda yalnizca dogrudan gercek haber kayitlari kullanilir.",
  ];
  parts.push(`Dogrudan haberli gun: ${directCount}; dogrudan haber olmayan gun: ${noNewsCount}.`);
  if (directCount) {
    parts.push(`Dogrudan eslesme sayisi: ${directCount}.`);
  }
  if (sources) {
    parts.push(`Haber kataloglari: ${sources}.`);
  }
  el.integrityNote.textContent = parts.join(" ");
}

function bindPlotlyEvents() {
  if (!window.Plotly || el.chartPlot.__boundPlotlyEvents) return;

  el.chartPlot.on("plotly_click", (payload) => {
    const point = payload?.points?.[0];
    const clickedDate = String(point?.customdata?.[0] || point?.x || "").slice(0, 10);
    if (!clickedDate) return;
    state.selectedDate = clickedDate;
    renderChart();
    renderDetail();
    renderTable();
  });

  el.chartPlot.on("plotly_relayout", (payload) => {
    if (!payload || payload["xaxis.autorange"] || payload["xaxis2.autorange"] || payload["xaxis3.autorange"] || payload["xaxis4.autorange"]) {
      state.chartRange = null;
      return;
    }

    const range0Key = Object.keys(payload).find((key) => key.endsWith(".range[0]"));
    const range1Key = Object.keys(payload).find((key) => key.endsWith(".range[1]"));
    if (range0Key && range1Key) {
      state.chartRange = [payload[range0Key], payload[range1Key]];
    }
  });

  el.chartPlot.__boundPlotlyEvents = true;
}

function renderChart() {
  const points = filteredDays();
  const selected = selectedDay();
  const lanes = laneVariables(points);

  el.chartTitle.textContent = `${metricLabel(state.metric)} ile ayrilmis anomaly timeline`;
  el.chartMeta.textContent = `${state.payload?.run?.label || "-"} | ${points.length} nokta | ${lanes.length} lane`;
  renderIntegrityNote(points, lanes);

  if (!window.Plotly) {
    el.chartPlot.innerHTML = `<div class="empty-state">Yerel Plotly bundle yuklenemedi; grafik cizilemedi.</div>`;
    return;
  }

  if (!points.length || !lanes.length) {
    window.Plotly.purge(el.chartPlot);
    el.chartPlot.innerHTML = `<div class="empty-state">Bu filtrede cizilecek veri yok.</div>`;
    return;
  }

  const traces = [];
  const gap = lanes.length > 1 ? 0.035 : 0;
  const domainHeight = (1 - gap * (lanes.length - 1)) / lanes.length;
  const layout = {
    height: Math.max(520, 210 * lanes.length + 120),
    margin: { l: 78, r: 34, t: 24, b: 88 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(255,255,255,0.78)",
    font: { family: '"Avenir Next", "Trebuchet MS", sans-serif', color: "#172d36", size: 12 },
    hovermode: "closest",
    hoverlabel: {
      bgcolor: "rgba(21,38,47,0.96)",
      bordercolor: "rgba(21,38,47,0.96)",
      font: { color: "#ffffff", size: 13 },
      align: "left",
    },
    showlegend: false,
    dragmode: "zoom",
  };

  lanes.forEach((variable, index) => {
    const lanePoints = points
      .filter((day) => day.topVariable === variable)
      .sort((left, right) => left.date.localeCompare(right.date));
    const xRef = plotAxisRef("x", index);
    const yRef = plotAxisRef("y", index);
    const xAxisKey = plotAxisLayoutKey("x", index);
    const yAxisKey = plotAxisLayoutKey("y", index);
    const top = 1 - index * (domainHeight + gap);
    const bottom = top - domainHeight;
    const values = lanePoints.map((day) => metricValue(day, state.metric)).filter((value) => value !== null);
    const bounds = values.length ? robustBounds(values, 0.02, 0.98) : { min: 0, max: 1 };
    const padding = bounds.max > bounds.min ? (bounds.max - bounds.min) * 0.12 : 1;

    layout[yAxisKey] = {
      domain: [bottom, top],
      title: { text: VARIABLE_LABELS[variable] || variable, standoff: 8 },
      showgrid: true,
      gridcolor: "rgba(31,86,102,0.10)",
      zerolinecolor: "rgba(31,86,102,0.16)",
      automargin: true,
      range: [bounds.min - padding, bounds.max + padding],
      ticks: "outside",
    };

    layout[xAxisKey] = {
      type: "date",
      anchor: yRef,
      domain: [0, 1],
      showgrid: true,
      gridcolor: "rgba(31,86,102,0.08)",
      showticklabels: index === lanes.length - 1,
      tickformat: "%Y",
      ticks: "outside",
      matches: index === 0 ? undefined : "x",
      range: state.chartRange || undefined,
      rangeslider:
        index === lanes.length - 1
          ? {
              visible: true,
              thickness: 0.09,
              bgcolor: "rgba(217,231,235,0.36)",
              bordercolor: "rgba(218,205,189,0.90)",
            }
          : undefined,
      rangeselector:
        index === lanes.length - 1
          ? {
              bgcolor: "rgba(255,253,248,0.92)",
              bordercolor: "rgba(218,205,189,0.96)",
              borderwidth: 1,
              x: 0,
              y: 1.1,
              buttons: [
                { count: 1, label: "1Y", step: "year", stepmode: "backward" },
                { count: 5, label: "5Y", step: "year", stepmode: "backward" },
                { count: 10, label: "10Y", step: "year", stepmode: "backward" },
                { count: 25, label: "25Y", step: "year", stepmode: "backward" },
                { step: "all", label: "Tum" },
              ],
            }
          : undefined,
    };

    const lineSeries = buildLineSeries(lanePoints);
    traces.push({
      type: "scattergl",
      mode: "lines",
      x: lineSeries.xs,
      y: lineSeries.ys,
      xaxis: xRef,
      yaxis: yRef,
      line: {
        color: VARIABLE_COLORS[variable] || "#666666",
        width: 1.7,
      },
      opacity: 0.34,
      hoverinfo: "skip",
      showlegend: false,
    });

    BUCKET_ORDER.forEach((bucket) => {
      const bucketPoints = lanePoints.filter((day) => displayBucketForDay(day) === bucket);
      if (!bucketPoints.length) return;
      if (STRICT_REAL_ONLY && !["direct", "none"].includes(bucket)) return;

      traces.push({
        type: "scattergl",
        mode: "markers",
        x: bucketPoints.map((day) => day.date),
        y: bucketPoints.map((day) => metricValue(day, state.metric)),
        xaxis: xRef,
        yaxis: yRef,
        customdata: bucketPoints.map((day) => [day.date]),
        hovertext: bucketPoints.map((day) => buildHoverHtml(day)),
        hovertemplate: "%{hovertext}<extra></extra>",
        marker: {
          size: bucketPoints.map((day) => bubbleSize(day)),
          color: VARIABLE_COLORS[variable] || "#666666",
          opacity: 0.92,
          line: {
            width: 2.6,
            color: BUCKET_COLORS[bucket] || BUCKET_COLORS.none,
          },
        },
        showlegend: false,
      });
    });

    if (selected && selected.topVariable === variable) {
      traces.push({
        type: "scatter",
        mode: "markers",
        x: [selected.date],
        y: [metricValue(selected, state.metric)],
        xaxis: xRef,
        yaxis: yRef,
        hoverinfo: "skip",
        marker: {
          size: bubbleSize(selected) + 8,
          color: "rgba(0,0,0,0)",
          line: {
            width: 2.8,
            color: "#172d36",
          },
          symbol: "diamond-open",
        },
        showlegend: false,
      });
    }
  });

  window.Plotly.react(
    el.chartPlot,
    traces,
    layout,
    {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "resetScale2d"],
      scrollZoom: true,
    }
  );
  bindPlotlyEvents();
}

function buildSearchLinks(day) {
  const year = day?.date?.slice(0, 4) || "";
  const variableLabel = VARIABLE_LABELS[day?.topVariable] || day?.topVariable || "meteoroloji";
  const query = `${year} ${variableLabel} meteoroloji`.trim();
  return [
    {
      label: "Cumhuriyet Arsivi",
      url: `https://www.google.com/search?q=${encodeURIComponent(`site:cumhuriyet.com.tr ${query}`)}`,
    },
    {
      label: "Hurriyet Arsivi",
      url: `https://www.google.com/search?q=${encodeURIComponent(`site:hurriyet.com.tr ${query}`)}`,
    },
  ];
}

function renderDetailBadges(day) {
  clearNode(el.detailBadges);
  const directNews = realNewsItems(day);
  const badges = [
    { label: displayBucketLabelForDay(day), kind: displayBucketForDay(day) },
    { label: `${day.scientificTier || "-"} tier`, kind: "none" },
    { label: `${day.eventCount || 0} event`, kind: "none" },
    { label: `${directNews.length || 0} dogrudan haber`, kind: directNews.length ? "direct" : "none" },
  ];
  if (day.dominantDirection) badges.push({ label: `Yon ${day.dominantDirection}`, kind: "none" });
  if (directNews[0]?.kindLabel) badges.push({ label: directNews[0].kindLabel, kind: "direct" });

  badges.forEach((item) => {
    const badge = document.createElement("span");
    badge.className = `inline-badge ${item.kind}`;
    badge.textContent = item.label;
    el.detailBadges.appendChild(badge);
  });
}

function newsRelationText(item) {
  if (!item) return "-";
  if (item.bucket === "direct") return "Ayni gun veya ayni event baglantisi";

  const parts = [];
  if (Number.isFinite(item.yearGap)) parts.push(`${Math.abs(Number(item.yearGap))} yil farki`);
  if (Number.isFinite(item.seasonalDayDiff)) parts.push(`${Math.abs(Number(item.seasonalDayDiff))} mevsim gun farki`);
  if (!parts.length && Number.isFinite(item.dayDiff)) parts.push(`${Math.abs(Number(item.dayDiff))} gun farki`);
  return parts.join(" | ") || "Ayni-gun olmayan arsiv referansi";
}

function renderDetail() {
  const day = selectedDay();
  if (!day) {
    el.detailTitle.textContent = "Veri yok";
    el.detailDate.textContent = "";
    el.detailSummary.innerHTML = `<div class="empty-state">Bu filtre icin detay gosterilemiyor.</div>`;
    clearNode(el.detailBadges);
    clearNode(el.climateGrid);
    el.newsCards.innerHTML = `<div class="empty-state">Haber bulunamadi.</div>`;
    el.eventCards.innerHTML = `<div class="empty-state">Event bulunamadi.</div>`;
    return;
  }

  const directNews = realNewsItems(day);
  const bestDirectScore = bestRealNewsScore(day);
  el.detailTitle.textContent = `${VARIABLE_LABELS[day.topVariable] || day.topVariable} anomalisi`;
  el.detailDate.textContent = formatDate(day.date);
  el.detailSummary.innerHTML = `
    <strong>${day.eventCount} event, ${day.anomalyPointCount} anomaly point</strong><br />
    Grafik metrigi ${metricLabel(state.metric)}: ${formatNumber(metricValue(day, state.metric), 2)}<br />
    Event siddeti ${formatNumber(day.eventSeverity, 2)} | Nokta siddeti ${formatNumber(day.pointSeverity, 2)}<br />
    Haber bagi ${displayBucketLabelForDay(day)} | En iyi dogrudan skor ${formatNumber(bestDirectScore, 2)}
  `;
  renderDetailBadges(day);

  clearNode(el.climateGrid);
  ["t_mean_c", "t_max_c", "t_min_c", "rh_mean_pct", "vpd_kpa", "es_minus_ea_kpa"].forEach((key) => {
    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `
      <p class="metric-label">${CLIMATE_LABELS[key] || key}</p>
      <p class="metric-value">${formatNumber(day.climate?.[key], 2)}</p>
    `;
    el.climateGrid.appendChild(card);
  });

  clearNode(el.newsCards);
  if (!directNews.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = `
      Bu gun icin dogrudan haber kaydi bulunmadi.
      <div class="search-actions">
        ${buildSearchLinks(day)
          .map((item) => `<a href="${item.url}" target="_blank" rel="noreferrer">${item.label}</a>`)
          .join("")}
      </div>
    `;
    el.newsCards.appendChild(empty);
  } else {
    directNews.forEach((item) => {
      const card = document.createElement("article");
      card.className = "news-card";
      card.innerHTML = `
        <div class="news-card-head">
          <span class="match-pill ${item.bucket || "none"}">${item.kindLabel || item.bucketLabel || "-"}</span>
          <span class="news-score">Skor ${formatNumber(item.score, 2)}</span>
        </div>
        <p class="news-meta">${item.source || "-"} | ${formatDate(item.date)} | ref ${formatDate(item.referenceDate)}</p>
        <h4>${escapeHtml(item.headline)}</h4>
        <p class="news-reason"><strong>Bag aciklamasi:</strong> ${escapeHtml(item.matchReason || "-")}</p>
        <p class="news-reason"><strong>Tarih iliskisi:</strong> ${escapeHtml(newsRelationText(item))}</p>
        ${item.url ? `<a href="${item.url}" target="_blank" rel="noreferrer">Haberi ac</a>` : ""}
      `;
      el.newsCards.appendChild(card);
    });
  }

  clearNode(el.eventCards);
  if (!day.events.length) {
    el.eventCards.innerHTML = `<div class="empty-state">Bu gun icin event ozeti yok.</div>`;
  } else {
    day.events.slice(0, 4).forEach((item) => {
      const card = document.createElement("article");
      card.className = "event-card";
      card.innerHTML = `
        <p class="event-meta">${item.eventId} | ${item.scientificTier || "-"} tier | ${item.internetConfidence || "-"}</p>
        <h4>${VARIABLE_LABELS[item.variable] || item.variable}</h4>
        <p class="news-reason">${escapeHtml(item.causeSummary || "Cause summary yok.")}</p>
      `;
      el.eventCards.appendChild(card);
    });
  }
}

function renderTable() {
  clearNode(el.dayTableBody);
  const days = filteredDays();
  el.tableMeta.textContent = `${days.length} satir`;

  days.forEach((day) => {
    const row = document.createElement("tr");
    if (state.selectedDate === day.date) row.classList.add("active");
    const directNews = realNewsItems(day);
    const leadHeadline = directNews[0]
      ? `${directNews[0].source || "-"} | ${directNews[0].headline || ""}`
      : "Dogrudan haber yok.";
    const bucket = displayBucketForDay(day);
    row.innerHTML = `
      <td>${formatDate(day.date)}</td>
      <td>${VARIABLE_LABELS[day.topVariable] || day.topVariable}</td>
      <td><span class="match-pill ${bucket}">${displayBucketLabelForDay(day)}</span></td>
      <td>${formatNumber(bestRealNewsScore(day), 2)}</td>
      <td class="headline-cell">${escapeHtml(shortText(leadHeadline, 112))}</td>
    `;
    row.addEventListener("click", () => {
      state.selectedDate = day.date;
      renderChart();
      renderDetail();
      renderTable();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    el.dayTableBody.appendChild(row);
  });
}

function renderAll() {
  renderRunSelect();
  renderMetricSelect();

  renderChipGroup(
    el.variableChips,
    availableVariables(),
    state.variable,
    (value) => {
      state.variable = value;
      state.selectedDate = filteredDays()[filteredDays().length - 1]?.date || null;
      renderAll();
    },
    (value) => VARIABLE_LABELS[value] || value
  );

  const tiers = Object.keys(state.payload?.stats?.tiers || {}).sort();
  renderChipGroup(el.tierChips, tiers, state.tier, (value) => {
    state.tier = value;
    state.selectedDate = filteredDays()[filteredDays().length - 1]?.date || null;
    renderAll();
  });

  const matchValues = (STRICT_REAL_ONLY ? ["direct", "none"] : BUCKET_ORDER).filter((bucket) =>
    bucket === "none"
      ? (state.payload?.days || []).some((day) => !hasRealNews(day))
      : (state.payload?.days || []).some((day) => displayBucketForDay(day) === bucket)
  );
  renderChipGroup(
    el.matchChips,
    matchValues,
    state.matchBucket,
    (value) => {
      state.matchBucket = value;
      state.selectedDate = filteredDays()[filteredDays().length - 1]?.date || null;
      renderAll();
    },
    (value) => BUCKET_LABELS[value] || value
  );

  const selected = selectedDay();
  state.selectedDate = selected?.date || null;
  el.newsOnlyToggle.checked = state.newsOnly;

  renderStats();
  renderLatestClimate();
  renderVariableBoards();
  renderCoverage();
  renderChart();
  renderDetail();
  renderTable();
}

function bindEvents() {
  el.runSelect.addEventListener("change", () => loadRun(el.runSelect.value));
  el.metricSelect.addEventListener("change", () => {
    state.metric = el.metricSelect.value;
    renderAll();
  });
  el.newsOnlyToggle.addEventListener("change", () => {
    state.newsOnly = el.newsOnlyToggle.checked;
    state.selectedDate = filteredDays()[filteredDays().length - 1]?.date || null;
    renderAll();
  });
  el.refreshBtn.addEventListener("click", async () => {
    await loadRuns();
    await loadRun(el.runSelect.value || state.runs[0]?.id || "");
  });
  window.addEventListener("resize", () => {
    if (window.Plotly && el.chartPlot?.data?.length) {
      window.Plotly.Plots.resize(el.chartPlot);
    }
  });
}

async function init() {
  bindEvents();
  await loadRuns();
  const params = new URLSearchParams(window.location.search);
  const initialRun = params.get("id") || state.runs[0]?.id || "";
  await loadRun(initialRun);
}

init().catch((error) => {
  console.error(error);
  el.chartTitle.textContent = "Veri yuklenemedi";
  el.chartMeta.textContent = String(error);
  if (el.integrityNote) el.integrityNote.textContent = "Grafik yuklenemedi.";
});
