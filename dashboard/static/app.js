const state = {
  runs: [],
  currentRunId: null,
  currentFeatureKey: null,
  currentDetail: null,
  liveQuant: null,
  runSearchText: "",
  runFilter: "all",
  presentationMode: "balanced",
  deckEnabled: false,
  focusIndex: -1,
  deckAutoPlay: false,
  deckTimerId: null,
  deckCycleStartMs: 0,
  deckIntervalSec: 8,
  sectionObserver: null,
  activeSectionId: "summaryCardsSection",
};

const el = {
  runList: document.getElementById("runList"),
  runSearchInput: document.getElementById("runSearchInput"),
  runArchiveStats: document.getElementById("runArchiveStats"),
  runFilterButtons: Array.from(document.querySelectorAll("#runFilterChips .filter-chip")),
  runTitle: document.getElementById("runTitle"),
  runMeta: document.getElementById("runMeta"),
  summaryCards: document.getElementById("summaryCardsSection"),
  heroMetrics: document.getElementById("heroMetrics"),
  sectionNav: document.getElementById("sectionNav"),
  runPulse: document.getElementById("runPulse"),
  commandCenterStatus: document.getElementById("commandCenterStatus"),
  presentationBadge: document.getElementById("presentationBadge"),
  presentationModeBtn: document.getElementById("presentationModeBtn"),
  presentationFocusNextBtn: document.getElementById("presentationFocusNextBtn"),
  presentationAutoBtn: document.getElementById("presentationAutoBtn"),
  presentationIntervalSelect: document.getElementById("presentationIntervalSelect"),
  presentationAutoStatus: document.getElementById("presentationAutoStatus"),
  presentationProgressTrack: document.getElementById("presentationProgressTrack"),
  presentationProgressFill: document.getElementById("presentationProgressFill"),
  summaryCardsSection: document.getElementById("summaryCardsSection"),
  visualBoardSection: document.getElementById("visualBoardSection"),
  integrationSection: document.getElementById("integrationSection"),
  executiveSection: document.getElementById("executiveSection"),
  presentationNoteSection: document.getElementById("presentationNoteSection"),
  presentationChartsSection: document.getElementById("presentationChartsSection"),
  healthSection: document.getElementById("healthSection"),
  resultSection: document.getElementById("resultSection"),
  visualGaugeGrid: document.getElementById("visualGaugeGrid"),
  visualHealthStack: document.getElementById("visualHealthStack"),
  visualRiskBars: document.getElementById("visualRiskBars"),
  visualModelMatrix: document.getElementById("visualModelMatrix"),
  integrationList: document.getElementById("integrationList"),
  executiveList: document.getElementById("executiveList"),
  presentationNote: document.getElementById("presentationNote"),
  presentationCopyBtn: document.getElementById("presentationCopyBtn"),
  presentationDownloadPdfBtn: document.getElementById("presentationDownloadPdfBtn"),
  presentationDownloadTxtBtn: document.getElementById("presentationDownloadTxtBtn"),
  presentationDownloadMdBtn: document.getElementById("presentationDownloadMdBtn"),
  presentationDownloadJsonBtn: document.getElementById("presentationDownloadJsonBtn"),
  presentationDownloadPackBtn: document.getElementById("presentationDownloadPackBtn"),
  presentationChartsPackBtn: document.getElementById("presentationChartsPackBtn"),
  presentationModeBalancedBtn: document.getElementById("presentationModeBalancedBtn"),
  presentationModeHealthBtn: document.getElementById("presentationModeHealthBtn"),
  presentationModePerformanceBtn: document.getElementById("presentationModePerformanceBtn"),
  presentationAssetsMeta: document.getElementById("presentationAssetsMeta"),
  presentationAssetsGallery: document.getElementById("presentationAssetsGallery"),
  healthSummaryCards: document.getElementById("healthSummaryCards"),
  healthTableBody: document.querySelector("#healthTable tbody"),
  resultTableBody: document.querySelector("#resultTable tbody"),
  logSection: document.getElementById("logSection"),
  fullRunCommand: document.getElementById("fullRunCommand"),
  fullRunCopyBtn: document.getElementById("fullRunCopyBtn"),
  modelCoverage: document.getElementById("modelCoverage"),
  featureInventory: document.getElementById("featureInventory"),
  featureTitle: document.getElementById("featureTitle"),
  featureDetail: document.getElementById("featureDetail"),
  reportDate: document.getElementById("reportDate"),
  refreshBtn: document.getElementById("refreshBtn"),
  liveQuantMeta: document.getElementById("liveQuantMeta"),
  liveQuantCards: document.getElementById("liveQuantCards"),
  liveQuantGallery: document.getElementById("liveQuantGallery"),
  liveQuantRefreshBtn: document.getElementById("liveQuantRefreshBtn"),
};

let liveQuantTimerId = null;

const PRESENTATION_MODE_LABELS = {
  balanced: "Dengeli",
  health: "Saglik Odakli",
  performance: "Performans Odakli",
};

const LIVE_QUANT_VAR_LABELS = {
  humidity: "Nem",
  precip: "Yagis",
  pressure: "Basinc",
  temp: "Sicaklik",
};

const SECTION_NAV_ITEMS = [
  { id: "summaryCardsSection", label: "Ozet" },
  { id: "liveQuantSection", label: "Canli Quant" },
  { id: "visualBoardSection", label: "Pano" },
  { id: "integrationSection", label: "Entegrasyon" },
  { id: "executiveSection", label: "Yonetici" },
  { id: "presentationChartsSection", label: "Grafikler" },
  { id: "healthSection", label: "Saglik" },
  { id: "resultSection", label: "Sonuclar" },
  { id: "technicalSection", label: "Teknik" },
  { id: "coverageSection", label: "Kapsam" },
  { id: "inventorySection", label: "Envanter" },
  { id: "featureDetailSection", label: "Detay" },
];

const urlParams = new URLSearchParams(window.location.search);
const initialModeFromUrl = urlParams.get("mode");
if (initialModeFromUrl) {
  state.presentationMode = normalizePresentationMode(initialModeFromUrl);
}
const initialDeckFromUrl = ["1", "true", "yes", "on"].includes(
  String(urlParams.get("deck") || "").trim().toLowerCase()
);
const initialAutoFromUrl = ["1", "true", "yes", "on"].includes(
  String(urlParams.get("auto") || "").trim().toLowerCase()
);
state.deckIntervalSec = normalizeDeckInterval(urlParams.get("sec") || state.deckIntervalSec);

function getPresentationFocusTargets() {
  return [
    el.summaryCardsSection,
    el.visualBoardSection,
    el.integrationSection,
    el.executiveSection,
    el.presentationNoteSection,
    el.presentationChartsSection,
    el.healthSection,
    el.resultSection,
  ].filter(Boolean);
}

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

function clearNode(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function normalizeDeckInterval(value) {
  const n = Number(value);
  if (n <= 6) return 5;
  if (n <= 10) return 8;
  return 12;
}

function clearFocusPanels() {
  getPresentationFocusTargets().forEach((panel) => panel.classList.remove("focused"));
}

function setDeckProgress(ratio) {
  if (!el.presentationProgressFill) return;
  const safe = Number.isFinite(ratio) ? Math.max(0, Math.min(1, ratio)) : 0;
  el.presentationProgressFill.style.width = `${safe * 100}%`;
}

function updateDeckAutoStatus() {
  if (el.presentationAutoBtn) {
    el.presentationAutoBtn.textContent = state.deckAutoPlay ? "Oto Durdur" : "Oto Baslat";
  }
  if (el.presentationAutoStatus) {
    if (state.deckAutoPlay) {
      el.presentationAutoStatus.textContent = `Otomatik Aktif | ${state.deckIntervalSec} sn`;
      el.presentationAutoStatus.classList.add("active");
    } else {
      el.presentationAutoStatus.textContent = "Otomatik Pasif";
      el.presentationAutoStatus.classList.remove("active");
    }
  }
  if (el.presentationIntervalSelect) {
    el.presentationIntervalSelect.value = String(state.deckIntervalSec);
  }
}

function stopDeckAutoPlay(resetProgress = true) {
  if (state.deckTimerId) {
    window.clearInterval(state.deckTimerId);
    state.deckTimerId = null;
  }
  state.deckAutoPlay = false;
  state.deckCycleStartMs = 0;
  if (resetProgress) setDeckProgress(0);
  updateDeckAutoStatus();
}

function startDeckAutoPlay() {
  if (!state.deckEnabled) setPresentationDeckEnabled(true);
  stopDeckAutoPlay(false);
  state.deckAutoPlay = true;
  state.deckCycleStartMs = Date.now();
  setDeckProgress(0);
  const durationMs = state.deckIntervalSec * 1000;
  state.deckTimerId = window.setInterval(() => {
    const elapsed = Date.now() - state.deckCycleStartMs;
    if (elapsed >= durationMs) {
      focusNextPanel(true);
      state.deckCycleStartMs = Date.now();
      setDeckProgress(0);
      return;
    }
    setDeckProgress(elapsed / durationMs);
  }, 120);
  updateDeckAutoStatus();
}

function toggleDeckAutoPlay() {
  if (state.deckAutoPlay) {
    stopDeckAutoPlay(true);
  } else {
    startDeckAutoPlay();
  }
}

function updatePresentationBadge() {
  if (!el.presentationBadge) return;
  if (state.deckEnabled) {
    const modeLabel = PRESENTATION_MODE_LABELS[normalizePresentationMode(state.presentationMode)] || "Dengeli";
    el.presentationBadge.textContent = `Sunum Modu Aktif | ${modeLabel}`;
    el.presentationBadge.classList.add("active");
  } else {
    el.presentationBadge.textContent = "Sunum Modu Pasif";
    el.presentationBadge.classList.remove("active");
  }
}

function setPresentationDeckEnabled(enabled) {
  state.deckEnabled = Boolean(enabled);
  document.body.classList.toggle("presentation-deck", state.deckEnabled);
  if (el.presentationModeBtn) {
    el.presentationModeBtn.textContent = state.deckEnabled ? "Sunumdan Cik" : "Sunum Modu";
  }
  if (!state.deckEnabled) {
    stopDeckAutoPlay(true);
    state.focusIndex = -1;
    clearFocusPanels();
  }
  updatePresentationBadge();
  updateDeckAutoStatus();
}

function focusPanelAt(index) {
  const targets = getPresentationFocusTargets();
  if (!targets.length) return;
  const safeIndex = ((index % targets.length) + targets.length) % targets.length;
  state.focusIndex = safeIndex;
  clearFocusPanels();
  const target = targets[state.focusIndex];
  target.classList.add("focused");
  target.scrollIntoView({ behavior: "smooth", block: "start" });
}

function focusNextPanel(fromAuto = false) {
  const targets = getPresentationFocusTargets();
  if (!targets.length) return;
  focusPanelAt(state.focusIndex + 1);
  if (state.deckAutoPlay && !fromAuto) {
    state.deckCycleStartMs = Date.now();
    setDeckProgress(0);
  }
}

function focusPrevPanel() {
  const targets = getPresentationFocusTargets();
  if (!targets.length) return;
  focusPanelAt(state.focusIndex - 1);
  if (state.deckAutoPlay) {
    state.deckCycleStartMs = Date.now();
    setDeckProgress(0);
  }
}

function p(text, className = "") {
  const node = document.createElement("p");
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function li(text) {
  const node = document.createElement("li");
  node.textContent = text;
  return node;
}

function toReadableDate(input) {
  if (!input) return "-";
  const date = new Date(input.replace(" ", "T"));
  if (Number.isNaN(date.getTime())) return input;
  return date.toLocaleString("tr-TR", {
    year: "numeric",
    month: "long",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function fmtNum(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function variableLabel(value) {
  return LIVE_QUANT_VAR_LABELS[String(value || "").trim().toLowerCase()] || String(value || "-");
}

function renderLiveQuant() {
  if (!el.liveQuantMeta || !el.liveQuantCards || !el.liveQuantGallery) return;
  clearNode(el.liveQuantCards);
  clearNode(el.liveQuantGallery);

  const payload = state.liveQuant;
  if (!payload || payload.error) {
    el.liveQuantMeta.textContent = payload?.error || "Canli quant verisi henuz yok.";
    return;
  }

  const run = payload.run || {};
  const items = Array.isArray(payload.items) ? payload.items : [];
  const quantItems = items.filter((item) => item.chartPng);
  el.liveQuantMeta.textContent =
    `Run: ${run.id || "-"} | guncelleme: ${toReadableDate(run.updatedAt || "")} | mod: ${run.analysisMode || "-"} | pencere: ${run.historyStartParam || "-"} -> ${run.historyEndParam || "-"} | quant grafik: ${quantItems.length}/${items.length}`;

  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "card";
    card.appendChild(p(variableLabel(item.variable), "label"));
    const hs = item.historyStart || "-";
    const he = item.historyEnd || "-";
    card.appendChild(p(`${hs} -> ${he}`, "value"));
    card.appendChild(p(`Anomali: ${item.anomalyRows || 0} | Tarihsel nokta: ${item.historyRows || 0}`, "muted"));
    if (item.chartPng) {
      card.appendChild(p("Grafik: Quant model secimi", "muted"));
    } else {
      card.appendChild(p("Grafik: Quant model grafigi yok", "muted"));
    }
    if (Number(item.forecastRows || 0) > 0) {
      card.appendChild(p(`Uyari: ${item.forecastRows} adet forecast satiri var`, "muted"));
    }
    el.liveQuantCards.appendChild(card);

    if (item.chartPng) {
      const figure = document.createElement("figure");
      const link = document.createElement("a");
      const cacheBuster = `ts=${Date.now()}`;
      const fileUrl = `/files/${encodeURIComponent(item.chartPng).replace(/%2F/g, "/")}`;
      link.href = fileUrl;
      link.target = "_blank";
      link.rel = "noopener noreferrer";

      const img = document.createElement("img");
      img.loading = "lazy";
      img.src = `${fileUrl}?${cacheBuster}`;
      img.alt = `${variableLabel(item.variable)} quant chart`;

      const cap = document.createElement("figcaption");
      cap.textContent = `${variableLabel(item.variable)} | quant model`;

      link.appendChild(img);
      figure.appendChild(link);
      figure.appendChild(cap);
      el.liveQuantGallery.appendChild(figure);
    }
  });

  if (!quantItems.length) {
    el.liveQuantGallery.appendChild(p("Bu kosuda gosterilecek quant model grafigi bulunamadi.", "muted"));
  }
}

async function loadLiveQuant() {
  try {
    const payload = await fetchJSON("/api/live-quant");
    state.liveQuant = payload;
  } catch (error) {
    state.liveQuant = { error: String(error) };
  }
  renderLiveQuant();
}

function startLiveQuantAutoRefresh() {
  if (liveQuantTimerId) window.clearInterval(liveQuantTimerId);
  liveQuantTimerId = window.setInterval(() => {
    loadLiveQuant();
  }, 20000);
}

function pickDefaultRun(runs) {
  if (!runs.length) return null;
  let best = runs[0];
  let bestScore = ((best.models_requested || []).length * 100) + ((best.models_ok || []).length * 10) + (best.health_available ? 25 : 0);
  runs.forEach((run) => {
    const score = ((run.models_requested || []).length * 100) + ((run.models_ok || []).length * 10) + (run.health_available ? 25 : 0);
    if (score > bestScore) {
      best = run;
      bestScore = score;
    }
  });
  return best.id;
}

function statusClass(status) {
  const map = {
    ok: "status-ok",
    failed: "status-fail",
    failed_missing_summary: "status-fail",
    requested: "status-requested",
    artifact: "status-artifact",
    not_requested: "status-not_requested",
    skipped_missing_inputs: "status-artifact",
    skipped_bad_schema: "status-artifact",
    skipped_date_mismatch: "status-artifact",
    skipped_no_overlap: "status-artifact",
  };
  return map[status] || "status-not_requested";
}

function statusLabel(status) {
  const map = {
    ok: "BASARILI",
    failed: "HATALI",
    failed_missing_summary: "HATALI",
    requested: "ISTENDI",
    artifact: "KLASOR VAR",
    not_requested: "YOK",
    skipped_missing_inputs: "ATLANDI",
    skipped_bad_schema: "ATLANDI",
    skipped_date_mismatch: "ATLANDI",
    skipped_no_overlap: "ATLANDI",
  };
  return map[status] || "YOK";
}

function runResultState(okValue) {
  if (okValue === true) return { label: "OK", className: "status-ok" };
  if (okValue === false) return { label: "FAIL", className: "status-fail" };
  return { label: "UNKNOWN", className: "status-artifact" };
}

function getFilteredRuns() {
  const needle = String(state.runSearchText || "").trim().toLowerCase();
  return state.runs.filter((run) => {
    if (state.runFilter === "suite" && run.kind !== "suite") return false;
    if (state.runFilter === "health" && !run.health_available) return false;
    if (state.runFilter === "artifact" && run.kind !== "artifact") return false;

    if (!needle) return true;
    const haystack = [
      run.id,
      run.updated_at,
      run.kind,
      run.observations_used,
      run.health_available ? "health saglik" : "",
      ...(run.models_requested || []),
      ...(run.models_ok || []),
      ...(run.models_failed || []),
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(needle);
  });
}

function renderRunFilters() {
  if (!el.runFilterButtons.length) return;
  el.runFilterButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.filter === state.runFilter);
  });
}

function renderArchiveStats() {
  if (!el.runArchiveStats) return;
  clearNode(el.runArchiveStats);

  const filteredRuns = getFilteredRuns();
  const stats = [
    { label: "Arsiv", value: state.runs.length },
    { label: "Gorunen", value: filteredRuns.length },
    { label: "Suite", value: state.runs.filter((run) => run.kind === "suite").length },
    { label: "Saglik", value: state.runs.filter((run) => run.health_available).length },
  ];

  stats.forEach((stat) => {
    const card = document.createElement("div");
    card.className = "archive-stat";
    card.appendChild(p(stat.label, "archive-stat-label"));
    card.appendChild(p(String(stat.value), "archive-stat-value"));
    el.runArchiveStats.appendChild(card);
  });
}

function reconcileCurrentRunWithFilters() {
  const filteredRuns = getFilteredRuns();
  if (!filteredRuns.length) return false;
  if (filteredRuns.some((run) => run.id === state.currentRunId)) return false;
  state.currentRunId = filteredRuns[0].id;
  state.currentFeatureKey = null;
  return true;
}

function getRunOverview(detail) {
  const summary = detail.summary || {};
  const health = detail.health_suite || {};
  const requested = (summary.models_requested || []).length;
  const ok = (summary.models_ok || []).length;
  const fail = (summary.models_failed || []).length;
  const healthRequested = (health.models_requested || []).length;
  const healthOk = (health.models_ok || []).length;
  const healthFail = (health.models_failed || []).length;
  const presentModels = Object.values(detail.model_status || {}).filter((row) => row.present).length;
  const supportedModels = (detail.supported_models || []).length;
  const totalFeatures = (detail.feature_inventory || []).length;
  const activeFeatures = (detail.feature_inventory || []).filter((item) => Number(item.count || 0) > 0).length;
  const presentationAssets = getPresentationAssetsForMode(detail);

  return {
    requested,
    ok,
    fail,
    healthRequested,
    healthOk,
    healthFail,
    healthAvailable: Boolean(health.available),
    presentModels,
    supportedModels,
    totalFeatures,
    activeFeatures,
    presentationAssetsCount: presentationAssets.length,
  };
}

function setCommandCenterStatus(text, tone = "") {
  if (!el.commandCenterStatus) return;
  el.commandCenterStatus.textContent = text;
  el.commandCenterStatus.className = `command-status${tone ? ` ${tone}` : ""}`;
}

function renderHeroMetrics(detail) {
  if (!el.heroMetrics) return;
  clearNode(el.heroMetrics);
  if (!detail) {
    setCommandCenterStatus("Run seciliyor");
    return;
  }

  const summary = detail.summary || {};
  const health = detail.health_suite || {};
  const overview = getRunOverview(detail);
  const healthStatus = overview.healthAvailable ? `${overview.healthOk}/${overview.healthRequested || 0}` : "Yok";
  const cards = [
    {
      label: "Model Tamamlama",
      value: `${overview.ok}/${overview.requested || 0}`,
      meta: overview.requested ? `${overview.fail} hatali kosu` : "model talebi kayitli degil",
    },
    {
      label: "Model Kapsami",
      value: `${overview.presentModels}/${overview.supportedModels || 0}`,
      meta: `${Math.round((overview.supportedModels ? overview.presentModels / overview.supportedModels : 0) * 100)}% artifact gorunurlugu`,
    },
    {
      label: "Saglik Suite",
      value: healthStatus,
      meta: overview.healthAvailable
        ? `${health.top_risk_model || "top risk yok"} | RR ${fmtNum(health.top_risk_rr, 3)}`
        : "health suite bulunmadi",
    },
    {
      label: "Aktif Feature",
      value: String(overview.activeFeatures),
      meta: `${detail.total_files || 0} dosya | kaynak ${summary.observations_used || "-"}`,
    },
  ];

  cards.forEach((metric) => {
    const card = document.createElement("article");
    card.className = "metric-card";
    card.appendChild(p(metric.label, "label"));
    card.appendChild(p(metric.value, "value"));
    card.appendChild(p(metric.meta, "meta"));
    el.heroMetrics.appendChild(card);
  });

  if (overview.fail || overview.healthFail) {
    setCommandCenterStatus(
      `Dikkat: ${overview.fail} model ve ${overview.healthFail} saglik akisi sorunlu`,
      "warn"
    );
    return;
  }
  if (!overview.healthAvailable && overview.requested) {
    setCommandCenterStatus(
      `Kismi hazirlik: ${overview.ok}/${overview.requested} model hazir, health suite eksik`,
      "warn"
    );
    return;
  }
  if (!overview.requested && detail.total_files) {
    setCommandCenterStatus(`Artifact arsivi secili: ${detail.total_files} dosya tarandi`, "ready");
    return;
  }
  setCommandCenterStatus(
    `Hazir: ${overview.ok}/${overview.requested || 0} model ve ${healthStatus} saglik akisi raporda`,
    "ready"
  );
}

function pulseTone(ratio, forceWarn = false) {
  if (forceWarn) return "warn";
  if (ratio >= 0.8) return "ok";
  if (ratio >= 0.45) return "info";
  return "warn";
}

function renderRunPulse(detail) {
  if (!el.runPulse) return;
  clearNode(el.runPulse);
  if (!detail) {
    el.runPulse.appendChild(p("Pulse verisi bekleniyor.", "muted"));
    return;
  }

  const overview = getRunOverview(detail);
  const rows = [
    {
      label: "Model Basari",
      ratio: overview.requested ? overview.ok / overview.requested : 0,
      value: `${overview.ok}/${overview.requested || 0}`,
      note: overview.requested ? `${overview.fail} model fail` : "model cagrisi kayitli degil",
      tone: pulseTone(overview.requested ? overview.ok / overview.requested : 0, overview.fail > 0),
    },
    {
      label: "Model Kapsami",
      ratio: overview.supportedModels ? overview.presentModels / overview.supportedModels : 0,
      value: `${overview.presentModels}/${overview.supportedModels || 0}`,
      note: "desteklenen model havuzu icinde gorunen artifact",
      tone: pulseTone(overview.supportedModels ? overview.presentModels / overview.supportedModels : 0),
    },
    {
      label: "Saglik Akisi",
      ratio: overview.healthAvailable && overview.healthRequested ? overview.healthOk / overview.healthRequested : 0,
      value: overview.healthAvailable ? `${overview.healthOk}/${overview.healthRequested || 0}` : "0/0",
      note: overview.healthAvailable ? `${overview.healthFail} saglik hata kaydi` : "health suite bulunmadi",
      tone: pulseTone(
        overview.healthAvailable && overview.healthRequested ? overview.healthOk / overview.healthRequested : 0,
        !overview.healthAvailable || overview.healthFail > 0
      ),
    },
    {
      label: "Feature Dolulugu",
      ratio: overview.totalFeatures ? overview.activeFeatures / overview.totalFeatures : 0,
      value: `${overview.activeFeatures}/${overview.totalFeatures || 0}`,
      note: `${overview.presentationAssetsCount} sunum gorseli secili`,
      tone: pulseTone(overview.totalFeatures ? overview.activeFeatures / overview.totalFeatures : 0),
    },
  ];

  rows.forEach((row) => {
    const wrap = document.createElement("div");
    wrap.className = "pulse-row";

    const head = document.createElement("div");
    head.className = "pulse-head";
    head.appendChild(p(row.label, "pulse-label"));
    head.appendChild(p(row.value, "pulse-value"));

    const track = document.createElement("div");
    track.className = "pulse-track";
    const fill = document.createElement("div");
    fill.className = `pulse-fill ${row.tone}`;
    fill.style.width = `${row.ratio > 0 ? Math.max(8, Math.round(row.ratio * 100)) : 0}%`;
    track.appendChild(fill);

    wrap.appendChild(head);
    wrap.appendChild(track);
    wrap.appendChild(p(row.note, "pulse-note"));
    el.runPulse.appendChild(wrap);
  });
}

function buildSectionNavItems() {
  return SECTION_NAV_ITEMS.map((item) => ({
    ...item,
    node: document.getElementById(item.id),
  })).filter((item) => item.node);
}

function setActiveSectionNav(id) {
  state.activeSectionId = id;
  if (!el.sectionNav) return;
  Array.from(el.sectionNav.querySelectorAll(".section-btn")).forEach((button) => {
    button.classList.toggle("active", button.dataset.target === id);
  });
}

function renderSectionNav() {
  if (!el.sectionNav) return;
  clearNode(el.sectionNav);

  const items = buildSectionNavItems();
  items.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "section-btn";
    button.dataset.target = item.id;
    button.textContent = item.label;
    button.addEventListener("click", () => {
      setActiveSectionNav(item.id);
      item.node.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    el.sectionNav.appendChild(button);
  });

  if (items.length) {
    const fallbackId = items.some((item) => item.id === state.activeSectionId)
      ? state.activeSectionId
      : items[0].id;
    setActiveSectionNav(fallbackId);
  }
}

function setupSectionNavObserver() {
  if (state.sectionObserver) {
    state.sectionObserver.disconnect();
    state.sectionObserver = null;
  }
  if (!("IntersectionObserver" in window)) return;

  const items = buildSectionNavItems();
  if (!items.length) return;

  state.sectionObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible && visible.target && visible.target.id) {
        setActiveSectionNav(visible.target.id);
      }
    },
    {
      threshold: [0.25, 0.45, 0.7],
      rootMargin: "-18% 0px -55% 0px",
    }
  );

  items.forEach((item) => state.sectionObserver.observe(item.node));
}

function renderRunList() {
  clearNode(el.runList);
  if (!state.runs.length) {
    el.runList.appendChild(p("Kosu bulunamadi.", "muted"));
    return;
  }

  const filteredRuns = getFilteredRuns();
  if (!filteredRuns.length) {
    el.runList.appendChild(p("Filtre ile eslesen kosu yok.", "muted"));
    return;
  }

  filteredRuns.forEach((run) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `run-item ${run.id === state.currentRunId ? "active" : ""}`;

    const runId = document.createElement("p");
    runId.className = "run-id";
    runId.textContent = run.id;

    const runMeta = document.createElement("p");
    runMeta.className = "run-meta";
    const requestedCount = (run.models_requested || []).length;
    runMeta.textContent = `${run.updated_at} | model: ${requestedCount} | ok: ${(run.models_ok || []).length} | health: ${run.health_available ? "var" : "yok"}`;

    const tags = document.createElement("div");
    tags.className = "run-tags";
    const kindTag = document.createElement("span");
    kindTag.className = `run-tag ${run.kind === "suite" ? "suite" : "artifact"}`;
    kindTag.textContent = run.kind === "suite" ? "Suite" : "Artifact";
    tags.appendChild(kindTag);

    if (run.health_available) {
      const healthTag = document.createElement("span");
      healthTag.className = "run-tag health";
      healthTag.textContent = "Health";
      tags.appendChild(healthTag);
    }

    if ((run.models_failed || []).length) {
      const warnTag = document.createElement("span");
      warnTag.className = "run-tag warn";
      warnTag.textContent = `${run.models_failed.length} fail`;
      tags.appendChild(warnTag);
    }

    item.appendChild(runId);
    item.appendChild(runMeta);
    item.appendChild(tags);
    item.addEventListener("click", () => {
      if (state.currentRunId === run.id) return;
      state.currentRunId = run.id;
      state.currentFeatureKey = null;
      renderRunList();
      loadRun(run.id);
    });
    el.runList.appendChild(item);
  });
}

function renderSummary(detail) {
  const summary = detail.summary || {};
  const requested = (summary.models_requested || []).length;
  const ok = (summary.models_ok || []).length;
  const fail = (summary.models_failed || []).length;
  const successRate = requested > 0 ? `${Math.round((ok / requested) * 100)}%` : "-";
  const activeFeatures = (detail.feature_inventory || []).filter((x) => x.count > 0).length;
  const stabilization = summary.stabilization || {};
  const stabilizationValue = stabilization.enabled === false
    ? "Kapali"
    : `${stabilization.applied ? "Aktif" : "Pasif"} | ${stabilization.selected_strategy || "-"}`;
  const robust = summary.robust_selection || {};
  const robustSummary = robust.summary || {};
  const robustCount = Array.isArray(robustSummary.variables_selected) ? robustSummary.variables_selected.length : 0;
  const robustValue = robust.enabled === false
    ? "Kapali"
    : `${robust.ok ? "Aktif" : "Hata"} | ${robustCount} degisken`;

  const cards = [
    { label: "Kosu Klasoru", value: detail.id },
    { label: "Guncelleme Zamani", value: detail.updated_at },
    { label: "Toplam Model", value: String(requested) },
    { label: "Basari Orani", value: successRate },
    { label: "Tahmin Stabilizasyonu", value: stabilizationValue },
    { label: "Robust Model Secimi", value: robustValue },
    { label: "Aktif Feature", value: String(activeFeatures) },
    { label: "Toplam Dosya", value: String(detail.total_files || 0) },
  ];

  clearNode(el.summaryCards);
  cards.forEach((card) => {
    const box = document.createElement("div");
    box.className = "card";
    box.appendChild(p(card.label, "label"));
    box.appendChild(p(card.value, "value"));
    el.summaryCards.appendChild(box);
  });

  el.runTitle.textContent = detail.id;
  el.runMeta.textContent = `Veri kaynagi: ${summary.observations_used || "-"}`;
  el.reportDate.textContent = toReadableDate(detail.updated_at);
}

function buildGaugeCard(label, ratio, valueText, color) {
  const card = document.createElement("article");
  card.className = "gauge-card";

  const safeRatio = Number.isFinite(ratio) ? Math.max(0, Math.min(1, ratio)) : 0;
  const radius = 34;
  const circumference = 2 * Math.PI * radius;
  const dash = circumference * safeRatio;
  const gap = circumference - dash;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 96 96");
  svg.classList.add("gauge-svg");

  const bg = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  bg.setAttribute("cx", "48");
  bg.setAttribute("cy", "48");
  bg.setAttribute("r", String(radius));
  bg.setAttribute("class", "gauge-bg");

  const fg = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  fg.setAttribute("cx", "48");
  fg.setAttribute("cy", "48");
  fg.setAttribute("r", String(radius));
  fg.setAttribute("class", "gauge-fg");
  fg.setAttribute("stroke", color);
  fg.setAttribute("stroke-dasharray", `${dash} ${gap}`);

  const value = document.createElementNS("http://www.w3.org/2000/svg", "text");
  value.setAttribute("x", "48");
  value.setAttribute("y", "53");
  value.setAttribute("text-anchor", "middle");
  value.setAttribute("class", "gauge-value");
  value.textContent = valueText;

  svg.appendChild(bg);
  svg.appendChild(fg);
  svg.appendChild(value);

  const title = document.createElement("p");
  title.className = "gauge-label";
  title.textContent = label;

  card.appendChild(svg);
  card.appendChild(title);
  return card;
}

function renderVisualBoard(detail) {
  const summary = detail.summary || {};
  const health = detail.health_suite || {};
  const requested = (summary.models_requested || []).length;
  const ok = (summary.models_ok || []).length;
  const fail = (summary.models_failed || []).length;
  const modelRatio = requested > 0 ? ok / requested : 0;
  const healthRequested = (health.models_requested || []).length;
  const healthOk = (health.models_ok || []).length;
  const healthSkip = (health.models_skipped || []).length;
  const healthFail = (health.models_failed || []).length;
  const healthRatio = healthRequested > 0 ? healthOk / healthRequested : 0;
  const modelPresent = Object.values(detail.model_status || {}).filter((x) => x.present).length;
  const coverageRatio = (detail.supported_models || []).length
    ? modelPresent / (detail.supported_models || []).length
    : 0;

  clearNode(el.visualGaugeGrid);
  el.visualGaugeGrid.appendChild(
    buildGaugeCard("Model Basari", modelRatio, `${Math.round(modelRatio * 100)}%`, "#1f6f3e")
  );
  el.visualGaugeGrid.appendChild(
    buildGaugeCard("Saglik Basari", healthRatio, `${Math.round(healthRatio * 100)}%`, "#355b8d")
  );
  el.visualGaugeGrid.appendChild(
    buildGaugeCard("Model Kapsami", coverageRatio, `${Math.round(coverageRatio * 100)}%`, "#8d6a2f")
  );

  clearNode(el.visualHealthStack);
  const totalHealth = healthOk + healthSkip + healthFail;
  if (!totalHealth) {
    el.visualHealthStack.appendChild(p("Saglik dagilim verisi yok.", "muted"));
  } else {
    const segments = [
      { label: "Basarili", value: healthOk, color: "#1f6f3e" },
      { label: "Atlanan", value: healthSkip, color: "#355b8d" },
      { label: "Basarisiz", value: healthFail, color: "#9c2f26" },
    ];
    const bar = document.createElement("div");
    bar.className = "stack-track";
    segments.forEach((seg) => {
      if (!seg.value) return;
      const block = document.createElement("div");
      block.className = "stack-seg";
      block.style.width = `${(seg.value / totalHealth) * 100}%`;
      block.style.background = seg.color;
      block.title = `${seg.label}: ${seg.value}`;
      bar.appendChild(block);
    });
    const legend = document.createElement("div");
    legend.className = "stack-legend";
    segments.forEach((seg) => {
      const item = document.createElement("p");
      item.className = "stack-legend-item";
      item.textContent = `${seg.label}: ${seg.value}`;
      legend.appendChild(item);
    });
    el.visualHealthStack.appendChild(bar);
    el.visualHealthStack.appendChild(legend);
  }

  clearNode(el.visualRiskBars);
  const riskRows = (health.rows || [])
    .filter((x) => x.status === "ok" && x.future_mean_rr !== null && x.future_mean_rr !== undefined)
    .sort((a, b) => Number(b.future_mean_rr) - Number(a.future_mean_rr))
    .slice(0, 6);
  if (!riskRows.length) {
    el.visualRiskBars.appendChild(p("Risk bar verisi yok.", "muted"));
  } else {
    const maxRR = Number(riskRows[0].future_mean_rr) || 1;
    riskRows.forEach((row) => {
      const wrap = document.createElement("div");
      wrap.className = "risk-row";
      const label = document.createElement("p");
      label.className = "risk-label";
      label.textContent = `${row.model} (${fmtNum(row.future_mean_rr, 3)})`;
      const track = document.createElement("div");
      track.className = "risk-track";
      const fill = document.createElement("div");
      fill.className = "risk-fill";
      fill.style.width = `${Math.max(6, (Number(row.future_mean_rr) / maxRR) * 100)}%`;
      track.appendChild(fill);
      wrap.appendChild(label);
      wrap.appendChild(track);
      el.visualRiskBars.appendChild(wrap);
    });
  }

  clearNode(el.visualModelMatrix);
  const matrixStatuses = detail.model_status || {};
  (detail.supported_models || []).forEach((model) => {
    const info = matrixStatuses[model] || { status: "not_requested" };
    const cell = document.createElement("div");
    cell.className = "matrix-cell";
    const name = document.createElement("p");
    name.className = "matrix-name";
    name.textContent = model;
    const status = document.createElement("p");
    status.className = `matrix-status ${statusClass(info.status)}`;
    status.textContent = statusLabel(info.status);
    cell.appendChild(name);
    cell.appendChild(status);
    el.visualModelMatrix.appendChild(cell);
  });
}

function renderIntegration(detail) {
  clearNode(el.integrationList);
  const summary = detail.summary || {};
  const healthInSummary = summary.health_suite || {};
  const healthSuite = detail.health_suite || {};
  const robust = summary.robust_selection || {};
  const robustSummary = robust.summary || {};
  const robustVars = Array.isArray(robustSummary.variables_selected) ? robustSummary.variables_selected : [];
  const robustVarsText = robustVars.length ? robustVars.join(", ") : "-";
  const healthSummaryStatus = healthInSummary.enabled
    ? (healthInSummary.ok ? "basarili" : "kismi/basarisiz")
    : "devre disi";
  const stabilization = summary.stabilization || {};
  const stabilizationEnabled = stabilization.enabled !== false;
  const stabilizationStatus = stabilization.status || (stabilizationEnabled ? "ok" : "disabled");
  const stabilizationStrategy = stabilization.selected_strategy || "-";
  const recentRatio = Number(stabilization.recent_ratio);
  const recentRatioText = Number.isFinite(recentRatio) ? `${Math.round(recentRatio * 100)}%` : "-";

  const integrationItems = [
    `Model suite: ${((summary.models_ok || []).length)}/${(summary.models_requested || []).length} model basarili.`,
    `Girdi stabilizasyonu: ${stabilizationEnabled ? `${stabilizationStatus} (${stabilizationStrategy}, recent oran ${recentRatioText})` : "kapali"}.`,
    `Robust model secimi: ${robust.enabled === false ? "kapali" : `${robust.ok ? "basarili" : "hatali"} (degisken: ${robustVarsText})`}.`,
    `Health suite (model summary): ${healthSummaryStatus}.`,
    `Health suite (dashboard): ${
      healthSuite.available
        ? `${(healthSuite.models_ok || []).length} basarili, ${(healthSuite.models_skipped || []).length} atlandi, ${(healthSuite.models_failed || []).length} hatali`
        : "bulunamadi"
    }.`,
    `Entegre komut: scripts/run_integrated_pipeline.sh ile model+health tek akista calisir.`,
  ];

  integrationItems.forEach((text) => {
    el.integrationList.appendChild(li(text));
  });
}

function renderPresentationNote(detail) {
  const note = detail.presentation_note || "Sunum notu bulunamadi.";
  el.presentationNote.textContent = note;
}

function normalizePresentationMode(mode) {
  const x = String(mode || "").trim().toLowerCase();
  if (x === "health" || x === "performance" || x === "balanced") return x;
  return "balanced";
}

function getPresentationAssetsForMode(detail) {
  const mode = normalizePresentationMode(state.presentationMode);
  const byMode = detail.presentation_assets_by_mode || {};
  const selected = byMode[mode];
  if (Array.isArray(selected)) return selected;
  return detail.presentation_assets || [];
}

function renderPresentationModeButtons() {
  const mode = normalizePresentationMode(state.presentationMode);
  const map = [
    { mode: "balanced", btn: el.presentationModeBalancedBtn },
    { mode: "health", btn: el.presentationModeHealthBtn },
    { mode: "performance", btn: el.presentationModePerformanceBtn },
  ];
  map.forEach((item) => {
    if (!item.btn) return;
    item.btn.classList.toggle("active", item.mode === mode);
  });
  updatePresentationBadge();
}

function renderPresentationAssets(detail) {
  const mode = normalizePresentationMode(state.presentationMode);
  const modeLabel = PRESENTATION_MODE_LABELS[mode] || mode;
  const assets = getPresentationAssetsForMode(detail);
  if (el.presentationAssetsMeta) {
    el.presentationAssetsMeta.textContent = assets.length
      ? `${modeLabel}: ${assets.length} quant model grafigi secildi. Bu set sunum paketine otomatik eklenir.`
      : "Bu kosuda sunuma uygun quant model grafigi bulunamadi.";
  }
  if (!el.presentationAssetsGallery) return;

  clearNode(el.presentationAssetsGallery);
  if (!assets.length) return;

  assets.forEach((asset) => {
    const figure = document.createElement("figure");
    const link = document.createElement("a");
    link.href = `/files/${encodeURIComponent(asset.path).replace(/%2F/g, "/")}`;
    link.target = "_blank";
    link.rel = "noopener noreferrer";

    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = `/files/${encodeURIComponent(asset.path).replace(/%2F/g, "/")}`;
    img.alt = asset.label || asset.path;

    const cap = document.createElement("figcaption");
    cap.textContent = `${asset.group || "run"} | ${asset.label || asset.path.split("/").slice(-1)[0]}`;

    link.appendChild(img);
    figure.appendChild(link);
    figure.appendChild(cap);
    el.presentationAssetsGallery.appendChild(figure);
  });
}

function triggerDownload(url) {
  const a = document.createElement("a");
  a.href = url;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function runExportUrl(kind) {
  if (!state.currentRunId) return "";
  const runId = encodeURIComponent(state.currentRunId);
  const mode = encodeURIComponent(normalizePresentationMode(state.presentationMode));
  if (kind === "note_pdf") return `/api/export_note?id=${runId}&format=pdf`;
  if (kind === "note_txt") return `/api/export_note?id=${runId}&format=txt`;
  if (kind === "note_md") return `/api/export_note?id=${runId}&format=md`;
  if (kind === "summary_json") return `/api/export_summary?id=${runId}`;
  if (kind === "pack_zip") return `/api/export_pack?id=${runId}&mode=${mode}`;
  if (kind === "charts_zip") return `/api/export_charts?id=${runId}&mode=${mode}`;
  return "";
}

function renderExecutive(detail) {
  const summary = detail.summary || {};
  const requested = summary.models_requested || [];
  const ok = summary.models_ok || [];
  const fail = summary.models_failed || [];
  const successRate = requested.length ? `${Math.round((ok.length / requested.length) * 100)}%` : "-";
  const modelsPresent = Object.values(detail.model_status || {}).filter((m) => m.present).length;
  const activeFeatures = (detail.feature_inventory || []).filter((x) => x.count > 0).length;
  const stabilization = summary.stabilization || {};
  const robust = summary.robust_selection || {};
  const robustSummary = robust.summary || {};

  clearNode(el.executiveList);
  el.executiveList.appendChild(
    li(
      `Bu kosuda ${requested.length} model istendi; ${ok.length} model basarili, ${fail.length} model hatali. Basari orani ${successRate}.`
    )
  );
  el.executiveList.appendChild(
    li(`Model kapsami: desteklenen ${detail.supported_models.length} modelin ${modelsPresent} tanesinde cikti klasoru mevcut.`)
  );
  el.executiveList.appendChild(
    li(`Feature kapsami: ${activeFeatures} farkli feature kategorisi aktif, toplam ${detail.total_files || 0} dosya tarandi.`)
  );
  if (stabilization.enabled !== false) {
    const ratio = Number(stabilization.recent_ratio);
    const ratioText = Number.isFinite(ratio) ? `${Math.round(ratio * 100)}%` : "-";
    const sourcePath = summary.observations_used || "-";
    el.executiveList.appendChild(
      li(`Tahmin stabilizasyonu: ${stabilization.selected_strategy || "-"} (durum: ${stabilization.status || "-"}, recent oran: ${ratioText}). Girdi: ${sourcePath}.`)
    );
  }
  if (robust.enabled !== false) {
    const selected = Array.isArray(robustSummary.selected_models) ? robustSummary.selected_models : [];
    const compact = selected
      .slice(0, 4)
      .map((x) => `${x.variable}:${x.model_key}(${x.confidence_grade || "-"})`)
      .join(", ");
    const more = selected.length > 4 ? ` (+${selected.length - 4})` : "";
    const statusText = robust.ok ? "ok" : "hata";
    el.executiveList.appendChild(
      li(`Robust secim: ${statusText}. Secilen model seti: ${compact || "-"}${more}.`)
    );
  }
  if (fail.length) {
    el.executiveList.appendChild(li(`Risk notu: hatali modeller (${fail.join(", ")}) icin tekrar kosu onerilir.`));
  } else {
    el.executiveList.appendChild(li("Risk notu: secili kosuda raporlanan model hatasi bulunmuyor."));
  }
}

function renderHealthSummary(detail) {
  clearNode(el.healthSummaryCards);
  clearNode(el.healthTableBody);

  const health = detail.health_suite || {};
  if (!health.available) {
    el.healthSummaryCards.appendChild(p("Saglik suite ozeti bu kosuda bulunmadi.", "muted"));
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 6;
    td.textContent = "Saglik analizi icin: python3 scripts/run_health_suite.py --run-dir <kosu_klasoru>";
    tr.appendChild(td);
    el.healthTableBody.appendChild(tr);
    return;
  }

  const cards = [
    { label: "Saglik Modeli (Istenen)", value: String((health.models_requested || []).length) },
    { label: "Basarili", value: String((health.models_ok || []).length) },
    { label: "Atlanan", value: String((health.models_skipped || []).length) },
    { label: "Basarisiz", value: String((health.models_failed || []).length) },
    { label: "En Riskli Model (RR)", value: health.top_risk_model ? `${health.top_risk_model} (${fmtNum(health.top_risk_rr, 3)})` : "-" },
    {
      label: "Future Donemi",
      value:
        health.future_start && health.future_end
          ? `${health.future_start}-${health.future_end}`
          : "-",
    },
  ];

  cards.forEach((card) => {
    const box = document.createElement("div");
    box.className = "card";
    box.appendChild(p(card.label, "label"));
    box.appendChild(p(card.value, "value"));
    el.healthSummaryCards.appendChild(box);
  });

  const rows = health.rows || [];
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 6;
    td.textContent = "Saglik satiri bulunamadi.";
    tr.appendChild(td);
    el.healthTableBody.appendChild(tr);
    return;
  }

  rows.forEach((row) => {
    const tr = document.createElement("tr");

    const tdModel = document.createElement("td");
    tdModel.textContent = row.model || "-";

    const tdStatus = document.createElement("td");
    tdStatus.textContent = statusLabel(row.status || "not_requested");
    tdStatus.className = statusClass(row.status || "not_requested");

    const tdRR = document.createElement("td");
    tdRR.textContent = fmtNum(row.future_mean_rr, 4);

    const tdHI = document.createElement("td");
    tdHI.textContent = fmtNum(row.future_mean_heat_index_c, 2);

    const tdShare = document.createElement("td");
    const share = row.future_high_risk_share;
    tdShare.textContent = share === null || share === undefined || Number.isNaN(Number(share)) ? "-" : `${fmtNum(Number(share) * 100, 1)}%`;

    const tdMsg = document.createElement("td");
    tdMsg.textContent = row.message || "-";

    tr.appendChild(tdModel);
    tr.appendChild(tdStatus);
    tr.appendChild(tdRR);
    tr.appendChild(tdHI);
    tr.appendChild(tdShare);
    tr.appendChild(tdMsg);
    el.healthTableBody.appendChild(tr);
  });
}

function renderResultTable(detail) {
  clearNode(el.resultTableBody);
  const rows = (detail.summary && detail.summary.results) || [];
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.textContent = "Model sonucu bulunamadi.";
    tr.appendChild(td);
    el.resultTableBody.appendChild(tr);
    return;
  }

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const tdName = document.createElement("td");
    const tdStatus = document.createElement("td");
    const tdCode = document.createElement("td");
    const tdOut = document.createElement("td");

    const rs = runResultState(row.ok);
    tdName.textContent = row.name || "-";
    tdStatus.textContent = rs.label;
    tdStatus.className = rs.className;
    tdCode.textContent = String(row.returncode ?? "-");
    tdOut.textContent = row.output_dir || "-";

    tr.appendChild(tdName);
    tr.appendChild(tdStatus);
    tr.appendChild(tdCode);
    tr.appendChild(tdOut);
    el.resultTableBody.appendChild(tr);
  });
}

function renderLogs(detail) {
  clearNode(el.logSection);
  const rows = (detail.summary && detail.summary.results) || [];
  if (!rows.length) {
    el.logSection.appendChild(p("Teknik log bulunamadi.", "muted"));
    return;
  }

  let hasAnyLog = false;
  rows.forEach((row) => {
    const logBlocks = [
      { title: `${row.name || "model"} stdout`, content: row.stdout_tail || "" },
      { title: `${row.name || "model"} stderr`, content: row.stderr_tail || "" },
    ];

    logBlocks.forEach((block) => {
      if (!block.content.trim()) return;
      hasAnyLog = true;
      const card = document.createElement("article");
      card.className = "log-card";
      const title = document.createElement("p");
      title.className = "log-title";
      title.textContent = block.title;
      const content = document.createElement("pre");
      content.className = "log-content";
      content.textContent = block.content;
      card.appendChild(title);
      card.appendChild(content);
      el.logSection.appendChild(card);
    });
  });

  if (!hasAnyLog) {
    el.logSection.appendChild(p("Kaydedilmis stdout/stderr ozeti bulunamadi.", "muted"));
  }
}

function renderFullRunCommand(detail) {
  const summary = detail.summary || {};
  const health = detail.health_suite || {};
  const stabilization = summary.stabilization || {};
  const stabilizationGap = Number(stabilization.gap_years);
  const stabilizationRecentRatio = Number(stabilization.min_recent_ratio);
  const gapYears = Number.isFinite(stabilizationGap) ? stabilizationGap : 5;
  const minRecentRatio = Number.isFinite(stabilizationRecentRatio) ? stabilizationRecentRatio : 0.25;
  const models = (detail.supported_models || []).join(",");
  const obsPath = summary.observations_used || "output/forecast_package/observations_with_graph.parquet";
  const sourceArg = obsPath
    ? `--prepared-observations ${obsPath}`
    : "--dataset /Users/yasinkaya/Hackhaton/DATA";
  const resultRows = summary.results || [];
  const walkforwardRow = resultRows.find((x) => x.name === "walkforward");
  const targetYearMatch = String((resultRows.find((x) => x.name === "quant") || {}).command || "").match(/--target-year\s+(\d{4})/);
  const startYearMatch = String((walkforwardRow || {}).command || "").match(/--start-year\s+(\d{4})/);
  const targetYear = Number(health.future_end || (targetYearMatch ? targetYearMatch[1] : 2035));
  const startYear = Number(health.future_start || (startYearMatch ? startYearMatch[1] : 2026));

  const suiteCmd = [
    "./scripts/run_integrated_pipeline.sh",
    sourceArg,
    `--models ${models}`,
    "--variables *",
    "--include-visuals true",
    '--graph-root "/Users/yasinkaya/Hackhaton/DATA/Graf Kağıtları Tarama "',
    "--stabilize-observations true",
    `--stabilization-gap-years ${gapYears}`,
    `--stabilization-min-recent-ratio ${minRecentRatio}`,
    "--run-robust-selection true",
    `--target-year ${targetYear}`,
    `--start-year ${startYear}`,
    "--walkforward-freqs YS,MS,W,D",
    "--best-meta-auto-select-combination true",
    "--run-health-suite true",
    "--health-models '*'",
    "--health-output-subdir health",
    `--output-dir output/model_suite_full_features_${targetYear}`,
  ].join(" \\\n  ");
  el.fullRunCommand.textContent = suiteCmd;
}

function renderModelCoverage(detail) {
  clearNode(el.modelCoverage);
  const statuses = detail.model_status || {};
  const supported = detail.supported_models || [];

  supported.forEach((model) => {
    const info = statuses[model] || { status: "not_requested", present: false, output_dir: "" };
    const card = document.createElement("article");
    card.className = "model-card";

    const name = document.createElement("p");
    name.className = "model-name";
    name.textContent = model;

    const status = document.createElement("p");
    status.className = statusClass(info.status);
    status.textContent = statusLabel(info.status);

    const meta = document.createElement("p");
    meta.className = "model-meta";
    meta.textContent = info.output_dir || "-";

    card.appendChild(name);
    card.appendChild(status);
    card.appendChild(meta);
    el.modelCoverage.appendChild(card);
  });
}

function renderFeatureInventory(detail) {
  clearNode(el.featureInventory);
  const inventory = detail.feature_inventory || [];
  if (!inventory.length) {
    el.featureInventory.appendChild(p("Feature envanteri bulunamadi.", "muted"));
    return;
  }

  const available = inventory.filter((f) => f.count > 0);
  if (!state.currentFeatureKey || !inventory.some((f) => f.key === state.currentFeatureKey)) {
    state.currentFeatureKey = available.length ? available[0].key : inventory[0].key;
  }

  inventory.forEach((feature) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `feature-card ${feature.key === state.currentFeatureKey ? "active" : ""}`;
    button.disabled = feature.count === 0;

    const title = document.createElement("p");
    title.className = "feature-name";
    title.textContent = feature.title;
    const count = document.createElement("p");
    count.className = "feature-count";
    count.textContent = `${feature.count} dosya`;

    button.appendChild(title);
    button.appendChild(count);
    button.addEventListener("click", () => {
      state.currentFeatureKey = feature.key;
      renderFeatureInventory(detail);
      renderFeatureDetail(detail);
    });
    el.featureInventory.appendChild(button);
  });
}

function renderCSVDetail(items) {
  const wrap = document.createElement("div");
  wrap.className = "preview-grid";
  items.forEach((file) => {
    const card = document.createElement("article");
    card.className = "preview-card";
    const header = document.createElement("p");
    header.className = "preview-title";
    header.textContent = file.path;
    card.appendChild(header);

    const table = document.createElement("table");
    table.className = "mini-table";
    const thead = document.createElement("thead");
    const headTr = document.createElement("tr");
    (file.columns || []).slice(0, 8).forEach((column) => {
      const th = document.createElement("th");
      th.textContent = column;
      headTr.appendChild(th);
    });
    thead.appendChild(headTr);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    (file.rows || []).forEach((row) => {
      const tr = document.createElement("tr");
      row.slice(0, 8).forEach((cell) => {
        const td = document.createElement("td");
        td.textContent = cell;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    card.appendChild(table);
    wrap.appendChild(card);
  });
  return wrap;
}

function renderTextDetail(items) {
  const wrap = document.createElement("div");
  wrap.className = "log-grid";
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "log-card";
    const title = document.createElement("p");
    title.className = "log-title";
    title.textContent = item.path;
    const content = document.createElement("pre");
    content.className = "log-content";
    content.textContent = item.snippet || "";
    card.appendChild(title);
    card.appendChild(content);
    wrap.appendChild(card);
  });
  return wrap;
}

function renderImageDetail(items) {
  const gallery = document.createElement("div");
  gallery.className = "feature-gallery";
  items.forEach((item) => {
    const figure = document.createElement("figure");
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = `/files/${encodeURIComponent(item.path).replace(/%2F/g, "/")}`;
    img.alt = item.path;
    const cap = document.createElement("figcaption");
    cap.textContent = item.path.split("/").slice(-1)[0];
    figure.appendChild(img);
    figure.appendChild(cap);
    gallery.appendChild(figure);
  });
  return gallery;
}

function renderFileDetail(items) {
  const list = document.createElement("ul");
  list.className = "file-list";
  items.forEach((item) => {
    const liNode = document.createElement("li");
    const link = document.createElement("a");
    link.href = `/files/${encodeURIComponent(item.path).replace(/%2F/g, "/")}`;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = item.path;
    liNode.appendChild(link);
    list.appendChild(liNode);
  });
  return list;
}

function renderFeatureDetail(detail) {
  clearNode(el.featureDetail);
  const key = state.currentFeatureKey;
  const data = (detail.feature_data || {})[key];
  if (!data) {
    el.featureTitle.textContent = "Feature secili degil.";
    el.featureDetail.appendChild(p("Detay bulunamadi.", "muted"));
    return;
  }

  el.featureTitle.textContent = `${data.title} | ${data.items.length} onizleme`;
  if (!data.items.length) {
    el.featureDetail.appendChild(p("Bu feature icin dosya yok.", "muted"));
    return;
  }

  if (data.type === "csv") {
    el.featureDetail.appendChild(renderCSVDetail(data.items));
    return;
  }
  if (data.type === "text") {
    el.featureDetail.appendChild(renderTextDetail(data.items));
    return;
  }
  if (data.type === "image") {
    el.featureDetail.appendChild(renderImageDetail(data.items));
    return;
  }
  el.featureDetail.appendChild(renderFileDetail(data.items));
}

async function loadRuns() {
  const data = await fetchJSON("/api/runs");
  state.runs = data.runs || [];

  if (!state.currentRunId || !state.runs.some((r) => r.id === state.currentRunId)) {
    state.currentRunId = pickDefaultRun(state.runs);
  }

  renderRunFilters();
  renderArchiveStats();
  reconcileCurrentRunWithFilters();
  renderRunList();
  if (state.currentRunId) {
    await loadRun(state.currentRunId);
  }
  await loadLiveQuant();
}

async function loadRun(runId) {
  try {
    const detail = await fetchJSON(`/api/run?id=${encodeURIComponent(runId)}`);
    state.currentDetail = detail;
    renderSummary(detail);
    renderHeroMetrics(detail);
    renderRunPulse(detail);
    renderVisualBoard(detail);
    renderIntegration(detail);
    renderExecutive(detail);
    renderPresentationNote(detail);
    renderPresentationModeButtons();
    renderPresentationAssets(detail);
    renderHealthSummary(detail);
    renderResultTable(detail);
    renderLogs(detail);
    renderFullRunCommand(detail);
    renderModelCoverage(detail);
    renderFeatureInventory(detail);
    renderFeatureDetail(detail);
  } catch (error) {
    state.currentDetail = null;
    clearFocusPanels();
    el.runTitle.textContent = "Hata";
    el.runMeta.textContent = "Kosu yuklenemedi.";
    el.reportDate.textContent = "-";
    renderHeroMetrics(null);
    renderRunPulse(null);
    setCommandCenterStatus("Kosu yuklenemedi", "warn");
    clearNode(el.summaryCards);
    clearNode(el.visualGaugeGrid);
    clearNode(el.visualHealthStack);
    clearNode(el.visualRiskBars);
    clearNode(el.visualModelMatrix);
    clearNode(el.integrationList);
    clearNode(el.executiveList);
    if (el.presentationNote) el.presentationNote.textContent = "";
    renderPresentationModeButtons();
    if (el.presentationAssetsMeta) el.presentationAssetsMeta.textContent = "";
    if (el.presentationAssetsGallery) clearNode(el.presentationAssetsGallery);
    clearNode(el.healthSummaryCards);
    clearNode(el.healthTableBody);
    clearNode(el.resultTableBody);
    clearNode(el.logSection);
    clearNode(el.modelCoverage);
    clearNode(el.featureInventory);
    clearNode(el.featureDetail);
    el.featureTitle.textContent = "";
    el.summaryCards.appendChild(p(String(error), "muted"));
  }
}

async function applyRunSidebarControls() {
  renderRunFilters();
  renderArchiveStats();
  const selectionChanged = reconcileCurrentRunWithFilters();
  renderRunList();
  if (selectionChanged && state.currentRunId) {
    await loadRun(state.currentRunId);
  }
}

el.refreshBtn.addEventListener("click", async () => {
  await loadRuns();
});

if (el.runSearchInput) {
  el.runSearchInput.addEventListener("input", async () => {
    state.runSearchText = el.runSearchInput.value || "";
    await applyRunSidebarControls();
  });
}

if (el.runFilterButtons.length) {
  el.runFilterButtons.forEach((button) => {
    button.addEventListener("click", async () => {
      const nextFilter = button.dataset.filter || "all";
      if (nextFilter === state.runFilter) return;
      state.runFilter = nextFilter;
      await applyRunSidebarControls();
    });
  });
}

if (el.liveQuantRefreshBtn) {
  el.liveQuantRefreshBtn.addEventListener("click", async () => {
    await loadLiveQuant();
  });
}

if (el.presentationModeBtn) {
  el.presentationModeBtn.addEventListener("click", () => {
    setPresentationDeckEnabled(!state.deckEnabled);
  });
}

if (el.presentationFocusNextBtn) {
  el.presentationFocusNextBtn.addEventListener("click", () => {
    if (!state.deckEnabled) setPresentationDeckEnabled(true);
    focusNextPanel();
  });
}

if (el.presentationAutoBtn) {
  el.presentationAutoBtn.addEventListener("click", () => {
    toggleDeckAutoPlay();
  });
}

if (el.presentationIntervalSelect) {
  el.presentationIntervalSelect.addEventListener("change", () => {
    state.deckIntervalSec = normalizeDeckInterval(el.presentationIntervalSelect.value);
    updateDeckAutoStatus();
    if (state.deckAutoPlay) {
      startDeckAutoPlay();
    }
  });
}

window.addEventListener("keydown", (event) => {
  const tag = (document.activeElement && document.activeElement.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea") return;
  const key = String(event.key || "").toLowerCase();
  if (key === "p") {
    setPresentationDeckEnabled(!state.deckEnabled);
  }
  if (key === "a") {
    toggleDeckAutoPlay();
  }
  if (event.key === "ArrowRight") {
    if (!state.deckEnabled) setPresentationDeckEnabled(true);
    focusNextPanel();
  }
  if (event.key === "ArrowLeft") {
    if (!state.deckEnabled) setPresentationDeckEnabled(true);
    focusPrevPanel();
  }
});

if (el.presentationCopyBtn) {
  el.presentationCopyBtn.addEventListener("click", async () => {
    const text = (el.presentationNote && el.presentationNote.textContent) || "";
    if (!text.trim()) return;
    try {
      await navigator.clipboard.writeText(text);
      el.presentationCopyBtn.textContent = "Kopyalandi";
      window.setTimeout(() => {
        el.presentationCopyBtn.textContent = "Notu Kopyala";
      }, 1600);
    } catch (_) {
      el.presentationCopyBtn.textContent = "Kopya Hatasi";
      window.setTimeout(() => {
        el.presentationCopyBtn.textContent = "Notu Kopyala";
      }, 1600);
    }
  });
}

if (el.presentationDownloadTxtBtn) {
  el.presentationDownloadTxtBtn.addEventListener("click", () => {
    const url = runExportUrl("note_txt");
    if (url) triggerDownload(url);
  });
}

if (el.presentationDownloadPdfBtn) {
  el.presentationDownloadPdfBtn.addEventListener("click", () => {
    const url = runExportUrl("note_pdf");
    if (url) triggerDownload(url);
  });
}

if (el.presentationDownloadMdBtn) {
  el.presentationDownloadMdBtn.addEventListener("click", () => {
    const url = runExportUrl("note_md");
    if (url) triggerDownload(url);
  });
}

if (el.presentationDownloadJsonBtn) {
  el.presentationDownloadJsonBtn.addEventListener("click", () => {
    const url = runExportUrl("summary_json");
    if (url) triggerDownload(url);
  });
}

if (el.presentationDownloadPackBtn) {
  el.presentationDownloadPackBtn.addEventListener("click", () => {
    const url = runExportUrl("pack_zip");
    if (url) triggerDownload(url);
  });
}

if (el.presentationChartsPackBtn) {
  el.presentationChartsPackBtn.addEventListener("click", () => {
    const url = runExportUrl("charts_zip");
    if (url) triggerDownload(url);
  });
}

function bindPresentationModeButton(button, mode) {
  if (!button) return;
  button.addEventListener("click", () => {
    state.presentationMode = normalizePresentationMode(mode);
    renderPresentationModeButtons();
    if (state.currentDetail) {
      renderRunPulse(state.currentDetail);
      renderPresentationAssets(state.currentDetail);
    }
  });
}

bindPresentationModeButton(el.presentationModeBalancedBtn, "balanced");
bindPresentationModeButton(el.presentationModeHealthBtn, "health");
bindPresentationModeButton(el.presentationModePerformanceBtn, "performance");

if (el.fullRunCopyBtn) {
  el.fullRunCopyBtn.addEventListener("click", async () => {
    const text = (el.fullRunCommand && el.fullRunCommand.textContent) || "";
    if (!text.trim()) return;
    try {
      await navigator.clipboard.writeText(text);
      el.fullRunCopyBtn.textContent = "Kopyalandi";
      window.setTimeout(() => {
        el.fullRunCopyBtn.textContent = "Komutu Kopyala";
      }, 1600);
    } catch (_) {
      el.fullRunCopyBtn.textContent = "Kopya Hatasi";
      window.setTimeout(() => {
        el.fullRunCopyBtn.textContent = "Komutu Kopyala";
      }, 1600);
    }
  });
}

setPresentationDeckEnabled(initialDeckFromUrl);
renderPresentationModeButtons();
renderSectionNav();
setupSectionNavObserver();
renderRunPulse(null);
setCommandCenterStatus("Run seciliyor");
loadRuns().then(() => {
  if (initialDeckFromUrl) {
    window.setTimeout(() => {
      focusNextPanel();
    }, 320);
  }
  if (initialAutoFromUrl) {
    window.setTimeout(() => {
      startDeckAutoPlay();
    }, 420);
  }
  startLiveQuantAutoRefresh();
});
