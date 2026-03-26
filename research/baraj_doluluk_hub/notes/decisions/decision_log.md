# Decision Log

## 2026-03-12

- New product-direction decision:
  after the forecasting core is stable, the project should expose an
  interactive web scenario interface where users can modify key variables and
  instantly see how the projected total dam occupancy path changes.
- This interface is not a separate side project; it is the main public-facing
  explanation layer for the forecasting model and scenario engine.
- The first version should focus on understandable controls:
  rainfall, reference evapotranspiration (ET0), aggregate demand,
  demand restriction / savings, physical water losses (NRW), and transfer /
  external-source assumptions where available.
- The first version should emphasize visual comparison:
  base case versus user-edited scenario,
  occupancy path over time,
  and plain-language explanation of which variable caused which change.
- New future-projection decision:
  the 15-year forward product should not be presented as a single deterministic
  line. It should be built as a scenario-based projection layer with at least a
  base case, a wet/mild case, and a hot-dry/high-demand stress case.
- New forcing decision:
  future climate inputs should come from bias-corrected and downscaled climate
  scenarios or official projection deltas, then be passed through the project
  water-balance stack rather than directly extrapolating occupancy.
- New demand decision:
  future demand should be decomposed into subscriber/population growth,
  per-capita use or aggregate use intensity, physical losses / NRW, reclaimed
  water, and external-transfer assumptions where available.
- New literature-integration decision:
  the three directly relevant dam papers are no longer only background reading;
  their actionable methods are promoted into backlog:
  `Extra Trees` benchmark,
  history-only versus hybrid benchmark split,
  open-water evaporation proxy,
  recovery-lag metric,
  and monthly-seasonal-annual joint reading.
- New forward-projection implementation decision:
  the first production-style forward package is now a monthly `2026-2040`
  scenario projection built from the canonical core monthly model bundle,
  source-backed scenario deltas, and recursive storage simulation.
- Model-selection decision for forward round 1:
  `ridge_full` remains the selected recursive projection model because its
  walk-forward RMSE (`4.30` percentage points) beat the first `Extra Trees`
  benchmark (`4.40`) on the common `2011-02` to `2024-02` training window.
- New scope decision:
  reclaimed water and transfer reliability remain conceptually important but are
  still held neutral in the first `2026-2040` numeric scenario engine because
  dense monthly forward assumptions are not yet parameterized.

## 2026-03-11

- Project focus collapsed to a single core line:
  Istanbul dam occupancy forecasting and scenario-based decision support.
- Modeling target changed from equal-weight system mean to capacity-weighted total occupancy.
- New-data rainfall is treated as primary where available.
- Post-2021 rainfall currently uses fallback monthly series only for continuity.
- ET0 remains in production but its radiation block is still proxy-driven until
  actinograph observations are ingested.
- Policy scenarios are now part of the core product, not an appendix:
  rainfall shock, ET0 shock, demand growth, demand restriction, rebound, and
  combined hot-dry-high-demand stress.
- External-source registry expanded with:
  WMO No. 8, ASCE standardized ET, USGS open-water evaporation references,
  ISKI tariff and tariff-regulation documents, ISKI financial report, and
  recent hybrid/explainable reservoir forecasting literature.
- New modeling decision:
  physical water losses and sectoral tariff-based demand segmentation are now
  promoted from "nice to have" to explicit priority feature candidates.
- New data-engineering decision:
  a single monthly feature store is now the canonical join layer for
  raw `new data`, ET0, occupancy, demand, and official loss proxies.
- New feature decision:
  temperature and humidity proxies stay in the core modeling path because they
  improved walk-forward RMSE over the baseline on the observed 2011-2021 window.
- New loss-block decision:
  official ISKI annual water-loss forms are accepted as a defensible annual
  NRW proxy even before monthly loss data is available.
- New operations-context decision:
  official annual active-subscriber and reclaimed-water metrics are now part of
  the project data layer and should be used as annual demand/operations context.
- New sector-split decision:
  sectoral demand split is promoted to `partial_active`, not `active`, because
  public monthly class-level water-use volumes are still missing.
- New decision-support decision:
  annual official policy leverage should be expressed in occupancy-equivalent
  percentage points to compare NRW, demand reduction, reclaimed water, and
  subscriber growth on the same scale.
- New official-data decision:
  the monthly official city-supply reconstruction is now extended to
  `2010-2023` and should be treated as the main public monthly validation
  series for aggregate urban water input.
- New dashboard-data decision:
  the public İSKİ baraj frontend API is now an approved operational source for
  short-horizon monitoring, presentation refresh, and same-day historical
  comparison views.
- New proxy-data decision:
  Kandilli-area reanalysis radiation and ET0 series are approved as temporary
  benchmark/backfill layers until actinograph-based radiation observations are
  integrated into the production ET0 path.
- New regime-feature decision:
  NOAA monthly NAO is approved as a screened exogenous regime feature candidate
  for winter rainfall-risk context; it should be tested as a seasonal or lagged
  context block, not as a standalone direct storage predictor.
- New data-packaging decision:
  all currently useful model inputs are now consolidated into a dedicated core
  and extended monthly bundle so future experiments run from one canonical
  monthly entry point instead of ad hoc file joins.

Why this matters:

This gives the project a defensible physical core, a clearer business story,
and a direct answer to "what changes the occupancy path and by how much?"

- 2026-03-11: Three-paper methods note added. Extra Trees benchmark, source-area evaporation proxy, and recovery-lag framing promoted in backlog.
- 2026-03-12: Official `Melen + Yeşilçay` annual totals and annual treated-water
  totals are now accepted as a forward-scenario anchor for external-transfer
  sensitivity. This block is encoded explicitly as a demand-equivalent burden
  or relief layer, not yet as a separate physical inflow model, because public
  monthly forward transfer reliability data is still missing.
- 2026-03-12: `2040` sensitivity-grid outputs are now part of the projection
  package. `Rain vs demand` is accepted as a presentation-ready decision
  surface. `ET0 vs transfer` is retained as a diagnostic surface because the
  isolated ET0 sign is not yet physically stable under the current proxy-based
  ET0 block.
- 2026-03-12: A stricter model-testbench protocol is now active for Istanbul
  dam modeling. Model selection is no longer based on one-step RMSE alone;
  it now also checks recursive `1/3/6/12` month backtests, directional
  accuracy, threshold-hit accuracy, and physical sign tests.
- 2026-03-12: `hybrid_ridge` remains the primary forward model after the
  stricter testbench. `extra_trees_full` is now the main challenger / guardrail
  model because it passes all current physical sign tests even though its
  recursive error is higher.
- 2026-03-12: The forward product is no longer treated as deterministic-only.
  A probabilistic wrapper is now approved for the public scenario stack using
  benchmark-weighted model mixing (`hybrid_ridge` + `extra_trees_full`) and
  stitched `12`-month historical residual blocks.
- 2026-03-12: Probabilistic risk reporting is now part of the forward package.
  Presentation and decision materials should prefer annual `%40` / `%30`
  risk curves and `P10-P50-P90` endpoint ranges over single endpoint values
  whenever future occupancy is discussed.

