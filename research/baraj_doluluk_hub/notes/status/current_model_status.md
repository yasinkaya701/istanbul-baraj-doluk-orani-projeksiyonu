# Current Model Status

Date: 2026-03-12

Current best operational framing:

- Target:
  capacity-weighted Istanbul total dam occupancy
- Time step:
  monthly
- Training window used in deep package:
  2011-01 to 2024-02
- New-data rainfall directly present:
  2011-01 to 2021-12
- Main exogenous blocks:
  rainfall, ET0, consumption, seasonal cycle, storage memory

Current best walk-forward result:

- model: `hybrid_ridge`
- RMSE: about `4.27` percentage points on the current strict forward-core testbench

Focused deep-feature result on the strictly observed `new data` window:

- window:
  `2011-03` to `2021-12`
- best model:
  `plus_temp_humidity`
- RMSE:
  about `4.45` percentage points
- baseline on the same window:
  about `4.62` percentage points

Short-window annual-context result:

- window:
  `2020-03` to `2023-12`
- best model:
  `plus_reuse_intensity`
- RMSE:
  about `5.29` percentage points
- baseline on the same short window:
  about `5.41` percentage points

Interpretation:

- Storage memory is still the strongest block.
- Rainfall adds clearer explanatory value than aggregate demand alone.
- Demand is useful for scenario design, but current aggregate consumption is not
  yet sector-separated enough to carry the full operational story by itself.
- Temperature and humidity add measurable signal on top of the current
  `rain + ET0 + demand + memory` structure.
- VPD and rain-minus-ET0 remain useful interpretation variables, but in the
  current tests they do not outperform the baseline by themselves.
- Annual official reuse and demand-intensity context adds a small signal on the
  short `2020-2023` window.
- Annual NRW alone is better treated as a decision and governance variable than
  a standalone monthly predictor on current public data.

Current known weaknesses:

- No direct reservoir inflow observations yet.
- No direct open-water evaporation observations yet.
- No explicit recorded restriction calendar in the model yet.
- No sector-level water withdrawal split yet.
- ET0 radiation term still uses proxy/climatology logic in part of the series.
- Physical losses now have an official annual proxy, but not yet a dense monthly series.
- Public demand segmentation is now partially grounded by official annual subscriber
  and reclaimed-water metrics, but there is still no public monthly tariff-class volume series.

Priority upgrades after latest literature round:

- Expand the new annual physical water-loss / non-revenue water proxy into a denser time series
- Add tariff-informed sector split for demand
- Add observed radiation from actinograph when available
- Keep ET0 and open-water evaporation conceptually separate
- Promote the future product layer to an interactive web scenario explorer so
  non-technical users can change inputs and immediately see the occupancy-path impact
- Move the forward product from a single-line forecast to a scenario-based
  2026-2040 projection layer using climate, demand, and operations assumptions

New active data assets:

- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_model_core_monthly.csv`
- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_model_extended_monthly.csv`
- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_source_current_context.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/istanbul_newdata_monthly_climate_panel.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/istanbul_dam_driver_panel.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_iski_water_loss_annual.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_iski_operational_context_annual.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_city_supply_monthly_2010_2023.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_supply_vs_model_consumption_monthly.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_iski_source_context.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/kandilli_openmeteo_daily_1940_present.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/kandilli_openmeteo_vs_local_et0_monthly.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/noaa_cpc_nao_monthly_1950_present.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/noaa_nao_vs_istanbul_djf_seasonal.csv`
- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/gunluk_ozet.csv`
- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/son_14_gun_toplam_doluluk.csv`
- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/yillik_yagis.csv`
- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/baraj_bazli_son_10_yil_doluluk.csv`

Latest official data expansion:

- Useful data blocks are now consolidated into a dedicated model bundle with:
  a core monthly matrix,
  an extended monthly matrix,
  and a source-aware current-context table.
- Official city-supply monthly coverage now reaches `2010-01` to `2023-12`
  from activity-report tables.
- Public İSKİ frontend API has been harvested into a reproducible local package.
- Official source-level context now includes basin area, annual yield, storage,
  and commissioning year for major reservoirs and regulators.
- A Kandilli-area daily reanalysis proxy layer is now available for radiation,
  ET0, wind, precipitation, and sunshine duration.
- A monthly NAO teleconnection layer is now available for winter regime context.
- That API gives:
  current per-dam storage,
  14-day total occupancy,
  1-year month-end occupancy,
  same-day 10-year occupancy comparison,
  annual rainfall,
  treated-water series,
  and Melen/Yeşilçay transfer history.

Operational interpretation:

- `istanbul_model_core_monthly.csv` should be the default first-pass training table.
- `istanbul_model_extended_monthly.csv` should be used for richer ablation and exogenous-feature tests.
- `istanbul_source_current_context.csv` should be used for source-aware explainability and scenario analysis, not as a replacement for the aggregate monthly target.
- The project now has a directly refreshable official daily snapshot layer in
  addition to the longer monthly reconstruction layer.
- Basin-area context is now available for rainfall-to-inflow proxy design and
  for prioritizing which reservoirs need explicit open-water evaporation area work.
- The reanalysis proxy is now available as a stopgap forcing/validation layer
  until actinograph-based radiation observations arrive.
- The NAO index is now available as a regime-scale exogenous feature candidate
  for winter rainfall and occupancy-risk screening.
- The public API is especially valuable for:
  short-horizon nowcasting,
  same-day year-over-year comparisons,
  and showing operational relevance in the presentation.
- A future web interface should sit on top of this model stack so users can
  adjust scenario inputs and inspect forecast-path changes without reading
  technical tables directly.
- The next research-backed forward step is a `2026-2040` monthly scenario
  projection stack rather than a single deterministic `2040` value.
- The most defensible first projection family is:
  base,
  wet-mild,
  hot-dry-high-demand,
  and management-improvement.
- That first `2026-2040` projection stack is now generated in round 1 and saved
  under `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040`.
- Round-1 selected projection model is `ridge_full`.
- After adding the formal benchmark split, the selected round-1 projection
  model is now labeled `hybrid_ridge`.
- `history_only_ridge` versus `hybrid_ridge` comparison is now explicit and
  confirms that external climate-demand features improve the forward core.
- Empirical month-specific residual bands are now attached to forward scenario
  paths, so the package includes a first uncertainty layer rather than only
  mean paths.
- Threshold-risk and recovery-lag style summaries are now attached to the
  forward package, including first-cross, first-recovery, longest-below-threshold
  spell, and permanent-cross dates for `%40` and `%30`.
- The forward package now also includes a driver-decomposition layer that
  separates climate-side and demand-side stress, and splits management gains
  into efficiency-side and NRW-side contributions.
- The forward package now includes an explicit `drivers first` layer:
  future rainfall, ET0, demand, temperature, humidity, VPD, water-balance, and
  occupancy are charted together before scenario interpretation.
- The forward package now includes an official external-transfer sensitivity
  layer anchored to `2021-2025` Melen-Yeşilçay totals and annual treated-water
  totals from the public İSKİ frontend API family.
- The current official external-transfer anchor is about `%47.16` of annual
  treated water on average over `2021-2025`; transfer stress and relief are
  now encoded as demand-equivalent sensitivity paths on top of the main
  occupancy scenarios.
- The forward package now also includes `2040` sensitivity grids for
  `rain vs demand` and `ET0 vs transfer`.
- `rain vs demand` behaves directionally as expected and is usable for
  decision-support framing.
- `ET0` isolated sensitivity currently shows a sign-stability problem in the
  hybrid model; this is treated as a model diagnostic, not as a fully trusted
  public decision surface, until actinograph-backed ET0 reconstruction is in place.
- A new round-2 model testbench now covers:
  one-step walk-forward,
  recursive multi-step backtest at `1/3/6/12` months,
  directional accuracy,
  threshold-hit accuracy,
  and physical sign tests.
- Under that stricter testbench, `hybrid_ridge` remains the best composite
  model, while `extra_trees_full` is retained as the main challenger because
  it passes all current physical-sign tests.
- Actinograph is still pending; ET0 scenario paths therefore still inherit the
  current proxy-backed ET0 baseline rather than observed-radiation recalculation.
- A new probabilistic forward package is now active under
  `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040_probabilistic`.
- That package combines the current best statistical model (`hybrid_ridge`) and
  the main physics-clean challenger (`extra_trees_full`) with benchmark-based
  ensemble weights of about `0.681` and `0.319`.
- The new package adds block-bootstrap path uncertainty on top of the
  deterministic scenario stack: `4000` simulations with `12`-month residual
  blocks.
- Forward outputs are now available not only as mean or median paths, but also
  as monthly `P10/P25/P50/P75/P90` occupancy envelopes.
- Annual risk tables now estimate the probability that any month within a year
  falls below `%40` or `%30`, rather than only reporting deterministic first-cross dates.
- Under the new probabilistic layer, the `base` scenario reaches a `>%50`
  annual chance of at least one `%30`-below month by `2036`, while
  `hot_dry_high_demand` reaches that level by `2027`.
- Under the same probabilistic layer, `wet_mild` and `management_improvement`
  do not cross the `>%50` annual `%30`-risk level within `2026-2040`.
- This probabilistic layer is intentionally described as an uncertainty wrapper
  around deterministic projections, not yet as a full physical stochastic inflow model.
