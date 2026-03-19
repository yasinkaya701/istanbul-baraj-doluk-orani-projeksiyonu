# Solar Model Literature Notes (v4.7)

This note documents the primary references used to update
`scripts/forecast_solar_potential.py` for a more physical and stable model.

## 1) Core references (physics + decomposition)

1. FAO-56 (Allen et al., 1998), Chapter 3 (Radiation)
   - Source: https://www.fao.org/4/X0490E/x0490e07.htm
   - Used for:
     - Extraterrestrial radiation context (`Ra`)
     - Clear-sky envelope relation:
       `Rso = (0.75 + 2e-5 * z) * Ra` (Eq. 37 in FAO-56 context)

2. Erbs, Klein, Duffie (1982)
   - Source record: https://www.osti.gov/biblio/5358527
   - DOI: https://doi.org/10.1016/0038-092X(82)90302-4
   - Used for:
     - Diffuse fraction correlation (`Kd`) as a function of clearness index (`Kt`)
     - Decomposition of global radiation into beam + diffuse components

3. Reindl, Beckman, Duffie (1990)
   - Source record: https://www.osti.gov/biblio/7121753
   - DOI: https://doi.org/10.1016/0038-092X(90)90060-P
   - Used for:
     - Weather-informed regression design pattern for solar radiation estimation
     - Motivation for combining meteorological predictors in a bounded radiation model

4. Thornton, Running (1999)
   - Source: https://www.sciencedirect.com/science/article/pii/S0034425799000556
   - Used for:
     - Hybrid meteorological estimation approach and practical error expectations
     - Guidance for robust climate-driven radiation estimation when direct radiation is absent

5. NREL PVWatts v5 Manual (Dobos, 2014)
   - Source: https://www.nrel.gov/docs/fy14osti/62641.pdf
   - Used for:
     - Module temperature coefficient defaults (`gamma_pdc`) by module type
     (standard / premium / thin-film)

## 2) Recent review and benchmark studies (2024-2026)

1. Pandzic & Capuder (2024), Energies
   - DOI: https://doi.org/10.3390/en17010097
   - "Advances in Short-Term Solar Forecasting: A Review and Benchmark ..."
   - Key takeaway:
     - Forecast quality strongly depends on horizon and data source.
     - Hybrid approaches (physics + ML) tend to be more robust than single-family models.

2. Chodakowska et al. (2024), Energies
   - DOI: https://doi.org/10.3390/en17133156
   - "Solar Radiation Forecasting: A Systematic Meta-Review ..."
   - Key takeaway:
     - No single model dominates all climates/horizons.
     - Data quality and uncertainty handling are central to reproducible improvements.

3. Systematic review across horizons (2025), Solar Compass
   - DOI: https://doi.org/10.1016/j.solcom.2025.100154
   - Key takeaway:
     - Best operational performance generally comes from matching model family to horizon.
     - Physics-informed AI is increasingly recommended when enough target data exists.

4. Scientific Reports benchmark (2025), Zafarana case
   - DOI: https://doi.org/10.1038/s41598-025-24853-4
   - Key takeaway:
     - Tree-based ML can outperform complex deep models under some data regimes.
     - Feature selection (temporal + meteorological) is often decisive.

5. Scientific Reports global ML/DL study (2026)
   - DOI: https://doi.org/10.1038/s41598-026-41357-x
   - Key takeaway:
     - Global-scale ML/DL can be effective, but requires broad/consistent data and careful validation.
     - Results vary by region; local calibration remains necessary.

## 3) Changes derived from literature

1. Clear-sky ceiling added:
   - `Rso = (0.75 + 2e-5*z) * Ra`
   - Prevents physically implausible monthly radiation spikes.

2. Clearness-index workflow:
   - Estimate bounded `Kt` from meteorological predictors.
   - Compute `Rs = Kt * Ra` and clip by `Rso`.

3. Beam/diffuse decomposition:
   - Apply Erbs `Kd(Kt)` to derive:
     - `Diffuse = Kd * Rs`
     - `Beam = Rs - Diffuse`

4. PV temperature derating:
   - Use PVWatts-style `gamma_pdc` defaults (or override).
   - Apply temperature efficiency factor to radiation potential.

5. Stability upgrades:
   - Forecast-only smoothing for median and interval widths.
   - Low-sample calibration guard via `--min-calibration-points` to avoid overfitting.

6. Scientific QA checks:
   - Quantile ordering, negativity, utilization ratio, diffuse bounds,
     beam+diffuse consistency, and month-to-month variability metrics.

7. Calibration policy update:
   - Minimum overlap requirement for calibration (`--min-calibration-points`) to avoid
     overfitting from 1-2 points (consistent with review guidance on data sufficiency).

8. Forecast stability control:
   - Forecast-only smoothing (`--forecast-smoothing-alpha`) to reduce unrealistically sharp
     month-to-month jumps while preserving historical segment.

9. Horizon-aware policy with data guard:
   - Following horizon-dependent evidence from recent reviews, long-horizon blending is supported.
   - To avoid bias when history is too short, blending is auto-disabled unless enough historical
     months exist (`--min-history-for-horizon-blend`, default 12).
   - Long-horizon blending is capped and growth-controlled
     (`--horizon-long-blend-growth-per-year`, `--horizon-blend-max`) to avoid over-regularization.

10. Driver sensitivity diagnostics:
   - One-sigma central-difference sensitivity is reported for
     temperature/humidity/precipitation/pressure.
   - This quantifies each variable's marginal effect on solar potential and provides ranking.

11. Visual output upgrade:
   - Main chart now includes:
     - P10-P90 uncertainty band + P50 series,
     - stochastic realization path (AR(1)),
     - global horizontal radiation reference,
     - forecast region shading,
     - standardized climate-driver panel to show multi-variable dynamics.

12. Scenario-path upgrade for realism:
   - Long-horizon monthly expected values can appear too regular when source forecasts
     are near-repeating.
   - A bounded AR(1) realization path is generated from the uncertainty envelope
     (`--scenario-enable`, `--scenario-ar1-rho`, `--scenario-scale`) to represent
     plausible month-to-month irregularity while preserving physical limits.

13. Input-quality diagnostics:
   - The report now includes `input_diagnostics` to detect near-repeating forecast inputs
     (e.g., very low interannual month-level variance), which directly limits realism.

14. Internet-derived extra predictors:
   - Open-Meteo climate API monthly extras are integrated when available:
     - `cloud_cover_mean` (%)
     - `wind_speed_10m_mean` (km/h)
     - `shortwave_radiation_sum` (converted to kWh/m2/day)
     - `temperature_2m_mean` (C)
   - These are used as additional modifiers for clearness, PV temperature derate, and radiation blending.
   - Output now includes `cloudiness_percent` and internet extra columns for auditability.
   - Report includes `internet_consistency` diagnostics (bias, MAE, correlation) between local series
     and internet-derived temperature/cloud/radiation signals.

15. Forecast temperature assimilation:
   - Forecast-period temperature can be blended with internet temperature signal
     (`--forecast-temp-internet-blend`).
   - This mitigates repeated-pattern bias in local long-horizon temperature projections while
     preserving local signal contribution.
   - Output columns:
     - `temperature_model_c`
     - `temperature_local_c`
     - `temperature_assimilation_adjustment_c`
   - Empirical sweep on current dataset favored a high blend for forecast period (`~0.90`)
     due to much lower temperature MAE versus internet signal while keeping similar solar stability.

16. Forecast shortwave/cloud assimilation:
   - Internet shortwave radiation now nudges forecast expected solar potential
     (`--forecast-shortwave-internet-blend`).
   - Cloudiness output is blended with internet cloud cover in forecast period
     (`--forecast-cloudiness-internet-blend`).
   - This improves internet consistency diagnostics, especially cloud correlation/bias.
   - Current tuned defaults in this workspace:
     - `forecast_temp_internet_blend = 0.90`
     - `forecast_shortwave_internet_blend = 0.00`
     - `forecast_cloudiness_internet_blend = 0.70`

17. Multi-variable internet assimilation:
   - Forecast-period blending now supports humidity and precipitation in addition to temperature.
   - Precipitation internet signal is unit-harmonized to monthly totals before blending
     (`precip_internet_mm_month`) to match local monthly precipitation scale.
   - Pressure blending remains available but depends on provider coverage and unit harmonization.
   - Current defaults:
     - `forecast_humidity_internet_blend = 0.65`
     - `forecast_precip_internet_blend = 0.55`
     - `forecast_pressure_internet_blend = 0.60` (applies only if coverage exists)

18. Internet variable-scale harmonization (new in v3.8):
   - Before forecast blending, internet variables are checked against local historical overlap.
   - If scale mismatch is detected (for example pressure in different units), an affine mapping
     (`local ~= a * internet + b`) is fitted on historical overlap and applied to forecast internet signal.
   - Alignment is applied only when overlap is sufficient, correlation is acceptable, and MAE improves,
     reducing instability from unit mismatches while preserving seasonal structure.

19. Internet cache integrity guard (new in v3.8):
   - Cached internet extras are now validated for requested date-window coverage and per-signal non-null coverage.
   - Incomplete/stale caches are auto-refreshed to avoid silently using sparse columns
     (notably for pressure when provider field availability changes).

20. Assimilation quality gate + overlap fallback (new in v3.9):
   - Internet assimilation now evaluates variable-level relationship quality before blending.
   - If historical overlap is too short, quality metrics automatically fall back to all available overlap
     to avoid unstable decisions from very small samples.
   - Variables with severe scale mismatch that cannot be reliably aligned are automatically excluded
     from blending (for example pressure in this workspace), improving stability and preventing
     physically implausible shifts.

21. Seasonal bias correction + reliability-weighted blending (new in v4.0):
   - Internet signals are seasonally bias-corrected before forecast blending using month-of-year offsets
     estimated from overlap periods.
   - Blend weights are now scaled by an overlap-based reliability score (correlation + normalized error),
     reducing over-correction when internet/local agreement is weak.
   - This produces smoother and more defensible multi-variable assimilation while preserving hard guards
     for severe unit mismatch.

22. Quantile mapping correction (new in v4.1):
   - Before seasonal bias correction, internet signals are optionally distribution-mapped to local signal
     using overlap-period empirical quantiles.
   - Mapping is applied only when overlap is sufficient, correlation is acceptable, and MAE improves.
   - This reduces distribution mismatch (especially for skewed variables such as precipitation) and
     improves stability of downstream blending.

23. Satellite reference fallback for sparse solar targets (new in v4.2):
   - When local solar observations are insufficient, NASA POWER monthly
     `ALLSKY_SFC_SW_DWN` is used as a fallback reference series.
   - Local observations still have priority on overlapping months, while reference fills missing periods.
   - This enables more stable calibration in low-observation regimes and reduces dependence on
     single-point local overlaps.

24. Conformal residual uncertainty blending (new in v4.3):
   - ML uncertainty intervals now combine normal approximation with empirical conformal residual quantiles
     from holdout months.
   - This improves interval calibration under asymmetric residual distributions while preserving
     smooth behavior when holdout sample is limited.
   - Holdout default was increased to 24 months for more reliable validation and interval estimation.

25. Lead-time decay in internet assimilation weights (new in v4.4):
   - Forecast-period internet blending weights now decay with lead time using a configurable half-life.
   - A non-zero minimum decay factor preserves long-horizon signal while reducing overreaction to
     uncertain distant projections.
   - Effective per-row assimilation weights are exported for auditability.
   - To protect baseline accuracy, this decay is optional and disabled by default (`half-life=0`).

26. Lagged/rolling ML features (new in v4.5):
   - Hybrid ML correction now uses lagged (`t-1`, `t-12`) and rolling (`3`, `12` month) features for
     core meteorological drivers and heuristic radiation proxy.
   - Trend term (`year_trend`) is added to capture slow structural drift.
   - This improves monthly sequence learning and generally stabilizes validation error in long histories.

27. Time-series CV ensemble weighting (new in v4.6):
   - Ensemble member weights are now estimated using expanding-window time-series cross-validation
     on the development segment instead of a single split.
   - This reduces weight instability and improves out-of-sample robustness for long-horizon forecasts.
   - Final member models are re-fitted on all available observed-target months after validation.

28. Post-blend robustness layer (new in v4.7):
   - After ML ensemble prediction, a holdout-validated blend with the calibrated physics baseline is
     evaluated and applied only when it does not materially worsen holdout MAE.
   - This provides an additional guard against occasional ML overfit while retaining data-driven gains.
   - Blend diagnostics (`w_ml`, mixed MAE/RMSE, applied/not-applied reason) are reported for auditability.

29. Positive-weight stacking blend + clearer diagnostics (new in v4.8):
   - In addition to inverse-error ensemble candidates, the pipeline now evaluates a non-negative
     holdout stacking blend (`LinearRegression(positive=True, fit_intercept=False)`), then normalizes
     coefficients to simplex weights.
   - Final model strategy is selected with a transparent score (`MAE + 0.45 * RMSE`) and reported
     with per-strategy diagnostics for reproducibility.
   - This improved bias-variance balance on the current dataset (lower RMSE while preserving MAE gains),
     and chart outputs were simplified into:
     1) main solar potential chart (less clutter),
     2) separate driver panel chart (temperature, humidity, precipitation, pressure anomalies).

## 4) Current data limitation

In this workspace, overlapping observed solar target is currently only 1 monthly point.
Because of this, the pipeline intentionally disables calibration/ML unless enough overlap
exists (default: `--min-calibration-points 6`, `--ml-min-months 24`).

## 5) Direct code mapping (for auditability)

Implemented in:
- `/Users/yasinkaya/Hackhaton/scripts/forecast_solar_potential.py`

Key mappings:
- FAO-56 Eq.37: clear-sky envelope cap (`Rso`) in `compute_heuristic_components`.
- Erbs 1982: diffuse fraction `Kd(Kt)` in `erbs_diffuse_fraction`.
- PVWatts gamma: module-type defaults + override via `--gamma-pdc`.
- Review-driven robustness:
  - `--source-mode auto` bundle selection.
  - Calibration guard (`--min-calibration-points`).
  - Horizon-aware guard (`--horizon-aware`, `--min-history-for-horizon-blend`).
  - Horizon cap/growth control (`--horizon-long-blend-growth-per-year`, `--horizon-blend-max`).
  - Stochastic scenario path (`--scenario-enable`, `--scenario-ar1-rho`, `--scenario-scale`).
  - Internet extras (`--internet-extra-mode`, `--openmeteo-model`, cache + refresh options).
  - Input diagnostics block (`input_diagnostics`) in report.
  - Driver sensitivity block (`driver_sensitivity`) in report.
  - Auto chart export (`--output-chart`).
  - QA block in `quality_checks` report output.
