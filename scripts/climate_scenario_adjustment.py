#!/usr/bin/env python3
"""Scenario-based climate drift adjustments for forecast outputs.

This module adds configurable deterministic drift terms to forecast values.
Main intent:
- temperature: add warming drift from selected scenario pathway/rate
- humidity: derive a relative-humidity drift from temperature drift
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Approximate near-term (2025+) warming slopes by scenario.
# Units: degrees C per year.
SCENARIO_TEMP_RATES_C_PER_YEAR: dict[str, float] = {
    "none": 0.0,
    "ssp126": 0.018,
    "ssp245": 0.028,
    "ssp370": 0.036,
    "ssp585": 0.046,
}

# IPCC AR6 WG1 SPM.8 panel-a, scenario mean GSAT anomaly pathways (C).
# Anchors are sampled every 5 years from 2020 to 2100.
SCENARIO_TEMP_PATHWAY_C: dict[str, dict[int, float]] = {
    "ssp119": {
        2020: 1.210,
        2025: 1.324,
        2030: 1.423,
        2035: 1.501,
        2040: 1.554,
        2045: 1.580,
        2050: 1.584,
        2055: 1.573,
        2060: 1.546,
        2065: 1.508,
        2070: 1.461,
        2075: 1.410,
        2080: 1.357,
        2085: 1.307,
        2090: 1.261,
        2095: 1.221,
        2100: 1.188,
    },
    "ssp126": {
        2020: 1.208,
        2025: 1.334,
        2030: 1.446,
        2035: 1.543,
        2040: 1.625,
        2045: 1.689,
        2050: 1.741,
        2055: 1.783,
        2060: 1.814,
        2065: 1.834,
        2070: 1.845,
        2075: 1.850,
        2080: 1.853,
        2085: 1.855,
        2090: 1.858,
        2095: 1.864,
        2100: 1.875,
    },
    "ssp245": {
        2020: 1.230,
        2025: 1.358,
        2030: 1.483,
        2035: 1.594,
        2040: 1.712,
        2045: 1.836,
        2050: 1.960,
        2055: 2.081,
        2060: 2.206,
        2065: 2.333,
        2070: 2.462,
        2075: 2.591,
        2080: 2.724,
        2085: 2.859,
        2090: 2.998,
        2095: 3.140,
        2100: 3.282,
    },
    "ssp370": {
        2020: 1.241,
        2025: 1.372,
        2030: 1.518,
        2035: 1.674,
        2040: 1.845,
        2045: 2.033,
        2050: 2.234,
        2055: 2.446,
        2060: 2.665,
        2065: 2.892,
        2070: 3.124,
        2075: 3.355,
        2080: 3.593,
        2085: 3.834,
        2090: 4.076,
        2095: 4.322,
        2100: 4.568,
    },
    "ssp585": {
        2020: 1.255,
        2025: 1.388,
        2030: 1.552,
        2035: 1.755,
        2040: 1.992,
        2045: 2.253,
        2050: 2.539,
        2055: 2.846,
        2060: 3.169,
        2065: 3.507,
        2070: 3.857,
        2075: 4.204,
        2080: 4.560,
        2085: 4.925,
        2090: 5.293,
        2095: 5.665,
        2100: 6.039,
    },
}


@dataclass
class ClimateAdjustmentConfig:
    enabled: bool
    scenario: str
    baseline_year: float
    temp_rate_c_per_year: float
    humidity_per_temp_c: float
    method: str


def canonical_scenario_name(text: Any) -> str:
    s = str(text).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "none": "none",
        "off": "none",
        "disabled": "none",
        "ssp126": "ssp126",
        "ssp245": "ssp245",
        "ssp370": "ssp370",
        "ssp585": "ssp585",
        "low": "ssp126",
        "mid": "ssp245",
        "medium": "ssp245",
        "high": "ssp585",
    }
    return aliases.get(s, "ssp245")


def canonical_adjustment_method(text: Any) -> str:
    s = str(text).strip().lower().replace("-", "_")
    aliases = {
        "pathway": "pathway",
        "ipcc": "pathway",
        "ipcc_ar6": "pathway",
        "ar6": "pathway",
        "linear": "linear",
        "rate": "linear",
    }
    return aliases.get(s, "pathway")


def _interp_with_linear_extrapolation(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    y = np.interp(x, xp, fp)

    if len(xp) >= 2:
        left_mask = x < xp[0]
        if np.any(left_mask):
            slope_left = (fp[1] - fp[0]) / max(xp[1] - xp[0], 1e-9)
            y[left_mask] = fp[0] + slope_left * (x[left_mask] - xp[0])

        right_mask = x > xp[-1]
        if np.any(right_mask):
            slope_right = (fp[-1] - fp[-2]) / max(xp[-1] - xp[-2], 1e-9)
            y[right_mask] = fp[-1] + slope_right * (x[right_mask] - xp[-1])
    return y


def from_args(args: Any) -> ClimateAdjustmentConfig:
    scenario = canonical_scenario_name(getattr(args, "climate_scenario", "ssp245"))
    baseline_year = float(getattr(args, "climate_baseline_year", float("nan")))
    humidity_per_temp_c = float(getattr(args, "humidity_per_temp_c", -2.0))
    method = canonical_adjustment_method(getattr(args, "climate_adjustment_method", "pathway"))

    # Optional override for temperature trend coefficient.
    temp_rate_override = float(getattr(args, "climate_temp_rate", np.nan))
    if np.isfinite(temp_rate_override):
        temp_rate = float(temp_rate_override)
    else:
        temp_rate = float(SCENARIO_TEMP_RATES_C_PER_YEAR.get(scenario, SCENARIO_TEMP_RATES_C_PER_YEAR["ssp245"]))

    enabled = (scenario != "none") and (not bool(getattr(args, "disable_climate_adjustment", False)))

    return ClimateAdjustmentConfig(
        enabled=enabled,
        scenario=scenario,
        baseline_year=baseline_year,
        temp_rate_c_per_year=temp_rate,
        humidity_per_temp_c=humidity_per_temp_c,
        method=method,
    )


def _temp_delta_from_cfg(year_f: np.ndarray, cfg: ClimateAdjustmentConfig) -> np.ndarray:
    yf = np.asarray(year_f, dtype=float)
    baseline = float(cfg.baseline_year)

    use_pathway = (cfg.method == "pathway") and (cfg.scenario in SCENARIO_TEMP_PATHWAY_C)
    if use_pathway:
        anchors = SCENARIO_TEMP_PATHWAY_C[cfg.scenario]
        xp = np.array(sorted(anchors.keys()), dtype=float)
        fp = np.array([anchors[int(y)] for y in xp], dtype=float)
        anom = _interp_with_linear_extrapolation(yf, xp, fp)
        base_anom = float(_interp_with_linear_extrapolation(np.array([baseline]), xp, fp)[0])
        return anom - base_anom

    return (yf - baseline) * float(cfg.temp_rate_c_per_year)


def _variable_key(variable: Any) -> str:
    s = str(variable).strip().lower()
    if any(k in s for k in ["temp", "sicak", "sıcak", "temperature", "t2m"]):
        return "temp"
    if any(k in s for k in ["humid", "nem", "rh"]):
        return "humidity"
    return s


def _fractional_year(ds: pd.Series | pd.DatetimeIndex | list[Any] | np.ndarray) -> np.ndarray:
    t = pd.to_datetime(ds)
    if isinstance(t, pd.Series):
        day = t.dt.dayofyear.astype(float).to_numpy()
        year = t.dt.year.astype(float).to_numpy()
    else:
        day = np.asarray(t.dayofyear, dtype=float)
        year = np.asarray(t.year, dtype=float)
    return year + (day - 1.0) / 365.2425


def climate_delta_series(
    ds: pd.Series | pd.DatetimeIndex | list[Any] | np.ndarray,
    variable: str,
    cfg: ClimateAdjustmentConfig,
) -> np.ndarray:
    if not cfg.enabled:
        return np.zeros(len(pd.to_datetime(ds)), dtype=float)

    v = _variable_key(variable)
    if v not in {"temp", "humidity"}:
        return np.zeros(len(pd.to_datetime(ds)), dtype=float)

    yf = _fractional_year(ds)
    temp_delta = _temp_delta_from_cfg(yf, cfg)
    if v == "temp":
        return np.asarray(temp_delta, dtype=float)

    # Relative humidity drift as pp/C * warming delta.
    hum_delta = temp_delta * float(cfg.humidity_per_temp_c)
    return np.asarray(hum_delta, dtype=float)


def climate_delta_scalar(ts: Any, variable: str, cfg: ClimateAdjustmentConfig) -> float:
    return float(climate_delta_series([pd.Timestamp(ts)], variable=variable, cfg=cfg)[0])


def with_series_baseline(cfg: ClimateAdjustmentConfig, last_observed_ts: Any) -> ClimateAdjustmentConfig:
    if np.isfinite(float(cfg.baseline_year)):
        return cfg
    base = float(_fractional_year([pd.Timestamp(last_observed_ts)])[0])
    return ClimateAdjustmentConfig(
        enabled=bool(cfg.enabled),
        scenario=str(cfg.scenario),
        baseline_year=base,
        temp_rate_c_per_year=float(cfg.temp_rate_c_per_year),
        humidity_per_temp_c=float(cfg.humidity_per_temp_c),
        method=str(cfg.method),
    )


def climate_strategy_suffix(variable: str, cfg: ClimateAdjustmentConfig) -> str:
    if not cfg.enabled:
        return ""
    v = _variable_key(variable)
    if v not in {"temp", "humidity"}:
        return ""
    return f"_climadj_{cfg.scenario}"
