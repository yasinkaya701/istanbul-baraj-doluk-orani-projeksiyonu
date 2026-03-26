#!/usr/bin/env python3
"""Unit tests for build_irrigation_crop_ml_bundle.py."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_irrigation_crop_ml_bundle import (
    add_common_features,
    build_irrigation_outputs,
    default_crop_calendar,
    kc_stage_and_value,
    run_ml_benchmark,
    split_idx,
)


class TestKcEngine(unittest.TestCase):
    def test_kc_stage_interpolation(self) -> None:
        row = pd.Series(
            {
                "season_length_days": 100,
                "l_ini": 20,
                "l_dev": 20,
                "l_mid": 40,
                "l_late": 20,
                "kc_ini": 0.4,
                "kc_mid": 1.2,
                "kc_end": 0.6,
            }
        )
        s1, k1 = kc_stage_and_value(10, row)
        s2, k2 = kc_stage_and_value(30, row)
        s3, k3 = kc_stage_and_value(50, row)
        s4, k4 = kc_stage_and_value(90, row)
        s5, k5 = kc_stage_and_value(120, row)

        self.assertEqual(s1, "initial")
        self.assertAlmostEqual(k1, 0.4, places=6)
        self.assertEqual(s2, "development")
        self.assertGreater(k2, 0.4)
        self.assertLessEqual(k2, 1.2)
        self.assertEqual(s3, "mid")
        self.assertAlmostEqual(k3, 1.2, places=6)
        self.assertEqual(s4, "late")
        self.assertLess(k4, 1.2)
        self.assertEqual(s5, "off_season")
        self.assertEqual(k5, 0.0)


class TestIrrigationMath(unittest.TestCase):
    def _mock_et0(self) -> pd.DataFrame:
        dates = pd.date_range("1987-01-01", periods=365, freq="D")
        et0 = pd.DataFrame(
            {
                "date": dates,
                "et0_completed_mm_day": 2.5 + 0.2 * np.sin(2 * np.pi * np.arange(365) / 365.0),
                "t_mean_c": 15.0,
                "t_min_c": 10.0,
                "t_max_c": 20.0,
                "rh_mean_pct": 60.0,
                "u2_m_s": 2.5,
                "rs_mj_m2_day": 12.0,
                "p_kpa": 100.0,
                "t_mean_nasa_c": 15.0,
                "rh_nasa_pct": 60.0,
                "source_temp": "local",
                "source_humidity": "local",
                "source_wind": "nasa",
                "source_radiation": "nasa",
            }
        )
        return add_common_features(et0, elevation_m=39.0, latitude=41.01, krs=0.19)

    def test_mm_to_m3_conversion(self) -> None:
        et0 = self._mock_et0()
        precip = pd.DataFrame(
            {
                "date": et0["date"],
                "month": et0["date"].dt.to_period("M").astype(str),
                "peff_proxy_mm_day": 0.2,
                "coverage_flag": "good",
            }
        )
        crop = default_crop_calendar(1987).head(1)
        daily, weekly, comparison = build_irrigation_outputs(
            et0=et0,
            precip=precip,
            crop_cal=crop,
            shifts=[0],
            area_ha=1.0,
            efficiency=0.75,
        )
        self.assertFalse(daily.empty)
        self.assertFalse(weekly.empty)
        self.assertFalse(comparison.empty)
        check = np.allclose(
            daily["dose_m3_day"].to_numpy(),
            (daily["gross_mm_day"] * 10.0).to_numpy(),
            atol=1e-9,
        )
        self.assertTrue(check)


class TestMlAndSplit(unittest.TestCase):
    def _mock_ml(self) -> pd.DataFrame:
        n = 365
        dates = pd.date_range("1987-01-01", periods=n, freq="D")
        x = np.arange(n, dtype=float)
        et0 = 2.0 + 0.005 * x + 0.5 * np.sin(2 * np.pi * x / 365.0)
        df = pd.DataFrame(
            {
                "date": dates,
                "et0_completed_mm_day": et0,
                "t_mean_c": 10 + 0.03 * x,
                "t_min_c": 5 + 0.02 * x,
                "t_max_c": 15 + 0.04 * x,
                "rh_mean_pct": 70 - 0.02 * x,
                "u2_m_s": 2.0 + 0.001 * x,
                "rs_mj_m2_day": 8 + 0.02 * x,
                "p_kpa": 100.0,
                "t_mean_nasa_c": 10 + 0.03 * x + 0.2,
                "rh_nasa_pct": 70 - 0.02 * x + 1.0,
                "source_temp": "local",
                "source_humidity": "local",
                "source_wind": "nasa",
                "source_radiation": "nasa",
            }
        )
        return add_common_features(df, elevation_m=39.0, latitude=41.01, krs=0.19)

    def test_split_idx(self) -> None:
        tr, va, te = split_idx(365)
        self.assertEqual(len(tr), 255)
        self.assertEqual(len(va), 54)
        self.assertEqual(len(te), 56)

    def test_ml_benchmark_runs(self) -> None:
        df = self._mock_ml()
        leaderboard, preds = run_ml_benchmark(df, model_keys=["linear"], elevation_m=39.0)
        self.assertFalse(leaderboard.empty)
        self.assertFalse(preds.empty)
        self.assertIn("test_rmse", leaderboard.columns)
        self.assertIn("y_pred_et0_mm_day", preds.columns)


if __name__ == "__main__":
    unittest.main()
