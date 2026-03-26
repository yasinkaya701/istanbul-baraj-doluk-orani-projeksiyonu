#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/yasinkaya/Hackhaton")
OUT_DIR = ROOT / "output" / "istanbul_backcast_pre2000"

OCCUPANCY_XLSX = ROOT / "external" / "raw" / "ibb" / "İstanbul_Barajları_Günlük_Doluluk_Oranları_af0b3902-cfd9-4096-85f7-e2c3017e4f21.xlsx"
PRECIP_DAILY = ROOT / "output" / "kandilli_et0_project" / "data" / "derived" / "daily" / "kandilli_precip_daily_1911_2024.csv"
ET0_DAILY = ROOT / "output" / "kandilli_et0_project" / "data" / "derived" / "daily" / "kandilli_et0_hargreaves_daily_1911_2021.csv"

CAPACITIES_MCM = {
    "Ömerli": 235.371,
    "Darlık": 107.500,
    "Elmalı": 9.600,
    "Terkos": 162.241,
    "Alibey": 34.143,
    "Büyükçekmece": 148.943,
    "Sazlıdere": 88.730,
    "Kazandere": 17.424,
    "Pabuçdere": 58.500,
    "Istrancalar": 6.231,
}


def month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def load_monthly_occupancy() -> pd.DataFrame:
    df = pd.read_excel(OCCUPANCY_XLSX)
    df["date"] = pd.to_datetime(df["Tarih"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    dam_cols = [c for c in df.columns if c not in {"Tarih", "date"}]
    weights = np.array([CAPACITIES_MCM.get(c, 0.0) for c in dam_cols], dtype=float)
    weights = weights / weights.sum()
    df["weighted_total_fill"] = (df[dam_cols].astype(float).values * weights).sum(axis=1)
    df["month"] = month_start(df["date"])
    monthly = df.groupby("month", as_index=False)["weighted_total_fill"].mean()
    return monthly.rename(columns={"month": "date"})


def load_monthly_climate() -> pd.DataFrame:
    precip = pd.read_csv(PRECIP_DAILY)
    precip["date"] = pd.to_datetime(precip["date"], errors="coerce")
    precip = precip.dropna(subset=["date"]).copy()
    precip["month"] = month_start(precip["date"])
    precip_m = precip.groupby("month", as_index=False)["precip_mm"].sum().rename(columns={"month": "date"})

    et0 = pd.read_csv(ET0_DAILY)
    et0["date"] = pd.to_datetime(et0["date"], errors="coerce")
    et0 = et0.dropna(subset=["date"]).copy()
    et0["month"] = month_start(et0["date"])
    et0_m = et0.groupby("month", as_index=False)["et0_harg_mm_day"].sum().rename(
        columns={"month": "date", "et0_harg_mm_day": "et0_mm"}
    )

    climate = precip_m.merge(et0_m, on="date", how="inner")
    climate["water_balance_mm"] = climate["precip_mm"] - climate["et0_mm"]
    climate["month_sin"] = np.sin(2.0 * np.pi * climate["date"].dt.month / 12.0)
    climate["month_cos"] = np.cos(2.0 * np.pi * climate["date"].dt.month / 12.0)
    return climate


def fit_models(train: pd.DataFrame, features: list[str]) -> dict[str, object]:
    ridge = Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-3, 3, 25)))])
    ridge.fit(train[features], train["weighted_total_fill"])

    et = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    et.fit(train[features], train["weighted_total_fill"])
    return {"ridge": ridge, "extra_trees": et}


def evaluate(model, df: pd.DataFrame, features: list[str]) -> float:
    pred = np.clip(np.asarray(model.predict(df[features]), dtype=float), 0.0, 1.0)
    return float(np.sqrt(mean_squared_error(df["weighted_total_fill"], pred)) * 100.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    occ = load_monthly_occupancy()
    climate = load_monthly_climate()

    merged = occ.merge(climate, on="date", how="inner")
    merged = merged.dropna(subset=["weighted_total_fill"]).copy()

    # Use 2000-2021 overlap for training/validation
    train = merged[(merged["date"] >= "2000-10-01") & (merged["date"] <= "2014-12-01")].copy()
    hold = merged[(merged["date"] >= "2015-01-01") & (merged["date"] <= "2021-12-01")].copy()

    features = ["precip_mm", "et0_mm", "water_balance_mm", "month_sin", "month_cos"]
    models = fit_models(train, features)

    scores = {name: evaluate(model, hold, features) for name, model in models.items()}
    best_name = min(scores, key=scores.get)
    best_model = models[best_name]

    # Backcast for 1912-1999 using climate-only features
    climate_only = climate[(climate["date"] >= "1912-01-01") & (climate["date"] <= "1999-12-01")].copy()
    back_pred = np.clip(np.asarray(best_model.predict(climate_only[features]), dtype=float), 0.0, 1.0)
    backcast = climate_only[["date"]].copy()
    backcast["weighted_total_fill"] = back_pred
    backcast["source"] = "backcast"

    observed = merged[["date", "weighted_total_fill"]].copy()
    observed["source"] = "observed"

    full = pd.concat([backcast, observed], ignore_index=True).sort_values("date").reset_index(drop=True)

    backcast.to_csv(OUT_DIR / "backcast_monthly_1912_1999.csv", index=False)
    observed.to_csv(OUT_DIR / "observed_monthly_2000_2021.csv", index=False)
    full.to_csv(OUT_DIR / "full_series_1912_2021.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12.5, 5.5), dpi=180)
    ax.plot(backcast["date"], backcast["weighted_total_fill"] * 100.0, color="#0ea5e9", label="Backcast (1912-1999)")
    ax.plot(observed["date"], observed["weighted_total_fill"] * 100.0, color="#111827", label="Gözlem (2000-2021)")
    ax.set_ylabel("Toplam doluluk (%)")
    ax.set_title("İstanbul baraj doluluğu: backcast + gözlem")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "backcast_vs_observed.png")
    plt.close(fig)

    summary = {
        "train_start": str(train["date"].min().date()),
        "train_end": str(train["date"].max().date()),
        "holdout_start": str(hold["date"].min().date()),
        "holdout_end": str(hold["date"].max().date()),
        "features": features,
        "model_scores_rmse_pp": scores,
        "best_model": best_name,
    }
    (OUT_DIR / "backcast_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
