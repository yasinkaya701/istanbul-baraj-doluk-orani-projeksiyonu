#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path("/Users/yasinkaya/Hackhaton")


def load_js_payload(path: Path, prefix: str) -> dict:
    raw = path.read_text().strip()
    if not raw.startswith(prefix):
        raise ValueError(f"Unexpected JS format in {path}")
    payload = raw[len(prefix):].strip()
    if payload.endswith(";"):
        payload = payload[:-1]
    return json.loads(payload)


def load_climate_baseline() -> pd.DataFrame:
    data = load_js_payload(ROOT / "assets/data/climate_baseline.js", "window.CLIMATE_BASELINE = ")
    rows = []
    for date_str, vals in data.items():
        rows.append({
            "date": pd.to_datetime(date_str),
            "rain_mm": vals.get("precip_mm_month"),
            "et0_mm_month": vals.get("et0_mm_month"),
        })
    return pd.DataFrame(rows).sort_values("date")


def load_usage_profile() -> list[float]:
    data = load_js_payload(ROOT / "assets/data/usage_monthly_profile.js", "window.USAGE_PROFILE = ")
    profile = data.get("profile") or []
    if len(profile) != 12:
        raise ValueError("usage profile must have 12 monthly weights")
    return [float(v) for v in profile]


def load_usage_trend() -> float:
    data = load_js_payload(ROOT / "assets/data/usage_trend_stats.js", "window.USAGE_TREND = ")
    return float(data.get("yoy_median") or data.get("cagr_2019_2023") or data.get("yoy_mean") or 0.0)


def apply_consumption_trend(df: pd.DataFrame, profile: list[float], growth: float) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month

    counts = out.groupby("year")["consumption_mean_monthly"].apply(lambda s: s.notna().sum())
    full_years = counts[counts == 12]
    if full_years.empty:
        return out
    base_year = int(full_years.index.max())

    base_df = out[out["year"] == base_year].copy()
    base_df["days"] = base_df["date"].dt.days_in_month
    base_annual = float((base_df["consumption_mean_monthly"] * base_df["days"]).sum())

    for year in sorted(out["year"].unique()):
        year_mask = out["year"] == year
        year_df = out[year_mask].copy()
        if year_df.empty:
            continue
        annual_target = base_annual * ((1 + growth) ** max(0, year - base_year))
        year_df["days"] = year_df["date"].dt.days_in_month
        existing = year_df[year_df["consumption_mean_monthly"].notna()].copy()
        existing_total = float((existing["consumption_mean_monthly"] * existing["days"]).sum())
        remaining = max(0.0, annual_target - existing_total)

        missing = year_df[year_df["consumption_mean_monthly"].isna()].copy()
        if missing.empty:
            continue
        weights = [profile[m - 1] for m in missing["month"]]
        weight_sum = sum(weights) if sum(weights) > 0 else 1.0
        for idx, row in missing.iterrows():
            weight = profile[int(row["month"]) - 1] / weight_sum
            monthly_total = remaining * weight
            daily_mean = monthly_total / row["days"] if row["days"] else 0.0
            out.loc[idx, "consumption_mean_monthly"] = daily_mean

    return out


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(1e-6, np.abs(y_true))
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(1e-6, (np.abs(y_true) + np.abs(y_pred)) / 2.0)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0


def corr(a, b):
    if len(a) < 2:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def eval_metrics(y_true, y_pred):
    return {
        "rmse_pp": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_pp": float(mean_absolute_error(y_true, y_pred)),
        "mape_pct": float(mape(y_true, y_pred)),
        "smape_pct": float(smape(y_true, y_pred)),
        "pearson_corr_pct": float(corr(y_true, y_pred) * 100.0),
    }


def load_panel(panel_path: Path) -> pd.DataFrame:
    df = pd.read_csv(panel_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    climate_panel = load_climate_baseline()
    df = df.merge(climate_panel, on="date", how="left", suffixes=("", "_panel"))
    for col in ["rain_mm", "et0_mm_month"]:
        panel_col = f"{col}_panel"
        if panel_col in df:
            df[col] = df[panel_col].combine_first(df[col])
            df = df.drop(columns=[panel_col])

    df = apply_consumption_trend(df, load_usage_profile(), load_usage_trend())
    df["fill_pct"] = df["weighted_total_fill"] * 100.0
    df["lag1_fill_pct"] = df["fill_pct"].shift(1)
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["consumption_lag1"] = df["consumption_mean_monthly"].shift(1)
    df["consumption_ma3"] = df["consumption_mean_monthly"].rolling(3, min_periods=1).mean()
    return df


def build_features(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    cols = [
        "rain_mm",
        "et0_mm_month",
        "consumption_mean_monthly",
        "consumption_lag1",
        "consumption_ma3",
        "t_mean_c",
        "rh_mean_pct",
        "pressure_kpa",
        "vpd_kpa_mean",
        "month_sin",
        "month_cos",
        "lag1_fill_pct",
    ]
    cols = [c for c in cols if c not in drop_cols]
    out = df[["date", "fill_pct"] + cols].copy()
    out = out.dropna(subset=cols + ["fill_pct"]).copy()
    return out


def split_window(df, train_start, train_end, test_start, test_end):
    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    return train, test


def rolling_window_predict(
    df: pd.DataFrame,
    model_template,
    lookback_months: int,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
):
    data = df.sort_values("date").reset_index(drop=True).copy()
    feat_cols = [c for c in data.columns if c not in ["date", "fill_pct"]]
    test = data[(data["date"] >= test_start) & (data["date"] <= test_end)].copy()

    dates = []
    y_true = []
    y_pred = []
    train_windows = []

    for _, row in test.iterrows():
        cur_date = pd.Timestamp(row["date"])
        train_end = (cur_date - pd.DateOffset(months=1)).normalize()
        train_start = (train_end - pd.DateOffset(months=lookback_months - 1)).normalize()
        train = data[(data["date"] >= train_start) & (data["date"] <= train_end)].copy()
        if len(train) < lookback_months:
            continue

        X_train = train[feat_cols].to_numpy(dtype=float)
        y_train = train["fill_pct"].to_numpy(dtype=float)
        X_cur = np.asarray(row[feat_cols], dtype=float).reshape(1, -1)

        model = clone(model_template)
        model.fit(X_train, y_train)
        pred = float(model.predict(X_cur)[0])

        dates.append(cur_date)
        y_true.append(float(row["fill_pct"]))
        y_pred.append(pred)
        train_windows.append((train_start, train_end))

    return (
        np.asarray(dates),
        np.asarray(y_true, dtype=float),
        np.asarray(y_pred, dtype=float),
        train_windows,
    )


def model_catalog():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "gbr": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
        "hgb": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=3, l2_regularization=0.1, random_state=42),
        "rf": RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=4, random_state=42, n_jobs=-1),
        "etr": ExtraTreesRegressor(n_estimators=100, max_depth=5, min_samples_leaf=4, random_state=42, n_jobs=-1),
    }


def usage_map():
    return {
        "ridge": "Doğrusal, stabil temel model; hızlı ve yorumlanabilir.",
        "gbr": "Doğrusal olmayan ilişkileri yakalar; kısa vadede dengeli.",
        "hgb": "Hızlı ve verimli gradient boosting; büyük veri için uygun.",
        "rf": "Dayanıklı ansambllar; uç değerlere karşı stabil.",
        "etr": "Rastgeleleştirilmiş ağaçlar; varyans azaltımı için güçlü.",
    }


def _plot_panel(ax, title, dates, y_true, y_pred, metrics, model_name):
    ax.plot(dates, y_true, color="#111111", linewidth=1.6, label="Gözlenen")
    ax.plot(dates, y_pred, color="#1F77B4", linewidth=1.4, label="Tahmin")
    ax.set_title(title)
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Doluluk (%)")
    ax.legend(loc="upper right")

    text = (
        f"RMSE: {metrics['rmse_pp']:.2f} yp\n"
        f"MAE: {metrics['mae_pp']:.2f} yp\n"
        f"MAPE: {metrics['mape_pct']:.2f}%\n"
        f"Pearson: {metrics['pearson_corr_pct']:.2f}%\n"
        f"Kullanım: {usage_map()[model_name]}"
    )
    ax.text(
        0.99,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="#F7F7F7", edgecolor="#CCCCCC"),
    )


def make_stacked_card(model_name, window_5, window_10, out_path: Path):
    (label5, dates5, y_true5, y_pred5, metrics5) = window_5
    (label10, dates10, y_true10, y_pred10, metrics10) = window_10

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 9), sharex=False)
    _plot_panel(axes[0], f"{model_name} | 5 Yıllık | {label5}", dates5, y_true5, y_pred5, metrics5, model_name)
    _plot_panel(axes[1], f"{model_name} | 10 Yıllık | {label10}", dates10, y_true10, y_pred10, metrics10, model_name)

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="output/newdata_feature_store/tables/istanbul_dam_driver_panel_2000_2026_extended.csv")
    p.add_argument("--drop-cols", default="")
    p.add_argument("--out", default="output/istanbul_model_cards_v4")
    args = p.parse_args()

    out_dir = ROOT / args.out
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    df = load_panel(ROOT / args.panel)
    feat = build_features(df, drop_cols)

    # define windows based on data max
    end_date = feat["date"].max()
    test5_start = (end_date - pd.DateOffset(months=59)).normalize()
    test5_end = end_date
    test10_start = (end_date - pd.DateOffset(months=119)).normalize()
    test10_end = end_date

    windows = {
        "5y": (60, test5_start, test5_end),
        "10y": (120, test10_start, test10_end),
    }

    models = model_catalog()
    rows = []
    per_model = {name: {} for name in models.keys()}

    for wlabel, (lookback_months, test_start, test_end) in windows.items():
        for name, model in models.items():
            dates, y_true, y_pred, train_windows = rolling_window_predict(
                feat,
                model,
                lookback_months=lookback_months,
                test_start=test_start,
                test_end=test_end,
            )
            if len(y_true) == 0:
                continue
            metrics = eval_metrics(y_true, y_pred)
            first_train = train_windows[0] if train_windows else (None, None)
            last_train = train_windows[-1] if train_windows else (None, None)
            rows.append({
                "model": name,
                "train_window": (
                    f"rolling_{lookback_months}m "
                    f"({first_train[0].date()} -> {first_train[1].date()}) .. "
                    f"({last_train[0].date()} -> {last_train[1].date()})"
                ),
                "window": f"{test_start.date()} -> {test_end.date()}",
                **metrics,
            })

            per_model[name][wlabel] = (
                f"Test: {test_start.date()} -> {test_end.date()} | Rolling egitim: {lookback_months} ay",
                dates,
                y_true,
                y_pred,
                metrics,
            )

    pd.DataFrame(rows).to_csv(out_dir / "model_cards_metrics.csv", index=False)

    for name in models.keys():
        if "5y" not in per_model[name] or "10y" not in per_model[name]:
            continue
        out_path = fig_dir / f"{name}_5y_10y.png"
        make_stacked_card(name, per_model[name]["5y"], per_model[name]["10y"], out_path)


if __name__ == "__main__":
    main()
