#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path("/Users/yasinkaya/Hackhaton")


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
    df["fill_pct"] = df["weighted_total_fill"] * 100.0
    df["lag1_fill_pct"] = df["fill_pct"].shift(1)
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    return df


def build_features(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    cols = [
        "rain_mm",
        "et0_mm_month",
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


def split_window(df, train_end, test_start, test_end):
    train = df[df["date"] <= train_end].copy()
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    return train, test


def model_catalog():
    return {
        "ridge": Ridge(alpha=1.0),
        "gbr": GradientBoostingRegressor(random_state=42),
        "hgb": HistGradientBoostingRegressor(random_state=42),
        "rf": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "etr": ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1),
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
    p.add_argument("--out", default="output/istanbul_model_cards_2026_03_18")
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
    train5_end = (test5_start - pd.DateOffset(months=1)).normalize()

    test10_start = (end_date - pd.DateOffset(months=119)).normalize()
    test10_end = end_date
    train10_end = (test10_start - pd.DateOffset(months=1)).normalize()

    windows = {
        "5y": (train5_end, test5_start, test5_end),
        "10y": (train10_end, test10_start, test10_end),
    }

    models = model_catalog()
    rows = []
    per_model = {name: {} for name in models.keys()}

    for wlabel, (train_end, test_start, test_end) in windows.items():
        train, test = split_window(feat, train_end, test_start, test_end)
        X_train = train.drop(columns=["date", "fill_pct"]).values
        y_train = train["fill_pct"].values
        X_test = test.drop(columns=["date", "fill_pct"]).values
        y_test = test["fill_pct"].values

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = eval_metrics(y_test, y_pred)
            rows.append({"model": name, "window": f"{test_start.date()} -> {test_end.date()}", **metrics})

            per_model[name][wlabel] = (
                f"{test_start.date()} -> {test_end.date()}",
                test["date"].values,
                y_test,
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
