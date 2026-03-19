#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CAPACITIES_MCM = {
    "Omerli": 235.371,
    "Darlik": 107.500,
    "Elmali": 9.600,
    "Terkos": 162.241,
    "Alibey": 34.143,
    "Buyukcekmece": 148.943,
    "Sazlidere": 88.730,
    "Kazandere": 17.424,
    "Pabucdere": 58.500,
    "Istrancalar": 6.231,
}


@dataclass
class ModelSpec:
    name: str
    features: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deepen Istanbul dam research package.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("output/istanbul_dam_forecast/istanbul_dam_monthly_history.csv"),
    )
    parser.add_argument(
        "--exog",
        type=Path,
        default=Path("output/istanbul_dam_quant_exog/tables/istanbul_dam_model_input_monthly.csv"),
    )
    parser.add_argument(
        "--et0",
        type=Path,
        default=Path("output/tarim_et0_quant/tables/tarim_et0_monthly_history.csv"),
    )
    parser.add_argument(
        "--rain-new",
        type=Path,
        default=Path("output/spreadsheet/rainfall_monthly_newdata.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/istanbul_dam_deep_research"),
    )
    return parser.parse_args()


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def load_base_frame(args: argparse.Namespace) -> pd.DataFrame:
    hist = pd.read_csv(args.history)
    hist["date"] = pd.to_datetime(hist["ds"])
    dam_cols = [c for c in CAPACITIES_MCM if c in hist.columns]
    weights = np.array([CAPACITIES_MCM[c] for c in dam_cols], dtype=float)
    hist["weighted_total_fill"] = (
        hist[dam_cols].mul(weights, axis=1).sum(axis=1) / weights.sum()
    )
    hist["overall_mean"] = hist["overall_mean"].astype(float)
    hist = hist[["date", "weighted_total_fill", "overall_mean"]].copy()

    exog = pd.read_csv(args.exog)
    exog["date"] = pd.to_datetime(exog["ds"])
    exog = exog[
        ["date", "rain_sum_monthly", "rain_mean_monthly", "consumption_mean_monthly"]
    ].copy()

    et0 = pd.read_csv(args.et0)
    et0["date"] = pd.to_datetime(et0["date"])
    et0 = et0[["date", "et0_mm_month"]].copy()

    rain_new = pd.read_csv(args.rain_new)
    rain_new["date"] = pd.to_datetime(rain_new["date"])
    rain_new = rain_new.rename(columns={"rain_mm": "rain_mm_new"})[["date", "rain_mm_new"]]

    df = hist.merge(exog, on="date", how="inner").merge(et0, on="date", how="left")
    df = df.merge(rain_new, on="date", how="left")
    df["rain_model_mm"] = df["rain_mm_new"].fillna(df["rain_sum_monthly"])
    df = df.sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    return df


def add_lags(df: pd.DataFrame, cols: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_lag = ["weighted_total_fill"]
    out = add_lags(df, cols_to_lag, (1, 2))
    out["rain_roll2"] = out["rain_model_mm"].rolling(2).mean()
    out["et0_roll2"] = out["et0_mm_month"].rolling(2).mean()
    out["cons_roll2"] = out["consumption_mean_monthly"].rolling(2).mean()
    out["rain_roll2_anom"] = out["rain_roll2"] - out.groupby("month")["rain_roll2"].transform("mean")
    out["et0_roll2_anom"] = out["et0_roll2"] - out.groupby("month")["et0_roll2"].transform("mean")
    out["cons_roll2_anom"] = out["cons_roll2"] - out.groupby("month")["cons_roll2"].transform("mean")
    out["delta_fill"] = out["weighted_total_fill"] - out["weighted_total_fill_lag1"]
    out = out.dropna().reset_index(drop=True)
    return out


def fit_predict_walkforward(
    df: pd.DataFrame,
    spec: ModelSpec,
    min_train: int = 60,
) -> tuple[pd.DataFrame, Pipeline]:
    preds: list[dict[str, float | str]] = []
    pipeline: Pipeline | None = None
    for i in range(min_train, len(df)):
        train = df.iloc[:i]
        test = df.iloc[[i]]
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
            ]
        )
        pipeline.fit(train[spec.features], train["delta_fill"])
        delta_hat = float(pipeline.predict(test[spec.features])[0])
        yhat = float(test["weighted_total_fill_lag1"].iloc[0] + delta_hat)
        preds.append(
            {
                "date": test["date"].iloc[0],
                "actual": float(test["weighted_total_fill"].iloc[0]),
                "pred": yhat,
                "model": spec.name,
            }
        )
    assert pipeline is not None
    pred_df = pd.DataFrame(preds)
    return pred_df, pipeline


def summarize_metrics(pred_df: pd.DataFrame) -> dict[str, float]:
    y_true = pred_df["actual"].to_numpy(dtype=float)
    y_pred = pred_df["pred"].to_numpy(dtype=float)
    return {
        "rmse_pp": float(np.sqrt(mean_squared_error(y_true, y_pred)) * 100.0),
        "mae_pp": float(mean_absolute_error(y_true, y_pred) * 100.0),
        "smape": smape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_specs() -> list[ModelSpec]:
    pattern = [
        "weighted_total_fill_lag1",
        "weighted_total_fill_lag2",
        "month_sin",
        "month_cos",
    ]
    climate = pattern + [
        "rain_roll2_anom",
        "et0_roll2_anom",
    ]
    demand = pattern + [
        "cons_roll2_anom",
    ]
    full = climate + ["cons_roll2_anom"]
    return [
        ModelSpec("pattern_only", pattern),
        ModelSpec("climate_plus_memory", climate),
        ModelSpec("demand_plus_memory", demand),
        ModelSpec("full_model", full),
    ]


def fit_final_model(df: pd.DataFrame, spec: ModelSpec) -> Pipeline:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )
    pipeline.fit(df[spec.features], df["delta_fill"])
    return pipeline


def build_climatology(df: pd.DataFrame) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]]]:
    raw = {
        "rain_model_mm": df.groupby("month")["rain_model_mm"].mean().to_dict(),
        "et0_mm_month": df.groupby("month")["et0_mm_month"].mean().to_dict(),
        "consumption_mean_monthly": df.groupby("month")["consumption_mean_monthly"].mean().to_dict(),
    }
    roll2 = {
        "rain_model_mm": df.groupby("month")["rain_roll2"].mean().to_dict(),
        "et0_mm_month": df.groupby("month")["et0_roll2"].mean().to_dict(),
        "consumption_mean_monthly": df.groupby("month")["cons_roll2"].mean().to_dict(),
    }
    return raw, roll2


def make_baseline_projection_frame(
    df: pd.DataFrame,
    raw_clim: dict[str, dict[int, float]],
    horizon: int = 12,
) -> pd.DataFrame:
    last_date = df["date"].max()
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    base = pd.DataFrame({"date": future_dates})
    base["month"] = base["date"].dt.month
    for col in ["rain_model_mm", "et0_mm_month", "consumption_mean_monthly"]:
        base[col] = base["month"].map(raw_clim[col])
    base["month_sin"] = np.sin(2 * np.pi * base["month"] / 12.0)
    base["month_cos"] = np.cos(2 * np.pi * base["month"] / 12.0)
    return base


def apply_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = df.copy()
    if scenario == "baseline":
        return out
    if scenario == "rain_plus10_3m":
        out.loc[:2, "rain_model_mm"] *= 1.10
    elif scenario == "et0_plus10_3m":
        out.loc[:2, "et0_mm_month"] *= 1.10
    elif scenario == "cons_plus10_3m":
        out.loc[:2, "consumption_mean_monthly"] *= 1.10
    elif scenario == "restriction_minus7_3m":
        out.loc[:2, "consumption_mean_monthly"] *= 0.93
    elif scenario == "restriction_minus15_3m":
        out.loc[:2, "consumption_mean_monthly"] *= 0.85
    elif scenario == "restriction_minus22_3m":
        out.loc[:2, "consumption_mean_monthly"] *= 0.78
    elif scenario == "restriction_minus15_with_rebound":
        out.loc[:2, "consumption_mean_monthly"] *= 0.85
        out.loc[3:4, "consumption_mean_monthly"] *= 1.05
    elif scenario == "hot_dry_high_demand":
        out.loc[:5, "rain_model_mm"] *= 0.85
        out.loc[:5, "et0_mm_month"] *= 1.10
        out.loc[:5, "consumption_mean_monthly"] *= 1.05
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return out


def simulate_recursive(
    hist_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    pipeline: Pipeline,
    spec: ModelSpec,
    roll2_clim: dict[str, dict[int, float]],
) -> pd.DataFrame:
    cols = [
        "date",
        "month",
        "month_sin",
        "month_cos",
        "rain_model_mm",
        "et0_mm_month",
        "consumption_mean_monthly",
    ]
    hist = hist_df[cols + ["weighted_total_fill"]].copy()
    future = scenario_df[cols].copy()
    sim_rows: list[dict[str, float | str]] = []

    past_y = hist["weighted_total_fill"].tolist()
    past_rain = hist["rain_model_mm"].tolist()
    past_et0 = hist["et0_mm_month"].tolist()
    past_cons = hist["consumption_mean_monthly"].tolist()

    for _, row in future.iterrows():
        feature_row = row.to_dict()
        month = int(feature_row["month"])
        feature_row["weighted_total_fill_lag1"] = past_y[-1]
        feature_row["weighted_total_fill_lag2"] = past_y[-2]
        feature_row["rain_roll2_anom"] = float(
            np.mean([past_rain[-1], feature_row["rain_model_mm"]]) - roll2_clim["rain_model_mm"][month]
        )
        feature_row["et0_roll2_anom"] = float(
            np.mean([past_et0[-1], feature_row["et0_mm_month"]]) - roll2_clim["et0_mm_month"][month]
        )
        feature_row["cons_roll2_anom"] = float(
            np.mean([past_cons[-1], feature_row["consumption_mean_monthly"]])
            - roll2_clim["consumption_mean_monthly"][month]
        )

        x = pd.DataFrame([{k: feature_row[k] for k in spec.features}])
        delta_hat = float(pipeline.predict(x)[0])
        yhat = float(past_y[-1] + delta_hat)
        yhat = float(np.clip(yhat, 0.0, 1.0))

        sim_rows.append({"date": row["date"], "pred": yhat})
        past_y.append(yhat)
        past_rain.append(float(row["rain_model_mm"]))
        past_et0.append(float(row["et0_mm_month"]))
        past_cons.append(float(row["consumption_mean_monthly"]))

    return pd.DataFrame(sim_rows)


def write_weighted_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out = df[["date", "overall_mean", "weighted_total_fill"]].copy()
    out["diff_pp"] = (out["weighted_total_fill"] - out["overall_mean"]) * 100.0
    out.to_csv(out_dir / "weighted_total_vs_mean.csv", index=False)
    return out


def plot_weighted_vs_mean(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    ax.plot(df["date"], df["overall_mean"] * 100.0, label="Esit agirlikli ortalama", color="#7c3aed", linewidth=1.6)
    ax.plot(df["date"], df["weighted_total_fill"] * 100.0, label="Hacim agirlikli toplam doluluk", color="#0f766e", linewidth=1.9)
    ax.set_title("Toplam Doluluk: Esit Ortalama vs Hacim Agirlikli Seri")
    ax.set_ylabel("Doluluk (%)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ablation(metrics_df: pd.DataFrame, out_path: Path) -> None:
    order = ["pattern_only", "climate_plus_memory", "demand_plus_memory", "full_model"]
    colors = ["#9ca3af", "#2563eb", "#059669", "#dc2626"]
    tmp = metrics_df.set_index("model").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    positions = np.arange(len(tmp))
    ax.bar(positions, tmp["rmse_pp"], color=colors)
    ax.set_title("Yuruyen Test: Model Ablasyonu (Hacim Agirlikli Doluluk)")
    ax.set_ylabel("RMSE (yuzde puan)")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        ["Pattern", "Iklim+Hafiza", "Talep+Hafiza", "Tam model"],
        rotation=10,
        ha="right",
    )
    for i, val in enumerate(tmp["rmse_pp"]):
        ax.text(i, val + 0.05, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scenario_paths(scenario_outputs: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.0), dpi=150)
    colors = {
        "baseline": "#111827",
        "restriction_minus15_3m": "#059669",
        "restriction_minus15_with_rebound": "#0ea5e9",
        "hot_dry_high_demand": "#dc2626",
        "rain_plus10_3m": "#2563eb",
    }
    labels = {
        "baseline": "Baz",
        "restriction_minus15_3m": "3 ay -%15 talep",
        "restriction_minus15_with_rebound": "3 ay -%15, sonra geri tepme",
        "hot_dry_high_demand": "Sicak-kurak-yuksek talep",
        "rain_plus10_3m": "3 ay +%10 yagis",
    }
    for key in ["baseline", "rain_plus10_3m", "restriction_minus15_3m", "restriction_minus15_with_rebound", "hot_dry_high_demand"]:
        df = scenario_outputs[key]
        ax.plot(df["date"], df["pred"] * 100.0, label=labels[key], color=colors[key], linewidth=1.8)
    ax.set_title("12 Aylik Senaryo Yollari (Hacim Agirlikli Toplam Doluluk)")
    ax.set_ylabel("Doluluk (%)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_group_importance(coeff_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=150)
    ax.bar(coeff_df["group"], coeff_df["abs_coef_sum"], color=["#6b7280", "#2563eb", "#dc2626", "#059669"])
    ax.set_title("Tam Modelde Grup Bazli Etki Buyuklugu")
    ax.set_ylabel("|standartlastirilmis katsayi| toplami")
    for i, val in enumerate(coeff_df["abs_coef_sum"]):
        ax.text(i, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_group_importance(pipeline: Pipeline, spec: ModelSpec) -> pd.DataFrame:
    coef = pipeline.named_steps["model"].coef_
    rows = pd.DataFrame({"feature": spec.features, "coef": coef})
    rows["group"] = "Diger"
    rows.loc[rows["feature"].str.contains("weighted_total_fill_lag"), "group"] = "Hafiza"
    rows.loc[rows["feature"].str.contains("month_"), "group"] = "Mevsimsellik"
    rows.loc[rows["feature"].str.contains("rain_"), "group"] = "Yagis-Kuraklik"
    rows.loc[rows["feature"].str.contains("et0_"), "group"] = "ET0"
    rows.loc[rows["feature"].str.contains("cons_"), "group"] = "Tuketim"
    out = rows.groupby("group", as_index=False)["coef"].agg(lambda s: np.abs(s).sum())
    out = out.rename(columns={"coef": "abs_coef_sum"})
    order = ["Hafiza", "Mevsimsellik", "Yagis-Kuraklik", "ET0", "Tuketim"]
    out["sort"] = out["group"].apply(lambda x: order.index(x) if x in order else 99)
    out = out.sort_values("sort").drop(columns="sort")
    return out


def build_scenario_summary(scenario_outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = scenario_outputs["baseline"].copy().rename(columns={"pred": "baseline"})
    rows = []
    horizons = [1, 3, 6, 12]
    for name, df in scenario_outputs.items():
        if name == "baseline":
            continue
        merged = base.merge(df.rename(columns={"pred": name}), on="date", how="inner")
        for h in horizons:
            delta = float((merged[name].iloc[h - 1] - merged["baseline"].iloc[h - 1]) * 100.0)
            rows.append({"scenario": name, "horizon_month": h, "delta_pp": delta})
    return pd.DataFrame(rows)


def save_sources(out_dir: Path) -> None:
    sources = [
        {
            "topic": "FAO-56 ve Penman-Monteith",
            "url": "https://www.fao.org/4/X0490E/x0490e06.htm",
        },
        {
            "topic": "HEC-HMS Penman-Monteith teknik referansi",
            "url": "https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/evaporation-and-transpiration/penman-monteith-method",
        },
        {
            "topic": "Reservoir evaporation ve su bulunurlugu senaryo calismasi",
            "url": "https://hess.copernicus.org/articles/28/3243/2024/index.html",
        },
        {
            "topic": "Open-water evaporation ve su yuzeyi etkileri",
            "url": "https://hess.copernicus.org/articles/30/67/2026/",
        },
        {
            "topic": "USGS hidroljik butce ve evaporasyonun buyuk payi",
            "url": "https://doi.org/10.3133/wsp2439",
        },
        {
            "topic": "AI reservoir volume forecasting (Journal of Hydrology 2023)",
            "url": "https://doi.org/10.1016/j.jhydrol.2022.128766",
        },
        {
            "topic": "Conditioned LSTM reservoir releases (Journal of Hydrology 2025)",
            "url": "https://doi.org/10.1016/j.jhydrol.2025.133750",
        },
        {
            "topic": "Demand restriction review, ortalama etki",
            "url": "https://doi.org/10.1111/j.1936-704X.2024.3402.x",
        },
        {
            "topic": "Mandatory vs voluntary restriction araligi",
            "url": "https://doi.org/10.22004/ag.econ.19327",
        },
        {
            "topic": "Los Angeles su kisiti etkisi",
            "url": "https://doi.org/10.1016/j.resconrec.2014.10.005",
        },
        {
            "topic": "ISKI su kaynaklari ve kapasite tablosu",
            "url": "https://iski.istanbul/kurumsal/hakkimizda/su-kaynaklari",
        },
        {
            "topic": "ISKI su yonetimi ve yakalar arasi dagitim",
            "url": "https://iski.istanbul/kurumsal/hakkimizda",
        },
        {
            "topic": "ISKI Omerli aritma tesisi kaynaklari",
            "url": "https://iski.istanbul/kurumsal/iski-tesisleri/icme-suyu-aritma-tesisleri/oemerli-icme-suyu-aritma-tesisleri/",
        },
        {
            "topic": "ISKI Ikitelli aritma tesisi kaynaklari",
            "url": "https://iski.istanbul/kurumsal/iski-tesisleri/icme-suyu-aritma-tesisleri/ikitelli-icme-suyu-aritma-tesisleri/",
        },
        {
            "topic": "ISKI Cumhuriyet tesisi ve Melen aktarimi",
            "url": "https://iski.istanbul/kurumsal/iski-tesisleri/icme-suyu-aritma-tesisleri/cumhuriyet-icme-suyu-aritma-tesisi",
        },
    ]
    (out_dir / "sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_base_frame(args)
    model_df = build_model_frame(base_df)
    specs = build_specs()

    metrics_rows = []
    pred_frames = []
    pipelines = {}
    for spec in specs:
        pred_df, _ = fit_predict_walkforward(model_df, spec)
        metrics = summarize_metrics(pred_df)
        metrics_rows.append({"model": spec.name, **metrics})
        pred_frames.append(pred_df)
        pipelines[spec.name] = fit_final_model(model_df, spec)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse_pp")
    metrics_df.to_csv(out_dir / "model_ablation_metrics.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / "walkforward_predictions.csv", index=False)

    weighted_cmp = write_weighted_summary(base_df, out_dir)
    plot_weighted_vs_mean(weighted_cmp, figures_dir / "weighted_total_vs_mean.png")
    plot_ablation(metrics_df, figures_dir / "ablation_rmse_weighted.png")

    full_spec = next(spec for spec in specs if spec.name == "full_model")
    full_model = pipelines["full_model"]
    coeff_df = build_group_importance(full_model, full_spec)
    coeff_df.to_csv(out_dir / "group_importance.csv", index=False)
    plot_group_importance(coeff_df, figures_dir / "group_importance.png")

    raw_clim, roll2_clim = build_climatology(model_df)
    baseline_future = make_baseline_projection_frame(model_df, raw_clim, horizon=12)
    scenario_names = [
        "baseline",
        "rain_plus10_3m",
        "et0_plus10_3m",
        "cons_plus10_3m",
        "restriction_minus7_3m",
        "restriction_minus15_3m",
        "restriction_minus22_3m",
        "restriction_minus15_with_rebound",
        "hot_dry_high_demand",
    ]
    scenario_outputs = {}
    for scenario in scenario_names:
        scenario_future = apply_scenario(baseline_future, scenario)
        scenario_outputs[scenario] = simulate_recursive(
            model_df,
            scenario_future,
            full_model,
            full_spec,
            roll2_clim=roll2_clim,
        )
    plot_scenario_paths(scenario_outputs, figures_dir / "scenario_paths_weighted.png")

    scenario_summary = build_scenario_summary(scenario_outputs)
    scenario_summary.to_csv(out_dir / "scenario_summary.csv", index=False)

    model_df.to_csv(out_dir / "model_frame.csv", index=False)
    save_sources(out_dir)

    summary = {
        "rain_new_available_until": str(base_df.loc[base_df["rain_mm_new"].notna(), "date"].max().date()),
        "history_start": str(base_df["date"].min().date()),
        "history_end": str(base_df["date"].max().date()),
        "weighted_vs_mean_avg_diff_pp": float(weighted_cmp["diff_pp"].mean()),
        "best_model": metrics_df.iloc[0]["model"],
        "best_rmse_pp": float(metrics_df.iloc[0]["rmse_pp"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
