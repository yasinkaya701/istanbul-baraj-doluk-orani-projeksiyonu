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
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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

    # find last full year with complete consumption data
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


def vpd_kpa_from_t_rh(t_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    es = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))
    ea = es * (rh_pct / 100.0)
    return es - ea


def model_catalog():
    from sklearn.ensemble import VotingRegressor
    linear = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return {
        "ridge": linear,
        "gbr": VotingRegressor(estimators=[("lin", linear), ("tree", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42))], weights=[0.4, 0.6]),
        "hgb": VotingRegressor(estimators=[("lin", linear), ("tree", HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=3, l2_regularization=0.1, random_state=42))], weights=[0.4, 0.6]),
        "rf": VotingRegressor(estimators=[("lin", linear), ("tree", RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=4, random_state=42, n_jobs=-1))], weights=[0.4, 0.6]),
        "etr": VotingRegressor(estimators=[("lin", linear), ("tree", ExtraTreesRegressor(n_estimators=100, max_depth=5, min_samples_leaf=4, random_state=42, n_jobs=-1))], weights=[0.4, 0.6]),
    }


def usage_map():
    return {
        "ridge": "Doğrusal, stabil temel model; hızlı ve yorumlanabilir.",
        "gbr": "Doğrusal olmayan ilişkileri yakalar; kısa vadede dengeli.",
        "hgb": "Hızlı ve verimli gradient boosting; büyük veri için uygun.",
        "rf": "Dayanıklı ansambllar; uç değerlere karşı stabil.",
        "etr": "Rastgeleleştirilmiş ağaçlar; varyans azaltımı için güçlü.",
    }


def build_driver_2000_2040(panel_path: Path, climate_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])

    climate = pd.read_csv(climate_path)
    climate["date"] = pd.to_datetime(climate["date"])
    climate_panel = load_climate_baseline()
    usage_profile = load_usage_profile()
    usage_growth = load_usage_trend()

    # Use climate projection for 2027-2040
    climate_future = climate[climate["date"].dt.year >= 2027].copy()
    climate_future = climate_future.rename(columns={
        "precip_mm_month": "rain_mm",
    })

    # monthly climatology from panel (2000-2026)
    panel["month"] = panel["date"].dt.month
    pressure_clim = panel.groupby("month")["pressure_kpa"].mean()
    rain_clim = panel.groupby("month")["rain_mm"].mean()
    et0_clim = panel.groupby("month")["et0_mm_month"].mean()
    tmean_clim = panel.groupby("month")["t_mean_c"].mean()
    rh_clim = panel.groupby("month")["rh_mean_pct"].mean()

    # base driver: all months 2000-2040
    full_idx = pd.date_range("2000-01-01", "2040-12-01", freq="MS")
    df = pd.DataFrame({"date": full_idx})
    df["month"] = df["date"].dt.month

    df = df.merge(panel[["date", "rain_mm", "et0_mm_month", "t_mean_c", "rh_mean_pct", "pressure_kpa", "vpd_kpa_mean", "weighted_total_fill", "consumption_mean_monthly"]],
                  on="date", how="left")

    df = df.merge(climate_future[["date", "rain_mm", "et0_mm_month", "t_mean_c", "rh_mean_pct"]],
                  on="date", how="left", suffixes=("", "_clim"))

    # fill future from climate projection
    for col in ["rain_mm", "et0_mm_month", "t_mean_c", "rh_mean_pct"]:
        df[col] = df[col].where(~df[f"{col}_clim"].notna(), df[f"{col}_clim"])
        if f"{col}_clim" in df:
            df = df.drop(columns=[f"{col}_clim"])

    # override rain/et0 with climate baseline (2010-2040 where available)
    df = df.merge(climate_panel, on="date", how="left", suffixes=("", "_panel"))
    for col in ["rain_mm", "et0_mm_month"]:
        panel_col = f"{col}_panel"
        if panel_col in df:
            df[col] = df[panel_col].combine_first(df[col])
            df = df.drop(columns=[panel_col])

    # fill by climatology
    df["pressure_kpa"] = df["pressure_kpa"].fillna(df["month"].map(pressure_clim))
    df["rain_mm"] = df["rain_mm"].fillna(df["month"].map(rain_clim))
    df["et0_mm_month"] = df["et0_mm_month"].fillna(df["month"].map(et0_clim))
    df["t_mean_c"] = df["t_mean_c"].fillna(df["month"].map(tmean_clim))
    df["rh_mean_pct"] = df["rh_mean_pct"].fillna(df["month"].map(rh_clim))

    # fallback global means if any remain
    for col in ["pressure_kpa", "rain_mm", "et0_mm_month", "t_mean_c", "rh_mean_pct"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    # compute vpd if missing
    df["vpd_kpa_mean"] = df["vpd_kpa_mean"].fillna(vpd_kpa_from_t_rh(df["t_mean_c"], df["rh_mean_pct"]))

    # features
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["climate_balance_mm"] = df["rain_mm"] - df["et0_mm_month"]
    # değişimler (delta)
    df["rain_delta"] = df["rain_mm"].diff(1)
    df["et0_delta"] = df["et0_mm_month"].diff(1)
    df["balance_delta"] = df["climate_balance_mm"].diff(1)
    df["tmean_delta"] = df["t_mean_c"].diff(1)
    df["rh_delta"] = df["rh_mean_pct"].diff(1)
    # gecikmeler (lag)
    df["rain_lag1"] = df["rain_mm"].shift(1)
    df["et0_lag1"] = df["et0_mm_month"].shift(1)
    df["balance_lag1"] = df["climate_balance_mm"].shift(1)
    df["tmean_lag1"] = df["t_mean_c"].shift(1)
    df["rh_lag1"] = df["rh_mean_pct"].shift(1)
    # 3 ve 6 aylık hareketli ortalama
    df["rain_ma3"] = df["rain_mm"].rolling(3, min_periods=1).mean()
    df["et0_ma3"] = df["et0_mm_month"].rolling(3, min_periods=1).mean()
    df["balance_ma3"] = df["climate_balance_mm"].rolling(3, min_periods=1).mean()
    df["rain_ma6"] = df["rain_mm"].rolling(6, min_periods=1).mean()
    df["et0_ma6"] = df["et0_mm_month"].rolling(6, min_periods=1).mean()
    df["balance_ma6"] = df["climate_balance_mm"].rolling(6, min_periods=1).mean()

    # consumption trend + features
    df = apply_consumption_trend(df, usage_profile, usage_growth)
    df["consumption_lag1"] = df["consumption_mean_monthly"].shift(1)
    df["consumption_ma3"] = df["consumption_mean_monthly"].rolling(3, min_periods=1).mean()

    # target
    df["fill_pct"] = df["weighted_total_fill"] * 100.0
    df["lag1_fill_pct"] = df["fill_pct"].shift(1)

    return df


def simulate_projection(
    df: pd.DataFrame,
    model,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    use_one_year_for_2027: bool = True,
    use_stochastic_future: bool = False,
    seed: int = 42,
) -> pd.Series:
    # copy to avoid mutation
    data = df.copy()

    # initial lag1 from observed
    data = data.sort_values("date").reset_index(drop=True)

    # prepare output series
    pred = pd.Series(index=data.index, dtype=float)

    # find start/end indices
    start_idx = data.index[data["date"] == start_date][0]
    end_idx = data.index[data["date"] == end_date][0]

    feat_cols = [
        "rain_mm",
        "et0_mm_month",
        "consumption_mean_monthly",
        "consumption_lag1",
        "consumption_ma3",
        "t_mean_c",
        "rh_mean_pct",
        "pressure_kpa",
        "vpd_kpa_mean",
        "climate_balance_mm",
        "rain_delta",
        "et0_delta",
        "balance_delta",
        "tmean_delta",
        "rh_delta",
        "rain_lag1",
        "et0_lag1",
        "balance_lag1",
        "tmean_lag1",
        "rh_lag1",
        "rain_ma3",
        "et0_ma3",
        "balance_ma3",
        "rain_ma6",
        "et0_ma6",
        "balance_ma6",
        "month_sin",
        "month_cos",
        "lag1_fill_pct",
    ]

    rng = np.random.default_rng(seed)
    # Smooth handover from observed->projected and damp year-turn jumps.
    hist = data[data["fill_pct"].notna()]["fill_pct"]
    hist_abs_delta = hist.diff().abs().dropna()
    last_obs_date = data.loc[data["fill_pct"].notna(), "date"].max()
    if len(hist_abs_delta):
        monthly_delta_cap = float(np.clip(np.percentile(hist_abs_delta, 75) * 1.15, 6.0, 12.0))
    else:
        monthly_delta_cap = 10.0
    first_year_turn = pd.Timestamp(f"{start_date.year + 1}-01-01")

    # iterate year-by-year (model refit each year)
    for year in range(start_date.year, end_date.year + 1):
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-01")

        # determine training window for the year
        if year == start_date.year and pd.notna(last_obs_date):
            desired_train_end = pd.Timestamp(last_obs_date)
        elif year == 2027:
            desired_train_end = pd.Timestamp("2026-12-01")
        elif year >= 2028:
            desired_train_end = pd.Timestamp(f"{year-1}-12-01")
        else:
            desired_train_end = year_start - pd.DateOffset(months=1)

        # Never train on non-observed horizon months; keep yearly refits continuous.
        if pd.notna(last_obs_date):
            train_end = min(pd.Timestamp(last_obs_date), desired_train_end)
        else:
            train_end = desired_train_end

        train = data[data["date"] <= train_end].copy()
        train = train.dropna(subset=feat_cols + ["fill_pct"])
        if train.empty:
            continue

        X_train = train[feat_cols].values
        y_train = train["fill_pct"].values
        model.fit(X_train, y_train)

        # predict months in this year within range
        year_mask = (data["date"] >= max(year_start, start_date)) & (data["date"] <= min(year_end, end_date))
        idxs = data.index[year_mask].tolist()
        for i in idxs:
            # update lag1
            if i == 0:
                lag1 = np.nan
            else:
                if not np.isnan(data.loc[i - 1, "fill_pct"]):
                    lag1 = data.loc[i - 1, "fill_pct"]
                elif not np.isnan(pred.loc[i - 1]):
                    lag1 = pred.loc[i - 1]
                else:
                    lag1 = np.nan

            if np.isnan(lag1):
                last_obs = data["fill_pct"].dropna()
                lag1 = float(last_obs.iloc[-1]) if not last_obs.empty else np.nan
            data.loc[i, "lag1_fill_pct"] = lag1

            X_pred = data.loc[i, feat_cols].values.reshape(1, -1)
            yhat = float(model.predict(X_pred)[0])
            if use_stochastic_future:
                # optional stochastic path (disabled for scientific mean path)
                yhat = yhat + rng.normal(0.0, 1.0)
                yhat = float(np.clip(yhat, 0.0, 100.0))

            # Transition damping: first projected month and January are softened.
            if np.isfinite(lag1):
                cur_date = data.loc[i, "date"]
                if i == start_idx:
                    alpha = 0.45
                    delta_cap = monthly_delta_cap
                elif cur_date == first_year_turn:
                    # Guard against visible break at the first Dec→Jan turn.
                    alpha = 0.45
                    delta_cap = min(monthly_delta_cap, 5.0)
                elif int(cur_date.month) == 1:
                    alpha = 0.60
                    delta_cap = monthly_delta_cap
                else:
                    alpha = 1.0
                    delta_cap = monthly_delta_cap
                yhat = lag1 + alpha * (yhat - lag1)
                yhat = lag1 + float(np.clip(yhat - lag1, -delta_cap, delta_cap))
                yhat = float(np.clip(yhat, 0.0, 100.0))
            pred.loc[i] = yhat
            data.loc[i, "fill_pct"] = yhat

    # merge predictions into output
    out = data[["date", "fill_pct"]].copy()
    out["fill_sim"] = out["fill_pct"]
    out.loc[start_idx:end_idx, "fill_sim"] = pred.loc[start_idx:end_idx].values
    return out.set_index("date")["fill_sim"]


def add_info_box(fig, model_name, metrics5, metrics10):
    text = (
        f"Model: {model_name} | {usage_map()[model_name]}\n"
        f"5y RMSE: {metrics5['rmse_pp']:.2f} | MAPE: {metrics5['mape_pct']:.2f}% | Pearson: {metrics5['pearson_corr_pct']:.2f}%\n"
        f"10y RMSE: {metrics10['rmse_pp']:.2f} | MAPE: {metrics10['mape_pct']:.2f}% | Pearson: {metrics10['pearson_corr_pct']:.2f}%\n"
        f"Not: 2027=2000-2026 eğitim, 2028+=2000-(önceki yıl) eğitim. Band: ±max(RMSE)."
    )
    fig.text(
        0.99, 0.01, text,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="#F7F7F7", edgecolor="#CCCCCC"),
    )


def compute_monthly_residual_quantiles(df: pd.DataFrame, model, feat_cols: list[str]) -> pd.DataFrame:
    hist = df[df["fill_pct"].notna()].copy()
    hist = hist.dropna(subset=feat_cols + ["fill_pct"])
    if hist.empty:
        return pd.DataFrame({"month": range(1, 13), "q10": 0.0, "q90": 0.0})
    X = hist[feat_cols].values
    y = hist["fill_pct"].values
    model.fit(X, y)
    yhat = model.predict(X)
    resid = y - yhat
    out = pd.DataFrame({"month": hist["date"].dt.month.values, "resid": resid})
    q = out.groupby("month")["resid"].quantile([0.1, 0.9]).unstack()
    q.columns = ["q10", "q90"]
    q = q.reindex(range(1, 13)).fillna(0.0).reset_index()
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="output/newdata_feature_store/tables/istanbul_dam_driver_panel_2000_2026_extended.csv")
    p.add_argument("--climate", default="output/scientific_climate_projection_2026_2040/climate_projection_2010_2040_monthly.csv")
    p.add_argument("--metrics", default="output/istanbul_model_cards_v4/model_cards_metrics.csv")
    p.add_argument("--out", default="output/istanbul_projection_2040_rolling")
    p.add_argument("--models", default="ridge,gbr,hgb,rf,etr")
    args = p.parse_args()

    out_dir = ROOT / args.out
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = build_driver_2000_2040(ROOT / args.panel, ROOT / args.climate)

    # determine simulation start: first month after last observed fill
    last_obs = df[df["fill_pct"].notna()]["date"].max()
    if pd.isna(last_obs):
        last_obs = pd.Timestamp("2024-02-01")
    start_date = (pd.Timestamp(last_obs) + pd.DateOffset(months=1)).normalize()
    end_date = pd.Timestamp("2040-12-01")

    metrics = pd.read_csv(ROOT / args.metrics)

    models_all = model_catalog()
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    models = {k: v for k, v in models_all.items() if k in requested}

    all_rows = []

    for name, model in models.items():
        is_stochastic = (name != "ridge")
        sim_series = simulate_projection(df, model, start_date, end_date, use_one_year_for_2027=True, use_stochastic_future=is_stochastic)

        out = df[["date", "fill_pct"]].copy()
        out["fill_sim"] = sim_series.values
        out["model"] = name
        out.to_csv(out_dir / f"projection_{name}.csv", index=False)

        # metrics for info box
        m5 = metrics[(metrics["model"] == name) & (metrics["window"].str.contains("2019-03-01"))].iloc[0]
        m10 = metrics[(metrics["model"] == name) & (metrics["window"].str.contains("2014-03-01"))].iloc[0]
        m5 = {"rmse_pp": m5["rmse_pp"], "mape_pct": m5["mape_pct"], "pearson_corr_pct": m5["pearson_corr_pct"]}
        m10 = {"rmse_pp": m10["rmse_pp"], "mape_pct": m10["mape_pct"], "pearson_corr_pct": m10["pearson_corr_pct"]}

        # uncertainty band using RMSE (scientific, conservative)
        rmse_band = max(m5["rmse_pp"], m10["rmse_pp"])

        # plot full
        plt.figure(figsize=(12, 6))
        # geçmiş (gözlenen)
        hist_mask = out["date"] <= last_obs
        fut_mask = out["date"] > last_obs
        plt.plot(out["date"][hist_mask], out["fill_pct"][hist_mask], color="#111111", linewidth=1.3, label="Gözlenen (Geçmiş)")
        # gelecek (simülasyon)
        plt.plot(out["date"][fut_mask], out["fill_sim"][fut_mask], color="#D55E00", linewidth=1.4, label="Simülasyon (Gelecek)")
        # belirsizlik bandı
        lower = np.clip(out["fill_sim"][fut_mask].values - rmse_band, 0, 100)
        upper = np.clip(out["fill_sim"][fut_mask].values + rmse_band, 0, 100)
        plt.fill_between(out["date"][fut_mask], lower, upper, color="#D55E00", alpha=0.15, linewidth=0)
        # geçiş çizgisi (son gözlenen)
        plt.axvline(last_obs, color="#999999", linewidth=0.8, linestyle="--", alpha=0.8)
        plt.title(f"{name} | 2000-2040 Simülasyon")
        plt.xlabel("Tarih")
        plt.ylabel("Doluluk (%)")
        plt.legend(loc="upper right")
        add_info_box(plt.gcf(), name, m5, m10)
        plt.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
        plt.savefig(fig_dir / f"{name}_projection_2000_2040.png", dpi=160)
        plt.close()

        # plot zoom future
        plt.figure(figsize=(12, 6))
        future_mask = out["date"] >= start_date
        plt.plot(out["date"][future_mask], out["fill_sim"][future_mask], color="#D55E00", linewidth=1.4, label="Simülasyon (Gelecek)")
        lower2 = np.clip(out["fill_sim"][future_mask].values - rmse_band, 0, 100)
        upper2 = np.clip(out["fill_sim"][future_mask].values + rmse_band, 0, 100)
        plt.fill_between(out["date"][future_mask], lower2, upper2, color="#D55E00", alpha=0.15, linewidth=0)
        plt.title(f"{name} | {start_date.date()}-2040 Simülasyon")
        plt.xlabel("Tarih")
        plt.ylabel("Doluluk (%)")
        plt.legend(loc="upper right")
        add_info_box(plt.gcf(), name, m5, m10)
        plt.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
        plt.savefig(fig_dir / f"{name}_projection_{start_date.date()}_2040.png", dpi=160)
        plt.savefig(fig_dir / f"{name}_projection_zoom_2040.png", dpi=160)
        plt.close()

        all_rows.append(out)

    pd.concat(all_rows).to_csv(out_dir / "projection_all_models.csv", index=False)


if __name__ == "__main__":
    main()
