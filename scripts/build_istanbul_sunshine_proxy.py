#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


ISTANBUL_LAT = 41.0082
ISTANBUL_LON = 28.9784
GSC_MJ_M2_MIN = 0.0820


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a monthly Istanbul sunshine-duration proxy and solar-radiation series "
            "using CRU cloud cover, CRU 10-minute sunshine climatology, and NASA POWER."
        )
    )
    parser.add_argument(
        "--cru-cld-nc-gz",
        type=Path,
        default=None,
        help="Path to CRU TS cloud-cover NetCDF GZip, e.g. cru_ts4.09.1901.2024.cld.dat.nc.gz",
    )
    parser.add_argument(
        "--cloud-csv",
        type=Path,
        default=None,
        help="Optional pre-extracted monthly cloud-cover CSV with timestamp and cld_pct columns.",
    )
    parser.add_argument(
        "--cru-sunp-dat-gz",
        type=Path,
        required=True,
        help="Path to CRU 10-minute sunshine climatology file, e.g. grid_10min_sunp.dat.gz",
    )
    parser.add_argument(
        "--nasa-power-json",
        type=Path,
        required=True,
        help="Path to NASA POWER monthly JSON with ALLSKY_SFC_SW_DWN.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1911,
        help="First year to emit. Defaults to 1911.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last year to emit. Defaults to 2025.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/istanbul_sunshine_proxy"),
        help="Directory for CSV/JSON outputs.",
    )
    return parser.parse_args()


def _adjust_lon(target_lon: float, lon_values: np.ndarray) -> float:
    lon_values = np.asarray(lon_values, dtype=float)
    if lon_values.size == 0:
        return float(target_lon)
    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))
    if lon_min >= 0.0 and lon_max > 180.0 and target_lon < 0.0:
        return float(target_lon % 360.0)
    if lon_max <= 180.0 and target_lon > 180.0:
        return float(((target_lon + 180.0) % 360.0) - 180.0)
    return float(target_lon)


def daily_solar_geometry(date: pd.Timestamp, latitude_deg: float) -> tuple[float, float]:
    lat_rad = math.radians(float(latitude_deg))
    j = float(date.dayofyear)
    dr = 1.0 + 0.033 * math.cos((2.0 * math.pi / 365.0) * j)
    delta = 0.409 * math.sin((2.0 * math.pi / 365.0) * j - 1.39)
    ws_arg = -math.tan(lat_rad) * math.tan(delta)
    ws_arg = max(-1.0, min(1.0, ws_arg))
    ws = math.acos(ws_arg)
    ra_mj_m2_day = (24.0 * 60.0 / math.pi) * GSC_MJ_M2_MIN * dr * (
        ws * math.sin(lat_rad) * math.sin(delta) + math.cos(lat_rad) * math.cos(delta) * math.sin(ws)
    )
    daylength_hours = 24.0 / math.pi * ws
    return daylength_hours, ra_mj_m2_day


def build_monthly_geometry(start: str, end: str, latitude_deg: float) -> pd.DataFrame:
    months = pd.date_range(start=start, end=end, freq="MS")
    rows: list[dict[str, Any]] = []
    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(0)
        days = pd.date_range(month_start, month_end, freq="D")
        daylengths: list[float] = []
        ras_mj: list[float] = []
        for day in days:
            daylength_h, ra_mj = daily_solar_geometry(day, latitude_deg)
            daylengths.append(daylength_h)
            ras_mj.append(ra_mj)
        rows.append(
            {
                "timestamp": month_start,
                "days_in_month": int(len(days)),
                "possible_sunshine_hours_month": float(np.sum(daylengths)),
                "possible_sunshine_hours_day": float(np.mean(daylengths)),
                "extra_radiation_mj_m2_month": float(np.sum(ras_mj)),
                "extra_radiation_mj_m2_day": float(np.mean(ras_mj)),
                "extra_radiation_kwh_m2_month": float(np.sum(ras_mj) * 0.2777777778),
                "extra_radiation_kwh_m2_day": float(np.mean(ras_mj) * 0.2777777778),
            }
        )
    return pd.DataFrame(rows)


def load_cru_sunp_climatology(path: Path, latitude_deg: float, longitude_deg: float) -> tuple[pd.Series, dict[str, float]]:
    best_line: list[float] | None = None
    best_dist = float("inf")
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("Mark New") or line.startswith("Data format"):
                continue
            parts = line.split()
            if len(parts) < 14:
                continue
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                vals = [float(v) for v in parts[2:14]]
            except ValueError:
                continue
            scale = math.cos(math.radians(latitude_deg))
            dist = ((lat - latitude_deg) ** 2) + (((lon - longitude_deg) * scale) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_line = [lat, lon, *vals]
    if best_line is None:
        raise RuntimeError(f"Could not locate a CRU sunp row in {path}")
    lat, lon, *vals = best_line
    series = pd.Series(vals, index=pd.Index(range(1, 13), name="month"), dtype=float, name="sunp_climo_pct")
    meta = {
        "grid_lat": float(lat),
        "grid_lon": float(lon),
        "distance_deg_sq": float(best_dist),
    }
    return series, meta


def load_cru_cloud_series(path: Path, latitude_deg: float, longitude_deg: float) -> tuple[pd.Series, dict[str, float]]:
    with gzip.open(path, "rb") as fh:
        ds = xr.open_dataset(fh, engine="scipy", decode_times=True, mask_and_scale=True)
        data_vars = list(ds.data_vars)
        if "cld" in data_vars:
            var_name = "cld"
        elif len(data_vars) == 1:
            var_name = data_vars[0]
        else:
            raise RuntimeError(f"Could not determine cloud-cover variable from {data_vars}")
        lon_target = _adjust_lon(float(longitude_deg), ds["lon"].values)
        point = ds[var_name].sel(lat=float(latitude_deg), lon=lon_target, method="nearest")
        lat_sel = float(point["lat"].item())
        lon_sel = float(point["lon"].item())
        index = pd.to_datetime(point["time"].values)
        series = pd.Series(point.values.astype(float), index=index, name="cld_pct").sort_index()
    series = series.replace([-9999.0, -999.0], np.nan)
    series = series[(series.index >= pd.Timestamp("1901-01-01")) & (series.index <= pd.Timestamp("2024-12-01"))]
    meta = {
        "grid_lat": lat_sel,
        "grid_lon": lon_sel,
    }
    return series, meta


def load_cloud_series_csv(path: Path) -> tuple[pd.Series, dict[str, float]]:
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns or "cld_pct" not in frame.columns:
        raise RuntimeError(f"{path} must contain timestamp and cld_pct columns")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["cld_pct"] = pd.to_numeric(frame["cld_pct"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    meta = {}
    if "grid_lat" in frame.columns:
        meta["grid_lat"] = float(pd.to_numeric(frame["grid_lat"], errors="coerce").dropna().iloc[0])
    if "grid_lon" in frame.columns:
        meta["grid_lon"] = float(pd.to_numeric(frame["grid_lon"], errors="coerce").dropna().iloc[0])
    series = frame.set_index("timestamp")["cld_pct"]
    return series, meta


def load_nasa_power_monthly(path: Path) -> pd.Series:
    payload = json.loads(path.read_text(encoding="utf-8"))
    param = ((payload.get("properties") or {}).get("parameter") or {}).get("ALLSKY_SFC_SW_DWN")
    if not isinstance(param, dict) or not param:
        raise RuntimeError("NASA POWER JSON missing properties.parameter.ALLSKY_SFC_SW_DWN")
    rows: list[dict[str, Any]] = []
    for k, v in param.items():
        ts = pd.to_datetime(str(k), format="%Y%m", errors="coerce")
        val = pd.to_numeric(v, errors="coerce")
        if pd.isna(ts) or pd.isna(val):
            continue
        fval = float(val)
        if fval < 0:
            continue
        rows.append({"timestamp": ts, "solar_kwh_m2_day": fval})
    if not rows:
        raise RuntimeError(f"NASA POWER file {path} yielded no monthly rows")
    frame = pd.DataFrame(rows).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return frame.set_index("timestamp")["solar_kwh_m2_day"]


def fit_cloud_to_sunp(cld_climo: pd.Series, sunp_climo: pd.Series) -> dict[str, float]:
    joined = (
        pd.concat([cld_climo.rename("cld"), sunp_climo.rename("sunp")], axis=1)
        .dropna()
        .sort_index()
    )
    if len(joined) < 6:
        raise RuntimeError("Need at least 6 monthly climatology pairs to fit cloud->sunshine relation")
    x = joined["cld"].to_numpy(dtype=float)
    y = joined["sunp"].to_numpy(dtype=float)
    x_center = x - float(np.mean(x))
    y_center = y - float(np.mean(y))
    denom = float(np.dot(x_center, x_center))
    slope = float(np.dot(x_center, y_center) / denom) if denom > 0 else -1.0
    pred = y.mean() + slope * x_center
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "slope_pct_sunp_per_pct_cld": slope,
        "r2": r2,
    }


def fit_angstrom_coeffs(overlap: pd.DataFrame) -> dict[str, float]:
    work = overlap.dropna(subset=["sunp_proxy_pct", "extra_radiation_kwh_m2_day", "solar_kwh_m2_day"]).copy()
    work = work[work["extra_radiation_kwh_m2_day"] > 0]
    if len(work) < 24:
        raise RuntimeError("Need at least 24 overlap months to calibrate Angstrom-Prescott coefficients")
    x = np.clip(work["sunp_proxy_pct"].to_numpy(dtype=float) / 100.0, 0.0, 1.0)
    y = np.divide(
        work["solar_kwh_m2_day"].to_numpy(dtype=float),
        np.maximum(work["extra_radiation_kwh_m2_day"].to_numpy(dtype=float), 1e-9),
    )
    A = np.column_stack([np.ones_like(x), x])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a_s = float(np.clip(coeffs[0], 0.10, 0.45))
    b_s = float(np.clip(coeffs[1], 0.20, 0.75))
    pred = a_s + b_s * x
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    corr = float(np.corrcoef(pred, y)[0, 1]) if len(y) > 1 else float("nan")
    return {
        "a_s": a_s,
        "b_s": b_s,
        "rmse_ratio": rmse,
        "corr": corr,
        "rows": int(len(work)),
    }


def build_proxy(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    geometry = build_monthly_geometry(f"{args.start_year}-01-01", f"{args.end_year}-12-01", ISTANBUL_LAT)
    if args.cloud_csv is not None:
        cloud_series, cloud_meta = load_cloud_series_csv(args.cloud_csv)
        cloud_input = {
            "mode": "cloud_csv",
            "path": str(args.cloud_csv),
        }
    elif args.cru_cld_nc_gz is not None:
        cloud_series, cloud_meta = load_cru_cloud_series(args.cru_cld_nc_gz, ISTANBUL_LAT, ISTANBUL_LON)
        cloud_input = {
            "mode": "cru_cld_nc_gz",
            "path": str(args.cru_cld_nc_gz),
        }
    else:
        raise RuntimeError("Provide either --cloud-csv or --cru-cld-nc-gz")
    sunp_climo, sunp_meta = load_cru_sunp_climatology(args.cru_sunp_dat_gz, ISTANBUL_LAT, ISTANBUL_LON)
    nasa_series = load_nasa_power_monthly(args.nasa_power_json)

    cloud_frame = cloud_series.rename_axis("timestamp").reset_index()
    merged = geometry.merge(cloud_frame, on="timestamp", how="left")
    merged = merged.merge(
        nasa_series.rename("solar_kwh_m2_day").rename_axis("timestamp").reset_index(),
        on="timestamp",
        how="left",
    )

    cld_climo = (
        cloud_series[(cloud_series.index >= pd.Timestamp("1961-01-01")) & (cloud_series.index <= pd.Timestamp("1990-12-01"))]
        .groupby(cloud_series[(cloud_series.index >= pd.Timestamp("1961-01-01")) & (cloud_series.index <= pd.Timestamp("1990-12-01"))].index.month)
        .mean()
    )
    rel_meta = fit_cloud_to_sunp(cld_climo, sunp_climo)
    slope = float(rel_meta["slope_pct_sunp_per_pct_cld"])

    month_idx = merged["timestamp"].dt.month
    sunp_lookup = month_idx.map(sunp_climo.to_dict()).astype(float)
    cld_lookup = month_idx.map(cld_climo.to_dict()).astype(float)
    merged["sunp_climo_pct"] = sunp_lookup
    merged["cld_climo_1961_1990_pct"] = cld_lookup
    merged["sunp_proxy_pct"] = np.where(
        np.isfinite(merged["cld_pct"]),
        np.clip(merged["sunp_climo_pct"] + slope * (merged["cld_pct"] - merged["cld_climo_1961_1990_pct"]), 0.0, 100.0),
        np.nan,
    )
    merged["sunshine_hours_month_proxy"] = (
        merged["possible_sunshine_hours_month"] * merged["sunp_proxy_pct"] / 100.0
    )
    merged["sunshine_hours_day_proxy"] = np.divide(
        merged["sunshine_hours_month_proxy"],
        np.maximum(merged["days_in_month"], 1),
    )

    overlap = merged[
        (merged["timestamp"] >= pd.Timestamp("1981-01-01")) & (merged["timestamp"] <= pd.Timestamp("2024-12-01"))
    ].copy()
    ang_meta = fit_angstrom_coeffs(overlap)
    a_s = float(ang_meta["a_s"])
    b_s = float(ang_meta["b_s"])

    merged["radiation_ratio_proxy"] = a_s + b_s * np.clip(merged["sunp_proxy_pct"] / 100.0, 0.0, 1.0)
    merged["radiation_kwh_m2_day_proxy"] = merged["radiation_ratio_proxy"] * merged["extra_radiation_kwh_m2_day"]
    merged["radiation_mj_m2_day_proxy"] = merged["radiation_kwh_m2_day_proxy"] / 0.2777777778
    merged["radiation_kwh_m2_month_proxy"] = merged["radiation_kwh_m2_day_proxy"] * merged["days_in_month"]
    merged["radiation_mj_m2_month_proxy"] = merged["radiation_mj_m2_day_proxy"] * merged["days_in_month"]

    fill_2025 = merged["timestamp"].dt.year.eq(2025) & merged["solar_kwh_m2_day"].notna()
    if fill_2025.any():
        merged.loc[fill_2025, "radiation_kwh_m2_day_proxy"] = merged.loc[fill_2025, "solar_kwh_m2_day"]
        merged.loc[fill_2025, "radiation_mj_m2_day_proxy"] = merged.loc[fill_2025, "solar_kwh_m2_day"] / 0.2777777778
        merged.loc[fill_2025, "radiation_kwh_m2_month_proxy"] = (
            merged.loc[fill_2025, "solar_kwh_m2_day"] * merged.loc[fill_2025, "days_in_month"]
        )
        merged.loc[fill_2025, "radiation_mj_m2_month_proxy"] = (
            merged.loc[fill_2025, "solar_kwh_m2_day"] / 0.2777777778 * merged.loc[fill_2025, "days_in_month"]
        )
        inv = np.divide(
            merged.loc[fill_2025, "solar_kwh_m2_day"],
            np.maximum(merged.loc[fill_2025, "extra_radiation_kwh_m2_day"], 1e-9),
        )
        inv = (inv - a_s) / max(b_s, 1e-9)
        merged.loc[fill_2025, "sunp_proxy_pct"] = np.clip(inv * 100.0, 0.0, 100.0)
        merged.loc[fill_2025, "sunshine_hours_month_proxy"] = (
            merged.loc[fill_2025, "possible_sunshine_hours_month"] * merged.loc[fill_2025, "sunp_proxy_pct"] / 100.0
        )
        merged.loc[fill_2025, "sunshine_hours_day_proxy"] = np.divide(
            merged.loc[fill_2025, "sunshine_hours_month_proxy"],
            np.maximum(merged.loc[fill_2025, "days_in_month"], 1),
        )

    fill_2025_climo = merged["timestamp"].dt.year.eq(2025) & merged["sunp_proxy_pct"].isna()
    if fill_2025_climo.any():
        merged.loc[fill_2025_climo, "sunp_proxy_pct"] = merged.loc[fill_2025_climo, "sunp_climo_pct"]
        merged.loc[fill_2025_climo, "sunshine_hours_month_proxy"] = (
            merged.loc[fill_2025_climo, "possible_sunshine_hours_month"]
            * merged.loc[fill_2025_climo, "sunp_proxy_pct"]
            / 100.0
        )
        merged.loc[fill_2025_climo, "sunshine_hours_day_proxy"] = np.divide(
            merged.loc[fill_2025_climo, "sunshine_hours_month_proxy"],
            np.maximum(merged.loc[fill_2025_climo, "days_in_month"], 1),
        )
        merged.loc[fill_2025_climo, "radiation_ratio_proxy"] = (
            a_s + b_s * np.clip(merged.loc[fill_2025_climo, "sunp_proxy_pct"] / 100.0, 0.0, 1.0)
        )
        merged.loc[fill_2025_climo, "radiation_kwh_m2_day_proxy"] = (
            merged.loc[fill_2025_climo, "radiation_ratio_proxy"]
            * merged.loc[fill_2025_climo, "extra_radiation_kwh_m2_day"]
        )
        merged.loc[fill_2025_climo, "radiation_mj_m2_day_proxy"] = (
            merged.loc[fill_2025_climo, "radiation_kwh_m2_day_proxy"] / 0.2777777778
        )
        merged.loc[fill_2025_climo, "radiation_kwh_m2_month_proxy"] = (
            merged.loc[fill_2025_climo, "radiation_kwh_m2_day_proxy"]
            * merged.loc[fill_2025_climo, "days_in_month"]
        )
        merged.loc[fill_2025_climo, "radiation_mj_m2_month_proxy"] = (
            merged.loc[fill_2025_climo, "radiation_mj_m2_day_proxy"]
            * merged.loc[fill_2025_climo, "days_in_month"]
        )

    merged["sunshine_source"] = "cru_cloud_proxy"
    merged.loc[fill_2025, "sunshine_source"] = "nasa_inversion"
    merged.loc[fill_2025_climo, "sunshine_source"] = "climatology_fill"
    merged["radiation_source"] = "angstrom_proxy"
    merged.loc[fill_2025, "radiation_source"] = "nasa_power_direct"
    merged.loc[fill_2025_climo, "radiation_source"] = "angstrom_climatology_fill"
    merged = merged[
        [
            "timestamp",
            "days_in_month",
            "cld_pct",
            "cld_climo_1961_1990_pct",
            "sunp_climo_pct",
            "sunp_proxy_pct",
            "possible_sunshine_hours_day",
            "possible_sunshine_hours_month",
            "sunshine_hours_day_proxy",
            "sunshine_hours_month_proxy",
            "extra_radiation_mj_m2_day",
            "extra_radiation_kwh_m2_day",
            "radiation_mj_m2_day_proxy",
            "radiation_kwh_m2_day_proxy",
            "radiation_mj_m2_month_proxy",
            "radiation_kwh_m2_month_proxy",
            "solar_kwh_m2_day",
            "sunshine_source",
            "radiation_source",
        ]
    ].copy()

    meta: dict[str, Any] = {
        "location": {
            "name": "Istanbul",
            "latitude": ISTANBUL_LAT,
            "longitude": ISTANBUL_LON,
        },
        "coverage": {
            "start": f"{args.start_year}-01-01",
            "end": f"{args.end_year}-12-01",
            "rows": int(len(merged)),
        },
        "cloud_input": cloud_input,
        "cloud_series_grid": cloud_meta,
        "sunp_climatology_grid": sunp_meta,
        "cloud_to_sunp_fit": rel_meta,
        "angstrom_calibration": ang_meta,
        "notes": [
            "Sunshine is a proxy, not a raw station observation.",
            "For 1911-2020, sunshine is inferred from monthly cloud cover using the 1961-1990 CRU 10-minute sunshine climatology as the monthly anchor.",
            "For 2021-2025 months with NASA POWER availability, sunshine is back-calculated from monthly solar radiation using calibrated Angstrom-Prescott coefficients.",
            "Any remaining 2025 gaps are filled with monthly climatology before converting back to radiation.",
        ],
        "sources": [
            {
                "name": "CRU TS v4.09 cloud cover",
                "url": "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.09/",
            },
            {
                "name": "CRU CL v2.0 10-minute sunshine climatology",
                "url": "https://crudata.uea.ac.uk/cru/data/hrg/tmc/",
            },
            {
                "name": "NASA POWER monthly ALLSKY_SFC_SW_DWN",
                "url": "https://power.larc.nasa.gov/api/temporal/monthly/point",
            },
        ],
    }
    return merged, meta


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, meta = build_proxy(args)

    csv_path = args.output_dir / "istanbul_monthly_sunshine_radiation_1911_2025.csv"
    json_path = args.output_dir / "istanbul_monthly_sunshine_radiation_1911_2025_meta.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(
        "Summary:",
        json.dumps(
            {
                "rows": int(len(frame)),
                "start": str(frame["timestamp"].min().date()),
                "end": str(frame["timestamp"].max().date()),
                "sunp_mean_pct": float(frame["sunp_proxy_pct"].mean(skipna=True)),
                "radiation_mean_kwh_m2_day": float(frame["radiation_kwh_m2_day_proxy"].mean(skipna=True)),
            },
            ensure_ascii=True,
        ),
    )


if __name__ == "__main__":
    main()
