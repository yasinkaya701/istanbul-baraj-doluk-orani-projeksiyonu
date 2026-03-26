#!/usr/bin/env python3
"""Join anomaly event days with the daily climate dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract daily climate rows for filtered anomaly events and anomaly days."
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/tum_asiri_olaylar_bilimsel_filtreli.csv"),
        help="Filtered event-level anomaly table.",
    )
    parser.add_argument(
        "--points-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/tum_asiri_olay_noktalari.csv"),
        help="Point-level anomaly table with timestamps.",
    )
    parser.add_argument(
        "--daily-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/spreadsheet/es_ea_newdata_daily.csv"),
        help="Daily climate dataset to join on date.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/anomaly_day_data"),
        help="Directory for extracted CSV outputs.",
    )
    return parser.parse_args()


def read_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path).copy()
    for col in ["start_time", "end_time", "center_time", "context_ds"]:
        if col in events.columns:
            events[col] = pd.to_datetime(events[col], errors="coerce")
    events["variable"] = events["variable"].astype(str).str.strip().str.lower()
    events["center_day"] = events["center_time"].dt.normalize()
    return events


def read_points(path: Path) -> pd.DataFrame:
    points = pd.read_csv(path).copy()
    points["timestamp"] = pd.to_datetime(points["timestamp"], errors="coerce")
    points["variable"] = points["variable"].astype(str).str.strip().str.lower()
    points["date"] = points["timestamp"].dt.normalize()
    rename_map = {
        "value": "point_value",
        "unit": "point_unit",
        "q_low": "point_q_low",
        "q_high": "point_q_high",
        "robust_z": "point_robust_z",
        "delta_prev": "point_delta_prev",
        "delta_abs": "point_delta_abs",
        "jump_threshold": "point_jump_threshold",
        "is_low_tail": "point_is_low_tail",
        "is_high_tail": "point_is_high_tail",
        "is_robust_outlier": "point_is_robust_outlier",
        "is_jump": "point_is_jump",
        "is_extreme": "point_is_extreme",
        "severity_score": "point_severity_score",
        "severity_level": "point_severity_level",
        "direction": "point_direction",
        "reason_tags": "point_reason_tags",
        "event_local_id": "point_event_local_id",
    }
    return points.rename(columns=rename_map)


def read_daily(path: Path) -> pd.DataFrame:
    daily = pd.read_csv(path).copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    return daily


def pipe_join(values: pd.Series) -> str:
    clean = []
    for value in values.dropna():
        text = str(value).strip()
        if text and text not in clean:
            clean.append(text)
    return " | ".join(clean)


def build_unique_day_summary(points_enriched: pd.DataFrame, daily_cols: list[str]) -> pd.DataFrame:
    ordered = points_enriched.sort_values(
        ["date", "point_severity_score", "peak_severity_score"],
        ascending=[True, False, False],
    )
    top_per_day = ordered.groupby("date", as_index=False).first()

    grouped = (
        ordered.groupby("date", as_index=False)
        .agg(
            anomaly_point_count=("timestamp", "size"),
            event_count=("event_id", "nunique"),
            variable_count=("variable", "nunique"),
            variables=("variable", pipe_join),
            event_ids=("event_id", pipe_join),
            point_timestamps=("timestamp", lambda s: pipe_join(s.dt.strftime("%Y-%m-%d %H:%M:%S"))),
            point_values=("point_value", lambda s: pipe_join(s.round(4).astype(str))),
            point_units=("point_unit", pipe_join),
            point_reason_tags=("point_reason_tags", pipe_join),
            event_severity_levels=("severity_level", pipe_join),
            scientific_tiers=("scientific_tier", pipe_join),
        )
    )

    top_cols = [
        "date",
        "event_id",
        "variable",
        "timestamp",
        "point_value",
        "point_unit",
        "point_severity_score",
        "point_severity_level",
        "peak_severity_score",
        "severity_level",
        "scientific_score",
        "scientific_tier",
    ]
    top_per_day = top_per_day[top_cols].rename(
        columns={
            "event_id": "top_event_id",
            "variable": "top_variable",
            "timestamp": "top_point_timestamp",
            "point_value": "top_point_value",
            "point_unit": "top_point_unit",
            "point_severity_score": "top_point_severity_score",
            "point_severity_level": "top_point_severity_level",
            "peak_severity_score": "top_event_peak_severity_score",
            "severity_level": "top_event_severity_level",
            "scientific_score": "top_scientific_score",
            "scientific_tier": "top_scientific_tier",
        }
    )

    keep_daily = ["date"] + daily_cols
    return grouped.merge(top_per_day, on="date", how="left").merge(
        ordered[keep_daily].drop_duplicates(subset=["date"]),
        on="date",
        how="left",
    ).sort_values(["date", "top_point_severity_score"], ascending=[True, False])


def nearest_available_dates(date_value: pd.Timestamp, available_dates: pd.Series) -> tuple[str, str]:
    if pd.isna(date_value):
        return "", ""
    before = available_dates[available_dates < date_value]
    after = available_dates[available_dates > date_value]
    prev_date = before.iloc[-1].strftime("%Y-%m-%d") if not before.empty else ""
    next_date = after.iloc[0].strftime("%Y-%m-%d") if not after.empty else ""
    return prev_date, next_date


def format_timestamp(value: pd.Timestamp | str | None, date_only: bool = False) -> str:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d" if date_only else "%Y-%m-%d %H:%M:%S")


def build_missing_daily_rows(
    events_enriched: pd.DataFrame,
    points_enriched: pd.DataFrame,
    daily: pd.DataFrame,
    daily_cols: list[str],
) -> pd.DataFrame:
    available_dates = daily["date"].dropna().drop_duplicates().sort_values().reset_index(drop=True)
    rows: list[dict[str, str]] = []

    event_daily_cols = [f"daily_{col}" for col in daily_cols]
    missing_events = events_enriched[events_enriched[event_daily_cols].isna().all(axis=1)]
    for _, row in missing_events.iterrows():
        prev_date, next_date = nearest_available_dates(row["center_day"], available_dates)
        rows.append(
            {
                "join_scope": "event_center_day",
                "event_id": row["event_id"],
                "variable": row["variable"],
                "reference_timestamp": format_timestamp(row["center_time"]),
                "missing_date": format_timestamp(row["center_day"], date_only=True),
                "prev_available_date": prev_date,
                "next_available_date": next_date,
            }
        )

    missing_points = points_enriched[points_enriched[daily_cols].isna().all(axis=1)]
    for _, row in missing_points.iterrows():
        prev_date, next_date = nearest_available_dates(row["date"], available_dates)
        rows.append(
            {
                "join_scope": "anomaly_point_day",
                "event_id": row["event_id"],
                "variable": row["variable"],
                "reference_timestamp": format_timestamp(row["timestamp"]),
                "missing_date": format_timestamp(row["date"], date_only=True),
                "prev_available_date": prev_date,
                "next_available_date": next_date,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "join_scope",
                "event_id",
                "variable",
                "reference_timestamp",
                "missing_date",
                "prev_available_date",
                "next_available_date",
            ]
        )
    return pd.DataFrame(rows).sort_values(["missing_date", "join_scope", "event_id"])


def write_summary(
    output_dir: Path,
    events_enriched: pd.DataFrame,
    points_enriched: pd.DataFrame,
    unique_days: pd.DataFrame,
    daily_cols: list[str],
    missing_daily_rows: pd.DataFrame,
) -> Path:
    event_daily_cols = [f"daily_{col}" for col in daily_cols]
    daily_missing_event = int(events_enriched[event_daily_cols].isna().all(axis=1).sum())
    daily_missing_point = int(points_enriched[daily_cols].isna().all(axis=1).sum())
    path = output_dir / "anomaly_day_data_summary.md"
    lines = [
        "# Anomaly Day Climate Extraction Summary",
        "",
        f"- Filtered events: **{len(events_enriched)}**",
        f"- Event center days: **{events_enriched['center_day'].nunique()}**",
        f"- Filtered anomaly points: **{len(points_enriched)}**",
        f"- Unique anomaly days from points: **{unique_days['date'].nunique()}**",
        f"- Missing daily climate rows for center-day join: **{daily_missing_event}**",
        f"- Missing daily climate rows for point-day join: **{daily_missing_point}**",
        "",
        "## Files",
        "",
        "- `anomaly_events_center_day_with_daily_climate.csv`: one row per filtered event, joined on `center_time` day.",
        "- `anomaly_points_with_daily_climate.csv`: one row per anomaly timestamp/day point, joined on daily date.",
        "- `anomaly_unique_days_with_daily_climate.csv`: one row per unique anomaly day, with aggregated events and daily climate fields.",
        "- `anomaly_days_missing_daily_climate.csv`: anomaly rows whose date is absent from the daily climate dataset.",
        "",
        "## Daily Climate Columns",
        "",
        f"- {', '.join(daily_cols)}",
        "",
    ]
    if not missing_daily_rows.empty:
        lines += [
            "## Missing Daily Dates",
            "",
            "|Join|Event ID|Variable|Missing Date|Prev Available|Next Available|",
            "|---|---|---|---|---|---|",
        ]
        for _, row in missing_daily_rows.iterrows():
            lines.append(
                f"|{row['join_scope']}|{row['event_id']}|{row['variable']}|{row['missing_date']}|{row['prev_available_date'] or '-'}|{row['next_available_date'] or '-'}|"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events = read_events(args.events_csv)
    points = read_points(args.points_csv)
    daily = read_daily(args.daily_csv)
    daily_cols = [col for col in daily.columns if col != "date"]

    filtered_pairs = events[["event_id", "variable"]].drop_duplicates()
    points_filtered = points.merge(filtered_pairs, on=["event_id", "variable"], how="inner")

    events_enriched = events.merge(
        daily.add_prefix("daily_").rename(columns={"daily_date": "center_day"}),
        on="center_day",
        how="left",
    )
    events_enriched = events_enriched.sort_values(
        ["center_day", "peak_severity_score", "event_id"],
        ascending=[True, False, True],
    )

    points_enriched = points_filtered.merge(events, on=["event_id", "variable"], how="left")
    points_enriched = points_enriched.merge(daily, on="date", how="left")
    points_enriched = points_enriched.sort_values(
        ["date", "point_severity_score", "peak_severity_score", "event_id"],
        ascending=[True, False, False, True],
    )

    unique_days = build_unique_day_summary(points_enriched, daily_cols)
    missing_daily_rows = build_missing_daily_rows(events_enriched, points_enriched, daily, daily_cols)

    center_day_csv = args.output_dir / "anomaly_events_center_day_with_daily_climate.csv"
    point_csv = args.output_dir / "anomaly_points_with_daily_climate.csv"
    unique_day_csv = args.output_dir / "anomaly_unique_days_with_daily_climate.csv"
    missing_csv = args.output_dir / "anomaly_days_missing_daily_climate.csv"

    events_enriched.to_csv(center_day_csv, index=False)
    points_enriched.to_csv(point_csv, index=False)
    unique_days.to_csv(unique_day_csv, index=False)
    missing_daily_rows.to_csv(missing_csv, index=False)

    summary_md = write_summary(
        output_dir=args.output_dir,
        events_enriched=events_enriched,
        points_enriched=points_enriched,
        unique_days=unique_days,
        daily_cols=daily_cols,
        missing_daily_rows=missing_daily_rows,
    )

    print(f"Wrote: {center_day_csv}")
    print(f"Wrote: {point_csv}")
    print(f"Wrote: {unique_day_csv}")
    print(f"Wrote: {missing_csv}")
    print(f"Wrote: {summary_md}")
    print(
        "Counts:",
        {
            "events": len(events_enriched),
            "center_days": int(events_enriched["center_day"].nunique()),
            "points": len(points_enriched),
            "unique_point_days": int(unique_days["date"].nunique()),
        },
    )


if __name__ == "__main__":
    main()
