#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_MODELS = [
    "quant",
    "prophet",
    "strong",
    "analog",
    "prophet_ultra",
    "walkforward",
    "best_meta",
    "literature",
    "stable_consensus",
]


@dataclass
class HealthRun:
    model: str
    status: str
    temp_csv: str
    humidity_csv: str
    temp_selection: str
    humidity_selection: str
    date_col: str
    output_dir: str
    message: str
    future_mean_rr: float | None
    future_high_risk_share: float | None
    future_mean_heat_index_c: float | None
    delta_mean_rr: float | None
    command: str
    returncode: int
    stderr_tail: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run human-health impact analysis across model outputs.")
    p.add_argument("--run-dir", type=Path, required=True, help="Model suite output directory")
    p.add_argument("--models", type=str, default="*", help="Comma-separated model names or *")
    p.add_argument("--output-subdir", type=str, default="health", help="Subdir under run-dir for health outputs")
    p.add_argument("--baseline-start", type=int, default=1991)
    p.add_argument("--baseline-end", type=int, default=2020)
    p.add_argument("--future-start", type=int, default=2026)
    p.add_argument("--future-end", type=int, default=-1, help="<=0 means infer from run-dir")
    p.add_argument("--python-bin", type=str, default=sys.executable)
    return p.parse_args()


def tail_text(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def parse_model_list(text: str) -> list[str]:
    if str(text).strip() == "*":
        return list(SUPPORTED_MODELS)
    out = []
    for tok in [x.strip().lower() for x in str(text).split(",") if x.strip()]:
        if tok in SUPPORTED_MODELS and tok not in out:
            out.append(tok)
    return out


def infer_target_year(run_dir: Path) -> int | None:
    years: list[int] = []
    for p in run_dir.rglob("*.csv"):
        name = p.name
        m = re.search(r"_to_(\d{4})\.csv$", name)
        if m:
            years.append(int(m.group(1)))
            continue
        m2 = re.search(r"_(\d{4})_(\d{4})\.csv$", name)
        if m2:
            years.append(int(m2.group(2)))
    if not years:
        return None
    years = [y for y in years if 1900 <= y <= 2200]
    return max(years) if years else None


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []


def detect_date_col(path: Path) -> str | None:
    cols = [c.strip() for c in read_csv_header(path)]
    for c in ("ds", "timestamp", "date", "datetime"):
        if c in cols:
            return c
    return cols[0] if cols else None


def _extract_year(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    m = re.search(r"(\d{4})", text)
    if not m:
        return None
    y = int(m.group(1))
    if 1800 <= y <= 2300:
        return y
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(dt.year)


def year_bounds(path: Path, date_col: str) -> tuple[int | None, int | None]:
    min_y: int | None = None
    max_y: int | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                y = _extract_year(str(row.get(date_col, "")))
                if y is None:
                    continue
                if min_y is None or y < min_y:
                    min_y = y
                if max_y is None or y > max_y:
                    max_y = y
    except OSError:
        return None, None
    return min_y, max_y


def _freq_rank(name: str) -> int:
    n = str(name).lower()
    if "_monthly_" in n or "_ms_" in n or n == "ms" or "monthly" in n:
        return 0
    if "_daily_" in n or "_d_" in n or n == "d" or "daily" in n:
        return 1
    if "_hourly_" in n or "_h_" in n or n == "h" or "hourly" in n:
        return 2
    if "_yearly_" in n or "_ys_" in n or n == "ys" or n == "y" or "yearly" in n:
        return 3
    return 9


def _to_float(value: Any) -> float | None:
    try:
        v = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _extract_year_hint(name: str) -> int:
    m = re.search(r"_to_(\d{4})(?:\D|$)", name)
    if m:
        return int(m.group(1))
    m2 = re.search(r"_(\d{4})_(\d{4})(?:\D|$)", name)
    if m2:
        return int(m2.group(2))
    return -1


def _metric_from_row(row: dict[str, str]) -> tuple[float, str] | None:
    # Lower is better.
    priority = [
        "score",
        "weighted_metric_proxy",
        "cv_rmse",
        "rmse_cv",
        "rmse",
        "best_cv_rmse",
        "mae_cv",
        "mae",
        "cv_mae",
        "smape_cv",
        "smape",
        "mape",
        "rank",
    ]
    for k in priority:
        if k not in row:
            continue
        v = _to_float(row.get(k))
        if v is not None:
            return v, k
    return None


def _normalize_forecast_path(raw: str, model_dir: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text or text.startswith("generated://"):
        return None
    p = Path(text)
    if p.exists():
        return p.resolve()
    p2 = (Path.cwd() / text).resolve()
    if p2.exists():
        return p2
    p3 = (model_dir / text).resolve()
    if p3.exists():
        return p3
    return None


def _metric_sources(model_dir: Path) -> list[Path]:
    out: list[Path] = []
    for pat in ("*index*.csv", "*metrics*.csv"):
        out.extend(sorted(model_dir.glob(pat)))
    leaderboards = model_dir / "leaderboards"
    if leaderboards.exists():
        out.extend(sorted(leaderboards.glob("*.csv")))
    # Keep deterministic unique order.
    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq


def _collect_metrics(model_dir: Path, variable: str) -> tuple[dict[Path, tuple[float, str]], list[tuple[float, int, str]]]:
    var = variable.lower()
    by_path: dict[Path, tuple[float, str]] = {}
    generic: list[tuple[float, int, str]] = []
    for src in _metric_sources(model_dir):
        try:
            with src.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                headers = [h.strip() for h in (reader.fieldnames or []) if h]
                if not headers:
                    continue
                src_has_var_col = "variable" in headers
                src_name_has_var = var in src.name.lower()
                src_freq_rank = _freq_rank(src.name)
                for row in reader:
                    row_var = str(row.get("variable", "")).strip().lower()
                    if src_has_var_col and row_var and row_var != var:
                        continue
                    metric = _metric_from_row(row)
                    if metric is None:
                        continue
                    metric_val, metric_key = metric
                    row_freq_rank = _freq_rank(str(row.get("frequency", "")).strip()) if "frequency" in row else src_freq_rank
                    desc = f"{src.name}:{metric_key}={metric_val:.6g}"
                    forecast_csv = row.get("forecast_csv", "")
                    fc_path = _normalize_forecast_path(forecast_csv, model_dir=model_dir)
                    generic_ok = src_has_var_col or src_name_has_var
                    if fc_path is not None and var in fc_path.name.lower():
                        best = by_path.get(fc_path)
                        if best is None or metric_val < best[0]:
                            by_path[fc_path] = (metric_val, desc)
                        # Keep as generic too; some indexes point to source member paths
                        # while candidate set may contain blended outputs.
                        if generic_ok:
                            generic.append((metric_val, row_freq_rank, desc))
                    elif generic_ok:
                        generic.append((metric_val, row_freq_rank, desc))
        except OSError:
            continue
    return by_path, generic


def pick_forecast_csv(model_dir: Path, variable: str) -> tuple[Path | None, str]:
    forecast_dir = model_dir / "forecasts"
    if not forecast_dir.exists():
        return None, "forecasts dir missing"
    candidates = [p for p in forecast_dir.glob("*.csv") if variable in p.name.lower()]
    if not candidates:
        return None, f"{variable} candidate missing"

    metric_by_path, generic_metrics = _collect_metrics(model_dir=model_dir, variable=variable)
    generic_best_by_freq: dict[int, tuple[float, str]] = {}
    generic_global_best: tuple[float, str] | None = None
    for metric_val, freq_rank, desc in generic_metrics:
        if (freq_rank not in generic_best_by_freq) or (metric_val < generic_best_by_freq[freq_rank][0]):
            generic_best_by_freq[freq_rank] = (metric_val, desc)
        if generic_global_best is None or metric_val < generic_global_best[0]:
            generic_global_best = (metric_val, desc)

    ranked: list[tuple[tuple[int, float, int, int, str], Path, str]] = []
    for p in candidates:
        rp = p.resolve()
        freq_rank = _freq_rank(p.name)
        metric = metric_by_path.get(rp)
        metric_note = ""
        if metric is None and freq_rank in generic_best_by_freq:
            metric = generic_best_by_freq[freq_rank]
            metric_note = " (freq-matched)"
        if metric is None and generic_global_best is not None:
            metric = generic_global_best
            metric_note = " (global)"

        if metric is not None:
            has_metric = 0
            metric_val = float(metric[0])
            reason = f"metric:{metric[1]}{metric_note}"
        else:
            has_metric = 1
            metric_val = float("inf")
            reason = "fallback:name+frequency"

        year_hint = _extract_year_hint(p.name)
        ranked.append(((has_metric, metric_val, freq_rank, -year_hint, p.name), p, reason))

    ranked.sort(key=lambda x: x[0])
    _, chosen, reason = ranked[0]
    return chosen, reason


def summarize_health(summary_path: Path) -> dict[str, Any]:
    d = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "future_mean_rr": d.get("future", {}).get("mean_proxy_relative_risk"),
        "future_high_risk_share": d.get("future", {}).get("high_risk_month_share"),
        "future_mean_heat_index_c": d.get("future", {}).get("mean_heat_index_c"),
        "delta_mean_rr": d.get("delta", {}).get("mean_proxy_relative_risk_delta"),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"run-dir not found: {run_dir}")

    models = parse_model_list(args.models)
    if not models:
        raise SystemExit("No valid model selected.")

    future_end = int(args.future_end)
    if future_end <= 0:
        inferred = infer_target_year(run_dir)
        future_end = inferred if inferred is not None else 2035
    future_start = int(args.future_start)
    if future_start > future_end:
        future_start = max(1900, future_end - 1)

    health_root = run_dir / args.output_subdir
    health_root.mkdir(parents=True, exist_ok=True)

    results: list[HealthRun] = []

    for model in models:
        model_dir = run_dir / model
        temp_csv, temp_selection = pick_forecast_csv(model_dir, "temp")
        humidity_csv, humidity_selection = pick_forecast_csv(model_dir, "humidity")

        if temp_csv is None or humidity_csv is None:
            results.append(
                HealthRun(
                    model=model,
                    status="skipped_missing_inputs",
                    temp_csv=str(temp_csv or ""),
                    humidity_csv=str(humidity_csv or ""),
                    temp_selection=temp_selection,
                    humidity_selection=humidity_selection,
                    date_col="",
                    output_dir=str(health_root / model),
                    message="temp/humidity forecast csv missing",
                    future_mean_rr=None,
                    future_high_risk_share=None,
                    future_mean_heat_index_c=None,
                    delta_mean_rr=None,
                    command="",
                    returncode=0,
                    stderr_tail="",
                )
            )
            continue

        date_col_temp = detect_date_col(temp_csv)
        date_col_hum = detect_date_col(humidity_csv)
        if not date_col_temp or not date_col_hum:
            results.append(
                HealthRun(
                    model=model,
                    status="skipped_bad_schema",
                    temp_csv=str(temp_csv),
                    humidity_csv=str(humidity_csv),
                    temp_selection=temp_selection,
                    humidity_selection=humidity_selection,
                    date_col="",
                    output_dir=str(health_root / model),
                    message="date column missing",
                    future_mean_rr=None,
                    future_high_risk_share=None,
                    future_mean_heat_index_c=None,
                    delta_mean_rr=None,
                    command="",
                    returncode=0,
                    stderr_tail="",
                )
            )
            continue
        if date_col_temp != date_col_hum:
            results.append(
                HealthRun(
                    model=model,
                    status="skipped_date_mismatch",
                    temp_csv=str(temp_csv),
                    humidity_csv=str(humidity_csv),
                    temp_selection=temp_selection,
                    humidity_selection=humidity_selection,
                    date_col=f"{date_col_temp}|{date_col_hum}",
                    output_dir=str(health_root / model),
                    message="date columns differ between temp/humidity files",
                    future_mean_rr=None,
                    future_high_risk_share=None,
                    future_mean_heat_index_c=None,
                    delta_mean_rr=None,
                    command="",
                    returncode=0,
                    stderr_tail="",
                )
            )
            continue

        temp_min_y, temp_max_y = year_bounds(temp_csv, date_col_temp)
        hum_min_y, hum_max_y = year_bounds(humidity_csv, date_col_hum)
        if temp_max_y is None or hum_max_y is None:
            results.append(
                HealthRun(
                    model=model,
                    status="skipped_bad_dates",
                    temp_csv=str(temp_csv),
                    humidity_csv=str(humidity_csv),
                    temp_selection=temp_selection,
                    humidity_selection=humidity_selection,
                    date_col=date_col_temp,
                    output_dir=str(health_root / model),
                    message="date parse failed",
                    future_mean_rr=None,
                    future_high_risk_share=None,
                    future_mean_heat_index_c=None,
                    delta_mean_rr=None,
                    command="",
                    returncode=0,
                    stderr_tail="",
                )
            )
            continue

        overlap_start = max(temp_min_y or -10**9, hum_min_y or -10**9)
        overlap_end = min(temp_max_y or 10**9, hum_max_y or 10**9)
        if overlap_start > overlap_end:
            results.append(
                HealthRun(
                    model=model,
                    status="skipped_no_year_overlap",
                    temp_csv=str(temp_csv),
                    humidity_csv=str(humidity_csv),
                    temp_selection=temp_selection,
                    humidity_selection=humidity_selection,
                    date_col=date_col_temp,
                    output_dir=str(health_root / model),
                    message=f"year ranges do not overlap ({temp_min_y}-{temp_max_y} vs {hum_min_y}-{hum_max_y})",
                    future_mean_rr=None,
                    future_high_risk_share=None,
                    future_mean_heat_index_c=None,
                    delta_mean_rr=None,
                    command="",
                    returncode=0,
                    stderr_tail="",
                )
            )
            continue

        if temp_max_y < future_start or hum_max_y < future_start:
            results.append(
                HealthRun(
                    model=model,
                    status="skipped_insufficient_future_coverage",
                    temp_csv=str(temp_csv),
                    humidity_csv=str(humidity_csv),
                    temp_selection=temp_selection,
                    humidity_selection=humidity_selection,
                    date_col=date_col_temp,
                    output_dir=str(health_root / model),
                    message=f"future start {future_start} not covered ({temp_max_y}, {hum_max_y})",
                    future_mean_rr=None,
                    future_high_risk_share=None,
                    future_mean_heat_index_c=None,
                    delta_mean_rr=None,
                    command="",
                    returncode=0,
                    stderr_tail="",
                )
            )
            continue

        out_dir = health_root / model
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python_bin,
            "scripts/health_impact_analysis.py",
            "--temp-csv",
            str(temp_csv),
            "--humidity-csv",
            str(humidity_csv),
            "--output-dir",
            str(out_dir),
            "--date-col",
            date_col_temp,
            "--baseline-start",
            str(args.baseline_start),
            "--baseline-end",
            str(args.baseline_end),
            "--future-start",
            str(future_start),
            "--future-end",
            str(future_end),
        ]

        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        stderr_tail = tail_text(proc.stderr or "")
        stdout_tail = tail_text(proc.stdout or "")

        if proc.returncode == 0:
            summary_path = out_dir / "health_impact_summary.json"
            if summary_path.exists():
                m = summarize_health(summary_path)
                status = "ok"
                message = "ok"
            else:
                m = {
                    "future_mean_rr": None,
                    "future_high_risk_share": None,
                    "future_mean_heat_index_c": None,
                    "delta_mean_rr": None,
                }
                status = "failed_missing_summary"
                message = "health_impact_summary.json missing after run"
            results.append(
                HealthRun(
                    model=model,
                    status=status,
                    temp_csv=str(temp_csv),
                    humidity_csv=str(humidity_csv),
                    temp_selection=temp_selection,
                    humidity_selection=humidity_selection,
                    date_col=date_col_temp,
                    output_dir=str(out_dir),
                    message=message,
                    future_mean_rr=m["future_mean_rr"],
                    future_high_risk_share=m["future_high_risk_share"],
                    future_mean_heat_index_c=m["future_mean_heat_index_c"],
                    delta_mean_rr=m["delta_mean_rr"],
                    command=" ".join(cmd),
                    returncode=proc.returncode,
                    stderr_tail=stderr_tail,
                )
            )
        else:
            fail_msg = stderr_tail or stdout_tail
            fail_status = "failed"
            if "do not overlap on date axis" in fail_msg.lower():
                fail_status = "skipped_no_overlap"
            results.append(
                HealthRun(
                    model=model,
                    status=fail_status,
                    temp_csv=str(temp_csv),
                    humidity_csv=str(humidity_csv),
                    temp_selection=temp_selection,
                    humidity_selection=humidity_selection,
                    date_col=date_col_temp,
                    output_dir=str(out_dir),
                    message=fail_msg.splitlines()[-1] if fail_msg else "health run failed",
                    future_mean_rr=None,
                    future_high_risk_share=None,
                    future_mean_heat_index_c=None,
                    delta_mean_rr=None,
                    command=" ".join(cmd),
                    returncode=proc.returncode,
                    stderr_tail=stderr_tail,
                )
            )

    rows = [
        {
            "model": r.model,
            "status": r.status,
            "temp_csv": r.temp_csv,
            "humidity_csv": r.humidity_csv,
            "temp_selection": r.temp_selection,
            "humidity_selection": r.humidity_selection,
            "date_col": r.date_col,
            "output_dir": r.output_dir,
            "message": r.message,
            "future_mean_rr": r.future_mean_rr,
            "future_high_risk_share": r.future_high_risk_share,
            "future_mean_heat_index_c": r.future_mean_heat_index_c,
            "delta_mean_rr": r.delta_mean_rr,
            "returncode": r.returncode,
            "command": r.command,
            "stderr_tail": r.stderr_tail,
        }
        for r in results
    ]

    ok_models = [r.model for r in results if r.status == "ok"]
    skipped_models = [r.model for r in results if r.status.startswith("skipped")]
    failed_models = [r.model for r in results if r.status.startswith("failed")]

    summary = {
        "run_dir": str(run_dir),
        "health_root": str(health_root),
        "models_requested": models,
        "models_ok": ok_models,
        "models_skipped": skipped_models,
        "models_failed": failed_models,
        "future_start": future_start,
        "future_end": future_end,
        "results": rows,
    }

    summary_json = health_root / "health_suite_summary.json"
    summary_csv = health_root / "health_suite_summary.csv"
    summary_md = health_root / "health_suite_summary.md"

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "status",
                "temp_selection",
                "humidity_selection",
                "future_mean_rr",
                "future_high_risk_share",
                "future_mean_heat_index_c",
                "delta_mean_rr",
                "output_dir",
                "message",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "model": r["model"],
                    "status": r["status"],
                    "temp_selection": r["temp_selection"],
                    "humidity_selection": r["humidity_selection"],
                    "future_mean_rr": r["future_mean_rr"],
                    "future_high_risk_share": r["future_high_risk_share"],
                    "future_mean_heat_index_c": r["future_mean_heat_index_c"],
                    "delta_mean_rr": r["delta_mean_rr"],
                    "output_dir": r["output_dir"],
                    "message": r["message"],
                }
            )

    lines = [
        "# Health Suite Summary",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Models requested: `{', '.join(models)}`",
        f"- OK: `{', '.join(ok_models) if ok_models else '-'}`",
        f"- Skipped: `{', '.join(skipped_models) if skipped_models else '-'}`",
        f"- Failed: `{', '.join(failed_models) if failed_models else '-'}`",
        "",
        "## Model Results",
    ]
    for r in results:
        lines.append(
            f"- `{r.model}` | {r.status} | rr={r.future_mean_rr} | hi_share={r.future_high_risk_share} | "
            f"temp_sel={r.temp_selection} | hum_sel={r.humidity_selection} | out={r.output_dir}"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_md}")
    if failed_models:
        print(f"Failed models: {', '.join(failed_models)}")
    if skipped_models:
        print(f"Skipped models: {', '.join(skipped_models)}")


if __name__ == "__main__":
    main()
