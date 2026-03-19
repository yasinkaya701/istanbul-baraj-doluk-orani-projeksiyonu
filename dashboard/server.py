#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import mimetypes
import zipfile
import re
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
DEFAULT_OUTPUT_ROOT = ROOT / "output"

SUPPORTED_MODELS = [
    "quant",
    "prophet",
    "strong",
    "analog",
    "prophet_ultra",
    "walkforward",
    "best_meta",
    "stable_consensus",
    "literature",
]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
TEXT_SUFFIXES = {".md", ".txt", ".json", ".log"}

FEATURE_ORDER = [
    "health_suite",
    "health_metrics",
    "health_reports",
    "forecasts",
    "anomalies",
    "leaderboards",
    "indexes",
    "reports",
    "datasets",
    "charts",
    "components",
    "parquet",
    "notes",
]

FEATURE_META = {
    "health_suite": {"title": "Saglik Suite Ozeti", "type": "text"},
    "health_metrics": {"title": "Saglik Metrikleri", "type": "csv"},
    "health_reports": {"title": "Saglik Raporlari", "type": "text"},
    "forecasts": {"title": "Forecast CSV", "type": "csv"},
    "anomalies": {"title": "Anomaly CSV", "type": "csv"},
    "leaderboards": {"title": "Leaderboard CSV", "type": "csv"},
    "indexes": {"title": "Index Dosyaları", "type": "csv"},
    "reports": {"title": "Raporlar", "type": "text"},
    "datasets": {"title": "Dataset CSV", "type": "csv"},
    "charts": {"title": "Grafikler", "type": "image"},
    "components": {"title": "Bilesen Grafikleri", "type": "image"},
    "parquet": {"title": "Parquet Dosyalari", "type": "file"},
    "notes": {"title": "Ek Notlar", "type": "text"},
}

MAX_PREVIEW_ROWS = 8
MAX_PREVIEW_FILES = 8
MAX_IMAGES = 24
MAX_SNIPPET_CHARS = 1000
MAX_SCAN_DEPTH = 8
MAX_PRESENTATION_CHARTS = 12
PRESENTATION_MODES = ("balanced", "health", "performance")
ANOMALY_NEWS_CATALOG = (
    DEFAULT_OUTPUT_ROOT / "extreme_events" / "news_expanded_v3_relaxed" / "top2_per_event_summary_hybrid_unique_sources.csv"
)
ANOMALY_NEWS_ARCHIVE_CSV = (
    DEFAULT_OUTPUT_ROOT / "extreme_events" / "news_expanded_v3_relaxed" / "meteoroloji_haber_baslik_katalogu.csv"
)
ANOMALY_NEWS_ENRICHED_CSV = (
    DEFAULT_OUTPUT_ROOT / "extreme_events" / "news_expanded_v3_relaxed" / "tum_asiri_olaylar_haber_enriched.csv"
)
LATEST_DAILY_CSV = DEFAULT_OUTPUT_ROOT / "spreadsheet" / "es_ea_newdata_daily.csv"
VARIABLE_ORDER = ["temp", "humidity", "pressure", "precip"]
MATCH_KIND_PRIORITY = {
    "exact_event": 0,
    "exact_day": 1,
    "seasonal_archive": 2,
    "analog_event": 3,
    "archive_catalog": 4,
}
MATCH_KIND_LABELS = {
    "exact_event": "Dogrudan eslesme",
    "exact_day": "Ayni gun eslesmesi",
    "analog_event": "Benzer olay",
    "seasonal_archive": "Ayni sezon arsivi",
    "archive_catalog": "Arsiv baglami",
}
MATCH_BUCKET_LABELS = {
    "direct": "Dogrudan",
    "analog": "Benzer olay",
    "archive": "Arsiv baglami",
    "none": "Eslesme yok",
}
ANOMALY_NEWS_CLIMATE_FIELDS = [
    "t_max_c",
    "t_min_c",
    "t_mean_c",
    "rh_mean_pct",
    "rh_max_pct",
    "rh_min_pct",
    "es_tmax_kpa",
    "es_tmin_kpa",
    "es_kpa",
    "ea_kpa",
    "es_minus_ea_kpa",
    "vpd_kpa",
]
ANOMALY_NEWS_TEXT_FIELDS = ["source", "source_temp", "source_humidity", "ea_formula"]


def _relative_posix(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _safe_resolve(base: Path, relative: str) -> Path:
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise FileNotFoundError("Path escapes base directory") from exc
    return candidate


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(v) for v in value]
    return value


def _format_ts(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def _csv_preview(path: Path, max_rows: int = MAX_PREVIEW_ROWS) -> dict:
    rows: list[list[str]] = []
    columns: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            columns = next(reader)
        except StopIteration:
            return {"columns": [], "rows": []}
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            rows.append(row)
    return {"columns": columns, "rows": rows}


def _text_snippet(path: Path, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _has_artifacts(run_dir: Path) -> bool:
    markers = {"charts", "forecasts", "reports", "leaderboards", "anomalies", "datasets", "components"}
    for item in run_dir.rglob("*"):
        if not item.is_dir():
            continue
        try:
            rel = item.relative_to(run_dir)
        except ValueError:
            continue
        if len(rel.parts) <= 3 and item.name.lower() in markers:
            return True
    return False


def _has_health_suite(run_dir: Path) -> bool:
    return (run_dir / "health" / "health_suite_summary.json").is_file()


def _find_runs(output_root: Path) -> list[dict]:
    runs: list[dict] = []
    summary_dirs: set[Path] = set()

    for summary_path in output_root.rglob("model_suite_summary.json"):
        try:
            data = _json_load(summary_path)
        except (json.JSONDecodeError, OSError):
            continue
        run_dir = summary_path.parent
        summary_dirs.add(run_dir.resolve())
        runs.append(
            {
                "id": _relative_posix(run_dir, ROOT),
                "summary_path": _relative_posix(summary_path, ROOT),
                "updated_at": _format_ts(summary_path),
                "models_requested": data.get("models_requested", []),
                "models_ok": data.get("models_ok", []),
                "models_failed": data.get("models_failed", []),
                "observations_used": data.get("observations_used", ""),
                "health_available": _has_health_suite(run_dir),
                "kind": "suite",
            }
        )

    for candidate in output_root.iterdir():
        if not candidate.is_dir():
            continue
        if candidate.resolve() in summary_dirs:
            continue
        if not _has_artifacts(candidate):
            continue
        runs.append(
            {
                "id": _relative_posix(candidate, ROOT),
                "summary_path": "",
                "updated_at": _format_ts(candidate),
                "models_requested": [],
                "models_ok": [],
                "models_failed": [],
                "observations_used": "",
                "health_available": _has_health_suite(candidate),
                "kind": "artifact",
            }
        )

    runs.sort(key=lambda item: item["updated_at"], reverse=True)
    return runs


def _scan_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in run_dir.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(run_dir)
        except ValueError:
            continue
        if len(rel.parts) <= MAX_SCAN_DEPTH:
            files.append(p)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _is_quant_chart_path(run_dir: Path, path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    rel = path.relative_to(run_dir)
    parts = [x.lower() for x in rel.parts]
    name = path.name.lower()
    return "charts" in parts and "regime_probs" not in name and "monthly_quant" in name


def _feature_keys_for_file(run_dir: Path, path: Path) -> list[str]:
    rel = path.relative_to(run_dir)
    parts = [x.lower() for x in rel.parts]
    name = path.name.lower()
    suffix = path.suffix.lower()

    keys: set[str] = set()

    if "forecasts" in parts and suffix == ".csv":
        keys.add("forecasts")
    if "anomalies" in parts and suffix == ".csv":
        keys.add("anomalies")
    if ("leaderboards" in parts or "leaderboard" in name) and suffix == ".csv":
        keys.add("leaderboards")
    if "index" in name and suffix in {".csv", ".parquet", ".json", ".md"}:
        keys.add("indexes")
    if "reports" in parts and suffix in {".json", ".md", ".txt", ".csv"}:
        keys.add("reports")
    if "datasets" in parts and suffix == ".csv":
        keys.add("datasets")
    if _is_quant_chart_path(run_dir, path):
        keys.add("charts")
    if "components" in parts and suffix in IMAGE_SUFFIXES:
        keys.add("components")
    if suffix == ".parquet":
        keys.add("parquet")
    if suffix in TEXT_SUFFIXES and "reports" not in parts and ("note" in name or "events" in name):
        keys.add("notes")
    if "health" in parts:
        if "health_suite_summary" in name:
            keys.add("health_suite")
        if suffix == ".csv" and any(
            tok in name
            for tok in [
                "health_monthly_metrics",
                "health_annual_summary",
                "health_period_comparison",
                "health_monthly_anomalies_vs_baseline",
                "health_suite_summary",
            ]
        ):
            keys.add("health_metrics")
        if suffix in {".md", ".json", ".txt"} and any(
            tok in name
            for tok in [
                "health_impact_report",
                "health_impact_summary",
                "health_suite_summary",
                "sensitivity_summary",
                "dlnm",
            ]
        ):
            keys.add("health_reports")

    return sorted(keys)


def _gather_chart_groups(run_dir: Path) -> list[dict]:
    chart_dirs = [
        d for d in run_dir.rglob("charts") if d.is_dir() and len(d.relative_to(run_dir).parts) <= 4
    ]
    chart_dirs.sort(key=lambda p: _relative_posix(p, run_dir))
    groups: list[dict] = []
    for chart_dir in chart_dirs:
        images = sorted(
            [p for p in chart_dir.iterdir() if p.is_file() and _is_quant_chart_path(run_dir, p)]
        )[:MAX_IMAGES]
        if not images:
            continue
        group_name = "/".join(chart_dir.relative_to(run_dir).parts[:-1]) or "charts"
        groups.append(
            {
                "name": group_name,
                "images": [_relative_posix(img, ROOT) for img in images],
            }
        )
    return groups


def _gather_csv_previews(run_dir: Path, pattern: str) -> list[dict]:
    csv_files = [
        p
        for p in run_dir.rglob(pattern)
        if p.is_file() and len(p.relative_to(run_dir).parts) <= 5 and p.suffix.lower() == ".csv"
    ]
    csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    previews: list[dict] = []
    for csv_path in csv_files[:MAX_PREVIEW_FILES]:
        preview = _csv_preview(csv_path)
        previews.append(
            {
                "path": _relative_posix(csv_path, ROOT),
                "columns": preview["columns"],
                "rows": preview["rows"],
            }
        )
    return previews


def _build_feature_views(run_dir: Path) -> tuple[list[dict], dict[str, dict], int]:
    files = _scan_files(run_dir)
    buckets: dict[str, list[Path]] = {k: [] for k in FEATURE_ORDER}

    for path in files:
        for key in _feature_keys_for_file(run_dir, path):
            if key in buckets:
                buckets[key].append(path)

    inventory: list[dict] = []
    feature_data: dict[str, dict] = {}

    for key in FEATURE_ORDER:
        meta = FEATURE_META[key]
        paths = buckets[key]
        inventory.append(
            {
                "key": key,
                "title": meta["title"],
                "count": len(paths),
                "samples": [_relative_posix(p, ROOT) for p in paths[:4]],
            }
        )

        type_name = meta["type"]
        items: list[dict] = []

        if type_name == "csv":
            for p in paths[:MAX_PREVIEW_FILES]:
                if p.suffix.lower() != ".csv":
                    items.append({"path": _relative_posix(p, ROOT), "columns": [], "rows": []})
                    continue
                preview = _csv_preview(p)
                items.append(
                    {
                        "path": _relative_posix(p, ROOT),
                        "columns": preview["columns"],
                        "rows": preview["rows"],
                    }
                )
        elif type_name == "image":
            items = [{"path": _relative_posix(p, ROOT)} for p in paths[:MAX_IMAGES]]
        elif type_name == "text":
            items = [
                {"path": _relative_posix(p, ROOT), "snippet": _text_snippet(p)}
                for p in paths[:MAX_PREVIEW_FILES]
            ]
        else:
            items = [{"path": _relative_posix(p, ROOT)} for p in paths[:MAX_PREVIEW_FILES]]

        feature_data[key] = {"type": type_name, "title": meta["title"], "items": items}

    return inventory, feature_data, len(files)


def _build_artifact_summary(run_dir: Path) -> dict:
    discovered = [m for m in SUPPORTED_MODELS if (run_dir / m).is_dir()]
    results = [
        {
            "name": model,
            "ok": None,
            "returncode": None,
            "output_dir": str(run_dir / model),
            "command": "",
            "stdout_tail": "",
            "stderr_tail": "",
        }
        for model in discovered
    ]
    return {
        "observations_used": "",
        "models_requested": discovered,
        "models_ok": [],
        "models_failed": [],
        "results": results,
    }


def _build_model_status(run_dir: Path, summary: dict) -> dict[str, dict]:
    requested = {str(x).lower() for x in summary.get("models_requested", [])}
    ok = {str(x).lower() for x in summary.get("models_ok", [])}
    failed = {str(x).lower() for x in summary.get("models_failed", [])}

    by_result: dict[str, dict] = {}
    for item in summary.get("results", []):
        name = str(item.get("name", "")).lower()
        if not name:
            continue
        by_result[name] = {
            "output_dir": item.get("output_dir", ""),
            "returncode": item.get("returncode"),
        }

    status: dict[str, dict] = {}
    for model in SUPPORTED_MODELS:
        model_dir = run_dir / model
        present = model_dir.is_dir()
        if model in failed:
            state = "failed"
        elif model in ok:
            state = "ok"
        elif model in requested:
            state = "requested"
        elif present:
            state = "artifact"
        else:
            state = "not_requested"

        info = by_result.get(model, {})
        output_dir = info.get("output_dir") or (_relative_posix(model_dir, ROOT) if present else "")
        status[model] = {
            "status": state,
            "present": present,
            "output_dir": output_dir,
            "returncode": info.get("returncode"),
        }
    return status


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _load_health_suite(run_dir: Path) -> dict:
    summary_path = run_dir / "health" / "health_suite_summary.json"
    if not summary_path.exists():
        return {
            "available": False,
            "path": "",
            "models_requested": [],
            "models_ok": [],
            "models_skipped": [],
            "models_failed": [],
            "future_start": None,
            "future_end": None,
            "top_risk_model": "",
            "top_risk_rr": None,
            "rows": [],
        }

    try:
        data = _json_load(summary_path)
    except (json.JSONDecodeError, OSError):
        return {
            "available": False,
            "path": _relative_posix(summary_path, ROOT),
            "models_requested": [],
            "models_ok": [],
            "models_skipped": [],
            "models_failed": [],
            "future_start": None,
            "future_end": None,
            "top_risk_model": "",
            "top_risk_rr": None,
            "rows": [],
        }

    rows: list[dict] = []
    top_model = ""
    top_rr: float | None = None

    for item in data.get("results", []):
        rr = _to_float_or_none(item.get("future_mean_rr"))
        high_share = _to_float_or_none(item.get("future_high_risk_share"))
        heat_idx = _to_float_or_none(item.get("future_mean_heat_index_c"))
        delta_rr = _to_float_or_none(item.get("delta_mean_rr"))
        row = {
            "model": item.get("model", ""),
            "status": item.get("status", ""),
            "future_mean_rr": rr,
            "future_high_risk_share": high_share,
            "future_mean_heat_index_c": heat_idx,
            "delta_mean_rr": delta_rr,
            "message": item.get("message", ""),
            "output_dir": item.get("output_dir", ""),
        }
        rows.append(row)
        if row["status"] == "ok" and rr is not None and (top_rr is None or rr > top_rr):
            top_rr = rr
            top_model = row["model"]

    return {
        "available": True,
        "path": _relative_posix(summary_path, ROOT),
        "models_requested": data.get("models_requested", []),
        "models_ok": data.get("models_ok", []),
        "models_skipped": data.get("models_skipped", []),
        "models_failed": data.get("models_failed", []),
        "future_start": data.get("future_start"),
        "future_end": data.get("future_end"),
        "top_risk_model": top_model,
        "top_risk_rr": top_rr,
        "rows": rows,
    }


def _merge_health_into_summary(summary: dict, health_suite: dict, run_dir: Path) -> dict:
    existing = summary.get("health_suite")
    if isinstance(existing, dict) and existing:
        return summary
    if not health_suite.get("available"):
        return summary

    merged = dict(summary)
    merged["health_suite"] = {
        "enabled": True,
        "ok": len(health_suite.get("models_failed", [])) == 0,
        "output_dir": _relative_posix(run_dir / "health", ROOT),
        "summary": {
            "models_requested": health_suite.get("models_requested", []),
            "models_ok": health_suite.get("models_ok", []),
            "models_skipped": health_suite.get("models_skipped", []),
            "models_failed": health_suite.get("models_failed", []),
            "future_start": health_suite.get("future_start"),
            "future_end": health_suite.get("future_end"),
        },
        "source": "dashboard_backfill",
    }
    return merged


def _build_presentation_note(
    run_id: str,
    updated_at: str,
    summary: dict,
    health_suite: dict,
    total_files: int,
    active_features: int,
) -> str:
    requested = summary.get("models_requested", [])
    ok = summary.get("models_ok", [])
    failed = summary.get("models_failed", [])
    stabilization = summary.get("stabilization", {}) if isinstance(summary.get("stabilization"), dict) else {}
    robust = summary.get("robust_selection", {}) if isinstance(summary.get("robust_selection"), dict) else {}
    robust_summary = robust.get("summary", {}) if isinstance(robust.get("summary"), dict) else {}
    requested_count = len(requested)
    ok_count = len(ok)
    fail_count = len(failed)
    success_rate = (ok_count / requested_count * 100.0) if requested_count else 0.0

    lines = [
        "RESMI SUNUM NOTU",
        f"Rapor Tarihi: {updated_at}",
        f"Kosu: {run_id}",
        "",
        "1) Model Kapsami ve Performans",
        (
            f"- Bu kosuda {requested_count} model talep edilmis, {ok_count} model basarili "
            f"ve {fail_count} model basarisizdir. Basari orani %{success_rate:.1f}."
        ),
        f"- Toplam {total_files} dosya ve {active_features} aktif feature kategorisi uretilmistir.",
        "",
        "2) Entegrasyon Durumu",
        "- Model suite ile health suite tek akista entegre sekilde calismaktadir.",
    ]
    if stabilization:
        stab_status = str(stabilization.get("status", "-"))
        stab_strategy = str(stabilization.get("selected_strategy", "-"))
        stab_ratio = _to_float_or_none(stabilization.get("recent_ratio"))
        ratio_txt = f"{stab_ratio * 100.0:.1f}%" if stab_ratio is not None else "-"
        lines.append(
            f"- Tahmin stabilizasyonu: durum={stab_status}, strateji={stab_strategy}, recent oran={ratio_txt}."
        )
    if robust and robust.get("enabled") is not False:
        robust_ok = bool(robust.get("ok"))
        selected_models = robust_summary.get("selected_models", [])
        if isinstance(selected_models, list):
            compact = ", ".join(
                f"{str(x.get('variable', '-'))}:{str(x.get('model_key', '-'))}({str(x.get('confidence_grade', '-'))})"
                for x in selected_models[:4]
                if isinstance(x, dict)
            )
            more = f" (+{len(selected_models) - 4})" if len(selected_models) > 4 else ""
        else:
            compact = ""
            more = ""
        lines.append(
            f"- Robust model secimi: {'basarili' if robust_ok else 'hatali'}; secilen set: {compact or '-'}{more}."
        )

    if health_suite.get("available"):
        h_req = len(health_suite.get("models_requested", []))
        h_ok = len(health_suite.get("models_ok", []))
        h_skip = len(health_suite.get("models_skipped", []))
        h_fail = len(health_suite.get("models_failed", []))
        top_model = str(health_suite.get("top_risk_model") or "-")
        top_rr = _to_float_or_none(health_suite.get("top_risk_rr"))
        top_rr_text = f"{top_rr:.3f}" if top_rr is not None else "-"
        fut_start = health_suite.get("future_start")
        fut_end = health_suite.get("future_end")
        period = f"{fut_start}-{fut_end}" if fut_start and fut_end else "-"
        lines.extend(
            [
                "",
                "3) Insan Sagligi Etki Ozeti",
                (
                    f"- Saglik analizinde {h_req} model degerlendirilmis; {h_ok} basarili, "
                    f"{h_skip} atlanan, {h_fail} basarisiz sonuc vardir."
                ),
                f"- Future analiz donemi: {period}.",
                f"- En yuksek goreli risk gosteren model: {top_model} (Future Mean RR: {top_rr_text}).",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "3) Insan Sagligi Etki Ozeti",
                "- Bu kosuda saglik suite ozeti bulunmamistir; health analizi tekrar kosulmalidir.",
            ]
        )

    if fail_count > 0:
        lines.extend(
            [
                "",
                "4) Risk ve Aksiyon",
                f"- Basarisiz modeller: {', '.join(str(x) for x in failed)}.",
                "- Aksiyon: ilgili modeller icin tekrar kosu ve log incelemesi onerilir.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "4) Risk ve Aksiyon",
                "- Kritik model hatasi raporlanmamistir; mevcut paket sunum ve karar destegi icin uygundur.",
            ]
        )
    return "\n".join(lines) + "\n"


def _normalize_presentation_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in PRESENTATION_MODES:
        return "balanced"
    return normalized


def _presentation_chart_score(path: Path, run_dir: Path, mode: str = "balanced") -> float:
    mode = _normalize_presentation_mode(mode)
    rel = path.relative_to(run_dir)
    parts = [x.lower() for x in rel.parts]
    name = path.name.lower()
    score = 0.0

    model_weight = {
        "best_meta": 30.0,
        "quant": 20.0,
        "prophet_ultra": 18.0,
        "strong": 16.0,
        "prophet": 14.0,
        "analog": 12.0,
        "walkforward": 10.0,
        "literature": 8.0,
    }
    for model, weight in model_weight.items():
        if model in parts:
            score += weight
            break

    if "health" in parts:
        score += 55.0
    if "charts" in parts:
        score += 24.0
    if "monthly_quant" in name:
        score += 35.0
    if "annual_compare" in name:
        score += 20.0
    if "regime_probs" in name:
        score -= 25.0
    if "diagnostics" in name:
        score -= 10.0
    if "components" in parts:
        score -= 8.0

    for token, weight in [("temp", 8.0), ("humidity", 8.0), ("pressure", 6.0), ("precip", 6.0)]:
        if token in name:
            score += weight
            break

    if mode == "health":
        if "health" in parts:
            score += 45.0
        if "temp" in name or "humidity" in name:
            score += 18.0
        if "best_meta" in parts:
            score += 10.0
        if "diagnostics" in name:
            score -= 6.0
    elif mode == "performance":
        if "best_meta" in parts:
            score += 30.0
        if "annual_compare" in name:
            score += 24.0
        if "health" in parts:
            score -= 22.0
        if "regime_probs" in name:
            score += 6.0

    return score


def _select_presentation_chart_files(
    run_dir: Path,
    max_items: int = MAX_PRESENTATION_CHARTS,
    mode: str = "balanced",
) -> list[Path]:
    mode = _normalize_presentation_mode(mode)
    images = [
        p
        for p in run_dir.rglob("*")
        if p.is_file()
        and _is_quant_chart_path(run_dir, p)
        and len(p.relative_to(run_dir).parts) <= 6
    ]
    if not images:
        return []

    ranked = sorted(
        images,
        key=lambda p: (_presentation_chart_score(p, run_dir, mode=mode), p.stat().st_mtime),
        reverse=True,
    )

    selected: list[Path] = []
    selected_set: set[Path] = set()
    used_groups: set[str] = set()

    # First pass: diversify by model/group.
    for p in ranked:
        rel = p.relative_to(run_dir)
        group = rel.parts[0].lower() if rel.parts else "misc"
        if mode != "health" and group in used_groups:
            continue
        selected.append(p)
        selected_set.add(p)
        used_groups.add(group)
        if len(selected) >= max_items:
            return selected

    # Second pass: fill the remaining slots by rank.
    for p in ranked:
        if p in selected_set:
            continue
        selected.append(p)
        selected_set.add(p)
        if len(selected) >= max_items:
            break
    return selected


def _build_presentation_assets(
    run_dir: Path,
    max_items: int = MAX_PRESENTATION_CHARTS,
    mode: str = "balanced",
) -> list[dict]:
    mode = _normalize_presentation_mode(mode)
    assets: list[dict] = []
    for path in _select_presentation_chart_files(run_dir, max_items=max_items, mode=mode):
        rel = path.relative_to(run_dir)
        parts = [x.lower() for x in rel.parts]
        if "health" in parts:
            kind = "health"
        elif "components" in parts:
            kind = "component"
        else:
            kind = "chart"
        assets.append(
            {
                "path": _relative_posix(path, ROOT),
                "label": path.name,
                "group": rel.parts[0] if rel.parts else "misc",
                "kind": kind,
                "mode": mode,
            }
        )
    return assets


def _pick_assets_for_mode(detail: dict, mode: str) -> list[dict]:
    mode = _normalize_presentation_mode(mode)
    by_mode = detail.get("presentation_assets_by_mode")
    if isinstance(by_mode, dict):
        selected = by_mode.get(mode)
        if isinstance(selected, list):
            return selected
    fallback = detail.get("presentation_assets")
    if isinstance(fallback, list):
        return fallback
    return []


def _wrap_text(text: str, max_chars: int = 110) -> list[str]:
    raw = text.strip()
    if not raw:
        return [""]
    words = raw.split()
    out: list[str] = []
    cur = ""
    for word in words:
        if not cur:
            cur = word
            continue
        if len(cur) + 1 + len(word) <= max_chars:
            cur = f"{cur} {word}"
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out or [""]


def _render_note_pdf(note: str, title: str) -> bytes | None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        return None

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle(title)
    width, height = A4
    margin = 46
    y = height - margin
    line_h = 13

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, title)
    y -= 24
    c.setFont("Helvetica", 10)

    for paragraph in note.splitlines():
        wrapped = _wrap_text(paragraph, max_chars=112)
        for line in wrapped:
            if y < margin:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - margin
            c.drawString(margin, y, line)
            y -= line_h
        if y < margin:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - margin
        else:
            y -= 3

    c.save()
    return buf.getvalue()


def _build_export_pack(detail: dict, mode: str = "balanced") -> bytes:
    mode = _normalize_presentation_mode(mode)
    run_id = str(detail.get("id", "run"))
    run_slug = run_id.replace("/", "_")
    note = str(detail.get("presentation_note", "")).strip() + "\n"
    summary_blob = json.dumps(_sanitize_json_value(detail), ensure_ascii=False, indent=2) + "\n"
    note_md = (
        "# Resmi Sunum Notu\n\n"
        f"**Kosu:** `{run_id}`\n\n"
        "```text\n"
        f"{note.rstrip()}\n"
        "```\n"
    )
    assets = _pick_assets_for_mode(detail, mode)

    pdf_blob = _render_note_pdf(note, f"{run_id} Sunum Notu")
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    readme_lines = [
        "Hackhaton Dashboard Sunum Paketi",
        f"Olusturulma: {created_at}",
        f"Kosu: {run_id}",
        "",
        "Icerik:",
        f"- {run_slug}_sunum_notu.txt",
        f"- {run_slug}_sunum_notu.md",
        f"- {run_slug}_run_summary.json",
    ]
    if pdf_blob is not None:
        readme_lines.append(f"- {run_slug}_sunum_notu.pdf")
    else:
        readme_lines.append("- PDF notu eklenemedi (reportlab yok).")
    readme_lines.append(f"- Grafik modu: {mode}")
    readme_lines.append(f"- Secili sunum grafikleri: {len(assets)} dosya (charts/ klasoru)")

    pack_buf = io.BytesIO()
    with zipfile.ZipFile(pack_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", "\n".join(readme_lines) + "\n")
        zf.writestr(f"{run_slug}_sunum_notu.txt", note)
        zf.writestr(f"{run_slug}_sunum_notu.md", note_md)
        zf.writestr(f"{run_slug}_run_summary.json", summary_blob)
        if pdf_blob is not None:
            zf.writestr(f"{run_slug}_sunum_notu.pdf", pdf_blob)
        for idx, asset in enumerate(assets, start=1):
            src = _safe_resolve(ROOT, str(asset.get("path", "")))
            if not src.is_file():
                continue
            clean_name = src.name
            zf.writestr(f"charts/{idx:02d}_{clean_name}", src.read_bytes())
    return pack_buf.getvalue()


def _build_chart_pack(detail: dict, mode: str = "balanced") -> bytes:
    mode = _normalize_presentation_mode(mode)
    run_id = str(detail.get("id", "run"))
    assets = _pick_assets_for_mode(detail, mode)
    pack_buf = io.BytesIO()
    with zipfile.ZipFile(pack_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        readme = [
            "Sunum Grafik Paketi",
            f"Kosu: {run_id}",
            f"Mod: {mode}",
            f"Toplam grafik: {len(assets)}",
            "",
            "Dosyalar charts/ altina yazilmistir.",
        ]
        zf.writestr("README.txt", "\n".join(readme) + "\n")
        for idx, asset in enumerate(assets, start=1):
            src = _safe_resolve(ROOT, str(asset.get("path", "")))
            if not src.is_file():
                continue
            zf.writestr(f"charts/{idx:02d}_{src.name}", src.read_bytes())
    return pack_buf.getvalue()


def _read_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _split_pipe_text(value: Any) -> list[str]:
    if value is None:
        return []
    out: list[str] = []
    for part in str(value).split("|"):
        cleaned = part.strip()
        if cleaned:
            out.append(cleaned)
    return out


def _resolve_anomaly_bundle(run_dir: Path) -> dict[str, Path]:
    candidates = [run_dir / "extreme_events", run_dir]
    for base_dir in candidates:
        unique_csv = base_dir / "anomaly_day_data" / "anomaly_unique_days_with_daily_climate.csv"
        filtered_csv = base_dir / "tum_asiri_olaylar_bilimsel_filtreli.csv"
        points_csv = base_dir / "anomaly_day_data" / "anomaly_points_with_daily_climate.csv"
        missing_csv = base_dir / "anomaly_day_data" / "anomaly_days_missing_daily_climate.csv"
        if unique_csv.is_file() and filtered_csv.is_file():
            return {
                "base_dir": base_dir,
                "unique_csv": unique_csv,
                "filtered_csv": filtered_csv,
                "points_csv": points_csv,
                "missing_csv": missing_csv,
            }
    raise FileNotFoundError("Anomaly-day bundle not found for run")


def _latest_daily_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    last_row: dict[str, str] | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            last_row = row
    if not last_row:
        return {}

    snapshot: dict[str, Any] = {
        "date": str(last_row.get("date", "")).strip(),
        "source": str(last_row.get("source", "")).strip(),
        "sourceTemp": str(last_row.get("source_temp", "")).strip(),
        "sourceHumidity": str(last_row.get("source_humidity", "")).strip(),
        "eaFormula": str(last_row.get("ea_formula", "")).strip(),
        "autoObsCount": _safe_float(last_row.get("auto_obs_count")),
    }
    for field in ANOMALY_NEWS_CLIMATE_FIELDS:
        snapshot[field] = _safe_float(last_row.get(field))
    return snapshot


def _find_anomaly_runs(output_root: Path) -> list[dict]:
    runs: list[dict] = []
    seen: set[str] = set()
    latest_symlink = output_root / "yeni_model_newdata_latest"
    latest_target: Path | None = None
    if latest_symlink.is_symlink():
        try:
            latest_target = latest_symlink.resolve(strict=False)
        except OSError:
            latest_target = None
    elif latest_symlink.exists():
        try:
            latest_target = latest_symlink.resolve()
        except OSError:
            latest_target = None

    for unique_csv in output_root.rglob("anomaly_unique_days_with_daily_climate.csv"):
        if unique_csv.parent.name != "anomaly_day_data":
            continue
        base_dir = unique_csv.parent.parent
        if base_dir.name == "extreme_events" and (base_dir.parent / "reports").is_dir():
            run_dir = base_dir.parent
        else:
            run_dir = base_dir
        run_id = _relative_posix(run_dir, ROOT)
        if run_id in seen:
            continue
        seen.add(run_id)
        try:
            bundle = _resolve_anomaly_bundle(run_dir)
        except FileNotFoundError:
            continue
        runs.append(
            {
                "id": run_id,
                "name": run_dir.name,
                "_is_latest": bool(latest_target and run_dir.resolve() == latest_target),
                "updated_at": _format_ts(bundle["unique_csv"]),
                "day_count": _count_csv_rows(bundle["unique_csv"]),
                "event_count": _count_csv_rows(bundle["filtered_csv"]),
                "bundle_root": _relative_posix(bundle["base_dir"], ROOT),
            }
        )

    runs.sort(key=lambda item: item["updated_at"], reverse=True)
    if runs and not any(item.get("_is_latest") for item in runs):
        runs[0]["_is_latest"] = True
    for item in runs:
        item["label"] = f"{item['name']} | latest" if item.pop("_is_latest", False) else item["name"]
        item.pop("name", None)
    return runs


def _safe_iso_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    text = text.split(" ")[0]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _seasonal_day_gap(lhs: date | None, rhs: date | None) -> int | None:
    if not lhs or not rhs:
        return None
    lhs_day = min(lhs.timetuple().tm_yday, 365)
    rhs_day = min(rhs.timetuple().tm_yday, 365)
    gap = abs(lhs_day - rhs_day)
    return min(gap, 365 - gap)


def _normalize_text(value: Any) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(value or ""))
    return " ".join(part for part in cleaned.split() if part)


def _match_bucket(kind: str) -> str:
    if kind in {"exact_event", "exact_day"}:
        return "direct"
    if kind == "analog_event":
        return "analog"
    if kind in {"seasonal_archive", "archive_catalog"}:
        return "archive"
    return "none"


def _hazard_preferences(
    variable: str,
    direction: str,
    reason_tags: list[str] | None = None,
    quant_cause: str = "",
) -> list[str]:
    var = str(variable).strip().lower()
    direction_text = str(direction).strip().lower()
    tags = {tag.strip().lower() for tag in (reason_tags or []) if tag}
    cause = _normalize_text(quant_cause)
    dry_bias = direction_text == "dusuk" or "low_tail" in tags or "acigi" in cause or "kurak" in cause
    wet_bias = direction_text in {"sicrama", "yuksek"} or "high_tail" in tags or "yagis" in cause

    if var == "precip":
        if dry_bias:
            return ["drought", "water_shortage", "dry_spell"]
        return ["flood", "heavy_rain", "flood_warning", "flood_damage", "hail", "storm"]
    if var == "humidity":
        if dry_bias:
            return ["drought", "heatwave", "water_shortage"]
        return ["heavy_rain", "flood", "storm", "hail"]
    if var == "pressure":
        if "blokaj" in cause or direction_text == "yuksek":
            return ["drought", "heatwave", "wildfire", "water_shortage"]
        if dry_bias:
            return ["drought", "water_shortage", "heatwave"]
        return ["storm", "heavy_rain", "flood", "hail"]
    if var == "temp":
        if direction_text == "dusuk":
            return ["coldwave", "snowstorm", "frost"]
        if dry_bias:
            return ["drought", "heatwave", "wildfire", "water_shortage"]
        if wet_bias:
            return ["heatwave", "drought", "wildfire"]
        return ["heatwave", "drought", "wildfire"]
    return ["flood", "storm", "drought", "heatwave"]


def _hazard_compatibility(hazard_type: str, preferred: list[str]) -> float:
    hazard = _normalize_text(hazard_type)
    if not hazard or not preferred:
        return 0.25
    preferred_text = [_normalize_text(item) for item in preferred if item]
    if hazard in preferred_text:
        return 1.0
    if any(item and (item in hazard or hazard in item) for item in preferred_text):
        return 0.82
    if hazard in {"flood_damage", "flood_warning"} and any(
        item in preferred_text for item in ["flood", "heavy_rain", "storm"]
    ):
        return 0.78
    if hazard in {"hail_damage", "hail_warning"} and any(item in preferred_text for item in ["hail", "storm"]):
        return 0.76
    if hazard == "attribution":
        return 0.55
    if hazard == "drought" and any(item in preferred_text for item in ["heatwave", "water_shortage", "wildfire"]):
        return 0.84
    if hazard in {"storm", "heavy_rain"} and any(item in preferred_text for item in ["flood", "hail"]):
        return 0.72
    return 0.20


def _headline_keyword_score(headline: str, hazard_type: str, variable: str, direction: str) -> float:
    haystack = _normalize_text(f"{headline} {hazard_type}")
    var = str(variable).strip().lower()
    direction_text = str(direction).strip().lower()
    if var == "temp":
        keywords = ["kar", "soguk", "don", "buz", "kis"] if direction_text == "dusuk" else ["sicak", "kurak", "susuz", "yangin", "bunalt"]
    elif var == "pressure":
        keywords = ["kurak", "susuz", "baraj", "su"] if direction_text == "yuksek" else ["firtina", "saganak", "yagis", "sel"]
    elif var == "humidity":
        keywords = ["kurak", "sicak", "susuz"] if direction_text == "dusuk" else ["nem", "yagis", "saganak", "sel", "dolu"]
    else:
        keywords = ["kurak", "susuz", "baraj", "su"] if direction_text == "dusuk" else ["sel", "yagis", "saganak", "dolu", "firtina"]

    hits = sum(1 for keyword in keywords if keyword and keyword in haystack)
    return min(1.0, hits / 2.0)


def _variable_similarity(target_variable: str, candidate_variable: str, hazard_type: str, preferred: list[str]) -> float:
    target = str(target_variable).strip().lower()
    candidate = str(candidate_variable).strip().lower()
    if target == candidate:
        return 1.0
    if {target, candidate} <= {"precip", "humidity"}:
        return 0.92
    hazard = _normalize_text(hazard_type)
    if target == "temp" and hazard in {"drought", "heatwave", "wildfire", "water_shortage"}:
        return 0.82
    if target == "pressure" and hazard in {"drought", "heavy_rain", "flood", "storm", "water_shortage"}:
        return 0.78
    if target == "humidity" and hazard in {"heavy_rain", "flood", "storm", "hail"}:
        return 0.80
    if target == "precip" and hazard in {"heavy_rain", "flood", "storm", "hail"}:
        return 0.86
    if _hazard_compatibility(hazard, preferred) >= 0.75:
        return 0.70
    return 0.35


def _seasonal_proximity_score(day_gap: int | None) -> float:
    if day_gap is None:
        return 0.25
    return float(math.exp(-abs(day_gap) / 40.0))


def _time_proximity_score(day_gap: int | None) -> float:
    if day_gap is None:
        return 0.25
    return float(math.exp(-abs(day_gap) / 45.0))


def _severity_similarity(current: float | None, candidate: float | None) -> float:
    if current is None or candidate is None:
        return 0.55
    lhs = math.log1p(abs(float(current)))
    rhs = math.log1p(abs(float(candidate)))
    return float(math.exp(-abs(lhs - rhs) / 0.9))


def _distance_rank(value: int | None) -> int:
    return abs(int(value)) if value is not None else 999


def _news_rank(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        MATCH_KIND_PRIORITY.get(str(item.get("kind", "")).strip(), 99),
        -float(item.get("score") or 0.0),
        _distance_rank(item.get("seasonalDayDiff")),
        _distance_rank(item.get("dayDiff")),
        _distance_rank(item.get("yearGap")),
    )


def _news_key(item: dict[str, Any]) -> str:
    return str(item.get("url") or f"{item.get('source','')}|{item.get('headline','')}|{item.get('date','')}").strip()


def _merge_news_items(items: list[dict[str, Any]], max_items: int = 4) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in items:
        if not str(item.get("headline", "")).strip():
            continue
        key = _news_key(item)
        existing = deduped.get(key)
        if existing is None or _news_rank(item) < _news_rank(existing):
            deduped[key] = item
    return sorted(deduped.values(), key=_news_rank)[:max_items]


def _build_summary_news_items(row: dict[str, str], match_kind: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    event_day = _safe_iso_date(row.get("event_day"))
    match_reason = "Ayni event id" if match_kind == "exact_event" else "Ayni gun ve degisken"
    for idx in (1, 2):
        headline = str(row.get(f"news_{idx}_headline", "")).strip()
        if not headline:
            continue
        news_date = _safe_iso_date(row.get(f"news_{idx}_date"))
        year_gap = abs(event_day.year - news_date.year) if event_day and news_date else None
        items.append(
            {
                "date": str(row.get(f"news_{idx}_date", "")).strip(),
                "source": str(row.get(f"news_{idx}_source", "")).strip(),
                "headline": headline,
                "url": str(row.get(f"news_{idx}_url", "")).strip(),
                "score": _safe_float(row.get(f"news_{idx}_score")),
                "dayDiff": _safe_int(row.get(f"news_{idx}_day_diff"), default=0),
                "seasonalDayDiff": _seasonal_day_gap(event_day, news_date),
                "yearGap": year_gap,
                "kind": match_kind,
                "kindLabel": MATCH_KIND_LABELS.get(match_kind, match_kind),
                "bucket": _match_bucket(match_kind),
                "bucketLabel": MATCH_BUCKET_LABELS.get(_match_bucket(match_kind), _match_bucket(match_kind)),
                "matchReason": match_reason,
                "referenceDate": str(row.get("event_day", "")).strip(),
                "referenceVariable": str(row.get("variable", "")).strip(),
            }
        )
    return items


def _build_enriched_reference(row: dict[str, str], match_kind: str) -> dict[str, Any]:
    center_date = _safe_iso_date(row.get("center_time"))
    news_date = _safe_iso_date(row.get("top_headline_date"))
    year_gap = abs(center_date.year - news_date.year) if center_date and news_date else None
    kind = str(match_kind).strip() or "analog_event"
    bucket = _match_bucket(kind)
    return {
        "eventId": str(row.get("event_id", "")).strip(),
        "variable": str(row.get("variable", "")).strip(),
        "centerDate": center_date,
        "peakSeverity": _safe_float(row.get("peak_severity_score")),
        "scientificTier": str(row.get("scientific_tier", "")).strip(),
        "direction": str(row.get("dominant_direction", "")).strip(),
        "hazardType": str(row.get("top_headline_hazard", "")).strip(),
        "headlineDate": news_date,
        "headline": str(row.get("top_headline", "")).strip(),
        "source": str(row.get("top_headline_source", "")).strip(),
        "url": str(row.get("top_headline_url", "")).strip(),
        "score": _safe_float(row.get("top_headline_match_score")),
        "dayDiff": _safe_int(row.get("top_headline_day_diff")),
        "seasonalDayDiff": _seasonal_day_gap(center_date, news_date),
        "yearGap": year_gap,
        "kind": kind,
        "kindLabel": MATCH_KIND_LABELS.get(kind, kind),
        "bucket": bucket,
        "bucketLabel": MATCH_BUCKET_LABELS.get(bucket, bucket),
        "matchReason": "Ayni event id" if kind == "exact_event" else "Benzer haberli olay",
        "referenceDate": str(row.get("center_time", "")).split(" ")[0],
        "referenceVariable": str(row.get("variable", "")).strip(),
    }


def _score_analog_reference(
    target_date: date | None,
    target_variable: str,
    target_direction: str,
    preferred_hazards: list[str],
    target_severity: float | None,
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    hazard_score = _hazard_compatibility(str(candidate.get("hazardType", "")), preferred_hazards)
    variable_score = _variable_similarity(
        target_variable,
        str(candidate.get("variable", "")),
        str(candidate.get("hazardType", "")),
        preferred_hazards,
    )
    seasonal_gap = _seasonal_day_gap(target_date, candidate.get("centerDate"))
    seasonal_score = _seasonal_proximity_score(seasonal_gap)
    severity_score = _severity_similarity(target_severity, candidate.get("peakSeverity"))
    keyword_score = _headline_keyword_score(
        str(candidate.get("headline", "")),
        str(candidate.get("hazardType", "")),
        target_variable,
        target_direction,
    )
    if hazard_score < 0.35 and keyword_score < 0.35 and variable_score < 0.85:
        return None
    base_score = float(candidate.get("score") or 0.55)
    total_score = (
        0.30 * variable_score
        + 0.23 * hazard_score
        + 0.18 * seasonal_score
        + 0.14 * severity_score
        + 0.10 * base_score
        + 0.05 * keyword_score
    )
    if total_score < 0.46:
        return None

    year_gap = abs(target_date.year - candidate["headlineDate"].year) if target_date and candidate.get("headlineDate") else None
    return {
        "date": candidate.get("headlineDate").isoformat() if candidate.get("headlineDate") else "",
        "source": str(candidate.get("source", "")).strip(),
        "headline": str(candidate.get("headline", "")).strip(),
        "url": str(candidate.get("url", "")).strip(),
        "score": round(total_score, 4),
        "dayDiff": candidate.get("dayDiff"),
        "seasonalDayDiff": seasonal_gap,
        "yearGap": year_gap,
        "kind": "analog_event",
        "kindLabel": MATCH_KIND_LABELS["analog_event"],
        "bucket": "analog",
        "bucketLabel": MATCH_BUCKET_LABELS["analog"],
        "matchReason": (
            f"Benzer {candidate.get('variable', '-') or '-'} olayi | "
            f"{seasonal_gap if seasonal_gap is not None else '-'} sezon gun farki | "
            f"hazard {candidate.get('hazardType', '-') or '-'}"
        ),
        "referenceDate": candidate.get("centerDate").isoformat() if candidate.get("centerDate") else "",
        "referenceVariable": str(candidate.get("variable", "")).strip(),
    }


def _score_archive_reference(
    target_date: date | None,
    target_variable: str,
    target_direction: str,
    preferred_hazards: list[str],
    row: dict[str, Any],
) -> dict[str, Any] | None:
    headline_date = row.get("headlineDate")
    seasonal_gap = _seasonal_day_gap(target_date, headline_date)
    seasonal_score = _seasonal_proximity_score(seasonal_gap)
    hazard_score = _hazard_compatibility(str(row.get("hazardType", "")), preferred_hazards)
    keyword_score = _headline_keyword_score(
        str(row.get("headline", "")),
        str(row.get("hazardType", "")),
        target_variable,
        target_direction,
    )
    if hazard_score < 0.35 and keyword_score < 0.35:
        return None
    year_gap = abs(target_date.year - headline_date.year) if target_date and headline_date else None
    year_score = float(math.exp(-(year_gap or 0) / 35.0)) if year_gap is not None else 0.35
    source_prior = float(row.get("sourcePrior") or 0.75)
    total_score = 0.34 * hazard_score + 0.24 * seasonal_score + 0.20 * keyword_score + 0.12 * source_prior + 0.10 * year_score
    if total_score < 0.38:
        return None

    kind = "seasonal_archive" if seasonal_score >= 0.68 else "archive_catalog"
    bucket = _match_bucket(kind)
    return {
        "date": headline_date.isoformat() if headline_date else "",
        "source": str(row.get("source", "")).strip(),
        "headline": str(row.get("headline", "")).strip(),
        "url": str(row.get("url", "")).strip(),
        "score": round(total_score, 4),
        "dayDiff": (headline_date - target_date).days if target_date and headline_date else None,
        "seasonalDayDiff": seasonal_gap,
        "yearGap": year_gap,
        "kind": kind,
        "kindLabel": MATCH_KIND_LABELS[kind],
        "bucket": bucket,
        "bucketLabel": MATCH_BUCKET_LABELS[bucket],
        "matchReason": (
            f"Arsiv basligi | {seasonal_gap if seasonal_gap is not None else '-'} sezon gun farki | "
            f"hazard {row.get('hazardType', '-') or '-'}"
        ),
        "referenceDate": headline_date.isoformat() if headline_date else "",
        "referenceVariable": target_variable,
    }


def _build_anomaly_news_detail(run_id: str, output_root: Path) -> dict:
    runs = _find_anomaly_runs(output_root)
    if not runs:
        raise FileNotFoundError("No anomaly-news runs found")

    selected_id = run_id or runs[0]["id"]
    run_dir = _safe_resolve(ROOT, selected_id)
    if not run_dir.is_dir():
        raise FileNotFoundError("Anomaly-news run directory not found")
    try:
        run_dir.relative_to(output_root.resolve())
    except ValueError as exc:
        raise FileNotFoundError("Run is outside output directory") from exc

    bundle = _resolve_anomaly_bundle(run_dir)
    unique_rows = _read_csv_records(bundle["unique_csv"])
    filtered_rows = _read_csv_records(bundle["filtered_csv"])
    missing_rows = _read_csv_records(bundle["missing_csv"]) if bundle["missing_csv"].is_file() else []
    news_rows = _read_csv_records(ANOMALY_NEWS_CATALOG) if ANOMALY_NEWS_CATALOG.is_file() else []
    archive_rows = _read_csv_records(ANOMALY_NEWS_ARCHIVE_CSV) if ANOMALY_NEWS_ARCHIVE_CSV.is_file() else []
    enriched_rows = _read_csv_records(ANOMALY_NEWS_ENRICHED_CSV) if ANOMALY_NEWS_ENRICHED_CSV.is_file() else []

    event_index: dict[str, dict[str, Any]] = {}
    for row in filtered_rows:
        event_id = str(row.get("event_id", "")).strip()
        if not event_id:
            continue
        event_index[event_id] = {
            "eventId": event_id,
            "variable": str(row.get("variable", "")).strip(),
            "centerTime": str(row.get("center_time", "")).strip(),
            "startTime": str(row.get("start_time", "")).strip(),
            "endTime": str(row.get("end_time", "")).strip(),
            "peakSeverityScore": _safe_float(row.get("peak_severity_score")),
            "severityLevel": str(row.get("severity_level", "")).strip(),
            "scientificTier": str(row.get("scientific_tier", "")).strip(),
            "internetConfidence": str(row.get("internet_confidence", "")).strip(),
            "causeSummary": str(row.get("internet_cause_summary", "")).strip(),
            "dominantDirection": str(row.get("dominant_direction", "")).strip(),
            "reasonTags": _split_pipe_text(row.get("reason_tags", "")),
            "quantCausePrimary": str(row.get("quant_cause_primary", "")).strip(),
        }

    news_index: dict[str, list[dict[str, Any]]] = {}
    news_day_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in news_rows:
        event_id = str(row.get("event_id", "")).strip()
        items = _build_summary_news_items(row, "exact_event")
        if items and event_id:
            news_index[event_id] = items
        event_day = str(row.get("event_day", "")).strip()
        event_var = str(row.get("variable", "")).strip()
        if items and event_day and event_var:
            news_day_index[(event_day, event_var)] = _build_summary_news_items(row, "exact_day")

    direct_enriched_index: dict[str, list[dict[str, Any]]] = {}
    analog_references: list[dict[str, Any]] = []
    for row in enriched_rows:
        if not str(row.get("top_headline", "")).strip():
            continue
        event_id = str(row.get("event_id", "")).strip()
        direct_item = _build_enriched_reference(row, "exact_event")
        if event_id:
            direct_enriched_index.setdefault(event_id, []).append(direct_item)
        analog_references.append(_build_enriched_reference(row, "analog_event"))

    archive_references: list[dict[str, Any]] = []
    for row in archive_rows:
        headline = str(row.get("headline", "")).strip()
        if not headline:
            continue
        archive_references.append(
            {
                "headlineDate": _safe_iso_date(row.get("headline_date")),
                "headline": headline,
                "source": str(row.get("source", "")).strip(),
                "url": str(row.get("url", "")).strip(),
                "hazardType": str(row.get("hazard_type", "")).strip(),
                "scope": str(row.get("scope", "")).strip(),
                "sourcePrior": _safe_float(row.get("source_prior")) or 0.75,
            }
        )

    points: list[dict[str, Any]] = []
    variable_counts: dict[str, int] = {}
    tier_counts: dict[str, int] = {}
    news_day_count = 0
    news_bucket_counts: dict[str, int] = {"direct": 0, "analog": 0, "archive": 0, "none": 0}
    max_severity = 0.0

    for row in unique_rows:
        day_text = str(row.get("date", "")).strip()
        target_date = _safe_iso_date(day_text)
        top_variable = str(row.get("top_variable", "")).strip()
        top_event_id = str(row.get("top_event_id", "")).strip()
        event_ids = _split_pipe_text(row.get("event_ids", ""))
        day_events: list[dict[str, Any]] = []

        for event_id in event_ids:
            meta = event_index.get(event_id)
            if meta:
                day_events.append(meta)

        day_events.sort(key=lambda item: (item["eventId"] != top_event_id, -(item.get("peakSeverityScore") or 0.0)))
        primary_event = day_events[0] if day_events else {}
        direction = str(primary_event.get("dominantDirection", "")).strip()
        reason_tags = primary_event.get("reasonTags") or _split_pipe_text(row.get("point_reason_tags", ""))
        quant_cause = str(primary_event.get("quantCausePrimary", "")).strip()
        preferred_hazards = _hazard_preferences(top_variable, direction, reason_tags, quant_cause)

        point_severity = _safe_float(row.get("top_point_severity_score"))
        event_severity = _safe_float(row.get("top_event_peak_severity_score"))
        target_severity = event_severity if event_severity is not None else point_severity
        candidate_news: list[dict[str, Any]] = []

        for event_id in event_ids:
            candidate_news.extend(news_index.get(event_id, []))
            candidate_news.extend(direct_enriched_index.get(event_id, []))

        candidate_news.extend(news_day_index.get((day_text, top_variable), []))

        for reference in analog_references:
            scored = _score_analog_reference(
                target_date,
                top_variable,
                direction,
                preferred_hazards,
                target_severity,
                reference,
            )
            if scored:
                candidate_news.append(scored)

        for reference in archive_references:
            scored = _score_archive_reference(target_date, top_variable, direction, preferred_hazards, reference)
            if scored:
                candidate_news.append(scored)

        day_news = _merge_news_items(candidate_news, max_items=4)
        best_bucket = str(day_news[0].get("bucket", "none")).strip() if day_news else "none"
        if day_news:
            news_day_count += 1
        news_bucket_counts[best_bucket] = news_bucket_counts.get(best_bucket, 0) + 1

        top_tier = str(row.get("top_scientific_tier", "")).strip()
        if top_variable:
            variable_counts[top_variable] = variable_counts.get(top_variable, 0) + 1
        if top_tier:
            tier_counts[top_tier] = tier_counts.get(top_tier, 0) + 1

        max_severity = max(max_severity, point_severity or 0.0, event_severity or 0.0)

        climate = {
            field: _safe_float(row.get(field))
            for field in ANOMALY_NEWS_CLIMATE_FIELDS
        }
        for field in ANOMALY_NEWS_TEXT_FIELDS:
            climate[field] = str(row.get(field, "")).strip()

        points.append(
            {
                "date": day_text,
                "topEventId": top_event_id,
                "topVariable": top_variable,
                "variables": _split_pipe_text(row.get("variables", "")),
                "eventIds": event_ids,
                "anomalyPointCount": _safe_int(row.get("anomaly_point_count")),
                "eventCount": _safe_int(row.get("event_count")),
                "variableCount": _safe_int(row.get("variable_count")),
                "pointTimestamp": str(row.get("top_point_timestamp", "")).strip(),
                "pointValue": _safe_float(row.get("top_point_value")),
                "pointUnit": str(row.get("top_point_unit", "")).strip(),
                "pointSeverity": point_severity,
                "pointSeverityLevel": str(row.get("top_point_severity_level", "")).strip(),
                "eventSeverity": event_severity,
                "eventSeverityLevel": str(row.get("top_event_severity_level", "")).strip(),
                "scientificScore": _safe_float(row.get("top_scientific_score")),
                "scientificTier": top_tier,
                "reasonTags": _split_pipe_text(row.get("point_reason_tags", "")),
                "pointValuesRaw": _split_pipe_text(row.get("point_values", "")),
                "hasNews": bool(day_news),
                "newsCount": len(day_news),
                "matchBucket": best_bucket,
                "matchBucketLabel": MATCH_BUCKET_LABELS.get(best_bucket, best_bucket),
                "bestNewsKind": str(day_news[0].get("kind", "")).strip() if day_news else "",
                "bestNewsScore": _safe_float(day_news[0].get("score")) if day_news else None,
                "preferredHazards": preferred_hazards,
                "dominantDirection": direction,
                "news": day_news,
                "events": day_events,
                "climate": climate,
            }
        )

    points.sort(key=lambda item: item.get("date", ""))
    selected_run = next((item for item in runs if item["id"] == selected_id), None)
    updated_at = selected_run["updated_at"] if selected_run else _format_ts(bundle["unique_csv"])

    return {
        "run": {
            "id": selected_id,
            "label": selected_run["label"] if selected_run else run_dir.name,
            "updatedAt": updated_at,
            "bundleRoot": _relative_posix(bundle["base_dir"], ROOT),
        },
        "stats": {
            "totalDays": len(points),
            "newsDays": news_day_count,
            "daysWithoutNews": len(points) - news_day_count,
            "newsBuckets": news_bucket_counts,
            "eventCount": len(filtered_rows),
            "maxSeverity": max_severity,
            "variables": variable_counts,
            "tiers": tier_counts,
        },
        "runs": runs,
        "metrics": [
            {"key": "eventSeverity", "label": "Event siddeti"},
            {"key": "pointSeverity", "label": "Anomali siddeti"},
            {"key": "t_mean_c", "label": "Ortalama sicaklik (C)"},
            {"key": "rh_mean_pct", "label": "Ortalama nem (%)"},
            {"key": "vpd_kpa", "label": "VPD (kPa)"},
            {"key": "es_minus_ea_kpa", "label": "Es-Ea farki (kPa)"},
        ],
        "days": points,
        "missingDates": missing_rows,
        "newsCatalogPath": _relative_posix(ANOMALY_NEWS_ARCHIVE_CSV, ROOT) if ANOMALY_NEWS_ARCHIVE_CSV.is_file() else "",
        "newsSummaryPath": _relative_posix(ANOMALY_NEWS_CATALOG, ROOT) if ANOMALY_NEWS_CATALOG.is_file() else "",
        "newsEnrichedPath": _relative_posix(ANOMALY_NEWS_ENRICHED_CSV, ROOT) if ANOMALY_NEWS_ENRICHED_CSV.is_file() else "",
        "latestClimate": _latest_daily_snapshot(LATEST_DAILY_CSV),
    }


def _resolve_latest_quant_run(output_root: Path) -> Path:
    symlink = output_root / "yeni_model_newdata_latest"
    if symlink.exists():
        resolved = symlink.resolve()
        if resolved.is_dir():
            return resolved

    candidates = [p for p in output_root.glob("yeni_model_newdata_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("No yeni_model_newdata run found")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_key_value_text(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _csv_history_bounds(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"historyStart": None, "historyEnd": None, "historyRows": 0, "forecastRows": 0}
    history_start = None
    history_end = None
    history_rows = 0
    forecast_rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ds = str(row.get("ds", "")).strip()
            is_fc_raw = str(row.get("is_forecast", "")).strip().lower()
            is_fc = is_fc_raw in {"1", "true", "yes"}
            if is_fc:
                forecast_rows += 1
                continue
            history_rows += 1
            if ds:
                if history_start is None or ds < history_start:
                    history_start = ds
                if history_end is None or ds > history_end:
                    history_end = ds
    return {
        "historyStart": history_start,
        "historyEnd": history_end,
        "historyRows": history_rows,
        "forecastRows": forecast_rows,
    }


def _discover_quant_file(
    run_dir: Path,
    subdir: str,
    variable: str,
    prefer_history: bool,
    suffix: str = ".csv",
    include_name_tokens: tuple[str, ...] = (),
    exclude_name_tokens: tuple[str, ...] = (),
) -> Path | None:
    base = run_dir / subdir
    if not base.is_dir():
        return None
    candidates = [p for p in base.glob(f"{variable}_*{suffix}") if p.is_file()]
    if include_name_tokens:
        candidates = [
            p for p in candidates
            if all(tok in p.name.lower() for tok in include_name_tokens)
        ]
    if exclude_name_tokens:
        candidates = [
            p for p in candidates
            if all(tok not in p.name.lower() for tok in exclude_name_tokens)
        ]
    if not candidates:
        return None

    def _score(path: Path) -> tuple[int, int, float]:
        name = path.name.lower()
        score_history = 1 if "history" in name else 0
        score_full = 1 if re.search(r"_to_\d{4}", name) else 0
        # prefer recent files
        ts = path.stat().st_mtime
        if prefer_history:
            return (score_history, score_full, ts)
        return (score_full, score_history, ts)

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def _build_live_quant(output_root: Path) -> dict[str, Any]:
    run_dir = _resolve_latest_quant_run(output_root)
    run_id = _relative_posix(run_dir, ROOT)
    model_meta = _read_key_value_text(run_dir / "YENI_MODEL.txt")
    analysis_mode = str(model_meta.get("analysis_mode", "")).strip().lower()
    prefer_history = analysis_mode == "anomalies_only"

    variables = ["humidity", "precip", "pressure", "temp"]
    items: list[dict[str, Any]] = []
    for variable in variables:
        forecast_csv = _discover_quant_file(run_dir, "forecasts", variable, prefer_history=prefer_history, suffix=".csv")
        anomaly_csv = _discover_quant_file(run_dir, "anomalies", variable, prefer_history=prefer_history, suffix=".csv")
        model_chart_png = _discover_quant_file(
            run_dir,
            "charts",
            variable,
            prefer_history=prefer_history,
            suffix=".png",
            include_name_tokens=("quant",),
            exclude_name_tokens=("regime_probs",),
        )
        if model_chart_png is None:
            model_chart_png = _discover_quant_file(
                run_dir,
                "charts",
                variable,
                prefer_history=prefer_history,
                suffix=".png",
                exclude_name_tokens=("regime_probs",),
            )

        bounds = _csv_history_bounds(forecast_csv) if forecast_csv is not None else {
            "historyStart": None,
            "historyEnd": None,
            "historyRows": 0,
            "forecastRows": 0,
        }
        anomaly_rows = _count_csv_rows(anomaly_csv) if anomaly_csv is not None else 0
        items.append(
            {
                "variable": variable,
                "forecastCsv": _relative_posix(forecast_csv, ROOT) if forecast_csv is not None else "",
                "anomaliesCsv": _relative_posix(anomaly_csv, ROOT) if anomaly_csv is not None else "",
                "chartPng": _relative_posix(model_chart_png, ROOT) if model_chart_png is not None else "",
                "realChartPng": "",
                "modelChartPng": _relative_posix(model_chart_png, ROOT) if model_chart_png is not None else "",
                "historyStart": bounds["historyStart"],
                "historyEnd": bounds["historyEnd"],
                "historyRows": int(bounds["historyRows"]),
                "forecastRows": int(bounds["forecastRows"]),
                "anomalyRows": int(anomaly_rows),
            }
        )

    return {
        "run": {
            "id": run_id,
            "updatedAt": _format_ts(run_dir),
            "analysisMode": analysis_mode or "unknown",
            "historyStartParam": model_meta.get("history_start", ""),
            "historyEndParam": model_meta.get("history_end", ""),
            "input": model_meta.get("input", ""),
            "denseWindowMode": model_meta.get("dense_window_mode", ""),
        },
        "items": items,
    }


def _run_detail(run_id: str, output_root: Path) -> dict:
    run_dir = _safe_resolve(ROOT, run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError("Run directory not found")
    try:
        run_dir.relative_to(output_root.resolve())
    except ValueError as exc:
        raise FileNotFoundError("Run is outside output directory") from exc

    summary_path = run_dir / "model_suite_summary.json"
    if summary_path.exists():
        summary = _json_load(summary_path)
        updated_at = _format_ts(summary_path)
    else:
        summary = _build_artifact_summary(run_dir)
        updated_at = _format_ts(run_dir)

    health_suite = _load_health_suite(run_dir)
    summary = _merge_health_into_summary(summary, health_suite, run_dir)
    feature_inventory, feature_data, total_files = _build_feature_views(run_dir)
    model_status = _build_model_status(run_dir, summary)
    presentation_assets_by_mode = {
        mode: _build_presentation_assets(run_dir, max_items=MAX_PRESENTATION_CHARTS, mode=mode)
        for mode in PRESENTATION_MODES
    }
    presentation_assets = presentation_assets_by_mode.get("balanced", [])
    active_features = len([x for x in feature_inventory if int(x.get("count", 0)) > 0])
    presentation_note = _build_presentation_note(
        run_id=_relative_posix(run_dir, ROOT),
        updated_at=updated_at,
        summary=summary,
        health_suite=health_suite,
        total_files=total_files,
        active_features=active_features,
    )

    return {
        "id": _relative_posix(run_dir, ROOT),
        "updated_at": updated_at,
        "summary": summary,
        "supported_models": SUPPORTED_MODELS,
        "model_status": model_status,
        "feature_inventory": feature_inventory,
        "feature_data": feature_data,
        "total_files": total_files,
        "health_suite": health_suite,
        "presentation_note": presentation_note,
        "presentation_assets": presentation_assets,
        "presentation_assets_by_mode": presentation_assets_by_mode,
        "presentation_modes": list(PRESENTATION_MODES),
        "chart_groups": _gather_chart_groups(run_dir),
        "indexes": _gather_csv_previews(run_dir, "*index*.csv"),
        "leaderboards": _gather_csv_previews(run_dir, "*leaderboard*.csv"),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    output_root = DEFAULT_OUTPUT_ROOT

    def _send_json(self, payload: dict | list, status: int = 200) -> None:
        safe_payload = _sanitize_json_value(payload)
        blob = json.dumps(safe_payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        filename: str | None = None,
        status: int = 200,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(
        self,
        text: str,
        content_type: str = "text/plain; charset=utf-8",
        filename: str | None = None,
        status: int = 200,
    ) -> None:
        self._send_bytes(text.encode("utf-8"), content_type=content_type, filename=filename, status=status)

    def _send_file(self, path: Path) -> None:
        content_type, _ = mimetypes.guess_type(path.name)
        data = path.read_bytes()
        self._send_bytes(data, content_type=content_type or "application/octet-stream", filename=None, status=200)

    def _serve_static(self, relative: str) -> None:
        try:
            file_path = _safe_resolve(STATIC_ROOT, relative)
        except FileNotFoundError:
            self.send_error(404, "Static file not found")
            return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "Static file not found")
            return
        self._send_file(file_path)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in {"/", "/index.html"}:
            self._serve_static("index.html")
            return
        if path in {"/anomaly-news", "/anomaly-news.html"}:
            self._serve_static("anomaly-news.html")
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if path.startswith("/static/"):
            self._serve_static(path.replace("/static/", "", 1))
            return
        if path == "/api/runs":
            self._send_json({"runs": _find_runs(self.output_root)})
            return
        if path == "/api/anomaly-news-runs":
            self._send_json({"runs": _find_anomaly_runs(self.output_root)})
            return
        if path == "/api/anomaly-news":
            run_id = query.get("id", [""])[0]
            try:
                self._send_json(_build_anomaly_news_detail(run_id, self.output_root))
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=404)
            return
        if path == "/api/run":
            run_id = query.get("id", [""])[0]
            if not run_id:
                self._send_json({"error": "Query param 'id' is required"}, status=400)
                return
            try:
                self._send_json(_run_detail(run_id, self.output_root))
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=404)
            return
        if path == "/api/live-quant":
            try:
                self._send_json(_build_live_quant(self.output_root))
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=404)
            return
        if path == "/api/export_note":
            run_id = query.get("id", [""])[0]
            fmt = str(query.get("format", ["txt"])[0]).strip().lower()
            if not run_id:
                self._send_json({"error": "Query param 'id' is required"}, status=400)
                return
            if fmt not in {"txt", "md", "pdf"}:
                self._send_json({"error": "format must be txt, md or pdf"}, status=400)
                return
            try:
                detail = _run_detail(run_id, self.output_root)
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=404)
                return
            note = str(detail.get("presentation_note", "")).strip()
            if fmt == "md":
                body = (
                    "# Resmi Sunum Notu\n\n"
                    f"**Kosu:** `{detail['id']}`\n\n"
                    "```text\n"
                    f"{note}\n"
                    "```\n"
                )
                filename = f"{detail['id'].replace('/', '_')}_sunum_notu.md"
                self._send_text(
                    body,
                    content_type="text/markdown; charset=utf-8",
                    filename=filename,
                )
            elif fmt == "pdf":
                pdf_blob = _render_note_pdf(note, f"{detail['id']} Sunum Notu")
                if pdf_blob is None:
                    self._send_json({"error": "PDF export icin reportlab paketi gerekli."}, status=501)
                    return
                filename = f"{detail['id'].replace('/', '_')}_sunum_notu.pdf"
                self._send_bytes(
                    pdf_blob,
                    content_type="application/pdf",
                    filename=filename,
                )
            else:
                filename = f"{detail['id'].replace('/', '_')}_sunum_notu.txt"
                self._send_text(note + "\n", content_type="text/plain; charset=utf-8", filename=filename)
            return
        if path == "/api/export_summary":
            run_id = query.get("id", [""])[0]
            if not run_id:
                self._send_json({"error": "Query param 'id' is required"}, status=400)
                return
            try:
                detail = _run_detail(run_id, self.output_root)
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=404)
                return
            payload = json.dumps(_sanitize_json_value(detail), ensure_ascii=False, indent=2) + "\n"
            filename = f"{detail['id'].replace('/', '_')}_run_summary.json"
            self._send_text(payload, content_type="application/json; charset=utf-8", filename=filename)
            return
        if path == "/api/export_pack":
            run_id = query.get("id", [""])[0]
            mode = _normalize_presentation_mode(query.get("mode", ["balanced"])[0])
            if not run_id:
                self._send_json({"error": "Query param 'id' is required"}, status=400)
                return
            try:
                detail = _run_detail(run_id, self.output_root)
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=404)
                return
            blob = _build_export_pack(detail, mode=mode)
            filename = f"{detail['id'].replace('/', '_')}_sunum_paketi_{mode}.zip"
            self._send_bytes(blob, content_type="application/zip", filename=filename)
            return
        if path == "/api/export_charts":
            run_id = query.get("id", [""])[0]
            mode = _normalize_presentation_mode(query.get("mode", ["balanced"])[0])
            if not run_id:
                self._send_json({"error": "Query param 'id' is required"}, status=400)
                return
            try:
                detail = _run_detail(run_id, self.output_root)
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=404)
                return
            blob = _build_chart_pack(detail, mode=mode)
            filename = f"{detail['id'].replace('/', '_')}_sunum_grafikleri_{mode}.zip"
            self._send_bytes(blob, content_type="application/zip", filename=filename)
            return
        if path.startswith("/files/"):
            relative = unquote(path.replace("/files/", "", 1))
            try:
                file_path = _safe_resolve(ROOT, relative)
            except FileNotFoundError:
                self.send_error(404, "File not found")
                return
            if not file_path.is_file():
                self.send_error(404, "File not found")
                return
            self._send_file(file_path)
            return

        self.send_error(404, "Not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hackhaton model suite dashboard server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    DashboardHandler.output_root = Path(args.output_root).resolve()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print(f"Output root: {DashboardHandler.output_root}")
    server.serve_forever()


if __name__ == "__main__":
    main()
