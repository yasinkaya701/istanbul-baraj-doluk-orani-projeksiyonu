#!/usr/bin/env python3
"""Run all climate models on any dataset with a single command.

Bu script, hackathon içinde üretilen modelleri tek bir girişten koşturur:
- quant_regime_projection
- prophet_climate_forecast
- prophet_ultra_500
- train_strong_consistent_model
- analog_pattern_forecast
- walkforward_retrain_multifreq
- best_climate_meta_ensemble
- literature_robust_forecast
- build_stable_consensus_forecast

Ek olarak, ham veri klasörü verildiğinde ortak gözlem tablosu üretir.
İsteğe bağlı olarak tüm TIFF görselleri de sayısallaştırıp girdiye katar.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    name: str
    command: list[str]
    ok: bool
    returncode: int
    output_dir: str
    stdout_tail: str
    stderr_tail: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tek komutla tüm modelleri her dataset üzerinde çalıştır."
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Ham dataset yolu (dosya veya klasör).",
    )
    p.add_argument(
        "--prepared-observations",
        type=Path,
        default=None,
        help="Hazır gözlem dosyası (timestamp,variable,value) varsa direkt bunu kullan.",
    )
    p.add_argument(
        "--graph-root",
        type=Path,
        default=None,
        help="Tüm TIFF görsellerin kök klasörü (opsiyonel).",
    )
    p.add_argument(
        "--include-visuals",
        type=str,
        default="false",
        help="true/false: graph-root içindeki tüm TIFF görselleri modele kat.",
    )
    p.add_argument(
        "--stabilize-observations",
        type=str,
        default="true",
        help="true/false: model eğitiminden önce otomatik kalibrasyon + rejim temizleme uygula.",
    )
    p.add_argument(
        "--stabilization-gap-years",
        type=float,
        default=5.0,
        help="Bundan büyük tarih boşluğu varsa son sürekli rejim seçilir (yıl).",
    )
    p.add_argument(
        "--stabilization-min-recent-ratio",
        type=float,
        default=0.25,
        help="Recent rejim satır oranı bu eşikten düşükse full calibrated fallback kullanılır.",
    )
    p.add_argument(
        "--stabilization-pressure-offset",
        type=float,
        default=900.0,
        help="Düşük ölçekli pressure numeric değerlerine eklenecek ofset.",
    )
    p.add_argument(
        "--stabilization-pressure-low-min",
        type=float,
        default=40.0,
        help="Pressure low-scale algılama alt sınırı.",
    )
    p.add_argument(
        "--stabilization-pressure-low-max",
        type=float,
        default=120.0,
        help="Pressure low-scale algılama üst sınırı.",
    )
    p.add_argument(
        "--stabilization-fail-on-error",
        type=str,
        default="false",
        help="true/false: stabilizasyon başarısız olursa koşuyu durdur.",
    )
    p.add_argument(
        "--models",
        type=str,
        default="quant,prophet,strong,analog,prophet_ultra,walkforward,best_meta,literature,stable_consensus",
        help="Çalıştırılacak modeller: quant,prophet,strong,analog,prophet_ultra,walkforward,best_meta,literature,stable_consensus",
    )
    p.add_argument(
        "--variables",
        type=str,
        default="*",
        help="Model değişken filtresi (örn: temp,humidity,pressure,precip veya *).",
    )
    p.add_argument("--target-year", type=int, default=2035)
    p.add_argument("--start-year", type=int, default=2026, help="Walkforward başlangıç yılı.")
    p.add_argument("--walkforward-freqs", type=str, default="YS,MS,W,D")
    p.add_argument(
        "--climate-scenario",
        type=str,
        default="ssp245",
        help="İklim senaryosu: none, ssp126, ssp245, ssp370, ssp585",
    )
    p.add_argument(
        "--climate-baseline-year",
        type=float,
        default=float("nan"),
        help="Senaryo düzeltmesi baz yılı; NaN ise seri son gözlem yılı otomatik alınır.",
    )
    p.add_argument(
        "--climate-temp-rate",
        type=float,
        default=float("nan"),
        help="Sıcaklık trend override (C/yıl). NaN ise senaryo varsayılanı.",
    )
    p.add_argument(
        "--humidity-per-temp-c",
        type=float,
        default=-2.0,
        help="Nem düzeltme katsayısı (yüzde puan / C).",
    )
    p.add_argument(
        "--climate-adjustment-method",
        type=str,
        default="pathway",
        help="Düzeltme metodu: pathway (IPCC AR6 SSP eğrisi) veya linear.",
    )
    p.add_argument(
        "--disable-climate-adjustment",
        type=str,
        default="false",
        help="true/false: senaryo katsayısı düzeltmesini kapat.",
    )
    p.add_argument(
        "--best-meta-auto-select-combination",
        type=str,
        default="true",
        help="true/false: best_meta içinde alt-kume kombinasyonunu rolling-CV ile otomatik seç.",
    )
    p.add_argument(
        "--best-meta-max-combo-size",
        type=int,
        default=3,
        help="best_meta otomatik seçimde maksimum model sayısı.",
    )
    p.add_argument(
        "--best-meta-combo-complexity-penalty",
        type=float,
        default=0.025,
        help="best_meta otomatik seçimde model sayısı ceza katsayısı.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/model_suite"),
        help="Ana çıktı klasörü.",
    )
    p.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Kullanılacak Python executable.",
    )
    p.add_argument(
        "--fail-fast",
        type=str,
        default="false",
        help="true/false: ilk model hatasında dur.",
    )
    p.add_argument(
        "--run-robust-selection",
        type=str,
        default="true",
        help="true/false: model kosusu sonrasinda stabilite+dogruluk bazli model secimi yap.",
    )
    p.add_argument(
        "--robust-selection-fail-on-error",
        type=str,
        default="false",
        help="true/false: robust model secimi hatasinda kosuyu basarisiz say.",
    )
    p.add_argument(
        "--run-health-suite",
        type=str,
        default="true",
        help="true/false: model suite sonrası sağlık etki analizlerini de çalıştır.",
    )
    p.add_argument(
        "--health-models",
        type=str,
        default="*",
        help="Sağlık suite için model listesi ('*' tüm destekli modeller).",
    )
    p.add_argument(
        "--health-output-subdir",
        type=str,
        default="health",
        help="Sağlık çıktılarının output-dir altında yazılacağı alt klasör.",
    )
    p.add_argument("--health-baseline-start", type=int, default=1991)
    p.add_argument("--health-baseline-end", type=int, default=2020)
    p.add_argument("--health-future-start", type=int, default=2026)
    p.add_argument(
        "--health-future-end",
        type=int,
        default=-1,
        help="<=0 ise hedef yıl otomatik alınır.",
    )
    p.add_argument(
        "--health-fail-on-error",
        type=str,
        default="false",
        help="true/false: sağlık suite başarısız olursa tüm koşuyu başarısız say.",
    )
    return p.parse_args()


def to_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_model_list(text: str) -> list[str]:
    allowed = {
        "quant",
        "prophet",
        "strong",
        "analog",
        "prophet_ultra",
        "walkforward",
        "best_meta",
        "literature",
        "stable_consensus",
    }
    out = []
    for tok in [x.strip().lower() for x in str(text).split(",") if x.strip()]:
        if tok in allowed and tok not in out:
            out.append(tok)
    return out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def tail_text(text: str, max_chars: int = 2500) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_command(
    name: str,
    cmd: list[str],
    output_dir: Path,
    env: dict[str, str],
) -> RunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    return RunResult(
        name=name,
        command=cmd,
        ok=(proc.returncode == 0),
        returncode=int(proc.returncode),
        output_dir=str(output_dir),
        stdout_tail=tail_text(proc.stdout or ""),
        stderr_tail=tail_text(proc.stderr or ""),
    )


def run_health_suite(
    args: argparse.Namespace,
    py: str,
    out_root: Path,
    env: dict[str, str],
    models: list[str],
) -> tuple[RunResult, dict[str, Any] | None]:
    health_root = out_root / str(args.health_output_subdir)
    health_models = str(args.health_models).strip()
    if health_models == "*":
        health_models = ",".join(models)
    cmd = [
        py,
        "scripts/run_health_suite.py",
        "--run-dir",
        str(out_root),
        "--models",
        health_models,
        "--output-subdir",
        str(args.health_output_subdir),
        "--baseline-start",
        str(args.health_baseline_start),
        "--baseline-end",
        str(args.health_baseline_end),
        "--future-start",
        str(args.health_future_start),
        "--future-end",
        str(args.health_future_end),
        "--python-bin",
        str(py),
    ]
    res = run_command("health_suite", cmd, health_root, env)
    summary_path = health_root / "health_suite_summary.json"
    health_summary = None
    if summary_path.exists():
        try:
            health_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            health_summary = None
    return res, health_summary


def run_robust_selection(
    py: str,
    out_root: Path,
    env: dict[str, str],
) -> tuple[RunResult, dict[str, Any] | None]:
    robust_root = out_root / "robust_selection"
    cmd = [
        py,
        "scripts/build_robust_model_selection.py",
        "--run-dir",
        str(out_root),
        "--output-dir",
        str(robust_root),
    ]
    res = run_command("robust_selection", cmd, robust_root, env)
    summary_path = robust_root / "robust_model_selection_summary.json"
    robust_summary = None
    if summary_path.exists():
        try:
            robust_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            robust_summary = None
    return res, robust_summary


def prepare_observations(
    args: argparse.Namespace,
    py: str,
    out_root: Path,
    env: dict[str, str],
) -> tuple[Path, list[RunResult]]:
    results: list[RunResult] = []

    if args.prepared_observations is not None:
        if not args.prepared_observations.exists():
            raise SystemExit(f"--prepared-observations bulunamadı: {args.prepared_observations}")
        obs_path = args.prepared_observations
    else:
        if args.dataset is None:
            raise SystemExit("Dataset vermelisin: --dataset veya --prepared-observations")
        if not args.dataset.exists():
            raise SystemExit(f"--dataset bulunamadı: {args.dataset}")

        if args.dataset.is_file():
            obs_path = args.dataset
        else:
            prep_dir = out_root / "prepare_universal"
            cmd = [
                py,
                "scripts/universal_climate_forecast_pipeline.py",
                "--input-dir",
                str(args.dataset),
                "--output-dir",
                str(prep_dir),
                "--freqs",
                "monthly",
            ]
            res = run_command("prepare_universal", cmd, prep_dir, env)
            results.append(res)
            if not res.ok:
                raise SystemExit("universal_climate_forecast_pipeline başarısız oldu.")
            obs_path = prep_dir / "observations_universal.parquet"
            if not obs_path.exists():
                raise SystemExit(f"Beklenen çıktı yok: {obs_path}")

    if to_bool(args.include_visuals):
        if args.graph_root is None:
            raise SystemExit("--include-visuals=true için --graph-root gerekli.")
        vis_dir = out_root / "prepare_visuals"
        cmd = [
            py,
            "scripts/process_all_visuals_to_quant.py",
            "--graph-root",
            str(args.graph_root),
            "--numeric-parquet",
            str(obs_path),
            "--output-dir",
            str(vis_dir),
        ]
        res = run_command("prepare_visuals", cmd, vis_dir, env)
        results.append(res)
        if not res.ok:
            raise SystemExit("process_all_visuals_to_quant başarısız oldu.")
        obs_path = vis_dir / "observations_with_all_visuals_for_quant.parquet"
        if not obs_path.exists():
            raise SystemExit(f"Beklenen görsel birleşik çıktı yok: {obs_path}")

    return obs_path, results


def _safe_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _coerce_existing_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    p = Path(text)
    return p if p.exists() else None


def stabilize_observations(
    args: argparse.Namespace,
    py: str,
    out_root: Path,
    env: dict[str, str],
    obs_path: Path,
) -> tuple[Path, RunResult | None, dict[str, Any]]:
    disabled = not to_bool(args.stabilize_observations)
    if disabled:
        return obs_path, None, {
            "enabled": False,
            "status": "disabled",
            "applied": False,
            "selected_strategy": "raw",
            "selected_path": str(obs_path),
        }

    stab_dir = out_root / "prepare_calibrated"
    cmd = [
        py,
        "scripts/calibrate_observations_for_forecast.py",
        "--input-observations",
        str(obs_path),
        "--output-dir",
        str(stab_dir),
        "--gap-years",
        str(args.stabilization_gap_years),
        "--pressure-offset",
        str(args.stabilization_pressure_offset),
        "--pressure-low-min",
        str(args.stabilization_pressure_low_min),
        "--pressure-low-max",
        str(args.stabilization_pressure_low_max),
    ]
    res = run_command("stabilize_observations", cmd, stab_dir, env)
    info: dict[str, Any] = {
        "enabled": True,
        "status": ("ok" if res.ok else "failed"),
        "applied": False,
        "selected_strategy": "raw_fallback",
        "selected_path": str(obs_path),
        "output_dir": str(stab_dir),
        "min_recent_ratio": float(args.stabilization_min_recent_ratio),
    }

    if not res.ok:
        info["reason"] = "calibration_step_failed"
        if to_bool(args.stabilization_fail_on_error):
            raise SystemExit("Gözlem stabilizasyonu başarısız oldu (--stabilization-fail-on-error=true).")
        return obs_path, res, info

    summary_path = stab_dir / "summary.json"
    summary = _safe_json(summary_path) or {}
    outputs = summary.get("outputs", {}) if isinstance(summary, dict) else {}

    full_path = _coerce_existing_path(outputs.get("full_parquet")) or (stab_dir / "observations_calibrated_full.parquet")
    recent_path = _coerce_existing_path(outputs.get("recent_parquet")) or (stab_dir / "observations_calibrated_recent_regime.parquet")

    rows_input = int(summary.get("rows_input", 0) or 0)
    rows_full = int(summary.get("rows_calibrated_full", 0) or 0)
    rows_recent = int(summary.get("rows_recent_regime", 0) or 0)
    recent_ratio = (float(rows_recent) / float(rows_full)) if rows_full > 0 else 0.0

    selected = obs_path
    strategy = "raw_fallback"
    if recent_path.exists() and rows_recent > 0 and recent_ratio >= float(args.stabilization_min_recent_ratio):
        selected = recent_path
        strategy = "recent_regime"
    elif full_path.exists() and rows_full > 0:
        selected = full_path
        strategy = "full_calibrated"
    elif recent_path.exists():
        selected = recent_path
        strategy = "recent_regime_no_summary"
    elif full_path.exists():
        selected = full_path
        strategy = "full_calibrated_no_summary"

    info.update(
        {
            "status": "ok",
            "applied": str(selected) != str(obs_path),
            "selected_strategy": strategy,
            "selected_path": str(selected),
            "rows_input": rows_input,
            "rows_calibrated_full": rows_full,
            "rows_recent_regime": rows_recent,
            "recent_ratio": recent_ratio,
            "pressure_fix_count": int(summary.get("pressure_fix_count", 0) or 0),
            "gap_years": float(summary.get("gap_years", args.stabilization_gap_years)),
        }
    )
    return selected, res, info


def model_commands(
    models: list[str],
    py: str,
    obs_path: Path,
    out_root: Path,
    args: argparse.Namespace,
) -> list[tuple[str, list[str], Path]]:
    out: list[tuple[str, list[str], Path]] = []

    if "quant" in models:
        q_dir = out_root / "quant"
        out.append(
            (
                "quant",
                [
                    py,
                    "scripts/quant_regime_projection.py",
                    "--observations",
                    str(obs_path),
                    "--output-dir",
                    str(q_dir),
                    "--variables",
                    str(args.variables),
                    "--target-year",
                    str(args.target_year),
                    "--climate-scenario",
                    str(args.climate_scenario),
                    "--climate-baseline-year",
                    str(args.climate_baseline_year),
                    "--climate-temp-rate",
                    str(args.climate_temp_rate),
                    "--humidity-per-temp-c",
                    str(args.humidity_per_temp_c),
                    "--climate-adjustment-method",
                    str(args.climate_adjustment_method),
                ],
                q_dir,
            )
        )
        if to_bool(args.disable_climate_adjustment):
            out[-1][1].append("--disable-climate-adjustment")

    if "prophet" in models:
        p_dir = out_root / "prophet"
        out.append(
            (
                "prophet",
                [
                    py,
                    "scripts/prophet_climate_forecast.py",
                    "--observations",
                    str(obs_path),
                    "--output-dir",
                    str(p_dir),
                    "--variables",
                    str(args.variables),
                    "--target-year",
                    str(args.target_year),
                    "--input-kind",
                    "auto",
                ],
                p_dir,
            )
        )

    if "strong" in models:
        s_dir = out_root / "strong"
        out.append(
            (
                "strong",
                [
                    py,
                    "scripts/train_strong_consistent_model.py",
                    "--observations",
                    str(obs_path),
                    "--output-dir",
                    str(s_dir),
                    "--variables",
                    str(args.variables),
                    "--target-year",
                    str(args.target_year),
                    "--input-kind",
                    "auto",
                ],
                s_dir,
            )
        )

    if "analog" in models:
        a_dir = out_root / "analog"
        out.append(
            (
                "analog",
                [
                    py,
                    "scripts/analog_pattern_forecast.py",
                    "--observations",
                    str(obs_path),
                    "--output-dir",
                    str(a_dir),
                    "--variables",
                    str(args.variables),
                    "--target-year",
                    str(args.target_year),
                    "--input-kind",
                    "auto",
                ],
                a_dir,
            )
        )

    if "prophet_ultra" in models:
        pu_dir = out_root / "prophet_ultra"
        out.append(
            (
                "prophet_ultra",
                [
                    py,
                    "scripts/prophet_ultra_500.py",
                    "--observations",
                    str(obs_path),
                    "--output-dir",
                    str(pu_dir),
                    "--variables",
                    str(args.variables),
                    "--target-year",
                    str(args.target_year),
                    "--input-kind",
                    "auto",
                ],
                pu_dir,
            )
        )

    if "walkforward" in models:
        w_dir = out_root / "walkforward"
        out.append(
            (
                "walkforward",
                [
                    py,
                    "scripts/walkforward_retrain_multifreq.py",
                    "--observations",
                    str(obs_path),
                    "--output-dir",
                    str(w_dir),
                    "--variables",
                    str(args.variables),
                    "--frequencies",
                    str(args.walkforward_freqs),
                    "--start-year",
                    str(args.start_year),
                    "--target-year",
                    str(args.target_year),
                    "--input-kind",
                    "auto",
                    "--climate-scenario",
                    str(args.climate_scenario),
                    "--climate-baseline-year",
                    str(args.climate_baseline_year),
                    "--climate-temp-rate",
                    str(args.climate_temp_rate),
                    "--humidity-per-temp-c",
                    str(args.humidity_per_temp_c),
                    "--climate-adjustment-method",
                    str(args.climate_adjustment_method),
                ],
                w_dir,
            )
        )
        if to_bool(args.disable_climate_adjustment):
            out[-1][1].append("--disable-climate-adjustment")

    if "best_meta" in models:
        bm_dir = out_root / "best_meta"
        has_all_base = all(k in models for k in ["quant", "strong", "prophet_ultra", "walkforward"])
        run_base = "false" if has_all_base else "true"
        out.append(
            (
                "best_meta",
                [
                    py,
                    "scripts/best_climate_meta_ensemble.py",
                    "--observations",
                    str(obs_path),
                    "--output-dir",
                    str(bm_dir),
                    "--variables",
                    str(args.variables),
                    "--target-year",
                    str(args.target_year),
                    "--walkforward-start-year",
                    str(args.start_year),
                    "--input-kind",
                    "auto",
                    "--run-base-models",
                    run_base,
                    "--reuse-existing",
                    "true",
                    "--base-dir-name",
                    "..",
                    "--auto-select-combination",
                    str(args.best_meta_auto_select_combination),
                    "--max-combo-size",
                    str(args.best_meta_max_combo_size),
                    "--combo-complexity-penalty",
                    str(args.best_meta_combo_complexity_penalty),
                    "--climate-scenario",
                    str(args.climate_scenario),
                    "--climate-baseline-year",
                    str(args.climate_baseline_year),
                    "--climate-temp-rate",
                    str(args.climate_temp_rate),
                    "--humidity-per-temp-c",
                    str(args.humidity_per_temp_c),
                    "--climate-adjustment-method",
                    str(args.climate_adjustment_method),
                ],
                bm_dir,
            )
        )
        if to_bool(args.disable_climate_adjustment):
            out[-1][1].append("--disable-climate-adjustment")

    if "literature" in models:
        lit_dir = out_root / "literature"
        out.append(
            (
                "literature",
                [
                    py,
                    "scripts/literature_robust_forecast.py",
                    "--observations",
                    str(obs_path),
                    "--output-dir",
                    str(lit_dir),
                    "--variables",
                    str(args.variables),
                    "--target-year",
                    str(args.target_year),
                    "--input-kind",
                    "auto",
                ],
                lit_dir,
            )
        )

    if "stable_consensus" in models:
        sc_dir = out_root / "stable_consensus"
        out.append(
            (
                "stable_consensus",
                [
                    py,
                    "scripts/build_stable_consensus_forecast.py",
                    "--run-dir",
                    str(out_root),
                    "--future-start",
                    str(args.health_future_start),
                    "--future-end",
                    str(args.target_year),
                    "--output-model-dir",
                    "stable_consensus",
                ],
                sc_dir,
            )
        )

    return out


def main() -> None:
    args = parse_args()
    if not (0.0 <= float(args.stabilization_min_recent_ratio) <= 1.0):
        raise SystemExit("--stabilization-min-recent-ratio 0 ile 1 arasinda olmali.")
    if float(args.stabilization_gap_years) <= 0:
        raise SystemExit("--stabilization-gap-years pozitif olmalı.")
    models = parse_model_list(args.models)
    if not models:
        raise SystemExit("Geçerli model yok. --models parametresini kontrol et.")

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    py = args.python_bin

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    observations_original, prep_results = prepare_observations(args, py=py, out_root=out_root, env=env)

    run_results: list[RunResult] = []
    run_results.extend(prep_results)
    observations_path, stabilization_result, stabilization_info = stabilize_observations(
        args=args,
        py=py,
        out_root=out_root,
        env=env,
        obs_path=observations_original,
    )
    if stabilization_result is not None:
        run_results.append(stabilization_result)
    fail_fast = to_bool(args.fail_fast)

    for name, cmd, out_dir in model_commands(models, py=py, obs_path=observations_path, out_root=out_root, args=args):
        res = run_command(name, cmd, out_dir, env)
        run_results.append(res)
        if fail_fast and not res.ok:
            break

    robust_summary: dict[str, Any] | None = None
    robust_result: RunResult | None = None
    if to_bool(args.run_robust_selection):
        robust_result, robust_summary = run_robust_selection(
            py=py,
            out_root=out_root,
            env=env,
        )
        run_results.append(robust_result)
        if to_bool(args.robust_selection_fail_on_error) and not robust_result.ok:
            raise SystemExit("Robust model secimi basarisiz oldu (--robust-selection-fail-on-error=true).")

    health_summary: dict[str, Any] | None = None
    health_result: RunResult | None = None
    if to_bool(args.run_health_suite):
        health_result, health_summary = run_health_suite(
            args=args, py=py, out_root=out_root, env=env, models=models
        )
        run_results.append(health_result)
        if to_bool(args.health_fail_on_error) and not health_result.ok:
            raise SystemExit("Health suite başarısız oldu (--health-fail-on-error=true).")

    model_only = [r for r in run_results if r.name in models]
    by_name = {r.name: r for r in model_only}
    models_ok = [m for m in models if m in by_name and by_name[m].ok]
    models_failed = [m for m in models if m not in by_name or not by_name[m].ok]

    summary = {
        "observations_original": str(observations_original),
        "observations_used": str(observations_path),
        "stabilization": stabilization_info,
        "models_requested": models,
        "models_ok": models_ok,
        "models_failed": models_failed,
        "robust_selection": {
            "enabled": to_bool(args.run_robust_selection),
            "ok": (robust_result.ok if robust_result is not None else None),
            "output_dir": (robust_result.output_dir if robust_result is not None else ""),
            "summary": robust_summary,
        },
        "health_suite": {
            "enabled": to_bool(args.run_health_suite),
            "ok": (health_result.ok if health_result is not None else None),
            "output_dir": (health_result.output_dir if health_result is not None else ""),
            "summary": health_summary,
        },
        "results": [
            {
                "name": r.name,
                "ok": r.ok,
                "returncode": r.returncode,
                "output_dir": r.output_dir,
                "command": " ".join(shlex.quote(x) for x in r.command),
                "stdout_tail": r.stdout_tail,
                "stderr_tail": r.stderr_tail,
            }
            for r in run_results
        ],
    }

    summary_json = out_root / "model_suite_summary.json"
    summary_md = out_root / "model_suite_summary.md"
    ensure_parent(summary_json)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Model Suite Özeti",
        "",
        f"- Orijinal gözlem dosyası: `{observations_original}`",
        f"- Modelde kullanılan gözlem dosyası: `{observations_path}`",
        (
            f"- Stabilizasyon: `{stabilization_info.get('status', '-')}`"
            f" | strateji: `{stabilization_info.get('selected_strategy', '-')}`"
        ),
        (
            f"- Stabilizasyon recent oranı: "
            f"`{(float(stabilization_info.get('recent_ratio', 0.0)) * 100.0):.1f}%`"
            if "recent_ratio" in stabilization_info
            else "- Stabilizasyon recent oranı: `-`"
        ),
        (
            f"- Robust secim: `{robust_result.ok if robust_result is not None else 'disabled'}`"
            f" | secilen degisken: `{len((robust_summary or {}).get('variables_selected', [])) if isinstance(robust_summary, dict) else 0}`"
        ),
        f"- İstenen modeller: `{', '.join(models)}`",
        f"- Başarılı: `{', '.join(models_ok) if models_ok else '-'}`",
        f"- Başarısız: `{', '.join(models_failed) if models_failed else '-'}`",
        f"- Sağlık suite: `{health_result.ok if health_result is not None else 'disabled'}`",
        "",
        "## Çalıştırma Sonuçları",
    ]
    for r in run_results:
        lines.append(f"- `{r.name}` | ok={r.ok} | code={r.returncode} | out={r.output_dir}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Model suite tamamlandı. Özet: {summary_json}")
    if summary["models_failed"]:
        print(f"Hata veren modeller: {', '.join(summary['models_failed'])}")
    else:
        print("Tüm istenen modeller başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
