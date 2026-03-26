#!/usr/bin/env python3
"""One-command data factory for hackathon climate datasets.

Stages:
1) Numeric ingest from spreadsheet/tabular inputs
2) Visual digitization from graph-paper images
3) Optional quant model run
4) Delivery manifest + package zip
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def to_bool(x: Any) -> bool:
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run end-to-end data factory for climate hackathon.")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/DATA"),
        help="Root dataset directory that contains numeric + graph folders.",
    )
    p.add_argument(
        "--numeric-dir",
        type=Path,
        default=None,
        help="Numeric input directory. Default: <dataset-root>/Sayısallaştırılmış Veri",
    )
    p.add_argument(
        "--graph-root",
        type=Path,
        default=None,
        help="Graph image root directory. Default: <dataset-root>/Graf Kağıtları Tarama",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/data_factory"),
        help="Base output directory for all artifacts.",
    )
    p.add_argument(
        "--run-quant",
        type=str,
        default="false",
        help="Run quant model after data preparation (true/false).",
    )
    p.add_argument(
        "--variables",
        type=str,
        default="temp,humidity,pressure,precip",
        help="Variables passed to quant model.",
    )
    p.add_argument("--target-year", type=int, default=2035, help="Quant forecast horizon year.")
    p.add_argument("--max-image-files", type=int, default=0, help="Debug cap for visual processing (0=all).")
    p.add_argument(
        "--image-exts",
        type=str,
        default=".tif,.tiff,.png,.jpg,.jpeg,.bmp,.webp",
        help="Image extensions for visual processing.",
    )
    return p.parse_args()


@dataclass
class StepResult:
    name: str
    ok: bool
    code: int
    cmd: list[str]
    started_at: str
    ended_at: str
    out_dir: str


def run_cmd(name: str, cmd: list[str], out_dir: Path) -> StepResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    env = dict(os.environ)
    mpl_cfg = out_dir / "mplconfig"
    mpl_cfg.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(mpl_cfg))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    proc = subprocess.run(cmd, cwd="/Users/yasinkaya/Hackhaton", text=True, capture_output=True, env=env)
    ended = datetime.now().isoformat(timespec="seconds")

    (out_dir / f"{name}.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / f"{name}.stderr.log").write_text(proc.stderr or "", encoding="utf-8")

    return StepResult(
        name=name,
        ok=(proc.returncode == 0),
        code=int(proc.returncode),
        cmd=cmd,
        started_at=started,
        ended_at=ended,
        out_dir=str(out_dir),
    )


def default_dir_if_none(value: Path | None, fallback: Path) -> Path:
    return value if value is not None else fallback


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_markdown_report(
    out_path: Path,
    args: argparse.Namespace,
    steps: list[StepResult],
    visual_summary: dict[str, Any],
    quant_dir: Path | None,
    zip_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Data Factory Calisma Ozeti")
    lines.append("")
    lines.append(f"- Tarih: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Dataset root: `{args.dataset_root}`")
    lines.append(f"- Numeric dir: `{args.numeric_dir}`")
    lines.append(f"- Graph root: `{args.graph_root}`")
    lines.append(f"- Output root: `{args.output_root}`")
    lines.append(f"- Run quant: `{args.run_quant}`")
    lines.append("")
    lines.append("## Adim Sonuclari")
    lines.append("")
    lines.append("|adim|ok|code|start|end|")
    lines.append("|---|---:|---:|---|---|")
    for s in steps:
        lines.append(f"|{s.name}|{s.ok}|{s.code}|{s.started_at}|{s.ended_at}|")
    lines.append("")

    if visual_summary:
        lines.append("## Gorsel Isleme Ozeti")
        lines.append("")
        for k in [
            "image_files_total",
            "visual_rows_all",
            "visual_rows_model_vars",
            "numeric_rows_all",
            "numeric_rows_model_vars",
            "silver_rows_all",
            "gold_rows_model_vars",
            "extract_success_count",
            "extract_fail_count",
        ]:
            if k in visual_summary:
                lines.append(f"- {k}: `{visual_summary[k]}`")
        outputs = visual_summary.get("outputs", {})
        if isinstance(outputs, dict):
            lines.append("")
            lines.append("### Cikti Dosyalari")
            lines.append("")
            for k, v in outputs.items():
                lines.append(f"- {k}: `{v}`")
        lines.append("")

    if quant_dir is not None:
        lines.append("## Quant Ciktilari")
        lines.append("")
        lines.append(f"- Quant klasoru: `{quant_dir}`")
        lines.append("")

    lines.append("## Paket")
    lines.append("")
    lines.append(f"- ZIP: `{zip_path}`")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def zip_tree(zip_path: Path, include_paths: list[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in include_paths:
            if not p.exists():
                continue
            if p.is_file():
                zf.write(p, arcname=p.name)
                continue
            for x in sorted(p.rglob("*")):
                if x.is_file():
                    zf.write(x, arcname=str(x.relative_to(p.parent)))


def main() -> None:
    args = parse_args()

    args.numeric_dir = default_dir_if_none(args.numeric_dir, args.dataset_root / "Sayısallaştırılmış Veri")
    args.graph_root = default_dir_if_none(args.graph_root, args.dataset_root / "Graf Kağıtları Tarama ")
    args.output_root.mkdir(parents=True, exist_ok=True)

    run_dir = args.output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # 1) Numeric ingest
    numeric_out = run_dir / "numeric"
    step1 = run_cmd(
        "ingest_numeric",
        [
            py,
            "scripts/ingest_numeric_and_plot.py",
            "--data-dir",
            str(args.numeric_dir),
            "--output-dir",
            str(numeric_out),
        ],
        logs_dir,
    )
    if not step1.ok:
        raise SystemExit(
            f"[ingest_numeric] failed with code={step1.code}. "
            f"See logs: {logs_dir / 'ingest_numeric.stderr.log'}"
        )

    numeric_parquet = numeric_out / "observations_numeric.parquet"
    if not numeric_parquet.exists():
        raise SystemExit(f"numeric parquet missing: {numeric_parquet}")

    # 2) Visual digitization + silver/gold datasets
    prepared_out = run_dir / "prepared"
    step2 = run_cmd(
        "process_visuals",
        [
            py,
            "scripts/process_all_visuals_to_quant.py",
            "--graph-root",
            str(args.graph_root),
            "--numeric-parquet",
            str(numeric_parquet),
            "--output-dir",
            str(prepared_out),
            "--max-files",
            str(args.max_image_files),
            "--image-exts",
            str(args.image_exts),
        ],
        logs_dir,
    )
    if not step2.ok:
        raise SystemExit(
            f"[process_visuals] failed with code={step2.code}. "
            f"See logs: {logs_dir / 'process_visuals.stderr.log'}"
        )

    gold_parquet = prepared_out / "observations_with_all_visuals_for_quant.parquet"
    if not gold_parquet.exists():
        raise SystemExit(f"gold parquet missing: {gold_parquet}")

    # 3) Optional quant run
    steps = [step1, step2]
    quant_dir: Path | None = None
    if to_bool(args.run_quant):
        quant_dir = run_dir / "quant"
        step3 = run_cmd(
            "quant_model",
            [
                py,
                "scripts/quant_regime_projection.py",
                "--observations",
                str(gold_parquet),
                "--output-dir",
                str(quant_dir),
                "--variables",
                str(args.variables),
                "--target-year",
                str(args.target_year),
                "--backtest-splits",
                "3",
                "--holdout-steps",
                "12",
                "--min-train-steps",
                "36",
                "--vol-model",
                "egarch",
                "--egarch-p",
                "1",
                "--egarch-o",
                "1",
                "--egarch-q",
                "1",
                "--egarch-dist",
                "t",
                "--regime-k",
                "2",
                "--regime-maxiter",
                "200",
                "--interval-alpha",
                "0.10",
                "--anomaly-z",
                "2.5",
                "--anomaly-top",
                "20",
            ],
            logs_dir,
        )
        steps.append(step3)
        if not step3.ok:
            raise SystemExit(
                f"[quant_model] failed with code={step3.code}. "
                f"See logs: {logs_dir / 'quant_model.stderr.log'}"
            )

    visual_summary = load_json(prepared_out / "summary.json")

    # 4) Delivery report + zip
    report_md = run_dir / "data_factory_report.md"
    zip_path = run_dir / "data_factory_package.zip"
    write_markdown_report(report_md, args, steps, visual_summary, quant_dir, zip_path)

    include: list[Path] = [report_md, logs_dir, numeric_out, prepared_out]
    if quant_dir is not None:
        include.append(quant_dir)
    zip_tree(zip_path, include)

    run_summary = {
        "run_dir": str(run_dir),
        "report_md": str(report_md),
        "zip_package": str(zip_path),
        "gold_observations_parquet": str(gold_parquet),
        "gold_observations_csv": str(prepared_out / "observations_with_all_visuals_for_quant.csv"),
        "silver_observations_parquet": str(prepared_out / "observations_lossless_silver.parquet"),
        "bronze_manifest_csv": str(prepared_out / "visual_process_report.csv"),
        "steps": [s.__dict__ for s in steps],
    }
    (run_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
