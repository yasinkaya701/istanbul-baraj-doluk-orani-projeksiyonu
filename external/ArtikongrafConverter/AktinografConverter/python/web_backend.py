#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import re
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

import extract_daily_series as extractor
from chart_models import ChartModel, get_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploaded"
OUTPUT_DIR = PROJECT_ROOT / "output"
DIFF_DIR = PROJECT_ROOT / "diff"
AUTO_OUTPUT_DIR = OUTPUT_DIR / "auto"
AUTO_DIFF_DIR = DIFF_DIR / "auto"
ALIGNED_DIR = OUTPUT_DIR / "aligned"
DB_PATH = OUTPUT_DIR / "webapp.sqlite3"

CSV_HEADERS = ["source_file", "x", "y", "hour_precise", "radiation"]


@dataclass
class ProcessResult:
    dataset_id: int
    ok: bool
    message: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sanitize_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^A-Za-z0-9_.\-]+", "_", base).strip("._")
    return base or "uploaded.png"


def _normalize_tiff_page_for_png(im: Image.Image) -> Image.Image:
    im = ImageOps.exif_transpose(im)
    mode = getattr(im, "mode", "")

    if mode == "P":
        return im.convert("RGBA")
    if mode == "1":
        return im.convert("L")
    if mode in {"L", "LA", "RGB", "RGBA"}:
        return im

    if mode.startswith("I;16") or mode in {"I", "F"}:
        arr = np.asarray(im, dtype=np.float32)
        finite = np.isfinite(arr)
        if not np.any(finite):
            arr8 = np.zeros(arr.shape, dtype=np.uint8)
        else:
            lo = float(np.nanpercentile(arr[finite], 1.0))
            hi = float(np.nanpercentile(arr[finite], 99.0))
            if hi <= lo + 1e-9:
                hi = lo + 1.0
            scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
            arr8 = np.rint(scaled * 255.0).astype(np.uint8)
        return Image.fromarray(arr8, mode="L")

    return im.convert("RGB")


def _maybe_convert_tiff_to_png(content: bytes, src_name: str, ext: str) -> Tuple[bytes, str, str]:
    # Returns: content_bytes, output_ext, display_name
    if ext not in {".tif", ".tiff"}:
        return content, ext, src_name

    out_name = f"{Path(src_name).stem}.png"
    errors: List[str] = []

    try:
        with Image.open(io.BytesIO(content)) as im:
            if int(getattr(im, "n_frames", 1) or 1) > 1:
                im.seek(0)
            im = _normalize_tiff_page_for_png(im)
            if im.mode == "RGBA":
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            elif im.mode != "RGB":
                im = im.convert("RGB")
            out = io.BytesIO()
            im.save(out, format="PNG")
            return out.getvalue(), ".png", out_name
    except Exception as e:
        errors.append(f"Pillow: {e}")

    try:
        arr = np.frombuffer(content, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("OpenCV decode sonucu bos")
        if decoded.ndim == 3 and decoded.shape[2] == 4:
            alpha = decoded[:, :, 3:4].astype(np.float32) / 255.0
            rgb = decoded[:, :, :3].astype(np.float32)
            bg = np.full_like(rgb, 255.0)
            decoded = np.rint(rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        ok, enc = cv2.imencode(".png", decoded)
        if not ok:
            raise ValueError("OpenCV PNG encode basarisiz")
        return enc.tobytes(), ".png", out_name
    except Exception as e:
        errors.append(f"OpenCV: {e}")

    raise ValueError(f"TIFF PNG'e donusturulemedi ({src_name}). " + " | ".join(errors))


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIFF_DIR.mkdir(parents=True, exist_ok=True)
    AUTO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUTO_DIFF_DIR.mkdir(parents=True, exist_ok=True)
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)


def connect_db() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                stem TEXT NOT NULL UNIQUE,
                stored_path TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                processed INTEGER NOT NULL DEFAULT 0,
                processed_at TEXT,
                has_manual_edit INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'imported',
                last_error TEXT NOT NULL DEFAULT '',
                spike_margin_int INTEGER NOT NULL DEFAULT 0,
                auto_csv_path TEXT,
                current_csv_path TEXT,
                auto_diff_path TEXT,
                current_diff_path TEXT,
                aligned_image_path TEXT
            )
            """
        )
        conn.commit()


def _models() -> Dict[str, ChartModel]:
    return get_models(PROJECT_ROOT)


def list_datasets() -> List[Dict[str, object]]:
    init_db()
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT * FROM datasets ORDER BY imported_at DESC, id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_dataset(dataset_id: int) -> Optional[Dict[str, object]]:
    init_db()
    with connect_db() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    return None if row is None else dict(row)


def import_uploaded_files(uploaded_files: Sequence[object]) -> List[int]:
    init_db()
    created_ids: List[int] = []
    errors: List[str] = []

    with connect_db() as conn:
        for uf in uploaded_files:
            if uf is None:
                continue
            src_name = _sanitize_filename(getattr(uf, "name", "uploaded.png"))
            try:
                ext = Path(src_name).suffix.lower()
                if ext not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                    errors.append(f"{src_name}: desteklenmeyen uzanti")
                    continue

                stem = f"{Path(src_name).stem}_{uuid.uuid4().hex[:8]}"
                stored_name = f"{stem}{ext}"
                stored_path = UPLOAD_DIR / stored_name

                if hasattr(uf, "getbuffer"):
                    content = bytes(uf.getbuffer())
                elif hasattr(uf, "read"):
                    content = uf.read()
                else:
                    content = bytes(uf)

                content, out_ext, display_name = _maybe_convert_tiff_to_png(content, src_name, ext)

                if out_ext != ext:
                    stored_name = f"{stem}{out_ext}"
                    stored_path = UPLOAD_DIR / stored_name
                stored_path.write_bytes(content)

                cur = conn.execute(
                    """
                    INSERT INTO datasets (
                        source_name, stem, stored_path, imported_at,
                        processed, processed_at, has_manual_edit, status, last_error, spike_margin_int
                    ) VALUES (?, ?, ?, ?, 0, NULL, 0, 'imported', '', 0)
                    """,
                    (display_name, stem, str(stored_path), _now_iso()),
                )
                created_ids.append(int(cur.lastrowid))
            except Exception as e:
                errors.append(f"{src_name}: {e}")
        conn.commit()

    if not created_ids and errors:
        raise RuntimeError("Import basarisiz:\n" + "\n".join(errors[:8]))

    return created_ids


def _hour_precise_to_float(hour_precise: str) -> float:
    hs = str(hour_precise).strip()
    if "." not in hs:
        return float(hs)
    hh, mm = hs.split(".", 1)
    return float(int(hh) + int(mm) / 60.0)


def _hour_wrap_to_label(hour_cont: float) -> str:
    h = int(np.floor(((hour_cont - 1.0) % 24.0) + 1.0 + 1e-9))
    return str(h)


def _radiation_to_y(model: ChartModel, x: np.ndarray, radiation: np.ndarray) -> np.ndarray:
    top = model.top_y(x)
    bottom = model.bottom_y(x)
    ratio = np.clip((radiation - model.value_min) / max(model.value_max - model.value_min, 1e-6), 0.0, 1.0)
    return bottom - ratio * (bottom - top)


def _rows_from_points(source_file: str, points: Sequence[Tuple[float, float, float, str, str, float]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for x, y, _hcont, _hlbl, hprec, val in points:
        rows.append(
            {
                "source_file": source_file,
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "hour_precise": str(hprec),
                "radiation": round(float(val), 4),
            }
        )
    return rows


def _points_from_rows(rows: Sequence[Dict[str, object]]) -> List[Tuple[float, float, float, str, str, float]]:
    pts: List[Tuple[float, float, float, str, str, float]] = []
    for i, r in enumerate(rows):
        h_cont = 20.0 + (i / 60.0)
        hour_precise = str(r["hour_precise"])
        pts.append(
            (
                float(r["x"]),
                float(r["y"]),
                h_cont,
                _hour_wrap_to_label(h_cont),
                hour_precise,
                float(r["radiation"]),
            )
        )
    return pts


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], headers: Sequence[str] = CSV_HEADERS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in headers})


def _read_csv_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _align_image_for_editor(source_path: Path, model: ChartModel, out_path: Path) -> Optional[Path]:
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    template = cv2.imread(str(model.template_path), cv2.IMREAD_COLOR)
    if image is None or template is None:
        return None
    aligned, _cc = extractor.load_and_align_to_template(image, template)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), aligned)
    return out_path


def process_dataset(dataset_id: int, spike_margin_int: int = 0) -> ProcessResult:
    init_db()
    row = get_dataset(dataset_id)
    if row is None:
        return ProcessResult(dataset_id=dataset_id, ok=False, message=f"Dataset {dataset_id} not found.")

    source_path = Path(str(row["stored_path"]))
    if not source_path.exists():
        with connect_db() as conn:
            conn.execute(
                "UPDATE datasets SET status='error', last_error=? WHERE id=?",
                (f"Source image not found: {source_path}", dataset_id),
            )
            conn.commit()
        return ProcessResult(dataset_id=dataset_id, ok=False, message=f"Source image not found: {source_path}")

    try:
        models = _models()
        model = models["radiation"]
        result = extractor.process_file(source_path, models, None, int(spike_margin_int))

        rows = _rows_from_points(str(row["source_name"]), result.points)
        current_csv = OUTPUT_DIR / f"{row['stem']}_series.csv"
        auto_csv = AUTO_OUTPUT_DIR / f"{row['stem']}_series_auto.csv"
        current_diff = DIFF_DIR / f"{row['stem']}_diff.png"
        auto_diff = AUTO_DIFF_DIR / f"{row['stem']}_diff_auto.png"
        aligned_png = ALIGNED_DIR / f"{row['stem']}_aligned.png"

        _write_csv(current_csv, rows, CSV_HEADERS)
        _write_csv(auto_csv, rows, CSV_HEADERS)
        extractor.save_diff_panel(source_path, result.points, model, current_diff)
        extractor.save_diff_panel(source_path, result.points, model, auto_diff)
        _align_image_for_editor(source_path, model, aligned_png)

        with connect_db() as conn:
            conn.execute(
                """
                UPDATE datasets
                SET processed=1,
                    processed_at=?,
                    has_manual_edit=0,
                    status='processed',
                    last_error='',
                    spike_margin_int=?,
                    auto_csv_path=?,
                    current_csv_path=?,
                    auto_diff_path=?,
                    current_diff_path=?,
                    aligned_image_path=?
                WHERE id=?
                """,
                (
                    _now_iso(),
                    int(spike_margin_int),
                    str(auto_csv),
                    str(current_csv),
                    str(auto_diff),
                    str(current_diff),
                    str(aligned_png),
                    dataset_id,
                ),
            )
            conn.commit()

        rebuild_all_series()
        return ProcessResult(dataset_id=dataset_id, ok=True, message=f"Processed: {row['source_name']}")
    except Exception as e:
        with connect_db() as conn:
            conn.execute(
                "UPDATE datasets SET status='error', last_error=? WHERE id=?",
                (str(e), dataset_id),
            )
            conn.commit()
        return ProcessResult(dataset_id=dataset_id, ok=False, message=f"Process error: {e}")


def process_many(dataset_ids: Iterable[int], spike_margin_int: int = 0) -> List[ProcessResult]:
    results: List[ProcessResult] = []
    for did in dataset_ids:
        results.append(process_dataset(int(did), spike_margin_int=spike_margin_int))
    return results


def rebuild_all_series() -> None:
    init_db()
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT current_csv_path
            FROM datasets
            WHERE processed = 1 AND current_csv_path IS NOT NULL AND current_csv_path != ''
            ORDER BY id ASC
            """
        ).fetchall()

    all_rows: List[Dict[str, object]] = []
    for r in rows:
        p = Path(str(r["current_csv_path"]))
        if not p.exists():
            continue
        all_rows.extend(_read_csv_rows(p))

    if all_rows:
        _write_csv(OUTPUT_DIR / "all_series.csv", all_rows, CSV_HEADERS)
    else:
        (OUTPUT_DIR / "all_series.csv").unlink(missing_ok=True)


def delete_dataset(dataset_id: int) -> bool:
    init_db()
    row = get_dataset(dataset_id)
    if row is None:
        return False

    delete_paths = [
        Path(str(row["stored_path"])) if row.get("stored_path") else None,
        Path(str(row["auto_csv_path"])) if row.get("auto_csv_path") else None,
        Path(str(row["current_csv_path"])) if row.get("current_csv_path") else None,
        Path(str(row["auto_diff_path"])) if row.get("auto_diff_path") else None,
        Path(str(row["current_diff_path"])) if row.get("current_diff_path") else None,
        Path(str(row["aligned_image_path"])) if row.get("aligned_image_path") else None,
    ]

    with connect_db() as conn:
        conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        conn.commit()

    for p in delete_paths:
        if p is None:
            continue
        p.unlink(missing_ok=True)

    rebuild_all_series()
    return True


def get_series_rows(dataset_id: int, use_auto: bool = False) -> List[Dict[str, object]]:
    row = get_dataset(dataset_id)
    if row is None:
        return []
    key = "auto_csv_path" if use_auto else "current_csv_path"
    p_str = row.get(key)
    if not p_str:
        return []
    return _read_csv_rows(Path(str(p_str)))


def apply_anchor_edits(
    dataset_id: int,
    base_rows: Sequence[Dict[str, object]],
    anchors: Sequence[Tuple[int, float]],
) -> List[Dict[str, object]]:
    row = get_dataset(dataset_id)
    if row is None:
        return [dict(r) for r in base_rows]

    models = _models()
    model = models["radiation"]

    out_rows = [dict(r) for r in base_rows]
    if not out_rows:
        return out_rows

    x = np.asarray([float(r["x"]) for r in out_rows], dtype=np.float64)
    rad = np.asarray([float(r["radiation"]) for r in out_rows], dtype=np.float64)

    latest_by_idx: Dict[int, float] = {}
    for idx, val in anchors:
        ii = int(idx)
        if 0 <= ii < rad.size:
            latest_by_idx[ii] = float(np.clip(val, 0.0, 2.0))

    if latest_by_idx:
        idxs = sorted(latest_by_idx.keys())
        for ii in idxs:
            rad[ii] = latest_by_idx[ii]
        for a, b in zip(idxs, idxs[1:]):
            if b > a:
                rad[a : b + 1] = np.linspace(rad[a], rad[b], b - a + 1)

    # Preserve day window rule.
    for i, rr in enumerate(out_rows):
        hh = _hour_precise_to_float(str(rr["hour_precise"]))
        if not (4.0 <= hh < 20.0):
            rad[i] = 0.0

    y = _radiation_to_y(model, x, rad)

    for i, rr in enumerate(out_rows):
        rr["x"] = f"{x[i]:.3f}"
        rr["y"] = f"{y[i]:.3f}"
        rr["radiation"] = f"{float(np.clip(rad[i], 0.0, 2.0)):.4f}"

    return out_rows


def save_manual_edit(dataset_id: int, rows: Sequence[Dict[str, object]]) -> ProcessResult:
    row = get_dataset(dataset_id)
    if row is None:
        return ProcessResult(dataset_id=dataset_id, ok=False, message=f"Dataset {dataset_id} not found.")

    current_csv = Path(str(row["current_csv_path"])) if row.get("current_csv_path") else OUTPUT_DIR / f"{row['stem']}_series.csv"
    _write_csv(current_csv, rows, CSV_HEADERS)

    source_path = Path(str(row["stored_path"]))
    model = _models()["radiation"]
    points = _points_from_rows(rows)
    current_diff = Path(str(row["current_diff_path"])) if row.get("current_diff_path") else DIFF_DIR / f"{row['stem']}_diff.png"
    extractor.save_diff_panel(source_path, points, model, current_diff)

    with connect_db() as conn:
        conn.execute(
            """
            UPDATE datasets
            SET processed=1,
                processed_at=?,
                has_manual_edit=1,
                status='processed',
                last_error='',
                current_csv_path=?,
                current_diff_path=?
            WHERE id=?
            """,
            (_now_iso(), str(current_csv), str(current_diff), dataset_id),
        )
        conn.commit()

    rebuild_all_series()
    return ProcessResult(dataset_id=dataset_id, ok=True, message=f"Saved manual edit: {row['source_name']}")


def ensure_editor_assets(dataset_id: int) -> Optional[Path]:
    row = get_dataset(dataset_id)
    if row is None:
        return None
    source_path = Path(str(row["stored_path"]))
    if not source_path.exists():
        return None

    aligned_str = row.get("aligned_image_path")
    if aligned_str:
        aligned = Path(str(aligned_str))
        if aligned.exists():
            return aligned

    aligned = ALIGNED_DIR / f"{row['stem']}_aligned.png"
    model = _models()["radiation"]
    out = _align_image_for_editor(source_path, model, aligned)
    if out is None:
        return None

    with connect_db() as conn:
        conn.execute("UPDATE datasets SET aligned_image_path=? WHERE id=?", (str(out), dataset_id))
        conn.commit()
    return out


def clear_all_outputs() -> None:
    # For future CLI/admin button: remove generated artifacts, keep DB entries as imported.
    for p in [OUTPUT_DIR, DIFF_DIR]:
        if p.exists():
            shutil.rmtree(p)
    _ensure_dirs()
    with connect_db() as conn:
        conn.execute(
            """
            UPDATE datasets
            SET processed=0,
                processed_at=NULL,
                has_manual_edit=0,
                status='imported',
                last_error='',
                auto_csv_path=NULL,
                current_csv_path=NULL,
                auto_diff_path=NULL,
                current_diff_path=NULL,
                aligned_image_path=NULL
            """
        )
        conn.commit()
