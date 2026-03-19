#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from docx import Document


CAPTION_REPLACEMENTS = {
    "\u015eekil 2. \u0130stanbul toplam doluluk (ayl\u0131k) ve entegre veri d\u00f6nemi (2011\u20132023).":
        "\u015eekil 2. \u0130stanbul toplam doluluk (ayl\u0131k) ve new data ile ortak veri d\u00f6nemi (2011\u20132021).",
    "\u015eekil 3. Ya\u011f\u0131\u015f, ET0, t\u00fcketim ve toplam doluluk (y\u0131ll\u0131k, normalize).":
        "\u015eekil 3. Ya\u011f\u0131\u015f, ET0, t\u00fcketim ve toplam doluluk (ayl\u0131k, normalize, new data ya\u011f\u0131\u015f\u0131).",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update Istanbul dam academic DOCX with new-data graphs."
    )
    parser.add_argument(
        "--docx",
        type=Path,
        default=Path("output/doc/istanbul_baraj_durum_ozeti_akademik.docx"),
        help="Input DOCX to update in place unless --out is provided.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("output/doc/figures"),
        help="Directory containing refreshed figure PNG files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional output DOCX path. Defaults to in-place update.",
    )
    return parser.parse_args()


def update_paragraph_text(docx_path: Path, out_path: Path) -> None:
    document = Document(docx_path)
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text in CAPTION_REPLACEMENTS:
            paragraph.text = CAPTION_REPLACEMENTS[text]
    document.save(out_path)


def replace_embedded_images(docx_path: Path, figure_dir: Path) -> None:
    replacements = {
        "word/media/image3.png": figure_dir / "drivers_normalized.png",
        "word/media/image4.png": figure_dir / "sensitivity_bar.png",
    }

    with tempfile.TemporaryDirectory(prefix="istanbul_docx_") as tmp_dir:
        unpack_dir = Path(tmp_dir) / "unpacked"
        unpack_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(docx_path) as archive:
            archive.extractall(unpack_dir)

        for internal_name, source_path in replacements.items():
            target_path = unpack_dir / internal_name
            if not source_path.exists():
                raise FileNotFoundError(f"Missing figure: {source_path}")
            if not target_path.exists():
                raise FileNotFoundError(f"Missing embedded image slot: {target_path}")
            shutil.copyfile(source_path, target_path)

        rebuilt_path = Path(tmp_dir) / "rebuilt.docx"
        with zipfile.ZipFile(rebuilt_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in sorted(unpack_dir.rglob("*")):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(unpack_dir))

        shutil.copyfile(rebuilt_path, docx_path)


def main() -> None:
    args = parse_args()
    docx_path = args.docx.resolve()
    figure_dir = args.figure_dir.resolve()
    out_path = args.out.resolve() if args.out else docx_path

    with tempfile.TemporaryDirectory(prefix="istanbul_doc_edit_") as tmp_dir:
        intermediate_path = Path(tmp_dir) / "caption_updated.docx"
        update_paragraph_text(docx_path, intermediate_path)
        shutil.copyfile(intermediate_path, out_path)

    replace_embedded_images(out_path, figure_dir)
    print(out_path)


if __name__ == "__main__":
    main()
