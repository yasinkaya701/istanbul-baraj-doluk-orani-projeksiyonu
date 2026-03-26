#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path("/Users/yasinkaya/Hackhaton")


def build_model_js(csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    df = df.sort_values(["model", "date"])
    out = {}
    for model, group in df.groupby("model"):
        rows = []
        for _, row in group.iterrows():
            fill = None if pd.isna(row["fill_sim"]) else float(row["fill_sim"])
            rows.append({"date": str(row["date"]), "fill": fill})
        out[model] = rows
    return "window.MODEL_DATA = " + json.dumps(out, ensure_ascii=False) + ";\n"


def write_js(js_text: str) -> None:
    for path in [
        ROOT / "assets/data/projection_all_models.js",
        ROOT / "baraj_web/assets/data/projection_all_models.js",
    ]:
        path.write_text(js_text)


def copy_figures() -> None:
    img_targets = [ROOT / "assets/img", ROOT / "baraj_web/assets/img"]
    for target in img_targets:
        target.mkdir(parents=True, exist_ok=True)

    proj_fig_dir = ROOT / "output/istanbul_v3/figures"
    if proj_fig_dir.exists():
        for src in proj_fig_dir.glob("*_projection_*.png"):
            for target in img_targets:
                shutil.copy2(src, target / src.name)

    card_fig_dir = ROOT / "output/istanbul_model_cards_2026_03_18/figures"
    if card_fig_dir.exists():
        for src in card_fig_dir.glob("*_5y_10y.png"):
            for target in img_targets:
                shutil.copy2(src, target / src.name)


def main() -> None:
    csv_path = ROOT / "output/istanbul_gemini/data/projection_all.csv"
    if not csv_path.exists():
        csv_path = ROOT / "output/istanbul_gemini/projection_all.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")
    js_text = build_model_js(csv_path)
    write_js(js_text)
    copy_figures()
    print("projection_all_models.js updated and figures synced.")


if __name__ == "__main__":
    main()
