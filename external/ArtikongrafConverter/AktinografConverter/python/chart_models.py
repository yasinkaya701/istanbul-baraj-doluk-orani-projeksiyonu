from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ChartModel:
    name: str
    template_path: Path
    image_size: Tuple[int, int]  # (width, height)
    hour_start: int
    hour_slots: int
    hour_step_px: float
    left_hour_poly_y: Sequence[float]  # x = a*y^2 + b*y + c
    top_poly_x: Sequence[float]  # y = a*x^2 + b*x + c
    bottom_poly_x: Sequence[float]  # y = a*x^2 + b*x + c
    value_min: float
    value_max: float
    humidity_ratio_nodes: Optional[Sequence[float]] = None
    humidity_percent_nodes: Optional[Sequence[float]] = None

    def top_y(self, x: np.ndarray) -> np.ndarray:
        a, b, c = self.top_poly_x
        return a * x * x + b * x + c

    def bottom_y(self, x: np.ndarray) -> np.ndarray:
        a, b, c = self.bottom_poly_x
        return a * x * x + b * x + c

    def left_hour_x(self, y: np.ndarray) -> np.ndarray:
        a, b, c = self.left_hour_poly_y
        return a * y * y + b * y + c

    def hour_label_from_slot(self, slot_int: int) -> str:
        hour = self.hour_start + slot_int
        while hour > 24:
            hour -= 24
        while hour <= 0:
            hour += 24
        return str(hour)

    def hour_continuous(self, x: np.ndarray, y: np.ndarray, values: Optional[np.ndarray] = None) -> np.ndarray:
        raw_slot = (x - self.left_hour_x(y)) / self.hour_step_px
        if self.name == "radiation" and values is not None:
            raw_slot = raw_slot - self._radiation_hour_bias(values)
        return self.hour_start + raw_slot

    def value_from_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        top = self.top_y(x)
        bottom = self.bottom_y(x)
        den = np.maximum(bottom - top, 1e-6)

        if self.name == "humidity":
            ratio = np.clip((y - top) / den, 0.0, 1.0)
            return self._humidity_ratio_to_percent(ratio)

        # radiation: top=2.0, bottom=0.0
        ratio = np.clip((bottom - y) / den, 0.0, 1.0)
        return ratio * (self.value_max - self.value_min) + self.value_min

    def _humidity_ratio_to_percent(self, ratio: np.ndarray) -> np.ndarray:
        if self.humidity_ratio_nodes is None or self.humidity_percent_nodes is None:
            return ratio * 100.0
        x = np.asarray(self.humidity_ratio_nodes, dtype=np.float64)
        y = np.asarray(self.humidity_percent_nodes, dtype=np.float64)
        return np.interp(ratio, x, y)

    @staticmethod
    def _radiation_hour_bias(values: np.ndarray) -> np.ndarray:
        # Java tarafindaki duzeltmeyle ayni mantik:
        # 1.50 ustunde saat slotu saga kaydigi icin -1 saatlik bias uygulanir.
        bias = np.zeros_like(values, dtype=np.float64)
        lo = values > 1.50
        hi = values >= 1.60
        mid = lo & (~hi)
        bias[hi] = 1.0
        bias[mid] = (values[mid] - 1.50) / 0.10
        return bias


def get_models(project_root: Path) -> dict:
    def resolve_template(candidates: Iterable[Path]) -> Path:
        candidate_list = list(candidates)
        for p in candidate_list:
            if p.exists():
                return p
        # Keep first as fallback so error message still shows an expected path.
        first = candidate_list[0] if candidate_list else None
        if first is None:
            raise RuntimeError("No template candidate path provided.")
        return first

    def glob_existing_images(base: Path, patterns: Sequence[str]) -> Sequence[Path]:
        out = []
        for pat in patterns:
            out.extend(sorted(base.glob(pat)))
        # Deterministic order, unique paths.
        seen = set()
        uniq = []
        for p in out:
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)
            uniq.append(p)
        return uniq

    data_dir = project_root / "data"
    uploaded_dir = data_dir / "uploaded"

    humidity_template = resolve_template(
        list(
            [
                data_dir / "2010_MAYIS-01.jpg",
                data_dir / "2010_MAYIS-01.jpeg",
                data_dir / "2010_MAYIS-01.png",
                data_dir / "2010_MAYIS-01_trace_score.png",
            ]
        )
        + list(glob_existing_images(data_dir, ["2010_MAYIS-*.png", "2010_MAYIS-*.jpg", "2010_MAYIS-*.jpeg"]))
        + list(glob_existing_images(uploaded_dir, ["2010_MAYIS-*.png", "2010_MAYIS-*.jpg", "2010_MAYIS-*.jpeg"]))
        + list(glob_existing_images(uploaded_dir, ["*.png", "*.jpg", "*.jpeg"]))
    )
    radiation_template = resolve_template(
        list(
            [
                data_dir / "2001_MAYIS-01.png",
                data_dir / "2001_MAYIS-01.jpg",
                data_dir / "2001_MAYIS-01.jpeg",
                data_dir / "2001_MAYIS-01_trace_score.png",
            ]
        )
        + list(glob_existing_images(data_dir, ["2001_MAYIS-*.png", "2001_MAYIS-*.jpg", "2001_MAYIS-*.jpeg"]))
        + list(glob_existing_images(uploaded_dir, ["2001_MAYIS-*.png", "2001_MAYIS-*.jpg", "2001_MAYIS-*.jpeg"]))
        + list(glob_existing_images(uploaded_dir, ["*.png", "*.jpg", "*.jpeg"]))
    )

    return {
        "humidity": ChartModel(
            name="humidity",
            template_path=humidity_template,
            image_size=(3519, 1092),
            hour_start=8,
            hour_slots=24,
            hour_step_px=135.0,
            left_hour_poly_y=(3.12805062e-04, -3.76540763e-01, 1.86913365e+02),
            top_poly_x=(2.64445361e-06, -1.55707440e-02, 9.50308911e01),
            bottom_poly_x=(-9.49039587e-07, 5.79229774e-03, 1.04262061e03),
            value_min=0.0,
            value_max=100.0,
            humidity_ratio_nodes=(0.0, 0.180, 0.326, 0.438, 0.534, 0.615, 0.687, 0.757, 0.826, 0.900, 1.0),
            humidity_percent_nodes=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
        ),
        "radiation": ChartModel(
            name="radiation",
            template_path=radiation_template,
            image_size=(3553, 1086),
            hour_start=20,
            hour_slots=25,
            hour_step_px=131.0,
            left_hour_poly_y=(0.0, -5.08e-02, 1.2859e02),
            top_poly_x=(-1.11697085e-06, 4.93723763e-03, 1.27732177e02),
            bottom_poly_x=(1.16534230e-06, 9.08074336e-04, 9.01562199e02),
            value_min=0.0,
            value_max=2.0,
        ),
    }
