#!/usr/bin/env python3
"""Shared visual building blocks for ET0 explanation charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def theme() -> dict[str, str]:
    return {
        "fig_bg": "#f4efe6",
        "panel_bg": "#ecdfcf",
        "panel_edge": "#c8ae91",
        "chart_bg": "#fffdf9",
        "text": "#2c231d",
        "muted": "#685c52",
        "grid": "#ddd0c0",
        "spine": "#c9b49b",
        "card_bg": "#fbf6ee",
        "card_alt": "#f5ede1",
        "card_soft": "#f1e5d6",
        "card_blue": "#e6f0ee",
        "card_gold": "#f4ead8",
        "card_red": "#f7e8e1",
        "accent": "#a54934",
        "accent_2": "#225860",
        "accent_3": "#c8772f",
        "accent_soft": "#b79a78",
        "daily": "#c8772f",
        "daily_ma": "#7b3010",
        "monthly_bar": "#d6b07b",
        "monthly_line": "#225860",
        "monthly_aux": "#a54934",
        "season_fill": "#ede3d4",
        "real_marker": "#225860",
        "synthetic_marker": "#b79a78",
    }


def style_axes(ax: plt.Axes, colors: dict[str, str]) -> None:
    ax.set_facecolor(colors["chart_bg"])
    ax.grid(axis="y", color=colors["grid"], linewidth=0.8, alpha=0.9)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(colors["spine"])
    ax.tick_params(colors=colors["text"])


def setup_panel_axis(ax: plt.Axes, colors: dict[str, str]) -> None:
    ax.set_facecolor(colors["panel_bg"])
    for spine in ax.spines.values():
        spine.set_color(colors["panel_edge"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def add_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    *,
    colors: dict[str, str],
    facecolor: str,
    edgecolor: str | None = None,
    title_color: str | None = None,
    body_color: str | None = None,
    title_size: float = 10.8,
    body_size: float = 9.4,
    formula: str | None = None,
    formula_size: float = 15.0,
) -> None:
    edge = edgecolor or colors["panel_edge"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.028",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)

    pad_x = x + 0.03
    top_y = y + h - 0.035
    ax.text(
        pad_x,
        top_y,
        title,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=title_size,
        fontweight="bold",
        color=title_color or colors["text"],
        family="DejaVu Sans",
    )

    current_y = top_y - min(0.055, h * 0.35)
    if formula:
        ax.text(
            pad_x,
            current_y,
            formula,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=formula_size,
            color=colors["text"],
            family="DejaVu Serif",
        )
        current_y -= min(0.085, h * 0.52)

    if body:
        ax.text(
            pad_x,
            current_y,
            body,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=body_size,
            color=body_color or colors["text"],
            family="DejaVu Sans",
            linespacing=1.32,
        )


def render_et0_panel(
    ax: plt.Axes,
    *,
    context_title: str,
    context_lines: list[str],
    assumption_lines: list[str],
    summary_lines: list[str],
    source_lines: list[str],
    model_lines: list[str] | None = None,
) -> None:
    colors = theme()
    setup_panel_axis(ax, colors)

    formula = (
        r"$ET_0 = \frac{0.408\,\Delta\,(R_n-G)"
        r" + \gamma \left(\frac{900}{T+273}\right)u_2(e_s-e_a)}"
        r"{\Delta + \gamma(1+0.34u_2)}$"
    )
    reading_lines = [
        "- Enerji terimi: net enerji etkisi.",
        "- Aerodinamik terim: rüzgar + kuruluk etkisi.",
        "- Payda: iki etkiyi dengeler.",
    ]
    term_lines = [
        "- Rn: net enerji.",
        "- Delta: sıcaklık duyarlılığı.",
        "- es-ea: hava kuruluk açığı.",
        "- u2: 2 m rüzgar hızı.",
        "- gamma: denge sabiti.",
        "- G: günlükte 0 alındı.",
    ]
    assumption_body = "\n".join(f"- {line}" for line in assumption_lines[:4])
    reading_body = "\n".join(reading_lines)
    term_body = "\n".join(term_lines)
    context_body = "\n".join(context_lines)
    summary_body = "\n".join(f"- {line}" for line in summary_lines[:3] + source_lines[:1])
    if model_lines:
        summary_body = "\n".join(f"- {line}" for line in summary_lines[:2] + model_lines[:1] + source_lines[:1])

    add_card(
        ax,
        0.04,
        0.885,
        0.92,
        0.085,
        context_title,
        context_body,
        colors=colors,
        facecolor=colors["card_alt"],
        title_size=12.2,
        body_size=8.8,
    )
    add_card(
        ax,
        0.04,
        0.705,
        0.92,
        0.16,
        "FAO-56 Penman-Monteith",
        "Gunluk referans evapotranspirasyonu (ET0).",
        colors=colors,
        facecolor=colors["card_bg"],
        formula=formula,
        formula_size=13.8,
        body_size=8.6,
    )
    add_card(
        ax,
        0.04,
        0.565,
        0.92,
        0.115,
        "Formül Nasıl Okunur",
        reading_body,
        colors=colors,
        facecolor=colors["card_blue"],
        body_size=8.5,
    )
    add_card(
        ax,
        0.04,
        0.39,
        0.92,
        0.145,
        "Terimler Ne İşe Yarar",
        term_body,
        colors=colors,
        facecolor=colors["card_soft"],
        body_size=8.35,
    )
    add_card(
        ax,
        0.04,
        0.215,
        0.92,
        0.155,
        "Kabuller ve Nedenleri",
        assumption_body,
        colors=colors,
        facecolor=colors["card_gold"],
        body_size=8.45,
    )
    add_card(
        ax,
        0.04,
        0.02,
        0.92,
        0.17,
        "Bu Grafikte",
        summary_body,
        colors=colors,
        facecolor=colors["card_red"],
        body_size=8.25,
    )


def build_formula_card(out_path: Path) -> None:
    colors = theme()
    fig = plt.figure(figsize=(13.6, 9.6), facecolor=colors["fig_bg"])
    ax = fig.add_subplot(111)
    setup_panel_axis(ax, colors)
    formula = (
        r"$ET_0 = \frac{0.408\,\Delta\,(R_n-G)"
        r" + \gamma \left(\frac{900}{T+273}\right)u_2(e_s-e_a)}"
        r"{\Delta + \gamma(1+0.34u_2)}$"
    )
    add_card(
        ax,
        0.04,
        0.885,
        0.92,
        0.085,
        "ET0 Formül Rehberi",
        "Referans yüzey: iyi sulanmış kısa çimen\nZaman adımı: günlük",
        colors=colors,
        facecolor=colors["card_alt"],
        title_size=12.8,
        body_size=9.2,
    )
    add_card(
        ax,
        0.04,
        0.69,
        0.92,
        0.17,
        "FAO-56 Penman-Monteith",
        "Bu formül günlük referans evapotranspirasyonu (ET0) verir.",
        colors=colors,
        facecolor=colors["card_bg"],
        formula=formula,
        formula_size=14.4,
        body_size=8.8,
    )
    add_card(
        ax,
        0.04,
        0.515,
        0.92,
        0.145,
        "Formül Nasıl Okunur",
        "\n".join(
            [
                "- Enerji terimi: net radyasyonun ET etkisi.",
                "- Aerodinamik terim: rüzgar + kuruluk etkisi.",
                "- Payda: iki mekanizmayı dengeler.",
                "- Sonuç: ET0 mm/gün olarak çıkar.",
            ]
        ),
        colors=colors,
        facecolor=colors["card_blue"],
        body_size=8.8,
    )
    add_card(
        ax,
        0.04,
        0.31,
        0.92,
        0.175,
        "Terimler Ne İşe Yarar",
        "\n".join(
            [
                "- Rn: ET için kullanılabilir net enerji.",
                "- Delta: sıcaklığın buharlaşma hassasiyeti.",
                "- es-ea: havanın kuruluk açığı.",
                "- u2: 2 m rüzgar hızı.",
                "- gamma: enerji ve hava terimini dengeler.",
                "- G: zemine giden ısı; günlükte 0 alınır.",
            ]
        ),
        colors=colors,
        facecolor=colors["card_soft"],
        body_size=8.8,
    )
    add_card(
        ax,
        0.04,
        0.12,
        0.92,
        0.16,
        "Kabuller ve Nedenleri",
        "\n".join(
            [
                "- Tmean = (Tmax + Tmin) / 2 → veriyle en tutarlı özet.",
                "- Delta = f(Tmean) → Delta, Tmean’in kendisi değil.",
                "- G = 0 → günlük ET0 için standart kabul.",
                "- u2 = 2.0 m/s → eksik rüzgar için fallback.",
                "- Rs dosyadan gelir → net radyasyonun ana girdisi.",
            ]
        ),
        colors=colors,
        facecolor=colors["card_gold"],
        body_size=8.75,
    )
    add_card(
        ax,
        0.04,
        0.02,
        0.92,
        0.08,
        "Ne İşe Yarar",
        "- Buharlaşma talebini temsil eder.\n- ET0, su dengesi ve ETc hesaplarının temelidir.",
        colors=colors,
        facecolor=colors["card_red"],
        body_size=8.8,
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
