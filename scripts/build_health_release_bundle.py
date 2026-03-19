#!/usr/bin/env python3
"""Build full health presentation release bundle for a given date label."""

from __future__ import annotations

import argparse
from datetime import date
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build health presentation release bundle")
    p.add_argument("--root-dir", type=Path, default=Path("output/health_impact"))
    p.add_argument("--fig-dir", type=Path, default=Path("output/health_impact/figures"))
    p.add_argument("--date-label", type=str, default=str(date.today()))
    p.add_argument("--logo-path", type=Path, default=None)
    p.add_argument("--python-bin", type=str, default="python3")
    p.add_argument(
        "--pdf-quality-target",
        choices=["any", "visual"],
        default="any",
        help="any allows fallback_text PDF; visual requires soffice/keynote visual conversion.",
    )
    p.add_argument(
        "--require-visual-pdf",
        action="store_true",
        help="Pass strict visual-PDF requirement to QC (fails on fallback_text).",
    )
    return p.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    cp = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}")


def maybe_copy(src: Path, dst: Path) -> None:
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


def copy_replace(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def has_visual_pdf_tool() -> bool:
    return bool(shutil.which("soffice") or shutil.which("libreoffice"))


def resolve_model_summary_path(root: Path) -> Path | None:
    for p in [
        root / "model_comparison_summary_stable_calibrated.csv",
        root / "model_comparison_summary_duzenlenmis_run.csv",
        root / "model_comparison_summary.csv",
    ]:
        if p.exists():
            return p
    return None


def load_pdf_conversion_summary(root: Path, date_label: str) -> list[str]:
    report_files = [
        root / f"sunum_6_slayt_{date_label}_v3_pdf_conversion.json",
        root / f"sunum_10_slayt_{date_label}_v4_detailed_pdf_conversion.json",
        root / f"sunum_13_slayt_{date_label}_v6_board_pdf_conversion.json",
    ]
    lines: list[str] = []
    for p in report_files:
        if not p.exists():
            lines.append(f"- {p.name}: rapor bulunamadi")
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            mode = str(data.get("mode", "unknown"))
            size = int(data.get("output_size_bytes", 0))
            slides = int(data.get("slide_count", 0))
            lines.append(f"- {p.name}: mode={mode}, slides={slides}, pdf_size={size} bytes")
        except Exception as e:
            lines.append(f"- {p.name}: rapor okuma hatasi ({e})")
    return lines


def load_stability_summary(root: Path, date_label: str) -> list[str]:
    diag_csv = root / f"model_stability_diagnostics_{date_label}.csv"
    if not diag_csv.exists():
        return ["- model_stability_diagnostics raporu bulunamadi."]
    try:
        d = pd.read_csv(diag_csv)
    except Exception as e:
        return [f"- model_stability_diagnostics okuma hatasi: {e}"]
    if d.empty:
        return ["- model_stability_diagnostics bos."]
    lines: list[str] = []
    for _, r in d.iterrows():
        lines.append(
            f"- {str(r.get('model_key', r.get('model', 'model')))}: "
            f"RR {float(r.get('orig_future_rr_mean', 0.0)):.4f} -> {float(r.get('calibrated_future_rr_mean', 0.0)):.4f}, "
            f"shrink={float(r.get('stability_calibration_shrink', 0.0)):.2f}, "
            f"stability={float(r.get('stability_score', 0.0)):.2f}, "
            f"confidence={str(r.get('prediction_confidence', 'na'))}"
        )
    return lines


def load_skin_focus_summary(root: Path, date_label: str) -> list[str]:
    candidates = [
        root / f"cilt_kanseri_odak_model_tablosu_{date_label}.csv",
        root / "cilt_kanseri_odak_model_tablosu_latest.csv",
    ]
    model_csv = None
    for p in candidates:
        if p.exists():
            model_csv = p
            break
    if model_csv is None:
        return ["- cilt_kanseri_odak_model_tablosu bulunamadi."]
    try:
        d = pd.read_csv(model_csv)
    except Exception as e:
        return [f"- cilt_kanseri_odak_model_tablosu okuma hatasi: {e}"]
    if d.empty:
        return ["- cilt_kanseri_odak_model_tablosu bos."]
    lines: list[str] = []
    for _, r in d.iterrows():
        lines.append(
            f"- {str(r.get('model_key', 'model'))}: "
            f"proxy_index={float(r.get('climate_uv_proxy_index_0_100', 0.0)):.1f}/100 "
            f"({str(r.get('climate_uv_proxy_level', 'na'))}), "
            f"sayisal_tahmin={'hayir' if not bool(r.get('numeric_skin_incidence_estimation_allowed', False)) else 'evet'}"
        )
    return lines


def load_skin_per10k_summary(root: Path, date_label: str) -> list[str]:
    candidates = [
        root / f"cilt_kanseri_10000_kisi_senaryo_{date_label}.csv",
        root / "cilt_kanseri_10000_kisi_senaryo_latest.csv",
    ]
    pth = None
    for p in candidates:
        if p.exists():
            pth = p
            break
    if pth is None:
        return ["- cilt_kanseri_10000_kisi_senaryo tablosu bulunamadi."]
    try:
        d = pd.read_csv(pth)
    except Exception as e:
        return [f"- cilt_kanseri_10000_kisi_senaryo okuma hatasi: {e}"]
    if d.empty:
        return ["- cilt_kanseri_10000_kisi_senaryo bos."]

    lines: list[str] = []
    base = d[d["model_key"].astype(str) == "baseline"]
    if not base.empty:
        r = base.iloc[0]
        lines.append(
            f"- Baseline: {float(r.get('baseline_cases_per_10000', 0.0)):.4f} /10.000 "
            f"(new_cases={float(r.get('baseline_new_cases', 0.0)):.0f}, pop={float(r.get('baseline_population', 0.0)):.0f})"
        )
        lines.append(
            f"- Solar ortalama degisimi: {float(r.get('solar_delta_pct', 0.0)):+.2f}% "
            f"({float(r.get('solar_baseline_mean_kwh_m2_day', 0.0)):.3f} -> {float(r.get('solar_future_mean_kwh_m2_day', 0.0)):.3f} kWh/m2/gun)"
        )

    for mk in ["quant", "strong"]:
        x = d[d["model_key"].astype(str) == mk]
        if x.empty:
            continue
        r = x.iloc[0]
        lines.append(
            f"- {mk}: projeksiyon={float(r.get('projected_cases_per_10000', 0.0)):.4f} /10.000, "
            f"ek={float(r.get('additional_cases_per_10000', 0.0)):+.4f} /10.000, "
            f"proxy_rr={float(r.get('proxy_rr_multiplier', 0.0)):.4f}"
        )
    return lines


def build_release_readout(root: Path, date_label: str, zip_path: Path, manifest: Path, qc_report: Path) -> Path:
    out = root / f"release_readout_{date_label}_v6.md"
    model_path = resolve_model_summary_path(root)

    lines = [
        f"# Release Readout ({date_label})",
        "",
        "## Paket Durumu",
        "",
        f"- Zip: `{zip_path.name}`",
        f"- Manifest: `{manifest.name}`",
        f"- QC: `{qc_report.name}`",
        "",
    ]

    if model_path is not None:
        try:
            df = pd.read_csv(model_path)
            lower = df["model"].astype(str).str.lower()
            strong = df[lower.str.contains("strong", regex=False)].iloc[0]
            quant = df[lower.str.contains("quant", regex=False)].iloc[0]
            lines += [
                "## Kilit Metrikler",
                "",
                f"- Strong future RR: {float(strong.get('future_rr_mean', float('nan'))):.4f}",
                f"- Strong delta RR: {float(strong.get('delta_rr_mean', float('nan'))):+.4f}",
                f"- Quant future RR: {float(quant.get('future_rr_mean', float('nan'))):.4f}",
                f"- Quant delta RR: {float(quant.get('delta_rr_mean', float('nan'))):+.4f}",
                f"- Strong threshold exceed: {float(strong.get('future_threshold_exceed_share', float('nan'))):.1%}",
                f"- Quant threshold exceed: {float(quant.get('future_threshold_exceed_share', float('nan'))):.1%}",
                "",
            ]
        except Exception as e:
            lines += ["## Kilit Metrikler", "", f"- Okuma hatasi: {e}", ""]
    else:
        lines += ["## Kilit Metrikler", "", "- Model summary dosyasi bulunamadi.", ""]

    lines += ["## Model Stabilite Kalibrasyonu", ""] + load_stability_summary(root, date_label) + [""]
    lines += ["## Cilt Kanseri Odak", ""] + load_skin_focus_summary(root, date_label) + [""]
    lines += ["## Cilt Kanseri (10.000 Kiside Senaryo)", ""] + load_skin_per10k_summary(root, date_label) + [""]

    priority_csv = root / f"halk_dili_oncelik_matrisi_{date_label}.csv"
    if priority_csv.exists():
        try:
            p = pd.read_csv(priority_csv)
            counts = p["aksiyon_onceligi"].astype(str).value_counts().to_dict()
            lines += [
                "## Aksiyon Onceligi (Hastalik Gruplari)",
                "",
                f"- Acil eylem: {int(counts.get('acil_eylem', 0))}",
                f"- Hedefli onlem: {int(counts.get('hedefli_onlem', 0))}",
                f"- Yakindan izlem: {int(counts.get('yakindan_izlem', 0))}",
                f"- Rutin izlem: {int(counts.get('rutin_izlem', 0))}",
                "",
            ]
        except Exception as e:
            lines += ["## Aksiyon Onceligi (Hastalik Gruplari)", "", f"- Okuma hatasi: {e}", ""]

    weekly_csv = root / f"halk_dili_haftalik_is_plani_{date_label}.csv"
    if weekly_csv.exists():
        try:
            w = pd.read_csv(weekly_csv)
            if not w.empty:
                r = w.iloc[0]
                lines += [
                    "## Haftalik Is Yuku (Ilk Hafta)",
                    "",
                    f"- Aktif gorev: {int(r.get('toplam_aktif_gorev', 0))}",
                    f"- Acil/Hedefli/Yakindan/Rutin: {int(r.get('acil_eylem', 0))}/"
                    f"{int(r.get('hedefli_onlem', 0))}/{int(r.get('yakindan_izlem', 0))}/{int(r.get('rutin_izlem', 0))}",
                    f"- Odak hastaliklar: {str(r.get('odak_hastaliklar', ''))}",
                    "",
                ]
        except Exception as e:
            lines += ["## Haftalik Is Yuku (Ilk Hafta)", "", f"- Okuma hatasi: {e}", ""]

    alarm_rules_csv = root / f"halk_dili_kpi_alarm_kurallari_{date_label}.csv"
    if alarm_rules_csv.exists():
        try:
            a = pd.read_csv(alarm_rules_csv)
            lines += [
                "## KPI Alarm Hazirlik Ozeti",
                "",
                f"- Toplam alarm kurali: {int(len(a))}",
                f"- Kapsanan hafta sayisi: {int(a['hafta'].nunique()) if 'hafta' in a.columns else 0}",
                f"- KPI sayisi: {int(a['kpi_adi'].nunique()) if 'kpi_adi' in a.columns else 0}",
                f"- Alt-esik alarm kurali: {int((a['alarm_yonu'] == 'altina_duserse').sum()) if 'alarm_yonu' in a.columns else 0}",
                f"- Ust-esik alarm kurali: {int((a['alarm_yonu'] == 'ustune_cikarsa').sum()) if 'alarm_yonu' in a.columns else 0}",
                "",
            ]
        except Exception as e:
            lines += ["## KPI Alarm Hazirlik Ozeti", "", f"- Okuma hatasi: {e}", ""]

    dashboard_png = root / f"halk_dili_entegre_bulgular_panosu_{date_label}.png"
    dashboard_md = root / f"halk_dili_entegre_bulgular_ozeti_{date_label}.md"
    if dashboard_png.exists() or dashboard_md.exists():
        lines += [
            "## Entegre Pano",
            "",
            f"- Pano gorseli: {'var' if dashboard_png.exists() else 'yok'}",
            f"- Pano ozeti: {'var' if dashboard_md.exists() else 'yok'}",
            "",
        ]

    scenario_csv = root / f"halk_dili_mudahale_senaryolari_{date_label}.csv"
    if scenario_csv.exists():
        try:
            s = pd.read_csv(scenario_csv).sort_values("beklenen_risk_azalimi_yuzde", ascending=False)
            if not s.empty:
                r = s.iloc[0]
                lines += [
                    "## Mudahale Senaryosu (En Iyi Etki)",
                    "",
                    f"- Senaryo: {str(r.get('senaryo', ''))}",
                    f"- Beklenen risk azalimi: %{float(r.get('beklenen_risk_azalimi_yuzde', 0.0)):.2f}",
                    f"- Kalan risk endeksi: {float(r.get('kalan_risk_endeksi_100', 0.0)):.2f}",
                    f"- Kalan kritik grup sayisi: {int(r.get('kritik_grup_sayisi_kalan', 0))}",
                    "",
                ]
        except Exception as e:
            lines += ["## Mudahale Senaryosu (En Iyi Etki)", "", f"- Okuma hatasi: {e}", ""]

    ops_csv = root / f"halk_dili_operasyon_hazirlik_matrisi_{date_label}.csv"
    if ops_csv.exists():
        try:
            o = pd.read_csv(ops_csv).sort_values("hazirlik_baski_puani", ascending=False)
            if not o.empty:
                r = o.iloc[0]
                lines += [
                    "## Operasyonel Hazirlik (En Kritik Birim)",
                    "",
                    f"- Birim: {str(r.get('sorumlu_birim', ''))}",
                    f"- Hazirlik baski puani: {float(r.get('hazirlik_baski_puani', 0.0)):.2f}",
                    f"- Toplam gorev: {int(r.get('toplam_gorev', 0))} | acil={int(r.get('acil_eylem', 0))} | hedefli={int(r.get('hedefli_onlem', 0))}",
                    f"- Ortalama baslangic gunu: {float(r.get('ort_baslangic_gunu', 0.0)):.1f}",
                    "",
                ]
        except Exception as e:
            lines += ["## Operasyonel Hazirlik (En Kritik Birim)", "", f"- Okuma hatasi: {e}", ""]

    lines += [
        "## Hızlı Erişim",
        "",
        f"- `sunum_6_slayt_{date_label}_v3.pptx`",
        f"- `sunum_10_slayt_{date_label}_v4_detailed.pptx`",
        f"- `sunum_13_slayt_{date_label}_v6_board.pptx`",
        f"- `sunum_6_slayt_{date_label}_v3.pdf`",
        f"- `sunum_10_slayt_{date_label}_v4_detailed.pdf`",
        f"- `sunum_13_slayt_{date_label}_v6_board.pdf`",
        f"- `sunum_pdf_kitabi_{date_label}_v1.pdf`",
        f"- `yonetici_brif_tek_sayfa_{date_label}.pdf`",
        f"- `pdf_donusum_kalite_raporu_{date_label}_v1.md`",
        f"- `yonetici_tek_cumle_ozet_{date_label}.md`",
        f"- `teknik_ekip_aksiyon_ozeti_{date_label}.md`",
        f"- `model_comparison_summary_stable_calibrated_{date_label}.csv`",
        f"- `model_stability_diagnostics_{date_label}.csv`",
        f"- `model_stability_notes_{date_label}.md`",
        f"- `model_stability_dashboard_{date_label}.png`",
        f"- `cilt_kanseri_odak_model_tablosu_{date_label}.csv`",
        f"- `cilt_kanseri_odak_etioloji_tablosu_{date_label}.csv`",
        f"- `cilt_kanseri_odak_ozet_{date_label}.md`",
        f"- `cilt_kanseri_odak_pano_{date_label}.png`",
        f"- `cilt_kanseri_10000_kisi_senaryo_{date_label}.csv`",
        f"- `cilt_kanseri_10000_kisi_gorsel_{date_label}.png`",
        f"- `halk_dili_sik_sorular_{date_label}.md`",
        f"- `halk_dili_hastalik_ozet_{date_label}.csv`",
        f"- `halk_dili_hastalik_karsilastirma_{date_label}.png`",
        f"- `halk_dili_oncelik_matrisi_{date_label}.csv`",
        f"- `halk_dili_oncelik_matrisi_{date_label}.png`",
        f"- `halk_dili_eylem_plani_{date_label}.csv`",
        f"- `halk_dili_eylem_takvimi_{date_label}.png`",
        f"- `halk_dili_haftalik_is_plani_{date_label}.csv`",
        f"- `halk_dili_haftalik_is_ozeti_{date_label}.md`",
        f"- `halk_dili_haftalik_is_yuku_{date_label}.png`",
        f"- `halk_dili_hafta1_gorev_listesi_{date_label}.csv`",
        f"- `halk_dili_hafta1_toplanti_ajandasi_{date_label}.md`",
        f"- `halk_dili_haftalik_kpi_takip_{date_label}.csv`",
        f"- `halk_dili_kpi_alarm_kurallari_{date_label}.csv`",
        f"- `halk_dili_kpi_alarm_ozeti_{date_label}.md`",
        f"- `halk_dili_kpi_alarm_panosu_{date_label}.png`",
        f"- `halk_dili_entegre_bulgular_panosu_{date_label}.png`",
        f"- `halk_dili_entegre_bulgular_ozeti_{date_label}.md`",
        f"- `halk_dili_mudahale_senaryolari_{date_label}.csv`",
        f"- `halk_dili_mudahale_senaryo_ozeti_{date_label}.md`",
        f"- `halk_dili_mudahale_senaryolari_{date_label}.png`",
        f"- `halk_dili_operasyon_hazirlik_matrisi_{date_label}.csv`",
        f"- `halk_dili_operasyon_hazirlik_ozeti_{date_label}.md`",
        f"- `halk_dili_operasyon_hazirlik_panosu_{date_label}.png`",
        "",
    ]
    lines += ["## PDF Donusum Ozeti", ""] + load_pdf_conversion_summary(root, date_label) + [""]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    cwd = Path("/Users/yasinkaya/Hackhaton")
    root = args.root_dir.resolve()
    fig = args.fig_dir.resolve()

    logo_args: list[str] = []
    if args.logo_path is not None:
        logo_args = ["--logo-path", str(args.logo_path.resolve())]
    strict_visual_pdf = args.require_visual_pdf or args.pdf_quality_target == "visual"
    pdf_strategy = "auto"
    if args.pdf_quality_target == "any" and not has_visual_pdf_tool():
        # Avoid repeated Keynote timeout attempts in non-visual environments.
        pdf_strategy = "fallback"

    # 0) Build stability-calibrated model summary (used by downstream scripts).
    run(
        [
            args.python_bin,
            "scripts/build_health_stable_calibration_pack.py",
            "--root-dir",
            str(root),
            "--date-label",
            args.date_label,
        ],
        cwd=cwd,
    )

    # 1) Build markdown+pdf brief
    run(
        [
            args.python_bin,
            "scripts/build_health_presentation_pack.py",
            "--root-dir",
            str(root),
            "--fig-dir",
            str(fig),
            "--date-label",
            args.date_label,
        ],
        cwd=cwd,
    )

    # 2) Build PPTX modes
    for mode in ["executive", "detailed", "board"]:
        run(
            [
                args.python_bin,
                "scripts/build_health_pptx.py",
                "--root-dir",
                str(root),
                "--fig-dir",
                str(fig),
                "--date-label",
                args.date_label,
                "--mode",
                mode,
                *logo_args,
            ],
            cwd=cwd,
        )

    # 2.1) Build dual-audience summaries
    run(
        [
            args.python_bin,
            "scripts/build_dual_audience_health_notes.py",
            "--root-dir",
            str(root),
            "--date-label",
            args.date_label,
        ],
        cwd=cwd,
    )

    # 2.2) Build public-friendly literature + disease outputs
    run(
        [
            args.python_bin,
            "scripts/build_health_literature_alignment_pack.py",
            "--output-dir",
            str(root),
            "--tag-date",
            args.date_label,
        ],
        cwd=cwd,
    )

    # 2.25) Build skin-cancer focused outputs (UV guardrail + proxy dashboard)
    run(
        [
            args.python_bin,
            "scripts/build_skin_cancer_focus_pack.py",
            "--root-dir",
            str(root),
            "--date-label",
            args.date_label,
        ],
        cwd=cwd,
    )

    # 2.3) Convert generated PPTX decks to PDF (tool-first, text-fallback).
    run(
        [
            args.python_bin,
            "scripts/convert_pptx_to_pdf.py",
            "--input-pptx",
            str(root / f"sunum_6_slayt_{args.date_label}_v3.pptx"),
            "--output-pdf",
            str(root / f"sunum_6_slayt_{args.date_label}_v3.pdf"),
            "--report-json",
            str(root / f"sunum_6_slayt_{args.date_label}_v3_pdf_conversion.json"),
            "--strategy",
            pdf_strategy,
            *(["--require-visual"] if strict_visual_pdf else []),
        ],
        cwd=cwd,
    )
    run(
        [
            args.python_bin,
            "scripts/convert_pptx_to_pdf.py",
            "--input-pptx",
            str(root / f"sunum_10_slayt_{args.date_label}_v4_detailed.pptx"),
            "--output-pdf",
            str(root / f"sunum_10_slayt_{args.date_label}_v4_detailed.pdf"),
            "--report-json",
            str(root / f"sunum_10_slayt_{args.date_label}_v4_detailed_pdf_conversion.json"),
            "--strategy",
            pdf_strategy,
            *(["--require-visual"] if strict_visual_pdf else []),
        ],
        cwd=cwd,
    )
    run(
        [
            args.python_bin,
            "scripts/convert_pptx_to_pdf.py",
            "--input-pptx",
            str(root / f"sunum_13_slayt_{args.date_label}_v6_board.pptx"),
            "--output-pdf",
            str(root / f"sunum_13_slayt_{args.date_label}_v6_board.pdf"),
            "--report-json",
            str(root / f"sunum_13_slayt_{args.date_label}_v6_board_pdf_conversion.json"),
            "--strategy",
            pdf_strategy,
            *(["--require-visual"] if strict_visual_pdf else []),
        ],
        cwd=cwd,
    )
    run(
        [
            args.python_bin,
            "scripts/build_health_pdf_book.py",
            "--root-dir",
            str(root),
            "--date-label",
            args.date_label,
        ],
        cwd=cwd,
    )
    run(
        [
            args.python_bin,
            "scripts/build_pdf_conversion_quality_report.py",
            "--root-dir",
            str(root),
            "--date-label",
            args.date_label,
        ],
        cwd=cwd,
    )

    # 3) Keep dated note-file aliases for package consistency.
    maybe_copy(
        root / "konusmaci_notlari_6_slayt_2026-03-05_v2.md",
        root / f"konusmaci_notlari_6_slayt_{args.date_label}.md",
    )
    maybe_copy(
        root / "konusmaci_notlari_10_slayt_2026-03-05_v4.md",
        root / f"konusmaci_notlari_10_slayt_{args.date_label}.md",
    )
    maybe_copy(
        root / "konusmaci_notlari_13_slayt_2026-03-06_v6.md",
        root / f"konusmaci_notlari_13_slayt_{args.date_label}.md",
    )

    # 4) Build zip bundle.
    zip_path = root / f"health_slide_pack_{args.date_label}_v6.zip"
    if zip_path.exists():
        zip_path.unlink()
    files_to_pack = [
        root / f"sunum_6_slayt_{args.date_label}_v3.pptx",
        root / f"sunum_10_slayt_{args.date_label}_v4_detailed.pptx",
        root / f"sunum_13_slayt_{args.date_label}_v6_board.pptx",
        root / f"sunum_6_slayt_{args.date_label}_v3.pdf",
        root / f"sunum_10_slayt_{args.date_label}_v4_detailed.pdf",
        root / f"sunum_13_slayt_{args.date_label}_v6_board.pdf",
        root / f"sunum_6_slayt_{args.date_label}_v3_pdf_conversion.json",
        root / f"sunum_10_slayt_{args.date_label}_v4_detailed_pdf_conversion.json",
        root / f"sunum_13_slayt_{args.date_label}_v6_board_pdf_conversion.json",
        root / f"sunum_pdf_kitabi_{args.date_label}_v1.pdf",
        root / f"pdf_donusum_kalite_raporu_{args.date_label}_v1.md",
        root / f"konusmaci_notlari_6_slayt_{args.date_label}.md",
        root / f"konusmaci_notlari_10_slayt_{args.date_label}.md",
        root / f"konusmaci_notlari_13_slayt_{args.date_label}.md",
        root / f"yonetici_brif_tek_sayfa_{args.date_label}.pdf",
        root / f"yonetici_tek_cumle_ozet_{args.date_label}.md",
        root / f"teknik_ekip_aksiyon_ozeti_{args.date_label}.md",
        root / f"model_comparison_summary_stable_calibrated_{args.date_label}.csv",
        root / f"model_stability_diagnostics_{args.date_label}.csv",
        root / f"model_stability_notes_{args.date_label}.md",
        root / f"model_stability_dashboard_{args.date_label}.png",
        root / f"cilt_kanseri_odak_model_tablosu_{args.date_label}.csv",
        root / f"cilt_kanseri_odak_etioloji_tablosu_{args.date_label}.csv",
        root / f"cilt_kanseri_odak_ozet_{args.date_label}.md",
        root / f"cilt_kanseri_odak_pano_{args.date_label}.png",
        root / f"cilt_kanseri_10000_kisi_senaryo_{args.date_label}.csv",
        root / f"cilt_kanseri_10000_kisi_gorsel_{args.date_label}.png",
        root / f"karar_mesaji_ozet_{args.date_label}.csv",
        root / f"karar_mesaji_gorsel_{args.date_label}.png",
        root / f"halk_dili_ozet_{args.date_label}.csv",
        root / f"halk_dili_aciklama_{args.date_label}.md",
        root / f"halk_dili_risk_gorsel_{args.date_label}.png",
        root / f"literatur_tutarlilik_kontrolu_{args.date_label}.csv",
        root / f"literatur_uyum_trafik_isigi_{args.date_label}.csv",
        root / f"literatur_uyum_trafik_isigi_{args.date_label}.png",
        root / f"devam_literatur_uyum_ozeti_{args.date_label}.md",
        root / f"halk_dili_hastalik_ozet_{args.date_label}.csv",
        root / f"halk_dili_hastalik_karsilastirma_{args.date_label}.png",
        root / f"halk_dili_sik_sorular_{args.date_label}.md",
        root / f"halk_dili_oncelik_matrisi_{args.date_label}.csv",
        root / f"halk_dili_oncelik_matrisi_{args.date_label}.png",
        root / f"halk_dili_eylem_plani_{args.date_label}.csv",
        root / f"halk_dili_eylem_takvimi_{args.date_label}.png",
        root / f"halk_dili_haftalik_is_plani_{args.date_label}.csv",
        root / f"halk_dili_haftalik_is_ozeti_{args.date_label}.md",
        root / f"halk_dili_haftalik_is_yuku_{args.date_label}.png",
        root / f"halk_dili_hafta1_gorev_listesi_{args.date_label}.csv",
        root / f"halk_dili_hafta1_toplanti_ajandasi_{args.date_label}.md",
        root / f"halk_dili_haftalik_kpi_takip_{args.date_label}.csv",
        root / f"halk_dili_kpi_alarm_kurallari_{args.date_label}.csv",
        root / f"halk_dili_kpi_alarm_ozeti_{args.date_label}.md",
        root / f"halk_dili_kpi_alarm_panosu_{args.date_label}.png",
        root / f"halk_dili_entegre_bulgular_panosu_{args.date_label}.png",
        root / f"halk_dili_entegre_bulgular_ozeti_{args.date_label}.md",
        root / f"halk_dili_mudahale_senaryolari_{args.date_label}.csv",
        root / f"halk_dili_mudahale_senaryo_ozeti_{args.date_label}.md",
        root / f"halk_dili_mudahale_senaryolari_{args.date_label}.png",
        root / f"halk_dili_operasyon_hazirlik_matrisi_{args.date_label}.csv",
        root / f"halk_dili_operasyon_hazirlik_ozeti_{args.date_label}.md",
        root / f"halk_dili_operasyon_hazirlik_panosu_{args.date_label}.png",
        root / "SLIDE_PACK_README.md",
        fig / "fig01_model_overview.png",
        fig / "fig06_all_disease_direct_scores.png",
        fig / "fig07_all_disease_etiology_coverage.png",
    ]

    existing = [p for p in files_to_pack if p.exists()]
    if not existing:
        raise RuntimeError("No files found to package.")

    rel_paths = [str(p.relative_to(root)) if p.is_relative_to(root) else str(p) for p in existing]
    zip_cmd = ["zip", "-q", "-r", str(zip_path)] + rel_paths
    run(zip_cmd, cwd=root)

    # 4.1) QC report after packaging
    run(
        [
            args.python_bin,
            "scripts/health_release_qc.py",
            "--root-dir",
            str(root),
            "--date-label",
            args.date_label,
            "--zip-name",
            zip_path.name,
            *(["--require-visual-pdf"] if strict_visual_pdf else []),
        ],
        cwd=cwd,
    )
    qc_report = root / f"release_qc_report_{args.date_label}_v6.md"
    if qc_report.exists():
        run(["zip", "-q", str(zip_path), qc_report.name], cwd=root)
        existing.append(qc_report)

    # 4.2) Human-readable release readout
    readout = build_release_readout(root, args.date_label, zip_path, manifest=root / f"release_manifest_{args.date_label}_v6.txt", qc_report=qc_report)
    if readout.exists():
        run(["zip", "-q", str(zip_path), readout.name], cwd=root)
        existing.append(readout)

    # 4.3) Release-vs-previous diff report
    run(
        [
            args.python_bin,
            "scripts/health_release_diff.py",
            "--root-dir",
            str(root),
            "--date-label",
            args.date_label,
            "--version-tag",
            "v6",
        ],
        cwd=cwd,
    )
    diff_report = root / f"release_diff_{args.date_label}_v6.md"
    if diff_report.exists():
        run(["zip", "-q", str(zip_path), diff_report.name], cwd=root)
        existing.append(diff_report)

    manifest = root / f"release_manifest_{args.date_label}_v6.txt"
    lines = [f"Release bundle: {zip_path.name}", "", "Included files:"]
    for p in existing:
        lines.append(f"- {p.name} ({p.stat().st_size} bytes)")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 5) Latest aliases for quick access
    latest_map = {
        root / f"sunum_6_slayt_{args.date_label}_v3.pptx": root / "sunum_6_slayt_latest.pptx",
        root / f"sunum_10_slayt_{args.date_label}_v4_detailed.pptx": root / "sunum_10_slayt_detailed_latest.pptx",
        root / f"sunum_13_slayt_{args.date_label}_v6_board.pptx": root / "sunum_13_slayt_board_latest.pptx",
        root / f"sunum_6_slayt_{args.date_label}_v3.pdf": root / "sunum_6_slayt_latest.pdf",
        root / f"sunum_10_slayt_{args.date_label}_v4_detailed.pdf": root / "sunum_10_slayt_detailed_latest.pdf",
        root / f"sunum_13_slayt_{args.date_label}_v6_board.pdf": root / "sunum_13_slayt_board_latest.pdf",
        root / f"sunum_6_slayt_{args.date_label}_v3_pdf_conversion.json": root / "sunum_6_slayt_pdf_conversion_latest.json",
        root / f"sunum_10_slayt_{args.date_label}_v4_detailed_pdf_conversion.json": root / "sunum_10_slayt_pdf_conversion_latest.json",
        root / f"sunum_13_slayt_{args.date_label}_v6_board_pdf_conversion.json": root / "sunum_13_slayt_pdf_conversion_latest.json",
        root / f"sunum_pdf_kitabi_{args.date_label}_v1.pdf": root / "sunum_pdf_kitabi_latest.pdf",
        root / f"pdf_donusum_kalite_raporu_{args.date_label}_v1.md": root / "pdf_donusum_kalite_raporu_latest.md",
        root / f"yonetici_brif_tek_sayfa_{args.date_label}.pdf": root / "yonetici_brif_tek_sayfa_latest.pdf",
        root / f"yonetici_tek_cumle_ozet_{args.date_label}.md": root / "yonetici_tek_cumle_ozet_latest.md",
        root / f"teknik_ekip_aksiyon_ozeti_{args.date_label}.md": root / "teknik_ekip_aksiyon_ozeti_latest.md",
        root / f"model_comparison_summary_stable_calibrated_{args.date_label}.csv": root / "model_comparison_summary_stable_calibrated_latest.csv",
        root / f"model_stability_diagnostics_{args.date_label}.csv": root / "model_stability_diagnostics_latest.csv",
        root / f"model_stability_notes_{args.date_label}.md": root / "model_stability_notes_latest.md",
        root / f"model_stability_dashboard_{args.date_label}.png": root / "model_stability_dashboard_latest.png",
        root / f"cilt_kanseri_odak_model_tablosu_{args.date_label}.csv": root / "cilt_kanseri_odak_model_tablosu_latest.csv",
        root / f"cilt_kanseri_odak_etioloji_tablosu_{args.date_label}.csv": root / "cilt_kanseri_odak_etioloji_tablosu_latest.csv",
        root / f"cilt_kanseri_odak_ozet_{args.date_label}.md": root / "cilt_kanseri_odak_ozet_latest.md",
        root / f"cilt_kanseri_odak_pano_{args.date_label}.png": root / "cilt_kanseri_odak_pano_latest.png",
        root / f"cilt_kanseri_10000_kisi_senaryo_{args.date_label}.csv": root / "cilt_kanseri_10000_kisi_senaryo_latest.csv",
        root / f"cilt_kanseri_10000_kisi_gorsel_{args.date_label}.png": root / "cilt_kanseri_10000_kisi_gorsel_latest.png",
        root / f"karar_mesaji_ozet_{args.date_label}.csv": root / "karar_mesaji_ozet_latest.csv",
        root / f"karar_mesaji_gorsel_{args.date_label}.png": root / "karar_mesaji_gorsel_latest.png",
        root / f"halk_dili_ozet_{args.date_label}.csv": root / "halk_dili_ozet_latest.csv",
        root / f"halk_dili_aciklama_{args.date_label}.md": root / "halk_dili_aciklama_latest.md",
        root / f"halk_dili_risk_gorsel_{args.date_label}.png": root / "halk_dili_risk_gorsel_latest.png",
        root / f"literatur_tutarlilik_kontrolu_{args.date_label}.csv": root / "literatur_tutarlilik_kontrolu_latest.csv",
        root / f"literatur_uyum_trafik_isigi_{args.date_label}.csv": root / "literatur_uyum_trafik_isigi_latest.csv",
        root / f"literatur_uyum_trafik_isigi_{args.date_label}.png": root / "literatur_uyum_trafik_isigi_latest.png",
        root / f"devam_literatur_uyum_ozeti_{args.date_label}.md": root / "devam_literatur_uyum_ozeti_latest.md",
        root / f"halk_dili_hastalik_ozet_{args.date_label}.csv": root / "halk_dili_hastalik_ozet_latest.csv",
        root / f"halk_dili_hastalik_karsilastirma_{args.date_label}.png": root / "halk_dili_hastalik_karsilastirma_latest.png",
        root / f"halk_dili_sik_sorular_{args.date_label}.md": root / "halk_dili_sik_sorular_latest.md",
        root / f"halk_dili_oncelik_matrisi_{args.date_label}.csv": root / "halk_dili_oncelik_matrisi_latest.csv",
        root / f"halk_dili_oncelik_matrisi_{args.date_label}.png": root / "halk_dili_oncelik_matrisi_latest.png",
        root / f"halk_dili_eylem_plani_{args.date_label}.csv": root / "halk_dili_eylem_plani_latest.csv",
        root / f"halk_dili_eylem_takvimi_{args.date_label}.png": root / "halk_dili_eylem_takvimi_latest.png",
        root / f"halk_dili_haftalik_is_plani_{args.date_label}.csv": root / "halk_dili_haftalik_is_plani_latest.csv",
        root / f"halk_dili_haftalik_is_ozeti_{args.date_label}.md": root / "halk_dili_haftalik_is_ozeti_latest.md",
        root / f"halk_dili_haftalik_is_yuku_{args.date_label}.png": root / "halk_dili_haftalik_is_yuku_latest.png",
        root / f"halk_dili_hafta1_gorev_listesi_{args.date_label}.csv": root / "halk_dili_hafta1_gorev_listesi_latest.csv",
        root / f"halk_dili_hafta1_toplanti_ajandasi_{args.date_label}.md": root / "halk_dili_hafta1_toplanti_ajandasi_latest.md",
        root / f"halk_dili_haftalik_kpi_takip_{args.date_label}.csv": root / "halk_dili_haftalik_kpi_takip_latest.csv",
        root / f"halk_dili_kpi_alarm_kurallari_{args.date_label}.csv": root / "halk_dili_kpi_alarm_kurallari_latest.csv",
        root / f"halk_dili_kpi_alarm_ozeti_{args.date_label}.md": root / "halk_dili_kpi_alarm_ozeti_latest.md",
        root / f"halk_dili_kpi_alarm_panosu_{args.date_label}.png": root / "halk_dili_kpi_alarm_panosu_latest.png",
        root / f"halk_dili_entegre_bulgular_panosu_{args.date_label}.png": root / "halk_dili_entegre_bulgular_panosu_latest.png",
        root / f"halk_dili_entegre_bulgular_ozeti_{args.date_label}.md": root / "halk_dili_entegre_bulgular_ozeti_latest.md",
        root / f"halk_dili_mudahale_senaryolari_{args.date_label}.csv": root / "halk_dili_mudahale_senaryolari_latest.csv",
        root / f"halk_dili_mudahale_senaryo_ozeti_{args.date_label}.md": root / "halk_dili_mudahale_senaryo_ozeti_latest.md",
        root / f"halk_dili_mudahale_senaryolari_{args.date_label}.png": root / "halk_dili_mudahale_senaryolari_latest.png",
        root / f"halk_dili_operasyon_hazirlik_matrisi_{args.date_label}.csv": root / "halk_dili_operasyon_hazirlik_matrisi_latest.csv",
        root / f"halk_dili_operasyon_hazirlik_ozeti_{args.date_label}.md": root / "halk_dili_operasyon_hazirlik_ozeti_latest.md",
        root / f"halk_dili_operasyon_hazirlik_panosu_{args.date_label}.png": root / "halk_dili_operasyon_hazirlik_panosu_latest.png",
        zip_path: root / "health_slide_pack_latest.zip",
        manifest: root / "release_manifest_latest.txt",
        qc_report: root / "release_qc_report_latest.md",
        readout: root / "release_readout_latest.md",
        diff_report: root / "release_diff_latest.md",
    }
    for src, dst in latest_map.items():
        copy_replace(src, dst)

    print(f"Wrote: {zip_path}")
    print(f"Wrote: {manifest}")
    print(f"Wrote: {qc_report}")
    print(f"Wrote: {readout}")
    if diff_report.exists():
        print(f"Wrote: {diff_report}")
    print(f"Updated latest aliases in: {root}")


if __name__ == "__main__":
    main()
