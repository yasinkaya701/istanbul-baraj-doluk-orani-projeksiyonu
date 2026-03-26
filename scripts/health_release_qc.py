#!/usr/bin/env python3
"""Quality checks for health presentation release artifacts."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd
from pptx import Presentation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QC checks for health release bundle")
    p.add_argument("--root-dir", type=Path, default=Path("output/health_impact"))
    p.add_argument("--date-label", type=str, required=True)
    p.add_argument("--zip-name", type=str, default=None, help="Optional explicit zip filename")
    p.add_argument(
        "--require-visual-pdf",
        action="store_true",
        help="Fail QC if any PDF conversion mode is fallback_text.",
    )
    return p.parse_args()


def check_file(path: Path, min_bytes: int = 1) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing: {path.name}"
    size = path.stat().st_size
    if size < min_bytes:
        return False, f"too small ({size} bytes): {path.name}"
    return True, f"ok ({size} bytes): {path.name}"


def check_pptx_slides(path: Path, expected: int) -> tuple[bool, str]:
    try:
        prs = Presentation(str(path))
    except Exception as e:
        return False, f"open failed ({path.name}): {e}"
    n = len(prs.slides)
    if n != expected:
        return False, f"slide count mismatch {path.name}: expected {expected}, got {n}"
    return True, f"slide count ok {path.name}: {n}"


def check_model_summary(root: Path) -> tuple[bool, str]:
    candidates = [
        root / "model_comparison_summary_stable_calibrated.csv",
        root / "model_comparison_summary_duzenlenmis_run.csv",
        root / "model_comparison_summary.csv",
    ]
    summary_path = None
    for p in candidates:
        if p.exists():
            summary_path = p
            break
    if summary_path is None:
        return False, "model summary missing"
    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        return False, f"model summary read failed ({summary_path.name}): {e}"

    if "model" not in df.columns:
        return False, f"model column missing ({summary_path.name})"
    lower = df["model"].astype(str).str.lower()
    if not (lower.str.contains("strong").any() and lower.str.contains("quant").any()):
        return False, f"strong/quant rows missing ({summary_path.name})"

    try:
        strong = df[lower.str.contains("strong")].iloc[0]
        quant = df[lower.str.contains("quant")].iloc[0]
        d_strong = float(strong.get("delta_rr_mean", 0.0))
        d_quant = float(quant.get("delta_rr_mean", 0.0))
        if d_strong < d_quant:
            return False, f"unexpected ordering: strong delta_rr({d_strong:.4f}) < quant({d_quant:.4f})"
        return True, f"model sanity ok ({summary_path.name}): strong delta_rr={d_strong:.4f}, quant={d_quant:.4f}"
    except Exception as e:
        return False, f"model sanity parse failed ({summary_path.name}): {e}"


def check_zip(path: Path, expected_members: list[str]) -> tuple[bool, str]:
    if not path.exists():
        return False, f"zip missing: {path.name}"
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
    except Exception as e:
        return False, f"zip open failed: {e}"

    missing = [m for m in expected_members if m not in names]
    if missing:
        return False, f"zip missing members: {', '.join(missing)}"
    return True, f"zip content ok: {path.name}"


def check_conversion_report(path: Path, require_visual_pdf: bool) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing: {path.name}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"json parse failed ({path.name}): {e}"
    mode = str(data.get("mode", ""))
    if mode not in {"soffice", "keynote", "fallback_text"}:
        return False, f"invalid mode ({path.name}): {mode}"
    if require_visual_pdf and mode == "fallback_text":
        return False, f"fallback mode not allowed in strict mode ({path.name})"
    if int(data.get("output_size_bytes", 0)) <= 0:
        return False, f"invalid output_size_bytes ({path.name})"
    return True, f"conversion report ok ({path.name}): mode={mode}"


def main() -> None:
    args = parse_args()
    root = args.root_dir.resolve()
    date_label = args.date_label

    f6 = root / f"sunum_6_slayt_{date_label}_v3.pptx"
    f10 = root / f"sunum_10_slayt_{date_label}_v4_detailed.pptx"
    f13 = root / f"sunum_13_slayt_{date_label}_v6_board.pptx"
    f6_pdf = root / f"sunum_6_slayt_{date_label}_v3.pdf"
    f10_pdf = root / f"sunum_10_slayt_{date_label}_v4_detailed.pdf"
    f13_pdf = root / f"sunum_13_slayt_{date_label}_v6_board.pdf"
    conv6_json = root / f"sunum_6_slayt_{date_label}_v3_pdf_conversion.json"
    conv10_json = root / f"sunum_10_slayt_{date_label}_v4_detailed_pdf_conversion.json"
    conv13_json = root / f"sunum_13_slayt_{date_label}_v6_board_pdf_conversion.json"
    pdf_book = root / f"sunum_pdf_kitabi_{date_label}_v1.pdf"
    pdf_quality_md = root / f"pdf_donusum_kalite_raporu_{date_label}_v1.md"
    pdf = root / f"yonetici_brif_tek_sayfa_{date_label}.pdf"
    one_liner = root / f"yonetici_tek_cumle_ozet_{date_label}.md"
    tech_note = root / f"teknik_ekip_aksiyon_ozeti_{date_label}.md"
    stable_summary_csv = root / f"model_comparison_summary_stable_calibrated_{date_label}.csv"
    stability_diag_csv = root / f"model_stability_diagnostics_{date_label}.csv"
    stability_md = root / f"model_stability_notes_{date_label}.md"
    stability_png = root / f"model_stability_dashboard_{date_label}.png"
    skin_model_csv = root / f"cilt_kanseri_odak_model_tablosu_{date_label}.csv"
    skin_eti_csv = root / f"cilt_kanseri_odak_etioloji_tablosu_{date_label}.csv"
    skin_md = root / f"cilt_kanseri_odak_ozet_{date_label}.md"
    skin_png = root / f"cilt_kanseri_odak_pano_{date_label}.png"
    skin_10k_csv = root / f"cilt_kanseri_10000_kisi_senaryo_{date_label}.csv"
    skin_10k_png = root / f"cilt_kanseri_10000_kisi_gorsel_{date_label}.png"
    decision_csv = root / f"karar_mesaji_ozet_{date_label}.csv"
    decision_png = root / f"karar_mesaji_gorsel_{date_label}.png"
    public_md = root / f"halk_dili_aciklama_{date_label}.md"
    faq_md = root / f"halk_dili_sik_sorular_{date_label}.md"
    disease_csv = root / f"halk_dili_hastalik_ozet_{date_label}.csv"
    disease_png = root / f"halk_dili_hastalik_karsilastirma_{date_label}.png"
    priority_csv = root / f"halk_dili_oncelik_matrisi_{date_label}.csv"
    priority_png = root / f"halk_dili_oncelik_matrisi_{date_label}.png"
    action_csv = root / f"halk_dili_eylem_plani_{date_label}.csv"
    action_png = root / f"halk_dili_eylem_takvimi_{date_label}.png"
    weekly_csv = root / f"halk_dili_haftalik_is_plani_{date_label}.csv"
    weekly_md = root / f"halk_dili_haftalik_is_ozeti_{date_label}.md"
    weekly_png = root / f"halk_dili_haftalik_is_yuku_{date_label}.png"
    week1_csv = root / f"halk_dili_hafta1_gorev_listesi_{date_label}.csv"
    week1_md = root / f"halk_dili_hafta1_toplanti_ajandasi_{date_label}.md"
    weekly_kpi_csv = root / f"halk_dili_haftalik_kpi_takip_{date_label}.csv"
    alarm_rules_csv = root / f"halk_dili_kpi_alarm_kurallari_{date_label}.csv"
    alarm_md = root / f"halk_dili_kpi_alarm_ozeti_{date_label}.md"
    alarm_png = root / f"halk_dili_kpi_alarm_panosu_{date_label}.png"
    dashboard_png = root / f"halk_dili_entegre_bulgular_panosu_{date_label}.png"
    dashboard_md = root / f"halk_dili_entegre_bulgular_ozeti_{date_label}.md"
    scenario_csv = root / f"halk_dili_mudahale_senaryolari_{date_label}.csv"
    scenario_md = root / f"halk_dili_mudahale_senaryo_ozeti_{date_label}.md"
    scenario_png = root / f"halk_dili_mudahale_senaryolari_{date_label}.png"
    ops_csv = root / f"halk_dili_operasyon_hazirlik_matrisi_{date_label}.csv"
    ops_md = root / f"halk_dili_operasyon_hazirlik_ozeti_{date_label}.md"
    ops_png = root / f"halk_dili_operasyon_hazirlik_panosu_{date_label}.png"
    lit_csv = root / f"literatur_tutarlilik_kontrolu_{date_label}.csv"
    zname = args.zip_name or f"health_slide_pack_{date_label}_v6.zip"
    zpath = root / zname

    checks: list[tuple[str, bool, str]] = []

    checks.append(("file", *check_file(f6, min_bytes=50_000)))
    checks.append(("file", *check_file(f10, min_bytes=50_000)))
    checks.append(("file", *check_file(f13, min_bytes=50_000)))
    checks.append(("file", *check_file(f6_pdf, min_bytes=20_000)))
    checks.append(("file", *check_file(f10_pdf, min_bytes=20_000)))
    checks.append(("file", *check_file(f13_pdf, min_bytes=20_000)))
    checks.append(("file", *check_file(pdf_book, min_bytes=30_000)))
    checks.append(("file", *check_file(pdf_quality_md, min_bytes=500)))
    checks.append(("conversion", *check_conversion_report(conv6_json, args.require_visual_pdf)))
    checks.append(("conversion", *check_conversion_report(conv10_json, args.require_visual_pdf)))
    checks.append(("conversion", *check_conversion_report(conv13_json, args.require_visual_pdf)))
    checks.append(("file", *check_file(pdf, min_bytes=20_000)))
    checks.append(("file", *check_file(one_liner, min_bytes=100)))
    checks.append(("file", *check_file(tech_note, min_bytes=200)))
    checks.append(("file", *check_file(stable_summary_csv, min_bytes=600)))
    checks.append(("file", *check_file(stability_diag_csv, min_bytes=400)))
    checks.append(("file", *check_file(stability_md, min_bytes=300)))
    checks.append(("file", *check_file(stability_png, min_bytes=40_000)))
    checks.append(("file", *check_file(skin_model_csv, min_bytes=500)))
    checks.append(("file", *check_file(skin_eti_csv, min_bytes=300)))
    checks.append(("file", *check_file(skin_md, min_bytes=500)))
    checks.append(("file", *check_file(skin_png, min_bytes=60_000)))
    checks.append(("file", *check_file(skin_10k_csv, min_bytes=500)))
    checks.append(("file", *check_file(skin_10k_png, min_bytes=60_000)))
    checks.append(("file", *check_file(decision_csv, min_bytes=150)))
    checks.append(("file", *check_file(decision_png, min_bytes=20_000)))
    checks.append(("file", *check_file(public_md, min_bytes=200)))
    checks.append(("file", *check_file(faq_md, min_bytes=300)))
    checks.append(("file", *check_file(disease_csv, min_bytes=800)))
    checks.append(("file", *check_file(disease_png, min_bytes=20_000)))
    checks.append(("file", *check_file(priority_csv, min_bytes=500)))
    checks.append(("file", *check_file(priority_png, min_bytes=20_000)))
    checks.append(("file", *check_file(action_csv, min_bytes=800)))
    checks.append(("file", *check_file(action_png, min_bytes=20_000)))
    checks.append(("file", *check_file(weekly_csv, min_bytes=600)))
    checks.append(("file", *check_file(weekly_md, min_bytes=300)))
    checks.append(("file", *check_file(weekly_png, min_bytes=20_000)))
    checks.append(("file", *check_file(week1_csv, min_bytes=500)))
    checks.append(("file", *check_file(week1_md, min_bytes=400)))
    checks.append(("file", *check_file(weekly_kpi_csv, min_bytes=250)))
    checks.append(("file", *check_file(alarm_rules_csv, min_bytes=1000)))
    checks.append(("file", *check_file(alarm_md, min_bytes=500)))
    checks.append(("file", *check_file(alarm_png, min_bytes=20_000)))
    checks.append(("file", *check_file(dashboard_png, min_bytes=50_000)))
    checks.append(("file", *check_file(dashboard_md, min_bytes=300)))
    checks.append(("file", *check_file(scenario_csv, min_bytes=400)))
    checks.append(("file", *check_file(scenario_md, min_bytes=500)))
    checks.append(("file", *check_file(scenario_png, min_bytes=60_000)))
    checks.append(("file", *check_file(ops_csv, min_bytes=500)))
    checks.append(("file", *check_file(ops_md, min_bytes=500)))
    checks.append(("file", *check_file(ops_png, min_bytes=80_000)))
    checks.append(("file", *check_file(lit_csv, min_bytes=200)))
    checks.append(("slides", *check_pptx_slides(f6, expected=6)))
    checks.append(("slides", *check_pptx_slides(f10, expected=10)))
    checks.append(("slides", *check_pptx_slides(f13, expected=13)))
    checks.append(("model", *check_model_summary(root)))

    expected_zip_members = [
        f6.name,
        f10.name,
        f13.name,
        f6_pdf.name,
        f10_pdf.name,
        f13_pdf.name,
        conv6_json.name,
        conv10_json.name,
        conv13_json.name,
        pdf_book.name,
        pdf_quality_md.name,
        f"konusmaci_notlari_6_slayt_{date_label}.md",
        f"konusmaci_notlari_10_slayt_{date_label}.md",
        f"konusmaci_notlari_13_slayt_{date_label}.md",
        pdf.name,
        one_liner.name,
        tech_note.name,
        stable_summary_csv.name,
        stability_diag_csv.name,
        stability_md.name,
        stability_png.name,
        skin_model_csv.name,
        skin_eti_csv.name,
        skin_md.name,
        skin_png.name,
        skin_10k_csv.name,
        skin_10k_png.name,
        decision_csv.name,
        decision_png.name,
        public_md.name,
        faq_md.name,
        disease_csv.name,
        disease_png.name,
        priority_csv.name,
        priority_png.name,
        action_csv.name,
        action_png.name,
        weekly_csv.name,
        weekly_md.name,
        weekly_png.name,
        week1_csv.name,
        week1_md.name,
        weekly_kpi_csv.name,
        alarm_rules_csv.name,
        alarm_md.name,
        alarm_png.name,
        dashboard_png.name,
        dashboard_md.name,
        scenario_csv.name,
        scenario_md.name,
        scenario_png.name,
        ops_csv.name,
        ops_md.name,
        ops_png.name,
        lit_csv.name,
        "SLIDE_PACK_README.md",
        "figures/fig01_model_overview.png",
        "figures/fig06_all_disease_direct_scores.png",
        "figures/fig07_all_disease_etiology_coverage.png",
    ]
    checks.append(("zip", *check_zip(zpath, expected_zip_members)))

    ok_count = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    overall_ok = ok_count == total

    md_lines = [
        f"# Release QC Report ({date_label})",
        "",
        f"- Root: `{root}`",
        f"- Overall: {'PASS' if overall_ok else 'FAIL'} ({ok_count}/{total})",
        "",
        "## Check Results",
        "",
    ]
    for kind, ok, msg in checks:
        status = "PASS" if ok else "FAIL"
        md_lines.append(f"- [{status}] ({kind}) {msg}")
    md_lines.append("")

    report = root / f"release_qc_report_{date_label}_v6.md"
    report.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {report}")
    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
