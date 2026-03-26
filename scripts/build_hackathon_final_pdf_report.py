#!/usr/bin/env python3
"""Create final Hackhaton PDF report with merged anomaly/news/forecast outputs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final Hackhaton PDF report")
    p.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/pdf/Hackhaton_Final_Raporu_2026-03-05.pdf"),
    )
    p.add_argument(
        "--events-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/tum_asiri_olaylar_bilimsel_filtreli.csv"),
    )
    p.add_argument(
        "--news-events-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/tum_asiri_olaylar_haber_enriched.csv"),
    )
    p.add_argument(
        "--continuous-summary-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/continuous/surekli_anomali_haber_ozet.csv"),
    )
    p.add_argument(
        "--fusion-summary-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/past_future_fusion/gecmis_gelecek_fusion_ozet.csv"),
    )
    p.add_argument(
        "--fusion-dashboard-png",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/past_future_fusion/gecmis_gelecek_fusion_dashboard.png"),
    )
    p.add_argument(
        "--continuous-overview-png",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/continuous/surekli_anomali_haber_overview.png"),
    )
    p.add_argument(
        "--v2-dashboard-png",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/charts_v2/anomali_haber_v2_dashboard.png"),
    )
    p.add_argument(
        "--var-fig-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/past_future_fusion"),
    )
    return p.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def fmt_num(x: Any, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"


def fit_image(path: Path, max_w: float, max_h: float) -> Image:
    img = PILImage.open(path)
    w, h = img.size
    ratio = min(max_w / float(w), max_h / float(h))
    rw = w * ratio
    rh = h * ratio
    return Image(str(path), width=rw, height=rh)


def page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawString(1.8 * cm, 1.2 * cm, "Bogazici KRDAE Hackhaton - Final Rapor")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.2 * cm, f"Sayfa {doc.page}")
    canvas.restoreState()


def build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=colors.HexColor("#0b2d45"),
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0b2d45"),
            spaceBefore=10,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1f5c8d"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=14,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=11.5,
            textColor=colors.HexColor("#4a4a4a"),
        ),
    }
    return styles


def build_summary_table(df: pd.DataFrame, cols: list[str], max_rows: int = 8) -> Table:
    show = df.copy()
    if len(show) > max_rows:
        show = show.head(max_rows)
    data = [cols]
    for _, r in show.iterrows():
        row = []
        for c in cols:
            row.append(str(r.get(c, "-")))
        data.append(row)
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b2d45")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8.8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f6f9fc")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bfc9d4")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def main() -> None:
    args = parse_args()
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)

    events = safe_read_csv(args.events_csv)
    news_events = safe_read_csv(args.news_events_csv)
    continuous_sum = safe_read_csv(args.continuous_summary_csv)
    fusion_sum = safe_read_csv(args.fusion_summary_csv)

    if events.empty:
        raise SystemExit(f"events csv missing/empty: {args.events_csv}")

    total_events = int(len(events))
    vars_count = events["variable"].astype(str).value_counts().to_dict() if "variable" in events.columns else {}
    news_matched = int(news_events["top_headline"].notna().sum()) if "top_headline" in news_events.columns else 0
    news_strict = int(
        pd.to_numeric(news_events.get("top_headline_match_score"), errors="coerce").ge(0.50).sum()
    ) if not news_events.empty else 0
    high_conf = int(events["internet_confidence"].astype(str).isin(["yuksek", "cok_yuksek"]).sum()) if "internet_confidence" in events.columns else 0
    tier_a = int((events.get("scientific_tier", pd.Series(dtype=str)).astype(str) == "A").sum())
    tier_b = int((events.get("scientific_tier", pd.Series(dtype=str)).astype(str) == "B").sum())

    # Top matched headlines table.
    top_news = pd.DataFrame()
    if not news_events.empty and "top_headline" in news_events.columns:
        top_news = news_events[news_events["top_headline"].notna()].copy()
        top_news["top_headline_match_score"] = pd.to_numeric(top_news["top_headline_match_score"], errors="coerce")
        top_news["center_time"] = pd.to_datetime(top_news["center_time"], errors="coerce")
        top_news = top_news.sort_values(
            ["top_headline_match_score", "peak_severity_score"],
            ascending=[False, False],
        ).head(15)
        top_news["date"] = top_news["center_time"].dt.strftime("%Y-%m-%d")
        top_news = top_news[
            [
                "date",
                "event_id",
                "variable",
                "top_headline_source",
                "top_headline_match_score",
                "top_headline",
            ]
        ].copy()
        top_news["top_headline_match_score"] = top_news["top_headline_match_score"].map(lambda x: fmt_num(x, 3))

    styles = build_styles()
    story = []

    # Cover
    story.append(Spacer(1, 1.8 * cm))
    story.append(Paragraph("115 Yillik Iklim Verisi Hackhaton", styles["title"]))
    story.append(Paragraph("Final Teknik Rapor - Anomali, Haber ve 2035 Projeksiyon Fusion", styles["h1"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["body"]))
    story.append(Paragraph("Kurulum: Bogazici Universitesi KRDAE Meteoroloji Laboratuvari", styles["body"]))
    story.append(Paragraph("Rapor kapsami: ham kagit veriden sayisallastirma, anomaly tespiti, internet/haber baglami, forecast fusion.", styles["body"]))
    story.append(Spacer(1, 0.8 * cm))
    story.append(
        Paragraph(
            f"<b>Ana metrikler</b><br/>"
            f"- Toplam bilimsel filtreli olay: <b>{total_events}</b><br/>"
            f"- Haber eslesen olay: <b>{news_matched}</b> (strict>=0.50: <b>{news_strict}</b>)<br/>"
            f"- Yuksek/cok_yuksek internet guven: <b>{high_conf}</b><br/>"
            f"- Scientific tier A/B: <b>{tier_a}/{tier_b}</b>",
            styles["body"],
        )
    )
    story.append(Spacer(1, 0.6 * cm))

    # Fusion dashboard on cover.
    if args.fusion_dashboard_png.exists():
        story.append(
            KeepTogether(
                [
                    fit_image(args.fusion_dashboard_png, max_w=17.5 * cm, max_h=10.5 * cm),
                    Spacer(1, 0.2 * cm),
                    Paragraph("Sekil 1: Gecmis + Gelecek (2035) tek dashboard", styles["small"]),
                ]
            )
        )

    story.append(PageBreak())

    # Section: Pipeline and methodology
    story.append(Paragraph("1) Yontem ve Akis", styles["h1"]))
    bullets = [
        "Farkli kagit tipleri (el yazisi/graf kagidi/farkli dil) sayisallastirildi ve ortak long format tabloya donusturuldu.",
        "Anomali tespiti: robust z-score, tail flags, iforest, rejim kirilmasi sinyalleri birlestirildi.",
        "Internet baglami: ENSO (ONI), NAO, global pencere eslesmeleri ve quant context birlestirildi.",
        "Haber baglami: gazete/haber baslik katalogu ile zaman+tehlike eslestirmesi yapildi.",
        "Forecast: degisken bazli en iyi pipeline ciktilari birlestirilerek 2035 ufku cikartildi.",
        "Son adim: anomaly + news + forecast tek zaman ekseninde fusion dashboard olarak sunuldu.",
    ]
    for b in bullets:
        story.append(Paragraph(f"- {b}", styles["body"]))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("2) Degisken Dagilimlari", styles["h2"]))
    if vars_count:
        vars_df = pd.DataFrame(
            [{"variable": k, "event_count": v} for k, v in vars_count.items()]
        ).sort_values("event_count", ascending=False)
        story.append(build_summary_table(vars_df, ["variable", "event_count"], max_rows=10))
    else:
        story.append(Paragraph("Degisken ozet tablosu olusturulamadi.", styles["body"]))

    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("3) Surekli Ozet ve Forecast Ozet", styles["h2"]))
    if not continuous_sum.empty:
        cs = continuous_sum.copy()
        cols = [c for c in ["variable", "event_count", "news_matched", "news_matched_strict", "median_severity"] if c in cs.columns]
        if cols:
            story.append(build_summary_table(cs, cols, max_rows=10))
            story.append(Spacer(1, 0.2 * cm))
    if not fusion_sum.empty:
        fs = fusion_sum.copy()
        cols = [c for c in ["variable", "rows", "forecast_months", "anomaly_ema_median", "forecast_yhat_median"] if c in fs.columns]
        if cols:
            story.append(build_summary_table(fs, cols, max_rows=10))

    if args.continuous_overview_png.exists():
        story.append(Spacer(1, 0.45 * cm))
        story.append(
            KeepTogether(
                [
                    fit_image(args.continuous_overview_png, max_w=17.5 * cm, max_h=8.8 * cm),
                    Paragraph("Sekil 2: Surekli aylik anomaly-news overview", styles["small"]),
                ]
            )
        )
        story.append(Spacer(1, 0.15 * cm))

    # Section: Top headline matches
    story.append(Paragraph("4) En Guclu Haber Eslesmeleri", styles["h1"]))
    story.append(
        Paragraph(
            "Asagidaki tablo anomaly olaylari ile en iyi eslesen gazete/haber basliklarini skora gore listeler.",
            styles["body"],
        )
    )
    if not top_news.empty:
        tbl = build_summary_table(
            top_news,
            ["date", "event_id", "variable", "top_headline_source", "top_headline_match_score", "top_headline"],
            max_rows=15,
        )
        story.append(tbl)
    else:
        story.append(Paragraph("Haber eslesme tablosu bulunamadi.", styles["body"]))

    # v2 dashboard
    if args.v2_dashboard_png.exists():
        story.append(Spacer(1, 0.35 * cm))
        story.append(
            KeepTogether(
                [
                    fit_image(args.v2_dashboard_png, max_w=17.5 * cm, max_h=8.7 * cm),
                    Paragraph("Sekil 3: V2 anomali-haber dashboard", styles["small"]),
                ]
            )
        )
        story.append(Spacer(1, 0.15 * cm))

    # Variable pages
    story.append(Paragraph("5) Degisken Bazli Grafikler", styles["h1"]))
    var_paths = [
        ("temp", args.var_fig_dir / "gecmis_gelecek_fusion_temp.png"),
        ("humidity", args.var_fig_dir / "gecmis_gelecek_fusion_humidity.png"),
        ("pressure", args.var_fig_dir / "gecmis_gelecek_fusion_pressure.png"),
        ("precip", args.var_fig_dir / "gecmis_gelecek_fusion_precip.png"),
    ]
    for var, p in var_paths:
        story.append(Paragraph(f"{var} - gecmis+gelecek fusion", styles["h2"]))
        if p.exists():
            story.append(fit_image(p, max_w=17.5 * cm, max_h=10.5 * cm))
        else:
            story.append(Paragraph(f"Grafik bulunamadi: {p}", styles["body"]))
        story.append(Spacer(1, 0.18 * cm))

    # Final notes + references
    story.append(Paragraph("6) Sonuc ve Sonraki Adimlar", styles["h1"]))
    final_items = [
        "Anomali analizi, haber baglami ve 2035 projeksiyonu tek pipeline icinde birlestirildi.",
        "Sistem her yeni veri geldiginde ayni scriptlerle tekrar calisabilir.",
        "Haber katalogu buyutuldukce eslesme kapsami artar; ozellikle 1980-2000 donemi icin arsiv genisletmesi onerilir.",
        "Model tarafinda walk-forward yeniden egitim ve fusion agirlik guncellemesi periyodik yapilmalidir.",
    ]
    for item in final_items:
        story.append(Paragraph(f"- {item}", styles["body"]))

    refs = [
        "NOAA CPC ONI: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php",
        "NOAA CPC NAO index: https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii",
        "NASA 2010 blocking event: https://earthobservatory.nasa.gov/images/44815/russian-fires-and-pakistan-floods-linked",
        "Reuters / AA / Euronews + yerel basin basliklari (Hurriyet, Cumhuriyet, Milliyet, TRT vb.)",
    ]
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("Kaynaklar", styles["h2"]))
    for r in refs:
        story.append(Paragraph(f"- {r}", styles["small"]))

    doc = SimpleDocTemplate(
        str(args.output_pdf),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.6 * cm,
        title="Hackhaton Final Raporu",
        author="KRDAE Hackhaton Pipeline",
        subject="Anomali + Haber + Forecast Fusion",
    )
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(f"Wrote: {args.output_pdf}")


if __name__ == "__main__":
    main()
