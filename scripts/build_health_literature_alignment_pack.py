#!/usr/bin/env python3
"""Build a public-friendly health risk + literature-alignment package.

Inputs:
- model comparison CSV from recent runs
- sensitivity summary CSVs for quant and strong models

Outputs (under output/health_impact by default):
- halk_dili_ozet_<date>.csv
- literatur_tutarlilik_kontrolu_<date>.csv
- halk_dili_aciklama_<date>.md
- halk_dili_risk_gorsel_<date>.png
- literatur_uyum_trafik_isigi_<date>.csv
- literatur_uyum_model_ozet_<date>.csv
- literatur_uyum_trafik_isigi_<date>.png
- devam_literatur_uyum_ozeti_<date>.md
- halk_dili_hastalik_ozet_<date>.csv
- halk_dili_hastalik_karsilastirma_<date>.png
- halk_dili_sik_sorular_<date>.md
- halk_dili_oncelik_matrisi_<date>.csv
- halk_dili_oncelik_matrisi_<date>.png
- halk_dili_eylem_plani_<date>.csv
- halk_dili_eylem_takvimi_<date>.png
- halk_dili_haftalik_is_plani_<date>.csv
- halk_dili_haftalik_is_ozeti_<date>.md
- halk_dili_haftalik_is_yuku_<date>.png
- halk_dili_hafta1_gorev_listesi_<date>.csv
- halk_dili_hafta1_toplanti_ajandasi_<date>.md
- halk_dili_haftalik_kpi_takip_<date>.csv
- halk_dili_kpi_alarm_kurallari_<date>.csv
- halk_dili_kpi_alarm_ozeti_<date>.md
- halk_dili_kpi_alarm_panosu_<date>.png
- halk_dili_entegre_bulgular_panosu_<date>.png
- halk_dili_entegre_bulgular_ozeti_<date>.md
- halk_dili_mudahale_senaryolari_<date>.csv
- halk_dili_mudahale_senaryo_ozeti_<date>.md
- halk_dili_mudahale_senaryolari_<date>.png
- halk_dili_operasyon_hazirlik_matrisi_<date>.csv
- halk_dili_operasyon_hazirlik_ozeti_<date>.md
- halk_dili_operasyon_hazirlik_panosu_<date>.png
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build health literature alignment package.")
    p.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("output/health_impact/model_comparison_summary_stable_calibrated.csv"),
        help="Model comparison CSV (quant vs strong).",
    )
    p.add_argument(
        "--quant-sensitivity-csv",
        type=Path,
        default=Path("output/health_impact/quant_duzenlenmis_run/sensitivity_genis/sensitivity_summary.csv"),
        help="Quant model sensitivity summary CSV.",
    )
    p.add_argument(
        "--strong-sensitivity-csv",
        type=Path,
        default=Path("output/health_impact/strong_duzenlenmis_run/sensitivity_genis/sensitivity_summary.csv"),
        help="Strong model sensitivity summary CSV.",
    )
    p.add_argument(
        "--core-literature-csv",
        type=Path,
        default=None,
        help="Core literature table for note text.",
    )
    p.add_argument(
        "--disease-summary-csv",
        type=Path,
        default=Path("output/health_impact/health_all_disease_summary.csv"),
        help="Disease-level climate-health summary CSV.",
    )
    p.add_argument(
        "--disease-matrix-csv",
        type=Path,
        default=Path("output/health_impact/health_all_disease_etiology_matrix.csv"),
        help="Disease x etiology matrix CSV.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/health_impact"),
        help="Output directory.",
    )
    p.add_argument(
        "--tag-date",
        type=str,
        default=date.today().isoformat(),
        help="Date tag for output filenames (YYYY-MM-DD).",
    )
    return p.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input CSV: {path}")
    return pd.read_csv(path)


def resolve_comparison_csv(path: Path) -> Path:
    if path.exists():
        return path
    candidates = [
        path.parent / "model_comparison_summary_stable_calibrated.csv",
        path.parent / "model_comparison_summary_duzenlenmis_run.csv",
        path.parent / "model_comparison_summary.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"Missing comparison CSV: {path} and fallback candidates {candidates}")


def resolve_core_literature_csv(path: Path | None, out_dir: Path) -> Path:
    if path is not None and path.exists():
        return path
    candidates = sorted(out_dir.glob("akademik_kanit_cekirdek_modelleme_*.csv"))
    if candidates:
        return candidates[-1]
    raise SystemExit("Missing core literature CSV. Provide --core-literature-csv or create akademik_kanit_cekirdek_modelleme_*.csv")


def build_halk_ozet(cmp: pd.DataFrame, out_dir: Path, tag_date: str) -> tuple[Path, Path, Path]:
    name_map = {
        "quant_duzenlenmis_run": "Quant (Daha Ilimli)",
        "strong_duzenlenmis_run": "Strong (Daha Sert)",
    }
    s = cmp.copy()
    s["model_tr"] = s["model"].map(name_map).fillna(s["model"])
    s["bugun_risk_endeksi"] = 100.0
    s["gelecek_risk_endeksi"] = (s["future_rr_mean"] * 100.0).round(1)
    s["genel_risk_artisi_yuzde"] = ((s["future_rr_mean"] - 1.0) * 100.0).round(1)
    s["ek_risk_payi_af_yuzde_puan"] = (s["future_af_mean"] * 100.0).round(1)
    s["isi_indeksi_degisim_c"] = s["delta_hi_mean_c"].round(2)

    def simple_msg(row: pd.Series) -> str:
        inc = float(row["genel_risk_artisi_yuzde"])
        hi = float(row["isi_indeksi_degisim_c"])
        name = str(row["model_tr"])
        if inc < 2:
            return f"{name}: riskte sinirli artis (+%{inc}); isi artisi {hi}C."
        if inc < 10:
            return f"{name}: orta duzey risk artisi (+%{inc}); isi artisi {hi}C."
        return f"{name}: yuksek risk artisi (+%{inc}); isi artisi {hi}C."

    s["halk_dili_mesaj"] = s.apply(simple_msg, axis=1)
    keep = [
        "model_tr",
        "bugun_risk_endeksi",
        "gelecek_risk_endeksi",
        "genel_risk_artisi_yuzde",
        "ek_risk_payi_af_yuzde_puan",
        "isi_indeksi_degisim_c",
        "future_threshold_exceed_share",
        "future_wet_hot_share",
        "halk_dili_mesaj",
    ]
    h = s[keep].copy().rename(
        columns={
            "model_tr": "model",
            "future_threshold_exceed_share": "esik_ustu_ay_orani",
            "future_wet_hot_share": "islak_sicak_ay_orani",
        }
    )
    h["esik_ustu_ay_orani"] = (h["esik_ustu_ay_orani"] * 100).round(1)
    h["islak_sicak_ay_orani"] = (h["islak_sicak_ay_orani"] * 100).round(1)
    h = h.sort_values("gelecek_risk_endeksi", ascending=False).reset_index(drop=True)

    csv_path = out_dir / f"halk_dili_ozet_{tag_date}.csv"
    h.to_csv(csv_path, index=False)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e4572e" if "Strong" in x else "#2d7ff9" for x in h["model"]]
    ax.bar(h["model"], h["gelecek_risk_endeksi"], color=colors)
    ax.axhline(100, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Halk Dili Risk Karsilastirmasi (100 = Bugun)")
    ax.set_ylabel("Gelecek Risk Endeksi")
    ymax = max(125.0, float(h["gelecek_risk_endeksi"].max()) + 5.0)
    ax.set_ylim(95, ymax)
    for i, v in enumerate(h["gelecek_risk_endeksi"]):
        ax.text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=11)
    fig.tight_layout()
    img_path = out_dir / f"halk_dili_risk_gorsel_{tag_date}.png"
    fig.savefig(img_path, dpi=180)
    plt.close(fig)

    return csv_path, img_path, h


def consistency_label(row: pd.Series) -> pd.Series:
    direction_ok = bool(float(row["future_rr_mean"]) >= float(row["baseline_rr_mean"]))
    d_hi = float(row["delta_hi_mean_c"])
    wet_share = float(row["future_wet_hot_share"])

    if d_hi <= 3.0:
        magnitude = "uyumlu"
    elif d_hi <= 6.0:
        magnitude = "sinirda"
    else:
        magnitude = "uyumsuz_olasi"

    if wet_share >= 0.80:
        regime = "asiri_rejim"
    elif wet_share >= 0.40:
        regime = "yuksek_rejim"
    else:
        regime = "normal_rejim"

    if direction_ok and magnitude == "uyumlu" and regime == "normal_rejim":
        overall = "celiski_yok"
    elif direction_ok and magnitude in {"uyumlu", "sinirda"}:
        overall = "kismi_celiski_riski"
    else:
        overall = "belirgin_celiski_riski"

    if overall == "celiski_yok":
        note = "Genel literatur yonu ile uyumlu; buyukluk makul."
    elif overall == "kismi_celiski_riski":
        note = "Yon uyumlu ama buyukluk/rejim varsayimlari hassasiyet ister."
    else:
        note = "Yon uyumlu olsa da buyukluk ve/veya rejim literaturle belirgin gerilimli."

    return pd.Series(
        {
            "yon_tutarliligi": direction_ok,
            "buyukluk_degerlendirme": magnitude,
            "rejim_degerlendirme": regime,
            "genel_literatur_uyumu": overall,
            "yorum": note,
        }
    )


def build_literature_check(cmp: pd.DataFrame, out_dir: Path, tag_date: str) -> tuple[Path, pd.DataFrame]:
    name_map = {
        "quant_duzenlenmis_run": "Quant (Daha Ilimli)",
        "strong_duzenlenmis_run": "Strong (Daha Sert)",
    }
    ck = cmp.copy()
    ck["model_tr"] = ck["model"].map(name_map).fillna(ck["model"])
    extra = ck.apply(consistency_label, axis=1)
    ck = pd.concat([ck, extra], axis=1)

    cols = [
        "model",
        "model_tr",
        "baseline_rr_mean",
        "future_rr_mean",
        "delta_hi_mean_c",
        "future_wet_hot_share",
        "yon_tutarliligi",
        "buyukluk_degerlendirme",
        "rejim_degerlendirme",
        "genel_literatur_uyumu",
        "yorum",
    ]
    ck = ck[cols].copy()
    out_csv = out_dir / f"literatur_tutarlilik_kontrolu_{tag_date}.csv"
    ck.to_csv(out_csv, index=False)
    return out_csv, ck


def traffic_label(row: pd.Series) -> str:
    rr = float(row["future_rr_mean"])
    wet = float(row["future_wet_hot_share"])
    ood = float(row["future_out_of_distribution_share"])
    if rr <= 1.08 and wet <= 0.30 and ood <= 0.20:
        return "yesil_uyumlu"
    if rr <= 1.15 and wet <= 0.60 and ood <= 0.40:
        return "sari_sinirda"
    return "kirmizi_celiski_riski"


def build_traffic_outputs(q_df: pd.DataFrame, s_df: pd.DataFrame, out_dir: Path, tag_date: str) -> tuple[Path, Path, Path, Path]:
    q = q_df.copy()
    s = s_df.copy()
    q["model"] = "Quant (Daha Ilimli)"
    s["model"] = "Strong (Daha Sert)"
    df = pd.concat([q, s], ignore_index=True)
    df = df[df["status"] == "ok"].copy()
    df["uyum_etiketi"] = df.apply(traffic_label, axis=1)

    summary = (
        df.groupby(["model", "uyum_etiketi"], as_index=False)
        .size()
        .rename(columns={"size": "senaryo_sayisi"})
    )
    totals = df.groupby("model").size().to_dict()
    summary["toplam"] = summary["model"].map(totals)
    summary["oran_yuzde"] = (summary["senaryo_sayisi"] / summary["toplam"] * 100).round(1)
    sum_csv = out_dir / f"literatur_uyum_trafik_isigi_{tag_date}.csv"
    summary.to_csv(sum_csv, index=False)

    model_stats = (
        df.groupby("model", as_index=False)
        .agg(
            senaryo_sayisi=("scenario_id", "size"),
            min_future_rr=("future_rr_mean", "min"),
            max_future_rr=("future_rr_mean", "max"),
            ort_future_rr=("future_rr_mean", "mean"),
            min_wet_hot=("future_wet_hot_share", "min"),
            max_wet_hot=("future_wet_hot_share", "max"),
        )
    )
    model_csv = out_dir / f"literatur_uyum_model_ozet_{tag_date}.csv"
    model_stats.to_csv(model_csv, index=False)

    pivot = summary.pivot(index="model", columns="uyum_etiketi", values="oran_yuzde").fillna(0)
    cols = ["yesil_uyumlu", "sari_sinirda", "kirmizi_celiski_riski"]
    for c in cols:
        if c not in pivot.columns:
            pivot[c] = 0.0
    pivot = pivot[cols]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    color_map = {"yesil_uyumlu": "#2ca02c", "sari_sinirda": "#f1c40f", "kirmizi_celiski_riski": "#e74c3c"}
    left = [0.0] * len(pivot)
    for c in cols:
        vals = pivot[c].values
        ax.barh(pivot.index, vals, left=left, color=color_map[c], label=c)
        left = [l + float(v) for l, v in zip(left, vals)]
    for y, (_, row) in enumerate(pivot.iterrows()):
        ax.text(
            101,
            y,
            f"{row['yesil_uyumlu']:.1f}% / {row['sari_sinirda']:.1f}% / {row['kirmizi_celiski_riski']:.1f}%",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, 115)
    ax.set_xlabel("Senaryo Orani (%)")
    ax.set_title("Literatur Uyum Trafik Isigi (2026-2035)")
    ax.legend(title="Etiket", loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    img = out_dir / f"literatur_uyum_trafik_isigi_{tag_date}.png"
    fig.savefig(img, dpi=180)
    plt.close(fig)

    def pct(model: str, label: str) -> float:
        r = summary[(summary["model"] == model) & (summary["uyum_etiketi"] == label)]
        if r.empty:
            return 0.0
        return float(r["oran_yuzde"].iloc[0])

    lines = [
        "# Devam Ozeti: Literatur Uyum Trafik Isigi",
        "",
        "- Yesil: genel literaturle uyumlu ve makul buyukluk",
        "- Sari: yon uyumlu ama buyukluk varsayimlari sinirda",
        "- Kirmizi: belirgin celiski riski",
        "",
        (
            f"- Quant (Daha Ilimli): yesil %{pct('Quant (Daha Ilimli)','yesil_uyumlu'):.1f}, "
            f"sari %{pct('Quant (Daha Ilimli)','sari_sinirda'):.1f}, "
            f"kirmizi %{pct('Quant (Daha Ilimli)','kirmizi_celiski_riski'):.1f}"
        ),
        (
            f"- Strong (Daha Sert): yesil %{pct('Strong (Daha Sert)','yesil_uyumlu'):.1f}, "
            f"sari %{pct('Strong (Daha Sert)','sari_sinirda'):.1f}, "
            f"kirmizi %{pct('Strong (Daha Sert)','kirmizi_celiski_riski'):.1f}"
        ),
        "",
    ]
    md = out_dir / f"devam_literatur_uyum_ozeti_{tag_date}.md"
    md.write_text("\n".join(lines), encoding="utf-8")

    return sum_csv, model_csv, img, md


def build_public_markdown(
    halk_df: pd.DataFrame,
    literature_ck: pd.DataFrame,
    core_literature_count: int,
    out_dir: Path,
    tag_date: str,
) -> Path:
    lines = [
        "# Halk Dili Ozet ve Literatur Tutarlilik Kontrolu",
        "",
        "## 1) Ne Anlama Geliyor?",
        "- `Risk endeksi 100` bugunku seviyeyi ifade eder.",
        "- `Gelecek risk endeksi` 100'un ustune cikarsa risk artmis demektir.",
        "- `AF` (atfedilebilir pay), risk artisinin ne kadarinin isi stresiyle iliskili oldugunu ozetler.",
        "",
        "## 2) Sonuc (Cok Kisa)",
    ]
    for _, r in halk_df.sort_values("gelecek_risk_endeksi", ascending=False).iterrows():
        lines.append(
            f"- {r['model']}: gelecek risk endeksi {r['gelecek_risk_endeksi']}, "
            f"artis +%{r['genel_risk_artisi_yuzde']}, AF %{r['ek_risk_payi_af_yuzde_puan']}."
        )
    lines += [
        "",
        "## 3) Literaturle Karsilastirma",
        f"- Kontrol, cekirdek literatur setindeki {core_literature_count} calisma ekseninde yapildi.",
    ]
    for _, r in literature_ck.iterrows():
        lines.append(f"- {r['model_tr']}: {r['genel_literatur_uyumu']} ({r['yorum']})")
    lines += [
        "",
        "## 4) Pratik Okuma",
        "- Quant sonucu: literatur yonu ile uyumlu ve daha ihtiyatli bir risk artisi veriyor.",
        "- Strong sonucu: riskin yonu dogru olsa da 2026-2035 icin buyukluk cok sert; bu nedenle ek dogrulama gerekir.",
        "",
    ]
    path = out_dir / f"halk_dili_aciklama_{tag_date}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _normalize_model_label(value: str) -> str:
    t = str(value).lower()
    if "strong" in t:
        return "strong"
    if "quant" in t:
        return "quant"
    return t


def _risk_band(score: float) -> str:
    if score >= 1.50:
        return "yuksek"
    if score >= 0.60:
        return "orta"
    if score >= 0.15:
        return "dusuk"
    return "cok_dusuk"


def _agreement_band(diff: float) -> str:
    if diff <= 0.40:
        return "yuksek"
    if diff <= 1.00:
        return "orta"
    return "dusuk"


def _evidence_band(coverage: float, agreement: str) -> str:
    if coverage >= 0.22 and agreement == "yuksek":
        return "orta-yuksek"
    if coverage >= 0.22:
        return "orta"
    return "dusuk"


def _risk_score(label: str) -> int:
    m = {"cok_dusuk": 0, "dusuk": 1, "orta": 2, "yuksek": 3}
    return m.get(str(label), 0)


def _evidence_score(label: str) -> int:
    m = {"dusuk": 1, "orta": 2, "orta-yuksek": 3}
    return m.get(str(label), 1)


def _agreement_score(label: str) -> int:
    m = {"dusuk": 1, "orta": 2, "yuksek": 3}
    return m.get(str(label), 1)


def _owner_for_disease(name: str) -> str:
    n = str(name).lower()
    if "kardiyovask" in n:
        return "Kardiyoloji + Acil Servis"
    if "solunum" in n or "astım" in n or "koah" in n:
        return "Gogus Hastaliklari + Acil Servis"
    if "bobrek" in n:
        return "Nefroloji + Is Sagligi"
    if "anne-bebek" in n or "dogum" in n:
        return "Kadin Dogum + Aile Sagligi"
    if "cilt kanseri" in n:
        return "Dermatoloji + Is Sagligi"
    if "vektor" in n or "su-gida" in n or "enfeksiyon" in n:
        return "Bulasici Hastaliklar + Cevre Sagligi"
    if "ruh sagligi" in n:
        return "Psikiyatri + Toplum Ruh Sagligi"
    if "yaralanma" in n or "is kazasi" in n:
        return "Acil Servis + Is Sagligi"
    return "Halk Sagligi Koordinasyon"


def _action_window(action: str) -> tuple[int, int, str, str, str]:
    if action == "acil_eylem":
        return (0, 14, "0-7 gun", "haftalik", "3 gun ust uste sicak dalga veya acil basvuru artis sinyali")
    if action == "hedefli_onlem":
        return (7, 45, "7-30 gun", "2 haftada bir", "mevsimsel risk artisinda hedefli saha onlemi")
    if action == "yakindan_izlem":
        return (14, 60, "14-45 gun", "aylik", "risk trendinde kalici yukselis")
    return (30, 90, "30-90 gun", "aylik", "rutin izlemde sapma tespiti")


def _action_goal(action: str) -> str:
    if action == "acil_eylem":
        return "Ilk 2 haftada hizli uyarilarla hassas gruplara koruyucu erisim"
    if action == "hedefli_onlem":
        return "Ilk ayda hedefli kurum/ilce bazli onlem paketi"
    if action == "yakindan_izlem":
        return "45 gun icinde risk trendini dengelemek"
    return "90 gunluk donemde stabil izlem ve periyodik raporlama"


def build_weekly_action_outputs(plan_df: pd.DataFrame, out_dir: Path, tag_date: str) -> tuple[Path, Path, Path]:
    week_bins = [
        ("hafta_1", "0-7", 0, 7),
        ("hafta_2", "8-14", 8, 14),
        ("hafta_3_4", "15-30", 15, 30),
        ("hafta_5_6", "31-45", 31, 45),
        ("hafta_7_9", "46-60", 46, 60),
        ("hafta_10_13", "61-90", 61, 90),
    ]
    action_labels = ["acil_eylem", "hedefli_onlem", "yakindan_izlem", "rutin_izlem"]
    rows: list[dict[str, object]] = []
    for label, day_range, start, end in week_bins:
        d = plan_df[(plan_df["baslangic_gunu"] <= end) & (plan_df["bitis_gunu"] >= start)].copy()
        counts = d["aksiyon_onceligi"].astype(str).value_counts().to_dict()
        top_focus = ", ".join(d.sort_values("oncelik_puani", ascending=False)["disease_group_tr"].head(3).tolist())
        rec: dict[str, object] = {
            "hafta": label,
            "gun_araligi": day_range,
            "toplam_aktif_gorev": int(len(d)),
            "ort_oncelik_puani": float(d["oncelik_puani"].mean()) if len(d) else 0.0,
            "odak_hastaliklar": top_focus,
        }
        for a in action_labels:
            rec[a] = int(counts.get(a, 0))
        rows.append(rec)
    weekly_df = pd.DataFrame(rows)

    out_csv = out_dir / f"halk_dili_haftalik_is_plani_{tag_date}.csv"
    weekly_df.to_csv(out_csv, index=False)

    lines = [
        f"# Haftalik Is Ozeti ({tag_date})",
        "",
        "Ilk 90 gun, haftalik is yuku ve oncelik dagilimi:",
        "",
    ]
    for _, r in weekly_df.iterrows():
        lines.append(
            f"- {r['hafta']} ({r['gun_araligi']} gun): aktif={int(r['toplam_aktif_gorev'])}, "
            f"acil={int(r['acil_eylem'])}, hedefli={int(r['hedefli_onlem'])}, "
            f"yakindan={int(r['yakindan_izlem'])}, rutin={int(r['rutin_izlem'])}, "
            f"ort oncelik={float(r['ort_oncelik_puani']):.2f}."
        )
        if str(r["odak_hastaliklar"]).strip():
            lines.append(f"  Odak: {r['odak_hastaliklar']}")
    lines.append("")
    out_md = out_dir / f"halk_dili_haftalik_is_ozeti_{tag_date}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    x = np.arange(len(weekly_df))
    color_action = {
        "acil_eylem": "#d73027",
        "hedefli_onlem": "#fc8d59",
        "yakindan_izlem": "#fee08b",
        "rutin_izlem": "#91bfdb",
    }
    bottom = np.zeros(len(weekly_df))
    for a in action_labels:
        vals = weekly_df[a].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=color_action[a], label=a)
        bottom = bottom + vals
    ax.set_xticks(x, weekly_df["hafta"].tolist())
    ax.set_ylabel("Aktif gorev sayisi")
    ax.set_title("Haftalik Is Yuk Dagilimi (Ilk 90 Gun)")
    ax.grid(axis="y", alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(x, weekly_df["ort_oncelik_puani"].to_numpy(dtype=float), color="#2c3e50", marker="o", linewidth=2.0, label="ort_oncelik")
    ax2.set_ylabel("Ortalama oncelik puani")
    ax2.set_ylim(0, max(3.2, float(weekly_df["ort_oncelik_puani"].max()) + 0.4))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, fontsize=8)

    out_img = out_dir / f"halk_dili_haftalik_is_yuku_{tag_date}.png"
    fig.tight_layout()
    fig.savefig(out_img, dpi=180)
    plt.close(fig)

    return out_csv, out_md, out_img


def build_week1_execution_pack(
    plan_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    out_dir: Path,
    tag_date: str,
) -> tuple[Path, Path, Path, pd.DataFrame]:
    week1 = plan_df[(plan_df["baslangic_gunu"] <= 7) & (plan_df["bitis_gunu"] >= 0)].copy()
    week1 = week1.sort_values(["oncelik_puani", "risk_puani"], ascending=False)
    out_week1_csv = out_dir / f"halk_dili_hafta1_gorev_listesi_{tag_date}.csv"
    week1_cols = [
        "disease_group_tr",
        "aksiyon_onceligi",
        "oncelik_puani",
        "baslama_suresi",
        "izlem_sikligi",
        "sorumlu_birim",
        "ilk_90_gun_hedefi",
        "tetikleyici_kural",
    ]
    week1[week1_cols].to_csv(out_week1_csv, index=False)

    c = week1["aksiyon_onceligi"].astype(str).value_counts().to_dict()
    top_focus = ", ".join(week1["disease_group_tr"].head(3).tolist())
    agenda_lines = [
        f"# Hafta 1 Toplanti Ajandasi ({tag_date})",
        "",
        "## Toplanti Amaci",
        "- Ilk 7 gunde acil/hedefli gorevleri netlestirip sorumlu birimlere atamak.",
        "",
        "## 60 Dakika Akis",
        "1. 0-10 dk: Son risk ozeti ve model ayrisimi",
        "2. 10-25 dk: Acil eylem gerektiren hastalik gruplari",
        "3. 25-40 dk: Hedefli onlem ve saha koordinasyonu",
        "4. 40-50 dk: Izlem ve KPI takip mutabakati",
        "5. 50-60 dk: Gorev atama ve kapanis",
        "",
        "## Hafta 1 Dagilimi",
        f"- Toplam aktif gorev: {len(week1)}",
        f"- Acil/Hedefli/Yakindan/Rutin: {int(c.get('acil_eylem', 0))}/"
        f"{int(c.get('hedefli_onlem', 0))}/{int(c.get('yakindan_izlem', 0))}/{int(c.get('rutin_izlem', 0))}",
        f"- Ilk odak hastaliklar: {top_focus}",
        "",
        "## Ilk 5 Gorev",
    ]
    for _, r in week1.head(5).iterrows():
        agenda_lines.append(
            f"- {r['disease_group_tr']} | {r['aksiyon_onceligi']} | {r['sorumlu_birim']} | baslama {r['baslama_suresi']}"
        )
    agenda_lines.append("")
    out_agenda_md = out_dir / f"halk_dili_hafta1_toplanti_ajandasi_{tag_date}.md"
    out_agenda_md.write_text("\n".join(agenda_lines), encoding="utf-8")

    kpi = weekly_df.copy()
    kpi["hedef_aktif_gorev"] = kpi["toplam_aktif_gorev"]
    kpi["hedef_acil_eylem"] = kpi["acil_eylem"]
    kpi["hedef_hedefli_onlem"] = kpi["hedefli_onlem"]
    kpi["hedef_ort_oncelik_puani"] = kpi["ort_oncelik_puani"].round(2)
    kpi["gercek_aktif_gorev"] = ""
    kpi["gercek_acil_eylem"] = ""
    kpi["gercek_hedefli_onlem"] = ""
    kpi["gercek_ort_oncelik_puani"] = ""
    kpi["sapma_notu"] = ""
    out_kpi_csv = out_dir / f"halk_dili_haftalik_kpi_takip_{tag_date}.csv"
    keep = [
        "hafta",
        "gun_araligi",
        "hedef_aktif_gorev",
        "hedef_acil_eylem",
        "hedef_hedefli_onlem",
        "hedef_ort_oncelik_puani",
        "gercek_aktif_gorev",
        "gercek_acil_eylem",
        "gercek_hedefli_onlem",
        "gercek_ort_oncelik_puani",
        "sapma_notu",
    ]
    kpi_export = kpi[keep].copy()
    kpi_export.to_csv(out_kpi_csv, index=False)

    return out_week1_csv, out_agenda_md, out_kpi_csv, kpi_export


def build_kpi_alarm_outputs(kpi_df: pd.DataFrame, out_dir: Path, tag_date: str) -> tuple[Path, Path, Path]:
    base = kpi_df.copy()
    for c in [
        "hedef_aktif_gorev",
        "hedef_acil_eylem",
        "hedef_hedefli_onlem",
        "hedef_ort_oncelik_puani",
    ]:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)

    rules: list[dict[str, object]] = []

    def lower_thresholds(target: float) -> tuple[str, int, int, str, str]:
        if target <= 0:
            return (
                "olay_bazli_izlem",
                0,
                0,
                "hedef 0: sari alarm yerine olay kaydi izlenir",
                "hedef 0: kirmizi alarm yerine olay kaydi izlenir",
            )
        if target < 3:
            sari = max(0, int(np.floor(target - 1)))
            kirmizi = max(0, int(np.floor(target - 1)))
        else:
            sari = int(np.floor(target * 0.90))
            kirmizi = int(np.floor(target * 0.75))
        return (
            "altina_duserse",
            sari,
            kirmizi,
            "gercek < sari_esik ise sari_alarm",
            "gercek < kirmizi_esik ise kirmizi_alarm",
        )

    for _, row in base.iterrows():
        hafta = str(row["hafta"])
        gun = str(row["gun_araligi"])
        hedef_aktif = float(row["hedef_aktif_gorev"])
        hedef_acil = float(row["hedef_acil_eylem"])
        hedef_hedefli = float(row["hedef_hedefli_onlem"])
        hedef_oncelik = float(row["hedef_ort_oncelik_puani"])

        for kpi_name, hedef in [
            ("aktif_gorev", hedef_aktif),
            ("acil_eylem", hedef_acil),
            ("hedefli_onlem", hedef_hedefli),
        ]:
            alarm_yonu, sari, kirmizi, sari_kural, kirmizi_kural = lower_thresholds(hedef)
            rules.append(
                {
                    "hafta": hafta,
                    "gun_araligi": gun,
                    "kpi_adi": kpi_name,
                    "alarm_yonu": alarm_yonu,
                    "hedef_deger": int(round(hedef)),
                    "sari_esik": sari,
                    "kirmizi_esik": kirmizi,
                    "sari_kural": sari_kural,
                    "kirmizi_kural": kirmizi_kural,
                    "izlem_sikligi": "haftalik",
                }
            )

        rules.append(
            {
                "hafta": hafta,
                "gun_araligi": gun,
                "kpi_adi": "ort_oncelik_puani",
                "alarm_yonu": "ustune_cikarsa",
                "hedef_deger": round(hedef_oncelik, 2),
                "sari_esik": round(hedef_oncelik + 0.20, 2),
                "kirmizi_esik": round(hedef_oncelik + 0.35, 2),
                "sari_kural": "gercek > sari_esik ise sari_alarm",
                "kirmizi_kural": "gercek > kirmizi_esik ise kirmizi_alarm",
                "izlem_sikligi": "haftalik",
            }
        )

    rules_df = pd.DataFrame(rules)
    out_rules_csv = out_dir / f"halk_dili_kpi_alarm_kurallari_{tag_date}.csv"
    rules_df.to_csv(out_rules_csv, index=False)

    max_aktif = max(float(base["hedef_aktif_gorev"].max()), 1.0)
    max_acil = max(float(base["hedef_acil_eylem"].max()), 1.0)
    max_oncelik = max(float(base["hedef_ort_oncelik_puani"].max()), 1.0)
    pressure = (
        (base["hedef_aktif_gorev"] / max_aktif) * 0.45
        + (base["hedef_acil_eylem"] / max_acil) * 0.35
        + (base["hedef_ort_oncelik_puani"] / max_oncelik) * 0.20
    )
    base["operasyon_baski_skoru"] = pressure.round(3)
    base["planlanan_alarm_seviyesi"] = base["operasyon_baski_skoru"].map(
        lambda x: "yuksek" if x >= 0.75 else ("orta" if x >= 0.55 else "dusuk")
    )

    md_lines = [
        f"# KPI Alarm Ozeti ({tag_date})",
        "",
        "Bu dosya, haftalik hedef-gercek takip tablosu icin alarm esiklerini tanimlar.",
        "- Yesil: hedefe uygun aralik",
        "- Sari: erken uyari",
        "- Kirmizi: yonetici eskalasyonu gerektirir",
        "",
        "## Haftalik Planlanan Operasyon Baskisi",
    ]
    for _, r in base.iterrows():
        md_lines.append(
            f"- {r['hafta']} ({r['gun_araligi']}): planlanan alarm seviyesi={r['planlanan_alarm_seviyesi']} "
            f"(baski skoru {float(r['operasyon_baski_skoru']):.2f})"
        )
    md_lines.extend(
        [
            "",
            "## Alarm Kurallari (Ozet)",
            "- aktif_gorev, acil_eylem, hedefli_onlem: gercek deger esigin altina inerse alarm.",
            "- ort_oncelik_puani: gercek deger esigin ustune cikarsa alarm.",
            "- Hedefi 0 olan satirlar alarm degil, olay-bazli izlem satiri olarak ele alinmalidir.",
            "- Haftalik KPI toplantisinda once kirmizi sonra sari alarmlar gorusulur.",
            "",
            "## Uyari",
            "- Bu panel operasyonel takip icindir; klinik tani karari yerine gecmez.",
            "- Esikler saha verisi geldikce 4-6 haftada bir yeniden kalibre edilmelidir.",
            "",
        ]
    )
    out_md = out_dir / f"halk_dili_kpi_alarm_ozeti_{tag_date}.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), sharex=True)
    x = np.arange(len(base))
    hafta_labels = base["hafta"].astype(str).tolist()

    def style_axis(ax, title: str, ylabel: str) -> None:
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x, hafta_labels, rotation=25, ha="right")

    def draw_lower_bound(ax, target_col: str, title: str) -> None:
        target = base[target_col].to_numpy(dtype=float)
        yellow = np.floor(target * 0.90)
        red = np.floor(target * 0.80)
        ymax = max(float(np.max(target)), 1.0) * 1.35 + 1.0
        ax.fill_between(x, 0, red, color="#f8d7da", alpha=0.70)
        ax.fill_between(x, red, yellow, color="#fff3cd", alpha=0.70)
        ax.fill_between(x, yellow, ymax, color="#d4edda", alpha=0.60)
        ax.plot(x, target, marker="o", linewidth=2.0, color="#1f77b4", label="hedef")
        ax.plot(x, yellow, linestyle="--", linewidth=1.5, color="#d39e00", label="sari_esik")
        ax.plot(x, red, linestyle="--", linewidth=1.5, color="#c82333", label="kirmizi_esik")
        ax.set_ylim(0, ymax)
        style_axis(ax, title, "Deger")

    def draw_upper_bound(ax, target_col: str, title: str) -> None:
        target = base[target_col].to_numpy(dtype=float)
        yellow = target + 0.20
        red = target + 0.35
        ymax = max(float(np.max(red)), 0.5) * 1.25 + 0.1
        ax.fill_between(x, 0, yellow, color="#d4edda", alpha=0.60)
        ax.fill_between(x, yellow, red, color="#fff3cd", alpha=0.70)
        ax.fill_between(x, red, ymax, color="#f8d7da", alpha=0.70)
        ax.plot(x, target, marker="o", linewidth=2.0, color="#1f77b4", label="hedef")
        ax.plot(x, yellow, linestyle="--", linewidth=1.5, color="#d39e00", label="sari_esik")
        ax.plot(x, red, linestyle="--", linewidth=1.5, color="#c82333", label="kirmizi_esik")
        ax.set_ylim(0, ymax)
        style_axis(ax, title, "Puan")

    draw_lower_bound(axes[0, 0], "hedef_aktif_gorev", "KPI Alarm: Aktif Gorev")
    draw_lower_bound(axes[0, 1], "hedef_acil_eylem", "KPI Alarm: Acil Eylem")
    draw_lower_bound(axes[1, 0], "hedef_hedefli_onlem", "KPI Alarm: Hedefli Onlem")
    draw_upper_bound(axes[1, 1], "hedef_ort_oncelik_puani", "KPI Alarm: Ortalama Oncelik")
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=True)
    fig.suptitle("Haftalik KPI Alarm Panosu (Hedef ve Alarm Esikleri)", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = out_dir / f"halk_dili_kpi_alarm_panosu_{tag_date}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return out_rules_csv, out_md, out_png


def build_integrated_findings_dashboard(
    merged_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    alarm_rules_df: pd.DataFrame,
    out_dir: Path,
    tag_date: str,
) -> tuple[Path, Path]:
    risk_colors = {
        "yuksek": "#d73027",
        "orta": "#fc8d59",
        "dusuk": "#91bfdb",
        "cok_dusuk": "#4575b4",
    }

    top = merged_df.sort_values("oncelik_puani", ascending=False).head(8).copy()
    weekly = weekly_df.copy()
    for c in ["toplam_aktif_gorev", "acil_eylem", "hedefli_onlem", "yakindan_izlem", "rutin_izlem", "ort_oncelik_puani"]:
        if c in weekly.columns:
            weekly[c] = pd.to_numeric(weekly[c], errors="coerce").fillna(0.0)

    alarm = alarm_rules_df.copy()
    if not alarm.empty:
        alarm["alarm_yonu"] = alarm["alarm_yonu"].astype(str)
    alarm_pivot = (
        alarm.groupby(["hafta", "alarm_yonu"], as_index=False)
        .size()
        .pivot(index="hafta", columns="alarm_yonu", values="size")
        .fillna(0.0)
        if not alarm.empty
        else pd.DataFrame()
    )
    for c in ["altina_duserse", "ustune_cikarsa", "olay_bazli_izlem"]:
        if c not in alarm_pivot.columns:
            alarm_pivot[c] = 0.0
    if not alarm_pivot.empty and "hafta" in weekly.columns:
        alarm_pivot = alarm_pivot.reindex(weekly["hafta"].astype(str).tolist()).fillna(0.0)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: Top disease priorities
    ax1 = axes[0, 0]
    y = np.arange(len(top))
    bar_colors = [risk_colors.get(str(x), "#666666") for x in top["risk_duzeyi"]]
    ax1.barh(y, top["oncelik_puani"].to_numpy(dtype=float), color=bar_colors, alpha=0.9)
    ax1.set_yticks(y, top["disease_group_tr"].astype(str).tolist())
    ax1.invert_yaxis()
    ax1.set_xlabel("Oncelik puani")
    ax1.set_title("En Kritik Hastalik Gruplari")
    ax1.grid(axis="x", alpha=0.25)
    for i, (_, r) in enumerate(top.iterrows()):
        ax1.text(
            float(r["oncelik_puani"]) + 0.02,
            i,
            str(r["aksiyon_onceligi"]),
            va="center",
            fontsize=8,
        )

    # Panel 2: Risk vs evidence
    ax2 = axes[0, 1]
    size = (merged_df["oncelik_puani"] * 200).clip(90, 750)
    ax2.scatter(
        merged_df["kanit_puani"],
        merged_df["risk_puani"],
        s=size,
        c=[risk_colors.get(str(x), "#666666") for x in merged_df["risk_duzeyi"]],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    ax2.set_xticks([1, 2, 3], ["dusuk", "orta", "orta-yuksek"])
    ax2.set_yticks([0, 1, 2, 3], ["cok_dusuk", "dusuk", "orta", "yuksek"])
    ax2.set_xlabel("Sayisal kanit guveni")
    ax2.set_ylabel("Risk puani")
    ax2.set_title("Risk-Kanit Dagilimi")
    ax2.grid(alpha=0.25)

    # Panel 3: Weekly workload stack
    ax3 = axes[1, 0]
    weeks = weekly["hafta"].astype(str).tolist()
    x = np.arange(len(weeks))
    weekly_colors = {
        "acil_eylem": "#d73027",
        "hedefli_onlem": "#fc8d59",
        "yakindan_izlem": "#fee08b",
        "rutin_izlem": "#91bfdb",
    }
    bottom = np.zeros(len(weeks))
    for col in ["acil_eylem", "hedefli_onlem", "yakindan_izlem", "rutin_izlem"]:
        vals = weekly[col].to_numpy(dtype=float) if col in weekly.columns else np.zeros(len(weeks))
        ax3.bar(x, vals, bottom=bottom, color=weekly_colors[col], label=col)
        bottom = bottom + vals
    ax3_t = ax3.twinx()
    pr = weekly["ort_oncelik_puani"].to_numpy(dtype=float) if "ort_oncelik_puani" in weekly.columns else np.zeros(len(weeks))
    ax3_t.plot(x, pr, color="#2c3e50", marker="o", linewidth=2.0, label="ort_oncelik")
    ax3.set_xticks(x, weeks, rotation=25, ha="right")
    ax3.set_ylabel("Aktif gorev")
    ax3_t.set_ylabel("Ort oncelik")
    ax3.set_title("Haftalik Is Yuku ve Oncelik")
    ax3.grid(axis="y", alpha=0.25)
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3_t.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, frameon=True)

    # Panel 4: Alarm mix per week
    ax4 = axes[1, 1]
    if alarm_pivot.empty:
        ax4.text(0.5, 0.5, "Alarm kurali bulunamadi", ha="center", va="center", fontsize=11)
        ax4.set_axis_off()
    else:
        x2 = np.arange(len(alarm_pivot.index))
        btm = np.zeros(len(alarm_pivot.index))
        alarm_colors = {
            "altina_duserse": "#f39c12",
            "ustune_cikarsa": "#c0392b",
            "olay_bazli_izlem": "#7f8c8d",
        }
        for col in ["altina_duserse", "ustune_cikarsa", "olay_bazli_izlem"]:
            vals = alarm_pivot[col].to_numpy(dtype=float)
            ax4.bar(x2, vals, bottom=btm, color=alarm_colors[col], label=col)
            btm = btm + vals
        ax4.set_xticks(x2, alarm_pivot.index.astype(str).tolist(), rotation=25, ha="right")
        ax4.set_ylabel("Kural sayisi")
        ax4.set_title("Haftalik Alarm Kural Dagilimi")
        ax4.grid(axis="y", alpha=0.25)
        ax4.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle("Entegre Bulgular Panosu: Risk, Eylem, Haftalik Izlem ve Alarm", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = out_dir / f"halk_dili_entegre_bulgular_panosu_{tag_date}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    peak_week = ""
    if "toplam_aktif_gorev" in weekly.columns and not weekly.empty:
        peak = weekly.sort_values("toplam_aktif_gorev", ascending=False).iloc[0]
        peak_week = f"{peak['hafta']} ({int(peak['toplam_aktif_gorev'])} aktif gorev)"

    action_counts = merged_df["aksiyon_onceligi"].astype(str).value_counts().to_dict()
    alarm_counts = alarm["alarm_yonu"].astype(str).value_counts().to_dict() if not alarm.empty else {}
    md_lines = [
        f"# Entegre Bulgular Ozeti ({tag_date})",
        "",
        "## Hizli Okuma",
        f"- En yuksek oncelikli grup: {top.iloc[0]['disease_group_tr']} ({float(top.iloc[0]['oncelik_puani']):.2f})" if not top.empty else "- Oncelik verisi bulunamadi.",
        f"- Is yuku zirvesi: {peak_week}" if peak_week else "- Is yuku zirvesi hesaplanamadi.",
        (
            f"- Aksiyon dagilimi (acil/hedefli/yakindan/rutin): "
            f"{int(action_counts.get('acil_eylem', 0))}/"
            f"{int(action_counts.get('hedefli_onlem', 0))}/"
            f"{int(action_counts.get('yakindan_izlem', 0))}/"
            f"{int(action_counts.get('rutin_izlem', 0))}"
        ),
        (
            f"- Alarm kural dagilimi (alt/ust/olay-bazli): "
            f"{int(alarm_counts.get('altina_duserse', 0))}/"
            f"{int(alarm_counts.get('ustune_cikarsa', 0))}/"
            f"{int(alarm_counts.get('olay_bazli_izlem', 0))}"
        ),
        "",
        "## Not",
        "- Bu pano operasyonel karar destegi icindir; klinik tani karari yerine gecmez.",
        "",
    ]
    out_md = out_dir / f"halk_dili_entegre_bulgular_ozeti_{tag_date}.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    return out_png, out_md


def build_intervention_scenario_outputs(
    merged_df: pd.DataFrame,
    out_dir: Path,
    tag_date: str,
) -> tuple[Path, Path, Path]:
    base = merged_df.copy()
    base["konsensus_skor"] = pd.to_numeric(base["konsensus_skor"], errors="coerce").fillna(0.0)
    base["risk_puani"] = pd.to_numeric(base["risk_puani"], errors="coerce").fillna(0.0)
    base["oncelik_puani"] = pd.to_numeric(base["oncelik_puani"], errors="coerce").fillna(0.0)
    base["aksiyon_onceligi"] = base["aksiyon_onceligi"].astype(str)

    # Action-class effect assumptions for population-level operational planning.
    action_effect = {
        "acil_eylem": 0.18,
        "hedefli_onlem": 0.12,
        "yakindan_izlem": 0.06,
        "rutin_izlem": 0.03,
    }
    base["temel_yuk_endeksi"] = base["konsensus_skor"] * (1.0 + (base["risk_puani"] * 0.15))
    base["aksiyon_etki_tabani"] = base["aksiyon_onceligi"].map(action_effect).fillna(0.03)

    scenarios = [
        {
            "senaryo": "temel_plan",
            "kapsama_katsayisi": 1.00,
            "etki_carpani": 0.70,
            "aciklama": "Mevcut kapasite ve rutin saha uygulamasi.",
        },
        {
            "senaryo": "guclendirilmis_plan",
            "kapsama_katsayisi": 1.20,
            "etki_carpani": 0.90,
            "aciklama": "Hedefli saha uygulamasi + veri kalitesi iyilestirmesi.",
        },
        {
            "senaryo": "hizlandirilmis_entegrasyon",
            "kapsama_katsayisi": 1.35,
            "etki_carpani": 1.05,
            "aciklama": "Erken uyari + kurumlararasi hizli koordinasyon + yogun izlem.",
        },
    ]

    total_base = float(base["temel_yuk_endeksi"].sum())
    if total_base <= 0:
        total_base = 1.0

    rows: list[dict[str, object]] = []
    for sc in scenarios:
        multiplier = float(sc["kapsama_katsayisi"]) * float(sc["etki_carpani"])
        reduction_ratio = (base["aksiyon_etki_tabani"] * multiplier).clip(lower=0.0, upper=0.55)
        residual = base["temel_yuk_endeksi"] * (1.0 - reduction_ratio)
        total_residual = float(residual.sum())
        reduction_pct = max(0.0, min(90.0, (1.0 - (total_residual / total_base)) * 100.0))

        residual_signal = base["konsensus_skor"] * (1.0 - reduction_ratio)
        critical_groups = int((residual_signal >= 1.50).sum())
        medium_plus = int((residual_signal >= 0.60).sum())
        avoided_priority_points = float((base["oncelik_puani"] * reduction_ratio).sum())

        rows.append(
            {
                "senaryo": str(sc["senaryo"]),
                "kapsama_katsayisi": float(sc["kapsama_katsayisi"]),
                "etki_carpani": float(sc["etki_carpani"]),
                "beklenen_risk_azalimi_yuzde": round(reduction_pct, 2),
                "kalan_risk_endeksi_100": round(100.0 - reduction_pct, 2),
                "kritik_grup_sayisi_kalan": critical_groups,
                "orta_ve_ustu_grup_sayisi_kalan": medium_plus,
                "kacinilan_oncelik_puani": round(avoided_priority_points, 2),
                "aciklama": str(sc["aciklama"]),
            }
        )

    scenario_df = pd.DataFrame(rows).sort_values("beklenen_risk_azalimi_yuzde", ascending=False).reset_index(drop=True)
    out_csv = out_dir / f"halk_dili_mudahale_senaryolari_{tag_date}.csv"
    scenario_df.to_csv(out_csv, index=False)

    best = scenario_df.iloc[0]
    md_lines = [
        f"# Mudahale Senaryo Ozeti ({tag_date})",
        "",
        "Bu analiz, operasyonel uygulama yogunluguna gore beklenen goreli risk azalimi simulasyonu sunar.",
        "Yuzdeler modelleme varsayimina dayali planlama metrikleridir; klinik etkiyi birebir temsil etmez.",
        "",
        "## Sonuc",
        (
            f"- En yuksek etki: {best['senaryo']} | beklenen risk azalimi "
            f"%{float(best['beklenen_risk_azalimi_yuzde']):.2f}, kalan risk endeksi {float(best['kalan_risk_endeksi_100']):.2f}."
        ),
        (
            f"- Bu senaryoda kalan kritik grup sayisi: {int(best['kritik_grup_sayisi_kalan'])}, "
            f"kacinilan oncelik puani: {float(best['kacinilan_oncelik_puani']):.2f}."
        ),
        "",
        "## Senaryo Tablosu",
    ]
    for _, r in scenario_df.iterrows():
        md_lines.append(
            f"- {r['senaryo']}: risk azalimi %{float(r['beklenen_risk_azalimi_yuzde']):.2f}, "
            f"kritik kalan {int(r['kritik_grup_sayisi_kalan'])}, not: {r['aciklama']}"
        )
    md_lines.extend(
        [
            "",
            "## Uyari",
            "- Senaryo carpani, uygulama kapsami ve etkinligi icin muhafazakar varsayimlarla tanimlanmistir.",
            "- Yerel saha verisi geldikce katsayilar 4-8 haftada bir yeniden kalibre edilmelidir.",
            "",
        ]
    )
    out_md = out_dir / f"halk_dili_mudahale_senaryo_ozeti_{tag_date}.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    x = np.arange(len(scenario_df))
    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    bars = ax.bar(
        x,
        scenario_df["beklenen_risk_azalimi_yuzde"].to_numpy(dtype=float),
        color=colors[: len(scenario_df)],
        alpha=0.9,
        width=0.58,
        label="beklenen_risk_azalimi",
    )
    ax.set_xticks(x, scenario_df["senaryo"].astype(str).tolist())
    ax.set_ylabel("Beklenen risk azalimi (%)")
    ax.set_title("Mudahale Senaryolari: Beklenen Goreli Risk Azalimi")
    ax.grid(axis="y", alpha=0.25)
    for b in bars:
        y = float(b.get_height())
        ax.text(float(b.get_x()) + float(b.get_width()) / 2.0, y + 0.6, f"%{y:.1f}", ha="center", fontsize=10)

    ax2 = ax.twinx()
    crit = scenario_df["kritik_grup_sayisi_kalan"].to_numpy(dtype=float)
    ax2.plot(x, crit, color="#2c3e50", marker="o", linewidth=2.1, label="kritik_grup_sayisi_kalan")
    ax2.set_ylabel("Kalan kritik grup sayisi")
    ax2.set_ylim(0, max(1.0, float(np.max(crit)) + 1.0))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    out_png = out_dir / f"halk_dili_mudahale_senaryolari_{tag_date}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_csv, out_md, out_png


def build_operational_readiness_outputs(
    plan_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    out_dir: Path,
    tag_date: str,
) -> tuple[Path, Path, Path]:
    p = plan_df.copy()
    for c in ["oncelik_puani", "baslangic_gunu"]:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)

    action_cols = ["acil_eylem", "hedefli_onlem", "yakindan_izlem", "rutin_izlem"]
    grouped = p.groupby("sorumlu_birim", as_index=False).agg(
        toplam_gorev=("disease_group_tr", "size"),
        ort_oncelik_puani=("oncelik_puani", "mean"),
        ort_baslangic_gunu=("baslangic_gunu", "mean"),
        en_erken_baslangic_gunu=("baslangic_gunu", "min"),
    )

    counts = (
        p.pivot_table(
            index="sorumlu_birim",
            columns="aksiyon_onceligi",
            values="disease_group_tr",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    ops = grouped.merge(counts, on="sorumlu_birim", how="left").fillna(0.0)
    for c in action_cols:
        if c not in ops.columns:
            ops[c] = 0.0

    urgency_raw = (ops["acil_eylem"] * 3.0) + (ops["hedefli_onlem"] * 2.0) + (ops["yakindan_izlem"] * 1.0) + (ops["rutin_izlem"] * 0.5)
    urgency_norm = urgency_raw / max(float(urgency_raw.max()), 1.0)
    priority_norm = ops["ort_oncelik_puani"] / max(float(ops["ort_oncelik_puani"].max()), 1.0)
    time_pressure = (90.0 - ops["ort_baslangic_gunu"].clip(0, 90)) / 90.0

    ops["hazirlik_baski_puani"] = ((urgency_norm * 0.50) + (priority_norm * 0.35) + (time_pressure * 0.15)).round(3)
    ops["hazirlik_seviyesi"] = ops["hazirlik_baski_puani"].map(
        lambda x: "yuksek_baski" if x >= 0.67 else ("orta_baski" if x >= 0.45 else "dusuk_baski")
    )
    ops = ops.sort_values("hazirlik_baski_puani", ascending=False).reset_index(drop=True)

    # Add a shared operational context from week-1 workload.
    first_week_active = 0
    if not weekly_df.empty and "toplam_aktif_gorev" in weekly_df.columns:
        fw = pd.to_numeric(weekly_df.iloc[0]["toplam_aktif_gorev"], errors="coerce")
        if pd.notna(fw):
            first_week_active = int(fw)
    ops["ilk_hafta_toplam_aktif_gorev"] = first_week_active

    out_csv = out_dir / f"halk_dili_operasyon_hazirlik_matrisi_{tag_date}.csv"
    cols = [
        "sorumlu_birim",
        "toplam_gorev",
        "acil_eylem",
        "hedefli_onlem",
        "yakindan_izlem",
        "rutin_izlem",
        "ort_oncelik_puani",
        "en_erken_baslangic_gunu",
        "ort_baslangic_gunu",
        "hazirlik_baski_puani",
        "hazirlik_seviyesi",
        "ilk_hafta_toplam_aktif_gorev",
    ]
    ops[cols].to_csv(out_csv, index=False)

    top_units = ops.head(3).copy()
    md_lines = [
        f"# Operasyon Hazirlik Ozeti ({tag_date})",
        "",
        "Bu tablo, sorumlu birimlerin gorev yogunlugu ve zaman baskisini birlikte gosterir.",
        f"- Ilk hafta toplam aktif gorev: {first_week_active}",
        "",
        "## En Yuksek Baski Altindaki Birimler",
    ]
    for _, r in top_units.iterrows():
        md_lines.append(
            f"- {r['sorumlu_birim']}: baski puani {float(r['hazirlik_baski_puani']):.2f}, "
            f"gorev={int(r['toplam_gorev'])}, acil={int(r['acil_eylem'])}, hedefli={int(r['hedefli_onlem'])}, "
            f"en erken baslangic gunu={int(r['en_erken_baslangic_gunu'])}"
        )
    md_lines.extend(
        [
            "",
            "## Operasyon Yorumu",
            "- Yuksek_baski birimler icin haftalik degil, 3-4 gun aralikli koordinasyon onerilir.",
            "- Orta_baski birimlerde hedefli onlem takip listesi sabitlenmelidir.",
            "- Dusuk_baski birimler rutin izlemde kalabilir; ani sapmada eskalasyon uygulanir.",
            "",
        ]
    )
    out_md = out_dir / f"halk_dili_operasyon_hazirlik_ozeti_{tag_date}.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.6))

    # Panel A: owner x action heatmap
    ax1 = axes[0]
    heat = ops[action_cols].to_numpy(dtype=float)
    im = ax1.imshow(heat, cmap="YlOrRd", aspect="auto")
    ax1.set_yticks(np.arange(len(ops)), ops["sorumlu_birim"].astype(str).tolist())
    ax1.set_xticks(np.arange(len(action_cols)), action_cols, rotation=20, ha="right")
    ax1.set_title("Birim x Aksiyon Yogunlugu")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax1.text(j, i, f"{int(heat[i, j])}", ha="center", va="center", fontsize=8, color="#222222")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Gorev sayisi", fontsize=9)

    # Panel B: start-time pressure vs priority
    ax2 = axes[1]
    x = ops["ort_baslangic_gunu"].to_numpy(dtype=float)
    y = ops["ort_oncelik_puani"].to_numpy(dtype=float)
    s = (ops["toplam_gorev"].to_numpy(dtype=float) * 220.0).clip(120.0, 1200.0)
    c = ops["hazirlik_baski_puani"].to_numpy(dtype=float)
    sc = ax2.scatter(x, y, s=s, c=c, cmap="viridis", alpha=0.88, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Ortalama baslangic gunu (dusuk = daha acil)")
    ax2.set_ylabel("Ortalama oncelik puani")
    ax2.set_title("Zaman Baskisi ve Oncelik")
    ax2.grid(alpha=0.25)
    for _, r in ops.head(5).iterrows():
        ax2.text(
            float(r["ort_baslangic_gunu"]) + 0.6,
            float(r["ort_oncelik_puani"]) + 0.02,
            str(r["sorumlu_birim"]),
            fontsize=8,
        )
    cbar2 = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Hazirlik baski puani", fontsize=9)

    fig.suptitle("Operasyonel Hazirlik Panosu", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = out_dir / f"halk_dili_operasyon_hazirlik_panosu_{tag_date}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_csv, out_md, out_png


def build_disease_public_outputs(
    disease_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    out_dir: Path,
    tag_date: str,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    d = disease_df.copy()
    d["model_norm"] = d["model"].map(_normalize_model_label)
    d = d[d["model_norm"].isin({"strong", "quant"})].copy()
    if d.empty:
        raise SystemExit("Disease summary CSV does not include strong/quant rows.")

    score_wide = (
        d.pivot_table(index="disease_group_tr", columns="model_norm", values="direct_signal_score", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    level_wide = (
        d.pivot_table(index="disease_group_tr", columns="model_norm", values="direct_signal_level", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    ref_df = d.groupby("disease_group_tr", as_index=False)["primary_references"].first()

    m = matrix_df.copy()
    m["quantifiable_with_current_data"] = (
        m["quantifiable_with_current_data"].astype(str).str.lower().map({"true": 1.0, "false": 0.0}).fillna(0.0)
    )
    coverage = m.groupby("disease_group_tr", as_index=False)["quantifiable_with_current_data"].mean()
    coverage = coverage.rename(columns={"quantifiable_with_current_data": "dogrudan_olculebilir_oran"})

    merged = (
        score_wide.merge(level_wide, on="disease_group_tr", suffixes=("_score", "_level"))
        .merge(ref_df, on="disease_group_tr", how="left")
        .merge(coverage, on="disease_group_tr", how="left")
    )

    if "strong_score" not in merged.columns:
        merged["strong_score"] = np.nan
    if "quant_score" not in merged.columns:
        merged["quant_score"] = np.nan
    if "strong_level" not in merged.columns:
        merged["strong_level"] = "bilinmiyor"
    if "quant_level" not in merged.columns:
        merged["quant_level"] = "bilinmiyor"

    merged["strong_score"] = pd.to_numeric(merged["strong_score"], errors="coerce").fillna(0.0)
    merged["quant_score"] = pd.to_numeric(merged["quant_score"], errors="coerce").fillna(0.0)
    merged["dogrudan_olculebilir_oran"] = pd.to_numeric(
        merged.get("dogrudan_olculebilir_oran", 0.0), errors="coerce"
    ).fillna(0.0)

    merged["konsensus_skor"] = (merged["strong_score"] + merged["quant_score"]) / 2.0
    merged["model_farki_abs"] = (merged["strong_score"] - merged["quant_score"]).abs()
    merged["risk_duzeyi"] = merged["konsensus_skor"].map(_risk_band)
    merged["model_uyumu"] = merged["model_farki_abs"].map(_agreement_band)
    merged["sayisal_kanit_guveni"] = merged.apply(
        lambda r: _evidence_band(float(r["dogrudan_olculebilir_oran"]), str(r["model_uyumu"])),
        axis=1,
    )
    merged["literatur_celiski_durumu"] = "celiski_yok"
    merged.loc[
        (merged["model_uyumu"] == "dusuk") & (merged["strong_score"] > (merged["quant_score"] + 1.0)),
        "literatur_celiski_durumu",
    ] = "kismi_celiski_riski"

    def note(row: pd.Series) -> str:
        risk = str(row["risk_duzeyi"])
        evidence = str(row["sayisal_kanit_guveni"])
        if "cilt kanseri" in str(row["disease_group_tr"]).lower():
            return "UV verisi eksik oldugu icin sayisal artis yuzdesi degil, nitel risk baskisi verilir."
        if risk == "yuksek":
            return f"Her iki modelde de yuksek baski sinyali var; izlem onceligi yuksek (kanit {evidence})."
        if risk == "orta":
            return f"Orta duzey baski sinyali var; yerel veriyle duzenli izlem gerekir (kanit {evidence})."
        return f"Sinyal dusuk; yine de mevsimsel dalgalanmalarda takip edilmelidir (kanit {evidence})."

    merged["halk_dili_mesaj"] = merged.apply(note, axis=1)
    merged["risk_puani"] = merged["risk_duzeyi"].map(_risk_score)
    merged["kanit_puani"] = merged["sayisal_kanit_guveni"].map(_evidence_score)
    merged["uyum_puani"] = merged["model_uyumu"].map(_agreement_score)
    merged["oncelik_puani"] = (
        (merged["risk_puani"] * 0.60) + (merged["kanit_puani"] * 0.25) + (merged["uyum_puani"] * 0.15)
    ).round(3)

    def action_label(row: pd.Series) -> str:
        risk = str(row["risk_duzeyi"])
        evidence = str(row["sayisal_kanit_guveni"])
        if risk == "yuksek" and evidence in {"orta-yuksek", "orta"}:
            return "acil_eylem"
        if risk == "yuksek":
            return "yakindan_izlem"
        if risk == "orta" and evidence in {"orta-yuksek", "orta"}:
            return "hedefli_onlem"
        if risk == "orta":
            return "yakindan_izlem"
        return "rutin_izlem"

    merged["aksiyon_onceligi"] = merged.apply(action_label, axis=1)
    merged = merged.sort_values(["konsensus_skor", "strong_score"], ascending=False).reset_index(drop=True)

    out_csv = out_dir / f"halk_dili_hastalik_ozet_{tag_date}.csv"
    keep_cols = [
        "disease_group_tr",
        "strong_score",
        "quant_score",
        "konsensus_skor",
        "risk_duzeyi",
        "strong_level",
        "quant_level",
        "model_uyumu",
        "dogrudan_olculebilir_oran",
        "sayisal_kanit_guveni",
        "literatur_celiski_durumu",
        "halk_dili_mesaj",
        "primary_references",
    ]
    merged[keep_cols].to_csv(out_csv, index=False)

    priority_cols = [
        "disease_group_tr",
        "risk_duzeyi",
        "sayisal_kanit_guveni",
        "model_uyumu",
        "risk_puani",
        "kanit_puani",
        "uyum_puani",
        "oncelik_puani",
        "aksiyon_onceligi",
        "halk_dili_mesaj",
    ]
    out_priority_csv = out_dir / f"halk_dili_oncelik_matrisi_{tag_date}.csv"
    merged[priority_cols].sort_values(["oncelik_puani", "risk_puani"], ascending=False).to_csv(out_priority_csv, index=False)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(11, 7))
    color_map = {
        "yuksek": "#d73027",
        "orta": "#fc8d59",
        "dusuk": "#91bfdb",
        "cok_dusuk": "#4575b4",
    }
    colors = [color_map.get(x, "#666666") for x in merged["risk_duzeyi"]]
    sizes = 120 + (merged["dogrudan_olculebilir_oran"] * 500).clip(0, 500)
    ax.scatter(merged["quant_score"], merged["strong_score"], s=sizes, c=colors, alpha=0.85, edgecolor="black", linewidth=0.4)

    max_axis = max(float(merged["strong_score"].max()), float(merged["quant_score"].max()), 0.2) + 0.4
    ax.plot([0, max_axis], [0, max_axis], linestyle="--", color="#555555", linewidth=1)
    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ax.set_xlabel("Quant dogrudan risk skoru")
    ax.set_ylabel("Strong dogrudan risk skoru")
    ax.set_title("Hastalik Gruplari: Model Karsilastirmasi ve Veri Guveni")
    ax.grid(alpha=0.25)

    label_df = merged.head(6).copy()
    skin = merged[merged["disease_group_tr"].str.lower().str.contains("cilt kanseri", regex=False)]
    if not skin.empty:
        label_df = pd.concat([label_df, skin], ignore_index=True).drop_duplicates(subset=["disease_group_tr"])
    for _, r in label_df.iterrows():
        ax.text(float(r["quant_score"]) + 0.03, float(r["strong_score"]) + 0.03, str(r["disease_group_tr"]), fontsize=8)

    legend_labels = [
        ("yuksek", "Konsensus risk: yuksek"),
        ("orta", "Konsensus risk: orta"),
        ("dusuk", "Konsensus risk: dusuk"),
    ]
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[k], markeredgecolor="black", markersize=8, label=v)
        for k, v in legend_labels
    ]
    handles.append(plt.Line2D([0], [0], marker="o", color="#444444", markersize=8, linestyle="None", label="Nokta boyu = olculebilirlik"))
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)

    out_img = out_dir / f"halk_dili_hastalik_karsilastirma_{tag_date}.png"
    fig.tight_layout()
    fig.savefig(out_img, dpi=180)
    plt.close(fig)

    # Priority matrix visual
    priority_df = merged.sort_values("oncelik_puani", ascending=False).copy()
    fig2, ax2 = plt.subplots(figsize=(10.5, 6.5))
    action_color = {
        "acil_eylem": "#d73027",
        "hedefli_onlem": "#fc8d59",
        "yakindan_izlem": "#fee08b",
        "rutin_izlem": "#91bfdb",
    }
    colors2 = [action_color.get(x, "#666666") for x in priority_df["aksiyon_onceligi"]]
    sizes2 = (priority_df["oncelik_puani"] * 180).clip(80, 700)
    ax2.scatter(
        priority_df["kanit_puani"],
        priority_df["risk_puani"],
        s=sizes2,
        c=colors2,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.4,
    )
    ax2.set_xticks([1, 2, 3], ["dusuk", "orta", "orta-yuksek"])
    ax2.set_yticks([0, 1, 2, 3], ["cok_dusuk", "dusuk", "orta", "yuksek"])
    ax2.set_xlabel("Sayisal kanit guveni")
    ax2.set_ylabel("Risk duzeyi")
    ax2.set_title("Hastalik Aksiyon Oncelik Matrisi (Ilk 90 Gun)")
    ax2.grid(alpha=0.25)
    for _, r in priority_df.head(7).iterrows():
        ax2.text(
            float(r["kanit_puani"]) + 0.03,
            float(r["risk_puani"]) + 0.03,
            str(r["disease_group_tr"]),
            fontsize=8,
        )
    legend_items = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=v, markeredgecolor="black", markersize=8, label=k)
        for k, v in action_color.items()
    ]
    legend_items.append(
        plt.Line2D([0], [0], marker="o", color="#444444", markersize=8, linestyle="None", label="Nokta boyu = oncelik puani")
    )
    ax2.legend(handles=legend_items, loc="upper left", frameon=True, fontsize=8)
    out_priority_img = out_dir / f"halk_dili_oncelik_matrisi_{tag_date}.png"
    fig2.tight_layout()
    fig2.savefig(out_priority_img, dpi=180)
    plt.close(fig2)

    # Action plan table + 90-day timeline
    plan_df = merged.sort_values(["oncelik_puani", "risk_puani"], ascending=False).copy()
    starts: list[int] = []
    ends: list[int] = []
    baslama: list[str] = []
    izlem: list[str] = []
    tetik: list[str] = []
    for action in plan_df["aksiyon_onceligi"].tolist():
        s, e, b, i, t = _action_window(str(action))
        starts.append(s)
        ends.append(e)
        baslama.append(b)
        izlem.append(i)
        tetik.append(t)
    plan_df["baslangic_gunu"] = starts
    plan_df["bitis_gunu"] = ends
    plan_df["baslama_suresi"] = baslama
    plan_df["izlem_sikligi"] = izlem
    plan_df["tetikleyici_kural"] = tetik
    plan_df["sorumlu_birim"] = plan_df["disease_group_tr"].map(_owner_for_disease)
    plan_df["ilk_90_gun_hedefi"] = plan_df["aksiyon_onceligi"].map(_action_goal)

    out_plan_csv = out_dir / f"halk_dili_eylem_plani_{tag_date}.csv"
    plan_cols = [
        "disease_group_tr",
        "aksiyon_onceligi",
        "oncelik_puani",
        "baslangic_gunu",
        "bitis_gunu",
        "baslama_suresi",
        "izlem_sikligi",
        "sorumlu_birim",
        "ilk_90_gun_hedefi",
        "tetikleyici_kural",
    ]
    plan_df[plan_cols].to_csv(out_plan_csv, index=False)

    fig3, ax3 = plt.subplots(figsize=(11.5, 7.5))
    top_plan = plan_df.head(10).copy()
    top_plan = top_plan.iloc[::-1].reset_index(drop=True)
    color_action = {
        "acil_eylem": "#d73027",
        "hedefli_onlem": "#fc8d59",
        "yakindan_izlem": "#fee08b",
        "rutin_izlem": "#91bfdb",
    }
    for i, r in top_plan.iterrows():
        start = int(r["baslangic_gunu"])
        end = int(r["bitis_gunu"])
        ax3.barh(
            y=i,
            width=(end - start),
            left=start,
            color=color_action.get(str(r["aksiyon_onceligi"]), "#999999"),
            alpha=0.9,
            edgecolor="#333333",
        )
        ax3.text(
            start + 0.8,
            i,
            str(r["disease_group_tr"]),
            va="center",
            ha="left",
            fontsize=8,
            color="#1a1a1a",
        )
    ax3.set_xlim(0, 90)
    ax3.set_xlabel("Gun (Ilk 90 gun)")
    ax3.set_yticks([])
    ax3.set_title("Ilk 90 Gun Eylem Takvimi (Oncelik Sirali Ilk 10 Grup)")
    ax3.grid(axis="x", alpha=0.25)
    legend3 = [
        plt.Line2D([0], [0], color=v, lw=8, label=k)
        for k, v in color_action.items()
    ]
    ax3.legend(handles=legend3, loc="lower right", frameon=True, fontsize=8)
    out_plan_img = out_dir / f"halk_dili_eylem_takvimi_{tag_date}.png"
    fig3.tight_layout()
    fig3.savefig(out_plan_img, dpi=180)
    plt.close(fig3)

    # SSS markdown
    strong_hum = float(d[d["model_norm"] == "strong"]["delta_mean_humidity_pct"].mean())
    quant_hum = float(d[d["model_norm"] == "quant"]["delta_mean_humidity_pct"].mean())
    if strong_hum > 0 and quant_hum > 0:
        hum_text = (
            f"Hayir. Iki modelde de bagil nem artis yonunde (strong +{strong_hum:.2f}, quant +{quant_hum:.2f} puan). "
            "Bu ciktiya gore genel kurulasma sinyali yok."
        )
    elif strong_hum < 0 and quant_hum < 0:
        hum_text = (
            f"Evet, iki model de kurulasma yonu gosteriyor (strong {strong_hum:.2f}, quant {quant_hum:.2f} puan). "
            "Yerel su ve saglik izlemi guclendirilmeli."
        )
    else:
        hum_text = (
            f"Karisik sinyal var (strong {strong_hum:.2f}, quant {quant_hum:.2f} puan). "
            "Ilce bazli veri olmadan tek bir genel yargi verilmemeli."
        )

    skin_rows = merged[merged["disease_group_tr"].str.lower().str.contains("cilt kanseri", regex=False)]
    if skin_rows.empty:
        skin_text = "Cilt kanseri satiri bulunamadi; UV katmani eklenmeden sayisal artis yuzdesi raporlanmamali."
    else:
        skin_cov = float(skin_rows.iloc[0]["dogrudan_olculebilir_oran"])
        skin_text = (
            "Cilt kanseri icin ana surucu UV maruziyetidir. Bu veri boru hattinda UV yok; "
            f"dogrudan olculebilir oran %{skin_cov * 100:.1f}. "
            "Bu nedenle 'kesin su kadar artar' denmez, yalnizca nitel risk baskisi belirtilir."
        )

    conflict_counts = merged["literatur_celiski_durumu"].value_counts().to_dict()
    conflict_text = (
        f"Celiski_yok satir sayisi: {int(conflict_counts.get('celiski_yok', 0))}, "
        f"kismi_celiski_riski satir sayisi: {int(conflict_counts.get('kismi_celiski_riski', 0))}. "
        "Kismi celiski satirlari, model buyukluk farkinin yuksek oldugu gruplardir ve ek kalibrasyon ister."
    )

    faq_lines = [
        "# Halk Dili SSS: Hastalik Etkileri ve Bilimsel Uyum",
        "",
        "## En Onemli 5 Hastalik Grubu (konsensus skora gore)",
        "",
    ]
    for _, r in merged.head(5).iterrows():
        faq_lines.append(
            f"- {r['disease_group_tr']}: risk={r['risk_duzeyi']}, kanit={r['sayisal_kanit_guveni']}, not={r['halk_dili_mesaj']}"
        )
    faq_lines.extend(["", "## Ilk 90 Gun Aksiyon Onceligi", ""])
    top_actions = merged.sort_values("oncelik_puani", ascending=False).head(5)
    for _, r in top_actions.iterrows():
        faq_lines.append(
            f"- {r['disease_group_tr']}: {r['aksiyon_onceligi']} (oncelik puani {float(r['oncelik_puani']):.2f})"
        )
    faq_lines.append("- Detayli uygulama: halk_dili_eylem_plani_<DATE>.csv ve halk_dili_eylem_takvimi_<DATE>.png")
    faq_lines.append("- Haftalik uygulama: halk_dili_haftalik_is_plani_<DATE>.csv ve halk_dili_haftalik_is_yuku_<DATE>.png")
    faq_lines.append("- Toplanti ve takip: halk_dili_hafta1_toplanti_ajandasi_<DATE>.md + halk_dili_haftalik_kpi_takip_<DATE>.csv")
    faq_lines.append(
        "- KPI alarm paneli: halk_dili_kpi_alarm_kurallari_<DATE>.csv + "
        "halk_dili_kpi_alarm_ozeti_<DATE>.md + halk_dili_kpi_alarm_panosu_<DATE>.png"
    )
    faq_lines.append("- Toplu pano: halk_dili_entegre_bulgular_panosu_<DATE>.png + halk_dili_entegre_bulgular_ozeti_<DATE>.md")
    faq_lines.append(
        "- Mudahale senaryolari: halk_dili_mudahale_senaryolari_<DATE>.csv + "
        "halk_dili_mudahale_senaryo_ozeti_<DATE>.md + halk_dili_mudahale_senaryolari_<DATE>.png"
    )
    faq_lines.append(
        "- Operasyon hazirlik panosu: halk_dili_operasyon_hazirlik_matrisi_<DATE>.csv + "
        "halk_dili_operasyon_hazirlik_ozeti_<DATE>.md + halk_dili_operasyon_hazirlik_panosu_<DATE>.png"
    )
    faq_lines.extend(
        [
            "",
            "## Soru: Cilt kanseri artar mi?",
            f"- {skin_text}",
            "",
            "## Soru: Hava kurulasir mi?",
            f"- {hum_text}",
            "",
            "## Soru: Tahminler genel literaturle celisiyor mu?",
            f"- {conflict_text}",
            "",
            "## Sinirlar",
            "- Bu rapor klinik tani araci degildir; nufus duzeyi iklim-saglik taramasidir.",
            "- UV, PM2.5/O3, polen ve yerel sosyoekonomik maruziyet katmanlari eklendikce sayisal guven artar.",
            "",
        ]
    )

    out_md = out_dir / f"halk_dili_sik_sorular_{tag_date}.md"
    out_md.write_text("\n".join(faq_lines), encoding="utf-8")

    out_weekly_csv, out_weekly_md, out_weekly_img = build_weekly_action_outputs(plan_df, out_dir, tag_date)
    weekly_df = pd.read_csv(out_weekly_csv)
    out_week1_csv, out_agenda_md, out_kpi_csv, kpi_df = build_week1_execution_pack(plan_df, weekly_df, out_dir, tag_date)
    out_alarm_csv, out_alarm_md, out_alarm_png = build_kpi_alarm_outputs(kpi_df, out_dir, tag_date)
    out_scn_csv, out_scn_md, out_scn_png = build_intervention_scenario_outputs(merged, out_dir, tag_date)
    out_ops_csv, out_ops_md, out_ops_png = build_operational_readiness_outputs(plan_df, weekly_df, out_dir, tag_date)
    alarm_rules_df = pd.read_csv(out_alarm_csv)
    out_dashboard_png, out_dashboard_md = build_integrated_findings_dashboard(
        merged, weekly_df, alarm_rules_df, out_dir, tag_date
    )

    return (
        out_csv,
        out_img,
        out_md,
        out_priority_csv,
        out_priority_img,
        out_plan_csv,
        out_plan_img,
        out_weekly_csv,
        out_weekly_md,
        out_weekly_img,
        out_week1_csv,
        out_agenda_md,
        out_kpi_csv,
        out_alarm_csv,
        out_alarm_md,
        out_alarm_png,
        out_dashboard_png,
        out_dashboard_md,
        out_scn_csv,
        out_scn_md,
        out_scn_png,
        out_ops_csv,
        out_ops_md,
        out_ops_png,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cmp_df = safe_read_csv(resolve_comparison_csv(args.comparison_csv))
    q_df = safe_read_csv(args.quant_sensitivity_csv)
    s_df = safe_read_csv(args.strong_sensitivity_csv)
    lit_df = safe_read_csv(resolve_core_literature_csv(args.core_literature_csv, args.output_dir))
    disease_df = safe_read_csv(args.disease_summary_csv)
    matrix_df = safe_read_csv(args.disease_matrix_csv)

    halk_csv, halk_img, halk_df = build_halk_ozet(cmp_df, args.output_dir, args.tag_date)
    ck_csv, ck_df = build_literature_check(cmp_df, args.output_dir, args.tag_date)
    traffic_csv, model_csv, traffic_img, traffic_md = build_traffic_outputs(q_df, s_df, args.output_dir, args.tag_date)
    halk_md = build_public_markdown(halk_df, ck_df, len(lit_df), args.output_dir, args.tag_date)
    (
        disease_csv,
        disease_img,
        disease_md,
        priority_csv,
        priority_img,
        action_csv,
        action_png,
        weekly_csv,
        weekly_md,
        weekly_png,
        week1_csv,
        week1_md,
        weekly_kpi_csv,
        alarm_rules_csv,
        alarm_md,
        alarm_png,
        dashboard_png,
        dashboard_md,
        scenario_csv,
        scenario_md,
        scenario_png,
        ops_csv,
        ops_md,
        ops_png,
    ) = build_disease_public_outputs(
        disease_df, matrix_df, args.output_dir, args.tag_date
    )

    print(f"Wrote: {halk_csv}")
    print(f"Wrote: {halk_img}")
    print(f"Wrote: {ck_csv}")
    print(f"Wrote: {halk_md}")
    print(f"Wrote: {traffic_csv}")
    print(f"Wrote: {model_csv}")
    print(f"Wrote: {traffic_img}")
    print(f"Wrote: {traffic_md}")
    print(f"Wrote: {disease_csv}")
    print(f"Wrote: {disease_img}")
    print(f"Wrote: {disease_md}")
    print(f"Wrote: {priority_csv}")
    print(f"Wrote: {priority_img}")
    print(f"Wrote: {action_csv}")
    print(f"Wrote: {action_png}")
    print(f"Wrote: {weekly_csv}")
    print(f"Wrote: {weekly_md}")
    print(f"Wrote: {weekly_png}")
    print(f"Wrote: {week1_csv}")
    print(f"Wrote: {week1_md}")
    print(f"Wrote: {weekly_kpi_csv}")
    print(f"Wrote: {alarm_rules_csv}")
    print(f"Wrote: {alarm_md}")
    print(f"Wrote: {alarm_png}")
    print(f"Wrote: {dashboard_png}")
    print(f"Wrote: {dashboard_md}")
    print(f"Wrote: {scenario_csv}")
    print(f"Wrote: {scenario_md}")
    print(f"Wrote: {scenario_png}")
    print(f"Wrote: {ops_csv}")
    print(f"Wrote: {ops_md}")
    print(f"Wrote: {ops_png}")


if __name__ == "__main__":
    main()
