#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BARAJ_PAGE_URL = "https://iski.istanbul/baraj-doluluk"
DEFAULT_OUT_DIR = Path("/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot")
BASE_URL_FALLBACK = "https://iskiapi.iski.istanbul/api/"
TIMEOUT = 30
MONTH_MAP_TR = {
    "Ocak": 1,
    "Şubat": 2,
    "Mart": 3,
    "Nisan": 4,
    "Mayıs": 5,
    "Haziran": 6,
    "Temmuz": 7,
    "Ağustos": 8,
    "Eylül": 9,
    "Ekim": 10,
    "Kasım": 11,
    "Aralık": 12,
}


@dataclass(frozen=True)
class EndpointSpec:
    slug: str
    path: str
    method: str = "GET"
    per_dam: bool = False


GLOBAL_ENDPOINTS = [
    EndpointSpec("genel_oran", "iski/baraj/genelOran/v2"),
    EndpointSpec("son_14_gun_toplam_doluluk", "iski/baraj/sonOnDortGunDoluluk/v2"),
    EndpointSpec("son_1_yil_ay_sonu_doluluk", "iski/baraj/sonBirYildakiAySonlariDoluluk/v2"),
    EndpointSpec("ayni_gun_yillara_gore_doluluk", "iski/baraj/sonGunDolulukOraniYillaraGore/v2"),
    EndpointSpec("barajlara_gore_mevcut_su_dagilimi", "iski/baraj/mevcutSuMiktarlarininBarajlaraGoreDagilimi/v2"),
    EndpointSpec("ayni_gun_yillara_gore_mevcut_su", "iski/baraj/sonGunMevcutSuYillaraGore/v2"),
    EndpointSpec("gunluk_ozet", "iski/baraj/gunlukOzet/v2"),
    EndpointSpec("yillik_yagis", "iski/baraj/yillikYagisMiktari/v2"),
    EndpointSpec("son_14_gun_verilen_su", "iski/baraj/icmeSuyuAritma/sonOndortGundeVerilenSu/v2"),
    EndpointSpec("son_1_yil_aylik_ortalama_verilen_su", "iski/baraj/icmeSuyuAritma/sonBirYildaVerilenAylikOrtalamaSu/v2"),
    EndpointSpec("son_10_yil_gunluk_ortalama_verilen_su", "iski/baraj/icmeSuyuAritma/sonOnYildaVerilenYillaraGoreGunlukOrtalamaSu/v2"),
    EndpointSpec("son_10_yil_toplam_verilen_su", "iski/baraj/icmeSuyuAritma/sonOnyildaVerilenToplamSu/v2"),
    EndpointSpec("melen_yesilcay", "iski/baraj/melenveYesilcayRegulatorlerindenAlinanSu/v2"),
    EndpointSpec("baraj_listesi", "iski/baraj/listesi/v2"),
]

PER_DAM_ENDPOINTS = [
    EndpointSpec("baraj_bazli_son_14_gun_doluluk", "iski/baraj/sonOnDortGundekiSu/v2?barajAdi={baraj}", method="POST", per_dam=True),
    EndpointSpec("baraj_bazli_son_14_gun_mevcut_su", "iski/baraj/sonOndortGundekiMevcutSuHacmi/v2?barajAdi={baraj}", method="POST", per_dam=True),
    EndpointSpec("baraj_bazli_son_1_yil_ay_sonu_doluluk", "iski/baraj/sonBirYildakiAySonuDolulukOranlari/v2?barajAdi={baraj}", method="POST", per_dam=True),
    EndpointSpec("baraj_bazli_son_10_yil_doluluk", "iski/baraj/sonOnYildakiDolulukOrani/v2?barajAdi={baraj}", method="POST", per_dam=True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a structured snapshot from İSKİ's public baraj-doluluk frontend API.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def fetch_text(url: str) -> str:
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.text


def discover_assets() -> tuple[str, str, list[str]]:
    html = fetch_text(BARAJ_PAGE_URL)
    script_paths = re.findall(r'<script[^>]+src="([^"]*_nuxt/[^"]+\.js)"', html)
    base_url = BASE_URL_FALLBACK
    token: str | None = None
    asset_urls: list[str] = []
    for script_path in script_paths:
        if script_path.startswith("http"):
            asset_url = script_path
        else:
            asset_url = f"https://iski.istanbul{script_path}"
        asset_urls.append(asset_url)
        js = fetch_text(asset_url)
        base_match = re.search(r'https://iskiapi\.iski\.istanbul/api/', js)
        if base_match:
            base_url = base_match.group(0)
        token_match = re.search(r'([0-9a-f]{200,})', js)
        if token_match:
            token = token_match.group(1)
    if token is None:
        raise RuntimeError("İSKİ frontend bundle içinde bearer token bulunamadı.")
    return base_url, token, asset_urls


def parse_date(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%d.%m.%Y", "%d/%m/%Y %H:%M:%S", "%Y"):
        try:
            dt = datetime.strptime(text, fmt)
            if fmt == "%Y":
                return dt.strftime("%Y-01-01")
            if fmt == "%d/%m/%Y %H:%M:%S":
                return dt.isoformat(sep=" ")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    month_match = re.fullmatch(r"([A-Za-zÇĞİÖŞÜçğıöşü]+)\s+(\d{4})", text)
    if month_match:
        month_name, year = month_match.groups()
        month = MONTH_MAP_TR.get(month_name)
        if month is not None:
            return f"{int(year):04d}-{month:02d}-01"
    return text


def to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if out[column].dtype != object:
            continue
        cleaned = out[column].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
        converted = pd.to_numeric(cleaned, errors="coerce")
        if converted.notna().sum() >= max(1, len(out) // 2):
            out[column] = converted
    return out


def normalize_table(payload: dict[str, Any]) -> pd.DataFrame:
    data = payload.get("data")
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame()
    if df.empty:
        return df
    for date_col in [c for c in df.columns if "tarih" in c.lower() or "zaman" in c.lower()]:
        df[date_col] = df[date_col].map(parse_date)
    df = to_numeric_frame(df)
    if "sonGuncellemeZamani" in payload:
        df["snapshot_updated_at"] = parse_date(payload["sonGuncellemeZamani"])
    return df


def fetch_json(base_url: str, token: str, spec: EndpointSpec, baraj: str | None = None) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    path = spec.path.format(baraj=baraj) if baraj else spec.path
    url = f"{base_url}{path}"
    if spec.method == "POST":
        response = requests.post(url, headers=headers, timeout=TIMEOUT)
    else:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    payload["_meta"] = {"url": url, "method": spec.method, "baraj": baraj}
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    readme = f"""# İSKİ Baraj API Snapshot

- Snapshot time: `{summary["snapshot_time"]}`
- Source page: `{BARAJ_PAGE_URL}`
- API base: `{summary["api_base_url"]}`
- Global endpoints fetched: `{summary["global_endpoint_count"]}`
- Per-dam endpoints fetched: `{summary["per_dam_endpoint_count"]}`
- Dam count: `{summary["dam_count"]}`

Useful tables:
- `tables/genel_oran.csv`
- `tables/gunluk_ozet.csv`
- `tables/son_14_gun_toplam_doluluk.csv`
- `tables/son_1_yil_ay_sonu_doluluk.csv`
- `tables/yillik_yagis.csv`
- `tables/son_14_gun_verilen_su.csv`
- `tables/melen_yesilcay.csv`
- `tables/baraj_bazli_son_10_yil_doluluk.csv`

Raw responses are stored under `raw/`.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir = args.out_dir / "raw"
    table_dir = args.out_dir / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    base_url, token, asset_urls = discover_assets()
    snapshot_time = datetime.now().isoformat(timespec="seconds")

    summary: dict[str, Any] = {
        "snapshot_time": snapshot_time,
        "api_base_url": base_url,
        "source_page": BARAJ_PAGE_URL,
        "frontend_assets": asset_urls,
    }

    tables_written: list[str] = []
    global_payloads: dict[str, dict[str, Any]] = {}
    for spec in GLOBAL_ENDPOINTS:
        payload = fetch_json(base_url, token, spec)
        global_payloads[spec.slug] = payload
        write_json(raw_dir / f"{spec.slug}.json", payload)
        df = normalize_table(payload)
        csv_path = table_dir / f"{spec.slug}.csv"
        df.to_csv(csv_path, index=False)
        tables_written.append(str(csv_path))

    dam_df = normalize_table(global_payloads["baraj_listesi"])
    dam_names = dam_df["kaynakAdi"].dropna().astype(str).tolist()

    per_dam_counts: dict[str, int] = {}
    for spec in PER_DAM_ENDPOINTS:
        frames: list[pd.DataFrame] = []
        count = 0
        for dam_name in dam_names:
            payload = fetch_json(base_url, token, spec, baraj=dam_name)
            write_json(raw_dir / f"{spec.slug}__{dam_name}.json", payload)
            df = normalize_table(payload)
            if df.empty:
                continue
            df.insert(0, "baraj_kod", dam_name)
            matching = dam_df.loc[dam_df["kaynakAdi"].astype(str) == dam_name, "baslikAdi"]
            if not matching.empty:
                df.insert(1, "baraj_adi", matching.iloc[0])
            frames.append(df)
            count += 1
        if frames:
            combined = pd.concat(frames, ignore_index=True)
        else:
            combined = pd.DataFrame()
        csv_path = table_dir / f"{spec.slug}.csv"
        combined.to_csv(csv_path, index=False)
        tables_written.append(str(csv_path))
        per_dam_counts[spec.slug] = count

    summary.update(
        {
            "global_endpoint_count": len(GLOBAL_ENDPOINTS),
            "per_dam_endpoint_count": len(PER_DAM_ENDPOINTS),
            "dam_count": len(dam_names),
            "dam_names": dam_names,
            "tables_written": tables_written,
            "per_dam_fetch_counts": per_dam_counts,
        }
    )

    (args.out_dir / "api_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    build_readme(args.out_dir, summary)

    print(args.out_dir / "api_manifest.json")
    print(args.out_dir / "README.md")
    for table in tables_written:
        print(table)


if __name__ == "__main__":
    main()
