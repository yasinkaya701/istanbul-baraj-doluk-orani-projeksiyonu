# İstanbul Baraj Projeksiyonu — Web Site

Bu repo, İstanbul baraj doluluk projeksiyonu çalışmasının **statik web sunumunu** içerir.

## İçerik
- `index.html`, `styles.css`: Tek sayfalık sunum
- `assets/docs/`: PDF raporlar
- `assets/et0/`: ET0 açıklama görselleri ve raporları
- `assets/img/`: Model kartları ve projeksiyon görselleri

## Yerelde açma
Tarayıcıda doğrudan:
- `index.html` dosyasını çift tıklayabilirsin

Veya basit bir statik sunucu:
```bash
python -m http.server 8000
```
Ardından `http://localhost:8000`.

## Raporları/ET0 paketini yeniden üretme
Bu repo **çıktı (artifact) deposudur**. Yeniden üretim için ana proje dizininde aşağıdaki adımlar kullanılır:

### PDF raporlar
```bash
python scripts/build_istanbul_current_status_pdf.py
python scripts/build_istanbul_current_status_detailed_pdf.py
# Hackathon final raporu: bağımlı pipeline verileri gerekir
python scripts/build_hackathon_final_pdf_report.py
```

### ET0 paket (gerçek radyasyon)
Gerekli girdiler üretildikten sonra:
```bash
python scripts/build_es_ea_newdata_csv.py \
  --temp-xlsx "<TEMP_XLSX>" \
  --humidity-xlsx "<HUMIDITY_XLSX>" \
  --auto-table1 "<AUTO_TABLE1>" \
  --auto-table2 "<AUTO_TABLE2>"

python scripts/build_complete_solar_dataset.py
python scripts/build_tarim_et0_real_radiation_package.py
python scripts/build_et0_formula_card.py
python scripts/build_one_year_explained_et0_charts.py --year 2004
python scripts/build_et0_trend_robust_chart.py
python scripts/build_et0_multiscale_charts.py
```

### Güncel dosyaları bu repo’ya kopyalama
Üretilen PDF ve görselleri `assets/docs/` ve `assets/et0/` altına kopyalayın.

## Veri Kaynakları (özet)
- İBB/İSKİ baraj doluluk ve havza yağışı
- İBB tüketim verileri
- Kandilli uzun dönem iklim serileri
- İklim projeksiyonları (2026–2040)

## Notlar
- Bu repo “sunum” amaçlıdır; model kodu ve veri işleme hatları ana projededir.
- Veri dosyaları bu repoda tutulmaz; site, veri yoluna bağlı içerik içermez.
- Hackathon final raporu, **extreme event pipeline** verilerine bağlıdır. Bu veri üretilmezse rapor yeniden oluşturulamaz.

## Lisans
İçerikler proje içi kullanım içindir.
