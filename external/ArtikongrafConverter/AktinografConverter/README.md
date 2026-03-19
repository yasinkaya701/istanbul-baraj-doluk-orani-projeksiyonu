# Aktinograf İşleyici

Bu klasor, sadece desktop uygulama icin ayrilmis minimal pakettir.

## Icerik

- `python/desktop_app.py`: Ana masaustu uygulama
- `python/web_backend.py`: Import/process/save backend
- `python/extract_daily_series.py`: OpenCV tabanli veri cikarma
- `python/chart_models.py`: Grafik modeli
- `python/convert_tiff_dataset.py`: Opsiyonel TIFF donusturucu
- `python/requirements.txt`: Gerekli Python paketleri

Veri klasorleri:

- `data/uploaded/`
- `output/`
- `diff/`

## Kurulum

```bash
cd AktinografConverter
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

## Calistirma

```bash
python3 python/desktop_app.py
```

veya:

```bash
./run_desktop.sh
```
