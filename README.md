<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:00C6FF,50:3F51B5,100:F7971E&height=180&section=header&text=Hackhaton&fontSize=54&fontColor=FFFFFF&animation=twinkling" />
  <h3>Su, iklim ve baraj analitigi calisma alani</h3>
  <p>Arastirma, modelleme ve karar destek panelleri bir arada.</p>
  <p><strong>Not:</strong> Bu calisma Kandilli Rasathanesi ve Bogazici Universitesi Bilgisayar Hackhatonu icin yapilmistir.</p>
  <p>
    <img src="https://img.shields.io/badge/Odak-Climate%20%26%20Water-00BCD4" />
    <img src="https://img.shields.io/badge/Stack-Python%20%2B%20JS-1E88E5" />
    <img src="https://img.shields.io/badge/Ciktilar-Reports%20%26%20Charts-F4511E" />
    <img src="https://img.shields.io/badge/Data-Ignored%20in%20Git-8E24AA" />
  </p>
</div>

---

## Hizli harita
```text
Hackhaton/
|-- dashboard/                      # Paneller icin frontend + server
|-- scripts/                        # Modelleme, veri hazirlama ve analiz akislari
|-- research/                       # Arastirma hub, loglar ve template'ler
|-- baraj_web/                      # Web uygulamasi (ayri git repo)
|-- external/ArtikongrafConverter/  # Harici arac (ayri git repo)
|-- hackhaton_model_kartlari_2026_03_18/  # Model kartlari (gorsel)
|-- hackhaton_projection_2040_2026_03_18/ # Projeksiyon gorselleri
|-- ADVERSARIAL_TEST/               # Test goruntuleri (git disi)
|-- DESKTOP_PROCESSING/             # Yerel isleme ciktilari (git disi)
|-- DATA/                           # Yerel veri setleri (git disi)
|-- new data/                       # Yerel veri setleri (git disi)
|-- output/                         # Uretilen ciktilar (git disi)
|-- tmp/                            # Gecici dosyalar (git disi)
```

## Icerik ozeti
- Baraj doluluk, ET0 ve iklim sinyalleri icin uctan uca modelleme scriptleri
- Karar destek dashboard'lari ve anomaly news gorunumleri
- Log, template ve sentez notlari ile arastirma hub yapisi
- Model kartlari ve projeksiyon gorselleri

## Kullanim dokumanlari
- `MODEL_DECISION_DASHBOARD_KULLANIM.md`
- `MODEL_SUITE_KULLANIM.md`
- `ET_ML_KULLANIM.md`
- `PROPHET_KULLANIM.md`
- `STRONG_MODEL_KULLANIM.md`
- `SOLAR_MODEL_LITERATURE.md`

## Veri notu
Veri setleri bilerek git disinda tutulur. Yerel verileri su dizinlere koy:
- `DATA/`
- `new data/`

Yaygin veri formatlari (ornegin `*.csv`, `*.parquet`, `*.xlsx`, `*.pkl`) `.gitignore` ile disarida tutulur.

## Sub-repo notu
- `baraj_web/` ve `external/ArtikongrafConverter/` icerikleri bu repoda tutulur.
  Eski git metadata yedekleri yerel olarak `.git_backup/` altinda saklanir ve git disindadir.

## Konvansiyonlar
- Buyuk binary ve uretilen ciktilari git disinda tut
- Yeni pipeline'lar icin `scripts/` altina kisa not ekle

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=0:3F51B5,100:00C6FF&height=16&section=footer" />
</div>
