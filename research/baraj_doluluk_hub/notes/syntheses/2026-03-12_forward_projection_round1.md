# 2026-03-12 Forward Projection Round 1

Bu not, ilk `2026-2040` aylik Istanbul toplam baraj doluluk projeksiyon
paketinin kaydidir.

## Uretilen ana paket

- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/scenario_projection_monthly_2026_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/scenario_projection_summary_2026_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/scenario_parameters_2026_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/model_selection_metrics.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/projection_round1_summary.md`

## Model secimi

Ortak egitim penceresi:
- `2011-02` -> `2024-02`

Ilk resmi benchmarklar:
- `history_only_ridge`: `RMSE = 5.35` yp
- `hybrid_ridge`: `RMSE = 4.30` yp
- `extra_trees_full`: `RMSE = 4.40` yp

Secilen model:
- `hybrid_ridge`

Gerekce:
- ilk recursive ileri projeksiyon turunda en iyi yuruyen-test hatasini verdi
- `yalniz tarihsel` modele gore belirgin iyilesme gosterdigi icin
  hibrit kurgu savunulabilir hale geldi

## Senaryo aileleri

- `base`
- `wet_mild`
- `hot_dry_high_demand`
- `management_improvement`

## Parametre mantigi

Iklim:
- yagis sonu-2040 delta
- mevsimsel yagis yeniden dagitimi
- ET0 sonu-2040 delta
- sicaklik ve bagil nem suruklenmesi
- VPD artisi

Talep ve isletme:
- subscriber / nufus buyumesi
- kisi-basi kullanim egilimi
- NRW azalimi

Not:
- `reclaimed water` ve `transfer reliability` ilk turda kavramsal olarak
  saklandi, fakat aylik sayisal motor icinde hala `neutral` tutuldu

## Sonuc ozeti

Round-1 `2026-2040` ortalama toplam doluluk:

- `wet_mild`: `60.91%`
- `management_improvement`: `55.60%`
- `base`: `51.70%`
- `hot_dry_high_demand`: `36.12%`

`2040-12` sonu:

- `wet_mild`: `44.18%`
- `management_improvement`: `38.43%`
- `base`: `30.38%`
- `hot_dry_high_demand`: `6.11%`

Kontrol noktasi farklari:

- `wet_mild` senaryosu `2040-12` noktasinda baza gore `+13.80` yp
- `management_improvement` senaryosu `2040-12` noktasinda baza gore `+8.05` yp
- `hot_dry_high_demand` senaryosu `2040-12` noktasinda baza gore `-24.27` yp

Ilk esik gecisleri:

- `base` senaryosu ilk kez `2036-11` doneminde `%30` altina iniyor
- `hot_dry_high_demand` senaryosu ilk kez `2027-11` doneminde `%30` altina iniyor
- `wet_mild` senaryosu bu turda `%30` altina inmiyor
- `management_improvement` senaryosu bu turda `%30` altina inmiyor

## Belirsizlik bandi

Secilen model icin aylik empirik artiklara dayali alt-ust bant tablosu da
uretilmistir:

- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/selected_model_empirical_interval_by_month.csv`

Ana grafik artik senaryo cizgilerinin etrafinda bu empirik bantlari da
gostermektedir.

## Yeni risk ve toparlanma katmani

Bu turda esik alti risk ve toparlanma bilgisi de eklendi:

- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/scenario_threshold_risk_summary_2026_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/figures/threshold_risk_below_30.png`

Ana okumalar:

- `hot_dry_high_demand` senaryosunda `%40` alti noktasal doluluk `108` ay,
  `%30` alti noktasal doluluk `64` ay
- ayni senaryoda `%40` icin kalici dusuk-donem gecisi `2037-06`,
  `%30` icin `2040-06`
- `base` senaryosunda `%40` icin kalici dusuk-donem gecisi `2040-09`
- `management_improvement` senaryosunda `%30` alti noktasal doluluk yok
- `wet_mild` senaryosunda `%30` alti noktasal doluluk yok

## Yeni bilesen ayrisimi

Bu turda iki ana senaryonun hangi bloktan beslendigi ayristirildi:

- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/scenario_driver_decomposition_2026_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/figures/driver_decomposition_2040.png`

Ana okumalar:

- `management_improvement` senaryosunda `2040-12` noktasinda baza gore
  `+8.05` yp farkin:
  - yaklasik `+4.21` yp kadari kullanim verimliligi / kisi-basi azalis
  - yaklasik `+4.02` yp kadari `NRW` iyilesmesi
- `hot_dry_high_demand` senaryosunda `2040-12` noktasinda baza gore
  `-24.27` yp farkin:
  - yaklasik `-9.26` yp kadari iklim blogu
  - yaklasik `-9.14` yp kadari talep blogu

Not:
- bu ayrisim tam nedensel parcalama degil,
  ayri model kosularindan gelen karar-destek tipi katkı okumasidir

## Yeni surucu-once okuma

Istek dogrultusunda ileri doluluktan once tum gelecek suruculerin yolu da
cizildi.

Yeni dosyalar:

- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/figures/future_driver_paths_2026_2040.png`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/figures/future_driver_deltas_2040.png`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/scenario_driver_annual_summary_2026_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/scenario_driver_checkpoints_2026_2040.csv`

Bu katman sayesinde once su sorular gorunur hale geldi:

- yagis hangi senaryoda nasil gidiyor
- ET0 ne kadar artiyor
- talep ne kadar yukseliyor veya dusuyor
- sicaklik, bagil nem ve VPD hangi yone kayiyor
- yagis eksi ET0 su dengesi nasil bozuluyor veya iyilesiyor

2040 kontrol noktasi ornegi:

- `base`: yagis `122.1 mm/ay`, ET0 `24.5 mm/ay`, tuketim `3.06 milyon m3/ay`, doluluk `30.38%`
- `wet_mild`: yagis `132.6 mm/ay`, ET0 `24.1 mm/ay`, tuketim `2.90 milyon m3/ay`, doluluk `44.18%`
- `management_improvement`: yagis `122.1 mm/ay`, ET0 `24.5 mm/ay`, tuketim `2.77 milyon m3/ay`, doluluk `38.43%`
- `hot_dry_high_demand`: yagis `108.4 mm/ay`, ET0 `25.9 mm/ay`, tuketim `3.38 milyon m3/ay`, doluluk `6.11%`

## Kaynak capalari

- `SRC-066`
- `SRC-067`
- `SRC-068`
- `SRC-069`
- `SRC-070`
- `SRC-071`

## Acik teknik sinir

Aktinograf gelmedigi icin ET0 tarafinda gercek gozlenmis radyasyonlu yeniden
hesaplama henuz yok. Bu nedenle bu tur, `proxy-backed ET0 baseline` ile
uretilmis ilk ileri projeksiyon turudur.

## Yeni dis transfer duyarliligi katmani

Bu turda resmi `Melen + Yesilcay` serisi de sayisal motorun ustune
duyarlilik katmani olarak eklendi.

Yeni dosyalar:

- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/official_transfer_dependency_annual_2021_2025.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/transfer_sensitivity_projection_2026_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/transfer_sensitivity_summary_2026_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/figures/transfer_dependency_history_2021_2025.png`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/figures/transfer_sensitivity_paths_2026_2040.png`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/figures/transfer_sensitivity_endpoints_2040.png`

Resmi `2021-2025` ortalama transfer payi:

- `Melen + Yesilcay / sehre verilen su = %47.16`

Modelleme notu:

- bu katman, ayri bir fiziksel giris-akimi modeli degil
- resmi transfer payini kullanarak `talep-esdegeri` bir yuk / rahatlama
  katsayisi olarak calisiyor

Ana okumalar:

- `base_transfer_relief` senaryosu `2040-12` noktasinda temele gore
  `+4.03` yp
- `base_transfer_stress` senaryosu `2040-12` noktasinda temele gore
  `-8.06` yp
- `hot_dry_transfer_stress` senaryosu `2040-12` noktasinda
  `hot_dry_high_demand` yoluna gore `-2.21` yp

Bu, dis kaynagin yalniz anlatisal degil, sayisal olarak da ciddi bir kaldirac
oldugunu gosteriyor.

## Yeni parametre duyarlilik izgarlari

Bu turda tekil senaryolarin yanina iki adet `2040 sonu doluluk` izgara testi
eklendi.

Yeni dosyalar:

- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/sensitivity_rain_demand_grid_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/sensitivity_et0_transfer_grid_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/figures/sensitivity_heatmap_rain_demand_2040.png`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040/figures/sensitivity_heatmap_et0_transfer_2040.png`

Ana okuma:

- `yagis + talep` izgarasi beklenen yone gidiyor
- `2040` sonunda yagis arttikca doluluk artiyor
- `2040` sonunda talep yuk bindikce doluluk dusuyor

Ornek:

- `yagis = -10%`, `talep = +10%` iken `2040-12` doluluk `13.35%`
- `yagis = +10%`, `talep = -10%` iken `2040-12` doluluk `48.31%`

Onemli tespit:

- `ET0 + transfer` izgarasi dis transfer tarafinda anlamli,
  fakat `ET0` tek basina izole edilince mevcut hibrit model fiziksel olarak
  beklenen negatif isareti tutarli vermiyor
- bu nedenle `ET0` tekil duyarlilik matrisi bu turda
  `karar metni`nden cok `model tanisi` olarak okunmali
- aktinograf geldiginde ve ET0/radyasyon blogu yeniden kuruldugunda bu test
  tekrar calistirilacak
