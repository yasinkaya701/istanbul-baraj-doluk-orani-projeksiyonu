# 2026-03-12 Probabilistic Projection Round 1

Bu not, Istanbul toplam baraj doluluk projeksiyonlarina eklenen yeni
olasiliksal katmani kaydeder.

## Paket

- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040_probabilistic/probabilistic_monthly_quantiles_2026_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040_probabilistic/probabilistic_yearly_risk_2026_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040_probabilistic/probabilistic_crossing_summary_2026_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040_probabilistic/probabilistic_endpoint_summary_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040_probabilistic/probabilistic_threshold_summary_2026_2040.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040_probabilistic/probabilistic_summary.md`

## Yontem

Bu turda projeksiyonlar tek cizgi olarak birakilmadi.

Kullanilan kurgu:

- benchmark tablosundan gelen agirliklarla iki modelli bir karisim
- `hybrid_ridge` + `extra_trees_full`
- model agirliklari: yaklasik `0.681` ve `0.319`
- her yol icin tek bir model secimi
- secilen modelin deterministik senaryo yoluna tarihsel hata bloklari eklenmesi
- hata blok boyu: `12` ay
- toplam simulasyon: `4000`

Bu, tam fiziksel bir akim simulasyonu degil.
Amaci, mevcut deterministik yolun etrafinda mevsimselligi ve model yapisi
belirsizligini birlikte gosteren daha savunulabilir bir olasilik zarfi
olusturmaktir.

## Ana sonuclar

2040 Aralik medyanlari:

- `wet_mild`: `45.99%`
- `management_improvement`: `40.23%`
- `base`: `31.94%`
- `hot_dry_high_demand`: `7.81%`

2040 Aralik `P10-P90` araliklari:

- `wet_mild`: `41.31% - 61.70%`
- `management_improvement`: `35.55% - 50.58%`
- `base`: `27.49% - 50.36%`
- `hot_dry_high_demand`: `3.24% - 18.24%`

Yillik risk yorumu:

- `base` senaryosunda yil icinde en az bir ay `%30` alti olma olasiligi
  `%50` esigini ilk kez `2036` yilinda geciyor.
- `wet_mild` senaryosunda bu turda `%30` icin `%50` ustu yillik risk yok.
- `management_improvement` senaryosunda da bu turda `%30` icin `%50` ustu
  yillik risk yok.
- `hot_dry_high_demand` senaryosunda yil icinde en az bir ay `%30` alti olma
  olasiligi `%50` esigini `2027` yilinda geciyor.

## Teknik anlam

Bu turla birlikte proje artik:

- tek bir tahmin cizgisi degil
- medyan yol + olasilik araliklari + yillik esik riskleri
  uretebiliyor
- model ayrismasini gizlemek yerine karar destegi girdisi olarak sakliyor

## Sinir

- olasiliksal katman, hata bloklarini deterministik yola ekliyor;
  hata geri beslemesini tam dinamik bir fiziksel simulasyon gibi tasimiyor
- bu nedenle bu paket `karar destegi icin guclu`, ama `tam fiziksel belirsizlik
  propagasyonu` olarak sunulmamalidir
- aktinograf geldikten sonra ET0 blogu guclendiginde bu olasilik katmani yeniden
  kurulmalidir
