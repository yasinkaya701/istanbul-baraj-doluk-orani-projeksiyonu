# Gelecek Projeksiyonu Araştırma Notu

Tarih: 2026-03-12

## Soru

Istanbul toplam baraj doluluk orani icin 15 yillik ileri projeksiyon nasil
kurulmali?

Bu notun amaci:
- hangi yontemlerin savunulabilir oldugunu netlestirmek
- Istanbul/Turkiye odakli iklim ve su yonetimi kaynaklarini projeye baglamak
- mevcut modele dogrudan uygulanabilir bir proje planina donusturmek

## Ana Sonuc

15 yillik ileri projeksiyon tek bir cizgi olarak kurulmamali.

Bilimsel olarak daha savunulabilir yapi:
- senaryo tabanli
- coklu iklim zorlugu iceren
- talep ve isletme parametrelerini ayri oynatan
- sonuc olarak tek tahmin yerine bir olasilik zarfi veya senaryo bandi veren
bir yapi olmalidir.

Bu proje icin en uygun omurga:

1. iklim zorlayicilari:
   yagis, sicaklik, gerekirse asagidan turetilmis ET0
2. hidrolojik cevap:
   inflow vekili / yagis-akis veya havza tepkisi
3. insan kullanimi:
   talep, kayip, geri kazanim, kisit, transfer
4. depolama guncellemesi:
   aylik su dengesi uzerinden toplam doluluk yolu

## Kaynaklardan Cikan Yontemler

### 1. Resmi Istanbul iklim ve nufus baglami

Kaynak:
- Istanbul Climate Change Action Plan (IBB, 2022)

Bulgular:
- IBB dokumani Istanbul nufusunun 2015'e gore 2030'da yaklasik `%22`,
  2040'ta yaklasik `%35` artarak sirasiyla `17.9 milyon` ve `19.7 milyon`
  seviyelerine cikacagini veriyor.
- Belgede Istanbul icin oncelikli riskler arasinda
  `heat waves`, `drought`, `changing precipitation regimes`, `water shortage`
  acikca sayiliyor.

Projeye anlami:
- 15 yillik talep projeksiyonunda nufus/subscriber artisi senaryosu resmi bir
  sehir baglamina baglanabilir.
- Doluluk projeksiyonunda sadece iklim degil, buyuyen sehir baskisi da
  cekirdek bir blok olmalidir.

### 2. Resmi Turkiye iklim projeksiyonlari

Kaynak:
- MGM, Yeni Senaryolar ile Turkiye Iklim Projeksiyonlari Raporu (2015)

Yontem:
- CMIP5 tabanli `HadGEM2-ES`, `MPI-ESM-MR`, `GFDL-ESM2M`
- `RCP4.5` ve `RCP8.5`
- `RegCM4.3.4` ile `20 km` bolgesel iklim projeksiyonu
- donemler:
  `2016-2040`, `2041-2070`, `2071-2099`

Projeye anlami:
- Resmi kaynak gelecegi tek modelle degil, coklu model + coklu senaryo ile
  kuruyor. Bizim de ayni mantigi izlememiz daha savunulabilir.
- Havza ozetlerinde tum havzalarda sicakligin artacagi, yagisin ise ilk
  donemde Marmara/Black Sea havzalarinda artis gosterebilse bile daha ileri
  donemlerde tutarsizlastigi veya azalis gosterdigi belirtiliyor.
- Bu, 2026-2040 icin "kesin kuruma" degil,
  "artan sicaklik + mevsimsellik kaymasi + yagis belirsizligi" okumasini
  daha dogru kiliyor.

### 3. Istanbul icin dogrudan su arz-talep gelecek modeli

Kaynak:
- Daloglu (2019), `Modeling water supply and demand balance of Istanbul under future climate and socio-economic change scenarios`

Yontem:
- `WEAP` kullaniliyor
- `Rainfall Runoff (Simplified Coefficient)` ile catchment hidrolojisi kuruyor
- senaryolar:
  `BAU`, `High Technology-Low Population`, `Low Technology-High Population`
- iklim:
  `RCP4.5` ve `RCP8.5`
- talep:
  nufus buyumesi, su kullanim orani, kayiplar, altyapi ve dis kaynak
  varsayimlari
- dis kaynak bagimliligi ozellikle `Melen` uzerinden ayrica test ediliyor

Ana sonuc:
- olumsuz iklim etkileri `2030` sonrasinda daha belirgin,
  `2040` sonrasinda daha daginik ve belirsiz hale geliyor
- dis kaynaklara, ozellikle `Melen`'e bagimlilik su guvenligi riskini buyutuyor

Projeye anlami:
- Gelecek projeksiyonunda sadece meteoroloji degil,
  `external source / transfer` blogu da olmalidir.
- Aylik toplam doluluk modelimizi orta vadede bir `hafif WEAP mantigi` ile
  genisletmek mumkun.

### 4. ET0 ve kuraklik talebi tarafinda yakin bolge bulgusu

Kaynak:
- `Modeling the Effect of Climate Change on Evapotranspiration in the Thrace Region`
  (Atmosphere, 2024)

Bulgular:
- bolgesel sicaklik artisi ve yagis degisimi ile `ET0` birlikte inceleniyor
- calisma, gelecekte `ET0` artisinin anlamli olabilecegini gosteriyor
- makale metnindeki ozet sayilar:
  `2016-2025` icin `ET0` yaklasik `%7` artiyor,
  `2046-2055` icin `%20`,
  `2076-2085` icin `%33`

Projeye anlami:
- ET0'u sabit alarak 15 yillik projeksiyon kurmak zayif kalir.
- En azindan orta ve stres senaryolarinda `ET0` artis blogu kullanilmalidir.
- Bu calisma dogrudan Istanbul degil, Thrace bolgesi oldugu icin
  sayilar birebir alinmamali; ancak yaklasik duyarlilik ve senaryo buyuklugu
  seciminde referans olabilir.

### 5. CMIP6 ile Turkiye geneli ekstremler

Kaynak:
- `Impacts of Climate Change on Extreme Climate Indices in Türkiye Driven by High-Resolution Downscaled CMIP6 Climate Models`
  (Sustainability, 2023)

Yontem:
- tum uygun `CMIP6` modelleri
- senaryolar:
  `SSP2-4.5`, `SSP5-8.5`
- `ERA5-Land` ile bias correction / downscaling
- yuksek cozumlu sicaklik ve yagis ekstremleri uretiliyor

Projeye anlami:
- Projeksiyon blogunu `RCP` ile sinirli birakmak zorunda degiliz;
  `CMIP6 SSP2-4.5` ve `SSP5-8.5` mantigina gecmek teknik olarak daha guncel.
- Yagisin sadece ortalamasina bakmak yeterli degil;
  asiri sicak ve asiri yagis indeksleri de sistem davranisini etkiliyor.

### 6. CMIP6 asiri yagis downscaling yontemi

Kaynak:
- `Quantifying future rainfall extremes in Türkiye: a CMIP6 ensemble approach with statistical downscaling`
  (Acta Geophysica, 2025)

Yontem:
- `6 high-performer member ensemble`
- `SSP2-4.5` ve `SSP5-8.5`
- `QDM` yani `Quantile Delta Mapping`
- donemler:
  `2015-2040`, `2041-2070`, `2071-2100`

Projeye anlami:
- En iyi projeksiyon yolu tek GCM degil,
  performansi iyi secilmis bir `ensemble` yapi.
- `QDM` veya benzeri bias-correction / delta mantigi bizim icin dogrudan
  uygulanabilir.
- Asiri yagis siddetinin artmasi, toplam yagis artsa bile depolama tepkisinin
  dogrusal olmayabilecegini gosteriyor.
  Yani "daha cok yagis = ayni oranda daha cok depolama" varsayimi zayif.

### 7. Istanbul havzalari icin hidrolojik cekirdek

Kaynak:
- `Assessing the Water-Resources Potential of Istanbul by Using a Soil and Water Assessment Tool (SWAT) Hydrological Model`
  (Water, 2017)

Yontem:
- `SWAT`
- `SWAT-CUP`
- `SUFI-2`
- `25` akim gozlem istasyonu ile `1977-2013` kalibrasyon/dogrulama

Projeye anlami:
- Uzun vadede baraj doluluk modelinin en buyuk eksigi `dogrudan inflow`
  gozleminin olmamasi.
- Bu kaynak Istanbul icin hidrolojik vekil katman kurmanin akademik olarak
  savunulabilir oldugunu gosteriyor.
- Kisa vadede yagis-lag-havza alani vekili, orta vadede ise
  `source-aware runoff emulator` gelistirmek mantikli.

## Attigin Uc Makaleden Dogrudan Eklenebilecekler

Bu bolum sadece daha once attigin uc makaleden projeye dogrudan
tasinabilecek maddeleri toplar.

### 1. `Extra Trees` benchmark zorunlu olmali

Kaynak:
- Sustainability 2024 Istanbul baraj doluluk calismasi

Dogrudan kullanim:
- bir sonraki model turunda `Extra Trees` resmi benchmark olarak calisacak
- yalnizca `LSTM` veya lineer modellerle yetinilmeyecek

Neden ise yarar:
- makale dogrudan Istanbul barajlari uzerinde calisiyor
- en iyi performans adayinin `Extra Trees` oldugunu raporluyor
- bizdeki mevcut `rain + ET0 + demand + memory` yapisina kolayca uygulanir

### 2. `yalniz tarihsel` ve `hibrit` benchmark ayrimi formalize edilmeli

Kaynak:
- Sustainability 2024 Istanbul baraj doluluk calismasi

Dogrudan kullanim:
- tum yeni deneylerde iki zorunlu benchmark tutulacak:
  - `yalniz tarihsel`: lagli doluluk + mevsimsellik
  - `hibrit`: lagli doluluk + iklim + talep + operasyon

Neden ise yarar:
- meteorolojik ve operasyonel bloklarin gercek katkisi boyle net gorulur
- sunumda "dis degisken eklenince ne fark etti" sorusuna acik cevap verir

### 3. Kisa ufukta `gunes radyasyonu`, `cig noktasi`, `gun uzunlugu` bloklari daha sistemli test edilmeli

Kaynak:
- Sustainability 2024 Istanbul baraj doluluk calismasi

Dogrudan kullanim:
- aktinograf geldikten sonra radyasyon blogu daha guclu sekilde modele girecek
- `cig noktasi` veya ona yakin nem turevleri ayri feature testi olarak eklenecek
- `gun uzunlugu` deterministik bir astronomik feature olarak kolayca uretilecek

Neden ise yarar:
- bu degiskenler makalede secilen hava suruculeri arasinda
- ozellikle kisa ve orta ufukta mevsimsel fiziksel sinyali guclendirir

### 4. `toprak isi akisinin sifir alinmasi` gunluk olcekte pratik bir kabul

Kaynak:
- Sustainability 2024 Istanbul baraj doluluk calismasi

Dogrudan kullanim:
- gunluk ET0 testlerinde ilk surumde `G = 0` kabulunu koruyabiliriz
- saatlik veya alt-gunluk modele gecmeden once bunu degistirmemize gerek yok

Neden ise yarar:
- ilk model karmasikligini dusurur
- mevcut veri kalitesi ile uyumludur

### 5. `acik su buharlasmasi` ET0'dan ayri tutulmali

Kaynak:
- Results in Engineering 2025 Iran dam evaporation calismasi

Dogrudan kullanim:
- `ET0` sadece atmosferik talep olarak kalacak
- reservoir kaybi icin ayri bir `acik su buharlasmasi vekili` kurulacak
- ilk vekil:
  `baraj alanı x iklim talebi x senaryo katsayisi`

Neden ise yarar:
- baraj yuzey kaybini daha fiziksel temsil eder
- yaz baskisini daha gercekci aciklar

### 6. Yeniden analiz / uydu tabanli buharlasma destek katmani savunulabilir

Kaynak:
- Results in Engineering 2025 Iran dam evaporation calismasi

Dogrudan kullanim:
- dogrudan olcum yoksa `ERA5` benzeri yeniden analiz urunleri destek katmani
  olarak kullanilabilir
- bu katman uretim modelinin yerine degil,
  `sanity check`, `backfill`, `duyarlilik` ve `alan tabanli kayip` tasarimi
  icin kullanilacak

Neden ise yarar:
- veri boslugunu daha savunulabilir sekilde kapatir
- actinograph gelene kadar gecis cozumune akademik dayanak verir

### 7. `toparlanma gecikmesi` metrik olarak eklenmeli

Kaynak:
- JRAS 2020 Istanbul baraj doluluk calismasi

Dogrudan kullanim:
- bir sok veya kurak donem sonrasi,
  dolulugun eski trendine donus suresi olculecek
- bu metrik hem senaryo ciktilarinda hem sunum dilinde yer alacak

Ornek yorum:
- "ayni yagis geri donse bile sistem onceki doluluk seviyesine
  hemen donmuyor"

Neden ise yarar:
- sadece minimum veya ortalama doluluk degil,
  sistemin toparlanma hizi da gorulur
- su kesintisi, talep kisiti veya dis kaynak artisi gibi mudahalelerin
  kalici etkisini tartismayi kolaylastirir

### 8. `aylik + mevsimsel + yillik` birlikte okunmali

Kaynak:
- JRAS 2020 Istanbul baraj doluluk calismasi

Dogrudan kullanim:
- gelecegi anlatirken sadece aylik yol vermeyecegiz
- ayni projeksiyon icin:
  - aylik yol
  - mevsimsel stres ozetleri
  - yillik kapanis veya yillik ortalama
  birlikte sunulacak

Neden ise yarar:
- teknik olmayan kullanicinin tabloyu anlamasini kolaylastirir
- baraj sistemindeki kalici egilim ile mevsimsel oynakligi ayirmamizi saglar

## Bizim Proje Icin En Dogru Gelecek Projeksiyonu Kurgusu

### A. Cekirdek mantik

Gelecek doluluk yolu su sekilde uretilmeli:

`gelecek doluluk = onceki depolama + iklim kaynakli giris - buharlasma/ET talebi - insan kullanim baskisi +/- transfer ve mudahale etkileri`

Burada her blok ayri senaryoya baglanabilir.

### B. Kullanilacak senaryo aileleri

Ilk asamada en savunulabilir minimum set:

1. `Temel senaryo`
   tarihsel iklim desenine yakin,
   resmi talep trendine yakin,
   mevcut kayip ve transfer yapisina yakin

2. `Ilik / islak senaryo`
   yagista hafif pozitif sapma,
   ET0'da zayif artis,
   talepte kontrollu artis

3. `Sicak-kurak-yuksek talep senaryosu`
   yagista azalis veya daha kotu mevsim dagilimi,
   ET0 artis,
   talep ve kayip baskisinin surmesi

4. `Yonetim iyilesme senaryosu`
   NRW dususu,
   geri kazanim artisi,
   kisit / tasarruf veya verimlilik etkisi

### C. Iklim verisini nasil uretecegiz

En pratik ve savunulabilir ilk yol:

1. mevcut tarihsel aylik seri kalir
2. resmi MGM / IBB / CMIP6 kaynaklarindan yakin gelecek delta mantigi cikartilir
3. bu deltalar tarihsel aylik profile uygulanir
4. yagis, sicaklik, gerekirse radyasyon ve nem uzerinden yeni ET0 uretilir

Bu yapi "ham GCM'i dogrudan modele basmak"tan daha saglamdir.

### D. ET0 tarafi

ET0 icin en dogru yol:
- tarihsel seride `Penman-Monteith`
- aktinograf geldikten sonra gozlenen radyasyon ile yeniden kalibrasyon
- gelecek senaryoda sicaklik/radyasyon/nem/ruzgar blogundan yeni ET0 uretimi

Eger tam meteorolojik degisken seti gelecekte kolay bulunamazsa:
- `delta ET0` yaklasimi
- veya `sicaklik + yagis + radyasyon vekili` ile senaryo ET0 katsayisi
  uygulanabilir

### E. Talep tarafi

Talep tek degisken olarak tutulmamali.

En azindan su bilesenlere ayrilmali:
- subscriber / nufus buyumesi
- kisi basi kullanim veya kullanim yogunlugu
- fiziksel su kaybi / NRW
- geri kazanilmis su
- kisit / tasarruf / mudahale etkisi
- dis kaynak varligi

### F. Cikti dili

Tek bir "2040 doluluk tahmini" vermek yerine:

- `P10 / P50 / P90` benzeri bant
- temel, iyi ve kotu senaryo
- belirli degiskenlerin etkisini ayri gosterme

daha dogru olur.

## Bizim Simdiki Veri Yapimizla Ne Kadarina Ulasabiliriz

Simdiden yapabileceklerimiz:
- aylik toplu doluluk hedefi
- yagis, ET0, sicaklik, nem, demand, memory bloklari
- resmi supply ve operasyon baglami
- senaryo motoru
- kaynak bazli baglam

Hemen yapamayacaklarimiz:
- dogrudan havza inflow olcumu
- gercek aylik NRW serisi
- gelecege ait tam gozlenmis radyasyon
- aylik sektor bazli cekim

Sonuc:
- `hafif su-denge + senaryo` tipi 15 yillik projeksiyon simdi kurulabilir
- `tam fiziksel kaynak-bazli hidrolojik model` icin ek veri gerekir

## Uygulanabilir Yol Haritasi

### Faz 1 - Simdi

- mevcut aylik model paketini koru
- `base / wet-mild / hot-dry-high-demand / management-improvement`
  senaryolarini kur
- resmi nufus ve iklim baglamiyla 2026-2040 aylik yol uret

### Faz 2 - Sonraki teknik tur

- aktinograf ile ET0 kalitesini artir
- havza alani + yagis + lag ile inflow proxy kur
- baraj yuzey alanina dayali acik su buharlasmasi vekili ekle

### Faz 3 - Guclu akademik surum

- `CMIP6 ensemble + QDM`
- kaynak bazli runoff emulator veya WEAP-lite
- senaryo bantlari ve belirsizlik zarfi

## Net Karar

Gelecek projeksiyonu icin en dogru is:

- tek model tek cizgi vermek degil
- `senaryo tabanli`, `ensemble mantikli`, `iklim + talep + isletme`
  birlesik bir sistem kurmak

Bu proje icin teknik olarak en savunulabilir ilk urun:

`2026-2040 aylik toplam baraj doluluk projeksiyonu`
`(temel / iyi / kotu / yonetim iyilesme senaryolari ile)`

Bu, hem mevcut veri yapimizla yapilabilir, hem de sunumda savunulabilir.
