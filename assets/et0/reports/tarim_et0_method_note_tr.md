# Tarimsal ET0 Calismasi - Yontem, Kabuller ve Nedenleri

## 1. Bu calismada ne yaptik?

Bu calismada tarim icin `referans evapotranspirasyon (ET0)` hesapladik.  
Amacimiz, mevcut meteorolojik veri ve radyasyon girdisi ile:

- gecmis ET0 davranisini hesaplamak,
- bunu grafiklerle okunur hale getirmek,
- genel trendi gormek,
- ve gelecege donuk bir ET0 oengorusu uretmekti.

Burada hedef degisken `ET0`'dir. Bu, iyi sulanmis referans bir bitki ortami icin atmosferin ne kadar su talep ettigini gosterir.

## 2. Neden tarimsal ET0 hesapladik?

Daha once konustugumuz gibi, baraj suyu icin kullanilan yaklasim ile tarim icin kullanilan yaklasim ayni degildir.

- Tarimda temel degisken: `ET0` veya `ETc`
- Acik su yuzeyinde temel degisken: `evaporasyon`

Bu calismada once tarim tarafini kurduk. Cunku senin istedigin ana model tarim icin ET hesabiydi. Sonra istersek bunu urun bazli `ETc = Kc * ET0` katmanina tasiyabiliriz.

## 3. Kullandigimiz ana formul

Tarimsal ET0 hesabinda `FAO-56 Penman-Monteith` formulunu kullandik:

`ET0 = [0.408*Delta*(Rn-G) + gamma*(900/(T+273))*u2*(es-ea)] / [Delta + gamma*(1 + 0.34*u2)]`

Bu formul iki ana etkiyi birlestirir:

- enerji etkisi
- aerodinamik etki

Yani hem yuzeye gelen net enerjiyi hem de havanin kurutma gucunu birlikte hesaba katar.

## 4. Formuldeki terimler ne ise yarar?

### `Rn`

Net radyasyondur. Yuzeyde evapotranspirasyon icin kullanilabilir enerjiyi temsil eder.

### `G`

Zemine giden isi akisidir. Gunluk tarimsal ET0 hesabinda genellikle `0` alinir.

### `Delta`

Doygun buhar basinici egri egimidir. Sicaklik degisimine karsi buharlasma duyarliligini gosterir.

### `gamma`

Psikrometrik sabittir. Enerji terimi ile aerodinamik terimi ayni olcekte dengeler.

### `u2`

2 metre yukseklikte ruzgar hizidir. Ruzgar arttikca havanin nem tasima kapasitesi artar.

### `es - ea`

Buhar basinici acigidir. Havanin ne kadar kuru oldugunu gosterir. Deger buyudukce atmosferin su cekme talebi artar.

## 5. Kullandigimiz kabuller ve nedenleri

### 5.1 `Tmean = (Tmax + Tmin) / 2`

Bunu secmemizin nedeni, veri yapimizin bu tanimi en temiz ve en tutarli sekilde desteklemesiydi.

Bu ifade `Delta` degildir.  
Bu sadece ortalama sicakligi verir. Sonra `Delta`, bu `Tmean` uzerinden fiziksel denklemle hesaplanir.

### 5.2 `Delta = f(Tmean)`

`Delta`'yi dogrudan ortalama sicaklik olarak almak yanlistir.  
Bu yuzden `Delta`, FAO-56'da verilen sicakliga bagli turev yapisiyla hesaplandi.

Bunu yapma nedenimiz:

- fiziksel olarak dogru olmak,
- literaturle uyumlu olmak,
- hesaplamayi savunulabilir yapmak.

### 5.2.1 `Delta` gun icinde neden degisir?

`Delta`, sicakliga bagli bir terimdir. Sicaklik gun boyunca sabit olmadigi icin `Delta` da sabit degildir.

Pratik yorum:

- sabah saatlerinde sicaklik dusuktur, bu yuzden `Delta` da daha dusuktur
- ogleden sonra sicaklik maksimuma yaklasir, bu yuzden `Delta` da genellikle en yuksektir

Ama bu paket gunluk olcekte kuruldugu icin saatlik `Delta` yerine tek bir gunluk `Delta` kullandik.

Gunluk yaklasim:

`Tmean = (Tmax + Tmin) / 2`

sonra:

`Delta = f(Tmean)`

Nedeni:

- kullandigimiz operasyonel ET0 paketi gunluk seri uzerine kurulu
- mevcut model penceresi `3747` gun, `120` ay ve `10` tam yildan olusuyor
- FAO-56 gunluk ET0 uygulamasiyla uyumlu

Eger elimizde ayni zaman adiminda:

- saatlik sicaklik
- saatlik nem
- saatlik ruzgar
- saatlik radyasyon

olsaydi, saatlik ET0 da hesaplayabilirdik. Bu daha ayrintili ama veri gereksinimi daha yuksek bir katmandir.

### 5.3 `G = 0`

Gunluk tarimsal ET0 hesabinda `G = 0` kabul edildi.

Nedeni:

- FAO-56'ya uygundur,
- gunluk olcekte makul bir standart kabuldur,
- gereksiz model karmasini engeller.

### 5.4 `u2 = 2.0 m/s`

Uzun ve surekli bir ruzgar serimiz olmadigi icin `u2 = 2.0 m/s` sabit fallback kullanildi.

Bunu yapma nedenimiz:

- tum seri boyunca hesap surekliligini korumak,
- veri boslugu yuzunden modeli durdurmamak,
- literaturde kullanilan pratik bir fallback'e dayanmak.

Bu, modelin en guclu degil ama en pratik kabullerinden biridir. Gercek ve uzun bir ruzgar serisi gelirse ilk iyilestirilmesi gereken noktalardan biri budur.

### 5.5 Basincin rakimdan turetilmesi

Psikrometrik sabit icin basinç gerekir. Uzun donem surekli basinç serisi yerine rakimdan turetilen sabit basinç kullanildi.

Nedeni:

- ET0 uzerindeki etkisi sinirlidir,
- seri boyunca tutarlilik saglar,
- eksik veri etkisini azaltir.

### 5.6 Radyasyonun dogrudan dosyadan alinmasi

Senin verdigin dosya:

`gunluk_gunes_radyasyonu_veri_seti.csv` (dosya yolu bu repoda paylasilmiyor)

Bu dosyayi ET0 hesabinda dogrudan kullandik.

Nedeni:

- radyasyon ET0 formulu icin ana girdilerden biridir,
- tahmini bir radyasyon yerine elindeki gercek/uretlenmis seri daha tutarli bir temel verir,
- modeli daha fiziksel hale getirir.

### 5.7 `coverage_frac >= 0.80` kuralinin kullanilmasi

Modeli gelistirirken en kritik duzeltmelerden biri buydu.

Ay bazinda eksik gunler varken o ayi dogrudan modele sokmak, aylik toplami yapay olarak dusuk gosterebiliyordu. Bu da trendi ve forecast'i bozuyordu.

Bu yuzden:

- her ay icin veri kapsama orani hesaplandi,
- sadece `%80` ve ustu kapsama sahip aylar modele alindi.

Nedeni:

- eksik aylarin sahte dusus gibi gozukmesini engellemek,
- yillik seriyi daha temiz kurmak,
- forecast tarafinda daha savunulabilir bir egitim penceresi elde etmek.

## 6. `es - ea` neden boyle kuruldu?

`es - ea`, havanin kuruluk talebini verir. Veri yapisina gore bu terimi sicaklik ve nem bilgilerinden turettik.

Bu terim onemlidir cunku:

- hava ne kadar kuruysa ET talebi o kadar artar,
- ayni sicaklikta bile farkli nem kosullari farkli ET0 uretir.

Yani ET0'yi sadece sicakliga baglamak yerine nem farkini da hesaba katmis olduk.

## 7. Modeli neden gelistirdik?

Ilk surum hesaplamayi uretse de yorum tarafinda iki sorun vardi:

### 7.1 Eksik aylar trendi bozuyordu

Kismi aylar bazen dusuk ET0 gibi davranip genel trendi yanlis yone cekebiliyordu.

Bu nedenle:

- aylik kapsama filtresi eklendi,
- yillik seri sadece guvenilir aylardan kuruldu.

### 7.2 Trend cizgisi asiri degerlerden etkileniyordu

Basit dogrusal trend bazen yillik oynakliktan fazla etkilenebilir.

Bu nedenle:

- genel trend icin `Theil-Sen`
- desen okumasi icin `LOWESS`

kullanildi.

Bu secimin nedeni:

- aykiri yillara karsi daha dayanikli olmak,
- cizginin veriyi daha dogru temsil etmesini saglamak,
- sadece tek dogrusal egim gostermek yerine gercek deseni gormek.

## 8. Neden `R kare` ve `p` degerlerini grafikten kaldirdik?

Bu degerleri teknik olarak anlatabiliriz, ama grafik uzerinde tutmak iki sorun olusturuyordu:

- gorseli kalabaliklastiriyordu,
- izleyicinin dikkatini ana desenden alip istatistik etiketine cekiyordu.

Bu nedenle grafiklerde:

- trendin kendisi,
- maksimum ve minimum noktalar,
- gercek radiation gunlerinin dagilimi

gosterildi.

Teknik istatistikler ise ayrica raporda tutulabilir.

## 9. Grafiklerde neden bu duzeni kullandik?

Gorsellerin sol tarafinda ayri bilgi bloklari olusturduk:

### `Formul`

Kullanilan fiziksel cekirdegi acikca gormek icin.

### `Formul nasil okunur`

Izleyicinin denklemi ezberlemeden mantigini anlamasi icin.

### `Terimler ne ise yarar`

`Rn`, `Delta`, `es-ea`, `u2`, `gamma`, `G` gibi terimlerin islevini kisa ve net anlatmak icin.

### `Kabuller ve nedenleri`

Modelin nerede fiziksel, nerede pratik fallback kullandigini acik gostermek icin.

### `Bu grafikte`

Grafik ozelinde hangi sonuc veya desenin one ciktigini tek bakista anlatmak icin.

Bu duzenin amaci, grafikleri sadece gorsel degil ayni zamanda aciklayici hale getirmekti.

## 10. Forecast tarafinda ne yaptik?

Tarihsel aylik ET0 serisini kullanip quant tabanli bir oengoru akisi kurduk.

Burada ana mantik suydu:

- once temiz ve guvenilir tarihsel pencereyi kurmak,
- sonra forecast modelini bunun uzerinde calistirmak,
- son olarak gelecekte ET0'nun yukari mi asagi mi gittigini okumak.

Bu asamada forecast, ET0 serisinin kendi desenine dayaniyor.  
Yani ileri donem ruzgar veya ileri donem radyasyon icin ayri senaryo vermedik.

## 11. Bu modelin guclu yonleri

- Fiziksel olarak dogru bir ET0 formulune dayaniyor.
- Gercek radyasyon dosyasini kullaniyor.
- Eksik aylarin bozucu etkisini filtreliyor.
- Trend yorumunda robust yontem kullaniyor.
- Gorsellerde model mantigini acikliyor.

## 11.1 Bu paketin sayisal ozeti

- Gunluk satir sayisi: `3747`
- Aylik satir sayisi: `120`
- Tam yillik seri: `10 yil`
- Model penceresi: `1995-01-01` -> `2004-12-01`
- Ortalama yillik ET0: `919.1 mm/yil`
- Trend: `+59.2 mm / 10 yil`
- `real_extracted` radyasyon gunu: `906`
- `synthetic` radyasyon gunu: `2841`

## 12. Bu modelin sinirlari

- Ruzgar sabit fallback ile temsil edildi.
- Radyasyon serisinin bir bolumu `synthetic`.
- Forecast tarafinda ileri donem dissal iklim girdileri ayri senaryo olarak verilmedi.
- Bu yapi referans ET0 verir; urun su tuketimi icin dogrudan `ETc` degildir.

## 13. Bir sonraki mantikli adimlar

1. Gercek uzun donem ruzgar serisi eklemek  
2. Urun bazli `ETc = Kc * ET0` katmani kurmak  
3. Forecast'i `Rs`, `VPD`, `Tmean` gibi dissal degiskenlerle guclendirmek  
4. Bu notu sunum/PDF icin daha kisa bir yoneci ozeti haline getirmek
