# Baraj ET0 Çalışması - Yöntem, Kabuller ve Nedenleri

## 1. Bu çalışmada ne yaptık?

Bu çalışmada baraj sistemi için `referans evapotranspirasyon (ET0)` hesabı yaptık.
ET0, doğrudan açık su yüzeyi buharlaşması değildir; ancak atmosferik buharlaşma talebini
fiziksel olarak tutarlı şekilde temsil eden standart göstergedir. Baraj su dengesi
modelinde bu değer, açık su buharlaşması hesabı için temel meteorolojik girdi olarak
kullanılır.

Bu paketin amacı:

- geçmiş ET0 davranışını hesaplamak,
- grafiklerle okunur hale getirmek,
- uzun dönem trendi görmek,
- ve ileriye dönük ET0 öngörüsü üretmektir.

Burada hedef değişken `ET0`'dır. Bu, iyi sulanmış referans yüzey için atmosferin
ne kadar su talep ettiğini gösterir.

## 2. Neden ET0 hesapladık?

Baraj suyu için ideal değişken, açık su yüzeyi buharlaşmasıdır. Ancak bunun doğrudan
hesabı, yerel pan/rezervuar katsayıları ve detaylı yüzey enerji dengesi gerektirir.
ET0 ise FAO-56 standardı ile hesaplanan, veri gereksinimi daha düşük ve karşılaştırılabilir
bir göstergedir.

Bu nedenle bu pakette:

- meteorolojik buharlaşma talebini ET0 ile temsil ediyoruz,
- baraj su dengesi modelinde ET0'ı buharlaşma hesabına ölçekli girdi olarak kullanıyoruz,
- ileride `E = K * ET0` gibi bir katsayı ile açık su yüzeyi buharlaşmasına dönüştürme
  seçeneğini açık bırakıyoruz.

## 3. Kullandığımız ana formül

Baraj ET0 hesabında `FAO-56 Penman-Monteith` formülünü kullandık:

`ET0 = [0.408*Delta*(Rn-G) + gamma*(900/(T+273))*u2*(es-ea)] / [Delta + gamma*(1 + 0.34*u2)]`

Bu formül iki ana etkiyi birleştirir:

- enerji etkisi
- aerodinamik etki

Yani hem yüzeye gelen net enerjiyi hem de havanın kurutma gücünü birlikte hesaba katar.

## 4. Formüldeki terimler ne işe yarar?

### `Rn`

Net radyasyondur. Yüzeyde evapotranspirasyon için kullanılabilir enerjiyi temsil eder.

### `G`

Zemine giden ısı akısıdır. Günlük ET0 hesabında genellikle `0` kabul edilir.

### `Delta`

Doygun buhar basıncı eğrisi eğimidir. Sıcaklık değişimine karşı buharlaşma
hassasiyetini gösterir.

### `gamma`

Psikrometrik sabittir. Enerji terimi ile aerodinamik terimi aynı ölçekte dengeler.

### `u2`

2 metre yükseklikte rüzgar hızıdır. Rüzgar arttıkça havanın nem taşıma kapasitesi artar.

### `es - ea`

Buhar basıncı açığıdır. Havanın ne kadar kuru olduğunu gösterir.
Değer büyüdükçe atmosferin su çekme talebi artar.

## 5. Kullandığımız kabuller ve nedenleri

### 5.1 `Tmean = (Tmax + Tmin) / 2`

Bunu seçmemizin nedeni, veri yapımızın bu tanımı en temiz ve en tutarlı şekilde
desteklemesidir.

Bu ifade `Delta` değildir.
Bu sadece ortalama sıcaklığı verir. Sonra `Delta`, bu `Tmean` üzerinden fiziksel
denklemle hesaplanır.

### 5.2 `Delta = f(Tmean)`

`Delta`'yı doğrudan ortalama sıcaklık olarak almak yanlıştır.
Bu yüzden `Delta`, FAO-56'da verilen sıcaklığa bağlı türev yapısıyla hesaplandı.

Bunu yapma nedenimiz:

- fiziksel olarak doğru olmak,
- literatürle uyumlu olmak,
- hesaplamayı savunulabilir yapmak.

### 5.2.1 `Delta` gün içinde neden değişir?

`Delta`, sıcaklığa bağlı bir terimdir. Sıcaklık gün boyunca sabit olmadığı için
`Delta` da sabit değildir.

Pratik yorum:

- sabah saatlerinde sıcaklık düşüktür, bu yüzden `Delta` da daha düşüktür
- öğleden sonra sıcaklık maksimuma yaklaşır, bu yüzden `Delta` genellikle en yüksektir

Ama bu paket günlük ölçekte kurulduğu için saatlik `Delta` yerine tek bir günlük
`Delta` kullandık.

Günlük yaklaşım:

`Tmean = (Tmax + Tmin) / 2`

sonra:

`Delta = f(Tmean)`

Nedeni:

- kullandığımız operasyonel ET0 paketi günlük seri üzerine kurulu
- mevcut model penceresi `3747` gün, `120` ay ve `10` tam yıldan oluşuyor
- FAO-56 günlük ET0 uygulamasıyla uyumlu

Eğer elimizde aynı zaman adımında:

- saatlik sıcaklık
- saatlik nem
- saatlik rüzgar
- saatlik radyasyon

olsaydı, saatlik ET0 da hesaplayabilirdik. Bu daha ayrıntılı ama veri gereksinimi
çok daha yüksek bir katmandır.

### 5.3 `G = 0`

Günlük ET0 hesabında `G = 0` kabul edildi.

Nedeni:

- FAO-56'ya uygundur,
- günlük ölçekte makul bir standart kabuldür,
- gereksiz model karmaşasını engeller.

### 5.4 `u2 = 2.0 m/s`

Uzun ve sürekli bir rüzgar serimiz olmadığı için `u2 = 2.0 m/s` sabit fallback
kullanıldı.

Bunu yapma nedenimiz:

- tüm seri boyunca hesap sürekliliğini korumak,
- veri boşluğu yüzünden modeli durdurmamak,
- literatürde kullanılan pratik bir fallback'e dayanmak.

Bu, modelin en güçlü değil ama en pratik kabullerinden biridir. Gerçek ve uzun bir
rüzgar serisi gelirse ilk iyileştirilmesi gereken noktalardan biri budur.

### 5.5 Basıncın rakımdan türetilmesi

Basınç ölçümü olmayan dönemlerde, basınç değeri istasyon rakımı üzerinden FAO-56
formülü ile türetildi.

Bunun nedeni:

- veri sürekliliğini korumak,
- basınç girdisini fiziksel bir yerçekimi modeline bağlamak,
- hesaplama kararlılığını artırmak.

## 6. Özet

Bu ET0 paketi, baraj su dengesi analizinde kullanılabilecek, fiziksel olarak tutarlı
ve savunulabilir bir meteorolojik buharlaşma talebi serisi üretir.

İleride yapılabilecek doğal geliştirmeler:

- açık su yüzeyi buharlaşması için `K` katsayısı kalibrasyonu,
- istasyon bazlı rüzgar serisi ile `u2`'nin güncellenmesi,
- saatlik ET0 hesabına geçiş (veri uygunsa).
