# Senaryo Web Arayuzu Fikri

Tarih: 2026-03-12

## Amac

Baraj doluluk tahmin modelini sadece teknik ekip icin degil, teknik olmayan
kullanicilar icin de anlasilir hale getirmek.

Temel fikir:
Kullanicinin bazi ana degiskenleri degistirebildigi bir web sitesi kurmak ve
girilen degerlere gore gelecek doluluk yolunun nasil degistigini aninda
gostermek.

## Neden Gerekli

- Modelin ne yaptigini tablo okumadan anlatir.
- Sunumda "hangi degisken neyi degistiriyor" sorusuna dogrudan cevap verir.
- Karar verici icin sadece tahmin degil, senaryo karsilastirma araci olur.
- Teknik olmayan kullanicilarin modeli anlamasini kolaylastirir.

## Ilk Surumde Olmasi Gerekenler

- Baslangic toplam doluluk orani
- Yagis degisimi
- Referans evapotranspirasyon (ET0) degisimi
- Toplam talep / tuketim degisimi
- Tasarruf veya kisit senaryosu
- Fiziksel su kaybi / gelir getirmeyen su (NRW) degisimi
- Dis kaynak / transfer katkisi varsa onun degisimi

## Gosterilecek Ciktilar

- Temel senaryo ile kullanicinin senaryosunu ayni grafikte gosterme
- Aylik toplam doluluk projeksiyonu
- "Bu senaryoda en buyuk etki hangi degiskenden geldi" ozeti
- Kritik esik uyari alanlari
- Istenirse baraj bazli ikinci gorunum

## Urun Dili

Arayuz sade olmali.
Teknik terim kullanilacaksa once Turkcesi, sonra parantez icinde teknik karsiligi verilmeli.

Ornek:
- referans evapotranspirasyon (ET0)
- fiziksel su kaybi / gelir getirmeyen su (NRW)
- acik su yuzeyi buharlasmasi

## Teknik Not

Bu arayuz yeni bir model kurmayacak.
Mevcut senaryo motorunun ve aylik model paketinin ustune oturacak.

Ilk veri tabani / girdi katmani:
- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_model_core_monthly.csv`
- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_model_extended_monthly.csv`
- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_source_current_context.csv`

## Ilk Demo Icin Minimum Kapsam

- Tek sayfa arayuz
- 5-6 degisken icin kaydirici veya sayisal giris
- 15 yillik projeksiyon grafigi
- Temel senaryo ve degistirilmis senaryonun karsilastirilmasi
- Kisa aciklama kutulari

## Karar

Bu fikir, projenin ilerideki ana sunum ve urunlestirme katmani olarak
saklanacaktir.
