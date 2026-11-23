# Türkçe Büyük Dil Modellerinde Nicemleme Tekniklerinin Morfolojik Bozulma Üzerindeki Etkisi

CEN401 Bitirme Tezi, takım **OEC-1049**
Fatma Ciran Akbaş (kaptan) · Öznur Beyazpınar
Danışman: Dr. Öğr. Üyesi Ömer Emin Çınar

Bu depo, tezin deney kodunu ve sonuç tablolarını içerir. Şu anki içerik **ön çalışma** aşamasıdır:
bir modelin FP16 (nicemlenmemiş), INT8 ve NF4 (4 bit) sürümlerinin Türkçe dil bilgisi minimal
çiftlerinde (TurBLiMP) ne kadar doğru karar verdiğini, bellek kullanımını ve hızını ölçer.

## Ön deney ne yapıyor?

1. TurBLiMP'in 16 dil bilgisi olgusundan her biri için aynı tohumla 300 minimal çift seçilir.
   Bir minimal çift, yalnızca bir yerde farklı olan iki cümledir: biri dil bilgisine uygun, biri değil.
2. Model her cümleye bir log olasılık verir. Uygun cümleye daha yüksek olasılık verdiyse çift **doğru** sayılır.
3. Aynı model üç kez yüklenir: FP16, INT8 (bitsandbytes LLM.int8) ve NF4 (bitsandbytes 4 bit).
4. Her sürüm için doğruluk, ağırlık belleği, tepe GPU belleği ve token başına üretim süresi kaydedilir.
5. `ozetle.py`, her olguda nicemlenmiş sürümün FP16'ya göre farkını hesaplar ve farkın anlamlı olup
   olmadığını eşleştirilmiş McNemar testiyle sınar.

Ön deneyde küçük bir model (Qwen2.5-1.5B-Instruct) kullanılmıştır; amaç hattın uçtan uca çalıştığını
göstermek ve ilk ölçümleri almaktır. Ana deneylerdeki 7–8B modeller ve GPTQ, AWQ, GGUF yöntemleri
Bahar döneminde eklenecektir.

## Depo yapısı

```
calistir.py                  ön deneyi çalıştırır (model × varyant → results/)
ozetle.py                    sonuçları özetler, tablo ve şekil üretir
src/veri.py                  TurBLiMP okuma, hatalı çiftleri ayıklama, örnekleme
src/model.py                 FP16 / INT8 / NF4 yükleme, cümle log olasılığı
src/olcum.py                 bellek, gecikme, sözcük başına token
tests/test_veri_ozet.py      GPU gerektirmeyen birim testleri
notebooks/on_calisma_colab.ipynb   Google Colab'da adım adım çalıştırma
veri_indir.sh                TurBLiMP verisini sabit commit'te indirir
results/                     deney çıktıları (CSV, JSON, PNG)
```

## Çalıştırma

En kolay yol `notebooks/on_calisma_colab.ipynb` defterini Google Colab'da T4 GPU ile açmaktır.
Komut satırından:

```bash
pip install -r requirements.txt
bash veri_indir.sh
python -m pytest -q                                   # GPU gerektirmez
python calistir.py --n 5 --varyantlar fp16 --cikti results_duman   # hızlı deneme
python calistir.py --model Qwen/Qwen2.5-1.5B-Instruct --n 300      # ön deney
python ozetle.py
```

CUDA destekli bir NVIDIA GPU gerekir (bitsandbytes nicemlemesi GPU üzerinde çalışır).

## Çıktılar

| Dosya | İçerik |
|---|---|
| `results/ciftler_<varyant>.csv` | Her çift için iki cümlenin log olasılığı ve doğru/yanlış |
| `results/kaynak.csv` | Varyant başına doğruluk, bellek, tepe bellek, gecikme, süre |
| `results/ortam.json` | GPU adı, CUDA, Python ve kütüphane sürümleri, model, veri ayarları |
| `results/olgu_tablosu.csv` | Olgu × varyant doğruluk, FP16'ya göre fark, McNemar p |
| `results/ozet.md` | Rapora konabilecek okunur tablo |
| `results/sekil_olgu_fark.png` | Olgu bazında FP16'ya göre doğruluk farkı |

## Ön deney sonuçları (23.11.2026)

Ortam: Kaggle, Tesla T4 (16 GB), CUDA 13.0, Python 3.13, PyTorch 2.11, Transformers 5.20.2,
bitsandbytes 0.50.2. Model: Qwen2.5-1.5B-Instruct. Veri: 16 olgu × 300 çift = 4.800 minimal çift.

| Varyant | Doğruluk (%) | Model belleği (GB) | Tepe GPU belleği (GB) | Token başına süre (ms) |
|---|---|---|---|---|
| FP16 | 78,9 | 2,88 | 4,51 | 42,7 |
| INT8 | 78,5 | 1,66 | 3,29 | 169,4 |
| NF4  | 76,7 | 1,05 | 2,76 | 60,0 |

Olgu bazındaki sonuçlar `results/ozet.md` ve `results/sekil_olgu_fark.png` dosyalarındadır.

## Veri hakkında notlar

- TurBLiMP: Başar vd., EMNLP 2025, <https://github.com/ezgibasar/turblimp> (CC BY 4.0).
  Veri depoya konmaz; `veri_indir.sh` commit `297de13` sürümünü indirir.
- `anaphor_agreement` ve `quantifiers` dosyalarının her birinde iyi ve kötü cümlesi birebir aynı olan
  10 çift vardır. Bu çiftler ayırt edici olmadığı için `src/veri.py` tarafından çıkarılır.
- `phenomenon` sütununda bazı değerlerin sonunda boşluk vardır ve geçişli/çift geçişli dosyaların ikisi
  de `argument_structure` yazar. Bu yüzden olgu adı dosya adından alınır (16 olgu).

## Yapay zekâ kullanımı

Bu depodaki kodun ilk sürümü (tüm `.py` dosyaları, `veri_indir.sh`, `notebooks/on_calisma_colab.ipynb`
ve bu README) **bir yapay zekâ asistanı desteğiyle yazılmıştır**. Kod takım üyeleri tarafından
satır satır incelenmiş ve Google Colab'da çalıştırılmıştır; sonuç dosyaları (`results/`) takımın kendi
çalıştırmasından gelmektedir. Sonraki değişikliklerde YZ desteği commit mesajında
`[YZ destekli]` etiketiyle belirtilecektir.
