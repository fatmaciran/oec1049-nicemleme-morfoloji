"""TurBLiMP minimal çiftlerini okuyan ve ön deney için örnekleyen modül."""

from pathlib import Path

import pandas as pd


def turblimp_yukle(klasor, n=None, tohum=42):
    """TurBLiMP'in 16 olgusuna ait CSV dosyalarını tek tabloda birleştirir.

    klasor : TurBLiMP deposundaki data/base klasörünün yolu
    n      : her olgudan alınacak çift sayısı (None ise hepsi)
    tohum  : örnekleme için rastgelelik tohumu (aynı tohum = aynı örnekler)

    Döndürür: (tablo, ayiklanan) -- ayiklanan, olgu başına çıkarılan
    hatalı (iyi ve kötü cümlesi aynı olan) çift sayısıdır.
    """
    dosyalar = sorted(Path(klasor).glob("augmented_*.csv"))
    if not dosyalar:
        raise FileNotFoundError(f"{klasor} içinde augmented_*.csv bulunamadı")

    parcalar = []
    ayiklanan = {}
    for dosya in dosyalar:
        # TurBLiMP dosyaları noktalı virgülle ayrılmıştır.
        tablo = pd.read_csv(dosya, sep=";", encoding="utf-8")
        # Olgu adını dosya adından alıyoruz: "phenomenon" sütununda bazı
        # satırların sonunda boşluk var ve geçişli/çift geçişli dosyaların ikisi
        # de "argument_structure" yazıyor; dosya adı ise 16 olguyu ayırıyor.
        olgu = dosya.stem.removeprefix("augmented_")
        tablo["phenomenon"] = olgu

        # Dosyadaki satır sırasını çift kimliği olarak saklıyoruz; FP16 ile
        # nicemlenmiş modelin aynı çift üzerindeki kararlarını eşleştirmek için.
        tablo["cift_id"] = [f"{olgu}_{i:04d}" for i in range(len(tablo))]

        # İyi ve kötü cümlesi birebir aynı olan çiftler ayırt edici değildir
        # (iki cümlenin olasılığı eşit çıkar); bu çiftleri çıkarıyoruz.
        ayni = tablo["good_sentence"] == tablo["bad_sentence"]
        ayiklanan[olgu] = int(ayni.sum())
        tablo = tablo[~ayni]

        if n is not None and n < len(tablo):
            tablo = tablo.sample(n=n, random_state=tohum)
        parcalar.append(tablo)

    tum = pd.concat(parcalar, ignore_index=True)
    tum = tum.rename(columns={"phenomenon": "olgu",
                              "good_sentence": "iyi",
                              "bad_sentence": "kotu"})
    return tum[["cift_id", "olgu", "iyi", "kotu"]], ayiklanan
