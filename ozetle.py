"""calistir.py çıktılarını özetler: olgu bazında doğruluk, FP16'ya göre fark, McNemar testi.

Örnek:
    python ozetle.py --cikti results

Üretilenler (results/ klasörü):
    olgu_tablosu.csv     olgu x varyant doğruluk, fark (yüzde puan) ve p değeri
    ozet.md              rapora konabilecek okunur tablo
    sekil_olgu_fark.png  her olguda FP16'ya göre doğruluk farkı
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # ekran olmadan (Colab/sunucu) dosyaya çizmek için
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest

TEMEL = "fp16"


def mcnemar_p(temel_dogru, varyant_dogru):
    """Kesin (binom) McNemar testi.

    Aynı çiftlerde iki modelin kararlarını karşılaştırır. Yalnızca iki modelin
    ayrıştığı çiftler sayılır: b = FP16 doğru / varyant yanlış, c = tersi.
    Fark yoksa b ve c birbirine yakın olmalıdır (b ~ Binom(b+c, 0.5)).
    """
    b = int((temel_dogru & ~varyant_dogru).sum())
    c = int((~temel_dogru & varyant_dogru).sum())
    if b + c == 0:
        return 1.0, b, c
    return binomtest(b, b + c, 0.5).pvalue, b, c


def olgu_tablosu(klasor):
    """ciftler_*.csv dosyalarından olgu x varyant özet tablosu üretir."""
    tablolar = {p.stem.removeprefix("ciftler_"): pd.read_csv(p)
                for p in sorted(Path(klasor).glob("ciftler_*.csv"))}
    if TEMEL not in tablolar:
        raise FileNotFoundError(f"{klasor}/ciftler_{TEMEL}.csv yok; önce FP16 çalıştırılmalı")

    temel = tablolar[TEMEL].set_index("cift_id")
    satirlar = []
    for varyant, tablo in tablolar.items():
        tablo = tablo.set_index("cift_id")
        # Olgu bazında ve en sonda tüm olgular için ("GENEL") birer satır.
        gruplar = list(tablo.groupby("olgu")) + [("GENEL", tablo)]
        for olgu, grup in gruplar:
            t = temel.loc[grup.index, "dogru"].astype(bool)   # aynı çiftler
            v = grup["dogru"].astype(bool)
            p, b, c = mcnemar_p(t.values, v.values)
            satirlar.append({
                "olgu": olgu, "varyant": varyant, "cift": len(grup),
                "dogruluk": v.mean() * 100,
                "fark_yp": (v.mean() - t.mean()) * 100,
                "fp16_dogru_varyant_yanlis": b, "fp16_yanlis_varyant_dogru": c,
                "p": p,
            })
    return pd.DataFrame(satirlar)


def sekil_ciz(ozet, dosya):
    """Her olgu için nicemlenmiş varyantların FP16'ya göre farkını çizer."""
    veri = ozet[(ozet["varyant"] != TEMEL) & (ozet["olgu"] != "GENEL")]
    tablo = veri.pivot(index="olgu", columns="varyant", values="fark_yp").sort_index()
    ax = tablo.plot.barh(figsize=(8, 7), width=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("FP16'ya göre doğruluk farkı (yüzde puan)")
    ax.set_ylabel("")
    ax.set_title("TurBLiMP olgularında nicemleme etkisi")
    plt.tight_layout()
    plt.savefig(dosya, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cikti", default="results")
    a = p.parse_args()
    klasor = Path(a.cikti)

    ozet = olgu_tablosu(klasor)
    ozet.to_csv(klasor / "olgu_tablosu.csv", index=False)
    sekil_ciz(ozet, klasor / "sekil_olgu_fark.png")

    # Okunur tablo: satır olgu, sütunlarda her varyantın doğruluğu ve farkı.
    genis = ozet.pivot(index="olgu", columns="varyant", values="dogruluk").round(1)
    for v in genis.columns:
        if v != TEMEL:
            fark = ozet[ozet["varyant"] == v].set_index("olgu")
            genis[f"{v} fark"] = fark["fark_yp"].round(1)
            genis[f"{v} p"] = fark["p"].map(lambda x: f"{x:.3f}")
    metin = "# Ön deney özeti\n\nDoğruluk (%), FP16'ya göre fark (yüzde puan) ve McNemar p değeri.\n\n"
    metin += genis.to_markdown() + "\n"
    kaynak = klasor / "kaynak.csv"
    if kaynak.exists():
        metin += "\n## Kaynak kullanımı\n\n" + pd.read_csv(kaynak).round(3).to_markdown(index=False) + "\n"
    (klasor / "ozet.md").write_text(metin, encoding="utf-8")
    print(metin)


if __name__ == "__main__":
    main()
