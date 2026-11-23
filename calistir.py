"""Ön deney: bir modeli FP16, INT8 ve NF4 olarak TurBLiMP üzerinde değerlendirir.

Örnek:
    python calistir.py --model Qwen/Qwen2.5-1.5B-Instruct --n 300

Çıktılar (results/ klasörü):
    ciftler_<varyant>.csv  her minimal çift için iki cümlenin puanı ve doğru/yanlış
    kaynak.csv             her varyant için bellek, tepe bellek, gecikme, süre
    ortam.json             GPU, kütüphane sürümleri, model ve veri bilgisi
"""

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import pandas as pd
import torch
import transformers

from src.model import VARYANTLAR, cumle_puanla, model_yukle
from src.olcum import bellek_gb, gecikme_ms, sozcuk_basina_token, tepe_bellek_gb
from src.veri import turblimp_yukle


def argumanlar():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--varyantlar", nargs="+", default=list(VARYANTLAR), choices=VARYANTLAR)
    p.add_argument("--n", type=int, default=300, help="olgu başına çift sayısı")
    p.add_argument("--tohum", type=int, default=42)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--veri", default="data/turblimp/data/base")
    p.add_argument("--cikti", default="results")
    return p.parse_args()


def surum(paket):
    """Kurulu paketin sürümünü döndürür; kurulu değilse 'yok'."""
    try:
        from importlib.metadata import version
        return version(paket)
    except Exception:
        return "yok"


def main():
    a = argumanlar()
    cikti = Path(a.cikti)
    cikti.mkdir(parents=True, exist_ok=True)

    ciftler, ayiklanan = turblimp_yukle(a.veri, n=a.n, tohum=a.tohum)
    print(f"{len(ciftler)} çift, {ciftler['olgu'].nunique()} olgu yüklendi")
    print("Aynı cümleli olduğu için çıkarılan çiftler:", {k: v for k, v in ayiklanan.items() if v})

    kaynak_satirlari = []
    for varyant in a.varyantlar:
        print(f"\n=== {varyant} ===")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model, tok = model_yukle(a.model, varyant)

        # İyi ve kötü cümleleri ayrı ayrı puanlıyoruz.
        iyi = cumle_puanla(model, tok, ciftler["iyi"].tolist(), batch=a.batch)
        kotu = cumle_puanla(model, tok, ciftler["kotu"].tolist(), batch=a.batch)

        sonuc = ciftler.copy()
        sonuc["lp_iyi"] = iyi
        sonuc["lp_kotu"] = kotu
        # Model dil bilgisine uygun cümleyi daha olası bulduysa çift doğrudur.
        sonuc["dogru"] = sonuc["lp_iyi"] > sonuc["lp_kotu"]
        sonuc.to_csv(cikti / f"ciftler_{varyant}.csv", index=False)
        puanlama_sn = time.perf_counter() - t0

        kaynak_satirlari.append({
            "varyant": varyant,
            "dogruluk": sonuc["dogru"].mean(),
            "model_bellek_gb": bellek_gb(model),
            "gecikme_ms_token": gecikme_ms(model, tok),
            "tepe_bellek_gb": tepe_bellek_gb(),
            "yukleme_puanlama_sn": puanlama_sn,
        })
        print(kaynak_satirlari[-1])

        # Sonraki varyant için GPU belleğini boşalt.
        del model
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(kaynak_satirlari).to_csv(cikti / "kaynak.csv", index=False)

    ortam = {
        "model": a.model,
        "olgu_basina_cift": a.n,
        "tohum": a.tohum,
        "toplam_cift": len(ciftler),
        "ayiklanan_ayni_cumleli_cift": ayiklanan,
        "sozcuk_basina_token": sozcuk_basina_token(tok, ciftler["iyi"].tolist()),
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bitsandbytes": surum("bitsandbytes"),
        "accelerate": surum("accelerate"),
    }
    (cikti / "ortam.json").write_text(json.dumps(ortam, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nOrtam bilgisi:", json.dumps(ortam, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
