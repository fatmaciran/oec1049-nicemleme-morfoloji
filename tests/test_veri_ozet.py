"""GPU gerektirmeyen parçaların testleri: veri okuma ve özetleme.

Çalıştırma:  python -m pytest -q
Buradaki puanlar yapaydır; yalnızca kodun doğru hesap yaptığını sınar.
"""

import pandas as pd

from ozetle import mcnemar_p, olgu_tablosu
from src.veri import turblimp_yukle


def test_ayni_cumleli_cift_ayiklanir(tmp_path):
    satirlar = [
        "good_sentence;bad_sentence;good_cue;bad_cue;critical_region;phenomenon",
        "A iyi.;A kötü.;x;y;z;olgu_a",
        "B aynı.;B aynı.;x;y;z;olgu_a",          # iyi == kötü -> çıkarılmalı
        "C iyi.;C kötü.;x;y;z;olgu_a",
    ]
    (tmp_path / "augmented_olgu_a.csv").write_text("\n".join(satirlar), encoding="utf-8")
    tablo, ayiklanan = turblimp_yukle(tmp_path)
    assert ayiklanan == {"olgu_a": 1}
    assert len(tablo) == 2
    assert list(tablo.columns) == ["cift_id", "olgu", "iyi", "kotu"]


def test_ornekleme_tohumla_tekrarlanabilir(tmp_path):
    satirlar = ["good_sentence;bad_sentence;good_cue;bad_cue;critical_region;phenomenon"]
    satirlar += [f"İyi {i}.;Kötü {i}.;x;y;z;olgu_b" for i in range(50)]
    (tmp_path / "augmented_olgu_b.csv").write_text("\n".join(satirlar), encoding="utf-8")
    t1, _ = turblimp_yukle(tmp_path, n=10, tohum=1)
    t2, _ = turblimp_yukle(tmp_path, n=10, tohum=1)
    assert len(t1) == 10 and list(t1["cift_id"]) == list(t2["cift_id"])


def test_mcnemar():
    import numpy as np
    t = np.array([True] * 10 + [False] * 5)
    v = np.array([False] * 10 + [False] * 5)   # 10 çiftte FP16 doğru, varyant yanlış
    p, b, c = mcnemar_p(t, v)
    assert (b, c) == (10, 0) and p < 0.01
    p, b, c = mcnemar_p(t, t)                  # hiç ayrışma yok
    assert (b, c) == (0, 0) and p == 1.0


def test_olgu_tablosu_fark(tmp_path):
    ortak = {"cift_id": ["a1", "a2", "a3", "a4"], "olgu": ["x"] * 4}
    pd.DataFrame({**ortak, "dogru": [True, True, True, False]}).to_csv(tmp_path / "ciftler_fp16.csv", index=False)
    pd.DataFrame({**ortak, "dogru": [True, False, True, False]}).to_csv(tmp_path / "ciftler_nf4.csv", index=False)
    ozet = olgu_tablosu(tmp_path).set_index(["olgu", "varyant"])
    assert ozet.loc[("x", "fp16"), "dogruluk"] == 75.0
    assert ozet.loc[("x", "nf4"), "fark_yp"] == -25.0
    assert ozet.loc[("GENEL", "nf4"), "fp16_dogru_varyant_yanlis"] == 1
