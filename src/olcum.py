"""Bellek, gecikme ve tokenizer ölçümleri."""

import statistics
import time

import torch

# Gecikme ölçümünde her varyantta aynı istem kullanılır.
GECIKME_ISTEMI = "Türkiye'nin başkenti Ankara'dır ve"


def bellek_gb(model):
    """Modelin ağırlıklarının bellekte kapladığı alan (GB)."""
    return model.get_memory_footprint() / 1024**3


def tepe_bellek_gb():
    """Son sıfırlamadan bu yana GPU'da ayrılan en yüksek bellek (GB)."""
    return torch.cuda.max_memory_allocated() / 1024**3


@torch.no_grad()
def gecikme_ms(model, tokenizer, yeni_token=64, tekrar=3):
    """Üretilen token başına ortanca süre (milisaniye).

    Önce kısa bir ısınma üretimi yapılır (ilk çağrıdaki CUDA hazırlık süresi
    ölçüme karışmasın diye). Sonra tam olarak `yeni_token` token, açgözlü
    (greedy) kod çözme ile `tekrar` kez üretilir ve ortanca alınır.
    """
    girdi = tokenizer(GECIKME_ISTEMI, return_tensors="pt").to(model.device)
    dolgu = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ortak = {"do_sample": False, "pad_token_id": dolgu}

    model.generate(**girdi, max_new_tokens=8, **ortak)  # ısınma
    sureler = []
    for _ in range(tekrar):
        torch.cuda.synchronize()          # GPU'daki önceki işler bitsin
        t0 = time.perf_counter()
        model.generate(**girdi, max_new_tokens=yeni_token,
                       min_new_tokens=yeni_token, **ortak)
        torch.cuda.synchronize()
        sureler.append((time.perf_counter() - t0) / yeni_token * 1000)
    return statistics.median(sureler)


def sozcuk_basina_token(tokenizer, cumleler):
    """Ortalama sözcük başına token sayısı (tokenizer'ın Türkçeyi ne kadar böldüğü)."""
    token = sum(len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in cumleler)
    sozcuk = sum(len(c.split()) for c in cumleler)
    return token / sozcuk
