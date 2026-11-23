"""Modeli FP16, INT8 veya NF4 olarak yükleyen ve cümle puanlayan modül."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

VARYANTLAR = ("fp16", "int8", "nf4")


def model_yukle(model_adi, varyant):
    """Hugging Face'teki modeli istenen hassasiyette GPU'ya yükler.

    fp16 : nicemlenmemiş temel çizgi (16 bit kayan nokta)
    int8 : bitsandbytes LLM.int8() ile 8 bit ağırlık nicemleme
    nf4  : bitsandbytes NF4 ile 4 bit ağırlık nicemleme (QLoRA'daki biçim)
    """
    if varyant == "fp16":
        ayar = {}
    elif varyant == "int8":
        ayar = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    elif varyant == "nf4":
        ayar = {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            # Hesaplama 16 bitte yapılır; yalnızca saklanan ağırlıklar 4 bittir.
            bnb_4bit_compute_dtype=torch.float16,
        )}
    else:
        raise ValueError(f"Bilinmeyen varyant: {varyant} (seçenekler: {VARYANTLAR})")

    tokenizer = AutoTokenizer.from_pretrained(model_adi)
    model = AutoModelForCausalLM.from_pretrained(
        model_adi,
        dtype=torch.float16,   # nicemlenmeyen katmanlar (ör. gömme) 16 bit kalır
        device_map="cuda:0",
        **ayar,
    )
    model.eval()  # dropout vb. kapalı; yalnızca çıkarım yapıyoruz
    return model, tokenizer


def baslangic_tokeni(tokenizer):
    """Cümlenin ilk sözcüğünün de puanlanabilmesi için başa konacak token.

    Nedensel bir model ilk token'ı tahmin edemez (öncesinde bağlam yoktur).
    Bu yüzden cümlenin önüne bir başlangıç token'ı ekliyoruz. Llama'da BOS
    token'ı vardır; Qwen'de yoktur, onun yerine dolgu (<|endoftext|>) kullanılır.
    """
    for aday in (tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id):
        if aday is not None:
            return aday
    raise ValueError("Tokenizer'da BOS, PAD veya EOS token'ı yok")


@torch.no_grad()
def cumle_puanla(model, tokenizer, cumleler, batch=16):
    """Her cümlenin toplam log olasılığını döndürür: log P(cümle).

    TurBLiMP makalesindeki gibi tüm cümle boyunca token log olasılıklarını
    topluyoruz. Değer ne kadar büyükse (sıfıra yakınsa) model cümleyi o kadar
    olası buluyor demektir.
    """
    bas = baslangic_tokeni(tokenizer)
    dolgu = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else bas
    puanlar = []

    for i in range(0, len(cumleler), batch):
        parca = cumleler[i:i + batch]
        kodlar = [[bas] + tokenizer(c, add_special_tokens=False)["input_ids"] for c in parca]
        uzunluk = max(len(k) for k in kodlar)

        # Kısa cümleleri sağdan dolguyla uzatıyoruz; maske dolguyu işaretler.
        ids = torch.full((len(kodlar), uzunluk), dolgu, dtype=torch.long)
        maske = torch.zeros((len(kodlar), uzunluk), dtype=torch.long)
        for j, k in enumerate(kodlar):
            ids[j, :len(k)] = torch.tensor(k)
            maske[j, :len(k)] = 1
        ids, maske = ids.to(model.device), maske.to(model.device)

        logits = model(input_ids=ids, attention_mask=maske).logits.float()
        # t konumundaki çıktı, t+1 konumundaki token'ın olasılığını verir.
        log_olas = torch.log_softmax(logits[:, :-1, :], dim=-1)
        hedef = ids[:, 1:]
        token_lp = log_olas.gather(2, hedef.unsqueeze(-1)).squeeze(-1)
        token_lp = token_lp * maske[:, 1:]          # dolgu token'larını sayma
        puanlar.extend(token_lp.sum(dim=1).tolist())

    return puanlar
