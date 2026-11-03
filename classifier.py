"""Klasifikasi status Telco Customer Churn menggunakan LLM (Groq API).
File ini dipakai saat EVALUASI (Modul 2A) dan saat DEPLOY di Streamlit (Modul 2B)."""
import json
import re
import time
import groq

# Peta nama fitur Telco Churn yang akan dibaca oleh LLM
NAMA_FITUR = {
    "tenure": "Lama Berlangganan (Bulan)",
    "MonthlyCharges": "Biaya Bulanan ($)",
    "TotalCharges": "Total Akumulasi Biaya ($)",
    "Contract": "Jenis Kontrak",
    "PaymentMethod": "Metode Pembayaran",
    "InternetService": "Layanan Internet",
    "PaperlessBilling": "Tagihan Tanpa Kertas",
    "gender": "Jenis Kelamin"
}

def load_config(path="config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def format_data(data):
    """Mengubah dictionary data pelanggan menjadi teks satu baris untuk LLM."""
    return "; ".join(f"{nama}: {data[kolom]}" for kolom, nama in NAMA_FITUR.items())

def build_messages(cfg, data):
    msgs = [{"role": "system", "content": cfg["system_prompt"]}]
    for ex in cfg.get("few_shot", []):
        msgs.append({"role": "user", "content": format_data(ex["data"])})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": format_data(data)})
    return msgs

def parse_label(output, labels):
    """Merapikan jawaban model: 'yes.' -> 'Yes'. Di luar daftar label -> 'invalid'."""
    jawab = output.strip().capitalize()
    # Bersihkan tanda baca ekstra jika ada
    jawab = re.sub(r"[^a-zA-Z]", "", jawab)
    peta = {label.lower(): label for label in labels}
    return peta.get(jawab.lower(), "invalid")

def classify(client, cfg, data, max_retry=3):
    for percobaan in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                messages=build_messages(cfg, data),
            )
            return parse_label(resp.choices[0].message.content, cfg["labels"])
        except groq.RateLimitError:
            time.sleep(15 * (percobaan + 1))  # Kena batas kuota: tunggu lalu coba lagi
    return "invalid"