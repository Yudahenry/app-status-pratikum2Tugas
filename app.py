import os
import time
import pandas as pd
import streamlit as st
from groq import Groq
from classifier import NAMA_FITUR, classify, load_config

st.set_page_config(page_title="Telco Churn (LLM)", page_icon="🤖", layout="centered")

# --- OTOMATIS BUAT FILE CONTOH DATA CSV JIKA BELUM ADA ---
if not os.path.exists("contoh_data_baru.csv"):
    sample_data = pd.DataFrame([
        {
            "tenure": 12, "MonthlyCharges": 65.00, "TotalCharges": 800.00,
            "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
            "InternetService": "DSL", "PaperlessBilling": "Yes", "gender": "Female"
        },
        {
            "tenure": 34, "MonthlyCharges": 56.95, "TotalCharges": 1889.50,
            "Contract": "One year", "PaymentMethod": "Mailed check",
            "InternetService": "DSL", "PaperlessBilling": "No", "gender": "Male"
        },
        {
            "tenure": 2, "MonthlyCharges": 70.70, "TotalCharges": 151.65,
            "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic", "PaperlessBilling": "Yes", "gender": "Female"
        },
        {
            "tenure": 45, "MonthlyCharges": 42.30, "TotalCharges": 1840.75,
            "Contract": "One year", "PaymentMethod": "Bank transfer (automatic)",
            "InternetService": "DSL", "PaperlessBilling": "No", "gender": "Male"
        },
        {
            "tenure": 62, "MonthlyCharges": 89.10, "TotalCharges": 5500.00,
            "Contract": "Two year", "PaymentMethod": "Credit card (automatic)",
            "InternetService": "Fiber optic", "PaperlessBilling": "Yes", "gender": "Male"
        }
    ])
    sample_data.to_csv("contoh_data_baru.csv", index=False)

# ---------- 1. Muat konfigurasi hasil evaluasi & siapkan client ----------
cfg = load_config("config.json")
FITUR = list(NAMA_FITUR)

try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY belum diatur. Isi di .streamlit/secrets.toml (lokal) "
             "atau menu Secrets (Streamlit Community Cloud).")
    st.stop()

if not API_KEY.strip():
    st.error("GROQ_API_KEY masih kosong. Isi di .streamlit/secrets.toml (lokal) "
             "atau menu Secrets (Streamlit Community Cloud).")
    st.stop()

@st.cache_resource
def get_client():
    return Groq(api_key=API_KEY)

client = get_client()

@st.cache_data(show_spinner=False)
def prediksi(tenure, monthly_charges, total_charges, contract, payment_method, internet_service, paperless_billing, gender):
    # Hasil disimpan di cache: data yang sama tidak memanggil API dua kali (hemat kuota)
    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "InternetService": internet_service,
        "PaperlessBilling": paperless_billing,
        "gender": gender
    }
    return classify(client, cfg, data)

# ---------- 2. Sidebar: informasi konfigurasi yang dikunci ----------
ev = cfg.get("evaluasi", {})
with st.sidebar:
    st.header("📌 Tentang Model LLM")
    st.write(f"**LLM:** `{cfg['model']}` (Groq)")
    st.write(f"**Few-shot:** {len(cfg.get('few_shot', []))} contoh")
    if "f1_macro" in ev:
        st.metric("F1 Macro LLM (Validasi)", f"{ev['f1_macro']:.4f}")
    if "accuracy" in ev:
        st.metric("Akurasi LLM", f"{ev['accuracy']*100:.2f}%")
    st.caption(f"Dievaluasi pada {ev.get('n', 30)} data uji, "
               f"tanggal {ev.get('tanggal', '2026')}")
    with st.expander("Lihat system prompt"):
        st.code(cfg["system_prompt"], language=None)

st.title("🤖 Prediksi Telco Customer Churn dengan LLM")
st.write("Aplikasi AI Model Service ini menggunakan LLM via Groq API untuk memprediksi apakah pelanggan telekomunikasi akan tetap berlangganan (**No**) atau berhenti berlangganan (**Yes / Churn**).")

tab1, tab2 = st.tabs(["Prediksi Satu Pelanggan", "Prediksi dari File CSV"])

# ---------- 3. Prediksi satu data pelanggan ----------
with tab1:
    st.subheader("Formulir Input Data Pelanggan")
    with st.form("form_prediksi_llm"):
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.number_input("Lama Berlangganan (Tenure dalam Bulan)", min_value=0, max_value=72, value=12)
            monthly_charges = st.number_input("Biaya Bulanan (Monthly Charges dalam $)", min_value=0.0, max_value=200.0, value=65.0)
            total_charges = st.number_input("Total Akumulasi Biaya (Total Charges dalam $)", min_value=0.0, max_value=10000.0, value=800.0)
            gender = st.selectbox("Jenis Kelamin (Gender)", ["Female", "Male"])
            
        with col2:
            contract = st.selectbox("Jenis Kontrak (Contract)", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Metode Pembayaran (Payment Method)", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            internet_service = st.selectbox("Layanan Internet (Internet Service)", ["DSL", "Fiber optic", "No"])
            paperless_billing = st.selectbox("Tagihan Tanpa Kertas (Paperless Billing)", ["Yes", "No"])
            
        kirim = st.form_submit_button("🔍 Jalankan Prediksi LLM")
        
    if kirim:
        with st.spinner("Menghubungi Groq API..."):
            hasil = prediksi(tenure, monthly_charges, total_charges, contract, payment_method, internet_service, paperless_billing, gender)
            
        if hasil == "Yes":
            st.error("⚠️ **Peringatan: Pelanggan Diprediksi CHURN (Berhenti Berlangganan)**")
        elif hasil == "No":
            st.success("✅ **Aman: Pelanggan Diprediksi TETAP AKTIF (No Churn)**")
        else:
            st.warning(f"Respon model: **{hasil}**")

# ---------- 4. Prediksi banyak data dari CSV ----------
with tab2:
    st.subheader("Unggah Data Massal (Format CSV)")
    st.write("Unggah file CSV dengan kolom fitur: " + ", ".join(f"`{c}`" for c in FITUR)
             + " (dibatasi maksimal 30 baris agar hemat kuota API Groq).")
    
    # 📥 Tombol unduh sampel CSV uji coba (Selalu tersedia)
    try:
        with open("contoh_data_baru.csv", "rb") as f:
            st.download_button(
                label="📥 Unduh Dataset / Contoh Format File CSV Uji Coba",
                data=f,
                file_name="contoh_data_baru.csv",
                mime="text/csv",
                help="Klik untuk mengunduh sampel data CSV untuk diunggah ke form di bawah."
            )
    except Exception as e:
        st.warning(f"Gagal memuat file unduhan: {e}")

    st.markdown("---")

    file = st.file_uploader("Pilih file CSV pelanggan", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        st.write("Pratinjau Data yang Diunggah:")
        st.dataframe(df.head())
        
        kurang = [c for c in FITUR if c not in df.columns]
        if kurang:
            st.error(f"Kolom berikut tidak ditemukan pada CSV Anda: {', '.join(kurang)}")
        else:
            df_proses = df.head(30)
            if st.button("🚀 Proses Prediksi Massal LLM"):
                bar = st.progress(0.0)
                hasil_prediksi = []
                for i, (_, b) in enumerate(df_proses.iterrows(), start=1):
                    res = prediksi(
                        int(b["tenure"]),
                        float(b["MonthlyCharges"]),
                        float(b["TotalCharges"]),
                        str(b["Contract"]),
                        str(b["PaymentMethod"]),
                        str(b["InternetService"]),
                        str(b["PaperlessBilling"]),
                        str(b["gender"])
                    )
                    hasil_prediksi.append(res)
                    bar.progress(i / len(df_proses))
                    time.sleep(2)               # Jaga batas request per menit (rate limit)
                    
                df_proses["Prediksi_Churn_LLM"] = hasil_prediksi
                st.success("Prediksi massal berhasil diselesaikan!")
                st.dataframe(df_proses)
                
                st.write("Distribusi Hasil Prediksi:")
                st.write(df_proses["Prediksi_Churn_LLM"].value_counts())
                
                csv_output = df_proses.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh Hasil Prediksi (CSV)",
                    data=csv_output,
                    file_name="hasil_prediksi_churn_llm.csv",
                    mime="text/csv"
                )