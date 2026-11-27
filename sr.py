

import streamlit as st
import pandas as pd
import joblib
from gm import analisis_gemini

st.set_page_config(
    page_title="Prediksi Kualitas Jagung",
    page_icon="🌽",
    layout="centered"
)

st.title("🌽 Cornsmart AI")
st.write("Prediksi kualitas jagung menggunakan metode XGBoost.")

df = pd.read_csv("Book.csv", header=0)
nama_input = df.columns[1:5]
nama_output = df.columns[5:9]

model_0 = joblib.load("model_0.pkl")
model_1 = joblib.load("model_1.pkl")
model_2 = joblib.load("model_2.pkl")
model_3 = joblib.load("model_3.pkl")
encoder = joblib.load("label_encoder.pkl")

st.subheader("Masukkan Data Jagung")

with st.form("form_prediksi"):
    kadar_air = st.number_input(str(nama_input[0]), value=0.0)
    suhu = st.number_input(str(nama_input[1]), value=0.0)
    rh = st.number_input(str(nama_input[2]), value=0.0)
    lama_simpan = st.number_input(str(nama_input[3]), value=0.0)

    tombol = st.form_submit_button("Prediksi")

if tombol:
    data_baru = pd.DataFrame(
        [[kadar_air, suhu, rh, lama_simpan]],
        columns=nama_input
    )

    hasil_0 = model_0.predict(data_baru)[0]
    hasil_1 = model_1.predict(data_baru)[0]
    hasil_2 = model_2.predict(data_baru)[0]
    hasil_3 = model_3.predict(data_baru)[0]

    kategori = encoder.inverse_transform([int(hasil_3)])[0]

    st.session_state["input_gemini"] = {
        "kadar_air": kadar_air,
        "suhu": suhu,
        "rh": rh,
        "lama_simpan": lama_simpan
    }

    st.session_state["hasil_prediksi"] = {
        str(nama_output[0]): round(float(hasil_0), 2),
        str(nama_output[1]): round(float(hasil_1), 2),
        str(nama_output[2]): round(float(hasil_2), 2),
        str(nama_output[3]): str(kategori)
    }

    st.session_state["prediksi_selesai"] = True


if st.session_state.get("prediksi_selesai", False):

    st.subheader("Hasil Prediksi")

    hasil = st.session_state["hasil_prediksi"]

    tabel = pd.DataFrame({
        "Variabel": list(hasil.keys()),
        "Hasil Prediksi": [str(v) for v in hasil.values()]
    })

    st.table(tabel)

    st.divider()

    st.subheader("Analisis Menggunakan Gemini")

    st.write(
        "Masukkan API Key Gemini milik Anda "
        "untuk melakukan analisis hasil prediksi."
    )

    api_key = st.text_input(
        "API Key Gemini",
        type="password",
        placeholder="Masukkan API Key Gemini"
    )

    if st.button("🔍 Lakukan Analisis"):

        if not api_key.strip():
            st.warning("Silakan masukkan API Key Gemini terlebih dahulu.")

        else:
            with st.spinner("Sedang melakukan analisis..."):

                hasil_analisis = analisis_gemini(
                    st.session_state["input_gemini"],
                    st.session_state["hasil_prediksi"],
                    api_key
                )

                st.subheader("Hasil Analisis Gemini")
                st.markdown(hasil_analisis)

else:
    st.info("Silakan masukkan data jagung dan lakukan prediksi terlebih dahulu.")
