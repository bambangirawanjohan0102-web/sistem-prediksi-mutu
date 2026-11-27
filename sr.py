import streamlit as st
import pandas as pd
import joblib
from gm import analisis_gemini
st.set_page_config(page_title="prediksi kualitas jagung", page_icon= "🌽", layout="centered")
st.title("🌽cornsmart AI")
st.write("prediksi metode XGBoost.")
df = pd.read_csv("Book.csv", header=0)
nama_input = df.columns[1:5]
nama_output = df.columns[5:9]
model_0 = joblib.load("model_0.pkl")
model_1 = joblib.load("model_1.pkl")
model_2 = joblib.load("model_2.pkl")
model_3 = joblib.load("model_3.pkl")
encoder = joblib.load("label_encoder.pkl")
st.subheader("Masukkan Data jagung")
with st.form("form_prediksi"): 
    kadar_air = st.number_input(str(nama_input[0]), value=0.0)
    suhu = st.number_input(str(nama_input[1]), value=0.0)
    rh = st.number_input(str(nama_input[2]), value=0.0)
    lama_simpan = st.number_input(str(nama_input[3]), value=0.0)
    tombol = st.form_submit_button("prediksi")
if tombol:
    data_baru = pd.DataFrame([[kadar_air, suhu, rh, lama_simpan]], columns=nama_input)
    hasil_0 = model_0.predict(data_baru)[0]
    hasil_1 = model_1.predict(data_baru)[0]
    hasil_2 = model_2.predict(data_baru)[0]
    hasil_3 = model_3.predict(data_baru)[0]
    kategori = encoder.inverse_transform([int(hasil_3)])[0]
    st.subheader("hasil prediksi")
    tabel = pd.DataFrame({"variabel": nama_output, "hasil_prediksi": [round(float(hasil_0), 2), round(float(hasil_1), 2), round(float(hasil_2), 2), kategori]})
    tabel["hasil_prediksi"] = tabel["hasil_prediksi"].astype(str)
    st.table(tabel)
    input_gemini = {"kadar air": kadar_air, "suhu": suhu, "rh": rh, "lama_simpan": lama_simpan}
    hasil_prediksi ={str(nama_output[0]): round(float(hasil_0), 2), 
    str(nama_output[1]): round(float(hasil_1 ), 2),
    str(nama_output[2]): round(float(hasil_2), 2), 
    str(nama_output[3]):str(kategori)}
    st.session_state["input_gemini"] = input_gemini
st.session_state["hasil_prediksi"] = hasil_prediksi
st.subheader("Analisis Berdasarkan Hasil Prediksi")

if st.button("🔍lakukan analisis"):
    with st.spinner("Gemini sedang menganalisis..."):
        try:
            hasil_analisis = analisis_gemini(input_gemini, hasil_prediksi)
            st.markdown(hasil_analisis)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat analisis Gemini: {e}")
