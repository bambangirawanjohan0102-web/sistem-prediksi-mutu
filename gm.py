
import streamlit as st
import time
import random
from google import genai
from google.genai import errors

def koneksi_gemini():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        Client = genai.Client(api_key=api_key)
        return Client

    except Exception as e:
        raise Exception(f"Gagal menghubungkan Gemini: {e}")

def analisis_gemini(data_input, hasil_prediksi):
    try:
        Client = koneksi_gemini()

        prompt = f"""
        Kamu adalah asisten AI ahli teknologi pascapanen
        dan penyimpanan jagung.

        Analisis kualitas jagung berdasarkan data berikut:

        1. Jelaskan hasil prediksi kualitas jagung berdasarkan output XGBoost.
        2. Analisis kadar air, suhu, RH dan lama simpan.
        3. Berikan rekomendasi penyimpanan jagung.
        4. Jelaskan potensi risiko kerusakan, jamur, perubahan mutu,
           peningkatan kadar air dan risiko mikotoksin.
        5. Berikan keputusan penanganan yang sesuai.
        6. Berikan kesimpulan dan tindakan yang sebaiknya dilakukan.

        TUGAS ANALISIS:
        - Jangan mengubah hasil prediksi XGBoost.
        - Jangan membuat data pengukuran baru.
        - Jangan menyatakan jagung aman dari mikotoksin.
        - Jika data tidak cukup, nyatakan keterbatasannya.
        - Berikan jawaban secara sistematis dengan subjudul.

        DATA INPUT JAGUNG:
        - Kadar air: {data_input.get('kadar_air')}
        - Suhu: {data_input.get('suhu')}
        - Kelembapan relatif (RH): {data_input.get('rh')}
        - Lama simpan: {data_input.get('lama simpan')}

        HASIL PREDIKSI XGBOOST:
        {hasil_prediksi}
        """

        max_percobaan = 5

        for percobaan in range(max_percobaan):
            try:
                response = Client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )

                if response.text:
                    return response.text

                raise Exception("Gemini memberikan respons kosong.")

            except errors.APIError as e:
                if e.code not in [408, 429, 500, 502, 503, 504]:
                    return f"Gemini mengalami gangguan: {e}"

                jeda = min(2 ** (percobaan + 1), 30)
                jeda += random.uniform(0, 1)

                print(
                    f"Gemini gagal. Percobaan "
                    f"{percobaan + 1}/{max_percobaan}. "
                    f"Mencoba kembali dalam {jeda:.1f} detik."
                )

                time.sleep(jeda)

        return "Gemini gagal memberikan respons setelah 5 kali percobaan."

    except Exception as e:
        return f"Terjadi kesalahan saat analisis Gemini: {e}"