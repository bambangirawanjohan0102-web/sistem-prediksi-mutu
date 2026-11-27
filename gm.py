

import time
import random
from google import genai
from google.genai import errors


def koneksi_gemini(api_key):
    if not api_key or not api_key.strip():
        raise ValueError("API Key Gemini belum dimasukkan.")

    return genai.Client(api_key=api_key.strip())


def analisis_gemini(data_input, hasil_prediksi, api_key):
    try:
        client = koneksi_gemini(api_key)

        prompt = f"""
Kamu adalah asisten AI ahli teknologi pascapanen
dan penyimpanan jagung.

Analisis kualitas jagung berdasarkan data berikut.

TUGAS ANALISIS:
1. Jelaskan hasil prediksi kualitas jagung berdasarkan output XGBoost.
2. Analisis kadar air, suhu, RH, dan lama penyimpanan.
3. Berikan rekomendasi penyimpanan jagung.
4. Jelaskan potensi risiko kerusakan, jamur, perubahan mutu,
   peningkatan kadar air, dan risiko mikotoksin.
5. Berikan keputusan penanganan yang sesuai.
6. Berikan kesimpulan dan tindakan yang sebaiknya dilakukan.

KETENTUAN:
- Jangan mengubah hasil prediksi XGBoost.
- Jangan membuat data pengukuran baru.
- Jangan menyatakan jagung aman dari mikotoksin hanya berdasarkan data ini.
- Jelaskan keterbatasan jika data tidak cukup.
- Bedakan hasil prediksi model dan interpretasi AI.
- Gunakan subjudul yang sistematis.

DATA INPUT JAGUNG:
- Kadar air: {data_input.get('kadar_air')}
- Suhu: {data_input.get('suhu')}
- RH: {data_input.get('rh')}
- Lama simpan: {data_input.get('lama_simpan')}

HASIL PREDIKSI XGBOOST:
{hasil_prediksi}
"""

        max_percobaan = 5

        for percobaan in range(max_percobaan):
            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )

                if response.text:
                    return response.text

                raise Exception("Gemini memberikan respons kosong.")

            except errors.APIError as e:
                if e.code not in [408, 429, 500, 502, 503, 504]:
                    return f"Gemini mengalami gangguan: {e}"

                if percobaan == max_percobaan - 1:
                    break

                jeda = min(2 ** (percobaan + 1), 30)
                jeda += random.uniform(0, 1)
                time.sleep(jeda)

        return "Gemini gagal memberikan respons setelah 5 kali percobaan."

    except Exception as e:
        return f"Terjadi kesalahan saat analisis Gemini: {e}"
