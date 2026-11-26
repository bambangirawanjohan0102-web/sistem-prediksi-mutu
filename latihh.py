import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
df = pd.read_csv("Book.csv")
df.columns = df.columns.str.strip()
print("nama kolom csv:")
for kolom in df.columns: print(repr(kolom))
X = df[["KA Awal (%)", "Suhu Ruang (°C)", "RH (%)", "Lama Simpan (Hari)"]]
Y_regresi = df[["Biji Rusak (%)", "KA Akhir (%)", "Biji Berjamur (%)"]]
Y_klarifikasi = df["Kategori Mutu"]
encoder = LabelEncoder()
Y_klarifikasi_encoded = encoder.fit_transform(Y_klarifikasi)
X_train, X_test, Y_reg_train, Y_reg_test, Y_klas_train, Y_klas_test = train_test_split(X, Y_regresi, Y_klarifikasi_encoded, test_size=0.2, random_state=42)
model = {}
hasil_latih = []
for kolom in Y_regresi.columns:
    model[kolom] = XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, objective="reg:squarederror", random_state=42)
    model[kolom].fit(X_train, Y_reg_train[kolom])
    hasil_latih.append({"variabel target": kolom, "jenis model": "regression", "data training": len(X_train), "status": "berhasil"})
model["kategori_mutu"] = XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, eval_metric="mlogloss", random_state=42)
model["kategori_mutu"].fit(X_train, Y_klas_train)
hasil_latih.append({"variabel target": "kategori_mutu", "jenis model": "classification", "data training": len(X_train), "status": "berhasil"})
tabel_hasil = pd.DataFrame(hasil_latih)
print("=========================================")
print("TABEL HASIL PELATIHAN XGBOOST")
print("=========================================")
print(tabel_hasil.to_string(index=False))
print("=========================================")
print("UKURAN DATA TRAINING:", X_train.shape)
print("UKURAN DATA TESTING:", X_test.shape)
print("=========================================")
