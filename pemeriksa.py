import pandas as pd
df = pd.read_csv("Book.csv", header=0)
ringkasan = pd.DataFrame({"TIPE DATA": df.dtypes.astype(str), "DATA KOSONG": df.isnull().sum(), "NILAI UNIK": df.nunique()})
print("DATA PEMERIKSAAN DATA")
print(ringkasan.to_string())
print("JUMLAH DATA DUPLIKAT:", df.duplicated().sum())
print("UKURAN DATA:", df.shape)
