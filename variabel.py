import pandas as pd
df = pd.read_csv("Book.csv", header=0)
X = df.iloc[:, 0:4]
Y = df.iloc[:, 4:8]
print("TABEL VARIABEL X (INDEPENDEN):")
print(X.to_string(index=False))
print("TABEL VARIABEL Y (DEPENDEN):")
print(Y.to_string(index=False))