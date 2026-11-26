import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("Book.csv", header=0)
X = df.iloc[:, 0:4]
Y = df.iloc[:, 4:8]
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=42)
print("TABEL X TRAINING")
print(X_train.to_string(index=False))
print("TABEL Y TRAINING:")
print(Y_train.to_string(index=False))
print("TABEL X Testing")
print(X_test.to_string(index=False))
print("TABEL Y TESTING:")
print(Y_test.to_string(index=False))
print("UKURAN DATA TRAINING:", X_train.shape)
print("UKURAN DATA TESTING:", X_test.shape)