import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
df = pd.read_csv("Book.csv", header=0)
X = df.iloc[:, 1:5]
Y = df.iloc[:, 5:8]
Y_kategori = df.iloc[:, 8]
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=42)
for kolom in Y.columns:
    model = XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)
    model.fit(X_train, Y_train[kolom])
    prediksi = model.predict(X_test)
    mae = mean_absolute_error(Y_test[kolom], prediksi)
    mse = mean_squared_error(Y_test[kolom], prediksi)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test[kolom], prediksi)
    print("Variabel:", kolom)
    print("nilai sebenarnya:", Y_test[kolom].values)
    print("hasil prediksi:", prediksi)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2:", r2)
