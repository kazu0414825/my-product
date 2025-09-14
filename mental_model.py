from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

def train_and_save(csv_path="mentalwave_input_data.csv"):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :11].values
    y = df.iloc[:, 11].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")  

    def create_sequences(X, y, time_steps=5):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    time_steps = 5
    X_seq, y_seq = create_sequences(X, y, time_steps)

    model = Sequential([
        LSTM(64, input_shape=(time_steps, X.shape[1])),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_seq, y_seq, epochs=50, batch_size=16, verbose=1)

    model.save("mental_model.h5")
    print("✅ 学習済みモデルを保存しました")
