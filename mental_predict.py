from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

# モデルとスケーラーは関数外でロード
model = load_model("mental_model.h5")
scaler = joblib.load("scaler.pkl")

def predict_future(days_ahead, csv_path="mentalwave_input_data.csv"):
    df = pd.read_csv(csv_path).tail(20)
    X = df.iloc[:, :11].values
    X = scaler.transform(X)

    time_steps = 5
    history_window = list(X[-time_steps:])

    predictions = []
    for _ in range(days_ahead):
        x_input = np.array(history_window[-time_steps:])
        x_input = np.expand_dims(x_input, axis=0)  # shape (1, time_steps, features)

        pred = model.predict(x_input, verbose=0)[0][0]
        predictions.append(pred)

        fake_features = history_window[-1].copy()
        fake_features[0] = pred  
        history_window.append(fake_features)

    return predictions
