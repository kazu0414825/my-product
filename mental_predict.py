from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

def predict_future(days_ahead, csv_path="mentalwave_input_data.csv"):
    model = load_model("mental_model.h5")
    scaler = joblib.load("scaler.pkl")

    df = pd.read_csv(csv_path)
    X = df.iloc[:, :11].values
    X = scaler.transform(X)

    time_steps = 5
    history_window = list(X[-time_steps:])

    prediction = None
    for _ in range(days_ahead):
        x_input = np.array(history_window[-time_steps:])
        x_input = np.expand_dims(x_input, axis=0)
        prediction = model.predict(x_input, verbose=0)[0][0]
        fake_features = history_window[-1].copy()
        fake_features[0] = prediction  
        history_window.append(fake_features)

    return prediction
