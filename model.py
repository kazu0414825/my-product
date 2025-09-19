import pandas as pd
from sklearn.linear_model import LinearRegression
import os

CSV_FILE = "data.csv"
CSV_COLUMNS = [
    "timestamp","mood","sleep_time","to_sleep_time",
    "training_time","weight","typing_speed"
]

def train_model():
    if not os.path.exists(CSV_FILE):
        return None
    
    df = pd.read_csv(CSV_FILE)
    if df.empty:
        return None
    
    for col in CSV_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X = df[["sleep_time","to_sleep_time","training_time","weight","typing_speed"]]
    y = df["mood"]

    model = LinearRegression()
    model.fit(X, y)
    return model


