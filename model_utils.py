import os
import csv
import pandas as pd
from datetime import datetime
import joblib
from sklearn.linear_model import LinearRegression

CSV_FILE = "data.csv"

# ---------------- CSV管理 ----------------
def append_to_csv(data_dict):
    """Heroku 上の CSV に追記"""
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "user_id","timestamp","mood","sleep_time","to_sleep_time",
            "training_time","weight","typing_speed","typing_accuracy"
        ])
        if not file_exists:
            writer.writeheader()
        row = {"timestamp": datetime.now().isoformat(), **data_dict}
        writer.writerow(row)

def load_csv():
    """CSV を読み込む"""
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        # 空のDataFrameを返す（timestampカラムも含む）
        return pd.DataFrame(columns=[
            "user_id","timestamp","mood","sleep_time","to_sleep_time",
            "training_time","weight","typing_speed","typing_accuracy"
        ])

# GitHub連携は不要なので無効化
def push_csv_to_github():
    pass

# ---------------- モデル管理 ----------------
MODEL_DIR = "/tmp/models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_model():
    return LinearRegression()

def get_model_path(user_id):
    return os.path.join(MODEL_DIR, f"{user_id}.pkl")

def save_model(model, user_id):
    joblib.dump(model, get_model_path(user_id))

def load_model(user_id):
    path = get_model_path(user_id)
    if os.path.exists(path):
        return joblib.load(path)
    return None
