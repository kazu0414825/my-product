import os
import boto3
import joblib
import pandas as pd
from models import db, TrainingData
from sklearn.linear_model import LinearRegression

BUCKET = os.environ.get("S3_BUCKET_NAME")

def get_model_key(user_id):
    return f"models/{user_id}.pkl"

def build_model():
    return LinearRegression()

def save_model_to_s3(model, user_id, local_path="model.pkl"):
    joblib.dump(model, local_path)
    if BUCKET:
        s3 = boto3.client("s3")
        s3.upload_file(local_path, BUCKET, get_model_key(user_id))

def load_model_from_s3(user_id, local_path="model.pkl"):
    if BUCKET:
        s3 = boto3.client("s3")
        try:
            s3.download_file(BUCKET, get_model_key(user_id), local_path)
            return joblib.load(local_path)
        except Exception:
            return None
    elif os.path.exists(local_path):
        return joblib.load(local_path)
    return None


BUCKET = os.environ.get("S3_BUCKET_NAME")

def get_user_csv_key(user_id):
    return f"user_data/{user_id}.csv"

def save_user_csv_to_s3(user_id):
    df = pd.DataFrame([
        {
            "mood": d.mood,
            "sleep_time": d.sleep_time,
            "to_sleep_time": d.to_sleep_time,
            "training_time": d.training_time,
            "weight": d.weight,
            "typing_speed": d.typing_speed,
            "typing_accuracy": d.typing_accuracy,
            "timestamp": d.timestamp
        } for d in TrainingData.query.filter_by(user_id=user_id).all()
    ])
    if df.empty:
        return
    tmp_file = f"/tmp/{user_id}.csv"
    df.to_csv(tmp_file, index=False)
    if BUCKET:
        s3 = boto3.client("s3")
        s3.upload_file(tmp_file, BUCKET, get_user_csv_key(user_id))

def restore_user_csv_from_s3(user_id):
    if not BUCKET:
        return
    tmp_file = f"/tmp/{user_id}.csv"
    s3 = boto3.client("s3")
    try:
        s3.download_file(BUCKET, get_user_csv_key(user_id), tmp_file)
        df = pd.read_csv(tmp_file)
        for _, row in df.iterrows():
            exists = TrainingData.query.filter_by(user_id=user_id, timestamp=row["timestamp"]).first()
            if exists:
                continue
            data = TrainingData(
                user_id=user_id,
                mood=row["mood"],
                sleep_time=row["sleep_time"],
                to_sleep_time=row["to_sleep_time"],
                training_time=row["training_time"],
                weight=row["weight"],
                typing_speed=row["typing_speed"],
                typing_accuracy=row["typing_accuracy"],
                timestamp=row["timestamp"]
            )
            db.session.add(data)
        db.session.commit()
    except Exception:
        pass

