import os
import boto3
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

BUCKET = os.environ.get("S3_BUCKET_NAME")

def get_model_key(user_id):
    return f"models/{user_id}.pkl"

def get_csv_key():
    return "user_data/data.csv"

def build_model():
    return LinearRegression()

# ---------------- モデル管理 ----------------
def save_model_to_s3(model, user_id, local_path="/tmp/model.pkl"):
    joblib.dump(model, local_path)
    if BUCKET:
        s3 = boto3.client("s3",
                          aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                          aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
        s3.upload_file(local_path, BUCKET, get_model_key(user_id))

def load_model_from_s3(user_id, local_path="/tmp/model.pkl"):
    if BUCKET:
        s3 = boto3.client("s3",
                          aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                          aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
        try:
            s3.download_file(BUCKET, get_model_key(user_id), local_path)
            return joblib.load(local_path)
        except:
            return None
    elif os.path.exists(local_path):
        return joblib.load(local_path)
    return None

# ---------------- CSV管理 ----------------
def save_csv_to_s3(local_path="data.csv"):
    if BUCKET and os.path.exists(local_path):
        s3 = boto3.client("s3",
                          aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                          aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
        s3.upload_file(local_path, BUCKET, get_csv_key())

def load_csv_from_s3(local_path="data.csv"):
    if BUCKET:
        s3 = boto3.client("s3",
                          aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                          aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
        try:
            s3.download_file(BUCKET, get_csv_key(), local_path)
            return pd.read_csv(local_path)
        except:
            return pd.DataFrame(columns=[
                "user_id","timestamp","mood","sleep_time","to_sleep_time",
                "training_time","weight","typing_speed","typing_accuracy"
            ])
    elif os.path.exists(local_path):
        return pd.read_csv(local_path)
    else:
        return pd.DataFrame(columns=[
            "user_id","timestamp","mood","sleep_time","to_sleep_time",
            "training_time","weight","typing_speed","typing_accuracy"
        ])

