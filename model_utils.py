import os
import boto3
import joblib
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
