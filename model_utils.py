import os
import boto3
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

BUCKET = os.environ.get("S3_BUCKET_NAME")

def get_model_key(user_id):
    return f"models/{user_id}.h5"

def build_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def save_model_to_s3(model, user_id, local_path="model.h5"):
    model.save(local_path)
    if BUCKET:
        s3 = boto3.client("s3")
        s3.upload_file(local_path, BUCKET, get_model_key(user_id))

def load_model_from_s3(user_id, local_path="model.h5"):
    if BUCKET:
        s3 = boto3.client("s3")
        try:
            s3.download_file(BUCKET, get_model_key(user_id), local_path)
            return load_model(local_path)
        except Exception:
            return None
    elif os.path.exists(local_path):
        return load_model(local_path)
    return None


