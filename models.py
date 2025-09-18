from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class TrainingData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(64), nullable=False)
    mood = db.Column(db.Float, nullable=False)
    sleep_time = db.Column(db.Float, nullable=False)
    to_sleep_time = db.Column(db.Float, nullable=False)
    training_time = db.Column(db.Float, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    typing_speed = db.Column(db.Float, nullable=False)
    typing_accuracy = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
