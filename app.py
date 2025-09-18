from flask import Flask, request, render_template, redirect, url_for, make_response
from model_utils import build_model, save_model_to_s3, load_model_from_s3, save_user_csv_to_s3, restore_user_csv_from_s3
from models import db, TrainingData
import numpy as np
import os
from datetime import datetime, timedelta
import pandas as pd

app = Flask(__name__)

# ----------------- データベース設定 -----------------
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")  # SQLAlchemy対応
else:
    DATABASE_URL = "sqlite:////tmp/mentalwave.db"  # ローカル用

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# ----------------- 質問リスト -----------------
positive_questions = [
    "今日は良い一日になると思う",
    "今朝は気分が前向きだ",
    "自分にはやる気がある",
    "他人と交流したい気分だ",
    "今日の仕事（学業）に集中できそうだ",
    "体調が良くエネルギーがある",
    "小さなことでも楽しいと感じる",
    "物事に希望を感じる",
    "心が穏やかで落ち着いている",
    "明るい気持ちで目覚めた"
]
negative_questions = [
    "また退屈な一日になりそうだ",
    "虚しさを感じ、エネルギーがない",
    "やる気が出ない",
    "たいていのことが面倒に感じる",
    "不安や緊張を強く感じている",
    "何事にも興味が持てない",
    "悲しみや落ち込みを感じる",
    "集中するのが難しい",
    "疲れが取れない",
    "自分に自信が持てない"
]

# ----------------- ユーザーID管理 -----------------
def get_user_id():
    uid = request.cookies.get("user_id")
    if not uid:
        uid = str(np.random.randint(1000000))
        restore_user_csv_from_s3(uid)
    return uid

# ----------------- ルーティング -----------------
@app.route('/')
def index():
    resp = make_response(render_template('index.html'))
    uid = get_user_id()
    resp.set_cookie("user_id", uid)
    return resp

@app.route('/question')
def question():
    import random
    pos_sel = random.sample(positive_questions, 3)
    neg_sel = random.sample(negative_questions, 3)
    combined = [{'text': q, 'polarity': 'positive'} for q in pos_sel] + \
               [{'text': q, 'polarity': 'negative'} for q in neg_sel]
    random.shuffle(combined)
    for i, item in enumerate(combined, start=1):
        item['id'] = f"q{i}"
    return render_template('question.html', questions=combined)

@app.route('/fluctuation')
def fluctuation():
    uid = get_user_id()
    data = TrainingData.query.filter_by(user_id=uid).order_by(TrainingData.timestamp).all()

    dates = [d.timestamp.strftime("%Y-%m-%d") for d in data]
    mood_list = [d.mood for d in data]
    sleep_time_list = [d.sleep_time for d in data]
    training_time_list = [d.training_time for d in data]
    weight_list = [d.weight for d in data]
    typing_speed_list = [d.typing_speed for d in data]
    typing_accuracy_list = [d.typing_accuracy for d in data]

    return render_template(
        "fluctuation.html",
        dates=dates,
        mood_list=mood_list,
        sleep_time_list=sleep_time_list,
        training_time_list=training_time_list,
        weight_list=weight_list,
        typing_speed_list=typing_speed_list,
        typing_accuracy_list=typing_accuracy_list
    )

@app.route('/form', methods=['POST'])
def form():
    uid = get_user_id()

    contribution_sum = 0.0
    for i in range(1, 7):
        try:
            val = float(request.form.get(f"q{i}", "0"))
        except:
            val = 0.0
        polarity = request.form.get(f"q{i}_polarity", "positive")
        contribution_sum += val if polarity == "positive" else -val
    mood = contribution_sum / 6.0

    # 睡眠時間計算
    sleep_time = 0.0
    sleep_start = request.form.get("sleep_start", "")
    wake_time = request.form.get("wake_time", "")
    if sleep_start and wake_time:
        try:
            t1 = datetime.strptime(sleep_start, "%H:%M")
            t2 = datetime.strptime(wake_time, "%H:%M")
            if t2 <= t1:
                t2 += timedelta(days=1)
            sleep_time = round((t2 - t1).total_seconds() / 3600.0, 2)
        except:
            sleep_time = 0.0

    # 入眠時間
    to_sleep_map = {"0-15": 7.5, "15-30": 22.5, "30-60": 45.0, "60+": 60.0}
    to_sleep_time = to_sleep_map.get(request.form.get("time_to_sleep", "0-15"), 0.0)

    def _getf(name):
        try:
            return float(request.form.get(name, 0))
        except:
            return 0.0

    training_time = _getf("training_time")
    weight = _getf("weight")
    typing_speed = _getf("typing_speed")
    typing_accuracy = _getf("typing_accuracy")

    # DB 保存
    data = TrainingData(
        user_id=uid,
        mood=mood,
        sleep_time=sleep_time,
        to_sleep_time=to_sleep_time,
        training_time=training_time,
        weight=weight,
        typing_speed=typing_speed,
        typing_accuracy=typing_accuracy
    )
    db.session.add(data)
    db.session.commit()

    # CSV保存
    save_user_csv_to_s3(uid)

    # モデル再学習
    df = TrainingData.query.filter_by(user_id=uid).order_by(TrainingData.timestamp).all()
    if len(df) >= 5:
        X = np.array([[d.sleep_time, d.to_sleep_time, d.training_time, d.weight, d.typing_speed, d.typing_accuracy] for d in df])
        y = np.array([d.mood for d in df])

        model = load_model_from_s3(uid)
        if model is None:
            model = build_model()
        model.fit(X, y)
        save_model_to_s3(model, uid)

    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    uid = get_user_id()
    if request.method == "POST":
        days = int(request.form['days'])
        model = load_model_from_s3(uid)
        if model is None:
            return "まだモデルが存在しません。データを入力してください。"

        df = TrainingData.query.filter_by(user_id=uid).order_by(TrainingData.timestamp).all()
        if len(df) < 5:
            return f"データが少なすぎます（{len(df)}件）"

        X = np.array([[d.sleep_time, d.to_sleep_time, d.training_time, d.weight, d.typing_speed, d.typing_accuracy] for d in df])
        last_features = X[-1].copy()
        predictions = []
        for _ in range(days):
            pred = float(model.predict(last_features.reshape(1, -1))[0])
            predictions.append(pred)
        return render_template("predict.html", prediction=predictions, days=days)

    return render_template("predict.html")

# --- DB初期化 ---
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)
