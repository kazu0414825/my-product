from flask import Flask, request, render_template, redirect, url_for, make_response
from model_utils import build_model, save_model, load_model, append_to_csv, load_csv
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import random
import os

CSV_FILE = "data.csv"
app = Flask(__name__)

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

# ---------------- ユーザー管理 ----------------
def get_user_id():
    uid = request.cookies.get("user_id")
    if not uid:
        uid = str(np.random.randint(1000000))
    return uid

def save_user_csv(uid, data_dict):
    """ユーザーの入力を CSV に保存"""
    row = {"user_id": uid, "timestamp": datetime.now().isoformat(), **data_dict}
    append_to_csv(row)

def load_user_csv(uid=None):
    """CSV を読み込み、ユーザーIDでフィルタ。空でも安全に返す"""
    df = load_csv()
    if df.empty:
        df = pd.DataFrame(columns=[
            "user_id","timestamp","mood","sleep_time","to_sleep_time",
            "training_time","weight","typing_speed","typing_accuracy"
        ])
    if uid is not None:
        df = df[df["user_id"] == uid]
    if "timestamp" in df.columns and not df.empty:
        df = df.sort_values("timestamp")
    return df

# ---------------- ルーティング ----------------
@app.route('/')
def index():
    resp = make_response(render_template('index.html'))
    uid = get_user_id()
    resp.set_cookie("user_id", uid)
    return resp

@app.route('/question')
def question():
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
    df = load_user_csv(uid)

    dates = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d").tolist() if not df.empty else []
    length = len(dates)
    
    mood_list = df["mood"].tolist() if not df.empty else [None]*length
    sleep_time_list = df["sleep_time"].tolist() if not df.empty else [None]*length
    training_time_list = df["training_time"].tolist() if not df.empty else [None]*length
    weight_list = df["weight"].tolist() if not df.empty else [None]*length
    typing_speed_list = df["typing_speed"].tolist() if not df.empty else [None]*length
    typing_accuracy_list = df["typing_accuracy"].tolist() if not df.empty else [None]*length

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

    # ---------------- mood計算 ----------------
    contribution_sum = 0.0
    for i in range(1, 7):
        val_str = request.form.get(f"q{i}", "0")
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            val = 0.0
        polarity = request.form.get(f"q{i}_polarity", "positive")
        if polarity not in ["positive", "negative"]:
            polarity = "positive"
        contribution_sum += val if polarity == "positive" else -val
    mood = contribution_sum / 6.0

    # ---------------- 睡眠時間計算 ----------------
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

    # ---------------- CSV保存 ----------------
    save_user_csv(uid, {
        "mood": mood,
        "sleep_time": sleep_time,
        "to_sleep_time": to_sleep_time,
        "training_time": training_time,
        "weight": weight,
        "typing_speed": typing_speed,
        "typing_accuracy": typing_accuracy
    })

    # ---------------- モデル学習（データ5件以上） ----------------
    df = load_user_csv()
    if len(df) >= 5:
        X = df[["sleep_time","to_sleep_time","training_time","weight","typing_speed","typing_accuracy"]].to_numpy()
        y = df["mood"].to_numpy()
        model = load_model("global_model") or build_model()
        model.fit(X, y)
        save_model(model, "global_model")

    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        days = int(request.form['days'])
        model = load_model("global_model")
        if model is None:
            return "まだモデルが存在しません。データを入力してください。"

        df = load_user_csv()
        if len(df) < 5:
            return f"データが少なすぎます（{len(df)}件）"

        X = df[["sleep_time","to_sleep_time","training_time","weight","typing_speed","typing_accuracy"]].to_numpy()
        last_features = X[-1].copy()
        predictions = []
        for _ in range(days):
            pred = float(model.predict(last_features.reshape(1, -1))[0])
            predictions.append(pred)
        return render_template("predict.html", prediction=predictions[-1], days=days)

    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)
