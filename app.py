from flask import Flask, request, render_template, redirect, url_for, make_response
from model_utils import build_model, save_model_to_s3, load_model_from_s3
from models import db, TrainingData
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# データベース設定
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")
else:
    DATABASE_URL = "sqlite:///mentalwave.db"

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
    data = TrainingData.query.order_by(TrainingData.timestamp).all()
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
    
    contribution_sum = 0
    for i in range(1, 7):  # 質問数に応じて変更
        val = float(request.form.get(f"q{i}", 0))
        polarity = request.form.get(f"q{i}_polarity", "positive")
        contribution_sum += val if polarity == "positive" else -val
    mood = contribution_sum / 6.0
    
    sleep_start = request.form.get("sleep_start", "")
    wake_time = request.form.get("wake_time", "")
    sleep_time = 0
    if sleep_start and wake_time:
        t1 = datetime.strptime(sleep_start, "%H:%M")
        t2 = datetime.strptime(wake_time, "%H:%M")
        if t2 <= t1:
            t2 += timedelta(days=1)
        sleep_time = round((t2 - t1).total_seconds() / 3600.0, 2)

    mood = float(request.form.get("mood", 0))
    sleep_time = float(request.form.get("sleep_time", 0))
    to_sleep_time = float(request.form.get("to_sleep_time", 0))
    training_time = float(request.form.get("training_time", 0))
    weight = float(request.form.get("weight", 0))
    typing_speed = float(request.form.get("typing_speed", 0))
    typing_accuracy = float(request.form.get("typing_accuracy", 0))

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


    # --- ユーザーごとに線形回帰モデルを再学習 ---
    df = TrainingData.query.filter_by(user_id=uid).order_by(TrainingData.timestamp).all()
    if len(df) >= 5:
        X = np.array([[d.sleep_time,d.to_sleep_time,d.training_time,d.weight,d.typing_speed,d.typing_accuracy] for d in df])
        y = np.array([d.mood for d in df])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = load_model_from_s3(uid)
        if model is None:
            model = build_model()

        model.fit(X_scaled, y)
        save_model_to_s3(model, uid)

    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    uid = get_user_id()
    if request.method=="POST":
        days = int(request.form['days'])
        model = load_model_from_s3(uid)
        if model is None:
            return "まだモデルが存在しません。データを入力してください。"

        df = TrainingData.query.filter_by(user_id=uid).order_by(TrainingData.timestamp).all()
        if len(df) < 5:
            return f"データが少なすぎます（{len(df)}件）"

        X = np.array([[d.sleep_time,d.to_sleep_time,d.training_time,d.weight,d.typing_speed,d.typing_accuracy] for d in df])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        last_features = X_scaled[-1]
        predictions = []
        for _ in range(days):
            pred = model.predict(last_features.reshape(1, -1))[0]
            predictions.append(pred)
            last_features[0] = pred  
        return render_template("predict.html", prediction=predictions, days=days)

    return render_template("predict.html")


with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        print(f"DB init skipped: {e}")

# ----------------- 初期化 -----------------
if __name__=="__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
