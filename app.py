from flask import Flask, request, render_template, redirect, url_for, make_response
from models import db, TrainingData  # ←ここでインポート
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mentalwave.db'
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

# ----------------- トップ -----------------
@app.route('/')
def index():
    resp = make_response(render_template('index.html'))
    uid = get_user_id()
    resp.set_cookie("user_id", uid)
    return resp

# ----------------- 質問ページ -----------------
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

# fluctuationページ
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


# ----------------- フォーム送信 -----------------
@app.route('/form', methods=['POST'])
def form():
    uid = get_user_id()
    # mood計算
    contribution_sum = 0
    for i in range(1, 7):
        val = float(request.form.get(f"q{i}", 0))
        polarity = request.form.get(f"q{i}_polarity", "positive")
        contribution_sum += val if polarity=="positive" else -val
    mood = contribution_sum / 6.0

    # sleep_time
    sleep_time = 0
    sleep_start = request.form.get("sleep_start","")
    wake_time = request.form.get("wake_time","")
    if sleep_start and wake_time:
        t1 = datetime.strptime(sleep_start,"%H:%M")
        t2 = datetime.strptime(wake_time,"%H:%M")
        if t2 <= t1: t2 += timedelta(days=1)
        sleep_time = round((t2-t1).total_seconds()/3600.0,2)

    # to_sleep_time
    to_sleep_time = {"0-15":7.5,"15-30":22.5,"30-60":45,"60+":60}.get(request.form.get("to_sleep_time","0-15"),0)
    training_time = float(request.form.get("training_time",0))
    weight = float(request.form.get("weight",0))
    typing_speed = float(request.form.get("typing_speed",0))
    typing_accuracy = float(request.form.get("typing_accuracy",0))

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

    # 送信後は自動でTOPにリダイレクト
    return redirect(url_for('index'))

# ----------------- 予測ページ -----------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    uid = get_user_id()
    if request.method=="POST":
        days = int(request.form['days'])

        # DBからユーザーのデータ取得
        df = db.session.query(TrainingData).filter_by(user_id=uid).order_by(TrainingData.timestamp).all()
        if len(df)<5:
            return f"データが少なすぎます（{len(df)}件）"

        # 7特徴量のみ
        X = np.array([[d.mood,d.sleep_time,d.to_sleep_time,d.training_time,d.weight,d.typing_speed,d.typing_accuracy] for d in df])
        y = X[:,0]  # moodを予測対象とする例

        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 時系列5ステップ
        time_steps = 5
        X_seq, y_seq = [], []
        for i in range(len(X_scaled)-time_steps):
            X_seq.append(X_scaled[i:i+time_steps])
            y_seq.append(y[i+time_steps])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # LSTMモデル構築・学習
        model = Sequential([
            LSTM(64,input_shape=(time_steps,X.shape[1])),
            Dense(32,activation='relu'),
            Dense(1,activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_seq, y_seq, epochs=30, batch_size=16, verbose=0)

        # 未来予測
        history_window = list(X_scaled[-time_steps:])
        predictions = []
        for _ in range(days):
            x_input = np.expand_dims(np.array(history_window[-time_steps:]),axis=0)
            pred = model.predict(x_input,verbose=0)[0][0]
            predictions.append(pred)
            fake_features = history_window[-1].copy()
            fake_features[0] = pred
            history_window.append(fake_features)

        return render_template("predict.html", prediction=predictions, days=days)

    return render_template("predict.html")
    
# ----------------- 初期化 -----------------
if __name__=="__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
