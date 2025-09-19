from flask import Flask, request, render_template, redirect, url_for
from datetime import datetime, timedelta
import pandas as pd
import random
import os
from model import train_model

app = Flask(__name__)
CSV_FILE = "data.csv"

# ---------------- CSV 設定 ----------------
CSV_COLUMNS = [
    "timestamp","mood","sleep_time","to_sleep_time",
    "training_time","weight","typing_speed"
]

def save_csv(row):
    df_row = pd.DataFrame([row], columns=CSV_COLUMNS)
    if not os.path.exists(CSV_FILE):
        df_row.to_csv(CSV_FILE, index=False)
        print(f"{CSV_FILE} を新規作成しました")
    else:
        df_row.to_csv(CSV_FILE, mode="a", header=False, index=False)
        print(f"{CSV_FILE} に行を追加しました: {row}")

    

CSV_COLUMNS = [
    "timestamp","mood","sleep_time","to_sleep_time",
    "training_time","weight","typing_speed"
]

def load_csv_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        for col in CSV_COLUMNS[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        return df
    else:
        return pd.DataFrame(columns=CSV_COLUMNS)

# ---------------- 質問リスト ----------------
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


@app.route('/')
def index():
    return render_template('index.html')

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

@app.route('/form', methods=['POST'])
def form():
    # ------- mood計算 -------
    contribution_sum = 0.0
    for i in range(1, 7):
        val_str = request.form.get(f"q{i}", "0")
        try:
            val = float(val_str)
        except:
            val = 0.0
        polarity = request.form.get(f"q{i}_polarity", "positive")
        contribution_sum += val if polarity == "positive" else -val
    mood = contribution_sum / 6.0

    # ------ 睡眠時間計算 ----
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

    row = {
        "timestamp": datetime.now().isoformat(),
        "mood": mood,
        "sleep_time": sleep_time,
        "to_sleep_time": to_sleep_time,
        "training_time": training_time,
        "weight": weight,
        "typing_speed": typing_speed
    }
    save_csv(row)

    return redirect(url_for('index'))

@app.route('/fluctuation')
def fluctuation():
    df = load_csv_data()
    if df.empty:
        return "データがまだありません"

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df = df.sort_values("timestamp")
    dates = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    return render_template(
        "fluctuation.html",
        dates=dates,
        mood_list=df["mood"].tolist(),
        sleep_time_list=df["sleep_time"].tolist(),
        training_time_list=df["training_time"].tolist(),
        weight_list=df["weight"].tolist(),
        typing_speed_list=df["typing_speed"].tolist(),
        to_sleep_time_list=df["to_sleep_time"].tolist()
    )
    
from model import train_model

@app.route('/predict')
def predict():
    model = train_model()

    df = load_csv_data()
    latest = df.iloc[-1][["sleep_time","to_sleep_time","training_time","weight","typing_speed"]].values.reshape(1, -1)
    pred = model.predict(latest)[0]

    return render_template("predict.html", prediction=round(pred, 2))


if __name__ == "__main__":
    app.run(debug=True)

