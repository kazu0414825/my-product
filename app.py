from flask import Flask, request, render_template, redirect, url_for
from mental_predict import predict_future
import csv
from datetime import datetime,timedelta
import os
import random

app = Flask(__name__)

# --- 質問リスト（例：日本語で10個ずつ） ---
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "mentalwave_input_data.csv")

# -------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


# /question: ランダムに3 positive + 3 negative を選び、順番をシャッフルして渡す
@app.route('/question')
def question():
    pos_sel = random.sample(positive_questions, 3)
    neg_sel = random.sample(negative_questions, 3)

    combined = []
    for q in pos_sel:
        combined.append({'text': q, 'polarity': 'positive'})
    for q in neg_sel:
        combined.append({'text': q, 'polarity': 'negative'})

    random.shuffle(combined)

    # id を付けて q1..q6 にする
    for i, item in enumerate(combined, start=1):
        item['id'] = f"q{i}"

    return render_template('question.html', questions=combined)


@app.route('/thanks')
def thanks():
    return render_template('thanks.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        days = int(request.form['days'])
        prediction = predict_future(days)
        return render_template('predict.html', prediction=prediction, days=days)
    return render_template('predict.html')


@app.route('/fluctuation')
def fluctuation():
    dates = []
    mood_list = []
    sleep_time_list = []
    to_sleep_time_list = []
    training_time_list = []
    weight_list = []
    typing_speed_list = []
    typing_accuracy_list = []

    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:  # datetime + 5項目
                    continue
                dates.append(row[0])
                mood_list.append(float(row[1]) if row[1] else None)
                sleep_time_list.append(float(row[2]) if row[2] else None)
                to_sleep_time_list.append(float(row[3]) if row[3] else None)
                training_time_list.append(float(row[4]) if row[4] else None)
                weight_list.append(float(row[5]) if row[5] else None)
                typing_speed_list.append(float(row["typing_speed"]) if row["typing_speed"] else 0.0)
                typing_accuracy_list.append(float(row["typing_accuracy"]) if row["typing_accuracy"] else 0.0)
    except FileNotFoundError:
        pass

    return render_template(
        "fluctuation.html",
        dates=dates,
        mood_list=mood_list,
        sleep_time_list=sleep_time_list,
        to_sleep_time_list=to_sleep_time_list,
        training_time_list=training_time_list,
        weight_list=weight_list,
        typing_speed_list=typing_speed_list,
        typing_accuracy_list=typing_accuracy_list
    )


# フォームから送られてくる q1..q6 と 他の質問を受け取り CSV に保存する
@app.route('/form', methods=['POST'])
def form():
    # --- mood の計算 ---
    contribution_sum = 0.0
    for i in range(1, 7):
        val_str = request.form.get(f"q{i}", "0")
        try:
            val = float(val_str)
        except:
            val = 0.0
        polarity = request.form.get(f"q{i}_polarity", "positive")
        if polarity == "positive":
            contribution_sum += val
        else:
            contribution_sum -= val
    mood = contribution_sum / 6.0

    # --- Q7, Q8 から睡眠時間を計算 ---
    sleep_start = request.form.get("sleep_start", "")  # HH:MM
    wake_time = request.form.get("wake_time", "")      # HH:MM
    sleep_time = ""
    if sleep_start and wake_time:
        try:
            t1 = datetime.strptime(sleep_start, "%H:%M")
            t2 = datetime.strptime(wake_time, "%H:%M")
            if t2 <= t1:
                t2 += timedelta(days=1)
            diff = (t2 - t1).total_seconds() / 3600.0
            sleep_time = round(diff, 2)
        except:
            sleep_time = ""

    # --- Q9 入眠までの時間 ---
    to_sleep_time = request.form.get("to_sleep_time", "")

    # --- Q10 トレーニング時間（分） ---
    training_time = request.form.get("training_time", "")

    # --- Q11 今日の体重 ---
    weight = request.form.get("weight", "")
    
     # --- タイピング ---
    typing_speed = request.form.get("typing_speed", "")
    typing_accuracy = request.form.get("typing_accuracy", "")

    # --- CSV に保存 ---
    write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "datetime",
                "mood",
                "sleep_time",
                "to_sleep_time",
                "training_time",
                "weight",
                "typing_speed",
                "typing_accuracy"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mood,
            sleep_time,
            to_sleep_time,
            training_time,
            weight,
            typing_speed,
            typing_accuracy
        ])

    return redirect(url_for('thanks'))


if __name__ == '__main__':
    app.run(debug=True)
