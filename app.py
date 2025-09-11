from flask import Flask, request, render_template, redirect, url_for
from mental_predict import predict_future
import csv
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/question')
def question():
    return render_template('question.html')

@app.route('/form', methods=['POST'])
def form():
    positive = request.form.get("emotion_positive")
    negative = request.form.get("emotion_negative")
    sleep_time = request.form.get("sleep_time")
    time_to_sleep = request.form.get("time_to_sleep")
    typing_speed = request.form.get("typing_speed")
    typing_accuracy = request.form.get("typing_accuracy")

    # CSVに保存
    with open("mentalwave_input_data.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            positive,
            negative,
            sleep_time,
            time_to_sleep,
            typing_speed,
            typing_accuracy
        ])
    
    return redirect(url_for('thanks'))


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

import csv
from flask import Flask, render_template

@app.route('/fluctuation')
def fluctuation():
    # CSV 読み込み
    dates = []
    positive_list = []
    negative_list = []
    sleep_time_list = []
    typing_speed_list = []
    typing_accuracy_list = []

    try:
        with open("mentalwave_input_data.csv", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 7:  
                    continue
                dates.append(row[0])
                positive_list.append(float(row[1]))
                negative_list.append(float(row[2]))
                sleep_time_list.append(float(row[3]))
                typing_speed_list.append(float(row[5]))
                typing_accuracy_list.append(float(row[6]))
    except FileNotFoundError:
        pass

    return render_template(
        "fluctuation.html",
        dates=dates,
        positive_list=positive_list,
        negative_list=negative_list,
        sleep_time_list=sleep_time_list,
        typing_speed_list=typing_speed_list,
        typing_accuracy_list=typing_accuracy_list
    )

if __name__ == '__main__':
    app.run(debug=True)