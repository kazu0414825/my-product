from flask import Flask, request, render_template, redirect, url_for
from mental_predict import predict_future
import csv
from datetime import datetime
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/question')
def question():
    return render_template('question.html')
@app.route('/fluctuation')
def fluctuation():
    dates = []
    positive_list = []
    negative_list = []
    sleep_time_list = []
    typing_speed_list = []
    typing_accuracy_list = []

    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app.py のあるディレクトリ
    CSV_PATH = os.path.join(BASE_DIR, "mentalwave_input_data.csv")

    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
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