from flask import Flask, request, render_template, redirect, url_for
from mental_predict import predict_future

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

    # TODO: CSVに保存したりDBに入れる処理

    
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
if __name__ == '__main__':
    app.run(debug=True)