from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('marks_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    study = float(request.form['study'])
    sleep = float(request.form['sleep'])
    attendance = float(request.form['attendance'])
    input_data = np.array([[study, sleep, attendance]])
    prediction = model.predict(input_data)[0]
    return render_template('index.html', result=f"Predicted Marks: {prediction:.2f}")

if __name__ == '_main_': 
    app.run(debug=True)


