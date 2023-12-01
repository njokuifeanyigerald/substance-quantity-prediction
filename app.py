# Flask App

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        hole_length = float(request.form['hole_length'])
        diameter = float(request.form['diameter'])
        washout_factor = float(request.form['washout_factor'])

        # Make a prediction
        input_data = np.array([[hole_length, diameter, washout_factor]])
        prediction = model.predict(input_data)

        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
