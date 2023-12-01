# app.py
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
        hole_length_ft = float(request.form['hole_length'])
        hole_size = float(request.form['hole_size'])
        washout_factor = float(request.form['washout_factor'])
        solid_waste_volume = float(request.form['solid_waste_volume'])

        # Convert hole length from feet to meters
        hole_length_m = hole_length_ft * 0.3048

        # Make a prediction
        input_data = np.array([[hole_length_m, hole_size, washout_factor, solid_waste_volume]])
        prediction = model.predict(input_data)

        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
