import joblib
import numpy as np
import time
from flask import Flask, request, jsonify, Response

# Import metrics dari prometheus_exporter.py
from prometheus_exporter import REQUEST_COUNT, REQUEST_ERRORS, INFERENCE_TIME, prometheus_metrics

# Load model, scaler, and label encoder
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder_gender.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Model is ready for inference!"

@app.route('/predict', methods=['POST'])
@INFERENCE_TIME.time()
def predict():
    REQUEST_COUNT.inc()
    try:
        data = request.get_json()

        age = data['Age']
        salary = data['AnnualSalary']
        gender = data['Gender']

        gender_encoded = label_encoder.transform([gender])[0]
        features = np.array([[age, salary, gender_encoded]])
        features_scaled = scaler.transform(features).astype(float)

        prediction = model.predict(features_scaled)
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        REQUEST_ERRORS.inc()
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    return prometheus_metrics()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
