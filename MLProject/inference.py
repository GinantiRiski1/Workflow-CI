import joblib
import numpy as np
import time
import psutil
from flask import Flask, request, jsonify, Response

# Import Prometheus metrics
from prometheus_exporter import (
    REQUEST_COUNT, REQUEST_ERRORS, INFERENCE_TIME, prometheus_metrics,
    IN_FLIGHT_REQUESTS, LATENCY_SUMMARY, CPU_USAGE_PERCENT, MEMORY_USAGE_PERCENT,
    REQUESTS_BY_USER_AGENT, REQUEST_PAYLOAD_SIZE, RESPONSE_PAYLOAD_SIZE
)

# Load model, scaler, and label encoder
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder_gender.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Model is ready for inference!"

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    IN_FLIGHT_REQUESTS.inc()
    
    # Hitung payload size request body
    request_size = request.content_length or 0
    REQUEST_PAYLOAD_SIZE.observe(request_size)

    # Ambil user-agent dari header
    user_agent = request.headers.get('User-Agent', 'unknown')
    REQUESTS_BY_USER_AGENT.labels(user_agent=user_agent).inc()

    REQUEST_COUNT.labels(endpoint='/predict', method='POST').inc()

    try:
        data = request.get_json()

        age = data['Age']
        salary = data['AnnualSalary']
        gender = data['Gender']

        gender_encoded = label_encoder.transform([gender])[0]
        features = np.array([[age, salary, gender_encoded]])
        features_scaled = scaler.transform(features).astype(float)

        # Measure inference time
        with INFERENCE_TIME.time():
            prediction = model.predict(features_scaled)

        response = jsonify({'prediction': int(prediction[0])})

        # Hitung payload size response body
        response_data = response.get_data()
        RESPONSE_PAYLOAD_SIZE.observe(len(response_data))

        latency = time.time() - start_time
        LATENCY_SUMMARY.labels(endpoint='/predict').observe(latency)

        # Update CPU and memory usage
        CPU_USAGE_PERCENT.set(psutil.cpu_percent())
        MEMORY_USAGE_PERCENT.set(psutil.virtual_memory().percent)

        IN_FLIGHT_REQUESTS.dec()

        return response

    except Exception as e:
        REQUEST_ERRORS.labels(endpoint='/predict').inc()
        IN_FLIGHT_REQUESTS.dec()
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    return prometheus_metrics()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
