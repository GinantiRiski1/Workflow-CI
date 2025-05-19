import joblib
import numpy as np
from flask import Flask, request, jsonify

# Load model, scaler, and label encoder
model = joblib.load('model.pkl')  # Trained model
scaler = joblib.load('scaler.pkl')  # Scaler
label_encoder = joblib.load('label_encoder_gender.pkl')  # LabelEncoder for 'Gender'

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Model is ready for inference!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from POST request
        data = request.get_json()

        # Extract fields
        age = data['Age']
        salary = data['AnnualSalary']
        gender = data['Gender']

        # Encode gender
        gender_encoded = label_encoder.transform([gender])[0]

        # Combine into feature array
        features = np.array([[age, salary, gender_encoded]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Ensure type is serializable
        features_scaled = features_scaled.astype(float)

        # Make prediction
        prediction = model.predict(features_scaled)

        # Return result
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    # Run on all interfaces so it can be accessed from Docker
    app.run(host='0.0.0.0', port=5000, debug=True)
