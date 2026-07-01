# app.py
# Voyage Analytics - Production Backend API with Real ML Models
# Purpose: Serve ML model predictions via REST API

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# MODEL LOADING (Startup - Load Once)
# ------------------------------------------------------------
print("🔄 Loading ML models into memory...")

# Path to model files
FLIGHT_MODEL_PATH = 'flight_price_model.pkl'
GENDER_MODEL_PATH = 'gender_classifier.pkl'

# Load models using joblib
try:
    flight_model = joblib.load(FLIGHT_MODEL_PATH)
    print(f"✅ Flight price model loaded from {FLIGHT_MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR loading flight model: {e}")
    flight_model = None

try:
    gender_model = joblib.load(GENDER_MODEL_PATH)
    print(f"✅ Gender classifier loaded from {GENDER_MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR loading gender model: {e}")
    gender_model = None

print("✅ Model loading complete!\n")

# Initialize Flask app
app = Flask(__name__)

# ------------------------------------------------------------
# ENDPOINT 1: /predict_flight
# Purpose: Predict flight ticket price based on input features
# ------------------------------------------------------------
@app.route('/predict_flight', methods=['POST'])
def predict_flight():
    """
    Expected JSON Input (API Contract):
    {
        "origin": "NYC",
        "destination": "LAX",
        "date": "2024-06-15",
        "airline": "Delta"
    }
    """
    # Check if model is loaded
    if flight_model is None:
        return jsonify({
            "status": "error",
            "message": "Flight price model not loaded. Check server logs."
        }), 500
    
    try:
        # Get the incoming JSON data from the request
        data = request.get_json()
        
        # Convert JSON to Pandas DataFrame (required for scikit-learn pipeline)
        input_df = pd.DataFrame([data])
        
        # Make prediction using the full pipeline
        prediction = flight_model.predict(input_df)
        
        # Convert NumPy array to native Python float for JSON serialization
        predicted_price = float(prediction[0])
        
        # Return successful response
        response = {
            "status": "success",
            "predicted_price": predicted_price,
            "currency": "USD",
            "model_version": "production_v1",
            "input_received": data
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        # Return detailed error for frontend debugging
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}",
            "error_type": type(e).__name__,
            "input_received": request.get_json()
        }), 400


# ------------------------------------------------------------
# ENDPOINT 2: /classify_gender
# Purpose: Classify gender based on user profile data
# ------------------------------------------------------------
@app.route('/classify_gender', methods=['POST'])
def classify_gender():
    """
    Expected JSON Input (API Contract):
    {
        "name": "Alex",
        "purchase_history": ["handbag", "laptop", "book"],
        "age": 28
    }
    """
    # Check if model is loaded
    if gender_model is None:
        return jsonify({
            "status": "error",
            "message": "Gender classifier model not loaded. Check server logs."
        }), 500
    
    try:
        # Get the incoming JSON data from the request
        data = request.get_json()
        
        # Convert JSON to Pandas DataFrame (required for scikit-learn pipeline)
        input_df = pd.DataFrame([data])
        
        # Make prediction using the full pipeline
        prediction = gender_model.predict(input_df)
        
        # Convert NumPy result to native Python string for JSON serialization
        predicted_gender = str(prediction[0])
        
        # Try to get prediction probabilities (if model supports it)
        try:
            probabilities = gender_model.predict_proba(input_df)
            confidence = float(np.max(probabilities))
        except AttributeError:
            # Model doesn't support predict_proba
            confidence = None
        
        # Return successful response
        response = {
            "status": "success",
            "predicted_gender": predicted_gender,
            "confidence": confidence,
            "model_version": "production_v1",
            "input_received": data
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        # Return detailed error for frontend debugging
        return jsonify({
            "status": "error",
            "message": f"Classification failed: {str(e)}",
            "error_type": type(e).__name__,
            "input_received": request.get_json()
        }), 400


# ------------------------------------------------------------
# ENDPOINT 3: /recommend_hotels
# Purpose: Recommend hotels based on user preferences
# (MOCK - Waiting for Member 2)
# ------------------------------------------------------------
@app.route('/recommend_hotels', methods=['POST'])
def recommend_hotels():
    """
    Expected JSON Input (API Contract):
    {
        "destination": "Paris",
        "budget": 150,
        "preferences": ["wifi", "breakfast", "gym"]
    }
    """
    data = request.get_json()
    
    # Mock response with a list of recommended hotels
    # TODO: Replace with real recommendation model (Member 2)
    mock_response = {
        "status": "success",
        "recommendations": [
            {
                "hotel_name": "Grand Hotel Paris",
                "price_per_night": 145,
                "rating": 4.5,
                "match_score": 0.92
            },
            {
                "hotel_name": "Budget Inn Central",
                "price_per_night": 120,
                "rating": 4.0,
                "match_score": 0.85
            },
            {
                "hotel_name": "Luxury Suites Eiffel",
                "price_per_night": 149,
                "rating": 4.7,
                "match_score": 0.88
            }
        ],
        "model_version": "mock_v1",
        "input_received": data
    }
    
    return jsonify(mock_response), 200


# ------------------------------------------------------------
# HEALTH CHECK ENDPOINT
# Purpose: Let frontend/DevOps check if the API is alive
# ------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple endpoint to verify the API is running
    Usage: Just open http://localhost:8000/health in browser
    """
    models_status = {
        "flight_model": "loaded" if flight_model is not None else "failed",
        "gender_model": "loaded" if gender_model is not None else "failed",
        "hotel_recommender": "mock"
    }
    
    return jsonify({
        "status": "healthy",
        "service": "Voyage Analytics API",
        "version": "1.0.0-production",
        "models": models_status
    }), 200


# ------------------------------------------------------------
# RUN THE APP
# ------------------------------------------------------------
if __name__ == '__main__':
    print("🚀 Starting Voyage Analytics Production API...")
    print("📍 Endpoints available:")
    print("   - POST http://localhost:8000/predict_flight (PRODUCTION)")
    print("   - POST http://localhost:8000/classify_gender (PRODUCTION)")
    print("   - POST http://localhost:8000/recommend_hotels (MOCK)")
    print("   - GET  http://localhost:8000/health")
    print("\n💡 Press CTRL+C to stop the server\n")
    
    # Run on port 8000 (MLflow is already using 5000)
    app.run(host='0.0.0.0', port=8000, debug=True)