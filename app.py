# app.py
# Voyage Analytics - Mock Backend API
# Purpose: Provide mock endpoints for ML model predictions until real models are ready

from flask import Flask, request, jsonify

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
    # Get the incoming JSON data from the request
    data = request.get_json()
    
    # For now, return a mock (hardcoded) response
    # TODO: Replace this with real model prediction when model is ready
    mock_response = {
        "status": "success",
        "predicted_price": 450.50,
        "currency": "USD",
        "model_version": "mock_v1",
        "input_received": data  # Echo back what was sent (helpful for debugging)
    }
    
    return jsonify(mock_response), 200


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
    data = request.get_json()
    
    # Mock response
    # TODO: Replace with real gender classification model
    mock_response = {
        "status": "success",
        "predicted_gender": "Female",
        "confidence": 0.87,
        "model_version": "mock_v1",
        "input_received": data
    }
    
    return jsonify(mock_response), 200


# ------------------------------------------------------------
# ENDPOINT 3: /recommend_hotels
# Purpose: Recommend hotels based on user preferences
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
    # TODO: Replace with real recommendation model
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
# HEALTH CHECK ENDPOINT (Bonus!)
# Purpose: Let frontend/DevOps check if the API is alive
# ------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple endpoint to verify the API is running
    Usage: Just open http://localhost:8000/health in browser
    """
    return jsonify({
        "status": "healthy",
        "service": "Voyage Analytics API",
        "version": "1.0.0-mock"
    }), 200


# ------------------------------------------------------------
# RUN THE APP
# ------------------------------------------------------------
if __name__ == '__main__':
    print("🚀 Starting Voyage Analytics Mock API...")
    print("📍 Endpoints available:")
    print("   - POST http://localhost:8000/predict_flight")
    print("   - POST http://localhost:8000/classify_gender")
    print("   - POST http://localhost:8000/recommend_hotels")
    print("   - GET  http://localhost:8000/health")
    print("\n💡 Press CTRL+C to stop the server\n")
    
    # Run on port 8000 (MLflow is already using 5000)
    app.run(host='0.0.0.0', port=8000, debug=True)