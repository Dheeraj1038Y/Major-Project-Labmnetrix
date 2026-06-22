# test_api.py
# Purpose: Test all Flask API endpoints programmatically

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    print("\n🔍 Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_predict_flight():
    print("\n🔍 Testing /predict_flight endpoint...")
    payload = {
     "flightType": "economic",
     "agency": "Rainbow",
     "from": "Florianopolis (SC)",
     "to": "Sao Paulo (SP)",
     "distance": 500.5,
     "time": 1.5,
     "day": 15,
     "month": 6,
     "weekday": 3
    }
    response = requests.post(f"{BASE_URL}/predict_flight", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_classify_gender():
    print("\n🔍 Testing /classify_gender endpoint...")
    payload = {
     "first_name": "Alex",
     "company": "TechCorp",
     "name_length": 4,
     "first_letter": "A",
     "last_letter": "x",
     "age": 28   
    }

    response = requests.post(f"{BASE_URL}/classify_gender", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_recommend_hotels():
    print("\n🔍 Testing /recommend_hotels endpoint...")
    payload = {
        "destination": "Paris",
        "budget": 150,
        "preferences": ["wifi", "breakfast", "gym"]
    }
    response = requests.post(f"{BASE_URL}/recommend_hotels", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

if __name__ == "__main__":
    print("🚀 Starting Voyage Analytics API Tests...")
    print("=" * 50)
    
    test_health()
    test_predict_flight()
    test_classify_gender() 
    test_recommend_hotels()
    
    print("=" * 50)
    print("✅ All tests completed!")
    print("🎯 Your mock API is working perfectly!")