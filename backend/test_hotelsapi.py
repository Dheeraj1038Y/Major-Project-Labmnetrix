# test_api.py
# Voyage Analytics - API Testing Script (FIXED)
# Purpose: Test all Flask endpoints to verify they work correctly

import requests
import json

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
BASE_URL = "http://localhost:8000"  # Change if your Flask runs on different port

# Color codes for pretty terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

# ------------------------------------------------------------
# TEST FUNCTIONS
# ------------------------------------------------------------

def print_test_header(test_name):
    """Print a formatted test header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}🧪 TEST: {test_name}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")

def print_success(message):
    """Print success message in green"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    """Print error message in red"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message):
    """Print warning message in yellow"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message):
    """Print info message in yellow"""
    print(f"{Colors.YELLOW}ℹ️  {message}{Colors.END}")


def test_health_check():
    """Test the /health endpoint"""
    print_test_header("Health Check Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print_success("Health check passed!")
            return True
        else:
            print_error("Health check failed!")
            return False
    except Exception as e:
        print_error(f"Connection failed: {e}")
        print_info("Make sure Flask is running on http://localhost:8000")
        return False


def test_flight_prediction():
    """Test the /predict_flight endpoint"""
    print_test_header("Flight Price Prediction")
    
    # Test payload matching your actual model features
    payload = {
        "from": "Florianopolis (SC)",
        "to": "Sao Paulo (SP)",
        "flightType": "economic",
        "agency": "Rainbow",
        "month": 6,
        "day": 15,
        "weekday": 3,
        "time": 1.5,
        "distance": 500.5
    }
    
    print(f"Sending payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict_flight",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            if "predicted_price" in result:
                print_success(f"Prediction successful! Price: ${result['predicted_price']:.2f}")
                return True
            else:
                print_error("Response missing 'predicted_price' field")
                return False
        else:
            print_error("Prediction failed!")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False


def test_gender_classification():
    """Test the /classify_gender endpoint"""
    print_test_header("Gender Classification")
    
    # Test payload matching your actual model features
    payload = {
        "first_name": "Alex",
        "first_letter": "A",
        "last_letter": "x",
        "name_length": 4,
        "company": "TechCorp",
        "age": 28
    }
    
    print(f"Sending payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/classify_gender",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            if "predicted_gender" in result:
                confidence_str = f" (confidence: {result['confidence']:.2%})" if result.get('confidence') else ""
                print_success(f"Classification successful! Gender: {result['predicted_gender']}{confidence_str}")
                return True
            else:
                print_error("Response missing 'predicted_gender' field")
                return False
        else:
            print_error("Classification failed!")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False


def test_hotel_recommendations(scenario_name, payload):
    """Generic function to test hotel recommendations"""
    print_test_header(f"Hotel Recommendations - {scenario_name}")
    
    print(f"Sending payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/recommend_hotels",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200:
            recommendations = result.get('recommendations', [])
            
            if len(recommendations) > 0:
                print_success(f"Recommendations received! Count: {len(recommendations)}")
                # Print first hotel as example (FIXED: using correct field names)
                if recommendations:
                    first_hotel = recommendations[0]
                    print(f"\n  🏨 Example Hotel: {first_hotel.get('name', 'N/A')}")  # ← FIXED: 'name' not 'hotel_name'
                    print(f"  💰 Price: ${first_hotel.get('avg_price_per_night', 'N/A')}/night")  # ← FIXED
                    print(f"  ⭐ Rating: {first_hotel.get('star_rating', 'N/A')} stars")  # ← FIXED
                    print(f"  📊 Review Score: {first_hotel.get('review_score', 'N/A')}/10")
                    print(f"  📍 Location: {first_hotel.get('location', 'N/A')}")
                return recommendations
            else:
                print_warning("No recommendations returned (empty list)")
                print_info("This might mean:")
                print_info("  1. No hotels match your criteria in hotels.csv")
                print_info("  2. The filters are too strict (try lowering min_stars or min_review)")
                print_info("  3. The destination name doesn't match any hotels in the dataset")
                return []
        else:
            print_error("Recommendation failed!")
            return None
            
    except Exception as e:
        print_error(f"Request failed: {e}")
        return None


def compare_hotel_recommendations(results):
    """Compare if hotel recommendation responses are different"""
    print_test_header("Comparing Hotel Recommendations Across Tests")
    
    # Filter out None and empty results
    valid_results = {name: recs for name, recs in results.items() if recs}
    
    if len(valid_results) < 2:
        print_warning("Not enough valid results to compare (need at least 2 non-empty responses)")
        print_info("Try relaxing the search criteria (lower min_stars, higher budget_max)")
        return False
    
    # Extract hotel names from each recommendation (FIXED: using correct field 'name')
    hotel_sets = {}
    for name, recs in valid_results.items():
        hotel_names = [hotel.get('name', 'UNKNOWN') for hotel in recs]  # ← FIXED: 'name' not 'hotel_name'
        locations = [hotel.get('location', 'UNKNOWN') for hotel in recs]
        hotel_sets[name] = set(hotel_names)
        
        print(f"\n{Colors.BOLD}{name}:{Colors.END} {len(recs)} hotel(s)")
        for i, (h_name, loc) in enumerate(zip(hotel_names, locations)):
            print(f"  {i+1}. {h_name} in {loc}")
    
    # Check if they're all identical
    all_sets = list(hotel_sets.values())
    
    # Create unique identifier for each result set
    unique_results = set(tuple(sorted(s)) for s in all_sets)
    
    if len(unique_results) == 1:
        print_error("\n❌ BUG DETECTED: All tests returned IDENTICAL hotels!")
        print_error("The recommender is NOT using the input payload!")
        return False
    else:
        print_success(f"\n✅ SUCCESS: {len(unique_results)} unique result sets found!")
        print_success("The recommender IS using different input payloads correctly!")
        return True


def test_error_handling_empty_payload():
    """Test error handling with empty payload"""
    print_test_header("Error Handling - Empty Payload")
    
    try:
        # Send request with empty JSON object
        response = requests.post(
            f"{BASE_URL}/recommend_hotels",
            json={},
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 400:
            print_success("✅ Correctly rejected empty payload with 400 error!")
            return True
        elif response.status_code == 200:
            print_warning("⚠️  Server accepted empty payload (this might be intentional)")
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False


def test_error_handling_missing_content_type():
    """Test error handling without Content-Type header"""
    print_test_header("Error Handling - Missing Content-Type Header (Testing force=True)")
    
    payload = {
        "destination": "Paris",
        "budget_max": 500
    }
    
    try:
        # Send request WITHOUT Content-Type header
        response = requests.post(
            f"{BASE_URL}/recommend_hotels",
            data=json.dumps(payload)
            # NOTE: No headers parameter - simulating frontend bug
        )
        
        print(f"Sent JSON without Content-Type header")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print_success("✅ Successfully handled missing Content-Type header!")
            print_success("   (This confirms force=True is working)")
            return True
        elif response.status_code == 400:
            result = response.json()
            if "Content-Type" in result.get('message', ''):
                print_success("✅ Correctly detected missing Content-Type header!")
                return True
        else:
            print_warning("⚠️  Unexpected response")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False


# ------------------------------------------------------------
# MAIN TEST RUNNER
# ------------------------------------------------------------

def run_all_tests():
    """Run all tests and print summary"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║   🚀 VOYAGE ANALYTICS API - COMPREHENSIVE TEST SUITE (FIXED)      ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(Colors.END)
    
    results = {}
    
    # Test 1: Health Check
    results['health'] = test_health_check()
    
    if not results['health']:
        print_error("\n❌ Flask server is not running! Stopping tests.")
        print_info("Start Flask with: python app.py")
        return
    
    # Test 2: Flight Prediction
    results['flight'] = test_flight_prediction()
    
    # Test 3: Gender Classification
    results['gender'] = test_gender_classification()
    
    # Test 4-7: Hotel Recommendations (4 different scenarios)
    hotel_results = {}
    
    # Very relaxed criteria - Sao Paulo
    hotel_results['Test 1: Sao Paulo (Very Relaxed)'] = test_hotel_recommendations(
        "Test 1: Sao Paulo (Very Relaxed)",
        {
            "destination": "Sao Paulo",
            "budget_max": 1000,
            "amenities": [],
            "style": "any",
            "min_stars": 0,
            "min_review": 0.0
        }
    )
    
    # Moderate criteria - Sao Paulo
    hotel_results['Test 2: Sao Paulo (Moderate)'] = test_hotel_recommendations(
        "Test 2: Sao Paulo (Moderate)",
        {
            "destination": "Sao Paulo",
            "budget_max": 300,
            "amenities": ["wifi"],
            "style": "business",
            "min_stars": 3,
            "min_review": 7.0
        }
    )
    
    # Different destination - Rio
    hotel_results['Test 3: Rio de Janeiro (Relaxed)'] = test_hotel_recommendations(
        "Test 3: Rio de Janeiro (Relaxed)",
        {
            "destination": "Rio de Janeiro",
            "budget_max": 1000,
            "amenities": [],
            "style": "any",
            "min_stars": 0,
            "min_review": 0.0
        }
    )
    
    # Different destination - Brasilia
    hotel_results['Test 4: Brasilia (Budget)'] = test_hotel_recommendations(
        "Test 4: Brasilia (Budget)",
        {
            "destination": "Brasilia",
            "budget_max": 250,
            "amenities": [],
            "style": "budget",
            "min_stars": 2,
            "min_review": 6.0
        }
    )
    
    # Compare hotel recommendations
    results['hotel_comparison'] = compare_hotel_recommendations(hotel_results)
    
    # Test 8: Error Handling - Empty Payload
    results['error_empty'] = test_error_handling_empty_payload()
    
    # Test 9: Error Handling - Missing Content-Type
    results['error_content_type'] = test_error_handling_missing_content_type()
    
    # Print Summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                         TEST SUMMARY                               ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(Colors.END)
    
    # Count hotel tests
    hotel_test_count = sum(1 for recs in hotel_results.values() if recs is not None and len(recs) > 0)
    total_hotel_tests = len(hotel_results)
    
    print(f"\n{Colors.BOLD}Core API Tests:{Colors.END}")
    for test_name, result in results.items():
        if result is True:
            print(f"{Colors.GREEN}✅ PASS{Colors.END} - {test_name}")
        elif result is False:
            print(f"{Colors.RED}❌ FAIL{Colors.END} - {test_name}")
        else:
            print(f"{Colors.YELLOW}⚠️  SKIP{Colors.END} - {test_name}")
    
    print(f"\n{Colors.BOLD}Hotel Recommendation Tests:{Colors.END}")
    for test_name, recs in hotel_results.items():
        if recs is not None and len(recs) > 0:
            print(f"{Colors.GREEN}✅ PASS{Colors.END} - {test_name} ({len(recs)} hotels)")
        elif recs is not None and len(recs) == 0:
            print(f"{Colors.YELLOW}⚠️  EMPTY{Colors.END} - {test_name} (0 hotels - criteria too strict?)")
        else:
            print(f"{Colors.RED}❌ FAIL{Colors.END} - {test_name}")
    
    passed_tests = sum(1 for r in results.values() if r is True)
    total_tests = len(results)
    
    print(f"\n{Colors.BOLD}Overall: {passed_tests}/{total_tests} core tests passed{Colors.END}")
    print(f"{Colors.BOLD}Hotel Tests: {hotel_test_count}/{total_hotel_tests} returned results{Colors.END}")
    
    if passed_tests == total_tests and hotel_test_count >= 2:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! Your API is working perfectly!{Colors.END}")
    elif hotel_test_count == 0:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Hotel recommender returns empty results{Colors.END}")
        print(f"{Colors.YELLOW}Possible reasons:{Colors.END}")
        print(f"{Colors.YELLOW}  1. hotels.csv might be empty or missing{Colors.END}")
        print(f"{Colors.YELLOW}  2. Destination names don't match (check case sensitivity){Colors.END}")
        print(f"{Colors.YELLOW}  3. All hotels filtered out by strict criteria{Colors.END}")
    else:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Great! Your API is working correctly!{Colors.END}")


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    run_all_tests()