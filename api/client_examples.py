"""
============================================================
🔌 ERAS AI Classifier - Client Integration Examples
============================================================
This file provides copy-pasteable examples of how to consume 
the ERAS Emergency Classifier API from other machines, 
servers, or frontend applications.

To call the API from another machine on the same network:
1. Run the server bound to all interfaces:
   uvicorn api.server:app --host 0.0.0.0 --port 8000
2. Replace 'localhost' in the URLs below with the host machine's IP address
   (e.g., 'http://192.168.1.50:8000/predict').
============================================================
"""

import json
import time

# ============================================================
# 🐍 PYTHON CLIENT EXAMPLE (using `requests`)
# ============================================================
# Install dependency: pip install requests
try:
    import requests
except ImportError:
    print("To run the Python examples, please install requests: pip install requests")
    requests = None

# Set the base URL and API Key of your API server.
# Change 'localhost' to the server's IP address if calling from another machine.
API_BASE_URL = "http://localhost:8000"
API_KEY = "eras_secure_api_key_2026"

def test_api_health():
    """Checks if the API is running and the model is loaded."""
    if not requests:
        return
    print("\n[1] Checking API Health...")
    url = f"{API_BASE_URL}/health"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("API is healthy!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"API returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to API at {url}. Make sure the server is running.")

def predict_single_emergency(text: str, language: str = "english"):
    """Classifies a single emergency report."""
    if not requests:
        return
    print(f"\n[2] Classifying Single Report: '{text}'")
    url = f"{API_BASE_URL}/predict"
    payload = {
        "text": text,
        "language": language
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"Prediction: {data['emergency_type']} (Confidence: {data['confidence']:.2%})")
            print(f"Primary Responder: {data['routing']['primary_responder']}")
            print(f"Needs Review? {data['needs_review']}")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

def predict_batch_emergencies(texts: list):
    """Classifies multiple emergency reports in a single request (up to 50)."""
    if not requests:
        return
    print(f"\n[3] Classifying Batch of {len(texts)} Reports...")
    url = f"{API_BASE_URL}/predict/batch"
    payload = {
        "texts": texts
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"Batch completed! Processed: {data['total']}, Flagged for review: {data['flagged_for_review']}")
            for idx, pred in enumerate(data['predictions']):
                print(f"  {idx+1}. Text: '{pred['text'][:30]}...' -> {pred['emergency_type']} ({pred['confidence']:.2%})")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")


# ============================================================
# 🌐 JAVASCRIPT / NODEJS FETCH EXAMPLE
# ============================================================
JS_CODE_SNIPPET = """
// --- JavaScript Fetch Example ---
// Can be run in browser console or Node.js (v18+)

const API_URL = 'http://localhost:8000/predict';

async function classifyEmergency(text) {
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': 'eras_secure_api_key_2026'
            },
            body: JSON.stringify({
                text: text,
                language: 'english'
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Emergency Type:', data.emergency_type);
        console.log('Routing:', data.routing.primary_responder);
        console.log('Full response:', data);
    } catch (error) {
        console.error('Request failed:', error);
    }
}

// Call the function
classifyEmergency("There is a bad accident on the highway, two cars collided.");
"""

# ============================================================
# 💻 CURL COMMAND EXAMPLES (Terminal / Bash)
# ============================================================
CURL_SNIPPET = """
# --- cURL CLI Examples ---

# 1. Health check (public endpoint, no API key required)
curl -X GET http://localhost:8000/health

# 2. Single prediction (API key required)
curl -X POST http://localhost:8000/predict \\
     -H "Content-Type: application/json" \\
     -H "X-API-Key: eras_secure_api_key_2026" \\
     -d '{"text": "House is on fire! Send help!", "language": "english"}'

# 3. Batch prediction (API key required)
curl -X POST http://localhost:8000/predict/batch \\
     -H "Content-Type: application/json" \\
     -H "X-API-Key: eras_secure_api_key_2026" \\
     -d '{"texts": ["Police emergency at the mall", "Patient needs an ambulance"]}'
"""


if __name__ == "__main__":
    print("=" * 60)
    print("ERAS CLIENT INTEGRATION RUNNER")
    print("=" * 60)
    
    # Run Python client demonstration
    test_api_health()
    
    predict_single_emergency(
        text="A fast spreading kitchen fire has filled the room with smoke.",
        language="english"
    )
    
    predict_batch_emergencies([
        "Child fell off a ladder and broke their arm.",
        "A robbery in progress at the corner convenience store.",
        "Just testing the emergency line, nothing to report."
    ])
    
    print("\n" + "=" * 60)
    print("JavaScript / Node.js Snippet:")
    print(JS_CODE_SNIPPET)
    
    print("=" * 60)
    print("cURL CLI Commands:")
    print(CURL_SNIPPET)
    print("=" * 60)
