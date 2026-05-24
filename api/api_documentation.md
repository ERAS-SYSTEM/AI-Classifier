# 🚨 ERAS Emergency Classification API Documentation

Welcome to the **Emergency Response Alert System (ERAS) AI Classifier API** documentation. This document explains how to authenticate, invoke, and integrate the ERAS classification server into other applications, microservices, or frontends.

---

## 🌐 Base URL & Server Deployment

When running the FastAPI server locally, it binds to port `8000` by default. 

| Environment | Base URL | Access Scope |
|:---|:---|:---|
| **Local Machine** | `http://localhost:8000` | Loopback access only |
| **Local Network (LAN)** | `http://<SERVER_IP>:8000` | Devices on the same Wi-Fi/Ethernet network |

> [!TIP]
> To allow other machines on the same network to connect, run the server with the host bound to all interfaces:
> ```bash
> uvicorn api.server:app --host 0.0.0.0 --port 8000
> ```
> Obtain the `<SERVER_IP>` by running `ipconfig` (Windows) or `ifconfig` (Linux/macOS) on the machine hosting the server.

---

## 🔑 Authentication

Protected prediction endpoints require a static API Key sent in the HTTP request headers.

*   **Header Name**: `X-API-Key`
*   **Default Key**: `eras_secure_api_key_2026`

> [!WARNING]
> If the server is deployed to production, change the API key by setting the `ERAS_API_KEY` environment variable on the server host.

---

## 📌 Endpoint Summary

| Method | Endpoint | Authentication | Description |
|:---|:---|:---|:---|
| **`GET`** | [`/health`](#get-health) | None (Public) | Check API server health and device status |
| **`GET`** | [`/model/info`](#get-modelinfo) | None (Public) | Fetch loaded model's configuration & labels |
| **`POST`** | [`/predict`](#post-predict) | **Required** (`X-API-Key`) | Classify a single emergency text |
| **`POST`** | [`/predict/batch`](#post-predictbatch) | **Required** (`X-API-Key`) | Classify a list of emergency texts (max 50) |

---

## 🔍 Endpoint Details

### `GET /health`
Verifies that the FastAPI server is running, the model is successfully loaded in memory, and indicates whether GPU (`cuda`) or CPU (`cpu`) acceleration is in use.

#### Request Example
```bash
curl -X GET http://localhost:8000/health
```

#### Response Example (`200 OK`)
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "timestamp": "2026-05-24T14:10:45.123456"
}
```

---

### `GET /model/info`
Returns information regarding the classification model's metadata, training settings, target labels mapping, and confidence thresholds.

#### Request Example
```bash
curl -X GET http://localhost:8000/model/info
```

#### Response Example (`200 OK`)
```json
{
  "model_name": "distilbert-base-uncased",
  "labels": {
    "0": "Fire Force",
    "1": "Medical/Hospital",
    "2": "Other",
    "3": "Police"
  },
  "num_labels": 4,
  "device": "cpu",
  "confidence_threshold": 0.7,
  "training_config": {
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 2e-05
  }
}
```

---

### `POST /predict`
Classifies a single emergency report text and yields confidence scores for each category along with recommended dispatch routing.

#### Request Headers
```http
Content-Type: application/json
X-API-Key: eras_secure_api_key_2026
```

#### Request Body Schema
```json
{
  "text": "String (1 to 1000 characters). The emergency description.",
  "language": "String (Optional, defaults to 'english'). e.g., 'english', 'amharic', 'afaan_oromo'."
}
```

#### Request Example
```json
{
  "text": "A fast spreading kitchen fire has filled the room with smoke.",
  "language": "english"
}
```

#### Response Example (`200 OK`)
```json
{
  "text": "A fast spreading kitchen fire has filled the room with smoke.",
  "emergency_type": "Fire Force",
  "confidence": 0.9982,
  "needs_review": false,
  "routing": {
    "primary_responder": "Fire Department",
    "requires_ambulance": false,
    "requires_police": false,
    "requires_fire": true,
    "is_non_emergency": false
  },
  "all_scores": {
    "Fire Force": 0.9982,
    "Medical/Hospital": 0.0006,
    "Other": 0.0009,
    "Police": 0.0003
  },
  "timestamp": "2026-05-24T14:12:01.789123"
}
```

> [!NOTE]
> If `needs_review` is `true`, it indicates the classifier's confidence score was below the safety threshold (`0.7`). It is highly recommended to flag these cases for human operator validation.

---

### `POST /predict/batch`
Classifies multiple emergency reports in a single network round-trip. Supports up to 50 reports per batch.

#### Request Headers
```http
Content-Type: application/json
X-API-Key: eras_secure_api_key_2026
```

#### Request Body Schema
```json
{
  "texts": ["List of strings. Each string represents an emergency report."]
}
```

#### Request Example
```json
{
  "texts": [
    "A massive fire broke out in the school kitchen.",
    "My neighbor fell down and is breathing heavily.",
    "Suspicious car parked outside, someone trying to break in."
  ]
}
```

#### Response Example (`200 OK`)
```json
{
  "predictions": [
    {
      "text": "A massive fire broke out in the school kitchen.",
      "emergency_type": "Fire Force",
      "confidence": 0.9991,
      "needs_review": false,
      "routing": {
        "primary_responder": "Fire Department",
        "requires_ambulance": false,
        "requires_police": false,
        "requires_fire": true,
        "is_non_emergency": false
      },
      "all_scores": {
        "Fire Force": 0.9991,
        "Medical/Hospital": 0.0003,
        "Other": 0.0004,
        "Police": 0.0002
      },
      "timestamp": "2026-05-24T14:15:20.111222"
    },
    {
      "text": "My neighbor fell down and is breathing heavily.",
      "emergency_type": "Medical/Hospital",
      "confidence": 0.9985,
      "needs_review": false,
      "routing": {
        "primary_responder": "Ambulance / Hospital",
        "requires_ambulance": true,
        "requires_police": false,
        "requires_fire": false,
        "is_non_emergency": false
      },
      "all_scores": {
        "Fire Force": 0.0004,
        "Medical/Hospital": 0.9985,
        "Other": 0.0008,
        "Police": 0.0003
      },
      "timestamp": "2026-05-24T14:15:20.111222"
    },
    {
      "text": "Suspicious car parked outside, someone trying to break in.",
      "emergency_type": "Police",
      "confidence": 0.9976,
      "needs_review": false,
      "routing": {
        "primary_responder": "Police Department",
        "requires_ambulance": false,
        "requires_police": true,
        "requires_fire": false,
        "is_non_emergency": false
      },
      "all_scores": {
        "Fire Force": 0.0002,
        "Medical/Hospital": 0.0005,
        "Other": 0.0017,
        "Police": 0.9976
      },
      "timestamp": "2026-05-24T14:15:20.111222"
    }
  ],
  "total": 3,
  "flagged_for_review": 0
}
```

---

## 🛠️ Code Integration Examples

````carousel
```python
# --- Python Integration (Requests) ---
import requests
import json

url = "http://localhost:8000/predict"
payload = {
    "text": "Gas leak in the basement",
    "language": "english"
}
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "eras_secure_api_key_2026"
}

response = requests.post(url, json=payload, headers=headers)
if response.status_code == 200:
    data = response.json()
    print(f"Classification: {data['emergency_type']}")
    print(f"Responder: {data['routing']['primary_responder']}")
else:
    print(f"Error {response.status_code}: {response.text}")
```
<!-- slide -->
```python
# --- Python Integration (Using Client SDK) ---
# Import the client SDK class from eras_client.py
from eras_client import ERASClient

client = ERASClient(
    base_url="http://localhost:8000",
    api_key="eras_secure_api_key_2026"
)

if client.is_healthy():
    result = client.predict("Medical alert: patient collapsed")
    print(f"Type: {result.emergency_type}")
    print(f"Requires Ambulance: {result.routing.requires_ambulance}")
```
<!-- slide -->
```javascript
// --- JavaScript (Fetch API) ---
const API_URL = 'http://localhost:8000/predict';

async function classifyReport(text) {
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': 'eras_secure_api_key_2026'
            },
            body: JSON.stringify({ text, language: 'english' })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log(`Routing Category: ${data.emergency_type}`);
        console.log(`Requires Police: ${data.routing.requires_police}`);
    } catch (error) {
        console.error('Request failed:', error);
    }
}

// Call the function
classifyReport("A robbery is taking place right now!");
```
<!-- slide -->
```bash
# --- Command Line (cURL) ---
# Single classification:
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -H "X-API-Key: eras_secure_api_key_2026" \
     -d '{"text": "Car crash on the highway with injuries.", "language": "english"}'

# Batch classification:
curl -X POST http://localhost:8000/predict/batch \
     -H "Content-Type: application/json" \
     -H "X-API-Key: eras_secure_api_key_2026" \
     -d '{"texts": ["Severe chest pain", "Fire at the warehouse"]}'
```
````

---

## ⚠️ Response Status Codes

| Code | Status | Meaning / Resolution |
|:---|:---|:---|
| **`200`** | `OK` | Request succeeded. Predictions included in response. |
| **`400`** | `Bad Request` | Request validation failed (e.g. empty text, text length > 1000, batch size > 50). |
| **`403`** | `Forbidden` | Missing or incorrect `X-API-Key` in request header. |
| **`422`** | `Unprocessable Entity` | Incorrect JSON request payload schema. |
| **`500`** | `Internal Server Error` | Inference engine failure or model weights missing. |
