"""
============================================================
🔌 ERAS AI Classifier - Python Client SDK
============================================================
This module provides a production-ready Python client SDK to 
interact with the ERAS Emergency Classifier API from other
applications, machines, or microservices.

Usage:
    from eras_client import ERASClient

    # Initialize client (point to your API server)
    client = ERASClient(base_url="http://192.168.1.15:8000")

    # Get server health
    if client.is_healthy():
        # Predict emergency category
        result = client.predict("Fire in the kitchen!")
        print(result.emergency_type)
        print(result.routing.primary_responder)
============================================================
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eras-client")

try:
    import requests
except ImportError:
    raise ImportError(
        "The 'requests' package is required by the ERAS client SDK. "
        "Install it using: pip install requests"
    )


class ERASPrediction:
    """Represents a structured prediction result from the ERAS API."""
    def __init__(self, data: Dict[str, Any]):
        self.text: str = data.get("text", "")
        self.emergency_type: str = data.get("emergency_type", "Other")
        self.confidence: float = data.get("confidence", 0.0)
        self.needs_review: bool = data.get("needs_review", True)
        self.all_scores: Dict[str, float] = data.get("all_scores", {})
        self.timestamp: str = data.get("timestamp", datetime.now().isoformat())
        
        # Parse routing structure
        routing_data = data.get("routing", {})
        self.routing = ERASRouting(routing_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction result back to dictionary representation."""
        return {
            "text": self.text,
            "emergency_type": self.emergency_type,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "all_scores": self.all_scores,
            "routing": self.routing.to_dict(),
            "timestamp": self.timestamp
        }


class ERASRouting:
    """Represents dispatch routing information for the classified emergency."""
    def __init__(self, data: Dict[str, Any]):
        self.primary_responder: str = data.get("primary_responder", "General Dispatch")
        self.requires_ambulance: bool = data.get("requires_ambulance", False)
        self.requires_police: bool = data.get("requires_police", False)
        self.requires_fire: bool = data.get("requires_fire", False)
        self.is_non_emergency: bool = data.get("is_non_emergency", True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_responder": self.primary_responder,
            "requires_ambulance": self.requires_ambulance,
            "requires_police": self.requires_police,
            "requires_fire": self.requires_fire,
            "is_non_emergency": self.is_non_emergency
        }


class ERASClient:
    """SDK client for interacting with the ERAS AI Classifier FastAPI server."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 10):
        """
        Initialize the ERAS API client.
        
        Args:
            base_url: The URL where the FastAPI server is running (e.g., http://192.168.1.15:8000)
            timeout: Default timeout in seconds for API requests
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_healthy(self) -> bool:
        """
        Checks if the API server is online and running.
        
        Returns:
            True if healthy, False otherwise.
        """
        url = f"{self.base_url}/health"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
        except requests.exceptions.RequestException as e:
            logger.warning(f"Health check failed for {url}: {e}")
        return False

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Fetch metadata about the loaded classifier model.
        
        Returns:
            Dictionary containing model configuration details, or None if request fails.
        """
        url = f"{self.base_url}/model/info"
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch model info: {e}")
        return None

    def predict(self, text: str, language: str = "english") -> ERASPrediction:
        """
        Classify a single emergency text report.
        
        Args:
            text: The emergency report text content
            language: The input text language (default: 'english')
            
        Returns:
            An ERASPrediction object containing labels, scores, and routing recommendations.
        """
        url = f"{self.base_url}/predict"
        payload = {
            "text": text,
            "language": language
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                return ERASPrediction(response.json())
            else:
                raise RuntimeError(
                    f"API returned status {response.status_code}: {response.text}"
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Single prediction request failed: {e}")
            raise

    def predict_batch(self, texts: List[str]) -> List[ERASPrediction]:
        """
        Classify multiple emergency text reports in a batch (maximum 50).
        
        Args:
            texts: List of emergency reports to classify
            
        Returns:
            List of ERASPrediction objects.
        """
        if not (1 <= len(texts) <= 50):
            raise ValueError("Batch size must be between 1 and 50 reports.")
            
        url = f"{self.base_url}/predict/batch"
        payload = {
            "texts": texts
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                predictions_data = response.json().get("predictions", [])
                return [ERASPrediction(pred) for pred in predictions_data]
            else:
                raise RuntimeError(
                    f"API batch request returned status {response.status_code}: {response.text}"
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Batch prediction request failed: {e}")
            raise
