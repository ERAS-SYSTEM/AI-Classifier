"""
============================================
🚀 ERAS - Emergency Classification API
============================================
FastAPI server that serves the trained DistilBERT
model for real-time emergency classification.

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict            → Classify a single emergency text
    POST /predict/batch      → Classify multiple texts at once
    GET  /health             → Health check
    GET  /model/info         → Model metadata & label info
============================================
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ============================================
# CONFIGURATION
# ============================================

# Path to the trained model (relative to project root)
# Use pathlib.Path so HuggingFace treats it as a local path (not a repo ID)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'eras_distilbert_final'

# Fallback to best_model if final doesn't exist yet
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_ROOT / 'models' / 'best_model'

# Confidence threshold — below this, flag for human review
CONFIDENCE_THRESHOLD = 0.7

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eras-api")

# ============================================
# REQUEST / RESPONSE MODELS
# ============================================

class PredictRequest(BaseModel):
    """Single prediction request"""
    text: str = Field(..., min_length=1, max_length=1000, description="Emergency report text")
    language: Optional[str] = Field("english", description="Language: english, amharic, afaan_oromo")

class PredictResponse(BaseModel):
    """Single prediction response"""
    text: str
    emergency_type: str
    confidence: float
    needs_review: bool
    routing: dict
    all_scores: dict
    timestamp: str

class BatchPredictRequest(BaseModel):
    """Batch prediction request"""
    texts: List[str] = Field(..., min_length=1, max_length=50, description="List of emergency texts")

class BatchPredictResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictResponse]
    total: int
    flagged_for_review: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    timestamp: str

class ModelInfoResponse(BaseModel):
    """Model info response"""
    model_name: str
    labels: dict
    num_labels: int
    device: str
    confidence_threshold: float
    training_config: Optional[dict] = None

# ============================================
# MODEL LOADER (singleton)
# ============================================

class ModelService:
    """Singleton service that loads and holds the model in memory."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = {}
        self.label_decoder = {}
        self.device = None
        self.model_name = "distilbert-base-uncased"
        self.training_config = None
        self._loaded = False
    
    def load(self, model_path=MODEL_PATH):
        """Load the trained model, tokenizer, and label encoder."""
        if self._loaded:
            return
        
        # Ensure it's a Path object (fixes spaces-in-path issue with HuggingFace)
        model_path = Path(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Device: {self.device}")
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train the model first: python training/train_distilbert.py"
            )
        
        # Load tokenizer & model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # Load label encoder
        encoder_path = model_path / 'label_encoder.json'
        with open(encoder_path, 'r') as f:
            encoders = json.load(f)
            self.label_encoder = encoders['label_encoder']
            self.label_decoder = encoders['label_decoder']
        
        # Load training config if available
        config_path = model_path / 'training_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.training_config = json.load(f)
                self.model_name = self.training_config.get('model_name', self.model_name)
        
        self._loaded = True
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"   Labels: {list(self.label_decoder.values())}")
    
    def predict(self, text: str) -> dict:
        """Run inference on a single text."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
        
        pred_label = self.label_decoder[str(prediction.item())]
        conf_value = confidence.item()
        
        # All class scores
        all_scores = {}
        for idx in range(probs.shape[1]):
            label = self.label_decoder[str(idx)]
            all_scores[label] = round(probs[0][idx].item(), 4)
        
        return {
            'emergency_type': pred_label,
            'confidence': round(conf_value, 4),
            'needs_review': conf_value < CONFIDENCE_THRESHOLD,
            'all_scores': all_scores,
            'routing': self._get_routing(pred_label)
        }
    
    @staticmethod
    def _get_routing(label: str) -> dict:
        """Map classification to dispatch routing."""
        responder_map = {
            'Medical/Hospital': 'Ambulance / Hospital',
            'Police': 'Police Department',
            'Fire Force': 'Fire Department',
            'Other': 'General Dispatch'
        }
        return {
            'primary_responder': responder_map.get(label, 'General Dispatch'),
            'requires_ambulance': 'Medical' in label,
            'requires_police': 'Police' in label,
            'requires_fire': 'Fire' in label,
            'is_non_emergency': label == 'Other'
        }


# Global model service instance
model_service = ModelService()

# ============================================
# FASTAPI APP
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("🚀 Starting ERAS API server...")
    model_service.load()
    yield
    logger.info("👋 Shutting down ERAS API server...")


app = FastAPI(
    title="ERAS Emergency Classification API",
    description=(
        "AI-powered emergency report classification for the "
        "Emergency Response Alert System (ERAS). "
        "Classifies emergency texts into: Medical/Hospital, Police, Fire Force, or Other."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow your frontend/backend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# ENDPOINTS
# ============================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the API and model are running."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_service._loaded,
        device=str(model_service.device),
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Get information about the loaded model."""
    return ModelInfoResponse(
        model_name=model_service.model_name,
        labels=model_service.label_encoder,
        num_labels=len(model_service.label_encoder),
        device=str(model_service.device),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        training_config=model_service.training_config
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Classify a single emergency report.
    
    Example request body:
    ```json
    {
        "text": "Fire at Adama Market! Flames spreading fast!",
        "language": "english"
    }
    ```
    """
    try:
        result = model_service.predict(request.text)
        
        return PredictResponse(
            text=request.text,
            emergency_type=result['emergency_type'],
            confidence=result['confidence'],
            needs_review=result['needs_review'],
            routing=result['routing'],
            all_scores=result['all_scores'],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Classify multiple emergency reports at once (max 50).
    
    Example request body:
    ```json
    {
        "texts": [
            "Fire at the market!",
            "Someone having a heart attack",
            "Robbery in progress"
        ]
    }
    ```
    """
    try:
        predictions = []
        for text in request.texts:
            result = model_service.predict(text)
            predictions.append(PredictResponse(
                text=text,
                emergency_type=result['emergency_type'],
                confidence=result['confidence'],
                needs_review=result['needs_review'],
                routing=result['routing'],
                all_scores=result['all_scores'],
                timestamp=datetime.now().isoformat()
            ))
        
        flagged = sum(1 for p in predictions if p.needs_review)
        
        return BatchPredictResponse(
            predictions=predictions,
            total=len(predictions),
            flagged_for_review=flagged
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ============================================
# RUN DIRECTLY (for development)
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
