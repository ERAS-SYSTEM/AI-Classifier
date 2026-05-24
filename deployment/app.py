"""
🚨 ERAS Emergency Classifier - Hugging Face Spaces Demo & API
========================================================
Serves BOTH a Gradio Web Interface at the root (/) and a secure
FastAPI REST API (/predict, /predict/batch, /health) on Hugging Face.

Deploy to: https://huggingface.co/spaces
"""

import os
import json
import torch
# pyrefly: ignore [missing-import]
import gradio as gr
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ============================================
# LOAD MODEL & CONFIG
# ============================================

MODEL_DIR = Path("model")
if not MODEL_DIR.exists():
    MODEL_DIR = Path(__file__).resolve().parent / "model"

print("Loading ERAS classifier...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(MODEL_DIR / "label_encoder.json", "r") as f:
    encoders = json.load(f)
    label_decoder = encoders["label_decoder"]

print(f"[SUCCESS] Model loaded! Labels: {list(label_decoder.values())}")

# Emoji & color mapping for Gradio UI
LABEL_INFO = {
    "Fire Force":        {"emoji": "🚒", "color": "#FF4500"},
    "Medical/Hospital":  {"emoji": "🏥", "color": "#00BFFF"},
    "Police":            {"emoji": "🚔", "color": "#4169E1"},
    "Other":             {"emoji": "💬", "color": "#808080"},
}

# ============================================
# FASTAPI APP & AUTH SETUP
# ============================================

app = FastAPI(
    title="ERAS Emergency Classification API",
    description="Secure REST API for ERAS emergency text classification.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
VALID_API_KEY = os.environ.get("ERAS_API_KEY", "eras_secure_api_key_2026")

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Validates the X-API-Key request header."""
    if api_key == VALID_API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials. Invalid or missing X-API-Key."
    )

# ============================================
# REQUEST / RESPONSE MODELS
# ============================================

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Emergency report text")
    language: Optional[str] = Field("english", description="Language: english, amharic, afaan_oromo")

class PredictResponse(BaseModel):
    text: str
    emergency_type: str
    confidence: float
    needs_review: bool
    routing: dict
    all_scores: dict
    timestamp: str

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50, description="List of emergency texts")

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    total: int
    flagged_for_review: int

# ============================================
# INFERENCE LOGIC SHARED FUNCTIONS
# ============================================

def run_model_inference(text: str) -> dict:
    """Helper function to run tokenizer and model sequence classification."""
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        
    pred_label = label_decoder[str(prediction.item())]
    conf_value = confidence.item()
    
    # Calculate all class scores
    all_scores = {}
    for idx in range(probs.shape[1]):
        label = label_decoder[str(idx)]
        all_scores[label] = round(probs[0][idx].item(), 4)
        
    responder_map = {
        'Medical/Hospital': 'Ambulance / Hospital',
        'Police': 'Police Department',
        'Fire Force': 'Fire Department',
        'Other': 'General Dispatch'
    }
    
    routing = {
        'primary_responder': responder_map.get(pred_label, 'General Dispatch'),
        'requires_ambulance': 'Medical' in pred_label,
        'requires_police': 'Police' in pred_label,
        'requires_fire': 'Fire' in pred_label,
        'is_non_emergency': pred_label == 'Other'
    }
    
    return {
        'emergency_type': pred_label,
        'confidence': round(conf_value, 4),
        'needs_review': conf_value < 0.7,
        'routing': routing,
        'all_scores': all_scores
    }

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(model.device if hasattr(model, 'device') else "cpu"),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", tags=["System"])
async def model_info():
    return {
        "model_name": "distilbert-base-uncased",
        "labels": label_decoder,
        "num_labels": len(label_decoder),
        "device": str(model.device if hasattr(model, 'device') else "cpu"),
        "confidence_threshold": 0.7
    }

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest, api_key: str = Depends(verify_api_key)):
    try:
        result = run_model_inference(request.text)
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest, api_key: str = Depends(verify_api_key)):
    try:
        predictions = []
        for text in request.texts:
            result = run_model_inference(text)
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
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# GRADIO INTERFACE LOGIC
# ============================================

def classify_emergency_ui(text):
    """Callback for the Gradio UI."""
    if not text or not text.strip():
        return {}, "Please enter an emergency report text."
        
    result = run_model_inference(text)
    
    # Build scores dict for Gradio Label component
    scores = {}
    for label, prob in result['all_scores'].items():
        info = LABEL_INFO.get(label, {"emoji": "❓"})
        display_name = f"{info['emoji']} {label}"
        scores[display_name] = prob
        
    conf_value = result['confidence']
    pred_label = result['emergency_type']
    info = LABEL_INFO.get(pred_label, {"emoji": "❓"})
    
    if conf_value >= 0.9:
        status_text = "✅ HIGH CONFIDENCE — Auto-dispatch recommended"
    elif conf_value >= 0.7:
        status_text = "🟡 ACCEPTABLE — Can proceed with dispatch"
    else:
        status_text = "⚠️ LOW CONFIDENCE — Route to human operator for review"
        
    summary = f"""
### {info['emoji']} Classification: **{pred_label}**

| Detail | Value |
|--------|-------|
| **Confidence** | {conf_value:.1%} |
| **Status** | {status_text} |
| **Route to** | {result['routing']['primary_responder']} |
| **Requires Ambulance** | {'Yes' if result['routing']['requires_ambulance'] else 'No'} |
| **Requires Police** | {'Yes' if result['routing']['requires_police'] else 'No'} |
| **Requires Fire Dept** | {'Yes' if result['routing']['requires_fire'] else 'No'} |
"""
    return scores, summary

# Build UI
example_texts = [
    ["Fire at Adama Market! Flames spreading fast!"],
    ["Someone is having a heart attack at the bus station"],
    ["Robbery in progress at the bank on Bole Road"],
    ["Armed suspect spotted near the school"],
    ["Hello, just testing the app"],
    ["Emergency! Gas leak reported at the factory"],
    ["እሳት ወጥቷል በአዳማ ገበያ"],
    ["Hospitaala nan barbaada"],
    ["Person unconscious at the park, not breathing"],
    ["What is the weather like today?"],
]

with gr.Blocks(
    title="ERAS Emergency Classifier",
    theme=gr.themes.Soft(primary_hue="red", secondary_hue="blue"),
) as demo:
    gr.Markdown("""
    # 🚨 ERAS — Emergency Response Alert System
    ### AI-Powered Emergency Text Classification
    
    Enter an emergency report text below and the DistilBERT model will classify it 
    into one of four categories: **Fire Force**, **Medical/Hospital**, **Police**, or **Other**.
    
    Supports: 🇬🇧 English · 🇪🇹 Amharic (አማርኛ) · Afaan Oromo
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Emergency Report",
                placeholder="Type or paste an emergency report here...",
                lines=3,
            )
            classify_btn = gr.Button("🔍 Classify Emergency", variant="primary", size="lg")
            gr.Examples(examples=example_texts, inputs=text_input, label="Try these examples:")
            
        with gr.Column(scale=1):
            label_output = gr.Label(label="Classification Scores", num_top_classes=4)
            summary_output = gr.Markdown(label="Details")
            
    classify_btn.click(
        fn=classify_emergency_ui,
        inputs=text_input,
        outputs=[label_output, summary_output],
    )
    
    text_input.submit(
        fn=classify_emergency_ui,
        inputs=text_input,
        outputs=[label_output, summary_output],
    )
    
    gr.Markdown("""
    ---
    *Built with DistilBERT · Fine-tuned on 6,000+ emergency reports · 96.8% accuracy*  
    *Part of the [ERAS-SYSTEM](https://github.com/ERAS-SYSTEM) project*
    """)

# ============================================
# MOUNT GRADIO ONTO FASTAPI & LAUNCH
# ============================================

# Mount Gradio onto the root (/) of the FastAPI server
# This makes Gradio UI available at the root URL, and API endpoints at /predict, /health, /docs etc.
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces port defaults to 7860
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)