"""
============================================
🔍 ERAS - Model Evaluation & Classifier
============================================
Production-ready classifier for the ERAS system.
Loads a trained DistilBERT model and provides
single-text and batch prediction capabilities.
============================================
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report

# Default model paths (relative to project root)
# Use pathlib.Path so HuggingFace treats it as a local path (not a repo ID)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / 'models' / 'eras_distilbert_final'
FALLBACK_MODEL_PATH = PROJECT_ROOT / 'models' / 'best_model'


class ERASEmergencyClassifier:
    """
    Production-ready emergency text classifier for ERAS.
    
    Usage:
        classifier = ERASEmergencyClassifier()
        label, confidence = classifier.predict("Fire at the market!")
    """
    
    def __init__(self, model_path=None):
        """
        Load a trained model, tokenizer, and label encoder.
        
        Args:
            model_path: Path to the saved model directory.
                        Defaults to models/eras_distilbert_final.
        """
        if model_path is None:
            # Use final model if it exists, otherwise fallback to best_model
            if DEFAULT_MODEL_PATH.exists():
                model_path = DEFAULT_MODEL_PATH
            elif FALLBACK_MODEL_PATH.exists():
                model_path = FALLBACK_MODEL_PATH
                print(f"⚠️  Final model not found, using best_model checkpoint")
            else:
                raise FileNotFoundError(
                    "No trained model found. Please run training first:\n"
                    "  python training/train_distilbert.py"
                )
        
        # Ensure it's a Path object (fixes spaces-in-path issue with HuggingFace)
        model_path = Path(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # Load label encoder
        encoder_path = model_path / 'label_encoder.json'
        with open(encoder_path, 'r') as f:
            encoders = json.load(f)
            self.label_encoder = encoders['label_encoder']
            self.label_decoder = encoders['label_decoder']
        
        print(f"✅ Loaded ERAS classifier from {model_path}")
        print(f"   Device: {self.device}")
        print(f"   Labels: {list(self.label_decoder.values())}")
    
    def predict(self, text, return_confidence=True):
        """
        Predict emergency type from a single text input.
        
        Args:
            text: The emergency report text
            return_confidence: If True, also returns the confidence score
        
        Returns:
            If return_confidence is True:  (label, confidence)
            If return_confidence is False: label
        """
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
        
        if return_confidence:
            return pred_label, confidence.item()
        return pred_label
    
    def predict_with_all_scores(self, text):
        """
        Predict and return confidence scores for ALL labels.
        
        Args:
            text: The emergency report text
        
        Returns:
            Dict with 'prediction', 'confidence', and 'all_scores'
        """
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
        
        all_scores = {}
        for idx in range(probs.shape[1]):
            label = self.label_decoder[str(idx)]
            all_scores[label] = probs[0][idx].item()
        
        return {
            'prediction': pred_label,
            'confidence': confidence.item(),
            'all_scores': all_scores
        }
    
    def batch_predict(self, texts, confidence_threshold=0.7):
        """
        Predict emergency types for multiple texts.
        
        Args:
            texts: List of emergency report texts
            confidence_threshold: Predictions below this threshold are flagged
        
        Returns:
            DataFrame with text, predicted_label, confidence, and needs_review
        """
        results = []
        for text in texts:
            label, conf = self.predict(text)
            results.append({
                'text': text,
                'predicted_label': label,
                'confidence': round(conf, 4),
                'needs_review': conf < confidence_threshold
            })
        return pd.DataFrame(results)


# ============================================
# MAIN - Run evaluation demo
# ============================================

if __name__ == "__main__":
    
    classifier = ERASEmergencyClassifier()
    
    # Test cases covering all categories and languages
    test_cases = [
        # English - Fire
        "Fire at Adama Market! Flames spreading fast",
        "Emergency! Gas leak reported at the factory",
        
        # English - Medical
        "Someone is having a heart attack at the bus station",
        "Urgent: Person unconscious at the park, not breathing",
        
        # English - Police
        "Robbery in progress at the bank on Bole Road",
        "Armed suspect spotted near the school",
        
        # English - Other
        "Hello, just testing the app",
        "What is the weather like today?",
        
        # Amharic
        "እሳት ወጥቷል በአዳማ ገበያ",          # Fire at Adama Market
        "እርዳታ ያስፈልገኛል በPiazza",         # I need help at Piazza
        
        # Afaan Oromo
        "Hospitaala nan barbaada",            # I need a hospital
        "Poolisiin na biliisa Piazza",        # Police help at Piazza
    ]
    
    print(f"\n{'='*60}")
    print(f"🔍 ERAS CLASSIFIER - EVALUATION")
    print(f"{'='*60}\n")
    
    for text in test_cases:
        label, conf = classifier.predict(text)
        
        if conf >= 0.9:
            status = "✅ HIGH CONFIDENCE"
        elif conf >= 0.7:
            status = "🟡 ACCEPTABLE"
        else:
            status = "⚠️  NEEDS REVIEW"
        
        display_text = text[:55] + "..." if len(text) > 55 else text
        print(f"Text: {display_text}")
        print(f"  → {label} (conf: {conf:.3f}) {status}")
        print()
    
    # Batch prediction demo
    print(f"\n{'='*60}")
    print(f"📊 BATCH PREDICTION DEMO")
    print(f"{'='*60}\n")
    
    batch_results = classifier.batch_predict(test_cases[:6])
    print(batch_results.to_string(index=False))
