"""
============================================
🎯 ERAS - Emergency Prediction Script
============================================
Integration-ready prediction endpoint for ERAS.
Can be called from CLI or imported as a module.

CLI Usage:
    python predict_emergency.py "Fire at Adama Market!"
    python predict_emergency.py "Heart attack at bus station" --language amharic
    
Module Usage:
    from predict_emergency import process_emergency_report
    result = process_emergency_report("Fire at the market!")
============================================
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path so we can import evaluate_model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_model import ERASEmergencyClassifier


def process_emergency_report(text, language='english', model_path=None):
    """
    Process an emergency report for the ERAS system.
    
    Args:
        text: The emergency report text
        language: 'english', 'amharic', or 'afaan_oromo'
        model_path: Path to model (uses default if None)
    
    Returns:
        Dict with classification results and routing info
    """
    classifier = ERASEmergencyClassifier(model_path)
    
    detailed = classifier.predict_with_all_scores(text)
    label = detailed['prediction']
    confidence = detailed['confidence']
    
    # Determine routing based on classification
    result = {
        'original_text': text,
        'language': language,
        'classification': {
            'emergency_type': label,
            'confidence': round(confidence, 4),
            'needs_operator_review': confidence < 0.7,
            'all_scores': {k: round(v, 4) for k, v in detailed['all_scores'].items()}
        },
        'routing': {
            'primary_responder': _get_primary_responder(label),
            'requires_ambulance': 'Medical' in label,
            'requires_police': 'Police' in label,
            'requires_fire': 'Fire' in label,
            'is_non_emergency': label == 'Other'
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return result


def _get_primary_responder(label):
    """Map label to primary responder unit."""
    responder_map = {
        'Medical/Hospital': 'Ambulance / Hospital',
        'Police': 'Police Department',
        'Fire Force': 'Fire Department',
        'Other': 'General Dispatch'
    }
    return responder_map.get(label, 'General Dispatch')


# ============================================
# CLI ENTRY POINT
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚨 ERAS Emergency Report Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_emergency.py "Fire at Adama Market!"
  python predict_emergency.py "Heart attack at bus station" --language amharic
  python predict_emergency.py "Robbery in progress" --model models/best_model
        """
    )
    
    parser.add_argument(
        'text',
        nargs='?',
        default="Fire at Mexico Square! Need help immediately!",
        help='Emergency report text to classify'
    )
    parser.add_argument(
        '--language', '-l',
        default='english',
        choices=['english', 'amharic', 'afaan_oromo'],
        help='Language of the report (default: english)'
    )
    parser.add_argument(
        '--model', '-m',
        default=None,
        help='Path to trained model directory'
    )
    
    args = parser.parse_args()
    
    print(f"\n🚨 ERAS Emergency Classification")
    print(f"{'='*50}")
    
    result = process_emergency_report(
        text=args.text,
        language=args.language,
        model_path=args.model
    )
    
    print(f"\n{json.dumps(result, indent=2, ensure_ascii=False)}")
    
    # Pretty summary
    cls = result['classification']
    rte = result['routing']
    
    print(f"\n{'='*50}")
    print(f"📋 SUMMARY")
    print(f"{'='*50}")
    print(f"   Input:      {args.text[:60]}...")
    print(f"   Type:       {cls['emergency_type']}")
    print(f"   Confidence: {cls['confidence']:.1%}")
    print(f"   Responder:  {rte['primary_responder']}")
    
    if cls['needs_operator_review']:
        print(f"   ⚠️  LOW CONFIDENCE — Routing to human operator for review")
    else:
        print(f"   ✅ AUTO-DISPATCH — Confidence above threshold")
    
    print()
