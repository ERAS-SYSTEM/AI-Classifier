"""
🚨 ERAS Emergency Classifier - Hugging Face Spaces Demo
========================================================
A simple Gradio web interface for testing the ERAS
DistilBERT emergency text classifier.

Deploy to: https://huggingface.co/spaces
"""

import json
import torch
import gradio as gr
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ============================================
# LOAD MODEL
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

# Emoji & color mapping
LABEL_INFO = {
    "Fire Force":        {"emoji": "🚒", "color": "#FF4500"},
    "Medical/Hospital":  {"emoji": "🏥", "color": "#00BFFF"},
    "Police":            {"emoji": "🚔", "color": "#4169E1"},
    "Other":             {"emoji": "💬", "color": "#808080"},
}

# ============================================
# PREDICTION FUNCTION
# ============================================

def classify_emergency(text):
    """Classify emergency text and return results."""
    if not text or not text.strip():
        return {}, "Please enter an emergency report text."

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

    # Build scores dict for Gradio Label component
    scores = {}
    for idx in range(probs.shape[1]):
        label = label_decoder[str(idx)]
        info = LABEL_INFO.get(label, {"emoji": "❓"})
        display_name = f"{info['emoji']} {label}"
        scores[display_name] = float(probs[0][idx])

    # Build summary text
    confidence, prediction = torch.max(probs, dim=1)
    pred_label = label_decoder[str(prediction.item())]
    conf_value = confidence.item()
    info = LABEL_INFO.get(pred_label, {"emoji": "❓"})

    if conf_value >= 0.9:
        status = "✅ HIGH CONFIDENCE — Auto-dispatch recommended"
    elif conf_value >= 0.7:
        status = "🟡 ACCEPTABLE — Can proceed with dispatch"
    else:
        status = "⚠️ LOW CONFIDENCE — Route to human operator for review"

    responder_map = {
        "Medical/Hospital": "Ambulance / Hospital",
        "Police": "Police Department",
        "Fire Force": "Fire Department",
        "Other": "General Dispatch (Non-emergency)",
    }

    summary = f"""
### {info['emoji']} Classification: **{pred_label}**

| Detail | Value |
|--------|-------|
| **Confidence** | {conf_value:.1%} |
| **Status** | {status} |
| **Route to** | {responder_map.get(pred_label, 'General Dispatch')} |
| **Requires Ambulance** | {'Yes' if 'Medical' in pred_label else 'No'} |
| **Requires Police** | {'Yes' if 'Police' in pred_label else 'No'} |
| **Requires Fire Dept** | {'Yes' if 'Fire' in pred_label else 'No'} |
"""

    return scores, summary


# ============================================
# GRADIO INTERFACE
# ============================================

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
        fn=classify_emergency,
        inputs=text_input,
        outputs=[label_output, summary_output],
    )

    text_input.submit(
        fn=classify_emergency,
        inputs=text_input,
        outputs=[label_output, summary_output],
    )

    gr.Markdown("""
    ---
    *Built with DistilBERT · Fine-tuned on 3,500 emergency reports · 96.8% accuracy*  
    *Part of the [ERAS-SYSTEM](https://github.com/ERAS-SYSTEM) project*
    """)

if __name__ == "__main__":
    demo.launch()
    