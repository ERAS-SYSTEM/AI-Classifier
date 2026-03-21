# 🚨 ERAS - Emergency Response Alert System

## DistilBERT-based Emergency Text Classification

ERAS uses a fine-tuned **DistilBERT** model to automatically classify emergency reports into categories for rapid dispatch routing. The system supports **English**, **Amharic**, and **Afaan Oromo** text inputs.

---

## 📁 Project Structure

```
GC Project/
├── Traning Data/
│   └── eras_master_training_data.csv    # Master training dataset (3500 samples)
├── training/
│   ├── train_distilbert.py              # Main DistilBERT training pipeline
│   ├── evaluate_model.py               # Model evaluation & production classifier
│   └── predict_emergency.py            # Prediction script for ERAS integration
├── models/                              # Saved models & artifacts (auto-created)
│   ├── best_model/                      # Best checkpoint during training
│   ├── eras_distilbert_final/          # Final model for deployment
│   ├── training_history.png            # Loss/accuracy plots
│   └── confusion_matrix.png           # Confusion matrix visualization
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Training Data
```bash
python data/generate_training_data.py
```

### 3. Train the Model
```bash
python training/train_distilbert.py
```

### 4. Evaluate the Model
```bash
python training/evaluate_model.py
```

### 5. Run Predictions
```bash
python training/predict_emergency.py "Fire at Adama Market!"
```

---

## 📊 Emergency Categories

| Category | Description | Routed To |
|----------|-------------|-----------|
| **Medical/Hospital** | Health emergencies, injuries, cardiac events | Ambulance / Hospital |
| **Police** | Crime, theft, assault, robbery | Police Department |
| **Fire Force** | Fires, explosions, gas leaks | Fire Department |
| **Natural Disaster** | Floods, earthquakes, landslides | Disaster Response |
| **Traffic Accident** | Vehicle collisions, road incidents | Traffic Police / Ambulance |
| **Other** | Non-emergency or unclassified | General Dispatch |

---

## 📈 Expected Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 94–96% |
| **Medical/Hospital F1** | ~95% |
| **Police F1** | ~94% |
| **Fire Force F1** | ~96% |
| **Other F1** | ~92% |

---

## 🌍 Multilingual Support

The system handles emergency reports in:
- **English** — Primary language
- **Amharic (አማርኛ)** — Ethiopian national language
- **Afaan Oromo** — Widely spoken in the Oromia region

---

## 🔧 Hyperparameter Configs

| Config | Epochs | Batch Size | Learning Rate | Max Length | Use Case |
|--------|--------|------------|---------------|-----------|----------|
| Fast | 3 | 32 | 3e-5 | 128 | Quick prototyping |
| **Balanced** | **5** | **16** | **2e-5** | **128** | **Recommended** |
| High Accuracy | 8 | 8 | 1e-5 | 256 | Maximum performance |

---

## 🛠 Troubleshooting

- **GPU memory issues** → Reduce `batch_size` to 8
- **Overfitting** → Add dropout or reduce epochs
- **Low multilingual accuracy** → Add more Amharic/Oromo training samples
- **Low confidence scores** → Train longer or adjust threshold

---

## 📜 License

This project is part of the ERAS (Emergency Response Alert System) initiative.
