"""
============================================
🚀 ERAS - Retrain on Synthetic Dataset
============================================
Retrains the DistilBERT classifier on the new
synthetic training data:
  eras_synthetic_training_data (1).csv

Run from the project root:
    python training/retrain_on_synthetic.py

Output:
  models/best_model/          ← checkpoint (saved mid-training)
  models/eras_distilbert_final/ ← final model used by the API
  models/training_history.png
  models/confusion_matrix.png
============================================
"""

import os
import sys

# Ensure project root is on the path so relative imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Re-use the shared trainer from train_distilbert.py
from training.train_distilbert import ERASDistilBERTTrainer

# ============================================
# PATHS
# ============================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    'Traning Data',
    'eras_synthetic_training_data (1).csv'
)

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("ERAS - RETRAIN ON SYNTHETIC DATA")
    print("=" * 60)
    print(f"Data : {DATA_PATH}")
    print(f"Output: {MODELS_DIR}")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Data file not found:\n   {DATA_PATH}")
        sys.exit(1)

    # 1. Initialize trainer
    trainer = ERASDistilBERTTrainer()

    # 2. Load & split the new synthetic dataset
    train_data, val_data, test_data = trainer.prepare_data(data_path=DATA_PATH)

    # 3. Build model head for the discovered label count
    trainer.initialize_model()

    # 4. Train
    #    - epochs=5  is solid for ~2 500 samples; bump to 7-8 if val_acc plateaus
    #    - batch_size=16 works on CPU; reduce to 8 if you hit memory errors
    #    - learning_rate=2e-5 is the standard DistilBERT sweet-spot
    best_model_path = trainer.train(
        train_data,
        val_data,
        epochs=5,
        batch_size=16,
        learning_rate=2e-5,
        max_len=128
    )

    # 5. Full evaluation on the held-out test set
    test_results = trainer.test_model(test_data, best_model_path)

    # 6. Visualisations
    trainer.plot_training_history(
        save_path=os.path.join(MODELS_DIR, 'training_history.png')
    )
    trainer.plot_confusion_matrix(
        test_results['true_labels'],
        test_results['predictions'],
        save_path=os.path.join(MODELS_DIR, 'confusion_matrix.png')
    )

    # 7. Save final model — this is what the API loads at startup
    trainer.save_model_and_tokenizer(
        path=os.path.join(MODELS_DIR, 'eras_distilbert_final')
    )

    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE!")
    print("  -> API will pick up the new model automatically on next restart.")
    print("=" * 60)
