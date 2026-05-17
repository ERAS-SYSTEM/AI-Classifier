"""
============================================
🚀 ERAS - Train on Combined Dataset
============================================
Merges the master training data and the synthetic
dataset, then trains the DistilBERT classifier on
the combined dataset.

Run from the project root:
    python training/train_combined.py

Output:
  models/best_model/            ← checkpoint (saved mid-training)
  models/eras_distilbert_final/  ← final model used by the API
  models/training_history.png
  models/confusion_matrix.png
============================================
"""

import os
import sys
import pandas as pd

# Ensure project root is on the path so relative imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Re-use the shared trainer from train_distilbert.py
from training.train_distilbert import ERASDistilBERTTrainer

# ============================================
# PATHS
# ============================================

MASTER_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    'Traning Data',
    'eras_master_training_data.csv'
)

SYNTHETIC_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    'Traning Data',
    'eras_synthetic_training_data (1).csv'
)

COMBINED_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    'Traning Data',
    'eras_combined_training_data.csv'
)

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("ERAS - COMBINING DATASETS & TRAINING MODEL")
    print("=" * 60)
    print(f"Master Data    : {MASTER_DATA_PATH}")
    print(f"Synthetic Data : {SYNTHETIC_DATA_PATH}")
    print(f"Combined Output: {COMBINED_DATA_PATH}")
    print(f"Models Directory: {MODELS_DIR}")
    print("=" * 60)

    # Validate that both input data files exist
    if not os.path.exists(MASTER_DATA_PATH):
        print(f"\nERROR: Master training data file not found:\n   {MASTER_DATA_PATH}")
        sys.exit(1)
        
    if not os.path.exists(SYNTHETIC_DATA_PATH):
        print(f"\nERROR: Synthetic training data file not found:\n   {SYNTHETIC_DATA_PATH}")
        sys.exit(1)

    # 1. Load both datasets
    print("\nLoading datasets...")
    df_master = pd.read_csv(MASTER_DATA_PATH)
    df_synthetic = pd.read_csv(SYNTHETIC_DATA_PATH)
    
    print(f"   -> Loaded {len(df_master)} master samples.")
    print(f"   -> Loaded {len(df_synthetic)} synthetic samples.")

    # 2. Combine and clean
    print("\nMerging datasets...")
    df_combined = pd.concat([df_master, df_synthetic], ignore_index=True)
    
    # Drop exact text duplicates to avoid overfitting on same samples
    initial_len = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['text'])
    dedup_len = len(df_combined)
    
    print(f"   -> Combined total: {initial_len} samples.")
    if initial_len != dedup_len:
        print(f"   -> Removed {initial_len - dedup_len} duplicate texts. New total: {dedup_len} unique samples.")

    # Drop rows without text or label
    df_combined = df_combined.dropna(subset=['text', 'label'])
    
    # Save combined dataset to CSV
    os.makedirs(os.path.dirname(COMBINED_DATA_PATH), exist_ok=True)
    df_combined.to_csv(COMBINED_DATA_PATH, index=False)
    print(f"✅ Combined dataset saved to: {COMBINED_DATA_PATH}")

    # 3. Initialize trainer
    trainer = ERASDistilBERTTrainer()

    # 4. Load & split the combined dataset
    train_data, val_data, test_data = trainer.prepare_data(data_path=COMBINED_DATA_PATH)

    # 5. Build model head for the discovered label count
    trainer.initialize_model()

    # 6. Train on the combined dataset
    #    - epochs=5 is balanced for ~6,000 samples.
    #    - batch_size=16 is standard. Reduce to 8 if you hit GPU memory errors (OOM).
    #    - learning_rate=2e-5 is standard for DistilBERT fine-tuning.
    best_model_path = trainer.train(
        train_data,
        val_data,
        epochs=5,
        batch_size=16,
        learning_rate=2e-5,
        max_len=128
    )

    # 7. Full evaluation on the held-out test set
    test_results = trainer.test_model(test_data, best_model_path)

    # 8. Visualizations
    trainer.plot_training_history(
        save_path=os.path.join(MODELS_DIR, 'training_history.png')
    )
    trainer.plot_confusion_matrix(
        test_results['true_labels'],
        test_results['predictions'],
        save_path=os.path.join(MODELS_DIR, 'confusion_matrix.png')
    )

    # 9. Save final model — this is what the API loads at startup
    trainer.save_model_and_tokenizer(
        path=os.path.join(MODELS_DIR, 'eras_distilbert_final')
    )

    print("\n" + "=" * 60)
    print("COMBINED RETRAINING COMPLETE!")
    print("  -> The API server will pick up the new model automatically upon restart.")
    print("=" * 60)
