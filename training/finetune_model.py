"""
============================================
🔄 ERAS - Fine-tune Existing Model
============================================
Continue training the already-trained DistilBERT
model on new/additional data. This preserves what
the model already learned and improves it further.

Usage:
    python training/finetune_model.py --data "path/to/new_data.csv"
    python training/finetune_model.py --data "path/to/new_data.csv" --epochs 3 --lr 1e-5
============================================
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================
# CONFIGURATION
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where to load the existing trained model from
EXISTING_MODEL_PATH = PROJECT_ROOT / 'models' / 'eras_distilbert_final'
FALLBACK_MODEL_PATH = PROJECT_ROOT / 'models' / 'best_model'

# Where to save the improved model
OUTPUT_MODEL_PATH = PROJECT_ROOT / 'models' / 'eras_distilbert_finetuned'


# ============================================
# DATASET
# ============================================

class EmergencyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ============================================
# FINE-TUNER
# ============================================

def finetune(
    new_data_path,
    epochs=3,
    batch_size=16,
    learning_rate=1e-5,
    max_len=128,
    val_split=0.15
):
    """
    Fine-tune the existing trained model on new data.
    
    Args:
        new_data_path: Path to new CSV (must have 'text' and 'label' columns)
        epochs: Number of fine-tuning epochs (2-3 is usually enough)
        batch_size: Batch size
        learning_rate: Use a LOWER rate than initial training (1e-5 recommended)
        max_len: Max token length
        val_split: Fraction of new data for validation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ----- 1. Find existing model -----
    if EXISTING_MODEL_PATH.exists():
        model_path = EXISTING_MODEL_PATH
    elif FALLBACK_MODEL_PATH.exists():
        model_path = FALLBACK_MODEL_PATH
    else:
        raise FileNotFoundError(
            "No trained model found. Run train_distilbert.py first."
        )
    
    print(f"\n{'='*60}")
    print(f"🔄 ERAS - FINE-TUNING PIPELINE")
    print(f"{'='*60}")
    print(f"   Base model:     {model_path}")
    print(f"   New data:       {new_data_path}")
    print(f"   Device:         {device}")
    print(f"   Epochs:         {epochs}")
    print(f"   Learning rate:  {learning_rate}")
    print(f"{'='*60}")
    
    # ----- 2. Load existing model & tokenizer -----
    print(f"\n📦 Loading existing trained model...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    
    # Load label encoder
    encoder_path = model_path / 'label_encoder.json'
    with open(encoder_path, 'r') as f:
        encoders = json.load(f)
        label_encoder = encoders['label_encoder']
        label_decoder = encoders['label_decoder']
    
    print(f"   Labels: {list(label_encoder.keys())}")
    print(f"   ✅ Model loaded!")
    
    # ----- 3. Load & prepare new data -----
    print(f"\n📂 Loading new data...")
    df = pd.read_csv(new_data_path)
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip()
    
    print(f"   New samples: {len(df)}")
    print(f"\n   Label distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"      {label}: {count}")
    
    # Validate labels match the existing model
    unknown_labels = set(df['label'].unique()) - set(label_encoder.keys())
    if unknown_labels:
        print(f"\n   ⚠️  WARNING: Unknown labels found: {unknown_labels}")
        print(f"   These will be SKIPPED. Known labels: {list(label_encoder.keys())}")
        df = df[df['label'].isin(label_encoder.keys())]
    
    # Encode labels
    df['encoded_label'] = df['label'].map(label_encoder)
    
    # Split into train/val
    train_df, val_df = train_test_split(
        df, test_size=val_split,
        stratify=df['encoded_label'],
        random_state=42
    )
    
    print(f"\n   Fine-tune split:")
    print(f"      Train: {len(train_df)} samples")
    print(f"      Val:   {len(val_df)} samples")
    
    # Create datasets
    train_dataset = EmergencyDataset(
        train_df['text'].values, train_df['encoded_label'].values,
        tokenizer, max_len
    )
    val_dataset = EmergencyDataset(
        val_df['text'].values, val_df['encoded_label'].values,
        tokenizer, max_len
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # ----- 4. Fine-tune -----
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\n📌 Epoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        # Train
        model.train()
        total_loss = 0
        steps = 0
        
        for batch in tqdm(train_loader, desc="   Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            total_loss += loss.item()
            steps += 1
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        
        avg_loss = total_loss / steps
        
        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f"\n   📊 Train Loss: {avg_loss:.4f}  |  Val Acc: {val_acc:.4f} ({val_acc*100:.1f}%)  |  Val F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"   🏆 New best! Saving...")
    
    # ----- 5. Save fine-tuned model -----
    print(f"\n💾 Saving fine-tuned model to {OUTPUT_MODEL_PATH}")
    OUTPUT_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(OUTPUT_MODEL_PATH)
    tokenizer.save_pretrained(OUTPUT_MODEL_PATH)
    
    with open(OUTPUT_MODEL_PATH / 'label_encoder.json', 'w') as f:
        json.dump({'label_encoder': label_encoder, 'label_decoder': label_decoder}, f, indent=2)
    
    config = {
        'model_name': 'distilbert-base-uncased (fine-tuned)',
        'base_model': str(model_path),
        'num_labels': len(label_encoder),
        'labels': label_encoder,
        'best_val_acc': best_val_acc,
        'finetuned_on': str(new_data_path),
        'finetuned_samples': len(df),
        'finetuned_at': datetime.now().isoformat()
    }
    
    with open(OUTPUT_MODEL_PATH / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ FINE-TUNING COMPLETE!")
    print(f"   Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"   Model saved to: {OUTPUT_MODEL_PATH}")
    print(f"{'='*60}")
    print(f"\n💡 To use the fine-tuned model, update your API or scripts to point to:")
    print(f"   models/eras_distilbert_finetuned")


# ============================================
# CLI
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🔄 Fine-tune ERAS model on new data")
    parser.add_argument('--data', '-d', required=True, help='Path to new CSV data file')
    parser.add_argument('--epochs', '-e', type=int, default=3, help='Fine-tuning epochs (default: 3)')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (default: 1e-5, lower than initial training)')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='Batch size (default: 16)')
    
    args = parser.parse_args()
    
    finetune(
        new_data_path=args.data,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
