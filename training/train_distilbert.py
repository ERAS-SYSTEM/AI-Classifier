"""
============================================
🚀 ERAS - DistilBERT Training Pipeline
============================================
Fine-tunes DistilBERT for emergency text classification.

Emergency Categories:
  - Medical/Hospital
  - Police
  - Fire Force
  - Other

Supports: English, Amharic (አማርኛ), Afaan Oromo
============================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

# ============================================
# CONFIGURATION
# ============================================

# Path to training data (relative to project root)
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Traning Data',
    'eras_master_training_data.csv'
)

# Output directories
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models'
)

# ============================================
# CUSTOM DATASET
# ============================================

class EmergencyDataset(Dataset):
    """PyTorch Dataset for emergency text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================
# MAIN TRAINER CLASS
# ============================================

class ERASDistilBERTTrainer:
    """
    End-to-end trainer for ERAS emergency classification.
    
    Usage:
        trainer = ERASDistilBERTTrainer()
        train_data, val_data, test_data = trainer.prepare_data()
        trainer.initialize_model()
        trainer.train(train_data, val_data)
        trainer.test_model(test_data)
        trainer.save_model_and_tokenizer()
    """
    
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.label_encoder = {}
        self.label_decoder = {}
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"🔧 Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    # ------------------------------------------
    # DATA PREPARATION
    # ------------------------------------------
    
    def prepare_data(self, data_path=None, test_size=0.15, val_size=0.15, random_state=42):
        """
        Load CSV, encode labels, and split into train/val/test sets.
        
        Args:
            data_path: Path to CSV file (defaults to DATA_PATH constant)
            test_size: Fraction of data for testing
            val_size: Fraction of remaining data for validation
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (train_data, val_data, test_data) DataFrames
        """
        if data_path is None:
            data_path = DATA_PATH
        
        print(f"\n📂 Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        print(f"   Total samples: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Clean data
        df = df.dropna(subset=['text', 'label'])
        df['text'] = df['text'].astype(str).str.strip()
        df['label'] = df['label'].astype(str).str.strip()
        
        # Show label distribution
        print(f"\n📊 Label Distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            pct = count / len(df) * 100
            print(f"   {label}: {count} ({pct:.1f}%)")
        
        # Create label encoders
        unique_labels = sorted(df['label'].unique())
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_decoder = {str(idx): label for label, idx in self.label_encoder.items()}
        
        print(f"\n🏷️  Label Encoding: {self.label_encoder}")
        
        # Encode labels
        df['encoded_label'] = df['label'].map(self.label_encoder)
        
        # Split: first separate test set, then split remainder into train/val
        train_val_df, test_df = train_test_split(
            df, test_size=test_size,
            stratify=df['encoded_label'],
            random_state=random_state
        )
        
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size / (1 - test_size),
            stratify=train_val_df['encoded_label'],
            random_state=random_state
        )
        
        print(f"\n✂️  Data Split:")
        print(f"   Train: {len(train_df)} samples")
        print(f"   Val:   {len(val_df)} samples")
        print(f"   Test:  {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    # ------------------------------------------
    # MODEL INITIALIZATION
    # ------------------------------------------
    
    def initialize_model(self):
        """Load DistilBERT tokenizer and model with correct number of labels."""
        
        num_labels = len(self.label_encoder)
        print(f"\n🤖 Initializing DistilBERT:")
        print(f"   Model: {self.model_name}")
        print(f"   Labels: {num_labels} ({list(self.label_encoder.keys())})")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable:    {trainable_params:,}")
        print(f"   ✅ Model loaded successfully!")
    
    # ------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------
    
    def train(self, train_data, val_data, epochs=5, batch_size=16,
              learning_rate=2e-5, max_len=128, warmup_ratio=0.1):
        """
        Train the DistilBERT model.
        
        Args:
            train_data: Training DataFrame with 'text' and 'encoded_label' columns
            val_data: Validation DataFrame
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Peak learning rate
            max_len: Maximum token sequence length
            warmup_ratio: Fraction of steps for learning rate warmup
        
        Returns:
            Path to the best model checkpoint
        """
        print(f"\n{'='*60}")
        print(f"🏋️  TRAINING CONFIGURATION")
        print(f"{'='*60}")
        print(f"   Epochs:        {epochs}")
        print(f"   Batch size:    {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Max length:    {max_len}")
        print(f"   Warmup ratio:  {warmup_ratio}")
        print(f"{'='*60}")
        
        # Create datasets
        train_dataset = EmergencyDataset(
            train_data['text'].values,
            train_data['encoded_label'].values,
            self.tokenizer,
            max_len
        )
        
        val_dataset = EmergencyDataset(
            val_data['text'].values,
            val_data['encoded_label'].values,
            self.tokenizer,
            max_len
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer & scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_acc = 0.0
        best_model_path = os.path.join(MODELS_DIR, 'best_model')
        os.makedirs(best_model_path, exist_ok=True)
        
        for epoch in range(epochs):
            print(f"\n📌 Epoch {epoch + 1}/{epochs}")
            print("-" * 40)
            
            # --- Train phase ---
            self.model.train()
            total_train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(train_loader, desc=f"   Training", leave=True)
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.model.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                train_steps += 1
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_train_loss / train_steps
            
            # --- Validation phase ---
            val_loss, val_acc, val_f1 = self._evaluate(val_loader)
            
            # Record history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            print(f"\n   📊 Results:")
            print(f"      Train Loss: {avg_train_loss:.4f}")
            print(f"      Val Loss:   {val_loss:.4f}")
            print(f"      Val Acc:    {val_acc:.4f} ({val_acc*100:.1f}%)")
            print(f"      Val F1:     {val_f1:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.model.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)
                # Also save label encoder so the checkpoint is self-contained
                with open(os.path.join(best_model_path, 'label_encoder.json'), 'w') as f:
                    json.dump({
                        'label_encoder': self.label_encoder,
                        'label_decoder': self.label_decoder
                    }, f, indent=2)
                print(f"      🏆 New best model saved! (acc: {best_val_acc:.4f})")
        
        print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.4f}")
        return best_model_path
    
    def _evaluate(self, data_loader):
        """Run evaluation on a data loader. Returns (loss, accuracy, f1)."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        steps = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                steps += 1
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / steps
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1
    
    # ------------------------------------------
    # TESTING
    # ------------------------------------------
    
    def test_model(self, test_data, model_path=None, max_len=128, batch_size=16):
        """
        Evaluate the model on the test set with full classification report.
        
        Args:
            test_data: Test DataFrame
            model_path: Path to saved model (uses best_model if None)
            max_len: Max token length
            batch_size: Batch size for evaluation
        
        Returns:
            Dict with true_labels, predictions, and classification_report
        """
        if model_path is not None:
            print(f"\n🔄 Loading model from {model_path}")
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path).to(self.device)
        
        test_dataset = EmergencyDataset(
            test_data['text'].values,
            test_data['encoded_label'].values,
            self.tokenizer,
            max_len
        )
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="   Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Decode labels back to strings
        true_labels = [self.label_decoder[str(l)] for l in all_labels]
        predictions = [self.label_decoder[str(p)] for p in all_preds]
        
        # Classification report
        label_names = list(self.label_encoder.keys())
        report = classification_report(true_labels, predictions, labels=label_names)
        accuracy = accuracy_score(true_labels, predictions)
        
        print(f"\n{'='*60}")
        print(f"📋 TEST RESULTS")
        print(f"{'='*60}")
        print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"\n{report}")
        
        return {
            'true_labels': true_labels,
            'predictions': predictions,
            'accuracy': accuracy,
            'report': report
        }
    
    # ------------------------------------------
    # VISUALIZATION
    # ------------------------------------------
    
    def plot_training_history(self, save_path=None):
        """Plot training loss, validation loss, and accuracy curves."""
        if save_path is None:
            save_path = os.path.join(MODELS_DIR, 'training_history.png')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs_range = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs_range, self.training_history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(epochs_range, self.training_history['val_loss'], label='Val Loss', marker='o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(epochs_range, self.training_history['val_acc'], label='Val Accuracy', marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ Training plot saved to {save_path}")
    
    def plot_confusion_matrix(self, true_labels, predictions, save_path=None):
        """Plot and save a confusion matrix heatmap."""
        if save_path is None:
            save_path = os.path.join(MODELS_DIR, 'confusion_matrix.png')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        labels = list(self.label_encoder.keys())
        cm = confusion_matrix(true_labels, predictions, labels=labels)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('ERAS Emergency Classification - Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ Confusion matrix saved to {save_path}")
    
    # ------------------------------------------
    # MODEL EXPORT
    # ------------------------------------------
    
    def save_model_and_tokenizer(self, path=None):
        """Save model, tokenizer, and label encoders for deployment."""
        if path is None:
            path = os.path.join(MODELS_DIR, 'eras_distilbert_final')
        
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save label encoder
        with open(os.path.join(path, 'label_encoder.json'), 'w') as f:
            json.dump({
                'label_encoder': self.label_encoder,
                'label_decoder': self.label_decoder
            }, f, indent=2)
        
        print(f"✅ Model and tokenizer saved to {path}")
        
        # Save training config
        config = {
            'model_name': self.model_name,
            'num_labels': len(self.label_encoder),
            'labels': self.label_encoder,
            'training_history': self.training_history,
            'best_val_acc': max(self.training_history['val_acc']) if self.training_history['val_acc'] else 0,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(path, 'training_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Training config saved to {path}/training_config.json")


# ============================================
# MAIN TRAINING PIPELINE
# ============================================

if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print("🚀 ERAS - DISTILBERT TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Initialize trainer
    trainer = ERASDistilBERTTrainer()
    
    # 2. Prepare data (uses the default DATA_PATH)
    train_data, val_data, test_data = trainer.prepare_data()
    
    # 3. Initialize model
    trainer.initialize_model()
    
    # 4. Train model
    best_model_path = trainer.train(
        train_data,
        val_data,
        epochs=5,              # Adjust based on your dataset size
        batch_size=16,          # Reduce to 8 if GPU memory issues
        learning_rate=2e-5,     # Standard for DistilBERT fine-tuning
        max_len=128             # Your texts are short, 128 is sufficient
    )
    
    # 5. Test model
    test_results = trainer.test_model(test_data, best_model_path)
    
    # 6. Plot training history
    trainer.plot_training_history()
    
    # 7. Plot confusion matrix
    trainer.plot_confusion_matrix(
        test_results['true_labels'],
        test_results['predictions']
    )
    
    # 8. Save final model
    trainer.save_model_and_tokenizer()
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE! Model ready for deployment.")
    print("=" * 60)
