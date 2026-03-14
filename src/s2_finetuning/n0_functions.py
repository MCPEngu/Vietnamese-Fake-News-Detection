"""
BERT Fine-tuning Module for Vietnamese Fake News Detection

Supports:
- vinai/phobert-base
- vinai/phobert-large-v2  
- uitnlp/visobert

Optimized for RTX 3050 4GB VRAM
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModel,
    get_linear_schedule_with_warmup,
    XLMRobertaTokenizer,
    RobertaTokenizer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# AVAILABLE MODELS
# =============================================================================

AVAILABLE_MODELS = {
    'phobert-v2': {
        'name': 'vinai/phobert-base-v2',
        'max_length': 256,
        'batch_size': 16,
        'description': 'PhoBERT v2 - 135M params'
    },
    'visobert': {
        'name': 'uitnlp/visobert',
        'max_length': 256,
        'batch_size': 16,
        'description': 'ViSoBERT - 110M params'
    },
    'phobert-large': {
        'name': 'vinai/phobert-large',
        'max_length': 256,
        'batch_size': 4,  # Model lớn, giảm batch để fit 4GB VRAM
        'description': 'PhoBERT Large - 370M params'
    }
}

# =============================================================================
# TRAINING DEFAULTS (dùng chung cho cả 3 models)
# =============================================================================
# - epochs: 3
# - learning_rate: 2e-5
# - warmup_ratio: 0.1
# - max_length: 256
# - Nếu cần thay đổi, sửa trong notebook hoặc truyền vào train()


# =============================================================================
# DATASET CLASS
# =============================================================================

class FakeNewsDataset(Dataset):
    """Dataset for Vietnamese Fake News Classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]) if pd.notna(self.texts[idx]) else ""
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# =============================================================================
# FINE-TUNER CLASS
# =============================================================================

class BertFineTuner:
    """
    Fine-tuner for BERT models on Vietnamese Fake News Detection
    
    Usage:
        tuner = BertFineTuner('phobert-base')
        tuner.load_data('data/processed/02_final.csv')
        tuner.train(epochs=3)
        tuner.save('models/bert_embedding/phobert-base-finetuned')
    """
    
    def __init__(self, model_key: str, device: str = None):
        """
        Initialize fine-tuner
        
        Args:
            model_key: Key from AVAILABLE_MODELS ('phobert-base', 'phobert-large-v2', 'visobert')
            device: 'cuda' or 'cpu'. Auto-detect if None.
        """
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Model must be one of: {list(AVAILABLE_MODELS.keys())}")
        
        self.model_key = model_key
        self.model_config = AVAILABLE_MODELS[model_key]
        self.model_name = self.model_config['name']
        self.max_length = self.model_config['max_length']
        self.batch_size = self.model_config['batch_size']
        
        # Device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"\nCUDA is available - using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                print("\nCUDA is not available - using CPU")
        else:
            self.device = torch.device(device)
        
        print(f"{'='*60}")
        print(f"BERT Fine-Tuner: {self.model_config['description']}")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Max Length: {self.max_length}")
        print(f"Batch Size: {self.batch_size}")
        
        # Load tokenizer and model
        print(f"\nLoading tokenizer and model...")
        # ViSoBERT dùng XLMRobertaTokenizer, PhoBERT dùng RobertaTokenizer
        if 'visobert' in self.model_name:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        self.model.to(self.device)
        
        print(f"✓ Model loaded!")
        
        # Data placeholders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    def load_data(self, 
                  data_path: str,
                  text_col: str = 'text_bert',
                  label_col: str = 'label',
                  test_size: float = 0.2,
                  val_size: float = 0.1,
                  random_state: int = 42):
        """
        Load and prepare data for training
        
        Args:
            data_path: Path to CSV file
            text_col: Column name for text
            label_col: Column name for labels
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining after test)
            random_state: Random seed
        """
        print(f"\nLoading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Filter out empty texts
        df = df[df[text_col].notna() & (df[text_col] != '')]
        
        texts = df[text_col].values
        labels = df[label_col].values
        
        print(f"Total samples: {len(texts)}")
        print(f"Label distribution: 0={sum(labels==0)}, 1={sum(labels==1)}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"\nData splits:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        
        # Create datasets
        train_dataset = FakeNewsDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = FakeNewsDataset(X_val, y_val, self.tokenizer, self.max_length)
        test_dataset = FakeNewsDataset(X_test, y_test, self.tokenizer, self.max_length)
        
        # Create dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"✓ Data loaded!")
    
    def train(self,
              epochs: int = 3,
              learning_rate: float = 2e-5,
              warmup_ratio: float = 0.1,
              gradient_accumulation_steps: int = 1,
              save_best: bool = True):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_ratio: Ratio of warmup steps
            gradient_accumulation_steps: Accumulate gradients over N steps (for larger effective batch)
            save_best: Save best model based on validation F1
        """
        if self.train_loader is None:
            raise ValueError("Please call load_data() first!")
        
        print(f"\n{'='*60}")
        print(f"TRAINING")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Effective Batch Size: {self.batch_size * gradient_accumulation_steps}")
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Scheduler
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_f1 = 0
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            
            # Training phase
            self.model.train()
            train_loss = 0
            optimizer.zero_grad()
            
            pbar = tqdm(self.train_loader, desc=f"Training")
            for step, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
                train_loss += outputs.loss.item()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                pbar.set_postfix({'loss': f'{loss.item()*gradient_accumulation_steps:.4f}'})
            
            avg_train_loss = train_loss / len(self.train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            val_loss, val_acc, val_f1 = self._evaluate(self.val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            # Save best model
            if save_best and val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"✓ New best model! F1: {best_f1:.4f}")
        
        # Load best model
        if save_best and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)
            print(f"\n✓ Loaded best model with F1: {best_f1:.4f}")
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"{'='*60}")
    
    def _evaluate(self, dataloader):
        """Evaluate model on a dataloader"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, accuracy, f1
    
    def evaluate_test(self):
        """Evaluate on test set and return detailed metrics"""
        if self.test_loader is None:
            raise ValueError("No test data loaded!")
        
        print(f"\n{'='*60}")
        print(f"TEST SET EVALUATION")
        print(f"{'='*60}")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
            'precision_macro': precision_score(all_labels, all_preds, average='macro'),
            'recall_macro': recall_score(all_labels, all_preds, average='macro'),
            'f1_fake': f1_score(all_labels, all_preds, pos_label=1),
            'f1_real': f1_score(all_labels, all_preds, pos_label=0)
        }
        
        print(f"\nResults:")
        print(f"  Accuracy:       {metrics['accuracy']:.4f}")
        print(f"  F1 (macro):     {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted):  {metrics['f1_weighted']:.4f}")
        print(f"  Precision:      {metrics['precision_macro']:.4f}")
        print(f"  Recall:         {metrics['recall_macro']:.4f}")
        print(f"  F1 (Fake):      {metrics['f1_fake']:.4f}")
        print(f"  F1 (Real):      {metrics['f1_real']:.4f}")
        
        return metrics
    
    def save(self, save_dir: str):
        """
        Save fine-tuned model and tokenizer
        
        Args:
            save_dir: Directory to save model
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving model to: {save_path}")
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save config and history
        config = {
            'model_key': self.model_key,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'history': self.history
        }
        
        with open(save_path / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Model saved!")
        print(f"  - model files: {save_path}")
        print(f"  - config: {save_path / 'training_config.json'}")
    
    def extract_embeddings(self, texts: list, pooling: str = 'cls') -> np.ndarray:
        """
        Extract embeddings from fine-tuned model (KHÔNG qua classification head)
        
        Args:
            texts: List of texts to extract embeddings
            pooling: 'cls' (CLS token) or 'mean' (mean of all tokens)
        
        Returns:
            np.ndarray of shape (n_samples, hidden_dim)
        """
        self.model.eval()
        all_embeddings = []
        
        # Get base model (without classification head)
        # PhoBERT/ViSoBERT dùng RoBERTa architecture
        if hasattr(self.model, 'roberta'):
            base_model = self.model.roberta
        elif hasattr(self.model, 'bert'):
            base_model = self.model.bert
        else:
            raise ValueError("Cannot find base model (roberta/bert)")
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = base_model(input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
                
                if pooling == 'cls':
                    # Lấy CLS token (position 0)
                    embeddings = hidden_states[:, 0, :]
                elif pooling == 'mean':
                    # Mean pooling (chỉ tính trên non-padding tokens)
                    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask, dim=1)
                    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling}")
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def extract_and_save_embeddings(self, 
                                     data_path: str,
                                     output_path: str,
                                     text_col: str = 'text_bert',
                                     pooling: str = 'cls'):
        """
        Extract embeddings từ data và save ra file
        
        Args:
            data_path: Path to input CSV
            output_path: Path to save embeddings (.npy)
            text_col: Column name for text
            pooling: 'cls' or 'mean'
        """
        print(f"\nExtracting embeddings from: {data_path}")
        
        df = pd.read_csv(data_path)
        texts = df[text_col].fillna('').tolist()
        
        embeddings = self.extract_embeddings(texts, pooling=pooling)
        
        # Save embeddings
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings)
        
        print(f"✓ Saved embeddings: {output_path}")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Pooling: {pooling}")
        
        return embeddings
    
    @classmethod
    def load(cls, load_dir: str, device: str = None):
        """
        Load a fine-tuned model
        
        Args:
            load_dir: Directory containing saved model
            device: Device to load model on
        
        Returns:
            Loaded model and tokenizer
        """
        load_path = Path(load_dir)
        
        # Load config
        with open(load_path / 'training_config.json', 'r') as f:
            config = json.load(f)
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check model type from config
        model_name = config.get('model_name', '')
        if 'visobert' in model_name:
            tokenizer = XLMRobertaTokenizer.from_pretrained(load_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(load_path)
        model.to(device)
        
        print(f"✓ Loaded model from: {load_path}")
        print(f"  Model: {config['model_name']}")
        
        return model, tokenizer, config


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_available_models():
    """Print available models"""
    print("Available models for fine-tuning:")
    print("-" * 60)
    for key, config in AVAILABLE_MODELS.items():
        print(f"  {key}:")
        print(f"    - Name: {config['name']}")
        print(f"    - Description: {config['description']}")
        print(f"    - Recommended batch size: {config['batch_size']}")
        print()


if __name__ == "__main__":
    list_available_models()
