"""
PhoBERT Large Embedding Extraction Module

Trích xuất embeddings từ:
- PhoBERT Large pre-trained (vinai/phobert-large)
- PhoBERT Large fine-tuned (models/bert_embedding/phobert-large-finetuned)

Output: numpy array (n_samples, 1024)
Note: PhoBERT Large có hidden_size = 1024 (lớn hơn base model 768)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    'pretrained_name': 'vinai/phobert-large',
    'finetuned_path': 'models/bert_embedding/phobert-large-finetuned',
    'max_length': 256,
    'batch_size': 4,  # Batch nhỏ hơn vì model lớn
    'hidden_size': 1024,  # PhoBERT Large có 1024 hidden units
}


# =============================================================================
# PHOBERT LARGE ENCODER CLASS
# =============================================================================

class PhoBertLargeEncoder:
    """
    PhoBERT Large Embedding Extractor
    
    Hỗ trợ cả pre-trained và fine-tuned models
    
    Usage:
        # Pre-trained
        encoder = PhoBertLargeEncoder(use_finetuned=False)
        embeddings = encoder.extract(texts)
        
        # Fine-tuned
        encoder = PhoBertLargeEncoder(use_finetuned=True)
        embeddings = encoder.extract(texts)
    
    Note: Output dimension is 1024 (not 768 like base models)
    """
    
    def __init__(self, 
                 use_finetuned: bool = False,
                 finetuned_path: Optional[str] = None,
                 max_length: int = MODEL_CONFIG['max_length'],
                 batch_size: int = MODEL_CONFIG['batch_size'],
                 device: str = None,
                 pooling: str = 'cls'):
        """
        Initialize PhoBERT Large Encoder
        
        Args:
            use_finetuned: Whether to use fine-tuned model
            finetuned_path: Path to fine-tuned model (if use_finetuned=True)
            max_length: Maximum sequence length
            batch_size: Batch size for inference (default 4 for large model)
            device: 'cuda' or 'cpu'. Auto-detect if None.
            pooling: 'cls' (CLS token) or 'mean' (mean pooling)
        """
        self.use_finetuned = use_finetuned
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model path
        if use_finetuned:
            if finetuned_path is None:
                # Default path
                base_dir = Path(__file__).parent.parent.parent
                finetuned_path = base_dir / MODEL_CONFIG['finetuned_path']
            self.model_path = str(finetuned_path)
            self.model_name = "PhoBERT Large (fine-tuned)"
        else:
            self.model_path = MODEL_CONFIG['pretrained_name']
            self.model_name = "PhoBERT Large (pre-trained)"
        
        print("="*60)
        print(f"PhoBERT Large Encoder: {self.model_name}")
        print("="*60)
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"⚠ Warning: Large model requires ~6GB VRAM")
        print(f"Max Length: {max_length}")
        print(f"Batch Size: {batch_size}")
        print(f"Pooling: {pooling}")
        print(f"Output Dimension: {MODEL_CONFIG['hidden_size']}")
        
        # Load tokenizer and model
        print(f"\nLoading model...")
        self._load_model()
        print(f"✓ Model loaded!")
    
    def _load_model(self):
        """Load tokenizer and model"""
        if self.use_finetuned:
            # Fine-tuned model (classification model, extract base)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
            full_model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            
            # Get base model (without classification head)
            if hasattr(full_model, 'roberta'):
                self.model = full_model.roberta
            elif hasattr(full_model, 'bert'):
                self.model = full_model.bert
            else:
                raise ValueError("Cannot find base model")
        else:
            # Pre-trained model
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_CONFIG['pretrained_name'], 
                use_fast=False
            )
            self.model = AutoModel.from_pretrained(MODEL_CONFIG['pretrained_name'])
        
        self.model.to(self.device)
        self.model.eval()
    
    def extract(self, texts: list, show_progress: bool = True) -> np.ndarray:
        """
        Extract embeddings from texts
        
        Args:
            texts: List of texts to extract embeddings from
            show_progress: Show progress bar
            
        Returns:
            np.ndarray of shape (n_texts, 1024)
        """
        all_embeddings = []
        
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting embeddings")
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + self.batch_size]
                
                # Handle None/NaN values
                batch_texts = [str(t) if pd.notna(t) else "" for t in batch_texts]
                
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
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state  # (batch, seq_len, 1024)
                
                # Pooling
                if self.pooling == 'cls':
                    # CLS token (position 0)
                    embeddings = hidden_states[:, 0, :]
                elif self.pooling == 'mean':
                    # Mean pooling (only non-padding tokens)
                    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask, dim=1)
                    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    raise ValueError(f"pooling must be 'cls' or 'mean', got {self.pooling}")
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings).astype(np.float32)
    
    def extract_and_save(self, 
                        data_path: str,
                        output_path: str,
                        text_col: str = 'text_bert',
                        id_col: str = 'id') -> dict:
        """
        Extract embeddings từ data file và save
        
        Args:
            data_path: Path to input CSV
            output_path: Path to save embeddings (.npz)
            text_col: Column name for text
            id_col: Column name for sample IDs
            
        Returns:
            dict with 'ids' and 'embeddings'
        """
        # Load data
        print(f"\nLoading data from: {data_path}")
        df = pd.read_csv(data_path)
        texts = df[text_col].tolist()
        ids = df[id_col].values
        print(f"Loaded {len(texts)} texts")
        
        # Extract embeddings
        embeddings = self.extract(texts)
        
        # Save as .npz with IDs
        output_path = Path(output_path)
        if output_path.suffix == '.npy':
            output_path = output_path.with_suffix('.npz')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, ids=ids, embeddings=embeddings)
        print(f"\n✓ Saved to: {output_path}")
        print(f"  IDs: {ids.shape}")
        print(f"  Embeddings: {embeddings.shape}")
        
        return {'ids': ids, 'embeddings': embeddings}


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_phobertlarge_embeddings(data_path: str,
                                    output_dir: str,
                                    text_col: str = 'text_bert',
                                    id_col: str = 'id',
                                    batch_size: int = 4) -> Tuple[dict, dict]:
    """
    Create both pre-trained and fine-tuned embeddings
    
    Args:
        data_path: Path to input CSV
        output_dir: Directory to save embeddings
        text_col: Column name for text
        id_col: Column name for sample IDs
        batch_size: Batch size for inference (default 4 for large model)
        
    Returns:
        Tuple of (pretrained_result, finetuned_result) dicts
    """
    print("\n" + "="*60)
    print("CREATING PHOBERT LARGE EMBEDDINGS")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data once
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    texts = df[text_col].tolist()
    ids = df[id_col].values
    print(f"Loaded {len(texts)} texts")
    
    # Pre-trained embeddings
    print("\n" + "-"*40)
    print("1. Pre-trained PhoBERT Large")
    print("-"*40)
    encoder_pretrained = PhoBertLargeEncoder(use_finetuned=False, batch_size=batch_size)
    emb_pretrained = encoder_pretrained.extract(texts)
    
    # Create subfolder
    phobertlarge_dir = output_dir / "phobertlarge"
    phobertlarge_dir.mkdir(parents=True, exist_ok=True)
    
    pretrained_path = phobertlarge_dir / "pretrained.npz"
    np.savez(pretrained_path, ids=ids, embeddings=emb_pretrained)
    print(f"✓ Saved: {pretrained_path}")
    print(f"  IDs: {ids.shape}, Embeddings: {emb_pretrained.shape}")
    
    # Clear GPU memory
    del encoder_pretrained
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Fine-tuned embeddings
    print("\n" + "-"*40)
    print("2. Fine-tuned PhoBERT Large")
    print("-"*40)
    encoder_finetuned = PhoBertLargeEncoder(use_finetuned=True, batch_size=batch_size)
    emb_finetuned = encoder_finetuned.extract(texts)
    
    finetuned_path = phobertlarge_dir / "finetuned.npz"
    np.savez(finetuned_path, ids=ids, embeddings=emb_finetuned)
    print(f"✓ Saved: {finetuned_path}")
    print(f"  IDs: {ids.shape}, Embeddings: {emb_finetuned.shape}")
    
    # Clear GPU memory
    del encoder_finetuned
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("\n" + "="*60)
    print("PHOBERT LARGE EMBEDDINGS CREATED!")
    print("="*60)
    
    return (
        {'ids': ids, 'embeddings': emb_pretrained},
        {'ids': ids, 'embeddings': emb_finetuned}
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_PATH = BASE_DIR / "data" / "processed" / "02_final.csv"
    OUTPUT_DIR = BASE_DIR / "data" / "encoded"
    
    # Create embeddings
    result_pre, result_ft = create_phobertlarge_embeddings(
        data_path=str(DATA_PATH),
        output_dir=str(OUTPUT_DIR),
        text_col='text_bert',
        id_col='id',
        batch_size=4
    )
    
    print(f"\nPre-trained: IDs {result_pre['ids'].shape}, Embeddings {result_pre['embeddings'].shape}")
    print(f"Fine-tuned: IDs {result_ft['ids'].shape}, Embeddings {result_ft['embeddings'].shape}")
