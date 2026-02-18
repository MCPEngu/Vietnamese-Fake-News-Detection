"""
Run Encoding Pipeline

Chạy toàn bộ encoding pipeline:
1. TF-IDF encoding (text_tfidf → tfidf_encoded.npz)
2. PhoBERT v2 embeddings (pretrained + finetuned)
3. ViSoBERT embeddings (pretrained + finetuned)
4. PhoBERT Large embeddings (pretrained + finetuned)

Output structure:
    data/encoded/
    ├── tfidf_encoded.npz          # TF-IDF vectors
    ├── phobertv2/
    │   ├── pretrained.npz         # PhoBERT v2 pretrained
    │   └── finetuned.npz          # PhoBERT v2 finetuned
    ├── visobert/
    │   ├── pretrained.npz         # ViSoBERT pretrained
    │   └── finetuned.npz          # ViSoBERT finetuned
    ├── phobertlarge/
    │   ├── pretrained.npz         # PhoBERT Large pretrained
    │   └── finetuned.npz          # PhoBERT Large finetuned
    ├── features.csv               # ID + label + 8 features
    └── manifest.json              # Embedding info

Mỗi file .npz chứa:
- ids: Sample IDs để mapping
- embeddings: Vectors
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.s3_encoding.f1_tfidf import TfidfEncoder
from src.s3_encoding.f2_phobertv2 import PhoBertV2Encoder
from src.s3_encoding.f3_visobert import ViSoBertEncoder
from src.s3_encoding.f4_phobertlarge import PhoBertLargeEncoder


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ENCODED_DIR = DATA_DIR / "encoded"
MODELS_DIR = BASE_DIR / "models"

# Input file
INPUT_FILE = PROCESSED_DIR / "02_final.csv"

# Selected features (from feature analysis)
SELECTED_FEATURES = [
    'feat_real_ratio',
    'feat_num_exclamation',
    'feat_avg_word_length',
    'feat_num_question',
    'feat_is_evening',
    'feat_num_urls',
    'feat_digit_ratio',
    'feat_num_sentences'
]

# TF-IDF configuration
TFIDF_CONFIG = {
    'use_svd': True,
    'variance_threshold': 0.85,  # Giữ 85% variance (optimal knee point)
}


# =============================================================================
# ENCODING FUNCTIONS
# =============================================================================

def create_tfidf_encoding(df: pd.DataFrame, output_dir: Path, models_dir: Path) -> dict:
    """
    Create TF-IDF encoding with IDs
    
    Returns:
        dict with 'ids' and 'embeddings'
    """
    print("\n" + "="*60)
    print("1. TF-IDF ENCODING")
    print("="*60)
    
    texts = df['text_tfidf'].fillna('').tolist()
    ids = df['id'].values
    
    # Create encoder
    encoder = TfidfEncoder(
        use_svd=TFIDF_CONFIG['use_svd'],
        variance_threshold=TFIDF_CONFIG['variance_threshold']
    )
    
    # Fit and transform
    embeddings = encoder.fit_transform(texts)
    
    # Save as .npz with IDs
    output_path = output_dir / 'tfidf_encoded.npz'
    np.savez(output_path, ids=ids, embeddings=embeddings)
    print(f"\n✓ Saved: {output_path}")
    print(f"  IDs: {ids.shape}, Embeddings: {embeddings.shape}")
    
    # Save encoder model
    encoder.save(str(models_dir / 'tfidf'))
    
    return {'ids': ids, 'embeddings': embeddings}


def create_bert_embeddings(df: pd.DataFrame, output_dir: Path, batch_size: int = 16) -> dict:
    """
    Create all BERT embeddings with IDs
    
    Saves to subfolders:
    - phobertv2/pretrained.npz, phobertv2/finetuned.npz
    - visobert/pretrained.npz, visobert/finetuned.npz
    - phobertlarge/pretrained.npz, phobertlarge/finetuned.npz
    
    Returns:
        dict with all embeddings
    """
    import torch
    
    texts = df['text_bert'].fillna('').tolist()
    ids = df['id'].values
    
    results = {}
    
    # =========================================================================
    # PhoBERT v2
    # =========================================================================
    print("\n" + "="*60)
    print("2. PHOBERT V2 EMBEDDINGS")
    print("="*60)
    
    phobertv2_dir = output_dir / 'phobertv2'
    phobertv2_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-trained
    print("\n--- Pre-trained ---")
    encoder = PhoBertV2Encoder(use_finetuned=False, batch_size=batch_size)
    emb = encoder.extract(texts)
    output_path = phobertv2_dir / 'pretrained.npz'
    np.savez(output_path, ids=ids, embeddings=emb)
    results['phobertv2_pretrained'] = {'ids': ids, 'embeddings': emb}
    print(f"✓ Saved: {output_path} | Shape: {emb.shape}")
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Fine-tuned
    print("\n--- Fine-tuned ---")
    encoder = PhoBertV2Encoder(use_finetuned=True, batch_size=batch_size)
    emb = encoder.extract(texts)
    output_path = phobertv2_dir / 'finetuned.npz'
    np.savez(output_path, ids=ids, embeddings=emb)
    results['phobertv2_finetuned'] = {'ids': ids, 'embeddings': emb}
    print(f"✓ Saved: {output_path} | Shape: {emb.shape}")
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # =========================================================================
    # ViSoBERT
    # =========================================================================
    print("\n" + "="*60)
    print("3. VISOBERT EMBEDDINGS")
    print("="*60)
    
    visobert_dir = output_dir / 'visobert'
    visobert_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-trained
    print("\n--- Pre-trained ---")
    encoder = ViSoBertEncoder(use_finetuned=False, batch_size=batch_size)
    emb = encoder.extract(texts)
    output_path = visobert_dir / 'pretrained.npz'
    np.savez(output_path, ids=ids, embeddings=emb)
    results['visobert_pretrained'] = {'ids': ids, 'embeddings': emb}
    print(f"✓ Saved: {output_path} | Shape: {emb.shape}")
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Fine-tuned
    print("\n--- Fine-tuned ---")
    encoder = ViSoBertEncoder(use_finetuned=True, batch_size=batch_size)
    emb = encoder.extract(texts)
    output_path = visobert_dir / 'finetuned.npz'
    np.savez(output_path, ids=ids, embeddings=emb)
    results['visobert_finetuned'] = {'ids': ids, 'embeddings': emb}
    print(f"✓ Saved: {output_path} | Shape: {emb.shape}")
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # =========================================================================
    # PhoBERT Large
    # =========================================================================
    print("\n" + "="*60)
    print("4. PHOBERT LARGE EMBEDDINGS")
    print("="*60)
    
    phobertlarge_dir = output_dir / 'phobertlarge'
    phobertlarge_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-trained (smaller batch for large model)
    print("\n--- Pre-trained ---")
    encoder = PhoBertLargeEncoder(use_finetuned=False, batch_size=4)
    emb = encoder.extract(texts)
    output_path = phobertlarge_dir / 'pretrained.npz'
    np.savez(output_path, ids=ids, embeddings=emb)
    results['phobertlarge_pretrained'] = {'ids': ids, 'embeddings': emb}
    print(f"✓ Saved: {output_path} | Shape: {emb.shape}")
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Fine-tuned
    print("\n--- Fine-tuned ---")
    encoder = PhoBertLargeEncoder(use_finetuned=True, batch_size=4)
    emb = encoder.extract(texts)
    output_path = phobertlarge_dir / 'finetuned.npz'
    np.savez(output_path, ids=ids, embeddings=emb)
    results['phobertlarge_finetuned'] = {'ids': ids, 'embeddings': emb}
    print(f"✓ Saved: {output_path} | Shape: {emb.shape}")
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results


def create_features_csv(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Create features.csv with ID, label, and selected features
    """
    print("\n" + "="*60)
    print("5. CREATING FEATURES CSV")
    print("="*60)
    
    columns = ['id', 'label'] + SELECTED_FEATURES
    df_features = df[columns].copy()
    
    df_features.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    print(f"  Shape: {df_features.shape}")
    print(f"  Columns: {list(df_features.columns)}")
    
    return df_features


def create_manifest(output_dir: Path, n_samples: int, tfidf_dim: int) -> dict:
    """Create embedding manifest file"""
    print("\n" + "="*60)
    print("6. CREATING MANIFEST")
    print("="*60)
    
    manifest = {
        'created_at': datetime.now().isoformat(),
        'n_samples': n_samples,
        'features': {
            'file': 'features.csv',
            'columns': ['id', 'label'] + SELECTED_FEATURES,
            'n_features': len(SELECTED_FEATURES)
        },
        'embeddings': {
            'tfidf': {
                'file': 'tfidf_encoded.npz',
                'dimension': tfidf_dim,
                'variance_threshold': TFIDF_CONFIG['variance_threshold']
            },
            'phobertv2_pretrained': {
                'file': 'phobertv2/pretrained.npz',
                'dimension': 768
            },
            'phobertv2_finetuned': {
                'file': 'phobertv2/finetuned.npz',
                'dimension': 768
            },
            'visobert_pretrained': {
                'file': 'visobert/pretrained.npz',
                'dimension': 768
            },
            'visobert_finetuned': {
                'file': 'visobert/finetuned.npz',
                'dimension': 768
            },
            'phobertlarge_pretrained': {
                'file': 'phobertlarge/pretrained.npz',
                'dimension': 1024
            },
            'phobertlarge_finetuned': {
                'file': 'phobertlarge/finetuned.npz',
                'dimension': 1024
            }
        }
    }
    
    # Save manifest
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved manifest: {manifest_path}")
    
    return manifest


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_encoding_pipeline(skip_bert: bool = False):
    """
    Run complete encoding pipeline
    
    Args:
        skip_bert: Skip BERT embeddings (only do TF-IDF)
    """
    print("\n" + "#"*60)
    print("#" + " "*20 + "ENCODING PIPELINE" + " "*21 + "#")
    print("#"*60)
    
    # Create output directory
    ENCODED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # 1. TF-IDF encoding
    tfidf_result = create_tfidf_encoding(df, ENCODED_DIR, MODELS_DIR)
    tfidf_dim = tfidf_result['embeddings'].shape[1]
    
    # 2. BERT embeddings
    if not skip_bert:
        bert_results = create_bert_embeddings(df, ENCODED_DIR)
    else:
        print("\n⚠ Skipping BERT embeddings (skip_bert=True)")
        bert_results = {}
    
    # 3. Create features CSV
    df_features = create_features_csv(df, ENCODED_DIR / 'features.csv')
    
    # 4. Create manifest
    manifest = create_manifest(ENCODED_DIR, len(df), tfidf_dim)
    
    # Summary
    print("\n" + "#"*60)
    print("#" + " "*20 + "PIPELINE COMPLETE!" + " "*20 + "#")
    print("#"*60)
    
    print(f"\nOutput directory: {ENCODED_DIR}")
    print(f"\nFiles created:")
    print(f"  ├── features.csv: {len(df)} rows × {len(df_features.columns)} columns")
    print(f"  ├── tfidf_encoded.npz: ({len(df)}, {tfidf_dim})")
    
    if not skip_bert:
        print(f"  ├── phobertv2/")
        print(f"  │   ├── pretrained.npz: ({len(df)}, 768)")
        print(f"  │   └── finetuned.npz: ({len(df)}, 768)")
        print(f"  ├── visobert/")
        print(f"  │   ├── pretrained.npz: ({len(df)}, 768)")
        print(f"  │   └── finetuned.npz: ({len(df)}, 768)")
        print(f"  ├── phobertlarge/")
        print(f"  │   ├── pretrained.npz: ({len(df)}, 1024)")
        print(f"  │   └── finetuned.npz: ({len(df)}, 1024)")
    
    print(f"  └── manifest.json")
    
    return df_features, manifest


def run_tfidf_only():
    """Run only TF-IDF encoding (skip BERT)"""
    return run_encoding_pipeline(skip_bert=True)


# =============================================================================
# UTILITY: LOAD EMBEDDINGS
# =============================================================================

def load_embedding(encoded_dir: Path, name: str) -> dict:
    """
    Load a single embedding file
    
    Args:
        encoded_dir: Path to encoded directory
        name: Embedding name (e.g., 'tfidf', 'phobertv2_pretrained', 'visobert_finetuned')
    
    Returns:
        dict with 'ids' and 'embeddings'
    """
    # Map name to file path
    file_map = {
        'tfidf': 'tfidf_encoded.npz',
        'phobertv2_pretrained': 'phobertv2/pretrained.npz',
        'phobertv2_finetuned': 'phobertv2/finetuned.npz',
        'visobert_pretrained': 'visobert/pretrained.npz',
        'visobert_finetuned': 'visobert/finetuned.npz',
        'phobertlarge_pretrained': 'phobertlarge/pretrained.npz',
        'phobertlarge_finetuned': 'phobertlarge/finetuned.npz'
    }
    
    if name not in file_map:
        raise ValueError(f"Unknown embedding: {name}. Valid: {list(file_map.keys())}")
    
    file_path = encoded_dir / file_map[name]
    data = np.load(file_path)
    
    return {
        'ids': data['ids'],
        'embeddings': data['embeddings']
    }


def load_all_embeddings(encoded_dir: str = None, names: list = None) -> dict:
    """
    Load multiple embeddings
    
    Args:
        encoded_dir: Path to encoded directory
        names: List of embedding names to load. If None, load all.
    
    Returns:
        dict with embedding data and features
    """
    if encoded_dir is None:
        encoded_dir = ENCODED_DIR
    else:
        encoded_dir = Path(encoded_dir)
    
    # Default: load all
    if names is None:
        names = [
            'tfidf',
            'phobertv2_pretrained', 'phobertv2_finetuned',
            'visobert_pretrained', 'visobert_finetuned',
            'phobertlarge_pretrained', 'phobertlarge_finetuned'
        ]
    
    # Load manifest
    with open(encoded_dir / 'manifest.json', 'r') as f:
        manifest = json.load(f)
    
    result = {
        'manifest': manifest,
        'features': pd.read_csv(encoded_dir / 'features.csv')
    }
    
    # Load embeddings
    for name in names:
        try:
            data = load_embedding(encoded_dir, name)
            result[name] = data
            print(f"✓ Loaded {name}: IDs {data['ids'].shape}, Embeddings {data['embeddings'].shape}")
        except Exception as e:
            print(f"⚠ Could not load {name}: {e}")
    
    return result


def get_combined_features(encoded_dir: str = None, 
                          include_tfidf: bool = True,
                          bert_models: list = None,
                          use_finetuned: bool = True) -> tuple:
    """
    Get combined feature matrix for training
    
    All embeddings are aligned by ID to ensure correct matching.
    
    Args:
        encoded_dir: Directory containing encoded data
        include_tfidf: Include TF-IDF features
        bert_models: List of BERT model names (e.g., ['phobertv2', 'visobert'])
                     If None, include all models
        use_finetuned: Use finetuned (True) or pretrained (False) versions
    
    Returns:
        X: Combined feature matrix
        y: Labels
        ids: Sample IDs
        feature_info: Dictionary with feature dimensions
    """
    if encoded_dir is None:
        encoded_dir = ENCODED_DIR
    else:
        encoded_dir = Path(encoded_dir)
    
    # Default: use all models
    if bert_models is None:
        bert_models = ['phobertv2', 'visobert', 'phobertlarge']
    
    # Build embedding names
    suffix = 'finetuned' if use_finetuned else 'pretrained'
    embedding_names = []
    if include_tfidf:
        embedding_names.append('tfidf')
    for model in bert_models:
        embedding_names.append(f"{model}_{suffix}")
    
    # Load features (has ID and label)
    df_features = pd.read_csv(encoded_dir / 'features.csv')
    ids = df_features['id'].values
    y = df_features['label'].values
    X_tabular = df_features[SELECTED_FEATURES].values
    
    feature_info = {
        'tabular_features': len(SELECTED_FEATURES)
    }
    
    # Load and align embeddings by ID
    X_parts = [X_tabular]
    
    for name in embedding_names:
        data = load_embedding(encoded_dir, name)
        emb_ids = data['ids']
        emb = data['embeddings']
        
        # Verify IDs match
        if not np.array_equal(ids, emb_ids):
            raise ValueError(f"IDs mismatch between features and {name}")
        
        X_parts.append(emb)
        feature_info[name] = emb.shape[1]
    
    # Concatenate
    X = np.hstack(X_parts)
    feature_info['total'] = X.shape[1]
    
    print(f"\n✓ Combined features: {X.shape}")
    print(f"  Feature breakdown:")
    for k, v in feature_info.items():
        print(f"    - {k}: {v}")
    
    return X, y, ids, feature_info


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run encoding pipeline')
    parser.add_argument('--skip-bert', action='store_true', 
                        help='Skip BERT embeddings (only TF-IDF)')
    args = parser.parse_args()
    
    run_encoding_pipeline(skip_bert=args.skip_bert)
