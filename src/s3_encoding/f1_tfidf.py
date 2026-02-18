"""
TF-IDF Encoding Module

Chuyển đổi text_tfidf thành vector representation:
- TF-IDF vectorization
- SVD dimensionality reduction với tự động chọn số chiều dựa trên variance threshold

Output: numpy array (n_samples, n_features)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib


# =============================================================================
# CONFIGURATION
# =============================================================================

TFIDF_CONFIG = {
    'max_features': 10000,       # Số features tối đa
    'min_df': 2,                 # Tối thiểu xuất hiện trong 2 documents
    'max_df': 0.95,              # Tối đa 95% documents
    'ngram_range': (1, 2),       # Unigrams và bigrams
    'sublinear_tf': True,        # Dùng 1 + log(tf) thay vì tf
}

SVD_CONFIG = {
    'variance_threshold': 0.85,  # Giữ lại 85% variance (optimal knee point)
    'max_components': 4000,      # Số components tối đa để fit ban đầu
    'random_state': 42,
}


# =============================================================================
# TFIDF ENCODER CLASS
# =============================================================================

class TfidfEncoder:
    """
    TF-IDF Encoder với SVD reduction dựa trên variance threshold
    
    Thay vì chọn cứng số chiều, tự động chọn số chiều nhỏ nhất
    mà vẫn giữ được >= variance_threshold thông tin.
    
    Usage:
        encoder = TfidfEncoder(variance_threshold=0.85)  # Giữ 85% variance
        encoder.fit(train_texts)
        vectors = encoder.transform(texts)
        encoder.save('models/tfidf')
        
        # Load later
        encoder = TfidfEncoder.load('models/tfidf')
    """
    
    def __init__(self, 
                 max_features: int = TFIDF_CONFIG['max_features'],
                 min_df: int = TFIDF_CONFIG['min_df'],
                 max_df: float = TFIDF_CONFIG['max_df'],
                 ngram_range: Tuple[int, int] = TFIDF_CONFIG['ngram_range'],
                 sublinear_tf: bool = TFIDF_CONFIG['sublinear_tf'],
                 use_svd: bool = True,
                 variance_threshold: float = SVD_CONFIG['variance_threshold'],
                 max_components: int = SVD_CONFIG['max_components'],
                 random_state: int = SVD_CONFIG['random_state']):
        """
        Initialize TF-IDF Encoder
        
        Args:
            max_features: Maximum number of features for TF-IDF
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: N-gram range (min, max)
            sublinear_tf: Use sublinear TF scaling
            use_svd: Apply SVD dimensionality reduction
            variance_threshold: Minimum variance to retain (0.0 to 1.0)
                              VD: 0.95 = giữ 95% thông tin
            max_components: Maximum SVD components to fit initially
            random_state: Random seed for reproducibility
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.use_svd = use_svd
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.random_state = random_state
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            dtype=np.float32
        )
        
        # SVD will be initialized during fit
        self.svd = None
        self.n_components_selected = None  # Số chiều được chọn sau khi fit
        
        self.is_fitted = False
        
        print("="*60)
        print("TF-IDF Encoder Initialized")
        print("="*60)
        print(f"Max features: {max_features}")
        print(f"N-gram range: {ngram_range}")
        print(f"Min DF: {min_df}, Max DF: {max_df}")
        print(f"Sublinear TF: {sublinear_tf}")
        print(f"Use SVD: {use_svd}")
        if use_svd:
            print(f"Variance threshold: {variance_threshold*100:.0f}%")
            print(f"Max components: {max_components}")
    
    def _find_optimal_components(self, explained_variance_ratio: np.ndarray) -> int:
        """
        Tìm số components tối thiểu để đạt variance threshold
        
        Args:
            explained_variance_ratio: Array of variance ratios per component
            
        Returns:
            Số components cần thiết
        """
        cumsum = np.cumsum(explained_variance_ratio)
        n_components = np.searchsorted(cumsum, self.variance_threshold) + 1
        return min(n_components, len(explained_variance_ratio))
    
    def fit(self, texts: list) -> 'TfidfEncoder':
        """
        Fit encoder on training texts
        
        Args:
            texts: List of texts to fit on
            
        Returns:
            self
        """
        print(f"\nFitting TF-IDF on {len(texts)} texts...")
        
        # Fit TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        n_samples, n_features = tfidf_matrix.shape
        print(f"✓ TF-IDF fitted: {tfidf_matrix.shape}")
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        # Fit SVD if needed
        if self.use_svd:
            print(f"\nFitting SVD to find optimal dimensions...")
            
            # Số components tối đa có thể
            max_possible = min(n_samples - 1, n_features - 1, self.max_components)
            
            # Fit SVD với nhiều components
            svd_full = TruncatedSVD(
                n_components=max_possible,
                random_state=self.random_state
            )
            svd_full.fit(tfidf_matrix)
            
            # Tìm số components tối ưu
            self.n_components_selected = self._find_optimal_components(
                svd_full.explained_variance_ratio_
            )
            
            # Tính variance với số components đã chọn
            cumsum = np.cumsum(svd_full.explained_variance_ratio_)
            actual_variance = cumsum[self.n_components_selected - 1]
            
            print(f"\n📊 SVD Analysis:")
            print(f"  - Initial fit: {max_possible} components")
            print(f"  - Target variance: {self.variance_threshold*100:.0f}%")
            print(f"  - Selected components: {self.n_components_selected}")
            print(f"  - Actual variance retained: {actual_variance*100:.2f}%")
            
            # Fit SVD mới với số components đã chọn
            self.svd = TruncatedSVD(
                n_components=self.n_components_selected,
                random_state=self.random_state
            )
            self.svd.fit(tfidf_matrix)
            
            # Print variance breakdown
            print(f"\n  Top 10 components variance contribution:")
            for i in range(min(10, self.n_components_selected)):
                var = self.svd.explained_variance_ratio_[i] * 100
                cumvar = np.cumsum(self.svd.explained_variance_ratio_)[i] * 100
                print(f"    Component {i+1}: {var:.2f}% | Cumulative: {cumvar:.2f}%")
        
        self.is_fitted = True
        return self
    
    def transform(self, texts: list) -> np.ndarray:
        """
        Transform texts to vectors
        
        Args:
            texts: List of texts to transform
            
        Returns:
            np.ndarray of shape (n_texts, n_components_selected)
        """
        if not self.is_fitted:
            raise ValueError("Encoder not fitted! Call fit() first.")
        
        # Transform with TF-IDF
        tfidf_matrix = self.vectorizer.transform(texts)
        
        # Apply SVD if needed
        if self.use_svd:
            vectors = self.svd.transform(tfidf_matrix)
        else:
            vectors = tfidf_matrix.toarray()
        
        return vectors.astype(np.float32)
    
    def fit_transform(self, texts: list) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            texts: List of texts
            
        Returns:
            np.ndarray of shape (n_texts, n_components_selected)
        """
        self.fit(texts)
        return self.transform(texts)
    
    def save(self, save_dir: str):
        """
        Save encoder to disk
        
        Args:
            save_dir: Directory to save encoder
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving TF-IDF encoder to: {save_path}")
        
        # Save vectorizer
        joblib.dump(self.vectorizer, save_path / 'tfidf_vectorizer.joblib')
        
        # Save SVD if exists
        if self.svd is not None:
            joblib.dump(self.svd, save_path / 'svd.joblib')
        
        # Save config
        config = {
            'max_features': int(self.max_features),
            'min_df': int(self.min_df),
            'max_df': float(self.max_df),
            'ngram_range': list(self.ngram_range),
            'sublinear_tf': bool(self.sublinear_tf),
            'use_svd': bool(self.use_svd),
            'variance_threshold': float(self.variance_threshold),
            'n_components_selected': int(self.n_components_selected) if self.n_components_selected else None,
            'random_state': int(self.random_state),
            'vocabulary_size': int(len(self.vectorizer.vocabulary_))
        }
        
        import json
        with open(save_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Encoder saved!")
        print(f"  - Selected dimensions: {self.n_components_selected}")
    
    @classmethod
    def load(cls, load_dir: str) -> 'TfidfEncoder':
        """
        Load encoder from disk
        
        Args:
            load_dir: Directory containing saved encoder
            
        Returns:
            Loaded TfidfEncoder
        """
        load_path = Path(load_dir)
        
        # Load config
        import json
        with open(load_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Create encoder with same config
        encoder = cls(
            max_features=config['max_features'],
            min_df=config['min_df'],
            max_df=config['max_df'],
            ngram_range=tuple(config['ngram_range']),
            sublinear_tf=config['sublinear_tf'],
            use_svd=config['use_svd'],
            variance_threshold=config['variance_threshold'],
            random_state=config['random_state']
        )
        
        # Load vectorizer
        encoder.vectorizer = joblib.load(load_path / 'tfidf_vectorizer.joblib')
        
        # Load SVD if exists
        if encoder.use_svd:
            encoder.svd = joblib.load(load_path / 'svd.joblib')
            encoder.n_components_selected = config['n_components_selected']
        
        encoder.is_fitted = True
        
        print(f"✓ Loaded TF-IDF encoder from: {load_path}")
        print(f"  Vocabulary size: {config['vocabulary_size']}")
        if config['use_svd']:
            print(f"  Selected dimensions: {config['n_components_selected']}")
            print(f"  Variance threshold: {config['variance_threshold']*100:.0f}%")
        
        return encoder
    
    def get_feature_names(self) -> list:
        """Get feature names (TF-IDF vocabulary)"""
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_output_dim(self) -> int:
        """Get output dimension"""
        if self.use_svd and self.n_components_selected:
            return self.n_components_selected
        return self.max_features


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_tfidf_encoding(data_path: str,
                          output_path: str,
                          model_save_path: str,
                          text_col: str = 'text_tfidf',
                          id_col: str = 'id',
                          use_svd: bool = True,
                          variance_threshold: float = 0.85) -> dict:
    """
    Create TF-IDF encoding từ data file
    
    Args:
        data_path: Path to input CSV
        output_path: Path to save embeddings (.npz)
        model_save_path: Path to save encoder model
        text_col: Column name for text
        id_col: Column name for sample IDs
        use_svd: Whether to use SVD reduction
        variance_threshold: Minimum variance to retain (default 0.85 = 85%)
        
    Returns:
        dict with 'ids' and 'embeddings'
    """
    print("\n" + "="*60)
    print("CREATING TF-IDF ENCODING")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    texts = df[text_col].fillna('').tolist()
    ids = df[id_col].values
    print(f"Loaded {len(texts)} texts")
    
    # Create and fit encoder
    encoder = TfidfEncoder(use_svd=use_svd, variance_threshold=variance_threshold)
    vectors = encoder.fit_transform(texts)
    
    print(f"\n✓ Created TF-IDF vectors: {vectors.shape}")
    
    # Save as .npz with IDs
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Change extension to .npz if needed
    if output_path.suffix == '.npy':
        output_path = output_path.with_suffix('.npz')
    
    np.savez(output_path, ids=ids, embeddings=vectors)
    print(f"✓ Saved to: {output_path}")
    print(f"  - IDs: {ids.shape}")
    print(f"  - Embeddings: {vectors.shape}")
    
    # Save encoder
    encoder.save(model_save_path)
    
    return {'ids': ids, 'embeddings': vectors}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_PATH = BASE_DIR / "data" / "processed" / "02_final.csv"
    OUTPUT_PATH = BASE_DIR / "data" / "encoded" / "tfidf_encoded.npz"
    MODEL_PATH = BASE_DIR / "models" / "tfidf"
    
    # Create encoding với 85% variance threshold
    result = create_tfidf_encoding(
        data_path=str(DATA_PATH),
        output_path=str(OUTPUT_PATH),
        model_save_path=str(MODEL_PATH),
        text_col='text_tfidf',
        id_col='id',
        use_svd=True,
        variance_threshold=0.85  # Giữ 85% thông tin (optimal knee point)
    )
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"IDs shape: {result['ids'].shape}")
    print(f"Embeddings shape: {result['embeddings'].shape}")
