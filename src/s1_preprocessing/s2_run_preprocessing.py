"""
Run Preprocessing Pipeline
Chạy toàn bộ pipeline preprocessing và lưu dataset mới
"""

import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.s1A_feature_engineering import create_all_features, get_feature_columns
from src.preprocessing.s1B_text_cleaning import clean_for_tfidf, clean_for_bert


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Input/Output files
INPUT_FILE = DATA_RAW_DIR / "data.csv"
OUTPUT_FILE = DATA_PROCESSED_DIR / "01_preprocessed.csv"


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def load_data(filepath: Path) -> pd.DataFrame:
    """Load raw data"""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_texts(df: pd.DataFrame, text_col: str = 'post_message') -> pd.DataFrame:
    """
    Làm sạch text và tạo 2 cột mới: text_tfidf và text_bert
    """
    print("\nCleaning texts...")
    result = df.copy()
    
    # Dùng tqdm để hiển thị progress
    tqdm.pandas(desc="Cleaning for TF-IDF")
    result['text_tfidf'] = result[text_col].progress_apply(
        lambda x: clean_for_tfidf(x) if pd.notna(x) else ""
    )
    
    tqdm.pandas(desc="Cleaning for BERT")
    result['text_bert'] = result[text_col].progress_apply(
        lambda x: clean_for_bert(x) if pd.notna(x) else ""
    )
    
    print("✓ Created text_tfidf and text_bert columns")
    return result


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo tất cả features"""
    print("\nCreating features...")
    return create_all_features(df)


def save_data(df: pd.DataFrame, filepath: Path):
    """Lưu processed data"""
    # Tạo thư mục nếu chưa tồn tại
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving data to: {filepath}")
    df.to_csv(filepath, index=False)
    print(f"✓ Saved {len(df)} rows, {len(df.columns)} columns")


def print_summary(df: pd.DataFrame):
    """In tóm tắt dataset"""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    # Basic info
    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Label distribution
    if 'label' in df.columns:
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
    
    # Feature columns
    feature_cols = get_feature_columns(df)
    print(f"\nNew feature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")
    
    # Text columns
    text_cols = ['text_tfidf', 'text_bert']
    existing_text_cols = [c for c in text_cols if c in df.columns]
    if existing_text_cols:
        print(f"\nText columns created: {existing_text_cols}")
        
        # Sample
        print("\nSample cleaned text (first row):")
        for col in existing_text_cols:
            sample = df[col].iloc[0][:100] + "..." if len(df[col].iloc[0]) > 100 else df[col].iloc[0]
            print(f"  {col}: {sample}")
    
    print("\n" + "="*60)


def run_preprocessing_pipeline(input_file: Path = None, 
                               output_file: Path = None,
                               text_col: str = 'post_message') -> pd.DataFrame:
    """
    Chạy toàn bộ preprocessing pipeline
    
    Args:
        input_file: Path to input CSV
        output_file: Path to save output CSV
        text_col: Name of text column
        
    Returns:
        Processed DataFrame
    """
    if input_file is None:
        input_file = INPUT_FILE
    if output_file is None:
        output_file = OUTPUT_FILE
    
    print("="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    # 1. Load data
    df = load_data(input_file)
    
    # 2. Clean texts
    df = clean_texts(df, text_col)
    
    # 3. Create features
    df = create_features(df)
    
    # 4. Save
    save_data(df, output_file)
    
    # 5. Summary
    print_summary(df)
    
    print("\n✓ Preprocessing completed!")
    return df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Input CSV file path")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV file path")
    parser.add_argument("--text-col", "-t", type=str, default="post_message",
                        help="Text column name")
    
    args = parser.parse_args()
    
    input_path = Path(args.input) if args.input else INPUT_FILE
    output_path = Path(args.output) if args.output else OUTPUT_FILE
    
    run_preprocessing_pipeline(input_path, output_path, args.text_col)
