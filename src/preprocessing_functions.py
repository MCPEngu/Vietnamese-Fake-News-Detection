"""
Data Preprocessing Functions for Vietnamese Fake News Detection
- Text cleaning for TF-IDF and PhoBERT
- Feature extraction (special characters, emojis, etc.)
- Account violation counting
"""

import pandas as pd
import numpy as np
import re
from collections import Counter

# ============================================
# 1. TEXT CLEANING FUNCTIONS
# ============================================

def clean_for_tfidf(text):
    """
    Lọc text tối ưu cho TF-IDF
    - Loại bỏ URL, hashtag, mention, emoji
    - Loại bỏ ký tự đặc biệt
    - Chuyển về lowercase
    - Giữ lại từ tiếng Việt
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+|< url >', '', text)
    
    # Remove hashtags (#something)
    text = re.sub(r'#\w+', '', text)
    
    # Remove mentions (@something)
    text = re.sub(r'@\w+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # Remove special characters (keep Vietnamese characters and spaces)
    text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lowercase
    text = text.lower()
    
    # Remove underscore (word segmentation marker)
    text = text.replace('_', ' ')
    
    return text


def clean_for_phobert(text):
    """
    Lọc text tối ưu cho PhoBERT
    - Giữ lại cấu trúc câu
    - Loại bỏ URL, HTML
    - Giữ lại một số ký tự đặc biệt quan trọng
    - PhoBERT cần word segmentation (dấu _)
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+|< url >', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # Remove hashtags but keep the text after #
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove mentions but keep the text after @
    text = re.sub(r'@(\w+)', r'\1', text)
    
    # Keep Vietnamese characters, basic punctuation, and underscores (for word segmentation)
    text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ.,!?_]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate to 256 tokens (PhoBERT max length)
    words = text.split()
    if len(words) > 200:
        text = ' '.join(words[:200])
    
    return text


def apply_text_cleaning(df, text_column='post_message'):
    """
    Áp dụng cả 2 hàm cleaning và tạo 2 cột mới
    
    Args:
        df: DataFrame
        text_column: Tên cột chứa text gốc
    
    Returns:
        df: DataFrame với 2 cột mới
    """
    print("[1] Cleaning text for TF-IDF...")
    df['text_tfidf'] = df[text_column].apply(clean_for_tfidf)
    
    print("[2] Cleaning text for PhoBERT...")
    df['text_phobert'] = df[text_column].apply(clean_for_phobert)
    
    print(f"    - Created columns: 'text_tfidf', 'text_phobert'")
    
    return df


# ============================================
# 2. FEATURE EXTRACTION FUNCTIONS
# ============================================

def count_special_features(text):
    """
    Đếm các ký tự đặc biệt trong text
    
    Returns:
        dict: Dictionary chứa các features
    """
    if pd.isna(text):
        text = ""
    text = str(text)
    
    features = {}
    
    # Count question marks (?)
    features['count_question'] = text.count('?')
    
    # Count exclamation marks (!)
    features['count_exclaim'] = text.count('!')
    
    # Count ellipsis (...)
    features['count_ellipsis'] = len(re.findall(r'\.{2,}', text))
    
    # Count URLs
    features['count_url'] = len(re.findall(r'http\S+|www\S+|https\S+|< url >', text))
    
    # Count hashtags (#)
    features['count_hashtag'] = len(re.findall(r'#\w+', text))
    
    # Count mentions (@)
    features['count_mention'] = len(re.findall(r'@\w+', text))
    
    # Count uppercase words
    features['count_uppercase'] = len(re.findall(r'\b[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]{2,}\b', text))
    
    # Count emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        "]+", flags=re.UNICODE)
    features['count_emoji'] = len(emoji_pattern.findall(text))
    
    # Count total words
    words = text.split()
    features['total_words'] = len(words)
    
    # Count total characters
    features['total_chars'] = len(text)
    
    # Ratio of uppercase to total words
    if features['total_words'] > 0:
        features['uppercase_ratio'] = features['count_uppercase'] / features['total_words']
    else:
        features['uppercase_ratio'] = 0
    
    return features


def extract_special_features(df, text_column='post_message'):
    """
    Trích xuất các features đặc biệt từ text và thêm vào DataFrame
    
    Args:
        df: DataFrame
        text_column: Tên cột chứa text
    
    Returns:
        df: DataFrame với các cột feature mới
    """
    print("[3] Extracting special features...")
    
    # Apply counting function
    features_df = df[text_column].apply(count_special_features).apply(pd.Series)
    
    # Add to original DataFrame
    for col in features_df.columns:
        df[col] = features_df[col]
    
    print(f"    - Created columns: {list(features_df.columns)}")
    
    return df


# ============================================
# 3. ACCOUNT VIOLATION COUNTING
# ============================================

def count_account_violations(df, account_column='user_name', label_column='label', fake_label=1):
    """
    Đếm số lần vi phạm (đăng tin sai sự thật) của mỗi tài khoản
    
    Args:
        df: DataFrame
        account_column: Tên cột chứa ID tài khoản
        label_column: Tên cột chứa nhãn (0=real, 1=fake)
        fake_label: Giá trị nhãn cho tin giả (mặc định là 1)
    
    Returns:
        df: DataFrame với cột 'benati' (số lần vi phạm)
    """
    print("[4] Counting account violations...")
    
    # Count fake news per account
    fake_counts = df[df[label_column] == fake_label].groupby(account_column).size()
    fake_counts = fake_counts.reset_index()
    fake_counts.columns = [account_column, 'benati']
    
    # Total posts per account
    total_posts = df.groupby(account_column).size().reset_index()
    total_posts.columns = [account_column, 'total_posts']
    
    # Merge back to original DataFrame
    df = df.merge(fake_counts, on=account_column, how='left')
    df['benati'] = df['benati'].fillna(0).astype(int)
    
    # Add total posts per account
    df = df.merge(total_posts, on=account_column, how='left')
    
    # Calculate violation ratio
    df['violation_ratio'] = df['benati'] / df['total_posts']
    
    # Statistics
    unique_accounts = df[account_column].nunique()
    violating_accounts = df[df['benati'] > 0][account_column].nunique()
    
    print(f"    - Total unique accounts: {unique_accounts}")
    print(f"    - Accounts with violations: {violating_accounts}")
    print(f"    - Created columns: 'benati', 'total_posts', 'violation_ratio'")
    
    return df


# ============================================
# MAIN PROCESSING FUNCTION
# ============================================

def process_all_features(df, text_column='post_message', account_column='user_name', label_column='label'):
    """
    Chạy tất cả các bước xử lý
    
    Args:
        df: DataFrame gốc
        text_column: Tên cột chứa text
        account_column: Tên cột chứa ID tài khoản
        label_column: Tên cột chứa nhãn
    
    Returns:
        df: DataFrame đã xử lý với tất cả features
    """
    print("=" * 50)
    print("PROCESSING ALL FEATURES")
    print("=" * 50)
    
    # 1. Clean text for TF-IDF and PhoBERT
    df = apply_text_cleaning(df, text_column)
    
    # 2. Extract special features
    df = extract_special_features(df, text_column)
    
    # 3. Count account violations
    df = count_account_violations(df, account_column, label_column)
    
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETED!")
    print("=" * 50)
    print(f"\nFinal columns: {df.columns.tolist()}")
    
    return df


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/raw/data.csv")
    print(f"Loaded data: {df.shape}")
    
    # Process all features
    df = process_all_features(
        df,
        text_column='post_message',
        account_column='user_name',
        label_column='label'
    )
    
    # Save processed data
    df.to_csv("data/processed/data_with_features.csv", index=False)
    print(f"\nSaved to: data/processed/data_with_features.csv")
    
    # Show sample
    print("\n" + "=" * 50)
    print("SAMPLE DATA")
    print("=" * 50)
    print(df[['text_tfidf', 'text_phobert', 'benati', 'count_emoji', 'total_words']].head())
