"""
Feature Engineering Module
Trích xuất đặc trưng từ post_message và các cột khác
"""

import re
import numpy as np
import pandas as pd
from typing import Optional


# =============================================================================
# TEXT-BASED FEATURES (từ post_message)
# =============================================================================

def count_chars(text: str) -> int:
    """Đếm số ký tự (không tính khoảng trắng)"""
    if pd.isna(text):
        return 0
    return len(text.replace(' ', ''))


def count_words(text: str) -> int:
    """Đếm số từ"""
    if pd.isna(text):
        return 0
    return len(text.split())


def count_sentences(text: str) -> int:
    """Đếm số câu (dựa trên dấu . ! ?)"""
    if pd.isna(text):
        return 0
    return len(re.findall(r'[.!?]+', text))


def count_exclamation(text: str) -> int:
    """Đếm số dấu chấm than"""
    if pd.isna(text):
        return 0
    return text.count('!')


def count_question(text: str) -> int:
    """Đếm số dấu hỏi"""
    if pd.isna(text):
        return 0
    return text.count('?')


def count_uppercase_words(text: str) -> int:
    """Đếm số từ viết hoa hoàn toàn"""
    if pd.isna(text):
        return 0
    words = text.split()
    return sum(1 for w in words if w.isupper() and len(w) > 1)


def calc_uppercase_ratio(text: str) -> float:
    """Tỷ lệ ký tự viết hoa"""
    if pd.isna(text) or len(text) == 0:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return 0.0
    upper_count = sum(1 for c in letters if c.isupper())
    return upper_count / len(letters)


def count_emojis(text: str) -> int:
    """Đếm số emoji"""
    if pd.isna(text):
        return 0
    # Unicode ranges for emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
        "]+", 
        flags=re.UNICODE
    )
    return len(emoji_pattern.findall(text))


def count_urls(text: str) -> int:
    """Đếm số URL"""
    if pd.isna(text):
        return 0
    # Đếm cả < url > placeholder và URL thực
    url_pattern = r'(https?://\S+|www\.\S+|< ?url ?>)'
    return len(re.findall(url_pattern, text, re.IGNORECASE))


def count_hashtags(text: str) -> int:
    """Đếm số hashtag"""
    if pd.isna(text):
        return 0
    return len(re.findall(r'#\w+', text))


def count_mentions(text: str) -> int:
    """Đếm số mention (@user)"""
    if pd.isna(text):
        return 0
    return len(re.findall(r'@\w+', text))


def count_digits(text: str) -> int:
    """Đếm số chữ số"""
    if pd.isna(text):
        return 0
    return sum(1 for c in text if c.isdigit())


def calc_digit_ratio(text: str) -> float:
    """Tỷ lệ chữ số trên tổng ký tự"""
    if pd.isna(text) or len(text) == 0:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if len(chars) == 0:
        return 0.0
    return sum(1 for c in chars if c.isdigit()) / len(chars)


def count_special_chars(text: str) -> int:
    """Đếm ký tự đặc biệt (!@#$%^&*...)"""
    if pd.isna(text):
        return 0
    special = set('!@#$%^&*()_+-=[]{}|;:\'",.<>?/~`')
    return sum(1 for c in text if c in special)


def calc_avg_word_length(text: str) -> float:
    """Độ dài trung bình của từ"""
    if pd.isna(text):
        return 0.0
    words = text.split()
    if len(words) == 0:
        return 0.0
    return np.mean([len(w) for w in words])


def count_long_words(text: str, min_length: int = 10) -> int:
    """Đếm số từ dài (>= min_length ký tự)"""
    if pd.isna(text):
        return 0
    words = text.split()
    return sum(1 for w in words if len(w) >= min_length)


def count_short_words(text: str, max_length: int = 3) -> int:
    """Đếm số từ ngắn (<= max_length ký tự)"""
    if pd.isna(text):
        return 0
    words = text.split()
    return sum(1 for w in words if len(w) <= max_length)


def count_quoted_text(text: str) -> int:
    """Đếm số đoạn text trong ngoặc kép"""
    if pd.isna(text):
        return 0
    return len(re.findall(r'["\'].*?["\']', text))


def has_all_caps_words(text: str) -> int:
    """Có từ viết hoa toàn bộ không (binary)"""
    if pd.isna(text):
        return 0
    words = text.split()
    return 1 if any(w.isupper() and len(w) > 2 for w in words) else 0


def count_repeated_chars(text: str) -> int:
    """Đếm số lần ký tự lặp liên tiếp >= 3 lần (vd: aaaaa, !!!!!)"""
    if pd.isna(text):
        return 0
    return len(re.findall(r'(.)\1{2,}', text))


def count_newlines(text: str) -> int:
    """Đếm số dòng mới"""
    if pd.isna(text):
        return 0
    return text.count('\n')


def calc_punctuation_ratio(text: str) -> float:
    """Tỷ lệ dấu câu trên tổng ký tự"""
    if pd.isna(text) or len(text) == 0:
        return 0.0
    punctuation = set('.,;:!?-()[]{}"\'/\\')
    chars = [c for c in text if not c.isspace()]
    if len(chars) == 0:
        return 0.0
    return sum(1 for c in chars if c in punctuation) / len(chars)


# =============================================================================
# ENGAGEMENT RATIO FEATURES
# =============================================================================

def calc_like_per_char(num_like: int, num_char: int) -> float:
    """Like trên mỗi ký tự"""
    if num_char == 0:
        return 0.0
    return num_like / num_char


def calc_cmt_per_char(num_cmt: int, num_char: int) -> float:
    """Comment trên mỗi ký tự"""
    if num_char == 0:
        return 0.0
    return num_cmt / num_char


def calc_share_per_char(num_share: int, num_char: int) -> float:
    """Share trên mỗi ký tự"""
    if num_char == 0:
        return 0.0
    return num_share / num_char


def calc_engagement_total(num_like: int, num_cmt: int, num_share: int) -> int:
    """Tổng engagement"""
    return num_like + num_cmt + num_share


def calc_like_cmt_ratio(num_like: int, num_cmt: int) -> float:
    """Tỷ lệ like/comment"""
    if num_cmt == 0:
        return 0.0
    return num_like / num_cmt


def calc_share_like_ratio(num_share: int, num_like: int) -> float:
    """Tỷ lệ share/like"""
    if num_like == 0:
        return 0.0
    return num_share / num_like


# =============================================================================
# USER-BASED FEATURES
# =============================================================================

def calc_fake_ratio(num_fake: int, num_post: int) -> float:
    """Tỷ lệ bài fake của user"""
    if num_post == 0:
        return 0.0
    return num_fake / num_post


def calc_real_ratio(num_real: int, num_post: int) -> float:
    """Tỷ lệ bài real của user"""
    if num_post == 0:
        return 0.0
    return num_real / num_post


# =============================================================================
# TIME-BASED FEATURES
# =============================================================================

def is_weekend(weekday: int) -> int:
    """Cuối tuần hay không (Saturday=5, Sunday=6)"""
    return 1 if weekday in [5, 6] else 0


def is_night(hour: int) -> int:
    """Ban đêm (22h-6h)"""
    return 1 if hour >= 22 or hour < 6 else 0


def is_morning(hour: int) -> int:
    """Buổi sáng (6h-12h)"""
    return 1 if 6 <= hour < 12 else 0


def is_afternoon(hour: int) -> int:
    """Buổi chiều (12h-18h)"""
    return 1 if 12 <= hour < 18 else 0


def is_evening(hour: int) -> int:
    """Buổi tối (18h-22h)"""
    return 1 if 18 <= hour < 22 else 0


def get_time_of_day(hour: int) -> str:
    """Thời điểm trong ngày (categorical)"""
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 22:
        return 'evening'
    else:
        return 'night'


# =============================================================================
# MAIN FEATURE CREATION FUNCTION
# =============================================================================

def create_text_features(df: pd.DataFrame, text_col: str = 'post_message') -> pd.DataFrame:
    """
    Tạo tất cả features từ text
    
    Args:
        df: DataFrame chứa dữ liệu
        text_col: Tên cột chứa text
        
    Returns:
        DataFrame với các features mới
    """
    result = df.copy()
    text = result[text_col]
    
    # Text-based features
    result['feat_num_chars'] = text.apply(count_chars)
    result['feat_num_words'] = text.apply(count_words)
    result['feat_num_sentences'] = text.apply(count_sentences)
    result['feat_num_exclamation'] = text.apply(count_exclamation)
    result['feat_num_question'] = text.apply(count_question)
    result['feat_num_uppercase_words'] = text.apply(count_uppercase_words)
    result['feat_uppercase_ratio'] = text.apply(calc_uppercase_ratio)
    result['feat_num_emojis'] = text.apply(count_emojis)
    result['feat_num_urls'] = text.apply(count_urls)
    result['feat_num_hashtags'] = text.apply(count_hashtags)
    result['feat_num_mentions'] = text.apply(count_mentions)
    result['feat_num_digits'] = text.apply(count_digits)
    result['feat_digit_ratio'] = text.apply(calc_digit_ratio)
    result['feat_num_special_chars'] = text.apply(count_special_chars)
    result['feat_avg_word_length'] = text.apply(calc_avg_word_length)
    result['feat_num_long_words'] = text.apply(count_long_words)
    result['feat_num_short_words'] = text.apply(count_short_words)
    result['feat_num_quoted'] = text.apply(count_quoted_text)
    result['feat_has_all_caps'] = text.apply(has_all_caps_words)
    result['feat_num_repeated_chars'] = text.apply(count_repeated_chars)
    result['feat_num_newlines'] = text.apply(count_newlines)
    result['feat_punctuation_ratio'] = text.apply(calc_punctuation_ratio)
    
    return result


def create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo features từ engagement metrics
    """
    result = df.copy()
    
    # Engagement features
    result['feat_engagement_total'] = df.apply(
        lambda x: calc_engagement_total(x['num_like'], x['num_cmt'], x['num_share']), axis=1
    )
    result['feat_like_per_char'] = df.apply(
        lambda x: calc_like_per_char(x['num_like'], x['num_char']), axis=1
    )
    result['feat_cmt_per_char'] = df.apply(
        lambda x: calc_cmt_per_char(x['num_cmt'], x['num_char']), axis=1
    )
    result['feat_share_per_char'] = df.apply(
        lambda x: calc_share_per_char(x['num_share'], x['num_char']), axis=1
    )
    result['feat_like_cmt_ratio'] = df.apply(
        lambda x: calc_like_cmt_ratio(x['num_like'], x['num_cmt']), axis=1
    )
    result['feat_share_like_ratio'] = df.apply(
        lambda x: calc_share_like_ratio(x['num_share'], x['num_like']), axis=1
    )
    
    return result


def create_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo features từ user behavior
    """
    result = df.copy()
    
    result['feat_fake_ratio'] = df.apply(
        lambda x: calc_fake_ratio(x['num_fake'], x['num_post']), axis=1
    )
    result['feat_real_ratio'] = df.apply(
        lambda x: calc_real_ratio(x['num_real'], x['num_post']), axis=1
    )
    
    return result


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo features từ timestamp
    """
    result = df.copy()
    
    result['feat_is_weekend'] = df['weekday'].apply(is_weekend)
    result['feat_is_night'] = df['hour'].apply(is_night)
    result['feat_is_morning'] = df['hour'].apply(is_morning)
    result['feat_is_afternoon'] = df['hour'].apply(is_afternoon)
    result['feat_is_evening'] = df['hour'].apply(is_evening)
    result['feat_time_of_day'] = df['hour'].apply(get_time_of_day)
    
    return result


def create_all_features(df: pd.DataFrame, text_col: str = 'post_message') -> pd.DataFrame:
    """
    Tạo tất cả features
    
    Args:
        df: DataFrame gốc
        text_col: Tên cột text
        
    Returns:
        DataFrame với tất cả features mới
    """
    print("Creating text features...")
    result = create_text_features(df, text_col)
    
    print("Creating engagement features...")
    result = create_engagement_features(result)
    
    print("Creating user features...")
    result = create_user_features(result)
    
    print("Creating time features...")
    result = create_time_features(result)
    
    # Get list of new features
    new_features = [col for col in result.columns if col.startswith('feat_')]
    print(f"\nCreated {len(new_features)} new features:")
    for feat in new_features:
        print(f"  - {feat}")
    
    return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> list:
    """Lấy danh sách các cột feature"""
    return [col for col in df.columns if col.startswith('feat_')]


def get_numeric_features(df: pd.DataFrame) -> list:
    """Lấy danh sách các feature số"""
    feature_cols = get_feature_columns(df)
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols
