"""
Text Cleaning Functions
Làm sạch text cho TF-IDF và BERT

Updated: Thêm placeholders cho date, money, phone
"""

import re
import unicodedata
import emoji


# =============================================================================
# BASIC CLEANING FUNCTIONS
# =============================================================================

def remove_html_tags(text: str) -> str:
    """Xóa HTML tags"""
    if not text:
        return ""
    return re.sub(r'<[^>]+>', ' ', text)


def remove_urls(text: str) -> str:
    """Xóa URLs và placeholder < url >"""
    if not text:
        return ""
    text = re.sub(r'<\s*url\s*>', ' <url> ', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+', ' <url> ', text)
    text = re.sub(r'www\.\S+', ' <url> ', text)
    return text


def remove_emails(text: str) -> str:
    """Xóa email addresses"""
    if not text:
        return ""
    return re.sub(r'\S+@\S+\.\S+', ' <email> ', text)


def remove_emojis(text: str) -> str:
    """Xóa tất cả emoji (dùng cho TF-IDF)"""
    if not text:
        return ""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002300-\U000023FF"
        "\U00002600-\U000026FF"
        "\U00002700-\U000027BF"
        "\U0000FE00-\U0000FE0F"
        "\U0001F000-\U0001F02F"
        "\U0001F0A0-\U0001F0FF"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(' ', text)


def translate_emojis_to_text(text: str) -> str:
    """
    Chuyển emoji thành text description (cho BERT)
    VD: 😱 → [face_screaming_in_fear]
    """
    if not text:
        return ""
    return emoji.demojize(text, delimiters=('[', ']'))


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode (NFC)"""
    if not text:
        return ""
    return unicodedata.normalize('NFC', text)


def normalize_whitespace(text: str) -> str:
    """Chuẩn hóa khoảng trắng"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_repeated_chars(text: str, max_repeat: int = 2) -> str:
    """Giảm ký tự lặp liên tiếp (aaaaa -> aa)"""
    if not text:
        return ""
    pattern = r'(.)\1{' + str(max_repeat) + r',}'
    return re.sub(pattern, r'\1' * max_repeat, text)


def remove_special_characters(text: str, keep_chars: str = '') -> str:
    """Xóa ký tự đặc biệt, giữ lại chữ cái, số, khoảng trắng và keep_chars"""
    if not text:
        return ""
    pattern = f'[^a-zA-ZÀ-ỹ0-9\\s{re.escape(keep_chars)}]'
    return re.sub(pattern, ' ', text)


# =============================================================================
# TF-IDF CLEANING PIPELINE
# =============================================================================

def clean_for_tfidf(text: str, keep_word_segmentation: bool = True) -> str:
    """
    Làm sạch text cho TF-IDF
    
    Pipeline:
    1. Xóa HTML tags
    2. Xóa URLs, emails
    3. Xóa emojis
    4. Xử lý hashtags/mentions
    5. Chuẩn hóa Unicode + lowercase
    6. Xóa ký tự đặc biệt + giảm lặp
    7. Chuẩn hóa khoảng trắng
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Remove HTML tags
    text = remove_html_tags(text)
    
    # 2. Remove URLs and emails
    text = re.sub(r'<\s*url\s*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    
    # 3. Remove emojis
    text = remove_emojis(text)
    
    # 4. Handle hashtags and mentions
    text = re.sub(r'#(\w+)', r'\1', text)  # Giữ text sau #
    text = re.sub(r'@\w+', ' ', text)  # Xóa mentions
    
    # 5. Normalize Unicode + lowercase
    text = normalize_unicode(text)
    text = text.lower()
    
    # 6. Remove special characters and reduce repeats
    keep_chars = '_' if keep_word_segmentation else ''
    text = remove_special_characters(text, keep_chars=keep_chars)
    text = remove_repeated_chars(text, max_repeat=2)
    
    # 7. Normalize whitespace
    text = normalize_whitespace(text)
    
    return text


# =============================================================================
# BERT CLEANING PIPELINE
# =============================================================================

def clean_for_bert(text: str, keep_word_segmentation: bool = True) -> str:
    """
    Làm sạch text cho BERT/PhoBERT/ViSoBERT
    
    BERT cần giữ lại nhiều thông tin hơn TF-IDF:
    - Emoji được chuyển thành text description
    - Giữ lại một số dấu câu quan trọng
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Remove HTML tags
    text = remove_html_tags(text)
    
    # 2. Remove URLs and emails
    text = re.sub(r'<\s*url\s*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    
    # 3. Normalize Unicode
    text = normalize_unicode(text)
    
    # 4. Handle hashtags and mentions
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'@(\w+)', r'\1', text)
    
    # 5. Translate emojis to text
    text = translate_emojis_to_text(text)
    
    # 6. Reduce repeated characters
    text = remove_repeated_chars(text, max_repeat=3)
    
    # 7. Normalize whitespace
    text = normalize_whitespace(text)
    
    return text


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def batch_clean_for_tfidf(texts: list, **kwargs) -> list:
    """Làm sạch batch texts cho TF-IDF"""
    return [clean_for_tfidf(text, **kwargs) for text in texts]


def batch_clean_for_bert(texts: list, **kwargs) -> list:
    """Làm sạch batch texts cho BERT"""
    return [clean_for_bert(text, **kwargs) for text in texts]
