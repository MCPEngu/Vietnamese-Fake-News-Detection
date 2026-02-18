"""
Text Cleaning Module
Làm sạch text cho TF-IDF và BERT
"""

import re
import unicodedata
from typing import Optional
import emoji


# =============================================================================
# COMMON CLEANING FUNCTIONS
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
    # Remove URL placeholders
    text = re.sub(r'<\s*url\s*>', ' ', text, flags=re.IGNORECASE)
    # Remove actual URLs
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    return text


def remove_emails(text: str) -> str:
    """Xóa email addresses"""
    if not text:
        return ""
    return re.sub(r'\S+@\S+\.\S+', ' ', text)


def remove_emojis(text: str) -> str:
    """Xóa tất cả emoji (dùng cho TF-IDF)"""
    if not text:
        return ""
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
        "\U00002300-\U000023FF"  # misc technical
        "\U00002600-\U000026FF"  # misc symbols
        "\U00002700-\U000027BF"  # dingbats
        "\U0000FE00-\U0000FE0F"  # variation selectors
        "\U0001F000-\U0001F02F"  # mahjong tiles
        "\U0001F0A0-\U0001F0FF"  # playing cards
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(' ', text)


def translate_emojis_to_text(text: str) -> str:
    """
    Chuyển emoji thành text description (cho BERT)
    
    VD: 😱 → [face_screaming_in_fear]
        🔥 → [fire]
        😂 → [face_with_tears_of_joy]
    """
    if not text:
        return ""
    
    # emoji.demojize converts emoji to text like :face_screaming_in_fear:
    # We wrap in [] to make it more distinctive for BERT
    text = emoji.demojize(text, delimiters=('[', ']'))
    return text


def remove_hashtags(text: str) -> str:
    """Xóa hashtags"""
    if not text:
        return ""
    return re.sub(r'#\w+', ' ', text)


def remove_mentions(text: str) -> str:
    """Xóa mentions (@user)"""
    if not text:
        return ""
    return re.sub(r'@\w+', ' ', text)


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode (NFC)"""
    if not text:
        return ""
    return unicodedata.normalize('NFC', text)


def normalize_whitespace(text: str) -> str:
    """Chuẩn hóa khoảng trắng (multiple spaces -> single space)"""
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


def remove_numbers(text: str) -> str:
    """Xóa tất cả số"""
    if not text:
        return ""
    return re.sub(r'\d+', ' ', text)


def remove_special_characters(text: str, keep_chars: str = '') -> str:
    """
    Xóa ký tự đặc biệt, giữ lại chữ cái, số, khoảng trắng và keep_chars
    
    Args:
        text: Input text
        keep_chars: Các ký tự muốn giữ lại (vd: '_' cho word segmentation)
    """
    if not text:
        return ""
    # Keep Vietnamese characters, letters, numbers, spaces, and keep_chars
    pattern = f'[^a-zA-ZÀ-ỹ0-9\\s{re.escape(keep_chars)}]'
    return re.sub(pattern, ' ', text)


def remove_extra_punctuation(text: str) -> str:
    """Xóa dấu câu thừa, giữ lại dấu cơ bản . , ! ?"""
    if not text:
        return ""
    # Remove multiple consecutive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[,]{2,}', ',', text)
    # Remove other special punctuation
    text = re.sub(r'[;:@#$%^&*()_+=\[\]{}|\\<>/~`]', ' ', text)
    return text


# =============================================================================
# TF-IDF SPECIFIC CLEANING
# =============================================================================

def clean_for_tfidf(text: str, 
                    remove_stopwords: bool = False,
                    keep_word_segmentation: bool = True) -> str:
    """
    Làm sạch text cho TF-IDF
    
    Pipeline:
    1. Xóa HTML tags
    2. Xóa URLs
    3. Xóa emails
    4. Xóa emojis
    5. Xóa hashtags (hoặc giữ text sau #)
    6. Xóa mentions
    7. Chuẩn hóa Unicode
    8. Lowercase
    9. Xóa dấu câu thừa
    10. Giảm ký tự lặp
    11. Chuẩn hóa khoảng trắng
    
    Args:
        text: Input text
        remove_stopwords: Có xóa stopwords không (default: False)
        keep_word_segmentation: Giữ dấu _ của word segmentation (default: True)
    
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Remove HTML tags
    text = remove_html_tags(text)
    
    # 2. Remove URLs
    text = remove_urls(text)
    
    # 3. Remove emails
    text = remove_emails(text)
    
    # 4. Remove emojis
    text = remove_emojis(text)
    
    # 5. Remove hashtags (or extract text)
    text = re.sub(r'#(\w+)', r'\1', text)  # Giữ text sau #
    
    # 6. Remove mentions
    text = remove_mentions(text)
    
    # 7. Normalize Unicode
    text = normalize_unicode(text)
    
    # 8. Lowercase
    text = text.lower()
    
    # 9. Remove extra punctuation
    text = remove_extra_punctuation(text)
    
    # 10. Remove special characters
    keep_chars = '_' if keep_word_segmentation else ''
    text = remove_special_characters(text, keep_chars=keep_chars)
    
    # 11. Reduce repeated characters
    text = remove_repeated_chars(text, max_repeat=2)
    
    # 12. Normalize whitespace
    text = normalize_whitespace(text)
    
    return text


def clean_for_tfidf_strict(text: str) -> str:
    """
    Làm sạch nghiêm ngặt cho TF-IDF (xóa nhiều hơn)
    - Xóa số
    - Xóa tất cả dấu câu
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = clean_for_tfidf(text)
    
    # Remove numbers
    text = remove_numbers(text)
    
    # Remove remaining punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace
    text = normalize_whitespace(text)
    
    return text


# =============================================================================
# BERT SPECIFIC CLEANING
# =============================================================================

def clean_for_bert(text: str,
                   keep_word_segmentation: bool = True) -> str:
    """
    Làm sạch text cho BERT/PhoBERT/ViSoBERT
    
    BERT cần giữ lại nhiều thông tin hơn TF-IDF vì nó hiểu ngữ cảnh.
    Emoji được chuyển thành text description để BERT có thể học.
    
    Pipeline:
    1. Xóa HTML tags
    2. Xóa URLs
    3. Xóa emails
    4. Chuẩn hóa Unicode
    5. Xử lý hashtags (giữ text)
    6. Xử lý mentions (giữ username)
    7. Dịch emoji → text (😱 → [face_screaming_in_fear])
    8. Giảm ký tự lặp
    9. Chuẩn hóa khoảng trắng
    
    Args:
        text: Input text
        keep_word_segmentation: Giữ dấu _ của word segmentation (default: True)
    
    Returns:
        Cleaned text với emoji đã dịch thành text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Remove HTML tags
    text = remove_html_tags(text)
    
    # 2. Remove URLs
    text = remove_urls(text)
    
    # 3. Remove emails
    text = remove_emails(text)
    
    # 4. Normalize Unicode
    text = normalize_unicode(text)
    
    # 5. Handle hashtags - keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 6. Handle mentions - remove @
    text = re.sub(r'@(\w+)', r'\1', text)
    
    # 7. Translate emojis to text descriptions (BERT can learn from these)
    # VD: 😱 → [face_screaming_in_fear]
    text = translate_emojis_to_text(text)
    
    # 8. Reduce repeated characters
    text = remove_repeated_chars(text, max_repeat=3)
    
    # 9. Normalize whitespace
    text = normalize_whitespace(text)
    
    return text


def clean_for_bert_with_special_tokens(text: str, 
                                        max_length: int = 256) -> str:
    """
    Làm sạch và chuẩn bị text cho BERT với giới hạn độ dài
    
    Args:
        text: Input text
        max_length: Giới hạn ký tự (sẽ được tokenizer xử lý thêm)
    
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Clean
    text = clean_for_bert(text)
    
    # Truncate if too long (rough estimate, tokenizer will handle exact)
    if len(text) > max_length * 4:  # ~4 chars per token estimate
        text = text[:max_length * 4]
    
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


# =============================================================================
# VIETNAMESE SPECIFIC
# =============================================================================

# Common Vietnamese stopwords (optional use)
VIETNAMESE_STOPWORDS = {
    'và', 'của', 'là', 'có', 'được', 'cho', 'với', 'này', 'các', 'trong',
    'để', 'đã', 'khi', 'thì', 'mà', 'nhưng', 'cũng', 'như', 'còn', 'hay',
    'vì', 'nếu', 'từ', 'đến', 'bị', 'tại', 'trên', 'dưới', 'sau', 'trước',
    'ra', 'vào', 'lại', 'đi', 'về', 'lên', 'xuống', 'theo', 'qua', 'nên',
    'rất', 'quá', 'hơn', 'nhất', 'nhiều', 'ít', 'một', 'hai', 'ba', 'những',
    'tôi', 'bạn', 'anh', 'chị', 'em', 'họ', 'chúng', 'ai', 'gì', 'nào',
    'thế', 'sao', 'làm', 'biết', 'muốn', 'cần', 'phải', 'nên', 'sẽ', 'đang'
}


def remove_vietnamese_stopwords(text: str, stopwords: set = None) -> str:
    """Xóa Vietnamese stopwords"""
    if not text:
        return ""
    if stopwords is None:
        stopwords = VIETNAMESE_STOPWORDS
    
    words = text.split()
    filtered = [w for w in words if w.lower() not in stopwords]
    return ' '.join(filtered)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_text_stats(text: str) -> dict:
    """Lấy thống kê của text"""
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'avg_word_length': 0
        }
    
    words = text.split()
    return {
        'char_count': len(text),
        'word_count': len(words),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0
    }


def compare_cleaning(text: str) -> dict:
    """So sánh kết quả của các phương pháp làm sạch"""
    return {
        'original': text,
        'tfidf': clean_for_tfidf(text),
        'tfidf_strict': clean_for_tfidf_strict(text),
        'bert': clean_for_bert(text)
    }
