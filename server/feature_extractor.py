"""
Feature Extractor
Trích xuất features từ text và metadata
"""

import re
from datetime import datetime
from typing import Optional
import numpy as np


class FeatureExtractor:
    """
    Trích xuất features từ content text và timestamp
    
    Features được extract:
    - feat_num_sentences: Số câu
    - feat_num_exclamation: Số dấu chấm than
    - feat_num_question: Số dấu hỏi
    - feat_avg_word_length: Độ dài trung bình từ
    - feat_num_urls: Số URL
    - feat_digit_ratio: Tỷ lệ chữ số
    - feat_is_evening: Có phải buổi tối không (18h-22h)
    """
    
    def extract(self, content_text: str, timestamp: Optional[str] = None) -> dict:
        """
        Extract tất cả features từ input
        
        Args:
            content_text: Nội dung bài đăng
            timestamp: Thời gian đăng (ISO format)
            
        Returns:
            dict chứa các features
        """
        features = {}
        
        # Text-based features
        features["feat_num_sentences"] = self.count_sentences(content_text)
        features["feat_num_exclamation"] = self.count_exclamation(content_text)
        features["feat_num_question"] = self.count_question(content_text)
        features["feat_avg_word_length"] = self.calc_avg_word_length(content_text)
        features["feat_num_urls"] = self.count_urls(content_text)
        features["feat_digit_ratio"] = self.calc_digit_ratio(content_text)
        
        # Time-based features
        features["feat_is_evening"] = self.is_evening(timestamp)
        
        return features
    
    # =========================================================================
    # TEXT FEATURES
    # =========================================================================
    def count_sentences(self, text: str) -> int:
        """Đếm số câu (dựa trên dấu . ! ?)"""
        if not text:
            return 0
        return len(re.findall(r'[.!?]+', text))
    
    def count_exclamation(self, text: str) -> int:
        """Đếm số dấu chấm than"""
        if not text:
            return 0
        return text.count('!')
    
    def count_question(self, text: str) -> int:
        """Đếm số dấu hỏi"""
        if not text:
            return 0
        return text.count('?')
    
    def calc_avg_word_length(self, text: str) -> float:
        """Độ dài trung bình của từ"""
        if not text:
            return 0.0
        words = text.split()
        if len(words) == 0:
            return 0.0
        return round(np.mean([len(w) for w in words]), 2)
    
    def count_urls(self, text: str) -> int:
        """Đếm số URL"""
        if not text:
            return 0
        # Pattern cho URL và placeholder
        url_pattern = r'(https?://\S+|www\.\S+|< ?url ?>)'
        return len(re.findall(url_pattern, text, re.IGNORECASE))
    
    def calc_digit_ratio(self, text: str) -> float:
        """Tỷ lệ chữ số trên tổng ký tự"""
        if not text or len(text) == 0:
            return 0.0
        chars = [c for c in text if not c.isspace()]
        if len(chars) == 0:
            return 0.0
        digit_count = sum(1 for c in chars if c.isdigit())
        return round(digit_count / len(chars), 4)
    
    # =========================================================================
    # TIME FEATURES
    # =========================================================================
    def is_evening(self, timestamp: Optional[str]) -> int:
        """
        Kiểm tra có phải buổi tối không (18h-22h)
        
        Args:
            timestamp: ISO format datetime string
            
        Returns:
            1 nếu buổi tối, 0 nếu không
        """
        if not timestamp:
            return 0
        
        try:
            # Parse ISO format
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = dt.hour
            return 1 if 18 <= hour < 22 else 0
        except:
            return 0


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    # Test với sample text
    sample_text = """
    BREAKING NEWS!!! Tin nóng nhất hôm nay! Bạn có biết không?
    Xem ngay tại https://example.com để biết thêm chi tiết.
    Số điện thoại liên hệ: 0123456789
    """
    
    sample_timestamp = "2024-01-15T19:30:00Z"
    
    features = extractor.extract(sample_text, sample_timestamp)
    
    print("=" * 50)
    print("Features extracted:")
    print("=" * 50)
    for key, value in features.items():
        print(f"  {key}: {value}")
