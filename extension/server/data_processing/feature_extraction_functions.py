"""
Feature extraction functions for online inference.
Only keeps the 11 features required by the trained B1_hour_sin_cos model.
"""

import math
import re
from datetime import datetime
from typing import Dict, Optional


def count_sentences(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"[.!?]+", text))


def count_exclamation(text: str) -> int:
    if not text:
        return 0
    return text.count("!")


def count_question(text: str) -> int:
    if not text:
        return 0
    return text.count("?")


def calc_avg_word_length(text: str) -> float:
    if not text:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    return round(sum(len(w) for w in words) / len(words), 6)


def count_urls(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"(https?://\S+|www\.\S+|<\s*url\s*>)", text, re.IGNORECASE))


def calc_digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    digit_count = sum(1 for c in chars if c.isdigit())
    return round(digit_count / len(chars), 6)


def calc_hour_sin(hour: int) -> float:
    return round(math.sin(2 * math.pi * hour / 24), 6)


def calc_hour_cos(hour: int) -> float:
    return round(math.cos(2 * math.pi * hour / 24), 6)


def calc_like_ratio(num_like: int, total_engagement: int) -> float:
    if total_engagement <= 0:
        return 0.0
    return round(num_like / total_engagement, 6)


def calc_comment_ratio(num_cmt: int, total_engagement: int) -> float:
    if total_engagement <= 0:
        return 0.0
    return round(num_cmt / total_engagement, 6)


def _parse_hour(timestamp: Optional[str]) -> int:
    if not timestamp:
        return 0
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.hour
    except Exception:
        return 0


def extract_required_features(
    content_text: str,
    timestamp: Optional[str] = None,
    num_like: int = 0,
    num_cmt: int = 0,
    num_share: int = 0,
    fake_ratio: float = 0.0,
) -> Dict[str, float]:
    """
    Main extraction function used by server inference.
    Returns exactly the 11 required features for Baseline1 + hour_sin_cos.
    """
    text = content_text or ""
    hour = _parse_hour(timestamp)
    total_engagement = max(0, int(num_like)) + max(0, int(num_cmt)) + max(0, int(num_share))

    features = {
        "feat_avg_word_length": calc_avg_word_length(text),
        "feat_comment_ratio": calc_comment_ratio(max(0, int(num_cmt)), total_engagement),
        "feat_digit_ratio": calc_digit_ratio(text),
        "feat_fake_ratio": round(float(fake_ratio), 6),
        "feat_hour_cos": calc_hour_cos(hour),
        "feat_hour_sin": calc_hour_sin(hour),
        "feat_like_ratio": calc_like_ratio(max(0, int(num_like)), total_engagement),
        "feat_num_exclamation": float(count_exclamation(text)),
        "feat_num_question": float(count_question(text)),
        "feat_num_sentences": float(count_sentences(text)),
        "feat_num_urls": float(count_urls(text)),
    }

    return features
