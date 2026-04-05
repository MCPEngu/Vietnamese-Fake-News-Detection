"""
Text normalization + cleaning helpers for inference.
- normalize_to_raw_style: convert incoming text toward dataset style.
- clean_for_bert / clean_for_tfidf: reuse s1 preprocessing logic.
- prepare_text_for_embeddings: one-call API for main.py.
"""

import re
import unicodedata
from typing import Dict

import emoji

try:
    from underthesea import word_tokenize
except Exception:  # pragma: no cover - optional dependency
    word_tokenize = None


def normalize_to_raw_style(text: str) -> str:
    """
    Convert text closer to data/raw/data.csv style (Vietnamese compounds joined by underscore).
    If underthesea is unavailable, falls back to whitespace normalization only.
    """
    if not text or not isinstance(text, str):
        return ""

    normalized = unicodedata.normalize("NFC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return ""

    if word_tokenize is None:
        return normalized

    try:
        segmented = word_tokenize(normalized, format="text")
        return segmented if isinstance(segmented, str) else normalized
    except Exception:
        return normalized


def _remove_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _remove_emojis(text: str) -> str:
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
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(" ", text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _remove_repeated_chars(text: str, max_repeat: int = 2) -> str:
    pattern = r"(.)\1{" + str(max_repeat) + r",}"
    return re.sub(pattern, r"\1" * max_repeat, text)


def clean_for_tfidf(text: str, keep_word_segmentation: bool = True) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = _remove_html_tags(text)
    text = re.sub(r"<\s*url\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = _remove_emojis(text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"@\w+", " ", text)
    text = unicodedata.normalize("NFC", text).lower()

    keep_chars = "_" if keep_word_segmentation else ""
    text = re.sub(f"[^a-zA-ZÀ-ỹ0-9\\s{re.escape(keep_chars)}]", " ", text)
    text = _remove_repeated_chars(text, max_repeat=2)
    return _normalize_whitespace(text)


def clean_for_bert(text: str, keep_word_segmentation: bool = True) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = _remove_html_tags(text)
    text = re.sub(r"<\s*url\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"@(\w+)", r"\1", text)
    text = emoji.demojize(text, delimiters=("[", "]"))
    text = _remove_repeated_chars(text, max_repeat=3)

    if not keep_word_segmentation:
        text = text.replace("_", " ")

    return _normalize_whitespace(text)


def prepare_text_for_embeddings(raw_text: str) -> Dict[str, str]:
    """
    Main text preprocessing function for inference.
    Input: original extension text
    Output: text_bert + text_tfidf
    """
    text_raw_style = normalize_to_raw_style(raw_text)
    text_bert = clean_for_bert(text_raw_style, keep_word_segmentation=True)
    text_tfidf = clean_for_tfidf(text_raw_style, keep_word_segmentation=True)
    return {
        "text_raw_style": text_raw_style,
        "text_bert": text_bert,
        "text_tfidf": text_tfidf,
    }
