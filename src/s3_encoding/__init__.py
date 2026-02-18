"""
Encoding Module for Vietnamese Fake News Detection

Bao gồm:
- TF-IDF vectorization
- BERT embeddings extraction (pre-trained & fine-tuned)

Models:
- PhoBERT v2 (vinai/phobert-base-v2)
- ViSoBERT (uitnlp/visobert)
- PhoBERT Large (vinai/phobert-large)
"""

from .f1_tfidf import TfidfEncoder, create_tfidf_encoding
from .f2_phobertv2 import PhoBertV2Encoder, create_phobertv2_embeddings
from .f3_visobert import ViSoBertEncoder, create_visobert_embeddings
from .f4_phobertlarge import PhoBertLargeEncoder, create_phobertlarge_embeddings

__all__ = [
    'TfidfEncoder',
    'create_tfidf_encoding',
    'PhoBertV2Encoder', 
    'create_phobertv2_embeddings',
    'ViSoBertEncoder',
    'create_visobert_embeddings',
    'PhoBertLargeEncoder',
    'create_phobertlarge_embeddings'
]
