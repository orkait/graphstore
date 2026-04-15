"""TinyBERT 64-label NER finetuning package."""

from .labels import BIO_LABELS, NUM_LABELS, SEMANTIC_LABELS, build_label_maps

__all__ = [
    "BIO_LABELS",
    "NUM_LABELS",
    "SEMANTIC_LABELS",
    "build_label_maps",
]
