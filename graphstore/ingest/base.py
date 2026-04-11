"""Ingestor protocol and data types."""
from dataclasses import dataclass, field
from typing import Any

from graphstore.algos.chunker import Chunk  # noqa: F401 — re-export


@dataclass
class ExtractedImage:
    data: bytes
    mime_type: str
    page: int | None = None
    caption: str | None = None
    description: str | None = None


@dataclass
class IngestResult:
    markdown: str
    chunks: list[Chunk] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    parser_used: str = ""
    confidence: float = 1.0

class Ingestor:
    """Base interface for document ingestors."""
    name: str = "base"
    supported_extensions: list[str] = []

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        raise NotImplementedError
