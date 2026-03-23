"""Tiered ingestor routing: deterministic parsers first, VLM as fallback."""
from pathlib import Path
from graphstore.ingest.base import IngestResult

_PLAINTEXT_EXTS = {"txt", "md"}

EXTENSION_MAP = {
    "txt": "markitdown", "md": "markitdown", "html": "markitdown", "htm": "markitdown",
    "csv": "markitdown", "json": "markitdown", "xml": "markitdown",
    "docx": "markitdown", "pptx": "markitdown", "xlsx": "markitdown",
    "zip": "markitdown",
    "pdf": "pymupdf4llm",
    "png": "markitdown", "jpg": "markitdown", "jpeg": "markitdown",
    "gif": "markitdown", "webp": "markitdown",
    # Audio (requires voice opt-in)
    "wav": "audio", "mp3": "audio", "ogg": "audio", "flac": "audio",
}

SUPPORTED_EXTENSIONS = set(EXTENSION_MAP.keys())

_ingestor_cache = {}


def _get_ingestor(name: str, **kwargs):
    cache_key = name
    if cache_key not in _ingestor_cache:
        if name == "markitdown":
            from graphstore.ingest.markitdown_ingestor import MarkItDownIngestor
            _ingestor_cache[cache_key] = MarkItDownIngestor(**kwargs)
        elif name == "pymupdf4llm":
            from graphstore.ingest.pymupdf4llm_ingestor import PyMuPDF4LLMIngestor
            _ingestor_cache[cache_key] = PyMuPDF4LLMIngestor()
        elif name == "docling":
            from graphstore.ingest.docling_ingestor import DoclingIngestor
            _ingestor_cache[cache_key] = DoclingIngestor()
        elif name == "audio":
            from graphstore.voice.stt import MoonshineSTT
            from graphstore.ingest.base import Ingestor, IngestResult

            class AudioIngestor(Ingestor):
                name = "audio"
                supported_extensions = ["wav", "mp3", "ogg", "flac"]

                def __init__(self):
                    self._stt = MoonshineSTT()

                def convert(self, file_path, **kwargs):
                    text = self._stt.transcribe_file(file_path)
                    return IngestResult(
                        markdown=text,
                        metadata={"source": file_path},
                        parser_used="moonshine",
                    )

            _ingestor_cache[cache_key] = AudioIngestor()
        else:
            raise ValueError(f"Unknown ingestor: {name!r}. Available: markitdown, pymupdf4llm, docling, audio")
    return _ingestor_cache[cache_key]


def select_ingestor(file_path: str, using: str | None = None) -> str:
    if using:
        return using
    ext = Path(file_path).suffix.lstrip(".").lower()
    if ext not in EXTENSION_MAP:
        raise ValueError(f"Unsupported format: .{ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    return EXTENSION_MAP[ext]


def ingest_file(file_path: str, using: str | None = None, **kwargs) -> IngestResult:
    ext = Path(file_path).suffix.lstrip(".").lower()
    if ext in _PLAINTEXT_EXTS and using is None:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        return IngestResult(markdown=text, parser_used="direct", confidence=1.0,
                           metadata={"source": file_path})
    name = select_ingestor(file_path, using)
    ingestor = _get_ingestor(name, **kwargs)
    return ingestor.convert(file_path)


def list_ingestors() -> list[dict]:
    return [
        {"name": "markitdown", "formats": ["txt", "md", "html", "csv", "json", "xml", "docx", "pptx", "xlsx", "pdf", "zip", "png", "jpg"], "tier": 1},
        {"name": "pymupdf4llm", "formats": ["pdf"], "tier": 2},
        {"name": "docling", "formats": ["pdf", "docx"], "tier": 3},
        {"name": "audio", "formats": ["wav", "mp3", "ogg", "flac"], "tier": 4, "opt_in": True},
    ]
