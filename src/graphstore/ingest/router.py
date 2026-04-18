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
}

# Formats only docling can handle - added when docling is installed
_DOCLING_EXCLUSIVE = {
    "tex": "docling",           # LaTeX
    "adoc": "docling",          # AsciiDoc
    "tif": "docling",           # TIFF images
    "tiff": "docling",
    "bmp": "docling",           # BMP images
}

try:
    import docling as _docling_check  # noqa
    EXTENSION_MAP.update(_DOCLING_EXCLUSIVE)
except ImportError:
    pass

SUPPORTED_EXTENSIONS = set(EXTENSION_MAP.keys())

_ingestor_cache = {}


def _kwargs_cache_key(kwargs: dict) -> tuple:
    """Produce a hashable, order-independent key from kwargs.

    Used to distinguish cached ingestors with different configuration
    (e.g., max_tokens=500 vs max_tokens=1000). Pre-fix, the cache keyed
    by ingestor name only, so a second construction with different
    kwargs silently returned the first instance with the ORIGINAL kwargs
    still active (bug #59). Values must be hashable; non-hashable values
    fall back to their repr.
    """
    items = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        try:
            hash(v)
            items.append((k, v))
        except TypeError:
            items.append((k, repr(v)))
    return tuple(items)


def _get_ingestor(name: str, **kwargs):
    cache_key = (name, _kwargs_cache_key(kwargs))
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
        else:
            raise ValueError(f"Unknown ingestor: {name!r}. Available: markitdown, pymupdf4llm, docling")
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
    _docling_formats = ["pdf", "docx", "pptx", "xlsx", "md", "html", "csv",
                        "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp",
                        "tex", "adoc", "m4a", "aac", "mp4", "avi", "mov"]
    try:
        import docling as _  # noqa
        docling_available = True
    except ImportError:
        docling_available = False

    return [
        {"name": "markitdown", "formats": ["txt", "md", "html", "csv", "json", "xml", "docx", "pptx", "xlsx", "pdf", "zip", "png", "jpg"], "tier": 1},
        {"name": "pymupdf4llm", "formats": ["pdf"], "tier": 2},
        {"name": "docling", "formats": _docling_formats, "tier": 3, "available": docling_available},
        {"name": "audio", "formats": ["wav", "mp3", "ogg", "flac"], "tier": 4, "opt_in": True},
    ]
