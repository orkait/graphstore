"""IngestorRegistry - pluggable extension-to-ingestor routing."""
from pathlib import Path
from graphstore.ingest.base import Ingestor, IngestResult


_BUILTIN_EXT_MAP: dict[str, str] = {
    "txt": "markitdown", "md": "markitdown", "html": "markitdown", "htm": "markitdown",
    "csv": "markitdown", "json": "markitdown", "xml": "markitdown",
    "docx": "markitdown", "pptx": "markitdown", "xlsx": "markitdown", "zip": "markitdown",
    "pdf": "pymupdf4llm",
    "png": "markitdown", "jpg": "markitdown", "jpeg": "markitdown",
    "gif": "markitdown", "webp": "markitdown",
    "wav": "audio", "mp3": "audio", "ogg": "audio", "flac": "audio",
}

_DOCLING_EXCLUSIVE: dict[str, str] = {
    "tex": "docling", "adoc": "docling",
    "tif": "docling", "tiff": "docling", "bmp": "docling",
    "m4a": "docling", "aac": "docling",
    "mp4": "docling", "avi": "docling", "mov": "docling",
}


def _build_builtin_ext_map() -> dict[str, str]:
    ext_map = dict(_BUILTIN_EXT_MAP)
    try:
        import docling as _  # noqa
        ext_map.update(_DOCLING_EXCLUSIVE)
    except ImportError:
        pass
    return ext_map


def _make_builtin_ingestor(name: str) -> Ingestor:
    """Lazily construct a built-in ingestor by name."""
    if name == "markitdown":
        from graphstore.ingest.markitdown_ingestor import MarkItDownIngestor
        return MarkItDownIngestor()
    if name == "pymupdf4llm":
        from graphstore.ingest.pymupdf4llm_ingestor import PyMuPDF4LLMIngestor
        return PyMuPDF4LLMIngestor()
    if name == "docling":
        from graphstore.ingest.docling_ingestor import DoclingIngestor
        return DoclingIngestor()
    if name == "audio":
        from graphstore.voice.stt import MoonshineSTT

        class _AudioIngestor(Ingestor):
            name = "audio"
            supported_extensions = ["wav", "mp3", "ogg", "flac"]

            def __init__(self):
                self._stt = MoonshineSTT()

            def convert(self, file_path: str, **kwargs) -> IngestResult:
                text = self._stt.transcribe_file(file_path)
                return IngestResult(
                    markdown=text,
                    metadata={"source": file_path},
                    parser_used="moonshine",
                )

        return _AudioIngestor()
    raise ValueError(f"Unknown built-in ingestor: {name!r}")


class IngestorRegistry:
    """Registry that maps file extensions to Ingestor instances.

    Built-in ingestors are loaded lazily. Custom ingestors registered via
    ``register()`` override built-ins for every extension they declare.
    """

    def __init__(self) -> None:
        self._ext_map: dict[str, str] = _build_builtin_ext_map()
        self._instances: dict[str, Ingestor] = {}

    def register(self, ingestor: Ingestor) -> None:
        """Register a custom ingestor. Overrides built-ins for all its extensions."""
        self._instances[ingestor.name] = ingestor
        for ext in ingestor.supported_extensions:
            self._ext_map[ext] = ingestor.name

    def resolve(self, file_path: str, using: str | None = None) -> Ingestor:
        """Return the ingestor for *file_path*, or the one named *using*."""
        if using:
            name = using
        else:
            ext = Path(file_path).suffix.lstrip(".").lower()
            if ext not in self._ext_map:
                supported = sorted(set(self._ext_map.keys()))
                raise ValueError(
                    f"Unsupported format: .{ext}. Supported: {supported}"
                )
            name = self._ext_map[ext]

        if name not in self._instances:
            self._instances[name] = _make_builtin_ingestor(name)
        return self._instances[name]

    def list(self) -> list[dict]:
        """Return all registered ingestors as dicts."""
        seen: dict[str, list[str]] = {}
        for ext, name in self._ext_map.items():
            seen.setdefault(name, []).append(ext)
        result = []
        for name, exts in seen.items():
            entry: dict = {"name": name, "formats": sorted(exts)}
            result.append(entry)
        return result
