"""Tests for extensibility injection points."""
import pytest
from graphstore.ingest.base import Ingestor, IngestResult


class _DummyIngestor(Ingestor):
    name = "dummy"
    supported_extensions = ["xyz"]

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        return IngestResult(markdown="dummy", parser_used="dummy")


class TestIngestorRegistry:
    def test_register_and_resolve_by_extension(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        ingestor = _DummyIngestor()
        reg.register(ingestor)
        resolved = reg.resolve("file.xyz")
        assert resolved is ingestor

    def test_resolve_unknown_extension_raises(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        with pytest.raises(ValueError, match="Unsupported format"):
            reg.resolve("file.unknownext999")

    def test_resolve_using_unknown_name_raises(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        # using= with an unknown builtin name raises ValueError from _make_builtin_ingestor
        with pytest.raises(ValueError):
            reg.resolve("file.txt", using="nonexistent_parser_xyz")

    def test_override_existing_extension(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()

        class MyPDFIngestor(Ingestor):
            name = "mypdf"
            supported_extensions = ["pdf"]
            def convert(self, path, **kwargs):
                return IngestResult(markdown="mypdf", parser_used="mypdf")

        reg.register(MyPDFIngestor())
        resolved = reg.resolve("report.pdf")
        assert resolved.name == "mypdf"

    def test_builtin_pdf_resolves(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        ingestor = reg.resolve("report.pdf")
        assert ingestor is not None
        assert ingestor.name in ("pymupdf4llm", "markitdown")

    def test_builtin_txt_resolves(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        ingestor = reg.resolve("notes.txt")
        assert ingestor.name == "markitdown"

    def test_list_returns_registered(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        reg.register(_DummyIngestor())
        entries = reg.list()
        names = [e["name"] for e in entries]
        assert "dummy" in names

    def test_using_override_bypasses_extension_map(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        reg.register(_DummyIngestor())
        resolved = reg.resolve("report.pdf", using="dummy")
        assert resolved.name == "dummy"
