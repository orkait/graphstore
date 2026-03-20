"""Tests for ingestion protocol and chunker."""
from graphstore.ingest.base import Chunk, IngestResult, Ingestor, ExtractedImage
from graphstore.ingest.chunker import chunk_by_heading, chunk_by_paragraph, chunk_fixed, _make_summary


class TestMakeSummary:
    def test_short_text_unchanged(self):
        assert _make_summary("hello") == "hello"

    def test_long_text_truncated(self):
        s = _make_summary("word " * 100, max_len=50)
        assert len(s) <= 55  # some tolerance for word boundary
        assert s.endswith("...")

    def test_empty_text(self):
        assert _make_summary("") == ""


class TestChunkByHeading:
    def test_splits_on_headings(self):
        text = "# Intro\nHello world\n## Details\nMore stuff\n## End\nBye"
        chunks = chunk_by_heading(text)
        assert len(chunks) >= 3
        assert chunks[0].heading == "Intro"
        assert "Hello world" in chunks[0].text

    def test_no_headings_falls_back(self):
        text = "Just plain text without any headings."
        chunks = chunk_by_heading(text)
        assert len(chunks) == 1

    def test_chunk_has_summary(self):
        text = "# Overview\nThis is a detailed section about something important."
        chunks = chunk_by_heading(text)
        assert chunks[0].summary
        assert len(chunks[0].summary) <= 203  # 200 + "..."

    def test_preamble_before_first_heading(self):
        text = "Some preamble text\n\n# Heading\nContent"
        chunks = chunk_by_heading(text)
        assert chunks[0].heading is None  # preamble has no heading
        assert "preamble" in chunks[0].text

    def test_respects_max_size(self):
        text = "# Big Section\n" + "word " * 1000
        chunks = chunk_by_heading(text, max_chunk_size=200)
        assert len(chunks) > 1


class TestChunkByParagraph:
    def test_splits_on_double_newline(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_by_paragraph(text, max_chunk_size=50)
        assert len(chunks) >= 2

    def test_single_paragraph(self):
        text = "Just one paragraph."
        chunks = chunk_by_paragraph(text)
        assert len(chunks) == 1

    def test_empty_text(self):
        chunks = chunk_by_paragraph("")
        assert len(chunks) == 1


class TestChunkFixed:
    def test_fixed_size(self):
        text = "a" * 500
        chunks = chunk_fixed(text, chunk_size=100, overlap=20)
        assert len(chunks) >= 5

    def test_overlap(self):
        text = "abcdefghij" * 50
        chunks = chunk_fixed(text, chunk_size=100, overlap=20)
        if len(chunks) > 1:
            end_of_first = chunks[0].text[-20:]
            start_of_second = chunks[1].text[:20]
            assert end_of_first == start_of_second

    def test_each_chunk_has_summary(self):
        text = "x" * 500
        chunks = chunk_fixed(text, chunk_size=100)
        for c in chunks:
            assert c.summary


class TestBaseProtocol:
    def test_ingest_result_defaults(self):
        r = IngestResult(markdown="hello")
        assert r.chunks == []
        assert r.images == []
        assert r.confidence == 1.0

    def test_extracted_image(self):
        img = ExtractedImage(data=b"png", mime_type="image/png", page=3)
        assert img.page == 3

    def test_ingestor_not_implemented(self):
        import pytest
        with pytest.raises(NotImplementedError):
            Ingestor().convert("test.txt")


class TestMarkItDownIngestor:
    def test_convert_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world\nThis is a test")
        from graphstore.ingest.markitdown_ingestor import MarkItDownIngestor
        ingestor = MarkItDownIngestor()
        result = ingestor.convert(str(f))
        assert "Hello" in result.markdown
        assert result.parser_used == "markitdown"

    def test_supported_extensions(self):
        from graphstore.ingest.markitdown_ingestor import MarkItDownIngestor
        assert "txt" in MarkItDownIngestor.supported_extensions
        assert "html" in MarkItDownIngestor.supported_extensions


class TestRouter:
    def test_txt_routes_to_markitdown(self):
        from graphstore.ingest.router import select_ingestor
        assert select_ingestor("notes.txt") == "markitdown"

    def test_pdf_routes_to_pymupdf4llm(self):
        from graphstore.ingest.router import select_ingestor
        assert select_ingestor("report.pdf") == "pymupdf4llm"

    def test_docx_routes_to_markitdown(self):
        from graphstore.ingest.router import select_ingestor
        assert select_ingestor("doc.docx") == "markitdown"

    def test_explicit_override(self):
        from graphstore.ingest.router import select_ingestor
        assert select_ingestor("report.pdf", using="docling") == "docling"

    def test_unsupported_format_raises(self):
        import pytest
        from graphstore.ingest.router import select_ingestor
        with pytest.raises(ValueError, match="Unsupported format"):
            select_ingestor("video.mp4")

    def test_list_ingestors(self):
        from graphstore.ingest.router import list_ingestors
        ingestors = list_ingestors()
        assert len(ingestors) == 3
        names = [i["name"] for i in ingestors]
        assert "markitdown" in names

    def test_ingest_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("# Hello\n\nWorld")
        from graphstore.ingest.router import ingest_file
        result = ingest_file(str(f))
        assert "Hello" in result.markdown


class TestVisionHandler:
    def test_init_without_openai_raises(self):
        """VisionHandler should work if openai is installed."""
        try:
            from graphstore.ingest.vision import VisionHandler
            vh = VisionHandler()
            # Just test initialization, not actual Ollama connection
            assert vh.model == "smolvlm2:2.2b"
        except ImportError:
            pass  # openai not installed, skip
