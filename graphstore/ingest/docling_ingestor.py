"""Docling ingestor: Tier 3 - hard PDFs with tables, OCR, formulas. Lazy import."""
from graphstore.ingest.base import Ingestor, IngestResult


class DoclingIngestor(Ingestor):
    name = "docling"
    supported_extensions = ["pdf", "docx"]

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        # Lazy import - docling is ~200MB, only loaded when explicitly requested
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(file_path)
        md_text = result.document.export_to_markdown()
        return IngestResult(
            markdown=md_text,
            metadata={"source": file_path},
            parser_used=self.name,
        )
