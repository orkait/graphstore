"""MarkItDown ingestor: Tier 1 - general files (Office, HTML, text, etc.)."""
from graphstore.ingest.base import Ingestor, IngestResult


class MarkItDownIngestor(Ingestor):
    name = "markitdown"
    supported_extensions = [
        "txt", "md", "html", "htm", "csv", "json", "xml",
        "docx", "pptx", "xlsx", "pdf", "zip",
    ]

    def __init__(self, llm_client=None, llm_model=None):
        from markitdown import MarkItDown

        kwargs = {}
        if llm_client:
            kwargs["llm_client"] = llm_client
            kwargs["llm_model"] = llm_model
            kwargs["enable_plugins"] = True
        self._md = MarkItDown(**kwargs)

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        result = self._md.convert(file_path)
        return IngestResult(
            markdown=result.text_content,
            metadata={"source": file_path},
            parser_used=self.name,
        )
