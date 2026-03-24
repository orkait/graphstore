"""PyMuPDF4LLM ingestor: Tier 2 - PDF with structure + image extraction."""
import logging
from graphstore.ingest.base import Ingestor, IngestResult, ExtractedImage

logger = logging.getLogger(__name__)


class PyMuPDF4LLMIngestor(Ingestor):
    name = "pymupdf4llm"
    supported_extensions = ["pdf"]

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        import pymupdf4llm
        import pymupdf

        md_text = pymupdf4llm.to_markdown(file_path)

        images = []
        doc = pymupdf.open(file_path)
        metadata = {"pages": len(doc), "source": file_path}
        if doc.metadata:
            metadata.update({k: v for k, v in doc.metadata.items() if v})

        for page_num, page in enumerate(doc):
            for img_info in page.get_images(full=True):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    if base_image:
                        images.append(ExtractedImage(
                            data=base_image["image"],
                            mime_type=f"image/{base_image['ext']}",
                            page=page_num,
                        ))
                except Exception as e:
                    logger.debug("image extraction skipped for xref %s: %s", img_info[0], e, exc_info=True)
        doc.close()

        # Scanned PDF detection
        confidence = 1.0
        page_count = metadata.get("pages", 1) or 1
        chars_per_page = len(md_text) / page_count
        if chars_per_page < 50:
            confidence = 0.3
            metadata["warning"] = "Low text extraction. Consider: USING docling"

        return IngestResult(
            markdown=md_text,
            images=images,
            metadata=metadata,
            parser_used=self.name,
            confidence=confidence,
        )
