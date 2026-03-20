"""Text chunking strategies for document ingestion."""
import re
from graphstore.ingest.base import Chunk

def _make_summary(text: str, max_len: int = 200) -> str:
    """First max_len chars, clean trailing."""
    s = text[:max_len].strip()
    if len(text) > max_len:
        s = s.rsplit(" ", 1)[0] + "..."
    return s

def chunk_by_heading(text: str, max_chunk_size: int = 2000) -> list[Chunk]:
    """Split on markdown headings. Falls back to paragraph if no headings."""
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    matches = list(heading_pattern.finditer(text))

    if not matches:
        return chunk_by_paragraph(text, max_chunk_size)

    chunks = []

    # Text before first heading
    if matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            chunks.append(Chunk(text=preamble, summary=_make_summary(preamble),
                               index=0, start_char=0))

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        heading = match.group(2).strip()

        if len(section) > max_chunk_size:
            sub_chunks = chunk_fixed(section, max_chunk_size, overlap=50)
            for sc in sub_chunks:
                sc.heading = heading
                sc.index = len(chunks)
                sc.start_char = start + sc.start_char
                chunks.append(sc)
        else:
            chunks.append(Chunk(
                text=section, summary=_make_summary(section),
                index=len(chunks), heading=heading, start_char=start))

    # Re-index
    for i, c in enumerate(chunks):
        c.index = i
    return chunks

def chunk_by_paragraph(text: str, max_chunk_size: int = 1000) -> list[Chunk]:
    """Split on double newlines."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""
    start = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current and (len(current) + len(para) > max_chunk_size or len(current.strip()) >= max_chunk_size // 2):
            chunks.append(Chunk(text=current.strip(), summary=_make_summary(current.strip()),
                               index=len(chunks), start_char=start))
            start += len(current)
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), summary=_make_summary(current.strip()),
                           index=len(chunks), start_char=start))
    if not chunks:
        chunks = [Chunk(text=text.strip(), summary=_make_summary(text.strip()), index=0, start_char=0)]
    return chunks

def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Fixed-size chunks with overlap."""
    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        chunk_text = text[pos:end]
        chunks.append(Chunk(text=chunk_text, summary=_make_summary(chunk_text),
                           index=len(chunks), start_char=pos))
        pos += chunk_size - overlap
        if pos >= len(text):
            break
    return chunks
