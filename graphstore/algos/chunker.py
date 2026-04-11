"""Pure text chunking primitives for ingestion.

Splits markdown / plaintext into Chunk records. No graphstore imports,
no I/O. Takes a string, returns a list of Chunks.
"""

import re
from dataclasses import dataclass

__all__ = [
    "Chunk",
    "make_summary",
    "chunk_by_heading",
    "chunk_by_paragraph",
    "chunk_fixed",
]


@dataclass
class Chunk:
    text: str
    summary: str
    index: int
    heading: str | None = None
    page: int | None = None
    start_char: int = 0


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def make_summary(text: str, max_len: int = 200) -> str:
    s = text[:max_len].strip()
    if len(text) > max_len:
        s = s.rsplit(" ", 1)[0] + "..."
    return s


def chunk_fixed(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    summary_max_len: int = 200,
) -> list[Chunk]:
    """Fixed-size sliding window chunks with overlap."""
    chunks: list[Chunk] = []
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        chunk_text = text[pos:end]
        chunks.append(
            Chunk(
                text=chunk_text,
                summary=make_summary(chunk_text, summary_max_len),
                index=len(chunks),
                start_char=pos,
            )
        )
        pos += chunk_size - overlap
        if pos >= len(text):
            break
    return chunks


def chunk_by_paragraph(
    text: str,
    max_chunk_size: int = 1000,
    summary_max_len: int = 200,
) -> list[Chunk]:
    """Split on double newlines, packing paragraphs up to max_chunk_size."""
    paragraphs = _PARA_SPLIT_RE.split(text)
    chunks: list[Chunk] = []
    current = ""
    start = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current and (
            len(current) + len(para) > max_chunk_size
            or len(current.strip()) >= max_chunk_size // 2
        ):
            chunks.append(
                Chunk(
                    text=current.strip(),
                    summary=make_summary(current.strip(), summary_max_len),
                    index=len(chunks),
                    start_char=start,
                )
            )
            start += len(current)
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(
            Chunk(
                text=current.strip(),
                summary=make_summary(current.strip(), summary_max_len),
                index=len(chunks),
                start_char=start,
            )
        )
    if not chunks:
        chunks = [
            Chunk(
                text=text.strip(),
                summary=make_summary(text.strip(), summary_max_len),
                index=0,
                start_char=0,
            )
        ]
    return chunks


def chunk_by_heading(
    text: str,
    max_chunk_size: int = 2000,
    summary_max_len: int = 200,
    overlap: int = 50,
) -> list[Chunk]:
    """Split on markdown headings; fall back to paragraph split if none found."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return chunk_by_paragraph(text, max_chunk_size, summary_max_len=summary_max_len)

    chunks: list[Chunk] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            chunks.append(
                Chunk(
                    text=preamble,
                    summary=make_summary(preamble, summary_max_len),
                    index=0,
                    start_char=0,
                )
            )

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        heading = match.group(2).strip()

        if len(section) > max_chunk_size:
            sub_chunks = chunk_fixed(
                section,
                chunk_size=max_chunk_size,
                overlap=overlap,
                summary_max_len=summary_max_len,
            )
            for sc in sub_chunks:
                sc.heading = heading
                sc.index = len(chunks)
                sc.start_char = start + sc.start_char
                chunks.append(sc)
        else:
            chunks.append(
                Chunk(
                    text=section,
                    summary=make_summary(section, summary_max_len),
                    index=len(chunks),
                    heading=heading,
                    start_char=start,
                )
            )

    for i, c in enumerate(chunks):
        c.index = i
    return chunks
