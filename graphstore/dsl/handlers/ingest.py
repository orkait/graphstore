"""Ingest and connect handlers for the DSL executor."""

import time
import hashlib
import logging
from pathlib import Path as _Path

import numpy as np

logger = logging.getLogger(__name__)

from graphstore.dsl.handlers._registry import handles
from graphstore.dsl.ast_nodes import IngestStmt, ConnectNode
from graphstore.core.errors import GraphStoreError
from graphstore.core.types import Result


class IngestHandlers:

    @handles(IngestStmt, write=True)
    def _ingest(self, q: IngestStmt) -> Result:
        """INGEST: parse file, chunk, create graph nodes + edges, store documents."""
        from graphstore.ingest.router import ingest_file, EXTENSION_MAP
        from graphstore.ingest.chunker import chunk_by_heading

        resolved = _Path(q.file_path).resolve()
        if self._ingest_root:
            root = _Path(self._ingest_root).resolve()
            if not str(resolved).startswith(str(root)):
                raise GraphStoreError(
                    f"Path traversal not allowed: {q.file_path} "
                    f"is outside ingest root {self._ingest_root}"
                )
        if not resolved.exists():
            raise GraphStoreError(f"File not found: {q.file_path}")

        safe_path = str(resolved)
        ext = resolved.suffix.lstrip(".").lower()

        image_exts = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"}
        if ext in image_exts and q.vision_model:
            return self._ingest_image_with_vision(q, safe_path, ext)

        result = ingest_file(safe_path, using=q.using)

        chunk_size = getattr(self, '_chunk_max_size', 2000)
        summary_len = getattr(self, '_summary_max_length', 200)
        chunk_overlap = getattr(self, '_chunk_overlap', 50)
        chunks = chunk_by_heading(result.markdown, max_chunk_size=chunk_size, summary_max_len=summary_len, overlap=chunk_overlap)

        parent_id = q.node_id
        if not parent_id:
            h = hashlib.sha256(q.file_path.encode()).hexdigest()[:12]
            parent_id = f"doc:{h}"

        parent_kind = q.kind or "document"

        existing = self.store.get_node(parent_id)
        if existing is not None:
            raise GraphStoreError(f"Node already exists: {parent_id}")

        metadata_fields = {
            "source": q.file_path,
            "parser": result.parser_used,
            "confidence": result.confidence,
        }
        metadata_fields.update({
            k: v for k, v in result.metadata.items()
            if isinstance(v, (str, int, float)) and k not in ("source",)
        })

        parent_slot = self.store.put_node(parent_id, parent_kind, metadata_fields)
        self.store.columns.set_reserved(parent_slot, "__blob_state__", "warm")

        if self._document_store:
            self._document_store.put_document(
                parent_slot, result.markdown.encode("utf-8"), "text/markdown")
            self._document_store.put_metadata(parent_slot, {
                "source_path": q.file_path,
                "pages": result.metadata.get("pages"),
                "author": result.metadata.get("author"),
                "title": result.metadata.get("title"),
                "parser_used": result.parser_used,
                "confidence": result.confidence,
                "ingested_at": int(time.time() * 1000),
            })

        embed_batch: list[tuple[int, str]] = []
        sections: dict[str, str] = {}
        section_slots: dict[str, int] = {}
        set_reserved = self.store.columns.set_reserved
        for chunk in chunks:
            if chunk.heading and chunk.heading not in sections:
                section_id = f"{parent_id}:section:{len(sections)}"
                sec_slot = self.store.put_node(section_id, "section", {
                    "heading": chunk.heading,
                    "summary": chunk.summary[:200],
                })
                set_reserved(sec_slot, "__confidence__", 0.6)
                set_reserved(sec_slot, "__blob_state__", "warm")
                self.store.put_edge(parent_id, section_id, "has_section")
                embed_text = f"{chunk.heading}: {chunk.text}" if chunk.heading else chunk.text
                embed_batch.append((sec_slot, embed_text))
                sections[chunk.heading] = section_id
                section_slots[chunk.heading] = sec_slot

        chunk_ids = []
        ds = self._document_store
        for chunk in chunks:
            chunk_id = f"{parent_id}:chunk:{chunk.index}"
            chunk_fields = {"summary": chunk.summary}
            if chunk.heading:
                chunk_fields["heading"] = chunk.heading
            if chunk.page is not None:
                chunk_fields["page"] = chunk.page

            chunk_slot = self.store.put_node(chunk_id, "chunk", chunk_fields)
            set_reserved(chunk_slot, "__blob_state__", "warm")
            chunk_ids.append(chunk_id)

            if ds:
                ds._conn.execute(
                    "INSERT OR REPLACE INTO documents (slot, content, content_type, size) VALUES (?, ?, ?, ?)",
                    (chunk_slot, chunk.text.encode("utf-8"), "text/markdown", len(chunk.text)))
                ds._conn.execute(
                    "INSERT OR REPLACE INTO summaries VALUES (?, ?, ?, ?, ?, ?)",
                    (chunk_slot, chunk.summary, chunk.heading, chunk.page, chunk.index, parent_slot))
                fts_text = chunk.text if getattr(self, '_fts_full_text', True) else chunk.summary
                ds._conn.execute(
                    "INSERT OR REPLACE INTO doc_fts (rowid, summary) VALUES (?, ?)",
                    (chunk_slot, fts_text))

            embed_text = f"{chunk.heading}: {chunk.text}" if chunk.heading else chunk.text
            embed_batch.append((chunk_slot, embed_text))

            if chunk.heading and chunk.heading in sections:
                self.store.put_edge(sections[chunk.heading], chunk_id, "has_chunk")
            else:
                self.store.put_edge(parent_id, chunk_id, "has_chunk")

        if ds:
            ds._conn.commit()

        image_count = 0
        vision_handler = None
        if q.vision_model:
            try:
                from graphstore.ingest.vision import VisionHandler
                vision_handler = VisionHandler(
                    model=q.vision_model,
                    base_url=getattr(self, '_vision_base_url', 'http://localhost:11434/v1'),
                    max_tokens=getattr(self, '_vision_max_tokens', 300),
                )
            except Exception as e:
                logger.debug("vision handler init failed: %s", e, exc_info=True)

        for i, img in enumerate(result.images):
            img_id = f"{parent_id}:image:{i}"
            img_fields = {}
            if img.page is not None:
                img_fields["page"] = img.page

            if not img.description and vision_handler:
                try:
                    img.description = vision_handler.describe(img.data, img.mime_type)
                except Exception as e:
                    logger.debug("image description failed: %s", e, exc_info=True)

            if img.description:
                img_fields["summary"] = img.description

            img_slot = self.store.put_node(img_id, "image", img_fields)
            set_reserved(img_slot, "__blob_state__", "warm")

            if ds:
                ds.put_image(img_slot, img.data, img.mime_type, img.page, img.description)

            if img.description:
                embed_batch.append((img_slot, img.description))

            self.store.put_edge(parent_id, img_id, "has_image")
            image_count += 1

        self._batch_embed_and_store(embed_batch)

        return Result(kind="ok", data={
            "doc_id": parent_id,
            "chunks": len(chunks),
            "sections": len(sections),
            "images": image_count,
            "parser": result.parser_used,
            "confidence": result.confidence,
        }, count=len(chunks))

    def _ingest_image_with_vision(self, q: IngestStmt, safe_path: str, ext: str) -> Result:
        """Handle standalone image ingest with VLM description."""
        from graphstore.ingest.vision import VisionHandler

        with open(safe_path, "rb") as f:
            image_bytes = f.read()

        mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
                    "tiff": "image/tiff"}
        mime_type = mime_map.get(ext, "image/png")

        vh = VisionHandler(
            model=q.vision_model,
            base_url=getattr(self, '_vision_base_url', 'http://localhost:11434/v1'),
            max_tokens=getattr(self, '_vision_max_tokens', 300),
        )
        description = vh.describe(image_bytes, mime_type)

        node_id = q.node_id
        if not node_id:
            h = hashlib.sha256(q.file_path.encode()).hexdigest()[:12]
            node_id = f"img:{h}"

        existing = self.store.get_node(node_id)
        if existing is not None:
            raise GraphStoreError(f"Node already exists: {node_id}")

        node_kind = q.kind or "image"
        fields = {
            "summary": description,
            "source": q.file_path,
            "mime_type": mime_type,
        }
        slot = self.store.put_node(node_id, node_kind, fields)
        self.store.columns.set_reserved(slot, "__blob_state__", "warm")

        if self._document_store:
            self._document_store.put_image(slot, image_bytes, mime_type, description=description)
            self._document_store.put_summary(slot, description)

        self._embed_and_store(slot, description)

        return Result(kind="ok", data={
            "doc_id": node_id,
            "chunks": 0,
            "sections": 0,
            "images": 1,
            "parser": "vision",
            "confidence": 0.8,
        }, count=1)

    @handles(ConnectNode, write=True)
    def _connect_node(self, q: ConnectNode) -> Result:
        """CONNECT NODE: wire one node to similar neighbors via vector similarity."""
        from graphstore.ingest.connector import connect_node as _connect_node_fn
        return _connect_node_fn(
            self.store, self._vector_store,
            q.node_id, threshold=q.threshold,
        )
