"""VaultSync: sync vault directory to graphstore graph."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from graphstore.vault.parser import (
    parse_frontmatter, parse_sections, extract_wikilinks, title_to_slug,
)
from graphstore.vault.manager import VaultManager


class VaultSync:
    """Syncs vault markdown files to graphstore nodes + edges."""

    def __init__(self, manager: VaultManager, store, schema=None,
                 embedder=None, vector_store=None, document_store=None,
                 summary_max_length: int = 200):
        self._manager = manager
        self._store = store
        self._schema = schema
        self._embedder = embedder
        self._vector_store = vector_store
        self._document_store = document_store
        self._summary_max_length = summary_max_length

    def sync_all(self) -> dict:
        """Walk vault dir, sync all notes to graph. Returns {synced, skipped, errors}.

        Two-pass approach: first upsert all nodes, then recreate edges.
        This ensures wikilink targets exist before edges are created.
        """
        synced = 0
        skipped = 0
        errors = 0

        slugs = self._manager.list_files()
        synced_slugs = []

        # Pass 1: upsert nodes
        for slug in slugs:
            try:
                node_id = f"note:{slug}"
                # Check if already in graph and up to date
                str_id = (self._store.string_table.intern(node_id)
                          if node_id in self._store.string_table else None)
                if str_id is not None:
                    slot = self._store.id_to_slot.get(str_id)
                    if slot is not None and slot not in self._store.node_tombstones:
                        file_mtime = self._manager.get_mtime(slug)
                        node_mtime = self._read_file_mtime(slot)
                        if node_mtime > 0 and file_mtime <= node_mtime:
                            skipped += 1
                            continue

                self._sync_node(slug)
                synced_slugs.append(slug)
                synced += 1
            except Exception as e:
                logger.debug("vault node sync failed for %r: %s", slug, e, exc_info=True)
                errors += 1

        # Pass 2: recreate edges for all synced notes
        for slug in synced_slugs:
            try:
                self._sync_edges(slug)
            except Exception as e:
                logger.debug("vault edge sync failed for %r: %s", slug, e, exc_info=True)

        return {"synced": synced, "skipped": skipped, "errors": errors}

    def sync_file(self, slug: str) -> str:
        """Parse + upsert node + recreate edges for one note. Returns node_id."""
        self._sync_node(slug)
        self._sync_edges(slug)
        return f"note:{slug}"

    def _read_file_mtime(self, slot: int) -> float:
        """Read __file_mtime__ directly from ColumnStore (reserved field, hidden from get_node)."""
        cols = self._store.columns
        field = "__file_mtime__"
        if not cols.has_column(field):
            return 0.0
        if not cols._presence[field][slot]:
            return 0.0
        return float(cols._columns[field][slot])

    def _sync_node(self, slug: str) -> int:
        """Upsert the node for a note file. Returns slot."""
        content = self._manager.read(slug)
        fm = parse_frontmatter(content)
        sections = parse_sections(content)

        node_id = f"note:{slug}"

        # Build node fields
        fields = {
            "note_kind": fm.get("kind", "memory"),
            "status": fm.get("status", "active"),
            "title": slug,
            "file": f"{slug}.md",
        }

        tags = fm.get("tags", [])
        if isinstance(tags, list):
            fields["tags"] = ",".join(str(t) for t in tags)
        elif isinstance(tags, str):
            fields["tags"] = tags

        if fm.get("agent"):
            fields["agent"] = fm["agent"]

        # Store summary for columnar search
        summary = sections.get("summary", "")
        if summary:
            fields["summary"] = summary[:self._summary_max_length]

        # Upsert node
        self._store.upsert_node(node_id, "note", fields)

        # Store file mtime as a reserved column for sync detection
        str_id = self._store.string_table.intern(node_id)
        slot = self._store.id_to_slot.get(str_id)

        if slot is not None:
            file_mtime = self._manager.get_mtime(slug)
            self._store.columns.set_reserved(slot, "__file_mtime__", file_mtime)

        # Store full content in DocumentStore
        if self._document_store and slot is not None:
            self._document_store.put_document(slot, content.encode("utf-8"), "text/markdown")
            if summary:
                self._document_store.put_summary(slot, summary[:self._summary_max_length], heading=None,
                                                  page=None, chunk_index=0, doc_slot=slot)

        # Embed summary
        if self._embedder and self._vector_store and slot is not None:
            body = sections.get("body", "")
            embed_text = f"{slug}: {summary} {body}" if summary else body
            if embed_text.strip():
                vec = self._embedder.encode_documents([embed_text])[0]
                if self._vector_store is not None:
                    self._vector_store.add(slot, vec)

        # Fact kind: auto-ASSERT into graphstore belief store
        if fm.get("kind") == "fact" and slot is not None:
            confidence = fm.get("confidence", 1.0)
            source = fm.get("source", "vault")
            if isinstance(confidence, (int, float)):
                self._store.columns.set_reserved(slot, "__confidence__", float(confidence))
            self._store.columns.set_reserved(slot, "__source__", str(source))
            self._store.columns.set_reserved(slot, "__retracted__", 0)

        return slot

    def _sync_edges(self, slug: str) -> None:
        """Recreate wikilink edges for a note."""
        content = self._manager.read(slug)
        wikilinks = extract_wikilinks(content)

        node_id = f"note:{slug}"
        if node_id not in self._store.string_table:
            return

        # Delete old link edges from this node
        existing_link_edges = self._store.get_edges_from(node_id, kind="links")
        for edge in existing_link_edges:
            try:
                self._store.delete_edge(edge["source"], edge["target"], "links")
            except Exception as e:
                logger.debug("vault link edge delete failed: %s", e, exc_info=True)

        # Create new link edges
        for target_slug in wikilinks:
            target_id = f"note:{target_slug}"
            if target_id in self._store.string_table:
                target_str_id = self._store.string_table.intern(target_id)
                if target_str_id in self._store.id_to_slot:
                    target_slot = self._store.id_to_slot[target_str_id]
                    if target_slot not in self._store.node_tombstones:
                        try:
                            self._store.put_edge(node_id, target_id, "links")
                        except Exception as e:
                            logger.debug("vault link edge creation failed: %s", e, exc_info=True)
