"""Tests for vault module: parser, manager, sync."""
import pytest
from pathlib import Path

from graphstore.vault.parser import (
    parse_frontmatter, parse_sections, extract_wikilinks,
    title_to_slug, write_frontmatter, write_section,
)
from graphstore.vault.manager import VaultManager


SAMPLE_NOTE = """---
kind: memory
tags: [research, ai]
created: 2026-03-21T10:00:00
updated: 2026-03-21T10:00:00
status: active
agent: claude
---

## Summary
This is a summary about AI research.

## Body
Detailed content here.

## Links
- [[related-note]]
- [[another-note]]
"""


class TestParser:
    def test_parse_frontmatter(self):
        fm = parse_frontmatter(SAMPLE_NOTE)
        assert fm["kind"] == "memory"
        assert fm["status"] == "active"
        assert "research" in fm["tags"]

    def test_parse_frontmatter_empty(self):
        assert parse_frontmatter("no frontmatter here") == {}

    def test_parse_sections(self):
        sections = parse_sections(SAMPLE_NOTE)
        assert "summary" in sections
        assert "body" in sections
        assert "links" in sections
        assert "AI research" in sections["summary"]

    def test_extract_wikilinks(self):
        links = extract_wikilinks(SAMPLE_NOTE)
        assert "related-note" in links
        assert "another-note" in links

    def test_title_to_slug(self):
        assert title_to_slug("My Cool Note!") == "my-cool-note"
        assert title_to_slug("2026-03-21") == "2026-03-21"
        assert title_to_slug("  spaces  ") == "spaces"

    def test_write_frontmatter(self):
        updated = write_frontmatter(SAMPLE_NOTE, {"status": "archived"})
        fm = parse_frontmatter(updated)
        assert fm["status"] == "archived"
        assert fm["kind"] == "memory"  # preserved

    def test_write_section(self):
        updated = write_section(SAMPLE_NOTE, "body", "New body content")
        sections = parse_sections(updated)
        assert sections["body"] == "New body content"
        assert "AI research" in sections["summary"]  # other sections preserved

    def test_write_section_nonexistent_appends(self):
        updated = write_section(SAMPLE_NOTE, "instructions", "Do this thing")
        sections = parse_sections(updated)
        assert "instructions" in sections
        assert "Do this thing" in sections["instructions"]


class TestManager:
    @pytest.fixture
    def vault(self, tmp_path):
        return VaultManager(tmp_path / "notes")

    def test_new_creates_file(self, vault):
        slug = vault.new("My Research Note", kind="memory", tags=["ai", "ml"])
        assert slug == "my-research-note"
        assert (vault.path / "my-research-note.md").exists()

    def test_new_duplicate_raises(self, vault):
        vault.new("Test Note")
        with pytest.raises(FileExistsError):
            vault.new("Test Note")

    def test_read(self, vault):
        vault.new("Read Test", body="Hello world")
        content = vault.read("Read Test")
        assert "Hello world" in content
        assert "## Summary" in content

    def test_read_not_found(self, vault):
        with pytest.raises(FileNotFoundError):
            vault.read("nonexistent")

    def test_write_section(self, vault):
        vault.new("Write Test", body="old body")
        vault.write_section("Write Test", "body", "new body")
        content = vault.read("Write Test")
        assert "new body" in content
        assert "old body" not in content

    def test_append_section(self, vault):
        vault.new("Append Test", body="line 1")
        vault.append_section("Append Test", "body", "line 2")
        content = vault.read("Append Test")
        assert "line 1" in content
        assert "line 2" in content

    def test_daily(self, vault):
        slug = vault.daily()
        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        assert slug == today
        assert (vault.path / f"{today}.md").exists()

    def test_daily_idempotent(self, vault):
        slug1 = vault.daily()
        slug2 = vault.daily()
        assert slug1 == slug2

    def test_archive(self, vault):
        vault.new("Archive Me")
        vault.archive("Archive Me")
        content = vault.read("Archive Me")
        fm = parse_frontmatter(content)
        assert fm["status"] == "archived"

    def test_list_files(self, vault):
        vault.new("Note A")
        vault.new("Note B")
        files = vault.list_files()
        assert "note-a" in files
        assert "note-b" in files


class TestVaultSync:
    def test_sync_creates_graph_nodes(self, tmp_path):
        from graphstore import GraphStore
        vault_path = tmp_path / "notes"
        vault_path.mkdir()

        # Write a note file manually
        (vault_path / "test-note.md").write_text("""---
kind: memory
tags: [test]
created: 2026-03-21T10:00:00
updated: 2026-03-21T10:00:00
status: active
---

## Summary
A test note for sync.

## Body
Content here.

## Links
""")

        g = GraphStore(path=str(tmp_path / "db"), vault=str(vault_path), embedder=None)
        # Sync should have run on init
        node = g.execute('NODE "note:test-note"')
        assert node.data is not None
        assert node.data["note_kind"] == "memory"
        assert node.data["status"] == "active"
        g.close()

    def test_sync_creates_wikilink_edges(self, tmp_path):
        from graphstore import GraphStore
        vault_path = tmp_path / "notes"
        vault_path.mkdir()

        (vault_path / "note-a.md").write_text("""---
kind: memory
status: active
---

## Summary
Note A

## Links
- [[note-b]]
""")
        (vault_path / "note-b.md").write_text("""---
kind: memory
status: active
---

## Summary
Note B
""")

        g = GraphStore(path=str(tmp_path / "db"), vault=str(vault_path), embedder=None)
        edges = g.execute('EDGES FROM "note:note-a"')
        link_edges = [e for e in edges.data if e["kind"] == "links"]
        assert len(link_edges) == 1
        assert link_edges[0]["target"] == "note:note-b"
        g.close()

    def test_vault_sync_api(self, tmp_path):
        """Test sync via Python API (VAULT SYNC DSL not yet wired)."""
        from graphstore import GraphStore
        vault_path = tmp_path / "notes"
        vault_path.mkdir()
        (vault_path / "sync-test.md").write_text("""---
kind: fact
status: active
---

## Summary
Sync test note
""")
        g = GraphStore(path=str(tmp_path / "db"), vault=str(vault_path), embedder=None)
        # Re-sync via Python API
        result = g._vault_sync.sync_all()
        assert result["synced"] >= 0
        assert result["errors"] == 0
        g.close()

    def test_sync_skips_unchanged(self, tmp_path):
        """Second sync_all should skip already-synced notes."""
        from graphstore import GraphStore
        vault_path = tmp_path / "notes"
        vault_path.mkdir()
        (vault_path / "stable-note.md").write_text("""---
kind: memory
status: active
---

## Summary
Stable note
""")
        g = GraphStore(path=str(tmp_path / "db"), vault=str(vault_path), embedder=None)
        # First sync happened in __init__, second should skip
        result = g._vault_sync.sync_all()
        assert result["skipped"] >= 1
        assert result["synced"] == 0
        g.close()
