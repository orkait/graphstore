"""VaultManager: file I/O for markdown notes."""
import os
from pathlib import Path
from datetime import datetime, timezone

import yaml

from graphstore.vault.parser import (
    parse_frontmatter, parse_sections, extract_wikilinks,
    title_to_slug, write_frontmatter, write_section as _write_section,
)


class VaultManager:
    """Manages markdown note files in a vault directory."""

    def __init__(self, vault_path: str | Path):
        self._path = Path(vault_path)
        self._path.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def new(self, title: str, kind: str = "memory", tags: list[str] | None = None,
            agent: str | None = None, body: str = "", summary: str = "") -> str:
        """Create a new note file. Returns the slug."""
        slug = title_to_slug(title)
        file_path = self._path / f"{slug}.md"

        if file_path.exists():
            raise FileExistsError(f"Note already exists: {slug}.md")

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fm = {
            "kind": kind,
            "tags": tags or [],
            "created": now,
            "updated": now,
            "status": "active",
        }
        if agent:
            fm["agent"] = agent

        fm_str = "---\n"
        fm_str += yaml.dump(fm, default_flow_style=False, sort_keys=False).strip()
        fm_str += "\n---\n"

        content = fm_str
        content += f"\n## Summary\n{summary or 'No summary yet.'}\n"
        content += f"\n## Body\n{body}\n"
        content += "\n## Links\n"

        file_path.write_text(content, encoding="utf-8")
        return slug

    def read(self, title_or_slug: str) -> str:
        """Read full note content. Accepts title or slug."""
        slug = title_to_slug(title_or_slug)
        file_path = self._path / f"{slug}.md"
        if not file_path.exists():
            # Try exact match
            file_path = self._path / f"{title_or_slug}.md"
        if not file_path.exists():
            raise FileNotFoundError(f"Note not found: {title_or_slug}")
        return file_path.read_text(encoding="utf-8")

    def write_section(self, title_or_slug: str, section: str, content: str) -> None:
        """Overwrite a section in a note."""
        slug = title_to_slug(title_or_slug)
        file_path = self._path / f"{slug}.md"
        if not file_path.exists():
            raise FileNotFoundError(f"Note not found: {title_or_slug}")

        old_content = file_path.read_text(encoding="utf-8")
        new_content = _write_section(old_content, section, content)
        new_content = write_frontmatter(new_content, {
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")
        })
        file_path.write_text(new_content, encoding="utf-8")

    def append_section(self, title_or_slug: str, section: str, content: str) -> None:
        """Append to a section in a note."""
        slug = title_to_slug(title_or_slug)
        file_path = self._path / f"{slug}.md"
        if not file_path.exists():
            raise FileNotFoundError(f"Note not found: {title_or_slug}")

        old_content = file_path.read_text(encoding="utf-8")
        sections = parse_sections(old_content)
        existing = sections.get(section.lower(), "")
        new_section_content = f"{existing}\n{content}" if existing else content
        new_content = _write_section(old_content, section, new_section_content)
        new_content = write_frontmatter(new_content, {
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")
        })
        file_path.write_text(new_content, encoding="utf-8")

    def daily(self, agent: str | None = None) -> str:
        """Create or return today's daily note. Returns slug (YYYY-MM-DD)."""
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = self._path / f"{today}.md"

        if not file_path.exists():
            self.new(today, kind="daily", agent=agent, summary=f"Daily note for {today}")

        return today

    def archive(self, title_or_slug: str) -> None:
        """Archive a note (set status = archived)."""
        slug = title_to_slug(title_or_slug)
        file_path = self._path / f"{slug}.md"
        if not file_path.exists():
            raise FileNotFoundError(f"Note not found: {title_or_slug}")

        content = file_path.read_text(encoding="utf-8")
        content = write_frontmatter(content, {
            "status": "archived",
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        file_path.write_text(content, encoding="utf-8")

    def list_files(self) -> list[str]:
        """List all .md files in vault. Returns slugs (without .md)."""
        return [f.stem for f in sorted(self._path.glob("*.md"))]

    def get_mtime(self, slug: str) -> float:
        """Get file modification time for a note."""
        file_path = self._path / f"{slug}.md"
        if not file_path.exists():
            return 0.0
        return file_path.stat().st_mtime
