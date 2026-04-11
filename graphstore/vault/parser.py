"""Parse markdown notes: frontmatter, sections, wikilinks."""
import re
from datetime import datetime

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "graphstore.vault requires the `vault` extra. "
        "Install with: pip install 'graphstore[vault]'"
    ) from e


def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown. Returns {} if no frontmatter."""
    match = re.match(r'^---\n(.*?\n)---\n', content, re.DOTALL)
    if not match:
        return {}
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}


def parse_sections(content: str) -> dict[str, str]:
    """Extract ## sections from markdown. Returns {section_name: content}."""
    # Strip frontmatter first
    body = re.sub(r'^---\n.*?\n---\n', '', content, count=1, flags=re.DOTALL)
    sections = {}
    current_name = None
    current_lines = []

    for line in body.split('\n'):
        heading_match = re.match(r'^## (.+)$', line)
        if heading_match:
            if current_name is not None:
                sections[current_name.lower()] = '\n'.join(current_lines).strip()
            current_name = heading_match.group(1)
            current_lines = []
        else:
            current_lines.append(line)

    if current_name is not None:
        sections[current_name.lower()] = '\n'.join(current_lines).strip()

    return sections


def extract_wikilinks(content: str) -> list[str]:
    """Extract [[wikilink]] targets from content. Returns list of slugs."""
    return [_title_to_slug(m) for m in re.findall(r'\[\[([^\]]+)\]\]', content)]


def _title_to_slug(title: str) -> str:
    """Convert title to kebab-case slug."""
    slug = title.lower().strip()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[\s]+', '-', slug)
    slug = re.sub(r'-+', '-', slug)
    return slug.strip('-')


def title_to_slug(title: str) -> str:
    """Public alias for slug conversion."""
    return _title_to_slug(title)


def write_frontmatter(content: str, updates: dict) -> str:
    """Update frontmatter fields in markdown content. Preserves existing fields."""
    fm = parse_frontmatter(content)
    fm.update(updates)
    # Rebuild content
    body = re.sub(r'^---\n.*?\n---\n', '', content, count=1, flags=re.DOTALL)
    fm_str = yaml.dump(fm, default_flow_style=False, sort_keys=False).strip()
    return f"---\n{fm_str}\n---\n{body}"


def write_section(content: str, section: str, new_content: str) -> str:
    """Replace a section's content in markdown. Case-insensitive section match."""
    lines = content.split('\n')
    result = []
    in_target = False
    replaced = False
    section_lower = section.lower()

    for line in lines:
        heading_match = re.match(r'^## (.+)$', line)
        if heading_match:
            if in_target:
                # End of target section - insert new content
                result.append(new_content)
                result.append('')
                in_target = False
                replaced = True
            if heading_match.group(1).lower() == section_lower:
                result.append(line)
                in_target = True
                continue
        if not in_target:
            result.append(line)

    if in_target:
        # Target section was the last section
        result.append(new_content)
        replaced = True

    if not replaced:
        # Section didn't exist, append it
        result.append(f'\n## {section.capitalize()}')
        result.append(new_content)

    return '\n'.join(result)
