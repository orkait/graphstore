"""Pytest hooks for the graphstore test suite.

Auto-skip tests that need an optional extra that isn't installed.

Two levels:

  1. Test files that import a feature module at module level (and would
     therefore crash during pytest collection) are listed in
     ``_FILES_REQUIRING`` - this conftest tells pytest to ``collect_ignore``
     them when their extra is missing.

  2. Test files that boot cleanly without the extra but whose test bodies
     hit the feature path use ``pytestmark = pytest.mark.needs_<extra>``
     (or a class-level ``@pytest.mark.needs_<extra>``). This conftest
     translates those markers into dynamic skips at collection time.

Either way, an individual test file never has to call ``importorskip``.
"""

from __future__ import annotations

import importlib.util

import pytest


_EXTRA_TO_DEP: dict[str, tuple[str, ...]] = {
    "needs_embedder": ("model2vec",),
    "needs_fastembed": ("fastembed",),
    "needs_ingest": ("markitdown", "pymupdf"),
    "needs_vault": ("yaml",),
    "needs_scheduler": ("croniter",),
    "needs_playground": ("fastapi", "pydantic"),
    "needs_gpu": ("onnxruntime",),
    "needs_voice": ("sounddevice",),
}

# Files that crash at collection time when the listed extra is missing
# because they import the feature module at the top. For these we use
# pytest's collect_ignore mechanism instead of per-item skip markers.
_FILES_REQUIRING: dict[str, str] = {
    "test_vault.py": "needs_vault",
    "test_server.py": "needs_playground",
    "test_server_endpoints.py": "needs_playground",
    "test_server_security.py": "needs_playground",
    "test_voice.py": "needs_voice",
    "test_ingest.py": "needs_ingest",
}


def _is_installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _extras_available() -> dict[str, bool]:
    return {
        marker: all(_is_installed(mod) for mod in deps)
        for marker, deps in _EXTRA_TO_DEP.items()
    }


collect_ignore = [
    fname
    for fname, marker in _FILES_REQUIRING.items()
    if not all(_is_installed(m) for m in _EXTRA_TO_DEP[marker])
]


def pytest_configure(config: pytest.Config) -> None:
    for marker in _EXTRA_TO_DEP:
        config.addinivalue_line(
            "markers",
            f"{marker}: skipped unless the matching graphstore extra is installed",
        )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    available = _extras_available()
    for item in items:
        for marker, ok in available.items():
            if ok:
                continue
            if item.get_closest_marker(marker) is None:
                continue
            missing = ", ".join(_EXTRA_TO_DEP[marker])
            item.add_marker(
                pytest.mark.skip(reason=f"requires optional deps: {missing}")
            )
