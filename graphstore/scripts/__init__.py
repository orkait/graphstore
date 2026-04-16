"""GraphStore scripts for downloading models and test fixtures.

Usage:
    from scripts import download_models, download_fixtures

    download_models.download_all()
    download_fixtures.download_all()

Or from command line:
    python -m scripts.download_models
    python -m scripts.download_fixtures
"""

from . import download_fixtures, download_models

__all__ = ["download_models", "download_fixtures"]
