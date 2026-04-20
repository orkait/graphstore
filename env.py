"""Repo-wide env loader. Reads /.env, exposes typed ENV.

Callers do ``ENV["openrouter_key"]`` or ``ENV.openrouter_key``.
Shell env wins over .env file. Missing .env is silent (CI / prod use
real env vars). Typos raise KeyError from __getitem__.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_DOTENV = Path(__file__).resolve().parent / ".env"
if _DOTENV.exists():
    for _line in _DOTENV.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class _Env:
    openrouter_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    ollama_key: str = os.environ.get("OLLAMA_API_KEY", "ollama")

    def __getitem__(self, k: str) -> str:
        return vars(self)[k]


ENV = _Env()
