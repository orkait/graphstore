"""Runtime-checkable protocols for voice and chunking subsystems.

Note: Python's isinstance() on runtime_checkable Protocols only checks
that required method *names* exist — it does NOT check signatures or
return types. Use mypy/pyright for full structural type checking.
"""
from typing import Protocol, runtime_checkable
from graphstore.ingest.base import Chunk


@runtime_checkable
class ChunkerProtocol(Protocol):
    """Protocol for text chunking implementations."""

    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        ...


@runtime_checkable
class STTProtocol(Protocol):
    """Protocol for speech-to-text implementations."""

    def transcribe_file(self, audio_path: str) -> str:
        ...

    def start_listening(self, on_text) -> None:
        ...

    def stop_listening(self) -> None:
        ...

    @property
    def is_listening(self) -> bool:
        ...


@runtime_checkable
class TTSProtocol(Protocol):
    """Protocol for text-to-speech implementations."""

    def speak(self, text: str) -> None:
        ...

    def synthesize(self, text: str) -> bytes:
        ...
