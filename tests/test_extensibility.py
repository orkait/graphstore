"""Tests for extensibility injection points."""
import pytest
from graphstore.ingest.base import Ingestor, IngestResult


class _DummyIngestor(Ingestor):
    name = "dummy"
    supported_extensions = ["xyz"]

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        return IngestResult(markdown="dummy", parser_used="dummy")


class TestIngestorRegistry:
    def test_register_and_resolve_by_extension(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        ingestor = _DummyIngestor()
        reg.register(ingestor)
        resolved = reg.resolve("file.xyz")
        assert resolved is ingestor

    def test_resolve_unknown_extension_raises(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        with pytest.raises(ValueError, match="Unsupported format"):
            reg.resolve("file.unknownext999")

    def test_resolve_using_unknown_name_raises(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        # using= with an unknown builtin name raises ValueError from _make_builtin_ingestor
        with pytest.raises(ValueError):
            reg.resolve("file.txt", using="nonexistent_parser_xyz")

    def test_override_existing_extension(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()

        class MyPDFIngestor(Ingestor):
            name = "mypdf"
            supported_extensions = ["pdf"]
            def convert(self, path, **kwargs):
                return IngestResult(markdown="mypdf", parser_used="mypdf")

        reg.register(MyPDFIngestor())
        resolved = reg.resolve("report.pdf")
        assert resolved.name == "mypdf"

    def test_builtin_pdf_resolves(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        ingestor = reg.resolve("report.pdf")
        assert ingestor is not None
        assert ingestor.name in ("pymupdf4llm", "markitdown")

    def test_builtin_txt_resolves(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        ingestor = reg.resolve("notes.txt")
        assert ingestor.name == "markitdown"

    def test_list_returns_registered(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        reg.register(_DummyIngestor())
        entries = reg.list()
        names = [e["name"] for e in entries]
        assert "dummy" in names

    def test_using_override_bypasses_extension_map(self):
        from graphstore.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        reg.register(_DummyIngestor())
        resolved = reg.resolve("report.pdf", using="dummy")
        assert resolved.name == "dummy"


class TestChunkerProtocol:
    def test_heading_chunker_implements_protocol(self):
        from graphstore.voice.protocol import ChunkerProtocol
        from graphstore.ingest.chunker import HeadingChunker
        chunker = HeadingChunker()
        assert isinstance(chunker, ChunkerProtocol)

    def test_heading_chunker_chunks_text(self):
        from graphstore.ingest.chunker import HeadingChunker
        chunker = HeadingChunker()
        chunks = chunker.chunk("# Heading\nsome text")
        assert len(chunks) >= 1
        assert chunks[0].heading == "Heading"

    def test_heading_chunker_passes_kwargs(self):
        from graphstore.ingest.chunker import HeadingChunker
        chunker = HeadingChunker()
        chunks = chunker.chunk("# H\n" + "word " * 1000, max_chunk_size=200)
        assert len(chunks) > 1  # kwargs were honored

    def test_custom_chunker_satisfies_protocol(self):
        from graphstore.voice.protocol import ChunkerProtocol
        from graphstore.ingest.base import Chunk

        class SingleChunker:
            def chunk(self, text: str, **kwargs):
                return [Chunk(text=text, summary=text[:50], index=0)]

        assert isinstance(SingleChunker(), ChunkerProtocol)

    def test_object_without_chunk_method_fails_protocol(self):
        from graphstore.voice.protocol import ChunkerProtocol

        class NotAChunker:
            pass

        assert not isinstance(NotAChunker(), ChunkerProtocol)


class TestVoiceProtocols:
    def test_moonshine_stt_satisfies_protocol_when_installed(self):
        pytest.importorskip("moonshine")
        from graphstore.voice.protocol import STTProtocol
        from graphstore.voice.stt import MoonshineSTT
        stt = MoonshineSTT()
        assert isinstance(stt, STTProtocol)

    def test_piper_tts_satisfies_protocol_when_installed(self):
        pytest.importorskip("piper")
        from graphstore.voice.protocol import TTSProtocol
        from graphstore.voice.tts import PiperTTS
        tts = PiperTTS.__new__(PiperTTS)
        assert isinstance(tts, TTSProtocol)

    def test_custom_stt_satisfies_protocol(self):
        from graphstore.voice.protocol import STTProtocol

        class StubSTT:
            def transcribe_file(self, audio_path: str) -> str:
                return "hello"
            def start_listening(self, on_text) -> None:
                pass
            def stop_listening(self) -> None:
                pass
            @property
            def is_listening(self) -> bool:
                return False

        stt = StubSTT()
        # runtime_checkable checks method names only (not signatures)
        assert isinstance(stt, STTProtocol)
        # Also verify callable contract
        assert stt.transcribe_file("x") == "hello"
        assert stt.is_listening is False

    def test_custom_tts_satisfies_protocol(self):
        from graphstore.voice.protocol import TTSProtocol

        class StubTTS:
            def speak(self, text: str) -> None:
                pass
            def synthesize(self, text: str) -> bytes:
                return b""

        tts = StubTTS()
        assert isinstance(tts, TTSProtocol)
        assert tts.synthesize("hi") == b""

    def test_object_missing_methods_fails_stt_protocol(self):
        from graphstore.voice.protocol import STTProtocol

        class IncompleteSTT:
            def transcribe_file(self, path: str) -> str:
                return ""
            # Missing start_listening, stop_listening, is_listening

        assert not isinstance(IncompleteSTT(), STTProtocol)
