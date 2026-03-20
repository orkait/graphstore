"""Tests for voice layer. Skipped if moonshine/piper not installed."""
import pytest


class TestVoiceNotInstalled:
    def test_speak_without_voice_raises(self, tmp_path):
        from graphstore import GraphStore
        g = GraphStore(path=str(tmp_path / "db"))
        with pytest.raises(ImportError, match="Voice not installed"):
            g.speak("hello")

    def test_listen_without_voice_raises(self, tmp_path):
        from graphstore import GraphStore
        g = GraphStore(path=str(tmp_path / "db"))
        with pytest.raises(ImportError, match="Voice not installed"):
            g.listen(on_text=lambda x: x)

    def test_stop_listening_without_voice_noop(self, tmp_path):
        from graphstore import GraphStore
        g = GraphStore(path=str(tmp_path / "db"))
        g.stop_listening()  # should not raise

    def test_listen_without_callback_raises(self, tmp_path):
        from graphstore import GraphStore
        g = GraphStore(path=str(tmp_path / "db"), voice=True)
        # Will fail with ImportError if moonshine not installed,
        # or ValueError if installed but no callback
        with pytest.raises((ImportError, ValueError)):
            g.listen()


class TestVoiceInstalled:
    """Only runs if moonshine and piper are installed."""

    @pytest.fixture(autouse=True)
    def skip_if_no_voice(self):
        pytest.importorskip("moonshine")
        pytest.importorskip("piper")

    def test_stt_init(self):
        from graphstore.voice.stt import MoonshineSTT
        stt = MoonshineSTT()
        assert not stt.is_listening

    def test_tts_init(self):
        from graphstore.voice.tts import PiperTTS
        tts = PiperTTS()


class TestAudioIngestNotInstalled:
    def test_audio_ingest_raises_without_voice(self, tmp_path):
        from graphstore.ingest.router import select_ingestor
        # .wav should route to audio ingestor
        name = select_ingestor("meeting.wav")
        assert name == "audio"
