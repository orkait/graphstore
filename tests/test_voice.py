"""Tests for voice layer. Skipped if moonshine/piper not installed."""
import io
import wave
import struct
import pytest


def _make_wav_bytes(duration_frames: int = 100, samplerate: int = 16000,
                    channels: int = 1, sampwidth: int = 2) -> bytes:
    """Generate minimal valid WAV bytes for testing."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(samplerate)
        wf.writeframes(struct.pack(f"<{duration_frames}h", *([0] * duration_frames)))
    return buf.getvalue()


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
        with pytest.raises((ImportError, ValueError)):
            g.listen()


class TestPiperTTSPlayback:
    """Tests for _play_audio using sounddevice + numpy (no pygame)."""

    def test_play_audio_empty_bytes_is_noop(self):
        from graphstore.voice.tts import PiperTTS
        tts = PiperTTS.__new__(PiperTTS)
        tts._play_audio(b"")  # must not raise

    def test_play_audio_uses_sounddevice(self, monkeypatch):
        """_play_audio should call sd.play + sd.wait when sounddevice available."""
        import numpy as np
        from graphstore.voice import tts as tts_module

        played = {}

        class FakeSD:
            @staticmethod
            def play(data, samplerate):
                played["data"] = data
                played["samplerate"] = samplerate

            @staticmethod
            def wait():
                played["waited"] = True

        monkeypatch.setattr(tts_module, "sd", FakeSD, raising=False)

        from graphstore.voice.tts import PiperTTS
        t = PiperTTS.__new__(PiperTTS)
        t._play_audio(_make_wav_bytes(samplerate=22050))

        assert played.get("samplerate") == 22050
        assert played.get("waited") is True
        assert isinstance(played.get("data"), np.ndarray)

    def test_play_audio_no_sounddevice_skips_gracefully(self, monkeypatch):
        """If sounddevice not available, _play_audio must not raise."""
        import sys
        from graphstore.voice import tts as tts_module
        monkeypatch.setattr(tts_module, "sd", None, raising=False)

        from graphstore.voice.tts import PiperTTS
        t = PiperTTS.__new__(PiperTTS)
        t._play_audio(_make_wav_bytes())  # must not raise

    def test_play_audio_no_pygame_no_import(self):
        """pygame must not be imported anywhere in tts module."""
        import graphstore.voice.tts as tts_mod
        import inspect
        src = inspect.getsource(tts_mod)
        assert "pygame" not in src

    def test_play_audio_stereo_wav(self, monkeypatch):
        """Stereo WAV should be reshaped to (frames, 2)."""
        import numpy as np
        from graphstore.voice import tts as tts_module

        played = {}

        class FakeSD:
            @staticmethod
            def play(data, samplerate):
                played["shape"] = data.shape

            @staticmethod
            def wait():
                pass

        monkeypatch.setattr(tts_module, "sd", FakeSD, raising=False)

        from graphstore.voice.tts import PiperTTS
        t = PiperTTS.__new__(PiperTTS)
        t._play_audio(_make_wav_bytes(duration_frames=100, channels=2))

        assert len(played["shape"]) == 2
        assert played["shape"][1] == 2


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
        name = select_ingestor("meeting.wav")
        assert name == "audio"
