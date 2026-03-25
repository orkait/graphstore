"""Piper TTS wrapper for text-to-speech. CPU only, < 100ms latency."""

import io
import wave
import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
except ImportError:
    sd = None


class PiperTTS:
    """Clear, functional text-to-speech via Piper. CPU only."""

    def __init__(self, voice: str = "en_US-lessac-medium"):
        try:
            import piper
        except ImportError:
            raise ImportError(
                "Piper TTS not installed. Run: graphstore install-voice"
            )
        self._voice_name = voice
        self._piper = piper

    def speak(self, text: str) -> None:
        """Generate and play audio from text."""
        audio = self.synthesize(text)
        self._play_audio(audio)

    def synthesize(self, text: str) -> bytes:
        """Generate WAV audio bytes from text (without playing)."""
        try:
            voice = self._piper.PiperVoice.load(self._voice_name)
            buf = io.BytesIO()
            voice.synthesize(text, buf)
            return buf.getvalue()
        except Exception as e:
            logger.debug("TTS synthesis failed: %s", e, exc_info=True)
            return b""

    def speak_to_file(self, text: str, output_path: str) -> None:
        """Generate audio and save to file."""
        audio = self.synthesize(text)
        with open(output_path, "wb") as f:
            f.write(audio)

    def _play_audio(self, audio_bytes: bytes) -> None:
        """Play WAV audio bytes via sounddevice + numpy."""
        if not audio_bytes:
            return
        if sd is None:
            logger.debug("sounddevice not available, audio playback skipped")
            return
        try:
            buf = io.BytesIO(audio_bytes)
            with wave.open(buf) as wf:
                samplerate = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            dtype = dtype_map.get(sampwidth, np.int16)
            audio = np.frombuffer(frames, dtype=dtype)
            if channels > 1:
                audio = audio.reshape(-1, channels)
            sd.play(audio, samplerate=samplerate)
            sd.wait()
        except Exception as e:
            logger.debug("audio playback failed: %s", e, exc_info=True)
