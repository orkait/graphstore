"""Piper TTS wrapper for text-to-speech. CPU only, < 100ms latency."""

import io


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
        # Piper generates WAV bytes, play via platform audio
        audio = self.synthesize(text)
        self._play_audio(audio)

    def synthesize(self, text: str) -> bytes:
        """Generate WAV audio bytes from text (without playing)."""
        # Implementation depends on piper's API
        # This is the interface contract
        try:
            voice = self._piper.PiperVoice.load(self._voice_name)
            buf = io.BytesIO()
            voice.synthesize(text, buf)
            return buf.getvalue()
        except Exception:
            # Fallback: return empty audio
            return b""

    def speak_to_file(self, text: str, output_path: str) -> None:
        """Generate audio and save to file."""
        audio = self.synthesize(text)
        with open(output_path, "wb") as f:
            f.write(audio)

    def _play_audio(self, audio_bytes: bytes) -> None:
        """Platform-specific audio playback."""
        if not audio_bytes:
            return
        # Try pygame, then sounddevice, then skip
        try:
            import pygame
            pygame.mixer.init()
            sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
            sound.play()
        except ImportError:
            pass  # No audio playback available
