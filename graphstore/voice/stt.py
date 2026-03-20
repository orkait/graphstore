"""Moonshine STT wrapper for real-time speech-to-text."""


class MoonshineSTT:
    """Real-time speech-to-text via Moonshine. CPU only, < 200ms latency."""

    def __init__(self):
        try:
            import moonshine
        except ImportError:
            raise ImportError(
                "Moonshine STT not installed. Run: graphstore install-voice"
            )
        self._moonshine = moonshine
        self._listening = False
        self._callback = None

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""
        result = self._moonshine.transcribe(audio_path)
        if isinstance(result, list):
            return " ".join(result)
        return str(result)

    def start_listening(self, on_text) -> None:
        """Start real-time streaming STT."""
        if self._listening:
            raise RuntimeError("Already listening. Call stop_listening() first.")
        self._callback = on_text
        self._listening = True
        # Note: actual streaming implementation depends on moonshine's API
        # This is the interface contract

    def stop_listening(self) -> None:
        """Stop streaming STT."""
        self._listening = False
        self._callback = None

    @property
    def is_listening(self) -> bool:
        return self._listening
