"""Moonshine Voice STT wrapper for real-time speech-to-text."""
import logging

logger = logging.getLogger(__name__)


class MoonshineSTT:
    """Real-time speech-to-text via Moonshine Voice. CPU only, < 200ms latency.

    Package: pip install moonshine-voice
    Native lib: libmoonshine.so must be on the system for transcription to work.
    See: https://github.com/usefulsensors/moonshine
    """

    def __init__(self, model: str = "tiny-en"):
        try:
            import moonshine_voice
        except ImportError:
            raise ImportError(
                "Moonshine STT not installed. Run: pip install moonshine-voice"
            )
        self._moonshine_voice = moonshine_voice
        self._model = model
        # Transcriber is lazily initialized on first use — construction succeeds
        # even if libmoonshine.so is absent; only transcription will raise.
        self._transcriber = None
        self._listening = False
        self._callback = None

    def _get_transcriber(self):
        if self._transcriber is None:
            from moonshine_voice.transcriber import Transcriber, ModelArch
            model_path = self._moonshine_voice.get_model_path(self._model)
            try:
                self._transcriber = Transcriber(str(model_path), ModelArch.TINY)
            except Exception as e:
                raise RuntimeError(
                    f"Moonshine STT failed to initialize: {e}. "
                    "Ensure libmoonshine.so is installed — "
                    "see https://github.com/usefulsensors/moonshine"
                ) from e
        return self._transcriber

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""
        t = self._get_transcriber()
        audio_data, sample_rate = self._moonshine_voice.load_wav_file(audio_path)
        transcript = t.transcribe_without_streaming(audio_data, sample_rate)
        return " ".join(line.text for line in transcript.lines if line.text).strip()

    def start_listening(self, on_text) -> None:
        """Start real-time streaming STT via event listener."""
        if self._listening:
            raise RuntimeError("Already listening. Call stop_listening() first.")
        from moonshine_voice.transcriber import TranscriptEventListener
        t = self._get_transcriber()
        callback = on_text

        class _Listener(TranscriptEventListener):
            def on_line_completed(self, event):
                if event.line.text:
                    callback(event.line.text)

        self._callback = on_text
        self._listening = True
        t.start()
        t.add_listener(_Listener())

    def stop_listening(self) -> None:
        """Stop streaming STT."""
        if self._listening and self._transcriber is not None:
            try:
                self._transcriber.stop()
                self._transcriber.remove_all_listeners()
            except Exception as e:
                logger.debug("stop_listening error: %s", e)
        self._listening = False
        self._callback = None

    @property
    def is_listening(self) -> bool:
        return self._listening
