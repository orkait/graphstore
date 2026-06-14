"""Cloud multimodal understanding: media bytes -> text, for ingestion.

graphstore OWNS understanding media (image / audio / ...). It is an ingestion
function - media in, understood text out, then stored + embedded like any other
DOCUMENT - the same shape as NL->DSL ingestion. Reuses the cloud LLMRunner +
free-first provider chain. A caller (e.g. a harness) only fetches the bytes and
hands them over; the understanding lives here, not in the caller.
"""
from __future__ import annotations

import base64

from graphstore.ingest.llm.resolve import build_provider_chain
from graphstore.llm_runner import LLMRunner

DEFAULT_VISION_MODELS = [
    "openrouter/openai/gpt-4o-mini",
    "openrouter/google/gemini-2.5-flash",
]
_IMAGE_PROMPT = (
    "Describe this image in detail - objects, any visible text, people, setting, "
    "colors, and anything notable. Be factual and concise."
)
_AUDIO_PROMPT = "Transcribe this audio. Return the transcript text only."


class MediaUnsupported(ValueError):
    """Raised when a mime type can't be understood by the cloud path."""


def _messages(data: bytes, mime: str, prompt: str | None) -> list[dict]:
    mime = (mime or "").lower()
    b64 = base64.b64encode(data).decode()
    if mime.startswith("image/"):
        return [{"role": "user", "content": [
            {"type": "text", "text": prompt or _IMAGE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}]
    if mime.startswith("audio/"):
        fmt = mime.split("/", 1)[1].split(";")[0] or "mp3"
        return [{"role": "user", "content": [
            {"type": "text", "text": prompt or _AUDIO_PROMPT},
            {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
        ]}]
    raise MediaUnsupported(
        f"no cloud understanding path for mime {mime!r} "
        "(supported: image/*, audio/*; video/doc need extraction first)"
    )


def understand_media(
    data: bytes, mime: str, *, models: list[str] | None = None,
    prompt: str | None = None, max_tokens: int = 512,
) -> str:
    """Understand media bytes into descriptive text via a cloud multimodal model.

    image/* -> a detailed visual description; audio/* -> a transcript. Raises
    MediaUnsupported for mimes with no inline path, RuntimeError if no provider
    key is configured.
    """
    messages = _messages(data, mime, prompt)
    chain = build_provider_chain(models or DEFAULT_VISION_MODELS)
    if not chain:
        raise RuntimeError(
            "no cloud providers resolved; set a provider key "
            "(e.g. OPENROUTER_API_KEY / GOOGLE_AISTUDIO_API_KEY)"
        )
    text = LLMRunner(chain).complete_messages(messages, max_tokens=max_tokens, temperature=0.0)
    return (text or "").strip()


__all__ = ["MediaUnsupported", "understand_media", "DEFAULT_VISION_MODELS"]
