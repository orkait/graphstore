"""VisionHandler: image understanding via Ollama (SmolVLM2 / Qwen3-VL)."""
import base64
import logging

logger = logging.getLogger(__name__)


class VisionHandler:
    """Connects to Ollama for image description. Tier 4 fallback."""

    def __init__(self, model: str = "smolvlm2:2.2b", base_url: str = "http://localhost:11434/v1",
                 max_tokens: int = 300):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("VisionHandler requires openai package. pip install openai")
        self._client = OpenAI(base_url=base_url, api_key="ollama")
        self._model = model
        self._max_tokens = max_tokens

    @property
    def client(self):
        return self._client

    @property
    def model(self):
        return self._model

    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        b64 = base64.b64encode(image_bytes).decode()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image concisely. Focus on data, text, and key elements."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                ],
            }],
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception as e:
            logger.debug("vision availability check failed: %s", e, exc_info=True)
            return False
