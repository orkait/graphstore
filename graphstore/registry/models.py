"""Supported models registry."""

SUPPORTED_MODELS = {
    "embeddinggemma-300m": {
        "family": "hf_onnx",
        "repo_id": "onnx-community/embeddinggemma-300m-ONNX",
        "description": "Google EmbeddingGemma 300M via ONNX. High quality, Matryoshka dims.",
        "max_length": 2048,
        "base_dims": 768,
        "allowed_dims": [768, 512, 256, 128],
        "default_dims": 256,
        "query_prefix": "task: search result | query: ",
        "doc_prefix_template": "title: {title} | text: ",
        "deps": ["onnxruntime", "tokenizers", "huggingface_hub"],
        "variants": {
            "fp32": {"files": ["onnx/model.onnx", "onnx/model.onnx_data"]},
            "q4": {"files": ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data"]},
        },
        "default_variant": "q4",
    },
}


def get_model_info(name: str) -> dict | None:
    return SUPPORTED_MODELS.get(name)


def list_models() -> list[dict]:
    return [{"name": k, **v} for k, v in SUPPORTED_MODELS.items()]
