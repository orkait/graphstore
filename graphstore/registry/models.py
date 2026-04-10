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
        "pooling": "mean",
        "deps": ["onnxruntime", "tokenizers", "huggingface_hub"],
        "variants": {
            "fp32": {"files": ["onnx/model.onnx", "onnx/model.onnx_data"]},
            "q4": {"files": ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data"]},
        },
        "default_variant": "q4",
    },
    "harrier-oss-v1-0.6b": {
        "family": "hf_onnx",
        "repo_id": "onnx-community/harrier-oss-v1-0.6b-ONNX",
        "description": (
            "Microsoft Harrier v1 0.6B (Qwen3-based) via ONNX. "
            "MTEB v2 69.0, 1024 dims, 32k context, multilingual, last-token pooling. "
            "fp16 is the default — q4 variants use GatherBlockQuantized with a 'bits' "
            "attribute unsupported by onnxruntime 1.20.x."
        ),
        "max_length": 2048,
        "base_dims": 1024,
        "allowed_dims": [1024],
        "default_dims": 1024,
        "query_prefix": (
            "Instruct: Given a web search query, retrieve relevant passages that "
            "answer the query\nQuery: "
        ),
        "doc_prefix_template": "",
        "pooling": "last_token",
        "deps": ["onnxruntime", "tokenizers", "huggingface_hub"],
        "variants": {
            "fp16": {"files": ["onnx/model_fp16.onnx", "onnx/model_fp16.onnx_data"]},
            "q4": {"files": ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data"]},
            "q4f16": {"files": ["onnx/model_q4f16.onnx", "onnx/model_q4f16.onnx_data"]},
        },
        "default_variant": "fp16",
    },
}


def get_model_info(name: str) -> dict | None:
    return SUPPORTED_MODELS.get(name)


def list_models() -> list[dict]:
    return [{"name": k, **v} for k, v in SUPPORTED_MODELS.items()]
