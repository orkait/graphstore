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
            "Default variant q4 - 40 ms/encode on CPU, 5x faster than fp16 with "
            "equivalent retrieval quality. Requires onnxruntime >= 1.21 for the "
            "GatherBlockQuantized operator."
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
        "default_variant": "q4",
    },
    "qwen3-embedding-0.6b": {
        "family": "gguf",
        "repo_id": "Qwen/Qwen3-Embedding-0.6B-GGUF",
        "description": "Qwen3 Embedding 0.6B via GGUF. MTEB 70.7, 1024 dims, multilingual.",
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
        "deps": ["llama_cpp", "huggingface_hub"],
        "variants": {
            "Q8_0": {"files": ["Qwen3-Embedding-0.6B-Q8_0.gguf"]},
            "f16": {"files": ["Qwen3-Embedding-0.6B-f16.gguf"]},
        },
        "default_variant": "Q8_0",
    },
    "harrier-oss-v1-0.6b-gguf": {
        "family": "gguf",
        "repo_id": "mradermacher/harrier-oss-v1-0.6b-GGUF",
        "description": "Microsoft Harrier v1 0.6B via GGUF. MTEB 69.0, 1024 dims, Q2-Q8 quant tiers.",
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
        "deps": ["llama_cpp", "huggingface_hub"],
        "variants": {
            "Q4_K_M": {"files": ["harrier-oss-v1-0.6b.Q4_K_M.gguf"]},
            "Q8_0": {"files": ["harrier-oss-v1-0.6b.Q8_0.gguf"]},
            "f16": {"files": ["harrier-oss-v1-0.6b.f16.gguf"]},
        },
        "default_variant": "Q4_K_M",
    },
}


def get_model_info(name: str) -> dict | None:
    return SUPPORTED_MODELS.get(name)


def list_models() -> list[dict]:
    return [{"name": k, **v} for k, v in SUPPORTED_MODELS.items()]
