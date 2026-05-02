# graphstore - Pro GPU image.
#
# CUDA 12.4 + cuDNN 9 runtime base. Installs graphstore[pro] and
# pre-downloads every model the default ProSpec selects (Bonsai TQ1_0
# GGUF, TinyBERT NER ONNX, Jina v5 small ONNX) at build time so the
# container starts cold-ready.
#
# Calibration is intentionally NOT run at build time - calibration is
# host-specific (per-CPU/per-GPU TPS measurements) and a value measured
# on a CI runner means nothing on the deployment host. Run it once on
# the target machine inside the container:
#
#     docker compose --profile pro run --rm graphstore-pro graphstore pro setup
#
# The calibration cache lives on the gs-cache volume and persists across
# container restarts.
#
# Build:  docker build -f Dockerfile.pro --cpus=8 --memory=16g -t graphstore-pro:latest .
# Run:    docker run --gpus all --cpus=8 --memory=16g -p 7200:7200 \
#               -v gs-data:/data -v gs-cache:/root/.cache/graphstore \
#               graphstore-pro:latest

ARG GRAPHSTORE_VERSION=0.6.0
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

FROM ${CUDA_IMAGE}

ARG GRAPHSTORE_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRAPHSTORE_DB_PATH=/data \
    GRAPHSTORE_HOST=0.0.0.0 \
    GRAPHSTORE_PORT=7200 \
    HF_HOME=/root/.cache/huggingface

# Python 3.12 from the deadsnakes PPA. cuDNN 9 already lives in the
# base image; we only need Python + curl + git for install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 100 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100 \
    && rm -rf /var/lib/apt/lists/*

# Install graphstore + pro extras. llama-cpp-python in [pro] is the
# CPU-only wheel; users who want GPU offload for Bonsai install the CUDA
# wheel themselves at runtime via:
#   pip install llama-cpp-python --extra-index-url \
#       https://abetlen.github.io/llama-cpp-python/whl/cu124
RUN python -m pip install --break-system-packages \
        "graphstore[pro]==${GRAPHSTORE_VERSION}"

# Pre-pull every model the default ProSpec selects so a cold container
# is functional immediately. ~5 GB total. Skip with --build-arg
# SKIP_MODEL_PREFETCH=1 for a slim image (first start downloads).
ARG SKIP_MODEL_PREFETCH=0
RUN if [ "$SKIP_MODEL_PREFETCH" = "0" ]; then \
        python -c "from huggingface_hub import hf_hub_download; \
            hf_hub_download('superkaiii/Ternary-Bonsai-4B-TQ1_0-GGUF', \
                            'Ternary-Bonsai-4B-TQ1_0.gguf')"; \
        python -c "from huggingface_hub import snapshot_download; \
            snapshot_download('jinaai/jina-embeddings-v3', \
                              allow_patterns=['*.onnx','*.json','*.txt'])"; \
        python -c "from huggingface_hub import snapshot_download; \
            snapshot_download('Xenova/distilbert-base-multilingual-cased-finetuned-conll03-english')"; \
    fi

COPY docker/entrypoint.sh /usr/local/bin/graphstore-entrypoint
RUN chmod +x /usr/local/bin/graphstore-entrypoint

WORKDIR /root
RUN mkdir -p /data /root/.cache/graphstore
VOLUME ["/data", "/root/.cache/graphstore", "/root/.cache/huggingface"]

EXPOSE 7200

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(3); \
        s.connect(('127.0.0.1', 7200)); s.close()" || exit 1

ENTRYPOINT ["/usr/local/bin/graphstore-entrypoint"]
