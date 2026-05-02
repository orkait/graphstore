# graphstore - CPU image (default).
#
# Multi-stage build. The builder installs the wheel + extras; the runtime
# stage copies only the populated site-packages, keeping the final image
# small (~250 MB). Playground HTTP server listens on 0.0.0.0:7200.
#
# Build:   docker build --cpus=8 --memory=16g -t graphstore:latest .
# Run:     docker run --cpus=8 --memory=16g -p 7200:7200 -v gs-data:/data graphstore:latest
#
# Override the version with --build-arg GRAPHSTORE_VERSION=0.6.0 to pin
# the wheel; default tracks the latest published release.

ARG GRAPHSTORE_VERSION=0.6.0

# ----- builder ------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ARG GRAPHSTORE_VERSION
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install into /install so the runtime stage can copy it as one layer.
RUN pip install --prefix=/install \
        "graphstore[playground]==${GRAPHSTORE_VERSION}"

# ----- runtime ------------------------------------------------------------
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRAPHSTORE_DB_PATH=/data \
    GRAPHSTORE_HOST=0.0.0.0 \
    GRAPHSTORE_PORT=7200

# Copy the installed package tree (no compiler toolchain needed at runtime).
COPY --from=builder /install /usr/local
COPY docker/entrypoint.sh /usr/local/bin/graphstore-entrypoint
RUN chmod +x /usr/local/bin/graphstore-entrypoint

# Persistent volume root + non-root user for least privilege.
RUN useradd --create-home --shell /bin/bash graphstore \
    && mkdir -p /data \
    && chown graphstore:graphstore /data
USER graphstore
WORKDIR /home/graphstore
VOLUME ["/data"]

EXPOSE 7200

# Healthcheck verifies the HTTP server is accepting connections - any
# response (even 401 from /api/* without auth) means the app is alive.
# We do not require 200 because the playground SPA is opt-in and the
# API enforces auth on all /api/* routes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(3); \
        s.connect(('127.0.0.1', 7200)); s.close()" || exit 1

# Entrypoint auto-generates an auth token on first boot (or honours
# GRAPHSTORE_AUTH_TOKEN / GRAPHSTORE_ALLOW_UNAUTH_BIND). Override the
# trailing CMD to run a different graphstore subcommand.
ENTRYPOINT ["/usr/local/bin/graphstore-entrypoint"]
