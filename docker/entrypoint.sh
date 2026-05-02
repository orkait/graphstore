#!/usr/bin/env sh
# graphstore docker entrypoint.
#
# The playground refuses to bind 0.0.0.0 without auth - good. To make
# the container "just start", we auto-generate a random token on first
# boot and write it to /data/.auth_token so it persists across
# restarts (volume-mounted /data is the default). Operators who want a
# fixed token set GRAPHSTORE_AUTH_TOKEN themselves; operators who
# accept a private-network deployment without auth set
# GRAPHSTORE_ALLOW_UNAUTH_BIND=1.
set -eu

DB="${GRAPHSTORE_DB_PATH:-/data}"
HOST="${GRAPHSTORE_HOST:-0.0.0.0}"
PORT="${GRAPHSTORE_PORT:-7200}"
TOKEN_FILE="${DB}/.auth_token"

if [ -z "${GRAPHSTORE_AUTH_TOKEN:-}" ] && [ "${GRAPHSTORE_ALLOW_UNAUTH_BIND:-0}" != "1" ]; then
    mkdir -p "${DB}"
    if [ ! -s "${TOKEN_FILE}" ]; then
        python -c "import secrets; print(secrets.token_urlsafe(32))" > "${TOKEN_FILE}"
        chmod 600 "${TOKEN_FILE}" || true
        echo "[entrypoint] generated new auth token at ${TOKEN_FILE}"
    fi
    GRAPHSTORE_AUTH_TOKEN="$(cat "${TOKEN_FILE}")"
    export GRAPHSTORE_AUTH_TOKEN
    echo "[entrypoint] auth token (use as 'Authorization: Bearer <token>'): ${GRAPHSTORE_AUTH_TOKEN}"
fi

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec graphstore playground \
    --host "${HOST}" \
    --port "${PORT}" \
    --no-browser \
    --db-path "${DB}"
