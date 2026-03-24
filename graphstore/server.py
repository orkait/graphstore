"""FastAPI server for the graphstore playground."""

from __future__ import annotations

import os
import time as _time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graphstore import GraphStore
from graphstore.core.errors import GraphStoreError

app = FastAPI(title="graphstore playground")

_CORS_ORIGINS = os.environ.get("GRAPHSTORE_CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth + Rate Limiting Middleware ---

_AUTH_TOKEN = os.environ.get("GRAPHSTORE_AUTH_TOKEN")
_RATE_LIMIT_RPM = int(os.environ.get("GRAPHSTORE_RATE_LIMIT_RPM", "120"))

_rate_buckets: dict[str, list[float]] = defaultdict(list)
_rate_cleanup_counter = 0


def _check_rate_limit(client_ip: str) -> bool:
    """Returns True if request is allowed."""
    global _rate_cleanup_counter
    now = _time.time()
    window = float(_RATE_LIMIT_WINDOW)
    bucket = _rate_buckets[client_ip]
    _rate_buckets[client_ip] = [t for t in bucket if now - t < window]
    if len(_rate_buckets[client_ip]) >= _RATE_LIMIT_RPM:
        return False
    _rate_buckets[client_ip].append(now)

    _rate_cleanup_counter += 1
    if _rate_cleanup_counter % 100 == 0:
        stale = [ip for ip, ts in _rate_buckets.items() if not ts or now - max(ts) > float(_RATE_LIMIT_WINDOW)]
        for ip in stale:
            del _rate_buckets[ip]

    return True


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    if _AUTH_TOKEN is not None:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header[7:] != _AUTH_TOKEN:
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"},
            headers={"Retry-After": str(_RATE_LIMIT_WINDOW)},
        )

    return await call_next(request)


# ---------------------------------------------------------------------------
# Module-level store (created lazily)
# ---------------------------------------------------------------------------

_store: GraphStore | None = None


def _get_store() -> GraphStore:
    global _store
    if _store is None:
        db_path = os.environ.get("GRAPHSTORE_DB_PATH")
        config_path = os.environ.get("GRAPHSTORE_CONFIG")
        ingest_root = os.environ.get("GRAPHSTORE_INGEST_ROOT", os.getcwd())
        kwargs: dict = {}
        if db_path:
            kwargs["path"] = db_path
        if config_path:
            kwargs["config_path"] = config_path
        if ingest_root:
            kwargs["ingest_root"] = ingest_root
        _store = GraphStore(**kwargs)
    return _store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_dict(result) -> dict[str, Any]:
    """Convert a Result to a JSON-serialisable dict, handling numpy arrays."""
    data = result.data
    if isinstance(data, np.ndarray):
        data = data.tolist()
    elif isinstance(data, list):
        data = [
            item.tolist() if isinstance(item, np.ndarray) else item
            for item in data
        ]
    elif isinstance(data, dict):
        data = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in data.items()
        }
    return {
        "kind": result.kind,
        "data": data,
        "count": result.count,
        "elapsed_us": result.elapsed_us,
    }


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ExecuteRequest(BaseModel):
    query: str


class BatchRequest(BaseModel):
    queries: list[str]


class ConfigRequest(BaseModel):
    ceiling_mb: int | None = None
    cost_threshold: int | None = None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

_MAX_QUERY_LENGTH = int(os.environ.get("GRAPHSTORE_MAX_QUERY_LENGTH", "10000"))
_MAX_BATCH_SIZE = int(os.environ.get("GRAPHSTORE_MAX_BATCH_SIZE", "1000"))
_RATE_LIMIT_WINDOW = int(os.environ.get("GRAPHSTORE_RATE_LIMIT_WINDOW", "60"))


def _validate_query(query: str) -> str | None:
    """Validate query input. Returns error message or None if valid."""
    if not query or not query.strip():
        return "Empty query"
    if len(query) > _MAX_QUERY_LENGTH:
        return f"Query exceeds maximum length ({_MAX_QUERY_LENGTH} chars)"
    if "\x00" in query:
        return "Query contains null bytes"
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/execute")
def execute(req: ExecuteRequest):
    err = _validate_query(req.query)
    if err:
        return {"kind": "error", "data": err, "count": 0, "elapsed_us": 0}
    store = _get_store()
    try:
        result = store.execute(req.query)
    except GraphStoreError as exc:
        # Return soft errors (NodeExists, duplicate edge) as 200 so the
        # frontend can continue executing subsequent queries in a batch.
        return {"kind": "error", "data": str(exc), "count": 0, "elapsed_us": 0}
    return _result_to_dict(result)


@app.post("/api/execute-batch")
def execute_batch(req: BatchRequest):
    if len(req.queries) > _MAX_BATCH_SIZE:
        return [{"kind": "error", "data": f"Batch exceeds {_MAX_BATCH_SIZE} queries", "count": 0, "elapsed_us": 0}]
    store = _get_store()
    results = []
    for q in req.queries:
        err = _validate_query(q)
        if err:
            results.append({"kind": "error", "data": err, "count": 0, "elapsed_us": 0})
            continue
        try:
            r = store.execute(q)
            results.append(_result_to_dict(r))
        except GraphStoreError as exc:
            results.append({"kind": "error", "data": str(exc), "count": 0, "elapsed_us": 0})
    return results


@app.get("/api/graph")
def get_graph():
    store = _get_store()
    nodes = store.get_all_nodes()
    edges = store.get_all_edges()
    return {"nodes": nodes, "edges": edges}


@app.post("/api/reset")
def reset():
    """Reset the in-memory graph. For persistent DBs, wipes memory and WAL
    but preserves the script metadata so Run All can repopulate cleanly."""
    global _store
    db_path = os.environ.get("GRAPHSTORE_DB_PATH")
    if db_path:
        # Preserve the script before wiping
        old_script = _store.get_script() if _store else None
        # Close old store
        if _store and _store._conn:
            _store._conn.close()
            _store._conn = None
        # Wipe the DB file contents but keep the file
        from graphstore.persistence.database import open_database, set_metadata
        db_file = Path(db_path) / "graphstore.db"
        conn = open_database(db_file)
        conn.execute("DELETE FROM blobs")
        conn.execute("DELETE FROM wal")
        conn.execute("DELETE FROM query_log")
        conn.execute("DELETE FROM metadata")
        conn.commit()
        # Restore script
        if old_script:
            set_metadata(conn, "playground_script", old_script)
        conn.close()
        # Create a fresh store from the now-empty DB
        _store = GraphStore(path=db_path)
    else:
        _store = GraphStore()
    return {"ok": True}


@app.get("/api/script")
def get_script():
    store = _get_store()
    script = store.get_script()
    return {"script": script}


@app.put("/api/script")
def put_script(req: ExecuteRequest):
    store = _get_store()
    store.set_script(req.query)
    return {"ok": True}


@app.post("/api/config")
def config(req: ConfigRequest):
    store = _get_store()
    if req.ceiling_mb is not None:
        store.ceiling_mb = req.ceiling_mb
    if req.cost_threshold is not None:
        store.cost_threshold = req.cost_threshold
    return {"ok": True}


@app.get("/api/logs")
def get_logs(
    limit: int = 50,
    tag: str | None = None,
    source: str | None = None,
    trace_id: str | None = None,
    since: str | None = None,
):
    """Query the enriched query log."""
    store = _get_store()
    conn = store._conn
    if conn is None:
        return []

    sql = "SELECT id, timestamp, query, elapsed_us, result_count, error, tag, trace_id, source, phase FROM query_log"
    params: list = []
    conditions = []

    if tag:
        conditions.append("tag = ?")
        params.append(tag)
    if source:
        conditions.append("source LIKE ?")
        params.append(f"%{source}%")
    if trace_id:
        conditions.append("trace_id = ?")
        params.append(trace_id)
    if since:
        import datetime
        dt = datetime.datetime.fromisoformat(since)
        conditions.append("timestamp >= ?")
        params.append(dt.timestamp())

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "id": r[0], "timestamp": r[1], "query": r[2],
            "elapsed_us": r[3], "result_count": r[4], "error": r[5],
            "tag": r[6], "trace_id": r[7], "source": r[8], "phase": r[9],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Static files helper
# ---------------------------------------------------------------------------


def mount_static(application: FastAPI, path: str | Path) -> None:
    """Mount a StaticFiles directory on / if it exists."""
    p = Path(path)
    if p.is_dir():
        application.mount(
            "/",
            StaticFiles(directory=str(p), html=True),
            name="playground",
        )
