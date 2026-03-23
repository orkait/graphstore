"""FastAPI server for the graphstore playground."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graphstore import GraphStore
from graphstore.core.errors import GraphStoreError

app = FastAPI(title="graphstore playground")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Module-level store (created lazily)
# ---------------------------------------------------------------------------

_store: GraphStore | None = None


import os

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
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/execute")
def execute(req: ExecuteRequest):
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
    store = _get_store()
    results = []
    for q in req.queries:
        try:
            r = store.execute(q)
            results.append(_result_to_dict(r))
        except GraphStoreError as exc:
            results.append(
                {"kind": "error", "data": str(exc), "count": 0, "elapsed_us": 0}
            )
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
