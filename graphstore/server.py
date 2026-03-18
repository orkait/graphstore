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
from graphstore.errors import GraphStoreError

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


def _get_store() -> GraphStore:
    global _store
    if _store is None:
        _store = GraphStore()
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
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=400,
            content={"kind": "error", "data": str(exc), "count": 0, "elapsed_us": 0},
        )
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
    nodes = store._store.get_all_nodes()
    edges = store._store.get_all_edges()
    return {"nodes": nodes, "edges": edges}


@app.post("/api/reset")
def reset():
    global _store
    _store = GraphStore()
    return {"ok": True}


@app.post("/api/config")
def config(req: ConfigRequest):
    store = _get_store()
    if req.ceiling_mb is not None:
        store._store._ceiling_bytes = req.ceiling_mb * 1_000_000
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
