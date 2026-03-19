"""Deserialize a CoreStore + SchemaRegistry from sqlite.

load() reads the full graph state from the blobs and metadata tables,
reconstructing the in-memory CoreStore with arrays, edge matrices,
secondary indices, and tombstones.
"""

import json

import numpy as np

from graphstore.store import CoreStore
from graphstore.strings import StringTable
from graphstore.schema import SchemaRegistry
from graphstore.persistence.database import SCHEMA_VERSION
from graphstore.errors import VersionMismatch


def load(conn) -> tuple[CoreStore, SchemaRegistry]:
    """Load graph from sqlite. Returns (store, schema).

    Raises VersionMismatch if schema_version doesn't match.
    Returns empty store if no data found.
    """
    # Check version
    row = conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    if row is None:
        # Fresh database, return empty store
        return CoreStore(), SchemaRegistry()

    if int(row[0]) != SCHEMA_VERSION:
        raise VersionMismatch(found=row[0], expected=SCHEMA_VERSION)

    # Load string table
    strings_row = conn.execute("SELECT data FROM blobs WHERE key='strings'").fetchone()
    if strings_row is None:
        return CoreStore(), SchemaRegistry()

    string_table = StringTable.from_list(json.loads(strings_row[0]))

    # Load store metadata
    meta_row = conn.execute("SELECT data FROM blobs WHERE key='store_meta'").fetchone()
    meta = json.loads(meta_row[0])

    # Create store and set its internals
    store = CoreStore()
    store.string_table = string_table
    store._next_slot = meta["next_slot"]
    store._count = meta["count"]

    # Ensure capacity
    capacity = meta.get("capacity", max(meta["next_slot"] * 2, 1024))
    store._capacity = capacity

    # Load node arrays
    ids_row = conn.execute("SELECT data, dtype FROM blobs WHERE key='node_ids'").fetchone()
    loaded_ids = np.frombuffer(ids_row[0], dtype=np.dtype(ids_row[1])).copy()
    store.node_ids = np.full(capacity, -1, dtype=np.int32)
    store.node_ids[:len(loaded_ids)] = loaded_ids

    kinds_row = conn.execute("SELECT data, dtype FROM blobs WHERE key='node_kinds'").fetchone()
    loaded_kinds = np.frombuffer(kinds_row[0], dtype=np.dtype(kinds_row[1])).copy()
    store.node_kinds = np.zeros(capacity, dtype=np.uint8)
    store.node_kinds[:len(loaded_kinds)] = loaded_kinds

    # Load node data
    data_row = conn.execute("SELECT data FROM blobs WHERE key='node_data'").fetchone()
    loaded_data = json.loads(data_row[0])
    store.node_data = loaded_data + [None] * (capacity - len(loaded_data))

    # Load tombstones
    tomb_row = conn.execute("SELECT data FROM blobs WHERE key='tombstones'").fetchone()
    store.node_tombstones = set(json.loads(tomb_row[0]))

    # Rebuild id_to_slot
    store.id_to_slot = {}
    for slot in range(store._next_slot):
        if slot not in store.node_tombstones and store.node_data[slot] is not None:
            str_id = int(store.node_ids[slot])
            if str_id >= 0:
                store.id_to_slot[str_id] = slot

    # Load raw edge lists
    store._edges_by_type = {}
    edge_rows = conn.execute(
        "SELECT key, data FROM blobs WHERE key LIKE 'raw_edges:%'"
    ).fetchall()
    for key, data in edge_rows:
        etype = key[len("raw_edges:"):]
        edge_list = json.loads(data)
        store._edges_by_type[etype] = [(s, t, d) for s, t, d in edge_list]

    # Rebuild edge keys set and edge matrices
    store._edge_keys = {
        (s, t, k)
        for k, edges in store._edges_by_type.items()
        for s, t, _d in edges
    }
    store._rebuild_edges()

    # Load indexed fields and rebuild indices
    idx_row = conn.execute("SELECT data FROM blobs WHERE key='indexed_fields'").fetchone()
    if idx_row:
        indexed_fields = json.loads(idx_row[0])
        for field in indexed_fields:
            store.add_index(field)

    # Load schema
    schema = SchemaRegistry()
    schema_row = conn.execute("SELECT data FROM blobs WHERE key='schema'").fetchone()
    if schema_row:
        schema = SchemaRegistry.from_dict(json.loads(schema_row[0]))

    return store, schema
