"""Serialize a CoreStore + SchemaRegistry to sqlite.

checkpoint() writes the full graph state into the blobs and metadata
tables within a single transaction.
"""

import time
from urllib.parse import quote

import msgspec.json as mjson

from graphstore.persistence.database import SCHEMA_VERSION
from graphstore.core.store import CoreStore
from graphstore.core.schema import SchemaRegistry


def checkpoint(store: CoreStore, schema: SchemaRegistry, conn):
    """Write full graph state to sqlite. All within a single transaction."""
    with conn:
        # String table
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("strings", mjson.encode(store.string_table.to_list()), "json"))

        # Node kinds array
        kinds_data = store.node_kinds[:store._next_slot]
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("node_kinds", kinds_data.tobytes(), str(kinds_data.dtype)))

        # Node IDs array
        ids_data = store.node_ids[:store._next_slot]
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("node_ids", ids_data.tobytes(), str(ids_data.dtype)))

        # Column store data (sole source of truth for node fields)
        conn.execute("DELETE FROM blobs WHERE key LIKE 'columns:%'")
        for field in store.columns._columns:
            col_data = store.columns._columns[field][:store._next_slot]
            pres_data = store.columns._presence[field][:store._next_slot]
            dtype_str = store.columns._dtypes[field]
            safe_field = quote(field, safe="")
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         (f"columns:{safe_field}:data", col_data.tobytes(), str(col_data.dtype)))
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         (f"columns:{safe_field}:presence", pres_data.tobytes(), str(pres_data.dtype)))
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         (f"columns:{safe_field}:dtype", dtype_str.encode(), "text"))

        # Edge matrices: one set of blobs per type
        # First, clear old edge blobs
        conn.execute("DELETE FROM blobs WHERE key LIKE 'edges:%'")
        conn.execute("DELETE FROM blobs WHERE key LIKE 'edge_data:%'")

        for etype, matrix in store.edge_matrices._typed.items():
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                        (f"edges:{etype}:indptr", matrix.indptr.tobytes(), str(matrix.indptr.dtype)))
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                        (f"edges:{etype}:indices", matrix.indices.tobytes(), str(matrix.indices.dtype)))
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                        (f"edges:{etype}:data", matrix.data.tobytes(), str(matrix.data.dtype)))
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                        (f"edges:{etype}:shape", mjson.encode(list(matrix.shape)), "json"))

        # Raw edge lists for store reconstruction
        conn.execute("DELETE FROM blobs WHERE key LIKE 'raw_edges:%'")
        for etype, edge_list in store._edges_by_type.items():
            # edge_list is [(src_slot, tgt_slot, data_dict), ...]
            serializable = [(int(s), int(t), d) for s, t, d in edge_list]
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                        (f"raw_edges:{etype}", mjson.encode(serializable), "json"))

        # Tombstones
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("tombstones", mjson.encode(list(store.node_tombstones)), "json"))

        # Secondary index field names (not data)
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("indexed_fields", mjson.encode(list(store._indexed_fields)), "json"))

        # Store metadata
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("store_meta", mjson.encode({
                         "next_slot": store._next_slot,
                         "count": store._count,
                         "capacity": store._capacity,
                     }), "json"))

        # Schema
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("schema", mjson.encode(schema.to_dict()), "json"))

        # Vector index
        vectors = getattr(store, 'vectors', None)
        if vectors is not None and vectors.count() > 0:
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         ("vector_index", vectors.save(), "usearch"))
            pres = vectors._has_vector[:store._next_slot]
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         ("vector_presence", pres.tobytes(), str(pres.dtype)))
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         ("vector_dims", str(vectors.dims).encode(), "text"))
        else:
            # Clear any stale vector blobs
            conn.execute("DELETE FROM blobs WHERE key IN ('vector_index', 'vector_presence', 'vector_dims')")

        # Version + timestamp
        conn.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                     ("schema_version", str(SCHEMA_VERSION)))
        conn.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                     ("last_checkpoint_time", str(time.time())))

        # Clear WAL after checkpoint
        conn.execute("DELETE FROM wal")
