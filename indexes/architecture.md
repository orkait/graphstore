# System Architecture

The overarching architecture of `graphstore` is composed of five main pillars:

## 1. Columnar Node Storage (`store.py:CoreStore`, `columns.py:ColumnStore`)
All node field data is stored in typed numpy arrays managed by `ColumnStore` - the sole source of truth.

- `node_ids`: NumPy array (`int32`) storing interned string IDs.
- `node_kinds`: NumPy array (`int32`) storing interned kind IDs.
- `ColumnStore._columns`: Dict of field name to typed numpy array (`int64`, `float64`, or `int32` interned strings).
- `ColumnStore._presence`: Dict of field name to boolean numpy mask (tracks which slots have a value set).
- **Reserved columns**: System-managed fields prefixed with `__` (e.g. `__created_at__`, `__updated_at__`, `__expires_at__`, `__retracted__`, `__confidence__`, `__source__`, `__context__`). Invisible in user-facing query results.
- **Tombstones**: Deleted slots marked in a `tombstones` set. Recycled on next insertion before expanding capacity.
- **Secondary Indices**: Optional quick-lookups on fields stored in standard dictionaries (`secondary_indices[field][val] = [slots...]`).
- **Materialization**: User-facing dicts are built on demand from column arrays only when needed for query results (`_materialize_slot`). At LIMIT 10, this costs ~5us.

## 2. Edge Storage (`edges.py:EdgeMatrices`)
The defining performance characteristic of `graphstore` is its usage of SciPy Compressed Sparse Row (CSR) matrices.

- **Matrix Types**: For every distinct edge `kind`, a dedicated N x N SciPy `csr_matrix` is instantiated.
- **Why CSR?**: CSR matrices allow for extremely fast queries on out-degrees and neighbor traversal without heavy object overhead. Also enables spreading activation via sparse matrix-vector multiply (`csr.dot(activation)`).
- **Combined & Transposed**: `edges.py` caches a combination of all edge matrices (the union) and transposed matrices (CSC) for rapid in-degree computation and reverse traversal (`ANCESTORS OF`).
- **Rebuilding**: Modifications are queued in an intermediate Python list. The actual matrix is rebuilt on demand or explicitly after a `COMMIT`.

## 3. String Interning (`strings.py:StringTable`)
Node IDs, kinds, and string field values are "interned" - mapped to unique integer IDs. This shrinks memory usage and accelerates comparisons (e.g. `node_kinds[slot] == kind_id`). String columns store int32 interned IDs, not raw strings.

## 4. Persistence (`persistence/database.py`)
`graphstore` uses an SQLite backend configured with `PRAGMA journal_mode=WAL`.

- **Write-Ahead Log**: Statements are written to the `wal` table sequentially rather than immediately flushing arrays.
- **Checkpoints**: On manual checkpoint or database close, column arrays are serialized as raw numpy byte blobs. Field names are URL-encoded to handle special characters (e.g. reserved `__` columns with colons).
- **Migration**: The deserializer handles legacy `node_data` JSON blobs from older versions and auto-migrates to columnar-only storage.

## 5. Memory Management (`memory.py`)
A strict size limit (`DEFAULT_CEILING_BYTES` = 256MB) is enforced prior to node/edge insertion. `check_ceiling` rejects operations that would exceed the limit with `CeilingExceeded`.

## 6. Visibility System (`store.py:compute_live_mask`, `executor.py:_compute_live_mask`)
A unified boolean mask computed once per query that determines which nodes are visible:

- **Tombstones**: deleted nodes excluded
- **TTL expiry**: nodes past `__expires_at__` excluded
- **Retracted beliefs**: nodes with `__retracted__ = 1` excluded
- **Context isolation**: when a context is bound, only nodes tagged with `__context__ = name` are visible

All query types (NODES, TRAVERSE, PATH, MATCH, RECALL, AGGREGATE, etc.) respect this mask.
