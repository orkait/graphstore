# System Architecture

The overarching architecture of `graphstore` is composed of four main pillars:

## 1. Node Storage (`store.py:CoreStore`)
Nodes are stored in pre-allocated arrays and lists to prevent frequent memory allocations and to make data highly contiguous where possible.

- `node_ids`: A NumPy array (`int32`) storing interned string IDs.
- `node_kinds`: A NumPy array (`uint8`) storing interned kind IDs.
- `node_data`: A standard Python list storing the properties (`dict`) of every node.
- **Tombstones**: When nodes are deleted, their active slot is marked in a `tombstones` set. A new node insertion (`put_node`) will recycle a tombstoned slot before expanding the internal capacity.
- **Secondary Indices**: Optional quick-lookups on fields are stored in standard dictionaries (`secondary_indices[field][val] = [slots...]`).

## 2. Edge Storage (`edges.py:EdgeMatrices`)
The defining performance characteristic of `graphstore` is its usage of SciPy Compressed Sparse Row (CSR) matrices.

- **Matrix Types**: For every distinct edge `kind`, a dedicated $N \times N$ SciPy `csr_matrix` is instantiated.
- **Why CSR?**: CSR matrices allow for extremely fast queries on out-degrees and neighbor traversal without heavy object overhead.
- **Combined & Transposed**: `edges.py` also caches a combination of all edge matrices (the *union*) and caches transposed matrices (CSC) to rapidly compute the in-degrees or follow reverse edges (`ANCESTORS OF`).
- **Rebuilding**: Modifications to the edges are queued in an intermediate Python list. The actual matrix is fully rebuilt (from `np.ones` and source/target arrays) either automatically on demand or explicitly after a `COMMIT`.

## 3. String Interning (`strings.py:StringTable`)
Since text operations are slow and wasteful, node IDs and types are "interned"—mapped to unique integer IDs. This shrinks memory usage and accelerates comparative logic (e.g. `node_kinds[slot] == kind_id`).

## 4. Persistence (`persistence/database.py`)
To ensure recoverability across restarts, `graphstore` utilizes an SQLite backend configuered with `PRAGMA journal_mode=WAL`.

- **Write-Ahead Log**: Rather than immediately flushing arrays to disk upon every command, statements are written to the `wal` table sequentially. 
- **Checkpoints**: When a manual checkpoint is triggered (or upon database close), the in-memory graph is serialized into compressed SQLite blobs (storing chunks of bytes using `pickle` / `numpy`), truncating the WAL.

## 5. Memory Management (`memory.py`)
Because the database runs entirely in-memory, a strict size limit (`DEFAULT_CEILING_BYTES` = 256MB) is monitored prior to node bounds expansion, utilizing `check_ceiling` to reject queries that exceed RAM limits with a `CeilingExceeded` exception.
